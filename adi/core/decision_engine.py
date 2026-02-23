"""ADI Decision Engine — the core orchestrator.

Pipeline: validate → load policy → normalize → score → uncertainty →
          apply constraints → explain → build output

This is the single entry point: decide(request) -> DecisionOutput.
It is pure Python — no LLM, no I/O side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from adi.schemas.decision_output import FeedbackInput, ProfileUpdate, ScenarioComparison

import numpy as np

from adi.core.explain import (
    compute_counterfactuals,
    compute_criterion_contributions,
    evaluate_constraints,
)
from adi.core.fuzzy import build_normalized_matrix
from adi.core.mcda import build_confidence_matrix, score_options
from adi.core.policy import PolicyRegistry, resolve_policy
from adi.core.uncertainty import (
    apply_confidence_adjustment,
    compute_all_confidences,
    robustness_check,
)
from adi.schemas.decision_output import (
    DataQualityReport,
    DecisionOutput,
    OptionResult,
    SensitivityReport,
)
from adi.schemas.decision_request import DecisionRequest, WeightProfile


def _effective_weights(request: DecisionRequest, criteria: list) -> np.ndarray:
    """Weights from criteria, optionally overridden by request.preferences or profile."""
    base = np.array([c.weight for c in criteria], dtype=float)
    prefs = getattr(request, "preferences", None)
    if prefs and getattr(prefs, "weights", None) and prefs.weights:
        for i, c in enumerate(criteria):
            if c.name in prefs.weights:
                base[i] = prefs.weights[c.name]
        total = base.sum()
        if total > 0:
            base = base / total
        else:
            base = np.ones(len(criteria)) / len(criteria)
    else:
        total = base.sum()
        if total == 0:
            base = np.ones(len(criteria)) / len(criteria)
        else:
            base = base / total
    return base


def decide(
    request: DecisionRequest,
    registry: PolicyRegistry | None = None,
) -> DecisionOutput:
    """
    Main decision pipeline.

    Args:
        request: fully validated DecisionRequest
        registry: optional custom PolicyRegistry; defaults to global registry

    Returns:
        DecisionOutput with full ranking, explanations, and audit trail
    """
    warnings: list[str] = []

    # 0. Load learned profile if profile_id set
    if getattr(request, "profile_id", None):
        from adi.core.learning import LearningEngine
        _engine = LearningEngine()
        loaded = _engine.load_profile(request.profile_id)
        if loaded and getattr(loaded, "weights", None):
            prefs = getattr(request, "preferences", None)
            if prefs and getattr(prefs, "weights", None):
                merged = dict(loaded.weights)
                merged.update(prefs.weights)
                request = request.model_copy(update={"preferences": WeightProfile(weights=merged)})
            else:
                request = request.model_copy(update={"preferences": loaded})

    # 1. Resolve policy
    policy = resolve_policy(
        name=request.policy_name,
        overrides=request.policy_overrides or None,
        registry=registry,
    )

    criteria = request.criteria
    options = request.options

    # 2. Build normalized decision matrix
    matrix, option_names, criterion_names, normalized_map = build_normalized_matrix(
        criteria, options
    )

    # 3. Compute weights (optionally merge profile preferences)
    weights = _effective_weights(request, criteria)

    # 4. Build confidence matrix for optional per-cell scoring
    confidence_matrix = build_confidence_matrix(options, criteria)

    # 5. Score options (raw + policy-adjusted; may use confidence in formula)
    raw_scores, adjusted_scores = score_options(
        matrix, weights, policy, confidence_matrix=confidence_matrix
    )

    # 6. Compute confidence per option
    confidences = compute_all_confidences(options, criteria, weights, policy)

    # 7. Apply confidence adjustment (skip when already in formula)
    uncertainty_factor = (
        0.0 if getattr(policy, "use_cell_confidence", False) else policy.uncertainty_penalty_factor
    )
    final_scores = apply_confidence_adjustment(
        adjusted_scores, confidences, uncertainty_factor
    )

    # 8. Evaluate constraints → eliminate violators, apply soft penalties
    elimination_map: dict[str, tuple[bool, str]] = {}
    constraint_reports_map: dict[str, list] = {}
    for option in options:
        eliminated, c_reports = evaluate_constraints(
            option, request.constraints, policy.constraint_priority_mode
        )
        elimination_map[option.name] = (eliminated, _first_elimination_reason(c_reports))
        constraint_reports_map[option.name] = c_reports

    # Apply soft constraint penalties to non-eliminated options
    for i, option in enumerate(options):
        eliminated, _ = elimination_map[option.name]
        if not eliminated:
            violated_soft = sum(
                1 for r in constraint_reports_map[option.name]
                if not r.satisfied and not r.hard
            )
            if violated_soft > 0:
                penalty = (1.0 - policy.soft_constraint_penalty) ** violated_soft
                final_scores[i] *= penalty
                warnings.append(
                    f"Option '{option.name}' has {violated_soft} soft constraint violation(s); "
                    f"score penalized by factor {penalty:.3f}"
                )

    # 9. Build per-option results
    active_scores = final_scores.copy()
    for i, option in enumerate(options):
        if elimination_map[option.name][0]:
            active_scores[i] = -1.0  # push to bottom

    sort_order = np.argsort(-active_scores)  # descending

    option_results: list[OptionResult] = []
    for rank_idx, opt_idx in enumerate(sort_order):
        option = options[opt_idx]
        opt_name = option.name
        eliminated, elim_reason = elimination_map[opt_name]

        contributions = compute_criterion_contributions(
            option=option,
            criteria=criteria,
            normalized_weights=weights,
            normalized_values=normalized_map[opt_name],
            final_score=float(final_scores[opt_idx]),
        )

        rank = rank_idx + 1
        counterfactuals = []
        if not eliminated:
            counterfactuals = compute_counterfactuals(
                option=option,
                criteria=criteria,
                normalized_weights=weights,
                all_option_names=option_names,
                all_scores=final_scores,
                current_rank=rank,
                normalized_values=normalized_map[opt_name],
                all_options=options,
            )

        option_results.append(
            OptionResult(
                option_name=opt_name,
                rank=rank,
                score=float(final_scores[opt_idx]),
                raw_score=float(raw_scores[opt_idx]),
                confidence=float(confidences[opt_idx]),
                eliminated=eliminated,
                elimination_reason=elim_reason,
                criterion_contributions=contributions,
                constraint_reports=constraint_reports_map[opt_name],
                counterfactuals=counterfactuals,
            )
        )

    # 10. Sensitivity analysis (on non-eliminated options)
    active_matrix = np.array([
        matrix[i] for i, opt in enumerate(options)
        if not elimination_map[opt.name][0]
    ])
    active_names = [
        opt.name for opt in options if not elimination_map[opt.name][0]
    ]

    if len(active_names) >= 2:
        stability, unstable_pairs, weight_sensitivity = robustness_check(
            active_matrix, weights, active_names, policy.sensitivity_perturbation_pct
        )
        # Map criterion indices back to names
        named_sensitivity = {
            criterion_names[int(k)]: v
            for k, v in weight_sensitivity.items()
            if int(k) < len(criterion_names)
        }
    else:
        stability = 1.0
        unstable_pairs = []
        named_sensitivity = {}
        if len(active_names) < 2:
            warnings.append("Fewer than 2 non-eliminated options; sensitivity analysis skipped.")

    sensitivity = SensitivityReport(
        overall_stability=stability,
        unstable_pairs=unstable_pairs,
        weight_sensitivity=named_sensitivity,
    )

    # 11. Determine best option
    non_eliminated = [r for r in option_results if not r.eliminated]
    if not non_eliminated:
        warnings.append("All options were eliminated by constraints.")
        best_option = option_results[0].option_name if option_results else "NONE"
    else:
        best_option = non_eliminated[0].option_name

    overall_confidence = float(np.mean([
        r.confidence for r in non_eliminated
    ])) if non_eliminated else 0.0

    # Data quality report from confidence matrix
    n_crit = len(criteria)
    n_opt = len(options)
    total_cells = n_opt * n_crit
    mean_conf = float(np.mean(confidence_matrix)) if total_cells > 0 else 0.0
    low_conf_list: list[dict[str, str | float]] = []
    missing_list: list[dict[str, str]] = []
    for i, opt in enumerate(options):
        for j, c in enumerate(criteria):
            conf = float(confidence_matrix[i, j])
            if opt.get_value(c.name) is None:
                missing_list.append({"option": opt.name, "criterion": c.name})
            elif conf < 0.5:
                low_conf_list.append({"option": opt.name, "criterion": c.name, "confidence": conf})
    deficit = sum(1.0 - confidence_matrix[i, j] for i in range(n_opt) for j in range(n_crit))
    confidence_impact = min(deficit / total_cells, 1.0) if total_cells > 0 else 0.0
    data_quality = DataQualityReport(
        mean_confidence=round(mean_conf, 4),
        low_confidence_features=low_conf_list,
        missing_features=missing_list,
        confidence_impact=round(confidence_impact, 4),
    )

    return DecisionOutput(
        best_option=best_option,
        ranking=option_results,
        overall_confidence=overall_confidence,
        policy_applied=policy.name,
        policy_parameters=policy.to_audit_dict(),
        sensitivity=sensitivity,
        warnings=warnings,
        context=request.context,
        data_quality=data_quality,
    )


def _first_elimination_reason(constraint_reports: list) -> str:
    for r in constraint_reports:
        if not r.satisfied and r.hard:
            return r.detail
    return ""


def what_if(
    request: DecisionRequest,
    changes: dict[str, Any],
    registry: PolicyRegistry | None = None,
) -> DecisionOutput:
    """Run a what-if analysis with modified inputs (dot-notation paths)."""
    from adi.core.scenarios import what_if as _what_if
    return _what_if(request, changes, lambda r: decide(r, registry))


def compare(
    request: DecisionRequest,
    scenarios: list[dict[str, Any]],
    registry: PolicyRegistry | None = None,
) -> "ScenarioComparison":
    """Compare multiple what-if scenarios; returns ScenarioComparison."""
    from adi.core.scenarios import compare_scenarios
    return compare_scenarios(request, scenarios, lambda r: decide(r, registry))


def feedback(
    feedback_input: "FeedbackInput",
    decision: DecisionOutput,
    request: DecisionRequest,
    registry: PolicyRegistry | None = None,
) -> "ProfileUpdate":
    """Process feedback and update the learning profile. Returns ProfileUpdate."""
    from adi.core.learning import LearningEngine
    from adi.schemas.decision_output import ProfileUpdate  # needed at runtime for construction
    engine = LearningEngine()
    update = engine.process_feedback(feedback_input, decision, request)
    profile_id = request.profile_id or "default"
    engine.update_profile(profile_id, update)
    raw = engine._load_raw_profile(profile_id)
    return ProfileUpdate(
        profile_id=profile_id,
        weight_adjustments=update.weight_adjustments,
        fuzzy_adjustments=update.fuzzy_adjustments,
        feedback_count=raw.feedback_count if raw else 0,
    )

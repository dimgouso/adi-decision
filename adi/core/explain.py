"""ADI Explain module — contributions, constraint reports, and counterfactuals.

Every decision output must be fully explainable. This module generates:
1. Per-criterion contribution breakdown for each option
2. Constraint satisfaction reports
3. Counterfactual analysis: "what would change if criterion X differed?"
"""

from __future__ import annotations

import numpy as np

from adi.schemas.decision_output import (
    ConstraintReport,
    CounterfactualItem,
    CriterionContribution,
)
from adi.schemas.decision_request import (
    Constraint,
    ConstraintType,
    Criterion,
    Option,
)
from adi.core.policy import ConstraintPriorityMode


def compute_criterion_contributions(
    option: Option,
    criteria: list[Criterion],
    normalized_weights: np.ndarray,
    normalized_values: dict[str, float | None],
    final_score: float,
) -> list[CriterionContribution]:
    """
    Compute how much each criterion contributed to the option's final score.
    """
    contributions = []
    total_weighted = sum(
        (normalized_values.get(c.name) or 0.0) * w
        for c, w in zip(criteria, normalized_weights)
    )

    for i, criterion in enumerate(criteria):
        weight = float(normalized_weights[i])
        raw_val = option.get_value(criterion.name)
        norm_val = normalized_values.get(criterion.name)
        missing = raw_val is None

        weighted_score = (norm_val or 0.0) * weight
        contribution_pct = (
            (weighted_score / total_weighted * 100.0) if total_weighted > 0 else 0.0
        )

        from adi.core.uncertainty import compute_evidence_quality

        eq = compute_evidence_quality(option, criterion)

        contributions.append(
            CriterionContribution(
                criterion_name=criterion.name,
                raw_value=raw_val,
                normalized_value=norm_val,
                weighted_score=weighted_score,
                contribution_pct=contribution_pct,
                evidence_quality=eq,
                missing=missing,
            )
        )

    return contributions


def evaluate_constraints(
    option: Option,
    constraints: list[Constraint],
    priority_mode: ConstraintPriorityMode = ConstraintPriorityMode.ELIMINATE,
) -> tuple[bool, list[ConstraintReport]]:
    """
    Evaluate all constraints for one option.

    priority_mode controls how hard constraint violations are handled at the policy level:
    - ELIMINATE: violators are removed from ranking entirely (default)
    - PENALIZE: violations are recorded but option is not eliminated (score penalized by caller)
    - WARN: violations are noted only; no score effect

    Returns:
        is_eliminated: True only when mode=ELIMINATE and a hard constraint is violated
        reports: list of ConstraintReport (one per applicable constraint)
    """
    reports = []
    is_eliminated = False

    for constraint in constraints:
        applicable, satisfied, detail = _check_single_constraint(option, constraint)
        if not applicable:
            continue

        if not satisfied and constraint.hard and priority_mode == ConstraintPriorityMode.ELIMINATE:
            is_eliminated = True

        reports.append(
            ConstraintReport(
                constraint_type=constraint.constraint_type.value,
                satisfied=satisfied,
                hard=constraint.hard,
                detail=detail,
            )
        )

    return is_eliminated, reports


def _check_single_constraint(
    option: Option,
    constraint: Constraint,
) -> tuple[bool, bool, str]:
    """
    Returns (applicable, satisfied, detail).
    applicable=False means this constraint doesn't target this option.
    """
    ct = constraint.constraint_type

    if ct == ConstraintType.MUST_INCLUDE:
        if constraint.option_name != option.name:
            return False, True, ""
        return True, True, f"Option '{option.name}' explicitly included"

    if ct == ConstraintType.MUST_EXCLUDE:
        if constraint.option_name != option.name:
            return False, True, ""
        return True, False, f"Option '{option.name}' must be excluded"

    if ct in (ConstraintType.MIN_VALUE, ConstraintType.MAX_VALUE):
        if constraint.option_name and constraint.option_name != option.name:
            return False, True, ""
        if not constraint.criterion_name or constraint.threshold is None:
            return False, True, ""

        raw = option.get_value(constraint.criterion_name)
        if raw is None:
            return True, False, (
                f"Missing value for '{constraint.criterion_name}' — cannot satisfy constraint"
            )

        if ct == ConstraintType.MIN_VALUE:
            satisfied = raw >= constraint.threshold
            detail = (
                f"{constraint.criterion_name}={raw:.3f} "
                f">= {constraint.threshold:.3f}: {'OK' if satisfied else 'FAIL'}"
            )
        else:
            satisfied = raw <= constraint.threshold
            detail = (
                f"{constraint.criterion_name}={raw:.3f} "
                f"<= {constraint.threshold:.3f}: {'OK' if satisfied else 'FAIL'}"
            )
        return True, satisfied, detail

    return False, True, ""


def compute_counterfactuals(
    option: Option,
    criteria: list[Criterion],
    normalized_weights: np.ndarray,
    all_option_names: list[str],
    all_scores: np.ndarray,
    current_rank: int,
    perturbation_pct: float = 20.0,
    normalized_values: dict[str, float | None] | None = None,
    all_options: list[Option] | None = None,
) -> list[CounterfactualItem]:
    """
    'What if criterion X improved/worsened by perturbation_pct%?'

    For each criterion, simulate increasing/decreasing the option's normalized value
    by perturbation_pct and recalculate its score to determine new rank.

    normalized_values: pre-computed {criterion_name: normalized_value} for this option
    all_options: full option list for proper normalization context (min-max needs population)

    Returns list of meaningful counterfactuals (only those that change rank).
    """
    from adi.core.fuzzy import normalize_criterion_values
    from adi.schemas.decision_request import OptionValue

    counterfactuals = []
    delta = perturbation_pct / 100.0

    for i, criterion in enumerate(criteria):
        weight = float(normalized_weights[i])
        raw = option.get_value(criterion.name)
        if raw is None:
            continue

        # Use pre-computed normalized value when available (avoids single-option min==max bug)
        if normalized_values is not None:
            current_norm = normalized_values.get(criterion.name)
        else:
            current_norm_values = normalize_criterion_values(criterion, [option])
            current_norm = current_norm_values.get(option.name)
        if current_norm is None:
            continue

        for direction, sign in [("increase", +1), ("decrease", -1)]:
            perturbed_raw = raw * (1 + sign * delta)
            fake_option = Option(
                name=option.name,
                values=[
                    ov if ov.criterion_name != criterion.name
                    else OptionValue(criterion_name=criterion.name, value=perturbed_raw)
                    for ov in option.values
                ],
            )
            # Normalize within the full population so min-max has meaningful range
            if all_options is not None:
                population = [
                    fake_option if o.name == option.name else o
                    for o in all_options
                ]
            else:
                population = [fake_option]
            new_norms = normalize_criterion_values(criterion, population)
            new_norm = new_norms.get(option.name)
            if new_norm is None:
                continue

            norm_delta = (new_norm - current_norm) * weight
            new_score = float(all_scores[all_option_names.index(option.name)]) + norm_delta
            new_score = max(0.0, min(1.0, new_score))

            updated_scores = all_scores.copy()
            idx = all_option_names.index(option.name)
            updated_scores[idx] = new_score
            new_rank = int(np.sum(updated_scores > new_score)) + 1
            rank_change = current_rank - new_rank  # positive = improved

            if rank_change != 0:
                counterfactuals.append(
                    CounterfactualItem(
                        criterion_name=criterion.name,
                        direction=direction,
                        delta_pct=perturbation_pct,
                        new_rank=new_rank,
                        rank_change=rank_change,
                    )
                )

    return counterfactuals

"""ADI Uncertainty module — confidence propagation and robustness analysis.

Computes per-option confidence scores based on:
- Evidence quality for each criterion value
- Missing value penalties (proportional to criterion weight)
- Robustness: how stable the ranking is under weight perturbations
"""

from __future__ import annotations

import numpy as np

from adi.core.policy import Policy
from adi.schemas.decision_request import Criterion, Option


def compute_evidence_quality(
    option: Option,
    criterion: Criterion,
) -> float:
    """
    Average evidence quality for an option-criterion pair.
    If option has explicit confidence for this criterion, use it; else derive from evidence.
    Returns 0.5 (neutral) if no evidence and no explicit confidence.
    """
    explicit = option.get_confidence(criterion.name)
    if explicit is not None:
        return float(explicit)
    evidence = option.get_evidence(criterion.name)
    if not evidence:
        return 0.5
    return float(np.mean([e.quality for e in evidence]))


def compute_option_confidence(
    option: Option,
    criteria: list[Criterion],
    normalized_weights: np.ndarray,
    policy: Policy,
) -> float:
    """
    Compute confidence score [0, 1] for a single option.

    Confidence reduces when:
    - Criterion values are missing (missingness_penalty_factor)
    - Evidence quality is low (uncertainty_penalty_factor)

    Base confidence = 1.0, reduced by penalties.
    """
    confidence = 1.0

    for i, criterion in enumerate(criteria):
        weight = float(normalized_weights[i])
        raw_value = option.get_value(criterion.name)

        if raw_value is None:
            # Missing value: penalty proportional to criterion weight
            confidence -= policy.missingness_penalty_factor * weight
        else:
            # Low evidence quality: penalty proportional to uncertainty factor
            eq = compute_evidence_quality(option, criterion)
            quality_deficit = 1.0 - eq  # how much quality is missing
            confidence -= policy.uncertainty_penalty_factor * quality_deficit * weight

    return max(0.0, min(1.0, confidence))


def compute_all_confidences(
    options: list[Option],
    criteria: list[Criterion],
    normalized_weights: np.ndarray,
    policy: Policy,
) -> np.ndarray:
    """
    Compute confidence scores for all options.
    Returns (n_options,) array of confidence values.
    """
    return np.array([
        compute_option_confidence(option, criteria, normalized_weights, policy)
        for option in options
    ])


def apply_confidence_adjustment(
    scores: np.ndarray,
    confidences: np.ndarray,
    uncertainty_penalty_factor: float,
) -> np.ndarray:
    """
    Adjust final scores by confidence. Low-confidence options score lower.
    adjusted = score * (1 - penalty_factor * (1 - confidence))
    """
    penalties = 1.0 - uncertainty_penalty_factor * (1.0 - confidences)
    return scores * np.clip(penalties, 0.0, 1.0)


def robustness_check(
    matrix: np.ndarray,
    weights: np.ndarray,
    option_names: list[str],
    perturbation_pct: float = 10.0,
) -> tuple[float, list[tuple[str, str]], dict[str, float]]:
    """
    Sensitivity analysis: how stable is the ranking under ±perturbation_pct weight shifts?

    For each criterion, perturb its weight up and down, renormalize all weights,
    recompute scores, and check if any pairwise rankings change.

    Returns:
        stability_score: fraction of perturbations that preserve the ranking [0, 1]
        unstable_pairs: list of (option_a, option_b) pairs whose ranking flips
        weight_sensitivity: {criterion_name: fraction of perturbations causing rank change}
    """
    from adi.core.mcda import weighted_sum_score

    n_crit = len(weights)
    criterion_indices = list(range(n_crit))
    delta = perturbation_pct / 100.0

    baseline_scores = weighted_sum_score(matrix, weights)
    baseline_ranks = _ranks_from_scores(baseline_scores)

    unstable_pairs_set: set[tuple[str, str]] = set()
    sensitivity_per_crit: dict[int, int] = {i: 0 for i in criterion_indices}
    total_perturbations = n_crit * 2  # up and down per criterion

    for i in criterion_indices:
        for sign in [+1, -1]:
            perturbed = weights.copy()
            perturbed[i] = max(0.001, perturbed[i] * (1 + sign * delta))
            perturbed = perturbed / perturbed.sum()

            p_scores = weighted_sum_score(matrix, perturbed)
            p_ranks = _ranks_from_scores(p_scores)

            if not np.array_equal(baseline_ranks, p_ranks):
                sensitivity_per_crit[i] += 1
                # find which pairs flipped
                n = len(option_names)
                for a in range(n):
                    for b in range(a + 1, n):
                        if (baseline_ranks[a] < baseline_ranks[b]) != (p_ranks[a] < p_ranks[b]):
                            pair = (option_names[a], option_names[b])
                            unstable_pairs_set.add(pair)

    stable_perturbations = sum(
        1 for count in sensitivity_per_crit.values() if count == 0
    ) * 2  # both up+down were stable
    stability_score = float(stable_perturbations) / total_perturbations if total_perturbations > 0 else 1.0

    weight_sensitivity = {
        str(i): float(sensitivity_per_crit[i]) / 2.0
        for i in criterion_indices
    }

    return (
        stability_score,
        list(unstable_pairs_set),
        weight_sensitivity,
    )


def _ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """Convert scores to rank array (0-indexed). Higher score = lower rank number."""
    order = np.argsort(-scores)  # descending
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    return ranks

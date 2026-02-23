"""ADI MCDA module — weighted scoring and TOPSIS.

Implements two scoring methods:
1. Weighted Sum Model (WSM): simple, interpretable
2. TOPSIS: distance-based, captures how close to ideal

Policy modifiers are applied after base scoring:
- risk_averse: variance penalty
- exploratory: exploration bonus for diverse options
"""

from __future__ import annotations

import numpy as np

from adi.core.policy import Policy, PolicyStrategy
from adi.schemas.decision_request import Criterion, Option


def build_confidence_matrix(
    options: list[Option],
    criteria: list[Criterion],
) -> np.ndarray:
    """
    Build (n_options, n_criteria) matrix of per-cell confidence [0, 1].
    Uses OptionValue.confidence if set, else mean of evidence.quality, else 0.5.
    """
    from adi.core.fuzzy import _effective_confidence

    n_opts = len(options)
    n_crit = len(criteria)
    out = np.zeros((n_opts, n_crit), dtype=np.float64)
    for i, opt in enumerate(options):
        for j, c in enumerate(criteria):
            out[i, j] = _effective_confidence(opt, c.name)
    return out


def compute_weights(criteria: list[Criterion]) -> np.ndarray:
    """Return normalized weight vector summing to 1."""
    weights = np.array([c.weight for c in criteria], dtype=float)
    total = weights.sum()
    if total == 0:
        return np.ones(len(criteria)) / len(criteria)
    return weights / total


def weighted_sum_score(
    matrix: np.ndarray,
    weights: np.ndarray,
    missing_fill: float = 0.0,
) -> np.ndarray:
    """
    Compute weighted sum score for each option.

    matrix: (n_options, n_criteria), NaN = missing
    weights: normalized (n_criteria,)
    missing_fill: value to substitute for NaN before scoring (applied at caller's weight)

    Returns: (n_options,) scores in [0, 1]
    """
    filled = np.where(np.isnan(matrix), missing_fill, matrix)
    return np.dot(filled, weights)


def topsis_score(
    matrix: np.ndarray,
    weights: np.ndarray,
    missing_fill: float = 0.5,
) -> np.ndarray:
    """
    TOPSIS: Technique for Order of Preference by Similarity to Ideal Solution.

    Steps:
    1. Fill missing values
    2. Weight the matrix
    3. Find ideal best (max) and ideal worst (min) per criterion
    4. Compute Euclidean distance to ideal best/worst
    5. Score = d_worst / (d_best + d_worst)

    Returns: (n_options,) scores in [0, 1], higher = better
    """
    filled = np.where(np.isnan(matrix), missing_fill, matrix)
    weighted = filled * weights[np.newaxis, :]

    ideal_best = np.max(weighted, axis=0)
    ideal_worst = np.min(weighted, axis=0)

    d_best = np.sqrt(np.sum((weighted - ideal_best) ** 2, axis=1))
    d_worst = np.sqrt(np.sum((weighted - ideal_worst) ** 2, axis=1))

    denom = d_best + d_worst
    scores = np.where(denom == 0, 0.5, d_worst / denom)
    return scores


def weighted_sum_score_with_confidence(
    matrix: np.ndarray,
    weights: np.ndarray,
    confidence_matrix: np.ndarray,
    missing_fill: float = 0.0,
) -> np.ndarray:
    """
    Confidence-aware weighted sum: S_i = sum_j w_j * v_ij * c_ij.
    matrix and confidence_matrix same shape (n_options, n_criteria).
    """
    filled = np.where(np.isnan(matrix), missing_fill, matrix)
    effective = filled * confidence_matrix
    return np.dot(effective, weights)


def topsis_score_with_confidence(
    matrix: np.ndarray,
    weights: np.ndarray,
    confidence_matrix: np.ndarray,
    missing_fill: float = 0.5,
) -> np.ndarray:
    """
    TOPSIS with per-cell confidence: weighted matrix is (filled * confidence) * weights.
    """
    filled = np.where(np.isnan(matrix), missing_fill, matrix)
    weighted = filled * confidence_matrix * weights[np.newaxis, :]

    ideal_best = np.max(weighted, axis=0)
    ideal_worst = np.min(weighted, axis=0)

    d_best = np.sqrt(np.sum((weighted - ideal_best) ** 2, axis=1))
    d_worst = np.sqrt(np.sum((weighted - ideal_worst) ** 2, axis=1))

    denom = d_best + d_worst
    scores = np.where(denom == 0, 0.5, d_worst / denom)
    return scores


def apply_variance_penalty(
    scores: np.ndarray,
    matrix: np.ndarray,
    weights: np.ndarray,
    variance_penalty_factor: float,
) -> np.ndarray:
    """
    RISK_AVERSE modifier: penalize options with high variance across weighted criteria.
    An option with very uneven criterion scores is riskier than one with balanced scores.
    """
    if variance_penalty_factor == 0.0:
        return scores

    filled = np.where(np.isnan(matrix), 0.0, matrix)
    weighted_vals = filled * weights[np.newaxis, :]
    # variance per option (higher = riskier)
    variance = np.var(weighted_vals, axis=1)
    max_var = variance.max()
    if max_var == 0.0:
        return scores

    normalized_var = variance / max_var
    penalties = 1.0 - (variance_penalty_factor * normalized_var)
    return scores * penalties


def apply_exploration_bonus(
    scores: np.ndarray,
    exploration_bonus_factor: float,
) -> np.ndarray:
    """
    EXPLORATORY modifier: boost options that are 'different' from the frontrunner.
    Options far from the current max score get a small bonus to encourage diversity.
    """
    if exploration_bonus_factor == 0.0:
        return scores

    max_score = scores.max()
    if max_score == 0.0:
        return scores

    distance_from_top = max_score - scores
    normalized_distance = distance_from_top / max_score
    bonuses = exploration_bonus_factor * normalized_distance
    return scores + bonuses


def score_options(
    matrix: np.ndarray,
    weights: np.ndarray,
    policy: Policy,
    confidence_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full scoring pipeline applying policy modifiers.
    When policy.use_cell_confidence and confidence_matrix is provided, uses
    per-cell confidence in the score formula; otherwise classic WSM/TOPSIS.
    Returns:
        raw_scores: scores before policy modifiers
        final_scores: scores after all policy adjustments
    """
    use_cell = getattr(policy, "use_cell_confidence", False) and confidence_matrix is not None

    if use_cell:
        if policy.topsis_enabled:
            raw_scores = topsis_score_with_confidence(matrix, weights, confidence_matrix)
        else:
            raw_scores = weighted_sum_score_with_confidence(
                matrix, weights, confidence_matrix
            )
    else:
        if policy.topsis_enabled:
            raw_scores = topsis_score(matrix, weights)
        else:
            raw_scores = weighted_sum_score(matrix, weights)

    adjusted = raw_scores.copy()

    if policy.strategy == PolicyStrategy.RISK_AVERSE:
        adjusted = apply_variance_penalty(
            adjusted, matrix, weights, policy.variance_penalty_factor
        )
    elif policy.strategy == PolicyStrategy.EXPLORATORY:
        adjusted = apply_exploration_bonus(adjusted, policy.exploration_bonus_factor)

    return raw_scores, adjusted

"""ADI Fuzzy module — fuzzification and normalization of criterion values.

Converts raw numeric values to normalized [0, 1] scores per criterion,
respecting direction (benefit vs cost) and optional fuzzy membership sets.
Supports confidence-aware widening (low confidence = wider MFs = more ambiguity).
"""

from __future__ import annotations

import math

import numpy as np

from adi.schemas.decision_request import (
    Criterion,
    CriterionDirection,
    FuzzySet,
    FuzzySetDefinition,
    Option,
)

# How much low confidence widens membership functions (1 + (1-conf)*this).
SPREAD_MULTIPLIER = 2.0
DEFUZZIFY_RESOLUTION = 500


def triangular_membership(x: float, a: float, b: float, c: float) -> float:
    """
    Triangular membership function μ(x) for point x given triangle (a, b, c).
    Returns value in [0, 1].
    """
    if math.isnan(x):
        return 0.0
    if x <= a or x >= c:
        return 0.0
    if a == b == c:
        return 1.0 if x == b else 0.0
    if x <= b:
        return (x - a) / (b - a) if b != a else 1.0
    return (c - x) / (c - b) if c != b else 1.0


def trapezoidal_membership(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoidal MF: 0 outside [a, d], 1.0 on [b, c]."""
    if math.isnan(x):
        return 0.0
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b != a else 1.0
    return (d - x) / (d - c) if d != c else 1.0


def gaussian_membership(x: float, mean: float, sigma: float) -> float:
    """Gaussian MF: exp(-0.5 * ((x - mean) / sigma)^2)."""
    if math.isnan(x) or sigma == 0.0:
        return 1.0 if not math.isnan(x) and x == mean else 0.0
    return math.exp(-0.5 * ((x - mean) / sigma) ** 2)


def _compute_membership(x: float, fs: FuzzySetDefinition) -> float:
    """Compute membership degree for x in fuzzy set definition."""
    t = fs.type
    p = fs.params
    if t == "triangular":
        if len(p) != 3:
            raise ValueError(f"Triangular MF needs 3 params, got {len(p)}")
        return triangular_membership(x, p[0], p[1], p[2])
    if t == "trapezoidal":
        if len(p) != 4:
            raise ValueError(f"Trapezoidal MF needs 4 params, got {len(p)}")
        return trapezoidal_membership(x, p[0], p[1], p[2], p[3])
    if t == "gaussian":
        if len(p) != 2:
            raise ValueError(f"Gaussian MF needs 2 params, got {len(p)}")
        return gaussian_membership(x, p[0], p[1])
    raise ValueError(f"Unknown MF type: {t!r}")


def _widen_params(fs: FuzzySetDefinition, spread_factor: float) -> list[float]:
    """Return widened MF params. spread_factor > 1 widens the sets."""
    p = list(fs.params)
    if fs.type == "triangular":
        a, b, c = p
        p[0] = b - (b - a) * spread_factor
        p[2] = b + (c - b) * spread_factor
    elif fs.type == "trapezoidal":
        a, b, c, d = p
        center = (b + c) / 2.0
        p[0] = center - (center - a) * spread_factor
        p[1] = center - (center - b) * spread_factor
        p[2] = center + (c - center) * spread_factor
        p[3] = center + (d - center) * spread_factor
    elif fs.type == "gaussian":
        p[1] = p[1] * spread_factor
    return p


def _defuzzify_centroid_definitions(
    memberships: dict[str, float],
    fuzzy_sets: list[FuzzySetDefinition],
) -> float:
    """Centroid defuzzification over combined fuzzy region (sampled)."""
    if not memberships or not fuzzy_sets:
        return 0.5
    all_bounds: list[float] = []
    for fs in fuzzy_sets:
        if fs.type == "triangular":
            all_bounds.extend([fs.params[0], fs.params[2]])
        elif fs.type == "trapezoidal":
            all_bounds.extend([fs.params[0], fs.params[3]])
        elif fs.type == "gaussian":
            mean, sigma = fs.params
            all_bounds.extend([mean - 4 * sigma, mean + 4 * sigma])
    if not all_bounds:
        return 0.5
    lo, hi = min(all_bounds), max(all_bounds)
    if lo == hi:
        return lo
    x_vals = np.linspace(lo, hi, DEFUZZIFY_RESOLUTION)
    aggregated = np.zeros_like(x_vals)
    for fs in fuzzy_sets:
        mu_weight = memberships.get(fs.name, 0.0)
        if mu_weight == 0.0:
            continue
        for i, x in enumerate(x_vals):
            raw = _compute_membership(float(x), fs)
            aggregated[i] = max(aggregated[i], min(raw, mu_weight))
    total_area = float(np.sum(aggregated))
    if total_area == 0.0:
        return (lo + hi) / 2.0
    return float(np.sum(x_vals * aggregated) / total_area)


def fuzzify_value_with_confidence(
    value: float,
    confidence: float,
    fuzzy_sets: list[FuzzySetDefinition],
    spread_multiplier: float = SPREAD_MULTIPLIER,
) -> float:
    """
    Fuzzify a numeric value with confidence-aware widening.
    Low confidence widens MFs so that multiple sets activate (genuine ambiguity).
    Returns crisp value in [0, 1].
    """
    if not fuzzy_sets:
        return 0.5
    if math.isnan(value):
        return 0.5
    spread_factor = 1.0 + (1.0 - confidence) * spread_multiplier
    result: dict[str, float] = {}
    for fs in fuzzy_sets:
        if spread_factor == 1.0:
            result[fs.name] = _compute_membership(value, fs)
        else:
            widened = _widen_params(fs, spread_factor)
            if fs.type == "triangular":
                result[fs.name] = triangular_membership(value, *widened)
            elif fs.type == "trapezoidal":
                result[fs.name] = trapezoidal_membership(value, *widened)
            elif fs.type == "gaussian":
                result[fs.name] = gaussian_membership(value, *widened)
            else:
                result[fs.name] = _compute_membership(value, fs)
    return _defuzzify_centroid_definitions(result, fuzzy_sets)


def defuzzify_centroid(memberships: dict[str, float], fuzzy_set: FuzzySet) -> float:
    """
    Simple centroid defuzzification: weighted average of set centers.
    Returns a value in [0, 1] representing crisp degree.
    """
    sets = {
        "low": fuzzy_set.low,
        "medium": fuzzy_set.medium,
        "high": fuzzy_set.high,
    }
    numerator = 0.0
    denominator = 0.0
    for label, (a, b, c) in sets.items():
        mu = memberships.get(label, 0.0)
        center = b  # apex of triangle
        numerator += mu * center
        denominator += mu
    if denominator == 0.0:
        return 0.5
    return max(0.0, min(1.0, numerator / denominator))


def fuzzify_value(value: float, fuzzy_set: FuzzySet) -> float:
    """
    Convert a raw value through fuzzy membership → defuzzify to [0, 1].
    The input value is assumed to already be in the fuzzy set's domain.
    """
    memberships = {
        "low": triangular_membership(value, *fuzzy_set.low),
        "medium": triangular_membership(value, *fuzzy_set.medium),
        "high": triangular_membership(value, *fuzzy_set.high),
    }
    return defuzzify_centroid(memberships, fuzzy_set)


def min_max_normalize(
    value: float,
    min_val: float,
    max_val: float,
    direction: CriterionDirection = CriterionDirection.BENEFIT,
) -> float:
    """
    Min-max normalization to [0, 1].
    For BENEFIT: higher = better (normalized value closer to 1 = better)
    For COST:    lower = better (inverted: 1 - normalized)
    """
    if math.isclose(max_val, min_val):
        return 0.5
    normalized = (value - min_val) / (max_val - min_val)
    normalized = max(0.0, min(1.0, normalized))
    if direction == CriterionDirection.COST:
        normalized = 1.0 - normalized
    return normalized


def _effective_confidence(option: Option, criterion_name: str) -> float:
    """Per-cell confidence: explicit or mean evidence quality. No dependency on uncertainty module."""
    c = option.get_confidence(criterion_name)
    if c is not None:
        return float(c)
    ev = option.get_evidence(criterion_name)
    if not ev:
        return 0.5
    return float(np.mean([e.quality for e in ev]))


def normalize_criterion_values(
    criterion: Criterion,
    options: list[Option],
) -> dict[str, float | None]:
    """
    Normalize all option values for a single criterion.
    Uses fuzzy_set_definitions with confidence widening when set; else fuzzy_set or min-max.
    Returns a dict: {option_name: normalized_value | None (if missing)}
    """
    raw_values: dict[str, float | None] = {}
    for opt in options:
        raw_values[opt.name] = opt.get_value(criterion.name)

    present_values = [v for v in raw_values.values() if v is not None]

    if not present_values:
        return {opt.name: None for opt in options}

    min_val = float(np.min(present_values))
    max_val = float(np.max(present_values))

    result: dict[str, float | None] = {}
    for opt in options:
        raw = raw_values[opt.name]
        if raw is None:
            result[opt.name] = None
            continue

        if criterion.fuzzy_set_definitions:
            conf = _effective_confidence(opt, criterion.name)
            result[opt.name] = fuzzify_value_with_confidence(
                raw, conf, criterion.fuzzy_set_definitions
            )
        elif criterion.fuzzy_set is not None:
            result[opt.name] = fuzzify_value(raw, criterion.fuzzy_set)
        else:
            result[opt.name] = min_max_normalize(raw, min_val, max_val, criterion.direction)

    return result


def build_normalized_matrix(
    criteria: list[Criterion],
    options: list[Option],
) -> tuple[np.ndarray, list[str], list[str], dict[str, dict[str, float | None]]]:
    """
    Build the full normalized decision matrix.

    Returns:
        matrix: shape (n_options, n_criteria), NaN for missing values
        option_names: list of option names (row order)
        criterion_names: list of criterion names (column order)
        normalized_map: {option_name: {criterion_name: normalized_value | None}}
    """
    option_names = [o.name for o in options]
    criterion_names = [c.name for c in criteria]

    normalized_map: dict[str, dict[str, float | None]] = {o.name: {} for o in options}

    for criterion in criteria:
        col = normalize_criterion_values(criterion, options)
        for opt_name, val in col.items():
            normalized_map[opt_name][criterion.name] = val

    n_opts = len(option_names)
    n_crit = len(criterion_names)
    matrix = np.full((n_opts, n_crit), np.nan)

    for i, opt_name in enumerate(option_names):
        for j, crit_name in enumerate(criterion_names):
            val = normalized_map[opt_name].get(crit_name)
            if val is not None:
                matrix[i, j] = val

    return matrix, option_names, criterion_names, normalized_map

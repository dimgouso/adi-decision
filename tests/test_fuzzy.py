"""Tests for ADI fuzzy normalization module."""

import numpy as np
import pytest

from adi.core.fuzzy import (
    build_normalized_matrix,
    fuzzify_value,
    min_max_normalize,
    normalize_criterion_values,
    triangular_membership,
)
from adi.schemas.decision_request import (
    Criterion,
    CriterionDirection,
    FuzzySet,
    Option,
    OptionValue,
)


def test_triangular_membership_apex():
    assert triangular_membership(5.0, 3.0, 5.0, 7.0) == 1.0


def test_triangular_membership_below():
    assert triangular_membership(1.0, 3.0, 5.0, 7.0) == 0.0


def test_triangular_membership_above():
    assert triangular_membership(9.0, 3.0, 5.0, 7.0) == 0.0


def test_triangular_membership_left_slope():
    val = triangular_membership(4.0, 3.0, 5.0, 7.0)
    assert 0.0 < val < 1.0


def test_min_max_normalize_benefit():
    val = min_max_normalize(7.0, 0.0, 10.0, CriterionDirection.BENEFIT)
    assert abs(val - 0.7) < 1e-9


def test_min_max_normalize_cost():
    val = min_max_normalize(7.0, 0.0, 10.0, CriterionDirection.COST)
    assert abs(val - 0.3) < 1e-9


def test_min_max_normalize_equal_range():
    val = min_max_normalize(5.0, 5.0, 5.0)
    assert val == 0.5


def test_normalize_criterion_values_benefit():
    criterion = Criterion(name="speed", weight=0.5, direction=CriterionDirection.BENEFIT)
    options = [
        Option(name="A", values=[OptionValue(criterion_name="speed", value=10.0)]),
        Option(name="B", values=[OptionValue(criterion_name="speed", value=0.0)]),
    ]
    result = normalize_criterion_values(criterion, options)
    assert abs(result["A"] - 1.0) < 1e-9
    assert abs(result["B"] - 0.0) < 1e-9


def test_normalize_handles_missing():
    criterion = Criterion(name="speed", weight=0.5)
    options = [
        Option(name="A", values=[OptionValue(criterion_name="speed", value=10.0)]),
        Option(name="B", values=[]),  # missing
    ]
    result = normalize_criterion_values(criterion, options)
    assert result["B"] is None
    assert result["A"] is not None


def test_build_normalized_matrix_shape():
    criteria = [
        Criterion(name="speed", weight=0.6),
        Criterion(name="cost", weight=0.4, direction=CriterionDirection.COST),
    ]
    options = [
        Option(name="A", values=[
            OptionValue(criterion_name="speed", value=8.0),
            OptionValue(criterion_name="cost", value=2.0),
        ]),
        Option(name="B", values=[
            OptionValue(criterion_name="speed", value=4.0),
            OptionValue(criterion_name="cost", value=6.0),
        ]),
    ]
    matrix, opt_names, crit_names, norm_map = build_normalized_matrix(criteria, options)
    assert matrix.shape == (2, 2)
    assert opt_names == ["A", "B"]
    assert crit_names == ["speed", "cost"]
    assert not np.any(np.isnan(matrix))


def test_fuzzify_with_fuzzy_set():
    fset = FuzzySet(
        low=(0.0, 0.0, 5.0),
        medium=(3.0, 5.0, 7.0),
        high=(5.0, 10.0, 10.0),
    )
    val = fuzzify_value(5.0, fset)
    assert 0.0 <= val <= 1.0

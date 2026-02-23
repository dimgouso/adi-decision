"""Tests for ADI explain module."""

import numpy as np
import pytest

from adi.core.explain import compute_criterion_contributions, evaluate_constraints
from adi.schemas.decision_request import (
    Constraint,
    ConstraintType,
    Criterion,
    CriterionDirection,
    Option,
    OptionValue,
)


def make_option(name: str, speed: float, cost: float) -> Option:
    return Option(
        name=name,
        values=[
            OptionValue(criterion_name="speed", value=speed),
            OptionValue(criterion_name="cost", value=cost),
        ],
    )


def test_contributions_count():
    option = make_option("A", 8.0, 3.0)
    criteria = [
        Criterion(name="speed", weight=0.6),
        Criterion(name="cost", weight=0.4, direction=CriterionDirection.COST),
    ]
    weights = np.array([0.6, 0.4])
    norm_vals = {"speed": 1.0, "cost": 0.8}
    contribs = compute_criterion_contributions(option, criteria, weights, norm_vals, 0.92)
    assert len(contribs) == 2
    assert contribs[0].criterion_name == "speed"
    assert contribs[1].criterion_name == "cost"


def test_contributions_missing_flag():
    option = Option(name="A", values=[OptionValue(criterion_name="speed", value=8.0)])
    criteria = [
        Criterion(name="speed", weight=0.6),
        Criterion(name="cost", weight=0.4),
    ]
    weights = np.array([0.6, 0.4])
    norm_vals = {"speed": 1.0, "cost": None}
    contribs = compute_criterion_contributions(option, criteria, weights, norm_vals, 0.6)
    cost_contrib = next(c for c in contribs if c.criterion_name == "cost")
    assert cost_contrib.missing


def test_must_exclude_constraint():
    option = Option(name="A", values=[])
    constraints = [
        Constraint(constraint_type=ConstraintType.MUST_EXCLUDE, option_name="A", hard=True)
    ]
    eliminated, reports = evaluate_constraints(option, constraints)
    assert eliminated
    assert len(reports) == 1
    assert not reports[0].satisfied


def test_must_include_not_eliminates():
    option = Option(name="A", values=[])
    constraints = [
        Constraint(constraint_type=ConstraintType.MUST_INCLUDE, option_name="A", hard=True)
    ]
    eliminated, reports = evaluate_constraints(option, constraints)
    assert not eliminated


def test_min_value_constraint_pass():
    option = make_option("A", speed=9.0, cost=1.0)
    constraints = [
        Constraint(
            constraint_type=ConstraintType.MIN_VALUE,
            criterion_name="speed",
            threshold=5.0,
            hard=True,
        )
    ]
    eliminated, reports = evaluate_constraints(option, constraints)
    assert not eliminated
    assert reports[0].satisfied


def test_min_value_constraint_fail():
    option = make_option("A", speed=3.0, cost=1.0)
    constraints = [
        Constraint(
            constraint_type=ConstraintType.MIN_VALUE,
            criterion_name="speed",
            threshold=5.0,
            hard=True,
        )
    ]
    eliminated, reports = evaluate_constraints(option, constraints)
    assert eliminated
    assert not reports[0].satisfied


def test_constraint_not_applicable_to_other_option():
    option = make_option("B", speed=3.0, cost=1.0)
    constraints = [
        Constraint(
            constraint_type=ConstraintType.MUST_EXCLUDE,
            option_name="A",  # different option
            hard=True,
        )
    ]
    eliminated, reports = evaluate_constraints(option, constraints)
    assert not eliminated
    assert len(reports) == 0

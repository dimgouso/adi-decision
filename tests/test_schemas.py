"""Tests for ADI schemas — DecisionRequest and DecisionOutput."""

import pytest
from pydantic import ValidationError

from adi.schemas.decision_request import (
    Constraint,
    ConstraintType,
    Criterion,
    CriterionDirection,
    DecisionRequest,
    EvidenceItem,
    Option,
    OptionValue,
)


def make_basic_request() -> DecisionRequest:
    return DecisionRequest(
        options=[
            Option(
                name="Option A",
                values=[
                    OptionValue(criterion_name="speed", value=8.0),
                    OptionValue(criterion_name="cost", value=3.0),
                ],
            ),
            Option(
                name="Option B",
                values=[
                    OptionValue(criterion_name="speed", value=5.0),
                    OptionValue(criterion_name="cost", value=7.0),
                ],
            ),
        ],
        criteria=[
            Criterion(name="speed", weight=0.6, direction=CriterionDirection.BENEFIT),
            Criterion(name="cost", weight=0.4, direction=CriterionDirection.COST),
        ],
    )


def test_valid_request():
    req = make_basic_request()
    assert len(req.options) == 2
    assert len(req.criteria) == 2


def test_requires_at_least_two_options():
    with pytest.raises(ValidationError):
        DecisionRequest(
            options=[Option(name="only one", values=[])],
            criteria=[Criterion(name="c", weight=0.5)],
        )


def test_unknown_criterion_in_option_value():
    with pytest.raises(ValidationError, match="unknown criterion"):
        DecisionRequest(
            options=[
                Option(name="A", values=[OptionValue(criterion_name="unknown", value=1.0)]),
                Option(name="B", values=[]),
            ],
            criteria=[Criterion(name="speed", weight=1.0)],
        )


def test_invalid_weight_zero():
    with pytest.raises(ValidationError):
        Criterion(name="speed", weight=0.0)


def test_constraint_unknown_option():
    with pytest.raises(ValidationError, match="unknown option"):
        DecisionRequest(
            options=[
                Option(name="A", values=[]),
                Option(name="B", values=[]),
            ],
            criteria=[Criterion(name="speed", weight=1.0)],
            constraints=[
                Constraint(
                    constraint_type=ConstraintType.MUST_EXCLUDE,
                    option_name="DOES_NOT_EXIST",
                )
            ],
        )


def test_evidence_item_quality_range():
    with pytest.raises(ValidationError):
        EvidenceItem(source="paper", quality=1.5)


def test_option_get_value():
    opt = Option(
        name="X",
        values=[OptionValue(criterion_name="speed", value=7.5)],
    )
    assert opt.get_value("speed") == 7.5
    assert opt.get_value("missing") is None


def test_default_policy_name():
    req = make_basic_request()
    assert req.policy_name == "balanced"

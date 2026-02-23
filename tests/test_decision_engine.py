"""Tests for ADI decision engine — end-to-end without LLM."""

import pytest

from adi.core.decision_engine import decide
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


def make_request(
    policy_name: str = "balanced",
    constraints: list | None = None,
) -> DecisionRequest:
    return DecisionRequest(
        options=[
            Option(
                name="Option A",
                values=[
                    OptionValue(
                        criterion_name="performance",
                        value=9.0,
                        evidence=[EvidenceItem(source="benchmark_2024", quality=0.9)],
                    ),
                    OptionValue(criterion_name="cost", value=5000.0),
                ],
            ),
            Option(
                name="Option B",
                values=[
                    OptionValue(criterion_name="performance", value=6.0),
                    OptionValue(criterion_name="cost", value=2000.0),
                ],
            ),
            Option(
                name="Option C",
                values=[
                    OptionValue(criterion_name="performance", value=7.5),
                    OptionValue(
                        criterion_name="cost",
                        value=3500.0,
                        evidence=[EvidenceItem(source="quote_sheet", quality=0.7)],
                    ),
                ],
            ),
        ],
        criteria=[
            Criterion(name="performance", weight=0.6, direction=CriterionDirection.BENEFIT),
            Criterion(name="cost", weight=0.4, direction=CriterionDirection.COST),
        ],
        policy_name=policy_name,
        constraints=constraints or [],
        context="Unit test decision",
    )


def test_decide_returns_valid_output():
    result = decide(make_request())
    assert result.best_option in {"Option A", "Option B", "Option C"}
    assert len(result.ranking) == 3
    assert result.policy_applied == "balanced"


def test_decide_ranks_are_unique():
    result = decide(make_request())
    ranks = [r.rank for r in result.ranking]
    assert sorted(ranks) == list(range(1, len(ranks) + 1))


def test_decide_best_is_rank_one():
    result = decide(make_request())
    top = next(r for r in result.ranking if r.rank == 1)
    assert top.option_name == result.best_option


def test_decide_confidence_in_range():
    result = decide(make_request())
    for r in result.ranking:
        assert 0.0 <= r.confidence <= 1.0
    assert 0.0 <= result.overall_confidence <= 1.0


def test_decide_criterion_contributions_sum():
    result = decide(make_request())
    for r in result.ranking:
        if not r.eliminated:
            total_pct = sum(c.contribution_pct for c in r.criterion_contributions)
            assert abs(total_pct - 100.0) < 1.0  # allow rounding


def test_decide_hard_constraint_eliminates():
    constraints = [
        Constraint(
            constraint_type=ConstraintType.MUST_EXCLUDE,
            option_name="Option A",
            hard=True,
        )
    ]
    result = decide(make_request(constraints=constraints))
    option_a = next(r for r in result.ranking if r.option_name == "Option A")
    assert option_a.eliminated
    assert result.best_option != "Option A"


def test_decide_min_value_constraint():
    constraints = [
        Constraint(
            constraint_type=ConstraintType.MIN_VALUE,
            criterion_name="performance",
            threshold=8.0,
            hard=True,
        )
    ]
    result = decide(make_request(constraints=constraints))
    for r in result.ranking:
        if r.eliminated:
            assert r.option_name != "Option A"  # Option A has performance=9.0, survives


def test_decide_risk_averse_policy():
    result = decide(make_request(policy_name="risk_averse"))
    assert result.policy_applied == "risk_averse"
    assert "variance_penalty_factor" in result.policy_parameters


def test_decide_exploratory_policy():
    result = decide(make_request(policy_name="exploratory"))
    assert result.policy_applied == "exploratory"


def test_decide_sensitivity_report():
    result = decide(make_request())
    assert 0.0 <= result.sensitivity.overall_stability <= 1.0
    assert isinstance(result.sensitivity.unstable_pairs, list)


def test_decide_policy_parameters_auditable():
    result = decide(make_request())
    params = result.policy_parameters
    assert "strategy" in params
    assert "uncertainty_penalty_factor" in params
    assert "missingness_penalty_factor" in params


def test_decide_with_missing_values():
    request = DecisionRequest(
        options=[
            Option(
                name="Complete",
                values=[
                    OptionValue(criterion_name="speed", value=8.0),
                    OptionValue(criterion_name="cost", value=3.0),
                ],
            ),
            Option(
                name="Incomplete",
                values=[
                    OptionValue(criterion_name="speed", value=9.0),
                    # cost is missing
                ],
            ),
        ],
        criteria=[
            Criterion(name="speed", weight=0.5),
            Criterion(name="cost", weight=0.5, direction=CriterionDirection.COST),
        ],
    )
    result = decide(request)
    incomplete = next(r for r in result.ranking if r.option_name == "Incomplete")
    assert incomplete.confidence < 1.0

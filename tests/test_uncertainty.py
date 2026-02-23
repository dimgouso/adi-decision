"""Tests for ADI uncertainty module."""

import numpy as np
import pytest

from adi.core.policy import Policy
from adi.core.uncertainty import (
    apply_confidence_adjustment,
    compute_all_confidences,
    compute_evidence_quality,
    compute_option_confidence,
    robustness_check,
)
from adi.schemas.decision_request import (
    Criterion,
    EvidenceItem,
    Option,
    OptionValue,
)


def make_option_with_evidence(quality: float) -> Option:
    return Option(
        name="A",
        values=[
            OptionValue(
                criterion_name="speed",
                value=8.0,
                evidence=[EvidenceItem(source="paper1", quality=quality)],
            )
        ],
    )


def test_evidence_quality_with_evidence():
    option = make_option_with_evidence(0.9)
    criterion = Criterion(name="speed", weight=0.5)
    q = compute_evidence_quality(option, criterion)
    assert abs(q - 0.9) < 1e-9


def test_evidence_quality_no_evidence():
    option = Option(name="A", values=[OptionValue(criterion_name="speed", value=5.0)])
    criterion = Criterion(name="speed", weight=0.5)
    q = compute_evidence_quality(option, criterion)
    assert q == 0.5


def test_confidence_perfect_evidence():
    option = make_option_with_evidence(1.0)
    criteria = [Criterion(name="speed", weight=1.0)]
    weights = np.array([1.0])
    policy = Policy.balanced()
    conf = compute_option_confidence(option, criteria, weights, policy)
    assert conf == pytest.approx(1.0, abs=1e-6)


def test_confidence_missing_value():
    option = Option(name="A", values=[])  # missing all criteria
    criteria = [Criterion(name="speed", weight=1.0)]
    weights = np.array([1.0])
    policy = Policy.balanced()
    conf = compute_option_confidence(option, criteria, weights, policy)
    assert conf < 1.0


def test_confidence_clamps_to_zero():
    option = Option(name="A", values=[])
    criteria = [Criterion(name="speed", weight=1.0)]
    weights = np.array([1.0])
    policy = Policy(
        name="strict",
        missingness_penalty_factor=1.0,
        uncertainty_penalty_factor=1.0,
    )
    conf = compute_option_confidence(option, criteria, weights, policy)
    assert conf >= 0.0


def test_apply_confidence_adjustment():
    scores = np.array([0.8, 0.8])
    confidences = np.array([1.0, 0.5])
    adjusted = apply_confidence_adjustment(scores, confidences, 0.4)
    assert adjusted[0] > adjusted[1]


def test_compute_all_confidences():
    options = [
        make_option_with_evidence(1.0),
        make_option_with_evidence(0.2),
    ]
    # rename options
    options[0] = Option(
        name="Good",
        values=[OptionValue(criterion_name="speed", value=8.0,
                            evidence=[EvidenceItem(source="p", quality=1.0)])],
    )
    options[1] = Option(
        name="Bad",
        values=[OptionValue(criterion_name="speed", value=5.0,
                            evidence=[EvidenceItem(source="p", quality=0.2)])],
    )
    criteria = [Criterion(name="speed", weight=1.0)]
    weights = np.array([1.0])
    policy = Policy.balanced()
    confs = compute_all_confidences(options, criteria, weights, policy)
    assert confs[0] >= confs[1]


def test_robustness_check_stable():
    matrix = np.array([
        [1.0, 0.5],
        [0.0, 0.0],
    ])
    weights = np.array([0.6, 0.4])
    stability, unstable_pairs, sensitivity = robustness_check(
        matrix, weights, ["A", "B"], perturbation_pct=10.0
    )
    assert 0.0 <= stability <= 1.0
    assert isinstance(unstable_pairs, list)

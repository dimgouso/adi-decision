"""Tests for ADI MCDA module."""

import numpy as np
import pytest

from adi.core.mcda import (
    apply_exploration_bonus,
    apply_variance_penalty,
    compute_weights,
    score_options,
    topsis_score,
    weighted_sum_score,
)
from adi.core.policy import Policy, PolicyStrategy
from adi.schemas.decision_request import Criterion


def test_compute_weights_normalizes():
    criteria = [
        Criterion(name="a", weight=0.3),
        Criterion(name="b", weight=0.7),
    ]
    w = compute_weights(criteria)
    assert abs(w.sum() - 1.0) < 1e-9
    assert abs(w[0] - 0.3) < 1e-9


def test_compute_weights_unequal_sum():
    # Weights don't have to sum to 1 in isolation; compute_weights normalizes them
    criteria = [
        Criterion(name="a", weight=0.25),
        Criterion(name="b", weight=0.75),
    ]
    w = compute_weights(criteria)
    assert abs(w.sum() - 1.0) < 1e-9
    assert abs(w[0] - 0.25) < 1e-9


def test_weighted_sum_score_basic():
    matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
    weights = np.array([0.6, 0.4])
    scores = weighted_sum_score(matrix, weights)
    assert abs(scores[0] - 0.8) < 1e-9
    assert abs(scores[1] - 0.7) < 1e-9


def test_weighted_sum_with_nan():
    matrix = np.array([[1.0, np.nan], [0.5, 1.0]])
    weights = np.array([0.6, 0.4])
    scores = weighted_sum_score(matrix, weights, missing_fill=0.0)
    assert scores[0] == pytest.approx(0.6)


def test_topsis_score_best_and_worst():
    matrix = np.array([
        [1.0, 1.0],  # ideal
        [0.0, 0.0],  # anti-ideal
    ])
    weights = np.array([0.5, 0.5])
    scores = topsis_score(matrix, weights)
    assert scores[0] > scores[1]


def test_variance_penalty_reduces_uneven():
    scores = np.array([0.8, 0.8])
    matrix = np.array([
        [1.0, 0.0],  # uneven
        [0.5, 0.5],  # even
    ])
    weights = np.array([0.5, 0.5])
    penalized = apply_variance_penalty(scores, matrix, weights, variance_penalty_factor=0.5)
    # uneven option (index 0) should be more penalized
    assert penalized[0] <= penalized[1]


def test_exploration_bonus_adds_diversity():
    scores = np.array([0.9, 0.3])
    boosted = apply_exploration_bonus(scores, exploration_bonus_factor=0.2)
    # option 1 (far from leader) should get bigger boost
    assert boosted[1] > scores[1]
    assert boosted[0] >= scores[0]  # leader stays same or slightly boosted


def test_score_options_balanced_policy():
    matrix = np.array([[1.0, 0.5], [0.4, 0.9]])
    weights = np.array([0.6, 0.4])
    policy = Policy.balanced()
    raw, final = score_options(matrix, weights, policy)
    assert len(raw) == 2
    assert len(final) == 2
    assert np.all(final >= 0.0)

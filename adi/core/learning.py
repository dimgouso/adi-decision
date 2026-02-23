"""ADI Learning Engine — preference learning from user/agent feedback.

Bayesian-style multiplicative weight updates. Profiles stored as JSON.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from adi.schemas.decision_output import (
    DecisionOutput,
    FeedbackInput,
    OptionResult,
    ProfileUpdate,
)
from adi.schemas.decision_request import DecisionRequest, WeightProfile

logger = logging.getLogger(__name__)

DEFAULT_PROFILES_DIR = Path.home() / ".adi" / "profiles"
MAX_FEEDBACK_HISTORY = 100
MIN_WEIGHT = 0.01
_ACCEPT_FACTOR = 0.02
_REJECT_FACTOR = 0.05
_OVERRIDE_FACTOR = 0.08


class UserProfile(BaseModel):
    """Persistent profile stored as JSON."""

    profile_id: str
    weights: dict[str, float] = Field(default_factory=dict)
    fuzzy_overrides: dict[str, list[dict]] = Field(default_factory=dict)
    feedback_history: list[dict] = Field(default_factory=list)
    feedback_count: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total == 0:
        n = len(weights)
        return {k: 1.0 / n for k in weights}
    # Divide by total, then clamp each weight to MIN_WEIGHT floor.
    # Re-normalize after clamping so the sum remains exactly 1.0.
    clamped = {k: max(v / total, MIN_WEIGHT) for k, v in weights.items()}
    clamped_total = sum(clamped.values())
    return {k: v / clamped_total for k, v in clamped.items()}


class LearningEngine:
    """Incremental preference learning from decision feedback."""

    def __init__(self, profiles_dir: Path | str = DEFAULT_PROFILES_DIR) -> None:
        self.profiles_dir = Path(profiles_dir)

    def _get_profile_path(self, profile_id: str) -> Path:
        return self.profiles_dir / f"{profile_id}.json"

    def _load_raw_profile(self, profile_id: str) -> UserProfile | None:
        path = self._get_profile_path(profile_id)
        if not path.exists():
            return None
        try:
            return UserProfile.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning("Failed to load profile '%s': %s", profile_id, exc)
            return None

    def save_profile(self, profile: UserProfile) -> None:
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        profile.updated_at = datetime.now(timezone.utc).isoformat()
        self._get_profile_path(profile.profile_id).write_text(
            profile.model_dump_json(indent=2), encoding="utf-8"
        )

    def load_profile(self, profile_id: str) -> WeightProfile | None:
        raw = self._load_raw_profile(profile_id)
        if raw is None or not raw.weights:
            return None
        return WeightProfile(weights=raw.weights, fuzzy_overrides=raw.fuzzy_overrides)

    def process_feedback(
        self,
        feedback: FeedbackInput,
        decision: DecisionOutput,
        request: DecisionRequest,
    ) -> ProfileUpdate:
        profile_id = request.profile_id or "default"
        adjustments: dict[str, float] = {}
        ranked_map = {r.option_name: r for r in decision.ranking}
        best_ranked = ranked_map.get(decision.best_option)

        if best_ranked is None:
            return ProfileUpdate(profile_id=profile_id, weight_adjustments={}, feedback_count=0)

        if feedback.action == "accept":
            adjustments = _accept_adjustments(best_ranked)
        elif feedback.action == "reject":
            adjustments = _reject_adjustments(best_ranked)
        elif feedback.action == "override" and feedback.chosen_option:
            chosen_ranked = ranked_map.get(feedback.chosen_option)
            if chosen_ranked is not None:
                adjustments = _override_adjustments(best_ranked, chosen_ranked)

        return ProfileUpdate(
            profile_id=profile_id,
            weight_adjustments=adjustments,
            feedback_count=0,
        )

    def update_profile(self, profile_id: str, update: ProfileUpdate) -> WeightProfile:
        raw = self._load_raw_profile(profile_id)
        if raw is None:
            raw = UserProfile(profile_id=profile_id)

        for criterion, delta in update.weight_adjustments.items():
            current = raw.weights.get(criterion, 1.0)
            new_weight = current * (1.0 + delta)
            raw.weights[criterion] = max(new_weight, MIN_WEIGHT)

        raw.weights = _normalize_weights(raw.weights)

        if update.fuzzy_adjustments:
            for criterion, params in update.fuzzy_adjustments.items():
                raw.fuzzy_overrides[criterion] = [
                    {"name": "learned", "type": "triangular", "params": params}
                ]

        raw.feedback_history.append(
            {"timestamp": datetime.now(timezone.utc).isoformat(), "adjustments": update.weight_adjustments}
        )
        if len(raw.feedback_history) > MAX_FEEDBACK_HISTORY:
            raw.feedback_history = raw.feedback_history[-MAX_FEEDBACK_HISTORY:]
        raw.feedback_count += 1
        self.save_profile(raw)
        return WeightProfile(weights=raw.weights, fuzzy_overrides=raw.fuzzy_overrides)

    def refine_fuzzy_sets(self, profile_id: str) -> dict[str, list[dict]] | None:
        raw = self._load_raw_profile(profile_id)
        if raw is None or raw.feedback_count < 10:
            return None
        criterion_deltas: dict[str, list[float]] = {}
        for entry in raw.feedback_history:
            for criterion, delta in entry.get("adjustments", {}).items():
                criterion_deltas.setdefault(criterion, []).append(delta)
        refinements: dict[str, list[dict]] = {}
        for criterion, deltas in criterion_deltas.items():
            if len(deltas) < 10:
                continue
            mean_delta = sum(deltas) / len(deltas)
            if abs(mean_delta) > 0.01:
                shift = mean_delta * 10.0
                center = max(0.0, min(1.0, 0.5 + shift))
                refinements[criterion] = [
                    {"name": "learned_preference", "type": "triangular", "params": [max(0, center - 0.3), center, min(1, center + 0.3)]}
                ]
        if not refinements:
            return None
        raw.fuzzy_overrides.update(refinements)
        self.save_profile(raw)
        return refinements


def _contrib_dict(opt: OptionResult) -> dict[str, float]:
    """criterion_name -> contribution (use weighted_score normalized by sum or contribution_pct)."""
    out: dict[str, float] = {}
    total = sum(c.weighted_score for c in opt.criterion_contributions)
    for c in opt.criterion_contributions:
        out[c.criterion_name] = (c.weighted_score / total) if total > 0 else 0.0
    return out


def _accept_adjustments(best: OptionResult) -> dict[str, float]:
    contrib = _contrib_dict(best)
    return {name: _ACCEPT_FACTOR * val for name, val in contrib.items()}


def _reject_adjustments(best: OptionResult) -> dict[str, float]:
    contrib = _contrib_dict(best)
    return {name: -_REJECT_FACTOR * val for name, val in contrib.items()}


def _override_adjustments(rejected: OptionResult, chosen: OptionResult) -> dict[str, float]:
    rej = _contrib_dict(rejected)
    ch = _contrib_dict(chosen)
    all_criteria = set(rej) | set(ch)
    adjustments: dict[str, float] = {}
    for criterion in all_criteria:
        diff = ch.get(criterion, 0.0) - rej.get(criterion, 0.0)
        adjustments[criterion] = _OVERRIDE_FACTOR * diff
    return adjustments

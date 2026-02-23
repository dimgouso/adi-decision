"""ADI DecisionOutput schema — the strict output contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CriterionContribution(BaseModel):
    """How much a specific criterion contributed to an option's score."""

    criterion_name: str
    raw_value: float | None
    normalized_value: float | None
    weighted_score: float
    contribution_pct: float = Field(description="Percentage of total score from this criterion")
    evidence_quality: float = Field(ge=0.0, le=1.0)
    missing: bool = False


class ConstraintReport(BaseModel):
    """Constraint evaluation result for one option."""

    constraint_type: str
    satisfied: bool
    hard: bool
    detail: str = ""


class CounterfactualItem(BaseModel):
    """'If criterion X changed, option Y would rank differently.'"""

    criterion_name: str
    direction: str = Field(description="'increase' or 'decrease'")
    delta_pct: float = Field(description="Magnitude of hypothetical change (%)")
    new_rank: int
    rank_change: int = Field(description="Positive = improved rank")


class OptionResult(BaseModel):
    """Full evaluation result for one option."""

    option_name: str
    rank: int = Field(ge=1)
    score: float = Field(description="Final weighted score after policy application")
    raw_score: float = Field(description="Score before uncertainty/policy adjustments")
    confidence: float = Field(ge=0.0, le=1.0)
    eliminated: bool = Field(default=False, description="Eliminated by hard constraint")
    elimination_reason: str = ""
    criterion_contributions: list[CriterionContribution] = Field(default_factory=list)
    constraint_reports: list[ConstraintReport] = Field(default_factory=list)
    counterfactuals: list[CounterfactualItem] = Field(default_factory=list)


class SensitivityReport(BaseModel):
    """How stable is the ranking across weight perturbations?"""

    overall_stability: float = Field(
        ge=0.0, le=1.0,
        description="1.0 = completely stable across weight perturbations",
    )
    unstable_pairs: list[tuple[str, str]] = Field(
        default_factory=list,
        description="Option pairs whose relative ranking is sensitive to weight changes",
    )
    weight_sensitivity: dict[str, float] = Field(
        default_factory=dict,
        description="Per-criterion: how much ranking changes with ±10% weight shift",
    )


class DecisionOutput(BaseModel):
    """Full output contract for ADI decision engine."""

    best_option: str = Field(description="Name of the top-ranked non-eliminated option")
    ranking: list[OptionResult] = Field(description="All options sorted by rank (1=best)")
    overall_confidence: float = Field(ge=0.0, le=1.0)
    policy_applied: str = Field(description="Policy name that was applied")
    policy_parameters: dict = Field(
        default_factory=dict,
        description="Resolved policy parameters for full auditability",
    )
    sensitivity: SensitivityReport
    warnings: list[str] = Field(default_factory=list)
    context: str = ""
    data_quality: DataQualityReport | None = Field(
        default=None,
        description="Optional summary of evidence quality",
    )

    @property
    def top_n(self) -> list[OptionResult]:
        return [r for r in self.ranking if not r.eliminated]


class ScenarioResult(BaseModel):
    """Result of a single scenario in a what-if comparison."""

    scenario_label: str
    changes: dict[str, float | str] = Field(default_factory=dict)
    best_option: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    ranking: list[str] = Field(
        default_factory=list,
        description="Option names in ranked order",
    )


class ScenarioComparison(BaseModel):
    """Result of comparing multiple what-if scenarios."""

    scenarios: list[ScenarioResult] = Field(default_factory=list)
    ranking_stability: float = Field(
        ge=0.0,
        le=1.0,
        default=0.0,
        description="Fraction of scenarios where #1 matches the base case",
    )
    robust_options: list[str] = Field(
        default_factory=list,
        description="Options that appear in top-2 in every scenario",
    )
    critical_thresholds: dict[str, float] = Field(
        default_factory=dict,
        description="Path -> value at which the best option flips",
    )


class DataQualityReport(BaseModel):
    """Evidence quality summary derived from per-cell confidence."""

    mean_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    low_confidence_features: list[dict[str, str | float]] = Field(default_factory=list)
    missing_features: list[dict[str, str]] = Field(default_factory=list)
    confidence_impact: float = Field(ge=0.0, le=1.0, default=0.0)


class FeedbackInput(BaseModel):
    """User/agent feedback on a decision — feeds the learning engine."""

    action: Literal["accept", "reject", "override"] = Field(
        description="accept: decision was correct; reject: wrong; override: user chose a different option",
    )
    chosen_option: str | None = Field(
        default=None,
        description="If action=override, the option the user actually chose",
    )
    reason: str | None = None


class ProfileUpdate(BaseModel):
    """Result of processing feedback — weight deltas applied to the profile."""

    profile_id: str = ""
    weight_adjustments: dict[str, float] = Field(default_factory=dict)
    fuzzy_adjustments: dict[str, list[float]] | None = None
    feedback_count: int = 0

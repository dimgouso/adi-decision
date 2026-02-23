"""ADI DecisionRequest schema — the strict input contract."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class CriterionDirection(str, Enum):
    """Benefit criteria: higher raw value is better. Cost: lower is better."""

    BENEFIT = "benefit"
    COST = "cost"


class FuzzySet(BaseModel):
    """Triangular or trapezoidal fuzzy set defining linguistic terms."""

    low: tuple[float, float, float] = Field(description="(a, b, c) triangular")
    medium: tuple[float, float, float] = Field(description="(a, b, c) triangular")
    high: tuple[float, float, float] = Field(description="(a, b, c) triangular")


class FuzzySetDefinition(BaseModel):
    """Single fuzzy membership function (for confidence-aware fuzzification)."""

    name: str = Field(description="Label: 'low', 'medium', 'high', etc.")
    type: str = Field(
        default="triangular",
        description="'triangular', 'trapezoidal', or 'gaussian'",
    )
    params: list[float] = Field(
        description="triangular=[a,b,c], trapezoidal=[a,b,c,d], gaussian=[mean,sigma]",
    )


class Criterion(BaseModel):
    """A single evaluation criterion with weight, direction, and optional fuzzy config."""

    name: str = Field(min_length=1, max_length=64)
    weight: float = Field(gt=0.0, le=1.0, description="Relative importance weight (0–1]")
    direction: CriterionDirection = CriterionDirection.BENEFIT
    fuzzy_set: FuzzySet | None = None
    fuzzy_set_definitions: list[FuzzySetDefinition] | None = Field(
        default=None,
        description="Optional list of MFs for confidence-aware fuzzification (overrides fuzzy_set when set)",
    )
    description: str = ""


class EvidenceItem(BaseModel):
    """A piece of evidence attached to an option's criterion value."""

    source: str = Field(description="Citation, URL, DOI, or internal ref")
    quality: float = Field(
        ge=0.0, le=1.0, default=0.5, description="Evidence quality score (0–1)"
    )
    note: str = ""


class OptionValue(BaseModel):
    """The raw value for one criterion of one option."""

    criterion_name: str
    value: float | None = Field(
        default=None, description="Raw numeric value; None means missing"
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Explicit confidence for this value (0–1). If None, derived from evidence.",
    )
    evidence: list[EvidenceItem] = Field(default_factory=list)


class Option(BaseModel):
    """A candidate option to be evaluated."""

    name: str = Field(min_length=1, max_length=128)
    values: list[OptionValue] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def get_value(self, criterion_name: str) -> float | None:
        for v in self.values:
            if v.criterion_name == criterion_name:
                return v.value
        return None

    def get_evidence(self, criterion_name: str) -> list[EvidenceItem]:
        for v in self.values:
            if v.criterion_name == criterion_name:
                return v.evidence
        return []

    def get_confidence(self, criterion_name: str) -> float | None:
        """Return explicit confidence for criterion if set, else None (caller may derive from evidence)."""
        for v in self.values:
            if v.criterion_name == criterion_name:
                return v.confidence
        return None


class ConstraintType(str, Enum):
    MUST_INCLUDE = "must_include"  # option must be in result
    MUST_EXCLUDE = "must_exclude"  # option must not be ranked
    MIN_VALUE = "min_value"        # option.criterion must be >= threshold
    MAX_VALUE = "max_value"        # option.criterion must be <= threshold


class Constraint(BaseModel):
    """A hard or soft constraint on the decision."""

    constraint_type: ConstraintType
    option_name: str | None = None
    criterion_name: str | None = None
    threshold: float | None = None
    hard: bool = Field(
        default=True, description="Hard=eliminates violators; soft=penalizes"
    )


class WeightProfile(BaseModel):
    """Learned or explicit weights per criterion (for learning/feedback)."""

    weights: dict[str, float] = Field(default_factory=dict)
    fuzzy_overrides: dict[str, list[dict]] | None = None


class DecisionRequest(BaseModel):
    """Full input contract for ADI decision engine. No free text passes core."""

    options: list[Option] = Field(min_length=2, description="At least 2 options required")
    criteria: list[Criterion] = Field(min_length=1)
    constraints: list[Constraint] = Field(default_factory=list)
    policy_name: str = Field(
        default="balanced",
        description="Name of the policy strategy to apply",
    )
    policy_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Override specific policy parameters inline",
    )
    profile_id: str | None = Field(
        default=None,
        description="Optional profile ID for loading learned weights",
    )
    preferences: WeightProfile | None = Field(
        default=None,
        description="Explicit weight overrides (e.g. from loaded profile)",
    )
    context: str = Field(
        default="",
        max_length=512,
        description="Optional human-readable context (for logging/explanation only)",
    )

    @field_validator("criteria")
    @classmethod
    def weights_must_be_positive(cls, criteria: list[Criterion]) -> list[Criterion]:
        for c in criteria:
            if c.weight <= 0:
                raise ValueError(f"Criterion '{c.name}' weight must be > 0")
        return criteria

    @model_validator(mode="after")
    def all_option_values_reference_known_criteria(self) -> "DecisionRequest":
        known = {c.name for c in self.criteria}
        for opt in self.options:
            for v in opt.values:
                if v.criterion_name not in known:
                    raise ValueError(
                        f"Option '{opt.name}' references unknown criterion '{v.criterion_name}'"
                    )
        return self

    @model_validator(mode="after")
    def constraint_references_valid(self) -> "DecisionRequest":
        known_options = {o.name for o in self.options}
        known_criteria = {c.name for c in self.criteria}
        for constraint in self.constraints:
            if constraint.option_name and constraint.option_name not in known_options:
                raise ValueError(
                    f"Constraint references unknown option '{constraint.option_name}'"
                )
            if constraint.criterion_name and constraint.criterion_name not in known_criteria:
                raise ValueError(
                    f"Constraint references unknown criterion '{constraint.criterion_name}'"
                )
        return self

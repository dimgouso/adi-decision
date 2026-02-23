"""ADI Policy module — first-class decision strategies.

A Policy defines HOW to make a decision, not what the domain criteria are.
Policies are loadable from YAML packs or constructed inline.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class PolicyStrategy(str, Enum):
    """High-level decision strategy."""

    BALANCED = "balanced"
    RISK_AVERSE = "risk_averse"
    EXPLORATORY = "exploratory"


class ConstraintPriorityMode(str, Enum):
    """How hard constraints interact with scoring."""

    ELIMINATE = "eliminate"    # violators are removed from ranking entirely
    PENALIZE = "penalize"      # violators receive a score penalty but remain
    WARN = "warn"              # violations are noted but do not affect score


class Policy(BaseModel):
    """
    Explicit, auditable policy for a decision.

    Every field is documented so the output explanation can reference
    exactly which parameters drove each scoring adjustment.
    """

    name: str = Field(default="balanced", description="Human-readable policy name")
    strategy: PolicyStrategy = PolicyStrategy.BALANCED

    uncertainty_penalty_factor: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description=(
            "How much low-confidence evidence penalizes score. "
            "0 = no penalty; 1 = full confidence penalty."
        ),
    )
    missingness_penalty_factor: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Penalty per missing criterion value (proportional to criterion weight).",
    )
    variance_penalty_factor: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "RISK_AVERSE: penalizes options with high score variance across criteria. "
            "0 = no variance penalty."
        ),
    )
    exploration_bonus_factor: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "EXPLORATORY: boosts options that score very differently from frontrunner. "
            "0 = no exploration bonus."
        ),
    )
    constraint_priority_mode: ConstraintPriorityMode = ConstraintPriorityMode.ELIMINATE
    soft_constraint_penalty: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Score multiplier penalty when a soft constraint is violated.",
    )
    topsis_enabled: bool = Field(
        default=False,
        description="Use TOPSIS distance-based scoring instead of weighted sum.",
    )
    use_cell_confidence: bool = Field(
        default=False,
        description=(
            "When True, confidence is applied per cell in MCDA (v_ij * c_ij). "
            "When False, confidence is applied as a post-score adjustment."
        ),
    )
    sensitivity_perturbation_pct: float = Field(
        default=10.0,
        gt=0.0,
        le=50.0,
        description="Weight perturbation percentage for sensitivity analysis.",
    )

    def to_audit_dict(self) -> dict[str, Any]:
        """Return all parameters for inclusion in DecisionOutput.policy_parameters."""
        return self.model_dump()

    @classmethod
    def balanced(cls) -> "Policy":
        return cls(
            name="balanced",
            strategy=PolicyStrategy.BALANCED,
            uncertainty_penalty_factor=0.2,
            missingness_penalty_factor=0.3,
            variance_penalty_factor=0.0,
            exploration_bonus_factor=0.0,
        )

    @classmethod
    def risk_averse(cls) -> "Policy":
        return cls(
            name="risk_averse",
            strategy=PolicyStrategy.RISK_AVERSE,
            uncertainty_penalty_factor=0.4,
            missingness_penalty_factor=0.5,
            variance_penalty_factor=0.3,
            exploration_bonus_factor=0.0,
            constraint_priority_mode=ConstraintPriorityMode.ELIMINATE,
        )

    @classmethod
    def exploratory(cls) -> "Policy":
        return cls(
            name="exploratory",
            strategy=PolicyStrategy.EXPLORATORY,
            uncertainty_penalty_factor=0.1,
            missingness_penalty_factor=0.1,
            variance_penalty_factor=0.0,
            exploration_bonus_factor=0.2,
            constraint_priority_mode=ConstraintPriorityMode.PENALIZE,
        )


_BUILT_IN_POLICIES: dict[str, Policy] = {
    "balanced": Policy.balanced(),
    "risk_averse": Policy.risk_averse(),
    "exploratory": Policy.exploratory(),
}


class PolicyRegistry:
    """
    Load and resolve policies by name.

    Resolution order:
    1. Built-in policies (balanced / risk_averse / exploratory)
    2. YAML pack files in registered pack directories
    3. Inline overrides from DecisionRequest.policy_overrides
    """

    def __init__(self) -> None:
        self._policies: dict[str, Policy] = dict(_BUILT_IN_POLICIES)
        self._pack_dirs: list[Path] = []

    def register_pack_dir(self, path: Path) -> None:
        self._pack_dirs.append(path)

    def load_from_yaml(self, path: Path) -> Policy:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        policy_data = data.get("policy", data)
        policy = Policy.model_validate(policy_data)
        self._policies[policy.name] = policy
        return policy

    def get(self, name: str, overrides: dict[str, Any] | None = None) -> Policy:
        """Resolve policy by name, then apply any inline overrides."""
        if name not in self._policies:
            self._try_load_from_packs(name)

        policy = self._policies.get(name)
        if policy is None:
            raise ValueError(
                f"Unknown policy '{name}'. "
                f"Available: {list(self._policies.keys())}"
            )

        if overrides:
            data = policy.model_dump()
            data.update(overrides)
            policy = Policy.model_validate(data)

        return policy

    def _try_load_from_packs(self, name: str) -> None:
        for pack_dir in self._pack_dirs:
            for yaml_path in pack_dir.rglob("*.yaml"):
                try:
                    p = self.load_from_yaml(yaml_path)
                    if p.name == name:
                        return
                except Exception:
                    continue

    def list_available(self) -> list[str]:
        return list(self._policies.keys())


_default_registry = PolicyRegistry()


def get_default_registry() -> PolicyRegistry:
    return _default_registry


def resolve_policy(
    name: str = "balanced",
    overrides: dict[str, Any] | None = None,
    registry: PolicyRegistry | None = None,
) -> Policy:
    """Convenience function to resolve a policy from the default registry."""
    reg = registry or _default_registry
    return reg.get(name, overrides)

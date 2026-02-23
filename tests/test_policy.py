"""Tests for ADI policy module."""

import pytest

from adi.core.policy import (
    ConstraintPriorityMode,
    Policy,
    PolicyRegistry,
    PolicyStrategy,
    resolve_policy,
)


def test_builtin_balanced():
    p = resolve_policy("balanced")
    assert p.strategy == PolicyStrategy.BALANCED
    assert p.uncertainty_penalty_factor == 0.2


def test_builtin_risk_averse():
    p = resolve_policy("risk_averse")
    assert p.strategy == PolicyStrategy.RISK_AVERSE
    assert p.variance_penalty_factor == 0.3
    assert p.uncertainty_penalty_factor == 0.4


def test_builtin_exploratory():
    p = resolve_policy("exploratory")
    assert p.strategy == PolicyStrategy.EXPLORATORY
    assert p.exploration_bonus_factor == 0.2


def test_unknown_policy_raises():
    with pytest.raises(ValueError, match="Unknown policy"):
        resolve_policy("nonexistent_policy")


def test_inline_overrides():
    p = resolve_policy("balanced", overrides={"uncertainty_penalty_factor": 0.99})
    assert p.uncertainty_penalty_factor == 0.99
    assert p.strategy == PolicyStrategy.BALANCED


def test_policy_audit_dict():
    p = Policy.balanced()
    d = p.to_audit_dict()
    assert "strategy" in d
    assert "uncertainty_penalty_factor" in d
    assert d["name"] == "balanced"


def test_registry_list_available():
    reg = PolicyRegistry()
    available = reg.list_available()
    assert "balanced" in available
    assert "risk_averse" in available
    assert "exploratory" in available


def test_registry_custom_policy():
    reg = PolicyRegistry()
    custom = Policy(
        name="custom_test",
        strategy=PolicyStrategy.BALANCED,
        uncertainty_penalty_factor=0.7,
    )
    reg._policies["custom_test"] = custom
    resolved = reg.get("custom_test")
    assert resolved.uncertainty_penalty_factor == 0.7


def test_policy_yaml_loading(tmp_path):
    import yaml
    policy_data = {
        "policy": {
            "name": "yaml_policy",
            "strategy": "risk_averse",
            "uncertainty_penalty_factor": 0.5,
            "missingness_penalty_factor": 0.4,
        }
    }
    yaml_file = tmp_path / "policy.yaml"
    yaml_file.write_text(yaml.dump(policy_data))

    reg = PolicyRegistry()
    p = reg.load_from_yaml(yaml_file)
    assert p.name == "yaml_policy"
    assert p.strategy == PolicyStrategy.RISK_AVERSE

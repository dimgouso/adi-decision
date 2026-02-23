"""ADI Agent Adapter — OpenAI and Claude function tool schemas.

Allows an LLM agent to call ADI as a structured tool, treating
the decision engine as a black box with a well-defined contract.
"""

from __future__ import annotations

from typing import Any

OPENAI_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "adi_decide",
        "description": (
            "Run a structured multi-criteria decision analysis. "
            "Given a list of options with numeric criterion values, weights, constraints, "
            "and a policy strategy, returns a ranked list with scores, confidence, "
            "and full explanation. Use this instead of reasoning about tradeoffs yourself."
        ),
        "parameters": {
            "type": "object",
            "required": ["options", "criteria"],
            "properties": {
                "options": {
                    "type": "array",
                    "description": "Candidate options to evaluate",
                    "items": {
                        "type": "object",
                        "required": ["name", "values"],
                        "properties": {
                            "name": {"type": "string"},
                            "values": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "required": ["criterion_name", "value"],
                                    "properties": {
                                        "criterion_name": {"type": "string"},
                                        "value": {"type": "number"},
                                        "evidence": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "source": {"type": "string"},
                                                    "quality": {"type": "number", "minimum": 0, "maximum": 1},
                                                },
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
                "criteria": {
                    "type": "array",
                    "description": "Evaluation criteria with weights",
                    "items": {
                        "type": "object",
                        "required": ["name", "weight"],
                        "properties": {
                            "name": {"type": "string"},
                            "weight": {"type": "number", "minimum": 0, "maximum": 1},
                            "direction": {
                                "type": "string",
                                "enum": ["benefit", "cost"],
                                "default": "benefit",
                            },
                            "description": {"type": "string"},
                        },
                    },
                },
                "policy_name": {
                    "type": "string",
                    "description": "Policy strategy: 'balanced', 'risk_averse', or 'exploratory'",
                    "enum": ["balanced", "risk_averse", "exploratory"],
                    "default": "balanced",
                },
                "constraints": {
                    "type": "array",
                    "description": "Hard or soft constraints on the decision",
                    "items": {
                        "type": "object",
                        "required": ["constraint_type"],
                        "properties": {
                            "constraint_type": {
                                "type": "string",
                                "enum": ["must_include", "must_exclude", "min_value", "max_value"],
                            },
                            "option_name": {"type": "string"},
                            "criterion_name": {"type": "string"},
                            "threshold": {"type": "number"},
                            "hard": {"type": "boolean", "default": True},
                        },
                    },
                },
                "context": {
                    "type": "string",
                    "description": "Optional context string for logging",
                },
            },
        },
    },
}

ANTHROPIC_TOOL_SCHEMA: dict[str, Any] = {
    "name": "adi_decide",
    "description": OPENAI_TOOL_SCHEMA["function"]["description"],
    "input_schema": OPENAI_TOOL_SCHEMA["function"]["parameters"],
}


def call_adi_tool(tool_input: dict[str, Any]) -> dict[str, Any]:
    """
    Execute an ADI decision from an agent tool call.

    Args:
        tool_input: dict matching the tool schema parameters

    Returns:
        DecisionOutput serialized as dict
    """
    from adi.core.decision_engine import decide
    from adi.schemas.decision_request import DecisionRequest

    request = DecisionRequest.model_validate(tool_input)
    output = decide(request)
    return output.model_dump()


def get_openai_tools() -> list[dict[str, Any]]:
    """Return list of tool dicts for OpenAI API (tools parameter)."""
    return [OPENAI_TOOL_SCHEMA]


def get_anthropic_tools() -> list[dict[str, Any]]:
    """Return list of tool dicts for Anthropic API."""
    return [ANTHROPIC_TOOL_SCHEMA]

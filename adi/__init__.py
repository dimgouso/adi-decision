"""ADI — Agent Decision Intelligence.

Library for structured multi-criteria decisions with policy, explainability,
and confidence. Use in-process (decide, what_if, compare, feedback) or via
agent tools (get_openai_tools, get_anthropic_tools, call_adi_tool).
"""

__version__ = "0.1.0"

from adi.core.decision_engine import compare, decide, feedback, what_if
from adi.interfaces.agent_adapter import call_adi_tool, get_anthropic_tools, get_openai_tools
from adi.schemas.decision_output import (
    DecisionOutput,
    FeedbackInput,
    ProfileUpdate,
    ScenarioComparison,
    ScenarioResult,
)
from adi.schemas.decision_request import DecisionRequest, WeightProfile

__all__ = [
    "__version__",
    "decide",
    "what_if",
    "compare",
    "feedback",
    "DecisionRequest",
    "DecisionOutput",
    "WeightProfile",
    "FeedbackInput",
    "ProfileUpdate",
    "ScenarioResult",
    "ScenarioComparison",
    "get_openai_tools",
    "get_anthropic_tools",
    "call_adi_tool",
]

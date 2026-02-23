# ADI — Agent Decision Intelligence

**Rank options. Get the why. No decision logic from scratch.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Pydantic](https://img.shields.io/badge/Pydantic-2.x-E92063?style=flat&logo=pydantic&logoColor=white)](https://pydantic.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26+-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)](LICENSE)

Structured multi-criteria decision making with policy, explainability, and confidence. Use as a **library** in your code or as a **tool** for LLM agents (OpenAI, Anthropic).

---

**What it does**  
You provide options (e.g. three suppliers) and criteria with weights (e.g. cost, quality, delivery time). ADI scores them, ranks them, and tells you which is best and **why** (how much each criterion contributed). You can run what-if analyses (e.g. what if I change the weight on cost?) and compare multiple scenarios. If you say “I would have chosen B”, it learns from that feedback and adjusts the weights for next time. You can use it in your own code, behind an API, or as a tool for LLM agents (OpenAI, Claude) without implementing the decision logic yourself.

| You provide | ADI returns |
|-------------|-------------|
| Options (e.g. three suppliers, five projects) | **Ranking** with scores and confidence |
| Criteria + weights (cost, quality, risk, …) | **Explanation**: which criterion contributed how much |
| Optional constraints (min/max, hard/soft) | **Sensitivity**: how stable the ranking is to weight changes |
| — | **What-if** and **scenario comparison** |
| Feedback ("I would have chosen B") | **Learning**: weights adapt for next time |

---

## 📦 Install

**pip (from repo root)**

```bash
pip install -e .
# or with dev deps: pip install -e ".[dev]"
```

**macOS (Homebrew)**

Install Python via Homebrew, then install ADI from this repo:

```bash
brew install python
cd /path/to/adi-decision
pip3 install -e .
```

Or, once the package is published to PyPI:

```bash
brew install python
pip3 install adi-decision
```

## 🚀 Quick start

```python
from adi import decide, DecisionRequest
from adi.schemas.decision_request import Option, OptionValue, Criterion

request = DecisionRequest(
    options=[
        Option(name="A", values=[
            OptionValue(criterion_name="cost", value=100),
            OptionValue(criterion_name="quality", value=0.8),
        ]),
        Option(name="B", values=[
            OptionValue(criterion_name="cost", value=80),
            OptionValue(criterion_name="quality", value=0.6),
        ]),
    ],
    criteria=[
        Criterion(name="cost", weight=0.4, direction="cost"),
        Criterion(name="quality", weight=0.6, direction="benefit"),
    ],
)
output = decide(request)
print(output.best_option, output.ranking, output.sensitivity, output.data_quality)
```

## 🤖 For agent developers

Register ADI as a tool so your LLM can call it:

```python
from adi import get_openai_tools, call_adi_tool

tools = get_openai_tools()  # or get_anthropic_tools() for Claude
# Pass tools to your OpenAI/Anthropic API.
# When the model calls "adi_decide", run:
result = call_adi_tool(tool_arguments)  # returns dict: best_option, ranking, etc.
```

## 🔀 What-if & compare

```python
from adi import what_if, compare

# Single scenario: what if option A had cost 50?
out = what_if(request, {"options.A.value.cost": 50})

# Compare multiple scenarios
cmp = compare(request, [
    {"criteria.cost": 0.6},
    {"criteria.quality": 0.8},
])
# cmp.ranking_stability, cmp.robust_options, cmp.critical_thresholds
```

## 📈 Learning from feedback

```python
from adi import decide, feedback
from adi.schemas.decision_output import FeedbackInput

# Request with profile_id to load/save learned weights
request.profile_id = "user_123"
output = decide(request)
# User says they prefer option B
feedback(FeedbackInput(action="override", chosen_option="B"), output, request)
# Next decide(request) with same profile_id uses updated weights
```

## ⌨️ CLI & API

```bash
adi decide --input request.json
uvicorn adi.interfaces.api:app --reload
# POST /decide with DecisionRequest JSON, GET /policies
```

## ✨ Features

- **Policies**: balanced, risk_averse, exploratory (uncertainty penalty, variance penalty, exploration bonus).
- **Confidence**: per-value or from evidence; optional `use_cell_confidence` in policy to bake confidence into the score.
- **Explainability**: criterion contributions, counterfactuals, constraint reports, sensitivity (weight perturbations).
- **Scenarios**: what-if and multi-scenario comparison with ranking stability and robust options.
- **Learning**: Bayesian-style weight updates from accept/reject/override feedback; profiles stored as JSON.

## ⚙️ Configurable parameters

Everything is driven by the **DecisionRequest** and optional **policy overrides**. No free text passes into the core engine.

### Decision request

| Parameter | Description |
|-----------|-------------|
| **options** | List of candidates. Each has `name`, `values` (list of `criterion_name` + `value`), optional `confidence` per value, optional `evidence` (source, quality), and `metadata`. |
| **criteria** | List of criteria: `name`, `weight` (0–1], `direction` (`benefit` or `cost`), optional `fuzzy_set` / `fuzzy_set_definitions`, `description`. |
| **constraints** | Optional list: `constraint_type` (`must_include`, `must_exclude`, `min_value`, `max_value`), `option_name`, `criterion_name`, `threshold`, `hard` (true = eliminate, false = penalize). |
| **policy_name** | Strategy: `"balanced"`, `"risk_averse"`, or `"exploratory"`. |
| **policy_overrides** | Dict to override specific policy parameters (see Policy, below). |
| **profile_id** | Optional ID for loading/saving learned weights (used with feedback). |
| **preferences** | Optional `WeightProfile`: explicit `weights` (and optional `fuzzy_overrides`) instead of defaults. |
| **context** | Optional short text for logging/explanation (max 512 chars). |

### Policy (via `policy_overrides` or YAML packs)

| Parameter | Description |
|-----------|-------------|
| **uncertainty_penalty_factor** | How much low-confidence evidence penalizes the score (0–1). |
| **missingness_penalty_factor** | Penalty per missing criterion value, scaled by weight (0–1). |
| **variance_penalty_factor** | (risk_averse) Penalty for options with high score variance across criteria (0–1). |
| **exploration_bonus_factor** | (exploratory) Bonus for options that score very differently from the frontrunner (0–1). |
| **constraint_priority_mode** | `eliminate`, `penalize`, or `warn`. |
| **soft_constraint_penalty** | Score multiplier when a soft constraint is violated (0–1). |
| **topsis_enabled** | If true, use TOPSIS distance-based scoring instead of weighted sum. |
| **use_cell_confidence** | If true, confidence is applied per cell (v_ij × c_ij); otherwise as a post-score adjustment. |
| **sensitivity_perturbation_pct** | Weight perturbation percentage for sensitivity analysis (e.g. 10). |

### Feedback (learning)

| Parameter | Description |
|-----------|-------------|
| **action** | `"accept"`, `"reject"`, or `"override"`. |
| **chosen_option** | Required when `action="override"`: the option the user actually chose. |
| **reason** | Optional free text. |

## 🧪 Tests

From the project root, install the package then run tests:

```bash
pip install -e .
pytest tests/ -v
```

Or without installing (from repo root): `PYTHONPATH=. pytest tests/ -v`

## License

MIT

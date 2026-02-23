# ADI — Agent Decision Intelligence (Skill)

## When to use

Use ADI when you or the user need to:

- Make a **structured multi-criteria decision** (e.g. choose among options with several criteria and weights).
- Get a **ranked result with explanation** (why one option won, counterfactuals, sensitivity).
- Run **what-if** or **scenario comparison** (how the decision changes if inputs change).
- Learn from **feedback** (accept/reject/override) to adapt weights over time.

Trigger when the user says: "help me decide", "compare options", "which is best given...", "what if we change...", "run a decision", "multi-criteria decision", "rank these options".

## How to use (library)

```python
from adi import decide, DecisionRequest, DecisionOutput
# Build request with options (name + values per criterion), criteria (name, weight, direction), optional constraints.
request = DecisionRequest(options=[...], criteria=[...])
output = decide(request)
# output.best_option, output.ranking, output.overall_confidence, output.sensitivity, output.data_quality
```

## How to use (agent tools)

Register the ADI tool with your LLM agent:

- **OpenAI**: `tools = get_openai_tools()` then pass to the API; when the model calls `adi_decide`, run `result = call_adi_tool(arguments)`.
- **Anthropic**: `tools = get_anthropic_tools()`; same handler `call_adi_tool(arguments)`.

Tool name: `adi_decide`. Input: `options` (array of {name, values: [{criterion_name, value}]}), `criteria` (array of {name, weight, direction}), optional `policy_name` ("balanced" | "risk_averse" | "exploratory"), `constraints`, `context`. Output: full decision with ranking, scores, explanation, sensitivity.

## What-if and compare

```python
from adi import what_if, compare
# Single scenario
out = what_if(request, {"options.A.value.cost": 100})
# Multiple scenarios
cmp = compare(request, [{"criteria.cost": 0.8}, {"criteria.quality": 0.8}])
# cmp.ranking_stability, cmp.robust_options, cmp.critical_thresholds
```

## Learning (feedback)

```python
from adi import decide, feedback, DecisionRequest, FeedbackInput
output = decide(request)
# User says "I actually prefer B"
up = feedback(FeedbackInput(action="override", chosen_option="B"), output, request)
# Next decide() with same profile_id will use updated weights (if profile_id was set).
```

## Key concepts

- **Policy**: balanced (default), risk_averse, exploratory. Controls uncertainty penalty and variance/exploration modifiers.
- **Confidence**: per value or from evidence.quality. Use `use_cell_confidence` in policy to bake confidence into the score formula.
- **Profiles**: set `request.profile_id` to load/save learned weights under that ID (e.g. per user).

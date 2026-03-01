# ADI — Agent Decision Intelligence

**Rank options, quantify confidence, and explain the decision.**

[![CI](https://github.com/dimgouso/adi-Agent-Decision-Intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/dimgouso/adi-Agent-Decision-Intelligence/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/dimgouso/adi-Agent-Decision-Intelligence?sort=semver)](https://github.com/dimgouso/adi-Agent-Decision-Intelligence/releases)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)

ADI is a Python library for structured multi-criteria decisions. Give it options, weighted criteria, constraints, and evidence confidence; it returns a ranked result with explainability, sensitivity analysis, and feedback-driven learning.

It is designed for two use cases:

- application code that needs auditable decision logic
- LLM agents that should call a deterministic ranking tool instead of improvising tradeoffs

Status: `alpha`. The core API is usable today, but expect refinement as real-world usage expands.

## Why ADI

- **Deterministic ranking**: scores and ranks options from explicit structured inputs.
- **Confidence-aware output**: low-confidence evidence can reduce score impact or lower final confidence.
- **Explainable decisions**: returns criterion contributions, constraint reports, and counterfactual guidance.
- **Scenario analysis**: supports what-if runs and cross-scenario comparison.
- **Learning loop**: feedback can update saved preference profiles for future runs.
- **Agent-ready**: ships OpenAI and Anthropic tool schemas out of the box.

## Install

From source:

```bash
git clone https://github.com/dimgouso/adi-Agent-Decision-Intelligence.git
cd adi-decision
pip install -e .
```

With development tooling:

```bash
pip install -e ".[dev]"
```

After PyPI publishing is enabled:

```bash
pip install adi-decision
```

## Quick Start

```python
from adi import DecisionRequest, decide
from adi.schemas.decision_request import Criterion, Option, OptionValue

request = DecisionRequest(
    options=[
        Option(
            name="Supplier A",
            values=[
                OptionValue(criterion_name="cost", value=100),
                OptionValue(criterion_name="quality", value=0.80, confidence=0.95),
            ],
        ),
        Option(
            name="Supplier B",
            values=[
                OptionValue(criterion_name="cost", value=85),
                OptionValue(criterion_name="quality", value=0.68, confidence=0.70),
            ],
        ),
    ],
    criteria=[
        Criterion(name="cost", weight=0.45, direction="cost"),
        Criterion(name="quality", weight=0.55, direction="benefit"),
    ],
    policy_name="balanced",
)

output = decide(request)

print(output.best_option)
for item in output.ranking:
    print(item.option_name, item.score, item.confidence)
```

## LLM Agent Integration

```python
from adi import call_adi_tool, get_openai_tools

tools = get_openai_tools()

# Pass `tools` to your OpenAI client.
# When the model calls `adi_decide`, route the arguments here:
result = call_adi_tool(tool_arguments)
```

## CLI And API

CLI:

```bash
adi validate request.json
adi decide --input request.json
adi decide --input request.json --policy risk_averse
```

API:

```bash
uvicorn adi.interfaces.api:app --reload
```

Key endpoints:

- `GET /health`
- `POST /decide`
- `GET /policies`
- `GET /policies/{policy_name}`

## Core Features

- **Policies**: `balanced`, `risk_averse`, and `exploratory`
- **Constraints**: hard elimination or soft penalties
- **Sensitivity**: robustness checks under weight perturbation
- **Confidence handling**: explicit per-value confidence or evidence-derived confidence
- **Feedback**: accept, reject, or override actions update user profiles
- **Typed schemas**: Pydantic models for request and response contracts

## Development

Run the full local verification suite:

```bash
pip install -e ".[dev]"
ruff check adi/
pytest
python -m build
twine check dist/*
```

## Release And Publishing

- Changelog: [CHANGELOG.md](CHANGELOG.md)
- Publishing guide: [docs/PUBLISHING.md](docs/PUBLISHING.md)
- GitHub Releases: [github.com/dimgouso/adi-Agent-Decision-Intelligence/releases](https://github.com/dimgouso/adi-Agent-Decision-Intelligence/releases)

## License

MIT

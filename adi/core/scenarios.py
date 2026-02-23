"""ADI Scenario & What-If Engine — explore how decisions change under different conditions.

Supports what-if analysis and multi-scenario comparison with dot-notation path changes.
Paths: options.<option_name>.value.<criterion_name>, criteria.<criterion_name>.weight,
       policy_overrides.<key>
"""

from __future__ import annotations

from typing import Any, Callable

from adi.schemas.decision_output import DecisionOutput, ScenarioComparison, ScenarioResult
from adi.schemas.decision_request import DecisionRequest


def _set_option_value(
    request: DecisionRequest,
    option_name: str,
    criterion_name: str,
    value: float | str,
) -> None:
    """Set the value for option's criterion. Creates OptionValue if missing."""
    from adi.schemas.decision_request import OptionValue

    option = next((o for o in request.options if o.name == option_name), None)
    if option is None:
        raise ValueError(f"Option '{option_name}' not found")
    known = {c.name for c in request.criteria}
    if criterion_name not in known:
        raise ValueError(f"Criterion '{criterion_name}' not found")
    for v in option.values:
        if v.criterion_name == criterion_name:
            v.value = value if isinstance(value, (int, float)) else None
            return
    option.values.append(
        OptionValue(criterion_name=criterion_name, value=value if isinstance(value, (int, float)) else None)
    )


def _set_criterion_weight(request: DecisionRequest, criterion_name: str, weight: float) -> None:
    criterion = next((c for c in request.criteria if c.name == criterion_name), None)
    if criterion is None:
        raise ValueError(f"Criterion '{criterion_name}' not found")
    criterion.weight = weight


def apply_changes(request: DecisionRequest, changes: dict[str, Any]) -> DecisionRequest:
    """
    Return a deep copy of request with the given changes applied.
    Paths: options.<option_name>.value.<criterion_name>, criteria.<criterion_name>.weight,
           policy_overrides.<key>
    """
    modified = request.model_copy(deep=True)

    for path, value in changes.items():
        parts = path.split(".")
        if not parts:
            continue
        root = parts[0]

        if root == "options" and len(parts) >= 4 and parts[2] == "value":
            # options.<option_name>.value.<criterion_name>
            option_name = parts[1]
            criterion_name = parts[3]
            _set_option_value(modified, option_name, criterion_name, value)
        elif root == "criteria" and len(parts) >= 2:
            # criteria.<criterion_name> or criteria.<criterion_name>.weight
            criterion_name = parts[1]
            if isinstance(value, (int, float)):
                _set_criterion_weight(modified, criterion_name, float(value))
        elif root == "policy_overrides" and len(parts) >= 2:
            key = parts[1]
            modified.policy_overrides[key] = value
        else:
            raise ValueError(
                f"Invalid change path '{path}': expected options.<name>.value.<criterion>, "
                "criteria.<name>, or policy_overrides.<key>"
            )

    return modified


def what_if(
    request: DecisionRequest,
    changes: dict[str, Any],
    decide_fn: Callable[[DecisionRequest], DecisionOutput],
) -> DecisionOutput:
    """Run decide_fn on a modified copy of request."""
    modified = apply_changes(request, changes)
    return decide_fn(modified)


def _auto_label(changes: dict[str, Any]) -> str:
    parts = []
    for path, val in changes.items():
        short = path.rsplit(".", maxsplit=1)[-1]
        parts.append(f"{short}={val}")
    return ", ".join(parts) if parts else "base"


def compare_scenarios(
    request: DecisionRequest,
    scenarios: list[dict[str, Any]],
    decide_fn: Callable[[DecisionRequest], DecisionOutput],
) -> ScenarioComparison:
    """Run base + each scenario and return aggregated comparison."""
    if not scenarios:
        raise ValueError("At least one scenario required for comparison")

    base_output = decide_fn(request)
    base_best = base_output.best_option
    base_ranking = [r.option_name for r in base_output.ranking]

    results: list[ScenarioResult] = [
        ScenarioResult(
            scenario_label="base",
            changes={},
            best_option=base_best,
            confidence=base_output.overall_confidence,
            ranking=base_ranking,
        )
    ]

    for ch in scenarios:
        out = what_if(request, ch, decide_fn)
        results.append(
            ScenarioResult(
                scenario_label=_auto_label(ch),
                changes=ch,
                best_option=out.best_option,
                confidence=out.overall_confidence,
                ranking=[r.option_name for r in out.ranking],
            )
        )

    same_as_base = sum(1 for r in results if r.best_option == base_best)
    ranking_stability = same_as_base / len(results)

    top2_sets = [set(r.ranking[:2]) for r in results if len(r.ranking) >= 2]
    robust_options = sorted(set.intersection(*top2_sets)) if top2_sets else []

    critical_thresholds: dict[str, float] = {}
    path_points: dict[str, list[tuple[float, str]]] = {}
    for idx, ch in enumerate(scenarios, start=1):
        for path, val in ch.items():
            if isinstance(val, (int, float)):
                path_points.setdefault(path, []).append((float(val), results[idx].best_option))

    for path, points in path_points.items():
        sorted_pts = sorted(points, key=lambda p: p[0])
        for i in range(len(sorted_pts) - 1):
            v1, opt1 = sorted_pts[i]
            v2, opt2 = sorted_pts[i + 1]
            if opt1 != opt2:
                critical_thresholds[path.rsplit(".", maxsplit=1)[-1]] = (v1 + v2) / 2.0
                break

    return ScenarioComparison(
        scenarios=results,
        ranking_stability=ranking_stability,
        robust_options=robust_options,
        critical_thresholds=critical_thresholds,
    )

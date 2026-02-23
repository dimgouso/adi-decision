"""ADI Typer CLI — command-line interface for the decision engine."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="adi",
    help="Agent Decision Intelligence — structured decision making with explainability.",
    add_completion=False,
)


@app.command()
def decide(
    input_file: Optional[Path] = typer.Option(
        None,
        "--input",
        "-i",
        help="Path to JSON file containing a DecisionRequest.",
    ),
    policy: Optional[str] = typer.Option(
        None,
        "--policy",
        "-p",
        help="Override policy name (balanced / risk_averse / exploratory).",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Write DecisionOutput JSON to this file (default: stdout).",
    ),
    pretty: bool = typer.Option(
        True,
        help="Pretty-print JSON output.",
    ),
) -> None:
    """Run the ADI decision pipeline on a DecisionRequest JSON file."""
    from adi.core.decision_engine import decide as _decide
    from adi.schemas.decision_request import DecisionRequest

    if input_file is None:
        if not sys.stdin.isatty():
            raw = sys.stdin.read()
        else:
            typer.echo("Error: provide --input or pipe JSON to stdin.", err=True)
            raise typer.Exit(1)
    else:
        raw = input_file.read_text()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        typer.echo(f"Error: invalid JSON — {e}", err=True)
        raise typer.Exit(1)

    if policy:
        data["policy_name"] = policy

    try:
        request = DecisionRequest.model_validate(data)
    except Exception as e:
        typer.echo(f"Error: invalid DecisionRequest — {e}", err=True)
        raise typer.Exit(1)

    result = _decide(request)

    indent = 2 if pretty else None
    output_json = result.model_dump_json(indent=indent)

    if output_file:
        output_file.write_text(output_json)
        typer.echo(f"Decision written to {output_file}")
    else:
        typer.echo(output_json)


@app.command()
def policies() -> None:
    """List all available policies."""
    from adi.core.policy import get_default_registry

    registry = get_default_registry()
    available = registry.list_available()
    typer.echo("Available policies:")
    for name in available:
        policy = registry.get(name)
        typer.echo(f"  {name:20s}  strategy={policy.strategy.value}")


@app.command()
def validate(
    input_file: Path = typer.Argument(..., help="Path to DecisionRequest JSON file."),
) -> None:
    """Validate a DecisionRequest JSON file without running the engine."""
    from adi.schemas.decision_request import DecisionRequest

    try:
        data = json.loads(input_file.read_text())
        DecisionRequest.model_validate(data)
        typer.echo(f"✓ Valid DecisionRequest in {input_file}")
    except Exception as e:
        typer.echo(f"✗ Validation failed: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

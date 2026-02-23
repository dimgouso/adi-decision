"""ADI FastAPI REST API."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException

from adi.schemas.decision_output import DecisionOutput
from adi.schemas.decision_request import DecisionRequest

app = FastAPI(
    title="ADI — Agent Decision Intelligence",
    description="Structured decision making with policy, explainability, and confidence.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "adi"}


@app.post("/decide", response_model=DecisionOutput)
def decide_endpoint(request: DecisionRequest) -> DecisionOutput:
    """Run the ADI decision pipeline on the provided DecisionRequest."""
    from adi.core.decision_engine import decide

    try:
        return decide(request)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decision engine error: {e}")


@app.get("/policies")
def list_policies() -> dict:
    """List all available policies with their parameters."""
    from adi.core.policy import get_default_registry

    registry = get_default_registry()
    result = {}
    for name in registry.list_available():
        policy = registry.get(name)
        result[name] = policy.to_audit_dict()
    return {"policies": result}


@app.get("/policies/{policy_name}")
def get_policy(policy_name: str) -> dict:
    """Get parameters for a specific policy."""
    from adi.core.policy import get_default_registry

    registry = get_default_registry()
    try:
        policy = registry.get(policy_name)
        return policy.to_audit_dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

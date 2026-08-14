"""FastAPI /arop/* gateway."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from neuroswarm_arm.evolution.factory import AROPRuntime


class OptimizeBody(BaseModel):
    force: bool = True


class CanaryBody(BaseModel):
    policy_id: str
    percent: float = Field(default=10.0, ge=0.0, le=100.0)


class GepaApproveBody(BaseModel):
    candidate_id: str
    reviewer: str = "operator"
    reason: str = "approved"


class GepaDeployBody(BaseModel):
    candidate_id: str
    require_approval: bool = True


def create_arop_router(runtime: AROPRuntime) -> APIRouter:
    router = APIRouter(prefix="/arop", tags=["arop"])

    @router.get("/health")
    def health() -> dict[str, Any]:
        return runtime.health()

    @router.get("/status")
    def status() -> dict[str, Any]:
        return runtime.status()

    @router.get("/policies")
    def policies() -> dict[str, Any]:
        return {
            "registry": runtime.registry.status(),
            "policies": [p.to_dict() for p in runtime.registry.list_policies()],
        }

    @router.post("/optimize")
    def optimize(body: OptimizeBody | None = None) -> dict[str, Any]:
        result = runtime.run_once()
        gepa_details = runtime.submit_gepa_best()
        details = dict(result.details or {})
        details.update(gepa_details)
        return {
            "status": result.status,
            "baseline_id": result.baseline_id,
            "candidate_id": result.candidate_id,
            "policy_id": result.policy_id,
            "message": result.message,
            "metrics": result.metrics,
            "details": details,
            "gepa_candidate_id": gepa_details.get("gepa_candidate_id"),
            "force": True if body is None else body.force,
        }

    @router.post("/rollback")
    def rollback() -> dict[str, Any]:
        rb = runtime.optimizer.deployment.rollback()
        return {
            "success": rb.success,
            "active_policy_id": rb.active_policy_id,
            "message": rb.message,
            "mode": rb.mode.value,
        }

    @router.post("/promote")
    def promote() -> dict[str, Any]:
        """Promote canary → 100% active (manual Option A gate; auto_promote stays off)."""
        dep = runtime.optimizer.deployment
        if hasattr(dep, "promote_canary"):
            result = dep.promote_canary()
        else:
            raise HTTPException(status_code=503, detail="promote_canary unavailable")
        return {
            "success": result.success,
            "active_policy_id": result.active_policy_id,
            "message": result.message,
            "mode": result.mode.value,
            "canary_percent": result.canary_percent,
            "details": dict(result.details or {}),
        }

    @router.get("/events")
    def events() -> dict[str, Any]:
        return {
            "events": [
                {"type": e.type.value, "at": e.at.isoformat(), "payload": dict(e.payload)}
                for e in runtime.bus.history(limit=50)
            ]
        }

    @router.get("/metrics")
    def metrics() -> dict[str, Any]:
        snap = runtime.aggregator.snapshot()
        return {
            "collected_at": snap.collected_at.isoformat(),
            "aggregate": dict(snap.aggregate),
            "providers": {k: dict(v) for k, v in snap.providers.items()},
        }

    @router.get("/gepa/pending")
    def gepa_pending() -> dict[str, Any]:
        gate = runtime.approval_gate
        if gate is None:
            return {"pending": []}
        return {
            "pending": [
                {
                    "id": c.id,
                    "version": c.version,
                    "content_hash": c.content_hash,
                    "components": list(c.components.keys()),
                    "scores": dict(getattr(c, "scores", {}) or {}),
                }
                for c in gate.pending()
            ]
        }

    @router.post("/gepa/approve")
    def gepa_approve(body: GepaApproveBody) -> dict[str, Any]:
        gate = runtime.approval_gate
        if gate is None:
            raise HTTPException(status_code=503, detail="GEPA approval gate unavailable")
        decision = gate.approve(body.candidate_id, reviewer=body.reviewer, reason=body.reason)
        return {
            "approved": decision.approved,
            "candidate_id": decision.candidate_id,
            "reviewer": decision.reviewer,
            "reason": decision.reason,
            "at": decision.at.isoformat(),
        }

    @router.post("/gepa/deploy")
    def gepa_deploy(body: GepaDeployBody) -> dict[str, Any]:
        gate = runtime.approval_gate
        deployer = runtime.text_deployer
        if deployer is None or gate is None:
            raise HTTPException(status_code=503, detail="GEPA deployer unavailable")
        candidate = None
        if runtime.gepa is not None:
            candidate = runtime.gepa.candidate_pool().get(body.candidate_id)
        if candidate is None:
            for c in gate.pending():
                if c.id == body.candidate_id:
                    candidate = c
                    break
        if candidate is None:
            raise HTTPException(status_code=404, detail=f"candidate not found: {body.candidate_id}")
        return deployer.deploy(candidate, require_approval=body.require_approval, gate=gate)

    return router

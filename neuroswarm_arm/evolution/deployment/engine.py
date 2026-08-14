"""Deployment engine — shadow / canary / promote / rollback via PolicyRegistry."""

from __future__ import annotations

from typing import Any

from neuroswarm_arm.evolution.deployment.adapters import DeploymentAdapter
from neuroswarm_arm.evolution.interfaces.deployment import (
    DeploymentController,
    DeploymentMode,
    DeploymentResult,
)
from neuroswarm_arm.evolution.models.experiment import CandidatePolicy
from neuroswarm_arm.evolution.models.policy import RuntimePolicy
from neuroswarm_arm.evolution.optimization.policy_registry import PolicyRegistry


class DeploymentEngine(DeploymentController):
    def __init__(
        self,
        registry: PolicyRegistry,
        *,
        adapters: list[DeploymentAdapter] | None = None,
    ) -> None:
        self.registry = registry
        self.adapters = list(adapters or [])
        self.rollback_count = 0

    def add_adapter(self, adapter: DeploymentAdapter) -> None:
        self.adapters.append(adapter)

    def _apply(self, policy: RuntimePolicy, *, dry_run: bool = False) -> dict[str, Any]:
        details: dict[str, Any] = {}
        for adapter in self.adapters:
            prev = adapter.state.dry_run
            adapter.state.dry_run = dry_run or prev
            try:
                applied = adapter.apply(policy.parameters)
                if applied:
                    details[adapter.layer] = applied
            finally:
                adapter.state.dry_run = prev
        return details

    def deploy_shadow(self, candidate: CandidatePolicy) -> DeploymentResult:
        self.registry.register(candidate.policy)
        self.registry.set_shadow(candidate.policy.id)
        details = self._apply(candidate.policy, dry_run=True)
        return DeploymentResult(
            success=True,
            mode=DeploymentMode.SHADOW,
            active_policy_id=self.registry.active().id if self.registry.active() else None,
            message="shadow registered (dry-run apply)",
            details=details,
        )

    def deploy_canary(self, candidate: CandidatePolicy, *, percent: float = 10.0) -> DeploymentResult:
        self.registry.register(candidate.policy)
        self.registry.set_canary(candidate.policy.id, percent=percent)
        details = self._apply(candidate.policy, dry_run=False)
        return DeploymentResult(
            success=True,
            mode=DeploymentMode.CANARY,
            active_policy_id=self.registry.active().id if self.registry.active() else None,
            canary_percent=percent,
            message=f"canary {percent}%",
            details=details,
        )

    def promote(self, candidate: CandidatePolicy) -> DeploymentResult:
        self.registry.register(candidate.policy)
        policy = self.registry.set_active(candidate.policy.id)
        details = self._apply(policy, dry_run=False)
        self.registry.clear_canary()
        self.registry.clear_shadow()
        return DeploymentResult(
            success=True,
            mode=DeploymentMode.FULL,
            active_policy_id=policy.id,
            message="promoted to active",
            details=details,
        )

    def promote_canary(self) -> DeploymentResult:
        """Promote the current canary policy to 100% active (Option A)."""
        canary = self.registry.canary()
        if canary is None:
            return DeploymentResult(
                success=False,
                mode=DeploymentMode.FULL,
                active_policy_id=self.registry.active().id if self.registry.active() else None,
                message="no canary policy to promote",
            )
        policy = self.registry.set_active(canary.id)
        details = self._apply(policy, dry_run=False)
        self.registry.clear_canary()
        self.registry.clear_shadow()
        return DeploymentResult(
            success=True,
            mode=DeploymentMode.FULL,
            active_policy_id=policy.id,
            canary_percent=0.0,
            message=f"promoted canary {policy.id} to active (100%)",
            details=details,
        )

    def rollback(self, *, to_policy: RuntimePolicy | None = None) -> DeploymentResult:
        target = to_policy
        active = self.registry.active()
        if target is None and active and active.rollback_policy_id:
            target = self.registry.get(active.rollback_policy_id)
        if target is None:
            # fall back to parent
            if active and active.parent_policy_id:
                target = self.registry.get(active.parent_policy_id)
        if target is None:
            return DeploymentResult(
                success=False,
                mode=DeploymentMode.ROLLBACK,
                active_policy_id=active.id if active else None,
                message="no rollback target",
            )
        self.registry.set_active(target.id)
        details = self._apply(target, dry_run=False)
        self.registry.clear_canary()
        self.rollback_count += 1
        return DeploymentResult(
            success=True,
            mode=DeploymentMode.ROLLBACK,
            active_policy_id=target.id,
            message=f"rolled back to {target.id}",
            details=details,
        )

    def active_policy(self) -> RuntimePolicy | None:
        return self.registry.active()

    def resolve_for_request(self, *, agent_id: str = "") -> RuntimePolicy | None:
        return self.registry.resolve(agent_id=agent_id)

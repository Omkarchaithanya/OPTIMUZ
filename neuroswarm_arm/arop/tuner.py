"""Rule-based AROP decision engine — one clamped param change per cycle."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from neuroswarm_arm.arop.metrics_parser import MetricsBundle
from neuroswarm_arm.arop.policy_state import (
    CLAMP_ACCEPT,
    CLAMP_DRAFT_K,
    CLAMP_GOVERNOR,
    PolicyState,
)

LOG = logging.getLogger(__name__)

# Latency slack: if avg_latency_ms is below this absolute ms budget, we have slack
# to grow draft_k. Tunable constant for v1 (not a silent metric default).
LATENCY_SLACK_MS = 2500.0


@dataclass(frozen=True, slots=True)
class Decision:
    action: str  # change | hold | skip | metric_unavailable
    param: str | None
    before: Any
    after: Any
    rule_id: str
    rationale: str
    metrics_used: dict[str, Any]


def _metrics_snapshot(bundle: MetricsBundle) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if bundle.hotspots is not None:
        out["top_function"] = bundle.hotspots.top_function
        out["top_pct"] = bundle.hotspots.top_pct
        out["unknown_symbol_pct"] = bundle.hotspots.unknown_symbol_pct
        out["contaminated"] = bundle.hotspots.contaminated
    if bundle.simd is not None:
        out["simd_instruction_pct"] = bundle.simd.simd_instruction_pct
        out["neon_pct"] = bundle.simd.neon_pct
        out["sve_pct"] = bundle.simd.sve_pct
    if bundle.cascade is not None:
        out["tier1_hit_rate"] = bundle.cascade.tier1_hit_rate
        out["tier2_hit_rate"] = bundle.cascade.tier2_hit_rate
        out["tier3_hit_rate"] = bundle.cascade.tier3_hit_rate
        out["overall_acceptance_rate"] = bundle.cascade.overall_acceptance_rate
        out["avg_latency_ms"] = bundle.cascade.avg_latency_ms
    if bundle.governor is not None:
        out["thinking_tokens_avg"] = bundle.governor.thinking_tokens_avg
        out["cap_b_used"] = bundle.governor.cap_b_used
    if bundle.throughput is not None:
        out["predicted_tokens_seconds"] = bundle.throughput.predicted_tokens_seconds
    if bundle.errors:
        out["errors"] = dict(bundle.errors)
    return out


def decide(
    bundle: MetricsBundle,
    policy: PolicyState,
    *,
    baseline_tok_s: float | None = None,
) -> Decision:
    """Evaluate rules in order R0 → R1 → R2 → R3 → R4. First match wins; one param max."""
    snap = _metrics_snapshot(bundle)
    clamped = policy.clamp()
    if baseline_tok_s is not None:
        snap["baseline_tok_s"] = float(baseline_tok_s)

    # R0 — contamination gate (Proposal A)
    if bundle.hotspots is None:
        return Decision(
            action="metric_unavailable",
            param=None,
            before=None,
            after=None,
            rule_id="R0",
            rationale="code_hotspots unavailable — cannot evaluate contamination; skip cycle",
            metrics_used=snap,
        )
    if bundle.hotspots.contaminated:
        return Decision(
            action="skip",
            param=None,
            before=None,
            after=None,
            rule_id="R0",
            rationale=(
                "profiling contaminated: "
                f"{bundle.hotspots.contamination_reason or 'unknown'}; skip tuning"
            ),
            metrics_used=snap,
        )

    # R1 — shrink draft_k: ggml-saturated top hotspot + low tier1 hit rate
    if bundle.cascade is None:
        LOG.info("R1/R2 skipped: cascade metrics unavailable")
    else:
        hs = bundle.hotspots
        ggml_hot = "ggml" in hs.top_function.lower() and hs.top_pct > 60.0
        low_t1 = bundle.cascade.tier1_hit_rate < 0.6
        if ggml_hot and low_t1:
            before = clamped.cascade_draft_k
            after = max(CLAMP_DRAFT_K[0], before - 1)
            if after == before:
                return Decision(
                    action="hold",
                    param="cascade_draft_k",
                    before=before,
                    after=after,
                    rule_id="R1",
                    rationale=(
                        f"R1 trigger (ggml top={hs.top_pct:.1f}%, "
                        f"tier1_hit={bundle.cascade.tier1_hit_rate:.3f}) "
                        f"but cascade_draft_k already at floor {CLAMP_DRAFT_K[0]}"
                    ),
                    metrics_used=snap,
                )
            return Decision(
                action="change",
                param="cascade_draft_k",
                before=before,
                after=after,
                rule_id="R1",
                rationale=(
                    f"drafter compute-saturated (top={hs.top_function!r} {hs.top_pct:.1f}%) "
                    f"and tier1_hit_rate={bundle.cascade.tier1_hit_rate:.3f}<0.6 → "
                    f"decrease cascade_draft_k {before}→{after}"
                ),
                metrics_used=snap,
            )

        # R2 — grow draft_k when tier1 is healthy and latency has slack
        high_t1 = bundle.cascade.tier1_hit_rate > 0.9
        latency_slack = bundle.cascade.avg_latency_ms < LATENCY_SLACK_MS
        if high_t1 and latency_slack:
            before = clamped.cascade_draft_k
            after = min(CLAMP_DRAFT_K[1], before + 1)
            if after == before:
                return Decision(
                    action="hold",
                    param="cascade_draft_k",
                    before=before,
                    after=after,
                    rule_id="R2",
                    rationale=(
                        f"R2 trigger (tier1_hit={bundle.cascade.tier1_hit_rate:.3f}, "
                        f"latency={bundle.cascade.avg_latency_ms:.1f}ms) "
                        f"but cascade_draft_k already at ceiling {CLAMP_DRAFT_K[1]}"
                    ),
                    metrics_used=snap,
                )
            return Decision(
                action="change",
                param="cascade_draft_k",
                before=before,
                after=after,
                rule_id="R2",
                rationale=(
                    f"tier1_hit_rate={bundle.cascade.tier1_hit_rate:.3f}>0.9 and "
                    f"latency_slack ({bundle.cascade.avg_latency_ms:.1f}ms<"
                    f"{LATENCY_SLACK_MS}) → increase cascade_draft_k {before}→{after}"
                ),
                metrics_used=snap,
            )

    # R3 — tighten governor cap when thinking tokens overrun
    if bundle.governor is None:
        LOG.info("R3 skipped: governor metrics unavailable")
    else:
        avg = bundle.governor.thinking_tokens_avg
        cap = clamped.governor_thinking_cap
        if avg > cap * 1.15:
            before = cap
            after = max(CLAMP_GOVERNOR[0], int(before * 0.9))
            if after == before:
                return Decision(
                    action="hold",
                    param="governor_thinking_cap",
                    before=before,
                    after=after,
                    rule_id="R3",
                    rationale=(
                        f"R3 trigger (thinking_avg={avg:.1f} > {cap}*1.15) "
                        f"but governor_thinking_cap already at floor {CLAMP_GOVERNOR[0]}"
                    ),
                    metrics_used=snap,
                )
            return Decision(
                action="change",
                param="governor_thinking_cap",
                before=before,
                after=after,
                rule_id="R3",
                rationale=(
                    f"thinking_tokens_avg={avg:.1f} > governor_thinking_cap={cap}*1.15 → "
                    f"tighten cap 10% {before}→{after}"
                ),
                metrics_used=snap,
            )

    # R4 — Option A: tok/s vs baseline → nudge accept_threshold (±0.05, clamped)
    # Lower threshold when throughput is strong; raise when degraded. No GGUF swap.
    if (
        baseline_tok_s is not None
        and baseline_tok_s > 0
        and bundle.throughput is not None
    ):
        tok = float(bundle.throughput.predicted_tokens_seconds)
        ratio = tok / float(baseline_tok_s)
        snap["tok_s"] = tok
        snap["tok_s_ratio"] = ratio
        before = clamped.tier_escalation_confidence
        if ratio > 0.95:
            after = max(CLAMP_ACCEPT[0], min(CLAMP_ACCEPT[1], before - 0.05))
            if after != before:
                return Decision(
                    action="change",
                    param="tier_escalation_confidence",
                    before=before,
                    after=after,
                    rule_id="R4",
                    rationale=(
                        f"tok/s={tok:.3f} > 95% of baseline={baseline_tok_s:.3f} "
                        f"(ratio={ratio:.3f}) → lower accept_threshold {before}→{after}"
                    ),
                    metrics_used=snap,
                )
        elif ratio < 0.80:
            after = max(CLAMP_ACCEPT[0], min(CLAMP_ACCEPT[1], before + 0.05))
            if after != before:
                return Decision(
                    action="change",
                    param="tier_escalation_confidence",
                    before=before,
                    after=after,
                    rule_id="R4",
                    rationale=(
                        f"tok/s={tok:.3f} < 80% of baseline={baseline_tok_s:.3f} "
                        f"(ratio={ratio:.3f}) → raise accept_threshold {before}→{after}"
                    ),
                    metrics_used=snap,
                )

    return Decision(
        action="hold",
        param=None,
        before=None,
        after=None,
        rule_id="none",
        rationale="no rule triggered",
        metrics_used=snap,
    )


def apply_decision(policy: PolicyState, decision: Decision) -> PolicyState:
    """Return a new PolicyState with the decision applied (or unchanged)."""
    if decision.action != "change" or decision.param is None:
        return policy.copy()
    new = policy.copy()
    if decision.param == "cascade_draft_k":
        new.cascade_draft_k = int(decision.after)
    elif decision.param == "tier_escalation_confidence":
        new.tier_escalation_confidence = float(decision.after)
    elif decision.param == "governor_thinking_cap":
        new.governor_thinking_cap = int(decision.after)
    else:
        LOG.warning("unknown param %s — leaving policy unchanged", decision.param)
        return policy.copy()
    return new.clamp()


def recommend_quant_preference(
    *,
    ggml_sample_share: float,
    hot_decode_pct: float = 50.0,
) -> str | None:
    """Policy bias only for CostRouter/AQR — never recommends GGUF file swaps.

    When decode is ggml-dominated, prefer existing Q4_0 containers.
    """
    if ggml_sample_share > hot_decode_pct:
        return "Q4_0"
    return None

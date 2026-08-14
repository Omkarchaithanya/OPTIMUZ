# AROP Architecture (Plane 5)

> **Naming:** NEXUS **Layer 5 = MAKS** (KV control plane). **AROP = Plane 5** Autonomic Runtime Optimization Plane. AROP optimizes policies across Layers 1–5; it does not replace MAKS.

## Pipeline

```
Observe → Normalize → Store → Analyze → Reflect → Generate Candidate Policies
→ Offline Evaluation → Shadow Execution → Statistical Validation → Safety Verification
→ Canary Deployment → Continuous Monitoring → Rollback → Knowledge Update
```

Never: Reflect → Deploy.

## Engines

| Engine | Role |
|--------|------|
| Observation | `ObservationProvider` backends (Performix, OTel, Prometheus, Linux perf, PMU, Runtime) |
| Knowledge | Mem0 runtime memory + OKF engineering memory |
| Reflection | `ReflectionStrategy` — Rule (knobs) / **GEPA (text-only Genetic-Pareto)** / Hybrid / Human |
| Optimization | Materializes immutable `RuntimePolicy` |
| Experiment | Offline / shadow / canary |
| Replay | Episode replay under candidate policies |
| Validation | Multi-metric scorecard + Welch / effect size |
| Safety | SLO / budget / regression gates |
| Deployment | Shadow / canary / promote / rollback + layer adapters |
| Evolution | Policy lineage → OKF |

## Dependency inversion

- GEPA is a **text-only ReflectionStrategy** (official Genetic-Pareto); not a knob deployer. See [gepa.md](gepa.md).
- Performix is an **ObservationProvider**, not the optimizer.
- ASCR consumes policies via `PolicyRegistryBackedAgent` / `RLAction` adapters.

## Package

`neuroswarm_arm/evolution/` — see `factory.build_arop()` and FastAPI `/arop/*`.

## AROP v1 CLI tuner (rule-based, shipping now)

Standalone module [`neuroswarm_arm/arop/`](../../neuroswarm_arm/arop/) — **independent of** the evolution `RuntimeOptimizer` pipeline for v1.

- Rule-based only (no PPO / GEPA / Mem0 in this path).
- Consumes honest `apx` JSON + benchmark outputs; fail-loud on missing/null fields (never invent `0`).
- Dry-run by default: `python -m neuroswarm_arm.arop.evolve_cycle`.
- Live apply restarts **gateway** only (`NSA_ASCR_DRAFT_LEN` / `NSA_ASCR_ACCEPT_THRESHOLD`); no runtime GGUF swap.
- Preflight: `python -m neuroswarm_arm.arop.preflight` / `scripts/arop-preflight.sh` — requires `NSA_PERFORMIX_ALLOW_DEMO=0`, `source=apx`, rejects `posix_fallocate`/low-sample captures; logs CPU features honestly (does not invent KleidiAI/SME2/CSS V3/MTE).
- **Not claimed on Axion MVP:** CSS V3, CXL, MTE, SME2 product acceleration, true P/D disaggregation, or dynamic multi-quant fleets.
- The `neuroswarm_arm/evolution/` Plane 5 pipeline closes the **rule/Performix** knob loop (ASCR thresholds, quant/tier policy bias, optional `NSA_AROP_LOOP`); GEPA remains text-only. Standalone [`neuroswarm_arm/arop/`](../../neuroswarm_arm/arop/) CLI tuner is still the judge-facing v1 path.
- **Why no PPO / GEPA-as-knobs / GRPO:** [ADR 0005](adr/0005-rule-based-closed-loop-not-rl.md). Axion cascade stays **0.5B / 3B / 7B Q4** — optimize knobs and KleidiAI, not model scale.
- See [`neuroswarm_arm/arop/README.md`](../../neuroswarm_arm/arop/README.md).

## Config

| Env | Default | Meaning |
|-----|---------|---------|
| `NSA_AROP_ENABLED` | `1` | Enable plane |
| `NSA_AROP_LOOP` | `0` | Background `RuntimeOptimizer.run_once` cadence (`NSA_AROP_INTERVAL`) |
| `NSA_AROP_PERFORMIX` | `0` | Real `apx` recipes |
| `NSA_AROP_REFLECTION` | `hybrid` | `rule\|performix\|gepa\|hybrid\|offline_llm` — hybrid knobs use PerformixAware rules; GEPA stays text-only |
| `NSA_AROP_CANARY_PCT` | `5` | Canary traffic % |
| `NSA_AROP_AUTO_PROMOTE` | `0` | Promote after canary (stays off for safety) |
| `NSA_AROP_MIN_IMPROVEMENT` | `0.01` | Primary metric delta |

## Closed loop (rule / Performix — not PPO)

Evolution path is closed for **rule + Performix** knobs:

1. `PerformixAwareRuleStrategy` proposes deltas only when `performix_available > 0` and honest keys (`ipc` / `hotspot_top_pct` / `cache_miss_rate`) are present — never invents zeros.
2. `ASCRDeploymentAdapter` → `ASCREngine.apply_rl_action` updates live thresholds.
3. `AQRDeploymentAdapter` biases `AQRQuantConnector` preference + `CostRouter` tier floor (`cascade_tier_bias`) — **policy bias only**, no GGUF path swap.
4. Optional `NSA_AROP_LOOP=1` runs `run_once` on an asyncio background task; `auto_promote` remains off.
5. Manual promote: `POST /arop/promote` calls `DeploymentEngine.promote_canary()` (canary → 100% active).

### Option A canary flow

```
run_once → validate/safety → deploy_canary(~5%) → monitor
                 ↓
         POST /arop/promote  (operator)
                 ↓
         canary policy becomes active; canary slot cleared
```

- Default canary share: `NSA_AROP_CANARY_PCT=5` (Option A).
- `NSA_AROP_AUTO_PROMOTE=0` — never auto-promote in the Axion demo path.
- Rollback: `POST /arop/rollback`.

### v1 CLI tok/s rule (R4)

[`neuroswarm_arm/arop/tuner.py`](../../neuroswarm_arm/arop/tuner.py): if `baseline_tok_s` is supplied and live tok/s &gt; 95% of baseline → lower `tier_escalation_confidence` by 0.05; if &lt; 80% → raise by 0.05 (clamped). Quant changes are **preference bias only** (`recommend_quant_preference` → existing Q4_0 containers via CostRouter/AQR — no GGUF path rewrite).

Performix GUI zip: [`neuroswarm_arm/arop/performix_zip.py`](../../neuroswarm_arm/arop/performix_zip.py) parses `functions-capture-periodic_sampling.csv` + `metadata.json`; falls back to ggml sample-share vs baseline when tok/s is absent.

Still out of scope: PPO / GRPO, GEPA numeric knobs, runtime weight swaps. See [ADR 0005](adr/0005-rule-based-closed-loop-not-rl.md).

## ADR

- [0001-plane5-not-maks.md](adr/0001-plane5-not-maks.md)
- [0002-propose-only-reflection.md](adr/0002-propose-only-reflection.md)
- [0003-immutable-policies.md](adr/0003-immutable-policies.md)
- [0004-gepa-text-only.md](adr/0004-gepa-text-only.md)
- [0005-rule-based-closed-loop-not-rl.md](adr/0005-rule-based-closed-loop-not-rl.md) — shipping closed loop is rule-based; no PPO/GRPO; GEPA text-only; Axion 0.5B/3B/7B Q4

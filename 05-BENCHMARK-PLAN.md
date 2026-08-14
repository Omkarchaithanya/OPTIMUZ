# Benchmark Plan — Prove Every Claim

> Replace targets with **measured** Axion numbers. Under-claim beats hype.

---

## Required: Arm Performix (GA recipes only)

Hackathon rules name Performix. Capture via host `apx` + `PerformixClient`:

```bash
apx recipe list
bash performix_capture.sh
# publishes to docs/evidence/performix/
```

| # | Recipe id | Why |
|---|---|---|
| 1 | `code_hotspots` | Flame / function attribution |
| 2 | `cpu_microarchitecture` | Topdown (frontend/backend bound) |
| 3 | `instruction_mix` | **SIMD / SVE2 / I8MM proof** (Arm’s own NEON example recipe) |
| 4 | `memory_access` | SPE load/store latency |
| 5 | `system_characterization` | ASCT preview — platform bring-up |
| 6 | `system_utilization` | Host utilization (present on this `apx` install — see `00-recipe-list.txt`) |

**CLI truth:** current `apx` uses `recipe run <id> --json` → `run export` (see `performix_client.py`). Do not document fake `--output` / `--duration` as primary flags (use `--timeout`).

### Stock vs KleidiAI

```bash
# Baseline
NSA_LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server docker compose up -d --force-recreate tier1 tier2 tier3
OUT_DIR=benchmarks/results/performix/stock bash performix_capture.sh

# Optimized
bash scripts/deploy-kleidiai-tiers.sh
OUT_DIR=benchmarks/results/performix bash performix_capture.sh
COMPARE=1 bash performix_capture.sh
```

Expect Instruction Mix / hotspots to show higher SIMD/I8MM share on Kleidi image.

---

## Application benches (real JSON)

```bash
uv sync --all-groups
uv run python benchmarks/run_all.py --out benchmarks/results/run_all.json
# or full evidence pack:
bash scripts/capture-evidence.sh
```

Key scripts:

| Script | Proves |
|---|---|
| `cascade_acceptance.py` | ASCR accept rate / latency |
| `router_accuracy.py` | Top-K tool routing |
| `governor_tokens.py` | RTG token caps |
| `kv_*.py` | KV share / compress / latency |
| `economics.py` | tokens/$ model |
| `speculative_tool_bench.py` | Speculative tool-call hit rate / latency saved vs sync MCP (`make bench-tool-spec`) |

Publish copies under `docs/evidence/latest/` (gitignored raw dir: `benchmarks/results/`).

### Closed-loop AROP benchmark

Use this to prove the optimization loop itself, not only before/after numbers:

```bash
uv run python benchmarks/arop_closed_loop_suite.py
# optional when gateway is running:
uv run python benchmarks/arop_closed_loop_suite.py --gateway-url http://127.0.0.1:8000
```

Artifacts:

| Artifact | Purpose |
|---|---|
| `work/benchmarks/arop_closed_loop/arop_closed_loop_report.json` | Machine-readable gate results and score |
| `work/benchmarks/arop_closed_loop/arop_closed_loop_report.md` | Company-style scorecard for review/demo |

What it validates:

| Gate | Why it matters |
|---|---|
| Clean `source=apx` profile can trigger R1 | Real Performix evidence drives an ASCR knob proposal |
| Contaminated profile skips tuning | Closed loop refuses unsafe/low-integrity evidence |
| Governor overrun triggers R3 | RTG optimization is checked independently from ASCR |
| Apply + recapture rollback path runs | A worse post-change metric restores the prior policy |
| Missing Performix metric is fail-loud | No fake zero/default metric can produce a false optimization |

---

## Topology honesty

On Axion `c4a-standard-8` (GCP C4A = **single UMA domain**):

```bash
numactl --hardware   # expect 1 node
lscpu | grep -i numa
bash scripts/probe-numa.sh   # → docs/evidence/latest/numa-status.json
# expect: cross_numa_penalty_applicable=false, locality_mode=cache_aware
# expect cpusets: tier1=0-1, tier2=2-4, tier3=5-7
```

| Configuration | Draft | Verifier | Notes |
|---|---|---|---|
| Baseline | OS default | OS default | Highest migration (control) |
| Affinity-aware (Axion) | CPU 0–1 | CPU 2–7 (split 2–4 / 5–7) | Defensible on 1-NUMA |
| NUMA-aware (future) | NUMA 0 | NUMA 1 | Only when `numa_nodes > 1` |

Do **not** claim measurable NUMA-split speedup or “NUMA activated” on this VM. Claim adaptive HAL + cache-aware affinity. Multi-NUMA bind is topology-gated (`NSA_NUMA_POLICY` / `NSA_LOCALITY_MODE`) and inactive when `numa_nodes==1`.

---

## Target → measured table (fill before submit)

| Claim | Target (aspirational) | Measured (Axion) | Artifact |
|---|---|---|---|
| Cascade latency / accept | — | ASCR accept **~0.42**; sample chat ~66s / tier3; cascade_acceptance tier_used=2 | `prometheus-metrics.txt`, `chat-completion.json`, `run_all.json` |
| Router schema reduction | ≥90% | **16%** avg token reduction (route top-k); tools-route sample **47%** prompt token cut; top-1 **83%**, top-3 **100%**; **6** tools indexed | `run_all.json` / `tools-route.json` |
| Kleidi vs stock tok/s | >1× | Kleidi image proven live (`nexus-arm/llama-kleidiai:server`); full tok/s A/B still optional | `kleidiai-runtime-gate.txt`, `docker-compose-ps.txt` |
| Instruction Mix SIMD share | up vs stock | Kleidi static mix: **Advanced SIMD (NEON) 1.61%** + **SVE 0.34%** (~**1.95%** SIMD-class); integer 44.4% / load-store 27.7% | `static_instruction_mix.csv`, `02-instruction_mix.json` |
| $/1M tokens vs H100 spot | ≥3.5× | sample RCIS **~$0.0016**/req; economics savings score **0.57** (under-claim; not H100 A/B yet) | `chat-completion.json`, `run_all.json` economics |
| Speculative tool call latency / hit rate | overlap MCP while cascade runs; warm-cache hits | **hit_rate 0.50**; avg_time_saved **46.8 ms**; p50 **57.9** / p95 **75.9** ms; latency_speedup **1.45×**; tokens_per_dollar_delta **+5723** (ref 12580×speedup−ref); predicted_correct_rate **0.50** (15/30 cache-hit-likely) | `docs/evidence/speculative_tool/` (`make bench-tool-spec`) |

---

## Reproduce for judges

```bash
uv sync --all-groups
cp .env.example .env
bash scripts/deploy-kleidiai-tiers.sh
bash scripts/capture-evidence.sh
bash performix_capture.sh   # requires apx + Arm account
```

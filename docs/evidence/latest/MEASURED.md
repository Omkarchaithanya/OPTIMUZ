# Measured (Axion suite `20260803-1321`)

<<<<<<< Updated upstream
Sources: [`BENCHMARKS.md`](../BENCHMARKS.md) ← `benchmarks/results/axion-20260803-1321/SUMMARY.json` + published copies under `docs/evidence/latest/` and `docs/evidence/performix/`.
=======
**Full before/after table:** [`../BENCHMARKS.md`](../BENCHMARKS.md)

Sources: `docs/evidence/latest/run_all.json` (2026-07-18) + live layer verify 2026-07-21 (`docs/evidence/latest/layer-verify/` + [`LAYER_SCORECARD.md`](LAYER_SCORECARD.md)).
>>>>>>> Stashed changes

| Metric | Measured | Notes |
|---|---|---|
| Router top-1 / top-3 | **100% / 100%** | `layer-verify/06-router_accuracy.json` (suite stamp) |
| Router avg token reduction | **87.4%** | Schema Top-K vs full set |
| Cascade `tier_used` (run_all) | **2** | `latest/run_all.json` |
| Cascade live acceptance | **1.0** (n=64) | `acceptance_rate_live.json`; avg_latency_ms=0 in artifact — do not treat as wall-clock TTFT |
| Governor mean cap | **268** | vs legacy **666** (`run_all`) → **−59.7%** |
| Governor live | **TBD — not measured** | Live run aborted (`RemoteDisconnected`); see SUMMARY `tbd` |
| Economics savings score | **0.78** | `run_all` economics |
| MAKS multi-agent dedup savings | **87.5%** | 8 agents × 20 prompts; `layer-verify/14-maks-dedup.json` |
| MAKS sharing savings | **100%** | shared_pages/pages on dedup run |
| MAKS pool bytes | **671184 → 83898** | control vs dedup |
| Kleidi tok/s (dated baselines) | Qwen0.5B **60.5→99.1 (+64%)**; Llama3.2-3B **19.8→25.0 (+26%)**; DeepSeek-R1-8B **6.8→9.1 (+34%)** | `kleidi_ab.json` measured 2026-07-22; live image `nexus-arm/llama-kleidiai:server` verified |
| Performix code_hotspots | **source=apx**, `libggml-cpu` **89.8%**, **88025** samples | PID **3592588** = DeepSeek-R1-Distill-Qwen-7B; load on `:8083`; `posix_fallocate` absent from top |
| Instruction Mix (live attach) | **TBD — not measured** | Prior dated A/B still in `performix/COMPARISON.md` — not this SUMMARY |
| TTFT | **TBD — not measured** | — |
| NUMA nodes | **1** | `locality_mode=cache_aware` |
| `NSA_PERFORMIX_ALLOW_DEMO` | **0** | — |

Older rows (2026-07-18 / 2026-07-21 packs) remain in git history; prefer this stamp for judge tables.

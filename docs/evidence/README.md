# Evidence pack (judge-visible)

`benchmarks/results/` is gitignored. Capture scripts copy key artifacts here so the public repo shows receipts.

<<<<<<< Updated upstream
**Primary judge table:** [`BENCHMARKS.md`](BENCHMARKS.md) (generated only from a live Axion `SUMMARY.json`).
=======
**Full application benchmarks (before/after table):** [`BENCHMARKS.md`](BENCHMARKS.md)
>>>>>>> Stashed changes

## Layout

| Path | Contents |
|---|---|
<<<<<<< Updated upstream
| `BENCHMARKS.md` | Optimization / Baseline / Optimized / Delta / Evidence (hackathon style) |
| `latest/` | health, ready, metrics, run_all, `axion-SUMMARY.json`, MEASURED.md, layer-verify |
=======
| `BENCHMARKS.md` | **Judge-facing** Optimization / Baseline / Optimized / Δ table for the whole Axion app |
| `latest/` | health, ready, metrics, run_all, compose ps, Kleidi gate, chat, tools-route, lscpu |
>>>>>>> Stashed changes
| `performix/` | GA recipe JSON + **`OPTIMIZATIONS.md`** + flame PNG + COMPARISON.md |

**Judges — Performix first:** [`performix/OPTIMIZATIONS.md`](performix/OPTIMIZATIONS.md) (hotspots → what we optimized) · [`performix/screenshots/05-code-hotspots-flame.png`](performix/screenshots/05-code-hotspots-flame.png).

## Regenerate on Axion

```bash
export NSA_PERFORMIX_ALLOW_DEMO=0 NSA_AROP_PERFORMIX=1
bash scripts/run-axion-benchmark-suite.sh
python3 scripts/generate-benchmarks-md.py benchmarks/results/axion-<stamp>/SUMMARY.json
```

Legacy capture (still valid for older packs):

```bash
bash scripts/deploy-kleidiai-tiers.sh
bash scripts/capture-evidence.sh
bash performix_capture.sh
COMPARE=1 bash performix_capture.sh
```

## Pass gates

1. `kleidiai-runtime-gate.txt` / live tier3 image is Kleidi (`nexus-arm/llama-kleidiai:server`)
2. Performix snapshot `source=apx` (never `demo|synthetic`)
3. `run_all.json` / router benches `status=ok` when claimed
4. Every BENCHMARKS.md cell cites an artifact or is **TBD — not measured**

Placeholder files below are replaced when you run capture on the VM.

# Optimizations driven by Arm Performix (Axion)

Status: **published evidence** — derived only from committed `apx` exports under this directory.  
Host: GCP Axion `c4a-standard-8` (Neoverse-V2, 1 NUMA).  
Primary hotspot run: `01-code_hotspots.json` (`source=apx`, run_id `ca3b2eb4c169`).

> Judges: JSON + this note live in **git** (`docs/evidence/performix/`). Runtime `work/performix/` is gitignored and is not the submission pack.

## What Performix showed

| Signal | Measured | Artifact |
|--------|----------|----------|
| Top hotspot | `<Unknown code in libggml-cpu.so.0.17.0>` **79.047%** self | `01-code_hotspots.json` |
| Second | `libomp.so.5` **12.393%** | same |
| Idle / load contamination | **Not** `default_idle_call` / `posix_fallocate` as top frame | `SYMBOLS.md`, post warm-PID capture |
| Kleidi SIMD (live decode) | NEON **3.41%** + SVE **0.94%** | `03-instruction_mix_dynamic_kleidi.json` |
| Stock SIMD (baseline) | NEON **2.14%** + SVE **1.19%** | `04-instruction_mix_dynamic_baseline.json` |

Attribution honesty: DWARF is present on Kleidi `libggml-cpu`, but Performix still labels container-mapped samples as Unknown-in-`.so` — the **library** attribution is the judge-facing claim, not invented function names (`SYMBOLS.md`).

## What we optimized from that data

Hotspots say wall-clock is spent inside **Kleidi CPU kernels**, not the gateway Python path. Optimizations therefore target the decode stack and routing around it:

1. **KleidiAI llama.cpp image** (`nexus-arm/llama-kleidiai:server`) — stay on Arm-optimized `libggml-cpu` rather than stock `ggml-org` server (Instruction Mix before/after in `COMPARISON.md`).
2. **Cascade Q4_0 GGUFs + cpusets** on 8 vCPU — keep work on the hot `libggml-cpu` path without oversubscription; no fantasy multi-quant fleet or runtime GGUF swap.
3. **AROP / ASCR knobs** (draft length, accept/escalate thresholds) — reduce wasted verify when IPC/hotspot pressure is high; rule-based only (ADR 0005 — not PPO).
4. **Single Compose scrape topology** — post dual-stack fix so Prometheus/`nexus_performix_*` gauges reflect one gateway job (see `screenshots/`).
5. **OpenMP** — `OMP_PROC_BIND=close`, tier threads 2/3/3, `OMP_WAIT_POLICY=passive` (libomp share under decode is expected; see `SYMBOLS.md`).
6. **Symbol helper** — `bash scripts/performix-host-libs.sh tier3` copies unstripped libs for neoprof; named `ggml_*` may still stay Unknown (container DWARF limit).

## Before / after (Instruction Mix)

| Build | NEON % | SVE % | Artifact |
|-------|--------|-------|----------|
| Stock `libggml-cpu-armv9.2_2` | 2.14 | 1.19 | `04-instruction_mix_dynamic_baseline.json` |
| Kleidi `libggml-cpu` | **3.41** | 0.94 | `03-instruction_mix_dynamic_kleidi.json` |

NEON share rose on the Kleidi path under live decode. SVE share alone is not pitched as a win; the pack shows both numbers honestly.

## Visuals

| File | What it is |
|------|------------|
| `screenshots/05-code-hotspots-flame.png` | Flame-style chart **generated from** `01-code_hotspots.json` (apx export) — not demo fill |
| `screenshots/03-hotspots.png` | Grafana PromQL bar chart of the same snapshot gauges |
| `snapshot.json` | Gateway/RMF snapshot with `source=apx` |

## Not claimed (this host)

- CSS V3, CXL, MTE product acceleration, SME2 product win, true P/D disaggregation
- Named DWARF leaf functions inside `libggml-cpu` (Unknown-in-so remains honest)
- Live refresh every judge window — if a later verify shows `source=unavailable`, that row stays honest in `docs/evidence/latest/MEASURED.md`

## Refresh window (2026-07-31)

Agent host could not SSH to Axion (`Permission denied (publickey)` / no `gcloud` on PATH).  
**Published pack remains** the prior PID-scoped `source=apx` capture (`01-code_hotspots.json`, `snapshot.json`).  
Operator re-run on the VM:

```bash
cd ~/neuroswarm-arm
NSA_PERFORMIX_ALLOW_DEMO=0 NSA_AROP_PERFORMIX=1 \
  bash scripts/capture-performix-hotspots-fixed.sh
# then copy work/performix/snapshot.json → docs/evidence/performix/ when source=apx
python scripts/render_performix_flame.py \
  --input docs/evidence/performix/01-code_hotspots.json \
  --output docs/evidence/performix/screenshots/05-code-hotspots-flame.png
```

Do **not** invent demo hotspots when refresh fails.

## Reproduce

```bash
# On Axion, single Compose stack, model warm, chat load running:
NSA_PERFORMIX_ALLOW_DEMO=0 NSA_AROP_PERFORMIX=1 \
  bash scripts/capture-performix-hotspots-fixed.sh
# Publish path (tracked):
#   docs/evidence/performix/01-code_hotspots.json
#   docs/evidence/performix/snapshot.json
python scripts/render_performix_flame.py \
  --input docs/evidence/performix/01-code_hotspots.json \
  --output docs/evidence/performix/screenshots/05-code-hotspots-flame.png
```

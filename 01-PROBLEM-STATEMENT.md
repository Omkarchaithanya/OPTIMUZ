# The Problem Statement — Optimuz

## The pitch (Devpost Project Overview)

> **Optimuz is a self-evolving, cost-optimized multi-agent AI runtime built natively for Arm Neoverse.** Cloud agents waste money on unused MCP tool schemas, duplicated KV caches, excess reasoning tokens, and GPU prices for memory-bound decode. We fix that with a three-tier CPU-CPU speculative cascade (0.5B → 3B → 8B) on **KleidiAI-optimized llama.cpp**, a semantic MCP tool router (live schema-token reduction ≈0.89 on Axion FastEmbed; aspirational 40→3 count cut is not the same metric), a reasoning-token governor, and a GEPA-style evolution loop driven by **Arm Performix** (Code Hotspots + Instruction Mix + CPU Microarchitecture).
> **Hardware honesty:** The live demo runs on **GCP Axion `c4a-standard-8`** (Neoverse-V2, SVE2/I8MM/BF16). Optimuz **auto-detects NUMA/CXL/MTE at runtime and degrades safely on single-NUMA VMs like Axion**, activating NUMA-split cascades and CXL KV pooling on multi-socket Neoverse hosts (e.g. Graviton4/5 `.16xlarge+`).  
> Result: measurable tokens/$ gains vs GPU spot, one-line Helm/Compose deploy, 6 MCP templates, Grafana cost dashboard, and Performix flame-graph evidence judges can re-run.

---

## Elevator pitch (60s video)

> Agentic AI on cloud is broken — most of the $0.40–$2.00 per request is waste. Optimuz is an Arm-native multi-agent runtime. On GCP Axion we run a three-tier speculative cascade with KleidiAI kernels, a semantic MCP router that ships top-K tools instead of all schemas, and Arm Performix in the loop so Instruction Mix proves SVE2/I8MM utilization. Topology is adaptive: single-NUMA Axion degrades cleanly; multi-socket Neoverse unlocks NUMA-split and CXL KV. One Compose/Helm command. Cheaper than GPU for agent decode — and we show the Performix receipts.

---

## Pain → fix

| Pain | Optimuz fix |
|---|---|
| Tool-schema flood | Semantic MCP router (TurboVec + Top-K) |
| Reasoning-token burn | RTG governor tied to confidence / KV pressure |
| KV duplication | Shared KV path; CXL when topology present |
| GPU lock-in for decode | CPU-CPU cascade + KleidiAI on Axion |
| Invisible cost | Grafana + Prometheus RMF |
| Untuned stacks | AROP + Performix recipes |

---

## Why Cloud AI / why Arm

- Track invites agentic + MCP + Arm cloud (Graviton / Cobalt / **Axion**) + llama.cpp.  
- Decode is memory-bound → Neoverse + KleidiAI I8MM/SVE2.  
- Performix named in rules → we capture real GA recipes, not invented ones.

---

## What we are NOT doing

| Skip | Why |
|---|---|
| Claiming 2-NUMA Graviton5 as the demo box | Demo is Axion single-NUMA — Option A |
| Invented Performix recipes (`system-utilization`) | GA set is five real recipes only |
| Stock llama.cpp while claiming KleidiAI | Image must be `nexus-arm/llama-kleidiai:server` |
| Building another chatbot only | We ship the runtime |

---

## Judges (live Devpost — 3 only)

- Avin Zarlez — Arm Staff SW Engineer, Developer Evangelist  
- Michael Hall — Arm Principal SW Engineer, Developer Evangelist  
- Gabriel Peterson — Arm Senior ML Engineer, Developer Evangelist  

They will run `numactl --hardware`, check Compose IMAGE tags, and look for Performix Instruction Mix. Match evidence to claims.

---

## Tagline

> **"Self-evolving Arm agent runtime on Axion — KleidiAI cascade + Performix Instruction Mix proof — topology that grows with the silicon."**

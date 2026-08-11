<p align="center">
  <img src="./Optimuz_logo.png" alt="Optimuz" width="340" style="border-radius: 10px;">
</p>

<p align="center">
  <strong>Optimuz â€” Optimize, Innovate, Elevate. A self-evolving, cost-optimized multi-agent AI runtime built natively for Arm Neoverse.</strong>
</p>

<p align="center">
  <a href="https://github.com/Omkarchaithanya/Neuroswarm">
    <img src="https://img.shields.io/badge/Architecture-ARM64-FF9900?style=for-the-badge&logo=arm&logoColor=white" alt="ARM64 Architecture">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+">
  </a>
  <a href="https://fastapi.tiangolo.com/">
    <img src="https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="MIT License">
  </a>
</p>

<p align="center">
  <a href="https://huggingface.co/blog/matryoshka">
    <img src="https://img.shields.io/badge/Matryoshka%20Embeddings-Truncating-9B59B6?style=for-the-badge&logoColor=white" alt="Matryoshka Embeddings">
  </a>
  <a href="https://github.com/ryancodrai/turbovec">
    <img src="https://img.shields.io/badge/TurboVec-Indexing-F39C12?style=for-the-badge&logo=github&logoColor=white" alt="TurboVec">
  </a>
  <a href="https://medium.com/ai-science/speculative-decoding-make-llm-inference-faster-c004501af120">
    <img src="https://img.shields.io/badge/Speculative-Decoding-E74C3C?style=for-the-badge&logoColor=white" alt="Speculative Decoding">
  </a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2512.15834">
    <img src="https://img.shields.io/badge/Speculative%20Tool-Calling-3498DB?style=for-the-badge&logo=arxiv&logoColor=white" alt="Speculative Tool Calling">
  </a>
  <a href="https://research.google/blog/speculative-cascades-a-hybrid-approach-for-smarter-faster-llm-inference/">
    <img src="https://img.shields.io/badge/Speculative-Cascading-1ABC9C?style=for-the-badge&logo=google&logoColor=white" alt="Speculative Cascading">
  </a>
  <a href="https://github.com/SalesforceAIResearch/xLAM">
    <img src="https://img.shields.io/badge/xLAM%20Model-Integration-34495E?style=for-the-badge&logo=github&logoColor=white" alt="xLAM Model">
  </a>
</p>

<p align="center">
  <a href="https://cloud.google.com/blog/products/data-analytics/how-the-open-knowledge-format-can-improve-data-sharing">
    <img src="https://img.shields.io/badge/Open%20Knowledge-Format-2ECC71?style=for-the-badge&logo=google-cloud&logoColor=white" alt="Open Knowledge Format">
  </a>
  <a href="https://github.com/mem0ai/mem0">
    <img src="https://img.shields.io/badge/Mem0-Memory%20Management-16A085?style=for-the-badge&logo=github&logoColor=white" alt="Mem0">
  </a>
</p>

**Optimuz** is an advanced MVP running on a single GCP Axion VM, redefining efficient AI workloads. By combining llama.cpp CPU inference on Arm64 with an innovative three-tier CPU cascade routing, semantic MCP tool selection, and reasoning-token governance, we bring the best of model capability to Arm platforms.

[Implementation Plan](04-IMPLEMENTATION-PLAN.md) Â· [Benchmarks](BENCHMARKS.md) Â· [Problem Statement](01-PROBLEM-STATEMENT.md)

---

## The Crisis This Solves

**Agentic AI on the cloud is broken.** Most of the $0.40â€“$2.00 per request is waste. Cloud agents waste money on unused MCP tool schemas, duplicated KV caches, excess reasoning tokens, and expensive GPU prices for memory-bound decode tasks. 

**Optimuz fixes this** with a three-tier CPU-CPU speculative cascade (0.5B â†’ 3B â†’ 8B) on **KleidiAI-optimized llama.cpp**, a semantic MCP tool router, a reasoning-token governor, and an evolution loop driven directly by **Arm Performix**.

<div align="center">

| Pain | The Optimuz Fix |
|---|---|
| **Tool-schema flood** | Semantic MCP router (TurboVec + Top-K schema injection) |
| **Reasoning-token burn** | RTG governor tied to confidence and KV pressure |
| **KV duplication** | Shared KV path; CXL pooling when topology is present |
| **GPU lock-in for decode** | CPU-CPU cascade + KleidiAI optimized exclusively for Axion |
| **Invisible cost** | Grafana + Prometheus RMF dashboards |
| **Untuned stacks** | AROP + Performix Instruction Mix recipes in the loop |

</div>

> [!IMPORTANT]
> **Hardware Honesty:** Our live demo runs on **GCP Axion `c4a-standard-8`** (Neoverse-V2, SVE2/I8MM/BF16). Optimuz auto-detects NUMA/CXL/MTE at runtime and degrades safely on single-NUMA VMs like Axion, while activating NUMA-split cascades and CXL KV pooling natively on multi-socket Neoverse hosts.

---

## Judge Scoring & Benchmark Highlights

We don't just claim performance; we measure it with Arm Performix Instruction Mix receipts. 

<div align="center">

| Metric | Baseline | Optimized | Delta | Verification |
|---|---|---|---|---|
| **KleidiAI Throughput (xLAM-2-1B)** | 60.5 tok/s | 99.08 tok/s | **+63.8%** | `benchmarks/kleidiai_baselines.json` |
| **NEON Instruction Share** | 2.14% | 3.41% | **+59%** | `03-instruction_mix_dynamic_kleidi.json` |
| **Semantic MCP Router Accuracy** | 16.6% (Random) | 1.00 (100%) | **+83.4 pp** | `latest/layer-verify/06-router_accuracy.json` |
| **MCP Token Context Reduction** | Full Catalog | Top-K (3) | **-92.7% tokens** | `work/benchmarks/router_mcpga.json` |
| **Cost Per Request (RCIS sample)** | ~$0.0308 | ~$0.00154 | **~95% cost drop** | `latest/layer-verify/08-economics.json` |
| **MAKS / Multi-Agent KV Dedup** | 671 KB | 83 KB | **-87.5% pool bytes** | `latest/layer-verify/14-maks-dedup.json` |

</div>

---

## The Optimuz Architecture

<div align="center">

| Optimization | Description |
|---|---|
| **Matryoshka Embeddings Truncating** | Dynamically scales embedding dimensions for optimal latency and accuracy, minimizing redundant computations. |
| **Turbovec Indexing** | Blazing-fast ANN (replaces FAISS) natively optimized for ARM architecture to perform rapid semantic searches. |
| **Speculative Decoding** | Generates multiple draft tokens in parallel, vastly increasing throughput for language model outputs. |
| **Speculative Tool Calling** | Overlaps draft tool prediction with main cascade generation (predict â†’ overlap MCP â†’ ToolOutputCache). |
| **Model Cascading** | Intelligent multi-tier CPU cascade routing (Inspired by Google Cascade) that shifts workloads based on reasoning demands. |
| **KV Cache Optimization** | Zero-waste MAKS multi-agent memory KV session deduplication, dropping memory bottlenecks at scale. |
| **xLAM Model Integration** | Powered by Qwen model finetuned (xLAM) for highly effective instruction following and reasoning. |
| **Mem0 (Zero) Memory Management Layer** | Highly efficient memory management layer that aggressively orchestrates active context windows. |
| **Open Knowledge Format (OKF)** | Structured ontology and knowledge files seamlessly compiled and injected into agent contexts. |
| **Quantization & Auto-Truncation** | Models are natively **Q4_0** quantized and automatically auto-truncated to **Q4_0_4_8** for extreme execution performance on Arm. |

</div>

---

## The 5-Plane Acronym Map

<div align="center">

| Acronym | One-liner | Where to Verify |
|---|---|---|
| **HAOE** | Layer-1 task-graph runtime (schedules work; never runs models) | `tests/runtime/haoe` |
| **DIPA** | Layer-2 inference kernel (planner â†’ routers â†’ cascade â†’ backends) | `tests/runtime/dipa` |
| **ASCR** | Adaptive speculative / quality cascade across CPU tiers | `docs/armcascade/` |
| **AROP** | Evolution / runtime optimization loop (Performix-fed policies) | `performix/` |
| **OKF** | Ontology / knowledge files compiled into agent context | `docs/` |
| **AQR** | Adaptive quantization routing metadata | `docs/` |
| **AWPP** | Arm weight / preference policy connector | `docs/` |
| **MAKS** | Memory / KV session services | `docs/evidence/` |
| **RTG** | Reasoning-token governor | `benchmarks/run_all.py` |
| **ACR** | Agent conversation / memory recall plane | `docs/` |

</div>

---

## Why Optimuz Is Unique

**This project:** Masters the hardware. We don't just run inference; we bend it to the will of the Arm architecture.

### Five properties that set this apart

<details>
<summary><b>1. &nbsp;Semantic MCP Tool Router (Turbovec Powered)</b></summary>

Replaces naÃ¯ve injection of all MCP tool schemas with Top-K semantic routing:
`nomic-embed-text-v1.5 â†’ TurboVec (2/4-bit TurboQuant when active; else exact NumPy) â†’ hybrid retrieval â†’ rerank â†’ Top-K schemas â†’ DIPA`

Default `NSA_ROUTER_TURBOVEC_MIN_TOOLS=0` so TurboVec runs whenever the ARM64 wheel imports. Advertised tool YAML IDs match FastMCP execute names natively.
</details>

<details>
<summary><b>2. &nbsp;Speculative Tool Calling (Zero-Latency Workflows)</b></summary>

Overlaps a draft **tool prediction** with the main cascade generation so MCP work can finish (or hit cache) before the actor emits the real `tool_call`. This is **tool-level** speculation.

Draws on [arXiv:2512.15834](https://arxiv.org/abs/2512.15834) (Speculative Tool Calling) and [arXiv:2510.04371](https://arxiv.org/abs/2510.04371) (Speculative Actions). Cache keys are canonical via `ToolOutputCache.make_key(tool_name, args)`.
</details>

<details>
<summary><b>3. &nbsp;HAOE (Layer 1) & DIPA (Layer 2)</b></summary>

**HAOE:** Chat requests execute as HAOE task graphs (route â†’ KV session â†’ DIPA â†’ checkpoint â†’ response). High-confidence turns take the gateway fast-path, lowering orchestration overhead significantly.
**DIPA:** Inference Runtime Kernel. Agents never call llama.cpp / vLLM directly â€” everything flows through DIPA (execution planner â†’ model routers â†’ ASCR â†’ streaming).
</details>

<details>
<summary><b>4. &nbsp;Arm Performix Benchmarked</b></summary>

Baseline checklist showed tier1 chat ~1116ms while `haoe_workflow_latency_ms` ~1970ms. Mitigated by:
1. **MCP process pool** (warm stdio servers).
2. **HAOE fast-path** (high-confidence chat skips full DAG).
3. **ASCR round-1** optimization.

*See `docs/evidence/performix/OPTIMIZATIONS.md` for verifiable flame PNGs and hotspots (`source=apx`, `libggml-cpu` ~79%).*
</details>

<details>
<summary><b>5. &nbsp;Advanced Quantization & Model Cascading</b></summary>

Models are strictly optimized with **Q4_0** quantization and auto-truncated to **Q4_0_4_8**. We use Google Cascade-inspired Model Cascading to optimally route reasoning effort across multiple tiers of compute.
</details>

---

## Repository Structure

```text
Optimuz/
â”œâ”€â”€ benchmarks/         # Arm Performix receipts, metrics, and JSON evidence
â”œâ”€â”€ docker/             # Container specs for CPU-cascade testing
â”œâ”€â”€ docs/               # Architecture ADRs, layer diagrams, and evidence packs
â”œâ”€â”€ helm/               # Kubernetes deployment assets for multi-node testing
â”œâ”€â”€ optimuz/     # Core runtime (HAOE Layer-1 and DIPA Layer-2)
â”œâ”€â”€ scripts/            # Bootstrap, deploy, and bench runner utilities
â””â”€â”€ tests/              # Pytest suite for runtime validation
```

---

## Getting Started (Local Axion MVP)

Run the entire suite locally or on an Axion VM with incredible simplicity:

```bash
# Sync dependencies
uv sync --all-groups

# Setup environment variables
cp .env.example .env   # Linux/macOS; on Windows: Copy-Item .env.example .env

# Fire up the Optimuz platform
docker compose up --build
```

The gateway listens natively on `http://VM_EXTERNAL_IP:8000`.

### Health & Ready Checks
```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
curl http://127.0.0.1:8000/v1/tools/cache
```

### Example Chat Request
```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Plan a cost-optimized ARM inference demo."}],"max_tokens":256}'
```

*For repeatable GCP setup, refer to `docs/gcp-axion-setup.md` or use `scripts/bootstrap-gcp.ps1` and `scripts/bootstrap-vm.sh`. Initial dev target: `c4a-standard-8` with `hyperdisk-balanced`.*

---

*Optimuz: Built for the ARM Cloud AI Optimization Challenge.*




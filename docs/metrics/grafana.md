# Grafana Guide

Dashboards are generated from `neuroswarm_arm.metrics.dashboards.default_dashboards()` into `ops/grafana/dashboards/rmf-*.json`.

## Boards

| UID | Title |
|-----|-------|
| rmf-runtime-overview | Runtime Overview |
| rmf-planner | Planner |
| rmf-routing | Routing |
| rmf-budget | Budget |
| rmf-inference | Inference |
| rmf-haoe | HAOE |
| rmf-dipa | DIPA |
| rmf-memory | Memory |
| rmf-kv | KV |
| rmf-energy | Energy |
| rmf-cost | Cost |
| rmf-arm-hardware | ARM Hardware |
| rmf-performix | Performix |
| rmf-planner-learning | Planner Learning |
| neuroswarm-benchmark-scoreboard | Benchmark Scoreboard |

## Provisioning

Datasource: `ops/grafana/provisioning/datasources/prometheus.yaml`  
Dashboard provider: `ops/grafana/provisioning/dashboards/neuroswarm.yaml` (loads JSON from dashboards dir).

## Regenerate

```bash
python -c "from neuroswarm_arm.metrics.dashboards import write_dashboards; write_dashboards('ops/grafana/dashboards')"
```

Existing `armora-budget-envelope` / `rcis-runtime-cost` boards remain for plane-specific deep dives; RMF boards are the cross-runtime view.

## Benchmark Scoreboard

Run the benchmark first so it writes `work/benchmarks/arop_closed_loop/benchmark.prom`:

```bash
python benchmarks/arop_closed_loop_suite.py --gateway-url http://127.0.0.1:8000
```

The gateway merges that file into `/metrics` via `NSA_BENCHMARK_METRICS_PATH`,
defaulting to `work/benchmarks/arop_closed_loop/benchmark.prom`.

Open Grafana and use the provisioned dashboard:

```text
NeuroSwarm / NeuroSwarm Benchmark Scoreboard
```

Useful checks:

```promql
neuroswarm_benchmark_closed_loop_score
neuroswarm_benchmark_scenario_pass_rate
neuroswarm_benchmark_scenario_pass
```

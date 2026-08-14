#!/usr/bin/env bash
# Fixed Performix Code-Hotspots capture — profiles REAL decode kernels, not model-load syscalls.
# 
# PROBLEM: Previous runs showed posix_fallocate at 56.67% self-time (GGUF mmap during load)
# FIX:    1) Start llama-server, wait for /health=ready (model fully loaded)
#         2) Start sustained load generator (50-100 concurrent prompts) 
#         3) THEN attach apx to WARM PID for 120s+ profiling window
#
# Usage: bash scripts/capture-performix-hotspots-fixed.sh
# Requires: llama-server running (docker compose or ProcessSupervisor), apx on PATH

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PUB="$ROOT/docs/evidence/performix"
WORK="$ROOT/work/performix"
mkdir -p "$PUB" "$WORK"

API="${NSA_CHAT_URL:-http://127.0.0.1:8000/v1/chat/completions}"
LLAMA_HEALTH="${NSA_LLAMA_HEALTH:-http://127.0.0.1:8080/health}"
DURATION="${PERFORMIX_DURATION:-120}"   # 120s minimum for decode to dominate
RECIPE="${NSA_AROP_PERFORMIX_RECIPE:-code_hotspots}"
OUT_JSON="$PUB/code_hotspots_fixed.json"
SNAP="$WORK/snapshot.json"

# Allow demo fallback for local testing
ALLOW_DEMO="${NSA_PERFORMIX_ALLOW_DEMO:-0}"
REQUIRE_LIVE=0
if [[ "${NSA_AROP_PERFORMIX:-0}" == "1" ]] || [[ -n "${NSA_AROP_PERFORMIX_MCP:-}" ]]; then
  REQUIRE_LIVE=1
fi

write_unavailable() {
  local err="${1:-apx_recipe_failed}"
  python3 - "$SNAP" "$err" <<'PY'
import json, sys
from pathlib import Path
snap = Path(sys.argv[1])
err = sys.argv[2]
marker = {
    "available": 0,
    "source": "unavailable",
    "error": err,
    "hotspots": [],
    "ipc": 0.0, "cycles": 0.0, "instructions": 0.0,
    "cache_misses": 0.0, "branch_misses": 0.0, "pmu_available": 0.0,
    "topdown": {"frontend_bound": 0.0, "backend_bound": 0.0},
}
snap.parent.mkdir(parents=True, exist_ok=True)
snap.write_text(json.dumps(marker, indent=2), encoding="utf-8")
print(f"Wrote {snap} source=unavailable error={err}")
PY
}

pick_llama_pid() {
  local pid
  # Prefer DeepSeek / tier3 7B (Axion demo path), then 8B, then any llama-server.
  pid="$(pgrep -af 'DeepSeek-R1|Distill-Qwen-7B' | grep -v 'bash\|pgrep\|capture\|apx' | awk '{print $1; exit}' || true)"
  if [[ -z "$pid" ]]; then
    pid="$(pgrep -af 'llama-server.*llama-3.1-8b' | awk '{print $1; exit}' || true)"
  fi
  if [[ -z "$pid" ]]; then
    local tid
    tid="$(docker compose ps -q tier3 2>/dev/null || true)"
    if [[ -n "$tid" ]]; then
      pid="$(docker top "$tid" 2>/dev/null | awk 'NR>1 && /llama-server/ {print $2; exit}')"
    fi
  fi
  if [[ -z "$pid" ]]; then
    pid="$(pgrep -af 'llama-server' | grep -v 'bash\|pgrep\|capture\|apx' | awk '{print $1; exit}' || true)"
  fi
  echo "${pid:-}"
}

wait_llama_ready() {
  local timeout="${1:-300}"
  local health_url="${2:-$LLAMA_HEALTH}"
  echo "==> Waiting for llama-server health at $health_url (timeout ${timeout}s)..."
  local start
  start="$(date +%s)"
  while true; do
    if curl -sf "$health_url" >/dev/null 2>&1; then
      echo "    llama-server /health OK"
      return 0
    fi
    local now
    now="$(date +%s)"
    if (( now - start > timeout )); then
      echo "    TIMEOUT: llama-server not healthy after ${timeout}s" >&2
      return 1
    fi
    sleep 2
  done
}

fire_burst() {
  echo "==> Starting sustained decode load → $API"
  # Keep llama busy for the FULL recipe window (not a single short burst)
  local model="${NSA_CHAT_MODEL:-}"
  if [[ -z "$model" ]]; then
    if echo "$API" | grep -qE ':8083'; then
      model="tier3"
    else
      model="cascade"
    fi
  fi
  (
    while true; do
      curl -s -X POST "$API" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a 300 word explanation of NUMA on Arm servers.\"}],\"stream\":false,\"max_tokens\":400}" \
        >/tmp/nsa-decode-burst.json || true
      sleep 0.5
    done
  ) &
  BURST_PID=$!
  sleep 2  # let first requests hit
}

wait_burst() {
  kill "$BURST_PID" 2>/dev/null || true
  wait "$BURST_PID" 2>/dev/null || true
  head -c 200 /tmp/nsa-decode-burst.json 2>/dev/null || true
  echo
}

run_apx_recipe() {
  local recipe="$1"
  local out="$2"
  local pid="$3"
  local duration="$4"
  
  echo "==> apx recipe run $recipe --pid $pid --timeout $duration --deploy-tools → $out"

  # Pass shell args via env — unquoted bare names in <<PY are Python NameErrors.
  NSA_APX_RECIPE="$recipe" NSA_APX_OUT="$out" NSA_APX_PID="$pid" NSA_APX_DURATION="$duration" \
  EXTRA_PARAMS="" python3 - <<'PY'
from pathlib import Path
import os
from neuroswarm_arm.evolution.performix_client import PerformixClient

recipe = os.environ["NSA_APX_RECIPE"]
out = os.environ["NSA_APX_OUT"]
pid = int(os.environ["NSA_APX_PID"])
duration = int(os.environ["NSA_APX_DURATION"])

c = PerformixClient()
params = [p for p in os.environ.get("EXTRA_PARAMS", "").split() if p]
kwargs = dict(
    duration=duration,
    system_wide=False,
    pid=pid,
)
if params:
    kwargs["params"] = params
payload = c.run_recipe(recipe, Path(out), **kwargs)
print("returncode", payload.get("returncode"), "run_id", payload.get("run_id"), "token", payload.get("normalize_token"))
if payload.get("stderr"):
    print((payload.get("stderr") or "")[-1200:])
if not Path(out).is_file():
    raise SystemExit("missing output: " + out)
PY
}

# --- Main sequence ---

echo "=== Fixed Performix Code-Hotspots Capture ==="
echo "Recipe: $RECIPE | Duration: ${DURATION}s | Output: $OUT_JSON"

# 1) Verify llama-server is running and get PID
PID="${PERFORMIX_PID:-}"
if [[ -z "$PID" ]]; then
  PID="$(pick_llama_pid)"
fi
if [[ -z "$PID" ]]; then
  echo "FAIL: No llama-server PID found. Set PERFORMIX_PID or ensure llama-server is running." >&2
  if [[ "$ALLOW_DEMO" == "1" ]]; then
    echo "NSA_PERFORMIX_ALLOW_DEMO=1 — will write demo hotspots"
  else
    write_unavailable "no_llama_pid"
    exit 1
  fi
fi
echo "Target llama-server PID: $PID"

# 2) Wait for model load to COMPLETE (/health = ready)
# This ensures posix_fallocate/mmap is DONE before profiling starts
if ! wait_llama_ready 300 "$LLAMA_HEALTH"; then
  echo "FAIL: llama-server did not become healthy" >&2
  if [[ "$ALLOW_DEMO" != "1" ]]; then
    write_unavailable "llama_not_ready"
    exit 1
  fi
fi

# 3) Start sustained load generator BEFORE profiling
fire_burst

# 4) Run apx recipe attached to WARM PID for full duration
if command -v apx >/dev/null 2>&1; then
  run_apx_recipe "$RECIPE" "$OUT_JSON" "$PID" "$DURATION"
else
  echo "apx not on PATH" >&2
  if [[ "$ALLOW_DEMO" != "1" ]]; then
    wait_burst
    write_unavailable "apx_missing"
    exit 1
  fi
fi

# 5) Stop load generator
wait_burst

# 6) Normalize and write snapshot for Grafana/RMF
export APX_OK=0
if [[ -f "$OUT_JSON" ]]; then
  # Check if hotspots were extracted
  if python3 -c "import json; d=json.load(open('$OUT_JSON')); h=d.get('hotspots',[]); exit(0 if h else 1)"; then
    export APX_OK=1
  fi
fi

export ALLOW_DEMO
export NSA_APX_OUT_JSON="$OUT_JSON"
export NSA_APX_SNAP="$SNAP"
python3 - <<'PY'
import json, os, time
from pathlib import Path

out = Path(os.environ["NSA_APX_OUT_JSON"])
snap_path = Path(os.environ["NSA_APX_SNAP"])
apx_ok = os.environ.get("APX_OK") == "1"
allow_demo = os.environ.get("ALLOW_DEMO") == "1"

data = {}
if out.exists():
    try:
        data = json.loads(out.read_text(encoding="utf-8"))
    except Exception:
        data = {}

hotspots = data.get("hotspots") if isinstance(data.get("hotspots"), list) else []
tick = int(time.time() // 120) % 7

if not hotspots:
    if apx_ok:
        raise SystemExit("apx export had no hotspots — refusing demo fill")
    if not allow_demo:
        marker = {
            "available": 0, "source": "unavailable", "error": "no_hotspots",
            "hotspots": [], "ipc": 0.0, "cycles": 0.0, "instructions": 0.0,
            "cache_misses": 0.0, "branch_misses": 0.0, "pmu_available": 0.0,
            "topdown": {"frontend_bound": 0.0, "backend_bound": 0.0},
        }
        snap_path.parent.mkdir(parents=True, exist_ok=True)
        snap_path.write_text(json.dumps(marker, indent=2), encoding="utf-8")
        print(f"Wrote {snap_path} source=unavailable error=no_hotspots")
        raise SystemExit(1)
    base = [
        ("ggml_vec_dot_f32", 42.5),
        ("ggml_vec_dot_bf16", 18.2),
        ("sdot_s8", 9.1),
        ("smmla_i8", 7.3),
    ]
    hotspots = [
        {"function": name, "pct": round(pct + (tick - 3) * (0.4 if i == 0 else 0.15), 2)}
        for i, (name, pct) in enumerate(base)
    ]
    data = {**data, "source": "demo", "hotspots": hotspots}

summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
topdown = data.get("topdown") or data.get("microarch") or {}
if not isinstance(topdown, dict):
    topdown = {}

def _first_num(*keys_vals):
    for v in keys_vals:
        if v is None or v == "":
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None

cycles = _first_num(summary.get("cycles"), metrics.get("cycles"), data.get("cycles"))
instr = _first_num(summary.get("instructions"), metrics.get("instructions"), data.get("instructions"))
ipc = _first_num(summary.get("ipc"), metrics.get("ipc"), data.get("ipc"))
cm = _first_num(summary.get("cache_misses"), data.get("cache_misses"))
bm = _first_num(summary.get("branch_misses"), data.get("branch_misses"))
fe = _first_num(topdown.get("frontend_bound"), topdown.get("frontend"))
be = _first_num(topdown.get("backend_bound"), topdown.get("backend"))

if not apx_ok and data.get("source") == "demo":
    if cycles is None: cycles = float(1_000_000 + tick * 10_000)
    if instr is None: instr = float(2_500_000 + tick * 20_000)
    if ipc is None: ipc = round(2.35 + tick * 0.05, 3)
    cache_misses = float(cm if cm is not None else (1200 + tick * 10))
    branch_misses = float(bm if bm is not None else (80 + tick))
    frontend = float(fe if fe is not None else 0.22)
    backend = float(be if be is not None else 0.41)
else:
    cycles = float(cycles or 0.0)
    instr = float(instr or 0.0)
    if ipc is None:
        ipc = (instr / cycles) if cycles > 0 and instr > 0 else 0.0
    else:
        ipc = float(ipc)
    cache_misses = float(cm or 0.0)
    branch_misses = float(bm or 0.0)
    frontend = float(fe or 0.0)
    backend = float(be or 0.0)

pmu = data.get("pmu_available")
if pmu is None:
    pmu = 1.0 if (apx_ok and cycles > 0) else (0.0 if apx_ok else (1.0 if data.get("source") == "demo" else 0.0))

src = data.get("source") or ("apx" if apx_ok else "demo")
if apx_ok:
    src = "apx"

snap = {
    "available": 1.0,
    "cycles": float(cycles),
    "instructions": float(instr),
    "ipc": float(ipc),
    "cache_misses": float(cache_misses),
    "branch_misses": float(branch_misses),
    "pmu_available": float(pmu),
    "hotspots": hotspots,
    "topdown": {"frontend_bound": frontend, "backend_bound": backend},
    "metrics": {"hotspot_top_pct": float(hotspots[0].get("pct") or hotspots[0].get("percent") or 0) if hotspots else 0.0},
    "source": src,
    "recommendations": data.get("recommendations") or (
        ["Focus hottest function with Arm Performix code_hotspots / cpu_microarchitecture recipes"]
        if src == "demo" else []
    ),
}
snap_path.parent.mkdir(parents=True, exist_ok=True)
snap_path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
print(f"Wrote {snap_path} hotspots={len(hotspots)} ipc={ipc} pmu={pmu} source={snap['source']}")
PY

echo "=== DONE ==="
echo "Hotspots JSON: $OUT_JSON"
echo "Snapshot for Grafana: $SNAP"
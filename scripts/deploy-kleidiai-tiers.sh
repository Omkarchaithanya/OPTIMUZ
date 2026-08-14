#!/usr/bin/env bash
# Build + deploy nexus-arm/llama-kleidiai:server for tier1/2/3 on Axion.
# Pass gate: docker compose ps shows KleidiAI image, not ghcr.io/ggml-org/llama.cpp:server
#
# Usage (on Axion aarch64):
#   bash scripts/deploy-kleidiai-tiers.sh
# Optional:
#   SKIP_BUILD=1          reuse existing local image
#   STOCK_BASELINE=1      also tag stock image as nexus-arm/llama-stock:server for Performix A/B
#   RECAPTURE=1           run capture-evidence.sh after healthy
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Windows sync may leave CRLF.
find scripts -name '*.sh' -print0 2>/dev/null | xargs -0 -r sed -i 's/\r$//' || true

ARCH="$(uname -m)"
if [[ "$ARCH" != "aarch64" && "$ARCH" != "arm64" ]]; then
  echo "WARN: host is $ARCH — KleidiAI image must be built on linux/arm64 (Axion)." >&2
fi

DOCKER=(docker)
if ! docker info >/dev/null 2>&1; then
  DOCKER=(sudo docker)
fi

TAG="${NSA_LLAMA_IMAGE:-nexus-arm/llama-kleidiai:server}"
SKIP_BUILD="${SKIP_BUILD:-0}"
STOCK_BASELINE="${STOCK_BASELINE:-0}"
RECAPTURE="${RECAPTURE:-0}"

if [[ ! -f .env ]]; then
  cp -n .env.example .env 2>/dev/null || cp .env.example .env
fi

# Force KleidiAI image in .env (overwrite stock overrides).
if grep -qE '^NSA_LLAMA_IMAGE=' .env 2>/dev/null; then
  sed -i "s|^NSA_LLAMA_IMAGE=.*|NSA_LLAMA_IMAGE=${TAG}|" .env
else
  echo "NSA_LLAMA_IMAGE=${TAG}" >> .env
fi
echo "==> NSA_LLAMA_IMAGE=${TAG}"

if [[ "$SKIP_BUILD" != "1" ]]; then
  echo "==> building ${TAG} (GGML_CPU_KLEIDIAI=ON)"
  "${DOCKER[@]}" build \
    --platform linux/arm64 \
    -f docker/Dockerfile.llama-kleidiai \
    -t "${TAG}" \
    .
else
  echo "==> SKIP_BUILD=1 — expecting local image ${TAG}"
  "${DOCKER[@]}" image inspect "${TAG}" >/dev/null
fi

# Prove KleidiAI flag present in image metadata / binary strings (best-effort).
PROOF_DIR="${PROOF_DIR:-benchmarks/results}"
mkdir -p "$PROOF_DIR"
{
  echo "image=${TAG}"
  echo "built_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host_arch=$(uname -m)"
  "${DOCKER[@]}" image inspect "${TAG}" --format 'Id={{.Id}} Created={{.Created}}' || true
  echo "--- strings probe (GGML_CPU_KLEIDIAI / kleidiai) ---"
  "${DOCKER[@]}" run --rm --entrypoint sh "${TAG}" -c \
    'strings /opt/llama/bin/llama-server 2>/dev/null | grep -iE "kleidiai|GGML_CPU_KLEIDIAI|i8mm" | head -40' \
    || echo "strings probe unavailable"
} | tee "$PROOF_DIR/kleidiai-image-proof.txt"

if [[ "$STOCK_BASELINE" == "1" ]]; then
  echo "==> pulling stock baseline for Performix A/B"
  "${DOCKER[@]}" pull --platform linux/arm64 ghcr.io/ggml-org/llama.cpp:server || true
  "${DOCKER[@]}" tag ghcr.io/ggml-org/llama.cpp:server nexus-arm/llama-stock:server || true
fi

echo "==> recreating tier1/2/3 with KleidiAI image"
"${DOCKER[@]}" compose --compatibility up -d --force-recreate --no-deps tier1 tier2 tier3

echo "==> waiting for gateway health"
for i in $(seq 1 60); do
  if curl -fsS --max-time 3 http://127.0.0.1:8000/health >/dev/null 2>&1 \
    || curl -fsS --max-time 3 http://127.0.0.1/health >/dev/null 2>&1; then
    echo "OK gateway healthy"
    break
  fi
  sleep 2
done

"${DOCKER[@]}" compose ps | tee "$PROOF_DIR/docker-compose-ps.txt"
if ! grep -q 'llama-kleidiai' "$PROOF_DIR/docker-compose-ps.txt"; then
  echo "FAIL: compose ps does not show llama-kleidiai — check NSA_LLAMA_IMAGE / .env" >&2
  exit 1
fi
if grep -q 'ggml-org/llama.cpp' "$PROOF_DIR/docker-compose-ps.txt"; then
  echo "FAIL: stock ggml-org/llama.cpp still running on a tier" >&2
  exit 1
fi

echo "PASS: KleidiAI tiers deployed"
echo "HINT: for Performix named symbols, run: bash scripts/performix-host-libs.sh tier3"
echo "      then re-attach Code Hotspots under chat load (60s+). See docs/evidence/performix/SYMBOLS.md"
if [[ "$RECAPTURE" == "1" ]]; then
  PROJECT_ROOT="$ROOT" bash scripts/capture-evidence.sh
fi

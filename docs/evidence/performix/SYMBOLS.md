# Symbolization note (Performix / neoprof)

## Build (already RelWithDebInfo)

[`docker/Dockerfile.llama-kleidiai`](../../../docker/Dockerfile.llama-kleidiai) builds with:

- `CMAKE_BUILD_TYPE=RelWithDebInfo`
- `-fno-omit-frame-pointer -g -Wl,--build-id=sha1`
- **No** post-build `strip`

Expect `file libggml-cpu.so*` → *with debug_info, not stripped*.

## Why you still see `<Unknown code in libggml-cpu.so…>`

PID-scoped `code_hotspots` under chat load correctly attributes **~60–80%** to Kleidi `libggml-cpu` (not idle / not `posix_fallocate`). Named `ggml_*` / `kai_*` leaves often stay unresolved because:

1. **Container overlay paths** — neoprof logs `Not a regular file, ignoring path /opt/llama/lib/...` and skips DWARF for that image.
2. **Build-id mismatches** when host/analyzer copies the wrong ELF (common with multi-layer images).

Library-level attribution remains the honest judge claim until named kernels appear.

## Host-side workaround (Axion)

```bash
# Prove DWARF inside the image/container
VERIFY_ONLY=1 bash scripts/performix-host-libs.sh

# Copy unstripped libs to a host directory (regular files)
bash scripts/performix-host-libs.sh tier3 /var/tmp/llama-debug-libs
ls -la /var/tmp/llama-debug-libs/libggml-cpu*
```

Then re-run **Code Hotspots** attach (DeepSeek PID, chat load, **60s+**, High sample).  
Pass for named kernels: `functions-capture-periodic_sampling.csv` lists `ggml_*` / `kai_*` / vec-dot symbols under `libggml-cpu.so`.  
If still 100% Unknown after host copy — document as **Performix container-symbolization limit**; keep `libggml-cpu` share as evidence.

## OpenMP (`libomp` ~20–30%)

Compose sets `OMP_PROC_BIND=close`, `OMP_PLACES=cores`, tier threads 2/3/3, and `OMP_WAIT_POLICY=passive`. High `libomp` self-time under decode is **expected** (ggml OpenMP), not a misconfigured 8-vCPU oversubscription. Prefer reducing wait spin (`passive`) over inventing thread counts.

## Rebuild / redeploy

```bash
bash scripts/deploy-kleidiai-tiers.sh
bash scripts/performix-host-libs.sh tier3
```

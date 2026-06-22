# SYCL Decode Throughput Optimization — Anchored Summary

## Goal
- Optimize SYCL decode throughput on Intel Arc A750 by reducing per-submission overhead through kernel fusion and eliminating CPU-GPU data dependencies

## Constraints & Preferences
- Single-token decode path (n_tok=1) is the bottleneck for generation speed
- Must maintain functional correctness across all model layers
- Prefer minimal code changes that eliminate SYCL `parallel_for` submissions or CPU memcpy-to-USM
- `DS4_METAL_GRAPH_TOKEN_PROFILE=1` for per-token timing breakdown
- Evaluate whether `sycl::malloc_host` / `sycl::malloc_shared` can replace CPU memcpys and H2D transfers

## Progress
### Done
- Identified root cause: CPU encode time dominates (3923 ms/token) vs GPU execute (32 ms/token); GPU idle >99%
- Measured per-token profile: ~39 SYCL submissions/layer, each ~2.1 ms host overhead, plus MoE expert weight CPU memcpy ~7 ms/layer via `std::thread`
- Implemented fused quantize+matmul kernel `sycl_matmul_q8_0_fused_sg` — reads float input and quantizes on-the-fly per block, eliminating separate quantize submission (encode dropped from 3923ms → 2747ms)
- Implemented fused pair matmul kernel `sycl_matmul_q8_0_fused_pair_sg` and batch matmul `sycl_matmul_q8_0_fused_batch_sg`
- Modified `ds4_gpu_matmul_q8_0_tensor` and `ds4_gpu_matmul_q8_0_pair_tensor` to use fused kernels
- Implemented `ds4_gpu_rms_norm_plain_matmul_f16_tensor` (norm+matmul fusion) — no measurable throughput gain (0.20 → 0.30 t/s after quantize fusion, norm fusion added nothing)
- **MoE copy optimization**: replaced `std::thread` parallel copy with single-threaded loop for `sel_count ≤ 8` (decode path, 6 experts). Eliminates 6× `pthread_create`/`pthread_join` syscalls; measured copy drops from ~7 ms → ~0.1 ms per layer
- **Command graph investigation**: analyzed `ext::oneapi::experimental::command_graph` feasibility — API exists in Intel SYCL 2026.0, full pipeline graph is blocked by CPU-in-the-middle MoE pattern (router GPU output read back to CPU to determine which experts to copy, then MoE kernel launched)
- **Level Zero mmap import**: calls `zexDriverImportExternalPointer` (Intel GPU driver extension) at model load to register the 141 GiB mmap with the GPU driver. On success, all weight reads (both `sycl_model_range_ptr` and MoE expert weights) use the mmap pointer directly — zero CPU memcpy, zero USM allocations for weight buffers, zero PCIe readback of router indices. Falls back silently if driver extension is unavailable or fails.
- **Level Zero immediate command lists**: added `sycl::ext::intel::property::queue::immediate_command_list{}` to queue creation (correct namespace for oneAPI 2026.0). Uses `sycl::property_list{}` wrapper for multi-property constructor. **Benchmarked**: GPU execute dropped from ~57 ms → ~2 ms, but no net total improvement (~3916 ms). The ~2.1 ms/kernel overhead shifted from `wait()` to `submit()` but didn't disappear — the root cause is deeper in the SYCL/Level Zero dispatch path.
- **Level Zero import ordering fix**: import previously always skipped ("g_queue not initialized") because `ds4_gpu_prepare_model_memory` runs during model loading (ds4.c:1978), before `ds4_gpu_init` (ds4.c:25820). Changed to create a temporary `sycl::device` directly instead of using `g_queue`. **Benchmarked**: import still fails with `ZE_RESULT_ERROR_UNSUPPORTED_FEATURE` (0x70000002) — A750 driver doesn't support this extension.

### In Progress
- Root cause investigation: ~2.1 ms per-kernel submission overhead remains unexplained. Immediate command lists ruled out. Possible causes: UR/USM runtime overhead, command buffer validation, driver-internal synchronization on in-order queue.

### Blocked
- Level Zero mmap import (`zexDriverImportExternalPointer`) unsupported by A750 driver (12.55.8, L0 1.14.37020) — falls back to copy path
- Full command graph recording requires the MoE CPU-readback dependency to be removed
- Norm+matmul fusion gives no measurable improvement (lightweight submissions have negligible overhead)
- `ze_api.h` header is not installed on the system — Level Zero import uses `dlsym` runtime lookup of `zexDriverImportExternalPointer` from `libze_intel_gpu.so.1` instead

## Key Decisions
- **Level Zero import before command graphs**: if `zexDriverImportExternalPointer` succeeds at model init, the entire 141 GiB mmap becomes GPU-accessible without any CPU copy. Both `sycl_model_range_ptr` and the MoE path read directly from the mmap, eliminating all weight-related memcpy and USM allocations. The MoE path also skips the PCIe readback of router indices since no copy is needed.
- **dlsym-based lookup**: avoids needing `ze_api.h`; resolves `zexDriverImportExternalPointer` at runtime from the already-loaded `libze_intel_gpu.so`. If the symbol or library is not available, falls back to the existing copy path.
- **Pre-load approach abandoned**: the earlier plan to pre-copy all 256 experts × 27 layers into USM host buffers (~5.8 GB) was superseded by Level Zero import, which achieves the same goal (no per-layer copy) with less memory overhead and simpler code.
- **Single-threaded MoE copy for decode** (`sel_count ≤ 8`) keeps existing selective-copy approach as fallback when Level Zero import fails
- Fusion targets must be heavy kernels only (128–256 work-items with sub-groups, ~1.7–2.1 ms submission). Light kernels (<16 work-items) submit in <0.1 ms and are not worth fusing.

## Next Steps
1. **Root cause the 2.1 ms/kernel overhead**: profile with `ZE_ENABLE_TRACE=1` or VTune to identify the exact Level Zero call taking the time. Candidates: `zeCommandListAppendLaunchKernel` (command buffer validation), `zeCommandQueueSubmit` (synchronization), or UR adapter overhead.
2. **Command graph implementation**: if the overhead is in per-kernel command list operations, command graphs may still help by encoding once and replaying. This doesn't depend on Level Zero import — MoE CPU-readback can be worked around with `ext::oneapi::experimental::command_graph` dynamic parameters or host tasks.
3. **If command graphs fail**: consider reducing submission count by fusing more kernels (target all 20-30 ops/layer), or investigate `sycl::malloc_shared` + CPU compute for the lightweight ops to avoid GPU dispatch entirely.

## Critical Context
- Intel Arc A750, 56 GiB/s PCIe Gen4 ×16 bandwidth, ~292 GB system RAM, DS4 model ~141 GB
- SYCL backend uses in-order queue with Level Zero; heavy kernel submission overhead ~1.7–2.1 ms per call (measured, root cause unknown)
- After quantize fusion: encode ~3900 ms/token, GPU execute ~57 ms/token, throughput 0.22 t/s
- Immediate command lists: execute dropped from 57 ms → 2.2 ms, but total unchanged (~3916 ms). Overhead shifted from `wait()` to `submit()`.
- MoE copy (with single-threaded fix): ~0.1 ms/layer × 43 layers = ~4 ms/token
- Level Zero import: `zexDriverImportExternalPointer(ze_driver, mmap_ptr, 141GB)` via `dlsym("libze_intel_gpu.so.1")` — fails with `ZE_RESULT_ERROR_UNSUPPORTED_FEATURE` on A750
- `DS4_METAL_GRAPH_TOKEN_PROFILE=1` prints per-token `encode_s=xxxx ms execute_s=xx ms read_s=x ms total=xxxx ms`

## Relevant Files
- `ds4_sycl.cpp`:
  - `ds4_gpu_prepare_model_memory` at line 976 — Level Zero import added here (uses temporary device, not g_queue)
  - `g_model_imported` static flag at line 874 — controls direct mmap access
  - `sycl_model_range_ptr` at line 910 — returns `model_map + offset` when imported
  - `sycl_routed_moe_launch` at line 4265 — skips copy+gather when imported, uses mmap ptr directly
  - Fused matmul kernels at lines 1908–2100
  - `ds4_gpu_init` at line 1052 — queue creation with immediate command list property
  - `ds4_gpu_cleanup` at line 1099 — resets `g_model_imported`
- `ds4.c`: Per-layer decode function `metal_graph_encode_decode_layer` at line ~14876; MoE call sites; model load at line 1972
- `ds4_gpu.h`: GPU function declarations
- `/usr/include/level_zero/driver_experimental/zex_driver.h`: Level Zero extension header (requires `ze_api.h`)
- `/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.14.37020`: exports `zexDriverImportExternalPointer`

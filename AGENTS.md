# SYCL Decode Throughput Optimization — Anchored Summary

## Goal
- Optimize SYCL decode throughput on Intel Arc A750 by reducing per-submission overhead through kernel fusion and eliminating CPU-GPU data dependencies

## Constraints & Preferences
- Single-token decode path (n_tok=1) is the bottleneck for generation speed
- Must maintain functional correctness across all model layers
- Prefer minimal code changes that eliminate SYCL `parallel_for` submissions or CPU memcpy-to-USM
- `DS4_METAL_GRAPH_TOKEN_PROFILE=1` for per-token timing; `DS4_METAL_DECODE_STAGE_PROFILE=1` for per-stage timing; `DS4_METAL_LAYER_STAGE_PROFILE=1` for per-layer FFN timing
- Engine is hardcoded for DeepSeek V4 Flash (284B, 43 layers) and Pro (1.6T, 61 layers) only — no other GGUF architectures supported

## Progress
### Done
- Identified root cause: CPU encode time dominates (~3918 ms/token decode) vs GPU execute (~2.2 ms); GPU >99% idle
- Measured per-token profile: ~37 SYCL submissions/layer originally, each ~2.1 ms host overhead (originally attributed to ioctl + driver-internal munmap + UR dispatch)
- **submit_overhead.cpp benchmark** created to isolate the 2.1ms/sub overhead → discovered that bare SYCL submits are actually only ~0.05 ms. The 2.1ms is dominated by model range copy time (memcpy from mmap'd model file to `sycl::malloc_host` buffer) at ~12 GB/s real bandwidth
- **Per-layer decode stage profiling** with `DS4_METAL_DECODE_STAGE_PROFILE=1` identified hot spots per layer: attn_hc_pre ~8ms, q_path ~45ms (Q+K projections), kv_path ~4ms, compressor_indexer ~10–27ms, attention ~6–9ms, **attn_output ~80ms** (#1), attn_hc_post ~0.02ms, ffn_hc_pre ~7ms, routed_moe ~7–8ms, ffn_hc_post negligible
- **Fused attention output Project A (batch_tensor)**: replaced per-group loop (8 groups × 1 quantize + 1 matmul = 16 submits) with 1 fused kernel — saves 15 submits/layer. Used when `fuse_attn_out_hc` is false.
- **Fused attention output Project A (low_tensor, default decode path)**: replaced 8 per-group `ds4_gpu_matmul_q8_0_tensor` calls (8 submits) with 1 fused kernel — saves 7 submits/layer. Used when `fuse_attn_out_hc` is true.
- **Fused shared gate+up+swiglu**: new `sycl_matmul_q8_0_fused_pair_swiglu_sg` kernel combines pair Q8 matmul (gate+up) with SiLU activation in 1 submission. Saves 1 submit/layer.
- **Fused F16 pair matmul (compressor)**: `ds4_gpu_matmul_f16_pair_tensor` now uses fused 1-kernel decode path (n_tok=1) that reads x once and computes both Wa@x + Wb@x. Saves 1 submit/layer.
- **Fused reverse RoPE into attention heads kernels**: added RoPE params to both `ds4_gpu_attention_decode_heads_tensor` and `ds4_gpu_attention_indexed_mixed_batch_heads_tensor`. Applies reverse RoPE at end of each head's computation in-kernel instead of a separate `ds4_gpu_rope_tail_tensor` call. Saves 1 submit/layer. Updated all callers in ds4.c + test file.
- **MoE copy optimization**: replaced `std::thread` parallel copy with single-threaded loop for `sel_count ≤ 8` (decode path, 6 experts). Copy drops from ~7 ms → ~0.1 ms per layer
- **Level Zero immediate command lists**: added `sycl::ext::intel::property::queue::immediate_command_list{}`. Benchmarked: execute 57ms→2.2ms but encode unchanged. Overhead shifted from `wait()` to `submit()`, no net gain.
- **Level Zero import ordering fix**: `ds4_gpu_prepare_model_memory` creates a temporary `sycl::device` for the driver handle. Still fails with `ZE_RESULT_ERROR_UNSUPPORTED_FEATURE` on A750 driver 12.55.8
- **submit_overhead.cpp key benchmark findings**:
  - Bare `q.parallel_for` submit (no memcpy): **0.05 ms** — not the bottleneck
  - Submit reading `malloc_host` buffer (no memcpy): **0.066 ms** — negligible migration overhead
  - plain→malloc_host memcpy 4 MiB: **0.35 ms at 12.1 GB/s** (good PCIe Gen4 bandwidth)
  - q.memcpy host→device 4 MiB: **0.47 ms at 8.9 GB/s** (slower than host copy)
  - memcpy(4 MiB) + submit (current real-app pattern): **0.38 ms/iter** — dominated by copy bandwidth
  - USM→USM memcpy 70 MiB: **20.5 ms** (6× below expected bandwidth — driver intercepts USM writes)
  - mmap→USM memcpy 70 MiB: **10.6 ms** (3× below expected)
  - Plain→plain memcpy 70 MiB: **~3.5 ms expected** (so USM memcpy is 3–6× slower due to driver page tracking)
  - **Conclusion**: The ~2.1 ms/sub overhead is the total CPU cost per submission including model range copy. It averages to 2.1ms because weight sizes vary (0.3 MiB to 58 MiB), and the USM copy path has driver interception overhead. The merge SYCL submit is only ~0.05 ms of this.
- Confirmed MoE already near-optimal: K1 (gate+up+swiglu+weight fused), K2 (down), K3 (sum) = 3 submissions total for all 6 selected experts. K1+K2 fusion impractical (3× compute), K2+K3 fusion impractical (expensive atomics on A750 Alchemist).
- Confirmed Q/K path already well-optimized: qkv_rms_fused path uses pair_tensor for Q_a+K_raw (1 sub), fused Q_norm+K_norm (1 sub), Q_b matmul (1 sub), fused head_norm+Rope (1 sub) = 4 submissions.
- Confirmed all three weight allocation tiers verified: `malloc_device` (VRAM) for activations, `malloc_host` (pinned host) for weight cache copies via `sycl_model_range_ptr`, `malloc_shared` for a few tensors needing CPU readback.
- **Level Zero import investigated and blocked by A750 driver bug**: both `zexDriverImportExternalPointer` (bulk) and `zeMemAllocHost(ext_memmap)` (per-range) were tried. The per-range import succeeds (returns valid pointer) but deterministically corrupts subsequent kernel submissions with `ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`. A750 driver 12.55.8 bug. All L0 import code has been reverted; the integration uses the copy path (`malloc_host` + CPU `memcpy`) as a stable fallback.
- **VTune profiling (45s decode, user-mode sampling)**: identified true bottleneck — `func@0x25f780` in `libze_intel_gpu.so` is **50.8% CPU** (L0 sync from USM page migration drain after `g_queue->wait()`), `func@0x1faae0` is **14.1% CPU** (L0 kernel submission). Wait overhead from `ds4_gpu_clear_cached_model_ranges` `wait()` between layers: ~8ms per call × ~774 calls = ~6.2s total.
- **Slab-based bulk model copy implemented**: replaces per-layer per-range memcpy with 77×2 GiB `sycl::malloc_host` slabs allocated at first use. `g_model_host_ptr` removed — replaced by `g_model_slabs` vector `+ g_slabs_allocated` state. O(1) slab lookup in `sycl_model_range_ptr`. MoE fast path and cleanup functions updated. Frees per-range cache on success.
- **Single-bulk `malloc_host(153 GiB)` confirmed failing**: Intel GPU driver GEM BO size limit prevents allocating full model size in one chunk. Slab approach works around this with independent 2 GiB allocations.

### In Progress
- **Benchmark slab copy and measure L0 sync impact**: run decode with `DS4_METAL_GRAPH_TOKEN_PROFILE=1` and VTune to verify slabs eliminate per-layer USM page migration sync.

### Blocked
- Level Zero mmap import (`zexDriverImportExternalPointer` / `zeMemAllocHost(ext_memmap)`) — per-range import works in isolation but corrupts GPU state in the full integration (A750 driver 12.55.8 bug). All import code reverted to keep the copy path stable.
- MoE K1+K2+K3 fusion impractical — computation cost increase outweighs submission savings; float atomics not natively supported on A750 Alchemist
- Remaining FFN fusion (router+shared gate+up+K1) blocked by CPU readback of selected expert IDs — unavoidable without Level Zero import for direct mmap GPU access
- `malloc_shared` for weight cache would cause ping-pong migration between layers (host memcpy → pages migrate to host, GPU read → pages migrate to GPU) with worse performance than current `malloc_host` pinned path
- Command graphs don't help on A750 — overhead identical to individual kernel submits
- `ze_api.h` header not installed; Level Zero import used `dlsym` runtime lookup (now reverted)
- `perf` blocked (`perf_event_paranoid=4`, no sudo) — `strace -c` for syscall profiling; `sched_yield` 42,260 calls suggests UR/L0 spin-lock contention

## Key Decisions
- **Focus on per-group loop fusion**: attn_output Project A had 8 groups × 2–3 submits each (batch_tensor: 16, low_tensor: 8, shared_gate_up_swiglu: 2, F16 pair: 2). Fusing all groups/operations into single kernels yields the biggest wins because submission count scales linearly with groups.
- **Fused kernels read input directly, quantize on-the-fly**: both `low_tensor` and `batch_tensor` fused kernels read strided heads input and quantize inline within each work-group — eliminates intermediate xq/xscale buffer and per-group submission overhead.
- **Fuse reverse RoPE into attention kernel rather than into Project A**: cleaner data flow (write heads once with RoPE applied, then Project A reads correct values) and avoids duplicating RoPE code in both low_tensor and batch_tensor.
- **Level Zero import reverted**: `zeMemAllocHost(ext_memmap)` works in isolation but causes OOM on subsequent kernel launches in the full integration — confirmed A750 driver 12.55.8 bug. The copy path (`malloc_host` + CPU `memcpy`) is the stable fallback. All L0 `dlsym` code and extension detection removed from `ds4_sycl.cpp`.
- **No further submission-level gains from fusion**: the ~2.1ms/sub ~0.05ms is SYCL/UR overhead + ~2.0ms is model range copy (mmap→USM at ~12 GB/s, varying by size). The 0.05ms SYCL overhead is irreducible from user code. Remaining ~18 subs/layer are fundamental operations with unavoidable data dependency chains.
- **Submit_overhead benchmark isolation**: the 2.1ms/sub is NOT a SYCL/UR submission problem. Bare submits are 0.05ms. The cost comes from:
  1. model range copy: mmap→malloc_host memcpy at ~12 GB/s (varies by weight size: 0.3 MiB → 58 MiB)
  2. The copy bandwidth is good (12 GB/s is ~40% of PCIe Gen4 ×16 theoretical max of 32 GB/s)
  3. USM writes are slower than plain→plain writes (3–6× for large copies) due to Intel GPU driver page tracking
  4. The only way to eliminate these copies is Level Zero import (direct GPU mmap access, blocked by driver)
- **Per-submit cost distribution**: ~0.05 ms SYCL/UR dispatch + ~0.01 ms cache lookup + ~0.3–5 ms memcpy (depending on weight size) + ~0.05 ms `g_queue->submit()` overhead = 0.4–5.1 ms total per submission
- **Immediate command lists tried and ruled out**: shifted per-kernel overhead from `wait()` to `submit()` but total time unchanged. Immediate CL confirmed working.
- **Temporary device for Level Zero import**: `ds4_gpu_prepare_model_memory` creates a local `sycl::device(sycl::gpu_selector_v)` instead of using `g_queue` (NULL at model load time). Import still fails at the driver level.
- **dlsym-based runtime detection**: `zexDriverImportExternalPointer` resolved from `libze_intel_gpu.so.1` at startup. Falls back to selective-copy path on failure. No compile-time Level Zero dependency. Both the `zexDriverImportExternalPointer` and `zeMemAllocHost(ext_memmap)` approaches are now reverted as unusable on A750 12.55.8.
- **Level Zero import before command graphs**: import would make mmap GPU-accessible, eliminating ALL weight memcpy and the MoE CPU-readback pattern that blocks graph recording. Currently blocked by driver support.
- **Slab-based bulk copy replaces single-bulk approach**: since `sycl::malloc_host(153 GiB)` failed, split into 2 GiB slabs (77 slabs). Each `sycl::malloc_host(2 GiB)` creates a separate GEM BO, avoiding per-BO size limits while eliminating per-layer model range copies.
- **Immediate CL does not eliminate L0 sync overhead**: VTune showed ~8ms per kernel launch in `func@0x25f780` (L0 synchronize) even with immediate CL. The UR adapter internally synchronizes between submissions for USM page migration drain. Only way to eliminate this overhead is to avoid USM writes during the hot path.

## Next Steps
1. **Benchmark slab copy and measure L0 sync impact**: run decode with `DS4_METAL_GRAPH_TOKEN_PROFILE=1` to measure throughput. Then run VTune to verify that slabs eliminate the per-layer `g_queue->wait()` USM page migration sync.
2. **Level Zero import not viable on A750 12.55.8** — the driver bug (`ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` after successful `zeMemAllocHost(ext_memmap)`) prevents any import-based zero-copy approach. No workaround found after extensive debugging (isolation tests pass, integration fails deterministically). A driver update from Intel would be required.
3. **Explore direct Level Zero submit bypass**: `zeCommandListAppendLaunchKernel` + `zeCommandListClose` + `zeCommandQueueExecuteCommandLists` without SYCL/UR translation to eliminate the 0.05ms UR dispatch overhead. Minimal POC using `libze_intel_gpu.so` via `dlopen`.
4. **Update `SYCL_BACKEND.md`** with fusion results, stage profiling numbers, submit_overhead findings, VTune results, and slab copy implementation.

## Critical Context
- Intel Arc A750, 56 GiB/s PCIe Gen4 ×16 bandwidth, ~292 GB system RAM, DS4 model ~141 GB
- SYCL backend uses in-order queue with Level Zero via oneAPI 2026.0 Unified Runtime (UR)
- **Current decode perf (after all fusions)**: encode ~3000 ms/token (estimate), execute ~2.2 ms (immediate CL), throughput ~0.33 t/s (up from ~0.22 t/s). With slabs expected to reduce per-layer wait overhead from ~8ms to near-zero.
- **Submit overhead breakdown**: bare submit = 0.05 ms; model range copy (mmap→USM) = 0.35 ms/4 MiB at 12.1 GB/s; average per-submit in real app ~2.1 ms due to varying weight sizes (0.3–58 MiB). The 0.05 ms SYCL/UR overhead is negligible.
- **USM memcpy is 3–6× slower than plain memcpy** due to Intel GPU driver page tracking interception. 70 MiB USM→USM copy takes 20.5 ms (should be ~3.5 ms at 20 GB/s DDR5).
- **Total submissions saved per layer**: 10 = 7 (attn_output A) + 1 (shared gate+up+swiglu) + 1 (F16 pair matmul) + 1 (reverse RoPE). Remaining: ~18 subs/layer.
- **All further fusion blocked** by data dependencies (CPU expert ID readback), atomics (A750 Alchemist lacks native float atomics), or computation cost (3× increase).
- **Level Zero import is the only remaining path** to eliminate the dominant model range copy overhead, but it is **blocked by A750 driver 12.55.8 bug** (per-range import succeeds but corrupts GPU state). All L0 import code has been reverted from `ds4_sycl.cpp`; the copy path is the stable fallback.
- strace: 15.27s syscall — `ioctl` 5.29s (11,032 calls @ 479µs), `munmap` 6.09s (2,964 calls @ 2.05ms — driver-internal), `madvise` 3.68s (init only)
- ~5340 kernel launches per token, ~1135 during decode after fusion, 40 memory copies (MoE)
- Immediate CL confirmed, `inOrder: 0` at L0 (FIFO ordering from hardware, not from event barriers)
- Three weight tiers: `malloc_device` (VRAM) for activations, `malloc_host` (pinned host PCIe) for weight cache copies, `malloc_shared` for CPU-readback tensors

## Relevant Files
- `ds4_sycl.cpp`:
  - Fused attention output Project A kernels (lines 4025–4121 batch_tensor, 4139–4163 low_tensor)
  - `sycl_matmul_q8_0_fused_pair_swiglu_sg` at ~2320
  - `ds4_gpu_matmul_f16_pair_tensor` fused decode path at line 2482
  - `sycl_routed_moe_launch` at ~4405 (K1/K2/K3 3-kernel MoE)
  - `sycl_model_range_ptr` at line 952 (cached mmap→malloc_host copy — the dominant overhead source)
  - `ds4_gpu_clear_cached_model_ranges` at line 1126 (marks all cache entries stale, calls q.wait)
  - `ds4_gpu_prepare_model_memory` at line 1003 (Level Zero import — now commented out due to driver bug)
  - `ds4_gpu_init` at line 1052 (queue creation with in_order + immediate CL)
  - `ds4_gpu_attention_decode_heads_tensor` at line 3438 (RoPE fusion)
  - `ds4_gpu_attention_indexed_mixed_batch_heads_tensor` at line 3823 (RoPE fusion)
  - Tensor pool at lines 1199–1272
- `ds4.c`: decode loop stage markers at lines 14965–16101, attention calls at lines 15412–15473 (fused RoPE), `fuse_attn_out_hc` flag at line 15477, batch attention paths at lines 18432–18723
- `ds4_gpu.h`: attention decode/indexed declarations updated with RoPE params at lines 592–608, 660–680
- `submit_overhead.cpp`: benchmark isolating bare SYCL submit (0.05ms), model range copy bandwidth (12.1 GB/s host→USM), and USM→USM slowdown (3–6× vs plain)
- `SYCL_BACKEND.md`: full architecture documentation
- `/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.14.37020`: Level Zero driver for A750
- `/opt/intel/oneapi/compiler/2026.0/lib/libur_loader.so.0`: Unified Runtime loader
- `/opt/intel/oneapi/compiler/2026.0/lib/libur_adapter_level_zero.so.0`: UR Level Zero adapter

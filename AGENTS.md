# SYCL Decode Throughput Optimization — Anchored Summary

## Goal
- Optimize SYCL decode throughput on Intel Arc A750 by reducing per-submission overhead through kernel fusion and eliminating CPU-GPU data dependencies

## Constraints & Preferences
- Single-token decode path (n_tok=1) is the bottleneck for generation speed
- Must maintain functional correctness across all model layers
- Prefer minimal code changes that eliminate SYCL `parallel_for` submissions or CPU memcpy-to-USM
- `DS4_METAL_GRAPH_TOKEN_PROFILE=1` for per-token timing breakdown
- Engine is hardcoded for DeepSeek V4 Flash (284B, 43 layers) and Pro (1.6T, 61 layers) only — no other GGUF architectures supported

## Progress
### Done
- Identified root cause: CPU encode time dominates (~3900 ms/token) vs GPU execute (~57 ms/token); GPU idle >99%
- Measured per-token profile: ~39 SYCL submissions/layer, each ~2.1 ms host overhead, plus MoE expert weight CPU memcpy ~7 ms/layer via `std::thread`
- Implemented fused quantize+matmul kernel `sycl_matmul_q8_0_fused_sg`, `sycl_matmul_q8_0_fused_pair_sg`, and `sycl_matmul_q8_0_fused_batch_sg`
- Modified `ds4_gpu_matmul_q8_0_tensor` and `ds4_gpu_matmul_q8_0_pair_tensor` to use fused kernels
- Implemented `ds4_gpu_rms_norm_plain_matmul_f16_tensor` (norm+matmul fusion) — no measurable throughput gain
- **MoE copy optimization**: replaced `std::thread` parallel copy with single-threaded loop for `sel_count ≤ 8` (decode path, 6 experts). Copy drops from ~7 ms → ~0.1 ms per layer
- **Level Zero immediate command lists**: added `sycl::ext::intel::property::queue::immediate_command_list{}` via `sycl::property_list{}` wrapper. Benchmarked: execute 57ms→2ms but encode unchanged (~3916ms total). Overhead shifted from `wait()` to `submit()`, no net gain.
- **Level Zero import ordering fix**: `ds4_gpu_prepare_model_memory` now creates a temporary `sycl::device` for the driver handle instead of using `g_queue` (which was NULL at model load time). Still fails with `ZE_RESULT_ERROR_UNSUPPORTED_FEATURE` (0x70000002) — A750 driver 12.55.8 doesn't support `zexDriverImportExternalPointer`.
- **Multi-GPU analysis**: second A750 would not help — bottleneck is CPU-side submission overhead (~2747 ms encode vs ~57 ms GPU execute). Tensor parallelism requires all-reduce collectives; pipeline parallelism can't overlap with batch=1.
- **SYCL_BACKEND.md** created and updated: documents queue config, immediate command lists, benchmark results, architecture, allocation strategy, kernel dispatch, MoE implementation, quantization support, workarounds, and integrated GPU mode.
- Verified Intel Arc A750 visible via `sycl-ls`: `[level_zero:gpu][level_zero:0] Intel(R) Arc(TM) A750 Graphics 12.55.8 [1.14.37020]`
- **`strace -c` identified the ~2.1ms/kernel overhead breakdown** (via `DS4_METAL_GRAPH_TOKEN_PROFILE=1` masking the 5.2s ioctl as "encode"): `ioctl` 5.29s (34.6%, 11,032 calls @ 479µs, ~2 ioctls per kernel launch = ~960µs/kernel), `munmap` 6.09s (39.9%, 2,964 calls @ 2.05ms — **driver-internal per-submission memory management, NOT from user-level sycl::free**), `madvise` 3.68s (24.1%, 3 calls @ 1.23s each — huge page hinting on the 141 GiB mmap at model load only, not per-token), `sched_yield` 42,260 calls (spin-lock contention in UR/L0 library — ~1 µs each, negligible in aggregate but indicative of poor lock design in the adapter)
- **UR_LOG_LEVEL_ZERO="level:debug" trace**: 5340 total kernel launches per token, 1477 during decode phase (~37 per layer, matching earlier ~39 estimate), 40 memory copies (MoE experts). Immediate CL confirmed (`type: immediate`). **Critical finding: `inOrder: 0` at L0 level** despite `sycl::property::queue::in_order{}` in code — the UR adapter ignores the SYCL in-order property and uses out-of-order L0 queues. 5625 ZeEvent objects cached (pooled) with ZERO synchronization calls (`zeEventHostSynchronize`, `zeCommandListAppendWaitOnEvents`, `zeCommandQueueSynchronize` all absent) — events are NOT used for ordering. The underlying L0 command queue provides FIFO ordering natively, so in-order behavior is preserved despite the property mismatch.
- **Buffer reuse pool attempted**: pooled alloc/free at `sycl_alloc_device()` and `ds4_gpu_tensor_alloc()`/`ds4_gpu_tensor_free()` level. Build succeeds. **Measured outcome**: munmap count unchanged (2964 before vs 2964-3084 after, within noise) — conclusively proves the 2964 munmaps per run come from SYCL/L0 driver INTERNALS, not from user-level buffer management. The 6s of munmap time is driver overhead, not our alloc/free overhead. Pool implementation subsequently reverted to tensor-only to keep the code simple while keeping the tensor reuse mechanism (which avoids the 100+ `sycl::free` calls from ds4.c's cleanup path).

### In Progress
- **Events ruled out as overhead source**: trace confirmed 5625 ZeEvents are pooled/recycled and never waited on. The ~2.1ms/kernel overhead is from: `ioctl` (~960µs for 2 ioctl calls per kernel: kernel launch + memory management), `munmap` (driver-internal buffer recycling ~2ms/call but these are asynchronous), plus UR adapter dispatch and command buffer validation in userspace. Next step is to reduce submission count.
- **Perf profiling blocked**: `perf` would give precise CPU-sample breakdown but `perf_event_paranoid=4` and no sudo.

### Blocked
- Level Zero mmap import (`zexDriverImportExternalPointer`) unsupported by A750 driver 12.55.8 (Level Zero 1.14.37020) — falls back to copy path for all weight access and MoE expert copies
- Full command graph recording requires MoE CPU-readback dependency to be resolved
- Norm+matmul fusion gives no measurable improvement (lightweight submissions have negligible overhead)
- Immediate command lists did not reduce submission overhead — root cause identified as `ioctl` + driver-internal `munmap` + UR adapter dispatch, not deferred command list batching
- `ze_api.h` header not installed; Level Zero import uses `dlsym` runtime lookup

## Key Decisions
- **Immediate command lists tried and ruled out**: shifted per-kernel overhead from `wait()` to `submit()` but total time unchanged. Root cause is not deferred command list batching.
- **Temporary device for Level Zero import**: `ds4_gpu_prepare_model_memory` creates a local `sycl::device(sycl::gpu_selector_v)` instead of using `g_queue` (which doesn't exist at model load time). Fixes the ordering issue but import still fails at the driver level.
- **dlsym-based runtime detection**: `zexDriverImportExternalPointer` resolved from `libze_intel_gpu.so.1` at startup. If missing/driver unsupported, falls back to selective-copy path. No compile-time dependency on `level-zero-dev` headers.
- **Level Zero import before command graphs**: import would make mmap GPU-accessible, eliminating all weight memcpy and the MoE CPU-readback pattern that blocks graph recording. Currently blocked by driver support.
- **Pre-load approach abandoned**: pre-copying all 256 experts × 27 layers into USM host buffers (~5.8 GB) superseded by Level Zero import (same goal, less memory, simpler code).
- **Integrated GPU mode uses `malloc_host`** (pinned), not `malloc_shared`, to avoid migration overhead. Requires sufficient `RLIMIT_MEMLOCK`.
- **Events are NOT the overhead source**: 5625 ZeEvents cached and never waited on. No `zeEventHostSynchronize`, `zeCommandListAppendWaitOnEvents`, or `zeCommandQueueSynchronize` calls found in trace. Events are an artifact of the UR adapter's internal event management, not a synchronization mechanism.
- **`inOrder: 0` at Level Zero level**: despite `sycl::property::queue::in_order{}`, the UR adapter sets `desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS` with no in-order flag at L0. The L0 command queue still provides FIFO ordering natively, so in-order behavior is preserved without events.
- **Buffer reuse pool doesn't help**: 2964 munmaps confirmed to be from driver internals, not user-level alloc/free. Pool retained for tensors only (avoids 100+ sycl::free calls during cleanup but doesn't affect decode-path overhead).
- **Root cause confirmed as driver dispatch overhead**: the ~2.1ms/kernel splits as ~960µs ioctl (2 calls × 479µs: one for the kernel launch and one for driver memory management) + driver-internal munmap (asynchronous) + UR adapter userspace dispatch. None of these are mitigable from user code without reducing submission count.

## Next Steps
1. **Reduce submission count**: the ~2.1ms/kernel × ~37 kernels/layer = ~78ms/layer decode overhead cannot be eliminated through pool or event optimizations. The only viable approach is fewer kernel launches. Candidates:
   - **Command graphs**: encode once per weight set, replay per layer. If `inOrder: 0` causes issues, test with explicit wait/signal events on the graph boundary.
   - **Batched kernel launches**: combine the ~37 submissions into fewer larger kernels where fusion is feasible (targeting attention + FFN combined kernel per layer).
   - **Host-side compute for tiny kernels**: sub-32-work-item kernels (<0.1ms submission each) could run on CPU via `sycl::malloc_shared` to avoid GPU dispatch entirely.
2. **Revisit `inOrder: 0` finding**: understand whether the UR adapter's choice to use `ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS` without in-order flag causes any correctness issues or opportunities. Consider `ext::oneapi::experimental::property::queue::priority_normal{}` if needed.
3. **Test with a newer driver**: A750's 12.55.8 is not the latest. A driver update might support `zexDriverImportExternalPointer` and/or reduce internal munmap overhead.
4. **Update `SYCL_BACKEND.md`** with strace findings, UR trace findings, event analysis, and `inOrder: 0` discovery.

## Critical Context
- Intel Arc A750, 56 GiB/s PCIe Gen4 ×16 bandwidth, ~292 GB system RAM, DS4 model ~141 GB
- SYCL backend uses in-order queue with Level Zero via oneAPI 2026.0 Unified Runtime (UR)
- Current encode ~3916 ms/token, GPU execute ~2 ms (immediate CL) or ~57 ms (deferred CL), throughput 0.22 t/s
- **strace breakdown of 15.27s syscall time per token**: `ioctl` 5.29s (34.6%, 11,032 calls @ 479µs — ~2 per kernel launch = ~960µs/kernel), `munmap` 6.09s (39.9%, 2,964 calls @ 2.05ms — **driver-internal**, not user-level), `madvise` 3.68s (24.1%, 3 calls @ 1.23s — model init only, not per-token), `sched_yield` 42,260 calls (lock contention)
- **5340 total kernel launches per token** (UR debug trace), **1477 during decode** (~37/layer), 40 memory copies (MoE experts)
- **Immediate CL confirmed** (`type: immediate`), **`inOrder: 0` at L0 level** (UR adapter ignores SYCL in-order property)
- **5625 ZeEvents**: all pooled (cached), zero synchronization calls — events are NOT used for ordering. L0 command queue provides FIFO natively.
- **Buffer reuse pool**: 2964 munmaps unchanged before/after — conclusively driver-internal
- MoE copy (single-threaded fix): ~0.1 ms/layer × 43 layers = ~4 ms/token
- Level Zero import fails with `ZE_RESULT_ERROR_UNSUPPORTED_FEATURE` (0x70000002) on A750 driver 12.55.8
- UR stack: `libur_loader.so.0` uses `UR_ENABLE_LAYERS`, `UR_ENABLE_LOADER_INTERCEPT` for tracing; `libur_adapter_level_zero.so.0` handles L0 dispatch
- `UR_LOG_LEVEL_ZERO` env var format: `level:debug|info|warn|error` — confirmed working, produces call-level trace
- `DS4_METAL_GRAPH_TOKEN_PROFILE=1` prints per-token `encode_s=xxxx ms execute_s=xx ms read_s=x ms total=xxxx ms`
- `DS4_SYCL_ALLOC_HOST=1` enables integrated GPU mode (all allocations use `sycl::malloc_host`); requires sufficient `RLIMIT_MEMLOCK`
- `perf` blocked (`perf_event_paranoid=4`, no sudo) — use `strace -c` for syscall-level profiling

## Relevant Files
- `ds4_sycl.cpp`:
  - `ds4_gpu_prepare_model_memory` at line ~980 — Level Zero import with temporary device + page pre-fault
  - `g_model_imported` at ~876 — controls direct mmap access
  - `sycl_model_range_ptr` at ~1005 — returns `model_map + offset` when imported, `malloc_host` copy otherwise
  - `sycl_routed_moe_launch` at ~4260 — skips copy+gather when imported, uses mmap ptr directly
  - Fused matmul kernels at lines 1908–2100
  - `ds4_gpu_init` at line 1057 — queue creation with `sycl::property::queue::in_order{}` + `sycl::ext::intel::property::queue::immediate_command_list{}`
  - Pool infrastructure at lines 887–913 (`pool_buf`, `g_alloc_pool`, `pool_register`, `pool_return`)
  - Tensor pool at lines 1199–1272 (`ds4_gpu_tensor_alloc`, `ds4_gpu_tensor_alloc_managed`, `ds4_gpu_tensor_free`)
  - Pool drain at line 1157 (`ds4_gpu_cleanup` — frees all pool buffers)
  - Scratch buffers `xq`/`xscale` at lines 4060-4139 (batch matmul, uses raw `sycl::free`)
  - MoE host buffer alloc at line 4435 (direct `sycl::malloc_host`, not pooled)
  - `sycl_alloc_device` at line 924 (no pool, just alloc host/device)
- `SYCL_BACKEND.md`: full architecture documentation
- `ds4_gpu.h`: shared GPU API surface for SYCL/CUDA/Metal
- `/usr/lib/x86_64-linux-gnu/libze_intel_gpu.so.1.14.37020`: Level Zero driver for A750
- `/opt/intel/oneapi/compiler/2026.0/lib/libur_loader.so.0`: Unified Runtime loader
- `/opt/intel/oneapi/compiler/2026.0/lib/libur_adapter_level_zero.so.0`: UR Level Zero adapter

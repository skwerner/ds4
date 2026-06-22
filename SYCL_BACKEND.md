# SYCL Backend — Architecture & Differences from Metal/CUDA

## File Organization

| Backend | Core File | Kernel Files | Compiler |
|---------|-----------|-------------|----------|
| **SYCL** | `ds4_sycl.cpp` (5332 lines) | `sycl/ds4_sycl_iq2_tables.hpp` | `icpx -fsycl` |
| **CUDA** | `ds4_cuda.cu` (13263 lines) | `ds4_iq2_tables_cuda.inc` | `nvcc` |
| **Metal** | `ds4_metal.m` (26629 lines) | 19 files under `metal/*.metal` | `clang` + Obj-C runtime |
| **ROCm** | `ds4_rocm.cu` (131 lines) | 22 files under `rocm/*.cuh` | `hipcc` |

All backends implement the same C interface declared in `ds4_gpu.h`, called from `ds4.c`.

---

## Model Weight Access — The Biggest Architectural Difference

### Metal (most elegant)
Wraps mmap pages directly as `MTLBuffer` objects — no copy at any point. Multiple overlapping views work around `maxBufferLength` limits. GPU reads weights over PCIe on demand. A warm-up kernel pre-faults GPU page tables at init.

### CUDA (two strategies)
- **Preferred**: `cudaHostRegisterMapped` + `cudaHostRegisterReadOnly` — pins the mmap and gives a device pointer for zero-copy PCIe reads.
- **Fallback** (`DS4_CUDA_COPY_MODEL`): `cudaMalloc(model_size)` + full H2D `cudaMemcpy` if the model fits in VRAM.

### SYCL (three tiers)
1. **Level Zero import** (`g_model_imported`, best case): calls `zexDriverImportExternalPointer` at model init via `dlsym` on `libze_intel_gpu.so.1`. Registers the entire mmap with the Intel GPU driver — GPU reads mmap directly, zero copy, zero USM allocations. Falls back silently if the driver extension is unavailable.
2. **USM host path** (current default): `sycl_model_range_ptr` allocates `sycl::malloc_host` buffers. These are pinned system RAM, accessible by both CPU and GPU. First layer does fresh allocations + CPU `memcpy`; subsequent layers reuse same buffers via a stale-cache mechanism (mark `offset=UINT64_MAX`, overwrite with `memcpy`). Total pinned footprint ~3.5 GiB (one layer of weights).
3. **CPU pre-fault**: `madvise(MADV_POPULATE_READ)` at model load avoids ~300K page faults during Level Zero page-pinning (saves ~600 ms per cold copy).

---

## Queue Configuration & Submission Overhead

The SYCL queue is configured with two properties:

```cpp
g_queue = new sycl::queue(*g_context, *g_device, ah,
    sycl::property::queue::in_order{},
    sycl::ext::oneapi::experimental::property::queue::immediate_command_list{});
```

- **`in_order`**: matches CUDA default stream semantics — commands execute in submission order without explicit dependencies.
- **`immediate_command_list`**: Intel DPC++ extension that uses Level Zero immediate command lists (`zeCommandListCreateImmediate`) instead of the two-step "create command list → submit to queue" path. Each `submit()` call dispatches work directly to the GPU command processor, eliminating the driver-level batching handshake that cost ~1.7-2.1 ms per kernel in the default (non-immediate) path.

Without immediate command lists, each `parallel_for` creates a command list, appends the kernel, submits the list to the queue, and synchronizes — a ~2.1 ms round trip entirely on the CPU side. Immediate mode bypasses the list creation and queue submission steps, pinning the bottleneck to the actual GPU dispatch latency (~tens of µs).

### Key differences
- SYCL has no `cudaHostRegister` equivalent — USM `malloc_host` is the closest, but requires explicit `memcpy` to fill (zero-copy `cudaHostRegisterMapped` does not).
- SYCL has no device-side VRAM cache for non-MoE weights (CUDA has `cuda_q8_f16_ranges` for dequantized weight caching).
- SYCL mmap uses `MAP_PRIVATE` (not `MAP_SHARED` like Metal) to avoid Darwin VM accounting issues on Linux.

---

## Memory Allocation

| Purpose | SYCL | CUDA | Metal |
|---------|------|------|-------|
| Activations, KV cache | `sycl::malloc_device` | `cudaMalloc` | `MTLResourceStorageModePrivate` |
| Host-visible scratch | `sycl::malloc_shared` | `cudaMallocManaged` | `MTLResourceStorageModeShared` |
| Model weight staging | `sycl::malloc_host` (pinned) | `cudaHostRegister` (zero-copy) | MTLBuffer wrapping mmap |
| Best-case weights | mmap direct (Level Zero import) | `cudaHostRegisterMapped` | MTLBuffer wrapping mmap |

---

## Kernel Dispatch

| Aspect | SYCL | CUDA | Metal |
|--------|------|------|-------|
| Syntax | `q.parallel_for(range, lambda)` or `h.parallel_for(nd_range, lambda)` | `kernel<<<grid, block>>>()` | `[enc dispatchThreadgroups:]` |
| Queue | Single in-order `sycl::queue` | Default stream (in-order) | Command buffer batching |
| BLAS | oneMKL via `oneapi::mkl::blas::gemm()` | cuBLAS | Custom Metal shaders |
| Warp/sub-group | `sycl::sub_group` + `reduce_over_group()` | `__syncthreads()`, warp shuffles | `simdgroup` ops |
| Async errors | Custom handler (logs, no terminate) | `cudaGetLastError()` | `cb.status` check |
| Kernel fusion | 3 fused quantize+matmul kernels (unique to SYCL) | No fusion | No fusion |

### Submission overhead
SYCL has a known issue: `parallel_for` submissions with 128-256 work-items incur **~1.7-2.1 ms** of CPU-side overhead per call (root cause unknown, likely Level Zero driver interaction). Lightweight kernels (<16 work-items) submit in <0.1 ms. This is the primary motivation for:
- **Fused quantize+matmul kernels** — eliminated one submission per layer (saved ~1.7 ms, dropped encode from 3923→2747 ms)
- **Norm+matmul fusion** — tried but showed no gain (norm is a lightweight kernel)
- **Command graph** investigation (see below)

CUDA and Metal do not exhibit this overhead pattern.

---

## MoE Implementation

### SYCL (`sycl_routed_moe_launch`)
```
if g_model_imported:
    direct mmap ptr (no copy, no CPU readback)
else:
    1. malloc_host USM buffers for gate/up/down (all 256 experts)
    2. memcpy(sel_host, selected->ptr, ...)  — PCIe readback of router output
    3. De-duplicate expert indices on CPU
    4. Copy selected experts from mmap → USM host (0.1 ms for 6 experts)
       - sel ≤ 8: single-threaded loop    ← optimized for decode
       - sel > 8: multi-threaded std::thread
    5. Kernel 1: gate + up + mid projection
    6. Kernel 2: down projection
```
- Quantization: IQ2_XXS gate + Q2_K down; Q4_K gate + Q4_K down
- No expert table preloading (`ds4_gpu_pro_q4_expert_table_auto_available` returns 0)

### CUDA
- Device-side expert cache with LRU eviction
- CuBLAS batched gemm for grouped expert computation
- Staging buffers for async upload overlap
- Expert table preloading with argument buffers

### Metal
- 19+ specialized pipeline states per quantization combo:
  - `moe_mul_mv_id_iq2_xxs`, `iq2_xxs_pair`, `iq2_xxs_pair_swiglu`
  - `moe_mul_mv_id_q2_k`, `q2_k_sum6`
  - `moe_mul_mv_id_q4_k`, `q4_k_pair`, `q4_k_pair_swiglu`, `q4_k_sum6`
  - Group variants: `group_q4_k_pair_swiglu`, `group6_`, `group8_`, `group24_`
- Argument encoders for efficient expert weight table binding
- `MTLSharedEvent` for pipelining selected-id readback with computation

---

## Quantization Support

| Type | SYCL | CUDA | Metal |
|------|------|------|-------|
| Q8_0 | Fused quantize+matmul; separate pre-quantize too | Separate quantize kernel + matmul, dequant to F16/F32 | `kernel_mul_mv_q8_0_f32` |
| Q4_K | Inline in MoE kernel | MoE and dense matmul | Multiple MoE variants |
| Q2_K | MoE down projection | MoE down | `moe_mul_mv_id_q2_k` |
| IQ2_XXS | `constexpr` tables in `.hpp` | `__device__` arrays in `.inc` | In `.metal` sources |
| F16 | oneMKL or custom kernel | cuBLAS or custom | `kernel_mul_mv_f16_f32` |
| FP8 KV | Yes | Yes | Yes |
| Q8→F16 cache | No | Yes (device-side) | No |

---

## Command Graph Status

**Metal**: Mature command buffer batching — all decode operations share a single reusable `MTLComputeCommandEncoder`. Uses `MTLSharedEvent` for signaling. Not a formal graph API but achieves similar effect.

**CUDA**: No command graph usage. Relies on stream-based concurrency with events.

**SYCL**: Investigated `ext::oneapi::experimental::command_graph` (available in Intel SYCL 2026.0). **Blocked** by the MoE dependency chain:
```
Router kernel → PCIe readback of selected indices → CPU → 
  memcpy(mmap → host) → MoE kernel
```
The CPU-in-the-middle break prevents recording a single graph. The Level Zero import eliminates the CPU readback and copy, making a full decode-layer graph possible. Target: reduce 162+ heavy submissions (~2.1 ms each) to 1-2 graph submissions per token, saving ~300-400 ms of encode time.

---

## SYCL-Specific Workarounds

| Issue | Workaround |
|-------|-----------|
| No `__device__` qualifier | IQ2 tables as `constexpr` arrays |
| No `cudaMemAdvise`/prefetch | `madvise(MADV_POPULATE_READ)` for CPU-side page pre-fault |
| No `ze_api.h` installed | `dlsym` runtime lookup from `libze_intel_gpu.so.1` |
| No `cudaHostRegister` | `sycl::malloc_host` + explicit `memcpy` |
| No tensor cores / WMMA | Sub-group `reduce_over_group()` for warp-level reduction |
| Heavy submission overhead | Fused quantize+matmul kernels (3 variants); Level Zero immediate command lists (`sycl::ext::oneapi::experimental::property::queue::immediate_command_list`) |
| MoE CPU-readback blocks graphs | Level Zero import eliminates the dependency |
| Async errors would terminate | Custom handler that logs and continues |

---

## Integrated GPU mode (`DS4_SYCL_ALLOC_HOST=1`)

When set, all memory allocations use `sycl::malloc_host` (pinned system RAM)
instead of `sycl::malloc_device`.  This is intended for integrated GPUs
(Meteor Lake, Lunar Lake, Arrow Lake, etc.) where device memory is the same
DRAM as host memory — there is no separate VRAM to exhaust.

All 12 `malloc_device` call sites redirect through `sycl_alloc_device()`:

| Category | Sites | Purpose |
|----------|-------|---------|
| Tensor alloc | `ds4_gpu_tensor_alloc` | Activations, KV cache, scratch tensors |
| Q8 scratch | `sycl_ensure_q8_bufs` | Quantized input buffers |
| F16 scratch | `sycl_ensure_f16_buf` | f16 conversion buffer |
| HC scratch | `ds4_gpu_hc_split_sinkhorn_tensor` | Hyper-connection quantize scratch |
| Stream selected | `sycl_stream_selected_ensure_bytes` | MoE selected-index device cache |
| Stream expert | `sycl_stream_expert_cache_try_alloc` | SSD streaming expert LRU cache |
| Model range | `ds4_gpu_cache_model_range` | Device copies of weight ranges |

Note: `malloc_host` memory is pinned and counts against the system's
`RLIMIT_MEMLOCK`.  Ensure the process has sufficient locked memory limits
(e.g. `ulimit -l unlimited`) when running with `DS4_SYCL_ALLOC_HOST=1`,
especially with large KV cache sizes.

---

## Key Measurements

| Metric | Value |
|--------|-------|
| Model size | ~141 GiB (DeepSeek V4 Flash) |
| System RAM | ~292 GB |
| GPU | Intel Arc A750 (12.55.8 driver) |
| PCIe | Gen4 ×16 (56 GiB/s) |
| Heavy kernel submission overhead | ~1.7-2.1 ms per `parallel_for` |
| Light kernel overhead | <0.1 ms |
| Encode time (baseline) | 3923 ms/token |
| Encode time (after quantize fusion) | 2747 ms/token |
| GPU execute time | ~57 ms/token |
| Throughput | 0.30 t/s |
| MoE expert copy (6 experts, decode) | ~0.1 ms/layer (after single-thread fix) |
| MoE expert copy (original pthreads) | ~7 ms/layer |
| Level Zero import target | ~2350 ms encode, 0.35 t/s |
| Command graph target | ~2000 ms encode, 0.40 t/s |

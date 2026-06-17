#include "ds4_gpu.h"

#include <sycl/sycl.hpp>
#include <oneapi/mkl/blas.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cctype>
#include <cerrno>
#include <cassert>
#include <fcntl.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <vector>
#include <unordered_map>
#include <mutex>

#define DS4_SYCL_UNUSED __attribute__((unused))
#define DS4_SYCL_MAX_STREAMS 4

#define DS4_GPU_BACKEND_NAME "SYCL"
#define DS4_GPU_LOG_PREFIX "ds4: SYCL "
#define DS4_GPU_BLAS_NAME "oneMKL"

/* =========================================================================
 * Tensor type (opaque to callers, defined here).
 * ========================================================================= */
struct ds4_gpu_tensor {
    void     *ptr;
    uint64_t  bytes;
    int       owner; /* 1 = allocated (must free), 0 = view */
};

/* QK_K constant (must match ds4_cuda.cu exactly). */
#define CUDA_QK_K 256

/* =========================================================================
 * Quantized block types (must match ds4_cuda.cu exactly).
 * ========================================================================= */
typedef struct {
    uint8_t scales[CUDA_QK_K / 16];
    uint8_t qs[CUDA_QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} sycl_block_q2_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[CUDA_QK_K / 2];
} sycl_block_q4_K;

typedef struct {
    float d;
    int8_t qs[CUDA_QK_K];
    int16_t bsums[CUDA_QK_K / 16];
} sycl_block_q8_K;

typedef struct {
    uint16_t d;
    uint16_t qs[CUDA_QK_K / 8];
} sycl_block_iq2_xxs;

/* IQ2 lookup tables — SYCL-portable constexpr arrays */
#include "sycl/ds4_sycl_iq2_tables.hpp"

/* C++ half-to-float via sycl::half (device-callable). */
static float sycl_f16_to_f32(uint16_t v) {
    return (float)sycl::bit_cast<sycl::half>(v);
}

/* Softplus (device-callable). */
static float sycl_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

/* IQ2_XXS dot product: one or more blocks (device-callable). */
static float sycl_iq2_xxs_dot_f32(const sycl_block_iq2_xxs *row, const float *x, uint32_t nb) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        const sycl_block_iq2_xxs *xb = row + b;
        const float d = sycl_f16_to_f32(xb->d);
        const uint16_t *q2 = xb->qs;
        const float *xf = x + (uint64_t)b * CUDA_QK_K;
        for (uint32_t ib32 = 0; ib32 < CUDA_QK_K / 32; ib32++) {
            uint32_t aux_g = (uint32_t)q2[0] | ((uint32_t)q2[1] << 16);
            uint32_t aux_s = (uint32_t)q2[2] | ((uint32_t)q2[3] << 16);
            q2 += 4;
            float dl = d * (0.5f + (float)(aux_s >> 28)) * 0.25f;
            uint8_t grids[4] = {
                (uint8_t)(aux_g & 0xffu),
                (uint8_t)((aux_g >> 8) & 0xffu),
                (uint8_t)((aux_g >> 16) & 0xffu),
                (uint8_t)((aux_g >> 24) & 0xffu),
            };
            for (uint32_t half = 0; half < 2; half++) {
                for (uint32_t g = 0; g < 2; g++) {
                    uint32_t gi = half * 2 + g;
                    uint64_t grid = sycl_iq2xxs_grid[grids[gi]];
                    uint8_t signs = sycl_ksigns_iq2xs[(aux_s >> (14u * half + 7u * g)) & 127u];
                    for (uint32_t i = 0; i < 8; i++) {
                        float w = (float)((grid >> (8u * i)) & 0xffu);
                        if (signs & (1u << i)) w = -w;
                        acc += dl * w * xf[ib32 * 32u + half * 16u + g * 8u + i];
                    }
                }
            }
        }
    }
    return acc;
}

/* Q2_K dot product: one or more blocks (device-callable). */
static float sycl_q2_K_dot_f32(const sycl_block_q2_K *row, const float *x, uint32_t nb) {
    float acc = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        const sycl_block_q2_K *xb = row + b;
        const float d = sycl_f16_to_f32(xb->d);
        const float dmin = sycl_f16_to_f32(xb->dmin);
        for (uint32_t il = 0; il < 16; il++) {
            uint32_t chunk = il / 8u;
            uint32_t pair  = il & 1u;
            uint32_t shift = ((il / 2u) & 3u) * 2u;
            uint8_t  sc    = xb->scales[il];
            float dl = d * (float)(sc & 0x0fu);
            float ml = dmin * (float)(sc >> 4);
            const uint8_t *q  = xb->qs + 32u * chunk + 16u * pair;
            const float   *xf = x + (uint64_t)b * CUDA_QK_K + chunk * 128u + ((il % 8u) / 2u) * 32u + pair * 16u;
            for (uint32_t i = 0; i < 16; i++) {
                float w = dl * (float)((q[i] >> shift) & 3u) - ml;
                acc += w * xf[i];
            }
        }
    }
    return acc;
}

/* Shared router-select kernel body (1 work-item per token). */
static void sycl_router_select_body(
        int32_t *sel, float *w, float *prob,
        const float *bias, const int32_t *hash,
        const float *log, const int32_t *tokens, int32_t token_scalar,
        uint32_t hash_rows, int has_bias, int hash_mode,
        float expert_weight_scale, uint32_t n_expert, uint32_t n_expert_used) {
    for (uint32_t i = 0; i < n_expert; i++)
        prob[i] = sqrtf(sycl_softplus(log[i]));
    if (hash_mode) {
        int32_t tok = tokens ? *tokens : token_scalar;
        if (tok < 0 || (uint32_t)tok >= hash_rows) tok = 0;
        const int32_t *row = hash + (uint64_t)tok * n_expert_used;
        for (uint32_t j = 0; j < n_expert_used; j++) sel[j] = row[j];
    } else {
        for (uint32_t j = 0; j < n_expert_used; j++) sel[j] = -1;
        for (uint32_t e = 0; e < n_expert; e++) {
            float score = prob[e] + (has_bias ? bias[e] : 0.0f);
            for (uint32_t j = 0; j < n_expert_used; j++) {
                int32_t sj = sel[j];
                if (sj < 0 || score > prob[(uint32_t)sj] + (has_bias ? bias[(uint32_t)sj] : 0.0f)) {
                    for (uint32_t k = n_expert_used - 1; k > j; k--) sel[k] = sel[k - 1];
                    sel[j] = (int32_t)e;
                    break;
                }
            }
        }
    }
    float sum = 0.0f;
    for (uint32_t j = 0; j < n_expert_used; j++) {
        int32_t e = sel[j];
        float v = (e >= 0 && (uint32_t)e < n_expert) ? prob[(uint32_t)e] : 0.0f;
        w[j] = v;
        sum += v;
    }
    const float eps = 6.103515625e-5f;
    sum = fmaxf(sum, eps);
    float scale = expert_weight_scale / sum;
    for (uint32_t j = 0; j < n_expert_used; j++) w[j] *= scale;
}

/* =========================================================================
 * Global state.
 * ========================================================================= */
static sycl::queue             *g_queue          = nullptr;
static sycl::device            *g_device         = nullptr;
static sycl::context           *g_context        = nullptr;
static int                      g_initialized    = 0;
static int                      g_quality_mode   = 0;
static int                      g_ssd_streaming  = 0;

/* Model mapping state */
static const void              *g_model_host_base       = nullptr;
static uint64_t                 g_model_registered_size = 0;
static int                      g_model_registered      = 0;
static int                      g_model_fd              = -1;

/* Error helper */
static int sycl_ok(bool cond, const char *what) {
    if (!cond) {
        fprintf(stderr, "ds4: SYCL %s failed\n", what);
        return 0;
    }
    return 1;
}

static int sycl_ok_event(sycl::event &ev, const char *what) {
    try {
        ev.wait_and_throw();
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL %s failed: %s\n", what, e.what());
        return 0;
    }
}

/* =========================================================================
 * Streaming expert cache
 * ========================================================================= */

/* Budget limits (matches CUDA defaults) */
#define SYCL_STREAM_EXPERT_DEFAULT 512u
#define SYCL_STREAM_EXPERT_MAX     23424u

struct sycl_stream_expert_cache_slot {
    int         valid;
    const void *model_map;
    uint64_t    model_size;
    uint32_t    layer;
    uint32_t    n_total_expert;
    uint32_t    expert;
    uint64_t    gate_offset;
    uint64_t    up_offset;
    uint64_t    down_offset;
    uint64_t    gate_expert_bytes;
    uint64_t    down_expert_bytes;
    uint64_t    age;
};

struct sycl_stream_expert_cache {
    int      valid;
    uint32_t capacity;
    uint32_t count;
    uint64_t tick;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    char    *gate_ptr;
    char    *up_ptr;
    char    *down_ptr;
    uint64_t gate_capacity;
    uint64_t up_capacity;
    uint64_t down_capacity;
    std::vector<sycl_stream_expert_cache_slot> slots;
};

struct sycl_stream_selected_cache {
    int      valid;
    const void *model_map;
    uint32_t layer;
    uint32_t n_total_expert;
    uint32_t n_selected;
    uint32_t slot_count;
    uint32_t compact_count;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
    uint64_t gate_expert_bytes;
    uint64_t down_expert_bytes;
    char    *gate_ptr;
    char    *up_ptr;
    char    *down_ptr;
    uint64_t gate_capacity;
    uint64_t up_capacity;
    uint64_t down_capacity;
    int32_t *slot_selected_ptr;
    uint64_t slot_selected_capacity;
    ds4_gpu_tensor slot_selected_tensor;
};

static sycl_stream_selected_cache  g_stream_selected_cache;
static sycl_stream_expert_cache    g_stream_expert_cache;
static uint32_t g_stream_expert_budget_override = 0;
static uint32_t g_stream_expert_runtime_cap = 0;
static uint32_t g_stream_expert_memory_cap_notice = 0;
static uint64_t g_stream_expert_runtime_gate_bytes = 0;
static uint64_t g_stream_expert_runtime_down_bytes = 0;

static void sycl_stream_selected_cache_invalidate(void) {
    g_stream_selected_cache.valid = 0;
}

static void sycl_stream_selected_cache_release_all(void) {
    if (g_stream_selected_cache.gate_ptr) {
        sycl::free(g_stream_selected_cache.gate_ptr, *g_context);
        g_stream_selected_cache.gate_ptr = nullptr;
    }
    if (g_stream_selected_cache.up_ptr) {
        sycl::free(g_stream_selected_cache.up_ptr, *g_context);
        g_stream_selected_cache.up_ptr = nullptr;
    }
    if (g_stream_selected_cache.down_ptr) {
        sycl::free(g_stream_selected_cache.down_ptr, *g_context);
        g_stream_selected_cache.down_ptr = nullptr;
    }
    if (g_stream_selected_cache.slot_selected_ptr) {
        sycl::free(g_stream_selected_cache.slot_selected_ptr, *g_context);
        g_stream_selected_cache.slot_selected_ptr = nullptr;
    }
    memset(&g_stream_selected_cache, 0, sizeof(g_stream_selected_cache));
}

static void sycl_stream_expert_cache_release_all(void) {
    if (g_stream_expert_cache.gate_ptr) {
        sycl::free(g_stream_expert_cache.gate_ptr, *g_context);
    }
    if (g_stream_expert_cache.up_ptr) {
        sycl::free(g_stream_expert_cache.up_ptr, *g_context);
    }
    if (g_stream_expert_cache.down_ptr) {
        sycl::free(g_stream_expert_cache.down_ptr, *g_context);
    }
    g_stream_expert_cache.slots.clear();
    memset(&g_stream_expert_cache, 0, sizeof(g_stream_expert_cache));
}

static uint32_t sycl_stream_expert_cache_requested_budget(void) {
    uint32_t cap = g_stream_expert_budget_override != 0 ?
        g_stream_expert_budget_override : SYCL_STREAM_EXPERT_DEFAULT;
    const char *env = getenv("DS4_CUDA_STREAMING_EXPERT_CACHE_N");
    if (env && env[0]) {
        char *end = nullptr;
        errno = 0;
        unsigned long v = strtoul(env, &end, 10);
        while (end && (*end == ' ' || *end == '\t')) end++;
        if (end != env && errno == 0 && end && *end == '\0') {
            cap = v > SYCL_STREAM_EXPERT_MAX ?
                SYCL_STREAM_EXPERT_MAX : (uint32_t)v;
        }
    }
    if (cap > SYCL_STREAM_EXPERT_MAX) cap = SYCL_STREAM_EXPERT_MAX;
    return cap;
}

static uint32_t sycl_stream_expert_cache_configured_budget(void) {
    uint32_t cap = sycl_stream_expert_cache_requested_budget();
    if (g_stream_expert_runtime_cap != 0 && cap > g_stream_expert_runtime_cap) {
        cap = g_stream_expert_runtime_cap;
    }
    return cap;
}

static int sycl_stream_expert_cache_budget_visible_to_shared(void) {
    if (!g_ssd_streaming) return 0;
    if (g_stream_expert_budget_override != 0) return 1;
    const char *env = getenv("DS4_CUDA_STREAMING_EXPERT_CACHE_N");
    if (env && env[0]) return 1;
    env = getenv("DS4_CUDA_ENABLE_STREAMING_EXPERT_HOTLIST");
    if (!env || !env[0]) {
        env = getenv("DS4_CUDA_STREAMING_EXPERT_HOTLIST");
    }
    return env && env[0] && strcmp(env, "0") != 0;
}

static uint64_t sycl_stream_expert_cache_expert_bytes(
        uint64_t gate_expert_bytes, uint64_t down_expert_bytes) {
    if (gate_expert_bytes == 0 || down_expert_bytes == 0 ||
        gate_expert_bytes > (UINT64_MAX - down_expert_bytes) / 2ull) {
        return 0;
    }
    return gate_expert_bytes * 2ull + down_expert_bytes;
}

static void sycl_stream_expert_cache_note_size(
        uint64_t gate_expert_bytes, uint64_t down_expert_bytes) {
    if (g_stream_expert_runtime_gate_bytes == gate_expert_bytes &&
        g_stream_expert_runtime_down_bytes == down_expert_bytes) return;
    g_stream_expert_runtime_gate_bytes = gate_expert_bytes;
    g_stream_expert_runtime_down_bytes = down_expert_bytes;
    g_stream_expert_runtime_cap = 0;
    g_stream_expert_memory_cap_notice = 0;
}

static int sycl_stream_selected_ensure_bytes(
        char **ptr, uint64_t *capacity, uint64_t needed, const char *what) {
    if (*capacity >= needed) return 1;
    if (*ptr) sycl::free(*ptr, *g_context);
    *ptr = (char *)sycl::malloc_device(needed, *g_queue);
    if (!*ptr) {
        *capacity = 0;
        fprintf(stderr, "ds4: SYCL selected %s allocation failed (%llu bytes)\n",
                what, (unsigned long long)needed);
        return 0;
    }
    *capacity = needed;
    return 1;
}

static int sycl_stream_selected_ensure_i32(
        int32_t **ptr, uint64_t *capacity, uint64_t needed, const char *what) {
    if (*capacity >= needed) return 1;
    if (*ptr) sycl::free(*ptr, *g_context);
    *ptr = (int32_t *)sycl::malloc_device(needed * sizeof(int32_t), *g_queue);
    if (!*ptr) {
        *capacity = 0;
        fprintf(stderr, "ds4: SYCL selected %s allocation failed (%llu elems)\n",
                what, (unsigned long long)needed);
        return 0;
    }
    *capacity = needed;
    return 1;
}

static uint32_t sycl_stream_expert_cache_lru_slot(
        sycl_stream_expert_cache *cache) {
    uint32_t lru = 0;
    uint64_t min_age = cache->slots[0].age;
    for (uint32_t i = 0; i < cache->capacity; i++) {
        if (!cache->slots[i].valid) return i;
        if (cache->slots[i].age < min_age) {
            min_age = cache->slots[i].age;
            lru = i;
        }
    }
    return lru;
}

static uint32_t sycl_stream_expert_cache_live_budget(
        uint32_t target_cap, uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes, uint64_t reclaim_bytes,
        int allow_reclaim) {
    (void)reclaim_bytes; (void)allow_reclaim;
    size_t free_bytes = 0, total_bytes = 0;
    try {
        total_bytes = g_device->get_info<sycl::info::device::global_mem_size>();
        /* SYCL doesn't expose free memory; use 80% heuristic */
        free_bytes = (size_t)(total_bytes * 0.8);
    } catch (...) { return 0; }
    if (gate_expert_bytes == 0 || down_expert_bytes == 0) return 0;
    uint64_t expert_bytes = sycl_stream_expert_cache_expert_bytes(
        gate_expert_bytes, down_expert_bytes);
    if (expert_bytes == 0) return 0;
    uint64_t max_by_mem = (uint64_t)free_bytes / (expert_bytes + 1024);
    if (max_by_mem > SYCL_STREAM_EXPERT_MAX) max_by_mem = SYCL_STREAM_EXPERT_MAX;
    if ((uint64_t)target_cap > max_by_mem) target_cap = (uint32_t)max_by_mem;
    return target_cap > 0 ? target_cap : 1;
}

static uint32_t sycl_stream_expert_cache_shrunken_cap(uint32_t cap) {
    if (cap <= 1) return 0;
    uint32_t next = cap / 2;
    if (next < 1) next = 1;
    return next;
}

static void sycl_stream_expert_cache_note_oom_cap(
        uint32_t cap, uint32_t new_cap, uint64_t expert_bytes,
        const char *what) {
    if (g_stream_expert_memory_cap_notice) return;
    if (new_cap != cap) {
        fprintf(stderr, "ds4: SYCL streaming expert OOM at cap=%u expert_bytes=%llu %s — retrying cap=%u\n",
                cap, (unsigned long long)expert_bytes,
                what ? what : "", new_cap);
    }
    g_stream_expert_memory_cap_notice =
        (expert_bytes > 0 && new_cap != 0) ? 0 : 1;
}

static int sycl_stream_expert_cache_try_alloc(
        uint32_t cap, uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes,
        char **gate_ptr, char **up_ptr, char **down_ptr,
        const char **alloc_error) {
    *gate_ptr = (char *)sycl::malloc_device((uint64_t)cap * gate_expert_bytes, *g_queue);
    if (!*gate_ptr) { *alloc_error = "gate"; return 0; }
    *up_ptr = (char *)sycl::malloc_device((uint64_t)cap * gate_expert_bytes, *g_queue);
    if (!*up_ptr) { sycl::free(*gate_ptr, *g_context); *gate_ptr = nullptr; *alloc_error = "up"; return 0; }
    *down_ptr = (char *)sycl::malloc_device((uint64_t)cap * down_expert_bytes, *g_queue);
    if (!*down_ptr) { sycl::free(*gate_ptr, *g_context); sycl::free(*up_ptr, *g_context); *gate_ptr = nullptr; *up_ptr = nullptr; *alloc_error = "down"; return 0; }
    *alloc_error = nullptr;
    return 1;
}

/* Copy expert weights from model to a cache slot via simple H2D (model is mmapped). */
static int sycl_stream_expert_cache_load_slot(
        sycl_stream_expert_cache *cache,
        const void *model_map, uint64_t model_size,
        uint32_t slot, uint32_t layer, uint32_t n_total_expert,
        uint32_t expert, uint64_t gate_offset, uint64_t up_offset,
        uint64_t down_offset, uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    const uint64_t gate_src = gate_offset + (uint64_t)expert * gate_expert_bytes;
    const uint64_t up_src   = up_offset   + (uint64_t)expert * gate_expert_bytes;
    const uint64_t down_src = down_offset + (uint64_t)expert * down_expert_bytes;
    const uint64_t gate_dst = (uint64_t)slot * gate_expert_bytes;
    const uint64_t down_dst = (uint64_t)slot * down_expert_bytes;
    try {
        g_queue->memcpy(cache->gate_ptr + gate_dst,
                        (const char *)model_map + gate_src, gate_expert_bytes);
        g_queue->memcpy(cache->up_ptr + gate_dst,
                        (const char *)model_map + up_src, gate_expert_bytes);
        g_queue->memcpy(cache->down_ptr + down_dst,
                        (const char *)model_map + down_src, down_expert_bytes);
        g_queue->wait_and_throw();
    } catch (...) { return 0; }
    sycl_stream_expert_cache_slot &entry = cache->slots[slot];
    entry.valid = 1;
    entry.model_map = model_map;
    entry.model_size = model_size;
    entry.layer = layer;
    entry.n_total_expert = n_total_expert;
    entry.expert = expert;
    entry.gate_offset = gate_offset;
    entry.up_offset = up_offset;
    entry.down_offset = down_offset;
    entry.gate_expert_bytes = gate_expert_bytes;
    entry.down_expert_bytes = down_expert_bytes;
    entry.age = ++cache->tick;
    return 1;
}

static int sycl_stream_expert_cache_find(
        sycl_stream_expert_cache *cache,
        const void *model_map, uint64_t model_size,
        uint32_t layer, uint32_t n_total_expert, uint32_t expert,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint64_t gate_expert_bytes, uint64_t down_expert_bytes) {
    if (!cache || !cache->valid) return -1;
    for (uint32_t i = 0; i < cache->capacity; i++) {
        const sycl_stream_expert_cache_slot &slot = cache->slots[i];
        if (slot.valid &&
            slot.model_map == model_map &&
            slot.model_size == model_size &&
            slot.layer == layer &&
            slot.n_total_expert == n_total_expert &&
            slot.expert == expert &&
            slot.gate_offset == gate_offset &&
            slot.up_offset == up_offset &&
            slot.down_offset == down_offset &&
            slot.gate_expert_bytes == gate_expert_bytes &&
            slot.down_expert_bytes == down_expert_bytes) {
            return (int)i;
        }
    }
    return -1;
}

static int sycl_stream_expert_cache_copy_to_compact(
        sycl_stream_expert_cache *cache,
        uint32_t cache_slot, uint32_t compact_slot,
        char *compact_gate, char *compact_up, char *compact_down) {
    const uint64_t src_gate = (uint64_t)cache_slot * cache->gate_expert_bytes;
    const uint64_t src_down = (uint64_t)cache_slot * cache->down_expert_bytes;
    const uint64_t dst_gate = (uint64_t)compact_slot * cache->gate_expert_bytes;
    const uint64_t dst_down = (uint64_t)compact_slot * cache->down_expert_bytes;
    try {
        g_queue->memcpy(compact_gate + dst_gate,
                        cache->gate_ptr + src_gate, cache->gate_expert_bytes);
        g_queue->memcpy(compact_up + dst_gate,
                        cache->up_ptr + src_gate, cache->gate_expert_bytes);
        g_queue->memcpy(compact_down + dst_down,
                        cache->down_ptr + src_down, cache->down_expert_bytes);
        g_queue->wait_and_throw();
        return 1;
    } catch (...) { return 0; }
}

static int sycl_stream_expert_cache_seed_one(
        sycl_stream_expert_cache *cache,
        const void *model_map, uint64_t model_size,
        uint32_t layer, uint32_t n_total_expert, uint32_t expert,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint64_t gate_expert_bytes, uint64_t down_expert_bytes) {
    int cache_slot = sycl_stream_expert_cache_find(cache, model_map, model_size,
        layer, n_total_expert, expert, gate_offset, up_offset, down_offset,
        gate_expert_bytes, down_expert_bytes);
    if (cache_slot >= 0) {
        cache->slots[(uint32_t)cache_slot].age = ++cache->tick;
        return 1;
    }
    const uint32_t load_slot = sycl_stream_expert_cache_lru_slot(cache);
    const int append = !cache->slots[load_slot].valid;
    if (!sycl_stream_expert_cache_load_slot(cache, model_map, model_size,
            load_slot, layer, n_total_expert, expert,
            gate_offset, up_offset, down_offset,
            gate_expert_bytes, down_expert_bytes)) return 0;
    if (append && cache->count < cache->capacity) cache->count++;
    return 1;
}

static sycl_stream_expert_cache *sycl_stream_expert_cache_prepare(
        uint64_t gate_expert_bytes, uint64_t down_expert_bytes,
        uint32_t target_cap) {
    const uint64_t expert_bytes =
        sycl_stream_expert_cache_expert_bytes(gate_expert_bytes, down_expert_bytes);
    if (expert_bytes == 0) return nullptr;
    sycl_stream_expert_cache_note_size(gate_expert_bytes, down_expert_bytes);
    const uint32_t requested_cap = sycl_stream_expert_cache_configured_budget();
    if (requested_cap == 0) return nullptr;
    if (target_cap == 0 || target_cap > requested_cap) target_cap = requested_cap;
    if (target_cap == 0) return nullptr;
    const int same_dims =
        g_stream_expert_cache.valid &&
        g_stream_expert_cache.gate_expert_bytes == gate_expert_bytes &&
        g_stream_expert_cache.down_expert_bytes == down_expert_bytes;
    if (!same_dims && g_stream_expert_cache.valid)
        sycl_stream_expert_cache_release_all();
    if (same_dims && g_stream_expert_cache.capacity != 0 &&
        g_stream_expert_cache.capacity >= target_cap &&
        g_stream_expert_cache.slots.size() == g_stream_expert_cache.capacity)
        return &g_stream_expert_cache;

    uint64_t reclaim_bytes = 0;
    if (same_dims && g_stream_expert_cache.capacity != 0 &&
        (uint64_t)g_stream_expert_cache.capacity <= UINT64_MAX / expert_bytes)
        reclaim_bytes = (uint64_t)g_stream_expert_cache.capacity * expert_bytes;
    uint32_t cap = sycl_stream_expert_cache_live_budget(
        target_cap, gate_expert_bytes, down_expert_bytes, reclaim_bytes,
        reclaim_bytes == 0);
    if (cap == 0) return nullptr;
    if (same_dims && g_stream_expert_cache.capacity != 0 &&
        g_stream_expert_cache.capacity >= cap &&
        g_stream_expert_cache.slots.size() == g_stream_expert_cache.capacity)
        return &g_stream_expert_cache;

    sycl_stream_expert_cache_release_all();
    while (cap != 0) {
        if ((uint64_t)cap > UINT64_MAX / gate_expert_bytes ||
            (uint64_t)cap > UINT64_MAX / down_expert_bytes) {
            fprintf(stderr, "ds4: SYCL streaming expert cache size overflow\n");
            return nullptr;
        }
        char *gate_ptr = nullptr, *up_ptr = nullptr, *down_ptr = nullptr;
        const char *alloc_error = nullptr;
        if (!sycl_stream_expert_cache_try_alloc(cap, gate_expert_bytes,
                down_expert_bytes, &gate_ptr, &up_ptr, &down_ptr, &alloc_error)) {
            const uint32_t new_cap = sycl_stream_expert_cache_shrunken_cap(cap);
            sycl_stream_expert_cache_note_oom_cap(cap, new_cap, expert_bytes, alloc_error);
            cap = new_cap;
            if (cap != 0)
                cap = sycl_stream_expert_cache_live_budget(cap, gate_expert_bytes,
                    down_expert_bytes, 0, 1);
            continue;
        }
        try { g_stream_expert_cache.slots.resize(cap); }
        catch (...) {
            fprintf(stderr, "ds4: SYCL streaming expert cache metadata allocation failed\n");
            sycl::free(gate_ptr, *g_context);
            sycl::free(up_ptr, *g_context);
            sycl::free(down_ptr, *g_context);
            sycl_stream_expert_cache_release_all();
            return nullptr;
        }
        g_stream_expert_cache.valid = 1;
        g_stream_expert_cache.capacity = cap;
        g_stream_expert_cache.count = 0;
        g_stream_expert_cache.tick = 0;
        g_stream_expert_cache.gate_expert_bytes = gate_expert_bytes;
        g_stream_expert_cache.down_expert_bytes = down_expert_bytes;
        g_stream_expert_cache.gate_ptr = gate_ptr;
        g_stream_expert_cache.up_ptr = up_ptr;
        g_stream_expert_cache.down_ptr = down_ptr;
        g_stream_expert_cache.gate_capacity = (uint64_t)cap * gate_expert_bytes;
        g_stream_expert_cache.up_capacity = (uint64_t)cap * gate_expert_bytes;
        g_stream_expert_cache.down_capacity = (uint64_t)cap * down_expert_bytes;
        return &g_stream_expert_cache;
    }
    return nullptr;
}

/* Simple H2D copy from model map (handles both mmapped and non-file cases). */
static int sycl_model_copy_to_device(
        char *dst, const void *model_map, uint64_t model_size,
        uint64_t offset, uint64_t bytes, const char *what) {
    (void)model_size; (void)what;
    if (!dst || !model_map || bytes == 0) return 0;
    try {
        g_queue->memcpy(dst, (const char *)model_map + offset, bytes);
        g_queue->wait_and_throw();
        return 1;
    } catch (...) { return 0; }
}

static int sycl_stream_selected_cache_begin_compact_load(
        const void    *model_map,
        uint64_t       model_size,
        uint32_t       layer,
        const int32_t *compact_ids,
        const int32_t *slot_ids,
        uint32_t       n_total_expert,
        uint32_t       compact_count,
        uint32_t       slot_count,
        uint64_t       gate_offset,
        uint64_t       up_offset,
        uint64_t       down_offset,
        uint64_t       gate_expert_bytes,
        uint64_t       down_expert_bytes,
        int            strict_failure,
        int            allow_global_cache) {
    sycl_stream_selected_cache_invalidate();
    if (!g_ssd_streaming) return 1;
    if (!model_map || !compact_ids || !slot_ids ||
        n_total_expert == 0 || compact_count == 0 ||
        compact_count > n_total_expert || slot_count == 0 ||
        gate_expert_bytes == 0 || down_expert_bytes == 0)
        return 0;
    if ((uint64_t)n_total_expert > UINT64_MAX / gate_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / down_expert_bytes ||
        (uint64_t)compact_count > UINT64_MAX / gate_expert_bytes ||
        (uint64_t)compact_count > UINT64_MAX / down_expert_bytes) {
        fprintf(stderr, "ds4: SYCL streaming selected expert size overflow\n");
        return 0;
    }
    const uint64_t full_gate_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
    const uint64_t full_down_bytes = (uint64_t)n_total_expert * down_expert_bytes;
    const uint64_t compact_gate_bytes = (uint64_t)compact_count * gate_expert_bytes;
    const uint64_t compact_down_bytes = (uint64_t)compact_count * down_expert_bytes;
    if (gate_offset > model_size || up_offset > model_size || down_offset > model_size ||
        full_gate_bytes > model_size - gate_offset ||
        full_gate_bytes > model_size - up_offset ||
        full_down_bytes > model_size - down_offset) {
        fprintf(stderr, "ds4: SYCL streaming selected expert range outside model map\n");
        return 0;
    }

    if (!allow_global_cache) sycl_stream_expert_cache_release_all();

    if (!sycl_stream_selected_ensure_bytes(&g_stream_selected_cache.gate_ptr,
            &g_stream_selected_cache.gate_capacity, compact_gate_bytes, "gate") ||
        !sycl_stream_selected_ensure_bytes(&g_stream_selected_cache.up_ptr,
            &g_stream_selected_cache.up_capacity, compact_gate_bytes, "up") ||
        !sycl_stream_selected_ensure_bytes(&g_stream_selected_cache.down_ptr,
            &g_stream_selected_cache.down_capacity, compact_down_bytes, "down") ||
        !sycl_stream_selected_ensure_i32(&g_stream_selected_cache.slot_selected_ptr,
            &g_stream_selected_cache.slot_selected_capacity, slot_count, "slots"))
        return strict_failure ? 0 : 1;

    if (allow_global_cache)
        sycl_stream_expert_cache_note_size(gate_expert_bytes, down_expert_bytes);

    const uint32_t configured_budget = sycl_stream_expert_cache_configured_budget();
    const int use_global_cache = allow_global_cache && configured_budget != 0;
    sycl_stream_expert_cache *expert_cache = use_global_cache ?
        sycl_stream_expert_cache_prepare(gate_expert_bytes, down_expert_bytes, configured_budget) : nullptr;
    int expert_cache_disabled = expert_cache == nullptr;

    for (uint32_t i = 0; i < compact_count; i++) {
        if (compact_ids[i] < 0 || (uint32_t)compact_ids[i] >= n_total_expert) {
            fprintf(stderr, "ds4: SYCL streaming selected expert id %d out of range at layer %u\n",
                    compact_ids[i], layer);
            return 0;
        }
        const uint64_t expert = (uint64_t)(uint32_t)compact_ids[i];
        const uint64_t gate_dst = (uint64_t)i * gate_expert_bytes;
        const uint64_t down_dst = (uint64_t)i * down_expert_bytes;
        int copied_from_cache = 0;

        if (!expert_cache_disabled) {
            int cs = sycl_stream_expert_cache_find(expert_cache, model_map, model_size,
                layer, n_total_expert, (uint32_t)expert,
                gate_offset, up_offset, down_offset,
                gate_expert_bytes, down_expert_bytes);
            if (cs >= 0) {
                expert_cache->slots[(uint32_t)cs].age = ++expert_cache->tick;
            } else {
                const uint32_t load_slot = sycl_stream_expert_cache_lru_slot(expert_cache);
                const int append = !expert_cache->slots[load_slot].valid;
                if (sycl_stream_expert_cache_load_slot(expert_cache, model_map, model_size,
                        load_slot, layer, n_total_expert, (uint32_t)expert,
                        gate_offset, up_offset, down_offset,
                        gate_expert_bytes, down_expert_bytes)) {
                    if (append && expert_cache->count < expert_cache->capacity)
                        expert_cache->count++;
                    cs = (int)load_slot;
                } else {
                    expert_cache_disabled = 1;
                    cs = -1;
                }
            }
            if (cs >= 0)
                copied_from_cache = sycl_stream_expert_cache_copy_to_compact(
                    expert_cache, (uint32_t)cs, i,
                    g_stream_selected_cache.gate_ptr,
                    g_stream_selected_cache.up_ptr,
                    g_stream_selected_cache.down_ptr);
            if (!copied_from_cache) expert_cache_disabled = 1;
        }

        if (!copied_from_cache) {
            const uint64_t gate_src = gate_offset + expert * gate_expert_bytes;
            const uint64_t up_src   = up_offset   + expert * gate_expert_bytes;
            const uint64_t down_src = down_offset + expert * down_expert_bytes;
            if (!sycl_model_copy_to_device(g_stream_selected_cache.gate_ptr + gate_dst,
                    model_map, model_size, gate_src, gate_expert_bytes, "gate") ||
                !sycl_model_copy_to_device(g_stream_selected_cache.up_ptr + gate_dst,
                    model_map, model_size, up_src, gate_expert_bytes, "up") ||
                !sycl_model_copy_to_device(g_stream_selected_cache.down_ptr + down_dst,
                    model_map, model_size, down_src, down_expert_bytes, "down")) {
                sycl_stream_selected_cache_invalidate();
                return strict_failure ? 0 : 1;
            }
        }
    }

    try {
        g_queue->memcpy(g_stream_selected_cache.slot_selected_ptr, slot_ids,
                        (size_t)slot_count * sizeof(int32_t));
        g_queue->wait_and_throw();
    } catch (...) {
        sycl_stream_selected_cache_invalidate();
        return strict_failure ? 0 : 1;
    }

    g_stream_selected_cache.model_map = model_map;
    g_stream_selected_cache.layer = layer;
    g_stream_selected_cache.n_total_expert = n_total_expert;
    g_stream_selected_cache.n_selected = slot_count;
    g_stream_selected_cache.slot_count = slot_count;
    g_stream_selected_cache.compact_count = compact_count;
    g_stream_selected_cache.gate_offset = gate_offset;
    g_stream_selected_cache.up_offset = up_offset;
    g_stream_selected_cache.down_offset = down_offset;
    g_stream_selected_cache.gate_expert_bytes = gate_expert_bytes;
    g_stream_selected_cache.down_expert_bytes = down_expert_bytes;
    g_stream_selected_cache.slot_selected_tensor.ptr =
        g_stream_selected_cache.slot_selected_ptr;
    g_stream_selected_cache.slot_selected_tensor.bytes =
        (uint64_t)slot_count * sizeof(int32_t);
    g_stream_selected_cache.slot_selected_tensor.owner = 0;
    g_stream_selected_cache.valid = 1;
    return 1;
}

/* =========================================================================
 * helpers: model range caching
 * ========================================================================= */
struct sycl_model_range {
    const void *host_base;
    uint64_t    offset;
    uint64_t    bytes;
    char       *device_ptr;
};

static std::vector<sycl_model_range> g_model_ranges;
static std::mutex                    g_model_ranges_mutex;

/* Helper: return device pointer for model range, caching on-the-fly. */
static const char *sycl_model_range_ptr(const void *model_map, uint64_t offset, uint64_t bytes, const char *what) {
    (void)what;
    if (!g_queue) return nullptr;
    if (bytes == 0) return (const char *)model_map + offset;
    {
        std::lock_guard<std::mutex> lock(g_model_ranges_mutex);
        for (auto &r : g_model_ranges) {
            if (r.host_base == model_map && offset >= r.offset && offset + bytes <= r.offset + r.bytes) {
                return r.device_ptr + (offset - r.offset);
            }
        }
    }
    try {
        char *dptr = (char *)sycl::malloc_device(bytes, *g_queue);
        if (!dptr) return nullptr;
        g_queue->memcpy(dptr, (const char *)model_map + offset, bytes);
        std::lock_guard<std::mutex> lock(g_model_ranges_mutex);
        g_model_ranges.push_back({model_map, offset, bytes, dptr});
        return dptr;
    } catch (...) { return nullptr; }
}

/* Helper: f32->f16 conversion kernel (used by f16 matmul batched path). */
static void sycl_convert_f32_f16(sycl::queue &q, uint64_t count, const float *src, sycl::half *dst) {
    q.parallel_for(sycl::range<1>(count), [=](sycl::id<1> i) {
        dst[i] = sycl::half(src[i]);
    });
}

/* =========================================================================
 * ds4_gpu_init / cleanup
 * ========================================================================= */
extern "C" int ds4_gpu_init(void) {
    if (g_initialized) return 1;
    try {
        sycl::device d(sycl::gpu_selector_v);
        g_device  = new sycl::device(d);
        g_context = new sycl::context(*g_device);
        /* In-order queue — matches CUDA default stream semantics */
        g_queue   = new sycl::queue(*g_context, *g_device,
                                    sycl::property::queue::in_order{});
        g_initialized = 1;
        fprintf(stderr, "ds4: SYCL backend initialized on %s\n",
                d.get_info<sycl::info::device::name>().c_str());
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL init failed: %s\n", e.what());
        return 0;
    }
}

/* Free all cached model ranges to reclaim GPU memory for the next layer. */
static void sycl_free_cached_model_ranges(void) {
    if (!g_context) {
        for (auto &r : g_model_ranges) {
            if (r.device_ptr) sycl::free(r.device_ptr, *g_queue);
        }
    } else {
        for (auto &r : g_model_ranges) {
            if (r.device_ptr) sycl::free(r.device_ptr, *g_context);
        }
    }
    g_model_ranges.clear();
}

extern "C" void ds4_gpu_clear_cached_model_ranges(void) {
    if (g_queue) g_queue->wait_and_throw();
    sycl_free_cached_model_ranges();
}

extern "C" void ds4_gpu_cleanup(void) {
    sycl_free_cached_model_ranges();
    g_model_host_base       = nullptr;
    g_model_registered_size = 0;
    g_model_registered      = 0;

    sycl_stream_selected_cache_release_all();
    sycl_stream_expert_cache_release_all();

    delete g_queue;
    delete g_context;
    delete g_device;
    g_queue       = nullptr;
    g_context     = nullptr;
    g_device      = nullptr;
    g_initialized = 0;
}

/* =========================================================================
 * Tensor allocation / lifecycle
 * ========================================================================= */
extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    if (!g_queue) return nullptr;
    if (bytes == 0) bytes = 1;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return nullptr;
    try {
        t->ptr = sycl::malloc_device(bytes, *g_queue);
        if (!t->ptr) { free(t); return nullptr; }
    } catch (...) { free(t); return nullptr; }
    t->bytes = bytes;
    t->owner = 1;
    return t;
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t bytes) {
    /* Managed memory fallback: use host shared memory via malloc_host */
    if (!g_queue) return nullptr;
    if (bytes == 0) bytes = 1;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return nullptr;
    try {
        t->ptr = sycl::malloc_shared(bytes, *g_queue);
        if (!t->ptr) { free(t); return nullptr; }
    } catch (...) { free(t); return nullptr; }
    t->bytes = bytes;
    t->owner = 1;
    return t;
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base,
                                                uint64_t offset, uint64_t bytes) {
    if (!base || offset > base->bytes || bytes > base->bytes - offset) return nullptr;
    ds4_gpu_tensor *t = (ds4_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return nullptr;
    t->ptr   = (char *)base->ptr + offset;
    t->bytes = bytes;
    t->owner = 0;
    return t;
}

extern "C" void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor) {
    if (!tensor) return;
    if (tensor->owner && tensor->ptr && g_context) {
        sycl::free(tensor->ptr, *g_context);
    }
    free(tensor);
}

extern "C" uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor) {
    return tensor ? tensor->bytes : 0;
}

extern "C" void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor) {
    if (!tensor) return nullptr;
    try {
        if (g_queue) g_queue->wait();
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL tensor_contents failed: %s\n", e.what());
        return nullptr;
    }
    return tensor->ptr;
}

extern "C" int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor,
                                       float value, uint64_t count) {
    if (!tensor || !g_queue || count > tensor->bytes / sizeof(float)) return 0;
    if (count == 0) return 1;
    try {
        g_queue->fill((float *)tensor->ptr, value, count);
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL fill failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset,
                                    const void *data, uint64_t bytes) {
    if (!tensor || !data || !g_queue ||
        offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    try {
        g_queue->memcpy((char *)tensor->ptr + offset, data, bytes);
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL write failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor,
                                   uint64_t offset, void *data, uint64_t bytes) {
    if (!tensor || !data || !g_queue ||
        offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    try {
        g_queue->memcpy(data, (const char *)tensor->ptr + offset, bytes);
        g_queue->wait();
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL read failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                                   const ds4_gpu_tensor *src, uint64_t src_offset,
                                   uint64_t bytes) {
    if (!dst || !src || !g_queue ||
        dst_offset > dst->bytes || src_offset > src->bytes ||
        bytes > dst->bytes - dst_offset || bytes > src->bytes - src_offset) return 0;
    if (bytes == 0) return 1;
    try {
        g_queue->memcpy((char *)dst->ptr + dst_offset,
                        (const char *)src->ptr + src_offset, bytes);
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL copy failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_tensor_copy_f32_to_f16(ds4_gpu_tensor *dst, uint64_t dst_offset,
                                              const ds4_gpu_tensor *src, uint64_t src_offset,
                                              uint64_t count) {
    (void)dst; (void)dst_offset; (void)src; (void)src_offset; (void)count;
    fprintf(stderr, "ds4: SYCL f32->f16 copy not yet implemented\n");
    return 0;
}

/* =========================================================================
 * Command scheduling
 * ========================================================================= */
extern "C" int ds4_gpu_begin_commands(void) { return 1; }

extern "C" int ds4_gpu_flush_commands(void) {
    if (!g_queue) return 0;
    try { g_queue->wait(); return 1; }
    catch (...) { return 0; }
}

extern "C" int ds4_gpu_signal_selected_readback_ready(uint64_t *event_value) {
    if (!g_queue) return 0;
    try {
        g_queue->wait();
        if (event_value) *event_value = 1;
        return 1;
    } catch (...) { return 0; }
}

extern "C" int ds4_gpu_commit_and_wait_selected_readback(uint64_t event_value,
                                                          const char *label) {
    (void)event_value; (void)label;
    if (!g_queue) return 0;
    try { g_queue->wait(); return 1; }
    catch (...) { return 0; }
}

extern "C" int ds4_gpu_wait_selected_readback_ready(uint64_t event_value,
                                                     const char *label) {
    (void)event_value; (void)label;
    if (!g_queue) return 0;
    try { g_queue->wait(); return 1; }
    catch (...) { return 0; }
}

extern "C" int ds4_gpu_end_commands(void) {
    if (!g_queue) return 0;
    try { g_queue->wait_and_throw(); return 1; }
    catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL end_commands failed: %s\n", e.what());
        return 0;
    }
    catch (...) {
        fprintf(stderr, "ds4: SYCL end_commands failed (unknown exception)\n");
        return 0;
    }
}

extern "C" int ds4_gpu_synchronize(void) {
    if (!g_queue) return 0;
    try { g_queue->wait_and_throw(); return 1; }
    catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL sync failed: %s\n", e.what());
        return 0;
    }
}

/* =========================================================================
 * Model mapping helpers.
 * ========================================================================= */
extern "C" int ds4_gpu_set_model_map(const void *model_map, uint64_t model_size) {
    g_model_host_base       = model_map;
    g_model_registered_size = model_size;
    g_model_registered      = 1;
    return 1;
}

extern "C" int ds4_gpu_set_model_fd(int fd) {
    g_model_fd = fd;
    return 1;
}

extern "C" int ds4_gpu_set_model_fd_for_map(int fd, const void *model_map) {
    g_model_fd            = fd;
    g_model_host_base     = model_map;
    return 1;
}

extern "C" int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size,
                                           uint64_t map_offset, uint64_t map_size,
                                           uint64_t max_tensor_bytes) {
    (void)model_map; (void)model_size; (void)map_offset; (void)map_size; (void)max_tensor_bytes;
    return 1;
}

extern "C" int ds4_gpu_set_model_map_spans(const void *model_map, uint64_t model_size,
                                           const uint64_t *offsets, const uint64_t *sizes,
                                           uint32_t count, uint64_t max_tensor_bytes) {
    (void)model_map; (void)model_size; (void)offsets; (void)sizes;
    (void)count; (void)max_tensor_bytes;
    return 1;
}

extern "C" int ds4_gpu_cache_model_range(const void *model_map, uint64_t model_size,
                                         uint64_t offset, uint64_t bytes,
                                         const char *label) {
    if (!g_queue) return 0;
    if (!model_map || offset + bytes > model_size) return 0;
    (void)label;
    std::lock_guard<std::mutex> lock(g_model_ranges_mutex);
    /* Allocate + copy */
    try {
        char *dptr = (char *)sycl::malloc_device(bytes, *g_queue);
        if (!dptr) return 0;
        g_queue->memcpy(dptr, (const char *)model_map + offset, bytes);
        g_model_ranges.push_back({model_map, offset, bytes, dptr});
        return 1;
    } catch (...) { return 0; }
}

extern "C" int ds4_gpu_cache_q8_f16_range(const void *model_map, uint64_t model_size,
                                           uint64_t offset, uint64_t bytes,
                                           uint64_t in_dim, uint64_t out_dim,
                                           const char *label) {
    /* Same as cache_model_range for now; Q8->F16 on-the-fly dequant later */
    return ds4_gpu_cache_model_range(model_map, model_size, offset, bytes, label);
    (void)in_dim; (void)out_dim;
}

extern "C" int ds4_gpu_pro_q4_expert_table_auto_available(void) { return 0; }

extern "C" int ds4_gpu_preload_q4_expert_tables(
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint64_t gate_expert_bytes, uint64_t down_expert_bytes,
        uint32_t n_total_expert) {
    (void)model_map; (void)model_size;
    (void)gate_offset; (void)up_offset; (void)down_offset;
    (void)gate_expert_bytes; (void)down_expert_bytes;
    (void)n_total_expert;
    return 1;
}

extern "C" int ds4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes,
                                                    uint64_t context_bytes) {
    (void)kv_cache_bytes; (void)context_bytes;
    return 0;
}

extern "C" void ds4_gpu_set_quality(bool quality) {
    g_quality_mode = quality ? 1 : 0;
}

extern "C" void ds4_gpu_set_ssd_streaming(bool enabled) {
    g_ssd_streaming = enabled ? 1 : 0;
}

extern "C" void ds4_gpu_set_streaming_expert_cache_budget(uint32_t experts) {
    g_stream_expert_budget_override = experts;
    g_stream_expert_runtime_cap = 0;
    g_stream_expert_runtime_gate_bytes = 0;
    g_stream_expert_runtime_down_bytes = 0;
    g_stream_expert_memory_cap_notice = 0;
    sycl_stream_selected_cache_invalidate();
    sycl_stream_expert_cache_release_all();
}

extern "C" void ds4_gpu_set_streaming_expert_cache_expert_bytes(uint64_t bytes) {
    (void)bytes;
}

extern "C" uint64_t ds4_gpu_recommended_working_set_size(void) { return 0; }

extern "C" uint32_t ds4_gpu_stream_expert_cache_configured_count(void) {
    if (!sycl_stream_expert_cache_budget_visible_to_shared()) return 0;
    return sycl_stream_expert_cache_configured_budget();
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_current_count(void) {
    return g_stream_expert_cache.count;
}

extern "C" void ds4_gpu_stream_expert_cache_reset_route_hotness(void) {}
extern "C" void ds4_gpu_stream_expert_cache_release_resident(void) {
    sycl_stream_expert_cache_release_all();
}

extern "C" uint32_t ds4_gpu_stream_expert_cache_budget_for_expert_size(
        uint64_t gate_expert_bytes, uint64_t down_expert_bytes) {
    if (!sycl_stream_expert_cache_budget_visible_to_shared() ||
        sycl_stream_expert_cache_expert_bytes(gate_expert_bytes, down_expert_bytes) == 0)
        return 0;
    sycl_stream_expert_cache_note_size(gate_expert_bytes, down_expert_bytes);
    return sycl_stream_expert_cache_configured_budget();
}

/* De-duplicate selected_ids and load into global LRU cache */
extern "C" int ds4_gpu_stream_expert_cache_seed_selected(
        const ds4_gpu_stream_expert_table *table,
        const int32_t *selected_ids, uint32_t n_selected) {
    if (!table || !selected_ids || n_selected == 0) return 0;
    const void *model_map = table->model_map;
    uint64_t model_size = table->model_size;
    uint32_t layer = table->layer;
    uint32_t n_total_expert = table->n_total_expert;
    uint64_t gate_offset = table->gate_offset;
    uint64_t up_offset = table->up_offset;
    uint64_t down_offset = table->down_offset;
    uint64_t gate_expert_bytes = table->gate_expert_bytes;
    uint64_t down_expert_bytes = table->down_expert_bytes;
    if (n_total_expert == 0 || gate_expert_bytes == 0 || down_expert_bytes == 0)
        return 0;
    if (!sycl_stream_expert_cache_prepare(gate_expert_bytes, down_expert_bytes,
            sycl_stream_expert_cache_configured_budget()))
        return 1;
    for (uint32_t i = 0; i < n_selected; i++) {
        int32_t eid = selected_ids[i];
        if (eid < 0 || (uint32_t)eid >= n_total_expert) continue;
        if (!sycl_stream_expert_cache_seed_one(&g_stream_expert_cache,
                model_map, model_size, layer, n_total_expert, (uint32_t)eid,
                gate_offset, up_offset, down_offset,
                gate_expert_bytes, down_expert_bytes))
            return 1;
    }
    return 1;
}

/* De-duplicate selected_ids and load into compact selected buffer */
extern "C" int ds4_gpu_stream_expert_cache_begin_selected_load(
        const ds4_gpu_stream_expert_table *table,
        const int32_t *selected_ids, uint32_t n_selected) {
    if (!table || !selected_ids || n_selected == 0) return 0;
    const void *model_map = table->model_map;
    uint64_t model_size = table->model_size;
    uint32_t layer = table->layer;
    uint32_t n_total_expert = table->n_total_expert;
    uint64_t gate_offset = table->gate_offset;
    uint64_t up_offset = table->up_offset;
    uint64_t down_offset = table->down_offset;
    uint64_t gate_expert_bytes = table->gate_expert_bytes;
    uint64_t down_expert_bytes = table->down_expert_bytes;
    if (n_total_expert == 0 || gate_expert_bytes == 0 || down_expert_bytes == 0)
        return 0;

    /* De-duplicate selected_ids */
    std::vector<int32_t> compact;
    compact.reserve(n_selected);
    for (uint32_t i = 0; i < n_selected; i++) {
        int32_t eid = selected_ids[i];
        if (eid < 0 || (uint32_t)eid >= n_total_expert) continue;
        bool found = false;
        for (auto c : compact) { if (c == eid) { found = true; break; } }
        if (!found) compact.push_back(eid);
    }
    uint32_t compact_count = (uint32_t)compact.size();
    if (compact_count == 0) return 0;

    /* Build slot_ids mapping */
    std::vector<int32_t> slot_ids(n_selected);
    for (uint32_t i = 0; i < n_selected; i++) {
        int32_t eid = selected_ids[i];
        int32_t idx = -1;
        for (uint32_t j = 0; j < compact_count; j++) {
            if (compact[j] == eid) { idx = (int32_t)j; break; }
        }
        slot_ids[i] = (idx >= 0) ? idx : 0;
    }

    return sycl_stream_selected_cache_begin_compact_load(
        model_map, model_size, layer,
        compact.data(), slot_ids.data(),
        n_total_expert, compact_count, n_selected,
        gate_offset, up_offset, down_offset,
        gate_expert_bytes, down_expert_bytes,
        0 /* strict_failure */, 1 /* allow_global_cache */);
}

/* Seed cache from prioritized hotlist of expert IDs */
extern "C" int ds4_gpu_stream_expert_cache_seed_experts(
        const ds4_gpu_stream_expert_table *table,
        const int32_t *expert_ids,
        const uint32_t *expert_priorities,
        uint32_t n_experts) {
    if (!table || !expert_ids || n_experts == 0) return 0;
    const void *model_map = table->model_map;
    uint64_t model_size = table->model_size;
    uint32_t layer = table->layer;
    uint32_t n_total_expert = table->n_total_expert;
    uint64_t gate_offset = table->gate_offset;
    uint64_t up_offset = table->up_offset;
    uint64_t down_offset = table->down_offset;
    uint64_t gate_expert_bytes = table->gate_expert_bytes;
    uint64_t down_expert_bytes = table->down_expert_bytes;
    if (n_total_expert == 0 || gate_expert_bytes == 0 || down_expert_bytes == 0)
        return 0;
    sycl_stream_expert_cache *cache =
        sycl_stream_expert_cache_prepare(gate_expert_bytes, down_expert_bytes,
            sycl_stream_expert_cache_configured_budget());
    if (!cache) return 1;

    /* Select top experts by priority (or descending index order) */
    struct entry { uint32_t idx; uint32_t priority; };
    std::vector<entry> chosen;
    chosen.reserve(n_experts);
    for (uint32_t i = 0; i < n_experts; i++) {
        int32_t eid = expert_ids[i];
        if (eid < 0 || (uint32_t)eid >= n_total_expert) continue;
        uint32_t prio = expert_priorities ? expert_priorities[i] : (n_experts - i);
        if (chosen.empty()) {
            chosen.push_back({(uint32_t)eid, prio});
        } else {
            auto it = chosen.begin();
            while (it != chosen.end() && it->priority >= prio) ++it;
            chosen.insert(it, {(uint32_t)eid, prio});
            if (chosen.size() > cache->capacity) chosen.pop_back();
        }
    }
    for (auto it = chosen.rbegin(); it != chosen.rend(); ++it) {
        sycl_stream_expert_cache_seed_one(cache, model_map, model_size,
            layer, n_total_expert, it->idx,
            gate_offset, up_offset, down_offset,
            gate_expert_bytes, down_expert_bytes);
    }
    return 1;
}

extern "C" void ds4_gpu_print_memory_report(const char *label) {
    if (!g_device) return;
    size_t free_bytes  = 0, total_bytes = 0;
    try {
        free_bytes  = g_device->get_info<sycl::info::device::global_mem_size>();
        /* SYCL does not expose per-allocation free; report device total */
        fprintf(stderr, "ds4: SYCL %s — Device memory: %.2f GiB total\n",
                label ? label : "", (double)free_bytes / 1073741824.0);
    } catch (...) {}
    (void)label;
}

/* =========================================================================
 * Embedding + Indexer stubs
 * ========================================================================= */
extern "C" int ds4_gpu_embed_token_hc_tensor(
        ds4_gpu_tensor *out_hc, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t n_vocab, uint32_t token,
        uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !model_map || n_embd == 0 || n_hc == 0) return 0;
    uint64_t weight_elems = (uint64_t)n_vocab * n_embd;
    if (weight_offset > model_size || weight_elems > (model_size - weight_offset) / sizeof(uint16_t) ||
        out_hc->bytes < (uint64_t)n_hc * n_embd * sizeof(float)) return 0;
    const char *wptr = sycl_model_range_ptr(model_map, weight_offset, (uint64_t)n_embd * sizeof(uint16_t), "embed_token_hc");
    if (!wptr) return 0;
    if (token >= n_vocab) token = 0;
    try {
        uint32_t n = n_embd * n_hc;
        g_queue->parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            uint32_t i = (uint32_t)idx;
            uint32_t e = i % n_embd;
            ((float *)out_hc->ptr)[i] = (float)((const sycl::half *)wptr)[(uint64_t)token * n_embd + e];
        });
        g_queue->wait_and_throw();
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL embed_token_hc failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_embed_tokens_hc_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *tokens,
        const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t n_vocab, uint32_t n_tokens,
        uint32_t n_embd, uint32_t n_hc) {
    if (!out_hc || !tokens || !model_map || n_tokens == 0 || n_embd == 0 || n_hc == 0) return 0;
    uint64_t weight_elems = (uint64_t)n_vocab * n_embd;
    if (weight_offset > model_size || weight_elems > (model_size - weight_offset) / sizeof(uint16_t) ||
        out_hc->bytes < (uint64_t)n_tokens * n_hc * n_embd * sizeof(float) ||
        tokens->bytes < (uint64_t)n_tokens * sizeof(int32_t)) return 0;
    const char *wptr = sycl_model_range_ptr(model_map, weight_offset, weight_elems * sizeof(uint16_t), "embed_tokens_hc");
    if (!wptr) return 0;
    try {
        uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
        g_queue->parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            uint64_t gid = (uint64_t)idx;
            uint32_t d = (uint32_t)(gid % n_embd);
            uint64_t tmp = gid / n_embd;
            uint32_t t = (uint32_t)(tmp / n_hc);
            int32_t tok_i = ((const int32_t *)tokens->ptr)[t];
            uint32_t tok = (tok_i < 0) ? 0 : (uint32_t)tok_i;
            if (tok >= n_vocab) tok = 0;
            ((float *)out_hc->ptr)[gid] = (float)((const sycl::half *)wptr)[(uint64_t)tok * n_embd + d];
        });
        g_queue->wait_and_throw();
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL embed_tokens_hc failed: %s\n", e.what());
        return 0;
    }
}

/* ---- Indexer score computation: scores[t][c] = sum_h max(dot(q[t][h], index_comp[c]), 0) * weights[t][h] * scale ---- */
static void indexer_scores_launch(
        ds4_gpu_tensor *scores, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights, const ds4_gpu_tensor *index_comp,
        uint32_t n_comp, uint32_t n_tokens, uint32_t pos0,
        uint32_t n_head, uint32_t head_dim, uint32_t ratio, float scale,
        int causal) {
    if (!scores || !q || !weights || !index_comp ||
        n_comp == 0 || n_tokens == 0 || n_head == 0 || head_dim == 0 ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        weights->bytes < (uint64_t)n_tokens * n_head * sizeof(float) ||
        index_comp->bytes < (uint64_t)n_comp * head_dim * sizeof(float) ||
        scores->bytes < (uint64_t)n_tokens * n_comp * sizeof(float)) return;
    sycl::queue *qptr = g_queue;
    if (!qptr) return;
    const float *q_data = (const float *)q->ptr;
    const float *w_data = (const float *)weights->ptr;
    const float *kc_data = (const float *)index_comp->ptr;
    float *s_data = (float *)scores->ptr;
    qptr->submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<2>(n_tokens, n_comp), [=](sycl::id<2> idx) {
            uint32_t t = idx[0];
            uint32_t c = idx[1];
            if (causal) {
                uint32_t n_visible = (pos0 + t + 1u) / ratio;
                if (c >= n_visible) {
                    s_data[(uint64_t)t * n_comp + c] = -INFINITY;
                    return;
                }
            }
            float total = 0.0f;
            for (uint32_t h = 0; h < n_head; h++) {
                const float *qh = q_data + ((uint64_t)t * n_head + h) * head_dim;
                const float *kh = kc_data + (uint64_t)c * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kh[d];
                total += sycl::fmax(dot, 0.0f) * w_data[(uint64_t)t * n_head + h];
            }
            s_data[(uint64_t)t * n_comp + c] = total * scale;
        });
    }).wait();
}

extern "C" int ds4_gpu_indexer_score_one_tensor(
        ds4_gpu_tensor *scores, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights, const ds4_gpu_tensor *index_comp,
        uint32_t n_comp, uint32_t n_head, uint32_t head_dim, float scale) {
    indexer_scores_launch(scores, q, weights, index_comp, n_comp, 1, 0,
                          n_head, head_dim, 1, scale, 0);
    return 1;
}

extern "C" int ds4_gpu_indexer_scores_prefill_tensor(
        ds4_gpu_tensor *scores, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights, const ds4_gpu_tensor *index_comp,
        uint32_t n_comp, uint32_t n_tokens, uint32_t n_head,
        uint32_t head_dim, uint32_t ratio, float scale) {
    indexer_scores_launch(scores, q, weights, index_comp, n_comp, n_tokens, 0,
                          n_head, head_dim, ratio, scale, 1);
    return 1;
}

extern "C" int ds4_gpu_indexer_scores_decode_batch_tensor(
        ds4_gpu_tensor *scores, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights, const ds4_gpu_tensor *index_comp,
        uint32_t n_comp, uint32_t n_tokens, uint32_t pos0,
        uint32_t n_head, uint32_t head_dim, uint32_t ratio, float scale) {
    indexer_scores_launch(scores, q, weights, index_comp, n_comp, n_tokens, pos0,
                          n_head, head_dim, ratio, scale, 1);
    return 1;
}

/* ---- Top-k: bitonic sort per token to find top_k largest scores ---- */
extern "C" int ds4_gpu_indexer_topk_tensor(
        ds4_gpu_tensor *selected, const ds4_gpu_tensor *scores,
        uint32_t n_comp, uint32_t n_tokens, uint32_t top_k) {
    if (!selected || !scores || n_comp == 0 || n_tokens == 0 || top_k == 0 ||
        top_k > n_comp ||
        scores->bytes < (uint64_t)n_tokens * n_comp * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * top_k * sizeof(uint32_t)) return 0;
    sycl::queue *qptr = g_queue;
    if (!qptr) return 0;
    const float *s_data = (const float *)scores->ptr;
    uint32_t *sel_data = (uint32_t *)selected->ptr;
    /* Simple insertion sort per token: for each score, insert into top-k list */
    qptr->submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(n_tokens), [=](sycl::id<1> tid) {
            uint32_t t = tid[0];
            const float *row = s_data + (uint64_t)t * n_comp;
            uint32_t *sel = sel_data + (uint64_t)t * top_k;
            /* Initialize with first k elements */
            uint32_t kk = top_k < n_comp ? top_k : n_comp;
            for (uint32_t i = 0; i < kk; i++) sel[i] = i;
            /* Sort initial list by score descending (simple insertion) */
            for (uint32_t i = 1; i < kk; i++) {
                uint32_t j = i;
                while (j > 0 && (row[sel[j]] > row[sel[j - 1]] ||
                       (row[sel[j]] == row[sel[j - 1]] && sel[j] < sel[j - 1]))) {
                    uint32_t tmp = sel[j];
                    sel[j] = sel[j - 1];
                    sel[j - 1] = tmp;
                    j--;
                }
            }
            /* Process remaining elements */
            for (uint32_t c = kk; c < n_comp; c++) {
                float v = row[c];
                /* If better than the worst in top-k */
                if (v > row[sel[kk - 1]] || (v == row[sel[kk - 1]] && c < sel[kk - 1])) {
                    /* Insert c into sorted position */
                    int32_t ins = (int32_t)kk - 1;
                    while (ins > 0 && (v > row[sel[ins - 1]] ||
                           (v == row[sel[ins - 1]] && c < sel[ins - 1]))) {
                        sel[ins] = sel[ins - 1];
                        ins--;
                    }
                    sel[ins] = c;
                }
            }
        });
    }).wait();
    return 1;
}

/* ---- Argmax: tree reduction over vocab ---- */
extern "C" int ds4_gpu_argmax_tensor(
        ds4_gpu_tensor *out_idx, const ds4_gpu_tensor *logits,
        uint32_t n_vocab) {
    if (!out_idx || !logits || n_vocab == 0 ||
        out_idx->bytes < sizeof(int32_t) ||
        logits->bytes < (uint64_t)n_vocab * sizeof(float)) return 0;
    sycl::queue *qptr = g_queue;
    if (!qptr) return 0;
    const float *l_data = (const float *)logits->ptr;
    int32_t *idx_data = (int32_t *)out_idx->ptr;
    enum { THREADS = 1024 };
    qptr->submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> sm_val(THREADS, h);
        sycl::local_accessor<int32_t, 1> sm_idx(THREADS, h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(THREADS), sycl::range<1>(THREADS)),
            [=](sycl::nd_item<1> item) {
            uint32_t tid = item.get_local_linear_id();
            float local_v = -INFINITY;
            int32_t local_i = 0;
            for (uint32_t i = tid; i < n_vocab; i += THREADS) {
                float v = l_data[i];
                if (v > local_v) { local_v = v; local_i = (int32_t)i; }
            }
            sm_val[tid] = local_v;
            sm_idx[tid] = local_i;
            item.barrier(sycl::access::fence_space::local_space);
            for (uint32_t s = THREADS / 2u; s > 0u; s >>= 1) {
                if (tid < s) {
                    float vr = sm_val[tid + s];
                    int32_t ir = sm_idx[tid + s];
                    float vl = sm_val[tid];
                    int32_t il = sm_idx[tid];
                    bool take_right = (vr > vl) || (vr == vl && ir < il);
                    if (take_right) { sm_val[tid] = vr; sm_idx[tid] = ir; }
                }
                item.barrier(sycl::access::fence_space::local_space);
            }
            if (tid == 0) *idx_data = sm_idx[0];
        });
    }).wait();
    return 1;
}

/* ---- Top-k mask: for each (t,c), set 0 if c is in top-k list, else -INF ---- */
extern "C" int ds4_gpu_dsv4_topk_mask_tensor(
        ds4_gpu_tensor *mask, const ds4_gpu_tensor *topk,
        uint32_t n_comp, uint32_t n_tokens, uint32_t top_k) {
    if (!mask || !topk || n_comp == 0 || n_tokens == 0 || top_k == 0 ||
        mask->bytes < (uint64_t)n_tokens * n_comp * sizeof(float) ||
        topk->bytes < (uint64_t)n_tokens * top_k * sizeof(uint32_t)) return 0;
    sycl::queue *qptr = g_queue;
    if (!qptr) return 0;
    float *m_data = (float *)mask->ptr;
    const uint32_t *tk_data = (const uint32_t *)topk->ptr;
    qptr->submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<2>(n_tokens, n_comp), [=](sycl::id<2> idx) {
            uint32_t t = idx[0];
            uint32_t c = idx[1];
            float v = -INFINITY;
            for (uint32_t k = 0; k < top_k; k++) {
                if (tk_data[(uint64_t)t * top_k + k] == c) { v = 0.0f; break; }
            }
            m_data[(uint64_t)t * n_comp + c] = v;
        });
    }).wait();
    return 1;
}

/* ---- DSv4 indexer QAT (quantization-aware training) no-op ---- */
extern "C" int ds4_gpu_dsv4_indexer_qat_tensor(
        ds4_gpu_tensor *x, uint32_t n_rows, uint32_t head_dim) {
    (void)x; (void)n_rows; (void)head_dim;
    return 1; /* no-op */
}

/* =========================================================================
 * Dense matmul stubs
 * ========================================================================= */
/* Decode an IEEE 754 binary16 stored in a uint16_t (the q8_0 block scale). */
static float sycl_half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (uint32_t)((h >> 10) & 0x1f);
    uint32_t mant = (uint32_t)(h & 0x3ff);
    if (exp == 0) {
        /* subnormal */
        if (mant == 0) return 0.0f;
        exp = 1u << 23u;
        while ((mant & 0x400u) == 0) { mant <<= 1; exp -= (1u << 23u); }
        mant &= 0x3ffu;
        mant <<= 13u;
        exp += 127u << 23u;
    } else if (exp == 31) {
        exp  = 255u << 23u;
        mant <<= 13u;
    } else {
        exp  = (exp + 112u) << 23u;
        mant <<= 13u;
    }
    uint32_t bits = sign | exp | mant;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* Quantise f32 activations to Q8_0 format (int8 + scale per 32-element block).
   xq: (n_tok * blocks_per_row * 32) int8 buffer
   xscale: (n_tok * blocks_per_row) float buffer */
static void sycl_quantize_q8_0(sycl::queue &q,
                                int8_t *xq, float *xscale,
                                const float *x, uint64_t in_dim,
                                uint64_t blocks_per_row, uint64_t n_tok) {
    q.parallel_for(sycl::range<2>(blocks_per_row, n_tok), [=](sycl::id<2> idx) {
        uint64_t b = (uint64_t)idx[0];
        uint64_t tok = (uint64_t)idx[1];
        uint64_t i0 = b * 32;
        uint64_t bn = (in_dim - i0 < 32) ? (in_dim - i0) : 32;
        const float *xr = x + tok * in_dim + i0;
        float amax = 0.0f;
        for (uint64_t i = 0; i < bn; i++) {
            float v = xr[i];
            if (v < 0) v = -v;
            if (v > amax) amax = v;
        }
        float d = amax / 127.0f;
        float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
        xscale[tok * blocks_per_row + b] = d;
        int8_t *dst = xq + (tok * blocks_per_row + b) * 32;
        for (uint64_t i = 0; i < bn; i++) {
            float scaled = xr[i] * id;
            int v = (int)(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
            if (v > 127) v = 127;
            else if (v < -128) v = -128;
            dst[i] = (int8_t)v;
        }
        for (uint64_t i = bn; i < 32; i++) dst[i] = 0;
    });
}

/* Sub-group optimised pre-quantised Q8_0 matmul: 8 rows per work-group,
   8 sub-groups (sg size 16), each sub-group handles one row.
   Each lane handles blocks round-robin and the sub-group reduces the
   per-block dot products. */
static void sycl_matmul_q8_0_preq_sg(sycl::queue &q,
                                      float *out,
                                      const uint8_t *w8,
                                      const int8_t *xq,
                                      const float *xscale,
                                      uint64_t out_dim,
                                      uint64_t blocks_per_row) {
    sycl::range<1> global(16 * ((out_dim + 15) / 16));
    sycl::range<1> local(16);
    q.parallel_for(sycl::nd_range<1>(global, local), [=](sycl::nd_item<1> item) {
        uint64_t row_base = item.get_group(0) * 16;
        uint64_t sg_id = item.get_sub_group().get_group_id();
        uint64_t row = row_base + sg_id;
        if (row >= out_dim) return;

        auto sg = item.get_sub_group();
        uint32_t lane = sg.get_local_id();
        float acc = 0.0f;
        for (uint64_t b = lane; b < blocks_per_row; b += 16) {
            const uint8_t *block = w8 + (row * blocks_per_row + b) * 34;
            uint16_t d_bits = (uint16_t)block[0] | ((uint16_t)block[1] << 8);
            float wscale = sycl_half_to_float(d_bits);
            const int8_t *wq = (const int8_t *)(block + 2);
            const int8_t *xqb = xq + b * 32;
            int dot = 0;
#pragma unroll
            for (int i = 0; i < 32; i++) dot += (int)wq[i] * (int)xqb[i];
            acc += wscale * xscale[b] * (float)dot;
        }
        acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
        if (lane == 0) out[row] = acc;
    });
}

/* Batched pre-quantised Q8_0 matmul with sub-group reduction.
   Work-group: 16 rows per group (one sub-group per row), token dim via blockIdx.y.
   Work-group size 16*1=16 matches sub-group size 16 so all lanes are active. */
static void sycl_matmul_q8_0_preq_batch_sg(sycl::queue &q,
                                             float *out,
                                             const uint8_t *w8,
                                             const int8_t *xq,
                                             const float *xscale,
                                             uint64_t out_dim,
                                             uint64_t blocks_per_row,
                                             uint64_t n_tok) {
    sycl::range<2> global(16 * ((out_dim + 15) / 16), n_tok);
    sycl::range<2> local(16, 1);
    q.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
        uint64_t row_base = (uint64_t)item.get_group(0) * 8;
        uint64_t tok = (uint64_t)item.get_group(1);
        uint64_t sg_id = item.get_sub_group().get_group_id();
        uint64_t row = row_base + sg_id;
        if (row >= out_dim) return;

        auto sg = item.get_sub_group();
        uint32_t lane = sg.get_local_id();
        const int8_t *xqr = xq + tok * blocks_per_row * 32;
        const float *xsr = xscale + tok * blocks_per_row;
        float acc = 0.0f;
        for (uint64_t b = lane; b < blocks_per_row; b += 16) {
            const uint8_t *block = w8 + (row * blocks_per_row + b) * 34;
            uint16_t d_bits = (uint16_t)block[0] | ((uint16_t)block[1] << 8);
            float wscale = sycl_half_to_float(d_bits);
            const int8_t *wq = (const int8_t *)(block + 2);
            const int8_t *xqb = xqr + b * 32;
            int dot = 0;
#pragma unroll
            for (int i = 0; i < 32; i++) dot += (int)wq[i] * (int)xqb[i];
            acc += wscale * xsr[b] * (float)dot;
        }
        acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
        if (lane == 0) out[tok * out_dim + row] = acc;
    });
}

extern "C" int ds4_gpu_matmul_q8_0_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    uint64_t blocks_per_row = (in_dim + 31) / 32;
    if (weight_offset > model_size || out_dim > UINT64_MAX / (blocks_per_row * 34)) return 0;
    uint64_t weight_bytes = out_dim * blocks_per_row * 34;
    if (weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const char *wptr = sycl_model_range_ptr(model_map, weight_offset, weight_bytes, "q8_0");
    if (!wptr) return 0;
    const uint8_t *w8 = (const uint8_t *)wptr;
    try {
        if (n_tok > 1 && blocks_per_row <= 32) {
            /* Pre-quantise + sub-group batch kernel for small block count. */
            uint64_t xq_bytes = n_tok * blocks_per_row * 32;
            int8_t *xq = (int8_t *)sycl::malloc_device(xq_bytes, *g_queue);
            float *xscale = (float *)sycl::malloc_device(n_tok * blocks_per_row * sizeof(float), *g_queue);
            if (!xq || !xscale) {
                sycl::free(xq, *g_queue); sycl::free(xscale, *g_queue);
                return 0;
            }
            sycl_quantize_q8_0(*g_queue, xq, xscale, (const float *)x->ptr,
                               in_dim, blocks_per_row, n_tok);
            sycl_matmul_q8_0_preq_batch_sg(*g_queue, (float *)out->ptr, w8,
                                            xq, xscale, out_dim, blocks_per_row, n_tok);
            g_queue->wait_and_throw();
            sycl::free(xq, *g_queue);
            sycl::free(xscale, *g_queue);
        } else if (n_tok > 1) {
            /* Many blocks: dequant weights to fp32 then oneMKL gemm. */
            uint64_t wf32_count = out_dim * in_dim;
            float *wf32 = (float *)sycl::malloc_device(wf32_count * sizeof(float), *g_queue);
            if (!wf32) return 0;
            g_queue->parallel_for(sycl::range<1>(out_dim * blocks_per_row), [=](sycl::id<1> idx) {
                uint64_t bid = (uint64_t)idx;
                uint64_t brow = bid / blocks_per_row;
                uint64_t bcol = bid % blocks_per_row;
                const uint8_t *block = w8 + (brow * blocks_per_row + bcol) * 34;
                uint16_t d_bits = (uint16_t)block[0] | ((uint16_t)block[1] << 8);
                float d = sycl_half_to_float(d_bits);
                const int8_t *qs = (const int8_t *)(block + 2);
                float *row_out = wf32 + brow * in_dim + bcol * 32;
                for (int i = 0; i < 32; i++) row_out[i] = d * (float)qs[i];
            });
            g_queue->wait_and_throw();
            const float alpha = 1.0f, beta = 0.0f;
            oneapi::mkl::blas::gemm(*g_queue,
                                    oneapi::mkl::transpose::trans,
                                    oneapi::mkl::transpose::nontrans,
                                    (int64_t)out_dim, (int64_t)n_tok, (int64_t)in_dim,
                                    alpha,
                                    wf32, (int64_t)in_dim,
                                    (const float *)x->ptr, (int64_t)in_dim,
                                    beta,
                                    (float *)out->ptr, (int64_t)out_dim);
            g_queue->wait_and_throw();
            sycl::free(wf32, *g_queue);
        } else {
            /* Single token: pre-quantise + sub-group kernel. */
            int8_t *xq = (int8_t *)sycl::malloc_device(blocks_per_row * 32, *g_queue);
            float *xscale = (float *)sycl::malloc_device(blocks_per_row * sizeof(float), *g_queue);
            if (!xq || !xscale) {
                sycl::free(xq, *g_queue); sycl::free(xscale, *g_queue);
                return 0;
            }
            sycl_quantize_q8_0(*g_queue, xq, xscale, (const float *)x->ptr,
                               in_dim, blocks_per_row, 1);
            sycl_matmul_q8_0_preq_sg(*g_queue, (float *)out->ptr, w8,
                                      xq, xscale, out_dim, blocks_per_row);
            g_queue->wait_and_throw();
            sycl::free(xq, *g_queue);
            sycl::free(xscale, *g_queue);
        }
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL matmul_q8_0 failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_matmul_q8_0_pair_tensor(
        ds4_gpu_tensor *out0, ds4_gpu_tensor *out1,
        const void *model_map, uint64_t model_size,
        uint64_t weight0_offset, uint64_t weight1_offset,
        uint64_t in_dim, uint64_t out0_dim, uint64_t out1_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out0 || !out1 || !x || !model_map || in_dim == 0 || out0_dim == 0 || out1_dim == 0 || n_tok == 0) return 0;
    return ds4_gpu_matmul_q8_0_tensor(out0, model_map, model_size, weight0_offset,
                                       in_dim, out0_dim, x, n_tok) &&
           ds4_gpu_matmul_q8_0_tensor(out1, model_map, model_size, weight1_offset,
                                       in_dim, out1_dim, x, n_tok);
}

extern "C" int ds4_gpu_matmul_q8_0_f16_out_tensor(
        ds4_gpu_tensor *out_h, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok) {
    (void)out_h; (void)model_map; (void)model_size; (void)weight_offset;
    (void)in_dim; (void)out_dim; (void)x; (void)n_tok;
    return 0;
}

extern "C" int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
        ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid, const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset,
        uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, float clamp) {
    return ds4_gpu_matmul_q8_0_pair_tensor(gate, up,
                                             model_map, model_size,
                                             gate_offset, up_offset,
                                             in_dim, out_dim, out_dim,
                                             x, 1) &&
           ds4_gpu_swiglu_tensor(mid, gate, up, (uint32_t)out_dim, clamp, 1.0f);
}

extern "C" int ds4_gpu_matmul_f16_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (weight_offset > model_size || out_dim > UINT64_MAX / in_dim) return 0;
    uint64_t weight_elems = out_dim * in_dim;
    if (weight_elems > UINT64_MAX / sizeof(sycl::half)) return 0;
    uint64_t weight_bytes = weight_elems * sizeof(sycl::half);
    if (weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const char *wptr = sycl_model_range_ptr(model_map, weight_offset, weight_bytes, "f16");
    if (!wptr) return 0;
    const sycl::half *w = (const sycl::half *)wptr;
    try {
        if (n_tok > 1) {
            const uint64_t xh_count = n_tok * in_dim;
            sycl::half *xh = (sycl::half *)sycl::malloc_device(xh_count * sizeof(sycl::half), *g_queue);
            if (!xh) return 0;
            sycl_convert_f32_f16(*g_queue, xh_count, (const float *)x->ptr, xh);
            const float alpha = 1.0f, beta = 0.0f;
            oneapi::mkl::blas::gemm(*g_queue,
                                    oneapi::mkl::transpose::trans,
                                    oneapi::mkl::transpose::nontrans,
                                    (int64_t)out_dim, (int64_t)n_tok, (int64_t)in_dim,
                                    alpha,
                                    w, (int64_t)in_dim,
                                    xh, (int64_t)in_dim,
                                    beta,
                                    (float *)out->ptr, (int64_t)out_dim);
            g_queue->wait_and_throw();
            sycl::free(xh, *g_queue);
        } else {
            g_queue->parallel_for(sycl::range<1>(out_dim), [=](sycl::id<1> idx) {
                uint64_t o = (uint64_t)idx;
                float sum = 0.0f;
                for (uint64_t i = 0; i < in_dim; i++)
                    sum += (float)w[o * in_dim + i] * ((const float *)x->ptr)[i];
                ((float *)out->ptr)[o] = sum;
            });
            g_queue->wait_and_throw();
        }
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL matmul_f16 failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_matmul_f16_pair_tensor(
        ds4_gpu_tensor *out_a, ds4_gpu_tensor *out_b,
        const void *model_map, uint64_t model_size,
        uint64_t weight_a_offset, uint64_t weight_b_offset,
        uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out_a || !out_b || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    return ds4_gpu_matmul_f16_tensor(out_a, model_map, model_size, weight_a_offset,
                                      in_dim, out_dim, x, n_tok) &&
           ds4_gpu_matmul_f16_tensor(out_b, model_map, model_size, weight_b_offset,
                                      in_dim, out_dim, x, n_tok);
}

extern "C" int ds4_gpu_matmul_f32_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok) {
    if (!out || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (weight_offset > model_size || out_dim > UINT64_MAX / in_dim) return 0;
    uint64_t weight_elems = out_dim * in_dim;
    if (weight_elems > UINT64_MAX / sizeof(float)) return 0;
    uint64_t weight_bytes = weight_elems * sizeof(float);
    if (weight_bytes > model_size - weight_offset) return 0;
    if (x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) return 0;
    const char *wptr = sycl_model_range_ptr(model_map, weight_offset, weight_bytes, "f32");
    if (!wptr) return 0;
    const float *w = (const float *)wptr;
    try {
        if (n_tok > 1) {
            const float alpha = 1.0f, beta = 0.0f;
            oneapi::mkl::blas::gemm(*g_queue,
                                    oneapi::mkl::transpose::trans,
                                    oneapi::mkl::transpose::nontrans,
                                    (int64_t)out_dim, (int64_t)n_tok, (int64_t)in_dim,
                                    alpha,
                                    w, (int64_t)in_dim,
                                    (const float *)x->ptr, (int64_t)in_dim,
                                    beta,
                                    (float *)out->ptr, (int64_t)out_dim);
        } else {
            g_queue->parallel_for(sycl::range<1>(out_dim), [=](sycl::id<1> idx) {
                uint64_t o = (uint64_t)idx;
                float sum = 0.0f;
                for (uint64_t i = 0; i < in_dim; i++)
                    sum += w[o * in_dim + i] * ((const float *)x->ptr)[i];
                ((float *)out->ptr)[o] = sum;
            });
        }
        g_queue->wait_and_throw();
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL matmul_f32 failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_repeat_hc_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *row,
        uint32_t n_embd, uint32_t n_hc) {
    if (!out || !row || n_embd == 0 || n_hc == 0 ||
        row->bytes < (uint64_t)n_embd * sizeof(float) ||
        out->bytes < (uint64_t)n_embd * n_hc * sizeof(float)) return 0;
    try {
        uint64_t n = (uint64_t)n_embd * n_hc;
        g_queue->parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            ((float *)out->ptr)[i] = ((const float *)row->ptr)[i % n_embd];
        });
        g_queue->wait_and_throw();
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL repeat_hc failed: %s\n", e.what());
        return 0;
    }
}

/* =========================================================================
 * RMS normalisation
 * ========================================================================= */
extern "C" int ds4_gpu_rms_norm_plain_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        uint32_t n, float eps) {
    if (!out || !x || out->bytes < (uint64_t)n * sizeof(float) ||
        x->bytes < (uint64_t)n * sizeof(float)) return 0;
    try {
        g_queue->parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
            float sum = 0.0f;
            const float *xr = (const float *)x->ptr;
            for (uint32_t i = 0; i < n; i++) sum += xr[i] * xr[i];
            float scale = sycl::rsqrt(sum / (float)n + eps);
            float *orow = (float *)out->ptr;
            for (uint32_t i = 0; i < n; i++) orow[i] = xr[i] * scale;
        });
        g_queue->wait_and_throw();
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL rms_norm_plain failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_rms_norm_plain_rows_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        uint32_t n, uint32_t rows, float eps) {
    if (!out || !x || out->bytes < (uint64_t)n * rows * sizeof(float) ||
        x->bytes < (uint64_t)n * rows * sizeof(float)) return 0;
    try {
        g_queue->parallel_for(sycl::range<1>(rows), [=](sycl::id<1> idx) {
            uint32_t row = (uint32_t)idx;
            const float *xr = (const float *)x->ptr + (uint64_t)row * n;
            float *orow = (float *)out->ptr + (uint64_t)row * n;
            float sum = 0.0f;
            for (uint32_t i = 0; i < n; i++) sum += xr[i] * xr[i];
            float scale = sycl::rsqrt(sum / (float)n + eps);
            for (uint32_t i = 0; i < n; i++) orow[i] = xr[i] * scale;
        });
        g_queue->wait_and_throw();
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL rms_norm_plain_rows failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_rms_norm_weight_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t n, float eps) {
    if (!out || !x || !model_map || weight_offset > model_size ||
        model_size - weight_offset < (uint64_t)n * sizeof(float) ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        x->bytes < (uint64_t)n * sizeof(float)) return 0;
    const char *wptr = sycl_model_range_ptr(model_map, weight_offset, (uint64_t)n * sizeof(float), "rms_weight");
    if (!wptr) return 0;
    const float *w = (const float *)wptr;
    try {
        g_queue->parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {
            float sum = 0.0f;
            const float *xr = (const float *)x->ptr;
            for (uint32_t i = 0; i < n; i++) sum += xr[i] * xr[i];
            float scale = sycl::rsqrt(sum / (float)n + eps);
            float *orow = (float *)out->ptr;
            for (uint32_t i = 0; i < n; i++) orow[i] = xr[i] * scale * w[i];
        });
        g_queue->wait_and_throw();
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL rms_norm_weight failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_rms_norm_weight_rows_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint32_t n, uint32_t rows, float eps) {
    if (!out || !x || !model_map || weight_offset > model_size ||
        model_size - weight_offset < (uint64_t)n * sizeof(float) ||
        out->bytes < (uint64_t)n * rows * sizeof(float) ||
        x->bytes < (uint64_t)n * rows * sizeof(float)) return 0;
    const char *wptr = sycl_model_range_ptr(model_map, weight_offset, (uint64_t)n * sizeof(float), "rms_weight");
    if (!wptr) return 0;
    const float *w = (const float *)wptr;
    try {
        g_queue->parallel_for(sycl::range<1>(rows), [=](sycl::id<1> idx) {
            uint32_t row = (uint32_t)idx;
            const float *xr = (const float *)x->ptr + (uint64_t)row * n;
            float *orow = (float *)out->ptr + (uint64_t)row * n;
            float sum = 0.0f;
            for (uint32_t i = 0; i < n; i++) sum += xr[i] * xr[i];
            float scale = sycl::rsqrt(sum / (float)n + eps);
            for (uint32_t i = 0; i < n; i++) orow[i] = xr[i] * scale * w[i];
        });
        g_queue->wait_and_throw();
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL rms_norm_weight_rows failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
        ds4_gpu_tensor *q_out, const ds4_gpu_tensor *q,
        const void *model_map, uint64_t model_size,
        uint64_t q_weight_offset, uint32_t q_n,
        ds4_gpu_tensor *kv_out, const ds4_gpu_tensor *kv,
        uint64_t kv_weight_offset, uint32_t kv_n,
        uint32_t rows, float eps) {
    if (!q_out || !q || !kv_out || !kv || !model_map ||
        q_weight_offset > model_size || kv_weight_offset > model_size ||
        q_out->bytes < (uint64_t)q_n * rows * sizeof(float) ||
        q->bytes < (uint64_t)q_n * rows * sizeof(float) ||
        kv_out->bytes < (uint64_t)kv_n * rows * sizeof(float) ||
        kv->bytes < (uint64_t)kv_n * rows * sizeof(float)) return 0;
    return ds4_gpu_rms_norm_weight_rows_tensor(q_out, q, model_map, model_size,
                                                q_weight_offset, q_n, rows, eps) &&
           ds4_gpu_rms_norm_weight_rows_tensor(kv_out, kv, model_map, model_size,
                                                kv_weight_offset, kv_n, rows, eps);
}

/* Yarn ramp helper for NTK-aware RoPE scaling. */
static float rope_yarn_ramp(float low, float high, int i0) {
    float y = ((float)(i0 / 2) - low) / sycl::fmax(0.001f, high - low);
    return 1.0f - sycl::fmin(1.0f, sycl::fmax(0.0f, y));
}

extern "C" int ds4_gpu_head_rms_norm_tensor(
        ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head,
        uint32_t head_dim, float eps) {
    if (!x || x->bytes < (uint64_t)n_tok * n_head * head_dim * sizeof(float)) return 0;
    try {
        uint32_t rows = n_tok * n_head;
        g_queue->parallel_for(sycl::range<1>(rows), [=](sycl::id<1> idx) {
            uint32_t row = (uint32_t)idx;
            float *xr = (float *)x->ptr + (uint64_t)row * head_dim;
            float sum = 0.0f;
            for (uint32_t i = 0; i < head_dim; i++) sum += xr[i] * xr[i];
            float scale = sycl::rsqrt(sum / (float)head_dim + eps);
            for (uint32_t i = 0; i < head_dim; i++) xr[i] *= scale;
        });
        g_queue->wait_and_throw();
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL head_rms_norm failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_head_rms_norm_rope_tail_tensor(
        ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head,
        uint32_t head_dim, uint32_t n_rot, uint32_t pos0,
        uint32_t n_ctx_orig, bool inverse, float freq_base,
        float freq_scale, float ext_factor, float attn_factor,
        float beta_fast, float beta_slow, float eps) {
    if (!x || n_rot > head_dim || (n_rot & 1) ||
        x->bytes < (uint64_t)n_tok * n_head * head_dim * sizeof(float)) return 0;
    try {
        uint32_t rows = n_tok * n_head;
        g_queue->parallel_for(sycl::range<1>(rows), [=](sycl::id<1> idx) {
            uint32_t row = (uint32_t)idx;
            uint32_t t = row / n_head;
            float *xr = (float *)x->ptr + (uint64_t)row * head_dim;
            float sum = 0.0f;
            for (uint32_t i = 0; i < head_dim; i++) sum += xr[i] * xr[i];
            const float scale = sycl::rsqrt(sum / (float)head_dim + eps);
            const uint32_t n_nope = head_dim - n_rot;
            for (uint32_t i = 0; i < n_nope; i++) xr[i] *= scale;

            float corr0 = 0.0f, corr1 = 0.0f;
            if (ext_factor != 0.0f) {
                float denom = 2.0f * sycl::log(freq_base);
                corr0 = sycl::floor((float)n_rot * sycl::log((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
                corr1 = sycl::ceil((float)n_rot * sycl::log((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
                corr0 = sycl::fmax(0.0f, corr0);
                corr1 = sycl::fmin((float)(n_rot - 1), corr1);
            }
            float *tail = xr + n_nope;
            for (uint32_t pair = 0; pair < n_rot / 2; pair++) {
                uint32_t i = pair * 2u;
                float theta_extrap = (float)(pos0 + t) * sycl::pow(freq_base, -((float)i) / (float)n_rot);
                float theta_interp = freq_scale * theta_extrap;
                float theta = theta_interp;
                float mscale = attn_factor;
                if (ext_factor != 0.0f) {
                    float ramp_mix = rope_yarn_ramp(corr0, corr1, (int)i) * ext_factor;
                    theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
                    mscale *= 1.0f + 0.1f * sycl::log(1.0f / freq_scale);
                }
                float c = sycl::cos(theta) * mscale;
                float s = sycl::sin(theta) * mscale;
                if (inverse) s = -s;
                float x0 = tail[i] * scale;
                float x1 = tail[i + 1] * scale;
                tail[i] = x0 * c - x1 * s;
                tail[i + 1] = x0 * s + x1 * c;
            }
        });
        g_queue->wait_and_throw();
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL head_rms_norm_rope_tail failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_rope_tail_tensor(
        ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head,
        uint32_t head_dim, uint32_t n_rot, uint32_t pos0,
        uint32_t n_ctx_orig, bool inverse, float freq_base,
        float freq_scale, float ext_factor, float attn_factor,
        float beta_fast, float beta_slow) {
    if (!x || n_rot > head_dim || (n_rot & 1) ||
        x->bytes < (uint64_t)n_tok * n_head * head_dim * sizeof(float)) return 0;
    try {
        uint32_t pairs = n_tok * n_head * (n_rot / 2);
        g_queue->parallel_for(sycl::range<1>(pairs), [=](sycl::id<1> gid) {
            uint32_t g = (uint32_t)gid;
            uint32_t pair = g % (n_rot / 2);
            uint32_t tmp = g / (n_rot / 2);
            uint32_t h = tmp % n_head;
            uint32_t t = tmp / n_head;
            uint32_t n_nope = head_dim - n_rot;
            uint32_t i = pair * 2;

            float corr0 = 0.0f, corr1 = 0.0f;
            if (ext_factor != 0.0f) {
                float denom = 2.0f * sycl::log(freq_base);
                corr0 = sycl::floor((float)n_rot * sycl::log((float)n_ctx_orig / (beta_fast * 2.0f * (float)M_PI)) / denom);
                corr1 = sycl::ceil((float)n_rot * sycl::log((float)n_ctx_orig / (beta_slow * 2.0f * (float)M_PI)) / denom);
                corr0 = sycl::fmax(0.0f, corr0);
                corr1 = sycl::fmin((float)(n_rot - 1), corr1);
            }
            float theta_extrap = (float)(pos0 + t) * sycl::pow(freq_base, -((float)i) / (float)n_rot);
            float theta_interp = freq_scale * theta_extrap;
            float theta = theta_interp;
            float mscale = attn_factor;
            if (ext_factor != 0.0f) {
                float ramp_mix = rope_yarn_ramp(corr0, corr1, (int)i) * ext_factor;
                theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
                mscale *= 1.0f + 0.1f * sycl::log(1.0f / freq_scale);
            }
            float c = sycl::cos(theta) * mscale;
            float s = sycl::sin(theta) * mscale;
            if (inverse) s = -s;
            float *tail = (float *)x->ptr + ((uint64_t)t * n_head + h) * head_dim + n_nope;
            float x0 = tail[i];
            float x1 = tail[i + 1];
            tail[i] = x0 * c - x1 * s;
            tail[i + 1] = x0 * s + x1 * c;
        });
        g_queue->wait_and_throw();
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL rope_tail failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *q_half,
        const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head,
        uint32_t head_dim, uint32_t n_rot, uint32_t pos0,
        uint32_t n_ctx_orig, bool inverse, float freq_base,
        float freq_scale, float ext_factor, float attn_factor,
        float beta_fast, float beta_slow, float eps) {
    /* Fused matmul(f16) + f32→f16 convert + head_rms_norm + rope_tail.
       Not yet fused — compose from individual ops. */
    if (!out || !q_half || !model_map || !x || in_dim == 0 || out_dim == 0) return 0;
    if (!ds4_gpu_matmul_f16_tensor(out, model_map, model_size, weight_offset,
                                    in_dim, out_dim, x, n_tok)) return 0;
    /* Convert f32→f16 for q_half -> copy needed for attention f16 path */
    uint64_t n_elems = out_dim * n_tok;
    if (q_half->bytes < n_elems * sizeof(sycl::half)) return 0;
    try {
        sycl_convert_f32_f16(*g_queue, n_elems, (const float *)out->ptr, (sycl::half *)q_half->ptr);
        g_queue->wait_and_throw();
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL attn_q_b_f16_head_rms_rope_tail failed: %s\n", e.what());
        return 0;
    }
    if (!ds4_gpu_head_rms_norm_rope_tail_tensor(
            out, n_tok, n_head, head_dim, n_rot, pos0,
            n_ctx_orig, inverse, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow, eps)) return 0;
    return 1;
}

static inline float sycl_e4m3fn_value(int i) {
    int exp = (i >> 3) & 15;
    int mant = i & 7;
    if (exp == 0) return (float)mant * 0.001953125f;
    return (1.0f + (float)mant * 0.125f) * exp2f((float)exp - 7.0f);
}
static inline float sycl_e4m3fn_dequant(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = fminf(fabsf(x), 448.0f);
    int lo = 0, hi = 126;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (sycl_e4m3fn_value(mid) <= ax) lo = mid;
        else hi = mid - 1;
    }
    int best = lo;
    if (best < 126) {
        float bd = fabsf(ax - sycl_e4m3fn_value(best));
        float nd = fabsf(ax - sycl_e4m3fn_value(best + 1));
        if (nd < bd || (nd == bd && (((best + 1) & 1) == 0) && ((best & 1) != 0))) best++;
    }
    return sign * sycl_e4m3fn_value(best);
}

extern "C" int ds4_gpu_dsv4_fp8_kv_quantize_tensor(
        ds4_gpu_tensor *x, uint32_t n_tok, uint32_t head_dim, uint32_t n_rot) {
    if (!x || n_tok == 0 || head_dim == 0 || n_rot > head_dim ||
        x->bytes < (uint64_t)n_tok * head_dim * sizeof(float)) return 0;
    sycl::queue *qptr = g_queue;
    if (!qptr) return 0;
    float *x_data = (float *)x->ptr;
    uint32_t n_nope = head_dim - n_rot;
    if (n_nope == 0) return 1;
    qptr->submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> scratch(64, h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(n_tok * 64), sycl::range<1>(64)),
            [=](sycl::nd_item<1> item) {
            uint32_t row = item.get_group(0);
            uint32_t tid = item.get_local_linear_id();
            if (row >= n_tok) return;
            float *xr = x_data + (uint64_t)row * head_dim;
            for (uint32_t off = 0; off < n_nope; off += 64) {
                float v = 0.0f;
                if (off + tid < n_nope) v = xr[off + tid];
                scratch[tid] = off + tid < n_nope ? fabsf(v) : 0.0f;
                item.barrier(sycl::access::fence_space::local_space);
                for (uint32_t stride = 32; stride > 0; stride >>= 1) {
                    if (tid < stride) scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
                    item.barrier(sycl::access::fence_space::local_space);
                }
                float scale = exp2f(ceilf(log2f(fmaxf(scratch[0], 1.0e-4f) / 448.0f)));
                if (off + tid < n_nope) {
                    float q = sycl_e4m3fn_dequant(fminf(448.0f, fmaxf(-448.0f, v / scale))) * scale;
                    xr[off + tid] = q;
                }
                item.barrier(sycl::access::fence_space::local_space);
            }
        });
    }).wait();
    return 1;
}

extern "C" int ds4_gpu_store_raw_kv_tensor(
        ds4_gpu_tensor *raw_cache, const ds4_gpu_tensor *kv,
        uint32_t raw_cap, uint32_t row, uint32_t head_dim) {
    return ds4_gpu_store_raw_kv_batch_tensor(raw_cache, kv, raw_cap, row, 1, head_dim);
}

extern "C" int ds4_gpu_store_raw_kv_batch_tensor(
        ds4_gpu_tensor *raw_cache, const ds4_gpu_tensor *kv,
        uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens, uint32_t head_dim) {
    if (!raw_cache || !kv || head_dim == 0 || n_tokens == 0 || raw_cap == 0 ||
        kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float) ||
        raw_cache->bytes < (uint64_t)raw_cap * head_dim * sizeof(float)) return 0;
    sycl::queue *qptr = g_queue;
    if (!qptr) return 0;
    float *raw_data = (float *)raw_cache->ptr;
    const float *kv_data = (const float *)kv->ptr;
    uint64_t n = (uint64_t)n_tokens * head_dim;
    qptr->submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> gid) {
            uint32_t d = gid % head_dim;
            uint32_t t = gid / head_dim;
            uint32_t dest_row = (pos0 + t) % raw_cap;
            float v = kv_data[(uint64_t)t * head_dim + d];
            raw_data[(uint64_t)dest_row * head_dim + d] = (float)(sycl::half)v;
        });
    }).wait();
    return 1;
}

extern "C" int ds4_gpu_kv_fp8_store_raw_tensor(
        ds4_gpu_tensor *kv, ds4_gpu_tensor *raw_cache,
        uint32_t raw_cap, uint32_t row, uint32_t head_dim, uint32_t n_rot) {
    if (!ds4_gpu_dsv4_fp8_kv_quantize_tensor(kv, 1, head_dim, n_rot)) return 0;
    return ds4_gpu_store_raw_kv_tensor(raw_cache, kv, raw_cap, row, head_dim);
}

/* ---- Compressor store batch ---- */
extern "C" int ds4_gpu_compressor_store_batch_tensor(
        const ds4_gpu_tensor *kv, const ds4_gpu_tensor *sc,
        ds4_gpu_tensor *state_kv, ds4_gpu_tensor *state_score,
        const void *model_map, uint64_t model_size,
        uint64_t ape_offset, uint32_t ape_type,
        uint32_t head_dim, uint32_t ratio,
        uint32_t pos0, uint32_t n_tokens) {
    if (!kv || !sc || !state_kv || !state_score || !model_map ||
        head_dim == 0 || ratio == 0 || n_tokens == 0 ||
        (ape_type != 0u && ape_type != 1u)) return 0;
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes) return 0;
    const char *ape = sycl_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
    if (!ape) return 0;
    sycl::queue *qptr = g_queue;
    if (!qptr) return 0;
    const float *kv_data = (const float *)kv->ptr;
    const float *sc_data = (const float *)sc->ptr;
    float *skv = (float *)state_kv->ptr;
    float *ssc = (float *)state_score->ptr;
    uint64_t n = (uint64_t)n_tokens * width;
    uint32_t apt = ape_type;
    qptr->submit([=](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> gid) {
            uint32_t t = gid / width;
            uint32_t j = gid - (uint64_t)t * width;
            uint32_t pos_mod = (pos0 + t) % ratio;
            uint32_t dst_row = ratio == 4u ? ratio + pos_mod : pos_mod;
            skv[(uint64_t)dst_row * width + j] = kv_data[(uint64_t)t * width + j];
            ssc[(uint64_t)dst_row * width + j] =
                sc_data[(uint64_t)t * width + j] + (apt == 1u ? (float)((const sycl::half *)ape)[(uint64_t)pos_mod * width + j] : ((const float *)ape)[(uint64_t)pos_mod * width + j]);
        });
    }).wait();
    return 1;
}

/* ---- Compressor update ---- */
extern "C" int ds4_gpu_compressor_update_tensor(
        const ds4_gpu_tensor *kv_cur, const ds4_gpu_tensor *sc_cur,
        ds4_gpu_tensor *state_kv, ds4_gpu_tensor *state_score,
        ds4_gpu_tensor *comp_cache,
        const void *model_map, uint64_t model_size,
        uint64_t ape_offset, uint32_t ape_type,
        uint64_t norm_offset, uint32_t norm_type,
        uint32_t head_dim, uint32_t ratio, uint32_t pos,
        uint32_t comp_row, uint32_t n_rot, uint32_t n_ctx_orig,
        float freq_base, float freq_scale, float ext_factor,
        float attn_factor, float beta_fast, float beta_slow, float rms_eps) {
    if (!kv_cur || !sc_cur || !state_kv || !state_score || !comp_cache ||
        !model_map || head_dim == 0 || ratio == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) || norm_type != 0u) return 0;
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t emit = ((pos + 1u) % ratio) == 0u ? 1u : 0u;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)(comp_row + (emit ? 1u : 0u)) * head_dim * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || norm_bytes > model_size - norm_offset ||
        kv_cur->bytes < kv_bytes || sc_cur->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        (emit && comp_cache->bytes < comp_bytes)) return 0;
    if (!ds4_gpu_compressor_store_batch_tensor(kv_cur, sc_cur, state_kv, state_score,
                                                 model_map, model_size, ape_offset, ape_type,
                                                 head_dim, ratio, pos, 1)) return 0;
    if (!emit) return 1;
    sycl::queue *qptr = g_queue;
    if (!qptr) return 0;
    float *comp_data = (float *)comp_cache->ptr;
    float *skv = (float *)state_kv->ptr;
    float *ssc = (float *)state_score->ptr;
    qptr->submit([=](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(head_dim), [=](sycl::id<1> d_id) {
            uint32_t d = d_id[0];
            float vals[128];
            float scores[128];
            float max_s = -INFINITY;
            uint32_t n_cand = 0;
            if (ratio == 4u) {
                for (uint32_t r = 0; r < 4; r++) {
                    vals[n_cand] = skv[(uint64_t)r * width + d];
                    scores[n_cand] = ssc[(uint64_t)r * width + d];
                    max_s = fmaxf(max_s, scores[n_cand++]);
                }
                for (uint32_t r = 0; r < 4; r++) {
                    vals[n_cand] = skv[(uint64_t)(ratio + r) * width + width/2 + d];
                    scores[n_cand] = ssc[(uint64_t)(ratio + r) * width + width/2 + d];
                    max_s = fmaxf(max_s, scores[n_cand++]);
                }
            } else {
                for (uint32_t r = 0; r < ratio; r++) {
                    vals[n_cand] = skv[(uint64_t)r * width + d];
                    scores[n_cand] = ssc[(uint64_t)r * width + d];
                    max_s = fmaxf(max_s, scores[n_cand++]);
                }
            }
            float den = 0.0f, acc = 0.0f;
            for (uint32_t i = 0; i < n_cand; i++) {
                float w = expf(scores[i] - max_s);
                den += w;
                acc += vals[i] * w;
            }
            comp_data[(uint64_t)comp_row * head_dim + d] = den != 0.0f ? acc / den : 0.0f;
        });
    }).wait();
    ds4_gpu_tensor *comp_row_view = ds4_gpu_tensor_view(
            comp_cache,
            (uint64_t)comp_row * head_dim * sizeof(float),
            (uint64_t)head_dim * sizeof(float));
    if (!comp_row_view) return 0;
    int ok = ds4_gpu_rms_norm_weight_rows_tensor(comp_row_view, comp_row_view,
                                                   model_map, model_size, norm_offset,
                                                   head_dim, 1, rms_eps);
    if (ok) ok = ds4_gpu_rope_tail_tensor(comp_row_view, 1, 1, head_dim, n_rot,
                                            pos + 1u - ratio, n_ctx_orig, false,
                                            freq_base, freq_scale, ext_factor, attn_factor,
                                            beta_fast, beta_slow);
    ds4_gpu_tensor_free(comp_row_view);
    if (ok && ratio == 4u) {
        uint64_t half = 4ull * width;
        qptr->submit([=](sycl::handler &h) {
            h.parallel_for(sycl::range<1>(half), [=](sycl::id<1> i) {
                float v = skv[half + i];
                float s = ssc[half + i];
                skv[i] = v;
                ssc[i] = s;
                skv[half + i] = v;
                ssc[half + i] = s;
            });
        }).wait();
    }
    return 1;
}

/* ---- Compressor set rows (shared helper for prefill) ---- */
static inline void compressor_set_rows_launch(
        float *state_kv, float *state_score,
        const float *kv, const float *sc,
        const char *ape, uint32_t ape_type,
        uint32_t width, uint32_t ratio, uint32_t pos0,
        uint32_t src0, uint32_t dst0, uint32_t rows) {
    sycl::queue *qptr = g_queue;
    if (!qptr) return;
    uint64_t n = (uint64_t)rows * width;
    uint32_t apt = ape_type;
    qptr->submit([=](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> gid) {
            uint32_t r = gid / width;
            uint32_t j = gid - (uint64_t)r * width;
            uint32_t src = src0 + r;
            uint32_t dst = dst0 + r;
            uint32_t phase = (pos0 + src) % ratio;
            state_kv[(uint64_t)dst * width + j] = kv[(uint64_t)src * width + j];
            state_score[(uint64_t)dst * width + j] =
                sc[(uint64_t)src * width + j] + (apt == 1u ? (float)((const sycl::half *)ape)[(uint64_t)phase * width + j] : ((const float *)ape)[(uint64_t)phase * width + j]);
        });
    }).wait();
}

/* ---- Compressor prefill pool ---- */
static inline void compressor_prefill_pool_launch(
        float *comp, const float *kv, const float *sc,
        const float *state_kv, const float *state_score,
        const char *ape, uint32_t ape_type,
        uint32_t head_dim, uint32_t ratio, uint32_t pos0, uint32_t n_comp, uint32_t replay) {
    sycl::queue *qptr = g_queue;
    if (!qptr) return;
    uint32_t apt = ape_type;
    qptr->submit([=](sycl::handler &h) {
        h.parallel_for(sycl::range<2>(n_comp, head_dim), [=](sycl::id<2> idx) {
            uint32_t c = idx[0];
            uint32_t d = idx[1];
            uint32_t coff = ratio == 4u ? 2u : 1u;
            uint32_t width = coff * head_dim;
            float vals[128];
            float scores[128];
            float max_s = -INFINITY;
            uint32_t n_cand = 0;
            if (ratio == 4u) {
                if (replay && c == 0) {
                    for (uint32_t r = 0; r < 4; r++) {
                        vals[n_cand] = state_kv[(uint64_t)r * width + d];
                        scores[n_cand] = state_score[(uint64_t)r * width + d];
                        max_s = fmaxf(max_s, scores[n_cand++]);
                    }
                } else if (c > 0) {
                    uint32_t base = (c - 1u) * ratio;
                    for (uint32_t r = 0; r < 4; r++) {
                        uint32_t t = base + r;
                        float ape_val = (apt == 1u ? (float)((const sycl::half *)ape)[(uint64_t)((pos0 + t) % ratio) * width + d] : ((const float *)ape)[(uint64_t)((pos0 + t) % ratio) * width + d]);
                        vals[n_cand] = kv[(uint64_t)t * width + d];
                        scores[n_cand] = sc[(uint64_t)t * width + d] + ape_val;
                        max_s = fmaxf(max_s, scores[n_cand++]);
                    }
                }
                uint32_t base = c * ratio;
                for (uint32_t r = 0; r < 4; r++) {
                    uint32_t t = base + r;
                    float ape_val2 = (apt == 1u ? (float)((const sycl::half *)ape)[(uint64_t)((pos0 + t) % ratio) * width + head_dim + d] : ((const float *)ape)[(uint64_t)((pos0 + t) % ratio) * width + head_dim + d]);
                    vals[n_cand] = kv[(uint64_t)t * width + head_dim + d];
                    scores[n_cand] = sc[(uint64_t)t * width + head_dim + d] + ape_val2;
                    max_s = fmaxf(max_s, scores[n_cand++]);
                }
            } else {
                uint32_t base = c * ratio;
                for (uint32_t r = 0; r < ratio; r++) {
                    uint32_t t = base + r;
                    float ape_val3 = (apt == 1u ? (float)((const sycl::half *)ape)[(uint64_t)((pos0 + t) % ratio) * width + d] : ((const float *)ape)[(uint64_t)((pos0 + t) % ratio) * width + d]);
                    vals[n_cand] = kv[(uint64_t)t * width + d];
                    scores[n_cand] = sc[(uint64_t)t * width + d] + ape_val3;
                    max_s = fmaxf(max_s, scores[n_cand++]);
                }
            }
            float den = 0.0f, acc = 0.0f;
            for (uint32_t i = 0; i < n_cand; i++) {
                float w = expf(scores[i] - max_s);
                den += w;
                acc += vals[i] * w;
            }
            comp[(uint64_t)c * head_dim + d] = den != 0.0f ? acc / den : 0.0f;
        });
    }).wait();
}

/* ---- Compressor prefill ---- */
extern "C" int ds4_gpu_compressor_prefill_tensor(
        ds4_gpu_tensor *comp_cache, ds4_gpu_tensor *state_kv,
        ds4_gpu_tensor *state_score,
        const ds4_gpu_tensor *kv, const ds4_gpu_tensor *sc,
        const void *model_map, uint64_t model_size,
        uint64_t ape_offset, uint32_t ape_type,
        uint64_t norm_offset, uint32_t norm_type,
        uint32_t head_dim, uint32_t ratio, uint32_t pos0,
        uint32_t n_tokens, uint32_t n_rot, uint32_t n_ctx_orig,
        bool quantize_fp8,
        float freq_base, float freq_scale, float ext_factor,
        float attn_factor, float beta_fast, float beta_slow, float rms_eps) {
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        head_dim == 0 || ratio == 0 || n_tokens == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) || norm_type != 0u) return 0;
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t n_comp = n_tokens / ratio;
    const uint32_t cutoff = n_comp * ratio;
    const uint32_t rem = n_tokens - cutoff;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || norm_bytes > model_size - norm_offset ||
        kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        (n_comp && comp_cache->bytes < comp_bytes)) return 0;
    const char *ape = sycl_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
    if (!ape) return 0;
    sycl::queue *qptr = g_queue;
    if (!qptr) return 0;
    float *skv = (float *)state_kv->ptr;
    float *ssc = (float *)state_score->ptr;
    uint64_t state_n = (uint64_t)state_rows * width;
    qptr->memset(skv, 0, state_n * sizeof(float));
    qptr->submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(state_n), [=](sycl::id<1> i) { ssc[(uint64_t)i] = -INFINITY; });
    }).wait();
    const float *kv_data = (const float *)kv->ptr;
    const float *sc_data = (const float *)sc->ptr;
    if (ratio == 4u) {
        if (cutoff >= ratio) {
            compressor_set_rows_launch(skv, ssc, kv_data, sc_data, ape, ape_type,
                                       width, ratio, pos0, cutoff - ratio, 0, ratio);
        }
        if (rem != 0) {
            compressor_set_rows_launch(skv, ssc, kv_data, sc_data, ape, ape_type,
                                       width, ratio, pos0, cutoff, ratio, rem);
        }
    } else if (rem != 0) {
        compressor_set_rows_launch(skv, ssc, kv_data, sc_data, ape, ape_type,
                                   width, ratio, pos0, cutoff, 0, rem);
    }
    float *comp_data = (float *)comp_cache->ptr;
    if (n_comp != 0) {
        compressor_prefill_pool_launch(comp_data, kv_data, sc_data, skv, ssc,
                                       ape, ape_type, head_dim, ratio, pos0, n_comp, 0);
        if (!ds4_gpu_rms_norm_weight_rows_tensor(comp_cache, comp_cache,
                                                   model_map, model_size, norm_offset,
                                                   head_dim, n_comp, rms_eps)) return 0;
        if (n_rot != 0) {
            for (uint32_t c = 0; c < n_comp; c++) {
                ds4_gpu_tensor row_view = { (char*)comp_data + (uint64_t)c * head_dim * sizeof(float), (uint64_t)head_dim * sizeof(float), 0 };
                if (!ds4_gpu_rope_tail_tensor(&row_view, 1, 1, head_dim, n_rot,
                                                pos0 + c * ratio, n_ctx_orig, false,
                                                freq_base, freq_scale, ext_factor, attn_factor,
                                                beta_fast, beta_slow)) return 0;
            }
        }
        if (quantize_fp8 && !ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim, n_rot)) return 0;
    }
    return 1;
}

/* ---- Compressor prefill ratio-4 replay ---- */
extern "C" int ds4_gpu_compressor_prefill_ratio4_replay_tensor(
        ds4_gpu_tensor *comp_cache, ds4_gpu_tensor *state_kv,
        ds4_gpu_tensor *state_score,
        const ds4_gpu_tensor *kv, const ds4_gpu_tensor *sc,
        const void *model_map, uint64_t model_size,
        uint64_t ape_offset, uint32_t ape_type,
        uint64_t norm_offset, uint32_t norm_type,
        uint32_t head_dim, uint32_t pos0, uint32_t n_tokens,
        uint32_t n_rot, uint32_t n_ctx_orig, bool quantize_fp8,
        float freq_base, float freq_scale, float ext_factor,
        float attn_factor, float beta_fast, float beta_slow, float rms_eps) {
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        head_dim == 0 || n_tokens == 0 || (n_tokens & 3u) != 0 || (pos0 & 3u) != 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) || norm_type != 0u) return 0;
    const uint32_t ratio = 4u;
    const uint32_t width = 2u * head_dim;
    const uint32_t state_rows = 8u;
    const uint32_t n_comp = n_tokens / ratio;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || norm_bytes > model_size - norm_offset ||
        kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        comp_cache->bytes < comp_bytes) return 0;
    const char *ape = sycl_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
    if (!ape) return 0;
    float *comp_data = (float *)comp_cache->ptr;
    float *skv = (float *)state_kv->ptr;
    float *ssc = (float *)state_score->ptr;
    const float *kv_data = (const float *)kv->ptr;
    const float *sc_data = (const float *)sc->ptr;
    compressor_prefill_pool_launch(comp_data, kv_data, sc_data, skv, ssc,
                                   ape, ape_type, head_dim, ratio, pos0, n_comp, 1);
    if (!ds4_gpu_rms_norm_weight_rows_tensor(comp_cache, comp_cache,
                                               model_map, model_size, norm_offset,
                                               head_dim, n_comp, rms_eps)) return 0;
    if (n_rot != 0) {
        for (uint32_t c = 0; c < n_comp; c++) {
            ds4_gpu_tensor row_view = { (char*)comp_data + (uint64_t)c * head_dim * sizeof(float), (uint64_t)head_dim * sizeof(float), 0 };
            if (!ds4_gpu_rope_tail_tensor(&row_view, 1, 1, head_dim, n_rot,
                                            pos0 + c * ratio, n_ctx_orig, false,
                                            freq_base, freq_scale, ext_factor, attn_factor,
                                            beta_fast, beta_slow)) return 0;
        }
    }
    if (quantize_fp8 && !ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim, n_rot)) return 0;
    sycl::queue *qptr2 = g_queue;
    if (!qptr2) return 0;
    uint64_t state_n = (uint64_t)state_rows * width;
    qptr2->memset(skv, 0, state_n * sizeof(float));
    qptr2->submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(state_n), [=](sycl::id<1> i) { ssc[(uint64_t)i] = -INFINITY; });
    }).wait();
    compressor_set_rows_launch(skv, ssc, kv_data, sc_data, ape, ape_type,
                               width, ratio, pos0, n_tokens - ratio, 0, ratio);
    return 1;
}

/* ---- Compressor prefill state ratio-4 ---- */
extern "C" int ds4_gpu_compressor_prefill_state_ratio4_tensor(
        ds4_gpu_tensor *state_kv, ds4_gpu_tensor *state_score,
        const ds4_gpu_tensor *kv_tail, const ds4_gpu_tensor *sc_tail,
        const void *model_map, uint64_t model_size,
        uint64_t ape_offset, uint32_t ape_type,
        uint32_t head_dim, uint32_t pos0) {
    if (!state_kv || !state_score || !kv_tail || !sc_tail || !model_map ||
        head_dim == 0 || (ape_type != 0u && ape_type != 1u)) return 0;
    const uint32_t ratio = 4u;
    const uint32_t width = 2u * head_dim;
    const uint32_t state_rows = 8u;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t tail_bytes = (uint64_t)ratio * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)ratio * width * elem_ape;
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        kv_tail->bytes < tail_bytes || sc_tail->bytes < tail_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes) return 0;
    const char *ape = sycl_model_range_ptr(model_map, ape_offset, ape_bytes, "compressor_ape");
    if (!ape) return 0;
    sycl::queue *qptr = g_queue;
    if (!qptr) return 0;
    float *skv = (float *)state_kv->ptr;
    float *ssc = (float *)state_score->ptr;
    uint64_t state_n = (uint64_t)state_rows * width;
    qptr->memset(skv, 0, state_n * sizeof(float));
    qptr->submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(state_n), [=](sycl::id<1> i) { ssc[(uint64_t)i] = -INFINITY; });
    }).wait();
    compressor_set_rows_launch(skv, ssc,
                               (const float *)kv_tail->ptr, (const float *)sc_tail->ptr,
                               ape, ape_type, width, ratio, pos0, 0, 0, ratio);
    return 1;
}

/* =========================================================================
 * Attention stubs
 * ========================================================================= */
extern "C" int ds4_gpu_attention_decode_heads_tensor(
        ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
        uint64_t sinks_offset, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv, uint32_t n_raw, uint32_t raw_cap,
        uint32_t raw_start, const ds4_gpu_tensor *comp_kv,
        uint32_t comp_kv_f16, uint32_t n_comp,
        const ds4_gpu_tensor *comp_mask, uint32_t use_mask,
        uint32_t n_head, uint32_t head_dim) {
    if (comp_kv_f16 || !heads || !q || !raw_kv || !model_map ||
        n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap ||
        (n_comp != 0 && !comp_kv) || (use_mask && !comp_mask) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp * head_dim * sizeof(float)) ||
        (use_mask && comp_mask->bytes < (uint64_t)n_comp * sizeof(float))) return 0;
    const char *sptr = sycl_model_range_ptr(model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sptr) return 0;
    const float *sinks = (const float *)sptr;
    try {
        uint32_t total_kv = n_raw + n_comp;
        g_queue->parallel_for(sycl::range<1>(n_head), [=](sycl::id<1> idx) {
            uint32_t h = (uint32_t)idx;
            const float *qh = (const float *)q->ptr + (uint64_t)h * head_dim;
            float *head_out = (float *)heads->ptr + (uint64_t)h * head_dim;
            float inv_scale = 1.0f / sycl::sqrt((float)head_dim);

            /* Compute scores */
            float max_score = -INFINITY;
            for (uint32_t i = 0; i < n_raw; i++) {
                uint32_t ki = (raw_start + i) % raw_cap;
                const float *kv = (const float *)raw_kv->ptr + (uint64_t)ki * head_dim;
                float d = 0.0f;
                for (uint32_t j = 0; j < head_dim; j++) d += qh[j] * kv[j];
                float s = d * inv_scale;
                if (i == 0) s += sinks[h];
                if (s > max_score) max_score = s;
            }
            for (uint32_t i = 0; i < n_comp; i++) {
                const float *kv = (const float *)comp_kv->ptr + (uint64_t)i * head_dim;
                float d = 0.0f;
                for (uint32_t j = 0; j < head_dim; j++) d += qh[j] * kv[j];
                float s = d * inv_scale;
                if (use_mask) s += ((const float *)comp_mask->ptr)[i];
                if (s > max_score) max_score = s;
            }

            /* Softmax: exp and sum */
            float exp_sum = 0.0f;
            for (uint32_t i = 0; i < n_raw; i++) {
                uint32_t ki = (raw_start + i) % raw_cap;
                const float *kv = (const float *)raw_kv->ptr + (uint64_t)ki * head_dim;
                float d = 0.0f;
                for (uint32_t j = 0; j < head_dim; j++) d += qh[j] * kv[j];
                float s = d * inv_scale;
                if (i == 0) s += sinks[h];
                exp_sum += sycl::exp(s - max_score);
            }
            for (uint32_t i = 0; i < n_comp; i++) {
                const float *kv = (const float *)comp_kv->ptr + (uint64_t)i * head_dim;
                float d = 0.0f;
                for (uint32_t j = 0; j < head_dim; j++) d += qh[j] * kv[j];
                float s = d * inv_scale;
                if (use_mask) s += ((const float *)comp_mask->ptr)[i];
                exp_sum += sycl::exp(s - max_score);
            }
            float inv_sum = 1.0f / exp_sum;

            /* Weighted sum of values */
            for (uint32_t j = 0; j < head_dim; j++) head_out[j] = 0.0f;
            for (uint32_t i = 0; i < n_raw; i++) {
                uint32_t ki = (raw_start + i) % raw_cap;
                const float *kv = (const float *)raw_kv->ptr + (uint64_t)ki * head_dim;
                float d = 0.0f;
                for (uint32_t j = 0; j < head_dim; j++) d += qh[j] * kv[j];
                float s = d * inv_scale;
                if (i == 0) s += sinks[h];
                float w = sycl::exp(s - max_score) * inv_sum;
                for (uint32_t j = 0; j < head_dim; j++) head_out[j] += w * kv[j];
            }
            for (uint32_t i = 0; i < n_comp; i++) {
                const float *kv = (const float *)comp_kv->ptr + (uint64_t)i * head_dim;
                float d = 0.0f;
                for (uint32_t j = 0; j < head_dim; j++) d += qh[j] * kv[j];
                float s = d * inv_scale;
                if (use_mask) s += ((const float *)comp_mask->ptr)[i];
                float w = sycl::exp(s - max_score) * inv_sum;
                for (uint32_t j = 0; j < head_dim; j++) head_out[j] += w * kv[j];
            }
        });
        g_queue->wait_and_throw();
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL attention_decode_heads failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_attention_prefill_raw_heads_tensor(
        ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
        uint64_t sinks_offset, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv, uint32_t n_tokens, uint32_t window,
        uint32_t n_head, uint32_t head_dim) {
    if (!heads || !q || !raw_kv || !model_map || sinks_offset > model_size ||
        model_size - sinks_offset < (uint64_t)n_head * sizeof(float) ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float) ||
        window > 256) return 0;
    const char *sptr = sycl_model_range_ptr(model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sptr) return 0;
    const float *sinks = (const float *)sptr;
    try {
        uint32_t pairs = n_tokens * n_head;
        g_queue->parallel_for(sycl::range<1>(pairs), [=](sycl::id<1> idx) {
            uint32_t p = (uint32_t)idx;
            uint32_t h = p % n_head;
            uint32_t t = p / n_head;
            uint32_t kv_start = (t > window) ? (t - window) : 0;
            uint32_t kv_end = t + 1;
            uint32_t n_kv = kv_end - kv_start;
            const float *qt = (const float *)q->ptr + ((uint64_t)t * n_head + h) * head_dim;
            float *head_out = (float *)heads->ptr + ((uint64_t)t * n_head + h) * head_dim;
            float inv_scale = 1.0f / sycl::sqrt((float)head_dim);

            float max_score = -INFINITY;
            for (uint32_t k = kv_start; k < kv_end; k++) {
                const float *kv = (const float *)raw_kv->ptr + (uint64_t)k * head_dim;
                float d = 0.0f;
                for (uint32_t j = 0; j < head_dim; j++) d += qt[j] * kv[j];
                float s = d * inv_scale;
                if (k == 0) s += sinks[h];
                if (s > max_score) max_score = s;
            }
            float exp_sum = 0.0f;
            for (uint32_t k = kv_start; k < kv_end; k++) {
                const float *kv = (const float *)raw_kv->ptr + (uint64_t)k * head_dim;
                float d = 0.0f;
                for (uint32_t j = 0; j < head_dim; j++) d += qt[j] * kv[j];
                float s = d * inv_scale;
                if (k == 0) s += sinks[h];
                exp_sum += sycl::exp(s - max_score);
            }
            float inv_sum = 1.0f / exp_sum;
            for (uint32_t j = 0; j < head_dim; j++) head_out[j] = 0.0f;
            for (uint32_t k = kv_start; k < kv_end; k++) {
                const float *kv = (const float *)raw_kv->ptr + (uint64_t)k * head_dim;
                float d = 0.0f;
                for (uint32_t j = 0; j < head_dim; j++) d += qt[j] * kv[j];
                float s = d * inv_scale;
                if (k == 0) s += sinks[h];
                float w = sycl::exp(s - max_score) * inv_sum;
                for (uint32_t j = 0; j < head_dim; j++) head_out[j] += w * kv[j];
            }
        });
        g_queue->wait_and_throw();
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL attention_prefill_raw_heads failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_attention_decode_raw_batch_heads_tensor(
        ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
        uint64_t sinks_offset, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv, uint32_t n_tokens, uint32_t pos0,
        uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start,
        uint32_t window, uint32_t n_head, uint32_t head_dim) {
    if (!heads || !q || !raw_kv || !model_map || n_tokens == 0 || n_raw == 0 ||
        raw_cap < n_raw || raw_start >= raw_cap ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)raw_cap * head_dim * sizeof(float)) return 0;
    const char *sptr = sycl_model_range_ptr(model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sptr) return 0;
    const float *sinks = (const float *)sptr;
    try {
        uint32_t pairs = n_tokens * n_head;
        g_queue->parallel_for(sycl::range<1>(pairs), [=](sycl::id<1> idx) {
            uint32_t p = (uint32_t)idx;
            uint32_t h = p % n_head;
            uint32_t t = p / n_head;
            uint32_t qpos = pos0 + t;
            uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
            const float *qh = (const float *)q->ptr + ((uint64_t)t * n_head + h) * head_dim;
            float *head_out = (float *)heads->ptr + ((uint64_t)t * n_head + h) * head_dim;
            float inv_scale = sycl::rsqrt((float)head_dim);

            /* Determine visible raw range */
            uint32_t raw_lo = 0, raw_hi = 0;
            if (n_tokens == 1u) {
                raw_lo = 0; raw_hi = n_raw;
            } else {
                uint32_t lo = first_raw_pos;
                uint32_t hi = first_raw_pos + n_raw - 1u;
                if (window != 0 && qpos + 1u > window) {
                    uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                if (qpos < hi) hi = qpos;
                if (hi >= lo) { raw_lo = lo - first_raw_pos; raw_hi = hi - first_raw_pos + 1u; }
            }

            /* Compute max score */
            float max_score = -INFINITY;
            for (uint32_t i = raw_lo; i < raw_hi; i++) {
                uint32_t kv_idx = (raw_start + i) % raw_cap;
                const float *kv = (const float *)raw_kv->ptr + (uint64_t)kv_idx * head_dim;
                float dot = 0.0f;
                for (uint32_t j = 0; j < head_dim; j++) dot += qh[j] * kv[j];
                float s = dot * inv_scale;
                if (i == 0) s += sinks[h];
                if (s > max_score) max_score = s;
            }

            /* Softmax sum */
            float exp_sum = 0.0f;
            for (uint32_t i = raw_lo; i < raw_hi; i++) {
                uint32_t kv_idx = (raw_start + i) % raw_cap;
                const float *kv = (const float *)raw_kv->ptr + (uint64_t)kv_idx * head_dim;
                float dot = 0.0f;
                for (uint32_t j = 0; j < head_dim; j++) dot += qh[j] * kv[j];
                float s = dot * inv_scale;
                if (i == 0) s += sinks[h];
                exp_sum += sycl::exp(s - max_score);
            }
            float inv_sum = 1.0f / exp_sum;

            /* Weighted sum */
            for (uint32_t j = 0; j < head_dim; j++) head_out[j] = 0.0f;
            for (uint32_t i = raw_lo; i < raw_hi; i++) {
                uint32_t kv_idx = (raw_start + i) % raw_cap;
                const float *kv = (const float *)raw_kv->ptr + (uint64_t)kv_idx * head_dim;
                float dot = 0.0f;
                for (uint32_t j = 0; j < head_dim; j++) dot += qh[j] * kv[j];
                float s = dot * inv_scale;
                if (i == 0) s += sinks[h];
                float w = sycl::exp(s - max_score) * inv_sum;
                for (uint32_t j = 0; j < head_dim; j++) head_out[j] += w * kv[j];
            }
        });
        g_queue->wait_and_throw();
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL attention_decode_raw_batch_heads failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_attention_decode_mixed_batch_heads_tensor(
        ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
        uint64_t sinks_offset, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv, const ds4_gpu_tensor *comp_kv,
        uint32_t comp_kv_f16, const ds4_gpu_tensor *comp_mask,
        uint32_t use_comp_mask, uint32_t n_tokens, uint32_t pos0,
        uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start,
        uint32_t n_comp, uint32_t window, uint32_t ratio,
        uint32_t n_head, uint32_t head_dim) {
    if (comp_kv_f16 || !heads || !q || !raw_kv || !model_map || n_tokens == 0 ||
        n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap ||
        (n_comp != 0 && !comp_kv) || (use_comp_mask && !comp_mask) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp * head_dim * sizeof(float)) ||
        (use_comp_mask && comp_mask->bytes < (uint64_t)n_tokens * n_comp * sizeof(float)))
        return 0;
    if (n_comp != 0 && ratio == 0) return 0;
    const char *sptr = sycl_model_range_ptr(model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sptr) return 0;
    const float *sinks = (const float *)sptr;
    try {
        uint32_t pairs = n_tokens * n_head;
        g_queue->parallel_for(sycl::range<1>(pairs), [=](sycl::id<1> idx) {
            uint32_t p = (uint32_t)idx;
            uint32_t h = p % n_head;
            uint32_t t = p / n_head;
            uint32_t qpos = pos0 + t;
            uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
            const float *qh = (const float *)q->ptr + ((uint64_t)t * n_head + h) * head_dim;
            float *head_out = (float *)heads->ptr + ((uint64_t)t * n_head + h) * head_dim;
            float inv_scale = sycl::rsqrt((float)head_dim);

            /* Visible raw range */
            uint32_t raw_lo = 0, raw_hi = 0;
            if (n_tokens == 1u) {
                raw_lo = 0; raw_hi = n_raw;
            } else {
                uint32_t lo = first_raw_pos;
                uint32_t hi = first_raw_pos + n_raw - 1u;
                if (window != 0 && qpos + 1u > window) {
                    uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                if (qpos < hi) hi = qpos;
                if (hi >= lo) { raw_lo = lo - first_raw_pos; raw_hi = hi - first_raw_pos + 1u; }
            }

            /* Visible compressed count */
            uint32_t visible_comp = (n_tokens == 1u) ? n_comp : ((qpos + 1u) / ratio);
            if (visible_comp > n_comp) visible_comp = n_comp;

            /* Max score */
            float max_score = -INFINITY;
            for (uint32_t i = raw_lo; i < raw_hi; i++) {
                uint32_t kv_idx = (raw_start + i) % raw_cap;
                const float *kv = (const float *)raw_kv->ptr + (uint64_t)kv_idx * head_dim;
                float dot = 0.0f;
                for (uint32_t j = 0; j < head_dim; j++) dot += qh[j] * kv[j];
                float s = dot * inv_scale;
                if (i == 0) s += sinks[h];
                if (s > max_score) max_score = s;
            }
            for (uint32_t c = 0; c < visible_comp; c++) {
                float add = use_comp_mask ? ((const float *)comp_mask->ptr)[(uint64_t)t * n_comp + c] : 0.0f;
                float s = -INFINITY;
                if (add > -1.0e20f) {
                    const float *kv = (const float *)comp_kv->ptr + (uint64_t)c * head_dim;
                    float dot = 0.0f;
                    for (uint32_t j = 0; j < head_dim; j++) dot += qh[j] * kv[j];
                    s = dot * inv_scale + add;
                }
                if (s > max_score) max_score = s;
            }

            /* Softmax sum */
            float exp_sum = 0.0f;
            for (uint32_t i = raw_lo; i < raw_hi; i++) {
                uint32_t kv_idx = (raw_start + i) % raw_cap;
                const float *kv = (const float *)raw_kv->ptr + (uint64_t)kv_idx * head_dim;
                float dot = 0.0f;
                for (uint32_t j = 0; j < head_dim; j++) dot += qh[j] * kv[j];
                float s = dot * inv_scale;
                if (i == 0) s += sinks[h];
                exp_sum += sycl::exp(s - max_score);
            }
            for (uint32_t c = 0; c < visible_comp; c++) {
                float add = use_comp_mask ? ((const float *)comp_mask->ptr)[(uint64_t)t * n_comp + c] : 0.0f;
                float s = -INFINITY;
                if (add > -1.0e20f) {
                    const float *kv = (const float *)comp_kv->ptr + (uint64_t)c * head_dim;
                    float dot = 0.0f;
                    for (uint32_t j = 0; j < head_dim; j++) dot += qh[j] * kv[j];
                    s = dot * inv_scale + add;
                }
                if (s > -1.0e20f) exp_sum += sycl::exp(s - max_score);
            }
            float inv_sum = exp_sum > 0.0f ? 1.0f / exp_sum : 0.0f;

            /* Weighted sum */
            for (uint32_t j = 0; j < head_dim; j++) head_out[j] = 0.0f;
            for (uint32_t i = raw_lo; i < raw_hi; i++) {
                uint32_t kv_idx = (raw_start + i) % raw_cap;
                const float *kv = (const float *)raw_kv->ptr + (uint64_t)kv_idx * head_dim;
                float dot = 0.0f;
                for (uint32_t j = 0; j < head_dim; j++) dot += qh[j] * kv[j];
                float s = dot * inv_scale;
                if (i == 0) s += sinks[h];
                float w = sycl::exp(s - max_score) * inv_sum;
                for (uint32_t j = 0; j < head_dim; j++) head_out[j] += w * kv[j];
            }
            for (uint32_t c = 0; c < visible_comp; c++) {
                float add = use_comp_mask ? ((const float *)comp_mask->ptr)[(uint64_t)t * n_comp + c] : 0.0f;
                float s = -INFINITY;
                if (add > -1.0e20f) {
                    const float *kv = (const float *)comp_kv->ptr + (uint64_t)c * head_dim;
                    float dot = 0.0f;
                    for (uint32_t j = 0; j < head_dim; j++) dot += qh[j] * kv[j];
                    s = dot * inv_scale + add;
                    float w = sycl::exp(s - max_score) * inv_sum;
                    for (uint32_t j = 0; j < head_dim; j++) head_out[j] += w * kv[j];
                }
            }
        });
        g_queue->wait_and_throw();
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL attention_decode_mixed_batch_heads failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
        ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
        uint64_t sinks_offset, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv, const ds4_gpu_tensor *comp_kv,
        uint32_t comp_kv_f16, const ds4_gpu_tensor *topk,
        uint32_t n_tokens, uint32_t pos0, uint32_t n_raw,
        uint32_t raw_cap, uint32_t raw_start, uint32_t n_comp,
        uint32_t top_k, uint32_t window, uint32_t ratio,
        uint32_t n_head, uint32_t head_dim) {
    if (comp_kv_f16 || !heads || !q || !raw_kv || !comp_kv || !topk || !model_map ||
        n_tokens == 0 || n_raw == 0 || raw_cap < n_raw || raw_start >= raw_cap ||
        n_comp == 0 || top_k == 0 ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)raw_cap * head_dim * sizeof(float) ||
        comp_kv->bytes < (uint64_t)n_comp * head_dim * sizeof(float) ||
        topk->bytes < (uint64_t)n_tokens * top_k * sizeof(uint32_t))
        return 0;
    const char *sptr = sycl_model_range_ptr(model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sptr) return 0;
    const float *sinks = (const float *)sptr;
    try {
        uint32_t pairs = n_tokens * n_head;
        g_queue->parallel_for(sycl::range<1>(pairs), [=](sycl::id<1> idx) {
            uint32_t p = (uint32_t)idx;
            uint32_t h = p % n_head;
            uint32_t t = p / n_head;
            uint32_t qpos = pos0 + t;
            uint32_t first_raw_pos = pos0 + n_tokens - n_raw;
            const float *qh = (const float *)q->ptr + ((uint64_t)t * n_head + h) * head_dim;
            float *head_out = (float *)heads->ptr + ((uint64_t)t * n_head + h) * head_dim;
            float inv_scale = sycl::rsqrt((float)head_dim);
            const uint32_t *topk_ptr = (const uint32_t *)topk->ptr + (uint64_t)t * top_k;

            /* Visible raw range */
            uint32_t raw_lo = 0, raw_hi = 0;
            if (n_tokens == 1u) {
                raw_lo = 0; raw_hi = n_raw;
            } else {
                uint32_t lo = first_raw_pos;
                uint32_t hi = first_raw_pos + n_raw - 1u;
                if (window != 0 && qpos + 1u > window) {
                    uint32_t wlo = qpos + 1u - window;
                    if (wlo > lo) lo = wlo;
                }
                if (qpos < hi) hi = qpos;
                if (hi >= lo) { raw_lo = lo - first_raw_pos; raw_hi = hi - first_raw_pos + 1u; }
            }

            /* Max score over raw + indexed comp */
            float max_score = -INFINITY;
            for (uint32_t i = raw_lo; i < raw_hi; i++) {
                uint32_t kv_idx = (raw_start + i) % raw_cap;
                const float *kv = (const float *)raw_kv->ptr + (uint64_t)kv_idx * head_dim;
                float dot = 0.0f;
                for (uint32_t j = 0; j < head_dim; j++) dot += qh[j] * kv[j];
                float s = dot * inv_scale;
                if (i == 0) s += sinks[h];
                if (s > max_score) max_score = s;
            }
            for (uint32_t k = 0; k < top_k; k++) {
                uint32_t c = topk_ptr[k];
                if (c < n_comp) {
                    const float *kv = (const float *)comp_kv->ptr + (uint64_t)c * head_dim;
                    float dot = 0.0f;
                    for (uint32_t j = 0; j < head_dim; j++) dot += qh[j] * kv[j];
                    float s = dot * inv_scale;
                    if (s > max_score) max_score = s;
                }
            }

            /* Softmax sum */
            float exp_sum = 0.0f;
            for (uint32_t i = raw_lo; i < raw_hi; i++) {
                uint32_t kv_idx = (raw_start + i) % raw_cap;
                const float *kv = (const float *)raw_kv->ptr + (uint64_t)kv_idx * head_dim;
                float dot = 0.0f;
                for (uint32_t j = 0; j < head_dim; j++) dot += qh[j] * kv[j];
                float s = dot * inv_scale;
                if (i == 0) s += sinks[h];
                exp_sum += sycl::exp(s - max_score);
            }
            for (uint32_t k = 0; k < top_k; k++) {
                uint32_t c = topk_ptr[k];
                if (c < n_comp) {
                    const float *kv = (const float *)comp_kv->ptr + (uint64_t)c * head_dim;
                    float dot = 0.0f;
                    for (uint32_t j = 0; j < head_dim; j++) dot += qh[j] * kv[j];
                    float s = dot * inv_scale;
                    exp_sum += sycl::exp(s - max_score);
                }
            }
            float inv_sum = exp_sum > 0.0f ? 1.0f / exp_sum : 0.0f;

            /* Weighted sum */
            for (uint32_t j = 0; j < head_dim; j++) head_out[j] = 0.0f;
            for (uint32_t i = raw_lo; i < raw_hi; i++) {
                uint32_t kv_idx = (raw_start + i) % raw_cap;
                const float *kv = (const float *)raw_kv->ptr + (uint64_t)kv_idx * head_dim;
                float dot = 0.0f;
                for (uint32_t j = 0; j < head_dim; j++) dot += qh[j] * kv[j];
                float s = dot * inv_scale;
                if (i == 0) s += sinks[h];
                float w = sycl::exp(s - max_score) * inv_sum;
                for (uint32_t j = 0; j < head_dim; j++) head_out[j] += w * kv[j];
            }
            for (uint32_t k = 0; k < top_k; k++) {
                uint32_t c = topk_ptr[k];
                if (c < n_comp) {
                    const float *kv = (const float *)comp_kv->ptr + (uint64_t)c * head_dim;
                    float dot = 0.0f;
                    for (uint32_t j = 0; j < head_dim; j++) dot += qh[j] * kv[j];
                    float s = dot * inv_scale;
                    float w = sycl::exp(s - max_score) * inv_sum;
                    for (uint32_t j = 0; j < head_dim; j++) head_out[j] += w * kv[j];
                }
            }
        });
        g_queue->wait_and_throw();
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL attention_indexed_mixed_batch_heads failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_attention_prefill_static_mixed_heads_tensor(
        ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
        uint64_t sinks_offset, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv, const ds4_gpu_tensor *comp_kv,
        uint32_t comp_kv_f16, uint32_t n_tokens, uint32_t n_comp,
        uint32_t window, uint32_t ratio, uint32_t n_head, uint32_t head_dim) {
    if (comp_kv_f16 || !heads || !q || !raw_kv || !model_map || n_tokens == 0 ||
        (n_comp != 0 && !comp_kv) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float) ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp * head_dim * sizeof(float)))
        return 0;
    const char *sptr = sycl_model_range_ptr(model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sptr) return 0;
    const float *sinks = (const float *)sptr;
    try {
        uint32_t pairs = n_tokens * n_head;
        g_queue->parallel_for(sycl::range<1>(pairs), [=](sycl::id<1> idx) {
            uint32_t p = (uint32_t)idx;
            uint32_t h = p % n_head;
            uint32_t t = p / n_head;
            const float *qh = (const float *)q->ptr + ((uint64_t)t * n_head + h) * head_dim;
            float *head_out = (float *)heads->ptr + ((uint64_t)t * n_head + h) * head_dim;
            float inv_scale = sycl::rsqrt((float)head_dim);

            /* Visible raw: positions 0..t with window */
            uint32_t raw_lo = 0, raw_hi = t + 1;
            if (window != 0 && t + 1 > window) raw_lo = t + 1 - window;

            /* Visible compressed: positions before current token */
            uint32_t visible_comp = (n_comp >= (t + 1u) / ratio) ? (t + 1u) / ratio : n_comp;

            /* Max score */
            float max_score = -INFINITY;
            for (uint32_t i = raw_lo; i < raw_hi; i++) {
                const float *kv = (const float *)raw_kv->ptr + (uint64_t)i * head_dim;
                float dot = 0.0f;
                for (uint32_t j = 0; j < head_dim; j++) dot += qh[j] * kv[j];
                float s = dot * inv_scale;
                if (i == 0) s += sinks[h];
                if (s > max_score) max_score = s;
            }
            for (uint32_t c = 0; c < visible_comp; c++) {
                const float *kv = (const float *)comp_kv->ptr + (uint64_t)c * head_dim;
                float dot = 0.0f;
                for (uint32_t j = 0; j < head_dim; j++) dot += qh[j] * kv[j];
                float s = dot * inv_scale;
                if (s > max_score) max_score = s;
            }

            /* Softmax sum */
            float exp_sum = 0.0f;
            for (uint32_t i = raw_lo; i < raw_hi; i++) {
                const float *kv = (const float *)raw_kv->ptr + (uint64_t)i * head_dim;
                float dot = 0.0f;
                for (uint32_t j = 0; j < head_dim; j++) dot += qh[j] * kv[j];
                float s = dot * inv_scale;
                if (i == 0) s += sinks[h];
                exp_sum += sycl::exp(s - max_score);
            }
            for (uint32_t c = 0; c < visible_comp; c++) {
                const float *kv = (const float *)comp_kv->ptr + (uint64_t)c * head_dim;
                float dot = 0.0f;
                for (uint32_t j = 0; j < head_dim; j++) dot += qh[j] * kv[j];
                float s = dot * inv_scale;
                exp_sum += sycl::exp(s - max_score);
            }
            float inv_sum = exp_sum > 0.0f ? 1.0f / exp_sum : 0.0f;

            /* Weighted sum */
            for (uint32_t j = 0; j < head_dim; j++) head_out[j] = 0.0f;
            for (uint32_t i = raw_lo; i < raw_hi; i++) {
                const float *kv = (const float *)raw_kv->ptr + (uint64_t)i * head_dim;
                float dot = 0.0f;
                for (uint32_t j = 0; j < head_dim; j++) dot += qh[j] * kv[j];
                float s = dot * inv_scale;
                if (i == 0) s += sinks[h];
                float w = sycl::exp(s - max_score) * inv_sum;
                for (uint32_t j = 0; j < head_dim; j++) head_out[j] += w * kv[j];
            }
            for (uint32_t c = 0; c < visible_comp; c++) {
                const float *kv = (const float *)comp_kv->ptr + (uint64_t)c * head_dim;
                float dot = 0.0f;
                for (uint32_t j = 0; j < head_dim; j++) dot += qh[j] * kv[j];
                float s = dot * inv_scale;
                float w = sycl::exp(s - max_score) * inv_sum;
                for (uint32_t j = 0; j < head_dim; j++) head_out[j] += w * kv[j];
            }
        });
        g_queue->wait_and_throw();
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL attention_prefill_static_mixed_heads failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_attention_prefill_masked_mixed_heads_tensor(
        ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
        uint64_t sinks_offset, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv, const ds4_gpu_tensor *comp_kv,
        uint32_t comp_kv_f16, const ds4_gpu_tensor *comp_mask,
        uint32_t n_tokens, uint32_t n_comp, uint32_t window,
        uint32_t ratio, uint32_t n_head, uint32_t head_dim) {
    if (comp_kv_f16 || !heads || !q || !raw_kv || !model_map || n_tokens == 0 ||
        (n_comp != 0 && (!comp_kv || !comp_mask)) ||
        sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        heads->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tokens * n_head * head_dim * sizeof(float) ||
        raw_kv->bytes < (uint64_t)n_tokens * head_dim * sizeof(float) ||
        (n_comp && comp_kv->bytes < (uint64_t)n_comp * head_dim * sizeof(float)) ||
        (n_comp && comp_mask->bytes < (uint64_t)n_tokens * n_comp * sizeof(float)))
        return 0;
    const char *sptr = sycl_model_range_ptr(model_map, sinks_offset, (uint64_t)n_head * sizeof(float), "attn_sinks");
    if (!sptr) return 0;
    const float *sinks = (const float *)sptr;
    try {
        uint32_t pairs = n_tokens * n_head;
        g_queue->parallel_for(sycl::range<1>(pairs), [=](sycl::id<1> idx) {
            uint32_t p = (uint32_t)idx;
            uint32_t h = p % n_head;
            uint32_t t = p / n_head;
            const float *qh = (const float *)q->ptr + ((uint64_t)t * n_head + h) * head_dim;
            float *head_out = (float *)heads->ptr + ((uint64_t)t * n_head + h) * head_dim;
            float inv_scale = sycl::rsqrt((float)head_dim);

            /* Visible raw: positions 0..t with window */
            uint32_t raw_lo = 0, raw_hi = t + 1;
            if (window != 0 && t + 1 > window) raw_lo = t + 1 - window;

            /* Visible compressed (all, but masked) */
            uint32_t visible_comp = (n_comp >= (t + 1u) / ratio) ? (t + 1u) / ratio : n_comp;
            const float *mask_row = (const float *)comp_mask->ptr + (uint64_t)t * n_comp;

            /* Max score */
            float max_score = -INFINITY;
            for (uint32_t i = raw_lo; i < raw_hi; i++) {
                const float *kv = (const float *)raw_kv->ptr + (uint64_t)i * head_dim;
                float dot = 0.0f;
                for (uint32_t j = 0; j < head_dim; j++) dot += qh[j] * kv[j];
                float s = dot * inv_scale;
                if (i == 0) s += sinks[h];
                if (s > max_score) max_score = s;
            }
            for (uint32_t c = 0; c < visible_comp; c++) {
                float add = mask_row[c];
                float s = -INFINITY;
                if (add > -1.0e20f) {
                    const float *kv = (const float *)comp_kv->ptr + (uint64_t)c * head_dim;
                    float dot = 0.0f;
                    for (uint32_t j = 0; j < head_dim; j++) dot += qh[j] * kv[j];
                    s = dot * inv_scale + add;
                }
                if (s > max_score) max_score = s;
            }

            /* Softmax sum */
            float exp_sum = 0.0f;
            for (uint32_t i = raw_lo; i < raw_hi; i++) {
                const float *kv = (const float *)raw_kv->ptr + (uint64_t)i * head_dim;
                float dot = 0.0f;
                for (uint32_t j = 0; j < head_dim; j++) dot += qh[j] * kv[j];
                float s = dot * inv_scale;
                if (i == 0) s += sinks[h];
                exp_sum += sycl::exp(s - max_score);
            }
            for (uint32_t c = 0; c < visible_comp; c++) {
                float add = mask_row[c];
                float s = -INFINITY;
                if (add > -1.0e20f) {
                    const float *kv = (const float *)comp_kv->ptr + (uint64_t)c * head_dim;
                    float dot = 0.0f;
                    for (uint32_t j = 0; j < head_dim; j++) dot += qh[j] * kv[j];
                    s = dot * inv_scale + add;
                    exp_sum += sycl::exp(s - max_score);
                }
            }
            float inv_sum = exp_sum > 0.0f ? 1.0f / exp_sum : 0.0f;

            /* Weighted sum */
            for (uint32_t j = 0; j < head_dim; j++) head_out[j] = 0.0f;
            for (uint32_t i = raw_lo; i < raw_hi; i++) {
                const float *kv = (const float *)raw_kv->ptr + (uint64_t)i * head_dim;
                float dot = 0.0f;
                for (uint32_t j = 0; j < head_dim; j++) dot += qh[j] * kv[j];
                float s = dot * inv_scale;
                if (i == 0) s += sinks[h];
                float w = sycl::exp(s - max_score) * inv_sum;
                for (uint32_t j = 0; j < head_dim; j++) head_out[j] += w * kv[j];
            }
            for (uint32_t c = 0; c < visible_comp; c++) {
                float add = mask_row[c];
                float s = -INFINITY;
                if (add > -1.0e20f) {
                    const float *kv = (const float *)comp_kv->ptr + (uint64_t)c * head_dim;
                    float dot = 0.0f;
                    for (uint32_t j = 0; j < head_dim; j++) dot += qh[j] * kv[j];
                    s = dot * inv_scale + add;
                    float w = sycl::exp(s - max_score) * inv_sum;
                    for (uint32_t j = 0; j < head_dim; j++) head_out[j] += w * kv[j];
                }
            }
        });
        g_queue->wait_and_throw();
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL attention_prefill_masked_mixed_heads failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_attention_output_q8_batch_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low,
        ds4_gpu_tensor *group_tmp, ds4_gpu_tensor *low_tmp,
        const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t out_b_offset,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups,
        uint64_t out_dim, const ds4_gpu_tensor *heads,
        uint32_t n_tokens) {
    (void)group_tmp; (void)low_tmp;
    if (!out || !low || !heads || !model_map ||
        group_dim == 0 || rank == 0 || n_groups == 0 || out_dim == 0 || n_tokens == 0) return 0;
    uint64_t low_dim = (uint64_t)n_groups * rank;
    uint64_t blocks_a = (group_dim + 31) / 32;
    uint64_t blocks_b = (low_dim + 31) / 32;
    uint64_t out_a_bytes = (uint64_t)n_groups * rank * blocks_a * 34;
    uint64_t out_b_bytes = out_dim * blocks_b * 34;
    if (out_a_offset > model_size || out_b_offset > model_size ||
        out_a_bytes > model_size - out_a_offset ||
        out_b_bytes > model_size - out_b_offset ||
        heads->bytes < (uint64_t)n_tokens * n_groups * group_dim * sizeof(float) ||
        low->bytes < (uint64_t)n_tokens * low_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float)) return 0;

    /* Project A (batched): launch one Q8_0 matmul per group for all tokens.
       heads  layout: [n_tokens][n_groups][group_dim] — strided per group
       low    layout: [n_tokens][n_groups][rank] — strided per group
       out_a  layout: [n_groups][rank][group_dim] (Q8) */
    for (uint32_t g = 0; g < n_groups; g++) {
        uint64_t group_offset = out_a_offset + (uint64_t)g * rank * blocks_a * 34;
        /* Build views: heads[:,g,:] is non-contiguous (strided by n_groups).
           Pre-quantize into a contiguous temp buffer for batch matmul. */
        uint64_t xq_bytes = n_tokens * blocks_a * 32;
        uint64_t xs_bytes = n_tokens * blocks_a * sizeof(float);
        int8_t *xq = (int8_t *)sycl::malloc_device(xq_bytes, *g_queue);
        float *xscale = (float *)sycl::malloc_device(xs_bytes, *g_queue);
        if (!xq || !xscale) {
            sycl::free(xq, *g_queue); sycl::free(xscale, *g_queue);
            return 0;
        }
        /* Quantize: strided read from heads[:,g,:] into contiguous xq/xscale */
        try {
            g_queue->submit([&](sycl::handler &h) {
                h.parallel_for(sycl::range<2>(blocks_a, n_tokens), [=](sycl::id<2> idx) {
                    uint32_t b = idx[0], t = idx[1];
                    const float *row = (const float *)heads->ptr +
                        ((uint64_t)t * n_groups + g) * group_dim + (uint64_t)b * 32;
                    int8_t *xq_row = xq + ((uint64_t)t * blocks_a + b) * 32;
                    float maxv = 0.0f;
                    for (int i = 0; i < 32; i++) {
                        float v = row[i];
                        if (v > maxv) maxv = v;
                        if (-v > maxv) maxv = -v;
                    }
                    float d = maxv / 127.0f;
                    if (d == 0.0f) d = 1.0f;
                    xscale[t * blocks_a + b] = d;
                    for (int i = 0; i < 32; i++) {
                        xq_row[i] = (int8_t)(row[i] / d);
                    }
                });
            });
            /* Batch Q8_0 matmul: all tokens at once for this group */
            g_queue->wait_and_throw();
            const char *wa_ptr = sycl_model_range_ptr(model_map, group_offset,
                rank * blocks_a * 34, "attn_out_a");
            if (!wa_ptr) { sycl::free(xq, *g_queue); sycl::free(xscale, *g_queue); return 0; }
            const uint8_t *w8_const = (const uint8_t *)wa_ptr;
            g_queue->submit([=](sycl::handler &h) {
                h.parallel_for(sycl::nd_range<2>(
                    sycl::range<2>(16 * ((rank + 15) / 16), n_tokens),
                    sycl::range<2>(16, 1)), [=](sycl::nd_item<2> item) {
                    uint64_t row_base = (uint64_t)item.get_group(0) * 16;
                    uint64_t tok = (uint64_t)item.get_group(1);
                    uint64_t sg_id = item.get_sub_group().get_group_id();
                    uint64_t row = row_base + sg_id;
                    if (row >= rank) return;

                    auto sg = item.get_sub_group();
                    uint32_t lane = sg.get_local_id();
                    const int8_t *xqr = xq + tok * blocks_a * 32;
                    const float *xsr = xscale + tok * blocks_a;
                    float acc = 0.0f;
                    for (uint64_t b = lane; b < blocks_a; b += 16) {
                        const uint8_t *block = w8_const + (row * blocks_a + b) * 34;
                        uint16_t d_bits = (uint16_t)block[0] | ((uint16_t)block[1] << 8);
                        float wscale = sycl_half_to_float(d_bits);
                        const int8_t *wq = (const int8_t *)(block + 2);
                        const int8_t *xqb = xqr + b * 32;
                        int dot = 0;
                        for (int i = 0; i < 32; i++) dot += (int)wq[i] * (int)xqb[i];
                        acc += wscale * xsr[b] * (float)dot;
                    }
                    acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
                    if (lane == 0) {
                        float *low_out = (float *)low->ptr +
                            ((uint64_t)tok * n_groups + g) * rank + row;
                        *low_out = acc;
                    }
                });
            });
            g_queue->wait_and_throw();
        } catch (sycl::exception &e) {
            fprintf(stderr, "ds4: SYCL attention_output_q8_batch project A failed: %s\n", e.what());
            sycl::free(xq, *g_queue); sycl::free(xscale, *g_queue);
            return 0;
        }
        sycl::free(xq, *g_queue);
        sycl::free(xscale, *g_queue);
    }

    /* Project B: low @ out_b^T -> out (single Q8 matmul) */
    return ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size,
                                       out_b_offset, low_dim, out_dim,
                                       low, n_tokens);
}

extern "C" int ds4_gpu_attention_output_q8_batch_f16_tensor(
        ds4_gpu_tensor *out_h, ds4_gpu_tensor *low,
        const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t out_b_offset,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups,
        uint64_t out_dim, const ds4_gpu_tensor *heads,
        uint32_t n_tokens) {
    (void)out_h; (void)low; (void)model_map; (void)model_size;
    (void)out_a_offset; (void)out_b_offset; (void)group_dim;
    (void)rank; (void)n_groups; (void)out_dim; (void)heads; (void)n_tokens;
    return 0;
}

extern "C" int ds4_gpu_attention_output_low_q8_tensor(
        ds4_gpu_tensor *low, const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t group_dim, uint64_t rank,
        uint32_t n_groups, const ds4_gpu_tensor *heads) {
    if (!low || !heads || !model_map || group_dim == 0 || rank == 0 || n_groups == 0)
        return 0;
    uint64_t low_dim = (uint64_t)n_groups * rank;
    uint64_t blocks_a = (group_dim + 31) / 32;
    uint64_t out_a_bytes = (uint64_t)n_groups * rank * blocks_a * 34;
    if (out_a_offset > model_size || out_a_bytes > model_size - out_a_offset ||
        heads->bytes < (uint64_t)n_groups * group_dim * sizeof(float) ||
        low->bytes < low_dim * sizeof(float))
        return 0;
    for (uint32_t g = 0; g < n_groups; g++) {
        uint64_t group_offset = out_a_offset + (uint64_t)g * rank * blocks_a * 34;
        ds4_gpu_tensor head_view = { (char *)heads->ptr + (uint64_t)g * group_dim * sizeof(float),
                                     group_dim * sizeof(float), 0 };
        ds4_gpu_tensor low_view = { (char *)low->ptr + (uint64_t)g * rank * sizeof(float),
                                    rank * sizeof(float), 0 };
        if (!ds4_gpu_matmul_q8_0_tensor(&low_view, model_map, model_size,
                                         group_offset, group_dim, rank,
                                         &head_view, 1)) return 0;
    }
    return 1;
}

/* =========================================================================
 * Router, Shared Expert, MoE stubs
 * ========================================================================= */
extern "C" int ds4_gpu_swiglu_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up, uint32_t n, float clamp, float weight) {
    if (!out || !gate || !up ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        gate->bytes < (uint64_t)n * sizeof(float) ||
        up->bytes < (uint64_t)n * sizeof(float)) return 0;
    try {
        g_queue->parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            uint32_t i = (uint32_t)idx;
            float g = ((const float *)gate->ptr)[i];
            float u = ((const float *)up->ptr)[i];
            if (clamp > 1.0e-6f) {
                g = sycl::fmin(g, clamp);
                u = sycl::fmin(sycl::fmax(u, -clamp), clamp);
            }
            float s = g / (1.0f + sycl::exp(-g));
            ((float *)out->ptr)[i] = s * u * weight;
        });
        g_queue->wait_and_throw();
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL swiglu failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_add_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b, uint32_t n) {
    if (!out || !a || !b ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        a->bytes < (uint64_t)n * sizeof(float) ||
        b->bytes < (uint64_t)n * sizeof(float)) return 0;
    try {
        g_queue->parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
            uint32_t i = (uint32_t)idx;
            ((float *)out->ptr)[i] = ((const float *)a->ptr)[i] + ((const float *)b->ptr)[i];
        });
        g_queue->wait_and_throw();
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL add failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_directional_steering_project_tensor(
        ds4_gpu_tensor *x, const ds4_gpu_tensor *directions,
        uint32_t layer, uint32_t width, uint32_t rows, float scale) {
    if (!x || !directions || width == 0 || rows == 0) return 0;
    if (x->bytes < (uint64_t)rows * width * sizeof(float) ||
        directions->bytes < (uint64_t)(layer + 1) * width * sizeof(float)) return 0;
    try {
        g_queue->parallel_for(sycl::range<1>(rows), [=](sycl::id<1> idx) {
            uint32_t row = (uint32_t)idx;
            float *xr = (float *)x->ptr + (uint64_t)row * width;
            const float *dir = (const float *)directions->ptr + (uint64_t)layer * width;
            float sum = 0.0f;
            for (uint32_t i = 0; i < width; i++) sum += xr[i] * dir[i];
            float coeff = scale * sum;
            for (uint32_t i = 0; i < width; i++) xr[i] -= coeff * dir[i];
        });
        g_queue->wait_and_throw();
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL directional_steering_project failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_router_select_tensor(
        ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
        ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size,
        uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows,
        uint32_t token, uint32_t n_expert, uint32_t n_expert_used,
        float expert_weight_scale, uint32_t n_expert_groups,
        uint32_t n_group_used, bool has_bias, bool hash_mode,
        const ds4_gpu_tensor *logits) {
    if (!g_queue || !selected || !weights || !probs || !logits ||
        n_expert == 0 || n_expert_used == 0 || n_expert_groups > 1 || n_group_used > 0)
        return 0;
    const float *bias = nullptr;
    const int32_t *hash = nullptr;
    if (has_bias && !hash_mode) {
        if (bias_offset > model_size || model_size - bias_offset < (uint64_t)n_expert * sizeof(float))
            return 0;
        bias = (const float *)sycl_model_range_ptr(model_map, bias_offset,
                                                    (uint64_t)n_expert * sizeof(float), "router_bias");
        if (!bias) return 0;
    }
    if (hash_mode) {
        uint64_t hash_bytes = (uint64_t)hash_rows * n_expert_used * sizeof(int32_t);
        if (hash_offset > model_size || hash_bytes > model_size - hash_offset)
            return 0;
        hash = (const int32_t *)sycl_model_range_ptr(model_map, hash_offset, hash_bytes, "router_hash");
        if (!hash) return 0;
    }
    int32_t *sel = (int32_t *)selected->ptr;
    float *w = (float *)weights->ptr;
    float *prob = (float *)probs->ptr;
    const float *log = (const float *)logits->ptr;
    try {
        g_queue->submit([&](sycl::handler &h) {
            h.single_task([=]() {
                sycl_router_select_body(sel, w, prob, bias, hash,
                                        log, nullptr, (int32_t)token,
                                        hash_rows, has_bias && !hash_mode, hash_mode,
                                        expert_weight_scale, n_expert, n_expert_used);
            });
        });
        g_queue->wait();
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL router_select failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_router_select_batch_tensor(
        ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
        ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size,
        uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows,
        uint32_t n_expert_groups, uint32_t n_group_used, bool has_bias,
        bool hash_mode, const ds4_gpu_tensor *logits,
        const ds4_gpu_tensor *tokens, uint32_t n_expert,
        uint32_t n_expert_used, float expert_weight_scale,
        uint32_t n_tokens) {
    if (!g_queue || !selected || !weights || !probs || !logits || !tokens ||
        n_tokens == 0 || n_expert == 0 || n_expert_used == 0 ||
        n_expert_groups > 1 || n_group_used > 0)
        return 0;
    const float *bias = nullptr;
    const int32_t *hash = nullptr;
    if (has_bias && !hash_mode) {
        if (bias_offset > model_size || model_size - bias_offset < (uint64_t)n_expert * sizeof(float))
            return 0;
        bias = (const float *)sycl_model_range_ptr(model_map, bias_offset,
                                                    (uint64_t)n_expert * sizeof(float), "router_bias");
        if (!bias) return 0;
    }
    if (hash_mode) {
        uint64_t hash_bytes = (uint64_t)hash_rows * n_expert_used * sizeof(int32_t);
        if (hash_offset > model_size || hash_bytes > model_size - hash_offset)
            return 0;
        hash = (const int32_t *)sycl_model_range_ptr(model_map, hash_offset, hash_bytes, "router_hash");
        if (!hash) return 0;
    }
    int32_t *sel = (int32_t *)selected->ptr;
    float *w = (float *)weights->ptr;
    float *prob = (float *)probs->ptr;
    const float *log = (const float *)logits->ptr;
    const int32_t *tok = (const int32_t *)tokens->ptr;
    try {
        g_queue->submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<1>(n_tokens), [=](sycl::id<1> tid) {
                uint32_t t = tid[0];
                int32_t *tsel = sel + (uint64_t)t * n_expert_used;
                float   *tw   = w   + (uint64_t)t * n_expert_used;
                float   *tprob= prob + (uint64_t)t * n_expert;
                const float *tlog = log + (uint64_t)t * n_expert;
                int32_t ttoken = tok ? tok[t] : 0;
                const float *tbias = bias;
                const int32_t *thash = hash;
                sycl_router_select_body(tsel, tw, tprob, tbias, thash,
                                        tlog, &ttoken, 0,
                                        hash_rows, has_bias && !hash_mode, hash_mode,
                                        expert_weight_scale, n_expert, n_expert_used);
            });
        });
        g_queue->wait();
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL router_select_batch failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_routed_moe_set_selected_override(const int32_t *selected,
                                                         uint32_t n_selected) {
    (void)selected; (void)n_selected;
    return 0;
}

/* Shared MoE launch: simplified f32 fallback path.
 * Supports gate_type=16 (IQ2_XXS) + down_type=10 (Q2_K).
 * Returns 1 on success, 0 on failure. */
static int sycl_routed_moe_launch(
        ds4_gpu_tensor *out, ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *down,
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint32_t gate_type, uint32_t down_type,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
        uint64_t down_expert_bytes, uint64_t down_row_bytes,
        uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        uint32_t n_total_expert, uint32_t n_expert, float clamp,
        const ds4_gpu_tensor *x, uint32_t layer_index, uint32_t n_tokens) {
    (void)layer_index;
    /* Validate parameters match the f32 fallback path */
    if (!g_queue || !out || !gate || !up || !mid || !down ||
        !model_map || !selected || !weights || !x ||
        n_tokens == 0 || n_total_expert == 0 || n_expert == 0 ||
        expert_in_dim % CUDA_QK_K != 0 || expert_mid_dim % CUDA_QK_K != 0 ||
        x->bytes < (uint64_t)n_tokens * expert_in_dim * sizeof(float) ||
        selected->bytes < (uint64_t)n_tokens * n_expert * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tokens * n_expert * sizeof(float) ||
        gate->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
        up->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
        mid->bytes < (uint64_t)n_tokens * n_expert * expert_mid_dim * sizeof(float) ||
        down->bytes < (uint64_t)n_tokens * n_expert * out_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tokens * out_dim * sizeof(float))
        return 0;
    /* Only IQ2_XXS gate + Q2_K down path (f32 fallback) */
    int q4k_path = (gate_type == 12u && down_type == 12u);
    if (!q4k_path && (gate_type != 16u || down_type != 10u)) return 0;
    uint64_t gate_bytes = (uint64_t)n_total_expert * gate_expert_bytes;
    uint64_t down_bytes = (uint64_t)n_total_expert * down_expert_bytes;
    if (gate_bytes > model_size - gate_offset ||
        gate_bytes > model_size - up_offset ||
        down_bytes > model_size - down_offset)
        return 0;
    /* Map model weights to device */
    const char *gate_w = sycl_model_range_ptr(model_map, gate_offset, gate_bytes, "moe_gate");
    const char *up_w   = sycl_model_range_ptr(model_map, up_offset,   gate_bytes, "moe_up");
    const char *down_w = sycl_model_range_ptr(model_map, down_offset, down_bytes, "moe_down");
    if (!gate_w || !up_w || !down_w) return 0;

    uint32_t pair_count = n_tokens * n_expert;
    const int32_t *sel_ptr = (const int32_t *)selected->ptr;
    const float   *wgt_ptr = (const float   *)weights->ptr;
    const float   *x_ptr   = (const float   *)x->ptr;
    float *gate_out = (float *)gate->ptr;
    float *up_out   = (float *)up->ptr;
    float *mid_out  = (float *)mid->ptr;
    float *down_out = (float *)down->ptr;
    float *out_ptr  = (float *)out->ptr;

    const uint32_t qk_k = CUDA_QK_K;

    /* Kernel 1: gate / up / mid for every (row, pair).
       Grid: (expert_mid_dim, pair_count), 1 work-item per element. */
    if (q4k_path) {
        /* Q4_K path: gate_type=12, down_type=12.
           For simplicity, dequantize each Q4_K block to f32 inline. */
        if (gate_row_bytes < sizeof(uint16_t) + 12 + 128) return 0;
        g_queue->submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<2>(expert_mid_dim, pair_count), [=](sycl::id<2> idx) {
                uint32_t row  = idx[0];
                uint32_t pair = idx[1];
                uint32_t tok  = pair / n_expert;
                uint32_t slot = pair - tok * n_expert;
                int32_t expert_i = sel_ptr[(uint64_t)tok * n_expert + slot];
                if (expert_i < 0) expert_i = 0;
                uint32_t expert = (uint32_t)expert_i;
                const uint8_t *gr = (const uint8_t *)(gate_w + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
                const uint8_t *ur = (const uint8_t *)(up_w   + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
                const float *xr = x_ptr + (uint64_t)tok * expert_in_dim;
                /* Dequantize Q4_K blocks and dot product */
                uint32_t nb = expert_in_dim / qk_k;
                float gate = 0.0f, up = 0.0f;
                for (uint32_t b = 0; b < nb; b++) {
                    const uint8_t *gblk = gr + b * (sizeof(uint16_t) + 12 + 128);
                    const uint8_t *ublk = ur + b * (sizeof(uint16_t) + 12 + 128);
                    const float *xblk = xr + (uint64_t)b * qk_k;
                    float gd = sycl_f16_to_f32(*(const uint16_t *)gblk);
                    float gdmin = sycl_f16_to_f32(*((const uint16_t *)gblk + 1));
                    float ud = sycl_f16_to_f32(*(const uint16_t *)ublk);
                    float udmin = sycl_f16_to_f32(*((const uint16_t *)ublk + 1));
                    const uint8_t *gsc = gblk + sizeof(uint16_t) * 2;
                    const uint8_t *gqs = gblk + sizeof(uint16_t) * 2 + 12;
                    const uint8_t *usc = ublk + sizeof(uint16_t) * 2;
                    const uint8_t *uqs = ublk + sizeof(uint16_t) * 2 + 12;
                    for (uint32_t il = 0; il < 16; il++) {
                        uint32_t chunk = il / 8u;
                        uint32_t pair_il = il & 1u;
                        uint32_t shift = ((il / 2u) & 3u) * 2u;
                        float gdl = gd * (float)((gsc[il] >> (shift & 3u)) & 3u); // Q4_K uses nibbles
                        float gml = gdmin * (float)((gsc[il] >> (4u + (shift & 3u))) & 3u); 
                        float udl = ud * (float)((usc[il] >> (shift & 3u)) & 3u);
                        float uml = udmin * (float)((usc[il] >> (4u + (shift & 3u))) & 3u);
                        const uint8_t *gqblk = gqs + chunk * 64u + pair_il * 32u;
                        const uint8_t *uqblk = uqs + chunk * 64u + pair_il * 32u;
                        const float *xblk_off = xblk + chunk * 128u + ((il % 8u) / 2u) * 32u + pair_il * 16u;
                        for (uint32_t i = 0; i < 16; i++) {
                            float gw = gdl * (float)((gqblk[i] >> (shift & 3u)) & 0x0fu) - gml;
                            float uw = udl * (float)((uqblk[i] >> (shift & 3u)) & 0x0fu) - uml;
                            gate += gw * xblk_off[i];
                            up   += uw * xblk_off[i];
                        }
                    }
                }
                float weight = wgt_ptr[(uint64_t)tok * n_expert + slot];
                float gate_act = (gate / (1.0f + expf(-gate)));
                if (clamp > 1.0e-6f) {
                    if (gate_act > clamp) gate_act = clamp;
                    if (up > clamp) up = clamp;
                    if (up < -clamp) up = -clamp;
                }
                uint64_t off = (uint64_t)pair * expert_mid_dim + row;
                gate_out[off] = gate;
                up_out[off]   = up;
                mid_out[off]  = gate_act * up * weight;
            });
        }).wait();
    } else {
        /* IQ2_XXS gate + Q2_K down path (gate_type=16, down_type=10) */
        g_queue->submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<2>(expert_mid_dim, pair_count), [=](sycl::id<2> idx) {
                uint32_t row  = idx[0];
                uint32_t pair = idx[1];
                uint32_t tok  = pair / n_expert;
                uint32_t slot = pair - tok * n_expert;
                int32_t expert_i = sel_ptr[(uint64_t)tok * n_expert + slot];
                if (expert_i < 0) expert_i = 0;
                uint32_t expert = (uint32_t)expert_i;
                const sycl_block_iq2_xxs *gr = (const sycl_block_iq2_xxs *)(gate_w + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
                const sycl_block_iq2_xxs *ur = (const sycl_block_iq2_xxs *)(up_w   + (uint64_t)expert * gate_expert_bytes + (uint64_t)row * gate_row_bytes);
                const float *xr = x_ptr + (uint64_t)tok * expert_in_dim;
                uint32_t nb = expert_in_dim / qk_k;
                float gate = sycl_iq2_xxs_dot_f32(gr, xr, nb);
                float up   = sycl_iq2_xxs_dot_f32(ur, xr, nb);
                float weight = wgt_ptr[(uint64_t)tok * n_expert + slot];
                float gate_act = gate * (1.0f / (1.0f + expf(-gate)));
                if (clamp > 1.0e-6f) {
                    if (gate_act > clamp) gate_act = clamp;
                    if (up > clamp) up = clamp;
                    if (up < -clamp) up = -clamp;
                }
                uint64_t off = (uint64_t)pair * expert_mid_dim + row;
                gate_out[off] = gate;
                up_out[off]   = up;
                mid_out[off]  = gate_act * up * weight;
            });
        }).wait();
    }

    /* Kernel 2: down projection for every (row, pair).
       Grid: (out_dim, pair_count), 1 work-item per element. */
    g_queue->submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<2>(out_dim, pair_count), [=](sycl::id<2> idx) {
            uint32_t row  = idx[0];
            uint32_t pair = idx[1];
            uint32_t tok  = pair / n_expert;
            uint32_t slot = pair - tok * n_expert;
            int32_t expert_i = sel_ptr[(uint64_t)tok * n_expert + slot];
            if (expert_i < 0) expert_i = 0;
            uint32_t expert = (uint32_t)expert_i;
            const sycl_block_q2_K *wr = (const sycl_block_q2_K *)(down_w + (uint64_t)expert * down_expert_bytes + (uint64_t)row * down_row_bytes);
            const float *xr = mid_out + (uint64_t)pair * expert_mid_dim;
            uint32_t nb = expert_mid_dim / qk_k;
            float acc = 0.0f;
            for (uint32_t b = 0; b < nb; b++) {
                const sycl_block_q2_K *xb = wr + b;
                float d    = sycl_f16_to_f32(xb->d);
                float dmin = sycl_f16_to_f32(xb->dmin);
                for (uint32_t il = 0; il < 16; il++) {
                    uint32_t chunk = il / 8u;
                    uint32_t pair_il = il & 1u;
                    uint32_t shift = ((il / 2u) & 3u) * 2u;
                    uint8_t sc = xb->scales[il];
                    float dl = d * (float)(sc & 0x0fu);
                    float ml = dmin * (float)(sc >> 4);
                    const uint8_t *q = xb->qs + 32u * chunk + 16u * pair_il;
                    const float *xf = xr + (uint64_t)b * qk_k + chunk * 128u + ((il % 8u) / 2u) * 32u + pair_il * 16u;
                    for (uint32_t i = 0; i < 16; i++) {
                        float w = dl * (float)((q[i] >> shift) & 3u) - ml;
                        acc += w * xf[i];
                    }
                }
            }
            down_out[(uint64_t)pair * out_dim + row] = acc;
        });
    }).wait();

    /* Kernel 3: sum across experts */
    uint64_t n = (uint64_t)n_tokens * out_dim;
    g_queue->submit([&](sycl::handler &h) {
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> gid) {
            uint32_t tok = gid / out_dim;
            uint32_t row = (uint32_t)(gid - (uint64_t)tok * out_dim);
            float acc = 0.0f;
            for (uint32_t e = 0; e < n_expert; e++)
                acc += down_out[((uint64_t)tok * n_expert + e) * out_dim + row];
            out_ptr[gid] = acc;
        });
    }).wait();

    return 1;
}

extern "C" int ds4_gpu_routed_moe_one_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *experts,
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint32_t gate_type, uint32_t down_type,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
        uint64_t down_expert_bytes, uint64_t down_row_bytes,
        uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        uint32_t n_total_expert, uint32_t n_expert, float clamp,
        const ds4_gpu_tensor *x, uint32_t layer_index) {
    if (!out || !gate || !up || !mid || !experts) return 0;
    return sycl_routed_moe_launch(
            out, gate, up, mid, experts,
            model_map, model_size,
            gate_offset, up_offset, down_offset,
            gate_type, down_type,
            gate_expert_bytes, gate_row_bytes,
            down_expert_bytes, down_row_bytes,
            expert_in_dim, expert_mid_dim, out_dim,
            selected, weights,
            n_total_expert, n_expert, clamp,
            x, layer_index, 1);
}

extern "C" int ds4_gpu_routed_moe_batch_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *gate,
        ds4_gpu_tensor *up, ds4_gpu_tensor *mid, ds4_gpu_tensor *experts,
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint32_t gate_type, uint32_t down_type,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
        uint64_t down_expert_bytes, uint64_t down_row_bytes,
        uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        uint32_t n_total_expert, uint32_t n_expert, float clamp,
        const ds4_gpu_tensor *x, uint32_t layer_index,
        uint32_t n_tokens, bool *mid_is_f16) {
    if (!out || !gate || !up || !mid || !experts) return 0;
    if (mid_is_f16) *mid_is_f16 = false;
    return sycl_routed_moe_launch(
            out, gate, up, mid, experts,
            model_map, model_size,
            gate_offset, up_offset, down_offset,
            gate_type, down_type,
            gate_expert_bytes, gate_row_bytes,
            down_expert_bytes, down_row_bytes,
            expert_in_dim, expert_mid_dim, out_dim,
            selected, weights,
            n_total_expert, n_expert, clamp,
            x, layer_index, n_tokens);
}

/* =========================================================================
 * Hyper-Connection kernels
 * ========================================================================= */

/* Device helper: Sinkhorn split for n_hc=4 (signature/code matches CUDA). */
/* Defined as static inline for use within SYCL kernel lambdas. */

extern "C" int ds4_gpu_hc_split_sinkhorn_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *mix,
        const void *model_map, uint64_t model_size,
        uint64_t scale_offset, uint64_t base_offset,
        uint32_t n_hc, uint32_t sinkhorn_iters, float eps) {
    if (!g_queue || !out || !mix || !model_map || n_hc != 4) return 0;
    uint64_t mix_hc = (uint64_t)(2u * n_hc + n_hc * n_hc);
    uint64_t n_rows = mix->bytes / ((uint64_t)mix_hc * sizeof(float));
    if (n_rows == 0 || out->bytes < n_rows * mix_hc * sizeof(float)) return 0;
    const float *scale = (const float *)sycl_model_range_ptr(model_map, scale_offset, 3 * sizeof(float), "hc_scale");
    const float *base  = (const float *)sycl_model_range_ptr(model_map, base_offset, mix->bytes, "hc_base");
    if (!scale || !base) return 0;
    float *out_ptr = (float *)out->ptr;
    const float *mix_ptr = (const float *)mix->ptr;
    try {
        g_queue->submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<1>(n_rows), [=](sycl::id<1> idx) {
                uint32_t row = idx[0];
                const float *lmix = mix_ptr + (uint64_t)row * mix_hc;
                float *lout = out_ptr + (uint64_t)row * mix_hc;
                float pre_scale = scale[0], post_scale = scale[1], comb_scale = scale[2];
                for (int i = 0; i < 4; i++) {
                    float z = lmix[i] * pre_scale + base[i];
                    lout[i] = 1.0f / (1.0f + expf(-z)) + eps;
                }
                for (int i = 0; i < 4; i++) {
                    float z = lmix[4 + i] * post_scale + base[4 + i];
                    lout[4 + i] = 2.0f / (1.0f + expf(-z));
                }
                float c[16];
                for (int r = 0; r < 4; r++) {
                    float m = -100.0f; /* softmax stable max */
                    for (int col = 0; col < 4; col++) {
                        float v = lmix[8 + r * 4 + col] * comb_scale + base[8 + r * 4 + col];
                        c[r * 4 + col] = v;
                        if (v > m) m = v;
                    }
                    float s = 0.0f;
                    for (int col = 0; col < 4; col++) {
                        float ev = expf(c[r * 4 + col] - m);
                        c[r * 4 + col] = ev;
                        s += ev;
                    }
                    for (int col = 0; col < 4; col++) c[r * 4 + col] = c[r * 4 + col] / s + eps;
                }
                for (int col = 0; col < 4; col++) {
                    float s = eps;
                    for (int r = 0; r < 4; r++) s += c[r * 4 + col];
                    for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
                }
                for (uint32_t iter = 1; iter < sinkhorn_iters; iter++) {
                    for (int r = 0; r < 4; r++) {
                        float s = eps;
                        for (int col = 0; col < 4; col++) s += c[r * 4 + col];
                        for (int col = 0; col < 4; col++) c[r * 4 + col] /= s;
                    }
                    for (int col = 0; col < 4; col++) {
                        float s = eps;
                        for (int r = 0; r < 4; r++) s += c[r * 4 + col];
                        for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
                    }
                }
                for (int i = 0; i < 16; i++) lout[8 + i] = c[i];
            });
        });
        g_queue->wait();
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL hc_split_sinkhorn failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_hc_weighted_sum_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *weights, uint32_t n_embd, uint32_t n_hc) {
    if (!g_queue || !out || !residual_hc || !weights ||
        n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = residual_hc->bytes / ((uint64_t)n_hc * n_embd * sizeof(float));
    if (n_tokens == 0) return 0;
    uint32_t weight_stride = n_hc; /* separate weights tensor */
    float *out_ptr = (float *)out->ptr;
    const float *x_ptr = (const float *)residual_hc->ptr;
    const float *w_ptr = (const float *)weights->ptr;
    try {
        g_queue->submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<1>((uint64_t)n_embd * n_tokens), [=](sycl::id<1> gid) {
                uint64_t d = gid % n_embd;
                uint32_t t = gid / n_embd;
                float acc = 0.0f;
                for (uint32_t h = 0; h < n_hc; h++) {
                    acc += x_ptr[(uint64_t)t * n_hc * n_embd + (uint64_t)h * n_embd + d] *
                           w_ptr[(uint64_t)t * weight_stride + h];
                }
                out_ptr[(uint64_t)t * n_embd + d] = acc;
            });
        });
        g_queue->wait();
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL hc_weighted_sum failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_hc_weighted_sum_split_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc) {
    if (!g_queue || !out || !residual_hc || !split ||
        n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = residual_hc->bytes / ((uint64_t)n_hc * n_embd * sizeof(float));
    if (n_tokens == 0) return 0;
    uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    float *out_ptr = (float *)out->ptr;
    const float *x_ptr = (const float *)residual_hc->ptr;
    const float *s_ptr = (const float *)split->ptr;
    try {
        g_queue->submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<1>((uint64_t)n_embd * n_tokens), [=](sycl::id<1> gid) {
                uint64_t d = gid % n_embd;
                uint32_t t = gid / n_embd;
                float acc = 0.0f;
                for (uint32_t h = 0; h < n_hc; h++) {
                    acc += x_ptr[(uint64_t)t * n_hc * n_embd + (uint64_t)h * n_embd + d] *
                           s_ptr[(uint64_t)t * mix_hc + h];
                }
                out_ptr[(uint64_t)t * n_embd + d] = acc;
            });
        });
        g_queue->wait();
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL hc_weighted_sum_split failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_hc_split_weighted_sum_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *split,
        const ds4_gpu_tensor *mix, const ds4_gpu_tensor *residual_hc,
        const void *model_map, uint64_t model_size,
        uint64_t scale_offset, uint64_t base_offset,
        uint32_t n_embd, uint32_t n_hc, uint32_t sinkhorn_iters, float eps) {
    if (!g_queue || !out || !split || !mix || !residual_hc || !model_map ||
        n_embd == 0 || n_hc != 4) return 0;
    uint32_t n_rows = mix->bytes / (24ull * sizeof(float));
    if (n_rows == 0 || n_rows != residual_hc->bytes / ((uint64_t)n_hc * n_embd * sizeof(float)))
        return 0;
    const float *scale = (const float *)sycl_model_range_ptr(model_map, scale_offset, 3 * sizeof(float), "hc_scale");
    const float *base  = (const float *)sycl_model_range_ptr(model_map, base_offset, 24ull * sizeof(float), "hc_base");
    if (!scale || !base) return 0;
    float *out_ptr = (float *)out->ptr;
    float *sp_ptr = (float *)split->ptr;
    const float *mx_ptr = (const float *)mix->ptr;
    const float *res_ptr = (const float *)residual_hc->ptr;
    try {
        g_queue->submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<1>(n_rows), [=](sycl::id<1> idx) {
                uint32_t t = idx[0];
                float *sp = sp_ptr + (uint64_t)t * 24;
                const float *lmix = mx_ptr + (uint64_t)t * 24;
                /* hc4_split_one inline */
                float pre_scale = scale[0], post_scale = scale[1], comb_scale = scale[2];
                for (int i = 0; i < 4; i++) {
                    float z = lmix[i] * pre_scale + base[i];
                    sp[i] = 1.0f / (1.0f + expf(-z)) + eps;
                }
                for (int i = 0; i < 4; i++) {
                    float z = lmix[4 + i] * post_scale + base[4 + i];
                    sp[4 + i] = 2.0f / (1.0f + expf(-z));
                }
                float c[16];
                for (int r = 0; r < 4; r++) {
                    float m = -100.0f;
                    for (int col = 0; col < 4; col++) {
                        float v = lmix[8 + r * 4 + col] * comb_scale + base[8 + r * 4 + col];
                        c[r * 4 + col] = v;
                        if (v > m) m = v;
                    }
                    float s = 0.0f;
                    for (int col = 0; col < 4; col++) {
                        float ev = expf(c[r * 4 + col] - m);
                        c[r * 4 + col] = ev;
                        s += ev;
                    }
                    for (int col = 0; col < 4; col++) c[r * 4 + col] = c[r * 4 + col] / s + eps;
                }
                for (int col = 0; col < 4; col++) {
                    float s = eps;
                    for (int r = 0; r < 4; r++) s += c[r * 4 + col];
                    for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
                }
                for (uint32_t iter = 1; iter < sinkhorn_iters; iter++) {
                    for (int r = 0; r < 4; r++) {
                        float s = eps;
                        for (int col = 0; col < 4; col++) s += c[r * 4 + col];
                        for (int col = 0; col < 4; col++) c[r * 4 + col] /= s;
                    }
                    for (int col = 0; col < 4; col++) {
                        float s = eps;
                        for (int r = 0; r < 4; r++) s += c[r * 4 + col];
                        for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
                    }
                }
                for (int i = 0; i < 16; i++) sp[8 + i] = c[i];
                /* Weighted sum: 1 block per token, 256 threads, cooperatively reduce over n_embd */
                float acc = 0.0f;
                for (uint32_t col = 0; col < n_embd; col++) {
                    acc = 0.0f;
                    for (uint32_t h = 0; h < 4; h++) {
                        acc += res_ptr[(uint64_t)t * 4u * n_embd + (uint64_t)h * n_embd + col] * sp[h];
                    }
                    out_ptr[(uint64_t)t * n_embd + col] = acc;
                }
            });
        });
        g_queue->wait();
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL hc_split_weighted_sum failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_hc_split_weighted_sum_norm_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *norm_out,
        ds4_gpu_tensor *split, const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void *model_map, uint64_t model_size,
        uint64_t scale_offset, uint64_t base_offset,
        uint64_t norm_weight_offset,
        uint32_t n_embd, uint32_t n_hc, uint32_t sinkhorn_iters,
        float eps, float norm_eps) {
    if (!g_queue || !out || !norm_out || !split || !mix || !residual_hc || !model_map ||
        n_embd == 0 || n_hc != 4) return 0;
    uint32_t n_rows = mix->bytes / (24ull * sizeof(float));
    if (n_rows == 0 || n_rows != residual_hc->bytes / ((uint64_t)n_hc * n_embd * sizeof(float)))
        return 0;
    const float *scale = (const float *)sycl_model_range_ptr(model_map, scale_offset, 3 * sizeof(float), "hc_scale");
    const float *base  = (const float *)sycl_model_range_ptr(model_map, base_offset, 24ull * sizeof(float), "hc_base");
    const float *norm_w = (const float *)sycl_model_range_ptr(model_map, norm_weight_offset,
                                                              (uint64_t)n_embd * sizeof(float), "hc_norm_w");
    if (!scale || !base || !norm_w) return 0;
    float *out_ptr = (float *)out->ptr;
    float *no_ptr = (float *)norm_out->ptr;
    float *sp_ptr = (float *)split->ptr;
    const float *mx_ptr = (const float *)mix->ptr;
    const float *res_ptr = (const float *)residual_hc->ptr;
    try {
        g_queue->submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<1>(n_rows), [=](sycl::id<1> idx) {
                uint32_t t = idx[0];
                float *sp = sp_ptr + (uint64_t)t * 24;
                const float *lmix = mx_ptr + (uint64_t)t * 24;
                /* hc4_split_one inline */
                float pre_scale = scale[0], post_scale = scale[1], comb_scale = scale[2];
                for (int i = 0; i < 4; i++) {
                    float z = lmix[i] * pre_scale + base[i];
                    sp[i] = 1.0f / (1.0f + expf(-z)) + eps;
                }
                for (int i = 0; i < 4; i++) {
                    float z = lmix[4 + i] * post_scale + base[4 + i];
                    sp[4 + i] = 2.0f / (1.0f + expf(-z));
                }
                float c[16];
                for (int r = 0; r < 4; r++) {
                    float m = -100.0f;
                    for (int col = 0; col < 4; col++) {
                        float v = lmix[8 + r * 4 + col] * comb_scale + base[8 + r * 4 + col];
                        c[r * 4 + col] = v;
                        if (v > m) m = v;
                    }
                    float s = 0.0f;
                    for (int col = 0; col < 4; col++) {
                        float ev = expf(c[r * 4 + col] - m);
                        c[r * 4 + col] = ev;
                        s += ev;
                    }
                    for (int col = 0; col < 4; col++) c[r * 4 + col] = c[r * 4 + col] / s + eps;
                }
                for (int col = 0; col < 4; col++) {
                    float s = eps;
                    for (int r = 0; r < 4; r++) s += c[r * 4 + col];
                    for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
                }
                for (uint32_t iter = 1; iter < sinkhorn_iters; iter++) {
                    for (int r = 0; r < 4; r++) {
                        float s = eps;
                        for (int col = 0; col < 4; col++) s += c[r * 4 + col];
                        for (int col = 0; col < 4; col++) c[r * 4 + col] /= s;
                    }
                    for (int col = 0; col < 4; col++) {
                        float s = eps;
                        for (int r = 0; r < 4; r++) s += c[r * 4 + col];
                        for (int r = 0; r < 4; r++) c[r * 4 + col] /= s;
                    }
                }
                for (int i = 0; i < 16; i++) sp[8 + i] = c[i];
                /* Weighted sum + RMS norm (single-thread per token) */
                float sum_sq = 0.0f;
                for (uint32_t col = 0; col < n_embd; col++) {
                    float acc = 0.0f;
                    for (uint32_t h = 0; h < 4; h++) {
                        acc += res_ptr[(uint64_t)t * 4u * n_embd + (uint64_t)h * n_embd + col] * sp[h];
                    }
                    out_ptr[(uint64_t)t * n_embd + col] = acc;
                    sum_sq += acc * acc;
                }
                float norm_scale = 1.0f / sqrtf(sum_sq / (float)n_embd + norm_eps);
                for (uint32_t col = 0; col < n_embd; col++) {
                    float v = out_ptr[(uint64_t)t * n_embd + col];
                    no_ptr[(uint64_t)t * n_embd + col] = v * norm_scale * norm_w[col];
                }
            });
        });
        g_queue->wait();
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL hc_split_weighted_sum_norm failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_output_hc_weights_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *pre,
        const void *model_map, uint64_t model_size,
        uint64_t scale_offset, uint64_t base_offset,
        uint32_t n_hc, float eps) {
    if (!g_queue || !out || !pre || !model_map || n_hc == 0) return 0;
    uint32_t n_tokens = pre->bytes / ((uint64_t)n_hc * sizeof(float));
    if (n_tokens == 0 || out->bytes < (uint64_t)n_tokens * n_hc * sizeof(float))
        return 0;
    const float *scale = (const float *)sycl_model_range_ptr(model_map, scale_offset, 3 * sizeof(float), "hc_scale");
    const float *base  = (const float *)sycl_model_range_ptr(model_map, base_offset, (uint64_t)n_hc * sizeof(float), "hc_base");
    if (!scale || !base) return 0;
    float *out_ptr = (float *)out->ptr;
    const float *pre_ptr = (const float *)pre->ptr;
    try {
        g_queue->submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<1>((uint64_t)n_tokens * n_hc), [=](sycl::id<1> gid) {
                uint32_t hc = gid % n_hc;
                float z = pre_ptr[gid] * scale[0] + base[hc];
                out_ptr[gid] = 1.0f / (1.0f + expf(-z)) + eps;
            });
        });
        g_queue->wait();
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL output_hc_weights failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_hc_expand_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *post,
        const ds4_gpu_tensor *comb, uint32_t n_embd, uint32_t n_hc) {
    if (!g_queue || !out_hc || !block_out || !residual_hc || !post || !comb ||
        n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = block_out->bytes / ((uint64_t)n_embd * sizeof(float));
    if (n_tokens == 0) return 0;
    const uint64_t hc_bytes = (uint64_t)n_tokens * n_hc * n_embd * sizeof(float);
    if (out_hc->bytes < hc_bytes || residual_hc->bytes < hc_bytes) return 0;
    const float *comb_ptr = (const float *)comb->ptr;
    const float *post_ptr = (const float *)post->ptr;
    float *o_ptr = (float *)out_hc->ptr;
    const float *bo_ptr = (const float *)block_out->ptr;
    const float *res_ptr = (const float *)residual_hc->ptr;
    uint32_t post_stride = n_hc;
    uint32_t comb_stride = (uint32_t)(n_hc * n_hc);
    try {
        g_queue->submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<1>((uint64_t)n_tokens * n_hc * n_embd), [=](sycl::id<1> gid) {
                uint32_t d = gid % n_embd;
                uint64_t tmp = gid / n_embd;
                uint32_t dst_hc = tmp % n_hc;
                uint32_t t = tmp / n_hc;
                float bv = bo_ptr[(uint64_t)t * n_embd + d];
                float acc = bv * post_ptr[(uint64_t)t * post_stride + dst_hc];
                for (uint32_t src = 0; src < n_hc; src++) {
                    float cv = comb_ptr[(uint64_t)t * comb_stride + dst_hc + (uint64_t)src * n_hc];
                    float rv = res_ptr[(uint64_t)t * n_hc * n_embd + (uint64_t)src * n_embd + d];
                    acc += cv * rv;
                }
                o_ptr[(uint64_t)t * n_hc * n_embd + (uint64_t)dst_hc * n_embd + d] = acc;
            });
        });
        g_queue->wait();
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL hc_expand failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_hc_expand_split_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split,
        uint32_t n_embd, uint32_t n_hc) {
    if (!g_queue || !out_hc || !block_out || !residual_hc || !split ||
        n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = block_out->bytes / ((uint64_t)n_embd * sizeof(float));
    if (n_tokens == 0) return 0;
    const uint64_t hc_bytes = (uint64_t)n_tokens * n_hc * n_embd * sizeof(float);
    if (out_hc->bytes < hc_bytes || residual_hc->bytes < hc_bytes) return 0;
    uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    const float *s_ptr = (const float *)split->ptr;
    const float *post_p = s_ptr + n_hc;
    const float *comb_p = s_ptr + 2u * n_hc;
    float *o_ptr = (float *)out_hc->ptr;
    const float *bo_ptr = (const float *)block_out->ptr;
    const float *res_ptr = (const float *)residual_hc->ptr;
    try {
        g_queue->submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<1>((uint64_t)n_tokens * n_hc * n_embd), [=](sycl::id<1> gid) {
                uint32_t d = gid % n_embd;
                uint64_t tmp = gid / n_embd;
                uint32_t dst_hc = tmp % n_hc;
                uint32_t t = tmp / n_hc;
                float bv = bo_ptr[(uint64_t)t * n_embd + d];
                float acc = bv * post_p[(uint64_t)t * mix_hc + dst_hc];
                for (uint32_t src = 0; src < n_hc; src++) {
                    float cv = comb_p[(uint64_t)t * mix_hc + dst_hc + (uint64_t)src * n_hc];
                    float rv = res_ptr[(uint64_t)t * n_hc * n_embd + (uint64_t)src * n_embd + d];
                    acc += cv * rv;
                }
                o_ptr[(uint64_t)t * n_hc * n_embd + (uint64_t)dst_hc * n_embd + d] = acc;
            });
        });
        g_queue->wait();
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL hc_expand_split failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_hc_expand_split_half_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out_h,
        const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split,
        uint32_t n_embd, uint32_t n_hc) {
    (void)out_hc; (void)block_out_h; (void)residual_hc; (void)split;
    (void)n_embd; (void)n_hc;
    fprintf(stderr, "ds4: SYCL hc_expand_split_half not implemented\n");
    return 0;
}

extern "C" int ds4_gpu_hc_expand_add_split_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split,
        uint32_t n_embd, uint32_t n_hc) {
    if (!g_queue || !out_hc || !block_out || !block_add || !residual_hc || !split ||
        n_embd == 0 || n_hc == 0) return 0;
    uint32_t n_tokens = block_out->bytes / ((uint64_t)n_embd * sizeof(float));
    if (n_tokens == 0) return 0;
    const uint64_t hc_bytes = (uint64_t)n_tokens * n_hc * n_embd * sizeof(float);
    if (out_hc->bytes < hc_bytes || residual_hc->bytes < hc_bytes) return 0;
    uint32_t mix_hc = 2u * n_hc + n_hc * n_hc;
    const float *s_ptr = (const float *)split->ptr;
    const float *post_p = s_ptr + n_hc;
    const float *comb_p = s_ptr + 2u * n_hc;
    float *o_ptr = (float *)out_hc->ptr;
    const float *bo_ptr = (const float *)block_out->ptr;
    const float *ba_ptr = (const float *)block_add->ptr;
    const float *res_ptr = (const float *)residual_hc->ptr;
    try {
        g_queue->submit([&](sycl::handler &h) {
            h.parallel_for(sycl::range<1>((uint64_t)n_tokens * n_hc * n_embd), [=](sycl::id<1> gid) {
                uint32_t d = gid % n_embd;
                uint64_t tmp = gid / n_embd;
                uint32_t dst_hc = tmp % n_hc;
                uint32_t t = tmp / n_hc;
                float bv = bo_ptr[(uint64_t)t * n_embd + d] + ba_ptr[(uint64_t)t * n_embd + d];
                float acc = bv * post_p[(uint64_t)t * mix_hc + dst_hc];
                for (uint32_t src = 0; src < n_hc; src++) {
                    float cv = comb_p[(uint64_t)t * mix_hc + dst_hc + (uint64_t)src * n_hc];
                    float rv = res_ptr[(uint64_t)t * n_hc * n_embd + (uint64_t)src * n_embd + d];
                    acc += cv * rv;
                }
                o_ptr[(uint64_t)t * n_hc * n_embd + (uint64_t)dst_hc * n_embd + d] = acc;
            });
        });
        g_queue->wait();
        return 1;
    } catch (sycl::exception &e) {
        fprintf(stderr, "ds4: SYCL hc_expand_add_split failed: %s\n", e.what());
        return 0;
    }
}

extern "C" int ds4_gpu_hc_expand_add_split_half_add_tensor(
        ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add_h,
        const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split,
        uint32_t n_embd, uint32_t n_hc) {
    (void)out_hc; (void)block_out; (void)block_add_h;
    (void)residual_hc; (void)split; (void)n_embd; (void)n_hc;
    fprintf(stderr, "ds4: SYCL hc_expand_add_split_half_add not implemented\n");
    return 0;
}

/* Fallback: decompose fused q8_0 matmul + hc_expand into separate steps. */
extern "C" int ds4_gpu_shared_down_hc_expand_q8_0_tensor(
        ds4_gpu_tensor *out_hc, ds4_gpu_tensor *shared_out,
        const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *shared_mid, const ds4_gpu_tensor *routed_out,
        const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split,
        uint32_t n_embd, uint32_t n_hc) {
    if (!g_queue || !out_hc || !shared_out || !model_map || !shared_mid ||
        !routed_out || !residual_hc || !split ||
        in_dim == 0 || out_dim == 0 || n_embd == 0 || n_hc == 0)
        return 0;
    if (!ds4_gpu_matmul_q8_0_tensor(shared_out, model_map, model_size,
                                     weight_offset, in_dim, out_dim,
                                     shared_mid, 1)) return 0;
    return ds4_gpu_hc_expand_add_split_tensor(out_hc, shared_out, routed_out,
                                              residual_hc, split, n_embd, n_hc);
}

extern "C" int ds4_gpu_matmul_q8_0_hc_expand_tensor(
        ds4_gpu_tensor *out_hc, ds4_gpu_tensor *block_out,
        const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t n_embd, uint32_t n_hc) {
    if (!g_queue || !out_hc || !block_out || !model_map || !x ||
        !residual_hc || !split ||
        in_dim == 0 || out_dim == 0 || n_embd == 0 || n_hc == 0 ||
        out_dim != n_embd)
        return 0;
    if (!ds4_gpu_matmul_q8_0_tensor(block_out, model_map, model_size,
                                     weight_offset, in_dim, out_dim, x, 1))
        return 0;
    return ds4_gpu_hc_expand_split_tensor(out_hc, block_out, residual_hc,
                                          split, n_embd, n_hc);
}

/* Always-provided helpers (non-Metal path) */
extern "C" int ds4_gpu_stream_expert_cache_prepare_selected_batch(
        const ds4_gpu_stream_expert_table *table,
        const int32_t *selected_ids, uint32_t n_tokens, uint32_t n_selected) {
    if (!table || !selected_ids || n_tokens == 0 || n_selected == 0) return 0;
    /* De-duplicate experts across all tokens */
    const void *model_map = table->model_map;
    uint64_t model_size = table->model_size;
    uint32_t layer = table->layer;
    uint32_t n_total_expert = table->n_total_expert;
    uint64_t gate_offset = table->gate_offset;
    uint64_t up_offset = table->up_offset;
    uint64_t down_offset = table->down_offset;
    uint64_t gate_expert_bytes = table->gate_expert_bytes;
    uint64_t down_expert_bytes = table->down_expert_bytes;
    if (n_total_expert == 0 || gate_expert_bytes == 0 || down_expert_bytes == 0)
        return 0;

    uint32_t total_slots = n_tokens * n_selected;
    std::vector<int32_t> compact;
    compact.reserve(total_slots);
    for (uint32_t i = 0; i < total_slots; i++) {
        int32_t eid = selected_ids[i];
        if (eid < 0 || (uint32_t)eid >= n_total_expert) continue;
        bool found = false;
        for (auto c : compact) { if (c == eid) { found = true; break; } }
        if (!found) compact.push_back(eid);
    }
    uint32_t compact_count = (uint32_t)compact.size();
    if (compact_count == 0) return 0;

    std::vector<int32_t> slot_ids(total_slots);
    for (uint32_t i = 0; i < total_slots; i++) {
        int32_t eid = selected_ids[i];
        int32_t idx = -1;
        for (uint32_t j = 0; j < compact_count; j++) {
            if (compact[j] == eid) { idx = (int32_t)j; break; }
        }
        slot_ids[i] = (idx >= 0) ? idx : 0;
    }

    return sycl_stream_selected_cache_begin_compact_load(
        model_map, model_size, layer,
        compact.data(), slot_ids.data(),
        n_total_expert, compact_count, total_slots,
        gate_offset, up_offset, down_offset,
        gate_expert_bytes, down_expert_bytes,
        1 /* strict_failure */, 0 /* allow_global_cache */);
}

#if defined(DS4_ROCM_BUILD) || defined(DS4_SYCL_BUILD)
extern "C" int ds4_gpu_tensor_read_after_selected_event(
        const ds4_gpu_tensor *tensor, uint64_t offset,
        void *data, uint64_t bytes, uint64_t event_value, const char *label) {
    (void)event_value; (void)label;
    return ds4_gpu_tensor_read(tensor, offset, data, bytes);
}

extern "C" void ds4_gpu_release_q8_f16_cache(void) {}

extern "C" int ds4_gpu_stream_expert_cache_load_layer(
        const ds4_gpu_stream_expert_table *table) {
    (void)table; return 1;
}

extern "C" int ds4_gpu_stream_expert_cache_seed_from_layer_selected(
        const ds4_gpu_stream_expert_table *table,
        const ds4_gpu_tensor *selected,
        uint32_t n_tokens, uint32_t n_seed_tokens, uint32_t n_selected) {
    (void)table; (void)selected; (void)n_tokens; (void)n_seed_tokens; (void)n_selected;
    return 1;
}

extern "C" int ds4_gpu_stream_expert_cache_release_layer_cache(void) { return 1; }
#endif

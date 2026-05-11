// Q8_0 KV-cache store and decode-attention kernels.
//
// Q8_0 = 32-element blocks of int8 quants + one f16 scale per block.
// Each block: 32 int8 quants store values in [-127, 127]; the scale
// d = absmax / 127, and reconstructed value = q * d.
//
// Layout (separate quants + scales buffers):
//   quants[ slot * num_kv_heads * head_dim + kv_head * head_dim + dim ] : int8
//   scales[ slot * num_kv_heads * (head_dim/32) + kv_head * (head_dim/32) + (dim/32) ] : f16 (as u16 bits)
//
// head_dim must be a multiple of 32.
//
// Compared to FP8 E4M3 (3-bit mantissa, ~12.5% ULP everywhere) this gives
// per-block effective ~7 bits of mantissa precision (int8 over a tight
// per-block range), which is enough to survive softmax @ V cancellation
// on Gemma-4 even with v_norm-induced channel concentration. Llama.cpp
// uses this format successfully on Gemma-4 26B per user report.

#ifndef QK8_0
#define QK8_0 32
#endif

// Helpers: pack/unpack f16 scale via raw 16-bit bit-pattern.
static __device__ __forceinline__ float q8_scale_to_float(unsigned short bits) {
    return __half2float(__ushort_as_half(bits));
}

static __device__ __forceinline__ unsigned short q8_scale_pack(float v) {
    return __half_as_ushort(__float2half_rn(v));
}

// ============================================================
//  Q8_0 KV STORE KERNELS
// ============================================================

// Decode-time single-position store (CUDA Graph friendly).
// Each thread handles one 32-element block: compute absmax → scale → quants.
// Grid: (num_blocks_total, 1, 1) where num_blocks_total = width / 32.
// Block: (1, 1, 1) — one thread per block of 32 elements (simple, prefill is the perf-critical path).
extern "C" __global__ void aegis_kv_store_q8_0_ptr(
    signed char*         key_quants,
    signed char*         value_quants,
    unsigned short*      key_scales,
    unsigned short*      value_scales,
    const float*         key,
    const float*         value,
    const unsigned int*  p_position,
    const unsigned int   width,            // num_kv_heads * head_dim
    const unsigned int   cache_capacity
) {
    const unsigned int position = *p_position;
    const unsigned int slot = (cache_capacity > 0u) ? (position % cache_capacity) : position;
    const unsigned int block_idx_in_width = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int num_blocks_per_slot = width / QK8_0;
    if (block_idx_in_width >= num_blocks_per_slot) { return; }

    const size_t elem_base   = (size_t)block_idx_in_width * QK8_0;
    const size_t cache_elem_base  = (size_t)slot * width + elem_base;
    const size_t cache_scale_base = (size_t)slot * num_blocks_per_slot + block_idx_in_width;

    // K
    {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; ++j) {
            float v = key[elem_base + j];
            amax = fmaxf(amax, fabsf(v));
        }
        float d = amax / 127.0f;
        float id = (d > 0.0f) ? 1.0f / d : 0.0f;
        key_scales[cache_scale_base] = q8_scale_pack(d);
        for (int j = 0; j < QK8_0; ++j) {
            float xq = key[elem_base + j] * id;
            int q = (int)roundf(xq);
            if (q > 127) q = 127; if (q < -127) q = -127;
            key_quants[cache_elem_base + j] = (signed char)q;
        }
    }
    // V
    {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; ++j) {
            float v = value[elem_base + j];
            amax = fmaxf(amax, fabsf(v));
        }
        float d = amax / 127.0f;
        float id = (d > 0.0f) ? 1.0f / d : 0.0f;
        value_scales[cache_scale_base] = q8_scale_pack(d);
        for (int j = 0; j < QK8_0; ++j) {
            float xq = value[elem_base + j] * id;
            int q = (int)roundf(xq);
            if (q > 127) q = 127; if (q < -127) q = -127;
            value_quants[cache_elem_base + j] = (signed char)q;
        }
    }
}

// Batched-slot store (prefill mirror, like fp8_slots_batched).
// Reads K/V scratch (already RoPE'd K) at [batch_idx, width] and stores into
// slot positions given by slot_mapping[batch_idx].
// Grid: (num_blocks_per_slot, batch, 1).  Block: (1, 1, 1).
extern "C" __global__ void aegis_kv_store_q8_0_slots_batched(
    signed char*         key_quants,
    signed char*         value_quants,
    unsigned short*      key_scales,
    unsigned short*      value_scales,
    const float*         key,
    const float*         value,
    const unsigned int*  slot_mapping,
    const unsigned int   batch,
    const unsigned int   width,
    const unsigned int   context_size
) {
    const unsigned int block_idx_in_width = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int batch_idx          = blockIdx.y;
    const unsigned int num_blocks_per_slot = width / QK8_0;
    if (batch_idx >= batch || block_idx_in_width >= num_blocks_per_slot) { return; }
    const unsigned int slot = slot_mapping[batch_idx];
    if (slot >= context_size) { return; }

    const size_t src_elem_base    = (size_t)batch_idx * width + (size_t)block_idx_in_width * QK8_0;
    const size_t dst_elem_base    = (size_t)slot * width + (size_t)block_idx_in_width * QK8_0;
    const size_t dst_scale_base   = (size_t)slot * num_blocks_per_slot + block_idx_in_width;

    // K
    {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; ++j) {
            float v = key[src_elem_base + j];
            amax = fmaxf(amax, fabsf(v));
        }
        float d = amax / 127.0f;
        float id = (d > 0.0f) ? 1.0f / d : 0.0f;
        key_scales[dst_scale_base] = q8_scale_pack(d);
        for (int j = 0; j < QK8_0; ++j) {
            float xq = key[src_elem_base + j] * id;
            int q = (int)roundf(xq);
            if (q > 127) q = 127; if (q < -127) q = -127;
            key_quants[dst_elem_base + j] = (signed char)q;
        }
    }
    // V
    {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; ++j) {
            float v = value[src_elem_base + j];
            amax = fmaxf(amax, fabsf(v));
        }
        float d = amax / 127.0f;
        float id = (d > 0.0f) ? 1.0f / d : 0.0f;
        value_scales[dst_scale_base] = q8_scale_pack(d);
        for (int j = 0; j < QK8_0; ++j) {
            float xq = value[src_elem_base + j] * id;
            int q = (int)roundf(xq);
            if (q > 127) q = 127; if (q < -127) q = -127;
            value_quants[dst_elem_base + j] = (signed char)q;
        }
    }
}

// Helper: f16 → float for V dequant (K8V16 hybrid path).
static __device__ __forceinline__ float k8v16_f16_to_float(unsigned short bits) {
    return __half2float(__ushort_as_half(bits));
}

// ============================================================
//  K8V16 hybrid: Q8_0 K + F16 V
// ============================================================
// Stores K as Q8_0 and V as plain F16. Preserves the softmax @ V cancellation
// (V stays exact) while saving ~25% of KV cache memory.
//
// Single-position decode-time K store (K-only Q8_0).
extern "C" __global__ void aegis_kv_store_q8_0_k_only_ptr(
    signed char*         key_quants,
    unsigned short*      key_scales,
    const float*         key,
    const unsigned int*  p_position,
    const unsigned int   width,
    const unsigned int   cache_capacity
) {
    const unsigned int position = *p_position;
    const unsigned int slot = (cache_capacity > 0u) ? (position % cache_capacity) : position;
    const unsigned int block_idx_in_width = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int num_blocks_per_slot = width / QK8_0;
    if (block_idx_in_width >= num_blocks_per_slot) { return; }

    const size_t elem_base   = (size_t)block_idx_in_width * QK8_0;
    const size_t cache_elem_base  = (size_t)slot * width + elem_base;
    const size_t cache_scale_base = (size_t)slot * num_blocks_per_slot + block_idx_in_width;

    float amax = 0.0f;
    for (int j = 0; j < QK8_0; ++j) {
        float v = key[elem_base + j];
        amax = fmaxf(amax, fabsf(v));
    }
    float d = amax / 127.0f;
    float id = (d > 0.0f) ? 1.0f / d : 0.0f;
    key_scales[cache_scale_base] = q8_scale_pack(d);
    for (int j = 0; j < QK8_0; ++j) {
        float xq = key[elem_base + j] * id;
        int q = (int)roundf(xq);
        if (q > 127) q = 127; if (q < -127) q = -127;
        key_quants[cache_elem_base + j] = (signed char)q;
    }
}

// Batched-slot K-only Q8_0 store (prefill mirror).
extern "C" __global__ void aegis_kv_store_q8_0_k_only_slots_batched(
    signed char*         key_quants,
    unsigned short*      key_scales,
    const float*         key,
    const unsigned int*  slot_mapping,
    const unsigned int   batch,
    const unsigned int   width,
    const unsigned int   context_size
) {
    const unsigned int block_idx_in_width = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int batch_idx          = blockIdx.y;
    const unsigned int num_blocks_per_slot = width / QK8_0;
    if (batch_idx >= batch || block_idx_in_width >= num_blocks_per_slot) { return; }
    const unsigned int slot = slot_mapping[batch_idx];
    if (slot >= context_size) { return; }

    const size_t src_elem_base    = (size_t)batch_idx * width + (size_t)block_idx_in_width * QK8_0;
    const size_t dst_elem_base    = (size_t)slot * width + (size_t)block_idx_in_width * QK8_0;
    const size_t dst_scale_base   = (size_t)slot * num_blocks_per_slot + block_idx_in_width;

    float amax = 0.0f;
    for (int j = 0; j < QK8_0; ++j) {
        float v = key[src_elem_base + j];
        amax = fmaxf(amax, fabsf(v));
    }
    float d = amax / 127.0f;
    float id = (d > 0.0f) ? 1.0f / d : 0.0f;
    key_scales[dst_scale_base] = q8_scale_pack(d);
    for (int j = 0; j < QK8_0; ++j) {
        float xq = key[src_elem_base + j] * id;
        int q = (int)roundf(xq);
        if (q > 127) q = 127; if (q < -127) q = -127;
        key_quants[dst_elem_base + j] = (signed char)q;
    }
}

// V-only F16 store (for K8V16 path). Mirrors aegis_kv_store_ptr but writes V only.
extern "C" __global__ void aegis_kv_store_f16_v_only_ptr(
    unsigned short* value_cache,
    const float* value,
    const unsigned int* p_position,
    const unsigned int width,
    const unsigned int cache_capacity
) {
    const unsigned int position = *p_position;
    const unsigned int slot = (cache_capacity > 0u) ? (position % cache_capacity) : position;
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width) {
        const size_t offset = (size_t)slot * width + idx;
        half h = __float2half_rn(value[idx]);
        value_cache[offset] = __half_as_ushort(h);
    }
}

// V-only F16 batched-slot store (for K8V16 prefill mirror).
extern "C" __global__ void aegis_kv_store_f16_v_only_slots_batched(
    unsigned short* value_cache,
    const float* value,
    const unsigned int* slot_mapping,
    const unsigned int batch,
    const unsigned int width,
    const unsigned int context_size
) {
    const unsigned int idx       = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int batch_idx = blockIdx.y;
    if (batch_idx < batch && idx < width) {
        const unsigned int slot = slot_mapping[batch_idx];
        if (slot < context_size) {
            const size_t src = (size_t)batch_idx * width + idx;
            const size_t dst = (size_t)slot * width + idx;
            half h = __float2half_rn(value[src]);
            value_cache[dst] = __half_as_ushort(h);
        }
    }
}

// K8V16 hybrid decode attention: K from Q8_0, V from F16 (u16 bits).
extern "C" __global__ void aegis_attention_decode_ptr_split_k8_v16(
    const signed char*    __restrict__ key_quants,
    const unsigned short* __restrict__ value_cache,   // f16 V
    const unsigned short* __restrict__ key_scales,
    const float*          __restrict__ query,
    const unsigned int*   __restrict__ p_seq_len,
    const unsigned int num_attention_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int split_k,
    const unsigned int max_chunk_len,
    const unsigned int window_size,
    const unsigned int cache_capacity,
    float* __restrict__ partial_acc,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l
) {
    const unsigned int seq_len   = *p_seq_len;
    const unsigned int head      = blockIdx.x;
    const unsigned int chunk_idx = blockIdx.y;
    const unsigned int tid       = threadIdx.x;
    const unsigned int warp_id   = tid >> 5u;
    const unsigned int lane      = tid & 31u;
    if (head >= num_attention_heads) { return; }

    const unsigned int window_start = (window_size > 0u && seq_len > window_size)
                                      ? seq_len - window_size : 0u;
    const unsigned int chunk_size  = (seq_len + split_k - 1u) / split_k;
    const unsigned int chunk_start = chunk_idx * chunk_size;
    const unsigned int out_idx     = head * split_k + chunk_idx;
    const unsigned int out_base    = out_idx * head_dim;

    if (chunk_start >= seq_len || (window_size > 0u && chunk_start + chunk_size <= window_start)) {
        if (tid == 0u) { partial_m[out_idx] = -3.402823466e38f; partial_l[out_idx] = 0.0f; }
        for (unsigned int d = tid; d < head_dim; d += blockDim.x) partial_acc[out_base + d] = 0.0f;
        return;
    }

    const unsigned int chunk_end = (chunk_start + chunk_size < seq_len)
                                   ? chunk_start + chunk_size : seq_len;
    const unsigned int chunk_len = chunk_end - chunk_start;

    extern __shared__ float shared[];
    float* scores       = shared;
    float* warp_partial = shared + max_chunk_len;
    float* vsum         = shared + max_chunk_len + 4u;

    const unsigned int group        = num_attention_heads / num_kv_heads;
    const unsigned int kv_head      = head / group;
    const float*       q            = query + (size_t)head * head_dim;
    const float        scale        = rsqrtf((float)head_dim);
    const unsigned int blocks_per_head = head_dim / QK8_0;
    const unsigned int width_blocks = num_kv_heads * blocks_per_head;

    /* Phase 1: Q·K, 1 warp per position */
    float warp_local_max = -3.402823466e38f;
    for (unsigned int pos = warp_id; pos < chunk_len; pos += 4u) {
        const unsigned int abs_pos = chunk_start + pos;
        float score;
        if (abs_pos < window_start) {
            score = -3.402823466e38f;
        } else {
            const unsigned int slot = (cache_capacity > 0u) ? (abs_pos % cache_capacity) : abs_pos;
            const signed char*    kq = key_quants + ((size_t)slot * num_kv_heads + kv_head) * head_dim;
            const unsigned short* ks = key_scales + ((size_t)slot * width_blocks + (size_t)kv_head * blocks_per_head);
            float partial = 0.0f;
            for (unsigned int d = lane * 4u; d < head_dim; d += 128u) {
                const unsigned int blk = d >> 5u;
                const float dscale = q8_scale_to_float(ks[blk]);
                partial += q[d+0u] * ((float)kq[d+0u]) * dscale;
                partial += q[d+1u] * ((float)kq[d+1u]) * dscale;
                partial += q[d+2u] * ((float)kq[d+2u]) * dscale;
                partial += q[d+3u] * ((float)kq[d+3u]) * dscale;
            }
            partial += __shfl_xor_sync(0xFFFFFFFFu, partial, 16u);
            partial += __shfl_xor_sync(0xFFFFFFFFu, partial, 8u);
            partial += __shfl_xor_sync(0xFFFFFFFFu, partial, 4u);
            partial += __shfl_xor_sync(0xFFFFFFFFu, partial, 2u);
            partial += __shfl_xor_sync(0xFFFFFFFFu, partial, 1u);
            score = partial * scale;
        }
        if (lane == 0u) scores[pos] = score;
        warp_local_max = fmaxf(warp_local_max, score);
    }

    if (lane == 0u) warp_partial[warp_id] = warp_local_max;
    __syncthreads();
    if (warp_id == 0u && lane < 4u) {
        float m = warp_partial[lane];
        m = fmaxf(m, __shfl_xor_sync(0xFu, m, 2u));
        m = fmaxf(m, __shfl_xor_sync(0xFu, m, 1u));
        if (lane == 0u) warp_partial[0] = m;
    }
    __syncthreads();
    const float chunk_max = warp_partial[0];

    /* Phase 2 */
    float local_sum = 0.0f;
    for (unsigned int pos = tid; pos < chunk_len; pos += blockDim.x) {
        float w = expf(scores[pos] - chunk_max);
        scores[pos] = w;
        local_sum  += w;
    }
    local_sum += __shfl_xor_sync(0xFFFFFFFFu, local_sum, 16u);
    local_sum += __shfl_xor_sync(0xFFFFFFFFu, local_sum, 8u);
    local_sum += __shfl_xor_sync(0xFFFFFFFFu, local_sum, 4u);
    local_sum += __shfl_xor_sync(0xFFFFFFFFu, local_sum, 2u);
    local_sum += __shfl_xor_sync(0xFFFFFFFFu, local_sum, 1u);
    if (lane == 0u) warp_partial[warp_id] = local_sum;
    __syncthreads();
    if (tid == 0u) {
        float s = warp_partial[0] + warp_partial[1] + warp_partial[2] + warp_partial[3];
        partial_m[out_idx] = chunk_max;
        partial_l[out_idx] = s;
    }

    /* Phase 3: weighted V sum, V from f16 cache (full precision). */
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned int pos = warp_id; pos < chunk_len; pos += 4u) {
        const unsigned int abs_pos_v = chunk_start + pos;
        const unsigned int slot_v = (cache_capacity > 0u) ? (abs_pos_v % cache_capacity) : abs_pos_v;
        const unsigned short* v = value_cache + ((size_t)slot_v * num_kv_heads + kv_head) * head_dim;
        float w = scores[pos];
        for (unsigned int d = lane * 4u; d < head_dim; d += 128u) {
            acc[0] += w * k8v16_f16_to_float(v[d+0u]);
            acc[1] += w * k8v16_f16_to_float(v[d+1u]);
            acc[2] += w * k8v16_f16_to_float(v[d+2u]);
            acc[3] += w * k8v16_f16_to_float(v[d+3u]);
        }
    }
    for (unsigned int i = 0u; i < 4u; ++i)
        vsum[warp_id * head_dim + lane * 4u + i] = acc[i];
    __syncthreads();
    for (unsigned int d = tid; d < head_dim; d += blockDim.x) {
        partial_acc[out_base + d] = vsum[0u * head_dim + d]
                                  + vsum[1u * head_dim + d]
                                  + vsum[2u * head_dim + d]
                                  + vsum[3u * head_dim + d];
    }
}

// ============================================================
//  Q8_0 DECODE ATTENTION KERNEL (split-K, CUDA Graph friendly)
// ============================================================
//
// Mirrors aegis_attention_decode_ptr_split_fp8. Reads quants (int8) and per-32
// scales (f16) for each (slot, kv_head, dim/32 block).
//
// Per-position dequant: for each 32-block, scale d = scales[slot, kv_head, block].
// Reconstructed value = q[dim] * d. Compute QK dot and weighted V sum in f32.
//
// Grid: (num_attention_heads, split_k, 1).  Block: (128, 1, 1).
extern "C" __global__ void aegis_attention_decode_ptr_split_q8_0(
    const signed char*    __restrict__ key_quants,
    const signed char*    __restrict__ value_quants,
    const unsigned short* __restrict__ key_scales,
    const unsigned short* __restrict__ value_scales,
    const float*          __restrict__ query,
    const unsigned int*   __restrict__ p_seq_len,
    const unsigned int num_attention_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int split_k,
    const unsigned int max_chunk_len,
    const unsigned int window_size,
    const unsigned int cache_capacity,
    float* __restrict__ partial_acc,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l
) {
    const unsigned int seq_len   = *p_seq_len;
    const unsigned int head      = blockIdx.x;
    const unsigned int chunk_idx = blockIdx.y;
    const unsigned int tid       = threadIdx.x;
    const unsigned int warp_id   = tid >> 5u;
    const unsigned int lane      = tid & 31u;
    if (head >= num_attention_heads) { return; }

    const unsigned int window_start = (window_size > 0u && seq_len > window_size)
                                      ? seq_len - window_size : 0u;
    const unsigned int chunk_size  = (seq_len + split_k - 1u) / split_k;
    const unsigned int chunk_start = chunk_idx * chunk_size;
    const unsigned int out_idx     = head * split_k + chunk_idx;
    const unsigned int out_base    = out_idx * head_dim;

    if (chunk_start >= seq_len || (window_size > 0u && chunk_start + chunk_size <= window_start)) {
        if (tid == 0u) { partial_m[out_idx] = -3.402823466e38f; partial_l[out_idx] = 0.0f; }
        for (unsigned int d = tid; d < head_dim; d += blockDim.x) partial_acc[out_base + d] = 0.0f;
        return;
    }

    const unsigned int chunk_end = (chunk_start + chunk_size < seq_len)
                                   ? chunk_start + chunk_size : seq_len;
    const unsigned int chunk_len = chunk_end - chunk_start;

    extern __shared__ float shared[];
    float* scores       = shared;
    float* warp_partial = shared + max_chunk_len;
    float* vsum         = shared + max_chunk_len + 4u;

    const unsigned int group        = num_attention_heads / num_kv_heads;
    const unsigned int kv_head      = head / group;
    const float*       q            = query + (size_t)head * head_dim;
    const float        scale        = rsqrtf((float)head_dim);
    const unsigned int blocks_per_head = head_dim / QK8_0;
    const unsigned int width_blocks = num_kv_heads * blocks_per_head;
    const unsigned int width        = num_kv_heads * head_dim;

    /* Phase 1: Q·K, 1 warp per position, coalesced loads (4 dims per lane). */
    float warp_local_max = -3.402823466e38f;
    for (unsigned int pos = warp_id; pos < chunk_len; pos += 4u) {
        const unsigned int abs_pos = chunk_start + pos;
        float score;
        if (abs_pos < window_start) {
            score = -3.402823466e38f;
        } else {
            const unsigned int slot = (cache_capacity > 0u) ? (abs_pos % cache_capacity) : abs_pos;
            const signed char*    kq = key_quants + ((size_t)slot * num_kv_heads + kv_head) * head_dim;
            const unsigned short* ks = key_scales + ((size_t)slot * width_blocks + (size_t)kv_head * blocks_per_head);
            float partial = 0.0f;
            for (unsigned int d = lane * 4u; d < head_dim; d += 128u) {
                // 4 contiguous dims; they're guaranteed to share at most 1 block
                // boundary only if (d % 32) >= 29. Since lane * 4 % 32 ∈ {0,4,8,...,28},
                // {d, d+1, d+2, d+3} all lie in the same 32-block.
                const unsigned int blk = d >> 5u;
                const float dscale = q8_scale_to_float(ks[blk]);
                partial += q[d+0u] * ((float)kq[d+0u]) * dscale;
                partial += q[d+1u] * ((float)kq[d+1u]) * dscale;
                partial += q[d+2u] * ((float)kq[d+2u]) * dscale;
                partial += q[d+3u] * ((float)kq[d+3u]) * dscale;
            }
            partial += __shfl_xor_sync(0xFFFFFFFFu, partial, 16u);
            partial += __shfl_xor_sync(0xFFFFFFFFu, partial, 8u);
            partial += __shfl_xor_sync(0xFFFFFFFFu, partial, 4u);
            partial += __shfl_xor_sync(0xFFFFFFFFu, partial, 2u);
            partial += __shfl_xor_sync(0xFFFFFFFFu, partial, 1u);
            score = partial * scale;
        }
        if (lane == 0u) scores[pos] = score;
        warp_local_max = fmaxf(warp_local_max, score);
    }

    if (lane == 0u) warp_partial[warp_id] = warp_local_max;
    __syncthreads();
    if (warp_id == 0u && lane < 4u) {
        float m = warp_partial[lane];
        m = fmaxf(m, __shfl_xor_sync(0xFu, m, 2u));
        m = fmaxf(m, __shfl_xor_sync(0xFu, m, 1u));
        if (lane == 0u) warp_partial[0] = m;
    }
    __syncthreads();
    const float chunk_max = warp_partial[0];

    /* Phase 2: softmax weights + sum */
    float local_sum = 0.0f;
    for (unsigned int pos = tid; pos < chunk_len; pos += blockDim.x) {
        float w = expf(scores[pos] - chunk_max);
        scores[pos] = w;
        local_sum  += w;
    }
    local_sum += __shfl_xor_sync(0xFFFFFFFFu, local_sum, 16u);
    local_sum += __shfl_xor_sync(0xFFFFFFFFu, local_sum, 8u);
    local_sum += __shfl_xor_sync(0xFFFFFFFFu, local_sum, 4u);
    local_sum += __shfl_xor_sync(0xFFFFFFFFu, local_sum, 2u);
    local_sum += __shfl_xor_sync(0xFFFFFFFFu, local_sum, 1u);
    if (lane == 0u) warp_partial[warp_id] = local_sum;
    __syncthreads();
    if (tid == 0u) {
        float s = warp_partial[0] + warp_partial[1] + warp_partial[2] + warp_partial[3];
        partial_m[out_idx] = chunk_max;
        partial_l[out_idx] = s;
    }

    /* Phase 3: weighted V sum, 1 warp per position */
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned int pos = warp_id; pos < chunk_len; pos += 4u) {
        const unsigned int abs_pos_v = chunk_start + pos;
        const unsigned int slot_v = (cache_capacity > 0u) ? (abs_pos_v % cache_capacity) : abs_pos_v;
        const signed char*    vq = value_quants + ((size_t)slot_v * num_kv_heads + kv_head) * head_dim;
        const unsigned short* vs = value_scales + ((size_t)slot_v * width_blocks + (size_t)kv_head * blocks_per_head);
        float w = scores[pos];
        for (unsigned int d = lane * 4u; d < head_dim; d += 128u) {
            const unsigned int blk = d >> 5u;
            const float dscale = q8_scale_to_float(vs[blk]);
            acc[0] += w * ((float)vq[d+0u]) * dscale;
            acc[1] += w * ((float)vq[d+1u]) * dscale;
            acc[2] += w * ((float)vq[d+2u]) * dscale;
            acc[3] += w * ((float)vq[d+3u]) * dscale;
        }
    }
    for (unsigned int i = 0u; i < 4u; ++i)
        vsum[warp_id * head_dim + lane * 4u + i] = acc[i];
    __syncthreads();
    for (unsigned int d = tid; d < head_dim; d += blockDim.x) {
        partial_acc[out_base + d] = vsum[0u * head_dim + d]
                                  + vsum[1u * head_dim + d]
                                  + vsum[2u * head_dim + d]
                                  + vsum[3u * head_dim + d];
    }
}

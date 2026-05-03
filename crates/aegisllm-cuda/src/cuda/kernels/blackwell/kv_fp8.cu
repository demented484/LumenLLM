// FP8 E4M3 KV-cache store and decode-attention kernels.
//
// All kernels mirror their F16 counterparts in norm_rope_kv.cu and
// attention_decode.cu, replacing `unsigned short*` cache buffers with
// `unsigned char*` and the F16 ↔ f32 conversion helpers with
// `float_to_fp8_e4m3_bits` / `fp8_e4m3_bits_to_float` from linear_utils.cuh.
//
// Conversion accuracy: round-to-nearest (no RTE tie-breaking) via
// fp32_to_ue4m3_halfbits.  Tolerance target: 5e-3 (gate_long_context_32k).
//
// The split-K combine kernel (aegis_attention_decode_ptr_combine) is shared
// between F16 and FP8 paths — it never touches the KV cache.

// ============================================================
//  FP8 KV STORE KERNELS
// ============================================================

extern "C" __global__ void aegis_kv_store_fp8_ptr(
    unsigned char*       key_cache,
    unsigned char*       value_cache,
    const float*         key,
    const float*         value,
    const unsigned int*  p_position,
    const unsigned int   width
) {
    const unsigned int position = *p_position;
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width) {
        const size_t offset = (size_t)position * width + idx;
        key_cache[offset]   = float_to_fp8_e4m3_bits(key[idx]);
        value_cache[offset] = float_to_fp8_e4m3_bits(value[idx]);
    }
}

extern "C" __global__ void aegis_kv_store_fp8(
    unsigned char*      key_cache,
    unsigned char*      value_cache,
    const float*        key,
    const float*        value,
    const unsigned int  position,
    const unsigned int  width
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width) {
        const size_t offset = (size_t)position * width + idx;
        key_cache[offset]   = float_to_fp8_e4m3_bits(key[idx]);
        value_cache[offset] = float_to_fp8_e4m3_bits(value[idx]);
    }
}

extern "C" __global__ void aegis_kv_store_fp8_batched(
    unsigned char*      key_cache,
    unsigned char*      value_cache,
    const float*        key,
    const float*        value,
    const unsigned int  start_position,
    const unsigned int  batch,
    const unsigned int  width
) {
    const unsigned int idx       = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int batch_idx = blockIdx.y;
    if (batch_idx < batch && idx < width) {
        const size_t src = (size_t)batch_idx * width + idx;
        const size_t dst = (size_t)(start_position + batch_idx) * width + idx;
        key_cache[dst]   = float_to_fp8_e4m3_bits(key[src]);
        value_cache[dst] = float_to_fp8_e4m3_bits(value[src]);
    }
}

extern "C" __global__ void aegis_kv_store_fp8_slots_batched(
    unsigned char*       key_cache,
    unsigned char*       value_cache,
    const float*         key,
    const float*         value,
    const unsigned int*  slot_mapping,
    const unsigned int   batch,
    const unsigned int   width,
    const unsigned int   context_size
) {
    const unsigned int idx       = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int batch_idx = blockIdx.y;
    if (batch_idx < batch && idx < width) {
        const unsigned int slot = slot_mapping[batch_idx];
        if (slot < context_size) {
            const size_t src = (size_t)batch_idx * width + idx;
            const size_t dst = (size_t)slot * width + idx;
            key_cache[dst]   = float_to_fp8_e4m3_bits(key[src]);
            value_cache[dst] = float_to_fp8_e4m3_bits(value[src]);
        }
    }
}

// RoPE + FP8 KV store — applies RoPE to key in-place, then stores both as FP8.
// Mirrors aegis_rope_kv_store_slots_batched with unsigned char* cache.
extern "C" __global__ void aegis_rope_kv_store_fp8_slots_batched(
    unsigned char*       key_cache,
    unsigned char*       value_cache,
    float*               key,
    const float*         value,
    const unsigned int*  positions,
    const unsigned int*  slot_mapping,
    const unsigned int   batch,
    const unsigned int   num_heads,
    const unsigned int   head_dim,
    const unsigned int   context_size,
    const float          theta,
    const float          factor,
    const float          low_freq_factor,
    const float          high_freq_factor,
    const unsigned int   original_max_position_embeddings,
    const unsigned int   partial_dim
) {
    const unsigned int idx       = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int batch_idx = blockIdx.y;
    const unsigned int width     = num_heads * head_dim;
    if (batch_idx >= batch || idx >= width) { return; }
    const unsigned int slot = slot_mapping[batch_idx];
    if (slot >= context_size) { return; }

    const size_t src_base = (size_t)batch_idx * width;
    const size_t dst_base = (size_t)slot * width;
    const unsigned int dim      = idx % head_dim;
    const unsigned int half_dim = head_dim / 2u;
    const unsigned int partial_half = (partial_dim > 0u) ? partial_dim / 2u : half_dim;

    value_cache[dst_base + idx] = float_to_fp8_e4m3_bits(value[src_base + idx]);
    if (dim < half_dim) {
        const unsigned int pair_idx = idx + half_dim;
        if (dim < partial_half) {
            const float angle = (float)positions[batch_idx] * rope_inv_freq_device(
                dim, head_dim, theta, factor, low_freq_factor, high_freq_factor,
                (float)original_max_position_embeddings);
            float sinv, cosv;
            sincosf(angle, &sinv, &cosv);
            const float x0 = key[src_base + idx];
            const float x1 = key[src_base + pair_idx];
            const float y0 = x0 * cosv - x1 * sinv;
            const float y1 = x0 * sinv + x1 * cosv;
            key[src_base + idx]      = y0;
            key[src_base + pair_idx] = y1;
            key_cache[dst_base + idx]      = float_to_fp8_e4m3_bits(y0);
            key_cache[dst_base + pair_idx] = float_to_fp8_e4m3_bits(y1);
        } else {
            key_cache[dst_base + idx]      = float_to_fp8_e4m3_bits(key[src_base + idx]);
            key_cache[dst_base + pair_idx] = float_to_fp8_e4m3_bits(key[src_base + pair_idx]);
        }
    }
}

// ============================================================
//  FP8 DECODE ATTENTION KERNELS
// ============================================================
//
// These are drop-in replacements for the F16 decode kernels.  The only change
// is the cache element type and the load conversion.  The arithmetic (QK dot,
// softmax, weighted V sum) remains in f32 throughout.

// --- Simple decode (no CUDA-Graph pointer variant) ---
extern "C" __global__ void aegis_attention_decode_fp8(
    const unsigned char* key_cache,
    const unsigned char* value_cache,
    const float*         query,
    const unsigned int   seq_len,
    const unsigned int   num_attention_heads,
    const unsigned int   num_kv_heads,
    const unsigned int   head_dim,
    const unsigned int   window_size,
    float*               output
) {
    const unsigned int head = blockIdx.x;
    const unsigned int tid  = threadIdx.x;
    if (head >= num_attention_heads) { return; }
    const unsigned int window_start = (window_size > 0u && seq_len > window_size)
                                      ? seq_len - window_size : 0u;
    extern __shared__ float shared[];
    float* scores  = shared;
    float* partial = shared + seq_len;
    const unsigned int group   = num_attention_heads / num_kv_heads;
    const unsigned int kv_head = head / group;
    const float*       q       = query + (size_t)head * head_dim;
    const float        scale   = rsqrtf((float)head_dim);

    float local_max = -3.402823466e38f;
    for (unsigned int pos = tid; pos < seq_len; pos += blockDim.x) {
        float score;
        if (pos < window_start) {
            score = -3.402823466e38f;
        } else {
            const unsigned char* k =
                key_cache + ((size_t)pos * num_kv_heads + kv_head) * head_dim;
            score = 0.0f;
            for (unsigned int dim = 0u; dim < head_dim; ++dim) {
                score += q[dim] * fp8_e4m3_bits_to_float(k[dim]);
            }
            score *= scale;
        }
        scores[pos] = score;
        local_max   = fmaxf(local_max, score);
    }
    partial[tid] = local_max;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) { partial[tid] = fmaxf(partial[tid], partial[tid + stride]); }
        __syncthreads();
    }
    const float max_score = partial[0];

    float local_sum = 0.0f;
    for (unsigned int pos = tid; pos < seq_len; pos += blockDim.x) {
        const float weight = expf(scores[pos] - max_score);
        scores[pos] = weight;
        local_sum  += weight;
    }
    partial[tid] = local_sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) { partial[tid] += partial[tid + stride]; }
        __syncthreads();
    }
    const float inv_sum = partial[0] > 0.0f ? 1.0f / partial[0] : 0.0f;

    float* out = output + (size_t)head * head_dim;
    for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
        float sum = 0.0f;
        for (unsigned int pos = 0u; pos < seq_len; ++pos) {
            const unsigned char* v =
                value_cache + ((size_t)pos * num_kv_heads + kv_head) * head_dim;
            sum += scores[pos] * inv_sum * fp8_e4m3_bits_to_float(v[dim]);
        }
        out[dim] = sum;
    }
}

// --- Pointer-based variant for CUDA Graph replay ---
extern "C" __global__ void aegis_attention_decode_ptr_fp8(
    const unsigned char* key_cache,
    const unsigned char* value_cache,
    const float*         query,
    const unsigned int*  p_seq_len,
    const unsigned int   num_attention_heads,
    const unsigned int   num_kv_heads,
    const unsigned int   head_dim,
    const unsigned int   window_size,
    float*               output
) {
    const unsigned int seq_len = *p_seq_len;
    const unsigned int head    = blockIdx.x;
    const unsigned int tid     = threadIdx.x;
    if (head >= num_attention_heads) { return; }
    const unsigned int window_start = (window_size > 0u && seq_len > window_size)
                                      ? seq_len - window_size : 0u;
    extern __shared__ float shared[];
    float* scores  = shared;
    float* partial = shared + seq_len;
    const unsigned int group   = num_attention_heads / num_kv_heads;
    const unsigned int kv_head = head / group;
    const float*       q       = query + (size_t)head * head_dim;
    const float        scale   = rsqrtf((float)head_dim);

    float local_max = -3.402823466e38f;
    for (unsigned int pos = tid; pos < seq_len; pos += blockDim.x) {
        float score;
        if (pos < window_start) {
            score = -3.402823466e38f;
        } else {
            const unsigned char* k =
                key_cache + ((size_t)pos * num_kv_heads + kv_head) * head_dim;
            score = 0.0f;
            for (unsigned int dim = 0u; dim < head_dim; ++dim) {
                score += q[dim] * fp8_e4m3_bits_to_float(k[dim]);
            }
            score *= scale;
        }
        scores[pos] = score;
        local_max   = fmaxf(local_max, score);
    }
    partial[tid] = local_max;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) { partial[tid] = fmaxf(partial[tid], partial[tid + stride]); }
        __syncthreads();
    }
    const float max_score = partial[0];

    float local_sum = 0.0f;
    for (unsigned int pos = tid; pos < seq_len; pos += blockDim.x) {
        const float weight = expf(scores[pos] - max_score);
        scores[pos] = weight;
        local_sum  += weight;
    }
    partial[tid] = local_sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) { partial[tid] += partial[tid + stride]; }
        __syncthreads();
    }
    const float inv_sum = partial[0] > 0.0f ? 1.0f / partial[0] : 0.0f;

    float* out = output + (size_t)head * head_dim;
    for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
        float sum = 0.0f;
        for (unsigned int pos = 0u; pos < seq_len; ++pos) {
            const unsigned char* v =
                value_cache + ((size_t)pos * num_kv_heads + kv_head) * head_dim;
            sum += scores[pos] * inv_sum * fp8_e4m3_bits_to_float(v[dim]);
        }
        out[dim] = sum;
    }
}

// --- FlashDecoding split-K variant for CUDA Graph replay ---
// Grid: (num_attention_heads, DECODE_SPLIT_K).  Block: (128,1).
// Shared: scores[max_chunk_len] + warp_partial[4] + vsum[4*head_dim].
extern "C" __global__ void aegis_attention_decode_ptr_split_fp8(
    const unsigned char* __restrict__ key_cache,
    const unsigned char* __restrict__ value_cache,
    const float*          __restrict__ query,
    const unsigned int*   __restrict__ p_seq_len,
    const unsigned int num_attention_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int split_k,
    const unsigned int max_chunk_len,
    const unsigned int window_size,
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

    const unsigned int group   = num_attention_heads / num_kv_heads;
    const unsigned int kv_head = head / group;
    const float*       q       = query + (size_t)head * head_dim;
    const float        scale   = rsqrtf((float)head_dim);

    /* Phase 1: Q·K, 1 warp per position, coalesced K loads (4 dims per lane). */
    float warp_local_max = -3.402823466e38f;
    for (unsigned int pos = warp_id; pos < chunk_len; pos += 4u) {
        const unsigned int abs_pos = chunk_start + pos;
        float score;
        if (abs_pos < window_start) {
            score = -3.402823466e38f;
        } else {
            const unsigned char* k =
                key_cache + ((size_t)abs_pos * num_kv_heads + kv_head) * head_dim;
            float partial = 0.0f;
            for (unsigned int d = lane * 4u; d < head_dim; d += 128u) {
                partial += q[d+0u] * fp8_e4m3_bits_to_float(k[d+0u]);
                partial += q[d+1u] * fp8_e4m3_bits_to_float(k[d+1u]);
                partial += q[d+2u] * fp8_e4m3_bits_to_float(k[d+2u]);
                partial += q[d+3u] * fp8_e4m3_bits_to_float(k[d+3u]);
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
        const unsigned char* v =
            value_cache + ((size_t)(chunk_start + pos) * num_kv_heads + kv_head) * head_dim;
        float w = scores[pos];
        for (unsigned int d = lane * 4u; d < head_dim; d += 128u) {
            acc[0] += w * fp8_e4m3_bits_to_float(v[d+0u]);
            acc[1] += w * fp8_e4m3_bits_to_float(v[d+1u]);
            acc[2] += w * fp8_e4m3_bits_to_float(v[d+2u]);
            acc[3] += w * fp8_e4m3_bits_to_float(v[d+3u]);
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

// --- Streaming decode variant for long contexts (O(1) shared memory) ---
extern "C" __global__ void aegis_attention_decode_streaming_fp8(
    const unsigned char* key_cache,
    const unsigned char* value_cache,
    const float*         query,
    const unsigned int   seq_len,
    const unsigned int   num_attention_heads,
    const unsigned int   num_kv_heads,
    const unsigned int   head_dim,
    const unsigned int   window_size,
    float*               output
) {
    const unsigned int head = blockIdx.x;
    const unsigned int tid  = threadIdx.x;
    if (head >= num_attention_heads) { return; }
    const unsigned int window_start = (window_size > 0u && seq_len > window_size)
                                      ? seq_len - window_size : 0u;
    extern __shared__ float shared[];
    float* partial  = shared;
    float* acc      = partial + blockDim.x;
    float* scalars  = acc + head_dim;
    const unsigned int group   = num_attention_heads / num_kv_heads;
    const unsigned int kv_head = head / group;
    const float*       q       = query + (size_t)head * head_dim;
    float*             out     = output + (size_t)head * head_dim;
    const float        scale   = rsqrtf((float)head_dim);

    if (tid == 0u) { scalars[0] = -3.402823466e38f; }
    __syncthreads();

    /* Pass 1: compute running max of QK scores */
    for (unsigned int pos = 0u; pos < seq_len; ++pos) {
        if (pos < window_start) {
            if (tid == 0u) scalars[0] = fmaxf(scalars[0], -3.402823466e38f);
            __syncthreads();
            continue;
        }
        const unsigned char* k =
            key_cache + ((size_t)pos * num_kv_heads + kv_head) * head_dim;
        float dot = 0.0f;
        for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
            dot += q[dim] * fp8_e4m3_bits_to_float(k[dim]);
        }
        partial[tid] = dot;
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride > 0u; stride >>= 1) {
            if (tid < stride) { partial[tid] += partial[tid + stride]; }
            __syncthreads();
        }
        if (tid == 0u) { scalars[0] = fmaxf(scalars[0], partial[0] * scale); }
        __syncthreads();
    }

    for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) { acc[dim] = 0.0f; }
    if (tid == 0u) { scalars[1] = 0.0f; }
    __syncthreads();

    /* Pass 2: weighted V accumulation */
    const float max_score = scalars[0];
    for (unsigned int pos = 0u; pos < seq_len; ++pos) {
        if (pos < window_start) {
            if (tid == 0u) { scalars[2] = 0.0f; }
            __syncthreads();
            continue;
        }
        const unsigned char* k =
            key_cache + ((size_t)pos * num_kv_heads + kv_head) * head_dim;
        float dot = 0.0f;
        for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
            dot += q[dim] * fp8_e4m3_bits_to_float(k[dim]);
        }
        partial[tid] = dot;
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride > 0u; stride >>= 1) {
            if (tid < stride) { partial[tid] += partial[tid + stride]; }
            __syncthreads();
        }
        if (tid == 0u) {
            scalars[2] = expf(partial[0] * scale - max_score);
            scalars[1] += scalars[2];
        }
        __syncthreads();

        const float weight = scalars[2];
        const unsigned char* v =
            value_cache + ((size_t)pos * num_kv_heads + kv_head) * head_dim;
        for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
            acc[dim] += weight * fp8_e4m3_bits_to_float(v[dim]);
        }
        __syncthreads();
    }

    const float denom = fmaxf(scalars[1], 1.0e-20f);
    for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
        out[dim] = acc[dim] / denom;
    }
}

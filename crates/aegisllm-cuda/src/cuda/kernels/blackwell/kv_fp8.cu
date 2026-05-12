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
    const unsigned int   width,
    const unsigned int   cache_capacity
) {
    const unsigned int position = *p_position;
    const unsigned int slot = (cache_capacity > 0u) ? (position % cache_capacity) : position;
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width) {
        const size_t offset = (size_t)slot * width + idx;
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
    const unsigned int  width,
    const unsigned int  cache_capacity
) {
    const unsigned int slot = (cache_capacity > 0u) ? (position % cache_capacity) : position;
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width) {
        const size_t offset = (size_t)slot * width + idx;
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
// FlashDecoding split-K decode attention — FP8 E4M3 KV variant.
// All math lives in `decode_split_attn_impl<unsigned char>()` in
// attention_decode_common.cuh — bit-identical structure to the f16
// path, only the cache element type and dequant function differ.
extern "C" __global__ void aegis_attention_decode_ptr_split_fp8(
    const unsigned char* __restrict__ key_cache,
    const unsigned char* __restrict__ value_cache,
    const float*         __restrict__ query,
    const unsigned int*  __restrict__ p_seq_len,
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
    decode_split_attn_impl<unsigned char>(
        key_cache, value_cache, query, p_seq_len,
        num_attention_heads, num_kv_heads, head_dim, split_k,
        max_chunk_len, window_size, cache_capacity,
        partial_acc, partial_m, partial_l);
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

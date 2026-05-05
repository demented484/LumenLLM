// FlashDecoding split-K decode attention — warp-level Q·K.
// Grid: (num_attention_heads, DECODE_SPLIT_K).  Block: (128,1) = 4 warps.
//
// KEY OPTIMISATION: uses 1 warp per K-position for the Q·K dot product.
//   Old: 1 thread per position, 128 serial u16 loads per thread (32 non-coalesced
//        cache-line misses per d-iteration × 128 iterations = 4096 misses per warp).
//   New: 1 warp per position, 32 threads × 4 u16 = 256 bytes fully coalesced
//        (2 cache lines per position, 32× fewer misses).
//
// Shared memory layout (pre-allocated for worst-case at graph capture time):
//   [0..max_chunk_len)           : scores[] (Q·K results, then softmax weights)
//   [max_chunk_len..+4)          : warp_partial[4] (cross-warp reductions)
//   [max_chunk_len+4..+4*head_dim): vsum[4 * head_dim] (per-warp V accumulators)
// Total = (max_chunk_len + 4 + 4*head_dim) * 4 bytes.
// With SPLIT_K=32, head_dim=128: (256 + 4 + 512) * 4 = 3072 bytes.
extern "C" __global__ void aegis_attention_decode_ptr_split(
    const unsigned short* __restrict__ key_cache,
    const unsigned short* __restrict__ value_cache,
    const float*          __restrict__ query,
    const unsigned int*   __restrict__ p_seq_len,
    const unsigned int num_attention_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int split_k,
    const unsigned int max_chunk_len,
    const unsigned int window_size,     /* 0 = full causal; >0 = sliding window */
    float* __restrict__ partial_acc,   /* [num_heads * split_k * head_dim] */
    float* __restrict__ partial_m,     /* [num_heads * split_k]            */
    float* __restrict__ partial_l      /* [num_heads * split_k]            */
) {
    const unsigned int seq_len   = *p_seq_len;
    const unsigned int head      = blockIdx.x;
    const unsigned int chunk_idx = blockIdx.y;
    const unsigned int tid       = threadIdx.x;
    const unsigned int warp_id   = tid >> 5u;
    const unsigned int lane      = tid & 31u;
    if (head >= num_attention_heads) return;

    /* Sliding-window: positions older than window_start are masked to -inf. */
    const unsigned int window_start = (window_size > 0u && seq_len > window_size)
                                      ? seq_len - window_size : 0u;

    const unsigned int chunk_size  = (seq_len + split_k - 1u) / split_k;
    const unsigned int chunk_start = chunk_idx * chunk_size;
    const unsigned int out_idx     = head * split_k + chunk_idx;
    const unsigned int out_base    = out_idx * head_dim;

    /* Entire chunk is outside the window — treat as all-masked (empty). */
    if (chunk_start >= seq_len || (window_size > 0u && chunk_start + chunk_size <= window_start)) {
        if (tid == 0) { partial_m[out_idx] = -3.402823466e38f; partial_l[out_idx] = 0.0f; }
        for (unsigned int d = tid; d < head_dim; d += blockDim.x) partial_acc[out_base + d] = 0.0f;
        return;
    }

    const unsigned int chunk_end = (chunk_start + chunk_size < seq_len) ? chunk_start + chunk_size : seq_len;
    const unsigned int chunk_len = chunk_end - chunk_start;

    extern __shared__ float shared[];
    float* scores      = shared;                        /* [max_chunk_len]  */
    float* warp_partial = shared + max_chunk_len;        /* [4]              */
    float* vsum        = shared + max_chunk_len + 4u;    /* [4 * head_dim]   */

    const unsigned int group   = num_attention_heads / num_kv_heads;
    const unsigned int kv_head = head / group;
    const float*       q       = query + (size_t)head * head_dim;
    const float        scale   = rsqrtf((float)head_dim);

    /* --- Phase 1: Q·K, 1 warp per position, coalesced K loads ---
     * Each warp handles positions warp_id, warp_id+4, warp_id+8, ...
     * Each lane handles head_dim/32 = 4 contiguous dims (lane*4 .. lane*4+3).
     * All 32 lanes read K[pos][lane*4..lane*4+3] → 256 bytes fully coalesced. */
    float warp_local_max = -3.402823466e38f;
    for (unsigned int pos = warp_id; pos < chunk_len; pos += 4u) {
        const unsigned int abs_pos = chunk_start + pos;
        float score;
        if (abs_pos < window_start) {
            /* Position is outside the sliding window — mask to -inf. */
            score = -3.402823466e38f;
        } else {
            const unsigned short* k = key_cache + ((size_t)abs_pos * num_kv_heads + kv_head) * head_dim;
            float partial = 0.0f;
            for (unsigned int d = lane * 4u; d < head_dim; d += 128u) {
                partial += q[d+0u] * f16_bits_to_float(k[d+0u]);
                partial += q[d+1u] * f16_bits_to_float(k[d+1u]);
                partial += q[d+2u] * f16_bits_to_float(k[d+2u]);
                partial += q[d+3u] * f16_bits_to_float(k[d+3u]);
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

    /* Cross-warp max reduction */
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

    /* --- Phase 2: softmax weights + sum, all 128 threads parallel over positions --- */
    float local_sum = 0.0f;
    for (unsigned int pos = tid; pos < chunk_len; pos += blockDim.x) {
        float w = expf(scores[pos] - chunk_max);
        scores[pos] = w;
        local_sum += w;
    }
    /* Cross-warp sum reduction (reuse warp_partial) */
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

    /* --- Phase 3: weighted V sum, 1 warp per position, coalesced V loads ---
     * Each lane accumulates 4 consecutive V dims per d-block. d-blocks step by 128
     * (32 lanes * 4 dims/lane). For head_dim=128 there is one d-block (Llama). For
     * head_dim=256/512 (Gemma 4 sliding/global) there are 2 / 4 d-blocks. The previous
     * implementation only had 4 accumulators total and silently summed contributions
     * from every d-block into the same slot — correct for head_dim=128, garbage
     * otherwise. We size the accumulator by the max supported head_dim (512 → 4
     * d-blocks). */
    constexpr unsigned int MAX_D_BLOCKS = 4u;  // supports head_dim up to 4*128 = 512
    float acc[MAX_D_BLOCKS][4] = { {0.0f, 0.0f, 0.0f, 0.0f} };
    const unsigned int d_blocks = (head_dim + 127u) / 128u;
    for (unsigned int pos = warp_id; pos < chunk_len; pos += 4u) {
        const unsigned short* v = value_cache + ((size_t)(chunk_start + pos) * num_kv_heads + kv_head) * head_dim;
        float w = scores[pos];
        for (unsigned int b = 0u; b < d_blocks; ++b) {
            const unsigned int d = b * 128u + lane * 4u;
            if (d >= head_dim) break;
            acc[b][0] += w * f16_bits_to_float(v[d+0u]);
            acc[b][1] += w * f16_bits_to_float(v[d+1u]);
            acc[b][2] += w * f16_bits_to_float(v[d+2u]);
            acc[b][3] += w * f16_bits_to_float(v[d+3u]);
        }
    }
    /* Write per-warp V accumulators to shared memory at vsum[warp_id*head_dim + d + i]. */
    for (unsigned int b = 0u; b < d_blocks; ++b) {
        const unsigned int d = b * 128u + lane * 4u;
        if (d >= head_dim) break;
        vsum[warp_id * head_dim + d + 0u] = acc[b][0];
        vsum[warp_id * head_dim + d + 1u] = acc[b][1];
        vsum[warp_id * head_dim + d + 2u] = acc[b][2];
        vsum[warp_id * head_dim + d + 3u] = acc[b][3];
    }
    __syncthreads();
    /* Reduce across 4 warps. Each thread covers head_dim/blockDim.x output dims. */
    for (unsigned int d = tid; d < head_dim; d += blockDim.x) {
        float sum = vsum[0u * head_dim + d]
                  + vsum[1u * head_dim + d]
                  + vsum[2u * head_dim + d]
                  + vsum[3u * head_dim + d];
        partial_acc[out_base + d] = sum;
    }
}

// Combine DECODE_SPLIT_K partial flash-decode results into a single output head vector.
// Grid: (num_attention_heads, 1).  Block: (128, 1).  No shared memory.
extern "C" __global__ void aegis_attention_decode_ptr_combine(
    const float* __restrict__ partial_acc,  /* [num_heads * split_k * head_dim] */
    const float* __restrict__ partial_m,    /* [num_heads * split_k]            */
    const float* __restrict__ partial_l,    /* [num_heads * split_k]            */
    const unsigned int head_dim,
    const unsigned int split_k,
    float* __restrict__ output              /* [num_heads * head_dim]            */
) {
    const unsigned int head = blockIdx.x;
    const unsigned int tid  = threadIdx.x;

    const float* my_m = partial_m + head * split_k;
    const float* my_l = partial_l + head * split_k;

    float global_max = -3.402823466e38f;
    for (unsigned int k = 0u; k < split_k; ++k) global_max = fmaxf(global_max, my_m[k]);

    float denom = 0.0f;
    for (unsigned int k = 0u; k < split_k; ++k) denom += expf(my_m[k] - global_max) * my_l[k];
    const float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;

    float* out = output + (size_t)head * head_dim;
    for (unsigned int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int k = 0u; k < split_k; ++k)
            acc += expf(my_m[k] - global_max) * partial_acc[(head * split_k + k) * head_dim + d];
        out[d] = acc * inv_denom;
    }
}

// Pointer-based variants for CUDA Graph replay: seq_len is read from device memory
// so the same captured graph works across multiple decode steps with growing seq_len.
extern "C" __global__ void aegis_attention_decode_ptr(
    const unsigned short* key_cache,
    const unsigned short* value_cache,
    const float* query,
    const unsigned int* p_seq_len,
    const unsigned int num_attention_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int window_size,     /* 0 = full causal; >0 = sliding window */
    float* output
) {
    const unsigned int seq_len = *p_seq_len;
    const unsigned int head = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (head >= num_attention_heads) { return; }
    const unsigned int window_start = (window_size > 0u && seq_len > window_size)
                                      ? seq_len - window_size : 0u;
    extern __shared__ float shared[];
    float* scores = shared;
    float* partial = shared + seq_len;
    const unsigned int group = num_attention_heads / num_kv_heads;
    const unsigned int kv_head = head / group;
    const float* q = query + size_t(head) * head_dim;
    const float scale = rsqrtf(float(head_dim));
    float local_max = -3.402823466e38f;
    for (unsigned int pos = tid; pos < seq_len; pos += blockDim.x) {
        float score;
        if (pos < window_start) {
            score = -3.402823466e38f;
        } else {
            const unsigned short* k = key_cache + (size_t(pos) * num_kv_heads + kv_head) * head_dim;
            score = 0.0f;
            for (unsigned int dim = 0u; dim < head_dim; ++dim) { score += q[dim] * f16_bits_to_float(k[dim]); }
            score *= scale;
        }
        scores[pos] = score;
        local_max = fmaxf(local_max, score);
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
        local_sum += weight;
    }
    partial[tid] = local_sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) { partial[tid] += partial[tid + stride]; }
        __syncthreads();
    }
    const float inv_sum = partial[0] > 0.0f ? 1.0f / partial[0] : 0.0f;
    float* out = output + size_t(head) * head_dim;
    for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
        float sum = 0.0f;
        for (unsigned int pos = 0u; pos < seq_len; ++pos) {
            const unsigned short* v = value_cache + (size_t(pos) * num_kv_heads + kv_head) * head_dim;
            sum += scores[pos] * inv_sum * f16_bits_to_float(v[dim]);
        }
        out[dim] = sum;
    }
}

extern "C" __global__ void aegis_attention_decode(
    const unsigned short* key_cache,
    const unsigned short* value_cache,
    const float* query,
    const unsigned int seq_len,
    const unsigned int num_attention_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int window_size,     /* 0 = full causal; >0 = sliding window */
    float* output
) {
    const unsigned int head = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (head >= num_attention_heads) {
        return;
    }
    const unsigned int window_start = (window_size > 0u && seq_len > window_size)
                                      ? seq_len - window_size : 0u;
    extern __shared__ float shared[];
    float* scores = shared;
    float* partial = shared + seq_len;
    const unsigned int group = num_attention_heads / num_kv_heads;
    const unsigned int kv_head = head / group;
    const float* q = query + size_t(head) * head_dim;
    const float scale = rsqrtf(float(head_dim));

    float local_max = -3.402823466e38f;
    for (unsigned int pos = tid; pos < seq_len; pos += blockDim.x) {
        float score;
        if (pos < window_start) {
            score = -3.402823466e38f;
        } else {
            const unsigned short* k = key_cache + (size_t(pos) * num_kv_heads + kv_head) * head_dim;
            score = 0.0f;
            for (unsigned int dim = 0u; dim < head_dim; ++dim) {
                score += q[dim] * f16_bits_to_float(k[dim]);
            }
            score *= scale;
        }
        scores[pos] = score;
        local_max = fmaxf(local_max, score);
    }
    partial[tid] = local_max;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            partial[tid] = fmaxf(partial[tid], partial[tid + stride]);
        }
        __syncthreads();
    }
    const float max_score = partial[0];

    float local_sum = 0.0f;
    for (unsigned int pos = tid; pos < seq_len; pos += blockDim.x) {
        const float weight = expf(scores[pos] - max_score);
        scores[pos] = weight;
        local_sum += weight;
    }
    partial[tid] = local_sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    const float inv_sum = partial[0] > 0.0f ? 1.0f / partial[0] : 0.0f;

    float* out = output + size_t(head) * head_dim;
    for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
        float sum = 0.0f;
        for (unsigned int pos = 0u; pos < seq_len; ++pos) {
            const unsigned short* v = value_cache + (size_t(pos) * num_kv_heads + kv_head) * head_dim;
            sum += scores[pos] * inv_sum * f16_bits_to_float(v[dim]);
        }
        out[dim] = sum;
    }
}

extern "C" __global__ void aegis_attention_decode_streaming(
    const unsigned short* key_cache,
    const unsigned short* value_cache,
    const float* query,
    const unsigned int seq_len,
    const unsigned int num_attention_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int window_size,     /* 0 = full causal; >0 = sliding window */
    float* output
) {
    const unsigned int head = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (head >= num_attention_heads) {
        return;
    }
    const unsigned int window_start = (window_size > 0u && seq_len > window_size)
                                      ? seq_len - window_size : 0u;

    extern __shared__ float shared[];
    float* partial = shared;
    float* acc = partial + blockDim.x;
    float* scalars = acc + head_dim;
    const unsigned int group = num_attention_heads / num_kv_heads;
    const unsigned int kv_head = head / group;
    const float* q = query + size_t(head) * head_dim;
    float* out = output + size_t(head) * head_dim;
    const float scale = rsqrtf(float(head_dim));

    if (tid == 0u) {
        scalars[0] = -3.402823466e38f;
    }
    __syncthreads();

    for (unsigned int pos = 0u; pos < seq_len; ++pos) {
        if (pos < window_start) {
            /* Masked position — contributes -inf to max; skip K load. */
            if (tid == 0u) scalars[0] = fmaxf(scalars[0], -3.402823466e38f);
            __syncthreads();
            continue;
        }
        const unsigned short* k =
            key_cache + (size_t(pos) * num_kv_heads + kv_head) * head_dim;
        float dot = 0.0f;
        for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
            dot += q[dim] * f16_bits_to_float(k[dim]);
        }
        partial[tid] = dot;
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride > 0u; stride >>= 1) {
            if (tid < stride) {
                partial[tid] += partial[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0u) {
            scalars[0] = fmaxf(scalars[0], partial[0] * scale);
        }
        __syncthreads();
    }

    for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
        acc[dim] = 0.0f;
    }
    if (tid == 0u) {
        scalars[1] = 0.0f;
    }
    __syncthreads();

    const float max_score = scalars[0];
    for (unsigned int pos = 0u; pos < seq_len; ++pos) {
        if (pos < window_start) {
            /* Masked: weight = exp(-inf - max) = 0; skip accumulation. */
            if (tid == 0u) { scalars[2] = 0.0f; }
            __syncthreads();
            continue;
        }
        const unsigned short* k =
            key_cache + (size_t(pos) * num_kv_heads + kv_head) * head_dim;
        float dot = 0.0f;
        for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
            dot += q[dim] * f16_bits_to_float(k[dim]);
        }
        partial[tid] = dot;
        __syncthreads();
        for (unsigned int stride = blockDim.x >> 1; stride > 0u; stride >>= 1) {
            if (tid < stride) {
                partial[tid] += partial[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0u) {
            scalars[2] = expf(partial[0] * scale - max_score);
            scalars[1] += scalars[2];
        }
        __syncthreads();

        const float weight = scalars[2];
        const unsigned short* v =
            value_cache + (size_t(pos) * num_kv_heads + kv_head) * head_dim;
        for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
            acc[dim] += weight * f16_bits_to_float(v[dim]);
        }
        __syncthreads();
    }

    const float denom = fmaxf(scalars[1], 1.0e-20f);
    for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
        out[dim] = acc[dim] / denom;
    }
}

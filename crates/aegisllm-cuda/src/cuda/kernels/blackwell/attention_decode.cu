// FlashDecoding split-K decode attention — F16 KV variant.
// All math lives in `decode_split_attn_impl<unsigned short>()` in
// attention_decode_common.cuh. See that file for shared-mem layout,
// cp.async pipelining, and the 3-phase structure. This file is just
// the extern "C" entry-point that the JIT loader picks up by name.
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
    const unsigned int window_size,
    const unsigned int cache_capacity,
    float* __restrict__ partial_acc,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l
) {
    decode_split_attn_impl<unsigned short>(
        key_cache, value_cache, query, p_seq_len,
        num_attention_heads, num_kv_heads, head_dim, split_k,
        max_chunk_len, window_size, cache_capacity,
        partial_acc, partial_m, partial_l);
}

// Stage G head-dim-partitioned single-pass variant (f16 KV). Opt-in via
// AEGIS_DECODE_HDPART=1; tiny shared (KQ[128]+scratch) for high occupancy.
extern "C" __global__ void aegis_attention_decode_ptr_split_hdpart(
    const unsigned short* __restrict__ key_cache,
    const unsigned short* __restrict__ value_cache,
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
    decode_split_attn_hdpart_impl<unsigned short>(
        key_cache, value_cache, query, p_seq_len,
        num_attention_heads, num_kv_heads, head_dim, split_k,
        max_chunk_len, window_size, cache_capacity,
        partial_acc, partial_m, partial_l);
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

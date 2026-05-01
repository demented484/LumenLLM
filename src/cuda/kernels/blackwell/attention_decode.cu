extern "C" __global__ void aegis_attention_decode(
    const unsigned short* key_cache,
    const unsigned short* value_cache,
    const float* query,
    const unsigned int seq_len,
    const unsigned int num_attention_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    float* output
) {
    const unsigned int head = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (head >= num_attention_heads) {
        return;
    }
    extern __shared__ float shared[];
    float* scores = shared;
    float* partial = shared + seq_len;
    const unsigned int group = num_attention_heads / num_kv_heads;
    const unsigned int kv_head = head / group;
    const float* q = query + size_t(head) * head_dim;
    const float scale = rsqrtf(float(head_dim));

    float local_max = -3.402823466e38f;
    for (unsigned int pos = tid; pos < seq_len; pos += blockDim.x) {
        const unsigned short* k = key_cache + (size_t(pos) * num_kv_heads + kv_head) * head_dim;
        float score = 0.0f;
        for (unsigned int dim = 0u; dim < head_dim; ++dim) {
            score += q[dim] * f16_bits_to_float(k[dim]);
        }
        score *= scale;
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
    float* output
) {
    const unsigned int head = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (head >= num_attention_heads) {
        return;
    }

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

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

extern "C" __global__ void aegis_attention_prefill_batched(
    const unsigned short* key_cache,
    const unsigned short* value_cache,
    const float* query,
    const unsigned int start_position,
    const unsigned int batch,
    const unsigned int num_attention_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    float* output
) {
    const unsigned int head = blockIdx.x;
    const unsigned int batch_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    if (batch_idx >= batch || head >= num_attention_heads) {
        return;
    }
    const unsigned int seq_len = start_position + batch_idx + 1u;
    extern __shared__ float shared[];
    float* scores = shared;
    float* partial = shared + (start_position + batch);
    const unsigned int group = num_attention_heads / num_kv_heads;
    const unsigned int kv_head = head / group;
    const float* q = query + (size_t(batch_idx) * num_attention_heads + head) * head_dim;
    float* out = output + (size_t(batch_idx) * num_attention_heads + head) * head_dim;
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
    const float denom = fmaxf(partial[0], 1.0e-20f);

    for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int pos = 0u; pos < seq_len; ++pos) {
            const float weight = scores[pos] / denom;
            const unsigned short* v = value_cache + (size_t(pos) * num_kv_heads + kv_head) * head_dim;
            acc += weight * f16_bits_to_float(v[dim]);
        }
        out[dim] = acc;
    }
}

extern "C" __global__ void aegis_attention_prefill_batched_mixed(
    const unsigned short* key_cache,
    const unsigned short* value_cache,
    const float* key_chunk,
    const float* value_chunk,
    const float* query,
    const unsigned int start_position,
    const unsigned int batch,
    const unsigned int num_attention_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    float* output
) {
    const unsigned int head = blockIdx.x;
    const unsigned int batch_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    if (batch_idx >= batch || head >= num_attention_heads) {
        return;
    }
    const unsigned int seq_len = start_position + batch_idx + 1u;
    extern __shared__ float shared[];
    float* scores = shared;
    float* partial = shared + (start_position + batch);
    const unsigned int group = num_attention_heads / num_kv_heads;
    const unsigned int kv_head = head / group;
    const float* q = query + (size_t(batch_idx) * num_attention_heads + head) * head_dim;
    float* out = output + (size_t(batch_idx) * num_attention_heads + head) * head_dim;
    const float scale = rsqrtf(float(head_dim));

    float local_max = -3.402823466e38f;
    for (unsigned int pos = tid; pos < seq_len; pos += blockDim.x) {
        float score = 0.0f;
        if (pos < start_position) {
            const unsigned short* k =
                key_cache + (size_t(pos) * num_kv_heads + kv_head) * head_dim;
            for (unsigned int dim = 0u; dim < head_dim; ++dim) {
                score += q[dim] * f16_bits_to_float(k[dim]);
            }
        } else {
            const unsigned int chunk_pos = pos - start_position;
            const float* k =
                key_chunk + (size_t(chunk_pos) * num_kv_heads + kv_head) * head_dim;
            for (unsigned int dim = 0u; dim < head_dim; ++dim) {
                score += q[dim] * k[dim];
            }
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
    const float denom = fmaxf(partial[0], 1.0e-20f);

    for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int pos = 0u; pos < seq_len; ++pos) {
            const float weight = scores[pos] / denom;
            if (pos < start_position) {
                const unsigned short* v =
                    value_cache + (size_t(pos) * num_kv_heads + kv_head) * head_dim;
                acc += weight * f16_bits_to_float(v[dim]);
            } else {
                const unsigned int chunk_pos = pos - start_position;
                const float* v =
                    value_chunk + (size_t(chunk_pos) * num_kv_heads + kv_head) * head_dim;
                acc += weight * v[dim];
            }
        }
        out[dim] = acc;
    }
}

extern "C" __global__ void aegis_attention_prefill_continuation(
    const unsigned short* key_cache,
    const unsigned short* value_cache,
    const float* key_chunk,
    const float* value_chunk,
    const float* query,
    const unsigned int start_position,
    const unsigned int batch,
    const unsigned int num_attention_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    float* output
) {
    const unsigned int head = blockIdx.x;
    const unsigned int batch_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    if (batch_idx >= batch || head >= num_attention_heads) {
        return;
    }

    extern __shared__ float shared[];
    float* partial = shared;
    float* acc = partial + blockDim.x;
    float* scalars = acc + head_dim;

    const unsigned int seq_len = start_position + batch_idx + 1u;
    const unsigned int group = num_attention_heads / num_kv_heads;
    const unsigned int kv_head = head / group;
    const float* q = query + (size_t(batch_idx) * num_attention_heads + head) * head_dim;
    float* out = output + (size_t(batch_idx) * num_attention_heads + head) * head_dim;
    const float scale = rsqrtf(float(head_dim));

    if (tid == 0u) {
        scalars[0] = -3.402823466e38f;
    }
    __syncthreads();

    for (unsigned int pos = 0u; pos < seq_len; ++pos) {
        float dot = 0.0f;
        const unsigned short* k =
            key_cache + (size_t(pos) * num_kv_heads + kv_head) * head_dim;
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
        float dot = 0.0f;
        const unsigned short* k =
            key_cache + (size_t(pos) * num_kv_heads + kv_head) * head_dim;
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

extern "C" __global__ void aegis_attention_prefill_paged_varlen(
    const unsigned short* key_cache,
    const unsigned short* value_cache,
    const float* query,
    const unsigned int* slot_mapping,
    const unsigned int* cu_q,
    const unsigned int* context_lens,
    const unsigned int* block_tables,
    const unsigned int num_sequences,
    const unsigned int total_q,
    const unsigned int num_attention_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int page_tokens,
    const unsigned int block_table_stride,
    const unsigned int physical_slots,
    float* output
) {
    const unsigned int head = blockIdx.x;
    const unsigned int global_q = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    if (global_q >= total_q || head >= num_attention_heads) {
        return;
    }
    const unsigned int current_slot = slot_mapping[global_q];
    if (current_slot == 0xffffffffu) {
        return;
    }

    unsigned int seq = 0u;
    unsigned int q_start = 0u;
    unsigned int q_end = 0u;
    for (; seq < num_sequences; ++seq) {
        q_start = cu_q[seq];
        q_end = cu_q[seq + 1u];
        if (global_q >= q_start && global_q < q_end) {
            break;
        }
    }
    if (seq >= num_sequences || q_end <= q_start) {
        return;
    }

    const unsigned int q_in_seq = global_q - q_start;
    const unsigned int q_len = q_end - q_start;
    const unsigned int context_len = context_lens[seq];
    const unsigned int hidden_future = q_len - q_in_seq - 1u;
    const unsigned int visible_len = context_len > hidden_future
        ? context_len - hidden_future
        : 0u;
    if (visible_len == 0u) {
        return;
    }

    extern __shared__ float shared[];
    float* partial = shared;
    float* acc = partial + blockDim.x;
    float* scalars = acc + head_dim;

    const unsigned int group = num_attention_heads / num_kv_heads;
    const unsigned int kv_head = head / group;
    const float* q = query + (size_t(global_q) * num_attention_heads + head) * head_dim;
    float* out = output + (size_t(global_q) * num_attention_heads + head) * head_dim;
    const float scale = rsqrtf(float(head_dim));

    for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
        acc[dim] = 0.0f;
    }
    if (tid == 0u) {
        scalars[0] = -3.402823466e38f; // online max
        scalars[1] = 0.0f;             // online denominator
    }
    __syncthreads();

    for (unsigned int pos = 0u; pos < visible_len; ++pos) {
        const unsigned int logical_page = page_tokens == 0u ? 0u : pos / page_tokens;
        const unsigned int page_offset = page_tokens == 0u ? pos : pos - logical_page * page_tokens;
        const unsigned int physical_page = block_tables[size_t(seq) * block_table_stride + logical_page];
        const unsigned int physical_slot = physical_page * page_tokens + page_offset;
        if (physical_slot >= physical_slots) {
            continue;
        }
        const unsigned short* k =
            key_cache + (size_t(physical_slot) * num_kv_heads + kv_head) * head_dim;

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
            const float score = partial[0] * scale;
            const float old_m = scalars[0];
            const float old_l = scalars[1];
            const float new_m = fmaxf(old_m, score);
            const float alpha = old_l > 0.0f ? expf(old_m - new_m) : 0.0f;
            const float beta = expf(score - new_m);
            scalars[2] = alpha;
            scalars[3] = beta;
            scalars[0] = new_m;
            scalars[1] = old_l * alpha + beta;
        }
        __syncthreads();

        const float alpha = scalars[2];
        const float beta = scalars[3];
        const unsigned short* v =
            value_cache + (size_t(physical_slot) * num_kv_heads + kv_head) * head_dim;
        for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
            acc[dim] = acc[dim] * alpha + beta * f16_bits_to_float(v[dim]);
        }
        __syncthreads();
    }

    const float denom = fmaxf(scalars[1], 1.0e-20f);
    for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
        out[dim] = acc[dim] / denom;
    }
}

extern "C" __global__ void aegis_attention_prefill_paged_varlen_halfq(
    const unsigned short* key_cache,
    const unsigned short* value_cache,
    const unsigned short* query,
    const unsigned int* slot_mapping,
    const unsigned int* cu_q,
    const unsigned int* context_lens,
    const unsigned int* block_tables,
    const unsigned int num_sequences,
    const unsigned int total_q,
    const unsigned int num_attention_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int page_tokens,
    const unsigned int block_table_stride,
    const unsigned int physical_slots,
    float* output
) {
    const unsigned int head = blockIdx.x;
    const unsigned int global_q = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    if (global_q >= total_q || head >= num_attention_heads) {
        return;
    }
    const unsigned int current_slot = slot_mapping[global_q];
    if (current_slot == 0xffffffffu) {
        return;
    }

    unsigned int seq = 0u;
    unsigned int q_start = 0u;
    unsigned int q_end = 0u;
    for (; seq < num_sequences; ++seq) {
        q_start = cu_q[seq];
        q_end = cu_q[seq + 1u];
        if (global_q >= q_start && global_q < q_end) {
            break;
        }
    }
    if (seq >= num_sequences || q_end <= q_start) {
        return;
    }

    const unsigned int q_in_seq = global_q - q_start;
    const unsigned int q_len = q_end - q_start;
    const unsigned int context_len = context_lens[seq];
    const unsigned int hidden_future = q_len - q_in_seq - 1u;
    const unsigned int visible_len = context_len > hidden_future
        ? context_len - hidden_future
        : 0u;
    if (visible_len == 0u) {
        return;
    }

    extern __shared__ float shared[];
    float* partial = shared;
    float* acc = partial + blockDim.x;
    float* scalars = acc + head_dim;
    float* q_shared = scalars + 4;

    const unsigned int group = num_attention_heads / num_kv_heads;
    const unsigned int kv_head = head / group;
    const unsigned short* q = query + (size_t(global_q) * num_attention_heads + head) * head_dim;
    float* out = output + (size_t(global_q) * num_attention_heads + head) * head_dim;
    const float scale = rsqrtf(float(head_dim));

    for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
        q_shared[dim] = f16_bits_to_float(q[dim]);
    }
    for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
        acc[dim] = 0.0f;
    }
    if (tid == 0u) {
        scalars[0] = -3.402823466e38f;
        scalars[1] = 0.0f;
    }
    __syncthreads();

    for (unsigned int pos = 0u; pos < visible_len; ++pos) {
        const unsigned int logical_page = page_tokens == 0u ? 0u : pos / page_tokens;
        const unsigned int page_offset = page_tokens == 0u ? pos : pos - logical_page * page_tokens;
        const unsigned int physical_page = block_tables[size_t(seq) * block_table_stride + logical_page];
        const unsigned int physical_slot = physical_page * page_tokens + page_offset;
        if (physical_slot >= physical_slots) {
            continue;
        }
        const unsigned short* k =
            key_cache + (size_t(physical_slot) * num_kv_heads + kv_head) * head_dim;

        float dot = 0.0f;
        for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
            dot += q_shared[dim] * f16_bits_to_float(k[dim]);
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
            const float score = partial[0] * scale;
            const float old_m = scalars[0];
            const float old_l = scalars[1];
            const float new_m = fmaxf(old_m, score);
            const float alpha = old_l > 0.0f ? expf(old_m - new_m) : 0.0f;
            const float beta = expf(score - new_m);
            scalars[2] = alpha;
            scalars[3] = beta;
            scalars[0] = new_m;
            scalars[1] = old_l * alpha + beta;
        }
        __syncthreads();

        const float alpha = scalars[2];
        const float beta = scalars[3];
        const unsigned short* v =
            value_cache + (size_t(physical_slot) * num_kv_heads + kv_head) * head_dim;
        for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
            acc[dim] = acc[dim] * alpha + beta * f16_bits_to_float(v[dim]);
        }
        __syncthreads();
    }

    const float denom = fmaxf(scalars[1], 1.0e-20f);
    for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
        out[dim] = acc[dim] / denom;
    }
}

extern "C" __global__ void aegis_attention_prefill_paged_varlen_halfq_block4(
    const unsigned short* key_cache,
    const unsigned short* value_cache,
    const unsigned short* query,
    const unsigned int* slot_mapping,
    const unsigned int* cu_q,
    const unsigned int* context_lens,
    const unsigned int* block_tables,
    const unsigned int num_sequences,
    const unsigned int total_q,
    const unsigned int num_attention_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int page_tokens,
    const unsigned int block_table_stride,
    const unsigned int physical_slots,
    float* output
) {
    const unsigned int q_block = 4u;
    const unsigned int head = blockIdx.x;
    const unsigned int global_q_base = blockIdx.y * q_block;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5u;
    const unsigned int nwarps = blockDim.x >> 5u;
    if (head >= num_attention_heads || global_q_base >= total_q || num_sequences != 1u) {
        return;
    }

    const unsigned int q_start = cu_q[0];
    const unsigned int q_end = cu_q[1];
    const unsigned int q_len = q_end - q_start;
    const unsigned int context_len = context_lens[0];
    const unsigned int group = num_attention_heads / num_kv_heads;
    const unsigned int kv_head = head / group;
    const float scale = rsqrtf(float(head_dim));

    unsigned int visible_len[4];
    bool valid[4];
    unsigned int max_visible = 0u;
    #pragma unroll
    for (unsigned int row = 0u; row < q_block; ++row) {
        const unsigned int global_q = global_q_base + row;
        valid[row] = global_q < total_q
            && global_q >= q_start
            && global_q < q_end
            && slot_mapping[global_q] != 0xffffffffu;
        if (valid[row]) {
            const unsigned int q_in_seq = global_q - q_start;
            const unsigned int hidden_future = q_len - q_in_seq - 1u;
            visible_len[row] = context_len > hidden_future
                ? context_len - hidden_future
                : 0u;
            valid[row] = visible_len[row] > 0u;
            max_visible = max(max_visible, visible_len[row]);
        } else {
            visible_len[row] = 0u;
        }
    }
    if (max_visible == 0u) {
        return;
    }

    extern __shared__ float shared[];
    float* partial = shared;
    float* q_shared = partial + q_block * nwarps;
    float* k_shared = q_shared + q_block * head_dim;
    float* v_shared = k_shared + head_dim;
    float* acc = v_shared + head_dim;
    float* scalars = acc + q_block * head_dim;

    for (unsigned int row = 0u; row < q_block; ++row) {
        if (!valid[row]) {
            continue;
        }
        const unsigned int global_q = global_q_base + row;
        const unsigned short* q =
            query + (size_t(global_q) * num_attention_heads + head) * head_dim;
        for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
            q_shared[row * head_dim + dim] = f16_bits_to_float(q[dim]);
            acc[row * head_dim + dim] = 0.0f;
        }
        if (tid == 0u) {
            scalars[row * 4u + 0u] = -3.402823466e38f;
            scalars[row * 4u + 1u] = 0.0f;
        }
    }
    __syncthreads();

    for (unsigned int pos = 0u; pos < max_visible; ++pos) {
        const unsigned int logical_page = page_tokens == 0u ? 0u : pos / page_tokens;
        const unsigned int page_offset = page_tokens == 0u ? pos : pos - logical_page * page_tokens;
        const unsigned int physical_page = block_tables[logical_page];
        const unsigned int physical_slot = physical_page * page_tokens + page_offset;
        const bool physical_valid = physical_slot < physical_slots;

        if (physical_valid) {
            const unsigned short* k =
                key_cache + (size_t(physical_slot) * num_kv_heads + kv_head) * head_dim;
            const unsigned short* v =
                value_cache + (size_t(physical_slot) * num_kv_heads + kv_head) * head_dim;
            for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
                k_shared[dim] = f16_bits_to_float(k[dim]);
                v_shared[dim] = f16_bits_to_float(v[dim]);
            }
        }
        __syncthreads();

        for (unsigned int row = 0u; row < q_block; ++row) {
            float dot = 0.0f;
            if (valid[row] && physical_valid && pos < visible_len[row]) {
                for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
                    dot += q_shared[row * head_dim + dim] * k_shared[dim];
                }
            }
            #pragma unroll
            for (unsigned int offset = 16u; offset > 0u; offset >>= 1) {
                dot += __shfl_down_sync(0xffffffffu, dot, offset);
            }
            if (lane == 0u) {
                partial[row * nwarps + warp] = dot;
            }
        }
        __syncthreads();

        if (tid < q_block) {
            float row_sum = 0.0f;
            for (unsigned int w = 0u; w < nwarps; ++w) {
                row_sum += partial[tid * nwarps + w];
            }
            partial[tid * nwarps] = row_sum;
        }
        __syncthreads();

        if (tid == 0u) {
            #pragma unroll
            for (unsigned int row = 0u; row < q_block; ++row) {
                if (valid[row] && physical_valid && pos < visible_len[row]) {
                    const float score = partial[row * nwarps] * scale;
                    const float old_m = scalars[row * 4u + 0u];
                    const float old_l = scalars[row * 4u + 1u];
                    const float new_m = fmaxf(old_m, score);
                    const float alpha = old_l > 0.0f ? expf(old_m - new_m) : 0.0f;
                    const float beta = expf(score - new_m);
                    scalars[row * 4u + 2u] = alpha;
                    scalars[row * 4u + 3u] = beta;
                    scalars[row * 4u + 0u] = new_m;
                    scalars[row * 4u + 1u] = old_l * alpha + beta;
                }
            }
        }
        __syncthreads();

        if (physical_valid) {
            for (unsigned int row = 0u; row < q_block; ++row) {
                if (!valid[row] || pos >= visible_len[row]) {
                    continue;
                }
                const float alpha = scalars[row * 4u + 2u];
                const float beta = scalars[row * 4u + 3u];
                for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
                    const size_t offset = size_t(row) * head_dim + dim;
                    acc[offset] = acc[offset] * alpha + beta * v_shared[dim];
                }
            }
        }
        __syncthreads();
    }

    for (unsigned int row = 0u; row < q_block; ++row) {
        if (!valid[row]) {
            continue;
        }
        const unsigned int global_q = global_q_base + row;
        float* out = output + (size_t(global_q) * num_attention_heads + head) * head_dim;
        const float denom = fmaxf(scalars[row * 4u + 1u], 1.0e-20f);
        for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
            out[dim] = acc[row * head_dim + dim] / denom;
        }
    }
}

extern "C" __global__ void aegis_attention_prefill_dense_halfq_block4(
    const unsigned short* key_cache,
    const unsigned short* value_cache,
    const unsigned short* query,
    const unsigned int start_position,
    const unsigned int total_q,
    const unsigned int context_len,
    const unsigned int num_attention_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    float* output
) {
    const unsigned int q_block = 4u;
    const unsigned int head = blockIdx.x;
    const unsigned int global_q_base = blockIdx.y * q_block;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5u;
    const unsigned int nwarps = blockDim.x >> 5u;
    if (head >= num_attention_heads || global_q_base >= total_q) {
        return;
    }

    const unsigned int group = num_attention_heads / num_kv_heads;
    const unsigned int kv_head = head / group;
    const float scale = rsqrtf(float(head_dim));

    unsigned int visible_len[4];
    bool valid[4];
    unsigned int max_visible = 0u;
    #pragma unroll
    for (unsigned int row = 0u; row < q_block; ++row) {
        const unsigned int global_q = global_q_base + row;
        valid[row] = global_q < total_q;
        if (valid[row]) {
            visible_len[row] = min(context_len, start_position + global_q + 1u);
            valid[row] = visible_len[row] > 0u;
            max_visible = max(max_visible, visible_len[row]);
        } else {
            visible_len[row] = 0u;
        }
    }
    if (max_visible == 0u) {
        return;
    }

    extern __shared__ float shared[];
    float* partial = shared;
    float* q_shared = partial + q_block * nwarps;
    float* k_shared = q_shared + q_block * head_dim;
    float* v_shared = k_shared + head_dim;
    float* acc = v_shared + head_dim;
    float* scalars = acc + q_block * head_dim;

    for (unsigned int row = 0u; row < q_block; ++row) {
        if (!valid[row]) {
            continue;
        }
        const unsigned int global_q = global_q_base + row;
        const unsigned short* q =
            query + (size_t(global_q) * num_attention_heads + head) * head_dim;
        for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
            q_shared[row * head_dim + dim] = f16_bits_to_float(q[dim]);
            acc[row * head_dim + dim] = 0.0f;
        }
        if (tid == 0u) {
            scalars[row * 4u + 0u] = -3.402823466e38f;
            scalars[row * 4u + 1u] = 0.0f;
        }
    }
    __syncthreads();

    for (unsigned int pos = 0u; pos < max_visible; ++pos) {
        const unsigned short* k =
            key_cache + (size_t(pos) * num_kv_heads + kv_head) * head_dim;
        const unsigned short* v =
            value_cache + (size_t(pos) * num_kv_heads + kv_head) * head_dim;
        for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
            k_shared[dim] = f16_bits_to_float(k[dim]);
            v_shared[dim] = f16_bits_to_float(v[dim]);
        }
        __syncthreads();

        for (unsigned int row = 0u; row < q_block; ++row) {
            float dot = 0.0f;
            if (valid[row] && pos < visible_len[row]) {
                for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
                    dot += q_shared[row * head_dim + dim] * k_shared[dim];
                }
            }
            #pragma unroll
            for (unsigned int offset = 16u; offset > 0u; offset >>= 1) {
                dot += __shfl_down_sync(0xffffffffu, dot, offset);
            }
            if (lane == 0u) {
                partial[row * nwarps + warp] = dot;
            }
        }
        __syncthreads();

        if (tid < q_block) {
            float row_sum = 0.0f;
            for (unsigned int w = 0u; w < nwarps; ++w) {
                row_sum += partial[tid * nwarps + w];
            }
            partial[tid * nwarps] = row_sum;
        }
        __syncthreads();

        if (tid == 0u) {
            #pragma unroll
            for (unsigned int row = 0u; row < q_block; ++row) {
                if (valid[row] && pos < visible_len[row]) {
                    const float score = partial[row * nwarps] * scale;
                    const float old_m = scalars[row * 4u + 0u];
                    const float old_l = scalars[row * 4u + 1u];
                    const float new_m = fmaxf(old_m, score);
                    const float alpha = old_l > 0.0f ? expf(old_m - new_m) : 0.0f;
                    const float beta = expf(score - new_m);
                    scalars[row * 4u + 2u] = alpha;
                    scalars[row * 4u + 3u] = beta;
                    scalars[row * 4u + 0u] = new_m;
                    scalars[row * 4u + 1u] = old_l * alpha + beta;
                }
            }
        }
        __syncthreads();

        for (unsigned int row = 0u; row < q_block; ++row) {
            if (!valid[row] || pos >= visible_len[row]) {
                continue;
            }
            const float alpha = scalars[row * 4u + 2u];
            const float beta = scalars[row * 4u + 3u];
            for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
                const size_t offset = size_t(row) * head_dim + dim;
                acc[offset] = acc[offset] * alpha + beta * v_shared[dim];
            }
        }
        __syncthreads();
    }

    for (unsigned int row = 0u; row < q_block; ++row) {
        if (!valid[row]) {
            continue;
        }
        const unsigned int global_q = global_q_base + row;
        float* out = output + (size_t(global_q) * num_attention_heads + head) * head_dim;
        const float denom = fmaxf(scalars[row * 4u + 1u], 1.0e-20f);
        for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
            out[dim] = acc[row * head_dim + dim] / denom;
        }
    }
}

extern "C" __global__ void aegis_attention_prefill_paged_varlen_halfq_block4_split(
    const unsigned short* key_cache,
    const unsigned short* value_cache,
    const unsigned short* query,
    const unsigned int* slot_mapping,
    const unsigned int* cu_q,
    const unsigned int* context_lens,
    const unsigned int* block_tables,
    const unsigned int num_sequences,
    const unsigned int total_q,
    const unsigned int num_attention_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int page_tokens,
    const unsigned int split_tokens,
    const unsigned int split_count,
    const unsigned int block_table_stride,
    const unsigned int physical_slots,
    float* partial_acc,
    float* partial_m,
    float* partial_l
) {
    const unsigned int q_block = 4u;
    const unsigned int head = blockIdx.x;
    const unsigned int q_block_idx = blockIdx.y;
    const unsigned int split = blockIdx.z;
    const unsigned int global_q_base = q_block_idx * q_block;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5u;
    const unsigned int nwarps = blockDim.x >> 5u;
    if (head >= num_attention_heads || global_q_base >= total_q || split >= split_count || num_sequences != 1u) {
        return;
    }

    const unsigned int q_start = cu_q[0];
    const unsigned int q_end = cu_q[1];
    const unsigned int q_len = q_end - q_start;
    const unsigned int context_len = context_lens[0];
    const unsigned int group = num_attention_heads / num_kv_heads;
    const unsigned int kv_head = head / group;
    const float scale = rsqrtf(float(head_dim));
    const unsigned int split_start = split * split_tokens;
    const unsigned int split_end = min(context_len, split_start + split_tokens);

    unsigned int visible_len[4];
    bool valid[4];
    bool any_valid = false;
    #pragma unroll
    for (unsigned int row = 0u; row < q_block; ++row) {
        const unsigned int global_q = global_q_base + row;
        valid[row] = global_q < total_q
            && global_q >= q_start
            && global_q < q_end
            && slot_mapping[global_q] != 0xffffffffu;
        if (valid[row]) {
            const unsigned int q_in_seq = global_q - q_start;
            const unsigned int hidden_future = q_len - q_in_seq - 1u;
            visible_len[row] = context_len > hidden_future
                ? context_len - hidden_future
                : 0u;
            valid[row] = visible_len[row] > split_start;
            any_valid = any_valid || valid[row];
        } else {
            visible_len[row] = 0u;
        }
    }

    const size_t stats_base =
        ((size_t(q_block_idx) * num_attention_heads + head) * split_count + split) * q_block;
    const size_t acc_base = stats_base * head_dim;
    if (!any_valid || split_start >= split_end) {
        for (unsigned int row = 0u; row < q_block; ++row) {
            for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
                partial_acc[(stats_base + row) * head_dim + dim] = 0.0f;
            }
            if (tid == 0u) {
                partial_m[stats_base + row] = -3.402823466e38f;
                partial_l[stats_base + row] = 0.0f;
            }
        }
        return;
    }

    extern __shared__ float shared[];
    float* partial = shared;
    float* q_shared = partial + q_block * nwarps;
    float* k_shared = q_shared + q_block * head_dim;
    float* v_shared = k_shared + head_dim;
    float* acc = v_shared + head_dim;
    float* scalars = acc + q_block * head_dim;

    for (unsigned int row = 0u; row < q_block; ++row) {
        const unsigned int global_q = global_q_base + row;
        if (valid[row]) {
            const unsigned short* q =
                query + (size_t(global_q) * num_attention_heads + head) * head_dim;
            for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
                q_shared[row * head_dim + dim] = f16_bits_to_float(q[dim]);
                acc[row * head_dim + dim] = 0.0f;
            }
        } else {
            for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
                acc[row * head_dim + dim] = 0.0f;
            }
        }
        if (tid == 0u) {
            scalars[row * 4u + 0u] = -3.402823466e38f;
            scalars[row * 4u + 1u] = 0.0f;
        }
    }
    __syncthreads();

    for (unsigned int pos = split_start; pos < split_end; ++pos) {
        const unsigned int logical_page = page_tokens == 0u ? 0u : pos / page_tokens;
        const unsigned int page_offset = page_tokens == 0u ? pos : pos - logical_page * page_tokens;
        const unsigned int physical_page = block_tables[logical_page];
        const unsigned int physical_slot = physical_page * page_tokens + page_offset;
        const bool physical_valid = physical_slot < physical_slots;

        if (physical_valid) {
            const unsigned short* k =
                key_cache + (size_t(physical_slot) * num_kv_heads + kv_head) * head_dim;
            const unsigned short* v =
                value_cache + (size_t(physical_slot) * num_kv_heads + kv_head) * head_dim;
            for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
                k_shared[dim] = f16_bits_to_float(k[dim]);
                v_shared[dim] = f16_bits_to_float(v[dim]);
            }
        }
        __syncthreads();

        for (unsigned int row = 0u; row < q_block; ++row) {
            float dot = 0.0f;
            if (valid[row] && physical_valid && pos < visible_len[row]) {
                for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
                    dot += q_shared[row * head_dim + dim] * k_shared[dim];
                }
            }
            #pragma unroll
            for (unsigned int offset = 16u; offset > 0u; offset >>= 1) {
                dot += __shfl_down_sync(0xffffffffu, dot, offset);
            }
            if (lane == 0u) {
                partial[row * nwarps + warp] = dot;
            }
        }
        __syncthreads();

        if (tid < q_block) {
            float row_sum = 0.0f;
            for (unsigned int w = 0u; w < nwarps; ++w) {
                row_sum += partial[tid * nwarps + w];
            }
            partial[tid * nwarps] = row_sum;
        }
        __syncthreads();

        if (tid == 0u) {
            #pragma unroll
            for (unsigned int row = 0u; row < q_block; ++row) {
                if (valid[row] && physical_valid && pos < visible_len[row]) {
                    const float score = partial[row * nwarps] * scale;
                    const float old_m = scalars[row * 4u + 0u];
                    const float old_l = scalars[row * 4u + 1u];
                    const float new_m = fmaxf(old_m, score);
                    const float alpha = old_l > 0.0f ? expf(old_m - new_m) : 0.0f;
                    const float beta = expf(score - new_m);
                    scalars[row * 4u + 2u] = alpha;
                    scalars[row * 4u + 3u] = beta;
                    scalars[row * 4u + 0u] = new_m;
                    scalars[row * 4u + 1u] = old_l * alpha + beta;
                }
            }
        }
        __syncthreads();

        if (physical_valid) {
            for (unsigned int row = 0u; row < q_block; ++row) {
                if (!valid[row] || pos >= visible_len[row]) {
                    continue;
                }
                const float alpha = scalars[row * 4u + 2u];
                const float beta = scalars[row * 4u + 3u];
                for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
                    const size_t offset = size_t(row) * head_dim + dim;
                    acc[offset] = acc[offset] * alpha + beta * v_shared[dim];
                }
            }
        }
        __syncthreads();
    }

    for (unsigned int row = 0u; row < q_block; ++row) {
        for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
            partial_acc[acc_base + size_t(row) * head_dim + dim] = acc[row * head_dim + dim];
        }
        if (tid == 0u) {
            partial_m[stats_base + row] = scalars[row * 4u + 0u];
            partial_l[stats_base + row] = scalars[row * 4u + 1u];
        }
    }
}

extern "C" __global__ void aegis_attention_prefill_paged_varlen_halfq_block4_combine(
    const float* partial_acc,
    const float* partial_m,
    const float* partial_l,
    const unsigned int total_q,
    const unsigned int num_attention_heads,
    const unsigned int head_dim,
    const unsigned int split_count,
    float* output
) {
    const unsigned int q_block = 4u;
    const unsigned int head = blockIdx.x;
    const unsigned int q_block_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int global_q_base = q_block_idx * q_block;
    if (head >= num_attention_heads || global_q_base >= total_q) {
        return;
    }

    extern __shared__ float shared[];
    float* acc = shared;
    float* scalars = acc + q_block * head_dim;
    if (tid == 0u) {
        #pragma unroll
        for (unsigned int row = 0u; row < q_block; ++row) {
            scalars[row * 4u + 0u] = -3.402823466e38f;
            scalars[row * 4u + 1u] = 0.0f;
            scalars[row * 4u + 2u] = 1.0f;
            scalars[row * 4u + 3u] = 0.0f;
        }
    }
    for (unsigned int row = 0u; row < q_block; ++row) {
        for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
            acc[row * head_dim + dim] = 0.0f;
        }
    }
    __syncthreads();

    for (unsigned int split = 0u; split < split_count; ++split) {
        const size_t stats_base =
            ((size_t(q_block_idx) * num_attention_heads + head) * split_count + split) * q_block;
        if (tid == 0u) {
            #pragma unroll
            for (unsigned int row = 0u; row < q_block; ++row) {
                const float local_l = partial_l[stats_base + row];
                if (local_l > 0.0f) {
                    const float local_m = partial_m[stats_base + row];
                    const float old_m = scalars[row * 4u + 0u];
                    const float old_l = scalars[row * 4u + 1u];
                    const float new_m = fmaxf(old_m, local_m);
                    const float alpha = old_l > 0.0f ? expf(old_m - new_m) : 0.0f;
                    const float beta = expf(local_m - new_m);
                    scalars[row * 4u + 0u] = new_m;
                    scalars[row * 4u + 1u] = old_l * alpha + local_l * beta;
                    scalars[row * 4u + 2u] = alpha;
                    scalars[row * 4u + 3u] = beta;
                } else {
                    scalars[row * 4u + 2u] = 1.0f;
                    scalars[row * 4u + 3u] = 0.0f;
                }
            }
        }
        __syncthreads();

        for (unsigned int row = 0u; row < q_block; ++row) {
            const float local_l = partial_l[stats_base + row];
            if (local_l <= 0.0f) {
                continue;
            }
            const float alpha = scalars[row * 4u + 2u];
            const float beta = scalars[row * 4u + 3u];
            const size_t base = stats_base * head_dim + size_t(row) * head_dim;
            for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
                acc[row * head_dim + dim] =
                    acc[row * head_dim + dim] * alpha + partial_acc[base + dim] * beta;
            }
        }
        __syncthreads();
    }

    for (unsigned int row = 0u; row < q_block; ++row) {
        const unsigned int global_q = global_q_base + row;
        if (global_q >= total_q) {
            continue;
        }
        float* out = output + (size_t(global_q) * num_attention_heads + head) * head_dim;
        const float denom = fmaxf(scalars[row * 4u + 1u], 1.0e-20f);
        for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
            out[dim] = acc[row * head_dim + dim] / denom;
        }
    }
}

extern "C" __global__ void aegis_attention_prefill_paged_varlen_warp(
    const unsigned short* key_cache,
    const unsigned short* value_cache,
    const float* query,
    const unsigned int* slot_mapping,
    const unsigned int* cu_q,
    const unsigned int* context_lens,
    const unsigned int* block_tables,
    const unsigned int num_sequences,
    const unsigned int total_q,
    const unsigned int num_attention_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int page_tokens,
    const unsigned int block_table_stride,
    const unsigned int physical_slots,
    float* output
) {
    const unsigned int head = blockIdx.x;
    const unsigned int global_q = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5u;
    const unsigned int nwarps = blockDim.x >> 5u;
    if (global_q >= total_q || head >= num_attention_heads || nwarps == 0u) {
        return;
    }
    const unsigned int current_slot = slot_mapping[global_q];
    if (current_slot == 0xffffffffu) {
        return;
    }

    unsigned int seq = 0u;
    unsigned int q_start = 0u;
    unsigned int q_end = 0u;
    for (; seq < num_sequences; ++seq) {
        q_start = cu_q[seq];
        q_end = cu_q[seq + 1u];
        if (global_q >= q_start && global_q < q_end) {
            break;
        }
    }
    if (seq >= num_sequences || q_end <= q_start) {
        return;
    }

    const unsigned int q_in_seq = global_q - q_start;
    const unsigned int q_len = q_end - q_start;
    const unsigned int context_len = context_lens[seq];
    const unsigned int hidden_future = q_len - q_in_seq - 1u;
    const unsigned int visible_len = context_len > hidden_future
        ? context_len - hidden_future
        : 0u;
    if (visible_len == 0u) {
        return;
    }

    extern __shared__ float shared[];
    float* warp_scores = shared;
    float* alphas = warp_scores + nwarps;
    float* betas = alphas + nwarps;
    float* acc = betas + nwarps;
    float* scalars = acc + head_dim;

    const unsigned int group = num_attention_heads / num_kv_heads;
    const unsigned int kv_head = head / group;
    const float* q = query + (size_t(global_q) * num_attention_heads + head) * head_dim;
    float* out = output + (size_t(global_q) * num_attention_heads + head) * head_dim;
    const float scale = rsqrtf(float(head_dim));

    for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
        acc[dim] = 0.0f;
    }
    if (tid == 0u) {
        scalars[0] = -3.402823466e38f;
        scalars[1] = 0.0f;
        scalars[2] = 0.0f;
        scalars[3] = 0.0f;
    }
    __syncthreads();

    for (unsigned int tile_base = 0u; tile_base < visible_len; tile_base += nwarps) {
        const unsigned int pos = tile_base + warp;
        float dot = 0.0f;
        if (pos < visible_len) {
            const unsigned int logical_page = page_tokens == 0u ? 0u : pos / page_tokens;
            const unsigned int page_offset = page_tokens == 0u ? pos : pos - logical_page * page_tokens;
            const unsigned int physical_page = block_tables[size_t(seq) * block_table_stride + logical_page];
            const unsigned int physical_slot = physical_page * page_tokens + page_offset;
            if (physical_slot < physical_slots) {
                const unsigned short* k =
                    key_cache + (size_t(physical_slot) * num_kv_heads + kv_head) * head_dim;

                for (unsigned int dim = lane; dim < head_dim; dim += 32u) {
                    dot += q[dim] * f16_bits_to_float(k[dim]);
                }
                for (unsigned int mask = 16u; mask > 0u; mask >>= 1u) {
                    dot += __shfl_down_sync(0xFFFFFFFFu, dot, mask, 32);
                }
            } else {
                dot = -3.402823466e38f / scale;
            }
        }
        if (lane == 0u) {
            warp_scores[warp] = pos < visible_len ? dot * scale : -3.402823466e38f;
        }
        __syncthreads();

        if (tid == 0u) {
            for (unsigned int w = 0u; w < nwarps; ++w) {
                const unsigned int candidate_pos = tile_base + w;
                if (candidate_pos < visible_len) {
                    const float score = warp_scores[w];
                    const float old_m = scalars[0];
                    const float old_l = scalars[1];
                    const float new_m = fmaxf(old_m, score);
                    const float alpha = old_l > 0.0f ? expf(old_m - new_m) : 0.0f;
                    const float beta = expf(score - new_m);
                    alphas[w] = alpha;
                    betas[w] = beta;
                    scalars[0] = new_m;
                    scalars[1] = old_l * alpha + beta;
                } else {
                    alphas[w] = 1.0f;
                    betas[w] = 0.0f;
                }
            }
        }
        __syncthreads();

        for (unsigned int w = 0u; w < nwarps; ++w) {
            const unsigned int candidate_pos = tile_base + w;
            if (candidate_pos >= visible_len) {
                continue;
            }
            const unsigned int logical_page = page_tokens == 0u ? 0u : candidate_pos / page_tokens;
            const unsigned int page_offset = page_tokens == 0u ? candidate_pos : candidate_pos - logical_page * page_tokens;
            const unsigned int physical_page = block_tables[size_t(seq) * block_table_stride + logical_page];
            const unsigned int physical_slot = physical_page * page_tokens + page_offset;
            if (physical_slot >= physical_slots) {
                continue;
            }
            const unsigned short* v =
                value_cache + (size_t(physical_slot) * num_kv_heads + kv_head) * head_dim;
            const float alpha = alphas[w];
            const float beta = betas[w];
            for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
                acc[dim] = acc[dim] * alpha + beta * f16_bits_to_float(v[dim]);
            }
            __syncthreads();
        }
    }

    const float denom = fmaxf(scalars[1], 1.0e-20f);
    for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
        out[dim] = acc[dim] / denom;
    }
}

extern "C" __global__ void aegis_attention_prefill_batched_warp(
    const unsigned short* key_cache,
    const unsigned short* value_cache,
    const float* query,
    const unsigned int start_position,
    const unsigned int batch,
    const unsigned int num_attention_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    float* output
) {
    const unsigned int head = blockIdx.x;
    const unsigned int batch_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5u;
    const unsigned int nwarps = blockDim.x >> 5u;
    if (batch_idx >= batch || head >= num_attention_heads || nwarps == 0u) {
        return;
    }
    const unsigned int seq_len = start_position + batch_idx + 1u;
    extern __shared__ float shared[];
    float* scores = shared;
    float* partial = shared + (start_position + batch);
    const unsigned int group = num_attention_heads / num_kv_heads;
    const unsigned int kv_head = head / group;
    const float* q = query + (size_t(batch_idx) * num_attention_heads + head) * head_dim;
    float* out = output + (size_t(batch_idx) * num_attention_heads + head) * head_dim;
    const float scale = rsqrtf(float(head_dim));

    float local_max = -3.402823466e38f;
    for (unsigned int pos = warp; pos < seq_len; pos += nwarps) {
        const unsigned short* k = key_cache + (size_t(pos) * num_kv_heads + kv_head) * head_dim;
        float score = 0.0f;
        for (unsigned int dim = lane; dim < head_dim; dim += 32u) {
            score += q[dim] * f16_bits_to_float(k[dim]);
        }
        for (unsigned int mask = 16u; mask > 0u; mask >>= 1u) {
            score += __shfl_down_sync(0xFFFFFFFFu, score, mask, 32);
        }
        if (lane == 0u) {
            score *= scale;
            scores[pos] = score;
            local_max = fmaxf(local_max, score);
        }
    }
    if (lane == 0u) {
        partial[warp] = local_max;
    }
    __syncthreads();
    if (tid == 0u) {
        float max_score = partial[0];
        for (unsigned int w = 1u; w < nwarps; ++w) {
            max_score = fmaxf(max_score, partial[w]);
        }
        partial[0] = max_score;
    }
    __syncthreads();
    const float max_score = partial[0];

    float local_sum = 0.0f;
    for (unsigned int pos = warp; pos < seq_len; pos += nwarps) {
        if (lane == 0u) {
            const float weight = expf(scores[pos] - max_score);
            scores[pos] = weight;
            local_sum += weight;
        }
    }
    if (lane == 0u) {
        partial[warp] = local_sum;
    }
    __syncthreads();
    if (tid == 0u) {
        float denom = partial[0];
        for (unsigned int w = 1u; w < nwarps; ++w) {
            denom += partial[w];
        }
        partial[0] = fmaxf(denom, 1.0e-20f);
    }
    __syncthreads();
    const float denom = partial[0];

    for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int pos = 0u; pos < seq_len; ++pos) {
            const unsigned short* v = value_cache + (size_t(pos) * num_kv_heads + kv_head) * head_dim;
            acc += (scores[pos] / denom) * f16_bits_to_float(v[dim]);
        }
        out[dim] = acc;
    }
}

extern "C" __global__ void aegis_copy_row_f32(
    const float* input,
    const unsigned int row,
    const unsigned int cols,
    float* output
) {
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        output[col] = input[size_t(row) * cols + col];
    }
}

#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <mma.h>

__device__ __forceinline__ float aegis_warp_reduce_max(float value) {
#pragma unroll
    for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

__device__ __forceinline__ float aegis_warp_reduce_sum(float value) {
#pragma unroll
    for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

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

extern "C" __global__ void aegis_attention_prefill_dense_halfq_warp_tile_hdim128(
    const unsigned short* __restrict__ key_cache,
    const unsigned short* __restrict__ value_cache,
    const unsigned short* __restrict__ query,
    const unsigned int start_position,
    const unsigned int total_q,
    const unsigned int context_len,
    const unsigned int num_attention_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    float* __restrict__ output
) {
    constexpr unsigned int hdim = 128u;
    constexpr unsigned int q_block = 16u;
    constexpr unsigned int k_tile = 32u;
    const unsigned int head = blockIdx.x;
    const unsigned int global_q_base = blockIdx.y * q_block;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5u;
    const unsigned int nwarps = blockDim.x >> 5u;
    if (head_dim != hdim || head >= num_attention_heads || warp >= q_block || nwarps < q_block) {
        return;
    }

    const unsigned int global_q = global_q_base + warp;
    const bool valid_q = global_q < total_q;
    const unsigned int last_q_in_block = min(total_q, global_q_base + q_block) - 1u;
    const unsigned int block_max_visible = global_q_base < total_q
        ? min(context_len, start_position + last_q_in_block + 1u)
        : 0u;
    const unsigned int visible_len = valid_q
        ? min(context_len, start_position + global_q + 1u)
        : 0u;
    if (block_max_visible == 0u) {
        return;
    }

    extern __shared__ __align__(16) unsigned char smem[];
    unsigned short* k_shared = reinterpret_cast<unsigned short*>(smem);
    unsigned short* v_shared = k_shared + k_tile * hdim;

    const unsigned int group = num_attention_heads / num_kv_heads;
    const unsigned int kv_head = head / group;
    const float scale = rsqrtf(float(hdim));

    float q_frag[4];
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (unsigned int i = 0u; i < 4u; ++i) {
        const unsigned int dim = lane + i * 32u;
        q_frag[i] = valid_q
            ? f16_bits_to_float(
                query[(size_t(global_q) * num_attention_heads + head) * hdim + dim])
            : 0.0f;
    }
    float running_m = -3.402823466e38f;
    float running_l = 0.0f;

    for (unsigned int tile_start = 0u; tile_start < block_max_visible; tile_start += k_tile) {
        const unsigned int tile_count = min(k_tile, block_max_visible - tile_start);
        for (unsigned int idx = tid; idx < k_tile * hdim; idx += blockDim.x) {
            const unsigned int col = idx / hdim;
            const unsigned int dim = idx - col * hdim;
            const unsigned int pos = tile_start + col;
            const bool valid_k = col < tile_count;
            const size_t kv_offset =
                (size_t(pos) * num_kv_heads + kv_head) * hdim + dim;
            k_shared[idx] = valid_k ? key_cache[kv_offset] : 0u;
            v_shared[idx] = valid_k ? value_cache[kv_offset] : 0u;
        }
        __syncthreads();

        float scores[k_tile];
        float tile_m = running_m;
#pragma unroll
        for (unsigned int col = 0u; col < k_tile; ++col) {
            float dot = 0.0f;
            if (valid_q && col < tile_count && tile_start + col < visible_len) {
#pragma unroll
                for (unsigned int i = 0u; i < 4u; ++i) {
                    const unsigned int dim = lane + i * 32u;
                    dot += q_frag[i] *
                        f16_bits_to_float(k_shared[col * hdim + dim]);
                }
            }
#pragma unroll
            for (unsigned int offset = 16u; offset > 0u; offset >>= 1) {
                dot += __shfl_down_sync(0xffffffffu, dot, offset);
            }
            float score = lane == 0u && valid_q && col < tile_count && tile_start + col < visible_len
                ? dot * scale
                : -3.402823466e38f;
            score = __shfl_sync(0xffffffffu, score, 0);
            scores[col] = score;
            tile_m = fmaxf(tile_m, score);
        }

        float tile_l = 0.0f;
        float weights[k_tile];
#pragma unroll
        for (unsigned int col = 0u; col < k_tile; ++col) {
            const float weight = scores[col] > -3.0e38f
                ? expf(scores[col] - tile_m)
                : 0.0f;
            weights[col] = weight;
            tile_l += weight;
        }
        const float alpha = running_l > 0.0f ? expf(running_m - tile_m) : 0.0f;

#pragma unroll
        for (unsigned int i = 0u; i < 4u; ++i) {
            float tile_acc = 0.0f;
            const unsigned int dim = lane + i * 32u;
#pragma unroll
            for (unsigned int col = 0u; col < k_tile; ++col) {
                tile_acc += weights[col] *
                    f16_bits_to_float(v_shared[col * hdim + dim]);
            }
            acc[i] = acc[i] * alpha + tile_acc;
        }
        running_m = tile_m;
        running_l = running_l * alpha + tile_l;
        __syncthreads();
    }

    const float inv_l = running_l > 0.0f ? 1.0f / running_l : 0.0f;
    if (!valid_q) {
        return;
    }
    float* out = output + (size_t(global_q) * num_attention_heads + head) * hdim;
#pragma unroll
    for (unsigned int i = 0u; i < 4u; ++i) {
        const unsigned int dim = lane + i * 32u;
        out[dim] = acc[i] * inv_l;
    }
}

extern "C" __global__ void aegis_attention_prefill_dense_halfq_wmma_hdim128(
    const unsigned short* __restrict__ key_cache,
    const unsigned short* __restrict__ value_cache,
    const unsigned short* __restrict__ query,
    const unsigned int start_position,
    const unsigned int total_q,
    const unsigned int context_len,
    const unsigned int num_attention_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    float* __restrict__ output
) {
    constexpr unsigned int hdim = 128u;
    constexpr unsigned int q_block = 16u;
    constexpr unsigned int k_tile = 32u;
    const unsigned int head = blockIdx.x;
    const unsigned int global_q_base = blockIdx.y * q_block;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5u;
    if (head_dim != hdim || head >= num_attention_heads || blockDim.x < 128u) {
        return;
    }

    const unsigned int last_q_in_block = min(total_q, global_q_base + q_block) - 1u;
    const unsigned int block_max_visible = global_q_base < total_q
        ? min(context_len, start_position + last_q_in_block + 1u)
        : 0u;
    if (block_max_visible == 0u) {
        return;
    }

    extern __shared__ __align__(16) unsigned char smem[];
    unsigned short* q_shared = reinterpret_cast<unsigned short*>(smem);
    unsigned short* k_shared = q_shared + q_block * hdim;
    unsigned short* v_shared = k_shared + k_tile * hdim;
    float* scores = reinterpret_cast<float*>(v_shared + k_tile * hdim);
    float* tile_acc = scores + q_block * k_tile;
    float* acc = tile_acc + q_block * hdim;
    float* scalars = acc + q_block * hdim;
    half* weights_half = reinterpret_cast<half*>(scalars + q_block * 3u);

    const unsigned int group = num_attention_heads / num_kv_heads;
    const unsigned int kv_head = head / group;
    const float scale = rsqrtf(float(hdim));
    const float log2e = 1.4426950408889634f;

    for (unsigned int idx = tid; idx < q_block * hdim; idx += blockDim.x) {
        const unsigned int row = idx / hdim;
        const unsigned int dim = idx - row * hdim;
        const unsigned int global_q = global_q_base + row;
        q_shared[idx] = global_q < total_q
            ? query[(size_t(global_q) * num_attention_heads + head) * hdim + dim]
            : 0u;
        acc[idx] = 0.0f;
    }
    for (unsigned int row = tid; row < q_block; row += blockDim.x) {
        scalars[row * 3u + 0u] = -3.402823466e38f;
        scalars[row * 3u + 1u] = 0.0f;
        scalars[row * 3u + 2u] = 0.0f;
    }
    __syncthreads();

    for (unsigned int tile_start = 0u; tile_start < block_max_visible; tile_start += k_tile) {
        const unsigned int tile_count = min(k_tile, block_max_visible - tile_start);
        for (unsigned int idx = tid; idx < k_tile * hdim; idx += blockDim.x) {
            const unsigned int col = idx / hdim;
            const unsigned int dim = idx - col * hdim;
            const unsigned int pos = tile_start + col;
            const bool valid_k = col < tile_count;
            const size_t kv_offset =
                (size_t(pos) * num_kv_heads + kv_head) * hdim + dim;
            k_shared[idx] = valid_k ? key_cache[kv_offset] : 0u;
            v_shared[idx] = valid_k ? value_cache[kv_offset] : 0u;
        }
        __syncthreads();

#if __CUDA_ARCH__ >= 800
        if (warp < 2u) {
            using namespace nvcuda;
            const unsigned int n_off = warp * 16u;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);
#pragma unroll
            for (unsigned int kk = 0u; kk < hdim; kk += 16u) {
                const half* a_ptr = reinterpret_cast<const half*>(q_shared + kk);
                const half* b_ptr = reinterpret_cast<const half*>(k_shared + n_off * hdim + kk);
                wmma::load_matrix_sync(a_frag, a_ptr, hdim);
                wmma::load_matrix_sync(b_frag, b_ptr, hdim);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(scores + n_off, c_frag, k_tile, wmma::mem_row_major);
        }
#endif
        __syncthreads();

        const unsigned int nwarps = blockDim.x >> 5u;
        for (unsigned int row = warp; row < q_block; row += nwarps) {
            const unsigned int global_q = global_q_base + row;
            const bool valid_q = global_q < total_q;
            const unsigned int visible_len = valid_q
                ? min(context_len, start_position + global_q + 1u)
                : 0u;
            const float old_m = scalars[row * 3u + 0u];
            const float old_l = scalars[row * 3u + 1u];
            const unsigned int pos = tile_start + lane;
            float score = (valid_q && lane < tile_count && pos < visible_len)
                ? scores[row * k_tile + lane] * scale
                : -3.402823466e38f;
            const float tile_m = aegis_warp_reduce_max(score);
            const float new_m = fmaxf(old_m, tile_m);
            float weight = 0.0f;
            if (score > -3.0e38f) {
                weight = exp2f((score - new_m) * log2e);
            }
            scores[row * k_tile + lane] = weight;
            const float tile_l = aegis_warp_reduce_sum(weight);
            if (lane == 0u) {
                const float alpha = old_l > 0.0f ? exp2f((old_m - new_m) * log2e) : 0.0f;
                scalars[row * 3u + 0u] = new_m;
                scalars[row * 3u + 1u] = old_l * alpha + tile_l;
                scalars[row * 3u + 2u] = alpha;
            }
        }
        __syncthreads();

        for (unsigned int idx = tid; idx < q_block * k_tile; idx += blockDim.x) {
            weights_half[idx] = __float2half_rn(scores[idx]);
        }
        __syncthreads();

#if __CUDA_ARCH__ >= 800
        if (warp < 8u) {
            using namespace nvcuda;
            const unsigned int n_off = warp * 16u;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> pv_frag;
            wmma::fill_fragment(pv_frag, 0.0f);
#pragma unroll
            for (unsigned int kk = 0u; kk < k_tile; kk += 16u) {
                const half* p_ptr = weights_half + kk;
                const half* v_ptr = reinterpret_cast<const half*>(v_shared + kk * hdim + n_off);
                wmma::load_matrix_sync(p_frag, p_ptr, k_tile);
                wmma::load_matrix_sync(v_frag, v_ptr, hdim);
                wmma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);
            }
            wmma::store_matrix_sync(tile_acc + n_off, pv_frag, hdim, wmma::mem_row_major);
        }
#endif
        __syncthreads();

        for (unsigned int idx = tid; idx < q_block * hdim; idx += blockDim.x) {
            const unsigned int row = idx / hdim;
            const unsigned int global_q = global_q_base + row;
            if (global_q < total_q) {
                acc[idx] = acc[idx] * scalars[row * 3u + 2u] + tile_acc[idx];
            }
        }
        __syncthreads();
    }

    for (unsigned int idx = tid; idx < q_block * hdim; idx += blockDim.x) {
        const unsigned int row = idx / hdim;
        const unsigned int dim = idx - row * hdim;
        const unsigned int global_q = global_q_base + row;
        if (global_q >= total_q) {
            continue;
        }
        const float denom = fmaxf(scalars[row * 3u + 1u], 1.0e-20f);
        output[(size_t(global_q) * num_attention_heads + head) * hdim + dim] = acc[idx] / denom;
    }
}

extern "C" __global__ void aegis_attention_prefill_dense_halfq_wmma_hdim128_fa(
    const unsigned short* __restrict__ key_cache,
    const unsigned short* __restrict__ value_cache,
    const unsigned short* __restrict__ query,
    const unsigned int start_position,
    const unsigned int total_q,
    const unsigned int context_len,
    const unsigned int num_attention_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    float* __restrict__ output
) {
    constexpr unsigned int hdim = 128u;
    constexpr unsigned int q_block = 16u;
    constexpr unsigned int k_tile = 32u;
    const unsigned int head = blockIdx.x;
    const unsigned int global_q_base = blockIdx.y * q_block;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5u;
    if (head_dim != hdim || head >= num_attention_heads || blockDim.x < 256u) {
        return;
    }

    const unsigned int last_q_in_block = min(total_q, global_q_base + q_block) - 1u;
    const unsigned int block_max_visible = global_q_base < total_q
        ? min(context_len, start_position + last_q_in_block + 1u)
        : 0u;
    if (block_max_visible == 0u) {
        return;
    }

    extern __shared__ __align__(16) unsigned char smem[];
    unsigned short* q_shared = reinterpret_cast<unsigned short*>(smem);
    unsigned short* k_shared = q_shared + q_block * hdim;
    unsigned short* v_shared = k_shared + k_tile * hdim;
    float* scores = reinterpret_cast<float*>(v_shared + k_tile * hdim);
    float* acc = scores + q_block * k_tile;
    float* scalars = acc + q_block * hdim;
    half* weights_half = reinterpret_cast<half*>(scalars + q_block * 3u);

    const unsigned int group = num_attention_heads / num_kv_heads;
    const unsigned int kv_head = head / group;
    const float scale = rsqrtf(float(hdim));
    const float log2e = 1.4426950408889634f;

    for (unsigned int idx = tid; idx < q_block * hdim; idx += blockDim.x) {
        const unsigned int row = idx / hdim;
        const unsigned int dim = idx - row * hdim;
        const unsigned int global_q = global_q_base + row;
        q_shared[idx] = global_q < total_q
            ? query[(size_t(global_q) * num_attention_heads + head) * hdim + dim]
            : 0u;
        acc[idx] = 0.0f;
    }
    for (unsigned int row = tid; row < q_block; row += blockDim.x) {
        scalars[row * 3u + 0u] = -3.402823466e38f;
        scalars[row * 3u + 1u] = 0.0f;
        scalars[row * 3u + 2u] = 0.0f;
    }
    __syncthreads();

    for (unsigned int tile_start = 0u; tile_start < block_max_visible; tile_start += k_tile) {
        const unsigned int tile_count = min(k_tile, block_max_visible - tile_start);
        for (unsigned int idx = tid; idx < k_tile * hdim; idx += blockDim.x) {
            const unsigned int col = idx / hdim;
            const unsigned int dim = idx - col * hdim;
            const unsigned int pos = tile_start + col;
            const bool valid_k = col < tile_count;
            const size_t kv_offset =
                (size_t(pos) * num_kv_heads + kv_head) * hdim + dim;
            k_shared[idx] = valid_k ? key_cache[kv_offset] : 0u;
            v_shared[idx] = valid_k ? value_cache[kv_offset] : 0u;
        }
        __syncthreads();

#if __CUDA_ARCH__ >= 800
        if (warp < 2u) {
            using namespace nvcuda;
            const unsigned int n_off = warp * 16u;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);
#pragma unroll
            for (unsigned int kk = 0u; kk < hdim; kk += 16u) {
                const half* a_ptr = reinterpret_cast<const half*>(q_shared + kk);
                const half* b_ptr = reinterpret_cast<const half*>(k_shared + n_off * hdim + kk);
                wmma::load_matrix_sync(a_frag, a_ptr, hdim);
                wmma::load_matrix_sync(b_frag, b_ptr, hdim);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(scores + n_off, c_frag, k_tile, wmma::mem_row_major);
        }
#endif
        __syncthreads();

        const unsigned int nwarps = blockDim.x >> 5u;
        for (unsigned int row = warp; row < q_block; row += nwarps) {
            const unsigned int global_q = global_q_base + row;
            const bool valid_q = global_q < total_q;
            const unsigned int visible_len = valid_q
                ? min(context_len, start_position + global_q + 1u)
                : 0u;
            const float old_m = scalars[row * 3u + 0u];
            const float old_l = scalars[row * 3u + 1u];
            const unsigned int pos = tile_start + lane;
            float score = (valid_q && lane < tile_count && pos < visible_len)
                ? scores[row * k_tile + lane] * scale
                : -3.402823466e38f;
            const float tile_m = aegis_warp_reduce_max(score);
            const float new_m = fmaxf(old_m, tile_m);
            float weight = 0.0f;
            if (score > -3.0e38f) {
                weight = exp2f((score - new_m) * log2e);
            }
            weights_half[row * k_tile + lane] = __float2half_rn(weight);
            const float tile_l = aegis_warp_reduce_sum(weight);
            if (lane == 0u) {
                const float alpha = old_l > 0.0f ? exp2f((old_m - new_m) * log2e) : 0.0f;
                scalars[row * 3u + 0u] = new_m;
                scalars[row * 3u + 1u] = old_l * alpha + tile_l;
                scalars[row * 3u + 2u] = alpha;
            }
        }
        __syncthreads();

        for (unsigned int idx = tid; idx < q_block * hdim; idx += blockDim.x) {
            const unsigned int row = idx / hdim;
            const unsigned int global_q = global_q_base + row;
            if (global_q < total_q) {
                acc[idx] *= scalars[row * 3u + 2u];
            }
        }
        __syncthreads();

#if __CUDA_ARCH__ >= 800
        if (warp < 8u) {
            using namespace nvcuda;
            const unsigned int n_off = warp * 16u;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> pv_frag;
            wmma::load_matrix_sync(pv_frag, acc + n_off, hdim, wmma::mem_row_major);
#pragma unroll
            for (unsigned int kk = 0u; kk < k_tile; kk += 16u) {
                const half* p_ptr = weights_half + kk;
                const half* v_ptr = reinterpret_cast<const half*>(v_shared + kk * hdim + n_off);
                wmma::load_matrix_sync(p_frag, p_ptr, k_tile);
                wmma::load_matrix_sync(v_frag, v_ptr, hdim);
                wmma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);
            }
            wmma::store_matrix_sync(acc + n_off, pv_frag, hdim, wmma::mem_row_major);
        }
#endif
        __syncthreads();
    }

    for (unsigned int idx = tid; idx < q_block * hdim; idx += blockDim.x) {
        const unsigned int row = idx / hdim;
        const unsigned int dim = idx - row * hdim;
        const unsigned int global_q = global_q_base + row;
        if (global_q >= total_q) {
            continue;
        }
        const float denom = fmaxf(scalars[row * 3u + 1u], 1.0e-20f);
        output[(size_t(global_q) * num_attention_heads + head) * hdim + dim] = acc[idx] / denom;
    }
}

extern "C" __global__ void aegis_attention_prefill_dense_halfq_wmma_hdim128_gqa4(
    const unsigned short* __restrict__ key_cache,
    const unsigned short* __restrict__ value_cache,
    const unsigned short* __restrict__ query,
    const unsigned int start_position,
    const unsigned int total_q,
    const unsigned int context_len,
    const unsigned int num_attention_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    float* __restrict__ output
) {
    constexpr unsigned int hdim = 128u;
    constexpr unsigned int q_tokens = 8u;
    constexpr unsigned int local_heads_max = 4u;
    constexpr unsigned int q_rows = q_tokens * local_heads_max;
    constexpr unsigned int k_tile = 32u;
    constexpr unsigned int score_stride = k_tile + 8u;
    constexpr unsigned int acc_stride = hdim + 8u;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5u;
    if (head_dim != hdim || num_kv_heads == 0u || num_attention_heads % num_kv_heads != 0u || blockDim.x < 256u) {
        return;
    }

    const unsigned int group = num_attention_heads / num_kv_heads;
    const unsigned int group_tiles = (group + local_heads_max - 1u) / local_heads_max;
    const unsigned int kv_head = blockIdx.x / group_tiles;
    const unsigned int local_head_base = (blockIdx.x - kv_head * group_tiles) * local_heads_max;
    const unsigned int global_q_base = blockIdx.y * q_tokens;
    if (kv_head >= num_kv_heads || global_q_base >= total_q) {
        return;
    }

    const unsigned int last_q_in_block = min(total_q, global_q_base + q_tokens) - 1u;
    const unsigned int block_max_visible = min(context_len, start_position + last_q_in_block + 1u);
    if (block_max_visible == 0u) {
        return;
    }

    extern __shared__ __align__(16) unsigned char smem[];
    unsigned short* q_shared = reinterpret_cast<unsigned short*>(smem);
    unsigned short* k_shared = q_shared + q_rows * hdim;
    unsigned short* v_shared = k_shared + k_tile * hdim;
    float* scores = reinterpret_cast<float*>(v_shared + k_tile * hdim);
    float* acc = scores + q_rows * score_stride;
    float* scalars = acc + q_rows * acc_stride;
    half* weights_half = reinterpret_cast<half*>(scalars + q_rows * 3u);

    const float scale = rsqrtf(float(hdim));
    const float log2e = 1.4426950408889634f;

    for (unsigned int idx = tid; idx < q_rows * hdim; idx += blockDim.x) {
        const unsigned int row = idx / hdim;
        const unsigned int dim = idx - row * hdim;
        const unsigned int local_head = row / q_tokens;
        const unsigned int token = row - local_head * q_tokens;
        const unsigned int head = kv_head * group + local_head_base + local_head;
        const unsigned int global_q = global_q_base + token;
        q_shared[idx] = (local_head_base + local_head < group && head < num_attention_heads && global_q < total_q)
            ? query[(size_t(global_q) * num_attention_heads + head) * hdim + dim]
            : 0u;
    }
    for (unsigned int idx = tid; idx < q_rows * acc_stride; idx += blockDim.x) {
        acc[idx] = 0.0f;
    }
    for (unsigned int row = tid; row < q_rows; row += blockDim.x) {
        scalars[row * 3u + 0u] = -3.402823466e38f;
        scalars[row * 3u + 1u] = 0.0f;
        scalars[row * 3u + 2u] = 0.0f;
    }
    __syncthreads();

#if __CUDA_ARCH__ >= 800
    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q_frag[8];
    if (warp < 4u) {
        const unsigned int row_block = warp >> 1u;
#pragma unroll
        for (unsigned int kk = 0u; kk < hdim; kk += 16u) {
            const half* q_ptr = reinterpret_cast<const half*>(q_shared + row_block * 16u * hdim + kk);
            wmma::load_matrix_sync(q_frag[kk / 16u], q_ptr, hdim);
        }
    }
#endif

    for (unsigned int tile_start = 0u; tile_start < block_max_visible; tile_start += k_tile) {
        const unsigned int tile_count = min(k_tile, block_max_visible - tile_start);
        for (unsigned int idx = tid; idx < k_tile * hdim; idx += blockDim.x) {
            const unsigned int col = idx / hdim;
            const unsigned int dim = idx - col * hdim;
            const unsigned int pos = tile_start + col;
            const bool valid_k = col < tile_count;
            const size_t kv_offset =
                (size_t(pos) * num_kv_heads + kv_head) * hdim + dim;
            k_shared[idx] = valid_k ? key_cache[kv_offset] : 0u;
            v_shared[idx] = valid_k ? value_cache[kv_offset] : 0u;
        }
        __syncthreads();

#if __CUDA_ARCH__ >= 800
        if (warp < 4u) {
            const unsigned int row_block = warp >> 1u;
            const unsigned int n_off = (warp & 1u) * 16u;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);
#pragma unroll
            for (unsigned int kk = 0u; kk < hdim; kk += 16u) {
                const half* b_ptr = reinterpret_cast<const half*>(k_shared + n_off * hdim + kk);
                wmma::load_matrix_sync(b_frag, b_ptr, hdim);
                wmma::mma_sync(c_frag, q_frag[kk / 16u], b_frag, c_frag);
            }
            wmma::store_matrix_sync(scores + row_block * 16u * score_stride + n_off, c_frag, score_stride, wmma::mem_row_major);
        }
#endif
        __syncthreads();

        const unsigned int nwarps = blockDim.x >> 5u;
        for (unsigned int row = warp; row < q_rows; row += nwarps) {
            const unsigned int local_head = row / q_tokens;
            const unsigned int token = row - local_head * q_tokens;
            const unsigned int head = kv_head * group + local_head_base + local_head;
            const unsigned int global_q = global_q_base + token;
            const bool valid_q = local_head_base + local_head < group && head < num_attention_heads && global_q < total_q;
            const unsigned int visible_len = valid_q
                ? min(context_len, start_position + global_q + 1u)
                : 0u;
            const float old_m = scalars[row * 3u + 0u];
            const float old_l = scalars[row * 3u + 1u];
            const unsigned int pos = tile_start + lane;
            float score = (valid_q && lane < tile_count && pos < visible_len)
                ? scores[row * score_stride + lane] * scale
                : -3.402823466e38f;
            const float tile_m = aegis_warp_reduce_max(score);
            const float new_m = fmaxf(old_m, tile_m);
            float weight = 0.0f;
            if (score > -3.0e38f) {
                weight = exp2f((score - new_m) * log2e);
            }
            weights_half[row * score_stride + lane] = __float2half_rn(weight);
            const float tile_l = aegis_warp_reduce_sum(weight);
            if (lane == 0u) {
                const float alpha = old_l > 0.0f ? exp2f((old_m - new_m) * log2e) : 0.0f;
                scalars[row * 3u + 0u] = new_m;
                scalars[row * 3u + 1u] = old_l * alpha + tile_l;
                scalars[row * 3u + 2u] = alpha;
            }
        }
        __syncthreads();

        for (unsigned int idx = tid; idx < q_rows * hdim; idx += blockDim.x) {
            const unsigned int row = idx / hdim;
            const unsigned int dim = idx - row * hdim;
            const unsigned int local_head = row / q_tokens;
            const unsigned int token = row - local_head * q_tokens;
            const unsigned int head = kv_head * group + local_head_base + local_head;
            const unsigned int global_q = global_q_base + token;
            if (local_head_base + local_head < group && head < num_attention_heads && global_q < total_q) {
                acc[row * acc_stride + dim] *= scalars[row * 3u + 2u];
            }
        }
        __syncthreads();

#if __CUDA_ARCH__ >= 800
        if (warp < 8u) {
            using namespace nvcuda;
            const unsigned int n_off = warp * 16u;
            for (unsigned int row_base = 0u; row_base < q_rows; row_base += 16u) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> pv_frag;
                wmma::load_matrix_sync(pv_frag, acc + row_base * acc_stride + n_off, acc_stride, wmma::mem_row_major);
#pragma unroll
                for (unsigned int kk = 0u; kk < k_tile; kk += 16u) {
                    const half* p_ptr = weights_half + row_base * score_stride + kk;
                    const half* v_ptr = reinterpret_cast<const half*>(v_shared + kk * hdim + n_off);
                    wmma::load_matrix_sync(p_frag, p_ptr, score_stride);
                    wmma::load_matrix_sync(v_frag, v_ptr, hdim);
                    wmma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);
                }
                wmma::store_matrix_sync(acc + row_base * acc_stride + n_off, pv_frag, acc_stride, wmma::mem_row_major);
            }
        }
#endif
        __syncthreads();
    }

    for (unsigned int idx = tid; idx < q_rows * hdim; idx += blockDim.x) {
        const unsigned int row = idx / hdim;
        const unsigned int dim = idx - row * hdim;
        const unsigned int local_head = row / q_tokens;
        const unsigned int token = row - local_head * q_tokens;
        const unsigned int head = kv_head * group + local_head_base + local_head;
        const unsigned int global_q = global_q_base + token;
        if (local_head_base + local_head >= group || head >= num_attention_heads || global_q >= total_q) {
            continue;
        }
        const float denom = fmaxf(scalars[row * 3u + 1u], 1.0e-20f);
        output[(size_t(global_q) * num_attention_heads + head) * hdim + dim] =
            acc[row * acc_stride + dim] / denom;
    }
}

extern "C" __global__ void aegis_attention_prefill_dense_halfq_wmma_hdim128_gqa4_split(
    const unsigned short* __restrict__ key_cache,
    const unsigned short* __restrict__ value_cache,
    const unsigned short* __restrict__ query,
    const unsigned int start_position,
    const unsigned int total_q,
    const unsigned int context_len,
    const unsigned int num_attention_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int split_tokens,
    const unsigned int split_count,
    float* __restrict__ partial_acc,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l
) {
    constexpr unsigned int hdim = 128u;
    constexpr unsigned int q_tokens = 8u;
    constexpr unsigned int local_heads_max = 4u;
    constexpr unsigned int q_rows = q_tokens * local_heads_max;
    constexpr unsigned int q_block_combine = 16u;
    constexpr unsigned int k_tile = 32u;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5u;
    const unsigned int split = blockIdx.z;
    if (head_dim != hdim || num_kv_heads == 0u || num_attention_heads % num_kv_heads != 0u || split >= split_count || blockDim.x < 256u) {
        return;
    }

    const unsigned int group = num_attention_heads / num_kv_heads;
    const unsigned int group_tiles = (group + local_heads_max - 1u) / local_heads_max;
    const unsigned int kv_head = blockIdx.x / group_tiles;
    const unsigned int local_head_base = (blockIdx.x - kv_head * group_tiles) * local_heads_max;
    const unsigned int global_q_base = blockIdx.y * q_tokens;
    if (kv_head >= num_kv_heads || global_q_base >= total_q) {
        return;
    }

    const unsigned int split_start = split * split_tokens;
    const unsigned int split_end = min(context_len, split_start + split_tokens);
    const unsigned int last_q_in_block = min(total_q, global_q_base + q_tokens) - 1u;
    const unsigned int block_max_visible = min(context_len, start_position + last_q_in_block + 1u);

    extern __shared__ __align__(16) unsigned char smem[];
    unsigned short* q_shared = reinterpret_cast<unsigned short*>(smem);
    unsigned short* k_shared = q_shared + q_rows * hdim;
    unsigned short* v_shared = k_shared + k_tile * hdim;
    float* scores = reinterpret_cast<float*>(v_shared + k_tile * hdim);
    float* acc = scores + q_rows * k_tile;
    float* scalars = acc + q_rows * hdim;
    half* weights_half = reinterpret_cast<half*>(scalars + q_rows * 3u);

    const float scale = rsqrtf(float(hdim));
    const float log2e = 1.4426950408889634f;

    for (unsigned int idx = tid; idx < q_rows * hdim; idx += blockDim.x) {
        const unsigned int row = idx / hdim;
        const unsigned int dim = idx - row * hdim;
        const unsigned int local_head = row / q_tokens;
        const unsigned int token = row - local_head * q_tokens;
        const unsigned int head = kv_head * group + local_head_base + local_head;
        const unsigned int global_q = global_q_base + token;
        q_shared[idx] = (local_head_base + local_head < group && head < num_attention_heads && global_q < total_q)
            ? query[(size_t(global_q) * num_attention_heads + head) * hdim + dim]
            : 0u;
        acc[idx] = 0.0f;
    }
    for (unsigned int row = tid; row < q_rows; row += blockDim.x) {
        scalars[row * 3u + 0u] = -3.402823466e38f;
        scalars[row * 3u + 1u] = 0.0f;
        scalars[row * 3u + 2u] = 0.0f;
    }
    __syncthreads();

    if (block_max_visible == 0u || split_start >= split_end || split_start >= block_max_visible) {
        for (unsigned int idx = tid; idx < q_rows * hdim; idx += blockDim.x) {
            const unsigned int row = idx / hdim;
            const unsigned int dim = idx - row * hdim;
            const unsigned int local_head = row / q_tokens;
            const unsigned int token = row - local_head * q_tokens;
            const unsigned int head = kv_head * group + local_head_base + local_head;
            const unsigned int global_q = global_q_base + token;
            if (local_head_base + local_head < group && head < num_attention_heads && global_q < total_q) {
                const unsigned int q_block_idx = global_q / q_block_combine;
                const unsigned int row_in_block = global_q - q_block_idx * q_block_combine;
                const size_t stats_index =
                    ((size_t(q_block_idx) * num_attention_heads + head) * split_count + split) * q_block_combine + row_in_block;
                partial_acc[stats_index * hdim + dim] = 0.0f;
            }
        }
        for (unsigned int row = tid; row < q_rows; row += blockDim.x) {
            const unsigned int local_head = row / q_tokens;
            const unsigned int token = row - local_head * q_tokens;
            const unsigned int head = kv_head * group + local_head_base + local_head;
            const unsigned int global_q = global_q_base + token;
            if (local_head_base + local_head < group && head < num_attention_heads && global_q < total_q) {
                const unsigned int q_block_idx = global_q / q_block_combine;
                const unsigned int row_in_block = global_q - q_block_idx * q_block_combine;
                const size_t stats_index =
                    ((size_t(q_block_idx) * num_attention_heads + head) * split_count + split) * q_block_combine + row_in_block;
                partial_m[stats_index] = -3.402823466e38f;
                partial_l[stats_index] = 0.0f;
            }
        }
        return;
    }

#if __CUDA_ARCH__ >= 800
    using namespace nvcuda;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q_frag[8];
    if (warp < 4u) {
        const unsigned int row_block = warp >> 1u;
#pragma unroll
        for (unsigned int kk = 0u; kk < hdim; kk += 16u) {
            const half* q_ptr = reinterpret_cast<const half*>(q_shared + row_block * 16u * hdim + kk);
            wmma::load_matrix_sync(q_frag[kk / 16u], q_ptr, hdim);
        }
    }
#endif

    const unsigned int split_visible_end = min(split_end, block_max_visible);
    for (unsigned int tile_start = split_start; tile_start < split_visible_end; tile_start += k_tile) {
        const unsigned int tile_count = min(k_tile, split_visible_end - tile_start);
        for (unsigned int idx = tid; idx < k_tile * hdim; idx += blockDim.x) {
            const unsigned int col = idx / hdim;
            const unsigned int dim = idx - col * hdim;
            const unsigned int pos = tile_start + col;
            const bool valid_k = col < tile_count;
            const size_t kv_offset =
                (size_t(pos) * num_kv_heads + kv_head) * hdim + dim;
            k_shared[idx] = valid_k ? key_cache[kv_offset] : 0u;
            v_shared[idx] = valid_k ? value_cache[kv_offset] : 0u;
        }
        __syncthreads();

#if __CUDA_ARCH__ >= 800
        if (warp < 4u) {
            const unsigned int row_block = warp >> 1u;
            const unsigned int n_off = (warp & 1u) * 16u;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);
#pragma unroll
            for (unsigned int kk = 0u; kk < hdim; kk += 16u) {
                const half* b_ptr = reinterpret_cast<const half*>(k_shared + n_off * hdim + kk);
                wmma::load_matrix_sync(b_frag, b_ptr, hdim);
                wmma::mma_sync(c_frag, q_frag[kk / 16u], b_frag, c_frag);
            }
            wmma::store_matrix_sync(scores + row_block * 16u * k_tile + n_off, c_frag, k_tile, wmma::mem_row_major);
        }
#endif
        __syncthreads();

        const unsigned int nwarps = blockDim.x >> 5u;
        for (unsigned int row = warp; row < q_rows; row += nwarps) {
            const unsigned int local_head = row / q_tokens;
            const unsigned int token = row - local_head * q_tokens;
            const unsigned int head = kv_head * group + local_head_base + local_head;
            const unsigned int global_q = global_q_base + token;
            const bool valid_q = local_head_base + local_head < group && head < num_attention_heads && global_q < total_q;
            const unsigned int visible_len = valid_q
                ? min(context_len, start_position + global_q + 1u)
                : 0u;
            const float old_m = scalars[row * 3u + 0u];
            const float old_l = scalars[row * 3u + 1u];
            const unsigned int pos = tile_start + lane;
            float score = (valid_q && lane < tile_count && pos < visible_len)
                ? scores[row * k_tile + lane] * scale
                : -3.402823466e38f;
            const float tile_m = aegis_warp_reduce_max(score);
            const float new_m = fmaxf(old_m, tile_m);
            float weight = 0.0f;
            if (score > -3.0e38f) {
                weight = exp2f((score - new_m) * log2e);
            }
            weights_half[row * k_tile + lane] = __float2half_rn(weight);
            const float tile_l = aegis_warp_reduce_sum(weight);
            if (lane == 0u) {
                const float alpha = old_l > 0.0f ? exp2f((old_m - new_m) * log2e) : 0.0f;
                scalars[row * 3u + 0u] = new_m;
                scalars[row * 3u + 1u] = old_l * alpha + tile_l;
                scalars[row * 3u + 2u] = alpha;
            }
        }
        __syncthreads();

        for (unsigned int idx = tid; idx < q_rows * hdim; idx += blockDim.x) {
            const unsigned int row = idx / hdim;
            const unsigned int local_head = row / q_tokens;
            const unsigned int token = row - local_head * q_tokens;
            const unsigned int head = kv_head * group + local_head_base + local_head;
            const unsigned int global_q = global_q_base + token;
            if (local_head_base + local_head < group && head < num_attention_heads && global_q < total_q) {
                acc[idx] *= scalars[row * 3u + 2u];
            }
        }
        __syncthreads();

#if __CUDA_ARCH__ >= 800
        if (warp < 8u) {
            using namespace nvcuda;
            const unsigned int n_off = warp * 16u;
            for (unsigned int row_base = 0u; row_base < q_rows; row_base += 16u) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> pv_frag;
                wmma::load_matrix_sync(pv_frag, acc + row_base * hdim + n_off, hdim, wmma::mem_row_major);
#pragma unroll
                for (unsigned int kk = 0u; kk < k_tile; kk += 16u) {
                    const half* p_ptr = weights_half + row_base * k_tile + kk;
                    const half* v_ptr = reinterpret_cast<const half*>(v_shared + kk * hdim + n_off);
                    wmma::load_matrix_sync(p_frag, p_ptr, k_tile);
                    wmma::load_matrix_sync(v_frag, v_ptr, hdim);
                    wmma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);
                }
                wmma::store_matrix_sync(acc + row_base * hdim + n_off, pv_frag, hdim, wmma::mem_row_major);
            }
        }
#endif
        __syncthreads();
    }

    for (unsigned int idx = tid; idx < q_rows * hdim; idx += blockDim.x) {
        const unsigned int row = idx / hdim;
        const unsigned int dim = idx - row * hdim;
        const unsigned int local_head = row / q_tokens;
        const unsigned int token = row - local_head * q_tokens;
        const unsigned int head = kv_head * group + local_head_base + local_head;
        const unsigned int global_q = global_q_base + token;
        if (local_head_base + local_head < group && head < num_attention_heads && global_q < total_q) {
            const unsigned int q_block_idx = global_q / q_block_combine;
            const unsigned int row_in_block = global_q - q_block_idx * q_block_combine;
            const size_t stats_index =
                ((size_t(q_block_idx) * num_attention_heads + head) * split_count + split) * q_block_combine + row_in_block;
            partial_acc[stats_index * hdim + dim] = acc[idx];
        }
    }
    for (unsigned int row = tid; row < q_rows; row += blockDim.x) {
        const unsigned int local_head = row / q_tokens;
        const unsigned int token = row - local_head * q_tokens;
        const unsigned int head = kv_head * group + local_head_base + local_head;
        const unsigned int global_q = global_q_base + token;
        if (local_head_base + local_head < group && head < num_attention_heads && global_q < total_q) {
            const unsigned int q_block_idx = global_q / q_block_combine;
            const unsigned int row_in_block = global_q - q_block_idx * q_block_combine;
            const size_t stats_index =
                ((size_t(q_block_idx) * num_attention_heads + head) * split_count + split) * q_block_combine + row_in_block;
            partial_m[stats_index] = scalars[row * 3u + 0u];
            partial_l[stats_index] = scalars[row * 3u + 1u];
        }
    }
}

extern "C" __global__ __cluster_dims__(2, 1, 1)
void aegis_attention_prefill_dense_halfq_wmma_hdim128_cluster2(
    const unsigned short* __restrict__ key_cache,
    const unsigned short* __restrict__ value_cache,
    const unsigned short* __restrict__ query,
    const unsigned int start_position,
    const unsigned int total_q,
    const unsigned int context_len,
    const unsigned int num_attention_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    float* __restrict__ output
) {
    namespace cg = cooperative_groups;
    constexpr unsigned int hdim = 128u;
    constexpr unsigned int q_block = 16u;
    constexpr unsigned int k_tile = 32u;
    constexpr unsigned int cluster_blocks = 2u;
    cg::cluster_group cluster = cg::this_cluster();
    const unsigned int cluster_rank = cluster.block_rank();
    const unsigned int head = blockIdx.x / cluster_blocks;
    const unsigned int global_q_base = blockIdx.y * q_block;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5u;
    if (head_dim != hdim || head >= num_attention_heads || blockDim.x < 256u
        || cluster.num_blocks() != cluster_blocks) {
        return;
    }

    const unsigned int last_q_in_block = min(total_q, global_q_base + q_block) - 1u;
    const unsigned int block_max_visible = global_q_base < total_q
        ? min(context_len, start_position + last_q_in_block + 1u)
        : 0u;
    if (block_max_visible == 0u) {
        return;
    }

    extern __shared__ __align__(16) unsigned char smem[];
    unsigned short* q_shared = reinterpret_cast<unsigned short*>(smem);
    unsigned short* k_shared = q_shared + q_block * hdim;
    unsigned short* v_shared = k_shared + k_tile * hdim;
    float* scores = reinterpret_cast<float*>(v_shared + k_tile * hdim);
    float* acc = scores + q_block * k_tile;
    float* scalars = acc + q_block * hdim;
    half* weights_half = reinterpret_cast<half*>(scalars + q_block * 3u);

    const unsigned int group = num_attention_heads / num_kv_heads;
    const unsigned int kv_head = head / group;
    const float scale = rsqrtf(float(hdim));
    const float log2e = 1.4426950408889634f;

    for (unsigned int idx = tid; idx < q_block * hdim; idx += blockDim.x) {
        const unsigned int row = idx / hdim;
        const unsigned int dim = idx - row * hdim;
        const unsigned int global_q = global_q_base + row;
        q_shared[idx] = global_q < total_q
            ? query[(size_t(global_q) * num_attention_heads + head) * hdim + dim]
            : 0u;
        acc[idx] = 0.0f;
    }
    for (unsigned int row = tid; row < q_block; row += blockDim.x) {
        scalars[row * 3u + 0u] = -3.402823466e38f;
        scalars[row * 3u + 1u] = 0.0f;
        scalars[row * 3u + 2u] = 0.0f;
    }
    __syncthreads();

    for (unsigned int tile_start = cluster_rank * k_tile;
         tile_start < block_max_visible;
         tile_start += cluster_blocks * k_tile) {
        const unsigned int tile_count = min(k_tile, block_max_visible - tile_start);
        for (unsigned int idx = tid; idx < k_tile * hdim; idx += blockDim.x) {
            const unsigned int col = idx / hdim;
            const unsigned int dim = idx - col * hdim;
            const unsigned int pos = tile_start + col;
            const bool valid_k = col < tile_count;
            const size_t kv_offset =
                (size_t(pos) * num_kv_heads + kv_head) * hdim + dim;
            k_shared[idx] = valid_k ? key_cache[kv_offset] : 0u;
            v_shared[idx] = valid_k ? value_cache[kv_offset] : 0u;
        }
        __syncthreads();

#if __CUDA_ARCH__ >= 800
        if (warp < 2u) {
            using namespace nvcuda;
            const unsigned int n_off = warp * 16u;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);
#pragma unroll
            for (unsigned int kk = 0u; kk < hdim; kk += 16u) {
                const half* a_ptr = reinterpret_cast<const half*>(q_shared + kk);
                const half* b_ptr = reinterpret_cast<const half*>(k_shared + n_off * hdim + kk);
                wmma::load_matrix_sync(a_frag, a_ptr, hdim);
                wmma::load_matrix_sync(b_frag, b_ptr, hdim);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(scores + n_off, c_frag, k_tile, wmma::mem_row_major);
        }
#endif
        __syncthreads();

        const unsigned int nwarps = blockDim.x >> 5u;
        for (unsigned int row = warp; row < q_block; row += nwarps) {
            const unsigned int global_q = global_q_base + row;
            const bool valid_q = global_q < total_q;
            const unsigned int visible_len = valid_q
                ? min(context_len, start_position + global_q + 1u)
                : 0u;
            const float old_m = scalars[row * 3u + 0u];
            const float old_l = scalars[row * 3u + 1u];
            const unsigned int pos = tile_start + lane;
            float score = (valid_q && lane < tile_count && pos < visible_len)
                ? scores[row * k_tile + lane] * scale
                : -3.402823466e38f;
            const float tile_m = aegis_warp_reduce_max(score);
            const float new_m = fmaxf(old_m, tile_m);
            float weight = 0.0f;
            if (score > -3.0e38f) {
                weight = exp2f((score - new_m) * log2e);
            }
            weights_half[row * k_tile + lane] = __float2half_rn(weight);
            const float tile_l = aegis_warp_reduce_sum(weight);
            if (lane == 0u) {
                const float alpha = old_l > 0.0f ? exp2f((old_m - new_m) * log2e) : 0.0f;
                scalars[row * 3u + 0u] = new_m;
                scalars[row * 3u + 1u] = old_l * alpha + tile_l;
                scalars[row * 3u + 2u] = alpha;
            }
        }
        __syncthreads();

        for (unsigned int idx = tid; idx < q_block * hdim; idx += blockDim.x) {
            const unsigned int row = idx / hdim;
            const unsigned int global_q = global_q_base + row;
            if (global_q < total_q) {
                acc[idx] *= scalars[row * 3u + 2u];
            }
        }
        __syncthreads();

#if __CUDA_ARCH__ >= 800
        if (warp < 8u) {
            using namespace nvcuda;
            const unsigned int n_off = warp * 16u;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> pv_frag;
            wmma::load_matrix_sync(pv_frag, acc + n_off, hdim, wmma::mem_row_major);
#pragma unroll
            for (unsigned int kk = 0u; kk < k_tile; kk += 16u) {
                const half* p_ptr = weights_half + kk;
                const half* v_ptr = reinterpret_cast<const half*>(v_shared + kk * hdim + n_off);
                wmma::load_matrix_sync(p_frag, p_ptr, k_tile);
                wmma::load_matrix_sync(v_frag, v_ptr, hdim);
                wmma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);
            }
            wmma::store_matrix_sync(acc + n_off, pv_frag, hdim, wmma::mem_row_major);
        }
#endif
        __syncthreads();
    }

    cluster.sync();
    if (cluster_rank == 0u) {
        float* remote_acc = cluster.map_shared_rank(acc, 1);
        float* remote_scalars = cluster.map_shared_rank(scalars, 1);
        for (unsigned int idx = tid; idx < q_block * hdim; idx += blockDim.x) {
            const unsigned int row = idx / hdim;
            const unsigned int dim = idx - row * hdim;
            const unsigned int global_q = global_q_base + row;
            if (global_q >= total_q) {
                continue;
            }
            const float m0 = scalars[row * 3u + 0u];
            const float l0 = scalars[row * 3u + 1u];
            const float m1 = remote_scalars[row * 3u + 0u];
            const float l1 = remote_scalars[row * 3u + 1u];
            const float m = fmaxf(m0, m1);
            const float a0 = l0 > 0.0f ? exp2f((m0 - m) * log2e) : 0.0f;
            const float a1 = l1 > 0.0f ? exp2f((m1 - m) * log2e) : 0.0f;
            const float denom = fmaxf(l0 * a0 + l1 * a1, 1.0e-20f);
            const float value = acc[idx] * a0 + remote_acc[idx] * a1;
            output[(size_t(global_q) * num_attention_heads + head) * hdim + dim] = value / denom;
        }
    }
    cluster.sync();
}

extern "C" __global__ void aegis_attention_prefill_dense_halfq_wmma_hdim128_q32(
    const unsigned short* __restrict__ key_cache,
    const unsigned short* __restrict__ value_cache,
    const unsigned short* __restrict__ query,
    const unsigned int start_position,
    const unsigned int total_q,
    const unsigned int context_len,
    const unsigned int num_attention_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    float* __restrict__ output
) {
    constexpr unsigned int hdim = 128u;
    constexpr unsigned int q_block = 32u;
    constexpr unsigned int k_tile = 32u;
    const unsigned int head = blockIdx.x;
    const unsigned int global_q_base = blockIdx.y * q_block;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5u;
    if (head_dim != hdim || head >= num_attention_heads || blockDim.x < 256u) {
        return;
    }

    const unsigned int last_q_in_block = min(total_q, global_q_base + q_block) - 1u;
    const unsigned int block_max_visible = global_q_base < total_q
        ? min(context_len, start_position + last_q_in_block + 1u)
        : 0u;
    if (block_max_visible == 0u) {
        return;
    }

    extern __shared__ __align__(16) unsigned char smem[];
    unsigned short* q_shared = reinterpret_cast<unsigned short*>(smem);
    unsigned short* k_shared = q_shared + q_block * hdim;
    unsigned short* v_shared = k_shared + k_tile * hdim;
    float* scores = reinterpret_cast<float*>(v_shared + k_tile * hdim);
    float* tile_acc = scores + q_block * k_tile;
    float* acc = tile_acc + q_block * hdim;
    float* scalars = acc + q_block * hdim;
    half* weights_half = reinterpret_cast<half*>(scalars + q_block * 3u);

    const unsigned int group = num_attention_heads / num_kv_heads;
    const unsigned int kv_head = head / group;
    const float scale = rsqrtf(float(hdim));
    const float log2e = 1.4426950408889634f;

    for (unsigned int idx = tid; idx < q_block * hdim; idx += blockDim.x) {
        const unsigned int row = idx / hdim;
        const unsigned int dim = idx - row * hdim;
        const unsigned int global_q = global_q_base + row;
        q_shared[idx] = global_q < total_q
            ? query[(size_t(global_q) * num_attention_heads + head) * hdim + dim]
            : 0u;
        acc[idx] = 0.0f;
    }
    for (unsigned int row = tid; row < q_block; row += blockDim.x) {
        scalars[row * 3u + 0u] = -3.402823466e38f;
        scalars[row * 3u + 1u] = 0.0f;
        scalars[row * 3u + 2u] = 0.0f;
    }
    __syncthreads();

    for (unsigned int tile_start = 0u; tile_start < block_max_visible; tile_start += k_tile) {
        const unsigned int tile_count = min(k_tile, block_max_visible - tile_start);
        for (unsigned int idx = tid; idx < k_tile * hdim; idx += blockDim.x) {
            const unsigned int col = idx / hdim;
            const unsigned int dim = idx - col * hdim;
            const unsigned int pos = tile_start + col;
            const bool valid_k = col < tile_count;
            const size_t kv_offset =
                (size_t(pos) * num_kv_heads + kv_head) * hdim + dim;
            k_shared[idx] = valid_k ? key_cache[kv_offset] : 0u;
            v_shared[idx] = valid_k ? value_cache[kv_offset] : 0u;
        }
        __syncthreads();

#if __CUDA_ARCH__ >= 800
        if (warp < 4u) {
            using namespace nvcuda;
            const unsigned int row_off = (warp >> 1u) * 16u;
            const unsigned int n_off = (warp & 1u) * 16u;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);
#pragma unroll
            for (unsigned int kk = 0u; kk < hdim; kk += 16u) {
                const half* a_ptr = reinterpret_cast<const half*>(q_shared + row_off * hdim + kk);
                const half* b_ptr = reinterpret_cast<const half*>(k_shared + n_off * hdim + kk);
                wmma::load_matrix_sync(a_frag, a_ptr, hdim);
                wmma::load_matrix_sync(b_frag, b_ptr, hdim);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(scores + row_off * k_tile + n_off, c_frag, k_tile, wmma::mem_row_major);
        }
#endif
        __syncthreads();

        const unsigned int nwarps = blockDim.x >> 5u;
        for (unsigned int row = warp; row < q_block; row += nwarps) {
            const unsigned int global_q = global_q_base + row;
            const bool valid_q = global_q < total_q;
            const unsigned int visible_len = valid_q
                ? min(context_len, start_position + global_q + 1u)
                : 0u;
            const float old_m = scalars[row * 3u + 0u];
            const float old_l = scalars[row * 3u + 1u];
            const unsigned int pos = tile_start + lane;
            float score = (valid_q && lane < tile_count && pos < visible_len)
                ? scores[row * k_tile + lane] * scale
                : -3.402823466e38f;
            const float tile_m = aegis_warp_reduce_max(score);
            const float new_m = fmaxf(old_m, tile_m);
            float weight = 0.0f;
            if (score > -3.0e38f) {
                weight = exp2f((score - new_m) * log2e);
            }
            scores[row * k_tile + lane] = weight;
            const float tile_l = aegis_warp_reduce_sum(weight);
            if (lane == 0u) {
                const float alpha = old_l > 0.0f ? exp2f((old_m - new_m) * log2e) : 0.0f;
                scalars[row * 3u + 0u] = new_m;
                scalars[row * 3u + 1u] = old_l * alpha + tile_l;
                scalars[row * 3u + 2u] = alpha;
            }
        }
        __syncthreads();

        for (unsigned int idx = tid; idx < q_block * k_tile; idx += blockDim.x) {
            weights_half[idx] = __float2half_rn(scores[idx]);
        }
        __syncthreads();

#if __CUDA_ARCH__ >= 800
        if (warp < 8u) {
            using namespace nvcuda;
            const unsigned int n_off = warp * 16u;
            for (unsigned int row_off = 0u; row_off < q_block; row_off += 16u) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> pv_frag;
                wmma::fill_fragment(pv_frag, 0.0f);
#pragma unroll
                for (unsigned int kk = 0u; kk < k_tile; kk += 16u) {
                    const half* p_ptr = weights_half + row_off * k_tile + kk;
                    const half* v_ptr = reinterpret_cast<const half*>(v_shared + kk * hdim + n_off);
                    wmma::load_matrix_sync(p_frag, p_ptr, k_tile);
                    wmma::load_matrix_sync(v_frag, v_ptr, hdim);
                    wmma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);
                }
                wmma::store_matrix_sync(tile_acc + row_off * hdim + n_off, pv_frag, hdim, wmma::mem_row_major);
            }
        }
#endif
        __syncthreads();

        for (unsigned int idx = tid; idx < q_block * hdim; idx += blockDim.x) {
            const unsigned int row = idx / hdim;
            const unsigned int global_q = global_q_base + row;
            if (global_q < total_q) {
                acc[idx] = acc[idx] * scalars[row * 3u + 2u] + tile_acc[idx];
            }
        }
        __syncthreads();
    }

    for (unsigned int idx = tid; idx < q_block * hdim; idx += blockDim.x) {
        const unsigned int row = idx / hdim;
        const unsigned int dim = idx - row * hdim;
        const unsigned int global_q = global_q_base + row;
        if (global_q >= total_q) {
            continue;
        }
        const float denom = fmaxf(scalars[row * 3u + 1u], 1.0e-20f);
        output[(size_t(global_q) * num_attention_heads + head) * hdim + dim] = acc[idx] / denom;
    }
}

extern "C" __global__ void aegis_attention_prefill_dense_halfq_wmma_hdim128_split(
    const unsigned short* __restrict__ key_cache,
    const unsigned short* __restrict__ value_cache,
    const unsigned short* __restrict__ query,
    const unsigned int start_position,
    const unsigned int total_q,
    const unsigned int context_len,
    const unsigned int num_attention_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int split_tokens,
    const unsigned int split_count,
    float* __restrict__ partial_acc,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l
) {
    constexpr unsigned int hdim = 128u;
    constexpr unsigned int q_block = 16u;
    constexpr unsigned int k_tile = 32u;
    const unsigned int head = blockIdx.x;
    const unsigned int q_block_idx = blockIdx.y;
    const unsigned int split = blockIdx.z;
    const unsigned int global_q_base = q_block_idx * q_block;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5u;
    if (head_dim != hdim || head >= num_attention_heads || split >= split_count || blockDim.x < 256u) {
        return;
    }

    const size_t stats_base =
        ((size_t(q_block_idx) * num_attention_heads + head) * split_count + split) * q_block;
    const size_t acc_base = stats_base * hdim;
    const unsigned int split_start = split * split_tokens;
    const unsigned int split_end = min(context_len, split_start + split_tokens);
    const unsigned int last_q_in_block = min(total_q, global_q_base + q_block) - 1u;
    const unsigned int block_max_visible = global_q_base < total_q
        ? min(context_len, start_position + last_q_in_block + 1u)
        : 0u;
    if (block_max_visible == 0u || split_start >= split_end || split_start >= block_max_visible) {
        for (unsigned int idx = tid; idx < q_block * hdim; idx += blockDim.x) {
            partial_acc[acc_base + idx] = 0.0f;
        }
        for (unsigned int row = tid; row < q_block; row += blockDim.x) {
            partial_m[stats_base + row] = -3.402823466e38f;
            partial_l[stats_base + row] = 0.0f;
        }
        return;
    }

    extern __shared__ __align__(16) unsigned char smem[];
    unsigned short* q_shared = reinterpret_cast<unsigned short*>(smem);
    unsigned short* k_shared = q_shared + q_block * hdim;
    unsigned short* v_shared = k_shared + k_tile * hdim;
    float* scores = reinterpret_cast<float*>(v_shared + k_tile * hdim);
    float* tile_acc = scores + q_block * k_tile;
    float* acc = tile_acc + q_block * hdim;
    float* scalars = acc + q_block * hdim;
    half* weights_half = reinterpret_cast<half*>(scalars + q_block * 3u);

    const unsigned int group = num_attention_heads / num_kv_heads;
    const unsigned int kv_head = head / group;
    const float scale = rsqrtf(float(hdim));
    const float log2e = 1.4426950408889634f;

    for (unsigned int idx = tid; idx < q_block * hdim; idx += blockDim.x) {
        const unsigned int row = idx / hdim;
        const unsigned int dim = idx - row * hdim;
        const unsigned int global_q = global_q_base + row;
        q_shared[idx] = global_q < total_q
            ? query[(size_t(global_q) * num_attention_heads + head) * hdim + dim]
            : 0u;
        acc[idx] = 0.0f;
    }
    for (unsigned int row = tid; row < q_block; row += blockDim.x) {
        scalars[row * 3u + 0u] = -3.402823466e38f;
        scalars[row * 3u + 1u] = 0.0f;
        scalars[row * 3u + 2u] = 0.0f;
    }
    __syncthreads();

    const unsigned int split_visible_end = min(split_end, block_max_visible);
    for (unsigned int tile_start = split_start; tile_start < split_visible_end; tile_start += k_tile) {
        const unsigned int tile_count = min(k_tile, split_visible_end - tile_start);
        for (unsigned int idx = tid; idx < k_tile * hdim; idx += blockDim.x) {
            const unsigned int col = idx / hdim;
            const unsigned int dim = idx - col * hdim;
            const unsigned int pos = tile_start + col;
            const bool valid_k = col < tile_count;
            const size_t kv_offset =
                (size_t(pos) * num_kv_heads + kv_head) * hdim + dim;
            k_shared[idx] = valid_k ? key_cache[kv_offset] : 0u;
            v_shared[idx] = valid_k ? value_cache[kv_offset] : 0u;
        }
        __syncthreads();

#if __CUDA_ARCH__ >= 800
        if (warp < 2u) {
            using namespace nvcuda;
            const unsigned int n_off = warp * 16u;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);
#pragma unroll
            for (unsigned int kk = 0u; kk < hdim; kk += 16u) {
                const half* a_ptr = reinterpret_cast<const half*>(q_shared + kk);
                const half* b_ptr = reinterpret_cast<const half*>(k_shared + n_off * hdim + kk);
                wmma::load_matrix_sync(a_frag, a_ptr, hdim);
                wmma::load_matrix_sync(b_frag, b_ptr, hdim);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(scores + n_off, c_frag, k_tile, wmma::mem_row_major);
        }
#endif
        __syncthreads();

        const unsigned int nwarps = blockDim.x >> 5u;
        for (unsigned int row = warp; row < q_block; row += nwarps) {
            const unsigned int global_q = global_q_base + row;
            const bool valid_q = global_q < total_q;
            const unsigned int visible_len = valid_q
                ? min(context_len, start_position + global_q + 1u)
                : 0u;
            const float old_m = scalars[row * 3u + 0u];
            const float old_l = scalars[row * 3u + 1u];
            const unsigned int pos = tile_start + lane;
            float score = (valid_q && lane < tile_count && pos < visible_len)
                ? scores[row * k_tile + lane] * scale
                : -3.402823466e38f;
            const float tile_m = aegis_warp_reduce_max(score);
            const float new_m = fmaxf(old_m, tile_m);
            float weight = 0.0f;
            if (score > -3.0e38f) {
                weight = exp2f((score - new_m) * log2e);
            }
            scores[row * k_tile + lane] = weight;
            const float tile_l = aegis_warp_reduce_sum(weight);
            if (lane == 0u) {
                const float alpha = old_l > 0.0f ? exp2f((old_m - new_m) * log2e) : 0.0f;
                scalars[row * 3u + 0u] = new_m;
                scalars[row * 3u + 1u] = old_l * alpha + tile_l;
                scalars[row * 3u + 2u] = alpha;
            }
        }
        __syncthreads();

        for (unsigned int idx = tid; idx < q_block * k_tile; idx += blockDim.x) {
            weights_half[idx] = __float2half_rn(scores[idx]);
        }
        __syncthreads();

#if __CUDA_ARCH__ >= 800
        if (warp < 8u) {
            using namespace nvcuda;
            const unsigned int n_off = warp * 16u;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> pv_frag;
            wmma::fill_fragment(pv_frag, 0.0f);
#pragma unroll
            for (unsigned int kk = 0u; kk < k_tile; kk += 16u) {
                const half* p_ptr = weights_half + kk;
                const half* v_ptr = reinterpret_cast<const half*>(v_shared + kk * hdim + n_off);
                wmma::load_matrix_sync(p_frag, p_ptr, k_tile);
                wmma::load_matrix_sync(v_frag, v_ptr, hdim);
                wmma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);
            }
            wmma::store_matrix_sync(tile_acc + n_off, pv_frag, hdim, wmma::mem_row_major);
        }
#endif
        __syncthreads();

        for (unsigned int idx = tid; idx < q_block * hdim; idx += blockDim.x) {
            const unsigned int row = idx / hdim;
            const unsigned int global_q = global_q_base + row;
            if (global_q < total_q) {
                acc[idx] = acc[idx] * scalars[row * 3u + 2u] + tile_acc[idx];
            }
        }
        __syncthreads();
    }

    for (unsigned int idx = tid; idx < q_block * hdim; idx += blockDim.x) {
        partial_acc[acc_base + idx] = acc[idx];
    }
    for (unsigned int row = tid; row < q_block; row += blockDim.x) {
        partial_m[stats_base + row] = scalars[row * 3u + 0u];
        partial_l[stats_base + row] = scalars[row * 3u + 1u];
    }
}

extern "C" __global__ void aegis_attention_prefill_dense_halfq_wmma_hdim128_combine(
    const float* __restrict__ partial_acc,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const unsigned int total_q,
    const unsigned int num_attention_heads,
    const unsigned int head_dim,
    const unsigned int split_count,
    float* __restrict__ output
) {
    constexpr unsigned int hdim = 128u;
    constexpr unsigned int q_block = 16u;
    const unsigned int head = blockIdx.x;
    const unsigned int q_block_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int global_q_base = q_block_idx * q_block;
    if (head_dim != hdim || head >= num_attention_heads || global_q_base >= total_q) {
        return;
    }

    extern __shared__ float shared[];
    float* acc = shared;
    float* scalars = acc + q_block * hdim;
    for (unsigned int idx = tid; idx < q_block * hdim; idx += blockDim.x) {
        acc[idx] = 0.0f;
    }
    for (unsigned int row = tid; row < q_block; row += blockDim.x) {
        scalars[row * 3u + 0u] = -3.402823466e38f;
        scalars[row * 3u + 1u] = 0.0f;
        scalars[row * 3u + 2u] = 1.0f;
    }
    __syncthreads();

    const float log2e = 1.4426950408889634f;
    for (unsigned int split = 0u; split < split_count; ++split) {
        const size_t stats_base =
            ((size_t(q_block_idx) * num_attention_heads + head) * split_count + split) * q_block;
        if (tid == 0u) {
#pragma unroll
            for (unsigned int row = 0u; row < q_block; ++row) {
                const float local_l = partial_l[stats_base + row];
                if (local_l > 0.0f) {
                    const float local_m = partial_m[stats_base + row];
                    const float old_m = scalars[row * 3u + 0u];
                    const float old_l = scalars[row * 3u + 1u];
                    const float new_m = fmaxf(old_m, local_m);
                    const float alpha = old_l > 0.0f ? exp2f((old_m - new_m) * log2e) : 0.0f;
                    const float beta = exp2f((local_m - new_m) * log2e);
                    scalars[row * 3u + 0u] = new_m;
                    scalars[row * 3u + 1u] = old_l * alpha + local_l * beta;
                    scalars[row * 3u + 2u] = alpha;
                } else {
                    scalars[row * 3u + 2u] = 1.0f;
                }
            }
        }
        __syncthreads();

        for (unsigned int row = 0u; row < q_block; ++row) {
            const float local_l = partial_l[stats_base + row];
            if (local_l <= 0.0f) {
                continue;
            }
            const float alpha = scalars[row * 3u + 2u];
            const float local_m = partial_m[stats_base + row];
            const float global_m = scalars[row * 3u + 0u];
            const float beta = exp2f((local_m - global_m) * log2e);
            const size_t base = stats_base * hdim + size_t(row) * hdim;
            for (unsigned int dim = tid; dim < hdim; dim += blockDim.x) {
                const unsigned int idx = row * hdim + dim;
                acc[idx] = acc[idx] * alpha + partial_acc[base + dim] * beta;
            }
        }
        __syncthreads();
    }

    for (unsigned int idx = tid; idx < q_block * hdim; idx += blockDim.x) {
        const unsigned int row = idx / hdim;
        const unsigned int dim = idx - row * hdim;
        const unsigned int global_q = global_q_base + row;
        if (global_q >= total_q) {
            continue;
        }
        const float denom = fmaxf(scalars[row * 3u + 1u], 1.0e-20f);
        output[(size_t(global_q) * num_attention_heads + head) * hdim + dim] = acc[idx] / denom;
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

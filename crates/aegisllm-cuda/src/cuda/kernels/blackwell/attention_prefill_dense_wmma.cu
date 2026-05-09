
// Ring-buffer slot lookup: maps an absolute token position to its slot in the
// KV cache. For sliding-window layers the cache is sized to `window_size`
// (`cache_capacity > 0`) and the slot wraps; for global layers the cache is
// sized to the full context (callers pass `cache_capacity == context_size`)
// and the modulo collapses to the identity for any `pos < context_size`.
__device__ __forceinline__ unsigned int kv_slot(unsigned int pos, unsigned int cache_capacity) {
    return (cache_capacity > 0u) ? (pos % cache_capacity) : pos;
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
    const unsigned int cache_capacity,
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
        const unsigned short* k = key_cache + (size_t(kv_slot(pos, cache_capacity)) * num_kv_heads + kv_head) * head_dim;
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
            const unsigned short* v = value_cache + (size_t(kv_slot(pos, cache_capacity)) * num_kv_heads + kv_head) * head_dim;
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
    const unsigned int cache_capacity,
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
                key_cache + (size_t(kv_slot(pos, cache_capacity)) * num_kv_heads + kv_head) * head_dim;
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
                    value_cache + (size_t(kv_slot(pos, cache_capacity)) * num_kv_heads + kv_head) * head_dim;
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
    const unsigned int cache_capacity,
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
            key_cache + (size_t(kv_slot(pos, cache_capacity)) * num_kv_heads + kv_head) * head_dim;
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
            key_cache + (size_t(kv_slot(pos, cache_capacity)) * num_kv_heads + kv_head) * head_dim;
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
            value_cache + (size_t(kv_slot(pos, cache_capacity)) * num_kv_heads + kv_head) * head_dim;
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
    const unsigned int cache_capacity,
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
            key_cache + (size_t(kv_slot(pos, cache_capacity)) * num_kv_heads + kv_head) * head_dim;
        const unsigned short* v =
            value_cache + (size_t(kv_slot(pos, cache_capacity)) * num_kv_heads + kv_head) * head_dim;
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
    const unsigned int cache_capacity,
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

    const uint4 zero_vec = make_uint4(0u, 0u, 0u, 0u);
    for (unsigned int tile_start = 0u; tile_start < block_max_visible; tile_start += k_tile) {
        const unsigned int tile_count = min(k_tile, block_max_visible - tile_start);
        constexpr unsigned int halfs_per_vec = sizeof(uint4) / sizeof(unsigned short);
        constexpr unsigned int kv_vecs = k_tile * hdim / halfs_per_vec;
        uint4* k_shared_vec = reinterpret_cast<uint4*>(k_shared);
        uint4* v_shared_vec = reinterpret_cast<uint4*>(v_shared);
        for (unsigned int vec = tid; vec < kv_vecs; vec += blockDim.x) {
            const unsigned int idx = vec * halfs_per_vec;
            const unsigned int col = idx / hdim;
            const unsigned int dim = idx - col * hdim;
            const unsigned int pos = tile_start + col;
            const bool valid_k = col < tile_count;
            const size_t kv_offset =
                (size_t(kv_slot(pos, cache_capacity)) * num_kv_heads + kv_head) * hdim + dim;
            k_shared_vec[vec] = valid_k
                ? *reinterpret_cast<const uint4*>(key_cache + kv_offset)
                : zero_vec;
            v_shared_vec[vec] = valid_k
                ? *reinterpret_cast<const uint4*>(value_cache + kv_offset)
                : zero_vec;
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

// Templated implementation of the WMMA-based dense prefill attention
// kernel. The compile-time `HDIM` parameter parametrises:
//   * shared-memory tile sizes (Q/K/V/scores/acc all scale with HDIM)
//   * the inner Q*K WMMA K-loop (HDIM/16 mma.sync calls per K-tile)
//   * the output P*V WMMA work distribution (HDIM/16 warps cooperate
//     on a 16×HDIM output tile)
//   * the launch constraint: blockDim.x ≥ (HDIM/16)*32
//
// `extern "C"` instantiations below dispatch by head_dim. HDIM ∈ {128,
// 256} go through this generic template. HDIM=512 (Gemma-4 global
// layers) has a separate bespoke kernel — `..._hdim512` below — that
// uses K_TILE=16 and drops the tile_acc double-buffer to fit the
// sm_120 100 KiB shared-mem cap.
// `window_size = 0` means full causal attention (no sliding-window
// clamp). `window_size > 0` (Gemma-4 sliding layers: 1024) clamps the
// K-tile loop to `[max(0, q_pos - window + 1), q_pos]`. For long
// contexts on sliding layers this turns O(seq²) into O(seq * window).
template<unsigned int HDIM>
__device__ __forceinline__ void aegis_attention_prefill_dense_halfq_wmma_impl(
    const unsigned short* __restrict__ key_cache,
    const unsigned short* __restrict__ value_cache,
    const unsigned short* __restrict__ query,
    const unsigned int start_position,
    const unsigned int total_q,
    const unsigned int context_len,
    const unsigned int num_attention_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int cache_capacity,
    const unsigned int window_size,
    float* __restrict__ output
) {
    static_assert(HDIM % 16u == 0u, "HDIM must be a multiple of WMMA tile size 16");
    constexpr unsigned int hdim = HDIM;
    constexpr unsigned int q_block = 16u;
    constexpr unsigned int k_tile = 32u;
    constexpr unsigned int output_warps = HDIM / 16u;
    const unsigned int head = blockIdx.x;
    const unsigned int global_q_base = blockIdx.y * q_block;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5u;
    if (head_dim != hdim || head >= num_attention_heads || blockDim.x < output_warps * 32u) {
        return;
    }

    const unsigned int last_q_in_block = min(total_q, global_q_base + q_block) - 1u;
    const unsigned int block_max_visible = global_q_base < total_q
        ? min(context_len, start_position + last_q_in_block + 1u)
        : 0u;
    if (block_max_visible == 0u) {
        return;
    }
    // Sliding-window lower bound for the earliest query in this block
    // (global_q = global_q_base). Earliest visible K position is
    // `q_pos - window + 1`. Round down to a k_tile boundary so the
    // K-loop still aligns with WMMA tile cadence.
    // window_size == 0 means full attention (no clamp).
    const unsigned int block_min_visible_raw = (window_size > 0u
        && start_position + global_q_base + 1u > window_size)
        ? (start_position + global_q_base + 1u - window_size)
        : 0u;
    const unsigned int block_min_tile_start =
        (block_min_visible_raw / k_tile) * k_tile;

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

    const uint4 zero_vec = make_uint4(0u, 0u, 0u, 0u);
    for (unsigned int tile_start = block_min_tile_start; tile_start < block_max_visible; tile_start += k_tile) {
        const unsigned int tile_count = min(k_tile, block_max_visible - tile_start);
        constexpr unsigned int halfs_per_vec = sizeof(uint4) / sizeof(unsigned short);
        constexpr unsigned int kv_vecs = k_tile * hdim / halfs_per_vec;
        uint4* k_shared_vec = reinterpret_cast<uint4*>(k_shared);
        uint4* v_shared_vec = reinterpret_cast<uint4*>(v_shared);
        for (unsigned int vec = tid; vec < kv_vecs; vec += blockDim.x) {
            const unsigned int idx = vec * halfs_per_vec;
            const unsigned int col = idx / hdim;
            const unsigned int dim = idx - col * hdim;
            const unsigned int pos = tile_start + col;
            const bool valid_k = col < tile_count;
            const size_t kv_offset =
                (size_t(kv_slot(pos, cache_capacity)) * num_kv_heads + kv_head) * hdim + dim;
            k_shared_vec[vec] = valid_k
                ? *reinterpret_cast<const uint4*>(key_cache + kv_offset)
                : zero_vec;
            v_shared_vec[vec] = valid_k
                ? *reinterpret_cast<const uint4*>(value_cache + kv_offset)
                : zero_vec;
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
            const unsigned int row_min_visible = (window_size > 0u && start_position + global_q + 1u > window_size)
                ? (start_position + global_q + 1u - window_size) : 0u;
            const float old_m = scalars[row * 3u + 0u];
            const float old_l = scalars[row * 3u + 1u];
            const unsigned int pos = tile_start + lane;
            float score = (valid_q && lane < tile_count && pos < visible_len && pos >= row_min_visible)
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
        if (warp < output_warps) {
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

// `extern "C"` instantiations consumed by the Rust dispatcher. Each
// targets a specific architectural head_dim; selection happens in
// `attention_prefill_dense_compat_device` based on the layer's metadata.
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
    const unsigned int cache_capacity,
    const unsigned int window_size,
    float* __restrict__ output
) {
    aegis_attention_prefill_dense_halfq_wmma_impl<128u>(
        key_cache, value_cache, query, start_position, total_q, context_len,
        num_attention_heads, num_kv_heads, head_dim, cache_capacity, window_size, output
    );
}

extern "C" __global__ void aegis_attention_prefill_dense_halfq_wmma_hdim256(
    const unsigned short* __restrict__ key_cache,
    const unsigned short* __restrict__ value_cache,
    const unsigned short* __restrict__ query,
    const unsigned int start_position,
    const unsigned int total_q,
    const unsigned int context_len,
    const unsigned int num_attention_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int cache_capacity,
    const unsigned int window_size,
    float* __restrict__ output
) {
    aegis_attention_prefill_dense_halfq_wmma_impl<256u>(
        key_cache, value_cache, query, start_position, total_q, context_len,
        num_attention_heads, num_kv_heads, head_dim, cache_capacity, window_size, output
    );
}

// HDIM=512 specialised implementation (Gemma-4 global-attention layers).
//
// Shared-memory pressure is the binding constraint at HDIM=512. With the
// generic `aegis_attention_prefill_dense_halfq_wmma_impl<512>` shape
// (q_block=16, k_tile=32) we'd need ~150 KiB per block, well above the
// 100 KiB sm_120 cap. Two changes shrink it to ~83 KiB:
//   1. K_TILE = 16 (one 16x16 Q*K WMMA tile per iteration instead of two).
//   2. Drop the `tile_acc` double-buffer entirely. Instead we rescale the
//      running `acc` in shared memory by the per-row alpha BEFORE the P*V
//      WMMA, then load `acc` as the WMMA accumulator's initial value and
//      let `mma_sync` fuse the new tile contribution. After the WMMA we
//      store back to `acc`. This saves q_block * hdim * sizeof(float) =
//      32 KiB at HDIM=512.
//
// Block size = (HDIM/16) * 32 = 1024 threads (the sm_120 max). For the
// Q*K stage only `warp < K_TILE/16 = 1` warp participates; for the P*V
// stage all 32 warps cooperate (one per 16-col output slice).
//
// `window_size = 0` (full causal) is the expected path for Gemma-4 global
// layers; the same `block_min_tile_start` logic as the generic template
// is preserved so a non-zero window would still work if a future model
// uses sliding+hdim=512.
extern "C" __global__ void aegis_attention_prefill_dense_halfq_wmma_hdim512(
    const unsigned short* __restrict__ key_cache,
    const unsigned short* __restrict__ value_cache,
    const unsigned short* __restrict__ query,
    const unsigned int start_position,
    const unsigned int total_q,
    const unsigned int context_len,
    const unsigned int num_attention_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int cache_capacity,
    const unsigned int window_size,
    float* __restrict__ output
) {
    constexpr unsigned int hdim = 512u;
    constexpr unsigned int q_block = 16u;
    constexpr unsigned int k_tile = 16u;
    constexpr unsigned int output_warps = hdim / 16u;  // 32
    const unsigned int head = blockIdx.x;
    const unsigned int global_q_base = blockIdx.y * q_block;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5u;
    if (head_dim != hdim || head >= num_attention_heads || blockDim.x < output_warps * 32u) {
        return;
    }

    const unsigned int last_q_in_block = min(total_q, global_q_base + q_block) - 1u;
    const unsigned int block_max_visible = global_q_base < total_q
        ? min(context_len, start_position + last_q_in_block + 1u)
        : 0u;
    if (block_max_visible == 0u) {
        return;
    }
    const unsigned int block_min_visible_raw = (window_size > 0u
        && start_position + global_q_base + 1u > window_size)
        ? (start_position + global_q_base + 1u - window_size)
        : 0u;
    const unsigned int block_min_tile_start =
        (block_min_visible_raw / k_tile) * k_tile;

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

    const uint4 zero_vec = make_uint4(0u, 0u, 0u, 0u);
    for (unsigned int tile_start = block_min_tile_start; tile_start < block_max_visible; tile_start += k_tile) {
        const unsigned int tile_count = min(k_tile, block_max_visible - tile_start);
        constexpr unsigned int halfs_per_vec = sizeof(uint4) / sizeof(unsigned short);
        constexpr unsigned int kv_vecs = k_tile * hdim / halfs_per_vec;
        uint4* k_shared_vec = reinterpret_cast<uint4*>(k_shared);
        uint4* v_shared_vec = reinterpret_cast<uint4*>(v_shared);
        for (unsigned int vec = tid; vec < kv_vecs; vec += blockDim.x) {
            const unsigned int idx = vec * halfs_per_vec;
            const unsigned int col = idx / hdim;
            const unsigned int dim = idx - col * hdim;
            const unsigned int pos = tile_start + col;
            const bool valid_k = col < tile_count;
            const size_t kv_offset =
                (size_t(kv_slot(pos, cache_capacity)) * num_kv_heads + kv_head) * hdim + dim;
            k_shared_vec[vec] = valid_k
                ? *reinterpret_cast<const uint4*>(key_cache + kv_offset)
                : zero_vec;
            v_shared_vec[vec] = valid_k
                ? *reinterpret_cast<const uint4*>(value_cache + kv_offset)
                : zero_vec;
        }
        __syncthreads();

#if __CUDA_ARCH__ >= 800
        // Q*K WMMA: k_tile=16 means a single 16x16 tile suffices, only
        // warp 0 participates. The K-loop walks the full HDIM via 16-wide
        // sub-tiles (HDIM/16 = 32 mma.sync calls).
        if (warp < 1u) {
            using namespace nvcuda;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);
#pragma unroll
            for (unsigned int kk = 0u; kk < hdim; kk += 16u) {
                const half* a_ptr = reinterpret_cast<const half*>(q_shared + kk);
                const half* b_ptr = reinterpret_cast<const half*>(k_shared + kk);
                wmma::load_matrix_sync(a_frag, a_ptr, hdim);
                wmma::load_matrix_sync(b_frag, b_ptr, hdim);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(scores, c_frag, k_tile, wmma::mem_row_major);
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
            const unsigned int row_min_visible = (window_size > 0u && start_position + global_q + 1u > window_size)
                ? (start_position + global_q + 1u - window_size) : 0u;
            const float old_m = scalars[row * 3u + 0u];
            const float old_l = scalars[row * 3u + 1u];
            // k_tile=16: only the lower half of each warp (lanes 0..15)
            // carries a real score. Lanes 16..31 contribute -inf to the
            // warp-wide reduce so they don't bias the row max/sum. The
            // ternary's true branch is only evaluated when `lane < k_tile`,
            // so the score load is safe.
            const unsigned int pos = tile_start + lane;
            float score = (valid_q && lane < k_tile && lane < tile_count && pos < visible_len && pos >= row_min_visible)
                ? scores[row * k_tile + lane] * scale
                : -3.402823466e38f;
            const float tile_m = aegis_warp_reduce_max(score);
            const float new_m = fmaxf(old_m, tile_m);
            float weight = 0.0f;
            if (score > -3.0e38f) {
                weight = exp2f((score - new_m) * log2e);
            }
            // Only the active lanes (< k_tile) write back; upper lanes
            // would clobber neighboring rows' scratch.
            if (lane < k_tile) {
                scores[row * k_tile + lane] = weight;
            }
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
        // Rescale running `acc` by per-row alpha BEFORE P*V WMMA. This is
        // the trick that lets us drop the tile_acc double-buffer and fit
        // in 96 KiB of shared memory at hdim=512.
        for (unsigned int idx = tid; idx < q_block * hdim; idx += blockDim.x) {
            const unsigned int row = idx / hdim;
            const unsigned int global_q = global_q_base + row;
            if (global_q < total_q) {
                acc[idx] = acc[idx] * scalars[row * 3u + 2u];
            }
        }
        __syncthreads();

#if __CUDA_ARCH__ >= 800
        // P*V WMMA with the running `acc` as accumulator-init. After the
        // mma loop, the fragment holds (acc * alpha) + (P @ V_tile),
        // which we store back to `acc`. K_TILE=16 means a single
        // 16x16 P fragment fully covers the column space.
        if (warp < output_warps) {
            using namespace nvcuda;
            const unsigned int n_off = warp * 16u;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> pv_frag;
            // Load existing rescaled `acc` slice as the initial accumulator.
            wmma::load_matrix_sync(pv_frag, acc + n_off, hdim, wmma::mem_row_major);
            // P[16x16] @ V[16x16] (k=16; single mma covers full k_tile).
            const half* p_ptr = weights_half;
            const half* v_ptr = reinterpret_cast<const half*>(v_shared + n_off);
            wmma::load_matrix_sync(p_frag, p_ptr, k_tile);
            wmma::load_matrix_sync(v_frag, v_ptr, hdim);
            wmma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);
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

// HDIM=512 register-resident-accumulator variant (Round 2 optimisation).
//
// Goal: lift block-per-SM residency from 1 to 2 by shrinking shared
// memory below ~50 KiB (sm_120 has 100 KiB shmem / SM).
//
// Differences vs the baseline `..._hdim512` kernel:
//   * Block size is 512 threads (16 warps) rather than 1024 (32 warps).
//     Each warp owns TWO 16-col output slices (cols w*32..w*32+32),
//     held as TWO persistent WMMA accumulator fragments in registers.
//     Register-resident acc removes the 16*512*4 = 32 KiB `acc` buffer
//     from shared memory entirely.
//   * For the per-row alpha rescale (which must touch every accumulator
//     element, but the wmma fragment lane→element mapping is opaque /
//     implementation-defined), each warp uses a SHARED scratch slot of
//     16*16*4 = 1 KiB to round-trip ONE c_frag at a time through shmem.
//     The scratch space overlays `k_shared` (which has been consumed by
//     the time we reach the rescale step). 16 warps * 1 KiB = 16 KiB,
//     well within the 16 KiB k_shared region.
//   * Q*K stage still runs on warp 0 only (k_tile=16 fits a single
//     16x16 wmma tile). Softmax distributes 16 rows across 16 warps
//     (one row per warp — perfect partition).
//
// Shared-memory budget (no acc buffer):
//   q_shared 16 KiB + k_shared 16 KiB + v_shared 16 KiB
//   + scores 1 KiB + scalars 192 B + weights_half 512 B
//   ≈ 49.7 KiB.
//
// Register pressure: 2 persistent f32 c_frag (8 floats/lane each)
//   = 16 regs/lane for acc, plus a_frag/b_frag/pv-temp working set.
//   At 512 threads * ~64 regs/thread = 32 KiB regs/block, two blocks
//   need ~64 KiB out of the 64 KiB / SM register file — tight but
//   feasible. We pin the per-thread register count via __launch_bounds__
//   so the compiler doesn't spill us out of 2-block residency.
//
// Numerical correctness preserved: same online-softmax math as the
// baseline kernel; the only added round-trip is store→multiply→load
// for the rescale, which is the same alpha multiply the baseline does
// (just on a smaller per-warp scratch instead of the global acc).
extern "C" __global__
__launch_bounds__(512, 2)
void aegis_attention_prefill_dense_halfq_wmma_hdim512_regacc(
    const unsigned short* __restrict__ key_cache,
    const unsigned short* __restrict__ value_cache,
    const unsigned short* __restrict__ query,
    const unsigned int start_position,
    const unsigned int total_q,
    const unsigned int context_len,
    const unsigned int num_attention_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int cache_capacity,
    const unsigned int window_size,
    float* __restrict__ output
) {
    constexpr unsigned int hdim = 512u;
    constexpr unsigned int q_block = 16u;
    constexpr unsigned int k_tile = 16u;
    constexpr unsigned int warps_per_block = 16u;       // 512 threads / 32
    constexpr unsigned int cols_per_warp = hdim / warps_per_block; // 32
    constexpr unsigned int frags_per_warp = cols_per_warp / 16u;   // 2
    const unsigned int head = blockIdx.x;
    const unsigned int global_q_base = blockIdx.y * q_block;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5u;
    if (head_dim != hdim || head >= num_attention_heads || blockDim.x < warps_per_block * 32u) {
        return;
    }

    const unsigned int last_q_in_block = min(total_q, global_q_base + q_block) - 1u;
    const unsigned int block_max_visible = global_q_base < total_q
        ? min(context_len, start_position + last_q_in_block + 1u)
        : 0u;
    if (block_max_visible == 0u) {
        return;
    }
    const unsigned int block_min_visible_raw = (window_size > 0u
        && start_position + global_q_base + 1u > window_size)
        ? (start_position + global_q_base + 1u - window_size)
        : 0u;
    const unsigned int block_min_tile_start =
        (block_min_visible_raw / k_tile) * k_tile;

    extern __shared__ __align__(16) unsigned char smem[];
    unsigned short* q_shared = reinterpret_cast<unsigned short*>(smem);
    unsigned short* k_shared = q_shared + q_block * hdim;
    unsigned short* v_shared = k_shared + k_tile * hdim;
    float* scores = reinterpret_cast<float*>(v_shared + k_tile * hdim);
    float* scalars = scores + q_block * k_tile;
    half* weights_half = reinterpret_cast<half*>(scalars + q_block * 3u);
    // Per-warp 16x16 = 1 KiB scratch reused for the alpha rescale of one
    // c_frag at a time. Overlays k_shared (which has been consumed by
    // the time we reach the rescale step). 16 warps * 1 KiB = 16 KiB,
    // and k_shared is 16 KiB (k_tile=16 * hdim=512 * 2 B). The base
    // pointer is identical, just reinterpreted as f32.
    float* acc_scratch = reinterpret_cast<float*>(k_shared);
    constexpr unsigned int acc_scratch_per_warp = 16u * 16u; // 256 floats per warp

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
    }
    for (unsigned int row = tid; row < q_block; row += blockDim.x) {
        scalars[row * 3u + 0u] = -3.402823466e38f;
        scalars[row * 3u + 1u] = 0.0f;
        scalars[row * 3u + 2u] = 0.0f;
    }
    __syncthreads();

#if __CUDA_ARCH__ >= 800
    using namespace nvcuda;
    // Persistent register-resident output accumulator: each warp owns
    // `frags_per_warp` (=2) WMMA fragments covering its 32-col slice.
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[frags_per_warp];
    wmma::fill_fragment(acc_frag[0], 0.0f);
    wmma::fill_fragment(acc_frag[1], 0.0f);
#endif

    const uint4 zero_vec = make_uint4(0u, 0u, 0u, 0u);
    for (unsigned int tile_start = block_min_tile_start; tile_start < block_max_visible; tile_start += k_tile) {
        const unsigned int tile_count = min(k_tile, block_max_visible - tile_start);
        constexpr unsigned int halfs_per_vec = sizeof(uint4) / sizeof(unsigned short);
        constexpr unsigned int kv_vecs = k_tile * hdim / halfs_per_vec;
        uint4* k_shared_vec = reinterpret_cast<uint4*>(k_shared);
        uint4* v_shared_vec = reinterpret_cast<uint4*>(v_shared);
        for (unsigned int vec = tid; vec < kv_vecs; vec += blockDim.x) {
            const unsigned int idx = vec * halfs_per_vec;
            const unsigned int col = idx / hdim;
            const unsigned int dim = idx - col * hdim;
            const unsigned int pos = tile_start + col;
            const bool valid_k = col < tile_count;
            const size_t kv_offset =
                (size_t(kv_slot(pos, cache_capacity)) * num_kv_heads + kv_head) * hdim + dim;
            k_shared_vec[vec] = valid_k
                ? *reinterpret_cast<const uint4*>(key_cache + kv_offset)
                : zero_vec;
            v_shared_vec[vec] = valid_k
                ? *reinterpret_cast<const uint4*>(value_cache + kv_offset)
                : zero_vec;
        }
        __syncthreads();

#if __CUDA_ARCH__ >= 800
        // Q*K WMMA: split HDIM=512 reduction across `qk_warps` warps so
        // 15/16 warps don't sit idle. Each of the first `qk_warps` warps
        // accumulates an independent 16x16 partial covering its slice of
        // HDIM; results are summed elementwise into `scores` afterwards.
        // Partial slots overlay k_shared (which is no longer needed once
        // all warp mma chains have read it for the Q*K reduction). 4
        // partials × 256 floats × 4 B = 4 KiB ≪ k_shared's 16 KiB.
        constexpr unsigned int qk_warps = 4u;
        constexpr unsigned int hdim_per_qk_warp = hdim / qk_warps;  // 128
        float* partial_scores = reinterpret_cast<float*>(k_shared);
        if (warp < qk_warps) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);
            const unsigned int kk_start = warp * hdim_per_qk_warp;
            const unsigned int kk_end   = kk_start + hdim_per_qk_warp;
#pragma unroll
            for (unsigned int kk = kk_start; kk < kk_end; kk += 16u) {
                const half* a_ptr = reinterpret_cast<const half*>(q_shared + kk);
                const half* b_ptr = reinterpret_cast<const half*>(k_shared + kk);
                wmma::load_matrix_sync(a_frag, a_ptr, hdim);
                wmma::load_matrix_sync(b_frag, b_ptr, hdim);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            // Sync so all qk_warps finish reading k_shared before any
            // warp writes partials into the same buffer (overlay).
            __syncthreads();
            wmma::store_matrix_sync(
                partial_scores + warp * 256u, c_frag, 16u, wmma::mem_row_major);
        } else {
            __syncthreads();
        }
        __syncthreads();
        // Reduce 4 partials (16x16 each = 256 floats) → scores. 256 elems
        // distributed across blockDim.x threads; idle threads at tail.
        for (unsigned int e = tid; e < 256u; e += blockDim.x) {
            scores[e] = partial_scores[0u * 256u + e]
                      + partial_scores[1u * 256u + e]
                      + partial_scores[2u * 256u + e]
                      + partial_scores[3u * 256u + e];
        }
#endif
        __syncthreads();

        // Softmax / online stats. We have exactly 16 warps and q_block=16
        // rows: each warp handles a single row (no inner stride loop).
        if (warp < q_block) {
            const unsigned int row = warp;
            const unsigned int global_q = global_q_base + row;
            const bool valid_q = global_q < total_q;
            const unsigned int visible_len = valid_q
                ? min(context_len, start_position + global_q + 1u)
                : 0u;
            const unsigned int row_min_visible = (window_size > 0u && start_position + global_q + 1u > window_size)
                ? (start_position + global_q + 1u - window_size) : 0u;
            const float old_m = scalars[row * 3u + 0u];
            const float old_l = scalars[row * 3u + 1u];
            // k_tile=16: only the lower half of each warp (lanes 0..15)
            // carries a real score. Lanes 16..31 contribute -inf to the
            // warp-wide reduce so they don't bias the row max/sum.
            const unsigned int pos = tile_start + lane;
            float score = (valid_q && lane < k_tile && lane < tile_count && pos < visible_len && pos >= row_min_visible)
                ? scores[row * k_tile + lane] * scale
                : -3.402823466e38f;
            const float tile_m = aegis_warp_reduce_max(score);
            const float new_m = fmaxf(old_m, tile_m);
            float weight = 0.0f;
            if (score > -3.0e38f) {
                weight = exp2f((score - new_m) * log2e);
            }
            if (lane < k_tile) {
                scores[row * k_tile + lane] = weight;
            }
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
        // Alpha rescale + P*V WMMA. For each of the warp's `frags_per_warp`
        // c_frags:
        //   1. Store frag to per-warp scratch (1 KiB, overlays k_shared).
        //   2. Multiply each scratch element by alpha[row].
        //   3. Reload into c_frag.
        //   4. P[16x16] @ V[16x16] (single mma covers the full k_tile).
        // The scratch round-trip is the price for not knowing the wmma
        // lane→(row,col) mapping at the C++ level. It's still cheap:
        // 256 f32 stores + 256 f32 loads per warp per K-iter.
        //
        // NOTE: k_shared has been consumed by the Q*K stage and the
        // softmax has already converted scores. Reusing k_shared as
        // acc_scratch is safe.
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;
        // Load P once per warp; identical for both column slices.
        wmma::load_matrix_sync(p_frag, weights_half, k_tile);
        float* warp_scratch = acc_scratch + warp * acc_scratch_per_warp;
#pragma unroll
        for (unsigned int f = 0u; f < frags_per_warp; ++f) {
            const unsigned int n_off = warp * cols_per_warp + f * 16u;
            // Step 1: store c_frag to warp-private scratch.
            wmma::store_matrix_sync(warp_scratch, acc_frag[f], 16u, wmma::mem_row_major);
            // Step 2: rescale by per-row alpha. 256 elements / 32 lanes
            // = 8 elements per lane. Stride elements by 32 (lane).
#pragma unroll
            for (unsigned int e = lane; e < 256u; e += 32u) {
                const unsigned int row = e / 16u;
                warp_scratch[e] *= scalars[row * 3u + 2u];
            }
            // Step 3: reload rescaled values back into the accumulator
            // fragment. This serves as the seed for the upcoming
            // mma_sync (acc + P*V).
            wmma::load_matrix_sync(acc_frag[f], warp_scratch, 16u, wmma::mem_row_major);
            // Step 4: load V slice and mma-fuse-add into acc_frag.
            const half* v_ptr = reinterpret_cast<const half*>(v_shared + n_off);
            wmma::load_matrix_sync(v_frag, v_ptr, hdim);
            wmma::mma_sync(acc_frag[f], p_frag, v_frag, acc_frag[f]);
        }
#endif
        __syncthreads();
    }

    // Final epilogue: divide by row-sum and write to global output.
    // We need each warp to write its 32-col slice of every row. The
    // wmma store puts data in row-major scratch; we then divide by
    // denom and write to global memory.
#if __CUDA_ARCH__ >= 800
    float* warp_scratch = acc_scratch + warp * acc_scratch_per_warp;
#pragma unroll
    for (unsigned int f = 0u; f < frags_per_warp; ++f) {
        const unsigned int n_off = warp * cols_per_warp + f * 16u;
        wmma::store_matrix_sync(warp_scratch, acc_frag[f], 16u, wmma::mem_row_major);
        // Each warp has 32 lanes; 16x16 = 256 elements; 8 per lane.
#pragma unroll
        for (unsigned int e = lane; e < 256u; e += 32u) {
            const unsigned int row = e / 16u;
            const unsigned int col = e - row * 16u;
            const unsigned int global_q = global_q_base + row;
            if (global_q >= total_q) {
                continue;
            }
            const float denom = fmaxf(scalars[row * 3u + 1u], 1.0e-20f);
            const unsigned int dim = n_off + col;
            output[(size_t(global_q) * num_attention_heads + head) * hdim + dim] =
                warp_scratch[e] / denom;
        }
    }
#endif
}

// =============================================================================
// HDIM=512 register-resident-accumulator twin with Q_BLOCK=32 (was 16).
// =============================================================================
//
// Goal: halve the K/V HBM bandwidth per output token at long context. With
// q_block=16 each K/V tile (32 KiB total per K-iter) is reused across 16 query
// rows. Doubling q_block to 32 means each K/V tile is reused across 32 query
// rows → ~2x reduction in K/V global-memory traffic per output token.
//
// At 38.4k context the parent kernel had attention = 51% of prefill stage
// time and per-token attention scales linearly with sequence length (full
// causal). Halving K/V bandwidth is a direct attack on the bottleneck.
//
// Structural changes vs `..._hdim512_regacc`:
//   * `q_block`               16 → 32
//   * `q_shared`              16 KiB → 32 KiB
//   * `scores`                 1 KiB →  2 KiB     (q_block * k_tile floats)
//   * `scalars`              48 B → 96 B          (q_block * 3 floats)
//   * `weights_half`           0.5 KiB → 1 KiB    (q_block * k_tile halfs)
//   * Per-warp persistent c_frags: 2 → 4
//       Each warp owns a 32-col slice spanning 2 row-strips x 2 col-strips.
//       Row-strip 0: rows  0..15, cols [warp*32 .. warp*32+15]  → frag[0]
//       Row-strip 0: rows  0..15, cols [warp*32+16 .. warp*32+31] → frag[1]
//       Row-strip 1: rows 16..31, cols [warp*32 .. warp*32+15]  → frag[2]
//       Row-strip 1: rows 16..31, cols [warp*32+16 .. warp*32+31] → frag[3]
//   * Q*K WMMA: still 4 warps split across HDIM, but each warp now produces
//     a 32x16 partial (= two 16x16 mma chains, one per row-strip) instead of
//     a single 16x16. 4 warps * 32x16 floats * 4 B = 8 KiB partial scratch,
//     still ≪ 16 KiB k_shared overlay region.
//   * Softmax: 32 rows distributed across 16 warps → each warp handles 2 rows
//     (`for (row = warp; row < q_block; row += warps_per_block)`).
//   * P*V: each warp's 4 c_frags rescaled and mma'd with V. Same scratch
//     round-trip pattern as the q_block=16 twin. 16 warps * 1 KiB scratch =
//     16 KiB, still overlays k_shared (16 KiB).
//   * Block grid: blockIdx.y covers ceil(total_q / 32).
//   * `__launch_bounds__(512, 1)` — accept 1 block/SM (vs 2 in q_block=16).
//
// Shared-mem layout (total ≤ 96 KiB sm_120 dynamic-shared cap):
//   q_shared       = 32 * 512 * 2  =     32 KiB
//   k_shared       = 16 * 512 * 2  =     16 KiB  (reused as acc_scratch)
//   v_shared       = 16 * 512 * 2  =     16 KiB
//   scores         = 32 * 16  * 4  =      2 KiB
//   scalars        = 32 *  3  * 4  =    384  B
//   weights_half   = 32 * 16  * 2  =      1 KiB
//                                  --------
//                                  ~67.5 KiB  (well under 96 KiB)
//
// Numerical correctness: same online softmax math, same per-element
// accumulation order within each c_frag. Results may bit-differ vs the
// q_block=16 twin (different reduction tree across warps for Q*K) but the
// algorithm is equivalent.

extern "C" __global__
__launch_bounds__(512, 1)
void aegis_attention_prefill_dense_halfq_wmma_hdim512_q32_regacc(
    const unsigned short* __restrict__ key_cache,
    const unsigned short* __restrict__ value_cache,
    const unsigned short* __restrict__ query,
    const unsigned int start_position,
    const unsigned int total_q,
    const unsigned int context_len,
    const unsigned int num_attention_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int cache_capacity,
    const unsigned int window_size,
    float* __restrict__ output
) {
    constexpr unsigned int hdim = 512u;
    constexpr unsigned int q_block = 32u;
    constexpr unsigned int k_tile = 16u;
    constexpr unsigned int warps_per_block = 16u;       // 512 threads / 32
    constexpr unsigned int cols_per_warp = hdim / warps_per_block; // 32
    constexpr unsigned int frags_per_warp_col = cols_per_warp / 16u; // 2
    constexpr unsigned int row_strips = q_block / 16u;               // 2
    constexpr unsigned int frags_per_warp = row_strips * frags_per_warp_col; // 4
    const unsigned int head = blockIdx.x;
    const unsigned int global_q_base = blockIdx.y * q_block;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5u;
    if (head_dim != hdim || head >= num_attention_heads || blockDim.x < warps_per_block * 32u) {
        return;
    }

    const unsigned int last_q_in_block = min(total_q, global_q_base + q_block) - 1u;
    const unsigned int block_max_visible = global_q_base < total_q
        ? min(context_len, start_position + last_q_in_block + 1u)
        : 0u;
    if (block_max_visible == 0u) {
        return;
    }
    const unsigned int block_min_visible_raw = (window_size > 0u
        && start_position + global_q_base + 1u > window_size)
        ? (start_position + global_q_base + 1u - window_size)
        : 0u;
    const unsigned int block_min_tile_start =
        (block_min_visible_raw / k_tile) * k_tile;

    extern __shared__ __align__(16) unsigned char smem[];
    unsigned short* q_shared = reinterpret_cast<unsigned short*>(smem);
    unsigned short* k_shared = q_shared + q_block * hdim;
    unsigned short* v_shared = k_shared + k_tile * hdim;
    float* scores = reinterpret_cast<float*>(v_shared + k_tile * hdim);
    float* scalars = scores + q_block * k_tile;
    half* weights_half = reinterpret_cast<half*>(scalars + q_block * 3u);
    // Per-warp 16x16 = 1 KiB scratch reused for the alpha rescale of one
    // c_frag at a time. Overlays k_shared (which has been consumed by
    // the time we reach the rescale step). 16 warps * 1 KiB = 16 KiB,
    // and k_shared is 16 KiB (k_tile=16 * hdim=512 * 2 B). Same overlay
    // pattern as the q_block=16 twin.
    float* acc_scratch = reinterpret_cast<float*>(k_shared);
    constexpr unsigned int acc_scratch_per_warp = 16u * 16u; // 256 floats per warp

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
    }
    for (unsigned int row = tid; row < q_block; row += blockDim.x) {
        scalars[row * 3u + 0u] = -3.402823466e38f;
        scalars[row * 3u + 1u] = 0.0f;
        scalars[row * 3u + 2u] = 0.0f;
    }
    __syncthreads();

#if __CUDA_ARCH__ >= 800
    using namespace nvcuda;
    // Persistent register-resident output accumulator: each warp owns
    // `frags_per_warp` (=4) WMMA fragments covering its 32-col slice
    // across both row-strips (rows 0..15 and rows 16..31).
    //   acc_frag[0] : rows  0..15, cols [warp*32 + 0  .. warp*32 + 15]
    //   acc_frag[1] : rows  0..15, cols [warp*32 + 16 .. warp*32 + 31]
    //   acc_frag[2] : rows 16..31, cols [warp*32 + 0  .. warp*32 + 15]
    //   acc_frag[3] : rows 16..31, cols [warp*32 + 16 .. warp*32 + 31]
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[frags_per_warp];
    wmma::fill_fragment(acc_frag[0], 0.0f);
    wmma::fill_fragment(acc_frag[1], 0.0f);
    wmma::fill_fragment(acc_frag[2], 0.0f);
    wmma::fill_fragment(acc_frag[3], 0.0f);
#endif

    const uint4 zero_vec = make_uint4(0u, 0u, 0u, 0u);
    for (unsigned int tile_start = block_min_tile_start; tile_start < block_max_visible; tile_start += k_tile) {
        const unsigned int tile_count = min(k_tile, block_max_visible - tile_start);
        constexpr unsigned int halfs_per_vec = sizeof(uint4) / sizeof(unsigned short);
        constexpr unsigned int kv_vecs = k_tile * hdim / halfs_per_vec;
        uint4* k_shared_vec = reinterpret_cast<uint4*>(k_shared);
        uint4* v_shared_vec = reinterpret_cast<uint4*>(v_shared);
        for (unsigned int vec = tid; vec < kv_vecs; vec += blockDim.x) {
            const unsigned int idx = vec * halfs_per_vec;
            const unsigned int col = idx / hdim;
            const unsigned int dim = idx - col * hdim;
            const unsigned int pos = tile_start + col;
            const bool valid_k = col < tile_count;
            const size_t kv_offset =
                (size_t(kv_slot(pos, cache_capacity)) * num_kv_heads + kv_head) * hdim + dim;
            k_shared_vec[vec] = valid_k
                ? *reinterpret_cast<const uint4*>(key_cache + kv_offset)
                : zero_vec;
            v_shared_vec[vec] = valid_k
                ? *reinterpret_cast<const uint4*>(value_cache + kv_offset)
                : zero_vec;
        }
        __syncthreads();

#if __CUDA_ARCH__ >= 800
        // Q*K WMMA: split HDIM=512 reduction across `qk_warps` warps so
        // 12/16 warps don't sit idle. Each of the first `qk_warps` warps
        // accumulates two independent 16x16 partials (one per row-strip)
        // covering its slice of HDIM. Layout per warp:
        //   row-strip 0 (Q rows  0..15) over kk in [warp*128, (warp+1)*128)
        //   row-strip 1 (Q rows 16..31) over kk in [warp*128, (warp+1)*128)
        // Results are summed elementwise into `scores` afterwards.
        // Partial slots overlay k_shared (which is no longer needed once
        // all warp mma chains have read it for the Q*K reduction).
        // 4 warps * 2 row-strips * 256 floats * 4 B = 8 KiB ≪ 16 KiB.
        constexpr unsigned int qk_warps = 4u;
        constexpr unsigned int hdim_per_qk_warp = hdim / qk_warps;  // 128
        constexpr unsigned int partial_stride = 256u * row_strips;  // floats per qk-warp
        float* partial_scores = reinterpret_cast<float*>(k_shared);
        if (warp < qk_warps) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[row_strips];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[row_strips];
            wmma::fill_fragment(c_frag[0], 0.0f);
            wmma::fill_fragment(c_frag[1], 0.0f);
            const unsigned int kk_start = warp * hdim_per_qk_warp;
            const unsigned int kk_end   = kk_start + hdim_per_qk_warp;
#pragma unroll
            for (unsigned int kk = kk_start; kk < kk_end; kk += 16u) {
                const half* a_ptr0 = reinterpret_cast<const half*>(q_shared +  0u * hdim + kk);
                const half* a_ptr1 = reinterpret_cast<const half*>(q_shared + 16u * hdim + kk);
                const half* b_ptr  = reinterpret_cast<const half*>(k_shared + kk);
                wmma::load_matrix_sync(a_frag[0], a_ptr0, hdim);
                wmma::load_matrix_sync(a_frag[1], a_ptr1, hdim);
                wmma::load_matrix_sync(b_frag,    b_ptr,  hdim);
                wmma::mma_sync(c_frag[0], a_frag[0], b_frag, c_frag[0]);
                wmma::mma_sync(c_frag[1], a_frag[1], b_frag, c_frag[1]);
            }
            // Sync so all qk_warps finish reading k_shared before any
            // warp writes partials into the same buffer (overlay).
            __syncthreads();
            wmma::store_matrix_sync(
                partial_scores + warp * partial_stride + 0u * 256u,
                c_frag[0], 16u, wmma::mem_row_major);
            wmma::store_matrix_sync(
                partial_scores + warp * partial_stride + 1u * 256u,
                c_frag[1], 16u, wmma::mem_row_major);
        } else {
            __syncthreads();
        }
        __syncthreads();
        // Reduce 4 partials per row-strip → scores[q_block * k_tile = 512].
        // Each partial is q_block(=32) rows * k_tile(=16) cols laid out as
        // two 16x16 row-strip blocks contiguously: [strip0 (256) | strip1 (256)].
        // We want scores[row * k_tile + col] for row in [0,32), col in [0,16).
        // For row in row-strip s (s = row / 16, r = row % 16), the source
        // index inside `partial_scores[w]` is `s * 256 + r * 16 + col`.
        // Since the layout per qk-warp is exactly that, indexing by linear
        // element e in [0, 512) gives:
        //   row = e / 16    (covers 0..31)
        //   col = e % 16
        //   source index inside qk-warp w = e   (because s*256 + r*16 + col == e)
        for (unsigned int e = tid; e < q_block * k_tile; e += blockDim.x) {
            scores[e] = partial_scores[0u * partial_stride + e]
                      + partial_scores[1u * partial_stride + e]
                      + partial_scores[2u * partial_stride + e]
                      + partial_scores[3u * partial_stride + e];
        }
#endif
        __syncthreads();

        // Softmax / online stats. q_block=32 rows distributed across
        // warps_per_block=16 warps → each warp handles 2 rows
        // (`for (row = warp; row < q_block; row += warps_per_block)`).
        for (unsigned int row = warp; row < q_block; row += warps_per_block) {
            const unsigned int global_q = global_q_base + row;
            const bool valid_q = global_q < total_q;
            const unsigned int visible_len = valid_q
                ? min(context_len, start_position + global_q + 1u)
                : 0u;
            const unsigned int row_min_visible = (window_size > 0u && start_position + global_q + 1u > window_size)
                ? (start_position + global_q + 1u - window_size) : 0u;
            const float old_m = scalars[row * 3u + 0u];
            const float old_l = scalars[row * 3u + 1u];
            // k_tile=16: only the lower half of each warp (lanes 0..15)
            // carries a real score. Lanes 16..31 contribute -inf to the
            // warp-wide reduce so they don't bias the row max/sum.
            const unsigned int pos = tile_start + lane;
            float score = (valid_q && lane < k_tile && lane < tile_count && pos < visible_len && pos >= row_min_visible)
                ? scores[row * k_tile + lane] * scale
                : -3.402823466e38f;
            const float tile_m = aegis_warp_reduce_max(score);
            const float new_m = fmaxf(old_m, tile_m);
            float weight = 0.0f;
            if (score > -3.0e38f) {
                weight = exp2f((score - new_m) * log2e);
            }
            if (lane < k_tile) {
                scores[row * k_tile + lane] = weight;
            }
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
        // Alpha rescale + P*V WMMA. For each of the warp's `frags_per_warp`
        // (=4) c_frags:
        //   1. Store frag to per-warp scratch (1 KiB, overlays k_shared).
        //   2. Multiply each scratch element by alpha[row].
        //      Row index for the rescale: row-strip s → rows [16s .. 16s+15].
        //      Within the 16x16 scratch, e = row_inside*16 + col, so
        //      scalars row index = s*16 + (e/16).
        //   3. Reload into c_frag.
        //   4. P[16x16] @ V[16x16] (single mma covers the full k_tile).
        // Two P fragments needed (one per row-strip): P_top = weights_half
        // rows  0..15, P_bot = weights_half rows 16..31.
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag[row_strips];
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;
        wmma::load_matrix_sync(p_frag[0], weights_half +  0u * k_tile, k_tile);
        wmma::load_matrix_sync(p_frag[1], weights_half + 16u * k_tile, k_tile);
        float* warp_scratch = acc_scratch + warp * acc_scratch_per_warp;
#pragma unroll
        for (unsigned int s = 0u; s < row_strips; ++s) {
            const unsigned int row_base = s * 16u;
#pragma unroll
            for (unsigned int f = 0u; f < frags_per_warp_col; ++f) {
                const unsigned int frag_idx = s * frags_per_warp_col + f;
                const unsigned int n_off = warp * cols_per_warp + f * 16u;
                // Step 1: store c_frag to warp-private scratch.
                wmma::store_matrix_sync(warp_scratch, acc_frag[frag_idx], 16u, wmma::mem_row_major);
                // Step 2: rescale by per-row alpha. 256 elements / 32 lanes
                // = 8 elements per lane. Stride elements by 32 (lane).
                //   row index inside scratch = e / 16 (covers 0..15)
                //   absolute row in q_block = row_base + (e / 16)
#pragma unroll
                for (unsigned int e = lane; e < 256u; e += 32u) {
                    const unsigned int row_in_strip = e / 16u;
                    const unsigned int abs_row = row_base + row_in_strip;
                    warp_scratch[e] *= scalars[abs_row * 3u + 2u];
                }
                // Step 3: reload rescaled values back into the accumulator
                // fragment. This serves as the seed for the upcoming
                // mma_sync (acc + P*V).
                wmma::load_matrix_sync(acc_frag[frag_idx], warp_scratch, 16u, wmma::mem_row_major);
                // Step 4: load V slice and mma-fuse-add into acc_frag.
                const half* v_ptr = reinterpret_cast<const half*>(v_shared + n_off);
                wmma::load_matrix_sync(v_frag, v_ptr, hdim);
                wmma::mma_sync(acc_frag[frag_idx], p_frag[s], v_frag, acc_frag[frag_idx]);
            }
        }
#endif
        __syncthreads();
    }

    // Final epilogue: divide by row-sum and write to global output.
    // Each warp writes its 32-col slice of every row across both row-strips.
#if __CUDA_ARCH__ >= 800
    float* warp_scratch = acc_scratch + warp * acc_scratch_per_warp;
#pragma unroll
    for (unsigned int s = 0u; s < row_strips; ++s) {
        const unsigned int row_base = s * 16u;
#pragma unroll
        for (unsigned int f = 0u; f < frags_per_warp_col; ++f) {
            const unsigned int frag_idx = s * frags_per_warp_col + f;
            const unsigned int n_off = warp * cols_per_warp + f * 16u;
            wmma::store_matrix_sync(warp_scratch, acc_frag[frag_idx], 16u, wmma::mem_row_major);
            // Each warp has 32 lanes; 16x16 = 256 elements; 8 per lane.
#pragma unroll
            for (unsigned int e = lane; e < 256u; e += 32u) {
                const unsigned int row_in_strip = e / 16u;
                const unsigned int col = e - row_in_strip * 16u;
                const unsigned int abs_row = row_base + row_in_strip;
                const unsigned int global_q = global_q_base + abs_row;
                if (global_q >= total_q) {
                    continue;
                }
                const float denom = fmaxf(scalars[abs_row * 3u + 1u], 1.0e-20f);
                const unsigned int dim = n_off + col;
                output[(size_t(global_q) * num_attention_heads + head) * hdim + dim] =
                    warp_scratch[e] / denom;
            }
        }
    }
#endif
}

// =============================================================================
// HDIM=512 register-resident-accumulator + cp.async pipelined K-only variant —
// Round 3 attention pipelining.
//
// Drop-in numerical-twin of `aegis_attention_prefill_dense_halfq_wmma_hdim512_regacc`.
// The only difference is HOW the K tile is staged from `key_cache` into
// shared memory:
//   * Synchronous twin:   global → shared (vectorised uint4 load) per K-iter,
//                         then __syncthreads(), then Q*K / softmax / P*V.
//   * This pipeline:      K is loaded via cp.async.ca.shared.global into a
//                         DOUBLE-BUFFERED pair of K slots, with one outstanding
//                         cp.async group so iter (k+1)'s K load is in-flight
//                         while iter (k)'s Q*K runs (Q*K is the bottleneck of
//                         the K critical path because softmax depends on it).
//
// Why K-only and not K+V (Option 1 of the round-3 follow-up):
//   * sm_120 (RTX 5070 Ti, Blackwell consumer) opt-in dynamic shared cap is
//     below 100 KiB — empirically 96 KiB is the safe ceiling. Pipelining BOTH
//     K and V (4 staged buffers × 16 KiB = 64 KiB of K/V alone) blows the cap.
//   * V is consumed late in the iter (after Q*K + softmax + rescale, in the
//     P*V WMMA), so a synchronous global→shared V load issued just after
//     the cp.async-K-wait runs in parallel with Q*K and softmax. The V load
//     is finished long before P*V touches v_shared.
//   * K dominates the latency-hiding win because Q*K is the FIRST compute
//     stage and the entire iter is gated on K being resident.
//
// Pipelining strategy (mirrors `aegis_nvfp4_grouped_prequant_gemm_wmma_bf16_t32_pipeline`,
//   restricted to the K side):
//   * Two K slots `k_shared[2][k_tile*hdim]`. Each slot is
//     k_tile=16 × hdim=512 × 2B = 16 KiB. 2 slots = 32 KiB.
//   * Single V slot `v_shared[k_tile*hdim]` = 16 KiB, loaded synchronously
//     each iter via uint4.
//   * Per K-iter we issue ONE cp.async commit_group for iter (k+1)'s K into
//     the next K buffer. `cp.async.wait_group<1>` keeps the current iter's K
//     ready while the next iter's K load runs in the background.
//   * Bit-identical K bytes vs. the synchronous twin's uint4 read.
//   * OOB rows (`pos >= context_len` etc.) for K use cp.async src_size=0
//     zero-fill; OOB rows for V use the same `valid_k ? *src : zero_vec`
//     clamp the synchronous twin uses.
//
// Per-warp rescale scratch: 16 KiB of warp-private scratch is needed during
// the alpha rescale step. We CANNOT overlay k_shared (it holds prefetched
// K[k+1] while we compute on K[k]). v_shared is alive during P*V. q_shared
// is alive across the whole loop. So a dedicated `acc_scratch[16 warps *
// 16 * 16]` = 16 KiB region is added.
//
// Shared-memory budget:
//   q_shared           16 KiB
//   k_shared[2]        32 KiB
//   v_shared[1]        16 KiB    (single buffer, synchronous load)
//   acc_scratch        16 KiB
//   scores              1 KiB
//   scalars + weights ~700 B
//   total            ~81.7 KiB → fits comfortably in the 96 KiB sm_120 cap.
//
// Block-per-SM residency: shared-mem footprint at ~82 KiB still keeps us at
// 1 block / SM on the 100 KiB sm_120 SM (synchronous twin at ~50 KiB gets
// 2 blocks). The win comes from cp.async-issued K load overlap with the
// previous iter's P*V and the current iter's V load + Q*K issue.
//
// Numerical correctness vs. the synchronous twin:
//   * cp.async copies bf16 bytes — same byte stream the uint4 load reads.
//   * OOB K rows: synchronous path stores `zero_vec` (16 zero bytes); cp.async
//     with src_size=0 zero-fills 16 destination bytes. Identical.
//   * V load is identical to the synchronous twin (same uint4 + zero_vec).
//   * Q*K, softmax, P*V, alpha rescale — all unchanged.
//   * `acc_frag[]` accumulation order is identical → bit-identical output.
// =============================================================================
extern "C" __global__
__launch_bounds__(512, 1)
void aegis_attention_prefill_dense_halfq_wmma_hdim512_regacc_pipeline(
    const unsigned short* __restrict__ key_cache,
    const unsigned short* __restrict__ value_cache,
    const unsigned short* __restrict__ query,
    const unsigned int start_position,
    const unsigned int total_q,
    const unsigned int context_len,
    const unsigned int num_attention_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int cache_capacity,
    const unsigned int window_size,
    float* __restrict__ output
) {
    constexpr unsigned int hdim = 512u;
    constexpr unsigned int q_block = 16u;
    constexpr unsigned int k_tile = 16u;
    constexpr unsigned int warps_per_block = 16u;
    constexpr unsigned int cols_per_warp = hdim / warps_per_block; // 32
    constexpr unsigned int frags_per_warp = cols_per_warp / 16u;   // 2
    const unsigned int head = blockIdx.x;
    const unsigned int global_q_base = blockIdx.y * q_block;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5u;
    if (head_dim != hdim || head >= num_attention_heads || blockDim.x < warps_per_block * 32u) {
        return;
    }

    const unsigned int last_q_in_block = min(total_q, global_q_base + q_block) - 1u;
    const unsigned int block_max_visible = global_q_base < total_q
        ? min(context_len, start_position + last_q_in_block + 1u)
        : 0u;
    if (block_max_visible == 0u) {
        return;
    }
    const unsigned int block_min_visible_raw = (window_size > 0u
        && start_position + global_q_base + 1u > window_size)
        ? (start_position + global_q_base + 1u - window_size)
        : 0u;
    const unsigned int block_min_tile_start =
        (block_min_visible_raw / k_tile) * k_tile;

    extern __shared__ __align__(16) unsigned char smem[];
    unsigned short* q_shared      = reinterpret_cast<unsigned short*>(smem);
    unsigned short* k_shared_buf  = q_shared + q_block * hdim;            // 2 * k_tile * hdim (cp.async double-buffered)
    unsigned short* v_shared      = k_shared_buf + 2u * k_tile * hdim;    // 1 * k_tile * hdim (synchronous)
    float*          acc_scratch   = reinterpret_cast<float*>(v_shared + k_tile * hdim);
    constexpr unsigned int acc_scratch_per_warp = 16u * 16u; // 256 floats per warp
    float*          scores        = acc_scratch + warps_per_block * acc_scratch_per_warp;
    float*          scalars       = scores + q_block * k_tile;
    half*           weights_half  = reinterpret_cast<half*>(scalars + q_block * 3u);

    const unsigned int group = num_attention_heads / num_kv_heads;
    const unsigned int kv_head = head / group;
    const float scale = rsqrtf(float(hdim));
    const float log2e = 1.4426950408889634f;

    // -------------------------------------------------------------------------
    // Q tile load (synchronous, identical to the twin).
    // -------------------------------------------------------------------------
    for (unsigned int idx = tid; idx < q_block * hdim; idx += blockDim.x) {
        const unsigned int row = idx / hdim;
        const unsigned int dim = idx - row * hdim;
        const unsigned int global_q = global_q_base + row;
        q_shared[idx] = global_q < total_q
            ? query[(size_t(global_q) * num_attention_heads + head) * hdim + dim]
            : 0u;
    }
    for (unsigned int row = tid; row < q_block; row += blockDim.x) {
        scalars[row * 3u + 0u] = -3.402823466e38f;
        scalars[row * 3u + 1u] = 0.0f;
        scalars[row * 3u + 2u] = 0.0f;
    }

#if __CUDA_ARCH__ >= 800
    using namespace nvcuda;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag[frags_per_warp];
    wmma::fill_fragment(acc_frag[0], 0.0f);
    wmma::fill_fragment(acc_frag[1], 0.0f);
#endif

    // -------------------------------------------------------------------------
    // cp.async helpers + per-iter K-tile prefetch (V is loaded synchronously).
    //
    // Each K tile is k_tile=16 rows × hdim=512 halfs = 8192 halfs = 16 KiB.
    // We map each thread to one 16-byte (8-half) chunk: 512 threads × 16B = 8 KiB.
    // That covers HALF of the tile per pass; we issue TWO 16-byte cp.async ops
    // per thread per tile (chunks 0 and 1). Together: 1024 chunks × 16B = 16 KiB
    // = exactly one tile.
    //
    // chunk layout (per pass):
    //   chunk_id_in_tile = pass_idx * 512 + tid       (0..1023)
    //   row              = chunk_id_in_tile / 64       (0..15)   — k_tile rows
    //   half_idx_in_row  = (chunk_id_in_tile % 64) * 8 (0..504, step 8)
    // -------------------------------------------------------------------------
#if __CUDA_ARCH__ >= 800
    auto cvt_smem = [] (const void* p) -> unsigned int {
        unsigned int s;
        asm volatile("{ .reg .u64 t;\n\t"
                     "  cvta.to.shared.u64 t, %1;\n\t"
                     "  cvt.u32.u64 %0, t; }\n"
                     : "=r"(s) : "l"(p));
        return s;
    };
    auto cp_async_16 = [] (unsigned int dst_smem, const void* src) {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                     :: "r"(dst_smem), "l"(src));
    };
    auto cp_async_zero_16 = [] (unsigned int dst_smem) {
        // src ptr is unused when src_size=0; pass any valid pointer.
        const unsigned long long any = 0ULL;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16, 0;\n"
                     :: "r"(dst_smem), "l"((const void*)&any));
    };
    auto cp_async_commit = [] () {
        asm volatile("cp.async.commit_group;\n" ::);
    };
    auto cp_async_wait_lt1 = [] () {
        asm volatile("cp.async.wait_group 1;\n" ::);
    };
    auto cp_async_wait_lt0 = [] () {
        asm volatile("cp.async.wait_group 0;\n" ::);
    };

    // Issue K[tile_start] prefetch into k_shared_buf[buf_idx]. Caller commits
    // the cp.async group separately. V is NOT prefetched here — it is loaded
    // synchronously inside the main loop after the K wait (V is consumed late
    // in the iter, after Q*K + softmax + rescale, so a synchronous load that
    // overlaps with Q*K via SM scoreboarding is enough).
    //
    // Bit-identical to the synchronous twin's tile-load for K: zero-fill any
    // row where `col >= tile_count` (where tile_count = min(k_tile,
    // block_max_visible - tile_start_arg)). For valid rows we issue a 16-byte
    // cp.async whose source matches the synchronous twin's uint4 read of
    // `key_cache + kv_offset`.
    auto issue_k_prefetch = [&] (unsigned int tile_start_arg, unsigned int buf_idx) {
        constexpr unsigned int halfs_per_chunk = 8u; // 16 bytes
        constexpr unsigned int chunks_per_row  = hdim / halfs_per_chunk;     // 64
        constexpr unsigned int chunks_per_tile = k_tile * chunks_per_row;    // 1024
        constexpr unsigned int passes          = chunks_per_tile / 512u;     // 2
        const unsigned int tile_count_arg = min(k_tile, block_max_visible - tile_start_arg);
        unsigned short* k_buf = k_shared_buf + buf_idx * (k_tile * hdim);
        #pragma unroll
        for (unsigned int p = 0u; p < passes; ++p) {
            const unsigned int chunk      = p * 512u + tid;
            const unsigned int row        = chunk >> 6;     // / 64  (0..15)
            const unsigned int half_off   = (chunk & 63u) << 3; // (chunk%64)*8 (0..504, step 8)
            const bool valid              = row < tile_count_arg;
            const unsigned int pos        = tile_start_arg + row;
            const size_t kv_offset = valid
                ? (size_t(kv_slot(pos, cache_capacity)) * num_kv_heads + kv_head) * hdim + half_off
                : (size_t)half_off;
            unsigned int k_dst_smem = cvt_smem(&k_buf[row * hdim + half_off]);
            if (valid) {
                cp_async_16(k_dst_smem, key_cache + kv_offset);
            } else {
                cp_async_zero_16(k_dst_smem);
            }
        }
    };
#endif

    // Number of K-iters that we'll execute in the loop.
    const unsigned int n_kiters = (block_max_visible > block_min_tile_start)
        ? ((block_max_visible - block_min_tile_start + k_tile - 1u) / k_tile)
        : 0u;

#if __CUDA_ARCH__ >= 800
    // Prologue: issue K prefetch for tile 0 into buffer 0, commit as group 0.
    if (n_kiters > 0u) {
        issue_k_prefetch(block_min_tile_start, 0u);
        cp_async_commit();
    }
#endif
    __syncthreads(); // ensure scalars/q_shared init visible before loop

    const uint4 zero_vec = make_uint4(0u, 0u, 0u, 0u);
    for (unsigned int it = 0u; it < n_kiters; ++it) {
        const unsigned int tile_start = block_min_tile_start + it * k_tile;
        const unsigned int tile_count = min(k_tile, block_max_visible - tile_start);
        const unsigned int buf_cur = it & 1u;
        const unsigned int buf_nxt = buf_cur ^ 1u;
        unsigned short* k_shared = k_shared_buf + buf_cur * (k_tile * hdim);

#if __CUDA_ARCH__ >= 800
        // Issue the next iter's K prefetch, then wait until ≤1 group remains
        // (i.e., the current iter's K data is ready).
        if (it + 1u < n_kiters) {
            issue_k_prefetch(block_min_tile_start + (it + 1u) * k_tile, buf_nxt);
            cp_async_commit();
            cp_async_wait_lt1();
        } else {
            cp_async_wait_lt0();
        }

        // Synchronous V load for the current iter. Bit-identical to the
        // synchronous twin's V uint4 path. Issued AFTER the K cp.async wait
        // so the K group is already in flight; the V load runs in parallel
        // with the next iter's K cp.async DMA via the SM's load/store unit.
        // V is consumed late (P*V), so by the time we touch v_shared the
        // load is comfortably visible after the post-Q*K sync.
        {
            constexpr unsigned int halfs_per_vec = sizeof(uint4) / sizeof(unsigned short);
            constexpr unsigned int v_vecs = k_tile * hdim / halfs_per_vec;
            uint4* v_shared_vec = reinterpret_cast<uint4*>(v_shared);
            for (unsigned int vec = tid; vec < v_vecs; vec += blockDim.x) {
                const unsigned int idx = vec * halfs_per_vec;
                const unsigned int col = idx / hdim;
                const unsigned int dim = idx - col * hdim;
                const unsigned int pos = tile_start + col;
                const bool valid_k = col < tile_count;
                const size_t kv_offset =
                    (size_t(kv_slot(pos, cache_capacity)) * num_kv_heads + kv_head) * hdim + dim;
                v_shared_vec[vec] = valid_k
                    ? *reinterpret_cast<const uint4*>(value_cache + kv_offset)
                    : zero_vec;
            }
        }
        __syncthreads();
#else
        // Pre-sm_80 fallback: synchronous load (matches the twin exactly).
        constexpr unsigned int halfs_per_vec = sizeof(uint4) / sizeof(unsigned short);
        constexpr unsigned int kv_vecs = k_tile * hdim / halfs_per_vec;
        uint4* k_shared_vec = reinterpret_cast<uint4*>(k_shared);
        uint4* v_shared_vec = reinterpret_cast<uint4*>(v_shared);
        for (unsigned int vec = tid; vec < kv_vecs; vec += blockDim.x) {
            const unsigned int idx = vec * halfs_per_vec;
            const unsigned int col = idx / hdim;
            const unsigned int dim = idx - col * hdim;
            const unsigned int pos = tile_start + col;
            const bool valid_k = col < tile_count;
            const size_t kv_offset =
                (size_t(kv_slot(pos, cache_capacity)) * num_kv_heads + kv_head) * hdim + dim;
            k_shared_vec[vec] = valid_k
                ? *reinterpret_cast<const uint4*>(key_cache + kv_offset)
                : zero_vec;
            v_shared_vec[vec] = valid_k
                ? *reinterpret_cast<const uint4*>(value_cache + kv_offset)
                : zero_vec;
        }
        __syncthreads();
        (void)buf_nxt; // unused on pre-sm_80
#endif

#if __CUDA_ARCH__ >= 800
        if (warp < 1u) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);
#pragma unroll
            for (unsigned int kk = 0u; kk < hdim; kk += 16u) {
                const half* a_ptr = reinterpret_cast<const half*>(q_shared + kk);
                const half* b_ptr = reinterpret_cast<const half*>(k_shared + kk);
                wmma::load_matrix_sync(a_frag, a_ptr, hdim);
                wmma::load_matrix_sync(b_frag, b_ptr, hdim);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(scores, c_frag, k_tile, wmma::mem_row_major);
        }
#endif
        __syncthreads();

        // Softmax / online stats — identical to the twin.
        if (warp < q_block) {
            const unsigned int row = warp;
            const unsigned int global_q = global_q_base + row;
            const bool valid_q = global_q < total_q;
            const unsigned int visible_len = valid_q
                ? min(context_len, start_position + global_q + 1u)
                : 0u;
            const unsigned int row_min_visible = (window_size > 0u && start_position + global_q + 1u > window_size)
                ? (start_position + global_q + 1u - window_size) : 0u;
            const float old_m = scalars[row * 3u + 0u];
            const float old_l = scalars[row * 3u + 1u];
            const unsigned int pos = tile_start + lane;
            float score = (valid_q && lane < k_tile && lane < tile_count && pos < visible_len && pos >= row_min_visible)
                ? scores[row * k_tile + lane] * scale
                : -3.402823466e38f;
            const float tile_m = aegis_warp_reduce_max(score);
            const float new_m = fmaxf(old_m, tile_m);
            float weight = 0.0f;
            if (score > -3.0e38f) {
                weight = exp2f((score - new_m) * log2e);
            }
            if (lane < k_tile) {
                scores[row * k_tile + lane] = weight;
            }
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
        // Alpha rescale + P*V — identical to the twin, but acc_scratch is
        // now its own region (NOT overlaying k_shared, which holds prefetched
        // K[k+1] for the next iter).
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;
        wmma::load_matrix_sync(p_frag, weights_half, k_tile);
        float* warp_scratch = acc_scratch + warp * acc_scratch_per_warp;
#pragma unroll
        for (unsigned int f = 0u; f < frags_per_warp; ++f) {
            const unsigned int n_off = warp * cols_per_warp + f * 16u;
            wmma::store_matrix_sync(warp_scratch, acc_frag[f], 16u, wmma::mem_row_major);
#pragma unroll
            for (unsigned int e = lane; e < 256u; e += 32u) {
                const unsigned int row = e / 16u;
                warp_scratch[e] *= scalars[row * 3u + 2u];
            }
            wmma::load_matrix_sync(acc_frag[f], warp_scratch, 16u, wmma::mem_row_major);
            const half* v_ptr = reinterpret_cast<const half*>(v_shared + n_off);
            wmma::load_matrix_sync(v_frag, v_ptr, hdim);
            wmma::mma_sync(acc_frag[f], p_frag, v_frag, acc_frag[f]);
        }
#endif
        __syncthreads();
    }

    // Final epilogue — same as the twin.
#if __CUDA_ARCH__ >= 800
    float* warp_scratch = acc_scratch + warp * acc_scratch_per_warp;
#pragma unroll
    for (unsigned int f = 0u; f < frags_per_warp; ++f) {
        const unsigned int n_off = warp * cols_per_warp + f * 16u;
        wmma::store_matrix_sync(warp_scratch, acc_frag[f], 16u, wmma::mem_row_major);
#pragma unroll
        for (unsigned int e = lane; e < 256u; e += 32u) {
            const unsigned int row = e / 16u;
            const unsigned int col = e - row * 16u;
            const unsigned int global_q = global_q_base + row;
            if (global_q >= total_q) {
                continue;
            }
            const float denom = fmaxf(scalars[row * 3u + 1u], 1.0e-20f);
            const unsigned int dim = n_off + col;
            output[(size_t(global_q) * num_attention_heads + head) * hdim + dim] =
                warp_scratch[e] / denom;
        }
    }
#endif
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
    const unsigned int cache_capacity,
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

    const uint4 zero_vec = make_uint4(0u, 0u, 0u, 0u);
    for (unsigned int tile_start = 0u; tile_start < block_max_visible; tile_start += k_tile) {
        const unsigned int tile_count = min(k_tile, block_max_visible - tile_start);
        constexpr unsigned int halfs_per_vec = sizeof(uint4) / sizeof(unsigned short);
        constexpr unsigned int kv_vecs = k_tile * hdim / halfs_per_vec;
        uint4* k_shared_vec = reinterpret_cast<uint4*>(k_shared);
        uint4* v_shared_vec = reinterpret_cast<uint4*>(v_shared);
        for (unsigned int vec = tid; vec < kv_vecs; vec += blockDim.x) {
            const unsigned int idx = vec * halfs_per_vec;
            const unsigned int col = idx / hdim;
            const unsigned int dim = idx - col * hdim;
            const unsigned int pos = tile_start + col;
            const bool valid_k = col < tile_count;
            const size_t kv_offset =
                (size_t(kv_slot(pos, cache_capacity)) * num_kv_heads + kv_head) * hdim + dim;
            k_shared_vec[vec] = valid_k
                ? *reinterpret_cast<const uint4*>(key_cache + kv_offset)
                : zero_vec;
            v_shared_vec[vec] = valid_k
                ? *reinterpret_cast<const uint4*>(value_cache + kv_offset)
                : zero_vec;
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
    const unsigned int cache_capacity,
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

    constexpr unsigned int q_halfs_per_vec = sizeof(uint4) / sizeof(unsigned short);
    constexpr unsigned int q_vecs = q_rows * hdim / q_halfs_per_vec;
    uint4* q_shared_vec = reinterpret_cast<uint4*>(q_shared);
    const uint4 zero_q_vec = make_uint4(0u, 0u, 0u, 0u);
    for (unsigned int vec = tid; vec < q_vecs; vec += blockDim.x) {
        const unsigned int idx = vec * q_halfs_per_vec;
        const unsigned int row = idx / hdim;
        const unsigned int dim = idx - row * hdim;
        const unsigned int local_head = row / q_tokens;
        const unsigned int token = row - local_head * q_tokens;
        const unsigned int head = kv_head * group + local_head_base + local_head;
        const unsigned int global_q = global_q_base + token;
        const bool valid_q = local_head_base + local_head < group
            && head < num_attention_heads
            && global_q < total_q;
        const size_t query_offset =
            (size_t(global_q) * num_attention_heads + head) * hdim + dim;
        q_shared_vec[vec] = valid_q
            ? *reinterpret_cast<const uint4*>(query + query_offset)
            : zero_q_vec;
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
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> pv_frag[2];
    if (warp < 4u) {
        const unsigned int row_block = warp >> 1u;
#pragma unroll
        for (unsigned int kk = 0u; kk < hdim; kk += 16u) {
            const half* q_ptr = reinterpret_cast<const half*>(q_shared + row_block * 16u * hdim + kk);
            wmma::load_matrix_sync(q_frag[kk / 16u], q_ptr, hdim);
        }
    }
    if (warp < 8u) {
        wmma::fill_fragment(pv_frag[0], 0.0f);
        wmma::fill_fragment(pv_frag[1], 0.0f);
    }
#endif

    const uint4 zero_vec = make_uint4(0u, 0u, 0u, 0u);
    for (unsigned int tile_start = 0u; tile_start < block_max_visible; tile_start += k_tile) {
        const unsigned int tile_count = min(k_tile, block_max_visible - tile_start);
        constexpr unsigned int halfs_per_vec = sizeof(uint4) / sizeof(unsigned short);
        constexpr unsigned int kv_vecs = k_tile * hdim / halfs_per_vec;
        uint4* k_shared_vec = reinterpret_cast<uint4*>(k_shared);
        uint4* v_shared_vec = reinterpret_cast<uint4*>(v_shared);
        for (unsigned int vec = tid; vec < kv_vecs; vec += blockDim.x) {
            const unsigned int idx = vec * halfs_per_vec;
            const unsigned int col = idx / hdim;
            const unsigned int dim = idx - col * hdim;
            const unsigned int pos = tile_start + col;
            const bool valid_k = col < tile_count;
            const size_t kv_offset =
                (size_t(kv_slot(pos, cache_capacity)) * num_kv_heads + kv_head) * hdim + dim;
            k_shared_vec[vec] = valid_k
                ? *reinterpret_cast<const uint4*>(key_cache + kv_offset)
                : zero_vec;
            v_shared_vec[vec] = valid_k
                ? *reinterpret_cast<const uint4*>(value_cache + kv_offset)
                : zero_vec;
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

#if __CUDA_ARCH__ >= 800
        if (warp < 8u) {
            using namespace nvcuda;
            const unsigned int n_off = warp * 16u;
            for (unsigned int row_base = 0u; row_base < q_rows; row_base += 16u) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;
                const unsigned int row_fragment = row_base >> 4u;
                aegis_scale_wmma_accumulator_m16n16_rows(pv_frag[row_fragment], scalars, row_base);
#pragma unroll
                for (unsigned int kk = 0u; kk < k_tile; kk += 16u) {
                    const half* p_ptr = weights_half + row_base * score_stride + kk;
                    const half* v_ptr = reinterpret_cast<const half*>(v_shared + kk * hdim + n_off);
                    wmma::load_matrix_sync(p_frag, p_ptr, score_stride);
                    wmma::load_matrix_sync(v_frag, v_ptr, hdim);
                    wmma::mma_sync(pv_frag[row_fragment], p_frag, v_frag, pv_frag[row_fragment]);
                }
            }
        }
#endif
        __syncthreads();
    }

#if __CUDA_ARCH__ >= 800
    if (warp < 8u) {
        using namespace nvcuda;
        const unsigned int n_off = warp * 16u;
        wmma::store_matrix_sync(acc + n_off, pv_frag[0], acc_stride, wmma::mem_row_major);
        wmma::store_matrix_sync(acc + 16u * acc_stride + n_off, pv_frag[1], acc_stride, wmma::mem_row_major);
    }
    __syncthreads();
#endif

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
    const unsigned int cache_capacity,
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
                (size_t(kv_slot(pos, cache_capacity)) * num_kv_heads + kv_head) * hdim + dim;
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
    const unsigned int cache_capacity,
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
                (size_t(kv_slot(pos, cache_capacity)) * num_kv_heads + kv_head) * hdim + dim;
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

// Q_BLOCK=32 variant of the hdim=128 dense WMMA prefill kernel.
//
// Identical online-softmax/WMMA structure to the generic
// `aegis_attention_prefill_dense_halfq_wmma_impl<128>` template
// (Q_BLOCK=16) but doubles the Q tile so each K/V tile load is
// amortised over twice as many query rows. For sliding-window layers
// (Gemma-4: 25/30 layers, window=1024) where the K-iteration count is
// bounded to `window/k_tile = 32` regardless of seq length, this halves
// the K/V shared-memory load traffic and the Q*K WMMA tile count grows
// from 2 to 4 (two 16x16 score sub-tiles per warp).
//
// `window_size = 0` means full causal attention. `window_size > 0`
// applies the same `block_min_tile_start` + per-row `row_min_visible`
// clamp as the generic template (commit 016a485) so sliding-layer
// behaviour matches the Q_BLOCK=16 reference bit-for-bit per output
// element: same K positions scanned in the same tile order, same
// online-softmax math per row, same hdim/k_tile WMMA accumulator
// reduction order.
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
    const unsigned int cache_capacity,
    const unsigned int window_size,
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
    // Sliding-window lower bound on the K-tile loop. Earliest visible K
    // position for the earliest query in this block (global_q =
    // global_q_base) is `start_position + global_q_base + 1 -
    // window_size`. Round down to a k_tile boundary so we still hit
    // tile_start cadence and per-row mask drops the residual.
    // window_size == 0 means full causal (no clamp).
    const unsigned int block_min_visible_raw = (window_size > 0u
        && start_position + global_q_base + 1u > window_size)
        ? (start_position + global_q_base + 1u - window_size)
        : 0u;
    const unsigned int block_min_tile_start =
        (block_min_visible_raw / k_tile) * k_tile;

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

    for (unsigned int tile_start = block_min_tile_start; tile_start < block_max_visible; tile_start += k_tile) {
        const unsigned int tile_count = min(k_tile, block_max_visible - tile_start);
        for (unsigned int idx = tid; idx < k_tile * hdim; idx += blockDim.x) {
            const unsigned int col = idx / hdim;
            const unsigned int dim = idx - col * hdim;
            const unsigned int pos = tile_start + col;
            const bool valid_k = col < tile_count;
            const size_t kv_offset =
                (size_t(kv_slot(pos, cache_capacity)) * num_kv_heads + kv_head) * hdim + dim;
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
            // Sliding-window per-row lower bound. Mirrors the generic
            // template's `row_min_visible` clamp so q_block=32 numerics
            // match q_block=16 bit-for-bit on sliding layers. When
            // window_size==0 this is 0 and the mask reduces to causal.
            const unsigned int row_min_visible = (window_size > 0u && start_position + global_q + 1u > window_size)
                ? (start_position + global_q + 1u - window_size) : 0u;
            const float old_m = scalars[row * 3u + 0u];
            const float old_l = scalars[row * 3u + 1u];
            const unsigned int pos = tile_start + lane;
            float score = (valid_q && lane < tile_count && pos < visible_len && pos >= row_min_visible)
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
    const unsigned int cache_capacity,
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
                (size_t(kv_slot(pos, cache_capacity)) * num_kv_heads + kv_head) * hdim + dim;
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
extern "C" __global__ void aegis_attention_prefill_batched_warp(
    const unsigned short* key_cache,
    const unsigned short* value_cache,
    const float* query,
    const unsigned int start_position,
    const unsigned int batch,
    const unsigned int num_attention_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int cache_capacity,
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
        const unsigned short* k = key_cache + (size_t(kv_slot(pos, cache_capacity)) * num_kv_heads + kv_head) * head_dim;
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
            const unsigned short* v = value_cache + (size_t(kv_slot(pos, cache_capacity)) * num_kv_heads + kv_head) * head_dim;
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

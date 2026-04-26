// Aegis-native FlashAttention-4 forward slice for Blackwell.
//
// Scope:
// - SM12.x dispatch is enforced in Rust.
// - f16 Q/K/V cache, f32 output.
// - head_dim == 128, causal single-sequence prefill, paged KV, GQA.
//
// This keeps ownership at the Aegis C ABI boundary. It follows the FA4 forward
// shape (larger Q/K tiles, shared-memory K/V staging, online softmax, paged KV
// indirection) without pulling in CuTeDSL/CUTLASS or PyTorch runtime pieces. The
// remaining gap to true FA4 is tensor-memory tcgen05/TMA warp specialization.
extern "C" __global__ void aegis_attention_prefill_paged_varlen_fa4_hdim128(
    const unsigned short* __restrict__ key_cache,
    const unsigned short* __restrict__ value_cache,
    const unsigned short* __restrict__ query,
    const unsigned int* __restrict__ slot_mapping,
    const unsigned int* __restrict__ cu_q,
    const unsigned int* __restrict__ context_lens,
    const unsigned int* __restrict__ block_tables,
    const unsigned int num_sequences,
    const unsigned int total_q,
    const unsigned int num_attention_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int page_tokens,
    const unsigned int block_table_stride,
    const unsigned int physical_slots,
    float* __restrict__ output
) {
    constexpr unsigned int q_block = 8u;
    constexpr unsigned int k_tile = 32u;
    constexpr unsigned int hdim = 128u;

    const unsigned int head = blockIdx.x;
    const unsigned int global_q_base = blockIdx.y * q_block;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5u;
    const unsigned int nwarps = blockDim.x >> 5u;

    if (head_dim != hdim
        || page_tokens == 0u
        || head >= num_attention_heads
        || global_q_base >= total_q
        || num_sequences != 1u
        || nwarps == 0u) {
        return;
    }

    const unsigned int q_start = cu_q[0];
    const unsigned int q_end = cu_q[1];
    const unsigned int q_len = q_end - q_start;
    const unsigned int context_len = context_lens[0];
    const unsigned int group = num_attention_heads / num_kv_heads;
    const unsigned int kv_head = head / group;
    const float scale = rsqrtf(float(hdim));
    const unsigned int* block_table = block_tables;

    unsigned int visible_len[q_block];
    bool valid[q_block];
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
            visible_len[row] = context_len > hidden_future ? context_len - hidden_future : 0u;
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
    float* partial = shared;                              // q_block * k_tile * nwarps
    float* weights = partial + q_block * k_tile * nwarps; // q_block * k_tile
    float* q_shared = weights + q_block * k_tile;         // q_block * hdim
    float* k_shared = q_shared + q_block * hdim;          // k_tile * hdim
    float* v_shared = k_shared + k_tile * hdim;           // k_tile * hdim
    float* acc = v_shared + k_tile * hdim;                // q_block * hdim
    float* scalars = acc + q_block * hdim;                // q_block * 4
    float* kv_valid = scalars + q_block * 4;              // k_tile

#pragma unroll
    for (unsigned int row = 0u; row < q_block; ++row) {
        const unsigned int global_q = global_q_base + row;
        if (valid[row]) {
            const unsigned short* q =
                query + (size_t(global_q) * num_attention_heads + head) * hdim;
            for (unsigned int dim = tid; dim < hdim; dim += blockDim.x) {
                q_shared[row * hdim + dim] = f16_bits_to_float(q[dim]);
                acc[row * hdim + dim] = 0.0f;
            }
        } else {
            for (unsigned int dim = tid; dim < hdim; dim += blockDim.x) {
                acc[row * hdim + dim] = 0.0f;
            }
        }
        if (tid == 0u) {
            scalars[row * 4u + 0u] = -3.402823466e38f; // running max
            scalars[row * 4u + 1u] = 0.0f;            // running denominator
            scalars[row * 4u + 2u] = 1.0f;            // accumulator rescale
            scalars[row * 4u + 3u] = 0.0f;
        }
    }
    __syncthreads();

    for (unsigned int tile_start = 0u; tile_start < max_visible; tile_start += k_tile) {
        const unsigned int tile_count = min(k_tile, max_visible - tile_start);

        for (unsigned int idx = tid; idx < tile_count * hdim; idx += blockDim.x) {
            const unsigned int col = idx / hdim;
            const unsigned int dim = idx - col * hdim;
            const unsigned int pos = tile_start + col;
            const unsigned int logical_page = pos / page_tokens;
            const unsigned int page_offset = pos - logical_page * page_tokens;
            const unsigned int physical_page =
                block_table[size_t(0) * block_table_stride + logical_page];
            const size_t physical_slot =
                size_t(physical_page) * size_t(page_tokens) + size_t(page_offset);
            const bool valid_slot = physical_slot < size_t(physical_slots);
            if (dim == 0u) {
                kv_valid[col] = valid_slot ? 1.0f : 0.0f;
            }
            if (valid_slot) {
                const size_t kv_offset =
                    (size_t(physical_slot) * num_kv_heads + kv_head) * hdim + dim;
                k_shared[col * hdim + dim] = f16_bits_to_float(key_cache[kv_offset]);
                v_shared[col * hdim + dim] = f16_bits_to_float(value_cache[kv_offset]);
            } else {
                k_shared[col * hdim + dim] = 0.0f;
                v_shared[col * hdim + dim] = 0.0f;
            }
        }
        __syncthreads();

#pragma unroll
        for (unsigned int row = 0u; row < q_block; ++row) {
#pragma unroll
            for (unsigned int col = 0u; col < k_tile; ++col) {
                const unsigned int pos = tile_start + col;
                float dot = 0.0f;
                if (col < tile_count && kv_valid[col] > 0.5f && valid[row] && pos < visible_len[row]) {
                    for (unsigned int dim = tid; dim < hdim; dim += blockDim.x) {
                        dot += q_shared[row * hdim + dim] * k_shared[col * hdim + dim];
                    }
                }
#pragma unroll
                for (unsigned int offset = 16u; offset > 0u; offset >>= 1) {
                    dot += __shfl_down_sync(0xffffffffu, dot, offset);
                }
                if (lane == 0u) {
                    partial[(row * k_tile + col) * nwarps + warp] = dot;
                }
            }
        }
        __syncthreads();

        for (unsigned int idx = tid; idx < q_block * k_tile; idx += blockDim.x) {
            float sum = 0.0f;
            for (unsigned int w = 0u; w < nwarps; ++w) {
                sum += partial[idx * nwarps + w];
            }
            const unsigned int row = idx / k_tile;
            const unsigned int col = idx - row * k_tile;
            const unsigned int pos = tile_start + col;
            weights[idx] =
                (col < tile_count && kv_valid[col] > 0.5f && valid[row] && pos < visible_len[row])
                    ? sum * scale
                    : -3.402823466e38f;
        }
        __syncthreads();

        if (tid < q_block) {
            const unsigned int row = tid;
            float tile_m = scalars[row * 4u + 0u];
#pragma unroll
            for (unsigned int col = 0u; col < k_tile; ++col) {
                tile_m = fmaxf(tile_m, weights[row * k_tile + col]);
            }
            const float old_m = scalars[row * 4u + 0u];
            const float old_l = scalars[row * 4u + 1u];
            float tile_l = 0.0f;
#pragma unroll
            for (unsigned int col = 0u; col < k_tile; ++col) {
                float weight = 0.0f;
                if (weights[row * k_tile + col] > -3.0e38f) {
                    weight = expf(weights[row * k_tile + col] - tile_m);
                }
                weights[row * k_tile + col] = weight;
                tile_l += weight;
            }
            const float alpha = old_l > 0.0f ? expf(old_m - tile_m) : 0.0f;
            scalars[row * 4u + 0u] = tile_m;
            scalars[row * 4u + 1u] = old_l * alpha + tile_l;
            scalars[row * 4u + 2u] = alpha;
        }
        __syncthreads();

#pragma unroll
        for (unsigned int row = 0u; row < q_block; ++row) {
            if (!valid[row]) {
                continue;
            }
            const float alpha = scalars[row * 4u + 2u];
            for (unsigned int dim = tid; dim < hdim; dim += blockDim.x) {
                float tile_acc = 0.0f;
#pragma unroll
                for (unsigned int col = 0u; col < k_tile; ++col) {
                    tile_acc += weights[row * k_tile + col] * v_shared[col * hdim + dim];
                }
                acc[row * hdim + dim] = acc[row * hdim + dim] * alpha + tile_acc;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (unsigned int row = 0u; row < q_block; ++row) {
        if (!valid[row]) {
            continue;
        }
        const unsigned int global_q = global_q_base + row;
        float* out = output + (size_t(global_q) * num_attention_heads + head) * hdim;
        const float denom = fmaxf(scalars[row * 4u + 1u], 1.0e-20f);
        for (unsigned int dim = tid; dim < hdim; dim += blockDim.x) {
            out[dim] = acc[row * hdim + dim] / denom;
        }
    }
}

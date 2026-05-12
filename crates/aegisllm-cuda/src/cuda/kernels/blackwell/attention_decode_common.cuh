// Common FlashDecoding split-K decode attention implementation, templated on
// the KV cache element type. The f16 (`unsigned short`) and fp8 e4m3
// (`unsigned char`) decode kernels are thin extern "C" wrappers around
// `decode_split_attn_impl<CacheElem>` defined here.
//
// Differences captured by the template parameter:
//   * Cache pointer type (`CacheElem*`) — sizeof(CacheElem) drives the
//     per-element byte count and the cp.async issue stride.
//   * Dequant via `dequant_cache<CacheElem>(x)` specialised below.
//   * Shared kv_pipe slot byte size = head_dim * sizeof(CacheElem).
// Everything else (Phase 1 Q·K with cp.async double-buffer, Phase 2 softmax,
// Phase 3 V·w sum with cp.async double-buffer) is dtype-agnostic.

// ----------------------------------------------------------------------------
// Dequant helpers — specialised per cache element type.
// ----------------------------------------------------------------------------

template<typename CacheElem>
__device__ __forceinline__ float dequant_cache(CacheElem x);

template<>
__device__ __forceinline__ float dequant_cache<unsigned short>(unsigned short x) {
    return f16_bits_to_float(x);
}

template<>
__device__ __forceinline__ float dequant_cache<unsigned char>(unsigned char x) {
    /* Hardware FP8 E4M3 → f32 via `cvt.rn.f16x2.e4m3x2` (sm_89+ — Ada/Hopper/
     * Blackwell). The PTX has no direct `e4m3 → f32` form; we convert to f16
     * via the pair-conversion intrinsic (using only the low half of the
     * packed pair) and then up-cast f16 → f32 with `__half2float`, which is
     * a free single-cycle conversion. Falls back to the software helper for
     * older arches. The software path uses branches + `exp2f` and dominates
     * inner-loop time at long context; the hardware path collapses to two
     * cheap instructions. */
#if __CUDA_ARCH__ >= 890
    unsigned int half_pair;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(half_pair) : "h"((unsigned short)x));
    return __half2float(__ushort_as_half((unsigned short)(half_pair & 0xFFFFu)));
#else
    return fp8_e4m3_bits_to_float(x);
#endif
}

// ----------------------------------------------------------------------------
// Templated split-K decode attention impl.
//
// Shared memory layout (allocated by caller, sized for worst case):
//   scores[max_chunk_len]                       (f32)
//   warp_partial[4]                             (f32)
//   vsum[4 * head_dim]                          (f32, Phase 3 acc)
//   kv_pipe[4 warps * 2 bufs * head_dim]        (CacheElem, cp.async K/V tile)
//
// Total bytes = (max_chunk_len + 4 + 4 * head_dim) * 4 + 8 * head_dim * sizeof(CacheElem).
// ----------------------------------------------------------------------------

template<typename CacheElem>
__device__ void decode_split_attn_impl(
    const CacheElem* __restrict__ key_cache,
    const CacheElem* __restrict__ value_cache,
    const float*     __restrict__ query,
    const unsigned int* __restrict__ p_seq_len,
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
    if (head >= num_attention_heads) return;

    const unsigned int window_start = (window_size > 0u && seq_len > window_size)
                                      ? seq_len - window_size : 0u;
    const unsigned int chunk_size  = (seq_len + split_k - 1u) / split_k;
    const unsigned int chunk_start = chunk_idx * chunk_size;
    const unsigned int out_idx     = head * split_k + chunk_idx;
    const unsigned int out_base    = out_idx * head_dim;

    if (chunk_start >= seq_len ||
        (window_size > 0u && chunk_start + chunk_size <= window_start)) {
        if (tid == 0) {
            partial_m[out_idx] = -3.402823466e38f;
            partial_l[out_idx] = 0.0f;
        }
        for (unsigned int d = tid; d < head_dim; d += blockDim.x)
            partial_acc[out_base + d] = 0.0f;
        return;
    }

    const unsigned int chunk_end = (chunk_start + chunk_size < seq_len)
                                   ? chunk_start + chunk_size : seq_len;
    const unsigned int chunk_len = chunk_end - chunk_start;

    extern __shared__ __align__(16) unsigned char smem_bytes[];
    float* scores       = reinterpret_cast<float*>(smem_bytes);
    float* warp_partial = scores + max_chunk_len;
    float* vsum         = warp_partial + 4u;
    /* kv_pipe placed after vsum. Stride per warp×buf = head_dim CacheElem. */
    CacheElem* kv_pipe  = reinterpret_cast<CacheElem*>(vsum + 4u * head_dim);
    const unsigned int kv_stride = head_dim;

    const unsigned int group   = num_attention_heads / num_kv_heads;
    const unsigned int kv_head = head / group;
    const float*       q       = query + (size_t)head * head_dim;
    const float        scale   = rsqrtf((float)head_dim);

#if __CUDA_ARCH__ >= 800
    /* cp.async issues: 16 bytes per call per participating lane.
     * Each call covers `elems_per_chunk = 16 / sizeof(CacheElem)` cache elements.
     * For head_dim coverage we need `total_chunks = head_dim / elems_per_chunk`
     * cp.async calls per K/V position, distributed across 32 lanes. If
     * total_chunks < 32, only the first total_chunks lanes participate (rest
     * sit idle on the cp.async issue). If total_chunks > 32, each lane loops. */
    constexpr unsigned int elems_per_chunk = 16u / sizeof(CacheElem);  // f16→8, fp8→16
    const unsigned int total_chunks = (head_dim + elems_per_chunk - 1u) / elems_per_chunk;
    const unsigned int chunks_per_lane = (total_chunks + 31u) / 32u;  // hd=256: 1; hd=512: 1 or 2

    auto buf_ptr = [&](unsigned int buf) -> CacheElem* {
        return kv_pipe + (size_t)(warp_id * 2u + buf) * kv_stride;
    };
    auto issue_load = [&](const CacheElem* cache,
                          unsigned int abs_pos, unsigned int buf, bool valid) {
        CacheElem* dst_base = buf_ptr(buf);
        #pragma unroll 4
        for (unsigned int li = 0u; li < chunks_per_lane; ++li) {
            const unsigned int chunk_idx_local = li * 32u + lane;
            if (chunk_idx_local >= total_chunks) break;
            const unsigned int elem_off = chunk_idx_local * elems_per_chunk;
            CacheElem* dst = dst_base + elem_off;
            unsigned int dst_smem;
            asm volatile("{ .reg .u64 smem64;\n\t"
                         "  cvta.to.shared.u64 smem64, %1;\n\t"
                         "  cvt.u32.u64 %0, smem64; }\n"
                         : "=r"(dst_smem) : "l"((const void*)dst));
            if (valid) {
                const unsigned int slot = (cache_capacity > 0u) ? (abs_pos % cache_capacity) : abs_pos;
                const CacheElem* src = cache +
                    ((size_t)slot * num_kv_heads + kv_head) * head_dim + elem_off;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                             :: "r"(dst_smem), "l"((const void*)src));
            } else {
                /* size=0 variant: zero-fills shared without dereferencing src. */
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, 0;\n"
                             :: "r"(dst_smem), "l"((const void*)cache));
            }
        }
    };
    auto cp_async_commit   = []() { asm volatile("cp.async.commit_group;\n" ::); };
    auto cp_async_wait_lt1 = []() { asm volatile("cp.async.wait_group 1;\n" ::); };
    auto cp_async_wait_lt0 = []() { asm volatile("cp.async.wait_group 0;\n" ::); };
#endif

    /* ───────── Phase 1: Q·K with cp.async-pipelined K loads ───────── */
    float warp_local_max = -3.402823466e38f;

#if __CUDA_ARCH__ >= 800
    {
        const unsigned int pos0 = warp_id;
        if (pos0 < chunk_len) {
            const unsigned int abs0 = chunk_start + pos0;
            issue_load(key_cache, abs0, 0u, abs0 >= window_start);
        }
        cp_async_commit();
    }

    for (unsigned int pos = warp_id; pos < chunk_len; pos += 4u) {
        const unsigned int abs_pos = chunk_start + pos;
        const unsigned int next_pos = pos + 4u;
        const unsigned int buf_cur = (pos >> 2u) & 1u;
        const unsigned int buf_nxt = buf_cur ^ 1u;

        if (next_pos < chunk_len) {
            const unsigned int next_abs = chunk_start + next_pos;
            issue_load(key_cache, next_abs, buf_nxt, next_abs >= window_start);
            cp_async_commit();
            cp_async_wait_lt1();
        } else {
            cp_async_wait_lt0();
        }
        __syncwarp();

        float score;
        if (abs_pos < window_start) {
            score = -3.402823466e38f;
        } else {
            const CacheElem* k = buf_ptr(buf_cur);
            float partial = 0.0f;
            for (unsigned int d = lane * 4u; d < head_dim; d += 128u) {
                partial += q[d+0u] * dequant_cache<CacheElem>(k[d+0u]);
                partial += q[d+1u] * dequant_cache<CacheElem>(k[d+1u]);
                partial += q[d+2u] * dequant_cache<CacheElem>(k[d+2u]);
                partial += q[d+3u] * dequant_cache<CacheElem>(k[d+3u]);
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
    cp_async_wait_lt0();
    __syncwarp();
#else
    /* sm_<800 fallback: synchronous K reads from global. */
    for (unsigned int pos = warp_id; pos < chunk_len; pos += 4u) {
        const unsigned int abs_pos = chunk_start + pos;
        float score;
        if (abs_pos < window_start) {
            score = -3.402823466e38f;
        } else {
            const unsigned int slot = (cache_capacity > 0u) ? (abs_pos % cache_capacity) : abs_pos;
            const CacheElem* k = key_cache +
                ((size_t)slot * num_kv_heads + kv_head) * head_dim;
            float partial = 0.0f;
            for (unsigned int d = lane * 4u; d < head_dim; d += 128u) {
                partial += q[d+0u] * dequant_cache<CacheElem>(k[d+0u]);
                partial += q[d+1u] * dequant_cache<CacheElem>(k[d+1u]);
                partial += q[d+2u] * dequant_cache<CacheElem>(k[d+2u]);
                partial += q[d+3u] * dequant_cache<CacheElem>(k[d+3u]);
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
#endif

    /* Cross-warp max reduction. */
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

    /* ───────── Phase 2: softmax weights + denominator ───────── */
    float local_sum = 0.0f;
    for (unsigned int pos = tid; pos < chunk_len; pos += blockDim.x) {
        float w = expf(scores[pos] - chunk_max);
        scores[pos] = w;
        local_sum += w;
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

    /* ───────── Phase 3: weighted V sum with cp.async-pipelined V loads ───────── */
    constexpr unsigned int MAX_D_BLOCKS = 4u;  // head_dim up to 4*128 = 512
    float acc[MAX_D_BLOCKS][4] = { {0.0f, 0.0f, 0.0f, 0.0f} };
    const unsigned int d_blocks = (head_dim + 127u) / 128u;

#if __CUDA_ARCH__ >= 800
    {
        const unsigned int pos0 = warp_id;
        if (pos0 < chunk_len) {
            const unsigned int abs0 = chunk_start + pos0;
            issue_load(value_cache, abs0, 0u, true);
        }
        cp_async_commit();
    }

    for (unsigned int pos = warp_id; pos < chunk_len; pos += 4u) {
        const unsigned int next_pos = pos + 4u;
        const unsigned int buf_cur = (pos >> 2u) & 1u;
        const unsigned int buf_nxt = buf_cur ^ 1u;

        if (next_pos < chunk_len) {
            const unsigned int next_abs = chunk_start + next_pos;
            issue_load(value_cache, next_abs, buf_nxt, true);
            cp_async_commit();
            cp_async_wait_lt1();
        } else {
            cp_async_wait_lt0();
        }
        __syncwarp();

        const CacheElem* v = buf_ptr(buf_cur);
        float w = scores[pos];
        for (unsigned int b = 0u; b < d_blocks; ++b) {
            const unsigned int d = b * 128u + lane * 4u;
            if (d >= head_dim) break;
            acc[b][0] += w * dequant_cache<CacheElem>(v[d+0u]);
            acc[b][1] += w * dequant_cache<CacheElem>(v[d+1u]);
            acc[b][2] += w * dequant_cache<CacheElem>(v[d+2u]);
            acc[b][3] += w * dequant_cache<CacheElem>(v[d+3u]);
        }
    }
    cp_async_wait_lt0();
    __syncwarp();
#else
    for (unsigned int pos = warp_id; pos < chunk_len; pos += 4u) {
        const unsigned int abs_pos_v = chunk_start + pos;
        const unsigned int slot_v = (cache_capacity > 0u) ? (abs_pos_v % cache_capacity) : abs_pos_v;
        const CacheElem* v = value_cache +
            ((size_t)slot_v * num_kv_heads + kv_head) * head_dim;
        float w = scores[pos];
        for (unsigned int b = 0u; b < d_blocks; ++b) {
            const unsigned int d = b * 128u + lane * 4u;
            if (d >= head_dim) break;
            acc[b][0] += w * dequant_cache<CacheElem>(v[d+0u]);
            acc[b][1] += w * dequant_cache<CacheElem>(v[d+1u]);
            acc[b][2] += w * dequant_cache<CacheElem>(v[d+2u]);
            acc[b][3] += w * dequant_cache<CacheElem>(v[d+3u]);
        }
    }
#endif

    for (unsigned int b = 0u; b < d_blocks; ++b) {
        const unsigned int d = b * 128u + lane * 4u;
        if (d >= head_dim) break;
        vsum[warp_id * head_dim + d + 0u] = acc[b][0];
        vsum[warp_id * head_dim + d + 1u] = acc[b][1];
        vsum[warp_id * head_dim + d + 2u] = acc[b][2];
        vsum[warp_id * head_dim + d + 3u] = acc[b][3];
    }
    __syncthreads();
    for (unsigned int d = tid; d < head_dim; d += blockDim.x) {
        partial_acc[out_base + d] = vsum[0u * head_dim + d]
                                  + vsum[1u * head_dim + d]
                                  + vsum[2u * head_dim + d]
                                  + vsum[3u * head_dim + d];
    }
}

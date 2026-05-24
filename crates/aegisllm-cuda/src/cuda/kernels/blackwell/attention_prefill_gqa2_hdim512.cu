// =============================================================================
// GQA-packed mma.sync FP16 FlashAttention prefill kernel (Stage H.2).
// head_dim = 512, dense, causal, GQA group == 2 (Gemma-4 global layers).
// =============================================================================
//
// Each block processes TWO query heads (head0 = kv_head*2, head1 = +1) that
// SHARE one kv_head. K and V are loaded into shared ONCE and each K/V register
// fragment is reused across both q-heads — halving the K/V global cp.async AND
// the shared (MIO) reads that dominate the kernel (ncu: the 1-head mma2 kernel
// is MIO/shared-load-bound). This is llama.cpp's ncols2=2 packing: deliberately
// 1 block/SM (the 2-head shared budget ~93 KiB forces it), where the win is
// per-fragment reuse, not occupancy.
//
// Reuses helpers from attention_prefill_mma_hdim512.cu (aegis_mma_load_a_m16k16,
// aegis_mma_load_b_n8k16_from_nk, aegis_mma_m16n8k16_f16, aegis_pack_f16x2,
// cp.async) and attention_prefill_mma2_hdim512.cu (aegis_mma2_m16n8k16_f16acc,
// aegis_h2_scale) — both included earlier in the concatenated TU.
//
// Shared (peak ~93 KiB, fits the 96 KiB sm_120 opt-in cap, 1 block/SM):
//   q_shared[2]   = 2 * q_block*hdim   bf16 = 64 KiB
//   kv_slab       =     kv_block*slab  bf16 =  8 KiB   (single-buf, shared)
//   s_shared[2]   = 2 * q_block*kv_block f32 = 16 KiB
//   weights_f16[2]= 2 * q_block*kv_block bf16=  4 KiB
//   scalars[2]    = 2 * q_block*3      f32  =  0.75 KiB
//
// Grid: (num_kv_heads, ceil(total_q / q_block)). Block: 256 threads (8 warps).
// =============================================================================

#if __CUDA_ARCH__ >= 800

extern "C" __global__
__launch_bounds__(256, 1)
void aegis_attention_prefill_dense_gqa2_hdim512(
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
    constexpr unsigned int hdim          = 512u;
    constexpr unsigned int q_block       = 32u;
    constexpr unsigned int kv_block      = 32u;
    constexpr unsigned int warps         = 8u;
    constexpr unsigned int block_threads = warps * 32u;             // 256
    constexpr unsigned int slab          = 128u;
    constexpr unsigned int n_slabs       = hdim / slab;             // 4
    constexpr unsigned int row_strips    = q_block / 16u;           // 2
    constexpr unsigned int kv_strips     = kv_block / 8u;           // 4
    constexpr unsigned int cols_per_warp = hdim / warps;            // 64
    constexpr unsigned int o_col_frags   = cols_per_warp / 8u;      // 8
    constexpr unsigned int o_frags       = row_strips * o_col_frags;// 16
    constexpr unsigned int warps_per_slab = slab / cols_per_warp;   // 2
    constexpr unsigned int slab_kk       = slab / 16u;              // 8
    constexpr unsigned int HP            = 2u;                      // heads packed

    const unsigned int global_q_base = blockIdx.y * q_block;
    const unsigned int tid           = threadIdx.x;
    const unsigned int lane          = tid & 31u;
    const unsigned int warp          = tid >> 5u;
    const unsigned int group         = num_attention_heads / num_kv_heads;
    // Each block processes HP=2 consecutive q-heads. They always share one
    // kv-head: for any group that is a multiple of HP, the pair {2b, 2b+1}
    // falls in the same kv-group (kv-group boundaries are at multiples of
    // `group`, which is even, so an even/odd consecutive pair never straddles
    // one). Gemma-4 global: group=8 (16 q / 2 kv) → 8 blocks, 4 per kv-head.
    const unsigned int head0   = blockIdx.x * HP;   // q-heads head0, head0+1
    const unsigned int kv_head = head0 / group;     // shared kv-head
    if (head_dim != hdim || (group % HP) != 0u || head0 + HP > num_attention_heads
        || blockDim.x < block_threads) {
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
        (block_min_visible_raw / kv_block) * kv_block;

    // --- shared layout ---
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned short* q_shared    = reinterpret_cast<unsigned short*>(smem);     // [HP][q_block*hdim]
    unsigned short* kv_slab     = q_shared + HP * q_block * hdim;              // [kv_block*slab]
    float*          s_shared    = reinterpret_cast<float*>(kv_slab + kv_block * slab); // [HP][q_block*kv_block]
    unsigned short* weights_f16 = reinterpret_cast<unsigned short*>(s_shared + HP * q_block * kv_block); // [HP][q_block*kv_block]
    float*          scalars     = reinterpret_cast<float*>(weights_f16 + HP * q_block * kv_block);       // [HP][q_block*3]

    const float scale  = rsqrtf(float(hdim));
    const float log2e  = 1.4426950408889634f;
    const float neg_inf = -3.402823466e38f;

    // --- load Q tiles (both heads) into shared, persistent ---
    {
        constexpr unsigned int halfs_per_vec = sizeof(uint4) / sizeof(unsigned short);
        constexpr unsigned int q_vecs = q_block * hdim / halfs_per_vec;   // per head
        const uint4 zero_vec = make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
        for (unsigned int hp = 0u; hp < HP; ++hp) {
            uint4* q_shared_vec = reinterpret_cast<uint4*>(q_shared + hp * q_block * hdim);
            const unsigned int head = head0 + hp;
            for (unsigned int vec = tid; vec < q_vecs; vec += block_threads) {
                const unsigned int idx = vec * halfs_per_vec;
                const unsigned int row = idx / hdim;
                const unsigned int dim = idx - row * hdim;
                const unsigned int global_q = global_q_base + row;
                q_shared_vec[vec] = global_q < total_q
                    ? *reinterpret_cast<const uint4*>(
                          query + (size_t(global_q) * num_attention_heads + head) * hdim + dim)
                    : zero_vec;
            }
        }
    }
    for (unsigned int i = tid; i < HP * q_block; i += block_threads) {
        scalars[i * 3u + 0u] = neg_inf;   // m
        scalars[i * 3u + 1u] = 0.0f;      // l
        scalars[i * 3u + 2u] = 0.0f;      // alpha
    }

    // --- persistent register-resident O accumulator (f16), per packed head ---
    unsigned o_acc[HP][o_frags][2];
#pragma unroll
    for (unsigned int hp = 0u; hp < HP; ++hp)
#pragma unroll
        for (unsigned int f = 0u; f < o_frags; ++f) {
            o_acc[hp][f][0] = 0u;
            o_acc[hp][f][1] = 0u;
        }

    // --- cp.async slab staging (K/V shared across both heads via kv_head) ---
    auto stage_slab = [&] (const unsigned short* __restrict__ cache,
                           unsigned int tile_start, unsigned int slab_idx,
                           unsigned short* buf, unsigned int tile_count) {
        constexpr unsigned int halfs_per_chunk = 8u;
        constexpr unsigned int chunks_per_row  = slab / halfs_per_chunk;       // 16
        constexpr unsigned int passes          = (kv_block * chunks_per_row) / block_threads; // 2
        const unsigned int hdim_base = slab_idx * slab;
#pragma unroll
        for (unsigned int p = 0u; p < passes; ++p) {
            const unsigned int chunk = p * block_threads + tid;
            const unsigned int row   = chunk / chunks_per_row;
            const unsigned int hoff  = (chunk % chunks_per_row) * halfs_per_chunk;
            const bool valid = row < tile_count;
            const unsigned int pos = tile_start + row;
            unsigned int dst = aegis_mma_cvta_smem(&buf[row * slab + hoff]);
            if (valid) {
                const size_t off =
                    (size_t(kv_slot(pos, cache_capacity)) * num_kv_heads + kv_head) * hdim
                    + hdim_base + hoff;
                aegis_mma_cp_async_16(dst, cache + off);
            } else {
                aegis_mma_cp_async_zero_16(dst);
            }
        }
    };

    const unsigned int n_kiters = (block_max_visible > block_min_tile_start)
        ? ((block_max_visible - block_min_tile_start + kv_block - 1u) / kv_block)
        : 0u;

    // Q*K warp -> S tile (per head): row_strips(2) * kv_strips(4) = 8 = warps.
    const unsigned int qk_row = warp / kv_strips;                 // 0..1
    const unsigned int qk_kv  = warp % kv_strips;                 // 0..3
    // P*V warp -> hdim slice.
    const unsigned int o_slab     = warp / warps_per_slab;        // 0..3
    const unsigned int o_col_base = (warp % warps_per_slab) * cols_per_warp;

    // Prologue: stage K slab 0.
    if (n_kiters > 0u) {
        const unsigned int tc0 = min(kv_block, block_max_visible - block_min_tile_start);
        stage_slab(key_cache, block_min_tile_start, 0u, kv_slab, tc0);
        aegis_mma_cp_commit();
    }
    __syncthreads();

    for (unsigned int it = 0u; it < n_kiters; ++it) {
        const unsigned int tile_start = block_min_tile_start + it * kv_block;
        const unsigned int tile_count = min(kv_block, block_max_visible - tile_start);

        // ======================= Q*K -> S (both heads, K reused) ============
        float s_acc[HP][4];
#pragma unroll
        for (unsigned int hp = 0u; hp < HP; ++hp) {
            s_acc[hp][0] = 0.0f; s_acc[hp][1] = 0.0f;
            s_acc[hp][2] = 0.0f; s_acc[hp][3] = 0.0f;
        }
        for (unsigned int sl = 0u; sl < n_slabs; ++sl) {
            unsigned short* k_buf = kv_slab;
            aegis_mma_cp_wait_all();
            __syncthreads();                       // RAW
#pragma unroll
            for (unsigned int kk = 0u; kk < slab_kk; ++kk) {
                const unsigned int a_row_base = qk_row * 16u;
                const unsigned int a_col_base = sl * slab + kk * 16u;
                const unsigned int b_row_base = qk_kv * 8u;
                const unsigned int b_col_base = kk * 16u;
                // K fragment loaded ONCE, reused across both q-heads.
                unsigned int b0, b1;
                aegis_mma_load_b_n8k16_from_nk(b0, b1,
                    k_buf + b_row_base * slab + b_col_base, slab);
#pragma unroll
                for (unsigned int hp = 0u; hp < HP; ++hp) {
                    unsigned int a0, a1, a2, a3;
                    aegis_mma_load_a_m16k16(a0, a1, a2, a3,
                        q_shared + hp * (q_block * hdim) + a_row_base * hdim + a_col_base, hdim);
                    aegis_mma_m16n8k16_f16(
                        s_acc[hp][0], s_acc[hp][1], s_acc[hp][2], s_acc[hp][3],
                        a0, a1, a2, a3, b0, b1);
                }
            }
            __syncthreads();                       // WAR
            if (sl + 1u < n_slabs) {
                stage_slab(key_cache, tile_start, sl + 1u, kv_slab, tile_count);
                aegis_mma_cp_commit();
            }
        }

        // Spill S tiles to s_shared[hp].
        {
            const unsigned int r_upper = qk_row * 16u + (lane >> 2);
            const unsigned int r_lower = r_upper + 8u;
            const unsigned int c_base  = qk_kv * 8u + (lane & 3u) * 2u;
#pragma unroll
            for (unsigned int hp = 0u; hp < HP; ++hp) {
                float* s = s_shared + hp * (q_block * kv_block);
                s[r_upper * kv_block + c_base + 0u] = s_acc[hp][0];
                s[r_upper * kv_block + c_base + 1u] = s_acc[hp][1];
                s[r_lower * kv_block + c_base + 0u] = s_acc[hp][2];
                s[r_lower * kv_block + c_base + 1u] = s_acc[hp][3];
            }
        }
        __syncthreads();

        // ======================= online softmax (per head) ==================
        constexpr unsigned int q_rows_per_warp = q_block / warps;   // 4
        constexpr unsigned int kv_per_lane      = kv_block / 32u;    // 1
#pragma unroll
        for (unsigned int hp = 0u; hp < HP; ++hp) {
            float* s   = s_shared + hp * (q_block * kv_block);
            float* sc_ = scalars  + hp * (q_block * 3u);
            unsigned short* w_ = weights_f16 + hp * (q_block * kv_block);
#pragma unroll
            for (unsigned int rr = 0u; rr < q_rows_per_warp; ++rr) {
                const unsigned int row = warp + rr * warps;
                const unsigned int global_q = global_q_base + row;
                const bool valid_q = global_q < total_q;
                const unsigned int visible_len = valid_q
                    ? min(context_len, start_position + global_q + 1u) : 0u;
                const unsigned int row_min_visible = (window_size > 0u
                    && start_position + global_q + 1u > window_size)
                    ? (start_position + global_q + 1u - window_size) : 0u;
                const float old_m = sc_[row * 3u + 0u];
                const float old_l = sc_[row * 3u + 1u];
                float sc[kv_per_lane];
                float tile_m = neg_inf;
#pragma unroll
                for (unsigned int c = 0u; c < kv_per_lane; ++c) {
                    const unsigned int col = lane + c * 32u;
                    const unsigned int pos = tile_start + col;
                    sc[c] = (valid_q && col < tile_count && pos < visible_len
                             && pos >= row_min_visible)
                        ? s[row * kv_block + col] * scale
                        : neg_inf;
                    tile_m = fmaxf(tile_m, sc[c]);
                }
                tile_m = aegis_warp_reduce_max(tile_m);
                const float new_m = fmaxf(old_m, tile_m);
                float w[kv_per_lane];
                float tile_l = 0.0f;
#pragma unroll
                for (unsigned int c = 0u; c < kv_per_lane; ++c) {
                    w[c] = (sc[c] > -3.0e38f) ? exp2f((sc[c] - new_m) * log2e) : 0.0f;
                    tile_l += w[c];
                    const unsigned int col = lane + c * 32u;
                    w_[row * kv_block + col] =
                        __half_as_ushort(__float2half_rn(w[c]));
                }
                tile_l = aegis_warp_reduce_sum(tile_l);
                if (lane == 0u) {
                    const float alpha = old_l > 0.0f ? exp2f((old_m - new_m) * log2e) : 1.0f;
                    sc_[row * 3u + 0u] = new_m;
                    sc_[row * 3u + 1u] = old_l * alpha + tile_l;
                    sc_[row * 3u + 2u] = alpha;
                }
            }
        }
        __syncthreads();

        // ======================= rescale O (per head) =======================
        {
            const unsigned int r_up = (lane >> 2);
#pragma unroll
            for (unsigned int hp = 0u; hp < HP; ++hp) {
                const float* sc_ = scalars + hp * (q_block * 3u);
#pragma unroll
                for (unsigned int rs = 0u; rs < row_strips; ++rs) {
                    const float a_up = sc_[(rs * 16u + r_up) * 3u + 2u];
                    const float a_lo = sc_[(rs * 16u + r_up + 8u) * 3u + 2u];
#pragma unroll
                    for (unsigned int cf = 0u; cf < o_col_frags; ++cf) {
                        unsigned* d = o_acc[hp][rs * o_col_frags + cf];
                        d[0] = aegis_h2_scale(d[0], a_up);
                        d[1] = aegis_h2_scale(d[1], a_lo);
                    }
                }
            }
        }

        // ======================= P*V -> O (both heads, V reused) ============
        stage_slab(value_cache, tile_start, 0u, kv_slab, tile_count);
        aegis_mma_cp_commit();
        for (unsigned int sl = 0u; sl < n_slabs; ++sl) {
            unsigned short* v_buf = kv_slab;
            aegis_mma_cp_wait_all();
            __syncthreads();                       // RAW
            if (o_slab == sl) {
                constexpr unsigned int pv_k_strips = kv_block / 16u;  // 2
#pragma unroll
                for (unsigned int ks = 0u; ks < pv_k_strips; ++ks) {
#pragma unroll
                    for (unsigned int cf = 0u; cf < o_col_frags; ++cf) {
                        const unsigned int v_row_base = ks * 16u;
                        const unsigned int v_col_base = o_col_base + cf * 8u;
                        // V fragment computed ONCE, reused across both q-heads.
                        const unsigned int n_idx = (lane >> 2);
                        const unsigned int k_lo  = (lane & 3u) << 1;
                        const unsigned int k_hi  = k_lo + 8u;
                        const unsigned short* src_v = v_buf + v_row_base * slab + v_col_base;
                        const unsigned int v0 = aegis_pack_f16x2(
                            src_v[(k_lo + 0u) * slab + n_idx],
                            src_v[(k_lo + 1u) * slab + n_idx]);
                        const unsigned int v1 = aegis_pack_f16x2(
                            src_v[(k_hi + 0u) * slab + n_idx],
                            src_v[(k_hi + 1u) * slab + n_idx]);
#pragma unroll
                        for (unsigned int hp = 0u; hp < HP; ++hp) {
                            unsigned short* w_ = weights_f16 + hp * (q_block * kv_block);
#pragma unroll
                            for (unsigned int rs = 0u; rs < row_strips; ++rs) {
                                const unsigned int p_row_base = rs * 16u;
                                const unsigned int p_col_base = ks * 16u;
                                unsigned int p0, p1, p2, p3;
                                aegis_mma_load_a_m16k16(p0, p1, p2, p3,
                                    w_ + p_row_base * kv_block + p_col_base, kv_block);
                                unsigned* d = o_acc[hp][rs * o_col_frags + cf];
                                aegis_mma2_m16n8k16_f16acc(
                                    d[0], d[1], p0, p1, p2, p3, v0, v1);
                            }
                        }
                    }
                }
            }
            __syncthreads();                       // WAR
            if (sl + 1u < n_slabs) {
                stage_slab(value_cache, tile_start, sl + 1u, kv_slab, tile_count);
                aegis_mma_cp_commit();
            }
        }

        // Prologue: stage next iter's K slab 0.
        if (it + 1u < n_kiters) {
            const unsigned int next_start = tile_start + kv_block;
            const unsigned int next_tc = min(kv_block, block_max_visible - next_start);
            stage_slab(key_cache, next_start, 0u, kv_slab, next_tc);
            aegis_mma_cp_commit();
        }
    }

    // ============================ epilogue (both heads) =====================
    __syncthreads();
#pragma unroll
    for (unsigned int hp = 0u; hp < HP; ++hp) {
        const unsigned int head = head0 + hp;
        const float* sc_ = scalars + hp * (q_block * 3u);
#pragma unroll
        for (unsigned int rs = 0u; rs < row_strips; ++rs) {
#pragma unroll
            for (unsigned int cf = 0u; cf < o_col_frags; ++cf) {
                const unsigned* d = o_acc[hp][rs * o_col_frags + cf];
                const __half2 c0 = *reinterpret_cast<const __half2*>(&d[0]);
                const __half2 c1 = *reinterpret_cast<const __half2*>(&d[1]);
                const unsigned int r_upper = rs * 16u + (lane >> 2);
                const unsigned int r_lower = r_upper + 8u;
                const unsigned int c_base  = o_slab * slab + o_col_base + cf * 8u + (lane & 3u) * 2u;
                const unsigned int gq_upper = global_q_base + r_upper;
                const unsigned int gq_lower = global_q_base + r_lower;
                if (gq_upper < total_q) {
                    const float inv = 1.0f / fmaxf(sc_[r_upper * 3u + 1u], 1.0e-20f);
                    output[(size_t(gq_upper) * num_attention_heads + head) * hdim + c_base + 0u] =
                        __low2float(c0) * inv;
                    output[(size_t(gq_upper) * num_attention_heads + head) * hdim + c_base + 1u] =
                        __high2float(c0) * inv;
                }
                if (gq_lower < total_q) {
                    const float inv = 1.0f / fmaxf(sc_[r_lower * 3u + 1u], 1.0e-20f);
                    output[(size_t(gq_lower) * num_attention_heads + head) * hdim + c_base + 0u] =
                        __low2float(c1) * inv;
                    output[(size_t(gq_lower) * num_attention_heads + head) * hdim + c_base + 1u] =
                        __high2float(c1) * inv;
                }
            }
        }
    }
}

#endif  // __CUDA_ARCH__ >= 800

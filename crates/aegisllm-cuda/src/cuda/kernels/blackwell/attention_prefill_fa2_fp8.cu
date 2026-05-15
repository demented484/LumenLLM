// =============================================================================
// FlashAttention-2 style prefill attention kernel for head_dim=512 — FP8 KV.
// =============================================================================
//
// Opt-in (AEGIS_ATTN_FP8=1) FP8-E4M3 KV-cache variant of
// `aegis_attention_prefill_dense_fa2_hdim512` (the q32 FA-2 kernel in
// attention_prefill_fa2.cu). It is selected ONLY when:
//   * the KV cache is FP8 (KvCacheQuantization::Fp8), AND
//   * head_dim == 512.
// Default OFF -> the prefill path is bit-equivalent to main.
//
// -----------------------------------------------------------------------------
// What FP8 changes vs the BF16 FA-2 kernel
// -----------------------------------------------------------------------------
// The BF16 FA-2 kernel reads an auxiliary BF16 KV cache (`prefill_f16_keys` /
// `prefill_f16_values`). This kernel reads the PERSISTENT FP8-E4M3 cache
// (`unsigned char`) directly — the same buffers the FP8 decode path
// (`decode_split_attn_impl<unsigned char>`) consumes. Benefits:
//   * KV HBM traffic in the prefill mainloop is HALVED (e4m3 = 1 byte/elem
//     vs BF16's 2). The cp.async slab transfers move half the bytes.
//   * The BF16 auxiliary cache becomes redundant for the head_dim=512 layers
//     (the dispatcher routes those layers straight to the FP8 cache).
//
// -----------------------------------------------------------------------------
// Why this is the dequant-to-half variant (option b), NOT raw FP8 MMA
// -----------------------------------------------------------------------------
// The FA-2 kernel's S accumulator, the online-softmax alpha rescale helper
// `aegis_scale_wmma_accumulator_m16n16_rows`, and the epilogue
// `store_matrix_sync` are ALL built around the `nvcuda::wmma` m16n16k16
// fragment layout. The SM120 FP8 e4m3 MMA is a raw
// `mma.sync.aligned.kind::f8f6f4.m16n8k32` instruction with an entirely
// different m16n8 fragment layout that `nvcuda::wmma` cannot express.
// Swapping it in is a full kernel rewrite (re-deriving every fragment->element
// mapping) that cannot be NVRTC-verified in a build-only workflow. To ship a
// kernel that is provably numerically equivalent to the proven BF16 FA-2
// path, this variant:
//   1. cp.async-loads the e4m3 K/V slab into a small staging buffer,
//   2. dequants e4m3 -> half IN SHARED MEMORY (HW `cvt.rn.f16x2.e4m3x2`,
//      the exact converter the FP8 decode path uses),
//   3. runs the IDENTICAL BF16 WMMA Q*K / softmax / P*V math.
// The accuracy cost is therefore exactly the e4m3 KV-cache rounding already
// accepted by `type-k: fp8` decode — no NEW precision loss vs FP8 decode.
//
// -----------------------------------------------------------------------------
// Shared-memory layout (peak ~76.5 KiB, identical footprint to BF16 FA-2)
// -----------------------------------------------------------------------------
//   q_shared      = 32*512*2   = 32 KiB   half  (Q tile, persistent)
//   e4m3_stage[2] = 2*64*128*1 = 16 KiB   uchar (cp.async double-buffered)
//   half_slab     = 64*128*2   = 16 KiB   half  (dequant target, WMMA reads)
//   s_shared      = 32*64*4    =  8 KiB   float (S tile, then P weights area)
//   weights_h     = 32*64*2    =  4 KiB   half  (P in half for the P*V WMMA)
//   scalars       = 32*3*4     =  0.4 KiB float
//                                ---------
//                                ~76.4 KiB  (fits the 96 KiB sm_120 cap)
//
// The e4m3 stage is double-buffered so slab sl+1's cp.async overlaps the
// dequant+WMMA of slab sl. half_slab is single-buffered: the dequant of slab
// sl writes it and the WMMA of slab sl reads it within the same iteration,
// separated by one __syncthreads(); slab sl+1's dequant cannot start until
// slab sl's WMMA has finished (the next iteration's top barrier orders that).
//
// Grid: (num_attention_heads, ceil(total_q / q_block)).  Block: 512 threads.
// =============================================================================

#if __CUDA_ARCH__ >= 800

extern "C" __global__
__launch_bounds__(512, 1)
void aegis_attention_prefill_dense_fa2_hdim512_fp8(
    const unsigned char* __restrict__ key_cache,
    const unsigned char* __restrict__ value_cache,
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
    using namespace nvcuda;
    constexpr unsigned int hdim          = 512u;
    constexpr unsigned int q_block       = 32u;
    constexpr unsigned int kv_block      = 64u;
    constexpr unsigned int warps         = 16u;
    constexpr unsigned int slab          = 128u;
    constexpr unsigned int n_slabs       = hdim / slab;          // 4
    constexpr unsigned int row_strips    = q_block / 16u;        // 2
    constexpr unsigned int kv_strips     = kv_block / 16u;       // 4
    constexpr unsigned int cols_per_warp = hdim / warps;         // 32
    constexpr unsigned int o_col_frags   = cols_per_warp / 16u;  // 2
    constexpr unsigned int o_frags       = row_strips * o_col_frags; // 4
    constexpr unsigned int warps_per_slab = slab / cols_per_warp;    // 4
    constexpr unsigned int slab_kk       = slab / 16u;           // 8

    const unsigned int head          = blockIdx.x;
    const unsigned int global_q_base = blockIdx.y * q_block;
    const unsigned int tid           = threadIdx.x;
    const unsigned int lane          = tid & 31u;
    const unsigned int warp          = tid >> 5u;
    if (head_dim != hdim || head >= num_attention_heads || blockDim.x < warps * 32u) {
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
    unsigned short* q_shared  = reinterpret_cast<unsigned short*>(smem);
    // e4m3 staging: 2 buffers of kv_block*slab bytes (cp.async double-buffer).
    unsigned char*  e4m3_stage = reinterpret_cast<unsigned char*>(
        q_shared + q_block * hdim);
    // half_slab: single dequant target, kv_block*slab halfs.
    unsigned short* half_slab  = reinterpret_cast<unsigned short*>(
        e4m3_stage + 2u * kv_block * slab);
    float*          s_shared  = reinterpret_cast<float*>(
        half_slab + kv_block * slab);
    half*           weights_h = reinterpret_cast<half*>(s_shared + q_block * kv_block);
    float*          scalars   = reinterpret_cast<float*>(weights_h + q_block * kv_block);
    // Epilogue scratch overlays e4m3_stage+half_slab (free once the mainloop
    // ends): 16 warps * 256 floats = 16 KiB <= (16 KiB e4m3_stage + 16 KiB
    // half_slab) = 32 KiB.
    float*          o_scratch = reinterpret_cast<float*>(e4m3_stage);

    const unsigned int group   = num_attention_heads / num_kv_heads;
    const unsigned int kv_head = head / group;
    const float scale  = rsqrtf(float(hdim));
    const float log2e  = 1.4426950408889634f;
    const float neg_inf = -3.402823466e38f;

    // --- load Q tile once (whole hdim, persistent) — Q stays BF16/half ---
    {
        constexpr unsigned int halfs_per_vec = sizeof(uint4) / sizeof(unsigned short);
        constexpr unsigned int q_vecs = q_block * hdim / halfs_per_vec;
        uint4* q_shared_vec = reinterpret_cast<uint4*>(q_shared);
        const uint4 zero_vec = make_uint4(0u, 0u, 0u, 0u);
        for (unsigned int vec = tid; vec < q_vecs; vec += blockDim.x) {
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
    for (unsigned int row = tid; row < q_block; row += blockDim.x) {
        scalars[row * 3u + 0u] = neg_inf;   // m
        scalars[row * 3u + 1u] = 0.0f;      // l
        scalars[row * 3u + 2u] = 0.0f;      // alpha
    }

    // --- persistent register-resident O accumulator ---
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag[o_frags];
#pragma unroll
    for (unsigned int f = 0u; f < o_frags; ++f) {
        wmma::fill_fragment(o_frag[f], 0.0f);
    }

    // --- cp.async helpers ---
    auto cvt_smem = [] (const void* p) -> unsigned int {
        unsigned int s;
        asm volatile("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }\n"
                     : "=r"(s) : "l"(p));
        return s;
    };
    auto cp_async_16 = [] (unsigned int dst, const void* src) {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
    };
    auto cp_async_zero_16 = [] (unsigned int dst) {
        const unsigned long long z = 0ULL;
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16, 0;\n"
                     :: "r"(dst), "l"((const void*)&z));
    };
    auto cp_commit   = [] () { asm volatile("cp.async.commit_group;\n" ::); };
    auto cp_wait_all = [] () { asm volatile("cp.async.wait_group 0;\n" ::); };

    // Stage one 128-wide hdim slab of e4m3 K or V for a KV block into an
    // e4m3 staging buffer (kv_block * slab BYTES = 8 KiB). Each thread copies
    // a 16-byte chunk = 16 e4m3 elements; kv_block*slab = 8192 bytes = 512
    // chunks, 512 threads -> 1 chunk each.
    auto stage_slab_e4m3 = [&] (const unsigned char* __restrict__ cache,
                                unsigned int tile_start, unsigned int slab_idx,
                                unsigned char* buf, unsigned int tile_count) {
        constexpr unsigned int bytes_per_chunk = 16u;
        constexpr unsigned int chunks_per_row  = slab / bytes_per_chunk;        // 8
        constexpr unsigned int total_chunks    = kv_block * chunks_per_row;     // 512
        const unsigned int hdim_base = slab_idx * slab;
        const unsigned int chunk = tid;                                        // 0..511
        if (chunk >= total_chunks) { return; }
        const unsigned int row  = chunk / chunks_per_row;                      // 0..63
        const unsigned int hoff = (chunk % chunks_per_row) * bytes_per_chunk;
        const bool valid = row < tile_count;
        const unsigned int pos = tile_start + row;
        unsigned int dst = cvt_smem(&buf[row * slab + hoff]);
        if (valid) {
            const size_t off =
                (size_t(kv_slot(pos, cache_capacity)) * num_kv_heads + kv_head) * hdim
                + hdim_base + hoff;
            cp_async_16(dst, cache + off);
        } else {
            cp_async_zero_16(dst);
        }
    };

    // Dequant the e4m3 staging buffer (kv_block*slab bytes) into the half slab
    // (kv_block*slab halfs) using the HW `cvt.rn.f16x2.e4m3x2` pair converter
    // — the exact path the FP8 decode kernel uses. Each thread converts a pair
    // of e4m3 bytes -> a packed half2; kv_block*slab = 8192 bytes = 4096 pairs,
    // 512 threads -> 8 pairs each.
    auto dequant_slab = [&] (const unsigned char* __restrict__ src,
                             unsigned short* __restrict__ dst) {
        constexpr unsigned int pairs = (kv_block * slab) / 2u;                  // 4096
        const unsigned short* src16 = reinterpret_cast<const unsigned short*>(src);
        unsigned int* dst32 = reinterpret_cast<unsigned int*>(dst);
#pragma unroll
        for (unsigned int p = tid; p < pairs; p += 512u) {
            // src16[p] packs two e4m3 bytes (lo = elem 2p, hi = elem 2p+1).
            // `cvt.rn.f16x2.e4m3x2` converts both to a half2 in one shot.
            unsigned int half_pair;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(half_pair) : "h"(src16[p]));
            dst32[p] = half_pair;
        }
    };

    const unsigned int n_kiters = (block_max_visible > block_min_tile_start)
        ? ((block_max_visible - block_min_tile_start + kv_block - 1u) / kv_block)
        : 0u;

    // Q*K warp -> S tile assignment (warps 0..7 active).
    const bool       qk_active = warp < (row_strips * kv_strips);  // 8
    const unsigned int qk_row  = warp / kv_strips;                 // 0..1
    const unsigned int qk_kv   = warp % kv_strips;                 // 0..3
    // P*V warp -> hdim slab / column ownership (all 16 warps).
    const unsigned int o_slab     = warp / warps_per_slab;         // 0..3
    const unsigned int o_col_base = (warp % warps_per_slab) * cols_per_warp;

    // e4m3 staging double-buffer pointers.
    auto e4m3_buf = [&] (unsigned int b) -> unsigned char* {
        return e4m3_stage + b * (kv_block * slab);
    };

    // Prologue: stage e4m3 K slab 0 of the first KV block.
    if (n_kiters > 0u) {
        const unsigned int tc0 = min(kv_block, block_max_visible - block_min_tile_start);
        stage_slab_e4m3(key_cache, block_min_tile_start, 0u, e4m3_buf(0u), tc0);
        cp_commit();
    }
    __syncthreads();

    // ----------------------------- mainloop --------------------------------
    for (unsigned int it = 0u; it < n_kiters; ++it) {
        const unsigned int tile_start = block_min_tile_start + it * kv_block;
        const unsigned int tile_count = min(kv_block, block_max_visible - tile_start);

        // ======================= Q*K -> S =======================
        // Per slab iteration: wait for slab sl's e4m3 cp.async, barrier,
        // prefetch slab sl+1's e4m3 (overlaps the dequant+WMMA below),
        // dequant slab sl e4m3->half, barrier, WMMA(sl).
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag;
        wmma::fill_fragment(s_frag, 0.0f);
        for (unsigned int sl = 0u; sl < n_slabs; ++sl) {
            const unsigned char* k_e4m3 = e4m3_buf(sl & 1u);
            cp_wait_all();     // slab sl's e4m3 cp.async landed (issuing-thread order)
            __syncthreads();   // slab sl's e4m3 visible block-wide AND slab sl-1
                               // WMMA done -> half_slab safe to overwrite.
            // Prefetch slab sl+1's e4m3 into the other staging buffer; its
            // cp.async overlaps the dequant + WMMA of slab sl.
            if (sl + 1u < n_slabs) {
                stage_slab_e4m3(key_cache, tile_start, sl + 1u,
                                e4m3_buf((sl + 1u) & 1u), tile_count);
                cp_commit();
            }
            // Dequant slab sl e4m3 -> half_slab.
            dequant_slab(k_e4m3, half_slab);
            __syncthreads();   // half_slab fully written before WMMA reads it.
            if (qk_active) {
                // s_frag[16q x 16kv] += Q[16q x 128] . K[16kv x 128]^T over
                // this slab.
#pragma unroll
                for (unsigned int kk = 0u; kk < slab_kk; ++kk) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
                    wmma::load_matrix_sync(a_frag,
                        reinterpret_cast<const half*>(
                            q_shared + qk_row * 16u * hdim + sl * slab + kk * 16u), hdim);
                    wmma::load_matrix_sync(b_frag,
                        reinterpret_cast<const half*>(
                            half_slab + qk_kv * 16u * slab + kk * 16u), slab);
                    wmma::mma_sync(s_frag, a_frag, b_frag, s_frag);
                }
            }
        }
        // Store each warp's complete S tile to s_shared [q_block, kv_block].
        if (qk_active) {
            wmma::store_matrix_sync(
                s_shared + qk_row * 16u * kv_block + qk_kv * 16u,
                s_frag, kv_block, wmma::mem_row_major);
        }
        __syncthreads();

        // ======================= online softmax =======================
        // Each warp owns 2 q rows: row = warp, warp + 16.
#pragma unroll
        for (unsigned int rr = 0u; rr < 2u; ++rr) {
            const unsigned int row = warp + rr * warps;          // 0..31
            const unsigned int global_q = global_q_base + row;
            const bool valid_q = global_q < total_q;
            const unsigned int visible_len = valid_q
                ? min(context_len, start_position + global_q + 1u) : 0u;
            const unsigned int row_min_visible = (window_size > 0u
                && start_position + global_q + 1u > window_size)
                ? (start_position + global_q + 1u - window_size) : 0u;
            const float old_m = scalars[row * 3u + 0u];
            const float old_l = scalars[row * 3u + 1u];
            float sc[2];
#pragma unroll
            for (unsigned int c = 0u; c < 2u; ++c) {
                const unsigned int col = lane + c * 32u;
                const unsigned int pos = tile_start + col;
                sc[c] = (valid_q && col < tile_count && pos < visible_len
                         && pos >= row_min_visible)
                    ? s_shared[row * kv_block + col] * scale
                    : neg_inf;
            }
            float tile_m = fmaxf(sc[0], sc[1]);
            tile_m = aegis_warp_reduce_max(tile_m);
            const float new_m = fmaxf(old_m, tile_m);
            float w[2];
            float tile_l = 0.0f;
#pragma unroll
            for (unsigned int c = 0u; c < 2u; ++c) {
                w[c] = (sc[c] > -3.0e38f) ? exp2f((sc[c] - new_m) * log2e) : 0.0f;
                tile_l += w[c];
                const unsigned int col = lane + c * 32u;
                weights_h[row * kv_block + col] = __float2half_rn(w[c]);
            }
            tile_l = aegis_warp_reduce_sum(tile_l);
            if (lane == 0u) {
                const float alpha = old_l > 0.0f ? exp2f((old_m - new_m) * log2e) : 1.0f;
                scalars[row * 3u + 0u] = new_m;
                scalars[row * 3u + 1u] = old_l * alpha + tile_l;
                scalars[row * 3u + 2u] = alpha;
            }
        }
        __syncthreads();

        // ======================= rescale O (in registers) =======================
#pragma unroll
        for (unsigned int rs = 0u; rs < row_strips; ++rs) {
#pragma unroll
            for (unsigned int cf = 0u; cf < o_col_frags; ++cf) {
                aegis_scale_wmma_accumulator_m16n16_rows(
                    o_frag[rs * o_col_frags + cf], scalars, rs * 16u);
            }
        }

        // ======================= P*V -> O =======================
        // weights_h holds P [q_block, kv_block] in half. Stream e4m3 V slabs:
        // stage e4m3 V slab 0, then per slab: wait, barrier, prefetch sl+1,
        // dequant sl, barrier, WMMA(sl).
        stage_slab_e4m3(value_cache, tile_start, 0u, e4m3_buf(0u), tile_count);
        cp_commit();
        for (unsigned int sl = 0u; sl < n_slabs; ++sl) {
            const unsigned char* v_e4m3 = e4m3_buf(sl & 1u);
            cp_wait_all();     // slab sl's e4m3 cp.async landed
            __syncthreads();   // e4m3 visible block-wide AND slab sl-1 WMMA done
            if (sl + 1u < n_slabs) {
                stage_slab_e4m3(value_cache, tile_start, sl + 1u,
                                e4m3_buf((sl + 1u) & 1u), tile_count);
                cp_commit();
            }
            dequant_slab(v_e4m3, half_slab);
            __syncthreads();   // half_slab fully written before WMMA reads it
            if (o_slab == sl) {
                // O[16q x 32] += P[16q x 64] . V[64 x 32] for this warp's cols.
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;
#pragma unroll
                for (unsigned int cf = 0u; cf < o_col_frags; ++cf) {
                    const unsigned int vcol = o_col_base + cf * 16u;
#pragma unroll
                    for (unsigned int ks = 0u; ks < kv_strips; ++ks) {
                        wmma::load_matrix_sync(v_frag,
                            reinterpret_cast<const half*>(
                                half_slab + ks * 16u * slab + vcol), slab);
                        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;
#pragma unroll
                        for (unsigned int rs = 0u; rs < row_strips; ++rs) {
                            wmma::load_matrix_sync(p_frag,
                                weights_h + rs * 16u * kv_block + ks * 16u, kv_block);
                            wmma::mma_sync(o_frag[rs * o_col_frags + cf], p_frag, v_frag,
                                           o_frag[rs * o_col_frags + cf]);
                        }
                    }
                }
            }
        }

        // Prologue for next iter: stage its e4m3 K slab 0. The next iter's
        // Q*K slab-0 cp_wait_all + top barrier provides RAW; the WAR (e4m3
        // buffer 0 last read by the P*V slab sl=2 dequant) is covered by P*V
        // slab sl=3's top barrier which ran before this stage.
        if (it + 1u < n_kiters) {
            const unsigned int next_start = tile_start + kv_block;
            const unsigned int next_tc = min(kv_block, block_max_visible - next_start);
            stage_slab_e4m3(key_cache, next_start, 0u, e4m3_buf(0u), next_tc);
            cp_commit();
        }
    }

    // ============================ epilogue ============================
    // O[row, col] /= l[row]; write to global output.
    __syncthreads();
    float* warp_scratch = o_scratch + warp * 256u;   // 16 warps * 256 = 16 KiB
#pragma unroll
    for (unsigned int rs = 0u; rs < row_strips; ++rs) {
#pragma unroll
        for (unsigned int cf = 0u; cf < o_col_frags; ++cf) {
            wmma::store_matrix_sync(warp_scratch,
                o_frag[rs * o_col_frags + cf], 16u, wmma::mem_row_major);
#pragma unroll
            for (unsigned int e = lane; e < 256u; e += 32u) {
                const unsigned int r = e >> 4u;          // 0..15
                const unsigned int c = e & 15u;          // 0..15
                const unsigned int row = rs * 16u + r;
                const unsigned int global_q = global_q_base + row;
                if (global_q >= total_q) continue;
                const unsigned int dim = o_slab * slab + o_col_base + cf * 16u + c;
                const float denom = fmaxf(scalars[row * 3u + 1u], 1.0e-20f);
                output[(size_t(global_q) * num_attention_heads + head) * hdim + dim] =
                    warp_scratch[e] / denom;
            }
        }
    }
}

#endif  // __CUDA_ARCH__ >= 800

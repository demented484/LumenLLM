// =============================================================================
// FlashAttention-2 style prefill attention kernel for head_dim=512.
// =============================================================================
//
// Opt-in (AEGIS_ATTN_FA2=1) rewrite of the head_dim=512 prefill attention
// path. The previous default kernel
// (`aegis_attention_prefill_dense_halfq_wmma_hdim512_q32_regacc`) ran at ~6%
// tensor-core utilisation. Root causes identified by code inspection:
//
//   1. K-tile of only 16 keys per iteration -> ~Nkeys/16 mainloop iterations,
//      each paying full WMMA fragment setup, an online-softmax update and a
//      __syncthreads(). FlashAttention-2 uses KV blocks of 64-128.
//   2. The online-softmax alpha rescale of the O accumulator round-trips
//      through shared memory (store fragment -> multiply in f32 -> reload).
//   3. 1 block/SM, occupancy too low to hide latency.
//
// FA-2 structural fixes applied here:
//   * KV block = 64 (vs 16). 4x fewer mainloop iterations, 4x fewer softmax
//     updates and __syncthreads(). This is the single biggest lever.
//   * The O accumulator stays register-resident across the whole KV mainloop
//     (`o_frags` WMMA accumulator fragments per warp, never spilled).
//   * The online-softmax alpha rescale is applied DIRECTLY to the register
//     accumulator fragment via `aegis_scale_wmma_accumulator_m16n16_rows`
//     (which exploits the documented m16n16 accumulator lane->row mapping) --
//     no store/reload-through-shared scratch.
//   * cp.async double-buffered K hdim-slab loads overlap HBM latency with the
//     Q*K WMMA chain.
//
// -----------------------------------------------------------------------------
// The head_dim=512 shared-memory wall
// -----------------------------------------------------------------------------
// A KV block of 64 positions x hdim=512 BF16 is 64 KiB for K and 64 KiB for V
// -- 128 KiB, far over the 96 KiB sm_120 opt-in dynamic-shared cap. This is
// exactly why the old kernel chose k_tile=16. The genuine FA-2 fix is NOT to
// stage the whole-hdim K/V tile in shared, but to STREAM the head dimension
// in slabs of 128: one K slab is 64 pos x 128 hdim x 2 B = 16 KiB. Both Q*K
// (the hdim contraction) and P*V (the hdim output) iterate the 4 slabs.
//
// Tiling (all compile-time constants):
//   q_block        = 32     query rows per block      (FA-2 kBlockM)
//   kv_block       = 64     KV positions per mainloop iter (kBlockN, 4x old)
//   hdim           = 512
//   slab           = 128    hdim slab width streamed through shared
//   n_slabs        = 4      hdim / slab
//   warps          = 16     512 threads / 32
//   row_strips     = 2      q_block / 16   (WMMA M tiles down Q)
//   kv_strips      = 4      kv_block / 16  (WMMA N tiles across the KV block)
//   cols_per_warp  = 32     hdim / warps   (O columns each warp owns)
//   o_col_frags    = 2      cols_per_warp / 16
//   o_frags        = 4      row_strips * o_col_frags  (persistent acc frags/warp)
//
// Shared-memory layout (peak ~76.5 KiB, within 96 KiB cap):
//   q_shared      = q_block*hdim    halfs = 32 KiB   (loaded once, persistent)
//   k_slab[2]     = 2*kv_block*slab halfs = 16 KiB   (cp.async double-buffered)
//   v_full        = kv_block*hdim   halfs = 64 KiB   <-- too big together.
//
// V needs the whole hdim for P*V (each O column needs its V column). But the
// O accumulator is split across warps -- warp w only needs V columns
// [w*32, w*32+32). So V is ALSO streamed in slabs: P*V iterates the 4 hdim
// slabs, accumulating into the O fragments for that slab's columns. V slab is
// 64 pos x 128 hdim x 2 B = 16 KiB, double-buffered = 32 KiB.
//
// Final shared layout (peak ~76.5 KiB):
//   q_shared      = 32*512*2  = 32 KiB    (persistent)
//   kv_slab[2]    = 2*64*128*2= 32 KiB    (cp.async double-buffered; K phase
//                                          and V phase reuse the same region)
//   s_shared      = 32*64*4   =  8 KiB    (S tile, then reused for P weights)
//   weights_h     = 32*64*2   =  4 KiB    (P in BF16 for the P*V WMMA)
//   scalars       = 32*4*4    =  0.5 KiB
//                              ---------
//                              ~76.5 KiB  (fits 96 KiB with headroom)
//
// Grid: (num_attention_heads, ceil(total_q / q_block)).  Block: 512 threads.
//
// -----------------------------------------------------------------------------
// Warp map
// -----------------------------------------------------------------------------
// Q*K: S is [q_block=32, kv_block=64] = 2 row strips x 4 kv strips = 8 WMMA
// tiles. 8 of the 16 warps each compute ONE complete S tile (full 512-deep
// hdim contraction); warps 8..15 idle during Q*K. Warp w (w<8) owns S tile
// (row_strip = w/4, kv_strip = w%4). It accumulates its s_frag across the 4
// hdim slabs as they stream in (8 kk-steps per slab, 32 mma total), so no
// cross-warp partial reduction is ever needed. Then it stores the tile to
// s_shared.
//
// P*V: O[q_block=32, hdim=512]. 16 warps; warp w owns hdim columns
// [w*32, w*32+32), which fall in slab w/4 (4 warps per 128-wide slab). For
// its 32 columns and 2 q row strips it holds o_frags = 4 persistent
// accumulator fragments. Per V slab streamed, only the 4 warps owning that
// slab do WMMA: row_strips(2) x o_col_frags(2) x kv_strips(4) = 16 mma.
//
// Online softmax: 32 q rows across 16 warps -> each warp owns 2 rows. Running
// m (max) and l (sum) per row in `scalars`; alpha (rescale factor) computed
// per KV block and applied to the register O accumulator directly.
// =============================================================================

#if __CUDA_ARCH__ >= 800

extern "C" __global__
__launch_bounds__(512, 1)
void aegis_attention_prefill_dense_fa2_hdim512(
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
    unsigned short* kv_slab   = q_shared + q_block * hdim;             // 2*kv_block*slab
    float*          s_shared  = reinterpret_cast<float*>(kv_slab + 2u * kv_block * slab);
    half*           weights_h = reinterpret_cast<half*>(s_shared + q_block * kv_block);
    float*          scalars   = reinterpret_cast<float*>(weights_h + q_block * kv_block);
    // scalars: 3 floats per q row -> [0]=m running max, [1]=l running sum,
    //          [2]=alpha (rescale factor for this KV block).
    // Stride 3 matches `aegis_scale_wmma_accumulator_m16n16_rows`, which reads
    // the per-row alpha at scalars[row*3 + 2].
    // Epilogue scratch overlays kv_slab (free once the mainloop ends):
    // 16 warps * 256 floats = 16 KiB <= kv_slab's 32 KiB.
    float*          o_scratch = reinterpret_cast<float*>(kv_slab);

    const unsigned int group   = num_attention_heads / num_kv_heads;
    const unsigned int kv_head = head / group;
    const float scale  = rsqrtf(float(hdim));
    const float log2e  = 1.4426950408889634f;
    const float neg_inf = -3.402823466e38f;

    // --- load Q tile once (whole hdim, persistent) ---
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
    auto cp_wait_lt1 = [] () { asm volatile("cp.async.wait_group 1;\n" ::); };
    auto cp_wait_all = [] () { asm volatile("cp.async.wait_group 0;\n" ::); };

    // Stage one 128-wide hdim slab of K or V for a KV block into a slab
    // buffer (kv_block * slab halfs = 16 KiB). Each thread copies 8 halfs
    // (16 B); kv_block*slab = 8192 halfs = 1024 chunks, 512 threads -> 2 each.
    auto stage_slab = [&] (const unsigned short* __restrict__ cache,
                           unsigned int tile_start, unsigned int slab_idx,
                           unsigned short* buf, unsigned int tile_count) {
        constexpr unsigned int halfs_per_chunk = 8u;
        constexpr unsigned int chunks_per_row  = slab / halfs_per_chunk;       // 16
        constexpr unsigned int passes          = (kv_block * chunks_per_row) / 512u; // 2
        const unsigned int hdim_base = slab_idx * slab;
#pragma unroll
        for (unsigned int p = 0u; p < passes; ++p) {
            const unsigned int chunk = p * 512u + tid;
            const unsigned int row   = chunk / chunks_per_row;                 // 0..63
            const unsigned int hoff  = (chunk % chunks_per_row) * halfs_per_chunk;
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
    const unsigned int o_col_base = (warp % warps_per_slab) * cols_per_warp; // col in slab

    // Prologue: stage K slab 0 of the first KV block.
    if (n_kiters > 0u) {
        const unsigned int tc0 = min(kv_block, block_max_visible - block_min_tile_start);
        stage_slab(key_cache, block_min_tile_start, 0u, kv_slab, tc0);
        cp_commit();
    }
    __syncthreads();

    // ----------------------------- mainloop --------------------------------
    for (unsigned int it = 0u; it < n_kiters; ++it) {
        const unsigned int tile_start = block_min_tile_start + it * kv_block;
        const unsigned int tile_count = min(kv_block, block_max_visible - tile_start);

        // ======================= Q*K -> S =======================
        // Software-pipelined hdim-slab streaming: ONE __syncthreads() per slab
        // (was two). Per slab iteration: wait for slab sl's load, ONE full
        // block barrier, prefetch slab sl+1, then WMMA(sl). The single barrier
        // simultaneously satisfies both hazards:
        //   (a) RAW -- slab sl's cp.async has landed block-wide before WMMA
        //       reads it (cp_wait orders the issuing thread; the barrier makes
        //       the visibility block-wide), and
        //   (b) WAR -- it runs after slab sl-1's WMMA, so the slab sl+1
        //       prefetch (issued just after the barrier, into buffer
        //       (sl+1)&1 == (sl-1)&1) cannot clobber data slab sl-1's WMMA is
        //       still reading.
        // The prefetch's cp.async overlaps WMMA(sl). Exactly one slab is in
        // flight at a time, so a plain cp_wait_all is correct and cheapest.
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag;
        wmma::fill_fragment(s_frag, 0.0f);
        for (unsigned int sl = 0u; sl < n_slabs; ++sl) {
            const unsigned int buf = sl & 1u;
            unsigned short* k_buf = kv_slab + buf * (kv_block * slab);
            cp_wait_all();     // slab sl's cp.async landed (issuing-thread order)
            __syncthreads();   // slab sl visible block-wide AND slab sl-1 WMMA done
            // Prefetch slab sl+1 into the other buffer; cp.async overlaps WMMA(sl).
            if (sl + 1u < n_slabs) {
                stage_slab(key_cache, tile_start, sl + 1u,
                           kv_slab + ((sl + 1u) & 1u) * (kv_block * slab), tile_count);
                cp_commit();
            }
            if (qk_active) {
                // s_frag[16q x 16kv] += Q[16q x 128] . K[16kv x 128]^T over
                // this slab. q rows  [qk_row*16 .. +16), kv [qk_kv*16 .. +16).
#pragma unroll
                for (unsigned int kk = 0u; kk < slab_kk; ++kk) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
                    wmma::load_matrix_sync(a_frag,
                        reinterpret_cast<const half*>(
                            q_shared + qk_row * 16u * hdim + sl * slab + kk * 16u), hdim);
                    wmma::load_matrix_sync(b_frag,
                        reinterpret_cast<const half*>(
                            k_buf + qk_kv * 16u * slab + kk * 16u), slab);
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
            // 64 kv columns, 32 lanes -> each lane handles columns lane, lane+32.
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
                // alpha = exp(old_m - new_m); for the first KV block (old_l==0)
                // the helper expects alpha=1.0 so the (zero) accumulator is a
                // no-op rescale and the early-out ballot fires.
                const float alpha = old_l > 0.0f ? exp2f((old_m - new_m) * log2e) : 1.0f;
                scalars[row * 3u + 0u] = new_m;
                scalars[row * 3u + 1u] = old_l * alpha + tile_l;
                scalars[row * 3u + 2u] = alpha;
            }
        }
        __syncthreads();

        // ======================= rescale O (in registers) =======================
        // alpha applied directly to the m16n16 accumulator fragment via the
        // shared helper (reads scalars[row*3 + 2], the documented m16n16
        // lane->row mapping). No store/reload-through-shared scratch.
        // NO trailing __syncthreads(): this phase only reads `scalars` (already
        // visible after the softmax barrier above) and writes per-thread WMMA
        // registers -- it touches no shared memory, so it creates no hazard for
        // the V-stage / P*V phase that follows. The softmax barrier already
        // separates the scalars writes from these reads.
#pragma unroll
        for (unsigned int rs = 0u; rs < row_strips; ++rs) {
#pragma unroll
            for (unsigned int cf = 0u; cf < o_col_frags; ++cf) {
                aegis_scale_wmma_accumulator_m16n16_rows(
                    o_frag[rs * o_col_frags + cf], scalars, rs * 16u);
            }
        }

        // ======================= P*V -> O =======================
        // weights_h holds P [q_block, kv_block] in BF16. Stream V slabs with
        // the same software-pipelined one-barrier-per-slab structure as Q*K:
        // V slab 0 is staged just above (after the softmax barrier, overlapping
        // the rescale); per slab iteration: wait, ONE barrier, prefetch slab
        // sl+1, WMMA(sl). The single barrier covers RAW (slab sl visible) and
        // WAR (slab sl-1's WMMA done before the slab sl+1 prefetch).
        stage_slab(value_cache, tile_start, 0u, kv_slab, tile_count);
        cp_commit();
        for (unsigned int sl = 0u; sl < n_slabs; ++sl) {
            const unsigned int buf = sl & 1u;
            unsigned short* v_buf = kv_slab + buf * (kv_block * slab);
            cp_wait_all();     // slab sl's cp.async landed (issuing-thread order)
            __syncthreads();   // slab sl visible block-wide AND slab sl-1 WMMA done
            if (sl + 1u < n_slabs) {
                stage_slab(value_cache, tile_start, sl + 1u,
                           kv_slab + ((sl + 1u) & 1u) * (kv_block * slab), tile_count);
                cp_commit();
            }
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
                                v_buf + ks * 16u * slab + vcol), slab);
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
            // NO trailing barrier: slab sl+1's top barrier (next iteration)
            // already separates this WMMA's v_buf reads from the slab sl+2
            // prefetch that overwrites the same buffer. For the last slab the
            // next-iter K slab-0 stage below overwrites buffer 0, last read by
            // P*V slab sl=2 -- separated by slab sl=3's top barrier.
        }

        // Prologue for next iter: stage its K slab 0. No barrier needed -- the
        // next iteration's Q*K slab-0 cp_wait_all + top barrier provides the
        // RAW; the WAR (buffer 0 last read by P*V slab 2) is covered by P*V
        // slab 3's top barrier which ran before this stage.
        if (it + 1u < n_kiters) {
            const unsigned int next_start = tile_start + kv_block;
            const unsigned int next_tc = min(kv_block, block_max_visible - next_start);
            stage_slab(key_cache, next_start, 0u, kv_slab, next_tc);
            cp_commit();
        }
    }

    // ============================ epilogue ============================
    // O[row, col] /= l[row]; write to global output. Each warp owns a 32-col
    // hdim slice and 2 row strips -> o_frags = 4 fragments. Store each
    // fragment to a per-warp 16x16 scratch (overlays the now-free kv_slab),
    // then write divided-by-l results to global memory. store_matrix_sync
    // avoids hand-deriving the m16n16 element->col mapping.
    __syncthreads();
    float* warp_scratch = o_scratch + warp * 256u;   // 16 warps * 256 = 16 KiB
#pragma unroll
    for (unsigned int rs = 0u; rs < row_strips; ++rs) {
#pragma unroll
        for (unsigned int cf = 0u; cf < o_col_frags; ++cf) {
            wmma::store_matrix_sync(warp_scratch,
                o_frag[rs * o_col_frags + cf], 16u, wmma::mem_row_major);
            // 16x16 = 256 elements, 32 lanes -> 8 per lane.
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

// =============================================================================
// FA-2 hdim=512 prefill, q_block=64 variant (Lever A: arithmetic-intensity).
// =============================================================================
//
// Disambiguation experiment turned production variant. The q_block=32 kernel
// re-reads every KV tile once per 32-query block: KV HBM traffic scales as
// (total_q / q_block) x KV_total. Doubling q_block to 64 HALVES the KV HBM
// re-read traffic and doubles arithmetic intensity (each loaded K/V element
// feeds 2x the MACs before eviction). If the kernel is memory-bound this is
// the dominant lever.
//
// The head_dim=512 shared wall forces a budget trade. With q_block=64:
//   q_shared   = 64*512*2          = 64 KiB   (persistent)
//   kv_slab[2] = 2 * kv_block*128*2            (cp.async double-buffered)
//   s_shared   = q_block*kv_block*4
//   weights_h  = q_block*kv_block*2
//   scalars    = q_block*3*4
// q_shared alone is 64 KiB, so kv_block is dropped 64 -> 32 to keep cp.async
// double-buffering AND fit the 96 KiB sm_120 cap:
//   q_shared 64 + kv_slab[2] 16 + s_shared 8 + weights_h 4 + scalars 0.75
//   = 92.75 KiB.
// KV HBM traffic depends ONLY on q_block (not kv_block) -- (total_q/64) full
// KV sweeps -- so kv_block=32 keeps the full 2x traffic win; it only adds
// mainloop-iteration / __syncthreads count, which is the latency-bound cost
// this experiment measures against.
//
// Tiling:
//   q_block=64, kv_block=32, hdim=512, slab=128, n_slabs=4, warps=16
//   row_strips = 4   (q_block/16)   kv_strips = 2  (kv_block/16)
//   cols_per_warp = 32  o_col_frags = 2  o_frags = 8  (4 row strips * 2 cols)
//   Q*K S tile [64,32] = 4*2 = 8 WMMA tiles -> warps 0..7 active, 8..15 idle.
//   O accumulator: 8 persistent acc frags/warp = 64 f32/thread. Feasible.
//   Softmax: 64 rows / 16 warps -> 4 rows per warp.
// =============================================================================

extern "C" __global__
__launch_bounds__(512, 1)
void aegis_attention_prefill_dense_fa2_hdim512_q64(
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
    using namespace nvcuda;
    constexpr unsigned int hdim          = 512u;
    constexpr unsigned int q_block       = 64u;
    constexpr unsigned int kv_block      = 32u;
    constexpr unsigned int warps         = 16u;
    constexpr unsigned int slab          = 128u;
    constexpr unsigned int n_slabs       = hdim / slab;          // 4
    constexpr unsigned int row_strips    = q_block / 16u;        // 4
    constexpr unsigned int kv_strips     = kv_block / 16u;       // 2
    constexpr unsigned int cols_per_warp = hdim / warps;         // 32
    constexpr unsigned int o_col_frags   = cols_per_warp / 16u;  // 2
    constexpr unsigned int o_frags       = row_strips * o_col_frags; // 8
    constexpr unsigned int warps_per_slab = slab / cols_per_warp;    // 4
    constexpr unsigned int slab_kk       = slab / 16u;           // 8
    constexpr unsigned int rows_per_warp = q_block / warps;      // 4

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
    unsigned short* kv_slab   = q_shared + q_block * hdim;             // 2*kv_block*slab
    float*          s_shared  = reinterpret_cast<float*>(kv_slab + 2u * kv_block * slab);
    half*           weights_h = reinterpret_cast<half*>(s_shared + q_block * kv_block);
    float*          scalars   = reinterpret_cast<float*>(weights_h + q_block * kv_block);
    // Epilogue scratch overlays kv_slab (free once the mainloop ends):
    // 16 warps * 256 floats = 16 KiB; kv_slab here is only 2*32*128*2 = 16 KiB,
    // so it fits exactly.
    float*          o_scratch = reinterpret_cast<float*>(kv_slab);

    const unsigned int group   = num_attention_heads / num_kv_heads;
    const unsigned int kv_head = head / group;
    const float scale  = rsqrtf(float(hdim));
    const float log2e  = 1.4426950408889634f;
    const float neg_inf = -3.402823466e38f;

    // --- load Q tile once (whole hdim, persistent) ---
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

    // --- persistent register-resident O accumulator (8 frags / warp) ---
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
    auto cp_wait_lt1 = [] () { asm volatile("cp.async.wait_group 1;\n" ::); };
    auto cp_wait_all = [] () { asm volatile("cp.async.wait_group 0;\n" ::); };

    // Stage one 128-wide hdim slab of K or V for a KV block (kv_block=32 rows)
    // into a slab buffer (kv_block*slab halfs = 8 KiB). Each thread copies
    // 8 halfs (16 B); kv_block*slab = 4096 halfs = 512 chunks, 512 threads ->
    // 1 pass.
    auto stage_slab = [&] (const unsigned short* __restrict__ cache,
                           unsigned int tile_start, unsigned int slab_idx,
                           unsigned short* buf, unsigned int tile_count) {
        constexpr unsigned int halfs_per_chunk = 8u;
        constexpr unsigned int chunks_per_row  = slab / halfs_per_chunk;       // 16
        constexpr unsigned int total_chunks    = kv_block * chunks_per_row;    // 512
        constexpr unsigned int passes          = (total_chunks + 511u) / 512u; // 1
        const unsigned int hdim_base = slab_idx * slab;
#pragma unroll
        for (unsigned int p = 0u; p < passes; ++p) {
            const unsigned int chunk = p * 512u + tid;
            if (chunk >= total_chunks) { continue; }
            const unsigned int row  = chunk / chunks_per_row;                  // 0..31
            const unsigned int hoff = (chunk % chunks_per_row) * halfs_per_chunk;
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
        }
    };

    const unsigned int n_kiters = (block_max_visible > block_min_tile_start)
        ? ((block_max_visible - block_min_tile_start + kv_block - 1u) / kv_block)
        : 0u;

    // Q*K warp -> S tile assignment (warps 0..7 active, S is [64,32] = 4*2).
    const bool       qk_active = warp < (row_strips * kv_strips);  // 8
    const unsigned int qk_row  = warp / kv_strips;                 // 0..3
    const unsigned int qk_kv   = warp % kv_strips;                 // 0..1
    // P*V warp -> hdim slab / column ownership (all 16 warps).
    const unsigned int o_slab     = warp / warps_per_slab;         // 0..3
    const unsigned int o_col_base = (warp % warps_per_slab) * cols_per_warp;

    // Prologue: stage K slab 0 of the first KV block.
    if (n_kiters > 0u) {
        const unsigned int tc0 = min(kv_block, block_max_visible - block_min_tile_start);
        stage_slab(key_cache, block_min_tile_start, 0u, kv_slab, tc0);
        cp_commit();
    }
    __syncthreads();

    // ----------------------------- mainloop --------------------------------
    for (unsigned int it = 0u; it < n_kiters; ++it) {
        const unsigned int tile_start = block_min_tile_start + it * kv_block;
        const unsigned int tile_count = min(kv_block, block_max_visible - tile_start);

        // ======================= Q*K -> S =======================
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag;
        wmma::fill_fragment(s_frag, 0.0f);
        for (unsigned int sl = 0u; sl < n_slabs; ++sl) {
            const unsigned int buf = sl & 1u;
            unsigned short* k_buf = kv_slab + buf * (kv_block * slab);
            if (sl + 1u < n_slabs) {
                stage_slab(key_cache, tile_start, sl + 1u,
                           kv_slab + ((sl + 1u) & 1u) * (kv_block * slab), tile_count);
                cp_commit();
                cp_wait_lt1();
            } else {
                cp_wait_all();
            }
            __syncthreads();
            if (qk_active) {
#pragma unroll
                for (unsigned int kk = 0u; kk < slab_kk; ++kk) {
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
                    wmma::load_matrix_sync(a_frag,
                        reinterpret_cast<const half*>(
                            q_shared + qk_row * 16u * hdim + sl * slab + kk * 16u), hdim);
                    wmma::load_matrix_sync(b_frag,
                        reinterpret_cast<const half*>(
                            k_buf + qk_kv * 16u * slab + kk * 16u), slab);
                    wmma::mma_sync(s_frag, a_frag, b_frag, s_frag);
                }
            }
            __syncthreads();
        }
        if (qk_active) {
            wmma::store_matrix_sync(
                s_shared + qk_row * 16u * kv_block + qk_kv * 16u,
                s_frag, kv_block, wmma::mem_row_major);
        }
        __syncthreads();

        // ======================= online softmax =======================
        // Each warp owns rows_per_warp = 4 q rows: warp, warp+16, warp+32, warp+48.
#pragma unroll
        for (unsigned int rr = 0u; rr < rows_per_warp; ++rr) {
            const unsigned int row = warp + rr * warps;          // 0..63
            const unsigned int global_q = global_q_base + row;
            const bool valid_q = global_q < total_q;
            const unsigned int visible_len = valid_q
                ? min(context_len, start_position + global_q + 1u) : 0u;
            const unsigned int row_min_visible = (window_size > 0u
                && start_position + global_q + 1u > window_size)
                ? (start_position + global_q + 1u - window_size) : 0u;
            const float old_m = scalars[row * 3u + 0u];
            const float old_l = scalars[row * 3u + 1u];
            // 32 kv columns, 32 lanes -> each lane handles one column.
            const unsigned int col = lane;
            const unsigned int pos = tile_start + col;
            const float sc = (valid_q && col < tile_count && pos < visible_len
                              && pos >= row_min_visible)
                ? s_shared[row * kv_block + col] * scale
                : neg_inf;
            float tile_m = aegis_warp_reduce_max(sc);
            const float new_m = fmaxf(old_m, tile_m);
            const float w = (sc > -3.0e38f) ? exp2f((sc - new_m) * log2e) : 0.0f;
            weights_h[row * kv_block + col] = __float2half_rn(w);
            const float tile_l = aegis_warp_reduce_sum(w);
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
        __syncthreads();

        // ======================= P*V -> O =======================
        for (unsigned int sl = 0u; sl < n_slabs; ++sl) {
            const unsigned int buf = sl & 1u;
            unsigned short* v_buf = kv_slab + buf * (kv_block * slab);
            if (sl == 0u) {
                stage_slab(value_cache, tile_start, 0u, kv_slab, tile_count);
                cp_commit();
            }
            if (sl + 1u < n_slabs) {
                stage_slab(value_cache, tile_start, sl + 1u,
                           kv_slab + ((sl + 1u) & 1u) * (kv_block * slab), tile_count);
                cp_commit();
                cp_wait_lt1();
            } else {
                cp_wait_all();
            }
            __syncthreads();
            if (o_slab == sl) {
                // O[16q x 32] += P[16q x 32] . V[32 x 32] for this warp's cols.
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;
#pragma unroll
                for (unsigned int cf = 0u; cf < o_col_frags; ++cf) {
                    const unsigned int vcol = o_col_base + cf * 16u;
#pragma unroll
                    for (unsigned int ks = 0u; ks < kv_strips; ++ks) {
                        wmma::load_matrix_sync(v_frag,
                            reinterpret_cast<const half*>(
                                v_buf + ks * 16u * slab + vcol), slab);
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
            __syncthreads();
        }

        // Prologue for next iter: stage its K slab 0.
        if (it + 1u < n_kiters) {
            const unsigned int next_start = tile_start + kv_block;
            const unsigned int next_tc = min(kv_block, block_max_visible - next_start);
            stage_slab(key_cache, next_start, 0u, kv_slab, next_tc);
            cp_commit();
            __syncthreads();
        }
    }

    // ============================ epilogue ============================
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

// =============================================================================
// FlashAttention-2 prefill attention for head_dim=512 — native FP8 e4m3 MMA.
// =============================================================================
//
// Opt-in (AEGIS_ATTN_FP8=1 / `compute-quantization: fp8`) FP8 prefill attention
// for Gemma-4's 5 head_dim=512 global layers. Selected ONLY when the KV cache
// is FP8 and head_dim == 512. Default OFF -> prefill is bit-equivalent to main.
//
// -----------------------------------------------------------------------------
// What this kernel is, vs the option-b `_fp8` kernel in attention_prefill_fa2_fp8
// -----------------------------------------------------------------------------
// The option-b kernel (`aegis_attention_prefill_dense_fa2_hdim512_fp8`) reads
// the e4m3 KV cache but DEQUANTS K/V to half in shared memory and runs the BF16
// `nvcuda::wmma` math. Its shared footprint is therefore the SAME ~76 KiB as
// the BF16 FA-2 kernel (it carries both an e4m3 staging buffer AND a half slab)
// -> still 1 block/SM.
//
// THIS kernel keeps K/V e4m3 IN shared memory and feeds the e4m3 bytes straight
// into the SM120 FP8 `mma.sync.aligned.kind::f8f6f4.m16n8k32` tensor-core MMA
// (the hardware-verified `aegis_mma_m16n8k32_e4m3_p` primitive — see
// fp8_mma.cuh / fp8_mma_smoke.cu). No half slab. The smem footprint roughly
// halves -> the kernel fits 2 thread-blocks per SM, the latency-hiding the BF16
// kernel structurally could not reach within the 100 KiB sm_120 cap. That
// occupancy jump is the real win; the raw FP8 MMA throughput is secondary
// because the BF16 kernel was diagnosed latency-bound, not roofline-bound.
//
// -----------------------------------------------------------------------------
// Tiling (all compile-time constants)
// -----------------------------------------------------------------------------
//   q_block   = 32     query rows per block
//   kv_block  = 64     KV positions per mainloop iteration
//   hdim      = 512
//   slab      = 128    hdim slab width streamed through shared (Q*K + P*V)
//   n_slabs   = 4      hdim / slab
//   warps     = 16     512 threads / 32
// MMA shape is m16 n8 k32 (NOT the wmma m16n16k16).
//   Q*K: S[q_block=32, kv_block=64] = 2 m-strips x 8 n-strips = 16 MMA tiles
//        -> all 16 warps active, warp w owns S tile (w/8, w%8). hdim=512
//        contraction = 16 k-tiles of 32; warp accumulates across all of them.
//   P*V: O[q_block=32, hdim=512]. m16 n8 -> 2 m-strips x 64 n-strips. 16 warps,
//        warp w owns hdim cols [w*32, w*32+32) = 4 n-tiles. Contraction over
//        kv_block=64 = 2 k-tiles of 32.
//
// -----------------------------------------------------------------------------
// Shared-memory map  (peak 42.6 KiB -> 2 blocks/SM within the 100 KiB cap)
// -----------------------------------------------------------------------------
//   q_e4m3      = q_block*hdim         bytes  = 32*512   = 16   KiB  (persistent)
//   kv_e4m3[2]  = 2*kv_block*slab      bytes  = 2*64*128 = 16   KiB  (cp.async db)
//   s_shared    = q_block*kv_block     floats = 32*64*4  =  8   KiB  (S / softmax)
//   p_e4m3      = q_block*kv_block     bytes  = 32*64    =  2   KiB  (P requant)
//   scalars     = q_block*3            floats = 32*3*4   =  0.375 KiB
//   q_scale     = q_block              floats = 32*4     =  0.125 KiB
//                                                          -----------
//                                                          ~42.5 KiB
//   2 blocks/SM = 85.0 KiB < 100 KiB sm_120 dynamic-shared cap.  ACHIEVED.
//
//   The f32 s_shared (8 KiB) does NOT shrink with FP8 — it is honestly budgeted.
//   The epilogue writes O straight to global from the register accumulator
//   (the m16n8 C/D fragment->element mapping is hand-derived, no scratch tile),
//   so no extra shared scratch is needed at all.
//
// -----------------------------------------------------------------------------
// Scale composition  (THE numerical correctness point — read carefully)
// -----------------------------------------------------------------------------
// K and V come from the persistent FP8 KV cache as RAW clamped e4m3 bytes with
// NO per-block scale (kv_fp8.cu writes via float_to_fp8_e4m3_bits directly).
// So the only score scaling is rsqrt(head_dim). Q and P are quantized here.
//
//   Q*K:  Q row m is quantized with a per-row absmax scale qs[m] (one f32 per
//         query row, stored in q_scale[m]). The MMA computes
//         acc = sum_k Qe4m3[m,k] * Ke4m3[n,k]. Because qs[m] is constant over
//         the contraction index k AND over n (it depends only on the M index),
//         it factors cleanly out of the k-sum:
//           true score sum_k Q[m,k]*K[n,k] = acc * qs[m].
//         K has NO scale. Final S[m,n] = acc * qs[m] * rsqrt(head_dim).
//
//   P*V:  After softmax, P[m,k] are the (unnormalized) exp weights
//         exp2((s - m_running) * log2e). Their MAX is exactly 1.0 (the term
//         where s == m_running). So P is bounded in [0, 1] with a known fixed
//         ceiling — there is NO need for a per-row / per-block P scale. We use
//         a SINGLE COMPILE-TIME constant scale: encode P*p_enc to e4m3 with
//         p_enc = 448 (e4m3 max), and divide the accumulator by p_enc once.
//           p_e4m3[m,k] = e4m3( P[m,k] * 448 )
//           MMA acc     = sum_k p_e4m3[m,k] * Ve4m3[k,n] = 448 * sum_k P*V
//           true O[m,n] = acc / 448.
//         448 is constant over m, k, n and over KV blocks -> it commutes with
//         both the f32 accumulation AND the cross-block online-softmax alpha
//         rescale, so it can be applied ONCE in the epilogue. V has NO scale.
//         (A per-block/per-row P scale would NOT commute with the cross-block
//         sum and would have to be folded per block — using the fixed 448
//         constant deliberately avoids that whole failure mode.)
//
// Composed: O_unnorm[m,n] = (sum over KV blocks, alpha-rescaled, of acc) / 448;
//           final O[m,n]  = O_unnorm[m,n] / l[m]   (l = online-softmax sum).
// A WRONG scale here (qs applied twice, 448 forgotten, rsqrt missing) makes the
// output a constant factor off or garbage — the classic FP8-attention failure
// and the first thing to check on hardware.
//
// Grid: (num_attention_heads, ceil(total_q / q_block)).  Block: 512 threads.
// =============================================================================

#if __CUDA_ARCH__ >= 800

extern "C" __global__
__launch_bounds__(512, 2)
void aegis_attention_prefill_dense_fa2_hdim512_fp8_mma(
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
    constexpr unsigned int hdim       = 512u;
    constexpr unsigned int q_block    = 32u;
    constexpr unsigned int kv_block   = 64u;
    constexpr unsigned int warps      = 16u;
    constexpr unsigned int slab       = 128u;
    constexpr unsigned int n_slabs    = hdim / slab;        // 4
    constexpr unsigned int k_per_slab = slab / 32u;         // 4  (k-tiles of 32)
    constexpr unsigned int kv_ktiles  = kv_block / 32u;     // 2  (P*V k-tiles)
    constexpr unsigned int s_mstrips  = q_block / 16u;      // 2
    constexpr unsigned int s_nstrips  = kv_block / 8u;      // 8  (S n-tiles, n=8)
    constexpr unsigned int o_dcols    = hdim / warps;       // 32 cols per warp
    constexpr unsigned int o_ntiles   = o_dcols / 8u;       // 4  (n=8 P*V tiles)
    constexpr float        e4m3_max   = 448.0f;
    constexpr float        p_enc      = 448.0f;  // fixed P-requant scale
    constexpr float        p_dec      = 1.0f / 448.0f;

    const unsigned int head          = blockIdx.x;
    const unsigned int global_q_base = blockIdx.y * q_block;
    const unsigned int tid           = threadIdx.x;
    const unsigned int lane          = tid & 31u;
    const unsigned int warp          = tid >> 5u;
    const unsigned int g             = lane >> 2u;          // groupID 0..7
    const unsigned int qq            = lane & 3u;           // thread-in-group 0..3
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

    // --- shared layout (see header map) ---
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char* q_e4m3   = smem;                                   // 16 KiB
    unsigned char* kv_e4m3  = q_e4m3 + q_block * hdim;                 // 16 KiB
    float*         s_shared = reinterpret_cast<float*>(
        kv_e4m3 + 2u * kv_block * slab);                              //  8 KiB
    unsigned char* p_e4m3   = reinterpret_cast<unsigned char*>(
        s_shared + q_block * kv_block);                               //  2 KiB
    float*         scalars  = reinterpret_cast<float*>(
        p_e4m3 + ((q_block * kv_block + 15u) & ~15u));                 //  m,l,alpha
    float*         q_scale  = scalars + q_block * 3u;                  //  per-Q-row

    const unsigned int group   = num_attention_heads / num_kv_heads;
    const unsigned int kv_head = head / group;
    const float scale   = rsqrtf(float(hdim));
    const float log2e   = 1.4426950408889634f;
    const float neg_inf = -3.402823466e38f;

    // ----------------------------------------------------------------------
    // Load Q tile -> quantize to e4m3 with a per-Q-row absmax scale.
    // Q arrives as half (BF16-class) from the RoPE'd prefill QKV path. Each
    // thread owns whole rows in a round-robin so the per-row absmax reduction
    // is a simple thread-local scan (no cross-thread reduction needed).
    // q_e4m3 is laid out [q_block][hdim] row-major; element (m,k) at m*hdim+k.
    // ----------------------------------------------------------------------
    for (unsigned int row = tid; row < q_block; row += blockDim.x) {
        const unsigned int global_q = global_q_base + row;
        const unsigned short* q_src = (global_q < total_q)
            ? query + (size_t(global_q) * num_attention_heads + head) * hdim
            : (const unsigned short*)0;
        // pass 1: per-row absmax over hdim.
        float amax = 0.0f;
        if (q_src) {
            for (unsigned int k = 0u; k < hdim; ++k) {
                amax = fmaxf(amax, fabsf(__half2float(__ushort_as_half(q_src[k]))));
            }
        }
        const float qs   = (amax > 0.0f) ? (amax / e4m3_max) : 1.0f;
        const float invq = (amax > 0.0f) ? (e4m3_max / amax) : 0.0f;
        q_scale[row] = qs;
        // pass 2: scale into e4m3 range and encode.
        unsigned char* q_row = q_e4m3 + row * hdim;
        for (unsigned int k = 0u; k < hdim; ++k) {
            const float v = q_src
                ? __half2float(__ushort_as_half(q_src[k])) * invq
                : 0.0f;
            q_row[k] = float_to_fp8_e4m3_bits(v);
        }
        scalars[row * 3u + 0u] = neg_inf;   // m running max
        scalars[row * 3u + 1u] = 0.0f;      // l running sum
        scalars[row * 3u + 2u] = 0.0f;      // alpha
    }

    // --- persistent register-resident O accumulator (f32) ---
    // o_acc[mstrip][ntile][4] : per warp, 2 m-strips x 4 n-tiles x 4 = 32 f32.
    float o_acc[s_mstrips][o_ntiles][4];
#pragma unroll
    for (unsigned int ms = 0u; ms < s_mstrips; ++ms)
#pragma unroll
        for (unsigned int nt = 0u; nt < o_ntiles; ++nt)
#pragma unroll
            for (unsigned int e = 0u; e < 4u; ++e) o_acc[ms][nt][e] = 0.0f;

    // --- cp.async helpers (e4m3 bytes; 16-byte chunks) ---
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

    // Stage one 128-wide hdim slab of e4m3 K or V for a KV block into a slab
    // buffer (kv_block*slab = 64*128 = 8192 bytes). Each thread copies a
    // 16-byte chunk = 16 e4m3 elements; 8192/16 = 512 chunks, 512 threads ->
    // 1 chunk each. Layout in shared: buf[row*slab + hoff].
    auto stage_slab_e4m3 = [&] (const unsigned char* __restrict__ cache,
                                unsigned int tile_start, unsigned int slab_idx,
                                unsigned char* buf, unsigned int tile_count) {
        constexpr unsigned int bytes_per_chunk = 16u;
        constexpr unsigned int chunks_per_row  = slab / bytes_per_chunk;     // 8
        constexpr unsigned int total_chunks    = kv_block * chunks_per_row;  // 512
        const unsigned int hdim_base = slab_idx * slab;
        const unsigned int chunk = tid;
        if (chunk >= total_chunks) { return; }
        const unsigned int row  = chunk / chunks_per_row;                    // 0..63
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

    auto kv_buf = [&] (unsigned int b) -> unsigned char* {
        return kv_e4m3 + b * (kv_block * slab);
    };

    const unsigned int n_kiters = (block_max_visible > block_min_tile_start)
        ? ((block_max_visible - block_min_tile_start + kv_block - 1u) / kv_block)
        : 0u;

    // Q*K warp -> S tile assignment: S[32,64], m16 n8 -> 2 m-strips x 8 n-strips.
    // 16 warps, warp w owns tile (s_ms = w/8, s_ns = w%8).
    const unsigned int s_ms = warp / s_nstrips;     // 0..1   (q rows [s_ms*16..])
    const unsigned int s_ns = warp % s_nstrips;     // 0..7   (kv cols [s_ns*8..])
    // P*V warp -> hdim column ownership: warp w owns hdim cols [w*32, w*32+32).
    const unsigned int o_dcol_base = warp * o_dcols;

    // Prologue: stage e4m3 K slab 0 of the first KV block.
    if (n_kiters > 0u) {
        const unsigned int tc0 = min(kv_block, block_max_visible - block_min_tile_start);
        stage_slab_e4m3(key_cache, block_min_tile_start, 0u, kv_buf(0u), tc0);
        cp_commit();
    }
    __syncthreads();

    // ============================ mainloop =================================
    for (unsigned int it = 0u; it < n_kiters; ++it) {
        const unsigned int tile_start = block_min_tile_start + it * kv_block;
        const unsigned int tile_count = min(kv_block, block_max_visible - tile_start);

        // ======================= Q*K -> S =======================
        // s_frag[4] is this warp's [16 q x 8 kv] accumulator across the whole
        // hdim=512 contraction (16 k-tiles of 32). The hdim is streamed in 4
        // slabs of 128; each slab carries k_per_slab=4 k-tiles.
        float s_frag[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (unsigned int sl = 0u; sl < n_slabs; ++sl) {
            unsigned char* k_buf = kv_buf(sl & 1u);
            cp_wait_all();     // slab sl's e4m3 cp.async landed (issuing order)
            __syncthreads();   // slab sl visible block-wide AND slab sl-1's MMA
                               // done -> buffer (sl+1)&1 safe to overwrite.
            if (sl + 1u < n_slabs) {
                stage_slab_e4m3(key_cache, tile_start, sl + 1u,
                                kv_buf((sl + 1u) & 1u), tile_count);
                cp_commit();
            }
            // 4 k-tiles of 32 in this slab.
#pragma unroll
            for (unsigned int kt = 0u; kt < k_per_slab; ++kt) {
                const unsigned int k_base = sl * slab + kt * 32u;  // hdim offset
                // --- A fragment: Q[16 q-rows x 32 k]. Row m of this warp's S
                //     tile maps to global q-tile row s_ms*16 + m.
                //     A layout: m = g + 8*v1, k = qq*4 + v0 + 16*v2,
                //               register r = v1 + 2*v2.
                aegis_u32 a[4];
#pragma unroll
                for (unsigned int v2 = 0u; v2 < 2u; ++v2)
#pragma unroll
                    for (unsigned int v1 = 0u; v1 < 2u; ++v1) {
                        unsigned char by[4];
#pragma unroll
                        for (unsigned int v0 = 0u; v0 < 4u; ++v0) {
                            const unsigned int m = s_ms * 16u + g + 8u * v1;
                            const unsigned int k = k_base + qq * 4u + v0 + 16u * v2;
                            by[v0] = q_e4m3[m * hdim + k];
                        }
                        a[v1 + 2u * v2] =
                            aegis_pack_e4m3x4_p(by[0], by[1], by[2], by[3]);
                    }
                // --- B fragment: K[8 kv-rows x 32 k]. Row in the slab buffer =
                //     s_ns*8 + n.  B[n,k] = K_slab[n][k].
                //     B layout: n = g, k = qq*4 + v0 + 16*v1, register r = v1.
                aegis_u32 b[2];
#pragma unroll
                for (unsigned int v1 = 0u; v1 < 2u; ++v1) {
                    unsigned char by[4];
#pragma unroll
                    for (unsigned int v0 = 0u; v0 < 4u; ++v0) {
                        const unsigned int n  = s_ns * 8u + g;
                        // k within slab buffer: kt*32 + qq*4 + v0 + 16*v1.
                        const unsigned int ks = kt * 32u + qq * 4u + v0 + 16u * v1;
                        by[v0] = k_buf[n * slab + ks];
                    }
                    b[v1] = aegis_pack_e4m3x4_p(by[0], by[1], by[2], by[3]);
                }
                float d[4];
                aegis_mma_m16n8k32_e4m3_p(d, a, b, s_frag);
                s_frag[0] = d[0]; s_frag[1] = d[1];
                s_frag[2] = d[2]; s_frag[3] = d[3];
            }
        }
        // Store this warp's S tile to s_shared [q_block, kv_block].
        // C/D layout: d0,d1 -> (row g,   col 2*qq+{0,1});
        //             d2,d3 -> (row g+8, col 2*qq+{0,1}).
        // q row  = s_ms*16 + {g, g+8};  kv col = s_ns*8 + 2*qq + {0,1}.
        // Dequant by q_scale[row] (K has NO scale). softmax_scale applied later.
        {
            const unsigned int r0 = s_ms * 16u + g;
            const unsigned int r1 = s_ms * 16u + g + 8u;
            const unsigned int c0 = s_ns * 8u + 2u * qq + 0u;
            const unsigned int c1 = s_ns * 8u + 2u * qq + 1u;
            s_shared[r0 * kv_block + c0] = s_frag[0] * q_scale[r0];
            s_shared[r0 * kv_block + c1] = s_frag[1] * q_scale[r0];
            s_shared[r1 * kv_block + c0] = s_frag[2] * q_scale[r1];
            s_shared[r1 * kv_block + c1] = s_frag[3] * q_scale[r1];
        }
        __syncthreads();

        // ======================= online softmax =======================
        // 32 q rows / 16 warps -> each warp owns 2 rows (warp, warp+16).
        // Each warp's 32 lanes cover the 64 kv columns (2 cols per lane).
        // The (unnormalized) exp weights are written back into s_shared.
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
                s_shared[row * kv_block + lane + c * 32u] = w[c];   // probs
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

        // ======================= requant P -> e4m3 =======================
        // p_e4m3[row][col] = e4m3( P[row][col] * p_enc ). P (the exp weights,
        // unnormalized) is bounded in [0, 1] with max exactly 1.0, so a single
        // FIXED scale p_enc = 448 maps it onto the full e4m3 range with no
        // per-row / per-block scale tracking. The matching 1/448 divide is
        // applied once in the epilogue (it commutes with the cross-block sum).
        for (unsigned int idx = tid; idx < q_block * kv_block; idx += blockDim.x) {
            p_e4m3[idx] = float_to_fp8_e4m3_bits(s_shared[idx] * p_enc);
        }
        __syncthreads();

        // ======================= rescale O (registers) =======================
        // O accumulator carried across KV blocks: rescale by alpha[row].
        // For the P*V MMA's C/D layout, d0,d1 own q row (ms*16 + g) and
        // d2,d3 own q row (ms*16 + g + 8).
#pragma unroll
        for (unsigned int ms = 0u; ms < s_mstrips; ++ms) {
            const float a0 = scalars[(ms * 16u + g) * 3u + 2u];
            const float a8 = scalars[(ms * 16u + g + 8u) * 3u + 2u];
#pragma unroll
            for (unsigned int nt = 0u; nt < o_ntiles; ++nt) {
                o_acc[ms][nt][0] *= a0; o_acc[ms][nt][1] *= a0;
                o_acc[ms][nt][2] *= a8; o_acc[ms][nt][3] *= a8;
            }
        }

        // ======================= P*V -> O =======================
        // O[32 q x 512 D] += P[32 q x 64 kv] . V[64 kv x 512 D].
        // V is streamed in 4 e4m3 slabs of 128 D. Warp w owns 32 D-cols, which
        // fall in slab w/4 -> only the 4 warps owning a slab do P*V for it.
        // Contraction over kv_block=64 = kv_ktiles=2 k-tiles of 32.
        // P is the A operand (e4m3, p_e4m3); V is the B operand (e4m3 slab).
        stage_slab_e4m3(value_cache, tile_start, 0u, kv_buf(0u), tile_count);
        cp_commit();
        const unsigned int o_slab = o_dcol_base / slab;   // 0..3
        for (unsigned int sl = 0u; sl < n_slabs; ++sl) {
            unsigned char* v_buf = kv_buf(sl & 1u);
            cp_wait_all();
            __syncthreads();
            if (sl + 1u < n_slabs) {
                stage_slab_e4m3(value_cache, tile_start, sl + 1u,
                                kv_buf((sl + 1u) & 1u), tile_count);
                cp_commit();
            }
            if (o_slab == sl) {
                // This warp's 32 D-cols lie at slab offset o_dcol_base - sl*128.
                const unsigned int dcol_in_slab = o_dcol_base - sl * slab;
#pragma unroll
                for (unsigned int kt = 0u; kt < kv_ktiles; ++kt) {
                    // --- A fragment: P[16 q x 32 kv]. For m-strip ms, q row
                    //     m = ms*16 + (g + 8*v1); kv = kt*32 + qq*4 + v0 + 16*v2.
                    //     A layout register r = v1 + 2*v2.
                    aegis_u32 a[s_mstrips][4];
#pragma unroll
                    for (unsigned int ms = 0u; ms < s_mstrips; ++ms)
#pragma unroll
                        for (unsigned int v2 = 0u; v2 < 2u; ++v2)
#pragma unroll
                            for (unsigned int v1 = 0u; v1 < 2u; ++v1) {
                                unsigned char by[4];
#pragma unroll
                                for (unsigned int v0 = 0u; v0 < 4u; ++v0) {
                                    const unsigned int m  = ms * 16u + g + 8u * v1;
                                    const unsigned int kv = kt * 32u + qq * 4u
                                                            + v0 + 16u * v2;
                                    by[v0] = p_e4m3[m * kv_block + kv];
                                }
                                a[ms][v1 + 2u * v2] =
                                    aegis_pack_e4m3x4_p(by[0], by[1], by[2], by[3]);
                            }
                    // --- 4 n-tiles of 8 D-cols. B = V^T: B[n,k] = V_slab[k][n].
                    //     n = D-col within tile; k = kv index.
                    //     B layout: n = g, k = qq*4 + v0 + 16*v1, register r = v1.
#pragma unroll
                    for (unsigned int nt = 0u; nt < o_ntiles; ++nt) {
                        aegis_u32 b[2];
#pragma unroll
                        for (unsigned int v1 = 0u; v1 < 2u; ++v1) {
                            unsigned char by[4];
#pragma unroll
                            for (unsigned int v0 = 0u; v0 < 4u; ++v0) {
                                // D-col within the slab: dcol_in_slab + nt*8 + g.
                                const unsigned int dcol = dcol_in_slab
                                                          + nt * 8u + g;
                                // kv index within this slab buffer's k-tile.
                                const unsigned int kv = kt * 32u + qq * 4u
                                                        + v0 + 16u * v1;
                                by[v0] = v_buf[kv * slab + dcol];
                            }
                            b[v1] = aegis_pack_e4m3x4_p(by[0], by[1], by[2], by[3]);
                        }
#pragma unroll
                        for (unsigned int ms = 0u; ms < s_mstrips; ++ms) {
                            float d[4];
                            aegis_mma_m16n8k32_e4m3_p(d, a[ms], b, o_acc[ms][nt]);
                            o_acc[ms][nt][0] = d[0]; o_acc[ms][nt][1] = d[1];
                            o_acc[ms][nt][2] = d[2]; o_acc[ms][nt][3] = d[3];
                        }
                    }
                }
            }
        }

        // Prologue for next iter: stage its e4m3 K slab 0. The next iter's Q*K
        // slab-0 cp_wait_all + top barrier provides RAW; the WAR (kv buffer 0
        // last read by P*V slab sl=2) is covered by P*V slab sl=3's top barrier
        // which ran before this stage.
        if (it + 1u < n_kiters) {
            const unsigned int next_start = tile_start + kv_block;
            const unsigned int next_tc = min(kv_block, block_max_visible - next_start);
            stage_slab_e4m3(key_cache, next_start, 0u, kv_buf(0u), next_tc);
            cp_commit();
        }
    }

    // ============================ epilogue ============================
    // O[m,n] = ( o_acc * p_dec ) / l[m].   p_dec = 1/448 undoes the fixed P
    // requant scale; l = the online-softmax running sum. o_acc is in units of
    // (p_e4m3 * v_e4m3) summed across all KV blocks (alpha-rescaled); since
    // p_enc and v have no per-block variation, p_dec applies once here.
    __syncthreads();
#pragma unroll
    for (unsigned int ms = 0u; ms < s_mstrips; ++ms) {
#pragma unroll
        for (unsigned int nt = 0u; nt < o_ntiles; ++nt) {
            // o_acc[ms][nt]: d0,d1 -> (row g,   col 2*qq+{0,1});
            //                d2,d3 -> (row g+8, col 2*qq+{0,1}).
            // q row  = ms*16 + {g, g+8};  D-col = o_dcol_base + nt*8 + 2*qq+{0,1}.
            const unsigned int r0 = ms * 16u + g;
            const unsigned int r1 = ms * 16u + g + 8u;
            const unsigned int cbase = nt * 8u + 2u * qq;
            const unsigned int rows[2]  = { r0, r1 };
            const float        v0[2]    = { o_acc[ms][nt][0], o_acc[ms][nt][1] };
            const float        v1[2]    = { o_acc[ms][nt][2], o_acc[ms][nt][3] };
            const float* const vals[2]  = { v0, v1 };
#pragma unroll
            for (unsigned int hh = 0u; hh < 2u; ++hh) {
                const unsigned int row = rows[hh];
                const unsigned int global_q = global_q_base + row;
                if (global_q >= total_q) continue;
                const float denom = fmaxf(scalars[row * 3u + 1u], 1.0e-20f);
                const float norm  = p_dec / denom;
#pragma unroll
                for (unsigned int cc = 0u; cc < 2u; ++cc) {
                    const unsigned int dim = o_dcol_base + cbase + cc;
                    output[(size_t(global_q) * num_attention_heads + head) * hdim
                           + dim] = vals[hh][cc] * norm;
                }
            }
        }
    }
}

#endif  // __CUDA_ARCH__ >= 800

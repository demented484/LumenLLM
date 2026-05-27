// =============================================================================
// Register-Softmax GQA-packed FP16 FlashAttention prefill kernel (Stage H.3).
// head_dim = 512, dense, causal+sliding, GQA group %% HP == 0. Faithful port of
// llama.cpp's fattn-mma-f16.cuh STRUCTURE (not just techniques) for D=512.
// =============================================================================
//
// MOTIVATION (re-derived 2026-05-24): every prior evolutionary attempt (mma2,
// gqa2, ldmatrix variants) failed because we kept spilling the Q·K score tile
// to shared (`s_shared`) and re-reading it in softmax, paying one __syncthreads
// AND a full MIO round-trip per KV tile. llama.cpp's fattn-mma keeps KQ_C
// REGISTER-RESIDENT across all nbatch_fa positions per warp, runs softmax
// directly on those registers via warp __shfl_xor_sync reductions, and feeds
// register-resident weights straight into P·V — no shared bounce, no sync
// between Q·K and softmax. THAT is the latency-hiding mechanism (the 1.33x
// kernel gap we never closed). It's a *structural* rewrite, not a feature graft.
//
// CONFIG (from llama.cpp ampere table for DKQ=DV=512, ncols=64):
//   nthreads        = 256                  (8 warps; cc12.0 / SM120)
//   occupancy       = 1                    (1 block/SM by design)
//   nbatch_fa       = 32                   (KV positions per tile)
//   nbatch_K2       = 128                  (K hdim slab width, half2)
//   nbatch_V2       = 128                  (V hdim slab width, half2)
//   nbatch_combine  = 128                  (epilogue combine slab)
//   nstages         = 1                    (single cp.async buffer)
//   Q_in_reg        = false                (Q in shared, reloaded via ldmatrix)
//   cols_per_warp   = 16                   (each warp owns 16 q-cols)
//   cols_per_thread = 2                    (per-lane KQ_max/rowsum cols)
//   np              = nwarps*cols_per_warp/ncols = 8*16/64 = 2
//   (np = warps sharing the same q-col-group; 4 col-groups × 2 warps = 8)
//
// CASE PARAMETERS for Gemma-4 global hd512 (group = 16q/2kv = 8):
//   q_block (=ncols1) = 32                 q rows per CTA
//   HP      (=ncols2) = 2                  consecutive q-heads packed
//   ncols             = 64                 32 q-rows × 2 heads, laid out as 64
//                                          "q-cols" in shared, packed
//                                          interleaved per row (col = row*HP + hp)
//   Grid              = (num_attention_heads/HP, ceil(total_q/q_block))
//   head0             = blockIdx.x * HP    consecutive q-heads share kv-head
//   kv_head           = head0 / group
//   guard             = group % HP == 0 (group=8 % 2 == 0 ✓ for Gemma-4 global)
//
// SHARED LAYOUT (target ~46 KiB at 1 block/SM, fits 96 KiB opt-in cap easily):
//   tile_Q             = ncols * DKQ/2 * sizeof(half2)        = 64 * 256 * 4 =  64 KiB  // wait below
//     ACTUALLY llama.cpp's tile_Q in shared is ncols * (DKQ/2 + 4 padding) half2.
//     For ncols=64 DKQ=512: 64 * (256+4) = 16640 half2 = ~33 KiB. Padding +4 is to
//     break shared-bank conflicts on ldmatrix. We use the same.
//   tile_K             = nbatch_fa * (nbatch_K2 + 4) half2     = 32 * 132 = 4224 half2 = ~8.5 KiB
//   tile_V             = nbatch_fa * (nbatch_V2 + 4) half2     = same as tile_K  = ~8.5 KiB
//   tile_mask          = ncols1 * (nbatch_fa + 8) half        = 32 * 40 * 2 B   = ~2.5 KiB
//   Total ≈ ~33 + 8.5 + 8.5 + 2.5 ≈ ~52.5 KiB. Within 96 KiB cap. 1 block/SM
//   is by design (occupancy=1) so the budget is fine.
//
// WARP DECOMPOSITION (np=2 warps per col-group, 4 col-groups × 2 warps = 8):
//   warp_group  = threadIdx.y / np    ∈ [0, nwarps/np) = [0, 4)
//   warp_in_np  = threadIdx.y % np    ∈ [0, np)        = [0, 2)
//   - warp_group selects which 16 q-cols of the 64 this warp handles
//   - warp_in_np selects which half of nbatch_fa (16+16) this warp handles
//   So each warp owns: 16 q-cols × 16 KV positions worth of D-tiles.
//   Per warp KQ_C array size: nbatch_fa/(np*T_C_KQ::J) = 32/(2*8) = 2 D-tiles.
//   Each D-tile (16q × 8n) is f32 accumulator with T_C_KQ::ne regs per lane.
//
// KEY OPS BY PHASE (matches fattn-mma-f16.cuh):
//
//   PHASE 1: load Q to tile_Q in shared (once per CTA, persistent across the
//   K-block iter loop). Layout: tile_Q[q_col, k=hdim/2 (half2)] for q_col in
//   [0, ncols). Padded stride DKQ/2 + 4.
//
//   PHASE 2: per-iter (one nbatch_fa-wide KV tile):
//     For k0_start from DKQ/2-1 down to 0, step nbatch_K2 (reverse iter for
//     MLA K reuse; we keep this since it doesn't hurt our case):
//       a) load_tile(K[k_VKQ_0..+nbatch_fa, k0_start..+nbatch_K2]) into tile_K
//          via cp_async if nstages==1 (we use nstages=1).
//       b) cp_async_wait_all(); __syncthreads().
//       c) Q·K accumulate into per-warp KQ_C registers, persistent across slabs.
//          NOTE: Q is in shared; KQ_C is in registers. NO spill of KQ to shared.
//
//   PHASE 3: register softmax. Each thread holds cols_per_thread=2 q-cols worth
//     of running KQ_max and KQ_rowsum (in registers, persistent across iters).
//     Find per-q-col max via warp __shfl_xor_sync across the lanes that share
//     the same q-col (4 lanes for Turing/Ampere wide; spread offsets 2, 1).
//     Apply expf(KQ - max), update rowsum, rescale VKQ_C by alpha = expf(old_max
//     - new_max).
//
//   PHASE 4: P·V into VKQ_C (per-warp register accumulator for the warp's hdim
//     slice). Load V via ldmatrix.x4.trans (V is the A operand in their
//     transposed formulation), MMA P (the just-softmaxed register KQ_C, used as
//     B) into VKQ_C. V is loaded once per (k_VKQ_step, dv_step); KQ_C is the
//     register array we just softmaxed (no shared spill of P).
//
//   EPILOGUE: each warp's VKQ_C / KQ_rowsum gets written to dst with proper
//     head/q-row mapping. For ncols2=2 (HP heads packed), the q-col splits into
//     head-of-pack via `j / ncols2` and `j % ncols2`.
//
// =============================================================================
//
// IMPLEMENTATION STATUS (Stage H.3, multi-session):
//   - This file is the design scaffold + foundation pieces ONLY. Phases 2-4
//     and the epilogue are intentionally STUB'd. Each will be filled in across
//     subsequent sessions with careful per-lane derivation and per-piece
//     cuda-attn-compare correctness gates. DO NOT enable in any config yet;
//     the launcher does not wire it in until the full kernel lands and beats
//     FA-2 at 256k.
//   - Foundation landed here: design comment, signature, constants, Q load,
//     shared layout, warp/lane id derivation.
//   - Reuses helpers from earlier kernel files in the concat TU:
//     aegis_mma_cp_async_16, aegis_mma_cp_async_zero_16, aegis_mma_cp_commit,
//     aegis_mma_cp_wait_all, aegis_mma_cvta_smem.
//
// =============================================================================

#if __CUDA_ARCH__ >= 800

extern "C" __global__
__launch_bounds__(256, 1)
void aegis_attention_prefill_dense_regsmx_hdim512(
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
    // ---- compile-time geometry ----
    constexpr unsigned int DKQ           = 512u;
    constexpr unsigned int DV            = 512u;
    constexpr unsigned int HP            = 2u;            // packed q-heads
    constexpr unsigned int q_block       = 32u;           // ncols1
    constexpr unsigned int ncols         = q_block * HP;  // 64
    constexpr unsigned int nthreads      = 256u;
    constexpr unsigned int nwarps        = nthreads / 32u;// 8
    constexpr unsigned int nbatch_fa     = 32u;
    constexpr unsigned int nbatch_K2     = 128u;          // half2 along DKQ
    constexpr unsigned int nbatch_V2     = 128u;          // half2 along DV
    constexpr unsigned int cols_per_warp = 16u;
    constexpr unsigned int np            = nwarps * cols_per_warp / ncols;  // 2
    constexpr unsigned int cols_per_thread = 2u;
    constexpr unsigned int q_pad         = 4u;            // half2 padding for ldmatrix
    constexpr unsigned int stride_tile_Q = DKQ/2u + q_pad; // half2 stride per q-col row
    constexpr unsigned int stride_tile_K = nbatch_K2 + q_pad;
    constexpr unsigned int stride_tile_V = nbatch_V2 + q_pad;
    (void)stride_tile_V; (void)DV;  // used in P·V phase (TODO)

    // ---- runtime args & guards ----
    const unsigned int group         = num_attention_heads / num_kv_heads;
    const unsigned int head0         = blockIdx.x * HP;
    const unsigned int kv_head       = head0 / group;
    const unsigned int global_q_base = blockIdx.y * q_block;
    const unsigned int tid           = threadIdx.x;
    const unsigned int lane          = tid & 31u;
    const unsigned int warp          = tid >> 5u;
    const unsigned int warp_group    = warp / np;            // 0..3
    const unsigned int warp_in_np    = warp % np;            // 0..1
    (void)warp_group; (void)warp_in_np;
    if (head_dim != DKQ || (group % HP) != 0u
        || head0 + HP > num_attention_heads
        || blockDim.x < nthreads) {
        return;
    }

    const unsigned int last_q_in_block = min(total_q, global_q_base + q_block) - 1u;
    const unsigned int block_max_visible = global_q_base < total_q
        ? min(context_len, start_position + last_q_in_block + 1u) : 0u;
    if (block_max_visible == 0u) return;
    const unsigned int block_min_visible_raw = (window_size > 0u
        && start_position + global_q_base + 1u > window_size)
        ? (start_position + global_q_base + 1u - window_size) : 0u;
    const unsigned int block_min_tile_start =
        (block_min_visible_raw / nbatch_fa) * nbatch_fa;

    // ---- shared layout ----
    extern __shared__ __align__(16) unsigned char smem_bytes[];
    unsigned short* tile_Q = reinterpret_cast<unsigned short*>(smem_bytes);
    // tile_Q occupies ncols rows × stride_tile_Q half2 = ncols × stride_tile_Q × 2 halfs
    unsigned short* tile_K = tile_Q + ncols * stride_tile_Q * 2u;
    // tile_K occupies nbatch_fa rows × stride_tile_K half2 = nbatch_fa × stride_tile_K × 2 halfs
    unsigned short* tile_V = tile_K + nbatch_fa * stride_tile_K * 2u;
    (void)tile_V;  // populated in PHASE 4 (TODO)

    // ---- PHASE 1: load Q (ncols q-cols × DKQ halfs) into tile_Q ----
    // Layout in shared: tile_Q[q_col, k] where q_col in [0, ncols), k in
    // [0, DKQ). For ncols2=HP=2 packing, q_col interleaves heads: q_col =
    // q_row * HP + hp. The corresponding query memory index is
    //   query[(global_q_base + q_row) * num_attention_heads + (head0 + hp)] [k]
    // Stride between q-cols in shared is stride_tile_Q half2 = (DKQ/2 + 4) half2.
    {
        constexpr unsigned int halfs_per_vec = sizeof(uint4) / sizeof(unsigned short);
        constexpr unsigned int chunks_per_row = DKQ / halfs_per_vec;  // 64
        for (unsigned int chunk = tid; chunk < ncols * chunks_per_row; chunk += nthreads) {
            const unsigned int q_col = chunk / chunks_per_row;        // 0..63
            const unsigned int hoff  = (chunk % chunks_per_row) * halfs_per_vec;
            const unsigned int q_row = q_col / HP;
            const unsigned int hp    = q_col % HP;
            const unsigned int head  = head0 + hp;
            const unsigned int global_q = global_q_base + q_row;
            uint4* dst = reinterpret_cast<uint4*>(
                tile_Q + (q_col * stride_tile_Q * 2u) + hoff);
            *dst = global_q < total_q
                ? *reinterpret_cast<const uint4*>(
                      query + (size_t(global_q) * num_attention_heads + head) * DKQ + hoff)
                : make_uint4(0u, 0u, 0u, 0u);
        }
    }
    __syncthreads();

    // ---- cp.async slab staging (K/V into tile_K/tile_V, single-buffered) ----
    // KV cache element layout: cache[slot, kv_head, hdim_col] (row-major).
    // tile dest layout: tile[kv_row in [0, nbatch_fa), k_half2 in [0, slab/2)]
    // stored with stride = stride_tile_K (or stride_tile_V) half2 per kv_row.
    // Each cp.async issues 16 bytes (= 8 halfs = 4 half2). nbatch_fa * (slab/8)
    // total chunks = 32 * 16 = 512 chunks → 256 threads × 2 chunks each.
    auto stage_slab = [&] (const unsigned short* __restrict__ cache,
                           unsigned short* __restrict__ tile_dst,
                           unsigned int   stride_tile_dst_h2,
                           unsigned int   slab_h2_base,    // k_half2 base of this slab
                           unsigned int   slab_h2_width,   // k_half2 width of this slab
                           unsigned int   tile_start,
                           unsigned int   tile_count) {
        constexpr unsigned int halfs_per_chunk  = 8u;
        const unsigned int chunks_per_row       = (slab_h2_width * 2u) / halfs_per_chunk;
        const unsigned int passes               = (nbatch_fa * chunks_per_row + nthreads - 1u) / nthreads;
#pragma unroll 2
        for (unsigned int p = 0u; p < passes; ++p) {
            const unsigned int chunk = p * nthreads + tid;
            if (chunk >= nbatch_fa * chunks_per_row) break;
            const unsigned int row  = chunk / chunks_per_row;
            const unsigned int hoff = (chunk % chunks_per_row) * halfs_per_chunk;
            const bool         ok   = row < tile_count;
            const unsigned int pos  = tile_start + row;
            // Destination in tile_dst (halfs). stride is half2; convert to halfs (*2).
            unsigned short* dst_h = tile_dst + row * (stride_tile_dst_h2 * 2u) + hoff;
            unsigned int dst_smem = aegis_mma_cvta_smem(dst_h);
            if (ok) {
                const size_t off =
                    (size_t(kv_slot(pos, cache_capacity)) * num_kv_heads + kv_head) * DKQ
                    + slab_h2_base * 2u + hoff;
                aegis_mma_cp_async_16(dst_smem, cache + off);
            } else {
                aegis_mma_cp_async_zero_16(dst_smem);
            }
        }
    };

    // Number of KV-block iterations covering [block_min_tile_start, block_max_visible).
    const unsigned int n_kiters = (block_max_visible > block_min_tile_start)
        ? ((block_max_visible - block_min_tile_start + nbatch_fa - 1u) / nbatch_fa)
        : 0u;
    if (n_kiters == 0u) {
        // (Output zeros handled in PHASE 4 epilogue; for the scaffold-only state
        // we have no output to write yet — guard added when the epilogue lands.)
        return;
    }

    // ---- persistent register state across iters (foundation declaration) ----
    // Each warp owns a per-warp-partition slice of (q-cols × kv-positions). The
    // exact slice (cols_per_warp=16 q-cols × nbatch_fa/np=16 kv) and the per-
    // lane fragment layout of KQ_C will be derived in the Phase 2 implementation
    // alongside the load_ldmatrix(tile<16,8>) primitives. This scaffold declares
    // the running scalars only — `KQ_max[cols_per_thread]` and
    // `KQ_rowsum[cols_per_thread]` per thread (cols_per_thread=2 for Turing/
    // Ampere wide), seeded with -inf / 0, updated each iter by the softmax.
    constexpr float neg_inf = -3.402823466e38f;
    float KQ_max   [cols_per_thread] = { neg_inf, neg_inf };
    float KQ_rowsum[cols_per_thread] = { 0.0f, 0.0f };
    // (VKQ_C[] register accumulator declared in Phase 4 once its size derived
    // from T_C_VKQ::ne is fixed.)

    // ============================================================
    // PHASE 2: K-block iter loop (cp.async-prologue + iter body with Q·K → KQ_C)
    //   Pseudo:
    //     stage K slab 0 (cp.async); cp_commit;
    //     for it in 0..n_kiters:
    //       cp_wait_all(); __syncthreads();          # K slab 0 ready
    //       for k0_start in DKQ/2-nbatch_K2 down to 0 step nbatch_K2:   # MLA reverse
    //         if next slab valid: stage K next slab (cp.async); cp_commit;
    //         load_ldmatrix(Q tile<16,8,half2>);
    //         load_ldmatrix(K tile<16,8,half2>);
    //         mma(KQ_C, Q, K);                       # f32-acc m16n8k16
    //         cp_wait_all(); __syncthreads();        # next slab ready (if any)
    //
    // PHASE 3: register softmax with __shfl_xor_sync row reductions + tiny
    //          cross-warp shared array (np=2 warps per q-col-group exchange
    //          their partial maxes/sums for a few hundred bytes).
    //
    // PHASE 4: P·V — load V via load_ldmatrix_trans (V as transposed-A operand),
    //          MMA the just-softmaxed register KQ_C as the B operand, accumulate
    //          into VKQ_C register array.
    //
    // EPILOGUE: VKQ_C / KQ_rowsum → output (per packed head & q_row).
    //
    // STATUS: PHASE 2 BODY remains to write. Required primitives not yet
    // vendored: load_ldmatrix(tile<16,8,half2>) variants (mma.cuh:790-798),
    // mma(tile<16,8,float>, tile<16,8,half2>, tile<8,8,half2>) (mma.cuh:1066+).
    // These get added in the next focused session, then the body fills in.
    // ============================================================

    // Suppress unused-var diagnostics for the (still-incomplete) scaffold.
    (void)start_position; (void)warp_group; (void)warp_in_np;
    (void)output; (void)lane; (void)KQ_max; (void)KQ_rowsum;
    (void)stride_tile_K; (void)stride_tile_V; (void)stage_slab; (void)tile_V;
}

#endif  // __CUDA_ARCH__ >= 800

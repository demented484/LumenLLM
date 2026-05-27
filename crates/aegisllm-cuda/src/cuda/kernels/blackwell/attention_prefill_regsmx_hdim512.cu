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

    // ============================================================
    // PHASE 2: K-block iter loop with register-resident KQ_C
    // PHASE 3: register softmax via __shfl_xor_sync (no s_shared bounce)
    // PHASE 4: P·V → VKQ_C register accumulator
    // EPILOGUE: VKQ_C / KQ_rowsum → output[2 heads × q_block × DV]
    //
    // STATUS: not yet implemented. Multi-session build planned. Foundation
    // (constants, geometry, shared layout, Q load) is in place above; the
    // remaining body is the careful per-lane MMA/shuffle/PV work that needs
    // dedicated focused sessions with cuda-attn-compare gates at each step.
    // The kernel is currently unused (launcher does not wire it in until the
    // full body lands and clears the parity gate vs FA-2).
    // ============================================================

    // Suppress unused-var diagnostics for the foundation-only scaffold.
    (void)key_cache; (void)value_cache; (void)start_position;
    (void)kv_head; (void)cache_capacity; (void)output;
    (void)block_min_tile_start; (void)lane; (void)cols_per_thread;
    (void)stride_tile_K;
}

#endif  // __CUDA_ARCH__ >= 800

// =============================================================================
// Hand-tuned mma.sync FP16 FlashAttention prefill kernel (Stage D.1).
// head_dim = 512, dense (non-paged), causal, GQA-aware.
// =============================================================================
//
// First-cut port of llama.cpp `fattn-mma-f16.cuh`'s structure to aegisllm:
// raw `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` PTX issuance with
// a register-resident online-softmax. No `nvcuda::wmma` — the entire MMA path
// is hand-issued PTX so register tiling, accumulator layout, and softmax
// rescale are all under direct control of the kernel.
//
// Data type: the engine's prefill path converts Q to FP16 (`f32_to_f16_device`)
// and the KV cache is FP16 in the global-layer dense path (norm_rope_kv stores
// `float_to_f16_bits`). The MMA must therefore be the FP16 variant — feeding
// FP16 bit-patterns into a `.bf16.bf16` MMA reads them as BF16 and produces
// garbage (this was the Stage D.1 catastrophic-drift bug, layer-29 cos 0.26).
//
// Tile geometry mirrors the FA-2 kernel
// (`attention_prefill_dense_fa2_hdim512`) so the host launcher's shared-memory
// budget is unchanged and well inside the 96 KiB sm_120 dynamic-shared cap:
//
//   q_block        = 32     query rows / block
//   kv_block       = 64     KV positions per mainloop iter
//   hdim           = 512
//   slab           = 128    hdim slab width streamed through shared
//   n_slabs        = 4      hdim / slab
//   warps          = 16
//   row_strips     = 2      q_block / 16         (MMA M tiles down Q)
//   kv_strips      = 8      kv_block / 8         (MMA N tiles across KV — N=8)
//   cols_per_warp  = 32     hdim / warps         (O columns each warp owns)
//   o_col_frags    = 4      cols_per_warp / 8    (O is N=8 frags wide per warp)
//   o_frags        = 8      row_strips * o_col_frags
//
// Compared to the FA-2 WMMA kernel:
//   * mma.sync `m16n8k16` is the native Ampere/Ada/Blackwell BF16 MMA shape.
//     wmma's 16x16x16 internally decomposes into two m16n8k16; using mma.sync
//     directly halves the f32 accumulator regs per tile (4 instead of 8) and
//     gives stable lane->element mapping for the rescale.
//   * Per-thread fragment layouts are well-defined and stable, so the online
//     softmax alpha rescale applies directly to the f32 accumulator with a
//     plain register-resident mul (no helper, no shared-memory round-trip).
//   * Q*K is fully warp-tiled: row_strips(2) * kv_strips(8) = 16 = warps, so
//     every warp owns ONE complete [16q, 8kv] S tile and there is no
//     cross-warp partial reduction. Each warp computes its slice across the
//     full hdim=512 contraction (4 slabs * 8 kk-steps = 32 MMA per S tile).
//   * P*V keeps the FA-2 plan: 16 warps cover hdim=512 in 32-col slices, each
//     warp accumulates 8 persistent f32 fragments (4 cols * 2 rows).
//
// Shared-memory layout (peak ~76.4 KiB, identical to FA-2 hdim=512):
//   q_shared      = q_block*hdim   bf16  = 32 KiB   (persistent)
//   kv_slab[2]    = 2*kv_block*128 bf16  = 32 KiB   (cp.async double-buffered)
//   s_shared      = q_block*kv_block f32 =  8 KiB   (S tile)
//   weights_f16  = q_block*kv_block bf16=  4 KiB   (P in BF16 for P*V)
//   scalars       = q_block*3       f32  =  0.4 KiB ([m, l, alpha] per q row)
//                                          --------
//                                          ~76.4 KiB
//
// Grid: (num_attention_heads, ceil(total_q / q_block)). Block: 512 threads.
//
// =============================================================================
// PTX m16n8k16.row.col.f32.f16.f16.f32 — per-thread fragment layout
// =============================================================================
// (from PTX ISA "Matrix Fragments for mma.m16n8k16"; layout for `.f16.f16` is
// identical to `.bf16.bf16` — the operand registers are 16-bit elements packed
// 2-per-u32 either way; only the instruction's numeric interpretation differs):
//
//   A (m=16, k=16, FP16), 4 u32 regs/thread, holding 8 FP16 elements:
//     a0..a1 (reg 0): A[lane/4,     (lane%4)*2 + 0..1]
//     a2..a3 (reg 1): A[lane/4 + 8, (lane%4)*2 + 0..1]
//     a4..a5 (reg 2): A[lane/4,     (lane%4)*2 + 8..9]
//     a6..a7 (reg 3): A[lane/4 + 8, (lane%4)*2 + 8..9]
//
//   B (n=8, k=16, FP16), 2 u32 regs/thread, holding 4 FP16 elements
//   (col-major in PTX: row=K, col=N):
//     b0..b1 (reg 0): B[(lane%4)*2 + 0..1, lane/4]
//     b2..b3 (reg 1): B[(lane%4)*2 + 8..9, lane/4]
//
//   D (m=16, n=8, f32), 4 f32 regs/thread:
//     d0: D[lane/4,     (lane%4)*2 + 0]
//     d1: D[lane/4,     (lane%4)*2 + 1]
//     d2: D[lane/4 + 8, (lane%4)*2 + 0]
//     d3: D[lane/4 + 8, (lane%4)*2 + 1]
// =============================================================================

#if __CUDA_ARCH__ >= 800

// 16 B cp.async load (HBM -> shared, .ca cache policy).
__device__ __forceinline__ void aegis_mma_cp_async_16(unsigned int dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :: "r"(dst), "l"(src));
}

// 16 B cp.async ZERO fill (zero-pads OOB rows of the KV tile so the MMA reads
// well-defined zeros at the tail of the final KV block — no mask needed).
__device__ __forceinline__ void aegis_mma_cp_async_zero_16(unsigned int dst) {
    const unsigned long long z = 0ULL;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, 0;\n"
                 :: "r"(dst), "l"((const void*)&z));
}

__device__ __forceinline__ void aegis_mma_cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void aegis_mma_cp_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::);
}

// Convert a generic pointer to a shared-memory address (.shared u32).
__device__ __forceinline__ unsigned int aegis_mma_cvta_smem(const void* p) {
    unsigned int s;
    asm volatile("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }\n"
                 : "=r"(s) : "l"(p));
    return s;
}

// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 — issue ONE MMA.
//   D[16,8] = A[16,16] * B[16,8] + D[16,8]
// (B is K-major in PTX semantics, i.e. B[k,n].)
__device__ __forceinline__ void aegis_mma_m16n8k16_f16(
    float& d0, float& d1, float& d2, float& d3,
    unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Pack two adjacent FP16 elements (stored as `unsigned short`) into one u32.
// PTX FP16 inputs to mma.sync are u32 with low half = element 0, high half =
// element 1 (little-endian f16x2 layout — same byte layout as bf16x2).
__device__ __forceinline__ unsigned int aegis_pack_f16x2(
    unsigned short lo, unsigned short hi
) {
    return (unsigned int)lo | ((unsigned int)hi << 16);
}

// Load the A operand (m16k16 BF16) for lane `lane` from a row-major [M=16, K]
// source in shared memory. `m_base` is the source row of element (0,0);
// `k_base` is the source col of element (0,0). `stride` is the row stride in
// BF16 elements.
//
// PTX A-fragment layout: each lane holds 8 BF16 elements packed into 4 u32:
//   reg 0 = pack(A[r0, c0+0], A[r0, c0+1])   r0 = m_base + lane/4
//   reg 1 = pack(A[r1, c0+0], A[r1, c0+1])   r1 = m_base + lane/4 + 8
//   reg 2 = pack(A[r0, c8+0], A[r0, c8+1])   c8 = c0 + 8
//   reg 3 = pack(A[r1, c8+0], A[r1, c8+1])
//   c0 = k_base + (lane%4)*2
__device__ __forceinline__ void aegis_mma_load_a_m16k16(
    unsigned int& r0, unsigned int& r1, unsigned int& r2, unsigned int& r3,
    const unsigned short* __restrict__ src, unsigned int stride
) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int row_upper = (lane >> 2);          // 0..7
    const unsigned int row_lower = row_upper + 8u;       // 8..15
    const unsigned int col_lo    = (lane & 3u) << 1;     // 0,2,4,6
    const unsigned int col_hi    = col_lo + 8u;
    const unsigned short* p_upper = src + row_upper * stride;
    const unsigned short* p_lower = src + row_lower * stride;
    r0 = aegis_pack_f16x2(p_upper[col_lo],     p_upper[col_lo + 1u]);
    r1 = aegis_pack_f16x2(p_lower[col_lo],     p_lower[col_lo + 1u]);
    r2 = aegis_pack_f16x2(p_upper[col_hi],     p_upper[col_hi + 1u]);
    r3 = aegis_pack_f16x2(p_lower[col_hi],     p_lower[col_hi + 1u]);
}

// Load the B operand (n8k16 BF16) for the "n=N-rows, k=K-cols logical
// transpose" case — when the source memory is [N=8, K=16] row-major (K
// contiguous), as both K-cache and V-cache are in their natural row-major
// "[kv_pos, hdim]" layout. Here `src` points to element (0,0) of the [8, K]
// tile and `stride` is the row stride in BF16 elements.
//
// PTX B-fragment layout (col-major: B[k,n]):
//   reg 0 = pack(B[k0+0, n], B[k0+1, n])    n  = lane/4
//   reg 1 = pack(B[k8+0, n], B[k8+1, n])    k0 = (lane%4)*2,  k8 = k0 + 8
//
// Since our source is [N, K] row-major (N rows, K cols), B[k, n] is src[n, k]:
//   B[k0+0, n] = src[n, k0+0]   etc.
__device__ __forceinline__ void aegis_mma_load_b_n8k16_from_nk(
    unsigned int& r0, unsigned int& r1,
    const unsigned short* __restrict__ src, unsigned int stride
) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int n_idx = (lane >> 2);              // 0..7
    const unsigned int k_lo  = (lane & 3u) << 1;         // 0,2,4,6
    const unsigned int k_hi  = k_lo + 8u;
    const unsigned short* p_n = src + n_idx * stride;
    r0 = aegis_pack_f16x2(p_n[k_lo],     p_n[k_lo + 1u]);
    r1 = aegis_pack_f16x2(p_n[k_hi],     p_n[k_hi + 1u]);
}

// Apply a per-row alpha (online-softmax rescale) to a 4-element m16n8 f32
// accumulator. Each lane owns rows {lane/4, lane/4 + 8}; elements 0,1 are in
// the upper row, 2,3 in the lower row. Warp-uniform early-out via ballot.
__device__ __forceinline__ void aegis_mma_scale_m16n8_rows(
    float& d0, float& d1, float& d2, float& d3,
    const float* __restrict__ scalars,
    unsigned int row_base
) {
    const unsigned int lane = threadIdx.x & 31u;
    const float a_upper = scalars[(row_base + (lane >> 2))      * 3u + 2u];
    const float a_lower = scalars[(row_base + (lane >> 2) + 8u) * 3u + 2u];
    if (__ballot_sync(0xffffffffu, (a_upper == 1.0f) & (a_lower == 1.0f)) == 0xffffffffu) {
        return;
    }
    d0 *= a_upper;
    d1 *= a_upper;
    d2 *= a_lower;
    d3 *= a_lower;
}

extern "C" __global__
__launch_bounds__(512, 1)
void aegis_attention_prefill_dense_mma_hdim512(
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
    constexpr unsigned int kv_block      = 64u;
    constexpr unsigned int warps         = 16u;
    constexpr unsigned int slab          = 128u;
    constexpr unsigned int n_slabs       = hdim / slab;          // 4
    constexpr unsigned int row_strips    = q_block / 16u;        // 2
    constexpr unsigned int kv_strips     = kv_block / 8u;        // 8
    constexpr unsigned int cols_per_warp = hdim / warps;         // 32
    constexpr unsigned int o_col_frags   = cols_per_warp / 8u;   // 4
    constexpr unsigned int o_frags       = row_strips * o_col_frags; // 8
    constexpr unsigned int warps_per_slab = slab / cols_per_warp;    // 4
    constexpr unsigned int slab_kk       = slab / 16u;           // 8

    const unsigned int head          = blockIdx.x;
    const unsigned int global_q_base = blockIdx.y * q_block;
    const unsigned int tid           = threadIdx.x;
    const unsigned int lane          = tid & 31u;
    const unsigned int warp          = tid >> 5u;
    (void)lane;
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

    // --- shared layout (identical to FA-2 hdim=512) ---
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned short* q_shared    = reinterpret_cast<unsigned short*>(smem);
    unsigned short* kv_slab     = q_shared + q_block * hdim;             // 2*kv_block*slab
    float*          s_shared    = reinterpret_cast<float*>(kv_slab + 2u * kv_block * slab);
    unsigned short* weights_f16 = reinterpret_cast<unsigned short*>(s_shared + q_block * kv_block);
    float*          scalars     = reinterpret_cast<float*>(weights_f16 + q_block * kv_block);

    const unsigned int group   = num_attention_heads / num_kv_heads;
    const unsigned int kv_head = head / group;
    const float scale  = rsqrtf(float(hdim));
    const float log2e  = 1.4426950408889634f;
    const float neg_inf = -3.402823466e38f;

    // --- load Q tile once (whole hdim, persistent in shared) ---
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

    // --- persistent register-resident O accumulator (8 frags/warp, 4 f32 each) ---
    float o_acc[o_frags][4];
#pragma unroll
    for (unsigned int f = 0u; f < o_frags; ++f) {
#pragma unroll
        for (unsigned int e = 0u; e < 4u; ++e) {
            o_acc[f][e] = 0.0f;
        }
    }

    // --- cp.async slab staging ---
    // kv_block * slab = 64 * 128 = 8192 halfs = 1024 16-B chunks. 512 threads
    // -> 2 chunks each.
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

    // Q*K warp -> S tile assignment. S is [q_block=32, kv_block=64] decomposed
    // into row_strips(2) * kv_strips(8) = 16 MMA tiles. Each warp owns exactly
    // ONE S tile of shape [16q, 8kv] — the warp grid is fully populated.
    const unsigned int qk_row = warp / kv_strips;                 // 0..1
    const unsigned int qk_kv  = warp % kv_strips;                 // 0..7

    // P*V warp -> hdim-slab / column ownership (all 16 warps active).
    const unsigned int o_slab     = warp / warps_per_slab;        // 0..3
    const unsigned int o_col_base = (warp % warps_per_slab) * cols_per_warp; // col in slab

    // Prologue: stage K slab 0 of the first KV block.
    if (n_kiters > 0u) {
        const unsigned int tc0 = min(kv_block, block_max_visible - block_min_tile_start);
        stage_slab(key_cache, block_min_tile_start, 0u, kv_slab, tc0);
        aegis_mma_cp_commit();
    }
    __syncthreads();

    // ----------------------------- mainloop --------------------------------
    for (unsigned int it = 0u; it < n_kiters; ++it) {
        const unsigned int tile_start = block_min_tile_start + it * kv_block;
        const unsigned int tile_count = min(kv_block, block_max_visible - tile_start);

        // ======================= Q*K -> S =======================
        // s_acc holds this warp's [16q, 8kv] S tile in 4 f32 regs/lane.
        float s_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        // Software-pipelined hdim-slab streaming with one __syncthreads per
        // slab (covers RAW for the just-arrived slab AND WAR for the prev
        // slab's MMA reads).
        for (unsigned int sl = 0u; sl < n_slabs; ++sl) {
            const unsigned int buf = sl & 1u;
            unsigned short* k_buf = kv_slab + buf * (kv_block * slab);
            aegis_mma_cp_wait_all();
            __syncthreads();
            if (sl + 1u < n_slabs) {
                stage_slab(key_cache, tile_start, sl + 1u,
                           kv_slab + ((sl + 1u) & 1u) * (kv_block * slab), tile_count);
                aegis_mma_cp_commit();
            }

            // For this slab: contract Q[qk_row*16..+16, sl*slab..+128] with
            // K[qk_kv*8..+8, sl*slab..+128]^T into s_acc[16q,8kv]. The MMA's
            // K dim is hdim (16 BF16 elements per MMA), so we decompose into
            // slab_kk=8 m16n8k16 MMAs.
#pragma unroll
            for (unsigned int kk = 0u; kk < slab_kk; ++kk) {
                const unsigned int a_row_base = qk_row * 16u;
                // Q is held in `q_shared` over the FULL hdim, so its col base
                // is hdim-absolute (slab offset + kk step).
                const unsigned int a_col_base = sl * slab + kk * 16u;
                const unsigned int b_row_base = qk_kv * 8u;
                // K is loaded one slab at a time into `k_buf` (only `slab`
                // cols), so its col base is in-slab — the hdim slab offset is
                // implicit in which slab `k_buf` currently holds. (Including
                // `sl * slab` here read past the slab buffer into adjacent
                // shared, causing CUDA_ERROR_ILLEGAL_ADDRESS for sl>0.)
                const unsigned int b_col_base = kk * 16u;
                unsigned int a0, a1, a2, a3, b0, b1;
                aegis_mma_load_a_m16k16(a0, a1, a2, a3,
                    q_shared + a_row_base * hdim + a_col_base, hdim);
                aegis_mma_load_b_n8k16_from_nk(b0, b1,
                    k_buf + b_row_base * slab + b_col_base, slab);
                aegis_mma_m16n8k16_f16(
                    s_acc[0], s_acc[1], s_acc[2], s_acc[3],
                    a0, a1, a2, a3, b0, b1);
            }
        }

        // Spill this warp's S tile to s_shared[16q, 8kv].
        // Lane mapping (m16n8 f32 accumulator):
        //   d0,d1 -> row = qk_row*16 + lane/4,     col = qk_kv*8 + (lane%4)*2 + {0,1}
        //   d2,d3 -> row = qk_row*16 + lane/4 + 8, col = qk_kv*8 + (lane%4)*2 + {0,1}
        {
            const unsigned int r_upper = qk_row * 16u + (lane >> 2);
            const unsigned int r_lower = r_upper + 8u;
            const unsigned int c_base  = qk_kv * 8u + (lane & 3u) * 2u;
            s_shared[r_upper * kv_block + c_base + 0u] = s_acc[0];
            s_shared[r_upper * kv_block + c_base + 1u] = s_acc[1];
            s_shared[r_lower * kv_block + c_base + 0u] = s_acc[2];
            s_shared[r_lower * kv_block + c_base + 1u] = s_acc[3];
        }
        __syncthreads();

        // ======================= online softmax =======================
        // Each warp owns 2 q rows (row, row + 16). 64 KV columns / 32 lanes ->
        // each lane handles columns {lane, lane + 32}.
#pragma unroll
        for (unsigned int rr = 0u; rr < 2u; ++rr) {
            const unsigned int row = warp + rr * warps;
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
                // P is consumed by the FP16 MMA in the P·V phase, so the
                // softmax weights are stored as raw FP16 bits (matches the
                // FP16 Q / FP16 KV-cache used elsewhere in the kernel).
                weights_f16[row * kv_block + col] =
                    __half_as_ushort(__float2half_rn(w[c]));
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
        // alpha applied to the m16n8 f32 accumulator. Elements 0,1 are in the
        // upper row (lane/4); 2,3 in the lower row (lane/4 + 8).
#pragma unroll
        for (unsigned int rs = 0u; rs < row_strips; ++rs) {
#pragma unroll
            for (unsigned int cf = 0u; cf < o_col_frags; ++cf) {
                float* d = o_acc[rs * o_col_frags + cf];
                aegis_mma_scale_m16n8_rows(d[0], d[1], d[2], d[3], scalars, rs * 16u);
            }
        }

        // ======================= P*V -> O =======================
        // O[q_block=32, hdim=512] += P[q_block=32, kv_block=64] . V[kv=64, hdim].
        // Each warp owns hdim cols [o_slab*slab + o_col_base, +32). Stream V
        // slabs in the same 1-barrier-per-slab pipeline as Q*K. Only the
        // warps owning the current slab do MMA work; the others coast.
        //
        // MMA decomposition (per warp, only when o_slab == sl):
        //   For ks in 0..kv_strips(8):
        //     Load B (V) once per (cf, ks): V[ks*16..+16 kv (K-dim),
        //                                     o_col_base+cf*8..+8 (N-dim)].
        //     For rs in 0..row_strips(2):
        //       Load A (P) [rs*16..+16, ks*16..+16] and MMA into o_acc[rs,cf].
        stage_slab(value_cache, tile_start, 0u, kv_slab, tile_count);
        aegis_mma_cp_commit();
        for (unsigned int sl = 0u; sl < n_slabs; ++sl) {
            const unsigned int buf = sl & 1u;
            unsigned short* v_buf = kv_slab + buf * (kv_block * slab);
            aegis_mma_cp_wait_all();
            __syncthreads();
            if (sl + 1u < n_slabs) {
                stage_slab(value_cache, tile_start, sl + 1u,
                           kv_slab + ((sl + 1u) & 1u) * (kv_block * slab), tile_count);
                aegis_mma_cp_commit();
            }
            if (o_slab == sl) {
                // P·V K dim contracts over kv_block (64 KV positions); MMA
                // k=16 → 4 ks-steps. `kv_strips=8` is Q·K's N=8 step count
                // and would OOB the kv_block-row slab buffer (reads rows
                // 64..127 from a 64-row buffer → ILLEGAL_ADDRESS).
                constexpr unsigned int pv_k_strips = kv_block / 16u;  // 4
#pragma unroll
                for (unsigned int ks = 0u; ks < pv_k_strips; ++ks) {
#pragma unroll
                    for (unsigned int cf = 0u; cf < o_col_frags; ++cf) {
                        const unsigned int v_row_base = ks * 16u;
                        const unsigned int v_col_base = o_col_base + cf * 8u;
                        // V is [kv_row, hdim] in shared: [K=16 kv, N=8 hdim]
                        // — but ldmatrix expects B with rows=N, so we use the
                        // "from_nk" loader (which interprets src as [N, K]).
                        // Here src is K-major; we need to swap. So instead use
                        // the n8k16 loader with the alternate layout: source
                        // is [K=16 rows, N=8 cols], B[k,n] = src[k, n], i.e.
                        // a simple 2D load with N varying along col.
                        //
                        // Compute the four BF16 elements per lane directly:
                        //   reg 0 = pack(src[k0+0, n], src[k0+1, n])
                        //   reg 1 = pack(src[k8+0, n], src[k8+1, n])
                        // with n  = lane/4
                        //      k0 = (lane%4)*2
                        //      k8 = k0 + 8
                        const unsigned int n_idx = (lane >> 2);
                        const unsigned int k_lo  = (lane & 3u) << 1;
                        const unsigned int k_hi  = k_lo + 8u;
                        const unsigned short* src_v = v_buf + v_row_base * slab + v_col_base;
                        const unsigned int n_col = n_idx;
                        const unsigned int v0 = aegis_pack_f16x2(
                            src_v[(k_lo + 0u) * slab + n_col],
                            src_v[(k_lo + 1u) * slab + n_col]);
                        const unsigned int v1 = aegis_pack_f16x2(
                            src_v[(k_hi + 0u) * slab + n_col],
                            src_v[(k_hi + 1u) * slab + n_col]);

#pragma unroll
                        for (unsigned int rs = 0u; rs < row_strips; ++rs) {
                            const unsigned int p_row_base = rs * 16u;
                            const unsigned int p_col_base = ks * 16u;
                            unsigned int p0, p1, p2, p3;
                            aegis_mma_load_a_m16k16(p0, p1, p2, p3,
                                weights_f16 + p_row_base * kv_block + p_col_base, kv_block);
                            float* d = o_acc[rs * o_col_frags + cf];
                            aegis_mma_m16n8k16_f16(
                                d[0], d[1], d[2], d[3],
                                p0, p1, p2, p3, v0, v1);
                        }
                    }
                }
            }
        }

        // Prologue for next iter: stage its K slab 0.
        if (it + 1u < n_kiters) {
            const unsigned int next_start = tile_start + kv_block;
            const unsigned int next_tc = min(kv_block, block_max_visible - next_start);
            stage_slab(key_cache, next_start, 0u, kv_slab, next_tc);
            aegis_mma_cp_commit();
        }
    }

    // ============================ epilogue ============================
    // For each (row_strip, col_frag) fragment of this warp, write its 4 f32
    // values divided by the per-row sum. Lane -> element mapping (m16n8 D):
    //   d0: (r_upper, c_base + 0)    r_upper = rs*16 + lane/4
    //   d1: (r_upper, c_base + 1)
    //   d2: (r_lower, c_base + 0)    r_lower = r_upper + 8
    //   d3: (r_lower, c_base + 1)
    //   c_base = o_slab*128 + o_col_base + cf*8 + (lane%4)*2
    __syncthreads();
#pragma unroll
    for (unsigned int rs = 0u; rs < row_strips; ++rs) {
#pragma unroll
        for (unsigned int cf = 0u; cf < o_col_frags; ++cf) {
            const float* d = o_acc[rs * o_col_frags + cf];
            const unsigned int r_upper = rs * 16u + (lane >> 2);
            const unsigned int r_lower = r_upper + 8u;
            const unsigned int c_base  = o_slab * slab + o_col_base + cf * 8u + (lane & 3u) * 2u;

            const unsigned int gq_upper = global_q_base + r_upper;
            const unsigned int gq_lower = global_q_base + r_lower;

            if (gq_upper < total_q) {
                const float denom = fmaxf(scalars[r_upper * 3u + 1u], 1.0e-20f);
                const float inv   = 1.0f / denom;
                output[(size_t(gq_upper) * num_attention_heads + head) * hdim + c_base + 0u] =
                    d[0] * inv;
                output[(size_t(gq_upper) * num_attention_heads + head) * hdim + c_base + 1u] =
                    d[1] * inv;
            }
            if (gq_lower < total_q) {
                const float denom = fmaxf(scalars[r_lower * 3u + 1u], 1.0e-20f);
                const float inv   = 1.0f / denom;
                output[(size_t(gq_lower) * num_attention_heads + head) * hdim + c_base + 0u] =
                    d[2] * inv;
                output[(size_t(gq_lower) * num_attention_heads + head) * hdim + c_base + 1u] =
                    d[3] * inv;
            }
        }
    }
}

#endif  // __CUDA_ARCH__ >= 800

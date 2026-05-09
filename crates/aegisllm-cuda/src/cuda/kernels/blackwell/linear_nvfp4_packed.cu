extern "C" __global__ void aegis_blackwell_nvfp4_linear_probe(
    const unsigned char* packed,
    const unsigned char* scales,
    const float* input,
    const unsigned int rows,
    const unsigned int cols,
    const float input_scale,
    const float output_scale,
    float* output
) {
    const unsigned int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    // This deliberately touches all operands so the Rust path validates residency,
    // stream ordering and launch argument ABI. On Blackwell it also executes a
    // real native FP4 block-scaled MMA instruction; the production kernel uses
    // a separate MXFP4 repacked layout rather than this ABI probe path.
    const unsigned int packed_cols = cols / 2u;
    const unsigned int scale_cols = (cols / 64u) * 4u;
    const unsigned int base = row * packed_cols;
    const unsigned int scale_base = row * scale_cols;
    const float guard = float(packed[base] & 0x0fu)
        + float(scales[scale_base])
        + input[0] * input_scale
        + output_scale * 0.0f;
#if __CUDA_ARCH__ >= 1200
    unsigned int a0 = 0u;
    unsigned int a1 = 0u;
    unsigned int a2 = 0u;
    unsigned int a3 = 0u;
    unsigned int b0 = 0u;
    unsigned int b1 = 0u;
    unsigned int scale_a = 0u;
    unsigned int scale_b = 0u;
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    asm volatile(
        "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
        "%10, {0, 0}, %11, {0, 0};"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(scale_a), "r"(scale_b));
    if (threadIdx.x == 0) {
        output[row] = guard * 0.0f + (d0 + d1 + d2 + d3) * 0.0f;
    }
#else
    if (threadIdx.x == 0) {
        output[row] = guard * 0.0f;
    }
#endif
}

extern "C" __global__ void aegis_nvfp4_linear_prequantized(
    const unsigned char* packed,
    const unsigned char* scales,
    const float* input,
    const unsigned int rows,
    const unsigned int cols,
    const float output_scale,
    float* output
) {
    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }

    const unsigned int packed_cols = cols / 2u;
    const unsigned int scale_cols = cols / 16u;
    const unsigned char* packed_row = packed + size_t(row) * packed_cols;
    const unsigned char* scale_row = scales + size_t(row) * scale_cols;
    extern __shared__ float partial[];

    float sum = 0.0f;
    for (unsigned int block_idx = tid; block_idx < scale_cols; block_idx += blockDim.x) {
        const float block_scale = decode_ue4m3_half(scale_row[block_idx]);
        const unsigned int input_base = block_idx * 16u;
        const unsigned int packed_base = block_idx * 8u;
        for (unsigned int j = 0u; j < 8u; ++j) {
            const unsigned int byte = packed_row[packed_base + j];
            const unsigned int lo_col = input_base + 2u*j;
            const unsigned int hi_col = lo_col + 1u;
            sum += float(decode_nvfp4_nibble(byte & 0x0Fu)) * block_scale * input[lo_col];
            sum += float(decode_nvfp4_nibble(byte >> 4)) * block_scale * input[hi_col];
        }
    }

    partial[tid] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0u) {
        output[row] = partial[0] * output_scale;
    }
}

extern "C" __global__ void aegis_nvfp4_linear_prequantized_batched(
    const unsigned char* packed,
    const unsigned char* scales,
    const float* input,
    const unsigned int rows,
    const unsigned int cols,
    const float output_scale,
    float* output
) {
    const unsigned int row = blockIdx.x;
    const unsigned int batch = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }

    const unsigned int packed_cols = cols / 2u;
    const unsigned int scale_cols = cols / 16u;
    const unsigned char* packed_row = packed + size_t(row) * packed_cols;
    const unsigned char* scale_row = scales + size_t(row) * scale_cols;
    const float* input_row = input + size_t(batch) * cols;
    extern __shared__ float partial[];

    float sum = 0.0f;
    for (unsigned int block_idx = tid; block_idx < scale_cols; block_idx += blockDim.x) {
        const float block_scale = decode_ue4m3_half(scale_row[block_idx]);
        const unsigned int input_base = block_idx * 16u;
        const unsigned int packed_base = block_idx * 8u;
        for (unsigned int j = 0u; j < 8u; ++j) {
            const unsigned int byte = packed_row[packed_base + j];
            const unsigned int lo_col = input_base + 2u*j;
            const unsigned int hi_col = lo_col + 1u;
            sum += float(decode_nvfp4_nibble(byte & 0x0Fu)) * block_scale * input_row[lo_col];
            sum += float(decode_nvfp4_nibble(byte >> 4)) * block_scale * input_row[hi_col];
        }
    }

    partial[tid] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0u) {
        output[size_t(batch) * rows + row] = partial[0] * output_scale;
    }
}

extern "C" __global__ void aegis_nvfp4_linear_reference(
    const unsigned char* packed,
    const unsigned char* scales,
    const float* input,
    const unsigned int rows,
    const unsigned int cols,
    const float input_scale,
    const float output_scale,
    float* output
) {
    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }

    const unsigned int packed_cols = cols / 2u;
    const unsigned int scale_cols = cols / 16u;
    const unsigned char* packed_row = packed + size_t(row) * packed_cols;
    const unsigned char* scale_row = scales + size_t(row) * scale_cols;
    extern __shared__ float partial[];

    float sum = 0.0f;
    for (unsigned int block_idx = tid; block_idx < scale_cols; block_idx += blockDim.x) {
        const float block_scale = decode_ue4m3_half(scale_row[block_idx]);
        const unsigned int input_base = block_idx * 16u;
        const unsigned int packed_base = block_idx * 8u;
        for (unsigned int j = 0u; j < 8u; ++j) {
            const unsigned int byte = packed_row[packed_base + j];
            const unsigned int lo_lane = 2u*j;
            const unsigned int hi_lane = lo_lane + 1u;
            const float input_lo = maybe_quantize_nvfp4_input(input, input_base, lo_lane, input_scale);
            const float input_hi = maybe_quantize_nvfp4_input(input, input_base, hi_lane, input_scale);
            sum += float(decode_nvfp4_nibble(byte & 0x0Fu)) * block_scale * input_lo;
            sum += float(decode_nvfp4_nibble(byte >> 4)) * block_scale * input_hi;
        }
    }

    partial[tid] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0u) {
        output[row] = partial[0] * output_scale;
    }
}

// Shared inner kernel of the prequantized batched GEMM and its grouped-MoE
// counterpart. Centralising the inner-loop arithmetic in a single
// __forceinline__ function guarantees both kernels compile to byte-identical
// SASS for the dot product, which keeps the moe.rs grouped-MoE path
// bit-exact with the per-expert reference path — Gemma 4's softmax+top-k
// router amplifies even single-ULP per-element drift into flipped expert
// selections within a few layers.
extern "C" __device__ __forceinline__ void aegis_nvfp4_prequant_gemm_inner(
    const unsigned char* sh_packed,
    const unsigned char* sh_scales,
    const float*         input_row,
    const unsigned int   scale_cols,
    const unsigned int   lane,
    float&               sum_inout
) {
    float sum = sum_inout;
    for (unsigned int block_idx = lane; block_idx < scale_cols; block_idx += 32u) {
        const float blk_scale = decode_ue4m3_half(sh_scales[block_idx]);
        const unsigned int input_base  = block_idx * 16u;
        const unsigned int packed_base = block_idx * 8u;
        for (unsigned int j = 0u; j < 8u; ++j) {
            const unsigned int byte = sh_packed[packed_base + j];
            sum += (float)decode_nvfp4_nibble(byte & 0x0Fu) * blk_scale
                 * input_row[input_base + 2u*j];
            sum += (float)decode_nvfp4_nibble(byte >> 4u)   * blk_scale
                 * input_row[input_base + 2u*j + 1u];
        }
    }
    for (unsigned int stride = 16u; stride > 0u; stride >>= 1u) {
        sum += __shfl_down_sync(0xffffffffu, sum, stride);
    }
    sum_inout = sum;
}

// Tiled batched GEMM for NVfp4 packed weights.
//
// Grid : (rows, ceil(total_batch / BATCH_PER_BLOCK))
// Block: 256 threads = BATCH_PER_BLOCK warps × WARP_SIZE lanes
//
// Each block handles ONE weight row × BATCH_PER_BLOCK batch elements.
// The weight row (packed + scales) is loaded into shared memory ONCE by
// all 256 threads collaboratively, then every warp reads it from L1
// while processing its own batch element.  This eliminates the 669×
// redundant global-memory reloads of the same row that the original
// per-(row,batch) kernel suffered from.
extern "C" __global__ void aegis_nvfp4_linear_prequantized_batched_gemm(
    const unsigned char* __restrict__ packed,
    const unsigned char* __restrict__ scales,
    const float*         __restrict__ input,
    const unsigned int rows,
    const unsigned int cols,
    const unsigned int total_batch,
    const float output_scale,
    float*               __restrict__ output
) {
    const unsigned int WARP_SIZE      = 32u;
    const unsigned int BATCH_PER_BLOCK = 8u;

    const unsigned int row        = blockIdx.x;
    const unsigned int batch_base = blockIdx.y * BATCH_PER_BLOCK;
    if (row >= rows) return;

    const unsigned int tid      = threadIdx.x;
    const unsigned int warp_id  = tid / WARP_SIZE;
    const unsigned int lane     = tid & (WARP_SIZE - 1u);
    const unsigned int batch_idx = batch_base + warp_id;

    const unsigned int packed_cols = cols / 2u;
    const unsigned int scale_cols  = cols / 16u;

    // Shared memory: [packed_cols bytes | scale_cols bytes]
    extern __shared__ unsigned char sh[];
    unsigned char* sh_packed = sh;
    unsigned char* sh_scales = sh + packed_cols;

    // Collaborative load of weight row by all 256 threads.
    const unsigned char* w_packed = packed + (size_t)row * packed_cols;
    const unsigned char* w_scales = scales + (size_t)row * scale_cols;
    for (unsigned int i = tid; i < packed_cols; i += 256u) sh_packed[i] = w_packed[i];
    for (unsigned int i = tid; i < scale_cols;  i += 256u) sh_scales[i] = w_scales[i];
    __syncthreads();

    if (batch_idx >= total_batch) return;

    const float* input_row = input + (size_t)batch_idx * cols;

    float sum = 0.0f;
    aegis_nvfp4_prequant_gemm_inner(sh_packed, sh_scales, input_row, scale_cols, lane, sum);
    if (lane == 0u) {
        output[(size_t)batch_idx * rows + row] = sum * output_scale;
    }
}

// =============================================================================
// WMMA BF16 tensor-core variant of the prequantized batched GEMM.
//
// Replaces the scalar 1-warp-per-row reduction in
// `aegis_nvfp4_linear_prequantized_batched_gemm` with a tiled BF16 WMMA path.
// Per K-tile we dequant a 16x16 block of NVFP4 weights to BF16 in shared
// memory, convert a 16x16 block of f32 input to BF16, then run one
// `mma.sync` (m16n16k16 bf16 → f32). Accumulates in f32 across all K-tiles
// and stores back as `acc * output_scale`.
//
// Portability: m16n16k16 BF16 WMMA is supported on SM 7.5+ (Turing and
// newer NVIDIA GPUs). Native FP4 mma.sync (mxf4nvf4) is Blackwell-only and
// stays as a separate optional kernel.
//
// Grid : (ceil(rows/16), ceil(total_batch/16))
// Block: 32 threads (one warp)
// Shared mem: 2 * 16*16 bytes for BF16 tiles = 1 KiB; plus 16*16 floats for
//   output staging = 1 KiB. Total ~2 KiB.
// =============================================================================
extern "C" __global__ void aegis_nvfp4_linear_prequantized_batched_gemm_wmma_bf16(
    const unsigned char* __restrict__ packed,
    const unsigned char* __restrict__ scales,
    const float*         __restrict__ input,
    const unsigned int rows,
    const unsigned int cols,
    const unsigned int total_batch,
    const float output_scale,
    float*               __restrict__ output
) {
    using namespace nvcuda;
    constexpr int M = 16, N = 16, K = 16;

    const unsigned int tile_row = blockIdx.x * (unsigned int)M;
    const unsigned int tile_col = blockIdx.y * (unsigned int)N;
    if (tile_row >= rows) return;

    const unsigned int tid = threadIdx.x;  // 0..31

    __shared__ __nv_bfloat16 sh_a[M * K];   // weight tile [M rows, K cols] row-major
    __shared__ __nv_bfloat16 sh_b[K * N];   // input  tile [K cols, N batch] col-major
    __shared__ float          sh_c[M * N];  // output tile [M rows, N batch] row-major

    wmma::fragment<wmma::matrix_a, M, N, K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    const unsigned int packed_cols = cols / 2u;
    const unsigned int scale_cols  = cols / 16u;

    // K-loop: iterate cols / K tiles.
    for (unsigned int k_tile = 0u; k_tile < cols; k_tile += (unsigned int)K) {
        // Load 16x16 weight tile. 256 elements / 32 threads = 8 elements per
        // thread. Each element decodes one NVFP4 nibble × per-block scale.
        for (unsigned int e = tid; e < (unsigned int)(M * K); e += 32u) {
            const unsigned int m = e / (unsigned int)K;
            const unsigned int k = e % (unsigned int)K;
            const unsigned int row_g = tile_row + m;
            const unsigned int col_g = k_tile + k;
            float v = 0.0f;
            if (row_g < rows && col_g < cols) {
                const size_t packed_idx = (size_t)row_g * packed_cols + (size_t)(col_g / 2u);
                const unsigned int byte = packed[packed_idx];
                const unsigned int nibble = (col_g & 1u) ? (byte >> 4u) : (byte & 0x0Fu);
                const size_t scale_idx = (size_t)row_g * scale_cols + (size_t)(col_g / 16u);
                const float blk_scale = decode_ue4m3_half(scales[scale_idx]);
                v = (float)decode_nvfp4_nibble(nibble) * blk_scale;
            }
            sh_a[m * (unsigned int)K + k] = __float2bfloat16(v);
        }

        // Load 16x16 input tile, col-major: sh_b[n*K + k] = input[batch=n, col=k].
        for (unsigned int e = tid; e < (unsigned int)(K * N); e += 32u) {
            const unsigned int n = e / (unsigned int)K;
            const unsigned int k = e % (unsigned int)K;
            const unsigned int batch_g = tile_col + n;
            const unsigned int col_g   = k_tile + k;
            float v = 0.0f;
            if (batch_g < total_batch && col_g < cols) {
                v = input[(size_t)batch_g * cols + col_g];
            }
            sh_b[n * (unsigned int)K + k] = __float2bfloat16(v);
        }
        __syncthreads();

        wmma::load_matrix_sync(a_frag, sh_a, K);
        wmma::load_matrix_sync(b_frag, sh_b, K);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    // Store 16x16 output tile to shared, then strided write to global with
    // boundary masking. Result layout: output[batch, row].
    wmma::store_matrix_sync(sh_c, c_frag, N, wmma::mem_row_major);
    for (unsigned int e = tid; e < (unsigned int)(M * N); e += 32u) {
        const unsigned int m = e / (unsigned int)N;
        const unsigned int n = e % (unsigned int)N;
        const unsigned int row_g   = tile_row + m;
        const unsigned int batch_g = tile_col + n;
        if (row_g < rows && batch_g < total_batch) {
            output[(size_t)batch_g * rows + row_g] = sh_c[m * (unsigned int)N + n] * output_scale;
        }
    }
}

// =============================================================================
// Grouped MoE NVFP4 GEMM with BF16 WMMA inner (Phase B.2).
//
// Single-launch kernel that processes ALL active experts of one projection
// (gate / up / down) in a single grid. Replaces the per-expert host loop +
// per-expert kernel call pattern (30 expert × 3 launches = 90 launches per
// layer with the per-expert path) with one launch per projection.
//
// Inputs:
//   * `packed_base`         — base ptr to a contiguous VRAM buffer holding
//                             ALL active experts' packed bytes.
//   * `scales_base`         — same for the per-block scales.
//   * `packed_offsets[ae]`  — byte offset into `packed_base` for the
//                             ae-th active expert's weights.
//   * `scales_offsets[ae]`  — same for scales.
//   * `output_scales[ae]`   — per-expert FP32 output scale (was the
//                             `output_scale` field on DeviceNvfp4Linear).
//   * `expert_token_offsets[ae+1]`  — prefix sum over active experts:
//                             tokens for expert ae live at
//                             `permuted_input[token_offsets[ae] : token_offsets[ae+1]]`.
//   * `permuted_input`      — `[total_tokens, cols]` activations sorted by
//                             expert (built by `aegis_permute_gather_f32`).
//   * `permuted_output`     — `[total_tokens, rows]` per-expert outputs
//                             sorted in same order; later un-permuted by
//                             `aegis_unpermute_scatter_add_f32`.
//
// Grid : (ceil(rows/16), max_tokens_per_expert/16, num_active_experts)
// Block: 32 threads (one warp) — same WMMA tile design as the per-expert
//        WMMA kernel above.
//
// Portability: same SM 7.5+ requirement as the per-expert WMMA kernel.
// =============================================================================
extern "C" __global__ void aegis_nvfp4_grouped_prequant_gemm_wmma_bf16(
    const unsigned char* __restrict__ packed_base,
    const unsigned char* __restrict__ scales_base,
    const unsigned int*  __restrict__ packed_offsets,        // [num_active_experts]
    const unsigned int*  __restrict__ scales_offsets,        // [num_active_experts]
    const float*         __restrict__ output_scales,          // [num_active_experts]
    const unsigned int*  __restrict__ expert_token_offsets,   // [num_active_experts + 1]
    const float*         __restrict__ permuted_input,         // [total_tokens, cols]
    const unsigned int rows,
    const unsigned int cols,
    float*               __restrict__ permuted_output         // [total_tokens, rows]
) {
    using namespace nvcuda;
    constexpr int M = 16, N = 16, K = 16;

    const unsigned int active_e = blockIdx.z;
    const unsigned int tok_start = expert_token_offsets[active_e];
    const unsigned int tok_end   = expert_token_offsets[active_e + 1];
    const unsigned int tok_count = tok_end - tok_start;

    const unsigned int tile_row = blockIdx.x * (unsigned int)M;
    const unsigned int tile_col_in_expert = blockIdx.y * (unsigned int)N;

    if (tile_row >= rows) return;
    if (tile_col_in_expert >= tok_count) return;  // expert has no tokens for this Y-tile

    const unsigned int tid = threadIdx.x;

    const unsigned char* packed = packed_base + (size_t)packed_offsets[active_e];
    const unsigned char* scales = scales_base + (size_t)scales_offsets[active_e];
    const float output_scale = output_scales[active_e];

    __shared__ __nv_bfloat16 sh_a[M * K];
    __shared__ __nv_bfloat16 sh_b[K * N];
    __shared__ float          sh_c[M * N];

    wmma::fragment<wmma::matrix_a, M, N, K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, M, N, K, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    const unsigned int packed_cols = cols / 2u;
    const unsigned int scale_cols  = cols / 16u;

    for (unsigned int k_tile = 0u; k_tile < cols; k_tile += (unsigned int)K) {
        // Weight tile: same as per-expert WMMA kernel, but using per-expert
        // packed/scales pointers.
        for (unsigned int e = tid; e < (unsigned int)(M * K); e += 32u) {
            const unsigned int m = e / (unsigned int)K;
            const unsigned int k = e % (unsigned int)K;
            const unsigned int row_g = tile_row + m;
            const unsigned int col_g = k_tile + k;
            float v = 0.0f;
            if (row_g < rows && col_g < cols) {
                const size_t packed_idx = (size_t)row_g * packed_cols + (size_t)(col_g / 2u);
                const unsigned int byte = packed[packed_idx];
                const unsigned int nibble = (col_g & 1u) ? (byte >> 4u) : (byte & 0x0Fu);
                const size_t scale_idx = (size_t)row_g * scale_cols + (size_t)(col_g / 16u);
                const float blk_scale = decode_ue4m3_half(scales[scale_idx]);
                v = (float)decode_nvfp4_nibble(nibble) * blk_scale;
            }
            sh_a[m * (unsigned int)K + k] = __float2bfloat16(v);
        }

        // Input tile: read from this expert's slice of permuted_input.
        // Global token index = tok_start + (tile_col_in_expert + n).
        for (unsigned int e = tid; e < (unsigned int)(K * N); e += 32u) {
            const unsigned int n = e / (unsigned int)K;
            const unsigned int k = e % (unsigned int)K;
            const unsigned int batch_in_e = tile_col_in_expert + n;
            const unsigned int col_g      = k_tile + k;
            float v = 0.0f;
            if (batch_in_e < tok_count && col_g < cols) {
                const unsigned int token_g = tok_start + batch_in_e;
                v = permuted_input[(size_t)token_g * cols + col_g];
            }
            sh_b[n * (unsigned int)K + k] = __float2bfloat16(v);
        }
        __syncthreads();

        wmma::load_matrix_sync(a_frag, sh_a, K);
        wmma::load_matrix_sync(b_frag, sh_b, K);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    wmma::store_matrix_sync(sh_c, c_frag, N, wmma::mem_row_major);
    for (unsigned int e = tid; e < (unsigned int)(M * N); e += 32u) {
        const unsigned int m = e / (unsigned int)N;
        const unsigned int n = e % (unsigned int)N;
        const unsigned int row_g     = tile_row + m;
        const unsigned int batch_in_e = tile_col_in_expert + n;
        if (row_g < rows && batch_in_e < tok_count) {
            const unsigned int token_g = tok_start + batch_in_e;
            permuted_output[(size_t)token_g * rows + row_g] =
                sh_c[m * (unsigned int)N + n] * output_scale;
        }
    }
}

// =============================================================================
// Grouped MoE NVFP4 GEMM with BF16 WMMA inner — 32x32 output tile (4 warps).
//
// Drop-in replacement for `aegis_nvfp4_grouped_prequant_gemm_wmma_bf16` that
// computes a 32-row × 32-col output region per block instead of 16×16. The
// block contains 4 warps (128 threads) and each warp owns one 16×16 sub-tile
// of the output. All four warps share the same K-slab loaded into shared
// memory once per K-iter, which (a) cuts redundant global loads of the
// dequantized weight tile by 2× across the M dimension, (b) cuts redundant
// loads of the input tile by 2× across the N dimension, and (c) issues 4×
// `mma.sync` per K-iter, raising tensor-core utilization vs. the 1-warp
// kernel.
//
// Bit-exactness vs. the 16×16 kernel: each warp's accumulation is a sum of
// the same per-K-tile WMMA products as the 16×16 kernel would compute for
// the same global (row, batch) pair. The K-iteration order, BF16 cast order,
// and `mma.sync` semantics are identical, so the per-element f32 accumulator
// is bit-identical. Only the global store pattern differs.
//
// Grid : (ceil(rows/32), ceil(max_tokens_per_active/32), num_active_experts)
// Block: 128 threads (4 warps)
// Shared: 32*16 BF16 sh_a (1 KiB) + 32*16 BF16 sh_b (1 KiB) + 32*32 f32 sh_c
//         (4 KiB) = ~6 KiB.
// =============================================================================
extern "C" __global__ void aegis_nvfp4_grouped_prequant_gemm_wmma_bf16_t32(
    const unsigned char* __restrict__ packed_base,
    const unsigned char* __restrict__ scales_base,
    const unsigned int*  __restrict__ packed_offsets,        // [num_active_experts]
    const unsigned int*  __restrict__ scales_offsets,        // [num_active_experts]
    const float*         __restrict__ output_scales,          // [num_active_experts]
    const unsigned int*  __restrict__ expert_token_offsets,   // [num_active_experts + 1]
    const float*         __restrict__ permuted_input,         // [total_tokens, cols]
    const unsigned int rows,
    const unsigned int cols,
    float*               __restrict__ permuted_output         // [total_tokens, rows]
) {
    using namespace nvcuda;
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
    constexpr int TILE_M = 32, TILE_N = 32;   // block output tile
    constexpr unsigned int WARPS_PER_BLOCK = 4u;  // 2x2 layout in (M,N)

    const unsigned int active_e = blockIdx.z;
    const unsigned int tok_start = expert_token_offsets[active_e];
    const unsigned int tok_end   = expert_token_offsets[active_e + 1];
    const unsigned int tok_count = tok_end - tok_start;

    const unsigned int tile_row = blockIdx.x * (unsigned int)TILE_M;
    const unsigned int tile_col_in_expert = blockIdx.y * (unsigned int)TILE_N;

    if (tile_row >= rows) return;
    if (tile_col_in_expert >= tok_count) return;

    const unsigned int tid    = threadIdx.x;       // 0..127
    const unsigned int warp_id = tid >> 5;          // 0..3
    // Lane index inside the warp is implicit in `wmma::*_sync` — the WMMA
    // ops handle per-lane operand routing internally. We only need warp_id
    // here to partition the 2x2 sub-tile grid across warps.
    const unsigned int warp_row = warp_id >> 1;     // 0..1 — owns rows [warp_row*16, +16)
    const unsigned int warp_col = warp_id & 1u;     // 0..1 — owns cols [warp_col*16, +16)

    const unsigned char* packed = packed_base + (size_t)packed_offsets[active_e];
    const unsigned char* scales = scales_base + (size_t)scales_offsets[active_e];
    const float output_scale = output_scales[active_e];

    __shared__ __nv_bfloat16 sh_a[TILE_M * WMMA_K];   // [TILE_M rows, WMMA_K cols] row-major
    __shared__ __nv_bfloat16 sh_b[TILE_N * WMMA_K];   // [TILE_N cols, WMMA_K] col-major: sh_b[n*K + k]
    __shared__ float          sh_c[TILE_M * TILE_N];  // [TILE_M rows, TILE_N cols] row-major

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    const unsigned int packed_cols = cols / 2u;
    const unsigned int scale_cols  = cols / 16u;

    constexpr unsigned int A_ELEMS = (unsigned)(TILE_M * WMMA_K);  // 512
    constexpr unsigned int B_ELEMS = (unsigned)(TILE_N * WMMA_K);  // 512
    constexpr unsigned int BLOCK_THREADS = WARPS_PER_BLOCK * 32u;  // 128

    for (unsigned int k_tile = 0u; k_tile < cols; k_tile += (unsigned int)WMMA_K) {
        // Load 32x16 weight tile (TILE_M rows × WMMA_K cols), row-major.
        // 512 elements / 128 threads = 4 elements per thread.
        for (unsigned int e = tid; e < A_ELEMS; e += BLOCK_THREADS) {
            const unsigned int m = e / (unsigned int)WMMA_K;
            const unsigned int k = e % (unsigned int)WMMA_K;
            const unsigned int row_g = tile_row + m;
            const unsigned int col_g = k_tile + k;
            float v = 0.0f;
            if (row_g < rows && col_g < cols) {
                const size_t packed_idx = (size_t)row_g * packed_cols + (size_t)(col_g / 2u);
                const unsigned int byte = packed[packed_idx];
                const unsigned int nibble = (col_g & 1u) ? (byte >> 4u) : (byte & 0x0Fu);
                const size_t scale_idx = (size_t)row_g * scale_cols + (size_t)(col_g / 16u);
                const float blk_scale = decode_ue4m3_half(scales[scale_idx]);
                v = (float)decode_nvfp4_nibble(nibble) * blk_scale;
            }
            sh_a[m * (unsigned int)WMMA_K + k] = __float2bfloat16(v);
        }

        // Load 32x16 input tile (TILE_N batch entries × WMMA_K cols), col-major:
        // sh_b[n*K + k] = input[batch=n, col=k].
        for (unsigned int e = tid; e < B_ELEMS; e += BLOCK_THREADS) {
            const unsigned int n = e / (unsigned int)WMMA_K;
            const unsigned int k = e % (unsigned int)WMMA_K;
            const unsigned int batch_in_e = tile_col_in_expert + n;
            const unsigned int col_g      = k_tile + k;
            float v = 0.0f;
            if (batch_in_e < tok_count && col_g < cols) {
                const unsigned int token_g = tok_start + batch_in_e;
                v = permuted_input[(size_t)token_g * cols + col_g];
            }
            sh_b[n * (unsigned int)WMMA_K + k] = __float2bfloat16(v);
        }
        __syncthreads();

        // Each warp loads its sub-tile of A/B and computes one 16x16 mma.
        const __nv_bfloat16* a_base = sh_a + (warp_row * (unsigned int)WMMA_M) * (unsigned int)WMMA_K;
        const __nv_bfloat16* b_base = sh_b + (warp_col * (unsigned int)WMMA_N) * (unsigned int)WMMA_K;
        wmma::load_matrix_sync(a_frag, a_base, WMMA_K);
        wmma::load_matrix_sync(b_frag, b_base, WMMA_K);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    // Store this warp's 16x16 sub-tile into sh_c at offset (warp_row*16, warp_col*16).
    float* c_base = sh_c
        + (warp_row * (unsigned int)WMMA_M) * (unsigned int)TILE_N
        + (warp_col * (unsigned int)WMMA_N);
    wmma::store_matrix_sync(c_base, c_frag, TILE_N, wmma::mem_row_major);
    __syncthreads();

    // Cooperative strided write of 32x32 fp32 staging buffer to global,
    // with output-scale and boundary masking. 1024 elements / 128 threads = 8/thread.
    constexpr unsigned int C_ELEMS = (unsigned)(TILE_M * TILE_N);  // 1024
    for (unsigned int e = tid; e < C_ELEMS; e += BLOCK_THREADS) {
        const unsigned int m = e / (unsigned int)TILE_N;
        const unsigned int n = e % (unsigned int)TILE_N;
        const unsigned int row_g     = tile_row + m;
        const unsigned int batch_in_e = tile_col_in_expert + n;
        if (row_g < rows && batch_in_e < tok_count) {
            const unsigned int token_g = tok_start + batch_in_e;
            permuted_output[(size_t)token_g * rows + row_g] =
                sh_c[m * (unsigned int)TILE_N + n] * output_scale;
        }
    }
}

// =============================================================================
// Grouped MoE NVFP4 GEMM with BF16 WMMA inner — 64×64 output tile (8 warps).
// Phase B.4 Round 4.
//
// Larger-tile sibling of `aegis_nvfp4_grouped_prequant_gemm_wmma_bf16_t32`.
// Computes a 64-row × 64-col output region per block. Block has 8 warps
// (256 threads) arranged in a 4 (row) × 2 (col) grid; each warp owns a
// 16-row × 32-col strip of the output tile = 2 horizontally-adjacent 16×16
// WMMA sub-tiles → 8 × 2 = 16 c_frags total, exactly covering 64×64 / 256 = 16
// 16×16 sub-tiles.
//
// Per K-iter, each warp loads ONE 16×16 a_frag (its row strip of A) and TWO
// 16×16 b_frags (its two col sub-tiles), then issues two mma.sync ops sharing
// the a_frag. This raises arithmetic per shared load relative to t32:
//   * t32: 1 a_frag + 1 b_frag → 1 mma per warp; A shared loads / mma = 1
//   * t32_big: 1 a_frag + 2 b_frags → 2 mmas per warp; A shared loads / mma = 0.5
//
// Bit-exactness vs. the t32 (and 16×16) kernel:
//   * Each output element (row_g, batch_g) is owned by exactly one warp.
//   * That warp issues `mma.sync(c, a_warp, b_warp, c)` once per K-iter where
//     `a_warp` is the same 16×16 of A[row_g..row_g+16, k_tile..k_tile+16] that
//     the t32 kernel uses for the same output element, and `b_warp` is the
//     same 16×16 of B[batch_g..batch_g+16, k_tile..k_tile+16]. Per-element f32
//     accumulator order is identical because mma.sync semantics are
//     element-wise sums; the only thing that differs is which warp/block
//     computes which output element. Cooperative shmem loads dequant exactly
//     the same NVFP4 nibble × ue4m3 scale and apply the same
//     __float2bfloat16 cast per element.
//
// Eligibility: rows%64==0 AND max_tokens_per_expert>=64. Outside that, the
// dispatcher falls back to the t32 (32×32) path. For Gemma-4-26B-A4B routed
// experts the per-expert weight rows are multiples of 64 (704 = 11×64,
// 2048 = 32×64); the M-tile boundary check is still kept inside the load
// loops in case future configs land odd row counts.
//
// Grid : (ceil(rows/64), ceil(max_tokens_per_active/64), num_active_experts)
// Block: 256 threads (8 warps) in 4 (row-warps) × 2 (col-warps) layout.
// Shared: 64×16 BF16 sh_a (2 KiB) + 64×16 BF16 sh_b (2 KiB) +
//         64×64 f32 sh_c (16 KiB) = ~20 KiB. Well under sm_120's 100 KiB cap.
// Registers per warp: 2 c_frags (8 f32 each = 16 regs) + 1 a_frag (8 bf16
//   packed = 4 regs) + 2 b_frags (8 bf16 packed = 4 regs each) ≈ 28 acc regs.
//   Plus loop-local f32/i32 ≈ 70-90 regs/thread total (verify with ptxas -v).
// =============================================================================
extern "C" __global__ void aegis_nvfp4_grouped_prequant_gemm_wmma_bf16_t32_big(
    const unsigned char* __restrict__ packed_base,
    const unsigned char* __restrict__ scales_base,
    const unsigned int*  __restrict__ packed_offsets,        // [num_active_experts]
    const unsigned int*  __restrict__ scales_offsets,        // [num_active_experts]
    const float*         __restrict__ output_scales,          // [num_active_experts]
    const unsigned int*  __restrict__ expert_token_offsets,   // [num_active_experts + 1]
    const float*         __restrict__ permuted_input,         // [total_tokens, cols]
    const unsigned int rows,
    const unsigned int cols,
    float*               __restrict__ permuted_output         // [total_tokens, rows]
) {
    using namespace nvcuda;
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
    constexpr int TILE_M = 64, TILE_N = 64;          // block output tile
    constexpr unsigned int WARPS_M = 4u;             // row-warps
    constexpr unsigned int WARPS_N = 2u;             // col-warps
    constexpr unsigned int WARPS_PER_BLOCK = WARPS_M * WARPS_N;  // 8
    constexpr unsigned int N_FRAGS_PER_WARP = 2u;    // each warp owns 2 col sub-tiles
    constexpr unsigned int BLOCK_THREADS = WARPS_PER_BLOCK * 32u;  // 256

    const unsigned int active_e = blockIdx.z;
    const unsigned int tok_start = expert_token_offsets[active_e];
    const unsigned int tok_end   = expert_token_offsets[active_e + 1];
    const unsigned int tok_count = tok_end - tok_start;

    const unsigned int tile_row = blockIdx.x * (unsigned int)TILE_M;
    const unsigned int tile_col_in_expert = blockIdx.y * (unsigned int)TILE_N;

    if (tile_row >= rows) return;
    if (tile_col_in_expert >= tok_count) return;

    const unsigned int tid     = threadIdx.x;        // 0..255
    const unsigned int warp_id = tid >> 5;            // 0..7
    // 4×2 warp grid in (M, N): warp_row ∈ [0..4), warp_col ∈ [0..2).
    const unsigned int warp_row = warp_id >> 1;       // 0..3 — owns rows [warp_row*16, +16)
    const unsigned int warp_col = warp_id & 1u;       // 0..1 — owns cols [warp_col*32, +32) = two 16-col sub-tiles

    const unsigned char* packed = packed_base + (size_t)packed_offsets[active_e];
    const unsigned char* scales = scales_base + (size_t)scales_offsets[active_e];
    const float output_scale = output_scales[active_e];

    __shared__ __nv_bfloat16 sh_a[TILE_M * WMMA_K];   // [TILE_M rows, WMMA_K cols] row-major
    __shared__ __nv_bfloat16 sh_b[TILE_N * WMMA_K];   // [TILE_N cols, WMMA_K] col-major: sh_b[n*K + k]
    __shared__ float          sh_c[TILE_M * TILE_N];  // [TILE_M rows, TILE_N cols] row-major

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag[N_FRAGS_PER_WARP];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[N_FRAGS_PER_WARP];
    #pragma unroll
    for (unsigned int j = 0u; j < N_FRAGS_PER_WARP; ++j) {
        wmma::fill_fragment(c_frag[j], 0.0f);
    }

    const unsigned int packed_cols = cols / 2u;
    const unsigned int scale_cols  = cols / 16u;

    constexpr unsigned int A_ELEMS = (unsigned)(TILE_M * WMMA_K);  // 1024
    constexpr unsigned int B_ELEMS = (unsigned)(TILE_N * WMMA_K);  // 1024

    for (unsigned int k_tile = 0u; k_tile < cols; k_tile += (unsigned int)WMMA_K) {
        // Load 64×16 weight tile (TILE_M rows × WMMA_K cols), row-major.
        // Scale-hoisted layout: thread covers 4 contiguous K positions in one
        // row (instead of 4 random rows), so:
        //   * 1 scale fetch+decode per thread (shared across all 4 elements)
        //   * 2 contiguous packed-byte loads per thread → 4 nibbles
        // Per-element decode order is byte/nibble/scale/f32-mul/bf16 cast —
        // bit-identical to the original (row_g, col_g) layout, just reordered
        // spatially. 256 threads × 4 elems = 1024 elems = full 64×16 tile.
        {
            const unsigned int m_local = tid >> 2;          // 0..63
            const unsigned int k_chunk = tid & 3u;          // 0..3
            const unsigned int k_start = k_chunk * 4u;      // 0/4/8/12
            const unsigned int row_g = tile_row + m_local;
            float blk_scale = 0.0f;
            unsigned int byte0 = 0u, byte1 = 0u;
            if (row_g < rows) {
                const size_t scale_idx =
                    (size_t)row_g * scale_cols + (size_t)(k_tile / 16u);
                blk_scale = decode_ue4m3_half(scales[scale_idx]);
                const size_t base_packed_idx =
                    (size_t)row_g * packed_cols
                    + (size_t)((k_tile + k_start) / 2u);
                byte0 = packed[base_packed_idx];
                byte1 = packed[base_packed_idx + 1u];
            }
            #pragma unroll
            for (unsigned int kk = 0u; kk < 4u; ++kk) {
                const unsigned int k = k_start + kk;
                const unsigned int byte = (kk < 2u) ? byte0 : byte1;
                const unsigned int nibble =
                    ((kk & 1u) == 1u) ? (byte >> 4u) : (byte & 0x0Fu);
                float v = 0.0f;
                if (row_g < rows) {
                    v = (float)decode_nvfp4_nibble(nibble) * blk_scale;
                }
                sh_a[m_local * (unsigned int)WMMA_K + k] = __float2bfloat16(v);
            }
        }

        // Load 64×16 input tile (TILE_N batch entries × WMMA_K cols), col-major:
        // sh_b[n*K + k] = input[batch=n, col=k]. Same per-element f32→bf16
        // conversion as the t32 kernel.
        for (unsigned int e = tid; e < B_ELEMS; e += BLOCK_THREADS) {
            const unsigned int n = e / (unsigned int)WMMA_K;
            const unsigned int k = e % (unsigned int)WMMA_K;
            const unsigned int batch_in_e = tile_col_in_expert + n;
            const unsigned int col_g      = k_tile + k;
            float v = 0.0f;
            if (batch_in_e < tok_count && col_g < cols) {
                const unsigned int token_g = tok_start + batch_in_e;
                v = permuted_input[(size_t)token_g * cols + col_g];
            }
            sh_b[n * (unsigned int)WMMA_K + k] = __float2bfloat16(v);
        }
        __syncthreads();

        // Each warp loads its 1 a_frag (row strip) and 2 b_frags (col sub-tiles),
        // then issues 2 mma.sync ops. The a_frag is reused across both mmas.
        const __nv_bfloat16* a_base = sh_a + (warp_row * (unsigned int)WMMA_M) * (unsigned int)WMMA_K;
        wmma::load_matrix_sync(a_frag, a_base, WMMA_K);
        #pragma unroll
        for (unsigned int j = 0u; j < N_FRAGS_PER_WARP; ++j) {
            // Sub-tile column index inside the block tile:
            //   warp_col=0 → sub-tile cols 0..16, 16..32   (j=0,1)
            //   warp_col=1 → sub-tile cols 32..48, 48..64  (j=0,1)
            const unsigned int sub_col = warp_col * N_FRAGS_PER_WARP + j;  // 0..3
            const __nv_bfloat16* b_base = sh_b + (sub_col * (unsigned int)WMMA_N) * (unsigned int)WMMA_K;
            wmma::load_matrix_sync(b_frag[j], b_base, WMMA_K);
            wmma::mma_sync(c_frag[j], a_frag, b_frag[j], c_frag[j]);
        }
        __syncthreads();
    }

    // Store each warp's 2 c_frags into sh_c at their respective sub-tile positions.
    // Row strip: rows [warp_row*16, +16). Col strip: cols [warp_col*32, +32),
    // split into two 16-col sub-tiles selected by `j`.
    #pragma unroll
    for (unsigned int j = 0u; j < N_FRAGS_PER_WARP; ++j) {
        const unsigned int sub_col = warp_col * N_FRAGS_PER_WARP + j;
        float* c_base = sh_c
            + (warp_row * (unsigned int)WMMA_M) * (unsigned int)TILE_N
            + (sub_col  * (unsigned int)WMMA_N);
        wmma::store_matrix_sync(c_base, c_frag[j], TILE_N, wmma::mem_row_major);
    }
    __syncthreads();

    // Cooperative strided write of 64×64 fp32 staging buffer to global with
    // output-scale and boundary masking. 4096 elements / 256 threads = 16/thread.
    constexpr unsigned int C_ELEMS = (unsigned)(TILE_M * TILE_N);  // 4096
    for (unsigned int e = tid; e < C_ELEMS; e += BLOCK_THREADS) {
        const unsigned int m = e / (unsigned int)TILE_N;
        const unsigned int n = e % (unsigned int)TILE_N;
        const unsigned int row_g     = tile_row + m;
        const unsigned int batch_in_e = tile_col_in_expert + n;
        if (row_g < rows && batch_in_e < tok_count) {
            const unsigned int token_g = tok_start + batch_in_e;
            permuted_output[(size_t)token_g * rows + row_g] =
                sh_c[m * (unsigned int)TILE_N + n] * output_scale;
        }
    }
}

// =============================================================================
// Grouped MoE NVFP4 GEMM with BF16 WMMA inner — 64×64 tile + cp.async pipelined
// B — Phase B.4 Round 5.
//
// Numerical-twin of `aegis_nvfp4_grouped_prequant_gemm_wmma_bf16_t32_big`.
// Mirrors the cp.async pipelining strategy already used by
// `_t32_pipeline` (Round 3), but applied to the 64×64 / 8-warp output-tile
// kernel from Round 4. Pipelines ONLY the input tile (B); the NVFP4 weight
// dequant path is left synchronous (its dequant is the longer per-iter
// critical path; reordering it could perturb numerics).
//
// Per-K-iter:
//   * Synchronous A: dequant 64×16 NVFP4 weight tile → bf16 in `sh_a` (same
//     code as the synchronous BIG kernel, byte-for-byte).
//   * cp.async B[k+1] (if any) into the next double-buffer slot, commit, then
//     `cp.async.wait_group<1>` so B[k] is ready while B[k+1] is in flight.
//   * Cast f32 → bf16 from the staging double-buffer slot for buf_cur into
//     `sh_b` (per-element __float2bfloat16 — same op the synchronous BIG
//     kernel applies to its global-loaded f32, so the resulting bf16 is
//     bit-identical at the per-element level).
//   * 8-warp / 4×2 / 2-c_frag mma sequence (identical to the BIG kernel).
//
// Tile/launch shape:
//   * Grid : (ceil(rows/64), ceil(max_tokens_per_active/64), num_active_experts)
//   * Block: 256 threads (8 warps) — same as BIG.
//   * Eligibility: rows%64==0 AND max_tokens_per_expert>=64 (matches BIG).
//
// cp.async layout (B-tile is 64 rows × 16 cols × 4 bytes = 4096 bytes):
//   * 256 threads × 16-byte chunk per thread → 4096 bytes / iter — full
//     bandwidth utilization, no thread idle.
//   * n_load       = tid / 4   (0..63)
//   * k_chunk_load = tid % 4   (0..3)
//   * k_start_load = k_chunk * 4 → {0,4,8,12}
//   * Each thread issues exactly one cp.async.ca.shared.global of 16 bytes
//     per (k_tile, buffer) pair.
//
// Shared memory budget:
//   * sh_a (bf16):     64 * 16 * 2 = 2 KiB
//   * sh_b (bf16):     64 * 16 * 2 = 2 KiB
//   * sh_c (f32):     64 * 64 * 4 = 16 KiB
//   * sh_b_raw (f32): 2 * 64 * 16 * 4 = 8 KiB  (double-buffer)
//   * Total ≈ 28 KiB. Well under sm_120's 100 KiB cap. Static shared, so no
//     `cudaFuncSetAttribute` needed.
//
// Numerical correctness vs. synchronous BIG:
//   * cp.async copies raw f32 bytes from `permuted_input` to `sh_b_raw` —
//     identical f32 values to what the synchronous load reads.
//   * The cast to bf16 happens via the same per-element `__float2bfloat16`
//     op as the synchronous BIG kernel uses on its global-loaded f32, so
//     `sh_b` bytes match bit-for-bit.
//   * `mma.sync` operand pairing per K-iter is identical → c_frag is
//     bit-identical to the synchronous BIG kernel at every iter.
//
// Requires sm_80+ for cp.async (Blackwell sm_120 supported).
// =============================================================================
extern "C" __global__ void aegis_nvfp4_grouped_prequant_gemm_wmma_bf16_t32_big_pipeline(
    const unsigned char* __restrict__ packed_base,
    const unsigned char* __restrict__ scales_base,
    const unsigned int*  __restrict__ packed_offsets,        // [num_active_experts]
    const unsigned int*  __restrict__ scales_offsets,        // [num_active_experts]
    const float*         __restrict__ output_scales,          // [num_active_experts]
    const unsigned int*  __restrict__ expert_token_offsets,   // [num_active_experts + 1]
    const float*         __restrict__ permuted_input,         // [total_tokens, cols]
    const unsigned int rows,
    const unsigned int cols,
    float*               __restrict__ permuted_output         // [total_tokens, rows]
) {
    using namespace nvcuda;
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
    constexpr int TILE_M = 64, TILE_N = 64;          // block output tile
    constexpr unsigned int WARPS_M = 4u;             // row-warps
    constexpr unsigned int WARPS_N = 2u;             // col-warps
    constexpr unsigned int WARPS_PER_BLOCK = WARPS_M * WARPS_N;  // 8
    constexpr unsigned int N_FRAGS_PER_WARP = 2u;
    constexpr unsigned int BLOCK_THREADS = WARPS_PER_BLOCK * 32u;  // 256

    const unsigned int active_e = blockIdx.z;
    const unsigned int tok_start = expert_token_offsets[active_e];
    const unsigned int tok_end   = expert_token_offsets[active_e + 1];
    const unsigned int tok_count = tok_end - tok_start;

    const unsigned int tile_row = blockIdx.x * (unsigned int)TILE_M;
    const unsigned int tile_col_in_expert = blockIdx.y * (unsigned int)TILE_N;

    if (tile_row >= rows) return;
    if (tile_col_in_expert >= tok_count) return;

    const unsigned int tid     = threadIdx.x;        // 0..255
    const unsigned int warp_id = tid >> 5;            // 0..7
    const unsigned int warp_row = warp_id >> 1;       // 0..3
    const unsigned int warp_col = warp_id & 1u;       // 0..1

    const unsigned char* packed = packed_base + (size_t)packed_offsets[active_e];
    const unsigned char* scales = scales_base + (size_t)scales_offsets[active_e];
    const float output_scale = output_scales[active_e];

    __shared__ __nv_bfloat16 sh_a[TILE_M * WMMA_K];   // 2 KiB
    __shared__ __nv_bfloat16 sh_b[TILE_N * WMMA_K];   // 2 KiB
    __shared__ float          sh_c[TILE_M * TILE_N];  // 16 KiB
    // Double-buffered f32 staging for the input tile, populated via cp.async.
    __shared__ float          sh_b_raw[2][TILE_N * WMMA_K];  // 8 KiB

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag[N_FRAGS_PER_WARP];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[N_FRAGS_PER_WARP];
    #pragma unroll
    for (unsigned int j = 0u; j < N_FRAGS_PER_WARP; ++j) {
        wmma::fill_fragment(c_frag[j], 0.0f);
    }

    const unsigned int packed_cols = cols / 2u;
    const unsigned int scale_cols  = cols / 16u;

    constexpr unsigned int A_ELEMS = (unsigned)(TILE_M * WMMA_K);  // 1024
    constexpr unsigned int B_ELEMS = (unsigned)(TILE_N * WMMA_K);  // 1024

    // cp.async chunking: 256 threads × 16B = 4096B = full B-tile.
    //   chunk_id = tid (0..255)
    //   n_load   = tid / 4   (0..63)
    //   k_chunk  = tid % 4   (0..3) → k_start = k_chunk * 4 ∈ {0,4,8,12}
    const unsigned int n_load       = tid >> 2;
    const unsigned int k_chunk_load = tid & 3u;
    const unsigned int k_start_load = k_chunk_load * 4u;

#if __CUDA_ARCH__ >= 800
    auto issue_b_prefetch = [&] (unsigned int k_tile_arg, unsigned int buf_idx) {
        const unsigned int batch_in_e = tile_col_in_expert + n_load;
        float* dst = &sh_b_raw[buf_idx][n_load * (unsigned int)WMMA_K + k_start_load];
        unsigned int dst_smem;
        asm volatile("{ .reg .u64 smem64;\n\t"
                     "  cvta.to.shared.u64 smem64, %1;\n\t"
                     "  cvt.u32.u64 %0, smem64; }\n"
                     : "=r"(dst_smem) : "l"((const void*)dst));
        if (batch_in_e < tok_count) {
            const unsigned int token_g = tok_start + batch_in_e;
            const unsigned int col_g   = k_tile_arg + k_start_load;
            const float* src = permuted_input + (size_t)token_g * cols + col_g;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                         :: "r"(dst_smem), "l"(src));
        } else {
            // OOB row: src_size=0 → zero-fills shared without dereferencing.
            const float* src = permuted_input;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, 0;\n"
                         :: "r"(dst_smem), "l"(src));
        }
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
#endif

    const unsigned int N_KITERS = cols / (unsigned int)WMMA_K;

#if __CUDA_ARCH__ >= 800
    // Prologue: issue B[0] prefetch into buffer 0, commit as group 0.
    if (N_KITERS > 0u) {
        issue_b_prefetch(0u, 0u);
        cp_async_commit();
    }
#endif

    for (unsigned int k_idx = 0u; k_idx < N_KITERS; ++k_idx) {
        const unsigned int k_tile  = k_idx * (unsigned int)WMMA_K;
        const unsigned int buf_cur = k_idx & 1u;
        const unsigned int buf_nxt = buf_cur ^ 1u;

        // ------ A weight tile (synchronous; scale-hoisted layout). ------
        // Thread covers 4 contiguous K positions in one row → 1 scale
        // decode + 2 contiguous packed-byte loads per thread per K-iter.
        // Bit-identical to the per-element original — same byte/nibble/
        // scale/f32-mul/bf16 cast chain, just reordered spatially.
        {
            const unsigned int m_local = tid >> 2;          // 0..63
            const unsigned int k_chunk = tid & 3u;          // 0..3
            const unsigned int k_start = k_chunk * 4u;      // 0/4/8/12
            const unsigned int row_g = tile_row + m_local;
            float blk_scale = 0.0f;
            unsigned int byte0 = 0u, byte1 = 0u;
            if (row_g < rows) {
                const size_t scale_idx =
                    (size_t)row_g * scale_cols + (size_t)(k_tile / 16u);
                blk_scale = decode_ue4m3_half(scales[scale_idx]);
                const size_t base_packed_idx =
                    (size_t)row_g * packed_cols
                    + (size_t)((k_tile + k_start) / 2u);
                byte0 = packed[base_packed_idx];
                byte1 = packed[base_packed_idx + 1u];
            }
            #pragma unroll
            for (unsigned int kk = 0u; kk < 4u; ++kk) {
                const unsigned int k = k_start + kk;
                const unsigned int byte = (kk < 2u) ? byte0 : byte1;
                const unsigned int nibble =
                    ((kk & 1u) == 1u) ? (byte >> 4u) : (byte & 0x0Fu);
                float v = 0.0f;
                if (row_g < rows) {
                    v = (float)decode_nvfp4_nibble(nibble) * blk_scale;
                }
                sh_a[m_local * (unsigned int)WMMA_K + k] = __float2bfloat16(v);
            }
        }

#if __CUDA_ARCH__ >= 800
        // Issue B[k+1] prefetch into the next buffer (if it exists), then
        // commit. After this we have either 1 or 2 outstanding cp.async groups.
        if (k_idx + 1u < N_KITERS) {
            issue_b_prefetch((k_idx + 1u) * (unsigned int)WMMA_K, buf_nxt);
            cp_async_commit();
            // 2 outstanding groups; wait until ≤1 remain → B[k] is ready.
            cp_async_wait_lt1();
        } else {
            // 1 outstanding group (B[k] only); wait until 0 remain.
            cp_async_wait_lt0();
        }
        __syncthreads();

        // Cast f32 → bf16 from the staging buffer into the bf16 sh_b that
        // wmma::load_matrix_sync reads. Per-element __float2bfloat16 — same
        // operation as the BIG kernel uses on its global-loaded f32, so the
        // resulting bf16 is bit-identical.
        // 1024 elems / 256 threads = 4 elems/thread.
        for (unsigned int e = tid; e < B_ELEMS; e += BLOCK_THREADS) {
            sh_b[e] = __float2bfloat16(sh_b_raw[buf_cur][e]);
        }
        __syncthreads();
#else
        // Fallback for pre-sm_80: synchronous load of B (matches BIG exactly).
        for (unsigned int e = tid; e < B_ELEMS; e += BLOCK_THREADS) {
            const unsigned int n = e / (unsigned int)WMMA_K;
            const unsigned int k = e % (unsigned int)WMMA_K;
            const unsigned int batch_in_e = tile_col_in_expert + n;
            const unsigned int col_g      = k_tile + k;
            float v = 0.0f;
            if (batch_in_e < tok_count && col_g < cols) {
                const unsigned int token_g = tok_start + batch_in_e;
                v = permuted_input[(size_t)token_g * cols + col_g];
            }
            sh_b[n * (unsigned int)WMMA_K + k] = __float2bfloat16(v);
        }
        __syncthreads();
#endif

        // ------ WMMA mma_sync (identical to BIG kernel). ------
        const __nv_bfloat16* a_base = sh_a + (warp_row * (unsigned int)WMMA_M) * (unsigned int)WMMA_K;
        wmma::load_matrix_sync(a_frag, a_base, WMMA_K);
        #pragma unroll
        for (unsigned int j = 0u; j < N_FRAGS_PER_WARP; ++j) {
            const unsigned int sub_col = warp_col * N_FRAGS_PER_WARP + j;
            const __nv_bfloat16* b_base = sh_b + (sub_col * (unsigned int)WMMA_N) * (unsigned int)WMMA_K;
            wmma::load_matrix_sync(b_frag[j], b_base, WMMA_K);
            wmma::mma_sync(c_frag[j], a_frag, b_frag[j], c_frag[j]);
        }
        __syncthreads();
    }

    // Store each warp's 2 c_frags into sh_c at their respective sub-tile positions.
    #pragma unroll
    for (unsigned int j = 0u; j < N_FRAGS_PER_WARP; ++j) {
        const unsigned int sub_col = warp_col * N_FRAGS_PER_WARP + j;
        float* c_base = sh_c
            + (warp_row * (unsigned int)WMMA_M) * (unsigned int)TILE_N
            + (sub_col  * (unsigned int)WMMA_N);
        wmma::store_matrix_sync(c_base, c_frag[j], TILE_N, wmma::mem_row_major);
    }
    __syncthreads();

    // Cooperative strided write of 64×64 fp32 staging buffer to global with
    // output-scale and boundary masking. 4096 elems / 256 threads = 16/thread.
    constexpr unsigned int C_ELEMS = (unsigned)(TILE_M * TILE_N);
    for (unsigned int e = tid; e < C_ELEMS; e += BLOCK_THREADS) {
        const unsigned int m = e / (unsigned int)TILE_N;
        const unsigned int n = e % (unsigned int)TILE_N;
        const unsigned int row_g     = tile_row + m;
        const unsigned int batch_in_e = tile_col_in_expert + n;
        if (row_g < rows && batch_in_e < tok_count) {
            const unsigned int token_g = tok_start + batch_in_e;
            permuted_output[(size_t)token_g * rows + row_g] =
                sh_c[m * (unsigned int)TILE_N + n] * output_scale;
        }
    }
}

// =============================================================================
// Grouped MoE NVFP4 GEMM with BF16 WMMA inner — t32 + cp.async pipelined B —
// Phase B.4 Round 3.
//
// Drop-in numerical-twin of `aegis_nvfp4_grouped_prequant_gemm_wmma_bf16_t32`.
// The only difference is HOW the input tile (B) is staged: instead of reading
// f32 from global → casting to bf16 → writing bf16 to shared synchronously,
// we issue the GLOBAL→SHARED bytes copy via `cp.async.ca.shared.global`
// against a double-buffered f32 staging slot in shared memory, then cast to
// bf16 only when the data is needed by `wmma::load_matrix_sync`. This
// overlaps the per-K-iter B-load latency with the synchronous A weight
// dequant.
//
// Pipelining strategy (option-b from the round-3 design doc):
//   * Pipeline ONLY the input tile (B). Weight A still goes through the same
//     synchronous dequant→cast→shared pattern.
//   * Two f32 staging buffers `sh_b_raw[2][TILE_N*WMMA_K]` (2 KiB each, 4 KiB
//     extra shared total).
//   * Software pipeline: at the start of K-iter `k`, the cp.async for B[k]
//     was issued at the end of K-iter `k-1` (or in a prologue for k=0).
//     We commit a fresh group for B[k+1] right after issuing A[k]'s dequant,
//     then `cp.async.wait_group<1>` so B[k]'s data is ready for our cast.
//   * After cast f32→bf16 into `sh_b`, the wmma path is bit-identical to t32.
//
// Numerical correctness vs. t32:
//   * The per-element f32 value loaded for B is the SAME f32 value: cp.async
//     copies raw f32 bytes; we cast via the same `__float2bfloat16` per
//     element. No precision loss vs. the synchronous path.
//   * A weight tile path is unchanged.
//   * `mma.sync` operands per K-iter are identical → `c_frag` accumulation
//     is bit-identical to the t32 kernel.
//
// Grid : (ceil(rows/32), ceil(max_tokens_per_active/32), num_active_experts)
// Block: 128 threads (4 warps), same partitioning as the standalone t32 kernel.
// Shared: 1 KiB sh_a + 1 KiB sh_b (bf16) + 4 KiB sh_c + 4 KiB sh_b_raw[2]
//         = ~10 KiB. Well under Blackwell's 100 KiB.
//
// Requires sm_80+ for cp.async (Blackwell sm_120 supported).
// =============================================================================
extern "C" __global__ void aegis_nvfp4_grouped_prequant_gemm_wmma_bf16_t32_pipeline(
    const unsigned char* __restrict__ packed_base,
    const unsigned char* __restrict__ scales_base,
    const unsigned int*  __restrict__ packed_offsets,        // [num_active_experts]
    const unsigned int*  __restrict__ scales_offsets,        // [num_active_experts]
    const float*         __restrict__ output_scales,          // [num_active_experts]
    const unsigned int*  __restrict__ expert_token_offsets,   // [num_active_experts + 1]
    const float*         __restrict__ permuted_input,         // [total_tokens, cols]
    const unsigned int rows,
    const unsigned int cols,
    float*               __restrict__ permuted_output         // [total_tokens, rows]
) {
    using namespace nvcuda;
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
    constexpr int TILE_M = 32, TILE_N = 32;   // block output tile
    constexpr unsigned int WARPS_PER_BLOCK = 4u;

    const unsigned int active_e = blockIdx.z;
    const unsigned int tok_start = expert_token_offsets[active_e];
    const unsigned int tok_end   = expert_token_offsets[active_e + 1];
    const unsigned int tok_count = tok_end - tok_start;

    const unsigned int tile_row = blockIdx.x * (unsigned int)TILE_M;
    const unsigned int tile_col_in_expert = blockIdx.y * (unsigned int)TILE_N;

    if (tile_row >= rows) return;
    if (tile_col_in_expert >= tok_count) return;

    const unsigned int tid     = threadIdx.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int warp_row = warp_id >> 1;
    const unsigned int warp_col = warp_id & 1u;

    const unsigned char* packed = packed_base + (size_t)packed_offsets[active_e];
    const unsigned char* scales = scales_base + (size_t)scales_offsets[active_e];
    const float output_scale = output_scales[active_e];

    __shared__ __nv_bfloat16 sh_a[TILE_M * WMMA_K];
    __shared__ __nv_bfloat16 sh_b[TILE_N * WMMA_K];
    __shared__ float          sh_c[TILE_M * TILE_N];
    // Double-buffered f32 staging for the input tile, populated via cp.async.
    __shared__ float          sh_b_raw[2][TILE_N * WMMA_K];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    const unsigned int packed_cols = cols / 2u;
    const unsigned int scale_cols  = cols / 16u;

    constexpr unsigned int A_ELEMS = (unsigned)(TILE_M * WMMA_K);    // 512
    constexpr unsigned int B_ELEMS = (unsigned)(TILE_N * WMMA_K);    // 512
    constexpr unsigned int BLOCK_THREADS = WARPS_PER_BLOCK * 32u;    // 128

    // Each B tile is 32 rows × 16 floats = 512 f32 = 2048 bytes.
    // We map each thread to one 16-byte (4-float) chunk: 128 threads × 16B = 2048B.
    //   chunk_id = tid (0..127)
    //   row n   = tid / 4   (0..31)
    //   k_chunk = tid % 4   (0..3) → k_start = k_chunk * 4
    const unsigned int n_load       = tid >> 2;            // 0..31
    const unsigned int k_chunk_load = tid & 3u;            // 0..3
    const unsigned int k_start_load = k_chunk_load * 4u;   // 0,4,8,12

#if __CUDA_ARCH__ >= 800
    // Issue an asynchronous 16-byte copy from `permuted_input` (global) into
    // `sh_b_raw[buf_idx]`, with bounds check on the row dimension. The K
    // dimension is always in-bounds because cols % 16 == 0 and we copy 4
    // contiguous floats starting at k_start_load ∈ {0,4,8,12}.
    auto issue_b_prefetch = [&] (unsigned int k_tile_arg, unsigned int buf_idx) {
        const unsigned int batch_in_e = tile_col_in_expert + n_load;
        float* dst = &sh_b_raw[buf_idx][n_load * (unsigned int)WMMA_K + k_start_load];
        // Convert the generic shared-memory pointer to a 32-bit SMEM address
        // for the cp.async PTX instruction. Done in inline PTX rather than
        // via the `__cvta_generic_to_shared` intrinsic to avoid any NVRTC
        // intrinsic-availability surprise.
        unsigned int dst_smem;
        asm volatile("{ .reg .u64 smem64;\n\t"
                     "  cvta.to.shared.u64 smem64, %1;\n\t"
                     "  cvt.u32.u64 %0, smem64; }\n"
                     : "=r"(dst_smem) : "l"((const void*)dst));
        if (batch_in_e < tok_count) {
            const unsigned int token_g = tok_start + batch_in_e;
            const unsigned int col_g   = k_tile_arg + k_start_load;
            const float* src = permuted_input + (size_t)token_g * cols + col_g;
            // 16-byte cp.async with full src_size (in-bounds row).
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                         :: "r"(dst_smem), "l"(src));
        } else {
            // Out-of-bounds row: cp.async with src_size=0 zero-fills the
            // 16-byte destination region without dereferencing the source.
            // Still issues a group member so the cp.async commit/wait
            // bookkeeping stays consistent across threads.
            const float* src = permuted_input;  // any valid pointer; src_size=0 means no read
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16, 0;\n"
                         :: "r"(dst_smem), "l"(src));
        }
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
#endif

    // Number of K-iters; cols is a multiple of 16 by precondition.
    const unsigned int N_KITERS = cols / (unsigned int)WMMA_K;

#if __CUDA_ARCH__ >= 800
    // Prologue: issue B[0] prefetch into buffer 0, commit as group 0.
    if (N_KITERS > 0u) {
        issue_b_prefetch(0u, 0u);
        cp_async_commit();
    }
#endif

    for (unsigned int k_idx = 0u; k_idx < N_KITERS; ++k_idx) {
        const unsigned int k_tile  = k_idx * (unsigned int)WMMA_K;
        const unsigned int buf_cur = k_idx & 1u;
        const unsigned int buf_nxt = buf_cur ^ 1u;

        // ------ A weight tile (synchronous, identical to t32 kernel). ------
        // 512 elements / 128 threads = 4 elements per thread.
        for (unsigned int e = tid; e < A_ELEMS; e += BLOCK_THREADS) {
            const unsigned int m = e / (unsigned int)WMMA_K;
            const unsigned int k = e % (unsigned int)WMMA_K;
            const unsigned int row_g = tile_row + m;
            const unsigned int col_g = k_tile + k;
            float v = 0.0f;
            if (row_g < rows && col_g < cols) {
                const size_t packed_idx = (size_t)row_g * packed_cols + (size_t)(col_g / 2u);
                const unsigned int byte = packed[packed_idx];
                const unsigned int nibble = (col_g & 1u) ? (byte >> 4u) : (byte & 0x0Fu);
                const size_t scale_idx = (size_t)row_g * scale_cols + (size_t)(col_g / 16u);
                const float blk_scale = decode_ue4m3_half(scales[scale_idx]);
                v = (float)decode_nvfp4_nibble(nibble) * blk_scale;
            }
            sh_a[m * (unsigned int)WMMA_K + k] = __float2bfloat16(v);
        }

#if __CUDA_ARCH__ >= 800
        // Issue B[k+1] prefetch into the next buffer (if it exists), then
        // commit. After this, we have either 1 or 2 outstanding cp.async
        // groups (1 if no K+1, 2 if K+1 exists).
        if (k_idx + 1u < N_KITERS) {
            issue_b_prefetch((k_idx + 1u) * (unsigned int)WMMA_K, buf_nxt);
            cp_async_commit();
            // 2 outstanding groups; wait until ≤1 remain → B[k] is ready.
            cp_async_wait_lt1();
        } else {
            // 1 outstanding group (B[k] only); wait until 0 remain.
            cp_async_wait_lt0();
        }
        __syncthreads();

        // Cast f32 → bf16 from the staging buffer into the bf16 sh_b that
        // wmma::load_matrix_sync reads. Per-element __float2bfloat16 — same
        // operation as the t32 kernel uses on the global-loaded f32, so the
        // resulting bf16 is bit-identical.
        // 512 elems / 128 threads = 4 elems/thread.
        for (unsigned int e = tid; e < B_ELEMS; e += BLOCK_THREADS) {
            sh_b[e] = __float2bfloat16(sh_b_raw[buf_cur][e]);
        }
        __syncthreads();
#else
        // Fallback for pre-sm_80: synchronous load of B (matches t32 exactly).
        for (unsigned int e = tid; e < B_ELEMS; e += BLOCK_THREADS) {
            const unsigned int n = e / (unsigned int)WMMA_K;
            const unsigned int k = e % (unsigned int)WMMA_K;
            const unsigned int batch_in_e = tile_col_in_expert + n;
            const unsigned int col_g      = k_tile + k;
            float v = 0.0f;
            if (batch_in_e < tok_count && col_g < cols) {
                const unsigned int token_g = tok_start + batch_in_e;
                v = permuted_input[(size_t)token_g * cols + col_g];
            }
            sh_b[n * (unsigned int)WMMA_K + k] = __float2bfloat16(v);
        }
        __syncthreads();
#endif

        // ------ WMMA mma_sync (identical to t32 kernel). ------
        const __nv_bfloat16* a_base = sh_a + (warp_row * (unsigned int)WMMA_M) * (unsigned int)WMMA_K;
        const __nv_bfloat16* b_base = sh_b + (warp_col * (unsigned int)WMMA_N) * (unsigned int)WMMA_K;
        wmma::load_matrix_sync(a_frag, a_base, WMMA_K);
        wmma::load_matrix_sync(b_frag, b_base, WMMA_K);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    // Store this warp's 16x16 sub-tile into sh_c at offset (warp_row*16, warp_col*16).
    float* c_base = sh_c
        + (warp_row * (unsigned int)WMMA_M) * (unsigned int)TILE_N
        + (warp_col * (unsigned int)WMMA_N);
    wmma::store_matrix_sync(c_base, c_frag, TILE_N, wmma::mem_row_major);
    __syncthreads();

    // Cooperative strided write: 1024 elems / 128 threads = 8/thread.
    constexpr unsigned int C_ELEMS = (unsigned)(TILE_M * TILE_N);
    for (unsigned int e = tid; e < C_ELEMS; e += BLOCK_THREADS) {
        const unsigned int m = e / (unsigned int)TILE_N;
        const unsigned int n = e % (unsigned int)TILE_N;
        const unsigned int row_g     = tile_row + m;
        const unsigned int batch_in_e = tile_col_in_expert + n;
        if (row_g < rows && batch_in_e < tok_count) {
            const unsigned int token_g = tok_start + batch_in_e;
            permuted_output[(size_t)token_g * rows + row_g] =
                sh_c[m * (unsigned int)TILE_N + n] * output_scale;
        }
    }
}

// =============================================================================
// Grouped MoE NVFP4 GEMM dual-output (gate+up fused) — Phase B.4 Round 2.
//
// Drop-in fused replacement for two consecutive launches of
// `aegis_nvfp4_grouped_prequant_gemm_wmma_bf16_t32` that share the same
// `permuted_input` and have identical per-expert weight shapes (rows × cols).
// Used for the gate and up projections of the routed-expert MLP, which both
// read `permuted_input` and produce two separate outputs of identical shape.
//
// Per-block work performed (32×32 output tile, 4 warps):
//   * Each block iterates its K-loop once.
//   * Each K-iter loads ONE input tile (32×16 bf16) into shared memory and
//     reuses it for BOTH projections (gate and up). This halves the in-kernel
//     B-load traffic compared to two separate launches.
//   * Each K-iter loads TWO weight tiles (32×16 bf16 gate + 32×16 bf16 up).
//     The total A-load traffic is the same as two separate launches, but the
//     dequant pipeline is interleaved.
//   * Each K-iter performs TWO mma.sync ops per warp (one for gate, one for up).
//
// Bit-exactness vs. the standalone t32 kernel: each warp's accumulation order,
// per-K-tile WMMA products, BF16 cast order, and `mma.sync` semantics are
// identical; only the launch shape and the load pattern of the shared input
// tile differ. The per-element f32 accumulator in `c_gate_frag` matches what
// a standalone t32 launch for gate would produce, and same for `c_up_frag`.
//
// Inputs:
//   * `packed_base_gate / scales_base_gate / packed_offsets_gate / scales_offsets_gate / output_scales_gate`
//     — gate-projection weight metadata (same layout as t32 kernel).
//   * `packed_base_up / ...` — up-projection weight metadata.
//   * `expert_token_offsets` — shared by both projections (both consume the
//     same `permuted_input`).
//   * `permuted_input` — `[total_tokens, cols]`.
//   * `permuted_output_gate / permuted_output_up` — `[total_tokens, rows]`.
//
// Grid : (ceil(rows/32), ceil(max_tokens_per_active/32), num_active_experts)
// Block: 128 threads (4 warps), same partitioning as the standalone t32 kernel.
// Shared: 32*16 BF16 sh_a_gate (1 KiB) + 32*16 BF16 sh_a_up (1 KiB) +
//         32*16 BF16 sh_b (1 KiB) + 32*32 f32 sh_c_gate (4 KiB) +
//         32*32 f32 sh_c_up (4 KiB) = ~11 KiB. Well under Blackwell's 100 KiB.
// =============================================================================
extern "C" __global__ void aegis_nvfp4_grouped_prequant_gemm_wmma_bf16_t32_dual(
    const unsigned char* __restrict__ packed_base_gate,
    const unsigned char* __restrict__ scales_base_gate,
    const unsigned int*  __restrict__ packed_offsets_gate,        // [num_active_experts]
    const unsigned int*  __restrict__ scales_offsets_gate,        // [num_active_experts]
    const float*         __restrict__ output_scales_gate,         // [num_active_experts]
    const unsigned char* __restrict__ packed_base_up,
    const unsigned char* __restrict__ scales_base_up,
    const unsigned int*  __restrict__ packed_offsets_up,          // [num_active_experts]
    const unsigned int*  __restrict__ scales_offsets_up,          // [num_active_experts]
    const float*         __restrict__ output_scales_up,           // [num_active_experts]
    const unsigned int*  __restrict__ expert_token_offsets,       // [num_active_experts + 1]
    const float*         __restrict__ permuted_input,             // [total_tokens, cols]
    const unsigned int rows,
    const unsigned int cols,
    float*               __restrict__ permuted_output_gate,       // [total_tokens, rows]
    float*               __restrict__ permuted_output_up          // [total_tokens, rows]
) {
    using namespace nvcuda;
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
    constexpr int TILE_M = 32, TILE_N = 32;
    constexpr unsigned int WARPS_PER_BLOCK = 4u;

    const unsigned int active_e = blockIdx.z;
    const unsigned int tok_start = expert_token_offsets[active_e];
    const unsigned int tok_end   = expert_token_offsets[active_e + 1];
    const unsigned int tok_count = tok_end - tok_start;

    const unsigned int tile_row = blockIdx.x * (unsigned int)TILE_M;
    const unsigned int tile_col_in_expert = blockIdx.y * (unsigned int)TILE_N;

    if (tile_row >= rows) return;
    if (tile_col_in_expert >= tok_count) return;

    const unsigned int tid    = threadIdx.x;
    const unsigned int warp_id = tid >> 5;
    const unsigned int warp_row = warp_id >> 1;
    const unsigned int warp_col = warp_id & 1u;

    const unsigned char* packed_g = packed_base_gate + (size_t)packed_offsets_gate[active_e];
    const unsigned char* scales_g = scales_base_gate + (size_t)scales_offsets_gate[active_e];
    const float output_scale_g    = output_scales_gate[active_e];
    const unsigned char* packed_u = packed_base_up   + (size_t)packed_offsets_up[active_e];
    const unsigned char* scales_u = scales_base_up   + (size_t)scales_offsets_up[active_e];
    const float output_scale_u    = output_scales_up[active_e];

    __shared__ __nv_bfloat16 sh_a_gate[TILE_M * WMMA_K];
    __shared__ __nv_bfloat16 sh_a_up  [TILE_M * WMMA_K];
    __shared__ __nv_bfloat16 sh_b     [TILE_N * WMMA_K];
    __shared__ float          sh_c_gate[TILE_M * TILE_N];
    __shared__ float          sh_c_up  [TILE_M * TILE_N];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_gate_frag;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_up_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_gate_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_up_frag;
    wmma::fill_fragment(c_gate_frag, 0.0f);
    wmma::fill_fragment(c_up_frag, 0.0f);

    const unsigned int packed_cols = cols / 2u;
    const unsigned int scale_cols  = cols / 16u;

    constexpr unsigned int A_ELEMS = (unsigned)(TILE_M * WMMA_K);
    constexpr unsigned int B_ELEMS = (unsigned)(TILE_N * WMMA_K);
    constexpr unsigned int BLOCK_THREADS = WARPS_PER_BLOCK * 32u;

    for (unsigned int k_tile = 0u; k_tile < cols; k_tile += (unsigned int)WMMA_K) {
        // Load gate weight tile (32×16 bf16, row-major).
        for (unsigned int e = tid; e < A_ELEMS; e += BLOCK_THREADS) {
            const unsigned int m = e / (unsigned int)WMMA_K;
            const unsigned int k = e % (unsigned int)WMMA_K;
            const unsigned int row_g = tile_row + m;
            const unsigned int col_g = k_tile + k;
            float v = 0.0f;
            if (row_g < rows && col_g < cols) {
                const size_t packed_idx = (size_t)row_g * packed_cols + (size_t)(col_g / 2u);
                const unsigned int byte = packed_g[packed_idx];
                const unsigned int nibble = (col_g & 1u) ? (byte >> 4u) : (byte & 0x0Fu);
                const size_t scale_idx = (size_t)row_g * scale_cols + (size_t)(col_g / 16u);
                const float blk_scale = decode_ue4m3_half(scales_g[scale_idx]);
                v = (float)decode_nvfp4_nibble(nibble) * blk_scale;
            }
            sh_a_gate[m * (unsigned int)WMMA_K + k] = __float2bfloat16(v);
        }

        // Load up weight tile (32×16 bf16, row-major).
        for (unsigned int e = tid; e < A_ELEMS; e += BLOCK_THREADS) {
            const unsigned int m = e / (unsigned int)WMMA_K;
            const unsigned int k = e % (unsigned int)WMMA_K;
            const unsigned int row_g = tile_row + m;
            const unsigned int col_g = k_tile + k;
            float v = 0.0f;
            if (row_g < rows && col_g < cols) {
                const size_t packed_idx = (size_t)row_g * packed_cols + (size_t)(col_g / 2u);
                const unsigned int byte = packed_u[packed_idx];
                const unsigned int nibble = (col_g & 1u) ? (byte >> 4u) : (byte & 0x0Fu);
                const size_t scale_idx = (size_t)row_g * scale_cols + (size_t)(col_g / 16u);
                const float blk_scale = decode_ue4m3_half(scales_u[scale_idx]);
                v = (float)decode_nvfp4_nibble(nibble) * blk_scale;
            }
            sh_a_up[m * (unsigned int)WMMA_K + k] = __float2bfloat16(v);
        }

        // Load shared input tile (32×16 bf16, col-major: sh_b[n*K + k]).
        for (unsigned int e = tid; e < B_ELEMS; e += BLOCK_THREADS) {
            const unsigned int n = e / (unsigned int)WMMA_K;
            const unsigned int k = e % (unsigned int)WMMA_K;
            const unsigned int batch_in_e = tile_col_in_expert + n;
            const unsigned int col_g      = k_tile + k;
            float v = 0.0f;
            if (batch_in_e < tok_count && col_g < cols) {
                const unsigned int token_g = tok_start + batch_in_e;
                v = permuted_input[(size_t)token_g * cols + col_g];
            }
            sh_b[n * (unsigned int)WMMA_K + k] = __float2bfloat16(v);
        }
        __syncthreads();

        // Each warp: load its sub-tiles of A_gate, A_up, B; mma both.
        const __nv_bfloat16* a_gate_base = sh_a_gate + (warp_row * (unsigned int)WMMA_M) * (unsigned int)WMMA_K;
        const __nv_bfloat16* a_up_base   = sh_a_up   + (warp_row * (unsigned int)WMMA_M) * (unsigned int)WMMA_K;
        const __nv_bfloat16* b_base      = sh_b      + (warp_col * (unsigned int)WMMA_N) * (unsigned int)WMMA_K;
        wmma::load_matrix_sync(a_gate_frag, a_gate_base, WMMA_K);
        wmma::load_matrix_sync(a_up_frag,   a_up_base,   WMMA_K);
        wmma::load_matrix_sync(b_frag,      b_base,      WMMA_K);
        // Same per-element accumulation order as the standalone t32 kernel:
        // mma.sync(c, a_warp, b_warp, c) with identical operands per K-iter.
        wmma::mma_sync(c_gate_frag, a_gate_frag, b_frag, c_gate_frag);
        wmma::mma_sync(c_up_frag,   a_up_frag,   b_frag, c_up_frag);
        __syncthreads();
    }

    // Store both warps' 16x16 sub-tiles to their respective sh_c buffers.
    float* c_gate_base = sh_c_gate
        + (warp_row * (unsigned int)WMMA_M) * (unsigned int)TILE_N
        + (warp_col * (unsigned int)WMMA_N);
    float* c_up_base = sh_c_up
        + (warp_row * (unsigned int)WMMA_M) * (unsigned int)TILE_N
        + (warp_col * (unsigned int)WMMA_N);
    wmma::store_matrix_sync(c_gate_base, c_gate_frag, TILE_N, wmma::mem_row_major);
    wmma::store_matrix_sync(c_up_base,   c_up_frag,   TILE_N, wmma::mem_row_major);
    __syncthreads();

    // Cooperative strided write: fan out 1024 elements per output to global.
    constexpr unsigned int C_ELEMS = (unsigned)(TILE_M * TILE_N);
    for (unsigned int e = tid; e < C_ELEMS; e += BLOCK_THREADS) {
        const unsigned int m = e / (unsigned int)TILE_N;
        const unsigned int n = e % (unsigned int)TILE_N;
        const unsigned int row_g     = tile_row + m;
        const unsigned int batch_in_e = tile_col_in_expert + n;
        if (row_g < rows && batch_in_e < tok_count) {
            const unsigned int token_g = tok_start + batch_in_e;
            const size_t out_idx = (size_t)token_g * rows + row_g;
            permuted_output_gate[out_idx] = sh_c_gate[m * (unsigned int)TILE_N + n] * output_scale_g;
            permuted_output_up  [out_idx] = sh_c_up  [m * (unsigned int)TILE_N + n] * output_scale_u;
        }
    }
}

extern "C" __global__ void aegis_nvfp4_linear_reference_batched(
    const unsigned char* packed,
    const unsigned char* scales,
    const float* input,
    const unsigned int rows,
    const unsigned int cols,
    const float input_scale,
    const float output_scale,
    float* output
) {
    const unsigned int row = blockIdx.x;
    const unsigned int batch = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }

    const unsigned int packed_cols = cols / 2u;
    const unsigned int scale_cols = cols / 16u;
    const unsigned char* packed_row = packed + size_t(row) * packed_cols;
    const unsigned char* scale_row = scales + size_t(row) * scale_cols;
    const float* input_row = input + size_t(batch) * cols;
    extern __shared__ float partial[];

    float sum = 0.0f;
    for (unsigned int block_idx = tid; block_idx < scale_cols; block_idx += blockDim.x) {
        const float block_scale = decode_ue4m3_half(scale_row[block_idx]);
        const unsigned int input_base = block_idx * 16u;
        const unsigned int packed_base = block_idx * 8u;
        for (unsigned int j = 0u; j < 8u; ++j) {
            const unsigned int byte = packed_row[packed_base + j];
            const unsigned int lo_lane = 2u*j;
            const unsigned int hi_lane = lo_lane + 1u;
            const float input_lo = maybe_quantize_nvfp4_input(input_row, input_base, lo_lane, input_scale);
            const float input_hi = maybe_quantize_nvfp4_input(input_row, input_base, hi_lane, input_scale);
            sum += float(decode_nvfp4_nibble(byte & 0x0Fu)) * block_scale * input_lo;
            sum += float(decode_nvfp4_nibble(byte >> 4)) * block_scale * input_hi;
        }
    }

    partial[tid] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0u) {
        output[size_t(batch) * rows + row] = partial[0] * output_scale;
    }

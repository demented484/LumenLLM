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

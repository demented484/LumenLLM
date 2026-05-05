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

// Per-expert NVFP4 input quantizer for the grouped MoE prefill pipeline.
//
// Mirrors `aegis_nvfp4_quantize_input_batched` but takes a per-expert
// `expert_input_scales[num_experts]` array and the same prefix-sum/count
// arrays the rest of the grouped pipeline uses, so each (expert, row-in-
// expert) pair applies its own input_scale during the NVFP4 round-trip.
//
// Grid: (ceil(cols/16), max_count, num_experts), block: (16, 1, 1).
// Reads `permuted_input` and writes `permuted_quantized` at the same row
// offsets — out-of-range (batch_in_expert >= count[expert]) blocks early-exit.
extern "C" __global__ void aegis_nvfp4_quantize_input_per_expert(
    const float*         __restrict__ permuted_input,
    const unsigned int*  __restrict__ expert_counts,
    const unsigned int*  __restrict__ expert_first_token_off,
    const float*         __restrict__ expert_input_scales,
    const unsigned int cols,
    float*               __restrict__ permuted_quantized
) {
    const unsigned int group = blockIdx.x;
    const unsigned int batch_in_expert = blockIdx.y;
    const unsigned int expert = blockIdx.z;
    const unsigned int lane = threadIdx.x;
    if (lane >= 16u) return;

    const unsigned int count = expert_counts[expert];
    if (batch_in_expert >= count) return;

    const unsigned int base = group * 16u;
    if (base + lane >= cols) return;

    const unsigned int abs_row = expert_first_token_off[expert] + batch_in_expert;
    const float input_scale = expert_input_scales[expert];

    const float* input_row  = permuted_input     + (size_t)abs_row * cols;
    float*       output_row = permuted_quantized + (size_t)abs_row * cols;

    if (!(input_scale > 0.0f)) {
        output_row[base + lane] = input_row[base + lane];
        return;
    }
    const float inv = 1.0f / input_scale;
    float amax = 0.0f;
    for (unsigned int j = 0u; j < 16u && base + j < cols; ++j) {
        amax = fmaxf(amax, fabsf(input_row[base + j] * inv));
    }
    if (amax == 0.0f) {
        output_row[base + lane] = 0.0f;
        return;
    }
    const float block_scale = decode_ue4m3_half(fp32_to_ue4m3_halfbits(amax / 6.0f));
    const unsigned int nibble = best_nvfp4_index(input_row[base + lane] * inv, block_scale);
    output_row[base + lane] = float(decode_nvfp4_nibble(nibble)) * block_scale * input_scale;
}

// Grouped NVFP4 prequantized GEMM for MoE prefill.
//
// Modelled byte-for-byte on `aegis_nvfp4_linear_prequantized_batched_gemm`
// (8 warps share one weight row in shared memory, 8 batch rows per block,
// warp-level shfl_down reduction, output_scale applied at write) — but
// adds an outer `blockIdx.z = expert` dimension and per-expert weight
// pointer / output-scale lookups.
//
// Pre-condition: `permuted_quantized_input` was produced by
// `aegis_nvfp4_quantize_input_per_expert` so each row already encodes the
// expert-specific input_scale; the kernel itself never touches input_scale.
//
// Grid: (rows, ceil(max_count / BATCH_PER_BLOCK), num_experts).
// Block: 256 threads (8 warps).
// Shared mem: packed_cols + scale_cols bytes (one weight row).
extern "C" __global__ void aegis_nvfp4_grouped_prequant_gemm(
    const unsigned char*       __restrict__ base_packed,
    const unsigned long long*  __restrict__ packed_offsets,   // [num_experts] u64 byte offsets
    const unsigned char*       __restrict__ base_scales,
    const unsigned long long*  __restrict__ scales_offsets,   // [num_experts] u64 byte offsets
    const unsigned int*        __restrict__ expert_counts,         // [num_experts] (cached_counts)
    const unsigned int*        __restrict__ expert_first_token_off, // [num_experts + 1] CSR start
    const float*               __restrict__ expert_output_scales,   // [num_experts]
    const unsigned int rows,
    const unsigned int cols,
    const float*               __restrict__ permuted_quantized_input, // [total_assignments, cols]
    float*                     __restrict__ permuted_output           // [total_assignments, rows]
) {
    const unsigned int WARP_SIZE       = 32u;
    const unsigned int BATCH_PER_BLOCK = 8u;

    const unsigned int row        = blockIdx.x;
    const unsigned int batch_base = blockIdx.y * BATCH_PER_BLOCK;
    const unsigned int expert     = blockIdx.z;
    if (row >= rows) return;

    const unsigned int count = expert_counts[expert];
    if (batch_base >= count) return;

    const unsigned int tid     = threadIdx.x;
    const unsigned int warp_id = tid / WARP_SIZE;
    const unsigned int lane    = tid & (WARP_SIZE - 1u);

    const unsigned int packed_cols = cols / 2u;
    const unsigned int scale_cols  = cols / 16u;

    const unsigned char* w_packed = base_packed + (size_t)packed_offsets[expert]
                                  + (size_t)row * packed_cols;
    const unsigned char* w_scales = base_scales + (size_t)scales_offsets[expert]
                                  + (size_t)row * scale_cols;

    extern __shared__ unsigned char sh[];
    unsigned char* sh_packed = sh;
    unsigned char* sh_scales = sh + packed_cols;

    // Collaborative load of the weight row — ALL 256 threads cooperate even
    // if their `batch_in_expert` slot is past `count`; otherwise we'd lose
    // bandwidth from idle warps.
    for (unsigned int i = tid; i < packed_cols; i += 256u) sh_packed[i] = w_packed[i];
    for (unsigned int i = tid; i < scale_cols;  i += 256u) sh_scales[i] = w_scales[i];
    __syncthreads();

    const unsigned int batch_in_expert = batch_base + warp_id;
    if (batch_in_expert >= count) return;

    const unsigned int abs_row = expert_first_token_off[expert] + batch_in_expert;
    const float* input_row = permuted_quantized_input + (size_t)abs_row * cols;

    float sum = 0.0f;
    aegis_nvfp4_prequant_gemm_inner(sh_packed, sh_scales, input_row, scale_cols, lane, sum);
    if (lane == 0u) {
        permuted_output[(size_t)abs_row * rows + row] = sum * expert_output_scales[expert];
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

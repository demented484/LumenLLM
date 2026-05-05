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
    for (unsigned int block_idx = lane; block_idx < scale_cols; block_idx += WARP_SIZE) {
        const float         blk_scale   = decode_ue4m3_half(sh_scales[block_idx]);
        const unsigned int  input_base  = block_idx * 16u;
        const unsigned int  packed_base = block_idx * 8u;
        for (unsigned int j = 0u; j < 8u; ++j) {
            const unsigned int byte = sh_packed[packed_base + j];
            sum += (float)decode_nvfp4_nibble(byte & 0x0Fu) * blk_scale * input_row[input_base + 2u*j];
            sum += (float)decode_nvfp4_nibble(byte >> 4u)   * blk_scale * input_row[input_base + 2u*j + 1u];
        }
    }

    // Warp-level reduction via shuffle.
    for (unsigned int stride = WARP_SIZE / 2u; stride > 0u; stride >>= 1u) {
        sum += __shfl_down_sync(0xffffffffu, sum, stride);
    }
    if (lane == 0u) {
        output[(size_t)batch_idx * rows + row] = sum * output_scale;
    }
}

// Grouped NVFP4 matvec (Phase 2 of perf overhaul, MoE prefill).
//
// One launch processes ALL active experts of a layer for one matmul-position
// (gate, up, OR down). Replaces the per-expert dispatch loop's ~50 launches
// with a single launch — kills launch overhead which dominates for small
// per-expert batches (~5 tokens × 30 layers × 3 matmuls = ~4500 launches/chunk
// → ~3 launches/chunk after this). Modeled after the vLLM `fused_moe_kernel`
// pattern: grid covers all (output_row × batch_in_expert × expert) tiles,
// per-block early-exit when the expert is empty or the batch slot is past
// the count.
//
// Inputs are in **permuted (expert-sorted) layout** as produced by
// `aegis_permute_gather_f32`: `permuted_input[expert_first_token_off[e] +
// batch_in_expert]` is the input row for expert e's `batch_in_expert`-th
// token. Outputs have the same layout.
//
// Weight pointers are taken from a single base buffer (typically the VRAM
// expert cache) plus per-expert offset arrays. For experts not in the cache
// the host loop dispatches the legacy per-expert kernel separately and
// passes 0 for the count here so the grouped kernel skips them.
extern "C" __global__ void aegis_nvfp4_grouped_matvec_packed(
    const unsigned char* __restrict__ base_packed,         // big buffer (e.g. cache)
    const unsigned int*  __restrict__ packed_offsets,       // [num_experts] byte offsets
    const unsigned char* __restrict__ base_scales,          // typically same as base_packed
    const unsigned int*  __restrict__ scales_offsets,       // [num_experts]
    const unsigned int*  __restrict__ expert_counts,        // [num_experts]
    const unsigned int*  __restrict__ expert_first_token_off, // [num_experts + 1]
    const float* __restrict__ expert_input_scales,           // [num_experts]
    const float* __restrict__ expert_output_scales,          // [num_experts]
    const unsigned int rows,                                // shared N (output dim)
    const unsigned int cols,                                // shared K (input dim)
    const float* __restrict__ permuted_input,               // [total_assignments, cols]
    float*       __restrict__ permuted_output               // [total_assignments, rows]
) {
    const unsigned int row = blockIdx.x;
    const unsigned int batch_in_expert = blockIdx.y;
    const unsigned int expert = blockIdx.z;
    if (row >= rows) return;
    const unsigned int count = expert_counts[expert];
    if (batch_in_expert >= count) return;

    const unsigned int abs_row = expert_first_token_off[expert] + batch_in_expert;
    const unsigned char* packed = base_packed + (size_t)packed_offsets[expert];
    const unsigned char* scales = base_scales + (size_t)scales_offsets[expert];
    const float input_scale = expert_input_scales[expert];
    const float output_scale = expert_output_scales[expert];

    const unsigned int packed_cols = cols / 2u;
    const unsigned int scale_cols = cols / 16u;
    const unsigned char* p_row = packed + (size_t)row * packed_cols;
    const unsigned char* s_row = scales + (size_t)row * scale_cols;
    const float* in_row = permuted_input + (size_t)abs_row * cols;

    extern __shared__ float partial[];
    float sum = 0.0f;
    const unsigned int tid = threadIdx.x;
    for (unsigned int blk = tid; blk < scale_cols; blk += blockDim.x) {
        const float bs = decode_ue4m3_half(s_row[blk]);
        const unsigned int input_base = blk * 16u;
        const unsigned int packed_base = blk * 8u;
        for (unsigned int j = 0u; j < 8u; ++j) {
            const unsigned int b = p_row[packed_base + j];
            const unsigned int lo_lane = 2u * j;
            const unsigned int hi_lane = lo_lane + 1u;
            const float input_lo = maybe_quantize_nvfp4_input(in_row, input_base, lo_lane, input_scale);
            const float input_hi = maybe_quantize_nvfp4_input(in_row, input_base, hi_lane, input_scale);
            sum += float(decode_nvfp4_nibble(b & 0x0Fu)) * bs * input_lo;
            sum += float(decode_nvfp4_nibble(b >> 4)) * bs * input_hi;
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
        permuted_output[(size_t)abs_row * rows + row] = partial[0] * output_scale;
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

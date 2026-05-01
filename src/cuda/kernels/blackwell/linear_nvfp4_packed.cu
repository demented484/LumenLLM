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

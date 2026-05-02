extern "C" __global__ void aegis_bf16_matvec_reference(
    const unsigned short* matrix,
    const float* input,
    const unsigned int rows,
    const unsigned int cols,
    float* output
) {
    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }

    extern __shared__ float partial[];
    float sum = 0.0f;
    const unsigned short* matrix_row = matrix + size_t(row) * cols;
    for (unsigned int col = tid; col < cols; col += blockDim.x) {
        sum += bf16_to_float(matrix_row[col]) * input[col];
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
        output[row] = partial[0];
    }
}

extern "C" __global__ void aegis_argmax_f32_blocks(
    const float* input,
    const unsigned int len,
    float* block_values,
    unsigned int* block_indices
) {
    __shared__ float values[256];
    __shared__ unsigned int indices[256];
    const unsigned int tid = threadIdx.x;
    const unsigned int idx = blockIdx.x * blockDim.x + tid;
    float value = -3.402823466e38f;
    unsigned int out_idx = 0xffffffffu;
    if (idx < len) {
        value = input[idx];
        out_idx = idx;
    }
    values[tid] = value;
    indices[tid] = out_idx;
    __syncthreads();

    for (unsigned int stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            const float other_value = values[tid + stride];
            const unsigned int other_idx = indices[tid + stride];
            const bool take_other = other_value > values[tid]
                || (other_value == values[tid] && other_idx < indices[tid]);
            if (take_other) {
                values[tid] = other_value;
                indices[tid] = other_idx;
            }
        }
        __syncthreads();
    }

    if (tid == 0u) {
        block_values[blockIdx.x] = values[0];
        block_indices[blockIdx.x] = indices[0];
    }
}

extern "C" __global__ void aegis_argmax_f32_finalize(
    const float* block_values,
    const unsigned int* block_indices,
    const unsigned int num_blocks,
    unsigned int* output_token
) {
    float best_value = -3.402823466e38f;
    unsigned int best_idx = 0u;
    for (unsigned int idx = 0u; idx < num_blocks; ++idx) {
        const float value = block_values[idx];
        const unsigned int token = block_indices[idx];
        if (value > best_value || (value == best_value && token < best_idx)) {
            best_value = value;
            best_idx = token;
        }
    }
    output_token[0] = best_idx;
}

#include <cuda_fp16.h>

extern "C" __device__ __forceinline__ int decode_nvfp4_nibble(unsigned int nibble) {
    switch (nibble & 0xFu) {
        case 0u: return 0;
        case 1u: return 1;
        case 2u: return 2;
        case 3u: return 3;
        case 4u: return 4;
        case 5u: return 6;
        case 6u: return 8;
        case 7u: return 12;
        case 8u: return 0;
        case 9u: return -1;
        case 10u: return -2;
        case 11u: return -3;
        case 12u: return -4;
        case 13u: return -6;
        case 14u: return -8;
        default: return -12;
    }
}

extern "C" __device__ __forceinline__ float decode_ue4m3_half(unsigned int byte) {
    byte &= 0x7Fu;
    if (byte == 0u || byte == 0x7Fu) {
        return 0.0f;
    }
    const int exponent = int((byte >> 3) & 0x0Fu);
    const float mantissa = float(byte & 0x07u);
    const float raw = exponent == 0
        ? mantissa * 0.001953125f
        : (1.0f + mantissa * 0.125f) * exp2f(float(exponent - 7));
    return raw * 0.5f;
}

extern "C" __device__ __forceinline__ unsigned int fp32_to_ue4m3_halfbits(float x) {
    if (!(x > 0.0f)) {
        return 0u;
    }
    if (x > 448.0f) {
        x = 448.0f;
    }
    const unsigned int bits = __float_as_uint(x);
    const int fp32_exp = int((bits >> 23) & 0xffu) - 127;
    const int fp32_man = int((bits >> 20) & 0x7u);
    int ue4m3_exp = fp32_exp + 7;
    if (ue4m3_exp <= 0) {
        int man = int(x * 512.0f + 0.5f);
        if (man > 7) man = 7;
        if (man < 1) return 0u;
        return (unsigned int)man;
    }
    if (ue4m3_exp >= 15) {
        return 0x7eu;
    }
    const int round_bit = int((bits >> 19) & 1u);
    int ue4m3_man = fp32_man + round_bit;
    if (ue4m3_man > 7) {
        ue4m3_man = 0;
        ue4m3_exp += 1;
        if (ue4m3_exp >= 15) {
            return 0x7eu;
        }
    }
    return (unsigned int)((ue4m3_exp << 3) | ue4m3_man);
}

extern "C" __device__ __forceinline__ unsigned int best_nvfp4_index(float x, float d) {
    if (d == 0.0f) {
        return 0u;
    }
    unsigned int best = 0u;
    float best_err = 3.402823466e38f;
    for (unsigned int idx = 0; idx < 16u; ++idx) {
        const float candidate = float(decode_nvfp4_nibble(idx)) * d;
        const float err = fabsf(candidate - x);
        if (err < best_err) {
            best = idx;
            best_err = err;
        }
    }
    return best;
}

extern "C" __device__ __forceinline__ float maybe_quantize_nvfp4_input(
    const float* input,
    const unsigned int base,
    const unsigned int lane,
    const float input_scale
) {
    float value = input[base + lane];
    if (!(input_scale > 0.0f)) {
        return value;
    }

    const float inv = 1.0f / input_scale;
    float amax = 0.0f;
    for (unsigned int j = 0u; j < 16u; ++j) {
        amax = fmaxf(amax, fabsf(input[base + j] * inv));
    }
    if (amax == 0.0f) {
        return 0.0f;
    }
    const float block_scale = decode_ue4m3_half(fp32_to_ue4m3_halfbits(amax / 6.0f));
    const unsigned int nibble = best_nvfp4_index(value * inv, block_scale);
    return float(decode_nvfp4_nibble(nibble)) * block_scale * input_scale;
}

extern "C" __global__ void aegis_nvfp4_quantize_input(
    const float* input,
    const unsigned int len,
    const float input_scale,
    float* output
) {
    const unsigned int base = blockIdx.x * 16u;
    const unsigned int lane = threadIdx.x;
    if (lane >= 16u || base + lane >= len) {
        return;
    }
    if (!(input_scale > 0.0f)) {
        output[base + lane] = input[base + lane];
        return;
    }

    const float inv = 1.0f / input_scale;
    float amax = 0.0f;
    for (unsigned int j = 0u; j < 16u && base + j < len; ++j) {
        amax = fmaxf(amax, fabsf(input[base + j] * inv));
    }
    if (amax == 0.0f) {
        output[base + lane] = 0.0f;
        return;
    }
    const float block_scale = decode_ue4m3_half(fp32_to_ue4m3_halfbits(amax / 6.0f));
    const unsigned int nibble = best_nvfp4_index(input[base + lane] * inv, block_scale);
    output[base + lane] = float(decode_nvfp4_nibble(nibble)) * block_scale * input_scale;
}

extern "C" __global__ void aegis_nvfp4_quantize_input_batched(
    const float* input,
    const unsigned int batch,
    const unsigned int len,
    const float input_scale,
    float* output
) {
    const unsigned int group = blockIdx.x;
    const unsigned int batch_idx = blockIdx.y;
    const unsigned int lane = threadIdx.x;
    const unsigned int base = group * 16u;
    if (batch_idx >= batch || lane >= 16u || base + lane >= len) {
        return;
    }
    const float* input_row = input + size_t(batch_idx) * len;
    float* output_row = output + size_t(batch_idx) * len;
    if (!(input_scale > 0.0f)) {
        output_row[base + lane] = input_row[base + lane];
        return;
    }

    const float inv = 1.0f / input_scale;
    float amax = 0.0f;
    for (unsigned int j = 0u; j < 16u && base + j < len; ++j) {
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

extern "C" __device__ __forceinline__ float e8m0_to_fp32(unsigned int value) {
    const unsigned int bits = value == 0u ? 0x00400000u : (value << 23);
    return __uint_as_float(bits);
}

extern "C" __device__ __forceinline__ unsigned int compute_e8m0_scale(float amax) {
    if (!(amax > 0.0f)) {
        return 0u;
    }
    int exponent = int(ceilf(log2f(amax / 6.0f))) + 127;
    exponent = exponent < 0 ? 0 : exponent;
    exponent = exponent > 254 ? 254 : exponent;
    return static_cast<unsigned int>(exponent);
}

extern "C" __device__ __forceinline__ float warp_max(float value) {
    for (unsigned int mask = 16u; mask > 0u; mask >>= 1u) {
        value = fmaxf(value, __shfl_xor_sync(0xFFFFFFFFu, value, mask, 32));
    }
    return value;
}

extern "C" __device__ __forceinline__ unsigned int load_u32_unaligned(const unsigned char* ptr) {
    return unsigned(ptr[0])
        | (unsigned(ptr[1]) << 8)
        | (unsigned(ptr[2]) << 16)
        | (unsigned(ptr[3]) << 24);
}

extern "C" __device__ __forceinline__ void store_u32_unaligned(unsigned char* ptr, unsigned int value) {
    ptr[0] = value & 0xFFu;
    ptr[1] = (value >> 8) & 0xFFu;
    ptr[2] = (value >> 16) & 0xFFu;
    ptr[3] = (value >> 24) & 0xFFu;
}

extern "C" __device__ __forceinline__ unsigned int pack_mxfp4_word_from_warp(
    const unsigned int q,
    const unsigned int physical_word
) {
    const unsigned int lane_base = (physical_word & 3u) * 4u;
    unsigned int word = 0u;
#pragma unroll
    for (unsigned int j = 0u; j < 4u; ++j) {
        const unsigned int lo = __shfl_sync(0xFFFFFFFFu, q, lane_base + j, 32) & 0x0Fu;
        const unsigned int hi = __shfl_sync(0xFFFFFFFFu, q, lane_base + j + 16u, 32) & 0x0Fu;
        word |= (lo | (hi << 4)) << (8u * j);
    }
    return word;
}

extern "C" __global__ void aegis_mxfp4_quantize_vector(
    const float* input,
    const unsigned int len,
    unsigned char* output
) {
    const unsigned int lane = threadIdx.x;
    const unsigned int batch = blockIdx.y;
    const unsigned int k_base = blockIdx.x * 64u;
    if (k_base + 63u >= len) {
        return;
    }
    input += size_t(batch) * len;
    output += size_t(batch) * (len / 64u) * 36u;

    const float x0 = input[k_base + lane];
    const float x1 = input[k_base + lane + 32u];
    const float amax0 = warp_max(fabsf(x0));
    const float amax1 = warp_max(fabsf(x1));
    const unsigned int e0 = compute_e8m0_scale(amax0);
    const unsigned int e1 = compute_e8m0_scale(amax1);
    const float d_scale0 = e8m0_to_fp32(e0) * 0.5f;
    const float d_scale1 = e8m0_to_fp32(e1) * 0.5f;
    const unsigned int q0 = best_nvfp4_index(x0, d_scale0);
    const unsigned int q1 = best_nvfp4_index(x1, d_scale1);

    unsigned char* block = output + blockIdx.x * 36u;
    if (lane == 0u) {
        store_u32_unaligned(block, e0 | (e1 << 8));
    }
    const unsigned int packed_word0 = pack_mxfp4_word_from_warp(q0, lane & 3u);
    const unsigned int packed_word1 = pack_mxfp4_word_from_warp(q1, lane & 3u);
    if (lane < 8u) {
        const unsigned int word = lane < 4u ? packed_word0 : packed_word1;
        store_u32_unaligned(block + 4u + lane * 4u, word);
    }
}

extern "C" __global__ void aegis_swiglu_mxfp4_quantize_batched(
    const float* gate,
    const float* up,
    const unsigned int batch,
    const unsigned int len,
    unsigned char* output
) {
    const unsigned int lane = threadIdx.x;
    const unsigned int batch_idx = blockIdx.y;
    const unsigned int k_base = blockIdx.x * 64u;
    if (batch_idx >= batch || k_base + 63u >= len) {
        return;
    }
    const float* gate_row = gate + size_t(batch_idx) * len;
    const float* up_row = up + size_t(batch_idx) * len;
    output += (size_t(batch_idx) * (len / 64u) + blockIdx.x) * 36u;

    const unsigned int idx0 = k_base + lane;
    const unsigned int idx1 = k_base + lane + 32u;
    const float g0 = gate_row[idx0];
    const float g1 = gate_row[idx1];
    const float x0 = (g0 / (1.0f + expf(-g0))) * up_row[idx0];
    const float x1 = (g1 / (1.0f + expf(-g1))) * up_row[idx1];
    const float amax0 = warp_max(fabsf(x0));
    const float amax1 = warp_max(fabsf(x1));
    const unsigned int e0 = compute_e8m0_scale(amax0);
    const unsigned int e1 = compute_e8m0_scale(amax1);
    const float d_scale0 = e8m0_to_fp32(e0) * 0.5f;
    const float d_scale1 = e8m0_to_fp32(e1) * 0.5f;
    const unsigned int q0 = best_nvfp4_index(x0, d_scale0);
    const unsigned int q1 = best_nvfp4_index(x1, d_scale1);

    if (lane == 0u) {
        store_u32_unaligned(output, e0 | (e1 << 8));
    }
    const unsigned int packed_word0 = pack_mxfp4_word_from_warp(q0, lane & 3u);
    const unsigned int packed_word1 = pack_mxfp4_word_from_warp(q1, lane & 3u);
    if (lane < 8u) {
        const unsigned int word = lane < 4u ? packed_word0 : packed_word1;
        store_u32_unaligned(output + 4u + lane * 4u, word);
    }
}

extern "C" __global__ void aegis_mxfp4_matvec_native(
    const unsigned char* mxfp4,
    const unsigned char* input_mxfp4,
    const unsigned int rows,
    const unsigned int cols,
    const unsigned int blocks_per_row,
    const float output_scale,
    float* output
) {
    const unsigned int lane = threadIdx.x;
    const unsigned int batch = blockIdx.y;
    const unsigned int row_block = blockIdx.x * 16u;
    const unsigned int input_blocks = cols / 64u;
    input_mxfp4 += size_t(batch) * input_blocks * 36u;
    output += size_t(batch) * rows;

#if __CUDA_ARCH__ >= 1200
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    const unsigned int k_tiles = cols / 64u;
    __shared__ __align__(16) int a_shared[16 * 8];

    for (unsigned int ktile = 0u; ktile < k_tiles; ++ktile) {
        const unsigned int block_base = ktile * 2u;

        for (unsigned int idx = lane; idx < 16u * 8u; idx += 32u) {
            const unsigned int row_in_tile = idx / 8u;
            const unsigned int physical_word = idx & 7u;
            const unsigned int row = row_block + row_in_tile;
            unsigned int word = 0u;
            if (row < rows) {
                const unsigned int k_block = block_base + physical_word / 4u;
                const unsigned int word_in_block = physical_word & 3u;
                const unsigned char* block = mxfp4 + (size_t(row) * blocks_per_row + k_block) * 17u;
                word = load_u32_unaligned(block + 1u + word_in_block * 4u);
            }
            a_shared[idx] = int(word);
        }
        __syncwarp();

        unsigned int a0 = 0u;
        unsigned int a1 = 0u;
        unsigned int a2 = 0u;
        unsigned int a3 = 0u;
        const int* a_ptr = a_shared + (lane % 16u) * 8u + (lane / 16u) * 4u;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
            : "l"(a_ptr));
        __syncwarp();

        const unsigned char* input_block = input_mxfp4 + ktile * 36u;
        const unsigned int b_word0 = lane & 3u;
        const unsigned int b_word1 = (lane & 3u) + 4u;
        const unsigned int b0 = load_u32_unaligned(input_block + 4u + b_word0 * 4u);
        const unsigned int b1 = load_u32_unaligned(input_block + 4u + b_word1 * 4u);
        const unsigned int scale_b = load_u32_unaligned(input_block);

        const unsigned int scale_row_in_tile = (lane / 4u) + ((lane & 1u) * 8u);
        const unsigned int scale_row = row_block + scale_row_in_tile;
        unsigned int scale_a = 0u;
        if (scale_row < rows) {
            const unsigned char* block0 = mxfp4 + (size_t(scale_row) * blocks_per_row + block_base) * 17u;
            const unsigned char* block1 = block0 + 17u;
            scale_a = unsigned(block0[0]) | (unsigned(block1[0]) << 8);
        }

        asm volatile(
            "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
            "%10, {0, 0}, %11, {0, 0};"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(scale_a), "r"(scale_b));
    }

    if ((lane & 3u) == 0u) {
        const unsigned int row0 = row_block + lane / 4u;
        const unsigned int row1 = row0 + 8u;
        if (row0 < rows) {
            output[row0] = d0 * output_scale;
        }
        if (row1 < rows) {
            output[row1] = d2 * output_scale;
        }
    }
#else
    if (lane == 0u) {
        for (unsigned int row = row_block; row < rows && row < row_block + 16u; ++row) {
            output[row] = 0.0f;
        }
    }
#endif
}

extern "C" __global__ void aegis_mxfp4_matmul_native_n8(
    const unsigned char* mxfp4,
    const unsigned char* input_mxfp4,
    const unsigned int rows,
    const unsigned int cols,
    const unsigned int blocks_per_row,
    const unsigned int batch,
    const float output_scale,
    float* output
) {
    const unsigned int lane = threadIdx.x;
    const unsigned int row_block = blockIdx.x * 16u;
    const unsigned int batch_tile = blockIdx.y * 8u;
    const unsigned int input_blocks = cols / 64u;

#if __CUDA_ARCH__ >= 1200
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    const unsigned int k_tiles = cols / 64u;
    __shared__ __align__(16) int a_shared[16 * 8];

    for (unsigned int ktile = 0u; ktile < k_tiles; ++ktile) {
        const unsigned int block_base = ktile * 2u;

        for (unsigned int idx = lane; idx < 16u * 8u; idx += 32u) {
            const unsigned int row_in_tile = idx / 8u;
            const unsigned int physical_word = idx & 7u;
            const unsigned int row = row_block + row_in_tile;
            unsigned int word = 0u;
            if (row < rows) {
                const unsigned int k_block = block_base + physical_word / 4u;
                const unsigned int word_in_block = physical_word & 3u;
                const unsigned char* block = mxfp4 + (size_t(row) * blocks_per_row + k_block) * 17u;
                word = load_u32_unaligned(block + 1u + word_in_block * 4u);
            }
            a_shared[idx] = int(word);
        }
        __syncwarp();

        unsigned int a0 = 0u;
        unsigned int a1 = 0u;
        unsigned int a2 = 0u;
        unsigned int a3 = 0u;
        const int* a_ptr = a_shared + (lane % 16u) * 8u + (lane / 16u) * 4u;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
            : "l"(a_ptr));
        __syncwarp();

        const unsigned int token_in_tile = lane / 4u;
        const unsigned int token = batch_tile + token_in_tile;
        const unsigned int word_in_token = lane & 3u;
        unsigned int b0 = 0u;
        unsigned int b1 = 0u;
        unsigned int scale_b = 0u;
        if (token < batch) {
            const unsigned char* input_block =
                input_mxfp4 + (size_t(token) * input_blocks + ktile) * 36u;
            b0 = load_u32_unaligned(input_block + 4u + word_in_token * 4u);
            b1 = load_u32_unaligned(input_block + 4u + (word_in_token + 4u) * 4u);
            scale_b = load_u32_unaligned(input_block);
        }

        const unsigned int scale_row_in_tile = (lane / 4u) + ((lane & 1u) * 8u);
        const unsigned int scale_row = row_block + scale_row_in_tile;
        unsigned int scale_a = 0u;
        if (scale_row < rows) {
            const unsigned char* block0 = mxfp4 + (size_t(scale_row) * blocks_per_row + block_base) * 17u;
            const unsigned char* block1 = block0 + 17u;
            scale_a = unsigned(block0[0]) | (unsigned(block1[0]) << 8);
        }

        asm volatile(
            "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
            "%10, {0, 0}, %11, {0, 0};"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(scale_a), "r"(scale_b));
    }

    const unsigned int row0 = row_block + lane / 4u;
    const unsigned int row1 = row0 + 8u;
    const unsigned int col0 = batch_tile + (lane & 3u) * 2u;
    const unsigned int col1 = col0 + 1u;
    if (row0 < rows && col0 < batch) {
        output[size_t(col0) * rows + row0] = d0 * output_scale;
    }
    if (row0 < rows && col1 < batch) {
        output[size_t(col1) * rows + row0] = d1 * output_scale;
    }
    if (row1 < rows && col0 < batch) {
        output[size_t(col0) * rows + row1] = d2 * output_scale;
    }
    if (row1 < rows && col1 < batch) {
        output[size_t(col1) * rows + row1] = d3 * output_scale;
    }
#else
    if (lane == 0u) {
        for (unsigned int token = batch_tile; token < batch && token < batch_tile + 8u; ++token) {
            for (unsigned int row = row_block; row < rows && row < row_block + 16u; ++row) {
                output[size_t(token) * rows + row] = 0.0f;
            }
        }
    }
#endif
}

extern "C" __global__ void aegis_mxfp4_matmul_native_tile_m16n32(
    const unsigned char* mxfp4,
    const unsigned char* input_mxfp4,
    const unsigned int rows,
    const unsigned int cols,
    const unsigned int blocks_per_row,
    const unsigned int batch,
    const float output_scale,
    float* output
) {
    const unsigned int warp = threadIdx.x >> 5u;
    const unsigned int lane = threadIdx.x & 31u;
    if (warp >= 4u) {
        return;
    }
    const unsigned int row_subtile = warp & 1u;
    const unsigned int token_subtile = warp >> 1u;
    const unsigned int row_block = blockIdx.x * 32u + row_subtile * 16u;
    const unsigned int batch_tile = blockIdx.y * 16u + token_subtile * 8u;
    const unsigned int input_blocks = cols / 64u;

#if __CUDA_ARCH__ >= 1200
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    const unsigned int k_tiles = cols / 64u;
    __shared__ __align__(16) int a_shared[4 * 16 * 8];
    int* warp_a_shared = a_shared + warp * 16u * 8u;

    for (unsigned int ktile = 0u; ktile < k_tiles; ++ktile) {
        const unsigned int block_base = ktile * 2u;

        for (unsigned int idx = lane; idx < 16u * 8u; idx += 32u) {
            const unsigned int row_in_tile = idx / 8u;
            const unsigned int physical_word = idx & 7u;
            const unsigned int row = row_block + row_in_tile;
            unsigned int word = 0u;
            if (row < rows) {
                const unsigned int k_block = block_base + physical_word / 4u;
                const unsigned int word_in_block = physical_word & 3u;
                const unsigned char* block = mxfp4 + (size_t(row) * blocks_per_row + k_block) * 17u;
                word = load_u32_unaligned(block + 1u + word_in_block * 4u);
            }
            warp_a_shared[idx] = int(word);
        }
        __syncwarp();

        unsigned int a0 = 0u;
        unsigned int a1 = 0u;
        unsigned int a2 = 0u;
        unsigned int a3 = 0u;
        const int* a_ptr = warp_a_shared + (lane % 16u) * 8u + (lane / 16u) * 4u;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
            : "l"(a_ptr));
        __syncwarp();

        const unsigned int token_in_tile = lane / 4u;
        const unsigned int token = batch_tile + token_in_tile;
        const unsigned int word_in_token = lane & 3u;
        unsigned int b0 = 0u;
        unsigned int b1 = 0u;
        unsigned int scale_b = 0u;
        if (token < batch) {
            const unsigned char* input_block =
                input_mxfp4 + (size_t(token) * input_blocks + ktile) * 36u;
            b0 = load_u32_unaligned(input_block + 4u + word_in_token * 4u);
            b1 = load_u32_unaligned(input_block + 4u + (word_in_token + 4u) * 4u);
            scale_b = load_u32_unaligned(input_block);
        }

        const unsigned int scale_row_in_tile = (lane / 4u) + ((lane & 1u) * 8u);
        const unsigned int scale_row = row_block + scale_row_in_tile;
        unsigned int scale_a = 0u;
        if (scale_row < rows) {
            const unsigned char* block0 = mxfp4 + (size_t(scale_row) * blocks_per_row + block_base) * 17u;
            const unsigned char* block1 = block0 + 17u;
            scale_a = unsigned(block0[0]) | (unsigned(block1[0]) << 8);
        }

        asm volatile(
            "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
            "%10, {0, 0}, %11, {0, 0};"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(scale_a), "r"(scale_b));
    }

    const unsigned int row0 = row_block + lane / 4u;
    const unsigned int row1 = row0 + 8u;
    const unsigned int col0 = batch_tile + (lane & 3u) * 2u;
    const unsigned int col1 = col0 + 1u;
    if (row0 < rows && col0 < batch) {
        output[size_t(col0) * rows + row0] = d0 * output_scale;
    }
    if (row0 < rows && col1 < batch) {
        output[size_t(col1) * rows + row0] = d1 * output_scale;
    }
    if (row1 < rows && col0 < batch) {
        output[size_t(col0) * rows + row1] = d2 * output_scale;
    }
    if (row1 < rows && col1 < batch) {
        output[size_t(col1) * rows + row1] = d3 * output_scale;
    }
#else
    if (lane == 0u) {
        for (unsigned int token = batch_tile; token < batch && token < batch_tile + 8u; ++token) {
            for (unsigned int row = row_block; row < rows && row < row_block + 16u; ++row) {
                output[size_t(token) * rows + row] = 0.0f;
            }
        }
    }
#endif
}

extern "C" __device__ __forceinline__ void mxfp4_matmul_tile_m16n64_core(
    const unsigned char* mxfp4,
    const unsigned char* input_mxfp4,
    const unsigned int rows,
    const unsigned int cols,
    const unsigned int blocks_per_row,
    const unsigned int batch,
    const float output_scale,
    float* output,
    int* a_shared
) {
    const unsigned int warp = threadIdx.x >> 5u;
    const unsigned int lane = threadIdx.x & 31u;
    if (warp >= 8u) {
        return;
    }
    const unsigned int row_subtile = warp & 3u;
    const unsigned int token_subtile = warp >> 2u;
    const unsigned int row_block = blockIdx.x * 64u + row_subtile * 16u;
    const unsigned int batch_tile = blockIdx.y * 16u + token_subtile * 8u;
    const unsigned int input_blocks = cols / 64u;

#if __CUDA_ARCH__ >= 1200
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    const unsigned int k_tiles = cols / 64u;
    int* warp_a_shared = a_shared + warp * 16u * 8u;

    for (unsigned int ktile = 0u; ktile < k_tiles; ++ktile) {
        const unsigned int block_base = ktile * 2u;

        for (unsigned int idx = lane; idx < 16u * 8u; idx += 32u) {
            const unsigned int row_in_tile = idx / 8u;
            const unsigned int physical_word = idx & 7u;
            const unsigned int row = row_block + row_in_tile;
            unsigned int word = 0u;
            if (row < rows) {
                const unsigned int k_block = block_base + physical_word / 4u;
                const unsigned int word_in_block = physical_word & 3u;
                const unsigned char* block = mxfp4 + (size_t(row) * blocks_per_row + k_block) * 17u;
                word = load_u32_unaligned(block + 1u + word_in_block * 4u);
            }
            warp_a_shared[idx] = int(word);
        }
        __syncwarp();

        unsigned int a0 = 0u;
        unsigned int a1 = 0u;
        unsigned int a2 = 0u;
        unsigned int a3 = 0u;
        const int* a_ptr = warp_a_shared + (lane % 16u) * 8u + (lane / 16u) * 4u;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
            : "l"(a_ptr));
        __syncwarp();

        const unsigned int token_in_tile = lane / 4u;
        const unsigned int token = batch_tile + token_in_tile;
        const unsigned int word_in_token = lane & 3u;
        unsigned int b0 = 0u;
        unsigned int b1 = 0u;
        unsigned int scale_b = 0u;
        if (token < batch) {
            const unsigned char* input_block =
                input_mxfp4 + (size_t(token) * input_blocks + ktile) * 36u;
            b0 = load_u32_unaligned(input_block + 4u + word_in_token * 4u);
            b1 = load_u32_unaligned(input_block + 4u + (word_in_token + 4u) * 4u);
            scale_b = load_u32_unaligned(input_block);
        }

        const unsigned int scale_row_in_tile = (lane / 4u) + ((lane & 1u) * 8u);
        const unsigned int scale_row = row_block + scale_row_in_tile;
        unsigned int scale_a = 0u;
        if (scale_row < rows) {
            const unsigned char* block0 = mxfp4 + (size_t(scale_row) * blocks_per_row + block_base) * 17u;
            const unsigned char* block1 = block0 + 17u;
            scale_a = unsigned(block0[0]) | (unsigned(block1[0]) << 8);
        }

        asm volatile(
            "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
            "%10, {0, 0}, %11, {0, 0};"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(scale_a), "r"(scale_b));
    }

    const unsigned int row0 = row_block + lane / 4u;
    const unsigned int row1 = row0 + 8u;
    const unsigned int col0 = batch_tile + (lane & 3u) * 2u;
    const unsigned int col1 = col0 + 1u;
    if (row0 < rows && col0 < batch) {
        output[size_t(col0) * rows + row0] = d0 * output_scale;
    }
    if (row0 < rows && col1 < batch) {
        output[size_t(col1) * rows + row0] = d1 * output_scale;
    }
    if (row1 < rows && col0 < batch) {
        output[size_t(col0) * rows + row1] = d2 * output_scale;
    }
    if (row1 < rows && col1 < batch) {
        output[size_t(col1) * rows + row1] = d3 * output_scale;
    }
#else
    if (lane == 0u) {
        for (unsigned int token = batch_tile; token < batch && token < batch_tile + 8u; ++token) {
            for (unsigned int row = row_block; row < rows && row < row_block + 16u; ++row) {
                output[size_t(token) * rows + row] = 0.0f;
            }
        }
    }
#endif
}

extern "C" __global__ void aegis_mxfp4_matmul_native_tile_m16n64(
    const unsigned char* mxfp4,
    const unsigned char* input_mxfp4,
    const unsigned int rows,
    const unsigned int cols,
    const unsigned int blocks_per_row,
    const unsigned int batch,
    const float output_scale,
    float* output
) {
    __shared__ __align__(16) int a_shared[8 * 16 * 8];
    mxfp4_matmul_tile_m16n64_core(
        mxfp4,
        input_mxfp4,
        rows,
        cols,
        blocks_per_row,
        batch,
        output_scale,
        output,
        a_shared
    );
}

extern "C" __device__ __forceinline__ void mxfp4_matmul_tile_m16n32_core(
    const unsigned char* mxfp4,
    const unsigned char* input_mxfp4,
    const unsigned int rows,
    const unsigned int cols,
    const unsigned int blocks_per_row,
    const unsigned int batch,
    const float output_scale,
    float* output,
    int* a_shared
) {
    const unsigned int warp = threadIdx.x >> 5u;
    const unsigned int lane = threadIdx.x & 31u;
    if (warp >= 4u) {
        return;
    }
    const unsigned int row_subtile = warp & 1u;
    const unsigned int token_subtile = warp >> 1u;
    const unsigned int row_block = blockIdx.x * 32u + row_subtile * 16u;
    const unsigned int batch_tile = blockIdx.y * 16u + token_subtile * 8u;
    const unsigned int input_blocks = cols / 64u;

#if __CUDA_ARCH__ >= 1200
    float d0 = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    float d3 = 0.0f;
    const unsigned int k_tiles = cols / 64u;
    int* warp_a_shared = a_shared + warp * 16u * 8u;

    for (unsigned int ktile = 0u; ktile < k_tiles; ++ktile) {
        const unsigned int block_base = ktile * 2u;

        for (unsigned int idx = lane; idx < 16u * 8u; idx += 32u) {
            const unsigned int row_in_tile = idx / 8u;
            const unsigned int physical_word = idx & 7u;
            const unsigned int row = row_block + row_in_tile;
            unsigned int word = 0u;
            if (row < rows) {
                const unsigned int k_block = block_base + physical_word / 4u;
                const unsigned int word_in_block = physical_word & 3u;
                const unsigned char* block = mxfp4 + (size_t(row) * blocks_per_row + k_block) * 17u;
                word = load_u32_unaligned(block + 1u + word_in_block * 4u);
            }
            warp_a_shared[idx] = int(word);
        }
        __syncwarp();

        unsigned int a0 = 0u;
        unsigned int a1 = 0u;
        unsigned int a2 = 0u;
        unsigned int a3 = 0u;
        const int* a_ptr = warp_a_shared + (lane % 16u) * 8u + (lane / 16u) * 4u;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
            : "l"(a_ptr));
        __syncwarp();

        const unsigned int token_in_tile = lane / 4u;
        const unsigned int token = batch_tile + token_in_tile;
        const unsigned int word_in_token = lane & 3u;
        unsigned int b0 = 0u;
        unsigned int b1 = 0u;
        unsigned int scale_b = 0u;
        if (token < batch) {
            const unsigned char* input_block =
                input_mxfp4 + (size_t(token) * input_blocks + ktile) * 36u;
            b0 = load_u32_unaligned(input_block + 4u + word_in_token * 4u);
            b1 = load_u32_unaligned(input_block + 4u + (word_in_token + 4u) * 4u);
            scale_b = load_u32_unaligned(input_block);
        }

        const unsigned int scale_row_in_tile = (lane / 4u) + ((lane & 1u) * 8u);
        const unsigned int scale_row = row_block + scale_row_in_tile;
        unsigned int scale_a = 0u;
        if (scale_row < rows) {
            const unsigned char* block0 = mxfp4 + (size_t(scale_row) * blocks_per_row + block_base) * 17u;
            const unsigned char* block1 = block0 + 17u;
            scale_a = unsigned(block0[0]) | (unsigned(block1[0]) << 8);
        }

        asm volatile(
            "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
            "%10, {0, 0}, %11, {0, 0};"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(scale_a), "r"(scale_b));
    }

    const unsigned int row0 = row_block + lane / 4u;
    const unsigned int row1 = row0 + 8u;
    const unsigned int col0 = batch_tile + (lane & 3u) * 2u;
    const unsigned int col1 = col0 + 1u;
    if (row0 < rows && col0 < batch) {
        output[size_t(col0) * rows + row0] = d0 * output_scale;
    }
    if (row0 < rows && col1 < batch) {
        output[size_t(col1) * rows + row0] = d1 * output_scale;
    }
    if (row1 < rows && col0 < batch) {
        output[size_t(col0) * rows + row1] = d2 * output_scale;
    }
    if (row1 < rows && col1 < batch) {
        output[size_t(col1) * rows + row1] = d3 * output_scale;
    }
#else
    if (lane == 0u) {
        for (unsigned int token = batch_tile; token < batch && token < batch_tile + 8u; ++token) {
            for (unsigned int row = row_block; row < rows && row < row_block + 16u; ++row) {
                output[size_t(token) * rows + row] = 0.0f;
            }
        }
    }
#endif
}

extern "C" __global__ void aegis_mxfp4_matmul_qkv_native_tile_m16n64(
    const unsigned char* q_mxfp4,
    const unsigned char* k_mxfp4,
    const unsigned char* v_mxfp4,
    const unsigned char* input_mxfp4,
    const unsigned int q_rows,
    const unsigned int kv_rows,
    const unsigned int cols,
    const unsigned int q_blocks_per_row,
    const unsigned int k_blocks_per_row,
    const unsigned int v_blocks_per_row,
    const unsigned int batch,
    const float q_output_scale,
    const float k_output_scale,
    const float v_output_scale,
    float* q_output,
    float* k_output,
    float* v_output
) {
    const unsigned int projection = blockIdx.z;
    const unsigned char* selected_mxfp4 = q_mxfp4;
    unsigned int selected_rows = q_rows;
    unsigned int selected_blocks = q_blocks_per_row;
    float selected_scale = q_output_scale;
    float* selected_output = q_output;
    if (projection == 1u) {
        selected_mxfp4 = k_mxfp4;
        selected_rows = kv_rows;
        selected_blocks = k_blocks_per_row;
        selected_scale = k_output_scale;
        selected_output = k_output;
    } else if (projection == 2u) {
        selected_mxfp4 = v_mxfp4;
        selected_rows = kv_rows;
        selected_blocks = v_blocks_per_row;
        selected_scale = v_output_scale;
        selected_output = v_output;
    }
    if (blockIdx.x * 64u >= selected_rows) {
        return;
    }
    __shared__ __align__(16) int a_shared[8 * 16 * 8];
    mxfp4_matmul_tile_m16n64_core(
        selected_mxfp4,
        input_mxfp4,
        selected_rows,
        cols,
        selected_blocks,
        batch,
        selected_scale,
        selected_output,
        a_shared
    );
}

extern "C" __global__ void aegis_mxfp4_matmul_qkv_native_tile_m16n32(
    const unsigned char* q_mxfp4,
    const unsigned char* k_mxfp4,
    const unsigned char* v_mxfp4,
    const unsigned char* input_mxfp4,
    const unsigned int q_rows,
    const unsigned int kv_rows,
    const unsigned int cols,
    const unsigned int q_blocks_per_row,
    const unsigned int k_blocks_per_row,
    const unsigned int v_blocks_per_row,
    const unsigned int batch,
    const float q_output_scale,
    const float k_output_scale,
    const float v_output_scale,
    float* q_output,
    float* k_output,
    float* v_output
) {
    const unsigned int projection = blockIdx.z;
    const unsigned char* selected_mxfp4 = q_mxfp4;
    unsigned int selected_rows = q_rows;
    unsigned int selected_blocks = q_blocks_per_row;
    float selected_scale = q_output_scale;
    float* selected_output = q_output;
    if (projection == 1u) {
        selected_mxfp4 = k_mxfp4;
        selected_rows = kv_rows;
        selected_blocks = k_blocks_per_row;
        selected_scale = k_output_scale;
        selected_output = k_output;
    } else if (projection == 2u) {
        selected_mxfp4 = v_mxfp4;
        selected_rows = kv_rows;
        selected_blocks = v_blocks_per_row;
        selected_scale = v_output_scale;
        selected_output = v_output;
    }
    if (blockIdx.x * 32u >= selected_rows) {
        return;
    }
    __shared__ __align__(16) int a_shared[4 * 16 * 8];
    mxfp4_matmul_tile_m16n32_core(
        selected_mxfp4,
        input_mxfp4,
        selected_rows,
        cols,
        selected_blocks,
        batch,
        selected_scale,
        selected_output,
        a_shared
    );
}

extern "C" __global__ void aegis_mxfp4_matmul_gate_up_native_tile_m16n64(
    const unsigned char* gate_mxfp4,
    const unsigned char* up_mxfp4,
    const unsigned char* input_mxfp4,
    const unsigned int gate_rows,
    const unsigned int up_rows,
    const unsigned int cols,
    const unsigned int gate_blocks_per_row,
    const unsigned int up_blocks_per_row,
    const unsigned int batch,
    const float gate_output_scale,
    const float up_output_scale,
    float* gate_output,
    float* up_output
) {
    const bool up_projection = blockIdx.z != 0u;
    const unsigned int selected_rows = up_projection ? up_rows : gate_rows;
    if (blockIdx.x * 64u >= selected_rows) {
        return;
    }
    __shared__ __align__(16) int a_shared[8 * 16 * 8];
    mxfp4_matmul_tile_m16n64_core(
        up_projection ? up_mxfp4 : gate_mxfp4,
        input_mxfp4,
        selected_rows,
        cols,
        up_projection ? up_blocks_per_row : gate_blocks_per_row,
        batch,
        up_projection ? up_output_scale : gate_output_scale,
        up_projection ? up_output : gate_output,
        a_shared
    );
}

extern "C" __global__ void aegis_mxfp4_matmul_gate_up_native_tile_m16n32(
    const unsigned char* gate_mxfp4,
    const unsigned char* up_mxfp4,
    const unsigned char* input_mxfp4,
    const unsigned int gate_rows,
    const unsigned int up_rows,
    const unsigned int cols,
    const unsigned int gate_blocks_per_row,
    const unsigned int up_blocks_per_row,
    const unsigned int batch,
    const float gate_output_scale,
    const float up_output_scale,
    float* gate_output,
    float* up_output
) {
    const bool up_projection = blockIdx.z != 0u;
    const unsigned int selected_rows = up_projection ? up_rows : gate_rows;
    if (blockIdx.x * 32u >= selected_rows) {
        return;
    }
    __shared__ __align__(16) int a_shared[4 * 16 * 8];
    mxfp4_matmul_tile_m16n32_core(
        up_projection ? up_mxfp4 : gate_mxfp4,
        input_mxfp4,
        selected_rows,
        cols,
        up_projection ? up_blocks_per_row : gate_blocks_per_row,
        batch,
        up_projection ? up_output_scale : gate_output_scale,
        up_projection ? up_output : gate_output,
        a_shared
    );
}

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

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>

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
// ---------------------------------------------------------------------------
// FP8 E4M3 signed conversion (NVIDIA convention: NaN=0x7f/0xff, max=448=0x7e)
// ---------------------------------------------------------------------------
//
// float_to_fp8_e4m3_bits: signed float → 8-bit E4M3 (1 sign, 4 exp, 3 mant)
//   Uses fp32_to_ue4m3_halfbits for the magnitude (already handles subnormals
//   and clamping to [0, 448]); prepends the sign bit.
extern "C" __device__ __forceinline__ unsigned char float_to_fp8_e4m3_bits(float x) {
    const unsigned int bits = __float_as_uint(x);
    const unsigned int sign = (bits >> 31u) & 1u;
    const float abs_val = __uint_as_float(bits & 0x7FFFFFFFu);
    const unsigned int mag = fp32_to_ue4m3_halfbits(abs_val);
    return (unsigned char)((sign << 7u) | mag);
}

// fp8_e4m3_bits_to_float: 8-bit E4M3 → float
//   0x7f / 0xff are NaN.
//   exp=0 → subnormal: mantissa * 2^(-9).
//   exp in [1,15] → normal: (1 + mantissa/8) * 2^(exp-7), max = 1.75*256 = 448.
extern "C" __device__ __forceinline__ float fp8_e4m3_bits_to_float(unsigned char x) {
    const unsigned int sign = (unsigned int)((x >> 7u) & 1u);
    const unsigned int mag  = (unsigned int)(x & 0x7Fu);
    if (mag == 0x7Fu) { return __uint_as_float(0x7FC00000u); } /* NaN */
    if (mag == 0u)    { return 0.0f; }
    const unsigned int exp_fp8 = (mag >> 3u) & 0xFu;
    const unsigned int mantissa = mag & 0x7u;
    float abs_val;
    if (exp_fp8 == 0u) {
        abs_val = (float)mantissa * 0.001953125f; /* mantissa * 2^(-9) */
    } else {
        abs_val = (1.0f + (float)mantissa * 0.125f) * exp2f((float)exp_fp8 - 7.0f);
    }
    return sign ? -abs_val : abs_val;
}
// ---------------------------------------------------------------------------

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

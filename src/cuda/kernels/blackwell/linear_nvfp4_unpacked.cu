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

extern "C" __global__ void aegis_split_qkv_scaled(
    const float* __restrict__ qkv,
    const unsigned int batch,
    const unsigned int q_rows,
    const unsigned int kv_rows,
    const float q_output_scale,
    const float k_output_scale,
    const float v_output_scale,
    float* __restrict__ q_output,
    float* __restrict__ k_output,
    float* __restrict__ v_output
) {
    const unsigned int qkv_rows = q_rows + kv_rows + kv_rows;
    const size_t total = size_t(batch) * qkv_rows;
    for (size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += size_t(blockDim.x) * gridDim.x) {
        const unsigned int row = unsigned(idx % qkv_rows);
        const unsigned int token = unsigned(idx / qkv_rows);
        const float value = qkv[idx];
        if (row < q_rows) {
            q_output[size_t(token) * q_rows + row] = value * q_output_scale;
        } else if (row < q_rows + kv_rows) {
            const unsigned int k_row = row - q_rows;
            k_output[size_t(token) * kv_rows + k_row] = value * k_output_scale;
        } else {
            const unsigned int v_row = row - q_rows - kv_rows;
            v_output[size_t(token) * kv_rows + v_row] = value * v_output_scale;
        }
    }
}

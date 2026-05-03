}

extern "C" __device__ __forceinline__ float bf16_to_float(unsigned short value) {
    return __uint_as_float(((unsigned int)value) << 16);
}

extern "C" __global__ void aegis_bf16_row_to_f32(
    const unsigned short* matrix,
    const unsigned int row,
    const unsigned int cols,
    float* output
) {
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        output[col] = bf16_to_float(matrix[size_t(row) * cols + col]);
    }
}

extern "C" __global__ void aegis_bf16_rows_to_f32(
    const unsigned short* matrix,
    const unsigned int* rows,
    const unsigned int batch,
    const unsigned int rows_total,
    const unsigned int cols,
    float* output
) {
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int batch_idx = blockIdx.y;
    if (batch_idx < batch && col < cols) {
        const unsigned int row = rows[batch_idx];
        output[size_t(batch_idx) * cols + col] =
            (row < rows_total) ? bf16_to_float(matrix[size_t(row) * cols + col]) : 0.0f;
    }
}

extern "C" __global__ void aegis_rms_norm(
    const float* input,
    const float* weight,
    const unsigned int len,
    const float eps,
    float* output
) {
    const unsigned int tid = threadIdx.x;
    extern __shared__ float partial[];
    float sum = 0.0f;
    for (unsigned int i = tid; i < len; i += blockDim.x) {
        const float value = input[i];
        sum += value * value;
    }
    partial[tid] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / float(len) + eps);
    for (unsigned int i = tid; i < len; i += blockDim.x) {
        output[i] = input[i] * scale * weight[i];
    }
}

extern "C" __global__ void aegis_rms_norm_batched(
    const float* input,
    const float* weight,
    const unsigned int batch,
    const unsigned int len,
    const float eps,
    float* output
) {
    const unsigned int batch_idx = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (batch_idx >= batch) {
        return;
    }
    const float* input_row = input + size_t(batch_idx) * len;
    float* output_row = output + size_t(batch_idx) * len;
    extern __shared__ float partial[];
    float sum = 0.0f;
    for (unsigned int i = tid; i < len; i += blockDim.x) {
        const float value = input_row[i];
        sum += value * value;
    }
    partial[tid] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / float(len) + eps);
    for (unsigned int i = tid; i < len; i += blockDim.x) {
        output_row[i] = input_row[i] * scale * weight[i];
    }
}

extern "C" __global__ void aegis_rms_norm_quant_nvfp4(
    const float* input,
    const float* weight,
    const unsigned int len,
    const float eps,
    const float input_scale,
    float* normed_output,
    float* quantized_output
) {
    const unsigned int tid = threadIdx.x;
    extern __shared__ float partial[];
    float sum = 0.0f;
    for (unsigned int i = tid; i < len; i += blockDim.x) {
        const float value = input[i];
        sum += value * value;
    }
    partial[tid] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    const float norm_scale = rsqrtf(partial[0] / float(len) + eps);

    const unsigned int groups = (len + 15u) / 16u;
    if (!(input_scale > 0.0f)) {
        for (unsigned int i = tid; i < len; i += blockDim.x) {
            const float normed = input[i] * norm_scale * weight[i];
            normed_output[i] = normed;
            quantized_output[i] = normed;
        }
        return;
    }

    const float inv = 1.0f / input_scale;
    for (unsigned int group = tid; group < groups; group += blockDim.x) {
        const unsigned int base = group * 16u;
        float values[16];
        float amax = 0.0f;
        #pragma unroll
        for (unsigned int lane = 0u; lane < 16u; ++lane) {
            const unsigned int idx = base + lane;
            float normed = 0.0f;
            if (idx < len) {
                normed = input[idx] * norm_scale * weight[idx];
                normed_output[idx] = normed;
                amax = fmaxf(amax, fabsf(normed * inv));
            }
            values[lane] = normed;
        }
        if (amax == 0.0f) {
            #pragma unroll
            for (unsigned int lane = 0u; lane < 16u; ++lane) {
                const unsigned int idx = base + lane;
                if (idx < len) {
                    quantized_output[idx] = 0.0f;
                }
            }
            continue;
        }
        const float block_scale = decode_ue4m3_half(fp32_to_ue4m3_halfbits(amax / 6.0f));
        #pragma unroll
        for (unsigned int lane = 0u; lane < 16u; ++lane) {
            const unsigned int idx = base + lane;
            if (idx < len) {
                const unsigned int nibble = best_nvfp4_index(values[lane] * inv, block_scale);
                quantized_output[idx] = float(decode_nvfp4_nibble(nibble)) * block_scale * input_scale;
            }
        }
    }
}

extern "C" __global__ void aegis_rms_norm_quant_nvfp4_batched(
    const float* input,
    const float* weight,
    const unsigned int batch,
    const unsigned int len,
    const float eps,
    const float input_scale,
    float* normed_output,
    float* quantized_output
) {
    const unsigned int batch_idx = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (batch_idx >= batch) {
        return;
    }
    const float* input_row = input + size_t(batch_idx) * len;
    float* normed_row = normed_output + size_t(batch_idx) * len;
    float* quantized_row = quantized_output + size_t(batch_idx) * len;
    extern __shared__ float partial[];
    float sum = 0.0f;
    for (unsigned int i = tid; i < len; i += blockDim.x) {
        const float value = input_row[i];
        sum += value * value;
    }
    partial[tid] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    const float norm_scale = rsqrtf(partial[0] / float(len) + eps);

    const unsigned int groups = (len + 15u) / 16u;
    if (!(input_scale > 0.0f)) {
        for (unsigned int i = tid; i < len; i += blockDim.x) {
            const float normed = input_row[i] * norm_scale * weight[i];
            normed_row[i] = normed;
            quantized_row[i] = normed;
        }
        return;
    }

    const float inv = 1.0f / input_scale;
    for (unsigned int group = tid; group < groups; group += blockDim.x) {
        const unsigned int base = group * 16u;
        float values[16];
        float amax = 0.0f;
        #pragma unroll
        for (unsigned int lane = 0u; lane < 16u; ++lane) {
            const unsigned int idx = base + lane;
            float normed = 0.0f;
            if (idx < len) {
                normed = input_row[idx] * norm_scale * weight[idx];
                normed_row[idx] = normed;
                amax = fmaxf(amax, fabsf(normed * inv));
            }
            values[lane] = normed;
        }
        if (amax == 0.0f) {
            #pragma unroll
            for (unsigned int lane = 0u; lane < 16u; ++lane) {
                const unsigned int idx = base + lane;
                if (idx < len) {
                    quantized_row[idx] = 0.0f;
                }
            }
            continue;
        }
        const float block_scale = decode_ue4m3_half(fp32_to_ue4m3_halfbits(amax / 6.0f));
        #pragma unroll
        for (unsigned int lane = 0u; lane < 16u; ++lane) {
            const unsigned int idx = base + lane;
            if (idx < len) {
                const unsigned int nibble = best_nvfp4_index(values[lane] * inv, block_scale);
                quantized_row[idx] = float(decode_nvfp4_nibble(nibble)) * block_scale * input_scale;
            }
        }
    }
}

extern "C" __global__ void aegis_vector_add(
    const float* a,
    const float* b,
    const unsigned int len,
    float* output
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        output[idx] = a[idx] + b[idx];
    }
}

extern "C" __global__ void aegis_vector_add_inplace(
    float* a,
    const float* b,
    const unsigned int len
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        a[idx] += b[idx];
    }
}

extern "C" __global__ void aegis_swiglu(
    const float* gate,
    const float* up,
    const unsigned int len,
    float* output
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        const float x = gate[idx];
        output[idx] = (x / (1.0f + expf(-x))) * up[idx];
    }
}

extern "C" __global__ void aegis_swiglu_inplace_gate(
    float* gate,
    const float* up,
    const unsigned int len
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        const float x = gate[idx];
        gate[idx] = (x / (1.0f + expf(-x))) * up[idx];
    }
}

extern "C" __device__ __forceinline__ float rope_inv_freq_device(
    const unsigned int index,
    const unsigned int head_dim,
    const float theta,
    const float factor,
    const float low_freq_factor,
    const float high_freq_factor,
    const float original_max_position_embeddings
) {
    const float freq = 1.0f / powf(theta, float(index * 2u) / float(head_dim));
    if (factor == 1.0f) {
        return freq;
    }
    const float wavelength = 6.283185307179586f / fmaxf(freq, 1.0e-12f);
    if (wavelength > original_max_position_embeddings / low_freq_factor) {
        return freq / factor;
    }
    if (wavelength < original_max_position_embeddings / high_freq_factor
        || fabsf(high_freq_factor - low_freq_factor) < 1.0e-12f) {
        return freq;
    }
    float smooth = ((original_max_position_embeddings / wavelength) - low_freq_factor)
        / (high_freq_factor - low_freq_factor);
    smooth = fminf(fmaxf(smooth, 0.0f), 1.0f);
    return (1.0f - smooth) * (freq / factor) + smooth * freq;
}

// Pointer-based variant for CUDA Graph replay: position is read from device memory.
extern "C" __global__ void aegis_apply_rope_ptr(
    float* values,
    const unsigned int* p_position,
    const unsigned int num_heads,
    const unsigned int head_dim,
    const float theta,
    const float factor,
    const float low_freq_factor,
    const float high_freq_factor,
    const unsigned int original_max_position_embeddings,
    const unsigned int partial_dim  /* 0 = full head_dim; >0 = first N dims rotated (p-RoPE) */
) {
    const unsigned int position = *p_position;
    const unsigned int head = blockIdx.x;
    const unsigned int i = threadIdx.x;
    const unsigned int half_dim = head_dim / 2u;
    const unsigned int partial_half = (partial_dim > 0u) ? partial_dim / 2u : half_dim;
    if (head >= num_heads || i >= half_dim) { return; }
    if (i >= partial_half) { return; }
    float* row = values + size_t(head) * head_dim;
    const float angle = float(position) * rope_inv_freq_device(i, head_dim, theta, factor,
        low_freq_factor, high_freq_factor, float(original_max_position_embeddings));
    float sinv, cosv;
    sincosf(angle, &sinv, &cosv);
    const float x0 = row[i], x1 = row[i + half_dim];
    row[i] = x0 * cosv - x1 * sinv;
    row[i + half_dim] = x0 * sinv + x1 * cosv;
}

extern "C" __global__ void aegis_apply_rope(
    float* values,
    const unsigned int position,
    const unsigned int num_heads,
    const unsigned int head_dim,
    const float theta,
    const float factor,
    const float low_freq_factor,
    const float high_freq_factor,
    const unsigned int original_max_position_embeddings
) {
    const unsigned int head = blockIdx.x;
    const unsigned int i = threadIdx.x;
    const unsigned int half_dim = head_dim / 2u;
    if (head >= num_heads || i >= half_dim) {
        return;
    }
    float* row = values + size_t(head) * head_dim;
    const float angle = float(position) * rope_inv_freq_device(
        i,
        head_dim,
        theta,
        factor,
        low_freq_factor,
        high_freq_factor,
        float(original_max_position_embeddings)
    );
    float sinv;
    float cosv;
    sincosf(angle, &sinv, &cosv);
    const float x0 = row[i];
    const float x1 = row[i + half_dim];
    row[i] = x0 * cosv - x1 * sinv;
    row[i + half_dim] = x0 * sinv + x1 * cosv;
}

extern "C" __global__ void aegis_apply_rope_batched(
    float* values,
    const unsigned int start_position,
    const unsigned int batch,
    const unsigned int num_heads,
    const unsigned int head_dim,
    const float theta,
    const float factor,
    const float low_freq_factor,
    const float high_freq_factor,
    const unsigned int original_max_position_embeddings
) {
    const unsigned int head = blockIdx.x;
    const unsigned int batch_idx = blockIdx.y;
    const unsigned int i = threadIdx.x;
    const unsigned int half_dim = head_dim / 2u;
    if (batch_idx >= batch || head >= num_heads || i >= half_dim) {
        return;
    }
    float* row = values + (size_t(batch_idx) * num_heads + head) * head_dim;
    const float angle = float(start_position + batch_idx) * rope_inv_freq_device(
        i,
        head_dim,
        theta,
        factor,
        low_freq_factor,
        high_freq_factor,
        float(original_max_position_embeddings)
    );
    float sinv;
    float cosv;
    sincosf(angle, &sinv, &cosv);
    const float x0 = row[i];
    const float x1 = row[i + half_dim];
    row[i] = x0 * cosv - x1 * sinv;
    row[i + half_dim] = x0 * sinv + x1 * cosv;
}

extern "C" __global__ void aegis_apply_rope_positions_batched(
    float* values,
    const unsigned int* positions,
    const unsigned int batch,
    const unsigned int num_heads,
    const unsigned int head_dim,
    const float theta,
    const float factor,
    const float low_freq_factor,
    const float high_freq_factor,
    const unsigned int original_max_position_embeddings
) {
    const unsigned int head = blockIdx.x;
    const unsigned int batch_idx = blockIdx.y;
    const unsigned int i = threadIdx.x;
    const unsigned int half_dim = head_dim / 2u;
    if (batch_idx >= batch || head >= num_heads || i >= half_dim) {
        return;
    }
    float* row = values + (size_t(batch_idx) * num_heads + head) * head_dim;
    const float angle = float(positions[batch_idx]) * rope_inv_freq_device(
        i,
        head_dim,
        theta,
        factor,
        low_freq_factor,
        high_freq_factor,
        float(original_max_position_embeddings)
    );
    float sinv;
    float cosv;
    sincosf(angle, &sinv, &cosv);
    const float x0 = row[i];
    const float x1 = row[i + half_dim];
    row[i] = x0 * cosv - x1 * sinv;
    row[i + half_dim] = x0 * sinv + x1 * cosv;
}

extern "C" __global__ void aegis_build_dense_prefill_metadata(
    const unsigned int start_position,
    const unsigned int batch,
    unsigned int* positions,
    unsigned int* slot_mapping
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch) {
        const unsigned int position = start_position + idx;
        positions[idx] = position;
        slot_mapping[idx] = position;
    }
}

extern "C" __device__ __forceinline__ unsigned short float_to_f16_bits(float value) {
    const unsigned int bits = __float_as_uint(value);
    const unsigned int sign = (bits >> 16) & 0x8000u;
    int exp = int((bits >> 23) & 0xffu) - 127 + 15;
    unsigned int mant = bits & 0x007fffffu;

    if (exp <= 0) {
        if (exp < -10) {
            return (unsigned short)sign;
        }
        mant |= 0x00800000u;
        const unsigned int shift = (unsigned int)(14 - exp);
        const unsigned int rounded = (mant + (1u << (shift - 1u)) - 1u + ((mant >> shift) & 1u)) >> shift;
        return (unsigned short)(sign | rounded);
    }
    if (exp >= 31) {
        return (unsigned short)(sign | 0x7c00u);
    }

    mant += 0x00001000u;
    if (mant & 0x00800000u) {
        mant = 0u;
        exp += 1;
        if (exp >= 31) {
            return (unsigned short)(sign | 0x7c00u);
        }
    }
    return (unsigned short)(sign | ((unsigned int)exp << 10) | (mant >> 13));
}

extern "C" __device__ __forceinline__ float f16_bits_to_float(unsigned short value) {
    return __half2float(__ushort_as_half(value));
}

extern "C" __global__ void aegis_f32_to_f16(
    const float* input,
    const unsigned int len,
    unsigned short* output
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        output[idx] = float_to_f16_bits(input[idx]);
    }
}

extern "C" __global__ void aegis_apply_rope_positions_batched_f16_out(
    float* values,
    const unsigned int* positions,
    const unsigned int batch,
    const unsigned int num_heads,
    const unsigned int head_dim,
    const float theta,
    const float factor,
    const float low_freq_factor,
    const float high_freq_factor,
    const unsigned int original_max_position_embeddings,
    unsigned short* output,
    const unsigned int partial_dim  /* 0 = full head_dim; >0 = first N dims rotated (p-RoPE) */
) {
    const unsigned int head = blockIdx.x;
    const unsigned int batch_idx = blockIdx.y;
    const unsigned int i = threadIdx.x;
    const unsigned int half_dim = head_dim / 2u;
    const unsigned int partial_half = (partial_dim > 0u) ? partial_dim / 2u : half_dim;
    if (batch_idx >= batch || head >= num_heads || i >= half_dim) {
        return;
    }
    float* row = values + (size_t(batch_idx) * num_heads + head) * head_dim;
    unsigned short* out = output + (size_t(batch_idx) * num_heads + head) * head_dim;
    float y0, y1;
    if (i < partial_half) {
        const float angle = float(positions[batch_idx]) * rope_inv_freq_device(
            i, head_dim, theta, factor, low_freq_factor, high_freq_factor,
            float(original_max_position_embeddings));
        float sinv, cosv;
        sincosf(angle, &sinv, &cosv);
        const float x0 = row[i], x1 = row[i + half_dim];
        y0 = x0 * cosv - x1 * sinv;
        y1 = x0 * sinv + x1 * cosv;
    } else {
        y0 = row[i];
        y1 = row[i + half_dim];
    }
    row[i] = y0;
    row[i + half_dim] = y1;
    out[i] = float_to_f16_bits(y0);
    out[i + half_dim] = float_to_f16_bits(y1);
}

// Pointer-based variant for CUDA Graph replay: position is read from device memory.
extern "C" __global__ void aegis_kv_store_ptr(
    unsigned short* key_cache,
    unsigned short* value_cache,
    const float* key,
    const float* value,
    const unsigned int* p_position,
    const unsigned int width
) {
    const unsigned int position = *p_position;
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width) {
        const size_t offset = size_t(position) * width + idx;
        key_cache[offset] = float_to_f16_bits(key[idx]);
        value_cache[offset] = float_to_f16_bits(value[idx]);
    }
}

extern "C" __global__ void aegis_kv_store(
    unsigned short* key_cache,
    unsigned short* value_cache,
    const float* key,
    const float* value,
    const unsigned int position,
    const unsigned int width
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width) {
        const size_t offset = size_t(position) * width + idx;
        key_cache[offset] = float_to_f16_bits(key[idx]);
        value_cache[offset] = float_to_f16_bits(value[idx]);
    }
}

extern "C" __global__ void aegis_kv_store_batched(
    unsigned short* key_cache,
    unsigned short* value_cache,
    const float* key,
    const float* value,
    const unsigned int start_position,
    const unsigned int batch,
    const unsigned int width
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int batch_idx = blockIdx.y;
    if (batch_idx < batch && idx < width) {
        const size_t src = size_t(batch_idx) * width + idx;
        const size_t dst = size_t(start_position + batch_idx) * width + idx;
        key_cache[dst] = float_to_f16_bits(key[src]);
        value_cache[dst] = float_to_f16_bits(value[src]);
    }
}

extern "C" __global__ void aegis_kv_store_slots_batched(
    unsigned short* key_cache,
    unsigned short* value_cache,
    const float* key,
    const float* value,
    const unsigned int* slot_mapping,
    const unsigned int batch,
    const unsigned int width,
    const unsigned int context_size
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int batch_idx = blockIdx.y;
    if (batch_idx < batch && idx < width) {
        const unsigned int slot = slot_mapping[batch_idx];
        if (slot < context_size) {
            const size_t src = size_t(batch_idx) * width + idx;
            const size_t dst = size_t(slot) * width + idx;
            key_cache[dst] = float_to_f16_bits(key[src]);
            value_cache[dst] = float_to_f16_bits(value[src]);
        }
    }
}

extern "C" __global__ void aegis_rope_kv_store_slots_batched(
    unsigned short* key_cache,
    unsigned short* value_cache,
    float* key,
    const float* value,
    const unsigned int* positions,
    const unsigned int* slot_mapping,
    const unsigned int batch,
    const unsigned int num_heads,
    const unsigned int head_dim,
    const unsigned int context_size,
    const float theta,
    const float factor,
    const float low_freq_factor,
    const float high_freq_factor,
    const unsigned int original_max_position_embeddings,
    const unsigned int partial_dim  /* 0 = full head_dim; >0 = first N dims rotated (p-RoPE) */
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int batch_idx = blockIdx.y;
    const unsigned int width = num_heads * head_dim;
    if (batch_idx >= batch || idx >= width) {
        return;
    }
    const unsigned int slot = slot_mapping[batch_idx];
    if (slot >= context_size) {
        return;
    }

    const size_t src_base = size_t(batch_idx) * width;
    const size_t dst_base = size_t(slot) * width;
    const unsigned int dim = idx % head_dim;
    const unsigned int half_dim = head_dim / 2u;
    const unsigned int partial_half = (partial_dim > 0u) ? partial_dim / 2u : half_dim;

    value_cache[dst_base + idx] = float_to_f16_bits(value[src_base + idx]);
    if (dim < half_dim) {
        const unsigned int pair_idx = idx + half_dim;
        if (dim < partial_half) {
            const float angle = float(positions[batch_idx]) * rope_inv_freq_device(
                dim, head_dim, theta, factor, low_freq_factor, high_freq_factor,
                float(original_max_position_embeddings));
            float sinv, cosv;
            sincosf(angle, &sinv, &cosv);
            const float x0 = key[src_base + idx];
            const float x1 = key[src_base + pair_idx];
            const float y0 = x0 * cosv - x1 * sinv;
            const float y1 = x0 * sinv + x1 * cosv;
            key[src_base + idx] = y0;
            key[src_base + pair_idx] = y1;
            key_cache[dst_base + idx] = float_to_f16_bits(y0);
            key_cache[dst_base + pair_idx] = float_to_f16_bits(y1);
        } else {
            key_cache[dst_base + idx] = float_to_f16_bits(key[src_base + idx]);
            key_cache[dst_base + pair_idx] = float_to_f16_bits(key[src_base + pair_idx]);
        }
    }
}

// AXPY: out[i] += alpha * src[i]   (used for MoE weighted expert accumulation)
extern "C" __global__ void aegis_axpy_f32(
    float* out,
    const float* src,
    const float alpha,
    const unsigned int len
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        out[idx] += alpha * src[idx];
    }
}

// Zero a float buffer (used to zero the MoE accumulator before expert dispatch)
extern "C" __global__ void aegis_zero_f32(
    float* out,
    const unsigned int len
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        out[idx] = 0.0f;
    }
}

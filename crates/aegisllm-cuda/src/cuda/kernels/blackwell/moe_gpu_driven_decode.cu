// GPU-driven MoE decode: gather routed experts' NVFP4 weights from
// device-mapped host RAM into a VRAM scratch in ONE launch, then run the
// per-expert GEMVs reading the gathered bytes + device-resident scales.
//
// Why this exists
// ---------------
// The host-streamed decode path (LinearStagingPool) drives a CPU round-trip
// per MoE layer: it memcpy_dtoh's the GPU router top-k, the host parses the
// expert indices, then the host issues top_k×3 per-expert H2D copies + GEMV
// launches. ~89% of decode time is the CPU issuing this launch storm, which
// also makes decode un-graphable (control flow is CPU-data-dependent).
//
// With the expert arena DEVICE-MAPPED (CU_MEMHOSTREGISTER_DEVICEMAP), the GPU
// can read host bytes directly. These kernels keep the top-k indices in GPU
// memory (never dtoh'd for control flow): a single gather kernel reads the
// on-device top-k index buffer, looks up each selected expert's host device
// pointer from per-layer pointer tables, and streams its packed+scales bytes
// into a fixed VRAM scratch slot (coalesced PCIe reads → VRAM, bandwidth
// friendly — NOT zero-copy reads inside the GEMM). The kernel sequence is then
// FIXED (slot k always feeds GEMV k), so the whole decode can be captured in a
// CUDA graph; the replay reads whatever indices are in the top-k buffer that
// token.
//
// Bit-identical to the per-expert staged path: same weights, same NVFP4
// dequant + accumulation order, same per-expert input/output scales and
// routing weight — the scales are merely sourced from device arrays indexed by
// slot instead of launch-time scalar args.


// ── Top-k packed-record layout (matches router_softmax_topk_packed) ─────────
//   packed_topk[2*k]     = expert index (u32)
//   packed_topk[2*k + 1] = bitcast<u32>(routing weight)
// The gather + axpy kernels read these straight from device memory; no host
// parse, no dtoh.

// Gather the selected experts' packed+scales bytes from device-mapped host RAM
// into a contiguous VRAM scratch, and populate the per-slot scale arrays.
//
// Layout of the bulk buffers (contiguous, slot-major; uniform per-expert sizes
// within a layer): for each routed slot k in 0..top_k, three projections in
// order gate, up, down. Slot k's projection p occupies
//   bulk_packed[(k*3 + p) * packed_bytes_per_proj .. +packed_bytes_per_proj]
//   bulk_scales[(k*3 + p) * scale_bytes_per_proj  .. +scale_bytes_per_proj ]
// (packed/scale bytes are uniform across gate/up/down for Gemma-4 experts:
//  gate==up by construction; down differs in `rows`/`cols` but the host caller
//  passes the per-projection byte stride, see below — here we keep it general
//  by taking three independent strides.)
//
// Pointer tables are device arrays of length num_experts holding the
// device-mapped-host base pointer for each expert's projection bytes:
//   gate_packed_ptrs[e], up_packed_ptrs[e], down_packed_ptrs[e]   (uint64)
//   gate_scale_ptrs[e],  up_scale_ptrs[e],  down_scale_ptrs[e]    (uint64)
// Scale-scalar tables are device arrays of length num_experts:
//   gate_in_scale[e], up_in_scale[e], down_in_scale[e]            (float)
//   gate_out_scale[e], up_out_scale[e], down_out_scale[e]         (float)
//
// One CTA per (slot, projection): blockIdx.y = slot k, blockIdx.x = projection
// (0=gate,1=up,2=down). Threads stream the packed bytes (and the much smaller
// scales) as 16-byte (uint4) chunks where aligned, else byte-wise tail.
extern "C" __global__ void aegis_moe_gather_experts(
    const unsigned int* __restrict__ packed_topk,     // [top_k*2] (idx,wbits) on device
    const unsigned int top_k,
    const unsigned int num_experts,
    // packed device pointers (host device-mapped), one per expert per proj
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ down_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const unsigned long long* __restrict__ down_scale_ptrs,
    // per-projection byte strides (uniform across experts within a layer)
    const unsigned int gate_packed_bytes,
    const unsigned int gate_scale_bytes,
    const unsigned int up_packed_bytes,
    const unsigned int up_scale_bytes,
    const unsigned int down_packed_bytes,
    const unsigned int down_scale_bytes,
    // per-expert scalar tables
    const float* __restrict__ gate_in_scale,
    const float* __restrict__ up_in_scale,
    const float* __restrict__ down_in_scale,
    const float* __restrict__ gate_out_scale,
    const float* __restrict__ up_out_scale,
    const float* __restrict__ down_out_scale,
    // destination VRAM scratch + per-slot scale arrays
    unsigned char* __restrict__ bulk_packed,
    unsigned char* __restrict__ bulk_scales,
    float* __restrict__ slot_in_scale,    // [top_k*3]  (gate,up,down per slot)
    float* __restrict__ slot_out_scale    // [top_k*3]
) {
    const unsigned int slot = blockIdx.y;
    const unsigned int proj = blockIdx.x;   // 0=gate, 1=up, 2=down
    if (slot >= top_k || proj >= 3u) return;

    const unsigned int expert = packed_topk[slot * 2u];
    if (expert >= num_experts) return;

    // Resolve this projection's source pointers + strides + scales.
    unsigned long long src_packed_u64;
    unsigned long long src_scale_u64;
    unsigned int packed_bytes;
    unsigned int scale_bytes;
    float in_s;
    float out_s;
    // Slot-major byte offset within the bulk buffers. Slot k holds gate, up,
    // down back-to-back, each at its own (uniform) stride.
    unsigned int packed_dst_off;
    unsigned int scale_dst_off;
    {
        const unsigned int gate_packed = gate_packed_bytes;
        const unsigned int up_packed   = up_packed_bytes;
        const unsigned int down_packed = down_packed_bytes;
        const unsigned int gate_scale  = gate_scale_bytes;
        const unsigned int up_scale    = up_scale_bytes;
        const unsigned int down_scale  = down_scale_bytes;
        const unsigned int per_slot_packed = gate_packed + up_packed + down_packed;
        const unsigned int per_slot_scale  = gate_scale + up_scale + down_scale;
        const unsigned int slot_packed_base = slot * per_slot_packed;
        const unsigned int slot_scale_base  = slot * per_slot_scale;
        if (proj == 0u) {
            src_packed_u64 = gate_packed_ptrs[expert];
            src_scale_u64  = gate_scale_ptrs[expert];
            packed_bytes = gate_packed; scale_bytes = gate_scale;
            in_s = gate_in_scale[expert]; out_s = gate_out_scale[expert];
            packed_dst_off = slot_packed_base;
            scale_dst_off  = slot_scale_base;
        } else if (proj == 1u) {
            src_packed_u64 = up_packed_ptrs[expert];
            src_scale_u64  = up_scale_ptrs[expert];
            packed_bytes = up_packed; scale_bytes = up_scale;
            in_s = up_in_scale[expert]; out_s = up_out_scale[expert];
            packed_dst_off = slot_packed_base + gate_packed;
            scale_dst_off  = slot_scale_base + gate_scale;
        } else {
            src_packed_u64 = down_packed_ptrs[expert];
            src_scale_u64  = down_scale_ptrs[expert];
            packed_bytes = down_packed; scale_bytes = down_scale;
            in_s = down_in_scale[expert]; out_s = down_out_scale[expert];
            packed_dst_off = slot_packed_base + gate_packed + up_packed;
            scale_dst_off  = slot_scale_base + gate_scale + up_scale;
        }
    }

    const unsigned char* src_packed = reinterpret_cast<const unsigned char*>(src_packed_u64);
    const unsigned char* src_scale  = reinterpret_cast<const unsigned char*>(src_scale_u64);
    unsigned char* dst_packed = bulk_packed + packed_dst_off;
    unsigned char* dst_scale  = bulk_scales + scale_dst_off;

    const unsigned int tid = threadIdx.x;
    const unsigned int nthreads = blockDim.x;

    // Copy packed bytes. Use uint4 (16B) chunks where both src and dst are
    // 16B-aligned (NVFP4 packed rows and the VRAM bulk slots are well aligned),
    // else fall back to per-byte. The source is mapped-host: coalesced reads
    // over PCIe land in VRAM — bandwidth-friendly streaming.
    {
        const bool aligned16 =
            ((reinterpret_cast<unsigned long long>(src_packed) & 0xF) == 0) &&
            ((reinterpret_cast<unsigned long long>(dst_packed) & 0xF) == 0);
        if (aligned16) {
            const unsigned int n16 = packed_bytes / 16u;
            const uint4* s4 = reinterpret_cast<const uint4*>(src_packed);
            uint4* d4 = reinterpret_cast<uint4*>(dst_packed);
            for (unsigned int i = tid; i < n16; i += nthreads) {
                d4[i] = s4[i];
            }
            for (unsigned int i = n16 * 16u + tid; i < packed_bytes; i += nthreads) {
                dst_packed[i] = src_packed[i];
            }
        } else {
            for (unsigned int i = tid; i < packed_bytes; i += nthreads) {
                dst_packed[i] = src_packed[i];
            }
        }
    }
    // Copy scale bytes (small).
    for (unsigned int i = tid; i < scale_bytes; i += nthreads) {
        dst_scale[i] = src_scale[i];
    }

    // First thread records the per-slot scales for the GEMV/quantize kernels.
    if (tid == 0u) {
        const unsigned int slot_proj = slot * 3u + proj;
        slot_in_scale[slot_proj] = in_s;
        slot_out_scale[slot_proj] = out_s;
    }
}

// ── Device-scalar quantize ──────────────────────────────────────────────────
// Same math as aegis_nvfp4_quantize_input but reads `input_scale` from a device
// float pointer (slot_in_scale + slot_proj) so the value is not baked into the
// launch (graph-capturable; the captured launch reads whatever the gather wrote
// this token).
extern "C" __global__ void aegis_nvfp4_quantize_input_dptr(
    const float* __restrict__ input,
    const unsigned int len,
    const float* __restrict__ input_scale_ptr,
    float* __restrict__ output
) {
    const unsigned int base = blockIdx.x * 16u;
    const unsigned int lane = threadIdx.x;
    if (lane >= 16u || base + lane >= len) {
        return;
    }
    const float input_scale = *input_scale_ptr;
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

// ── Device-scalar prequantized GEMV ─────────────────────────────────────────
// Same math as aegis_nvfp4_linear_prequantized but reads `output_scale` from a
// device float pointer (slot_out_scale + slot_proj). packed/scales are views
// into the gathered bulk buffer (passed as device pointers + base offsets).
extern "C" __global__ void aegis_nvfp4_linear_prequantized_dptr(
    const unsigned char* __restrict__ packed,
    const unsigned char* __restrict__ scales,
    const float* __restrict__ input,
    const unsigned int rows,
    const unsigned int cols,
    const float* __restrict__ output_scale_ptr,
    float* __restrict__ output
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
        output[row] = partial[0] * (*output_scale_ptr);
    }
}

// ── axpy with routing weight read from the device top-k buffer ───────────────
// out[i] += weight_k * src[i], where weight_k = bitcast<float>(packed_topk[2*slot+1]).
// Keeps the routing weight on-device so the accumulation is graph-capturable.
extern "C" __global__ void aegis_axpy_f32_topk_weight(
    float* __restrict__ out,
    const float* __restrict__ src,
    const unsigned int* __restrict__ packed_topk,
    const unsigned int slot,
    const unsigned int len
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        const float alpha = __uint_as_float(packed_topk[slot * 2u + 1u]);
        out[idx] += alpha * src[idx];
    }
}

// ════════════════════════════════════════════════════════════════════════════
// BATCHED (grouped-over-experts) decode MoE kernels. The per-slot path above
// issues top_k×[3 quant + 3 GEMV + 1 geglu + 1 axpy] tiny serial launches per
// MoE layer; these collapse each STAGE into ONE launch with the expert/slot on
// grid.y (0..top_k). Per-(slot,row,element) scalar math is byte-identical to the
// single-expert kernels above, so output stays bit-identical; only the Rust
// for-loop moves onto a grid axis. All shapes are fixed → graph-capturable.
// Opt-in via AEGIS_BATCHED_DECODE_MOE (executor/mlp.rs).
// ════════════════════════════════════════════════════════════════════════════

// 1. BATCHED INPUT-QUANT over top_k slots. Identical 16-group math to
// aegis_nvfp4_quantize_input_dptr. `input_stride`=0 for gate/up (every slot
// quantizes the SAME shared `hidden` vector, each with its own scale) or =the
// per-slot swiglu stride for down. `output` is per-slot at `output_stride`.
// Scale = slot_in_scale[slot*3 + proj_off].
extern "C" __global__ void aegis_nvfp4_quantize_input_batched_dptr(
    const float* __restrict__ input,
    const unsigned int input_stride,
    const unsigned int len,
    const unsigned int output_stride,
    const float* __restrict__ slot_in_scale,
    const unsigned int proj_off,
    const unsigned int top_k,
    float* __restrict__ output
) {
    const unsigned int slot = blockIdx.y;
    if (slot >= top_k) {
        return;
    }
    const unsigned int base = blockIdx.x * 16u;
    const unsigned int lane = threadIdx.x;
    if (lane >= 16u || base + lane >= len) {
        return;
    }
    const float* in = input + size_t(slot) * input_stride;
    float* out = output + size_t(slot) * output_stride;
    const float input_scale = slot_in_scale[slot * 3u + proj_off];
    if (!(input_scale > 0.0f)) {
        out[base + lane] = in[base + lane];
        return;
    }
    const float inv = 1.0f / input_scale;
    float amax = 0.0f;
    for (unsigned int j = 0u; j < 16u && base + j < len; ++j) {
        amax = fmaxf(amax, fabsf(in[base + j] * inv));
    }
    if (amax == 0.0f) {
        out[base + lane] = 0.0f;
        return;
    }
    const float block_scale = decode_ue4m3_half(fp32_to_ue4m3_halfbits(amax / 6.0f));
    const unsigned int nibble = best_nvfp4_index(in[base + lane] * inv, block_scale);
    out[base + lane] = float(decode_nvfp4_nibble(nibble)) * block_scale * input_scale;
}

// 2. BATCHED PREQUANTIZED GEMV over top_k slots. Identical accumulation +
// tree-reduction math to aegis_nvfp4_linear_prequantized_dptr. The single-expert
// kernel got PRE-SLICED packed/scales views; this one derives the slot/proj base
// itself from the slot-major bulk layout (slot k = [gate,up,down] back-to-back),
// so it is byte-identical on the same input bytes. Output per (slot,row).
extern "C" __global__ void aegis_nvfp4_linear_prequantized_batched_dptr(
    const unsigned char* __restrict__ bulk_packed,
    const unsigned char* __restrict__ bulk_scales,
    const unsigned int per_slot_packed,
    const unsigned int per_slot_scale,
    const unsigned int proj_packed_off,
    const unsigned int proj_scale_off,
    const float* __restrict__ input,
    const unsigned int input_stride,
    const unsigned int rows,
    const unsigned int cols,
    const float* __restrict__ slot_out_scale,
    const unsigned int proj_off,
    const unsigned int output_stride,
    const unsigned int top_k,
    float* __restrict__ output
) {
    const unsigned int slot = blockIdx.y;
    if (slot >= top_k) {
        return;
    }
    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }
    const unsigned int packed_cols = cols / 2u;
    const unsigned int scale_cols = cols / 16u;
    const unsigned char* packed = bulk_packed + size_t(slot) * per_slot_packed + proj_packed_off;
    const unsigned char* scales = bulk_scales + size_t(slot) * per_slot_scale + proj_scale_off;
    const unsigned char* packed_row = packed + size_t(row) * packed_cols;
    const unsigned char* scale_row = scales + size_t(row) * scale_cols;
    const float* in = input + size_t(slot) * input_stride;
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
            sum += float(decode_nvfp4_nibble(byte & 0x0Fu)) * block_scale * in[lo_col];
            sum += float(decode_nvfp4_nibble(byte >> 4)) * block_scale * in[hi_col];
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
        output[size_t(slot) * output_stride + row] = partial[0] * slot_out_scale[slot * 3u + proj_off];
    }
}

// 2b. BATCHED prequantized GEMV — FAST warp-per-row variant (mmvq-style).
// Same ABI + slot-major bulk layout as aegis_nvfp4_linear_prequantized_batched_dptr,
// but ONE WARP owns one (slot,row) output: 8 warps/block emit 8 rows, the 32 lanes
// stride over the cols/16 NVFP4 groups, and the cross-lane reduction is a 5-step
// warp-shuffle butterfly — ZERO __syncthreads, ZERO shared memory. f32-accurate
// (numerically equivalent to the block-per-row kernel; only the reduction order
// differs). grid=(ceil(rows/8), top_k, 1) block=(32,8,1) shmem=0.
extern "C" __global__ void aegis_nvfp4_linear_prequantized_batched_dptr_warp(
    const unsigned char* __restrict__ bulk_packed,
    const unsigned char* __restrict__ bulk_scales,
    const unsigned int per_slot_packed,
    const unsigned int per_slot_scale,
    const unsigned int proj_packed_off,
    const unsigned int proj_scale_off,
    const float* __restrict__ input,
    const unsigned int input_stride,
    const unsigned int rows,
    const unsigned int cols,
    const float* __restrict__ slot_out_scale,
    const unsigned int proj_off,
    const unsigned int output_stride,
    const unsigned int top_k,
    float* __restrict__ output
) {
    // block=(128,1,1) grid=(rows, top_k, 1): 128 threads (4 warps) per (slot,row),
    // warp-shuffle reduction + single-barrier 4-partial combine (see
    // aegis_nvfp4_gemv_warp). Same coalesced loads + K-parallelism as the naive
    // batched_dptr kernel, minus the 7 extra barriers.
    const unsigned int slot = blockIdx.y;
    if (slot >= top_k) {
        return;
    }
    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;             // 0..127
    if (row >= rows) {
        return;
    }
    const unsigned int packed_cols = cols / 2u;
    const unsigned int scale_cols = cols / 16u;
    const unsigned char* packed = bulk_packed + size_t(slot) * per_slot_packed + proj_packed_off;
    const unsigned char* scales = bulk_scales + size_t(slot) * per_slot_scale + proj_scale_off;
    const unsigned char* packed_row = packed + size_t(row) * packed_cols;
    const unsigned char* scale_row = scales + size_t(row) * scale_cols;
    const float* in = input + size_t(slot) * input_stride;

    float sum = 0.0f;
    for (unsigned int block_idx = tid; block_idx < scale_cols; block_idx += 128u) {
        const float block_scale = decode_ue4m3_half(scale_row[block_idx]);
        const unsigned int input_base = block_idx * 16u;
        const unsigned int packed_base = block_idx * 8u;
        // 128-bit loads (uint2 packed + 4x float4 input), bit-identical to the
        // byte-by-byte loop (same products + order). See aegis_nvfp4_gemv_warp.
        const uint2 pw = *reinterpret_cast<const uint2*>(packed_row + packed_base);
        const float4* in4 = reinterpret_cast<const float4*>(in + input_base);
        const float4 i0 = in4[0]; const float4 i1 = in4[1];
        const float4 i2 = in4[2]; const float4 i3 = in4[3];
        const float in_arr[16] = {
            i0.x, i0.y, i0.z, i0.w, i1.x, i1.y, i1.z, i1.w,
            i2.x, i2.y, i2.z, i2.w, i3.x, i3.y, i3.z, i3.w
        };
        #pragma unroll
        for (unsigned int j = 0u; j < 8u; ++j) {
            const unsigned int byte = (j < 4u) ? ((pw.x >> (j * 8u)) & 0xFFu)
                                               : ((pw.y >> ((j - 4u) * 8u)) & 0xFFu);
            sum += float(decode_nvfp4_nibble(byte & 0x0Fu)) * block_scale * in_arr[2u*j];
            sum += float(decode_nvfp4_nibble(byte >> 4)) * block_scale * in_arr[2u*j + 1u];
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, (unsigned int)offset, 32);
    }
    __shared__ float warp_sums[4];
    const unsigned int warp_id = tid >> 5;
    if ((tid & 31u) == 0u) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();
    if (tid == 0u) {
        output[size_t(slot) * output_stride + row] =
            (warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3]) * slot_out_scale[slot * 3u + proj_off];
    }
}

// 3. BATCHED strided GeGLU over top_k slots. Identical gelu_pytorch_tanh literals
// + op order to aegis_geglu_tanh_batched (sampling.cu), but per-slot strided so it
// works with the over-allocated [top_k * inter_stride] scratch. Renamed to avoid a
// duplicate-symbol collision with sampling.cu's aegis_geglu_tanh_batched (both .cu
// files compile into one NVRTC translation unit).
extern "C" __global__ void aegis_moe_geglu_tanh_batched_slots(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const unsigned int len,
    const unsigned int stride,
    const unsigned int top_k,
    float* __restrict__ output
) {
    const unsigned int slot = blockIdx.y;
    if (slot >= top_k) {
        return;
    }
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        const size_t off = size_t(slot) * stride + idx;
        const float x = gate[off];
        const float k = 0.7978845608028654f;
        const float k2 = 0.044715f;
        const float inner = k * (x + k2 * x * x * x);
        const float gelu = 0.5f * x * (1.0f + tanhf(inner));
        output[off] = gelu * up[off];
    }
}

// 4. BATCHED WEIGHTED ACCUMULATE: out[i] = sum_{k=0..top_k-1} w[k] * expert_out[k][i].
// Replaces the top_k serial aegis_axpy_f32_topk_weight launches. Each thread folds
// the slots in FIXED ascending order with `acc += w * e` as a SINGLE expression —
// under the module's --fmad=true this fuses to one fma per step, matching the
// single-expression axpy (`out[idx] += alpha * src[idx]`) exactly. Do NOT split
// into separate mul/add or use fmaf (would change rounding vs the per-slot path).
// `out[i] = acc` overwrites (acc absorbs the routed_acc=0 init).
extern "C" __global__ void aegis_moe_weighted_accumulate(
    float* __restrict__ out,
    const float* __restrict__ expert_out,
    const unsigned int output_stride,
    const unsigned int* __restrict__ packed_topk,
    const unsigned int top_k,
    const unsigned int len
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) {
        return;
    }
    float acc = 0.0f;
    for (unsigned int k = 0u; k < top_k; ++k) {
        const float w = __uint_as_float(packed_topk[k * 2u + 1u]);
        acc += w * expert_out[size_t(k) * output_stride + idx];
    }
    out[idx] = acc;
}

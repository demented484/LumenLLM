// Standalone FP8 E4M3 linear weight kernels.
//
// Used by `shared-MLP-quantization = "fp8"` and (planned)
// `attention-quantization = "fp8"`. The load-time quantizer in
// `cuda/loader.rs` calls `aegis_quantize_bf16_to_fp8_per_row` once per
// projection to convert BF16 weights to FP8 + per-row FP32 scales; at
// inference time, `aegis_fp8_matvec` (decode) and
// `aegis_fp8_matmul_batched` (prefill) dequant on the fly.
//
// Layout:
//   data[r * cols + c] = E4M3 quantized value in [-448, 448]
//   row_scales[r]      = FP32 dequant scale; original ≈ E4M3(data) * scale
//
// Per-row (rather than per-group) gives a tiny scale buffer (rows*4
// bytes) and is sufficient for shared-MLP/attention BF16 weights whose
// row-wise dynamic range is well-behaved. If quality drifts more than
// ~1% PPL we'll revisit with per-group scales.

#ifndef AEGIS_FP8_LINEAR_CU
#define AEGIS_FP8_LINEAR_CU

extern "C" __global__ void aegis_quantize_bf16_to_fp8_per_row(
    const unsigned short* __restrict__ bf16,  // [rows, cols] BF16 packed as u16
    unsigned char* __restrict__ fp8_out,       // [rows, cols] E4M3
    float* __restrict__ row_scales_out,        // [rows] FP32 per-row scales
    const unsigned int rows,
    const unsigned int cols)
{
    const unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int block_dim = blockDim.x;

    const unsigned short* row_bf16 = bf16 + row * cols;
    unsigned char* row_fp8 = fp8_out + row * cols;

    // Pass 1: per-row absmax via warp/block reduction.
    float local_amax = 0.0f;
    for (unsigned int c = tid; c < cols; c += block_dim) {
        unsigned int bits = ((unsigned int)row_bf16[c]) << 16u;
        float v = __uint_as_float(bits);
        float a = fabsf(v);
        if (a > local_amax) local_amax = a;
    }
    // Block-wide reduction in shared mem.
    extern __shared__ float sdata[];  // size = block_dim
    sdata[tid] = local_amax;
    __syncthreads();
    for (unsigned int s = block_dim / 2u; s > 0u; s >>= 1) {
        if (tid < s) {
            float other = sdata[tid + s];
            if (other > sdata[tid]) sdata[tid] = other;
        }
        __syncthreads();
    }
    float amax = sdata[0];

    // Scale: divide by E4M3 max (448) so quantized values fit. Guard
    // against amax==0 (empty/zero row).
    const float fp8_max = 448.0f;
    float scale = (amax > 0.0f) ? (amax / fp8_max) : 1.0f;
    float inv_scale = 1.0f / scale;

    if (tid == 0) {
        row_scales_out[row] = scale;
    }

    // Pass 2: quantize each value as E4M3.
    for (unsigned int c = tid; c < cols; c += block_dim) {
        unsigned int bits = ((unsigned int)row_bf16[c]) << 16u;
        float v = __uint_as_float(bits);
        float scaled = v * inv_scale;
        row_fp8[c] = float_to_fp8_e4m3_bits(scaled);
    }
}

// Single-token matvec: out[r] = sum_c { dequant(fp8[r,c]) * input[c] }
//                            = scale[r] * sum_c { e4m3_to_f32(fp8[r,c]) * input[c] }
//
// Block per output row; threads cooperatively reduce along K.
extern "C" __global__ void aegis_fp8_matvec(
    const unsigned char* __restrict__ fp8,      // [rows, cols]
    const float* __restrict__ row_scales,        // [rows]
    const float* __restrict__ input,             // [cols]
    const unsigned int rows,
    const unsigned int cols,
    float* __restrict__ output)                  // [rows]
{
    const unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int block_dim = blockDim.x;

    const unsigned char* row_fp8 = fp8 + row * cols;

    float partial = 0.0f;
    for (unsigned int c = tid; c < cols; c += block_dim) {
        float w = fp8_e4m3_bits_to_float(row_fp8[c]);
        partial += w * input[c];
    }

    extern __shared__ float sdata[];
    sdata[tid] = partial;
    __syncthreads();
    for (unsigned int s = block_dim / 2u; s > 0u; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        output[row] = sdata[0] * row_scales[row];
    }
}

// Batched matmul (prefill path): out[b, r] = sum_c { dequant(fp8[r,c]) * input[b, c] }
//
// Grid: (rows, batch). Block: 128 threads. One block per (row, token) pair;
// threads reduce along K. Simple and correct — performance optimization
// is a follow-up (tile/wmma fusion).
extern "C" __global__ void aegis_fp8_matmul_batched(
    const unsigned char* __restrict__ fp8,      // [rows, cols]
    const float* __restrict__ row_scales,        // [rows]
    const float* __restrict__ input,             // [batch, cols]
    const unsigned int rows,
    const unsigned int cols,
    const unsigned int batch,
    float* __restrict__ output)                  // [batch, rows]
{
    const unsigned int row = blockIdx.x;
    const unsigned int b   = blockIdx.y;
    if (row >= rows || b >= batch) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int block_dim = blockDim.x;

    const unsigned char* row_fp8 = fp8 + row * cols;
    const float* row_in = input + b * cols;

    float partial = 0.0f;
    for (unsigned int c = tid; c < cols; c += block_dim) {
        float w = fp8_e4m3_bits_to_float(row_fp8[c]);
        partial += w * row_in[c];
    }

    extern __shared__ float sdata[];
    sdata[tid] = partial;
    __syncthreads();
    for (unsigned int s = block_dim / 2u; s > 0u; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        output[b * rows + row] = sdata[0] * row_scales[row];
    }
}

// Dequantize a standalone-FP8 weight matrix into a BF16 scratch buffer:
//   bf16_out[r, c] = bf16( e4m3_to_f32(fp8[r, c]) * row_scales[r] )
//
// This unlocks the existing BF16 cuBLASLt tensor-core GEMM path for FP8
// weights at the cost of one streaming dequant pass per call. Memory
// traffic: rows*cols B (read FP8) + rows*4 B (read scales) + rows*cols*2 B
// (write BF16). At ~700 GB/s HBM the dequant is microseconds for typical
// projection sizes; the cuBLASLt GEMM dominates wall time.
//
// Grid: (rows, ceil(cols/blockDim.x)). Block: 256 threads.
extern "C" __global__ void aegis_dequant_fp8_to_bf16(
    const unsigned char* __restrict__ fp8,      // [rows, cols]
    const float* __restrict__ row_scales,        // [rows]
    unsigned short* __restrict__ bf16_out,       // [rows, cols] BF16 packed as u16
    const unsigned int rows,
    const unsigned int cols)
{
    const unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    const float scale = row_scales[row];
    const float v = fp8_e4m3_bits_to_float(fp8[row * cols + col]) * scale;
    // BF16 round-to-nearest-even via raw bit manipulation: take the high 16
    // bits of f32, with bias for round-to-nearest (drop NaN/Inf handling
    // since dequant output is already in a tame range).
    unsigned int bits = __float_as_uint(v);
    unsigned int rounded = bits + 0x7FFFu + ((bits >> 16u) & 1u);
    bf16_out[row * cols + col] = (unsigned short)(rounded >> 16u);
}

#endif  // AEGIS_FP8_LINEAR_CU

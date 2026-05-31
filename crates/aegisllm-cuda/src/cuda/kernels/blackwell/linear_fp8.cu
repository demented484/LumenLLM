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

// FP8 BLOCK-scaled matvec (DeepSeek-style W8 with [blk×blk] weight_scale_inv):
//   output[r] = sum_c { e4m3_to_f32(fp8[r,c]) * scale[r/blk, c/blk] * input[c] }
// The scale varies along BOTH axes (unlike per-row), so it is folded inside the
// K-loop. Block per output row; threads cooperatively reduce along K.
extern "C" __global__ void aegis_fp8_block_matvec(
    const unsigned char* __restrict__ fp8,      // [rows, cols]
    const float* __restrict__ block_scales,      // [ceil(rows/blk), scale_cols]
    const float* __restrict__ input,             // [cols]
    const unsigned int rows,
    const unsigned int cols,
    const unsigned int blk,
    const unsigned int scale_cols,
    float* __restrict__ output)                  // [rows]
{
    const unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int block_dim = blockDim.x;
    const unsigned char* row_fp8 = fp8 + (size_t)row * cols;
    const unsigned int srow = (row / blk) * scale_cols;

    // Decode LUT in shared memory: the e4m3→float decode (branches + exp2f) is
    // otherwise evaluated 16×/thread and makes the GEMV ALU-bound. Each block
    // computes the 256-entry table once (1 KiB smem) → one smem load per weight.
    __shared__ float e4m3_lut[256];
    for (unsigned int i = tid; i < 256u; i += block_dim) {
        e4m3_lut[i] = fp8_e4m3_bits_to_float((unsigned char)i);
    }
    __syncthreads();

    // Vectorized: each thread reads 16 fp8 weights (one 128-bit uint4 load) and
    // 16 inputs (four float4 loads) per chunk. cols and the weight row base are
    // 16-aligned (hidden/intermediate are multiples of 16), so the loads are
    // aligned and coalesced — vs 1-byte scalar loads that ran at ~25% of HBM.
    // A 16-element chunk never crosses a 128-element scale block (base%128 ≤
    // 112 for any 16-aligned base), so one scale lookup covers all 16.
    float partial = 0.0f;
    const unsigned int nvec = cols / 16u;
    const uint4* row_fp8_v = reinterpret_cast<const uint4*>(row_fp8);
    const float4* input_v = reinterpret_cast<const float4*>(input);
    for (unsigned int ch = tid; ch < nvec; ch += block_dim) {
        uint4 wpack = row_fp8_v[ch];
        const unsigned char* wb = reinterpret_cast<const unsigned char*>(&wpack);
        float s = block_scales[srow + (ch * 16u) / blk];
        float inbuf[16];
        *reinterpret_cast<float4*>(&inbuf[0])  = input_v[ch * 4u + 0u];
        *reinterpret_cast<float4*>(&inbuf[4])  = input_v[ch * 4u + 1u];
        *reinterpret_cast<float4*>(&inbuf[8])  = input_v[ch * 4u + 2u];
        *reinterpret_cast<float4*>(&inbuf[12]) = input_v[ch * 4u + 3u];
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            partial += e4m3_lut[wb[k]] * s * inbuf[k];
        }
    }
    // Scalar tail for any cols not divisible by 16 (none of the current shapes,
    // but keeps the kernel correct if that changes).
    for (unsigned int c = nvec * 16u + tid; c < cols; c += block_dim) {
        partial += e4m3_lut[row_fp8[c]] * block_scales[srow + c / blk] * input[c];
    }
    // Warp-shuffle reduction: each warp reduces in registers, then warp 0
    // combines the per-warp sums (1 barrier vs the 8-barrier smem tree).
    for (int off = 16; off > 0; off >>= 1)
        partial += __shfl_down_sync(0xffffffffu, partial, off);
    __shared__ float warp_sums[32];
    const unsigned int lane = tid & 31u, warp = tid >> 5;
    if (lane == 0) warp_sums[warp] = partial;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < (block_dim >> 5)) ? warp_sums[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) output[row] = v;
    }
}

// Batched (prefill) variant of the FP8 block-scaled matmul.
// out[b, r] = sum_c { e4m3(fp8[r,c]) * scale[r/blk, c/blk] * input[b, c] }
// Grid: (rows, batch). Block: threads reduce along K.
extern "C" __global__ void aegis_fp8_block_matmul_batched(
    const unsigned char* __restrict__ fp8,      // [rows, cols]
    const float* __restrict__ block_scales,      // [ceil(rows/blk), scale_cols]
    const float* __restrict__ input,             // [batch, cols]
    const unsigned int rows,
    const unsigned int cols,
    const unsigned int blk,
    const unsigned int scale_cols,
    const unsigned int batch,
    float* __restrict__ output)                  // [batch, rows]
{
    const unsigned int row = blockIdx.x;
    const unsigned int b   = blockIdx.y;
    if (row >= rows || b >= batch) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int block_dim = blockDim.x;
    const unsigned char* row_fp8 = fp8 + (size_t)row * cols;
    const float* row_in = input + (size_t)b * cols;
    const unsigned int srow = (row / blk) * scale_cols;

    float partial = 0.0f;
    for (unsigned int c = tid; c < cols; c += block_dim) {
        float w = fp8_e4m3_bits_to_float(row_fp8[c]);
        float s = block_scales[srow + c / blk];
        partial += w * s * row_in[c];
    }
    extern __shared__ float sdata[];
    sdata[tid] = partial;
    __syncthreads();
    for (unsigned int s = block_dim / 2u; s > 0u; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) output[(size_t)b * rows + row] = sdata[0];
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

// Block-scaled FP8 → BF16 dequant (DeepSeek-style): bf16_out[r,c] =
//   bf16( e4m3(fp8[r,c]) * block_scales[r/blk, c/blk] ).
// Unlocks the cuBLASLt BF16 GEMM (weight read once, amortized over M tokens) for
// the block-scaled FP8 weights used by Qwen3.5-9B prefill.
// Grid: (rows, ceil(cols/blockDim.x)). Block: 256.
extern "C" __global__ void aegis_dequant_fp8_block_to_bf16(
    const unsigned char* __restrict__ fp8,      // [rows, cols]
    const float* __restrict__ block_scales,      // [ceil(rows/blk), scale_cols]
    unsigned short* __restrict__ bf16_out,       // [rows, cols] BF16 packed as u16
    const unsigned int rows,
    const unsigned int cols,
    const unsigned int blk,
    const unsigned int scale_cols)
{
    const unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    const float s = block_scales[(row / blk) * scale_cols + (col / blk)];
    const float v = fp8_e4m3_bits_to_float(fp8[(size_t)row * cols + col]) * s;
    unsigned int bits = __float_as_uint(v);
    unsigned int rounded = bits + 0x7FFFu + ((bits >> 16u) & 1u);
    bf16_out[(size_t)row * cols + col] = (unsigned short)(rounded >> 16u);
}

#endif  // AEGIS_FP8_LINEAR_CU

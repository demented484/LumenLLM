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

#ifndef AEGIS_CUDA_FP8_H_INCLUDED
#define AEGIS_CUDA_FP8_H_INCLUDED
#include <cuda_fp8.h>   // __nv_fp8_e4m3 (hardware round-to-nearest-even e4m3 encoder)
#endif

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

// ============================================================================
// Native FP8 block-scaled W8A8 tiled GEMM (Qwen3.5-9B prefill, no dequant).
//   out[M,N] = A[M,K] @ W[N,K]^T   (row.col MMA form)
// A = activation e4m3 [M,K] with per-(token,128-K-group) scale a_scale[M,K/128];
// W = weight e4m3 [N,K] with weight_scale_inv[N/128,K/128]. Both scales MULTIPLY.
// Scales depend on k/128, so the f32 accumulator is rescaled+flushed every 128 K.
// ============================================================================

// Quantize f32 activation [M,K] -> e4m3 [M,K] + per-(token,128-K-group) scale.
// scale = max(|group|,1e-10)/448 (stored, MULTIPLY on dequant); q = RNE_e4m3(a/scale).
// Grid (M, K/128); block 128 (one thread per K-element in the group).
extern "C" __global__ void aegis_quantize_f32_to_fp8_token_group(
    const float* __restrict__ a,        // [M, K]
    unsigned char* __restrict__ a_q,    // [M, K]
    float* __restrict__ a_scale,        // [M, n_kgroups]
    const unsigned int M,
    const unsigned int K,
    const unsigned int n_kgroups)
{
    const unsigned int m = blockIdx.x;
    const unsigned int g = blockIdx.y;
    const unsigned int tid = threadIdx.x;     // 0..127
    const unsigned int k = g * 128u + tid;
    const float v = (m < M && k < K) ? a[(size_t)m * K + k] : 0.0f;
    __shared__ float sm[128];
    sm[tid] = fabsf(v);
    __syncthreads();
    for (unsigned int s = 64u; s > 0u; s >>= 1) { if (tid < s) sm[tid] = fmaxf(sm[tid], sm[tid + s]); __syncthreads(); }
    const float scale = fmaxf(sm[0], 1.0e-10f) / 448.0f;
    if (tid == 0u && m < M) a_scale[(size_t)m * n_kgroups + g] = scale;
    if (m < M && k < K) {
        const float q = fminf(fmaxf(v / scale, -448.0f), 448.0f);
        // Hardware round-to-nearest-even e4m3 encoder. The hand-rolled
        // `float_to_fp8_e4m3_bits` is round-half-UP, which over a 4096-wide
        // RMS-normed activation accumulates a ~12% upward magnitude bias —
        // catastrophic in the recurrent GDN prefill path (gibberish). The
        // `__nv_fp8_e4m3` constructor (cuda_fp8.h, NVRTC-compilable on SM120,
        // the same encoder the HW-verified fp8_mma_smoke harness uses) is
        // unbiased and bit-matches the e4m3 the SM120 MMA decodes. Brought
        // act-quant rel-err 30%->2.6% and the GEMM-vs-decode cos 0.9975->0.99996.
        const __nv_fp8_e4m3 e = __nv_fp8_e4m3(q);
        a_q[(size_t)m * K + k] = *reinterpret_cast<const unsigned char*>(&e);
    }
}

// Tiled FP8 block-scaled GEMM. Block = 64x64 C tile, 256 threads (8 warps, 4 M x
// 2 N); each warp owns [16,32] = 4 n8 MMA tiles. K processed in 128-blocks
// (4 m16n8k32 e4m3 MMA steps), rescaled by a_scale[m,g]*w_scale[n/128,g] per block.
// Assumes N%64==0, K%128==0 (true for Qwen3.5-9B). M-tail guarded.
extern "C" __global__ void aegis_fp8_block_gemm(
    const unsigned char* __restrict__ a_q,    // [M, K] e4m3
    const float* __restrict__ a_scale,         // [M, K/128]
    const unsigned char* __restrict__ w,       // [N, K] e4m3
    const float* __restrict__ w_scale,         // [N/128, K/128]
    float* __restrict__ out,                   // [M, N] f32
    const unsigned int M,
    const unsigned int N,
    const unsigned int K,
    const unsigned int n_kgroups)              // K/128
{
    __shared__ unsigned char As[64 * 128];     // 8 KiB
    __shared__ unsigned char Bs[64 * 128];     // 8 KiB
    const unsigned int tile_m = blockIdx.y * 64u;
    const unsigned int tile_n = blockIdx.x * 64u;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;        // 0..7
    const unsigned int lane = tid & 31u;
    const unsigned int g_l  = lane >> 2;        // 0..7
    const unsigned int q_l  = lane & 3u;        // 0..3
    const unsigned int warp_m = warp & 3u;      // 0..3
    const unsigned int warp_n = warp >> 2;      // 0..1
    const unsigned int w_scale_n = tile_n / 128u;

    float out_acc[4][4];
    #pragma unroll
    for (int nt = 0; nt < 4; ++nt) { out_acc[nt][0]=0.f; out_acc[nt][1]=0.f; out_acc[nt][2]=0.f; out_acc[nt][3]=0.f; }

    for (unsigned int g = 0u; g < n_kgroups; ++g) {
        const unsigned int k0 = g * 128u;
        // Cooperative coalesced load of As[64,128] + Bs[64,128] (uint4 = 16 e4m3).
        for (unsigned int i = tid; i < 512u; i += 256u) {     // 512 uint4 per tile
            const unsigned int row = i >> 3;                   // 0..63
            const unsigned int cu4 = (i & 7u) * 16u;           // col byte offset
            const unsigned int am = tile_m + row;
            *reinterpret_cast<uint4*>(&As[row * 128u + cu4]) =
                (am < M) ? *reinterpret_cast<const uint4*>(&a_q[(size_t)am * K + k0 + cu4])
                         : make_uint4(0u, 0u, 0u, 0u);
            *reinterpret_cast<uint4*>(&Bs[row * 128u + cu4]) =
                *reinterpret_cast<const uint4*>(&w[(size_t)(tile_n + row) * K + k0 + cu4]);
        }
        __syncthreads();

        float blk[4][4];
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) { blk[nt][0]=0.f; blk[nt][1]=0.f; blk[nt][2]=0.f; blk[nt][3]=0.f; }

        #pragma unroll
        for (int kk = 0; kk < 4; ++kk) {        // 4 k32 steps in the 128-block
            const unsigned int kbase = kk * 32u;
            // A fragment (16x32) from As[16*warp_m .. , kbase ..]: a[v1+2*v2].
            aegis_u32 af[4];
            #pragma unroll
            for (int v2 = 0; v2 < 2; ++v2)
                #pragma unroll
                for (int v1 = 0; v1 < 2; ++v1) {
                    const unsigned int rr = 16u * warp_m + g_l + 8u * v1;
                    const unsigned int cc = kbase + q_l * 4u + 16u * v2;
                    const unsigned char* p = &As[rr * 128u + cc];
                    af[v1 + 2 * v2] = aegis_pack_e4m3x4_p(p[0], p[1], p[2], p[3]);
                }
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                // B fragment (8x32) from Bs[32*warp_n + 8*nt .., kbase ..]: b[v1].
                aegis_u32 bf[2];
                #pragma unroll
                for (int v1 = 0; v1 < 2; ++v1) {
                    const unsigned int rr = 32u * warp_n + 8u * (unsigned)nt + g_l;
                    const unsigned int cc = kbase + q_l * 4u + 16u * v1;
                    const unsigned char* p = &Bs[rr * 128u + cc];
                    bf[v1] = aegis_pack_e4m3x4_p(p[0], p[1], p[2], p[3]);
                }
                aegis_mma_m16n8k32_e4m3_p(blk[nt], af, bf, blk[nt]);
            }
        }
        __syncthreads();    // As/Bs reusable next g

        // Rescale this 128-K block and accumulate. 2 distinct activation rows.
        const unsigned int m_base = tile_m + 16u * warp_m;
        const float ws  = w_scale[(size_t)w_scale_n * n_kgroups + g];
        const float as0 = ((m_base + g_l)     < M) ? a_scale[(size_t)(m_base + g_l)     * n_kgroups + g] : 0.f;
        const float as1 = ((m_base + g_l + 8u) < M) ? a_scale[(size_t)(m_base + g_l + 8u) * n_kgroups + g] : 0.f;
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            out_acc[nt][0] += blk[nt][0] * as0 * ws;
            out_acc[nt][1] += blk[nt][1] * as0 * ws;
            out_acc[nt][2] += blk[nt][2] * as1 * ws;
            out_acc[nt][3] += blk[nt][3] * as1 * ws;
        }
    }

    // Epilogue: write [16,32] per warp.
    const unsigned int m_base = tile_m + 16u * warp_m;
    #pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
        const unsigned int n_base = tile_n + 32u * warp_n + 8u * (unsigned)nt;
        const unsigned int r0 = m_base + g_l, r1 = m_base + g_l + 8u;
        const unsigned int c0 = n_base + 2u * q_l, c1 = n_base + 2u * q_l + 1u;
        if (r0 < M) { out[(size_t)r0 * N + c0] = out_acc[nt][0]; out[(size_t)r0 * N + c1] = out_acc[nt][1]; }
        if (r1 < M) { out[(size_t)r1 * N + c0] = out_acc[nt][2]; out[(size_t)r1 * N + c1] = out_acc[nt][3]; }
    }
}

#endif  // AEGIS_FP8_LINEAR_CU

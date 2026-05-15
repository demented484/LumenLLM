// =============================================================================
// FP8 e4m3 m16n8k32 tensor-core MMA smoke kernels (SM120 / Blackwell consumer).
// =============================================================================
//
// Synthetic, model-free verification of the raw FP8 MMA primitive that a
// from-scratch FP8 FlashAttention kernel will be built on:
//
//   mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32
//
// This is the SM120 form (note the `kind::f8f6f4` prefix — the bare SM89
// `mma...m16n8k32...e4m3` form does NOT exist on SM120). nvcuda::wmma
// (m16n16k16) cannot express it, so it must be inline PTX.
//
// Operand register counts (from CUTLASS cute/arch/mma_sm120.hpp,
// SM120_16x8x32_TN<float_e4m3_t,float_e4m3_t,float>):
//   A = uint32_t[4]   (16 e4m3 bytes per thread)
//   B = uint32_t[2]   ( 8 e4m3 bytes per thread)
//   C = float[4], D = float[4]
//
// Fragment->thread layout is the universal int8/fp8 m16n8k32 layout, inherited
// in CUTLASS from MMA_Traits<SM80_16x8x32_S32S8S8S32_TN>:
//   ALayout = ((_4,_8),(_4,_2,_2)) : ((_64,_1),(_16,_8,_256))
//   BLayout = ((_4,_8),(_4,_2))    : ((_32,_1),(_8,_128))
//   CLayout = SM80_16x8_Row
//
// Decoded (groupID g = lane/4, threadID-in-group q = lane%4):
//   A[m,k]:  m = g + 8*v1,         k = q*4 + v0 + 16*v2
//            register r in [0,4), byte v0 in [0,4):  r = v1 + 2*v2
//   B[n,k]:  n = g,                k = q*4 + v0 + 16*v1
//            register r in [0,2), byte v0 in [0,4):  r = v1
//   C/D[m,n]: c0,c1 -> row g,   col 2*q + {0,1}
//             c2,c3 -> row g+8, col 2*q + {0,1}
//
// (Register<->v1/v2 packing order is the one assumption verified empirically by
// Stage 1 with distinguishable integer inputs; if it were swapped the Stage-1
// bit-exact check would fail loudly.)
//
// All kernels here load NO model — a few small device buffers only. Safe to run
// compute-sanitizer memcheck/racecheck on.
// =============================================================================

#include <cuda_fp8.h>

#ifndef AEGIS_FP8_SMOKE_GUARD
#define AEGIS_FP8_SMOKE_GUARD

// NVRTC has no <cstdint>; the kernel only needs a 32-bit unsigned type for the
// MMA register operands.
typedef unsigned int uint32_t;

// -----------------------------------------------------------------------------
// Stage 1 primitive: the bare 16x8x32 e4m3 MMA.
// d[4] = A(16x32 e4m3) * B(8x32 e4m3)^T + c[4], all in the canonical fragment
// layout. This is THE device function the full FP8 FA kernel will call.
// -----------------------------------------------------------------------------
__device__ __forceinline__ void aegis_mma_m16n8k32_e4m3(
    float d[4],
    const uint32_t a[4],
    const uint32_t b[2],
    const float c[4]) {
#if (__CUDA_ARCH__ >= 1200)
  asm volatile(
      "mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,  %1,  %2,  %3},"
      "{%4,  %5,  %6,  %7},"
      "{%8,  %9},"
      "{%10, %11, %12, %13};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b[0]), "r"(b[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
#else
  // Non-SM120 fallback: keep the TU compilable for other arches. The smoke
  // command guards on compute capability and will not launch this there.
  d[0] = c[0]; d[1] = c[1]; d[2] = c[2]; d[3] = c[3];
#endif
}

// e4m3 byte helpers ----------------------------------------------------------
// Pack four e4m3 bytes (already encoded) into one b32 register, little-endian
// (byte 0 = bits 0..7), matching how the MMA consumes a .b32 A/B fragment.
__device__ __forceinline__ uint32_t aegis_pack_e4m3x4(
    unsigned char b0, unsigned char b1, unsigned char b2, unsigned char b3) {
  return (uint32_t)b0 | ((uint32_t)b1 << 8) | ((uint32_t)b2 << 16) |
         ((uint32_t)b3 << 24);
}

// Encode an f32 value to an e4m3 byte (round-to-nearest, HW path).
__device__ __forceinline__ unsigned char aegis_f32_to_e4m3(float x) {
  __nv_fp8_e4m3 v = __nv_fp8_e4m3(x);
  return *reinterpret_cast<unsigned char*>(&v);
}

// Decode an e4m3 byte back to f32 (for CPU-parity dequant on device if needed).
__device__ __forceinline__ float aegis_e4m3_to_f32(unsigned char b) {
  __nv_fp8_e4m3 v;
  *reinterpret_cast<unsigned char*>(&v) = b;
  return float(v);
}

// =============================================================================
// Stage 1 kernel: single 16x8x32 tile.
// Inputs A_e4m3 (16*32 bytes, row-major M-major: A[m*32+k]) and
//        B_e4m3 ( 8*32 bytes, row-major N-major: B[n*32+k]).
// One warp (32 threads). Each thread loads its fragment slice per the canonical
// layout, runs ONE MMA with c=0, writes its 4 accumulators to out[m*8+n].
// =============================================================================
extern "C" __global__ void aegis_fp8_mma_smoke_stage1(
    const unsigned char* __restrict__ A_e4m3,  // 16 x 32
    const unsigned char* __restrict__ B_e4m3,  //  8 x 32
    float* __restrict__ out) {                 // 16 x 8
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;        // groupID 0..7
  const int q = lane & 3;         // threadID-in-group 0..3

  uint32_t a[4];
  uint32_t b[2];
  float c[4] = {0.f, 0.f, 0.f, 0.f};
  float d[4];

  // A fragment: register r = v1 + 2*v2 ; byte v0.
  //   m = g + 8*v1 ; k = q*4 + v0 + 16*v2
  for (int v2 = 0; v2 < 2; ++v2) {
    for (int v1 = 0; v1 < 2; ++v1) {
      int r = v1 + 2 * v2;
      unsigned char bytes[4];
      for (int v0 = 0; v0 < 4; ++v0) {
        int m = g + 8 * v1;
        int k = q * 4 + v0 + 16 * v2;
        bytes[v0] = A_e4m3[m * 32 + k];
      }
      a[r] = aegis_pack_e4m3x4(bytes[0], bytes[1], bytes[2], bytes[3]);
    }
  }

  // B fragment: register r = v1 ; byte v0.
  //   n = g ; k = q*4 + v0 + 16*v1
  for (int v1 = 0; v1 < 2; ++v1) {
    unsigned char bytes[4];
    for (int v0 = 0; v0 < 4; ++v0) {
      int n = g;
      int k = q * 4 + v0 + 16 * v1;
      bytes[v0] = B_e4m3[n * 32 + k];
    }
    b[v1] = aegis_pack_e4m3x4(bytes[0], bytes[1], bytes[2], bytes[3]);
  }

  aegis_mma_m16n8k32_e4m3(d, a, b, c);

  // C/D layout: c0,c1 -> (row g,   col 2q+{0,1})
  //             c2,c3 -> (row g+8, col 2q+{0,1})
  out[(g) * 8 + (2 * q + 0)] = d[0];
  out[(g) * 8 + (2 * q + 1)] = d[1];
  out[(g + 8) * 8 + (2 * q + 0)] = d[2];
  out[(g + 8) * 8 + (2 * q + 1)] = d[3];
}

// =============================================================================
// Stage 2 kernel: tiled FP8 GEMM, D[M,N] = A[M,K] * B[N,K]^T.
// M, N are multiples of 16 / 8 respectively; K a multiple of 32.
// One CTA = one warp computes one 16x8 output tile, looping over K in steps
// of 32. A is row-major (A[m*K+k]); B is row-major N-major (B[n*K+k]) — i.e.
// the .row.col (TN) convention the MMA wants natively.
// Grid: (N/8, M/16). Block: 32 threads.
// =============================================================================
extern "C" __global__ void aegis_fp8_mma_smoke_stage2(
    const unsigned char* __restrict__ A_e4m3,  // M x K
    const unsigned char* __restrict__ B_e4m3,  // N x K
    float* __restrict__ D,                     // M x N
    int M, int N, int K) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int q = lane & 3;

  const int tile_m = blockIdx.y * 16;
  const int tile_n = blockIdx.x * 8;
  if (tile_m >= M || tile_n >= N) return;

  float acc[4] = {0.f, 0.f, 0.f, 0.f};

  for (int k0 = 0; k0 < K; k0 += 32) {
    uint32_t a[4];
    uint32_t b[2];

    for (int v2 = 0; v2 < 2; ++v2) {
      for (int v1 = 0; v1 < 2; ++v1) {
        int r = v1 + 2 * v2;
        unsigned char bytes[4];
        for (int v0 = 0; v0 < 4; ++v0) {
          int m = tile_m + g + 8 * v1;
          int k = k0 + q * 4 + v0 + 16 * v2;
          bytes[v0] = A_e4m3[m * K + k];
        }
        a[r] = aegis_pack_e4m3x4(bytes[0], bytes[1], bytes[2], bytes[3]);
      }
    }
    for (int v1 = 0; v1 < 2; ++v1) {
      unsigned char bytes[4];
      for (int v0 = 0; v0 < 4; ++v0) {
        int n = tile_n + g;
        int k = k0 + q * 4 + v0 + 16 * v1;
        bytes[v0] = B_e4m3[n * K + k];
      }
      b[v1] = aegis_pack_e4m3x4(bytes[0], bytes[1], bytes[2], bytes[3]);
    }

    float d[4];
    aegis_mma_m16n8k32_e4m3(d, a, b, acc);
    acc[0] = d[0]; acc[1] = d[1]; acc[2] = d[2]; acc[3] = d[3];
  }

  D[(tile_m + g) * N + (tile_n + 2 * q + 0)] = acc[0];
  D[(tile_m + g) * N + (tile_n + 2 * q + 1)] = acc[1];
  D[(tile_m + g + 8) * N + (tile_n + 2 * q + 0)] = acc[2];
  D[(tile_m + g + 8) * N + (tile_n + 2 * q + 1)] = acc[3];
}

// =============================================================================
// Stage 3 kernel: tiny synthetic FP8 attention, one CTA per (head, q-tile).
// head_dim = 512, causal. Q/K/V are pre-quantized to e4m3 on the host with a
// PER-ROW absmax scale (e4m3 has only 3 mantissa bits — per-row scaling of
// Q/K/V is REQUIRED). The kernel receives e4m3 tensors plus per-row f32 scales.
//
// Layout (host-side):
//   Q_e4m3 : [n_heads][q_tile=16][head_dim]   row-major
//   K_e4m3 : [n_heads][ctx]      [head_dim]   row-major
//   V_e4m3 : [n_heads][ctx]      [head_dim]   row-major
//   q_scale: [n_heads][q_tile]   (f32, dequant multiplier per Q row)
//   k_scale: [n_heads][ctx]      (f32)
//   v_scale: [n_heads][ctx]      (f32)
//
// Algorithm (one warp per (head, q-tile=16 rows)):
//   S[16,ctx] = Q.K^T  via FP8 MMA, dequant scores by q_scale*k_scale
//   S *= rsqrt(head_dim); causal mask; f32 online softmax over ctx
//   O[16,head_dim] = P.V   — P re-quantized to e4m3 with a per-row scale
//                            (online; this is the documented Option A).
//
// ctx is a multiple of 8 (N tiles), head_dim a multiple of 32 (K tiles for
// Q.K) and of 8 (N tiles for P.V). q_tile fixed 16. One warp; ctx<=256 and
// head_dim<=512 kept small so scores live in shared memory.
// =============================================================================
extern "C" __global__ void aegis_fp8_mma_smoke_stage3(
    const unsigned char* __restrict__ Q_e4m3,  // [H][16][D]
    const unsigned char* __restrict__ K_e4m3,  // [H][ctx][D]
    const unsigned char* __restrict__ V_e4m3,  // [H][ctx][D]
    const float* __restrict__ q_scale,         // [H][16]
    const float* __restrict__ k_scale,         // [H][ctx]
    const float* __restrict__ v_scale,         // [H][ctx]
    float* __restrict__ O,                     // [H][16][D]
    int H, int ctx, int D) {
  const int head = blockIdx.x;
  if (head >= H) return;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int q = lane & 3;

  extern __shared__ float smem[];
  // scores: 16 x ctx  (f32). Followed by P_e4m3 staging: 16 x ctx bytes.
  float* scores = smem;                              // 16*ctx floats
  unsigned char* P_e4m3 =
      reinterpret_cast<unsigned char*>(scores + 16 * ctx);  // 16*ctx bytes
  float* p_scale = reinterpret_cast<float*>(
      P_e4m3 + ((16 * ctx + 3) & ~3));                      // 16 floats

  const unsigned char* Qh = Q_e4m3 + (size_t)head * 16 * D;
  const unsigned char* Kh = K_e4m3 + (size_t)head * ctx * D;
  const unsigned char* Vh = V_e4m3 + (size_t)head * ctx * D;
  const float* qsh = q_scale + (size_t)head * 16;
  const float* ksh = k_scale + (size_t)head * ctx;
  const float* vsh = v_scale + (size_t)head * ctx;
  float* Oh = O + (size_t)head * 16 * D;

  const float softmax_scale = rsqrtf((float)D);

  // ---- S = Q.K^T -----------------------------------------------------------
  // For each N-tile of 8 keys, accumulate over K-tiles (head_dim).
  for (int n0 = 0; n0 < ctx; n0 += 8) {
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k0 = 0; k0 < D; k0 += 32) {
      uint32_t a[4];
      uint32_t b[2];
      for (int v2 = 0; v2 < 2; ++v2) {
        for (int v1 = 0; v1 < 2; ++v1) {
          int r = v1 + 2 * v2;
          unsigned char by[4];
          for (int v0 = 0; v0 < 4; ++v0) {
            int m = g + 8 * v1;            // Q row
            int k = k0 + q * 4 + v0 + 16 * v2;
            by[v0] = Qh[m * D + k];
          }
          a[r] = aegis_pack_e4m3x4(by[0], by[1], by[2], by[3]);
        }
      }
      for (int v1 = 0; v1 < 2; ++v1) {
        unsigned char by[4];
        for (int v0 = 0; v0 < 4; ++v0) {
          int n = n0 + g;                  // K row
          int k = k0 + q * 4 + v0 + 16 * v1;
          by[v0] = Kh[n * D + k];
        }
        b[v1] = aegis_pack_e4m3x4(by[0], by[1], by[2], by[3]);
      }
      float d[4];
      aegis_mma_m16n8k32_e4m3(d, a, b, acc);
      acc[0]=d[0]; acc[1]=d[1]; acc[2]=d[2]; acc[3]=d[3];
    }
    // Dequant + write to shared scores. acc holds raw e4m3*e4m3 sums;
    // multiply by q_scale[row]*k_scale[col].
    int row0 = g,      col0 = n0 + 2 * q + 0;
    int row1 = g + 8,  col1 = n0 + 2 * q + 1;
    scores[row0 * ctx + col0] = acc[0] * qsh[row0] * ksh[col0];
    scores[row0 * ctx + col1] = acc[1] * qsh[row0] * ksh[col1];
    scores[row1 * ctx + col0] = acc[2] * qsh[row1] * ksh[col0];
    scores[row1 * ctx + col1] = acc[3] * qsh[row1] * ksh[col1];
  }
  __syncwarp();

  // ---- scale + causal mask + online softmax (per row) ----------------------
  // 16 rows, 32 lanes -> lanes 0..15 each own one row.
  for (int row = lane; row < 16; row += 32) {
    float m_run = -1e30f;
    for (int c = 0; c < ctx; ++c) {
      float s = scores[row * ctx + c] * softmax_scale;
      // causal: query row `row` attends keys c <= row + (ctx-16).
      // q-tile is the LAST 16 positions of a length-ctx sequence.
      int qpos = (ctx - 16) + row;
      if (c > qpos) s = -1e30f;
      scores[row * ctx + c] = s;
      m_run = fmaxf(m_run, s);
    }
    float denom = 0.f;
    for (int c = 0; c < ctx; ++c) {
      float e = __expf(scores[row * ctx + c] - m_run);
      scores[row * ctx + c] = e;
      denom += e;
    }
    float inv = denom > 0.f ? 1.f / denom : 0.f;
    // Normalize P and find per-row absmax for e4m3 requant.
    float amax = 0.f;
    for (int c = 0; c < ctx; ++c) {
      float p = scores[row * ctx + c] * inv;
      scores[row * ctx + c] = p;
      amax = fmaxf(amax, fabsf(p));
    }
    float ps = amax > 0.f ? amax / 448.0f : 1.0f;  // e4m3 max magnitude 448
    p_scale[row] = ps;
    float invps = ps > 0.f ? 1.f / ps : 0.f;
    for (int c = 0; c < ctx; ++c) {
      float pq = scores[row * ctx + c] * invps;
      P_e4m3[row * ctx + c] = aegis_f32_to_e4m3(pq);
    }
  }
  __syncwarp();

  // ---- O = P.V -------------------------------------------------------------
  // P is [16 x ctx] e4m3 (row-major); V is [ctx x D] e4m3. Contraction over
  // ctx. Output O[16 x D]. N-tiles of 8 over D, K-tiles of 32 over ctx.
  for (int n0 = 0; n0 < D; n0 += 8) {
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k0 = 0; k0 < ctx; k0 += 32) {
      uint32_t a[4];
      uint32_t b[2];
      for (int v2 = 0; v2 < 2; ++v2) {
        for (int v1 = 0; v1 < 2; ++v1) {
          int r = v1 + 2 * v2;
          unsigned char by[4];
          for (int v0 = 0; v0 < 4; ++v0) {
            int m = g + 8 * v1;             // P row (query)
            int k = k0 + q * 4 + v0 + 16 * v2;
            by[v0] = P_e4m3[m * ctx + k];
          }
          a[r] = aegis_pack_e4m3x4(by[0], by[1], by[2], by[3]);
        }
      }
      // B = V^T: the MMA wants B[n,k] with n the output col (a D index) and
      // k the contraction (a ctx index). V is [ctx][D], so B[n,k] = V[k,n].
      for (int v1 = 0; v1 < 2; ++v1) {
        unsigned char by[4];
        for (int v0 = 0; v0 < 4; ++v0) {
          int n = n0 + g;                   // D index
          int k = k0 + q * 4 + v0 + 16 * v1;  // ctx index
          by[v0] = Vh[k * D + n];
        }
        b[v1] = aegis_pack_e4m3x4(by[0], by[1], by[2], by[3]);
      }
      float d[4];
      aegis_mma_m16n8k32_e4m3(d, a, b, acc);
      acc[0]=d[0]; acc[1]=d[1]; acc[2]=d[2]; acc[3]=d[3];
    }
    // Dequant: acc = sum_k P_e4m3[m,k] * V_e4m3[k,n].
    // P true value = P_e4m3 * p_scale[m]; V true value = V_e4m3 * v_scale[k].
    // v_scale varies per k -> we CANNOT pull it out of the MMA sum. To keep
    // the smoke harness exact-as-possible we require the host to feed V with a
    // SINGLE per-head v scale broadcast across rows (v_scale[head][k] constant
    // over k). Then dequant = acc * p_scale[m] * v_scale_head.
    int row0 = g,      col0 = n0 + 2 * q + 0;
    int row1 = g + 8,  col1 = n0 + 2 * q + 1;
    float vsc = vsh[0];  // constant-over-k per-head V scale (see host note)
    Oh[row0 * D + col0] = acc[0] * p_scale[row0] * vsc;
    Oh[row0 * D + col1] = acc[1] * p_scale[row0] * vsc;
    Oh[row1 * D + col0] = acc[2] * p_scale[row1] * vsc;
    Oh[row1 * D + col1] = acc[3] * p_scale[row1] * vsc;
  }
}

#endif  // AEGIS_FP8_SMOKE_GUARD

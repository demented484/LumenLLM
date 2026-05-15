// =============================================================================
// FP8 e4m3 m16n8k32 tensor-core MMA primitive — shared header.
// =============================================================================
//
// The device-side FP8 MMA primitive, extracted so the production FP8
// FlashAttention prefill kernel (`attention_prefill_fa2_fp8.cu`) builds on the
// EXACT same instruction the model-free smoke harness (`fp8_mma_smoke.cu`)
// hardware-verified. The smoke file keeps its own private copy guarded by
// `AEGIS_FP8_SMOKE_GUARD` (do not disturb — it is verified); this header is a
// byte-identical re-statement of that proven primitive for the production
// translation unit, which compiles in a different NVRTC module.
//
// The MMA instruction and fragment->thread layout are documented and verified
// in `fp8_mma_smoke.cu` — see that file's header for the decoded layout:
//   groupID g = lane/4, threadID-in-group q = lane%4.
//   A[m,k]:  m = g + 8*v1,  k = q*4 + v0 + 16*v2,  register r = v1 + 2*v2
//   B[n,k]:  n = g,         k = q*4 + v0 + 16*v1,  register r = v1
//   C/D[m,n]: c0,c1 -> (row g,   col 2q+{0,1}); c2,c3 -> (row g+8, col 2q+{0,1})
// Stage-1 of the smoke harness verified that register packing empirically with
// distinguishable integer inputs.
//
// e4m3 byte<->float conversion is NOT defined here: the production module
// already includes `linear_utils.cuh`, whose `float_to_fp8_e4m3_bits` is the
// EXACT encoder the FP8 KV cache is written with (kv_fp8.cu). Reusing it keeps
// the kernel's Q/P quantization bit-consistent with the cache and avoids a
// second, divergent e4m3 encoder.
// =============================================================================

#ifndef AEGIS_FP8_MMA_CUH
#define AEGIS_FP8_MMA_CUH

// NVRTC has no <cstdint>; the MMA register operands only need a 32-bit unsigned.
#ifndef AEGIS_U32_TYPEDEF
#define AEGIS_U32_TYPEDEF
typedef unsigned int aegis_u32;
#endif

// -----------------------------------------------------------------------------
// The bare 16x8x32 e4m3 MMA (SM120 `kind::f8f6f4` form).
//   d[4] = A(16x32 e4m3) * B(8x32 e4m3)^T + c[4]
// A = uint32_t[4], B = uint32_t[2], C/D = float[4], canonical fragment layout.
// Byte-identical to `aegis_mma_m16n8k32_e4m3` in fp8_mma_smoke.cu.
// -----------------------------------------------------------------------------
__device__ __forceinline__ void aegis_mma_m16n8k32_e4m3_p(
    float d[4],
    const aegis_u32 a[4],
    const aegis_u32 b[2],
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
  // Non-SM120 fallback: keep the TU compilable for other arches. The dispatch
  // gates on compute capability and will not select this kernel there.
  d[0] = c[0]; d[1] = c[1]; d[2] = c[2]; d[3] = c[3];
#endif
}

// Pack four e4m3 bytes (already encoded) into one b32, little-endian
// (byte 0 = bits 0..7) — the order the MMA consumes a .b32 A/B fragment.
__device__ __forceinline__ aegis_u32 aegis_pack_e4m3x4_p(
    unsigned char b0, unsigned char b1, unsigned char b2, unsigned char b3) {
  return (aegis_u32)b0 | ((aegis_u32)b1 << 8) | ((aegis_u32)b2 << 16) |
         ((aegis_u32)b3 << 24);
}

#endif  // AEGIS_FP8_MMA_CUH

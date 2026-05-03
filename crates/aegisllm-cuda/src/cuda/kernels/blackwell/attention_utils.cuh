#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <mma.h>

#if __CUDA_ARCH__ >= 800
template <typename Fragment>
__device__ __forceinline__ void aegis_scale_wmma_accumulator_m16n16_rows(
    Fragment& frag,
    const float* __restrict__ scalars,
    const unsigned int row_base
) {
    const unsigned int lane_row_base = (threadIdx.x & 31u) >> 2u;
    // Each lane owns rows [lane_row_base] and [lane_row_base + 8].
    const float a0 = scalars[(row_base + lane_row_base) * 3u + 2u];
    const float a8 = scalars[(row_base + lane_row_base + 8u) * 3u + 2u];
    // Warp-uniform early exit: all 32 lanes must agree that both rows are alpha=1.0
    // before we can skip. Uses ballot so the branch is never divergent.
    if (__ballot_sync(0xffffffffu, a0 == 1.0f & a8 == 1.0f) == 0xffffffffu) return;
#pragma unroll
    for (unsigned int element = 0u; element < Fragment::num_elements; ++element) {
        frag.x[element] *= (element & 2u) ? a8 : a0;
    }
}

#endif

// Butterfly reduction via __shfl_xor_sync delivers the full reduction to ALL
// 32 lanes (unlike __shfl_down_sync which leaves the result only in lane 0).
// Required by the WMMA prefill kernels which consume the reduced max/sum
// per-lane to compute online-softmax weights — using a down-only reduction
// here yields garbage in lanes 1..31 and silently corrupts the attention output.
__device__ __forceinline__ float aegis_warp_reduce_max(float value) {
#pragma unroll
    for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    }
    return value;
}

__device__ __forceinline__ float aegis_warp_reduce_sum(float value) {
#pragma unroll
    for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}


// =============================================================================
// Row-softmax kernel for the vision tower's bidirectional attention.
// =============================================================================
//
// Input/output: f32 scores [n_rows, n_cols] row-major. In-place.
// Optionally: pre-scale every element by `scale` (= 1/sqrt(head_dim)) before
// the softmax — saves a separate scale pass.
//
// Launch: grid = (n_rows, 1, 1), block = (256, 1, 1). One block per row;
// 256 threads cooperate to reduce row-max and row-sum via warp shuffles +
// shared memory.
//
// Algorithm:
//   max = -inf
//   for c in [tid..n_cols..256]: max = fmax(max, scores[c] * scale)
//   max = warp_max(max)
//   if lane==0: smem[warp] = max
//   __syncthreads()
//   if warp==0: max = warp_max(smem[lane])  // 8 warps for 256 threads
//   row_max = broadcast max from lane 0
//   __syncthreads()
//
//   sum = 0
//   for c in [tid..n_cols..256]:
//       e = expf(scores[c]*scale - row_max)
//       scores[c] = e
//       sum += e
//   sum = warp_sum(sum)
//   if lane==0: smem[warp] = sum
//   __syncthreads()
//   if warp==0: sum = warp_sum(smem[lane])
//   row_sum = broadcast from lane 0
//   __syncthreads()
//
//   inv = 1 / row_sum
//   for c in [tid..n_cols..256]: scores[c] *= inv

#if __CUDA_ARCH__ >= 800

extern "C" __global__
__launch_bounds__(256, 1)
void aegis_vision_row_softmax(
    float* __restrict__ scores,     // [n_rows, n_cols]
    const unsigned int n_rows,
    const unsigned int n_cols,
    const float scale               // pre-multiply by this before softmax
) {
    const unsigned int row = blockIdx.x;
    if (row >= n_rows) return;
    float* row_ptr = scores + (size_t)row * n_cols;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5u;

    __shared__ float smem[8];   // 8 warps × 1 float

    // ── Phase 1: row max over scaled scores.
    float m = -3.402823466e38f;
    for (unsigned int c = tid; c < n_cols; c += 256u) {
        m = fmaxf(m, row_ptr[c] * scale);
    }
    // Warp-reduce max.
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFFu, m, off, 32);
        m = fmaxf(m, other);
    }
    if (lane == 0) smem[warp] = m;
    __syncthreads();
    if (warp == 0) {
        float v = lane < 8u ? smem[lane] : -3.402823466e38f;
        for (int off = 4; off > 0; off >>= 1) {
            float other = __shfl_xor_sync(0xFFu, v, off, 32);
            v = fmaxf(v, other);
        }
        if (lane == 0) smem[0] = v;
    }
    __syncthreads();
    const float row_max = smem[0];

    // ── Phase 2: exp + write back + accumulate sum.
    float s = 0.0f;
    for (unsigned int c = tid; c < n_cols; c += 256u) {
        float e = expf(row_ptr[c] * scale - row_max);
        row_ptr[c] = e;
        s += e;
    }
    // Warp-reduce sum.
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_xor_sync(0xFFFFFFFFu, s, off, 32);
    }
    if (lane == 0) smem[warp] = s;
    __syncthreads();
    if (warp == 0) {
        float v = lane < 8u ? smem[lane] : 0.0f;
        for (int off = 4; off > 0; off >>= 1) {
            v += __shfl_xor_sync(0xFFu, v, off, 32);
        }
        if (lane == 0) smem[0] = v;
    }
    __syncthreads();
    const float inv = 1.0f / smem[0];

    // ── Phase 3: normalize.
    for (unsigned int c = tid; c < n_cols; c += 256u) {
        row_ptr[c] *= inv;
    }
}

// ─────────────────────────────────────────────────────────────────────────
// In-place row softmax on BF16 storage (compute in F32). Replaces the
// BF16→F32→softmax→F32→BF16 round-trip that the BF16 attention path
// otherwise needs. Same launch shape as the F32 variant: grid (n_rows),
// block 256.
//
// BF16 ↔ F32 round-trip per element: high 16 bits of an IEEE-754 binary32
// is exactly bfloat16. So `f32 = (u32)bf16 << 16` and `bf16 = f32 >> 16
// (with RNE)` are bit-exact for finite values.
// ─────────────────────────────────────────────────────────────────────────
__device__ __forceinline__ float bf16_to_f32_dev(unsigned short b) {
    unsigned int u = ((unsigned int)b) << 16;
    return __int_as_float((int)u);
}
__device__ __forceinline__ unsigned short f32_to_bf16_rne(float f) {
    unsigned int u = __float_as_int(f);
    // Round-to-nearest-even: add 0x7FFF + LSB-of-truncated-mantissa.
    unsigned int rounding_bias = 0x7FFFu + ((u >> 16) & 1u);
    u += rounding_bias;
    return (unsigned short)(u >> 16);
}

extern "C" __global__
__launch_bounds__(256, 1)
void aegis_vision_row_softmax_bf16(
    unsigned short* __restrict__ scores,   // [n_rows, n_cols] BF16 in-place
    const unsigned int n_rows,
    const unsigned int n_cols,
    const float scale
) {
    const unsigned int row = blockIdx.x;
    if (row >= n_rows) return;
    unsigned short* row_ptr = scores + (size_t)row * n_cols;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5u;

    __shared__ float smem[8];

    // Phase 1: row max (reads BF16, compares in F32).
    float m = -3.402823466e38f;
    for (unsigned int c = tid; c < n_cols; c += 256u) {
        float v = bf16_to_f32_dev(row_ptr[c]) * scale;
        m = fmaxf(m, v);
    }
    for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFFu, m, off, 32);
        m = fmaxf(m, other);
    }
    if (lane == 0) smem[warp] = m;
    __syncthreads();
    if (warp == 0) {
        float v = lane < 8u ? smem[lane] : -3.402823466e38f;
        for (int off = 4; off > 0; off >>= 1) {
            float other = __shfl_xor_sync(0xFFu, v, off, 32);
            v = fmaxf(v, other);
        }
        if (lane == 0) smem[0] = v;
    }
    __syncthreads();
    const float row_max = smem[0];

    // Phase 2: exp + write back BF16 + accumulate sum in F32.
    float s = 0.0f;
    for (unsigned int c = tid; c < n_cols; c += 256u) {
        float v = bf16_to_f32_dev(row_ptr[c]) * scale;
        float e = expf(v - row_max);
        row_ptr[c] = f32_to_bf16_rne(e);
        s += e;
    }
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_xor_sync(0xFFFFFFFFu, s, off, 32);
    }
    if (lane == 0) smem[warp] = s;
    __syncthreads();
    if (warp == 0) {
        float v = lane < 8u ? smem[lane] : 0.0f;
        for (int off = 4; off > 0; off >>= 1) {
            float other = __shfl_xor_sync(0xFFu, v, off, 32);
            v += other;
        }
        if (lane == 0) smem[0] = v;
    }
    __syncthreads();
    const float inv = 1.0f / smem[0];

    // Phase 3: normalize. Read back the exp(BF16), multiply by inv, store.
    for (unsigned int c = tid; c < n_cols; c += 256u) {
        float e = bf16_to_f32_dev(row_ptr[c]);
        row_ptr[c] = f32_to_bf16_rne(e * inv);
    }
}

#endif  // __CUDA_ARCH__ >= 800

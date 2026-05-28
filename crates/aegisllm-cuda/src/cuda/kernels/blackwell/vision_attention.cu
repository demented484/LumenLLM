// =============================================================================
// Bidirectional (non-causal) attention kernel for the Gemma-4 vision tower.
// =============================================================================
//
// One launch per layer's attention. Inputs:
//   q, k, v : f32 [n_tok, n_heads, head_dim]  row-major
//   out     : f32 [n_tok, n_heads, head_dim]  row-major (output)
//
// Per-token softmax:
//   scores[i, j] = (Q[i] · K[j]) * scale         for j in 0..n_tok
//   p[i, j]      = softmax_j(scores[i, j])
//   out[i]       = sum_j p[i, j] * V[j]
//
// Grid:  (n_heads, n_tok, 1)  — one block per (head, query_row).
// Block: (256, 1, 1)          — threads cooperate to compute the row.
//
// Algorithm:
//   1. Load Q[q_row, head, :] into registers (head_dim ≤ 256, so each thread
//      holds head_dim/256 ≈ 1 element for head_dim=72).
//   2. Phase A — compute scores[q_row, :] into a dynamically-allocated shared
//      array of size n_tok floats:
//        for kv in [tid..n_tok..256]:
//          dot = sum over d of (Q[d] * K[kv, head, d])   // online dot
//          scores[kv] = dot * scale
//      Cooperative loop: each thread reads its threads' Q values via the
//      lane-strided dot.
//   3. __syncthreads()
//   4. Phase B — row-softmax in shared scores[] in-place (max → exp → sum →
//      normalize). 256 threads cooperate via warp shuffles + smem reductions.
//   5. __syncthreads()
//   6. Phase C — output[q_row, head, :] = sum_kv scores[kv] * V[kv, head, :]
//      Per-thread accumulator; one block of 256 threads splits the head_dim
//      across lanes if head_dim >= 256, otherwise threads chunk over kv.
//      We use the latter: each thread accumulates one chunk of kv positions
//      then warp-reduces into a final output element.

#if __CUDA_ARCH__ >= 800

extern "C" __global__
__launch_bounds__(256, 1)
void aegis_vision_bidi_attn(
    const float* __restrict__ q,   // [n_tok, n_heads, head_dim]
    const float* __restrict__ k,
    const float* __restrict__ v,
    const unsigned int n_tok,
    const unsigned int n_heads,
    const unsigned int head_dim,
    const float scale,             // 1/sqrt(head_dim) baked in
    float*       __restrict__ out  // [n_tok, n_heads, head_dim]
) {
    const unsigned int head  = blockIdx.x;
    const unsigned int q_row = blockIdx.y;
    if (head >= n_heads || q_row >= n_tok) return;

    const unsigned int tid  = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5u;
    const unsigned int nwarps = 8u;

    const unsigned int h = n_heads;
    const unsigned int d = head_dim;

    // Dynamic shared: scores[n_tok] + reduction scratch[8 warps] + Q[d].
    // Use the file-shared extern smem name (declared as unsigned char in
    // another TU) reinterpreted as floats — avoids redeclaration conflict
    // in the concatenated kernel translation unit.
    extern __shared__ __align__(16) unsigned char smem[];
    float* scores  = reinterpret_cast<float*>(smem);        // [n_tok]
    float* warpred = scores + n_tok;                        // [8]
    float* q_buf   = warpred + 8;                           // [d]

    // ── Load Q[q_row, head, :] into shared (small: head_dim=72).
    for (unsigned int dd = tid; dd < d; dd += 256u) {
        q_buf[dd] = q[((size_t)q_row * h + head) * d + dd];
    }
    __syncthreads();

    // ── Phase A: scores[kv] = (Q · K[kv]) * scale  for kv in [0, n_tok).
    // Each thread strides over kv positions; for each kv it computes the
    // full dot over head_dim (≤ 256 elems) sequentially.
    for (unsigned int kv = tid; kv < n_tok; kv += 256u) {
        const float* kp = k + ((size_t)kv * h + head) * d;
        float dot = 0.0f;
        // Manual unroll by 4 for head_dim multiples (Gemma-4 has hd=72).
        unsigned int dd = 0;
        for (; dd + 3 < d; dd += 4) {
            dot += q_buf[dd  ] * kp[dd  ];
            dot += q_buf[dd+1] * kp[dd+1];
            dot += q_buf[dd+2] * kp[dd+2];
            dot += q_buf[dd+3] * kp[dd+3];
        }
        for (; dd < d; ++dd) {
            dot += q_buf[dd] * kp[dd];
        }
        scores[kv] = dot * scale;
    }
    __syncthreads();

    // ── Phase B: row-softmax in place.
    // Step 1: row max via per-thread strided scan + warp/cross-warp reduce.
    float m = -3.402823466e38f;
    for (unsigned int kv = tid; kv < n_tok; kv += 256u) {
        m = fmaxf(m, scores[kv]);
    }
    for (int off = 16; off > 0; off >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, off, 32));
    }
    if (lane == 0) warpred[warp] = m;
    __syncthreads();
    if (warp == 0) {
        float v_ = lane < nwarps ? warpred[lane] : -3.402823466e38f;
        for (int off = 4; off > 0; off >>= 1) {
            v_ = fmaxf(v_, __shfl_xor_sync(0xFFu, v_, off, 32));
        }
        if (lane == 0) warpred[0] = v_;
    }
    __syncthreads();
    const float row_max = warpred[0];

    // Step 2: exp + accumulate row_sum.
    float s = 0.0f;
    for (unsigned int kv = tid; kv < n_tok; kv += 256u) {
        float e = expf(scores[kv] - row_max);
        scores[kv] = e;
        s += e;
    }
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_xor_sync(0xFFFFFFFFu, s, off, 32);
    }
    if (lane == 0) warpred[warp] = s;
    __syncthreads();
    if (warp == 0) {
        float v_ = lane < nwarps ? warpred[lane] : 0.0f;
        for (int off = 4; off > 0; off >>= 1) {
            v_ += __shfl_xor_sync(0xFFu, v_, off, 32);
        }
        if (lane == 0) warpred[0] = v_;
    }
    __syncthreads();
    const float inv_row_sum = 1.0f / warpred[0];

    // Step 3: normalize.
    for (unsigned int kv = tid; kv < n_tok; kv += 256u) {
        scores[kv] *= inv_row_sum;
    }
    __syncthreads();

    // ── Phase C: out[q_row, head, dd] = sum_kv scores[kv] * V[kv, head, dd]
    // for each output element dd in 0..head_dim. Each thread handles one
    // dd (with head_dim=72 ≤ 256). Threads with tid >= head_dim idle this
    // phase.
    if (tid < d) {
        float acc = 0.0f;
        for (unsigned int kv = 0; kv < n_tok; ++kv) {
            acc += scores[kv] * v[((size_t)kv * h + head) * d + tid];
        }
        out[((size_t)q_row * h + head) * d + tid] = acc;
    }
}

#endif  // __CUDA_ARCH__ >= 800

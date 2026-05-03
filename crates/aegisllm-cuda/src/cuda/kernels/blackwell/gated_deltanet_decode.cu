// Gated DeltaNet single-token decode step (Phase 6.3).
//
// Implements the delta-rule state update for one query token:
//   beta_eff = beta / dot(k, k)
//   delta[i]  = v[i] - dot(S[i, :], k)      for each row i in [0, d_v)
//   S[i, j]  += beta_eff * delta[i] * k[j]  (outer-product update)
//   y[i]      = dot(S_new[i, :], q)          (readout)
//
// Each thread block handles one attention head. blockDim.x must be equal to
// the next power-of-2 that is >= d_k and <= 256.  If d_k < blockDim.x some
// thread lanes are inactive (their k/q values are treated as 0).
//
// Grid:  (num_heads, 1, 1)
// Block: (block_k, 1, 1)   block_k = next_pow2(d_k), clamped to 256
// Smem:  block_k * sizeof(float) bytes for reductions
extern "C" __global__ void aegis_gated_deltanet_decode(
    float* __restrict__ state,          // [num_heads, d_v, d_k]  in-place
    const float* __restrict__ q,        // [num_heads, d_k]
    const float* __restrict__ k,        // [num_heads, d_k]
    const float* __restrict__ v,        // [num_heads, d_v]
    const float* __restrict__ beta,     // [num_heads]  scalar gate per head
    float* __restrict__ output,         // [num_heads, d_v]
    const unsigned int d_k,
    const unsigned int d_v
) {
    const unsigned int head       = blockIdx.x;
    const unsigned int tid        = threadIdx.x;
    const unsigned int block_size = blockDim.x;

    float* __restrict__ S = state + (unsigned long long)head * d_v * d_k;
    const float* __restrict__ q_h = q + head * d_k;
    const float* __restrict__ k_h = k + head * d_k;
    const float* __restrict__ v_h = v + head * d_v;
    float* __restrict__ out_h = output + head * d_v;

    const float beta_h = beta[head];

    extern __shared__ float gdn_smem[];

    // Per-thread k and q values (0 for inactive lanes).
    const float kj = (tid < d_k) ? k_h[tid] : 0.0f;
    const float qj = (tid < d_k) ? q_h[tid] : 0.0f;

    // Phase 1: k_norm_sq = dot(k, k)
    gdn_smem[tid] = kj * kj;
    __syncthreads();
    for (unsigned int s = block_size >> 1u; s > 0u; s >>= 1u) {
        if (tid < s) gdn_smem[tid] += gdn_smem[tid + s];
        __syncthreads();
    }
    const float k_norm_sq = gdn_smem[0];
    const float beta_eff  = (k_norm_sq > 1.0e-8f) ? (beta_h / k_norm_sq) : 0.0f;
    __syncthreads();

    // Phase 2: per-row update and output projection.
    for (unsigned int i = 0u; i < d_v; i++) {
        const float s_ij = (tid < d_k) ? S[i * d_k + tid] : 0.0f;

        // Sk[i] = dot(S[i, :], k)
        gdn_smem[tid] = s_ij * kj;
        __syncthreads();
        for (unsigned int s = block_size >> 1u; s > 0u; s >>= 1u) {
            if (tid < s) gdn_smem[tid] += gdn_smem[tid + s];
            __syncthreads();
        }
        const float sk_i    = gdn_smem[0];
        const float delta_i = v_h[i] - sk_i;
        __syncthreads();

        // Update S[i, tid] in-place.
        const float s_new_ij = s_ij + beta_eff * delta_i * kj;
        if (tid < d_k) {
            S[i * d_k + tid] = s_new_ij;
        }

        // y[i] = dot(S_new[i, :], q) via reduction.
        gdn_smem[tid] = s_new_ij * qj;
        __syncthreads();
        for (unsigned int s = block_size >> 1u; s > 0u; s >>= 1u) {
            if (tid < s) gdn_smem[tid] += gdn_smem[tid + s];
            __syncthreads();
        }
        if (tid == 0u) {
            out_h[i] = gdn_smem[0];
        }
        __syncthreads();
    }
}

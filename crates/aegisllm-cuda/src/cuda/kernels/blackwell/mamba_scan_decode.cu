// Mamba selective-scan single-token decode step (Phase 7.3).
//
// Implements the SSM state update for one token:
//   A_bar[s] = exp(dt[h] * A_log[h, s])          -- discrete decay per state dim
//   B_bar[s] = dt[h] * B[h, s]                   -- input expansion
//   state[h, s, j] = A_bar[s] * state[h, s, j] + B_bar[s] * x[h, j]
//   y[h, j] = D[h, j] * x[h, j] + sum_s(C[h, s] * state[h, s, j])
//
// Each thread handles one column j of the per-head state matrix, iterating
// over d_state rows independently (no shared memory or barriers needed).
//
// Grid:  (num_heads, 1, 1)
// Block: (head_dim, 1, 1)   head_dim = d_inner / num_heads
// Smem:  0 bytes
//
// A_log convention: A_log[h, s] = log(-A[h, s]) where A < 0 (standard Mamba),
// so A_bar = exp(dt * A_log) ∈ (0, 1) for positive dt and negative A.
extern "C" __global__ void aegis_mamba_scan_decode(
    float* __restrict__ state,          // [num_heads, d_state, head_dim]  in-place
    const float* __restrict__ x,        // [num_heads, head_dim]  SSM input
    const float* __restrict__ dt,       // [num_heads]  delta time step (after softplus)
    const float* __restrict__ a_log,    // [num_heads, d_state]  log(-A) decay param
    const float* __restrict__ B,        // [num_heads, d_state]  input-dep B projection
    const float* __restrict__ C,        // [num_heads, d_state]  output-dep C projection
    const float* __restrict__ D,        // [num_heads, head_dim]  skip connection
    float* __restrict__ output,         // [num_heads, head_dim]
    const unsigned int d_state,
    const unsigned int head_dim
) {
    const unsigned int head = blockIdx.x;
    const unsigned int j    = threadIdx.x;

    if (j >= head_dim) return;

    float* __restrict__ S_h   = state + (unsigned long long)head * d_state * head_dim;
    const float* __restrict__ x_h    = x     + head * head_dim;
    const float* __restrict__ a_h    = a_log + head * d_state;
    const float* __restrict__ B_h    = B     + head * d_state;
    const float* __restrict__ C_h    = C     + head * d_state;
    const float* __restrict__ D_h    = D     + head * head_dim;
    float* __restrict__ out_h = output + head * head_dim;

    const float dt_h = dt[head];
    const float x_j  = x_h[j];

    float y_j = D_h[j] * x_j;

    for (unsigned int s = 0u; s < d_state; s++) {
        const float a_bar_s  = expf(dt_h * a_h[s]);
        const float b_bar_s  = dt_h * B_h[s];
        const float new_state = a_bar_s * S_h[s * head_dim + j] + b_bar_s * x_j;
        S_h[s * head_dim + j] = new_state;
        y_j += C_h[s] * new_state;
    }

    out_h[j] = y_j;
}

// Gated DeltaNet single-token decode step (Qwen3-Next / Qwen3.5-3.6).
//
// Implements the GATED delta-rule recurrence for one query token. Inputs q,k
// are expected ALREADY L2-normalized over d_k (and q pre-scaled by 1/sqrt(d_k))
// and beta = sigmoid(b), g = -exp(A_log)*softplus(a+dt_bias) precomputed by the
// caller. State layout here is S[d_v, d_k] (the transpose of the CPU reference's
// S[d_k, d_v]; the recurrence is symmetric under transpose).
//
//   S        *= exp(g)                          (decay — THE gate; was missing)
//   delta[i]  = v[i] - dot(S[i, :], k)           for each row i in [0, d_v)
//   S[i, j]  += beta * delta[i] * k[j]           (outer-product update; beta
//                                                 direct — k is unit-norm, so the
//                                                 old beta/dot(k,k) is dropped)
//   y[i]      = dot(S_new[i, :], q)              (readout from updated state)
//
// Each thread block handles one (value) head. blockDim.x = next_pow2(d_k) ≤ 256.
// Lanes with tid >= d_k are inactive (their k/q treated as 0).
//
// Grid:  (num_heads, 1, 1)
// Block: (block_k, 1, 1)   block_k = next_pow2(d_k), clamped to 256
// Smem:  block_k * sizeof(float) bytes for reductions
// Warp-per-row formulation: one warp owns row `i` of one head (rows are
// independent across d_v and heads). The two d_k-length dot products (sk_i and
// y_i) are warp-shuffle reductions — no shared memory, no __syncthreads, and
// num_heads*d_v warps fill the GPU (vs the old one-block-per-head with a serial
// 128-row loop + ~1800 barriers/head). Math is identical to the reference.
// d_k ≤ 256 (power of 2) ⇒ ≤ 8 j-values per lane (cached in registers).
extern "C" __global__ void aegis_gated_deltanet_decode(
    float* __restrict__ state,          // [num_heads, d_v, d_k]  in-place
    const float* __restrict__ q,        // [num_heads, d_k]  (normed + scaled)
    const float* __restrict__ k,        // [num_heads, d_k]  (normed)
    const float* __restrict__ v,        // [num_heads, d_v]
    const float* __restrict__ beta,     // [num_heads]  sigmoid(b)
    const float* __restrict__ g,        // [num_heads]  -exp(A_log)*softplus(...)
    float* __restrict__ output,         // [num_heads, d_v]
    const unsigned int num_heads,
    const unsigned int d_k,
    const unsigned int d_v
) {
    const unsigned int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5u;
    const unsigned int lane    = threadIdx.x & 31u;
    const unsigned int total_rows = num_heads * d_v;
    if (warp_id >= total_rows) return;

    const unsigned int head = warp_id / d_v;
    const unsigned int row  = warp_id % d_v;

    float* __restrict__ S = state + ((unsigned long long)head * d_v + row) * d_k; // S[head, row, :]
    const float* __restrict__ q_h = q + (unsigned long long)head * d_k;
    const float* __restrict__ k_h = k + (unsigned long long)head * d_k;

    const float beta_h  = beta[head];
    const float decay_h = expf(g[head]);                       // exp(g) ∈ (0, 1]
    const float v_ir    = v[(unsigned long long)head * d_v + row];

    // Decay first (gate forgets old state); cache the decayed row in registers
    // and accumulate sk_i = dot(decayed S[i,:], k) in the same pass.
    float s_dec[8];
    float sk = 0.0f;
    unsigned int nj = 0u;
    for (unsigned int j = lane; j < d_k; j += 32u) {
        float sd = S[j] * decay_h;
        s_dec[nj] = sd;
        sk += sd * k_h[j];
        nj++;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sk += __shfl_down_sync(0xffffffffu, sk, off);
    sk = __shfl_sync(0xffffffffu, sk, 0);                      // broadcast to all lanes
    const float delta_i = (v_ir - sk) * beta_h;

    // Update S[i, j] in-place (decayed base + outer-product) and read out y_i.
    float y = 0.0f;
    nj = 0u;
    for (unsigned int j = lane; j < d_k; j += 32u) {
        float s_new = s_dec[nj] + delta_i * k_h[j];
        S[j] = s_new;
        y += s_new * q_h[j];
        nj++;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) y += __shfl_down_sync(0xffffffffu, y, off);
    if (lane == 0u) output[(unsigned long long)head * d_v + row] = y;
}

// GDN per-token preprocessing: L2-normalize q,k over d_k and GQA-expand from
// n_k key heads to n_v value heads (each key head feeds n_v/n_k value heads).
// q is additionally scaled by 1/sqrt(d_k). Inputs are the post-conv q,k.
//
//   kh        = h / (n_v / n_k)                 (source key head for value head h)
//   k_out[h]  = k_in[kh] / ||k_in[kh]||_2
//   q_out[h]  = q_in[kh] / ||q_in[kh]||_2 * (1/sqrt(d_k))
//
// Grid:  (n_v, 1, 1)   one block per value head
// Block: (block_k, 1, 1)   block_k = next_pow2(d_k) ≤ 256
// Smem:  block_k * sizeof(float)
extern "C" __global__ void aegis_gdn_qk_norm_expand(
    const float* __restrict__ q_in,     // [n_k, d_k]
    const float* __restrict__ k_in,     // [n_k, d_k]
    float* __restrict__ q_out,          // [n_v, d_k]  normed + scaled + expanded
    float* __restrict__ k_out,          // [n_v, d_k]  normed + expanded
    const unsigned int n_k,
    const unsigned int d_k,
    const unsigned int expand           // n_v / n_k
) {
    const unsigned int h          = blockIdx.x;       // value head
    const unsigned int kh         = h / expand;        // source key head
    const unsigned int tid        = threadIdx.x;
    const unsigned int block_size = blockDim.x;
    const float eps               = 1.0e-6f;

    const float qv = (tid < d_k) ? q_in[kh * d_k + tid] : 0.0f;
    const float kv = (tid < d_k) ? k_in[kh * d_k + tid] : 0.0f;

    extern __shared__ float sm[];

    // ||q||^2
    sm[tid] = qv * qv;
    __syncthreads();
    for (unsigned int s = block_size >> 1u; s > 0u; s >>= 1u) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    const float inv_q = rsqrtf(sm[0] + eps) * rsqrtf((float)d_k);
    __syncthreads();

    // ||k||^2
    sm[tid] = kv * kv;
    __syncthreads();
    for (unsigned int s = block_size >> 1u; s > 0u; s >>= 1u) {
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    const float inv_k = rsqrtf(sm[0] + eps);

    if (tid < d_k) {
        q_out[h * d_k + tid] = qv * inv_q;
        k_out[h * d_k + tid] = kv * inv_k;
    }
}

// GDN gate computation (per value head, n_v scalars):
//   beta = sigmoid(b)
//   g    = -exp(A_log) * softplus(a + dt_bias)        (negative → exp(g) decays)
// softplus stable: max(x,0) + log1p(exp(-|x|)).
//
// Grid: (1,1,1)  Block: (n_v,1,1)   (n_v small, e.g. 32)
extern "C" __global__ void aegis_gdn_gate(
    const float* __restrict__ b,        // [n_v]
    const float* __restrict__ a,        // [n_v]
    const float* __restrict__ a_log,    // [n_v]
    const float* __restrict__ dt_bias,  // [n_v]
    float* __restrict__ beta_out,       // [n_v]
    float* __restrict__ g_out,          // [n_v]
    const unsigned int n_v
) {
    const unsigned int h = threadIdx.x;
    if (h >= n_v) return;
    beta_out[h] = 1.0f / (1.0f + expf(-b[h]));
    const float x  = a[h] + dt_bias[h];
    const float sp = fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
    g_out[h] = -expf(a_log[h]) * sp;
}

// GDN output gated RMSNorm (Qwen3-Next Qwen3NextRMSNormGated), per value head.
// Exact HF order (modeling_qwen3_next.py:73-80): normalize o FIRST, scale by
// PLAIN weight, THEN gate by silu(z):
//   normed = o * rsqrt(mean(o^2) + eps)
//   out    = weight * normed * silu(z)
// `weight` is shared across heads ([d_v]). Operates over d_v per head.
//
// Grid:  (n_v, 1, 1)   one block per value head
// Block: (block_v, 1, 1)   block_v = next_pow2(d_v) ≤ 1024
// Smem:  block_v * sizeof(float)
extern "C" __global__ void aegis_gdn_gated_rmsnorm(
    const float* __restrict__ o,        // [n_v, d_v]
    const float* __restrict__ z,        // [n_v, d_v]
    const float* __restrict__ weight,   // [d_v]
    float* __restrict__ out,            // [n_v, d_v]
    const unsigned int d_v,
    const float eps
) {
    const unsigned int h          = blockIdx.x;
    const unsigned int tid        = threadIdx.x;
    const unsigned int block_size = blockDim.x;
    const unsigned int base       = h * d_v;

    const float ov = (tid < d_v) ? o[base + tid] : 0.0f;
    const float zv = (tid < d_v) ? z[base + tid] : 0.0f;

    // mean of squares over the UN-gated o (HF normalizes o, not o·gate).
    extern __shared__ float rms_sm[];
    rms_sm[tid] = ov * ov;
    __syncthreads();
    for (unsigned int s = block_size >> 1u; s > 0u; s >>= 1u) {
        if (tid < s) rms_sm[tid] += rms_sm[tid + s];
        __syncthreads();
    }
    const float inv    = rsqrtf(rms_sm[0] / (float)d_v + eps);
    const float silu_z = zv / (1.0f + expf(-zv));
    if (tid < d_v) {
        out[base + tid] = weight[tid] * (ov * inv) * silu_z;
    }
}

// GDN streaming (decode) depthwise causal conv1d + SiLU for ONE token.
// Per channel c, the K-tap window is [conv_state[c,0..K-2], x_new[c]]:
//   acc   = Σ_{t<K-1} w[c,t]·conv_state[c,t] + w[c,K-1]·x_new[c]   (+ bias)
//   out[c] = silu(acc)
// Then the conv state is shifted left by one and x_new[c] appended, so the
// next token sees the updated K-1 history. conv_weight is [C, 1, K] → w[c*K+t];
// conv_state is [C, K-1] → s[c*(K-1)+t].
//
// Grid:  (ceil(C / 256), 1, 1)   Block: (256, 1, 1)   one thread per channel.
extern "C" __global__ void aegis_gdn_conv1d_decode(
    const float* __restrict__ x_new,        // [C]
    float* __restrict__ conv_state,         // [C, K-1]  updated in-place
    const float* __restrict__ conv_weight,  // [C, 1, K]
    float* __restrict__ out,                // [C]
    const unsigned int channels,
    const unsigned int kernel
) {
    const unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const unsigned int km1 = kernel - 1u;
    const float xc = x_new[c];

    float acc = 0.0f;   // GDN conv1d has no bias
    // taps 0..K-2 read the history; tap K-1 reads the new token.
    for (unsigned int t = 0u; t < km1; t++) {
        acc += conv_weight[c * kernel + t] * conv_state[c * km1 + t];
    }
    acc += conv_weight[c * kernel + km1] * xc;
    // SiLU
    out[c] = acc / (1.0f + expf(-acc));

    // shift history left, append new token.
    for (unsigned int t = 0u; t + 1u < km1; t++) {
        conv_state[c * km1 + t] = conv_state[c * km1 + t + 1u];
    }
    if (km1 > 0u) {
        conv_state[c * km1 + (km1 - 1u)] = xc;
    }
}

// Qwen3-Next attention output gate: q_proj outputs [num_heads, 2*head_dim]
// where each head's first head_dim is the query and the second is the gate.
// De-interleave into separate contiguous query/gate buffers (each [nh, hd]).
extern "C" __global__ void aegis_deinterleave_gated_q(
    const float* __restrict__ q_full,   // [nh, 2*hd]
    float* __restrict__ query,          // [nh, hd]
    float* __restrict__ gate,           // [nh, hd]
    const unsigned int num_heads,
    const unsigned int head_dim
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = num_heads * head_dim;
    if (i >= total) return;
    const unsigned int h = i / head_dim;
    const unsigned int d = i % head_dim;
    const unsigned int base = h * 2u * head_dim;
    query[i] = q_full[base + d];
    gate[i]  = q_full[base + head_dim + d];
}

// De-interleave Qwen3-Next in_proj_qkv from the per-key-head packed layout
// (HF fix_query_key_value_ordering): the projection output is grouped by key
// head as [kh: q(hd_k), k(hd_k), v(expand*hd_v)] for kh in [0, n_k). Reorder to
// the contiguous [all_q (n_k*hd_k) | all_k (n_k*hd_k) | all_v (n_v*hd_v)] layout
// that the depthwise conv1d weight (HF conv_dim channel order) expects.
//   expand = n_v / n_k ; per-head stride = 2*hd_k + expand*hd_v
extern "C" __global__ void aegis_gdn_deinterleave_qkv(
    const float* __restrict__ in_packed,   // [n_k * (2*hd_k + expand*hd_v)]
    float* __restrict__ out,                // [n_k*hd_k + n_k*hd_k + n_v*hd_v]
    const unsigned int n_k,
    const unsigned int hd_k,
    const unsigned int hd_v,
    const unsigned int expand
) {
    const unsigned int stride = 2u * hd_k + expand * hd_v;
    const unsigned int total  = n_k * stride;
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const unsigned int kh  = idx / stride;
    const unsigned int off = idx % stride;
    const float val = in_packed[idx];
    const unsigned int all_q = n_k * hd_k;
    const unsigned int all_k = n_k * hd_k;
    if (off < hd_k) {
        out[kh * hd_k + off] = val;                                   // q
    } else if (off < 2u * hd_k) {
        out[all_q + kh * hd_k + (off - hd_k)] = val;                  // k
    } else {
        out[all_q + all_k + kh * (expand * hd_v) + (off - 2u * hd_k)] = val; // v
    }
}

// HF/GPT-NeoX partial RoPE (Qwen3-Next): rotate ONLY the first `rotary_dim`
// dims of each head, with the half-split taken WITHIN the rotary block (pairs
// (i, i+rotary_dim/2)) and the inv-freq divisor = rotary_dim. Dims
// [rotary_dim, head_dim) pass through unchanged. This differs from the engine's
// default p-RoPE (which pairs (i, i+head_dim/2) with divisor head_dim).
//
// Grid: (num_heads, 1, 1)   Block: (rotary_dim/2, 1, 1)
extern "C" __global__ void aegis_apply_rope_ptr_neox_partial(
    float* __restrict__ values,
    const unsigned int* __restrict__ p_position,
    const unsigned int num_heads,
    const unsigned int head_dim,
    const float theta,
    const unsigned int rotary_dim
) {
    const unsigned int head = blockIdx.x;
    const unsigned int i    = threadIdx.x;
    const unsigned int half = rotary_dim / 2u;
    if (head >= num_heads || i >= half) return;
    float* row = values + (size_t)head * head_dim;
    const float position = (float)(*p_position);
    const float freq = 1.0f / powf(theta, (float)(2u * i) / (float)rotary_dim);
    float sinv, cosv;
    sincosf(position * freq, &sinv, &cosv);
    const float x0 = row[i];
    const float x1 = row[i + half];
    row[i]        = x0 * cosv - x1 * sinv;
    row[i + half] = x1 * cosv + x0 * sinv;
}

// Batched HF/GPT-NeoX partial RoPE (Qwen3-Next full-attention prefill): same
// math as aegis_apply_rope_ptr_neox_partial but over a whole T-token chunk,
// where each token carries its own position from p_positions[t].
// Grid: (num_heads, seq_len, 1)   Block: (rotary_dim/2, 1, 1)
// `values` layout: [T, num_heads, head_dim] (row-major).
extern "C" __global__ void aegis_apply_rope_neox_partial_batched(
    float* __restrict__ values,
    const unsigned int* __restrict__ p_positions,   // [T]
    const unsigned int num_heads,
    const unsigned int head_dim,
    const float theta,
    const unsigned int rotary_dim
) {
    const unsigned int head = blockIdx.x;
    const unsigned int t    = blockIdx.y;
    const unsigned int i    = threadIdx.x;
    const unsigned int half = rotary_dim / 2u;
    if (head >= num_heads || i >= half) return;
    float* row = values + (((unsigned long long)t * num_heads) + head) * head_dim;
    const float position = (float)(p_positions[t]);
    const float freq = 1.0f / powf(theta, (float)(2u * i) / (float)rotary_dim);
    float sinv, cosv;
    sincosf(position * freq, &sinv, &cosv);
    const float x0 = row[i];
    const float x1 = row[i + half];
    row[i]        = x0 * cosv - x1 * sinv;
    row[i + half] = x1 * cosv + x0 * sinv;
}

// Multiply x by sigmoid(g) elementwise (Qwen3-Next attention output gating,
// applied to the attention context before o_proj).
extern "C" __global__ void aegis_sigmoid_gate_mul_f32(
    float* __restrict__ x,
    const float* __restrict__ g,
    const unsigned int n
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gv = g[i];
    x[i] = x[i] * (1.0f / (1.0f + expf(-gv)));
}

// ============================================================================
// Batched (chunked-prefill) GDN kernels. These process a whole T-token chunk in
// one launch, mirroring the single-token decode kernels above. The matmuls
// (in_proj/out_proj) are batched separately (dequant→cuBLASLt). Together with
// these, GDN prefill goes from token-by-token to chunk-batched.
// ============================================================================

// Strided 2D copy: dst[r, 0..copy_len) = src[r, src_off .. src_off+copy_len).
// Used to split the contiguous [T, conv_dim] conv output into q/k/v buffers.
extern "C" __global__ void aegis_strided_copy_2d(
    const float* __restrict__ src,
    float* __restrict__ dst,
    const unsigned int rows,
    const unsigned int copy_len,
    const unsigned int src_stride,
    const unsigned int dst_stride,
    const unsigned int src_off
) {
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long total = (unsigned long long)rows * copy_len;
    if (i >= total) return;
    const unsigned int r = (unsigned int)(i / copy_len);
    const unsigned int c = (unsigned int)(i % copy_len);
    dst[(unsigned long long)r * dst_stride + c] = src[(unsigned long long)r * src_stride + src_off + c];
}

// Batched GDN delta-rule over a T-token chunk. One warp per (value head, d_v row);
// loops over the T tokens sequentially through the recurrence (S carries forward),
// so the math is IDENTICAL to aegis_gated_deltanet_decode applied T times. State
// is updated in place to the post-chunk value. d_k ≤ 256 ⇒ ≤8 j per lane.
extern "C" __global__ void aegis_gated_deltanet_prefill(
    float* __restrict__ state,          // [n_v, d_v, d_k] in-place
    const float* __restrict__ q,        // [T, n_v, d_k]  (normed + scaled + expanded)
    const float* __restrict__ k,        // [T, n_v, d_k]  (normed + expanded)
    const float* __restrict__ v,        // [T, n_v, d_v]
    const float* __restrict__ beta,     // [T, n_v]
    const float* __restrict__ g,        // [T, n_v]
    float* __restrict__ output,         // [T, n_v, d_v]
    const unsigned int seq_len,
    const unsigned int num_heads,
    const unsigned int d_k,
    const unsigned int d_v
) {
    const unsigned int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5u;
    const unsigned int lane    = threadIdx.x & 31u;
    const unsigned int total_rows = num_heads * d_v;
    if (warp_id >= total_rows) return;
    const unsigned int head = warp_id / d_v;
    const unsigned int row  = warp_id % d_v;

    float* __restrict__ S = state + ((unsigned long long)head * d_v + row) * d_k;
    float s_reg[8];
    unsigned int nj = 0u;
    for (unsigned int j = lane; j < d_k; j += 32u) { s_reg[nj] = S[j]; nj++; }

    for (unsigned int t = 0u; t < seq_len; t++) {
        const float beta_h  = beta[(unsigned long long)t * num_heads + head];
        const float decay_h = expf(g[(unsigned long long)t * num_heads + head]);
        const float v_ir    = v[(((unsigned long long)t * num_heads) + head) * d_v + row];
        const float* __restrict__ q_t = q + (((unsigned long long)t * num_heads) + head) * d_k;
        const float* __restrict__ k_t = k + (((unsigned long long)t * num_heads) + head) * d_k;

        float sk = 0.0f;
        nj = 0u;
        for (unsigned int j = lane; j < d_k; j += 32u) {
            s_reg[nj] *= decay_h;                 // decay first
            sk += s_reg[nj] * k_t[j];
            nj++;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) sk += __shfl_down_sync(0xffffffffu, sk, off);
        sk = __shfl_sync(0xffffffffu, sk, 0);
        const float delta_i = (v_ir - sk) * beta_h;

        float y = 0.0f;
        nj = 0u;
        for (unsigned int j = lane; j < d_k; j += 32u) {
            s_reg[nj] += delta_i * k_t[j];        // S_new = decayed + delta·k
            y += s_reg[nj] * q_t[j];
            nj++;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) y += __shfl_down_sync(0xffffffffu, y, off);
        if (lane == 0u) output[(((unsigned long long)t * num_heads) + head) * d_v + row] = y;
    }
    nj = 0u;
    for (unsigned int j = lane; j < d_k; j += 32u) { S[j] = s_reg[nj]; nj++; }
}

// Batched depthwise causal conv1d + SiLU over a T-token chunk. One thread per
// channel; slides the K-tap window across the T tokens using conv_state as the
// left-context, then writes the last K-1 inputs back to conv_state. K ≤ 4.
extern "C" __global__ void aegis_gdn_conv1d_prefill(
    const float* __restrict__ x,            // [T, C]
    float* __restrict__ conv_state,         // [C, K-1] in-place
    const float* __restrict__ conv_weight,  // [C, 1, K]
    float* __restrict__ out,                // [T, C]
    const unsigned int seq_len,
    const unsigned int channels,
    const unsigned int kernel
) {
    const unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const unsigned int km1 = kernel - 1u;
    float win[4];                            // K ≤ 4
    for (unsigned int t = 0u; t < km1; t++) win[t] = conv_state[c * km1 + t];
    for (unsigned int t = 0u; t < seq_len; t++) {
        const float xc = x[(unsigned long long)t * channels + c];
        float acc = 0.0f;
        for (unsigned int kk = 0u; kk < km1; kk++) acc += conv_weight[c * kernel + kk] * win[kk];
        acc += conv_weight[c * kernel + km1] * xc;
        out[(unsigned long long)t * channels + c] = acc / (1.0f + expf(-acc));
        for (unsigned int kk = 0u; kk + 1u < km1; kk++) win[kk] = win[kk + 1u];
        if (km1 > 0u) win[km1 - 1u] = xc;
    }
    for (unsigned int t = 0u; t < km1; t++) conv_state[c * km1 + t] = win[t];
}

// Batched qk-norm + GQA-expand. Grid (n_v, T): blockIdx.x=value head, .y=token.
// q/k inputs are the contiguous post-conv split buffers [T, n_k, d_k].
extern "C" __global__ void aegis_gdn_qk_norm_expand_batched(
    const float* __restrict__ q_in,     // [T, n_k, d_k]
    const float* __restrict__ k_in,     // [T, n_k, d_k]
    float* __restrict__ q_out,          // [T, n_v, d_k]
    float* __restrict__ k_out,          // [T, n_v, d_k]
    const unsigned int n_k,
    const unsigned int d_k,
    const unsigned int expand
) {
    const unsigned int h   = blockIdx.x;        // value head
    const unsigned int t   = blockIdx.y;        // token
    const unsigned int n_v = gridDim.x;
    const unsigned int kh  = h / expand;
    const unsigned int tid = threadIdx.x;
    const unsigned int block_size = blockDim.x;
    const float eps = 1.0e-6f;

    const float* __restrict__ qin = q_in + (((unsigned long long)t * n_k) + kh) * d_k;
    const float* __restrict__ kin = k_in + (((unsigned long long)t * n_k) + kh) * d_k;
    const float qv = (tid < d_k) ? qin[tid] : 0.0f;
    const float kv = (tid < d_k) ? kin[tid] : 0.0f;

    extern __shared__ float sm[];
    sm[tid] = qv * qv; __syncthreads();
    for (unsigned int s = block_size >> 1u; s > 0u; s >>= 1u) { if (tid < s) sm[tid] += sm[tid + s]; __syncthreads(); }
    const float inv_q = rsqrtf(sm[0] + eps) * rsqrtf((float)d_k);
    __syncthreads();
    sm[tid] = kv * kv; __syncthreads();
    for (unsigned int s = block_size >> 1u; s > 0u; s >>= 1u) { if (tid < s) sm[tid] += sm[tid + s]; __syncthreads(); }
    const float inv_k = rsqrtf(sm[0] + eps);

    if (tid < d_k) {
        const unsigned long long o = (((unsigned long long)t * n_v) + h) * d_k + tid;
        q_out[o] = qv * inv_q;
        k_out[o] = kv * inv_k;
    }
}

// Batched gate: beta=sigmoid(b), g=-exp(A_log)*softplus(a+dt_bias), over T*n_v.
extern "C" __global__ void aegis_gdn_gate_batched(
    const float* __restrict__ b,        // [T, n_v]
    const float* __restrict__ a,        // [T, n_v]
    const float* __restrict__ a_log,    // [n_v]
    const float* __restrict__ dt_bias,  // [n_v]
    float* __restrict__ beta_out,       // [T, n_v]
    float* __restrict__ g_out,          // [T, n_v]
    const unsigned int seq_len,
    const unsigned int n_v
) {
    const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long total = (unsigned long long)seq_len * n_v;
    if (idx >= total) return;
    const unsigned int h = (unsigned int)(idx % n_v);
    beta_out[idx] = 1.0f / (1.0f + expf(-b[idx]));
    const float x  = a[idx] + dt_bias[h];
    const float sp = fmaxf(x, 0.0f) + log1pf(expf(-fabsf(x)));
    g_out[idx] = -expf(a_log[h]) * sp;
}

// Batched gated RMSNorm (norm o, ×weight, ×silu(z)). Grid (n_v, T).
extern "C" __global__ void aegis_gdn_gated_rmsnorm_batched(
    const float* __restrict__ o,        // [T, n_v, d_v]
    const float* __restrict__ z,        // [T, n_v, d_v]
    const float* __restrict__ weight,   // [d_v]
    float* __restrict__ out,            // [T, n_v, d_v]
    const unsigned int d_v,
    const float eps
) {
    const unsigned int h   = blockIdx.x;
    const unsigned int t   = blockIdx.y;
    const unsigned int n_v = gridDim.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int block_size = blockDim.x;
    const unsigned long long base = (((unsigned long long)t * n_v) + h) * d_v;

    const float ov = (tid < d_v) ? o[base + tid] : 0.0f;
    const float zv = (tid < d_v) ? z[base + tid] : 0.0f;
    extern __shared__ float rms_sm[];
    rms_sm[tid] = ov * ov; __syncthreads();
    for (unsigned int s = block_size >> 1u; s > 0u; s >>= 1u) { if (tid < s) rms_sm[tid] += rms_sm[tid + s]; __syncthreads(); }
    const float inv    = rsqrtf(rms_sm[0] / (float)d_v + eps);
    const float silu_z = zv / (1.0f + expf(-zv));
    if (tid < d_v) out[base + tid] = weight[tid] * (ov * inv) * silu_z;
}

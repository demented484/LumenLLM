// =============================================================================
// Gemma-4 audio tower (USM/Conformer) per-token CUDA kernels.
// =============================================================================
//
// The audio tower is a 12-layer Conformer encoder (Gemma-3n / Gemma-4 audio
// architecture). The heavy matmuls (Q/K/V/post projections, FFN, conv
// pointwise linears) reuse the existing cuBLASLt BF16 GEMM wrapper; only the
// genuinely audio-specific elementwise ops live here:
//
//   * depthwise causal conv1d over time (kernel_size K, left-pad K-1)
//   * GLU (split-half gate) activation
//   * per_dim_scale application to the attention query (q * q_scale * softplus(s))
//   * gradient clipping (clamp ±C) used between Conformer sub-blocks
//   * SiLU in-place — used both for the LightConv1d post-conv activation AND
//     for the ungated Macaron-FFN activation (see audio.rs::feed_forward).
//     TODO(gpu-verify): confirm the FFN is ungated SiLU and not a SwiGLU split.
//
// All buffers are f32 row-major unless stated otherwise. SM_120 (Blackwell)
// target; guarded for arch >= 800.
//
// TODO(gpu-verify): every numeric detail below is implemented from the HF
// `modeling_gemma3n.py` reference math but has NOT been GPU-validated. Cross
// check against an HF activation dump before trusting outputs.

#if __CUDA_ARCH__ >= 800

// ─────────────────────────────────────────────────────────────────────────
// GLU split-half: out[t, c] = a[t, c] * sigmoid(a[t, c + half])
// Input  `a`   : [n_frames, 2*half]  row-major f32 (concat of value || gate).
// Output `out` : [n_frames, half]    row-major f32.
// torch.nn.functional.glu(x, dim=-1): first half is the value, second half is
// the gate that is passed through sigmoid. out = first * sigmoid(second).
// Launch: grid = (n_frames, ceil(half/256)), block = 256.
// ─────────────────────────────────────────────────────────────────────────
extern "C" __global__
void aegis_audio_glu_halfsplit(
    const float* __restrict__ a,      // [n_frames, 2*half]
    float*       __restrict__ out,    // [n_frames, half]
    const unsigned int n_frames,
    const unsigned int half
) {
    const unsigned int t = blockIdx.x;
    const unsigned int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= n_frames || c >= half) return;
    const size_t base = (size_t)t * (2u * half);
    const float val  = a[base + c];
    const float gate = a[base + half + c];
    const float sig  = 1.0f / (1.0f + expf(-gate));
    out[(size_t)t * half + c] = val * sig;
}

// ─────────────────────────────────────────────────────────────────────────
// Depthwise causal conv1d over the time axis.
//   in  : [n_frames, channels]  row-major f32
//   w   : [channels, kernel]    row-major f32 (one filter per channel)
//   out : [n_frames, channels]  row-major f32
// Causal: out[t, c] = sum_{j=0..K-1} in[t - (K-1) + j, c] * w[c, j], with
// left-edge frames (t - (K-1) + j < 0) treated as zero (left padding K-1).
// This matches HF: F.pad(x, (K-1, 0)) then Conv1d(groups=channels) with no
// extra padding. The depthwise_conv1d weight is stored [channels, 1, kernel];
// we pass it flattened to [channels, kernel].
// Launch: grid = (n_frames, ceil(channels/256)), block = 256.
// ─────────────────────────────────────────────────────────────────────────
extern "C" __global__
void aegis_audio_depthwise_causal_conv1d(
    const float* __restrict__ in,     // [n_frames, channels]
    const float* __restrict__ w,      // [channels, kernel]
    float*       __restrict__ out,    // [n_frames, channels]
    const unsigned int n_frames,
    const unsigned int channels,
    const unsigned int kernel
) {
    const unsigned int t = blockIdx.x;
    const unsigned int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= n_frames || c >= channels) return;
    float acc = 0.0f;
    // tap j corresponds to input frame  t - (kernel-1) + j.
    for (unsigned int j = 0; j < kernel; ++j) {
        const int src_t = (int)t - (int)(kernel - 1) + (int)j;
        if (src_t < 0) continue;                      // left zero-pad
        const float xv = in[(size_t)src_t * channels + c];
        const float wv = w[(size_t)c * kernel + j];
        acc += xv * wv;
    }
    out[(size_t)t * channels + c] = acc;
}

// ─────────────────────────────────────────────────────────────────────────
// per_dim_scale apply to the attention query.
//   q : [n_frames, n_heads, head_dim]  row-major f32, modified IN PLACE.
//   per_dim_scale : [head_dim] f32 (raw learned param; softplus applied here).
// HF: per_dim_scale_sp = softplus(per_dim_scale);
//     q = q * q_scale * per_dim_scale_sp     (broadcast over heads & frames)
// where q_scale = head_dim^-0.5 / softplus(0) is a precomputed scalar passed in.
// softplus(x) = log1p(exp(x)); numerically stable form for large x.
// Launch: grid = (n_frames, n_heads), block = (head_dim).
// ─────────────────────────────────────────────────────────────────────────
extern "C" __global__
void aegis_audio_per_dim_scale(
    float*       __restrict__ q,             // [n_frames, n_heads, head_dim]
    const float* __restrict__ per_dim_scale, // [head_dim]
    const unsigned int n_heads,
    const unsigned int head_dim,
    const float q_scale
) {
    const unsigned int t = blockIdx.x;
    const unsigned int head = blockIdx.y;
    const unsigned int d = threadIdx.x;
    if (d >= head_dim) return;
    const float s = per_dim_scale[d];
    // softplus(s) = log(1 + exp(s)); stable: max(s,0) + log1p(exp(-|s|))
    const float sp = fmaxf(s, 0.0f) + log1pf(expf(-fabsf(s)));
    const size_t idx = ((size_t)t * n_heads + head) * head_dim + d;
    q[idx] = q[idx] * q_scale * sp;
}

// ─────────────────────────────────────────────────────────────────────────
// Gradient clipping (HF clamps to ±gradient_clipping between sub-blocks).
//   x : flat f32 buffer of length n, clamped in place to [-c, c].
// Launch: grid = ceil(n/256), block = 256.
// ─────────────────────────────────────────────────────────────────────────
extern "C" __global__
void aegis_audio_clamp_inplace(
    float* __restrict__ x,
    const unsigned int n,
    const float c
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    x[i] = fminf(fmaxf(v, -c), c);
}

// ─────────────────────────────────────────────────────────────────────────
// SiLU in place: x = x * sigmoid(x). Used for the LightConv1d post-conv
// activation AND the (assumed ungated) Macaron-FFN activation. Flat f32
// buffer of length n.
// Launch: grid = ceil(n/256), block = 256.
// ─────────────────────────────────────────────────────────────────────────
extern "C" __global__
void aegis_audio_silu_inplace(
    float* __restrict__ x,
    const unsigned int n
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    x[i] = v / (1.0f + expf(-v));
}

// ─────────────────────────────────────────────────────────────────────────
// Add a learned bias vector to each row.
//   x    : [n_rows, dim]  row-major f32, modified IN PLACE.
//   bias : [dim] f32.
// Used by output_proj (which carries a bias) in the audio tail.
// Launch: grid = (n_rows, ceil(dim/256)), block = 256.
// ─────────────────────────────────────────────────────────────────────────
extern "C" __global__
void aegis_audio_add_bias_rows(
    float*       __restrict__ x,
    const float* __restrict__ bias,
    const unsigned int dim
) {
    const unsigned int row = blockIdx.x;
    const unsigned int c   = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= dim) return;
    x[(size_t)row * dim + c] += bias[c];
}

#endif  // __CUDA_ARCH__ >= 800

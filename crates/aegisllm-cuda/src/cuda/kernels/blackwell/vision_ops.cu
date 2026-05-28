// =============================================================================
// Vision tower per-token CUDA kernels (Stage I.3 GPU-only forward).
// =============================================================================
//
// Pixel rescale, 2D-axial position embedding add, per-head QK/V RMSNorm,
// 2D multidimensional RoPE, final standardization (subtract-then-scale),
// 3x3 average pooling with sqrt(hidden) scale.

#if __CUDA_ARCH__ >= 800

// ─────────────────────────────────────────────────────────────────────────
// Pixel rescale: pixel = 2 * (pixel - 0.5)
// In-place over a flat buffer of `n` elements.
// Launch: grid_dim = ceil(n / 256), block_dim = 256.
// ─────────────────────────────────────────────────────────────────────────
extern "C" __global__
void aegis_vision_pixel_rescale(
    float* __restrict__ pixels,
    const unsigned int n
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    pixels[i] = 2.0f * (pixels[i] - 0.5f);
}

// ─────────────────────────────────────────────────────────────────────────
// 2D-axial position embedding add: per patch (ph, pw), hidden[tok, c] +=
// table[bank=0, x=pw, c] + table[bank=1, y=ph, c]. position_table is
// pre-flattened [2 * n_table_rows, hidden] BF16 (bank 0 is rows 0..n_table,
// bank 1 is rows n_table..2*n_table).
// Launch: grid_dim = (n_patches, ceil(hidden/256)), block_dim = 256.
// ─────────────────────────────────────────────────────────────────────────
extern "C" __global__
void aegis_vision_pos_embed_add(
    float*                __restrict__ hidden_buf,            // [n_patches, hidden_size]
    const unsigned short* __restrict__ position_table_bf16,   // [2*n_table_rows, hidden_size]
    const unsigned int n_patches_h,
    const unsigned int n_patches_w,
    const unsigned int n_table_rows,
    const unsigned int hidden_size
) {
    const unsigned int tok = blockIdx.x;
    const unsigned int c   = blockIdx.y * blockDim.x + threadIdx.x;
    if (tok >= n_patches_h * n_patches_w || c >= hidden_size) return;
    const unsigned int ph = tok / n_patches_w;
    const unsigned int pw = tok - ph * n_patches_w;
    const unsigned int x_idx = pw < n_table_rows ? pw : (n_table_rows - 1);
    const unsigned int y_idx = ph < n_table_rows ? ph : (n_table_rows - 1);
    // BF16 → F32: shift the 16 bits into the high half of a 32-bit float.
    const unsigned short x_h = position_table_bf16[(size_t)x_idx * hidden_size + c];
    const unsigned short y_h = position_table_bf16[(size_t)(n_table_rows + y_idx) * hidden_size + c];
    const float x_f = __int_as_float((int)((unsigned)x_h) << 16);
    const float y_f = __int_as_float((int)((unsigned)y_h) << 16);
    hidden_buf[(size_t)tok * hidden_size + c] += x_f + y_f;
}

// ─────────────────────────────────────────────────────────────────────────
// Per-head per-token RMSNorm of Q/K (with weight) or V (without weight).
// Input: x [n_tok, n_heads, head_dim] row-major f32.
// Weight (optional): [head_dim] f32. Pass null/zero-length to skip.
// In-place.
// Launch: grid = (n_tok, n_heads), block = (head_dim, 1, 1). One block per
// (token, head); threads cooperate over head_dim. head_dim must be ≤ 1024.
// ─────────────────────────────────────────────────────────────────────────
extern "C" __global__
void aegis_vision_head_rmsnorm(
    float*       __restrict__ x,           // [n_tok, n_heads, head_dim]
    const float* __restrict__ weight,      // [head_dim] or nullptr
    const unsigned int n_heads,
    const unsigned int head_dim,
    const float eps,
    const unsigned int with_weight
) {
    const unsigned int tok  = blockIdx.x;
    const unsigned int head = blockIdx.y;
    const unsigned int tid  = threadIdx.x;
    if (tid >= head_dim) return;
    const size_t base = ((size_t)tok * n_heads + head) * head_dim;
    const float xv = x[base + tid];
    // Block-wide sum of xv*xv.
    __shared__ float ssum;
    if (tid == 0) ssum = 0.0f;
    __syncthreads();
    atomicAdd(&ssum, xv * xv);
    __syncthreads();
    const float rms = rsqrtf(ssum / (float)head_dim + eps);
    float out = xv * rms;
    if (with_weight) out *= weight[tid];
    x[base + tid] = out;
}

// ─────────────────────────────────────────────────────────────────────────
// 2D multidimensional RoPE: applies position rotation to Q or K (in place).
// head_dim split into 2 chunks of (head_dim/2) each (x-dim, y-dim). Within
// each chunk, rotate-half pairs at offset (head_dim/4).
// freqs[i] = 1 / theta^(2i / (head_dim/2)) for i in [0, head_dim/4)
// cos/sin(chunk_x at token t): freqs[i] * pos_x[t], repeated twice.
//
// Launch: grid = (n_tok, n_heads), block = (head_dim, 1, 1). Each thread
// handles one head_dim element of one (tok, head).
// ─────────────────────────────────────────────────────────────────────────
extern "C" __global__
void aegis_vision_rope_2d(
    float* __restrict__ x,                       // [n_tok, n_heads, head_dim]
    const unsigned int n_patches_w,
    const unsigned int n_heads,
    const unsigned int head_dim,
    const float rope_theta
) {
    const unsigned int tok  = blockIdx.x;
    const unsigned int head = blockIdx.y;
    const unsigned int k    = threadIdx.x;
    if (k >= head_dim) return;
    const unsigned int spatial_dim = head_dim / 2;   // 36 for hd=72
    const unsigned int n_freqs     = spatial_dim / 2; // 18

    const unsigned int ph = tok / n_patches_w;
    const unsigned int pw = tok - ph * n_patches_w;

    // Which spatial chunk is k in?
    const unsigned int chunk = (k < spatial_dim) ? 0u : 1u;
    const unsigned int k_in_chunk = (k < spatial_dim) ? k : (k - spatial_dim);
    const unsigned int chunk_base_in_hd = (chunk == 0u) ? 0u : spatial_dim;

    // Position for this chunk: x for chunk 0, y for chunk 1.
    const float pos = (chunk == 0u) ? (float)pw : (float)ph;

    // Frequency index inside the spatial_dim (mirrored at n_freqs).
    const unsigned int freq_i = (k_in_chunk < n_freqs) ? k_in_chunk : (k_in_chunk - n_freqs);
    const float exponent = (float)(2u * freq_i) / (float)spatial_dim;
    const float inv_freq = 1.0f / powf(rope_theta, exponent);
    const float freq     = pos * inv_freq;
    const float cos_v    = cosf(freq);
    const float sin_v    = sinf(freq);

    // rotate_half pair: chunk[k_in_chunk] pairs with chunk[k_in_chunk±n_freqs];
    // rot = -x[pair] if k_in_chunk < n_freqs, else +x[pair].
    const unsigned int pair_in_chunk = (k_in_chunk < n_freqs)
        ? (k_in_chunk + n_freqs) : (k_in_chunk - n_freqs);
    const unsigned int pair_k = chunk_base_in_hd + pair_in_chunk;

    const size_t base = ((size_t)tok * n_heads + head) * head_dim;
    // Read both before writing (in-place safe via two reads then one write per thread).
    const float my_val   = x[base + k];
    const float pair_val = x[base + pair_k];
    const float rot      = (k_in_chunk < n_freqs) ? -pair_val : pair_val;
    // Each thread writes only its own k → no race.
    x[base + k] = my_val * cos_v + rot * sin_v;
}

// ─────────────────────────────────────────────────────────────────────────
// Standardization: x = (x - bias) * scale, per-channel affine. In-place.
// Launch: grid_dim = (n_rows, ceil(hidden/256)), block_dim = 256.
// ─────────────────────────────────────────────────────────────────────────
extern "C" __global__
void aegis_vision_standardize(
    float*       __restrict__ x,         // [n_rows, hidden]
    const float* __restrict__ scale,     // [hidden]
    const float* __restrict__ bias,      // [hidden]
    const unsigned int hidden_size
) {
    const unsigned int row = blockIdx.x;
    const unsigned int c   = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= hidden_size) return;
    const size_t idx = (size_t)row * hidden_size + c;
    x[idx] = (x[idx] - bias[c]) * scale[c];
}

// ─────────────────────────────────────────────────────────────────────────
// 3x3 (stride 3, no overlap) average pool with sqrt(hidden) scale factor.
// Input:  src [n_ph * n_pw, hidden]   row-major f32 (patch grid)
// Output: dst [n_th * n_tw, hidden]   row-major f32 (pooled tokens)
//   where n_th = n_ph / pool, n_tw = n_pw / pool.
// Each output value = (sum of pool*pool input values) / pool² * sqrt(hidden).
//
// Launch: grid_dim = (n_th * n_tw, ceil(hidden/256)), block_dim = 256.
// ─────────────────────────────────────────────────────────────────────────
extern "C" __global__
void aegis_vision_pool3x3_scale(
    const float* __restrict__ src,
    float*       __restrict__ dst,
    const unsigned int n_ph,
    const unsigned int n_pw,
    const unsigned int n_tw,
    const unsigned int hidden_size,
    const unsigned int pool,
    const float        out_scale            // sqrt(hidden) / pool^2
) {
    const unsigned int tok = blockIdx.x;
    const unsigned int c   = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= hidden_size) return;
    const unsigned int th = tok / n_tw;
    const unsigned int tw = tok - th * n_tw;
    const unsigned int ph_base = th * pool;
    const unsigned int pw_base = tw * pool;
    float sum = 0.0f;
    for (unsigned int dh = 0; dh < pool; ++dh) {
        const unsigned int ph = ph_base + dh;
        if (ph >= n_ph) continue;
        for (unsigned int dw = 0; dw < pool; ++dw) {
            const unsigned int pw = pw_base + dw;
            if (pw >= n_pw) continue;
            sum += src[((size_t)ph * n_pw + pw) * hidden_size + c];
        }
    }
    dst[((size_t)tok) * hidden_size + c] = sum * out_scale;
}

#endif  // __CUDA_ARCH__ >= 800

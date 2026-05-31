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

// =============================================================================
// Qwen3-VL native-ViT kernels (LayerNorm+bias, gelu_tanh batched, 2D vision
// RoPE in (row,col) rotate-half form, add-bias-rows).
// =============================================================================

// ─────────────────────────────────────────────────────────────────────────
// LayerNorm with weight + bias (per-row over `dim`), f32 in/out.
//   y = (x - mean) / sqrt(var + eps) * weight + bias    (var = E[x²]-E[x]²)
// Launch: grid = (n_rows, 1, 1), block = (256, 1, 1). One block per row;
// 256 threads cooperate over `dim` via a 2-pass shared reduction.
// ─────────────────────────────────────────────────────────────────────────
extern "C" __global__
void aegis_vision_layernorm_bias(
    const float* __restrict__ x,        // [n_rows, dim]
    const float* __restrict__ weight,   // [dim]
    const float* __restrict__ bias,     // [dim]
    const unsigned int dim,
    const float eps,
    float* __restrict__ out             // [n_rows, dim]
) {
    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int nthreads = blockDim.x;
    const size_t base = (size_t)row * dim;

    __shared__ float s_sum[256];
    __shared__ float s_sumsq[256];

    float local_sum = 0.0f;
    float local_sumsq = 0.0f;
    for (unsigned int i = tid; i < dim; i += nthreads) {
        const float v = x[base + i];
        local_sum += v;
        local_sumsq += v * v;
    }
    s_sum[tid] = local_sum;
    s_sumsq[tid] = local_sumsq;
    __syncthreads();
    for (unsigned int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid]   += s_sum[tid + stride];
            s_sumsq[tid] += s_sumsq[tid + stride];
        }
        __syncthreads();
    }
    const float mean = s_sum[0] / (float)dim;
    const float var  = s_sumsq[0] / (float)dim - mean * mean;
    const float inv  = rsqrtf(var + eps);
    for (unsigned int i = tid; i < dim; i += nthreads) {
        const float v = x[base + i];
        out[base + i] = (v - mean) * inv * weight[i] + bias[i];
    }
}

// ─────────────────────────────────────────────────────────────────────────
// GELU tanh approximation (per element, in place over a flat buffer of n):
//   0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 x³) ))
// Same formula as aegis_gelu_tanh_inplace_f32; kept here so the vision PTX
// module is self-contained (avoids relying on which TU defines it first).
// Launch: grid = ceil(n/256), block = 256.
// ─────────────────────────────────────────────────────────────────────────
extern "C" __global__
void aegis_vision_gelu_tanh(
    float* __restrict__ x,
    const unsigned int n
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    const float c = 0.7978845608028654f; // sqrt(2/pi)
    const float inner = c * (v + 0.044715f * v * v * v);
    x[i] = 0.5f * v * (1.0f + tanhf(inner));
}

// ─────────────────────────────────────────────────────────────────────────
// Exact (erf-based) GELU, in place over n elements:  0.5 x (1 + erf(x/√2)).
// HF nn.GELU() default — used by the Qwen vision merger fc1 activation
// (the per-block MLP uses gelu_pytorch_tanh instead).
// Launch: grid = ceil(n/256), block = 256.
// ─────────────────────────────────────────────────────────────────────────
extern "C" __global__
void aegis_vision_gelu_erf(
    float* __restrict__ x,
    const unsigned int n
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    x[i] = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f)); // 1/√2
}

// ─────────────────────────────────────────────────────────────────────────
// Add a per-column bias to every row.  x[row, c] += bias[c].  In place.
// Launch: grid = (n_rows, ceil(dim/256)), block = 256.
// ─────────────────────────────────────────────────────────────────────────
extern "C" __global__
void aegis_vision_add_bias_rows(
    float*       __restrict__ x,
    const float* __restrict__ bias,
    const unsigned int dim
) {
    const unsigned int row = blockIdx.x;
    const unsigned int c   = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= dim) return;
    x[(size_t)row * dim + c] += bias[c];
}

// ─────────────────────────────────────────────────────────────────────────
// Qwen3-VL vision RoPE (in place over Q or K).  Standard rotate-half form,
// NOT Gemma's multidim-split.  Per HF apply_rotary_pos_emb_vision:
//
//   rotary_pos_emb[tok] = [ row_id[tok] * inv_freq(0..n_freqs),
//                            col_id[tok] * inv_freq(0..n_freqs) ]   (len head_dim/2)
//   emb = cat([rotary, rotary])                                     (len head_dim)
//   cos/sin = emb.cos()/emb.sin()
//   y = x * cos + rotate_half(x) * sin
//   rotate_half(x) = [-x[d/2:], x[:d/2]]
//
// where inv_freq(i) = 1 / theta^(2i / (head_dim/2)),  i in [0, head_dim/4).
// row_id/col_id are the per-token (h,w) ids in merge-block order, supplied
// as a device array `pos_ids` of shape [n_tok, 2] (row, col) int32.
//
// Launch: grid = (n_tok, n_heads), block = (head_dim, 1, 1). Each thread
// owns one head_dim element of one (tok, head); reads its rotate-half pair
// then writes only its own slot (race-free).
// ─────────────────────────────────────────────────────────────────────────
extern "C" __global__
void aegis_vision_rope_qwen(
    float*             __restrict__ x,        // [n_tok, n_heads, head_dim]
    const int*         __restrict__ pos_ids,  // [n_tok, 2] (row, col)
    const unsigned int n_heads,
    const unsigned int head_dim,
    const float        rope_theta
) {
    const unsigned int tok  = blockIdx.x;
    const unsigned int head = blockIdx.y;
    const unsigned int k    = threadIdx.x;
    if (k >= head_dim) return;
    const unsigned int half    = head_dim / 2;     // 36 for hd=72
    const unsigned int n_freqs = half / 2;         // 18 (inv_freq count per axis)

    // Which entry of `emb` (len head_dim) is this? emb = cat([rotary, rotary]),
    // rotary = [row*invf(0..nf), col*invf(0..nf)] (len half).
    // So index k in [0, head_dim): the "rotary index" is (k % half).
    const unsigned int r = (k < half) ? k : (k - half);  // index into rotary[0..half)
    // r in [0, n_freqs)      -> row axis, freq r
    // r in [n_freqs, half)   -> col axis, freq (r - n_freqs)
    int   pos_id;
    unsigned int freq_i;
    if (r < n_freqs) {
        pos_id = pos_ids[tok * 2 + 0];   // row
        freq_i = r;
    } else {
        pos_id = pos_ids[tok * 2 + 1];   // col
        freq_i = r - n_freqs;
    }
    const float exponent = (float)(2u * freq_i) / (float)half;
    const float inv_freq = 1.0f / powf(rope_theta, exponent);
    const float angle    = (float)pos_id * inv_freq;
    const float cos_v    = cosf(angle);
    const float sin_v    = sinf(angle);

    // rotate_half over the FULL head_dim: pair(k) = k+half if k<half else k-half,
    // with sign -1 for the lower half, +1 for the upper half.
    const unsigned int pair_k = (k < half) ? (k + half) : (k - half);
    const float sign = (k < half) ? -1.0f : 1.0f;

    const size_t base = ((size_t)tok * n_heads + head) * head_dim;
    const float my_val   = x[base + k];
    const float pair_val = x[base + pair_k];
    x[base + k] = my_val * cos_v + sign * pair_val * sin_v;
}

#endif  // __CUDA_ARCH__ >= 800

//! Gated DeltaNet (GDN) linear-attention — pure-f32 CPU **reference oracle**.
//!
//! This module is the unambiguous ground truth for Qwen3-Next (Qwen3.5/3.6)
//! linear-attention layers. The CUDA decode/prefill kernels are validated
//! against it (per-op cosine > 0.999). It is deliberately written as the plain
//! sequential recurrence — no chunking, no triangular solve — so its
//! correctness is by-inspection. The chunked-parallel algorithm (a GPU
//! optimization) is checked against THIS, not the other way round.
//!
//! Canonical math (reconciled HF `modeling_qwen3_next.py` + vLLM + flashinfer):
//!   - in_proj_qkvz → q,k:[n_k,d_k], v,z:[n_v,d_v]; in_proj_ba → b,a:[n_v]
//!   - depthwise causal conv1d (k=4) + SiLU over cat[q,k,v] (NOT z)
//!   - beta = sigmoid(b);  g = -exp(A_log) * softplus(a + dt_bias)   (NEGATIVE)
//!   - l2norm(q,k) over d_k (eps 1e-6); GQA-expand q,k by n_v/n_k; q *= 1/sqrt(d_k)
//!   - per value-head recurrence with state S[d_k, d_v]:
//!       S *= exp(g);  kv = Sᵀk;  delta = (v - kv)·beta;  S += k⊗delta;  y = Sᵀq
//!   - per-head gated RMSNorm of y by `norm` gated by silu(z), then out_proj
//!
//! State layout decision (must match the CUDA kernels): **S[d_k, d_v]** per
//! value head, row-major → `S[h*d_k*d_v + i*d_v + j]` is `S_h[i, j]`.

/// Numerically-stable softplus: `log(1 + exp(x))`.
#[inline]
pub fn softplus(x: f32) -> f32 {
    // max(x,0) + log1p(exp(-|x|)) avoids overflow for large |x|.
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// L2-normalizes `v` in place over its full length (eps under the sqrt).
pub fn l2_normalize(v: &mut [f32], eps: f32) {
    let ss: f32 = v.iter().map(|&x| x * x).sum();
    let inv = 1.0 / (ss + eps).sqrt();
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// Per-token gate scalar `g = -exp(A_log) * softplus(a + dt_bias)` (negative;
/// drives the `exp(g)` state decay). `a_log`, `dt_bias`, `a` are per value-head.
#[inline]
pub fn delta_gate(a: f32, dt_bias: f32, a_log: f32) -> f32 {
    -a_log.exp() * softplus(a + dt_bias)
}

/// Static dimensions of the recurrence.
#[derive(Debug, Clone, Copy)]
pub struct GdnShape {
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub k_head_dim: usize,
    pub v_head_dim: usize,
}

impl GdnShape {
    #[inline]
    pub fn state_len(&self) -> usize {
        self.num_v_heads * self.k_head_dim * self.v_head_dim
    }
    /// GQA expansion ratio (value heads per key head).
    #[inline]
    pub fn expand(&self) -> usize {
        self.num_v_heads / self.num_k_heads
    }
    /// Key head that feeds value head `h`.
    #[inline]
    pub fn key_head_of(&self, h: usize) -> usize {
        h / self.expand()
    }
}

/// One GDN decode step. Mutates `state` (length `shape.state_len()`) and writes
/// `out` (length `n_v * d_v`).
///
/// Inputs are POST-conv, POST-norm:
///   - `q_norm`, `k_norm`: `[n_k, d_k]` L2-normed (and q pre-scaled by 1/√d_k)
///   - `v`: `[n_v, d_v]`
///   - `beta`, `g`: `[n_v]`
pub fn decode_step(
    shape: &GdnShape,
    state: &mut [f32],
    q_norm: &[f32],
    k_norm: &[f32],
    v: &[f32],
    beta: &[f32],
    g: &[f32],
    out: &mut [f32],
) {
    let (d_k, d_v) = (shape.k_head_dim, shape.v_head_dim);
    for h in 0..shape.num_v_heads {
        let kh = shape.key_head_of(h);
        let q = &q_norm[kh * d_k..(kh + 1) * d_k];
        let k = &k_norm[kh * d_k..(kh + 1) * d_k];
        let vh = &v[h * d_v..(h + 1) * d_v];
        let base = h * d_k * d_v;
        let s = &mut state[base..base + d_k * d_v];

        // 1. decay the whole state by exp(g_h).
        let decay = g[h].exp();
        for x in s.iter_mut() {
            *x *= decay;
        }
        // 2. per output dim j: kv_j = Σ_i S[i,j]·k_i, delta_j = (v_j - kv_j)·β,
        //    then S[i,j] += k_i·delta_j  (column j is independent).
        let b = beta[h];
        for j in 0..d_v {
            let mut kv = 0.0f32;
            for i in 0..d_k {
                kv += s[i * d_v + j] * k[i];
            }
            let delta = (vh[j] - kv) * b;
            for i in 0..d_k {
                s[i * d_v + j] += k[i] * delta;
            }
        }
        // 3. y_j = Σ_i S[i,j]·q_i  (post-update state).
        let oh = &mut out[h * d_v..(h + 1) * d_v];
        for (j, oj) in oh.iter_mut().enumerate() {
            let mut y = 0.0f32;
            for i in 0..d_k {
                y += s[i * d_v + j] * q[i];
            }
            *oj = y;
        }
    }
}

/// Per-head gated RMSNorm (Qwen3-Next `Qwen3NextRMSNormGated`). HF order
/// (modeling_qwen3_next.py): normalize `y` over `d_v` FIRST, scale by plain
/// `weight`, THEN gate by silu(z): `out = weight * (y * rsqrt(mean(y^2)+eps)) *
/// silu(z)`. Operates per value head; `weight` is `[d_v]`. Writes `out`.
pub fn gated_rmsnorm(
    shape: &GdnShape,
    y: &[f32],
    z: &[f32],
    weight: &[f32],
    eps: f32,
    out: &mut [f32],
) {
    let d_v = shape.v_head_dim;
    for h in 0..shape.num_v_heads {
        let yh = &y[h * d_v..(h + 1) * d_v];
        let zh = &z[h * d_v..(h + 1) * d_v];
        let oh = &mut out[h * d_v..(h + 1) * d_v];
        // mean of squares over the UN-gated y.
        let ms: f32 = yh.iter().map(|&v| v * v).sum::<f32>() / d_v as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for j in 0..d_v {
            oh[j] = weight[j] * (yh[j] * inv) * silu(zh[j]);
        }
    }
}

/// Depthwise **causal** conv1d (groups = channels), left-padded with `K-1`
/// zeros, followed by SiLU. `input` is `[T, C]` row-major, `weight` is `[C, K]`,
/// optional per-channel `bias` is `[C]`. Returns `[T, C]`. This matches the
/// Qwen3-Next short-conv applied to cat[q, k, v].
pub fn causal_conv1d_silu(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    seq_len: usize,
    channels: usize,
    kernel: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; seq_len * channels];
    for t in 0..seq_len {
        for c in 0..channels {
            let mut acc = bias.map_or(0.0, |b| b[c]);
            for kk in 0..kernel {
                // causal tap: input position t - (K-1) + kk
                let src = t as isize - (kernel as isize - 1) + kk as isize;
                if src >= 0 {
                    acc += weight[c * kernel + kk] * input[src as usize * channels + c];
                }
            }
            out[t * channels + c] = silu(acc);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(&x, &y)| x * y).sum();
        let na: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
        dot / (na * nb + 1e-12)
    }

    #[test]
    fn softplus_known_values() {
        assert!(approx(softplus(0.0), std::f32::consts::LN_2, 1e-6));
        // large positive ≈ x; large negative ≈ 0; both numerically stable.
        assert!(approx(softplus(30.0), 30.0, 1e-4));
        assert!(softplus(-40.0) >= 0.0 && softplus(-40.0) < 1e-10);
    }

    #[test]
    fn delta_gate_is_negative() {
        // g = -exp(A_log)·softplus(...) is always ≤ 0 → exp(g) ∈ (0,1] decays.
        let g = delta_gate(0.3, -0.1, 0.0);
        assert!(g <= 0.0);
        assert!(g.exp() > 0.0 && g.exp() <= 1.0);
    }

    #[test]
    fn l2_normalize_unit_length() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v, 1e-6);
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(approx(n, 1.0, 1e-4));
    }

    #[test]
    fn conv1d_is_causal() {
        // C=1, K=4, identity-on-last-tap weight → out[t] = silu(input[t]).
        let inp = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let w = vec![0.0, 0.0, 0.0, 1.0]; // only the current tap
        let out = causal_conv1d_silu(&inp, &w, None, 5, 1, 4);
        for t in 0..5 {
            assert!(approx(out[t], silu(inp[t]), 1e-5));
        }
        // a future-only weight would be non-causal; verify tap 0 looks back 3.
        let w2 = vec![1.0, 0.0, 0.0, 0.0]; // tap at t-3
        let out2 = causal_conv1d_silu(&inp, &w2, None, 5, 1, 4);
        assert!(approx(out2[0], silu(0.0), 1e-5)); // t-3 < 0 → padded 0
        assert!(approx(out2[3], silu(inp[0]), 1e-5)); // t=3 sees input[0]
    }

    #[test]
    fn decode_step_empty_state_is_beta_weighted_outer() {
        // With S=0: kv=0, delta = v·β, S = k⊗(v·β), y = (qᵀk)·(v·β).
        let shape = GdnShape { num_k_heads: 1, num_v_heads: 1, k_head_dim: 2, v_head_dim: 2 };
        let mut state = vec![0.0f32; shape.state_len()];
        let mut q = vec![0.6f32, 0.8]; // unit
        let mut k = vec![1.0f32, 0.0];
        let v = vec![2.0f32, -1.0];
        let beta = vec![0.5f32];
        let g = vec![-0.1f32];
        l2_normalize(&mut q, 1e-6);
        l2_normalize(&mut k, 1e-6);
        let mut out = vec![0.0f32; 2];
        decode_step(&shape, &mut state, &q, &k, &v, &beta, &g, &mut out);
        // qᵀk = 0.6 (since k=[1,0], q=[0.6,0.8]); expected y = 0.6 * (v*β).
        let qtk = q[0] * k[0] + q[1] * k[1];
        for j in 0..2 {
            assert!(approx(out[j], qtk * v[j] * beta[0], 1e-5));
        }
    }

    #[test]
    fn state_decays_toward_zero_under_repeated_g() {
        // Repeated steps with v=0 must shrink the state (exp(g)<1) and outputs.
        let shape = GdnShape { num_k_heads: 1, num_v_heads: 1, k_head_dim: 4, v_head_dim: 4 };
        let mut state = vec![0.0f32; shape.state_len()];
        let mut q = vec![1.0, 0.5, -0.3, 0.2];
        let mut k = vec![0.4, -0.2, 0.7, 0.1];
        l2_normalize(&mut q, 1e-6);
        l2_normalize(&mut k, 1e-6);
        let beta = vec![0.7f32];
        let g = vec![-0.5f32];
        let mut out = vec![0.0f32; 4];
        // prime with a non-zero v
        decode_step(&shape, &mut state, &q, &k, &vec![1.0, 2.0, 3.0, 4.0], &beta, &g, &mut out);
        let mag0: f32 = state.iter().map(|x| x * x).sum();
        // then drive with v=0 → only decay + (kv mismatch) shrinks it
        for _ in 0..20 {
            decode_step(&shape, &mut state, &q, &k, &vec![0.0; 4], &beta, &g, &mut out);
        }
        let mag1: f32 = state.iter().map(|x| x * x).sum();
        assert!(mag1 < mag0, "state should decay: {mag0} -> {mag1}");
    }

    #[test]
    fn sequential_state_carries_across_calls() {
        // Running [t0,t1] in one loop must equal t0 then t1 in two calls
        // (the recurrence is the ground truth — state must persist verbatim).
        let shape = GdnShape { num_k_heads: 2, num_v_heads: 2, k_head_dim: 3, v_head_dim: 3 };
        let mk = |seed: f32| {
            let mut q = vec![seed, seed + 0.1, -seed, 0.2, -0.3, seed * 0.5];
            let mut k = vec![0.3, -seed, 0.2, seed, 0.1, -0.4];
            for h in 0..2 {
                l2_normalize(&mut q[h * 3..(h + 1) * 3], 1e-6);
                l2_normalize(&mut k[h * 3..(h + 1) * 3], 1e-6);
            }
            let v = vec![seed, 1.0, -1.0, 0.5, seed, 0.3];
            (q, k, v)
        };
        let beta = vec![0.6f32, 0.4];
        let g = vec![-0.2f32, -0.7];

        // combined
        let mut s_comb = vec![0.0f32; shape.state_len()];
        let mut o = vec![0.0f32; 6];
        let (q0, k0, v0) = mk(0.3);
        let (q1, k1, v1) = mk(0.9);
        decode_step(&shape, &mut s_comb, &q0, &k0, &v0, &beta, &g, &mut o);
        let o0_comb = o.clone();
        decode_step(&shape, &mut s_comb, &q1, &k1, &v1, &beta, &g, &mut o);

        // split (fresh state, replay)
        let mut s_split = vec![0.0f32; shape.state_len()];
        let mut o0 = vec![0.0f32; 6];
        decode_step(&shape, &mut s_split, &q0, &k0, &v0, &beta, &g, &mut o0);
        let mut o1 = vec![0.0f32; 6];
        decode_step(&shape, &mut s_split, &q1, &k1, &v1, &beta, &g, &mut o1);

        assert!(cosine(&o0_comb, &o0) > 0.99999);
        assert!(cosine(&o, &o1) > 0.99999);
        assert!(cosine(&s_comb, &s_split) > 0.99999);
    }

    #[test]
    fn gated_rmsnorm_normalizes() {
        let shape = GdnShape { num_k_heads: 1, num_v_heads: 1, k_head_dim: 4, v_head_dim: 4 };
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let z = vec![0.5, 0.5, 0.5, 0.5];
        let w = vec![1.0, 1.0, 1.0, 1.0];
        let mut out = vec![0.0; 4];
        gated_rmsnorm(&shape, &y, &z, &w, 1e-6, &mut out);
        // HF order: out = (1)·norm(y)·silu(z). With z constant 0.5 and weight 1,
        // norm(y) has unit RMS, so out's RMS = silu(0.5).
        let rms = (out.iter().map(|x| x * x).sum::<f32>() / 4.0).sqrt();
        assert!(approx(rms, silu(0.5), 1e-3), "rms={rms}");
    }
}

//! Per-head RMS norm for Gemma-4 q/k/v norms.
//!
//! Mirrors the CUDA `rms_norm_batched_device` (weighted, used for q_norm/k_norm)
//! and `rms_norm_batched_no_weight_device` (no weight, used for v_norm) kernels
//! — `crates/aegisllm-cuda/src/executor/attention.rs:139-183`,
//! kernel math `norm_rope_kv.cu:66-132`.
//!
//! Each `[head_dim]` slice is an independent RMS-norm row:
//!   scale  = 1 / sqrt(mean(x²) + eps)
//!   out[i] = x[i] * scale * w[i]      (weighted variant)
//!   out[i] = x[i] * scale             (no-weight variant, v_norm)
//!
//! This is the exact body of the model-wide `math::rms_norm_into`, looped per
//! head over `head_dim`.

use crate::cpu::simd;

/// Weighted per-head RMS norm (q_norm / k_norm). `input` and `out` are
/// `[n_heads * head_dim]` flat row-major; `weight` is `[head_dim]` (shared
/// across heads). `out` may NOT alias `input` (two-pass, like the kernel).
pub(crate) fn rms_norm_per_head_into(
    input: &[f32],
    weight: &[f32],
    n_heads: usize,
    head_dim: usize,
    eps: f32,
    out: &mut [f32],
) {
    debug_assert_eq!(input.len(), n_heads * head_dim);
    debug_assert_eq!(out.len(), n_heads * head_dim);
    debug_assert_eq!(weight.len(), head_dim);
    for head in 0..n_heads {
        let base = head * head_dim;
        let x = &input[base..base + head_dim];
        let mean_square = simd::dot_f32(x, x) / head_dim as f32;
        let scale = 1.0 / (mean_square + eps).sqrt();
        simd::rms_scale(x, weight, &mut out[base..base + head_dim], scale);
    }
}

/// Unweighted per-head RMS norm (Gemma-4 v_norm, `with_scale=false`).
/// `out[i] = x[i] * rsqrt(mean(x²) + eps)`.
pub(crate) fn rms_norm_per_head_no_weight_into(
    input: &[f32],
    n_heads: usize,
    head_dim: usize,
    eps: f32,
    out: &mut [f32],
) {
    debug_assert_eq!(input.len(), n_heads * head_dim);
    debug_assert_eq!(out.len(), n_heads * head_dim);
    for head in 0..n_heads {
        let base = head * head_dim;
        let x = &input[base..base + head_dim];
        let mean_square = simd::dot_f32(x, x) / head_dim as f32;
        let scale = 1.0 / (mean_square + eps).sqrt();
        let dst = &mut out[base..base + head_dim];
        dst.copy_from_slice(x);
        simd::scale_in_place(dst, scale);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_rms(x: &[f32], w: Option<&[f32]>, eps: f32) -> Vec<f32> {
        let n = x.len();
        let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let scale = 1.0 / (ms + eps).sqrt();
        (0..n)
            .map(|i| x[i] * scale * w.map(|w| w[i]).unwrap_or(1.0))
            .collect()
    }

    #[test]
    fn per_head_weighted_matches_hand_computed() {
        // 2 heads, head_dim=4, distinct per-head magnitudes.
        let input = [1.0f32, 2.0, 3.0, 4.0, 10.0, 0.0, -10.0, 0.0];
        let weight = [0.5f32, 1.0, 2.0, 1.5];
        let eps = 1e-6;
        let mut out = [0.0f32; 8];
        rms_norm_per_head_into(&input, &weight, 2, 4, eps, &mut out);

        let h0 = ref_rms(&input[0..4], Some(&weight), eps);
        let h1 = ref_rms(&input[4..8], Some(&weight), eps);
        for i in 0..4 {
            assert!((out[i] - h0[i]).abs() < 1e-5, "head0 i={i}: {} vs {}", out[i], h0[i]);
            assert!((out[4 + i] - h1[i]).abs() < 1e-5, "head1 i={i}");
        }
    }

    #[test]
    fn per_head_no_weight_matches_hand_computed() {
        let input = [3.0f32, 4.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let eps = 0.0;
        let mut out = [0.0f32; 8];
        rms_norm_per_head_no_weight_into(&input, 2, 4, eps, &mut out);

        // head0: [3,4,0,0], ms=25/4=6.25, rsqrt=1/2.5=0.4 → [1.2,1.6,0,0]
        assert!((out[0] - 1.2).abs() < 1e-5);
        assert!((out[1] - 1.6).abs() < 1e-5);
        assert!((out[2]).abs() < 1e-5);
        // head1: all 1, ms=1, rsqrt=1 → unchanged
        for i in 4..8 {
            assert!((out[i] - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn no_weight_equals_weighted_with_ones() {
        let input: Vec<f32> = (0..16).map(|i| (i as f32 - 7.0) * 0.3).collect();
        let ones = vec![1.0f32; 8];
        let eps = 1e-5;
        let mut a = vec![0.0f32; 16];
        let mut b = vec![0.0f32; 16];
        rms_norm_per_head_into(&input, &ones, 2, 8, eps, &mut a);
        rms_norm_per_head_no_weight_into(&input, 2, 8, eps, &mut b);
        for i in 0..16 {
            assert!((a[i] - b[i]).abs() < 1e-6, "i={i}");
        }
    }
}

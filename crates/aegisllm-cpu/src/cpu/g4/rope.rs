//! Gemma-4 RoPE: half-split rotation with optional partial RoPE and
//! per-layer theta. Mirrors the CUDA `aegis_apply_rope_ptr` kernel
//! (`norm_rope_kv.cu:465-492`) and `rope_inv_freq_device`
//! (`norm_rope_kv.cu:437-462`).
//!
//! - half-split: pair `i` is `(row[i], row[i + head_dim/2])`.
//! - partial RoPE (global layers): only the first `partial_dim/2` pairs are
//!   rotated; dims `[partial_dim, head_dim)` pass through unchanged.
//!   `partial_dim == 0` means full RoPE.
//! - `inv_freq(i) = 1 / theta^(2i / head_dim)` — denominator is the FULL
//!   head_dim even under partial RoPE — plus optional llama3-style scaling.
//! - per-layer theta: sliding layers use `rope_theta_sliding` (10k), global
//!   layers use `rope_theta_global` (1M).

use aegisllm_base::error::{AegisError, Result};

/// Per-layer RoPE configuration. `theta` is the already-resolved per-layer
/// base (sliding vs global). The llama3-scaling fields mirror the CUDA
/// `DeviceRopeConfig` / CPU `RopeConfig`.
#[derive(Debug, Clone)]
pub(crate) struct G4RopeConfig {
    pub(crate) theta: f32,
    pub(crate) factor: f32,
    pub(crate) low_freq_factor: Option<f32>,
    pub(crate) high_freq_factor: Option<f32>,
    pub(crate) original_max_position_embeddings: Option<usize>,
}

impl G4RopeConfig {
    fn inv_freq(&self, index: usize, head_dim: usize) -> f32 {
        let freq = 1.0 / self.theta.powf((index * 2) as f32 / head_dim as f32);
        if self.factor == 1.0 {
            return freq;
        }
        let low = self.low_freq_factor.unwrap_or(1.0);
        let high = self.high_freq_factor.unwrap_or(low);
        let original = self.original_max_position_embeddings.unwrap_or(8192) as f32;
        let wavelength = 2.0 * std::f32::consts::PI / freq.max(1e-12);
        if wavelength > original / low {
            freq / self.factor
        } else if wavelength < original / high || (high - low).abs() < 1e-12 {
            freq
        } else {
            let smooth = ((original / wavelength) - low) / (high - low);
            let smooth = smooth.clamp(0.0, 1.0);
            (1.0 - smooth) * (freq / self.factor) + smooth * freq
        }
    }
}

/// Apply half-split RoPE in place over `n_heads` rows of `head_dim`.
/// `partial_dim == 0` rotates the full head; otherwise only the first
/// `partial_dim/2` pairs are rotated.
pub(crate) fn apply_rope_partial_in_place(
    values: &mut [f32],
    position: usize,
    n_heads: usize,
    head_dim: usize,
    partial_dim: usize,
    rope: &G4RopeConfig,
) -> Result<()> {
    if values.len() != n_heads * head_dim {
        return Err(AegisError::InvalidPlan(format!(
            "rope shape mismatch: expected {}, got {}",
            n_heads * head_dim,
            values.len()
        )));
    }
    let half_dim = head_dim / 2;
    let partial_half = if partial_dim > 0 {
        (partial_dim / 2).min(half_dim)
    } else {
        half_dim
    };
    // Precompute cos/sin for the rotated pairs once for all heads at this position.
    let mut cos_table = vec![0.0_f32; partial_half];
    let mut sin_table = vec![0.0_f32; partial_half];
    for i in 0..partial_half {
        let angle = position as f32 * rope.inv_freq(i, head_dim);
        let (s, c) = angle.sin_cos();
        cos_table[i] = c;
        sin_table[i] = s;
    }
    for head in 0..n_heads {
        let row = &mut values[head * head_dim..(head + 1) * head_dim];
        for i in 0..partial_half {
            let x0 = row[i];
            let x1 = row[i + half_dim];
            let c = cos_table[i];
            let s = sin_table[i];
            row[i] = x0 * c - x1 * s;
            row[i + half_dim] = x0 * s + x1 * c;
        }
        // dims [partial_half, half_dim) and their partners pass through unchanged.
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rope_no_scaling(theta: f32) -> G4RopeConfig {
        G4RopeConfig {
            theta,
            factor: 1.0,
            low_freq_factor: None,
            high_freq_factor: None,
            original_max_position_embeddings: None,
        }
    }

    #[test]
    fn full_rope_position_zero_is_identity() {
        let rope = rope_no_scaling(10_000.0);
        let mut v = vec![1.0f32, 2.0, 3.0, 4.0];
        let orig = v.clone();
        apply_rope_partial_in_place(&mut v, 0, 1, 4, 0, &rope).unwrap();
        for i in 0..4 {
            assert!((v[i] - orig[i]).abs() < 1e-6, "pos0 must be identity");
        }
    }

    #[test]
    fn full_rope_matches_hand_computed_pair() {
        // head_dim=2, single pair. theta irrelevant for i=0 (inv_freq=1).
        // angle = position * 1.0 = 1.0 rad. row=[x0,x1].
        let rope = rope_no_scaling(10_000.0);
        let mut v = vec![1.0f32, 0.0]; // x0=1, x1=0
        apply_rope_partial_in_place(&mut v, 1, 1, 2, 0, &rope).unwrap();
        let (s, c) = 1.0f32.sin_cos();
        // row[0] = x0*c - x1*s = c ; row[1] = x0*s + x1*c = s
        assert!((v[0] - c).abs() < 1e-6);
        assert!((v[1] - s).abs() < 1e-6);
    }

    #[test]
    fn partial_rope_leaves_tail_unrotated() {
        // head_dim=8, half=4, partial_dim=4 → partial_half=2. Pairs (0,4) and
        // (1,5) rotate; pairs (2,6),(3,7) pass through.
        let rope = rope_no_scaling(1_000_000.0);
        let input: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let mut v = input.clone();
        apply_rope_partial_in_place(&mut v, 3, 1, 8, 4, &rope).unwrap();
        // Indices 2,3 (first-halves not rotated) and 6,7 (their partners) unchanged.
        assert!((v[2] - input[2]).abs() < 1e-6);
        assert!((v[3] - input[3]).abs() < 1e-6);
        assert!((v[6] - input[6]).abs() < 1e-6);
        assert!((v[7] - input[7]).abs() < 1e-6);
        // Pair (0,4) rotated.
        let half = 4usize;
        let c0 = (3.0f32 * rope.inv_freq(0, 8)).cos();
        let s0 = (3.0f32 * rope.inv_freq(0, 8)).sin();
        let exp0 = input[0] * c0 - input[half] * s0;
        let exp4 = input[0] * s0 + input[half] * c0;
        assert!((v[0] - exp0).abs() < 1e-5);
        assert!((v[half] - exp4).abs() < 1e-5);
    }

    #[test]
    fn inv_freq_denominator_uses_full_head_dim() {
        // Under partial RoPE the inv_freq denominator must still be the full
        // head_dim (256/512), NOT partial_dim. Index 1, head_dim=512.
        let rope = rope_no_scaling(1_000_000.0);
        let want = 1.0 / 1_000_000.0f32.powf(2.0 / 512.0);
        assert!((rope.inv_freq(1, 512) - want).abs() < 1e-9);
    }
}

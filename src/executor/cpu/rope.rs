use crate::artifact::{HfRopeScaling, ModelArtifact};
use crate::error::{AegisError, Result};

#[derive(Debug, Clone)]
pub(super) struct RopeConfig {
    pub(super) theta: f32,
    factor: f32,
    low_freq_factor: Option<f32>,
    high_freq_factor: Option<f32>,
    original_max_position_embeddings: Option<usize>,
}

impl RopeConfig {
    pub(super) fn from_artifact(artifact: &ModelArtifact) -> Self {
        let scaling = artifact.config.rope_scaling.as_ref();
        Self {
            theta: artifact.config.rope_theta.unwrap_or(10_000.0) as f32,
            factor: scaling.and_then(|value| value.factor).unwrap_or(1.0) as f32,
            low_freq_factor: scaling_f32(scaling, |value| value.low_freq_factor),
            high_freq_factor: scaling_f32(scaling, |value| value.high_freq_factor),
            original_max_position_embeddings: scaling
                .and_then(|value| value.original_max_position_embeddings),
        }
    }
}

fn scaling_f32(
    scaling: Option<&HfRopeScaling>,
    field: impl FnOnce(&HfRopeScaling) -> Option<f64>,
) -> Option<f32> {
    scaling.and_then(field).map(|value| value as f32)
}

pub(super) fn apply_rope_in_place(
    values: &mut [f32],
    position: usize,
    num_heads: usize,
    head_dim: usize,
    rope: &RopeConfig,
) -> Result<()> {
    if values.len() != num_heads * head_dim {
        return Err(AegisError::InvalidPlan(format!(
            "rope shape mismatch: expected {}, got {}",
            num_heads * head_dim,
            values.len()
        )));
    }
    let half_dim = head_dim / 2;
    for head in 0..num_heads {
        let row = &mut values[head * head_dim..(head + 1) * head_dim];
        for i in 0..half_dim {
            let angle = position as f32 * rope_inv_freq(i, head_dim, rope);
            let (sin, cos) = angle.sin_cos();
            let x0 = row[i];
            let x1 = row[i + half_dim];
            row[i] = x0 * cos - x1 * sin;
            row[i + half_dim] = x0 * sin + x1 * cos;
        }
    }
    Ok(())
}

fn rope_inv_freq(index: usize, head_dim: usize, rope: &RopeConfig) -> f32 {
    let freq = 1.0 / rope.theta.powf((index * 2) as f32 / head_dim as f32);
    if rope.factor == 1.0 {
        return freq;
    }

    let low = rope.low_freq_factor.unwrap_or(1.0);
    let high = rope.high_freq_factor.unwrap_or(low);
    let original = rope.original_max_position_embeddings.unwrap_or(8192) as f32;
    let wavelength = 2.0 * std::f32::consts::PI / freq.max(1e-12);
    if wavelength > original / low {
        freq / rope.factor
    } else if wavelength < original / high || (high - low).abs() < 1e-12 {
        freq
    } else {
        let smooth = ((original / wavelength) - low) / (high - low);
        let smooth = smooth.clamp(0.0, 1.0);
        (1.0 - smooth) * (freq / rope.factor) + smooth * freq
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_scaling_without_llama3_fields_is_identity() {
        let rope = RopeConfig {
            theta: 10_000.0,
            factor: 1.0,
            low_freq_factor: None,
            high_freq_factor: None,
            original_max_position_embeddings: None,
        };
        assert!((rope_inv_freq(0, 128, &rope) - 1.0).abs() < 1e-6);
    }
}

use aegisllm_base::artifact::{HfRopeScaling, ModelArtifact};
use crate::cuda::DeviceRopeConfig;
use aegisllm_base::error::{AegisError, Result};

#[derive(Debug, Clone, Copy)]
pub(super) struct RopeConfig {
    theta: f32,
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

    pub(super) fn to_device(self) -> Result<DeviceRopeConfig> {
        let low = self.low_freq_factor.unwrap_or(1.0);
        let original_max_position_embeddings =
            u32::try_from(self.original_max_position_embeddings.unwrap_or(8192)).map_err(|_| {
                AegisError::InvalidPlan(format!(
                    "RoPE original_max_position_embeddings exceeds u32: {:?}",
                    self.original_max_position_embeddings
                ))
            })?;
        Ok(DeviceRopeConfig {
            theta: self.theta,
            factor: self.factor,
            low_freq_factor: low,
            high_freq_factor: self.high_freq_factor.unwrap_or(low),
            original_max_position_embeddings,
        })
    }
}

fn scaling_f32(
    scaling: Option<&HfRopeScaling>,
    field: impl FnOnce(&HfRopeScaling) -> Option<f64>,
) -> Option<f32> {
    scaling.and_then(field).map(|value| value as f32)
}

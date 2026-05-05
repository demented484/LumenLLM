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

    /// Produce a `DeviceRopeConfig` for a specific layer.
    /// `partial_dim` > 0 enables p-RoPE for Gemma 4 global layers; 0 = full RoPE.
    pub(super) fn to_device_with_partial_dim(self, partial_dim: usize) -> Result<DeviceRopeConfig> {
        self.to_device_with_partial_dim_and_theta(partial_dim, None)
    }

    /// Same as `to_device_with_partial_dim` but allows overriding `theta` per-layer.
    /// Gemma 4 uses different `rope_theta` for sliding (10k) and global (1M) layers.
    pub(super) fn to_device_with_partial_dim_and_theta(
        self,
        partial_dim: usize,
        theta_override: Option<f32>,
    ) -> Result<DeviceRopeConfig> {
        let low = self.low_freq_factor.unwrap_or(1.0);
        let original_max_position_embeddings =
            u32::try_from(self.original_max_position_embeddings.unwrap_or(8192)).map_err(|_| {
                AegisError::InvalidPlan(format!(
                    "RoPE original_max_position_embeddings exceeds u32: {:?}",
                    self.original_max_position_embeddings
                ))
            })?;
        Ok(DeviceRopeConfig {
            theta: theta_override.unwrap_or(self.theta),
            factor: self.factor,
            low_freq_factor: low,
            high_freq_factor: self.high_freq_factor.unwrap_or(low),
            original_max_position_embeddings,
            partial_dim: partial_dim as u32,
        })
    }

}

fn scaling_f32(
    scaling: Option<&HfRopeScaling>,
    field: impl FnOnce(&HfRopeScaling) -> Option<f64>,
) -> Option<f32> {
    scaling.and_then(field).map(|value| value as f32)
}

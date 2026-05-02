use cudarc::driver::{LaunchConfig, PushKernelArg};

use super::{CudaRuntime, ceil_div, map_cuda_err};
use crate::cuda::DeviceBuffer;
use aegisllm_base::error::{AegisError, Result};

impl CudaRuntime {
    pub fn argmax_f32_device(
        &self,
        logits: &DeviceBuffer<f32>,
        block_values: &mut DeviceBuffer<f32>,
        block_indices: &mut DeviceBuffer<u32>,
        output_token: &mut DeviceBuffer<u32>,
    ) -> Result<()> {
        if logits.len() == 0 || output_token.len() != 1 {
            return Err(AegisError::InvalidPlan(format!(
                "argmax shape mismatch: logits={} output={}",
                logits.len(),
                output_token.len()
            )));
        }
        let blocks = ceil_div(logits.len() as u32, 256);
        if block_values.len() != blocks as usize || block_indices.len() != blocks as usize {
            return Err(AegisError::InvalidPlan(format!(
                "argmax scratch mismatch: expected blocks={} values={} indices={}",
                blocks,
                block_values.len(),
                block_indices.len()
            )));
        }
        let len = logits.len() as u32;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.argmax_blocks)
                .arg(&logits.slice)
                .arg(&len)
                .arg(&mut block_values.slice)
                .arg(&mut block_indices.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch argmax block reduce"))?;

        let finalize_cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.argmax_finalize)
                .arg(&block_values.slice)
                .arg(&block_indices.slice)
                .arg(&blocks)
                .arg(&mut output_token.slice)
                .launch(finalize_cfg)
        }
        .map_err(map_cuda_err("launch argmax finalize"))?;
        Ok(())
    }
}

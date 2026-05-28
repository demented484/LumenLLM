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
        if logits.is_empty() || output_token.len() != 1 {
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

    /// Speculative-decoding sparse lm_head matvec.
    ///
    /// Evaluates the dense BF16 lm_head ONLY over the explicit list of
    /// `candidate_rows` token ids: `logits[i] = lm_head[candidate_rows[i], :] ·
    /// hidden`. One block per candidate (mirrors `aegis_bf16_matvec_reference`).
    ///
    /// `lm_head` MUST be VRAM-resident — the candidate-gather kernel indexes the
    /// matrix rows directly on device. The draft's tied embed/lm_head is small
    /// (262144 × 256 BF16 ≈ 134 MiB) so it always loads VRAM-resident.
    ///
    /// TODO(gpu-verify): the centroid → candidate-row mapping is computed on the
    /// host (see `executor::speculative`); this kernel only consumes the row
    /// list, so verify the row ids against a reference centroid decode.
    pub fn spec_sparse_lm_head_matvec_device(
        &self,
        lm_head: &crate::cuda::DeviceBf16Matrix,
        hidden: &DeviceBuffer<f32>,
        candidate_rows: &DeviceBuffer<u32>,
        num_candidates: usize,
        logits: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if lm_head.is_host_resident() {
            return Err(AegisError::InvalidPlan(format!(
                "spec sparse lm_head `{}` must be VRAM-resident",
                lm_head.name
            )));
        }
        if hidden.len() < lm_head.cols {
            return Err(AegisError::InvalidPlan(format!(
                "spec sparse lm_head hidden too small: have {} need {}",
                hidden.len(),
                lm_head.cols
            )));
        }
        if candidate_rows.len() < num_candidates || logits.len() < num_candidates {
            return Err(AegisError::InvalidPlan(format!(
                "spec sparse lm_head buffer mismatch: candidates={} rows_cap={} logits_cap={}",
                num_candidates,
                candidate_rows.len(),
                logits.len()
            )));
        }
        if num_candidates == 0 {
            return Ok(());
        }
        let cols = lm_head.cols as u32;
        let n = num_candidates as u32;
        let block_dim = 256u32;
        let cfg = LaunchConfig {
            grid_dim: (n, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: block_dim * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.spec_sparse_lm_head_matvec)
                .arg(lm_head.values_u16())
                .arg(&hidden.slice)
                .arg(&candidate_rows.slice)
                .arg(&n)
                .arg(&cols)
                .arg(&mut logits.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch spec sparse lm_head matvec"))?;
        Ok(())
    }
}

use super::attention::forward_attention_device;
use super::block::{CudaLayerBlockExecutor, CudaLayerBlockState};
use super::mlp::forward_mlp_device;
use super::state::{CudaLayer, CudaLayerState, CudaLlamaExecutor, CudaLlamaState, CudaScratch};
use crate::cuda::{CudaRuntime, DeviceBuffer, DeviceRopeConfig};
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::generation::SamplingConfig;

#[derive(Debug, Clone, Copy)]
pub(super) struct CudaLayerForwardParams {
    pub(super) rms_norm_eps: f32,
    pub(super) position: usize,
    pub(super) num_attention_heads: usize,
    pub(super) num_kv_heads: usize,
    pub(super) head_dim: usize,
    pub(super) kv_context_size: usize,
    pub(super) rope: DeviceRopeConfig,
}

impl CudaLlamaExecutor {
    pub(super) fn forward_hidden(&self, state: &mut CudaLlamaState, token_id: usize) -> Result<()> {
        if state.position >= self.kv_context_size {
            return Err(AegisError::InvalidPlan(format!(
                "kv cache context exhausted: position={} context={}",
                state.position, self.kv_context_size
            )));
        }
        self.runtime
            .bf16_row_to_f32_device(&self.embed_tokens, token_id, &mut state.hidden)?;
        let rope = self.rope.to_device()?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let layer_state = &mut state.layers[layer_idx];
            forward_cuda_layer_device(
                &self.runtime,
                layer,
                layer_state,
                &mut state.hidden,
                &mut state.scratch,
                CudaLayerForwardParams {
                    rms_norm_eps: self.rms_norm_eps,
                    position: state.position,
                    num_attention_heads: self.num_attention_heads,
                    num_kv_heads: self.num_kv_heads,
                    head_dim: self.head_dim,
                    kv_context_size: self.kv_context_size,
                    rope,
                },
            )?;
        }
        state.position += 1;
        Ok(())
    }

    fn read_logits(&self, state: &mut CudaLlamaState) -> Result<Vec<f32>> {
        self.runtime.rms_norm_device(
            &state.hidden,
            &self.final_norm,
            self.rms_norm_eps,
            &mut state.scratch.final_hidden,
        )?;
        self.runtime.matvec_bf16_reference_device(
            &self.lm_head,
            &state.scratch.final_hidden,
            &mut state.logits,
        )?;
        self.runtime.download_f32(&state.logits)
    }

    pub(super) fn sample_next_from_current_hidden(
        &self,
        state: &mut CudaLlamaState,
        sampling: &SamplingConfig,
    ) -> Result<usize> {
        self.runtime.rms_norm_device(
            &state.hidden,
            &self.final_norm,
            self.rms_norm_eps,
            &mut state.scratch.final_hidden,
        )?;
        self.runtime.matvec_bf16_reference_device(
            &self.lm_head,
            &state.scratch.final_hidden,
            &mut state.logits,
        )?;
        if sampling.temperature > 0.0 && sampling.top_k != 1 {
            let logits = self.runtime.download_f32(&state.logits)?;
            return aegisllm_base::executor::generation::sample_next_token(&logits, sampling);
        }
        self.runtime.argmax_f32_device(
            &state.logits,
            &mut state.scratch.argmax_block_values,
            &mut state.scratch.argmax_block_indices,
            &mut state.sampled_token,
        )?;
        let token = self.runtime.download_u32(&state.sampled_token)?;
        token
            .first()
            .copied()
            .map(|token| token as usize)
            .ok_or_else(|| AegisError::InvalidPlan("CUDA argmax returned no token".into()))
    }

    pub(super) fn forward_logits(
        &self,
        state: &mut CudaLlamaState,
        token_id: usize,
    ) -> Result<Vec<f32>> {
        self.forward_hidden(state, token_id)?;
        self.read_logits(state)
    }

    pub(super) fn forward_next_token(
        &self,
        state: &mut CudaLlamaState,
        token_id: usize,
        sampling: &SamplingConfig,
    ) -> Result<usize> {
        if sampling.temperature > 0.0 && sampling.top_k != 1 {
            let logits = self.forward_logits(state, token_id)?;
            return aegisllm_base::executor::generation::sample_next_token(&logits, sampling);
        }
        self.forward_hidden(state, token_id)?;
        self.sample_next_from_current_hidden(state, sampling)
    }
}

impl CudaLayerBlockExecutor {
    #[allow(dead_code)]
    pub fn forward_layer_host(
        &self,
        state: &mut CudaLayerBlockState,
        layer_idx: usize,
        position: usize,
        hidden: &[f32],
    ) -> Result<Vec<f32>> {
        if hidden.len() != self.hidden_size {
            return Err(AegisError::InvalidPlan(format!(
                "hybrid CUDA layer input mismatch: expected {}, got {}",
                self.hidden_size,
                hidden.len()
            )));
        }
        state.hidden = self.runtime.upload_f32(hidden)?;
        self.forward_layer_device(state, layer_idx, position)?;
        self.runtime.download_f32(&state.hidden)
    }

    #[allow(dead_code)]
    fn forward_layer_device(
        &self,
        state: &mut CudaLayerBlockState,
        layer_idx: usize,
        position: usize,
    ) -> Result<()> {
        let layer = self.layers.get(&layer_idx).ok_or_else(|| {
            AegisError::InvalidPlan(format!("missing CUDA hybrid layer `{layer_idx}`"))
        })?;
        let layer_state = state.layers.get_mut(&layer_idx).ok_or_else(|| {
            AegisError::InvalidPlan(format!("missing CUDA hybrid layer state `{layer_idx}`"))
        })?;
        let rope = self.rope.to_device()?;
        forward_cuda_layer_device(
            &self.runtime,
            layer,
            layer_state,
            &mut state.hidden,
            &mut state.scratch,
            CudaLayerForwardParams {
                rms_norm_eps: self.rms_norm_eps,
                position,
                num_attention_heads: self.num_attention_heads,
                num_kv_heads: self.num_kv_heads,
                head_dim: self.head_dim,
                kv_context_size: self.kv_context_size,
                rope,
            },
        )?;
        Ok(())
    }
}

pub(super) fn forward_cuda_layer_device(
    runtime: &CudaRuntime,
    layer: &CudaLayer,
    layer_state: &mut CudaLayerState,
    hidden: &mut DeviceBuffer<f32>,
    scratch: &mut CudaScratch,
    params: CudaLayerForwardParams,
) -> Result<()> {
    forward_attention_device(
        runtime,
        layer,
        layer_state,
        hidden,
        scratch,
        params.rms_norm_eps,
        params.position,
        params.num_attention_heads,
        params.num_kv_heads,
        params.head_dim,
        params.kv_context_size,
        params.rope,
    )?;
    forward_mlp_device(runtime, layer, scratch, params.rms_norm_eps)?;
    std::mem::swap(hidden, &mut scratch.hidden_out);
    Ok(())
}

mod attention;
mod batch;
mod gemm;
mod kv;
mod layer;
mod scheduler;
mod timings;

use std::time::Instant;

use layer::{CudaPrefillForwardParams, forward_cuda_layer_prefill_chunk_device};
use timings::{print_prefill_stage_timings, record_prefill_stage};

use super::state::{CudaLlamaExecutor, CudaLlamaState};
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::generation::SamplingConfig;

impl CudaLlamaExecutor {
    pub(super) fn prefill_prompt(
        &self,
        state: &mut CudaLlamaState,
        prompt_tokens: &[usize],
        sampling: &SamplingConfig,
    ) -> Result<usize> {
        let Some((&last, prefix)) = prompt_tokens.split_last() else {
            return Err(AegisError::InvalidConfig(
                "prompt produced no tokens".into(),
            ));
        };
        let end_position = state
            .position
            .checked_add(prompt_tokens.len())
            .ok_or_else(|| {
                AegisError::InvalidPlan(format!(
                    "prefill position overflow: position={} prompt_tokens={}",
                    state.position,
                    prompt_tokens.len()
                ))
            })?;
        if end_position > self.kv_context_size {
            return Err(AegisError::InvalidPlan(format!(
                "prefill would exhaust kv cache: position={} prompt_tokens={} context={}",
                state.position,
                prompt_tokens.len(),
                self.kv_context_size
            )));
        }
        if prompt_tokens.len() > 1 && state.prefill.is_some() {
            return self.prefill_prompt_chunked(state, prompt_tokens, sampling);
        }
        for &token in prefix {
            self.forward_hidden(state, token)?;
        }
        self.forward_next_token(state, last, sampling)
    }

    fn prefill_prompt_chunked(
        &self,
        state: &mut CudaLlamaState,
        prompt_tokens: &[usize],
        sampling: &SamplingConfig,
    ) -> Result<usize> {
        state.prefill_timings.reset();
        let rope = self.rope.to_device()?;
        let chunk_size = state.prefill.as_ref().map(|s| s.chunk_size).unwrap_or(1);
        for chunk in prompt_tokens.chunks(chunk_size) {
            let start_position = state.position;
            let prefill = state
                .prefill
                .as_mut()
                .ok_or_else(|| AegisError::InvalidPlan("CUDA prefill scratch is missing".into()))?;
            let prepare_start = Instant::now();
            let batch_meta = prefill.prepare_dense_batch(
                &self.runtime,
                chunk,
                start_position,
                self.kv_context_size,
                self.embed_tokens.rows,
            )?;
            record_prefill_stage(
                &self.runtime,
                &mut state.prefill_timings,
                prepare_start,
                |timings, elapsed| timings.prepare_us += elapsed,
            )?;
            state.prefill_timings.chunks += 1;

            let embed_start = Instant::now();
            self.runtime.bf16_rows_to_f32_device(
                &self.embed_tokens,
                &prefill.tokens,
                batch_meta.num_prefill_tokens,
                &mut prefill.hidden,
            )?;
            record_prefill_stage(
                &self.runtime,
                &mut state.prefill_timings,
                embed_start,
                |timings, elapsed| timings.embed_us += elapsed,
            )?;

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                let layer_state = &mut state.layers[layer_idx];
                let layer_start = Instant::now();
                forward_cuda_layer_prefill_chunk_device(
                    &self.runtime,
                    layer,
                    layer_state,
                    prefill,
                    CudaPrefillForwardParams {
                        rms_norm_eps: self.rms_norm_eps,
                        start_position: batch_meta.start_position,
                        batch: batch_meta.num_prefill_tokens,
                        num_sequences: batch_meta.num_sequences,
                        dense_metadata: batch_meta.dense_metadata,
                        num_attention_heads: self.num_attention_heads,
                        num_kv_heads: self.num_kv_heads,
                        head_dim: self.head_dim,
                        kv_context_size: self.kv_context_size,
                        rope,
                    },
                    &mut state.prefill_timings,
                )?;
                record_prefill_stage(
                    &self.runtime,
                    &mut state.prefill_timings,
                    layer_start,
                    |timings, elapsed| timings.layer_total_us += elapsed,
                )?;
            }
            self.runtime.copy_row_f32_device(
                &prefill.hidden,
                batch_meta.num_prefill_tokens - 1,
                self.hidden_size,
                &mut state.hidden,
            )?;
            state.position += batch_meta.num_prefill_tokens;
        }

        let sample_start = Instant::now();
        let next = self.sample_next_from_current_hidden(state, sampling)?;
        record_prefill_stage(
            &self.runtime,
            &mut state.prefill_timings,
            sample_start,
            |timings, elapsed| timings.sample_us += elapsed,
        )?;
        print_prefill_stage_timings(state.prefill_timings);
        Ok(next)
    }
}

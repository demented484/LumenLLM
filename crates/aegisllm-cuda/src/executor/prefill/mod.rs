mod attention;
mod batch;
mod gemm;
mod kv;
mod layer;
mod moe;
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
            // Gemma 4 ScaledWordEmbedding: embeddings get multiplied by sqrt(hidden_size).
            // The decode path applies this in `forward_hidden`; chunked prefill must mirror.
            // Without it the first layer's RMS-norm sees magnitudes ~53× too small,
            // attention collapses, and the rest of the model diverges into garbage.
            if let Some(scale) = self.embed_scale {
                let total = batch_meta
                    .num_prefill_tokens
                    .checked_mul(self.hidden_size)
                    .ok_or_else(|| AegisError::InvalidPlan(
                        "prefill embed_scale length overflow".into(),
                    ))?;
                self.runtime
                    .scale_f32_device_len(scale, &mut prefill.hidden, total)?;
            }
            if let Ok(tag) = std::env::var("AEGIS_DUMP_EMBED") {
                let h = self.runtime.download_f32(&prefill.hidden)?;
                eprintln!("[DUMP {tag} EMBED] first8={:?}", &h[0..8.min(h.len())]);
            }

            // Stage I.2 image injection. After embed lookup + embed_scale,
            // overwrite slots whose token id == image_token_id with rows from
            // the VRAM-resident image-embeddings buffer. Image embeddings are
            // already scaled by sqrt(hidden) by the vision pooler — DO NOT
            // apply embed_scale to them. We mirror that by overwriting AFTER
            // the embed_scale step.
            // Implementation: CPU-side splice via download/modify/upload.
            // Tolerable cost: hidden ≈ 9 KB/token × ~280 image tokens + ~chunk
            // text tokens = a few MB per prefill chunk, one-shot.
            if state.image_embeds.is_some()
                && state.image_n_tokens > 0
                && chunk.iter().any(|&t| t as u32 == state.image_token_id)
            {
                let h = self.hidden_size;
                let n_img = state.image_n_tokens;
                let img_tok_id = state.image_token_id;
                let img_data = self.runtime.download_f32(
                    state.image_embeds.as_ref().unwrap()
                )?;
                let mut hidden_host = self.runtime.download_f32(&prefill.hidden)?;
                let mut img_row_idx = 0usize;
                for (slot, &tok) in chunk.iter().enumerate() {
                    if tok as u32 != img_tok_id { continue; }
                    let src_row = img_row_idx % n_img;
                    img_row_idx += 1;
                    let src_off = src_row * h;
                    let dst_off = slot * h;
                    hidden_host[dst_off..dst_off + h]
                        .copy_from_slice(&img_data[src_off..src_off + h]);
                }
                self.runtime.upload_f32_slice_to_device(&hidden_host, &mut prefill.hidden)?;
            }
            record_prefill_stage(
                &self.runtime,
                &mut state.prefill_timings,
                embed_start,
                |timings, elapsed| timings.embed_us += elapsed,
            )?;

            let staging_ptr = state.scratch.staging_pool
                .as_deref_mut()
                .map_or(std::ptr::null_mut(), |p| p as *mut _);
            let kv_staging_ptr = state.scratch.kv_staging
                .as_deref_mut()
                .map_or(std::ptr::null_mut(), |p| p as *mut _);
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
                        staging_ptr,
                        kv_staging_ptr,
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

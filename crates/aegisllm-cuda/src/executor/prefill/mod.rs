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
                let mut slot_positions: Vec<usize> = Vec::new();
                for (slot, &tok) in chunk.iter().enumerate() {
                    if tok as u32 != img_tok_id { continue; }
                    let src_row = img_row_idx % n_img;
                    img_row_idx += 1;
                    let src_off = src_row * h;
                    let dst_off = slot * h;
                    hidden_host[dst_off..dst_off + h]
                        .copy_from_slice(&img_data[src_off..src_off + h]);
                    slot_positions.push(slot);
                }
                if std::env::var("AEGIS_DEBUG_INJECT").is_ok() {
                    eprintln!(
                        "[inject] chunk_len={} n_img={} injected_rows={}",
                        chunk.len(), n_img, img_row_idx
                    );
                    eprintln!(
                        "[inject] first slots: {:?}, last slots: {:?}",
                        &slot_positions[..slot_positions.len().min(5)],
                        &slot_positions[slot_positions.len().saturating_sub(5)..]
                    );
                    // Sample pre/at/post values to verify injection: hidden[slot_positions[0]][0..6]
                    if !slot_positions.is_empty() {
                        let s0 = slot_positions[0];
                        let pre = if s0 > 0 { s0 - 1 } else { 0 };
                        eprintln!(
                            "[inject] pre-slot {} hidden[0..6] = {:?}",
                            pre, &hidden_host[pre * h..pre * h + 6]
                        );
                        eprintln!(
                            "[inject] img-slot {} hidden[0..6] = {:?}",
                            s0, &hidden_host[s0 * h..s0 * h + 6]
                        );
                        eprintln!(
                            "[inject] img_data row 0[0..6]  = {:?}",
                            &img_data[..6]
                        );
                    }
                }
                self.runtime.upload_f32_slice_to_device(&hidden_host, &mut prefill.hidden)?;
            }

            // Audio soft-token injection — exact parallel of the image splice
            // above. Overwrite slots whose token id == audio_token_id with
            // consecutive rows from the VRAM-resident audio-embeddings buffer
            // (already in text-hidden space; NOT scaled by embed_scale, so we
            // overwrite AFTER the embed_scale step). Gated so non-audio prompts
            // are unaffected.
            if state.audio_embeds.is_some()
                && state.audio_n_tokens > 0
                && chunk.iter().any(|&t| t as u32 == state.audio_token_id)
            {
                let h = self.hidden_size;
                let n_aud = state.audio_n_tokens;
                let aud_tok_id = state.audio_token_id;
                let aud_data = self.runtime.download_f32(
                    state.audio_embeds.as_ref().unwrap()
                )?;
                // TODO(gpu-verify): this copies `h = hidden_size` floats per row
                // and assumes the audio embeddings were produced with stride
                // `hidden_size` (i.e. embed_audio.rows == model hidden_size,
                // 2560 for E4B). If the projector's output width differs from
                // hidden_size the row stride here is wrong — guard/assert on GPU.
                let mut hidden_host = self.runtime.download_f32(&prefill.hidden)?;
                let mut aud_row_idx = 0usize;
                for (slot, &tok) in chunk.iter().enumerate() {
                    if tok as u32 != aud_tok_id { continue; }
                    let src_row = aud_row_idx % n_aud;
                    aud_row_idx += 1;
                    let src_off = src_row * h;
                    let dst_off = slot * h;
                    hidden_host[dst_off..dst_off + h]
                        .copy_from_slice(&aud_data[src_off..src_off + h]);
                }
                if std::env::var("AEGIS_DEBUG_INJECT").is_ok() {
                    eprintln!(
                        "[inject-audio] chunk_len={} n_aud={} injected_rows={}",
                        chunk.len(), n_aud, aud_row_idx
                    );
                }
                self.runtime.upload_f32_slice_to_device(&hidden_host, &mut prefill.hidden)?;
            }
            // PLE (Per-Layer Embeddings) token-entry compute for this chunk —
            // mirrors `forward_next_token`/`forward_hidden` on the decode side.
            // Without this call, the chunked prefill ran 42 layers WITHOUT the
            // per-layer additive contribution → bias accumulates → prompt-final
            // hidden picks the wrong first decode token. This was the root
            // cause of the "coherent text in wrong language" output on E4B.
            if let Some(ref ple) = self.ple {
                crate::executor::ple::compute_per_layer_inputs_prefill_chunk(
                    &self.runtime, ple,
                    chunk,
                    self.hidden_size,
                    self.layers.len(),
                    prefill,
                    self.rms_norm_eps,
                )?;
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
            let n_layers = self.layers.len();
            for layer_idx in 0..self.layers.len() {
                let layer = &self.layers[layer_idx];
                // KV-cache sharing (Gemma-4 E4B/E2B): the last `num_kv_shared_layers`
                // layers attend with their OWN query against a PARENT layer's cached
                // K/V (their own k/v weights are vestigial at inference — see the
                // decode path in forward.rs). Split-borrow `state.layers` so we hold a
                // mutable borrow of THIS layer's state and an immutable borrow of the
                // parent's KV (parent index < layer_idx by construction). Without this
                // the chunked prefill computed each shared layer's own K/V and attended
                // to it — diverging hard at layer `n_layers - n_shared` (the first
                // shared layer) and corrupting the prompt-final hidden state.
                let (left, right) = state.layers.split_at_mut(layer_idx);
                let layer_state = &mut right[0];
                let kv_shared_override = layer
                    .kv_shared_from
                    .and_then(|parent_idx| left.get(parent_idx).map(|s| &s.kv));
                let layer_start = Instant::now();
                forward_cuda_layer_prefill_chunk_device(
                    &self.runtime,
                    layer,
                    layer_idx,
                    self.ple.as_ref(),
                    n_layers,
                    kv_shared_override,
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

    /// Speculative-decode BATCHED VERIFY. Runs ONE target forward over the
    /// `verify_tokens` = `[last_committed, prop0, .., prop_{K-1}]` (K+1 tokens)
    /// at positions `[base_pos, base_pos+K]` via the chunked-prefill machinery,
    /// and returns the K+1 per-position greedy argmax predictions. Does NOT
    /// advance `state.position` or sample — the caller computes the accept
    /// length and rewinds `state.position` to `base_pos + accepted + 1`
    /// (rejected-tail KV slots are positionally overwritten next round; no
    /// explicit truncation needed because the KV is position-addressed).
    ///
    /// `prefill.hidden` is left holding the K+1 hidden rows so the caller can
    /// `copy_row` the correction's conditioning hidden into `state.hidden`.
    pub(super) fn verify_batched(
        &self,
        state: &mut CudaLlamaState,
        verify_tokens: &[usize],
        base_pos: usize,
    ) -> Result<Vec<usize>> {
        let prefill = state
            .prefill
            .as_mut()
            .ok_or_else(|| AegisError::InvalidPlan("spec verify: prefill scratch missing".into()))?;
        let batch_meta = prefill.prepare_dense_batch(
            &self.runtime,
            verify_tokens,
            base_pos,
            self.kv_context_size,
            self.embed_tokens.rows,
        )?;
        let n = batch_meta.num_prefill_tokens;
        // embed + Gemma-4 sqrt(hidden) scale (mirrors prefill_prompt_chunked).
        self.runtime.bf16_rows_to_f32_device(
            &self.embed_tokens, &prefill.tokens, n, &mut prefill.hidden,
        )?;
        if let Some(scale) = self.embed_scale {
            let total = n.checked_mul(self.hidden_size).ok_or_else(|| {
                AegisError::InvalidPlan("spec verify embed_scale overflow".into())
            })?;
            self.runtime.scale_f32_device_len(scale, &mut prefill.hidden, total)?;
        }
        // PLE per-token compute (no-op for non-PLE models). verify_tokens are
        // plain text tokens — no image/audio injection needed.
        if let Some(ref ple) = self.ple {
            crate::executor::ple::compute_per_layer_inputs_prefill_chunk(
                &self.runtime, ple, verify_tokens, self.hidden_size,
                self.layers.len(), prefill, self.rms_norm_eps,
            )?;
        }
        let staging_ptr = state.scratch.staging_pool
            .as_deref_mut().map_or(std::ptr::null_mut(), |p| p as *mut _);
        let kv_staging_ptr = state.scratch.kv_staging
            .as_deref_mut().map_or(std::ptr::null_mut(), |p| p as *mut _);
        let n_layers = self.layers.len();
        for layer_idx in 0..self.layers.len() {
            let layer = &self.layers[layer_idx];
            let (left, right) = state.layers.split_at_mut(layer_idx);
            let layer_state = &mut right[0];
            let kv_shared_override = layer
                .kv_shared_from
                .and_then(|parent_idx| left.get(parent_idx).map(|s| &s.kv));
            forward_cuda_layer_prefill_chunk_device(
                &self.runtime,
                layer,
                layer_idx,
                self.ple.as_ref(),
                n_layers,
                kv_shared_override,
                layer_state,
                prefill,
                CudaPrefillForwardParams {
                    rms_norm_eps: self.rms_norm_eps,
                    start_position: batch_meta.start_position,
                    batch: n,
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
        }
        // ── Per-row logits: final_norm (batched) → lm_head (batched) → argmax. ──
        // No new kernels; greedy argmax is invariant under the tanh logit
        // softcap so we skip it. prefill.hidden = [n, hidden] from the loop.
        let mut normed = self.runtime.alloc_f32(n * self.hidden_size)?;
        self.runtime.rms_norm_batched_device(
            &prefill.hidden, &self.final_norm, n, self.rms_norm_eps, &mut normed,
        )?;
        let vocab = self.lm_head.rows;
        let mut logits = self.runtime.alloc_f32(n * vocab)?;
        // Prefer cuBLASLt BF16 tensor cores for the K+1-row lm_head over the
        // full 262144 vocab — the F32 reference batched matvec (1 block/row,
        // no tensor cores) dominates the per-round spec-decode overhead. Falls
        // back to the reference kernel for host-resident lm_head.
        if self.runtime.cublaslt_bf16_enabled_for(&self.lm_head) {
            let mut in_bf16 = self.runtime.alloc_u16(n * self.hidden_size)?;
            let mut out_bf16 = self.runtime.alloc_u16(n * vocab)?;
            self.runtime.matmul_bf16_cublaslt_device(
                &self.lm_head, &normed, n, &mut in_bf16, &mut out_bf16, &mut logits,
            )?;
        } else {
            self.runtime
                .matmul_bf16_reference_batched_device(&self.lm_head, &normed, n, &mut logits)?;
        }
        let host = self.runtime.download_f32(&logits)?;
        let mut preds = Vec::with_capacity(n);
        for row in 0..n {
            let slice = &host[row * vocab..(row + 1) * vocab];
            let mut best = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (j, &v) in slice.iter().enumerate() {
                if v > best_v {
                    best_v = v;
                    best = j;
                }
            }
            preds.push(best);
        }
        Ok(preds)
    }
}

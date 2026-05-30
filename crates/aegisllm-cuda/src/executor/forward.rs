use super::attention::forward_attention_device;
use super::block::{CudaLayerBlockExecutor, CudaLayerBlockState};
use super::mlp::forward_mlp_device;
use super::state::{CudaLayer, CudaLayerState, CudaLlamaExecutor, CudaLlamaState, CudaScratch, SendCudaGraph};
use crate::cuda::{CUDA_GRAPH_ATTN_MAX_SEQ_LEN, CudaRuntime, DeviceBuffer};
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::generation::SamplingConfig;

#[derive(Debug, Clone, Copy)]
pub(super) struct CudaLayerForwardParams {
    pub(super) rms_norm_eps: f32,
    pub(super) num_attention_heads: usize,
    pub(super) num_kv_heads: usize,
    pub(super) head_dim: usize,
    pub(super) kv_context_size: usize,
    /// Host-side token position (0-indexed). Needed by host-resident KV staging:
    /// upload `position` existing entries before store, writeback at `position` after.
    pub(super) position: usize,
    /// Host-side sequence length = position + 1.
    pub(super) seq_len: usize,
    /// When `Some(idx)`, layer is host-resident and uses `kv_staging.slots[idx]`.
    pub(super) staging_slot_idx: Option<usize>,
}

impl CudaLlamaExecutor {
    /// Run all transformer layers for the current decode step.
    /// decode_position and decode_seq_len must already be updated before calling this.
    ///
    /// For host-resident KV layers, this orchestrates an async PCIe transfer pipeline:
    /// layer L+1's H2D upload runs on the dedicated transfer stream while layer L's
    /// compute runs on the compute stream. The two ping-pong staging slots ensure
    /// no buffer conflicts. After each host-resident layer's attention, the new KV
    /// slot is asynchronously written back via the transfer stream.
    fn forward_layers_device(&self, state: &mut CudaLlamaState, position: usize) -> Result<()> {
        let base_params = CudaLayerForwardParams {
            rms_norm_eps: self.rms_norm_eps,
            num_attention_heads: self.num_attention_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            kv_context_size: self.kv_context_size,
            position,
            seq_len: position + 1,
            staging_slot_idx: None,
        };
        let kv_width = self.num_kv_heads * self.head_dim;
        let CudaLlamaState {
            ref mut layers,
            ref mut hidden,
            ref mut scratch,
            ref decode_position,
            ref decode_seq_len,
            ..
        } = *state;

        // Pre-issue H2D for layer 0 if it's host-resident (no overlap possible).
        // Holds the H2D event for the layer whose iteration is *about to start*.
        let mut current_h2d_event: Option<cudarc::driver::CudaEvent> =
            issue_h2d_for_layer(&self.runtime, scratch, layers, 0, position, kv_width)?;

        let prof = std::env::var("AEGIS_PROFILE_LAYERS").ok().is_some_and(|v| !v.is_empty());
        let mut layer_times: Vec<(usize, f64)> = Vec::with_capacity(self.layers.len());
        for layer_idx in 0..self.layers.len() {
            let lt0 = std::time::Instant::now();
            let host_resident = layers[layer_idx].kv.is_host_resident();
            let staging_slot_idx = if host_resident { Some(layer_idx % 2) } else { None };

            // Prefetch H2D for the next layer onto the transfer stream BEFORE we
            // start compute on this layer, so it overlaps with this layer's kernels.
            // This is queued on the transfer stream after the current H2D but before
            // any subsequent D2H — so it can run in parallel with compute on the
            // compute stream.
            let next_h2d_event = if layer_idx + 1 < self.layers.len() {
                issue_h2d_for_layer(
                    &self.runtime,
                    scratch,
                    layers,
                    layer_idx + 1,
                    position,
                    kv_width,
                )?
            } else {
                None
            };

            // Make compute stream wait for THIS layer's H2D (if host-resident).
            if host_resident {
                let evt = current_h2d_event.as_ref().ok_or_else(|| {
                    AegisError::InvalidPlan(
                        "missing H2D event for host-resident layer".into(),
                    )
                })?;
                self.runtime.compute_wait_event(evt)?;
            }

            let mut params = base_params;
            params.staging_slot_idx = staging_slot_idx;

            let layer = &self.layers[layer_idx];
            // Split-borrow `layers` so we can hand `forward_attention_device`
            // a mutable borrow of layer N's state AND an immutable borrow of
            // the parent layer's KV cache (when KV-share is active for this
            // layer). split_at_mut returns disjoint halves; the parent index
            // lives in `left` (it's strictly less than this layer's index by
            // construction in `load_cuda_layer`'s post-load pass).
            let (left, right) = layers.split_at_mut(layer_idx);
            let layer_state = &mut right[0];
            let kv_shared_override = layer.kv_shared_from
                .and_then(|parent_idx| left.get(parent_idx).map(|s| &s.kv));
            let lt_attn = std::time::Instant::now();
            super::attention::forward_attention_device(
                &self.runtime,
                layer,
                layer_state,
                kv_shared_override,
                hidden,
                scratch,
                decode_position,
                decode_seq_len,
                params.rms_norm_eps,
                params.num_attention_heads,
                layer.layer_num_kv_heads,
                layer.layer_head_dim,
                params.kv_context_size,
                layer.rope,
                params.staging_slot_idx,
                params.position,
                params.seq_len,
            )?;
            let lt_attn_done = lt_attn.elapsed().as_secs_f64() * 1000.0;
            if let Ok(layer_str) = std::env::var("AEGIS_DUMP_LAYER") {
                if let Ok(target_layer) = layer_str.parse::<usize>() {
                    if layer_idx == target_layer {
                        let tag = std::env::var("AEGIS_DUMP_TAG").unwrap_or_else(|_| "?".into());
                        let h = self.runtime.download_f32(&scratch.residual)?;
                        eprintln!(
                            "[DUMP {tag} L{}] residual first8={:?}",
                            layer_idx,
                            &h[0..8.min(h.len())],
                        );
                    }
                }
            }
            let lt_mlp = std::time::Instant::now();
            super::mlp::forward_mlp_device(
                &self.runtime, layer, layer_idx, self.ple.as_ref(), scratch, params.rms_norm_eps,
            )?;
            let lt_mlp_done = lt_mlp.elapsed().as_secs_f64() * 1000.0;
            std::mem::swap(hidden, &mut scratch.hidden_out);
            if prof {
                eprintln!(
                    "[PROF L{:>2}] attn={:>6.1}ms mlp={:>6.1}ms",
                    layer_idx, lt_attn_done, lt_mlp_done
                );
            }

            // Schedule async D2H of the new KV slot on the transfer stream after
            // the compute stream finishes the attention store + read. This overlaps
            // with the next layer's compute.
            if host_resident {
                let compute_evt = self.runtime.record_compute_event()?;
                self.runtime.transfer_wait_event(&compute_evt)?;
                let slot_idx = staging_slot_idx.unwrap();
                // Re-borrow the pool to schedule the D2H. Disjoint from the
                // current and next H2D since this writes the host KV (a separate
                // memory region) and reads the staging slot we just finished.
                let pool = scratch.kv_staging.as_mut().ok_or_else(|| {
                    AegisError::InvalidPlan("missing kv_staging pool".into())
                })?;
                let slot = &pool.slots[slot_idx];
                let host = layers[layer_idx]
                    .kv
                    .host
                    .as_mut()
                    .ok_or_else(|| AegisError::InvalidPlan("layer host kv missing".into()))?;
                // Take an immutable view into staging since slot is a different
                // borrow than host.
                self.runtime
                    .writeback_kv_slot_async(&mut host.keys, &slot.keys, position, kv_width)?;
                self.runtime
                    .writeback_kv_slot_async(&mut host.values, &slot.values, position, kv_width)?;
            }

            current_h2d_event = next_h2d_event;
            let _ = lt0; // attn/mlp printed above when AEGIS_PROFILE_LAYERS=1
        }
        let _ = layer_times;

        Ok(())
    }

    pub(super) fn forward_hidden(&self, state: &mut CudaLlamaState, token_id: usize) -> Result<()> {
        if state.position >= self.kv_context_size {
            return Err(AegisError::InvalidPlan(format!(
                "kv cache context exhausted: position={} context={}",
                state.position, self.kv_context_size
            )));
        }
        let prof = std::env::var("AEGIS_PROFILE").ok().is_some_and(|v| !v.is_empty());
        let t0 = std::time::Instant::now();
        self.runtime
            .bf16_row_to_f32_device(&self.embed_tokens, token_id, &mut state.hidden)?;
        if let Some(scale) = self.embed_scale {
            self.runtime.scale_f32_device(scale, &mut state.hidden)?;
        }
        if let Ok(tag) = std::env::var("AEGIS_DUMP_EMBED") {
            let h = self.runtime.download_f32(&state.hidden)?;
            eprintln!("[DUMP {tag} EMBED tok={}] first8={:?}", token_id, &h[0..8.min(h.len())]);
        }
        // PLE token-entry compute (Gemma-4 E4B/E2B): combines the
        // `embed_tokens_per_layer` lookup with the `per_layer_model_projection`
        // of `hidden`, then RMSNorm-combined into `state.scratch.per_layer_inputs`.
        // No-op for non-PLE models. (This `forward_hidden` is the prefill/
        // hidden-only path; decode uses `forward_next_token` below which has
        // its own PLE call site.)
        if let Some(ple) = &self.ple {
            crate::executor::ple::compute_per_layer_inputs_decode(
                &self.runtime, ple, token_id, &state.hidden, self.layers.len(),
                &mut state.scratch, self.rms_norm_eps,
            )?;
        }

        let seq_len = state.position + 1;
        self.runtime.copy_u32_to_device(
            &[state.position as u32],
            &mut state.decode_position,
        )?;
        self.runtime.copy_u32_to_device(
            &[seq_len as u32],
            &mut state.decode_seq_len,
        )?;

        let t1 = std::time::Instant::now();
        self.forward_layers_device(state, state.position)?;
        let t2 = std::time::Instant::now();
        if prof {
            // Force a sync to attribute time to where it was actually spent on GPU.
            self.runtime.synchronize()?;
            let t3 = std::time::Instant::now();
            eprintln!(
                "[PROF] embed+setup={:>6.1}ms layers_dispatch={:>6.1}ms gpu_work={:>6.1}ms total={:>6.1}ms",
                (t1 - t0).as_secs_f64() * 1000.0,
                (t2 - t1).as_secs_f64() * 1000.0,
                (t3 - t2).as_secs_f64() * 1000.0,
                (t3 - t0).as_secs_f64() * 1000.0,
            );
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
        let mut logits = self.runtime.download_f32(&state.logits)?;
        if let Some(cap) = self.lm_head_softcap {
            aegisllm_base::executor::generation::apply_logit_softcap(&mut logits, cap);
        }
        Ok(logits)
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
        let non_greedy = sampling.temperature > 0.0 && sampling.top_k != 1;
        if non_greedy || self.lm_head_softcap.is_some() {
            let mut logits = self.runtime.download_f32(&state.logits)?;
            if let Some(cap) = self.lm_head_softcap {
                aegisllm_base::executor::generation::apply_logit_softcap(&mut logits, cap);
            }
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
        let non_greedy = sampling.temperature > 0.0 && sampling.top_k != 1;
        // Diagnostic: split decode time into "CPU issuing kernels" vs "CPU
        // waiting for GPU after issuing". Toggle with `AEGIS_DECODE_TIMING=1`.
        // This pins down whether decode tps is gated by Rust/cudarc launch
        // overhead (T_cpu_issuing dominates) or by actual GPU compute
        // (T_gpu_wait dominates).
        let dec_timing =
            std::env::var("AEGIS_DECODE_TIMING").ok().is_some_and(|v| !v.is_empty());
        let t_step_start = if dec_timing { Some(std::time::Instant::now()) } else { None };
        // Real per-token H2D streaming volume: snapshot the staging-pool byte
        // counter at step entry; the delta at the end is exactly what this token
        // streamed (NVFP4 experts, BF16-streamed layers, etc.).
        let h2d_bytes_start = if dec_timing {
            Some(crate::cuda::staging::STAGING_H2D_BYTES.load(std::sync::atomic::Ordering::Relaxed))
        } else {
            None
        };

        if state.position >= self.kv_context_size {
            return Err(AegisError::InvalidPlan(format!(
                "kv cache context exhausted: position={} context={}",
                state.position, self.kv_context_size
            )));
        }

        self.runtime
            .bf16_row_to_f32_device(&self.embed_tokens, token_id, &mut state.hidden)?;
        // Gemma 4 ScaledWordEmbedding: multiply embeddings by sqrt(hidden_size) so that
        // token embedding magnitudes match downstream RMS-norm hidden states.
        if let Some(scale) = self.embed_scale {
            self.runtime.scale_f32_device(scale, &mut state.hidden)?;
        }
        // PLE (Per-Layer Embeddings) token-entry compute. Must run OUTSIDE
        // the CUDA Graph capture/replay window because it stages a host
        // BF16 row through cuMemcpyHtoD per token (the embed_tokens_per_layer
        // table is 5.4 GiB host-resident — can't fit in VRAM). The per-layer
        // additive contribution inside each decoder block runs from
        // `apply_ple_contribution_decode` (mlp.rs).
        if let Some(ple) = &self.ple {
            crate::executor::ple::compute_per_layer_inputs_decode(
                &self.runtime, ple, token_id, &state.hidden, self.layers.len(),
                &mut state.scratch, self.rms_norm_eps,
            )?;
        }

        let seq_len = state.position + 1;
        // Update dynamic decode params BEFORE capture/replay (outside the captured graph).
        self.runtime.copy_u32_to_device(
            &[state.position as u32],
            &mut state.decode_position,
        )?;
        self.runtime.copy_u32_to_device(
            &[seq_len as u32],
            &mut state.decode_seq_len,
        )?;

        // The decode graph captures the split-decode kernel's shared-mem
        // allocation, which is sized for chunk_len ≤ DECODE_MAX_CHUNK_LEN
        // (= CUDA_GRAPH_ATTN_MAX_SEQ_LEN / DECODE_SPLIT_K). Replaying the
        // captured graph at seq_len > CUDA_GRAPH_ATTN_MAX_SEQ_LEN would
        // overflow `scores[chunk_len]` past the captured shared
        // allocation. Fall back to the eager path (which sizes shared mem
        // from the live seq_len) when we exceed the captured envelope.
        let can_replay = seq_len <= CUDA_GRAPH_ATTN_MAX_SEQ_LEN;
        if let (true, Some(ref graph)) = (can_replay, state.decode_graph.as_ref()) {
            // Hot path: replay the previously captured graph (32 layers + norm + lm_head + argmax).
            // For non-greedy we'll download logits from state.logits; the argmax is a no-op for us.
            self.runtime.replay_decode_graph(&graph.0)?;
        } else {
            let can_capture = seq_len <= CUDA_GRAPH_ATTN_MAX_SEQ_LEN
                && !self.has_staged_layers
                && !self.has_staged_kv
                && state.decode_graph.is_none();
            if can_capture {
                self.runtime.begin_decode_graph_capture()?;
            }

            let position = state.position;
            self.forward_layers_device(state, position)?;

            // Final norm + lm_head + argmax (graph-compatible: no dynamic scalar params).
            // Argmax is always captured in the graph even for non-greedy — we'll just ignore
            // the sampled_token result and download state.logits directly if non_greedy.
            {
                let CudaLlamaState {
                    ref hidden,
                    ref mut scratch,
                    ref mut logits,
                    ref mut sampled_token,
                    ..
                } = *state;
                self.runtime.rms_norm_device(
                    hidden,
                    &self.final_norm,
                    self.rms_norm_eps,
                    &mut scratch.final_hidden,
                )?;
                self.runtime.matvec_bf16_reference_device(
                    &self.lm_head,
                    &scratch.final_hidden,
                    logits,
                )?;
                self.runtime.argmax_f32_device(
                    logits,
                    &mut scratch.argmax_block_values,
                    &mut scratch.argmax_block_indices,
                    sampled_token,
                )?;
            }

            if can_capture
                && let Some(graph) = self.runtime.end_decode_graph_capture()?
            {
                // The capture recorded kernels but did NOT execute them.
                // Launch the graph now so that results are correct for this step.
                self.runtime.replay_decode_graph(&graph)?;
                state.decode_graph = Some(SendCudaGraph(graph));
            }
        }

        state.position += 1;

        // CPU has finished issuing all launches; the next download forces a
        // stream sync and any time it spends waiting is GPU-side.
        let t_cpu_done = t_step_start.map(|_| std::time::Instant::now());

        if non_greedy || self.lm_head_softcap.is_some() {
            // Download all logits and sample on CPU (also needed for soft-cap).
            let mut logits = self.runtime.download_f32(&state.logits)?;
            if let Some(cap) = self.lm_head_softcap {
                aegisllm_base::executor::generation::apply_logit_softcap(&mut logits, cap);
            }
            if let (Some(t0), Some(t_cpu)) = (t_step_start, t_cpu_done) {
                let h2d = crate::cuda::staging::STAGING_H2D_BYTES
                    .load(std::sync::atomic::Ordering::Relaxed)
                    .wrapping_sub(h2d_bytes_start.unwrap_or(0));
                report_decode_split(t0, t_cpu, h2d);
            }
            return aegisllm_base::executor::generation::sample_next_token(&logits, sampling);
        }

        // Greedy (no soft-cap): download the argmax result (also synchronizes the stream).
        let token = self.runtime.download_u32(&state.sampled_token)?;
        if let (Some(t0), Some(t_cpu)) = (t_step_start, t_cpu_done) {
            let h2d = crate::cuda::staging::STAGING_H2D_BYTES
                .load(std::sync::atomic::Ordering::Relaxed)
                .wrapping_sub(h2d_bytes_start.unwrap_or(0));
            report_decode_split(t0, t_cpu, h2d);
        }
        token
            .first()
            .copied()
            .map(|token| token as usize)
            .ok_or_else(|| AegisError::InvalidPlan("CUDA decode argmax returned no token".into()))
    }
}

/// Print the CPU-issuing vs GPU-waiting split for one decode token.
/// `t0` = function entry; `t_cpu_done` = right after all kernels are issued
/// (state.position bumped, BEFORE any sync). The current time minus
/// `t_cpu_done` is the time the CPU spent blocked waiting for the GPU to
/// finish (download_* synchronously drains the stream), so it's a lower
/// bound on extra GPU work past the CPU-issuing window.
fn report_decode_split(t0: std::time::Instant, t_cpu_done: std::time::Instant, h2d_bytes: u64) {
    let total = t_cpu_done.elapsed() + (t_cpu_done - t0);
    let cpu_issuing_ms = (t_cpu_done - t0).as_secs_f64() * 1000.0;
    let gpu_wait_ms = t_cpu_done.elapsed().as_secs_f64() * 1000.0;
    let total_ms = total.as_secs_f64() * 1000.0;
    let pct = |x: f64| -> f64 { if total_ms > 0.0 { x / total_ms * 100.0 } else { 0.0 } };
    // H2D streamed this token + the sustained rate over the whole step (lower
    // bound on achieved PCIe bandwidth; if decode is transfer-bound this ≈ the
    // real link rate). MiB and GB/s (1e9) so it compares directly to the
    // ~55 GB/s PCIe-5.0-x16 ceiling.
    let mib = h2d_bytes as f64 / (1024.0 * 1024.0);
    let gbps = if total_ms > 0.0 { h2d_bytes as f64 / (total_ms / 1000.0) / 1e9 } else { 0.0 };
    eprintln!(
        "[DECODE-TIMING] total={:>5.2}ms  cpu_issuing={:>5.2}ms ({:>4.1}%)  gpu_wait={:>5.2}ms ({:>4.1}%)  h2d={:>7.1} MiB ({:>5.1} GB/s)",
        total_ms,
        cpu_issuing_ms,
        pct(cpu_issuing_ms),
        gpu_wait_ms,
        pct(gpu_wait_ms),
        mib,
        gbps,
    );
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

    /// Hybrid Gemma-4 DENSE per-layer forward (CPU+GPU heterogeneous path).
    ///
    /// Runs ONE Gemma-4 dense decoder layer on the GPU, reusing the EXACT
    /// full-model per-layer math (`forward_attention_device` + the full
    /// `forward_mlp_device` WITH PLE) — no duplicated kernels. The shared PLE
    /// feed `per_layer_inputs` (computed once per token on the CPU) is uploaded
    /// to `state.scratch.per_layer_inputs` so this layer's on-device PLE additive
    /// is bit-identical to the all-GPU path; `layer_scalar` is applied on-device
    /// (inside `forward_mlp_device`), BEFORE the PLE additive — wait: PLE additive
    /// runs BEFORE layer_scalar (see mlp.rs), matching HF `Gemma4DecoderLayer`.
    ///
    /// `per_layer_inputs` is `[total_num_layers * ple_dim]` (empty/ignored for
    /// non-PLE models). KV-share (`layer.kv_shared_from`) reads the PARENT
    /// layer's KV cache, which the hybrid guarantees is co-located on this same
    /// device (the parent is GPU-scheduled and present in `state.layers`).
    ///
    /// Mirrors the per-layer body of `forward_layers_device` but for a single
    /// layer via the block-executor state.
    #[allow(dead_code)]
    pub fn forward_g4_layer_host(
        &self,
        state: &mut CudaLayerBlockState,
        layer_idx: usize,
        position: usize,
        hidden: &[f32],
        per_layer_inputs: &[f32],
    ) -> Result<Vec<f32>> {
        if hidden.len() != self.hidden_size {
            return Err(AegisError::InvalidPlan(format!(
                "hybrid CUDA G4 layer input mismatch: expected {}, got {}",
                self.hidden_size,
                hidden.len()
            )));
        }
        let layer = self.layers.get(&layer_idx).ok_or_else(|| {
            AegisError::InvalidPlan(format!("missing CUDA hybrid G4 layer `{layer_idx}`"))
        })?;
        if layer.moe.is_some() {
            return Err(AegisError::Unsupported(format!(
                "hybrid Gemma-4 layer `{layer_idx}` is MoE; the per-layer CPU+GPU hybrid \
                 supports DENSE Gemma-4 (E2B/E4B) only"
            )));
        }
        // Upload the token's hidden + the shared PLE feed for THIS step. The PLE
        // feed is the same vector the CPU computed at token entry, so on-device
        // PLE matches the CPU-side PLE additive exactly.
        state.hidden = self.runtime.upload_f32(hidden)?;
        if self.ple.is_some() && !per_layer_inputs.is_empty() {
            self.runtime
                .upload_f32_slice_to_device(per_layer_inputs, &mut state.scratch.per_layer_inputs)?;
        }
        self.runtime
            .copy_u32_to_device(&[position as u32], &mut state.p_position)?;
        self.runtime
            .copy_u32_to_device(&[(position + 1) as u32], &mut state.p_seq_len)?;

        // Borrow the layer's mutable KV state and (for shared layers) the
        // parent's immutable KV cache from the same BTreeMap. The two indices
        // are distinct (parent < layer_idx by construction), so the regions do
        // not alias; raw pointers express this to the borrow checker (same
        // pattern as the full-model split-borrow). The parent's KV is on THIS
        // device — guaranteed by the hybrid's co-location validation.
        let kv_shared_parent = layer.kv_shared_from;
        let layers_ptr: *mut std::collections::BTreeMap<usize, CudaLayerState> = &mut state.layers;
        let parent_kv: Option<*const super::state::CudaKvCache> = match kv_shared_parent {
            Some(parent_idx) => {
                let parent = unsafe { &*layers_ptr }.get(&parent_idx).ok_or_else(|| {
                    AegisError::InvalidPlan(format!(
                        "hybrid G4 KV-share: parent layer `{parent_idx}` of layer `{layer_idx}` \
                         is not on this CUDA device (co-locate the parent and child)"
                    ))
                })?;
                Some(&parent.kv as *const _)
            }
            None => None,
        };
        let layer_state = unsafe { &mut *layers_ptr }.get_mut(&layer_idx).ok_or_else(|| {
            AegisError::InvalidPlan(format!("missing CUDA hybrid G4 layer state `{layer_idx}`"))
        })?;
        // SAFETY: `parent_kv` points at a DIFFERENT map entry (parent_idx !=
        // layer_idx), so it does not alias `layer_state`. The reference lives
        // only for this call. forward_attention_device reads it immutably.
        let kv_shared_override = parent_kv.map(|p| unsafe { &*p });

        super::attention::forward_attention_device(
            &self.runtime,
            layer,
            layer_state,
            kv_shared_override,
            &state.hidden,
            &mut state.scratch,
            &state.p_position,
            &state.p_seq_len,
            self.rms_norm_eps,
            self.num_attention_heads,
            layer.layer_num_kv_heads,
            layer.layer_head_dim,
            self.kv_context_size,
            layer.rope,
            None,
            position,
            position + 1,
        )?;
        // Full Gemma-4 MLP WITH PLE additive + layer_scalar (on-device), exactly
        // as the all-GPU path runs it. `self.ple.as_ref()` is the loaded global
        // apparatus; `forward_mlp_device` reads `scratch.per_layer_inputs` (just
        // uploaded) for this layer's additive contribution.
        super::mlp::forward_mlp_device(
            &self.runtime,
            layer,
            layer_idx,
            self.ple.as_ref(),
            &mut state.scratch,
            self.rms_norm_eps,
        )?;
        std::mem::swap(&mut state.hidden, &mut state.scratch.hidden_out);
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
        // Reuse pooled `p_position` / `p_seq_len` device buffers
        // instead of `alloc_u32(1)` × 2 fresh allocations per layer
        // per token (each round-trips through cudaMallocAsync).
        self.runtime.copy_u32_to_device(&[position as u32], &mut state.p_position)?;
        self.runtime.copy_u32_to_device(&[(position + 1) as u32], &mut state.p_seq_len)?;
        let layer_state = state.layers.get_mut(&layer_idx).ok_or_else(|| {
            AegisError::InvalidPlan(format!("missing CUDA hybrid layer state `{layer_idx}`"))
        })?;
        forward_cuda_layer_device(
            &self.runtime,
            layer,
            layer_state,
            &mut state.hidden,
            &mut state.scratch,
            &state.p_position,
            &state.p_seq_len,
            CudaLayerForwardParams {
                rms_norm_eps: self.rms_norm_eps,
                num_attention_heads: self.num_attention_heads,
                num_kv_heads: self.num_kv_heads,
                head_dim: self.head_dim,
                kv_context_size: self.kv_context_size,
                position,
                seq_len: position + 1,
                staging_slot_idx: None,
            },
        )?;
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn forward_cuda_layer_device(
    runtime: &CudaRuntime,
    layer: &CudaLayer,
    layer_state: &mut CudaLayerState,
    hidden: &mut DeviceBuffer<f32>,
    scratch: &mut CudaScratch,
    p_position: &DeviceBuffer<u32>,
    p_seq_len: &DeviceBuffer<u32>,
    params: CudaLayerForwardParams,
) -> Result<()> {
    // Per-layer head_dim and num_kv_heads override model-wide params
    // (e.g. Gemma 4 global layers use head_dim=512, num_kv_heads=2).
    // Block-level executor: no KV-share (single-layer unit tests).
    forward_attention_device(
        runtime,
        layer,
        layer_state,
        None,
        hidden,
        scratch,
        p_position,
        p_seq_len,
        params.rms_norm_eps,
        params.num_attention_heads,
        layer.layer_num_kv_heads,
        layer.layer_head_dim,
        params.kv_context_size,
        layer.rope,
        params.staging_slot_idx,
        params.position,
        params.seq_len,
    )?;
    if let Ok(layer_str) = std::env::var("AEGIS_DUMP_LAYER") {
        if let Ok(target_layer) = layer_str.parse::<usize>() {
            let tag = std::env::var("AEGIS_DUMP_TAG").unwrap_or_else(|_| "?".into());
            thread_local! {
                static CALL_COUNT: std::cell::RefCell<usize> = std::cell::RefCell::new(0);
            }
            CALL_COUNT.with(|c| {
                let mut c = c.borrow_mut();
                if *c == target_layer {
                    let h = runtime.download_f32(&scratch.residual).unwrap();
                    eprintln!(
                        "[DUMP {tag} L{}] residual first8={:?}",
                        *c,
                        &h[0..8.min(h.len())],
                    );
                }
                *c += 1;
            });
        }
    }
    // Block-level path (unit tests). PLE is wired only on the full-model
    // executor (`CudaLlamaExecutor::forward_hidden`); this path always
    // passes None and layer_idx=0.
    forward_mlp_device(runtime, layer, 0, None, scratch, params.rms_norm_eps)?;
    std::mem::swap(hidden, &mut scratch.hidden_out);
    Ok(())
}

/// Issues asynchronous H2D upload of `position` existing KV entries for a single
/// layer onto the dedicated transfer stream. Returns the recorded transfer event
/// (the compute stream must wait on this before reading the staging slot).
/// Returns `None` if the layer is not host-resident (no upload needed).
fn issue_h2d_for_layer(
    runtime: &CudaRuntime,
    scratch: &mut CudaScratch,
    layers: &mut [CudaLayerState],
    layer_idx: usize,
    position: usize,
    kv_width: usize,
) -> Result<Option<cudarc::driver::CudaEvent>> {
    if !layers[layer_idx].kv.is_host_resident() {
        return Ok(None);
    }
    let pool = scratch.kv_staging.as_mut().ok_or_else(|| {
        AegisError::InvalidPlan("missing kv_staging pool for host-resident layer".into())
    })?;
    let slot_idx = layer_idx % 2;
    let host = layers[layer_idx].kv.host.as_ref().ok_or_else(|| {
        AegisError::InvalidPlan(format!("layer {layer_idx} marked host-resident but missing host kv"))
    })?;
    let slot = &mut pool.slots[slot_idx];
    runtime.upload_kv_slice_async(&mut slot.keys, &host.keys, position * kv_width)?;
    runtime.upload_kv_slice_async(&mut slot.values, &host.values, position * kv_width)?;
    Ok(Some(runtime.record_transfer_event()?))
}

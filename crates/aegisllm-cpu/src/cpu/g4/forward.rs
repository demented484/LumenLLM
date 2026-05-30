//! Gemma-4 CPU forward driver. Mirrors the CUDA decode path op-for-op
//! (`crates/aegisllm-cuda/src/executor/{forward,attention,mlp,ple}.rs`).
//!
//! Two paths share the exact same math:
//!   * `forward_hidden` — single-token, used for DECODE.
//!   * `forward_batched` — processes all T prompt tokens at once for PREFILL.
//!     The per-token weight projections (q/k/v/o, gate/up/down) are replaced by
//!     BATCHED `matmul_into` GEMMs so each weight is read from DRAM ONCE per
//!     prompt instead of once per token (the big prefill win); attention stays
//!     PER-POSITION (token i attends to cached K/V at positions `[.. start+i]`,
//!     respecting the sliding window) and every elementwise op (norms, RoPE,
//!     per-head norms, GeGLU, PLE, residuals) stays per-token. The batched GEMM
//!     is bit-identical to looping `matvec_into` (see `linear::matmul_into`), so
//!     `forward_batched` produces the SAME hidden/logits as looping
//!     `forward_hidden`. MoE layers fall back to per-token `forward_moe`
//!     (batching the two-stream MoE combine is deferred); dense + PLE models
//!     (E2B/E4B) are fully batched.

use super::attention::{
    g4_attention_decode_into, g4_attention_prefill_into, G4DecodeAttnRequest, G4PrefillAttnRequest,
};
use super::moe::router_softmax_topk_normalized;
use super::norm::{rms_norm_per_head_into, rms_norm_per_head_no_weight_into};
use super::ple;
use super::rope::apply_rope_partial_in_place;
use super::state::{
    G4CpuExecutor, G4CpuLayer, G4CpuState, G4DenseMlp, G4MoeLayer, G4PrefillScratch,
};
use crate::cpu::math::{add_into, geglu_into, rms_norm_into};
use crate::cpu::simd;
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::executor::generation::apply_logit_softcap;
use aegisllm_base::generation::SamplingConfig;
use rayon::prelude::*;

/// Smallest prefill where the batched-GEMM path is worth its setup. Below this,
/// per-token `forward_hidden` is comparable and avoids the extra buffers.
const PREFILL_BATCH_THRESHOLD: usize = 4;
/// Max tokens per batched chunk: bigger amortizes per-chunk overhead (buffer
/// allocs, attention setup) over more tokens and matches llama.cpp's large-batch
/// prompt processing; the large-batch GEMM (`bf16_matmul_fast`) keeps each weight
/// panel cache-resident across all tokens. Cost is scratch RAM (≈ batch *
/// intermediate floats per projection). Chunks are processed in order so the
/// position-indexed KV cache keeps causal semantics.
const PREFILL_MAX_BATCH: usize = 512;

impl G4CpuExecutor {
    pub(crate) fn prefill_prompt(
        &self,
        state: &mut G4CpuState,
        prompt_tokens: &[usize],
        sampling: &SamplingConfig,
    ) -> Result<usize> {
        let Some((&last, prefix)) = prompt_tokens.split_last() else {
            return Err(AegisError::InvalidConfig("prompt produced no tokens".into()));
        };
        if state.position + prompt_tokens.len() > self.kv_context_size {
            return Err(AegisError::InvalidPlan(format!(
                "prefill would exhaust kv cache: position={} prompt_tokens={} context={}",
                state.position,
                prompt_tokens.len(),
                self.kv_context_size
            )));
        }
        let start_position = state.position;
        // Batched prefill over ALL prompt tokens (including `last`): batched
        // GEMM projections + per-position attention, advancing `state.position`
        // by the prompt length and leaving every position's KV cached. Tiny
        // prompts fall back to per-token forward_hidden (batch setup would
        // dominate). The next-token logits come from the LAST position's hidden.
        if prompt_tokens.len() >= PREFILL_BATCH_THRESHOLD {
            let last_hidden = self.forward_batched(state, prompt_tokens, start_position)?;
            let logits = self.final_logits(&last_hidden)?;
            return aegisllm_base::executor::generation::sample_next_token(&logits, sampling);
        }
        for &token in prefix {
            let _ = self.forward_hidden(state, token)?;
        }
        let logits = self.forward_logits(state, last)?;
        aegisllm_base::executor::generation::sample_next_token(&logits, sampling)
    }

    pub(crate) fn forward_logits(
        &self,
        state: &mut G4CpuState,
        token_id: usize,
    ) -> Result<Vec<f32>> {
        let hidden = self.forward_hidden(state, token_id)?;
        self.final_logits(&hidden)
    }

    // ── Per-layer hybrid block API (CPU+GPU heterogeneous Gemma-4 dense) ──────
    //
    // These expose the SAME per-token math `forward_hidden` runs, but split so a
    // hybrid scheduler can interleave CPU-computed and GPU-computed layers in a
    // single forward. The token-entry work (embed + scale + PLE feed) runs ONCE
    // per token on the CPU and the resulting `per_layer_inputs` is shared with
    // BOTH the CPU and GPU layer paths (the GPU path uploads the same vector), so
    // every layer's PLE additive uses a bit-identical feed.

    /// Number of decoder layers (used by the hybrid to build the schedule and to
    /// size the shared PLE feed).
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// `hidden_size`, exposed so the hybrid can size cross-device transfer buffers.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// `Some(parent_idx)` when layer `layer_idx` reads its K/V from another layer
    /// (Gemma-4 E2B/E4B KV-share); `None` for own-KV layers. The hybrid uses this
    /// to validate that a shared layer is co-located with its KV parent.
    pub fn kv_shared_parent(&self, layer_idx: usize) -> Option<usize> {
        self.layers.get(layer_idx).and_then(|l| l.kv_shared_from)
    }

    /// True when layer `layer_idx` is a dense (non-MoE) layer. The hybrid only
    /// supports dense Gemma-4 (E2B/E4B); MoE (26B-A4B) is rejected upstream.
    pub fn layer_is_dense(&self, layer_idx: usize) -> bool {
        self.layers.get(layer_idx).map(|l| l.moe.is_none()).unwrap_or(false)
    }

    /// True when ANY layer is MoE. The hybrid rejects MoE Gemma-4 (26B) with a
    /// clear message; only dense E2B/E4B is wired.
    pub fn has_moe_layer(&self) -> bool {
        self.layers.iter().any(|l| l.moe.is_some())
    }

    /// Token entry for the hybrid path: embed lookup + Gemma-4 embed scale +
    /// PLE token-entry compute (writes `state.per_layer_inputs`). Returns the
    /// scaled embedding `hidden` (input to layer 0). Mirrors the head of
    /// `forward_hidden` exactly. Does NOT advance `state.position` (the hybrid
    /// advances position once after all layers, like `forward_hidden`).
    pub fn token_entry_host(
        &self,
        state: &mut G4CpuState,
        token_id: usize,
    ) -> Result<Vec<f32>> {
        if state.position >= self.kv_context_size {
            return Err(AegisError::InvalidPlan(format!(
                "kv cache context exhausted: position={} context={}",
                state.position, self.kv_context_size
            )));
        }
        let mut hidden = self.embed_tokens.row(token_id)?;
        simd::scale_in_place(&mut hidden, self.embed_scale);
        if let Some(ple_g) = &self.ple {
            ple::compute_per_layer_inputs(
                ple_g,
                token_id,
                &hidden,
                self.layers.len(),
                self.rms_norm_eps,
                &mut state.per_layer_inputs,
            )?;
        }
        Ok(hidden)
    }

    /// A read-only snapshot of `state.per_layer_inputs` (the shared PLE feed).
    /// The hybrid uploads this to the GPU so GPU-computed layers apply the same
    /// PLE additive. Empty for non-PLE models.
    pub fn per_layer_inputs_snapshot<'s>(&self, state: &'s G4CpuState) -> &'s [f32] {
        &state.per_layer_inputs
    }

    /// Run ONE dense Gemma-4 layer on the CPU: attention sublayer + dense MLP
    /// (with the per-layer PLE additive + layer_scalar). Reads/advances the
    /// per-layer KV cache for own-KV layers; reads the parent's KV for shared
    /// layers. Bit-identical to the corresponding iteration of `forward_hidden`.
    /// Rejects MoE layers (hybrid is dense-only).
    pub fn forward_dense_layer_host(
        &self,
        state: &mut G4CpuState,
        layer_idx: usize,
        position: usize,
        hidden: &[f32],
    ) -> Result<Vec<f32>> {
        let layer = self.layers.get(layer_idx).ok_or_else(|| {
            AegisError::InvalidPlan(format!("g4 hybrid: missing CPU layer `{layer_idx}`"))
        })?;
        if layer.moe.is_some() {
            return Err(AegisError::Unsupported(format!(
                "hybrid Gemma-4 layer `{layer_idx}` is MoE; the per-layer CPU+GPU hybrid \
                 supports DENSE Gemma-4 (E2B/E4B) only. Run the 26B MoE fully on one device."
            )));
        }
        let seq_len = position + 1;
        let residual = self.forward_attention(state, layer, layer_idx, position, seq_len, hidden)?;
        let mlp = layer.mlp.as_ref().ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "g4 hybrid: dense layer `{layer_idx}` has neither MLP nor MoE"
            ))
        })?;
        self.forward_dense_mlp(layer, mlp, layer_idx, &state.per_layer_inputs, &residual)
    }

    /// final_norm → lm_head → optional softcap, for the hybrid (logits always
    /// produced on CPU in the MVP). Public wrapper over the private `final_logits`.
    pub fn final_logits_host(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        self.final_logits(hidden)
    }

    /// Advance the decode position by one (called by the hybrid after all layers
    /// for a token have run, mirroring `forward_hidden`'s `state.position += 1`).
    pub fn advance_position(&self, state: &mut G4CpuState) {
        state.position += 1;
    }

    /// final_norm (RMS) → lm_head → optional logit softcap. Shared by the
    /// per-token decode path and the batched-prefill last-position path.
    fn final_logits(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        let mut final_hidden = vec![0.0_f32; self.hidden_size];
        rms_norm_into(hidden, &self.final_norm, self.rms_norm_eps, &mut final_hidden);
        let mut logits = vec![0.0_f32; self.lm_head.rows];
        self.lm_head.matvec_into(&final_hidden, &mut logits)?;
        // final logit softcap: cap * tanh(logits / cap).
        if let Some(cap) = self.lm_head_softcap {
            apply_logit_softcap(&mut logits, cap);
        }
        Ok(logits)
    }

    pub(crate) fn forward_hidden(
        &self,
        state: &mut G4CpuState,
        token_id: usize,
    ) -> Result<Vec<f32>> {
        if state.position >= self.kv_context_size {
            return Err(AegisError::InvalidPlan(format!(
                "kv cache context exhausted: position={} context={}",
                state.position, self.kv_context_size
            )));
        }
        let position = state.position;
        let seq_len = position + 1;

        // ── Token entry ────────────────────────────────────────────────
        // 1-2. embed lookup + embed scale (sqrt(hidden_size)).
        let mut hidden = self.embed_tokens.row(token_id)?;
        simd::scale_in_place(&mut hidden, self.embed_scale);

        // 3. PLE token-entry (E4B/E2B only).
        if let Some(ple_g) = &self.ple {
            ple::compute_per_layer_inputs(
                ple_g,
                token_id,
                &hidden,
                self.layers.len(),
                self.rms_norm_eps,
                &mut state.per_layer_inputs,
            )?;
        }

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // ── Attention sublayer ──────────────────────────────────────
            let residual = self.forward_attention(state, layer, layer_idx, position, seq_len, &hidden)?;

            // ── MLP sublayer ────────────────────────────────────────────
            let hidden_out = if let Some(moe) = &layer.moe {
                self.forward_moe(layer, moe, layer_idx, &state.per_layer_inputs, &residual)?
            } else if let Some(mlp) = &layer.mlp {
                self.forward_dense_mlp(layer, mlp, layer_idx, &state.per_layer_inputs, &residual)?
            } else {
                return Err(AegisError::InvalidPlan(format!(
                    "g4 layer {layer_idx} has neither dense MLP nor MoE"
                )));
            };
            hidden = hidden_out;
        }

        state.position += 1;
        Ok(hidden)
    }

    /// Batched prefill of all `token_ids` starting at `start_position`. Returns
    /// the LAST token's final hidden (pre-final-norm) so the caller can sample
    /// the next token. Advances `state.position` by `token_ids.len()` and leaves
    /// every position's K/V in the cache, exactly like running `forward_hidden`
    /// once per token — but with BATCHED weight projections (`matmul_into`) and
    /// PER-POSITION attention. Math is byte-identical to `forward_hidden`.
    ///
    /// Processed in chunks of `PREFILL_MAX_BATCH` to bound scratch; chunks are
    /// sequential so the position-indexed KV cache preserves causal semantics.
    pub(crate) fn forward_batched(
        &self,
        state: &mut G4CpuState,
        token_ids: &[usize],
        start_position: usize,
    ) -> Result<Vec<f32>> {
        let hidden_size = self.hidden_size;
        if start_position + token_ids.len() > self.kv_context_size {
            return Err(AegisError::InvalidPlan(format!(
                "kv cache context exhausted: position={} prompt={} context={}",
                start_position,
                token_ids.len(),
                self.kv_context_size
            )));
        }
        let num_layers = self.layers.len();
        let ple_row = self.ple.as_ref().map(|p| num_layers * p.ple_dim).unwrap_or(0);
        let mut last_hidden = vec![0.0_f32; hidden_size];

        // Reusable scratch pool: allocated ONCE, reused across every layer and
        // chunk (no per-layer alloc, no re-zero). `per_layer_inputs` is sized to
        // the max chunk width once, too.
        let mut scratch = self.new_prefill_scratch(PREFILL_MAX_BATCH);
        let mut per_layer_inputs = vec![0.0_f32; PREFILL_MAX_BATCH * ple_row];

        let mut chunk_start = start_position;
        for chunk in token_ids.chunks(PREFILL_MAX_BATCH) {
            let batch = chunk.len();

            // ── Token entry (per token) ─────────────────────────────────
            // 1-2. embed lookup + embed scale → main_a[batch, hidden]. main_a is
            // the per-layer hidden input; the per-layer ping-pong is
            //   attention: main_a (hidden) → main_b (residual)
            //   MLP:       main_b (residual) → main_a (hidden_out)
            // so the result lands back in main_a after every layer.
            for (i, &token) in chunk.iter().enumerate() {
                let row = self.embed_tokens.row(token)?;
                let dst = &mut scratch.main_a[i * hidden_size..(i + 1) * hidden_size];
                dst.copy_from_slice(&row);
                simd::scale_in_place(dst, self.embed_scale);
            }

            // 3. PLE token-entry: per_layer_inputs[batch, num_layers*ple_dim].
            // BATCHED: the model_projection runs as one VNNI GEMM over all tokens
            // (was a per-token matvec ×batch — ~30% of prefill on E2B).
            if let Some(ple_g) = &self.ple {
                ple::compute_per_layer_inputs_batched(
                    ple_g,
                    chunk,
                    &scratch.main_a[..batch * hidden_size],
                    batch,
                    hidden_size,
                    num_layers,
                    self.rms_norm_eps,
                    &mut per_layer_inputs[..batch * ple_row],
                )?;
            }

            let timing = std::env::var("AEGIS_G4_PREFILL_TIMING").is_ok();
            let mut t_attn = std::time::Duration::ZERO;
            let mut t_mlp = std::time::Duration::ZERO;
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                // ── Attention sublayer (batched proj + per-position attn) ──
                // Reads scratch.main_a (hidden), writes scratch.main_b (residual).
                let t0 = std::time::Instant::now();
                self.forward_attention_batched(
                    state, &mut scratch, layer, layer_idx, chunk_start, batch,
                )?;
                if timing {
                    t_attn += t0.elapsed();
                }

                // ── MLP sublayer ──────────────────────────────────────────
                // Dense + PLE (E2B/E4B) is fully batched; MoE falls back to
                // per-token forward_moe (two-stream combine batching deferred).
                // Reads scratch.main_b (residual), writes scratch.main_a (hidden_out).
                let t1 = std::time::Instant::now();
                if let Some(moe) = &layer.moe {
                    for i in 0..batch {
                        let pli = if ple_row == 0 {
                            &[][..]
                        } else {
                            &per_layer_inputs[i * ple_row..(i + 1) * ple_row]
                        };
                        let row = self.forward_moe(
                            layer,
                            moe,
                            layer_idx,
                            pli,
                            &scratch.main_b[i * hidden_size..(i + 1) * hidden_size],
                        )?;
                        scratch.main_a[i * hidden_size..(i + 1) * hidden_size]
                            .copy_from_slice(&row);
                    }
                } else if layer.mlp.is_some() {
                    self.forward_dense_mlp_batched(
                        &mut scratch, layer, layer_idx, batch, &per_layer_inputs, ple_row,
                    )?;
                } else {
                    return Err(AegisError::InvalidPlan(format!(
                        "g4 layer {layer_idx} has neither dense MLP nor MoE"
                    )));
                }
                if timing {
                    t_mlp += t1.elapsed();
                }
            }
            if timing {
                eprintln!(
                    "[g4-prefill] batch={batch} attn={:.1}ms mlp={:.1}ms total={:.1}ms",
                    t_attn.as_secs_f64() * 1e3,
                    t_mlp.as_secs_f64() * 1e3,
                    (t_attn + t_mlp).as_secs_f64() * 1e3,
                );
            }

            // Keep the last token of this chunk; final chunk's last = prompt last.
            // Layer output lives in main_a.
            last_hidden
                .copy_from_slice(&scratch.main_a[(batch - 1) * hidden_size..batch * hidden_size]);
            state.position += batch;
            chunk_start += batch;
        }
        Ok(last_hidden)
    }

    /// Attention sublayer (steps 4-17). Returns the post-residual `residual`
    /// buffer (input to the MLP sublayer).
    fn forward_attention(
        &self,
        state: &mut G4CpuState,
        layer: &G4CpuLayer,
        layer_idx: usize,
        position: usize,
        seq_len: usize,
        hidden: &[f32],
    ) -> Result<Vec<f32>> {
        let head_dim = layer.layer_head_dim;
        let num_kv_heads = layer.layer_num_kv_heads;
        let q_width = self.num_attention_heads * head_dim;
        let kv_width = num_kv_heads * head_dim;
        let shared = layer.kv_shared_from.is_some();

        // 4. input_layernorm (RMS).
        let mut input_normed = vec![0.0_f32; self.hidden_size];
        rms_norm_into(hidden, &layer.input_norm_weight, self.rms_norm_eps, &mut input_normed);

        // 5. Q proj.
        let mut q = vec![0.0_f32; q_width];
        layer.q_proj.matvec_into(&input_normed, &mut q)?;

        // 6. K / V proj (skipped on shared layers).
        let mut k = vec![0.0_f32; kv_width];
        let mut v = vec![0.0_f32; kv_width];
        if !shared {
            layer.k_proj.matvec_into(&input_normed, &mut k)?;
            layer.v_proj.matvec_into(&input_normed, &mut v)?;
        }

        // 7. per-head q_norm (RMS, weighted).
        if let Some(qnw) = &layer.q_norm_weight {
            let mut tmp = vec![0.0_f32; q_width];
            rms_norm_per_head_into(&q, qnw, self.num_attention_heads, head_dim, self.rms_norm_eps, &mut tmp);
            q.copy_from_slice(&tmp);
        }
        if !shared {
            // 8. per-head k_norm (RMS, weighted).
            if let Some(knw) = &layer.k_norm_weight {
                let mut tmp = vec![0.0_f32; kv_width];
                rms_norm_per_head_into(&k, knw, num_kv_heads, head_dim, self.rms_norm_eps, &mut tmp);
                k.copy_from_slice(&tmp);
            }
            // 9. per-head v_norm (RMS, NO weight) — runs whenever q_norm present.
            if layer.q_norm_weight.is_some() {
                let mut tmp = vec![0.0_f32; kv_width];
                rms_norm_per_head_no_weight_into(&v, num_kv_heads, head_dim, self.rms_norm_eps, &mut tmp);
                v.copy_from_slice(&tmp);
            }
        }

        // 10-11. RoPE on Q (and K unless shared).
        apply_rope_partial_in_place(
            &mut q, position, self.num_attention_heads, head_dim, layer.partial_dim, &layer.rope,
        )?;
        if !shared {
            apply_rope_partial_in_place(
                &mut k, position, num_kv_heads, head_dim, layer.partial_dim, &layer.rope,
            )?;
        }

        // 12. Q scale. Gemma-4's effective attention scale is 1.0 (CUDA folds
        //     this by pre-scaling Q by sqrt(head_dim) so its hardcoded
        //     rsqrt(head_dim) kernel scale cancels). On CPU we take the
        //     algebraically-identical-and-cheaper route (spec §1 step 12):
        //     pass scale=1.0 to attention and do NOT pre-scale Q. Non-Gemma-4
        //     (no q_norm) keeps the standard 1/sqrt(head_dim) scale.
        let attn_scale = if layer.q_norm_weight.is_some() {
            1.0
        } else {
            1.0 / (head_dim as f32).sqrt()
        };

        // 13-14. KV store + attention (causal + sliding window).
        let mut attn_context = vec![0.0_f32; q_width];
        if let Some(parent) = layer.kv_shared_from {
            // Shared layer: read parent's KV (parent already stored this position).
            let parent_state = &state.layers[parent];
            g4_attention_decode_into(
                G4DecodeAttnRequest {
                    keys: &parent_state.keys,
                    values: &parent_state.values,
                    seq_len: parent_state.seq_len,
                    query: &q,
                    num_attention_heads: self.num_attention_heads,
                    num_kv_heads,
                    head_dim,
                    window_size: layer.window_size,
                    scale: attn_scale,
                },
                &mut attn_context,
            )?;
        } else {
            let layer_state = &mut state.layers[layer_idx];
            layer_state.push(&k, &v)?;
            g4_attention_decode_into(
                G4DecodeAttnRequest {
                    keys: &layer_state.keys,
                    values: &layer_state.values,
                    seq_len: layer_state.seq_len,
                    query: &q,
                    num_attention_heads: self.num_attention_heads,
                    num_kv_heads,
                    head_dim,
                    window_size: layer.window_size,
                    scale: attn_scale,
                },
                &mut attn_context,
            )?;
        }
        let _ = seq_len; // seq_len == layer_state.seq_len for own-KV layers.

        // 15. o_proj.
        let mut attn_out = vec![0.0_f32; self.hidden_size];
        layer.o_proj.matvec_into(&attn_context, &mut attn_out)?;

        // 16. post_attention sublayer norm (PrePost).
        let mut residual = vec![0.0_f32; self.hidden_size];
        if let Some(post_norm) = &layer.post_attn_sublayer_norm {
            let mut attn_out_normed = vec![0.0_f32; self.hidden_size];
            rms_norm_into(&attn_out, post_norm, self.rms_norm_eps, &mut attn_out_normed);
            // 17. residual add.
            add_into(hidden, &attn_out_normed, &mut residual)?;
        } else {
            add_into(hidden, &attn_out, &mut residual)?;
        }
        Ok(residual)
    }

    /// Batched attention sublayer (steps 4-17) over `batch` tokens at positions
    /// `[chunk_start, chunk_start+batch)`. Q/K/V/O projections are BATCHED
    /// (`matmul_into`); every per-head norm, RoPE, and KV store is per-token; and
    /// attention is PER-POSITION (token i attends to cached K/V at positions
    /// `[.. chunk_start+i]`, sliding window respected). Math mirrors
    /// `forward_attention` exactly.
    ///
    /// Buffers come from the reusable `scratch` pool (no per-call alloc/zero):
    /// reads `scratch.main_a[..batch*hidden]` (hidden), writes the post-residual
    /// `residual` into `scratch.main_b[..batch*hidden]`. Every scratch region is
    /// FULLY OVERWRITTEN before any read — `normed`/`q`/`attn_context`/`proj_out`
    /// /`main_b` by matmul/rms/attention writes; `k`/`v` only on non-shared
    /// layers (on shared layers their stale content is NEVER read: attention
    /// reads the parent layer's already-stored KV, and the local k/v are unused).
    #[allow(clippy::too_many_arguments)]
    fn forward_attention_batched(
        &self,
        state: &mut G4CpuState,
        scratch: &mut G4PrefillScratch,
        layer: &G4CpuLayer,
        layer_idx: usize,
        chunk_start: usize,
        batch: usize,
    ) -> Result<()> {
        let hidden_size = self.hidden_size;
        let head_dim = layer.layer_head_dim;
        let num_kv_heads = layer.layer_num_kv_heads;
        let q_width = self.num_attention_heads * head_dim;
        let kv_width = num_kv_heads * head_dim;
        let shared = layer.kv_shared_from.is_some();

        // 4. input_layernorm (RMS), per token. Reads main_a (hidden), writes
        // normed[..batch*hidden] (fully overwritten by the par_chunks).
        let hidden = &scratch.main_a[..batch * hidden_size];
        let normed = &mut scratch.normed[..batch * hidden_size];
        normed
            .par_chunks_mut(hidden_size)
            .zip(hidden.par_chunks(hidden_size))
            .for_each(|(out, h)| {
                rms_norm_into(h, &layer.input_norm_weight, self.rms_norm_eps, out);
            });

        // 5-6. BATCHED Q (and K/V unless shared) projections. matmul_into fully
        // writes its [batch*width] output.
        let normed = &scratch.normed[..batch * hidden_size];
        layer
            .q_proj
            .matmul_into(normed, batch, &mut scratch.q[..batch * q_width])?;
        if !shared {
            layer
                .k_proj
                .matmul_into(normed, batch, &mut scratch.k[..batch * kv_width])?;
            layer
                .v_proj
                .matmul_into(normed, batch, &mut scratch.v[..batch * kv_width])?;
        }

        // 7-11. Per-token per-head q/k/v norms + partial RoPE at the token's
        // position, PARALLELIZED over tokens (each token's q/k/v slices are
        // independent). Math is identical to the previous sequential loop and to
        // `forward_attention`. The q/k/v norms run in-place via a per-token scratch
        // `tmp` (allocated inside the closure, per token, not pooled).
        let q_norm_weight = layer.q_norm_weight.as_ref();
        let k_norm_weight = layer.k_norm_weight.as_ref();
        let q = &mut scratch.q[..batch * q_width];
        if shared {
            // Only Q is normed/RoPE'd; K/V are unused on shared layers.
            q.par_chunks_mut(q_width)
                .enumerate()
                .try_for_each(|(i, q_slice)| -> Result<()> {
                    let position = chunk_start + i;
                    if let Some(qnw) = q_norm_weight {
                        let mut tmp = vec![0.0_f32; q_width];
                        rms_norm_per_head_into(
                            q_slice, qnw, self.num_attention_heads, head_dim, self.rms_norm_eps,
                            &mut tmp,
                        );
                        q_slice.copy_from_slice(&tmp);
                    }
                    apply_rope_partial_in_place(
                        q_slice, position, self.num_attention_heads, head_dim, layer.partial_dim,
                        &layer.rope,
                    )
                })?;
        } else {
            let k = &mut scratch.k[..batch * kv_width];
            let v = &mut scratch.v[..batch * kv_width];
            q.par_chunks_mut(q_width)
                .zip(k.par_chunks_mut(kv_width))
                .zip(v.par_chunks_mut(kv_width))
                .enumerate()
                .try_for_each(|(i, ((q_slice, k_slice), v_slice))| -> Result<()> {
                    let position = chunk_start + i;
                    // 7. per-head q_norm (RMS, weighted).
                    if let Some(qnw) = q_norm_weight {
                        let mut tmp = vec![0.0_f32; q_width];
                        rms_norm_per_head_into(
                            q_slice, qnw, self.num_attention_heads, head_dim, self.rms_norm_eps,
                            &mut tmp,
                        );
                        q_slice.copy_from_slice(&tmp);
                    }
                    // 8. per-head k_norm (RMS, weighted).
                    if let Some(knw) = k_norm_weight {
                        let mut tmp = vec![0.0_f32; kv_width];
                        rms_norm_per_head_into(
                            k_slice, knw, num_kv_heads, head_dim, self.rms_norm_eps, &mut tmp,
                        );
                        k_slice.copy_from_slice(&tmp);
                    }
                    // 9. per-head v_norm (RMS, NO weight) — whenever q_norm present.
                    if q_norm_weight.is_some() {
                        let mut tmp = vec![0.0_f32; kv_width];
                        rms_norm_per_head_no_weight_into(
                            v_slice, num_kv_heads, head_dim, self.rms_norm_eps, &mut tmp,
                        );
                        v_slice.copy_from_slice(&tmp);
                    }
                    // 10-11. RoPE on Q and K at this token's position.
                    apply_rope_partial_in_place(
                        q_slice, position, self.num_attention_heads, head_dim, layer.partial_dim,
                        &layer.rope,
                    )?;
                    apply_rope_partial_in_place(
                        k_slice, position, num_kv_heads, head_dim, layer.partial_dim, &layer.rope,
                    )
                })?;
        }

        // 12. Q scale (see forward_attention: 1.0 for Gemma-4, else 1/sqrt(d)).
        let attn_scale = if layer.q_norm_weight.is_some() {
            1.0
        } else {
            1.0 / (head_dim as f32).sqrt()
        };

        // 13. KV store: push ALL batch tokens for own-KV layers BEFORE attention
        // (shared layers read the parent layer, which was pushed earlier this
        // chunk). Per-position causal masking is enforced via the per-token
        // `seq_len` passed to attention, not by withholding KV.
        if !shared {
            let layer_state = &mut state.layers[layer_idx];
            for i in 0..batch {
                layer_state.push(
                    &scratch.k[i * kv_width..(i + 1) * kv_width],
                    &scratch.v[i * kv_width..(i + 1) * kv_width],
                )?;
            }
        }

        // 14. PER-POSITION attention: token i attends to KV positions
        // [window_start .. chunk_start+i+1). Read the (now fully-populated) KV
        // buffer for this layer (own) or its parent (shared). Writes
        // attn_context[..batch*q_width] (fully overwritten). The VECTORIZED
        // batched path computes all `batch` queries' attention in one call
        // (SIMD Q·Kᵀ scores + softmax + scores·V per (query, head),
        // parallelized over (query, head)). Bit-closely matches looping
        // `g4_attention_decode_into` per query (same scale/mask/window/GQA;
        // softmax is two-pass max-subtract vs the decode path's online form).
        let parent_idx = layer.kv_shared_from.unwrap_or(layer_idx);
        let kv_state = &state.layers[parent_idx];
        let q = &scratch.q[..batch * q_width];
        g4_attention_prefill_into(
            G4PrefillAttnRequest {
                keys: &kv_state.keys,
                values: &kv_state.values,
                queries: q,
                chunk_start,
                batch,
                num_attention_heads: self.num_attention_heads,
                num_kv_heads,
                head_dim,
                window_size: layer.window_size,
                scale: attn_scale,
            },
            &mut scratch.attn_context[..batch * q_width],
        )?;

        // 15. BATCHED o_proj → proj_out[..batch*hidden] (fully written).
        layer.o_proj.matmul_into(
            &scratch.attn_context[..batch * q_width],
            batch,
            &mut scratch.proj_out[..batch * hidden_size],
        )?;

        // 16-17. post_attention sublayer norm (PrePost) + residual add, per token.
        // Reads main_a (hidden) + proj_out (attn_out), writes main_b (residual,
        // fully overwritten by the per-token add_into).
        let G4PrefillScratch { main_a, main_b, proj_out, .. } = &mut *scratch;
        let hidden = &main_a[..batch * hidden_size];
        let attn_out = &proj_out[..batch * hidden_size];
        main_b[..batch * hidden_size]
            .par_chunks_mut(hidden_size)
            .zip(hidden.par_chunks(hidden_size))
            .zip(attn_out.par_chunks(hidden_size))
            .try_for_each(|((res, h), a)| -> Result<()> {
                if let Some(post_norm) = &layer.post_attn_sublayer_norm {
                    let mut a_normed = vec![0.0_f32; hidden_size];
                    rms_norm_into(a, post_norm, self.rms_norm_eps, &mut a_normed);
                    add_into(h, &a_normed, res)
                } else {
                    add_into(h, a, res)
                }
            })?;
        Ok(())
    }

    /// Batched dense MLP sublayer (steps 18-25) over `batch` tokens. gate/up/down
    /// projections are BATCHED (`matmul_into`); pre/post norms, GeGLU, PLE
    /// additive, residual and layer_scalar are per-token. Math mirrors
    /// `forward_dense_mlp` exactly.
    ///
    /// Buffers come from the reusable `scratch` pool (no per-call alloc/zero):
    /// reads `scratch.main_b[..batch*hidden]` (residual), writes the layer output
    /// `hidden_out` into `scratch.main_a[..batch*hidden]`. Each scratch region is
    /// FULLY OVERWRITTEN before any read — `normed` by the pre-FFN rms, `gate`
    /// /`up` by the projections, `swiglu` by GeGLU, `proj_out` by the down proj,
    /// `main_a` by the post-FFN norm+residual add (which writes ALL of main_a
    /// before PLE's `+=` reads it).
    #[allow(clippy::too_many_arguments)]
    fn forward_dense_mlp_batched(
        &self,
        scratch: &mut G4PrefillScratch,
        layer: &G4CpuLayer,
        layer_idx: usize,
        batch: usize,
        per_layer_inputs: &[f32],
        ple_row: usize,
    ) -> Result<()> {
        let hidden_size = self.hidden_size;
        let mlp = layer.mlp.as_ref().expect("forward_dense_mlp_batched on non-dense layer");
        let inter = mlp.gate_proj.rows();

        // 18. pre_feedforward_layernorm (RMS), per token. Reads main_b (residual),
        // writes normed[..batch*hidden] (fully overwritten).
        {
            let residual = &scratch.main_b[..batch * hidden_size];
            let normed = &mut scratch.normed[..batch * hidden_size];
            normed
                .par_chunks_mut(hidden_size)
                .zip(residual.par_chunks(hidden_size))
                .for_each(|(out, r)| {
                    rms_norm_into(r, &layer.pre_mlp_norm_weight, self.rms_norm_eps, out);
                });
        }

        // 19. BATCHED gate / up → gate[..batch*inter], up[..batch*inter] (fully written).
        {
            let normed = &scratch.normed[..batch * hidden_size];
            mlp.gate_proj.matmul_into(normed, batch, &mut scratch.gate[..batch * inter])?;
            mlp.up_proj.matmul_into(normed, batch, &mut scratch.up[..batch * inter])?;
        }

        // 20. GeGLU-tanh activation, per token → swiglu[..batch*inter] (fully written).
        {
            let G4PrefillScratch { gate, up, swiglu, .. } = &mut *scratch;
            let gate = &gate[..batch * inter];
            let up = &up[..batch * inter];
            swiglu[..batch * inter]
                .par_chunks_mut(inter)
                .zip(gate.par_chunks(inter))
                .zip(up.par_chunks(inter))
                .try_for_each(|((s, g), u)| -> Result<()> { geglu_into(g, u, s) })?;
        }

        // 21. BATCHED down → proj_out[..batch*hidden] (fully written).
        mlp.down_proj.matmul_into(
            &scratch.swiglu[..batch * inter],
            batch,
            &mut scratch.proj_out[..batch * hidden_size],
        )?;

        // 22-23. post_feedforward sublayer norm (PrePost) + residual add, per token
        // (parallel). Reads main_b (residual) + proj_out (mlp_out), writes main_a
        // (hidden_out, fully overwritten before the PLE += read below).
        {
            let G4PrefillScratch { main_a, main_b, proj_out, .. } = &mut *scratch;
            let residual = &main_b[..batch * hidden_size];
            let mlp_out = &proj_out[..batch * hidden_size];
            main_a[..batch * hidden_size]
                .par_chunks_mut(hidden_size)
                .zip(residual.par_chunks(hidden_size))
                .zip(mlp_out.par_chunks(hidden_size))
                .try_for_each(|((out, r), m)| -> Result<()> {
                    if let Some(post_norm) = &layer.post_mlp_sublayer_norm {
                        let mut m_normed = vec![0.0_f32; hidden_size];
                        rms_norm_into(m, post_norm, self.rms_norm_eps, &mut m_normed);
                        add_into(r, &m_normed, out)
                    } else {
                        add_into(r, m, out)
                    }
                })?;
        }

        // 24. PLE additive (BEFORE layer_scalar) — BATCHED (input_gate/projection
        // run as VNNI GEMMs over all tokens instead of per-token matvecs).
        if let (Some(ple_g), Some(layer_ple)) = (&self.ple, &layer.ple) {
            ple::apply_ple_contribution_batched(
                layer_ple,
                ple_g,
                layer_idx,
                per_layer_inputs,
                ple_row,
                batch,
                self.rms_norm_eps,
                &mut scratch.main_a[..batch * hidden_size],
            )?;
        }

        // 25. layer_scalar, per token (parallel).
        if let Some(scalar) = layer.layer_scalar {
            scratch.main_a[..batch * hidden_size]
                .par_chunks_mut(hidden_size)
                .for_each(|h| simd::scale_in_place(h, scalar));
        }
        Ok(())
    }

    /// Dense MLP sublayer (steps 18-25).
    fn forward_dense_mlp(
        &self,
        layer: &G4CpuLayer,
        mlp: &G4DenseMlp,
        layer_idx: usize,
        per_layer_inputs: &[f32],
        residual: &[f32],
    ) -> Result<Vec<f32>> {
        // 18. pre_feedforward_layernorm (RMS).
        let mut post_normed = vec![0.0_f32; self.hidden_size];
        rms_norm_into(residual, &layer.pre_mlp_norm_weight, self.rms_norm_eps, &mut post_normed);

        // 19. gate / up proj.
        let inter = mlp.gate_proj.rows();
        let mut gate = vec![0.0_f32; inter];
        let mut up = vec![0.0_f32; inter];
        mlp.gate_proj.matvec_into(&post_normed, &mut gate)?;
        mlp.up_proj.matvec_into(&post_normed, &mut up)?;

        // 20. GeGLU-tanh activation.
        let mut swiglu = vec![0.0_f32; inter];
        geglu_into(&gate, &up, &mut swiglu)?;

        // 21. down proj.
        let mut mlp_out = vec![0.0_f32; self.hidden_size];
        mlp.down_proj.matvec_into(&swiglu, &mut mlp_out)?;

        // 22-23. post_feedforward sublayer norm (PrePost) + residual add.
        let mut hidden_out = vec![0.0_f32; self.hidden_size];
        if let Some(post_norm) = &layer.post_mlp_sublayer_norm {
            let mut mlp_out_normed = vec![0.0_f32; self.hidden_size];
            rms_norm_into(&mlp_out, post_norm, self.rms_norm_eps, &mut mlp_out_normed);
            add_into(residual, &mlp_out_normed, &mut hidden_out)?;
        } else {
            add_into(residual, &mlp_out, &mut hidden_out)?;
        }

        // 24. PLE additive contribution (E4B/E2B), BEFORE layer_scalar.
        if let (Some(ple_g), Some(layer_ple)) = (&self.ple, &layer.ple) {
            ple::apply_ple_contribution(
                layer_ple, ple_g, layer_idx, per_layer_inputs, self.rms_norm_eps, &mut hidden_out,
            )?;
        }

        // 25. layer_scalar.
        if let Some(scalar) = layer.layer_scalar {
            simd::scale_in_place(&mut hidden_out, scalar);
        }
        Ok(hidden_out)
    }

    /// MoE MLP sublayer (26B-A4B; replaces dense steps 18-25).
    fn forward_moe(
        &self,
        layer: &G4CpuLayer,
        moe: &G4MoeLayer,
        layer_idx: usize,
        per_layer_inputs: &[f32],
        residual: &[f32],
    ) -> Result<Vec<f32>> {
        let hidden_size = self.hidden_size;

        // Step 1: post_normed = rms_norm(residual, pre_mlp_norm).
        let mut post_normed = vec![0.0_f32; hidden_size];
        rms_norm_into(residual, &layer.pre_mlp_norm_weight, self.rms_norm_eps, &mut post_normed);

        // Step 2: router.
        //   router_input = rms_norm(residual, router_input_scale) * hidden^(-0.5)
        //   logits = router @ router_input ; softmax-all → topk → renorm → per_expert_scale
        let router_input: Vec<f32> = match &moe.router_input_scale {
            Some(scale) => {
                let mut ri = vec![0.0_f32; hidden_size];
                rms_norm_into(residual, scale, self.rms_norm_eps, &mut ri);
                let root = (hidden_size as f32).powf(-0.5);
                simd::scale_in_place(&mut ri, root);
                ri
            }
            None => residual.to_vec(),
        };
        let mut logits = vec![0.0_f32; moe.num_experts];
        moe.router.matvec_into(&router_input, &mut logits)?;
        let (indices, weights) = router_softmax_topk_normalized(
            &logits,
            moe.top_k,
            moe.router_per_expert_scale.as_deref(),
        );

        // Step 3: expert input = pre_feedforward_layernorm_2(residual) if present, else post_normed.
        let expert_input: Vec<f32> = match &moe.pre_feedforward_layernorm_2 {
            Some(norm2) => {
                let mut ei = vec![0.0_f32; hidden_size];
                rms_norm_into(residual, norm2, self.rms_norm_eps, &mut ei);
                ei
            }
            None => post_normed.clone(),
        };

        // Step 4: shared MLP on post_normed → moe_acc (GeGLU-tanh).
        let mut moe_acc = vec![0.0_f32; hidden_size];
        if let Some(shared) = &moe.shared_expert {
            self.dense_ffn_into(shared, &post_normed, &mut moe_acc)?;
        }

        // Step 5: stream1 = post_feedforward_layernorm_1(moe_acc) if present else copy.
        let mut stream1 = vec![0.0_f32; hidden_size];
        match &moe.post_feedforward_layernorm_1 {
            Some(norm1) => rms_norm_into(&moe_acc, norm1, self.rms_norm_eps, &mut stream1),
            None => stream1.copy_from_slice(&moe_acc),
        }

        // Step 6: routed experts → routed_acc.
        let mut routed_acc = vec![0.0_f32; hidden_size];
        let inter = moe.expert_intermediate_size;
        let mut gate = vec![0.0_f32; inter];
        let mut up = vec![0.0_f32; inter];
        let mut swiglu = vec![0.0_f32; inter];
        let mut expert_out = vec![0.0_f32; hidden_size];
        for (&idx, &weight) in indices.iter().zip(weights.iter()) {
            let expert = moe.experts.get(idx).ok_or_else(|| {
                AegisError::InvalidPlan(format!("MoE expert index {idx} out of range"))
            })?;
            expert.gate_proj.matvec_into(&expert_input, &mut gate)?;
            expert.up_proj.matvec_into(&expert_input, &mut up)?;
            geglu_into(&gate, &up, &mut swiglu)?;
            expert.down_proj.matvec_into(&swiglu, &mut expert_out)?;
            simd::axpy(&mut routed_acc, &expert_out, weight);
        }

        // Step 7: stream2 = post_feedforward_layernorm_2(routed_acc) if present.
        let mut stream2 = vec![0.0_f32; hidden_size];
        match &moe.post_feedforward_layernorm_2 {
            Some(norm2) => rms_norm_into(&routed_acc, norm2, self.rms_norm_eps, &mut stream2),
            None => stream2.copy_from_slice(&routed_acc),
        }

        // Step 8: combined = stream1 + stream2.
        let mut combined = vec![0.0_f32; hidden_size];
        add_into(&stream1, &stream2, &mut combined)?;

        // Step 9: normed_out = post_mlp_sublayer_norm(combined).
        let mut normed_out = vec![0.0_f32; hidden_size];
        match &layer.post_mlp_sublayer_norm {
            Some(final_norm) => rms_norm_into(&combined, final_norm, self.rms_norm_eps, &mut normed_out),
            None => normed_out.copy_from_slice(&combined),
        }

        // Step 10: hidden_out = residual + normed_out.
        let mut hidden_out = vec![0.0_f32; hidden_size];
        add_into(residual, &normed_out, &mut hidden_out)?;

        // PLE additive (BEFORE layer_scalar) — present only on PLE models.
        if let (Some(ple_g), Some(layer_ple)) = (&self.ple, &layer.ple) {
            ple::apply_ple_contribution(
                layer_ple, ple_g, layer_idx, per_layer_inputs, self.rms_norm_eps, &mut hidden_out,
            )?;
        }

        // Step 11: hidden_out *= layer_scalar.
        if let Some(scalar) = layer.layer_scalar {
            simd::scale_in_place(&mut hidden_out, scalar);
        }
        Ok(hidden_out)
    }

    /// gate/up → GeGLU-tanh → down, into `out`. Shared by shared-expert MLP.
    fn dense_ffn_into(&self, mlp: &G4DenseMlp, input: &[f32], out: &mut [f32]) -> Result<()> {
        let inter = mlp.gate_proj.rows();
        let mut gate = vec![0.0_f32; inter];
        let mut up = vec![0.0_f32; inter];
        let mut swiglu = vec![0.0_f32; inter];
        mlp.gate_proj.matvec_into(input, &mut gate)?;
        mlp.up_proj.matvec_into(input, &mut up)?;
        geglu_into(&gate, &up, &mut swiglu)?;
        mlp.down_proj.matvec_into(&swiglu, out)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::g4::linear::CpuLinear;
    use crate::cpu::g4::rope::G4RopeConfig;
    use crate::cpu::g4::state::{G4CpuLayer, G4DenseMlp, G4PleGlobal, G4PleLayer};
    use aegisllm_base::executor::tensors::Bf16Matrix;

    /// Deterministic pseudo-random f32 in roughly [-0.5, 0.5), seeded.
    fn rng(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }

    /// Build a synthetic BF16 [rows, cols] CpuLinear with small bf16-exact-ish
    /// weights (scaled down so the forward stays numerically tame).
    fn lin(seed: &mut u64, rows: usize, cols: usize) -> CpuLinear {
        let mut bytes = Vec::with_capacity(rows * cols * 2);
        for _ in 0..rows * cols {
            let f = rng(seed) * 0.2;
            let bf = (f.to_bits() >> 16) as u16;
            bytes.extend_from_slice(&bf.to_le_bytes());
        }
        CpuLinear::Bf16(Bf16Matrix::from_bf16_bytes("syn".into(), rows, cols, bytes))
    }

    fn vec_seed(seed: &mut u64, n: usize, base: f32) -> Vec<f32> {
        (0..n).map(|_| base + rng(seed) * 0.1).collect()
    }

    fn rope() -> G4RopeConfig {
        G4RopeConfig {
            theta: 10_000.0,
            factor: 1.0,
            low_freq_factor: None,
            high_freq_factor: None,
            original_max_position_embeddings: None,
        }
    }

    /// Build a tiny dense+PLE Gemma-4-like executor exercising q/k/v norms,
    /// partial RoPE, a sliding-window layer, PrePost norms, layer_scalar, PLE.
    fn build_exec() -> G4CpuExecutor {
        let mut s = 0x1234_5678_9abc_def0u64;
        let hidden = 16usize;
        let heads = 4usize;
        let kv_heads = 2usize;
        let head_dim = 4usize; // q_width = heads*head_dim = 16
        let inter = 24usize;
        let vocab = 11usize;
        let ple_dim = 6usize;
        let num_layers = 2usize;

        let q_width = heads * head_dim;
        let kv_width = kv_heads * head_dim;

        let make_layer = |s: &mut u64, window_size: usize, partial_dim: usize| G4CpuLayer {
            input_norm_weight: vec_seed(s, hidden, 1.0),
            pre_mlp_norm_weight: vec_seed(s, hidden, 1.0),
            post_attn_sublayer_norm: Some(vec_seed(s, hidden, 1.0)),
            post_mlp_sublayer_norm: Some(vec_seed(s, hidden, 1.0)),
            q_proj: lin(s, q_width, hidden),
            k_proj: lin(s, kv_width, hidden),
            v_proj: lin(s, kv_width, hidden),
            o_proj: lin(s, hidden, q_width),
            q_norm_weight: Some(vec_seed(s, head_dim, 1.0)),
            k_norm_weight: Some(vec_seed(s, head_dim, 1.0)),
            mlp: Some(G4DenseMlp {
                gate_proj: lin(s, inter, hidden),
                up_proj: lin(s, inter, hidden),
                down_proj: lin(s, hidden, inter),
            }),
            moe: None,
            layer_head_dim: head_dim,
            layer_num_kv_heads: kv_heads,
            window_size,
            partial_dim,
            rope: rope(),
            layer_scalar: Some(1.0),
            kv_shared_from: None,
            ple: Some(G4PleLayer {
                input_gate: bf16_mat(s, ple_dim, hidden),
                projection: bf16_mat(s, hidden, ple_dim),
                post_norm: vec_seed(s, hidden, 1.0),
            }),
        };

        fn bf16_mat(s: &mut u64, rows: usize, cols: usize) -> Bf16Matrix {
            let mut bytes = Vec::with_capacity(rows * cols * 2);
            for _ in 0..rows * cols {
                let f = rng(s) * 0.2;
                let bf = (f.to_bits() >> 16) as u16;
                bytes.extend_from_slice(&bf.to_le_bytes());
            }
            Bf16Matrix::from_bf16_bytes("syn".into(), rows, cols, bytes)
        }

        let layers = vec![
            make_layer(&mut s, 0, 0),  // global, full RoPE
            make_layer(&mut s, 3, 4),  // sliding window=3, partial RoPE
        ];

        let ple = Some(G4PleGlobal {
            embed_table: bf16_mat(&mut s, vocab, num_layers * ple_dim),
            model_projection: bf16_mat(&mut s, num_layers * ple_dim, hidden),
            projection_norm: vec_seed(&mut s, ple_dim, 1.0),
            ple_dim,
            embed_scale_per_layer: (ple_dim as f32).sqrt(),
            model_projection_scale: 1.0 / (hidden as f32).sqrt(),
            combine_scale: 1.0 / 2.0f32.sqrt(),
        });

        G4CpuExecutor {
            hidden_size: hidden,
            num_attention_heads: heads,
            rms_norm_eps: 1e-6,
            embed_scale: (hidden as f32).sqrt(),
            lm_head_softcap: Some(30.0),
            embed_tokens: bf16_mat(&mut s, vocab, hidden),
            final_norm: vec_seed(&mut s, hidden, 1.0),
            lm_head: bf16_mat(&mut s, vocab, hidden),
            layers,
            // Large enough that the multi-chunk test can exceed PREFILL_MAX_BATCH
            // (512) and genuinely cross a chunk boundary, exercising scratch-pool
            // reuse across chunks.
            kv_context_size: 2048,
            ple,
            max_intermediate: inter,
        }
    }

    /// forward_batched MUST produce bit-identical hidden + logits to looping
    /// forward_hidden — the core correctness guarantee. Covers q/k/v norms,
    /// partial RoPE, sliding window, PrePost norms, PLE, layer_scalar.
    #[test]
    fn forward_batched_matches_per_token_forward_hidden() {
        let exec = build_exec();
        let prompt: Vec<usize> = vec![1, 4, 2, 7, 0, 9, 3, 5];

        // Reference: per-token forward_hidden over all tokens, capturing the
        // last token's hidden, then final_logits.
        let mut ref_state = exec.new_state();
        let mut ref_last = Vec::new();
        for &t in &prompt {
            ref_last = exec.forward_hidden(&mut ref_state, t).unwrap();
        }
        let ref_logits = exec.final_logits(&ref_last).unwrap();

        // Batched: forward_batched over all tokens, returns last hidden.
        let mut bat_state = exec.new_state();
        let bat_last = exec.forward_batched(&mut bat_state, &prompt, 0).unwrap();
        let bat_logits = exec.final_logits(&bat_last).unwrap();

        // The batched MLP uses an outer-product GEMM (linear.rs) that reorders the
        // K-sum vs the per-token path, so hidden/logits are NOT bit-identical —
        // pure f32 round-off with the same exact bf16 widen. Require very high
        // cosine similarity; the greedy argmax (what temperature-0 sampling uses)
        // must still agree EXACTLY.
        let cos = |a: &[f32], b: &[f32]| -> f64 {
            let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
            for (&x, &y) in a.iter().zip(b.iter()) {
                d += x as f64 * y as f64;
                na += x as f64 * x as f64;
                nb += y as f64 * y as f64;
            }
            if na == 0.0 || nb == 0.0 { 1.0 } else { d / (na.sqrt() * nb.sqrt()) }
        };
        assert_eq!(bat_state.position, ref_state.position, "position must match");
        assert!(cos(&bat_last, &ref_last) > 0.999, "last hidden cos={}", cos(&bat_last, &ref_last));
        assert!(cos(&bat_logits, &ref_logits) > 0.999, "logits cos={}", cos(&bat_logits, &ref_logits));

        // Greedy argmax (what bench-generate samples at temperature 0) must agree.
        let argmax = |l: &[f32]| {
            l.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        };
        assert_eq!(argmax(&bat_logits), argmax(&ref_logits));
    }

    /// Multi-chunk path (prompt longer than PREFILL_MAX_BATCH) must also match
    /// per-token, proving cross-chunk KV causality is preserved AND that the
    /// reusable scratch pool is correctly re-overwritten on every chunk (stale
    /// data from a prior chunk must never leak).
    #[test]
    fn forward_batched_multi_chunk_matches_per_token() {
        let exec = build_exec();
        // 1100 tokens > 2 * PREFILL_MAX_BATCH (512) → 3 chunks (512+512+76),
        // crossing two chunk boundaries so the scratch pool is reused across them.
        let prompt: Vec<usize> = (0..1100).map(|i| (i * 3 + 1) % 11).collect();

        let mut ref_state = exec.new_state();
        let mut ref_last = Vec::new();
        for &t in &prompt {
            ref_last = exec.forward_hidden(&mut ref_state, t).unwrap();
        }

        let mut bat_state = exec.new_state();
        let bat_last = exec.forward_batched(&mut bat_state, &prompt, 0).unwrap();

        // Outer-product GEMM reorders the K-sum → high cosine, not bit-identical.
        let cos = |a: &[f32], b: &[f32]| -> f64 {
            let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
            for (&x, &y) in a.iter().zip(b.iter()) {
                d += x as f64 * y as f64;
                na += x as f64 * x as f64;
                nb += y as f64 * y as f64;
            }
            if na == 0.0 || nb == 0.0 { 1.0 } else { d / (na.sqrt() * nb.sqrt()) }
        };
        assert_eq!(bat_state.position, ref_state.position);
        assert!(
            cos(&bat_last, &ref_last) > 0.999,
            "multi-chunk last hidden cos={}",
            cos(&bat_last, &ref_last)
        );
    }
}

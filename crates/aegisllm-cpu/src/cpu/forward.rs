use super::attention::attention_into;
use super::math::{add_into, rms_norm_into, swiglu_into};
use super::rope::apply_rope_in_place;
use super::state::{CpuLlamaExecutor, CpuLlamaState, CpuScratch};
use crate::attention::{
    ReferenceAttentionPrefillRequest, reference_attention_prefill_f32_into,
};
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::generation::SamplingConfig;
use rayon::prelude::*;

/// Smallest prefill batch where the matmul-batched path is worth it.
/// Below this, per-token forward_hidden is comparable and avoids the extra setup.
const PREFILL_BATCH_THRESHOLD: usize = 4;
/// Maximum tokens per batched prefill chunk: bigger gives more cache reuse but uses
/// more scratch RAM (batch * intermediate_size floats).
const PREFILL_MAX_BATCH: usize = 64;

impl CpuLlamaExecutor {
    pub(super) fn prefill_prompt(
        &self,
        state: &mut CpuLlamaState,
        prompt_tokens: &[usize],
        sampling: &SamplingConfig,
    ) -> Result<usize> {
        let Some((&last, prefix)) = prompt_tokens.split_last() else {
            return Err(AegisError::InvalidConfig(
                "prompt produced no tokens".into(),
            ));
        };
        if state.position + prompt_tokens.len() > self.kv_context_size {
            return Err(AegisError::InvalidPlan(format!(
                "prefill would exhaust kv cache: position={} prompt_tokens={} context={}",
                state.position,
                prompt_tokens.len(),
                self.kv_context_size
            )));
        }
        // Batched prefill: process N tokens at once with matmul instead of N matvecs.
        // Falls back to per-token forward_hidden for tiny prompts (overhead would dominate).
        if prefix.len() >= PREFILL_BATCH_THRESHOLD {
            self.prefill_batched(state, prefix)?;
        } else {
            for &token in prefix {
                let _hidden = self.forward_hidden(state, token)?;
            }
        }
        let logits = self.forward_logits(state, last)?;
        aegisllm_base::executor::generation::sample_next_token(&logits, sampling)
    }

    /// Batched prefill: processes `tokens` through all layers using matrix-batched projections.
    /// KV cache is updated in order; attention uses the prefill (causal) variant.
    fn prefill_batched(
        &self,
        state: &mut CpuLlamaState,
        tokens: &[usize],
    ) -> Result<()> {
        let hidden_size = self.hidden_size;
        let attn_width = self.num_attention_heads * self.head_dim;
        let kv_width = self.num_kv_heads * self.head_dim;
        let intermediate = self
            .layers
            .first()
            .map(|l| l.gate_proj.rows)
            .unwrap_or(hidden_size);

        // Process in chunks to bound scratch memory. Each chunk's KV cache is appended
        // sequentially so causal attention semantics are preserved across chunks.
        for chunk in tokens.chunks(PREFILL_MAX_BATCH) {
            let batch = chunk.len();
            let start_position = state.position;

            // Embed all tokens in chunk into a flat batch buffer.
            let mut hidden = vec![0.0_f32; batch * hidden_size];
            for (idx, &token) in chunk.iter().enumerate() {
                let row = self.embed_tokens.row(token)?;
                hidden[idx * hidden_size..(idx + 1) * hidden_size].copy_from_slice(&row);
            }

            for (layer_idx, layer) in self.layers.iter().enumerate() {
                // Batched RMS norm.
                let mut input_normed = vec![0.0_f32; batch * hidden_size];
                input_normed
                    .par_chunks_mut(hidden_size)
                    .zip(hidden.par_chunks(hidden_size))
                    .for_each(|(out, h)| {
                        rms_norm_into(h, &layer.input_norm_weight, self.rms_norm_eps, out);
                    });

                // Batched Q/K/V projections (the GPU-equivalent saving here).
                let mut q = vec![0.0_f32; batch * attn_width];
                let mut k = vec![0.0_f32; batch * kv_width];
                let mut v = vec![0.0_f32; batch * kv_width];
                layer.q_proj.matmul_into(&input_normed, batch, &mut q)?;
                layer.k_proj.matmul_into(&input_normed, batch, &mut k)?;
                layer.v_proj.matmul_into(&input_normed, batch, &mut v)?;

                // Per-token RoPE (positions are sequential in the chunk).
                for token in 0..batch {
                    let pos = start_position + token;
                    let q_slice =
                        &mut q[token * attn_width..(token + 1) * attn_width];
                    apply_rope_in_place(
                        q_slice,
                        pos,
                        self.num_attention_heads,
                        self.head_dim,
                        &self.rope,
                    )?;
                    let k_slice =
                        &mut k[token * kv_width..(token + 1) * kv_width];
                    apply_rope_in_place(
                        k_slice,
                        pos,
                        self.num_kv_heads,
                        self.head_dim,
                        &self.rope,
                    )?;
                }

                // Append this chunk's K/V to the layer's KV cache.
                let layer_state = &mut state.layers[layer_idx];
                for token in 0..batch {
                    let k_slice = &k[token * kv_width..(token + 1) * kv_width];
                    let v_slice = &v[token * kv_width..(token + 1) * kv_width];
                    layer_state.push(k_slice, v_slice, kv_width)?;
                }

                // Prefill attention with full causal mask: each token attends to itself + past.
                let mut attn_context = vec![0.0_f32; batch * attn_width];
                reference_attention_prefill_f32_into(
                    ReferenceAttentionPrefillRequest {
                        keys: &layer_state.keys,
                        values: &layer_state.values,
                        start_position,
                        batch,
                        query: &q,
                        num_attention_heads: self.num_attention_heads,
                        num_kv_heads: self.num_kv_heads,
                        head_dim: self.head_dim,
                    },
                    &mut attn_context,
                )?;

                // Batched O projection.
                let mut attn_out = vec![0.0_f32; batch * hidden_size];
                layer.o_proj.matmul_into(&attn_context, batch, &mut attn_out)?;

                // Residual: hidden = hidden + attn_out (in place).
                hidden
                    .par_chunks_mut(hidden_size)
                    .zip(attn_out.par_chunks(hidden_size))
                    .for_each(|(h, a)| {
                        super::simd::add_in_place(h, a);
                    });

                // Batched post-attention RMS norm.
                let mut post_normed = vec![0.0_f32; batch * hidden_size];
                post_normed
                    .par_chunks_mut(hidden_size)
                    .zip(hidden.par_chunks(hidden_size))
                    .for_each(|(out, h)| {
                        rms_norm_into(h, &layer.post_attention_norm_weight, self.rms_norm_eps, out);
                    });

                // Batched gate/up.
                let mut gate = vec![0.0_f32; batch * intermediate];
                let mut up = vec![0.0_f32; batch * intermediate];
                layer.gate_proj.matmul_into(&post_normed, batch, &mut gate)?;
                layer.up_proj.matmul_into(&post_normed, batch, &mut up)?;

                // Batched SwiGLU.
                let mut swiglu = vec![0.0_f32; batch * intermediate];
                swiglu
                    .par_chunks_mut(intermediate)
                    .zip(gate.par_chunks(intermediate))
                    .zip(up.par_chunks(intermediate))
                    .for_each(|((s, g), u)| {
                        super::simd::swiglu_into_simd(g, u, s);
                    });

                // Batched down.
                let mut mlp_out = vec![0.0_f32; batch * hidden_size];
                layer.down_proj.matmul_into(&swiglu, batch, &mut mlp_out)?;

                // Residual: hidden = hidden + mlp_out (in place).
                hidden
                    .par_chunks_mut(hidden_size)
                    .zip(mlp_out.par_chunks(hidden_size))
                    .for_each(|(h, m)| {
                        super::simd::add_in_place(h, m);
                    });
            }

            state.position += batch;
        }
        Ok(())
    }

    pub(super) fn forward_hidden(
        &self,
        state: &mut CpuLlamaState,
        token_id: usize,
    ) -> Result<Vec<f32>> {
        if state.position >= self.kv_context_size {
            return Err(AegisError::InvalidPlan(format!(
                "kv cache context exhausted: position={} context={}",
                state.position, self.kv_context_size
            )));
        }
        let mut hidden = self.embed_tokens.row(token_id)?;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let scratch = &mut state.scratch;
            rms_norm_into(
                &hidden,
                &layer.input_norm_weight,
                self.rms_norm_eps,
                &mut scratch.input_normed,
            );
            layer
                .q_proj
                .matvec_into(&scratch.input_normed, &mut scratch.q)?;
            layer
                .k_proj
                .matvec_into(&scratch.input_normed, &mut scratch.k)?;
            layer
                .v_proj
                .matvec_into(&scratch.input_normed, &mut scratch.v)?;

            apply_rope_in_place(
                &mut scratch.q,
                state.position,
                self.num_attention_heads,
                self.head_dim,
                &self.rope,
            )?;
            apply_rope_in_place(
                &mut scratch.k,
                state.position,
                self.num_kv_heads,
                self.head_dim,
                &self.rope,
            )?;
            let layer_state = &mut state.layers[layer_idx];
            layer_state.push(&scratch.k, &scratch.v, self.num_kv_heads * self.head_dim)?;
            attention_into(
                layer_state,
                &scratch.q,
                self.num_attention_heads,
                self.num_kv_heads,
                self.head_dim,
                &mut scratch.attn_context,
            )?;
            layer
                .o_proj
                .matvec_into(&scratch.attn_context, &mut scratch.attn_out)?;
            add_into(&hidden, &scratch.attn_out, &mut scratch.residual)?;

            rms_norm_into(
                &scratch.residual,
                &layer.post_attention_norm_weight,
                self.rms_norm_eps,
                &mut scratch.post_normed,
            );
            layer
                .gate_proj
                .matvec_into(&scratch.post_normed, &mut scratch.gate)?;
            layer
                .up_proj
                .matvec_into(&scratch.post_normed, &mut scratch.up)?;
            swiglu_into(&scratch.gate, &scratch.up, &mut scratch.swiglu)?;
            layer
                .down_proj
                .matvec_into(&scratch.swiglu, &mut scratch.mlp_out)?;
            add_into(&scratch.residual, &scratch.mlp_out, &mut scratch.hidden_out)?;
            std::mem::swap(&mut hidden, &mut scratch.hidden_out);
        }

        state.position += 1;
        Ok(hidden)
    }

    pub(super) fn forward_logits(
        &self,
        state: &mut CpuLlamaState,
        token_id: usize,
    ) -> Result<Vec<f32>> {
        let hidden = self.forward_hidden(state, token_id)?;
        self.final_logits_host_with_scratch(&hidden, &mut state.scratch)
    }

    #[allow(dead_code)]
    pub(super) fn forward_layer_host(
        &self,
        state: &mut CpuLlamaState,
        layer_idx: usize,
        position: usize,
        hidden: &[f32],
    ) -> Result<Vec<f32>> {
        let layer = self
            .layers
            .get(layer_idx)
            .ok_or_else(|| AegisError::InvalidPlan(format!("missing CPU layer `{layer_idx}`")))?;
        let mut scratch = CpuScratch::new(self);
        rms_norm_into(
            hidden,
            &layer.input_norm_weight,
            self.rms_norm_eps,
            &mut scratch.input_normed,
        );
        layer
            .q_proj
            .matvec_into(&scratch.input_normed, &mut scratch.q)?;
        layer
            .k_proj
            .matvec_into(&scratch.input_normed, &mut scratch.k)?;
        layer
            .v_proj
            .matvec_into(&scratch.input_normed, &mut scratch.v)?;

        apply_rope_in_place(
            &mut scratch.q,
            position,
            self.num_attention_heads,
            self.head_dim,
            &self.rope,
        )?;
        apply_rope_in_place(
            &mut scratch.k,
            position,
            self.num_kv_heads,
            self.head_dim,
            &self.rope,
        )?;
        let layer_state = state.layers.get_mut(layer_idx).ok_or_else(|| {
            AegisError::InvalidPlan(format!("missing CPU layer state `{layer_idx}`"))
        })?;
        layer_state.push(&scratch.k, &scratch.v, self.num_kv_heads * self.head_dim)?;
        attention_into(
            layer_state,
            &scratch.q,
            self.num_attention_heads,
            self.num_kv_heads,
            self.head_dim,
            &mut scratch.attn_context,
        )?;
        layer
            .o_proj
            .matvec_into(&scratch.attn_context, &mut scratch.attn_out)?;
        add_into(hidden, &scratch.attn_out, &mut scratch.residual)?;

        rms_norm_into(
            &scratch.residual,
            &layer.post_attention_norm_weight,
            self.rms_norm_eps,
            &mut scratch.post_normed,
        );
        layer
            .gate_proj
            .matvec_into(&scratch.post_normed, &mut scratch.gate)?;
        layer
            .up_proj
            .matvec_into(&scratch.post_normed, &mut scratch.up)?;
        swiglu_into(&scratch.gate, &scratch.up, &mut scratch.swiglu)?;
        layer
            .down_proj
            .matvec_into(&scratch.swiglu, &mut scratch.mlp_out)?;
        add_into(&scratch.residual, &scratch.mlp_out, &mut scratch.hidden_out)?;
        Ok(scratch.hidden_out)
    }

    #[allow(dead_code)]
    pub(super) fn final_logits_host(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        let mut scratch = CpuScratch::new(self);
        self.final_logits_host_with_scratch(hidden, &mut scratch)
    }

    pub(super) fn final_logits_host_with_scratch(
        &self,
        hidden: &[f32],
        scratch: &mut CpuScratch,
    ) -> Result<Vec<f32>> {
        rms_norm_into(
            hidden,
            &self.final_norm,
            self.rms_norm_eps,
            &mut scratch.final_hidden,
        );
        let mut logits = vec![0.0; self.lm_head.rows];
        self.lm_head
            .matvec_into(&scratch.final_hidden, &mut logits)?;
        Ok(logits)
    }
}

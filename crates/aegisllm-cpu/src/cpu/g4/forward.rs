//! Gemma-4 CPU forward driver. Mirrors the CUDA decode path op-for-op
//! (`crates/aegisllm-cuda/src/executor/{forward,attention,mlp,ple}.rs`).
//!
//! Decode single-token is the primary path; batched prefill reuses
//! `forward_hidden` per token (correctness-first — a batched-matmul prefill is
//! a follow-up).

use super::attention::{g4_attention_decode_into, G4DecodeAttnRequest};
use super::moe::router_softmax_topk_normalized;
use super::norm::{rms_norm_per_head_into, rms_norm_per_head_no_weight_into};
use super::ple;
use super::rope::apply_rope_partial_in_place;
use super::state::{G4CpuExecutor, G4CpuLayer, G4CpuState, G4DenseMlp, G4MoeLayer};
use crate::cpu::math::{add_into, geglu_into, rms_norm_into};
use crate::cpu::simd;
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::executor::generation::apply_logit_softcap;
use aegisllm_base::generation::SamplingConfig;

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
        // Correctness-first: per-token prefill (causal semantics preserved by
        // the position-indexed KV cache and the attention mask).
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
        // final_norm (RMS).
        let mut final_hidden = vec![0.0_f32; self.hidden_size];
        rms_norm_into(&hidden, &self.final_norm, self.rms_norm_eps, &mut final_hidden);
        // lm_head.
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

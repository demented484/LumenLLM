use super::attention::attention_into;
use super::math::{add_into, rms_norm_into, swiglu_into};
use super::rope::apply_rope_in_place;
use super::state::{CpuLlamaExecutor, CpuLlamaState, CpuScratch};
use crate::error::{AegisError, Result};
use crate::generation::SamplingConfig;

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
        for &token in prefix {
            let _hidden = self.forward_hidden(state, token)?;
        }
        let logits = self.forward_logits(state, last)?;
        crate::executor::generation::sample_next_token(&logits, sampling)
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

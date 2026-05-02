use super::rope::RopeConfig;
use super::CpuNvfp4Linear;
use crate::error::{AegisError, Result};
use crate::executor::tensors::Bf16Matrix;

#[derive(Debug)]
pub(super) struct CpuLlamaExecutor {
    pub(super) hidden_size: usize,
    pub(super) num_attention_heads: usize,
    pub(super) num_kv_heads: usize,
    pub(super) head_dim: usize,
    pub(super) rms_norm_eps: f32,
    pub(super) rope: RopeConfig,
    pub(super) embed_tokens: Bf16Matrix,
    pub(super) final_norm: Vec<f32>,
    pub(super) lm_head: Bf16Matrix,
    pub(super) layers: Vec<CpuLayer>,
    pub(super) kv_context_size: usize,
}

#[derive(Debug)]
pub(super) struct CpuLayer {
    pub(super) input_norm_weight: Vec<f32>,
    pub(super) post_attention_norm_weight: Vec<f32>,
    pub(super) q_proj: CpuNvfp4Linear,
    pub(super) k_proj: CpuNvfp4Linear,
    pub(super) v_proj: CpuNvfp4Linear,
    pub(super) o_proj: CpuNvfp4Linear,
    pub(super) gate_proj: CpuNvfp4Linear,
    pub(super) up_proj: CpuNvfp4Linear,
    pub(super) down_proj: CpuNvfp4Linear,
}

#[derive(Debug)]
pub(in crate::executor) struct CpuLlamaState {
    pub(super) position: usize,
    pub(super) layers: Vec<CpuLayerState>,
    pub(super) scratch: CpuScratch,
}

#[derive(Debug)]
pub(super) struct CpuLayerState {
    pub(super) keys: Vec<f32>,
    pub(super) values: Vec<f32>,
    pub(super) seq_len: usize,
}

impl CpuLayerState {
    pub(super) fn push(&mut self, key: &[f32], value: &[f32], width: usize) -> Result<()> {
        if key.len() != width || value.len() != width {
            return Err(AegisError::InvalidPlan(format!(
                "kv cache push shape mismatch: expected {width}, got key={} value={}",
                key.len(),
                value.len()
            )));
        }
        self.keys.extend_from_slice(key);
        self.values.extend_from_slice(value);
        self.seq_len += 1;
        Ok(())
    }
}

impl CpuLlamaExecutor {
    pub(super) fn new_state(&self) -> CpuLlamaState {
        let kv_width = self.num_kv_heads * self.head_dim;
        CpuLlamaState {
            position: 0,
            layers: (0..self.layers.len())
                .map(|_| CpuLayerState {
                    keys: Vec::with_capacity(self.kv_context_size.min(256) * kv_width),
                    values: Vec::with_capacity(self.kv_context_size.min(256) * kv_width),
                    seq_len: 0,
                })
                .collect(),
            scratch: CpuScratch::new(self),
        }
    }

    #[allow(dead_code)]
    pub(super) fn embed_token(&self, token_id: usize) -> Result<Vec<f32>> {
        self.embed_tokens.row(token_id)
    }
}

#[derive(Debug)]
pub(super) struct CpuScratch {
    pub(super) input_normed: Vec<f32>,
    pub(super) q: Vec<f32>,
    pub(super) k: Vec<f32>,
    pub(super) v: Vec<f32>,
    pub(super) attn_context: Vec<f32>,
    pub(super) attn_out: Vec<f32>,
    pub(super) residual: Vec<f32>,
    pub(super) post_normed: Vec<f32>,
    pub(super) gate: Vec<f32>,
    pub(super) up: Vec<f32>,
    pub(super) swiglu: Vec<f32>,
    pub(super) mlp_out: Vec<f32>,
    pub(super) hidden_out: Vec<f32>,
    pub(super) final_hidden: Vec<f32>,
}

impl CpuScratch {
    pub(super) fn new_for_shape(
        hidden: usize,
        attn: usize,
        kv: usize,
        intermediate: usize,
    ) -> Self {
        Self {
            input_normed: vec![0.0; hidden],
            q: vec![0.0; attn],
            k: vec![0.0; kv],
            v: vec![0.0; kv],
            attn_context: vec![0.0; attn],
            attn_out: vec![0.0; hidden],
            residual: vec![0.0; hidden],
            post_normed: vec![0.0; hidden],
            gate: vec![0.0; intermediate],
            up: vec![0.0; intermediate],
            swiglu: vec![0.0; intermediate],
            mlp_out: vec![0.0; hidden],
            hidden_out: vec![0.0; hidden],
            final_hidden: vec![0.0; hidden],
        }
    }

    pub(super) fn new(model: &CpuLlamaExecutor) -> Self {
        let hidden = model.hidden_size;
        let attn = model.num_attention_heads * model.head_dim;
        let kv = model.num_kv_heads * model.head_dim;
        let intermediate = model
            .layers
            .first()
            .map(|layer| layer.gate_proj.rows)
            .unwrap_or(hidden);
        Self::new_for_shape(hidden, attn, kv, intermediate)
    }
}

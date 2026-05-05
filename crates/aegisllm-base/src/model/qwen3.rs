//! Qwen 3.5 / 3.6 architecture family.
//!
//! Dense variants are plain decoders (Llama-compatible tensor names).
//! Hybrid variants (35B-A3B, 397B-A17B) interleave GDN linear-attention
//! layers (every 3 out of 4 layers) with full-attention layers, plus sparse MoE.
//!
//! Layer type is determined by (in priority order):
//!   1. Explicit `layer_types` array in config.json.
//!   2. `full_attention_interval` (every N-th layer is full, rest GDN).
//!   3. `linear_attn_every_n_layers` legacy field.
//!   4. If `use_linear_attention` is false / absent → all dense.

use crate::artifact::{HfConfig, ModelArtifact};
use crate::error::Result;
use crate::graph::ModelGraph;
use crate::model::{
    AttentionPattern, LayerKind, ModelArchitecture, NormPattern, RopeConfig,
};
use crate::model::llama::rope_from_hf_config;

#[derive(Debug, Clone, Copy)]
pub struct Qwen3Architecture;

impl ModelArchitecture for Qwen3Architecture {
    fn name(&self) -> &'static str {
        "qwen3"
    }

    fn build_graph(&self, artifact: &ModelArtifact) -> Result<ModelGraph> {
        ModelGraph::build_llama_style(artifact, self)
    }

    fn layer_kind(&self, layer_idx: usize, config: &HfConfig) -> LayerKind {
        let num_experts = config.num_experts.unwrap_or(0);
        let has_moe = num_experts > 1;
        let is_linear = is_linear_attention_layer(layer_idx, config);

        if is_linear {
            return LayerKind::LinearAttentionDecoder;
        }
        if has_moe {
            let top_k = config.num_experts_per_tok.unwrap_or(2);
            let has_shared = config.num_shared_experts.unwrap_or(0) > 0;
            return LayerKind::MoEDecoder {
                num_experts,
                top_k,
                has_shared_expert: has_shared,
            };
        }
        LayerKind::DenseDecoder
    }

    fn attention_pattern(&self, layer_idx: usize, config: &HfConfig) -> AttentionPattern {
        match self.layer_kind(layer_idx, config) {
            LayerKind::LinearAttentionDecoder => AttentionPattern::LinearGatedDeltaNet,
            _ => AttentionPattern::FullCausal,
        }
    }

    fn norm_pattern(&self) -> NormPattern {
        NormPattern::PreOnly
    }

    fn lm_head_softcap(&self, _config: &HfConfig) -> Option<f32> {
        None
    }

    fn rope_config(&self, config: &HfConfig) -> RopeConfig {
        let partial = config.partial_rotary_factor.unwrap_or(1.0);
        rope_from_hf_config(config, partial)
    }
}

/// Returns true if layer `layer_idx` is a Gated DeltaNet linear-attention layer.
pub fn is_linear_attention_layer(layer_idx: usize, config: &HfConfig) -> bool {
    // 1. Explicit per-layer type array (Qwen3.5-9B, Qwen3.6-35B).
    if let Some(types) = &config.layer_types {
        if let Some(t) = types.get(layer_idx) {
            return t == "linear_attention";
        }
    }
    // 2. full_attention_interval: N → every N-th layer is full, others are GDN.
    if let Some(interval) = config.full_attention_interval {
        if interval > 0 {
            return layer_idx % interval != (interval - 1);
        }
    }
    // 3. Legacy explicit flag / frequency.
    if config.use_linear_attention != Some(true) {
        return false;
    }
    if let Some(n) = config.num_linear_attention_layers {
        return layer_idx < n;
    }
    if let Some(freq) = config.linear_attn_every_n_layers {
        return layer_idx % freq != (freq - 1);
    }
    false
}

//! Qwen 3.5 / 3.6 architecture family.
//!
//! Dense variants (0.8B, 2B, 4B, 9B, 27B) are plain decoders — structurally
//! Llama-compatible tensor names. Hybrid MoE variants (35B-A3B, 397B-A17B)
//! add Gated DeltaNet (GDN) linear-attention layers plus sparse MoE.
//! The default tensor naming follows Llama conventions.

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
        // Hybrid MoE: interleave MoE and GDN layers.
        // Pattern: every `linear_attn_every_n_layers`-th layer is GDN; the rest are MoE or dense.
        if let Some(num_experts) = config.num_experts.filter(|&n| n > 1) {
            let top_k = config.num_experts_per_tok.unwrap_or(2);
            let has_shared = config.num_shared_experts.unwrap_or(0) > 0;

            // Check if this layer is a linear-attention layer.
            if is_linear_attention_layer(layer_idx, config) {
                return LayerKind::LinearAttentionDecoder;
            }
            return LayerKind::MoEDecoder {
                num_experts,
                top_k,
                has_shared_expert: has_shared,
            };
        }
        // Pure dense (no MoE flag)
        if is_linear_attention_layer(layer_idx, config) {
            return LayerKind::LinearAttentionDecoder;
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
        rope_from_hf_config(config, 1.0)
    }
}

/// Returns true if the layer at `layer_idx` is a linear-attention (GDN) layer.
fn is_linear_attention_layer(layer_idx: usize, config: &HfConfig) -> bool {
    if config.use_linear_attention != Some(true) {
        return false;
    }
    // Explicit count: the first N layers are linear-attention.
    if let Some(n) = config.num_linear_attention_layers {
        return layer_idx < n;
    }
    // Frequency-based: every `linear_attn_every_n_layers`-th layer group has one GDN layer.
    // Pattern for Qwen 3.5 MoE: 3 GDN : 1 full-attention repeating.
    if let Some(freq) = config.linear_attn_every_n_layers {
        // freq=4 → positions 0,1,2 in each group of 4 are GDN, position 3 is full.
        return layer_idx % freq != (freq - 1);
    }
    false
}

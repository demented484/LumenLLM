//! Nemotron 3 Nano Omni architecture.
//!
//! Nemotron 3 is a hybrid backbone interleaving:
//! - 23 Mamba2 SSM layers
//! - 23 Mixture-of-Experts (MoE) FFN layers
//! - 6 standard GQA (full-causal) attention layers
//!
//! The exact interleaving pattern is encoded in the model config.
//! When not specified, we use the published 30B-A3B pattern.

use crate::artifact::{HfConfig, ModelArtifact};
use crate::error::Result;
use crate::graph::ModelGraph;
use crate::model::{
    AttentionPattern, LayerKind, ModelArchitecture, NormPattern, RopeConfig,
};
use crate::model::llama::rope_from_hf_config;

/// Nemotron 3 30B-A3B published layer pattern (0=Mamba, 1=MoE, 2=GQA).
/// Length must equal num_hidden_layers (52 total for 30B-A3B).
const NEMOTRON3_30B_PATTERN: &[u8] = &[
    0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1,
    2, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
    2, 0, 0, 2,
];

#[derive(Debug, Clone, Copy)]
pub struct Nemotron3Architecture;

impl ModelArchitecture for Nemotron3Architecture {
    fn name(&self) -> &'static str {
        "nemotron3"
    }

    fn build_graph(&self, artifact: &ModelArtifact) -> Result<ModelGraph> {
        ModelGraph::build_llama_style(artifact, self)
    }

    fn layer_kind(&self, layer_idx: usize, config: &HfConfig) -> LayerKind {
        let num_experts = config.num_experts.unwrap_or(0);
        let top_k = config.num_experts_per_tok.unwrap_or(6);
        let has_shared = config.num_shared_experts.unwrap_or(0) > 0;

        match layer_type_code(layer_idx, config) {
            0 => LayerKind::MambaDecoder,
            1 => LayerKind::MoEDecoder { num_experts: num_experts.max(128), top_k, has_shared_expert: has_shared },
            2 => LayerKind::DenseDecoder,
            _ => LayerKind::DenseDecoder,
        }
    }

    fn attention_pattern(&self, layer_idx: usize, config: &HfConfig) -> AttentionPattern {
        match layer_type_code(layer_idx, config) {
            0 => AttentionPattern::Mamba,
            2 => AttentionPattern::FullCausal,
            _ => AttentionPattern::None, // MoE uses no separate attention
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

/// Returns 0=Mamba, 1=MoE, 2=GQA for a given layer index.
fn layer_type_code(layer_idx: usize, config: &HfConfig) -> u8 {
    // Use the published pattern if the layer count matches.
    if config.num_hidden_layers == NEMOTRON3_30B_PATTERN.len() {
        return NEMOTRON3_30B_PATTERN
            .get(layer_idx)
            .copied()
            .unwrap_or(2);
    }
    // Generic fallback: interleave Mamba / MoE / GQA in 3:3:1 ratio.
    match layer_idx % 7 {
        0..=2 => 0, // Mamba
        3..=5 => 1, // MoE
        _ => 2,     // GQA (index 6)
    }
}

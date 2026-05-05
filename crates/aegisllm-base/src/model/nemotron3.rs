//! Nemotron 3 / NemotronH hybrid backbone.
//!
//! The layer interleaving is controlled by the `hybrid_override_pattern` string
//! in the model config:
//!   'M' = Mamba2 SSM layer
//!   'E' = Mixture-of-Experts FFN layer
//!   '*' = Standard full-causal attention (GQA) layer
//!
//! For Nemotron-3-Nano-Omni-30B-A3B the pattern is:
//!   "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
//!   (52 layers: 23 Mamba, 23 MoE, 6 attention)
//!
//! When `hybrid_override_pattern` is absent the published 30B-A3B hardcoded
//! pattern is used as fallback.

use crate::artifact::{HfConfig, ModelArtifact};
use crate::error::Result;
use crate::graph::ModelGraph;
use crate::model::{
    AttentionPattern, LayerKind, ModelArchitecture, NormPattern, RopeConfig,
};
use crate::model::llama::rope_from_hf_config;

/// Published 30B-A3B pattern used as fallback when `hybrid_override_pattern`
/// is absent. 'M'=77, 'E'=69, '*'=42 in ASCII.
const NEMOTRON3_30B_PATTERN: &str =
    "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME";

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
        let num_experts = config.num_experts.unwrap_or(128);
        let top_k = config.num_experts_per_tok.unwrap_or(6);
        let has_shared = config.num_shared_experts.unwrap_or(0) > 0;

        match layer_char(layer_idx, config) {
            b'M' => LayerKind::MambaDecoder,
            b'E' => LayerKind::MoEDecoder { num_experts, top_k, has_shared_expert: has_shared },
            _ => LayerKind::DenseDecoder, // '*' = GQA full attention
        }
    }

    fn attention_pattern(&self, layer_idx: usize, config: &HfConfig) -> AttentionPattern {
        match layer_char(layer_idx, config) {
            b'M' => AttentionPattern::Mamba,
            b'E' => AttentionPattern::None,
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

/// Returns the pattern character for `layer_idx`.
fn layer_char(layer_idx: usize, config: &HfConfig) -> u8 {
    let pattern = config
        .hybrid_override_pattern
        .as_deref()
        .unwrap_or(NEMOTRON3_30B_PATTERN);
    pattern.as_bytes().get(layer_idx).copied().unwrap_or(b'*')
}

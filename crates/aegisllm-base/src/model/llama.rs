use crate::artifact::{HfConfig, ModelArtifact};
use crate::error::Result;
use crate::graph::ModelGraph;
use crate::model::{
    AttentionPattern, LayerKind, ModelArchitecture, NormPattern, RopeConfig, RopeScalingConfig,
};

#[derive(Debug, Clone, Copy)]
pub struct LlamaArchitecture;

impl ModelArchitecture for LlamaArchitecture {
    fn name(&self) -> &'static str {
        "llama"
    }

    fn build_graph(&self, artifact: &ModelArtifact) -> Result<ModelGraph> {
        ModelGraph::build_llama_style(artifact, self)
    }

    fn layer_kind(&self, _layer_idx: usize, config: &HfConfig) -> LayerKind {
        // Mistral / Llama 3.x with sliding window
        if let Some(w) = config.sliding_window.filter(|&w| w > 0) {
            return LayerKind::SlidingWindowDecoder { window: w };
        }
        LayerKind::DenseDecoder
    }

    fn attention_pattern(&self, layer_idx: usize, config: &HfConfig) -> AttentionPattern {
        match self.layer_kind(layer_idx, config) {
            LayerKind::SlidingWindowDecoder { window } => AttentionPattern::SlidingWindow { size: window },
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

pub(crate) fn rope_from_hf_config(config: &HfConfig, partial_factor: f32) -> RopeConfig {
    let theta = config.rope_theta.unwrap_or(10_000.0) as f32;
    let scaling = config.rope_scaling.as_ref().map(|rs| RopeScalingConfig {
        rope_type: rs
            .rope_type
            .clone()
            .unwrap_or_else(|| "linear".to_string()),
        factor: rs.factor.unwrap_or(1.0) as f32,
        low_freq_factor: rs.low_freq_factor.map(|v| v as f32),
        high_freq_factor: rs.high_freq_factor.map(|v| v as f32),
        original_max_position_embeddings: rs.original_max_position_embeddings,
    });
    RopeConfig {
        theta,
        partial_factor,
        scaling,
    }
}

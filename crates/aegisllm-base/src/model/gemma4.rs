//! Gemma 4 architecture family.
//!
//! All Gemma 4 models interleave "local" (sliding-window) and "global"
//! (full-causal) attention layers. The default pattern is every 5th layer
//! uses global attention (`global_attn_every_n_layers = 5`).
//!
//! Gemma 4 also uses:
//! - Pre+Post RMSNorm (after attention and MLP output, in addition to pre-norm).
//! - Logit soft-capping on attention logits and final lm_head logits.
//! - Proportional RoPE (partial_rotary_factor < 1.0) on global layers.
//! - MatFormer nested-parameter checkpoints for E2B/E4B variants.

use crate::artifact::{HfConfig, ModelArtifact};
use crate::error::Result;
use crate::graph::ModelGraph;
use crate::model::{
    AttentionPattern, LayerKind, ModelArchitecture, NormPattern, RopeConfig,
};
use crate::model::llama::rope_from_hf_config;

/// Default: every 5th layer is global; others are sliding-window.
const DEFAULT_GLOBAL_EVERY_N: usize = 5;
/// Default sliding-window size for Gemma 4 (tokens).
const DEFAULT_WINDOW: usize = 1024;

#[derive(Debug, Clone, Copy)]
pub struct Gemma4Architecture;

impl ModelArchitecture for Gemma4Architecture {
    fn name(&self) -> &'static str {
        "gemma4"
    }

    fn build_graph(&self, artifact: &ModelArtifact) -> Result<ModelGraph> {
        ModelGraph::build_llama_style(artifact, self)
    }

    fn layer_kind(&self, layer_idx: usize, config: &HfConfig) -> LayerKind {
        if is_global_layer(layer_idx, config) {
            LayerKind::GlobalDecoder
        } else {
            let window = sliding_window(config);
            LayerKind::SlidingWindowDecoder { window }
        }
    }

    fn attention_pattern(&self, layer_idx: usize, config: &HfConfig) -> AttentionPattern {
        if is_global_layer(layer_idx, config) {
            AttentionPattern::FullCausal
        } else {
            AttentionPattern::SlidingWindow { size: sliding_window(config) }
        }
    }

    fn norm_pattern(&self) -> NormPattern {
        NormPattern::PrePost
    }

    fn lm_head_softcap(&self, config: &HfConfig) -> Option<f32> {
        config.final_logit_softcapping.filter(|&v| v > 0.0)
    }

    fn rope_config(&self, config: &HfConfig) -> RopeConfig {
        // Global layers use partial RoPE; local layers use full RoPE.
        // We return the global layer config here; the executor should check
        // per-layer whether it's global or local.
        let partial = config.partial_rotary_factor.unwrap_or(1.0);
        rope_from_hf_config(config, partial)
    }
}

fn is_global_layer(layer_idx: usize, config: &HfConfig) -> bool {
    let every_n = config
        .global_attn_every_n_layers
        .unwrap_or(DEFAULT_GLOBAL_EVERY_N);
    if every_n == 0 {
        return false;
    }
    // Gemma 4 convention: layers whose 0-based index is a multiple of every_n
    // use global attention. E.g. every_n=5 → layers 0,5,10,... are global.
    layer_idx.is_multiple_of(every_n)
}

fn sliding_window(config: &HfConfig) -> usize {
    config.sliding_window.unwrap_or(DEFAULT_WINDOW).max(1)
}


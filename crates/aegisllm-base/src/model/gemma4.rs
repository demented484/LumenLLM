//! Gemma 4 architecture family.
//!
//! Gemma 4 interleaves "sliding" (local) and "global" (full-causal) attention
//! layers. The exact pattern is given by the `layer_types` array in config.json;
//! the legacy `global_attn_every_n_layers` field is a fallback.
//!
//! - Pre+Post RMSNorm on every sublayer (4 norms per layer vs Llama's 2).
//! - Logit soft-capping on attention logits and lm_head output.
//! - Proportional RoPE (`partial_rotary_factor`) on global layers.
//! - Global layers use `global_head_dim` (512), sliding layers use `head_dim` (256).
//! - MatFormer nested-parameter checkpoints for E2B/E4B variants (Phase 9).

use crate::artifact::{HfConfig, ModelArtifact};
use crate::error::Result;
use crate::graph::ModelGraph;
use crate::model::{
    AttentionPattern, LayerKind, ModelArchitecture, NormPattern, RopeConfig,
};
use crate::model::llama::rope_from_hf_config;

/// Default sliding-window size (tokens) when `sliding_window` is absent.
const DEFAULT_WINDOW: usize = 1024;
/// Default: every 5th layer is global when `layer_types` is absent.
const DEFAULT_GLOBAL_EVERY_N: usize = 5;

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
        if is_moe_layer(layer_idx, config) {
            let num_experts = config.num_experts.unwrap_or(1);
            let top_k = config.num_experts_per_tok.unwrap_or(1);
            let has_shared = config.num_shared_experts.unwrap_or(0) > 0;
            LayerKind::MoEDecoder { num_experts, top_k, has_shared_expert: has_shared }
        } else if is_global_layer(layer_idx, config) {
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

    fn embed_scale(&self, config: &HfConfig) -> Option<f32> {
        // Gemma 4 ScaledWordEmbedding: out = embed_lookup * sqrt(hidden_size).
        Some((config.hidden_size as f32).sqrt())
    }

    fn lm_head_softcap(&self, config: &HfConfig) -> Option<f32> {
        config.final_logit_softcapping.filter(|&v| v > 0.0)
    }

    fn rope_config(&self, config: &HfConfig) -> RopeConfig {
        let partial = config.partial_rotary_factor.unwrap_or(1.0);
        rope_from_hf_config(config, partial)
    }
}

/// Returns true when layer `layer_idx` uses full (global) causal attention.
pub fn is_global_layer(layer_idx: usize, config: &HfConfig) -> bool {
    // Prefer the explicit per-layer type array when present.
    if let Some(types) = &config.layer_types {
        if let Some(t) = types.get(layer_idx) {
            return t == "full_attention";
        }
    }
    // Fallback: every N-th layer (0-based multiple) is global.
    let every_n = config.global_attn_every_n_layers.unwrap_or(DEFAULT_GLOBAL_EVERY_N);
    if every_n == 0 {
        return false;
    }
    layer_idx % every_n == 0
}

/// Returns true when layer `layer_idx` is a MoE block.
fn is_moe_layer(_layer_idx: usize, config: &HfConfig) -> bool {
    // Gemma 4 26B: enable_moe_block=true → ALL layers have MoE FFN.
    // Dense (non-MoE) Gemma 4 layers: enable_moe_block absent/false → false.
    config.enable_moe_block.unwrap_or(false) && config.num_experts.unwrap_or(0) > 1
}

/// Head dimension for a specific layer — global layers use `global_head_dim`.
pub fn head_dim_for_layer(layer_idx: usize, config: &HfConfig) -> Option<usize> {
    if is_global_layer(layer_idx, config) {
        config.global_head_dim.or(config.head_dim)
    } else {
        config.head_dim
    }
}

pub fn sliding_window(config: &HfConfig) -> usize {
    config.sliding_window.unwrap_or(DEFAULT_WINDOW).max(1)
}

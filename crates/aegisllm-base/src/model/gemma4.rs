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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::detect_architecture;

    /// Build an `HfConfig` matching google/gemma-4-31B's text config (dense
    /// variant: no PLE, no MoE, no layer_scalar). Verifies the architecture
    /// plumbing derives the right per-layer metadata WITHOUT the checkpoint.
    fn gemma4_31b_config() -> HfConfig {
        // 60 layers, 5 sliding : 1 full, full at i%6==5 (last layer 59 is full).
        let layer_types = (0..60)
            .map(|i| if i % 6 == 5 { "full_attention" } else { "sliding_attention" }.to_string())
            .collect::<Vec<_>>();
        HfConfig {
            architectures: Some(vec!["Gemma4ForConditionalGeneration".into()]),
            model_type: "gemma4_text".into(),
            hidden_size: 5376,
            intermediate_size: Some(21504),
            num_hidden_layers: 60,
            num_attention_heads: 32,
            num_key_value_heads: Some(16),
            num_global_key_value_heads: Some(4),
            head_dim: Some(256),
            global_head_dim: Some(512),
            sliding_window: Some(1024),
            layer_types: Some(layer_types),
            partial_rotary_factor: Some(0.25),
            final_logit_softcapping: Some(30.0),
            // Dense: PLE off, MoE off.
            hidden_size_per_layer_input: None,
            enable_moe_block: Some(false),
            num_experts: None,
            num_kv_shared_layers: Some(0),
            ..Default::default()
        }
    }

    #[test]
    fn gemma4_31b_dense_is_detected_as_gemma4() {
        let cfg = gemma4_31b_config();
        let arch = detect_architecture(&cfg).expect("Gemma4ForConditionalGeneration -> gemma4");
        assert_eq!(arch.name(), "gemma4");
    }

    #[test]
    fn gemma4_31b_per_layer_attention_geometry() {
        let cfg = gemma4_31b_config();
        let arch = Gemma4Architecture;
        for layer in 0..cfg.num_hidden_layers {
            let is_global = layer % 6 == 5;
            assert_eq!(is_global_layer(layer, &cfg), is_global, "layer {layer} global?");
            // head_dim: sliding 256, global 512.
            assert_eq!(
                head_dim_for_layer(layer, &cfg),
                Some(if is_global { 512 } else { 256 }),
                "layer {layer} head_dim"
            );
            // Per-layer KV head count derivation (mirrors graph::build_llama_style):
            // global layers use num_global_key_value_heads (4), sliding use 16.
            let base_kv = cfg.num_key_value_heads.unwrap();
            let layer_kv = if is_global_layer(layer, &cfg) {
                cfg.num_global_key_value_heads.unwrap_or(base_kv)
            } else {
                base_kv
            };
            assert_eq!(layer_kv, if is_global { 4 } else { 16 }, "layer {layer} kv heads");
            // Only global layers use partial RoPE; both are FullCausal vs SlidingWindow.
            match arch.attention_pattern(layer, &cfg) {
                AttentionPattern::FullCausal => assert!(is_global),
                AttentionPattern::SlidingWindow { size } => {
                    assert!(!is_global);
                    assert_eq!(size, 1024);
                }
                other => panic!("gemma4 layer {layer} unexpected pattern {other:?}"),
            }
        }
        // Last layer is global (Gemma-4 change vs Gemma 3).
        assert!(is_global_layer(59, &cfg));
    }

    #[test]
    fn gemma4_31b_dense_has_no_moe_or_ple() {
        let cfg = gemma4_31b_config();
        let arch = Gemma4Architecture;
        // Dense: every layer is a plain decoder (sliding or global), never MoE.
        for layer in 0..cfg.num_hidden_layers {
            assert!(
                !matches!(arch.layer_kind(layer, &cfg), LayerKind::MoEDecoder { .. }),
                "layer {layer} must not be MoE for the dense 31B"
            );
        }
        // Embedding scale = sqrt(hidden), lm_head softcap = 30.
        assert_eq!(arch.embed_scale(&cfg), Some((5376f32).sqrt()));
        assert_eq!(arch.lm_head_softcap(&cfg), Some(30.0));
    }
}

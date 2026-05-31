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

    fn layer_kind(&self, _layer_idx: usize, config: &HfConfig) -> LayerKind {
        // `layer_kind` describes the MLP sublayer (Dense vs MoE). The MIXER type
        // (full-attention vs Gated DeltaNet) is carried by `attention_pattern`,
        // because Qwen3-Next's GDN layers ALSO have an MoE/dense MLP — the two
        // axes are independent. Keeping them separate lets a GDN layer correctly
        // load MoE experts (35B) while the loader picks the mixer by tensor name.
        let num_experts = config.num_experts.unwrap_or(0);
        if num_experts > 1 {
            let top_k = config.num_experts_per_tok.unwrap_or(2);
            // Qwen3-Next sets only `shared_expert_intermediate_size` (no
            // `num_shared_experts` count), so detect a shared expert from either.
            let has_shared = config.num_shared_experts.unwrap_or(0) > 0
                || config.shared_expert_intermediate_size.is_some();
            return LayerKind::MoEDecoder {
                num_experts,
                top_k,
                has_shared_expert: has_shared,
            };
        }
        LayerKind::DenseDecoder
    }

    fn attention_pattern(&self, layer_idx: usize, config: &HfConfig) -> AttentionPattern {
        if is_linear_attention_layer(layer_idx, config) {
            AttentionPattern::LinearGatedDeltaNet
        } else {
            AttentionPattern::FullCausal
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

/// Static dimensions of a Gated DeltaNet (linear-attention) block, resolved
/// from the HF config. All GDN layers in a model share these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GdnDims {
    /// Number of key/query heads (q,k share this count). Qwen3-Next: 16.
    pub num_k_heads: usize,
    /// Number of value heads (v,z share this; GQA-expands q,k by num_v/num_k). 32.
    pub num_v_heads: usize,
    /// Per-head key/query dimension. 128.
    pub k_head_dim: usize,
    /// Per-head value dimension. 128.
    pub v_head_dim: usize,
    /// Depthwise causal conv1d kernel width. 4.
    pub conv_kernel: usize,
}

impl GdnDims {
    /// Recurrent state size per value head: `[k_head_dim, v_head_dim]` f32.
    pub fn state_elems_per_v_head(&self) -> usize {
        self.k_head_dim * self.v_head_dim
    }
    /// in_proj_qkvz output width: q,k = num_k*k_dim each; v,z = num_v*v_dim each.
    pub fn qkvz_proj_dim(&self) -> usize {
        2 * self.num_k_heads * self.k_head_dim + 2 * self.num_v_heads * self.v_head_dim
    }
    /// in_proj_ba output width: b,a = num_v each.
    pub fn ba_proj_dim(&self) -> usize {
        2 * self.num_v_heads
    }
    /// Channels carried through the depthwise conv1d: cat[q, k, v].
    pub fn conv_channels(&self) -> usize {
        self.num_k_heads * self.k_head_dim * 2 + self.num_v_heads * self.v_head_dim
    }
}

/// Reads the Gated DeltaNet dimensions, if the config describes a GDN model.
/// Returns `None` when the required `linear_*` fields are absent.
pub fn gdn_dims(config: &HfConfig) -> Option<GdnDims> {
    let num_k_heads = config.linear_num_key_heads?;
    let k_head_dim = config.linear_key_head_dim?;
    Some(GdnDims {
        num_k_heads,
        // value heads default to key heads when unspecified (no GQA expand).
        num_v_heads: config.linear_num_value_heads.unwrap_or(num_k_heads),
        k_head_dim,
        v_head_dim: config.linear_value_head_dim.unwrap_or(k_head_dim),
        conv_kernel: config.linear_conv_kernel_dim.unwrap_or(4),
    })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::HfConfig;

    /// Minimal config mirroring Qwen3.6-35B-A3B (MoE + GDN hybrid).
    fn qwen3_moe_config() -> HfConfig {
        HfConfig {
            model_type: "qwen3_5_moe".into(),
            hidden_size: 2048,
            num_hidden_layers: 40,
            num_experts: Some(256),
            num_experts_per_tok: Some(8),
            // The bug: count is absent, only the intermediate size is set.
            num_shared_experts: None,
            shared_expert_intermediate_size: Some(512),
            layer_types: Some(
                ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
            ),
            linear_num_key_heads: Some(16),
            linear_num_value_heads: Some(32),
            linear_key_head_dim: Some(128),
            linear_value_head_dim: Some(128),
            linear_conv_kernel_dim: Some(4),
            attn_output_gate: Some(true),
            ..Default::default()
        }
    }

    #[test]
    fn layer_types_drives_linear_vs_full() {
        let cfg = qwen3_moe_config();
        assert!(is_linear_attention_layer(0, &cfg));
        assert!(is_linear_attention_layer(2, &cfg));
        assert!(!is_linear_attention_layer(3, &cfg)); // full_attention
    }

    #[test]
    fn moe_layer_detects_shared_expert_from_intermediate_size() {
        let cfg = qwen3_moe_config();
        // Every layer is MoE in the 35B (mixer type is orthogonal). Layer 3 is
        // full_attention → MoEDecoder with shared expert + full-causal mixer.
        match Qwen3Architecture.layer_kind(3, &cfg) {
            LayerKind::MoEDecoder { num_experts, top_k, has_shared_expert } => {
                assert_eq!(num_experts, 256);
                assert_eq!(top_k, 8);
                assert!(has_shared_expert, "shared expert missed despite intermediate_size set");
            }
            other => panic!("expected MoEDecoder, got {other:?}"),
        }
        assert_eq!(Qwen3Architecture.attention_pattern(3, &cfg), AttentionPattern::FullCausal);
    }

    #[test]
    fn linear_layer_is_moe_with_gdn_mixer() {
        let cfg = qwen3_moe_config();
        // GDN layers are STILL MoE for the MLP (decoupled axes) but use the
        // Gated-DeltaNet mixer.
        assert!(matches!(
            Qwen3Architecture.layer_kind(0, &cfg),
            LayerKind::MoEDecoder { .. }
        ));
        assert_eq!(
            Qwen3Architecture.attention_pattern(0, &cfg),
            AttentionPattern::LinearGatedDeltaNet
        );
    }

    #[test]
    fn gdn_dims_reads_linear_fields() {
        let cfg = qwen3_moe_config();
        let d = gdn_dims(&cfg).expect("gdn dims");
        assert_eq!(d.num_k_heads, 16);
        assert_eq!(d.num_v_heads, 32);
        assert_eq!(d.k_head_dim, 128);
        assert_eq!(d.conv_kernel, 4);
        // q,k: 2*16*128 = 4096; v,z: 2*32*128 = 8192 → 12288.
        assert_eq!(d.qkvz_proj_dim(), 12288);
        assert_eq!(d.ba_proj_dim(), 64); // 2 * 32
        assert_eq!(d.conv_channels(), 16 * 128 * 2 + 32 * 128); // q+k+v
        assert_eq!(d.state_elems_per_v_head(), 128 * 128);
    }

    #[test]
    fn dense_qwen_has_no_moe_no_shared() {
        // Qwen3.5-9B: dense, still GDN hybrid, no experts.
        let cfg = HfConfig {
            model_type: "qwen3_5".into(),
            hidden_size: 4096,
            num_hidden_layers: 32,
            layer_types: Some(
                ["linear_attention", "full_attention"]
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
            ),
            linear_num_key_heads: Some(16),
            linear_key_head_dim: Some(128),
            ..Default::default()
        };
        // Dense 9B: no experts → DenseDecoder MLP on every layer; mixer type
        // still alternates via attention_pattern.
        assert!(matches!(
            Qwen3Architecture.layer_kind(0, &cfg),
            LayerKind::DenseDecoder
        ));
        assert_eq!(
            Qwen3Architecture.attention_pattern(0, &cfg),
            AttentionPattern::LinearGatedDeltaNet
        );
        assert_eq!(
            Qwen3Architecture.attention_pattern(1, &cfg),
            AttentionPattern::FullCausal
        );
        assert!(gdn_dims(&cfg).is_some());
    }
}

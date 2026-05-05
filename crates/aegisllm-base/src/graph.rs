use std::collections::BTreeMap;

use crate::artifact::ModelArtifact;
use crate::error::Result;
use crate::model::{
    detect_architecture, AttentionPattern, LayerKind, ModelArchitecture, NormPattern,
};
use crate::model::gemma4::{head_dim_for_layer, is_global_layer};
use crate::tensor::TensorInfo;
use crate::tensor::quant::WeightQuantization;

// ── Core types ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub struct ModelGraph {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: Option<usize>,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: Option<usize>,
    pub weight_quantization: WeightQuantization,
    pub regions: Vec<GraphRegion>,

    // ── Per-layer heterogeneous metadata ──────────────────────────────────
    /// One entry per transformer layer; drives per-layer dispatch decisions.
    pub layer_metadata: Vec<LayerMetadata>,
    /// Pre-only or pre+post RMSNorm style (Llama vs Gemma 4).
    pub norm_pattern: NormPattern,
    /// Logit soft-cap on lm_head output (`cap * tanh(logits / cap)`).
    pub lm_head_softcap: Option<f32>,
    /// Logit soft-cap applied inside attention before softmax (Gemma 4).
    pub attn_logit_softcap: Option<f32>,
    /// Multiplicative scale applied to token embeddings after lookup.
    /// Gemma 4 uses `sqrt(hidden_size)`. Other architectures: None (treated as 1.0).
    pub embed_scale: Option<f32>,
    /// Detected architecture name (for display / debugging).
    pub architecture: String,
    /// True when the graph was built from a MatFormer nested-param checkpoint
    /// with an `effective_size` that slices to a smaller sub-model (e.g. E2B).
    pub is_sliced: bool,
    /// Tensor-name prefix for the text decoder. `"model."` for Llama / Qwen3,
    /// `"model.language_model."` for HuggingFace multimodal wrappers
    /// (Qwen3.5-9B, Gemma 4, Nemotron Omni). Includes trailing dot.
    pub text_prefix: String,
}

/// Detect the text-decoder tensor-name prefix for an artifact. Returns
/// `"model.language_model."` when the embedding is wrapped under
/// `language_model` (multimodal HF layout), otherwise `"model."`.
pub fn detect_text_prefix(artifact: &ModelArtifact) -> String {
    if artifact.tensors.has("model.language_model.embed_tokens.weight") {
        "model.language_model.".to_string()
    } else {
        "model.".to_string()
    }
}

/// Per-layer dispatch information built during graph construction.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerMetadata {
    pub layer_idx: usize,
    pub kind: LayerKind,
    pub attention_pattern: AttentionPattern,
    /// Effective head_dim for this layer. Gemma 4 global layers use
    /// `global_head_dim` (512); sliding layers use `head_dim` (256).
    /// For all other architectures this equals `ModelGraph::head_dim`.
    pub head_dim: usize,
    /// Per-layer KV head count. Gemma 4 global layers use
    /// `num_global_key_value_heads` (2); sliding layers use `num_key_value_heads` (8).
    /// For all other architectures this equals `ModelGraph::num_kv_heads`.
    pub num_kv_heads: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraphRegion {
    pub id: RegionId,
    pub kind: GraphRegionKind,
    pub layer_index: Option<usize>,
    pub tensors: Vec<GraphTensor>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RegionId(pub String);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphRegionKind {
    TokenEmbedding,
    TransformerBlock,
    Attention,
    Mlp,
    FinalNorm,
    LmHead,
    KvCache,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GraphTensor {
    pub role: TensorRole,
    pub info: TensorInfo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorRole {
    TokenEmbedding,
    AttentionNorm,
    /// Post-sublayer norm (Gemma 4 PrePost pattern).
    PostAttentionNorm,
    /// Post-MLP norm (Gemma 4 PrePost pattern).
    PostMlpNorm,
    Query,
    Key,
    Value,
    Output,
    MlpNorm,
    /// Additional pre-FFN norm present in Gemma 4 (pre_feedforward_layernorm).
    /// In Llama/Qwen the single post_attention_layernorm serves as MlpNorm;
    /// Gemma 4 has both post_attention_layernorm (PostAttentionNorm) AND
    /// pre_feedforward_layernorm (MlpPreNorm).
    MlpPreNorm,
    Gate,
    Up,
    Down,
    FinalNorm,
    LmHead,
    /// Per-query-head RMS norm (Gemma 4).
    QNorm,
    /// Per-key-head RMS norm (Gemma 4).
    KNorm,
    WeightScale,
    InputScale,
    OutputScale,
    /// MoE router weight matrix [hidden, num_experts].
    MoeRouter,
    /// Shared / always-active expert weights.
    MoeSharedGate,
    MoeSharedUp,
    MoeSharedDown,
    /// Mamba / SSM parameters.
    SsmInProj,
    SsmOutProj,
    SsmA,
    SsmD,
    SsmDt,
    SsmConv1d,
    /// Generic catch-all.
    Other,
}

// ── ModelGraph construction ───────────────────────────────────────────────────

impl ModelGraph {
    /// Dispatch to the correct architecture and build the graph.
    pub fn from_artifact(artifact: &ModelArtifact) -> Result<Self> {
        let arch = detect_architecture(&artifact.config)?;
        arch.build_graph(artifact)
    }

    /// Shared "Llama-style" builder used by Llama, Qwen 3, Gemma 4, and Nemotron 3.
    /// All four families share the same HuggingFace tensor naming convention.
    pub fn build_llama_style(
        artifact: &ModelArtifact,
        arch: &dyn ModelArchitecture,
    ) -> Result<Self> {
        let config = &artifact.config;
        let text_prefix = detect_text_prefix(artifact);
        let mut regions = Vec::new();

        // Embedding — try the detected text prefix first, then bare fallbacks.
        let embed_candidate = format!("{text_prefix}embed_tokens.weight");
        push_single_tensor_region(
            &mut regions,
            artifact,
            "embed",
            GraphRegionKind::TokenEmbedding,
            None,
            TensorRole::TokenEmbedding,
            &[embed_candidate.as_str(), "model.embed_tokens.weight"],
        );

        let mut layer_metadata = Vec::with_capacity(config.num_hidden_layers);

        let is_gemma4 = arch.norm_pattern() == NormPattern::PrePost;
        let num_experts = config.num_experts.unwrap_or(0);

        for layer in 0..config.num_hidden_layers {
            let prefix = format!("{text_prefix}layers.{layer}");
            let mut tensors = Vec::new();

            // ── Pre-attention norm ────────────────────────────────────────
            add_known_tensor(&mut tensors, artifact, TensorRole::AttentionNorm,
                &format!("{prefix}.input_layernorm.weight"));

            // ── QKV projections + QK norms (Gemma 4) ─────────────────────
            add_known_tensor(&mut tensors, artifact, TensorRole::Query,
                &format!("{prefix}.self_attn.q_proj.weight"));
            add_quant_aux_tensors(&mut tensors, artifact, &format!("{prefix}.self_attn.q_proj"));
            add_known_tensor(&mut tensors, artifact, TensorRole::QNorm,
                &format!("{prefix}.self_attn.q_norm.weight"));

            add_known_tensor(&mut tensors, artifact, TensorRole::Key,
                &format!("{prefix}.self_attn.k_proj.weight"));
            add_quant_aux_tensors(&mut tensors, artifact, &format!("{prefix}.self_attn.k_proj"));
            add_known_tensor(&mut tensors, artifact, TensorRole::KNorm,
                &format!("{prefix}.self_attn.k_norm.weight"));

            add_known_tensor(&mut tensors, artifact, TensorRole::Value,
                &format!("{prefix}.self_attn.v_proj.weight"));
            add_quant_aux_tensors(&mut tensors, artifact, &format!("{prefix}.self_attn.v_proj"));

            add_known_tensor(&mut tensors, artifact, TensorRole::Output,
                &format!("{prefix}.self_attn.o_proj.weight"));
            add_quant_aux_tensors(&mut tensors, artifact, &format!("{prefix}.self_attn.o_proj"));

            // ── Post-attention norm ───────────────────────────────────────
            // Gemma 4: post_attention_layernorm = post-attn (PostAttentionNorm)
            // Llama/Qwen: post_attention_layernorm = pre-MLP (MlpNorm)
            if is_gemma4 {
                add_known_tensor(&mut tensors, artifact, TensorRole::PostAttentionNorm,
                    &format!("{prefix}.post_attention_layernorm.weight"));
                // Gemma 4 pre-MLP norm is a separate tensor
                add_known_tensor(&mut tensors, artifact, TensorRole::MlpNorm,
                    &format!("{prefix}.pre_feedforward_layernorm.weight"));
            } else {
                // Llama/Qwen: the single norm after attention serves as MLP norm
                add_known_tensor(&mut tensors, artifact, TensorRole::MlpNorm,
                    &format!("{prefix}.post_attention_layernorm.weight"));
            }

            // Dense MLP (present on non-MoE layers and as shared expert)
            add_known_tensor(&mut tensors, artifact, TensorRole::Gate,
                &format!("{prefix}.mlp.gate_proj.weight"));
            add_quant_aux_tensors(&mut tensors, artifact, &format!("{prefix}.mlp.gate_proj"));

            add_known_tensor(&mut tensors, artifact, TensorRole::Up,
                &format!("{prefix}.mlp.up_proj.weight"));
            add_quant_aux_tensors(&mut tensors, artifact, &format!("{prefix}.mlp.up_proj"));

            add_known_tensor(&mut tensors, artifact, TensorRole::Down,
                &format!("{prefix}.mlp.down_proj.weight"));
            add_quant_aux_tensors(&mut tensors, artifact, &format!("{prefix}.mlp.down_proj"));

            // ── Post-MLP norm (Gemma 4 PrePost) ──────────────────────────
            add_known_tensor(&mut tensors, artifact, TensorRole::PostMlpNorm,
                &format!("{prefix}.post_mlp_norm.weight"));
            add_known_tensor(&mut tensors, artifact, TensorRole::PostMlpNorm,
                &format!("{prefix}.post_feedforward_layernorm.weight"));

            // ── MoE router (Gemma 4: router.proj, Qwen: mlp.router_logits) ─
            add_known_tensor(&mut tensors, artifact, TensorRole::MoeRouter,
                &format!("{prefix}.router.proj.weight"));
            add_known_tensor(&mut tensors, artifact, TensorRole::MoeRouter,
                &format!("{prefix}.mlp.router_logits.weight"));
            add_known_tensor(&mut tensors, artifact, TensorRole::MoeRouter,
                &format!("{prefix}.block_sparse_moe.gate.weight"));

            // ── Per-expert tensors (Gemma 4 MoE: experts.N.{gate|up|down}_proj) ─
            // We record individual expert regions as Other tensors so the loader
            // can access them; the executor stacks them at load time.
            if num_experts > 1 {
                for expert_idx in 0..num_experts {
                    let ep = format!("{prefix}.experts.{expert_idx}");
                    for (proj, role) in [
                        ("gate_proj", TensorRole::Gate),
                        ("up_proj", TensorRole::Up),
                        ("down_proj", TensorRole::Down),
                    ] {
                        add_known_tensor(&mut tensors, artifact, role,
                            &format!("{ep}.{proj}.weight"));
                        add_quant_aux_tensors(&mut tensors, artifact, &format!("{ep}.{proj}"));
                    }
                }
            }

            // ── Shared expert (Nemotron 3 / Qwen 3.x) ────────────────────
            add_known_tensor(&mut tensors, artifact, TensorRole::MoeSharedGate,
                &format!("{prefix}.mlp.shared_expert.gate_proj.weight"));
            add_known_tensor(&mut tensors, artifact, TensorRole::MoeSharedUp,
                &format!("{prefix}.mlp.shared_expert.up_proj.weight"));
            add_known_tensor(&mut tensors, artifact, TensorRole::MoeSharedDown,
                &format!("{prefix}.mlp.shared_expert.down_proj.weight"));

            // ── SSM / Mamba tensors (Nemotron 3) ─────────────────────────
            add_known_tensor(&mut tensors, artifact, TensorRole::SsmInProj,
                &format!("{prefix}.mamba.in_proj.weight"));
            add_known_tensor(&mut tensors, artifact, TensorRole::SsmOutProj,
                &format!("{prefix}.mamba.out_proj.weight"));
            add_known_tensor(&mut tensors, artifact, TensorRole::SsmA,
                &format!("{prefix}.mamba.A_log"));
            add_known_tensor(&mut tensors, artifact, TensorRole::SsmD,
                &format!("{prefix}.mamba.D"));
            add_known_tensor(&mut tensors, artifact, TensorRole::SsmDt,
                &format!("{prefix}.mamba.dt_bias"));
            add_known_tensor(&mut tensors, artifact, TensorRole::SsmConv1d,
                &format!("{prefix}.mamba.conv1d.weight"));

            // Per-layer head_dim (Gemma 4 global layers use global_head_dim).
            let layer_head_dim = if is_gemma4 {
                head_dim_for_layer(layer, config)
                    .unwrap_or_else(|| artifact.head_dim())
            } else {
                artifact.head_dim()
            };
            // Per-layer KV head count (Gemma 4 global layers use num_global_key_value_heads).
            let base_kv_heads = config.num_key_value_heads.unwrap_or_else(|| {
                config.num_attention_heads
            });
            let layer_num_kv_heads = if is_gemma4 && is_global_layer(layer, config) {
                config.num_global_key_value_heads.unwrap_or(base_kv_heads)
            } else {
                base_kv_heads
            };

            regions.push(GraphRegion {
                id: RegionId(format!("layer.{layer}")),
                kind: GraphRegionKind::TransformerBlock,
                layer_index: Some(layer),
                tensors,
            });

            layer_metadata.push(LayerMetadata {
                layer_idx: layer,
                kind: arch.layer_kind(layer, config),
                attention_pattern: arch.attention_pattern(layer, config),
                head_dim: layer_head_dim,
                num_kv_heads: layer_num_kv_heads,
            });
        }

        // Final norm
        let final_norm_candidate = format!("{text_prefix}norm.weight");
        push_single_tensor_region(
            &mut regions,
            artifact,
            "final_norm",
            GraphRegionKind::FinalNorm,
            None,
            TensorRole::FinalNorm,
            &[final_norm_candidate.as_str(), "model.norm.weight"],
        );

        // LM head (may be tied to embeddings).
        let embed_candidate_lmhead = format!("{text_prefix}embed_tokens.weight");
        push_single_tensor_region(
            &mut regions,
            artifact,
            "lm_head",
            GraphRegionKind::LmHead,
            None,
            TensorRole::LmHead,
            &[
                "lm_head.weight",
                embed_candidate_lmhead.as_str(),
                "model.embed_tokens.weight",
            ],
        );

        let eff = config.effective_dims();
        // Use the base (sliding) head_dim for the graph-level field.
        // Per-layer head_dim is stored in LayerMetadata.head_dim for Gemma 4.
        let head_dim = artifact.head_dim();
        let orig_q = config.num_attention_heads;
        let orig_kv = config.num_key_value_heads.unwrap_or(orig_q);
        let (num_attention_heads, num_kv_heads) = if eff.is_sliced && head_dim > 0 {
            let eff_q = eff.hidden_size / head_dim;
            let eff_kv = (eff_q * orig_kv / orig_q).max(1);
            (eff_q, eff_kv)
        } else {
            (orig_q, orig_kv)
        };

        Ok(Self {
            model_type: config.model_type.clone(),
            architecture: arch.name().to_string(),
            hidden_size: eff.hidden_size,
            intermediate_size: eff.intermediate_size,
            num_layers: config.num_hidden_layers,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            vocab_size: config.vocab_size,
            weight_quantization: artifact.infer_weight_quantization(),
            regions,
            layer_metadata,
            norm_pattern: arch.norm_pattern(),
            lm_head_softcap: arch.lm_head_softcap(config),
            attn_logit_softcap: config.attn_logit_softcapping.filter(|&v| v > 0.0),
            embed_scale: arch.embed_scale(config),
            is_sliced: eff.is_sliced,
            text_prefix,
        })
    }

    pub fn total_weight_bytes(&self) -> u64 {
        self.regions.iter().map(GraphRegion::weight_bytes).sum()
    }

    pub fn regions_by_id(&self) -> BTreeMap<&RegionId, &GraphRegion> {
        self.regions
            .iter()
            .map(|region| (&region.id, region))
            .collect()
    }

    /// Returns the `LayerMetadata` for a given layer index (None if out of range).
    pub fn layer(&self, idx: usize) -> Option<&LayerMetadata> {
        self.layer_metadata.get(idx)
    }
}

impl GraphRegion {
    pub fn weight_bytes(&self) -> u64 {
        self.tensors
            .iter()
            .map(|tensor| tensor.info.data_len_bytes())
            .sum()
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

pub fn push_single_tensor_region(
    regions: &mut Vec<GraphRegion>,
    artifact: &ModelArtifact,
    id: &str,
    kind: GraphRegionKind,
    layer_index: Option<usize>,
    role: TensorRole,
    names: &[&str],
) {
    let tensors = names
        .iter()
        .find_map(|name| artifact.tensors.get(name).cloned())
        .map(|info| vec![GraphTensor { role, info }]);
    if let Some(tensors) = tensors {
        regions.push(GraphRegion {
            id: RegionId(id.to_string()),
            kind,
            layer_index,
            tensors,
        });
    }
}

pub fn add_known_tensor(
    tensors: &mut Vec<GraphTensor>,
    artifact: &ModelArtifact,
    role: TensorRole,
    name: &str,
) {
    if let Some(info) = artifact.tensors.get(name) {
        tensors.push(GraphTensor {
            role,
            info: info.clone(),
        });
    }
}

pub fn add_quant_aux_tensors(
    tensors: &mut Vec<GraphTensor>,
    artifact: &ModelArtifact,
    prefix: &str,
) {
    add_known_tensor(tensors, artifact, TensorRole::WeightScale,
        &format!("{prefix}.weight_scale"));
    add_known_tensor(tensors, artifact, TensorRole::OutputScale,
        &format!("{prefix}.weight_scale_2"));
    add_known_tensor(tensors, artifact, TensorRole::InputScale,
        &format!("{prefix}.input_scale"));
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_weight_bytes_are_summed() {
        let region = GraphRegion {
            id: RegionId("x".into()),
            kind: GraphRegionKind::LmHead,
            layer_index: None,
            tensors: vec![GraphTensor {
                role: TensorRole::LmHead,
                info: TensorInfo {
                    name: "x".into(),
                    dtype: crate::tensor::TensorDType::F32,
                    shape: vec![2],
                    num_elements: 2,
                    data_offsets: (0, 8),
                    file_offsets: (10, 18),
                    shard_name: "s".into(),
                    shard_path: "s".into(),
                },
            }],
        };
        assert_eq!(region.weight_bytes(), 8);
    }

    #[test]
    fn layer_kind_debug_is_deterministic() {
        let k = LayerKind::DenseDecoder;
        assert_eq!(format!("{k:?}"), "DenseDecoder");

        let m = LayerKind::MoEDecoder { num_experts: 128, top_k: 2, has_shared_expert: false };
        assert!(format!("{m:?}").contains("MoEDecoder"));
    }

    #[test]
    fn effective_head_count_math_e2b() {
        // Simulate the head-count computation for a Gemma 4 E2B slice:
        // full: hidden=4096, 16 q-heads, 8 kv-heads, head_dim=256
        // e2b:  eff_hidden=2048 → eff_q=2048/256=8, eff_kv=8*8/16=4
        let orig_q: usize = 16;
        let orig_kv: usize = 8;
        let head_dim: usize = 256;
        let eff_hidden: usize = 2048;
        let eff_q = eff_hidden / head_dim;
        let eff_kv = (eff_q * orig_kv / orig_q).max(1);
        assert_eq!(eff_q, 8);
        assert_eq!(eff_kv, 4);
        // ratio preserved: q/kv = 2:1 in both full and sliced
        assert_eq!(eff_q / eff_kv, orig_q / orig_kv);
    }
}

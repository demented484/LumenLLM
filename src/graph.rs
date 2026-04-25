use std::collections::BTreeMap;

use crate::artifact::ModelArtifact;
use crate::error::{AegisError, Result};
use crate::tensor::TensorInfo;
use crate::tensor::quant::WeightQuantization;

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
    Query,
    Key,
    Value,
    Output,
    MlpNorm,
    Gate,
    Up,
    Down,
    FinalNorm,
    LmHead,
    WeightScale,
    InputScale,
    OutputScale,
    Other,
}

impl ModelGraph {
    pub fn from_artifact(artifact: &ModelArtifact) -> Result<Self> {
        if !artifact
            .config
            .model_type
            .to_ascii_lowercase()
            .contains("llama")
        {
            return Err(AegisError::Unsupported(format!(
                "graph builder currently understands llama-style tensor names, got `{}`",
                artifact.config.model_type
            )));
        }

        let mut regions = Vec::new();
        push_single_tensor_region(
            &mut regions,
            artifact,
            "embed",
            GraphRegionKind::TokenEmbedding,
            None,
            TensorRole::TokenEmbedding,
            &["model.embed_tokens.weight"],
        );

        for layer in 0..artifact.config.num_hidden_layers {
            let prefix = format!("model.layers.{layer}");
            let mut tensors = Vec::new();
            add_known_tensor(
                &mut tensors,
                artifact,
                TensorRole::AttentionNorm,
                &format!("{prefix}.input_layernorm.weight"),
            );
            add_known_tensor(
                &mut tensors,
                artifact,
                TensorRole::Query,
                &format!("{prefix}.self_attn.q_proj.weight"),
            );
            add_quant_aux_tensors(
                &mut tensors,
                artifact,
                &format!("{prefix}.self_attn.q_proj"),
            );
            add_known_tensor(
                &mut tensors,
                artifact,
                TensorRole::Key,
                &format!("{prefix}.self_attn.k_proj.weight"),
            );
            add_quant_aux_tensors(
                &mut tensors,
                artifact,
                &format!("{prefix}.self_attn.k_proj"),
            );
            add_known_tensor(
                &mut tensors,
                artifact,
                TensorRole::Value,
                &format!("{prefix}.self_attn.v_proj.weight"),
            );
            add_quant_aux_tensors(
                &mut tensors,
                artifact,
                &format!("{prefix}.self_attn.v_proj"),
            );
            add_known_tensor(
                &mut tensors,
                artifact,
                TensorRole::Output,
                &format!("{prefix}.self_attn.o_proj.weight"),
            );
            add_quant_aux_tensors(
                &mut tensors,
                artifact,
                &format!("{prefix}.self_attn.o_proj"),
            );
            add_known_tensor(
                &mut tensors,
                artifact,
                TensorRole::MlpNorm,
                &format!("{prefix}.post_attention_layernorm.weight"),
            );
            add_known_tensor(
                &mut tensors,
                artifact,
                TensorRole::Gate,
                &format!("{prefix}.mlp.gate_proj.weight"),
            );
            add_quant_aux_tensors(&mut tensors, artifact, &format!("{prefix}.mlp.gate_proj"));
            add_known_tensor(
                &mut tensors,
                artifact,
                TensorRole::Up,
                &format!("{prefix}.mlp.up_proj.weight"),
            );
            add_quant_aux_tensors(&mut tensors, artifact, &format!("{prefix}.mlp.up_proj"));
            add_known_tensor(
                &mut tensors,
                artifact,
                TensorRole::Down,
                &format!("{prefix}.mlp.down_proj.weight"),
            );
            add_quant_aux_tensors(&mut tensors, artifact, &format!("{prefix}.mlp.down_proj"));
            if tensors.is_empty() {
                return Err(AegisError::Unsupported(format!(
                    "no tensors found for llama layer {layer}"
                )));
            }
            regions.push(GraphRegion {
                id: RegionId(format!("layer.{layer}")),
                kind: GraphRegionKind::TransformerBlock,
                layer_index: Some(layer),
                tensors,
            });
        }

        push_single_tensor_region(
            &mut regions,
            artifact,
            "final_norm",
            GraphRegionKind::FinalNorm,
            None,
            TensorRole::FinalNorm,
            &["model.norm.weight"],
        );
        push_single_tensor_region(
            &mut regions,
            artifact,
            "lm_head",
            GraphRegionKind::LmHead,
            None,
            TensorRole::LmHead,
            &["lm_head.weight", "model.embed_tokens.weight"],
        );

        Ok(Self {
            model_type: artifact.config.model_type.clone(),
            hidden_size: artifact.config.hidden_size,
            intermediate_size: artifact.config.intermediate_size,
            num_layers: artifact.config.num_hidden_layers,
            num_attention_heads: artifact.config.num_attention_heads,
            num_kv_heads: artifact
                .config
                .num_key_value_heads
                .unwrap_or(artifact.config.num_attention_heads),
            head_dim: artifact.head_dim(),
            vocab_size: artifact.config.vocab_size,
            weight_quantization: artifact.infer_weight_quantization(),
            regions,
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
}

impl GraphRegion {
    pub fn weight_bytes(&self) -> u64 {
        self.tensors
            .iter()
            .map(|tensor| tensor.info.data_len_bytes())
            .sum()
    }
}

fn push_single_tensor_region(
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

fn add_known_tensor(
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

fn add_quant_aux_tensors(tensors: &mut Vec<GraphTensor>, artifact: &ModelArtifact, prefix: &str) {
    add_known_tensor(
        tensors,
        artifact,
        TensorRole::WeightScale,
        &format!("{prefix}.weight_scale"),
    );
    add_known_tensor(
        tensors,
        artifact,
        TensorRole::OutputScale,
        &format!("{prefix}.weight_scale_2"),
    );
    add_known_tensor(
        tensors,
        artifact,
        TensorRole::InputScale,
        &format!("{prefix}.input_scale"),
    );
}

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
}

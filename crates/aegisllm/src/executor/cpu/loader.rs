use std::collections::BTreeMap;

use super::rope::RopeConfig;
use super::state::{CpuLayer, CpuLlamaExecutor};
use crate::artifact::ModelArtifact;
use crate::error::{AegisError, Result};
use crate::executor::tensors::{Bf16Matrix, read_dense_vector, require_tensor};
use crate::graph::{GraphRegionKind, ModelGraph, RegionId};
use crate::planning::materialization::LinearMaterializationCache;
use crate::planning::placement::{ResolvedPlacement, StoragePlacement};
use crate::planning::runtime::RuntimePlan;
use crate::tensor::layout::{LinearResidentLayout, MaterializationPolicy};
use crate::tensor::storage::{TensorResidencyPlan, TensorStorageLoader};

impl CpuLlamaExecutor {
    pub(super) fn from_artifact(
        artifact: &ModelArtifact,
        graph: &ModelGraph,
        placement: &ResolvedPlacement,
        runtime: &RuntimePlan,
    ) -> Result<Self> {
        let region_placements = placement.region_map();
        let runtime_layouts = runtime_linear_layouts_by_region(runtime);
        let mut loader = TensorStorageLoader::new();

        let embed_region = region_placements
            .get(&RegionId("embed".into()))
            .ok_or_else(|| AegisError::InvalidPlan("missing embed placement".into()))?;
        let final_norm_region = region_placements
            .get(&RegionId("final_norm".into()))
            .ok_or_else(|| AegisError::InvalidPlan("missing final_norm placement".into()))?;
        let lm_head_region = region_placements
            .get(&RegionId("lm_head".into()))
            .ok_or_else(|| AegisError::InvalidPlan("missing lm_head placement".into()))?;

        let embed_tokens = Bf16Matrix::from_named_tensor(
            artifact,
            "model.embed_tokens.weight",
            embed_region.store,
            &mut loader,
        )?;
        let final_norm = read_dense_vector(
            artifact
                .tensors
                .get("model.norm.weight")
                .ok_or_else(|| AegisError::InvalidPlan("missing `model.norm.weight`".into()))?,
            final_norm_region.store,
            &mut loader,
        )?;
        let lm_head = Bf16Matrix::from_first_existing_tensor(
            artifact,
            &["lm_head.weight", "model.embed_tokens.weight"],
            lm_head_region.store,
            &mut loader,
        )?;

        let mut materialization = LinearMaterializationCache::new();
        let mut layers = Vec::with_capacity(graph.num_layers);
        for layer in 0..graph.num_layers {
            let region_id = RegionId(format!("layer.{layer}"));
            let region = graph
                .regions
                .iter()
                .find(|region| region.id == region_id)
                .ok_or_else(|| {
                    AegisError::InvalidPlan(format!("missing graph region `{}`", region_id.0))
                })?;
            if region.kind != GraphRegionKind::TransformerBlock {
                return Err(AegisError::InvalidPlan(format!(
                    "region `{}` is not a transformer block",
                    region_id.0
                )));
            }
            let placement = region_placements.get(&region_id).ok_or_else(|| {
                AegisError::InvalidPlan(format!("missing placement for `{}`", region_id.0))
            })?;
            let runtime_layout = runtime_layouts
                .get(&region_id.0)
                .copied()
                .unwrap_or_default();
            let prefix = format!("model.layers.{layer}");
            layers.push(CpuLayer {
                input_norm_weight: read_dense_vector(
                    require_tensor(artifact, &format!("{prefix}.input_layernorm.weight"))?,
                    placement.store,
                    &mut loader,
                )?,
                post_attention_norm_weight: read_dense_vector(
                    require_tensor(
                        artifact,
                        &format!("{prefix}.post_attention_layernorm.weight"),
                    )?,
                    placement.store,
                    &mut loader,
                )?,
                q_proj: materialization.load_cpu_nvfp4_linear(
                    artifact,
                    &format!("{prefix}.self_attn.q_proj"),
                    placement.store,
                    cpu_residency_for_store(placement.store)?,
                    runtime_layout.resident_layout,
                    runtime_layout.materialization,
                )?,
                k_proj: materialization.load_cpu_nvfp4_linear(
                    artifact,
                    &format!("{prefix}.self_attn.k_proj"),
                    placement.store,
                    cpu_residency_for_store(placement.store)?,
                    runtime_layout.resident_layout,
                    runtime_layout.materialization,
                )?,
                v_proj: materialization.load_cpu_nvfp4_linear(
                    artifact,
                    &format!("{prefix}.self_attn.v_proj"),
                    placement.store,
                    cpu_residency_for_store(placement.store)?,
                    runtime_layout.resident_layout,
                    runtime_layout.materialization,
                )?,
                o_proj: materialization.load_cpu_nvfp4_linear(
                    artifact,
                    &format!("{prefix}.self_attn.o_proj"),
                    placement.store,
                    cpu_residency_for_store(placement.store)?,
                    runtime_layout.resident_layout,
                    runtime_layout.materialization,
                )?,
                gate_proj: materialization.load_cpu_nvfp4_linear(
                    artifact,
                    &format!("{prefix}.mlp.gate_proj"),
                    placement.store,
                    cpu_residency_for_store(placement.store)?,
                    runtime_layout.resident_layout,
                    runtime_layout.materialization,
                )?,
                up_proj: materialization.load_cpu_nvfp4_linear(
                    artifact,
                    &format!("{prefix}.mlp.up_proj"),
                    placement.store,
                    cpu_residency_for_store(placement.store)?,
                    runtime_layout.resident_layout,
                    runtime_layout.materialization,
                )?,
                down_proj: materialization.load_cpu_nvfp4_linear(
                    artifact,
                    &format!("{prefix}.mlp.down_proj"),
                    placement.store,
                    cpu_residency_for_store(placement.store)?,
                    runtime_layout.resident_layout,
                    runtime_layout.materialization,
                )?,
            });
        }

        Ok(Self {
            hidden_size: graph.hidden_size,
            num_attention_heads: graph.num_attention_heads,
            num_kv_heads: graph.num_kv_heads,
            head_dim: graph.head_dim,
            rms_norm_eps: artifact.config.rms_norm_eps.unwrap_or(1e-5) as f32,
            rope: RopeConfig::from_artifact(artifact),
            embed_tokens,
            final_norm,
            lm_head,
            layers,
            kv_context_size: placement.kv_cache.context_size,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct RuntimeLinearLayout {
    resident_layout: LinearResidentLayout,
    materialization: MaterializationPolicy,
}

impl Default for RuntimeLinearLayout {
    fn default() -> Self {
        Self {
            resident_layout: LinearResidentLayout::PackedSource,
            materialization: MaterializationPolicy::Lazy,
        }
    }
}

fn runtime_linear_layouts_by_region(
    runtime: &RuntimePlan,
) -> BTreeMap<String, RuntimeLinearLayout> {
    runtime
        .kernels
        .iter()
        .map(|kernel| {
            (
                kernel.name.clone(),
                RuntimeLinearLayout {
                    resident_layout: kernel.linear_layout.resident_layout,
                    materialization: kernel.linear_layout.materialization,
                },
            )
        })
        .collect()
}

fn cpu_residency_for_store(store: StoragePlacement) -> Result<TensorResidencyPlan> {
    match store {
        StoragePlacement::Ram => Ok(TensorResidencyPlan::RamResident),
        StoragePlacement::Mmap => Ok(TensorResidencyPlan::FileBackedMmap),
        StoragePlacement::Vram { device } => Ok(TensorResidencyPlan::StagedDeviceToHost { device }),
    }
}

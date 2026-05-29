use std::collections::{BTreeMap, BTreeSet};

use super::attention::attention_into;
use super::math::{add_into, rms_norm_into, swiglu_into};
use super::rope::{RopeConfig, apply_rope_in_place};
use super::state::{CpuLayer, CpuLlamaState, CpuScratch};
use aegisllm_base::artifact::ModelArtifact;
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::executor::tensors::{Bf16Matrix, read_dense_vector, require_tensor};
use aegisllm_base::graph::{GraphRegionKind, ModelGraph, RegionId};
use crate::materialization::LinearMaterializationCache;
use aegisllm_base::planning::placement::{ComputePlacement, ResolvedPlacement, StoragePlacement};
use aegisllm_base::planning::runtime::RuntimePlan;
use aegisllm_base::tensor::layout::{LinearResidentLayout, MaterializationPolicy};
use aegisllm_base::tensor::storage::TensorStorageLoader;

#[derive(Debug)]
pub struct CpuLayerBlockExecutor {
    hidden_size: usize,
    num_attention_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rms_norm_eps: f32,
    rope: RopeConfig,
    embed_tokens: Bf16Matrix,
    final_norm: Vec<f32>,
    lm_head: Bf16Matrix,
    layers: BTreeMap<usize, CpuLayer>,
    kv_context_size: usize,
}

impl CpuLayerBlockExecutor {
    pub fn from_artifact(
        artifact: &ModelArtifact,
        graph: &ModelGraph,
        placement: &ResolvedPlacement,
        runtime: &RuntimePlan,
        selected_layers: &BTreeSet<usize>,
    ) -> Result<Self> {
        // The hybrid CPU-layer block path is Llama-style only: it does NOT
        // implement the Gemma-4 PrePost norms, per-head q/k/v norm, partial
        // RoPE, per-layer head_dim/kv, PLE, or MoE. Running a Gemma-4 model
        // through it would silently miscompute, so reject it loudly. The
        // pure-CPU Gemma-4 forward (compute=cpu for ALL regions) is handled by
        // `G4CpuExecutor`; a hybrid Gemma-4 per-layer path is a follow-up.
        if aegisllm_base::model::detect_architecture(&artifact.config)
            .map(|arch| arch.name() == "gemma4")
            .unwrap_or(false)
        {
            return Err(AegisError::Unsupported(
                "hybrid CPU-layer execution is not yet implemented for Gemma-4; \
                 run Gemma-4 fully on CPU (all regions compute=cpu) or fully on GPU"
                    .into(),
            ));
        }
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
            host_store_for_cpu(embed_region.store),
            &mut loader,
        )?;
        let final_norm = read_dense_vector(
            require_tensor(artifact, "model.norm.weight")?,
            host_store_for_cpu(final_norm_region.store),
            &mut loader,
        )?;
        let lm_head = Bf16Matrix::from_first_existing_tensor(
            artifact,
            &["lm_head.weight", "model.embed_tokens.weight"],
            host_store_for_cpu(lm_head_region.store),
            &mut loader,
        )?;

        let mut materialization = LinearMaterializationCache::new();
        let mut layers = BTreeMap::new();
        for &layer in selected_layers {
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
            if placement.compute != ComputePlacement::Cpu {
                return Err(AegisError::InvalidPlan(format!(
                    "selected CPU hybrid layer `{}` has compute={}",
                    region_id.0, placement.compute
                )));
            }
            let runtime_layout = runtime_layouts
                .get(&region_id.0)
                .copied()
                .unwrap_or_default();
            let store = host_store_for_cpu(placement.store);
            let prefix = format!("model.layers.{layer}");
            layers.insert(
                layer,
                CpuLayer {
                    input_norm_weight: read_dense_vector(
                        require_tensor(artifact, &format!("{prefix}.input_layernorm.weight"))?,
                        store,
                        &mut loader,
                    )?,
                    post_attention_norm_weight: read_dense_vector(
                        require_tensor(
                            artifact,
                            &format!("{prefix}.post_attention_layernorm.weight"),
                        )?,
                        store,
                        &mut loader,
                    )?,
                    q_proj: materialization.load_cpu_nvfp4_linear(
                        artifact,
                        &format!("{prefix}.self_attn.q_proj"),
                        store,
                        cpu_residency_for_store(store),
                        runtime_layout.resident_layout,
                        runtime_layout.materialization,
                    )?,
                    k_proj: materialization.load_cpu_nvfp4_linear(
                        artifact,
                        &format!("{prefix}.self_attn.k_proj"),
                        store,
                        cpu_residency_for_store(store),
                        runtime_layout.resident_layout,
                        runtime_layout.materialization,
                    )?,
                    v_proj: materialization.load_cpu_nvfp4_linear(
                        artifact,
                        &format!("{prefix}.self_attn.v_proj"),
                        store,
                        cpu_residency_for_store(store),
                        runtime_layout.resident_layout,
                        runtime_layout.materialization,
                    )?,
                    o_proj: materialization.load_cpu_nvfp4_linear(
                        artifact,
                        &format!("{prefix}.self_attn.o_proj"),
                        store,
                        cpu_residency_for_store(store),
                        runtime_layout.resident_layout,
                        runtime_layout.materialization,
                    )?,
                    gate_proj: materialization.load_cpu_nvfp4_linear(
                        artifact,
                        &format!("{prefix}.mlp.gate_proj"),
                        store,
                        cpu_residency_for_store(store),
                        runtime_layout.resident_layout,
                        runtime_layout.materialization,
                    )?,
                    up_proj: materialization.load_cpu_nvfp4_linear(
                        artifact,
                        &format!("{prefix}.mlp.up_proj"),
                        store,
                        cpu_residency_for_store(store),
                        runtime_layout.resident_layout,
                        runtime_layout.materialization,
                    )?,
                    down_proj: materialization.load_cpu_nvfp4_linear(
                        artifact,
                        &format!("{prefix}.mlp.down_proj"),
                        store,
                        cpu_residency_for_store(store),
                        runtime_layout.resident_layout,
                        runtime_layout.materialization,
                    )?,
                },
            );
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

    pub fn new_state(&self) -> CpuLlamaState {
        let kv_width = self.num_kv_heads * self.head_dim;
        CpuLlamaState {
            position: 0,
            layers: (0..self.layers.keys().next_back().copied().unwrap_or(0) + 1)
                .map(|_| super::state::CpuLayerState {
                    keys: Vec::with_capacity(self.kv_context_size.min(256) * kv_width),
                    values: Vec::with_capacity(self.kv_context_size.min(256) * kv_width),
                    seq_len: 0,
                })
                .collect(),
            scratch: CpuScratch::new_for_shape(
                self.hidden_size,
                self.num_attention_heads * self.head_dim,
                self.num_kv_heads * self.head_dim,
                self.layers
                    .values()
                    .next()
                    .map(|layer| layer.gate_proj.rows)
                    .unwrap_or(self.hidden_size),
            ),
        }
    }

    pub fn embed_token(&self, token_id: usize) -> Result<Vec<f32>> {
        self.embed_tokens.row(token_id)
    }

    pub fn forward_layer_host(
        &self,
        state: &mut CpuLlamaState,
        layer_idx: usize,
        position: usize,
        hidden: &[f32],
    ) -> Result<Vec<f32>> {
        let layer = self
            .layers
            .get(&layer_idx)
            .ok_or_else(|| AegisError::InvalidPlan(format!("missing CPU layer `{layer_idx}`")))?;
        let scratch = &mut state.scratch;
        rms_norm_into(
            hidden,
            &layer.input_norm_weight,
            self.rms_norm_eps,
            &mut scratch.input_normed,
        );
        layer
            .q_proj
            .matvec_into(&scratch.input_normed, &mut scratch.q)?;
        layer
            .k_proj
            .matvec_into(&scratch.input_normed, &mut scratch.k)?;
        layer
            .v_proj
            .matvec_into(&scratch.input_normed, &mut scratch.v)?;

        apply_rope_in_place(
            &mut scratch.q,
            position,
            self.num_attention_heads,
            self.head_dim,
            &self.rope,
        )?;
        apply_rope_in_place(
            &mut scratch.k,
            position,
            self.num_kv_heads,
            self.head_dim,
            &self.rope,
        )?;
        let layer_state = state.layers.get_mut(layer_idx).ok_or_else(|| {
            AegisError::InvalidPlan(format!("missing CPU layer state `{layer_idx}`"))
        })?;
        layer_state.push(&scratch.k, &scratch.v, self.num_kv_heads * self.head_dim)?;
        attention_into(
            layer_state,
            &scratch.q,
            self.num_attention_heads,
            self.num_kv_heads,
            self.head_dim,
            &mut scratch.attn_context,
        )?;
        layer
            .o_proj
            .matvec_into(&scratch.attn_context, &mut scratch.attn_out)?;
        add_into(hidden, &scratch.attn_out, &mut scratch.residual)?;

        rms_norm_into(
            &scratch.residual,
            &layer.post_attention_norm_weight,
            self.rms_norm_eps,
            &mut scratch.post_normed,
        );
        layer
            .gate_proj
            .matvec_into(&scratch.post_normed, &mut scratch.gate)?;
        layer
            .up_proj
            .matvec_into(&scratch.post_normed, &mut scratch.up)?;
        swiglu_into(&scratch.gate, &scratch.up, &mut scratch.swiglu)?;
        layer
            .down_proj
            .matvec_into(&scratch.swiglu, &mut scratch.mlp_out)?;
        add_into(&scratch.residual, &scratch.mlp_out, &mut scratch.hidden_out)?;
        Ok(scratch.hidden_out.clone())
    }

    pub fn final_logits_host_with_state(
        &self,
        state: &mut CpuLlamaState,
        hidden: &[f32],
    ) -> Result<Vec<f32>> {
        rms_norm_into(
            hidden,
            &self.final_norm,
            self.rms_norm_eps,
            &mut state.scratch.final_hidden,
        );
        let mut logits = vec![0.0; self.lm_head.rows];
        self.lm_head
            .matvec_into(&state.scratch.final_hidden, &mut logits)?;
        Ok(logits)
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

fn host_store_for_cpu(store: StoragePlacement) -> StoragePlacement {
    match store {
        StoragePlacement::Ram => StoragePlacement::Ram,
        StoragePlacement::Mmap | StoragePlacement::Vram { .. } => StoragePlacement::Mmap,
    }
}

fn cpu_residency_for_store(store: StoragePlacement) -> aegisllm_base::tensor::storage::TensorResidencyPlan {
    match store {
        StoragePlacement::Ram => aegisllm_base::tensor::storage::TensorResidencyPlan::RamResident,
        StoragePlacement::Mmap | StoragePlacement::Vram { .. } => {
            aegisllm_base::tensor::storage::TensorResidencyPlan::FileBackedMmap
        }
    }
}

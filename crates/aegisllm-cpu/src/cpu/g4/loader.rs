//! Gemma-4 CPU loader. Consumes the same descriptor data the CUDA loader
//! uses (`ModelGraph.layer_metadata`, `HfConfig`) so the CPU forward stays
//! consistent with the CUDA ground truth
//! (`crates/aegisllm-cuda/src/executor/{full,loader}.rs`).

use super::linear::CpuLinear;
use super::rope::G4RopeConfig;
use super::state::{
    G4CpuExecutor, G4CpuLayer, G4DenseMlp, G4MoeExpert, G4MoeLayer, G4PleGlobal, G4PleLayer,
};
use crate::materialization::LinearMaterializationCache;
use aegisllm_base::artifact::ModelArtifact;
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::executor::tensors::{read_dense_vector, require_tensor, Bf16Matrix};
use aegisllm_base::graph::{ModelGraph, RegionId};
use aegisllm_base::model::{detect_architecture, AttentionPattern};
use aegisllm_base::planning::placement::{ResolvedPlacement, StoragePlacement};
use aegisllm_base::planning::runtime::RuntimePlan;
use aegisllm_base::tensor::layout::{LinearResidentLayout, MaterializationPolicy};
use aegisllm_base::tensor::storage::{TensorResidencyPlan, TensorStorageLoader};
use std::collections::BTreeMap;

impl G4CpuExecutor {
    pub(crate) fn from_artifact(
        artifact: &ModelArtifact,
        graph: &ModelGraph,
        placement: &ResolvedPlacement,
        runtime: &RuntimePlan,
    ) -> Result<Self> {
        let region_placements = placement.region_map();
        let runtime_layouts = runtime_linear_layouts_by_region(runtime);
        let mut loader = TensorStorageLoader::new();
        let mut materialization = LinearMaterializationCache::new();
        let text_prefix = &graph.text_prefix;

        let embed_region = region_placements
            .get(&RegionId("embed".into()))
            .ok_or_else(|| AegisError::InvalidPlan("missing embed placement".into()))?;
        let final_norm_region = region_placements
            .get(&RegionId("final_norm".into()))
            .ok_or_else(|| AegisError::InvalidPlan("missing final_norm placement".into()))?;
        let lm_head_region = region_placements
            .get(&RegionId("lm_head".into()))
            .ok_or_else(|| AegisError::InvalidPlan("missing lm_head placement".into()))?;

        let embed_tokens = Bf16Matrix::from_first_existing_tensor(
            artifact,
            &[
                &format!("{text_prefix}embed_tokens.weight"),
                "model.embed_tokens.weight",
            ],
            host_store(embed_region.store),
            &mut loader,
        )?;
        let final_norm = read_dense_vector(
            require_tensor(
                artifact,
                first_existing(artifact, &[&format!("{text_prefix}norm.weight"), "model.norm.weight"])?,
            )?,
            host_store(final_norm_region.store),
            &mut loader,
        )?;
        let lm_head = Bf16Matrix::from_first_existing_tensor(
            artifact,
            &[
                "lm_head.weight",
                &format!("{text_prefix}embed_tokens.weight"),
                "model.embed_tokens.weight",
            ],
            host_store(lm_head_region.store),
            &mut loader,
        )?;

        // PLE global apparatus (E4B / E2B).
        let ple = load_ple_global(artifact, graph, &mut loader)?;

        let num_layers = graph.num_layers;
        let arch = detect_architecture(&artifact.config)?;
        let rms_norm_eps = artifact.config.rms_norm_eps.unwrap_or(1e-5) as f32;
        let base_rope = base_rope_config(artifact);

        let mut layers = Vec::with_capacity(num_layers);
        let mut max_intermediate = 0usize;
        for layer in 0..num_layers {
            let region_id = RegionId(format!("layer.{layer}"));
            let placement = region_placements.get(&region_id).ok_or_else(|| {
                AegisError::InvalidPlan(format!("missing placement for `{}`", region_id.0))
            })?;
            let runtime_layout = runtime_layouts.get(&region_id.0).copied().unwrap_or_default();
            let store = host_store(placement.store);
            let prefix = format!("{text_prefix}layers.{layer}");

            let meta = graph.layer(layer);
            let layer_head_dim = meta.map(|m| m.head_dim).unwrap_or(graph.head_dim);
            let layer_num_kv_heads = meta.map(|m| m.num_kv_heads).unwrap_or(graph.num_kv_heads);
            let is_global = matches!(
                meta.map(|m| m.attention_pattern),
                Some(AttentionPattern::FullCausal)
            );
            let window_size = match meta.map(|m| m.attention_pattern) {
                Some(AttentionPattern::SlidingWindow { size }) => size,
                _ => 0,
            };
            // partial RoPE only on global layers; uses layer_head_dim.
            let partial_dim = artifact
                .config
                .partial_rotary_factor
                .filter(|&f| is_global && f < 1.0)
                .map(|f| (f as f64 * layer_head_dim as f64).round() as usize)
                .unwrap_or(0);
            // per-layer theta: sliding=rope_theta_sliding(10k), global=rope_theta_global(1M).
            let mut rope = base_rope.clone();
            let theta_override = if window_size > 0 {
                artifact.config.rope_theta_sliding
            } else {
                artifact.config.rope_theta_global
            };
            if let Some(t) = theta_override {
                rope.theta = t as f32;
            }

            // ── Norms (PrePost) ────────────────────────────────────────
            let has_pre_ffn = artifact
                .tensors
                .has(&format!("{prefix}.pre_feedforward_layernorm.weight"));
            let input_norm_weight = load_norm(artifact, &format!("{prefix}.input_layernorm.weight"), store, &mut loader)?;
            // pre-MLP norm: Gemma-4 uses pre_feedforward_layernorm; Llama-style
            // falls back to post_attention_layernorm.
            let pre_mlp_norm_weight = if has_pre_ffn {
                load_norm(artifact, &format!("{prefix}.pre_feedforward_layernorm.weight"), store, &mut loader)?
            } else {
                load_norm(artifact, &format!("{prefix}.post_attention_layernorm.weight"), store, &mut loader)?
            };
            // post-attn sublayer norm = post_attention_layernorm, ONLY when the
            // dedicated pre-MLP norm exists (Gemma-4 PrePost); otherwise it was
            // already consumed as the pre-MLP norm (Llama).
            let post_attn_sublayer_norm = if has_pre_ffn {
                load_optional_norm(
                    artifact, store, &mut loader,
                    &[
                        &format!("{prefix}.post_attention_layernorm.weight"),
                        &format!("{prefix}.post_attention_norm.weight"),
                    ],
                )?
            } else {
                None
            };
            let post_mlp_sublayer_norm = load_optional_norm(
                artifact, store, &mut loader,
                &[
                    &format!("{prefix}.post_mlp_norm.weight"),
                    &format!("{prefix}.post_feedforward_layernorm.weight"),
                ],
            )?;

            // ── Attention projections ──────────────────────────────────
            let q_proj = load_linear(
                artifact, &format!("{prefix}.self_attn.q_proj"), store, runtime_layout,
                &mut loader, &mut materialization,
            )?;
            let k_proj = load_linear(
                artifact, &format!("{prefix}.self_attn.k_proj"), store, runtime_layout,
                &mut loader, &mut materialization,
            )?;
            // Gemma-4 global layers may have attention_k_eq_v=true: no separate
            // v_proj → fall back to k_proj tensor.
            let v_prefix = if artifact.tensors.has(&format!("{prefix}.self_attn.v_proj.weight"))
                || artifact.tensors.has(&format!("{prefix}.self_attn.v_proj.weight_scale"))
            {
                format!("{prefix}.self_attn.v_proj")
            } else {
                format!("{prefix}.self_attn.k_proj")
            };
            let v_proj = load_linear(
                artifact, &v_prefix, store, runtime_layout, &mut loader, &mut materialization,
            )?;
            let o_proj = load_linear(
                artifact, &format!("{prefix}.self_attn.o_proj"), store, runtime_layout,
                &mut loader, &mut materialization,
            )?;
            let q_norm_weight = load_optional_norm(
                artifact, store, &mut loader, &[&format!("{prefix}.self_attn.q_norm.weight")],
            )?;
            let k_norm_weight = load_optional_norm(
                artifact, store, &mut loader, &[&format!("{prefix}.self_attn.k_norm.weight")],
            )?;

            // ── MLP / MoE ──────────────────────────────────────────────
            let layer_kind = meta.map(|m| m.kind);
            let is_moe = matches!(
                layer_kind,
                Some(aegisllm_base::model::LayerKind::MoEDecoder { .. })
            );
            let (mlp, moe) = if is_moe {
                let moe = load_moe(
                    artifact, &prefix, store, runtime_layout, &mut loader, &mut materialization,
                    &layer_kind,
                )?;
                max_intermediate = max_intermediate.max(moe.expert_intermediate_size);
                if let Some(shared) = &moe.shared_expert {
                    max_intermediate = max_intermediate.max(shared.gate_proj.rows());
                }
                (None, Some(moe))
            } else {
                let gate_proj = load_linear(
                    artifact, &format!("{prefix}.mlp.gate_proj"), store, runtime_layout,
                    &mut loader, &mut materialization,
                )?;
                let up_proj = load_linear(
                    artifact, &format!("{prefix}.mlp.up_proj"), store, runtime_layout,
                    &mut loader, &mut materialization,
                )?;
                let down_proj = load_linear(
                    artifact, &format!("{prefix}.mlp.down_proj"), store, runtime_layout,
                    &mut loader, &mut materialization,
                )?;
                max_intermediate = max_intermediate.max(gate_proj.rows());
                (Some(G4DenseMlp { gate_proj, up_proj, down_proj }), None)
            };

            // ── layer_scalar (BF16 [1]) ────────────────────────────────
            let layer_scalar = match artifact.tensors.get(&format!("{prefix}.layer_scalar")) {
                Some(t) => Some(read_scalar_f32(t, store, &mut loader)?),
                None => None,
            };

            // ── per-layer PLE weights ──────────────────────────────────
            let layer_ple = load_ple_layer(artifact, &prefix, store, &mut loader)?;

            layers.push(G4CpuLayer {
                input_norm_weight,
                pre_mlp_norm_weight,
                post_attn_sublayer_norm,
                post_mlp_sublayer_norm,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm_weight,
                k_norm_weight,
                mlp,
                moe,
                layer_head_dim,
                layer_num_kv_heads,
                window_size,
                partial_dim,
                rope,
                layer_scalar,
                kv_shared_from: None,
                ple: layer_ple,
            });
        }

        // KV-share resolution (E4B / E2B): layers ≥ num_layers - num_kv_shared_layers
        // read K/V from the most recent pre-boundary layer of matching attention type.
        if let Some(n_shared) = artifact.config.num_kv_shared_layers
            && n_shared > 0
            && n_shared < layers.len()
        {
            let first_shared = layers.len() - n_shared;
            let layer_is_global: Vec<bool> = (0..layers.len())
                .map(|i| matches!(arch.attention_pattern(i, &artifact.config), AttentionPattern::FullCausal))
                .collect();
            for li in first_shared..layers.len() {
                let need_global = layer_is_global[li];
                let parent = (0..first_shared).rev().find(|&k| layer_is_global[k] == need_global);
                match parent {
                    Some(p) => layers[li].kv_shared_from = Some(p),
                    None => {
                        return Err(AegisError::InvalidPlan(format!(
                            "KV-share: no pre-boundary parent of {} type for layer {}",
                            if need_global { "global" } else { "sliding" },
                            li
                        )));
                    }
                }
            }
        }

        let embed_scale = graph
            .embed_scale
            .unwrap_or_else(|| (graph.hidden_size as f32).sqrt());

        Ok(Self {
            hidden_size: graph.hidden_size,
            num_attention_heads: graph.num_attention_heads,
            rms_norm_eps,
            embed_scale,
            lm_head_softcap: graph.lm_head_softcap,
            embed_tokens,
            final_norm,
            lm_head,
            layers,
            kv_context_size: placement.kv_cache.context_size,
            ple,
            max_intermediate,
        })
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn base_rope_config(artifact: &ModelArtifact) -> G4RopeConfig {
    let scaling = artifact.config.rope_scaling.as_ref();
    G4RopeConfig {
        theta: artifact.config.rope_theta.unwrap_or(10_000.0) as f32,
        factor: scaling.and_then(|v| v.factor).map(|v| v as f32).unwrap_or(1.0),
        low_freq_factor: scaling.and_then(|v| v.low_freq_factor).map(|v| v as f32),
        high_freq_factor: scaling.and_then(|v| v.high_freq_factor).map(|v| v as f32),
        original_max_position_embeddings: scaling
            .and_then(|v| v.original_max_position_embeddings),
    }
}

/// CPU host store mapping: VRAM placements fall back to Mmap (host-resident).
fn host_store(store: StoragePlacement) -> StoragePlacement {
    match store {
        StoragePlacement::Ram => StoragePlacement::Ram,
        StoragePlacement::Mmap | StoragePlacement::Vram { .. } => StoragePlacement::Mmap,
    }
}

fn cpu_residency(store: StoragePlacement) -> TensorResidencyPlan {
    match store {
        StoragePlacement::Ram => TensorResidencyPlan::RamResident,
        StoragePlacement::Mmap | StoragePlacement::Vram { .. } => {
            TensorResidencyPlan::FileBackedMmap
        }
    }
}

fn first_existing<'a>(artifact: &ModelArtifact, names: &[&'a str]) -> Result<&'a str> {
    names
        .iter()
        .find(|n| artifact.tensors.has(n))
        .copied()
        .ok_or_else(|| AegisError::InvalidPlan(format!("missing any of tensors: {}", names.join(", "))))
}

fn load_norm(
    artifact: &ModelArtifact,
    name: &str,
    store: StoragePlacement,
    loader: &mut TensorStorageLoader,
) -> Result<Vec<f32>> {
    read_dense_vector(require_tensor(artifact, name)?, store, loader)
}

fn load_optional_norm(
    artifact: &ModelArtifact,
    store: StoragePlacement,
    loader: &mut TensorStorageLoader,
    names: &[&str],
) -> Result<Option<Vec<f32>>> {
    for name in names {
        if let Some(tensor) = artifact.tensors.get(name) {
            return Ok(Some(read_dense_vector(tensor, store, loader)?));
        }
    }
    Ok(None)
}

/// Read a 1-element BF16/F32 scalar tensor to f32.
fn read_scalar_f32(
    tensor: &aegisllm_base::tensor::TensorInfo,
    store: StoragePlacement,
    loader: &mut TensorStorageLoader,
) -> Result<f32> {
    let loaded = loader.load_for_store(tensor, store)?;
    let bytes = loaded.as_bytes();
    use aegisllm_base::tensor::TensorDType;
    match tensor.dtype {
        TensorDType::BF16 => {
            let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
            Ok(f32::from_bits((bits as u32) << 16))
        }
        TensorDType::F32 => Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])),
        other => Err(AegisError::InvalidPlan(format!(
            "layer_scalar `{}` must be BF16 or F32, got {:?}",
            tensor.name, other
        ))),
    }
}

/// Load one linear projection: NVFP4 when `{prefix}.weight_scale` is present,
/// else BF16.
fn load_linear(
    artifact: &ModelArtifact,
    prefix: &str,
    store: StoragePlacement,
    runtime_layout: RuntimeLinearLayout,
    loader: &mut TensorStorageLoader,
    materialization: &mut LinearMaterializationCache,
) -> Result<CpuLinear> {
    if artifact.tensors.has(&format!("{prefix}.weight_scale")) {
        let lin = materialization.load_cpu_nvfp4_linear(
            artifact,
            prefix,
            store,
            cpu_residency(store),
            runtime_layout.resident_layout,
            runtime_layout.materialization,
        )?;
        Ok(CpuLinear::Nvfp4(lin))
    } else {
        let m = Bf16Matrix::from_named_tensor(artifact, &format!("{prefix}.weight"), store, loader)?;
        Ok(CpuLinear::Bf16(m))
    }
}

#[allow(clippy::too_many_arguments)]
fn load_moe(
    artifact: &ModelArtifact,
    prefix: &str,
    store: StoragePlacement,
    runtime_layout: RuntimeLinearLayout,
    loader: &mut TensorStorageLoader,
    materialization: &mut LinearMaterializationCache,
    layer_kind: &Option<aegisllm_base::model::LayerKind>,
) -> Result<G4MoeLayer> {
    use aegisllm_base::model::LayerKind;
    let (num_experts, top_k, has_shared) = match layer_kind {
        Some(LayerKind::MoEDecoder { num_experts, top_k, has_shared_expert }) => {
            (*num_experts, *top_k, *has_shared_expert)
        }
        _ => return Err(AegisError::InvalidPlan("load_moe on non-MoE layer".into())),
    };

    // Router weight [num_experts, hidden_size], BF16.
    let router_candidates = [
        format!("{prefix}.router.proj.weight"),
        format!("{prefix}.mlp.router_logits.weight"),
        format!("{prefix}.block_sparse_moe.gate.weight"),
    ];
    let router_name = router_candidates
        .iter()
        .find(|n| artifact.tensors.has(n))
        .ok_or_else(|| {
            AegisError::InvalidPlan(format!("missing MoE router for `{prefix}`"))
        })?;
    let router = Bf16Matrix::from_named_tensor(artifact, router_name, store, loader)?;
    let router_input_scale = load_optional_norm(
        artifact, store, loader, &[&format!("{prefix}.router.scale")],
    )?;
    let router_per_expert_scale = match artifact.tensors.get(&format!("{prefix}.router.per_expert_scale")) {
        Some(t) => Some(read_dense_vector(t, store, loader)?),
        None => None,
    };

    // Per-expert projections. Gemma-4: {prefix}.experts.N.*; Qwen: {prefix}.mlp.experts.N.*
    let expert_base = if artifact.tensors.has(&format!("{prefix}.experts.0.gate_proj.weight")) {
        format!("{prefix}.experts")
    } else {
        format!("{prefix}.mlp.experts")
    };
    let mut experts = Vec::with_capacity(num_experts);
    let mut expert_intermediate = 0usize;
    for e in 0..num_experts {
        let ep = format!("{expert_base}.{e}");
        let gate_proj = load_linear(artifact, &format!("{ep}.gate_proj"), store, runtime_layout, loader, materialization)?;
        let up_proj = load_linear(artifact, &format!("{ep}.up_proj"), store, runtime_layout, loader, materialization)?;
        let down_proj = load_linear(artifact, &format!("{ep}.down_proj"), store, runtime_layout, loader, materialization)?;
        expert_intermediate = gate_proj.rows();
        experts.push(G4MoeExpert { gate_proj, up_proj, down_proj });
    }

    let shared_expert = if has_shared
        && artifact.tensors.has(&format!("{prefix}.mlp.shared_expert.gate_proj.weight"))
    {
        let gate_proj = load_linear(artifact, &format!("{prefix}.mlp.shared_expert.gate_proj"), store, runtime_layout, loader, materialization)?;
        let up_proj = load_linear(artifact, &format!("{prefix}.mlp.shared_expert.up_proj"), store, runtime_layout, loader, materialization)?;
        let down_proj = load_linear(artifact, &format!("{prefix}.mlp.shared_expert.down_proj"), store, runtime_layout, loader, materialization)?;
        Some(G4DenseMlp { gate_proj, up_proj, down_proj })
    } else if has_shared && artifact.tensors.has(&format!("{prefix}.shared_experts.gate_proj.weight")) {
        let gate_proj = load_linear(artifact, &format!("{prefix}.shared_experts.gate_proj"), store, runtime_layout, loader, materialization)?;
        let up_proj = load_linear(artifact, &format!("{prefix}.shared_experts.up_proj"), store, runtime_layout, loader, materialization)?;
        let down_proj = load_linear(artifact, &format!("{prefix}.shared_experts.down_proj"), store, runtime_layout, loader, materialization)?;
        Some(G4DenseMlp { gate_proj, up_proj, down_proj })
    } else {
        None
    };

    let post_feedforward_layernorm_1 = load_optional_norm(
        artifact, store, loader, &[&format!("{prefix}.post_feedforward_layernorm_1.weight")],
    )?;
    let pre_feedforward_layernorm_2 = load_optional_norm(
        artifact, store, loader, &[&format!("{prefix}.pre_feedforward_layernorm_2.weight")],
    )?;
    let post_feedforward_layernorm_2 = load_optional_norm(
        artifact, store, loader, &[&format!("{prefix}.post_feedforward_layernorm_2.weight")],
    )?;

    Ok(G4MoeLayer {
        router,
        router_input_scale,
        router_per_expert_scale,
        experts,
        shared_expert,
        top_k,
        num_experts,
        expert_intermediate_size: expert_intermediate,
        post_feedforward_layernorm_1,
        pre_feedforward_layernorm_2,
        post_feedforward_layernorm_2,
    })
}

fn load_ple_global(
    artifact: &ModelArtifact,
    graph: &ModelGraph,
    loader: &mut TensorStorageLoader,
) -> Result<Option<G4PleGlobal>> {
    let ple_dim = match artifact.config.hidden_size_per_layer_input {
        Some(d) if d > 0 => d,
        _ => return Ok(None),
    };
    let text_prefix = &graph.text_prefix;
    let hidden = graph.hidden_size;
    // PLE tables are host-resident; use Ram store on CPU.
    let store = StoragePlacement::Ram;
    let embed_table = Bf16Matrix::from_named_tensor(
        artifact, &format!("{text_prefix}embed_tokens_per_layer.weight"), store, loader,
    )?;
    let model_projection = Bf16Matrix::from_named_tensor(
        artifact, &format!("{text_prefix}per_layer_model_projection.weight"), store, loader,
    )?;
    let projection_norm = read_dense_vector(
        require_tensor(artifact, &format!("{text_prefix}per_layer_projection_norm.weight"))?,
        store, loader,
    )?;
    Ok(Some(G4PleGlobal {
        embed_table,
        model_projection,
        projection_norm,
        ple_dim,
        embed_scale_per_layer: (ple_dim as f32).sqrt(),
        model_projection_scale: 1.0 / (hidden as f32).sqrt(),
        combine_scale: 1.0 / (2.0_f32).sqrt(),
    }))
}

fn load_ple_layer(
    artifact: &ModelArtifact,
    prefix: &str,
    store: StoragePlacement,
    loader: &mut TensorStorageLoader,
) -> Result<Option<G4PleLayer>> {
    if artifact.config.hidden_size_per_layer_input.unwrap_or(0) == 0 {
        return Ok(None);
    }
    let input_gate = Bf16Matrix::from_named_tensor(
        artifact, &format!("{prefix}.per_layer_input_gate.weight"), store, loader,
    )?;
    let projection = Bf16Matrix::from_named_tensor(
        artifact, &format!("{prefix}.per_layer_projection.weight"), store, loader,
    )?;
    let post_norm = read_dense_vector(
        require_tensor(artifact, &format!("{prefix}.post_per_layer_input_norm.weight"))?,
        store, loader,
    )?;
    Ok(Some(G4PleLayer { input_gate, projection, post_norm }))
}

// ── Runtime linear layout (mirrors loader.rs / block.rs) ────────────────────

#[derive(Debug, Clone, Copy)]
pub(crate) struct RuntimeLinearLayout {
    pub(crate) resident_layout: LinearResidentLayout,
    pub(crate) materialization: MaterializationPolicy,
}

impl Default for RuntimeLinearLayout {
    fn default() -> Self {
        Self {
            resident_layout: LinearResidentLayout::PackedSource,
            materialization: MaterializationPolicy::Lazy,
        }
    }
}

fn runtime_linear_layouts_by_region(runtime: &RuntimePlan) -> BTreeMap<String, RuntimeLinearLayout> {
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

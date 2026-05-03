use std::collections::BTreeMap;

use aegisllm_base::artifact::ModelArtifact;
use crate::cuda::CudaWeightLoader;
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::graph::GraphRegionKind;
use aegisllm_base::model::LayerKind;
use aegisllm_base::planning::placement::{ComputePlacement, RegionPlacement, StoragePlacement};
use aegisllm_base::planning::runtime::RuntimePlan;
use aegisllm_base::tensor::TensorInfo;
use aegisllm_base::tensor::layout::LinearResidentLayout;
use aegisllm_base::tensor::storage::TensorResidencyPlan;

use crate::cuda::DeviceBuffer;
use super::rope::RopeConfig;
use super::state::{CudaLayer, CudaMoE, CudaMoEExpert, CudaMoEShared};
use aegisllm_base::executor::tensors::require_tensor;

#[derive(Debug, Clone)]
pub(super) struct CudaLayerShape {
    pub(super) hidden_size: usize,
    pub(super) intermediate_size: Option<usize>,
    pub(super) num_attention_heads: usize,
    pub(super) num_kv_heads: usize,
    pub(super) head_dim: usize,
    /// True when this graph was built from a MatFormer nested-param checkpoint
    /// (e.g. Gemma 4 E2B). When set, linear layers are loaded via submatrix
    /// slicing rather than loading the full tensor.
    pub(super) is_sliced: bool,
    /// Tensor-name prefix for the text decoder (e.g. `"model."` or
    /// `"model.language_model."`). Includes trailing dot.
    pub(super) text_prefix: String,
}

pub(super) fn cuda_residency_for_store(
    store: StoragePlacement,
    device: usize,
) -> Result<TensorResidencyPlan> {
    match store {
        StoragePlacement::Vram {
            device: store_device,
        } if store_device == device => Ok(TensorResidencyPlan::VramResident { device }),
        StoragePlacement::Ram | StoragePlacement::Mmap => {
            Ok(TensorResidencyPlan::StagedHostToDevice { device })
        }
        StoragePlacement::Vram {
            device: store_device,
        } => Err(AegisError::Unsupported(format!(
            "cross-device CUDA residency is not implemented: store=vram:{store_device} compute=cuda:{device}"
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn load_cuda_layer(
    cuda: &CudaWeightLoader<'_>,
    artifact: &ModelArtifact,
    layer: usize,
    region_kind: GraphRegionKind,
    layer_kind: LayerKind,
    placement: &RegionPlacement,
    resident_layout: LinearResidentLayout,
    shape: CudaLayerShape,
    window_size: usize,
    partial_dim: usize,
    loader: &mut aegisllm_base::tensor::storage::TensorStorageLoader,
) -> Result<CudaLayer> {
    if region_kind != GraphRegionKind::TransformerBlock {
        return Err(AegisError::InvalidPlan(format!(
            "region `{}` is not a transformer block",
            placement.region_id.0
        )));
    }
    match placement.compute {
        ComputePlacement::Cuda { device } if device == cuda.device_index() => {}
        other => {
            return Err(AegisError::InvalidPlan(format!(
                "CUDA layer `{}` has compute={other}, expected cuda:{}",
                placement.region_id.0,
                cuda.device_index()
            )));
        }
    }

    let prefix = format!("{}layers.{layer}", shape.text_prefix);
    let residency = cuda_residency_for_store(placement.store, cuda.device_index())?;

    let is_moe = matches!(layer_kind, LayerKind::MoEDecoder { .. });
    let is_sliced = shape.is_sliced;

    // Effective dimensions used for sliced (MatFormer) loads.
    let hidden = shape.hidden_size;
    let q_width = shape.num_attention_heads * shape.head_dim;
    let kv_width = shape.num_kv_heads * shape.head_dim;
    let intermediate = shape.intermediate_size.unwrap_or(hidden);

    let (gate_proj, up_proj, down_proj) = if is_moe {
        (
            cuda.alloc_dummy_nvfp4_linear(&format!("{prefix}.mlp.gate_proj"))?,
            cuda.alloc_dummy_nvfp4_linear(&format!("{prefix}.mlp.up_proj"))?,
            cuda.alloc_dummy_nvfp4_linear(&format!("{prefix}.mlp.down_proj"))?,
        )
    } else {
        (
            load_nvfp4_maybe_sliced_linear(
                cuda, artifact, &format!("{prefix}.mlp.gate_proj"),
                placement.store, residency, resident_layout,
                intermediate, hidden, is_sliced, loader,
            )?,
            load_nvfp4_maybe_sliced_linear(
                cuda, artifact, &format!("{prefix}.mlp.up_proj"),
                placement.store, residency, resident_layout,
                intermediate, hidden, is_sliced, loader,
            )?,
            load_nvfp4_maybe_sliced_linear(
                cuda, artifact, &format!("{prefix}.mlp.down_proj"),
                placement.store, residency, resident_layout,
                hidden, intermediate, is_sliced, loader,
            )?,
        )
    };

    let moe = if let LayerKind::MoEDecoder { num_experts, top_k, has_shared_expert } = layer_kind {
        Some(Box::new(load_cuda_moe(
            cuda,
            artifact,
            layer,
            num_experts,
            top_k,
            has_shared_expert,
            &shape.text_prefix,
            placement.store,
            residency,
            resident_layout,
            loader,
        )?))
    } else {
        None
    };

    let layer_out = CudaLayer {
        input_norm_weight: cuda.load_dense_vector_with_store(
            require_tensor(artifact, &format!("{prefix}.input_layernorm.weight"))?,
            placement.store,
            loader,
        )?,
        post_attention_norm_weight: cuda.load_dense_vector_with_store(
            require_tensor(
                artifact,
                &format!("{prefix}.post_attention_layernorm.weight"),
            )?,
            placement.store,
            loader,
        )?,
        post_attn_sublayer_norm: load_optional_norm_weight(
            cuda, artifact, placement.store, loader,
            &[
                &format!("{prefix}.post_attention_norm.weight"),
                &format!("{prefix}.post_attn_layernorm.weight"),
            ],
        )?,
        post_mlp_sublayer_norm: load_optional_norm_weight(
            cuda, artifact, placement.store, loader,
            &[
                &format!("{prefix}.post_mlp_norm.weight"),
                &format!("{prefix}.post_feedforward_layernorm.weight"),
            ],
        )?,
        q_proj: load_nvfp4_maybe_sliced_linear(
            cuda, artifact, &format!("{prefix}.self_attn.q_proj"),
            placement.store, residency, resident_layout,
            q_width, hidden, is_sliced, loader,
        )?,
        k_proj: load_nvfp4_maybe_sliced_linear(
            cuda, artifact, &format!("{prefix}.self_attn.k_proj"),
            placement.store, residency, resident_layout,
            kv_width, hidden, is_sliced, loader,
        )?,
        v_proj: load_nvfp4_maybe_sliced_linear(
            cuda, artifact, &format!("{prefix}.self_attn.v_proj"),
            placement.store, residency, resident_layout,
            kv_width, hidden, is_sliced, loader,
        )?,
        // CUTLASS fused QKV group is not supported for sliced models.
        qkv_proj: if is_sliced {
            None
        } else {
            cuda.load_cutlass_qkv_group_with_layout(
                artifact,
                &format!("{prefix}.self_attn.q_proj"),
                &format!("{prefix}.self_attn.k_proj"),
                &format!("{prefix}.self_attn.v_proj"),
                placement.store,
                residency,
                resident_layout,
                loader,
            )?
        },
        o_proj: load_nvfp4_maybe_sliced_linear(
            cuda, artifact, &format!("{prefix}.self_attn.o_proj"),
            placement.store, residency, resident_layout,
            hidden, q_width, is_sliced, loader,
        )?,
        gate_proj,
        up_proj,
        down_proj,
        window_size,
        rope: RopeConfig::from_artifact(artifact).to_device_with_partial_dim(partial_dim)?,
        moe,
    };
    if !is_moe {
        validate_cuda_layer_shape(&layer_out, shape)?;
    }
    Ok(layer_out)
}

#[allow(clippy::too_many_arguments)]
fn load_cuda_moe(
    cuda: &CudaWeightLoader<'_>,
    artifact: &ModelArtifact,
    layer: usize,
    num_experts: usize,
    top_k: usize,
    has_shared_expert: bool,
    text_prefix: &str,
    store: StoragePlacement,
    residency: TensorResidencyPlan,
    resident_layout: LinearResidentLayout,
    loader: &mut aegisllm_base::tensor::storage::TensorStorageLoader,
) -> Result<CudaMoE> {
    let prefix = format!("{text_prefix}layers.{layer}");

    // Router weight matrix [num_experts, hidden_size] — BF16
    let router_tensor = first_existing_tensor(
        artifact,
        &[
            &format!("{prefix}.mlp.router_logits.weight"),
            &format!("{prefix}.block_sparse_moe.gate.weight"),
        ],
    )?;
    let router = cuda.load_bf16_matrix_with_store(router_tensor, store, residency, loader)?;

    // Per-expert projections
    let mut experts = Vec::with_capacity(num_experts);
    for expert_idx in 0..num_experts {
        let ep = format!("{prefix}.mlp.experts.{expert_idx}");
        let gate_proj = cuda.load_nvfp4_linear_with_layout(
            artifact, &format!("{ep}.gate_proj"), store, residency, resident_layout, loader,
        )?;
        let up_proj = cuda.load_nvfp4_linear_with_layout(
            artifact, &format!("{ep}.up_proj"), store, residency, resident_layout, loader,
        )?;
        let down_proj = cuda.load_nvfp4_linear_with_layout(
            artifact, &format!("{ep}.down_proj"), store, residency, resident_layout, loader,
        )?;
        experts.push(CudaMoEExpert { gate_proj, up_proj, down_proj });
    }

    let expert_intermediate_size = experts
        .first()
        .map(|e| e.gate_proj.rows)
        .ok_or_else(|| AegisError::InvalidPlan("MoE layer has no experts".into()))?;

    // Optional shared (always-active) expert
    let shared_expert = if has_shared_expert {
        let sp = format!("{prefix}.mlp.shared_expert");
        let gate_proj = cuda.load_nvfp4_linear_with_layout(
            artifact, &format!("{sp}.gate_proj"), store, residency, resident_layout, loader,
        )?;
        let up_proj = cuda.load_nvfp4_linear_with_layout(
            artifact, &format!("{sp}.up_proj"), store, residency, resident_layout, loader,
        )?;
        let down_proj = cuda.load_nvfp4_linear_with_layout(
            artifact, &format!("{sp}.down_proj"), store, residency, resident_layout, loader,
        )?;
        Some(CudaMoEShared { gate_proj, up_proj, down_proj })
    } else {
        None
    };

    Ok(CudaMoE {
        router,
        experts,
        shared_expert,
        top_k,
        num_experts,
        expert_intermediate_size,
    })
}

pub(super) fn runtime_layouts_by_region(
    runtime: &RuntimePlan,
) -> BTreeMap<String, LinearResidentLayout> {
    runtime
        .kernels
        .iter()
        .map(|kernel| (kernel.name.clone(), kernel.linear_layout.resident_layout))
        .collect()
}

pub(super) fn first_existing_tensor<'a>(
    artifact: &'a ModelArtifact,
    names: &[&str],
) -> Result<&'a TensorInfo> {
    names
        .iter()
        .find_map(|name| artifact.tensors.get(name))
        .ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "missing any of dense matrix tensors: {}",
                names.join(", ")
            ))
        })
}

/// Tries each name in `names` in order; loads the first one found, returns None if none exist.
fn load_optional_norm_weight(
    cuda: &CudaWeightLoader<'_>,
    artifact: &ModelArtifact,
    store: StoragePlacement,
    loader: &mut aegisllm_base::tensor::storage::TensorStorageLoader,
    names: &[&str],
) -> Result<Option<DeviceBuffer<f32>>> {
    for name in names {
        if let Some(tensor) = artifact.tensors.get(name) {
            return cuda
                .load_dense_vector_with_store(tensor, store, loader)
                .map(Some);
        }
    }
    Ok(None)
}

fn load_nvfp4_maybe_sliced_linear(
    cuda: &CudaWeightLoader<'_>,
    artifact: &ModelArtifact,
    prefix: &str,
    store: StoragePlacement,
    residency: TensorResidencyPlan,
    resident_layout: LinearResidentLayout,
    eff_rows: usize,
    eff_logical_cols: usize,
    is_sliced: bool,
    loader: &mut aegisllm_base::tensor::storage::TensorStorageLoader,
) -> Result<crate::cuda::DeviceNvfp4Linear> {
    if is_sliced {
        cuda.load_nvfp4_linear_sliced_with_layout(
            artifact,
            prefix,
            store,
            residency,
            resident_layout,
            eff_rows,
            eff_logical_cols,
            loader,
        )
    } else {
        cuda.load_nvfp4_linear_with_layout(artifact, prefix, store, residency, resident_layout, loader)
    }
}

fn validate_cuda_layer_shape(layer: &CudaLayer, shape: CudaLayerShape) -> Result<()> {
    let hidden = shape.hidden_size;
    let q_width = shape.num_attention_heads * shape.head_dim;
    let kv_width = shape.num_kv_heads * shape.head_dim;
    let intermediate = shape.intermediate_size.unwrap_or(layer.gate_proj.rows);

    require_vector_len(
        "input_layernorm.weight",
        layer.input_norm_weight.len(),
        hidden,
    )?;
    require_vector_len(
        "post_attention_layernorm.weight",
        layer.post_attention_norm_weight.len(),
        hidden,
    )?;
    require_linear_shape(
        &layer.q_proj.name,
        layer.q_proj.rows,
        layer.q_proj.cols,
        q_width,
        hidden,
    )?;
    require_linear_shape(
        &layer.k_proj.name,
        layer.k_proj.rows,
        layer.k_proj.cols,
        kv_width,
        hidden,
    )?;
    require_linear_shape(
        &layer.v_proj.name,
        layer.v_proj.rows,
        layer.v_proj.cols,
        kv_width,
        hidden,
    )?;
    require_linear_shape(
        &layer.o_proj.name,
        layer.o_proj.rows,
        layer.o_proj.cols,
        hidden,
        q_width,
    )?;
    require_linear_shape(
        &layer.gate_proj.name,
        layer.gate_proj.rows,
        layer.gate_proj.cols,
        intermediate,
        hidden,
    )?;
    require_linear_shape(
        &layer.up_proj.name,
        layer.up_proj.rows,
        layer.up_proj.cols,
        intermediate,
        hidden,
    )?;
    require_linear_shape(
        &layer.down_proj.name,
        layer.down_proj.rows,
        layer.down_proj.cols,
        hidden,
        intermediate,
    )?;
    Ok(())
}

fn require_vector_len(name: &str, actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        return Err(AegisError::InvalidPlan(format!(
            "CUDA layer tensor `{name}` shape mismatch: expected len={expected}, got len={actual}"
        )));
    }
    Ok(())
}

fn require_linear_shape(
    name: &str,
    rows: usize,
    cols: usize,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<()> {
    if rows != expected_rows || cols != expected_cols {
        return Err(AegisError::InvalidPlan(format!(
            "CUDA linear `{name}` shape mismatch: expected {expected_rows}x{expected_cols}, got {rows}x{cols}"
        )));
    }
    Ok(())
}

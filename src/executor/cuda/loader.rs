use std::collections::BTreeMap;

use crate::artifact::ModelArtifact;
use crate::cuda::CudaWeightLoader;
use crate::error::{AegisError, Result};
use crate::graph::GraphRegionKind;
use crate::planning::placement::{ComputePlacement, RegionPlacement, StoragePlacement};
use crate::planning::runtime::RuntimePlan;
use crate::tensor::TensorInfo;
use crate::tensor::layout::LinearResidentLayout;
use crate::tensor::storage::TensorResidencyPlan;

use super::state::CudaLayer;
use crate::executor::tensors::require_tensor;

#[derive(Debug, Clone, Copy)]
pub(super) struct CudaLayerShape {
    pub(super) hidden_size: usize,
    pub(super) intermediate_size: Option<usize>,
    pub(super) num_attention_heads: usize,
    pub(super) num_kv_heads: usize,
    pub(super) head_dim: usize,
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

pub(super) fn load_cuda_layer(
    cuda: &CudaWeightLoader<'_>,
    artifact: &ModelArtifact,
    layer: usize,
    region_kind: GraphRegionKind,
    placement: &RegionPlacement,
    resident_layout: LinearResidentLayout,
    shape: CudaLayerShape,
    loader: &mut crate::tensor::storage::TensorStorageLoader,
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

    let prefix = format!("model.layers.{layer}");
    let residency = cuda_residency_for_store(placement.store, cuda.device_index())?;
    let layer = CudaLayer {
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
        q_proj: cuda.load_nvfp4_linear_with_layout(
            artifact,
            &format!("{prefix}.self_attn.q_proj"),
            placement.store,
            residency,
            resident_layout,
            loader,
        )?,
        k_proj: cuda.load_nvfp4_linear_with_layout(
            artifact,
            &format!("{prefix}.self_attn.k_proj"),
            placement.store,
            residency,
            resident_layout,
            loader,
        )?,
        v_proj: cuda.load_nvfp4_linear_with_layout(
            artifact,
            &format!("{prefix}.self_attn.v_proj"),
            placement.store,
            residency,
            resident_layout,
            loader,
        )?,
        o_proj: cuda.load_nvfp4_linear_with_layout(
            artifact,
            &format!("{prefix}.self_attn.o_proj"),
            placement.store,
            residency,
            resident_layout,
            loader,
        )?,
        gate_proj: cuda.load_nvfp4_linear_with_layout(
            artifact,
            &format!("{prefix}.mlp.gate_proj"),
            placement.store,
            residency,
            resident_layout,
            loader,
        )?,
        up_proj: cuda.load_nvfp4_linear_with_layout(
            artifact,
            &format!("{prefix}.mlp.up_proj"),
            placement.store,
            residency,
            resident_layout,
            loader,
        )?,
        down_proj: cuda.load_nvfp4_linear_with_layout(
            artifact,
            &format!("{prefix}.mlp.down_proj"),
            placement.store,
            residency,
            resident_layout,
            loader,
        )?,
    };
    validate_cuda_layer_shape(&layer, shape)?;
    Ok(layer)
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
        .find_map(|name| artifact.tensors.get(*name))
        .ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "missing any of dense matrix tensors: {}",
                names.join(", ")
            ))
        })
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

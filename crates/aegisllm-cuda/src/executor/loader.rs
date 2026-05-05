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
use super::state::{CudaLayer, CudaLinear, CudaMoE, CudaMoEExpert, CudaMoEShared};
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
    let layer_head_dim = shape.head_dim;
    let layer_num_kv_heads = shape.num_kv_heads;

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
        // Gemma 4 (PrePost): post_attention_layernorm = post-attn sublayer norm,
        //                     pre_feedforward_layernorm = pre-MLP norm.
        // Llama / Qwen (PreOnly): post_attention_layernorm = pre-MLP norm only.
        // Detect by checking whether pre_feedforward_layernorm is present.
        post_attention_norm_weight: {
            let pre_mlp_name = if artifact
                .tensors
                .has(&format!("{prefix}.pre_feedforward_layernorm.weight"))
            {
                // Gemma 4 PrePost: use the dedicated pre-MLP norm.
                format!("{prefix}.pre_feedforward_layernorm.weight")
            } else {
                // Llama / Qwen: the single norm after attention is the pre-MLP norm.
                format!("{prefix}.post_attention_layernorm.weight")
            };
            cuda.load_dense_vector_with_store(
                require_tensor(artifact, &pre_mlp_name)?,
                placement.store,
                loader,
            )?
        },
        post_attn_sublayer_norm: load_optional_norm_weight(
            cuda, artifact, placement.store, loader,
            &[
                // Gemma 4: post_attention_layernorm is the post-attn sublayer norm.
                &format!("{prefix}.post_attention_layernorm.weight"),
                // Alternative naming used in some variants:
                &format!("{prefix}.post_attention_norm.weight"),
                &format!("{prefix}.post_attn_layernorm.weight"),
            ],
        ).map(|opt| {
            // For Llama/Qwen, post_attention_layernorm was loaded as post_attention_norm_weight
            // above. We must NOT also load it as post_attn_sublayer_norm or we'd double-apply it.
            // Clear the optional if pre_feedforward_layernorm was absent (Llama/Qwen case).
            if artifact.tensors.has(&format!("{prefix}.pre_feedforward_layernorm.weight")) {
                opt
            } else {
                None
            }
        })?,
        post_mlp_sublayer_norm: load_optional_norm_weight(
            cuda, artifact, placement.store, loader,
            &[
                &format!("{prefix}.post_mlp_norm.weight"),
                &format!("{prefix}.post_feedforward_layernorm.weight"),
            ],
        )?,
        // Gemma 4 MoE: post-norm on shared-MLP stream (post_feedforward_layernorm_1).
        post_feedforward_layernorm_1: load_optional_norm_weight(
            cuda, artifact, placement.store, loader,
            &[&format!("{prefix}.post_feedforward_layernorm_1.weight")],
        )?,
        // Gemma 4 MoE: separate pre-norm for expert inputs (pre_feedforward_layernorm_2).
        pre_feedforward_layernorm_2: load_optional_norm_weight(
            cuda, artifact, placement.store, loader,
            &[&format!("{prefix}.pre_feedforward_layernorm_2.weight")],
        )?,
        // Gemma 4 MoE: post-norm on routed-expert stream (post_feedforward_layernorm_2).
        post_feedforward_layernorm_2: load_optional_norm_weight(
            cuda, artifact, placement.store, loader,
            &[&format!("{prefix}.post_feedforward_layernorm_2.weight")],
        )?,
        // Gemma 4: per-layer scalar (BF16 tensor of shape [1]).
        layer_scalar: {
            use crate::cuda::loader::read_scalar_f32_with_loader;
            artifact.tensors.get(&format!("{prefix}.layer_scalar"))
                .map(|t| read_scalar_f32_with_loader(loader, t, placement.store))
                .transpose()?
        },
        q_proj: load_cuda_linear(
            cuda, artifact, &format!("{prefix}.self_attn.q_proj"),
            placement.store, residency, resident_layout,
            q_width, hidden, is_sliced, loader,
        )?,
        k_proj: load_cuda_linear(
            cuda, artifact, &format!("{prefix}.self_attn.k_proj"),
            placement.store, residency, resident_layout,
            kv_width, hidden, is_sliced, loader,
        )?,
        v_proj: {
            // Gemma 4 global layers have attention_k_eq_v=true: no separate v_proj,
            // K and V use the same projection. Fall back to k_proj weights when absent.
            let v_has_weight = artifact.tensors.has(&format!("{prefix}.self_attn.v_proj.weight"))
                || artifact.tensors.has(&format!("{prefix}.self_attn.v_proj.weight_scale"));
            let v_prefix = if v_has_weight {
                format!("{prefix}.self_attn.v_proj")
            } else {
                format!("{prefix}.self_attn.k_proj")
            };
            load_cuda_linear(
                cuda, artifact, &v_prefix,
                placement.store, residency, resident_layout,
                kv_width, hidden, is_sliced, loader,
            )?
        },
        // CUTLASS fused QKV group is not supported for sliced models.
        // Only attempt fused QKV for NVFP4 layers (BF16 attn layers skip this).
        qkv_proj: if is_sliced {
            None
        } else if artifact.tensors.has(&format!("{prefix}.self_attn.q_proj.weight_scale")) {
            cuda.load_cutlass_qkv_group_with_layout(
                artifact,
                &format!("{prefix}.self_attn.q_proj"),
                &format!("{prefix}.self_attn.k_proj"),
                &format!("{prefix}.self_attn.v_proj"),
                placement.store,
                residency,
                resident_layout,
                loader,
            )?.map(CudaLinear::Nvfp4)
        } else {
            None
        },
        o_proj: load_cuda_linear(
            cuda, artifact, &format!("{prefix}.self_attn.o_proj"),
            placement.store, residency, resident_layout,
            hidden, q_width, is_sliced, loader,
        )?,
        // Gemma 4: per-head RMS norm on Q (applied between q_proj and RoPE).
        q_norm_weight: load_optional_norm_weight(
            cuda, artifact, placement.store, loader,
            &[&format!("{prefix}.self_attn.q_norm.weight")],
        )?,
        // Gemma 4: per-head RMS norm on K (applied between k_proj and RoPE).
        k_norm_weight: load_optional_norm_weight(
            cuda, artifact, placement.store, loader,
            &[&format!("{prefix}.self_attn.k_norm.weight")],
        )?,
        gate_proj,
        up_proj,
        down_proj,
        window_size,
        rope: {
            // Gemma 4 has different rope_theta for sliding (10k) and global (1M) layers.
            // Detect via `is_sliced` field shape: window_size > 0 means sliding layer.
            let theta_override = if window_size > 0 {
                artifact.config.rope_theta_sliding.map(|v| v as f32)
            } else {
                artifact.config.rope_theta_global.map(|v| v as f32)
            };
            RopeConfig::from_artifact(artifact)
                .to_device_with_partial_dim_and_theta(partial_dim, theta_override)?
        },
        moe,
        layer_head_dim,
        layer_num_kv_heads,
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
    // Gemma 4: {prefix}.router.proj.weight
    // Qwen 3.x: {prefix}.mlp.router_logits.weight
    // Mixtral-style: {prefix}.block_sparse_moe.gate.weight
    let router_tensor = first_existing_tensor(
        artifact,
        &[
            &format!("{prefix}.router.proj.weight"),
            &format!("{prefix}.mlp.router_logits.weight"),
            &format!("{prefix}.block_sparse_moe.gate.weight"),
        ],
    )?;
    // Router runs every layer's MoE block; host-resident BF16 matvec goes through the
    // CPU rayon path which is ~30× slower than GPU. Force to VRAM (router is tiny,
    // [num_experts, hidden_size] BF16 = ~720 KB per layer).
    let router =
        cuda.load_bf16_matrix_with_store_opts(router_tensor, store, residency, loader, true)?;

    // Gemma 4: per-input-dim scaling vector at `{prefix}.router.scale` (BF16, len=hidden_size).
    // Applied element-wise to router input BEFORE projection.
    let router_input_scale = load_optional_norm_weight(
        cuda, artifact, store, loader,
        &[&format!("{prefix}.router.scale")],
    )?;
    // Gemma 4: per-expert scaling vector at `{prefix}.router.per_expert_scale` (BF16, len=num_experts).
    // Applied to top-k routing weights AFTER softmax (in Python: `top_k_weights *= per_expert_scale[idx]`).
    // Cached on host because the top-k selection already happens on CPU.
    let router_per_expert_scale_host: Option<Vec<f32>> = artifact
        .tensors
        .get(&format!("{prefix}.router.per_expert_scale"))
        .map(|tensor| {
            let bytes = loader
                .load_for_store(tensor, store)?
                .as_bytes()
                .to_vec();
            // BF16 → f32 conversion (Gemma 4 stores per_expert_scale in BF16).
            if tensor.dtype == aegisllm_base::tensor::TensorDType::BF16 {
                Ok(bytes
                    .chunks_exact(2)
                    .map(|c| {
                        let bits = u16::from_le_bytes([c[0], c[1]]);
                        f32::from_bits((bits as u32) << 16)
                    })
                    .collect::<Vec<f32>>())
            } else if tensor.dtype == aegisllm_base::tensor::TensorDType::F32 {
                Ok(bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect::<Vec<f32>>())
            } else {
                Err(AegisError::InvalidPlan(format!(
                    "router.per_expert_scale must be BF16 or F32, got {:?}",
                    tensor.dtype
                )))
            }
        })
        .transpose()?;

    // Per-expert projections.
    // Gemma 4: {prefix}.experts.{i}.{proj}  (no "mlp." prefix)
    // Qwen 3.x: {prefix}.mlp.experts.{i}.{proj}
    let expert_base_prefix = if artifact
        .tensors
        .has(&format!("{prefix}.experts.0.gate_proj.weight"))
    {
        format!("{prefix}.experts")
    } else {
        format!("{prefix}.mlp.experts")
    };

    let mut experts = Vec::with_capacity(num_experts);
    for expert_idx in 0..num_experts {
        let ep = format!("{expert_base_prefix}.{expert_idx}");
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
        // Qwen/Nemotron style NVFP4 shared expert at mlp.shared_expert.*
        let sp = format!("{prefix}.mlp.shared_expert");
        let gate_proj = CudaLinear::Nvfp4(cuda.load_nvfp4_linear_with_layout(
            artifact, &format!("{sp}.gate_proj"), store, residency, resident_layout, loader,
        )?);
        let up_proj = CudaLinear::Nvfp4(cuda.load_nvfp4_linear_with_layout(
            artifact, &format!("{sp}.up_proj"), store, residency, resident_layout, loader,
        )?);
        let down_proj = CudaLinear::Nvfp4(cuda.load_nvfp4_linear_with_layout(
            artifact, &format!("{sp}.down_proj"), store, residency, resident_layout, loader,
        )?);
        Some(CudaMoEShared { gate_proj, up_proj, down_proj })
    } else if artifact.tensors.has(&format!("{prefix}.mlp.gate_proj.weight")) {
        // Gemma 4 style: mlp.* is always-active shared expert (BF16). Force-VRAM —
        // shared MLP runs every layer (3 matvecs/layer) and the CPU path bottlenecks
        // the entire forward pass. Total VRAM cost: ~1 GB across 30 layers.
        let sp = format!("{prefix}.mlp");
        let gate_tensor = require_tensor(artifact, &format!("{sp}.gate_proj.weight"))?;
        let up_tensor = require_tensor(artifact, &format!("{sp}.up_proj.weight"))?;
        let down_tensor = require_tensor(artifact, &format!("{sp}.down_proj.weight"))?;
        let residency_store = cuda_residency_for_store(store, cuda.device_index())?;
        Some(CudaMoEShared {
            gate_proj: CudaLinear::Bf16(cuda.load_bf16_matrix_with_store_opts(gate_tensor, store, residency_store, loader, true)?),
            up_proj: CudaLinear::Bf16(cuda.load_bf16_matrix_with_store_opts(up_tensor, store, residency_store, loader, true)?),
            down_proj: CudaLinear::Bf16(cuda.load_bf16_matrix_with_store_opts(down_tensor, store, residency_store, loader, true)?),
        })
    } else {
        None
    };

    // Build the always-present device-resident per-expert scale buffer.
    // GPU `router_softmax_topk_device` always multiplies by this; if the model
    // has no per-expert scale we use an identity (all 1.0) so the kernel stays
    // branch-free.
    let per_expert_scale_data: Vec<f32> = match &router_per_expert_scale_host {
        Some(v) => v.clone(),
        None => vec![1.0_f32; num_experts],
    };
    let mut router_per_expert_scale_device = cuda.runtime().alloc_f32(num_experts)?;
    cuda.runtime()
        .upload_f32_slice_to_device(&per_expert_scale_data, &mut router_per_expert_scale_device)?;

    Ok(CudaMoE {
        router,
        router_input_scale,
        router_per_expert_scale_host,
        router_per_expert_scale_device,
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

/// Loads a projection as NVFP4 if `{prefix}.weight_scale` exists, otherwise as BF16.
#[allow(clippy::too_many_arguments)]
fn load_cuda_linear(
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
) -> Result<CudaLinear> {
    let has_scale = artifact.tensors.has(&format!("{prefix}.weight_scale"));
    if has_scale {
        let l = load_nvfp4_maybe_sliced_linear(
            cuda, artifact, prefix, store, residency, resident_layout,
            eff_rows, eff_logical_cols, is_sliced, loader,
        )?;
        Ok(CudaLinear::Nvfp4(l))
    } else {
        let tensor = require_tensor(artifact, &format!("{prefix}.weight"))?;
        // BF16 host-resident matvec routes through `matvec_bf16_host_resident_device`
        // which does D2H(input) → CPU rayon matmul → H2D(output). For Gemma 4 attention
        // (Q/K/V/O × 30 layers = 120 calls/token) this dominates wall time. Force these
        // weights into VRAM — total ~600 MB across all attention layers, well within
        // a free-VRAM budget of several GB. Routed-expert weights remain streamed
        // because they're an order of magnitude larger.
        let m = cuda.load_bf16_matrix_with_store_opts(tensor, store, residency, loader, true)?;
        Ok(CudaLinear::Bf16(m))
    }
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
        layer.q_proj.name(),
        layer.q_proj.rows(),
        layer.q_proj.cols(),
        q_width,
        hidden,
    )?;
    require_linear_shape(
        layer.k_proj.name(),
        layer.k_proj.rows(),
        layer.k_proj.cols(),
        kv_width,
        hidden,
    )?;
    require_linear_shape(
        layer.v_proj.name(),
        layer.v_proj.rows(),
        layer.v_proj.cols(),
        kv_width,
        hidden,
    )?;
    require_linear_shape(
        layer.o_proj.name(),
        layer.o_proj.rows(),
        layer.o_proj.cols(),
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

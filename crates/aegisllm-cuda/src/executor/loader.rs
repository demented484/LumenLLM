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
use super::state::{CudaLayer, CudaLinear, CudaMoE, CudaMoEExpert, CudaMoEShared, PleGlobal, PleLayer};
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
    shared_mlp_quantization: aegisllm_base::planning::placement::WeightQuantOverride,
    attention_quantization: aegisllm_base::planning::placement::WeightQuantOverride,
    attention_store_override: Option<aegisllm_base::planning::placement::StoragePlacement>,
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
    // Attention sublayers (Q/K/V/O) follow `attention.store` from config when
    // it overrides the enclosing transformer-block region's store.
    let attn_store = attention_store_override.unwrap_or(placement.store);
    let attn_residency = cuda_residency_for_store(attn_store, cuda.device_index())?;

    let is_moe = matches!(layer_kind, LayerKind::MoEDecoder { .. });
    let is_sliced = shape.is_sliced;

    // Effective dimensions used for sliced (MatFormer) loads.
    let hidden = shape.hidden_size;
    let q_width = shape.num_attention_heads * shape.head_dim;
    let kv_width = shape.num_kv_heads * shape.head_dim;
    let intermediate = shape.intermediate_size.unwrap_or(hidden);
    let layer_head_dim = shape.head_dim;
    let layer_num_kv_heads = shape.num_kv_heads;

    // Dense MLP load is format-aware (NVFP4 / BF16 / FP8) via `load_cuda_linear`.
    // For MoE layers the dense slots are stubbed; the real per-expert weights
    // are loaded in `load_cuda_moe` below.
    let (gate_proj, up_proj, down_proj) = if is_moe {
        (
            CudaLinear::Nvfp4(cuda.alloc_dummy_nvfp4_linear(&format!("{prefix}.mlp.gate_proj"))?),
            CudaLinear::Nvfp4(cuda.alloc_dummy_nvfp4_linear(&format!("{prefix}.mlp.up_proj"))?),
            CudaLinear::Nvfp4(cuda.alloc_dummy_nvfp4_linear(&format!("{prefix}.mlp.down_proj"))?),
        )
    } else {
        (
            load_cuda_linear(
                cuda, artifact, &format!("{prefix}.mlp.gate_proj"),
                placement.store, residency, resident_layout,
                intermediate, hidden, is_sliced, shared_mlp_quantization, loader,
            )?,
            load_cuda_linear(
                cuda, artifact, &format!("{prefix}.mlp.up_proj"),
                placement.store, residency, resident_layout,
                intermediate, hidden, is_sliced, shared_mlp_quantization, loader,
            )?,
            load_cuda_linear(
                cuda, artifact, &format!("{prefix}.mlp.down_proj"),
                placement.store, residency, resident_layout,
                hidden, intermediate, is_sliced, shared_mlp_quantization, loader,
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
            shared_mlp_quantization,
            attn_store,
            attn_residency,
            loader,
        )?))
    } else {
        None
    };

    // Qwen3-Next Gated DeltaNet layers replace self-attention with the GDN
    // mixer (detected by the presence of `linear_attn.in_proj_qkv.weight`). The
    // q/k/v/o slots become dummies; the MLP/MoE sublayer is loaded as usual.
    let is_gdn = artifact
        .tensors
        .has(&format!("{prefix}.linear_attn.in_proj_qkv.weight"));
    // Qwen3-Next uses zero-centered RMSNorm weights (norm·(1+weight)); fold the
    // +1 into the regular norms at load (NOT the GDN gated norm, which is plain).
    let qwen_unit_norm = {
        let mt = artifact.config.model_type.as_str();
        mt.contains("qwen3_5") || mt.contains("qwen3_next")
    };
    let gdn = if is_gdn {
        Some(Box::new(load_gdn(
            cuda, artifact, &prefix, attn_store, attn_residency, resident_layout,
            attention_quantization, loader,
        )?))
    } else {
        None
    };
    let (q_proj, k_proj, v_proj, qkv_proj, o_proj) = if is_gdn {
        (
            CudaLinear::Nvfp4(cuda.alloc_dummy_nvfp4_linear(&format!("{prefix}.self_attn.q_proj"))?),
            CudaLinear::Nvfp4(cuda.alloc_dummy_nvfp4_linear(&format!("{prefix}.self_attn.k_proj"))?),
            CudaLinear::Nvfp4(cuda.alloc_dummy_nvfp4_linear(&format!("{prefix}.self_attn.v_proj"))?),
            None,
            CudaLinear::Nvfp4(cuda.alloc_dummy_nvfp4_linear(&format!("{prefix}.self_attn.o_proj"))?),
        )
    } else {
        // Qwen3-Next full-attention output gate: q_proj outputs 2×q_width
        // ([query | gate] interleaved per head). Load the full gated width.
        let gated = artifact.config.attn_output_gate == Some(true);
        let q_proj_rows = if gated { 2 * q_width } else { q_width };
        let q_proj = load_cuda_linear(
            cuda, artifact, &format!("{prefix}.self_attn.q_proj"),
            attn_store, attn_residency, resident_layout,
            q_proj_rows, hidden, is_sliced, attention_quantization, loader,
        )?;
        let k_proj = load_cuda_linear(
            cuda, artifact, &format!("{prefix}.self_attn.k_proj"),
            attn_store, attn_residency, resident_layout,
            kv_width, hidden, is_sliced, attention_quantization, loader,
        )?;
        let v_proj = {
            // Gemma 4 global layers have attention_k_eq_v=true: no separate v_proj.
            let v_has_weight = artifact.tensors.has(&format!("{prefix}.self_attn.v_proj.weight"))
                || artifact.tensors.has(&format!("{prefix}.self_attn.v_proj.weight_scale"));
            let v_prefix = if v_has_weight {
                format!("{prefix}.self_attn.v_proj")
            } else {
                format!("{prefix}.self_attn.k_proj")
            };
            load_cuda_linear(
                cuda, artifact, &v_prefix,
                attn_store, attn_residency, resident_layout,
                kv_width, hidden, is_sliced, attention_quantization, loader,
            )?
        };
        let qkv_proj = if is_sliced || gated {
            // Gated attention can't use a fused QKV group (q is double-width).
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
        };
        let o_proj = load_cuda_linear(
            cuda, artifact, &format!("{prefix}.self_attn.o_proj"),
            attn_store, attn_residency, resident_layout,
            hidden, q_width, is_sliced, attention_quantization, loader,
        )?;
        (q_proj, k_proj, v_proj, qkv_proj, o_proj)
    };

    let layer_out = CudaLayer {
        input_norm_weight: {
            let b = cuda.load_dense_vector_with_store(
                require_tensor(artifact, &format!("{prefix}.input_layernorm.weight"))?,
                placement.store,
                loader,
            )?;
            if qwen_unit_norm { cuda.plus_one_norm(b)? } else { b }
        },
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
            let b = cuda.load_dense_vector_with_store(
                require_tensor(artifact, &pre_mlp_name)?,
                placement.store,
                loader,
            )?;
            if qwen_unit_norm { cuda.plus_one_norm(b)? } else { b }
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
        q_proj,
        k_proj,
        v_proj,
        qkv_proj,
        o_proj,
        // Gemma 4 / Qwen3-Next: per-head RMS norm on Q (between q_proj and RoPE).
        q_norm_weight: {
            let o = load_optional_norm_weight(
                cuda, artifact, placement.store, loader,
                &[&format!("{prefix}.self_attn.q_norm.weight")],
            )?;
            if qwen_unit_norm { o.map(|b| cuda.plus_one_norm(b)).transpose()? } else { o }
        },
        // Gemma 4 / Qwen3-Next: per-head RMS norm on K (between k_proj and RoPE).
        k_norm_weight: {
            let o = load_optional_norm_weight(
                cuda, artifact, placement.store, loader,
                &[&format!("{prefix}.self_attn.k_norm.weight")],
            )?;
            if qwen_unit_norm { o.map(|b| cuda.plus_one_norm(b)).transpose()? } else { o }
        },
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
        ple: load_ple_layer(cuda, artifact, &prefix, placement.store, residency, loader)?,
        // kv_shared_from is filled in by a post-load pass that has access to
        // the full layer list (and thus per-layer `layer_type`s for parent
        // resolution).
        kv_shared_from: None,
        gdn,
        attn_output_gate: !is_gdn && artifact.config.attn_output_gate == Some(true),
        dense_activation: {
            // Pick activation from the architecture descriptor: `silu` (Llama
            // / Qwen) → SwiGLU; `gelu_pytorch_tanh` (Gemma-4 E4B / 26B-A4B
            // dense layers) → GeGLU-tanh. Anything else is rejected at load
            // time rather than silently miscomputed at decode.
            use super::mlp::DenseActivation;
            match artifact.config.hidden_activation.as_deref() {
                None | Some("silu") | Some("swiglu") => DenseActivation::Swiglu,
                Some("gelu_pytorch_tanh") | Some("gelu_tanh") | Some("gelu") =>
                    DenseActivation::GeluTanh,
                Some(other) => return Err(AegisError::InvalidPlan(format!(
                    "dense MLP: unsupported hidden_activation `{other}` — supported: silu/swiglu, gelu_pytorch_tanh"
                ))),
            }
        },
    };
    if !is_moe && !is_gdn {
        validate_cuda_layer_shape(&layer_out, shape)?;
    }
    Ok(layer_out)
}

/// Load the Qwen3-Next Gated DeltaNet mixer weights for one layer.
#[allow(clippy::too_many_arguments)]
fn load_gdn(
    cuda: &CudaWeightLoader<'_>,
    artifact: &ModelArtifact,
    prefix: &str,
    store: StoragePlacement,
    residency: TensorResidencyPlan,
    resident_layout: LinearResidentLayout,
    quant_override: aegisllm_base::planning::placement::WeightQuantOverride,
    loader: &mut aegisllm_base::tensor::storage::TensorStorageLoader,
) -> Result<super::gdn::CudaGdn> {
    let cfg = &artifact.config;
    let n_k = cfg.linear_num_key_heads.ok_or_else(|| {
        AegisError::InvalidPlan("GDN: missing linear_num_key_heads".into())
    })?;
    let d_k = cfg.linear_key_head_dim.ok_or_else(|| {
        AegisError::InvalidPlan("GDN: missing linear_key_head_dim".into())
    })?;
    let n_v = cfg.linear_num_value_heads.unwrap_or(n_k);
    let d_v = cfg.linear_value_head_dim.unwrap_or(d_k);
    let kc = cfg.linear_conv_kernel_dim.unwrap_or(4);
    let dims = super::gdn::GdnDims {
        num_k_heads: n_k,
        num_v_heads: n_v,
        k_head_dim: d_k,
        v_head_dim: d_v,
        conv_kernel: kc,
    };
    let qkv_out = 2 * n_k * d_k + n_v * d_v;
    let zp = format!("{prefix}.linear_attn");
    let hidden = artifact.config.hidden_size;
    let in_proj_qkv = load_cuda_linear(
        cuda, artifact, &format!("{zp}.in_proj_qkv"),
        store, residency, resident_layout, qkv_out, hidden, false, quant_override, loader,
    )?;
    let in_proj_z = load_cuda_linear(
        cuda, artifact, &format!("{zp}.in_proj_z"),
        store, residency, resident_layout, n_v * d_v, hidden, false, quant_override, loader,
    )?;
    let in_proj_b = load_cuda_linear(
        cuda, artifact, &format!("{zp}.in_proj_b"),
        store, residency, resident_layout, n_v, hidden, false, quant_override, loader,
    )?;
    let in_proj_a = load_cuda_linear(
        cuda, artifact, &format!("{zp}.in_proj_a"),
        store, residency, resident_layout, n_v, hidden, false, quant_override, loader,
    )?;
    let out_proj = load_cuda_linear(
        cuda, artifact, &format!("{zp}.out_proj"),
        store, residency, resident_layout, hidden, n_v * d_v, false, quant_override, loader,
    )?;
    let conv1d_weight = cuda.load_dense_vector_with_store(
        require_tensor(artifact, &format!("{zp}.conv1d.weight"))?, store, loader,
    )?;
    let a_log = cuda.load_dense_vector_with_store(
        require_tensor(artifact, &format!("{zp}.A_log"))?, store, loader,
    )?;
    let dt_bias = cuda.load_dense_vector_with_store(
        require_tensor(artifact, &format!("{zp}.dt_bias"))?, store, loader,
    )?;
    let norm_weight = cuda.load_dense_vector_with_store(
        require_tensor(artifact, &format!("{zp}.norm.weight"))?, store, loader,
    )?;
    Ok(super::gdn::CudaGdn {
        in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj,
        conv1d_weight, a_log, dt_bias, norm_weight, dims,
    })
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
    shared_mlp_quantization: aegisllm_base::planning::placement::WeightQuantOverride,
    // Store for the BF16 dense parts of the MoE block (router + shared MLP).
    // Defaults to `store` when no `attention.store` override is set; when
    // the user sets `attention.store=vram` we route shared MLP and routers
    // there too because the BF16 batched matmul kernel doesn't support
    // host-resident weights. Routed-expert NVFP4 weights still follow the
    // main `store` because they have a per-call H2D streaming path.
    bf16_dense_store: StoragePlacement,
    bf16_dense_residency: TensorResidencyPlan,
    loader: &mut aegisllm_base::tensor::storage::TensorStorageLoader,
) -> Result<CudaMoE> {
    let prefix = format!("{text_prefix}layers.{layer}");

    // Router weight matrix [num_experts, hidden_size] — BF16
    // Gemma 4: {prefix}.router.proj.weight
    // Qwen 3.x: {prefix}.mlp.router_logits.weight
    // Qwen3-Next: {prefix}.mlp.gate.weight
    // Mixtral-style: {prefix}.block_sparse_moe.gate.weight
    let router_tensor = first_existing_tensor(
        artifact,
        &[
            &format!("{prefix}.router.proj.weight"),
            &format!("{prefix}.mlp.router_logits.weight"),
            &format!("{prefix}.mlp.gate.weight"),
            &format!("{prefix}.block_sparse_moe.gate.weight"),
        ],
    )?;
    // Router is BF16 [num_experts, hidden_size]. Follows the BF16-dense store
    // (defaults to attention.store override) because host-resident BF16
    // matvec routes through the slow CPU rayon path.
    let router = cuda.load_bf16_matrix_with_store(
        router_tensor,
        bf16_dense_store,
        bf16_dense_residency,
        loader,
    )?;

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
    let t_experts = std::time::Instant::now();

    // Parallel-prefetch fast path: when experts are host-resident with
    // the default packed-source layout (no native-MXFP4 / cutlass-NVFP4
    // repack), dispatch all `num_experts × 3` tensor reads to rayon
    // workers concurrently via `prefetch_host_nvfp4_pairs_par`. The
    // arena's atomic bump-pointer alloc lets workers write disjoint
    // slots without coordination; per-tensor reads themselves use 8-way
    // chunked pread, so even one big tensor saturates the NVMe. Outer
    // rayon parallelism then hides per-tensor `File::open` overhead
    // across the ~384 expert tensors per layer. Combined this brings
    // load throughput from ~500 MB/s (QD=1) to multi-GB/s.
    // Host-resident (streamed) experts read packed NVFP4 bytes and run the
    // prequantized GEMV (native_mxfp4 is None for the streamed path), so the
    // planned NativeTensorCore on-device repack is irrelevant here — those bytes
    // are the same PackedSource layout. Treat both as prefetch-eligible so the
    // 26B experts (planned NativeTensorCore) use the PARALLEL shard-buffered
    // prefetch instead of the serial single-threaded per-expert path (~33s).
    let prefetch_eligible = matches!(residency, TensorResidencyPlan::StagedHostToDevice { .. })
        && matches!(
            resident_layout,
            LinearResidentLayout::PackedSource | LinearResidentLayout::NativeTensorCore
        )
        && cuda.arena().is_some();
    if prefetch_eligible {
        let mut prefixes = Vec::with_capacity(num_experts * 3);
        for expert_idx in 0..num_experts {
            let ep = format!("{expert_base_prefix}.{expert_idx}");
            prefixes.push(format!("{ep}.gate_proj"));
            prefixes.push(format!("{ep}.up_proj"));
            prefixes.push(format!("{ep}.down_proj"));
        }
        cuda.report_status(&format!("layer {layer}: prefetch experts (host)"));
        let prefetched = cuda.prefetch_host_nvfp4_pairs_par(artifact, &prefixes)?;
        let mut iter = prefetched.into_iter();
        let mut prefix_iter = prefixes.iter();
        for expert_idx in 0..num_experts {
            if expert_idx % 16 == 0 {
                cuda.report_status(&format!(
                    "layer {layer}: experts {expert_idx}/{num_experts}"
                ));
            }
            let gate_p = prefix_iter.next().expect("prefix available");
            let (gate_packed, gate_scales) = iter.next().expect("prefetched gate");
            let gate_proj = cuda.finalize_host_nvfp4_with_prefetched(
                artifact, gate_p, residency, store, gate_packed, gate_scales, loader,
            )?;
            let up_p = prefix_iter.next().expect("prefix available");
            let (up_packed, up_scales) = iter.next().expect("prefetched up");
            let up_proj = cuda.finalize_host_nvfp4_with_prefetched(
                artifact, up_p, residency, store, up_packed, up_scales, loader,
            )?;
            let down_p = prefix_iter.next().expect("prefix available");
            let (down_packed, down_scales) = iter.next().expect("prefetched down");
            let down_proj = cuda.finalize_host_nvfp4_with_prefetched(
                artifact, down_p, residency, store, down_packed, down_scales, loader,
            )?;
            experts.push(CudaMoEExpert { gate_proj, up_proj, down_proj });
        }
    } else {
        for expert_idx in 0..num_experts {
            if expert_idx % 8 == 0 {
                cuda.report_status(&format!(
                    "layer {layer}: experts {expert_idx}/{num_experts}"
                ));
            }
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
    }
    // Skip per-layer expert timing on a TTY so it doesn't break the
    // in-place progress bar (the bar already shows expert progress via
    // `cuda.report_status(...)` ticks). Logs and pipes still get the
    // detailed per-layer breakdown.
    if !std::io::IsTerminal::is_terminal(&std::io::stderr()) {
        eprintln!(
            "load-timing:   experts ({} × 3 NVFP4)  {:>5.2}s",
            num_experts,
            t_experts.elapsed().as_secs_f64()
        );
    }

    let expert_intermediate_size = experts
        .first()
        .map(|e| e.gate_proj.rows)
        .ok_or_else(|| AegisError::InvalidPlan("MoE layer has no experts".into()))?;

    // Optional shared (always-active) expert
    let shared_expert = if has_shared_expert {
        // Qwen/Nemotron style NVFP4 shared expert at mlp.shared_expert.*
        // The shared expert runs on EVERY token (unlike the sparse routed
        // experts), so it must be VRAM-resident — route it to the dense store
        // (`attention.store`, VRAM) not the streamed routed-expert `store`.
        // Streaming it per token was a large chunk of the decode H2D volume.
        let sp = format!("{prefix}.mlp.shared_expert");
        let gate_proj = CudaLinear::Nvfp4(cuda.load_nvfp4_linear_with_layout(
            artifact, &format!("{sp}.gate_proj"), bf16_dense_store, bf16_dense_residency.clone(), resident_layout, loader,
        )?);
        let up_proj = CudaLinear::Nvfp4(cuda.load_nvfp4_linear_with_layout(
            artifact, &format!("{sp}.up_proj"), bf16_dense_store, bf16_dense_residency.clone(), resident_layout, loader,
        )?);
        let down_proj = CudaLinear::Nvfp4(cuda.load_nvfp4_linear_with_layout(
            artifact, &format!("{sp}.down_proj"), bf16_dense_store, bf16_dense_residency.clone(), resident_layout, loader,
        )?);
        // Qwen3-Next: sigmoid gate on the shared expert (`shared_expert_gate`).
        let shared_gate = artifact
            .tensors
            .get(&format!("{prefix}.mlp.shared_expert_gate.weight"))
            .map(|t| cuda.load_bf16_matrix_with_store(t, bf16_dense_store, bf16_dense_residency.clone(), loader))
            .transpose()?;
        Some(CudaMoEShared { gate_proj, up_proj, down_proj, gate_up_fused: None, shared_gate })
    } else if artifact.tensors.has(&format!("{prefix}.mlp.gate_proj.weight")) {
        // Gemma 4 style: mlp.* is always-active shared expert. Stored as
        // BF16 in the checkpoint; the user can opt into a smaller format
        // via `hidden-layers.shared-MLP-quantization`. Default keeps BF16
        // (force-VRAM since the CPU rayon path is the only host-resident
        // alternative and bottlenecks everything).
        let sp = format!("{prefix}.mlp");
        let gate_tensor = require_tensor(artifact, &format!("{sp}.gate_proj.weight"))?;
        let up_tensor   = require_tensor(artifact, &format!("{sp}.up_proj.weight"))?;
        let down_tensor = require_tensor(artifact, &format!("{sp}.down_proj.weight"))?;
        use aegisllm_base::planning::placement::WeightQuantOverride as Wq;
        match shared_mlp_quantization {
            Wq::Default => {
                // Shared MLP is BF16 dense — uses the dense override.
                // Load gate and up individually (force-VRAM by `bf16_dense_*`)
                // then fuse them into a single row-stacked `[2*intermediate,
                // hidden]` matrix so the runtime hot path can use one cuBLASLt
                // BF16 GEMM + strided GeGLU instead of two GEMMs + standalone
                // GeGLU. Per-MoE-layer: 4 BF16 launches → 3.
                let gate_mat = cuda.load_bf16_matrix_with_store(
                    gate_tensor, bf16_dense_store, bf16_dense_residency, loader,
                )?;
                let up_mat = cuda.load_bf16_matrix_with_store(
                    up_tensor, bf16_dense_store, bf16_dense_residency, loader,
                )?;
                let down_mat = cuda.load_bf16_matrix_with_store(
                    down_tensor, bf16_dense_store, bf16_dense_residency, loader,
                )?;
                let (fused, gate_stub, up_stub) = cuda.fuse_bf16_gate_up(gate_mat, up_mat)?;
                Some(CudaMoEShared {
                    gate_proj: CudaLinear::Bf16(gate_stub),
                    up_proj:   CudaLinear::Bf16(up_stub),
                    down_proj: CudaLinear::Bf16(down_mat),
                    gate_up_fused: Some(fused),
                    shared_gate: None,
                })
            }
            Wq::Fp8 => Some(CudaMoEShared {
                gate_proj: CudaLinear::Fp8(cuda.load_bf16_as_fp8_linear(gate_tensor, loader)?),
                up_proj:   CudaLinear::Fp8(cuda.load_bf16_as_fp8_linear(up_tensor,   loader)?),
                down_proj: CudaLinear::Fp8(cuda.load_bf16_as_fp8_linear(down_tensor, loader)?),
                gate_up_fused: None,
                shared_gate: None,
            }),
            other => return Err(AegisError::Unsupported(format!(
                "shared-MLP-quantization={other:?} not yet wired into the loader"
            ))),
        }
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
        // Populated post-load in `full.rs` when the expert arena is
        // device-mapped (AEGIS_GPU_DRIVEN_MOE). Defaults to host-streamed decode.
        device_tables: None,
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

/// Loads a projection as NVFP4 if `{prefix}.weight_scale` exists, otherwise as
/// BF16 (or FP8 when `quant_override == Fp8`). NVFP4 layers ignore the quant
/// override since they're already 4-bit.
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
    quant_override: aegisllm_base::planning::placement::WeightQuantOverride,
    loader: &mut aegisllm_base::tensor::storage::TensorStorageLoader,
) -> Result<CudaLinear> {
    let has_scale = artifact.tensors.has(&format!("{prefix}.weight_scale"));
    if has_scale {
        let l = load_nvfp4_maybe_sliced_linear(
            cuda, artifact, prefix, store, residency, resident_layout,
            eff_rows, eff_logical_cols, is_sliced, loader,
        )?;
        return Ok(CudaLinear::Nvfp4(l));
    }
    let tensor = require_tensor(artifact, &format!("{prefix}.weight"))?;
    // DeepSeek-style FP8 block-scaled (F8_E4M3 weight + `weight_scale_inv`
    // [rows/128, cols/128]): keep the FP8 weights in VRAM (9 GB fits the 16 GB
    // GPU vs 18 GB dequanted) and dequant-on-the-fly in the matvec. Covers
    // Qwen3.5-9B-FP8 (dense) — every linear routes here.
    if tensor.dtype == aegisllm_base::tensor::TensorDType::F8E4M3 {
        let scale = require_tensor(artifact, &format!("{prefix}.weight_scale_inv"))?;
        let l = cuda.load_fp8_block_linear(tensor, scale, loader)?;
        return Ok(CudaLinear::Fp8(l));
    }
    use aegisllm_base::planning::placement::WeightQuantOverride as Wq;
    match quant_override {
        Wq::Default => {
            // BF16 host-resident matvec routes through `matvec_bf16_host_resident_device`
            // which does D2H(input) → CPU rayon matmul → H2D(output). For Gemma 4 attention
            // (Q/K/V/O × 30 layers = 120 calls/token) this dominates wall time. Force these
            // weights into VRAM — total ~600 MB across all attention layers, well within
            // a free-VRAM budget of several GB. Routed-expert weights remain streamed
            // because they're an order of magnitude larger.
            let m = cuda.load_bf16_matrix_with_store(tensor, store, residency, loader)?;
            Ok(CudaLinear::Bf16(m))
        }
        Wq::Fp8 => Ok(CudaLinear::Fp8(cuda.load_bf16_as_fp8_linear(tensor, loader)?)),
        other => Err(AegisError::Unsupported(format!(
            "attention-quantization={other:?} not yet wired into the loader"
        ))),
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
    // Derive the MLP intermediate from THIS layer's own gate_proj rather than
    // the model-wide scalar. Gemma-4 E2B uses `use_double_wide_mlp`: layers
    // 0-14 have intermediate 6144, layers 15-34 have 12288. The runtime MLP
    // math is already per-layer correct (GEMM dims come from each loaded
    // matrix; activations are element-wise; scratch is sized to the max
    // intermediate across layers in full.rs). The only thing that rejected the
    // wide layers was this validation comparing against the scalar — so we
    // instead check the three MLP projections are mutually consistent
    // (up.rows == gate.rows, down.cols == gate.rows) plus hidden on the other
    // axis. Uniform models (E4B, Llama, Qwen) are unaffected: gate.rows equals
    // their scalar intermediate.
    let intermediate = layer.gate_proj.rows();

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
        // Qwen3-Next full-attention output gate: q_proj emits 2×q_width
        // ([query | gate] per head), so the loaded matrix is the gated width.
        if layer.attn_output_gate { 2 * q_width } else { q_width },
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
        layer.gate_proj.name(),
        layer.gate_proj.rows(),
        layer.gate_proj.cols(),
        intermediate,
        hidden,
    )?;
    require_linear_shape(
        layer.up_proj.name(),
        layer.up_proj.rows(),
        layer.up_proj.cols(),
        intermediate,
        hidden,
    )?;
    require_linear_shape(
        layer.down_proj.name(),
        layer.down_proj.rows(),
        layer.down_proj.cols(),
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

/// Load global PLE state (embed table + model projection + projection norm)
/// from the artifact. Returns `None` when the model has no PLE config
/// (Gemma-4-26B-A4B-NVFP4 and earlier targets). Returns `Some(PleGlobal)`
/// for Gemma-4 E4B / E2B and any future checkpoint that exposes
/// `hidden_size_per_layer_input` in its top-level config.
///
/// The 5.4 GiB embed_table is stored host-resident (mmap-streamed per
/// token) to keep the GPU VRAM budget viable on 16 GiB cards. The smaller
/// model projection + norm sit on VRAM alongside the model weights.
pub(super) fn load_ple_global(
    cuda: &CudaWeightLoader<'_>,
    artifact: &ModelArtifact,
    device_index: usize,
    text_prefix: &str,
    loader: &mut aegisllm_base::tensor::storage::TensorStorageLoader,
) -> Result<Option<PleGlobal>> {
    let ple_dim = match artifact.config.hidden_size_per_layer_input {
        Some(d) if d > 0 => d,
        _ => return Ok(None),
    };
    let hidden = artifact.config.hidden_size;
    let table_store = StoragePlacement::Ram;
    let table_residency = cuda_residency_for_store(table_store, device_index)?;
    let proj_store = StoragePlacement::Vram { device: device_index };
    let proj_residency = cuda_residency_for_store(proj_store, device_index)?;

    let embed_table = cuda.load_bf16_matrix_with_store(
        require_tensor(artifact, &format!("{text_prefix}embed_tokens_per_layer.weight"))?,
        table_store, table_residency, loader,
    )?;
    let model_projection = cuda.load_bf16_matrix_with_store(
        require_tensor(artifact, &format!("{text_prefix}per_layer_model_projection.weight"))?,
        proj_store, proj_residency, loader,
    )?;
    let projection_norm = cuda.load_dense_vector_with_store(
        require_tensor(artifact, &format!("{text_prefix}per_layer_projection_norm.weight"))?,
        proj_store, loader,
    )?;
    let embed_scale_per_layer = (ple_dim as f32).sqrt();
    let model_projection_scale = 1.0 / (hidden as f32).sqrt();
    let combine_scale = 1.0 / (2.0_f32).sqrt();
    Ok(Some(PleGlobal {
        embed_table,
        model_projection,
        projection_norm,
        ple_dim,
        embed_scale_per_layer,
        model_projection_scale,
        combine_scale,
    }))
}

/// Load per-layer PLE weights (gate / projection / post_norm). Returns
/// `None` when the model has no PLE config, mirroring `load_ple_global`.
///
/// PLE per-layer linears are tiny (256×2560 BF16 = 1.3 MB each × 3 × 42
/// layers ≈ 160 MB total at E4B), so they're forced VRAM-resident regardless
/// of the enclosing `hidden-layers.store` config — host-resident BF16 GEMM
/// would route through the CPU rayon path which is far too slow for the
/// per-layer / per-decode-step call pattern.
pub(super) fn load_ple_layer(
    cuda: &CudaWeightLoader<'_>,
    artifact: &ModelArtifact,
    prefix: &str,
    _layer_store: StoragePlacement,
    _layer_residency: TensorResidencyPlan,
    loader: &mut aegisllm_base::tensor::storage::TensorStorageLoader,
) -> Result<Option<PleLayer>> {
    if artifact.config.hidden_size_per_layer_input.unwrap_or(0) == 0 {
        return Ok(None);
    }
    let vram_store = StoragePlacement::Vram { device: cuda.device_index() };
    let vram_residency = cuda_residency_for_store(vram_store, cuda.device_index())?;
    let input_gate = cuda.load_bf16_matrix_with_store(
        require_tensor(artifact, &format!("{prefix}.per_layer_input_gate.weight"))?,
        vram_store, vram_residency.clone(), loader,
    )?;
    let projection = cuda.load_bf16_matrix_with_store(
        require_tensor(artifact, &format!("{prefix}.per_layer_projection.weight"))?,
        vram_store, vram_residency, loader,
    )?;
    let post_norm = cuda.load_dense_vector_with_store(
        require_tensor(artifact, &format!("{prefix}.post_per_layer_input_norm.weight"))?,
        vram_store, loader,
    )?;
    Ok(Some(PleLayer { input_gate, projection, post_norm }))
}

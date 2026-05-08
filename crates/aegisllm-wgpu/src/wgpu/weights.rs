//! Model weights uploaded to wgpu storage buffers.
//!
//! Bridges the host-side `ModelArtifact` (safetensors-derived tensor
//! registry on disk + RAM) to the `wgpu::Buffer` form the device-resident
//! forward path consumes. For the first iteration we keep it simple:
//!
//! * BF16 weights → upcast to f32 at load time, uploaded as f32 storage
//!   buffers. WGSL has no native bf16 type, so this saves us writing
//!   bit-twiddle decoders inside every matmul shader; the cost is 2× VRAM
//!   for BF16 weights vs. native, which is fine for the synthetic tests
//!   that drive the orchestration; real Gemma-4 will later use a packed
//!   `array<u32>` storage + per-element decode shader.
//! * F32 weights → uploaded directly.
//! * NVFP4 weights → uploaded as a (packed, scales) buffer pair using the
//!   existing [`upload_padded_u8_buf`] helper. Decode happens on-device
//!   via [`dequant_nvfp4_device`] before being consumed by the matmul.
//!
//! Tensor naming matches the safetensors layout produced by the
//! Hugging Face `transformers` checkpoint format that aegisllm targets:
//!   * `model.embed_tokens.weight`, `model.norm.weight`, `lm_head.weight`
//!   * `model.layers.{L}.input_layernorm.weight` etc.
//!   * `model.layers.{L}.self_attn.q_proj.weight` etc.
//!   * NVFP4 weights pair with `{name}.weight_scale` for UE4M3 block scales.
//!
//! Shape conventions match what the `_device` primitives expect:
//!   * Attention `q_proj` is `[num_q_heads * head_dim, hidden_size]`.
//!   * Attention `k_proj` / `v_proj` is `[num_kv_heads * head_dim, hidden_size]`.
//!   * Attention `o_proj` is `[hidden_size, num_q_heads * head_dim]`.
//!   * MLP `gate_proj` / `up_proj` is `[intermediate_size, hidden_size]`.
//!   * MLP `down_proj` is `[hidden_size, intermediate_size]`.

use std::sync::Arc;

use aegisllm_base::artifact::ModelArtifact;
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::planning::placement::StoragePlacement;
use aegisllm_base::tensor::core::TensorDType;
use aegisllm_base::tensor::storage::TensorStorageLoader;

use super::forward::{upload_bf16_packed_buf, upload_f32_buf, upload_padded_u8_buf};
use super::loader::WgpuContext;

/// A linear weight matrix uploaded to the device. Either dense f32,
/// BF16 stored as packed `array<u32>` (decoded on the fly via
/// `dequant_bf16_device`), or NVFP4 packed.
pub enum WgpuLinear {
    /// f32 row-major `[rows, cols]`.
    Dense {
        weight: wgpu::Buffer,
        rows: usize,
        cols: usize,
    },
    /// BF16 packed: each u32 holds 2 BF16 values. Total buffer size =
    /// `(rows * cols + 1) / 2 * 4` bytes. Saves ~50 % VRAM vs `Dense`
    /// when the source weight is BF16 — required to fit Gemma-4-26B
    /// shared-MLP / embedding tensors in 16 GiB VRAM.
    Bf16Packed {
        weight: wgpu::Buffer,
        rows: usize,
        cols: usize,
    },
    /// NVFP4: packed nibbles `[rows, cols/2]` bytes + UE4M3 scales
    /// `[rows, cols/16]` bytes + per-tensor `output_scale`. Decoded on
    /// device via `dequant_nvfp4_device` into an f32 scratch before use.
    Nvfp4 {
        packed: wgpu::Buffer,
        scales: wgpu::Buffer,
        rows: usize,
        cols: usize,
        output_scale: f32,
    },
}

impl std::fmt::Debug for WgpuLinear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dense { rows, cols, .. } => f
                .debug_struct("WgpuLinear::Dense")
                .field("rows", rows)
                .field("cols", cols)
                .finish(),
            Self::Bf16Packed { rows, cols, .. } => f
                .debug_struct("WgpuLinear::Bf16Packed")
                .field("rows", rows)
                .field("cols", cols)
                .finish(),
            Self::Nvfp4 { rows, cols, output_scale, .. } => f
                .debug_struct("WgpuLinear::Nvfp4")
                .field("rows", rows)
                .field("cols", cols)
                .field("output_scale", output_scale)
                .finish(),
        }
    }
}

impl WgpuLinear {
    pub fn rows(&self) -> usize {
        match self {
            Self::Dense { rows, .. }
            | Self::Bf16Packed { rows, .. }
            | Self::Nvfp4 { rows, .. } => *rows,
        }
    }
    pub fn cols(&self) -> usize {
        match self {
            Self::Dense { cols, .. }
            | Self::Bf16Packed { cols, .. }
            | Self::Nvfp4 { cols, .. } => *cols,
        }
    }
}

/// Per-layer attention weights for a vanilla Llama / Gemma-style block.
/// Gemma-4 specifics (per-head Q/K norm, sub-layer norms) will land
/// alongside the Gemma-4 forward variant; this is the minimum-viable set
/// that drives the existing `forward_attention_block_device`.
pub struct WgpuAttentionWeightsFull {
    pub norm_weight: wgpu::Buffer,
    pub q_proj: WgpuLinear,
    pub k_proj: WgpuLinear,
    pub v_proj: WgpuLinear,
    pub o_proj: WgpuLinear,
    /// Gemma-4: per-head Q norm `[head_dim]` applied between Q proj and
    /// RoPE. `None` for vanilla Llama (no Q norm).
    pub q_norm: Option<wgpu::Buffer>,
    /// Gemma-4: per-head K norm `[head_dim]` applied between K proj and
    /// RoPE. `None` for vanilla Llama.
    pub k_norm: Option<wgpu::Buffer>,
    /// Gemma-4: when `q_norm` is present, V also gets a per-head RMS
    /// norm with no learned weight (the shader binds an all-ones
    /// vector of length `head_dim`). Buffer is allocated once per
    /// model and reused across layers; `None` for vanilla Llama.
    pub v_norm_unit: Option<wgpu::Buffer>,
    /// Gemma-4 PrePost: full-RMS-norm `[hidden_size]` applied to
    /// the attention block output BEFORE adding to the residual.
    /// `None` for vanilla Llama.
    pub post_attn_sublayer_norm: Option<wgpu::Buffer>,
}

impl std::fmt::Debug for WgpuAttentionWeightsFull {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuAttentionWeightsFull").finish_non_exhaustive()
    }
}

/// One routed-expert weight bundle. Gemma-4 has 128 of these per
/// MoE layer; top-k=2 means each token activates 2.
pub struct WgpuMoeExpert {
    pub gate_proj: WgpuLinear,
    pub up_proj: WgpuLinear,
    pub down_proj: WgpuLinear,
}

impl std::fmt::Debug for WgpuMoeExpert {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuMoeExpert").finish_non_exhaustive()
    }
}

/// Mixture-of-Experts weights for one Gemma-4 transformer layer.
///
/// Gemma-4 every layer is MoE (no dense-only layers). The router projects
/// the post-norm input to `num_experts` logits, top-k indices are selected
/// (currently on host — future: GPU softmax+topk shader). Each token
/// activates `top_k` routed experts plus an always-active shared expert.
pub struct WgpuMoeWeights {
    /// `[num_experts, hidden_size]` router projection. Maps the router
    /// input to per-expert logits.
    pub router: WgpuLinear,
    /// Optional `[hidden_size]` element-wise scale applied to the
    /// router input BEFORE projection. Gemma-4 sets this; vanilla MoE
    /// architectures may leave it `None`.
    pub router_input_scale: Option<wgpu::Buffer>,
    /// Host-side `[num_experts]` calibration scale applied to the
    /// top-k routing weights AFTER softmax (`top_k_w[i] *=
    /// per_expert_scale[indices[i]]`). Gemma-4-specific. The host-side
    /// representation is a Vec because top-k indexing happens on host.
    pub per_expert_scale: Vec<f32>,
    /// All routed experts. Length must equal `num_experts`.
    pub experts: Vec<WgpuMoeExpert>,
    /// Always-active shared expert (Gemma-4 has one). Run unconditionally
    /// alongside the routed experts and combined into the layer output.
    pub shared_expert: Option<WgpuMlpWeightsFull>,
    pub num_experts: usize,
    pub top_k: usize,
    pub intermediate_size: usize,
    /// Gemma-4: pre-norm applied to the residual specifically for the
    /// routed-expert stream's input (separate from the shared expert's
    /// input). When `None`, the routed experts reuse the same pre-FFN
    /// normed input as the shared expert.
    pub pre_feedforward_layernorm_2: Option<wgpu::Buffer>,
    /// Gemma-4: post-norm on the shared-expert stream output before
    /// combining with routed experts.
    pub post_feedforward_layernorm_1: Option<wgpu::Buffer>,
    /// Gemma-4: post-norm on the routed-expert stream output before
    /// combining with the shared stream.
    pub post_feedforward_layernorm_2: Option<wgpu::Buffer>,
}

impl std::fmt::Debug for WgpuMoeWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuMoeWeights")
            .field("num_experts", &self.num_experts)
            .field("top_k", &self.top_k)
            .field("intermediate_size", &self.intermediate_size)
            .field("has_shared_expert", &self.shared_expert.is_some())
            .finish()
    }
}

/// Per-layer dense MLP weights (gate/up/down + post-attn norm).
/// MoE layers will use a different struct that holds router + per-expert
/// weights; this is the simple path.
pub struct WgpuMlpWeightsFull {
    pub norm_weight: wgpu::Buffer,
    pub gate_proj: WgpuLinear,
    pub up_proj: WgpuLinear,
    pub down_proj: WgpuLinear,
    /// Gemma-4 PrePost: full-RMS-norm `[hidden_size]` applied to the
    /// MLP block output BEFORE adding to the residual. `None` for
    /// vanilla Llama.
    pub post_mlp_sublayer_norm: Option<wgpu::Buffer>,
}

impl std::fmt::Debug for WgpuMlpWeightsFull {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuMlpWeightsFull").finish_non_exhaustive()
    }
}

/// Full weights for one Llama-style layer.
pub struct WgpuLayerWeights {
    pub attention: WgpuAttentionWeightsFull,
    pub mlp: WgpuMlpWeightsFull,
    /// When present, this layer is MoE: `mlp` holds the shared-expert
    /// fields (or is unused if Gemma-4 routes the shared expert through
    /// `WgpuMoeWeights::shared_expert` instead) and `moe` carries the
    /// router + routed experts. `None` for vanilla Llama.
    pub moe: Option<WgpuMoeWeights>,
    /// Gemma-4: per-layer multiplicative scalar applied after the MLP
    /// block's residual add. `None` for vanilla Llama.
    pub layer_scalar: Option<f32>,
    /// Gemma-4 sliding-window layers cap attention to the most recent
    /// `window_size` positions. `None` means full causal attention
    /// (Gemma-4 global layers and vanilla Llama).
    pub attention_window_size: Option<u32>,
    /// Gemma-4 global layers may use a different head_dim (512) than
    /// the model's "default" sliding head_dim (256). When `Some`,
    /// overrides the model-level head_dim for THIS layer's attention.
    pub head_dim_override: Option<usize>,
    /// Gemma-4 global layers use a different num_kv_heads (2) than
    /// the model's default sliding count (8). When `Some`, overrides
    /// `model.num_kv_heads` for this layer.
    pub num_kv_heads_override: Option<usize>,
}

impl std::fmt::Debug for WgpuLayerWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuLayerWeights").finish_non_exhaustive()
    }
}

/// Whole-model weights uploaded to wgpu storage. Layer 0 lives at index 0,
/// layer N-1 at index N-1.
pub struct WgpuModel {
    pub ctx: Arc<WgpuContext>,
    /// Token embedding table. `WgpuLinear::Dense` (f32) for vanilla
    /// loads; `WgpuLinear::Bf16Packed` for Gemma-4 (saves ~50 % VRAM
    /// on the 2.95 GiB → 1.48 GiB embedding). The provider's row
    /// lookup dispatches the right path based on the variant.
    pub embed_tokens: WgpuLinear,
    pub embed_tokens_rows: usize,
    pub embed_tokens_cols: usize,
    pub final_norm: wgpu::Buffer,
    pub lm_head: WgpuLinear,
    pub layers: Vec<WgpuLayerWeights>,
    /// Architectural shapes carried alongside so the forward orchestration
    /// doesn't need to re-derive them from buffer sizes.
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    /// Gemma-4: scalar multiplier applied to embeddings after lookup
    /// (`sqrt(hidden_size)` for Gemma-4). `None` for vanilla Llama.
    pub embed_scale: Option<f32>,
    /// True when `lm_head.weight` is absent and the lm_head matmul
    /// should reuse the embedding table directly (tie_word_embeddings).
    /// Saves N GiB of VRAM for large vocab models. When `true`,
    /// `lm_head` is a placeholder and the forward path uses
    /// `embed_tokens` for the final matmul.
    pub lm_head_tied: bool,
}

impl std::fmt::Debug for WgpuModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuModel")
            .field("num_layers", &self.layers.len())
            .field("hidden_size", &self.hidden_size)
            .field("intermediate_size", &self.intermediate_size)
            .field("num_q_heads", &self.num_q_heads)
            .field("num_kv_heads", &self.num_kv_heads)
            .field("head_dim", &self.head_dim)
            .field("vocab_size", &self.vocab_size)
            .finish()
    }
}

/// Architectural shapes the loader expects to drive a Llama-style model.
/// In a real load these come from the artifact's `config.json` /
/// `ModelGraph`; the loader takes them as input rather than re-parsing,
/// to keep this module free of artifact-format wrangling.
#[derive(Debug, Clone, Copy)]
pub struct WgpuModelShape {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
}

/// Convert raw BF16 bytes (`u16` little-endian) to f32 by zero-extending
/// to the high 16 bits of the f32 mantissa. BF16 is just f32 with the
/// low 16 bits truncated, so this is exact.
fn bf16_bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 2 != 0 {
        return Err(AegisError::InvalidPlan(format!(
            "BF16 byte slice length {} is not a multiple of 2",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            f32::from_bits((bits as u32) << 16)
        })
        .collect())
}

/// Read raw f32 bytes (`f32` little-endian) into a host `Vec<f32>`.
fn f32_bytes_to_vec(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return Err(AegisError::InvalidPlan(format!(
            "F32 byte slice length {} is not a multiple of 4",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

/// Load any-dtype tensor bytes from the artifact and upload it as an f32
/// storage buffer. BF16 / F32 are upcast/copied as-is; FP16 is converted
/// via `f16::to_f32`. Errors on quantised dtypes — those need the NVFP4
/// pair-load path.
fn load_dense_as_f32(
    ctx: &WgpuContext,
    loader: &mut TensorStorageLoader,
    artifact: &ModelArtifact,
    name: &str,
    label: &'static str,
) -> Result<wgpu::Buffer> {
    let info = artifact.tensors.tensors.get(name).ok_or_else(|| {
        AegisError::InvalidPlan(format!("artifact missing tensor `{name}`"))
    })?;
    let host = loader.load_for_store(info, StoragePlacement::Ram)?;
    let bytes = host.as_bytes();
    let values = match info.dtype {
        TensorDType::BF16 => bf16_bytes_to_f32(bytes)?,
        TensorDType::F32 => f32_bytes_to_vec(bytes)?,
        TensorDType::F16 => bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        other => {
            return Err(AegisError::Unsupported(format!(
                "load_dense_as_f32: tensor `{name}` has unsupported dtype {other:?} \
                 (expected BF16/F16/F32)"
            )));
        }
    };
    Ok(upload_f32_buf(ctx, &values, label))
}

/// Load a linear weight from the artifact. If a companion
/// `{name}.weight_scale` tensor exists, treat as NVFP4; otherwise treat as
/// dense (BF16/F16/F32 → f32). The `_label` is used as the wgpu-buffer
/// debug label.
fn load_linear(
    ctx: &WgpuContext,
    loader: &mut TensorStorageLoader,
    artifact: &ModelArtifact,
    name: &str,
    label: &'static str,
) -> Result<WgpuLinear> {
    let weight_info = artifact.tensors.tensors.get(name).ok_or_else(|| {
        AegisError::InvalidPlan(format!("artifact missing tensor `{name}`"))
    })?;
    let scale_name = format!("{name}_scale");
    if let Some(scale_info) = artifact.tensors.tensors.get(&scale_name) {
        // NVFP4 path: packed nibbles + UE4M3 block scales.
        if weight_info.dtype != TensorDType::U8 && weight_info.dtype != TensorDType::I8 {
            return Err(AegisError::InvalidPlan(format!(
                "tensor `{name}` has companion scale `{scale_name}` (NVFP4 marker) \
                 but dtype is {:?}, not U8/I8",
                weight_info.dtype
            )));
        }
        if weight_info.shape.len() != 2 || scale_info.shape.len() != 2 {
            return Err(AegisError::InvalidPlan(format!(
                "NVFP4 weight + scale must be 2-D: `{name}` shape {:?}, `{scale_name}` shape {:?}",
                weight_info.shape, scale_info.shape
            )));
        }
        let rows = weight_info.shape[0];
        // packed col count = original col / 2 (each byte = 2 nibbles).
        let packed_cols = weight_info.shape[1];
        let cols = packed_cols * 2;
        if scale_info.shape[0] != rows || scale_info.shape[1] != cols / 16 {
            return Err(AegisError::InvalidPlan(format!(
                "NVFP4 scale shape mismatch for `{name}`: weight rows×cols/16 = {}×{}, \
                 scale shape = {:?}",
                rows, cols / 16, scale_info.shape
            )));
        }
        let packed_bytes = loader.load_for_store(weight_info, StoragePlacement::Ram)?;
        let scales_bytes = loader.load_for_store(scale_info, StoragePlacement::Ram)?;
        let packed_buf = upload_padded_u8_buf(ctx, packed_bytes.as_bytes(), label);
        let scales_buf = upload_padded_u8_buf(ctx, scales_bytes.as_bytes(), label);
        // NVFP4 stores the per-tensor output dequant scale at
        // `{prefix}.weight_scale_2` (scalar BF16/F32). The CUDA loader
        // uses this name; we mirror it. Default 1.0 if absent.
        // The base name strips the trailing `.weight` so the lookup
        // matches the artifact's `prefix.weight_scale_2`.
        let scale_2_name = if let Some(stripped) = name.strip_suffix(".weight") {
            format!("{stripped}.weight_scale_2")
        } else {
            format!("{name}.weight_scale_2")
        };
        Ok(WgpuLinear::Nvfp4 {
            packed: packed_buf,
            scales: scales_buf,
            rows,
            cols,
            output_scale: load_optional_scalar(artifact, &scale_2_name).unwrap_or(1.0),
        })
    } else {
        // Dense path. For BF16 source: store packed (each u32 = 2 bf16
        // values) and dequant on the fly during matmul. Saves ~50 % VRAM.
        // For F32 / F16 source: upload as f32 (small enough that the
        // packing complexity isn't worth it).
        if weight_info.shape.len() != 2 {
            return Err(AegisError::InvalidPlan(format!(
                "dense linear weight `{name}` must be 2-D, got shape {:?}",
                weight_info.shape
            )));
        }
        let rows = weight_info.shape[0];
        let cols = weight_info.shape[1];
        if weight_info.dtype == TensorDType::BF16 {
            let host = loader.load_for_store(weight_info, StoragePlacement::Ram)?;
            let buf = upload_bf16_packed_buf(ctx, host.as_bytes(), label)?;
            Ok(WgpuLinear::Bf16Packed { weight: buf, rows, cols })
        } else {
            let buf = load_dense_as_f32(ctx, loader, artifact, name, label)?;
            Ok(WgpuLinear::Dense { weight: buf, rows, cols })
        }
    }
}

/// Read a scalar (1-element) BF16/F32 tensor as a host f32. Returns
/// `None` if the tensor is absent.
fn load_optional_scalar(artifact: &ModelArtifact, name: &str) -> Option<f32> {
    let info = artifact.tensors.tensors.get(name)?;
    let mut loader = TensorStorageLoader::new();
    let host = loader.load_for_store(info, StoragePlacement::Ram).ok()?;
    let bytes = host.as_bytes();
    match info.dtype {
        TensorDType::F32 if bytes.len() == 4 => {
            Some(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        }
        TensorDType::BF16 if bytes.len() == 2 => {
            let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
            Some(f32::from_bits((bits as u32) << 16))
        }
        _ => None,
    }
}

/// Load a vanilla Llama-style model into wgpu buffers. Uses the standard
/// HuggingFace `transformers` tensor naming with prefix `model.`. For
/// Gemma-4 (which adds q_norm/k_norm + sub-layer norms + MoE), this is
/// the foundation; the Gemma-4-specific extensions add to this set rather
/// than replacing it.
pub fn load_vanilla_llama_model(
    ctx: Arc<WgpuContext>,
    artifact: &ModelArtifact,
    shape: WgpuModelShape,
) -> Result<WgpuModel> {
    let mut loader = TensorStorageLoader::new();

    let embed_name = "model.embed_tokens.weight";
    let embed_info = artifact.tensors.tensors.get(embed_name).ok_or_else(|| {
        AegisError::InvalidPlan(format!("artifact missing tensor `{embed_name}`"))
    })?;
    let embed_rows = embed_info.shape[0];
    let embed_cols = embed_info.shape[1];
    let embed_linear = load_linear(&ctx, &mut loader, artifact, embed_name, "embed_tokens")?;

    let final_norm =
        load_dense_as_f32(&ctx, &mut loader, artifact, "model.norm.weight", "final_norm")?;

    // Tied lm_head: try lm_head.weight, fall back to embed_tokens.weight.
    let lm_head_name = if artifact.tensors.tensors.contains_key("lm_head.weight") {
        "lm_head.weight"
    } else {
        embed_name
    };
    let lm_head = load_linear(&ctx, &mut loader, artifact, lm_head_name, "lm_head")?;

    let mut layers = Vec::with_capacity(shape.num_layers);
    for layer_idx in 0..shape.num_layers {
        let attn_norm = load_dense_as_f32(
            &ctx, &mut loader, artifact,
            &format!("model.layers.{layer_idx}.input_layernorm.weight"),
            "attn_norm",
        )?;
        let q = load_linear(&ctx, &mut loader, artifact,
            &format!("model.layers.{layer_idx}.self_attn.q_proj.weight"), "q_proj")?;
        let k = load_linear(&ctx, &mut loader, artifact,
            &format!("model.layers.{layer_idx}.self_attn.k_proj.weight"), "k_proj")?;
        let v = load_linear(&ctx, &mut loader, artifact,
            &format!("model.layers.{layer_idx}.self_attn.v_proj.weight"), "v_proj")?;
        let o = load_linear(&ctx, &mut loader, artifact,
            &format!("model.layers.{layer_idx}.self_attn.o_proj.weight"), "o_proj")?;
        let mlp_norm = load_dense_as_f32(
            &ctx, &mut loader, artifact,
            &format!("model.layers.{layer_idx}.post_attention_layernorm.weight"),
            "mlp_norm",
        )?;
        let gate = load_linear(&ctx, &mut loader, artifact,
            &format!("model.layers.{layer_idx}.mlp.gate_proj.weight"), "gate_proj")?;
        let up = load_linear(&ctx, &mut loader, artifact,
            &format!("model.layers.{layer_idx}.mlp.up_proj.weight"), "up_proj")?;
        let down = load_linear(&ctx, &mut loader, artifact,
            &format!("model.layers.{layer_idx}.mlp.down_proj.weight"), "down_proj")?;
        layers.push(WgpuLayerWeights {
            attention: WgpuAttentionWeightsFull {
                norm_weight: attn_norm,
                q_proj: q,
                k_proj: k,
                v_proj: v,
                o_proj: o,
                // Vanilla Llama has no per-head Q/K/V norms or sub-layer norms.
                q_norm: None,
                k_norm: None,
                v_norm_unit: None,
                post_attn_sublayer_norm: None,
            },
            mlp: WgpuMlpWeightsFull {
                norm_weight: mlp_norm,
                gate_proj: gate,
                up_proj: up,
                down_proj: down,
                post_mlp_sublayer_norm: None,
            },
            moe: None,
            layer_scalar: None,
            attention_window_size: None,
            head_dim_override: None,
            num_kv_heads_override: None,
        });
    }

    Ok(WgpuModel {
        ctx,
        embed_tokens: embed_linear,
        embed_tokens_rows: embed_rows,
        embed_tokens_cols: embed_cols,
        final_norm,
        lm_head,
        layers,
        hidden_size: shape.hidden_size,
        intermediate_size: shape.intermediate_size,
        num_q_heads: shape.num_q_heads,
        num_kv_heads: shape.num_kv_heads,
        head_dim: shape.head_dim,
        vocab_size: shape.vocab_size,
        rms_norm_eps: shape.rms_norm_eps,
        // Vanilla Llama doesn't scale embeddings.
        embed_scale: None,
        lm_head_tied: false,
    })
}

/// Optional dense norm-weight loader. Returns the first existing
/// tensor across `candidates`; `None` when none exist. BF16/F16/F32 →
/// f32 device buffer.
fn load_optional_dense(
    ctx: &WgpuContext,
    loader: &mut TensorStorageLoader,
    artifact: &ModelArtifact,
    candidates: &[&str],
    label: &'static str,
) -> Result<Option<wgpu::Buffer>> {
    for name in candidates {
        if artifact.tensors.tensors.contains_key(*name) {
            return Ok(Some(load_dense_as_f32(ctx, loader, artifact, name, label)?));
        }
    }
    Ok(None)
}

/// Load `model.layers.{L}.router.per_expert_scale` as a host `Vec<f32>`.
/// Returns identity (`vec![1.0; num_experts]`) if the tensor is absent.
fn load_per_expert_scale_host(
    loader: &mut TensorStorageLoader,
    artifact: &ModelArtifact,
    layer_idx: usize,
    text_prefix: &str,
    num_experts: usize,
) -> Result<Vec<f32>> {
    let name = format!("{text_prefix}layers.{layer_idx}.router.per_expert_scale");
    let info = match artifact.tensors.tensors.get(&name) {
        Some(t) => t,
        None => return Ok(vec![1.0; num_experts]),
    };
    let host = loader.load_for_store(info, StoragePlacement::Ram)?;
    let bytes = host.as_bytes();
    match info.dtype {
        TensorDType::BF16 => bf16_bytes_to_f32(bytes),
        TensorDType::F32 => f32_bytes_to_vec(bytes),
        TensorDType::F16 => Ok(bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()),
        other => Err(AegisError::Unsupported(format!(
            "router.per_expert_scale[{layer_idx}] has unsupported dtype {other:?}"
        ))),
    }
}

/// Load a Gemma-4 model into wgpu buffers. Drives off the artifact
/// `HfConfig` for per-layer global/sliding detection and the gemma4
/// helper functions in `aegisllm_base::model::gemma4`.
///
/// Populates every Gemma-4-specific optional field on
/// `WgpuLayerWeights` and `WgpuMoeWeights`:
///   * Per-head Q/K norms + V `v_norm_unit` (one shared all-ones buffer
///     of length max(head_dim) across all layers).
///   * PrePost sub-layer norms when present.
///   * `embed_scale = sqrt(hidden_size)` and per-layer `layer_scalar`.
///   * MoE: router (NVFP4 or BF16), `router.scale` input scale,
///     `per_expert_scale` host vec, 128 routed experts (NVFP4),
///     shared expert (BF16 dense → upcast to f32), 3 PrePost FFN norms.
///   * Per-layer `attention_window_size = Some(1024)` for sliding
///     layers, `None` for global; `head_dim_override = Some(512)` for
///     globals (vs the model-level `head_dim = 256`).
///   * V-proj fallback: when `self_attn.v_proj.weight` is absent on a
///     global layer (Gemma-4's `attention_k_eq_v=true` quirk), reuses
///     `k_proj` weights.
pub fn load_gemma4_model(
    ctx: Arc<WgpuContext>,
    artifact: &ModelArtifact,
) -> Result<WgpuModel> {
    let cfg = &artifact.config;
    let hidden = cfg.hidden_size;
    let num_layers = cfg.num_hidden_layers;
    let intermediate = cfg
        .intermediate_size
        .ok_or_else(|| AegisError::InvalidPlan("Gemma-4 config missing intermediate_size".into()))?;
    let num_q_heads = cfg.num_attention_heads;
    let num_kv_heads = cfg
        .num_key_value_heads
        .unwrap_or(num_q_heads);
    let model_head_dim = cfg.head_dim.unwrap_or(hidden / num_q_heads);
    let vocab = cfg
        .vocab_size
        .ok_or_else(|| AegisError::InvalidPlan("Gemma-4 config missing vocab_size".into()))?;
    let rms_norm_eps = cfg.rms_norm_eps.unwrap_or(1e-6) as f32;
    let num_experts = cfg
        .num_experts
        .ok_or_else(|| AegisError::InvalidPlan("Gemma-4 config missing num_experts".into()))?;
    let top_k = cfg.num_experts_per_tok.unwrap_or(2);
    let moe_intermediate = cfg.moe_intermediate_size.unwrap_or(intermediate);
    let window = aegisllm_base::model::gemma4::sliding_window(cfg) as u32;

    // Detect text prefix ("model." or "model.language_model." for multimodal).
    let text_prefix = if artifact
        .tensors
        .tensors
        .contains_key("model.language_model.embed_tokens.weight")
    {
        "model.language_model."
    } else {
        "model."
    };

    let mut loader = TensorStorageLoader::new();

    // Embeddings + final norm + lm_head.
    let embed_name = format!("{text_prefix}embed_tokens.weight");
    let embed_info = artifact.tensors.tensors.get(&embed_name).ok_or_else(|| {
        AegisError::InvalidPlan(format!("Gemma-4 artifact missing tensor `{embed_name}`"))
    })?;
    let embed_rows = embed_info.shape[0];
    let embed_cols = embed_info.shape[1];
    if std::env::var("AEGIS_WGPU_TRACE").is_ok() {
        eprintln!(
            "[g4-loader] cfg: hidden={} inter={} num_q={} num_kv={} head_dim={} vocab={} \
             num_layers={} num_experts={} top_k={} text_prefix={}",
            hidden, intermediate, num_q_heads, num_kv_heads, model_head_dim, vocab,
            num_layers, num_experts, top_k, text_prefix
        );
        eprintln!(
            "[g4-loader] embed tensor `{embed_name}` shape={:?} bytes={}",
            embed_info.shape,
            embed_info.data_len_bytes()
        );
    }
    // Load embed_tokens via `load_linear` so BF16 source ends up
    // stored as Bf16Packed (saves ~50 % VRAM vs f32 upcast).
    let embed_linear = load_linear(&ctx, &mut loader, artifact, &embed_name, "g4_embed")?;
    if std::env::var("AEGIS_WGPU_TRACE").is_ok() {
        eprintln!(
            "[g4-loader] embed loaded as {:?}",
            embed_linear,
        );
    }
    let final_norm = load_dense_as_f32(
        &ctx,
        &mut loader,
        artifact,
        &format!("{text_prefix}norm.weight"),
        "g4_final_norm",
    )?;
    let lm_head_present = artifact.tensors.tensors.contains_key("lm_head.weight");
    let (lm_head, lm_head_tied) = if lm_head_present {
        (load_linear(&ctx, &mut loader, artifact, "lm_head.weight", "g4_lm_head")?, false)
    } else {
        // Tied: lm_head reuses the embedding table at forward time.
        // Construct a placeholder Dense linear pointing at a tiny dummy
        // buffer (forward path checks `lm_head_tied` before reading
        // this). Prevents allocating a 2.95 GiB duplicate of embed.
        (
            WgpuLinear::Dense {
                weight: upload_f32_buf(&ctx, &[0.0_f32], "g4_lm_head_tied_placeholder"),
                rows: embed_rows,
                cols: embed_cols,
            },
            true,
        )
    };

    // V-norm unit buffer: an all-ones vector sized to the LARGEST head_dim
    // any layer uses (typically Gemma-4 global head_dim=512). One buffer
    // shared across all layers.
    let max_head_dim = (0..num_layers)
        .map(|l| {
            aegisllm_base::model::gemma4::head_dim_for_layer(l, cfg).unwrap_or(model_head_dim)
        })
        .max()
        .unwrap_or(model_head_dim);
    let v_norm_unit_buf = upload_f32_buf(
        &ctx,
        &vec![1.0_f32; max_head_dim],
        "g4_v_norm_unit",
    );

    let mut layers = Vec::with_capacity(num_layers);
    for layer_idx in 0..num_layers {
        let prefix = format!("{text_prefix}layers.{layer_idx}");
        let is_global = aegisllm_base::model::gemma4::is_global_layer(layer_idx, cfg);
        let layer_head_dim =
            aegisllm_base::model::gemma4::head_dim_for_layer(layer_idx, cfg).unwrap_or(model_head_dim);
        let layer_kv_heads = if is_global {
            cfg.num_global_key_value_heads.unwrap_or(num_kv_heads)
        } else {
            num_kv_heads
        };
        let q_width = num_q_heads * layer_head_dim;
        let kv_width = layer_kv_heads * layer_head_dim;

        // Attention norms.
        let input_norm = load_dense_as_f32(
            &ctx,
            &mut loader,
            artifact,
            &format!("{prefix}.input_layernorm.weight"),
            "g4_input_norm",
        )?;

        // Q/K/V/O projections. V-proj falls back to k_proj when absent
        // (Gemma-4 global layers can have attention_k_eq_v=true).
        let q_proj = load_linear(&ctx, &mut loader, artifact,
            &format!("{prefix}.self_attn.q_proj.weight"), "g4_q_proj")?;
        let k_proj = load_linear(&ctx, &mut loader, artifact,
            &format!("{prefix}.self_attn.k_proj.weight"), "g4_k_proj")?;
        let v_proj_name = if artifact.tensors.tensors.contains_key(&format!("{prefix}.self_attn.v_proj.weight"))
            || artifact.tensors.tensors.contains_key(&format!("{prefix}.self_attn.v_proj.weight_scale"))
        {
            format!("{prefix}.self_attn.v_proj.weight")
        } else {
            format!("{prefix}.self_attn.k_proj.weight")
        };
        let v_proj = load_linear(&ctx, &mut loader, artifact, &v_proj_name, "g4_v_proj")?;
        let o_proj = load_linear(&ctx, &mut loader, artifact,
            &format!("{prefix}.self_attn.o_proj.weight"), "g4_o_proj")?;

        // Per-head Q/K norms (Gemma-4 always present).
        let q_norm = load_optional_dense(
            &ctx, &mut loader, artifact,
            &[&format!("{prefix}.self_attn.q_norm.weight")],
            "g4_q_norm",
        )?;
        let k_norm = load_optional_dense(
            &ctx, &mut loader, artifact,
            &[&format!("{prefix}.self_attn.k_norm.weight")],
            "g4_k_norm",
        )?;
        // V-norm unit: bind only when q_norm is present (Gemma-4 marker).
        let v_norm_unit = if q_norm.is_some() {
            // Need a fresh handle per layer; reupload each time. Cheap.
            Some(upload_f32_buf(
                &ctx,
                &vec![1.0_f32; layer_head_dim],
                "g4_v_norm_unit_layer",
            ))
        } else {
            None
        };
        let _ = &v_norm_unit_buf; // shared one not used; per-layer simpler

        // Sub-layer norms (PrePost). The "pre-MLP" norm comes from
        // `pre_feedforward_layernorm.weight` (Gemma-4) or
        // `post_attention_layernorm.weight` (Llama fallback) — same slot.
        let mlp_norm = if let Some(buf) = load_optional_dense(
            &ctx, &mut loader, artifact,
            &[&format!("{prefix}.pre_feedforward_layernorm.weight")],
            "g4_mlp_norm",
        )? {
            buf
        } else {
            load_dense_as_f32(
                &ctx, &mut loader, artifact,
                &format!("{prefix}.post_attention_layernorm.weight"),
                "g4_mlp_norm",
            )?
        };

        // Post-attn sublayer norm: if Gemma-4 PrePost (pre_feedforward
        // exists), then post_attention_layernorm is the post-attn
        // sublayer norm. If only post_attention_layernorm exists, it
        // was already used as `mlp_norm` and there's no sublayer norm.
        let has_pre_ffn_norm = artifact
            .tensors
            .tensors
            .contains_key(&format!("{prefix}.pre_feedforward_layernorm.weight"));
        let post_attn_sublayer_norm = if has_pre_ffn_norm {
            load_optional_dense(
                &ctx, &mut loader, artifact,
                &[
                    &format!("{prefix}.post_attention_layernorm.weight"),
                    &format!("{prefix}.post_attention_norm.weight"),
                    &format!("{prefix}.post_attn_layernorm.weight"),
                ],
                "g4_post_attn_subnorm",
            )?
        } else {
            None
        };
        let post_mlp_sublayer_norm = load_optional_dense(
            &ctx, &mut loader, artifact,
            &[
                &format!("{prefix}.post_feedforward_layernorm.weight"),
                &format!("{prefix}.post_mlp_norm.weight"),
            ],
            "g4_post_mlp_subnorm",
        )?;

        // ── MoE: router + 128 routed experts + shared expert ──────────
        let router = load_linear(
            &ctx, &mut loader, artifact,
            &format!("{prefix}.router.proj.weight"),
            "g4_router",
        )
        // Some checkpoints have just `router.weight` instead of `router.proj.weight`.
        .or_else(|_| load_linear(&ctx, &mut loader, artifact,
            &format!("{prefix}.router.weight"), "g4_router_alt"))?;
        let router_input_scale = load_optional_dense(
            &ctx, &mut loader, artifact,
            &[&format!("{prefix}.router.scale")],
            "g4_router_input_scale",
        )?;
        let per_expert_scale =
            load_per_expert_scale_host(&mut loader, artifact, layer_idx, text_prefix, num_experts)?;

        // Routed expert prefix: Gemma-4 native uses `{prefix}.experts.{E}.*`,
        // Qwen 3.x style uses `{prefix}.mlp.experts.{E}.*`.
        let expert_base = if artifact
            .tensors
            .tensors
            .contains_key(&format!("{prefix}.experts.0.gate_proj.weight"))
        {
            format!("{prefix}.experts")
        } else {
            format!("{prefix}.mlp.experts")
        };
        let mut experts = Vec::with_capacity(num_experts);
        for expert_idx in 0..num_experts {
            let ep = format!("{expert_base}.{expert_idx}");
            let gate = load_linear(&ctx, &mut loader, artifact,
                &format!("{ep}.gate_proj.weight"), "g4_expert_gate")?;
            let up = load_linear(&ctx, &mut loader, artifact,
                &format!("{ep}.up_proj.weight"), "g4_expert_up")?;
            let down = load_linear(&ctx, &mut loader, artifact,
                &format!("{ep}.down_proj.weight"), "g4_expert_down")?;
            experts.push(WgpuMoeExpert {
                gate_proj: gate,
                up_proj: up,
                down_proj: down,
            });
        }

        // Shared expert: Gemma-4 stores at `{prefix}.mlp.{gate,up,down}_proj.weight` (BF16).
        let shared_expert = if artifact
            .tensors
            .tensors
            .contains_key(&format!("{prefix}.mlp.gate_proj.weight"))
        {
            let gate = load_linear(&ctx, &mut loader, artifact,
                &format!("{prefix}.mlp.gate_proj.weight"), "g4_shared_gate")?;
            let up = load_linear(&ctx, &mut loader, artifact,
                &format!("{prefix}.mlp.up_proj.weight"), "g4_shared_up")?;
            let down = load_linear(&ctx, &mut loader, artifact,
                &format!("{prefix}.mlp.down_proj.weight"), "g4_shared_down")?;
            Some(WgpuMlpWeightsFull {
                norm_weight: load_dense_as_f32(
                    &ctx, &mut loader, artifact,
                    &format!("{prefix}.input_layernorm.weight"),  // dummy: shared expert reuses pre-FFN norm
                    "g4_shared_norm_dummy",
                )?,
                gate_proj: gate,
                up_proj: up,
                down_proj: down,
                post_mlp_sublayer_norm: None,
            })
        } else {
            None
        };

        // PrePost FFN norms (Gemma-4 MoE specific).
        let pre_ff_2 = load_optional_dense(
            &ctx, &mut loader, artifact,
            &[&format!("{prefix}.pre_feedforward_layernorm_2.weight")],
            "g4_pre_ff_2",
        )?;
        let post_ff_1 = load_optional_dense(
            &ctx, &mut loader, artifact,
            &[&format!("{prefix}.post_feedforward_layernorm_1.weight")],
            "g4_post_ff_1",
        )?;
        let post_ff_2 = load_optional_dense(
            &ctx, &mut loader, artifact,
            &[&format!("{prefix}.post_feedforward_layernorm_2.weight")],
            "g4_post_ff_2",
        )?;

        let moe = WgpuMoeWeights {
            router,
            router_input_scale,
            per_expert_scale,
            experts,
            shared_expert,
            num_experts,
            top_k,
            intermediate_size: moe_intermediate,
            pre_feedforward_layernorm_2: pre_ff_2,
            post_feedforward_layernorm_1: post_ff_1,
            post_feedforward_layernorm_2: post_ff_2,
        };

        // Optional layer_scalar.
        let layer_scalar =
            load_optional_scalar(artifact, &format!("{prefix}.layer_scalar"));

        layers.push(WgpuLayerWeights {
            attention: WgpuAttentionWeightsFull {
                norm_weight: input_norm,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                q_norm,
                k_norm,
                v_norm_unit,
                post_attn_sublayer_norm,
            },
            mlp: WgpuMlpWeightsFull {
                norm_weight: mlp_norm,
                // Gemma-4 layers route through MoE; the dense MLP fields
                // here hold the SHARED expert weights so the existing
                // dense forward path stays usable for diagnostics. The
                // MoE forward uses `moe.shared_expert` instead.
                gate_proj: WgpuLinear::Dense {
                    weight: upload_f32_buf(&ctx, &[0.0_f32], "g4_dummy_gate"),
                    rows: moe_intermediate,
                    cols: hidden,
                },
                up_proj: WgpuLinear::Dense {
                    weight: upload_f32_buf(&ctx, &[0.0_f32], "g4_dummy_up"),
                    rows: moe_intermediate,
                    cols: hidden,
                },
                down_proj: WgpuLinear::Dense {
                    weight: upload_f32_buf(&ctx, &[0.0_f32], "g4_dummy_down"),
                    rows: hidden,
                    cols: moe_intermediate,
                },
                post_mlp_sublayer_norm,
            },
            moe: Some(moe),
            layer_scalar,
            attention_window_size: if is_global { None } else { Some(window) },
            head_dim_override: if is_global { Some(layer_head_dim) } else { None },
            num_kv_heads_override: if is_global {
                Some(layer_kv_heads)
            } else {
                None
            },
        });
        // Sanity: q_width / kv_width are computed but only documented for clarity.
        let _ = (q_width, kv_width);
    }

    Ok(WgpuModel {
        ctx,
        embed_tokens: embed_linear,
        embed_tokens_rows: embed_rows,
        embed_tokens_cols: embed_cols,
        final_norm,
        lm_head,
        layers,
        hidden_size: hidden,
        intermediate_size: intermediate,
        num_q_heads,
        num_kv_heads,
        head_dim: model_head_dim,
        vocab_size: vocab,
        rms_norm_eps,
        // Gemma-4 ScaledWordEmbedding: out = embed_lookup * sqrt(hidden_size).
        embed_scale: Some((hidden as f32).sqrt()),
        lm_head_tied,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_roundtrip_matches_f32_for_representable_values() {
        // BF16 representable f32 values: those whose low 16 bits are zero.
        // 1.0 = 0x3F800000, low 16 = 0 → exact.
        let src: [f32; 4] = [1.0, -2.0, 0.5, 0.0];
        let mut bytes = Vec::with_capacity(src.len() * 2);
        for v in src {
            // BF16 is f32 with low 16 bits truncated.
            let bf16 = (v.to_bits() >> 16) as u16;
            bytes.extend_from_slice(&bf16.to_le_bytes());
        }
        let decoded = bf16_bytes_to_f32(&bytes).unwrap();
        for (a, b) in src.iter().zip(decoded.iter()) {
            assert_eq!(a, b, "BF16-representable round trip failed");
        }
    }

    #[test]
    fn bf16_byte_length_validates() {
        let err = bf16_bytes_to_f32(&[1, 2, 3]).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("not a multiple of 2"));
    }

    #[test]
    fn f32_byte_length_validates() {
        let err = f32_bytes_to_vec(&[1, 2, 3]).unwrap_err();
        let msg = format!("{err:?}");
        assert!(msg.contains("not a multiple of 4"));
    }
}

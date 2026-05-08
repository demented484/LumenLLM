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

use super::forward::{upload_f32_buf, upload_padded_u8_buf};
use super::loader::WgpuContext;

/// A linear weight matrix uploaded to the device. Either dense f32 (BF16
/// or F32 source upcast at load time) or NVFP4 packed.
pub enum WgpuLinear {
    /// f32 row-major `[rows, cols]`.
    Dense {
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
            Self::Dense { rows, .. } | Self::Nvfp4 { rows, .. } => *rows,
        }
    }
    pub fn cols(&self) -> usize {
        match self {
            Self::Dense { cols, .. } | Self::Nvfp4 { cols, .. } => *cols,
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
}

impl std::fmt::Debug for WgpuAttentionWeightsFull {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuAttentionWeightsFull").finish_non_exhaustive()
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
    pub embed_tokens: wgpu::Buffer,
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
        Ok(WgpuLinear::Nvfp4 {
            packed: packed_buf,
            scales: scales_buf,
            rows,
            cols,
            // Per-tensor output scale: artifact may store it as a
            // separate `{name}.output_scale` scalar; default 1.0 if
            // absent. NVFP4 tensors without an output scale are common.
            output_scale: load_optional_scalar(artifact, &format!("{name}.output_scale")).unwrap_or(1.0),
        })
    } else {
        // Dense path.
        if weight_info.shape.len() != 2 {
            return Err(AegisError::InvalidPlan(format!(
                "dense linear weight `{name}` must be 2-D, got shape {:?}",
                weight_info.shape
            )));
        }
        let rows = weight_info.shape[0];
        let cols = weight_info.shape[1];
        let buf = load_dense_as_f32(ctx, loader, artifact, name, label)?;
        Ok(WgpuLinear::Dense { weight: buf, rows, cols })
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
    let embed_buf = load_dense_as_f32(&ctx, &mut loader, artifact, embed_name, "embed_tokens")?;

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
            },
            mlp: WgpuMlpWeightsFull {
                norm_weight: mlp_norm,
                gate_proj: gate,
                up_proj: up,
                down_proj: down,
            },
        });
    }

    Ok(WgpuModel {
        ctx,
        embed_tokens: embed_buf,
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

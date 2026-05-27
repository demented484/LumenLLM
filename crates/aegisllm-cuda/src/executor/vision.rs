//! Vision tower + multimodal projector loader (Stage I).
//!
//! Loads `model.vision_tower.*` + `model.embed_vision.*` from the artifact
//! into CUDA-resident BF16 tensors. The Gemma-4 vision tower is a SigLIP-style
//! ViT with QK-norm and the 4-LN-per-block layout reused from the text model.
//!
//! Tensor inventory (per the NVFP4 artifact's safetensors index):
//!
//! Tower-level:
//!   model.vision_tower.patch_embedder.input_proj.weight      [hidden, P*P*3]
//!   model.vision_tower.patch_embedder.position_embedding_table  [2, 10240, hidden]
//!   model.vision_tower.std_scale                             [hidden]
//!   model.vision_tower.std_bias                              [hidden]
//!
//! Per layer (× num_layers):
//!   .self_attn.{q,k,v,o}_proj.linear.weight                  [hidden, hidden]
//!   .self_attn.{q,k}_norm.weight                             [head_dim]
//!   .input_layernorm.weight                                  [hidden]
//!   .post_attention_layernorm.weight                         [hidden]
//!   .pre_feedforward_layernorm.weight                        [hidden]
//!   .post_feedforward_layernorm.weight                       [hidden]
//!   .mlp.{gate,up}_proj.linear.weight                        [intermediate, hidden]
//!   .mlp.down_proj.linear.weight                             [hidden, intermediate]
//!
//! Projector (vision-hidden → text-hidden):
//!   model.embed_vision.embedding_projection.weight           [text_hidden, vision_hidden]
//!
//! Stage I.1 ships the LOADER + data structures only. The forward pass
//! (patch-embed → 27 layers → std norm → pooling → projection) lands in I.2.

use aegisllm_base::artifact::ModelArtifact;
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::tensor::TensorInfo;
use aegisllm_base::tensor::storage::TensorStorageLoader;

use super::loader::cuda_residency_for_store;
use crate::cuda::loader::CudaWeightLoader;
use crate::cuda::{DeviceBf16Matrix, DeviceBuffer};
use aegisllm_base::planning::placement::StoragePlacement;

/// Configuration for one vision encoder, derived from the model's
/// `vision_config` (config.json).
#[derive(Debug, Clone)]
pub struct VisionEncoderShape {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub position_embedding_size: usize,
}

impl VisionEncoderShape {
    /// Hard-coded Gemma-4 vision config (matches config.json["vision_config"]).
    pub fn gemma4() -> Self {
        Self {
            hidden_size: 1152,
            intermediate_size: 4304,
            num_layers: 27,
            num_attention_heads: 16,
            head_dim: 72,
            patch_size: 16,
            pooling_kernel_size: 3,
            rms_norm_eps: 1.0e-6,
            rope_theta: 100.0,
            position_embedding_size: 10240,
        }
    }
}

/// One transformer block's BF16 device-resident weights.
pub struct VisionLayerWeights {
    pub q_proj: DeviceBf16Matrix,
    pub k_proj: DeviceBf16Matrix,
    pub v_proj: DeviceBf16Matrix,
    pub o_proj: DeviceBf16Matrix,
    pub q_norm: DeviceBuffer<f32>,
    pub k_norm: DeviceBuffer<f32>,
    pub input_layernorm: DeviceBuffer<f32>,
    pub post_attention_layernorm: DeviceBuffer<f32>,
    pub pre_feedforward_layernorm: DeviceBuffer<f32>,
    pub post_feedforward_layernorm: DeviceBuffer<f32>,
    pub mlp_gate: DeviceBf16Matrix,
    pub mlp_up: DeviceBf16Matrix,
    pub mlp_down: DeviceBf16Matrix,
}

/// The full vision encoder (tower + projector). Everything CUDA-resident.
pub struct VisionTower {
    pub shape: VisionEncoderShape,
    pub patch_embed: DeviceBf16Matrix,
    pub position_table: DeviceBf16Matrix,
    pub std_scale: DeviceBuffer<f32>,
    pub std_bias: DeviceBuffer<f32>,
    pub layers: Vec<VisionLayerWeights>,
    pub projector: DeviceBf16Matrix,
}

impl VisionTower {
    /// Load the vision tower + projector from the artifact. All weights
    /// uploaded to VRAM as device-resident BF16. Returns Err if any required
    /// tensor is missing or has the wrong shape/dtype.
    pub fn from_artifact(
        artifact: &ModelArtifact,
        shape: VisionEncoderShape,
        cuda_weights: &CudaWeightLoader<'_>,
        device_index: usize,
        loader: &mut TensorStorageLoader,
    ) -> Result<Self> {
        let store = StoragePlacement::Vram { device: device_index };
        let residency = cuda_residency_for_store(store, device_index)?;

        let get = |name: &str| -> Result<&TensorInfo> {
            artifact.tensors.tensors.get(name).ok_or_else(|| {
                AegisError::InvalidPlan(format!("vision tower: tensor `{name}` missing"))
            })
        };

        let patch_embed = cuda_weights.load_bf16_matrix_with_store(
            get("model.vision_tower.patch_embedder.input_proj.weight")?,
            store, residency.clone(), loader,
        )?;
        // position_embedding_table is shape [2, N, H] but stored contiguously;
        // load as a 2-D view [2*N, H]. The forward indexes slot 0 vs 1.
        let position_table = cuda_weights.load_bf16_matrix_with_store(
            get("model.vision_tower.patch_embedder.position_embedding_table")?,
            store, residency.clone(), loader,
        )?;
        let std_scale = cuda_weights.load_dense_vector_with_store(
            get("model.vision_tower.std_scale")?, store, loader,
        )?;
        let std_bias = cuda_weights.load_dense_vector_with_store(
            get("model.vision_tower.std_bias")?, store, loader,
        )?;

        let mut layers = Vec::with_capacity(shape.num_layers);
        for li in 0..shape.num_layers {
            let p = |suffix: &str| format!("model.vision_tower.encoder.layers.{li}.{suffix}");
            let layer = VisionLayerWeights {
                q_proj: cuda_weights.load_bf16_matrix_with_store(
                    get(&p("self_attn.q_proj.linear.weight"))?,
                    store, residency.clone(), loader)?,
                k_proj: cuda_weights.load_bf16_matrix_with_store(
                    get(&p("self_attn.k_proj.linear.weight"))?,
                    store, residency.clone(), loader)?,
                v_proj: cuda_weights.load_bf16_matrix_with_store(
                    get(&p("self_attn.v_proj.linear.weight"))?,
                    store, residency.clone(), loader)?,
                o_proj: cuda_weights.load_bf16_matrix_with_store(
                    get(&p("self_attn.o_proj.linear.weight"))?,
                    store, residency.clone(), loader)?,
                q_norm: cuda_weights.load_dense_vector_with_store(
                    get(&p("self_attn.q_norm.weight"))?, store, loader)?,
                k_norm: cuda_weights.load_dense_vector_with_store(
                    get(&p("self_attn.k_norm.weight"))?, store, loader)?,
                input_layernorm: cuda_weights.load_dense_vector_with_store(
                    get(&p("input_layernorm.weight"))?, store, loader)?,
                post_attention_layernorm: cuda_weights.load_dense_vector_with_store(
                    get(&p("post_attention_layernorm.weight"))?, store, loader)?,
                pre_feedforward_layernorm: cuda_weights.load_dense_vector_with_store(
                    get(&p("pre_feedforward_layernorm.weight"))?, store, loader)?,
                post_feedforward_layernorm: cuda_weights.load_dense_vector_with_store(
                    get(&p("post_feedforward_layernorm.weight"))?, store, loader)?,
                mlp_gate: cuda_weights.load_bf16_matrix_with_store(
                    get(&p("mlp.gate_proj.linear.weight"))?,
                    store, residency.clone(), loader)?,
                mlp_up: cuda_weights.load_bf16_matrix_with_store(
                    get(&p("mlp.up_proj.linear.weight"))?,
                    store, residency.clone(), loader)?,
                mlp_down: cuda_weights.load_bf16_matrix_with_store(
                    get(&p("mlp.down_proj.linear.weight"))?,
                    store, residency.clone(), loader)?,
            };
            layers.push(layer);
        }

        let projector = cuda_weights.load_bf16_matrix_with_store(
            get("model.embed_vision.embedding_projection.weight")?,
            store, residency.clone(), loader,
        )?;

        Ok(Self { shape, patch_embed, position_table, std_scale, std_bias, layers, projector })
    }
}

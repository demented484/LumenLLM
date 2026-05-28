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
use rayon::prelude::*;

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

    /// Forward pass: preprocessed image pixels → image-soft-token embeddings
    /// in TEXT embedding space.
    ///
    /// Inputs:
    ///   * `patches`: `[n_patches, P*P*3]` row-major f32, output of
    ///     `ImageProcessor::load(image).patches`.
    ///   * `n_patches_h`, `n_patches_w`: patch grid (`H/P`, `W/P`).
    ///
    /// Output: `Vec<f32>` of shape `[n_tokens × text_hidden]` row-major,
    /// where `n_tokens = (n_patches_h/pool) × (n_patches_w/pool)` are the
    /// post-pool image-soft-tokens ready to be injected at `<|image|>`
    /// positions in the prompt's embedding stream.
    ///
    /// CORRECTNESS-FIRST implementation: attention softmax happens on CPU
    /// (download Q·K^T → softmax → upload). Slow (~seconds for 2376 patches)
    /// but uses only existing primitives. A fused GPU softmax + bidirectional
    /// attention kernel is a future perf step; this validates end-to-end first.
    pub fn forward(
        &self,
        runtime: &crate::cuda::CudaRuntime,
        patches: &[f32],
        n_patches_h: usize,
        n_patches_w: usize,
    ) -> Result<Vec<f32>> {
        let s = &self.shape;
        let n_patches = n_patches_h * n_patches_w;
        let patch_dim = 3 * s.patch_size * s.patch_size;
        if patches.len() != n_patches * patch_dim {
            return Err(AegisError::InvalidPlan(format!(
                "vision forward: patches len={} != n_patches({}) * patch_dim({}) = {}",
                patches.len(), n_patches, patch_dim, n_patches * patch_dim
            )));
        }

        // ── Patch embedding: [n_patches, P²·3] @ patch_embed.T  →  [n_patches, hidden]
        // patch_embed.weight is stored as [hidden, P²·3] (PyTorch row-major
        // out × in), so cuBLASLt BF16 GEMM with `weight @ input.T` style
        // produces `[n_patches, hidden]` directly. matmul_bf16_cublaslt
        // computes  out[i,j] = sum_k weight[j,k] * input[i,k]  (where
        // weight.rows = out_dim, weight.cols = in_dim, batch = n_patches).
        // HF Gemma4VisionPatchEmbedder.forward: pixel_values = 2 * (pixel_values - 0.5)
        // i.e. remap [0, 1] (post-rescale) → [-1, 1] BEFORE the patch_embed linear.
        // (The preprocessor only rescales by 1/255, doesn't normalize; the [-1,1]
        // mapping happens inside the model.)
        let patches_rescaled: Vec<f32> = patches.iter().map(|&x| 2.0 * (x - 0.5)).collect();
        let patches_f32 = runtime.upload_f32(&patches_rescaled)?;
        let mut patches_bf16 = runtime.alloc_u16(patches.len())?;
        runtime.f32_to_bf16_device(&patches_f32, patches.len(), &mut patches_bf16)?;

        let mut hidden_bf16 = runtime.alloc_u16(n_patches * s.hidden_size)?;
        let mut hidden_f32  = runtime.alloc_f32(n_patches * s.hidden_size)?;
        runtime.matmul_bf16_cublaslt_with_input_bf16_device(
            &self.patch_embed,
            &patches_bf16,
            n_patches,
            &mut hidden_bf16,
            &mut hidden_f32,
        )?;

        // ── Add position embeddings (2D-axial: row_emb + col_emb summed).
        // position_table is `[2, N, hidden]` flattened to `[2N, hidden]`.
        // Slot 0 (rows 0..N) is the row-position table; slot 1 (rows N..2N)
        // is the col-position table. For a [H, W] patch grid:
        //   tok[ph, pw] += row_emb[interp_row(ph)] + col_emb[interp_col(pw)]
        // The interpolation maps the [n_patches_h, n_patches_w] grid into
        // the fixed [N, N] table size by nearest-position lookup. Production
        // models use bilinear interp; for the smoke nearest is fine to
        // validate end-to-end shape correctness; we'll upgrade to bilinear
        // once the rest of the pipeline is proven.
        let n_table = s.position_embedding_size;
        let pos_table_f32 = {
            // Download once; n_table * hidden * 2 = 10240*1152*2 = 23M f32 = 94 MiB.
            // Fine for one-time vision init.
            let buf = runtime.download_u16_slice(self.position_table.values_u16())?;
            buf.into_iter().map(|h| {
                f32::from_bits((h as u32) << 16)
            }).collect::<Vec<f32>>()
        };

        // HF Gemma4VisionPatchEmbedder._position_embeddings: patches are
        // indexed by their (x_grid, y_grid) position directly into the
        // 2-bank position_embedding_table [2, N, H]. Bank 0 = x-positions,
        // bank 1 = y-positions; the two are SUMMED per patch.
        // Patch order in HF processor is "xy" meshgrid: (x, y) iterates
        // x fastest, then y. We patchify in (ph, pw) order with pw fastest.
        // In HF position_ids each row of `stacked_grid.reshape(N, 2)` is
        // (x, y) = (pw, ph) — i.e. column-first reading of the patches.
        // OUR patches[] is in (ph, pw) row-major order (row-first); to match
        // the position_id ordering we just need (x, y) = (pw, ph) per token.
        let mut hidden_host = runtime.download_f32(&hidden_f32)?;
        for ph in 0..n_patches_h {
            for pw in 0..n_patches_w {
                let tok = ph * n_patches_w + pw;
                let x_idx = (pw.min(n_table - 1)) as usize;
                let y_idx = (ph.min(n_table - 1)) as usize;
                // bank 0 = x table; bank 1 = y table (offset by n_table rows)
                let x_off = x_idx * s.hidden_size;
                let y_off = (n_table + y_idx) * s.hidden_size;
                for k in 0..s.hidden_size {
                    hidden_host[tok * s.hidden_size + k] +=
                        pos_table_f32[x_off + k] + pos_table_f32[y_off + k];
                }
            }
        }

        // From here on, the per-layer transformer body lives on the GPU.
        // We use a CPU-side reference attention with download/upload because
        // there's no standalone non-causal-attention CUDA primitive yet.
        let mut state = hidden_host;
        let nh = s.num_attention_heads;
        let hd = s.head_dim;
        let h  = s.hidden_size;
        let i  = s.intermediate_size;
        let eps = s.rms_norm_eps;

        // ── Precompute 2D RoPE cos/sin tables (per HF Gemma4VisionRotaryEmbedding).
        // ndim = 2 spatial dims, head_dim split into half per dim (36 each).
        // For each dim: inv_freq = 1 / theta^(arange(0, spatial_dim, 2) / spatial_dim)
        //   spatial_dim = head_dim / 2 = 36, so arange(0,36,2) = 18 freqs.
        // For each token & dim: freqs = pos_id_for_dim * inv_freq → [n_patches, 18].
        // emb_for_dim = cat([freqs, freqs], dim=-1) → [n_patches, 36].
        // cos_full = cat([cos_x, cos_y], dim=-1) → [n_patches, head_dim=72].
        // (Identical for sin.) Then apply_multidimensional_rope splits x into
        // 2 chunks of size 36 and rotates each with that dim's (cos, sin).
        let spatial_dim = hd / 2;
        let n_freqs = spatial_dim / 2;
        let mut inv_freq = Vec::with_capacity(n_freqs);
        for i_f in 0..n_freqs {
            let exponent = (2 * i_f) as f32 / spatial_dim as f32;
            inv_freq.push(1.0 / s.rope_theta.powf(exponent));
        }
        let mut cos_table = vec![0f32; n_patches * hd];
        let mut sin_table = vec![0f32; n_patches * hd];
        for ph in 0..n_patches_h {
            for pw in 0..n_patches_w {
                let tok = ph * n_patches_w + pw;
                let pos_x = pw as f32;
                let pos_y = ph as f32;
                // x-dim freqs (first spatial_dim entries of cos/sin).
                for i_f in 0..n_freqs {
                    let f = pos_x * inv_freq[i_f];
                    cos_table[tok * hd + i_f] = f.cos();
                    cos_table[tok * hd + n_freqs + i_f] = f.cos();
                    sin_table[tok * hd + i_f] = f.sin();
                    sin_table[tok * hd + n_freqs + i_f] = f.sin();
                }
                // y-dim freqs (next spatial_dim entries).
                for i_f in 0..n_freqs {
                    let f = pos_y * inv_freq[i_f];
                    cos_table[tok * hd + spatial_dim + i_f] = f.cos();
                    cos_table[tok * hd + spatial_dim + n_freqs + i_f] = f.cos();
                    sin_table[tok * hd + spatial_dim + i_f] = f.sin();
                    sin_table[tok * hd + spatial_dim + n_freqs + i_f] = f.sin();
                }
            }
        }
        // Helper: apply 2D RoPE to a per-token [hd] slice. Out-of-place to
        // avoid the in-place rotate-half overlap bug.
        // num_rotated_channels_per_dim = 2 * (hd // (2*ndim)) = 36 for hd=72.
        // Split [hd] into 2 chunks of 36 (x-dim, y-dim); within each chunk,
        // rotate_half(chunk)[k] = -chunk[k+18] if k<18, else chunk[k-18].
        // Apply: out[k] = chunk[k]*cos[k] + rotate_half(chunk)[k]*sin[k].
        let apply_rope = |x: &[f32]| -> Vec<f32> {
            let mut out = vec![0f32; x.len()];
            for tok in 0..n_patches {
                let cb = tok * hd;
                for head in 0..nh {
                    let base = tok * h + head * hd;
                    // Chunk x-dim (indices 0..spatial_dim) and Chunk y-dim
                    // (indices spatial_dim..hd) are independent rotations.
                    for chunk_start in [0usize, spatial_dim] {
                        for k in 0..spatial_dim {
                            let i_in = base + chunk_start + k;
                            let pair_k = if k < n_freqs { k + n_freqs } else { k - n_freqs };
                            let pair_in = base + chunk_start + pair_k;
                            let rot = if k < n_freqs { -x[pair_in] } else { x[pair_in] };
                            let cos_k = cos_table[cb + chunk_start + k];
                            let sin_k = sin_table[cb + chunk_start + k];
                            out[i_in] = x[i_in] * cos_k + rot * sin_k;
                        }
                    }
                }
            }
            out
        };


        let log_progress = std::env::var("AEGIS_VISION_PROGRESS").is_ok();
        for li in 0..s.num_layers {
            let t_layer = std::time::Instant::now();
            let layer = &self.layers[li];

            // Upload current state f32 [n_patches, h] → BF16 input for matmul.
            let cur_f32 = runtime.upload_f32(&state)?;
            let mut cur_bf16 = runtime.alloc_u16(n_patches * h)?;
            runtime.f32_to_bf16_device(&cur_f32, n_patches * h, &mut cur_bf16)?;

            // input_layernorm(state).
            let mut normed_f32 = runtime.alloc_f32(n_patches * h)?;
            runtime.rms_norm_batched_device(
                &cur_f32, &layer.input_layernorm, n_patches, eps, &mut normed_f32)?;
            let mut normed_bf16 = runtime.alloc_u16(n_patches * h)?;
            runtime.f32_to_bf16_device(&normed_f32, n_patches * h, &mut normed_bf16)?;

            // Q = normed @ q_proj.T  ; K, V similarly. q_proj is [h, h].
            let proj_attn = |w: &crate::cuda::DeviceBf16Matrix| -> Result<Vec<f32>> {
                let mut out_bf16 = runtime.alloc_u16(n_patches * h)?;
                let mut out_f32  = runtime.alloc_f32(n_patches * h)?;
                runtime.matmul_bf16_cublaslt_with_input_bf16_device(
                    w, &normed_bf16, n_patches, &mut out_bf16, &mut out_f32)?;
                runtime.download_f32(&out_f32)
            };
            let q = proj_attn(&layer.q_proj)?;
            let k = proj_attn(&layer.k_proj)?;
            let v = proj_attn(&layer.v_proj)?;

            // Per-head QK-norm + V-norm (no-scale per HF: v_norm has with_scale=False).
            // RMSNorm: x * rsqrt(mean(x²) + eps) [* weight]
            let qn_w = runtime.download_f32(&layer.q_norm)?;
            let kn_w = runtime.download_f32(&layer.k_norm)?;
            let mut q_n = vec![0f32; q.len()];
            let mut k_n = vec![0f32; k.len()];
            let mut v_n = vec![0f32; v.len()];
            for t in 0..n_patches {
                for head in 0..nh {
                    let mut q_sum = 0f32;
                    let mut k_sum = 0f32;
                    let mut v_sum = 0f32;
                    for d in 0..hd {
                        let off = t * h + head * hd + d;
                        q_sum += q[off] * q[off];
                        k_sum += k[off] * k[off];
                        v_sum += v[off] * v[off];
                    }
                    let q_rms = 1.0 / ((q_sum / hd as f32) + eps).sqrt();
                    let k_rms = 1.0 / ((k_sum / hd as f32) + eps).sqrt();
                    let v_rms = 1.0 / ((v_sum / hd as f32) + eps).sqrt();
                    for d in 0..hd {
                        let off = t * h + head * hd + d;
                        q_n[off] = q[off] * q_rms * qn_w[d];
                        k_n[off] = k[off] * k_rms * kn_w[d];
                        v_n[off] = v[off] * v_rms;  // no scale
                    }
                }
            }
            let v = v_n;  // use normed V for the attention
            // Apply 2D RoPE to Q and K (V does NOT get RoPE).
            let q_n = apply_rope(&q_n);
            let k_n = apply_rope(&k_n);

            // Bidirectional attention on GPU via aegis_vision_bidi_attn kernel.
            // Uploads normed Q/K/V to VRAM, runs the fused QK·softmax·PV
            // kernel, downloads the [n_patches, h] result.
            let scale = 1.0 / (hd as f32).sqrt();
            let q_gpu = runtime.upload_f32(&q_n)?;
            let k_gpu = runtime.upload_f32(&k_n)?;
            let v_gpu = runtime.upload_f32(&v)?;
            let mut attn_gpu = runtime.alloc_f32(n_patches * h)?;
            runtime.vision_bidi_attn_device(
                &q_gpu, &k_gpu, &v_gpu,
                n_patches, nh, hd, scale,
                &mut attn_gpu,
            )?;
            let attn_out = runtime.download_f32(&attn_gpu)?;

            // o_proj: attn_out @ o_proj.T → [n_patches, h].
            let attn_f32 = runtime.upload_f32(&attn_out)?;
            let mut attn_bf16 = runtime.alloc_u16(attn_out.len())?;
            runtime.f32_to_bf16_device(&attn_f32, attn_out.len(), &mut attn_bf16)?;
            let mut o_bf16 = runtime.alloc_u16(n_patches * h)?;
            let mut o_f32  = runtime.alloc_f32(n_patches * h)?;
            runtime.matmul_bf16_cublaslt_with_input_bf16_device(
                &layer.o_proj, &attn_bf16, n_patches, &mut o_bf16, &mut o_f32)?;

            // post_attention_layernorm( o_proj_out ) — applied to the attention
            // output BEFORE residual add (Gemma 4 4-LN-per-block convention).
            let mut o_post = runtime.alloc_f32(n_patches * h)?;
            runtime.rms_norm_batched_device(
                &o_f32, &layer.post_attention_layernorm, n_patches, eps, &mut o_post)?;
            // residual: state += o_post.
            let o_post_host = runtime.download_f32(&o_post)?;
            for (i, x) in state.iter_mut().enumerate() { *x += o_post_host[i]; }

            // ── MLP branch.
            let cur_f32 = runtime.upload_f32(&state)?;
            let mut pre_ff = runtime.alloc_f32(n_patches * h)?;
            runtime.rms_norm_batched_device(
                &cur_f32, &layer.pre_feedforward_layernorm, n_patches, eps, &mut pre_ff)?;
            let mut pre_ff_bf16 = runtime.alloc_u16(n_patches * h)?;
            runtime.f32_to_bf16_device(&pre_ff, n_patches * h, &mut pre_ff_bf16)?;
            // gate = pre_ff @ gate_proj.T  ;  up = pre_ff @ up_proj.T  →  [n_patches, intermediate]
            let mut gate_bf16 = runtime.alloc_u16(n_patches * i)?;
            let mut gate_f32  = runtime.alloc_f32(n_patches * i)?;
            runtime.matmul_bf16_cublaslt_with_input_bf16_device(
                &layer.mlp_gate, &pre_ff_bf16, n_patches, &mut gate_bf16, &mut gate_f32)?;
            let mut up_bf16 = runtime.alloc_u16(n_patches * i)?;
            let mut up_f32  = runtime.alloc_f32(n_patches * i)?;
            runtime.matmul_bf16_cublaslt_with_input_bf16_device(
                &layer.mlp_up, &pre_ff_bf16, n_patches, &mut up_bf16, &mut up_f32)?;
            // swiglu(gate, up) → [n_patches, intermediate]
            let mut activated = runtime.alloc_f32(n_patches * i)?;
            runtime.swiglu_device(&gate_f32, &up_f32, &mut activated)?;
            // down: activated @ down_proj.T → [n_patches, h]
            let mut act_bf16 = runtime.alloc_u16(n_patches * i)?;
            runtime.f32_to_bf16_device(&activated, n_patches * i, &mut act_bf16)?;
            let mut down_bf16 = runtime.alloc_u16(n_patches * h)?;
            let mut down_f32  = runtime.alloc_f32(n_patches * h)?;
            runtime.matmul_bf16_cublaslt_with_input_bf16_device(
                &layer.mlp_down, &act_bf16, n_patches, &mut down_bf16, &mut down_f32)?;
            // post_feedforward_layernorm(down_out)  +  residual.
            let mut post_ff = runtime.alloc_f32(n_patches * h)?;
            runtime.rms_norm_batched_device(
                &down_f32, &layer.post_feedforward_layernorm, n_patches, eps, &mut post_ff)?;
            let post_ff_host = runtime.download_f32(&post_ff)?;
            for (i, x) in state.iter_mut().enumerate() { *x += post_ff_host[i]; }
            if log_progress {
                eprintln!("  vision layer {:>2}/{}: {:.2}s", li + 1, s.num_layers, t_layer.elapsed().as_secs_f64());
            }
        }

        // ── Final standardization (per HF Gemma4VisionModel.config.standardize):
        // hidden_states = (hidden_states - std_bias) * std_scale
        // NOTE the order: subtract bias FIRST, then multiply by scale.
        let std_scale = runtime.download_f32(&self.std_scale)?;
        let std_bias  = runtime.download_f32(&self.std_bias)?;
        for t in 0..n_patches {
            for c in 0..h {
                let v = state[t * h + c];
                state[t * h + c] = (v - std_bias[c]) * std_scale[c];
            }
        }

        // ── 3×3 average pool over the patch grid (stride 3, no overlap).
        // Per HF Gemma4VisionPooler: pooled = avg_pool(hidden) * sqrt(hidden_size).
        let pool = s.pooling_kernel_size;
        let n_tok_h = n_patches_h / pool;
        let n_tok_w = n_patches_w / pool;
        let n_tokens = n_tok_h * n_tok_w;
        let mut pooled = vec![0f32; n_tokens * h];
        let pool_norm = 1.0 / (pool * pool) as f32;
        let root_h = (h as f32).sqrt();
        for th in 0..n_tok_h {
            for tw in 0..n_tok_w {
                for c in 0..h {
                    let mut sum = 0f32;
                    for dh in 0..pool {
                        for dw in 0..pool {
                            let ph = th * pool + dh;
                            let pw = tw * pool + dw;
                            sum += state[(ph * n_patches_w + pw) * h + c];
                        }
                    }
                    pooled[(th * n_tok_w + tw) * h + c] = sum * pool_norm * root_h;
                }
            }
        }

        // ── Multimodal embedder (Gemma4MultimodalEmbedder):
        //   1. RMSNorm-no-scale over the pooler output (per-token, dim=vision_hidden).
        //      Brings the per-token RMS to ~1 before the projection — without
        //      this, the projector output magnitude blows up by ~sqrt(hidden)
        //      and the LLM treats it as noise.
        //   2. Linear projection vision_hidden → text_hidden.
        let mut pooled_normed = vec![0f32; pooled.len()];
        for t in 0..n_tokens {
            let mut sum = 0f32;
            for c in 0..h {
                let v = pooled[t * h + c];
                sum += v * v;
            }
            let rms = 1.0 / ((sum / h as f32) + eps).sqrt();
            for c in 0..h {
                pooled_normed[t * h + c] = pooled[t * h + c] * rms;
            }
        }

        // Projector: [n_tokens, vision_hidden] @ projector.T → [n_tokens, text_hidden].
        let pooled_f32 = runtime.upload_f32(&pooled_normed)?;
        let mut pooled_bf16 = runtime.alloc_u16(n_tokens * h)?;
        runtime.f32_to_bf16_device(&pooled_f32, n_tokens * h, &mut pooled_bf16)?;
        let text_hidden = self.projector.rows;
        let mut out_bf16 = runtime.alloc_u16(n_tokens * text_hidden)?;
        let mut out_f32  = runtime.alloc_f32(n_tokens * text_hidden)?;
        runtime.matmul_bf16_cublaslt_with_input_bf16_device(
            &self.projector, &pooled_bf16, n_tokens, &mut out_bf16, &mut out_f32)?;

        runtime.download_f32(&out_f32)
    }

    /// GPU-only forward pass (Stage I.4). Same math as `forward()` but every
    /// op (pixel rescale, position embedding add, QK/V norm, 2D RoPE, attention,
    /// residual adds, std-norm, 3×3 avg pool, projector) executes on the GPU.
    /// One upload of raw patches, one download of the projected output. The
    /// per-layer download/upload round-trips in the CPU forward are gone.
    pub fn forward_gpu(
        &self,
        runtime: &crate::cuda::CudaRuntime,
        patches: &[f32],
        n_patches_h: usize,
        n_patches_w: usize,
    ) -> Result<Vec<f32>> {
        let s = &self.shape;
        let n_patches = n_patches_h * n_patches_w;
        let patch_dim = 3 * s.patch_size * s.patch_size;
        if patches.len() != n_patches * patch_dim {
            return Err(AegisError::InvalidPlan(format!(
                "vision forward_gpu: patches len={} != n_patches({}) * patch_dim({}) = {}",
                patches.len(), n_patches, patch_dim, n_patches * patch_dim
            )));
        }
        let h  = s.hidden_size;
        let nh = s.num_attention_heads;
        let hd = s.head_dim;
        let i_  = s.intermediate_size;
        let eps = s.rms_norm_eps;
        let log_progress = std::env::var("AEGIS_VISION_PROGRESS").is_ok();
        let dump_prefix = std::env::var("AEGIS_VISION_DUMP").ok();
        let dump = |stage: &str, buf: &crate::cuda::DeviceBuffer<f32>| -> Result<()> {
            if let Some(ref p) = dump_prefix {
                let v = runtime.download_f32(buf)?;
                let path = format!("{}.{}.bin", p, stage);
                let bytes: Vec<u8> = v.iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect();
                std::fs::write(&path, bytes)
                    .map_err(|e| AegisError::InvalidPlan(format!("dump write {path}: {e}")))?;
                eprintln!("  dump {stage}: {} f32 → {path}", v.len());
            }
            Ok(())
        };

        // ── Phase 1: pixel rescale + patch_embed.
        // Upload patches (already in [0, 1] from preprocess), rescale on GPU
        // to [-1, 1] per HF Gemma4VisionPatchEmbedder.
        let mut patches_f32 = runtime.upload_f32(patches)?;
        runtime.vision_pixel_rescale_device(&mut patches_f32, patches.len())?;
        dump("after_rescale", &patches_f32)?;
        let mut patches_bf16 = runtime.alloc_u16(patches.len())?;
        runtime.f32_to_bf16_device(&patches_f32, patches.len(), &mut patches_bf16)?;

        // patch_embed: [n_patches, P²·3] @ patch_embed.T → [n_patches, hidden].
        let mut hidden_bf16 = runtime.alloc_u16(n_patches * h)?;
        let mut state = runtime.alloc_f32(n_patches * h)?;
        runtime.matmul_bf16_cublaslt_with_input_bf16_device(
            &self.patch_embed, &patches_bf16, n_patches,
            &mut hidden_bf16, &mut state,
        )?;
        dump("after_patch_embed", &state)?;

        // ── Phase 2: add 2D-axial position embeddings (GPU kernel).
        runtime.vision_pos_embed_add_device(
            &mut state,
            self.position_table.values_u16(),
            n_patches_h, n_patches_w,
            s.position_embedding_size,
            h,
        )?;
        dump("after_pos_embed", &state)?;

        // ── Phase 3: 27 transformer layers, all on GPU.
        // Scratch buffers reused across layers (allocate once).
        let mut normed = runtime.alloc_f32(n_patches * h)?;
        let mut normed_bf16 = runtime.alloc_u16(n_patches * h)?;
        let mut q_buf = runtime.alloc_f32(n_patches * h)?;
        let mut k_buf = runtime.alloc_f32(n_patches * h)?;
        let mut v_buf = runtime.alloc_f32(n_patches * h)?;
        let mut qkv_bf16 = runtime.alloc_u16(n_patches * h)?;
        let mut attn_out = runtime.alloc_f32(n_patches * h)?;
        let mut attn_bf16 = runtime.alloc_u16(n_patches * h)?;
        let mut o_out_bf16 = runtime.alloc_u16(n_patches * h)?;
        let mut o_out = runtime.alloc_f32(n_patches * h)?;
        let mut o_post = runtime.alloc_f32(n_patches * h)?;
        let mut pre_ff = runtime.alloc_f32(n_patches * h)?;
        let mut pre_ff_bf16 = runtime.alloc_u16(n_patches * h)?;
        let mut gate_bf16 = runtime.alloc_u16(n_patches * i_)?;
        let mut gate_f32 = runtime.alloc_f32(n_patches * i_)?;
        let mut up_bf16 = runtime.alloc_u16(n_patches * i_)?;
        let mut up_f32 = runtime.alloc_f32(n_patches * i_)?;
        let mut act_f32 = runtime.alloc_f32(n_patches * i_)?;
        let mut act_bf16 = runtime.alloc_u16(n_patches * i_)?;
        let mut down_bf16 = runtime.alloc_u16(n_patches * h)?;
        let mut down_f32 = runtime.alloc_f32(n_patches * h)?;
        let mut post_ff = runtime.alloc_f32(n_patches * h)?;
        // HF Gemma4VisionAttention sets `self.scaling = 1.0` and passes that
        // directly into eager_attention_forward — no `/sqrt(head_dim)` factor.
        // Q/K already pass through per-head RMSNorm so their magnitudes are
        // bounded, but the attention logits still get the raw QK^T scale.
        let scale = 1.0_f32;

        // Optional fast attention path: replaces the naive `vision_bidi_attn`
        // kernel (one block per (head, q_row), bandwidth-bound at large n_tok)
        // with two cuBLASLt strided-batched F32 GEMMs around a row-softmax.
        // Q-tiled (Bq = 128) so the scores chunk is `nh × Bq × n_tok × 4 B`
        // (~71 MB at 8694 tok × 16 heads, vs ~4.8 GB for a single un-tiled
        // call) — fits alongside the 262K-context KV cache pre-reservation.
        let fast_attn = std::env::var("AEGIS_VISION_FAST_ATTN").is_ok();
        let bq: usize = std::env::var("AEGIS_VISION_ATTN_Q_TILE")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(128);
        let mut scores_buf = if fast_attn {
            runtime.alloc_f32(nh * bq * n_patches)?
        } else {
            runtime.alloc_f32(1)?
        };

        for li in 0..s.num_layers {
            let t_layer = std::time::Instant::now();
            let layer = &self.layers[li];

            // input_layernorm(state) → normed
            runtime.rms_norm_batched_device(
                &state, &layer.input_layernorm, n_patches, eps, &mut normed)?;
            runtime.f32_to_bf16_device(&normed, n_patches * h, &mut normed_bf16)?;

            // Q/K/V projections.
            let mut proj = |w: &crate::cuda::DeviceBf16Matrix, out: &mut crate::cuda::DeviceBuffer<f32>| -> Result<()> {
                runtime.matmul_bf16_cublaslt_with_input_bf16_device(
                    w, &normed_bf16, n_patches, &mut qkv_bf16, out)?;
                Ok(())
            };
            proj(&layer.q_proj, &mut q_buf)?;
            proj(&layer.k_proj, &mut k_buf)?;
            proj(&layer.v_proj, &mut v_buf)?;

            // Per-head Q-norm, K-norm (with weight), V-norm (no weight).
            runtime.vision_head_rmsnorm_device(
                &mut q_buf, Some(&layer.q_norm), n_patches, nh, hd, eps)?;
            runtime.vision_head_rmsnorm_device(
                &mut k_buf, Some(&layer.k_norm), n_patches, nh, hd, eps)?;
            runtime.vision_head_rmsnorm_device(
                &mut v_buf, None, n_patches, nh, hd, eps)?;

            // 2D RoPE on Q and K (V does NOT get RoPE).
            runtime.vision_rope_2d_device(
                &mut q_buf, n_patches, n_patches_w, nh, hd, s.rope_theta)?;
            runtime.vision_rope_2d_device(
                &mut k_buf, n_patches, n_patches_w, nh, hd, s.rope_theta)?;

            if fast_attn {
                // Q-tiled cuBLASLt batched attention. Loop over Q in chunks
                // of `bq` rows; per chunk compute scores[h, i in q_chunk, j] =
                // Q[i,h,:]·K[j,h,:] via batched F32 GEMM (TF32 TC), then
                // row-softmax, then out[i,h,d] = scores · V via batched F32
                // GEMM. Scores chunk is `nh * bq * n_patches × 4 B` (~71 MB
                // at bq=128, n_patches=8694, nh=16) — fits with KV cache.
                let mut q_start = 0usize;
                while q_start < n_patches {
                    let q_end = (q_start + bq).min(n_patches);
                    let q_len = q_end - q_start;
                    // QK^T: per head h, scores_chunk[h, i, j] = Q[i+q_start, h, :] · K[j, h, :]
                    // A=K (per head offset h*hd, lda=nh*hd, stride_a=hd) — covers all n_patches K rows.
                    // B=Q chunk (per head offset q_start*nh*hd + h*hd, ldb=nh*hd, stride_b=hd) — q_len rows.
                    // C=scores_chunk (per head [q_len, n_patches] tight, ldc=n_patches, stride_c=q_len*n_patches).
                    runtime.f32_strided_batched_gemm_device(
                        &k_buf, 0, n_patches * nh * hd,
                        &q_buf, q_start * nh * hd, q_len * nh * hd,
                        &mut scores_buf, 0, nh * q_len * n_patches,
                        /* transa */ true, /* transb */ false,
                        /* m */ n_patches, /* n */ q_len, /* k */ hd,
                        /* lda */ nh * hd, /* ldb */ nh * hd,
                        /* ldc */ n_patches,
                        /* stride_a */ hd, /* stride_b */ hd,
                        /* stride_c */ q_len * n_patches,
                        /* batch */ nh, /* alpha */ scale, /* beta */ 0.0,
                    )?;
                    // Row-softmax over [nh*q_len, n_patches] in place.
                    runtime.vision_row_softmax_device(
                        &mut scores_buf, nh * q_len, n_patches, 1.0,
                    )?;
                    // PV: per head h, out_chunk[i, h, d] = sum_j scores_chunk[h, i, j] * V[j, h, d]
                    // A=V (per head offset h*hd, lda=nh*hd, stride_a=hd).
                    // B=scores_chunk per head [q_len, n_patches] (ldb=n_patches, stride_b=q_len*n_patches).
                    // C=attn_out chunk (q_start*nh*hd + h*hd, ldc=nh*hd, stride_c=hd).
                    runtime.f32_strided_batched_gemm_device(
                        &v_buf, 0, n_patches * nh * hd,
                        &scores_buf, 0, nh * q_len * n_patches,
                        &mut attn_out, q_start * nh * hd, q_len * nh * hd,
                        /* transa */ false, /* transb */ false,
                        /* m */ hd, /* n */ q_len, /* k */ n_patches,
                        /* lda */ nh * hd, /* ldb */ n_patches,
                        /* ldc */ nh * hd,
                        /* stride_a */ hd,
                        /* stride_b */ q_len * n_patches,
                        /* stride_c */ hd,
                        /* batch */ nh, /* alpha */ 1.0, /* beta */ 0.0,
                    )?;
                    q_start = q_end;
                }
            } else {
                // Naive fused kernel (correct, but bandwidth-bound past ~3k tok).
                runtime.vision_bidi_attn_device(
                    &q_buf, &k_buf, &v_buf,
                    n_patches, nh, hd, scale,
                    &mut attn_out,
                )?;
            }

            // o_proj: attn_out @ o_proj.T → o_out.
            runtime.f32_to_bf16_device(&attn_out, n_patches * h, &mut attn_bf16)?;
            runtime.matmul_bf16_cublaslt_with_input_bf16_device(
                &layer.o_proj, &attn_bf16, n_patches,
                &mut o_out_bf16, &mut o_out,
            )?;

            // post_attention_layernorm(o_out) → o_post; then state += o_post.
            runtime.rms_norm_batched_device(
                &o_out, &layer.post_attention_layernorm, n_patches, eps, &mut o_post)?;
            runtime.add_inplace_device_len(&mut state, &o_post, n_patches * h)?;

            // pre_feedforward_layernorm(state) → pre_ff.
            runtime.rms_norm_batched_device(
                &state, &layer.pre_feedforward_layernorm, n_patches, eps, &mut pre_ff)?;
            runtime.f32_to_bf16_device(&pre_ff, n_patches * h, &mut pre_ff_bf16)?;
            // gate, up.
            runtime.matmul_bf16_cublaslt_with_input_bf16_device(
                &layer.mlp_gate, &pre_ff_bf16, n_patches,
                &mut gate_bf16, &mut gate_f32)?;
            runtime.matmul_bf16_cublaslt_with_input_bf16_device(
                &layer.mlp_up, &pre_ff_bf16, n_patches,
                &mut up_bf16, &mut up_f32)?;
            // swiglu(gate, up) → activated.
            runtime.swiglu_device(&gate_f32, &up_f32, &mut act_f32)?;
            // down: act @ down_proj.T → down_f32.
            runtime.f32_to_bf16_device(&act_f32, n_patches * i_, &mut act_bf16)?;
            runtime.matmul_bf16_cublaslt_with_input_bf16_device(
                &layer.mlp_down, &act_bf16, n_patches,
                &mut down_bf16, &mut down_f32)?;
            // post_feedforward_layernorm(down) → post_ff; state += post_ff.
            runtime.rms_norm_batched_device(
                &down_f32, &layer.post_feedforward_layernorm, n_patches, eps, &mut post_ff)?;
            runtime.add_inplace_device_len(&mut state, &post_ff, n_patches * h)?;

            if log_progress {
                eprintln!(
                    "  vision-gpu layer {:>2}/{}: {:.3}s",
                    li + 1, s.num_layers, t_layer.elapsed().as_secs_f64()
                );
            }
            if li == 0 {
                dump("after_layer0", &state)?;
            }
        }
        dump("after_all_layers", &state)?;

        // ── 3×3 average pool + sqrt(hidden) pooler scale on GPU.
        // Order matches HF `Gemma4VisionModel.forward` EXACTLY:
        //   encoder → pool*sqrt(hidden) → standardize → projector(RMSNorm+linear).
        // We previously had standardize BEFORE pool which produced a bias
        // term off by `sqrt(hidden) ≈ 34×` and degraded after_projector
        // cosine vs HF (visible as cos=0.84 at after_all_layers compounding).
        let pool = s.pooling_kernel_size;
        let n_th = n_patches_h / pool;
        let n_tw = n_patches_w / pool;
        let n_tokens = n_th * n_tw;
        let pooler_scale = (h as f32).sqrt() / (pool * pool) as f32;
        let mut pooled = runtime.alloc_f32(n_tokens * h)?;
        runtime.vision_pool3x3_scale_device(
            &state, &mut pooled,
            n_patches_h, n_patches_w, n_th, n_tw,
            h, pool, pooler_scale,
        )?;
        dump("after_pool", &pooled)?;

        // ── Final standardization on the POOLED tokens (n_tokens, h).
        runtime.vision_standardize_device(
            &mut pooled, &self.std_scale, &self.std_bias, n_tokens, h,
        )?;
        dump("after_std", &pooled)?;

        // ── Multimodal embedder: RMSNorm-no-scale then projector.
        let mut pooled_normed = runtime.alloc_f32(n_tokens * h)?;
        runtime.rms_norm_batched_no_weight_device(
            &pooled, n_tokens, h, eps, &mut pooled_normed,
        )?;
        let mut pooled_bf16 = runtime.alloc_u16(n_tokens * h)?;
        runtime.f32_to_bf16_device(&pooled_normed, n_tokens * h, &mut pooled_bf16)?;

        // Projector: pooled @ projector.T → [n_tokens, text_hidden].
        let text_hidden = self.projector.rows;
        let mut out_bf16 = runtime.alloc_u16(n_tokens * text_hidden)?;
        let mut out_f32 = runtime.alloc_f32(n_tokens * text_hidden)?;
        runtime.matmul_bf16_cublaslt_with_input_bf16_device(
            &self.projector, &pooled_bf16, n_tokens,
            &mut out_bf16, &mut out_f32,
        )?;
        dump("after_projector", &out_f32)?;

        runtime.download_f32(&out_f32)
    }
}

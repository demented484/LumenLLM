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
        let patches_f32 = runtime.upload_f32(patches)?;
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

        let mut hidden_host = runtime.download_f32(&hidden_f32)?;
        for ph in 0..n_patches_h {
            let r_idx = (ph as f32 / n_patches_h.max(1) as f32 * n_table as f32) as usize;
            let r_idx = r_idx.min(n_table - 1);
            for pw in 0..n_patches_w {
                let c_idx = (pw as f32 / n_patches_w.max(1) as f32 * n_table as f32) as usize;
                let c_idx = c_idx.min(n_table - 1);
                let tok = ph * n_patches_w + pw;
                for k in 0..s.hidden_size {
                    let row_emb = pos_table_f32[r_idx * s.hidden_size + k];
                    // slot 1 starts at n_table rows
                    let col_emb = pos_table_f32[(n_table + c_idx) * s.hidden_size + k];
                    hidden_host[tok * s.hidden_size + k] += row_emb + col_emb;
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

            // Per-head QK-norm.
            // Download q_norm/k_norm weights once (shape [head_dim] each).
            let qn_w = runtime.download_f32(&layer.q_norm)?;
            let kn_w = runtime.download_f32(&layer.k_norm)?;
            let mut q_n = vec![0f32; q.len()];
            let mut k_n = vec![0f32; k.len()];
            for t in 0..n_patches {
                for head in 0..nh {
                    // Compute RMS over head_dim.
                    let mut q_sum = 0f32;
                    let mut k_sum = 0f32;
                    for d in 0..hd {
                        let off = t * h + head * hd + d;
                        q_sum += q[off] * q[off];
                        k_sum += k[off] * k[off];
                    }
                    let q_rms = 1.0 / ((q_sum / hd as f32) + eps).sqrt();
                    let k_rms = 1.0 / ((k_sum / hd as f32) + eps).sqrt();
                    for d in 0..hd {
                        let off = t * h + head * hd + d;
                        q_n[off] = q[off] * q_rms * qn_w[d];
                        k_n[off] = k[off] * k_rms * kn_w[d];
                    }
                }
            }

            // Bidirectional attention. Per-head parallel via rayon — heads are
            // independent. Each head's work is O(n²·hd) ≈ 0.4M·72 = 30M FMAs
            // for the score grid, identical for the V-sum. With 16 heads
            // running in parallel across ~16 cores this brings the 27-layer
            // wall from minutes to seconds even with CPU softmax.
            let scale = 1.0 / (hd as f32).sqrt();
            let q_n_ref = &q_n;
            let k_n_ref = &k_n;
            let v_ref   = &v;
            let head_outputs: Vec<Vec<f32>> = (0..nh).into_par_iter().map(|head| {
                // Slice per-head views (just indices into the [n_patches, h] arrays).
                let mut scores = vec![0f32; n_patches * n_patches];
                // Q·K^T scaled.
                for ti in 0..n_patches {
                    let q_off = ti * h + head * hd;
                    for tj in 0..n_patches {
                        let k_off = tj * h + head * hd;
                        let mut s_ij = 0f32;
                        for d in 0..hd {
                            s_ij += q_n_ref[q_off + d] * k_n_ref[k_off + d];
                        }
                        scores[ti * n_patches + tj] = s_ij * scale;
                    }
                }
                // Row-softmax.
                for ti in 0..n_patches {
                    let row = &mut scores[ti * n_patches .. (ti + 1) * n_patches];
                    let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0f32;
                    for x in row.iter_mut() { *x = (*x - mx).exp(); sum += *x; }
                    let inv = 1.0 / sum;
                    for x in row.iter_mut() { *x *= inv; }
                }
                // P·V: per-head output [n_patches, hd].
                let mut head_out = vec![0f32; n_patches * hd];
                for ti in 0..n_patches {
                    for d in 0..hd {
                        let mut acc = 0f32;
                        for tj in 0..n_patches {
                            acc += scores[ti * n_patches + tj] * v_ref[tj * h + head * hd + d];
                        }
                        head_out[ti * hd + d] = acc;
                    }
                }
                head_out
            }).collect();
            // Scatter heads back into [n_patches, h].
            let mut attn_out = vec![0f32; n_patches * h];
            for (head, head_out) in head_outputs.iter().enumerate() {
                for ti in 0..n_patches {
                    for d in 0..hd {
                        attn_out[ti * h + head * hd + d] = head_out[ti * hd + d];
                    }
                }
            }

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

        // ── Final standardization: state = std_scale * state + std_bias
        // (per-channel affine, NOT RMSNorm — it's a learned scale+bias
        // applied without any normalization).
        let std_scale = runtime.download_f32(&self.std_scale)?;
        let std_bias  = runtime.download_f32(&self.std_bias)?;
        for t in 0..n_patches {
            for c in 0..h {
                let v = state[t * h + c];
                state[t * h + c] = v * std_scale[c] + std_bias[c];
            }
        }

        // ── 3×3 average pool over the patch grid (stride 3, no overlap).
        let pool = s.pooling_kernel_size;
        let n_tok_h = n_patches_h / pool;
        let n_tok_w = n_patches_w / pool;
        let n_tokens = n_tok_h * n_tok_w;
        let mut pooled = vec![0f32; n_tokens * h];
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
                    pooled[(th * n_tok_w + tw) * h + c] = sum / (pool * pool) as f32;
                }
            }
        }

        // ── Projector: [n_tokens, vision_hidden] @ projector.T  →  [n_tokens, text_hidden]
        // projector.weight = [text_hidden=2816, vision_hidden=1152].
        let pooled_f32 = runtime.upload_f32(&pooled)?;
        let mut pooled_bf16 = runtime.alloc_u16(n_tokens * h)?;
        runtime.f32_to_bf16_device(&pooled_f32, n_tokens * h, &mut pooled_bf16)?;
        let text_hidden = self.projector.rows;
        let mut out_bf16 = runtime.alloc_u16(n_tokens * text_hidden)?;
        let mut out_f32  = runtime.alloc_f32(n_tokens * text_hidden)?;
        runtime.matmul_bf16_cublaslt_with_input_bf16_device(
            &self.projector, &pooled_bf16, n_tokens, &mut out_bf16, &mut out_f32)?;

        runtime.download_f32(&out_f32)
    }
}

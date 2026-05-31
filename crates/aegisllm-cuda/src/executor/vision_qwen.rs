//! Qwen3-VL (Qwen3.5 / 3.6) native-ViT vision tower loader + GPU forward.
//!
//! Distinct from the Gemma `VisionTower` (different tensor map, biases on every
//! linear, LayerNorm-with-bias instead of RMSNorm, gelu_tanh MLP, 2×2 merger
//! instead of spatial pooling, standard rotate-half 2D RoPE instead of Gemma's
//! multidim-split). Loads all 333 `model.visual.*` BF16 tensors into VRAM.
//!
//! Tensor inventory (per the safetensors index):
//!   Tower-level:
//!     model.visual.patch_embed.proj.weight   [1152, 3, 2, 16, 16] → [1152, 1536]
//!     model.visual.patch_embed.proj.bias     [1152]
//!     model.visual.pos_embed.weight          [2304, 1152]   (48×48 learned grid)
//!   Per block (× depth=27):
//!     .attn.qkv.{weight,bias}                [3456, 1152] / [3456]
//!     .attn.proj.{weight,bias}               [1152, 1152] / [1152]
//!     .norm1.{weight,bias}                   [1152]
//!     .norm2.{weight,bias}                   [1152]
//!     .mlp.linear_fc1.{weight,bias}          [4304, 1152] / [4304]
//!     .mlp.linear_fc2.{weight,bias}          [1152, 4304] / [1152]
//!   Merger:
//!     .merger.norm.{weight,bias}             [1152]
//!     .merger.linear_fc1.{weight,bias}       [4608, 4608] / [4608]
//!     .merger.linear_fc2.{weight,bias}       [out_hidden, 4608] / [out_hidden]
//!
//! Forward (HF Qwen3_5VisionModel.forward, single still image → grid_t=1):
//!   patch_embed(linear+bias) → + bilinear-interp pos_embed → 27 blocks
//!   (LN1+bias → fused-QKV+bias → split → per-head RoPE on Q,K → full bidi attn
//!    scale 1/√hd → proj+bias → residual; LN2+bias → fc1+bias → gelu_tanh →
//!    fc2+bias → residual) → merger (norm+bias → 2×2 reshape → fc1 → GELU → fc2)
//!   → [n/4, out_hidden] image embeddings.

use aegisllm_base::artifact::ModelArtifact;
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::tensor::TensorInfo;
use aegisllm_base::tensor::storage::TensorStorageLoader;

use super::loader::cuda_residency_for_store;
use crate::cuda::loader::CudaWeightLoader;
use crate::cuda::{DeviceBf16Matrix, DeviceBuffer};
use aegisllm_base::planning::placement::StoragePlacement;

/// Static shape of the Qwen vision tower, from `vision_config`.
#[derive(Debug, Clone)]
pub struct QwenVisionShape {
    pub hidden_size: usize,      // 1152
    pub intermediate_size: usize, // 4304
    pub depth: usize,            // 27
    pub num_heads: usize,        // 16
    pub head_dim: usize,         // 72 (hidden/heads)
    pub patch_size: usize,       // 16
    pub temporal_patch_size: usize, // 2
    pub spatial_merge_size: usize,  // 2
    pub out_hidden_size: usize,  // 4096 (9B) / 2048 (35B)
    pub num_pos_embeddings: usize, // 2304 = 48×48
    pub num_grid_per_side: usize,  // 48
    pub in_channels: usize,      // 3
    pub rope_theta: f32,         // 10000
    pub ln_eps: f32,             // 1e-6
}

impl QwenVisionShape {
    pub fn from_artifact(artifact: &ModelArtifact) -> Result<Self> {
        let vc = artifact.config.vision_config.as_ref().ok_or_else(|| {
            AegisError::InvalidPlan(
                "Qwen vision tower requested but config.json has no `vision_config`".into(),
            )
        })?;
        let hidden = vc.hidden_size;
        let heads = vc.num_attention_heads;
        if heads == 0 || hidden % heads != 0 {
            return Err(AegisError::InvalidPlan(format!(
                "Qwen vision: hidden {hidden} not divisible by heads {heads}"
            )));
        }
        let head_dim = hidden / heads;
        let npos = vc.position_embedding_size;
        let side = (npos as f64).sqrt().round() as usize;
        if side * side != npos {
            return Err(AegisError::InvalidPlan(format!(
                "Qwen vision: num_position_embeddings {npos} is not a perfect square"
            )));
        }
        Ok(Self {
            hidden_size: hidden,
            intermediate_size: vc.intermediate_size,
            depth: vc.num_hidden_layers,
            num_heads: heads,
            head_dim,
            patch_size: vc.patch_size,
            temporal_patch_size: vc.temporal_patch_size.max(1),
            spatial_merge_size: vc.spatial_merge_size.max(1),
            out_hidden_size: if vc.out_hidden_size > 0 {
                vc.out_hidden_size
            } else {
                // Fallback: project to the text hidden via the merger fc2 rows
                // at load time; record 0 here and let the loader assert.
                0
            },
            num_pos_embeddings: npos,
            num_grid_per_side: side,
            in_channels: if vc.in_channels > 0 { vc.in_channels } else { 3 },
            rope_theta: vc
                .rope_parameters
                .as_ref()
                .map(|rp| rp.rope_theta as f32)
                .unwrap_or(10000.0),
            ln_eps: 1e-6,
        })
    }
}

/// One Qwen vision block's device-resident weights (all BF16 matrices; LN
/// weights + all biases as f32 vectors).
pub struct QwenVisionBlock {
    pub norm1_w: DeviceBuffer<f32>,
    pub norm1_b: DeviceBuffer<f32>,
    pub qkv: DeviceBf16Matrix, // [3*hidden, hidden]
    pub qkv_b: DeviceBuffer<f32>, // [3*hidden]
    pub proj: DeviceBf16Matrix, // [hidden, hidden]
    pub proj_b: DeviceBuffer<f32>, // [hidden]
    pub norm2_w: DeviceBuffer<f32>,
    pub norm2_b: DeviceBuffer<f32>,
    pub fc1: DeviceBf16Matrix, // [intermediate, hidden]
    pub fc1_b: DeviceBuffer<f32>,
    pub fc2: DeviceBf16Matrix, // [hidden, intermediate]
    pub fc2_b: DeviceBuffer<f32>,
}

pub struct QwenVisionTower {
    pub shape: QwenVisionShape,
    pub patch_embed: DeviceBf16Matrix, // [hidden, 1536]
    pub patch_embed_b: DeviceBuffer<f32>, // [hidden]
    /// Learned position-embedding table [num_pos, hidden] BF16 (kept on device
    /// for the bilinear-interp gather; we read it back to host once at forward).
    pub pos_embed: DeviceBf16Matrix, // [num_pos, hidden]
    pub blocks: Vec<QwenVisionBlock>,
    pub merger_norm_w: DeviceBuffer<f32>, // [hidden]
    pub merger_norm_b: DeviceBuffer<f32>,
    pub merger_fc1: DeviceBf16Matrix, // [hidden*merge², hidden*merge²]
    pub merger_fc1_b: DeviceBuffer<f32>,
    pub merger_fc2: DeviceBf16Matrix, // [out_hidden, hidden*merge²]
    pub merger_fc2_b: DeviceBuffer<f32>,
}

impl QwenVisionTower {
    pub fn from_artifact(
        artifact: &ModelArtifact,
        shape: QwenVisionShape,
        cuda_weights: &CudaWeightLoader<'_>,
        device_index: usize,
        loader: &mut TensorStorageLoader,
    ) -> Result<Self> {
        let store = StoragePlacement::Vram { device: device_index };
        let residency = cuda_residency_for_store(store, device_index)?;

        let get = |name: &str| -> Result<&TensorInfo> {
            artifact.tensors.tensors.get(name).ok_or_else(|| {
                AegisError::InvalidPlan(format!("Qwen vision: tensor `{name}` missing"))
            })
        };
        let mat = |name: &str, loader: &mut TensorStorageLoader| -> Result<DeviceBf16Matrix> {
            cuda_weights.load_bf16_matrix_with_store(get(name)?, store, residency.clone(), loader)
        };
        let vec = |name: &str, loader: &mut TensorStorageLoader| -> Result<DeviceBuffer<f32>> {
            cuda_weights.load_dense_vector_with_store(get(name)?, store, loader)
        };

        let h = shape.hidden_size;
        let inter = shape.intermediate_size;
        let merge_dim = h * shape.spatial_merge_size * shape.spatial_merge_size; // 4608

        // patch_embed.proj.weight is the Conv3d weight [hidden, 3, 2, 16, 16] =
        // the linear [hidden, 3*2*16*16=1536]. The generic matrix loader would
        // collapse it to [hidden*3*2*16, 16]; load with explicit (rows, cols).
        let expect_cols = shape.in_channels * shape.temporal_patch_size
            * shape.patch_size * shape.patch_size;
        let patch_embed = cuda_weights.load_bf16_matrix_explicit_dims(
            get("model.visual.patch_embed.proj.weight")?, h, expect_cols, loader)?;
        let patch_embed_b = vec("model.visual.patch_embed.proj.bias", loader)?;
        let pos_embed = mat("model.visual.pos_embed.weight", loader)?;
        if pos_embed.rows != shape.num_pos_embeddings || pos_embed.cols != h {
            return Err(AegisError::InvalidPlan(format!(
                "Qwen vision pos_embed: got [{}, {}] expected [{}, {}]",
                pos_embed.rows, pos_embed.cols, shape.num_pos_embeddings, h
            )));
        }

        let mut blocks = Vec::with_capacity(shape.depth);
        for li in 0..shape.depth {
            let p = |s: &str| format!("model.visual.blocks.{li}.{s}");
            let qkv = mat(&p("attn.qkv.weight"), loader)?;
            if qkv.rows != 3 * h || qkv.cols != h {
                return Err(AegisError::InvalidPlan(format!(
                    "Qwen vision block{li} qkv: got [{}, {}] expected [{}, {}]",
                    qkv.rows, qkv.cols, 3 * h, h
                )));
            }
            let fc1 = mat(&p("mlp.linear_fc1.weight"), loader)?;
            if fc1.rows != inter || fc1.cols != h {
                return Err(AegisError::InvalidPlan(format!(
                    "Qwen vision block{li} fc1: got [{}, {}] expected [{}, {}]",
                    fc1.rows, fc1.cols, inter, h
                )));
            }
            blocks.push(QwenVisionBlock {
                norm1_w: vec(&p("norm1.weight"), loader)?,
                norm1_b: vec(&p("norm1.bias"), loader)?,
                qkv,
                qkv_b: vec(&p("attn.qkv.bias"), loader)?,
                proj: mat(&p("attn.proj.weight"), loader)?,
                proj_b: vec(&p("attn.proj.bias"), loader)?,
                norm2_w: vec(&p("norm2.weight"), loader)?,
                norm2_b: vec(&p("norm2.bias"), loader)?,
                fc1,
                fc1_b: vec(&p("mlp.linear_fc1.bias"), loader)?,
                fc2: mat(&p("mlp.linear_fc2.weight"), loader)?,
                fc2_b: vec(&p("mlp.linear_fc2.bias"), loader)?,
            });
        }

        let merger_fc1 = mat("model.visual.merger.linear_fc1.weight", loader)?;
        if merger_fc1.rows != merge_dim || merger_fc1.cols != merge_dim {
            return Err(AegisError::InvalidPlan(format!(
                "Qwen vision merger fc1: got [{}, {}] expected [{}, {}]",
                merger_fc1.rows, merger_fc1.cols, merge_dim, merge_dim
            )));
        }
        let merger_fc2 = mat("model.visual.merger.linear_fc2.weight", loader)?;
        if merger_fc2.cols != merge_dim {
            return Err(AegisError::InvalidPlan(format!(
                "Qwen vision merger fc2: cols={} expected {}",
                merger_fc2.cols, merge_dim
            )));
        }
        // merger fc2 rows = out_hidden (project to text hidden).
        if shape.out_hidden_size != 0 && merger_fc2.rows != shape.out_hidden_size {
            return Err(AegisError::InvalidPlan(format!(
                "Qwen vision merger fc2 rows={} != out_hidden_size {}",
                merger_fc2.rows, shape.out_hidden_size
            )));
        }

        Ok(Self {
            shape,
            patch_embed,
            patch_embed_b,
            pos_embed,
            blocks,
            merger_norm_w: vec("model.visual.merger.norm.weight", loader)?,
            merger_norm_b: vec("model.visual.merger.norm.bias", loader)?,
            merger_fc1,
            merger_fc1_b: vec("model.visual.merger.linear_fc1.bias", loader)?,
            merger_fc2,
            merger_fc2_b: vec("model.visual.merger.linear_fc2.bias", loader)?,
        })
    }

    /// Text-hidden dim the merger projects image embeddings into.
    pub fn out_hidden(&self) -> usize {
        self.merger_fc2.rows
    }

    /// GPU forward: packed pixel patches → merged image embeddings in text space.
    ///
    /// Inputs:
    ///   * `pixel_values`: `[n_patches, embed_dim=1536]` row-major f32, the
    ///     Qwen2-VL packed patch matrix (merge-block ordered rows). This is the
    ///     output of `ImageProcessor::preprocess_qwen2vl(...).pixel_values`.
    ///   * `grid_h`, `grid_w`: the pre-merge patch grid (`H/patch`, `W/patch`).
    ///
    /// Output: `[n_merged, out_hidden]` row-major f32, ready to splice at
    /// `<|image_pad|>` slots, where `n_merged = (grid_h/m)·(grid_w/m)`.
    pub fn forward_gpu(
        &self,
        runtime: &crate::cuda::CudaRuntime,
        pixel_values: &[f32],
        grid_h: usize,
        grid_w: usize,
    ) -> Result<Vec<f32>> {
        let s = &self.shape;
        let h = s.hidden_size;
        let nh = s.num_heads;
        let hd = s.head_dim;
        let inter = s.intermediate_size;
        let m = s.spatial_merge_size;
        let eps = s.ln_eps;
        let n = grid_h * grid_w;
        let embed_dim = self.patch_embed.cols;
        if pixel_values.len() != n * embed_dim {
            return Err(AegisError::InvalidPlan(format!(
                "Qwen vision forward: pixel_values len={} != n({})*embed_dim({})={}",
                pixel_values.len(), n, embed_dim, n * embed_dim
            )));
        }
        if grid_h % m != 0 || grid_w % m != 0 {
            return Err(AegisError::InvalidPlan(format!(
                "Qwen vision forward: grid {grid_h}x{grid_w} not a multiple of merge {m}"
            )));
        }

        let log_progress = std::env::var("AEGIS_VISION_PROGRESS").is_ok();
        let dump_prefix = std::env::var("AEGIS_VISION_DUMP").ok();
        let dump = |stage: &str, buf: &DeviceBuffer<f32>, count: usize| -> Result<()> {
            if let Some(ref p) = dump_prefix {
                let v = runtime.download_f32(buf)?;
                let v = &v[..count.min(v.len())];
                let path = format!("{p}.{stage}.bin");
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                std::fs::write(&path, bytes)
                    .map_err(|e| AegisError::InvalidPlan(format!("dump {path}: {e}")))?;
                eprintln!("  dump {stage}: {} f32 → {path}", v.len());
            }
            Ok(())
        };

        // ── Patch embed: [n, 1536] @ patch_embed.T → [n, hidden], + bias.
        let px_f32 = runtime.upload_f32(pixel_values)?;
        let mut px_bf16 = runtime.alloc_u16(n * embed_dim)?;
        runtime.f32_to_bf16_device(&px_f32, n * embed_dim, &mut px_bf16)?;
        let mut state = runtime.alloc_f32(n * h)?;
        let mut state_bf16 = runtime.alloc_u16(n * h)?;
        runtime.matmul_bf16_cublaslt_with_input_bf16_device(
            &self.patch_embed, &px_bf16, n, &mut state_bf16, &mut state)?;
        runtime.vision_add_bias_rows_device(&mut state, &self.patch_embed_b, n, h)?;
        dump("after_patch_embed", &state, n * h)?;

        // ── Bilinear-interp position embedding (host gather, one-time add).
        // Pull pos_embed table to host, gather the 4 corners per token with the
        // host-computed indices/weights (merge-block ordered), add to state.
        {
            use aegisllm_base::modalities::mrope::qwen_bilinear_pos_indices_weights;
            let (idx, wts) = qwen_bilinear_pos_indices_weights(grid_h, grid_w, s.num_grid_per_side, m);
            let table = runtime.download_u16_slice(self.pos_embed.values_u16())?;
            let table_f32: Vec<f32> = table.iter()
                .map(|&hb| f32::from_bits((hb as u32) << 16))
                .collect();
            let mut pe = vec![0f32; n * h];
            for tok in 0..n {
                for c in 0..4 {
                    let ti = idx[c * n + tok] as usize;
                    let w = wts[c * n + tok];
                    let off = ti * h;
                    let dst = tok * h;
                    for k in 0..h {
                        pe[dst + k] += w * table_f32[off + k];
                    }
                }
            }
            let pe_dev = runtime.upload_f32(&pe)?;
            runtime.add_inplace_device_len(&mut state, &pe_dev, n * h)?;
        }
        dump("after_pos_embed", &state, n * h)?;

        // ── Vision RoPE position ids (row, col), merge-block ordered.
        let pos_ids = {
            use aegisllm_base::modalities::mrope::qwen_vision_position_ids;
            let host = qwen_vision_position_ids(grid_h, grid_w, m);
            runtime.upload_i32(&host)?
        };

        // ── 27 transformer blocks. Scratch reused across blocks.
        let mut normed = runtime.alloc_f32(n * h)?;
        let mut normed_bf16 = runtime.alloc_u16(n * h)?;
        // fused QKV output [n, 3h]; split into q/k/v [n, h].
        let mut qkv_bf16 = runtime.alloc_u16(n * 3 * h)?;
        let mut qkv_f32 = runtime.alloc_f32(n * 3 * h)?;
        let mut q_buf = runtime.alloc_f32(n * h)?;
        let mut k_buf = runtime.alloc_f32(n * h)?;
        let mut v_buf = runtime.alloc_f32(n * h)?;
        let mut attn_bf16 = runtime.alloc_u16(n * h)?;
        let mut o_bf16 = runtime.alloc_u16(n * h)?;
        let mut o_f32 = runtime.alloc_f32(n * h)?;
        let mut fc1_bf16 = runtime.alloc_u16(n * inter)?;
        let mut fc1_f32 = runtime.alloc_f32(n * inter)?;
        let mut fc1_act_bf16 = runtime.alloc_u16(n * inter)?;
        let mut fc2_bf16 = runtime.alloc_u16(n * h)?;
        let mut fc2_f32 = runtime.alloc_f32(n * h)?;

        // BF16 attention scratch (Q-tiled, same pattern as Gemma fast-attn).
        let bq: usize = std::env::var("AEGIS_VISION_ATTN_Q_TILE")
            .ok().and_then(|s| s.parse().ok()).unwrap_or(128);
        let mut q_bf16 = runtime.alloc_u16(n * h)?;
        let mut k_bf16 = runtime.alloc_u16(n * h)?;
        let mut v_bf16 = runtime.alloc_u16(n * h)?;
        let mut scores_bf16 = runtime.alloc_u16(nh * bq * n)?;
        let scale = 1.0_f32 / (hd as f32).sqrt();

        for li in 0..s.depth {
            let t0 = std::time::Instant::now();
            let blk = &self.blocks[li];

            // norm1 (LayerNorm + bias) → normed.
            runtime.vision_layernorm_bias_device(
                &state, &blk.norm1_w, &blk.norm1_b, n, h, eps, &mut normed)?;
            runtime.f32_to_bf16_device(&normed, n * h, &mut normed_bf16)?;

            // fused QKV: [n, 3h] = normed @ qkv.T + qkv_b.
            runtime.matmul_bf16_cublaslt_with_input_bf16_device(
                &blk.qkv, &normed_bf16, n, &mut qkv_bf16, &mut qkv_f32)?;
            runtime.vision_add_bias_rows_device(&mut qkv_f32, &blk.qkv_b, n, 3 * h)?;

            // Split [n, 3h] as HF reshape(n, 3, nh, hd): for token t, the row is
            // [q(nh*hd), k(nh*hd), v(nh*hd)] contiguous. Extract each via a
            // strided 2D copy (src_stride=3h, copy_len=h, per-block offset).
            runtime.strided_copy_2d(&qkv_f32, &mut q_buf, n, h, 3 * h, h, 0)?;
            runtime.strided_copy_2d(&qkv_f32, &mut k_buf, n, h, 3 * h, h, h)?;
            runtime.strided_copy_2d(&qkv_f32, &mut v_buf, n, h, 3 * h, h, 2 * h)?;

            // Vision RoPE on Q, K (standard rotate-half, (row,col) pos ids).
            runtime.vision_rope_qwen_device(&mut q_buf, &pos_ids, n, nh, hd, s.rope_theta)?;
            runtime.vision_rope_qwen_device(&mut k_buf, &pos_ids, n, nh, hd, s.rope_theta)?;
            if li == 0 {
                dump("q_layer0_after_rope", &q_buf, n * h)?;
            }

            // Full bidirectional attention (single image = one cu_seqlens block),
            // Q-tiled cuBLASLt BF16 GEMMs around a row-softmax.
            runtime.f32_to_bf16_device(&q_buf, n * h, &mut q_bf16)?;
            runtime.f32_to_bf16_device(&k_buf, n * h, &mut k_bf16)?;
            runtime.f32_to_bf16_device(&v_buf, n * h, &mut v_bf16)?;
            let mut q_start = 0usize;
            while q_start < n {
                let q_end = (q_start + bq).min(n);
                let q_len = q_end - q_start;
                runtime.bf16_strided_batched_gemm_device(
                    &k_bf16, 0, n * nh * hd,
                    &q_bf16, q_start * nh * hd, q_len * nh * hd,
                    &mut scores_bf16, 0, nh * q_len * n,
                    true, false,
                    n, q_len, hd,
                    nh * hd, nh * hd, n,
                    hd, hd, q_len * n,
                    nh, scale, 0.0,
                )?;
                runtime.vision_row_softmax_bf16_device(
                    &mut scores_bf16, nh * q_len, n, 1.0)?;
                runtime.bf16_strided_batched_gemm_device(
                    &v_bf16, 0, n * nh * hd,
                    &scores_bf16, 0, nh * q_len * n,
                    &mut attn_bf16, q_start * nh * hd, q_len * nh * hd,
                    false, false,
                    hd, q_len, n,
                    nh * hd, n, nh * hd,
                    hd, q_len * n, hd,
                    nh, 1.0, 0.0,
                )?;
                q_start = q_end;
            }

            // proj: attn @ proj.T + proj_b → o.
            runtime.matmul_bf16_cublaslt_with_input_bf16_device(
                &blk.proj, &attn_bf16, n, &mut o_bf16, &mut o_f32)?;
            runtime.vision_add_bias_rows_device(&mut o_f32, &blk.proj_b, n, h)?;
            // residual: state += o.
            runtime.add_inplace_device_len(&mut state, &o_f32, n * h)?;
            if li == 0 {
                dump("after_block0_attn", &state, n * h)?;
            }

            // MLP: norm2 → fc1+bias → gelu_tanh → fc2+bias → residual.
            runtime.vision_layernorm_bias_device(
                &state, &blk.norm2_w, &blk.norm2_b, n, h, eps, &mut normed)?;
            runtime.f32_to_bf16_device(&normed, n * h, &mut normed_bf16)?;
            runtime.matmul_bf16_cublaslt_with_input_bf16_device(
                &blk.fc1, &normed_bf16, n, &mut fc1_bf16, &mut fc1_f32)?;
            runtime.vision_add_bias_rows_device(&mut fc1_f32, &blk.fc1_b, n, inter)?;
            runtime.vision_gelu_tanh_device(&mut fc1_f32, n * inter)?;
            runtime.f32_to_bf16_device(&fc1_f32, n * inter, &mut fc1_act_bf16)?;
            runtime.matmul_bf16_cublaslt_with_input_bf16_device(
                &blk.fc2, &fc1_act_bf16, n, &mut fc2_bf16, &mut fc2_f32)?;
            runtime.vision_add_bias_rows_device(&mut fc2_f32, &blk.fc2_b, n, h)?;
            runtime.add_inplace_device_len(&mut state, &fc2_f32, n * h)?;

            if log_progress {
                eprintln!("  qwen-vis block {:>2}/{}: {:.3}s",
                    li + 1, s.depth, t0.elapsed().as_secs_f64());
            }
            if li == 0 {
                dump("after_block0", &state, n * h)?;
            }
            if li == 5 { dump("after_block5", &state, n * h)?; }
            if li == 13 { dump("after_block13", &state, n * h)?; }
        }
        dump("after_all_blocks", &state, n * h)?;

        // ── Merger: LayerNorm+bias (on hidden) → reshape [n/m², h*m²] →
        // fc1+bias → GELU → fc2+bias.
        let merge_dim = h * m * m; // 4608
        let n_merged = n / (m * m);
        // norm over the 1152-dim, pre-shuffle (use_postshuffle_norm=False).
        let mut merged_norm = runtime.alloc_f32(n * h)?;
        runtime.vision_layernorm_bias_device(
            &state, &self.merger_norm_w, &self.merger_norm_b, n, h, eps, &mut merged_norm)?;
        // reshape [n, h] → [n_merged, merge_dim] is a contiguous view (rows are
        // already merge-block ordered): row j = concat of rows 4j..4j+4. So
        // treat merged_norm as [n_merged, merge_dim] directly for the GEMM.
        let mut mn_bf16 = runtime.alloc_u16(n * h)?;
        runtime.f32_to_bf16_device(&merged_norm, n * h, &mut mn_bf16)?;
        // fc1: [n_merged, merge_dim] @ fc1.T + fc1_b → [n_merged, merge_dim].
        let mut fc1m_bf16 = runtime.alloc_u16(n_merged * merge_dim)?;
        let mut fc1m_f32 = runtime.alloc_f32(n_merged * merge_dim)?;
        runtime.matmul_bf16_cublaslt_with_input_bf16_device(
            &self.merger_fc1, &mn_bf16, n_merged, &mut fc1m_bf16, &mut fc1m_f32)?;
        runtime.vision_add_bias_rows_device(&mut fc1m_f32, &self.merger_fc1_b, n_merged, merge_dim)?;
        // GELU (nn.GELU() default = exact erf-based, not tanh!). Use erf-gelu.
        runtime.gelu_erf_inplace_device(&mut fc1m_f32, n_merged * merge_dim)?;
        let mut fc1m_act_bf16 = runtime.alloc_u16(n_merged * merge_dim)?;
        runtime.f32_to_bf16_device(&fc1m_f32, n_merged * merge_dim, &mut fc1m_act_bf16)?;
        // fc2: [n_merged, merge_dim] @ fc2.T + fc2_b → [n_merged, out_hidden].
        let out_hidden = self.merger_fc2.rows;
        let mut out_bf16 = runtime.alloc_u16(n_merged * out_hidden)?;
        let mut out_f32 = runtime.alloc_f32(n_merged * out_hidden)?;
        runtime.matmul_bf16_cublaslt_with_input_bf16_device(
            &self.merger_fc2, &fc1m_act_bf16, n_merged, &mut out_bf16, &mut out_f32)?;
        runtime.vision_add_bias_rows_device(&mut out_f32, &self.merger_fc2_b, n_merged, out_hidden)?;
        dump("after_merger", &out_f32, n_merged * out_hidden)?;

        runtime.download_f32(&out_f32)
    }
}

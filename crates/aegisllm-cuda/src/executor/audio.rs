//! Gemma-4 E4B audio encoder (USM/Conformer tower).
//!
//! Loads `model.audio_tower.*` + `model.embed_audio.*` and runs the forward
//! pass: log-mel features → subsample conv stack → 12 Conformer layers →
//! output_proj → embed_audio → audio soft-token embeddings in the LLM's text
//! hidden space (2560). The result is spliced into the prompt embedding stream
//! at `audio_token_id` positions, exactly mirroring the vision-tower image path.
//!
//! ARCHITECTURE (Gemma-3n / Gemma-4 audio, from HF `modeling_gemma3n.py`):
//!
//!   Subsampler (`subsample_conv_projection`):
//!     in: [n_frames, 128] log-mel → unsqueeze channel → [1, n_frames, 128]
//!     layer0: Conv2d(1→128, k=3, s=2, pad=1) → CumulativeGroupNorm → ReLU
//!     layer1: Conv2d(128→32, k=3, s=2, pad=1) → CumulativeGroupNorm → ReLU
//!     → permute [C,T,F]→[T,F,C], flatten F*C = 32*32 = 1024
//!     → input_proj_linear [1024 → 1024]
//!
//!   Per Conformer layer (`layers.{L}`), block ordering:
//!     1. feed_forward1 (Macaron): pre_ln → linear1[4096,1024] → SiLU
//!        → linear2[1024,4096] → post_ln → residual + out*residual_weight(0.5)
//!     2. self_attn: norm_pre_attn → q/k/v_proj[1024,1024], per_dim_scale[128]
//!        on Q, relative_k_proj[1024,1024] rel-pos bias, chunked-local mask
//!        (chunk=12, left=13, right=0), tanh logit-softcap(50), post[1024,1024]
//!        → norm_post_attn → residual
//!     3. lconv1d: pre_ln → linear_start[2048,1024] → GLU → depthwise causal
//!        conv1d(k=5) → conv_norm → SiLU → linear_end[1024,1024] → residual
//!     4. feed_forward2 (Macaron): same as feed_forward1
//!     5. norm_out (RMSNorm)
//!   Gradient clipping (clamp ±1e10) is applied between sub-blocks.
//!
//!   Tail: output_proj [1024 → 1536] (+ bias) → embed_audio.embedding_projection
//!         [1536 → 2560].
//!
//! IMPLEMENTATION NOTE: This is a CORRECTNESS-FIRST CPU-reference forward that
//! mirrors `vision.rs::forward()`. The heavy matmuls run on the GPU via the
//! existing cuBLASLt BF16 wrapper; the audio-specific elementwise ops use the
//! new `audio_*` CUDA kernels; the rel-pos chunked-local attention runs on the
//! CPU (download Q·K^T → mask → softcap → softmax → upload), exactly like the
//! vision tower's first-pass CPU attention. A fused GPU Conformer attention
//! kernel is a future perf step.
//!
//! Every numeric detail is marked `// TODO(gpu-verify)` where it was derived
//! from the HF reference but not validated against an activation dump on GPU.

use aegisllm_base::artifact::ModelArtifact;
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::tensor::TensorInfo;
use aegisllm_base::tensor::storage::TensorStorageLoader;

use super::loader::cuda_residency_for_store;
use crate::cuda::loader::CudaWeightLoader;
use crate::cuda::{DeviceBf16Matrix, DeviceBuffer};
use aegisllm_base::planning::placement::StoragePlacement;

/// Shape parameters for the audio encoder, derived from `audio_config`.
#[derive(Debug, Clone)]
pub struct AudioEncoderShape {
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    /// hidden_size / num_attention_heads (Gemma-4: 1024 / 8 = 128).
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub output_proj_dims: usize,
    /// `[128, 32]` for Gemma-4: subsample conv output channels per layer.
    pub subsampling_conv_channels: Vec<usize>,
    pub conv_kernel_size: usize,
    pub attention_chunk_size: usize,
    pub attention_context_left: usize,
    pub attention_context_right: usize,
    pub attention_logit_cap: f32,
    pub attention_invalid_logits_value: f32,
    pub residual_weight: f32,
    pub gradient_clipping: f32,
    pub use_clipped_linears: bool,
    /// Number of log-mel bins per frame (Gemma-4: 128). Fixed by the front-end.
    pub n_mel_bins: usize,
}

impl AudioEncoderShape {
    pub fn from_artifact(artifact: &ModelArtifact) -> Result<Self> {
        let ac = artifact.config.audio_config.as_ref().ok_or_else(|| {
            AegisError::InvalidPlan(
                "audio tower requested but config.json has no `audio_config` section".into(),
            )
        })?;
        if ac.num_attention_heads == 0 {
            return Err(AegisError::InvalidPlan(
                "audio_config.num_attention_heads must be > 0".into(),
            ));
        }
        let head_dim = ac.hidden_size / ac.num_attention_heads;
        Ok(Self {
            hidden_size: ac.hidden_size,
            num_layers: ac.num_hidden_layers,
            num_attention_heads: ac.num_attention_heads,
            head_dim,
            rms_norm_eps: ac.rms_norm_eps as f32,
            output_proj_dims: ac.output_proj_dims,
            subsampling_conv_channels: ac.subsampling_conv_channels.clone(),
            conv_kernel_size: ac.conv_kernel_size,
            attention_chunk_size: ac.attention_chunk_size,
            attention_context_left: ac.attention_context_left,
            attention_context_right: ac.attention_context_right,
            attention_logit_cap: ac.attention_logit_cap,
            attention_invalid_logits_value: ac.attention_invalid_logits_value,
            residual_weight: ac.residual_weight,
            // gradient_clipping is not yet parsed in HfAudioConfig; Gemma-4
            // uses 1e10 (effectively "no clip" until activations explode).
            // TODO(gpu-verify): wire gradient_clipping through HfAudioConfig if
            // the clamp ever matters numerically (it shouldn't at 1e10).
            gradient_clipping: 1.0e10,
            use_clipped_linears: ac.use_clipped_linears,
            // 128 log-mel bins — fixed by the Gemma-4 audio front-end
            // (frame=320 hop=160 fft=512 @ 16 kHz → 100 frames/s, 128 mel).
            n_mel_bins: 128,
        })
    }
}

/// A `ClippableLinear`: a BF16 weight matrix plus optional input/output clamp
/// scalars (when `use_clipped_linears`). The clamp bounds are scalar BF16
/// tensors `input_min/max`, `output_min/max` in the checkpoint.
pub struct AudioClippableLinear {
    pub weight: DeviceBf16Matrix,
    /// (input_min, input_max, output_min, output_max). All `None` when the
    /// model does not use clipped linears.
    pub clamp: Option<AudioClipBounds>,
}

#[derive(Debug, Clone, Copy)]
pub struct AudioClipBounds {
    pub input_min: f32,
    pub input_max: f32,
    pub output_min: f32,
    pub output_max: f32,
}

/// A Macaron feed-forward sub-block (feed_forward1 / feed_forward2).
pub struct AudioFeedForward {
    pub pre_layer_norm: DeviceBuffer<f32>,
    pub ffw_layer_1: AudioClippableLinear, // [4*hidden, hidden]
    pub ffw_layer_2: AudioClippableLinear, // [hidden, 4*hidden]
    pub post_layer_norm: DeviceBuffer<f32>,
}

/// The LightConv1d sub-block.
pub struct AudioLightConv1d {
    pub pre_layer_norm: DeviceBuffer<f32>,
    pub linear_start: AudioClippableLinear, // [2*hidden, hidden] (GLU input)
    pub conv_norm: DeviceBuffer<f32>,
    pub depthwise_conv1d: DeviceBuffer<f32>, // [hidden, kernel] (flattened)
    pub linear_end: AudioClippableLinear,    // [hidden, hidden]
}

/// The self-attention sub-block.
pub struct AudioSelfAttn {
    pub q_proj: AudioClippableLinear,
    pub k_proj: AudioClippableLinear,
    pub v_proj: AudioClippableLinear,
    pub relative_k_proj: DeviceBf16Matrix, // [hidden, hidden], no clamp/bias
    pub per_dim_scale: DeviceBuffer<f32>,  // [head_dim]
    pub post: AudioClippableLinear,        // [hidden, hidden]
}

/// One Conformer layer.
pub struct AudioConformerLayer {
    pub feed_forward1: AudioFeedForward,
    pub norm_pre_attn: DeviceBuffer<f32>,
    pub self_attn: AudioSelfAttn,
    pub norm_post_attn: DeviceBuffer<f32>,
    pub lconv1d: AudioLightConv1d,
    pub feed_forward2: AudioFeedForward,
    pub norm_out: DeviceBuffer<f32>,
}

/// 2D conv subsample block (Conv2d k=3 s=2 pad=1 + norm + ReLU).
pub struct AudioSubsampleConvBlock {
    /// Conv2d weight [out_ch, in_ch, 3, 3] flattened — kept on host f32 for the
    /// reference conv (the conv runs on CPU in this first pass; see forward()).
    pub conv_weight: Vec<f32>,
    pub out_channels: usize,
    pub in_channels: usize,
    pub norm_weight: Vec<f32>, // [out_ch] CumulativeGroupNorm scale
}

/// The full audio tower (subsampler + 12 Conformer layers + tail).
pub struct AudioTower {
    pub shape: AudioEncoderShape,
    pub subsample0: AudioSubsampleConvBlock,
    pub subsample1: AudioSubsampleConvBlock,
    pub input_proj: DeviceBf16Matrix, // [hidden, f_out*c_out=1024]
    pub layers: Vec<AudioConformerLayer>,
    pub output_proj: DeviceBf16Matrix, // [output_proj_dims, hidden]
    pub output_proj_bias: DeviceBuffer<f32>, // [output_proj_dims]
    pub embed_audio: DeviceBf16Matrix, // [text_hidden, output_proj_dims]
}

impl AudioTower {
    /// Load the audio tower + projector from the artifact. All matmul weights
    /// VRAM-resident BF16; norm/scale vectors uploaded f32; conv weights kept
    /// host-side f32 (the subsample conv runs on the CPU in this first pass).
    pub fn from_artifact(
        artifact: &ModelArtifact,
        shape: AudioEncoderShape,
        cuda_weights: &CudaWeightLoader<'_>,
        device_index: usize,
        loader: &mut TensorStorageLoader,
    ) -> Result<Self> {
        let store = StoragePlacement::Vram { device: device_index };
        let residency = cuda_residency_for_store(store, device_index)?;

        let get = |name: &str| -> Result<&TensorInfo> {
            artifact.tensors.tensors.get(name).ok_or_else(|| {
                AegisError::InvalidPlan(format!("audio tower: tensor `{name}` missing"))
            })
        };

        // Helper to load a ClippableLinear (weight + optional clamp scalars).
        let load_clip = |prefix: &str,
                         loader: &mut TensorStorageLoader|
         -> Result<AudioClippableLinear> {
            let weight = cuda_weights.load_bf16_matrix_with_store(
                get(&format!("{prefix}.linear.weight"))?,
                store,
                residency.clone(),
                loader,
            )?;
            let clamp = if shape.use_clipped_linears {
                // input/output_min/max are scalar BF16 tensors (shape []).
                // Load each as a 1-element vector on host. We read them via the
                // dense-vector loader which downloads to host f32 first.
                // TODO(gpu-verify): confirm the scalar clamp is applied as
                // input clamp BEFORE the GEMM and output clamp AFTER — and
                // whether E4B's bounds are wide enough to be inert.
                let scalar = |name: &str,
                              loader: &mut TensorStorageLoader|
                 -> Result<f32> {
                    read_scalar_bf16(artifact, name, store, loader)
                };
                Some(AudioClipBounds {
                    input_min: scalar(&format!("{prefix}.input_min"), loader)?,
                    input_max: scalar(&format!("{prefix}.input_max"), loader)?,
                    output_min: scalar(&format!("{prefix}.output_min"), loader)?,
                    output_max: scalar(&format!("{prefix}.output_max"), loader)?,
                })
            } else {
                None
            };
            Ok(AudioClippableLinear { weight, clamp })
        };

        let load_vec = |name: &str, loader: &mut TensorStorageLoader| {
            cuda_weights.load_dense_vector_with_store(
                match artifact.tensors.tensors.get(name) {
                    Some(t) => t,
                    None => {
                        return Err(AegisError::InvalidPlan(format!(
                            "audio tower: tensor `{name}` missing"
                        )));
                    }
                },
                store,
                loader,
            )
        };

        // ── Subsampler conv blocks (host f32 conv weight + norm scale). ──
        let load_conv_block = |idx: usize,
                               in_ch: usize,
                               loader: &mut TensorStorageLoader|
         -> Result<AudioSubsampleConvBlock> {
            let cw = get(&format!(
                "model.audio_tower.subsample_conv_projection.layer{idx}.conv.weight"
            ))?;
            let out_ch = cw.shape[0];
            let conv_weight = download_bf16_tensor(artifact, cw, store, loader)?;
            let norm_weight = download_bf16_tensor(
                artifact,
                get(&format!(
                    "model.audio_tower.subsample_conv_projection.layer{idx}.norm.weight"
                ))?,
                store,
                loader,
            )?;
            Ok(AudioSubsampleConvBlock {
                conv_weight,
                out_channels: out_ch,
                in_channels: in_ch,
                norm_weight,
            })
        };
        let subsample0 = load_conv_block(0, 1, loader)?;
        let subsample1 = load_conv_block(1, subsample0.out_channels, loader)?;

        let input_proj = cuda_weights.load_bf16_matrix_with_store(
            get("model.audio_tower.subsample_conv_projection.input_proj_linear.weight")?,
            store,
            residency.clone(),
            loader,
        )?;

        // ── Conformer layers. ──
        let mut layers = Vec::with_capacity(shape.num_layers);
        for li in 0..shape.num_layers {
            let p = |s: &str| format!("model.audio_tower.layers.{li}.{s}");

            let load_ffw = |which: &str,
                            loader: &mut TensorStorageLoader|
             -> Result<AudioFeedForward> {
                Ok(AudioFeedForward {
                    pre_layer_norm: load_vec(&p(&format!("{which}.pre_layer_norm.weight")), loader)?,
                    ffw_layer_1: load_clip(&p(&format!("{which}.ffw_layer_1")), loader)?,
                    ffw_layer_2: load_clip(&p(&format!("{which}.ffw_layer_2")), loader)?,
                    post_layer_norm: load_vec(&p(&format!("{which}.post_layer_norm.weight")), loader)?,
                })
            };

            let feed_forward1 = load_ffw("feed_forward1", loader)?;
            let norm_pre_attn = load_vec(&p("norm_pre_attn.weight"), loader)?;
            let self_attn = AudioSelfAttn {
                q_proj: load_clip(&p("self_attn.q_proj"), loader)?,
                k_proj: load_clip(&p("self_attn.k_proj"), loader)?,
                v_proj: load_clip(&p("self_attn.v_proj"), loader)?,
                relative_k_proj: cuda_weights.load_bf16_matrix_with_store(
                    get(&p("self_attn.relative_k_proj.weight"))?,
                    store,
                    residency.clone(),
                    loader,
                )?,
                per_dim_scale: load_vec(&p("self_attn.per_dim_scale"), loader)?,
                post: load_clip(&p("self_attn.post"), loader)?,
            };
            let norm_post_attn = load_vec(&p("norm_post_attn.weight"), loader)?;
            let lconv1d = AudioLightConv1d {
                pre_layer_norm: load_vec(&p("lconv1d.pre_layer_norm.weight"), loader)?,
                linear_start: load_clip(&p("lconv1d.linear_start"), loader)?,
                conv_norm: load_vec(&p("lconv1d.conv_norm.weight"), loader)?,
                // depthwise_conv1d.weight is [hidden, 1, kernel] (3-D); the
                // dense-vector loader rejects >1-D, so download to host f32 and
                // upload as a flattened [hidden, kernel] buffer.
                depthwise_conv1d: {
                    let host = download_bf16_tensor(
                        artifact,
                        get(&p("lconv1d.depthwise_conv1d.weight"))?,
                        store,
                        loader,
                    )?;
                    cuda_weights.runtime().upload_f32(&host)?
                },
                linear_end: load_clip(&p("lconv1d.linear_end"), loader)?,
            };
            let feed_forward2 = load_ffw("feed_forward2", loader)?;
            let norm_out = load_vec(&p("norm_out.weight"), loader)?;

            layers.push(AudioConformerLayer {
                feed_forward1,
                norm_pre_attn,
                self_attn,
                norm_post_attn,
                lconv1d,
                feed_forward2,
                norm_out,
            });
        }

        // ── Tail. ──
        let output_proj = cuda_weights.load_bf16_matrix_with_store(
            get("model.audio_tower.output_proj.weight")?,
            store,
            residency.clone(),
            loader,
        )?;
        let output_proj_bias = load_vec("model.audio_tower.output_proj.bias", loader)?;
        let embed_audio = cuda_weights.load_bf16_matrix_with_store(
            get("model.embed_audio.embedding_projection.weight")?,
            store,
            residency.clone(),
            loader,
        )?;

        Ok(Self {
            shape,
            subsample0,
            subsample1,
            input_proj,
            layers,
            output_proj,
            output_proj_bias,
            embed_audio,
        })
    }

    /// Forward pass: precomputed log-mel features `[n_frames, n_mel_bins]`
    /// row-major f32 → audio soft-token embeddings `[n_audio_tokens, text_hidden]`
    /// row-major f32, ready to splice at `audio_token_id` positions.
    ///
    /// CORRECTNESS-FIRST: matmuls on GPU (cuBLASLt BF16), audio-specific
    /// elementwise on GPU (audio_* kernels), attention on CPU.
    pub fn forward(
        &self,
        runtime: &crate::cuda::CudaRuntime,
        mel: &[f32],
        n_frames: usize,
    ) -> Result<Vec<f32>> {
        let s = &self.shape;
        if mel.len() != n_frames * s.n_mel_bins {
            return Err(AegisError::InvalidPlan(format!(
                "audio forward: mel len={} != n_frames({}) * n_mel_bins({}) = {}",
                mel.len(),
                n_frames,
                s.n_mel_bins,
                n_frames * s.n_mel_bins
            )));
        }
        let h = s.hidden_size;
        let eps = s.rms_norm_eps;

        // ── Subsample conv stack → [t_out, hidden]. ──
        // TODO(gpu-verify): the subsample conv (Conv2d k3 s2 pad1 + cumulative
        // group norm + ReLU) is implemented on the CPU here. Confirm the
        // padding convention (HF uses asymmetric manual padding; this uses
        // symmetric pad=1) and the CumulativeGroupNorm cumulative-over-time
        // statistics against an HF dump before trusting downstream values.
        let (mut state, t_out) = self.subsample_forward(mel, n_frames)?;
        // state: [t_out, hidden] row-major f32.

        let log = std::env::var("AEGIS_AUDIO_PROGRESS").is_ok();

        // q_scale = head_dim^-0.5 / softplus(0).  softplus(0) = ln(2).
        let q_scale = (s.head_dim as f32).powf(-0.5) / (2.0f32.ln());

        for li in 0..s.num_layers {
            let t_layer = std::time::Instant::now();
            let layer = &self.layers[li];

            // 1. feed_forward1 (Macaron).
            self.feed_forward(runtime, &mut state, t_out, &layer.feed_forward1)?;
            self.clamp(runtime, &mut state, t_out * h)?;

            // 2. self-attention.
            self.self_attention(runtime, &mut state, t_out, layer, q_scale)?;
            self.clamp(runtime, &mut state, t_out * h)?;

            // 3. lconv1d.
            self.light_conv1d(runtime, &mut state, t_out, &layer.lconv1d)?;
            self.clamp(runtime, &mut state, t_out * h)?;

            // 4. feed_forward2 (Macaron).
            self.feed_forward(runtime, &mut state, t_out, &layer.feed_forward2)?;
            self.clamp(runtime, &mut state, t_out * h)?;

            // 5. norm_out (RMSNorm).
            self.rms_norm_inplace(runtime, &mut state, t_out, &layer.norm_out, eps)?;

            if log {
                eprintln!(
                    "  audio layer {:>2}/{}: {:.3}s",
                    li + 1,
                    s.num_layers,
                    t_layer.elapsed().as_secs_f64()
                );
            }
        }

        // ── Tail: output_proj (+ bias) → embed_audio. ──
        // output_proj: [t_out, hidden] @ output_proj.T → [t_out, output_proj_dims].
        let proj_dim = s.output_proj_dims;
        let mut proj = self.matmul_host(runtime, &state, t_out, &self.output_proj)?;
        // add bias.
        {
            let bias = runtime.download_f32(&self.output_proj_bias)?;
            for t in 0..t_out {
                for c in 0..proj_dim {
                    proj[t * proj_dim + c] += bias[c];
                }
            }
        }
        // embed_audio: [t_out, output_proj_dims] @ embed_audio.T → [t_out, text_hidden].
        let out = self.matmul_host(runtime, &proj, t_out, &self.embed_audio)?;
        Ok(out)
    }

    /// CPU subsample conv stack. Returns (flattened [t_out, hidden] f32, t_out).
    fn subsample_forward(&self, mel: &[f32], n_frames: usize) -> Result<(Vec<f32>, usize)> {
        let s = &self.shape;
        // layer0: input [1, T0=n_frames, F0=n_mel_bins] → [C1, T1, F1].
        let (x0, c1, t1, f1) =
            conv2d_norm_relu(mel, 1, n_frames, s.n_mel_bins, &self.subsample0)?;
        // layer1: input [C1, T1, F1] → [C2, T2, F2].
        let (x1, c2, t2, f2) = conv2d_norm_relu(&x0, c1, t1, f1, &self.subsample1)?;
        // permute [C2, T2, F2] → [T2, F2, C2], flatten → [T2, F2*C2].
        let flat_dim = f2 * c2;
        let mut flat = vec![0f32; t2 * flat_dim];
        for t in 0..t2 {
            for f in 0..f2 {
                for c in 0..c2 {
                    let src = (c * t2 + t) * f2 + f; // [C2, T2, F2]
                    let dst = t * flat_dim + (f * c2 + c); // [T2, F2, C2]
                    flat[dst] = x1[src];
                }
            }
        }
        // TODO(gpu-verify): confirm the [T,F,C] flatten order matches HF
        // `permute(0,2,3,1).view(b, t_out, f_out*c_out)` exactly (F-major then
        // C-minor as done here).
        if flat_dim != self.input_proj.cols {
            return Err(AegisError::InvalidPlan(format!(
                "audio subsample: flat_dim={flat_dim} != input_proj.cols={}",
                self.input_proj.cols
            )));
        }
        Ok((flat, t2))
    }

    /// matmul host helper: input host f32 [batch, in] → host f32 [batch, rows].
    fn matmul_host(
        &self,
        runtime: &crate::cuda::CudaRuntime,
        input: &[f32],
        batch: usize,
        weight: &DeviceBf16Matrix,
    ) -> Result<Vec<f32>> {
        let in_dim = weight.cols;
        let out_dim = weight.rows;
        let in_f32 = runtime.upload_f32(input)?;
        let mut in_bf16 = runtime.alloc_u16(batch * in_dim)?;
        runtime.f32_to_bf16_device(&in_f32, batch * in_dim, &mut in_bf16)?;
        let mut out_bf16 = runtime.alloc_u16(batch * out_dim)?;
        let mut out_f32 = runtime.alloc_f32(batch * out_dim)?;
        runtime.matmul_bf16_cublaslt_with_input_bf16_device(
            weight,
            &in_bf16,
            batch,
            &mut out_bf16,
            &mut out_f32,
        )?;
        runtime.download_f32(&out_f32)
    }

    /// Apply a clipped linear: optional input clamp → GEMM → optional output
    /// clamp. Input/output are host f32 [batch, *].
    fn clip_linear_host(
        &self,
        runtime: &crate::cuda::CudaRuntime,
        input: &[f32],
        batch: usize,
        lin: &AudioClippableLinear,
    ) -> Result<Vec<f32>> {
        let mut x = input.to_vec();
        if let Some(b) = lin.clamp {
            for v in x.iter_mut() {
                *v = v.clamp(b.input_min, b.input_max);
            }
        }
        let mut out = self.matmul_host(runtime, &x, batch, &lin.weight)?;
        if let Some(b) = lin.clamp {
            for v in out.iter_mut() {
                *v = v.clamp(b.output_min, b.output_max);
            }
        }
        Ok(out)
    }

    /// In-place RMSNorm over [batch, hidden] with a learned scale vector.
    fn rms_norm_inplace(
        &self,
        runtime: &crate::cuda::CudaRuntime,
        state: &mut [f32],
        batch: usize,
        weight: &DeviceBuffer<f32>,
        eps: f32,
    ) -> Result<()> {
        let h = self.shape.hidden_size;
        let inp = runtime.upload_f32(state)?;
        let mut out = runtime.alloc_f32(batch * h)?;
        runtime.rms_norm_batched_device(&inp, weight, batch, eps, &mut out)?;
        let host = runtime.download_f32(&out)?;
        state.copy_from_slice(&host[..batch * h]);
        Ok(())
    }

    /// RMSNorm to a fresh buffer (not in place).
    fn rms_norm_to(
        &self,
        runtime: &crate::cuda::CudaRuntime,
        state: &[f32],
        batch: usize,
        weight: &DeviceBuffer<f32>,
        eps: f32,
    ) -> Result<Vec<f32>> {
        let h = self.shape.hidden_size;
        let inp = runtime.upload_f32(state)?;
        let mut out = runtime.alloc_f32(batch * h)?;
        runtime.rms_norm_batched_device(&inp, weight, batch, eps, &mut out)?;
        runtime.download_f32(&out)
    }

    /// Gradient-clip clamp on host (cheap; the threshold is 1e10 so usually inert).
    fn clamp(
        &self,
        _runtime: &crate::cuda::CudaRuntime,
        state: &mut [f32],
        _n: usize,
    ) -> Result<()> {
        let c = self.shape.gradient_clipping;
        for v in state.iter_mut() {
            *v = v.clamp(-c, c);
        }
        Ok(())
    }

    /// Macaron feed-forward sub-block: state += residual_weight * postLN(FFN(preLN(state))).
    fn feed_forward(
        &self,
        runtime: &crate::cuda::CudaRuntime,
        state: &mut [f32],
        batch: usize,
        ff: &AudioFeedForward,
    ) -> Result<()> {
        let h = self.shape.hidden_size;
        let eps = self.shape.rms_norm_eps;
        // pre_layer_norm.
        let normed = self.rms_norm_to(runtime, state, batch, &ff.pre_layer_norm, eps)?;
        // linear1 → SiLU → linear2.
        let h1 = self.clip_linear_host(runtime, &normed, batch, &ff.ffw_layer_1)?;
        // SiLU on GPU.
        let mut h1_dev = runtime.upload_f32(&h1)?;
        runtime.audio_silu_inplace_device(&mut h1_dev, h1.len())?;
        let h1_act = runtime.download_f32(&h1_dev)?;
        let ff_out = self.clip_linear_host(runtime, &h1_act, batch, &ff.ffw_layer_2)?;
        // post_layer_norm.
        let post = self.rms_norm_to(runtime, &ff_out, batch, &ff.post_layer_norm, eps)?;
        // residual + out * residual_weight.
        let rw = self.shape.residual_weight;
        for t in 0..batch {
            for c in 0..h {
                state[t * h + c] += rw * post[t * h + c];
            }
        }
        Ok(())
    }

    /// LightConv1d sub-block: state += linear_end(SiLU(conv_norm(dwconv(GLU(linear_start(preLN(state))))))).
    fn light_conv1d(
        &self,
        runtime: &crate::cuda::CudaRuntime,
        state: &mut [f32],
        batch: usize,
        lc: &AudioLightConv1d,
    ) -> Result<()> {
        let h = self.shape.hidden_size;
        let eps = self.shape.rms_norm_eps;
        let k = self.shape.conv_kernel_size;

        // pre_layer_norm.
        let normed = self.rms_norm_to(runtime, state, batch, &lc.pre_layer_norm, eps)?;
        // linear_start → [batch, 2*hidden]; GLU → [batch, hidden].
        let ls = self.clip_linear_host(runtime, &normed, batch, &lc.linear_start)?;
        let ls_dev = runtime.upload_f32(&ls)?;
        let mut glu_dev = runtime.alloc_f32(batch * h)?;
        runtime.audio_glu_halfsplit_device(&ls_dev, &mut glu_dev, batch, h)?;
        // depthwise causal conv1d over time.
        let mut conv_dev = runtime.alloc_f32(batch * h)?;
        runtime.audio_depthwise_causal_conv1d_device(
            &glu_dev,
            &lc.depthwise_conv1d,
            &mut conv_dev,
            batch,
            h,
            k,
        )?;
        let conv = runtime.download_f32(&conv_dev)?;
        // conv_norm (RMSNorm) → SiLU.
        let mut normed2 = self.rms_norm_to(runtime, &conv, batch, &lc.conv_norm, eps)?;
        let mut n2_dev = runtime.upload_f32(&normed2)?;
        runtime.audio_silu_inplace_device(&mut n2_dev, batch * h)?;
        normed2 = runtime.download_f32(&n2_dev)?;
        // linear_end.
        let end = self.clip_linear_host(runtime, &normed2, batch, &lc.linear_end)?;
        // residual add.
        for i in 0..batch * h {
            state[i] += end[i];
        }
        Ok(())
    }

    /// Self-attention sub-block (rel-pos chunked-local). Updates state in place
    /// with the residual add. Attention scores/softmax run on the CPU.
    fn self_attention(
        &self,
        runtime: &crate::cuda::CudaRuntime,
        state: &mut [f32],
        batch: usize,
        layer: &AudioConformerLayer,
        q_scale: f32,
    ) -> Result<()> {
        let s = &self.shape;
        let h = s.hidden_size;
        let nh = s.num_attention_heads;
        let hd = s.head_dim;
        let eps = s.rms_norm_eps;
        let attn = &layer.self_attn;

        // norm_pre_attn.
        let normed = self.rms_norm_to(runtime, state, batch, &layer.norm_pre_attn, eps)?;
        // Q/K/V projections (clipped linears, [hidden, hidden]).
        let q = self.clip_linear_host(runtime, &normed, batch, &attn.q_proj)?;
        let k = self.clip_linear_host(runtime, &normed, batch, &attn.k_proj)?;
        let v = self.clip_linear_host(runtime, &normed, batch, &attn.v_proj)?;

        // Apply per_dim_scale to Q on GPU: q = q * q_scale * softplus(per_dim_scale).
        let mut q_dev = runtime.upload_f32(&q)?;
        runtime.audio_per_dim_scale_device(
            &mut q_dev,
            &attn.per_dim_scale,
            batch,
            nh,
            hd,
            q_scale,
        )?;
        let q = runtime.download_f32(&q_dev)?;

        // ── Relative position bias term (term_bd). ──
        // HF builds a sinusoidal timing signal over relative positions
        // [max_backward .. -max_forward], projects via relative_k_proj-derived
        // pos_proj, and applies a relative-shift. For this first pass we compute
        // a simplified content-only attention (term_ac) and leave term_bd as a
        // TODO. This is the single biggest numeric gap in the audio path.
        //
        // TODO(gpu-verify): implement the full relative-position bias.
        //   max_backward = attention_context_left - 1 (= 12)
        //   max_forward  = attention_context_right     (= 0)
        //   pos_indices  = arange(max_backward, -max_forward-1, -1)
        //   sin_emb      = timing_signal_1d(pos_indices)            [F_span, hidden]
        //   proj         = sin_emb @ relative_k_proj.T              [F_span, hidden]
        //   term_bd[i,j] = sum_d Q[i,d] * proj[rel(i,j), d]  (then _relative_shift)
        // Until then attention uses content scores only (term_ac).
        let _rel_k = &attn.relative_k_proj;

        // ── Chunked-local masked attention (term_ac + softcap + softmax · V). ──
        // chunk = attention_chunk_size; each query at frame i attends to keys in
        // [chunk_start - context_left, chunk_end + context_right) where the chunk
        // is floor(i/chunk)*chunk .. that+chunk. context_right=0 → strictly causal
        // within the right edge.
        // TODO(gpu-verify): exact block/context window semantics vs HF
        // _convert_to_block / _extract_block_context. This implements the
        // equivalent per-query allowed-key set directly.
        let chunk = s.attention_chunk_size.max(1);
        let ctx_left = s.attention_context_left;
        let ctx_right = s.attention_context_right;
        let cap = s.attention_logit_cap;

        let mut attn_out = vec![0f32; batch * h];
        for head in 0..nh {
            for i in 0..batch {
                let chunk_idx = i / chunk;
                let chunk_start = chunk_idx * chunk;
                let chunk_end = (chunk_start + chunk).min(batch);
                // allowed key window for this query.
                let lo = chunk_start.saturating_sub(ctx_left);
                let hi = (chunk_end + ctx_right).min(batch);
                // scores over [lo, hi).
                let mut scores = Vec::with_capacity(hi - lo);
                let mut max_s = f32::NEG_INFINITY;
                for j in lo..hi {
                    let mut dot = 0f32;
                    for d in 0..hd {
                        dot += q[i * h + head * hd + d] * k[j * h + head * hd + d];
                    }
                    // tanh logit softcap: cap * tanh(score / cap).
                    let capped = if cap > 0.0 {
                        cap * (dot / cap).tanh()
                    } else {
                        dot
                    };
                    if capped > max_s {
                        max_s = capped;
                    }
                    scores.push(capped);
                }
                // softmax.
                let mut sum = 0f32;
                for sc in scores.iter_mut() {
                    *sc = (*sc - max_s).exp();
                    sum += *sc;
                }
                let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                // weighted V.
                for d in 0..hd {
                    let mut acc = 0f32;
                    for (idx, j) in (lo..hi).enumerate() {
                        acc += scores[idx] * inv * v[j * h + head * hd + d];
                    }
                    attn_out[i * h + head * hd + d] = acc;
                }
            }
        }

        // post projection (clipped linear, [hidden, hidden]).
        let post = self.clip_linear_host(runtime, &attn_out, batch, &attn.post)?;
        // norm_post_attn then residual add.
        let post_n = self.rms_norm_to(runtime, &post, batch, &layer.norm_post_attn, eps)?;
        for i in 0..batch * h {
            state[i] += post_n[i];
        }
        Ok(())
    }
}

/// Read a scalar BF16 tensor (shape `[]`) to f32 on the host.
fn read_scalar_bf16(
    artifact: &ModelArtifact,
    name: &str,
    store: StoragePlacement,
    loader: &mut TensorStorageLoader,
) -> Result<f32> {
    let tensor = artifact.tensors.tensors.get(name).ok_or_else(|| {
        AegisError::InvalidPlan(format!("audio tower: scalar tensor `{name}` missing"))
    })?;
    let loaded = loader.load_for_store(tensor, store)?;
    let bytes = loaded.as_bytes();
    if bytes.len() < 2 {
        return Err(AegisError::InvalidPlan(format!(
            "audio tower: scalar `{name}` too short ({} bytes)",
            bytes.len()
        )));
    }
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    Ok(f32::from_bits((bits as u32) << 16))
}

/// Download a BF16 tensor of any rank to a flat host f32 vector (row-major).
fn download_bf16_tensor(
    artifact: &ModelArtifact,
    tensor: &TensorInfo,
    store: StoragePlacement,
    loader: &mut TensorStorageLoader,
) -> Result<Vec<f32>> {
    let _ = artifact;
    let loaded = loader.load_for_store(tensor, store)?;
    let bytes = loaded.as_bytes();
    Ok(bytes
        .chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect())
}

/// CPU Conv2d(k=3, s=2, pad=1) + CumulativeGroupNorm + ReLU.
///
/// Input  `x`: [in_ch, t_in, f_in] row-major f32 (channel-major).
/// Weight: block.conv_weight [out_ch, in_ch, 3, 3] row-major f32.
/// Output: [out_ch, t_out, f_out] row-major f32, with
///   t_out = (t_in + 2*pad - k)/stride + 1, f_out likewise.
///
/// CumulativeGroupNorm: per (time, channel-group) cumulative-over-time mean/var
/// across the reduction axes (here group = all channels + freq). This impl uses
/// a per-frame (across channels & freq) running cumulative mean/variance to
/// mirror HF's cumulative statistics.
///
/// TODO(gpu-verify): the cumulative group-norm reduction axes + the symmetric
/// pad=1 (HF uses asymmetric manual padding) are the two details most likely to
/// differ from HF; cross-check against a dump.
fn conv2d_norm_relu(
    x: &[f32],
    in_ch: usize,
    t_in: usize,
    f_in: usize,
    block: &AudioSubsampleConvBlock,
) -> Result<(Vec<f32>, usize, usize, usize)> {
    let k = 3usize;
    let stride = 2usize;
    let pad = 1usize;
    let out_ch = block.out_channels;
    if block.in_channels != in_ch {
        return Err(AegisError::InvalidPlan(format!(
            "audio conv: block.in_channels={} != in_ch={}",
            block.in_channels, in_ch
        )));
    }
    let t_out = (t_in + 2 * pad - k) / stride + 1;
    let f_out = (f_in + 2 * pad - k) / stride + 1;
    let mut conv = vec![0f32; out_ch * t_out * f_out];

    for oc in 0..out_ch {
        for ot in 0..t_out {
            for of in 0..f_out {
                let mut acc = 0f32;
                for ic in 0..in_ch {
                    for kt in 0..k {
                        let it = ot * stride + kt;
                        if it < pad {
                            continue;
                        }
                        let it = it - pad;
                        if it >= t_in {
                            continue;
                        }
                        for kf in 0..k {
                            let iff = of * stride + kf;
                            if iff < pad {
                                continue;
                            }
                            let iff = iff - pad;
                            if iff >= f_in {
                                continue;
                            }
                            let w = block.conv_weight
                                [((oc * in_ch + ic) * k + kt) * k + kf];
                            let xv = x[(ic * t_in + it) * f_in + iff];
                            acc += w * xv;
                        }
                    }
                }
                conv[(oc * t_out + ot) * f_out + of] = acc;
            }
        }
    }

    // CumulativeGroupNorm: cumulative-over-time mean/var across (channel, freq)
    // for each time index, scaled by norm_weight[channel], then ReLU.
    let eps = 1.0e-3f32; // TODO(gpu-verify): confirm group-norm eps (HF default).
    let mut out = vec![0f32; out_ch * t_out * f_out];
    let mut run_sum = 0f64;
    let mut run_sumsq = 0f64;
    let mut run_count = 0f64;
    let per_t = (out_ch * f_out) as f64;
    for ot in 0..t_out {
        // accumulate this time slice's sum/sumsq across all channels & freqs.
        for oc in 0..out_ch {
            for of in 0..f_out {
                let val = conv[(oc * t_out + ot) * f_out + of] as f64;
                run_sum += val;
                run_sumsq += val * val;
            }
        }
        run_count += per_t;
        let mean = run_sum / run_count;
        let var = (run_sumsq / run_count) - mean * mean;
        let inv_std = 1.0 / (var + eps as f64).sqrt();
        for oc in 0..out_ch {
            for of in 0..f_out {
                let idx = (oc * t_out + ot) * f_out + of;
                let normed = ((conv[idx] as f64 - mean) * inv_std) as f32;
                let scaled = normed * block.norm_weight[oc];
                out[idx] = scaled.max(0.0); // ReLU
            }
        }
    }

    Ok((out, out_ch, t_out, f_out))
}

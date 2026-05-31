//! Qwen3-Next Gated DeltaNet (linear-attention) mixer — load + forward.
//!
//! A GDN layer replaces the self-attention sublayer (the MLP sublayer is
//! unchanged: dense for the 9B, MoE for the 35B). The decode forward chains the
//! GPU-validated GDN kernels (see `cuda/kernels/blackwell/gated_deltanet_decode.cu`
//! and the launchers in `cuda/runtime/state_cache.rs`):
//!
//!   x → rms_norm(input_norm) → in_proj_qkv ─┐
//!                              in_proj_z → z │
//!                              in_proj_b → b │
//!                              in_proj_a → a │
//!   qkv → conv1d_decode(+state)+SiLU → split q[n_k,d_k] k[n_k,d_k] v[n_v,d_v]
//!   (q,k) → qk_norm_expand → q_n,k_n[n_v,d_k];  (b,a) → gate → beta,g[n_v]
//!   decode_step(state, q_n,k_n,v,beta,g) → o[n_v,d_v]
//!   gated_rmsnorm(o, z, norm) → o_norm;  out_proj(o_norm) → mixer_out[H]
//!   residual = hidden + mixer_out
//!
//! Buffers are allocated per call (correctness-first bring-up — GDN models are
//! new, so there is no decode-tps baseline to regress; pooling is a follow-up).

use aegisllm_base::error::{AegisError, Result};
use crate::cuda::DeviceBuffer;
use crate::cuda::runtime::CudaRuntime;
use super::linear_ops::matvec_cuda_linear_with_scratch;
use super::state::{
    CudaLayer, CudaLayerState, CudaLinear, CudaPrefillScratch, CudaScratch,
};

/// Resolved GDN dimensions for one layer (all GDN layers share these).
#[derive(Debug, Clone, Copy)]
pub(super) struct GdnDims {
    pub(super) num_k_heads: usize,
    pub(super) num_v_heads: usize,
    pub(super) k_head_dim: usize,
    pub(super) v_head_dim: usize,
    pub(super) conv_kernel: usize,
}

impl GdnDims {
    /// cat[q,k,v] width carried through the conv (= in_proj_qkv output width).
    pub(super) fn conv_channels(&self) -> usize {
        2 * self.num_k_heads * self.k_head_dim + self.num_v_heads * self.v_head_dim
    }
    pub(super) fn qk_width(&self) -> usize {
        self.num_k_heads * self.k_head_dim
    }
    pub(super) fn v_width(&self) -> usize {
        self.num_v_heads * self.v_head_dim
    }
    /// Recurrent state elements: [n_v, d_v, d_k] (kernel layout).
    pub(super) fn state_elems(&self) -> usize {
        self.num_v_heads * self.v_head_dim * self.k_head_dim
    }
    pub(super) fn conv_state_elems(&self) -> usize {
        self.conv_channels() * (self.conv_kernel - 1)
    }
}

/// Persistent per-decode-step scratch for the GDN mixer. All GDN layers share
/// the same dims, so one set of buffers (sized to the GDN dims) is reused across
/// every GDN layer and every token — replacing the ~16 per-call `alloc_f32`s
/// that previously churned the stream allocator 16 × (#GDN layers) times per
/// token (480/token on the 35B's 30 GDN layers). Buffers are fully written
/// before read each step, so reuse is safe and bit-identical to fresh allocs.
#[derive(Debug)]
pub(super) struct GdnDecodeScratch {
    pub(super) normed: DeviceBuffer<f32>,
    pub(super) qkv: DeviceBuffer<f32>,
    pub(super) z: DeviceBuffer<f32>,
    pub(super) b: DeviceBuffer<f32>,
    pub(super) a: DeviceBuffer<f32>,
    pub(super) qkv_conv: DeviceBuffer<f32>,
    pub(super) q_raw: DeviceBuffer<f32>,
    pub(super) k_raw: DeviceBuffer<f32>,
    pub(super) v: DeviceBuffer<f32>,
    pub(super) q_n: DeviceBuffer<f32>,
    pub(super) k_n: DeviceBuffer<f32>,
    pub(super) beta: DeviceBuffer<f32>,
    pub(super) g: DeviceBuffer<f32>,
    pub(super) o: DeviceBuffer<f32>,
    pub(super) o_norm: DeviceBuffer<f32>,
    pub(super) mixer_out: DeviceBuffer<f32>,
}

impl GdnDecodeScratch {
    /// Allocate once, sized to the (shared) GDN dims + the model hidden size.
    pub(super) fn new(runtime: &CudaRuntime, dims: GdnDims, hidden_size: usize) -> Result<Self> {
        let conv_ch = dims.conv_channels();
        let qk = dims.qk_width();
        let v_width = dims.v_width();
        let n_v = dims.num_v_heads;
        let d_k = dims.k_head_dim;
        Ok(Self {
            normed: runtime.alloc_f32(hidden_size)?,
            qkv: runtime.alloc_f32(conv_ch)?,
            z: runtime.alloc_f32(v_width)?,
            b: runtime.alloc_f32(n_v)?,
            a: runtime.alloc_f32(n_v)?,
            qkv_conv: runtime.alloc_f32(conv_ch)?,
            q_raw: runtime.alloc_f32(qk)?,
            k_raw: runtime.alloc_f32(qk)?,
            v: runtime.alloc_f32(v_width)?,
            q_n: runtime.alloc_f32(n_v * d_k)?,
            k_n: runtime.alloc_f32(n_v * d_k)?,
            beta: runtime.alloc_f32(n_v)?,
            g: runtime.alloc_f32(n_v)?,
            o: runtime.alloc_f32(v_width)?,
            o_norm: runtime.alloc_f32(v_width)?,
            mixer_out: runtime.alloc_f32(hidden_size)?,
        })
    }
}

/// Loaded Gated DeltaNet weights for one layer.
#[derive(Debug)]
pub(super) struct CudaGdn {
    pub(super) in_proj_qkv: CudaLinear,
    pub(super) in_proj_z: CudaLinear,
    pub(super) in_proj_b: CudaLinear,
    pub(super) in_proj_a: CudaLinear,
    pub(super) out_proj: CudaLinear,
    /// `conv1d.weight` flattened `[channels, 1, kernel]` → `channels*kernel` f32.
    pub(super) conv1d_weight: DeviceBuffer<f32>,
    pub(super) a_log: DeviceBuffer<f32>,
    pub(super) dt_bias: DeviceBuffer<f32>,
    /// `norm.weight` `[v_head_dim]` for the gated RMSNorm.
    pub(super) norm_weight: DeviceBuffer<f32>,
    pub(super) dims: GdnDims,
}

/// One GDN decode step. Reads `hidden` (residual stream), writes the post-mixer
/// residual into `scratch.residual` (same contract as `forward_attention_device`:
/// `residual = hidden + mixer_out`). Threads `layer_state.recurrent` and
/// `layer_state.conv_state` in place.
pub(super) fn forward_gdn_mixer_decode(
    runtime: &CudaRuntime,
    layer: &CudaLayer,
    layer_state: &mut CudaLayerState,
    hidden: &DeviceBuffer<f32>,
    scratch: &mut CudaScratch,
    rms_norm_eps: f32,
) -> Result<()> {
    let gdn = layer
        .gdn
        .as_ref()
        .ok_or_else(|| AegisError::InvalidPlan("forward_gdn_mixer: layer has no GDN weights".into()))?;
    let d = gdn.dims;
    let (n_k, n_v, d_k, d_v, kc) =
        (d.num_k_heads, d.num_v_heads, d.k_head_dim, d.v_head_dim, d.conv_kernel);
    let conv_ch = d.conv_channels();
    let qk = d.qk_width();
    let v_width = d.v_width();

    // Persistent per-step scratch (one alloc set, reused across all GDN layers
    // and all tokens). Replaces the per-call `alloc_f32` churn. Borrow the GDN
    // scratch + the shared quant/mxfp4/residual buffers disjointly via raw
    // pointers (distinct fields of `scratch`).
    // SAFETY: `gdn_decode`, `quant_hidden`, `mxfp4_hidden`, `residual` are
    // distinct fields of `scratch`; the raw pointers only sidestep the borrow
    // checker over the single `&mut scratch`.
    let gs: *mut GdnDecodeScratch = scratch
        .gdn_decode
        .as_deref_mut()
        .ok_or_else(|| AegisError::InvalidPlan(
            "forward_gdn_mixer: GDN decode scratch not allocated".into(),
        ))? as *mut _;
    let quant_hidden = &mut scratch.quant_hidden as *mut DeviceBuffer<f32>;
    let mxfp4_hidden = &mut scratch.mxfp4_hidden as *mut DeviceBuffer<u8>;
    let gs = unsafe { &mut *gs };

    // 1. input RMSNorm.
    runtime.rms_norm_device(hidden, &layer.input_norm_weight, rms_norm_eps, &mut gs.normed)?;

    // 2. input projections.
    // in_proj_qkv already emits the contiguous [all_q | all_k | all_v] layout
    // (HF `torch.split(mixed_qkv, [key_dim, key_dim, value_dim])`), which is
    // exactly the channel order the depthwise conv1d weight expects — no
    // per-head de-interleave (an earlier wrong assumption corrupted the conv
    // channel↔filter mapping and the q/k/v split, giving cos≈0.5 at layer 0).
    matvec_cuda_linear_with_scratch(
        runtime, &gdn.in_proj_qkv, &gs.normed,
        unsafe { &mut *quant_hidden }, unsafe { &mut *mxfp4_hidden }, &mut gs.qkv, None,
    )?;
    matvec_cuda_linear_with_scratch(
        runtime, &gdn.in_proj_z, &gs.normed,
        unsafe { &mut *quant_hidden }, unsafe { &mut *mxfp4_hidden }, &mut gs.z, None,
    )?;
    matvec_cuda_linear_with_scratch(
        runtime, &gdn.in_proj_b, &gs.normed,
        unsafe { &mut *quant_hidden }, unsafe { &mut *mxfp4_hidden }, &mut gs.b, None,
    )?;
    matvec_cuda_linear_with_scratch(
        runtime, &gdn.in_proj_a, &gs.normed,
        unsafe { &mut *quant_hidden }, unsafe { &mut *mxfp4_hidden }, &mut gs.a, None,
    )?;

    // 3. streaming depthwise causal conv1d + SiLU over cat[q,k,v].
    {
        let conv_state = layer_state.conv_state.as_mut().ok_or_else(|| {
            AegisError::InvalidPlan("forward_gdn_mixer: missing conv state".into())
        })?;
        runtime.gdn_conv1d_decode(&gs.qkv, conv_state, &gdn.conv1d_weight, &mut gs.qkv_conv, conv_ch, kc)?;
    }

    // 4. split q/k/v out of the fused conv output.
    {
        let qkv_conv = &gs.qkv_conv as *const DeviceBuffer<f32>;
        // SAFETY: qkv_conv (read) is distinct from q_raw/k_raw/v (write).
        runtime.copy_f32_d2d_range(unsafe { &*qkv_conv }, 0, &mut gs.q_raw, 0, qk)?;
        runtime.copy_f32_d2d_range(unsafe { &*qkv_conv }, qk, &mut gs.k_raw, 0, qk)?;
        runtime.copy_f32_d2d_range(unsafe { &*qkv_conv }, 2 * qk, &mut gs.v, 0, v_width)?;
    }

    // 5. L2-norm + GQA-expand q,k → [n_v, d_k]; compute beta,g.
    {
        let q_raw = &gs.q_raw as *const DeviceBuffer<f32>;
        let k_raw = &gs.k_raw as *const DeviceBuffer<f32>;
        // SAFETY: q_raw/k_raw (read) are distinct from q_n/k_n (write).
        runtime.gdn_qk_norm_expand(
            unsafe { &*q_raw }, unsafe { &*k_raw }, &mut gs.q_n, &mut gs.k_n, n_k, n_v, d_k,
        )?;
    }
    {
        let b = &gs.b as *const DeviceBuffer<f32>;
        let a = &gs.a as *const DeviceBuffer<f32>;
        // SAFETY: b/a (read) are distinct from beta/g (write).
        runtime.gdn_gate(
            unsafe { &*b }, unsafe { &*a }, &gdn.a_log, &gdn.dt_bias, &mut gs.beta, &mut gs.g, n_v,
        )?;
    }

    // 6. recurrent delta-rule step.
    {
        let state = layer_state.recurrent.as_mut().ok_or_else(|| {
            AegisError::InvalidPlan("forward_gdn_mixer: missing recurrent state".into())
        })?;
        runtime.gated_deltanet_decode_step(
            state, &gs.q_n, &gs.k_n, &gs.v, &gs.beta, &gs.g, &mut gs.o, n_v, d_k, d_v,
        )?;
    }

    // 7. gated RMSNorm (gate-first by silu(z)), then out_proj.
    {
        let o = &gs.o as *const DeviceBuffer<f32>;
        let z = &gs.z as *const DeviceBuffer<f32>;
        // SAFETY: o/z (read) are distinct from o_norm (write).
        runtime.gdn_gated_rmsnorm(
            unsafe { &*o }, unsafe { &*z }, &gdn.norm_weight, &mut gs.o_norm, n_v, d_v, rms_norm_eps,
        )?;
    }
    matvec_cuda_linear_with_scratch(
        runtime, &gdn.out_proj, &gs.o_norm,
        unsafe { &mut *quant_hidden }, unsafe { &mut *mxfp4_hidden }, &mut gs.mixer_out, None,
    )?;

    // 8. residual add (Qwen is PreOnly — no post-sublayer norm).
    runtime.add_device(hidden, &gs.mixer_out, &mut scratch.residual)?;
    Ok(())
}

/// Batched (chunked-prefill) Gated DeltaNet mixer over a `batch`-token chunk.
/// Mirrors `forward_gdn_mixer_decode` but processes the whole chunk in one pass
/// through each kernel. The recurrent + conv state in `layer_state` are threaded
/// in place (they persist across chunks; zero-initialized for a fresh prompt).
///
/// Reads `prefill.hidden` (`[batch, hidden]`, the residual stream) and writes the
/// post-mixer residual back into `prefill.hidden` in place. FP8 in/out
/// projections use the native block-scaled tensor-core GEMM (no dequant);
/// BF16 in_proj_a/b use the batched BF16 reference matmul.
pub(super) fn forward_gdn_mixer_prefill_chunk(
    runtime: &CudaRuntime,
    layer: &CudaLayer,
    layer_state: &mut CudaLayerState,
    prefill: &mut CudaPrefillScratch,
    batch: usize,
    hidden_size: usize,
    rms_norm_eps: f32,
) -> Result<()> {
    let gdn = layer
        .gdn
        .as_ref()
        .ok_or_else(|| AegisError::InvalidPlan("forward_gdn_mixer_prefill: layer has no GDN".into()))?;
    let d = gdn.dims;
    let (n_k, n_v, d_k, d_v, kc) =
        (d.num_k_heads, d.num_v_heads, d.k_head_dim, d.v_head_dim, d.conv_kernel);
    let conv_ch = d.conv_channels();
    let qk = d.qk_width();
    let v_width = d.v_width();

    // 1. input RMSNorm (batched). prefill.input_normed = [batch, hidden].
    runtime.rms_norm_batched_device(
        &prefill.hidden,
        &layer.input_norm_weight,
        batch,
        rms_norm_eps,
        &mut prefill.input_normed,
    )?;

    // 2. input projections (batched). FP8-block → native tensor-core GEMM
    //    (no dequant); BF16 → batched reference matmul. in_proj_qkv emits the
    //    contiguous [all_q | all_k | all_v] layout (no de-interleave — see the
    //    decode path comment).
    gdn_proj_batched(runtime, &gdn.in_proj_qkv, &prefill.input_normed, batch,
        &mut prefill.fp8_a_q, &mut prefill.fp8_a_scale, &mut prefill.gdn_qkv)?;
    gdn_proj_batched(runtime, &gdn.in_proj_z, &prefill.input_normed, batch,
        &mut prefill.fp8_a_q, &mut prefill.fp8_a_scale, &mut prefill.gdn_z)?;
    gdn_proj_batched(runtime, &gdn.in_proj_b, &prefill.input_normed, batch,
        &mut prefill.fp8_a_q, &mut prefill.fp8_a_scale, &mut prefill.gdn_b)?;
    gdn_proj_batched(runtime, &gdn.in_proj_a, &prefill.input_normed, batch,
        &mut prefill.fp8_a_q, &mut prefill.fp8_a_scale, &mut prefill.gdn_a)?;

    // 3. batched depthwise causal conv1d + SiLU over cat[q,k,v]. Threads the
    //    per-channel conv_state in place across the chunk.
    let conv_state = layer_state.conv_state.as_mut().ok_or_else(|| {
        AegisError::InvalidPlan("forward_gdn_mixer_prefill: missing conv state".into())
    })?;
    runtime.gdn_conv1d_prefill(
        &prefill.gdn_qkv, conv_state, &gdn.conv1d_weight, &mut prefill.gdn_conv_out,
        batch, conv_ch, kc,
    )?;

    // 4. split q/k/v out of the fused conv output ([batch, conv_ch]).
    runtime.strided_copy_2d(&prefill.gdn_conv_out, &mut prefill.gdn_q_raw, batch, qk, conv_ch, qk, 0)?;
    runtime.strided_copy_2d(&prefill.gdn_conv_out, &mut prefill.gdn_k_raw, batch, qk, conv_ch, qk, qk)?;
    runtime.strided_copy_2d(&prefill.gdn_conv_out, &mut prefill.gdn_v, batch, v_width, conv_ch, v_width, 2 * qk)?;

    // 5. L2-norm + GQA-expand q,k → [batch, n_v, d_k]; compute beta,g.
    let expand = n_v / n_k;
    runtime.gdn_qk_norm_expand_batched(
        &prefill.gdn_q_raw, &prefill.gdn_k_raw, &mut prefill.gdn_q_n, &mut prefill.gdn_k_n,
        batch, n_k, n_v, d_k, expand,
    )?;
    runtime.gdn_gate_batched(
        &prefill.gdn_b, &prefill.gdn_a, &gdn.a_log, &gdn.dt_bias,
        &mut prefill.gdn_beta, &mut prefill.gdn_g, batch, n_v,
    )?;

    // 6. batched recurrent delta-rule over the chunk.
    let state = layer_state.recurrent.as_mut().ok_or_else(|| {
        AegisError::InvalidPlan("forward_gdn_mixer_prefill: missing recurrent state".into())
    })?;
    runtime.gated_deltanet_prefill_step(
        state, &prefill.gdn_q_n, &prefill.gdn_k_n, &prefill.gdn_v,
        &prefill.gdn_beta, &prefill.gdn_g, &mut prefill.gdn_o,
        batch, n_v, d_k, d_v,
    )?;

    // 7. gated RMSNorm (gate-first by silu(z)), then out_proj.
    runtime.gdn_gated_rmsnorm_batched(
        &prefill.gdn_o, &prefill.gdn_z, &gdn.norm_weight, &mut prefill.gdn_o_norm,
        batch, n_v, d_v, rms_norm_eps,
    )?;
    gdn_proj_batched(runtime, &gdn.out_proj, &prefill.gdn_o_norm, batch,
        &mut prefill.fp8_a_q, &mut prefill.fp8_a_scale, &mut prefill.gdn_mixer_out)?;

    // 8. residual add (Qwen is PreOnly — no post-sublayer norm).
    runtime.add_inplace_device_len(&mut prefill.hidden, &prefill.gdn_mixer_out, batch * hidden_size)?;
    Ok(())
}

/// Batched GDN projection dispatch: FP8-block → native tensor-core GEMM (no
/// dequant), BF16 → batched reference matmul. NVFP4 GDN projections are not
/// produced by any current checkpoint (GDN weights are FP8/BF16).
fn gdn_proj_batched(
    runtime: &CudaRuntime,
    linear: &CudaLinear,
    input: &DeviceBuffer<f32>,
    batch: usize,
    a_q: &mut DeviceBuffer<u8>,
    a_scale: &mut DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
) -> Result<()> {
    match linear {
        CudaLinear::Fp8(f) if f.is_block_scaled() => {
            runtime.matmul_fp8_block_native_batched(f, input, batch, a_q, a_scale, output)
        }
        CudaLinear::Bf16(m) => {
            runtime.matmul_bf16_reference_batched_device(m, input, batch, output)
        }
        CudaLinear::Fp8(f) => runtime.matmul_fp8_standalone_batched_device(f, input, batch, output),
        CudaLinear::Nvfp4(_) => Err(AegisError::Unsupported(
            "GDN prefill: NVFP4 in/out projection not supported".into(),
        )),
    }
}

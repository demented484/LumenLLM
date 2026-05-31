use super::linear_ops::{
    matvec_cuda_linear_with_scratch, matvec_nvfp4_device_with_scratch,
    matvec_nvfp4_prepared_device_reuse, native_mxfp4_enabled, nvfp4_a16_enabled,
    prepare_nvfp4_input,
};
use super::state::{CudaLayer, CudaLinear, CudaMoE, CudaMoEScratch, CudaScratch};
use crate::cuda::{CudaRuntime, DeviceBuffer};
use crate::cuda::staging::LinearStagingPool;
use aegisllm_base::error::{AegisError, Result};

/// Opt-in (default OFF): batch the GPU-driven decode MoE expert GEMVs over all
/// top_k experts (slot on grid.y) instead of the per-slot serial loop. Removes
/// 56 launches/layer and fills the GPU; only beneficial when experts are
/// VRAM-resident (fast gather). Output is bit-identical to the per-slot path.
/// Read once (decode hot path) — env lookup is not per-token.
fn batched_decode_moe_enabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("AEGIS_BATCHED_DECODE_MOE").is_ok())
}

pub(super) fn forward_mlp_device(
    runtime: &CudaRuntime,
    layer: &CudaLayer,
    layer_idx: usize,
    ple_global: Option<&crate::executor::state::PleGlobal>,
    scratch: &mut CudaScratch,
    rms_norm_eps: f32,
) -> Result<()> {
    if let Some(ref moe) = layer.moe {
        // SAFETY: both raw pointers point to distinct fields of `scratch` — `moe` and
        // `staging_pool` are separate from each other and from `residual`/`post_normed`/
        // `hidden_out`.  We use raw pointers so the borrow checker sees no conflicting
        // mutable borrows of the same struct.
        let moe_scratch_ptr: *mut CudaMoEScratch = scratch
            .moe
            .as_deref_mut()
            .expect("CudaMoEScratch must be allocated for MoE layers");
        let staging_ptr: *mut LinearStagingPool = scratch
            .staging_pool
            .as_deref_mut()
            .map_or(std::ptr::null_mut(), |p| p as *mut _);
        return forward_moe_decode_device(
            runtime,
            layer,
            moe,
            &scratch.residual,
            &mut scratch.post_normed,
            &mut scratch.hidden_out,
            unsafe { &mut *moe_scratch_ptr },
            if staging_ptr.is_null() { None } else { Some(unsafe { &mut *staging_ptr }) },
            rms_norm_eps,
        );
    }
    // Dense MLP: dispatch on weight variant. NVFP4-only path (all three of
    // gate/up/down are NVFP4) keeps the existing native-MXFP4 / CUTLASS /
    // unfused fallbacks. BF16 and FP8 variants take the cuBLASLt dense path
    // below. Mixed variants are rejected — checkpoint format is uniform per
    // dense layer.
    let nvfp4_triple = layer.gate_proj.as_nvfp4()
        .zip(layer.up_proj.as_nvfp4())
        .zip(layer.down_proj.as_nvfp4());
    if nvfp4_triple.is_none() {
        return forward_dense_mlp_non_nvfp4_device(
            runtime, layer, layer_idx, ple_global, scratch, rms_norm_eps,
        );
    }
    let (gate_proj_nvfp4, up_proj_nvfp4, down_proj_nvfp4) = {
        let ((g, u), d) = nvfp4_triple.unwrap();
        (g, u, d)
    };
    // From here on out, the NVFP4-specific code uses the unwrapped refs.
    // For decode (M=1), native MXFP4 MATVEC is strongly preferred over CUTLASS.
    // CUTLASS tiles are 128×128, so M=1 uses <1% of each tile. Native MXFP4 GEMV
    // with hardware mxf4 MMA instructions is purpose-built for this shape.
    let use_native_gate_up = native_mxfp4_enabled(runtime, gate_proj_nvfp4)
        && native_mxfp4_enabled(runtime, up_proj_nvfp4);
    if use_native_gate_up {
        runtime.rms_norm_device(
            &scratch.residual,
            &layer.post_attention_norm_weight,
            rms_norm_eps,
            &mut scratch.post_normed,
        )?;
        // Gate: quantize post_normed to MXFP4, run native GEMV.
        let mxfp4_valid = matvec_nvfp4_prepared_device_reuse(
            runtime,
            gate_proj_nvfp4,
            &scratch.post_normed,
            &scratch.quant_hidden,
            &mut scratch.mxfp4_hidden,
            false,
            &mut scratch.gate,
            scratch.staging_pool.as_deref_mut(),
        )?;
        // Up: reuse MXFP4-quantized input (same post_normed), skip re-quantize.
        matvec_nvfp4_prepared_device_reuse(
            runtime,
            up_proj_nvfp4,
            &scratch.post_normed,
            &scratch.quant_hidden,
            &mut scratch.mxfp4_hidden,
            mxfp4_valid,
            &mut scratch.up,
            scratch.staging_pool.as_deref_mut(),
        )?;
    } else if runtime.cutlass_nvfp4_inference_enabled_for(gate_proj_nvfp4)
        && runtime.cutlass_nvfp4_inference_enabled_for(up_proj_nvfp4)
    {
        runtime.rms_norm_device(
            &scratch.residual,
            &layer.post_attention_norm_weight,
            rms_norm_eps,
            &mut scratch.post_normed,
        )?;
        runtime.quantize_cutlass_nvfp4_activation_device(
            &scratch.post_normed,
            1,
            gate_proj_nvfp4.cols,
            &mut scratch.cutlass_payload,
            &mut scratch.cutlass_scales,
        )?;
        runtime.matmul_cutlass_nvfp4_prepacked_prefill_device(
            gate_proj_nvfp4,
            &scratch.cutlass_payload,
            &scratch.cutlass_scales,
            1,
            &mut scratch.cutlass_workspace,
            &mut scratch.gate,
        )?;
        runtime.matmul_cutlass_nvfp4_prepacked_prefill_device(
            up_proj_nvfp4,
            &scratch.cutlass_payload,
            &scratch.cutlass_scales,
            1,
            &mut scratch.cutlass_workspace,
            &mut scratch.up,
        )?;
    } else {
        runtime.rms_norm_quant_nvfp4_device(
            &scratch.residual,
            &layer.post_attention_norm_weight,
            rms_norm_eps,
            gate_proj_nvfp4.input_scale,
            &mut scratch.post_normed,
            &mut scratch.quant_hidden,
        )?;
        let mut quant_scale = Some(gate_proj_nvfp4.input_scale);
        let mxfp4_valid = matvec_nvfp4_prepared_device_reuse(
            runtime,
            gate_proj_nvfp4,
            &scratch.post_normed,
            &scratch.quant_hidden,
            &mut scratch.mxfp4_hidden,
            false,
            &mut scratch.gate,
            scratch.staging_pool.as_deref_mut(),
        )?;
        prepare_nvfp4_input(
            runtime,
            up_proj_nvfp4,
            &scratch.post_normed,
            &mut quant_scale,
            &mut scratch.quant_hidden,
        )?;
        matvec_nvfp4_prepared_device_reuse(
            runtime,
            up_proj_nvfp4,
            &scratch.post_normed,
            &scratch.quant_hidden,
            &mut scratch.mxfp4_hidden,
            mxfp4_valid,
            &mut scratch.up,
            scratch.staging_pool.as_deref_mut(),
        )?;
    }
    let use_native_down = native_mxfp4_enabled(runtime, down_proj_nvfp4);
    if use_native_down {
        runtime.swiglu_device(&scratch.gate, &scratch.up, &mut scratch.swiglu)?;
        matvec_nvfp4_device_with_scratch(
            runtime,
            down_proj_nvfp4,
            &scratch.swiglu,
            &mut scratch.quant_intermediate,
            &mut scratch.mxfp4_intermediate,
            &mut scratch.mlp_out,
            scratch.staging_pool.as_deref_mut(),
        )?;
    } else if runtime.cutlass_nvfp4_inference_enabled_for(down_proj_nvfp4) {
        runtime.swiglu_quantize_cutlass_nvfp4_activation_device(
            &scratch.gate,
            &scratch.up,
            1,
            down_proj_nvfp4.cols,
            &mut scratch.cutlass_payload,
            &mut scratch.cutlass_scales,
        )?;
        runtime.matmul_cutlass_nvfp4_prepacked_prefill_device(
            down_proj_nvfp4,
            &scratch.cutlass_payload,
            &scratch.cutlass_scales,
            1,
            &mut scratch.cutlass_workspace,
            &mut scratch.mlp_out,
        )?;
    } else {
        runtime.swiglu_device(&scratch.gate, &scratch.up, &mut scratch.swiglu)?;
        matvec_nvfp4_device_with_scratch(
            runtime,
            down_proj_nvfp4,
            &scratch.swiglu,
            &mut scratch.quant_intermediate,
            &mut scratch.mxfp4_intermediate,
            &mut scratch.mlp_out,
            scratch.staging_pool.as_deref_mut(),
        )?;
    }
    if let Some(ref post_norm) = layer.post_mlp_sublayer_norm {
        // Gemma 4 PrePost: normalize MLP output before adding to residual.
        // scratch.post_normed is free at this point (pre-MLP norm is done).
        runtime.rms_norm_device(&scratch.mlp_out, post_norm, rms_norm_eps, &mut scratch.post_normed)?;
        runtime.add_device(&scratch.residual, &scratch.post_normed, &mut scratch.hidden_out)?;
    } else {
        runtime.add_device(&scratch.residual, &scratch.mlp_out, &mut scratch.hidden_out)?;
    }
    // PLE per-layer additive contribution — applied BEFORE layer_scalar to
    // match HF Gemma4DecoderLayer.forward (gemma4.py:739-752). No-op for
    // non-PLE models (layer.ple is None) and when the global apparatus
    // wasn't loaded (Gemma-4-26B-A4B et al.).
    if let Some(ple_g) = ple_global {
        crate::executor::ple::apply_ple_contribution_decode(
            runtime, layer, ple_g, layer_idx, scratch, rms_norm_eps,
        )?;
    }
    if let Some(scalar) = layer.layer_scalar {
        runtime.scale_f32_device(scalar, &mut scratch.hidden_out)?;
    }
    Ok(())
}

/// Dense MLP decode forward for BF16 and FP8 weight variants. Mirrors the
/// NVFP4 path's structure (RMSNorm → gate/up GEMMs → SwiGLU/GeGLU →
/// down GEMM → residual + optional post-norm + layer_scalar) but routes
/// through cuBLASLt BF16 or FP8 GEMMs instead of NVFP4 matvec kernels.
///
/// Selects SwiGLU vs GeGLU-tanh based on the layer's `dense_activation`
/// (driven by the architecture descriptor's `hidden_activation`). For
/// Gemma-4 E4B that's `gelu_pytorch_tanh`; for Llama / Qwen text it's
/// `silu` (SwiGLU).
fn forward_dense_mlp_non_nvfp4_device(
    runtime: &CudaRuntime,
    layer: &CudaLayer,
    layer_idx: usize,
    ple_global: Option<&crate::executor::state::PleGlobal>,
    scratch: &mut CudaScratch,
    rms_norm_eps: f32,
) -> Result<()> {
    runtime.rms_norm_device(
        &scratch.residual,
        &layer.post_attention_norm_weight,
        rms_norm_eps,
        &mut scratch.post_normed,
    )?;
    // Gate / up projection: dispatch on the (uniform) variant of the triple.
    // Mixed variants are rejected at load time by `load_cuda_linear` because
    // a single dense layer's checkpoint format is uniform across the three
    // sub-projections, so we only need to look at gate_proj here.
    match &layer.gate_proj {
        CudaLinear::Bf16(_) => {
            matvec_cuda_linear_with_scratch(
                runtime,
                &layer.gate_proj,
                &scratch.post_normed,
                &mut scratch.quant_hidden,
                &mut scratch.mxfp4_hidden,
                &mut scratch.gate,
                scratch.staging_pool.as_deref_mut(),
            )?;
            matvec_cuda_linear_with_scratch(
                runtime,
                &layer.up_proj,
                &scratch.post_normed,
                &mut scratch.quant_hidden,
                &mut scratch.mxfp4_hidden,
                &mut scratch.up,
                scratch.staging_pool.as_deref_mut(),
            )?;
        }
        CudaLinear::Fp8(_) => {
            matvec_cuda_linear_with_scratch(
                runtime,
                &layer.gate_proj,
                &scratch.post_normed,
                &mut scratch.quant_hidden,
                &mut scratch.mxfp4_hidden,
                &mut scratch.gate,
                scratch.staging_pool.as_deref_mut(),
            )?;
            matvec_cuda_linear_with_scratch(
                runtime,
                &layer.up_proj,
                &scratch.post_normed,
                &mut scratch.quant_hidden,
                &mut scratch.mxfp4_hidden,
                &mut scratch.up,
                scratch.staging_pool.as_deref_mut(),
            )?;
        }
        CudaLinear::Nvfp4(_) => unreachable!("NVFP4 path handled upstream"),
    }
    // Activation: dispatch on the architecture's MLP activation.
    match layer.dense_activation {
        DenseActivation::Swiglu => {
            runtime.swiglu_device(&scratch.gate, &scratch.up, &mut scratch.swiglu)?;
        }
        DenseActivation::GeluTanh => {
            runtime.geglu_tanh_device(&scratch.gate, &scratch.up, &mut scratch.swiglu)?;
        }
    }
    // Down projection.
    matvec_cuda_linear_with_scratch(
        runtime,
        &layer.down_proj,
        &scratch.swiglu,
        &mut scratch.quant_intermediate,
        &mut scratch.mxfp4_intermediate,
        &mut scratch.mlp_out,
        scratch.staging_pool.as_deref_mut(),
    )?;
    if let Some(ref post_norm) = layer.post_mlp_sublayer_norm {
        runtime.rms_norm_device(&scratch.mlp_out, post_norm, rms_norm_eps, &mut scratch.post_normed)?;
        runtime.add_device(&scratch.residual, &scratch.post_normed, &mut scratch.hidden_out)?;
    } else {
        runtime.add_device(&scratch.residual, &scratch.mlp_out, &mut scratch.hidden_out)?;
    }
    // PLE per-layer additive contribution — applied BEFORE layer_scalar to
    // match HF Gemma4DecoderLayer.forward (gemma4.py:739-752). No-op for
    // non-PLE models (layer.ple is None) and when the global apparatus
    // wasn't loaded (Gemma-4-26B-A4B et al.).
    if let Some(ple_g) = ple_global {
        crate::executor::ple::apply_ple_contribution_decode(
            runtime, layer, ple_g, layer_idx, scratch, rms_norm_eps,
        )?;
    }
    if let Some(scalar) = layer.layer_scalar {
        runtime.scale_f32_device(scalar, &mut scratch.hidden_out)?;
    }
    Ok(())
}

/// Dense MLP activation kind, decided per-architecture at load time. Gemma-4
/// E4B uses GeGLU-tanh (`gelu_pytorch_tanh`); Llama / Qwen text uses SwiGLU.
#[derive(Debug, Clone, Copy)]
pub(super) enum DenseActivation {
    Swiglu,
    GeluTanh,
}

#[allow(clippy::too_many_arguments)]
fn forward_moe_decode_device(
    runtime: &CudaRuntime,
    layer: &CudaLayer,
    moe: &CudaMoE,
    residual: &crate::cuda::DeviceBuffer<f32>,
    post_normed: &mut crate::cuda::DeviceBuffer<f32>,
    hidden_out: &mut crate::cuda::DeviceBuffer<f32>,
    moe_scratch: &mut CudaMoEScratch,
    staging: Option<&mut LinearStagingPool>,
    rms_norm_eps: f32,
) -> Result<()> {
    // Gemma 4 MoE forward follows the transformers Gemma4TextDecoderLayer:
    //   1. pre_feedforward_layernorm(residual)   → shared MLP input
    //   2. shared_mlp(normed)                    → shared_out
    //   3. post_feedforward_layernorm_1(shared_out) → stream1   [if present]
    //   4. router on residual (unnormed)
    //   5. pre_feedforward_layernorm_2(residual) → expert input [if present; else reuse normed]
    //   6. Σ top-k experts(expert_input)         → expert_combined
    //   7. post_feedforward_layernorm_2(combined) → stream2     [if present]
    //   8. combined = stream1 + stream2
    //   9. post_feedforward_layernorm(combined)  → normed_out   [post_mlp_sublayer_norm]
    //  10. hidden_out = residual + normed_out
    //  11. hidden_out *= layer_scalar             [if present]

    // Step 1: pre_feedforward_layernorm(residual) → post_normed (shared MLP input)
    runtime.rms_norm_device(residual, &layer.post_attention_norm_weight, rms_norm_eps, post_normed)?;

    // === Async-overlap router (decode) =====================================
    //
    // Goal: eliminate the per-MoE-layer host sync that previously serialized
    // attention output → router math → expert dispatch. Previous structure
    // issued `download_f32(router_logits)` which drained the compute stream,
    // blocking the CPU for ~24 MoE layers per token.
    //
    // New pipeline (single transfer stream, one CudaEvent per layer):
    //
    //   compute : rms_norm + scale + matvec_router + topk_packed → packed_topk_device
    //   compute : RECORD event_topk_ready
    //   transfer: WAIT event_topk_ready → memcpy_dtoh(packed_topk_pinned) [async]
    //
    //   compute : shared MLP (matvec gate_up, geglu, matvec down) → moe_acc
    //   compute : post_feedforward_layernorm_1(moe_acc) → post_normed (stream1)
    //
    //   host    : packed_topk_pinned.as_slice()  [waits on pinned's internal
    //             event = dtoh completion. By now the compute stream has
    //             issued shared MLP launches; the host wait is near-zero.]
    //   host    : parse u32 records → router_top_indices / router_top_weights
    //
    //   compute : expert pre-norm (or copy) → hidden_out
    //   compute : routed experts → routed_acc
    //   compute : post_feedforward_layernorm_2(routed_acc) → expert_out (stream2)
    //   compute : combine (post_normed + expert_out → moe_acc), final norm,
    //             residual add, scalar.
    //
    // Bit-equivalence: GPU softmax+topk produces ULP-level differences from
    // the prior CPU path (e.g. accumulation order of `expf`), but the order
    // of operations (renormalize → per_expert_scale) is identical. Same
    // behaviour as the prefill `router_softmax_topk_device` path that has
    // shipped since the GPU router landed.

    // Gemma 4 router (matches transformers Gemma4TextRouter.forward):
    //   hidden  = rms_norm(residual)            (no weight, just unit-variance normalize)
    //   hidden *= router.scale                  (per-input-dim BF16 vector)
    //   hidden *= 1 / sqrt(hidden_size)         (scalar_root_size)
    //   logits  = proj(hidden)                  (matmul, [num_experts, hidden_size])
    //   probs   = softmax(logits)
    //   weights, indices = topk(probs, k)
    //   weights /= sum(weights)                  (renormalize per token)
    //   weights *= per_expert_scale[indices]     (per-expert calibration on weights)
    let hidden_size = residual.len();
    let router_input: &crate::cuda::DeviceBuffer<f32> = match &moe.router_input_scale {
        Some(input_scale) => {
            runtime.rms_norm_device(
                residual,
                input_scale,
                rms_norm_eps,
                &mut moe_scratch.router_input_scratch,
            )?;
            let scalar_root_size = (hidden_size as f32).powf(-0.5);
            runtime.scale_f32_device(scalar_root_size, &mut moe_scratch.router_input_scratch)?;
            &moe_scratch.router_input_scratch
        }
        // Standard MoE (Qwen3-Next, Mixtral, …): the router runs on the SAME
        // post-attention-layernorm'd hidden as the experts/shared (HF
        // `gate(hidden_states)`), NOT the raw residual. Gemma uses its own
        // `router.scale` norm via the `Some` arm above.
        None => &*post_normed,
    };
    runtime.matvec_bf16_reference_device(&moe.router, router_input, &mut moe_scratch.router_logits)?;
    let top_k = moe.top_k;
    let num_experts = moe.num_experts;
    // GPU fused softmax+topk with per-expert scale renormalization. Writes
    // `top_k * 2` interleaved u32 words `(idx, bitcast(weight))` into
    // `packed_topk_device`.
    runtime.router_softmax_topk_packed_device(
        &moe_scratch.router_logits,
        &moe.router_per_expert_scale_device,
        1,
        num_experts,
        top_k,
        &mut moe_scratch.packed_topk_device,
    )?;
    // GPU-driven decode: the routed experts are gathered from device-mapped
    // host RAM by a GPU kernel reading the on-device top-k buffer — NO dtoh of
    // the top-k, NO host parse, NO host-issued per-expert copies. Keyed purely on
    // this layer's device tables having been built — which only happens at load
    // when AEGIS_GPU_DRIVEN_MOE is set AND the arena device-mapped AND every
    // expert resolved a host device pointer. Avoids an env syscall per MoE layer
    // in the (non-graphed first) decode step. Otherwise the host-streamed path
    // runs unchanged.
    let gpu_driven = moe.device_tables.is_some();
    if !gpu_driven {
        // Record compute-stream completion; transfer stream waits on it before
        // issuing the dtoh.
        runtime.record_into_compute(&moe_scratch.event_topk_ready)?;
        runtime.transfer_wait_event(&moe_scratch.event_topk_ready)?;
        // Single fused dtoh: `top_k * 8` bytes onto the pinned host buffer. The
        // pinned slice's internal event is auto-recorded by cudarc after the copy
        // completes; the host `as_slice()` call below synchronizes on it.
        let packed_words = top_k.checked_mul(2).ok_or_else(|| {
            aegisllm_base::error::AegisError::InvalidPlan(format!(
                "MoE packed top-k overflow: top_k={top_k}"
            ))
        })?;
        runtime.download_u32_to_pinned_async(
            &moe_scratch.packed_topk_device,
            &mut moe_scratch.packed_topk_pinned,
            packed_words,
        )?;
    }

    // Issue expert pre-norm BEFORE the host sync so it's queued behind shared
    // MLP launches. When `pre_feedforward_layernorm_2` is present this also
    // makes the post_normed buffer free for the shared-MLP path to reuse as
    // its output later.
    if let Some(ref norm2) = layer.pre_feedforward_layernorm_2 {
        runtime.rms_norm_device(residual, norm2, rms_norm_eps, hidden_out)?;
    } else {
        // Expert input == pre-MLP norm output. Copy it out NOW because the
        // shared-MLP write to `post_normed` further below would otherwise
        // clobber it.
        runtime.copy_f32_device(post_normed, hidden_out)?;
    }

    let staging_ptr: *mut LinearStagingPool =
        staging.map_or(std::ptr::null_mut(), |p| p as *mut _);

    // Steps 2-3: shared MLP on post_normed → moe_acc. Independent of router
    // top-k; runs concurrently with the dtoh and provides the overlap window
    // that makes the upcoming host sync cheap.
    if let Some(ref shared) = moe.shared_expert {
        if let Some(ref fused) = shared.gate_up_fused {
            let intermediate = shared.gate_proj.rows();
            runtime.matvec_bf16_reference_device(
                fused,
                post_normed,
                &mut moe_scratch.shared_gate_up_fused,
            )?;
            runtime.geglu_tanh_strided_device(
                &moe_scratch.shared_gate_up_fused,
                1,
                intermediate,
                &mut moe_scratch.expert_swiglu,
            )?;
        } else {
            matvec_cuda_linear_with_scratch(
                runtime, &shared.gate_proj, post_normed,
                &mut moe_scratch.quant_expert, &mut moe_scratch.mxfp4_expert,
                &mut moe_scratch.expert_gate,
                if staging_ptr.is_null() { None } else { Some(unsafe { &mut *staging_ptr }) },
            )?;
            matvec_cuda_linear_with_scratch(
                runtime, &shared.up_proj, post_normed,
                &mut moe_scratch.quant_expert, &mut moe_scratch.mxfp4_expert,
                &mut moe_scratch.expert_up,
                if staging_ptr.is_null() { None } else { Some(unsafe { &mut *staging_ptr }) },
            )?;
            // Qwen3-Next shared expert is SwiGLU (silu); Gemma is GeGLU-tanh.
            match layer.dense_activation {
                super::mlp::DenseActivation::Swiglu => runtime.swiglu_device(
                    &moe_scratch.expert_gate, &moe_scratch.expert_up, &mut moe_scratch.expert_swiglu,
                )?,
                super::mlp::DenseActivation::GeluTanh => runtime.geglu_tanh_device(
                    &moe_scratch.expert_gate, &moe_scratch.expert_up, &mut moe_scratch.expert_swiglu,
                )?,
            }
        }
        matvec_cuda_linear_with_scratch(
            runtime, &shared.down_proj, &moe_scratch.expert_swiglu,
            &mut moe_scratch.quant_expert, &mut moe_scratch.mxfp4_expert,
            &mut moe_scratch.moe_acc,
            if staging_ptr.is_null() { None } else { Some(unsafe { &mut *staging_ptr }) },
        )?;
        // Qwen3-Next: scale the shared-expert output by sigmoid(shared_gate · x),
        // where x is the (pre-shared-MLP) normed hidden still in `post_normed`.
        // Kept fully on-device: the gate logit lands in a persistent [1] scratch
        // and `scale_by_sigmoid_scalar` reads it directly — no host download / no
        // per-call alloc. Previously this did a blocking `download_f32` + host
        // sigmoid + relaunch PER MoE LAYER PER TOKEN (36 syncs/token on the 35B),
        // each stalling the CPU launch pipeline (cpu_issuing-bound decode).
        if let Some(ref sgate) = shared.shared_gate {
            runtime.matvec_bf16_reference_device(
                sgate,
                post_normed,
                &mut moe_scratch.shared_gate_logit,
            )?;
            let n = moe_scratch.moe_acc.len();
            let logit = &moe_scratch.shared_gate_logit as *const DeviceBuffer<f32>;
            // SAFETY: shared_gate_logit (read) and moe_acc (write) are distinct
            // fields of moe_scratch; the raw ptr only sidesteps the borrow
            // checker over the shared &mut moe_scratch.
            runtime.scale_by_sigmoid_scalar(
                &mut moe_scratch.moe_acc,
                unsafe { &*logit },
                n,
            )?;
        }
    } else {
        runtime.zero_f32_device(&mut moe_scratch.moe_acc)?;
    }

    // Step 3: post_feedforward_layernorm_1(moe_acc=shared_out) → post_normed (stream1)
    if let Some(ref norm1) = layer.post_feedforward_layernorm_1 {
        runtime.rms_norm_device(&moe_scratch.moe_acc, norm1, rms_norm_eps, post_normed)?;
    } else {
        runtime.copy_f32_device(&moe_scratch.moe_acc, post_normed)?;
    }

    // Host sync: wait for the packed dtoh, then parse the records into the
    // pooled top-k arrays. By this point the compute stream has dispatched
    // shared MLP gate_up / geglu / down_proj + post_norm_1 — enough work to
    // hide the dtoh latency. SKIPPED on the GPU-driven path: the top-k stays on
    // the device and is consumed by the gather kernel; the host never sees it.
    if !gpu_driven {
        let packed_host = moe_scratch
            .packed_topk_pinned
            .as_slice()
            .map_err(|e| aegisllm_base::error::AegisError::Unsupported(
                format!("pinned packed topk slice sync failed: {e:?}"),
            ))?;
        moe_scratch.router_top_indices.clear();
        moe_scratch.router_top_weights.clear();
        for k in 0..top_k {
            let idx_word = packed_host[k * 2];
            let weight_word = packed_host[k * 2 + 1];
            moe_scratch.router_top_indices.push(idx_word as usize);
            moe_scratch.router_top_weights.push(f32::from_bits(weight_word));
        }
    }
    // On the host path `active_top_k` is the parsed count; on the GPU-driven
    // path the router always selects exactly `top_k` experts (the kernel writes
    // `top_k` records), so we process every slot.
    let active_top_k = if gpu_driven { top_k } else { moe_scratch.router_top_indices.len() };

    // Routed experts → routed_acc (separate accumulator so it does not alias
    // with `moe_acc` which already holds the shared-MLP output).
    runtime.zero_f32_device(&mut moe_scratch.routed_acc)?;

    // ── GPU-driven expert dispatch (device-mapped-host gather) ───────────────
    // No CPU round-trip: a single gather kernel reads the on-device top-k index
    // buffer, streams the selected experts' packed+scales from device-mapped
    // host RAM into the bulk VRAM scratch (fixed slot-major layout) + writes the
    // per-slot NVFP4 scales, then per-slot GEMVs read those. The whole sequence
    // is FIXED (slot k → GEMV k) → graph-capturable. Bit-identical to the host
    // path: same experts (the gather reads the same indices the host would have
    // parsed), same weights/scales, same NVFP4 dequant + accumulation order.
    if gpu_driven {
        let tables = moe.device_tables.as_ref().expect("gpu_driven implies device_tables");
        let (bulk_packed, bulk_scales) = match (
            moe_scratch.bulk_expert_packed.as_ref(),
            moe_scratch.bulk_expert_scales.as_ref(),
        ) {
            (Some(_), Some(_)) => (true, true),
            _ => (false, false),
        };
        if !(bulk_packed && bulk_scales) {
            return Err(AegisError::InvalidPlan(
                "GPU-driven MoE decode requires the bulk expert buffers to be allocated".into(),
            ));
        }
        // Slot-major byte layout: slot k holds gate, up, down back-to-back at
        // their uniform strides (matches the gather kernel).
        let per_slot_packed =
            tables.gate_packed_bytes + tables.up_packed_bytes + tables.down_packed_bytes;
        let per_slot_scale =
            tables.gate_scale_bytes + tables.up_scale_bytes + tables.down_scale_bytes;
        // Account the gathered PCIe traffic in the shared H2D counter so
        // AEGIS_DECODE_TIMING's MiB/token + GB/s stay meaningful (the gather reads
        // device-mapped host directly, bypassing the staging pool that normally
        // increments this). Same bytes/token as the host path — only the path
        // (in-graph gather over mapped host vs host-issued memcpy) differs.
        // NOTE: under CUDA-graph REPLAY this Rust code doesn't run, so the
        // counter only ticks on the capture token; the per-token PCIe volume is
        // unchanged across replays, it just isn't re-counted host-side.
        crate::cuda::staging::STAGING_H2D_BYTES.fetch_add(
            ((per_slot_packed + per_slot_scale) * top_k) as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        // 1) Gather: device-mapped host → bulk VRAM + per-slot scale arrays.
        {
            let bp = moe_scratch.bulk_expert_packed.as_mut().unwrap() as *mut DeviceBuffer<u8>;
            let bs = moe_scratch.bulk_expert_scales.as_mut().unwrap() as *mut DeviceBuffer<u8>;
            // SAFETY: bp/bs/slot_*_scale are distinct fields of moe_scratch; we
            // use raw pointers only to satisfy the borrow checker across the
            // shared &mut moe_scratch.
            let sin = &mut moe_scratch.slot_in_scale as *mut DeviceBuffer<f32>;
            let sout = &mut moe_scratch.slot_out_scale as *mut DeviceBuffer<f32>;
            runtime.moe_gather_experts_device(
                &moe_scratch.packed_topk_device,
                top_k,
                num_experts,
                tables,
                unsafe { &mut *bp },
                unsafe { &mut *bs },
                unsafe { &mut *sin },
                unsafe { &mut *sout },
            )?;
        }
        // 2) GEMVs reading the gathered bulk buffer + device scales.
        // Expert projection shapes are uniform across experts (we use the first).
        let gate = &moe.experts[0].gate_proj;
        let up = &moe.experts[0].up_proj;
        let down = &moe.experts[0].down_proj;
        if batched_decode_moe_enabled() {
            // BATCHED path: collapse the top_k×[3 quant + 3 GEMV + 1 geglu +
            // 1 axpy] tiny serial launches into 9 launches/layer with the
            // expert/slot on grid.y. Per-(slot,row,element) math is identical to
            // the per-slot loop below (bit-identical output); only the for-loop
            // moves onto a grid axis, raising GPU fill + removing 56 launches.
            let tk = active_top_k;
            // Per-slot strides = the batched buffers' per-slot alloc widths
            // (sized [max_top_k * width] in full.rs). Derive max_top_k from
            // packed_topk_device ([max_top_k*2]).
            let max_top_k = moe_scratch.packed_topk_device.len() / 2;
            let inter_stride = moe_scratch.expert_gate_b.len() / max_top_k; // = max_expert_intermediate
            let quant_stride = moe_scratch.quant_b.len() / max_top_k; // = max_input
            let hidden_stride = down.rows; // = hidden_size (expert_out_b per-slot alloc width)
            let bp = moe_scratch.bulk_expert_packed.as_ref().unwrap() as *const DeviceBuffer<u8>;
            let bs = moe_scratch.bulk_expert_scales.as_ref().unwrap() as *const DeviceBuffer<u8>;
            // GATE: batched quant of the shared hidden (input_stride=0) → batched GEMV
            runtime.quantize_nvfp4_input_batched_dptr_device(
                hidden_out, 0, gate.cols, quant_stride,
                &moe_scratch.slot_in_scale, 0, tk, &mut moe_scratch.quant_b,
            )?;
            runtime.matvec_nvfp4_prequantized_batched_dptr_device(
                unsafe { &*bp }, unsafe { &*bs }, per_slot_packed, per_slot_scale,
                0, 0,
                &moe_scratch.quant_b, quant_stride, gate.rows, gate.cols,
                &moe_scratch.slot_out_scale, 0, inter_stride, tk, &mut moe_scratch.expert_gate_b,
            )?;
            // UP
            runtime.quantize_nvfp4_input_batched_dptr_device(
                hidden_out, 0, up.cols, quant_stride,
                &moe_scratch.slot_in_scale, 1, tk, &mut moe_scratch.quant_b,
            )?;
            runtime.matvec_nvfp4_prequantized_batched_dptr_device(
                unsafe { &*bp }, unsafe { &*bs }, per_slot_packed, per_slot_scale,
                tables.gate_packed_bytes, tables.gate_scale_bytes,
                &moe_scratch.quant_b, quant_stride, up.rows, up.cols,
                &moe_scratch.slot_out_scale, 1, inter_stride, tk, &mut moe_scratch.expert_up_b,
            )?;
            // GEGLU (batched, strided over slots)
            {
                let eg = &moe_scratch.expert_gate_b as *const DeviceBuffer<f32>;
                let eu = &moe_scratch.expert_up_b as *const DeviceBuffer<f32>;
                // SAFETY: eg/eu (read) are distinct from expert_swiglu_b (write).
                runtime.geglu_tanh_batched_slots_device(
                    unsafe { &*eg }, unsafe { &*eu }, gate.rows, inter_stride, tk,
                    &mut moe_scratch.expert_swiglu_b,
                )?;
            }
            // DOWN: batched quant of per-slot swiglu (input_stride=inter_stride) → batched GEMV
            {
                let es = &moe_scratch.expert_swiglu_b as *const DeviceBuffer<f32>;
                // SAFETY: es (read) is distinct from quant_b (write).
                runtime.quantize_nvfp4_input_batched_dptr_device(
                    unsafe { &*es }, inter_stride, down.cols, quant_stride,
                    &moe_scratch.slot_in_scale, 2, tk, &mut moe_scratch.quant_b,
                )?;
            }
            runtime.matvec_nvfp4_prequantized_batched_dptr_device(
                unsafe { &*bp }, unsafe { &*bs }, per_slot_packed, per_slot_scale,
                tables.gate_packed_bytes + tables.up_packed_bytes,
                tables.gate_scale_bytes + tables.up_scale_bytes,
                &moe_scratch.quant_b, quant_stride, down.rows, down.cols,
                &moe_scratch.slot_out_scale, 2, hidden_stride, tk, &mut moe_scratch.expert_out_b,
            )?;
            // WEIGHTED ACCUMULATE: routed_acc = Σ_k w[k]·expert_out[k]
            // (fixed ascending fold + single-expr FMA = bit-identical to the
            //  serial axpy chain; routed_acc was zeroed earlier — overwrite here).
            {
                let eo = &moe_scratch.expert_out_b as *const DeviceBuffer<f32>;
                let ptk = &moe_scratch.packed_topk_device as *const DeviceBuffer<u32>;
                // SAFETY: eo/ptk (read) are distinct from routed_acc (write).
                runtime.moe_weighted_accumulate_device(
                    &mut moe_scratch.routed_acc, unsafe { &*eo }, hidden_stride,
                    unsafe { &*ptk }, tk, down.rows,
                )?;
            }
        } else {
        for k in 0..active_top_k {
            let packed_base = k * per_slot_packed;
            let scale_base = k * per_slot_scale;
            let gate_p_off = packed_base;
            let gate_s_off = scale_base;
            let up_p_off = packed_base + tables.gate_packed_bytes;
            let up_s_off = scale_base + tables.gate_scale_bytes;
            let down_p_off = packed_base + tables.gate_packed_bytes + tables.up_packed_bytes;
            let down_s_off = scale_base + tables.gate_scale_bytes + tables.up_scale_bytes;
            let slot_gate = k * 3;
            let slot_up = k * 3 + 1;
            let slot_down = k * 3 + 2;
            let bp = moe_scratch.bulk_expert_packed.as_ref().unwrap() as *const DeviceBuffer<u8>;
            let bs = moe_scratch.bulk_expert_scales.as_ref().unwrap() as *const DeviceBuffer<u8>;
            // SAFETY: bp/bs are read-only views; the GEMV writes only into
            // distinct expert_* scratch fields. Raw ptrs avoid a borrow-checker
            // conflict with the &mut field writes below.
            // gate
            runtime.quantize_nvfp4_input_dptr_device(
                hidden_out, &moe_scratch.slot_in_scale, slot_gate, &mut moe_scratch.quant_expert,
            )?;
            runtime.matvec_nvfp4_prequantized_dptr_bulk_device(
                unsafe { &*bp }, unsafe { &*bs },
                gate_p_off, tables.gate_packed_bytes, gate_s_off, tables.gate_scale_bytes,
                gate.rows, gate.cols,
                &moe_scratch.slot_out_scale, slot_gate,
                &moe_scratch.quant_expert, &mut moe_scratch.expert_gate,
            )?;
            // up
            runtime.quantize_nvfp4_input_dptr_device(
                hidden_out, &moe_scratch.slot_in_scale, slot_up, &mut moe_scratch.quant_expert,
            )?;
            runtime.matvec_nvfp4_prequantized_dptr_bulk_device(
                unsafe { &*bp }, unsafe { &*bs },
                up_p_off, tables.up_packed_bytes, up_s_off, tables.up_scale_bytes,
                up.rows, up.cols,
                &moe_scratch.slot_out_scale, slot_up,
                &moe_scratch.quant_expert, &mut moe_scratch.expert_up,
            )?;
            runtime.geglu_tanh_device(
                &moe_scratch.expert_gate, &moe_scratch.expert_up, &mut moe_scratch.expert_swiglu,
            )?;
            // down
            runtime.quantize_nvfp4_input_dptr_device(
                &moe_scratch.expert_swiglu, &moe_scratch.slot_in_scale, slot_down,
                &mut moe_scratch.quant_expert,
            )?;
            runtime.matvec_nvfp4_prequantized_dptr_bulk_device(
                unsafe { &*bp }, unsafe { &*bs },
                down_p_off, tables.down_packed_bytes, down_s_off, tables.down_scale_bytes,
                down.rows, down.cols,
                &moe_scratch.slot_out_scale, slot_down,
                &moe_scratch.quant_expert, &mut moe_scratch.expert_out,
            )?;
            // accumulate with the routing weight read from the device top-k buffer
            {
                let eo = &moe_scratch.expert_out as *const DeviceBuffer<f32>;
                let ptk = &moe_scratch.packed_topk_device as *const DeviceBuffer<u32>;
                // SAFETY: eo (read) and packed_topk_device (read) are distinct
                // from routed_acc (write).
                runtime.axpy_f32_topk_weight_device(
                    &mut moe_scratch.routed_acc,
                    unsafe { &*eo },
                    unsafe { &*ptk },
                    k,
                )?;
            }
        }
        } // end per-slot fallback (else of batched_decode_moe_enabled)
    }

    // ── Coalesced expert H2D (decode PCIe-saturation fix) ────────────────
    //
    // Per token, the routed experts of every MoE layer must be streamed from
    // host RAM. The old structure issued `top_k × 3` (gate/up/down) tiny
    // (~3.2 MB) H2D transfers through the 4-slot staging pool, with a kernel
    // launch interleaved between each — leaving the PCIe link idle between
    // bursts and capping throughput at ~28 GB/s on a 55 GB/s link.
    //
    // Mirror the PREFILL grouped path's bulk staging: concatenate the active
    // experts' packed+scales bytes for all three projections into one
    // contiguous VRAM buffer with back-to-back `copy_host_u8_to_device_at_offset_async`
    // calls on the transfer stream (no interleaved kernels/syncs → the driver
    // pipelines them into one saturated burst), then run the per-expert GEMVs
    // reading views into that buffer. Same bytes/token, but as one large burst
    // per layer instead of 24 stop-start transfers. Output is bit-identical:
    // same weights, same `nvfp4_prequant` kernel, same per-expert input
    // quantization + accumulation order.
    //
    // Gated on host-resident experts AND the bulk buffers being allocated.
    // VRAM-resident experts (cache) have no H2D and keep the per-expert path.
    let experts_host_resident = moe
        .experts
        .first()
        .map(|e| e.gate_proj.is_host_resident())
        .unwrap_or(false);
    // OPT-IN (default OFF): measured a REGRESSION (36→25 tps, 28→19 GB/s) — the
    // per-expert path already overlaps transfer+compute via the 4-slot staging
    // pool, while this bulk path serializes a whole-layer transfer then compute
    // (+ a cross-layer WAR fence) and never became a single transfer. The decode
    // bottleneck is CPU launch/sync orchestration (~89% cpu_issuing), not the
    // transfer shape; the real fix is a grouped single-launch GEMM. Kept behind
    // a flag for A/B until that lands.
    let bulk_ready = experts_host_resident
        && moe_scratch.bulk_expert_packed.is_some()
        && moe_scratch.bulk_expert_scales.is_some()
        && std::env::var("AEGIS_DECODE_BULK_MOE_ENABLE").is_ok();

    // ── Batched-staged routed-expert decode (default ON for host-resident NVFP4
    //    experts; toggle off with AEGIS_DECODE_BATCHED_STAGED_MOE=0) ──────────
    //
    // The per-slot path (the `else` fallback below) issues, per MoE layer,
    // `top_k × (3 quant + 3 GEMV + 1 geglu + 1 axpy)` = 64 compute launches plus
    // 24 staged H2Ds — ~88 host ops/layer × 36 layers ≈ 3000 host ops/token. The
    // 35B decode is CPU-launch/issue-bound (~93% cpu_issuing, gpu_wait ~1.5ms),
    // so that issuance rate gates the whole token.
    //
    // This path keeps the proven-fast PINNED per-expert H2D (24 small DMAs into a
    // slot-major bulk VRAM buffer — NOT one giant transfer, which regressed) but
    // COLLAPSES the compute into 8 batched launches/layer (3 batched quant + 3
    // batched GEMV with the expert/slot on grid.y + 1 batched geglu + 1 weighted
    // accumulate). Net ~32 host ops/layer vs ~88 → ~1150 ops/token. Output is
    // bit-identical to the per-slot path: same weights, same NVFP4 dequant, same
    // per-slot input quant, same fixed ascending weighted fold (the batched
    // kernels are the exact loop-on-grid.y analogue, already validated on the
    // gpu_driven path). Differs from AEGIS_DECODE_BULK_MOE_ENABLE: that one
    // coalesced only the TRANSFER and kept the 64-launch per-expert compute loop
    // + a full-burst transfer→compute fence (no overlap) → regression. Here the
    // launch count itself drops, which is the actual lever.
    // OPT-IN (default OFF): measured a small REGRESSION on the 35B (36.5→34.4
    // tps, 26.7→25.1 GB/s). Collapsing compute to 8 batched launches/layer did
    // cut launch count, but it forfeits the per-slot staging pool's
    // transfer/compute OVERLAP (it must stage all 8 experts before the batched
    // GEMV can read any slot) — and the decode wall turned out to be the
    // ~27 GB/s pinned-H2D transfer with overlap, not the launch count. Kept
    // behind a flag for A/B; the per-slot path stays default.
    let batched_staged = !gpu_driven
        && experts_host_resident
        && moe_scratch.bulk_expert_packed.is_some()
        && moe_scratch.bulk_expert_scales.is_some()
        && std::env::var("AEGIS_DECODE_BATCHED_STAGED_MOE").map(|v| v == "1").unwrap_or(false);

    if gpu_driven {
        // Routed experts already dispatched by the GPU-driven gather path above.
    } else if batched_staged {
        forward_moe_decode_batched_staged(
            runtime, moe, moe_scratch, hidden_out, active_top_k,
        )?;
    } else if bulk_ready {
        // Build the per-expert/per-projection byte-offset layout host-side and
        // issue all H2Ds in one burst. Layout (contiguous): for each active
        // expert e in router order → gate(e), up(e), down(e).
        // `proj_meta[i] = (gate_off, up_off, down_off)` byte offsets into the
        // bulk buffers; sizes are uniform within a layer.
        // WAR hazard: the bulk buffer is reused across all MoE layers in this
        // token. Block the transfer stream until the PREVIOUS layer's expert
        // GEMVs have finished reading it (skip on the first layer — the event
        // has no recorded workload yet). Without this the burst could clobber
        // the buffer mid-read on the compute stream.
        if moe_scratch.bulk_expert_primed {
            runtime.transfer_wait_event(&moe_scratch.bulk_expert_compute_event)?;
        }
        let bulk_packed = moe_scratch.bulk_expert_packed.as_mut().unwrap();
        let bulk_scales = moe_scratch.bulk_expert_scales.as_mut().unwrap();
        let mut packed_off = 0usize;
        let mut scales_off = 0usize;
        // (gate_p, gate_s, up_p, up_s, down_p, down_s) byte offsets per expert.
        let mut layout: Vec<[usize; 6]> = Vec::with_capacity(active_top_k);
        for i in 0..active_top_k {
            let expert = &moe.experts[moe_scratch.router_top_indices[i]];
            let mut slot = [0usize; 6];
            let projs = [&expert.gate_proj, &expert.up_proj, &expert.down_proj];
            for (pi, proj) in projs.iter().enumerate() {
                let (pb, sb) = proj
                    .host_packed_scales_bytes()
                    .ok_or_else(|| AegisError::InvalidPlan(format!(
                        "decode bulk MoE: expert proj `{}` is not host-resident",
                        proj.name
                    )))??;
                slot[pi * 2] = packed_off;
                slot[pi * 2 + 1] = scales_off;
                runtime.copy_host_u8_to_device_at_offset_async(pb, bulk_packed, packed_off)?;
                runtime.copy_host_u8_to_device_at_offset_async(sb, bulk_scales, scales_off)?;
                packed_off += pb.len();
                scales_off += sb.len();
            }
            layout.push(slot);
        }
        // Account the burst in the shared H2D counter so AEGIS_DECODE_TIMING's
        // MiB/token + GB/s reading stays accurate (these bytes bypass the
        // staging pool, which is what increments the counter on the per-expert
        // path). Total bytes/token is unchanged vs the per-expert path — same
        // experts, same weights — only the transfer shape (one burst vs 24
        // tiny transfers) differs, so MiB/token should match while GB/s rises.
        crate::cuda::staging::STAGING_H2D_BYTES.fetch_add(
            (packed_off + scales_off) as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        // One transfer→compute fence for the whole burst.
        runtime.record_into_transfer(&moe_scratch.bulk_expert_event)?;
        runtime.compute_wait_event(&moe_scratch.bulk_expert_event)?;

        // Per-expert GEMVs reading the resident bulk buffers. Compute is tiny;
        // identical sequence to the per-expert path (quantize input with the
        // linear's own input_scale, then prequantized matvec).
        let bulk_packed = moe_scratch.bulk_expert_packed.as_ref().unwrap();
        let bulk_scales = moe_scratch.bulk_expert_scales.as_ref().unwrap();
        for i in 0..active_top_k {
            let expert = &moe.experts[moe_scratch.router_top_indices[i]];
            let weight = moe_scratch.router_top_weights[i];
            let off = layout[i];
            // gate
            runtime.quantize_nvfp4_input_device(
                hidden_out, expert.gate_proj.input_scale, &mut moe_scratch.quant_expert,
            )?;
            runtime.matvec_nvfp4_prequantized_bulk_views_device(
                bulk_packed, bulk_scales,
                off[0], expert.gate_proj.packed_bytes,
                off[1], expert.gate_proj.scale_bytes,
                expert.gate_proj.rows, expert.gate_proj.cols,
                expert.gate_proj.output_scale,
                &moe_scratch.quant_expert, &mut moe_scratch.expert_gate,
            )?;
            // up
            runtime.quantize_nvfp4_input_device(
                hidden_out, expert.up_proj.input_scale, &mut moe_scratch.quant_expert,
            )?;
            runtime.matvec_nvfp4_prequantized_bulk_views_device(
                bulk_packed, bulk_scales,
                off[2], expert.up_proj.packed_bytes,
                off[3], expert.up_proj.scale_bytes,
                expert.up_proj.rows, expert.up_proj.cols,
                expert.up_proj.output_scale,
                &moe_scratch.quant_expert, &mut moe_scratch.expert_up,
            )?;
            runtime.geglu_tanh_device(
                &moe_scratch.expert_gate, &moe_scratch.expert_up, &mut moe_scratch.expert_swiglu,
            )?;
            // down
            runtime.quantize_nvfp4_input_device(
                &moe_scratch.expert_swiglu, expert.down_proj.input_scale, &mut moe_scratch.quant_expert,
            )?;
            runtime.matvec_nvfp4_prequantized_bulk_views_device(
                bulk_packed, bulk_scales,
                off[4], expert.down_proj.packed_bytes,
                off[5], expert.down_proj.scale_bytes,
                expert.down_proj.rows, expert.down_proj.cols,
                expert.down_proj.output_scale,
                &moe_scratch.quant_expert, &mut moe_scratch.expert_out,
            )?;
            runtime.axpy_f32_device(weight, &moe_scratch.expert_out, &mut moe_scratch.routed_acc)?;
        }
        // Signal the compute stream is done reading the bulk buffer, so the
        // next layer's burst can safely overwrite it.
        runtime.record_into_compute(&moe_scratch.bulk_expert_compute_event)?;
        moe_scratch.bulk_expert_primed = true;
    } else {
        // ── Once-per-layer gate/up activation quantize (decode CPU-op-count fix) ──
        //
        // gate_proj and up_proj of EVERY routed expert read the SAME input
        // (`hidden_out`, this layer's expert-input hidden) AND — in the Qwen3.x
        // NVFP4 checkpoint — the SAME `input_scale`: verified constant across
        // all 256 experts (e.g. 0.007634) and gate==up within each expert.
        // Since the fp4 input quant is a pure function of (input vector,
        // input_scale), the quantized buffer is BYTE-IDENTICAL for every
        // expert's gate/up GEMV. So quantize `hidden_out` ONCE per layer into a
        // persistent `quant_gate_up` buffer (the per-expert `quant_expert` can't
        // hold it because each expert's down_proj quant clobbers it), then every
        // expert's gate/up GEMV reads that one buffer. Removes `top_k-1`
        // redundant identical quantize launches per MoE layer (~7×30 = 210
        // fewer/token on the 35B). Bit-identical: same bytes feed the same
        // GEMVs in the same order.
        //
        // Guarded: only fires when ALL active experts are host-resident NVFP4
        // (not native-MXFP4, not A16) AND share the gate==up==first-expert
        // input_scale. Any mismatch (other model / checkpoint) falls back to the
        // per-expert shared-quant path — still bit-identical, just not hoisted.
        let layer_gate_up_quant = active_top_k > 0
            && !nvfp4_a16_enabled()
            && (0..active_top_k).all(|i| {
                let e = &moe.experts[moe_scratch.router_top_indices[i]];
                e.gate_proj.is_host_resident()
                    && !e.gate_proj.is_host_resident_with_native_mxfp4()
                    && e.gate_proj.input_scale == e.up_proj.input_scale
                    && e.gate_proj.input_scale
                        == moe.experts[moe_scratch.router_top_indices[0]].gate_proj.input_scale
            });
        if layer_gate_up_quant {
            let scale = moe.experts[moe_scratch.router_top_indices[0]].gate_proj.input_scale;
            runtime.quantize_nvfp4_input_device(
                hidden_out, scale, &mut moe_scratch.quant_gate_up,
            )?;
        }
        for i in 0..active_top_k {
            let expert_idx = moe_scratch.router_top_indices[i];
            let weight = moe_scratch.router_top_weights[i];
            let expert = &moe.experts[expert_idx];
            // Per-expert fallback gate/up shared-quant (when the once-per-layer
            // hoist above didn't fire): gate_proj and up_proj share input +
            // input_scale, so quantize once per expert into `quant_expert`.
            let gate_up_share_quant = !layer_gate_up_quant
                && expert.gate_proj.is_host_resident()
                && !nvfp4_a16_enabled()
                && !expert.gate_proj.is_host_resident_with_native_mxfp4()
                && expert.gate_proj.input_scale == expert.up_proj.input_scale;
            if layer_gate_up_quant {
                // Both gate/up GEMVs read the once-per-layer quantized input.
                let staging_gate =
                    if staging_ptr.is_null() { None } else { Some(unsafe { &mut *staging_ptr }) };
                matvec_nvfp4_prepared_device_reuse(
                    runtime, &expert.gate_proj, hidden_out, &moe_scratch.quant_gate_up,
                    &mut moe_scratch.mxfp4_expert, false, &mut moe_scratch.expert_gate,
                    staging_gate,
                )?;
                let staging_up =
                    if staging_ptr.is_null() { None } else { Some(unsafe { &mut *staging_ptr }) };
                matvec_nvfp4_prepared_device_reuse(
                    runtime, &expert.up_proj, hidden_out, &moe_scratch.quant_gate_up,
                    &mut moe_scratch.mxfp4_expert, false, &mut moe_scratch.expert_up,
                    staging_up,
                )?;
            } else if gate_up_share_quant {
                // Quantize hidden_out ONCE with the shared gate/up scale into
                // quant_expert, then run both GEMVs reading that buffer.
                runtime.quantize_nvfp4_input_device(
                    hidden_out, expert.gate_proj.input_scale, &mut moe_scratch.quant_expert,
                )?;
                let staging_gate =
                    if staging_ptr.is_null() { None } else { Some(unsafe { &mut *staging_ptr }) };
                matvec_nvfp4_prepared_device_reuse(
                    runtime, &expert.gate_proj, hidden_out, &moe_scratch.quant_expert,
                    &mut moe_scratch.mxfp4_expert, false, &mut moe_scratch.expert_gate,
                    staging_gate,
                )?;
                let staging_up =
                    if staging_ptr.is_null() { None } else { Some(unsafe { &mut *staging_ptr }) };
                matvec_nvfp4_prepared_device_reuse(
                    runtime, &expert.up_proj, hidden_out, &moe_scratch.quant_expert,
                    &mut moe_scratch.mxfp4_expert, false, &mut moe_scratch.expert_up,
                    staging_up,
                )?;
            } else {
                matvec_nvfp4_device_with_scratch(
                    runtime, &expert.gate_proj, hidden_out,
                    &mut moe_scratch.quant_expert, &mut moe_scratch.mxfp4_expert,
                    &mut moe_scratch.expert_gate,
                    if staging_ptr.is_null() { None } else { Some(unsafe { &mut *staging_ptr }) },
                )?;
                matvec_nvfp4_device_with_scratch(
                    runtime, &expert.up_proj, hidden_out,
                    &mut moe_scratch.quant_expert, &mut moe_scratch.mxfp4_expert,
                    &mut moe_scratch.expert_up,
                    if staging_ptr.is_null() { None } else { Some(unsafe { &mut *staging_ptr }) },
                )?;
            }
            runtime.geglu_tanh_device(&moe_scratch.expert_gate, &moe_scratch.expert_up, &mut moe_scratch.expert_swiglu)?;
            matvec_nvfp4_device_with_scratch(
                runtime, &expert.down_proj, &moe_scratch.expert_swiglu,
                &mut moe_scratch.quant_expert, &mut moe_scratch.mxfp4_expert,
                &mut moe_scratch.expert_out,
                if staging_ptr.is_null() { None } else { Some(unsafe { &mut *staging_ptr }) },
            )?;
            runtime.axpy_f32_device(weight, &moe_scratch.expert_out, &mut moe_scratch.routed_acc)?;
        }
    }

    // Step 7: post_feedforward_layernorm_2(routed_acc) → expert_out (stream2)
    if let Some(ref norm2) = layer.post_feedforward_layernorm_2 {
        runtime.rms_norm_device(&moe_scratch.routed_acc, norm2, rms_norm_eps, &mut moe_scratch.expert_out)?;
    } else {
        runtime.copy_f32_device(&moe_scratch.routed_acc, &mut moe_scratch.expert_out)?;
    }

    // Step 8: combined = stream1 (post_normed) + stream2 (expert_out) → moe_acc
    runtime.add_device(post_normed, &moe_scratch.expert_out, &mut moe_scratch.moe_acc)?;

    // Step 9: post_feedforward_layernorm(combined) → hidden_out
    if let Some(ref final_norm) = layer.post_mlp_sublayer_norm {
        runtime.rms_norm_device(&moe_scratch.moe_acc, final_norm, rms_norm_eps, hidden_out)?;
    } else {
        runtime.copy_f32_device(&moe_scratch.moe_acc, hidden_out)?;
    }

    // Step 10: hidden_out += residual  (in-place residual add)
    runtime.add_inplace_device(hidden_out, residual)?;

    // Step 11: hidden_out *= layer_scalar
    if let Some(scalar) = layer.layer_scalar {
        runtime.scale_f32_device(scalar, hidden_out)?;
    }

    Ok(())
}

/// Batched-staged routed-expert decode for host-resident NVFP4 experts.
///
/// Streams the `active_top_k` active experts' gate/up/down NVFP4 weights from
/// the pinned host arena into a SLOT-MAJOR bulk VRAM buffer via per-expert async
/// H2Ds (the proven-fast pinned DMA path — small back-to-back transfers, NOT one
/// giant transfer), then runs the routed-expert FFN as 8 BATCHED launches (slot
/// on grid.y) instead of the 64-launch per-slot loop. Math is bit-identical to
/// the per-slot path: same weights, same NVFP4 input-quant + dequant, same fixed
/// ascending weighted accumulation (the batched kernels are the validated
/// loop-on-grid.y analogue used by the gpu_driven path).
///
/// `hidden_out` holds the (post pre_ffn_norm_2) routed-expert input. On return
/// `moe_scratch.routed_acc = Σ_k w[k] · expert_k(hidden_out)`.
fn forward_moe_decode_batched_staged(
    runtime: &CudaRuntime,
    moe: &CudaMoE,
    moe_scratch: &mut CudaMoEScratch,
    hidden_out: &DeviceBuffer<f32>,
    active_top_k: usize,
) -> Result<()> {
    if active_top_k == 0 {
        return Ok(());
    }
    // Uniform per-expert projection shapes (all experts share gate/up/down dims).
    let e0 = &moe.experts[0];
    let gate = &e0.gate_proj;
    let up = &e0.up_proj;
    let down = &e0.down_proj;
    let gate_pb = gate.packed_bytes;
    let up_pb = up.packed_bytes;
    let down_pb = down.packed_bytes;
    let gate_sb = gate.scale_bytes;
    let up_sb = up.scale_bytes;
    let down_sb = down.scale_bytes;
    let per_slot_packed = gate_pb + up_pb + down_pb;
    let per_slot_scale = gate_sb + up_sb + down_sb;

    // ── 1) Stage each active expert's gate/up/down into the slot-major bulk
    //       buffer + collect its NVFP4 input/output scales (slot-major:
    //       slot k → [gate, up, down]). WAR fence: the bulk buffer + slot-scale
    //       arrays are reused across MoE layers in this token, so block the
    //       transfer stream until the PREVIOUS layer's batched GEMVs finished
    //       reading them (no-op on the first layer). ──────────────────────────
    if moe_scratch.bulk_expert_primed {
        runtime.transfer_wait_event(&moe_scratch.bulk_expert_compute_event)?;
    }
    let mut in_scales: Vec<f32> = Vec::with_capacity(active_top_k * 3);
    let mut out_scales: Vec<f32> = Vec::with_capacity(active_top_k * 3);
    let mut h2d_bytes = 0usize;
    {
        let bulk_packed = moe_scratch.bulk_expert_packed.as_mut().unwrap();
        let bulk_scales = moe_scratch.bulk_expert_scales.as_mut().unwrap();
        for i in 0..active_top_k {
            let expert = &moe.experts[moe_scratch.router_top_indices[i]];
            let packed_base = i * per_slot_packed;
            let scale_base = i * per_slot_scale;
            let projs = [&expert.gate_proj, &expert.up_proj, &expert.down_proj];
            let mut p_off = packed_base;
            let mut s_off = scale_base;
            for proj in projs.iter() {
                let (pb, sb) = proj
                    .host_packed_scales_bytes()
                    .ok_or_else(|| AegisError::InvalidPlan(format!(
                        "batched-staged MoE: expert proj `{}` is not host-resident",
                        proj.name
                    )))??;
                runtime.copy_host_u8_to_device_at_offset_async(pb, bulk_packed, p_off)?;
                runtime.copy_host_u8_to_device_at_offset_async(sb, bulk_scales, s_off)?;
                p_off += pb.len();
                s_off += sb.len();
                h2d_bytes += pb.len() + sb.len();
                in_scales.push(proj.input_scale);
                out_scales.push(proj.output_scale);
            }
        }
    }
    // Account the streamed bytes in the shared H2D counter (these bypass the
    // staging pool which normally increments it) so AEGIS_DECODE_TIMING's
    // MiB/token + GB/s stay accurate. Same bytes/token as the per-slot path.
    crate::cuda::staging::STAGING_H2D_BYTES
        .fetch_add(h2d_bytes as u64, std::sync::atomic::Ordering::Relaxed);

    // Upload the per-slot NVFP4 scales (tiny: top_k*3 f32 each) on the transfer
    // stream so they land before the compute fence.
    runtime.upload_f32_slice_to_device_async(&in_scales, &mut moe_scratch.slot_in_scale)?;
    runtime.upload_f32_slice_to_device_async(&out_scales, &mut moe_scratch.slot_out_scale)?;

    // One transfer→compute fence for the whole layer's staging.
    runtime.record_into_transfer(&moe_scratch.bulk_expert_event)?;
    runtime.compute_wait_event(&moe_scratch.bulk_expert_event)?;

    // ── 2) Batched compute (mirrors the gpu_driven batched branch) ───────────
    let tk = active_top_k;
    let max_top_k = moe_scratch.packed_topk_device.len() / 2;
    let inter_stride = moe_scratch.expert_gate_b.len() / max_top_k; // max_expert_intermediate
    let quant_stride = moe_scratch.quant_b.len() / max_top_k; // max_input
    let hidden_stride = down.rows; // = hidden_size
    let bp = moe_scratch.bulk_expert_packed.as_ref().unwrap() as *const DeviceBuffer<u8>;
    let bs = moe_scratch.bulk_expert_scales.as_ref().unwrap() as *const DeviceBuffer<u8>;
    // GATE
    runtime.quantize_nvfp4_input_batched_dptr_device(
        hidden_out, 0, gate.cols, quant_stride,
        &moe_scratch.slot_in_scale, 0, tk, &mut moe_scratch.quant_b,
    )?;
    runtime.matvec_nvfp4_prequantized_batched_dptr_device(
        unsafe { &*bp }, unsafe { &*bs }, per_slot_packed, per_slot_scale,
        0, 0,
        &moe_scratch.quant_b, quant_stride, gate.rows, gate.cols,
        &moe_scratch.slot_out_scale, 0, inter_stride, tk, &mut moe_scratch.expert_gate_b,
    )?;
    // UP
    runtime.quantize_nvfp4_input_batched_dptr_device(
        hidden_out, 0, up.cols, quant_stride,
        &moe_scratch.slot_in_scale, 1, tk, &mut moe_scratch.quant_b,
    )?;
    runtime.matvec_nvfp4_prequantized_batched_dptr_device(
        unsafe { &*bp }, unsafe { &*bs }, per_slot_packed, per_slot_scale,
        gate_pb, gate_sb,
        &moe_scratch.quant_b, quant_stride, up.rows, up.cols,
        &moe_scratch.slot_out_scale, 1, inter_stride, tk, &mut moe_scratch.expert_up_b,
    )?;
    // GEGLU (batched, strided over slots) — matches the per-slot path's
    // geglu_tanh_device exactly (loop-on-grid.y analogue).
    {
        let eg = &moe_scratch.expert_gate_b as *const DeviceBuffer<f32>;
        let eu = &moe_scratch.expert_up_b as *const DeviceBuffer<f32>;
        // SAFETY: eg/eu (read) are distinct from expert_swiglu_b (write).
        runtime.geglu_tanh_batched_slots_device(
            unsafe { &*eg }, unsafe { &*eu }, gate.rows, inter_stride, tk,
            &mut moe_scratch.expert_swiglu_b,
        )?;
    }
    // DOWN
    {
        let es = &moe_scratch.expert_swiglu_b as *const DeviceBuffer<f32>;
        // SAFETY: es (read) is distinct from quant_b (write).
        runtime.quantize_nvfp4_input_batched_dptr_device(
            unsafe { &*es }, inter_stride, down.cols, quant_stride,
            &moe_scratch.slot_in_scale, 2, tk, &mut moe_scratch.quant_b,
        )?;
    }
    runtime.matvec_nvfp4_prequantized_batched_dptr_device(
        unsafe { &*bp }, unsafe { &*bs }, per_slot_packed, per_slot_scale,
        gate_pb + up_pb, gate_sb + up_sb,
        &moe_scratch.quant_b, quant_stride, down.rows, down.cols,
        &moe_scratch.slot_out_scale, 2, hidden_stride, tk, &mut moe_scratch.expert_out_b,
    )?;
    // WEIGHTED ACCUMULATE: routed_acc = Σ_k w[k]·expert_out[k] (fixed ascending
    // fold; routed_acc was zeroed by the caller — this overwrites it).
    {
        let eo = &moe_scratch.expert_out_b as *const DeviceBuffer<f32>;
        let ptk = &moe_scratch.packed_topk_device as *const DeviceBuffer<u32>;
        // SAFETY: eo/ptk (read) are distinct from routed_acc (write).
        runtime.moe_weighted_accumulate_device(
            &mut moe_scratch.routed_acc, unsafe { &*eo }, hidden_stride,
            unsafe { &*ptk }, tk, down.rows,
        )?;
    }
    // Signal the compute stream is done reading the bulk buffer + slot scales so
    // the next layer's staging burst can safely overwrite them.
    runtime.record_into_compute(&moe_scratch.bulk_expert_compute_event)?;
    moe_scratch.bulk_expert_primed = true;
    Ok(())
}

/// Gemma 4 routing post-processing (matches Gemma4TextRouter.forward):
///   probs  = softmax(logits)
///   topk_w, topk_i = topk(probs, k)
///   topk_w /= sum(topk_w)                       # renormalize so top-k weights sum to 1
///   topk_w *= per_expert_scale[topk_i]           # if provided
///
/// Decode no longer calls this — replaced by the GPU async-overlap router
/// (`router_softmax_topk_packed_device` + pinned dtoh). Kept for tests and
/// any non-decode fallback path that may still want the CPU implementation.
#[allow(dead_code)]
fn softmax_top_k_normalized_into(
    logits: &[f32],
    top_k: usize,
    per_expert_scale: Option<&[f32]>,
    probs_buf: &mut Vec<f32>,
    indexed_buf: &mut Vec<(usize, f32)>,
    out_indices: &mut Vec<usize>,
    out_weights: &mut Vec<f32>,
) {
    softmax_top_k_into(logits, top_k, probs_buf, indexed_buf, out_indices, out_weights);
    let sum: f32 = out_weights.iter().sum();
    if sum > 0.0 {
        for w in out_weights.iter_mut() {
            *w /= sum;
        }
    }
    if let Some(pes) = per_expert_scale {
        for (i, w) in out_indices.iter().zip(out_weights.iter_mut()) {
            if let Some(s) = pes.get(*i) {
                *w *= *s;
            }
        }
    }
}

/// Softmax over `logits`, write top-k `(indices, weights)` into the
/// provided buffers in descending weight order. `probs_buf` and
/// `indexed_buf` are scratch reused across calls; `out_indices` /
/// `out_weights` are cleared and re-extended.
fn softmax_top_k_into(
    logits: &[f32],
    top_k: usize,
    probs_buf: &mut Vec<f32>,
    indexed_buf: &mut Vec<(usize, f32)>,
    out_indices: &mut Vec<usize>,
    out_weights: &mut Vec<f32>,
) {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    probs_buf.clear();
    probs_buf.extend(logits.iter().map(|&x| (x - max).exp()));
    let sum: f32 = probs_buf.iter().sum();
    if sum > 0.0 {
        for p in probs_buf.iter_mut() {
            *p /= sum;
        }
    }

    let k = top_k.min(probs_buf.len());
    indexed_buf.clear();
    indexed_buf.extend(probs_buf.iter().cloned().enumerate());
    if k > 0 {
        // Partial sort: place the k largest at the front.
        indexed_buf.select_nth_unstable_by(k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    let top = &mut indexed_buf[..k];
    top.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    out_indices.clear();
    out_weights.clear();
    out_indices.extend(top.iter().map(|(i, _)| *i));
    out_weights.extend(top.iter().map(|(_, w)| *w));
}

/// Allocating wrapper kept for tests and for the prefill MoE path
/// that hasn't been converted to scratch-based yet. The hot decode
/// path goes through `softmax_top_k_into` directly.
#[cfg(test)]
fn softmax_top_k(logits: &[f32], top_k: usize) -> (Vec<usize>, Vec<f32>) {
    let mut probs_buf = Vec::with_capacity(logits.len());
    let mut indexed_buf = Vec::with_capacity(logits.len());
    let mut indices = Vec::with_capacity(top_k);
    let mut weights = Vec::with_capacity(top_k);
    softmax_top_k_into(
        logits,
        top_k,
        &mut probs_buf,
        &mut indexed_buf,
        &mut indices,
        &mut weights,
    );
    (indices, weights)
}

#[cfg(test)]
mod tests {
    use super::softmax_top_k;

    #[test]
    fn softmax_top_k_returns_highest_prob_experts() {
        // Logits: expert 2 is highest, then expert 0, then expert 1.
        let logits = vec![1.0f32, 0.0, 3.0, -1.0];
        let (indices, weights) = softmax_top_k(&logits, 2);
        assert_eq!(indices.len(), 2);
        assert_eq!(weights.len(), 2);
        // Expert 2 should be first (highest logit → highest weight).
        assert_eq!(indices[0], 2);
        assert_eq!(indices[1], 0);
        // Weights must be positive and sum to ≤ 1.0 (top-2 of 4).
        assert!(weights[0] > weights[1]);
        assert!(weights.iter().all(|&w| w > 0.0));
        let wsum: f32 = weights.iter().sum();
        assert!(wsum <= 1.0 + 1e-5);
    }

    #[test]
    fn softmax_top_k_handles_k_equals_len() {
        let logits = vec![0.0f32, 1.0, 2.0];
        let (indices, weights) = softmax_top_k(&logits, 3);
        assert_eq!(indices.len(), 3);
        let wsum: f32 = weights.iter().sum();
        assert!((wsum - 1.0).abs() < 1e-5);
    }
}

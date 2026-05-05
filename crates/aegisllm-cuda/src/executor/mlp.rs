use super::linear_ops::{
    matvec_cuda_linear_with_scratch, matvec_nvfp4_device_with_scratch,
    matvec_nvfp4_prepared_device_reuse, native_mxfp4_enabled, prepare_nvfp4_input,
};
use super::state::{CudaLayer, CudaMoE, CudaMoEScratch, CudaScratch};
use crate::cuda::CudaRuntime;
use crate::cuda::staging::LinearStagingPool;
use aegisllm_base::error::Result;

pub(super) fn forward_mlp_device(
    runtime: &CudaRuntime,
    layer: &CudaLayer,
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
    // For decode (M=1), native MXFP4 MATVEC is strongly preferred over CUTLASS.
    // CUTLASS tiles are 128×128, so M=1 uses <1% of each tile. Native MXFP4 GEMV
    // with hardware mxf4 MMA instructions is purpose-built for this shape.
    let use_native_gate_up = native_mxfp4_enabled(runtime, &layer.gate_proj)
        && native_mxfp4_enabled(runtime, &layer.up_proj);
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
            &layer.gate_proj,
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
            &layer.up_proj,
            &scratch.post_normed,
            &scratch.quant_hidden,
            &mut scratch.mxfp4_hidden,
            mxfp4_valid,
            &mut scratch.up,
            scratch.staging_pool.as_deref_mut(),
        )?;
    } else if runtime.cutlass_nvfp4_inference_enabled_for(&layer.gate_proj)
        && runtime.cutlass_nvfp4_inference_enabled_for(&layer.up_proj)
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
            layer.gate_proj.cols,
            &mut scratch.cutlass_payload,
            &mut scratch.cutlass_scales,
        )?;
        runtime.matmul_cutlass_nvfp4_prepacked_prefill_device(
            &layer.gate_proj,
            &scratch.cutlass_payload,
            &scratch.cutlass_scales,
            1,
            &mut scratch.cutlass_workspace,
            &mut scratch.gate,
        )?;
        runtime.matmul_cutlass_nvfp4_prepacked_prefill_device(
            &layer.up_proj,
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
            layer.gate_proj.input_scale,
            &mut scratch.post_normed,
            &mut scratch.quant_hidden,
        )?;
        let mut quant_scale = Some(layer.gate_proj.input_scale);
        let mxfp4_valid = matvec_nvfp4_prepared_device_reuse(
            runtime,
            &layer.gate_proj,
            &scratch.post_normed,
            &scratch.quant_hidden,
            &mut scratch.mxfp4_hidden,
            false,
            &mut scratch.gate,
            scratch.staging_pool.as_deref_mut(),
        )?;
        prepare_nvfp4_input(
            runtime,
            &layer.up_proj,
            &scratch.post_normed,
            &mut quant_scale,
            &mut scratch.quant_hidden,
        )?;
        matvec_nvfp4_prepared_device_reuse(
            runtime,
            &layer.up_proj,
            &scratch.post_normed,
            &scratch.quant_hidden,
            &mut scratch.mxfp4_hidden,
            mxfp4_valid,
            &mut scratch.up,
            scratch.staging_pool.as_deref_mut(),
        )?;
    }
    let use_native_down = native_mxfp4_enabled(runtime, &layer.down_proj);
    if use_native_down {
        runtime.swiglu_device(&scratch.gate, &scratch.up, &mut scratch.swiglu)?;
        matvec_nvfp4_device_with_scratch(
            runtime,
            &layer.down_proj,
            &scratch.swiglu,
            &mut scratch.quant_intermediate,
            &mut scratch.mxfp4_intermediate,
            &mut scratch.mlp_out,
            scratch.staging_pool.as_deref_mut(),
        )?;
    } else if runtime.cutlass_nvfp4_inference_enabled_for(&layer.down_proj) {
        runtime.swiglu_quantize_cutlass_nvfp4_activation_device(
            &scratch.gate,
            &scratch.up,
            1,
            layer.down_proj.cols,
            &mut scratch.cutlass_payload,
            &mut scratch.cutlass_scales,
        )?;
        runtime.matmul_cutlass_nvfp4_prepacked_prefill_device(
            &layer.down_proj,
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
            &layer.down_proj,
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
    if let Some(scalar) = layer.layer_scalar {
        runtime.scale_f32_device(scalar, &mut scratch.hidden_out)?;
    }
    Ok(())
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
            // hidden = rms_norm(residual) * router.scale  (single fused rms_norm with weight)
            runtime.rms_norm_device(
                residual,
                input_scale,
                rms_norm_eps,
                &mut moe_scratch.router_input_scratch,
            )?;
            // hidden *= 1 / sqrt(hidden_size)
            let scalar_root_size = (hidden_size as f32).powf(-0.5);
            runtime.scale_f32_device(scalar_root_size, &mut moe_scratch.router_input_scratch)?;
            &moe_scratch.router_input_scratch
        }
        None => residual,
    };
    runtime.matvec_bf16_reference_device(&moe.router, router_input, &mut moe_scratch.router_logits)?;
    let logits = runtime.download_f32(&moe_scratch.router_logits)?;
    let (top_k_indices, top_k_weights) = softmax_top_k_normalized(
        &logits,
        moe.top_k,
        moe.router_per_expert_scale_host.as_deref(),
    );

    // Step 5: pre_feedforward_layernorm_2(residual) → hidden_out (expert input)
    // We store this in hidden_out temporarily; it will be overwritten later.
    if let Some(ref norm2) = layer.pre_feedforward_layernorm_2 {
        runtime.rms_norm_device(residual, norm2, rms_norm_eps, hidden_out)?;
    } else {
        // No separate expert pre-norm: copy post_normed into hidden_out.
        runtime.copy_f32_device(post_normed, hidden_out)?;
    }

    // Steps 6-7: Accumulate routed experts on expert_input (hidden_out), then post-norm.
    let staging_ptr: *mut LinearStagingPool =
        staging.map_or(std::ptr::null_mut(), |p| p as *mut _);

    runtime.zero_f32_device(&mut moe_scratch.moe_acc)?;
    for (expert_idx, weight) in top_k_indices.iter().zip(top_k_weights.iter()) {
        let expert = &moe.experts[*expert_idx];
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
        // Gemma 4 uses gelu_pytorch_tanh as MoE expert activation, not silu/SwiGLU.
        runtime.geglu_tanh_device(&moe_scratch.expert_gate, &moe_scratch.expert_up, &mut moe_scratch.expert_swiglu)?;
        matvec_nvfp4_device_with_scratch(
            runtime, &expert.down_proj, &moe_scratch.expert_swiglu,
            &mut moe_scratch.quant_expert, &mut moe_scratch.mxfp4_expert,
            &mut moe_scratch.expert_out,
            if staging_ptr.is_null() { None } else { Some(unsafe { &mut *staging_ptr }) },
        )?;
        runtime.axpy_f32_device(*weight, &moe_scratch.expert_out, &mut moe_scratch.moe_acc)?;
    }

    // Step 7: post_feedforward_layernorm_2(moe_acc) → expert_out (stream2)
    if let Some(ref norm2) = layer.post_feedforward_layernorm_2 {
        runtime.rms_norm_device(&moe_scratch.moe_acc, norm2, rms_norm_eps, &mut moe_scratch.expert_out)?;
    } else {
        runtime.copy_f32_device(&moe_scratch.moe_acc, &mut moe_scratch.expert_out)?;
    }

    // Steps 2-3: Shared MLP on post_normed (step 1 output) → result in moe_acc
    if let Some(ref shared) = moe.shared_expert {
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
        // Gemma 4 shared MLP also uses gelu_pytorch_tanh activation.
        runtime.geglu_tanh_device(&moe_scratch.expert_gate, &moe_scratch.expert_up, &mut moe_scratch.expert_swiglu)?;
        matvec_cuda_linear_with_scratch(
            runtime, &shared.down_proj, &moe_scratch.expert_swiglu,
            &mut moe_scratch.quant_expert, &mut moe_scratch.mxfp4_expert,
            &mut moe_scratch.moe_acc,
            if staging_ptr.is_null() { None } else { Some(unsafe { &mut *staging_ptr }) },
        )?;
    } else {
        runtime.zero_f32_device(&mut moe_scratch.moe_acc)?;
    }

    // Step 3: post_feedforward_layernorm_1(moe_acc=shared_out) → post_normed (stream1)
    // Overwrite post_normed — shared MLP input is no longer needed.
    if let Some(ref norm1) = layer.post_feedforward_layernorm_1 {
        runtime.rms_norm_device(&moe_scratch.moe_acc, norm1, rms_norm_eps, post_normed)?;
    } else {
        runtime.copy_f32_device(&moe_scratch.moe_acc, post_normed)?;
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

/// Gemma 4 routing post-processing (matches Gemma4TextRouter.forward):
///   probs  = softmax(logits)
///   topk_w, topk_i = topk(probs, k)
///   topk_w /= sum(topk_w)                       # renormalize so top-k weights sum to 1
///   topk_w *= per_expert_scale[topk_i]           # if provided
fn softmax_top_k_normalized(
    logits: &[f32],
    top_k: usize,
    per_expert_scale: Option<&[f32]>,
) -> (Vec<usize>, Vec<f32>) {
    let (mut indices, mut weights) = softmax_top_k(logits, top_k);
    let sum: f32 = weights.iter().sum();
    if sum > 0.0 {
        for w in weights.iter_mut() {
            *w /= sum;
        }
    }
    if let Some(pes) = per_expert_scale {
        for (i, w) in indices.iter().zip(weights.iter_mut()) {
            if let Some(s) = pes.get(*i) {
                *w *= *s;
            }
        }
    }
    (indices, weights)
}

/// Softmax over `logits`, return top-k `(indices, weights)` in descending weight order.
fn softmax_top_k(logits: &[f32], top_k: usize) -> (Vec<usize>, Vec<f32>) {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

    let k = top_k.min(probs.len());
    let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    // Partial sort: place the k largest at the front
    indexed.select_nth_unstable_by(k - 1, |a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut top = indexed[..k].to_vec();
    top.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let indices: Vec<usize> = top.iter().map(|(i, _)| *i).collect();
    let weights: Vec<f32> = top.iter().map(|(_, w)| *w).collect();
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

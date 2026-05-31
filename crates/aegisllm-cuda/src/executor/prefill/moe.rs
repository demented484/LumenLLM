//! Chunked MoE prefill — batched 3-stream forward (shared MLP + routed experts).
//!
//! Architecture mirrors `forward_moe_decode_device` but works over a chunk of
//! `batch` tokens at once. Routed experts use **grouped GEMM**: per active expert
//! we gather the rows of tokens routing to it, run a batched expert matmul, and
//! scatter-add results into the accumulator weighted by the per-token routing
//! weight. This is the standard MoE prefill recipe (vLLM `FusedMoE`,
//! transformers `Gemma4TextExperts`).
//!
//! Buffers reuse the prefill scratch where possible. Sequence:
//!   1. `input_normed = rms_norm(residual, pre_ffn_norm)`               batched
//!   2. shared MLP (BF16 batched): gate, up, geglu_tanh, down → `gather_out`
//!   3. `stream1 = rms_norm(gather_out, post_ffn_norm_1)`               batched
//!   4. `expert_input = rms_norm(residual, pre_ffn_norm_2)`             batched
//!   5. `router_input = rms_norm(residual, router.scale) * root_size`   batched
//!   6. `router_logits = matmul(router, router_input)`                  batched
//!   7. host-side per-token softmax + topk + per_expert_scale + renorm
//!   8. per active expert: gather → batched gate/up/geglu/down → scatter-add
//!   9. `moe_acc = rms_norm(moe_acc, post_ffn_norm_2)`                  batched in-place
//!  10. `moe_acc += stream1`                                            batched
//!  11. `gather_out = rms_norm(moe_acc, post_ffn_norm)`                 batched
//!  12. `prefill.hidden += gather_out`                                  batched
//!  13. `prefill.hidden *= layer_scalar`                                if present

use crate::cuda::staging::LinearStagingPool;
use crate::cuda::{CudaRuntime, DeviceBuffer};
use crate::executor::linear_ops::matvec_nvfp4_batched_device_with_scratch;
use crate::executor::state::{
    CudaLayer, CudaMoE, CudaPrefillScratch, CudaPrefillStageTimings,
};
use aegisllm_base::error::{AegisError, Result};

#[allow(clippy::too_many_arguments)]
pub(super) fn forward_moe_prefill_chunk_device(
    runtime: &CudaRuntime,
    layer: &CudaLayer,
    moe: &CudaMoE,
    prefill: &mut CudaPrefillScratch,
    batch: usize,
    hidden_size: usize,
    rms_norm_eps: f32,
    staging_ptr: *mut LinearStagingPool,
    _timings: &mut CudaPrefillStageTimings,
) -> Result<()> {
    let profile = std::env::var("AEGIS_MOE_PROFILE").is_ok();
    let mark = |label: &'static str| -> (&'static str, std::time::Instant) {
        if profile {
            let _ = runtime.synchronize();
            (label, std::time::Instant::now())
        } else { (label, std::time::Instant::now()) }
    };
    let report = |checkpoint: (&'static str, std::time::Instant), tag: &'static str| {
        if profile {
            let _ = runtime.synchronize();
            let elapsed_us = checkpoint.1.elapsed().as_micros();
            eprintln!("[MOE_PROF {}->{}] {}us", checkpoint.0, tag, elapsed_us);
        }
    };
    let batch_hidden = batch
        .checked_mul(hidden_size)
        .ok_or_else(|| AegisError::InvalidPlan("MoE prefill batch_hidden overflow".into()))?;

    // SAFETY: we hold &mut prefill but need disjoint &mut access to
    // prefill.{hidden,input_normed,gate,up,…} and prefill.moe.{router_input,…}.
    // Splitting through raw pointers keeps the borrow checker happy without
    // changing CudaPrefillScratch field ordering.
    let pf_ptr: *mut CudaPrefillScratch = prefill as *mut _;
    let moe_scratch = prefill
        .moe
        .as_deref_mut()
        .ok_or_else(|| AegisError::InvalidPlan("MoE prefill scratch not allocated".into()))?;
    let pf = unsafe { &mut *pf_ptr };

    // ── Step 1: input_normed = rms_norm(residual, pre_ffn_norm) ─────────────
    runtime.rms_norm_batched_device(
        &pf.hidden,
        &layer.post_attention_norm_weight,
        batch,
        rms_norm_eps,
        &mut pf.input_normed,
    )?;

    // ── Step 2: shared MLP (BF16 or load-time-quantized MXFP4, batched) ────
    // Reuses `gather_intermediate` (cs × max_intermediate) for gate, then
    // `gather_swiglu` for up. After geglu we run down → `gather_out`.
    let shared = moe.shared_expert.as_ref().ok_or_else(|| {
        AegisError::InvalidPlan(
            "Gemma 4 MoE prefill requires a shared MLP (mlp.gate/up/down_proj.weight)".into(),
        )
    })?;
    let intermediate = shared.gate_proj.rows();

    let cp_shared = mark("");
    use crate::executor::state::CudaLinear as CL;
    match (&shared.gate_proj, &shared.up_proj, &shared.down_proj) {
        (CL::Bf16(g), CL::Bf16(u), CL::Bf16(d)) => {
            // cuBLASLt BF16 GEMM. Shared MLP weights are force-VRAM at load,
            // so the cublaslt path applies. Falls back to reference if any
            // reason VRAM-residency is unmet.
            if runtime.cublaslt_bf16_enabled_for(g) {
                if let Some(ref fused) = shared.gate_up_fused {
                    // Fused gate+up: one cuBLASLt GEMM produces
                    // `[batch, 2*intermediate]` row-major, then the strided
                    // GeGLU kernel reads gate from the first `intermediate`
                    // floats and up from the next `intermediate`.
                    runtime.matmul_bf16_cublaslt_device(
                        fused, &pf.input_normed, batch,
                        &mut pf.bf16_in_scratch, &mut pf.bf16_out_scratch,
                        &mut moe_scratch.gather_shared_gate_up_fused,
                    )?;
                    runtime.geglu_tanh_strided_device(
                        &moe_scratch.gather_shared_gate_up_fused,
                        batch,
                        intermediate,
                        &mut moe_scratch.gather_swiglu,
                    )?;
                } else {
                    runtime.matmul_bf16_cublaslt_device(
                        g, &pf.input_normed, batch,
                        &mut pf.bf16_in_scratch, &mut pf.bf16_out_scratch,
                        &mut moe_scratch.gather_intermediate,
                    )?;
                    runtime.matmul_bf16_cublaslt_device(
                        u, &pf.input_normed, batch,
                        &mut pf.bf16_in_scratch, &mut pf.bf16_out_scratch,
                        &mut moe_scratch.gather_swiglu,
                    )?;
                    runtime.geglu_tanh_in_place_device(
                        &moe_scratch.gather_intermediate,
                        &mut moe_scratch.gather_swiglu,
                        batch * intermediate,
                    )?;
                }
                runtime.matmul_bf16_cublaslt_device(
                    d, &moe_scratch.gather_swiglu, batch,
                    &mut pf.bf16_in_scratch, &mut pf.bf16_out_scratch,
                    &mut moe_scratch.gather_out,
                )?;
            } else {
                runtime.matmul_bf16_reference_batched_device(
                    g, &pf.input_normed, batch, &mut moe_scratch.gather_intermediate,
                )?;
                runtime.matmul_bf16_reference_batched_device(
                    u, &pf.input_normed, batch, &mut moe_scratch.gather_swiglu,
                )?;
                runtime.geglu_tanh_in_place_device(
                    &moe_scratch.gather_intermediate,
                    &mut moe_scratch.gather_swiglu,
                    batch * intermediate,
                )?;
                runtime.matmul_bf16_reference_batched_device(
                    d, &moe_scratch.gather_swiglu, batch, &mut moe_scratch.gather_out,
                )?;
            }
        }
        (CL::Fp8(g), CL::Fp8(u), CL::Fp8(d)) => {
            // Standalone FP8 shared expert via dequant-to-BF16 + cuBLASLt
            // tensor-core path. Each projection dequants its weight into
            // the shared `pf.fp8_dequant_scratch` (one buffer reused across
            // all four GEMMs in the chunk; safe because each call's
            // weight-dequant precedes its own matmul).
            runtime.matmul_fp8_via_bf16_cublaslt_device(
                g, &mut pf.fp8_dequant_scratch,
                &pf.input_normed, batch,
                &mut pf.bf16_in_scratch, &mut pf.bf16_out_scratch,
                &mut moe_scratch.gather_intermediate,
            )?;
            runtime.matmul_fp8_via_bf16_cublaslt_device(
                u, &mut pf.fp8_dequant_scratch,
                &pf.input_normed, batch,
                &mut pf.bf16_in_scratch, &mut pf.bf16_out_scratch,
                &mut moe_scratch.gather_swiglu,
            )?;
            runtime.geglu_tanh_in_place_device(
                &moe_scratch.gather_intermediate,
                &mut moe_scratch.gather_swiglu,
                batch * intermediate,
            )?;
            runtime.matmul_fp8_via_bf16_cublaslt_device(
                d, &mut pf.fp8_dequant_scratch,
                &moe_scratch.gather_swiglu, batch,
                &mut pf.bf16_in_scratch, &mut pf.bf16_out_scratch,
                &mut moe_scratch.gather_out,
            )?;
        }
        (CL::Nvfp4(g), CL::Nvfp4(u), CL::Nvfp4(d)) => {
            // Qwen3-Next NVFP4 shared expert (the 35B-A3B case). Mirrors the
            // per-token decode shared-MLP path exactly, batched over the chunk:
            // gate/up via the batched NVFP4 GEMM, the layer's configured dense
            // activation (SwiGLU for Qwen `silu`; GeGLU-tanh for Gemma), then
            // down. The shared_gate sigmoid (below, after this match) scales
            // the result per token. Routed experts use their own GeGLU-tanh
            // path (matching decode) — do NOT conflate the two activations.
            let sp = if staging_ptr.is_null() {
                None
            } else {
                Some(unsafe { &mut *staging_ptr })
            };
            matvec_nvfp4_batched_device_with_scratch(
                runtime, g, &pf.input_normed, batch,
                &mut moe_scratch.gather_quant, &mut moe_scratch.gather_mxfp4,
                &mut moe_scratch.gather_intermediate, sp,
            )?;
            let sp = if staging_ptr.is_null() {
                None
            } else {
                Some(unsafe { &mut *staging_ptr })
            };
            matvec_nvfp4_batched_device_with_scratch(
                runtime, u, &pf.input_normed, batch,
                &mut moe_scratch.gather_quant, &mut moe_scratch.gather_mxfp4,
                &mut moe_scratch.gather_swiglu, sp,
            )?;
            // Activation. SwiGLU writes into `gather_intermediate` (silu(gate)*up),
            // GeGLU-tanh writes into `gather_swiglu` (gelu_tanh(gate)*up). Track
            // which buffer holds the activated value so `down` reads the right
            // one — matching the decode `match layer.dense_activation` arms.
            use crate::executor::mlp::DenseActivation;
            let act_in_intermediate = match layer.dense_activation {
                DenseActivation::Swiglu => {
                    runtime.swiglu_inplace_gate_device_len(
                        &mut moe_scratch.gather_intermediate,
                        &moe_scratch.gather_swiglu,
                        batch * intermediate,
                    )?;
                    true
                }
                DenseActivation::GeluTanh => {
                    runtime.geglu_tanh_in_place_device(
                        &moe_scratch.gather_intermediate,
                        &mut moe_scratch.gather_swiglu,
                        batch * intermediate,
                    )?;
                    false
                }
            };
            let sp = if staging_ptr.is_null() {
                None
            } else {
                Some(unsafe { &mut *staging_ptr })
            };
            // SAFETY: the activation buffer (read) and gather_out (write) are
            // distinct fields of moe_scratch; the raw split sidesteps the
            // borrow checker over the shared &mut moe_scratch.
            if act_in_intermediate {
                let act = &moe_scratch.gather_intermediate as *const DeviceBuffer<f32>;
                matvec_nvfp4_batched_device_with_scratch(
                    runtime, d, unsafe { &*act }, batch,
                    &mut moe_scratch.gather_quant, &mut moe_scratch.gather_mxfp4,
                    &mut moe_scratch.gather_out, sp,
                )?;
            } else {
                let act = &moe_scratch.gather_swiglu as *const DeviceBuffer<f32>;
                matvec_nvfp4_batched_device_with_scratch(
                    runtime, d, unsafe { &*act }, batch,
                    &mut moe_scratch.gather_quant, &mut moe_scratch.gather_mxfp4,
                    &mut moe_scratch.gather_out, sp,
                )?;
            }
        }
        _ => return Err(AegisError::InvalidPlan(
            "MoE prefill expects shared expert with all three projections in the same \
             format (BF16, FP8, or NVFP4)".into(),
        )),
    }

    // ── Qwen3-Next shared-expert gate ───────────────────────────────────────
    // Scale each token's shared-MLP output by `sigmoid(shared_gate · x)`, where
    // x is the (pre-shared-MLP) normed hidden in `pf.input_normed`. Mirrors the
    // decode path (`scale_by_sigmoid_scalar` per token); here the [1, hidden]
    // gate produces a `[batch]` logit vector via the batched matvec and a
    // per-row sigmoid scale broadcasts over the hidden dim. `None` for Gemma.
    if let Some(ref sgate) = shared.shared_gate {
        runtime.matmul_bf16_reference_batched_device(
            sgate,
            &pf.input_normed,
            batch,
            &mut moe_scratch.shared_gate_logit,
        )?;
        runtime.scale_by_sigmoid_rows(
            &mut moe_scratch.gather_out,
            &moe_scratch.shared_gate_logit,
            batch,
            hidden_size,
        )?;
    }

    report(cp_shared, "shared_mlp_done");
    let cp_router = mark("");
    // ── Step 3: stream1 = post_ffn_norm_1(shared MLP output) ────────────────
    if let Some(ref n1) = layer.post_feedforward_layernorm_1 {
        runtime.rms_norm_batched_device(
            &moe_scratch.gather_out,
            n1,
            batch,
            rms_norm_eps,
            &mut moe_scratch.stream1,
        )?;
    } else {
        // Without an explicit post-norm, just copy.
        runtime.copy_prefix_f32_device(
            &moe_scratch.gather_out,
            &mut moe_scratch.stream1,
            batch_hidden,
        )?;
    }

    // ── Step 4: expert_input = pre_ffn_norm_2(residual) ─────────────────────
    if let Some(ref n2) = layer.pre_feedforward_layernorm_2 {
        runtime.rms_norm_batched_device(
            &pf.hidden,
            n2,
            batch,
            rms_norm_eps,
            &mut moe_scratch.expert_input,
        )?;
    } else {
        runtime.copy_prefix_f32_device(
            &pf.input_normed,
            &mut moe_scratch.expert_input,
            batch_hidden,
        )?;
    }

    // ── Step 5: router input ────────────────────────────────────────────────
    // Gemma 4 router pre-processing: rms_norm(residual) * router.scale * root_size
    // (matches transformers Gemma4TextRouter / vLLM Gemma4Router).
    let router_input: &DeviceBuffer<f32> = match &moe.router_input_scale {
        Some(input_scale) => {
            runtime.rms_norm_batched_device(
                &pf.hidden,
                input_scale,
                batch,
                rms_norm_eps,
                &mut moe_scratch.router_input,
            )?;
            let root = (hidden_size as f32).powf(-0.5);
            runtime.scale_f32_device_len(root, &mut moe_scratch.router_input, batch_hidden)?;
            &moe_scratch.router_input
        }
        None => &pf.hidden,
    };

    // ── Step 6: router_logits = router(router_input) ────────────────────────
    if runtime.cublaslt_bf16_enabled_for(&moe.router) {
        runtime.matmul_bf16_cublaslt_device(
            &moe.router, router_input, batch,
            &mut pf.bf16_in_scratch, &mut pf.bf16_out_scratch,
            &mut moe_scratch.router_logits,
        )?;
    } else {
        runtime.matmul_bf16_reference_batched_device(
            &moe.router, router_input, batch, &mut moe_scratch.router_logits,
        )?;
    }

    report(cp_router, "router_done");
    let cp_topk = mark("");
    // ── Step 7: GPU-resident softmax + top-k + per-expert-scale + bucket sort.
    // All data stays on the device; host downloads only the small
    // `expert_counts[num_experts]` array to drive the per-expert dispatch.
    let num_experts = moe.num_experts;
    let top_k = moe.top_k;
    runtime.router_softmax_topk_device(
        &moe_scratch.router_logits,
        &moe.router_per_expert_scale_device,
        batch,
        num_experts,
        top_k,
        &mut moe_scratch.topk_idx,
        &mut moe_scratch.topk_weights,
    )?;
    runtime.router_zero_expert_counts_device(&mut moe_scratch.expert_counts, num_experts)?;
    let stride = moe_scratch.expert_list_stride;
    runtime.router_bucket_sort_device(
        &moe_scratch.topk_idx,
        &moe_scratch.topk_weights,
        batch,
        top_k,
        stride,
        &mut moe_scratch.expert_token_lists,
        &mut moe_scratch.expert_weight_lists,
        &mut moe_scratch.expert_counts,
    )?;
    // Download just the counts (~512 bytes for 128 experts) to size each
    // expert's per-token batch.
    let counts_host = runtime.download_u32(&moe_scratch.expert_counts)?;

    report(cp_topk, "topk_done");
    let cp_experts = mark("");
    // ── Step 8: zero accumulator, then dispatch experts. ───────────────────
    runtime.zero_f32_device_len(&mut moe_scratch.moe_acc, batch_hidden)?;

    // Build the list of active experts (count > 0) for this chunk.
    let mut active_experts: Vec<usize> = Vec::new();
    for e in 0..num_experts {
        if counts_host[e] > 0 {
            active_experts.push(e);
        }
    }

    // ── GROUPED PATH (B.2) ────────────────────────────────────────────────
    // Single-launch grouped GEMM per projection (gate/up/down) instead of
    // 30-active-experts × 3-projections = 90 launches. Activated by default
    // when the NVFP4 routed-expert weights are host-resident (the common
    // case for `hidden-layers.store=ram`). Set `AEGIS_GROUPED_MOE_DISABLE=1`
    // to fall back to the per-expert path for diagnostics.
    let grouped_disabled = std::env::var("AEGIS_GROUPED_MOE_DISABLE").is_ok();
    let first_expert_host_resident = active_experts
        .first()
        .map(|&e| moe.experts[e].gate_proj.is_host_resident())
        .unwrap_or(false);
    let use_grouped = !grouped_disabled
        && !active_experts.is_empty()
        && first_expert_host_resident;

    // CUTLASS NVFP4 grouped MoE: opt-in via AEGIS_CUTLASS_NVFP4_GROUPED=1
    // AND build-time AEGIS_CUTLASS_NVFP4_GROUPED_BUILD=1. Routes the large
    // active experts (M >= AEGIS_CUTLASS_NVFP4_GROUPED_M_THRESHOLD, default
    // 128) through CUTLASS grouped GEMM; routes small experts through the
    // existing per-expert path for correctness (CUTLASS rejects M<128).
    let cutlass_enabled = std::env::var("AEGIS_CUTLASS_NVFP4_GROUPED").is_ok()
        && CudaRuntime::cutlass_nvfp4_moe_grouped_built()
        && moe_scratch.cutlass.is_some();

    if use_grouped && cutlass_enabled {
        forward_moe_cutlass_split_routed_experts(
            runtime, moe, moe_scratch, &active_experts, &counts_host,
            stride, num_experts, hidden_size, batch, top_k, staging_ptr,
        )?;
        report(cp_experts, "experts_done");
    } else if use_grouped {
        forward_moe_grouped_routed_experts(
            runtime, moe, moe_scratch, &active_experts, &counts_host,
            stride, num_experts, hidden_size, batch, top_k,
        )?;
        report(cp_experts, "experts_done");
        // Skip the per-expert dispatch below.
    } else {
    // Per-expert dispatch. Each active expert: copy its bucketed indices /
    // weights into gather_*, gather rows from `expert_input`, run the three
    // NVFP4 matmuls + GeGLU, then scatter-add weighted into `moe_acc`.
    for &expert_idx in &active_experts {
        let count = counts_host[expert_idx] as usize;
        if count == 0 {
            continue;
        }
        let bucket_off = expert_idx * stride;
        runtime.copy_u32_d2d_range(
            &moe_scratch.expert_token_lists,
            bucket_off,
            &mut moe_scratch.gather_indices,
            0,
            count,
        )?;
        runtime.copy_f32_d2d_range(
            &moe_scratch.expert_weight_lists,
            bucket_off,
            &mut moe_scratch.gather_weights,
            0,
            count,
        )?;
        runtime.gather_rows_f32_device(
            &moe_scratch.expert_input,
            &moe_scratch.gather_indices,
            count,
            hidden_size,
            &mut moe_scratch.gather_input,
        )?;
        let expert = &moe.experts[expert_idx];
        let exp_intermediate = expert.gate_proj.rows;
        matvec_nvfp4_batched_device_with_scratch(
            runtime, &expert.gate_proj, &moe_scratch.gather_input, count,
            &mut moe_scratch.gather_quant, &mut moe_scratch.gather_mxfp4,
            &mut moe_scratch.gather_intermediate,
            if staging_ptr.is_null() { None } else { Some(unsafe { &mut *staging_ptr }) },
        )?;
        matvec_nvfp4_batched_device_with_scratch(
            runtime, &expert.up_proj, &moe_scratch.gather_input, count,
            &mut moe_scratch.gather_quant, &mut moe_scratch.gather_mxfp4,
            &mut moe_scratch.gather_swiglu,
            if staging_ptr.is_null() { None } else { Some(unsafe { &mut *staging_ptr }) },
        )?;
        runtime.geglu_tanh_in_place_device(
            &moe_scratch.gather_intermediate,
            &mut moe_scratch.gather_swiglu,
            count * exp_intermediate,
        )?;
        matvec_nvfp4_batched_device_with_scratch(
            runtime,
            &expert.down_proj,
            &moe_scratch.gather_swiglu,
            count,
            &mut moe_scratch.gather_quant,
            &mut moe_scratch.gather_mxfp4,
            &mut moe_scratch.gather_out,
            if staging_ptr.is_null() { None } else { Some(unsafe { &mut *staging_ptr }) },
        )?;
        runtime.scatter_add_weighted_f32_device(
            &moe_scratch.gather_out,
            &moe_scratch.gather_indices,
            &moe_scratch.gather_weights,
            count,
            hidden_size,
            &mut moe_scratch.moe_acc,
        )?;
    }
    report(cp_experts, "experts_done");
    }
    // ── Step 9: post_ffn_norm_2(moe_acc) (stream2). In-place is safe — each
    //   thread reads/writes its own column index per the rms_norm kernel layout.
    if let Some(ref n2) = layer.post_feedforward_layernorm_2 {
        runtime.rms_norm_batched_device(
            &moe_scratch.moe_acc,
            n2,
            batch,
            rms_norm_eps,
            &mut moe_scratch.gather_out,
        )?;
        // Move stream2 into moe_acc for the upcoming add.
        runtime.copy_prefix_f32_device(
            &moe_scratch.gather_out,
            &mut moe_scratch.moe_acc,
            batch_hidden,
        )?;
    }

    // ── Step 10: moe_acc += stream1  (combined = stream1 + stream2) ─────────
    runtime.add_inplace_device_len(
        &mut moe_scratch.moe_acc,
        &moe_scratch.stream1,
        batch_hidden,
    )?;

    // ── Step 11: post_ffn_norm(combined) → gather_out ───────────────────────
    if let Some(ref final_norm) = layer.post_mlp_sublayer_norm {
        runtime.rms_norm_batched_device(
            &moe_scratch.moe_acc,
            final_norm,
            batch,
            rms_norm_eps,
            &mut moe_scratch.gather_out,
        )?;
    } else {
        runtime.copy_prefix_f32_device(
            &moe_scratch.moe_acc,
            &mut moe_scratch.gather_out,
            batch_hidden,
        )?;
    }

    // ── Step 12: prefill.hidden += gather_out ────────────────────────────────
    runtime.add_inplace_device_len(&mut pf.hidden, &moe_scratch.gather_out, batch_hidden)?;

    // ── Step 13: prefill.hidden *= layer_scalar ─────────────────────────────
    if let Some(scalar) = layer.layer_scalar {
        runtime.scale_f32_device_len(scalar, &mut pf.hidden, batch_hidden)?;
    }

    Ok(())
}

/// Gemma 4 routing post-processing per token. Mirrors the decode-side helper.
fn softmax_top_k_normalized(
    logits: &[f32],
    top_k: usize,
    per_expert_scale: Option<&[f32]>,
) -> (Vec<usize>, Vec<f32>) {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
    let k = top_k.min(probs.len());
    let mut idx: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    idx.select_nth_unstable_by(k - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut top = idx[..k].to_vec();
    top.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut indices: Vec<usize> = top.iter().map(|(i, _)| *i).collect();
    let mut weights: Vec<f32> = top.iter().map(|(_, w)| *w).collect();
    let wsum: f32 = weights.iter().sum();
    if wsum > 0.0 {
        for w in weights.iter_mut() {
            *w /= wsum;
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

// ─────────────────────────────────────────────────────────────────────────────
// Grouped MoE forward (Phase B.2): single-launch GEMM per projection.
// Replaces ~90 per-expert kernel calls with 3 grouped calls (gate/up/down)
// + one fused permute_gather + one fused unpermute_scatter_add.
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum ExpertProjKind {
    Gate,
    Up,
    Down,
}

/// Stage one projection of all active experts into the bulk VRAM buffers
/// **on the dedicated transfer stream** (Phase B.3 dual-stream overlap).
/// Returns a `CudaEvent` recorded on the transfer stream after all H2D
/// ops complete. The caller arranges for the compute stream to `wait`
/// on this event before launching the consumer GEMM, so kernel work for
/// projection N runs concurrently with H2D for projection N+1.
#[allow(clippy::too_many_arguments)]
fn stage_active_experts_projection_async(
    runtime: &CudaRuntime,
    experts: &[crate::executor::state::CudaMoEExpert],
    active_experts: &[usize],
    projection: ExpertProjKind,
    bulk_packed: &mut DeviceBuffer<u8>,
    bulk_scales: &mut DeviceBuffer<u8>,
    bulk_packed_offsets: &mut DeviceBuffer<u32>,
    bulk_scales_offsets: &mut DeviceBuffer<u32>,
    bulk_output_scales: &mut DeviceBuffer<f32>,
) -> Result<cudarc::driver::CudaEvent> {
    let mut packed_offsets_host: Vec<u32> = Vec::with_capacity(active_experts.len());
    let mut scales_offsets_host: Vec<u32> = Vec::with_capacity(active_experts.len());
    let mut output_scales_host: Vec<f32> = Vec::with_capacity(active_experts.len());

    let mut packed_off: usize = 0;
    let mut scales_off: usize = 0;
    for &expert_idx in active_experts {
        let expert = &experts[expert_idx];
        let proj = match projection {
            ExpertProjKind::Gate => &expert.gate_proj,
            ExpertProjKind::Up => &expert.up_proj,
            ExpertProjKind::Down => &expert.down_proj,
        };
        let (packed_bytes, scales_bytes) = proj
            .host_packed_scales_bytes()
            .ok_or_else(|| AegisError::InvalidPlan(format!(
                "grouped MoE staging: expert `{}` is not host-resident", proj.name
            )))??;
        packed_offsets_host.push(packed_off as u32);
        scales_offsets_host.push(scales_off as u32);
        output_scales_host.push(proj.output_scale);
        runtime.copy_host_u8_to_device_at_offset_async(packed_bytes, bulk_packed, packed_off)?;
        runtime.copy_host_u8_to_device_at_offset_async(scales_bytes, bulk_scales, scales_off)?;
        packed_off += packed_bytes.len();
        scales_off += scales_bytes.len();
    }

    runtime.upload_u32_slice_to_device_async(&packed_offsets_host, bulk_packed_offsets)?;
    runtime.upload_u32_slice_to_device_async(&scales_offsets_host, bulk_scales_offsets)?;
    runtime.upload_f32_slice_to_device_async(&output_scales_host, bulk_output_scales)?;
    runtime.record_transfer_event()
}

/// Run all routed-expert GEMMs (gate / up / down) for one chunk in three
/// grouped kernel launches instead of `30 active × 3 = 90` per-expert
/// launches. Result is written into `moe_scratch.moe_acc` exactly as the
/// per-expert path does (atomic scatter-add of weighted per-token outputs).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn forward_moe_grouped_routed_experts(
    runtime: &CudaRuntime,
    moe: &CudaMoE,
    moe_scratch: &mut crate::executor::state::CudaMoEPrefillScratch,
    active_experts: &[usize],
    counts_host: &[u32],
    stride: usize,
    num_experts: usize,
    hidden_size: usize,
    batch: usize,
    top_k: usize,
) -> Result<()> {
    if active_experts.is_empty() {
        return Ok(());
    }

    // Per-active-expert token-offset prefix sum (used by the grouped GEMM).
    // Indexes into the permuted_input layout: since inactive experts have
    // 0-width slots, active prefix-sum offsets coincide with the all-experts
    // offsets at active positions.
    let mut active_token_offsets_host: Vec<u32> = Vec::with_capacity(active_experts.len() + 1);
    active_token_offsets_host.push(0);
    let mut max_tokens_per_active = 0usize;
    for &expert_idx in active_experts {
        let count = counts_host[expert_idx];
        max_tokens_per_active = max_tokens_per_active.max(count as usize);
        let prev = *active_token_offsets_host.last().unwrap();
        active_token_offsets_host.push(prev + count);
    }
    let total_active_tokens = *active_token_offsets_host.last().unwrap() as usize;
    if std::env::var("AEGIS_MOE_M_DUMP").is_ok() {
        let mut counts: Vec<u32> = active_experts.iter().map(|&e| counts_host[e]).collect();
        counts.sort_unstable();
        let n = counts.len();
        let p50 = counts[n / 2];
        let p90 = counts[(n * 9) / 10];
        let mut lt64 = 0usize; let mut lt128 = 0usize;
        let mut ge128 = 0usize; let mut ge256 = 0usize;
        for &c in &counts {
            if c < 64 { lt64 += 1; } else if c < 128 { lt128 += 1; }
            else if c < 256 { ge128 += 1; } else { ge256 += 1; }
        }
        eprintln!(
            "[MOE-M] active={} min={} p50={} p90={} max={} | <64:{}  64-127:{}  128-255:{}  >=256:{}",
            n, counts[0], p50, p90, counts[n-1], lt64, lt128, ge128, ge256,
        );
    }
    runtime.upload_u32_slice_to_device(
        &active_token_offsets_host,
        &mut moe_scratch.bulk_token_offsets,
    )?;

    // All-experts prefix-sum (used by permute_gather / unpermute_scatter).
    runtime.router_expert_offsets_device(
        &moe_scratch.expert_counts,
        num_experts,
        &mut moe_scratch.expert_offsets,
    )?;

    // Permute: expert_input[batch, hidden] → permuted_input[total_routed, hidden].
    let max_per_expert_overall = counts_host.iter().copied().max().unwrap_or(0) as usize;
    runtime.permute_gather_f32_device(
        &moe_scratch.expert_input,
        &moe_scratch.expert_token_lists,
        &moe_scratch.expert_counts,
        &moe_scratch.expert_offsets,
        stride,
        num_experts,
        max_per_expert_overall,
        hidden_size,
        &mut moe_scratch.permuted_input,
    )?;

    // Shape constants. All routed experts share dimensions for the same
    // projection family on Gemma-4 (704 / 2816). Read from first active expert.
    let first = &moe.experts[active_experts[0]];
    let exp_intermediate = first.gate_proj.rows;  // gate/up output rows
    let exp_cols_in     = first.gate_proj.cols;   // gate/up cols (== hidden)
    let down_rows       = first.down_proj.rows;   // == hidden
    let down_cols       = first.down_proj.cols;   // == intermediate
    let num_active = active_experts.len();

    // ── Phase B.3: dual-stream H2D / compute overlap. Issue all 3
    //   projections' H2Ds upfront on the transfer stream so the driver
    //   can pipeline them; compute stream waits per-projection events
    //   right before each grouped GEMM. Each projection has its own slot
    //   so transfer-stream writes to slot N+1 don't race with the kernel
    //   reading slot N's metadata. ──
    let gate_event = stage_active_experts_projection_async(
        runtime, &moe.experts, active_experts, ExpertProjKind::Gate,
        &mut moe_scratch.bulk_slots[0].bulk_packed,
        &mut moe_scratch.bulk_slots[0].bulk_scales,
        &mut moe_scratch.bulk_slots[0].bulk_packed_offsets,
        &mut moe_scratch.bulk_slots[0].bulk_scales_offsets,
        &mut moe_scratch.bulk_slots[0].bulk_output_scales,
    )?;
    let up_event = stage_active_experts_projection_async(
        runtime, &moe.experts, active_experts, ExpertProjKind::Up,
        &mut moe_scratch.bulk_slots[1].bulk_packed,
        &mut moe_scratch.bulk_slots[1].bulk_scales,
        &mut moe_scratch.bulk_slots[1].bulk_packed_offsets,
        &mut moe_scratch.bulk_slots[1].bulk_scales_offsets,
        &mut moe_scratch.bulk_slots[1].bulk_output_scales,
    )?;
    let down_event = stage_active_experts_projection_async(
        runtime, &moe.experts, active_experts, ExpertProjKind::Down,
        &mut moe_scratch.bulk_slots[2].bulk_packed,
        &mut moe_scratch.bulk_slots[2].bulk_scales,
        &mut moe_scratch.bulk_slots[2].bulk_packed_offsets,
        &mut moe_scratch.bulk_slots[2].bulk_scales_offsets,
        &mut moe_scratch.bulk_slots[2].bulk_output_scales,
    )?;

    // ── Phase B.4 Round 2: fused gate+up grouped GEMM (opt-in via
    //   AEGIS_NVFP4_GROUPED_DUAL_ENABLE=1). Default-off: empirical bench
    //   regressed -5% at 9.6k because (a) shared-mem doubles → lower
    //   occupancy, and (b) fusing waits for both H2Ds before launch,
    //   killing the gate_GEMM || up_H2D overlap. Kept in-tree for future
    //   tuning (cp.async + smaller shmem may flip the win). ──
    let dual_enabled = std::env::var("AEGIS_NVFP4_GROUPED_DUAL_ENABLE").is_ok();
    if dual_enabled {
        runtime.compute_wait_event(&gate_event)?;
        runtime.compute_wait_event(&up_event)?;
        let (slots_lo, slots_hi) = moe_scratch.bulk_slots.split_at(1);
        let slot_gate = &slots_lo[0];
        let slot_up   = &slots_hi[0];
        runtime.matmul_nvfp4_grouped_prequant_wmma_bf16_dual_device(
            &slot_gate.bulk_packed, &slot_gate.bulk_scales,
            &slot_gate.bulk_packed_offsets, &slot_gate.bulk_scales_offsets,
            &slot_gate.bulk_output_scales,
            &slot_up.bulk_packed, &slot_up.bulk_scales,
            &slot_up.bulk_packed_offsets, &slot_up.bulk_scales_offsets,
            &slot_up.bulk_output_scales,
            &moe_scratch.bulk_token_offsets,
            &moe_scratch.permuted_input,
            exp_intermediate, exp_cols_in, max_tokens_per_active, num_active,
            &mut moe_scratch.permuted_intermediate,
            &mut moe_scratch.permuted_swiglu,
        )?;
    } else {
        // Default: gate first, then up. up_H2D overlaps gate_GEMM.
        runtime.compute_wait_event(&gate_event)?;
        {
            let slot = &moe_scratch.bulk_slots[0];
            runtime.matmul_nvfp4_grouped_prequant_wmma_bf16_device(
                &slot.bulk_packed, &slot.bulk_scales,
                &slot.bulk_packed_offsets, &slot.bulk_scales_offsets,
                &slot.bulk_output_scales, &moe_scratch.bulk_token_offsets,
                &moe_scratch.permuted_input,
                exp_intermediate, exp_cols_in, max_tokens_per_active, num_active,
                &mut moe_scratch.permuted_intermediate,
            )?;
        }
        runtime.compute_wait_event(&up_event)?;
        {
            let slot = &moe_scratch.bulk_slots[1];
            runtime.matmul_nvfp4_grouped_prequant_wmma_bf16_device(
                &slot.bulk_packed, &slot.bulk_scales,
                &slot.bulk_packed_offsets, &slot.bulk_scales_offsets,
                &slot.bulk_output_scales, &moe_scratch.bulk_token_offsets,
                &moe_scratch.permuted_input,
                exp_intermediate, exp_cols_in, max_tokens_per_active, num_active,
                &mut moe_scratch.permuted_swiglu,
            )?;
        }
    }

    // GeGLU (Gemma 4): permuted_swiglu = geglu_tanh(permuted_intermediate, permuted_swiglu).
    runtime.geglu_tanh_in_place_device(
        &moe_scratch.permuted_intermediate,
        &mut moe_scratch.permuted_swiglu,
        total_active_tokens * exp_intermediate,
    )?;

    // ── Down projection ──
    runtime.compute_wait_event(&down_event)?;
    {
        let slot = &moe_scratch.bulk_slots[2];
        runtime.matmul_nvfp4_grouped_prequant_wmma_bf16_device(
            &slot.bulk_packed, &slot.bulk_scales,
            &slot.bulk_packed_offsets, &slot.bulk_scales_offsets,
            &slot.bulk_output_scales, &moe_scratch.bulk_token_offsets,
            &moe_scratch.permuted_swiglu,
            down_rows, down_cols, max_tokens_per_active, num_active,
            &mut moe_scratch.permuted_output,
        )?;
    }

    // Inverse permute + weighted scatter into moe_acc (deterministic — see
    // `unpermute_scatter_add_f32_device`). `moe_acc` was zeroed at step 8.
    runtime.unpermute_scatter_add_f32_device(
        &moe_scratch.permuted_output,
        &moe_scratch.expert_token_lists,
        &moe_scratch.expert_weight_lists,
        &moe_scratch.expert_counts,
        &moe_scratch.expert_offsets,
        stride,
        num_experts,
        max_per_expert_overall,
        hidden_size,
        batch,
        top_k,
        &mut moe_scratch.unpermute_rows,
        &mut moe_scratch.unpermute_wbits,
        &mut moe_scratch.unpermute_count,
        &mut moe_scratch.moe_acc,
    )?;

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// CUTLASS NVFP4 grouped split-dispatch:
//   - large experts (M >= threshold) → CUTLASS NVFP4 grouped GEMM
//   - small experts (M < threshold) → per-expert NVFP4 matvec
// CUTLASS path requires the compile-time `aegis_cutlass_nvfp4_grouped` cfg.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(not(aegis_cutlass_nvfp4_grouped))]
#[allow(clippy::too_many_arguments)]
fn forward_moe_cutlass_split_routed_experts(
    _runtime: &CudaRuntime,
    _moe: &CudaMoE,
    _moe_scratch: &mut crate::executor::state::CudaMoEPrefillScratch,
    _active_experts: &[usize],
    _counts_host: &[u32],
    _stride: usize,
    _num_experts: usize,
    _hidden_size: usize,
    _batch: usize,
    _top_k: usize,
    _staging_ptr: *mut LinearStagingPool,
) -> Result<()> {
    Err(AegisError::Unsupported(
        "CUTLASS NVFP4 grouped MoE not compiled into this build; \
         rebuild with AEGIS_CUTLASS_NVFP4_GROUPED_BUILD=1".into(),
    ))
}

#[cfg(aegis_cutlass_nvfp4_grouped)]
#[allow(clippy::too_many_arguments)]
fn forward_moe_cutlass_split_routed_experts(
    runtime: &CudaRuntime,
    moe: &CudaMoE,
    moe_scratch: &mut crate::executor::state::CudaMoEPrefillScratch,
    active_experts: &[usize],
    counts_host: &[u32],
    stride: usize,
    num_experts: usize,
    hidden_size: usize,
    batch: usize,
    top_k: usize,
    staging_ptr: *mut LinearStagingPool,
) -> Result<()> {
    if active_experts.is_empty() {
        return Ok(());
    }

    let threshold = std::env::var("AEGIS_CUTLASS_NVFP4_GROUPED_M_THRESHOLD")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(128usize);

    // Split active experts (ascending by global id) into large / small.
    // Keep ascending order so the existing permuted_input layout
    // (which the upstream `router_expert_offsets_device` + `permute_gather`
    // populated by global-expert order) stays valid for both subsets.
    let mut large_experts: Vec<usize> = Vec::with_capacity(active_experts.len());
    let mut small_experts: Vec<usize> = Vec::with_capacity(active_experts.len());
    for &e in active_experts {
        if (counts_host[e] as usize) >= threshold {
            large_experts.push(e);
        } else {
            small_experts.push(e);
        }
    }

    // If everything is small, defer to the existing grouped path — no CUTLASS
    // work to do, no payoff. (CUTLASS will reject M<128 anyway.)
    if large_experts.is_empty() {
        return forward_moe_grouped_routed_experts(
            runtime, moe, moe_scratch, active_experts, counts_host,
            stride, num_experts, hidden_size, batch, top_k,
        );
    }

    // Per-active-expert prefix-sum offsets for the existing grouped path:
    // used for staging metadata and (when we fall through) the t32_big kernel.
    let mut active_token_offsets_host: Vec<u32> = Vec::with_capacity(active_experts.len() + 1);
    active_token_offsets_host.push(0);
    let mut max_tokens_per_active = 0usize;
    for &expert_idx in active_experts {
        let count = counts_host[expert_idx];
        max_tokens_per_active = max_tokens_per_active.max(count as usize);
        let prev = *active_token_offsets_host.last().unwrap();
        active_token_offsets_host.push(prev + count);
    }
    let _total_active_tokens = *active_token_offsets_host.last().unwrap() as usize;
    let _ = max_tokens_per_active;
    runtime.upload_u32_slice_to_device(
        &active_token_offsets_host,
        &mut moe_scratch.bulk_token_offsets,
    )?;

    // Compute global per-expert prefix-sum offsets (the layout of permuted_input).
    runtime.router_expert_offsets_device(
        &moe_scratch.expert_counts,
        num_experts,
        &mut moe_scratch.expert_offsets,
    )?;
    // Mirror it host-side so we can compute absolute byte offsets for CUTLASS
    // per-group A/D pointers. `expert_offsets[e]` = sum of counts[0..e].
    let mut expert_offsets_host = vec![0u32; num_experts + 1];
    for e in 0..num_experts {
        expert_offsets_host[e + 1] = expert_offsets_host[e] + counts_host[e];
    }
    let total_routed = expert_offsets_host[num_experts] as usize;

    // permute_gather over all active experts (large + small).
    let max_per_expert_overall = counts_host.iter().copied().max().unwrap_or(0) as usize;
    runtime.permute_gather_f32_device(
        &moe_scratch.expert_input,
        &moe_scratch.expert_token_lists,
        &moe_scratch.expert_counts,
        &moe_scratch.expert_offsets,
        stride,
        num_experts,
        max_per_expert_overall,
        hidden_size,
        &mut moe_scratch.permuted_input,
    )?;

    let first = &moe.experts[active_experts[0]];
    let exp_intermediate = first.gate_proj.rows;  // gate/up output rows
    let exp_cols_in     = first.gate_proj.cols;   // gate/up cols (== hidden)
    let down_rows       = first.down_proj.rows;   // == hidden
    let down_cols       = first.down_proj.cols;   // == intermediate

    // Zero permuted_intermediate / permuted_swiglu / permuted_output so that
    // small-expert rows are 0-filled. GeGLU and the final unpermute_scatter
    // are run over the entire buffer; with zero-filled small-expert rows,
    // geglu(0,0)=0 and scatter-add(0)=no-op. Small experts contribute via
    // the per-expert path that writes directly into moe_acc.
    runtime.zero_f32_device_len(
        &mut moe_scratch.permuted_intermediate,
        total_routed * exp_intermediate,
    )?;
    runtime.zero_f32_device_len(
        &mut moe_scratch.permuted_swiglu,
        total_routed * exp_intermediate,
    )?;
    runtime.zero_f32_device_len(
        &mut moe_scratch.permuted_output,
        total_routed * hidden_size,
    )?;

    // Stage all three projections' bulk weights asynchronously (matches the
    // existing grouped path so we share the staging pipeline).
    let gate_event = stage_active_experts_projection_async(
        runtime, &moe.experts, active_experts, ExpertProjKind::Gate,
        &mut moe_scratch.bulk_slots[0].bulk_packed,
        &mut moe_scratch.bulk_slots[0].bulk_scales,
        &mut moe_scratch.bulk_slots[0].bulk_packed_offsets,
        &mut moe_scratch.bulk_slots[0].bulk_scales_offsets,
        &mut moe_scratch.bulk_slots[0].bulk_output_scales,
    )?;
    let up_event = stage_active_experts_projection_async(
        runtime, &moe.experts, active_experts, ExpertProjKind::Up,
        &mut moe_scratch.bulk_slots[1].bulk_packed,
        &mut moe_scratch.bulk_slots[1].bulk_scales,
        &mut moe_scratch.bulk_slots[1].bulk_packed_offsets,
        &mut moe_scratch.bulk_slots[1].bulk_scales_offsets,
        &mut moe_scratch.bulk_slots[1].bulk_output_scales,
    )?;
    let down_event = stage_active_experts_projection_async(
        runtime, &moe.experts, active_experts, ExpertProjKind::Down,
        &mut moe_scratch.bulk_slots[2].bulk_packed,
        &mut moe_scratch.bulk_slots[2].bulk_scales,
        &mut moe_scratch.bulk_slots[2].bulk_packed_offsets,
        &mut moe_scratch.bulk_slots[2].bulk_scales_offsets,
        &mut moe_scratch.bulk_slots[2].bulk_output_scales,
    )?;

    // Walk active_experts AGAIN host-side and record the per-active-expert
    // packed/scales byte offsets — same arithmetic as `stage_active_experts_projection_async`
    // but we need it here too for building the CUTLASS pointer arrays.
    let projections = [ExpertProjKind::Gate, ExpertProjKind::Up, ExpertProjKind::Down];
    let mut per_proj_packed_offsets_host: [Vec<u64>; 3] = Default::default();
    let mut per_proj_scales_offsets_host: [Vec<u64>; 3] = Default::default();
    let mut per_proj_alpha_host: [Vec<f32>; 3] = Default::default();
    for (pi, proj) in projections.iter().enumerate() {
        let mut packed_off: u64 = 0;
        let mut scales_off: u64 = 0;
        let mut packed = Vec::with_capacity(active_experts.len());
        let mut scales = Vec::with_capacity(active_experts.len());
        let mut alphas = Vec::with_capacity(active_experts.len());
        for &expert_idx in active_experts {
            let expert = &moe.experts[expert_idx];
            let p = match *proj {
                ExpertProjKind::Gate => &expert.gate_proj,
                ExpertProjKind::Up => &expert.up_proj,
                ExpertProjKind::Down => &expert.down_proj,
            };
            packed.push(packed_off);
            scales.push(scales_off);
            alphas.push(p.output_scale);
            packed_off += p.packed_bytes as u64;
            scales_off += p.scale_bytes as u64;
        }
        per_proj_packed_offsets_host[pi] = packed;
        per_proj_scales_offsets_host[pi] = scales;
        per_proj_alpha_host[pi] = alphas;
    }

    // Indices of large experts in the `active_experts` array (positions in
    // bulk staging arrays).
    let mut large_pos_in_active: Vec<usize> = Vec::with_capacity(large_experts.len());
    {
        let mut ae_iter = 0usize;
        for &ge in &large_experts {
            while ae_iter < active_experts.len() && active_experts[ae_iter] != ge {
                ae_iter += 1;
            }
            large_pos_in_active.push(ae_iter);
            ae_iter += 1;
        }
    }

    let num_large = large_experts.len();
    let cutlass_scratch = moe_scratch.cutlass.as_deref_mut().ok_or_else(|| {
        AegisError::InvalidPlan("CUTLASS MoE scratch missing in cutlass-enabled path".into())
    })?;

    // Per-large-expert payload offsets (cumulative).
    let cols_hidden_div_2 = hidden_size / 2;
    let cols_intermediate_div_2 = exp_intermediate / 2;
    let padded_scale_cols_hidden = (hidden_size / 16).div_ceil(4) * 4;
    let padded_scale_cols_intermediate = (exp_intermediate / 16).div_ceil(4) * 4;

    let mut payload_off_hidden_host: Vec<u64> = Vec::with_capacity(num_large);
    let mut sfa_off_hidden_host: Vec<u64> = Vec::with_capacity(num_large);
    let mut payload_off_intermediate_host: Vec<u64> = Vec::with_capacity(num_large);
    let mut sfa_off_intermediate_host: Vec<u64> = Vec::with_capacity(num_large);
    let mut payload_cum_h: u64 = 0;
    let mut sfa_cum_h: u64 = 0;
    let mut payload_cum_i: u64 = 0;
    let mut sfa_cum_i: u64 = 0;
    for &ge in &large_experts {
        let m_g = counts_host[ge] as usize;
        payload_off_hidden_host.push(payload_cum_h);
        sfa_off_hidden_host.push(sfa_cum_h);
        payload_off_intermediate_host.push(payload_cum_i);
        sfa_off_intermediate_host.push(sfa_cum_i);
        let padded_rows = m_g.div_ceil(128) * 128;
        payload_cum_h += (m_g * cols_hidden_div_2) as u64;
        sfa_cum_h += (padded_rows * padded_scale_cols_hidden) as u64;
        payload_cum_i += (m_g * cols_intermediate_div_2) as u64;
        sfa_cum_i += (padded_rows * padded_scale_cols_intermediate) as u64;
    }

    // Build a single multi-group offset blob host-side, upload ONCE per K-type,
    // then issue per-group quantize launches that stride into the blob via
    // raw-pointer offset. This avoids per-iter implicit HTOD-sync stalls.
    //
    // Layout:
    //   token_offsets:    2*N entries (per group: [global_start, global_end]).
    //   payload_offsets:  N entries (byte offset into input_packed_*).
    //   sfa_offsets:      N entries (byte offset into input_sfa_*).
    let mut tok_blob_hidden: Vec<u32> = Vec::with_capacity(num_large * 2);
    let mut max_m_hidden = 0u32;
    for &ge in &large_experts {
        let m_g = counts_host[ge];
        max_m_hidden = max_m_hidden.max(m_g);
        tok_blob_hidden.push(expert_offsets_host[ge]);
        tok_blob_hidden.push(expert_offsets_host[ge] + m_g);
    }
    let payload_blob_hidden: Vec<u64> = payload_off_hidden_host.clone();
    let sfa_blob_hidden: Vec<u64> = sfa_off_hidden_host.clone();
    let payload_blob_intermediate: Vec<u64> = payload_off_intermediate_host.clone();
    let sfa_blob_intermediate: Vec<u64> = sfa_off_intermediate_host.clone();
    let tok_blob_intermediate = tok_blob_hidden.clone(); // same token rows

    // Upload all four arrays once each.
    runtime.upload_u32_slice_to_device(&tok_blob_hidden, &mut cutlass_scratch.token_offsets)?;
    // Pre-allocated quant_payload_off_scratch is size=1; we need num_large.
    // Reuse the cutlass_scratch.payload_offsets / sfa_offsets — they were
    // sized to max_active in the scratch allocator, so num_large fits.
    runtime.upload_u64_slice_to_device(&payload_blob_hidden, &mut cutlass_scratch.payload_offsets)?;
    runtime.upload_u64_slice_to_device(&sfa_blob_hidden, &mut cutlass_scratch.sfa_offsets)?;
    let _ = tok_blob_intermediate;

    // Per-large-group quantize for gate/up shared input (K=hidden) — N launches
    // but ZERO H2D between them.
    for (lg, &ge) in large_experts.iter().enumerate() {
        let m_g = counts_host[ge] as usize;
        let cs_raw: *mut super::super::state::CutlassMoeScratch = &mut *cutlass_scratch;
        let cs = unsafe { &mut *cs_raw };
        runtime.cutlass_moe_nvfp4_quantize_input_single_strided(
            &moe_scratch.permuted_input,
            hidden_size,
            &cs.token_offsets, lg * 2, // [start_lg, end_lg] live at indices 2*lg, 2*lg+1
            &cs.payload_offsets, lg,
            &cs.sfa_offsets, lg,
            m_g,
            &mut cs.input_packed_hidden,
            &mut cs.input_sfa_hidden,
        )?;
    }

    // Wait on all three weight-staging events upfront. Weights are needed
    // before swizzle and GEMM. (Could overlap better — Agent 3's tuning.)
    runtime.compute_wait_event(&gate_event)?;
    runtime.compute_wait_event(&up_event)?;
    runtime.compute_wait_event(&down_event)?;

    // For each projection, swizzle large-only weight scales.
    for pi in 0..3 {
        let (n_g, k_g) = match projections[pi] {
            ExpertProjKind::Gate | ExpertProjKind::Up => (exp_intermediate, exp_cols_in),
            ExpertProjKind::Down => (down_rows, down_cols),
        };
        // SFB per group is the same for all groups in this projection (depends
        // only on n_g, k_g — not m). Query once with the largest m as a probe.
        let (_sfa_per_g, sfb_per_g) = runtime.cutlass_nvfp4_moe_grouped_sfa_sfb_bytes(
            counts_host[large_experts[0]] as usize,
            n_g, k_g,
        )?;

        let mut src_off_host: Vec<u64> = Vec::with_capacity(num_large);
        let mut dst_off_host: Vec<u64> = Vec::with_capacity(num_large);
        let mut dst_cum: u64 = 0;
        for &lpos in &large_pos_in_active {
            src_off_host.push(per_proj_scales_offsets_host[pi][lpos]);
            dst_off_host.push(dst_cum);
            dst_cum += sfb_per_g as u64;
        }

        let slot = &mut cutlass_scratch.slots[pi];
        runtime.upload_u64_slice_to_device(&src_off_host, &mut slot.src_offsets)?;
        runtime.upload_u64_slice_to_device(&dst_off_host, &mut slot.dst_offsets)?;

        let src_cols = k_g / 16;
        // Split borrow: bulk_slot is in moe_scratch.bulk_slots; slot is in cutlass_scratch.slots.
        // No overlapping borrow since `slot` is &mut cutlass_scratch.* and we
        // borrow bulk_slot from moe_scratch immutably.
        let bulk_slot = &moe_scratch.bulk_slots[pi];
        runtime.cutlass_moe_nvfp4_swizzle_weight_scales_grouped(
            &bulk_slot.bulk_scales,
            n_g,
            src_cols,
            num_large,
            &slot.src_offsets,
            &slot.dst_offsets,
            &mut slot.weight_sfb,
        )?;
    }

    // Upload per-large alpha values (all projections packed contiguously).
    let mut alphas_packed: Vec<f32> = Vec::with_capacity(num_large * 3);
    for pi in 0..3 {
        for &lpos in &large_pos_in_active {
            alphas_packed.push(per_proj_alpha_host[pi][lpos]);
        }
    }
    runtime.upload_f32_slice_to_device(&alphas_packed, &mut cutlass_scratch.alpha_values)?;

    // ── Per-projection: build blobs, pointers, launch GEMM ──────────────
    let blob_sa = cutlass_scratch.blob_stride_a;
    let blob_sb = cutlass_scratch.blob_stride_b;
    let blob_sd = cutlass_scratch.blob_stride_d;
    let blob_lsfa = cutlass_scratch.blob_layout_sfa;
    let blob_lsfb = cutlass_scratch.blob_layout_sfb;
    let blob_ps = cutlass_scratch.blob_problem_shape;
    let max_active_slots = cutlass_scratch.stride_a.len() / blob_sa;
    if num_large > max_active_slots {
        return Err(AegisError::Unsupported(format!(
            "CUTLASS MoE: num_large={num_large} > max_active_slots={max_active_slots}"
        )));
    }

    for pi in 0..3 {
        let (n_g, k_g) = match projections[pi] {
            ExpertProjKind::Gate | ExpertProjKind::Up => (exp_intermediate, exp_cols_in),
            ExpertProjKind::Down => (down_rows, down_cols),
        };

        // Before Down GEMM: run GeGLU, then quantize permuted_swiglu per group.
        if matches!(projections[pi], ExpertProjKind::Down) {
            runtime.geglu_tanh_in_place_device(
                &moe_scratch.permuted_intermediate,
                &mut moe_scratch.permuted_swiglu,
                total_routed * exp_intermediate,
            )?;
            // Token-offsets are the same (same large-expert row ranges); we
            // re-uploaded them above for the hidden-K path. Upload only the
            // K=intermediate payload/sfa offsets, then issue N strided launches.
            runtime.upload_u64_slice_to_device(
                &payload_blob_intermediate,
                &mut cutlass_scratch.payload_offsets,
            )?;
            runtime.upload_u64_slice_to_device(
                &sfa_blob_intermediate,
                &mut cutlass_scratch.sfa_offsets,
            )?;
            for (lg, &ge) in large_experts.iter().enumerate() {
                let m_g = counts_host[ge] as usize;
                let cs_raw: *mut super::super::state::CutlassMoeScratch = &mut *cutlass_scratch;
                let cs = unsafe { &mut *cs_raw };
                runtime.cutlass_moe_nvfp4_quantize_input_single_strided(
                    &moe_scratch.permuted_swiglu,
                    exp_intermediate,
                    &cs.token_offsets, lg * 2,
                    &cs.payload_offsets, lg,
                    &cs.sfa_offsets, lg,
                    m_g,
                    &mut cs.input_packed_intermediate,
                    &mut cs.input_sfa_intermediate,
                )?;
                let _ = ge;
            }
        }

        // Build per-large stride/layout/problem-shape blobs (host) and upload.
        let mut stride_a_blob = vec![0u8; blob_sa * num_large];
        let mut stride_b_blob = vec![0u8; blob_sb * num_large];
        let mut stride_d_blob = vec![0u8; blob_sd * num_large];
        let mut layout_sfa_blob = vec![0u8; blob_lsfa * num_large];
        let mut layout_sfb_blob = vec![0u8; blob_lsfb * num_large];
        let mut problem_shape_blob = vec![0u8; blob_ps * num_large];
        for (lg, &ge) in large_experts.iter().enumerate() {
            let m_g = counts_host[ge] as usize;
            runtime.cutlass_moe_nvfp4_compute_strides(
                m_g, n_g, k_g,
                &mut stride_a_blob[lg * blob_sa..(lg + 1) * blob_sa],
                &mut stride_b_blob[lg * blob_sb..(lg + 1) * blob_sb],
                &mut stride_d_blob[lg * blob_sd..(lg + 1) * blob_sd],
                &mut layout_sfa_blob[lg * blob_lsfa..(lg + 1) * blob_lsfa],
                &mut layout_sfb_blob[lg * blob_lsfb..(lg + 1) * blob_lsfb],
            )?;
            let slot_ps = &mut problem_shape_blob[lg * blob_ps..(lg + 1) * blob_ps];
            if slot_ps.len() < 12 {
                return Err(AegisError::Unsupported(format!(
                    "problem_shape slot too small: {} bytes", slot_ps.len()
                )));
            }
            let m_i32 = m_g as i32;
            slot_ps[0..4].copy_from_slice(&m_i32.to_le_bytes());
            slot_ps[4..8].copy_from_slice(&(n_g as i32).to_le_bytes());
            slot_ps[8..12].copy_from_slice(&(k_g as i32).to_le_bytes());
        }
        runtime.upload_u8_slice_to_device(&stride_a_blob, &mut cutlass_scratch.stride_a)?;
        runtime.upload_u8_slice_to_device(&stride_b_blob, &mut cutlass_scratch.stride_b)?;
        runtime.upload_u8_slice_to_device(&stride_d_blob, &mut cutlass_scratch.stride_d)?;
        runtime.upload_u8_slice_to_device(&layout_sfa_blob, &mut cutlass_scratch.layout_sfa)?;
        runtime.upload_u8_slice_to_device(&layout_sfb_blob, &mut cutlass_scratch.layout_sfb)?;
        runtime.upload_u8_slice_to_device(&problem_shape_blob, &mut cutlass_scratch.problem_shapes)?;

        // Build per-group device pointer arrays (host vectors, then upload).
        let mut a_ptrs = vec![0u64; num_large];
        let mut b_ptrs = vec![0u64; num_large];
        let mut sfa_ptrs = vec![0u64; num_large];
        let mut sfb_ptrs = vec![0u64; num_large];
        let mut d_ptrs = vec![0u64; num_large];
        let mut alpha_ptrs = vec![0u64; num_large];

        // Resolve per-projection base pointers via runtime helpers (avoids
        // direct .slice access from outside the cuda module).
        let bulk_packed_base = runtime.device_ptr_u8(&moe_scratch.bulk_slots[pi].bulk_packed);
        let sfb_base = runtime.device_ptr_u8(&cutlass_scratch.slots[pi].weight_sfb);
        let alpha_base = runtime.device_ptr_f32(&cutlass_scratch.alpha_values);
        let d_base = match projections[pi] {
            ExpertProjKind::Gate => runtime.device_ptr_f32_mut(&mut moe_scratch.permuted_intermediate),
            ExpertProjKind::Up => runtime.device_ptr_f32_mut(&mut moe_scratch.permuted_swiglu),
            ExpertProjKind::Down => runtime.device_ptr_f32_mut(&mut moe_scratch.permuted_output),
        };
        let (input_packed_base, input_sfa_base) = match projections[pi] {
            ExpertProjKind::Gate | ExpertProjKind::Up => (
                runtime.device_ptr_u8_mut(&mut cutlass_scratch.input_packed_hidden),
                runtime.device_ptr_u8_mut(&mut cutlass_scratch.input_sfa_hidden),
            ),
            ExpertProjKind::Down => (
                runtime.device_ptr_u8_mut(&mut cutlass_scratch.input_packed_intermediate),
                runtime.device_ptr_u8_mut(&mut cutlass_scratch.input_sfa_intermediate),
            ),
        };

        let payload_off_pi = match projections[pi] {
            ExpertProjKind::Gate | ExpertProjKind::Up => &payload_off_hidden_host,
            ExpertProjKind::Down => &payload_off_intermediate_host,
        };
        let sfa_off_pi = match projections[pi] {
            ExpertProjKind::Gate | ExpertProjKind::Up => &sfa_off_hidden_host,
            ExpertProjKind::Down => &sfa_off_intermediate_host,
        };

        // SFB per group: same for all groups in this projection. Query once.
        let (_, sfb_per_g) = runtime.cutlass_nvfp4_moe_grouped_sfa_sfb_bytes(
            counts_host[large_experts[0]] as usize, n_g, k_g,
        )?;

        for (lg, &ge) in large_experts.iter().enumerate() {
            let lpos = large_pos_in_active[lg];
            a_ptrs[lg] = input_packed_base + payload_off_pi[lg];
            b_ptrs[lg] = bulk_packed_base + per_proj_packed_offsets_host[pi][lpos];
            sfa_ptrs[lg] = input_sfa_base + sfa_off_pi[lg];
            sfb_ptrs[lg] = sfb_base + (lg as u64) * (sfb_per_g as u64);
            d_ptrs[lg] = d_base + (expert_offsets_host[ge] as u64) * (n_g as u64) * 4;
            alpha_ptrs[lg] = alpha_base + ((pi * num_large + lg) as u64) * 4;
        }

        let slot_mut = &mut cutlass_scratch.slots[pi];
        runtime.upload_u64_slice_to_device(&a_ptrs, &mut slot_mut.a_ptrs)?;
        runtime.upload_u64_slice_to_device(&b_ptrs, &mut slot_mut.b_ptrs)?;
        runtime.upload_u64_slice_to_device(&sfa_ptrs, &mut slot_mut.sfa_ptrs)?;
        runtime.upload_u64_slice_to_device(&sfb_ptrs, &mut slot_mut.sfb_ptrs)?;
        runtime.upload_u64_slice_to_device(&d_ptrs, &mut slot_mut.d_ptrs)?;
        runtime.upload_u64_slice_to_device(&alpha_ptrs, &mut slot_mut.alpha_ptrs)?;

        // Now split the borrow: we need an immutable ref to cutlass_scratch
        // for the stride/layout/problem-shape blobs (lives across the call),
        // plus a mutable ref to slot_mut for the per-group pointer + workspace
        // arrays. We pre-extract the cutlass_scratch references via raw
        // pointer split to satisfy the borrow checker.
        let cs_raw: *mut super::super::state::CutlassMoeScratch = &mut *cutlass_scratch;
        // SAFETY: we're inside the &mut borrow of cutlass_scratch (via the
        // outer `as_deref_mut`); slot_mut aliases cutlass_scratch.slots[pi]
        // while we also need &mut cutlass_scratch.stride_a etc — those are
        // disjoint fields, so the split is safe.
        let cs = unsafe { &mut *cs_raw };
        runtime.cutlass_moe_nvfp4_grouped_run(
            num_large,
            &cs.problem_shapes,
            &mut cs.slots[pi].a_ptrs,
            &mut cs.slots[pi].b_ptrs,
            &mut cs.slots[pi].sfa_ptrs,
            &mut cs.slots[pi].sfb_ptrs,
            &mut cs.slots[pi].d_ptrs,
            &mut cs.stride_a,
            &mut cs.stride_b,
            &mut cs.stride_d,
            &mut cs.layout_sfa,
            &mut cs.layout_sfb,
            &mut cs.slots[pi].alpha_ptrs,
            &mut cs.slots[pi].workspace,
        )?;
    }

    // Unpermute + scatter-add: large experts contribute CUTLASS results,
    // small experts contribute zeros (their positions stay zero throughout).
    runtime.unpermute_scatter_add_f32_device(
        &moe_scratch.permuted_output,
        &moe_scratch.expert_token_lists,
        &moe_scratch.expert_weight_lists,
        &moe_scratch.expert_counts,
        &moe_scratch.expert_offsets,
        stride,
        num_experts,
        max_per_expert_overall,
        hidden_size,
        batch,
        top_k,
        &mut moe_scratch.unpermute_rows,
        &mut moe_scratch.unpermute_wbits,
        &mut moe_scratch.unpermute_count,
        &mut moe_scratch.moe_acc,
    )?;

    // ── Small-expert dispatch via grouped t32_big NVFP4 GEMM ───────────
    // Previously this used the per-expert matvec path (3 launches per small
    // expert + gather/scatter), which dominated wall-time when 10-20 small
    // experts were live. The refactor below packs the small-expert subset
    // into a compact permuted layout and runs ONE grouped GEMM per
    // projection (gate / up / down) using the same `t32_big` kernel the
    // pure-grouped path already uses.
    //
    // Invariants:
    //   * `bulk_packed[pi]` / `bulk_scales[pi]` were staged for ALL active
    //     experts (large + small) at the top of the function; we just need
    //     to point at the small subset via fresh offset arrays.
    //   * `permuted_input` / `permuted_intermediate` / `permuted_swiglu`
    //     / `permuted_output` are free to reuse: the large-CUTLASS path
    //     completed its `unpermute_scatter_add` above, so no further reads
    //     from them are outstanding on the compute stream.
    //   * `expert_counts` / `expert_offsets` on the device are also free
    //     to overwrite (their large-path consumers have run).
    if !small_experts.is_empty() {
        let _ = staging_ptr; // not used in grouped small path

        // ── small-only host-side metadata ───────────────────────────────
        // small_counts[e] = counts[e] if e is small, else 0.
        let mut small_counts_host = vec![0u32; num_experts];
        for &e in &small_experts {
            small_counts_host[e] = counts_host[e];
        }
        // small_offsets[e+1] = small_offsets[e] + small_counts[e].
        // Each small expert's compact row range starts at small_offsets[e].
        let mut small_offsets_host = vec![0u32; num_experts + 1];
        for e in 0..num_experts {
            small_offsets_host[e + 1] = small_offsets_host[e] + small_counts_host[e];
        }
        let total_small_tokens = small_offsets_host[num_experts] as usize;

        // bulk_token_offsets for the grouped GEMM: contiguous prefix sum
        // over small_experts in order (matches the compact layout that
        // permute_gather will produce because each small expert's start
        // equals the sum of preceding small experts' counts).
        let mut small_bulk_tok_offsets_host: Vec<u32> =
            Vec::with_capacity(small_experts.len() + 1);
        small_bulk_tok_offsets_host.push(0);
        let mut max_small_count = 0usize;
        for &e in &small_experts {
            let c = counts_host[e];
            max_small_count = max_small_count.max(c as usize);
            let prev = *small_bulk_tok_offsets_host.last().unwrap();
            small_bulk_tok_offsets_host.push(prev + c);
        }
        let num_small = small_experts.len();

        // small_pos_in_active: positions of small experts in the active_experts
        // array (i.e. row index into bulk_packed / bulk_scales staging order).
        let mut small_pos_in_active: Vec<usize> = Vec::with_capacity(num_small);
        {
            let mut ae_iter = 0usize;
            for &ge in &small_experts {
                while ae_iter < active_experts.len() && active_experts[ae_iter] != ge {
                    ae_iter += 1;
                }
                small_pos_in_active.push(ae_iter);
                ae_iter += 1;
            }
        }

        // ── upload metadata ─────────────────────────────────────────────
        // Overwrite expert_counts / expert_offsets / bulk_token_offsets
        // (the large CUTLASS path has finished consuming them).
        runtime.upload_u32_slice_to_device(
            &small_counts_host,
            &mut moe_scratch.expert_counts,
        )?;
        runtime.upload_u32_slice_to_device(
            &small_offsets_host,
            &mut moe_scratch.expert_offsets,
        )?;
        runtime.upload_u32_slice_to_device(
            &small_bulk_tok_offsets_host,
            &mut moe_scratch.bulk_token_offsets,
        )?;

        // Per-projection: build small-only packed/scales offsets + output
        // scales (subset of the all-active arrays already on the device).
        for pi in 0..3 {
            let mut packed_offs: Vec<u32> = Vec::with_capacity(num_small);
            let mut scales_offs: Vec<u32> = Vec::with_capacity(num_small);
            let mut output_scales: Vec<f32> = Vec::with_capacity(num_small);
            for &spos in &small_pos_in_active {
                packed_offs.push(per_proj_packed_offsets_host[pi][spos] as u32);
                scales_offs.push(per_proj_scales_offsets_host[pi][spos] as u32);
                output_scales.push(per_proj_alpha_host[pi][spos]);
            }
            let slot = &mut moe_scratch.bulk_slots[pi];
            runtime.upload_u32_slice_to_device(&packed_offs, &mut slot.bulk_packed_offsets)?;
            runtime.upload_u32_slice_to_device(&scales_offs, &mut slot.bulk_scales_offsets)?;
            runtime.upload_f32_slice_to_device(&output_scales, &mut slot.bulk_output_scales)?;
        }

        // ── permute_gather: pack small experts' inputs contiguously into
        //    permuted_input (rows [0, total_small_tokens)). ───────────────
        runtime.permute_gather_f32_device(
            &moe_scratch.expert_input,
            &moe_scratch.expert_token_lists,
            &moe_scratch.expert_counts,
            &moe_scratch.expert_offsets,
            stride,
            num_experts,
            max_small_count,
            hidden_size,
            &mut moe_scratch.permuted_input,
        )?;

        // ── grouped GEMM: gate, up, GeGLU, down ─────────────────────────
        // Gate.
        {
            let slot = &moe_scratch.bulk_slots[0];
            runtime.matmul_nvfp4_grouped_prequant_wmma_bf16_device(
                &slot.bulk_packed, &slot.bulk_scales,
                &slot.bulk_packed_offsets, &slot.bulk_scales_offsets,
                &slot.bulk_output_scales, &moe_scratch.bulk_token_offsets,
                &moe_scratch.permuted_input,
                exp_intermediate, exp_cols_in, max_small_count, num_small,
                &mut moe_scratch.permuted_intermediate,
            )?;
        }
        // Up.
        {
            let slot = &moe_scratch.bulk_slots[1];
            runtime.matmul_nvfp4_grouped_prequant_wmma_bf16_device(
                &slot.bulk_packed, &slot.bulk_scales,
                &slot.bulk_packed_offsets, &slot.bulk_scales_offsets,
                &slot.bulk_output_scales, &moe_scratch.bulk_token_offsets,
                &moe_scratch.permuted_input,
                exp_intermediate, exp_cols_in, max_small_count, num_small,
                &mut moe_scratch.permuted_swiglu,
            )?;
        }
        // GeGLU(gate, up) → permuted_swiglu (in-place over up).
        runtime.geglu_tanh_in_place_device(
            &moe_scratch.permuted_intermediate,
            &mut moe_scratch.permuted_swiglu,
            total_small_tokens * exp_intermediate,
        )?;
        // Down.
        {
            let slot = &moe_scratch.bulk_slots[2];
            runtime.matmul_nvfp4_grouped_prequant_wmma_bf16_device(
                &slot.bulk_packed, &slot.bulk_scales,
                &slot.bulk_packed_offsets, &slot.bulk_scales_offsets,
                &slot.bulk_output_scales, &moe_scratch.bulk_token_offsets,
                &moe_scratch.permuted_swiglu,
                down_rows, down_cols, max_small_count, num_small,
                &mut moe_scratch.permuted_output,
            )?;
        }

        // ── unpermute_scatter_add: write small-expert contributions back
        //    into moe_acc using small-only counts/offsets. The serial kernel
        //    uses `+=`, so this composes with the large-expert call above.
        runtime.unpermute_scatter_add_f32_device(
            &moe_scratch.permuted_output,
            &moe_scratch.expert_token_lists,
            &moe_scratch.expert_weight_lists,
            &moe_scratch.expert_counts,
            &moe_scratch.expert_offsets,
            stride,
            num_experts,
            max_small_count,
            hidden_size,
            batch,
            top_k,
            &mut moe_scratch.unpermute_rows,
            &mut moe_scratch.unpermute_wbits,
            &mut moe_scratch.unpermute_count,
            &mut moe_scratch.moe_acc,
        )?;
    } else {
        let _ = staging_ptr; // suppress unused warning when no small experts
    }

    Ok(())
}

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
        _ => return Err(AegisError::InvalidPlan(
            "MoE prefill expects shared expert with all three projections in the same \
             format (BF16 or FP8)".into(),
        )),
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

    if use_grouped {
        forward_moe_grouped_routed_experts(
            runtime, moe, moe_scratch, &active_experts, &counts_host,
            stride, num_experts, hidden_size,
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
fn forward_moe_grouped_routed_experts(
    runtime: &CudaRuntime,
    moe: &CudaMoE,
    moe_scratch: &mut crate::executor::state::CudaMoEPrefillScratch,
    active_experts: &[usize],
    counts_host: &[u32],
    stride: usize,
    num_experts: usize,
    hidden_size: usize,
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

    // Inverse permute + weighted scatter into moe_acc.
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
        &mut moe_scratch.moe_acc,
    )?;

    Ok(())
}

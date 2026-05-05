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

    // ── Step 2: shared MLP (BF16, batched) ──────────────────────────────────
    // Reuses `gather_intermediate` (cs × max_intermediate) for gate, then
    // `gather_swiglu` for up. After geglu we run down → `gather_out`.
    let shared = moe.shared_expert.as_ref().ok_or_else(|| {
        AegisError::InvalidPlan(
            "Gemma 4 MoE prefill requires a shared MLP (mlp.gate/up/down_proj.weight)".into(),
        )
    })?;
    let shared_gate = shared.gate_proj.as_bf16().ok_or_else(|| {
        AegisError::InvalidPlan("MoE prefill expects BF16 shared MLP gate_proj".into())
    })?;
    let shared_up = shared.up_proj.as_bf16().ok_or_else(|| {
        AegisError::InvalidPlan("MoE prefill expects BF16 shared MLP up_proj".into())
    })?;
    let shared_down = shared.down_proj.as_bf16().ok_or_else(|| {
        AegisError::InvalidPlan("MoE prefill expects BF16 shared MLP down_proj".into())
    })?;
    let intermediate = shared_gate.rows;

    let cp_shared = mark("");
    // cuBLASLt BF16 GEMM (Phase C). Shared MLP weights are force-VRAM at load,
    // so the cublaslt path applies. Falls back to reference if any reason
    // VRAM-residency is unmet.
    if runtime.cublaslt_bf16_enabled_for(shared_gate) {
        runtime.matmul_bf16_cublaslt_device(
            shared_gate, &pf.input_normed, batch,
            &mut pf.bf16_in_scratch, &mut pf.bf16_out_scratch,
            &mut moe_scratch.gather_intermediate,
        )?;
        runtime.matmul_bf16_cublaslt_device(
            shared_up, &pf.input_normed, batch,
            &mut pf.bf16_in_scratch, &mut pf.bf16_out_scratch,
            &mut moe_scratch.gather_swiglu,
        )?;
        runtime.geglu_tanh_in_place_device(
            &moe_scratch.gather_intermediate,
            &mut moe_scratch.gather_swiglu,
            batch * intermediate,
        )?;
        runtime.matmul_bf16_cublaslt_device(
            shared_down, &moe_scratch.gather_swiglu, batch,
            &mut pf.bf16_in_scratch, &mut pf.bf16_out_scratch,
            &mut moe_scratch.gather_out,
        )?;
    } else {
        runtime.matmul_bf16_reference_batched_device(
            shared_gate, &pf.input_normed, batch, &mut moe_scratch.gather_intermediate,
        )?;
        runtime.matmul_bf16_reference_batched_device(
            shared_up, &pf.input_normed, batch, &mut moe_scratch.gather_swiglu,
        )?;
        runtime.geglu_tanh_in_place_device(
            &moe_scratch.gather_intermediate,
            &mut moe_scratch.gather_swiglu,
            batch * intermediate,
        )?;
        runtime.matmul_bf16_reference_batched_device(
            shared_down, &moe_scratch.gather_swiglu, batch, &mut moe_scratch.gather_out,
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

    // Phase 2 grouped path lights up when the VRAM expert cache is installed
    // (env-gated by AEGIS_VRAM_EXPERT_CACHE=1 in `from_artifact`). For cached
    // experts we run the full layer's gate/up/down via three single grouped
    // kernel launches; uncached experts fall back to the per-expert loop
    // immediately below. Both paths atomically accumulate into `moe_acc` so
    // they coexist without further synchronisation.
    let cache_handle = runtime.expert_cache().cloned();

    let max_count = counts_host.iter().copied().max().unwrap_or(0) as usize;

    // Build host-side per-layer offset arrays + cached/uncached partitioning.
    // Done unconditionally (cheap: ~128 HashMap lookups + 6 small Vec::push)
    // so we know which experts the grouped pass covers and which need the
    // fallback.
    // u64 byte offsets into the cache buffer — the buffer can exceed 4 GB
    // on Gemma-4-26B (~9 GB cache) so 32-bit offsets silently wrap around
    // layer ~10 and the kernel reads weights from the wrong layer.
    let mut gate_packed = vec![0u64; num_experts];
    let mut gate_scales = vec![0u64; num_experts];
    let mut up_packed = vec![0u64; num_experts];
    let mut up_scales = vec![0u64; num_experts];
    let mut down_packed = vec![0u64; num_experts];
    let mut down_scales = vec![0u64; num_experts];
    // Per-expert input/output scales for the three matmuls. Each NVFP4 expert
    // weight has its own pair, so we cannot use a single scalar across the
    // grouped GEMM — per-expert arrays are uploaded next to the offset arrays.
    let mut gate_in_scales = vec![0.0f32; num_experts];
    let mut gate_out_scales = vec![0.0f32; num_experts];
    let mut up_in_scales = vec![0.0f32; num_experts];
    let mut up_out_scales = vec![0.0f32; num_experts];
    let mut down_in_scales = vec![0.0f32; num_experts];
    let mut down_out_scales = vec![0.0f32; num_experts];
    let mut cached_counts_host = vec![0u32; num_experts];
    let mut uncached_experts: Vec<usize> = Vec::new();
    if let Some(cache) = cache_handle.as_ref() {
        for e in 0..num_experts {
            let count = counts_host[e];
            if count == 0 {
                continue;
            }
            let expert = &moe.experts[e];
            let g = cache.get(&expert.gate_proj.name);
            let u = cache.get(&expert.up_proj.name);
            let d = cache.get(&expert.down_proj.name);
            if let (Some(g), Some(u), Some(d)) = (g, u, d) {
                gate_packed[e] = g.packed_offset as u64;
                gate_scales[e] = g.scales_offset as u64;
                up_packed[e] = u.packed_offset as u64;
                up_scales[e] = u.scales_offset as u64;
                down_packed[e] = d.packed_offset as u64;
                down_scales[e] = d.scales_offset as u64;
                gate_in_scales[e] = expert.gate_proj.input_scale;
                gate_out_scales[e] = expert.gate_proj.output_scale;
                up_in_scales[e] = expert.up_proj.input_scale;
                up_out_scales[e] = expert.up_proj.output_scale;
                down_in_scales[e] = expert.down_proj.input_scale;
                down_out_scales[e] = expert.down_proj.output_scale;
                cached_counts_host[e] = count;
            } else {
                uncached_experts.push(e);
            }
        }
    } else {
        // No cache → everything goes through the per-expert fallback.
        for e in 0..num_experts {
            if counts_host[e] > 0 {
                uncached_experts.push(e);
            }
        }
    }

    // Phase 2 grouped MoE prefill is opt-in via AEGIS_GROUPED_MOE_ENABLE=1.
    // When disabled (default), every triggered expert goes through the
    // per-expert fallback path (which transparently uses cache views or the
    // staging pool depending on whether the expert weight is in the cache).
    //
    // The grouped kernels (gate/up/geglu/down) produce numerically correct
    // outputs per-expert and the resulting `moe_acc` matches a fallback
    // recompute to ~1e-6, but end-to-end generation diverges from the
    // fallback baseline — the source of that divergence is not yet pinned
    // down. Until it is, default behaviour is the verified-correct fallback.
    let grouped_enabled = std::env::var("AEGIS_GROUPED_MOE_ENABLE").is_ok();
    let cache_active = grouped_enabled
        && cache_handle.is_some()
        && cached_counts_host.iter().any(|&c| c > 0);
    if !grouped_enabled {
        uncached_experts.clear();
        for e in 0..num_experts {
            if counts_host[e] > 0 {
                uncached_experts.push(e);
            }
        }
    }

    if cache_active {
        let cache = cache_handle.as_ref().unwrap();
        // Upload per-layer offset arrays + cached_counts to the device.
        runtime.upload_u32_slice_to_device(&cached_counts_host, &mut moe_scratch.cached_counts)?;
        runtime.upload_u64_slice_to_device(&gate_packed, &mut moe_scratch.gate_packed_offsets)?;
        runtime.upload_u64_slice_to_device(&gate_scales, &mut moe_scratch.gate_scales_offsets)?;
        runtime.upload_u64_slice_to_device(&up_packed, &mut moe_scratch.up_packed_offsets)?;
        runtime.upload_u64_slice_to_device(&up_scales, &mut moe_scratch.up_scales_offsets)?;
        runtime.upload_u64_slice_to_device(&down_packed, &mut moe_scratch.down_packed_offsets)?;
        runtime.upload_u64_slice_to_device(&down_scales, &mut moe_scratch.down_scales_offsets)?;
        runtime.upload_f32_slice_to_device(&gate_in_scales, &mut moe_scratch.gate_input_scales)?;
        runtime.upload_f32_slice_to_device(&gate_out_scales, &mut moe_scratch.gate_output_scales)?;
        runtime.upload_f32_slice_to_device(&up_in_scales, &mut moe_scratch.up_input_scales)?;
        runtime.upload_f32_slice_to_device(&up_out_scales, &mut moe_scratch.up_output_scales)?;
        runtime.upload_f32_slice_to_device(&down_in_scales, &mut moe_scratch.down_input_scales)?;
        runtime.upload_f32_slice_to_device(&down_out_scales, &mut moe_scratch.down_output_scales)?;

        // CSR prefix sum over FULL expert_counts: lays out the permuted
        // activation buffer covering *all* experts (cached + uncached). The
        // grouped GEMM only writes outputs for cached entries (cached_counts
        // is 0 for uncached), and unpermute_scatter_add only contributes
        // from cached entries — uncached rows stay zero in permuted_down.
        runtime.router_expert_offsets_device(
            &moe_scratch.expert_counts,
            num_experts,
            &mut moe_scratch.expert_offsets,
        )?;

        // Permute gather: scatter expert_input rows into permuted_input by
        // expert. Uses full counts because the same buffer is referenced
        // from both grouped (cached) and fallback (uncached) paths — but
        // fallback reads expert_input directly, so this is technically
        // wasted for uncached. Acceptable for first iteration.
        runtime.permute_gather_f32_device(
            &moe_scratch.expert_input,
            &moe_scratch.expert_token_lists,
            &moe_scratch.expert_counts,
            &moe_scratch.expert_offsets,
            stride,
            num_experts,
            max_count,
            hidden_size,
            &mut moe_scratch.permuted_input,
        )?;

        // Discover the expert intermediate dim from any cached expert.
        let exp_intermediate = moe
            .experts
            .iter()
            .find(|_| true)
            .map(|e| e.gate_proj.rows)
            .ok_or_else(|| {
                aegisllm_base::error::AegisError::InvalidPlan(
                    "MoE layer has no experts (grouped path)".into(),
                )
            })?;

        // Pipeline (mirrors vLLM `fused_moe` / TRT-LLM permuted-activation
        // grouped GEMM):
        //   1. NVFP4-quantize permuted_input with each cached expert's
        //      gate.input_scale → permuted_input_quant.
        //   2. Grouped prequant GEMM (8-warp tiled, shared-mem weight load,
        //      one launch handles all cached experts) → permuted_gate.
        //   3. Re-quantize permuted_input with each expert's up.input_scale
        //      and run grouped GEMM → permuted_up.
        //   4. GeGLU(permuted_gate, permuted_up) → permuted_up (in-place).
        //   5. Quantize permuted_up with each expert's down.input_scale and
        //      run grouped GEMM (rows=hidden, cols=intermediate) → permuted_down.
        let total_assignments = batch * top_k;
        let _ = total_assignments;

        // Step 1+2: gate
        runtime.nvfp4_quantize_input_per_expert_device(
            &moe_scratch.permuted_input,
            &moe_scratch.cached_counts,
            &moe_scratch.expert_offsets,
            &moe_scratch.gate_input_scales,
            hidden_size,
            num_experts,
            max_count,
            &mut moe_scratch.permuted_input_quant,
        )?;
        runtime.nvfp4_grouped_prequant_gemm_device(
            cache.buffer(),
            &moe_scratch.gate_packed_offsets,
            cache.buffer(),
            &moe_scratch.gate_scales_offsets,
            &moe_scratch.cached_counts,
            &moe_scratch.expert_offsets,
            &moe_scratch.gate_output_scales,
            exp_intermediate,
            hidden_size,
            &moe_scratch.permuted_input_quant,
            &mut moe_scratch.permuted_gate,
            num_experts,
            max_count,
        )?;
        // Step 3: up (re-quantize input with up.input_scale, then grouped GEMM)
        runtime.nvfp4_quantize_input_per_expert_device(
            &moe_scratch.permuted_input,
            &moe_scratch.cached_counts,
            &moe_scratch.expert_offsets,
            &moe_scratch.up_input_scales,
            hidden_size,
            num_experts,
            max_count,
            &mut moe_scratch.permuted_input_quant,
        )?;
        runtime.nvfp4_grouped_prequant_gemm_device(
            cache.buffer(),
            &moe_scratch.up_packed_offsets,
            cache.buffer(),
            &moe_scratch.up_scales_offsets,
            &moe_scratch.cached_counts,
            &moe_scratch.expert_offsets,
            &moe_scratch.up_output_scales,
            exp_intermediate,
            hidden_size,
            &moe_scratch.permuted_input_quant,
            &mut moe_scratch.permuted_up,
            num_experts,
            max_count,
        )?;

        // Step 4: GeGLU(gate, up) → permuted_up (in-place). Iterates flat
        // over total_assignments × intermediate; positions for non-cached
        // experts get processed but later down/scatter steps skip them via
        // cached_counts so the garbage never contributes to moe_acc.
        runtime.geglu_tanh_in_place_device(
            &moe_scratch.permuted_gate,
            &mut moe_scratch.permuted_up,
            batch * top_k * exp_intermediate,
        )?;

        // Step 5: down (quantize permuted_up with down.input_scale, grouped GEMM)
        runtime.nvfp4_quantize_input_per_expert_device(
            &moe_scratch.permuted_up,
            &moe_scratch.cached_counts,
            &moe_scratch.expert_offsets,
            &moe_scratch.down_input_scales,
            exp_intermediate,
            num_experts,
            max_count,
            &mut moe_scratch.permuted_input_quant,
        )?;
        runtime.nvfp4_grouped_prequant_gemm_device(
            cache.buffer(),
            &moe_scratch.down_packed_offsets,
            cache.buffer(),
            &moe_scratch.down_scales_offsets,
            &moe_scratch.cached_counts,
            &moe_scratch.expert_offsets,
            &moe_scratch.down_output_scales,
            hidden_size,
            exp_intermediate,
            &moe_scratch.permuted_input_quant,
            &mut moe_scratch.permuted_down,
            num_experts,
            max_count,
        )?;
        // Per-expert serial scatter: matches the per-expert fallback's
        // accumulation order bit-exactly so Gemma 4's top-k router sees the
        // same logits as the cache-on/grouped-disabled path. (A single fused
        // unpermute_scatter_add kernel relies on atomicAdd ordering, which
        // produces 1-ULP per-element drift that the router amplifies into
        // flipped expert selections within ~3 layers.)
        //
        // Each expert's permuted_down slice and expert_token_lists/
        // expert_weight_lists slice is already contiguous in VRAM, so we
        // launch one `scatter_add_weighted_f32_views` per cached expert
        // pointing directly at those views — no copies, just one kernel per
        // expert. The `counts_host` prefix sum is computed host-side so we
        // don't need to download `expert_offsets`.
        let mut off_host: usize = 0;
        for e in 0..num_experts {
            let count = cached_counts_host[e] as usize;
            let full_count = counts_host[e] as usize;
            if count == 0 {
                off_host += full_count;
                continue;
            }
            let bucket_off = e * stride;
            let perm_row_off = off_host * hidden_size;
            runtime.scatter_add_weighted_f32_subslice(
                &moe_scratch.permuted_down, perm_row_off,
                &moe_scratch.expert_token_lists, bucket_off,
                &moe_scratch.expert_weight_lists, bucket_off,
                count, hidden_size,
                &mut moe_scratch.moe_acc,
            )?;
            off_host += full_count;
        }
    }

    // Per-expert fallback: covers all experts when no cache is installed,
    // and only uncached experts when the grouped pass ran above.
    for &expert_idx in &uncached_experts {
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

use std::time::Instant;

use super::gemm::{
    prefill_gate_up_mxfp4_native_device, prefill_linear_batched_device_with_scratch,
    prefill_linear_cutlass_nvfp4_device, prefill_linear_cutlass_nvfp4_enabled,
    prefill_linear_cutlass_nvfp4_prepacked_device, prefill_linear_native_mxfp4_enabled,
    prefill_linear_prepare_nvfp4_input, prefill_linear_prepared_batched_device,
    prefill_qkv_mxfp4_native_device,
};
use super::timings::record_prefill_stage;
use crate::cuda::{CudaRuntime, DensePrefillMetadataProof};
use crate::cuda::staging::LinearStagingPool;
use aegisllm_base::error::{AegisError, Result};
use crate::executor::state::{
    CudaLayer, CudaLayerState, CudaLinear, CudaPrefillScratch, CudaPrefillStageTimings,
    KvStagingPool, KvStagingSlot,
};

#[derive(Debug, Clone, Copy)]
pub(super) struct CudaPrefillForwardParams {
    pub(super) rms_norm_eps: f32,
    pub(super) start_position: usize,
    pub(super) batch: usize,
    pub(super) num_sequences: usize,
    pub(super) dense_metadata: DensePrefillMetadataProof,
    pub(super) num_attention_heads: usize,
    pub(super) num_kv_heads: usize,
    pub(super) head_dim: usize,
    pub(super) kv_context_size: usize,
    /// Raw pointer to the shared linear staging pool; null when no staged layers exist.
    /// Using a raw pointer avoids lifetime entanglement with `CudaLlamaState` fields
    /// that are already mutably borrowed when this params struct is constructed.
    pub(super) staging_ptr: *mut LinearStagingPool,
    /// Raw pointer to the shared KV staging slot; null when KV is VRAM-resident.
    pub(super) kv_staging_ptr: *mut KvStagingPool,
}

pub(super) fn forward_cuda_layer_prefill_chunk_device(
    runtime: &CudaRuntime,
    layer: &CudaLayer,
    layer_state: &mut CudaLayerState,
    prefill: &mut CudaPrefillScratch,
    params: CudaPrefillForwardParams,
    timings: &mut CudaPrefillStageTimings,
) -> Result<()> {
    let hidden_size = layer.o_proj.rows();
    // For MoE layers, gate_proj is a dummy (rows=1). The real intermediate lives in
    // CudaMoE::expert_intermediate_size, but prefill MoE is guarded below.
    let intermediate = if layer.moe.is_some() { 1 } else { layer.gate_proj.rows };
    // Prefill scratch is lifetime-pooled manually here: Q lives in `gate`
    // until attention finishes, K lives in `up`, and attention context lives
    // in `qkv` after QKV split has consumed it. MLP overwrites gate/up later.
    let qkv_start = Instant::now();
    // Dispatch: BF16 path if q_proj is BF16 (e.g. Gemma4), NVFP4 path otherwise.
    if let (Some(q_nvfp4), Some(k_nvfp4), Some(v_nvfp4)) = (
        layer.q_proj.as_nvfp4(), layer.k_proj.as_nvfp4(), layer.v_proj.as_nvfp4(),
    ) {
        // NVFP4 path (Llama, Qwen, etc.)
        let qkv_fused = layer.qkv_proj.as_ref()
            .and_then(|l| l.as_nvfp4())
            .filter(|l| prefill_linear_cutlass_nvfp4_enabled(runtime, l));
        if let Some(qkv_proj) = qkv_fused {
            runtime.rms_norm_batched_device(
                &prefill.hidden,
                &layer.input_norm_weight,
                params.batch,
                params.rms_norm_eps,
                &mut prefill.input_normed,
            )?;
            runtime.quantize_cutlass_nvfp4_activation_device(
                &prefill.input_normed,
                params.batch,
                qkv_proj.cols,
                &mut prefill.cutlass_payload,
                &mut prefill.cutlass_scales,
            )?;
            prefill_linear_cutlass_nvfp4_prepacked_device(
                runtime,
                qkv_proj,
                &prefill.cutlass_payload,
                &prefill.cutlass_scales,
                params.batch,
                &mut prefill.cutlass_workspace,
                &mut prefill.qkv,
            )?;
            runtime.split_qkv_scaled_device(
                &prefill.qkv,
                params.batch,
                q_nvfp4.rows,
                k_nvfp4.rows,
                q_nvfp4.output_scale,
                k_nvfp4.output_scale,
                v_nvfp4.output_scale,
                &mut prefill.gate,
                &mut prefill.up,
                &mut prefill.v,
            )?;
        } else if prefill_linear_native_mxfp4_enabled(runtime, q_nvfp4)
            && prefill_linear_native_mxfp4_enabled(runtime, k_nvfp4)
            && prefill_linear_native_mxfp4_enabled(runtime, v_nvfp4)
        {
            runtime.rms_norm_batched_device(
                &prefill.hidden,
                &layer.input_norm_weight,
                params.batch,
                params.rms_norm_eps,
                &mut prefill.input_normed,
            )?;
            runtime.quantize_mxfp4_input_batched_device(
                &prefill.input_normed,
                params.batch,
                q_nvfp4.cols,
                &mut prefill.mxfp4_hidden,
            )?;
            prefill_qkv_mxfp4_native_device(
                runtime,
                q_nvfp4,
                k_nvfp4,
                v_nvfp4,
                &prefill.mxfp4_hidden,
                params.batch,
                &mut prefill.gate,
                &mut prefill.up,
                &mut prefill.v,
            )?;
        } else if prefill_linear_cutlass_nvfp4_enabled(runtime, q_nvfp4)
            && prefill_linear_cutlass_nvfp4_enabled(runtime, k_nvfp4)
            && prefill_linear_cutlass_nvfp4_enabled(runtime, v_nvfp4)
        {
            runtime.rms_norm_batched_device(
                &prefill.hidden,
                &layer.input_norm_weight,
                params.batch,
                params.rms_norm_eps,
                &mut prefill.input_normed,
            )?;
            runtime.quantize_cutlass_nvfp4_activation_device(
                &prefill.input_normed,
                params.batch,
                q_nvfp4.cols,
                &mut prefill.cutlass_payload,
                &mut prefill.cutlass_scales,
            )?;
            prefill_linear_cutlass_nvfp4_prepacked_device(
                runtime,
                q_nvfp4,
                &prefill.cutlass_payload,
                &prefill.cutlass_scales,
                params.batch,
                &mut prefill.cutlass_workspace,
                &mut prefill.gate,
            )?;
            prefill_linear_cutlass_nvfp4_prepacked_device(
                runtime,
                k_nvfp4,
                &prefill.cutlass_payload,
                &prefill.cutlass_scales,
                params.batch,
                &mut prefill.cutlass_workspace,
                &mut prefill.up,
            )?;
            prefill_linear_cutlass_nvfp4_prepacked_device(
                runtime,
                v_nvfp4,
                &prefill.cutlass_payload,
                &prefill.cutlass_scales,
                params.batch,
                &mut prefill.cutlass_workspace,
                &mut prefill.v,
            )?;
        } else {
            // SAFETY: staging_ptr points to scratch.staging_pool which lives at least as long as this fn.
            // We reborrow it as `&mut` individually per call; only one reborrow is alive at a time.
            let sp = params.staging_ptr;
            runtime.rms_norm_quant_nvfp4_batched_device(
                &prefill.hidden,
                &layer.input_norm_weight,
                params.batch,
                params.rms_norm_eps,
                q_nvfp4.input_scale,
                &mut prefill.input_normed,
                &mut prefill.quant_hidden,
            )?;
            prefill_linear_prepared_batched_device(
                runtime,
                q_nvfp4,
                &prefill.input_normed,
                &prefill.quant_hidden,
                params.batch,
                &mut prefill.mxfp4_hidden,
                &mut prefill.gate,
                if sp.is_null() { None } else { Some(unsafe { &mut *sp }) },
            )?;
            let mut quant_scale = Some(q_nvfp4.input_scale);
            prefill_linear_prepare_nvfp4_input(
                runtime,
                k_nvfp4,
                &prefill.input_normed,
                params.batch,
                &mut quant_scale,
                &mut prefill.quant_hidden,
            )?;
            prefill_linear_prepared_batched_device(
                runtime,
                k_nvfp4,
                &prefill.input_normed,
                &prefill.quant_hidden,
                params.batch,
                &mut prefill.mxfp4_hidden,
                &mut prefill.up,
                if sp.is_null() { None } else { Some(unsafe { &mut *sp }) },
            )?;
            prefill_linear_prepare_nvfp4_input(
                runtime,
                v_nvfp4,
                &prefill.input_normed,
                params.batch,
                &mut quant_scale,
                &mut prefill.quant_hidden,
            )?;
            prefill_linear_prepared_batched_device(
                runtime,
                v_nvfp4,
                &prefill.input_normed,
                &prefill.quant_hidden,
                params.batch,
                &mut prefill.mxfp4_hidden,
                &mut prefill.v,
                if sp.is_null() { None } else { Some(unsafe { &mut *sp }) },
            )?;
        }
    } else {
        // BF16 attention path (Gemma 4 26B): batched matmul Q/K/V with VRAM-resident
        // weights (host-resident BF16 attention is forced to VRAM at load time). All
        // attention norms are run batched — q_norm, k_norm, v_norm (no learned weight).
        let q_bf16 = layer.q_proj.as_bf16().ok_or_else(|| AegisError::InvalidPlan(
            "BF16 attention prefill: q_proj must be BF16".into()))?;
        let k_bf16 = layer.k_proj.as_bf16().ok_or_else(|| AegisError::InvalidPlan(
            "BF16 attention prefill: k_proj must be BF16".into()))?;
        let v_bf16 = layer.v_proj.as_bf16().ok_or_else(|| AegisError::InvalidPlan(
            "BF16 attention prefill: v_proj must be BF16".into()))?;
        runtime.rms_norm_batched_device(
            &prefill.hidden,
            &layer.input_norm_weight,
            params.batch,
            params.rms_norm_eps,
            &mut prefill.input_normed,
        )?;
        // Phase C: cuBLASLt BF16 GEMM for Q/K/V projections. Falls back to
        // reference if any of the three is host-resident (shouldn't happen in
        // practice — Gemma 4 attention BF16 weights are force-VRAM).
        if runtime.cublaslt_bf16_enabled_for(q_bf16)
            && runtime.cublaslt_bf16_enabled_for(k_bf16)
            && runtime.cublaslt_bf16_enabled_for(v_bf16)
        {
            runtime.matmul_bf16_cublaslt_device(
                q_bf16, &prefill.input_normed, params.batch,
                &mut prefill.bf16_in_scratch, &mut prefill.bf16_out_scratch,
                &mut prefill.gate,
            )?;
            runtime.matmul_bf16_cublaslt_device(
                k_bf16, &prefill.input_normed, params.batch,
                &mut prefill.bf16_in_scratch, &mut prefill.bf16_out_scratch,
                &mut prefill.up,
            )?;
            runtime.matmul_bf16_cublaslt_device(
                v_bf16, &prefill.input_normed, params.batch,
                &mut prefill.bf16_in_scratch, &mut prefill.bf16_out_scratch,
                &mut prefill.v,
            )?;
        } else {
            runtime.matmul_bf16_reference_batched_device(
                q_bf16, &prefill.input_normed, params.batch, &mut prefill.gate,
            )?;
            runtime.matmul_bf16_reference_batched_device(
                k_bf16, &prefill.input_normed, params.batch, &mut prefill.up,
            )?;
            runtime.matmul_bf16_reference_batched_device(
                v_bf16, &prefill.input_normed, params.batch, &mut prefill.v,
            )?;
        }
        // Per-head q_norm/k_norm (with weight) + v_norm (no weight). Each per-head
        // RMS norm acts on a row of head_dim values across (batch * num_heads) rows.
        if let Some(ref qnw) = layer.q_norm_weight {
            // Treat scratch.gate as (batch * num_heads) rows of head_dim.
            runtime.rms_norm_batched_device(
                &prefill.gate,
                qnw,
                params.batch * params.num_attention_heads,
                params.rms_norm_eps,
                &mut prefill.qkv,  // scratch reuse
            )?;
            // Copy back into prefill.gate.
            runtime.copy_prefix_f32_device(
                &prefill.qkv,
                &mut prefill.gate,
                params.batch * layer.q_proj.rows(),
            )?;
        }
        if let Some(ref knw) = layer.k_norm_weight {
            runtime.rms_norm_batched_device(
                &prefill.up,
                knw,
                params.batch * layer.layer_num_kv_heads,
                params.rms_norm_eps,
                &mut prefill.qkv,
            )?;
            runtime.copy_prefix_f32_device(
                &prefill.qkv,
                &mut prefill.up,
                params.batch * layer.k_proj.rows(),
            )?;
        }
        if layer.q_norm_weight.is_some() {
            // V norm with no weight (Gemma 4 v_norm has_weight=False).
            runtime.rms_norm_batched_no_weight_device(
                &prefill.v,
                params.batch * layer.layer_num_kv_heads,
                layer.layer_head_dim,
                params.rms_norm_eps,
                &mut prefill.qkv,
            )?;
            runtime.copy_prefix_f32_device(
                &prefill.qkv,
                &mut prefill.v,
                params.batch * layer.v_proj.rows(),
            )?;
        }
        if let Ok(tag) = std::env::var("AEGIS_DUMP_QKV") {
            let q = runtime.download_f32(&prefill.gate).unwrap();
            let k = runtime.download_f32(&prefill.up).unwrap();
            let v = runtime.download_f32(&prefill.v).unwrap();
            eprintln!("[DUMP {tag} Q post-norm] first8={:?}", &q[0..8]);
            eprintln!("[DUMP {tag} K post-norm] first8={:?}", &k[0..8]);
            eprintln!("[DUMP {tag} V post-norm] first8={:?}", &v[0..8]);
        }
        // Compensate kernel's 1/sqrt(d) so effective scaling = 1.0 (Gemma 4).
        if layer.q_norm_weight.is_some() {
            let sqrt_d = (layer.layer_head_dim as f32).sqrt();
            runtime.scale_f32_device_len(
                sqrt_d,
                &mut prefill.gate,
                params.batch * layer.q_proj.rows(),
            )?;
        }
    }
    let qkv_flops = prefill_gemm_flops(
        params.batch,
        layer.q_proj.rows() + layer.k_proj.rows() + layer.v_proj.rows(),
        layer.q_proj.cols(),
    );
    record_prefill_stage(runtime, timings, qkv_start, |timings, elapsed| {
        timings.qkv_us += elapsed;
        timings.qkv_tflops = timings.qkv_tflops.max(tflops(qkv_flops, elapsed));
    })?;

    let rope_start = Instant::now();
    runtime.apply_rope_positions_batched_f16_out_device(
        &mut prefill.gate,
        &prefill.positions,
        params.batch,
        params.num_attention_heads,
        layer.layer_head_dim,
        layer.rope,
        &mut prefill.q_half,
    )?;
    record_prefill_stage(runtime, timings, rope_start, |timings, elapsed| {
        timings.rope_us += elapsed
    })?;
    if let Ok(tag) = std::env::var("AEGIS_DUMP_QROPE") {
        let q = runtime.download_f32(&prefill.gate).unwrap();
        let head_dim = layer.layer_head_dim;
        let q_width = params.num_attention_heads * head_dim;
        eprintln!("[DUMP {tag} Q rope tok0_h0] {:?}", &q[0..8]);
        if params.batch >= 2 {
            eprintln!("[DUMP {tag} Q rope tok1_h0] {:?}", &q[q_width..q_width+8]);
        }
    }

    let kv_store_start = Instant::now();
    let kv_is_host = layer_state.kv.is_host_resident();
    // Prefill uses a single staging slot (slot 0). Async transfer pipelining is
    // currently a decode-only optimization; prefill remains on the synchronous
    // upload/writeback path here.
    let mut kv_staging: Option<&mut KvStagingSlot> = if kv_is_host && !params.kv_staging_ptr.is_null() {
        let pool = unsafe { &mut *params.kv_staging_ptr };
        Some(&mut pool.slots[0])
    } else {
        None
    };

    if let Some(ref mut staging) = kv_staging.as_deref_mut().filter(|_| kv_is_host) {
        // Host-resident KV: upload existing entries, store into staging, writeback new batch.
        let kv_width = layer.layer_num_kv_heads * layer.layer_head_dim;
        {
            let host = layer_state.kv.host.as_ref().unwrap();
            runtime.upload_kv_slice_device(
                &mut staging.keys,
                &host.keys,
                params.start_position * kv_width,
            )?;
            runtime.upload_kv_slice_device(
                &mut staging.values,
                &host.values,
                params.start_position * kv_width,
            )?;
        }
        runtime.store_kv_slots_batched_rope_key_device(
            &mut staging.keys,
            &mut staging.values,
            &mut prefill.up,
            &prefill.v,
            &prefill.positions,
            &prefill.slot_mapping,
            params.batch,
            layer.layer_num_kv_heads,
            layer.layer_head_dim,
            params.kv_context_size,
            params.dense_metadata,
            layer.rope,
        )?;
        record_prefill_stage(runtime, timings, kv_store_start, |timings, elapsed| {
            timings.kv_store_us += elapsed
        })?;

        let attention_start = Instant::now();
        runtime.attention_prefill_dense_compat_device(
            &staging.keys,
            &staging.values,
            &prefill.up,
            &prefill.v,
            &prefill.gate,
            &mut prefill.q_half,
            true,
            &mut prefill.attn_split_acc,
            &mut prefill.attn_split_m,
            &mut prefill.attn_split_l,
            &prefill.slot_mapping,
            &prefill.cu_q,
            &prefill.cu_k,
            &prefill.context_lens,
            &prefill.block_tables,
            params.num_sequences,
            params.start_position,
            params.batch,
            params.num_attention_heads,
            layer.layer_num_kv_heads,
            layer.layer_head_dim,
            &mut prefill.qkv,
            params.dense_metadata,
        )?;
        record_prefill_stage(runtime, timings, attention_start, |timings, elapsed| {
            timings.attention_us += elapsed
        })?;

        // Writeback the newly-stored batch to host pinned RAM.
        let kv_width = layer.layer_num_kv_heads * layer.layer_head_dim;
        let host = layer_state.kv.host.as_mut().unwrap();
        runtime.writeback_kv_batch_device(
            &mut host.keys,
            &staging.keys,
            params.start_position,
            params.batch,
            kv_width,
        )?;
        runtime.writeback_kv_batch_device(
            &mut host.values,
            &staging.values,
            params.start_position,
            params.batch,
            kv_width,
        )?;
    } else {
        runtime.store_kv_slots_batched_rope_key_device(
            &mut layer_state.kv.keys,
            &mut layer_state.kv.values,
            &mut prefill.up,
            &prefill.v,
            &prefill.positions,
            &prefill.slot_mapping,
            params.batch,
            layer.layer_num_kv_heads,
            layer.layer_head_dim,
            params.kv_context_size,
            params.dense_metadata,
            layer.rope,
        )?;
        record_prefill_stage(runtime, timings, kv_store_start, |timings, elapsed| {
            timings.kv_store_us += elapsed
        })?;

        let attention_start = Instant::now();
        runtime.attention_prefill_dense_compat_device(
            &layer_state.kv.keys,
            &layer_state.kv.values,
            &prefill.up,
            &prefill.v,
            &prefill.gate,
            &mut prefill.q_half,
            true,
            &mut prefill.attn_split_acc,
            &mut prefill.attn_split_m,
            &mut prefill.attn_split_l,
            &prefill.slot_mapping,
            &prefill.cu_q,
            &prefill.cu_k,
            &prefill.context_lens,
            &prefill.block_tables,
            params.num_sequences,
            params.start_position,
            params.batch,
            params.num_attention_heads,
            layer.layer_num_kv_heads,
            layer.layer_head_dim,
            &mut prefill.qkv,
            params.dense_metadata,
        )?;
        record_prefill_stage(runtime, timings, attention_start, |timings, elapsed| {
            timings.attention_us += elapsed
        })?;
    }

    if let Ok(tag) = std::env::var("AEGIS_DUMP_ATTNOUT") {
        thread_local! { static AC: std::cell::RefCell<usize> = std::cell::RefCell::new(0); }
        let target = std::env::var("AEGIS_DUMP_ATTNOUT_LAYER")
            .ok().and_then(|s| s.parse::<usize>().ok());
        let idx = AC.with(|c| { let v = *c.borrow(); *c.borrow_mut() = v + 1; v });
        if target.is_none() || target == Some(idx) {
            let q = runtime.download_f32(&prefill.qkv).unwrap();
            let attn_width = params.num_attention_heads * layer.layer_head_dim;
            eprintln!("[DUMP {tag} L{} attn_out tok0]={:?}", idx, &q[0..8]);
            if params.batch >= 2 {
                eprintln!("[DUMP {tag} L{} attn_out tok1]={:?}", idx, &q[attn_width..attn_width+8]);
            }
        }
    }
    let o_proj_start = Instant::now();
    match &layer.o_proj {
        CudaLinear::Nvfp4(o) if prefill_linear_cutlass_nvfp4_enabled(runtime, o) => {
            prefill_linear_cutlass_nvfp4_device(
                runtime,
                o,
                &prefill.qkv,
                params.batch,
                &mut prefill.cutlass_payload,
                &mut prefill.cutlass_scales,
                &mut prefill.cutlass_workspace,
                &mut prefill.input_normed,
            )?;
        }
        CudaLinear::Nvfp4(o) => {
            let sp = params.staging_ptr;
            prefill_linear_batched_device_with_scratch(
                runtime,
                o,
                &prefill.qkv,
                params.batch,
                &mut prefill.quant_hidden,
                &mut prefill.mxfp4_hidden,
                &mut prefill.input_normed,
                if sp.is_null() { None } else { Some(unsafe { &mut *sp }) },
            )?;
        }
        CudaLinear::Bf16(o) => {
            // Gemma 4 BF16 o_proj. Phase C: cuBLASLt BF16 GEMM.
            if runtime.cublaslt_bf16_enabled_for(o) {
                runtime.matmul_bf16_cublaslt_device(
                    o, &prefill.qkv, params.batch,
                    &mut prefill.bf16_in_scratch, &mut prefill.bf16_out_scratch,
                    &mut prefill.input_normed,
                )?;
            } else {
                runtime.matmul_bf16_reference_batched_device(
                    o, &prefill.qkv, params.batch, &mut prefill.input_normed,
                )?;
            }
        }
    }
    record_prefill_stage(runtime, timings, o_proj_start, |timings, elapsed| {
        timings.o_proj_us += elapsed
    })?;

    let batch_hidden = params.batch * hidden_size;
    if prefill.hidden.len() < batch_hidden || prefill.input_normed.len() < batch_hidden {
        return Err(AegisError::InvalidPlan(
            "CUDA prefill hidden scratch is too small".into(),
        ));
    }
    if let Some(ref post_norm) = layer.post_attn_sublayer_norm {
        // Gemma 4 PrePost: normalize attention output (post-o_proj) before adding to residual.
        // Mirrors the decode path in executor/attention.rs.
        if prefill.qkv.len() < batch_hidden {
            return Err(AegisError::InvalidPlan(
                "CUDA prefill qkv scratch too small for post-attn norm".into(),
            ));
        }
        runtime.rms_norm_batched_device(
            &prefill.input_normed,
            post_norm,
            params.batch,
            params.rms_norm_eps,
            &mut prefill.qkv,
        )?;
        runtime.add_inplace_device_len(&mut prefill.hidden, &prefill.qkv, batch_hidden)?;
    } else {
        runtime.add_inplace_device_len(&mut prefill.hidden, &prefill.input_normed, batch_hidden)?;
    }

    if let Ok(layer_str) = std::env::var("AEGIS_DUMP_LAYER") {
        if let Ok(target_layer) = layer_str.parse::<usize>() {
            let tag = std::env::var("AEGIS_DUMP_TAG").unwrap_or_else(|_| "?".into());
            thread_local! {
                static CALL_COUNT: std::cell::RefCell<usize> = std::cell::RefCell::new(0);
            }
            CALL_COUNT.with(|c| {
                let mut c = c.borrow_mut();
                if *c == target_layer {
                    let h = runtime.download_f32(&prefill.hidden).unwrap();
                    eprintln!(
                        "[DUMP {tag} L{}] post_attn_residual tok0={:?}",
                        *c,
                        &h[0..8.min(h.len())],
                    );
                    if params.batch >= 2 && h.len() >= hidden_size + 8 {
                        eprintln!(
                            "[DUMP {tag} L{}] post_attn_residual tok1={:?}",
                            *c,
                            &h[hidden_size..hidden_size + 8],
                        );
                    }
                }
                *c += 1;
            });
        }
    }

    let mlp_start = Instant::now();
    if let Some(ref moe) = layer.moe {
        super::moe::forward_moe_prefill_chunk_device(
            runtime,
            layer,
            moe,
            prefill,
            params.batch,
            hidden_size,
            params.rms_norm_eps,
            params.staging_ptr,
            timings,
        )?;
        record_prefill_stage(runtime, timings, mlp_start, |timings, elapsed| {
            timings.mlp_us += elapsed
        })?;
        return Ok(());
    }
    if prefill_linear_cutlass_nvfp4_enabled(runtime, &layer.gate_proj)
        && prefill_linear_cutlass_nvfp4_enabled(runtime, &layer.up_proj)
    {
        runtime.rms_norm_batched_device(
            &prefill.hidden,
            &layer.post_attention_norm_weight,
            params.batch,
            params.rms_norm_eps,
            &mut prefill.input_normed,
        )?;
        runtime.quantize_cutlass_nvfp4_activation_device(
            &prefill.input_normed,
            params.batch,
            layer.gate_proj.cols,
            &mut prefill.cutlass_payload,
            &mut prefill.cutlass_scales,
        )?;
        runtime.matmul_cutlass_nvfp4_pair_prepacked_prefill_device(
            &layer.gate_proj,
            &layer.up_proj,
            &prefill.cutlass_payload,
            &prefill.cutlass_scales,
            params.batch,
            &mut prefill.cutlass_workspace,
            &mut prefill.gate,
            &mut prefill.up,
        )?;
    } else if prefill_linear_native_mxfp4_enabled(runtime, &layer.gate_proj)
        && prefill_linear_native_mxfp4_enabled(runtime, &layer.up_proj)
    {
        runtime.rms_norm_batched_device(
            &prefill.hidden,
            &layer.post_attention_norm_weight,
            params.batch,
            params.rms_norm_eps,
            &mut prefill.input_normed,
        )?;
        runtime.quantize_mxfp4_input_batched_device(
            &prefill.input_normed,
            params.batch,
            layer.gate_proj.cols,
            &mut prefill.mxfp4_hidden,
        )?;
        prefill_gate_up_mxfp4_native_device(
            runtime,
            &layer.gate_proj,
            &layer.up_proj,
            &prefill.mxfp4_hidden,
            params.batch,
            &mut prefill.gate,
            &mut prefill.up,
        )?;
    } else {
        let sp = params.staging_ptr;
        runtime.rms_norm_quant_nvfp4_batched_device(
            &prefill.hidden,
            &layer.post_attention_norm_weight,
            params.batch,
            params.rms_norm_eps,
            layer.gate_proj.input_scale,
            &mut prefill.input_normed,
            &mut prefill.quant_hidden,
        )?;
        prefill_linear_prepared_batched_device(
            runtime,
            &layer.gate_proj,
            &prefill.input_normed,
            &prefill.quant_hidden,
            params.batch,
            &mut prefill.mxfp4_hidden,
            &mut prefill.gate,
            if sp.is_null() { None } else { Some(unsafe { &mut *sp }) },
        )?;
        let mut quant_scale = Some(layer.gate_proj.input_scale);
        prefill_linear_prepare_nvfp4_input(
            runtime,
            &layer.up_proj,
            &prefill.input_normed,
            params.batch,
            &mut quant_scale,
            &mut prefill.quant_hidden,
        )?;
        prefill_linear_prepared_batched_device(
            runtime,
            &layer.up_proj,
            &prefill.input_normed,
            &prefill.quant_hidden,
            params.batch,
            &mut prefill.mxfp4_hidden,
            &mut prefill.up,
            if sp.is_null() { None } else { Some(unsafe { &mut *sp }) },
        )?;
    }
    if prefill_linear_cutlass_nvfp4_enabled(runtime, &layer.down_proj) {
        runtime.swiglu_quantize_cutlass_nvfp4_activation_device(
            &prefill.gate,
            &prefill.up,
            params.batch,
            intermediate,
            &mut prefill.cutlass_payload,
            &mut prefill.cutlass_scales,
        )?;
        prefill_linear_cutlass_nvfp4_prepacked_device(
            runtime,
            &layer.down_proj,
            &prefill.cutlass_payload,
            &prefill.cutlass_scales,
            params.batch,
            &mut prefill.cutlass_workspace,
            &mut prefill.input_normed,
        )?;
    } else if prefill_linear_native_mxfp4_enabled(runtime, &layer.down_proj) {
        runtime.swiglu_mxfp4_quantize_batched_device(
            &prefill.gate,
            &prefill.up,
            params.batch,
            intermediate,
            &mut prefill.mxfp4_intermediate,
        )?;
        runtime.matmul_mxfp4_native_prepacked_prefill_device(
            &layer.down_proj,
            &prefill.mxfp4_intermediate,
            params.batch,
            &mut prefill.input_normed,
        )?;
    } else {
        let sp = params.staging_ptr;
        runtime.swiglu_inplace_gate_device_len(
            &mut prefill.gate,
            &prefill.up,
            params.batch * intermediate,
        )?;
        prefill_linear_batched_device_with_scratch(
            runtime,
            &layer.down_proj,
            &prefill.gate,
            params.batch,
            &mut prefill.quant_intermediate,
            &mut prefill.mxfp4_intermediate,
            &mut prefill.input_normed,
            if sp.is_null() { None } else { Some(unsafe { &mut *sp }) },
        )?;
    }
    if let Some(ref post_norm) = layer.post_mlp_sublayer_norm {
        // Gemma 4 PrePost: normalize MLP output before residual add. Mirrors decode in mlp.rs.
        if prefill.qkv.len() < batch_hidden {
            return Err(AegisError::InvalidPlan(
                "CUDA prefill qkv scratch too small for post-mlp norm".into(),
            ));
        }
        runtime.rms_norm_batched_device(
            &prefill.input_normed,
            post_norm,
            params.batch,
            params.rms_norm_eps,
            &mut prefill.qkv,
        )?;
        runtime.add_inplace_device_len(&mut prefill.hidden, &prefill.qkv, batch_hidden)?;
    } else {
        runtime.add_inplace_device_len(&mut prefill.hidden, &prefill.input_normed, batch_hidden)?;
    }
    let mlp_flops =
        prefill_gemm_flops(
            params.batch,
            layer.gate_proj.rows + layer.up_proj.rows,
            layer.gate_proj.cols,
        ) + prefill_gemm_flops(params.batch, layer.down_proj.rows, layer.down_proj.cols);
    record_prefill_stage(runtime, timings, mlp_start, |timings, elapsed| {
        timings.mlp_us += elapsed;
        timings.mlp_tflops = timings.mlp_tflops.max(tflops(mlp_flops, elapsed));
    })?;
    Ok(())
}

fn prefill_gemm_flops(tokens: usize, output_channels: usize, hidden: usize) -> f64 {
    2.0 * tokens as f64 * output_channels as f64 * hidden as f64
}

fn tflops(flops: f64, elapsed_micros: u128) -> f64 {
    if elapsed_micros == 0 {
        0.0
    } else {
        flops / (elapsed_micros as f64 / 1_000_000.0) / 1.0e12
    }
}

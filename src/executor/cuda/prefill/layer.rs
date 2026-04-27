use std::time::Instant;

use super::gemm::{
    prefill_gate_up_mxfp4_native_device, prefill_linear_batched_device_with_scratch,
    prefill_linear_cutlass_nvfp4_device, prefill_linear_cutlass_nvfp4_enabled,
    prefill_linear_cutlass_nvfp4_prepacked_device, prefill_linear_native_mxfp4_enabled,
    prefill_linear_prepare_nvfp4_input, prefill_linear_prepared_batched_device,
    prefill_qkv_mxfp4_native_device,
};
use super::timings::record_prefill_stage;
use crate::cuda::{CudaRuntime, DensePrefillMetadataProof, DeviceRopeConfig};
use crate::error::{AegisError, Result};
use crate::executor::cuda::state::{
    CudaLayer, CudaLayerState, CudaPrefillScratch, CudaPrefillStageTimings,
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
    pub(super) rope: DeviceRopeConfig,
}

pub(super) fn forward_cuda_layer_prefill_chunk_device(
    runtime: &CudaRuntime,
    layer: &CudaLayer,
    layer_state: &mut CudaLayerState,
    prefill: &mut CudaPrefillScratch,
    params: CudaPrefillForwardParams,
    timings: &mut CudaPrefillStageTimings,
) -> Result<()> {
    let hidden_size = layer.o_proj.rows;
    let intermediate = layer.gate_proj.rows;
    let qkv_start = Instant::now();
    if let Some(qkv_proj) = layer
        .qkv_proj
        .as_ref()
        .filter(|linear| prefill_linear_cutlass_nvfp4_enabled(runtime, linear))
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
            layer.q_proj.rows,
            layer.k_proj.rows,
            layer.q_proj.output_scale,
            layer.k_proj.output_scale,
            layer.v_proj.output_scale,
            &mut prefill.q,
            &mut prefill.k,
            &mut prefill.v,
        )?;
    } else if prefill_linear_native_mxfp4_enabled(runtime, &layer.q_proj)
        && prefill_linear_native_mxfp4_enabled(runtime, &layer.k_proj)
        && prefill_linear_native_mxfp4_enabled(runtime, &layer.v_proj)
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
            layer.q_proj.cols,
            &mut prefill.mxfp4_hidden,
        )?;
        prefill_qkv_mxfp4_native_device(
            runtime,
            &layer.q_proj,
            &layer.k_proj,
            &layer.v_proj,
            &prefill.mxfp4_hidden,
            params.batch,
            &mut prefill.q,
            &mut prefill.k,
            &mut prefill.v,
        )?;
    } else if prefill_linear_cutlass_nvfp4_enabled(runtime, &layer.q_proj)
        && prefill_linear_cutlass_nvfp4_enabled(runtime, &layer.k_proj)
        && prefill_linear_cutlass_nvfp4_enabled(runtime, &layer.v_proj)
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
            layer.q_proj.cols,
            &mut prefill.cutlass_payload,
            &mut prefill.cutlass_scales,
        )?;
        prefill_linear_cutlass_nvfp4_prepacked_device(
            runtime,
            &layer.q_proj,
            &prefill.cutlass_payload,
            &prefill.cutlass_scales,
            params.batch,
            &mut prefill.cutlass_workspace,
            &mut prefill.q,
        )?;
        prefill_linear_cutlass_nvfp4_prepacked_device(
            runtime,
            &layer.k_proj,
            &prefill.cutlass_payload,
            &prefill.cutlass_scales,
            params.batch,
            &mut prefill.cutlass_workspace,
            &mut prefill.k,
        )?;
        prefill_linear_cutlass_nvfp4_prepacked_device(
            runtime,
            &layer.v_proj,
            &prefill.cutlass_payload,
            &prefill.cutlass_scales,
            params.batch,
            &mut prefill.cutlass_workspace,
            &mut prefill.v,
        )?;
    } else {
        runtime.rms_norm_quant_nvfp4_batched_device(
            &prefill.hidden,
            &layer.input_norm_weight,
            params.batch,
            params.rms_norm_eps,
            layer.q_proj.input_scale,
            &mut prefill.input_normed,
            &mut prefill.quant_hidden,
        )?;
        prefill_linear_prepared_batched_device(
            runtime,
            &layer.q_proj,
            &prefill.input_normed,
            &prefill.quant_hidden,
            params.batch,
            &mut prefill.mxfp4_hidden,
            &mut prefill.q,
        )?;
        let mut quant_scale = Some(layer.q_proj.input_scale);
        prefill_linear_prepare_nvfp4_input(
            runtime,
            &layer.k_proj,
            &prefill.input_normed,
            params.batch,
            &mut quant_scale,
            &mut prefill.quant_hidden,
        )?;
        prefill_linear_prepared_batched_device(
            runtime,
            &layer.k_proj,
            &prefill.input_normed,
            &prefill.quant_hidden,
            params.batch,
            &mut prefill.mxfp4_hidden,
            &mut prefill.k,
        )?;
        prefill_linear_prepare_nvfp4_input(
            runtime,
            &layer.v_proj,
            &prefill.input_normed,
            params.batch,
            &mut quant_scale,
            &mut prefill.quant_hidden,
        )?;
        prefill_linear_prepared_batched_device(
            runtime,
            &layer.v_proj,
            &prefill.input_normed,
            &prefill.quant_hidden,
            params.batch,
            &mut prefill.mxfp4_hidden,
            &mut prefill.v,
        )?;
    }
    let qkv_flops = prefill_gemm_flops(
        params.batch,
        layer.q_proj.rows + layer.k_proj.rows + layer.v_proj.rows,
        layer.q_proj.cols,
    );
    record_prefill_stage(runtime, timings, qkv_start, |timings, elapsed| {
        timings.qkv_us += elapsed;
        timings.qkv_tflops = timings.qkv_tflops.max(tflops(qkv_flops, elapsed));
    })?;

    let rope_start = Instant::now();
    runtime.apply_rope_positions_batched_f16_out_device(
        &mut prefill.q,
        &prefill.positions,
        params.batch,
        params.num_attention_heads,
        params.head_dim,
        params.rope,
        &mut prefill.q_half,
    )?;
    record_prefill_stage(runtime, timings, rope_start, |timings, elapsed| {
        timings.rope_us += elapsed
    })?;

    let kv_store_start = Instant::now();
    runtime.store_kv_slots_batched_rope_key_device(
        &mut layer_state.kv.keys,
        &mut layer_state.kv.values,
        &mut prefill.k,
        &prefill.v,
        &prefill.positions,
        &prefill.slot_mapping,
        params.batch,
        params.num_kv_heads,
        params.head_dim,
        params.kv_context_size,
        params.dense_metadata,
        params.rope,
    )?;
    record_prefill_stage(runtime, timings, kv_store_start, |timings, elapsed| {
        timings.kv_store_us += elapsed
    })?;

    let attention_start = Instant::now();
    runtime.attention_prefill_dense_compat_device(
        &layer_state.kv.keys,
        &layer_state.kv.values,
        &prefill.k,
        &prefill.v,
        &prefill.q,
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
        params.num_kv_heads,
        params.head_dim,
        &mut prefill.attn_context,
        params.dense_metadata,
    )?;
    record_prefill_stage(runtime, timings, attention_start, |timings, elapsed| {
        timings.attention_us += elapsed
    })?;

    let o_proj_start = Instant::now();
    if prefill_linear_cutlass_nvfp4_enabled(runtime, &layer.o_proj) {
        prefill_linear_cutlass_nvfp4_device(
            runtime,
            &layer.o_proj,
            &prefill.attn_context,
            params.batch,
            &mut prefill.cutlass_payload,
            &mut prefill.cutlass_scales,
            &mut prefill.cutlass_workspace,
            &mut prefill.attn_out,
        )?;
    } else {
        prefill_linear_batched_device_with_scratch(
            runtime,
            &layer.o_proj,
            &prefill.attn_context,
            params.batch,
            &mut prefill.quant_hidden,
            &mut prefill.mxfp4_hidden,
            &mut prefill.attn_out,
        )?;
    }
    record_prefill_stage(runtime, timings, o_proj_start, |timings, elapsed| {
        timings.o_proj_us += elapsed
    })?;

    let batch_hidden = params.batch * hidden_size;
    if prefill.hidden.len() < batch_hidden || prefill.attn_out.len() < batch_hidden {
        return Err(AegisError::InvalidPlan(
            "CUDA prefill hidden scratch is too small".into(),
        ));
    }
    runtime.add_inplace_device_len(&mut prefill.hidden, &prefill.attn_out, batch_hidden)?;

    let mlp_start = Instant::now();
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
            &mut prefill.mlp_out,
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
            &mut prefill.mlp_out,
        )?;
    } else {
        runtime.swiglu_device_len(
            &prefill.gate,
            &prefill.up,
            &mut prefill.swiglu,
            params.batch * intermediate,
        )?;
        prefill_linear_batched_device_with_scratch(
            runtime,
            &layer.down_proj,
            &prefill.swiglu,
            params.batch,
            &mut prefill.quant_intermediate,
            &mut prefill.mxfp4_intermediate,
            &mut prefill.mlp_out,
        )?;
    }
    runtime.add_inplace_device_len(&mut prefill.hidden, &prefill.mlp_out, batch_hidden)?;
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

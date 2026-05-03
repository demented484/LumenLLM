use super::linear_ops::{
    matvec_nvfp4_device_with_scratch, matvec_nvfp4_prepared_device_reuse, native_mxfp4_enabled,
    prepare_nvfp4_input,
};
use super::state::{CudaLayer, CudaLayerState, CudaScratch};
use crate::cuda::{CudaRuntime, DeviceBuffer, DeviceRopeConfig};
use aegisllm_base::error::Result;

/// Forward attention for a single layer.
///
/// `staging_slot_idx`: when `Some(idx)`, the layer's KV is host-resident and the
/// caller has pre-uploaded the prior KV onto `scratch.kv_staging.slots[idx]` via
/// the transfer stream and event-synchronized the compute stream against it.
/// This function only runs store_kv + attention against the slot; the caller is
/// responsible for scheduling the post-compute D2H writeback on the transfer stream.
/// When `None`, the layer's KV is fully VRAM-resident.
#[allow(clippy::too_many_arguments)]
pub(super) fn forward_attention_device(
    runtime: &CudaRuntime,
    layer: &CudaLayer,
    layer_state: &mut CudaLayerState,
    hidden: &DeviceBuffer<f32>,
    scratch: &mut CudaScratch,
    p_position: &DeviceBuffer<u32>,
    p_seq_len: &DeviceBuffer<u32>,
    rms_norm_eps: f32,
    num_attention_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_context_size: usize,
    rope: DeviceRopeConfig,
    staging_slot_idx: Option<usize>,
    _position: usize,
    _seq_len: usize,
) -> Result<()> {
    let kv_width = num_kv_heads * head_dim;

    runtime.rms_norm_quant_nvfp4_device(
        hidden,
        &layer.input_norm_weight,
        rms_norm_eps,
        layer.q_proj.input_scale,
        &mut scratch.input_normed,
        &mut scratch.quant_hidden,
    )?;
    let mut quant_scale = Some(layer.q_proj.input_scale);
    // Q projection: quantize input_normed to mxfp4_hidden (native path) or quant_hidden (legacy).
    let mxfp4_valid = matvec_nvfp4_prepared_device_reuse(
        runtime,
        &layer.q_proj,
        &scratch.input_normed,
        &scratch.quant_hidden,
        &mut scratch.mxfp4_hidden,
        false,
        &mut scratch.q,
        scratch.staging_pool.as_deref_mut(),
    )?;
    // K/V projections share the same input_normed — skip MXFP4 re-quantize in native path.
    prepare_nvfp4_input(
        runtime,
        &layer.k_proj,
        &scratch.input_normed,
        &mut quant_scale,
        &mut scratch.quant_hidden,
    )?;
    matvec_nvfp4_prepared_device_reuse(
        runtime,
        &layer.k_proj,
        &scratch.input_normed,
        &scratch.quant_hidden,
        &mut scratch.mxfp4_hidden,
        mxfp4_valid,
        &mut scratch.k,
        scratch.staging_pool.as_deref_mut(),
    )?;
    prepare_nvfp4_input(
        runtime,
        &layer.v_proj,
        &scratch.input_normed,
        &mut quant_scale,
        &mut scratch.quant_hidden,
    )?;
    matvec_nvfp4_prepared_device_reuse(
        runtime,
        &layer.v_proj,
        &scratch.input_normed,
        &scratch.quant_hidden,
        &mut scratch.mxfp4_hidden,
        mxfp4_valid,
        &mut scratch.v,
        scratch.staging_pool.as_deref_mut(),
    )?;
    runtime.apply_rope_ptr_device(
        &mut scratch.q,
        p_position,
        num_attention_heads,
        head_dim,
        rope,
    )?;
    runtime.apply_rope_ptr_device(&mut scratch.k, p_position, num_kv_heads, head_dim, rope)?;

    if let Some(idx) = staging_slot_idx {
        // Host-resident KV: caller has pre-uploaded prior KV onto the staging slot
        // and the compute stream is synchronized against the H2D event. Run store +
        // attention against this slot; caller will schedule the D2H writeback.
        let pool = scratch.kv_staging.as_mut().ok_or_else(|| {
            aegisllm_base::error::AegisError::InvalidPlan(
                "host-resident KV cache requires kv_staging pool".into(),
            )
        })?;
        let staging = &mut pool.slots[idx];
        runtime.store_kv_ptr_device(
            &mut staging.keys,
            &mut staging.values,
            &scratch.k,
            &scratch.v,
            p_position,
            kv_width,
            kv_context_size,
        )?;
        runtime.attention_decode_split_ptr_device(
            &staging.keys,
            &staging.values,
            &scratch.q,
            p_seq_len,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            layer.window_size,
            &mut scratch.attn_split_acc,
            &mut scratch.attn_split_m,
            &mut scratch.attn_split_l,
            &mut scratch.attn_context,
        )?;
    } else {
        runtime.store_kv_ptr_device(
            &mut layer_state.kv.keys,
            &mut layer_state.kv.values,
            &scratch.k,
            &scratch.v,
            p_position,
            kv_width,
            kv_context_size,
        )?;
        runtime.attention_decode_split_ptr_device(
            &layer_state.kv.keys,
            &layer_state.kv.values,
            &scratch.q,
            p_seq_len,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            layer.window_size,
            &mut scratch.attn_split_acc,
            &mut scratch.attn_split_m,
            &mut scratch.attn_split_l,
            &mut scratch.attn_context,
        )?;
    }
    if native_mxfp4_enabled(runtime, &layer.o_proj) {
        matvec_nvfp4_device_with_scratch(
            runtime,
            &layer.o_proj,
            &scratch.attn_context,
            &mut scratch.quant_hidden,
            &mut scratch.mxfp4_hidden,
            &mut scratch.attn_out,
            scratch.staging_pool.as_deref_mut(),
        )?;
    } else if runtime.cutlass_nvfp4_inference_enabled_for(&layer.o_proj) {
        runtime.matmul_cutlass_nvfp4_prefill_device(
            &layer.o_proj,
            &scratch.attn_context,
            1,
            &mut scratch.cutlass_payload,
            &mut scratch.cutlass_scales,
            &mut scratch.cutlass_workspace,
            &mut scratch.attn_out,
        )?;
    } else {
        matvec_nvfp4_device_with_scratch(
            runtime,
            &layer.o_proj,
            &scratch.attn_context,
            &mut scratch.quant_hidden,
            &mut scratch.mxfp4_hidden,
            &mut scratch.attn_out,
            scratch.staging_pool.as_deref_mut(),
        )?;
    }
    if let Some(ref post_norm) = layer.post_attn_sublayer_norm {
        // Gemma 4 PrePost: normalize attention output before adding to residual.
        runtime.rms_norm_device(&scratch.attn_out, post_norm, rms_norm_eps, &mut scratch.post_normed)?;
        runtime.add_device(hidden, &scratch.post_normed, &mut scratch.residual)?;
    } else {
        runtime.add_device(hidden, &scratch.attn_out, &mut scratch.residual)?;
    }
    Ok(())
}

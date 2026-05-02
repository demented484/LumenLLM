use super::linear_ops::{
    matvec_nvfp4_device_with_scratch, matvec_nvfp4_prepared_device, prepare_nvfp4_input,
};
use super::state::{CudaLayer, CudaLayerState, CudaScratch};
use crate::cuda::{CudaRuntime, DeviceBuffer, DeviceRopeConfig};
use aegisllm_base::error::Result;

pub(super) fn forward_attention_device(
    runtime: &CudaRuntime,
    layer: &CudaLayer,
    layer_state: &mut CudaLayerState,
    hidden: &DeviceBuffer<f32>,
    scratch: &mut CudaScratch,
    rms_norm_eps: f32,
    position: usize,
    num_attention_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_context_size: usize,
    rope: DeviceRopeConfig,
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
    matvec_nvfp4_prepared_device(
        runtime,
        &layer.q_proj,
        &scratch.input_normed,
        &scratch.quant_hidden,
        &mut scratch.mxfp4_hidden,
        &mut scratch.q,
    )?;
    prepare_nvfp4_input(
        runtime,
        &layer.k_proj,
        &scratch.input_normed,
        &mut quant_scale,
        &mut scratch.quant_hidden,
    )?;
    matvec_nvfp4_prepared_device(
        runtime,
        &layer.k_proj,
        &scratch.input_normed,
        &scratch.quant_hidden,
        &mut scratch.mxfp4_hidden,
        &mut scratch.k,
    )?;
    prepare_nvfp4_input(
        runtime,
        &layer.v_proj,
        &scratch.input_normed,
        &mut quant_scale,
        &mut scratch.quant_hidden,
    )?;
    matvec_nvfp4_prepared_device(
        runtime,
        &layer.v_proj,
        &scratch.input_normed,
        &scratch.quant_hidden,
        &mut scratch.mxfp4_hidden,
        &mut scratch.v,
    )?;
    runtime.apply_rope_device(
        &mut scratch.q,
        position,
        num_attention_heads,
        head_dim,
        rope,
    )?;
    runtime.apply_rope_device(&mut scratch.k, position, num_kv_heads, head_dim, rope)?;
    runtime.store_kv_device(
        &mut layer_state.kv.keys,
        &mut layer_state.kv.values,
        &scratch.k,
        &scratch.v,
        position,
        kv_width,
        kv_context_size,
    )?;
    runtime.attention_decode_device(
        &layer_state.kv.keys,
        &layer_state.kv.values,
        &scratch.q,
        position + 1,
        num_attention_heads,
        num_kv_heads,
        head_dim,
        &mut scratch.attn_context,
    )?;
    if runtime.cutlass_nvfp4_inference_enabled_for(&layer.o_proj) {
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
        )?;
    }
    runtime.add_device(hidden, &scratch.attn_out, &mut scratch.residual)?;
    Ok(())
}

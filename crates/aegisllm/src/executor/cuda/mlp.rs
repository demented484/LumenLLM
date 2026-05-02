use super::linear_ops::{
    matvec_nvfp4_device_with_scratch, matvec_nvfp4_prepared_device, prepare_nvfp4_input,
};
use super::state::{CudaLayer, CudaScratch};
use crate::cuda::CudaRuntime;
use crate::error::Result;

pub(super) fn forward_mlp_device(
    runtime: &CudaRuntime,
    layer: &CudaLayer,
    scratch: &mut CudaScratch,
    rms_norm_eps: f32,
) -> Result<()> {
    if runtime.cutlass_nvfp4_inference_enabled_for(&layer.gate_proj)
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
        matvec_nvfp4_prepared_device(
            runtime,
            &layer.gate_proj,
            &scratch.post_normed,
            &scratch.quant_hidden,
            &mut scratch.mxfp4_hidden,
            &mut scratch.gate,
        )?;
        prepare_nvfp4_input(
            runtime,
            &layer.up_proj,
            &scratch.post_normed,
            &mut quant_scale,
            &mut scratch.quant_hidden,
        )?;
        matvec_nvfp4_prepared_device(
            runtime,
            &layer.up_proj,
            &scratch.post_normed,
            &scratch.quant_hidden,
            &mut scratch.mxfp4_hidden,
            &mut scratch.up,
        )?;
    }
    if runtime.cutlass_nvfp4_inference_enabled_for(&layer.down_proj) {
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
        )?;
    }
    runtime.add_device(&scratch.residual, &scratch.mlp_out, &mut scratch.hidden_out)?;
    Ok(())
}

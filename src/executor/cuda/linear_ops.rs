use crate::cuda::{CudaRuntime, DeviceBuffer, DeviceNvfp4Linear};
use crate::error::Result;

pub(super) fn prepare_nvfp4_input(
    runtime: &CudaRuntime,
    linear: &DeviceNvfp4Linear,
    input: &DeviceBuffer<f32>,
    current_scale: &mut Option<f32>,
    scratch: &mut DeviceBuffer<f32>,
) -> Result<()> {
    if current_scale
        .map(|scale| scale_differs(scale, linear.input_scale))
        .unwrap_or(true)
    {
        runtime.quantize_nvfp4_input_device(input, linear.input_scale, scratch)?;
        *current_scale = Some(linear.input_scale);
    }
    Ok(())
}

pub(super) fn prepare_nvfp4_input_batched(
    runtime: &CudaRuntime,
    linear: &DeviceNvfp4Linear,
    input: &DeviceBuffer<f32>,
    batch: usize,
    current_scale: &mut Option<f32>,
    scratch: &mut DeviceBuffer<f32>,
) -> Result<()> {
    if current_scale
        .map(|scale| scale_differs(scale, linear.input_scale))
        .unwrap_or(true)
    {
        runtime.quantize_nvfp4_input_batched_device(
            input,
            linear.input_scale,
            batch,
            linear.cols,
            scratch,
        )?;
        *current_scale = Some(linear.input_scale);
    }
    Ok(())
}

pub(super) fn matvec_nvfp4_prepared_device(
    runtime: &CudaRuntime,
    linear: &DeviceNvfp4Linear,
    native_input: &DeviceBuffer<f32>,
    quantized_input: &DeviceBuffer<f32>,
    mxfp4_input: &mut DeviceBuffer<u8>,
    output: &mut DeviceBuffer<f32>,
) -> Result<()> {
    if native_mxfp4_enabled(runtime, linear) {
        runtime.quantize_mxfp4_input_device(native_input, mxfp4_input)?;
        runtime.matvec_mxfp4_native_prepacked_device(linear, mxfp4_input, output)
    } else {
        runtime.matvec_nvfp4_prequantized_device(linear, quantized_input, output)
    }
}

pub(super) fn matvec_nvfp4_device_with_scratch(
    runtime: &CudaRuntime,
    linear: &DeviceNvfp4Linear,
    input: &DeviceBuffer<f32>,
    quantized_input: &mut DeviceBuffer<f32>,
    mxfp4_input: &mut DeviceBuffer<u8>,
    output: &mut DeviceBuffer<f32>,
) -> Result<()> {
    if native_mxfp4_enabled(runtime, linear) {
        runtime.quantize_mxfp4_input_device(input, mxfp4_input)?;
        runtime.matvec_mxfp4_native_prepacked_device(linear, mxfp4_input, output)
    } else {
        runtime.quantize_nvfp4_input_device(input, linear.input_scale, quantized_input)?;
        runtime.matvec_nvfp4_prequantized_device(linear, quantized_input, output)
    }
}

pub(super) fn matvec_nvfp4_prepared_batched_device(
    runtime: &CudaRuntime,
    linear: &DeviceNvfp4Linear,
    native_input: &DeviceBuffer<f32>,
    quantized_input: &DeviceBuffer<f32>,
    batch: usize,
    mxfp4_input: &mut DeviceBuffer<u8>,
    output: &mut DeviceBuffer<f32>,
) -> Result<()> {
    if native_mxfp4_enabled(runtime, linear) {
        runtime.quantize_mxfp4_input_batched_device(
            native_input,
            batch,
            linear.cols,
            mxfp4_input,
        )?;
        runtime.matvec_mxfp4_native_prepacked_batched_device(linear, mxfp4_input, batch, output)
    } else {
        runtime.matvec_nvfp4_prequantized_batched_device(linear, quantized_input, batch, output)
    }
}

pub(super) fn matvec_nvfp4_batched_device_with_scratch(
    runtime: &CudaRuntime,
    linear: &DeviceNvfp4Linear,
    input: &DeviceBuffer<f32>,
    batch: usize,
    quantized_input: &mut DeviceBuffer<f32>,
    mxfp4_input: &mut DeviceBuffer<u8>,
    output: &mut DeviceBuffer<f32>,
) -> Result<()> {
    if native_mxfp4_enabled(runtime, linear) {
        runtime.quantize_mxfp4_input_batched_device(input, batch, linear.cols, mxfp4_input)?;
        runtime.matvec_mxfp4_native_prepacked_batched_device(linear, mxfp4_input, batch, output)
    } else {
        runtime.quantize_nvfp4_input_batched_device(
            input,
            linear.input_scale,
            batch,
            linear.cols,
            quantized_input,
        )?;
        runtime.matvec_nvfp4_prequantized_batched_device(linear, quantized_input, batch, output)
    }
}

pub(super) fn native_mxfp4_enabled(runtime: &CudaRuntime, linear: &DeviceNvfp4Linear) -> bool {
    runtime.native_mxfp4_inference_enabled_for(linear)
}

fn scale_differs(a: f32, b: f32) -> bool {
    (a - b).abs() > 1.0e-12
}

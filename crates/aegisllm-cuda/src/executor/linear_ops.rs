use crate::cuda::{CudaRuntime, DeviceBuffer, DeviceNvfp4Linear};
use crate::cuda::staging::LinearStagingPool;
use aegisllm_base::error::{AegisError, Result};
use super::state::CudaLinear;

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

/// Like `matvec_nvfp4_prepared_device_reuse` but skips the MXFP4 input quantization step when
/// `mxfp4_already_valid` is true (i.e. `mxfp4_input` was already filled from the same
/// `native_input` by a previous projection in the same layer). Returns whether `mxfp4_input`
/// is now valid so callers can chain projections without tracking the flag themselves.
pub(super) fn matvec_nvfp4_prepared_device_reuse(
    runtime: &CudaRuntime,
    linear: &DeviceNvfp4Linear,
    native_input: &DeviceBuffer<f32>,
    quantized_input: &DeviceBuffer<f32>,
    mxfp4_input: &mut DeviceBuffer<u8>,
    mxfp4_already_valid: bool,
    output: &mut DeviceBuffer<f32>,
    staging: Option<&mut LinearStagingPool>,
) -> Result<bool> {
    // Host-resident (StagedHostToDevice): stream weights from RAM to staging VRAM first.
    if linear.is_host_resident() {
        let staging = staging.ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "staging pool required for host-resident linear `{}`",
                linear.name
            ))
        })?;
        if linear.is_host_resident_with_native_mxfp4() {
            // Tensor-core path: quantize input to MXFP4 (reuse if already valid), stage weight.
            if !mxfp4_already_valid {
                runtime.quantize_mxfp4_input_device(native_input, mxfp4_input)?;
            }
            runtime.matvec_native_mxfp4_staged_device(linear, staging, mxfp4_input, output)?;
            return Ok(true);
        }
        runtime.matvec_nvfp4_staged_prequantized_device(linear, staging, quantized_input, output)?;
        return Ok(false);
    }
    if native_mxfp4_enabled(runtime, linear) {
        if !mxfp4_already_valid {
            runtime.quantize_mxfp4_input_device(native_input, mxfp4_input)?;
        }
        runtime.matvec_mxfp4_native_prepacked_device(linear, mxfp4_input, output)?;
        Ok(true)
    } else {
        runtime.matvec_nvfp4_prequantized_device(linear, quantized_input, output)?;
        Ok(false)
    }
}

pub(super) fn matvec_nvfp4_device_with_scratch(
    runtime: &CudaRuntime,
    linear: &DeviceNvfp4Linear,
    input: &DeviceBuffer<f32>,
    quantized_input: &mut DeviceBuffer<f32>,
    mxfp4_input: &mut DeviceBuffer<u8>,
    output: &mut DeviceBuffer<f32>,
    staging: Option<&mut LinearStagingPool>,
) -> Result<()> {
    // Host-resident: quantize activations then stream weights from RAM.
    if linear.is_host_resident() {
        let staging = staging.ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "staging pool required for host-resident linear `{}`",
                linear.name
            ))
        })?;
        if linear.is_host_resident_with_native_mxfp4() {
            runtime.quantize_mxfp4_input_device(input, mxfp4_input)?;
            return runtime.matvec_native_mxfp4_staged_device(linear, staging, mxfp4_input, output);
        }
        runtime.quantize_nvfp4_input_device(input, linear.input_scale, quantized_input)?;
        return runtime.matvec_nvfp4_staged_prequantized_device(
            linear, staging, quantized_input, output,
        );
    }
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
    staging: Option<&mut LinearStagingPool>,
) -> Result<()> {
    if linear.is_host_resident() {
        let staging = staging.ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "staging pool required for host-resident linear `{}`",
                linear.name
            ))
        })?;
        if linear.is_host_resident_with_native_mxfp4() {
            runtime.quantize_mxfp4_input_batched_device(native_input, batch, linear.cols, mxfp4_input)?;
            return runtime.matvec_native_mxfp4_staged_batched_device(
                linear, staging, mxfp4_input, batch, output,
            );
        }
        return runtime.matvec_nvfp4_staged_prequantized_batched_device(
            linear, staging, quantized_input, batch, output,
        );
    }
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
    staging: Option<&mut LinearStagingPool>,
) -> Result<()> {
    if linear.is_host_resident() {
        let staging = staging.ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "staging pool required for host-resident linear `{}`",
                linear.name
            ))
        })?;
        if linear.is_host_resident_with_native_mxfp4() {
            runtime.quantize_mxfp4_input_batched_device(input, batch, linear.cols, mxfp4_input)?;
            return runtime.matvec_native_mxfp4_staged_batched_device(
                linear, staging, mxfp4_input, batch, output,
            );
        }
        runtime.quantize_nvfp4_input_batched_device(
            input, linear.input_scale, batch, linear.cols, quantized_input,
        )?;
        return runtime.matvec_nvfp4_staged_prequantized_batched_device(
            linear, staging, quantized_input, batch, output,
        );
    }
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

/// Matvec dispatch for a CudaLinear (BF16 or NVFP4 path).
pub(super) fn matvec_cuda_linear_with_scratch(
    runtime: &CudaRuntime,
    linear: &CudaLinear,
    input: &DeviceBuffer<f32>,
    quant_hidden: &mut DeviceBuffer<f32>,
    mxfp4_hidden: &mut DeviceBuffer<u8>,
    output: &mut DeviceBuffer<f32>,
    staging: Option<&mut LinearStagingPool>,
) -> Result<()> {
    match linear {
        CudaLinear::Nvfp4(l) => {
            matvec_nvfp4_device_with_scratch(runtime, l, input, quant_hidden, mxfp4_hidden, output, staging)
        }
        CudaLinear::Bf16(m) => {
            runtime.matvec_bf16_reference_device(m, input, output)
        }
        CudaLinear::Fp8(m) => {
            runtime.matvec_fp8_standalone_device(m, input, output)
        }
    }
}

fn scale_differs(a: f32, b: f32) -> bool {
    (a - b).abs() > 1.0e-12
}

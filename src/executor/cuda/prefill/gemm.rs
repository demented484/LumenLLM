use crate::cuda::{CudaRuntime, DeviceBuffer, DeviceNvfp4Linear};
use crate::error::Result;
use crate::executor::cuda::linear_ops::{
    matvec_nvfp4_batched_device_with_scratch, matvec_nvfp4_prepared_batched_device,
    native_mxfp4_enabled, prepare_nvfp4_input_batched,
};

pub(super) fn prefill_linear_native_mxfp4_enabled(
    runtime: &CudaRuntime,
    linear: &DeviceNvfp4Linear,
) -> bool {
    native_mxfp4_enabled(runtime, linear)
}

pub(super) fn prefill_linear_prepare_nvfp4_input(
    runtime: &CudaRuntime,
    linear: &DeviceNvfp4Linear,
    input: &DeviceBuffer<f32>,
    batch: usize,
    quant_scale: &mut Option<f32>,
    quantized_input: &mut DeviceBuffer<f32>,
) -> Result<()> {
    prepare_nvfp4_input_batched(runtime, linear, input, batch, quant_scale, quantized_input)
}

pub(super) fn prefill_linear_prepared_batched_device(
    runtime: &CudaRuntime,
    linear: &DeviceNvfp4Linear,
    input: &DeviceBuffer<f32>,
    quantized_input: &DeviceBuffer<f32>,
    batch: usize,
    input_mxfp4: &mut DeviceBuffer<u8>,
    output: &mut DeviceBuffer<f32>,
) -> Result<()> {
    matvec_nvfp4_prepared_batched_device(
        runtime,
        linear,
        input,
        quantized_input,
        batch,
        input_mxfp4,
        output,
    )
}

pub(super) fn prefill_linear_batched_device_with_scratch(
    runtime: &CudaRuntime,
    linear: &DeviceNvfp4Linear,
    input: &DeviceBuffer<f32>,
    batch: usize,
    quantized_input: &mut DeviceBuffer<f32>,
    input_mxfp4: &mut DeviceBuffer<u8>,
    output: &mut DeviceBuffer<f32>,
) -> Result<()> {
    matvec_nvfp4_batched_device_with_scratch(
        runtime,
        linear,
        input,
        batch,
        quantized_input,
        input_mxfp4,
        output,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prefill_qkv_mxfp4_native_device(
    runtime: &CudaRuntime,
    q_proj: &DeviceNvfp4Linear,
    k_proj: &DeviceNvfp4Linear,
    v_proj: &DeviceNvfp4Linear,
    input_mxfp4: &DeviceBuffer<u8>,
    batch: usize,
    q: &mut DeviceBuffer<f32>,
    k: &mut DeviceBuffer<f32>,
    v: &mut DeviceBuffer<f32>,
) -> Result<()> {
    runtime.matmul_mxfp4_native_qkv_prefill_device(
        q_proj,
        k_proj,
        v_proj,
        input_mxfp4,
        batch,
        q,
        k,
        v,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prefill_gate_up_mxfp4_native_device(
    runtime: &CudaRuntime,
    gate_proj: &DeviceNvfp4Linear,
    up_proj: &DeviceNvfp4Linear,
    input_mxfp4: &DeviceBuffer<u8>,
    batch: usize,
    gate: &mut DeviceBuffer<f32>,
    up: &mut DeviceBuffer<f32>,
) -> Result<()> {
    runtime.matmul_mxfp4_native_gate_up_prefill_device(
        gate_proj,
        up_proj,
        input_mxfp4,
        batch,
        gate,
        up,
    )
}

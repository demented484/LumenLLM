use std::ffi::c_void;

use cudarc::driver::{DevicePtr, DevicePtrMut};

use super::CudaRuntime;
use crate::cuda::{DeviceBuffer, DeviceNvfp4Linear, cutlass_bridge};
use crate::error::{AegisError, Result};
use crate::planning::runtime::KernelFamily;

fn i32_arg(name: &str, value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| {
        AegisError::InvalidPlan(format!(
            "CUTLASS FP4 argument {name} exceeds i32 range: {value}"
        ))
    })
}

fn checked_len(label: &str, lhs: usize, rhs: usize) -> Result<usize> {
    lhs.checked_mul(rhs).ok_or_else(|| {
        AegisError::InvalidPlan(format!(
            "CUTLASS FP4 {label} length overflow: {lhs} * {rhs}"
        ))
    })
}

impl CudaRuntime {
    pub fn cutlass_nvfp4_activation_payload_bytes(rows: usize, cols: usize) -> Result<usize> {
        if cols % 32 != 0 {
            return Err(AegisError::InvalidPlan(format!(
                "CUTLASS NVFP4 activation payload requires cols divisible by 32, got {cols}"
            )));
        }
        checked_len("activation payload", rows, cols / 2)
    }

    pub fn cutlass_nvfp4_activation_scale_bytes(rows: usize, cols: usize) -> Result<usize> {
        if cols % 32 != 0 {
            return Err(AegisError::InvalidPlan(format!(
                "CUTLASS NVFP4 activation scales require cols divisible by 32, got {cols}"
            )));
        }
        let scale_rows = rows.div_ceil(128) * 128;
        let scale_cols = (cols / 16).div_ceil(4) * 4;
        checked_len("activation scales", scale_rows, scale_cols)
    }

    pub fn cutlass_nvfp4_workspace_bytes(
        &self,
        tokens: usize,
        output_channels: usize,
        hidden: usize,
    ) -> Result<usize> {
        let m = i32_arg("m", tokens)?;
        let n = i32_arg("n", output_channels)?;
        let k = i32_arg("k", hidden)?;
        cutlass_bridge::workspace_size(m, n, k).map_err(|error| {
            AegisError::Unsupported(format!("CUTLASS FP4 workspace query failed: {error}"))
        })
    }

    pub fn cutlass_nvfp4_inference_enabled_for(&self, linear: &DeviceNvfp4Linear) -> bool {
        linear.cutlass_nvfp4.is_some()
            && matches!(
                linear.kernel_family,
                KernelFamily::CudaCutlassFp4TensorCores | KernelFamily::CudaNativeFp4TensorCores
            )
    }

    pub fn quantize_cutlass_nvfp4_activation_device(
        &self,
        input: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
        payload: &mut DeviceBuffer<u8>,
        scales: &mut DeviceBuffer<u8>,
    ) -> Result<()> {
        let expected_input = checked_len("activation input", rows, cols)?;
        let expected_payload = Self::cutlass_nvfp4_activation_payload_bytes(rows, cols)?;
        let expected_scales = Self::cutlass_nvfp4_activation_scale_bytes(rows, cols)?;
        if input.len() < expected_input
            || payload.len() < expected_payload
            || scales.len() < expected_scales
        {
            return Err(AegisError::InvalidPlan(format!(
                "CUTLASS NVFP4 activation quant buffers too small: input={} expected_input={} payload={} expected_payload={} scales={} expected_scales={}",
                input.len(),
                expected_input,
                payload.len(),
                expected_payload,
                scales.len(),
                expected_scales
            )));
        }

        let rows = i32_arg("rows", rows)?;
        let cols = i32_arg("cols", cols)?;
        let (input_ptr, _input_read) = input.slice.device_ptr(&self.stream);
        let (payload_ptr, _payload_write) = payload.slice.device_ptr_mut(&self.stream);
        let (scales_ptr, _scales_write) = scales.slice.device_ptr_mut(&self.stream);
        let stream = self.stream.cu_stream().cast::<c_void>();
        unsafe {
            cutlass_bridge::quantize_f32(
                input_ptr as *const f32,
                rows,
                cols,
                payload_ptr as *mut u8,
                scales_ptr as *mut u8,
                stream,
            )
        }
        .map_err(|error| {
            AegisError::Unsupported(format!(
                "CUTLASS FP4 activation quantization failed: {error}"
            ))
        })
    }

    pub fn swiglu_quantize_cutlass_nvfp4_activation_device(
        &self,
        gate: &DeviceBuffer<f32>,
        up: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
        payload: &mut DeviceBuffer<u8>,
        scales: &mut DeviceBuffer<u8>,
    ) -> Result<()> {
        let expected_input = checked_len("SwiGLU activation input", rows, cols)?;
        let expected_payload = Self::cutlass_nvfp4_activation_payload_bytes(rows, cols)?;
        let expected_scales = Self::cutlass_nvfp4_activation_scale_bytes(rows, cols)?;
        if gate.len() < expected_input
            || up.len() < expected_input
            || payload.len() < expected_payload
            || scales.len() < expected_scales
        {
            return Err(AegisError::InvalidPlan(format!(
                "CUTLASS NVFP4 SwiGLU quant buffers too small: gate={} up={} expected_input={} payload={} expected_payload={} scales={} expected_scales={}",
                gate.len(),
                up.len(),
                expected_input,
                payload.len(),
                expected_payload,
                scales.len(),
                expected_scales
            )));
        }

        let rows = i32_arg("rows", rows)?;
        let cols = i32_arg("cols", cols)?;
        let (gate_ptr, _gate_read) = gate.slice.device_ptr(&self.stream);
        let (up_ptr, _up_read) = up.slice.device_ptr(&self.stream);
        let (payload_ptr, _payload_write) = payload.slice.device_ptr_mut(&self.stream);
        let (scales_ptr, _scales_write) = scales.slice.device_ptr_mut(&self.stream);
        let stream = self.stream.cu_stream().cast::<c_void>();
        unsafe {
            cutlass_bridge::swiglu_quantize_f32(
                gate_ptr as *const f32,
                up_ptr as *const f32,
                rows,
                cols,
                payload_ptr as *mut u8,
                scales_ptr as *mut u8,
                stream,
            )
        }
        .map_err(|error| {
            AegisError::Unsupported(format!(
                "CUTLASS FP4 SwiGLU activation quantization failed: {error}"
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matmul_cutlass_nvfp4_prefill_device(
        &self,
        linear: &DeviceNvfp4Linear,
        input: &DeviceBuffer<f32>,
        batch: usize,
        activation_payload: &mut DeviceBuffer<u8>,
        activation_scales: &mut DeviceBuffer<u8>,
        workspace: &mut DeviceBuffer<u8>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        self.quantize_cutlass_nvfp4_activation_device(
            input,
            batch,
            linear.cols,
            activation_payload,
            activation_scales,
        )?;
        self.matmul_cutlass_nvfp4_prepacked_prefill_device(
            linear,
            activation_payload,
            activation_scales,
            batch,
            workspace,
            output,
        )
    }

    pub fn matmul_cutlass_nvfp4_prepacked_prefill_device(
        &self,
        linear: &DeviceNvfp4Linear,
        activation_payload: &DeviceBuffer<u8>,
        activation_scales: &DeviceBuffer<u8>,
        batch: usize,
        workspace: &mut DeviceBuffer<u8>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let Some(weight) = linear.cutlass_nvfp4.as_ref() else {
            return Err(AegisError::InvalidPlan(format!(
                "CUTLASS FP4 inference requested for `{}`, but no CUDA_R_4F_E2M1+UE4M3 resident layout was materialized",
                linear.name
            )));
        };
        let expected_a = Self::cutlass_nvfp4_activation_payload_bytes(batch, linear.cols)?;
        let expected_a_sf = Self::cutlass_nvfp4_activation_scale_bytes(batch, linear.cols)?;
        let expected_d = checked_len("output", batch, linear.rows)?;
        if activation_payload.len() < expected_a
            || activation_scales.len() < expected_a_sf
            || output.len() < expected_d
        {
            return Err(AegisError::InvalidPlan(format!(
                "CUTLASS FP4 GEMM buffers too small for `{}`: a={} expected_a={} a_sf={} expected_a_sf={} d={} expected_d={}",
                linear.name,
                activation_payload.len(),
                expected_a,
                activation_scales.len(),
                expected_a_sf,
                output.len(),
                expected_d
            )));
        }
        let expected_b = linear.rows * (linear.cols / 2);
        if weight.payload_e2m1.len() != expected_b
            || weight.layout.logical_n != linear.rows
            || weight.layout.logical_k != linear.cols
        {
            return Err(AegisError::InvalidPlan(format!(
                "CUTLASS FP4 resident weight shape mismatch for `{}`",
                linear.name
            )));
        }

        let workspace_required =
            self.cutlass_nvfp4_workspace_bytes(batch, linear.rows, linear.cols)?;
        if workspace.len() < workspace_required.max(1) {
            return Err(AegisError::InvalidPlan(format!(
                "CUTLASS FP4 workspace too small for `{}`: got={} required={}",
                linear.name,
                workspace.len(),
                workspace_required
            )));
        }

        let workspace_len = workspace.len();
        let (a_ptr, _a_read) = activation_payload.slice.device_ptr(&self.stream);
        let (b_ptr, _b_read) = weight.payload_e2m1.device_ptr(&self.stream);
        let (a_sf_ptr, _a_sf_read) = activation_scales.slice.device_ptr(&self.stream);
        let (b_sf_ptr, _b_sf_read) = weight.scales_ue4m3.device_ptr(&self.stream);
        let (workspace_ptr, _workspace_write) = workspace.slice.device_ptr_mut(&self.stream);
        let (d_ptr, _d_write) = output.slice.device_ptr_mut(&self.stream);
        let stream = self.stream.cu_stream().cast::<c_void>();
        unsafe {
            cutlass_bridge::gemm_f32(
                a_ptr as *const c_void,
                b_ptr as *const c_void,
                a_sf_ptr as *const c_void,
                b_sf_ptr as *const c_void,
                d_ptr as *mut f32,
                workspace_ptr as *mut c_void,
                workspace_len,
                i32_arg("m", batch)?,
                i32_arg("n", linear.rows)?,
                i32_arg("k", linear.cols)?,
                linear.output_scale,
                stream,
            )
        }
        .map_err(|error| {
            AegisError::Unsupported(format!(
                "CUTLASS FP4 GEMM failed for `{}`: {error}",
                linear.name
            ))
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matmul_cutlass_nvfp4_pair_prepacked_prefill_device(
        &self,
        first: &DeviceNvfp4Linear,
        second: &DeviceNvfp4Linear,
        activation_payload: &DeviceBuffer<u8>,
        activation_scales: &DeviceBuffer<u8>,
        batch: usize,
        workspace: &mut DeviceBuffer<u8>,
        first_output: &mut DeviceBuffer<f32>,
        second_output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if first.cols != second.cols {
            return Err(AegisError::InvalidPlan(format!(
                "CUTLASS FP4 paired GEMM shape mismatch: first={}x{} second={}x{}",
                first.rows, first.cols, second.rows, second.cols
            )));
        }
        let Some(first_weight) = first.cutlass_nvfp4.as_ref() else {
            return Err(AegisError::InvalidPlan(format!(
                "CUTLASS FP4 paired GEMM requested for `{}`, but no resident layout was materialized",
                first.name
            )));
        };
        let Some(second_weight) = second.cutlass_nvfp4.as_ref() else {
            return Err(AegisError::InvalidPlan(format!(
                "CUTLASS FP4 paired GEMM requested for `{}`, but no resident layout was materialized",
                second.name
            )));
        };
        let expected_a = Self::cutlass_nvfp4_activation_payload_bytes(batch, first.cols)?;
        let expected_a_sf = Self::cutlass_nvfp4_activation_scale_bytes(batch, first.cols)?;
        let expected_first = checked_len("paired first output", batch, first.rows)?;
        let expected_second = checked_len("paired second output", batch, second.rows)?;
        if activation_payload.len() < expected_a
            || activation_scales.len() < expected_a_sf
            || first_output.len() < expected_first
            || second_output.len() < expected_second
        {
            return Err(AegisError::InvalidPlan(format!(
                "CUTLASS FP4 paired GEMM buffers too small: a={} expected_a={} a_sf={} expected_a_sf={} first_out={} expected_first={} second_out={} expected_second={}",
                activation_payload.len(),
                expected_a,
                activation_scales.len(),
                expected_a_sf,
                first_output.len(),
                expected_first,
                second_output.len(),
                expected_second
            )));
        }
        let first_workspace = self.cutlass_nvfp4_workspace_bytes(batch, first.rows, first.cols)?;
        let second_workspace =
            self.cutlass_nvfp4_workspace_bytes(batch, second.rows, second.cols)?;
        let workspace_required = first_workspace.max(second_workspace);
        if workspace.len() < workspace_required.max(1) {
            return Err(AegisError::InvalidPlan(format!(
                "CUTLASS FP4 paired workspace too small: got={} required={}",
                workspace.len(),
                workspace_required
            )));
        }

        let workspace_len = workspace.len();
        let (a_ptr, _a_read) = activation_payload.slice.device_ptr(&self.stream);
        let (a_sf_ptr, _a_sf_read) = activation_scales.slice.device_ptr(&self.stream);
        let (first_b_ptr, _first_b_read) = first_weight.payload_e2m1.device_ptr(&self.stream);
        let (first_b_sf_ptr, _first_b_sf_read) = first_weight.scales_ue4m3.device_ptr(&self.stream);
        let (second_b_ptr, _second_b_read) = second_weight.payload_e2m1.device_ptr(&self.stream);
        let (second_b_sf_ptr, _second_b_sf_read) =
            second_weight.scales_ue4m3.device_ptr(&self.stream);
        let (workspace_ptr, _workspace_write) = workspace.slice.device_ptr_mut(&self.stream);
        let (first_d_ptr, _first_d_write) = first_output.slice.device_ptr_mut(&self.stream);
        let (second_d_ptr, _second_d_write) = second_output.slice.device_ptr_mut(&self.stream);
        let stream = self.stream.cu_stream().cast::<c_void>();
        unsafe {
            cutlass_bridge::gemm2_f32(
                a_ptr as *const c_void,
                first_b_ptr as *const c_void,
                second_b_ptr as *const c_void,
                a_sf_ptr as *const c_void,
                first_b_sf_ptr as *const c_void,
                second_b_sf_ptr as *const c_void,
                first_d_ptr as *mut f32,
                second_d_ptr as *mut f32,
                workspace_ptr as *mut c_void,
                workspace_len,
                i32_arg("m", batch)?,
                i32_arg("first n", first.rows)?,
                i32_arg("second n", second.rows)?,
                i32_arg("k", first.cols)?,
                first.output_scale,
                second.output_scale,
                stream,
            )
        }
        .map_err(|error| {
            AegisError::Unsupported(format!(
                "CUTLASS FP4 paired GEMM failed for `{}` + `{}`: {error}",
                first.name, second.name
            ))
        })
    }
}

use std::ffi::c_void;

use cudarc::driver::{DevicePtr, DevicePtrMut};

use super::CudaRuntime;
use crate::cuda::{DeviceBuffer, DeviceNvfp4Linear, cutlass_bridge};
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::planning::runtime::KernelFamily;

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
    /// True iff the CUTLASS NVFP4 grouped GEMM TU was compiled into the
    /// bridge archive (env AEGIS_CUTLASS_NVFP4_GROUPED_BUILD=1 at build
    /// time). The dispatcher uses this to fail fast if the runtime flag
    /// is set but the symbols are absent.
    pub fn cutlass_nvfp4_moe_grouped_built() -> bool {
        cutlass_bridge::moe_grouped_supported()
    }

    /// Per-group CUTLASS blob sizes. Required to pre-size the
    /// stride/layout/problem-shape buffers in the MoE prefill scratch.
    /// Returns `Err` if the build was not compiled with CUTLASS grouped
    /// support.
    #[cfg(aegis_cutlass_nvfp4_grouped)]
    pub fn cutlass_nvfp4_moe_grouped_blob_sizes(&self) -> Result<CutlassMoeGroupedBlobSizes> {
        let sz = cutlass_bridge::moe_grouped_blob_sizes()
            .map_err(AegisError::Unsupported)?;
        Ok(CutlassMoeGroupedBlobSizes {
            stride_a: sz.stride_a,
            stride_b: sz.stride_b,
            stride_d: sz.stride_d,
            layout_sfa: sz.layout_sfa,
            layout_sfb: sz.layout_sfb,
            problem_shape: sz.problem_shape,
        })
    }

    /// SFA / SFB byte sizes for one problem shape. Used to size the
    /// per-group quantized-activation SFA buffer and the swizzled
    /// weight-scale SFB buffer worst-case at executor init.
    #[cfg(aegis_cutlass_nvfp4_grouped)]
    pub fn cutlass_nvfp4_moe_grouped_sfa_sfb_bytes(
        &self,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(usize, usize)> {
        let m_i = i32_arg("m", m)?;
        let n_i = i32_arg("n", n)?;
        let k_i = i32_arg("k", k)?;
        cutlass_bridge::moe_grouped_sfa_sfb_bytes(m_i, n_i, k_i)
            .map_err(AegisError::Unsupported)
    }
}

/// Public mirror of `cutlass_bridge::MoeGroupedBlobSizes` so callers
/// outside the `cuda::cutlass_bridge` module can size scratch buffers.
#[cfg(aegis_cutlass_nvfp4_grouped)]
#[derive(Debug, Clone, Copy)]
pub struct CutlassMoeGroupedBlobSizes {
    pub stride_a: usize,
    pub stride_b: usize,
    pub stride_d: usize,
    pub layout_sfa: usize,
    pub layout_sfb: usize,
    pub problem_shape: usize,
}

impl CudaRuntime {
    /// Dummy placeholder so the symbol exists even when the CUTLASS
    /// grouped build is disabled; gated callers use the cfg flag to
    /// avoid calling this.
    #[cfg(not(aegis_cutlass_nvfp4_grouped))]
    pub fn cutlass_nvfp4_moe_grouped_blob_sizes(&self) -> Result<()> {
        Err(AegisError::Unsupported(
            "CUTLASS NVFP4 grouped not compiled into this build".into(),
        ))
    }

    /// Quantize one or more groups of f32 activations into NVFP4 e2m1 +
    /// ue4m3 scale-factor blob for CUTLASS grouped GEMM consumption.
    #[cfg(aegis_cutlass_nvfp4_grouped)]
    #[allow(clippy::too_many_arguments)]
    pub fn cutlass_moe_nvfp4_quantize_input_grouped(
        &self,
        input: &DeviceBuffer<f32>,
        cols: usize,
        num_groups: usize,
        token_offsets: &DeviceBuffer<u32>,
        payload_offsets: &DeviceBuffer<u64>,
        sfa_offsets: &DeviceBuffer<u64>,
        max_padded_rows_per_group: usize,
        payload_out: &mut DeviceBuffer<u8>,
        sfa_out: &mut DeviceBuffer<u8>,
    ) -> Result<()> {
        let (input_ptr, _) = input.slice.device_ptr(&self.stream);
        let (tok_ptr, _) = token_offsets.slice.device_ptr(&self.stream);
        let (po_ptr, _) = payload_offsets.slice.device_ptr(&self.stream);
        let (so_ptr, _) = sfa_offsets.slice.device_ptr(&self.stream);
        let (payload_ptr, _) = payload_out.slice.device_ptr_mut(&self.stream);
        let (sfa_ptr, _) = sfa_out.slice.device_ptr_mut(&self.stream);
        let stream = self.stream.cu_stream().cast::<c_void>();
        unsafe {
            cutlass_bridge::moe_grouped_quantize_input(
                input_ptr as *const f32,
                i32_arg("cols", cols)?,
                i32_arg("num_groups", num_groups)?,
                tok_ptr as *const u32,
                po_ptr as *const u64,
                so_ptr as *const u64,
                i32_arg("max_padded_rows_per_group", max_padded_rows_per_group)?,
                payload_ptr as *mut u8,
                sfa_ptr as *mut u8,
                stream,
            )
        }
        .map_err(AegisError::Unsupported)
    }

    /// Swizzle raw row-major weight-scale blobs (one per group, concatenated)
    /// into the CUTLASS SFB layout.
    #[cfg(aegis_cutlass_nvfp4_grouped)]
    #[allow(clippy::too_many_arguments)]
    pub fn cutlass_moe_nvfp4_swizzle_weight_scales_grouped(
        &self,
        src: &DeviceBuffer<u8>,
        rows_per_group: usize,
        src_cols: usize,
        num_groups: usize,
        src_offsets: &DeviceBuffer<u64>,
        dst_offsets: &DeviceBuffer<u64>,
        dst: &mut DeviceBuffer<u8>,
    ) -> Result<()> {
        let (src_ptr, _) = src.slice.device_ptr(&self.stream);
        let (src_off_ptr, _) = src_offsets.slice.device_ptr(&self.stream);
        let (dst_off_ptr, _) = dst_offsets.slice.device_ptr(&self.stream);
        let (dst_ptr, _) = dst.slice.device_ptr_mut(&self.stream);
        let stream = self.stream.cu_stream().cast::<c_void>();
        unsafe {
            cutlass_bridge::moe_grouped_swizzle_weight_scales(
                src_ptr as *const u8,
                i32_arg("rows_per_group", rows_per_group)?,
                i32_arg("src_cols", src_cols)?,
                i32_arg("num_groups", num_groups)?,
                src_off_ptr as *const u64,
                dst_off_ptr as *const u64,
                dst_ptr as *mut u8,
                stream,
            )
        }
        .map_err(AegisError::Unsupported)
    }

    /// Launch the CUTLASS NVFP4 grouped GEMM. Per-group pointer arrays,
    /// stride/layout/problem-shape blobs, alpha pointers, and workspace
    /// must already be uploaded.
    #[cfg(aegis_cutlass_nvfp4_grouped)]
    #[allow(clippy::too_many_arguments)]
    pub fn cutlass_moe_nvfp4_grouped_run(
        &self,
        num_groups: usize,
        problem_shapes: &DeviceBuffer<u8>,
        a_ptrs: &mut DeviceBuffer<u64>,
        b_ptrs: &mut DeviceBuffer<u64>,
        sfa_ptrs: &mut DeviceBuffer<u64>,
        sfb_ptrs: &mut DeviceBuffer<u64>,
        d_ptrs: &mut DeviceBuffer<u64>,
        stride_a: &mut DeviceBuffer<u8>,
        stride_b: &mut DeviceBuffer<u8>,
        stride_d: &mut DeviceBuffer<u8>,
        layout_sfa: &mut DeviceBuffer<u8>,
        layout_sfb: &mut DeviceBuffer<u8>,
        alpha_ptrs: &mut DeviceBuffer<u64>,
        workspace: &mut DeviceBuffer<u8>,
    ) -> Result<()> {
        let (ps_ptr, _) = problem_shapes.slice.device_ptr(&self.stream);
        let (a_pp, _) = a_ptrs.slice.device_ptr_mut(&self.stream);
        let (b_pp, _) = b_ptrs.slice.device_ptr_mut(&self.stream);
        let (sfa_pp, _) = sfa_ptrs.slice.device_ptr_mut(&self.stream);
        let (sfb_pp, _) = sfb_ptrs.slice.device_ptr_mut(&self.stream);
        let (d_pp, _) = d_ptrs.slice.device_ptr_mut(&self.stream);
        let (sa_ptr, _) = stride_a.slice.device_ptr_mut(&self.stream);
        let (sb_ptr, _) = stride_b.slice.device_ptr_mut(&self.stream);
        let (sd_ptr, _) = stride_d.slice.device_ptr_mut(&self.stream);
        let (lsfa_ptr, _) = layout_sfa.slice.device_ptr_mut(&self.stream);
        let (lsfb_ptr, _) = layout_sfb.slice.device_ptr_mut(&self.stream);
        let (alpha_pp, _) = alpha_ptrs.slice.device_ptr_mut(&self.stream);
        let (ws_ptr, _) = workspace.slice.device_ptr_mut(&self.stream);
        let ws_bytes = workspace.len();
        let stream = self.stream.cu_stream().cast::<c_void>();
        unsafe {
            cutlass_bridge::moe_grouped_run(
                i32_arg("num_groups", num_groups)?,
                ps_ptr as *mut c_void,
                std::ptr::null(),
                a_pp as *mut *mut c_void,
                b_pp as *mut *mut c_void,
                sfa_pp as *mut *mut c_void,
                sfb_pp as *mut *mut c_void,
                d_pp as *mut *mut c_void,
                sa_ptr as *mut c_void,
                sb_ptr as *mut c_void,
                sd_ptr as *mut c_void,
                lsfa_ptr as *mut c_void,
                lsfb_ptr as *mut c_void,
                alpha_pp as *mut *mut f32,
                ws_ptr as *mut c_void,
                ws_bytes,
                stream,
            )
        }
        .map_err(AegisError::Unsupported)
    }

    /// Get raw device pointer (u64) for an f32 buffer. Used by the MoE
    /// dispatcher to build per-group pointer arrays without exposing the
    /// inner cudarc slice type to non-`cuda::` callers.
    #[cfg(aegis_cutlass_nvfp4_grouped)]
    pub fn device_ptr_f32(&self, buf: &DeviceBuffer<f32>) -> u64 {
        let (p, _) = buf.slice.device_ptr(&self.stream);
        p
    }
    #[cfg(aegis_cutlass_nvfp4_grouped)]
    pub fn device_ptr_f32_mut(&self, buf: &mut DeviceBuffer<f32>) -> u64 {
        let (p, _) = buf.slice.device_ptr_mut(&self.stream);
        p
    }
    #[cfg(aegis_cutlass_nvfp4_grouped)]
    pub fn device_ptr_u8(&self, buf: &DeviceBuffer<u8>) -> u64 {
        let (p, _) = buf.slice.device_ptr(&self.stream);
        p
    }
    #[cfg(aegis_cutlass_nvfp4_grouped)]
    pub fn device_ptr_u8_mut(&self, buf: &mut DeviceBuffer<u8>) -> u64 {
        let (p, _) = buf.slice.device_ptr_mut(&self.stream);
        p
    }

    /// Variant of `cutlass_moe_nvfp4_quantize_input_grouped` that takes
    /// pre-uploaded multi-group offset buffers and an `offset_index` selecting
    /// which group's entry to use. Lets the caller upload `2*N` token-offsets,
    /// `N` payload-offsets, `N` sfa-offsets in a SINGLE H2D, then issue
    /// `N` quantize launches without touching the offset buffers.
    /// The kernel always runs with `num_groups=1` (reads `[0]` and `[1]`
    /// for token_offsets, `[0]` for payload/sfa offsets — strided via the
    /// raw pointer offset).
    #[cfg(aegis_cutlass_nvfp4_grouped)]
    #[allow(clippy::too_many_arguments)]
    pub fn cutlass_moe_nvfp4_quantize_input_single_strided(
        &self,
        input: &DeviceBuffer<f32>,
        cols: usize,
        token_offsets_base: &DeviceBuffer<u32>,
        token_offsets_idx: usize, // entry index for [start, end] pair = 2*token_offsets_idx
        payload_offsets_base: &DeviceBuffer<u64>,
        payload_offsets_idx: usize,
        sfa_offsets_base: &DeviceBuffer<u64>,
        sfa_offsets_idx: usize,
        max_padded_rows_per_group: usize,
        payload_out: &mut DeviceBuffer<u8>,
        sfa_out: &mut DeviceBuffer<u8>,
    ) -> Result<()> {
        let (input_ptr, _) = input.slice.device_ptr(&self.stream);
        let (tok_ptr, _) = token_offsets_base.slice.device_ptr(&self.stream);
        let (po_ptr, _) = payload_offsets_base.slice.device_ptr(&self.stream);
        let (so_ptr, _) = sfa_offsets_base.slice.device_ptr(&self.stream);
        let (payload_ptr, _) = payload_out.slice.device_ptr_mut(&self.stream);
        let (sfa_ptr, _) = sfa_out.slice.device_ptr_mut(&self.stream);
        let stream = self.stream.cu_stream().cast::<c_void>();
        let tok_off = tok_ptr + (token_offsets_idx as u64) * 4 /* u32 */;
        let payload_off = po_ptr + (payload_offsets_idx as u64) * 8 /* u64 */;
        let sfa_off = so_ptr + (sfa_offsets_idx as u64) * 8;
        unsafe {
            cutlass_bridge::moe_grouped_quantize_input(
                input_ptr as *const f32,
                i32_arg("cols", cols)?,
                1, // num_groups
                tok_off as *const u32,
                payload_off as *const u64,
                sfa_off as *const u64,
                i32_arg("max_padded_rows_per_group", max_padded_rows_per_group)?,
                payload_ptr as *mut u8,
                sfa_ptr as *mut u8,
                stream,
            )
        }
        .map_err(AegisError::Unsupported)
    }

    /// Compute one group's CUTLASS internal stride / layout blobs for
    /// a given (m, n, k). Used to pre-populate the per-group blob arrays
    /// before launching `cutlass_moe_nvfp4_grouped_run`.
    #[cfg(aegis_cutlass_nvfp4_grouped)]
    #[allow(clippy::too_many_arguments)]
    pub fn cutlass_moe_nvfp4_compute_strides(
        &self,
        m: usize, n: usize, k: usize,
        stride_a_out: &mut [u8],
        stride_b_out: &mut [u8],
        stride_d_out: &mut [u8],
        layout_sfa_out: &mut [u8],
        layout_sfb_out: &mut [u8],
    ) -> Result<()> {
        cutlass_bridge::moe_grouped_compute_strides(
            i32_arg("m", m)?, i32_arg("n", n)?, i32_arg("k", k)?,
            stride_a_out, stride_b_out, stride_d_out, layout_sfa_out, layout_sfb_out,
        )
        .map_err(AegisError::Unsupported)
    }

    pub fn cutlass_nvfp4_activation_payload_bytes(rows: usize, cols: usize) -> Result<usize> {
        if !cols.is_multiple_of(32) {
            return Err(AegisError::InvalidPlan(format!(
                "CUTLASS NVFP4 activation payload requires cols divisible by 32, got {cols}"
            )));
        }
        checked_len("activation payload", rows, cols / 2)
    }

    pub fn cutlass_nvfp4_activation_scale_bytes(rows: usize, cols: usize) -> Result<usize> {
        if !cols.is_multiple_of(32) {
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

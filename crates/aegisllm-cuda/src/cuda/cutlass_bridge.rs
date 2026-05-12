use std::ffi::c_void;
use std::os::raw::{c_char, c_int};

unsafe extern "C" {
    fn aegis_cutlass_fp4_sm120_workspace_size(
        m: c_int,
        n: c_int,
        k: c_int,
        workspace_bytes: *mut usize,
        error: *mut c_char,
        error_len: usize,
    ) -> c_int;

    fn aegis_cutlass_fp4_sm120_gemm_f32(
        a: *const c_void,
        b: *const c_void,
        a_sf: *const c_void,
        b_sf: *const c_void,
        d: *mut f32,
        workspace: *mut c_void,
        workspace_bytes: usize,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: f32,
        stream: *mut c_void,
        error: *mut c_char,
        error_len: usize,
    ) -> c_int;

    fn aegis_cutlass_fp4_sm120_gemm2_f32(
        a: *const c_void,
        b0: *const c_void,
        b1: *const c_void,
        a_sf: *const c_void,
        b0_sf: *const c_void,
        b1_sf: *const c_void,
        d0: *mut f32,
        d1: *mut f32,
        workspace: *mut c_void,
        workspace_bytes: usize,
        m: c_int,
        n0: c_int,
        n1: c_int,
        k: c_int,
        alpha0: f32,
        alpha1: f32,
        stream: *mut c_void,
        error: *mut c_char,
        error_len: usize,
    ) -> c_int;

    fn aegis_cutlass_fp4_quantize_f32(
        input: *const f32,
        rows: c_int,
        cols: c_int,
        payload: *mut u8,
        scales: *mut u8,
        stream: *mut c_void,
        error: *mut c_char,
        error_len: usize,
    ) -> c_int;

    fn aegis_cutlass_fp4_swiglu_quantize_f32(
        gate: *const f32,
        up: *const f32,
        rows: c_int,
        cols: c_int,
        payload: *mut u8,
        scales: *mut u8,
        stream: *mut c_void,
        error: *mut c_char,
        error_len: usize,
    ) -> c_int;

}

pub(super) fn workspace_size(m: i32, n: i32, k: i32) -> Result<usize, String> {
    let mut workspace_bytes = 0usize;
    let mut error = ErrorBuffer::new();
    let code = unsafe {
        aegis_cutlass_fp4_sm120_workspace_size(
            m,
            n,
            k,
            &mut workspace_bytes,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    status(code, &error)?;
    Ok(workspace_bytes)
}

#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn gemm_f32(
    a: *const c_void,
    b: *const c_void,
    a_sf: *const c_void,
    b_sf: *const c_void,
    d: *mut f32,
    workspace: *mut c_void,
    workspace_bytes: usize,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    stream: *mut c_void,
) -> Result<(), String> {
    let mut error = ErrorBuffer::new();
    let code = unsafe {
        aegis_cutlass_fp4_sm120_gemm_f32(
            a,
            b,
            a_sf,
            b_sf,
            d,
            workspace,
            workspace_bytes,
            m,
            n,
            k,
            alpha,
            stream,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    status(code, &error)
}

#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn gemm2_f32(
    a: *const c_void,
    b0: *const c_void,
    b1: *const c_void,
    a_sf: *const c_void,
    b0_sf: *const c_void,
    b1_sf: *const c_void,
    d0: *mut f32,
    d1: *mut f32,
    workspace: *mut c_void,
    workspace_bytes: usize,
    m: i32,
    n0: i32,
    n1: i32,
    k: i32,
    alpha0: f32,
    alpha1: f32,
    stream: *mut c_void,
) -> Result<(), String> {
    let mut error = ErrorBuffer::new();
    let code = unsafe {
        aegis_cutlass_fp4_sm120_gemm2_f32(
            a,
            b0,
            b1,
            a_sf,
            b0_sf,
            b1_sf,
            d0,
            d1,
            workspace,
            workspace_bytes,
            m,
            n0,
            n1,
            k,
            alpha0,
            alpha1,
            stream,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    status(code, &error)
}

pub(super) unsafe fn quantize_f32(
    input: *const f32,
    rows: i32,
    cols: i32,
    payload: *mut u8,
    scales: *mut u8,
    stream: *mut c_void,
) -> Result<(), String> {
    let mut error = ErrorBuffer::new();
    let code = unsafe {
        aegis_cutlass_fp4_quantize_f32(
            input,
            rows,
            cols,
            payload,
            scales,
            stream,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    status(code, &error)
}

pub(super) unsafe fn swiglu_quantize_f32(
    gate: *const f32,
    up: *const f32,
    rows: i32,
    cols: i32,
    payload: *mut u8,
    scales: *mut u8,
    stream: *mut c_void,
) -> Result<(), String> {
    let mut error = ErrorBuffer::new();
    let code = unsafe {
        aegis_cutlass_fp4_swiglu_quantize_f32(
            gate,
            up,
            rows,
            cols,
            payload,
            scales,
            stream,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    status(code, &error)
}

fn status(code: i32, error: &ErrorBuffer) -> Result<(), String> {
    if code == 0 {
        Ok(())
    } else {
        let message = error.message();
        if message.is_empty() {
            Err(format!("CUTLASS bridge failed with code {code}"))
        } else {
            Err(format!("CUTLASS bridge failed with code {code}: {message}"))
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CUTLASS NVFP4 grouped GEMM SM120 — FFI surface for the MoE prefill path.
// Linked only when build.rs was invoked with
// AEGIS_CUTLASS_NVFP4_GROUPED_BUILD=1 (cfg aegis_cutlass_nvfp4_grouped).
// See crates/aegisllm-cuda/src/cuda/cutlass_bridge_moe.cu for the C++ side.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(aegis_cutlass_nvfp4_grouped)]
unsafe extern "C" {
    fn aegis_cutlass_moe_nvfp4_stride_sizes_sm120(
        stride_a_bytes: *mut usize,
        stride_b_bytes: *mut usize,
        stride_d_bytes: *mut usize,
        layout_sfa_bytes: *mut usize,
        layout_sfb_bytes: *mut usize,
        problem_shape_bytes: *mut usize,
    ) -> c_int;

    fn aegis_cutlass_moe_nvfp4_sfa_sfb_bytes_sm120(
        m: c_int,
        n: c_int,
        k: c_int,
        sfa_bytes_out: *mut usize,
        sfb_bytes_out: *mut usize,
    ) -> c_int;

    fn aegis_cutlass_moe_nvfp4_quantize_input_grouped(
        input: *const f32,
        cols: c_int,
        num_groups: c_int,
        token_offsets_device: *const u32,
        payload_offsets_device: *const u64,
        sfa_offsets_device: *const u64,
        max_padded_rows_per_group: c_int,
        payload_out: *mut u8,
        sfa_out: *mut u8,
        stream: *mut c_void,
    ) -> c_int;

    fn aegis_cutlass_moe_nvfp4_swizzle_weight_scales_grouped(
        src: *const u8,
        rows_per_group: c_int,
        src_cols: c_int,
        num_groups: c_int,
        src_offsets_device: *const u64,
        dst_offsets_device: *const u64,
        dst: *mut u8,
        stream: *mut c_void,
    ) -> c_int;

    fn aegis_cutlass_moe_nvfp4_compute_strides_sm120(
        m: c_int,
        n: c_int,
        k: c_int,
        stride_a_out: *mut c_void,
        stride_b_out: *mut c_void,
        stride_d_out: *mut c_void,
        layout_sfa_out: *mut c_void,
        layout_sfb_out: *mut c_void,
    ) -> c_int;

    fn aegis_cutlass_moe_nvfp4_workspace_size_sm120(
        num_groups: c_int,
        device_problem_sizes: *mut c_void,
        host_problem_sizes_or_null: *const c_void,
        workspace_bytes: *mut usize,
        error: *mut c_char,
        error_len: usize,
    ) -> c_int;

    #[allow(clippy::too_many_arguments)]
    fn aegis_cutlass_moe_nvfp4_grouped_gemm_sm120(
        num_groups: c_int,
        device_problem_sizes: *mut c_void,
        host_problem_sizes_or_null: *const c_void,
        ptr_a: *mut *mut c_void,
        ptr_b: *mut *mut c_void,
        ptr_sfa: *mut *mut c_void,
        ptr_sfb: *mut *mut c_void,
        ptr_d: *mut *mut c_void,
        stride_a: *mut c_void,
        stride_b: *mut c_void,
        stride_d: *mut c_void,
        layout_sfa: *mut c_void,
        layout_sfb: *mut c_void,
        alpha_device: *mut *mut f32,
        workspace: *mut c_void,
        workspace_bytes: usize,
        stream: *mut c_void,
        error: *mut c_char,
        error_len: usize,
    ) -> c_int;
}

/// Per-group blob sizes returned by the C++ side. Each blob is opaque
/// (CUTLASS Internal{Stride,Layout}*) and must be uploaded as-is.
#[cfg(aegis_cutlass_nvfp4_grouped)]
#[derive(Debug, Clone, Copy)]
pub(super) struct MoeGroupedBlobSizes {
    pub stride_a: usize,
    pub stride_b: usize,
    pub stride_d: usize,
    pub layout_sfa: usize,
    pub layout_sfb: usize,
    pub problem_shape: usize,
}

#[cfg(aegis_cutlass_nvfp4_grouped)]
pub(super) fn moe_grouped_blob_sizes() -> Result<MoeGroupedBlobSizes, String> {
    let mut sizes = MoeGroupedBlobSizes {
        stride_a: 0,
        stride_b: 0,
        stride_d: 0,
        layout_sfa: 0,
        layout_sfb: 0,
        problem_shape: 0,
    };
    let code = unsafe {
        aegis_cutlass_moe_nvfp4_stride_sizes_sm120(
            &mut sizes.stride_a,
            &mut sizes.stride_b,
            &mut sizes.stride_d,
            &mut sizes.layout_sfa,
            &mut sizes.layout_sfb,
            &mut sizes.problem_shape,
        )
    };
    if code != 0 {
        return Err(format!(
            "CUTLASS MoE NVFP4 grouped blob sizes query failed with code {code}"
        ));
    }
    Ok(sizes)
}

/// Compute one group's per-shape stride/layout blobs. The caller is
/// expected to pre-size each buffer using `moe_grouped_blob_sizes()`.
#[cfg(aegis_cutlass_nvfp4_grouped)]
#[allow(clippy::too_many_arguments)]
pub(super) fn moe_grouped_compute_strides(
    m: i32,
    n: i32,
    k: i32,
    stride_a_out: &mut [u8],
    stride_b_out: &mut [u8],
    stride_d_out: &mut [u8],
    layout_sfa_out: &mut [u8],
    layout_sfb_out: &mut [u8],
) -> Result<(), String> {
    let code = unsafe {
        aegis_cutlass_moe_nvfp4_compute_strides_sm120(
            m,
            n,
            k,
            stride_a_out.as_mut_ptr().cast(),
            stride_b_out.as_mut_ptr().cast(),
            stride_d_out.as_mut_ptr().cast(),
            layout_sfa_out.as_mut_ptr().cast(),
            layout_sfb_out.as_mut_ptr().cast(),
        )
    };
    if code != 0 {
        return Err(format!(
            "CUTLASS MoE NVFP4 compute_strides failed (m={m}, n={n}, k={k}) with code {code}"
        ));
    }
    Ok(())
}

/// Query the workspace bytes required for a grouped GEMM launch. The
/// device-side problem-size array must already be uploaded; the host
/// copy is optional (pass null to defer the grid sizing to the kernel).
#[cfg(aegis_cutlass_nvfp4_grouped)]
pub(super) unsafe fn moe_grouped_workspace_size(
    num_groups: i32,
    device_problem_sizes: *mut c_void,
    host_problem_sizes_or_null: *const c_void,
) -> Result<usize, String> {
    let mut bytes: usize = 0;
    let mut error = ErrorBuffer::new();
    let code = unsafe {
        aegis_cutlass_moe_nvfp4_workspace_size_sm120(
            num_groups,
            device_problem_sizes,
            host_problem_sizes_or_null,
            &mut bytes,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if code != 0 {
        let msg = error.message();
        return Err(if msg.is_empty() {
            format!("CUTLASS MoE NVFP4 workspace_size failed with code {code}")
        } else {
            format!("CUTLASS MoE NVFP4 workspace_size failed with code {code}: {msg}")
        });
    }
    Ok(bytes)
}

/// Launch the CUTLASS NVFP4 grouped GEMM. All device-side arrays are
/// caller-owned. `alpha_device` is a device array of `num_groups`
/// `*mut f32` pointers, each pointing to a single per-group alpha
/// value (typically the expert's `output_scale`).
#[cfg(aegis_cutlass_nvfp4_grouped)]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn moe_grouped_run(
    num_groups: i32,
    device_problem_sizes: *mut c_void,
    host_problem_sizes_or_null: *const c_void,
    ptr_a: *mut *mut c_void,
    ptr_b: *mut *mut c_void,
    ptr_sfa: *mut *mut c_void,
    ptr_sfb: *mut *mut c_void,
    ptr_d: *mut *mut c_void,
    stride_a: *mut c_void,
    stride_b: *mut c_void,
    stride_d: *mut c_void,
    layout_sfa: *mut c_void,
    layout_sfb: *mut c_void,
    alpha_device: *mut *mut f32,
    workspace: *mut c_void,
    workspace_bytes: usize,
    stream: *mut c_void,
) -> Result<(), String> {
    let mut error = ErrorBuffer::new();
    let code = unsafe {
        aegis_cutlass_moe_nvfp4_grouped_gemm_sm120(
            num_groups,
            device_problem_sizes,
            host_problem_sizes_or_null,
            ptr_a,
            ptr_b,
            ptr_sfa,
            ptr_sfb,
            ptr_d,
            stride_a,
            stride_b,
            stride_d,
            layout_sfa,
            layout_sfb,
            alpha_device,
            workspace,
            workspace_bytes,
            stream,
            error.as_mut_ptr(),
            error.len(),
        )
    };
    if code != 0 {
        let msg = error.message();
        return Err(if msg.is_empty() {
            format!("CUTLASS MoE NVFP4 grouped GEMM run failed with code {code}")
        } else {
            format!("CUTLASS MoE NVFP4 grouped GEMM run failed with code {code}: {msg}")
        });
    }
    Ok(())
}

/// Query the per-expert SFA/SFB byte sizes for a given (M, N, K)
/// problem shape. Returns (sfa_bytes, sfb_bytes). SFA depends on (M, K),
/// SFB on (N, K); both expressed as
/// `cosize(Sm1xxBlkScaledConfig::tile_atom_to_shape_SF{A,B}) * sizeof(ElementSF)`.
#[cfg(aegis_cutlass_nvfp4_grouped)]
pub(super) fn moe_grouped_sfa_sfb_bytes(m: i32, n: i32, k: i32) -> Result<(usize, usize), String> {
    let mut sfa: usize = 0;
    let mut sfb: usize = 0;
    let code = unsafe {
        aegis_cutlass_moe_nvfp4_sfa_sfb_bytes_sm120(m, n, k, &mut sfa, &mut sfb)
    };
    if code != 0 {
        return Err(format!(
            "CUTLASS MoE NVFP4 sfa_sfb_bytes failed (m={m}, n={n}, k={k}) with code {code}"
        ));
    }
    Ok((sfa, sfb))
}

/// Launch the per-expert NVFP4 activation quantizer. All buffer offsets
/// are device-resident u64 arrays sized `num_groups`. `token_offsets` is
/// a u32 prefix-sum of length `num_groups+1`.
#[cfg(aegis_cutlass_nvfp4_grouped)]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn moe_grouped_quantize_input(
    input: *const f32,
    cols: i32,
    num_groups: i32,
    token_offsets_device: *const u32,
    payload_offsets_device: *const u64,
    sfa_offsets_device: *const u64,
    max_padded_rows_per_group: i32,
    payload_out: *mut u8,
    sfa_out: *mut u8,
    stream: *mut c_void,
) -> Result<(), String> {
    let code = unsafe {
        aegis_cutlass_moe_nvfp4_quantize_input_grouped(
            input,
            cols,
            num_groups,
            token_offsets_device,
            payload_offsets_device,
            sfa_offsets_device,
            max_padded_rows_per_group,
            payload_out,
            sfa_out,
            stream,
        )
    };
    if code != 0 {
        Err(format!(
            "CUTLASS MoE NVFP4 grouped input quantize failed with code {code}"
        ))
    } else {
        Ok(())
    }
}

/// Launch the per-expert weight-scale row-major → swizzled transform.
#[cfg(aegis_cutlass_nvfp4_grouped)]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn moe_grouped_swizzle_weight_scales(
    src: *const u8,
    rows_per_group: i32,
    src_cols: i32,
    num_groups: i32,
    src_offsets_device: *const u64,
    dst_offsets_device: *const u64,
    dst: *mut u8,
    stream: *mut c_void,
) -> Result<(), String> {
    let code = unsafe {
        aegis_cutlass_moe_nvfp4_swizzle_weight_scales_grouped(
            src,
            rows_per_group,
            src_cols,
            num_groups,
            src_offsets_device,
            dst_offsets_device,
            dst,
            stream,
        )
    };
    if code != 0 {
        Err(format!(
            "CUTLASS MoE NVFP4 grouped weight-scale swizzle failed with code {code}"
        ))
    } else {
        Ok(())
    }
}

#[cfg(not(aegis_cutlass_nvfp4_grouped))]
pub(super) fn moe_grouped_supported() -> bool {
    false
}

#[cfg(aegis_cutlass_nvfp4_grouped)]
pub(super) fn moe_grouped_supported() -> bool {
    true
}

struct ErrorBuffer {
    bytes: [u8; 512],
}

impl ErrorBuffer {
    fn new() -> Self {
        Self { bytes: [0; 512] }
    }

    fn as_mut_ptr(&mut self) -> *mut c_char {
        self.bytes.as_mut_ptr().cast()
    }

    fn len(&self) -> usize {
        self.bytes.len()
    }

    fn message(&self) -> String {
        let end = self
            .bytes
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(self.bytes.len());
        String::from_utf8_lossy(&self.bytes[..end]).into_owned()
    }
}

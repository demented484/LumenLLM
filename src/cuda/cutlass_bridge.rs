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

    fn aegis_cutlass_fp4_swiglu_quantize_grouped_f32(
        gate_up: *const f32,
        rows: c_int,
        cols: c_int,
        gate_scale: f32,
        up_scale: f32,
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

pub(super) unsafe fn swiglu_quantize_grouped_f32(
    gate_up: *const f32,
    rows: i32,
    cols: i32,
    gate_scale: f32,
    up_scale: f32,
    payload: *mut u8,
    scales: *mut u8,
    stream: *mut c_void,
) -> Result<(), String> {
    let mut error = ErrorBuffer::new();
    let code = unsafe {
        aegis_cutlass_fp4_swiglu_quantize_grouped_f32(
            gate_up,
            rows,
            cols,
            gate_scale,
            up_scale,
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

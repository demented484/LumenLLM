//! cuBLASLt-backed BF16 tensor-core matmul (Phase C of perf overhaul).
//!
//! Replaces the naive `aegis_bf16_matmul_reference_batched` shared-mem-reduction
//! kernel for BF16-weighted ops (attention Q/K/V/O, shared MLP within MoE, router,
//! lm_head). Inputs come in as F32 activations; weights are stored as BF16 (the
//! `DeviceBf16Matrix.values` slice is `CudaSlice<u16>` whose bit pattern is
//! BF16). The flow:
//!
//! 1. Convert F32 input → BF16 scratch (`aegis_f32_to_bf16` kernel).
//! 2. cuBLASLt BF16×BF16 → BF16 GEMM with F32 accumulation (Blackwell SM_120
//!    tensor cores: `mma.sync.aligned.kind::bf16.m16n8k16` ~150 TFLOPs).
//! 3. Convert BF16 output → F32 (`aegis_bf16_to_f32` kernel).
//!
//! The two conversion kernels are negligible in cost (one byte read/write per
//! element) compared to the GEMM itself, which is the actual hot loop.
//!
//! Shape convention: weight is `[rows, cols]` row-major BF16, input is
//! `[batch, cols]` row-major F32, output is `[batch, rows]` row-major F32.
//! Equivalent to row-major `output = input @ weight^T`. We feed cuBLASLt the
//! standard row-major-to-col-major-with-flipped-args trick.

use cudarc::cublaslt::{Matmul, MatmulConfig};

use super::CudaRuntime;
use super::map_cuda_err;
use crate::cuda::{DeviceBf16Matrix, DeviceBuffer, StandaloneFp8Linear};
use aegisllm_base::error::{AegisError, Result};

impl CudaRuntime {
    /// True when this BF16 weight matrix can be matmul'd via cuBLASLt (i.e. is
    /// VRAM-resident). Host-resident matrices still require the CPU rayon
    /// fallback (`matvec_bf16_host_resident_device`) since cuBLASLt cannot read
    /// host-pinned weights directly.
    pub(crate) fn cublaslt_bf16_enabled_for(&self, matrix: &DeviceBf16Matrix) -> bool {
        !matrix.is_host_resident()
    }

    /// Compute `output = input @ weight^T` via cuBLASLt BF16 tensor cores.
    ///
    /// * `weight` — BF16 `[rows, cols]` row-major, must be VRAM-resident.
    /// * `input` — F32 `[batch, cols]` row-major.
    /// * `batch` — number of token rows (M dimension of the row-major view).
    /// * `input_bf16` / `output_bf16` — scratch buffers sized for `batch*cols`
    ///   and `batch*rows` respectively. Reused across calls; the caller is
    ///   responsible for sizing them once at construction time.
    /// * `output` — F32 `[batch, rows]` row-major result.
    pub fn matmul_bf16_cublaslt_device(
        &self,
        weight: &DeviceBf16Matrix,
        input: &DeviceBuffer<f32>,
        batch: usize,
        input_bf16: &mut DeviceBuffer<u16>,
        output_bf16: &mut DeviceBuffer<u16>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if weight.is_host_resident() {
            return Err(AegisError::InvalidPlan(format!(
                "cuBLASLt BF16 GEMM requires VRAM-resident weight matrix `{}`; host-resident weights must use the CPU rayon path",
                weight.name
            )));
        }
        let rows = weight.rows;
        let cols = weight.cols;
        let in_len = batch
            .checked_mul(cols)
            .ok_or_else(|| AegisError::InvalidPlan("cublaslt bf16 input len overflow".into()))?;
        let out_len = batch
            .checked_mul(rows)
            .ok_or_else(|| AegisError::InvalidPlan("cublaslt bf16 output len overflow".into()))?;
        if input.len() < in_len {
            return Err(AegisError::InvalidPlan(format!(
                "cuBLASLt BF16 input shape: input={} need batch*cols={}*{}={}",
                input.len(), batch, cols, in_len
            )));
        }
        if input_bf16.len() < in_len || output_bf16.len() < out_len {
            return Err(AegisError::InvalidPlan(format!(
                "cuBLASLt BF16 scratch too small: input_bf16={} need {}, output_bf16={} need {}",
                input_bf16.len(), in_len, output_bf16.len(), out_len
            )));
        }
        if output.len() < out_len {
            return Err(AegisError::InvalidPlan(format!(
                "cuBLASLt BF16 output shape: output={} need batch*rows={}*{}={}",
                output.len(), batch, rows, out_len
            )));
        }

        // Step 1: F32 input → BF16 scratch.
        self.f32_to_bf16_device(input, in_len, input_bf16)?;

        // Step 2: BF16 GEMM via cuBLASLt.
        //
        // Row-major math: C[batch, rows] = A[batch, cols] * W^T[cols, rows].
        // cuBLASLt reads memory as col-major. Standard trick: feed `weight` as
        // the first matrix with transa=true (so its row-major (rows, cols)
        // layout, viewed col-major as (cols, rows), is logically transposed
        // back to (rows, cols)) and `input` as the second matrix transb=false
        // (its row-major (batch, cols) layout, viewed col-major, is (cols,
        // batch)).
        //
        // Output C is col-major (rows, batch) which is row-major (batch, rows). ✓
        let cfg = MatmulConfig {
            transa: true,
            transb: false,
            transc: false,
            m: rows as u64,
            n: batch as u64,
            k: cols as u64,
            alpha: 1.0,
            lda: cols as i64,
            ldb: cols as i64,
            beta: 0.0,
            ldc: rows as i64,
            stride_a: None,
            stride_b: None,
            stride_c: None,
            stride_bias: None,
            batch_size: None,
        };

        // Reinterpret CudaSlice<u16> ↔ CudaSlice<half::bf16>. half::bf16 is
        // `repr(transparent)` over u16; the buffers have identical layout.
        // Length matches the weight matrix exactly so transmute returns Some.
        let weight_view = unsafe { weight.values.transmute::<half::bf16>(weight.values.len()) }
            .ok_or_else(|| {
                AegisError::Unsupported(format!(
                    "weight u16 → bf16 transmute failed for `{}` (len={})",
                    weight.name, weight.values.len()
                ))
            })?;
        let in_view = unsafe { input_bf16.slice.transmute::<half::bf16>(in_len) }
            .ok_or_else(|| {
                AegisError::Unsupported("input u16 → bf16 transmute failed".into())
            })?;
        let mut out_view = unsafe { output_bf16.slice.transmute_mut::<half::bf16>(out_len) }
            .ok_or_else(|| {
                AegisError::Unsupported("output u16 → bf16 transmute failed".into())
            })?;

        unsafe {
            self.cublas_lt
                .matmul(cfg, &weight_view, &in_view, &mut out_view, None, None)
        }
        .map_err(|e| {
            AegisError::Unsupported(format!(
                "cuBLASLt BF16 matmul failed for `{}` (m={} n={} k={}): {e:?}",
                weight.name, rows, batch, cols
            ))
        })?;

        // Step 3: BF16 output → F32.
        self.bf16_to_f32_device(output_bf16, out_len, output)?;
        Ok(())
    }

    /// Variant of `matmul_bf16_cublaslt_device` that takes input already
    /// quantized to BF16. Used for QKV projection where Q, K, and V share the
    /// same input — caller quantizes once via `f32_to_bf16_device` and reuses
    /// `input_bf16` across three GEMMs, skipping two redundant conversions.
    /// Bit-identical to the f32-input variant for the same input bytes.
    pub fn matmul_bf16_cublaslt_with_input_bf16_device(
        &self,
        weight: &DeviceBf16Matrix,
        input_bf16: &DeviceBuffer<u16>,
        batch: usize,
        output_bf16: &mut DeviceBuffer<u16>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if weight.is_host_resident() {
            return Err(AegisError::InvalidPlan(format!(
                "cuBLASLt BF16 GEMM requires VRAM-resident weight matrix `{}`; host-resident weights must use the CPU rayon path",
                weight.name
            )));
        }
        let rows = weight.rows;
        let cols = weight.cols;
        let in_len = batch
            .checked_mul(cols)
            .ok_or_else(|| AegisError::InvalidPlan("cublaslt bf16 input len overflow".into()))?;
        let out_len = batch
            .checked_mul(rows)
            .ok_or_else(|| AegisError::InvalidPlan("cublaslt bf16 output len overflow".into()))?;
        if input_bf16.len() < in_len || output_bf16.len() < out_len {
            return Err(AegisError::InvalidPlan(format!(
                "cuBLASLt BF16 scratch too small: input_bf16={} need {}, output_bf16={} need {}",
                input_bf16.len(), in_len, output_bf16.len(), out_len
            )));
        }
        if output.len() < out_len {
            return Err(AegisError::InvalidPlan(format!(
                "cuBLASLt BF16 output shape: output={} need batch*rows={}*{}={}",
                output.len(), batch, rows, out_len
            )));
        }
        let cfg = MatmulConfig {
            transa: true,
            transb: false,
            transc: false,
            m: rows as u64,
            n: batch as u64,
            k: cols as u64,
            alpha: 1.0,
            lda: cols as i64,
            ldb: cols as i64,
            beta: 0.0,
            ldc: rows as i64,
            stride_a: None,
            stride_b: None,
            stride_c: None,
            stride_bias: None,
            batch_size: None,
        };
        let weight_view = unsafe { weight.values.transmute::<half::bf16>(weight.values.len()) }
            .ok_or_else(|| {
                AegisError::Unsupported(format!(
                    "weight u16 → bf16 transmute failed for `{}` (len={})",
                    weight.name, weight.values.len()
                ))
            })?;
        let in_view = unsafe { input_bf16.slice.transmute::<half::bf16>(in_len) }
            .ok_or_else(|| {
                AegisError::Unsupported("input u16 → bf16 transmute failed".into())
            })?;
        let mut out_view = unsafe { output_bf16.slice.transmute_mut::<half::bf16>(out_len) }
            .ok_or_else(|| {
                AegisError::Unsupported("output u16 → bf16 transmute failed".into())
            })?;
        unsafe {
            self.cublas_lt
                .matmul(cfg, &weight_view, &in_view, &mut out_view, None, None)
        }
        .map_err(|e| {
            AegisError::Unsupported(format!(
                "cuBLASLt BF16 matmul failed for `{}` (m={} n={} k={}): {e:?}",
                weight.name, rows, batch, cols
            ))
        })?;
        self.bf16_to_f32_device(output_bf16, out_len, output)?;
        Ok(())
    }

    /// Public wrapper around the F32→BF16 conversion kernel for callers that
    /// want to share a quantized input across multiple GEMMs.
    pub fn f32_to_bf16_into_device(
        &self,
        input: &DeviceBuffer<f32>,
        len: usize,
        out: &mut DeviceBuffer<u16>,
    ) -> Result<()> {
        self.f32_to_bf16_device(input, len, out)
    }

    /// Compute `output = input @ weight^T` for an FP8 standalone weight by
    /// dequantizing into a BF16 scratch and routing through the existing
    /// BF16 cuBLASLt tensor-core path. Activates Blackwell SM_120 BF16
    /// tensor cores (~150 TFLOPs) at the cost of one streaming dequant per
    /// call. Native FP8 tensor cores (~700 TFLOPs) require raw cuBLASLt FFI
    /// and will land as a follow-up.
    ///
    /// * `weight` — standalone FP8 `[rows, cols]`, VRAM-resident.
    /// * `weight_dequant_scratch` — BF16 scratch sized for `rows*cols`, reused
    ///   across all FP8 GEMMs in the same chunk; caller-allocated.
    pub fn matmul_fp8_via_bf16_cublaslt_device(
        &self,
        weight: &StandaloneFp8Linear,
        weight_dequant_scratch: &mut DeviceBuffer<u16>,
        input: &DeviceBuffer<f32>,
        batch: usize,
        input_bf16: &mut DeviceBuffer<u16>,
        output_bf16: &mut DeviceBuffer<u16>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let rows = weight.rows;
        let cols = weight.cols;
        let weight_elems = rows
            .checked_mul(cols)
            .ok_or_else(|| AegisError::InvalidPlan("fp8 dequant weight elem overflow".into()))?;
        if weight_dequant_scratch.len() < weight_elems {
            return Err(AegisError::InvalidPlan(format!(
                "fp8 cublaslt scratch too small for `{}`: have {} need {}",
                weight.name, weight_dequant_scratch.len(), weight_elems
            )));
        }
        let in_len = batch
            .checked_mul(cols)
            .ok_or_else(|| AegisError::InvalidPlan("fp8 cublaslt input len overflow".into()))?;
        let out_len = batch
            .checked_mul(rows)
            .ok_or_else(|| AegisError::InvalidPlan("fp8 cublaslt output len overflow".into()))?;
        if input.len() < in_len || input_bf16.len() < in_len
            || output_bf16.len() < out_len || output.len() < out_len
        {
            return Err(AegisError::InvalidPlan(format!(
                "fp8 cublaslt shape mismatch for `{}`: input={}/{} input_bf16={}/{} output_bf16={}/{} output={}/{}",
                weight.name,
                input.len(), in_len,
                input_bf16.len(), in_len,
                output_bf16.len(), out_len,
                output.len(), out_len,
            )));
        }

        // Step 1: FP8 weight → BF16 scratch (dequant).
        self.dequant_fp8_to_bf16_device(weight, weight_dequant_scratch)?;

        // Step 2: F32 input → BF16 scratch.
        self.f32_to_bf16_device(input, in_len, input_bf16)?;

        // Step 3: BF16 GEMM via cuBLASLt. Same row-major-via-col-major-flip
        // pattern as `matmul_bf16_cublaslt_device`.
        let cfg = MatmulConfig {
            transa: true,
            transb: false,
            transc: false,
            m: rows as u64,
            n: batch as u64,
            k: cols as u64,
            alpha: 1.0,
            lda: cols as i64,
            ldb: cols as i64,
            beta: 0.0,
            ldc: rows as i64,
            stride_a: None,
            stride_b: None,
            stride_c: None,
            stride_bias: None,
            batch_size: None,
        };

        let weight_view = unsafe {
            weight_dequant_scratch.slice.transmute::<half::bf16>(weight_elems)
        }
        .ok_or_else(|| {
            AegisError::Unsupported(format!(
                "fp8 dequant scratch u16→bf16 transmute failed for `{}` (len={})",
                weight.name, weight_elems
            ))
        })?;
        let in_view = unsafe { input_bf16.slice.transmute::<half::bf16>(in_len) }
            .ok_or_else(|| {
                AegisError::Unsupported("fp8 cublaslt input u16→bf16 transmute failed".into())
            })?;
        let mut out_view = unsafe { output_bf16.slice.transmute_mut::<half::bf16>(out_len) }
            .ok_or_else(|| {
                AegisError::Unsupported("fp8 cublaslt output u16→bf16 transmute failed".into())
            })?;

        unsafe {
            self.cublas_lt
                .matmul(cfg, &weight_view, &in_view, &mut out_view, None, None)
        }
        .map_err(|e| {
            AegisError::Unsupported(format!(
                "fp8-dequant cuBLASLt BF16 matmul failed for `{}` (m={} n={} k={}): {e:?}",
                weight.name, rows, batch, cols
            ))
        })?;

        // Step 4: BF16 output → F32.
        self.bf16_to_f32_device(output_bf16, out_len, output)?;
        Ok(())
    }

    /// Strided-batched F32 GEMM via cuBLASLt (TF32 tensor cores on SM_120).
    /// Used by the vision tower's bidirectional attention to compute
    /// `scores = Q @ K^T` and `out = scores @ V` as two batched GEMMs across
    /// `n_heads`, replacing the naive per-(head, q_row) `aegis_vision_bidi_attn`
    /// kernel that was bandwidth-bound at large `n_tok`.
    ///
    /// `*_offset` / `*_len` define a window into each buffer so callers can
    /// tile over Q-rows without slicing the full buffer (cudarc slices
    /// require non-overlapping mutable borrows — tiling logic is cleaner
    /// with explicit element offsets). All matrices are F32. cuBLASLt views
    /// memory column-major; the caller provides `lda/ldb/ldc` and `stride_*`
    /// in the column-major sense and chooses `transa/transb` so the
    /// effective math is the desired row-major operation.
    /// Strided-batched BF16 GEMM via cuBLASLt (BF16 mma.sync tensor cores on
    /// SM_120, ~3× the throughput of the TF32 F32 path for the same shape and
    /// 2× less memory traffic across the scores buffer). Same calling
    /// convention as the F32 variant; `a`/`b`/`c` are u16 buffers whose bit
    /// pattern is BF16. `aux_f32` is unused storage required so the underlying
    /// call site has somewhere to write — pass any small scratch.
    #[allow(clippy::too_many_arguments)]
    pub fn bf16_strided_batched_gemm_device(
        &self,
        a: &DeviceBuffer<u16>,
        a_offset: usize,
        a_len: usize,
        b: &DeviceBuffer<u16>,
        b_offset: usize,
        b_len: usize,
        c: &mut DeviceBuffer<u16>,
        c_offset: usize,
        c_len: usize,
        transa: bool,
        transb: bool,
        m: usize,
        n: usize,
        k: usize,
        lda: usize,
        ldb: usize,
        ldc: usize,
        stride_a: usize,
        stride_b: usize,
        stride_c: usize,
        batch_size: usize,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        let cfg = MatmulConfig {
            transa,
            transb,
            transc: false,
            m: m as u64,
            n: n as u64,
            k: k as u64,
            alpha,
            lda: lda as i64,
            ldb: ldb as i64,
            beta,
            ldc: ldc as i64,
            stride_a: Some(stride_a as i64),
            stride_b: Some(stride_b as i64),
            stride_c: Some(stride_c as i64),
            stride_bias: None,
            batch_size: Some(batch_size as i32),
        };
        let a_total = a.slice.len();
        let b_total = b.slice.len();
        let c_total = c.slice.len();
        let a_view = a.slice.try_slice(a_offset..a_offset + a_len)
            .ok_or_else(|| AegisError::InvalidPlan(format!(
                "bf16 batched gemm: a slice [{a_offset}..{}] OOB (len={a_total})",
                a_offset + a_len
            )))?;
        let b_view = b.slice.try_slice(b_offset..b_offset + b_len)
            .ok_or_else(|| AegisError::InvalidPlan(format!(
                "bf16 batched gemm: b slice [{b_offset}..{}] OOB (len={b_total})",
                b_offset + b_len
            )))?;
        let mut c_view = c.slice.try_slice_mut(c_offset..c_offset + c_len)
            .ok_or_else(|| AegisError::InvalidPlan(format!(
                "bf16 batched gemm: c slice [{c_offset}..{}] OOB (len={c_total})",
                c_offset + c_len
            )))?;
        // Reinterpret u16 ↔ half::bf16 (repr(transparent) — same layout).
        let a_bf16 = unsafe { a_view.transmute::<half::bf16>(a_len) }
            .ok_or_else(|| AegisError::Unsupported(
                "bf16 batched gemm: a u16 → bf16 transmute failed".into()))?;
        let b_bf16 = unsafe { b_view.transmute::<half::bf16>(b_len) }
            .ok_or_else(|| AegisError::Unsupported(
                "bf16 batched gemm: b u16 → bf16 transmute failed".into()))?;
        let mut c_bf16 = unsafe { c_view.transmute_mut::<half::bf16>(c_len) }
            .ok_or_else(|| AegisError::Unsupported(
                "bf16 batched gemm: c u16 → bf16 transmute failed".into()))?;
        unsafe {
            self.cublas_lt
                .matmul(cfg, &a_bf16, &b_bf16, &mut c_bf16, None, None)
        }
        .map_err(|e| {
            AegisError::Unsupported(format!(
                "cuBLASLt BF16 strided-batched matmul failed (m={m} n={n} k={k} \
                 lda={lda} ldb={ldb} ldc={ldc} stride_a={stride_a} stride_b={stride_b} \
                 stride_c={stride_c} batch={batch_size}): {e:?}"
            ))
        })?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn f32_strided_batched_gemm_device(
        &self,
        a: &DeviceBuffer<f32>,
        a_offset: usize,
        a_len: usize,
        b: &DeviceBuffer<f32>,
        b_offset: usize,
        b_len: usize,
        c: &mut DeviceBuffer<f32>,
        c_offset: usize,
        c_len: usize,
        transa: bool,
        transb: bool,
        m: usize,
        n: usize,
        k: usize,
        lda: usize,
        ldb: usize,
        ldc: usize,
        stride_a: usize,
        stride_b: usize,
        stride_c: usize,
        batch_size: usize,
        alpha: f32,
        beta: f32,
    ) -> Result<()> {
        let cfg = MatmulConfig {
            transa,
            transb,
            transc: false,
            m: m as u64,
            n: n as u64,
            k: k as u64,
            alpha,
            lda: lda as i64,
            ldb: ldb as i64,
            beta,
            ldc: ldc as i64,
            stride_a: Some(stride_a as i64),
            stride_b: Some(stride_b as i64),
            stride_c: Some(stride_c as i64),
            stride_bias: None,
            batch_size: Some(batch_size as i32),
        };
        let a_view = a.slice.try_slice(a_offset..a_offset + a_len)
            .ok_or_else(|| AegisError::InvalidPlan(format!(
                "f32 batched gemm: a slice [{a_offset}..{}] OOB (len={})",
                a_offset + a_len, a.slice.len()
            )))?;
        let b_view = b.slice.try_slice(b_offset..b_offset + b_len)
            .ok_or_else(|| AegisError::InvalidPlan(format!(
                "f32 batched gemm: b slice [{b_offset}..{}] OOB (len={})",
                b_offset + b_len, b.slice.len()
            )))?;
        let c_total = c.slice.len();
        let mut c_view = c.slice.try_slice_mut(c_offset..c_offset + c_len)
            .ok_or_else(|| AegisError::InvalidPlan(format!(
                "f32 batched gemm: c slice [{c_offset}..{}] OOB (len={c_total})",
                c_offset + c_len
            )))?;
        unsafe {
            self.cublas_lt
                .matmul(cfg, &a_view, &b_view, &mut c_view, None, None)
        }
        .map_err(|e| {
            AegisError::Unsupported(format!(
                "cuBLASLt F32 strided-batched matmul failed (m={m} n={n} k={k} \
                 lda={lda} ldb={ldb} ldc={ldc} stride_a={stride_a} stride_b={stride_b} \
                 stride_c={stride_c} batch={batch_size}): {e:?}"
            ))
        })?;
        Ok(())
    }
}

#[allow(dead_code)]
fn _ensure_compiles(_: &CudaRuntime) {
    // Placeholder so unused-import lints don't fire when the module is empty.
    let _ = map_cuda_err;
}

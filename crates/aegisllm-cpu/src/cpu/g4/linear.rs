//! Unified CPU linear projection for Gemma-4: BF16 (E2B/E4B/31B-dense checkpoints)
//! or NVFP4 (quantized checkpoints). Mirrors the CUDA `CudaLinear` enum
//! (`crates/aegisllm-cuda/src/executor/state.rs:47`), but only the two storage
//! formats the CPU path supports.
//!
//! Correctness-first: BF16 weights are kept as raw BF16 bytes inside
//! `Bf16Matrix` and converted to f32 lazily per matvec row (rayon-parallel),
//! exactly like `Bf16Matrix::matvec_into`. A fast blocked SIMD BF16 GEMM is a
//! follow-up; this version prioritizes matching the CUDA forward math.

use crate::cpu::CpuNvfp4Linear;
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::executor::tensors::Bf16Matrix;

#[derive(Debug)]
pub(crate) enum CpuLinear {
    Bf16(Bf16Matrix),
    Nvfp4(CpuNvfp4Linear),
}

impl CpuLinear {
    pub(crate) fn rows(&self) -> usize {
        match self {
            Self::Bf16(m) => m.rows,
            Self::Nvfp4(l) => l.rows,
        }
    }

    // Part of the documented CpuLinear contract; used by the batched-prefill
    // follow-up and shape validation.
    #[allow(dead_code)]
    pub(crate) fn cols(&self) -> usize {
        match self {
            Self::Bf16(m) => m.cols,
            Self::Nvfp4(l) => l.cols,
        }
    }

    /// Single-vector projection: `out[r] = Σ_c W[r,c] * input[c]`.
    pub(crate) fn matvec_into(&self, input: &[f32], out: &mut [f32]) -> Result<()> {
        match self {
            Self::Bf16(m) => m.matvec_into(input, out),
            Self::Nvfp4(l) => l.matvec_into(input, out),
        }
    }

    /// Batched projection over `batch` tokens. Input/output are row-major
    /// `[batch, cols]` / `[batch, rows]`. The BF16 path loops `matvec_into`
    /// per token (correctness-first; the NVFP4 path already dequantizes each
    /// weight row once and dots all tokens). Reserved for the batched-prefill
    /// follow-up (decode currently drives prefill per-token).
    #[allow(dead_code)]
    pub(crate) fn matmul_into(&self, input: &[f32], batch: usize, out: &mut [f32]) -> Result<()> {
        match self {
            Self::Bf16(m) => {
                let cols = m.cols;
                let rows = m.rows;
                if input.len() != batch * cols || out.len() != batch * rows {
                    return Err(AegisError::InvalidPlan(format!(
                        "bf16 matmul shape mismatch: expected input={} output={} (batch={} rows={} cols={})",
                        batch * cols,
                        batch * rows,
                        batch,
                        rows,
                        cols
                    )));
                }
                for token in 0..batch {
                    let in_row = &input[token * cols..(token + 1) * cols];
                    let out_row = &mut out[token * rows..(token + 1) * rows];
                    m.matvec_into(in_row, out_row)?;
                }
                Ok(())
            }
            Self::Nvfp4(l) => l.matmul_into(input, batch, out),
        }
    }
}

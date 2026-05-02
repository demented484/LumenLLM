use aegisllm_base::error::{AegisError, Result};
use super::simd;

pub(super) fn rms_norm_into(input: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let mean_square = simd::dot_f32(input, input) / input.len() as f32;
    let scale = 1.0 / (mean_square + eps).sqrt();
    simd::rms_scale(input, weight, out, scale);
}

pub(super) fn add_into(a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> {
    if a.len() != b.len() || a.len() != out.len() {
        return Err(AegisError::InvalidPlan("vector add shape mismatch".into()));
    }
    simd::add_into_simd(a, b, out);
    Ok(())
}

pub(super) fn swiglu_into(gate: &[f32], up: &[f32], out: &mut [f32]) -> Result<()> {
    if gate.len() != up.len() || gate.len() != out.len() {
        return Err(AegisError::InvalidPlan("swiglu shape mismatch".into()));
    }
    simd::swiglu_into_simd(gate, up, out);
    Ok(())
}

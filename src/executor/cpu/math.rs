use crate::error::{AegisError, Result};

pub(super) fn rms_norm_into(input: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let mean_square = input.iter().map(|value| value * value).sum::<f32>() / input.len() as f32;
    let scale = 1.0 / (mean_square + eps).sqrt();
    for i in 0..input.len() {
        out[i] = input[i] * scale * weight[i];
    }
}

pub(super) fn add_into(a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> {
    if a.len() != b.len() || a.len() != out.len() {
        return Err(AegisError::InvalidPlan("vector add shape mismatch".into()));
    }
    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
    Ok(())
}

pub(super) fn swiglu_into(gate: &[f32], up: &[f32], out: &mut [f32]) -> Result<()> {
    if gate.len() != up.len() || gate.len() != out.len() {
        return Err(AegisError::InvalidPlan("swiglu shape mismatch".into()));
    }
    for i in 0..gate.len() {
        out[i] = silu(gate[i]) * up[i];
    }
    Ok(())
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub(super) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

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

/// GeGLU (Gemma-4 / Qwen MLP activation): out = gelu_pytorch_tanh(gate) * up.
/// First CPU-side Gemma-4 primitive toward architecture-agnostic CPU inference.
#[allow(dead_code)] // wired by the upcoming Gemma-4 CPU forward (task #56)
pub(super) fn geglu_into(gate: &[f32], up: &[f32], out: &mut [f32]) -> Result<()> {
    if gate.len() != up.len() || gate.len() != out.len() {
        return Err(AegisError::InvalidPlan("geglu shape mismatch".into()));
    }
    simd::geglu_into_simd(gate, up, out);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gelu_tanh_matches_reference_values() {
        // HF gelu_pytorch_tanh reference points.
        let cases = [(0.0f32, 0.0f32), (1.0, 0.841_192), (-1.0, -0.158_808), (2.0, 1.954_598)];
        for (x, want) in cases {
            let got = simd::gelu_tanh_scalar(x);
            assert!((got - want).abs() < 1e-3, "gelu_tanh({x}) = {got}, want {want}");
        }
    }

    #[test]
    fn geglu_equals_gelu_gate_times_up() {
        let gate = [1.0f32, -1.0, 2.0, 0.5];
        let up = [2.0f32, 3.0, -1.0, 4.0];
        let mut out = [0.0f32; 4];
        geglu_into(&gate, &up, &mut out).unwrap();
        for i in 0..4 {
            // `geglu_into_simd` uses a vectorized tanh approximation; the libm
            // `gelu_tanh_scalar` reference is matched only to the kernel's accuracy
            // target (< 2e-3 abs), not bit-exactly.
            let want = simd::gelu_tanh_scalar(gate[i]) * up[i];
            assert!((out[i] - want).abs() < 2e-3);
        }
    }
}

//! Layer-block forward primitives (device-resident).
//!
//! These compose the per-primitive `*_device` functions from
//! [`super::forward`] into the layer-shaped operations a model actually
//! runs (e.g. dense MLP block: rms_norm → gate/up → swiglu → down →
//! residual). Inputs and outputs are persistent `wgpu::Buffer`s; nothing
//! goes to host between calls.

use aegisllm_base::error::{AegisError, Result};

use super::forward::{matmul_f32_device, residual_add_device, rms_norm_device, swiglu_device};
use super::loader::WgpuContext;
use super::state::WgpuLlamaState;

/// Weights for one dense (non-MoE) Llama-style MLP block, in device memory.
///
/// `norm_weight`: `[hidden_size]` rms-norm scale vector.
/// `gate_proj`, `up_proj`: `[intermediate_size, hidden_size]` row-major.
/// `down_proj`: `[hidden_size, intermediate_size]` row-major.
///
/// All buffers are f32 storage. NVFP4 / BF16 weight formats will land
/// alongside `forward_dense_mlp_block_quant_device` once the on-device
/// dequant pipe is wired into this path; for now this is the f32 reference
/// route used to validate the chain end-to-end.
pub struct WgpuDenseMlpWeights {
    pub norm_weight: wgpu::Buffer,
    pub gate_proj: wgpu::Buffer,
    pub up_proj: wgpu::Buffer,
    pub down_proj: wgpu::Buffer,
}

impl std::fmt::Debug for WgpuDenseMlpWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuDenseMlpWeights").finish_non_exhaustive()
    }
}

/// Run one dense-MLP block on the wgpu backend.
///
/// Input: `state.residual` holds the layer's input activation.
/// Output: `state.residual` is updated in place to `residual + mlp(residual)`.
///
/// Pipeline (all on device, single host readback at the very end of decode):
///   1. `post_normed = rms_norm(residual, norm_weight)`
///   2. `gate = matmul(post_normed, gate_proj^T)`
///   3. `up   = matmul(post_normed, up_proj^T)`
///   4. `swiglu_out = silu(gate) * up`
///   5. `mlp_out = matmul(swiglu_out, down_proj^T)`
///   6. `residual = residual + mlp_out`
pub fn forward_dense_mlp_block_device(
    ctx: &WgpuContext,
    state: &mut WgpuLlamaState,
    weights: &WgpuDenseMlpWeights,
    rms_norm_eps: f32,
) -> Result<()> {
    let hidden = state.hidden_size;
    let intermediate = state.intermediate_size;
    let residual = state
        .residual
        .as_ref()
        .ok_or_else(|| AegisError::InvalidPlan("WgpuLlamaState missing residual buffer".into()))?;
    let post_normed = state
        .post_normed
        .as_ref()
        .ok_or_else(|| AegisError::InvalidPlan("WgpuLlamaState missing post_normed buffer".into()))?;
    let gate = state
        .gate
        .as_ref()
        .ok_or_else(|| AegisError::InvalidPlan("WgpuLlamaState missing gate buffer".into()))?;
    let up = state
        .up
        .as_ref()
        .ok_or_else(|| AegisError::InvalidPlan("WgpuLlamaState missing up buffer".into()))?;
    let swiglu_out = state
        .swiglu_out
        .as_ref()
        .ok_or_else(|| AegisError::InvalidPlan("WgpuLlamaState missing swiglu_out buffer".into()))?;
    let mlp_out = state
        .mlp_out
        .as_ref()
        .ok_or_else(|| AegisError::InvalidPlan("WgpuLlamaState missing mlp_out buffer".into()))?;

    rms_norm_device(ctx, residual, &weights.norm_weight, post_normed, hidden, rms_norm_eps)?;
    matmul_f32_device(ctx, post_normed, &weights.gate_proj, gate, 1, intermediate, hidden)?;
    matmul_f32_device(ctx, post_normed, &weights.up_proj, up, 1, intermediate, hidden)?;
    swiglu_device(ctx, gate, up, swiglu_out, intermediate)?;
    matmul_f32_device(ctx, swiglu_out, &weights.down_proj, mlp_out, 1, hidden, intermediate)?;

    // residual += mlp_out  (read-modify-write the residual buffer; we
    // route through `post_normed` as a scratch since wgpu primitives
    // don't yet have an in-place add).
    residual_add_device(ctx, residual, mlp_out, post_normed, hidden)?;
    // Copy post_normed → residual to leave state ready for the next block.
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("dense_mlp_block writeback"),
    });
    enc.copy_buffer_to_buffer(post_normed, 0, residual, 0, (hidden * 4) as u64);
    ctx.queue.submit(std::iter::once(enc.finish()));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wgpu::forward::{download_f32_buf, upload_f32_buf};

    /// CPU reference: fwd(residual) = residual + W_down * silu(W_gate * rms_norm(residual)) * (W_up * rms_norm(residual)).
    /// Matches the shader convention: B is row-major `[N, K]`, so
    /// `out[i] = Σ_k normed[k] * B[i, k]`.
    fn cpu_dense_mlp(
        residual: &[f32],
        norm_w: &[f32],
        gate_w: &[f32], // [I, H]
        up_w: &[f32],   // [I, H]
        down_w: &[f32], // [H, I]
        h: usize,
        i: usize,
        eps: f32,
    ) -> Vec<f32> {
        let mean_sq: f32 = residual.iter().map(|v| v * v).sum::<f32>() / h as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        let normed: Vec<f32> = residual
            .iter()
            .zip(norm_w.iter())
            .map(|(v, w)| v * inv_rms * w)
            .collect();
        let mut gate = vec![0.0_f32; i];
        let mut up = vec![0.0_f32; i];
        for row in 0..i {
            let mut g = 0.0_f32;
            let mut u = 0.0_f32;
            for col in 0..h {
                g += normed[col] * gate_w[row * h + col];
                u += normed[col] * up_w[row * h + col];
            }
            gate[row] = g;
            up[row] = u;
        }
        let swig: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
            .collect();
        let mut mlp = vec![0.0_f32; h];
        for row in 0..h {
            let mut acc = 0.0_f32;
            for col in 0..i {
                acc += swig[col] * down_w[row * i + col];
            }
            mlp[row] = acc;
        }
        residual.iter().zip(mlp.iter()).map(|(r, m)| r + m).collect()
    }

    /// End-to-end: tiny synthetic dense-MLP block, GPU vs CPU agree within 1e-4.
    /// Gated behind `AEGIS_WGPU_SMOKE=1`.
    #[test]
    fn dense_mlp_block_matches_cpu_reference() {
        if std::env::var("AEGIS_WGPU_SMOKE").is_err() {
            eprintln!("skipping; set AEGIS_WGPU_SMOKE=1 to run on a host with Vulkan/Metal/D3D12");
            return;
        }
        let ctx = WgpuContext::new(0).expect("wgpu ctx");
        let h = 16;
        let i = 32;
        let eps = 1e-6_f32;

        // Deterministic small random inputs (seeded by index, no rand crate).
        let residual_host: Vec<f32> = (0..h).map(|k| ((k * 13 + 7) % 23) as f32 * 0.05 - 0.5).collect();
        let norm_w_host: Vec<f32> = (0..h).map(|k| 1.0 + (k as f32) * 0.01).collect();
        let gate_w_host: Vec<f32> = (0..(i * h))
            .map(|k| ((k * 17 + 3) % 31) as f32 * 0.02 - 0.3)
            .collect();
        let up_w_host: Vec<f32> = (0..(i * h))
            .map(|k| ((k * 19 + 5) % 29) as f32 * 0.02 - 0.25)
            .collect();
        let down_w_host: Vec<f32> = (0..(h * i))
            .map(|k| ((k * 23 + 11) % 37) as f32 * 0.02 - 0.35)
            .collect();

        let cpu = cpu_dense_mlp(
            &residual_host, &norm_w_host, &gate_w_host, &up_w_host, &down_w_host, h, i, eps,
        );

        // GPU run: upload weights, build state, run block, read back residual.
        let weights = WgpuDenseMlpWeights {
            norm_weight: upload_f32_buf(&ctx, &norm_w_host, "norm_w"),
            gate_proj: upload_f32_buf(&ctx, &gate_w_host, "gate_w"),
            up_proj: upload_f32_buf(&ctx, &up_w_host, "up_w"),
            down_proj: upload_f32_buf(&ctx, &down_w_host, "down_w"),
        };
        let mut state = WgpuLlamaState::new_for_dense_mlp(&ctx, h, i).expect("state");
        // Seed `residual` with the input activation.
        ctx.queue.write_buffer(
            state.residual.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&residual_host),
        );

        forward_dense_mlp_block_device(&ctx, &mut state, &weights, eps).expect("forward");

        let gpu = download_f32_buf(&ctx, state.residual.as_ref().unwrap(), h, "result").unwrap();
        for (k, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            assert!(
                (g - c).abs() < 1e-4,
                "mismatch at k={k}: gpu={g} cpu={c}",
            );
        }
    }
}

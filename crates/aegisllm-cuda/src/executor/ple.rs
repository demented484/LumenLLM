//! Per-Layer Embeddings (PLE) forward path for Gemma-4 E4B / E2B.
//!
//! HF Gemma-4 splits the per-token signal into a **main embedding** (the
//! usual `embed_tokens` lookup) and a **per-layer embedding** that gets
//! added back inside every decoder block. The per-layer feed is the sum of:
//!
//!   1. A **token-identity** component: row `t` of `embed_tokens_per_layer`,
//!      shape `[num_layers, ple_dim]`, scaled by `sqrt(ple_dim)`.
//!   2. A **context-aware** component: `per_layer_model_projection(hidden)`
//!      scaled by `1/sqrt(hidden_size)`, reshaped to `[num_layers, ple_dim]`,
//!      then RMSNorm'd with `per_layer_projection_norm`.
//!
//! The two are combined by `(token_identity + context) * (1/sqrt(2))` —
//! the precomputed `combine_scale` in `PleGlobal`.
//!
//! Inside each decoder layer's MLP forward, the per-layer additive
//! contribution is computed as:
//!
//! ```text
//!   gate    = per_layer_input_gate(hidden_out)          # [ple_dim]
//!   gate    = gelu_pytorch_tanh(gate)
//!   gated   = gate * per_layer_inputs[layer_idx, :]     # elementwise
//!   contrib = per_layer_projection(gated)               # [hidden]
//!   contrib = post_per_layer_input_norm(contrib)        # RMSNorm
//!   hidden_out += contrib
//! ```

use crate::cuda::CudaRuntime;
use crate::executor::state::{CudaLayer, CudaScratch, PleGlobal};
use aegisllm_base::error::{AegisError, Result};

/// Compute `per_layer_inputs` for a single decode token. Result lives in
/// `scratch.per_layer_inputs` as `[num_layers, ple_dim]` row-major f32 and
/// is consumed by each layer's [`apply_ple_contribution`] call.
///
/// Reads `embed_tokens_per_layer[token_id, :]` from host-pinned memory,
/// runs `hidden @ model_projection.T` via cuBLASLt BF16 GEMM, applies the
/// projection scale + per-row RMSNorm via existing kernels, then combines
/// the two contributions on the device.
pub(super) fn compute_per_layer_inputs_decode(
    runtime: &CudaRuntime,
    ple: &PleGlobal,
    token_id: usize,
    hidden: &crate::cuda::DeviceBuffer<f32>,
    num_layers: usize,
    scratch: &mut CudaScratch,
    rms_norm_eps: f32,
) -> Result<()> {
    let row_len = num_layers * ple.ple_dim;
    // 1. Lookup `embed_tokens_per_layer[token_id, :]` — the host arena
    //    holds the table as BF16. Copy that row into BF16 staging on
    //    device, then dequant+scale by `sqrt(ple_dim)`.
    let host_bytes = ple.embed_table.host_values_u16().ok_or_else(|| {
        AegisError::InvalidPlan(
            "PLE embed_table is not host-resident (unexpected; loader sets store=ram)".into()
        )
    })?;
    let table_cols = ple.embed_table.cols;
    if row_len != table_cols {
        return Err(AegisError::InvalidPlan(format!(
            "PLE: expected row_len={row_len} (num_layers*ple_dim) but table_cols={table_cols}"
        )));
    }
    let row_start = token_id.checked_mul(table_cols).ok_or_else(|| {
        AegisError::InvalidPlan(format!("PLE: token_id={token_id} × cols={table_cols} overflow"))
    })?;
    let row_end = row_start + table_cols;
    if row_end > host_bytes.len() {
        return Err(AegisError::InvalidPlan(format!(
            "PLE: token_id={token_id} OOB of embed_tokens_per_layer (rows={})",
            host_bytes.len() / table_cols
        )));
    }
    runtime.upload_u16_slice_to_device(&host_bytes[row_start..row_end], &mut scratch.ple_bf16_in)?;
    // Convert BF16 → F32 into `per_layer_inputs` and scale by sqrt(ple_dim)
    // in a single pass. Reuses the existing `bf16_to_f32` kernel followed by
    // an in-place scale.
    runtime.bf16_to_f32_device(&scratch.ple_bf16_in, row_len, &mut scratch.per_layer_inputs)?;
    runtime.scale_f32_device_len(
        ple.embed_scale_per_layer, &mut scratch.per_layer_inputs, row_len,
    )?;

    // 2. Context projection: `hidden @ per_layer_model_projection.T`. The
    //    matrix is `[num_layers * ple_dim, hidden]` BF16 VRAM-resident.
    //    BF16 input = the F32 `hidden` vector quantized at length `hidden`.
    let hidden_len = ple.model_projection.cols;
    runtime.f32_to_bf16_device(hidden, hidden_len, &mut scratch.ple_bf16_in)?;
    runtime.matmul_bf16_cublaslt_with_input_bf16_device(
        &ple.model_projection,
        &scratch.ple_bf16_in,
        1, // batch=1 for decode
        &mut scratch.ple_bf16_out,
        &mut scratch.ple_projection,
    )?;
    // 3. Scale projection by `1/sqrt(hidden)`.
    runtime.scale_f32_device_len(ple.model_projection_scale, &mut scratch.ple_projection, row_len)?;

    // 4. RMSNorm each `[ple_dim]` row in the `[num_layers, ple_dim]` view
    //    against `projection_norm`.
    runtime.rms_norm_batched_device(
        &scratch.ple_projection, &ple.projection_norm, num_layers, rms_norm_eps,
        &mut scratch.ple_projection_normed,
    )?;

    // 5. per_layer_inputs += projection_normed; then * combine_scale.
    runtime.add_inplace_device_len(
        &mut scratch.per_layer_inputs, &scratch.ple_projection_normed, row_len,
    )?;
    runtime.scale_f32_device_len(ple.combine_scale, &mut scratch.per_layer_inputs, row_len)?;
    Ok(())
}

/// Apply the per-layer PLE additive contribution to `scratch.hidden_out`
/// **before** `layer_scalar`. Reads the `[ple_dim]` slice of
/// `per_layer_inputs` belonging to this layer.
pub(super) fn apply_ple_contribution_decode(
    runtime: &CudaRuntime,
    layer: &CudaLayer,
    ple: &PleGlobal,
    layer_idx: usize,
    scratch: &mut CudaScratch,
    rms_norm_eps: f32,
) -> Result<()> {
    let layer_ple = match &layer.ple {
        Some(p) => p,
        None => return Ok(()),
    };
    let ple_dim = ple.ple_dim;
    // 1. gate = hidden_out @ input_gate.T   → [ple_dim]
    runtime.f32_to_bf16_device(&scratch.hidden_out, layer_ple.input_gate.cols, &mut scratch.ple_bf16_in)?;
    runtime.matmul_bf16_cublaslt_with_input_bf16_device(
        &layer_ple.input_gate,
        &scratch.ple_bf16_in,
        1,
        &mut scratch.ple_bf16_out,
        &mut scratch.ple_gate,
    )?;
    // 2. gate = gelu_pytorch_tanh(gate) in-place. We have geglu_tanh_in_place,
    //    but that's a gate*up fused activation; here it's a single tensor.
    //    Use the standalone `gelu_tanh_in_place_device` if available, else
    //    a tiny helper. For now, route through `gelu_tanh_strided_device`
    //    with up=ones is wasteful — use `gelu_tanh_in_place_device`.
    runtime.gelu_tanh_inplace_device(&mut scratch.ple_gate, ple_dim)?;
    // 3. gate *= per_layer_inputs[layer_idx, :ple_dim]
    runtime.mul_vec_inplace_slice_device(
        &mut scratch.ple_gate,
        &scratch.per_layer_inputs,
        layer_idx * ple_dim,
        ple_dim,
    )?;
    // 4. contrib = gate @ projection.T → [hidden]
    runtime.f32_to_bf16_device(&scratch.ple_gate, ple_dim, &mut scratch.ple_bf16_in)?;
    runtime.matmul_bf16_cublaslt_with_input_bf16_device(
        &layer_ple.projection,
        &scratch.ple_bf16_in,
        1,
        &mut scratch.ple_bf16_out,
        &mut scratch.ple_contrib,
    )?;
    // 5. contrib = RMSNorm(contrib, post_norm) into ple_contrib_normed.
    runtime.rms_norm_device(
        &scratch.ple_contrib, &layer_ple.post_norm, rms_norm_eps, &mut scratch.ple_contrib_normed,
    )?;
    // 6. hidden_out += contrib_normed
    runtime.add_inplace_device_len(
        &mut scratch.hidden_out, &scratch.ple_contrib_normed, layer_ple.projection.rows,
    )?;
    Ok(())
}

//! Per-Layer Embeddings (PLE) for Gemma-4 E4B / E2B. Ports the CUDA
//! `compute_per_layer_inputs_decode` and `apply_ple_contribution_decode`
//! (`crates/aegisllm-cuda/src/executor/ple.rs`).
//!
//! Token-entry (once per token):
//!   token_identity = embed_tokens_per_layer[t, :]  * sqrt(ple_dim)   [num_layers, ple_dim]
//!   context        = (hidden @ model_projection.T) * 1/sqrt(hidden)  [num_layers, ple_dim]
//!   context        = rms_norm(context, projection_norm) per [ple_dim] row
//!   per_layer_inputs = (token_identity + context) * (1/sqrt(2))
//!
//! Per-layer (inside each block's MLP, BEFORE layer_scalar):
//!   gate    = input_gate @ hidden_out            [ple_dim]
//!   gate    = gelu_pytorch_tanh(gate)
//!   gate   *= per_layer_inputs[layer_idx, :]
//!   contrib = projection @ gate                  [hidden]
//!   contrib = rms_norm(contrib, post_norm)
//!   hidden_out += contrib

use super::state::{G4PleGlobal, G4PleLayer};
use crate::cpu::math::rms_norm_into;
use crate::cpu::simd;
use aegisllm_base::error::Result;
use rayon::prelude::*;

/// Compute `per_layer_inputs` for one decode token into `out_per_layer_inputs`
/// (`[num_layers * ple_dim]`).
pub(crate) fn compute_per_layer_inputs(
    ple: &G4PleGlobal,
    token_id: usize,
    hidden: &[f32],
    num_layers: usize,
    eps: f32,
    out_per_layer_inputs: &mut [f32],
) -> Result<()> {
    let row_len = num_layers * ple.ple_dim;
    debug_assert_eq!(out_per_layer_inputs.len(), row_len);

    // 1. token identity: embed_tokens_per_layer[token_id, :] * sqrt(ple_dim).
    let mut token_identity = ple.embed_table.row(token_id)?;
    simd::scale_in_place(&mut token_identity, ple.embed_scale_per_layer);

    // 2. context = hidden @ model_projection.T  → [num_layers * ple_dim].
    let mut context = vec![0.0_f32; row_len];
    ple.model_projection.matvec_into(hidden, &mut context)?;
    simd::scale_in_place(&mut context, ple.model_projection_scale);

    // 3. RMS-norm each [ple_dim] row by projection_norm.
    let mut context_normed = vec![0.0_f32; row_len];
    for layer in 0..num_layers {
        let base = layer * ple.ple_dim;
        rms_norm_into(
            &context[base..base + ple.ple_dim],
            &ple.projection_norm,
            eps,
            &mut context_normed[base..base + ple.ple_dim],
        );
    }

    // 4. per_layer_inputs = (token_identity + context_normed) * combine_scale.
    for i in 0..row_len {
        out_per_layer_inputs[i] =
            (token_identity[i] + context_normed[i]) * ple.combine_scale;
    }
    Ok(())
}

/// Apply the per-layer PLE additive contribution to `hidden_out` (in place),
/// BEFORE layer_scalar. `per_layer_inputs` is the global `[num_layers, ple_dim]`
/// feed; this reads the `layer_idx` slice.
pub(crate) fn apply_ple_contribution(
    layer_ple: &G4PleLayer,
    ple: &G4PleGlobal,
    layer_idx: usize,
    per_layer_inputs: &[f32],
    eps: f32,
    hidden_out: &mut [f32],
) -> Result<()> {
    let ple_dim = ple.ple_dim;

    // 1. gate = input_gate @ hidden_out → [ple_dim].
    let mut gate = vec![0.0_f32; layer_ple.input_gate.rows];
    layer_ple.input_gate.matvec_into(hidden_out, &mut gate)?;

    // 2. gate = gelu_pytorch_tanh(gate) in place.
    for g in gate.iter_mut() {
        *g = simd::gelu_tanh_scalar(*g);
    }

    // 3. gate *= per_layer_inputs[layer_idx, :ple_dim].
    let base = layer_idx * ple_dim;
    let pli = &per_layer_inputs[base..base + ple_dim];
    for (g, &p) in gate.iter_mut().zip(pli.iter()) {
        *g *= p;
    }

    // 4. contrib = projection @ gate → [hidden].
    let mut contrib = vec![0.0_f32; layer_ple.projection.rows];
    layer_ple.projection.matvec_into(&gate, &mut contrib)?;

    // 5. contrib = rms_norm(contrib, post_norm).
    let mut contrib_normed = vec![0.0_f32; contrib.len()];
    rms_norm_into(&contrib, &layer_ple.post_norm, eps, &mut contrib_normed);

    // 6. hidden_out += contrib_normed.
    simd::add_in_place(hidden_out, &contrib_normed);
    Ok(())
}

/// Batched PLE token-entry — the prefill equivalent of `compute_per_layer_inputs`
/// over `batch` tokens. Identical math, but the expensive `model_projection`
/// (`[num_layers*ple_dim, hidden]`) runs as ONE batched VNNI GEMM over all tokens
/// instead of a per-token matvec, and the per-row RMS-norm + combine parallelize.
/// `hidden` is `[batch, hidden_size]`; `out` is `[batch, num_layers*ple_dim]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_per_layer_inputs_batched(
    ple: &G4PleGlobal,
    token_ids: &[usize],
    hidden: &[f32],
    batch: usize,
    hidden_size: usize,
    num_layers: usize,
    eps: f32,
    out: &mut [f32],
) -> Result<()> {
    let ple_dim = ple.ple_dim;
    let row_len = num_layers * ple_dim;

    // 2. context[batch, row_len] = hidden[batch, hidden] @ model_projection.T — BATCHED.
    let mut context = vec![0.0_f32; batch * row_len];
    super::linear::bf16_matmul_fast(
        ple.model_projection.weight_bytes(),
        row_len,
        hidden_size,
        hidden,
        batch,
        &mut context,
    );

    // 1,3,4 per token (parallel): token identity + scale, scale context, RMS-norm
    // each [ple_dim] row into `out`, then combine. Each `out` row fully written.
    out.par_chunks_mut(row_len)
        .zip(context.par_chunks_mut(row_len))
        .enumerate()
        .try_for_each(|(t, (out_row, ctx_row))| -> Result<()> {
            // 1. token identity: embed_tokens_per_layer[token, :] * sqrt(ple_dim).
            let mut token_identity = ple.embed_table.row(token_ids[t])?;
            simd::scale_in_place(&mut token_identity, ple.embed_scale_per_layer);
            // 2. scale context by 1/sqrt(hidden).
            simd::scale_in_place(ctx_row, ple.model_projection_scale);
            // 3. RMS-norm each [ple_dim] row → out_row (context_normed).
            for layer in 0..num_layers {
                let base = layer * ple_dim;
                rms_norm_into(
                    &ctx_row[base..base + ple_dim],
                    &ple.projection_norm,
                    eps,
                    &mut out_row[base..base + ple_dim],
                );
            }
            // 4. out = (token_identity + context_normed) * combine_scale.
            for i in 0..row_len {
                out_row[i] = (token_identity[i] + out_row[i]) * ple.combine_scale;
            }
            Ok(())
        })
}

/// Batched PLE contribution over `batch` tokens — the prefill equivalent of
/// `apply_ple_contribution`. Same math, but the two projections run as batched
/// `matmul_into` GEMMs (the fast BF16-VNNI path) instead of per-token `matvec`,
/// the gelu·per_layer_input fuses through the vectorized `geglu_into_simd`, and
/// the post-norm + residual add parallelize over tokens. `hidden_out` is
/// `[batch, hidden]`; `per_layer_inputs` is `[batch, num_layers*ple_dim]`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_ple_contribution_batched(
    layer_ple: &G4PleLayer,
    ple: &G4PleGlobal,
    layer_idx: usize,
    per_layer_inputs: &[f32],
    ple_row: usize,
    batch: usize,
    eps: f32,
    hidden_out: &mut [f32],
) -> Result<()> {
    let ple_dim = ple.ple_dim;
    let gate_dim = layer_ple.input_gate.rows;
    let hidden = layer_ple.projection.rows;
    let base = layer_idx * ple_dim;

    // 1. gate[batch, gate_dim] = input_gate @ hidden_out[batch, hidden]. PLE
    // weights are raw `Bf16Matrix`, so drive the batched VNNI kernel directly.
    let mut gate = vec![0.0_f32; batch * gate_dim];
    super::linear::bf16_matmul_fast(
        layer_ple.input_gate.weight_bytes(),
        gate_dim,
        layer_ple.input_gate.cols,
        hidden_out,
        batch,
        &mut gate,
    );

    // 2-3. gated = gelu_pytorch_tanh(gate) * per_layer_inputs[layer_idx] — the
    // vectorized GeGLU fuses both steps; parallel over tokens. (gate_dim==ple_dim.)
    let mut gated = vec![0.0_f32; batch * gate_dim];
    gated
        .par_chunks_mut(gate_dim)
        .enumerate()
        .for_each(|(t, out)| {
            let g = &gate[t * gate_dim..t * gate_dim + gate_dim];
            let pli = &per_layer_inputs[t * ple_row + base..t * ple_row + base + ple_dim];
            simd::geglu_into_simd(g, pli, out);
        });

    // 4. contrib[batch, hidden] = projection @ gated[batch, gate_dim].
    let mut contrib = vec![0.0_f32; batch * hidden];
    super::linear::bf16_matmul_fast(
        layer_ple.projection.weight_bytes(),
        hidden,
        layer_ple.projection.cols,
        &gated,
        batch,
        &mut contrib,
    );

    // 5-6. contrib = rms_norm(contrib, post_norm); hidden_out += contrib (per token, parallel).
    hidden_out
        .par_chunks_mut(hidden)
        .zip(contrib.par_chunks(hidden))
        .for_each(|(out, c)| {
            let mut cn = vec![0.0_f32; hidden];
            rms_norm_into(c, &layer_ple.post_norm, eps, &mut cn);
            simd::add_in_place(out, &cn);
        });
    Ok(())
}

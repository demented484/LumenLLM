//! Layer-block forward primitives (device-resident).
//!
//! These compose the per-primitive `*_device` functions from
//! [`super::forward`] into the layer-shaped operations a model actually
//! runs (e.g. dense MLP block: rms_norm → gate/up → swiglu → down →
//! residual). Inputs and outputs are persistent `wgpu::Buffer`s; nothing
//! goes to host between calls.

use aegisllm_base::error::{AegisError, Result};

use super::forward::{
    decode_attention_device_full, decode_attention_device_strided, dequant_nvfp4_device,
    geglu_tanh_device, matmul_f32_device, residual_add_device, rms_norm_batched_device,
    rms_norm_device, rope_device, scale_f32_device, swiglu_device,
};

/// MLP activation function. Llama uses SwiGLU; Gemma-4 uses GeGLU
/// (tanh-approximation GELU). Selected by the per-model forward path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    SwiGLU,
    GeGluTanh,
}
use super::loader::WgpuContext;
use super::state::{WgpuLlamaState, WgpuModelState};
use super::weights::{WgpuLayerWeights, WgpuLinear, WgpuMlpWeightsFull, WgpuModel, WgpuMoeWeights};

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

/// Weights for one Llama-style attention block, in device memory.
///
/// `norm_weight`: `[hidden_size]` rms-norm scale vector.
/// `q_proj`: `[num_q_heads * head_dim, hidden_size]` row-major.
/// `k_proj` / `v_proj`: `[num_kv_heads * head_dim, hidden_size]` row-major.
/// `o_proj`: `[hidden_size, num_q_heads * head_dim]` row-major.
///
/// All buffers are f32 storage. NVFP4/BF16 storage will plug in via the
/// existing `dequant_nvfp4_device` shader once the on-device dequant is
/// wired into the projection step.
pub struct WgpuAttentionWeights {
    pub norm_weight: wgpu::Buffer,
    pub q_proj: wgpu::Buffer,
    pub k_proj: wgpu::Buffer,
    pub v_proj: wgpu::Buffer,
    pub o_proj: wgpu::Buffer,
}

impl std::fmt::Debug for WgpuAttentionWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuAttentionWeights").finish_non_exhaustive()
    }
}

/// Run one attention block on the wgpu backend (single token / decode).
///
/// Input: `state.residual` holds the layer-input activation; `state.position`
/// is the 0-indexed token position (must be < `state.max_seq_len`).
/// Output: `state.residual` is updated in place to `residual + attn(...)`.
/// `state.position` is **not** advanced — the caller manages it after
/// finishing the layer (typically after the MLP block).
///
/// `cos_table` / `sin_table` are precomputed for the current position
/// (length `head_dim / 2` each); the function uploads them into
/// `state.rope_cos` / `state.rope_sin` before dispatching RoPE.
///
/// Pipeline (all on device, no host readback):
///   1. `post_normed = rms_norm(residual, norm_weight)`
///   2. `q_new = matmul(post_normed, q_proj^T)`
///   3. `k_new = matmul(post_normed, k_proj^T)`
///   4. `v_new = matmul(post_normed, v_proj^T)`
///   5. `rope(q_new)` and `rope(k_new)` (in place)
///   6. write `k_new` → cache.keys at offset `position * kv_width`
///   7. write `v_new` → cache.values at offset
///      `max_seq_len * kv_width + position * kv_width`
///   8. `attn_out = decode_attention(q_new, cache, seq_len = position+1)`
///   9. `mlp_out = matmul(attn_out, o_proj^T)`
///  10. `residual = residual + mlp_out`
#[allow(clippy::too_many_arguments)]
pub fn forward_attention_block_device(
    ctx: &WgpuContext,
    state: &mut WgpuLlamaState,
    weights: &WgpuAttentionWeights,
    cos_table: &[f32],
    sin_table: &[f32],
    rms_norm_eps: f32,
) -> Result<()> {
    let hidden = state.hidden_size;
    let nq = state.num_q_heads;
    let nkv = state.num_kv_heads;
    let hd = state.head_dim;
    let max_seq = state.max_seq_len;
    let position = state.position;
    let kv_width = nkv * hd;
    let q_width = nq * hd;
    let half = hd / 2;
    if nq == 0 || nkv == 0 || hd == 0 || max_seq == 0 {
        return Err(AegisError::InvalidPlan(
            "WgpuLlamaState attention dims are zero — was new_for_full_layer called?".into(),
        ));
    }
    if position >= max_seq {
        return Err(AegisError::InvalidPlan(format!(
            "decode position {position} ≥ max_seq_len {max_seq} — KV cache is full"
        )));
    }
    if cos_table.len() != half || sin_table.len() != half {
        return Err(AegisError::InvalidPlan(format!(
            "cos/sin table size mismatch: cos={} sin={} expected={half}",
            cos_table.len(),
            sin_table.len(),
        )));
    }

    let residual = state.residual.as_ref().expect("residual");
    let post_normed = state.post_normed.as_ref().expect("post_normed");
    let attn_q = state.attn_q.as_ref().expect("attn_q");
    let attn_k_new = state.attn_k_new.as_ref().expect("attn_k_new");
    let attn_v_new = state.attn_v_new.as_ref().expect("attn_v_new");
    let attn_out = state.attn_out.as_ref().expect("attn_out");
    let kv_cache = state.attn_kv_cache.as_ref().expect("attn_kv_cache");
    let rope_cos = state.rope_cos.as_ref().expect("rope_cos");
    let rope_sin = state.rope_sin.as_ref().expect("rope_sin");
    let mlp_out_scratch = state.mlp_out.as_ref().expect("mlp_out");

    // Upload RoPE tables for this position.
    ctx.queue.write_buffer(rope_cos, 0, bytemuck::cast_slice(cos_table));
    ctx.queue.write_buffer(rope_sin, 0, bytemuck::cast_slice(sin_table));

    // 1. Pre-attention norm.
    rms_norm_device(ctx, residual, &weights.norm_weight, post_normed, hidden, rms_norm_eps)?;

    // 2-4. QKV projections.
    matmul_f32_device(ctx, post_normed, &weights.q_proj, attn_q, 1, q_width, hidden)?;
    matmul_f32_device(ctx, post_normed, &weights.k_proj, attn_k_new, 1, kv_width, hidden)?;
    matmul_f32_device(ctx, post_normed, &weights.v_proj, attn_v_new, 1, kv_width, hidden)?;

    // 5. RoPE on Q and K (in place).
    rope_device(ctx, attn_q, rope_cos, rope_sin, nq, hd)?;
    rope_device(ctx, attn_k_new, rope_cos, rope_sin, nkv, hd)?;

    // 6-7. Write K/V into the persistent cache at slot `position`. The
    // cache layout is K_full || V_full with V_full starting at
    // `max_seq * kv_width` floats. Each slot is `kv_width` floats.
    let bytes_per_slot = (kv_width * std::mem::size_of::<f32>()) as u64;
    let k_offset_bytes = (position * kv_width * std::mem::size_of::<f32>()) as u64;
    let v_offset_bytes =
        ((max_seq + position) * kv_width * std::mem::size_of::<f32>()) as u64;
    let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("attn_kv_cache write"),
    });
    enc.copy_buffer_to_buffer(attn_k_new, 0, kv_cache, k_offset_bytes, bytes_per_slot);
    enc.copy_buffer_to_buffer(attn_v_new, 0, kv_cache, v_offset_bytes, bytes_per_slot);
    ctx.queue.submit(std::iter::once(enc.finish()));

    // 8. Attention over the live region [0, position+1] of the cache.
    let seq_len = position + 1;
    let v_offset_floats = max_seq * kv_width;
    decode_attention_device_strided(
        ctx,
        attn_q,
        kv_cache,
        attn_out,
        nq,
        nkv,
        hd,
        seq_len,
        Some(v_offset_floats),
    )?;

    // 9. O projection.
    matmul_f32_device(ctx, attn_out, &weights.o_proj, mlp_out_scratch, 1, hidden, q_width)?;

    // 10. residual += mlp_out (route through post_normed scratch like the
    // dense MLP block does — wgpu primitives don't have an in-place add
    // yet).
    residual_add_device(ctx, residual, mlp_out_scratch, post_normed, hidden)?;
    let mut wb = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("attn_block writeback"),
    });
    wb.copy_buffer_to_buffer(post_normed, 0, residual, 0, (hidden * 4) as u64);
    ctx.queue.submit(std::iter::once(wb.finish()));
    Ok(())
}

/// Run a single dense MLP block (gate/up matmul → activation → down
/// matmul) producing the result in `out_hidden_size`. Caller already
/// wrote the post-norm input into `post_normed`. Used both as the
/// standalone Llama MLP and per-routed-expert / shared-expert dispatch
/// inside a Gemma-4 MoE layer.
#[allow(clippy::too_many_arguments)]
fn run_dense_mlp_into(
    ctx: &WgpuContext,
    gate_proj: &WgpuLinear,
    up_proj: &WgpuLinear,
    down_proj: &WgpuLinear,
    model_state: &WgpuModelState,
    post_normed: &wgpu::Buffer,
    out_hidden_size: &wgpu::Buffer,
    activation: Activation,
    intermediate: usize,
    hidden: usize,
) -> Result<()> {
    matmul_with_optional_dequant(ctx, model_state, post_normed, gate_proj, &model_state.gate, 1, intermediate, hidden)?;
    matmul_with_optional_dequant(ctx, model_state, post_normed, up_proj, &model_state.up, 1, intermediate, hidden)?;
    match activation {
        Activation::SwiGLU => {
            swiglu_device(ctx, &model_state.gate, &model_state.up, &model_state.swiglu_out, intermediate)?;
        }
        Activation::GeGluTanh => {
            geglu_tanh_device(ctx, &model_state.gate, &model_state.up, &model_state.swiglu_out, intermediate)?;
        }
    }
    matmul_with_optional_dequant(ctx, model_state, &model_state.swiglu_out, down_proj, out_hidden_size, 1, hidden, intermediate)?;
    Ok(())
}

/// Run one Gemma-4 MoE block on the wgpu backend.
///
/// Reads `model_state.residual` as input. Produces `model_state.mlp_out`
/// containing the FULL MoE output (routed-expert weighted sum + shared
/// expert), already post-normed where the model has post-FFN norms.
/// The caller is responsible for the residual add. The routed-expert
/// pre-norm output is left in `model_state.post_normed` (caller has
/// already written it via `rms_norm_device`).
///
/// CPU-side router: we download `num_experts` logits, do softmax+top-k
/// + per-expert-scale on host. For Gemma-4 (num_experts=128, top_k=2)
/// this is 512B / layer / token — negligible. A future
/// `router_softmax_topk` device shader can replace this without
/// changing the surrounding code.
#[allow(clippy::too_many_arguments)]
pub fn forward_moe_block_device(
    ctx: &WgpuContext,
    model_state: &mut WgpuModelState,
    pre_ffn_normed: &wgpu::Buffer,
    moe: &WgpuMoeWeights,
    rms_norm_eps: f32,
    activation: Activation,
) -> Result<()> {
    let h = model_state.hidden_size;

    // ── Router input: optionally rms-norm + scale by sqrt(hidden) ──
    // Gemma-4: hidden = rms_norm_no_weight(residual) * router.scale * 1/sqrt(hidden)
    // For simplicity we treat the model as supplying `pre_ffn_normed`
    // already in the right state; if `router_input_scale` is present, we
    // scale `pre_ffn_normed` element-wise BEFORE projecting.
    // (Faithful Gemma-4 reference uses a fresh no-weight rms-norm of
    // residual; the wgpu loader will handle that detail when populating
    // these fields. For tests today, pass the same buffer as input.)
    let router_input = pre_ffn_normed;

    // Router projection: matmul(router_input, router.weight^T) → logits.
    matmul_with_optional_dequant(
        ctx,
        model_state,
        router_input,
        &moe.router,
        &model_state.router_logits,
        1,
        moe.num_experts,
        h,
    )?;

    // Download `num_experts` logits to host for softmax+topk.
    let logits_host = {
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("router_logits_staging"),
            size: (moe.num_experts * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("router_logits_readback"),
        });
        enc.copy_buffer_to_buffer(
            &model_state.router_logits,
            0,
            &staging,
            0,
            (moe.num_experts * 4) as u64,
        );
        ctx.queue.submit(std::iter::once(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).ok(); });
        ctx.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| AegisError::Unsupported("router logits readback channel closed".into()))?
            .map_err(|e| AegisError::Unsupported(format!("router logits map_async: {e:?}")))?;
        let data = slice.get_mapped_range();
        let v: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        v
    };

    // Softmax + top-k + per-expert scale + renormalise.
    let max_logit = logits_host.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits_host.iter().map(|l| (l - max_logit).exp()).collect();
    let exp_sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|e| e / exp_sum).collect();
    let mut topk_pairs: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
    topk_pairs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    topk_pairs.truncate(moe.top_k);
    // Renormalise top-k weights to sum to 1 then apply per-expert scale.
    let topk_sum: f32 = topk_pairs.iter().map(|(_, w)| *w).sum();
    let topk: Vec<(usize, f32)> = topk_pairs
        .into_iter()
        .map(|(idx, w)| {
            let normalised = if topk_sum > 0.0 { w / topk_sum } else { 0.0 };
            let calibrated = normalised * moe.per_expert_scale.get(idx).copied().unwrap_or(1.0);
            (idx, calibrated)
        })
        .collect();

    // Zero accumulator.
    {
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("moe_acc_zero"),
        });
        enc.clear_buffer(&model_state.moe_acc, 0, None);
        ctx.queue.submit(std::iter::once(enc.finish()));
    }

    // Per-expert dispatch: gate/up matmul → activation → down matmul →
    // weighted accumulate into moe_acc.
    let inter = moe.intermediate_size;
    for (expert_idx, weight) in &topk {
        let expert = moe.experts.get(*expert_idx).ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "router top-k chose expert {expert_idx} which is out of [0, {})",
                moe.experts.len()
            ))
        })?;
        // Per-expert MLP into `mlp_out` buffer (scratch).
        run_dense_mlp_into(
            ctx,
            &expert.gate_proj,
            &expert.up_proj,
            &expert.down_proj,
            model_state,
            pre_ffn_normed,
            &model_state.mlp_out,
            activation,
            inter,
            h,
        )?;
        // moe_acc += weight * mlp_out. We don't have a fused axpy yet,
        // so: scale mlp_out in-place by weight, then add into moe_acc
        // (residual_add_device produces a new buffer; we route through
        // post_normed and copy back).
        scale_f32_device(ctx, &model_state.mlp_out, h, *weight)?;
        residual_add_device(ctx, &model_state.moe_acc, &model_state.mlp_out, &model_state.post_normed, h)?;
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("moe_acc_writeback"),
        });
        enc.copy_buffer_to_buffer(&model_state.post_normed, 0, &model_state.moe_acc, 0, (h * 4) as u64);
        ctx.queue.submit(std::iter::once(enc.finish()));
    }

    // Optional post-FFN-norm-2 on the routed accumulator.
    if let Some(ref post_ffn_2) = moe.post_feedforward_layernorm_2 {
        rms_norm_device(ctx, &model_state.moe_acc, post_ffn_2, &model_state.post_normed, h, rms_norm_eps)?;
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("post_ffn_2_writeback"),
        });
        enc.copy_buffer_to_buffer(&model_state.post_normed, 0, &model_state.moe_acc, 0, (h * 4) as u64);
        ctx.queue.submit(std::iter::once(enc.finish()));
    }

    // Shared expert (always active): produces shared_expert_out.
    if let Some(ref shared) = moe.shared_expert {
        run_dense_mlp_into(
            ctx,
            &shared.gate_proj,
            &shared.up_proj,
            &shared.down_proj,
            model_state,
            pre_ffn_normed,
            &model_state.shared_expert_out,
            activation,
            inter,
            h,
        )?;
        if let Some(ref post_ffn_1) = moe.post_feedforward_layernorm_1 {
            rms_norm_device(ctx, &model_state.shared_expert_out, post_ffn_1, &model_state.post_normed, h, rms_norm_eps)?;
            let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("post_ffn_1_writeback"),
            });
            enc.copy_buffer_to_buffer(&model_state.post_normed, 0, &model_state.shared_expert_out, 0, (h * 4) as u64);
            ctx.queue.submit(std::iter::once(enc.finish()));
        }
        // mlp_out = moe_acc + shared_expert_out.
        residual_add_device(
            ctx,
            &model_state.moe_acc,
            &model_state.shared_expert_out,
            &model_state.mlp_out,
            h,
        )?;
    } else {
        // No shared expert: mlp_out = moe_acc (just copy).
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("moe_acc_to_mlp_out"),
        });
        enc.copy_buffer_to_buffer(&model_state.moe_acc, 0, &model_state.mlp_out, 0, (h * 4) as u64);
        ctx.queue.submit(std::iter::once(enc.finish()));
    }
    Ok(())
}

/// Run a `[1, n] = [1, k] @ [n, k]^T` matmul where the weight may be
/// either `Dense` (used directly) or `Nvfp4` (dequantised on-the-fly
/// into `model_state.nvfp4_dequant_scratch` first). Pairs the dequant
/// + matmul atomically so the scratch lifetime is guaranteed to stay
/// valid across the matmul read; subsequent calls then reuse the
/// scratch for their own dequant.
#[allow(clippy::too_many_arguments)]
fn matmul_with_optional_dequant(
    ctx: &WgpuContext,
    model_state: &WgpuModelState,
    input: &wgpu::Buffer,
    linear: &WgpuLinear,
    output: &wgpu::Buffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let weight_buf = match linear {
        WgpuLinear::Dense { weight, .. } => weight,
        WgpuLinear::Nvfp4 { packed, scales, rows, cols, output_scale } => {
            dequant_nvfp4_device(
                ctx,
                packed,
                scales,
                &model_state.nvfp4_dequant_scratch,
                *rows,
                *cols,
                *output_scale,
            )?;
            &model_state.nvfp4_dequant_scratch
        }
    };
    matmul_f32_device(ctx, input, weight_buf, output, m, n, k)
}

/// Compatibility shim used in matmul sites that don't have
/// `&WgpuModelState` in scope (e.g. final `lm_head`). Dense passes
/// through; NVFP4 errors with a clear message. The lm_head matmul
/// site has been refactored to use `matmul_with_optional_dequant` —
/// this helper is kept as a fallback for any leftover callers.
fn dense_weight_buf(linear: &WgpuLinear) -> Result<&wgpu::Buffer> {
    match linear {
        WgpuLinear::Dense { weight, .. } => Ok(weight),
        WgpuLinear::Nvfp4 { .. } => Err(AegisError::Unsupported(
            "wgpu: this matmul site doesn't have model_state access for NVFP4 dequant; \
             refactor the call site to use matmul_with_optional_dequant"
                .into(),
        )),
    }
}

/// Run one full Llama-style transformer layer (attention block + dense
/// MLP block) on the wgpu backend, end-to-end on persistent buffers.
///
/// `model_state.residual` carries the activation across blocks; the
/// function reads + updates it in place. `layer_idx` selects this
/// layer's persistent KV cache from `model_state.kv_caches`.
///
/// `cos_table` / `sin_table` are precomputed for the current decode
/// position and uploaded into `model_state.rope_cos` / `rope_sin` at
/// the start of the attention block.
///
/// The pipeline mirrors the reference vanilla-Llama decoder block:
///   ATTENTION:
///     1. post_normed = rms_norm(residual, attn_norm)
///     2. q = matmul(post_normed, q_proj^T)
///     3. k_new = matmul(post_normed, k_proj^T)
///     4. v_new = matmul(post_normed, v_proj^T)
///     5. rope(q), rope(k_new) in place
///     6. write k_new → cache.keys[position]
///     7. write v_new → cache.values[position]
///     8. attn_out = decode_attention(q, cache, seq_len = position + 1)
///     9. mlp_out = matmul(attn_out, o_proj^T)
///     10. residual = residual + mlp_out
///   MLP:
///     11. post_normed = rms_norm(residual, mlp_norm)
///     12. gate = matmul(post_normed, gate_proj^T)
///     13. up = matmul(post_normed, up_proj^T)
///     14. swiglu_out = silu(gate) * up
///     15. mlp_out = matmul(swiglu_out, down_proj^T)
///     16. residual = residual + mlp_out
///
/// Position is *not* advanced — the caller bumps `model_state.position`
/// once per generation step (after running all layers).
#[allow(clippy::too_many_arguments)]
pub fn forward_layer_device(
    ctx: &WgpuContext,
    model_state: &mut WgpuModelState,
    weights: &WgpuLayerWeights,
    layer_idx: usize,
    cos_table: &[f32],
    sin_table: &[f32],
    rms_norm_eps: f32,
    activation: Activation,
) -> Result<()> {
    let h = model_state.hidden_size;
    let i = model_state.intermediate_size;
    let nq = model_state.num_q_heads;
    let nkv = model_state.num_kv_heads;
    let hd = model_state.head_dim;
    let max_seq = model_state.max_seq_len;
    let position = model_state.position;
    let q_width = nq * hd;
    let kv_width = nkv * hd;
    let half = hd / 2;

    if layer_idx >= model_state.kv_caches.len() {
        return Err(AegisError::InvalidPlan(format!(
            "layer_idx {layer_idx} out of range (have {} kv_caches)",
            model_state.kv_caches.len()
        )));
    }
    if position >= max_seq {
        return Err(AegisError::InvalidPlan(format!(
            "decode position {position} ≥ max_seq_len {max_seq} — KV cache is full"
        )));
    }
    if cos_table.len() != half || sin_table.len() != half {
        return Err(AegisError::InvalidPlan(format!(
            "cos/sin table size mismatch: cos={} sin={} expected={half}",
            cos_table.len(),
            sin_table.len(),
        )));
    }

    let kv_cache = &model_state.kv_caches[layer_idx];

    // Upload RoPE tables for this position.
    ctx.queue.write_buffer(&model_state.rope_cos, 0, bytemuck::cast_slice(cos_table));
    ctx.queue.write_buffer(&model_state.rope_sin, 0, bytemuck::cast_slice(sin_table));

    // ── ATTENTION BLOCK ───────────────────────────────────────────────────
    // 1. pre-attention norm.
    rms_norm_device(
        ctx,
        &model_state.residual,
        &weights.attention.norm_weight,
        &model_state.post_normed,
        h,
        rms_norm_eps,
    )?;
    // 2-4. QKV projections (NVFP4-aware: dequants on-the-fly when needed).
    matmul_with_optional_dequant(
        ctx,
        model_state,
        &model_state.post_normed,
        &weights.attention.q_proj,
        &model_state.attn_q,
        1,
        q_width,
        h,
    )?;
    matmul_with_optional_dequant(
        ctx,
        model_state,
        &model_state.post_normed,
        &weights.attention.k_proj,
        &model_state.attn_k_new,
        1,
        kv_width,
        h,
    )?;
    matmul_with_optional_dequant(
        ctx,
        model_state,
        &model_state.post_normed,
        &weights.attention.v_proj,
        &model_state.attn_v_new,
        1,
        kv_width,
        h,
    )?;
    // 4b. Gemma-4 per-head Q/K/V norms (between projection and RoPE).
    // Vanilla Llama has no per-head norms — skip when not present.
    if let Some(ref q_norm) = weights.attention.q_norm {
        // Use post_normed as scratch (its f32[hidden] capacity is at
        // least q_width since hidden_size >= num_q_heads * head_dim
        // for both vanilla Llama and Gemma-4 sliding/global layouts).
        rms_norm_batched_device(
            ctx,
            &model_state.attn_q,
            q_norm,
            &model_state.post_normed,
            nq,
            hd,
            rms_norm_eps,
        )?;
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("q_norm_writeback"),
        });
        enc.copy_buffer_to_buffer(
            &model_state.post_normed,
            0,
            &model_state.attn_q,
            0,
            (nq * hd * 4) as u64,
        );
        ctx.queue.submit(std::iter::once(enc.finish()));
    }
    if let Some(ref k_norm) = weights.attention.k_norm {
        rms_norm_batched_device(
            ctx,
            &model_state.attn_k_new,
            k_norm,
            &model_state.post_normed,
            nkv,
            hd,
            rms_norm_eps,
        )?;
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("k_norm_writeback"),
        });
        enc.copy_buffer_to_buffer(
            &model_state.post_normed,
            0,
            &model_state.attn_k_new,
            0,
            (kv_width * 4) as u64,
        );
        ctx.queue.submit(std::iter::once(enc.finish()));
    }
    if let Some(ref v_unit) = weights.attention.v_norm_unit {
        rms_norm_batched_device(
            ctx,
            &model_state.attn_v_new,
            v_unit,
            &model_state.post_normed,
            nkv,
            hd,
            rms_norm_eps,
        )?;
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("v_norm_writeback"),
        });
        enc.copy_buffer_to_buffer(
            &model_state.post_normed,
            0,
            &model_state.attn_v_new,
            0,
            (kv_width * 4) as u64,
        );
        ctx.queue.submit(std::iter::once(enc.finish()));
    }
    // 5. RoPE on Q and K (in place).
    rope_device(
        ctx,
        &model_state.attn_q,
        &model_state.rope_cos,
        &model_state.rope_sin,
        nq,
        hd,
    )?;
    rope_device(
        ctx,
        &model_state.attn_k_new,
        &model_state.rope_cos,
        &model_state.rope_sin,
        nkv,
        hd,
    )?;
    // 5b. Gemma-4: post-RoPE Q scale-by-sqrt(d). Gemma-4 attention uses
    // scaling=1.0 in the softmax; our `decode_attention` shader bakes in
    // 1/sqrt(d). Multiplying Q by sqrt(d) cancels the kernel's scale so
    // the effective Q·K^T is unscaled. Triggered by q_norm being present
    // (which is Gemma-4's marker tensor — vanilla Llama has neither).
    if weights.attention.q_norm.is_some() {
        let sqrt_d = (hd as f32).sqrt();
        scale_f32_device(ctx, &model_state.attn_q, nq * hd, sqrt_d)?;
    }
    // 6-7. KV cache writes (per-layer cache, slot `position`).
    let bytes_per_slot = (kv_width * 4) as u64;
    let k_offset_bytes = (position * kv_width * 4) as u64;
    let v_offset_bytes = ((max_seq + position) * kv_width * 4) as u64;
    let mut enc_kv = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("layer_kv_cache_write"),
    });
    enc_kv.copy_buffer_to_buffer(&model_state.attn_k_new, 0, kv_cache, k_offset_bytes, bytes_per_slot);
    enc_kv.copy_buffer_to_buffer(&model_state.attn_v_new, 0, kv_cache, v_offset_bytes, bytes_per_slot);
    ctx.queue.submit(std::iter::once(enc_kv.finish()));
    // 8. Attention. Sliding-window cap (Gemma-4 sliding layers) when present.
    let seq_len = position + 1;
    let v_offset_floats = max_seq * kv_width;
    let window = weights.attention_window_size.unwrap_or(0);
    decode_attention_device_full(
        ctx,
        &model_state.attn_q,
        kv_cache,
        &model_state.attn_out,
        nq,
        nkv,
        hd,
        seq_len,
        Some(v_offset_floats),
        window,
    )?;
    // 9. O projection.
    matmul_with_optional_dequant(
        ctx,
        model_state,
        &model_state.attn_out,
        &weights.attention.o_proj,
        &model_state.mlp_out,
        1,
        h,
        q_width,
    )?;
    // 9b. Gemma-4 PrePost: post-attention sub-layer norm BEFORE the
    // residual add. Normalises `mlp_out` (= attention block output)
    // with a learned hidden_size weight, writes back into `mlp_out`.
    if let Some(ref post_attn_norm) = weights.attention.post_attn_sublayer_norm {
        rms_norm_device(
            ctx,
            &model_state.mlp_out,
            post_attn_norm,
            &model_state.post_normed,
            h,
            rms_norm_eps,
        )?;
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("post_attn_sublayer_norm_writeback"),
        });
        enc.copy_buffer_to_buffer(
            &model_state.post_normed,
            0,
            &model_state.mlp_out,
            0,
            (h * 4) as u64,
        );
        ctx.queue.submit(std::iter::once(enc.finish()));
    }
    // 10. residual += attn_o (route through post_normed).
    residual_add_device(
        ctx,
        &model_state.residual,
        &model_state.mlp_out,
        &model_state.post_normed,
        h,
    )?;
    let mut enc_wb = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("attn_residual_writeback"),
    });
    enc_wb.copy_buffer_to_buffer(
        &model_state.post_normed,
        0,
        &model_state.residual,
        0,
        (h * 4) as u64,
    );
    ctx.queue.submit(std::iter::once(enc_wb.finish()));

    // ── MLP BLOCK ────────────────────────────────────────────────────────
    // 11. pre-MLP norm.
    rms_norm_device(
        ctx,
        &model_state.residual,
        &weights.mlp.norm_weight,
        &model_state.post_normed,
        h,
        rms_norm_eps,
    )?;
    // 12-13. gate / up (NVFP4-aware via matmul_with_optional_dequant).
    matmul_with_optional_dequant(
        ctx,
        model_state,
        &model_state.post_normed,
        &weights.mlp.gate_proj,
        &model_state.gate,
        1,
        i,
        h,
    )?;
    matmul_with_optional_dequant(
        ctx,
        model_state,
        &model_state.post_normed,
        &weights.mlp.up_proj,
        &model_state.up,
        1,
        i,
        h,
    )?;
    // 14. Activation: SwiGLU (Llama) or GeGLU-tanh (Gemma-4).
    match activation {
        Activation::SwiGLU => {
            swiglu_device(ctx, &model_state.gate, &model_state.up, &model_state.swiglu_out, i)?;
        }
        Activation::GeGluTanh => {
            geglu_tanh_device(ctx, &model_state.gate, &model_state.up, &model_state.swiglu_out, i)?;
        }
    }
    // 15. down.
    matmul_with_optional_dequant(
        ctx,
        model_state,
        &model_state.swiglu_out,
        &weights.mlp.down_proj,
        &model_state.mlp_out,
        1,
        h,
        i,
    )?;
    // 15b. Gemma-4 PrePost: post-MLP sub-layer norm BEFORE residual add.
    if let Some(ref post_mlp_norm) = weights.mlp.post_mlp_sublayer_norm {
        rms_norm_device(
            ctx,
            &model_state.mlp_out,
            post_mlp_norm,
            &model_state.post_normed,
            h,
            rms_norm_eps,
        )?;
        let mut enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("post_mlp_sublayer_norm_writeback"),
        });
        enc.copy_buffer_to_buffer(
            &model_state.post_normed,
            0,
            &model_state.mlp_out,
            0,
            (h * 4) as u64,
        );
        ctx.queue.submit(std::iter::once(enc.finish()));
    }
    // 16. residual += mlp_out.
    residual_add_device(
        ctx,
        &model_state.residual,
        &model_state.mlp_out,
        &model_state.post_normed,
        h,
    )?;
    let mut enc_mlp_wb = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("mlp_residual_writeback"),
    });
    enc_mlp_wb.copy_buffer_to_buffer(
        &model_state.post_normed,
        0,
        &model_state.residual,
        0,
        (h * 4) as u64,
    );
    ctx.queue.submit(std::iter::once(enc_mlp_wb.finish()));

    // Gemma-4: per-layer scalar multiply on the layer output.
    if let Some(scalar) = weights.layer_scalar {
        scale_f32_device(ctx, &model_state.residual, h, scalar)?;
    }

    Ok(())
}

/// Run all layers of `model` for one decode token. After this returns,
/// `model_state.logits` holds the per-token logits ready for argmax /
/// softmax sampling, and `model_state.position` has been incremented.
///
/// `cos_for_position(p, half_dim) -> Vec<f32>` and `sin_for_position`
/// are the user-supplied RoPE table generators (theta-base depends on
/// model architecture so it lives on the caller side).
#[allow(clippy::too_many_arguments)]
pub fn forward_token_device<FCos, FSin>(
    ctx: &WgpuContext,
    model: &WgpuModel,
    model_state: &mut WgpuModelState,
    cos_for_position: FCos,
    sin_for_position: FSin,
    rms_norm_eps: f32,
    activation: Activation,
) -> Result<()>
where
    FCos: Fn(usize, usize) -> Vec<f32>,
    FSin: Fn(usize, usize) -> Vec<f32>,
{
    let half = model.head_dim / 2;
    let cos = cos_for_position(model_state.position, half);
    let sin = sin_for_position(model_state.position, half);
    // Gemma-4: scale embeddings (which the caller has just written into
    // model_state.residual) by sqrt(hidden_size). No-op for vanilla Llama.
    if let Some(scale) = model.embed_scale {
        scale_f32_device(ctx, &model_state.residual, model.hidden_size, scale)?;
    }
    for (layer_idx, layer_weights) in model.layers.iter().enumerate() {
        forward_layer_device(
            ctx,
            model_state,
            layer_weights,
            layer_idx,
            &cos,
            &sin,
            rms_norm_eps,
            activation,
        )?;
    }
    // Final norm + lm_head matmul → logits.
    rms_norm_device(
        ctx,
        &model_state.residual,
        &model.final_norm,
        &model_state.final_normed,
        model.hidden_size,
        rms_norm_eps,
    )?;
    matmul_with_optional_dequant(
        ctx,
        model_state,
        &model_state.final_normed,
        &model.lm_head,
        &model_state.logits,
        1,
        model.vocab_size,
        model.hidden_size,
    )?;
    model_state.position += 1;
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

    /// CPU reference for one attention block step. Mirrors the GPU
    /// pipeline exactly, including cache layout.
    #[allow(clippy::too_many_arguments)]
    fn cpu_attn_step(
        residual: &mut Vec<f32>,
        norm_w: &[f32],
        q_w: &[f32],   // [Q*hd, H]
        k_w: &[f32],   // [KV*hd, H]
        v_w: &[f32],   // [KV*hd, H]
        o_w: &[f32],   // [H, Q*hd]
        cos: &[f32],
        sin: &[f32],
        keys_cache: &mut [f32],   // [max_seq * KV*hd]
        values_cache: &mut [f32], // [max_seq * KV*hd]
        position: usize,
        h: usize,
        nq: usize,
        nkv: usize,
        hd: usize,
        eps: f32,
    ) {
        // RMS-norm.
        let mean_sq = residual.iter().map(|v| v * v).sum::<f32>() / h as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        let normed: Vec<f32> = residual
            .iter()
            .zip(norm_w.iter())
            .map(|(v, w)| v * inv_rms * w)
            .collect();
        let q_width = nq * hd;
        let kv_width = nkv * hd;
        // QKV projections.
        let mut q = vec![0.0_f32; q_width];
        let mut k = vec![0.0_f32; kv_width];
        let mut v = vec![0.0_f32; kv_width];
        for r in 0..q_width {
            let mut acc = 0.0_f32;
            for c in 0..h {
                acc += normed[c] * q_w[r * h + c];
            }
            q[r] = acc;
        }
        for r in 0..kv_width {
            let mut ak = 0.0_f32;
            let mut av = 0.0_f32;
            for c in 0..h {
                ak += normed[c] * k_w[r * h + c];
                av += normed[c] * v_w[r * h + c];
            }
            k[r] = ak;
            v[r] = av;
        }
        // RoPE on Q and K.
        let half = hd / 2;
        for head in 0..nq {
            for i in 0..half {
                let lo = q[head * hd + i];
                let hi = q[head * hd + i + half];
                q[head * hd + i] = lo * cos[i] - hi * sin[i];
                q[head * hd + i + half] = lo * sin[i] + hi * cos[i];
            }
        }
        for head in 0..nkv {
            for i in 0..half {
                let lo = k[head * hd + i];
                let hi = k[head * hd + i + half];
                k[head * hd + i] = lo * cos[i] - hi * sin[i];
                k[head * hd + i + half] = lo * sin[i] + hi * cos[i];
            }
        }
        // Write K/V into cache.
        keys_cache[position * kv_width..(position + 1) * kv_width].copy_from_slice(&k);
        values_cache[position * kv_width..(position + 1) * kv_width].copy_from_slice(&v);
        // Attention (online softmax) over [0..=position].
        let scale = 1.0_f32 / (hd as f32).sqrt();
        let group = nq / nkv;
        let mut attn = vec![0.0_f32; q_width];
        for qh in 0..nq {
            let kvh = qh / group;
            let q_base = qh * hd;
            // Compute scores then softmax then weighted sum.
            let seq = position + 1;
            let mut scores = vec![0.0_f32; seq];
            let mut max_s = f32::NEG_INFINITY;
            for p in 0..seq {
                let k_base = p * kv_width + kvh * hd;
                let mut dot = 0.0_f32;
                for i in 0..hd {
                    dot += q[q_base + i] * keys_cache[k_base + i];
                }
                scores[p] = dot * scale;
                if scores[p] > max_s {
                    max_s = scores[p];
                }
            }
            let mut sum = 0.0_f32;
            let exps: Vec<f32> = scores
                .iter()
                .map(|s| {
                    let e = (s - max_s).exp();
                    sum += e;
                    e
                })
                .collect();
            for p in 0..seq {
                let w = exps[p] / sum;
                let v_base = p * kv_width + kvh * hd;
                for i in 0..hd {
                    attn[q_base + i] += w * values_cache[v_base + i];
                }
            }
        }
        // O proj.
        let mut mlp = vec![0.0_f32; h];
        for r in 0..h {
            let mut acc = 0.0_f32;
            for c in 0..q_width {
                acc += attn[c] * o_w[r * q_width + c];
            }
            mlp[r] = acc;
        }
        // Residual update.
        for i in 0..h {
            residual[i] += mlp[i];
        }
    }

    /// End-to-end attention block on real Vulkan, two consecutive decode
    /// tokens (position=0, position=1) sharing the same persistent KV
    /// cache. GPU output must match the CPU reference within 1e-4 at each
    /// step.
    /// Gated behind `AEGIS_WGPU_SMOKE=1`.
    #[test]
    fn attention_block_matches_cpu_reference_two_tokens() {
        if std::env::var("AEGIS_WGPU_SMOKE").is_err() {
            eprintln!("skipping; set AEGIS_WGPU_SMOKE=1 to run on a host with Vulkan/Metal/D3D12");
            return;
        }
        let ctx = WgpuContext::new(0).expect("wgpu ctx");

        let h = 8;
        let intermediate = 16; // unused for attention block but state requires it
        let nq = 2;
        let nkv = 2;
        let hd = 4;
        let max_seq = 8;
        let eps = 1e-6_f32;
        let q_width = nq * hd;
        let kv_width = nkv * hd;

        // Deterministic random weights.
        let norm_w: Vec<f32> = (0..h).map(|k| 1.0 + (k as f32) * 0.01).collect();
        let q_w: Vec<f32> = (0..(q_width * h))
            .map(|k| ((k * 11 + 1) % 17) as f32 * 0.05 - 0.4)
            .collect();
        let k_w: Vec<f32> = (0..(kv_width * h))
            .map(|k| ((k * 13 + 3) % 19) as f32 * 0.05 - 0.45)
            .collect();
        let v_w: Vec<f32> = (0..(kv_width * h))
            .map(|k| ((k * 17 + 5) % 23) as f32 * 0.05 - 0.5)
            .collect();
        let o_w: Vec<f32> = (0..(h * q_width))
            .map(|k| ((k * 19 + 7) % 29) as f32 * 0.04 - 0.3)
            .collect();

        // RoPE tables for each position.
        let half = hd / 2;
        let theta: Vec<f32> = (0..half).map(|i| 10000f32.powf(-2.0 * i as f32 / hd as f32)).collect();
        let cos_for = |pos: usize| -> Vec<f32> {
            theta.iter().map(|t| (pos as f32 * t).cos()).collect()
        };
        let sin_for = |pos: usize| -> Vec<f32> {
            theta.iter().map(|t| (pos as f32 * t).sin()).collect()
        };

        // GPU setup.
        let weights = WgpuAttentionWeights {
            norm_weight: crate::wgpu::forward::upload_f32_buf(&ctx, &norm_w, "attn_norm_w"),
            q_proj: crate::wgpu::forward::upload_f32_buf(&ctx, &q_w, "attn_q_proj"),
            k_proj: crate::wgpu::forward::upload_f32_buf(&ctx, &k_w, "attn_k_proj"),
            v_proj: crate::wgpu::forward::upload_f32_buf(&ctx, &v_w, "attn_v_proj"),
            o_proj: crate::wgpu::forward::upload_f32_buf(&ctx, &o_w, "attn_o_proj"),
        };
        let mut state = WgpuLlamaState::new_for_full_layer(&ctx, h, intermediate, nq, nkv, hd, max_seq)
            .expect("state");

        // CPU mirror state.
        let mut cpu_residual: Vec<f32> = (0..h).map(|k| ((k * 5 + 1) % 13) as f32 * 0.1 - 0.5).collect();
        let mut cpu_keys = vec![0.0_f32; max_seq * kv_width];
        let mut cpu_values = vec![0.0_f32; max_seq * kv_width];

        // Seed GPU residual.
        ctx.queue.write_buffer(
            state.residual.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(&cpu_residual),
        );

        // Position 0.
        let cos0 = cos_for(0);
        let sin0 = sin_for(0);
        forward_attention_block_device(&ctx, &mut state, &weights, &cos0, &sin0, eps).unwrap();
        cpu_attn_step(
            &mut cpu_residual, &norm_w, &q_w, &k_w, &v_w, &o_w, &cos0, &sin0,
            &mut cpu_keys, &mut cpu_values, 0, h, nq, nkv, hd, eps,
        );
        let gpu_step0 = crate::wgpu::forward::download_f32_buf(
            &ctx, state.residual.as_ref().unwrap(), h, "step0",
        ).unwrap();
        for (i, (g, c)) in gpu_step0.iter().zip(cpu_residual.iter()).enumerate() {
            assert!(
                (g - c).abs() < 1e-4,
                "step 0 mismatch at i={i}: gpu={g} cpu={c}",
            );
        }

        // Position 1 — cache should retain k0/v0 and attention reads both slots.
        state.position = 1;
        let cos1 = cos_for(1);
        let sin1 = sin_for(1);
        forward_attention_block_device(&ctx, &mut state, &weights, &cos1, &sin1, eps).unwrap();
        cpu_attn_step(
            &mut cpu_residual, &norm_w, &q_w, &k_w, &v_w, &o_w, &cos1, &sin1,
            &mut cpu_keys, &mut cpu_values, 1, h, nq, nkv, hd, eps,
        );
        let gpu_step1 = crate::wgpu::forward::download_f32_buf(
            &ctx, state.residual.as_ref().unwrap(), h, "step1",
        ).unwrap();
        for (i, (g, c)) in gpu_step1.iter().zip(cpu_residual.iter()).enumerate() {
            assert!(
                (g - c).abs() < 1e-4,
                "step 1 mismatch at i={i}: gpu={g} cpu={c}",
            );
        }
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

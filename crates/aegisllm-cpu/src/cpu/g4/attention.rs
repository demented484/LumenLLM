//! Gemma-4 decode attention: causal + sliding-window, GQA, scale override.
//!
//! Mirrors the CUDA `attention_decode_split` path
//! (`crates/aegisllm-cuda/src/executor/attention.rs:239-321`) and the kernel
//! masking (`attention_decode_common.cuh:82,192-209`):
//!
//!   window_start = (window_size>0 && seq_len>window_size) ? seq_len-window_size : 0
//!   score(pos)   = dot(q, k[pos]) * scale     for pos in [window_start, seq_len)
//!   positions < window_start are masked (skipped).
//!   online softmax → Σ weight * v[pos].
//!
//! Gemma-4 folds its effective attention scale of 1.0 by pre-scaling Q by
//! `sqrt(head_dim)`; therefore `scale` here is passed as 1.0 (the equivalent,
//! cheaper formulation noted in the spec §1 step 12). There is NO attention
//! logit softcap on Gemma-4 (attn_logit_softcapping is null/unused).
//!
//! KV layout: `keys`/`values` are `[seq_len, num_kv_heads * head_dim]`
//! row-major (linear, position-indexed). Correctness-first: the CPU path does
//! not use a sliding-window ring buffer; it stores all positions and masks at
//! attention time. This is bit-identical to the GPU result (the ring buffer is
//! purely a VRAM-footprint optimization).

use crate::cpu::simd;
use aegisllm_base::error::{AegisError, Result};
use rayon::prelude::*;

pub(crate) struct G4DecodeAttnRequest<'a> {
    pub(crate) keys: &'a [f32],
    pub(crate) values: &'a [f32],
    pub(crate) seq_len: usize,
    pub(crate) query: &'a [f32],
    pub(crate) num_attention_heads: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) head_dim: usize,
    /// 0 = full causal; >0 = attend only to the last `window_size` positions.
    pub(crate) window_size: usize,
    /// Softmax scale folded into scores. 1.0 for Gemma-4 (Q pre-scaled).
    pub(crate) scale: f32,
}

pub(crate) fn g4_attention_decode_into(
    req: G4DecodeAttnRequest<'_>,
    out: &mut [f32],
) -> Result<()> {
    if req.num_attention_heads == 0 || req.num_kv_heads == 0 || req.head_dim == 0 {
        return Err(AegisError::InvalidPlan(format!(
            "g4 attention dims must be non-zero: q_heads={} kv_heads={} head_dim={}",
            req.num_attention_heads, req.num_kv_heads, req.head_dim
        )));
    }
    if !req.num_attention_heads.is_multiple_of(req.num_kv_heads) {
        return Err(AegisError::InvalidPlan(
            "g4 attention: attention heads must be divisible by kv heads".into(),
        ));
    }
    let q_width = req.num_attention_heads * req.head_dim;
    let kv_width = req.num_kv_heads * req.head_dim;
    if req.query.len() != q_width || out.len() != q_width {
        return Err(AegisError::InvalidPlan(format!(
            "g4 attention query/output shape mismatch: query={} output={} expected={}",
            req.query.len(),
            out.len(),
            q_width
        )));
    }
    let required_kv = req.seq_len * kv_width;
    if req.keys.len() < required_kv || req.values.len() < required_kv {
        return Err(AegisError::InvalidPlan(format!(
            "g4 attention KV too small: need {} keys={} values={}",
            required_kv,
            req.keys.len(),
            req.values.len()
        )));
    }

    let group = req.num_attention_heads / req.num_kv_heads;
    let window_start = if req.window_size > 0 && req.seq_len > req.window_size {
        req.seq_len - req.window_size
    } else {
        0
    };

    out.par_chunks_mut(req.head_dim)
        .enumerate()
        .for_each(|(head, head_out)| {
            let kv_head = head / group;
            let q = &req.query[head * req.head_dim..(head + 1) * req.head_dim];
            attention_head_into(
                req.keys,
                req.values,
                window_start,
                req.seq_len,
                q,
                req.num_kv_heads,
                req.head_dim,
                kv_head,
                req.scale,
                head_out,
            );
        });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn attention_head_into(
    keys: &[f32],
    values: &[f32],
    window_start: usize,
    seq_len: usize,
    q: &[f32],
    num_kv_heads: usize,
    head_dim: usize,
    kv_head: usize,
    scale: f32,
    out: &mut [f32],
) {
    let mut max_score = f32::NEG_INFINITY;
    let mut score_sum = 0.0_f32;
    out.fill(0.0);

    for pos in window_start..seq_len {
        let off = (pos * num_kv_heads + kv_head) * head_dim;
        let k = &keys[off..off + head_dim];
        let score = simd::dot_f32(q, k) * scale;
        if score > max_score {
            let rescale = (max_score - score).exp();
            simd::scale_in_place(out, rescale);
            score_sum *= rescale;
            max_score = score;
        }
        let weight = (score - max_score).exp();
        score_sum += weight;
        let v = &values[off..off + head_dim];
        simd::axpy(out, v, weight);
    }

    if score_sum > 0.0 {
        simd::scale_in_place(out, 1.0 / score_sum);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_one_unscaled_dot() {
        // Single head, head_dim=2, seq_len=2, scale=1.0.
        let keys = [1.0f32, 0.0, 0.0, 1.0];
        let values = [10.0f32, 0.0, 0.0, 20.0];
        let query = [1.0f32, 0.0];
        let mut out = [0.0f32; 2];
        g4_attention_decode_into(
            G4DecodeAttnRequest {
                keys: &keys,
                values: &values,
                seq_len: 2,
                query: &query,
                num_attention_heads: 1,
                num_kv_heads: 1,
                head_dim: 2,
                window_size: 0,
                scale: 1.0,
            },
            &mut out,
        )
        .unwrap();
        // scores: pos0 = q·k0 = 1.0; pos1 = q·k1 = 0.0.
        let w0 = 1.0f32; // exp(score-max)=exp(0)
        let w1 = (-1.0f32).exp();
        let denom = w0 + w1;
        assert!((out[0] - 10.0 * w0 / denom).abs() < 1e-5);
        assert!((out[1] - 20.0 * w1 / denom).abs() < 1e-5);
    }

    #[test]
    fn sliding_window_masks_old_positions() {
        // seq_len=4, window_size=2 → window_start=2. Positions 0,1 masked.
        // head_dim=1 so dot = q*k.
        let keys = [1.0f32, 1.0, 1.0, 1.0];
        let values = [100.0f32, 200.0, 3.0, 7.0];
        let query = [0.0f32]; // all scores 0 → uniform over attended positions.
        let mut out = [0.0f32; 1];
        g4_attention_decode_into(
            G4DecodeAttnRequest {
                keys: &keys,
                values: &values,
                seq_len: 4,
                query: &query,
                num_attention_heads: 1,
                num_kv_heads: 1,
                head_dim: 1,
                window_size: 2,
                scale: 1.0,
            },
            &mut out,
        )
        .unwrap();
        // Only positions 2,3 attended, equal weights → mean(3,7) = 5.
        assert!((out[0] - 5.0).abs() < 1e-5, "got {}", out[0]);
    }

    #[test]
    fn window_inactive_until_exceeded() {
        // seq_len == window_size → window_start = 0 (all attended).
        let keys = [1.0f32, 1.0];
        let values = [4.0f32, 6.0];
        let query = [0.0f32];
        let mut out = [0.0f32; 1];
        g4_attention_decode_into(
            G4DecodeAttnRequest {
                keys: &keys,
                values: &values,
                seq_len: 2,
                query: &query,
                num_attention_heads: 1,
                num_kv_heads: 1,
                head_dim: 1,
                window_size: 2,
                scale: 1.0,
            },
            &mut out,
        )
        .unwrap();
        assert!((out[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn gqa_head_mapping() {
        // 2 q-heads, 1 kv-head, head_dim=2, seq_len=1.
        let keys = [1.0f32, 0.0];
        let values = [1.0f32, 2.0];
        let query = [1.0f32, 0.0, 0.0, 1.0];
        let mut out = [0.0f32; 4];
        g4_attention_decode_into(
            G4DecodeAttnRequest {
                keys: &keys,
                values: &values,
                seq_len: 1,
                query: &query,
                num_attention_heads: 2,
                num_kv_heads: 1,
                head_dim: 2,
                window_size: 0,
                scale: 1.0,
            },
            &mut out,
        )
        .unwrap();
        // seq_len=1 → both heads attend the single position → value [1,2].
        assert_eq!(out, [1.0, 2.0, 1.0, 2.0]);
    }
}

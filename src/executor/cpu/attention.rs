use rayon::prelude::*;

use super::math::dot_f32;
use super::state::CpuLayerState;
use crate::error::{AegisError, Result};

pub(super) fn attention_into(
    state: &CpuLayerState,
    query: &[f32],
    num_attention_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) -> Result<()> {
    if query.len() != num_attention_heads * head_dim || out.len() != query.len() {
        return Err(AegisError::InvalidPlan("attention shape mismatch".into()));
    }
    if num_attention_heads % num_kv_heads != 0 {
        return Err(AegisError::InvalidPlan(
            "attention heads must be divisible by kv heads".into(),
        ));
    }
    let group = num_attention_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    out.par_chunks_mut(head_dim)
        .enumerate()
        .for_each(|(head, head_out)| {
            attention_head_into(
                &state.keys,
                &state.values,
                state.seq_len,
                query,
                head,
                group,
                num_kv_heads,
                head_dim,
                scale,
                head_out,
            );
        });
    Ok(())
}

fn attention_head_into(
    keys: &[f32],
    values: &[f32],
    seq_len: usize,
    query: &[f32],
    head: usize,
    group: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    out: &mut [f32],
) {
    let kv_head = head / group;
    let q = &query[head * head_dim..(head + 1) * head_dim];
    let mut max_score = f32::NEG_INFINITY;
    let mut score_sum = 0.0_f32;
    out.fill(0.0);

    for pos in 0..seq_len {
        let key_offset = (pos * num_kv_heads + kv_head) * head_dim;
        let k = &keys[key_offset..key_offset + head_dim];
        let score = dot_f32(q, k) * scale;
        if score > max_score {
            let rescale = (max_score - score).exp();
            for value in out.iter_mut() {
                *value *= rescale;
            }
            score_sum *= rescale;
            max_score = score;
        }
        let weight = (score - max_score).exp();
        score_sum += weight;
        let value_offset = (pos * num_kv_heads + kv_head) * head_dim;
        let v = &values[value_offset..value_offset + head_dim];
        for i in 0..head_dim {
            out[i] += weight * v[i];
        }
    }

    if score_sum > 0.0 {
        let inv = 1.0 / score_sum;
        for value in out.iter_mut() {
            *value *= inv;
        }
    }
}

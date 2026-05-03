use rayon::prelude::*;

use aegisllm_base::error::{AegisError, Result};
use super::cpu::simd;

#[derive(Debug, Clone, Copy)]
pub struct ReferenceAttentionDecodeRequest<'a> {
    pub keys: &'a [f32],
    pub values: &'a [f32],
    pub seq_len: usize,
    pub query: &'a [f32],
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct ReferenceAttentionPrefillRequest<'a> {
    pub keys: &'a [f32],
    pub values: &'a [f32],
    pub start_position: usize,
    pub batch: usize,
    pub query: &'a [f32],
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl ReferenceAttentionDecodeRequest<'_> {
    fn validate(self, out: &[f32]) -> Result<()> {
        if self.num_attention_heads == 0 || self.num_kv_heads == 0 || self.head_dim == 0 {
            return Err(AegisError::InvalidPlan(format!(
                "reference attention dimensions must be non-zero: q_heads={} kv_heads={} head_dim={}",
                self.num_attention_heads, self.num_kv_heads, self.head_dim
            )));
        }
        if !self.num_attention_heads.is_multiple_of(self.num_kv_heads) {
            return Err(AegisError::InvalidPlan(
                "reference attention attention heads must be divisible by kv heads".into(),
            ));
        }
        let q_width = self.num_attention_heads * self.head_dim;
        let kv_width = self.num_kv_heads * self.head_dim;
        if self.query.len() != q_width || out.len() != q_width {
            return Err(AegisError::InvalidPlan(format!(
                "reference attention query/output shape mismatch: query={} output={} expected={}",
                self.query.len(),
                out.len(),
                q_width
            )));
        }
        let required_kv = self.seq_len.checked_mul(kv_width).ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "reference attention KV length overflow: seq_len={} kv_width={}",
                self.seq_len, kv_width
            ))
        })?;
        if self.keys.len() < required_kv || self.values.len() < required_kv {
            return Err(AegisError::InvalidPlan(format!(
                "reference attention KV cache too small: seq_len={} kv_width={} keys={} values={}",
                self.seq_len,
                kv_width,
                self.keys.len(),
                self.values.len()
            )));
        }
        Ok(())
    }
}

#[allow(dead_code)]
impl ReferenceAttentionPrefillRequest<'_> {
    fn validate(self, out: &[f32]) -> Result<()> {
        if self.num_attention_heads == 0 || self.num_kv_heads == 0 || self.head_dim == 0 {
            return Err(AegisError::InvalidPlan(format!(
                "reference attention prefill dimensions must be non-zero: q_heads={} kv_heads={} head_dim={}",
                self.num_attention_heads, self.num_kv_heads, self.head_dim
            )));
        }
        if !self.num_attention_heads.is_multiple_of(self.num_kv_heads) {
            return Err(AegisError::InvalidPlan(
                "reference attention prefill attention heads must be divisible by kv heads".into(),
            ));
        }
        let q_width = self.num_attention_heads * self.head_dim;
        let kv_width = self.num_kv_heads * self.head_dim;
        let query_len = self.batch.checked_mul(q_width).ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "reference attention prefill query length overflow: batch={} q_width={}",
                self.batch, q_width
            ))
        })?;
        if self.query.len() != query_len || out.len() != query_len {
            return Err(AegisError::InvalidPlan(format!(
                "reference attention prefill query/output shape mismatch: query={} output={} expected={}",
                self.query.len(),
                out.len(),
                query_len
            )));
        }
        let max_seq_len = self.start_position.checked_add(self.batch).ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "reference attention prefill max sequence overflow: start={} batch={}",
                self.start_position, self.batch
            ))
        })?;
        let required_kv = max_seq_len.checked_mul(kv_width).ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "reference attention prefill KV length overflow: seq_len={} kv_width={}",
                max_seq_len, kv_width
            ))
        })?;
        if self.keys.len() < required_kv || self.values.len() < required_kv {
            return Err(AegisError::InvalidPlan(format!(
                "reference attention prefill KV cache too small: seq_len={} kv_width={} keys={} values={}",
                max_seq_len,
                kv_width,
                self.keys.len(),
                self.values.len()
            )));
        }
        Ok(())
    }
}

pub fn reference_attention_decode_f32_into(
    request: ReferenceAttentionDecodeRequest<'_>,
    out: &mut [f32],
) -> Result<()> {
    request.validate(out)?;
    let group = request.num_attention_heads / request.num_kv_heads;
    let scale = 1.0 / (request.head_dim as f32).sqrt();
    out.par_chunks_mut(request.head_dim)
        .enumerate()
        .for_each(|(head, head_out)| {
            reference_attention_decode_head_f32_into(request, head, group, scale, head_out);
        });
    Ok(())
}

#[allow(dead_code)]
pub fn reference_attention_prefill_f32_into(
    request: ReferenceAttentionPrefillRequest<'_>,
    out: &mut [f32],
) -> Result<()> {
    request.validate(out)?;
    let group = request.num_attention_heads / request.num_kv_heads;
    let scale = 1.0 / (request.head_dim as f32).sqrt();
    let q_width = request.num_attention_heads * request.head_dim;
    out.par_chunks_mut(q_width)
        .enumerate()
        .for_each(|(batch_idx, token_out)| {
            let seq_len = request.start_position + batch_idx + 1;
            let query_base = batch_idx * q_width;
            token_out
                .par_chunks_mut(request.head_dim)
                .enumerate()
                .for_each(|(head, head_out)| {
                    let query_offset = query_base + head * request.head_dim;
                    reference_attention_head_f32_into(
                        request.keys,
                        request.values,
                        seq_len,
                        &request.query[query_offset..query_offset + request.head_dim],
                        request.num_kv_heads,
                        request.head_dim,
                        head / group,
                        scale,
                        head_out,
                    );
                });
        });
    Ok(())
}

fn reference_attention_decode_head_f32_into(
    request: ReferenceAttentionDecodeRequest<'_>,
    head: usize,
    group: usize,
    scale: f32,
    out: &mut [f32],
) {
    let kv_head = head / group;
    let q = &request.query[head * request.head_dim..(head + 1) * request.head_dim];
    reference_attention_head_f32_into(
        request.keys,
        request.values,
        request.seq_len,
        q,
        request.num_kv_heads,
        request.head_dim,
        kv_head,
        scale,
        out,
    );
}

#[allow(clippy::too_many_arguments)]
fn reference_attention_head_f32_into(
    keys: &[f32],
    values: &[f32],
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

    for pos in 0..seq_len {
        let key_offset = (pos * num_kv_heads + kv_head) * head_dim;
        let k = &keys[key_offset..key_offset + head_dim];
        let score = simd::dot_f32(q, k) * scale;
        if score > max_score {
            let rescale = (max_score - score).exp();
            simd::scale_in_place(out, rescale);
            score_sum *= rescale;
            max_score = score;
        }
        let weight = (score - max_score).exp();
        score_sum += weight;
        let value_offset = (pos * num_kv_heads + kv_head) * head_dim;
        let v = &values[value_offset..value_offset + head_dim];
        simd::axpy(out, v, weight);
    }

    if score_sum > 0.0 {
        let inv = 1.0 / score_sum;
        simd::scale_in_place(out, inv);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ReferenceAttentionDecodeRequest, ReferenceAttentionPrefillRequest,
        reference_attention_decode_f32_into, reference_attention_prefill_f32_into,
    };

    #[test]
    fn reference_attention_decode_matches_manual_single_head() {
        let inv_sqrt2 = 1.0_f32 / 2.0_f32.sqrt();
        let keys = [1.0, 0.0, 0.0, 1.0];
        let values = [10.0, 0.0, 0.0, 20.0];
        let query = [1.0, 0.0];
        let mut out = [0.0; 2];
        reference_attention_decode_f32_into(
            ReferenceAttentionDecodeRequest {
                keys: &keys,
                values: &values,
                seq_len: 2,
                query: &query,
                num_attention_heads: 1,
                num_kv_heads: 1,
                head_dim: 2,
            },
            &mut out,
        )
        .unwrap();

        let w0 = 1.0_f32;
        let w1 = (-inv_sqrt2).exp();
        let denom = w0 + w1;
        assert!((out[0] - 10.0 * w0 / denom).abs() < 1.0e-5);
        assert!((out[1] - 20.0 * w1 / denom).abs() < 1.0e-5);
    }

    #[test]
    fn reference_attention_decode_supports_gqa_head_mapping() {
        let keys = [1.0, 0.0, 0.0, 1.0];
        let values = [1.0, 2.0, 3.0, 4.0];
        let query = [1.0, 0.0, 0.0, 1.0];
        let mut out = [0.0; 4];
        reference_attention_decode_f32_into(
            ReferenceAttentionDecodeRequest {
                keys: &keys,
                values: &values,
                seq_len: 1,
                query: &query,
                num_attention_heads: 2,
                num_kv_heads: 1,
                head_dim: 2,
            },
            &mut out,
        )
        .unwrap();
        assert_eq!(out, [1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn reference_attention_prefill_matches_decode_for_each_causal_row() {
        let keys = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let values = [10.0, 0.0, 0.0, 20.0, 5.0, 5.0];
        let query = [1.0, 0.0, 0.0, 1.0];
        let mut prefill = [0.0; 4];
        reference_attention_prefill_f32_into(
            ReferenceAttentionPrefillRequest {
                keys: &keys,
                values: &values,
                start_position: 0,
                batch: 2,
                query: &query,
                num_attention_heads: 1,
                num_kv_heads: 1,
                head_dim: 2,
            },
            &mut prefill,
        )
        .unwrap();

        let mut row0 = [0.0; 2];
        reference_attention_decode_f32_into(
            ReferenceAttentionDecodeRequest {
                keys: &keys,
                values: &values,
                seq_len: 1,
                query: &query[0..2],
                num_attention_heads: 1,
                num_kv_heads: 1,
                head_dim: 2,
            },
            &mut row0,
        )
        .unwrap();
        let mut row1 = [0.0; 2];
        reference_attention_decode_f32_into(
            ReferenceAttentionDecodeRequest {
                keys: &keys,
                values: &values,
                seq_len: 2,
                query: &query[2..4],
                num_attention_heads: 1,
                num_kv_heads: 1,
                head_dim: 2,
            },
            &mut row1,
        )
        .unwrap();
        assert_eq!(prefill[0..2], row0);
        assert_eq!(prefill[2..4], row1);
    }

    #[test]
    fn reference_attention_decode_rejects_bad_shapes() {
        let mut out = [0.0; 2];
        let error = reference_attention_decode_f32_into(
            ReferenceAttentionDecodeRequest {
                keys: &[0.0],
                values: &[0.0],
                seq_len: 1,
                query: &[0.0, 0.0],
                num_attention_heads: 2,
                num_kv_heads: 3,
                head_dim: 1,
            },
            &mut out,
        )
        .unwrap_err();
        assert!(error.to_string().contains("divisible"));
    }
}

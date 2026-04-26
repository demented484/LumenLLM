use rayon::prelude::*;

use crate::error::{AegisError, Result};

#[derive(Debug, Clone, Copy)]
pub(super) struct SdpaDecodeRequest<'a> {
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
pub(super) struct SdpaPrefillRequest<'a> {
    pub keys: &'a [f32],
    pub values: &'a [f32],
    pub start_position: usize,
    pub batch: usize,
    pub query: &'a [f32],
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl SdpaDecodeRequest<'_> {
    fn validate(self, out: &[f32]) -> Result<()> {
        if self.num_attention_heads == 0 || self.num_kv_heads == 0 || self.head_dim == 0 {
            return Err(AegisError::InvalidPlan(format!(
                "SDPA dimensions must be non-zero: q_heads={} kv_heads={} head_dim={}",
                self.num_attention_heads, self.num_kv_heads, self.head_dim
            )));
        }
        if self.num_attention_heads % self.num_kv_heads != 0 {
            return Err(AegisError::InvalidPlan(
                "SDPA attention heads must be divisible by kv heads".into(),
            ));
        }
        let q_width = self.num_attention_heads * self.head_dim;
        let kv_width = self.num_kv_heads * self.head_dim;
        if self.query.len() != q_width || out.len() != q_width {
            return Err(AegisError::InvalidPlan(format!(
                "SDPA query/output shape mismatch: query={} output={} expected={}",
                self.query.len(),
                out.len(),
                q_width
            )));
        }
        let required_kv = self.seq_len.checked_mul(kv_width).ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "SDPA KV length overflow: seq_len={} kv_width={}",
                self.seq_len, kv_width
            ))
        })?;
        if self.keys.len() < required_kv || self.values.len() < required_kv {
            return Err(AegisError::InvalidPlan(format!(
                "SDPA KV cache too small: seq_len={} kv_width={} keys={} values={}",
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
impl SdpaPrefillRequest<'_> {
    fn validate(self, out: &[f32]) -> Result<()> {
        if self.num_attention_heads == 0 || self.num_kv_heads == 0 || self.head_dim == 0 {
            return Err(AegisError::InvalidPlan(format!(
                "SDPA prefill dimensions must be non-zero: q_heads={} kv_heads={} head_dim={}",
                self.num_attention_heads, self.num_kv_heads, self.head_dim
            )));
        }
        if self.num_attention_heads % self.num_kv_heads != 0 {
            return Err(AegisError::InvalidPlan(
                "SDPA prefill attention heads must be divisible by kv heads".into(),
            ));
        }
        let q_width = self.num_attention_heads * self.head_dim;
        let kv_width = self.num_kv_heads * self.head_dim;
        let query_len = self.batch.checked_mul(q_width).ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "SDPA prefill query length overflow: batch={} q_width={}",
                self.batch, q_width
            ))
        })?;
        if self.query.len() != query_len || out.len() != query_len {
            return Err(AegisError::InvalidPlan(format!(
                "SDPA prefill query/output shape mismatch: query={} output={} expected={}",
                self.query.len(),
                out.len(),
                query_len
            )));
        }
        let max_seq_len = self.start_position.checked_add(self.batch).ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "SDPA prefill max sequence overflow: start={} batch={}",
                self.start_position, self.batch
            ))
        })?;
        let required_kv = max_seq_len.checked_mul(kv_width).ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "SDPA prefill KV length overflow: seq_len={} kv_width={}",
                max_seq_len, kv_width
            ))
        })?;
        if self.keys.len() < required_kv || self.values.len() < required_kv {
            return Err(AegisError::InvalidPlan(format!(
                "SDPA prefill KV cache too small: seq_len={} kv_width={} keys={} values={}",
                max_seq_len,
                kv_width,
                self.keys.len(),
                self.values.len()
            )));
        }
        Ok(())
    }
}

pub(super) fn sdpa_decode_f32_into(request: SdpaDecodeRequest<'_>, out: &mut [f32]) -> Result<()> {
    request.validate(out)?;
    let group = request.num_attention_heads / request.num_kv_heads;
    let scale = 1.0 / (request.head_dim as f32).sqrt();
    out.par_chunks_mut(request.head_dim)
        .enumerate()
        .for_each(|(head, head_out)| {
            sdpa_decode_head_f32_into(request, head, group, scale, head_out);
        });
    Ok(())
}

#[allow(dead_code)]
pub(super) fn sdpa_prefill_f32_into(
    request: SdpaPrefillRequest<'_>,
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
                    sdpa_head_f32_into(
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

fn sdpa_decode_head_f32_into(
    request: SdpaDecodeRequest<'_>,
    head: usize,
    group: usize,
    scale: f32,
    out: &mut [f32],
) {
    let kv_head = head / group;
    let q = &request.query[head * request.head_dim..(head + 1) * request.head_dim];
    sdpa_head_f32_into(
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
fn sdpa_head_f32_into(
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
        for dim in 0..head_dim {
            out[dim] += weight * v[dim];
        }
    }

    if score_sum > 0.0 {
        let inv = 1.0 / score_sum;
        for value in out.iter_mut() {
            *value *= inv;
        }
    }
}

fn dot_f32(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter().zip(rhs).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use super::{
        SdpaDecodeRequest, SdpaPrefillRequest, sdpa_decode_f32_into, sdpa_prefill_f32_into,
    };

    #[test]
    fn sdpa_decode_matches_manual_single_head() {
        let inv_sqrt2 = 1.0_f32 / 2.0_f32.sqrt();
        let keys = [1.0, 0.0, 0.0, 1.0];
        let values = [10.0, 0.0, 0.0, 20.0];
        let query = [1.0, 0.0];
        let mut out = [0.0; 2];
        sdpa_decode_f32_into(
            SdpaDecodeRequest {
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
    fn sdpa_decode_supports_gqa_head_mapping() {
        let keys = [1.0, 0.0, 0.0, 1.0];
        let values = [1.0, 2.0, 3.0, 4.0];
        let query = [1.0, 0.0, 0.0, 1.0];
        let mut out = [0.0; 4];
        sdpa_decode_f32_into(
            SdpaDecodeRequest {
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
    fn sdpa_prefill_matches_decode_for_each_causal_row() {
        let keys = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let values = [10.0, 0.0, 0.0, 20.0, 5.0, 5.0];
        let query = [1.0, 0.0, 0.0, 1.0];
        let mut prefill = [0.0; 4];
        sdpa_prefill_f32_into(
            SdpaPrefillRequest {
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
        sdpa_decode_f32_into(
            SdpaDecodeRequest {
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
        sdpa_decode_f32_into(
            SdpaDecodeRequest {
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
    fn sdpa_decode_rejects_bad_shapes() {
        let mut out = [0.0; 2];
        let error = sdpa_decode_f32_into(
            SdpaDecodeRequest {
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

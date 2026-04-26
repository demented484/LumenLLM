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

fn sdpa_decode_head_f32_into(
    request: SdpaDecodeRequest<'_>,
    head: usize,
    group: usize,
    scale: f32,
    out: &mut [f32],
) {
    let kv_head = head / group;
    let q = &request.query[head * request.head_dim..(head + 1) * request.head_dim];
    let mut max_score = f32::NEG_INFINITY;
    let mut score_sum = 0.0_f32;
    out.fill(0.0);

    for pos in 0..request.seq_len {
        let key_offset = (pos * request.num_kv_heads + kv_head) * request.head_dim;
        let k = &request.keys[key_offset..key_offset + request.head_dim];
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
        let value_offset = (pos * request.num_kv_heads + kv_head) * request.head_dim;
        let v = &request.values[value_offset..value_offset + request.head_dim];
        for dim in 0..request.head_dim {
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
    use super::{SdpaDecodeRequest, sdpa_decode_f32_into};

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

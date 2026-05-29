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

/// Batched chunk-attention request for PREFILL. Computes attention for ALL
/// `batch` queries of a chunk at positions `[chunk_start, chunk_start+batch)`
/// in a single call, vectorized per head, replacing the per-query
/// `g4_attention_decode_into` loop. The math is identical to looping the decode
/// path per query (same scale, same causal + sliding-window boundary, same GQA
/// mapping, same softmax) — only the softmax is a two-pass max-subtract instead
/// of the online single-pass form (numerically equivalent).
pub(crate) struct G4PrefillAttnRequest<'a> {
    /// KV cache `[total_seq, num_kv_heads * head_dim]`, row-major, f32-widened.
    pub(crate) keys: &'a [f32],
    pub(crate) values: &'a [f32],
    /// Query block `[batch, num_attention_heads * head_dim]`, row-major.
    pub(crate) queries: &'a [f32],
    /// Absolute position of query row 0 (query i has position `chunk_start+i`).
    pub(crate) chunk_start: usize,
    /// Number of query rows in this chunk.
    pub(crate) batch: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) head_dim: usize,
    /// 0 = full causal; >0 = attend only to the last `window_size` positions.
    pub(crate) window_size: usize,
    /// Softmax scale folded into scores. 1.0 for Gemma-4 (Q pre-scaled).
    pub(crate) scale: f32,
}

/// Vectorized batched chunk-attention. Writes `out` as
/// `[batch, num_attention_heads * head_dim]`. Parallelized over (query, head):
/// each task owns one `[head_dim]` output slot, mirroring the decode path's
/// `par_chunks_mut(head_dim)` granularity but across the whole chunk.
///
/// Per (query i, head):
///   - `pos = chunk_start + i`, `seq_len = pos + 1`;
///   - `window_start = (window_size>0 && seq_len>window_size) ? seq_len-window_size : 0`
///     — IDENTICAL boundary to `g4_attention_decode_into` (so the off-by-one is
///     shared, not re-derived);
///   - `score(j) = scale * dot(q, k[j])` for `j in [window_start, seq_len)`
///     (causal + sliding-window: position `j > pos` is never reached because the
///     upper bound is `seq_len = pos+1`);
///   - softmax over those scores (subtract the max for stability);
///   - `out = Σ_j softmax(j) * v[j]`.
/// GQA: `kv_head = head / (num_attention_heads / num_kv_heads)`. No softcap
/// (Gemma-4 `attn_logit_softcapping` is null — matches the decode path exactly).
pub(crate) fn g4_attention_prefill_into(
    req: G4PrefillAttnRequest<'_>,
    out: &mut [f32],
) -> Result<()> {
    if req.num_attention_heads == 0 || req.num_kv_heads == 0 || req.head_dim == 0 {
        return Err(AegisError::InvalidPlan(format!(
            "g4 prefill attention dims must be non-zero: q_heads={} kv_heads={} head_dim={}",
            req.num_attention_heads, req.num_kv_heads, req.head_dim
        )));
    }
    if !req.num_attention_heads.is_multiple_of(req.num_kv_heads) {
        return Err(AegisError::InvalidPlan(
            "g4 prefill attention: attention heads must be divisible by kv heads".into(),
        ));
    }
    let q_width = req.num_attention_heads * req.head_dim;
    let kv_width = req.num_kv_heads * req.head_dim;
    let expected = req.batch * q_width;
    if req.queries.len() != expected || out.len() != expected {
        return Err(AegisError::InvalidPlan(format!(
            "g4 prefill attention query/output shape mismatch: queries={} output={} expected={}",
            req.queries.len(),
            out.len(),
            expected
        )));
    }
    // Last query (row batch-1) attends up to position chunk_start+batch-1, so the
    // cache must hold at least chunk_start+batch positions.
    let required_kv = (req.chunk_start + req.batch) * kv_width;
    if req.keys.len() < required_kv || req.values.len() < required_kv {
        return Err(AegisError::InvalidPlan(format!(
            "g4 prefill attention KV too small: need {} keys={} values={}",
            required_kv,
            req.keys.len(),
            req.values.len()
        )));
    }

    let group = req.num_attention_heads / req.num_kv_heads;
    let head_dim = req.head_dim;
    let num_kv_heads = req.num_kv_heads;
    let q_heads = req.num_attention_heads;

    // Parallelize over (query, head) — one `[head_dim]` output slot per task,
    // exactly the decode path's parallel granularity but spanning the chunk.
    out.par_chunks_mut(head_dim)
        .enumerate()
        .for_each(|(slot, head_out)| {
            let i = slot / q_heads; // query row within the chunk
            let head = slot % q_heads; // attention head
            let kv_head = head / group;
            let pos = req.chunk_start + i;
            let seq_len = pos + 1;
            let window_start = if req.window_size > 0 && seq_len > req.window_size {
                seq_len - req.window_size
            } else {
                0
            };
            let q = &req.queries[(i * q_heads + head) * head_dim..(i * q_heads + head + 1) * head_dim];
            prefill_head_into(
                req.keys,
                req.values,
                window_start,
                seq_len,
                q,
                num_kv_heads,
                head_dim,
                kv_head,
                req.scale,
                head_out,
            );
        });
    Ok(())
}

/// Two-pass (max-subtract) softmax attention for one (query, head). Equivalent
/// to `attention_head_into`'s online softmax; kept separate so a per-query
/// scores buffer can be vectorized: pass 1 fills `scores[j] = scale*dot(q,k[j])`
/// (SIMD dot over head_dim) and tracks the max; pass 2 exps + sums; pass 3
/// SIMD-axpy's `weight*v[j]` and finally normalizes.
#[allow(clippy::too_many_arguments)]
fn prefill_head_into(
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
    out.fill(0.0);
    let n = seq_len - window_start;
    if n == 0 {
        return;
    }

    // Pass 1: scores + running max.
    let mut scores = vec![0.0_f32; n];
    let mut max_score = f32::NEG_INFINITY;
    for (s, pos) in scores.iter_mut().zip(window_start..seq_len) {
        let off = (pos * num_kv_heads + kv_head) * head_dim;
        let k = &keys[off..off + head_dim];
        let sc = simd::dot_f32(q, k) * scale;
        *s = sc;
        if sc > max_score {
            max_score = sc;
        }
    }

    // Pass 2: exp(score - max) + sum.
    let mut score_sum = 0.0_f32;
    for s in scores.iter_mut() {
        let w = (*s - max_score).exp();
        *s = w;
        score_sum += w;
    }

    // Pass 3: weighted sum of V.
    for (&weight, pos) in scores.iter().zip(window_start..seq_len) {
        let off = (pos * num_kv_heads + kv_head) * head_dim;
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

    // ── batched prefill attention: must equal looping the decode path ─────────

    /// Exact f32 → f16 → f32 round-trip (round-to-nearest-even) so the test KV
    /// cache carries genuine f16 precision (the cache is f16-origin), without
    /// pulling in the `half` dependency. Subnormals/Inf/NaN handled.
    fn f16_round_trip(x: f32) -> f32 {
        let bits = x.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let mut exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        let mant = bits & 0x007f_ffff;
        if ((bits >> 23) & 0xff) == 0xff {
            // Inf/NaN.
            let half = sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
            return f16_bits_to_f32(half as u16);
        }
        let half: u16 = if exp >= 0x1f {
            (sign | 0x7c00) as u16 // overflow → Inf
        } else if exp <= 0 {
            // Subnormal or zero.
            if exp < -10 {
                sign as u16
            } else {
                let m = mant | 0x0080_0000;
                let shift = (14 - exp) as u32;
                let mut h = m >> shift;
                // round to nearest even
                let rem = m & ((1 << shift) - 1);
                let halfway = 1u32 << (shift - 1);
                if rem > halfway || (rem == halfway && (h & 1) == 1) {
                    h += 1;
                }
                (sign | h) as u16
            }
        } else {
            let mut h = (exp << 10) as u32 | (mant >> 13);
            let rem = mant & 0x1fff;
            if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
                h += 1;
                if (h & 0x7c00) == 0x7c00 {
                    h = (sign | 0x7c00) | (h & 0x03ff); // carry into exp; clamp at Inf below
                }
                let _ = &mut exp;
            }
            (sign | h) as u16
        };
        f16_bits_to_f32(half)
    }

    fn f16_bits_to_f32(h: u16) -> f32 {
        let sign = ((h as u32) & 0x8000) << 16;
        let exp = ((h as u32) >> 10) & 0x1f;
        let mant = (h as u32) & 0x03ff;
        let bits = if exp == 0 {
            if mant == 0 {
                sign
            } else {
                // subnormal → normalize
                let mut e = -1i32;
                let mut m = mant;
                while (m & 0x0400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                m &= 0x03ff;
                let exp32 = (e + 1 + (127 - 15)) as u32;
                sign | (exp32 << 23) | (m << 13)
            }
        } else if exp == 0x1f {
            sign | 0x7f80_0000 | (mant << 13)
        } else {
            let exp32 = (exp as i32 - 15 + 127) as u32;
            sign | (exp32 << 23) | (mant << 13)
        };
        f32::from_bits(bits)
    }

    // Tiny xorshift RNG so the test is deterministic with no extra deps.
    fn next_rand(s: &mut u64) -> f32 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        // map to [-1, 1)
        ((*s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    /// Run the batched prefill path AND the decode-per-query reference over the
    /// same random Q + f16-precision K/V; assert cosine > 0.9999 element-for-
    /// element. Exercises GQA, a chunk_start offset, and the given window.
    fn check_prefill_matches_decode(
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        window_size: usize,
        chunk_start: usize,
        batch: usize,
        scale: f32,
        seed: u64,
    ) -> f64 {
        let mut s = seed | 1;
        let q_width = q_heads * head_dim;
        let kv_width = kv_heads * head_dim;
        let total_seq = chunk_start + batch;

        let queries: Vec<f32> = (0..batch * q_width).map(|_| next_rand(&mut s)).collect();
        // K/V are f16-origin: generate f32, round-trip through f16.
        let keys: Vec<f32> = (0..total_seq * kv_width)
            .map(|_| f16_round_trip(next_rand(&mut s)))
            .collect();
        let values: Vec<f32> = (0..total_seq * kv_width)
            .map(|_| f16_round_trip(next_rand(&mut s)))
            .collect();

        // Batched path.
        let mut batched = vec![0.0_f32; batch * q_width];
        g4_attention_prefill_into(
            G4PrefillAttnRequest {
                keys: &keys,
                values: &values,
                queries: &queries,
                chunk_start,
                batch,
                num_attention_heads: q_heads,
                num_kv_heads: kv_heads,
                head_dim,
                window_size,
                scale,
            },
            &mut batched,
        )
        .unwrap();

        // Reference: loop the decode path per query, exactly like the old code.
        let mut reference = vec![0.0_f32; batch * q_width];
        for i in 0..batch {
            let seq_len = chunk_start + i + 1;
            g4_attention_decode_into(
                G4DecodeAttnRequest {
                    keys: &keys,
                    values: &values,
                    seq_len,
                    query: &queries[i * q_width..(i + 1) * q_width],
                    num_attention_heads: q_heads,
                    num_kv_heads: kv_heads,
                    head_dim,
                    window_size,
                    scale,
                },
                &mut reference[i * q_width..(i + 1) * q_width],
            )
            .unwrap();
        }

        // Cosine over the full chunk output.
        let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for (&x, &y) in batched.iter().zip(reference.iter()) {
            d += x as f64 * y as f64;
            na += x as f64 * x as f64;
            nb += y as f64 * y as f64;
        }
        if na == 0.0 || nb == 0.0 {
            1.0
        } else {
            d / (na.sqrt() * nb.sqrt())
        }
    }

    #[test]
    fn prefill_matches_decode_full_causal() {
        // Full causal (window_size=0), GQA 8→2, head_dim 256, chunk starts at 0.
        let cos = check_prefill_matches_decode(8, 2, 256, 0, 0, 96, 1.0, 0xA11CE);
        assert!(cos > 0.9999, "full-causal cosine {cos}");
    }

    #[test]
    fn prefill_matches_decode_full_causal_chunk_offset() {
        // Full causal with a chunk_start > 0 (mid-prompt chunk).
        let cos = check_prefill_matches_decode(8, 2, 256, 0, 137, 64, 1.0, 0xBEEF1);
        assert!(cos > 0.9999, "full-causal chunk-offset cosine {cos}");
    }

    #[test]
    fn prefill_matches_decode_sliding_window() {
        // Sliding window=32, chunk_start=137 so EVERY query in the chunk has
        // pos > window (seq_len > window) → window_start is active for all of
        // them, exercising the boundary hard (the off-by-one would surface here).
        let cos = check_prefill_matches_decode(8, 2, 256, 32, 137, 64, 1.0, 0xC0FFEE);
        assert!(cos > 0.9999, "sliding-window cosine {cos}");
    }

    #[test]
    fn prefill_matches_decode_sliding_window_boundary_crossing() {
        // chunk_start=20, window=32, batch=40: early queries (pos<32) attend
        // from 0 (window inactive), later queries (pos≥32) start clipping —
        // the chunk straddles the seq_len == window_size boundary in BOTH the
        // inactive and active regimes, the most boundary-sensitive case.
        let cos = check_prefill_matches_decode(4, 1, 128, 32, 20, 40, 1.0, 0xD15EA5E);
        assert!(cos > 0.9999, "sliding-window boundary-crossing cosine {cos}");
    }

    #[test]
    fn prefill_matches_decode_scaled() {
        // Non-1.0 scale (the non-Gemma-4 1/sqrt(d) path), full causal.
        let scale = 1.0 / (256f32).sqrt();
        let cos = check_prefill_matches_decode(8, 2, 256, 0, 5, 80, scale, 0x5CA1E);
        assert!(cos > 0.9999, "scaled cosine {cos}");
    }

    /// Microbench: vectorized batched prefill vs the old decode-per-query loop,
    /// at seq_len ∈ {64, 256, 512}, head_dim 256, GQA 8→2. Ignored by default;
    /// run with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn prefill_attention_microbench() {
        use std::time::Instant;
        let q_heads = 8usize;
        let kv_heads = 2usize;
        let head_dim = 256usize;
        let q_width = q_heads * head_dim;
        let kv_width = kv_heads * head_dim;
        let scale = 1.0f32;

        for &seq_len in &[64usize, 256, 512] {
            // A full chunk of `seq_len` queries at chunk_start=0.
            let batch = seq_len;
            let mut s = 0x1234_5678u64;
            let queries: Vec<f32> = (0..batch * q_width).map(|_| next_rand(&mut s)).collect();
            let keys: Vec<f32> = (0..seq_len * kv_width)
                .map(|_| f16_round_trip(next_rand(&mut s)))
                .collect();
            let values: Vec<f32> = (0..seq_len * kv_width)
                .map(|_| f16_round_trip(next_rand(&mut s)))
                .collect();

            let iters = (2_000_000 / (seq_len * seq_len)).max(20);

            // Old path: loop decode per query.
            let mut out_old = vec![0.0_f32; batch * q_width];
            let run_old = |out: &mut [f32]| {
                for i in 0..batch {
                    let sl = i + 1;
                    g4_attention_decode_into(
                        G4DecodeAttnRequest {
                            keys: &keys,
                            values: &values,
                            seq_len: sl,
                            query: &queries[i * q_width..(i + 1) * q_width],
                            num_attention_heads: q_heads,
                            num_kv_heads: kv_heads,
                            head_dim,
                            window_size: 0,
                            scale,
                        },
                        &mut out[i * q_width..(i + 1) * q_width],
                    )
                    .unwrap();
                }
            };
            run_old(&mut out_old); // warm
            let t0 = Instant::now();
            for _ in 0..iters {
                run_old(&mut out_old);
            }
            let old_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

            // New path: batched prefill.
            let mut out_new = vec![0.0_f32; batch * q_width];
            let run_new = |out: &mut [f32]| {
                g4_attention_prefill_into(
                    G4PrefillAttnRequest {
                        keys: &keys,
                        values: &values,
                        queries: &queries,
                        chunk_start: 0,
                        batch,
                        num_attention_heads: q_heads,
                        num_kv_heads: kv_heads,
                        head_dim,
                        window_size: 0,
                        scale,
                    },
                    out,
                )
                .unwrap();
            };
            run_new(&mut out_new); // warm
            let t1 = Instant::now();
            for _ in 0..iters {
                run_new(&mut out_new);
            }
            let new_ms = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;

            eprintln!(
                "prefill_attention_microbench seq_len={seq_len:4} head_dim={head_dim}: \
                 old(decode-loop) = {old_ms:.4} ms, new(batched) = {new_ms:.4} ms, \
                 speedup = {:.2}x ({iters} iters)",
                old_ms / new_ms
            );
        }
    }
}

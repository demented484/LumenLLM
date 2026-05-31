//! Fused CPU INT4 (GPTQ-packed) GEMV with integer dot-products (W4A8).
//!
//! # Why this exists: testing "int4 experts beat nvfp4 experts on CPU"
//!
//! The experts-on-CPU decode path for Qwen3.6-35B-A3B streams ~540 MiB of packed
//! 4-bit expert weights from DRAM per decoded token. [`crate::nvfp4_gemv`] already
//! does this fused (dequant-in-register, weights never widened in RAM) and is
//! **bandwidth-bound at ~65.7 GB/s** on a large contiguous working set. The open
//! question this module settles: does an INT4 path with integer dot-products
//! (`vpdpbusd`, W4A8) beat NVFP4 *on the fragmented ~18 MiB per-layer chunks*,
//! where the kernel may be compute-bound rather than DRAM-bound?
//!
//! The hypothesis: NVFP4 must, per 16-element block, `vpermps`-gather 16 FP32
//! codes from a LUT, broadcast-multiply by the block scale, and FP32-FMA. INT4
//! with VNNI instead does a single `vpdpbusd` (4-wide int8 dot → int32) per 4
//! elements with NO per-element FP work, applying the (weight_scale × act_scale)
//! only once per 128-group. If the per-element FP dequant ALU is the limiter on
//! the small chunks, INT4 wins. If both are purely moving the same 4-bit bytes
//! from DRAM, INT4 cannot win (same bytes, same wall). **The microbench tests
//! decide; see [`tests`].**
//!
//! # GPTQ packing (confirmed against the real checkpoint)
//!
//! `packing_format = "auto_round:auto_gptq"`, `bits=4`, `group_size=128`,
//! `sym=true`. Per linear, on disk:
//!   * `qweight` I32 `[in/8, out]` — 8 int4 packed per int32 along the IN dim,
//!     low-nibble-first: in-position `8*j + i` is `(qweight[j,o] >> (4*i)) & 0xF`.
//!   * `scales`   F16 `[in/128, out]` — one scale per (group, output column).
//!   * `qzeros`   I32 `[in/128, out/8]` — packed int4 zero-points. Verified on
//!     the real weights: **every unpacked zero == 7** (symmetric AutoGPTQ uses
//!     `2^(bits-1)-1 = 7`, NOT 8). Dequant is `w = (nibble - 7) * scale`.
//!
//! This kernel stores the *transposed* row-major layout `W[out, in]` (each output
//! row's IN dim packed contiguously, low-nibble-first, 2 nibbles/byte) so the
//! GEMV parallelizes over output rows exactly like [`crate::nvfp4_gemv`] and reads
//! the identical 4-bit byte volume — the apples-to-apples comparison. Transposing
//! the on-disk `[in/8, out]` into `[out, in/2]` is a one-time load-time concern,
//! not a steady-state kernel concern, so it does not affect the throughput number.
//!
//! # The fused W4A8 dot (in-register, no RAM materialization)
//!
//! Per output row, per 128-element group:
//!   1. The activation is dynamically quantized to int8 once per token:
//!      `act_scale = max|x| / 127`, `q[k] = round(x[k] / act_scale)`. (Done once
//!      per token across ALL rows — it's an input property, not per-row.)
//!   2. The packed weight nibbles are unpacked to **unsigned** u8 in `[0,15]`
//!      (the raw nibble — no sign work) and fed to `vpdpbusd(u8_nibble, i8_act)`,
//!      accumulating `sum(nibble[k] * q[k])` as int32 over the group.
//!   3. The zero-point correction folds in via the identity
//!      `sum((nibble-7)*q) = sum(nibble*q) - 7*sum(q)`, where `sum(q)` over the
//!      group is precomputed once per token (an activation property). So the
//!      dequanted integer dot is `idot = raw_dot - 7*qsum_group`.
//!   4. Scale once per group: `acc += (idot as f32) * weight_scale * act_scale`,
//!      FP32-accumulating `acc` across the row's groups.
//!
//! The only weight RAM reads are the packed nibble bytes + the F16 group scales —
//! exactly the on-disk quant size, same as NVFP4. No per-element FP dequant.

use rayon::prelude::*;

/// Elements per quant group (GPTQ group_size).
pub const GROUP: usize = 128;
/// The symmetric AutoGPTQ zero-point for 4-bit (verified == 7 on the checkpoint).
pub const GPTQ_ZERO: i32 = 7;

// ── Packed weight view (transposed row-major W[out, in]) ──────────────────────

/// A row-major INT4 GPTQ weight `W[rows=out, cols=in]`: nibble-packed weights
/// (2 per byte, low-nibble-first along IN), per-(group,row) F16 scales. Borrows
/// the bytes — the kernel reads from here and nothing else, so weight DRAM
/// traffic == `packed.len() + scales.len()*2`.
#[derive(Clone, Copy)]
pub struct GptqWeights<'a> {
    pub rows: usize, // out features
    pub cols: usize, // in features (multiple of GROUP)
    /// `rows * cols/2` packed nibble bytes, row-major. Byte `b` of row `r` holds
    /// in-positions `2b` (low nibble) and `2b+1` (high nibble).
    pub packed: &'a [u8],
    /// `rows * cols/GROUP` F16 (bit pattern in u16) group scales, row-major:
    /// scale of row `r`, group `g` is `scales[r * (cols/GROUP) + g]`.
    pub scales: &'a [u16],
}

impl<'a> GptqWeights<'a> {
    pub fn new(rows: usize, cols: usize, packed: &'a [u8], scales: &'a [u16]) -> Self {
        assert_eq!(cols % GROUP, 0, "cols must be a multiple of {GROUP}");
        assert_eq!(packed.len(), rows * cols / 2, "packed byte count mismatch");
        assert_eq!(scales.len(), rows * cols / GROUP, "scale count mismatch");
        Self { rows, cols, packed, scales }
    }

    #[inline]
    fn packed_cols(&self) -> usize {
        self.cols / 2
    }
    #[inline]
    fn scale_cols(&self) -> usize {
        self.cols / GROUP
    }
}

/// Decode an F16 bit pattern to f32 (IEEE half → single). Small, branch-light;
/// only called once per group (cols/128 times per row), never per element.
#[inline(always)]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1f;
    let mant = (h & 0x3ff) as u32;
    let f = if exp == 0 {
        if mant == 0 {
            (sign as u32) << 31
        } else {
            // subnormal half → normalized single
            let mut e = -1i32;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            let exp32 = (127 - 15 + e) as u32;
            ((sign as u32) << 31) | (exp32 << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        ((sign as u32) << 31) | (0xff << 23) | (mant << 13)
    } else {
        let exp32 = (exp as i32 - 15 + 127) as u32;
        ((sign as u32) << 31) | (exp32 << 23) | (mant << 13)
    };
    f32::from_bits(f)
}

// ── Dynamic per-token int8 activation quantization ────────────────────────────

/// Per-token int8-quantized activation: int8 codes + the scalar `act_scale`
/// (= max|x|/127), plus the per-group `sum(q)` used by the zero-point correction.
pub struct QuantizedActivation {
    pub q: Vec<i8>,
    pub scale: f32,
    /// `sum(q[g*GROUP .. (g+1)*GROUP])` per group — the zero-point correction term.
    pub group_qsum: Vec<i32>,
}

/// Dynamically quantize one token's activation to int8 (symmetric, per-tensor):
/// `scale = max|x| / 127`, `q[k] = round(x[k] / scale)` clamped to `[-127, 127]`.
/// Also precomputes the per-group `sum(q)` (cols/GROUP entries) once.
pub fn quantize_activation(x: &[f32]) -> QuantizedActivation {
    assert_eq!(x.len() % GROUP, 0, "activation length must be a multiple of {GROUP}");
    let amax = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    // Guard the all-zero token: scale 0 → all codes 0 (and qsum 0), exact.
    let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
    let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
    let mut q = vec![0i8; x.len()];
    let ngroups = x.len() / GROUP;
    let mut group_qsum = vec![0i32; ngroups];
    // `g` indexes the group's slice of `q`/`x` (offset arithmetic) and the
    // `group_qsum` slot — the indexed form is clearest for the offset stride.
    #[allow(clippy::needless_range_loop)]
    for g in 0..ngroups {
        let mut s = 0i32;
        for k in 0..GROUP {
            let idx = g * GROUP + k;
            let v = (x[idx] * inv).round();
            let qi = v.clamp(-127.0, 127.0) as i32;
            q[idx] = qi as i8;
            s += qi;
        }
        group_qsum[g] = s;
    }
    QuantizedActivation { q, scale, group_qsum }
}

// ── Scalar references ─────────────────────────────────────────────────────────

/// W4A16 reference (the QUALITY oracle): full GPTQ dequant of each weight against
/// the **raw FP32** activation, accumulated in f64. This is the highest-fidelity
/// path (FP activation, like NVFP4's CPU kernel) and the baseline the W4A8 path's
/// quality cost is measured against. `y[r] = sum_c (nibble(r,c)-7)*scale(r,g)*x[c]`.
pub fn gemv_reference_w4a16(w: &GptqWeights, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), w.cols);
    assert_eq!(y.len(), w.rows);
    let pc = w.packed_cols();
    let sc = w.scale_cols();
    #[allow(clippy::needless_range_loop)]
    for r in 0..w.rows {
        let prow = &w.packed[r * pc..(r + 1) * pc];
        let srow = &w.scales[r * sc..(r + 1) * sc];
        let mut acc = 0.0f64;
        for g in 0..sc {
            let gscale = f16_to_f32(srow[g]) as f64;
            let mut gacc = 0.0f64;
            for kk in 0..GROUP {
                let col = g * GROUP + kk;
                let byte = prow[col / 2];
                let nib = if col & 1 == 0 { byte & 0x0f } else { byte >> 4 };
                let wv = (nib as i32 - GPTQ_ZERO) as f64;
                gacc += wv * x[col] as f64;
            }
            acc += gacc * gscale;
        }
        y[r] = acc as f32;
    }
}

/// W4A8 reference (the integer-path oracle): the activation is first quantized to
/// int8, then the dot is done in integers with the zero-point correction, scaled
/// per group. This defines the FUSED kernel's exact semantics (the fused AVX path
/// must match THIS, bit-close), and its delta vs [`gemv_reference_w4a16`] is the
/// quality cost of int8 activations.
pub fn gemv_reference_w4a8(w: &GptqWeights, qa: &QuantizedActivation, y: &mut [f32]) {
    assert_eq!(qa.q.len(), w.cols);
    assert_eq!(y.len(), w.rows);
    let pc = w.packed_cols();
    let sc = w.scale_cols();
    let act_scale = qa.scale;
    #[allow(clippy::needless_range_loop)]
    for r in 0..w.rows {
        let prow = &w.packed[r * pc..(r + 1) * pc];
        let srow = &w.scales[r * sc..(r + 1) * sc];
        let mut acc = 0.0f32;
        for g in 0..sc {
            let gscale = f16_to_f32(srow[g]);
            let mut raw = 0i32; // sum(nibble * q)
            for kk in 0..GROUP {
                let col = g * GROUP + kk;
                let byte = prow[col / 2];
                let nib = if col & 1 == 0 { byte & 0x0f } else { byte >> 4 } as i32;
                raw += nib * qa.q[col] as i32;
            }
            // sum((nibble-7)*q) = raw - 7*sum(q)
            let idot = raw - GPTQ_ZERO * qa.group_qsum[g];
            acc += idot as f32 * gscale * act_scale;
        }
        y[r] = acc;
    }
}

// ── Fused W4A8 dot for ONE row (the in-register core) ─────────────────────────

/// Fused INT4 W4A8 dequant-dot of one weight row against one int8 activation.
/// Returns `sum_c (nibble(c)-7)*weight_scale(group(c)) * q[c] * act_scale`.
/// Reads only `packed_row` (cols/2 bytes) + `scale_row` (cols/128 F16) for the
/// weights. Dispatches to AVX-512 VNNI when present, else a scalar fused loop.
#[inline]
fn fused_row_dot_w4a8(packed_row: &[u8], scale_row: &[u16], qa: &QuantizedActivation) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx512vbmi")
        {
            // SAFETY: gated on runtime avx512f+bw+vnni+vbmi; all slices sized to
            // `cols` (multiple of GROUP=128) and only indexed within them.
            return unsafe { fused_row_dot_avx512_vnni(packed_row, scale_row, qa) };
        }
    }
    fused_row_dot_w4a8_scalar(packed_row, scale_row, qa)
}

/// Portable fused scalar fallback (same math as the VNNI path, no widening).
#[inline]
fn fused_row_dot_w4a8_scalar(
    packed_row: &[u8],
    scale_row: &[u16],
    qa: &QuantizedActivation,
) -> f32 {
    let q = &qa.q;
    let act_scale = qa.scale;
    let mut acc = 0.0f32;
    for (g, &sh) in scale_row.iter().enumerate() {
        let gscale = f16_to_f32(sh);
        let pbase = g * (GROUP / 2);
        let xbase = g * GROUP;
        let mut raw = 0i32;
        for j in 0..(GROUP / 2) {
            let byte = packed_row[pbase + j];
            raw += (byte & 0x0f) as i32 * q[xbase + 2 * j] as i32;
            raw += (byte >> 4) as i32 * q[xbase + 2 * j + 1] as i32;
        }
        let idot = raw - GPTQ_ZERO * qa.group_qsum[g];
        acc += idot as f32 * gscale * act_scale;
    }
    acc
}

/// AVX-512 VNNI fused INT4 W4A8 dequant-dot of one row. The 4-bit unpack →
/// unsigned-u8 nibbles + `vpdpbusd(u8_nibble, i8_act)` int32 accumulate happen
/// entirely in zmm registers; the only weight RAM reads are the 64 packed bytes
/// (128 nibbles) + 1 F16 scale per group. NO per-element FP dequant — the scale
/// is applied once per 128-group, the zero-point correction once per group.
///
/// Per group (128 elements = 64 packed bytes): load the 64 bytes, split into low
/// nibbles (`& 0x0f`) and high nibbles (`>> 4`) — both unsigned u8 in `[0,15]`.
/// The 128 activation int8 are interleaved as `[lo0,hi0,lo1,hi1,...]` (byte j →
/// in-positions 2j, 2j+1), so we de-interleave the int8 to a low stream (even
/// positions) and a high stream (odd positions) once per group and `vpdpbusd`
/// each nibble stream against its matching int8 stream. Accumulating in a single
/// int32 zmm gives `sum(nibble*q)`; subtract `7*qsum`, scale by
/// `weight_scale * act_scale`, FP32-add into the row accumulator.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni,avx512vbmi")]
fn fused_row_dot_avx512_vnni(
    packed_row: &[u8],
    scale_row: &[u16],
    qa: &QuantizedActivation,
) -> f32 {
    use std::arch::x86_64::*;

    // SAFETY: caller guarantees avx512f+bw+vnni at runtime; `scale_row` has one
    // F16 per 128-element group, `packed_row` has 64 bytes/group, and `qa.q` has
    // 128 i8/group — every access below stays within those bounds.
    unsafe {
        let q = qa.q.as_ptr();
        let act_scale = qa.scale;
        let low_mask = _mm512_set1_epi8(0x0f);

        // De-interleave index vectors (built ONCE, hoisted out of the group loop).
        // The 128 activation int8 for a group live in two 64-byte registers q0,q1
        // (q0 = x[0..64], q1 = x[64..128]). We want a 64-lane "even" register
        // holding x[0,2,4,...,126] (pairs with the low nibbles) and an "odd"
        // register holding x[1,3,...,127] (pairs with the high nibbles).
        // `permutex2var_epi8` gathers from a 128-byte (two-register) source by a
        // 7-bit per-lane index: index `i` selects byte `i` of the concatenation
        // [q0(0..64) , q1(0..64)]. So even_idx[i] = 2i, odd_idx[i] = 2i+1.
        let mut even_b = [0i8; 64];
        let mut odd_b = [0i8; 64];
        for i in 0..64 {
            even_b[i] = (2 * i) as i8; // 0,2,...,126
            odd_b[i] = (2 * i + 1) as i8; // 1,3,...,127
        }
        let even_idx = _mm512_loadu_si512(even_b.as_ptr() as *const _);
        let odd_idx = _mm512_loadu_si512(odd_b.as_ptr() as *const _);

        let mut acc = 0.0f32;
        let ngroups = scale_row.len();
        #[allow(clippy::needless_range_loop)]
        for g in 0..ngroups {
            // --- load 64 packed weight bytes (128 nibbles) for this group ------
            let pbase = g * (GROUP / 2);
            let wbytes = _mm512_loadu_si512(packed_row.as_ptr().add(pbase) as *const _);
            // low nibbles (in-positions 0,2,4,...) and high (1,3,5,...), each u8.
            let w_lo = _mm512_and_si512(wbytes, low_mask); // nibble at even in-pos
            let w_hi = _mm512_and_si512(_mm512_srli_epi16(wbytes, 4), low_mask); // odd in-pos

            // --- load 128 int8 activations and split even/odd positions --------
            // w_lo[b] pairs with x[2b], w_hi[b] pairs with x[2b+1].
            let xbase = g * GROUP;
            let q0 = _mm512_loadu_si512(q.add(xbase) as *const _); // x[0..64]
            let q1 = _mm512_loadu_si512(q.add(xbase + 64) as *const _); // x[64..128]
            // One permutex2var across the 128-byte (q0:q1) source per stream.
            let q_even = _mm512_permutex2var_epi8(q0, even_idx, q1);
            let q_odd = _mm512_permutex2var_epi8(q0, odd_idx, q1);

            // --- VNNI: int32 accumulate sum(nibble * q) over the 128 elements --
            // vpdpbusd(unsigned u8, signed i8): w_lo/w_hi are u8 [0,15], q_* i8.
            let mut idot32 = _mm512_setzero_si512();
            idot32 = _mm512_dpbusd_epi32(idot32, w_lo, q_even);
            idot32 = _mm512_dpbusd_epi32(idot32, w_hi, q_odd);
            // Horizontal-reduce the 16 int32 lanes → scalar sum(nibble*q).
            let raw = _mm512_reduce_add_epi32(idot32);

            // --- zero-point correction + per-group scale -----------------------
            let idot = raw - GPTQ_ZERO * qa.group_qsum[g];
            let gscale = f16_to_f32(scale_row[g]);
            acc += idot as f32 * gscale * act_scale;
        }
        acc
    }
}

// ── Public fused GEMV / GEMM (multi-threaded over rows) ───────────────────────

/// Fused INT4 W4A8 GEMV, M=1: `y[rows] = W[rows,cols] · x[cols]`, threaded over
/// output rows (rayon). The activation is int8-quantized ONCE (shared across all
/// rows); each row reads its packed bytes from DRAM once and dots in-register.
pub fn gemv_into(w: &GptqWeights, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), w.cols, "input length must equal cols");
    assert_eq!(y.len(), w.rows, "output length must equal rows");
    let qa = quantize_activation(x);
    gemv_into_prequantized(w, &qa, y);
}

/// Fused INT4 W4A8 GEMV against an already-int8-quantized activation — the form
/// the real path uses (the activation is quantized once per token, then reused
/// across the gate/up projections that share the same input hidden).
pub fn gemv_into_prequantized(w: &GptqWeights, qa: &QuantizedActivation, y: &mut [f32]) {
    assert_eq!(qa.q.len(), w.cols, "activation length must equal cols");
    assert_eq!(y.len(), w.rows, "output length must equal rows");
    let pc = w.packed_cols();
    let sc = w.scale_cols();
    let packed = w.packed;
    let scales = w.scales;
    y.par_iter_mut().enumerate().for_each(|(r, slot)| {
        let prow = &packed[r * pc..(r + 1) * pc];
        let srow = &scales[r * sc..(r + 1) * sc];
        *slot = fused_row_dot_w4a8(prow, srow, qa);
    });
}

/// Fused INT4 W4A8 GEMM for small M: `Y[m, rows] = W · X[m, cols]`, X/Y token-
/// major. Each weight row is read once and dotted against all M tokens (row bytes
/// stay hot across the inner token loop) — weight DRAM traffic independent of M.
pub fn gemm_into(w: &GptqWeights, x: &[f32], m: usize, y: &mut [f32]) {
    if m == 0 {
        return;
    }
    if m == 1 {
        return gemv_into(w, x, y);
    }
    assert_eq!(x.len(), m * w.cols, "input length must equal m*cols");
    assert_eq!(y.len(), m * w.rows, "output length must equal m*rows");
    let qas: Vec<QuantizedActivation> =
        (0..m).map(|t| quantize_activation(&x[t * w.cols..(t + 1) * w.cols])).collect();
    let pc = w.packed_cols();
    let sc = w.scale_cols();
    let rows = w.rows;
    let packed = w.packed;
    let scales = w.scales;
    let row_cols: Vec<Vec<f32>> = (0..rows)
        .into_par_iter()
        .map(|r| {
            let prow = &packed[r * pc..(r + 1) * pc];
            let srow = &scales[r * sc..(r + 1) * sc];
            let mut out = vec![0.0f32; m];
            for (t, slot) in out.iter_mut().enumerate() {
                *slot = fused_row_dot_w4a8(prow, srow, &qas[t]);
            }
            out
        })
        .collect();
    for (r, col) in row_cols.iter().enumerate() {
        for (t, &v) in col.iter().enumerate() {
            y[t * rows + r] = v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nvfp4_gemv::{self, PackedWeights};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use std::time::Instant;

    const SEED: u64 = 0x4751_5054_5134_0000; // "GPTQ4\0\0"

    // ── helpers ───────────────────────────────────────────────────────────────

    /// Build a random GPTQ-packed weight `[rows=out, cols=in]` (transposed
    /// row-major layout) whose dequanted values mimic real LLM weights: per
    /// (group,row) scale ~ small positive, nibbles distributed around the zero
    /// point 7 so `(nibble-7)*scale` is roughly N(0, scale*few). Returns
    /// (packed bytes, F16 scale bits).
    fn random_gptq(rng: &mut SmallRng, rows: usize, cols: usize) -> (Vec<u8>, Vec<u16>) {
        let packed: Vec<u8> = (0..rows * cols / 2).map(|_| rng.random::<u8>()).collect();
        let ngroups = cols / GROUP;
        let scales: Vec<u16> = (0..rows * ngroups)
            .map(|_| {
                // small positive scale ~ [0.003, 0.02], like the real checkpoint.
                let s = 0.003f32 + rng.random::<f32>() * 0.017;
                half_from_f32(s)
            })
            .collect();
        (packed, scales)
    }

    /// f32 → F16 bit pattern (round-to-nearest-even, good enough for test scales).
    fn half_from_f32(x: f32) -> u16 {
        let bits = x.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        let mant = bits & 0x7f_ffff;
        if exp <= 0 {
            return sign; // flush tiny to zero (test scales are well above this)
        }
        if exp >= 0x1f {
            return sign | 0x7c00;
        }
        // round mantissa to 10 bits
        let m = mant >> 13;
        let round = (mant >> 12) & 1;
        let h = sign | ((exp as u16) << 10) | (m as u16);
        h.wrapping_add(round as u16)
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let (mut d, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b.iter()) {
            d += x as f64 * y as f64;
            na += (x as f64).powi(2);
            nb += (y as f64).powi(2);
        }
        if na == 0.0 || nb == 0.0 {
            return 1.0;
        }
        d / (na.sqrt() * nb.sqrt())
    }

    fn max_rel_err(reference: &[f32], got: &[f32]) -> f64 {
        let n = reference.len().max(1) as f64;
        let rms = (reference.iter().map(|&r| (r as f64).powi(2)).sum::<f64>() / n).sqrt();
        let denom = rms.max(1e-12);
        reference
            .iter()
            .zip(got.iter())
            .map(|(&r, &g)| (r as f64 - g as f64).abs() / denom)
            .fold(0.0f64, f64::max)
    }

    /// Qwen3.6-35B-A3B expert shapes (rows=out, cols=in) + block-edge shapes.
    const EXPERT_SHAPES: &[(usize, usize, &str)] = &[
        (512, 2048, "gate/up [out512 x in2048]"),
        (2048, 512, "down [out2048 x in512]"),
        (4, 128, "single-group rows [4x128]"),
        (16, 256, "two-group [16x256]"),
        (128, 1024, "wide [128x1024]"),
    ];

    // ── correctness ─────────────────────────────────────────────────────────

    /// f16_to_f32 must invert our test f16 encoder on representative scales.
    #[test]
    fn f16_roundtrip() {
        for &v in &[0.003f32, 0.01, 0.0177, 0.5, 1.0, 2.5, 0.04477] {
            let h = half_from_f32(v);
            let back = f16_to_f32(h);
            assert!((back - v).abs() < 1e-3 * v.max(1e-3), "f16 roundtrip {v} -> {back}");
        }
    }

    /// FUSED W4A8 (AVX-VNNI or scalar) must bit-closely match the W4A8 reference
    /// (the integer-path oracle) — same activation int8, same zero correction.
    #[test]
    fn fused_matches_w4a8_reference() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        for &(rows, cols, label) in EXPERT_SHAPES {
            let (packed, scales) = random_gptq(&mut rng, rows, cols);
            let w = GptqWeights::new(rows, cols, &packed, &scales);
            let x: Vec<f32> = (0..cols).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
            let qa = quantize_activation(&x);

            let mut y_ref = vec![0.0f32; rows];
            let mut y_fast = vec![0.0f32; rows];
            gemv_reference_w4a8(&w, &qa, &mut y_ref);
            gemv_into_prequantized(&w, &qa, &mut y_fast);

            let cos = cosine(&y_ref, &y_fast);
            let rel = max_rel_err(&y_ref, &y_fast);
            assert!(cos > 0.99999, "{label}: fused-vs-w4a8ref cosine {cos}");
            assert!(rel < 1e-4, "{label}: fused-vs-w4a8ref max-rel-err {rel}");
            eprintln!("{label}: fused vs W4A8-ref  cos={cos:.9} rel={rel:.2e}");
        }
    }

    /// Scalar fused path must also match the W4A8 reference (pins the non-AVX
    /// fallback and proves the VNNI path isn't the only correct one).
    #[test]
    fn scalar_fused_matches_w4a8_reference() {
        let mut rng = SmallRng::seed_from_u64(SEED ^ 0xABCD);
        for &(rows, cols, label) in EXPERT_SHAPES {
            let (packed, scales) = random_gptq(&mut rng, rows, cols);
            let w = GptqWeights::new(rows, cols, &packed, &scales);
            let x: Vec<f32> = (0..cols).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
            let qa = quantize_activation(&x);
            let pc = cols / 2;
            let sc = cols / GROUP;

            let mut y_ref = vec![0.0f32; rows];
            gemv_reference_w4a8(&w, &qa, &mut y_ref);
            let mut y_scalar = vec![0.0f32; rows];
            for r in 0..rows {
                let prow = &packed[r * pc..(r + 1) * pc];
                let srow = &scales[r * sc..(r + 1) * sc];
                y_scalar[r] = fused_row_dot_w4a8_scalar(prow, srow, &qa);
            }
            let cos = cosine(&y_ref, &y_scalar);
            assert!(cos > 0.99999, "{label}: scalar-vs-w4a8ref cosine {cos}");
        }
    }

    /// THE QUALITY-COST MEASUREMENT: W4A8 (int8 activation) vs W4A16 (FP32
    /// activation, the high-fidelity oracle NVFP4's CPU kernel also uses). Reports
    /// cosine + max-rel-err so we know the accuracy we trade for the integer dot.
    /// Per-token symmetric int8 (max|x|/127) on a single dot loses ~one part in a
    /// few hundred; we assert it stays well within a sane bound and PRINT it.
    #[test]
    fn w4a8_quality_cost_vs_w4a16() {
        let mut rng = SmallRng::seed_from_u64(SEED ^ 0xA8A8);
        let mut worst_cos = 1.0f64;
        let mut worst_rel = 0.0f64;
        for &(rows, cols, label) in EXPERT_SHAPES {
            let (packed, scales) = random_gptq(&mut rng, rows, cols);
            let w = GptqWeights::new(rows, cols, &packed, &scales);
            // realistic activation magnitudes: roughly unit-scale with outliers.
            let x: Vec<f32> = (0..cols)
                .map(|_| {
                    let base = rng.random::<f32>() * 2.0 - 1.0;
                    base * if rng.random::<f32>() < 0.02 { 6.0 } else { 1.0 }
                })
                .collect();

            let mut y_a16 = vec![0.0f32; rows];
            gemv_reference_w4a16(&w, &x, &mut y_a16);
            let qa = quantize_activation(&x);
            let mut y_a8 = vec![0.0f32; rows];
            gemv_reference_w4a8(&w, &qa, &mut y_a8);

            let cos = cosine(&y_a16, &y_a8);
            let rel = max_rel_err(&y_a16, &y_a8);
            worst_cos = worst_cos.min(cos);
            worst_rel = worst_rel.max(rel);
            eprintln!("{label}: W4A8 vs W4A16  cos={cos:.6}  max_rel_err={rel:.3e}");
        }
        eprintln!(
            "── W4A8 quality cost (worst over shapes): cosine={worst_cos:.6}  max_rel_err={worst_rel:.3e} ──"
        );
        // int8 per-token symmetric activation on a ~K-term dot: cosine should stay
        // very high (>0.999); this is the real, quantitative tradeoff vs FP act.
        assert!(worst_cos > 0.999, "W4A8 quality regressed: cosine {worst_cos}");
    }

    /// Batched (small-M) GEMM must match per-token GEMV row-for-row.
    #[test]
    fn gemm_matches_per_token_gemv() {
        let mut rng = SmallRng::seed_from_u64(SEED ^ 0x1234);
        let (rows, cols) = (512usize, 2048usize);
        let (packed, scales) = random_gptq(&mut rng, rows, cols);
        let w = GptqWeights::new(rows, cols, &packed, &scales);
        for &m in &[2usize, 4, 8] {
            let x: Vec<f32> = (0..m * cols).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
            let mut y_batch = vec![0.0f32; m * rows];
            gemm_into(&w, &x, m, &mut y_batch);
            let mut y_expected = vec![0.0f32; m * rows];
            for t in 0..m {
                let mut yt = vec![0.0f32; rows];
                gemv_into(&w, &x[t * cols..(t + 1) * cols], &mut yt);
                y_expected[t * rows..(t + 1) * rows].copy_from_slice(&yt);
            }
            for i in 0..y_batch.len() {
                assert!(
                    (y_batch[i] - y_expected[i]).abs() < 1e-4 * (1.0 + y_expected[i].abs()),
                    "m={m} i={i}: batch={} expected={}",
                    y_batch[i],
                    y_expected[i]
                );
            }
        }
    }

    /// Sanity: all-zero activation → zero output.
    #[test]
    fn zero_activation_gives_zero() {
        let mut rng = SmallRng::seed_from_u64(SEED ^ 0x0);
        let (rows, cols) = (8usize, 256usize);
        let (packed, scales) = random_gptq(&mut rng, rows, cols);
        let w = GptqWeights::new(rows, cols, &packed, &scales);
        let x = vec![0.0f32; cols];
        let mut y = vec![1.0f32; rows];
        gemv_into(&w, &x, &mut y);
        assert!(y.iter().all(|&v| v == 0.0), "expected all-zero output");
    }

    // ── COMPARATIVE MICROBENCH (the key deliverable) ──────────────────────────
    //
    // Compares INT4 (W4A8 VNNI) against the NVFP4 fused kernel at the SAME byte
    // sizes, in TWO regimes:
    //   (a) FULL active set per token (~540 MiB across 8 experts × 3 proj × 40
    //       layers) — the large contiguous streaming regime (NVFP4 ~65.7 GB/s).
    //   (b) PER-LAYER chunk (~18 MiB = one layer's 8 active experts) — the
    //       fragmented integrated regime (NVFP4 ~23 GB/s). This is where compute
    //       could matter and where INT4 must prove itself.
    //
    // Both kernels read the identical 4-bit byte volume + scale bytes (INT4 scale
    // is F16=2 bytes/group/128 elems; NVFP4 is 1 byte/16 elems = 8 bytes/128).
    // Run with: cargo test -p aegisllm-cpu --release int4_vs_nvfp4 -- --ignored --nocapture

    const HIDDEN: usize = 2048;
    const INTER: usize = 512;
    const EXPERTS_PER_TOK: usize = 8;
    const MOE_LAYERS: usize = 40;

    struct Int4Expert {
        gate: (Vec<u8>, Vec<u16>),
        up: (Vec<u8>, Vec<u16>),
        down: (Vec<u8>, Vec<u16>),
    }
    struct Nvfp4Expert {
        gate: (Vec<u8>, Vec<u8>),
        up: (Vec<u8>, Vec<u8>),
        down: (Vec<u8>, Vec<u8>),
    }

    fn build_nvfp4(rng: &mut SmallRng, rows: usize, cols: usize) -> (Vec<u8>, Vec<u8>) {
        let packed: Vec<u8> = (0..rows * cols / 2).map(|_| rng.random::<u8>()).collect();
        let scales: Vec<u8> =
            (0..rows * cols / 16).map(|_| 0x10u8 + (rng.random::<u8>() % 0x50)).collect();
        (packed, scales)
    }
    fn build_int4(rng: &mut SmallRng, rows: usize, cols: usize) -> (Vec<u8>, Vec<u16>) {
        random_gptq(rng, rows, cols)
    }

    fn int4_bytes_per_expert() -> usize {
        // packed nibbles (cols/2 per row) + F16 scales (cols/128 per row * 2 bytes)
        let gu = INTER * (HIDDEN / 2) + INTER * (HIDDEN / GROUP) * 2;
        let dn = HIDDEN * (INTER / 2) + HIDDEN * (INTER / GROUP) * 2;
        gu * 2 + dn
    }
    fn nvfp4_bytes_per_expert() -> usize {
        let gu = INTER * (HIDDEN / 2) + INTER * (HIDDEN / 16);
        let dn = HIDDEN * (INTER / 2) + HIDDEN * (INTER / 16);
        gu * 2 + dn
    }

    /// Run INT4 over a >LLC pool of distinct experts, `calls` expert-calls total.
    fn run_int4(pool: &[Int4Expert], xh: &[f32], xi: &[f32], calls: usize) -> (f64, f32) {
        let mut yg = vec![0.0f32; INTER];
        let mut yu = vec![0.0f32; INTER];
        let mut yh = vec![0.0f32; HIDDEN];
        let qa_h = quantize_activation(xh);
        let qa_i = quantize_activation(xi);
        // warm
        for i in 0..pool.len().min(40) {
            let e = &pool[i];
            let g = GptqWeights::new(INTER, HIDDEN, &e.gate.0, &e.gate.1);
            gemv_into_prequantized(&g, &qa_h, &mut yg);
        }
        let mut sink = 0.0f32;
        let t0 = Instant::now();
        for i in 0..calls {
            let e = &pool[i % pool.len()];
            let g = GptqWeights::new(INTER, HIDDEN, &e.gate.0, &e.gate.1);
            let u = GptqWeights::new(INTER, HIDDEN, &e.up.0, &e.up.1);
            let d = GptqWeights::new(HIDDEN, INTER, &e.down.0, &e.down.1);
            gemv_into_prequantized(&g, &qa_h, &mut yg);
            gemv_into_prequantized(&u, &qa_h, &mut yu);
            gemv_into_prequantized(&d, &qa_i, &mut yh);
            sink += yh[0] + yg[0] + yu[0];
        }
        (t0.elapsed().as_secs_f64(), sink)
    }

    fn run_nvfp4(pool: &[Nvfp4Expert], xh: &[f32], xi: &[f32], calls: usize) -> (f64, f32) {
        let mut yg = vec![0.0f32; INTER];
        let mut yu = vec![0.0f32; INTER];
        let mut yh = vec![0.0f32; HIDDEN];
        const GSCALE: f32 = 0.5;
        for i in 0..pool.len().min(40) {
            let e = &pool[i];
            let g = PackedWeights::new(INTER, HIDDEN, &e.gate.0, &e.gate.1, GSCALE);
            nvfp4_gemv::gemv_into(&g, xh, &mut yg);
        }
        let mut sink = 0.0f32;
        let t0 = Instant::now();
        for i in 0..calls {
            let e = &pool[i % pool.len()];
            let g = PackedWeights::new(INTER, HIDDEN, &e.gate.0, &e.gate.1, GSCALE);
            let u = PackedWeights::new(INTER, HIDDEN, &e.up.0, &e.up.1, GSCALE);
            let d = PackedWeights::new(HIDDEN, INTER, &e.down.0, &e.down.1, GSCALE);
            nvfp4_gemv::gemv_into(&g, xh, &mut yg);
            nvfp4_gemv::gemv_into(&u, xh, &mut yu);
            nvfp4_gemv::gemv_into(&d, xi, &mut yh);
            sink += yh[0] + yg[0] + yu[0];
        }
        (t0.elapsed().as_secs_f64(), sink)
    }

    #[test]
    #[ignore]
    fn int4_vs_nvfp4_comparative_microbench() {
        let mut rng = SmallRng::seed_from_u64(SEED ^ 0xC0DE);

        // Build a >LLC pool of DISTINCT experts (cold-DRAM streaming, the honest
        // decode number). One INT4 expert ≈ 1.6 MiB; 64 ≈ 105 MiB >> 32 MiB LLC.
        const POOL: usize = 64;
        let int4_pool: Vec<Int4Expert> = (0..POOL)
            .map(|_| Int4Expert {
                gate: build_int4(&mut rng, INTER, HIDDEN),
                up: build_int4(&mut rng, INTER, HIDDEN),
                down: build_int4(&mut rng, HIDDEN, INTER),
            })
            .collect();
        let nvfp4_pool: Vec<Nvfp4Expert> = (0..POOL)
            .map(|_| Nvfp4Expert {
                gate: build_nvfp4(&mut rng, INTER, HIDDEN),
                up: build_nvfp4(&mut rng, INTER, HIDDEN),
                down: build_nvfp4(&mut rng, HIDDEN, INTER),
            })
            .collect();

        let xh: Vec<f32> = (0..HIDDEN).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let xi: Vec<f32> = (0..INTER).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();

        let int4_bpe = int4_bytes_per_expert();
        let nvfp4_bpe = nvfp4_bytes_per_expert();
        // 2 FLOP per weight element (mul + add), same element count for both.
        let elems_per_expert = INTER * HIDDEN * 2 + HIDDEN * INTER;
        let flops_per_expert = elems_per_expert * 2;
        let calls_per_token = EXPERTS_PER_TOK * MOE_LAYERS; // 320

        // ----- Regime (a): FULL active set per token -----
        // Many tokens' worth of calls to amortize the timer & stay steady-state.
        const TOKENS_FULL: usize = 30;
        let calls_full = TOKENS_FULL * calls_per_token;
        let (t_i4, s_i4) = run_int4(&int4_pool, &xh, &xi, calls_full);
        let (t_nv, s_nv) = run_nvfp4(&nvfp4_pool, &xh, &xi, calls_full);

        let i4_gbps = (int4_bpe * calls_full) as f64 / t_i4 / 1e9;
        let nv_gbps = (nvfp4_bpe * calls_full) as f64 / t_nv / 1e9;
        let i4_gflops = (flops_per_expert * calls_full) as f64 / t_i4 / 1e9;
        let nv_gflops = (flops_per_expert * calls_full) as f64 / t_nv / 1e9;
        let i4_tok_ms = t_i4 / TOKENS_FULL as f64 * 1e3;
        let nv_tok_ms = t_nv / TOKENS_FULL as f64 * 1e3;

        eprintln!("════════════════════════════════════════════════════════════");
        eprintln!(" INT4 (W4A8 VNNI) vs NVFP4 fused CPU GEMV — comparative bench");
        eprintln!(" hw: Ryzen 5 7500F (Zen4, 6c/12t, AVX-512 VNNI), Qwen3.6-35B-A3B shapes");
        eprintln!("════════════════════════════════════════════════════════════");
        eprintln!("(a) FULL active set / token  ({calls_per_token} expert-calls/tok, {TOKENS_FULL} tok)");
        eprintln!("    INT4  : {i4_gbps:6.1} GB/s  {i4_gflops:7.1} GFLOP/s  {i4_tok_ms:6.2} ms/tok  (bytes/exp={int4_bpe})  sink={s_i4:.2}");
        eprintln!("    NVFP4 : {nv_gbps:6.1} GB/s  {nv_gflops:7.1} GFLOP/s  {nv_tok_ms:6.2} ms/tok  (bytes/exp={nvfp4_bpe})  sink={s_nv:.2}");
        eprintln!("    speedup INT4/NVFP4 (by time): {:.2}×", t_nv / t_i4);

        // ----- Regime (b): PER-LAYER chunk (~18 MiB) -----
        // One layer = EXPERTS_PER_TOK experts × 3 proj = 24 GEMVs ≈ 8×1.6=12.8MiB
        // INT4 / ~14.2 MiB NVFP4. Time MANY single-layer chunks back-to-back but
        // each from a DISTINCT pool slot so it's a cold ~chunk-sized working set.
        const LAYER_REPS: usize = 30 * MOE_LAYERS; // same total work as (a)
        let chunk_calls = EXPERTS_PER_TOK; // one layer's experts (gate+up+down each)

        // Measure: per "layer", run EXPERTS_PER_TOK experts then move to the next
        // pool slots — the kernel sees ~18 MiB of fresh weights per measured unit.
        let bench_chunk_int4 = |reps: usize| -> (f64, f32) {
            let mut yg = vec![0.0f32; INTER];
            let mut yu = vec![0.0f32; INTER];
            let mut yh = vec![0.0f32; HIDDEN];
            let qa_h = quantize_activation(&xh);
            let qa_i = quantize_activation(&xi);
            let mut sink = 0.0f32;
            let t0 = Instant::now();
            let mut idx = 0usize;
            for _ in 0..reps {
                for _ in 0..chunk_calls {
                    let e = &int4_pool[idx % POOL];
                    idx += 1;
                    let g = GptqWeights::new(INTER, HIDDEN, &e.gate.0, &e.gate.1);
                    let u = GptqWeights::new(INTER, HIDDEN, &e.up.0, &e.up.1);
                    let d = GptqWeights::new(HIDDEN, INTER, &e.down.0, &e.down.1);
                    gemv_into_prequantized(&g, &qa_h, &mut yg);
                    gemv_into_prequantized(&u, &qa_h, &mut yu);
                    gemv_into_prequantized(&d, &qa_i, &mut yh);
                    sink += yh[0] + yg[0] + yu[0];
                }
            }
            (t0.elapsed().as_secs_f64(), sink)
        };
        let bench_chunk_nvfp4 = |reps: usize| -> (f64, f32) {
            const GSCALE: f32 = 0.5;
            let mut yg = vec![0.0f32; INTER];
            let mut yu = vec![0.0f32; INTER];
            let mut yh = vec![0.0f32; HIDDEN];
            let mut sink = 0.0f32;
            let t0 = Instant::now();
            let mut idx = 0usize;
            for _ in 0..reps {
                for _ in 0..chunk_calls {
                    let e = &nvfp4_pool[idx % POOL];
                    idx += 1;
                    let g = PackedWeights::new(INTER, HIDDEN, &e.gate.0, &e.gate.1, GSCALE);
                    let u = PackedWeights::new(INTER, HIDDEN, &e.up.0, &e.up.1, GSCALE);
                    let d = PackedWeights::new(HIDDEN, INTER, &e.down.0, &e.down.1, GSCALE);
                    nvfp4_gemv::gemv_into(&g, &xh, &mut yg);
                    nvfp4_gemv::gemv_into(&u, &xh, &mut yu);
                    nvfp4_gemv::gemv_into(&d, &xi, &mut yh);
                    sink += yh[0] + yg[0] + yu[0];
                }
            }
            (t0.elapsed().as_secs_f64(), sink)
        };
        // warm
        let _ = bench_chunk_int4(2);
        let _ = bench_chunk_nvfp4(2);
        let (tc_i4, sc_i4) = bench_chunk_int4(LAYER_REPS);
        let (tc_nv, sc_nv) = bench_chunk_nvfp4(LAYER_REPS);
        let chunk_bytes_i4 = int4_bpe * chunk_calls; // ~ one layer
        let chunk_bytes_nv = nvfp4_bpe * chunk_calls;
        let ci4_gbps = (chunk_bytes_i4 * LAYER_REPS) as f64 / tc_i4 / 1e9;
        let cnv_gbps = (chunk_bytes_nv * LAYER_REPS) as f64 / tc_nv / 1e9;
        let ci4_gflops = (flops_per_expert * chunk_calls * LAYER_REPS) as f64 / tc_i4 / 1e9;
        let cnv_gflops = (flops_per_expert * chunk_calls * LAYER_REPS) as f64 / tc_nv / 1e9;

        eprintln!("(b) PER-LAYER chunk (~{:.1} MiB INT4 / {:.1} MiB NVFP4, one layer's {chunk_calls} experts)",
            chunk_bytes_i4 as f64 / 1048576.0, chunk_bytes_nv as f64 / 1048576.0);
        eprintln!("    INT4  : {ci4_gbps:6.1} GB/s  {ci4_gflops:7.1} GFLOP/s   sink={sc_i4:.2}");
        eprintln!("    NVFP4 : {cnv_gbps:6.1} GB/s  {cnv_gflops:7.1} GFLOP/s   sink={sc_nv:.2}");
        eprintln!("    speedup INT4/NVFP4 (by time): {:.2}×", tc_nv / tc_i4);
        eprintln!("════════════════════════════════════════════════════════════");
        eprintln!(" VERDICT (full): INT4 is {:.2}× NVFP4 by wall time", t_nv / t_i4);
        eprintln!(" VERDICT (18MiB chunk): INT4 is {:.2}× NVFP4 by wall time", tc_nv / tc_i4);
        eprintln!("════════════════════════════════════════════════════════════");
        assert!(s_i4.is_finite() && s_nv.is_finite() && sc_i4.is_finite() && sc_nv.is_finite());
    }

    /// Single-thread per-core kernel bandwidth for INT4 (no rayon), to compare to
    /// the NVFP4 single-thread number and isolate the inner kernel from threading.
    #[test]
    #[ignore]
    fn int4_single_thread_kernel_bandwidth() {
        const COLS: usize = 2048;
        const ROWS: usize = 64 * 1024; // ~67 MiB packed >> LLC
        let mut rng = SmallRng::seed_from_u64(SEED ^ 0x57);
        let (packed, scales) = random_gptq(&mut rng, ROWS, COLS);
        let x: Vec<f32> = (0..COLS).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let qa = quantize_activation(&x);
        let pc = COLS / 2;
        let sc = COLS / GROUP;
        let bytes_per_row = pc + sc * 2; // packed + F16 scales

        let kernel = |start: usize, count: usize| -> f32 {
            let mut s = 0.0f32;
            for r in start..start + count {
                let rr = r % ROWS;
                let prow = &packed[rr * pc..(rr + 1) * pc];
                let srow = &scales[rr * sc..(rr + 1) * sc];
                s += fused_row_dot_w4a8(prow, srow, &qa);
            }
            s
        };
        let mut sink = kernel(0, ROWS);
        const WINDOW: usize = ROWS * 4;
        let mut best = 0.0f64;
        for w in 0..5 {
            let t0 = Instant::now();
            sink += kernel(w * 7, WINDOW);
            best = best.max((WINDOW * bytes_per_row) as f64 / t0.elapsed().as_secs_f64() / 1e9);
        }
        eprintln!("── INT4 W4A8 fused kernel: SINGLE-THREAD per-core bandwidth ──");
        eprintln!("best single-core: {best:.1} GB/s  (×6 ≈ {:.0} GB/s ceiling)  sink={sink:.1}", best * 6.0);
        assert!(sink.is_finite());
    }
}

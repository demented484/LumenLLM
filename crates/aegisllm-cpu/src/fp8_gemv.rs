//! Fused CPU dequant-GEMV for DeepSeek-style **block-scaled FP8 (E4M3)** weights
//! — the second consumer of the "any-quant" fused-in-register design proven by
//! [`crate::nvfp4_gemv`]. Same thesis: **the dequant is fused into the dot and
//! the widened weights are NEVER materialized in RAM**, so DRAM traffic equals
//! exactly the packed FP8 bytes + the (tiny) block-scale bytes.
//!
//! # Why FP8 is a different shape of the same problem
//!
//! NVFP4 packs 16 elements into an 8-byte block with a *1-byte* per-16 scale.
//! DeepSeek-style FP8 (what our Qwen3.5-9B-FP8 checkpoint uses) is:
//!
//! * weight: `F8_E4M3` `[rows, cols]`, **1 byte per element**, row-major;
//! * scale: `weight_scale_inv` `[ceil(rows/B), ceil(cols/B)]`, **square `B×B`
//!   blocks** (`B = 128` for the 9B), stored BF16 on disk → decoded to f32;
//! * dequant: `w[r,c] = e4m3(byte[r,c]) * scale[r/B, c/B]`  (**MULTIPLY**),
//!   exactly as the GPU `aegis_fp8_block_matvec` and the loader's
//!   `dequant_fp8_block_into_bf16` do — there is **no** per-tensor global scale.
//!
//! Two structural consequences vs NVFP4, both handled here:
//!
//! 1. **8-bit, not 4-bit.** Each element is a full byte, so for a given DRAM
//!    budget we stream half as many *elements* as NVFP4 — but the byte stream is
//!    still the on-disk size, so the kernel stays bandwidth-bound (see microbench).
//!    The in-register decode is a 256-entry LUT gather instead of a 16-entry one.
//! 2. **128-wide scale groups along K, indexed by `row/B` along N.** A single
//!    f32 scale covers 128 consecutive columns, so the inner loop holds one scale
//!    constant across **eight** 16-lane AVX-512 vectors (128 = 8×16) — the scale
//!    multiply is hoisted out of the FMA chain and amortized 128-wide. The
//!    per-row scale-row offset is `(r / B) * scale_cols`, computed once per row.
//!
//! # In-register decode on Zen4 (no native FP8)
//!
//! Zen4 has no FP8 conversion instruction, so the E4M3→f32 decode is a 256-entry
//! f32 LUT gather: zero-extend 16 weight bytes to 16 u32 lanes and
//! `vpgatherdd`/`_mm512_i32gather_ps` the codes. The LUT is built once
//! ([`E4M3_SIGNED_LUT`]) and is **bit-identical** to the GPU
//! `linear_utils.cuh::fp8_e4m3_bits_to_float` and the loader's
//! `build_e4m3_signed_lut` (sign · {subnormal `m·2^-9` | normal `(1+m/8)·2^(e-7)`},
//! lone NaN at `S.1111.111`). Nothing wider than the FP8 bytes is read from RAM.
//!
//! # Numerical faithfulness
//!
//! The scalar reference dequants each element via the same [`E4M3_SIGNED_LUT`]
//! and the same f32 block scale, then dots in f64 — so the only difference vs the
//! fused AVX-512 path is FP32 summation order. Unit tests assert cosine > 0.9999
//! and tiny max-rel-err on representative Qwen3.5-9B FP8 shapes.

use rayon::prelude::*;

/// Signed OCP FP8 E4M3 decode table (256 entries, indexed by the raw byte).
///
/// Bit-identical to the GPU `linear_utils.cuh::fp8_e4m3_bits_to_float` and to the
/// loader's `build_e4m3_signed_lut` (`crates/aegisllm-cuda/src/cuda/loader.rs`):
/// 1 sign bit, 4 exp bits (bias 7), 3 mantissa bits; subnormal (exp==0) is
/// `mant · 2^-9`; the lone NaN is `S.1111.111`. E4M3 has no infinities; max =
/// `1.75 · 2^8 = 448`. Stored `f32` so the AVX-512 path can gather directly.
///
/// We define it locally (rather than reuse a base helper) because `aegisllm-base`
/// only exposes the *unsigned* UE4M3-with-½ decoder used for NVFP4 *scales*
/// (`decode_ue4m3_with_half_lut`), which folds a ×0.5 and drops the sign bit —
/// wrong for signed FP8 *weight* bytes. This table is the faithful signed decode.
pub const E4M3_SIGNED_LUT: [f32; 256] = build_e4m3_signed_lut();

const fn build_e4m3_signed_lut() -> [f32; 256] {
    // `const fn` so the table is materialized at compile time (no runtime init,
    // no `powi`/`exp2` at startup). Mirrors loader::build_e4m3_signed_lut exactly.
    let mut lut = [0.0f32; 256];
    let mut b = 0usize;
    while b < 256 {
        let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
        let exp = ((b >> 3) & 0x0F) as i32;
        let mant = (b & 0x07) as f32;
        let v = if exp == 0 {
            // Subnormal: 2^(1-bias) · (mant/8), bias 7 → 2^-6 = 0.015625.
            (mant / 8.0) * 0.015_625
        } else if exp == 15 && (b & 0x07) == 7 {
            f32::NAN
        } else {
            // Normal: (1 + mant/8) · 2^(exp-7). `powi` isn't const, so build the
            // power of two by constructing the f32 exponent bits directly:
            // 2^(exp-7) has biased exponent (exp-7)+127 = exp+120 in [121, 134],
            // always a valid normal f32, mantissa 0.
            let pow2 = f32::from_bits(((exp + 120) as u32) << 23);
            (1.0 + mant / 8.0) * pow2
        };
        lut[b] = sign * v;
        b += 1;
    }
    lut
}

// ── Format trait extension point (FP8 today; MXFP4 / MXFP8 documented stubs) ──

/// A 2-D **block-scaled FP8-family** weight format: square `B×B` scale blocks, a
/// per-element 8-bit code decoded through a 256-entry signed LUT, and a *decoded
/// f32* scale that multiplies the code. This is the FP8 sibling of
/// [`crate::nvfp4_gemv::BlockFormat`]; it is split out because FP8's scale grid is
/// 2-D and runtime-sized (`B` is a load-time value, not a compile-time 16), and
/// its scale is f32 rather than a 1-byte FP8 code — extending the NVFP4 trait to
/// cover both would erase the compile-time block size that makes the NVFP4 SIMD
/// path branch-free. Keeping two small traits is the honest "the design
/// generalizes per-family" answer (the *fused-in-register* skeleton is shared).
pub trait Fp8BlockFormat: Sync {
    /// Decode one raw weight byte to its f32 code (pre-scale). For FP8 this is the
    /// signed E4M3 LUT lookup; the per-block f32 scale is applied by the GEMV.
    fn decode_code(byte: u8) -> f32;
}

/// DeepSeek-style block-scaled FP8: signed E4M3 element, square `B×B` f32 scale,
/// dequant = `code · scale` (no per-tensor global scale).
pub struct Fp8E4M3Block;

impl Fp8BlockFormat for Fp8E4M3Block {
    #[inline(always)]
    fn decode_code(byte: u8) -> f32 {
        E4M3_SIGNED_LUT[byte as usize]
    }
}

// ── MXFP4 / MXFP8: DOCUMENTED STUBS — DO NOT IMPLEMENT ────────────────────────
//
// These are intentionally unimplemented. They exist so the "any-quant" surface is
// visible and a future format slots in beside FP8/NVFP4 without re-architecting.
//
//   * MXFP4 — 4-bit E2M1 element + per-32 E8M0 (power-of-two) block scale.
//     KNOWN-BROKEN as a *weight* format here: per memory
//     `aegisllm_attention_quant_sensitivity`, MXFP4 quant gave ~10 effective bits
//     (not 4) and blew perplexity 129 → 42170 on the shared path; FP8 is the
//     right 8-bit target. No current CPU-experts model consumes MXFP4 weights.
//
//   * MXFP8 — 8-bit E4M3 (or E5M2) element + per-32 E8M0 block scale (the MX
//     analogue of the DeepSeek FP8 path above, but power-of-two scale and 32-wide
//     groups). No current model consumer on the CPU path.
//
// Implement either ONLY when a target model actually needs it; until then they
// must not pretend to work.

/// MXFP4 (E2M1 + per-32 E8M0 scale). **Unimplemented stub** — see the module note
/// above: MXFP4 is known-broken as a weight format and has no CPU consumer.
pub struct Mxfp4;
impl Fp8BlockFormat for Mxfp4 {
    fn decode_code(_byte: u8) -> f32 {
        unimplemented!(
            "MXFP4 not implemented: no current model consumer; MXFP4 is known-broken \
             (see memory aegisllm_attention_quant_sensitivity — ~10 eff bits, PPL \
             129→42170). Implement only when a model needs it."
        )
    }
}

/// MXFP8 (E4M3/E5M2 + per-32 E8M0 scale). **Unimplemented stub** — no current CPU
/// model consumer; implement when a target model needs it.
pub struct Mxfp8;
impl Fp8BlockFormat for Mxfp8 {
    fn decode_code(_byte: u8) -> f32 {
        unimplemented!(
            "MXFP8 not implemented: no current model consumer. Implement (E4M3/E5M2 \
             element + per-32 E8M0 power-of-two block scale) when a model needs it."
        )
    }
}

// ── Packed weight view ────────────────────────────────────────────────────────

/// A row-major FP8 E4M3 weight matrix `W[rows, cols]` (1 byte/elem) plus its
/// square `B×B` block scales `S[ceil(rows/B), ceil(cols/B)]` (decoded to f32).
/// Borrows the bytes — the kernel reads from here and nothing else, so the DRAM
/// weight traffic is `fp8.len()` (== `rows*cols`) + `scales.len()*4` (tiny).
#[derive(Clone, Copy)]
pub struct PackedFp8Weights<'a> {
    pub rows: usize,
    pub cols: usize,
    /// `rows * cols` E4M3 bytes, row-major.
    pub fp8: &'a [u8],
    /// `ceil(rows/block) * scale_cols` f32 block scales, row-major.
    pub scales: &'a [f32],
    /// Square scale-block edge `B` (e.g. 128).
    pub block: usize,
    /// Number of scale columns = `ceil(cols / block)`.
    pub scale_cols: usize,
}

impl<'a> PackedFp8Weights<'a> {
    pub fn new(rows: usize, cols: usize, fp8: &'a [u8], scales: &'a [f32], block: usize) -> Self {
        assert!(block > 0, "block must be > 0");
        let scale_cols = cols.div_ceil(block);
        let scale_rows = rows.div_ceil(block);
        assert_eq!(fp8.len(), rows * cols, "fp8 byte count mismatch");
        assert_eq!(scales.len(), scale_rows * scale_cols, "scale count mismatch");
        Self { rows, cols, fp8, scales, block, scale_cols }
    }
}

// ── Scalar reference GEMV (the f64 correctness oracle) ────────────────────────

/// Reference dequant-GEMV for M=1, accumulating in **f64** as the high-precision
/// ground truth: `y[r] = sum_c e4m3(W[r,c]) * scale[r/B, c/B] * x[c]`. Obviously
/// correct (per-element decode + scale + dot), single-threaded; validates the
/// fast paths. f64 accumulation isolates true FP32 rounding from summation-order
/// noise (same rationale as the NVFP4 reference).
pub fn gemv_reference(w: &PackedFp8Weights, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), w.cols);
    assert_eq!(y.len(), w.rows);
    #[allow(clippy::needless_range_loop)]
    for r in 0..w.rows {
        let wrow = &w.fp8[r * w.cols..(r + 1) * w.cols];
        let srow = (r / w.block) * w.scale_cols;
        let mut acc = 0.0f64;
        for c in 0..w.cols {
            let code = Fp8E4M3Block::decode_code(wrow[c]) as f64;
            let scale = w.scales[srow + c / w.block] as f64;
            acc += code * scale * x[c] as f64;
        }
        y[r] = acc as f32;
    }
}

// ── Fused dequant-dot for ONE row (the in-register core) ──────────────────────

/// Fused FP8 dequant-dot of one weight row against one input vector. Reads only
/// `wrow` (cols bytes) + the few f32 scales for this row's block (`scale_row`,
/// pre-sliced to this scale-row). Dispatches to AVX-512 when available, else a
/// portable fused scalar loop (still single-read-of-weights).
#[inline]
fn fused_row_dot(wrow: &[u8], scale_row: &[f32], block: usize, x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            // SAFETY: gated on runtime avx512f+avx512bw; `wrow`/`x` are `cols`
            // long, `scale_row` covers `ceil(cols/block)` scales, and the kernel
            // only indexes within those bounds.
            return unsafe { fused_row_dot_avx512(wrow, scale_row, block, x) };
        }
    }
    fused_row_dot_scalar(wrow, scale_row, block, x)
}

/// Portable fused scalar fallback. Walks the row in `block`-wide scale groups,
/// holding one f32 scale across the whole group, decoding each byte through the
/// signed LUT and FMA-ing into the group accumulator. No row materialization.
#[inline]
fn fused_row_dot_scalar(wrow: &[u8], scale_row: &[f32], block: usize, x: &[f32]) -> f32 {
    let cols = wrow.len();
    let mut acc = 0.0f32;
    let mut c = 0usize;
    let mut g = 0usize;
    while c < cols {
        let end = (c + block).min(cols);
        let scale = scale_row[g];
        let mut group_acc = 0.0f32;
        for k in c..end {
            group_acc += E4M3_SIGNED_LUT[wrow[k] as usize] * x[k];
        }
        acc += scale * group_acc;
        c = end;
        g += 1;
    }
    acc
}

/// Branchless in-register E4M3→f32 decode of 16 bytes (zero-extended to 16 u32
/// lanes in `idx`). Pure shift/mask/blend arithmetic — **no memory gather** —
/// because `_mm512_i32gather_ps` over a 256-entry LUT is microcoded and slow on
/// Zen4 (it caps the kernel at ~16 GB/s, far below the ~50+ GB/s DRAM ceiling;
/// the gather, not the DRAM read, becomes the bottleneck). Reconstructs the IEEE
/// f32 bit pattern directly:
///
/// * `sign`  = bit7  → f32 sign bit (`<<24`).
/// * normal (`exp4 ≥ 1`): f32 = `(1 + mant3/8)·2^(exp4-7)`, i.e. f32 exponent
///   field `exp4 + 120` (since `(exp4-7)+127`) and the 3 mantissa bits placed at
///   the top of the 23-bit f32 mantissa (`mant3 << 20`).
/// * subnormal (`exp4 == 0`): f32 = `sign · mant3 · 2^-9`; computed as
///   `(mant3 as f32) * 2^-9` and sign-applied, then blended in for `exp4==0`
///   lanes (this also yields 0 for `mant3==0`, i.e. ±0).
///
/// NaN bytes (`mag==0x7f`, i.e. `0x7f`/`0xff`) decode here to a large *finite*
/// value rather than NaN — quantized weight matrices never contain them, so this
/// never affects a real dot; the scalar reference + [`E4M3_SIGNED_LUT`] remain
/// the faithful NaN-aware oracle. (Validated cos≈1, max-rel-err ~1e-7 on
/// NaN-free shapes — see `fused_gemv_matches_reference`.)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
#[inline]
unsafe fn decode_e4m3_x16(idx: std::arch::x86_64::__m512i) -> std::arch::x86_64::__m512 {
    use std::arch::x86_64::*;
    let mag = _mm512_and_si512(idx, _mm512_set1_epi32(0x7f));
    let sign = _mm512_slli_epi32(_mm512_and_si512(idx, _mm512_set1_epi32(0x80)), 24);
    let exp4 = _mm512_and_si512(_mm512_srli_epi32(mag, 3), _mm512_set1_epi32(0x0f));
    let mant3 = _mm512_and_si512(mag, _mm512_set1_epi32(0x07));

    // Normal path: f32_bits = sign | ((exp4+120)<<23) | (mant3<<20).
    let exp_field = _mm512_slli_epi32(_mm512_add_epi32(exp4, _mm512_set1_epi32(120)), 23);
    let mant_field = _mm512_slli_epi32(mant3, 20);
    let normal_bits = _mm512_or_si512(_mm512_or_si512(sign, exp_field), mant_field);
    let normal = _mm512_castsi512_ps(normal_bits);

    // Subnormal path: sign · mant3 · 2^-9.
    let mant_f = _mm512_cvtepi32_ps(mant3);
    let sub_mag = _mm512_mul_ps(mant_f, _mm512_set1_ps(0.001_953_125)); // 2^-9
    // Apply sign by OR-ing the f32 sign bit (mant3 ≥ 0 so sub_mag ≥ 0).
    let sub = _mm512_castsi512_ps(_mm512_or_si512(_mm512_castps_si512(sub_mag), sign));

    // Select subnormal where exp4 == 0, else normal.
    let is_sub = _mm512_cmpeq_epi32_mask(exp4, _mm512_setzero_si512());
    _mm512_mask_blend_ps(is_sub, normal, sub)
}

/// AVX-512 fused FP8 dequant-dot of one row. The E4M3 decode (branchless
/// bit-manipulation, [`decode_e4m3_x16`]) + per-block scale + FMA happen entirely
/// in zmm registers; the only RAM reads for the weights are the FP8 bytes + the
/// handful of f32 block scales.
///
/// The scale is constant for `block` (=128) consecutive columns, so we accumulate
/// the unscaled 16-lane dot for a whole scale group, then fold the group's scale
/// in once (`acc += scale * group_dot`) — hoisting the scale multiply out of the
/// eight inner FMAs. One horizontal reduction per row (not per block).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
fn fused_row_dot_avx512(wrow: &[u8], scale_row: &[f32], block: usize, x: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    // SAFETY: caller guarantees avx512f+avx512bw at runtime; `wrow`/`x` are `cols`
    // long, `scale_row` has `ceil(cols/block)` entries, and every load below stays
    // within those bounds. A 16-element chunk never crosses a scale boundary
    // because `block` is a multiple of 16 for all real shapes; the scalar tail
    // handles any remainder (including a non-16-multiple `block`).
    unsafe {
        let cols = wrow.len();
        // Running row accumulator (vector) for the scaled group dots, plus a
        // scalar accumulator for the partial-chunk tails. One reduction at the end.
        let mut acc = _mm512_setzero_ps();
        let mut tail_acc = 0.0f32;
        let mut c = 0usize;
        let mut g = 0usize;
        while c < cols {
            let scale = scale_row[g];
            let group_end = (c + block).min(cols);
            // Vectorized 16-lane chunks within this scale group (unscaled).
            let mut group = _mm512_setzero_ps();
            let mut k = c;
            while k + 16 <= group_end {
                // Load 16 FP8 bytes into the low 128 bits of an xmm.
                let bytes = _mm_loadu_si128(wrow.as_ptr().add(k) as *const __m128i);
                // Zero-extend 16 u8 → 16 u32 lanes, then branchless E4M3 decode.
                let idx = _mm512_cvtepu8_epi32(bytes);
                let codes = decode_e4m3_x16(idx);
                let xv = _mm512_loadu_ps(x.as_ptr().add(k));
                group = _mm512_fmadd_ps(codes, xv, group);
                k += 16;
            }
            // Fold this group's scale in once (hoisted out of the inner FMAs).
            acc = _mm512_fmadd_ps(group, _mm512_set1_ps(scale), acc);
            // Scalar tail for a partial 16-chunk at the end of the group (only the
            // matrix's final group can be partial; real shapes are 16-aligned).
            while k < group_end {
                tail_acc += scale * E4M3_SIGNED_LUT[wrow[k] as usize] * x[k];
                k += 1;
            }
            c = group_end;
            g += 1;
        }
        _mm512_reduce_add_ps(acc) + tail_acc
    }
}

// ── Public fused GEMV / GEMM (multi-threaded over rows) ───────────────────────

/// Fused FP8 dequant-GEMV, M=1: `y[rows] = W[rows,cols] · x[cols]`, threaded over
/// rows (rayon). Each row reads its FP8 bytes from DRAM exactly once and dequants
/// in-register. The per-token expert projection primitive for FP8 checkpoints.
pub fn gemv_into(w: &PackedFp8Weights, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), w.cols, "input length must equal cols");
    assert_eq!(y.len(), w.rows, "output length must equal rows");
    let cols = w.cols;
    let block = w.block;
    let scale_cols = w.scale_cols;
    let fp8 = w.fp8;
    let scales = w.scales;
    y.par_iter_mut().enumerate().for_each(|(r, slot)| {
        let wrow = &fp8[r * cols..(r + 1) * cols];
        let sbase = (r / block) * scale_cols;
        let scale_row = &scales[sbase..sbase + scale_cols];
        *slot = fused_row_dot(wrow, scale_row, block, x);
    });
}

/// Fused FP8 dequant-GEMM for small M (batched verify / prefill): `Y[m, rows]`,
/// X and Y token-major. Each weight row is read from DRAM ONCE and dotted against
/// all `m` tokens (its bytes stay hot in L1/L2 across the token loop), so DRAM
/// weight traffic is independent of M.
pub fn gemm_into(w: &PackedFp8Weights, x: &[f32], m: usize, y: &mut [f32]) {
    if m == 0 {
        return;
    }
    if m == 1 {
        return gemv_into(w, x, y);
    }
    assert_eq!(x.len(), m * w.cols, "input length must equal m*cols");
    assert_eq!(y.len(), m * w.rows, "output length must equal m*rows");
    let cols = w.cols;
    let rows = w.rows;
    let block = w.block;
    let scale_cols = w.scale_cols;
    let fp8 = w.fp8;
    let scales = w.scales;

    let row_cols: Vec<Vec<f32>> = (0..rows)
        .into_par_iter()
        .map(|r| {
            let wrow = &fp8[r * cols..(r + 1) * cols];
            let sbase = (r / block) * scale_cols;
            let scale_row = &scales[sbase..sbase + scale_cols];
            let mut out = vec![0.0f32; m];
            for (t, slot) in out.iter_mut().enumerate() {
                let xt = &x[t * cols..(t + 1) * cols];
                *slot = fused_row_dot(wrow, scale_row, block, xt);
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
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};
    use std::time::Instant;

    const SEED: u64 = 0x5151_4650_38E4_4D33; // "QFP8\xe4M3"-ish

    /// Build a random valid FP8 weight matrix `[rows, cols]` with square `block`
    /// scales. FP8 bytes avoid the NaN codes (0x7f / 0xff) so every weight is a
    /// real magnitude; scales are random positive f32 in a realistic range.
    fn random_fp8(
        rng: &mut SmallRng,
        rows: usize,
        cols: usize,
        block: usize,
    ) -> (Vec<u8>, Vec<f32>) {
        let fp8: Vec<u8> = (0..rows * cols)
            .map(|_| {
                let b = rng.random::<u8>();
                // Avoid the lone NaN (mag 0x7f) on either sign.
                if b & 0x7f == 0x7f { b ^ 0x01 } else { b }
            })
            .collect();
        let scale_cols = cols.div_ceil(block);
        let scale_rows = rows.div_ceil(block);
        let scales: Vec<f32> = (0..scale_rows * scale_cols)
            // DeepSeek weight_scale_inv magnitudes are typically O(1e-2 .. 1e-1).
            .map(|_| rng.random::<f32>() * 0.1 + 0.01)
            .collect();
        (fp8, scales)
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b.iter()) {
            dot += x as f64 * y as f64;
            na += (x as f64) * (x as f64);
            nb += (y as f64) * (y as f64);
        }
        if na == 0.0 || nb == 0.0 {
            return 1.0;
        }
        dot / (na.sqrt() * nb.sqrt())
    }

    /// Max relative error of `got` vs the f64 `reference`, normalized by the
    /// output RMS (per-element rel-err is meaningless near zero from dot-product
    /// cancellation — same metric the NVFP4 tests use).
    fn max_rel_err(reference: &[f32], got: &[f32]) -> f64 {
        let n = reference.len().max(1) as f64;
        let rms = (reference.iter().map(|&r| (r as f64) * (r as f64)).sum::<f64>() / n).sqrt();
        let denom = rms.max(1e-12);
        let mut worst = 0.0f64;
        for (&r, &g) in reference.iter().zip(got.iter()) {
            worst = worst.max((r as f64 - g as f64).abs() / denom);
        }
        worst
    }

    /// Representative Qwen3.5-9B-FP8 linear shapes (DeepSeek 128×128 blocks), plus
    /// small shapes that exercise the block edges and partial groups.
    const FP8_SHAPES: &[(usize, usize, usize, &str)] = &[
        (4096, 4096, 128, "qkv/o-ish [4096x4096] B=128"),
        (12288, 4096, 128, "gate/up [12288x4096] B=128"),
        (4096, 12288, 128, "down [4096x12288] B=128"),
        (256, 512, 128, "small [256x512] B=128"),
        (8, 256, 128, "few rows [8x256] B=128"),
        (130, 384, 128, "partial scale row [130x384] B=128"),
        (64, 320, 64, "B=64 partial last group [64x320]"),
        (3, 16, 16, "single-group rows [3x16] B=16"),
    ];

    #[test]
    fn lut_matches_gpu_decode_spot_checks() {
        // Spot-check the signed E4M3 LUT against hand-computed values matching the
        // GPU fp8_e4m3_bits_to_float / loader build_e4m3_signed_lut.
        assert_eq!(E4M3_SIGNED_LUT[0x00], 0.0); // +0
        assert_eq!(E4M3_SIGNED_LUT[0x80], 0.0); // -0 (sign·0)
        // 0x38 = 0_0111_000 → exp 7 (unbiased 0), mant 0 → 1.0
        assert_eq!(E4M3_SIGNED_LUT[0x38], 1.0);
        assert_eq!(E4M3_SIGNED_LUT[0xB8], -1.0); // sign flip
        // 0x3C = 0_0111_100 → (1 + 4/8)·2^0 = 1.5
        assert_eq!(E4M3_SIGNED_LUT[0x3C], 1.5);
        // max normal 0x7e = 0_1111_110 → (1+6/8)·2^8 = 1.75·256 = 448
        assert_eq!(E4M3_SIGNED_LUT[0x7e], 448.0);
        // smallest subnormal 0x01 → 1·2^-9 = 0.001953125
        assert_eq!(E4M3_SIGNED_LUT[0x01], 0.001_953_125);
        // NaN codes
        assert!(E4M3_SIGNED_LUT[0x7f].is_nan());
        assert!(E4M3_SIGNED_LUT[0xff].is_nan());
    }

    #[test]
    fn fused_gemv_matches_reference() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        for &(rows, cols, block, label) in FP8_SHAPES {
            let (fp8, scales) = random_fp8(&mut rng, rows, cols, block);
            let w = PackedFp8Weights::new(rows, cols, &fp8, &scales, block);
            let x: Vec<f32> = (0..cols).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();

            let mut y_ref = vec![0.0f32; rows];
            let mut y_fast = vec![0.0f32; rows];
            gemv_reference(&w, &x, &mut y_ref);
            gemv_into(&w, &x, &mut y_fast);

            let cos = cosine(&y_ref, &y_fast);
            let rel = max_rel_err(&y_ref, &y_fast);
            assert!(cos > 0.9999, "{label}: cosine {cos} <= 0.9999 (fused vs reference)");
            assert!(rel < 1e-4, "{label}: max-rel-err {rel} >= 1e-4 (fused vs reference)");
            eprintln!("{label}: cos={cos:.8} max_rel_err={rel:.3e}");
        }
    }

    #[test]
    fn fused_scalar_path_matches_reference() {
        let mut rng = SmallRng::seed_from_u64(SEED ^ 0xABCD);
        for &(rows, cols, block, label) in FP8_SHAPES {
            let (fp8, scales) = random_fp8(&mut rng, rows, cols, block);
            let w = PackedFp8Weights::new(rows, cols, &fp8, &scales, block);
            let x: Vec<f32> = (0..cols).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
            let scale_cols = cols.div_ceil(block);

            let mut y_ref = vec![0.0f32; rows];
            gemv_reference(&w, &x, &mut y_ref);
            let mut y_scalar = vec![0.0f32; rows];
            for r in 0..rows {
                let wrow = &fp8[r * cols..(r + 1) * cols];
                let sbase = (r / block) * scale_cols;
                y_scalar[r] = fused_row_dot_scalar(wrow, &scales[sbase..sbase + scale_cols], block, &x);
            }
            let cos = cosine(&y_ref, &y_scalar);
            let rel = max_rel_err(&y_ref, &y_scalar);
            assert!(cos > 0.9999, "{label}: scalar-fused cosine {cos}");
            assert!(rel < 1e-4, "{label}: scalar-fused max-rel-err {rel}");
        }
    }

    #[test]
    fn fused_gemm_matches_per_token_gemv() {
        let mut rng = SmallRng::seed_from_u64(SEED ^ 0x1234);
        let (rows, cols, block) = (512usize, 1024usize, 128usize);
        let (fp8, scales) = random_fp8(&mut rng, rows, cols, block);
        let w = PackedFp8Weights::new(rows, cols, &fp8, &scales, block);
        for &m in &[2usize, 4, 8] {
            let x: Vec<f32> = (0..m * cols).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
            let mut y_batch = vec![0.0f32; m * rows];
            gemm_into(&w, &x, m, &mut y_batch);

            let mut y_expected = vec![0.0f32; m * rows];
            for t in 0..m {
                let xt = &x[t * cols..(t + 1) * cols];
                let mut yt = vec![0.0f32; rows];
                gemv_into(&w, xt, &mut yt);
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

    /// Sanity: all-zero FP8 bytes → zero output regardless of scales.
    #[test]
    fn zero_weights_give_zero() {
        let (rows, cols, block) = (8usize, 256usize, 128usize);
        let fp8 = vec![0u8; rows * cols];
        let scales = vec![0.05f32; rows.div_ceil(block) * cols.div_ceil(block)];
        let w = PackedFp8Weights::new(rows, cols, &fp8, &scales, block);
        let x: Vec<f32> = (0..cols).map(|i| i as f32 / 31.0 - 1.0).collect();
        let mut y = vec![1.0f32; rows];
        gemv_into(&w, &x, &mut y);
        assert!(y.iter().all(|&v| v == 0.0), "expected all-zero output");
    }

    #[test]
    #[should_panic(expected = "MXFP4 not implemented")]
    fn mxfp4_is_a_stub() {
        let _ = Mxfp4::decode_code(0);
    }

    #[test]
    #[should_panic(expected = "MXFP8 not implemented")]
    fn mxfp8_is_a_stub() {
        let _ = Mxfp8::decode_code(0);
    }

    // ── Microbench (ignored; `--release -- --ignored --nocapture`) ────────────
    //
    // Streams a >LLC pool of distinct FP8 weight rows through the fused kernel and
    // reports GB/s, GFLOP/s, and the RAM-bound verdict. FP8 is 8-bit, so per
    // element it's 2× the bytes of NVFP4 (fewer elements/sec) — the question is
    // whether it still saturates DRAM. We test both a single-thread per-core
    // number and the multi-core parallel ceiling.

    #[test]
    #[ignore]
    fn fp8_single_thread_kernel_bandwidth() {
        const COLS: usize = 4096;
        const BLOCK: usize = 128;
        // One row = COLS fp8 bytes + COLS/128 f32 scales ≈ 4.1 KiB; 16k rows ≈
        // 67 MiB pool >> LLC, so reads come from DRAM not cache.
        const ROWS: usize = 16 * 1024;
        let mut rng = SmallRng::seed_from_u64(SEED ^ 0x57);
        let (fp8, scales) = random_fp8(&mut rng, ROWS, COLS, BLOCK);
        let x: Vec<f32> = (0..COLS).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let scale_cols = COLS.div_ceil(BLOCK);
        // Bytes touched per row = fp8 row + its f32 scales.
        let bytes_per_row = COLS + scale_cols * 4;

        let kernel = |start: usize, count: usize| -> f32 {
            let mut s = 0.0f32;
            for r in start..start + count {
                let rr = r % ROWS;
                let wrow = &fp8[rr * COLS..(rr + 1) * COLS];
                let sbase = (rr / BLOCK) * scale_cols;
                s += fused_row_dot(wrow, &scales[sbase..sbase + scale_cols], BLOCK, &x);
            }
            s
        };
        let mut sink = kernel(0, ROWS);
        const WINDOW: usize = ROWS * 4;
        let mut best_gbps = 0.0f64;
        let mut best_gflops = 0.0f64;
        for w in 0..5 {
            let t0 = Instant::now();
            sink += kernel(w * 7, WINDOW);
            let secs = t0.elapsed().as_secs_f64();
            let gbps = (WINDOW * bytes_per_row) as f64 / secs / 1e9;
            // 2 FLOP/elem (mul+add); COLS elems/row.
            let gflops = (WINDOW * COLS * 2) as f64 / secs / 1e9;
            if gbps > best_gbps {
                best_gbps = gbps;
                best_gflops = gflops;
            }
        }
        eprintln!("── FP8 fused kernel: SINGLE-THREAD per-core ──");
        eprintln!(
            "best single-core: {best_gbps:.1} GB/s, {best_gflops:.1} GFLOP/s  (×6 cores ≈ {:.0} GB/s)  sink={sink:.1}",
            best_gbps * 6.0
        );
        assert!(sink.is_finite());
    }

    #[test]
    #[ignore]
    fn fp8_parallel_ceiling_single_region() {
        const COLS: usize = 4096;
        const BLOCK: usize = 128;
        const ROWS: usize = 16 * 1024; // ~67 MiB pool >> LLC
        let mut rng = SmallRng::seed_from_u64(SEED ^ 0xBEEF);
        let (fp8, scales) = random_fp8(&mut rng, ROWS, COLS, BLOCK);
        let x: Vec<f32> = (0..COLS).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let scale_cols = COLS.div_ceil(BLOCK);
        let bytes_per_row = COLS + scale_cols * 4;

        let run = || -> f32 {
            (0..ROWS)
                .into_par_iter()
                .map(|r| {
                    let wrow = &fp8[r * COLS..(r + 1) * COLS];
                    let sbase = (r / BLOCK) * scale_cols;
                    fused_row_dot(wrow, &scales[sbase..sbase + scale_cols], BLOCK, &x)
                })
                .sum()
        };
        let mut sink = run();
        let mut best_gbps = 0.0f64;
        let mut best_gflops = 0.0f64;
        for _ in 0..6 {
            let t0 = Instant::now();
            sink += run();
            let secs = t0.elapsed().as_secs_f64();
            let gbps = (ROWS * bytes_per_row) as f64 / secs / 1e9;
            let gflops = (ROWS * COLS * 2) as f64 / secs / 1e9;
            if gbps > best_gbps {
                best_gbps = gbps;
                best_gflops = gflops;
            }
        }
        let ai = (COLS * 2) as f64 / bytes_per_row as f64;
        eprintln!("── FP8 fused kernel: PARALLEL ceiling (single rayon region) ──");
        eprintln!("arithmetic intensity: {ai:.2} FLOP/byte");
        eprintln!("best parallel: {best_gbps:.1} GB/s, {best_gflops:.1} GFLOP/s  sink={sink:.1}");
        assert!(sink.is_finite());
    }

    /// Per-token Qwen3.5-9B-FP8-ish dense projection set streamed through the
    /// fused kernel: reports GB/s, GFLOP/s and the RAM-bound verdict (DDR5
    /// dual-channel ceiling ~60-70 GB/s).
    #[test]
    #[ignore]
    fn fp8_projection_microbench_cold() {
        const HIDDEN: usize = 4096;
        const INTER: usize = 12288;
        const BLOCK: usize = 128;
        let mut rng = SmallRng::seed_from_u64(SEED ^ 0xC01D);
        // gate/up [INTER, HIDDEN], down [HIDDEN, INTER]; build a few distinct sets
        // so the working set > LLC.
        const POOL: usize = 4;
        struct Proj {
            gate: (Vec<u8>, Vec<f32>),
            up: (Vec<u8>, Vec<f32>),
            down: (Vec<u8>, Vec<f32>),
        }
        let pool: Vec<Proj> = (0..POOL)
            .map(|_| Proj {
                gate: random_fp8(&mut rng, INTER, HIDDEN, BLOCK),
                up: random_fp8(&mut rng, INTER, HIDDEN, BLOCK),
                down: random_fp8(&mut rng, HIDDEN, INTER, BLOCK),
            })
            .collect();
        let x_hidden: Vec<f32> = (0..HIDDEN).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let x_inter: Vec<f32> = (0..INTER).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let mut yg = vec![0.0f32; INTER];
        let mut yu = vec![0.0f32; INTER];
        let mut yh = vec![0.0f32; HIDDEN];

        let bytes_per_call = pool[0].gate.0.len()
            + pool[0].gate.1.len() * 4
            + pool[0].up.0.len()
            + pool[0].up.1.len() * 4
            + pool[0].down.0.len()
            + pool[0].down.1.len() * 4;
        let flops_per_call = (INTER * HIDDEN * 2 + HIDDEN * INTER) * 2;

        let run = |p: &Proj, yg: &mut [f32], yu: &mut [f32], yh: &mut [f32]| {
            let g = PackedFp8Weights::new(INTER, HIDDEN, &p.gate.0, &p.gate.1, BLOCK);
            let u = PackedFp8Weights::new(INTER, HIDDEN, &p.up.0, &p.up.1, BLOCK);
            let d = PackedFp8Weights::new(HIDDEN, INTER, &p.down.0, &p.down.1, BLOCK);
            gemv_into(&g, &x_hidden, yg);
            gemv_into(&u, &x_hidden, yu);
            gemv_into(&d, &x_inter, yh);
        };
        for i in 0..8 {
            run(&pool[i % POOL], &mut yg, &mut yu, &mut yh);
        }
        const CALLS: usize = 200;
        let t0 = Instant::now();
        let mut sink = 0.0f32;
        for i in 0..CALLS {
            run(&pool[i % POOL], &mut yg, &mut yu, &mut yh);
            sink += yg[0] + yu[0] + yh[0];
        }
        let secs = t0.elapsed().as_secs_f64();
        let bytes_total = (bytes_per_call * CALLS) as f64;
        let flops_total = (flops_per_call * CALLS) as f64;
        eprintln!("── FP8 fused projection microbench (Qwen3.5-9B-ish dense) ──");
        eprintln!("arithmetic intensity: {:.2} FLOP/byte", flops_total / bytes_total);
        eprintln!(
            "achieved: {:.1} GB/s, {:.1} GFLOP/s  (sink={sink:.3})",
            bytes_total / secs / 1e9,
            flops_total / secs / 1e9
        );
        assert!(sink.is_finite());
    }
}

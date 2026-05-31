//! Fused CPU dequant-GEMV for block-scaled 4-bit weights (NVFP4 today; FP8 /
//! MXFP4 / MXFP8 are clean extension points behind the [`BlockFormat`] trait).
//!
//! # The whole point: dequant FUSED into the dot, never materialized in RAM
//!
//! The "experts-on-CPU" path for Qwen3.6-35B-A3B streams ~540 MiB of packed
//! NVFP4 expert weights from DRAM **per decoded token** (8 active experts ×
//! 3 projections × 40 MoE layers). A naive "dequant a row to a BF16/FP32 buffer
//! in RAM, then GEMV" blows that up: write ~2 GiB of widened weights, then
//! re-read 2 GiB to dot them — ~8× the DRAM traffic of the packed bytes, which
//! is the bandwidth wall on a memory-bound kernel.
//!
//! This kernel reads **only the packed bytes + per-block scale bytes** from DRAM
//! (i.e. exactly the on-disk quant size). The 4-bit unpack + per-block scale are
//! applied entirely in AVX-512 registers and fused straight into an FP32 FMA
//! accumulate. The weight matrix is never widened in memory — same strategy
//! llama.cpp uses to stay fast on quantized CPU weights.
//!
//! ## How the dequant stays in-register (NVFP4)
//!
//! An NVFP4 16-element block = 8 packed bytes (16 signed E2M1 nibbles) + 1 FP8
//! E4M3 block-scale byte. The decoded weight is
//! `w[k] = e2m1_code(nibble_k) * block_scale`, where `e2m1_code` is one of the
//! 16 small signed integers `{0,±1,±2,±3,±4,±6,±8,±12}` (the ×0.5 magnitude
//! factor is folded into `block_scale`, exactly as the GPU does — see
//! `linear_utils.cuh::decode_nvfp4_nibble`/`decode_ue4m3_half`). Therefore the
//! block's contribution to the dot factors as
//!
//! ```text
//!   sum_k w[k]*x[k] = block_scale * sum_k ( code_k * x[k] )
//! ```
//!
//! so we (1) load 8 bytes, (2) explode them to 16 nibbles, (3) gather the 16
//! integer codes from a 16-entry FP32 LUT via `vpermps`, (4) FMA them against
//! the 16 input lanes into a per-block FP32 accumulator, and (5) multiply by the
//! single decoded `block_scale` once per block. The per-tensor `output_scale`
//! (global scale) is applied once at the very end of the row. Nothing wider than
//! the 8 packed bytes is ever read from or written to RAM for the weights.
//!
//! ## Numerical faithfulness
//!
//! The scalar reference reuses the exact public decode helpers from
//! `aegisllm-base` (`decode_nvfp4_nibble_i8`, `decode_ue4m3_with_half_lut`),
//! which are bit-for-bit the same math as the GPU `linear_utils.cuh` path
//! (integer code LUT + 128-entry UE4M3 LUT with the ×0.5 folded in). The fused
//! AVX-512 path gathers from the *same* integer LUT and uses the *same*
//! `block_scale` and `output_scale`, so the only difference vs. the reference is
//! FP32 summation order. Unit tests assert cosine > 0.9999 and tiny max-rel-err.

use aegisllm_base::tensor::quant::{
    QK_NVFP4_SUB, decode_nvfp4_nibble_i8, decode_ue4m3_with_half_lut,
};
use rayon::prelude::*;

/// The 16 signed integer E2M1 codes, indexed by nibble. Bit-identical to the
/// GPU `decode_nvfp4_nibble` and to `aegisllm_base::...::decode_nvfp4_nibble_i8`.
/// Stored as `f32` so the AVX-512 path can `vpermps`-gather them directly.
const NVFP4_CODE_LUT_F32: [f32; 16] = [
    0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0,
];

const SUB: usize = QK_NVFP4_SUB; // 16 — elements per block / per FP8 scale byte
const PACKED_PER_BLOCK: usize = SUB / 2; // 8 packed bytes per 16-element block

// ── Format trait (extension point for FP8 / MXFP4 / MXFP8) ────────────────────

/// A block-scaled quantized weight format. Implementors describe their packing
/// (bytes-per-block, scale decoding) and provide a *scalar* block dequant; the
/// generic GEMV skeleton ([`gemv_into`]) and threading are shared across formats.
///
/// The SIMD fast path is currently NVFP4-specific (see [`Nvfp4`]). New formats
/// implement this trait for correctness + the reference path immediately, and
/// can add their own SIMD kernel later behind the same row/threading skeleton.
pub trait BlockFormat: Sync {
    /// Elements contributing to one scale group (NVFP4 / MXFP4: 16).
    fn elems_per_block() -> usize;
    /// Packed weight bytes per block (NVFP4: 8 = 16 nibbles).
    fn packed_bytes_per_block() -> usize;
    /// Scale bytes per block (NVFP4: 1 FP8 E4M3 byte).
    fn scale_bytes_per_block() -> usize;

    /// Scalar, obviously-correct dequant of one block's weights into `out`
    /// (`elems_per_block()` long): `out[k] = code(packed) * decoded_scale`.
    /// This defines the format's semantics and is the unit-test oracle.
    fn dequant_block_scalar(packed: &[u8], scale: &[u8], out: &mut [f32]);
}

/// NVFP4 = 4-bit E2M1 elements, per-16 block scale in FP8 E4M3, × per-tensor
/// global scale (applied by the GEMV, not the block dequant).
pub struct Nvfp4;

impl BlockFormat for Nvfp4 {
    #[inline(always)]
    fn elems_per_block() -> usize {
        SUB
    }
    #[inline(always)]
    fn packed_bytes_per_block() -> usize {
        PACKED_PER_BLOCK
    }
    #[inline(always)]
    fn scale_bytes_per_block() -> usize {
        1
    }

    #[inline]
    fn dequant_block_scalar(packed: &[u8], scale: &[u8], out: &mut [f32]) {
        // Block scale is the single FP8 E4M3 byte (×0.5 already folded into the LUT).
        let block_scale = decode_ue4m3_with_half_lut(scale[0]);
        for j in 0..PACKED_PER_BLOCK {
            let byte = packed[j];
            // Byte j holds columns 2j (low nibble) and 2j+1 (high nibble) — the
            // exact ordering the GPU `aegis_nvfp4_linear_prequantized` uses.
            out[2 * j] = decode_nvfp4_nibble_i8(byte & 0x0f) as f32 * block_scale;
            out[2 * j + 1] = decode_nvfp4_nibble_i8(byte >> 4) as f32 * block_scale;
        }
    }
}

// ── Packed weight view ────────────────────────────────────────────────────────

/// A row-major packed NVFP4 weight matrix `W[rows, cols]` plus its block scales
/// and global (per-tensor) scale. Borrows the packed bytes — the kernel reads
/// from here and nothing else, so DRAM traffic == `packed.len() + scales.len()`.
#[derive(Clone, Copy)]
pub struct PackedWeights<'a> {
    pub rows: usize,
    pub cols: usize,
    /// `rows * cols/2` packed nibble bytes, row-major.
    pub packed: &'a [u8],
    /// `rows * cols/16` FP8 E4M3 block-scale bytes, row-major.
    pub scales: &'a [u8],
    /// Per-tensor global scale (applied once per output element).
    pub output_scale: f32,
}

impl<'a> PackedWeights<'a> {
    pub fn new(
        rows: usize,
        cols: usize,
        packed: &'a [u8],
        scales: &'a [u8],
        output_scale: f32,
    ) -> Self {
        assert_eq!(cols % SUB, 0, "cols must be a multiple of {SUB}");
        assert_eq!(packed.len(), rows * cols / 2, "packed byte count mismatch");
        assert_eq!(scales.len(), rows * cols / SUB, "scale byte count mismatch");
        Self { rows, cols, packed, scales, output_scale }
    }

    #[inline]
    fn packed_cols(&self) -> usize {
        self.cols / 2
    }
    #[inline]
    fn scale_cols(&self) -> usize {
        self.cols / SUB
    }
}

// ── Scalar reference GEMV (the correctness oracle) ────────────────────────────

/// Reference dequant-GEMV for M=1: `y[r] = (sum_c dequant(W[r,c]) * x[c]) * gscale`.
/// Dequants each block to a tiny stack buffer, then dots — simple and obviously
/// correct. Single-threaded; used only to validate the fast paths.
///
/// Accumulates in **f64** so it serves as the high-precision ground truth: the
/// FP32 fused/scalar paths are validated against it, and the only differences
/// are FP32 rounding + summation order (no algorithmic difference — same LUT,
/// same scales). This is what makes the accuracy numbers meaningful (the FP32
/// reference would itself carry summation-order noise comparable to the fast
/// path, masking the true error).
pub fn gemv_reference(w: &PackedWeights, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), w.cols);
    assert_eq!(y.len(), w.rows);
    let packed_cols = w.packed_cols();
    let scale_cols = w.scale_cols();
    let mut block = [0.0f32; SUB];
    // `r` indexes packed/scales (different strides) and y; the indexed form is
    // the clearest expression of the three-way row slicing.
    #[allow(clippy::needless_range_loop)]
    for r in 0..w.rows {
        let prow = &w.packed[r * packed_cols..(r + 1) * packed_cols];
        let srow = &w.scales[r * scale_cols..(r + 1) * scale_cols];
        let mut acc = 0.0f64;
        for b in 0..scale_cols {
            Nvfp4::dequant_block_scalar(
                &prow[b * PACKED_PER_BLOCK..(b + 1) * PACKED_PER_BLOCK],
                &srow[b..b + 1],
                &mut block,
            );
            let xblk = &x[b * SUB..(b + 1) * SUB];
            for k in 0..SUB {
                acc += block[k] as f64 * xblk[k] as f64;
            }
        }
        y[r] = (acc * w.output_scale as f64) as f32;
    }
}

// ── Fused dequant-dot for ONE row (the in-register core) ──────────────────────

/// Fused NVFP4 dequant-dot of one weight row against one input vector. Returns
/// `(sum_c dequant(W[r,c]) * x[c])` WITHOUT the global scale (the caller applies
/// it). Reads only `packed_row` (cols/2 bytes) + `scale_row` (cols/16 bytes).
///
/// Dispatches to the AVX-512 path when available, else a portable scalar fused
/// loop (still single-read-of-weights, just not vectorized).
#[inline]
fn fused_row_dot(packed_row: &[u8], scale_row: &[u8], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw") {
            // SAFETY: gated on runtime avx512f+avx512bw; all slices are sized to
            // `cols` (multiple of 16), and the kernel only indexes within them.
            return unsafe { fused_row_dot_avx512(packed_row, scale_row, x) };
        }
    }
    fused_row_dot_scalar(packed_row, scale_row, x)
}

/// Portable fused scalar fallback: unpack a block's nibbles into FP32 codes on
/// the stack, FMA against the input block, scale by the block scale. No full-row
/// materialization (only a 16-elem stack buffer), so DRAM weight reads stay 1×.
#[inline]
fn fused_row_dot_scalar(packed_row: &[u8], scale_row: &[u8], x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (b, &sbyte) in scale_row.iter().enumerate() {
        let block_scale = decode_ue4m3_with_half_lut(sbyte);
        let pbase = b * PACKED_PER_BLOCK;
        let xbase = b * SUB;
        let mut block_acc = 0.0f32;
        for j in 0..PACKED_PER_BLOCK {
            let byte = packed_row[pbase + j];
            block_acc += NVFP4_CODE_LUT_F32[(byte & 0x0f) as usize] * x[xbase + 2 * j];
            block_acc += NVFP4_CODE_LUT_F32[(byte >> 4) as usize] * x[xbase + 2 * j + 1];
        }
        acc += block_scale * block_acc;
    }
    acc
}

/// AVX-512 fused NVFP4 dequant-dot of one row. The 4-bit unpack + LUT gather +
/// scale + FMA happen entirely in zmm registers; the only RAM reads for the
/// weights are the 8 packed bytes + 1 scale byte per 16-element block.
///
/// Per block: explode 8 packed bytes → 16 nibble indices, `vpermps`-gather the
/// 16 signed integer codes from the LUT, multiply by the broadcast block scale,
/// then FMA `(codes*scale) * x` into a 16-lane FP32 accumulator. The horizontal
/// reduction happens ONCE per row (not per block), so the inner loop is pure
/// load + permute + FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
fn fused_row_dot_avx512(packed_row: &[u8], scale_row: &[u8], x: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    // SAFETY: caller guarantees avx512f+avx512bw at runtime; `scale_row` has one
    // byte per 16-element block, `packed_row` has 8 bytes per block, and `x` has
    // 16 f32 per block — every access below stays within those bounds.
    unsafe {
        // 16-entry integer-code LUT in a single zmm for vpermps gathers.
        let lut = _mm512_loadu_ps(NVFP4_CODE_LUT_F32.as_ptr());
        // pshufb control to DUPLICATE byte j into output bytes 2j and 2j+1:
        //   out[0..16] = [b0,b0,b1,b1,b2,b2,b3,b3,b4,b4,b5,b5,b6,b6,b7,b7]
        // After this, byte 2j and 2j+1 both equal packed byte j; a per-lane
        // variable shift then extracts the low nibble (even lanes) / high nibble
        // (odd lanes). All in-register: no stack round-trip on the hot path.
        let dup_ctrl = _mm_setr_epi8(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
        // Lane-shift pattern: even lanes >>0 (low nibble), odd lanes >>4 (high).
        let shift = _mm512_setr_epi32(0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4);
        let low_nibble_mask = _mm512_set1_epi32(0x0f);

        let mut acc = _mm512_setzero_ps();
        let nblocks = scale_row.len();
        // `b` indexes packed (×8), scales (×1) and x (×16) at different strides.
        #[allow(clippy::needless_range_loop)]
        for b in 0..nblocks {
            // --- read exactly 8 packed bytes (one block) from DRAM -------------
            let pbase = b * PACKED_PER_BLOCK;
            // Load the 8 packed bytes into the low 64 bits of an xmm register.
            let packed_xmm = _mm_loadl_epi64(packed_row.as_ptr().add(pbase) as *const _);
            // Duplicate byte j into lanes 2j,2j+1 (pshufb) → 16 bytes in-reg.
            let dup_xmm = _mm_shuffle_epi8(packed_xmm, dup_ctrl);
            // Zero-extend 16 u8 → 16 u32 lanes.
            let bytes32 = _mm512_cvtepu8_epi32(dup_xmm);
            // Per-lane variable right-shift, then mask to the nibble index.
            let shifted = _mm512_srlv_epi32(bytes32, shift);
            let idx = _mm512_and_si512(shifted, low_nibble_mask);

            // --- gather the 16 integer codes from the LUT (vpermps) -----------
            let codes = _mm512_permutexvar_ps(idx, lut);

            // --- scale by the per-block FP8 scale (broadcast) -----------------
            let block_scale = decode_ue4m3_with_half_lut(scale_row[b]);
            let scaled = _mm512_mul_ps(codes, _mm512_set1_ps(block_scale));

            // --- FMA against the 16 input lanes into the row accumulator ------
            let xv = _mm512_loadu_ps(x.as_ptr().add(b * SUB));
            acc = _mm512_fmadd_ps(scaled, xv, acc);
        }
        _mm512_reduce_add_ps(acc)
    }
}

// ── Public fused GEMV / GEMM (multi-threaded over rows) ───────────────────────

/// Fused NVFP4 dequant-GEMV, M=1: `y[rows] = W[rows,cols] · x[cols]`, threaded
/// over rows (rayon). Each row reads its packed bytes from DRAM exactly once and
/// dequants in-register. This is the per-token expert projection primitive.
pub fn gemv_into(w: &PackedWeights, x: &[f32], y: &mut [f32]) {
    assert_eq!(x.len(), w.cols, "input length must equal cols");
    assert_eq!(y.len(), w.rows, "output length must equal rows");
    let packed_cols = w.packed_cols();
    let scale_cols = w.scale_cols();
    let output_scale = w.output_scale;
    let packed = w.packed;
    let scales = w.scales;
    y.par_iter_mut().enumerate().for_each(|(r, slot)| {
        let prow = &packed[r * packed_cols..(r + 1) * packed_cols];
        let srow = &scales[r * scale_cols..(r + 1) * scale_cols];
        *slot = fused_row_dot(prow, srow, x) * output_scale;
    });
}

/// Fused NVFP4 dequant-GEMM for small M (the future MTP batched-verify path):
/// `Y[m, rows] = W[rows, cols] · X[m, cols]`, X and Y token-major. Each weight
/// row is read from DRAM ONCE and dotted against all `m` input tokens (the
/// row's packed bytes stay hot in L1/L2 across the inner token loop), so DRAM
/// weight traffic is independent of M — the whole reason to batch.
pub fn gemm_into(w: &PackedWeights, x: &[f32], m: usize, y: &mut [f32]) {
    if m == 0 {
        return;
    }
    if m == 1 {
        return gemv_into(w, x, y);
    }
    assert_eq!(x.len(), m * w.cols, "input length must equal m*cols");
    assert_eq!(y.len(), m * w.rows, "output length must equal m*rows");
    let packed_cols = w.packed_cols();
    let scale_cols = w.scale_cols();
    let rows = w.rows;
    let cols = w.cols;
    let output_scale = w.output_scale;
    let packed = w.packed;
    let scales = w.scales;

    // Row-parallel: compute the full output column for one weight row (all M
    // tokens) then scatter to the token-major Y. Reusing the row's bytes across
    // M tokens is the cache-blocking win.
    let row_cols: Vec<Vec<f32>> = (0..rows)
        .into_par_iter()
        .map(|r| {
            let prow = &packed[r * packed_cols..(r + 1) * packed_cols];
            let srow = &scales[r * scale_cols..(r + 1) * scale_cols];
            let mut out = vec![0.0f32; m];
            for (t, slot) in out.iter_mut().enumerate() {
                let xt = &x[t * cols..(t + 1) * cols];
                *slot = fused_row_dot(prow, srow, xt) * output_scale;
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

    const SEED: u64 = 0x5151_4E56_4650_3400; // "QNVFP4\0\0"

    /// Build a random valid NVFP4 packed weight matrix `[rows, cols]`: random
    /// nibbles (0..16) and random *valid* FP8 E4M3 scale bytes (avoiding the
    /// 0/0x7f NaN-ish codes so block_scale != 0 and the test exercises real
    /// magnitudes), plus a random positive global scale.
    fn random_packed(
        rng: &mut SmallRng,
        rows: usize,
        cols: usize,
    ) -> (Vec<u8>, Vec<u8>, f32) {
        let packed: Vec<u8> = (0..rows * cols / 2).map(|_| rng.random::<u8>()).collect();
        let scales: Vec<u8> = (0..rows * cols / SUB)
            .map(|_| {
                // Pick a 7-bit ue4m3 code in [0x10, 0x60] → block_scale in a
                // reasonable non-trivial range; never 0 or 0x7f.
                0x10u8 + (rng.random::<u8>() % 0x50)
            })
            .collect();
        let gscale = rng.random::<f32>() * 0.5 + 0.25;
        (packed, scales, gscale)
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
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

    /// Max relative error of `got` vs the f64-precision `reference`, normalized by
    /// the output RMS. Per-element relative error is meaningless for GEMV outputs
    /// near zero (catastrophic cancellation of ~K random terms), so we gate the
    /// denominator at the row's RMS magnitude — the standard way to score a dot
    /// product. This isolates true FP32 rounding error from cancellation noise.
    fn max_rel_err(reference: &[f32], got: &[f32]) -> f64 {
        let n = reference.len().max(1) as f64;
        let rms = (reference.iter().map(|&r| (r as f64) * (r as f64)).sum::<f64>() / n).sqrt();
        let denom = rms.max(1e-12);
        let mut worst = 0.0f64;
        for (&r, &g) in reference.iter().zip(got.iter()) {
            let e = (r as f64 - g as f64).abs() / denom;
            worst = worst.max(e);
        }
        worst
    }

    /// Representative Qwen3.6-35B-A3B expert projection shapes:
    /// gate/up = [moe_intermediate=512, hidden=2048]; down = [hidden=2048,
    /// moe_intermediate=512]. Plus small shapes that exercise the block edges.
    const EXPERT_SHAPES: &[(usize, usize, &str)] = &[
        (512, 2048, "gate/up [512x2048]"),
        (2048, 512, "down [2048x512]"),
        (2, 64, "tiny [2x64]"),
        (3, 16, "single-block rows [3x16]"),
        (17, 256, "odd rows [17x256]"),
        (128, 4096, "wide [128x4096]"),
    ];

    #[test]
    fn fused_gemv_matches_reference() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        for &(rows, cols, label) in EXPERT_SHAPES {
            let (packed, scales, gscale) = random_packed(&mut rng, rows, cols);
            let w = PackedWeights::new(rows, cols, &packed, &scales, gscale);
            let x: Vec<f32> = (0..cols).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();

            let mut y_ref = vec![0.0f32; rows];
            let mut y_fast = vec![0.0f32; rows];
            gemv_reference(&w, &x, &mut y_ref);
            gemv_into(&w, &x, &mut y_fast);

            let cos = cosine(&y_ref, &y_fast);
            let rel = max_rel_err(&y_ref, &y_fast);
            assert!(
                cos > 0.9999,
                "{label}: cosine {cos} <= 0.9999 (fused vs reference)"
            );
            assert!(
                rel < 1e-4,
                "{label}: max-rel-err {rel} >= 1e-4 (fused vs reference)"
            );
            eprintln!("{label}: cos={cos:.8} max_rel_err={rel:.3e}");
        }
    }

    /// The portable scalar fused path must also match the reference (covers
    /// machines without AVX-512 and pins the fallback's correctness).
    #[test]
    fn fused_scalar_path_matches_reference() {
        let mut rng = SmallRng::seed_from_u64(SEED ^ 0xABCD);
        for &(rows, cols, label) in EXPERT_SHAPES {
            let (packed, scales, gscale) = random_packed(&mut rng, rows, cols);
            let w = PackedWeights::new(rows, cols, &packed, &scales, gscale);
            let x: Vec<f32> = (0..cols).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
            let packed_cols = cols / 2;
            let scale_cols = cols / SUB;

            let mut y_ref = vec![0.0f32; rows];
            gemv_reference(&w, &x, &mut y_ref);
            let mut y_scalar = vec![0.0f32; rows];
            for r in 0..rows {
                let prow = &packed[r * packed_cols..(r + 1) * packed_cols];
                let srow = &scales[r * scale_cols..(r + 1) * scale_cols];
                y_scalar[r] = fused_row_dot_scalar(prow, srow, &x) * gscale;
            }
            let cos = cosine(&y_ref, &y_scalar);
            let rel = max_rel_err(&y_ref, &y_scalar);
            assert!(cos > 0.9999, "{label}: scalar-fused cosine {cos}");
            assert!(rel < 1e-4, "{label}: scalar-fused max-rel-err {rel}");
        }
    }

    /// Batched (small-M) GEMM must match per-token GEMV row-for-row.
    #[test]
    fn fused_gemm_matches_per_token_gemv() {
        let mut rng = SmallRng::seed_from_u64(SEED ^ 0x1234);
        let (rows, cols) = (512usize, 2048usize); // gate/up shape
        let (packed, scales, gscale) = random_packed(&mut rng, rows, cols);
        let w = PackedWeights::new(rows, cols, &packed, &scales, gscale);
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

    /// Sanity: all-zero nibbles → zero output regardless of scales.
    #[test]
    fn zero_nibbles_give_zero() {
        let (rows, cols) = (8usize, 256usize);
        let packed = vec![0u8; rows * cols / 2];
        let scales = vec![0x40u8; rows * cols / SUB];
        let w = PackedWeights::new(rows, cols, &packed, &scales, 0.75);
        let x: Vec<f32> = (0..cols).map(|i| i as f32 / 31.0 - 1.0).collect();
        let mut y = vec![1.0f32; rows];
        gemv_into(&w, &x, &mut y);
        assert!(y.iter().all(|&v| v == 0.0), "expected all-zero output");
    }

    // ── Microbench (ignored; run with `--release -- --ignored --nocapture`) ───
    //
    // Streams the full per-token Qwen3.6-35B-A3B active expert set
    // (8 experts × {gate,up,down} × 40 MoE layers ≈ 540 MiB packed) through the
    // fused kernel and reports achieved GB/s, GFLOP/s, and the RAM-bound vs
    // compute-bound verdict + projected per-token expert time and decode tps.

    struct ExpertSet {
        gate: (Vec<u8>, Vec<u8>),
        up: (Vec<u8>, Vec<u8>),
        down: (Vec<u8>, Vec<u8>),
    }

    fn build_expert(rng: &mut SmallRng, rows: usize, cols: usize) -> (Vec<u8>, Vec<u8>) {
        let (p, s, _g) = random_packed(rng, rows, cols);
        (p, s)
    }

    #[test]
    #[ignore]
    fn nvfp4_expert_microbench() {
        const HIDDEN: usize = 2048;
        const INTER: usize = 512;
        const EXPERTS_PER_TOK: usize = 8;
        const MOE_LAYERS: usize = 40;
        const GSCALE: f32 = 0.5;

        // Build ONE physical expert per (gate/up/down) and reuse its bytes for
        // all 8×40 active slots — we measure the kernel's throughput, and reusing
        // buffers keeps the test's own allocation modest while still issuing the
        // full active FLOP/byte workload. (For a pure cold-DRAM number we'd want
        // distinct buffers > LLC; see the "distinct buffers" variant below.)
        let mut rng = SmallRng::seed_from_u64(SEED);
        let expert = ExpertSet {
            gate: build_expert(&mut rng, INTER, HIDDEN), // [512, 2048]
            up: build_expert(&mut rng, INTER, HIDDEN),   // [512, 2048]
            down: build_expert(&mut rng, HIDDEN, INTER), // [2048, 512]
        };

        let x_hidden: Vec<f32> =
            (0..HIDDEN).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let x_inter: Vec<f32> = (0..INTER).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let mut y_inter_g = vec![0.0f32; INTER];
        let mut y_inter_u = vec![0.0f32; INTER];
        let mut y_hidden = vec![0.0f32; HIDDEN];

        let gate_w =
            PackedWeights::new(INTER, HIDDEN, &expert.gate.0, &expert.gate.1, GSCALE);
        let up_w = PackedWeights::new(INTER, HIDDEN, &expert.up.0, &expert.up.1, GSCALE);
        let down_w =
            PackedWeights::new(HIDDEN, INTER, &expert.down.0, &expert.down.1, GSCALE);

        // One "expert call" = gate + up + down GEMV.
        let run_expert = |yg: &mut [f32], yu: &mut [f32], yh: &mut [f32]| {
            gemv_into(&gate_w, &x_hidden, yg);
            gemv_into(&up_w, &x_hidden, yu);
            gemv_into(&down_w, &x_inter, yh);
        };

        // Bytes & FLOPs per expert call (packed weights + scale bytes read).
        let bytes_per_expert = expert.gate.0.len()
            + expert.gate.1.len()
            + expert.up.0.len()
            + expert.up.1.len()
            + expert.down.0.len()
            + expert.down.1.len();
        // 2 FLOP per weight element (multiply + add).
        let elems_per_expert = INTER * HIDDEN * 2 + HIDDEN * INTER; // gate+up+down weights
        let flops_per_expert = elems_per_expert * 2;

        let calls_per_token = EXPERTS_PER_TOK * MOE_LAYERS; // 320
        let total_bytes_per_token = bytes_per_expert * calls_per_token;
        let total_flops_per_token = flops_per_expert * calls_per_token;

        // Warm up.
        for _ in 0..20 {
            run_expert(&mut y_inter_g, &mut y_inter_u, &mut y_hidden);
        }

        // Time enough expert calls to amortize the timer; report per-token.
        const TOKENS: usize = 30;
        let total_calls = TOKENS * calls_per_token;
        let t0 = Instant::now();
        let mut sink = 0.0f32;
        for _ in 0..total_calls {
            run_expert(&mut y_inter_g, &mut y_inter_u, &mut y_hidden);
            sink += y_hidden[0] + y_inter_g[0] + y_inter_u[0];
        }
        let secs = t0.elapsed().as_secs_f64();

        let bytes_total = (bytes_per_expert * total_calls) as f64;
        let flops_total = (flops_per_expert * total_calls) as f64;
        let gbps = bytes_total / secs / 1e9;
        let gflops = flops_total / secs / 1e9;
        let per_token_ms = secs / TOKENS as f64 * 1e3;
        let implied_tps = 1.0 / (secs / TOKENS as f64);

        // Arithmetic intensity & roofline verdict. DDR5-class single-socket
        // dual-channel ceiling ~ 60-70 GB/s; Zen4 AVX-512 FP32 FMA peak is far
        // above what 3.7 FLOP/byte can use, so if we're near the DRAM ceiling the
        // kernel is bandwidth-bound (good: dequant hides under the read).
        let ai = flops_total / bytes_total; // FLOP/byte
        eprintln!("── NVFP4 fused expert-GEMV microbench (Qwen3.6-35B-A3B) ──");
        eprintln!(
            "per-token active set: {:.1} MiB packed+scales, {:.2} GFLOP ({} expert calls)",
            total_bytes_per_token as f64 / 1024.0 / 1024.0,
            total_flops_per_token as f64 / 1e9,
            calls_per_token
        );
        eprintln!("arithmetic intensity: {ai:.2} FLOP/byte");
        eprintln!(
            "achieved: {gbps:.1} GB/s, {gflops:.1} GFLOP/s   (sink={sink:.3})"
        );
        eprintln!(
            "projected per-token EXPERT compute: {per_token_ms:.2} ms  →  implied decode {implied_tps:.1} tps (experts only)"
        );
        eprintln!(
            "NOTE: buffers are reused (hot in cache for the larger shapes); see \
             nvfp4_expert_microbench_cold for a >LLC distinct-buffer number."
        );
        assert!(sink.is_finite());
    }

    /// Cold-ish variant: allocate a pool of DISTINCT expert buffers larger than
    /// the LLC (~32 MiB on Zen4) and round-robin through them, so each expert
    /// call reads weights that are NOT resident in cache — this is the honest
    /// "streaming from DRAM" number that the real decode path hits.
    #[test]
    #[ignore]
    fn nvfp4_expert_microbench_cold() {
        const HIDDEN: usize = 2048;
        const INTER: usize = 512;
        const EXPERTS_PER_TOK: usize = 8;
        const MOE_LAYERS: usize = 40;
        const GSCALE: f32 = 0.5;

        let mut rng = SmallRng::seed_from_u64(SEED ^ 0xC01D);
        // Build enough distinct experts that the working set >> LLC. One expert
        // ≈ 1.77 MiB; 64 experts ≈ 113 MiB >> 32 MiB LLC.
        const POOL: usize = 64;
        let experts: Vec<ExpertSet> = (0..POOL)
            .map(|_| ExpertSet {
                gate: build_expert(&mut rng, INTER, HIDDEN),
                up: build_expert(&mut rng, INTER, HIDDEN),
                down: build_expert(&mut rng, HIDDEN, INTER),
            })
            .collect();

        let x_hidden: Vec<f32> =
            (0..HIDDEN).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let x_inter: Vec<f32> = (0..INTER).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let mut yg = vec![0.0f32; INTER];
        let mut yu = vec![0.0f32; INTER];
        let mut yh = vec![0.0f32; HIDDEN];

        let bytes_per_expert = experts[0].gate.0.len()
            + experts[0].gate.1.len()
            + experts[0].up.0.len()
            + experts[0].up.1.len()
            + experts[0].down.0.len()
            + experts[0].down.1.len();
        let elems_per_expert = INTER * HIDDEN * 2 + HIDDEN * INTER;
        let flops_per_expert = elems_per_expert * 2;
        let calls_per_token = EXPERTS_PER_TOK * MOE_LAYERS;

        let run = |e: &ExpertSet, yg: &mut [f32], yu: &mut [f32], yh: &mut [f32]| {
            let g = PackedWeights::new(INTER, HIDDEN, &e.gate.0, &e.gate.1, GSCALE);
            let u = PackedWeights::new(INTER, HIDDEN, &e.up.0, &e.up.1, GSCALE);
            let d = PackedWeights::new(HIDDEN, INTER, &e.down.0, &e.down.1, GSCALE);
            gemv_into(&g, &x_hidden, yg);
            gemv_into(&u, &x_hidden, yu);
            gemv_into(&d, &x_inter, yh);
        };

        for i in 0..40 {
            run(&experts[i % POOL], &mut yg, &mut yu, &mut yh);
        }

        const TOKENS: usize = 30;
        let total_calls = TOKENS * calls_per_token;
        let mut sink = 0.0f32;
        let t0 = Instant::now();
        for i in 0..total_calls {
            run(&experts[i % POOL], &mut yg, &mut yu, &mut yh);
            sink += yh[0] + yg[0] + yu[0];
        }
        let secs = t0.elapsed().as_secs_f64();

        let bytes_total = (bytes_per_expert * total_calls) as f64;
        let flops_total = (flops_per_expert * total_calls) as f64;
        let gbps = bytes_total / secs / 1e9;
        let gflops = flops_total / secs / 1e9;
        let per_token_ms = secs / TOKENS as f64 * 1e3;
        let implied_tps = 1.0 / (secs / TOKENS as f64);
        let ai = flops_total / bytes_total;
        eprintln!("── NVFP4 fused expert-GEMV microbench (COLD, >LLC working set) ──");
        eprintln!("pool: {POOL} distinct experts ≈ {:.0} MiB", POOL as f64 * bytes_per_expert as f64 / 1024.0 / 1024.0);
        eprintln!("arithmetic intensity: {ai:.2} FLOP/byte");
        eprintln!("achieved: {gbps:.1} GB/s, {gflops:.1} GFLOP/s   (sink={sink:.3})");
        eprintln!(
            "projected per-token EXPERT compute: {per_token_ms:.2} ms  →  implied decode {implied_tps:.1} tps (experts only)"
        );
        assert!(sink.is_finite());
    }

    /// Single-thread, single-core kernel throughput: streams a >LLC pool of
    /// `down`-shaped rows through `fused_row_dot` with NO rayon, so it isolates
    /// the inner kernel's per-core bandwidth from threading overhead. Multiply by
    /// ~6 cores to estimate the parallel ceiling and compare to the multi-thread
    /// numbers above (if the MT number is well below 6×ST, threading/launch
    /// overhead dominates; if ST is itself far below DRAM/core, the dequant ALU
    /// is the limit).
    #[test]
    #[ignore]
    fn nvfp4_single_thread_kernel_bandwidth() {
        const COLS: usize = 2048;
        // Build a >LLC pool of distinct rows so reads come from DRAM, not cache.
        // One row = COLS/2 packed + COLS/16 scale ≈ 1.06 KiB; 64k rows ≈ 69 MiB.
        const ROWS: usize = 64 * 1024;
        let mut rng = SmallRng::seed_from_u64(SEED ^ 0x57);
        let packed: Vec<u8> = (0..ROWS * COLS / 2).map(|_| rng.random::<u8>()).collect();
        let scales: Vec<u8> = (0..ROWS * COLS / SUB)
            .map(|_| 0x10u8 + (rng.random::<u8>() % 0x50))
            .collect();
        let x: Vec<f32> = (0..COLS).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let packed_cols = COLS / 2;
        let scale_cols = COLS / SUB;
        let bytes_per_row = packed_cols + scale_cols;

        let kernel = |start: usize, count: usize| -> f32 {
            let mut s = 0.0f32;
            for r in start..start + count {
                let rr = r % ROWS;
                let prow = &packed[rr * packed_cols..(rr + 1) * packed_cols];
                let srow = &scales[rr * scale_cols..(rr + 1) * scale_cols];
                s += fused_row_dot(prow, srow, &x);
            }
            s
        };
        // Warm.
        let mut sink = kernel(0, ROWS);
        // Time several windows, keep the best (steady-state, turbo-warm).
        const WINDOW: usize = ROWS * 4;
        let mut best_gbps = 0.0f64;
        for w in 0..5 {
            let t0 = Instant::now();
            sink += kernel(w * 7, WINDOW);
            let secs = t0.elapsed().as_secs_f64();
            let gbps = (WINDOW * bytes_per_row) as f64 / secs / 1e9;
            best_gbps = best_gbps.max(gbps);
        }
        eprintln!("── NVFP4 fused kernel: SINGLE-THREAD per-core bandwidth ──");
        eprintln!(
            "best single-core: {best_gbps:.1} GB/s  (×6 cores ≈ {:.0} GB/s parallel ceiling)  sink={sink:.1}",
            best_gbps * 6.0
        );
        assert!(sink.is_finite());
    }

    /// Parallel ceiling in ONE big rayon region (no per-GEMV launch overhead):
    /// streams a >LLC pool of rows through `fused_row_dot` in a single
    /// `into_par_iter`, so it measures the true multi-core DRAM bandwidth the
    /// kernel can sustain. Compare to `nvfp4_expert_microbench_cold` (which does
    /// 320 small GEMV launches/token): a large gap means rayon per-launch
    /// overhead — not the kernel — caps the real per-token path, motivating a
    /// fused "all active experts in one parallel region" dispatch.
    #[test]
    #[ignore]
    fn nvfp4_parallel_ceiling_single_region() {
        const COLS: usize = 2048;
        const ROWS: usize = 64 * 1024; // ~69 MiB pool >> LLC
        let mut rng = SmallRng::seed_from_u64(SEED ^ 0xBEEF);
        let packed: Vec<u8> = (0..ROWS * COLS / 2).map(|_| rng.random::<u8>()).collect();
        let scales: Vec<u8> = (0..ROWS * COLS / SUB)
            .map(|_| 0x10u8 + (rng.random::<u8>() % 0x50))
            .collect();
        let x: Vec<f32> = (0..COLS).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
        let packed_cols = COLS / 2;
        let scale_cols = COLS / SUB;
        let bytes_per_row = packed_cols + scale_cols;

        let run = || -> f32 {
            (0..ROWS)
                .into_par_iter()
                .map(|r| {
                    let prow = &packed[r * packed_cols..(r + 1) * packed_cols];
                    let srow = &scales[r * scale_cols..(r + 1) * scale_cols];
                    fused_row_dot(prow, srow, &x)
                })
                .sum()
        };
        let mut sink = run();
        let mut best_gbps = 0.0f64;
        for _ in 0..6 {
            let t0 = Instant::now();
            sink += run();
            let secs = t0.elapsed().as_secs_f64();
            best_gbps = best_gbps.max((ROWS * bytes_per_row) as f64 / secs / 1e9);
        }
        eprintln!("── NVFP4 fused kernel: PARALLEL ceiling (single rayon region) ──");
        eprintln!("best parallel: {best_gbps:.1} GB/s  sink={sink:.1}");
        assert!(sink.is_finite());
    }
}

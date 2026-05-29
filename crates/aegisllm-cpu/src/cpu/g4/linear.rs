//! Unified CPU linear projection for Gemma-4: BF16 (E2B/E4B/31B-dense checkpoints)
//! or NVFP4 (quantized checkpoints). Mirrors the CUDA `CudaLinear` enum
//! (`crates/aegisllm-cuda/src/executor/state.rs:47`), but only the two storage
//! formats the CPU path supports.
//!
//! Single-token GEMV keeps the rayon + in-register BF16-widen dot
//! (`bf16_matvec_fast`). The batched GEMM (`matmul_into`) uses a blocked SIMD
//! kernel: on AVX512_BF16 hardware (Zen 4 / Sapphire Rapids) it computes
//! NATIVELY in BF16 via `VDPBF16PS` (bf16×bf16 → f32-accumulate, ~2× the f32
//! FMA rate AND no widen), exactly like llama.cpp's CPU prefill. On any other
//! CPU it falls back to a portable f32 outer-product kernel that widens the
//! BF16 weights to f32. Both produce row-major `[batch, rows]` output.

use crate::cpu::CpuNvfp4Linear;
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::executor::tensors::Bf16Matrix;
use rayon::prelude::*;

/// Fast BF16×f32 GEMV: `out[r] = Σ_c W[r,c]*input[c]`, parallel over output rows,
/// each row a SIMD bf16-widen+FMA dot (no full-matrix f32 copy). `bytes` is the
/// row-major `[rows, cols]` LE-BF16 weight; `out.len()` == rows.
fn bf16_matvec_fast(bytes: &[u8], cols: usize, input: &[f32], out: &mut [f32]) {
    let row_bytes = cols * 2;
    out.par_iter_mut().enumerate().for_each(|(r, o)| {
        let row = &bytes[r * row_bytes..(r + 1) * row_bytes];
        *o = crate::cpu::simd::dot_bf16_f32(row, input);
    });
}

#[derive(Debug)]
pub(crate) enum CpuLinear {
    Bf16(Bf16Matrix),
    Nvfp4(CpuNvfp4Linear),
}

impl CpuLinear {
    pub(crate) fn rows(&self) -> usize {
        match self {
            Self::Bf16(m) => m.rows,
            Self::Nvfp4(l) => l.rows,
        }
    }

    // Part of the documented CpuLinear contract; used by the batched-prefill
    // follow-up and shape validation.
    #[allow(dead_code)]
    pub(crate) fn cols(&self) -> usize {
        match self {
            Self::Bf16(m) => m.cols,
            Self::Nvfp4(l) => l.cols,
        }
    }

    /// Single-vector projection: `out[r] = Σ_c W[r,c] * input[c]`.
    pub(crate) fn matvec_into(&self, input: &[f32], out: &mut [f32]) -> Result<()> {
        match self {
            Self::Bf16(m) => {
                if input.len() != m.cols || out.len() != m.rows {
                    return Err(AegisError::InvalidPlan(format!(
                        "bf16 matvec shape mismatch for {}: input={} cols={} output={} rows={}",
                        m.name(), input.len(), m.cols, out.len(), m.rows
                    )));
                }
                // Fast path: rayon over rows + SIMD bf16-widen+FMA dot, reading the
                // BF16 weights once from DRAM (no per-call full-matrix f32 copy).
                bf16_matvec_fast(m.weight_bytes(), m.cols, input, out);
                Ok(())
            }
            Self::Nvfp4(l) => l.matvec_into(input, out),
        }
    }

    /// Batched projection over `batch` tokens. Input/output are row-major
    /// `[batch, cols]` / `[batch, rows]`. The BF16 path runs a blocked SIMD GEMM
    /// (native BF16 VNNI when available, f32 fallback otherwise); the NVFP4 path
    /// dequantizes each weight row once and dots all tokens.
    #[allow(dead_code)]
    pub(crate) fn matmul_into(&self, input: &[f32], batch: usize, out: &mut [f32]) -> Result<()> {
        match self {
            Self::Bf16(m) => {
                let cols = m.cols;
                let rows = m.rows;
                if input.len() != batch * cols || out.len() != batch * rows {
                    return Err(AegisError::InvalidPlan(format!(
                        "bf16 matmul shape mismatch: expected input={} output={} (batch={} rows={} cols={})",
                        batch * cols,
                        batch * rows,
                        batch,
                        rows,
                        cols
                    )));
                }
                bf16_matmul_fast(m.weight_bytes(), rows, cols, input, batch, out);
                Ok(())
            }
            Self::Nvfp4(l) => l.matmul_into(input, batch, out),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Batched BF16 GEMM
//
// Computes `out[b, r] = Σ_c W[r, c] * input[b, c]` where `W` is row-major
// `[rows, cols]` LE-BF16 and `input`/`out` are row-major `[batch, cols]` /
// `[batch, rows]`. Internally we work in the transposed layout `temp[r, b]`
// (rows × batch) — the natural VNNI/outer-product output shape — then transpose
// to the caller's `[batch, rows]`.
//
// Output f32 lanes map to 16 token columns. `NB = ceil(batch / 16)` is the
// number of 16-wide column blocks; with `batch ≤ 64` that's ≤ 4.
// ─────────────────────────────────────────────────────────────────────────────

const COL_LANES: usize = 16; // f32 lanes in a ZMM == token columns per block
/// Microkernel row tiling (weight rows handled together; share the packed input).
/// MR=6 keeps MR*NB (≤24) accumulators + NB input panels inside the 32 ZMM file;
/// MR≥8 spills and tanks throughput (measured on Zen 4).
const MR: usize = 6;
/// Column-blocks per N-tile in the large-batch kernel: NB_TILE=4 → 64 tokens per
/// N-block. The `[MR][NB_TILE]` f32 accumulator grid is `6*4 = 24` ZMM, plus 4
/// input panels + 1 weight broadcast = 29 ≤ 32 — stays register-resident.
const NB_TILE: usize = 4;
/// K-pairs per K-block in the large-batch kernel (cache blocking for the WIDE
/// case). A K-block slices the weight panel to `MR * 2*KC_PAIRS * 2` bytes; with
/// KC_PAIRS=1024 (=2048 cols) that's `6*2048*2 = 24 KiB`, fitting the 32 KiB L1d
/// so the slice stays L1-resident while sweeping all N-blocks of the panel.
const KC_PAIRS: usize = 1024;

#[inline]
fn col_blocks(batch: usize) -> usize {
    batch.div_ceil(COL_LANES)
}

/// Round f32 → bf16 bits with round-to-nearest-even.
///
/// We use RNE (not the top-16-bit truncation `Bf16Matrix` uses for weights):
/// the weights are already stored as bf16, so the only new rounding here is the
/// f32 *activations* → bf16, and RNE halves the rounding bias vs truncation for
/// the same cost. Documented in NUMERICS below.
#[inline(always)]
fn f32_to_bf16_rne(x: f32) -> u16 {
    let bits = x.to_bits();
    // round-half-to-even on the 16 discarded low bits
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

/// Pack f32 activations `input[batch, cols]` into the VNNI input layout, rounding
/// to bf16. Layout (raw bf16 bits, 2 contiguous bf16 == one 32-bit word):
///
///   packed[(kp * NB + blk) * 32 + c * 2 + s]
///     = bf16(input[(blk*16 + c) * cols + 2*kp + s])   for s ∈ {0, 1}
///
/// i.e. for a given k-pair `kp` and 16-col block `blk`, the 16 columns' (k, k+1)
/// bf16 pairs are laid out as 32 contiguous bf16 — exactly one `__m512bh`. The
/// last column block is zero-padded for `batch % 16 != 0`; an odd `cols` leaves
/// the final k-pair's high bf16 zero (kp covers `ceil(cols/2)` pairs).
fn pack_input_bf16(input: &[f32], batch: usize, cols: usize) -> Vec<u16> {
    let nb = col_blocks(batch);
    let kpairs = cols.div_ceil(2);
    let mut packed = vec![0u16; kpairs * nb * 32];
    // Parallelize over k-pairs: each writes a disjoint contiguous region.
    packed
        .par_chunks_mut(nb * 32)
        .enumerate()
        .for_each(|(kp, dst)| {
            let k0 = 2 * kp;
            let k1 = k0 + 1;
            let has_k1 = k1 < cols;
            for blk in 0..nb {
                let base = blk * 32;
                for c in 0..COL_LANES {
                    let col = blk * COL_LANES + c;
                    if col >= batch {
                        break; // remaining lanes stay zero (masked on store)
                    }
                    let row = &input[col * cols..col * cols + cols];
                    dst[base + c * 2] = f32_to_bf16_rne(row[k0]);
                    if has_k1 {
                        dst[base + c * 2 + 1] = f32_to_bf16_rne(row[k1]);
                    }
                }
            }
        });
    packed
}

/// Dispatch the batched BF16 GEMM. Picks the native-BF16 VNNI kernel when the
/// CPU advertises AVX512_BF16, otherwise the portable f32-widen fallback.
/// `pub(crate)` so the batched PLE path (`g4::ple`) can drive raw `Bf16Matrix`
/// weights through the same kernel without wrapping them in `CpuLinear`.
pub(crate) fn bf16_matmul_fast(
    weight: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
    batch: usize,
    out: &mut [f32],
) {
    if batch == 0 || rows == 0 {
        return;
    }
    if batch == 1 {
        // Degenerate batch: the GEMV path is already optimal and avoids packing.
        bf16_matvec_fast(weight, cols, input, out);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512bf16") {
            // SAFETY: gated on runtime avx512f+avx512bf16 detection.
            unsafe {
                bf16_matmul_vnni(weight, rows, cols, input, batch, out);
            }
            return;
        }
    }

    bf16_matmul_f32_fallback(weight, rows, cols, input, batch, out);
}

// ─────────────────────────────────────────────────────────────────────────────
// f32 fallback kernel (portable; runs when AVX512_BF16 is absent)
//
// Widens the BF16 weights to f32 on the fly and does a register-blocked
// outer-product FMA: a single weight scalar is broadcast across a 16-wide tile of
// token columns. This is the same blocking the VNNI kernel uses (so the two share
// the pack + transpose), but with plain f32 arrays the autovectorizer maps it to
// AVX512 / AVX2 / NEON on whatever target lacks AVX512_BF16. Correctness path:
// the GEMM never depends on a CPU feature for *correctness*, only for speed.
// ─────────────────────────────────────────────────────────────────────────────

fn bf16_matmul_f32_fallback(
    weight: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
    batch: usize,
    out: &mut [f32],
) {
    // Pack input into f32 column-tiles: panel[(k*nt + blk)*16 + c] = input[col, k]
    // where col = blk*16 + c (zero-padded tail). One broadcast-FMA tile per (k,blk).
    let nt = col_blocks(batch);
    let mut panels = vec![0.0f32; cols * nt * COL_LANES];
    panels
        .par_chunks_mut(nt * COL_LANES)
        .enumerate()
        .for_each(|(k, dst)| {
            for blk in 0..nt {
                let base = blk * COL_LANES;
                for c in 0..COL_LANES {
                    let col = blk * COL_LANES + c;
                    if col >= batch {
                        break;
                    }
                    dst[base + c] = input[col * cols + k];
                }
            }
        });

    let mut temp = vec![0.0f32; rows * batch];
    let row_bytes = cols * 2;
    let nthreads = rayon::current_num_threads().max(1);
    let micro_tiles = rows.div_ceil(MR);
    let tiles_per_chunk = micro_tiles.div_ceil(nthreads);
    let chunk_rows = (tiles_per_chunk * MR).max(MR);

    macro_rules! dispatch_f32 {
        ($nt:literal) => {
            temp.par_chunks_mut(chunk_rows * batch)
                .enumerate()
                .for_each(|(ci, tchunk)| {
                    let r_base = ci * chunk_rows;
                    let chunk_n = tchunk.len() / batch;
                    let mut ro = 0;
                    while ro + MR <= chunk_n {
                        f32_strip::<MR, $nt>(
                            weight, r_base + ro, cols, row_bytes, &panels, batch,
                            &mut tchunk[ro * batch..(ro + MR) * batch],
                        );
                        ro += MR;
                    }
                    while ro < chunk_n {
                        f32_strip::<1, $nt>(
                            weight, r_base + ro, cols, row_bytes, &panels, batch,
                            &mut tchunk[ro * batch..(ro + 1) * batch],
                        );
                        ro += 1;
                    }
                })
        };
    }
    match nt {
        1 => dispatch_f32!(1),
        2 => dispatch_f32!(2),
        3 => dispatch_f32!(3),
        4 => dispatch_f32!(4),
        _ => f32_strip_generic(weight, cols, row_bytes, &panels, nt, batch, &mut temp),
    }
    transpose_into(&temp, rows, batch, out);
}

/// Portable register-blocked f32 micro-kernel: `MR` weight rows × `NT` 16-col
/// blocks into `tchunk` (`[MR, batch]`). Plain f32 arrays so it autovectorizes
/// to AVX512 / AVX2 / NEON; `MR` and `NT` const so accumulators stay in regs.
#[inline(always)]
fn f32_strip<const MR: usize, const NT: usize>(
    weight: &[u8],
    r0: usize,
    cols: usize,
    row_bytes: usize,
    panels: &[f32],
    batch: usize,
    tchunk: &mut [f32],
) {
    use aegisllm_base::executor::tensors::bf16_to_f32;
    let mut acc = [[[0.0f32; COL_LANES]; NT]; MR];
    for k in 0..cols {
        let mut w = [0.0f32; MR];
        for (r, wr) in w.iter_mut().enumerate() {
            let off = (r0 + r) * row_bytes + k * 2;
            *wr = bf16_to_f32(u16::from_le_bytes([weight[off], weight[off + 1]]));
        }
        for blk in 0..NT {
            let pbase = (k * NT + blk) * COL_LANES;
            let panel = &panels[pbase..pbase + COL_LANES];
            for r in 0..MR {
                let wr = w[r];
                let a = &mut acc[r][blk];
                for c in 0..COL_LANES {
                    a[c] += wr * panel[c];
                }
            }
        }
    }
    for (r, acc_row) in acc.iter().enumerate() {
        let base = r * batch;
        for (blk, a) in acc_row.iter().enumerate() {
            let col0 = blk * COL_LANES;
            let n = (batch - col0).min(COL_LANES);
            tchunk[base + col0..base + col0 + n].copy_from_slice(&a[..n]);
        }
    }
}

/// Scalar f32 fallback for batch > 64 (nt > 4): one row at a time, all columns.
/// Rare in the prefill path; correctness over speed.
fn f32_strip_generic(
    weight: &[u8],
    cols: usize,
    row_bytes: usize,
    panels: &[f32],
    nt: usize,
    batch: usize,
    temp: &mut [f32],
) {
    use aegisllm_base::executor::tensors::bf16_to_f32;
    temp.par_chunks_mut(batch).enumerate().for_each(|(r, trow)| {
        for (col, t) in trow.iter_mut().enumerate() {
            let blk = col / COL_LANES;
            let c = col % COL_LANES;
            let mut acc = 0.0f32;
            for k in 0..cols {
                let off = r * row_bytes + k * 2;
                let w = bf16_to_f32(u16::from_le_bytes([weight[off], weight[off + 1]]));
                acc += w * panels[(k * nt + blk) * COL_LANES + c];
            }
            *t = acc;
        }
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Native BF16 VNNI kernel (AVX512_BF16 / VDPBF16PS)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn bf16_matmul_vnni(
    weight: &[u8],
    rows: usize,
    cols: usize,
    input: &[f32],
    batch: usize,
    out: &mut [f32],
) {
    let nb = col_blocks(batch);
    let kpairs = cols.div_ceil(2);
    // input[batch, cols] f32 → packed bf16 in VNNI layout (once per matmul).
    let packed = pack_input_bf16(input, batch, cols);

    // Weight as u16 (LE-bf16). `weight` is 2*rows*cols bytes; reinterpret as a
    // `&[u16]` so it's `Sync` for the rayon closure (raw `*const u16` is not).
    debug_assert_eq!(weight.len(), rows * cols * 2);
    // SAFETY: bf16 weights are 2-byte aligned LE-u16; len is exactly 2*rows*cols.
    let wu16: &[u16] =
        unsafe { std::slice::from_raw_parts(weight.as_ptr() as *const u16, rows * cols) };

    let mut temp = vec![0.0f32; rows * batch];

    // Coarse row chunking: one chunk per rayon worker (rounded to MR), each doing
    // many MR-row micro-tiles. Fine-grained `par_chunks_mut(MR*batch)` would spawn
    // rows/MR tiny tasks and the scheduler overhead alone caps throughput at
    // ~single-thread; coarse chunks scale near-linearly across cores.
    let nthreads = rayon::current_num_threads().max(1);
    let micro_tiles = rows.div_ceil(MR);
    let tiles_per_chunk = micro_tiles.div_ceil(nthreads);
    let chunk_rows = (tiles_per_chunk * MR).max(MR);

    if nb <= NB_TILE {
        // Small batch (≤ NB_TILE*16 = 64 tokens): the whole token range fits one
        // N-block, so the existing single-pass kernel already reads each weight
        // panel from DRAM exactly once. Dispatch `nb` to a const generic so the
        // `MR*nb` accumulators + `nb` panels stay register-resident (a runtime
        // `nb` keeps the accumulators in memory and tanks throughput).
        macro_rules! dispatch_nb {
            ($nb:literal) => {
                temp.par_chunks_mut(chunk_rows * batch)
                    .enumerate()
                    .for_each(|(ci, tchunk)| {
                        let r_base = ci * chunk_rows;
                        let chunk_n = tchunk.len() / batch;
                        let mut ro = 0;
                        // SAFETY: avx512f+avx512bf16 verified by caller; in bounds.
                        unsafe {
                            while ro + MR <= chunk_n {
                                let dst = &mut tchunk[ro * batch..(ro + MR) * batch];
                                kernel_strip::<MR, $nb>(
                                    wu16, r_base + ro, cols, kpairs, &packed, batch, dst,
                                );
                                ro += MR;
                            }
                            while ro < chunk_n {
                                let dst = &mut tchunk[ro * batch..(ro + 1) * batch];
                                kernel_strip::<1, $nb>(
                                    wu16, r_base + ro, cols, kpairs, &packed, batch, dst,
                                );
                                ro += 1;
                            }
                        }
                    })
            };
        }
        match nb {
            1 => dispatch_nb!(1),
            2 => dispatch_nb!(2),
            3 => dispatch_nb!(3),
            _ => dispatch_nb!(4),
        }
    } else {
        // Large batch (> 64 tokens): a single kernel that reads each weight panel
        // from DRAM ONCE and reuses it across ALL N-blocks. For each MR-row panel
        // (rayon over panels), sweep the `nb` 16-token blocks in NB_TILE-wide
        // N-blocks; the panel (or, when wide, each KC_PAIRS K-slice of it) stays
        // L1/L2-resident so N-blocks after the first hit cache instead of DRAM.
        temp.par_chunks_mut(chunk_rows * batch)
            .enumerate()
            .for_each(|(ci, tchunk)| {
                let r_base = ci * chunk_rows;
                let chunk_n = tchunk.len() / batch;
                let mut ro = 0;
                // SAFETY: avx512f+avx512bf16 verified by caller; in bounds.
                unsafe {
                    while ro + MR <= chunk_n {
                        let dst = &mut tchunk[ro * batch..(ro + MR) * batch];
                        kernel_strip_large::<MR>(
                            wu16, r_base + ro, cols, kpairs, &packed, nb, batch, dst,
                        );
                        ro += MR;
                    }
                    while ro < chunk_n {
                        let dst = &mut tchunk[ro * batch..(ro + 1) * batch];
                        kernel_strip_large::<1>(
                            wu16, r_base + ro, cols, kpairs, &packed, nb, batch, dst,
                        );
                        ro += 1;
                    }
                }
            });
    }

    transpose_into(&temp, rows, batch, out);
}

/// Large-batch micro-kernel: `MR` weight rows over ALL `nb` 16-token column-blocks
/// (`batch` tokens), reading the weight panel from DRAM ONCE and reusing it across
/// every N-block. Writes `tchunk` (row-major `[MR, batch]`).
///
/// Loop nest (mirrors llama.cpp's blocked GEMM order so the panel stays cached):
///   for N-block (NB_TILE=4 col-blocks = 64 tokens):
///       acc[MR][nb_this] = 0                    ← ZMM-resident across all K
///       for kc-block (KC_PAIRS k-pairs):        ← cache blocking for wide cols
///           for kp in kc-range:
///               load nb_this input panels; broadcast each row's weight word; dpbf16
///       store acc → tchunk
///
/// The accumulator grid for an N-block persists in registers across all kc-blocks
/// (so a wide row's partials are summed before being written out). Because the
/// weight panel — or, when `cols > 2*KC_PAIRS`, each KC_PAIRS slice of it — is
/// touched again on the next N-block while still cache-resident, DRAM reads it
/// once per panel regardless of `batch`. Full NB_TILE-wide N-blocks pin the panel
/// count at compile time (NB_TILE accumulators + panels); the final partial
/// N-block (`batch % 64 != 0`) is handled by the const-generic `kernel_strip`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn kernel_strip_large<const MR: usize>(
    wu16: &[u16],
    r0: usize,
    cols: usize,
    kpairs: usize,
    packed: &[u16],
    nb: usize,
    batch: usize,
    tchunk: &mut [f32],
) {
    use std::arch::x86_64::*;

    // SAFETY: target_feature verified by caller; all indexing stays within
    // `wu16` (rows*cols), `packed` (kpairs*nb*32) and `tchunk` (MR*batch);
    // stores past `batch` in the final 16-lane block are masked.
    unsafe {
        let packed_ptr = packed.as_ptr();
        let wptr = wu16.as_ptr();

        // Sweep full NB_TILE-wide N-blocks (64 tokens each). Only count groups of
        // NB_TILE FULLY-16-wide column-blocks: `full16 = batch / 16` is the number
        // of complete 16-token blocks, so `full16 / NB_TILE` N-blocks are entirely
        // in-bounds (16-lane unmasked stores). Any leftover full blocks plus the
        // final partial block (`batch % 16 != 0`) go to the masked tail — this is
        // why the split is on `full16`, NOT `nb`: when `nb % NB_TILE == 0` but
        // `batch % 16 != 0`, the last block is partial yet sits in the final
        // NB_TILE group, and an unmasked store there would write past `batch`.
        let full16 = batch / COL_LANES;
        let full_nblocks = full16 / NB_TILE;
        for nblk in 0..full_nblocks {
            let blk0 = nblk * NB_TILE; // first 16-token column-block of this N-block
            let mut acc = [[_mm512_setzero_ps(); NB_TILE]; MR];

            // K-blocking: each kc-block touches a KC_PAIRS-pair slice of the panel
            // (≤ 24 KiB), L1-resident while we sweep this N-block's K range. For
            // narrow `cols` (≤ 2*KC_PAIRS) this is a single pass over K.
            let mut kc0 = 0;
            while kc0 < kpairs {
                let kc1 = (kc0 + KC_PAIRS).min(kpairs);
                for kp in kc0..kc1 {
                    // Load this N-block's NB_TILE input panels for k-pair `kp`.
                    let mut b_panels = [_mm512_setzero_si512(); NB_TILE];
                    for (j, panel) in b_panels.iter_mut().enumerate() {
                        let off = (kp * nb + blk0 + j) * 32;
                        *panel = _mm512_loadu_si512(packed_ptr.add(off) as *const _);
                    }
                    for r in 0..MR {
                        let widx = (r0 + r) * cols + 2 * kp;
                        let word: u32 = if 2 * kp + 1 < cols {
                            (wptr.add(widx) as *const u32).read_unaligned()
                        } else {
                            *wptr.add(widx) as u32
                        };
                        let a_bh: __m512bh = std::mem::transmute(_mm512_set1_epi32(word as i32));
                        for j in 0..NB_TILE {
                            let b_bh: __m512bh = std::mem::transmute(b_panels[j]);
                            acc[r][j] = _mm512_dpbf16_ps(acc[r][j], a_bh, b_bh);
                        }
                    }
                }
                kc0 = kc1;
            }

            // Store this N-block's 64 token columns (no partial block — every full
            // N-block ends within `batch`, so all NB_TILE blocks are 16 wide).
            for (r, acc_row) in acc.iter().enumerate() {
                let trow_base = r * batch + blk0 * COL_LANES;
                for (j, a) in acc_row.iter().enumerate() {
                    _mm512_storeu_ps(tchunk.as_mut_ptr().add(trow_base + j * COL_LANES), *a);
                }
            }
        }

        // Tail: any leftover column-blocks after the full NB_TILE groups
        // (`nb - full_nblocks*NB_TILE`, which is 1..=NB_TILE — the leftover full
        // blocks plus the possibly-partial final block). The const-generic tail
        // kernel masks the final block's store; its weight re-read is a single
        // panel over a short token range, negligible vs the full-N-block sweeps.
        let tail_blocks = nb - full_nblocks * NB_TILE;
        if tail_blocks != 0 {
            let blk0 = full_nblocks * NB_TILE;
            let col0 = blk0 * COL_LANES;
            match tail_blocks {
                1 => kernel_strip_tail::<MR, 1>(
                    wu16, r0, cols, kpairs, packed, nb, blk0, batch, col0, tchunk,
                ),
                2 => kernel_strip_tail::<MR, 2>(
                    wu16, r0, cols, kpairs, packed, nb, blk0, batch, col0, tchunk,
                ),
                3 => kernel_strip_tail::<MR, 3>(
                    wu16, r0, cols, kpairs, packed, nb, blk0, batch, col0, tchunk,
                ),
                _ => kernel_strip_tail::<MR, 4>(
                    wu16, r0, cols, kpairs, packed, nb, blk0, batch, col0, tchunk,
                ),
            }
        }
    }
}

/// Tail handler for the large-batch kernel: computes `TB` (1..=3) trailing 16-token
/// column-blocks starting at global block `blk0` (token column `col0`) for `MR`
/// rows, writing into `tchunk[r*batch + col0 ..]` with the final block masked to
/// `batch`. Same VNNI math as `kernel_strip`, but indexed against the full-batch
/// packing (`kp*nb + blk0 + j`) and writing at the column offset.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn kernel_strip_tail<const MR: usize, const TB: usize>(
    wu16: &[u16],
    r0: usize,
    cols: usize,
    kpairs: usize,
    packed: &[u16],
    nb: usize,
    blk0: usize,
    batch: usize,
    col0: usize,
    tchunk: &mut [f32],
) {
    use std::arch::x86_64::*;
    // SAFETY: target_feature verified by caller; indexing stays within `packed`
    // (kpairs*nb*32), `wu16` (rows*cols) and `tchunk` (MR*batch); the final
    // 16-lane store is masked to the live token count.
    unsafe {
        let packed_ptr = packed.as_ptr();
        let wptr = wu16.as_ptr();
        let mut acc = [[_mm512_setzero_ps(); TB]; MR];
        for kp in 0..kpairs {
            let mut b_panels = [_mm512_setzero_si512(); TB];
            for (j, panel) in b_panels.iter_mut().enumerate() {
                let off = (kp * nb + blk0 + j) * 32;
                *panel = _mm512_loadu_si512(packed_ptr.add(off) as *const _);
            }
            for r in 0..MR {
                let widx = (r0 + r) * cols + 2 * kp;
                let word: u32 = if 2 * kp + 1 < cols {
                    (wptr.add(widx) as *const u32).read_unaligned()
                } else {
                    *wptr.add(widx) as u32
                };
                let a_bh: __m512bh = std::mem::transmute(_mm512_set1_epi32(word as i32));
                for j in 0..TB {
                    let b_bh: __m512bh = std::mem::transmute(b_panels[j]);
                    acc[r][j] = _mm512_dpbf16_ps(acc[r][j], a_bh, b_bh);
                }
            }
        }
        for (r, acc_row) in acc.iter().enumerate() {
            let trow_base = r * batch + col0;
            for (j, a) in acc_row.iter().enumerate() {
                let cglob = col0 + j * COL_LANES;
                let n = (batch - cglob).min(COL_LANES);
                let dst = tchunk.as_mut_ptr().add(trow_base + j * COL_LANES);
                if n == COL_LANES {
                    _mm512_storeu_ps(dst, *a);
                } else {
                    let mask: u16 = ((1u32 << n) - 1) as u16;
                    _mm512_mask_storeu_ps(dst, mask, *a);
                }
            }
        }
    }
}

/// Compute `MR` weight rows × `NB` column-blocks (`NB*16` token columns) into
/// `tchunk` (row-major `[MR, batch]`). `acc[r][blk]` holds 16 token columns of
/// row `r`; both `MR` and `NB` are const so the `MR*NB` accumulators and the
/// panel array live entirely in ZMM registers.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bf16")]
unsafe fn kernel_strip<const MR: usize, const NB: usize>(
    wu16: &[u16],
    r0: usize,
    cols: usize,
    kpairs: usize,
    packed: &[u16],
    batch: usize,
    tchunk: &mut [f32],
) {
    use std::arch::x86_64::*;

    // SAFETY: target_feature avx512f+avx512bf16 verified by the caller; all
    // pointer arithmetic stays within `wu16` (rows*cols), `packed`
    // (kpairs*NB*32) and `tchunk` (MR*batch), and stores past `batch` are masked.
    unsafe {
        let mut acc = [[_mm512_setzero_ps(); NB]; MR];

        let packed_ptr = packed.as_ptr();
        let wptr = wu16.as_ptr();

        for kp in 0..kpairs {
            // Load the NB input panels for this k-pair once.
            let mut b_panels = [_mm512_setzero_si512(); NB];
            for (blk, panel) in b_panels.iter_mut().enumerate() {
                let off = (kp * NB + blk) * 32;
                // 32 u16 == 512 bits.
                *panel = _mm512_loadu_si512(packed_ptr.add(off) as *const _);
            }
            // For each row, broadcast its (k, k+1) weight word and dpbf16 into acc.
            for r in 0..MR {
                let row = r0 + r;
                // 32-bit word = two contiguous bf16 at &W[row*cols + 2*kp].
                let widx = row * cols + 2 * kp;
                let word: u32 = if 2 * kp + 1 < cols {
                    // two valid bf16 — read the 32-bit word in one load.
                    (wptr.add(widx) as *const u32).read_unaligned()
                } else {
                    // odd cols tail: high bf16 must be zero (pad).
                    *wptr.add(widx) as u32
                };
                let a = _mm512_set1_epi32(word as i32);
                let a_bh: __m512bh = std::mem::transmute(a);
                for blk in 0..NB {
                    let b_bh: __m512bh = std::mem::transmute(b_panels[blk]);
                    acc[r][blk] = _mm512_dpbf16_ps(acc[r][blk], a_bh, b_bh);
                }
            }
        }

        // Store acc → tchunk[r*batch + col]. Mask the last (partial) col block.
        for (r, acc_row) in acc.iter().enumerate() {
            let trow_base = r * batch;
            for blk in 0..NB {
                let col0 = blk * COL_LANES;
                let n = (batch - col0).min(COL_LANES);
                let dst = tchunk.as_mut_ptr().add(trow_base + col0);
                if n == COL_LANES {
                    _mm512_storeu_ps(dst, acc_row[blk]);
                } else {
                    let mask: u16 = ((1u32 << n) - 1) as u16;
                    _mm512_mask_storeu_ps(dst, mask, acc_row[blk]);
                }
            }
        }
    }
}

/// Transpose `temp[rows, batch]` (row-major) into `out[batch, rows]` (row-major).
fn transpose_into(temp: &[f32], rows: usize, batch: usize, out: &mut [f32]) {
    // Parallelize over output rows (= tokens); each reads a strided column of temp.
    out.par_chunks_mut(rows)
        .enumerate()
        .for_each(|(b, orow)| {
            for (r, o) in orow.iter_mut().enumerate() {
                *o = temp[r * batch + b];
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    const SEED: u64 = 0x5151_4242_7373_9090;

    /// Build a row-major `[rows, cols]` LE-bf16 weight (top-16-bit truncation,
    /// matching how `Bf16Matrix` stores weights) and the f32 reference values.
    fn make_bf16_weight(rng: &mut SmallRng, rows: usize, cols: usize) -> (Vec<u8>, Vec<f32>) {
        let mut bytes = Vec::with_capacity(rows * cols * 2);
        let mut widened = vec![0f32; rows * cols];
        for w in widened.iter_mut() {
            let f: f32 = rng.random_range(-2.0..2.0);
            let bf = (f.to_bits() >> 16) as u16; // truncate to bf16 (weight storage)
            bytes.extend_from_slice(&bf.to_le_bytes());
            *w = f32::from_bits((bf as u32) << 16);
        }
        (bytes, widened)
    }

    /// Reference: pure f32 matmul `out[b,r] = Σ_c W[r,c]*input[b,c]` with the
    /// (already-bf16) weights kept in f32. Activations are NOT rounded — this is
    /// the looped `matvec` reference the kernel is checked against.
    fn reference(
        widened: &[f32],
        rows: usize,
        cols: usize,
        input: &[f32],
        batch: usize,
    ) -> Vec<f32> {
        let mut out = vec![0f32; batch * rows];
        for b in 0..batch {
            for r in 0..rows {
                let mut acc = 0f32;
                for c in 0..cols {
                    acc += widened[r * cols + c] * input[b * cols + c];
                }
                out[b * rows + r] = acc;
            }
        }
        out
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let mut dot = 0f64;
        let mut na = 0f64;
        let mut nb = 0f64;
        for (&x, &y) in a.iter().zip(b.iter()) {
            dot += x as f64 * y as f64;
            na += (x as f64) * (x as f64);
            nb += (y as f64) * (y as f64);
        }
        dot / (na.sqrt() * nb.sqrt())
    }

    fn rand_vec(rng: &mut SmallRng, n: usize) -> Vec<f32> {
        (0..n).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect()
    }

    /// VNNI kernel (or f32 fallback when AVX512_BF16 absent) vs the f32 reference.
    /// Activations are rounded to bf16, so we require cosine > 0.999 (NOT max-abs).
    #[test]
    fn bf16_matmul_into_matches_reference_cosine() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        // Mix of shapes: square, E2B-ish, odd cols, odd rows, batch not %16.
        let shapes: &[(usize, usize, usize)] = &[
            (16, 16, 16),
            (32, 64, 8),
            (33, 65, 17), // odd rows, odd cols, batch not multiple of 16
            (128, 256, 64),
            (1536, 256, 40),
            (64, 1536, 31),
            // Large-batch (> 64) coverage: exercises kernel_strip_large + the
            // partial-N-block tail and K-blocking (wide cols=6144 > 2*KC_PAIRS).
            (256, 512, 128),  // exact 2 N-blocks
            (200, 300, 200),  // batch not a multiple of 64 (tail = 8 tokens)
            (300, 6144, 256), // wide cols → K-blocking, batch=256
            (1536, 6144, 512), // E2B `down` at full large batch
            (6144, 1536, 512), // E2B `gate/up` at full large batch
            (130, 257, 333),  // odd rows/cols + batch not %16 and not %64
            // Tail-masking edge cases for kernel_strip_large's full/tail split:
            (96, 128, 250),   // nb=16 (nb%NB_TILE==0) but batch%16!=0 → partial
            //                   block sits in the last NB_TILE group; must be MASKED.
            (96, 128, 180),   // full16=11 → tail_blocks=4 (3 full + 1 partial block)
            (96, 128, 240),   // full16=15 → tail_blocks=3, all 16-wide
            (96, 128, 49),    // small >64: nb=4 full + ... actually nb=4 → small path
        ];
        for &(rows, cols, batch) in shapes {
            let (bytes, widened) = make_bf16_weight(&mut rng, rows, cols);
            let input = rand_vec(&mut rng, batch * cols);
            let mut out = vec![0f32; batch * rows];
            bf16_matmul_fast(&bytes, rows, cols, &input, batch, &mut out);
            let refv = reference(&widened, rows, cols, &input, batch);
            let cos = cosine(&out, &refv);
            println!("cosine[rows={rows} cols={cols} batch={batch}] = {cos:.6}");
            assert!(
                cos > 0.999,
                "rows={rows} cols={cols} batch={batch}: cosine={cos} < 0.999"
            );
        }
    }

    /// argmax (greedy-token) stability per row vs the f32 reference. bf16
    /// activations shouldn't flip the dominant output for well-separated values.
    #[test]
    fn bf16_matmul_into_argmax_stable() {
        let mut rng = SmallRng::seed_from_u64(SEED ^ 0xABCD);
        let (rows, cols, batch) = (256usize, 512usize, 32usize);
        let (bytes, widened) = make_bf16_weight(&mut rng, rows, cols);
        let input = rand_vec(&mut rng, batch * cols);
        let mut out = vec![0f32; batch * rows];
        bf16_matmul_fast(&bytes, rows, cols, &input, batch, &mut out);
        let refv = reference(&widened, rows, cols, &input, batch);
        let argmax = |v: &[f32]| -> usize {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap()
        };
        let mut flips = 0;
        for b in 0..batch {
            let o = &out[b * rows..(b + 1) * rows];
            let rf = &refv[b * rows..(b + 1) * rows];
            if argmax(o) != argmax(rf) {
                flips += 1;
            }
        }
        // Allow a tiny number of ties to flip; with random data flips are rare.
        assert!(
            flips <= 1,
            "argmax flipped on {flips}/{batch} rows (bf16 activation rounding)"
        );
    }

    /// The portable f32-widen fallback must match the reference on its own (it is
    /// what runs on non-AVX512_BF16 CPUs and must be correct everywhere).
    #[test]
    fn bf16_matmul_into_f32_fallback_matches_reference() {
        let mut rng = SmallRng::seed_from_u64(SEED ^ 0x1357);
        let shapes: &[(usize, usize, usize)] = &[(33, 65, 17), (128, 256, 40)];
        for &(rows, cols, batch) in shapes {
            let (bytes, widened) = make_bf16_weight(&mut rng, rows, cols);
            let input = rand_vec(&mut rng, batch * cols);
            let mut out = vec![0f32; batch * rows];
            bf16_matmul_f32_fallback(&bytes, rows, cols, &input, batch, &mut out);
            let refv = reference(&widened, rows, cols, &input, batch);
            // f32 fallback keeps weights bf16 but activations f32 → near-exact.
            let cos = cosine(&out, &refv);
            assert!(cos > 0.9999, "rows={rows} cols={cols} batch={batch}: cosine={cos}");
        }
    }

    /// matmul_into with batch=1 must equal the GEMV path (degenerate dispatch).
    #[test]
    fn bf16_matmul_batch_one_matches_gemv() {
        let mut rng = SmallRng::seed_from_u64(SEED ^ 0x2468);
        let (rows, cols) = (200usize, 300usize);
        let (bytes, _widened) = make_bf16_weight(&mut rng, rows, cols);
        let input = rand_vec(&mut rng, cols);
        let mut out_mm = vec![0f32; rows];
        let mut out_gemv = vec![0f32; rows];
        bf16_matmul_fast(&bytes, rows, cols, &input, 1, &mut out_mm);
        bf16_matvec_fast(&bytes, cols, &input, &mut out_gemv);
        for r in 0..rows {
            assert_eq!(out_mm[r], out_gemv[r], "row {r}");
        }
    }

    #[test]
    fn f32_to_bf16_rne_roundtrips_exact_bf16() {
        // Exact bf16 values (low 16 bits zero) round to themselves.
        for &v in &[1.0f32, -2.0, 0.5, 0.0, 123.0] {
            let bf = f32_to_bf16_rne(v);
            let back = f32::from_bits((bf as u32) << 16);
            assert_eq!(back, v, "v={v}");
        }
        // RNE rounds the half-way bit to even, not always up.
        let just_above_one = f32::from_bits(0x3f80_8000); // 1.0 + 0.5ulp(bf16)
        let bf = f32_to_bf16_rne(just_above_one);
        // ties-to-even: mantissa lsb of 1.0's bf16 is 0 → rounds down to 1.0.
        assert_eq!(bf, 0x3f80, "ties-to-even should keep 1.0");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Microbenchmark: run with
    //   cargo test -p aegisllm-cpu --release bf16_gemm_microbench -- --ignored --nocapture
    // Prints f32-fallback vs native-BF16-VNNI GFLOP/s for the E2B MLP shapes,
    // plus the input-pack cost.
    // ─────────────────────────────────────────────────────────────────────────
    #[test]
    #[ignore]
    fn bf16_gemm_microbench() {
        use std::time::Instant;

        fn bench_shape(name: &str, rows: usize, cols: usize, batch: usize) {
            let mut rng = SmallRng::seed_from_u64(0xBEEF);
            let bytes: Vec<u8> = (0..rows * cols * 2).map(|_| rng.random::<u8>()).collect();
            let input: Vec<f32> =
                (0..batch * cols).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
            let mut out = vec![0f32; batch * rows];
            // 2 FLOP per MAC.
            let flops = 2.0 * rows as f64 * cols as f64 * batch as f64;

            // Prior baseline: per-token GEMV loop (what `matmul_into` used to do).
            let gemv_loop = |bytes: &[u8], out: &mut [f32]| {
                for token in 0..batch {
                    let in_row = &input[token * cols..(token + 1) * cols];
                    let out_row = &mut out[token * rows..(token + 1) * rows];
                    bf16_matvec_fast(bytes, cols, in_row, out_row);
                }
            };
            for _ in 0..3 {
                gemv_loop(&bytes, &mut out);
            }
            let iters = 50;
            let t = Instant::now();
            for _ in 0..iters {
                gemv_loop(&bytes, &mut out);
            }
            let gemv_secs = t.elapsed().as_secs_f64() / iters as f64;
            let gemv_gflops = flops / gemv_secs / 1e9;

            // Prior large-batch strategy: split into ≤64-token VNNI sub-calls.
            // This re-reads the WHOLE weight from DRAM once per 64-token chunk —
            // the bandwidth waste the single-large-batch kernel removes. (For
            // batch ≤ 64 it's a single call == the new path.)
            let split64 = |bytes: &[u8], out: &mut [f32]| {
                let mut b0 = 0;
                while b0 < batch {
                    let n = (batch - b0).min(64);
                    let in_chunk = &input[b0 * cols..(b0 + n) * cols];
                    let out_chunk = &mut out[b0 * rows..(b0 + n) * rows];
                    bf16_matmul_fast(bytes, rows, cols, in_chunk, n, out_chunk);
                    b0 += n;
                }
            };
            for _ in 0..3 {
                split64(&bytes, &mut out);
            }
            let t = Instant::now();
            for _ in 0..iters {
                split64(&bytes, &mut out);
            }
            let split_secs = t.elapsed().as_secs_f64() / iters as f64;
            let split_gflops = flops / split_secs / 1e9;

            // warm up + time the portable f32-widen blocked fallback
            for _ in 0..3 {
                bf16_matmul_f32_fallback(&bytes, rows, cols, &input, batch, &mut out);
            }
            let t = Instant::now();
            for _ in 0..iters {
                bf16_matmul_f32_fallback(&bytes, rows, cols, &input, batch, &mut out);
            }
            let f32_secs = t.elapsed().as_secs_f64() / iters as f64;
            let f32_gflops = flops / f32_secs / 1e9;

            // time native VNNI if available, else note absence.
            #[cfg(target_arch = "x86_64")]
            let have_bf16 = std::is_x86_feature_detected!("avx512bf16");
            #[cfg(not(target_arch = "x86_64"))]
            let have_bf16 = false;

            if have_bf16 {
                for _ in 0..3 {
                    bf16_matmul_fast(&bytes, rows, cols, &input, batch, &mut out);
                }
                let t = Instant::now();
                for _ in 0..iters {
                    bf16_matmul_fast(&bytes, rows, cols, &input, batch, &mut out);
                }
                let v_secs = t.elapsed().as_secs_f64() / iters as f64;
                let v_gflops = flops / v_secs / 1e9;

                // isolate pack cost
                let t = Instant::now();
                for _ in 0..iters {
                    let p = pack_input_bf16(&input, batch, cols);
                    std::hint::black_box(&p);
                }
                let pack_us = t.elapsed().as_secs_f64() / iters as f64 * 1e6;

                println!(
                    "[{name}] rows={rows} cols={cols} batch={batch}  \
                     GEMV-loop={gemv_gflops:7.1}  \
                     VNNI-64split(before)={split_gflops:7.1}  \
                     VNNI-large(after)={v_gflops:7.1} GFLOP/s  \
                     (after vs 64split {:.2}x)  \
                     pack={pack_us:.1}us ({:.0}% of step)",
                    v_gflops / split_gflops,
                    pack_us / (v_secs * 1e6) * 100.0
                );
                let _ = f32_gflops;
            } else {
                println!(
                    "[{name}] rows={rows} cols={cols} batch={batch}  \
                     prior-GEMV-loop={gemv_gflops:7.1}  \
                     f32-blocked={f32_gflops:7.1} GFLOP/s  (no AVX512_BF16 on this CPU)"
                );
            }
        }

        println!();
        // E2B MLP shapes across the large-batch range. The batch ≥ 256 numbers
        // are the ones that matter (weight-DRAM-read amortized over many tokens);
        // a flat/rising curve toward bf16 peak (~1.9 TFLOP/s) confirms the weight
        // is read from DRAM ONCE per panel rather than once per 64-token sub-call.
        for &batch in &[64usize, 128, 256, 512] {
            bench_shape("gate/up", 6144, 1536, batch);
        }
        for &batch in &[64usize, 128, 256, 512] {
            bench_shape("down", 1536, 6144, batch);
        }
        // A weight that does NOT fit L3 (144 MiB ≫ 32 MiB): here the old 64-split
        // RE-READS the weight from DRAM per chunk while the large-batch kernel
        // reads it ONCE — the gap is the DRAM-read-once win the E2B shapes hide
        // (their 18.9 MiB weight stays L3-resident across the split chunks).
        for &batch in &[256usize, 512] {
            bench_shape("L3-spill", 12288, 6144, batch);
        }
    }
}

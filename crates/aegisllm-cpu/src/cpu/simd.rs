use pulp::{Arch, Simd};
use std::cell::RefCell;
use std::sync::OnceLock;

pub fn arch() -> Arch {
    static ARCH: OnceLock<Arch> = OnceLock::new();
    *ARCH.get_or_init(Arch::new)
}

// ── dot product ──────────────────────────────────────────────────────────────

struct DotF32<'a> {
    a: &'a [f32],
    b: &'a [f32],
}

impl pulp::WithSimd for DotF32<'_> {
    type Output = f32;
    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> f32 {
        let (a_head, a_tail) = S::as_simd_f32s(self.a);
        let (b_head, b_tail) = S::as_simd_f32s(self.b);
        let mut acc = simd.splat_f32s(0.0);
        for (&av, &bv) in a_head.iter().zip(b_head.iter()) {
            acc = simd.mul_add_f32s(av, bv, acc);
        }
        let mut sum = simd.reduce_sum_f32s(acc);
        for (&av, &bv) in a_tail.iter().zip(b_tail.iter()) {
            sum += av * bv;
        }
        sum
    }
}

pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    arch().dispatch(DotF32 { a: &a[..len], b: &b[..len] })
}

// ── bf16 widen + dot ──────────────────────────────────────────────────────────
//
// A BF16 value is the top 16 bits of the equivalent f32 bit pattern, so widening
// is exact: `f32::from_bits((bf16 as u32) << 16)`. We never materialize the whole
// weight matrix as f32 — instead we widen one BF16 weight row (or a small K-block)
// into a reusable f32 scratch, then SIMD-dot it against the f32 input. Callers
// reuse one row's scratch across an entire GEMM batch, so each weight byte is read
// from DRAM exactly once (the cache-blocking win on a memory-bound kernel).

/// Widen `cols` little-endian BF16 values (raw bytes, `2*cols` long) into the f32
/// `out` slice (`cols` long). Pure shift + reinterpret — the loop auto-vectorizes
/// and is bounded by the 2-byte/elem DRAM read, which is the bandwidth we want to
/// hit. This is the ONLY BF16→f32 expansion in the fast path.
#[inline]
pub fn widen_bf16_row_into(row_bytes: &[u8], out: &mut [f32]) {
    let n = out.len().min(row_bytes.len() / 2);
    let pairs = &row_bytes[..n * 2];
    for (dst, chunk) in out[..n].iter_mut().zip(pairs.chunks_exact(2)) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]) as u32;
        *dst = f32::from_bits(bits << 16);
    }
}

/// Fused BF16×f32 dot: widen one BF16 weight row in small K-blocks (kept in L1)
/// and SIMD-FMA each block against the matching slice of the f32 input, so a row
/// is never fully materialized as f32 and the reduction is vectorized. Used by the
/// single-token GEMV path where there is no batch to amortize a full-row widen.
pub fn dot_bf16_f32(row_bytes: &[u8], input: &[f32]) -> f32 {
    const BLOCK: usize = 256; // 1 KiB f32 scratch — comfortably L1-resident.
    let cols = input.len().min(row_bytes.len() / 2);
    let mut acc = 0.0_f32;
    let mut scratch = [0.0_f32; BLOCK];
    let mut col = 0;
    while col < cols {
        let len = (cols - col).min(BLOCK);
        widen_bf16_row_into(&row_bytes[col * 2..(col + len) * 2], &mut scratch[..len]);
        acc += dot_f32(&scratch[..len], &input[col..col + len]);
        col += len;
    }
    acc
}

// ── scale in place ───────────────────────────────────────────────────────────

struct ScaleInPlace<'a> {
    out: &'a mut [f32],
    scale: f32,
}

impl pulp::WithSimd for ScaleInPlace<'_> {
    type Output = ();
    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let sv = simd.splat_f32s(self.scale);
        let (head, tail) = S::as_mut_simd_f32s(self.out);
        for chunk in head.iter_mut() {
            *chunk = simd.mul_f32s(*chunk, sv);
        }
        for v in tail.iter_mut() {
            *v *= self.scale;
        }
    }
}

pub fn scale_in_place(out: &mut [f32], scale: f32) {
    arch().dispatch(ScaleInPlace { out, scale });
}

// ── axpy: out += weight * v ───────────────────────────────────────────────────

struct Axpy<'a> {
    out: &'a mut [f32],
    v: &'a [f32],
    weight: f32,
}

impl pulp::WithSimd for Axpy<'_> {
    type Output = ();
    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let wv = simd.splat_f32s(self.weight);
        let (out_head, out_tail) = S::as_mut_simd_f32s(self.out);
        let (v_head, v_tail) = S::as_simd_f32s(self.v);
        for (o, &vv) in out_head.iter_mut().zip(v_head.iter()) {
            *o = simd.mul_add_f32s(wv, vv, *o);
        }
        for (o, &vv) in out_tail.iter_mut().zip(v_tail.iter()) {
            *o += self.weight * vv;
        }
    }
}

pub fn axpy(out: &mut [f32], v: &[f32], weight: f32) {
    let len = out.len().min(v.len());
    arch().dispatch(Axpy { out: &mut out[..len], v: &v[..len], weight });
}

// ── add_into ─────────────────────────────────────────────────────────────────

struct AddIntoSimd<'a> {
    a: &'a [f32],
    b: &'a [f32],
    out: &'a mut [f32],
}

impl pulp::WithSimd for AddIntoSimd<'_> {
    type Output = ();
    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let (a_head, a_tail) = S::as_simd_f32s(self.a);
        let (b_head, b_tail) = S::as_simd_f32s(self.b);
        let (out_head, out_tail) = S::as_mut_simd_f32s(self.out);
        for ((&av, &bv), o) in a_head.iter().zip(b_head.iter()).zip(out_head.iter_mut()) {
            *o = simd.add_f32s(av, bv);
        }
        for ((&av, &bv), o) in a_tail.iter().zip(b_tail.iter()).zip(out_tail.iter_mut()) {
            *o = av + bv;
        }
    }
}

pub fn add_into_simd(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = a.len().min(b.len()).min(out.len());
    arch().dispatch(AddIntoSimd { a: &a[..len], b: &b[..len], out: &mut out[..len] });
}

// ── add in-place: a += b ─────────────────────────────────────────────────────

struct AddInPlace<'a> {
    a: &'a mut [f32],
    b: &'a [f32],
}

impl pulp::WithSimd for AddInPlace<'_> {
    type Output = ();
    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let (a_head, a_tail) = S::as_mut_simd_f32s(self.a);
        let (b_head, b_tail) = S::as_simd_f32s(self.b);
        for (a, &bv) in a_head.iter_mut().zip(b_head.iter()) {
            *a = simd.add_f32s(*a, bv);
        }
        for (a, &bv) in a_tail.iter_mut().zip(b_tail.iter()) {
            *a += bv;
        }
    }
}

pub fn add_in_place(a: &mut [f32], b: &[f32]) {
    let len = a.len().min(b.len());
    arch().dispatch(AddInPlace { a: &mut a[..len], b: &b[..len] });
}

// ── swiglu ───────────────────────────────────────────────────────────────────

pub fn swiglu_into_simd(gate: &[f32], up: &[f32], out: &mut [f32]) {
    let len = gate.len().min(up.len()).min(out.len());
    for i in 0..len {
        out[i] = silu_scalar(gate[i]) * up[i];
    }
}

#[inline(always)]
fn silu_scalar(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// ── GeGLU (Gemma-4 / Qwen MLP activation) ────────────────────────────────────
// out = gelu_pytorch_tanh(gate) * up, matching HF `gelu_pytorch_tanh`:
//   0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))

#[inline(always)]
pub fn gelu_tanh_scalar(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_56; // sqrt(2/pi)
    let inner = SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

pub fn geglu_into_simd(gate: &[f32], up: &[f32], out: &mut [f32]) {
    let len = gate.len().min(up.len()).min(out.len());
    for i in 0..len {
        out[i] = gelu_tanh_scalar(gate[i]) * up[i];
    }
}

// ── RoPE pair ────────────────────────────────────────────────────────────────

pub fn rope_apply_pair(row: &mut [f32], cos_table: &[f32], sin_table: &[f32]) {
    let half = row.len() / 2;
    let len = half.min(cos_table.len()).min(sin_table.len());
    for i in 0..len {
        let x0 = row[i];
        let x1 = row[i + half];
        let c = cos_table[i];
        let s = sin_table[i];
        row[i] = x0 * c - x1 * s;
        row[i + half] = x0 * s + x1 * c;
    }
}

// ── RMS norm scale pass ───────────────────────────────────────────────────────

struct RmsScale<'a> {
    x: &'a [f32],
    w: &'a [f32],
    out: &'a mut [f32],
    scale: f32,
}

impl pulp::WithSimd for RmsScale<'_> {
    type Output = ();
    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let sv = simd.splat_f32s(self.scale);
        let (x_head, x_tail) = S::as_simd_f32s(self.x);
        let (w_head, w_tail) = S::as_simd_f32s(self.w);
        let (out_head, out_tail) = S::as_mut_simd_f32s(self.out);
        for ((&xv, &wv), o) in x_head.iter().zip(w_head.iter()).zip(out_head.iter_mut()) {
            *o = simd.mul_f32s(simd.mul_f32s(xv, sv), wv);
        }
        for ((&xv, &wv), o) in x_tail.iter().zip(w_tail.iter()).zip(out_tail.iter_mut()) {
            *o = xv * self.scale * wv;
        }
    }
}

pub fn rms_scale(x: &[f32], w: &[f32], out: &mut [f32], scale: f32) {
    let len = x.len().min(w.len()).min(out.len());
    arch().dispatch(RmsScale { x: &x[..len], w: &w[..len], out: &mut out[..len], scale });
}

// ── NVFP4 helpers (available for future wiring into linear.rs) ────────────────

thread_local! {
    static DEQUANT_SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

#[allow(dead_code)]
pub fn nvfp4_unpacked_block_dot(weights_i8: &[i8], input: &[f32], block_scale: f32) -> f32 {
    let n = weights_i8.len().min(input.len());
    let mut acc = 0.0_f32;
    for (w, v) in weights_i8[..n].iter().zip(input[..n].iter()) {
        acc += *w as f32 * *v;
    }
    acc * block_scale
}

#[allow(dead_code)]
pub fn packed_nvfp4_row_dot(
    packed_row: &[u8],
    input: &[f32],
    decode_nibble: impl Fn(u8) -> f32,
    block_size: usize,
    decode_block_scale: impl Fn(u8) -> f32,
    scale_row: &[u8],
) -> f32 {
    let cols = input.len();
    DEQUANT_SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        scratch.resize(cols, 0.0_f32);
        let scale_cols = cols / block_size;
        for (block_idx, &scale_byte) in scale_row[..scale_cols].iter().enumerate() {
            let block_scale = decode_block_scale(scale_byte);
            let input_base = block_idx * block_size;
            let packed_base = block_idx * (block_size / 2);
            for j in 0..(block_size / 2) {
                let lo = input_base + j * 2;
                let hi = lo + 1;
                if hi < cols {
                    let byte = packed_row[packed_base + j];
                    scratch[lo] = decode_nibble(byte & 0x0f) * block_scale;
                    scratch[hi] = decode_nibble(byte >> 4) * block_scale;
                }
            }
        }
        dot_f32(&scratch, input)
    })
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    const LENGTHS: &[usize] = &[0, 1, 7, 16, 17, 31, 1024, 4096];
    const SEED: u64 = 0xDEAD_BEEF_1234_5678;

    #[test]
    fn dot_bf16_f32_matches_widened_reference() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        for &cols in &[1usize, 7, 16, 31, 256, 257, 1024, 2560] {
            // Build BF16 weight bytes (top-16-bits truncation of random f32) + f32 input.
            let mut bytes = Vec::with_capacity(cols * 2);
            let mut widened = vec![0f32; cols];
            for w in widened.iter_mut() {
                let f: f32 = rng.gen_range(-2.0..2.0);
                let bf = (f.to_bits() >> 16) as u16; // truncate to bf16
                bytes.extend_from_slice(&bf.to_le_bytes());
                *w = f32::from_bits((bf as u32) << 16);
            }
            let input = rand_vec(&mut rng, cols);
            let fast = dot_bf16_f32(&bytes, &input);
            let reference = dot_f32(&widened, &input); // dot of explicitly-widened weights
            assert!(
                (fast - reference).abs() <= 1e-3 * (reference.abs() + 1.0),
                "cols={cols}: fast={fast} reference={reference}"
            );
        }
    }

    fn rand_vec(rng: &mut SmallRng, n: usize) -> Vec<f32> {
        (0..n).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect()
    }

    fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn dot_f32_matches_scalar() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        for &n in LENGTHS {
            let a = rand_vec(&mut rng, n);
            let b = rand_vec(&mut rng, n);
            let expected = dot_scalar(&a, &b);
            let got = dot_f32(&a, &b);
            if n == 0 {
                assert_eq!(got, 0.0);
            } else {
                assert!(
                    (got - expected).abs() < 1e-5 * (1.0 + expected.abs()),
                    "len={n} got={got} expected={expected}"
                );
            }
        }
    }

    #[test]
    fn scale_in_place_matches_scalar() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        for &n in LENGTHS {
            let a = rand_vec(&mut rng, n);
            let scale = rng.random::<f32>() + 0.1;
            let mut got = a.clone();
            scale_in_place(&mut got, scale);
            for i in 0..n {
                assert!((got[i] - a[i] * scale).abs() < 1e-5, "len={n} i={i}");
            }
        }
    }

    #[test]
    fn axpy_matches_scalar() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        for &n in LENGTHS {
            let v = rand_vec(&mut rng, n);
            let mut out = rand_vec(&mut rng, n);
            let weight = rng.random::<f32>();
            let expected: Vec<f32> = out.iter().zip(&v).map(|(o, &vi)| o + weight * vi).collect();
            axpy(&mut out, &v, weight);
            for i in 0..n {
                assert!((out[i] - expected[i]).abs() < 1e-5, "len={n} i={i}");
            }
        }
    }

    #[test]
    fn add_into_simd_is_exact() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        for &n in LENGTHS {
            let a = rand_vec(&mut rng, n);
            let b = rand_vec(&mut rng, n);
            let mut out = vec![0.0_f32; n];
            add_into_simd(&a, &b, &mut out);
            for i in 0..n {
                assert_eq!(out[i], a[i] + b[i], "len={n} i={i}");
            }
        }
    }

    #[test]
    fn swiglu_matches_scalar() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        for &n in LENGTHS {
            let gate = rand_vec(&mut rng, n);
            let up = rand_vec(&mut rng, n);
            let mut out = vec![0.0_f32; n];
            swiglu_into_simd(&gate, &up, &mut out);
            for i in 0..n {
                let expected = (gate[i] / (1.0 + (-gate[i]).exp())) * up[i];
                assert!(
                    (out[i] - expected).abs() < 1e-4,
                    "len={n} i={i} got={} expected={}",
                    out[i],
                    expected
                );
            }
        }
    }

    #[test]
    fn rms_scale_matches_scalar() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        for &n in LENGTHS {
            let x = rand_vec(&mut rng, n);
            let w = rand_vec(&mut rng, n);
            let scale = rng.random::<f32>() + 0.1;
            let mut out = vec![0.0_f32; n];
            rms_scale(&x, &w, &mut out, scale);
            for i in 0..n {
                let expected = x[i] * scale * w[i];
                assert!((out[i] - expected).abs() < 1e-5, "len={n} i={i}");
            }
        }
    }

    #[test]
    fn rope_apply_pair_matches_scalar() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        for &half in &[0usize, 4, 16, 64] {
            let n = half * 2;
            let mut row = rand_vec(&mut rng, n);
            let cos_t: Vec<f32> = (0..half).map(|_| rng.random::<f32>()).collect();
            let sin_t: Vec<f32> = (0..half).map(|_| rng.random::<f32>()).collect();
            let orig = row.clone();
            rope_apply_pair(&mut row, &cos_t, &sin_t);
            for i in 0..half {
                let expected0 = orig[i] * cos_t[i] - orig[i + half] * sin_t[i];
                let expected1 = orig[i] * sin_t[i] + orig[i + half] * cos_t[i];
                assert!((row[i] - expected0).abs() < 1e-5, "half={half} i={i}");
                assert!(
                    (row[i + half] - expected1).abs() < 1e-5,
                    "half={half} i+half={}",
                    i + half
                );
            }
        }
    }
}

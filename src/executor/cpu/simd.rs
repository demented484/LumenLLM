use pulp::{Arch, Simd};
use std::cell::RefCell;
use std::sync::OnceLock;

pub(crate) fn arch() -> Arch {
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

pub(crate) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    arch().dispatch(DotF32 { a: &a[..len], b: &b[..len] })
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

pub(crate) fn scale_in_place(out: &mut [f32], scale: f32) {
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

pub(crate) fn axpy(out: &mut [f32], v: &[f32], weight: f32) {
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

pub(crate) fn add_into_simd(a: &[f32], b: &[f32], out: &mut [f32]) {
    let len = a.len().min(b.len()).min(out.len());
    arch().dispatch(AddIntoSimd { a: &a[..len], b: &b[..len], out: &mut out[..len] });
}

// ── swiglu ───────────────────────────────────────────────────────────────────

pub(crate) fn swiglu_into_simd(gate: &[f32], up: &[f32], out: &mut [f32]) {
    let len = gate.len().min(up.len()).min(out.len());
    for i in 0..len {
        out[i] = silu_scalar(gate[i]) * up[i];
    }
}

#[inline(always)]
fn silu_scalar(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// ── RoPE pair ────────────────────────────────────────────────────────────────

pub(crate) fn rope_apply_pair(row: &mut [f32], cos_table: &[f32], sin_table: &[f32]) {
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

pub(crate) fn rms_scale(x: &[f32], w: &[f32], out: &mut [f32], scale: f32) {
    let len = x.len().min(w.len()).min(out.len());
    arch().dispatch(RmsScale { x: &x[..len], w: &w[..len], out: &mut out[..len], scale });
}

// ── NVFP4 helpers (available for future wiring into linear.rs) ────────────────

thread_local! {
    static DEQUANT_SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

#[allow(dead_code)]
pub(crate) fn nvfp4_unpacked_block_dot(weights_i8: &[i8], input: &[f32], block_scale: f32) -> f32 {
    let n = weights_i8.len().min(input.len());
    let mut acc = 0.0_f32;
    for (w, v) in weights_i8[..n].iter().zip(input[..n].iter()) {
        acc += *w as f32 * *v;
    }
    acc * block_scale
}

#[allow(dead_code)]
pub(crate) fn packed_nvfp4_row_dot(
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

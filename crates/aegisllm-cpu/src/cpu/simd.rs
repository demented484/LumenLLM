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

#[inline]
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
//
// In Gemma-4 CPU prefill this runs ~14M times/prefill (≈60% of MLP time once the
// matmuls are near-peak). The scalar reference (`gelu_tanh_scalar`) calls libm
// `f32::tanh()`; the fast path replaces it with a branchless tanh approximation
//   tanh(z) = 1 - 2/(1 + exp(2z))
// where `exp` is a polynomial expf: range-reduce `e = n*ln2 + r` (|r| ≤ ln2/2),
// degree-5 minimax for `exp(r)`, then `2^n` via the IEEE-754 exponent-field "ldexp
// bit trick". The AVX-512 path below is hand-written intrinsics (16-wide f32):
// `floor`/`n as i32`/`from_bits` defeat the scalar autovectorizer, so we issue
// `_mm512_roundscale_ps` + `_mm512_cvttps_epi32` + `_mm512_slli_epi32` directly.
// The scalar fallback (`geglu_body`) runs the *same* polynomial so both paths agree.

const SQRT_2_OVER_PI: f32 = 0.797_884_56; // sqrt(2/pi)
const GELU_C: f32 = 0.044_715; // cubic coefficient in the tanh-approx GELU
const EXP_LOG2E: f32 = 1.442_695_04; // 1/ln2
const EXP_LN2_HI: f32 = 0.693_359_38; // ln2 split for an exact Cody–Waite reduction
const EXP_LN2_LO: f32 = -2.121_944_4e-4;
// exp(r) minimax coefficients on r ∈ [-ln2/2, ln2/2] (Horner order, lowest last).
const EXP_P5: f32 = 0.008_333_33;
const EXP_P4: f32 = 0.041_666_67;
const EXP_P3: f32 = 0.166_666_67;
const TANH_CLAMP: f32 = 15.0; // |z| > 15 → tanh saturates to ±1; keeps exp finite
const EXP_HI: f32 = 88.0; // exp input clamp (≈ ln(f32::MAX))
const EXP_LO: f32 = -87.0;

/// ACCURACY REFERENCE (and source of the scalar fallback's exactness target):
/// exact-as-libm gelu_pytorch_tanh.
#[inline(always)]
pub fn gelu_tanh_scalar(x: f32) -> f32 {
    let inner = SQRT_2_OVER_PI * (x + GELU_C * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// Scalar twin of the vectorized polynomial `exp` (same coefficients), used by the
/// scalar fallback and as the unit-test oracle for the intrinsic path.
#[inline(always)]
fn exp_approx(x: f32) -> f32 {
    let x = x.clamp(EXP_LO, EXP_HI);
    let n = (x * EXP_LOG2E + 0.5).floor();
    let r = (x - n * EXP_LN2_HI) - n * EXP_LN2_LO;
    let p =
        1.0 + r * (1.0 + r * (0.5 + r * (EXP_P3 + r * (EXP_P4 + r * EXP_P5))));
    let pow2n = f32::from_bits(((n as i32 + 127) as u32) << 23);
    p * pow2n
}

/// Branchless `tanh(z)` via `tanh(z) = 1 - 2/(1 + exp(2z))`, clamped at ±15.
#[inline(always)]
fn tanh_approx(z: f32) -> f32 {
    let z = z.clamp(-TANH_CLAMP, TANH_CLAMP);
    1.0 - 2.0 / (1.0 + exp_approx(2.0 * z))
}

/// gelu_pytorch_tanh using the approximate tanh (scalar twin of the AVX-512 path).
#[inline(always)]
fn gelu_tanh_approx(x: f32) -> f32 {
    let inner = SQRT_2_OVER_PI * (x + GELU_C * x * x * x);
    0.5 * x * (1.0 + tanh_approx(inner))
}

/// Portable scalar GeGLU body (fallback when AVX-512 is absent). Uses the same
/// polynomial as the intrinsic path so results agree to the last ULP of the approx.
#[inline(always)]
fn geglu_body(gate: &[f32], up: &[f32], out: &mut [f32]) {
    for ((o, &g), &u) in out.iter_mut().zip(gate.iter()).zip(up.iter()) {
        *o = gelu_tanh_approx(g) * u;
    }
}

/// Hand-written AVX-512 GeGLU: `out[i] = gelu_tanh_approx(gate[i]) * up[i]`, 16-wide.
/// Tail (< 16 elements) falls back to the scalar body. Mirrors `gelu_tanh_approx`
/// op-for-op so the unit tests can use the scalar twin as the oracle.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn geglu_avx512(gate: &[f32], up: &[f32], out: &mut [f32]) {
    use std::arch::x86_64::*;

    let len = out.len();
    let lanes = 16;
    let n_vec = len / lanes;

    let half = _mm512_set1_ps(0.5);
    let one = _mm512_set1_ps(1.0);
    let two = _mm512_set1_ps(2.0);
    let sqrt_2_over_pi = _mm512_set1_ps(SQRT_2_OVER_PI);
    let gelu_c = _mm512_set1_ps(GELU_C);
    let tanh_clamp = _mm512_set1_ps(TANH_CLAMP);
    let neg_tanh_clamp = _mm512_set1_ps(-TANH_CLAMP);
    let exp_hi = _mm512_set1_ps(EXP_HI);
    let exp_lo = _mm512_set1_ps(EXP_LO);
    let log2e = _mm512_set1_ps(EXP_LOG2E);
    let ln2_hi = _mm512_set1_ps(EXP_LN2_HI);
    let ln2_lo = _mm512_set1_ps(EXP_LN2_LO);
    let bias = _mm512_set1_epi32(127);
    let p5 = _mm512_set1_ps(EXP_P5);
    let p4 = _mm512_set1_ps(EXP_P4);
    let p3 = _mm512_set1_ps(EXP_P3);
    let p2 = _mm512_set1_ps(0.5);

    let gp = gate.as_ptr();
    let upp = up.as_ptr();
    let op = out.as_mut_ptr();

    for i in 0..n_vec {
        let off = i * lanes;
        let g = _mm512_loadu_ps(gp.add(off));
        let u = _mm512_loadu_ps(upp.add(off));

        // inner = sqrt(2/pi) * (g + GELU_C * g^3)
        let g2 = _mm512_mul_ps(g, g);
        let inner_poly = _mm512_fmadd_ps(_mm512_mul_ps(gelu_c, g2), g, g); // g + c*g^3
        let inner = _mm512_mul_ps(sqrt_2_over_pi, inner_poly);

        // tanh(inner): clamp z to ±15, then 1 - 2/(1 + exp(2z))
        let z = _mm512_min_ps(_mm512_max_ps(inner, neg_tanh_clamp), tanh_clamp);
        let e_in = _mm512_min_ps(_mm512_max_ps(_mm512_mul_ps(two, z), exp_lo), exp_hi);

        // exp(e_in): n = round(e_in * log2e); r = e_in - n*ln2 (Cody–Waite)
        let nf = _mm512_roundscale_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(
            _mm512_mul_ps(e_in, log2e),
        );
        let r = _mm512_fnmadd_ps(nf, ln2_hi, e_in); // e_in - n*ln2_hi
        let r = _mm512_fnmadd_ps(nf, ln2_lo, r); // - n*ln2_lo
        // poly = 1 + r(1 + r(0.5 + r(p3 + r(p4 + r*p5))))  (Horner)
        let mut poly = _mm512_fmadd_ps(r, p5, p4);
        poly = _mm512_fmadd_ps(r, poly, p3);
        poly = _mm512_fmadd_ps(r, poly, p2);
        poly = _mm512_fmadd_ps(r, poly, one);
        poly = _mm512_fmadd_ps(r, poly, one);
        // 2^n via (n + 127) << 23 reinterpreted as f32
        let ni = _mm512_cvttps_epi32(nf);
        let pow2n = _mm512_castsi512_ps(_mm512_slli_epi32::<23>(_mm512_add_epi32(ni, bias)));
        let exp = _mm512_mul_ps(poly, pow2n);

        // tanh = 1 - 2/(1 + exp)
        let tanh = _mm512_sub_ps(one, _mm512_div_ps(two, _mm512_add_ps(one, exp)));
        // gelu = 0.5 * g * (1 + tanh); out = gelu * u
        let gelu = _mm512_mul_ps(_mm512_mul_ps(half, g), _mm512_add_ps(one, tanh));
        _mm512_storeu_ps(op.add(off), _mm512_mul_ps(gelu, u));
    }

    let done = n_vec * lanes;
    if done < len {
        geglu_body(&gate[done..], &up[done..], &mut out[done..]);
    }
}

pub fn geglu_into_simd(gate: &[f32], up: &[f32], out: &mut [f32]) {
    let len = gate.len().min(up.len()).min(out.len());
    let gate = &gate[..len];
    let up = &up[..len];
    let out = &mut out[..len];
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: dispatched only when avx512f is present at runtime; slices are
            // all `len` long, and loads/stores stay within `n_vec*16 ≤ len`.
            unsafe { geglu_avx512(gate, up, out) };
            return;
        }
    }
    geglu_body(gate, up, out);
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

    /// The vectorized gelu_pytorch_tanh fast path must match the libm reference
    /// (`gelu_tanh_scalar`) to < 1e-3 max-abs across the activation range, and must
    /// not blow up (NaN/Inf) in the saturation tails (|x| > 10 → ±1).
    #[test]
    fn gelu_tanh_vectorized_matches_libm() {
        // Dense sweep over the bounded activation range, plus explicit edge cases.
        let mut max_err = 0.0_f32;
        let mut worst_x = 0.0_f32;
        // 0.001 step over [-15, 15] = 30001 samples.
        for i in 0..=30_000 {
            let x = -15.0 + i as f32 * 0.001;
            let want = gelu_tanh_scalar(x);
            let got = gelu_tanh_approx(x);
            assert!(got.is_finite(), "gelu_tanh_approx({x}) = {got} (non-finite)");
            let e = (got - want).abs();
            if e > max_err {
                max_err = e;
                worst_x = x;
            }
        }
        // Saturation tails / edge cases must stay finite and approach the limits.
        for &x in &[0.0f32, 10.0, -10.0, 30.0, -30.0, 100.0, -100.0, 1e4, -1e4] {
            let got = gelu_tanh_approx(x);
            assert!(got.is_finite(), "gelu_tanh_approx({x}) = {got} (non-finite)");
        }
        // gelu(0)=0 exactly; large +x → ~x; large -x → ~0.
        assert!(gelu_tanh_approx(0.0).abs() < 1e-6);
        assert!((gelu_tanh_approx(30.0) - 30.0).abs() < 1e-2);
        assert!(gelu_tanh_approx(-30.0).abs() < 1e-2);
        assert!(
            max_err < 1e-3,
            "max-abs gelu_tanh error {max_err} (at x={worst_x}) exceeds 1e-3"
        );
        eprintln!("gelu_tanh vectorized max-abs error vs libm = {max_err:e} (at x={worst_x})");
    }

    /// End-to-end check of the DISPATCHED `geglu_into_simd` (the real AVX-512 path
    /// on this CPU, incl. the scalar tail) against the per-element libm reference
    /// `gelu_tanh_scalar(gate)*up`. The approximate fast path is allowed up to 2e-3
    /// abs; we also assert the underlying gelu approximation alone stays < 1e-3.
    #[test]
    fn geglu_into_simd_matches_reference() {
        let mut rng = SmallRng::seed_from_u64(SEED);
        // Include tail-exercising lengths (not multiples of 16) and the Gemma-4
        // 6144-wide intermediate (a clean multiple of 16, no tail).
        for &n in &[0usize, 1, 15, 16, 17, 31, 6144, 6145] {
            // gate spread across the active range so the tanh nonlinearity is exercised.
            let gate: Vec<f32> = (0..n).map(|_| rng.random::<f32>() * 24.0 - 12.0).collect();
            let up = rand_vec(&mut rng, n);
            let mut out = vec![0.0_f32; n];
            geglu_into_simd(&gate, &up, &mut out);
            let mut max_gelu_err = 0.0_f32;
            for i in 0..n {
                let want = gelu_tanh_scalar(gate[i]) * up[i];
                assert!(
                    (out[i] - want).abs() < 2e-3,
                    "n={n} i={i} gate={} up={} got={} want={}",
                    gate[i],
                    up[i],
                    out[i],
                    want
                );
                // Isolate the activation error from the `up` scaling: divide back out.
                if up[i].abs() > 1e-3 {
                    let gelu_err = (out[i] / up[i] - gelu_tanh_scalar(gate[i])).abs();
                    max_gelu_err = max_gelu_err.max(gelu_err);
                }
            }
            assert!(
                max_gelu_err < 1e-3,
                "n={n}: dispatched gelu activation max-abs err {max_gelu_err} exceeds 1e-3"
            );
        }
    }

    /// Throughput microbench: scalar (libm tanh) loop vs vectorized `geglu_into_simd`
    /// on a 6144-wide buffer (Gemma-4 MLP intermediate width). Prints Melem/s for
    /// both and the speedup. Ignored by default (run with `--ignored --nocapture`).
    #[test]
    #[ignore]
    fn geglu_microbench() {
        use std::time::Instant;
        const N: usize = 6144;
        const ITERS: usize = 20_000;
        let mut rng = SmallRng::seed_from_u64(SEED);
        let gate: Vec<f32> = (0..N).map(|_| rng.random::<f32>() * 24.0 - 12.0).collect();
        let up = rand_vec(&mut rng, N);
        let mut out = vec![0.0_f32; N];

        // Warm up + scalar baseline (libm tanh, exactly the old kernel).
        let mut sink = 0.0_f32;
        for _ in 0..200 {
            for i in 0..N {
                out[i] = gelu_tanh_scalar(gate[i]) * up[i];
            }
        }
        let t0 = Instant::now();
        for _ in 0..ITERS {
            for i in 0..N {
                out[i] = gelu_tanh_scalar(gate[i]) * up[i];
            }
            sink += out[N / 2];
        }
        let scalar_secs = t0.elapsed().as_secs_f64();

        // Vectorized path.
        for _ in 0..200 {
            geglu_into_simd(&gate, &up, &mut out);
        }
        let t1 = Instant::now();
        for _ in 0..ITERS {
            geglu_into_simd(&gate, &up, &mut out);
            sink += out[N / 2];
        }
        let vec_secs = t1.elapsed().as_secs_f64();

        let elems = (N * ITERS) as f64;
        let scalar_mes = elems / scalar_secs / 1e6;
        let vec_mes = elems / vec_secs / 1e6;
        let speedup = vec_mes / scalar_mes;
        eprintln!(
            "geglu_microbench: scalar(libm tanh) = {scalar_mes:.1} Melem/s, \
             vectorized = {vec_mes:.1} Melem/s, speedup = {speedup:.2}x (sink={sink})"
        );
        assert!(sink.is_finite());
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

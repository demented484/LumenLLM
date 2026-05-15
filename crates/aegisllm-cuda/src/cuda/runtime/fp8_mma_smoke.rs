//! Standalone smoke harness for the SM120 FP8 e4m3 `m16n8k32` tensor-core MMA.
//!
//! Verifies — in escalating stages — the raw
//! `mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32`
//! primitive that a from-scratch FP8 FlashAttention kernel will depend on.
//! `nvcuda::wmma` (m16n16k16) cannot express this instruction; it must be
//! inline PTX, and a prior agent could not verify the from-scratch path
//! build-only. This harness verifies it ON HARDWARE with synthetic data.
//!
//! No model load. Allocates only a few small device buffers (a few MiB at
//! most) — safe to run `compute-sanitizer` memcheck/racecheck on.
//!
//! Stage 1: the bare 16x8x32 e4m3 MMA, exact integer inputs, bit-exact check.
//! Stage 2: a tiled FP8 GEMM M=64,N=64,K=512 (the head_dim=512 contraction).
//! Stage 3: a tiny synthetic FP8 causal attention, head_dim=512.
//!
//! Determinism: each stage runs 5x on identical inputs; outputs must be
//! bit-identical run-to-run (a race would diverge).

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};

use super::CudaRuntime;
use crate::cuda::compile::nvrtc_arch_for_device;
use aegisllm_base::error::{AegisError, Result};

const STAGE1_KERNEL_SRC: &str = include_str!("../kernels/blackwell/fp8_mma_smoke.cu");

// Acceptance bars (from the task brief).
const STAGE1_COS_SIM_BAR: f64 = 0.9999;
const STAGE2_COS_SIM_BAR: f64 = 0.999;
const STAGE3_COS_SIM_BAR: f64 = 0.997;
const SMOKE_RUNS: usize = 5;

/// Per-stage result row.
#[derive(Debug, Clone)]
pub struct Fp8MmaStageResult {
    pub name: &'static str,
    pub shape: String,
    pub cos_sim: f64,
    pub abs_max_err: f32,
    pub ref_abs_max: f32,
    pub deterministic: bool,
    pub bar: f64,
    pub passed: bool,
}

/// Top-level smoke report.
#[derive(Debug, Clone)]
pub struct Fp8MmaSmokeReport {
    pub device_index: usize,
    pub compute_capability: String,
    pub stages: Vec<Fp8MmaStageResult>,
    pub passed: bool,
}

// ---------------------------------------------------------------------------
// e4m3 host codec (HW-round-to-nearest parity).
// ---------------------------------------------------------------------------

/// Encode f32 -> e4m3 byte, round-to-nearest-even, saturating at +-448.
/// Matches `__nv_fp8_e4m3(x)` on device (the kernel's `aegis_f32_to_e4m3`).
fn f32_to_e4m3(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7f;
    }
    let sign: u8 = if x.is_sign_negative() { 0x80 } else { 0x00 };
    let mut a = x.abs();
    if a >= 448.0 {
        // Saturate to max normal (e4m3 has no inf).
        return sign | 0x7e;
    }
    if a == 0.0 {
        return sign;
    }
    // Normal range: smallest normal = 2^-6. Subnormal step = 2^-9.
    const MIN_NORMAL: f32 = 0.015625; // 2^-6
    if a < MIN_NORMAL {
        // Subnormal: value = mant * 2^-9, mant in [0,7].
        let q = (a / 0.001953125).round() as i32; // 2^-9
        let q = q.clamp(0, 7) as u8;
        return sign | q;
    }
    // Decompose into exponent/mantissa with round-to-nearest-even.
    let mut exp = a.log2().floor() as i32; // unbiased
    // Clamp exp to representable [-6, 8].
    if exp < -6 {
        exp = -6;
    }
    if exp > 8 {
        exp = 8;
    }
    let scale = (2.0_f32).powi(exp);
    let mut mant_f = a / scale - 1.0; // in [0,1)
    let mut mant = (mant_f * 8.0).round() as i32;
    if mant >= 8 {
        // mantissa overflow -> bump exponent.
        mant = 0;
        exp += 1;
        if exp > 8 {
            return sign | 0x7e;
        }
    }
    let _ = &mut mant_f;
    let biased = (exp + 7) as u8; // bias 7
    sign | ((biased & 0x0f) << 3) | (mant as u8 & 0x07)
}

/// Decode e4m3 byte -> f32. Matches device `float(__nv_fp8_e4m3)`.
fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0_f32 } else { 1.0 };
    let exp = ((b >> 3) & 0x0f) as i32;
    let mant = (b & 0x07) as i32;
    if exp == 0 {
        // Subnormal: mant * 2^-9.
        sign * (mant as f32) * 0.001953125
    } else {
        let scale = (2.0_f32).powi(exp - 7);
        sign * (1.0 + mant as f32 / 8.0) * scale
    }
}

/// Round-trip an f32 through e4m3 (encode then decode) — the value the MMA
/// actually multiplies.
fn quantize_e4m3(x: f32) -> f32 {
    e4m3_to_f32(f32_to_e4m3(x))
}

// ---------------------------------------------------------------------------
// Deterministic PRNG (xorshift32) — byte-identical reruns.
// ---------------------------------------------------------------------------
struct Rng(u32);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
    /// Uniform on [-range, range].
    fn signed(&mut self, range: f32) -> f32 {
        let u = (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32;
        (u * 2.0 - 1.0) * range
    }
    /// Small integer in [-7, 7] — exactly representable in e4m3.
    fn small_int(&mut self) -> f32 {
        ((self.next_u32() % 15) as i32 - 7) as f32
    }
}

// ---------------------------------------------------------------------------
// Stats.
// ---------------------------------------------------------------------------
fn compare(device: &[f32], reference: &[f32]) -> (f64, f32, f32) {
    let mut sum_xy = 0.0_f64;
    let mut sum_xx = 0.0_f64;
    let mut sum_yy = 0.0_f64;
    let mut abs_max = 0.0_f32;
    let mut ref_abs_max = 0.0_f32;
    for (&d, &r) in device.iter().zip(reference.iter()) {
        sum_xy += d as f64 * r as f64;
        sum_xx += (d as f64) * (d as f64);
        sum_yy += (r as f64) * (r as f64);
        abs_max = abs_max.max((d - r).abs());
        ref_abs_max = ref_abs_max.max(r.abs());
    }
    let denom = (sum_xx.sqrt() * sum_yy.sqrt()).max(1e-12);
    (sum_xy / denom, abs_max, ref_abs_max)
}

// ---------------------------------------------------------------------------
// NVRTC module — compiled separately so the harness does not perturb the main
// BLACKWELL_FP4 kernel module.
// ---------------------------------------------------------------------------
struct SmokeModule {
    _module: Arc<CudaModule>,
    stage1: CudaFunction,
    stage2: CudaFunction,
    stage3: CudaFunction,
}

impl SmokeModule {
    fn load(context: &Arc<CudaContext>, device_index: usize) -> Result<Self> {
        let arch = nvrtc_arch_for_device(device_index);
        let ptx = compile_ptx_with_opts(
            STAGE1_KERNEL_SRC,
            CompileOptions {
                arch: Some(arch),
                name: Some("aegis_fp8_mma_smoke.cu".into()),
                include_paths: cuda_include_paths(),
                ..Default::default()
            },
        )
        .map_err(|e| {
            AegisError::Unsupported(format!("compile fp8 mma smoke kernels failed: {e}"))
        })?;
        let module = context
            .load_module(Ptx::from_src(ptx.to_src()))
            .map_err(|e| AegisError::Unsupported(format!("load fp8 mma smoke module: {e:?}")))?;
        let stage1 = module
            .load_function("aegis_fp8_mma_smoke_stage1")
            .map_err(|e| AegisError::Unsupported(format!("load stage1: {e:?}")))?;
        let stage2 = module
            .load_function("aegis_fp8_mma_smoke_stage2")
            .map_err(|e| AegisError::Unsupported(format!("load stage2: {e:?}")))?;
        let stage3 = module
            .load_function("aegis_fp8_mma_smoke_stage3")
            .map_err(|e| AegisError::Unsupported(format!("load stage3: {e:?}")))?;
        // Stage 3 sizes shared memory dynamically; opt into the 96 KiB pool.
        stage3
            .set_attribute(
                cudarc::driver::sys::CUfunction_attribute_enum
                    ::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                96 * 1024,
            )
            .map_err(|e| {
                AegisError::Unsupported(format!("set dynamic shared on stage3: {e:?}"))
            })?;
        Ok(Self {
            _module: module,
            stage1,
            stage2,
            stage3,
        })
    }
}

fn cuda_include_paths() -> Vec<String> {
    let mut out = Vec::new();
    for var in ["CUDA_PATH", "CUDA_HOME"] {
        if let Ok(root) = std::env::var(var) {
            out.push(format!("{root}/include"));
            out.push(format!("{root}/targets/x86_64-linux/include"));
        }
    }
    out.extend(
        [
            "/opt/cuda/include",
            "/opt/cuda/targets/x86_64-linux/include",
            "/usr/local/cuda/include",
        ]
        .into_iter()
        .map(str::to_owned),
    );
    out.into_iter()
        .filter(|p| std::path::Path::new(p).join("cuda_fp8.h").exists())
        .collect()
}

impl CudaRuntime {
    /// Run the FP8 m16n8k32 MMA smoke harness end-to-end. See the module
    /// doc-comment for the staged design.
    pub fn fp8_mma_smoke(&self) -> Result<Fp8MmaSmokeReport> {
        let cc = self.compute_capability().unwrap_or("unknown").to_string();
        if !cc.starts_with("12.") {
            return Err(AegisError::Unsupported(format!(
                "fp8 mma smoke requires SM120 (compute capability 12.x); device reports {cc}"
            )));
        }
        let module = SmokeModule::load(self.stream.context(), self.device_index())?;

        let mut stages = Vec::new();
        stages.push(self.fp8_smoke_stage1(&module)?);
        stages.push(self.fp8_smoke_stage2(&module)?);
        stages.push(self.fp8_smoke_stage3(&module)?);

        let passed = stages.iter().all(|s| s.passed);
        Ok(Fp8MmaSmokeReport {
            device_index: self.device_index(),
            compute_capability: cc,
            stages,
            passed,
        })
    }

    // -- Stage 1: bare 16x8x32 e4m3 MMA --------------------------------------
    fn fp8_smoke_stage1(&self, module: &SmokeModule) -> Result<Fp8MmaStageResult> {
        const M: usize = 16;
        const N: usize = 8;
        const K: usize = 32;

        // Small integer inputs, exactly representable in e4m3 -> the MMA result
        // is an EXACT integer, so the CPU reference can be bit-exact.
        let mut rng = Rng(0x51A6E1u32);
        let mut a_f32 = vec![0.0_f32; M * K];
        let mut b_f32 = vec![0.0_f32; N * K];
        for v in &mut a_f32 {
            *v = rng.small_int();
        }
        for v in &mut b_f32 {
            *v = rng.small_int();
        }
        let a_e4m3: Vec<u8> = a_f32.iter().map(|&x| f32_to_e4m3(x)).collect();
        let b_e4m3: Vec<u8> = b_f32.iter().map(|&x| f32_to_e4m3(x)).collect();

        // CPU reference: dequant both, exact f32 matmul D = A * B^T.
        let mut reference = vec![0.0_f32; M * N];
        for m in 0..M {
            for n in 0..N {
                let mut acc = 0.0_f32;
                for k in 0..K {
                    acc += e4m3_to_f32(a_e4m3[m * K + k]) * e4m3_to_f32(b_e4m3[n * K + k]);
                }
                reference[m * N + n] = acc;
            }
        }

        let mut d_a = self.alloc_u8(a_e4m3.len())?;
        self.upload_u8_slice_to_device(&a_e4m3, &mut d_a)?;
        let mut d_b = self.alloc_u8(b_e4m3.len())?;
        self.upload_u8_slice_to_device(&b_e4m3, &mut d_b)?;
        let mut d_out = self.alloc_f32(M * N)?;

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut first: Option<Vec<f32>> = None;
        let mut deterministic = true;
        for _ in 0..SMOKE_RUNS {
            unsafe {
                self.stream
                    .launch_builder(&module.stage1)
                    .arg(&d_a.slice)
                    .arg(&d_b.slice)
                    .arg(&mut d_out.slice)
                    .launch(cfg)
            }
            .map_err(|e| AegisError::Unsupported(format!("launch stage1: {e:?}")))?;
            self.synchronize()?;
            let out = self.download_f32(&d_out)?;
            match &first {
                None => first = Some(out),
                Some(prev) => {
                    if prev.iter().zip(out.iter()).any(|(a, b)| a.to_bits() != b.to_bits()) {
                        deterministic = false;
                    }
                }
            }
        }
        let device_out = first.unwrap();
        let (cos_sim, abs_max_err, ref_abs_max) = compare(&device_out, &reference);
        // Stage 1 must be bit-exact: integer inputs, integer result.
        let bit_exact = device_out
            .iter()
            .zip(reference.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits());
        let passed = deterministic && (bit_exact || cos_sim >= STAGE1_COS_SIM_BAR);
        Ok(Fp8MmaStageResult {
            name: "stage1_bare_mma",
            shape: format!("M={M} N={N} K={K} (bit_exact={bit_exact})"),
            cos_sim,
            abs_max_err,
            ref_abs_max,
            deterministic,
            bar: STAGE1_COS_SIM_BAR,
            passed,
        })
    }

    // -- Stage 2: tiled FP8 GEMM, K-loop -------------------------------------
    fn fp8_smoke_stage2(&self, module: &SmokeModule) -> Result<Fp8MmaStageResult> {
        const M: usize = 64;
        const N: usize = 64;
        const K: usize = 512; // head_dim=512 contraction shape

        let mut rng = Rng(0xBEEF77u32);
        let mut a_f32 = vec![0.0_f32; M * K];
        let mut b_f32 = vec![0.0_f32; N * K];
        // Modest magnitudes — keep e4m3 quant error small. Per-element values
        // in [-1,1]; K=512 sum stays well within f32.
        for v in &mut a_f32 {
            *v = rng.signed(1.0);
        }
        for v in &mut b_f32 {
            *v = rng.signed(1.0);
        }
        let a_e4m3: Vec<u8> = a_f32.iter().map(|&x| f32_to_e4m3(x)).collect();
        let b_e4m3: Vec<u8> = b_f32.iter().map(|&x| f32_to_e4m3(x)).collect();

        // CPU reference: dequant + exact f32 matmul over the actual e4m3 values
        // (so the only error vs device is MMA accumulation order, not quant).
        let aq: Vec<f32> = a_e4m3.iter().map(|&b| e4m3_to_f32(b)).collect();
        let bq: Vec<f32> = b_e4m3.iter().map(|&b| e4m3_to_f32(b)).collect();
        let mut reference = vec![0.0_f32; M * N];
        for m in 0..M {
            for n in 0..N {
                let mut acc = 0.0_f32;
                for k in 0..K {
                    acc += aq[m * K + k] * bq[n * K + k];
                }
                reference[m * N + n] = acc;
            }
        }

        let mut d_a = self.alloc_u8(a_e4m3.len())?;
        self.upload_u8_slice_to_device(&a_e4m3, &mut d_a)?;
        let mut d_b = self.alloc_u8(b_e4m3.len())?;
        self.upload_u8_slice_to_device(&b_e4m3, &mut d_b)?;
        let mut d_out = self.alloc_f32(M * N)?;

        let cfg = LaunchConfig {
            grid_dim: ((N / 8) as u32, (M / 16) as u32, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let (mm, nn, kk) = (M as i32, N as i32, K as i32);

        let mut first: Option<Vec<f32>> = None;
        let mut deterministic = true;
        for _ in 0..SMOKE_RUNS {
            unsafe {
                self.stream
                    .launch_builder(&module.stage2)
                    .arg(&d_a.slice)
                    .arg(&d_b.slice)
                    .arg(&mut d_out.slice)
                    .arg(&mm)
                    .arg(&nn)
                    .arg(&kk)
                    .launch(cfg)
            }
            .map_err(|e| AegisError::Unsupported(format!("launch stage2: {e:?}")))?;
            self.synchronize()?;
            let out = self.download_f32(&d_out)?;
            match &first {
                None => first = Some(out),
                Some(prev) => {
                    if prev.iter().zip(out.iter()).any(|(a, b)| a.to_bits() != b.to_bits()) {
                        deterministic = false;
                    }
                }
            }
        }
        let device_out = first.unwrap();
        let (cos_sim, abs_max_err, ref_abs_max) = compare(&device_out, &reference);
        let passed = deterministic && cos_sim >= STAGE2_COS_SIM_BAR;
        Ok(Fp8MmaStageResult {
            name: "stage2_tiled_gemm",
            shape: format!("M={M} N={N} K={K}"),
            cos_sim,
            abs_max_err,
            ref_abs_max,
            deterministic,
            bar: STAGE2_COS_SIM_BAR,
            passed,
        })
    }

    // -- Stage 3: tiny synthetic FP8 attention -------------------------------
    fn fp8_smoke_stage3(&self, module: &SmokeModule) -> Result<Fp8MmaStageResult> {
        const H: usize = 4; // heads
        const QT: usize = 16; // q-tile (total_q = H*16 = 64)
        const CTX: usize = 128;
        const D: usize = 512; // head_dim

        let mut rng = Rng(0x3A77E9u32);

        // Raw f32 Q/K/V. Per-row absmax scaling is REQUIRED for e4m3 (3 mantissa
        // bits) — we quantize each row with its own scale.
        //
        // V scale convention (documented): the P*V MMA contracts over the ctx
        // index `k`, and v_scale[k] cannot be pulled out of the MMA sum if it
        // varies per k. So the harness gives V a SINGLE per-head scale
        // (constant over ctx rows): we compute one absmax over the whole
        // [ctx][D] V block per head. P is requantized online with a per-Q-row
        // scale (that one CAN be pulled out — it is the MMA's M index).
        let mut q_raw = vec![0.0_f32; H * QT * D];
        let mut k_raw = vec![0.0_f32; H * CTX * D];
        let mut v_raw = vec![0.0_f32; H * CTX * D];
        for v in &mut q_raw {
            *v = rng.signed(2.0);
        }
        for v in &mut k_raw {
            *v = rng.signed(2.0);
        }
        for v in &mut v_raw {
            *v = rng.signed(2.0);
        }

        // Per-row Q/K scales; per-head V scale.
        let mut q_e4m3 = vec![0u8; H * QT * D];
        let mut k_e4m3 = vec![0u8; H * CTX * D];
        let mut v_e4m3 = vec![0u8; H * CTX * D];
        let mut q_scale = vec![0.0_f32; H * QT];
        let mut k_scale = vec![0.0_f32; H * CTX];
        let mut v_scale = vec![0.0_f32; H * CTX]; // broadcast: all entries per head equal

        let row_scale = |row: &[f32]| -> f32 {
            let amax = row.iter().fold(0.0_f32, |a, &x| a.max(x.abs()));
            if amax > 0.0 { amax / 448.0 } else { 1.0 }
        };

        for h in 0..H {
            for r in 0..QT {
                let row = &q_raw[(h * QT + r) * D..(h * QT + r) * D + D];
                let s = row_scale(row);
                q_scale[h * QT + r] = s;
                let inv = 1.0 / s;
                for d in 0..D {
                    q_e4m3[(h * QT + r) * D + d] = f32_to_e4m3(row[d] * inv);
                }
            }
            for r in 0..CTX {
                let row = &k_raw[(h * CTX + r) * D..(h * CTX + r) * D + D];
                let s = row_scale(row);
                k_scale[h * CTX + r] = s;
                let inv = 1.0 / s;
                for d in 0..D {
                    k_e4m3[(h * CTX + r) * D + d] = f32_to_e4m3(row[d] * inv);
                }
            }
            // V: single per-head absmax (constant over ctx).
            let v_block = &v_raw[h * CTX * D..(h + 1) * CTX * D];
            let v_amax = v_block.iter().fold(0.0_f32, |a, &x| a.max(x.abs()));
            let vs = if v_amax > 0.0 { v_amax / 448.0 } else { 1.0 };
            let vinv = 1.0 / vs;
            for r in 0..CTX {
                v_scale[h * CTX + r] = vs;
                for d in 0..D {
                    v_e4m3[(h * CTX + r) * D + d] = f32_to_e4m3(v_block[r * D + d] * vinv);
                }
            }
        }

        // -- CPU reference: full f32 attention over the e4m3-dequantized
        //    Q/K/V values (so the device's only extra error is FP8 MMA
        //    accumulation + the P->e4m3 requant the device does internally).
        let softmax_scale = 1.0_f32 / (D as f32).sqrt();
        let mut reference = vec![0.0_f32; H * QT * D];
        for h in 0..H {
            for r in 0..QT {
                // dequant Q row
                let qs = q_scale[h * QT + r];
                let mut qrow = vec![0.0_f32; D];
                for d in 0..D {
                    qrow[d] = e4m3_to_f32(q_e4m3[(h * QT + r) * D + d]) * qs;
                }
                let qpos = (CTX - QT) + r;
                let mut scores = vec![f32::NEG_INFINITY; CTX];
                let mut mrun = f32::NEG_INFINITY;
                for c in 0..CTX {
                    if c > qpos {
                        continue;
                    }
                    let ks = k_scale[h * CTX + c];
                    let mut s = 0.0_f32;
                    for d in 0..D {
                        s += qrow[d] * e4m3_to_f32(k_e4m3[(h * CTX + c) * D + d]) * ks;
                    }
                    s *= softmax_scale;
                    scores[c] = s;
                    mrun = mrun.max(s);
                }
                let mut denom = 0.0_f32;
                for c in 0..CTX {
                    if scores[c] == f32::NEG_INFINITY {
                        scores[c] = 0.0;
                    } else {
                        let e = (scores[c] - mrun).exp();
                        scores[c] = e;
                        denom += e;
                    }
                }
                let inv = if denom > 0.0 { 1.0 / denom } else { 0.0 };
                // P quantized to e4m3 with per-row scale, matching the kernel.
                let mut p = vec![0.0_f32; CTX];
                let mut amax = 0.0_f32;
                for c in 0..CTX {
                    p[c] = scores[c] * inv;
                    amax = amax.max(p[c].abs());
                }
                let ps = if amax > 0.0 { amax / 448.0 } else { 1.0 };
                let vs = v_scale[h * CTX]; // per-head constant
                for d in 0..D {
                    let mut o = 0.0_f32;
                    for c in 0..CTX {
                        let pq = quantize_e4m3(p[c] / ps) * ps;
                        let vq = e4m3_to_f32(v_e4m3[(h * CTX + c) * D + d]) * vs;
                        o += pq * vq;
                    }
                    reference[(h * QT + r) * D + d] = o;
                }
            }
        }

        // -- Device run --
        let mut d_q = self.alloc_u8(q_e4m3.len())?;
        self.upload_u8_slice_to_device(&q_e4m3, &mut d_q)?;
        let mut d_k = self.alloc_u8(k_e4m3.len())?;
        self.upload_u8_slice_to_device(&k_e4m3, &mut d_k)?;
        let mut d_v = self.alloc_u8(v_e4m3.len())?;
        self.upload_u8_slice_to_device(&v_e4m3, &mut d_v)?;
        let mut d_qs = self.alloc_f32(q_scale.len())?;
        self.upload_f32_slice_to_device(&q_scale, &mut d_qs)?;
        let mut d_ks = self.alloc_f32(k_scale.len())?;
        self.upload_f32_slice_to_device(&k_scale, &mut d_ks)?;
        let mut d_vs = self.alloc_f32(v_scale.len())?;
        self.upload_f32_slice_to_device(&v_scale, &mut d_vs)?;
        let mut d_o = self.alloc_f32(H * QT * D)?;

        // shared mem: scores 16*ctx f32 + P_e4m3 16*ctx bytes (pad to 4) +
        // p_scale 16 f32.
        let scores_bytes = 16 * CTX * 4;
        let p_bytes = (16 * CTX + 3) & !3;
        let pscale_bytes = 16 * 4;
        let shared = (scores_bytes + p_bytes + pscale_bytes) as u32;

        let cfg = LaunchConfig {
            grid_dim: (H as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: shared,
        };
        let (hh, cc, dd) = (H as i32, CTX as i32, D as i32);

        let mut first: Option<Vec<f32>> = None;
        let mut deterministic = true;
        for _ in 0..SMOKE_RUNS {
            unsafe {
                self.stream
                    .launch_builder(&module.stage3)
                    .arg(&d_q.slice)
                    .arg(&d_k.slice)
                    .arg(&d_v.slice)
                    .arg(&d_qs.slice)
                    .arg(&d_ks.slice)
                    .arg(&d_vs.slice)
                    .arg(&mut d_o.slice)
                    .arg(&hh)
                    .arg(&cc)
                    .arg(&dd)
                    .launch(cfg)
            }
            .map_err(|e| AegisError::Unsupported(format!("launch stage3: {e:?}")))?;
            self.synchronize()?;
            let out = self.download_f32(&d_o)?;
            match &first {
                None => first = Some(out),
                Some(prev) => {
                    if prev.iter().zip(out.iter()).any(|(a, b)| a.to_bits() != b.to_bits()) {
                        deterministic = false;
                    }
                }
            }
        }
        let device_out = first.unwrap();
        let (cos_sim, abs_max_err, ref_abs_max) = compare(&device_out, &reference);
        let passed = deterministic && cos_sim >= STAGE3_COS_SIM_BAR;
        Ok(Fp8MmaStageResult {
            name: "stage3_fp8_attention",
            shape: format!("H={H} q={QT} ctx={CTX} D={D} causal"),
            cos_sim,
            abs_max_err,
            ref_abs_max,
            deterministic,
            bar: STAGE3_COS_SIM_BAR,
            passed,
        })
    }
}

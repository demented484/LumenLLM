//! Standalone smoke test for the CUTLASS NVFP4 grouped GEMM bridge.
//!
//! Exercises the three helpers added in commit 4935403:
//!   * `aegis_cutlass_moe_nvfp4_sfa_sfb_bytes_sm120`
//!   * `aegis_cutlass_moe_nvfp4_swizzle_weight_scales_grouped`
//!   * `aegis_cutlass_moe_nvfp4_quantize_input_grouped`
//! plus the actual grouped GEMM launch via `aegis_cutlass_moe_nvfp4_grouped_gemm_sm120`.
//!
//! No model load is required — synthetic per-expert data is generated with
//! a deterministic seed so reruns are reproducible. Compares device output
//! to a CPU reference that runs the same NVFP4 quant + dequant + f32 matmul.
//!
//! Build flag: `AEGIS_CUTLASS_NVFP4_GROUPED_BUILD=1`. Without it, the
//! `cutlass_nvfp4_moe_grouped_built()` runtime probe returns `false` and
//! the smoke command errors out with a friendly message.

use std::ffi::c_void;
use std::time::Instant;

use super::CudaRuntime;
use crate::cuda::cutlass_bridge;
use aegisllm_base::error::{AegisError, Result};
/// Standard NVFP4 e2m1 nibble decode (matching CUTLASS's
/// `cutlass::float_e2m1_t` → float). Magnitudes are {0, 0.5, 1, 1.5,
/// 2, 3, 4, 6}, sign in bit 3. Note this is HALF the value returned by
/// `decode_nvfp4_nibble_i8` from aegisllm_base::tensor::quant (which
/// returns integer values 0, 1, 2, ..., 12 — those are 2× the true
/// float values, paired with the half-LUT scale decoder).
fn decode_e2m1_float(nibble: u8) -> f32 {
    const VALUES: [f32; 16] = [
        0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,
        0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    VALUES[(nibble & 0xf) as usize]
}

/// "True" E4M3 decoding matching `__nv_fp8_e4m3` round-trip (no half-LUT
/// 0.5 factor that the legacy aegis NVFP4 kernels use). The kernel's
/// `quantize_grouped_f32_to_e2m1_ue4m3_kernel` writes the byte via
/// `__nv_fp8_e4m3(scale)` and then CUTLASS reads it back as a standard
/// E4M3 value. To match, our CPU reference must also use true E4M3.
fn decode_ue4m3_true(byte: u8) -> f32 {
    let byte = byte & 0x7f;
    if byte == 0 || byte == 0x7f {
        return 0.0;
    }
    let exponent = ((byte >> 3) & 0x0f) as i32;
    let mantissa = (byte & 0x07) as f32;
    if exponent == 0 {
        // Subnormal: mantissa * 2^(1-7) / 8 = mantissa / 1024
        mantissa / 1024.0
    } else {
        (1.0 + mantissa * 0.125) * (2.0_f32).powi(exponent - 7)
    }
}

/// Per-expert summary printed by the smoke command.
#[derive(Debug, Clone)]
pub struct CutlassMoeSmokeExpert {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub output_scale: f32,
    /// Per-element cosine similarity between device output and CPU reference.
    pub cos_sim: f64,
    /// Max absolute error |device - reference| across the M×N output tile.
    pub abs_max_err: f32,
    /// Reference output L2 norm (sanity check that we're not comparing zeros).
    pub ref_l2: f64,
    /// Best-fit scalar `c` such that `device ≈ c * reference`. A value far
    /// from 1.0 with cos_sim ≈ 1 indicates a global scaling mismatch
    /// (e.g. alpha applied twice, or scale-LUT discrepancy).
    pub scale_ratio: f64,
    /// Max absolute reference value (denominator for relative error).
    pub ref_abs_max: f32,
}

/// Top-level smoke report.
#[derive(Debug, Clone)]
pub struct CutlassMoeSmokeReport {
    pub num_experts: usize,
    pub experts: Vec<CutlassMoeSmokeExpert>,
    pub workspace_bytes: usize,
    pub gemm_ms: f64,
    /// Per-group SFA/SFB bytes returned by the helper, indexed by expert.
    pub sfa_sfb_bytes: Vec<(usize, usize)>,
    /// Acceptance threshold used (cos_sim ≥ this per expert).
    pub cos_sim_threshold: f64,
    pub passed: bool,
}

const COS_SIM_THRESHOLD: f64 = 0.998;

/// Per-block NVFP4 quantization mirroring the device-side
/// `quantize_grouped_f32_to_e2m1_ue4m3_kernel`: amax/6 → ue4m3 scale,
/// then `moe_best_e2m1` rounding-to-nearest using boundaries
/// {0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0}.
///
/// Returns (nibbles[16], scale_byte).
fn quantize_block_like_kernel(values: &[f32; 16]) -> ([u8; 16], u8) {
    let mut amax = 0.0_f32;
    for &v in values {
        amax = amax.max(v.abs());
    }
    let scale = if amax > 0.0 { amax / 6.0 } else { 0.0 };
    let scale_byte = fp32_to_ue4m3_kernel_style(scale);
    let decoded_scale = decode_ue4m3_true(scale_byte);
    let mut nibbles = [0u8; 16];
    for (i, &v) in values.iter().enumerate() {
        nibbles[i] = moe_best_e2m1_like_kernel(v, decoded_scale);
    }
    (nibbles, scale_byte)
}

/// Mirrors `__nv_fp8_e4m3(scale)` construction on device (round-to-nearest).
/// We use the existing CPU `fp32_to_ue4m3` from quant.rs via the public
/// `quantize_input_nvfp4_into` path — but that helper takes a full block
/// and we want just the byte. Reimplement here matching the CUDA cast.
fn fp32_to_ue4m3_kernel_style(x: f32) -> u8 {
    // Match the CPU implementation in aegisllm_base::tensor::quant (private fn).
    // E4M3 with bias 7. Subnormals: mantissa = round(x * 512). Clip to 448.
    if x <= 0.0 {
        return 0;
    }
    let mut x = x;
    if x > 448.0 {
        x = 448.0;
    }
    let bits = x.to_bits();
    let fp32_exp = ((bits >> 23) & 0xff) as i32 - 127;
    let fp32_man = ((bits >> 20) & 0x7) as i32;
    let mut ue4m3_exp = fp32_exp + 7;
    if ue4m3_exp <= 0 {
        let mut man = (x * 512.0 + 0.5) as i32;
        if man > 7 {
            man = 7;
        }
        if man < 1 {
            return 0;
        }
        return man as u8;
    }
    if ue4m3_exp >= 15 {
        return 0x7e;
    }
    let round_bit = ((bits >> 19) & 1) as i32;
    let mut ue4m3_man = fp32_man + round_bit;
    if ue4m3_man > 7 {
        ue4m3_man = 0;
        ue4m3_exp += 1;
        if ue4m3_exp >= 15 {
            return 0x7e;
        }
    }
    ((ue4m3_exp as u8) << 3) | ue4m3_man as u8
}

/// Mirrors device `moe_best_e2m1`: piecewise nearest-neighbor mapping
/// to e2m1 magnitudes {0, 1, 2, 3, 4, 6, 8, 12} (via mag thresholds
/// {0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0}), sign in bit 3.
fn moe_best_e2m1_like_kernel(value: f32, scale: f32) -> u8 {
    if !(scale > 0.0) {
        return 0;
    }
    let scaled = value / scale;
    if !scaled.is_finite() {
        return 0;
    }
    let mag = scaled.abs();
    let code: u8 = if mag <= 0.25 {
        0
    } else if mag <= 0.75 {
        1
    } else if mag <= 1.25 {
        2
    } else if mag <= 1.75 {
        3
    } else if mag <= 2.5 {
        4
    } else if mag <= 3.5 {
        5
    } else if mag <= 5.0 {
        6
    } else {
        7
    };
    if scaled < 0.0 && code != 0 {
        code | 0x8
    } else {
        code
    }
}

/// Deterministic xorshift32 PRNG so reruns produce byte-identical results.
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
    fn next_unit(&mut self) -> f32 {
        // Uniform on [0, 1).
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
    fn next_signed(&mut self, range: f32) -> f32 {
        (self.next_unit() * 2.0 - 1.0) * range
    }
    fn next_nibble(&mut self) -> u8 {
        (self.next_u32() & 0xf) as u8
    }
    /// Random ue4m3 scale byte that produces a "reasonable" decoded scale
    /// (≈ 0.5 .. 2.0). We pick bytes in [0x38, 0x42] which give decoded
    /// values around 1.0 (exp = 7 → 0.5, exp = 8 → 1.0..2.0).
    fn next_ue4m3_scale(&mut self) -> u8 {
        0x38 + (self.next_u32() & 0xf) as u8
    }
}

impl CudaRuntime {
    /// Run the CUTLASS NVFP4 grouped GEMM bridge end-to-end on synthetic
    /// data and verify the output matches a CPU reference. See the module
    /// doc-comment for details on the test design.
    pub fn cutlass_nvfp4_moe_grouped_smoke(&self) -> Result<CutlassMoeSmokeReport> {
        if !Self::cutlass_nvfp4_moe_grouped_built() {
            return Err(AegisError::Unsupported(
                "CUTLASS NVFP4 grouped bridge is not compiled into this build; \
                 rebuild with AEGIS_CUTLASS_NVFP4_GROUPED_BUILD=1"
                    .into(),
            ));
        }
        smoke_impl(self)
    }
}

#[cfg(not(aegis_cutlass_nvfp4_grouped))]
fn smoke_impl(_runtime: &CudaRuntime) -> Result<CutlassMoeSmokeReport> {
    Err(AegisError::Unsupported(
        "CUTLASS NVFP4 grouped bridge is not compiled into this build; \
         rebuild with AEGIS_CUTLASS_NVFP4_GROUPED_BUILD=1"
            .into(),
    ))
}

#[cfg(aegis_cutlass_nvfp4_grouped)]
fn smoke_impl(runtime: &CudaRuntime) -> Result<CutlassMoeSmokeReport> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};

    // -- Test problem definition ------------------------------------------------
    let expert_ms: [usize; 4] = [128, 256, 512, 1024];
    let n: usize = 128;
    let k: usize = 128;
    let num_groups = expert_ms.len();
    let total_tokens: usize = expert_ms.iter().sum();

    // -- Generate synthetic activations + weights -------------------------------
    let mut rng = Rng(0xC0FFEEEEu32);

    // Activations: total_tokens rows × k cols, values in [-2, 2].
    let mut act = vec![0.0_f32; total_tokens * k];
    for v in &mut act {
        *v = rng.next_signed(2.0);
    }

    // Per-expert weight nibbles (B^T-style row-major as it ships from the loader:
    // rows = N, cols = K, packed to N × K/2 bytes per expert).
    // Per-expert raw row-major scales: N × (K/16) bytes per expert.
    let weight_payload_per_g = n * k / 2;
    let weight_scales_per_g = n * (k / 16);
    let mut weights_b = vec![0u8; num_groups * weight_payload_per_g];
    let mut weights_b_scales_raw = vec![0u8; num_groups * weight_scales_per_g];
    for byte in &mut weights_b {
        // Two nibbles per byte. Keep magnitudes modest by occasionally zeroing.
        *byte = rng.next_nibble() | (rng.next_nibble() << 4);
    }
    for byte in &mut weights_b_scales_raw {
        *byte = rng.next_ue4m3_scale();
    }

    // Per-expert alpha (output_scale).
    let alphas: Vec<f32> = (0..num_groups)
        .map(|g| 0.5 + 0.25 * g as f32) // {0.5, 0.75, 1.0, 1.25}
        .collect();

    // -- Token offsets (prefix-sum) --------------------------------------------
    let mut token_offsets = vec![0u32; num_groups + 1];
    for g in 0..num_groups {
        token_offsets[g + 1] = token_offsets[g] + expert_ms[g] as u32;
    }

    // -- Per-expert payload + SFA layout offsets (in bytes) --------------------
    // payload_g size = m_g * (k/2). sfa_g size = padded_rows(m_g) * padded_scale_cols(k).
    let scale_cols = k / 16;
    let padded_scale_cols = scale_cols.div_ceil(4) * 4; // == 8 for k=128
    let max_padded_rows = expert_ms.iter().copied().max().unwrap().div_ceil(128) * 128;

    let mut payload_offsets = vec![0u64; num_groups];
    let mut sfa_offsets = vec![0u64; num_groups];
    let mut payload_total: u64 = 0;
    let mut sfa_total: u64 = 0;
    for g in 0..num_groups {
        payload_offsets[g] = payload_total;
        sfa_offsets[g] = sfa_total;
        payload_total += (expert_ms[g] * (k / 2)) as u64;
        let padded_rows_g = expert_ms[g].div_ceil(128) * 128;
        sfa_total += (padded_rows_g * padded_scale_cols) as u64;
    }

    // -- Per-expert SFB layout offsets (in bytes) ------------------------------
    // Per-group raw weight-scale offsets (input to swizzle):
    let mut wscale_src_offsets = vec![0u64; num_groups];
    for g in 0..num_groups {
        wscale_src_offsets[g] = (g * weight_scales_per_g) as u64;
    }
    // Per-group SFB destination offsets — N is shared, so SFB byte size
    // is identical across groups: padded_rows(N) * padded_scale_cols(K).
    let sfb_per_g = (n.div_ceil(128) * 128) * padded_scale_cols;
    let mut sfb_offsets = vec![0u64; num_groups];
    for g in 0..num_groups {
        sfb_offsets[g] = (g * sfb_per_g) as u64;
    }
    let sfb_total = (num_groups * sfb_per_g) as u64;

    // -- Cross-check sfa/sfb sizes against bridge helper -----------------------
    let mut sfa_sfb_bytes = Vec::with_capacity(num_groups);
    for g in 0..num_groups {
        let m_g = expert_ms[g] as i32;
        let n_g = n as i32;
        let k_g = k as i32;
        let (sfa_b, sfb_b) = cutlass_bridge::moe_grouped_sfa_sfb_bytes(m_g, n_g, k_g)
            .map_err(|e| AegisError::Unsupported(format!("sfa_sfb_bytes(m={m_g},n={n_g},k={k_g}): {e}")))?;
        sfa_sfb_bytes.push((sfa_b, sfb_b));
        // Our manual blob math:
        let padded_rows_g = expert_ms[g].div_ceil(128) * 128;
        let manual_sfa = padded_rows_g * padded_scale_cols;
        let manual_sfb = (n.div_ceil(128) * 128) * padded_scale_cols;
        if manual_sfa != sfa_b {
            return Err(AegisError::Unsupported(format!(
                "expert {g} SFA bytes mismatch: helper={sfa_b} manual={manual_sfa} (m={m_g} k={k_g})"
            )));
        }
        if manual_sfb != sfb_b {
            return Err(AegisError::Unsupported(format!(
                "expert {g} SFB bytes mismatch: helper={sfb_b} manual={manual_sfb} (n={n_g} k={k_g})"
            )));
        }
    }

    // -- Allocate device buffers ----------------------------------------------
    let mut d_act = runtime.alloc_f32(act.len())?;
    runtime.upload_f32_slice_to_device(&act, &mut d_act)?;

    let mut d_token_offsets = runtime.alloc_u32(token_offsets.len())?;
    runtime.upload_u32_slice_to_device(&token_offsets, &mut d_token_offsets)?;
    let mut d_payload_offsets = runtime.alloc_u64(payload_offsets.len())?;
    runtime.upload_u64_slice_to_device(&payload_offsets, &mut d_payload_offsets)?;
    let mut d_sfa_offsets = runtime.alloc_u64(sfa_offsets.len())?;
    runtime.upload_u64_slice_to_device(&sfa_offsets, &mut d_sfa_offsets)?;

    let mut d_payload = runtime.alloc_u8(payload_total as usize)?;
    let mut d_sfa = runtime.alloc_u8(sfa_total as usize)?;

    let mut d_wb = runtime.alloc_u8(weights_b.len())?;
    runtime.upload_u8_slice_to_device(&weights_b, &mut d_wb)?;
    let mut d_wb_scales_raw = runtime.alloc_u8(weights_b_scales_raw.len())?;
    runtime.upload_u8_slice_to_device(&weights_b_scales_raw, &mut d_wb_scales_raw)?;
    let mut d_wb_src_offsets = runtime.alloc_u64(wscale_src_offsets.len())?;
    runtime.upload_u64_slice_to_device(&wscale_src_offsets, &mut d_wb_src_offsets)?;
    let mut d_sfb_offsets = runtime.alloc_u64(sfb_offsets.len())?;
    runtime.upload_u64_slice_to_device(&sfb_offsets, &mut d_sfb_offsets)?;
    let mut d_sfb = runtime.alloc_u8(sfb_total as usize)?;

    // Per-expert output: total_tokens × n.
    let mut d_out = runtime.alloc_f32(total_tokens * n)?;

    // Per-expert alphas. Need one *float per group (alpha_ptr_array points
    // to an array of pointers, each pointer to a single f32). We allocate
    // a single f32 buffer of length num_groups, and an array of u64 pointers.
    let mut d_alphas = runtime.alloc_f32(num_groups)?;
    runtime.upload_f32_slice_to_device(&alphas, &mut d_alphas)?;

    let stream_ptr: *mut c_void = runtime.stream.cu_stream().cast::<c_void>();

    // -- Step 1: swizzle weight scales (Option B device-side) -----------------
    {
        let (src_ptr, _src_read) = d_wb_scales_raw.slice.device_ptr(&runtime.stream);
        let (src_off_ptr, _so) = d_wb_src_offsets.slice.device_ptr(&runtime.stream);
        let (dst_off_ptr, _do) = d_sfb_offsets.slice.device_ptr(&runtime.stream);
        let (dst_ptr, _dst_w) = d_sfb.slice.device_ptr_mut(&runtime.stream);
        unsafe {
            cutlass_bridge::moe_grouped_swizzle_weight_scales(
                src_ptr as *const u8,
                n as i32,
                scale_cols as i32,
                num_groups as i32,
                src_off_ptr as *const u64,
                dst_off_ptr as *const u64,
                dst_ptr as *mut u8,
                stream_ptr,
            )
        }
        .map_err(|e| AegisError::Unsupported(format!("swizzle_weight_scales: {e}")))?;
    }

    // -- Step 2: quantize activations per expert into payload + SFA -----------
    {
        let (act_ptr, _ar) = d_act.slice.device_ptr(&runtime.stream);
        let (tok_ptr, _tr) = d_token_offsets.slice.device_ptr(&runtime.stream);
        let (po_ptr, _po) = d_payload_offsets.slice.device_ptr(&runtime.stream);
        let (so_ptr, _so) = d_sfa_offsets.slice.device_ptr(&runtime.stream);
        let (payload_ptr, _pw) = d_payload.slice.device_ptr_mut(&runtime.stream);
        let (sfa_ptr, _sw) = d_sfa.slice.device_ptr_mut(&runtime.stream);
        unsafe {
            cutlass_bridge::moe_grouped_quantize_input(
                act_ptr as *const f32,
                k as i32,
                num_groups as i32,
                tok_ptr as *const u32,
                po_ptr as *const u64,
                so_ptr as *const u64,
                max_padded_rows as i32,
                payload_ptr as *mut u8,
                sfa_ptr as *mut u8,
                stream_ptr,
            )
        }
        .map_err(|e| AegisError::Unsupported(format!("quantize_input_grouped: {e}")))?;
    }
    runtime.synchronize()?;

    // -- Step 3: build per-group strides + layouts ---------------------------
    let sizes = cutlass_bridge::moe_grouped_blob_sizes()
        .map_err(|e| AegisError::Unsupported(format!("moe_grouped_blob_sizes: {e}")))?;

    let mut stride_a_blob = vec![0u8; sizes.stride_a * num_groups];
    let mut stride_b_blob = vec![0u8; sizes.stride_b * num_groups];
    let mut stride_d_blob = vec![0u8; sizes.stride_d * num_groups];
    let mut layout_sfa_blob = vec![0u8; sizes.layout_sfa * num_groups];
    let mut layout_sfb_blob = vec![0u8; sizes.layout_sfb * num_groups];
    let mut problem_shape_blob = vec![0u8; sizes.problem_shape * num_groups];

    // Per-expert problem_sizes are an array of (m, n, k, [batch?]) — they're
    // emitted as cute Shape<int,int,int>. We don't know the exact byte layout
    // a priori; query the C++ side once per group to populate it.
    // We have `sizes.problem_shape` (the per-group struct size). The host
    // representation is 3 × i32 (Shape<int,int,int>) but CuTe may pad — use
    // memcpy of (i32, i32, i32) at the start of each slot.
    for g in 0..num_groups {
        let m_g = expert_ms[g] as i32;
        let n_g = n as i32;
        let k_g = k as i32;
        cutlass_bridge::moe_grouped_compute_strides(
            m_g, n_g, k_g,
            &mut stride_a_blob[g * sizes.stride_a..(g + 1) * sizes.stride_a],
            &mut stride_b_blob[g * sizes.stride_b..(g + 1) * sizes.stride_b],
            &mut stride_d_blob[g * sizes.stride_d..(g + 1) * sizes.stride_d],
            &mut layout_sfa_blob[g * sizes.layout_sfa..(g + 1) * sizes.layout_sfa],
            &mut layout_sfb_blob[g * sizes.layout_sfb..(g + 1) * sizes.layout_sfb],
        )
        .map_err(AegisError::Unsupported)?;
        // Write (m, n, k) at the start of the problem-shape slot. We size
        // by sizes.problem_shape (the actual struct size).
        let slot = &mut problem_shape_blob[g * sizes.problem_shape..(g + 1) * sizes.problem_shape];
        // Defensive: at minimum 3*i32 = 12 bytes.
        if slot.len() < 12 {
            return Err(AegisError::Unsupported(format!(
                "problem_shape slot too small: {} bytes (need ≥ 12)",
                slot.len()
            )));
        }
        slot[0..4].copy_from_slice(&m_g.to_le_bytes());
        slot[4..8].copy_from_slice(&n_g.to_le_bytes());
        slot[8..12].copy_from_slice(&k_g.to_le_bytes());
    }

    let mut d_stride_a = runtime.alloc_u8(stride_a_blob.len())?;
    runtime.upload_u8_slice_to_device(&stride_a_blob, &mut d_stride_a)?;
    let mut d_stride_b = runtime.alloc_u8(stride_b_blob.len())?;
    runtime.upload_u8_slice_to_device(&stride_b_blob, &mut d_stride_b)?;
    let mut d_stride_d = runtime.alloc_u8(stride_d_blob.len())?;
    runtime.upload_u8_slice_to_device(&stride_d_blob, &mut d_stride_d)?;
    let mut d_layout_sfa = runtime.alloc_u8(layout_sfa_blob.len())?;
    runtime.upload_u8_slice_to_device(&layout_sfa_blob, &mut d_layout_sfa)?;
    let mut d_layout_sfb = runtime.alloc_u8(layout_sfb_blob.len())?;
    runtime.upload_u8_slice_to_device(&layout_sfb_blob, &mut d_layout_sfb)?;
    let mut d_problem_shapes = runtime.alloc_u8(problem_shape_blob.len())?;
    runtime.upload_u8_slice_to_device(&problem_shape_blob, &mut d_problem_shapes)?;

    // -- Step 4: build per-group device pointer arrays -----------------------
    // For each group g, A_ptr[g] = d_payload + payload_offsets[g],
    // B_ptr[g] = d_wb + g * weight_payload_per_g, SFA_ptr[g] = d_sfa + sfa_offsets[g],
    // SFB_ptr[g] = d_sfb + sfb_offsets[g], D_ptr[g] = d_out + (token_offsets[g] * n) * 4.
    // alpha_ptr[g] = d_alphas + g (one f32 per group).
    let mut a_ptrs = vec![0u64; num_groups];
    let mut b_ptrs = vec![0u64; num_groups];
    let mut sfa_ptrs = vec![0u64; num_groups];
    let mut sfb_ptrs = vec![0u64; num_groups];
    let mut d_ptrs = vec![0u64; num_groups];
    let mut alpha_ptrs = vec![0u64; num_groups];

    {
        let (payload_base, _pb) = d_payload.slice.device_ptr(&runtime.stream);
        let (wb_base, _wbb) = d_wb.slice.device_ptr(&runtime.stream);
        let (sfa_base, _sb) = d_sfa.slice.device_ptr(&runtime.stream);
        let (sfb_base, _sfb_b) = d_sfb.slice.device_ptr(&runtime.stream);
        let (out_base, _ob) = d_out.slice.device_ptr_mut(&runtime.stream);
        let (alpha_base, _ab) = d_alphas.slice.device_ptr(&runtime.stream);
        for g in 0..num_groups {
            a_ptrs[g] = payload_base + payload_offsets[g];
            b_ptrs[g] = wb_base + (g * weight_payload_per_g) as u64;
            sfa_ptrs[g] = sfa_base + sfa_offsets[g];
            sfb_ptrs[g] = sfb_base + sfb_offsets[g];
            d_ptrs[g] = out_base + (token_offsets[g] as u64) * (n as u64) * 4;
            alpha_ptrs[g] = alpha_base + (g as u64) * 4;
        }
    }
    let mut d_a_ptrs = runtime.upload_u64_slice(&a_ptrs)?;
    let mut d_b_ptrs = runtime.upload_u64_slice(&b_ptrs)?;
    let mut d_sfa_ptrs = runtime.upload_u64_slice(&sfa_ptrs)?;
    let mut d_sfb_ptrs = runtime.upload_u64_slice(&sfb_ptrs)?;
    let mut d_d_ptrs = runtime.upload_u64_slice(&d_ptrs)?;
    let mut d_alpha_ptrs = runtime.upload_u64_slice(&alpha_ptrs)?;

    // -- Step 5: workspace size + run ----------------------------------------
    let (d_ps_ptr, _dps) = d_problem_shapes.slice.device_ptr(&runtime.stream);
    let workspace_bytes = unsafe {
        cutlass_bridge::moe_grouped_workspace_size(
            num_groups as i32,
            d_ps_ptr as *mut c_void,
            std::ptr::null(),
        )
    }
    .map_err(|e| AegisError::Unsupported(format!("moe_grouped_workspace_size: {e}")))?;
    let mut d_workspace = runtime.alloc_u8(workspace_bytes.max(1))?;

    let (d_ws_ptr, _dws) = d_workspace.slice.device_ptr_mut(&runtime.stream);
    let (a_pp_ptr, _) = d_a_ptrs.slice.device_ptr_mut(&runtime.stream);
    let (b_pp_ptr, _) = d_b_ptrs.slice.device_ptr_mut(&runtime.stream);
    let (sfa_pp_ptr, _) = d_sfa_ptrs.slice.device_ptr_mut(&runtime.stream);
    let (sfb_pp_ptr, _) = d_sfb_ptrs.slice.device_ptr_mut(&runtime.stream);
    let (d_pp_ptr, _) = d_d_ptrs.slice.device_ptr_mut(&runtime.stream);
    let (alpha_pp_ptr, _) = d_alpha_ptrs.slice.device_ptr_mut(&runtime.stream);
    let (sa_ptr, _) = d_stride_a.slice.device_ptr_mut(&runtime.stream);
    let (sb_ptr, _) = d_stride_b.slice.device_ptr_mut(&runtime.stream);
    let (sd_ptr, _) = d_stride_d.slice.device_ptr_mut(&runtime.stream);
    let (lsfa_ptr, _) = d_layout_sfa.slice.device_ptr_mut(&runtime.stream);
    let (lsfb_ptr, _) = d_layout_sfb.slice.device_ptr_mut(&runtime.stream);

    runtime.synchronize()?;
    let t0 = Instant::now();
    unsafe {
        cutlass_bridge::moe_grouped_run(
            num_groups as i32,
            d_ps_ptr as *mut c_void,
            std::ptr::null(),
            a_pp_ptr as *mut *mut c_void,
            b_pp_ptr as *mut *mut c_void,
            sfa_pp_ptr as *mut *mut c_void,
            sfb_pp_ptr as *mut *mut c_void,
            d_pp_ptr as *mut *mut c_void,
            sa_ptr as *mut c_void,
            sb_ptr as *mut c_void,
            sd_ptr as *mut c_void,
            lsfa_ptr as *mut c_void,
            lsfb_ptr as *mut c_void,
            alpha_pp_ptr as *mut *mut f32,
            d_ws_ptr as *mut c_void,
            workspace_bytes,
            stream_ptr,
        )
    }
    .map_err(|e| AegisError::Unsupported(format!("moe_grouped_run: {e}")))?;
    runtime.synchronize()?;
    let gemm_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // -- Step 6: download output + CPU reference comparison -------------------
    let device_out = runtime.download_f32(&d_out)?;

    let mut experts = Vec::with_capacity(num_groups);
    let mut all_passed = true;
    for g in 0..num_groups {
        let m_g = expert_ms[g];
        let row_start = token_offsets[g] as usize;

        // Build reference: per-block CPU NVFP4 quant of activations
        // (matches the kernel byte-exactly), then CPU dequant of both A
        // and B (using the same nibble + ue4m3 tables), then f32 matmul,
        // then × alpha.
        let mut dequant_a = vec![0.0_f32; m_g * k];
        for row in 0..m_g {
            for blk in 0..(k / 16) {
                let mut values = [0.0_f32; 16];
                for i in 0..16 {
                    values[i] = act[(row_start + row) * k + blk * 16 + i];
                }
                let (nibbles, scale_byte) = quantize_block_like_kernel(&values);
                let decoded_scale = decode_ue4m3_true(scale_byte);
                for i in 0..16 {
                    let dq = decode_e2m1_float(nibbles[i]) * decoded_scale;
                    dequant_a[row * k + blk * 16 + i] = dq;
                }
            }
        }

        // Weights B for group g: layout = row-major N × K nibbles, packed
        // 2 nibbles/byte. Scale layout = row-major N × (K/16) ue4m3 bytes.
        // CPU dequant produces a f32 N × K matrix.
        let mut dequant_b = vec![0.0_f32; n * k];
        let wb_base = g * weight_payload_per_g;
        let ws_base = g * weight_scales_per_g;
        for row in 0..n {
            for blk in 0..(k / 16) {
                let scale_byte = weights_b_scales_raw[ws_base + row * (k / 16) + blk];
                let decoded_scale = decode_ue4m3_true(scale_byte);
                for pair in 0..8 {
                    let byte = weights_b[wb_base + row * (k / 2) + blk * 8 + pair];
                    let lo = byte & 0x0f;
                    let hi = byte >> 4;
                    let lo_val = decode_e2m1_float(lo) * decoded_scale;
                    let hi_val = decode_e2m1_float(hi) * decoded_scale;
                    dequant_b[row * k + blk * 16 + pair * 2] = lo_val;
                    dequant_b[row * k + blk * 16 + pair * 2 + 1] = hi_val;
                }
            }
        }

        // Reference matmul: out[r, c] = alpha * sum_k A[r,k] * B[c,k] (B is N×K row-major,
        // mapped to LayoutB=ColumnMajor for CUTLASS — i.e. CUTLASS sees B^T with N
        // as the contraction-output dimension, K as the contraction dim).
        let alpha = alphas[g];
        let mut reference = vec![0.0_f32; m_g * n];
        for row in 0..m_g {
            for col in 0..n {
                let mut acc = 0.0_f32;
                for kk in 0..k {
                    acc += dequant_a[row * k + kk] * dequant_b[col * k + kk];
                }
                reference[row * n + col] = alpha * acc;
            }
        }

        // Device tile for group g: out[(row_start + r) * n + c].
        let mut sum_xy = 0.0_f64;
        let mut sum_xx = 0.0_f64;
        let mut sum_yy = 0.0_f64;
        let mut abs_max = 0.0_f32;
        let mut ref_abs_max = 0.0_f32;
        for row in 0..m_g {
            for col in 0..n {
                let dev = device_out[(row_start + row) * n + col];
                let refv = reference[row * n + col];
                let d = (dev as f64) * (refv as f64);
                sum_xy += d;
                sum_xx += (dev as f64).powi(2);
                sum_yy += (refv as f64).powi(2);
                abs_max = abs_max.max((dev - refv).abs());
                ref_abs_max = ref_abs_max.max(refv.abs());
            }
        }
        let denom = (sum_xx.sqrt() * sum_yy.sqrt()).max(1e-12);
        let cos_sim = sum_xy / denom;
        // Best-fit c minimizing ||dev - c*ref||^2  is c = <dev, ref> / <ref, ref>.
        let scale_ratio = if sum_yy > 0.0 { sum_xy / sum_yy } else { 0.0 };
        let passed = cos_sim >= COS_SIM_THRESHOLD;
        all_passed &= passed;
        experts.push(CutlassMoeSmokeExpert {
            m: m_g,
            n,
            k,
            output_scale: alpha,
            cos_sim,
            abs_max_err: abs_max,
            ref_l2: sum_yy.sqrt(),
            scale_ratio,
            ref_abs_max,
        });
    }

    Ok(CutlassMoeSmokeReport {
        num_experts: num_groups,
        experts,
        workspace_bytes,
        gemm_ms,
        sfa_sfb_bytes,
        cos_sim_threshold: COS_SIM_THRESHOLD,
        passed: all_passed,
    })
}

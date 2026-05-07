/// Integration test for the wgpu NVFP4 dequant shader.
///
/// Verifies that GPU-side dequant matches the CPU reference (CUDA-side
/// helpers `decode_nvfp4_nibble` × `decode_ue4m3_half`) within f32 round-
/// trip tolerance. Skipped by default; set `AEGIS_WGPU_SMOKE=1` to run on
/// a host with Vulkan / Metal / D3D12 support.
#[test]
fn wgpu_dequant_nvfp4_matches_cpu_reference() {
    if std::env::var("AEGIS_WGPU_SMOKE").is_err() {
        eprintln!("skipping wgpu_dequant_nvfp4 test; set AEGIS_WGPU_SMOKE=1 to run");
        return;
    }

    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    // Small but realistic shape: rows×cols multiple of 16 along cols.
    let rows = 32_usize;
    let cols = 64_usize;
    let mut rng = SmallRng::seed_from_u64(0xDEC0_DE17);
    let packed: Vec<u8> = (0..rows * cols / 2).map(|_| rng.random::<u8>()).collect();
    let scales: Vec<u8> = (0..rows * cols / 16).map(|_| rng.random::<u8>()).collect();
    let output_scale: f32 = 0.873;

    // CPU reference. Mirrors CUDA `linear_utils.cuh` exactly.
    fn decode_nibble(n: u8) -> i32 {
        match n & 0xF {
            0 => 0,  1 => 1,  2 => 2,  3 => 3,
            4 => 4,  5 => 6,  6 => 8,  7 => 12,
            8 => 0,  9 => -1, 10 => -2, 11 => -3,
            12 => -4, 13 => -6, 14 => -8, _ => -12,
        }
    }
    fn decode_ue4m3(byte: u8) -> f32 {
        let masked = byte & 0x7F;
        if masked == 0 || masked == 0x7F { return 0.0; }
        let exponent = ((masked >> 3) & 0x0F) as i32;
        let mantissa = (masked & 0x07) as f32;
        let raw = if exponent == 0 {
            mantissa * 0.001953125
        } else {
            (1.0 + mantissa * 0.125) * (exponent as f32 - 7.0).exp2()
        };
        raw * 0.5
    }
    let cpu_out: Vec<f32> = (0..rows * cols)
        .map(|i| {
            let row = i / cols;
            let col = i % cols;
            let packed_idx = row * (cols / 2) + col / 2;
            let byte = packed[packed_idx];
            let nibble = if col & 1 == 0 { byte & 0x0F } else { byte >> 4 };
            let scale_idx = row * (cols / 16) + col / 16;
            let scale = decode_ue4m3(scales[scale_idx]);
            (decode_nibble(nibble) as f32) * scale * output_scale
        })
        .collect();

    // GPU
    let ctx = aegisllm::WgpuContext::new(0).expect("WgpuContext::new");
    let gpu_out = aegisllm::dequant_nvfp4_gpu(&ctx, &packed, &scales, rows, cols, output_scale)
        .expect("dequant_nvfp4_gpu");

    assert_eq!(gpu_out.len(), cpu_out.len());
    let max_diff = gpu_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    // f32 round-trip + exp2 — anything above 1e-5 indicates a real
    // semantic mismatch (bad nibble decode or scale formula).
    assert!(
        max_diff < 1e-5,
        "wgpu dequant_nvfp4 max abs diff {max_diff} exceeds 1e-5"
    );
}

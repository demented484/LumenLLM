/// Integration test for the wgpu RMS-norm kernel.
/// Skipped by default; set AEGIS_WGPU_SMOKE=1 to run on a host with
/// Vulkan / Metal / D3D12 support.
#[test]
fn wgpu_rms_norm_matches_cpu_reference() {
    if std::env::var("AEGIS_WGPU_SMOKE").is_err() {
        eprintln!("skipping wgpu_rms_norm test; set AEGIS_WGPU_SMOKE=1 to run");
        return;
    }

    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    let len = 4096_usize;
    let eps = 1e-5_f32;
    let mut rng = SmallRng::seed_from_u64(0xAE61_5ABC);
    let input: Vec<f32> = (0..len).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
    let weight: Vec<f32> = (0..len).map(|_| rng.random::<f32>()).collect();

    // GPU result
    let ctx = aegisllm::WgpuContext::new(0).expect("failed to create WgpuContext");
    let gpu_out = aegisllm::rms_norm_gpu(&ctx, &input, &weight, eps).expect("rms_norm_gpu failed");

    // CPU reference
    let mean_sq: f32 = input.iter().map(|v| v * v).sum::<f32>() / len as f32;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    let cpu_out: Vec<f32> = input
        .iter()
        .zip(weight.iter())
        .map(|(x, w)| x * scale * w)
        .collect();

    assert_eq!(gpu_out.len(), cpu_out.len());
    let max_diff = gpu_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(a, b): (&f32, &f32)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff < 1e-4,
        "wgpu rms_norm max abs diff {max_diff} exceeds 1e-4"
    );
}

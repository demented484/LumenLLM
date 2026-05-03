/// Integration test for the wgpu f32 matmul kernel.
/// Skipped by default; set AEGIS_WGPU_SMOKE=1 to run on a host with
/// Vulkan / Metal / D3D12 support.
#[test]
fn wgpu_matmul_matches_cpu_reference() {
    if std::env::var("AEGIS_WGPU_SMOKE").is_err() {
        eprintln!("skipping wgpu_matmul test; set AEGIS_WGPU_SMOKE=1 to run");
        return;
    }

    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    // Modest sizes that exercise both dimensions but stay under 1MB per buffer.
    let m = 64_usize;
    let n = 96_usize;
    let k = 128_usize;
    let mut rng = SmallRng::seed_from_u64(0xC0FF_EE42);
    let a: Vec<f32> = (0..m * k).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
    let b: Vec<f32> = (0..n * k).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();

    let ctx = aegisllm::WgpuContext::new(0).expect("failed to create WgpuContext");
    let gpu_out = aegisllm::matmul_f32_gpu(&ctx, &a, &b, m, n, k).expect("matmul_f32_gpu failed");

    // CPU reference: c[r,col] = sum_k a[r,k] * b[col,k]
    let mut cpu_out = vec![0.0_f32; m * n];
    for r in 0..m {
        for col in 0..n {
            let mut acc = 0.0_f32;
            for ki in 0..k {
                acc += a[r * k + ki] * b[col * k + ki];
            }
            cpu_out[r * n + col] = acc;
        }
    }

    assert_eq!(gpu_out.len(), cpu_out.len());
    let max_diff = gpu_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(a, b): (&f32, &f32)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let scale_max = cpu_out.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
    assert!(
        max_diff < 1e-3 * scale_max.max(1.0),
        "wgpu matmul max abs diff {max_diff} exceeds tolerance (scale {scale_max})"
    );
}

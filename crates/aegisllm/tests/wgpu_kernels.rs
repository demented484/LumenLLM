/// Integration tests for the wgpu compute kernels.
/// Skipped by default; set AEGIS_WGPU_SMOKE=1 to run on a host with
/// Vulkan / Metal / D3D12 support.
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

fn smoke_enabled() -> bool {
    std::env::var("AEGIS_WGPU_SMOKE").is_ok()
}

fn ctx() -> aegisllm::WgpuContext {
    aegisllm::WgpuContext::new(0).expect("failed to create WgpuContext")
}

#[test]
fn wgpu_swiglu_matches_cpu_reference() {
    if !smoke_enabled() {
        eprintln!("skipping wgpu_swiglu test; set AEGIS_WGPU_SMOKE=1 to run");
        return;
    }
    let len = 4096_usize;
    let mut rng = SmallRng::seed_from_u64(0xA1B2_C3D4);
    let gate: Vec<f32> = (0..len).map(|_| rng.random::<f32>() * 4.0 - 2.0).collect();
    let up: Vec<f32> = (0..len).map(|_| rng.random::<f32>() * 4.0 - 2.0).collect();

    let gpu_out = aegisllm::swiglu_gpu(&ctx(), &gate, &up).expect("swiglu_gpu failed");

    let cpu_out: Vec<f32> = gate
        .iter()
        .zip(up.iter())
        .map(|(g, u)| {
            let silu = g / (1.0 + (-g).exp());
            silu * u
        })
        .collect();

    assert_eq!(gpu_out.len(), cpu_out.len());
    let max_diff = gpu_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(a, b): (&f32, &f32)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_diff < 5e-5, "wgpu swiglu max abs diff {max_diff} >= 5e-5");
}

#[test]
fn wgpu_residual_add_matches_cpu_reference() {
    if !smoke_enabled() {
        eprintln!("skipping wgpu_residual_add test; set AEGIS_WGPU_SMOKE=1 to run");
        return;
    }
    let len = 8192_usize;
    let mut rng = SmallRng::seed_from_u64(0xDEAD_BEEF);
    let a: Vec<f32> = (0..len).map(|_| rng.random::<f32>() * 4.0 - 2.0).collect();
    let b: Vec<f32> = (0..len).map(|_| rng.random::<f32>() * 4.0 - 2.0).collect();

    let gpu_out = aegisllm::residual_add_gpu(&ctx(), &a, &b).expect("residual_add_gpu failed");

    for i in 0..len {
        let expected = a[i] + b[i];
        assert!(
            (gpu_out[i] - expected).abs() < 1e-5,
            "mismatch at {i}: gpu={} expected={}",
            gpu_out[i],
            expected
        );
    }
}

#[test]
fn wgpu_embedding_matches_cpu_reference() {
    if !smoke_enabled() {
        eprintln!("skipping wgpu_embedding test; set AEGIS_WGPU_SMOKE=1 to run");
        return;
    }
    let vocab = 256_usize;
    let hidden = 1024_usize;
    let mut rng = SmallRng::seed_from_u64(0x4242_BEEF);
    let table: Vec<f32> = (0..vocab * hidden).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
    let token_id: u32 = 137;

    let gpu_out = aegisllm::embedding_gpu(&ctx(), &table, token_id, hidden)
        .expect("embedding_gpu failed");

    let expected = &table[(token_id as usize) * hidden..(token_id as usize + 1) * hidden];
    assert_eq!(gpu_out.len(), hidden);
    for i in 0..hidden {
        assert_eq!(gpu_out[i], expected[i], "mismatch at {i}");
    }
}

#[test]
fn wgpu_decode_attention_matches_cpu_reference() {
    if !smoke_enabled() {
        eprintln!("skipping wgpu_decode_attention test; set AEGIS_WGPU_SMOKE=1 to run");
        return;
    }
    let num_q_heads = 4_usize;
    let num_kv_heads = 2_usize;
    let head_dim = 64_usize;
    let seq_len = 16_usize;
    let group_size = num_q_heads / num_kv_heads;

    let mut rng = SmallRng::seed_from_u64(0xA77E_BEEF_u64.wrapping_mul(7));
    let q: Vec<f32> = (0..num_q_heads * head_dim)
        .map(|_| rng.random::<f32>() * 2.0 - 1.0)
        .collect();
    let kv_width = num_kv_heads * head_dim;
    let keys: Vec<f32> = (0..seq_len * kv_width)
        .map(|_| rng.random::<f32>() * 2.0 - 1.0)
        .collect();
    let values: Vec<f32> = (0..seq_len * kv_width)
        .map(|_| rng.random::<f32>() * 2.0 - 1.0)
        .collect();

    let gpu_out = aegisllm::decode_attention_gpu(
        &ctx(), &q, &keys, &values, num_q_heads, num_kv_heads, head_dim, seq_len,
    )
    .expect("decode_attention_gpu failed");

    // CPU reference: online softmax per head.
    let scale = 1.0_f32 / (head_dim as f32).sqrt();
    let mut cpu_out = vec![0.0_f32; num_q_heads * head_dim];
    for q_head in 0..num_q_heads {
        let kv_head = q_head / group_size;
        let q_slice = &q[q_head * head_dim..(q_head + 1) * head_dim];
        let mut max_score = f32::NEG_INFINITY;
        let mut sum = 0.0_f32;
        let mut acc = vec![0.0_f32; head_dim];
        for pos in 0..seq_len {
            let k_slice = &keys[(pos * kv_width + kv_head * head_dim)..(pos * kv_width + (kv_head + 1) * head_dim)];
            let dot: f32 = q_slice.iter().zip(k_slice.iter()).map(|(a, b)| a * b).sum();
            let score = dot * scale;
            if score > max_score {
                let rescale = (max_score - score).exp();
                for v in acc.iter_mut() { *v *= rescale; }
                sum *= rescale;
                max_score = score;
            }
            let weight = (score - max_score).exp();
            sum += weight;
            let v_slice = &values[(pos * kv_width + kv_head * head_dim)..(pos * kv_width + (kv_head + 1) * head_dim)];
            for (a, &v) in acc.iter_mut().zip(v_slice.iter()) {
                *a += weight * v;
            }
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        for (i, v) in acc.iter().enumerate() {
            cpu_out[q_head * head_dim + i] = v * inv;
        }
    }

    let max_diff = gpu_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(a, b): (&f32, &f32)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff < 5e-5,
        "wgpu decode_attention max abs diff {max_diff} >= 5e-5"
    );
}

#[test]
fn wgpu_rope_matches_cpu_reference() {
    if !smoke_enabled() {
        eprintln!("skipping wgpu_rope test; set AEGIS_WGPU_SMOKE=1 to run");
        return;
    }
    let num_heads = 8_usize;
    let head_dim = 128_usize;
    let half = head_dim / 2;
    let mut rng = SmallRng::seed_from_u64(0x1234_5678);
    let values: Vec<f32> = (0..num_heads * head_dim)
        .map(|_| rng.random::<f32>() * 2.0 - 1.0)
        .collect();
    // Build cos/sin tables as if for some position.
    let cos_table: Vec<f32> = (0..half).map(|i| (i as f32 * 0.01).cos()).collect();
    let sin_table: Vec<f32> = (0..half).map(|i| (i as f32 * 0.01).sin()).collect();

    let gpu_out = aegisllm::rope_gpu(&ctx(), &values, &cos_table, &sin_table, num_heads, head_dim)
        .expect("rope_gpu failed");

    // CPU reference: rotate each head's pairs.
    let mut cpu_out = values.clone();
    for h in 0..num_heads {
        for i in 0..half {
            let lo = h * head_dim + i;
            let hi = lo + half;
            let x0 = values[lo];
            let x1 = values[hi];
            let c = cos_table[i];
            let s = sin_table[i];
            cpu_out[lo] = x0 * c - x1 * s;
            cpu_out[hi] = x0 * s + x1 * c;
        }
    }

    let max_diff = gpu_out
        .iter()
        .zip(cpu_out.iter())
        .map(|(a, b): (&f32, &f32)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_diff < 1e-5, "wgpu rope max abs diff {max_diff} >= 1e-5");
}

//! End-to-end token-forward test: tiny synthetic Llama model on the
//! wgpu backend matches a faithful CPU reference within 1e-3.
//!
//! Bypasses the artifact loader and constructs `WgpuModel` directly from
//! in-memory random weights, so this test isolates the multi-layer
//! orchestration (`forward_token_device`) from the safetensors-parsing
//! path. The artifact loader is exercised separately via unit tests.
//!
//! Gated behind `AEGIS_WGPU_SMOKE=1`; on hosts without Vulkan/Metal/
//! D3D12 the test prints a "skipping" line and returns success.

use std::sync::Arc;

use aegisllm_wgpu::{
    forward_token_device, upload_f32_buf, Activation, WgpuAttentionWeightsFull, WgpuContext,
    WgpuLayerWeights, WgpuLinear, WgpuMlpWeightsFull, WgpuModel, WgpuModelState,
};

/// Deterministic-random f32 vector.
fn det_rand(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15);
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let bits = (state >> 33) as u32;
            // Map to roughly [-0.5, 0.5).
            (bits as f32 / u32::MAX as f32) - 0.5
        })
        .collect()
}

/// CPU reference: full Llama-style forward for one token at `position`.
/// Mutates `kv_keys` / `kv_values` (one entry per layer, each
/// [max_seq, kv_width]) by appending this position's K/V. Returns
/// the post-`lm_head` logits (length `vocab`).
#[allow(clippy::too_many_arguments)]
struct LlamaShape {
    h: usize,
    inter: usize,
    nq: usize,
    nkv: usize,
    hd: usize,
    max_seq: usize,
    num_layers: usize,
    vocab: usize,
    eps: f32,
}

#[allow(clippy::too_many_arguments)]
fn cpu_llama_forward(
    s: &LlamaShape,
    embed_table: &[f32],     // [vocab, h]
    final_norm_w: &[f32],    // [h]
    lm_head_w: &[f32],       // [vocab, h]
    layers: &[CpuLayer],
    kv_keys: &mut [Vec<f32>],   // [num_layers] of [max_seq * kv_width]
    kv_values: &mut [Vec<f32>], // same
    position: usize,
    token_id: u32,
    cos: &[f32],
    sin: &[f32],
) -> Vec<f32> {
    let kv_width = s.nkv * s.hd;
    let q_width = s.nq * s.hd;
    let half = s.hd / 2;

    // Embedding lookup.
    let mut residual: Vec<f32> = embed_table[token_id as usize * s.h..(token_id as usize + 1) * s.h]
        .to_vec();

    for (layer_idx, layer) in layers.iter().enumerate() {
        // ── Attention ────────────────────────────────────────────────
        let mean_sq = residual.iter().map(|v| v * v).sum::<f32>() / s.h as f32;
        let inv_rms = 1.0 / (mean_sq + s.eps).sqrt();
        let normed: Vec<f32> = residual
            .iter()
            .zip(layer.attn_norm.iter())
            .map(|(v, w)| v * inv_rms * w)
            .collect();
        let mut q = vec![0.0_f32; q_width];
        let mut k = vec![0.0_f32; kv_width];
        let mut v = vec![0.0_f32; kv_width];
        for r in 0..q_width {
            let mut acc = 0.0;
            for c in 0..s.h {
                acc += normed[c] * layer.q_proj[r * s.h + c];
            }
            q[r] = acc;
        }
        for r in 0..kv_width {
            let mut ak = 0.0;
            let mut av = 0.0;
            for c in 0..s.h {
                ak += normed[c] * layer.k_proj[r * s.h + c];
                av += normed[c] * layer.v_proj[r * s.h + c];
            }
            k[r] = ak;
            v[r] = av;
        }
        // RoPE.
        for head in 0..s.nq {
            for i in 0..half {
                let lo = q[head * s.hd + i];
                let hi = q[head * s.hd + i + half];
                q[head * s.hd + i] = lo * cos[i] - hi * sin[i];
                q[head * s.hd + i + half] = lo * sin[i] + hi * cos[i];
            }
        }
        for head in 0..s.nkv {
            for i in 0..half {
                let lo = k[head * s.hd + i];
                let hi = k[head * s.hd + i + half];
                k[head * s.hd + i] = lo * cos[i] - hi * sin[i];
                k[head * s.hd + i + half] = lo * sin[i] + hi * cos[i];
            }
        }
        // Cache write.
        kv_keys[layer_idx][position * kv_width..(position + 1) * kv_width].copy_from_slice(&k);
        kv_values[layer_idx][position * kv_width..(position + 1) * kv_width]
            .copy_from_slice(&v);
        // Online softmax attention over [0..=position].
        let scale = 1.0_f32 / (s.hd as f32).sqrt();
        let group = s.nq / s.nkv;
        let mut attn = vec![0.0_f32; q_width];
        for qh in 0..s.nq {
            let kvh = qh / group;
            let q_base = qh * s.hd;
            let seq = position + 1;
            let mut scores = vec![0.0_f32; seq];
            let mut max_s = f32::NEG_INFINITY;
            for p in 0..seq {
                let k_base = p * kv_width + kvh * s.hd;
                let mut dot = 0.0_f32;
                for i in 0..s.hd {
                    dot += q[q_base + i] * kv_keys[layer_idx][k_base + i];
                }
                scores[p] = dot * scale;
                if scores[p] > max_s {
                    max_s = scores[p];
                }
            }
            let mut sum = 0.0;
            let exps: Vec<f32> = scores
                .iter()
                .map(|sc| {
                    let e = (sc - max_s).exp();
                    sum += e;
                    e
                })
                .collect();
            for p in 0..seq {
                let w = exps[p] / sum;
                let v_base = p * kv_width + kvh * s.hd;
                for i in 0..s.hd {
                    attn[q_base + i] += w * kv_values[layer_idx][v_base + i];
                }
            }
        }
        // O proj.
        let mut o_out = vec![0.0_f32; s.h];
        for r in 0..s.h {
            let mut acc = 0.0;
            for c in 0..q_width {
                acc += attn[c] * layer.o_proj[r * q_width + c];
            }
            o_out[r] = acc;
        }
        for i in 0..s.h {
            residual[i] += o_out[i];
        }
        // ── MLP ─────────────────────────────────────────────────────
        let mean_sq = residual.iter().map(|v| v * v).sum::<f32>() / s.h as f32;
        let inv_rms = 1.0 / (mean_sq + s.eps).sqrt();
        let normed: Vec<f32> = residual
            .iter()
            .zip(layer.mlp_norm.iter())
            .map(|(v, w)| v * inv_rms * w)
            .collect();
        let mut gate = vec![0.0_f32; s.inter];
        let mut up = vec![0.0_f32; s.inter];
        for r in 0..s.inter {
            let mut g = 0.0;
            let mut u = 0.0;
            for c in 0..s.h {
                g += normed[c] * layer.gate_proj[r * s.h + c];
                u += normed[c] * layer.up_proj[r * s.h + c];
            }
            gate[r] = g;
            up[r] = u;
        }
        let swig: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
            .collect();
        let mut down_out = vec![0.0_f32; s.h];
        for r in 0..s.h {
            let mut acc = 0.0;
            for c in 0..s.inter {
                acc += swig[c] * layer.down_proj[r * s.inter + c];
            }
            down_out[r] = acc;
        }
        for i in 0..s.h {
            residual[i] += down_out[i];
        }
    }
    // Final norm + lm_head.
    let mean_sq = residual.iter().map(|v| v * v).sum::<f32>() / s.h as f32;
    let inv_rms = 1.0 / (mean_sq + s.eps).sqrt();
    let final_normed: Vec<f32> = residual
        .iter()
        .zip(final_norm_w.iter())
        .map(|(v, w)| v * inv_rms * w)
        .collect();
    let mut logits = vec![0.0_f32; s.vocab];
    for r in 0..s.vocab {
        let mut acc = 0.0;
        for c in 0..s.h {
            acc += final_normed[c] * lm_head_w[r * s.h + c];
        }
        logits[r] = acc;
    }
    logits
}

struct CpuLayer {
    attn_norm: Vec<f32>,
    q_proj: Vec<f32>,
    k_proj: Vec<f32>,
    v_proj: Vec<f32>,
    o_proj: Vec<f32>,
    mlp_norm: Vec<f32>,
    gate_proj: Vec<f32>,
    up_proj: Vec<f32>,
    down_proj: Vec<f32>,
}

#[test]
fn forward_token_matches_cpu_reference_two_tokens_two_layers() {
    if std::env::var("AEGIS_WGPU_SMOKE").is_err() {
        eprintln!("skipping; set AEGIS_WGPU_SMOKE=1 to run on a host with Vulkan/Metal/D3D12");
        return;
    }

    let s = LlamaShape {
        h: 8,
        inter: 16,
        nq: 2,
        nkv: 2,
        hd: 4,
        max_seq: 4,
        num_layers: 2,
        vocab: 12,
        eps: 1e-6,
    };

    // ── Build random weights (deterministic) ─────────────────────────
    let embed = det_rand(s.vocab * s.h, 1);
    let final_norm_w = det_rand(s.h, 2).into_iter().map(|v| v + 1.0).collect::<Vec<_>>();
    let lm_head = det_rand(s.vocab * s.h, 3);
    let q_width = s.nq * s.hd;
    let kv_width = s.nkv * s.hd;
    let mut cpu_layers = Vec::with_capacity(s.num_layers);
    for l in 0..s.num_layers {
        let seed_base = (l as u64 + 1) * 100;
        cpu_layers.push(CpuLayer {
            attn_norm: det_rand(s.h, seed_base + 1).into_iter().map(|v| v + 1.0).collect(),
            q_proj: det_rand(q_width * s.h, seed_base + 2),
            k_proj: det_rand(kv_width * s.h, seed_base + 3),
            v_proj: det_rand(kv_width * s.h, seed_base + 4),
            o_proj: det_rand(s.h * q_width, seed_base + 5),
            mlp_norm: det_rand(s.h, seed_base + 6).into_iter().map(|v| v + 1.0).collect(),
            gate_proj: det_rand(s.inter * s.h, seed_base + 7),
            up_proj: det_rand(s.inter * s.h, seed_base + 8),
            down_proj: det_rand(s.h * s.inter, seed_base + 9),
        });
    }

    // ── Build wgpu model directly from the same weights ──────────────
    let ctx = Arc::new(WgpuContext::new(0).expect("wgpu ctx"));
    let embed_buf = upload_f32_buf(&ctx, &embed, "embed");
    let final_norm_buf = upload_f32_buf(&ctx, &final_norm_w, "final_norm");
    let lm_head_buf = WgpuLinear::Dense {
        weight: upload_f32_buf(&ctx, &lm_head, "lm_head"),
        rows: s.vocab,
        cols: s.h,
    };
    let make_dense = |data: &[f32], rows: usize, cols: usize, label: &'static str| -> WgpuLinear {
        WgpuLinear::Dense {
            weight: upload_f32_buf(&ctx, data, label),
            rows,
            cols,
        }
    };
    let layers: Vec<WgpuLayerWeights> = cpu_layers
        .iter()
        .map(|cl| WgpuLayerWeights {
            attention: WgpuAttentionWeightsFull {
                norm_weight: upload_f32_buf(&ctx, &cl.attn_norm, "attn_norm"),
                q_proj: make_dense(&cl.q_proj, q_width, s.h, "q_proj"),
                k_proj: make_dense(&cl.k_proj, kv_width, s.h, "k_proj"),
                v_proj: make_dense(&cl.v_proj, kv_width, s.h, "v_proj"),
                o_proj: make_dense(&cl.o_proj, s.h, q_width, "o_proj"),
            },
            mlp: WgpuMlpWeightsFull {
                norm_weight: upload_f32_buf(&ctx, &cl.mlp_norm, "mlp_norm"),
                gate_proj: make_dense(&cl.gate_proj, s.inter, s.h, "gate_proj"),
                up_proj: make_dense(&cl.up_proj, s.inter, s.h, "up_proj"),
                down_proj: make_dense(&cl.down_proj, s.h, s.inter, "down_proj"),
            },
        })
        .collect();
    let model = WgpuModel {
        ctx: ctx.clone(),
        embed_tokens: embed_buf,
        embed_tokens_rows: s.vocab,
        embed_tokens_cols: s.h,
        final_norm: final_norm_buf,
        lm_head: lm_head_buf,
        layers,
        hidden_size: s.h,
        intermediate_size: s.inter,
        num_q_heads: s.nq,
        num_kv_heads: s.nkv,
        head_dim: s.hd,
        vocab_size: s.vocab,
        rms_norm_eps: s.eps,
    };

    // ── Build model state ─────────────────────────────────────────────
    let mut state = WgpuModelState::new(
        &ctx, s.num_layers, s.h, s.inter, s.nq, s.nkv, s.hd, s.vocab, s.max_seq,
    )
    .expect("state");

    // RoPE table generators (Llama-style, theta_base = 10000).
    let theta: Vec<f32> = (0..(s.hd / 2))
        .map(|i| 10000f32.powf(-2.0 * i as f32 / s.hd as f32))
        .collect();
    let cos_for = |pos: usize, half: usize| -> Vec<f32> {
        (0..half).map(|i| (pos as f32 * theta[i]).cos()).collect()
    };
    let sin_for = |pos: usize, half: usize| -> Vec<f32> {
        (0..half).map(|i| (pos as f32 * theta[i]).sin()).collect()
    };

    // CPU-side mirror state.
    let mut cpu_keys: Vec<Vec<f32>> = (0..s.num_layers)
        .map(|_| vec![0.0_f32; s.max_seq * kv_width])
        .collect();
    let mut cpu_values: Vec<Vec<f32>> = (0..s.num_layers)
        .map(|_| vec![0.0_f32; s.max_seq * kv_width])
        .collect();

    // Token sequence: first decode token id=3 at position=0, then id=7 at position=1.
    for (step_idx, &token_id) in [3u32, 7u32].iter().enumerate() {
        // GPU side: write the embedding for this token into state.residual.
        // (We're testing forward_token_device which accepts the residual
        // pre-seeded; in a real provider this comes from the embedding
        // shader, but here we mirror what the CPU does and seed directly.)
        let token_embed: Vec<f32> = embed[token_id as usize * s.h..(token_id as usize + 1) * s.h]
            .to_vec();
        ctx.queue().write_buffer(&state.residual, 0, bytemuck::cast_slice(&token_embed));

        forward_token_device(
            &ctx,
            &model,
            &mut state,
            cos_for,
            sin_for,
            s.eps,
            Activation::SwiGLU,
        )
        .expect("forward_token");

        // CPU reference for this step.
        let cos = cos_for(step_idx, s.hd / 2);
        let sin = sin_for(step_idx, s.hd / 2);
        let cpu_logits = cpu_llama_forward(
            &s, &embed, &final_norm_w, &lm_head, &cpu_layers, &mut cpu_keys, &mut cpu_values,
            step_idx, token_id, &cos, &sin,
        );

        // Read GPU logits.
        let staging = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("e2e_staging"),
            size: (s.vocab * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = ctx
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("e2e_readback"),
            });
        enc.copy_buffer_to_buffer(&state.logits, 0, &staging, 0, (s.vocab * 4) as u64);
        ctx.queue().submit(std::iter::once(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        ctx.device().poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let gpu_logits: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();

        for (i, (g, c)) in gpu_logits.iter().zip(cpu_logits.iter()).enumerate() {
            assert!(
                (g - c).abs() < 1e-3,
                "step {step_idx}: logits mismatch at i={i}: gpu={g} cpu={c}",
            );
        }
    }
}

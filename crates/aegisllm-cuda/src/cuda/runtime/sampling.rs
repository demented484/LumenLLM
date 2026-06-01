use cudarc::driver::{LaunchConfig, PushKernelArg};

use super::{CudaRuntime, ceil_div, map_cuda_err};
use crate::cuda::DeviceBuffer;
use aegisllm_base::error::{AegisError, Result};

/// Top-k capacity of the GPU sampler — MUST match `SAMPLER_KCAP` in
/// `kernels/blackwell/sampling.cu`. The config's `top_k` (≤50) is clamped to
/// this; larger `top_k` requests silently use this many candidates (the CPU
/// reference also only ever truncates to `top_k`, so as long as KCAP ≥ top_k
/// the candidate sets are identical).
pub const SAMPLER_KCAP: u32 = 64;

/// Thread count for the single-block fused sampler. Each round of the iterated
/// top-k is a block-parallel reduction over the full vocab, so more threads =
/// fewer reads/thread; 1024 keeps a 248K-vocab round at ~243 reads/thread with a
/// 10-step shared reduction. Shared mem = `block_dim*(f32+u32) + KCAP*(f32+u32)`,
/// well within the SM cap.
pub const SAMPLER_BLOCK_DIM: u32 = 1024;

impl CudaRuntime {
    /// On-device multinomial sampler — replaces the per-token DtoH of the full
    /// vocab logit vector + CPU top-k/top-p/min-p sort with ONE fused
    /// single-block kernel. Selects the global top-`top_k` by iterated
    /// block-parallel argmax, then applies temperature → top-p → min-p →
    /// renormalise → multinomial draw with the host-supplied uniform `u`
    /// (byte-for-byte the CPU `sample_next_token` order + draw semantics), and
    /// writes the sampled token id to `out_token[0]`. Caller downloads a single
    /// u32 instead of 248K floats.
    ///
    /// `u` MUST be the same `rand::random::<f32>()` draw the CPU path would use,
    /// so a given RNG draw selects the same token on GPU and CPU.
    pub fn sample_token_device(
        &self,
        logits: &DeviceBuffer<f32>,
        top_k: usize,
        temperature: f32,
        top_p: f32,
        min_p: f32,
        u: f32,
        out_token: &mut DeviceBuffer<u32>,
    ) -> Result<()> {
        if logits.is_empty() || out_token.len() != 1 {
            return Err(AegisError::InvalidPlan(format!(
                "sampler shape mismatch: logits={} output={}",
                logits.len(),
                out_token.len()
            )));
        }
        let vocab = logits.len() as u32;
        let top_k_u = top_k as u32;
        let block_dim = SAMPLER_BLOCK_DIM;
        // Shared: reduction scratch (block_dim × (val,idx)) + winner set
        // (KCAP × (val,idx)). All f32/u32 (4 bytes each).
        let shared = (block_dim + SAMPLER_KCAP) * (std::mem::size_of::<f32>() as u32 * 2);
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: shared,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.sampler_topk_fused)
                .arg(&logits.slice)
                .arg(&vocab)
                .arg(&top_k_u)
                .arg(&temperature)
                .arg(&top_p)
                .arg(&min_p)
                .arg(&u)
                .arg(&mut out_token.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch fused gpu sampler"))?;
        Ok(())
    }

    pub fn argmax_f32_device(
        &self,
        logits: &DeviceBuffer<f32>,
        block_values: &mut DeviceBuffer<f32>,
        block_indices: &mut DeviceBuffer<u32>,
        output_token: &mut DeviceBuffer<u32>,
    ) -> Result<()> {
        if logits.is_empty() || output_token.len() != 1 {
            return Err(AegisError::InvalidPlan(format!(
                "argmax shape mismatch: logits={} output={}",
                logits.len(),
                output_token.len()
            )));
        }
        let blocks = ceil_div(logits.len() as u32, 256);
        if block_values.len() != blocks as usize || block_indices.len() != blocks as usize {
            return Err(AegisError::InvalidPlan(format!(
                "argmax scratch mismatch: expected blocks={} values={} indices={}",
                blocks,
                block_values.len(),
                block_indices.len()
            )));
        }
        let len = logits.len() as u32;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.argmax_blocks)
                .arg(&logits.slice)
                .arg(&len)
                .arg(&mut block_values.slice)
                .arg(&mut block_indices.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch argmax block reduce"))?;

        let finalize_cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.argmax_finalize)
                .arg(&block_values.slice)
                .arg(&block_indices.slice)
                .arg(&blocks)
                .arg(&mut output_token.slice)
                .launch(finalize_cfg)
        }
        .map_err(map_cuda_err("launch argmax finalize"))?;
        Ok(())
    }

    /// Speculative-decoding sparse lm_head matvec.
    ///
    /// Evaluates the dense BF16 lm_head ONLY over the explicit list of
    /// `candidate_rows` token ids: `logits[i] = lm_head[candidate_rows[i], :] ·
    /// hidden`. One block per candidate (mirrors `aegis_bf16_matvec_reference`).
    ///
    /// `lm_head` MUST be VRAM-resident — the candidate-gather kernel indexes the
    /// matrix rows directly on device. The draft's tied embed/lm_head is small
    /// (262144 × 256 BF16 ≈ 134 MiB) so it always loads VRAM-resident.
    ///
    /// TODO(gpu-verify): the centroid → candidate-row mapping is computed on the
    /// host (see `executor::speculative`); this kernel only consumes the row
    /// list, so verify the row ids against a reference centroid decode.
    pub fn spec_sparse_lm_head_matvec_device(
        &self,
        lm_head: &crate::cuda::DeviceBf16Matrix,
        hidden: &DeviceBuffer<f32>,
        candidate_rows: &DeviceBuffer<u32>,
        num_candidates: usize,
        logits: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if lm_head.is_host_resident() {
            return Err(AegisError::InvalidPlan(format!(
                "spec sparse lm_head `{}` must be VRAM-resident",
                lm_head.name
            )));
        }
        if hidden.len() < lm_head.cols {
            return Err(AegisError::InvalidPlan(format!(
                "spec sparse lm_head hidden too small: have {} need {}",
                hidden.len(),
                lm_head.cols
            )));
        }
        if candidate_rows.len() < num_candidates || logits.len() < num_candidates {
            return Err(AegisError::InvalidPlan(format!(
                "spec sparse lm_head buffer mismatch: candidates={} rows_cap={} logits_cap={}",
                num_candidates,
                candidate_rows.len(),
                logits.len()
            )));
        }
        if num_candidates == 0 {
            return Ok(());
        }
        let cols = lm_head.cols as u32;
        let n = num_candidates as u32;
        let block_dim = 256u32;
        let cfg = LaunchConfig {
            grid_dim: (n, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: block_dim * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.spec_sparse_lm_head_matvec)
                .arg(lm_head.values_u16())
                .arg(&hidden.slice)
                .arg(&candidate_rows.slice)
                .arg(&n)
                .arg(&cols)
                .arg(&mut logits.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch spec sparse lm_head matvec"))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::runtime::CudaRuntime;

    /// Byte-for-byte re-implementation of the CPU `sample_next_token`
    /// (`aegisllm-base::executor::generation`) with the `rand::random::<f32>()`
    /// draw REPLACED by an injected `u`. This is the correctness oracle: we run
    /// the GPU sampler with the *same* `u` and assert the same token. The op
    /// order is identical — temperature → top_k (pre-exp logit sort/truncate) →
    /// exp weights → top_p → min_p → renormalise → cumulative draw. Kept here
    /// (not shared) so a refactor of the production CPU path can't silently move
    /// the oracle with it.
    fn cpu_reference_with_u(
        logits: &[f32],
        top_k: usize,
        temperature: f32,
        top_p: f32,
        min_p: f32,
        u: f32,
    ) -> usize {
        let argmax = |xs: &[f32]| -> usize {
            xs.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap_or(0)
        };
        if temperature <= 0.0 || top_k == 1 {
            return argmax(logits);
        }
        let temperature = temperature.max(1e-6);
        let mut candidates: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .filter_map(|(idx, &l)| l.is_finite().then_some((idx, l)))
            .collect();
        if candidates.is_empty() {
            return argmax(logits);
        }
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
        if top_k > 0 && top_k < candidates.len() {
            candidates.truncate(top_k);
        }
        let max_logit = candidates
            .iter()
            .map(|(_, l)| *l)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut weighted: Vec<(usize, f32)> = candidates
            .into_iter()
            .map(|(idx, l)| (idx, ((l - max_logit) / temperature).exp()))
            .filter(|(_, w)| w.is_finite() && *w > 0.0)
            .collect();
        if weighted.is_empty() {
            return argmax(logits);
        }
        let total: f32 = weighted.iter().map(|(_, w)| *w).sum();
        if top_p > 0.0 && top_p < 1.0 && total > 0.0 {
            let mut cum = 0.0f32;
            let cutoff = total * top_p;
            let mut keep = 0usize;
            for (_, w) in &weighted {
                cum += *w;
                keep += 1;
                if cum >= cutoff {
                    break;
                }
            }
            weighted.truncate(keep.max(1));
        }
        if min_p > 0.0 {
            let max_weight = weighted[0].1;
            let cutoff = max_weight * min_p;
            weighted.retain(|(_, w)| *w >= cutoff);
            if weighted.is_empty() {
                return argmax(logits);
            }
        }
        let total: f32 = weighted.iter().map(|(_, w)| *w).sum();
        if total <= 0.0 {
            return argmax(logits);
        }
        let mut draw = u * total;
        for (idx, w) in weighted {
            if draw <= w {
                return idx;
            }
            draw -= w;
        }
        argmax(logits)
    }

    /// Run the GPU sampler once for `logits` with the given config + uniform `u`.
    fn gpu_sample(
        rt: &CudaRuntime,
        logits: &[f32],
        top_k: usize,
        temperature: f32,
        top_p: f32,
        min_p: f32,
        u: f32,
    ) -> usize {
        let dlogits = rt.upload_f32(logits).expect("upload logits");
        let mut out = rt.alloc_u32(1).expect("out token");
        rt.sample_token_device(&dlogits, top_k, temperature, top_p, min_p, u, &mut out)
            .expect("gpu sample");
        rt.download_u32(&out).expect("download token")[0] as usize
    }

    /// A small deterministic LCG so the test is reproducible without pulling a
    /// seeded RNG dependency into the test (avoids flakiness across rand
    /// versions). Returns f32 in [0,1).
    struct Lcg(u64);
    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            // Numerical Recipes LCG constants.
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
        fn next_f32(&mut self) -> f32 {
            (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
        }
        /// Logit in roughly [-8, 8].
        fn next_logit(&mut self) -> f32 {
            self.next_f32() * 16.0 - 8.0
        }
    }

    /// Same-`u` distribution-correctness: across a battery of random logit
    /// vectors and the full sampling-config grid, the GPU sampler must select
    /// the EXACT same token as the CPU reference given the same uniform draw.
    /// This validates the on-device top-k extraction, the temperature/top-p/
    /// min-p filter order, and the multinomial walk all match the CPU semantics.
    #[test]
    fn gpu_sampler_matches_cpu_reference_same_u() {
        let rt = match CudaRuntime::new(0) {
            Ok(rt) => rt,
            Err(e) => {
                eprintln!("skip gpu_sampler_matches_cpu_reference_same_u: {e:?}");
                return;
            }
        };

        // Config grid: covers greedy-ish (top_k small) through wide nucleus,
        // including min-p on/off and the production qwen config (1.0/0.95/50/0.05).
        let configs: &[(usize, f32, f32, f32)] = &[
            (50, 1.0, 0.95, 0.05), // production qwen3.6 sampling
            (40, 0.8, 0.9, 0.0),
            (64, 1.2, 1.0, 0.1),   // top_p disabled (==1.0)
            (10, 0.7, 0.95, 0.0),
            (50, 1.0, 1.0, 0.0),   // pure top-k
            (50, 2.0, 0.5, 0.2),   // aggressive min-p + tight nucleus
            (2, 1.0, 1.0, 0.0),    // tiny top-k
            (50, 0.5, 0.99, 0.0),
        ];

        // Several vocab sizes incl. a large one near the qwen 248K vocab so the
        // 256-block striding + 64-cap merge is exercised at scale.
        let vocabs = [257usize, 1024, 4096, 65_536, 248_320];

        let mut lcg = Lcg(0x1234_5678_9abc_def0);
        let mut total_cases = 0usize;
        let mut mismatches = 0usize;
        for &vocab in &vocabs {
            for trial in 0..4 {
                let logits: Vec<f32> = (0..vocab).map(|_| lcg.next_logit()).collect();
                for &(top_k, temperature, top_p, min_p) in configs {
                    // A handful of uniform draws per case to exercise different
                    // points of the cumulative walk.
                    for _ in 0..6 {
                        let u = lcg.next_f32();
                        let cpu = cpu_reference_with_u(&logits, top_k, temperature, top_p, min_p, u);
                        let gpu = gpu_sample(&rt, &logits, top_k, temperature, top_p, min_p, u);
                        total_cases += 1;
                        if cpu != gpu {
                            mismatches += 1;
                            if mismatches <= 12 {
                                eprintln!(
                                    "MISMATCH vocab={vocab} trial={trial} k={top_k} t={temperature} \
                                     p={top_p} mp={min_p} u={u}: cpu={cpu} gpu={gpu} \
                                     (cpu_logit={} gpu_logit={})",
                                    logits[cpu], logits[gpu]
                                );
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(
            mismatches, 0,
            "{mismatches}/{total_cases} GPU-vs-CPU same-u sampling mismatches"
        );
        eprintln!("gpu_sampler same-u: {total_cases} cases, 0 mismatches");
    }

    /// Duplicate-logit / tie-breaking correctness: when many tokens share the
    /// same logit value, the GPU top-k must rank ties by SMALLER index exactly
    /// like the CPU's stable descending sort, and pick the same token.
    #[test]
    fn gpu_sampler_handles_ties_like_cpu() {
        let Ok(rt) = CudaRuntime::new(0) else {
            eprintln!("skip gpu_sampler_handles_ties_like_cpu: no CUDA device");
            return;
        };
        let vocab = 5000usize;
        // Big plateau of equal logits + a few distinct peaks, so top-k must cut
        // through the tie plateau and tie-breaking (lower index) decides the set.
        let mut logits = vec![1.0f32; vocab];
        logits[4000] = 5.0;
        logits[10] = 5.0; // equal peak at a lower index — must out-rank 4000 on tie
        logits[2500] = 3.0;
        let mut lcg = Lcg(0xdead_beef_cafe_0001);
        for &(top_k, temperature, top_p, min_p) in
            &[(50usize, 1.0f32, 0.95f32, 0.05f32), (3, 1.0, 1.0, 0.0), (50, 1.0, 0.3, 0.0)]
        {
            for _ in 0..32 {
                let u = lcg.next_f32();
                let cpu = cpu_reference_with_u(&logits, top_k, temperature, top_p, min_p, u);
                let gpu = gpu_sample(&rt, &logits, top_k, temperature, top_p, min_p, u);
                assert_eq!(cpu, gpu, "tie-break mismatch at u={u} k={top_k} p={top_p} mp={min_p}");
            }
        }
    }

    /// Empirical histogram test: draw MANY samples from a fixed logit vector with
    /// the GPU sampler (feeding fresh uniforms) and with the CPU reference (same
    /// stream of uniforms), then assert the per-token empirical distributions
    /// match within sampling noise. Because both consume the SAME uniform stream,
    /// they should be near-identical; we still allow a small tolerance for the
    /// measure-zero FP-boundary cases.
    #[test]
    fn gpu_sampler_distribution_matches_cpu_histogram() {
        let Ok(rt) = CudaRuntime::new(0) else {
            eprintln!("skip gpu_sampler_distribution_matches_cpu_histogram: no CUDA device");
            return;
        };
        let vocab = 2048usize;
        let mut lcg = Lcg(0x00c0_ffee_1234_5678);
        let logits: Vec<f32> = (0..vocab).map(|_| lcg.next_logit()).collect();
        let (top_k, temperature, top_p, min_p) = (50usize, 1.0f32, 0.95f32, 0.05f32);

        let draws = 4_000usize;
        let mut cpu_hist = std::collections::HashMap::<usize, u32>::new();
        let mut gpu_hist = std::collections::HashMap::<usize, u32>::new();
        let mut draw_lcg = Lcg(0x9999_1111_2222_3333);
        let mut disagree = 0usize;
        for _ in 0..draws {
            let u = draw_lcg.next_f32();
            let cpu = cpu_reference_with_u(&logits, top_k, temperature, top_p, min_p, u);
            let gpu = gpu_sample(&rt, &logits, top_k, temperature, top_p, min_p, u);
            *cpu_hist.entry(cpu).or_default() += 1;
            *gpu_hist.entry(gpu).or_default() += 1;
            if cpu != gpu {
                disagree += 1;
            }
        }
        // Same-u disagreement must be vanishingly small (FP boundary only).
        let disagree_frac = disagree as f64 / draws as f64;
        assert!(
            disagree_frac < 0.001,
            "{disagree}/{draws} same-u disagreements ({disagree_frac:.4}) — distribution diverged"
        );
        // Total-variation distance between the two empirical distributions.
        let mut tv = 0.0f64;
        let keys: std::collections::HashSet<usize> =
            cpu_hist.keys().chain(gpu_hist.keys()).copied().collect();
        for k in keys {
            let c = *cpu_hist.get(&k).unwrap_or(&0) as f64 / draws as f64;
            let g = *gpu_hist.get(&k).unwrap_or(&0) as f64 / draws as f64;
            tv += (c - g).abs();
        }
        tv *= 0.5;
        assert!(tv < 0.01, "GPU/CPU empirical distributions differ: TV={tv:.4}");
        eprintln!(
            "histogram: {draws} draws, {} distinct tokens, TV={tv:.5}, disagree={disagree}",
            gpu_hist.len()
        );
    }
}

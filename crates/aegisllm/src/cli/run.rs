use std::env;
use std::path::Path;

use super::gates::run_gates;
use super::generate::{print_generate_bench, print_generate_bench_sweep};
use super::smoke::{
    cpu_materialize_smoke, cpu_smoke, cuda_attn_compare, cuda_attn_fp8_smoke, cuda_attn_ref_check,
    cuda_chain_smoke, cuda_compare, cuda_cutlass_nvfp4_smoke, cuda_dense_smoke,
    cuda_prefill_compare, cuda_prefill_sweep, cuda_smoke, inspect_hardware, mvp_check,
    quality_smoke, storage_smoke, vision_load_smoke,
};
use super::{Command, parse_args};
use crate::engine::bench::run_generation_bench;
use crate::engine::eval_mmlu_pro::{print_eval_summary, run_eval_mmlu_pro};
use crate::engine::perplexity::compute_perplexity;
use crate::engine::sample_diversity::run_sample_diversity;
use crate::engine::{AegisEngine, EngineConfig};
use aegisllm_base::error::{AegisError, Result};
use crate::executor::readiness_for_plan;

/// Greedy-generation snapshot/diff for quality regression detection.
///
/// First run with a given reference path: writes the current generation to it
/// (snapshot mode). Subsequent runs: compares the current text to the saved
/// reference and prints a `loss` metric — `mismatched_chars / max_chars`,
/// where 0.0 is byte-identical and larger numbers indicate divergence.
///
/// The metric is intentionally simple — character-level rather than tokens or
/// log-prob — so it works without exposing logits or the tokenizer to the
/// CLI. For more sensitive regression detection use a longer prompt + larger
/// `--max-tokens`; the metric scales with the test surface area.
fn run_quality_diff(current: &str, reference_path: &Path) -> Result<()> {
    if !reference_path.exists() {
        std::fs::write(reference_path, current.as_bytes()).map_err(|e| {
            AegisError::InvalidConfig(format!(
                "quality-diff: failed to write snapshot at {}: {e}",
                reference_path.display(),
            ))
        })?;
        println!(
            "quality-diff SNAPSHOT saved={} bytes path={}",
            current.len(),
            reference_path.display(),
        );
        return Ok(());
    }
    let reference_bytes = std::fs::read(reference_path).map_err(|e| {
        AegisError::InvalidConfig(format!(
            "quality-diff: failed to read reference at {}: {e}",
            reference_path.display(),
        ))
    })?;
    let reference = String::from_utf8_lossy(&reference_bytes);
    let max_len = reference.chars().count().max(current.chars().count());
    if max_len == 0 {
        println!("quality-diff PASS loss=0.0000 (both empty)");
        return Ok(());
    }
    let mut mismatched = 0usize;
    let mut first_diff: Option<usize> = None;
    let mut ref_chars = reference.chars();
    let mut cur_chars = current.chars();
    let mut idx = 0usize;
    loop {
        match (ref_chars.next(), cur_chars.next()) {
            (None, None) => break,
            (Some(a), Some(b)) if a == b => {}
            (Some(_), Some(_)) | (Some(_), None) | (None, Some(_)) => {
                mismatched += 1;
                if first_diff.is_none() {
                    first_diff = Some(idx);
                }
            }
        }
        idx += 1;
    }
    let loss = mismatched as f32 / max_len as f32;
    let status = if mismatched == 0 { "PASS" } else { "FAIL" };
    let preview_len = 80usize;
    let ref_preview: String = reference.chars().take(preview_len).collect();
    let cur_preview: String = current.chars().take(preview_len).collect();
    println!(
        "quality-diff {status} loss={loss:.4} mismatched={mismatched}/{max_len} \
         first_diff={first_diff:?}",
    );
    if mismatched > 0 {
        println!("  reference: {ref_preview}");
        println!("  current:   {cur_preview}");
    }
    Ok(())
}

pub fn run_env() -> Result<()> {
    match parse_args(env::args().skip(1))? {
        Command::InspectHardware => inspect_hardware(),
        Command::ShowPlan(config) => {
            let engine = AegisEngine::build(config)?;
            print!("{}", engine.report());
        }
        Command::MvpCheck(config) => mvp_check(config)?,
        Command::QualitySmoke(config) => quality_smoke(config)?,
        Command::StorageSmoke(config) => storage_smoke(config)?,
        Command::CpuSmoke(config) => cpu_smoke(config)?,
        Command::CpuMaterializeSmoke(config) => cpu_materialize_smoke(config)?,
        Command::CudaSmoke(config) => cuda_smoke(config)?,
        Command::VisionLoadSmoke(config) => vision_load_smoke(config)?,
        Command::CudaCutlassNvfp4Smoke => cuda_cutlass_nvfp4_smoke()?,
        Command::CudaAttnFp8Smoke => cuda_attn_fp8_smoke()?,
        Command::CudaAttnRefCheck => cuda_attn_ref_check()?,
        Command::CudaDenseSmoke(config) => cuda_dense_smoke(config)?,
        Command::CudaChainSmoke(config) => cuda_chain_smoke(config)?,
        Command::CudaCompare(config) => cuda_compare(config)?,
        Command::CudaPrefillCompare(config) => cuda_prefill_compare(config)?,
        Command::CudaPrefillSweep(config) => cuda_prefill_sweep(config)?,
        Command::CudaAttnCompare(config, prompt, reference) => {
            cuda_attn_compare(config, prompt, reference)?
        }
        Command::Gates(config, gates) => run_gates(config, gates)?,
        Command::Generate(config, mut request, image, audio_mel) => {
            let engine = AegisEngine::build(config)?;
            // Stage I.2 multimodal: if --image was passed, load + preprocess
            // it + run the vision tower to produce image-soft-token embeddings
            // in text-embedding space, then attach to the request so the
            // engine's prefill step splices them at `<|image|>` positions.
            if let Some(path) = image {
                let injection = compute_image_injection(&engine, &path)?;
                eprintln!(
                    "vision: {} tokens × {} hidden",
                    injection.n_tokens, injection.hidden
                );
                // The chat template emits a single image marker; HF's
                // image processor expands that into the multimodal
                // image-block: BOI + N × image_soft_token + EOI. We mirror
                // that expansion here, reading every token string from the
                // tokenizer (via the model's config-declared token IDs) so
                // the same code works for any vision-capable checkpoint —
                // no hardcoded `<|image>` / `<image|>` literals.
                let cfg = &engine.artifact.config;
                let text = aegisllm_base::text::TextProcessor::from_artifact(&engine.artifact)?;
                let img_tok_u32 = injection.image_token_id as u32;
                let image_marker = text.token_string(img_tok_u32)
                    .ok_or_else(|| AegisError::InvalidPlan(format!(
                        "vision: tokenizer has no string for image_token_id={}",
                        img_tok_u32
                    )))?;
                let boi = cfg.boi_token_id.and_then(|id| text.token_string(id));
                let eoi = cfg.eoi_token_id.and_then(|id| text.token_string(id));
                if request.prompt.contains(&image_marker) {
                    let boi_s = boi.as_deref().unwrap_or("");
                    let eoi_s = eoi.as_deref().unwrap_or("");
                    let mut block = String::with_capacity(
                        boi_s.len() + image_marker.len() * injection.n_tokens + eoi_s.len()
                    );
                    block.push_str(boi_s);
                    for _ in 0..injection.n_tokens { block.push_str(&image_marker); }
                    block.push_str(eoi_s);
                    request.prompt = request.prompt.replacen(&image_marker, &block, 1);
                    eprintln!(
                        "vision: expanded `{}` to {}{}×{}{} block",
                        image_marker, boi_s, injection.n_tokens, image_marker, eoi_s,
                    );
                }
                request.image_injection = Some(injection);
            }
            // Audio multimodal: if --audio-mel was passed, load the precomputed
            // log-mel features + run the audio tower to produce audio-soft-token
            // embeddings in text-embedding space, then attach to the request so
            // the engine's prefill splices them at `<audio_soft_token>` positions.
            // Mirrors the image path (BOA + N audio tokens + EOA expansion).
            if let Some(mel_path) = audio_mel {
                let injection = compute_audio_injection(&engine, &mel_path)?;
                eprintln!(
                    "audio: {} tokens × {} hidden",
                    injection.n_tokens, injection.hidden
                );
                let cfg = &engine.artifact.config;
                let text = aegisllm_base::text::TextProcessor::from_artifact(&engine.artifact)?;
                let aud_tok_u32 = injection.audio_token_id as u32;
                let audio_marker = text.token_string(aud_tok_u32)
                    .ok_or_else(|| AegisError::InvalidPlan(format!(
                        "audio: tokenizer has no string for audio_token_id={}",
                        aud_tok_u32
                    )))?;
                let boa = cfg.boa_token_id.and_then(|id| text.token_string(id));
                let eoa = cfg.eoa_token_id.and_then(|id| text.token_string(id));
                if request.prompt.contains(&audio_marker) {
                    let boa_s = boa.as_deref().unwrap_or("");
                    let eoa_s = eoa.as_deref().unwrap_or("");
                    let mut block = String::with_capacity(
                        boa_s.len() + audio_marker.len() * injection.n_tokens + eoa_s.len()
                    );
                    block.push_str(boa_s);
                    for _ in 0..injection.n_tokens { block.push_str(&audio_marker); }
                    block.push_str(eoa_s);
                    request.prompt = request.prompt.replacen(&audio_marker, &block, 1);
                    eprintln!(
                        "audio: expanded `{}` to {}{}×{}{} block",
                        audio_marker, boa_s, injection.n_tokens, audio_marker, eoa_s,
                    );
                }
                request.audio_injection = Some(injection);
            }
            let output = engine.generate(request)?;
            println!("{}", output.text);
            eprintln!(
                "finish={} prompt_tokens={} completion_tokens={}",
                output.finish_reason, output.prompt_tokens, output.completion_tokens
            );
        }
        Command::QualityDiff(config, request, reference_path) => {
            let engine = AegisEngine::build(config)?;
            let output = engine.generate(request)?;
            run_quality_diff(&output.text, &reference_path)?;
        }
        Command::BenchGenerate(config, request, prompt_repeat, format) => {
            let metrics = run_generation_bench(config, request)?;
            print_generate_bench(&metrics, prompt_repeat, format);
        }
        Command::BenchGenerateSweep(
            config,
            request,
            prompt_repeats,
            chunk_sizes,
            warmup_runs,
            measured_runs,
            format,
        ) => {
            let mut results = Vec::new();
            for chunk_size in chunk_sizes {
                for prompt_repeat in &prompt_repeats {
                    let mut config = config.clone();
                    config.cuda.prefill_chunk_size = Some(chunk_size);
                    let mut generate = request.clone();
                    generate.prompt = std::iter::repeat_n(generate.prompt.as_str(), *prompt_repeat)
                        .collect::<Vec<_>>()
                        .join("\n");
                    let metrics = run_generation_bench(
                        config,
                        crate::engine::bench::BenchGenerateRequest {
                            generate,
                            warmup_runs,
                            measured_runs,
                        },
                    )?;
                    results.push((*prompt_repeat, metrics));
                }
            }
            print_generate_bench_sweep(&results, format);
        }
        Command::Perplexity(config, request) => {
            let result = compute_perplexity(config, request)?;
            println!(
                "perplexity: tokens_scored={} mean_neg_logp={:.6} ppl={:.4}",
                result.num_tokens_scored, result.mean_neg_log_prob, result.perplexity,
            );
        }
        Command::SampleDiversity(config, request) => {
            let prompt_preview = request.prompt.clone();
            let result = run_sample_diversity(config, request)?;
            println!(
                "sample-diversity: runs={} sampling=temp={:.2}/top_k={}/top_p={:.2}/min_p={:.3}",
                result.runs,
                result.sampling.temperature,
                result.sampling.top_k,
                result.sampling.top_p,
                result.sampling.min_p,
            );
            println!("prompt: {prompt_preview:?}");
            println!("first-token distribution:");
            for (tok, count) in &result.first_token_distribution {
                let pct = (*count as f64 / result.runs as f64) * 100.0;
                println!("  {count:>3}/{} ({pct:>5.1}%) — {tok:?}", result.runs);
            }
            println!("completion distribution (top 5):");
            for (text, count) in result.distribution.iter().take(5) {
                let preview: String = text.chars().take(80).collect();
                println!("  {count:>3}× {preview:?}");
            }
        }
        Command::EvalMmluPro(config, request) => {
            let result = run_eval_mmlu_pro(config, request)?;
            print_eval_summary(&result);
        }
        Command::Serve(config) => {
            let mut default_sampling = config.engine.generation;
            let engine_config = EngineConfig {
                model_path: config.engine.model_path,
                policy: config.engine.policy,
                enable_executor: false,
                cuda: config.engine.cuda,
                // Spec-decode for `serve` comes from the config `draft` section
                // (plumbed through ServeConfig/EngineConfigFragment). Absent → plain.
                draft_model: config.engine.draft_model,
                num_draft_tokens: config.engine.num_draft_tokens,
            };
            // Build the preview engine WITHOUT the executor first so we can
            // compute readiness from the placement + runtime plan. If the
            // plan is runnable, promote the preview in-place by attaching
            // the executor — this reuses the already-parsed artifact and
            // plan instead of re-running `ModelArtifact::from_local_path`
            // (which scans every safetensors shard via `parse_lfs_pointer`
            // and used to be a hidden ~38s + ~17 GiB-of-disk-reads pass).
            let preview = AegisEngine::build(engine_config)?;
            let readiness = readiness_for_plan(&preview.placement, &preview.runtime);
            let engine = if readiness.runnable {
                preview.with_executor()?
            } else {
                preview
            };
            // If the params config left sampling at the greedy default,
            // adopt the model's own recommended sampling from
            // generation_config.json. Greedy decode degenerates into
            // repetition loops on reasoning models (Gemma-4).
            if default_sampling == aegisllm_base::generation::SamplingConfig::default() {
                if let Some(recommended) = engine
                    .artifact
                    .generation_config
                    .as_ref()
                    .and_then(|g| g.recommended_sampling())
                {
                    eprintln!(
                        "sampling: no sampling set in params config — using model's \
                         generation_config.json (temp={:.2} top_k={} top_p={:.2})",
                        recommended.temperature, recommended.top_k, recommended.top_p,
                    );
                    default_sampling = recommended;
                }
            }
            eprintln!("{}", engine.report());
            crate::server::serve_http(
                config.host,
                config.port,
                config.api,
                config.api_keys,
                engine,
                readiness,
                default_sampling,
            )?;
        }
    }
    Ok(())
}

/// Load an image, preprocess, run the vision tower, return an
/// `ImageInjection` ready to splice into the prompt's embedding stream.
/// Every model-specific value (vision arch, soft-token budget, image_token_id,
/// boi/eoi markers) is read from the artifact's parsed config.json — no
/// per-model hardcodes in this function.
fn compute_image_injection(
    engine: &AegisEngine,
    image_path: &Path,
) -> Result<aegisllm_base::generation::ImageInjection> {
    use aegisllm_base::modalities::image_preprocess::ImageProcessor;
    use aegisllm_base::tensor::storage::TensorStorageLoader;
    use aegisllm_cuda::executor::vision::{VisionEncoderShape, VisionTower};

    let device = engine
        .inventory
        .gpus
        .first()
        .map(|gpu| gpu.index)
        .ok_or_else(|| AegisError::Unsupported("no CUDA device for --image".into()))?;
    let cuda_config = engine.cuda;
    let cuda = aegisllm_cuda::cuda::CudaRuntime::new_with_config(device, cuda_config)?;
    let cuda_weights = cuda.weight_loader();
    let mut loader = TensorStorageLoader::new();

    let shape = VisionEncoderShape::from_artifact(&engine.artifact)?;
    eprintln!("vision: loading tower ({}L hidden={} head_dim={} standardize={})...",
        shape.num_layers, shape.hidden_size, shape.head_dim, shape.standardize);
    let tower = VisionTower::from_artifact(
        &engine.artifact, shape, &cuda_weights, device, &mut loader,
    )?;

    eprintln!("vision: preprocessing {image_path:?}...");
    let vision_cfg = engine.artifact.config.vision_config.as_ref()
        .ok_or_else(|| AegisError::InvalidPlan("vision: config.json missing vision_config".into()))?;
    let max_soft_tokens = engine.artifact.config.vision_soft_tokens_per_image
        .ok_or_else(|| AegisError::InvalidPlan(
            "vision: config.json missing `vision_soft_tokens_per_image` (top-level key)".into()
        ))?;
    let processor = ImageProcessor::from_artifact_vision(vision_cfg, max_soft_tokens);
    let img = processor.load(image_path)?;
    eprintln!(
        "vision: {}x{} → {} patches → {} soft tokens",
        img.height, img.width, img.num_patches(), img.num_tokens()
    );

    let text_hidden = tower.projector.rows;
    let n_tokens = img.num_tokens();

    // Diagnostic: when AEGIS_INJECT_FROM_FILE is set, load that .bin (f32,
    // shape [n_tokens, text_hidden]) instead of running the vision tower.
    // Lets us validate the injection mechanism with a known-good reference
    // (e.g. HF Gemma4 projector dump).
    let embeds = if let Ok(path) = std::env::var("AEGIS_INJECT_FROM_FILE") {
        eprintln!("vision: loading embeds from {path} (bypassing tower forward)");
        let bytes = std::fs::read(&path)
            .map_err(|e| AegisError::InvalidPlan(format!("read {path}: {e}")))?;
        let expected = n_tokens * text_hidden * 4;
        if bytes.len() != expected {
            return Err(AegisError::InvalidPlan(format!(
                "inject-from-file: got {} bytes, expected {} ({}×{}×4)",
                bytes.len(), expected, n_tokens, text_hidden
            )));
        }
        let mut v = vec![0.0_f32; n_tokens * text_hidden];
        for (i, chunk) in bytes.chunks_exact(4).enumerate() {
            v[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        v
    } else {
        eprintln!("vision: running tower forward (~55s)...");
        let t0 = std::time::Instant::now();
        let e = tower.forward_gpu(&cuda, &img.patches, img.num_patches_h, img.num_patches_w)?;
        eprintln!("vision: forward done in {:.2}s", t0.elapsed().as_secs_f64());
        e
    };

    let image_token_id = engine.artifact.config.image_token_id.ok_or_else(|| {
        AegisError::InvalidPlan(
            "vision: config.json missing `image_token_id` (top-level)".into(),
        )
    })?;
    Ok(aegisllm_base::generation::ImageInjection {
        data: embeds,
        n_tokens,
        hidden: text_hidden,
        image_token_id: image_token_id as usize,
    })
}

/// Load precomputed log-mel features from a raw f32 `.bin` (`[n_frames, 128]`),
/// run the audio tower, and return an `AudioInjection` ready to splice into the
/// prompt's embedding stream. Mirrors `compute_image_injection`.
///
/// v1 INPUT: the `.bin` holds raw little-endian f32, row-major
/// `[n_frames, n_mel_bins]`. We infer `n_frames = byte_len / (4 * n_mel_bins)`.
///
/// TODO(gpu-verify): the real mel front-end (frame=320 hop=160 fft=512 @16kHz →
/// 100 frames/s, 128 log-mel bins, per-feature normalization) is NOT
/// implemented this pass — the caller must precompute the mel features. Add an
/// FFT/ffmpeg front-end in a later pass.
fn compute_audio_injection(
    engine: &AegisEngine,
    mel_path: &Path,
) -> Result<aegisllm_base::generation::AudioInjection> {
    use aegisllm_base::tensor::storage::TensorStorageLoader;
    use aegisllm_cuda::executor::audio::{AudioEncoderShape, AudioTower};

    let device = engine
        .inventory
        .gpus
        .first()
        .map(|gpu| gpu.index)
        .ok_or_else(|| AegisError::Unsupported("no CUDA device for --audio-mel".into()))?;
    let cuda_config = engine.cuda;
    let cuda = aegisllm_cuda::cuda::CudaRuntime::new_with_config(device, cuda_config)?;
    let cuda_weights = cuda.weight_loader();
    let mut loader = TensorStorageLoader::new();

    let shape = AudioEncoderShape::from_artifact(&engine.artifact)?;
    eprintln!(
        "audio: loading tower ({}L hidden={} heads={} head_dim={})...",
        shape.num_layers, shape.hidden_size, shape.num_attention_heads, shape.head_dim
    );
    let n_mel = shape.n_mel_bins;
    let tower = AudioTower::from_artifact(
        &engine.artifact, shape, &cuda_weights, device, &mut loader,
    )?;

    // Load the raw f32 [n_frames, n_mel] log-mel features.
    let bytes = std::fs::read(mel_path)
        .map_err(|e| AegisError::InvalidPlan(format!("read {mel_path:?}: {e}")))?;
    if bytes.len() % (4 * n_mel) != 0 {
        return Err(AegisError::InvalidPlan(format!(
            "audio-mel: {} bytes not a multiple of 4*n_mel_bins ({})",
            bytes.len(), 4 * n_mel
        )));
    }
    let n_frames = bytes.len() / (4 * n_mel);
    let mut mel = vec![0.0_f32; n_frames * n_mel];
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        mel[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    eprintln!("audio: {n_frames} frames × {n_mel} mel → running tower forward...");

    let t0 = std::time::Instant::now();
    let embeds = tower.forward(&cuda, &mel, n_frames)?;
    eprintln!("audio: forward done in {:.2}s", t0.elapsed().as_secs_f64());

    let text_hidden = tower.embed_audio.rows;
    let n_tokens = if text_hidden > 0 { embeds.len() / text_hidden } else { 0 };

    let audio_token_id = engine.artifact.config.audio_token_id.ok_or_else(|| {
        AegisError::InvalidPlan("audio: config.json missing `audio_token_id` (top-level)".into())
    })?;
    Ok(aegisllm_base::generation::AudioInjection {
        data: embeds,
        n_tokens,
        hidden: text_hidden,
        audio_token_id: audio_token_id as usize,
    })
}

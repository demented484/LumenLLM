use std::env;
use std::path::Path;

use super::gates::run_gates;
use super::generate::{print_generate_bench, print_generate_bench_sweep};
use super::smoke::{
    cpu_materialize_smoke, cpu_smoke, cuda_chain_smoke, cuda_compare, cuda_cutlass_nvfp4_smoke,
    cuda_dense_smoke, cuda_prefill_compare, cuda_prefill_sweep, cuda_smoke, inspect_hardware,
    mvp_check, quality_smoke, storage_smoke,
};
use super::{Command, parse_args};
use crate::engine::bench::run_generation_bench;
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
        Command::CudaCutlassNvfp4Smoke => cuda_cutlass_nvfp4_smoke()?,
        Command::CudaDenseSmoke(config) => cuda_dense_smoke(config)?,
        Command::CudaChainSmoke(config) => cuda_chain_smoke(config)?,
        Command::CudaCompare(config) => cuda_compare(config)?,
        Command::CudaPrefillCompare(config) => cuda_prefill_compare(config)?,
        Command::CudaPrefillSweep(config) => cuda_prefill_sweep(config)?,
        Command::Gates(config, gates) => run_gates(config, gates)?,
        Command::Generate(config, request) => {
            let engine = AegisEngine::build(config)?;
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
        Command::Serve(config) => {
            let default_sampling = config.engine.generation;
            let engine_config = EngineConfig {
                model_path: config.engine.model_path,
                policy: config.engine.policy,
                enable_executor: false,
                cuda: config.engine.cuda,
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
            eprintln!("{}", engine.report());
            crate::server::serve_http(
                config.host,
                config.port,
                config.api,
                engine,
                readiness,
                default_sampling,
            )?;
        }
    }
    Ok(())
}

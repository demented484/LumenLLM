use std::time::Duration;

use serde_json::json;

use super::command::BenchOutputFormat;
use crate::engine::bench::GenerateBenchMetrics;
use crate::generation::PrefillStageTimings;

pub(super) fn print_generate_bench(
    metrics: &GenerateBenchMetrics,
    prompt_repeat: usize,
    format: BenchOutputFormat,
) {
    match format {
        BenchOutputFormat::Text => print_generate_bench_text(metrics, prompt_repeat),
        BenchOutputFormat::Json => print_generate_bench_json(metrics, prompt_repeat),
        BenchOutputFormat::Csv => print_generate_bench_csv(metrics, prompt_repeat),
    }
}

pub(super) fn print_generate_bench_sweep(
    results: &[(usize, GenerateBenchMetrics)],
    format: BenchOutputFormat,
) {
    match format {
        BenchOutputFormat::Text => {
            println!("bench-generate-sweep:");
            for (prompt_repeat, metrics) in results {
                print_generate_bench_text(metrics, *prompt_repeat);
            }
        }
        BenchOutputFormat::Json => {
            let results = results
                .iter()
                .map(|(prompt_repeat, metrics)| {
                    json!({
                        "prompt_repeat": prompt_repeat,
                        "prefill_chunk_size": metrics.prefill_chunk_size,
                        "prompt_tokens": metrics.prompt_tokens,
                        "completion_tokens": metrics.completion_tokens,
                        "average_prefill_elapsed_ms": millis(metrics.average_prefill_elapsed),
                        "average_decode_elapsed_ms": millis(metrics.average_decode_elapsed),
                        "prefill_tok_per_s": tokens_per_second(metrics.prompt_tokens, metrics.average_prefill_elapsed),
                        "decode_tok_per_s": tokens_per_second(metrics.completion_tokens, metrics.average_decode_elapsed),
                        "attention_requested": metrics.attention_requested,
                        "attention_auto_target": metrics.attention_auto_target,
                        "attention_logical_backend": metrics.attention_logical_backend,
                        "attention_effective_path": metrics.attention_effective_path,
                        "attention_reason": metrics.attention_reason,
                        "selection_context_tokens": metrics.selection_context_tokens,
                        "average_prefill_stage_timings": metrics.average_prefill_stage_timings.map(stage_timings_json),
                    })
                })
                .collect::<Vec<_>>();
            println!(
                "{}",
                json!({ "command": "bench-generate-sweep", "results": results })
            );
        }
        BenchOutputFormat::Csv => {
            println!(
                "prompt_repeat,prefill_chunk_size,prompt_tokens,completion_tokens,prefill_ms,decode_ms,prefill_tok_per_s,decode_tok_per_s,attention_requested,attention_logical_backend,attention_effective_path,stage_qkv_us,stage_attention_us,stage_mlp_us,stage_qkv_tflops,stage_mlp_tflops"
            );
            for (prompt_repeat, metrics) in results {
                let stages = metrics.average_prefill_stage_timings;
                println!(
                    "{},{},{},{},{:.3},{:.3},{:.3},{:.3},{},{},{},{},{},{},{:.3},{:.3}",
                    prompt_repeat,
                    metrics
                        .prefill_chunk_size
                        .map(|chunk| chunk.to_string())
                        .unwrap_or_else(|| "auto".into()),
                    metrics.prompt_tokens,
                    metrics.completion_tokens,
                    millis(metrics.average_prefill_elapsed),
                    millis(metrics.average_decode_elapsed),
                    tokens_per_second(metrics.prompt_tokens, metrics.average_prefill_elapsed),
                    tokens_per_second(metrics.completion_tokens, metrics.average_decode_elapsed),
                    csv_escape(metrics.attention_requested.as_deref().unwrap_or("unknown")),
                    csv_escape(
                        metrics
                            .attention_logical_backend
                            .as_deref()
                            .unwrap_or("unknown")
                    ),
                    csv_escape(
                        metrics
                            .attention_effective_path
                            .as_deref()
                            .unwrap_or("unknown")
                    ),
                    stages
                        .map(|stage| stage.qkv_us.to_string())
                        .unwrap_or_default(),
                    stages
                        .map(|stage| stage.attention_us.to_string())
                        .unwrap_or_default(),
                    stages
                        .map(|stage| stage.mlp_us.to_string())
                        .unwrap_or_default(),
                    stages.map(|stage| stage.qkv_tflops).unwrap_or_default(),
                    stages.map(|stage| stage.mlp_tflops).unwrap_or_default(),
                );
            }
        }
    }
}

fn print_generate_bench_text(metrics: &GenerateBenchMetrics, prompt_repeat: usize) {
    let total_tokens = metrics.prompt_tokens + metrics.completion_tokens;
    let load_run_elapsed = metrics.load_elapsed + metrics.average_run_elapsed;
    println!("bench-generate:");
    println!(
        "  backend={}",
        metrics.backend.as_deref().unwrap_or("unknown")
    );
    println!(
        "  attention_requested={}",
        metrics.attention_requested.as_deref().unwrap_or("unknown")
    );
    println!(
        "  attention_auto_target={}",
        metrics.attention_auto_target.as_deref().unwrap_or("none")
    );
    println!(
        "  attention_logical_backend={}",
        metrics
            .attention_logical_backend
            .as_deref()
            .unwrap_or("unknown")
    );
    println!(
        "  attention_effective_path={}",
        metrics
            .attention_effective_path
            .as_deref()
            .unwrap_or("unknown")
    );
    println!(
        "  attention_reason={}",
        metrics.attention_reason.as_deref().unwrap_or("unknown")
    );
    println!(
        "  prefill_chunk_size={}",
        metrics
            .prefill_chunk_size
            .map(|chunk| chunk.to_string())
            .unwrap_or_else(|| "auto".into())
    );
    println!(
        "  selection_context_tokens={}",
        metrics.selection_context_tokens
    );
    if let Some(stages) = metrics.average_prefill_stage_timings {
        println!(
            "  avg_prefill_stages: chunks={} prepare_us={} embed_us={} qkv_us={} qkv_tflops={:.3} rope_us={} kv_store_us={} attention_us={} o_proj_us={} mlp_us={} mlp_tflops={:.3} layer_total_us={} sample_us={}",
            stages.chunks,
            stages.prepare_us,
            stages.embed_us,
            stages.qkv_us,
            stages.qkv_tflops,
            stages.rope_us,
            stages.kv_store_us,
            stages.attention_us,
            stages.o_proj_us,
            stages.mlp_us,
            stages.mlp_tflops,
            stages.layer_total_us,
            stages.sample_us
        );
    }
    println!("  prompt_repeat={prompt_repeat}");
    println!("  warmup_runs={}", metrics.warmup_runs);
    println!("  measured_runs={}", metrics.measured_runs);
    println!("  finish={}", metrics.finish_reason);
    println!("  load_elapsed_ms={:.3}", millis(metrics.load_elapsed));
    println!(
        "  avg_run_elapsed_ms={:.3}",
        millis(metrics.average_run_elapsed)
    );
    println!(
        "  avg_tokenize_elapsed_ms={:.3}",
        millis(metrics.average_tokenize_elapsed)
    );
    println!(
        "  avg_prefill_elapsed_ms={:.3}",
        millis(metrics.average_prefill_elapsed)
    );
    println!(
        "  avg_decode_elapsed_ms={:.3}",
        millis(metrics.average_decode_elapsed)
    );
    println!("  load_run_elapsed_ms={:.3}", millis(load_run_elapsed));
    println!("  prompt_tokens={}", metrics.prompt_tokens);
    println!("  completion_tokens={}", metrics.completion_tokens);
    println!("  total_tokens={total_tokens}");
    println!(
        "  total_tok_per_s={:.3}",
        tokens_per_second(total_tokens, metrics.average_run_elapsed)
    );
    println!(
        "  load_run_tok_per_s={:.3}",
        tokens_per_second(total_tokens, load_run_elapsed)
    );
    println!(
        "  prefill_tok_per_s={:.3}",
        tokens_per_second(metrics.prompt_tokens, metrics.average_prefill_elapsed)
    );
    println!(
        "  decode_tok_per_s={:.3}",
        tokens_per_second(metrics.completion_tokens, metrics.average_decode_elapsed)
    );
    for run in &metrics.runs {
        println!(
            "  run[{}]: total_ms={:.3} tokenize_ms={:.3} prefill_ms={:.3} decode_ms={:.3}",
            run.run_index,
            millis(run.total_elapsed),
            millis(run.tokenize_elapsed),
            millis(run.prefill_elapsed),
            millis(run.decode_elapsed),
        );
    }
}

fn print_generate_bench_json(metrics: &GenerateBenchMetrics, prompt_repeat: usize) {
    let runs = metrics
        .runs
        .iter()
        .map(|run| {
            json!({
                "run_index": run.run_index,
                "total_ms": millis(run.total_elapsed),
                "tokenize_ms": millis(run.tokenize_elapsed),
                "prefill_ms": millis(run.prefill_elapsed),
                "decode_ms": millis(run.decode_elapsed),
                "prompt_tokens": run.prompt_tokens,
                "completion_tokens": run.completion_tokens,
                "finish_reason": run.finish_reason,
                "prefill_stage_timings": run.prefill_stage_timings.map(stage_timings_json),
            })
        })
        .collect::<Vec<_>>();
    println!(
        "{}",
        json!({
            "command": "bench-generate",
            "backend": metrics.backend,
            "attention_requested": metrics.attention_requested,
            "attention_auto_target": metrics.attention_auto_target,
            "attention_logical_backend": metrics.attention_logical_backend,
            "attention_effective_path": metrics.attention_effective_path,
            "attention_reason": metrics.attention_reason,
            "prefill_chunk_size": metrics.prefill_chunk_size,
            "selection_context_tokens": metrics.selection_context_tokens,
            "average_prefill_stage_timings": metrics.average_prefill_stage_timings.map(stage_timings_json),
            "prompt_repeat": prompt_repeat,
            "warmup_runs": metrics.warmup_runs,
            "measured_runs": metrics.measured_runs,
            "finish_reason": metrics.finish_reason,
            "load_elapsed_ms": millis(metrics.load_elapsed),
            "average_run_elapsed_ms": millis(metrics.average_run_elapsed),
            "average_tokenize_elapsed_ms": millis(metrics.average_tokenize_elapsed),
            "average_prefill_elapsed_ms": millis(metrics.average_prefill_elapsed),
            "average_decode_elapsed_ms": millis(metrics.average_decode_elapsed),
            "prompt_tokens": metrics.prompt_tokens,
            "completion_tokens": metrics.completion_tokens,
            "prefill_tok_per_s": tokens_per_second(metrics.prompt_tokens, metrics.average_prefill_elapsed),
            "decode_tok_per_s": tokens_per_second(metrics.completion_tokens, metrics.average_decode_elapsed),
            "runs": runs,
        })
    );
}

fn print_generate_bench_csv(metrics: &GenerateBenchMetrics, prompt_repeat: usize) {
    println!(
        "run_index,backend,attention_requested,attention_auto_target,attention_logical_backend,attention_effective_path,attention_reason,prefill_chunk_size,selection_context_tokens,prompt_repeat,warmup_runs,measured_runs,total_ms,tokenize_ms,prefill_ms,decode_ms,prompt_tokens,completion_tokens,prefill_tok_per_s,decode_tok_per_s,stage_chunks,stage_prepare_us,stage_embed_us,stage_qkv_us,stage_qkv_tflops,stage_rope_us,stage_kv_store_us,stage_attention_us,stage_o_proj_us,stage_mlp_us,stage_mlp_tflops,stage_layer_total_us,stage_sample_us,finish_reason"
    );
    let backend = metrics.backend.as_deref().unwrap_or("unknown");
    let attention_requested = metrics.attention_requested.as_deref().unwrap_or("unknown");
    let attention_auto_target = metrics.attention_auto_target.as_deref().unwrap_or("none");
    let attention_logical_backend = metrics
        .attention_logical_backend
        .as_deref()
        .unwrap_or("unknown");
    let attention_effective_path = metrics
        .attention_effective_path
        .as_deref()
        .unwrap_or("unknown");
    let attention_reason = metrics.attention_reason.as_deref().unwrap_or("unknown");
    let prefill_chunk_size = metrics
        .prefill_chunk_size
        .map(|chunk| chunk.to_string())
        .unwrap_or_else(|| "auto".into());
    for run in &metrics.runs {
        println!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{},{},{:.3},{:.3},{},{},{},{},{:.3},{},{},{},{},{},{:.3},{},{},{}",
            run.run_index,
            csv_escape(backend),
            csv_escape(attention_requested),
            csv_escape(attention_auto_target),
            csv_escape(attention_logical_backend),
            csv_escape(attention_effective_path),
            csv_escape(attention_reason),
            csv_escape(&prefill_chunk_size),
            metrics.selection_context_tokens,
            prompt_repeat,
            metrics.warmup_runs,
            metrics.measured_runs,
            millis(run.total_elapsed),
            millis(run.tokenize_elapsed),
            millis(run.prefill_elapsed),
            millis(run.decode_elapsed),
            run.prompt_tokens,
            run.completion_tokens,
            tokens_per_second(run.prompt_tokens, run.prefill_elapsed),
            tokens_per_second(run.completion_tokens, run.decode_elapsed),
            run.prefill_stage_timings
                .map(|stage| stage.chunks.to_string())
                .unwrap_or_default(),
            run.prefill_stage_timings
                .map(|stage| stage.prepare_us.to_string())
                .unwrap_or_default(),
            run.prefill_stage_timings
                .map(|stage| stage.embed_us.to_string())
                .unwrap_or_default(),
            run.prefill_stage_timings
                .map(|stage| stage.qkv_us.to_string())
                .unwrap_or_default(),
            run.prefill_stage_timings
                .map(|stage| format!("{:.3}", stage.qkv_tflops))
                .unwrap_or_default(),
            run.prefill_stage_timings
                .map(|stage| stage.rope_us.to_string())
                .unwrap_or_default(),
            run.prefill_stage_timings
                .map(|stage| stage.kv_store_us.to_string())
                .unwrap_or_default(),
            run.prefill_stage_timings
                .map(|stage| stage.attention_us.to_string())
                .unwrap_or_default(),
            run.prefill_stage_timings
                .map(|stage| stage.o_proj_us.to_string())
                .unwrap_or_default(),
            run.prefill_stage_timings
                .map(|stage| stage.mlp_us.to_string())
                .unwrap_or_default(),
            run.prefill_stage_timings
                .map(|stage| format!("{:.3}", stage.mlp_tflops))
                .unwrap_or_default(),
            run.prefill_stage_timings
                .map(|stage| stage.layer_total_us.to_string())
                .unwrap_or_default(),
            run.prefill_stage_timings
                .map(|stage| stage.sample_us.to_string())
                .unwrap_or_default(),
            csv_escape(&run.finish_reason),
        );
    }
}

fn stage_timings_json(stage: PrefillStageTimings) -> serde_json::Value {
    json!({
        "chunks": stage.chunks,
        "prepare_us": stage.prepare_us,
        "embed_us": stage.embed_us,
        "qkv_us": stage.qkv_us,
        "qkv_tflops": stage.qkv_tflops,
        "rope_us": stage.rope_us,
        "kv_store_us": stage.kv_store_us,
        "attention_us": stage.attention_us,
        "o_proj_us": stage.o_proj_us,
        "mlp_us": stage.mlp_us,
        "mlp_tflops": stage.mlp_tflops,
        "layer_total_us": stage.layer_total_us,
        "sample_us": stage.sample_us,
    })
}

fn csv_escape(value: &str) -> String {
    if value.contains([',', '"', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn tokens_per_second(tokens: usize, elapsed: Duration) -> f64 {
    let seconds = elapsed.as_secs_f64();
    if seconds > 0.0 {
        tokens as f64 / seconds
    } else {
        0.0
    }
}

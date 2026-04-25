use std::time::Duration;

use serde_json::json;

use super::command::BenchOutputFormat;
use crate::engine::bench::GenerateBenchMetrics;

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

fn print_generate_bench_text(metrics: &GenerateBenchMetrics, prompt_repeat: usize) {
    let total_tokens = metrics.prompt_tokens + metrics.completion_tokens;
    let load_run_elapsed = metrics.load_elapsed + metrics.average_run_elapsed;
    println!("bench-generate:");
    println!(
        "  backend={}",
        metrics.backend.as_deref().unwrap_or("unknown")
    );
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
            })
        })
        .collect::<Vec<_>>();
    println!(
        "{}",
        json!({
            "command": "bench-generate",
            "backend": metrics.backend,
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
        "run_index,backend,prompt_repeat,warmup_runs,measured_runs,total_ms,tokenize_ms,prefill_ms,decode_ms,prompt_tokens,completion_tokens,prefill_tok_per_s,decode_tok_per_s,finish_reason"
    );
    let backend = metrics.backend.as_deref().unwrap_or("unknown");
    for run in &metrics.runs {
        println!(
            "{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{},{},{:.3},{:.3},{}",
            run.run_index,
            csv_escape(backend),
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
            csv_escape(&run.finish_reason),
        );
    }
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

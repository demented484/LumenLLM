use std::time::{Duration, Instant};

use crate::error::{AegisError, Result};
use crate::generation::GenerateRequest;

use super::{AegisEngine, EngineConfig};

#[derive(Debug, Clone, PartialEq)]
pub struct BenchGenerateRequest {
    pub generate: GenerateRequest,
    pub warmup_runs: usize,
    pub measured_runs: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenerateBenchMetrics {
    pub load_elapsed: Duration,
    pub runs: Vec<GenerateBenchRun>,
    pub average_run_elapsed: Duration,
    pub average_tokenize_elapsed: Duration,
    pub average_prefill_elapsed: Duration,
    pub average_decode_elapsed: Duration,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: String,
    pub backend: Option<String>,
    pub attention_requested: Option<String>,
    pub attention_auto_target: Option<String>,
    pub attention_logical_backend: Option<String>,
    pub attention_effective_path: Option<String>,
    pub attention_reason: Option<String>,
    pub prefill_chunk_size: Option<usize>,
    pub warmup_runs: usize,
    pub measured_runs: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenerateBenchRun {
    pub run_index: usize,
    pub total_elapsed: Duration,
    pub tokenize_elapsed: Duration,
    pub prefill_elapsed: Duration,
    pub decode_elapsed: Duration,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: String,
}

pub fn run_generation_bench(
    config: EngineConfig,
    request: BenchGenerateRequest,
) -> Result<GenerateBenchMetrics> {
    if request.measured_runs == 0 {
        return Err(AegisError::InvalidConfig(
            "bench-generate requires at least one measured run".into(),
        ));
    }
    let load_start = Instant::now();
    let engine = AegisEngine::build(config)?;
    let load_elapsed = load_start.elapsed();
    let backend = engine
        .executor_info()
        .map(|info| format!("{} {:?}", info.name, info.backends));
    let compute_capability = engine
        .inventory
        .gpus
        .first()
        .and_then(|gpu| gpu.compute_capability.as_deref());

    for _ in 0..request.warmup_runs {
        let _ = engine.generate_timed(request.generate.clone())?;
    }

    let mut runs = Vec::with_capacity(request.measured_runs);
    for run_index in 0..request.measured_runs {
        let timed = engine.generate_timed(request.generate.clone())?;
        runs.push(GenerateBenchRun {
            run_index,
            total_elapsed: timed.total_elapsed,
            tokenize_elapsed: timed.tokenize_elapsed,
            prefill_elapsed: timed.prefill_elapsed,
            decode_elapsed: timed.decode_elapsed,
            prompt_tokens: timed.output.prompt_tokens,
            completion_tokens: timed.output.completion_tokens,
            finish_reason: timed.output.finish_reason,
        });
    }

    let first = runs
        .first()
        .ok_or_else(|| AegisError::InvalidPlan("bench produced no measured runs".into()))?;
    let prompt_tokens = first.prompt_tokens;
    let completion_tokens = first.completion_tokens;
    let finish_reason = first.finish_reason.clone();
    for run in &runs {
        if run.prompt_tokens != prompt_tokens || run.completion_tokens != completion_tokens {
            return Err(AegisError::InvalidPlan(format!(
                "bench run token counts changed: first prompt={} completion={}, run{} prompt={} completion={}",
                prompt_tokens,
                completion_tokens,
                run.run_index,
                run.prompt_tokens,
                run.completion_tokens
            )));
        }
    }

    Ok(GenerateBenchMetrics {
        load_elapsed,
        average_run_elapsed: average_duration(runs.iter().map(|run| run.total_elapsed)),
        average_tokenize_elapsed: average_duration(runs.iter().map(|run| run.tokenize_elapsed)),
        average_prefill_elapsed: average_duration(runs.iter().map(|run| run.prefill_elapsed)),
        average_decode_elapsed: average_duration(runs.iter().map(|run| run.decode_elapsed)),
        prompt_tokens,
        completion_tokens,
        finish_reason,
        backend,
        attention_requested: Some(engine.cuda.prefill_attention.canonical_name().into()),
        attention_auto_target: {
            let selection = engine.cuda.prefill_attention_selection(
                compute_capability,
                prompt_tokens,
                engine.graph.head_dim,
            );
            selection
                .auto_target
                .map(|backend| backend.canonical_name().to_string())
        },
        attention_logical_backend: {
            let selection = engine.cuda.prefill_attention_selection(
                compute_capability,
                prompt_tokens,
                engine.graph.head_dim,
            );
            Some(selection.logical_backend.canonical_name().into())
        },
        attention_effective_path: {
            let selection = engine.cuda.prefill_attention_selection(
                compute_capability,
                prompt_tokens,
                engine.graph.head_dim,
            );
            Some(selection.effective_path.canonical_name().into())
        },
        attention_reason: {
            let selection = engine.cuda.prefill_attention_selection(
                compute_capability,
                prompt_tokens,
                engine.graph.head_dim,
            );
            Some(selection.reason.into())
        },
        prefill_chunk_size: engine.cuda.prefill_chunk_size,
        warmup_runs: request.warmup_runs,
        measured_runs: request.measured_runs,
        runs,
    })
}

fn average_duration(values: impl Iterator<Item = Duration>) -> Duration {
    let mut count = 0_u32;
    let mut total = Duration::ZERO;
    for value in values {
        total += value;
        count += 1;
    }
    if count == 0 {
        Duration::ZERO
    } else {
        total / count
    }
}

use std::time::{Duration, Instant};

use crate::error::{AegisError, Result};
use crate::generation::{GenerateRequest, PrefillStageTimings};

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
    pub selection_context_tokens: usize,
    pub attention_requested: Option<String>,
    pub attention_auto_target: Option<String>,
    pub attention_logical_backend: Option<String>,
    pub attention_effective_path: Option<String>,
    pub attention_reason: Option<String>,
    pub prefill_chunk_size: Option<usize>,
    pub warmup_runs: usize,
    pub measured_runs: usize,
    pub average_prefill_stage_timings: Option<PrefillStageTimings>,
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
    pub prefill_stage_timings: Option<PrefillStageTimings>,
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
            prefill_stage_timings: timed.prefill_stage_timings,
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
        selection_context_tokens: prompt_tokens,
        attention_requested: Some(engine.cuda.prefill_attention.canonical_name().into()),
        attention_auto_target: {
            let selection = engine.cuda.prefill_attention_selection(
                compute_capability,
                prompt_tokens,
                engine.graph.head_dim,
                engine.graph.num_attention_heads,
                engine.graph.num_kv_heads,
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
                engine.graph.num_attention_heads,
                engine.graph.num_kv_heads,
            );
            Some(selection.logical_backend.canonical_name().into())
        },
        attention_effective_path: {
            let selection = engine.cuda.prefill_attention_selection(
                compute_capability,
                prompt_tokens,
                engine.graph.head_dim,
                engine.graph.num_attention_heads,
                engine.graph.num_kv_heads,
            );
            Some(selection.effective_path.canonical_name().into())
        },
        attention_reason: {
            let selection = engine.cuda.prefill_attention_selection(
                compute_capability,
                prompt_tokens,
                engine.graph.head_dim,
                engine.graph.num_attention_heads,
                engine.graph.num_kv_heads,
            );
            Some(selection.reason.into())
        },
        prefill_chunk_size: engine.cuda.prefill_chunk_size,
        warmup_runs: request.warmup_runs,
        measured_runs: request.measured_runs,
        average_prefill_stage_timings: average_stage_timings(
            runs.iter().filter_map(|run| run.prefill_stage_timings),
        ),
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

fn average_stage_timings(
    values: impl Iterator<Item = PrefillStageTimings>,
) -> Option<PrefillStageTimings> {
    let mut count = 0_u128;
    let mut total = PrefillStageTimings {
        chunks: 0,
        prepare_us: 0,
        embed_us: 0,
        qkv_us: 0,
        qkv_tflops: 0.0,
        rope_us: 0,
        kv_store_us: 0,
        attention_us: 0,
        o_proj_us: 0,
        mlp_us: 0,
        mlp_tflops: 0.0,
        layer_total_us: 0,
        sample_us: 0,
    };
    for value in values {
        count += 1;
        total.chunks += value.chunks;
        total.prepare_us += value.prepare_us;
        total.embed_us += value.embed_us;
        total.qkv_us += value.qkv_us;
        total.qkv_tflops = total.qkv_tflops.max(value.qkv_tflops);
        total.rope_us += value.rope_us;
        total.kv_store_us += value.kv_store_us;
        total.attention_us += value.attention_us;
        total.o_proj_us += value.o_proj_us;
        total.mlp_us += value.mlp_us;
        total.mlp_tflops = total.mlp_tflops.max(value.mlp_tflops);
        total.layer_total_us += value.layer_total_us;
        total.sample_us += value.sample_us;
    }
    (count > 0).then(|| PrefillStageTimings {
        chunks: (total.chunks as u128 / count) as usize,
        prepare_us: total.prepare_us / count,
        embed_us: total.embed_us / count,
        qkv_us: total.qkv_us / count,
        qkv_tflops: total.qkv_tflops,
        rope_us: total.rope_us / count,
        kv_store_us: total.kv_store_us / count,
        attention_us: total.attention_us / count,
        o_proj_us: total.o_proj_us / count,
        mlp_us: total.mlp_us / count,
        mlp_tflops: total.mlp_tflops,
        layer_total_us: total.layer_total_us / count,
        sample_us: total.sample_us / count,
    })
}

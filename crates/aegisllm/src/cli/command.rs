use std::path::PathBuf;

use crate::cli::gates::GatesConfig;
use crate::engine::EngineConfig;
use crate::engine::bench::BenchGenerateRequest;
use crate::engine::eval_mmlu_pro::EvalMmluProRequest;
use crate::engine::perplexity::PerplexityRequest;
use crate::engine::sample_diversity::SampleDiversityRequest;
use aegisllm_base::generation::GenerateRequest;
use crate::params::ServeConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchOutputFormat {
    Text,
    Json,
    Csv,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    InspectHardware,
    ShowPlan(EngineConfig),
    MvpCheck(EngineConfig),
    QualitySmoke(EngineConfig),
    StorageSmoke(EngineConfig),
    CpuSmoke(EngineConfig),
    CpuMaterializeSmoke(EngineConfig),
    CudaSmoke(EngineConfig),
    CudaCutlassNvfp4Smoke,
    CudaAttnFp8Smoke,
    CudaDenseSmoke(EngineConfig),
    CudaChainSmoke(EngineConfig),
    CudaCompare(EngineConfig),
    CudaPrefillCompare(EngineConfig),
    CudaPrefillSweep(EngineConfig),
    Generate(EngineConfig, GenerateRequest),
    /// Greedy generation + character-level diff against a reference text.
    /// If the reference file doesn't exist yet, the current generation is saved
    /// to it (snapshot mode). On subsequent runs, the new generation is
    /// compared and a `loss` metric (mismatched_chars / max_chars) printed —
    /// 0.0 means byte-identical, larger means quality regression.
    QualityDiff(EngineConfig, GenerateRequest, PathBuf),
    BenchGenerate(EngineConfig, BenchGenerateRequest, usize, BenchOutputFormat),
    BenchGenerateSweep(
        EngineConfig,
        GenerateRequest,
        Vec<usize>,
        Vec<usize>,
        usize,
        usize,
        BenchOutputFormat,
    ),
    Gates(EngineConfig, GatesConfig),
    Serve(ServeConfig),
    /// Compute perplexity on a small built-in (or user-supplied) text via
    /// teacher forcing. Useful as a fitness function for quantization
    /// changes — coherent text is too noisy a signal.
    Perplexity(EngineConfig, PerplexityRequest),
    /// Run the same prompt N times under non-greedy sampling (one model
    /// load) and report the distribution of completions / first tokens.
    /// Diagnostic for "are token probabilities sane" — under reasonable
    /// settings (temp=1.0, top-k=50, min-p=0.05) a factual prompt should
    /// concentrate on the right answer but show varied phrasing.
    SampleDiversity(EngineConfig, SampleDiversityRequest),
    /// Run the MMLU-Pro benchmark (5-shot CoT by default) against the
    /// loaded model and report overall + per-subject accuracy. Used to
    /// validate the engine end-to-end against NVIDIA's published number
    /// and to measure the accuracy cost of FP8 attention/KV quantization.
    EvalMmluPro(EngineConfig, EvalMmluProRequest),
}

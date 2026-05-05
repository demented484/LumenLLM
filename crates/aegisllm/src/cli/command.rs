use std::path::PathBuf;

use crate::cli::gates::GatesConfig;
use crate::engine::EngineConfig;
use crate::engine::bench::BenchGenerateRequest;
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
}

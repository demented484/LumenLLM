use crate::engine::EngineConfig;
use crate::engine::bench::BenchGenerateRequest;
use crate::generation::GenerateRequest;
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
    CudaSdpaSweep(EngineConfig),
    Generate(EngineConfig, GenerateRequest),
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
    Serve(ServeConfig),
}

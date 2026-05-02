use std::path::PathBuf;

use aegisllm_cuda::cuda::CudaRuntimeConfig;
use aegisllm_base::generation::SamplingConfig;
use aegisllm_base::planning::placement::PlacementPolicy;

#[derive(Debug, Clone, PartialEq)]
pub struct ServeConfig {
    pub host: String,
    pub port: u16,
    pub api: String,
    pub engine: EngineConfigFragment,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EngineConfigFragment {
    pub model_path: PathBuf,
    pub policy: PlacementPolicy,
    pub cuda: CudaRuntimeConfig,
    pub generation: SamplingConfig,
}

use std::path::PathBuf;

use crate::cuda::CudaRuntimeConfig;
use crate::generation::SamplingConfig;
use crate::planning::placement::PlacementPolicy;

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

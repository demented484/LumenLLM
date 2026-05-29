use std::path::PathBuf;

use aegisllm_cuda::cuda::CudaRuntimeConfig;
use aegisllm_base::generation::SamplingConfig;
use aegisllm_base::planning::placement::PlacementPolicy;

#[derive(Debug, Clone, PartialEq)]
pub struct ServeConfig {
    pub host: String,
    pub port: u16,
    pub api: String,
    /// Accepted API keys (from `server-parameters.api-keys` + `AEGIS_API_KEY`).
    /// Empty → server is open (no auth). Non-empty → /v1/* requests must match.
    pub api_keys: Vec<String>,
    pub engine: EngineConfigFragment,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EngineConfigFragment {
    pub model_path: PathBuf,
    pub policy: PlacementPolicy,
    pub cuda: CudaRuntimeConfig,
    pub generation: SamplingConfig,
    /// EAGLE/MTP draft model from the config `draft` section (None = no
    /// spec-decode). An explicit `--draft-model` flag overrides this.
    pub draft_model: Option<PathBuf>,
    /// Tokens proposed per speculative round (config `draft.num-draft-tokens`,
    /// default 4). Only meaningful when `draft_model` is Some.
    pub num_draft_tokens: usize,
}

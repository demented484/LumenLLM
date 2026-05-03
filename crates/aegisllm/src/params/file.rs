use std::path::PathBuf;

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ParametersFile {
    #[serde(rename = "server-bin")]
    pub server_bin: Option<ServerBinSection>,
    #[serde(rename = "server-parameters")]
    pub server: Option<ServerSection>,
    pub model: ModelSection,
    pub layers: Option<LayersSection>,
    #[serde(rename = "kv-cache")]
    pub kv_cache: Option<KvCacheSection>,
    #[serde(rename = "linear-layout")]
    pub linear_layout: Option<LinearLayoutSection>,
    #[serde(rename = "other-parameters")]
    pub other: Option<OtherSection>,
    pub cuda: Option<CudaSection>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ServerBinSection {
    pub path: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ServerSection {
    pub host: Option<String>,
    pub port: Option<u16>,
    #[serde(rename = "server-api")]
    pub server_api: Option<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct ModelSection {
    pub path: PathBuf,
    pub store: Option<String>,
    pub compute: Option<String>,
    pub mmap: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct LayersSection {
    pub number: Option<usize>,
    pub store: Option<String>,
    pub compute: Option<String>,
    #[serde(rename = "rest-store")]
    pub rest_store: Option<String>,
    #[serde(rename = "rest-compute")]
    pub rest_compute: Option<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct KvCacheSection {
    #[serde(rename = "context-size")]
    pub context_size: Option<usize>,
    pub store: Option<String>,
    pub compute: Option<String>,
    #[serde(rename = "type-k")]
    pub type_k: Option<String>,
    #[serde(rename = "type-v")]
    pub type_v: Option<String>,
    #[serde(rename = "cache-prompt")]
    pub cache_prompt: Option<bool>,
    /// Number of layers (from layer 0) that use `store-first` instead of `store`.
    #[serde(rename = "store-first-n")]
    pub store_first_n: Option<usize>,
    /// Storage backend for the first `store-first-n` layers.
    #[serde(rename = "store-first")]
    pub store_first: Option<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct OtherSection {
    pub temperature: Option<f32>,
    #[serde(rename = "top-p")]
    pub top_p: Option<f32>,
    #[serde(rename = "top-k")]
    pub top_k: Option<usize>,
    #[serde(rename = "batch-size")]
    pub batch_size: Option<usize>,
    #[serde(rename = "ubatch-size")]
    pub ubatch_size: Option<usize>,
    #[serde(rename = "flash-attention")]
    pub flash_attention: Option<bool>,
    #[serde(rename = "cpu-linear-layout")]
    pub cpu_linear_layout: Option<String>,
    #[serde(rename = "cuda-linear-layout")]
    pub cuda_linear_layout: Option<String>,
    #[serde(rename = "linear-materialize")]
    pub linear_materialize: Option<String>,
    pub threads: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct LinearLayoutSection {
    pub mode: Option<String>,
    pub cpu: Option<String>,
    pub cuda: Option<String>,
    pub materialize: Option<String>,
    #[serde(rename = "max-extra-memory")]
    pub max_extra_memory: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct CudaSection {
    pub device: Option<usize>,
    #[serde(rename = "native-mxfp4-repack")]
    pub native_mxfp4_repack: Option<bool>,
    #[serde(rename = "cutlass-nvfp4-repack")]
    pub cutlass_nvfp4_repack: Option<bool>,
    #[serde(rename = "native-mxfp4-inference")]
    pub native_mxfp4_inference: Option<bool>,
    #[serde(rename = "prefill-attention")]
    pub prefill_attention: Option<String>,
    #[serde(rename = "prefill-chunk-size")]
    pub prefill_chunk_size: Option<usize>,
    #[serde(rename = "prefill-stage-timings")]
    pub prefill_stage_timings: Option<bool>,
}

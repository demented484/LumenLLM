use std::path::PathBuf;

use serde::Deserialize;

/// Top-level schema for `parameters.*.json` files.
///
/// Layout (post-rename):
/// ```jsonc
/// {
///   "model": {
///     "path": "...",
///     "store": "vram",      // default tier for non-block weights (embed, lm_head, final_norm);
///                           // also fallback for hidden-layers.{weights,kv-cache} when not overridden
///     "compute": "cuda:0"
///   },
///   "hidden-layers": {
///     "compute": "cuda:0",  // optional default compute for both sub-sections
///     "weights":  { "number": ..., "store": ..., "compute": ...,
///                   "fallback-store": ..., "fallback-compute": ... },
///     "kv-cache": { "number": ..., "context-size": ...,
///                   "store": ..., "fallback-store": ...,
///                   "type-k": ..., "type-v": ... }
///   },
///   "linear-layout":   { ... },
///   "other-parameters":{ ... },
///   "cuda":            { ... }
/// }
/// ```
///
/// Old top-level `layers` / `kv-cache` keys are rejected by `deny_unknown_fields`
/// to produce a clear error pointing users at the new path.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ParametersFile {
    #[serde(rename = "server-bin")]
    pub server_bin: Option<ServerBinSection>,
    #[serde(rename = "server-parameters")]
    pub server: Option<ServerSection>,
    pub model: ModelSection,
    /// Embed-tokens placement override (per-tensor, applies before layers load).
    #[serde(rename = "input-layer")]
    pub input_layer: Option<InputLayerSection>,
    /// LM-head placement override (per-tensor, applies before layers load).
    #[serde(rename = "output-layer")]
    pub output_layer: Option<OutputLayerSection>,
    /// Per-layer attention (Q/K/V/O) placement override; takes effect inside
    /// each `layer.N` region independently of the layer's MLP/expert weights.
    pub attention: Option<AttentionSection>,
    #[serde(rename = "hidden-layers")]
    pub hidden_layers: Option<HiddenLayersSection>,
    #[serde(rename = "linear-layout")]
    pub linear_layout: Option<LinearLayoutSection>,
    #[serde(rename = "other-parameters")]
    pub other: Option<OtherSection>,
    pub cuda: Option<CudaSection>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct InputLayerSection {
    pub store: Option<String>,
    pub compute: Option<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OutputLayerSection {
    pub store: Option<String>,
    pub compute: Option<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AttentionSection {
    /// Attention kernel/mechanism selector — passed through to
    /// `cuda.prefill_attention` (e.g. `"aegis-varlen"`, `"reference"`).
    pub mechanism: Option<String>,
    pub store: Option<String>,
    pub compute: Option<String>,
    /// Re-quantize the attention Q/K/V/O projections at load time.
    /// One of: "bf16" (default), "nvfp4", "fp8", "int8", "int4".
    /// Same semantics as `hidden-layers.shared-MLP-quantization`.
    #[serde(rename = "attention-quantization")]
    pub attention_quantization: Option<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ServerBinSection {
    pub path: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ServerSection {
    pub host: Option<String>,
    pub port: Option<u16>,
    #[serde(rename = "server-api")]
    pub server_api: Option<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModelSection {
    pub path: PathBuf,
    pub store: Option<String>,
    pub compute: Option<String>,
    pub mmap: Option<bool>,
}

/// Wrapper for the two sub-sections that together describe per-hidden-layer placement.
/// `compute` here (if set) is the **default compute target** for both sub-sections; each
/// sub-section may still override its own `compute`. There is no `store` at this level
/// by design — store must be explicit per sub-section, or fall back to `model.store`.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct HiddenLayersSection {
    pub compute: Option<String>,
    /// Shorthand for `weights.store`: applies to all hidden-layer weights.
    /// When both `store` and `weights.store` are present, `weights.store`
    /// wins (longer form is more specific).
    pub store: Option<String>,
    /// Re-quantize the shared expert (always-active MLP) at load time.
    /// One of: "bf16" (default — keep as stored), "nvfp4", "fp8", "int8",
    /// "int4". The checkpoint stores it as BF16; setting this to e.g.
    /// `"nvfp4"` runs a load-time quantizer (per-block absmax, no
    /// calibration) so the weights live in VRAM in the requested format
    /// and use the matching tensor-core GEMM during inference.
    #[serde(rename = "shared-MLP-quantization")]
    pub shared_mlp_quantization: Option<String>,
    pub weights: Option<HiddenLayerWeightsSection>,
    #[serde(rename = "kv-cache")]
    pub kv_cache: Option<HiddenLayerKvCacheSection>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct HiddenLayerWeightsSection {
    /// First `number` hidden layers use `store`/`compute`; the remaining hidden layers
    /// use `fallback-store`/`fallback-compute` (or fall back to `model.{store,compute}`).
    /// If omitted, applies to all hidden layers.
    /// If `number > num_hidden_layers`, clamped down with a warning (llama.cpp style).
    pub number: Option<usize>,
    pub store: Option<String>,
    pub compute: Option<String>,
    #[serde(rename = "fallback-store")]
    pub fallback_store: Option<String>,
    #[serde(rename = "fallback-compute")]
    pub fallback_compute: Option<String>,
}

/// KV cache section. Note: no `compute` / `fallback-compute` fields by design — KV cache
/// is read by the attention kernel that runs on the same compute target as the matching
/// layer's `weights`. Specifying compute separately would be redundant or contradictory.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct HiddenLayerKvCacheSection {
    pub number: Option<usize>,
    #[serde(rename = "context-size")]
    pub context_size: Option<usize>,
    pub store: Option<String>,
    #[serde(rename = "fallback-store")]
    pub fallback_store: Option<String>,
    #[serde(rename = "type-k")]
    pub type_k: Option<String>,
    #[serde(rename = "type-v")]
    pub type_v: Option<String>,
    #[serde(rename = "cache-prompt")]
    pub cache_prompt: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OtherSection {
    pub temperature: Option<f32>,
    #[serde(rename = "top-p")]
    pub top_p: Option<f32>,
    #[serde(rename = "top-k")]
    pub top_k: Option<usize>,
    #[serde(rename = "min-p")]
    pub min_p: Option<f32>,
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
#[serde(deny_unknown_fields)]
pub struct LinearLayoutSection {
    pub mode: Option<String>,
    pub cpu: Option<String>,
    pub cuda: Option<String>,
    pub materialize: Option<String>,
    #[serde(rename = "max-extra-memory")]
    pub max_extra_memory: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
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

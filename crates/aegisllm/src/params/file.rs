use std::path::PathBuf;

use serde::Deserialize;

/// Top-level schema for `parameters.*.json` files (schema v2).
///
/// All model + placement config lives under `model`. Optional modality towers
/// are top-level: present → load that tower (if the checkpoint has one) and
/// enable that input; absent → the tower is NOT loaded even when the checkpoint
/// contains one (saves VRAM/load time).
/// ```jsonc
/// {
///   "model": {
///     "path": "...",
///     "store": "vram",      // default tier for non-block weights (embed, lm_head, final_norm);
///                           // also fallback for hidden-layers.{weights,kv-cache} when not overridden
///     "compute": "cuda:0",
///     "input-layer":     { ... },
///     "output-layer":    { ... },
///     "attention":       { ... },
///     "hidden-layers":   { "compute": ..., "weights": {...}, "kv-cache": {...} },
///     "linear-layout":   { ... },
///     "other-parameters":{ ... }
///   },
///   "vision": { "compute": "cuda:0", "store": "vram" },   // omit -> vision NOT loaded
///   "audio":  { "compute": "cuda:0", "store": "vram" },   // omit -> audio NOT loaded
///   "cuda":   { ... }
/// }
/// ```
///
/// `deny_unknown_fields` rejects the old flat layout (top-level `input-layer`,
/// `hidden-layers`, etc.) with a clear error so users migrate to nesting under
/// `model`. Migrated configs live under the repo; pre-v2 backups in .configbak_pre_v2/.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ParametersFile {
    #[serde(rename = "server-bin")]
    pub server_bin: Option<ServerBinSection>,
    #[serde(rename = "server-parameters")]
    pub server: Option<ServerSection>,
    /// Everything that describes the *text model* and its placement lives under
    /// `model` (schema v2): the path, the model-wide store/compute defaults, and
    /// the per-section placement overrides (input-layer, output-layer, attention,
    /// hidden-layers, linear-layout, other-parameters).
    pub model: ModelSection,
    /// Vision encoder + multimodal projector placement. Schema v2: top-level,
    /// optional. PRESENT → load the vision tower (if the model artifact supplies
    /// one) and enable image input. ABSENT → the vision tower is NOT loaded even
    /// when the checkpoint contains one (saves VRAM/load time). `{compute,store}`
    /// default to the model section's values when unset.
    pub vision: Option<ModalitySection>,
    /// Audio encoder placement. Schema v2: top-level, optional. PRESENT → load
    /// the audio tower (if the model artifact supplies one) and enable audio
    /// input. ABSENT → the audio tower is NOT loaded even when the checkpoint
    /// contains one. `{compute,store}` default to the model section's values.
    pub audio: Option<ModalitySection>,
    pub cuda: Option<CudaSection>,
}

/// Placement for an optional modality tower (vision / audio). Mere presence of
/// the section enables loading that modality; absence disables it.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ModalitySection {
    /// Compute device for the modality tower + its projector. Defaults to the
    /// model section's `compute` when unset.
    pub compute: Option<String>,
    /// Storage tier for the modality tower + projector weights. Defaults to the
    /// model section's `store` when unset.
    pub store: Option<String>,
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
    /// Re-quantize the attention Q/K/V/O projection **weights** at load
    /// time. One of: "bf16" (default), "nvfp4", "fp8", "int8", "int4".
    /// Same semantics as `hidden-layers.shared-MLP-quantization`.
    ///
    /// NOTE: this is the *weight* quantization. To pick the precision the
    /// attention KERNEL runs in, use `compute-quantization` below — the two
    /// are independent knobs.
    #[serde(rename = "attention-quantization")]
    pub attention_quantization: Option<String>,
    /// Precision the prefill/decode attention KERNEL runs in.
    /// One of:
    ///   * "default" — historical dispatch (env gates only); bit-equivalent
    ///                 to leaving this unset.
    ///   * "bf16"    — explicit BF16 (half) attention kernels.
    ///   * "bf16-fa2"— BF16 FlashAttention-2 rewrite (head_dim=512 path);
    ///                 equivalent to exporting `AEGIS_ATTN_FA2=1`.
    ///   * "fp8"     — FP8 (E4M3) attention kernels; equivalent to exporting
    ///                 `AEGIS_ATTN_FP8=1`. Requires the KV cache to be FP8
    ///                 (`hidden-layers.kv-cache.type-k`/`type-v: fp8`)
    ///                 because the FP8 kernel reads FP8 K/V directly.
    ///
    /// This is the attention *compute* path — distinct from
    /// `attention-quantization` (weight quant) and from
    /// `hidden-layers.kv-cache.type-k/type-v` (KV storage dtype).
    #[serde(rename = "compute-quantization")]
    pub compute_quantization: Option<String>,
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

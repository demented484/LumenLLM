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
    /// EAGLE/MTP speculative-decoding draft model. Schema v2: top-level, optional.
    /// PRESENT → load the draft model alongside the target and enable
    /// speculative decoding. ABSENT → plain (non-speculative) decode. This
    /// mirrors the optional `vision`/`audio` sections: the draft is a model
    /// dependency, so it belongs in the config, not in CLI flags. An explicit
    /// `--draft-model` flag still overrides the config for quick experiments.
    pub draft: Option<DraftSection>,
    pub cuda: Option<CudaSection>,
}

/// Speculative-decoding draft model. Accepts the SAME placement block as the
/// primary `model` (path, store, compute, input-layer, output-layer, attention,
/// hidden-layers, ...) — flattened in — plus `num-draft-tokens`, so the draft is
/// configured exactly like the target.
///
/// NOTE on what the engine honors: the draft is an EAGLE/MTP model — tiny
/// (~152 MiB), and it SHARES the target's KV cache while running interleaved on
/// the target's device. So it is pinned to the target's device and has NO
/// separate KV cache. The meaningful knobs are `path`, `num-draft-tokens`, and
/// `store`/`compute` (where the draft's own weights live). The `hidden-layers`
/// `kv-cache` and per-section attention sub-fields are accepted for config
/// symmetry but inherited from the target (the draft does not own a KV cache).
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct DraftSection {
    #[serde(flatten)]
    pub model: ModelSection,
    /// Tokens proposed per speculative round. Defaults to 4 when unset.
    #[serde(rename = "num-draft-tokens")]
    pub num_draft_tokens: Option<usize>,
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
    /// API keys accepted by the server. When non-empty, every /v1/* generation
    /// request must present a matching key (OpenAI: `Authorization: Bearer <key>`;
    /// Anthropic: `x-api-key: <key>`). Empty/absent → server is open (local use).
    /// The `AEGIS_API_KEY` env var (comma-separated) is merged in at parse time.
    #[serde(rename = "api-keys")]
    pub api_keys: Option<Vec<String>>,
}

// NOTE: no `#[serde(deny_unknown_fields)]` here — ModelSection is `#[serde(flatten)]`
// into DraftSection (which adds `num-draft-tokens`), and serde flatten is
// incompatible with deny_unknown_fields. Typos are still caught by the leaf
// sections (attention, hidden-layers, kv-cache, …) which keep deny_unknown_fields,
// and by ParametersFile's deny_unknown_fields at the top level.
#[derive(Debug, Clone, Deserialize, PartialEq)]
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
    pub weights: Option<HiddenLayerWeightsSection>,
    /// Placement for the ROUTED (sparse, top-k) experts only — distinct from the
    /// shared expert / GDN / attention, which follow `hidden-layers.compute`.
    /// Sits beside `weights` / `kv-cache`. When `compute: cpu`, the routed-expert
    /// decode GEMV runs on the CPU (read in place from the host arena — no
    /// 540 MiB/token H2D stream) instead of `cuda:0`. `store` controls routed-
    /// expert residency (ram/mmap = host arena). ABSENT → routed experts fall
    /// back to `hidden-layers.compute` (the unchanged GPU path).
    pub experts: Option<HiddenLayerExpertsSection>,
    #[serde(rename = "kv-cache")]
    pub kv_cache: Option<HiddenLayerKvCacheSection>,
    /// Arbitrary per-layer-range placement, e.g. a 4-way CPU/GPU split:
    /// ```jsonc
    /// "ranges": [
    ///   { "start": 0,  "end": 17, "store": "vram", "compute": "cuda:0" },
    ///   { "start": 17, "end": 42, "store": "ram",  "compute": "cpu" }
    /// ]
    /// ```
    /// Each entry resolves to a `PlacementRule { selector: Range{start,end}, .. }`
    /// (half-open `[start, end)`). Ranges are applied in array order AFTER the
    /// `weights` first-N rule, so a later range overrides an earlier one for any
    /// layer it covers. This is the only way to express a layer split that is
    /// not first-N (e.g. layers 17..42 on CPU). Coexists with `weights`/`store`/
    /// `compute`: omit it and the legacy first-N/shorthand path is unchanged.
    pub ranges: Option<Vec<HiddenLayerRangeSection>>,
}

/// One `[start, end)` layer range with its own store/compute placement.
/// At least one of `store` / `compute` must be set (an empty entry is a no-op
/// and rejected at parse time). `compute` falls back to the enclosing
/// `hidden-layers.compute` when unset; `store` has no fallback here (it inherits
/// the resolved per-layer store from earlier rules / `model.store`).
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct HiddenLayerRangeSection {
    pub start: usize,
    pub end: usize,
    pub store: Option<String>,
    pub compute: Option<String>,
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

/// Routed-expert placement. Mirrors the `weights` / `kv-cache` sub-sections but
/// applies ONLY to the sparse top-k routed experts of a MoE layer (NVFP4 in our
/// Qwen3.x / Gemma-4 checkpoints). The shared expert, GDN/attention, and router
/// stay on `hidden-layers.compute` regardless — they are not the routed experts.
///
/// Semantics:
///   * `compute: cpu`    → the routed-expert decode GEMV runs on the CPU
///                         (`aegisllm-cpu::moe_layer_experts_into`), reading the
///                         packed NVFP4 bytes IN PLACE from the host arena. This
///                         keeps the 540 MiB/token expert stream OFF the PCIe link.
///                         Requires the experts to be host-resident (`store: ram`
///                         or `mmap`) — enforced at materialization, not parse.
///   * `compute: cuda:N` → the unchanged GPU streaming path.
///   * `compute` omitted → falls back to `hidden-layers.compute` → `model.compute`
///                         (so every existing config behaves EXACTLY as before).
///   * `store`           → routed-expert residency (`ram`/`mmap` host arena, or
///                         `vram`). Falls back to the layer-region store.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct HiddenLayerExpertsSection {
    pub compute: Option<String>,
    pub store: Option<String>,
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
    pub threads: Option<usize>,
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

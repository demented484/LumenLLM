use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::{AegisError, Result};
use crate::tensor::TensorRegistry;
use crate::tensor::quant::WeightQuantization;

#[derive(Debug, Clone, PartialEq)]
pub struct ModelArtifact {
    pub root: PathBuf,
    pub config: HfConfig,
    pub generation_config: Option<HfGenerationConfig>,
    pub tokenizer_config: Option<HfTokenizerConfig>,
    pub weights: WeightManifest,
    pub tensors: TensorRegistry,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelArtifactSummary {
    pub root: PathBuf,
    pub shard_count: usize,
    pub available_shards: usize,
    pub total_tensors: usize,
    pub total_size_bytes: Option<u64>,
    pub total_parameters: Option<u64>,
    pub complete: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WeightManifest {
    pub index_path: PathBuf,
    pub weight_map: BTreeMap<String, String>,
    pub shard_files: Vec<ShardFile>,
    pub total_tensors: usize,
    pub total_size_bytes: Option<u64>,
    pub total_parameters: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardFile {
    pub name: String,
    pub path: PathBuf,
    pub tensor_count: usize,
    pub bytes_on_disk: u64,
    pub lfs_pointer: bool,
    pub expected_bytes: Option<u64>,
}

#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct HfConfig {
    // ── Universal fields ───────────────────────────────────────────────────
    pub architectures: Option<Vec<String>>,
    #[serde(default)]
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: Option<usize>,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub head_dim: Option<usize>,
    pub max_position_embeddings: Option<usize>,
    pub rms_norm_eps: Option<f64>,
    pub rope_scaling: Option<HfRopeScaling>,
    pub rope_theta: Option<f64>,
    /// Gemma 4: explicit per-attention-type rope_theta (sliding=10k, global=1M).
    /// Hoisted from `rope_parameters.{sliding_attention,full_attention}.rope_theta`.
    #[serde(default)]
    pub rope_theta_sliding: Option<f64>,
    #[serde(default)]
    pub rope_theta_global: Option<f64>,
    pub torch_dtype: Option<String>,
    pub quantization_config: Option<HfQuantizationConfig>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<serde_json::Value>,
    pub vocab_size: Option<usize>,

    // ── Sliding-window attention (Gemma 4, Qwen 3.6) ──────────────────────
    /// Attention window size; None or 0 = full causal attention.
    pub sliding_window: Option<usize>,
    /// Gemma 4 pattern: every Nth layer uses full (global) attention.
    pub global_attn_every_n_layers: Option<usize>,

    // ── Logit soft-capping (Gemma 4) ──────────────────────────────────────
    /// `logits = cap * tanh(logits / cap)` applied before softmax.
    pub attn_logit_softcapping: Option<f32>,
    /// Applied to the final lm_head logits.
    pub final_logit_softcapping: Option<f32>,

    // ── Proportional RoPE (Gemma 4 global layers) ─────────────────────────
    /// Fraction of head_dim that gets RoPE; the rest passes through.
    pub partial_rotary_factor: Option<f32>,

    // ── Pre+Post RMSNorm (Gemma 4) ────────────────────────────────────────
    /// When true, a second RMSNorm is applied after attention/MLP output.
    pub post_attention_layernorm: Option<bool>,

    // ── Per-layer type list (Gemma 4, Qwen 3.5/3.6 hybrid) ──────────────
    /// Explicit per-layer attention-type strings. When present, overrides
    /// interval-based calculations. Values: "sliding_attention",
    /// "full_attention", "linear_attention".
    pub layer_types: Option<Vec<String>>,

    // ── Gemma 4 per-attention-kind dimensions ─────────────────────────────
    /// Head dimension for global (full) attention layers in Gemma 4.
    /// Defaults to `head_dim` when absent.
    pub global_head_dim: Option<usize>,
    /// Number of KV heads for global attention layers (Gemma 4 26B).
    pub num_global_key_value_heads: Option<usize>,
    /// Last N layers share their KV cache with an earlier layer of the same
    /// attention type (Gemma 4 E4B feature, optional skip in Phase 4).
    pub num_kv_shared_layers: Option<usize>,
    /// When true, K and V projections share the same weight matrix.
    pub attention_k_eq_v: Option<bool>,

    // ── Mixture-of-Experts (Gemma 4 MoE, Qwen 3.x, Nemotron 3) ──────────
    /// When true this model has MoE blocks (Gemma 4 26B).
    pub enable_moe_block: Option<bool>,
    #[serde(alias = "n_routed_experts")]
    pub num_experts: Option<usize>,
    /// Top-k experts selected per token.
    #[serde(alias = "top_k_experts", alias = "num_experts_per_tok")]
    pub num_experts_per_tok: Option<usize>,
    /// Per-expert intermediate size (Qwen 3 MoE: "moe_intermediate_size").
    pub moe_intermediate_size: Option<usize>,
    /// Intermediate size for shared (always-active) expert if present.
    pub shared_expert_intermediate_size: Option<usize>,
    #[serde(alias = "n_shared_experts")]
    pub num_shared_experts: Option<usize>,
    /// Interleave pattern: how often a non-MoE (dense) layer appears.
    pub moe_layer_freq: Option<usize>,

    // ── Gated DeltaNet (Qwen 3.5/3.6 hybrid) ─────────────────────────────
    /// When true, linear-attention (GDN) layers exist in the model.
    pub use_linear_attention: Option<bool>,
    /// Number of GDN linear-attention layers (if not derivable from freq).
    pub num_linear_attention_layers: Option<usize>,
    /// Alternation pattern: every N-th layer is GDN (e.g. N=4 → 3 GDN + 1 full).
    pub linear_attn_every_n_layers: Option<usize>,
    /// Fallback interval: full attention every N layers, GDN for the rest.
    pub full_attention_interval: Option<usize>,
    // GDN per-head dimensions
    pub linear_num_key_heads: Option<usize>,
    pub linear_key_head_dim: Option<usize>,
    pub linear_value_head_dim: Option<usize>,
    pub linear_num_value_heads: Option<usize>,
    pub linear_conv_kernel_dim: Option<usize>,
    /// When true, a gated output norm is applied after the GDN output projection.
    pub attn_output_gate: Option<bool>,

    // ── Mamba / SSM (Nemotron 3) ──────────────────────────────────────────
    /// SSM state dimension (d_state in Mamba / ssm_state_size in Nemotron).
    #[serde(alias = "ssm_state_size")]
    pub state_size: Option<usize>,
    /// Number of SSM heads.
    pub mamba_num_heads: Option<usize>,
    /// Head dimension for SSM.
    pub mamba_head_dim: Option<usize>,
    /// Mamba expansion factor (d_inner = expand * hidden_size).
    pub expand: Option<usize>,
    /// Chunk size for chunked Mamba prefill.
    pub chunk_size: Option<usize>,
    /// Convolutional kernel size in Mamba layers.
    pub conv_kernel: Option<usize>,
    /// Hybrid layer pattern string (Nemotron): 'M'=Mamba, 'E'=MoE, '*'=attn.
    pub hybrid_override_pattern: Option<String>,

    // ── Multimodal / Omni (Nemotron 3 Omni) ─────────────────────────────
    /// Modalities the model can consume.
    pub supported_modalities: Option<Vec<String>>,

    // ── MatFormer / nested params (Gemma 4 E2B/E4B) ──────────────────────
    /// Active model size within a nested-param checkpoint (e.g. "e2b").
    pub effective_size: Option<String>,
    /// Granularity of nested param blocks (e.g. ["1.0b", "2.0b", "4.0b"]).
    pub nested_param_sizes: Option<Vec<String>>,

    // ── Qwen-specific ────────────────────────────────────────────────────
    /// Number of attention layers when using hybrid GDN model.
    pub num_attention_heads_per_layer: Option<Vec<usize>>,
}

/// Effective dimensions for a MatFormer-style nested-param checkpoint.
///
/// Returned by `HfConfig::effective_dims()`. When the config has no
/// `effective_size` field, callers should use `hidden_size`/`intermediate_size`
/// directly without a slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EffectiveDims {
    /// Effective hidden dimension after applying the nested-param slice.
    pub hidden_size: usize,
    /// Effective FFN intermediate dimension; `None` if not present in config.
    pub intermediate_size: Option<usize>,
    /// True when the slice is strictly smaller than the full checkpoint.
    pub is_sliced: bool,
}

impl HfConfig {
    /// Resolve the active hidden / intermediate dimensions for this config,
    /// applying the MatFormer `effective_size` slice if present.
    ///
    /// Returns the full checkpoint dims when `effective_size` is `None` or
    /// unrecognized; returns sliced dims when set to `"e2b"` or `"e4b"`.
    pub fn effective_dims(&self) -> EffectiveDims {
        let label = match self.effective_size.as_deref() {
            Some(s) => s,
            None => {
                return EffectiveDims {
                    hidden_size: self.hidden_size,
                    intermediate_size: self.intermediate_size,
                    is_sliced: false,
                };
            }
        };
        let scale = match label.to_ascii_lowercase().as_str() {
            "e2b" | "2b" => 0.5f32,
            "e4b" | "4b" => 1.0f32,
            _ => {
                return EffectiveDims {
                    hidden_size: self.hidden_size,
                    intermediate_size: self.intermediate_size,
                    is_sliced: false,
                };
            }
        };
        let hidden_eff = ((self.hidden_size as f32) * scale).round().max(1.0) as usize;
        let interm_eff = self
            .intermediate_size
            .map(|d| ((d as f32) * scale).round().max(1.0) as usize);
        EffectiveDims {
            hidden_size: hidden_eff,
            intermediate_size: interm_eff,
            is_sliced: scale < 1.0,
        }
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct HfQuantizationConfig {
    pub quant_algo: Option<String>,
    /// HuggingFace standard alternative key for quantization name (FP8 / GPTQ / AWQ).
    pub quant_method: Option<String>,
    pub kv_cache_scheme: Option<HfKvCacheScheme>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct HfKvCacheScheme {
    pub num_bits: Option<u8>,
    #[serde(rename = "type")]
    pub value_type: Option<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct HfRopeScaling {
    pub factor: Option<f64>,
    pub low_freq_factor: Option<f64>,
    pub high_freq_factor: Option<f64>,
    pub original_max_position_embeddings: Option<usize>,
    pub rope_type: Option<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct HfGenerationConfig {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub do_sample: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct HfTokenizerConfig {
    pub tokenizer_class: Option<String>,
    /// HuggingFace tokenizer configs sometimes set this to a sentinel like
    /// `1e30` to signal "no limit", which overflows usize. Deserialize as
    /// `f64` and clamp at use sites.
    pub model_max_length: Option<f64>,
    pub chat_template: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct SafetensorsIndex {
    metadata: Option<SafetensorsIndexMetadata>,
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Deserialize)]
struct SafetensorsIndexMetadata {
    total_parameters: Option<u64>,
    total_size: Option<u64>,
}

impl ModelArtifact {
    pub fn from_local_path(path: impl AsRef<Path>) -> Result<Self> {
        let root = path.as_ref();
        let root = if root.is_file() {
            root.parent().ok_or_else(|| {
                AegisError::InvalidConfig(format!(
                    "cannot infer model root from file path {}",
                    root.display()
                ))
            })?
        } else {
            root
        };
        let root = root.canonicalize()?;
        let config = read_hf_config(&root.join("config.json"))?;
        let generation_config = read_json_optional(&root.join("generation_config.json"))?;
        let tokenizer_config = read_json_optional(&root.join("tokenizer_config.json"))?;
        let weights = WeightManifest::from_root(&root)?;
        let tensors = TensorRegistry::from_index_and_shards(&root, &weights.weight_map)?;

        Ok(Self {
            root,
            config,
            generation_config,
            tokenizer_config,
            weights,
            tensors,
        })
    }

    pub fn summary(&self) -> ModelArtifactSummary {
        let available_shards = self
            .weights
            .shard_files
            .iter()
            .filter(|shard| !shard.lfs_pointer && shard.bytes_on_disk > 0)
            .count();
        ModelArtifactSummary {
            root: self.root.clone(),
            shard_count: self.weights.shard_files.len(),
            available_shards,
            total_tensors: self.weights.total_tensors,
            total_size_bytes: self.weights.total_size_bytes,
            total_parameters: self.weights.total_parameters,
            complete: available_shards == self.weights.shard_files.len(),
        }
    }

    pub fn infer_weight_quantization(&self) -> WeightQuantization {
        if let Some(qcfg) = self.config.quantization_config.as_ref() {
            if let Some(quant) = qcfg.quant_algo.as_deref() {
                return WeightQuantization::parse_guess(quant);
            }
            if let Some(method) = qcfg.quant_method.as_deref() {
                return WeightQuantization::parse_guess(method);
            }
        }
        WeightQuantization::parse_guess(self.config.torch_dtype.as_deref().unwrap_or("none"))
    }

    pub fn head_dim(&self) -> usize {
        self.config
            .head_dim
            .unwrap_or(self.config.hidden_size / self.config.num_attention_heads)
    }
}

impl WeightManifest {
    pub fn from_root(root: &Path) -> Result<Self> {
        let index_path = root.join("model.safetensors.index.json");
        // Single-shard models (e.g. Gemma 4 E4B) ship only `model.safetensors`
        // without a companion index.json. Build a synthetic manifest by reading
        // the safetensors header directly.
        if !index_path.exists() {
            let single = root.join("model.safetensors");
            if single.exists() {
                return Self::from_single_shard(root, &single);
            }
        }
        let index: SafetensorsIndex = read_json_required(&index_path)?;
        let mut shard_names = BTreeSet::new();
        let mut tensor_counts = BTreeMap::<String, usize>::new();
        for shard_name in index.weight_map.values() {
            shard_names.insert(shard_name.clone());
            *tensor_counts.entry(shard_name.clone()).or_insert(0) += 1;
        }

        let mut shard_files = Vec::new();
        for shard_name in shard_names {
            let path = root.join(&shard_name);
            let bytes_on_disk = fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
            let pointer = path
                .exists()
                .then(|| parse_lfs_pointer(&path))
                .transpose()?
                .flatten();
            shard_files.push(ShardFile {
                name: shard_name.clone(),
                path,
                tensor_count: tensor_counts.get(&shard_name).copied().unwrap_or(0),
                bytes_on_disk,
                lfs_pointer: pointer.is_some(),
                expected_bytes: pointer,
            });
        }

        Ok(Self {
            index_path,
            total_tensors: index.weight_map.len(),
            total_size_bytes: index.metadata.as_ref().and_then(|m| m.total_size),
            total_parameters: index.metadata.as_ref().and_then(|m| m.total_parameters),
            weight_map: index.weight_map,
            shard_files,
        })
    }

    /// Builds a manifest from a single `model.safetensors` file by reading
    /// its embedded JSON header (no companion `.index.json` needed).
    fn from_single_shard(root: &Path, shard_path: &Path) -> Result<Self> {
        let shard_name = "model.safetensors".to_string();
        let header = read_safetensors_header(shard_path)?;
        let mut weight_map = BTreeMap::new();
        let mut tensor_count = 0usize;
        for key in header.keys() {
            if key != "__metadata__" {
                weight_map.insert(key.clone(), shard_name.clone());
                tensor_count += 1;
            }
        }
        let bytes_on_disk = fs::metadata(shard_path).map(|m| m.len()).unwrap_or(0);
        Ok(Self {
            index_path: root.join("model.safetensors.index.json"),
            total_tensors: tensor_count,
            total_size_bytes: None,
            total_parameters: None,
            weight_map,
            shard_files: vec![ShardFile {
                name: shard_name,
                path: shard_path.to_path_buf(),
                tensor_count,
                bytes_on_disk,
                lfs_pointer: false,
                expected_bytes: None,
            }],
        })
    }
}

/// Reads the safetensors header JSON from the first bytes of a `.safetensors`
/// file. The format is: [8 bytes LE u64 N][N bytes JSON header].
/// Returns a map of tensor_name → {dtype, shape, data_offsets}.
fn read_safetensors_header(
    path: &Path,
) -> Result<serde_json::Map<String, serde_json::Value>> {
    use std::io::Read;
    let mut f = std::io::BufReader::new(fs::File::open(path)?);
    let mut len_bytes = [0u8; 8];
    f.read_exact(&mut len_bytes)
        .map_err(|e| AegisError::InvalidConfig(format!("safetensors read: {e}")))?;
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    if header_len > 256 * 1024 * 1024 {
        return Err(AegisError::InvalidConfig(format!(
            "safetensors header too large: {header_len} bytes"
        )));
    }
    let mut header_bytes = vec![0u8; header_len];
    f.read_exact(&mut header_bytes)
        .map_err(|e| AegisError::InvalidConfig(format!("safetensors header read: {e}")))?;
    let value: serde_json::Value = serde_json::from_slice(&header_bytes)
        .map_err(|e| AegisError::InvalidConfig(format!("safetensors header JSON: {e}")))?;
    value
        .as_object()
        .cloned()
        .ok_or_else(|| AegisError::InvalidConfig("safetensors header not an object".into()))
}

fn read_json_required<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    if !path.exists() {
        return Err(AegisError::MissingFile(path.to_path_buf()));
    }
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

/// Reads `config.json` and accommodates the HuggingFace multimodal layout where
/// the text-model fields are nested under `text_config` (Qwen3.5, Gemma 4,
/// Nemotron Omni). When `hidden_size` is absent at root but present in
/// `text_config`, the `text_config` object is flattened into the root before
/// deserialization. Other nested configs (`vision_config`, `audio_config`) are
/// preserved as opaque siblings; the planner currently ignores them.
fn read_hf_config(path: &Path) -> Result<HfConfig> {
    if !path.exists() {
        return Err(AegisError::MissingFile(path.to_path_buf()));
    }
    let bytes = fs::read(path)?;
    let mut value: serde_json::Value = serde_json::from_slice(&bytes)?;
    // Flatten text_config (Qwen3.5, Gemma4, Nemotron Omni outer wrapper)
    // or llm_config (Nemotron Omni inner LLM config) when hidden_size is
    // absent at the top level.
    let needs_flatten = value
        .as_object()
        .map(|obj| !obj.contains_key("hidden_size") && obj.contains_key("text_config"))
        .unwrap_or(false);
    if needs_flatten {
        if let Some(obj) = value.as_object_mut() {
            if let Some(serde_json::Value::Object(text_cfg)) = obj.remove("text_config") {
                for (k, v) in text_cfg {
                    obj.entry(k).or_insert(v);
                }
            }
        }
    }
    // Second pass: llm_config (Nemotron Omni wraps the LLM under llm_config)
    let needs_flatten_llm = value
        .as_object()
        .map(|obj| !obj.contains_key("hidden_size") && obj.contains_key("llm_config"))
        .unwrap_or(false);
    if needs_flatten_llm {
        if let Some(obj) = value.as_object_mut() {
            if let Some(serde_json::Value::Object(llm_cfg)) = obj.remove("llm_config") {
                for (k, v) in llm_cfg {
                    obj.entry(k).or_insert(v);
                }
            }
        }
    }
    // HF moved `torch_dtype` → `dtype` in transformers v5; mirror it back so
    // existing downstream consumers (`infer_weight_quantization`, etc.) still work.
    if let Some(obj) = value.as_object_mut() {
        if !obj.contains_key("torch_dtype") {
            if let Some(dtype) = obj.get("dtype").cloned() {
                obj.insert("torch_dtype".into(), dtype);
            }
        }
    }
    // Gemma 4 nests per-attention-type rope params under `rope_parameters`.
    // Hoist `partial_rotary_factor` from `rope_parameters.full_attention` so
    // that HfConfig.partial_rotary_factor gets populated correctly.
    if let Some(obj) = value.as_object_mut() {
        if !obj.contains_key("partial_rotary_factor") {
            let factor = obj
                .get("rope_parameters")
                .and_then(|rp| rp.get("full_attention"))
                .and_then(|fa| fa.get("partial_rotary_factor"))
                .and_then(|f| f.as_f64());
            if let Some(f) = factor {
                obj.insert("partial_rotary_factor".into(), serde_json::json!(f));
            }
        }
        // Per-layer-type rope_theta (Gemma 4: sliding=10k, global=1M).
        for (key, lt) in [
            ("rope_theta_sliding", "sliding_attention"),
            ("rope_theta_global", "full_attention"),
        ] {
            if !obj.contains_key(key)
                && let Some(theta) = obj
                    .get("rope_parameters")
                    .and_then(|rp| rp.get(lt))
                    .and_then(|p| p.get("rope_theta"))
                    .and_then(|f| f.as_f64())
            {
                obj.insert(key.into(), serde_json::json!(theta));
            }
        }
        // Fallback for top-level rope_theta — many Gemma 4 configs only set it under
        // rope_parameters.{layer_type}; without this hoist, RopeConfig::from_artifact
        // would silently default to 10_000.
        if !obj.contains_key("rope_theta") {
            let global = obj.get("rope_theta_global").and_then(|f| f.as_f64());
            let sliding = obj.get("rope_theta_sliding").and_then(|f| f.as_f64());
            if let Some(theta) = sliding.or(global) {
                obj.insert("rope_theta".into(), serde_json::json!(theta));
            }
        }
    }
    Ok(serde_json::from_value(value)?)
}

fn read_json_optional<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Option<T>> {
    if path.exists() {
        Ok(Some(read_json_required(path)?))
    } else {
        Ok(None)
    }
}

fn parse_lfs_pointer(path: &Path) -> Result<Option<u64>> {
    // Git-LFS pointer files are tiny (~134 bytes); anything > 512 is not a
    // pointer. Check the file size BEFORE reading — the previous version
    // unconditionally loaded the whole file into a Vec via `fs::read`, then
    // discarded everything if the size exceeded the cap. For 17 GiB of
    // safetensors shards in a Gemma-4-26B layout, this wasted ~34 GiB of
    // disk reads + transient Vec allocations on every artifact-open
    // (and the Serve CLI opens twice — preview + real — so 68 GiB total).
    // The runaway anon allocations were the visible "sawtooth" RAM pattern.
    let size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    if size > 512 {
        return Ok(None);
    }
    let bytes = fs::read(path)?;
    let text = String::from_utf8_lossy(&bytes);
    if !text.starts_with("version https://git-lfs.github.com/spec/v1") {
        return Ok(None);
    }
    Ok(text.lines().find_map(|line| {
        line.strip_prefix("size ")
            .and_then(|value| value.trim().parse::<u64>().ok())
    }))
}


#[cfg(test)]
mod tests {
    use super::HfConfig;

    fn make_config(effective: Option<&str>) -> HfConfig {
        HfConfig {
            model_type: "gemma4".into(),
            hidden_size: 4096,
            intermediate_size: Some(16384),
            num_hidden_layers: 32,
            num_attention_heads: 8,
            effective_size: effective.map(String::from),
            ..Default::default()
        }
    }

    #[test]
    fn effective_dims_returns_full_when_unset() {
        let cfg = make_config(None);
        let d = cfg.effective_dims();
        assert_eq!(d.hidden_size, 4096);
        assert_eq!(d.intermediate_size, Some(16384));
        assert!(!d.is_sliced);
    }

    #[test]
    fn effective_dims_halves_for_e2b() {
        let cfg = make_config(Some("e2b"));
        let d = cfg.effective_dims();
        assert_eq!(d.hidden_size, 2048);
        assert_eq!(d.intermediate_size, Some(8192));
        assert!(d.is_sliced);
    }

    #[test]
    fn effective_dims_full_for_e4b() {
        let cfg = make_config(Some("e4b"));
        let d = cfg.effective_dims();
        assert_eq!(d.hidden_size, 4096);
        assert_eq!(d.intermediate_size, Some(16384));
        assert!(!d.is_sliced);
    }

    #[test]
    fn effective_dims_falls_back_when_unknown() {
        let cfg = make_config(Some("eXX"));
        let d = cfg.effective_dims();
        assert_eq!(d.hidden_size, 4096);
        assert!(!d.is_sliced);
    }
}


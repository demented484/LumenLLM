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

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct HfConfig {
    pub architectures: Option<Vec<String>>,
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
    pub torch_dtype: Option<String>,
    pub quantization_config: Option<HfQuantizationConfig>,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<serde_json::Value>,
    pub vocab_size: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct HfQuantizationConfig {
    pub quant_algo: Option<String>,
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
    pub model_max_length: Option<usize>,
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
        let config = read_json_required(&root.join("config.json"))?;
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
        if let Some(quant) = self
            .config
            .quantization_config
            .as_ref()
            .and_then(|config| config.quant_algo.as_deref())
        {
            return WeightQuantization::parse_guess(quant);
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
}

fn read_json_required<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    if !path.exists() {
        return Err(AegisError::MissingFile(path.to_path_buf()));
    }
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn read_json_optional<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Option<T>> {
    if path.exists() {
        Ok(Some(read_json_required(path)?))
    } else {
        Ok(None)
    }
}

fn parse_lfs_pointer(path: &Path) -> Result<Option<u64>> {
    let bytes = fs::read(path)?;
    if bytes.len() > 512 {
        return Ok(None);
    }
    let text = String::from_utf8_lossy(&bytes);
    if !text.starts_with("version https://git-lfs.github.com/spec/v1") {
        return Ok(None);
    }
    Ok(text.lines().find_map(|line| {
        line.strip_prefix("size ")
            .and_then(|value| value.trim().parse::<u64>().ok())
    }))
}

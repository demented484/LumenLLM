use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::cuda::{CudaPrefillAttentionKernel, CudaRuntimeConfig};
use crate::error::{AegisError, Result};
use crate::generation::SamplingConfig;
use crate::planning::placement::{
    ComputePlacement, LayerSelector, PlacementPolicy, PlacementRule, StoragePlacement,
};
use crate::tensor::layout::{LinearLayoutChoice, LinearLayoutPolicy, MaterializationPolicy};
use crate::tensor::quant::KvCacheQuantization;

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

impl ParametersFile {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        Ok(serde_json::from_slice(&fs::read(path)?)?)
    }

    pub fn into_serve_config(self, default_policy: PlacementPolicy) -> Result<ServeConfig> {
        let host = self
            .server
            .as_ref()
            .and_then(|server| server.host.clone())
            .unwrap_or_else(|| "127.0.0.1".into());
        let port = self
            .server
            .as_ref()
            .and_then(|server| server.port)
            .unwrap_or(1337);
        let api = self
            .server
            .as_ref()
            .and_then(|server| server.server_api.clone())
            .unwrap_or_else(|| "openai".into());

        Ok(ServeConfig {
            host,
            port,
            api,
            engine: self.into_engine_fragment(default_policy)?,
        })
    }

    pub fn into_engine_fragment(self, mut policy: PlacementPolicy) -> Result<EngineConfigFragment> {
        let cuda_device = self.cuda.as_ref().and_then(|cuda| cuda.device).unwrap_or(0);
        if let Some(device) = self.cuda.as_ref().and_then(|cuda| cuda.device) {
            retarget_cuda_policy(&mut policy, device);
        }
        let mut cuda_runtime = CudaRuntimeConfig::from_env();
        let mut explicit_cuda_prefill_attention = false;
        if let Some(cuda) = &self.cuda {
            if let Some(value) = cuda.native_mxfp4_repack {
                cuda_runtime.native_mxfp4_repack = value;
            }
            if let Some(value) = cuda.cutlass_nvfp4_repack {
                cuda_runtime.cutlass_nvfp4_repack = value;
            }
            if let Some(value) = cuda.native_mxfp4_inference {
                cuda_runtime.native_mxfp4_inference = value;
            }
            if let Some(value) = cuda.prefill_attention.as_deref() {
                cuda_runtime.prefill_attention = CudaPrefillAttentionKernel::parse(value)?;
                explicit_cuda_prefill_attention = true;
            }
            if let Some(value) = cuda.prefill_chunk_size {
                cuda_runtime.prefill_chunk_size = Some(value);
            }
            if let Some(value) = cuda.prefill_stage_timings {
                cuda_runtime.prefill_stage_timings = value;
            }
        }
        let model_path = self.model.path;

        let mmap_enabled = self.model.mmap.unwrap_or(true);
        if let Some(store) = self.model.store {
            policy.weights_store = parse_storage(&store, cuda_device)?;
        } else if !mmap_enabled && policy.weights_store == StoragePlacement::Mmap {
            policy.weights_store = StoragePlacement::Ram;
            policy.spill_store = StoragePlacement::Ram;
        }
        if let Some(compute) = self.model.compute {
            let compute = parse_compute(&compute, cuda_device)?;
            policy.weights_compute = compute;
            policy.spill_compute = compute;
        }

        if let Some(kv) = self.kv_cache {
            if let Some(context_size) = kv.context_size {
                policy.context_size = context_size;
            }
            if let Some(store) = kv.store {
                policy.kv_store = parse_storage(&store, cuda_device)?;
            }
            if let Some(compute) = kv.compute {
                policy.kv_compute = parse_compute(&compute, cuda_device)?;
            }
            if let Some(value) = kv.type_k.or(kv.type_v) {
                policy.kv_quantization = KvCacheQuantization::parse(&value).ok_or_else(|| {
                    AegisError::InvalidConfig(format!(
                        "unsupported kv cache quantization `{value}`"
                    ))
                })?;
            }
        }

        if let Some(layout) = self.linear_layout {
            apply_linear_layout_section(&mut policy.linear_layout, layout)?;
        }
        if let Some(other) = &self.other {
            if let Some(value) = other.cpu_linear_layout.as_deref() {
                policy.linear_layout.cpu = LinearLayoutChoice::parse(value)?;
            }
            if let Some(value) = other.cuda_linear_layout.as_deref() {
                policy.linear_layout.cuda = LinearLayoutChoice::parse(value)?;
            }
            if let Some(value) = other.linear_materialize.as_deref() {
                policy.linear_layout.materialization = MaterializationPolicy::parse(value)?;
            }
            if !explicit_cuda_prefill_attention && let Some(value) = other.flash_attention {
                cuda_runtime.prefill_attention = if value {
                    CudaPrefillAttentionKernel::Auto
                } else {
                    CudaPrefillAttentionKernel::Reference
                };
            }
            if cuda_runtime.prefill_chunk_size.is_none() {
                cuda_runtime.prefill_chunk_size = other.ubatch_size.or(other.batch_size);
            }
        }
        let mut generation = SamplingConfig::default();
        if let Some(other) = &self.other {
            if let Some(value) = other.temperature {
                generation.temperature = value;
            }
            if let Some(value) = other.top_p {
                generation.top_p = value;
            }
            if let Some(value) = other.top_k {
                generation.top_k = value;
            }
        }

        policy.rules.clear();
        if let Some(layers) = self.layers {
            let n = layers.number.unwrap_or(0);
            let base_store = layers
                .rest_store
                .as_ref()
                .or_else(|| if n == 0 { layers.store.as_ref() } else { None });
            let base_compute = layers.rest_compute.as_ref().or_else(|| {
                if n == 0 {
                    layers.compute.as_ref()
                } else {
                    None
                }
            });
            if let Some(store) = base_store {
                policy.weights_store = parse_storage(store, cuda_device)?;
                policy.spill_store = policy.weights_store;
            }
            if let Some(compute) = base_compute {
                policy.weights_compute = parse_compute(compute, cuda_device)?;
                policy.spill_compute = policy.weights_compute;
            }
            if n > 0 {
                policy.rules.push(PlacementRule {
                    selector: LayerSelector::FirstN { n },
                    store: layers
                        .store
                        .as_deref()
                        .map(|value| parse_storage(value, cuda_device))
                        .transpose()?,
                    compute: layers
                        .compute
                        .as_deref()
                        .map(|value| parse_compute(value, cuda_device))
                        .transpose()?,
                });
            }
        }

        Ok(EngineConfigFragment {
            model_path,
            policy,
            cuda: cuda_runtime,
            generation,
        })
    }
}

pub(crate) fn retarget_cuda_policy(policy: &mut PlacementPolicy, device: usize) {
    policy.weights_store = retarget_store(policy.weights_store, device);
    policy.spill_store = retarget_store(policy.spill_store, device);
    policy.kv_store = retarget_store(policy.kv_store, device);
    policy.weights_compute = retarget_compute(policy.weights_compute, device);
    policy.spill_compute = retarget_compute(policy.spill_compute, device);
    policy.kv_compute = retarget_compute(policy.kv_compute, device);
    for rule in &mut policy.rules {
        rule.store = rule.store.map(|store| retarget_store(store, device));
        rule.compute = rule
            .compute
            .map(|compute| retarget_compute(compute, device));
    }
}

fn retarget_store(store: StoragePlacement, device: usize) -> StoragePlacement {
    match store {
        StoragePlacement::Vram { .. } => StoragePlacement::Vram { device },
        other => other,
    }
}

fn retarget_compute(compute: ComputePlacement, device: usize) -> ComputePlacement {
    match compute {
        ComputePlacement::Cuda { .. } => ComputePlacement::Cuda { device },
        other => other,
    }
}

fn apply_linear_layout_section(
    policy: &mut LinearLayoutPolicy,
    layout: LinearLayoutSection,
) -> Result<()> {
    if let Some(mode) = layout.mode.as_deref() {
        let choice = LinearLayoutChoice::parse(mode)?;
        policy.cpu = choice;
        policy.cuda = choice;
    }
    if let Some(cpu) = layout.cpu.as_deref() {
        policy.cpu = LinearLayoutChoice::parse(cpu)?;
    }
    if let Some(cuda) = layout.cuda.as_deref() {
        policy.cuda = LinearLayoutChoice::parse(cuda)?;
    }
    if let Some(materialize) = layout.materialize.as_deref() {
        policy.materialization = MaterializationPolicy::parse(materialize)?;
    }
    if let Some(value) = layout.max_extra_memory {
        policy.max_extra_memory_bytes = parse_optional_bytes(value)?;
    }
    Ok(())
}

fn parse_optional_bytes(value: serde_json::Value) -> Result<Option<u64>> {
    match value {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::String(value) if value.eq_ignore_ascii_case("auto") => Ok(None),
        serde_json::Value::String(value) => parse_bytes_string(&value).map(Some),
        serde_json::Value::Number(value) => value.as_u64().map(Some).ok_or_else(|| {
            AegisError::InvalidConfig("max-extra-memory must be a positive integer".into())
        }),
        other => Err(AegisError::InvalidConfig(format!(
            "unsupported max-extra-memory value `{other}`"
        ))),
    }
}

fn parse_bytes_string(value: &str) -> Result<u64> {
    let raw = value.trim().to_ascii_lowercase();
    let (digits, multiplier) = if let Some(number) = raw.strip_suffix("gib") {
        (number.trim(), 1024_u64.pow(3))
    } else if let Some(number) = raw.strip_suffix("gb") {
        (number.trim(), 1_000_000_000)
    } else if let Some(number) = raw.strip_suffix("mib") {
        (number.trim(), 1024_u64.pow(2))
    } else if let Some(number) = raw.strip_suffix("mb") {
        (number.trim(), 1_000_000)
    } else {
        (raw.as_str(), 1)
    };
    let value = digits
        .parse::<u64>()
        .map_err(|_| AegisError::InvalidConfig(format!("invalid byte value `{}`", value.trim())))?;
    value
        .checked_mul(multiplier)
        .ok_or_else(|| AegisError::InvalidConfig(format!("byte value `{}` overflows", value)))
}

pub fn parse_storage(value: &str, default_device: usize) -> Result<StoragePlacement> {
    match value.to_ascii_lowercase().as_str() {
        "ram" => Ok(StoragePlacement::Ram),
        "mmap" => Ok(StoragePlacement::Mmap),
        "vram" | "gpu" => Ok(StoragePlacement::Vram {
            device: default_device,
        }),
        other if other.starts_with("vram:") => Ok(StoragePlacement::Vram {
            device: other
                .trim_start_matches("vram:")
                .parse::<usize>()
                .map_err(|_| AegisError::InvalidConfig(format!("invalid storage `{value}`")))?,
        }),
        _ => Err(AegisError::InvalidConfig(format!(
            "unsupported storage placement `{value}`"
        ))),
    }
}

pub fn parse_compute(value: &str, default_device: usize) -> Result<ComputePlacement> {
    match value.to_ascii_lowercase().as_str() {
        "cpu" => Ok(ComputePlacement::Cpu),
        "cuda" | "gpu" => Ok(ComputePlacement::Cuda {
            device: default_device,
        }),
        other if other.starts_with("cuda:") => Ok(ComputePlacement::Cuda {
            device: other
                .trim_start_matches("cuda:")
                .parse::<usize>()
                .map_err(|_| AegisError::InvalidConfig(format!("invalid compute `{value}`")))?,
        }),
        _ => Err(AegisError::InvalidConfig(format!(
            "unsupported compute placement `{value}`"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::HardwareInventory;

    #[test]
    fn cuda_runtime_flags_come_from_parameters() {
        let params: ParametersFile = serde_json::from_value(serde_json::json!({
            "model": {
                "path": "/tmp/model",
                "store": "vram",
                "compute": "cuda"
            },
            "cuda": {
                "device": 2,
                "native-mxfp4-repack": true,
                "native-mxfp4-inference": false,
                "prefill-attention": "warp-flash"
            }
        }))
        .expect("parameters should parse");

        let fragment = params
            .into_engine_fragment(PlacementPolicy::auto_for(&HardwareInventory::detect()))
            .expect("parameters should become an engine fragment");

        assert_eq!(
            fragment.policy.weights_store,
            StoragePlacement::Vram { device: 2 }
        );
        assert_eq!(
            fragment.policy.weights_compute,
            ComputePlacement::Cuda { device: 2 }
        );
        assert!(fragment.cuda.native_mxfp4_repack);
        assert!(!fragment.cuda.native_mxfp4_inference);
        assert_eq!(
            fragment.cuda.prefill_attention,
            CudaPrefillAttentionKernel::WarpFlash
        );
    }

    #[test]
    fn legacy_flash_attention_flag_controls_cuda_prefill_attention() {
        let params: ParametersFile = serde_json::from_value(serde_json::json!({
            "model": {
                "path": "/tmp/model"
            },
            "other-parameters": {
                "flash-attention": false
            }
        }))
        .expect("parameters should parse");

        let fragment = params
            .into_engine_fragment(PlacementPolicy::auto_for(&HardwareInventory::detect()))
            .expect("parameters should become an engine fragment");

        assert_eq!(
            fragment.cuda.prefill_attention,
            CudaPrefillAttentionKernel::Reference
        );
    }

    #[test]
    fn explicit_cuda_prefill_attention_wins_over_legacy_flash_attention_flag() {
        let params: ParametersFile = serde_json::from_value(serde_json::json!({
            "model": {
                "path": "/tmp/model"
            },
            "cuda": {
                "prefill-attention": "reference"
            },
            "other-parameters": {
                "flash-attention": true
            }
        }))
        .expect("parameters should parse");

        let fragment = params
            .into_engine_fragment(PlacementPolicy::auto_for(&HardwareInventory::detect()))
            .expect("parameters should become an engine fragment");

        assert_eq!(
            fragment.cuda.prefill_attention,
            CudaPrefillAttentionKernel::Reference
        );
    }

    #[test]
    fn model_mmap_false_uses_ram_when_store_is_not_explicit() {
        let params: ParametersFile = serde_json::from_value(serde_json::json!({
            "model": {
                "path": "/tmp/model",
                "mmap": false
            }
        }))
        .expect("parameters should parse");

        let mut policy = PlacementPolicy::auto_for(&HardwareInventory {
            cpu: crate::hardware::CpuInfo {
                model_name: "test-cpu".into(),
                physical_cores: 1,
                logical_threads: 1,
                ram_total_bytes: 8 * 1024 * 1024 * 1024,
                ram_available_bytes: Some(8 * 1024 * 1024 * 1024),
                avx2: false,
                avx512: false,
                bf16: false,
            },
            gpus: Vec::new(),
        });
        policy.weights_store = StoragePlacement::Mmap;

        let fragment = params
            .into_engine_fragment(policy)
            .expect("parameters should become an engine fragment");

        assert_eq!(fragment.policy.weights_store, StoragePlacement::Ram);
        assert_eq!(fragment.policy.spill_store, StoragePlacement::Ram);
    }

    #[test]
    fn cuda_device_retargets_auto_cuda_policy() {
        let params: ParametersFile = serde_json::from_value(serde_json::json!({
            "model": {
                "path": "/tmp/model"
            },
            "cuda": {
                "device": 2
            }
        }))
        .expect("parameters should parse");

        let mut policy = PlacementPolicy::auto_for(&HardwareInventory {
            cpu: crate::hardware::CpuInfo {
                model_name: "test-cpu".into(),
                physical_cores: 1,
                logical_threads: 1,
                ram_total_bytes: 8 * 1024 * 1024 * 1024,
                ram_available_bytes: Some(8 * 1024 * 1024 * 1024),
                avx2: false,
                avx512: false,
                bf16: false,
            },
            gpus: Vec::new(),
        });
        policy.weights_store = StoragePlacement::Vram { device: 0 };
        policy.weights_compute = ComputePlacement::Cuda { device: 0 };
        policy.kv_store = StoragePlacement::Vram { device: 0 };
        policy.kv_compute = ComputePlacement::Cuda { device: 0 };

        let fragment = params
            .into_engine_fragment(policy)
            .expect("parameters should become an engine fragment");

        assert_eq!(
            fragment.policy.weights_store,
            StoragePlacement::Vram { device: 2 }
        );
        assert_eq!(
            fragment.policy.weights_compute,
            ComputePlacement::Cuda { device: 2 }
        );
        assert_eq!(
            fragment.policy.kv_store,
            StoragePlacement::Vram { device: 2 }
        );
        assert_eq!(
            fragment.policy.kv_compute,
            ComputePlacement::Cuda { device: 2 }
        );
    }
}

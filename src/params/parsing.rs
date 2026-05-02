use std::fs;
use std::path::Path;

use crate::cuda::{CudaPrefillAttentionKernel, CudaRuntimeConfig};
use crate::error::{AegisError, Result};
use crate::generation::SamplingConfig;
use crate::planning::placement::{
    ComputePlacement, LayerSelector, PlacementPolicy, PlacementRule, StoragePlacement,
};
use crate::tensor::layout::{LinearLayoutChoice, LinearLayoutPolicy, MaterializationPolicy};
use crate::tensor::quant::KvCacheQuantization;

use super::file::*;
use super::runtime::{EngineConfigFragment, ServeConfig};

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

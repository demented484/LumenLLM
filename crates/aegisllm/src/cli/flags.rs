use std::path::PathBuf;

use aegisllm_cuda::cuda::{CudaPrefillAttentionKernel, CudaRuntimeConfig};
use crate::engine::EngineConfig;
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::generation::SamplingConfig;
use aegisllm_base::hardware::HardwareInventory;
use crate::params::ParametersFile;
use aegisllm_base::planning::placement::{
    ComputePlacement, LayerSelector, PlacementPolicy, PlacementRule, StoragePlacement,
};
use aegisllm_base::tensor::layout::{LinearLayoutChoice, MaterializationPolicy};
use aegisllm_base::tensor::quant::KvCacheQuantization;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ParsedEngineFlags {
    pub(super) model_path: PathBuf,
    pub(super) policy: PlacementPolicy,
    pub(super) cuda: CudaRuntimeConfig,
    pub(super) generation: SamplingConfig,
}

impl ParsedEngineFlags {
    pub(super) fn engine_config(self, enable_executor: bool) -> EngineConfig {
        EngineConfig {
            model_path: self.model_path,
            policy: self.policy,
            enable_executor,
            cuda: self.cuda,
        }
    }
}

pub(super) fn parse_engine_flags(args: &[String]) -> Result<ParsedEngineFlags> {
    let mut model_path = None;
    let mut loaded_policy = None;
    let mut loaded_generation = None;
    let mut ctx_size = 8192;
    let mut ctx_size_explicit = false;
    let mut cuda_device = 0;
    let mut cuda_device_explicit = false;
    let mut n_gpu_layers: Option<usize> = None;
    let mut weights_store: Option<String> = None;
    let mut weights_compute: Option<String> = None;
    let mut kv_store: Option<String> = None;
    let mut kv_compute: Option<String> = None;
    let mut kv_quantization = KvCacheQuantization::F16;
    let mut kv_quantization_explicit = false;
    let mut linear_layout = None;
    let mut cpu_linear_layout = None;
    let mut cuda_linear_layout = None;
    let mut linear_materialize = None;
    let mut cuda_runtime = CudaRuntimeConfig::from_env();

    let mut i = 0;
    while i < args.len() {
        let flag = &args[i];
        match flag.as_str() {
            "--config" | "--parameters" => {
                let path = take_value(args, &mut i, flag)?;
                let inventory = HardwareInventory::detect();
                let fragment = ParametersFile::from_path(&path)?
                    .into_engine_fragment(PlacementPolicy::auto_for(&inventory))?;
                cuda_device = cuda_device_from_policy(&fragment.policy).unwrap_or(cuda_device);
                model_path = Some(fragment.model_path);
                loaded_policy = Some(fragment.policy);
                loaded_generation = Some(fragment.generation);
                cuda_runtime = fragment.cuda;
            }
            "--model" => model_path = Some(take_value(args, &mut i, flag)?.into()),
            "--ctx-size" => {
                ctx_size = parse_value(args, &mut i, flag)?;
                ctx_size_explicit = true;
            }
            "--cuda-device" => {
                cuda_device = parse_value(args, &mut i, flag)?;
                cuda_device_explicit = true;
            }
            "--n-gpu-layers" => n_gpu_layers = Some(parse_value(args, &mut i, flag)?),
            "--weights-store" => weights_store = Some(take_value(args, &mut i, flag)?),
            "--weights-compute" => weights_compute = Some(take_value(args, &mut i, flag)?),
            "--kv-store" => kv_store = Some(take_value(args, &mut i, flag)?),
            "--kv-compute" => kv_compute = Some(take_value(args, &mut i, flag)?),
            "--cache-type-k" | "--cache-type-v" | "--kv-quant" => {
                let value = take_value(args, &mut i, flag)?;
                kv_quantization = KvCacheQuantization::parse(&value).ok_or_else(|| {
                    AegisError::InvalidConfig(format!(
                        "unsupported kv cache quantization `{value}`"
                    ))
                })?;
                kv_quantization_explicit = true;
            }
            "--linear-layout" => {
                linear_layout = Some(LinearLayoutChoice::parse(&take_value(args, &mut i, flag)?)?)
            }
            "--cpu-linear-layout" => {
                cpu_linear_layout =
                    Some(LinearLayoutChoice::parse(&take_value(args, &mut i, flag)?)?)
            }
            "--cuda-linear-layout" => {
                cuda_linear_layout =
                    Some(LinearLayoutChoice::parse(&take_value(args, &mut i, flag)?)?)
            }
            "--linear-materialize" => {
                linear_materialize = Some(MaterializationPolicy::parse(&take_value(
                    args, &mut i, flag,
                )?)?)
            }
            "--native-mxfp4-repack" => cuda_runtime.native_mxfp4_repack = true,
            "--cutlass-nvfp4-repack" => cuda_runtime.cutlass_nvfp4_repack = true,
            "--native-mxfp4-inference" => cuda_runtime.native_mxfp4_inference = true,
            "--cuda-stage-timings" => cuda_runtime.prefill_stage_timings = true,
            "--cuda-prefill-attention" => {
                cuda_runtime.prefill_attention =
                    CudaPrefillAttentionKernel::parse(&take_value(args, &mut i, flag)?)?
            }
            "--cuda-prefill-chunk-size" => {
                let value = take_value(args, &mut i, flag)?;
                cuda_runtime.prefill_chunk_size = Some(value.parse::<usize>().map_err(|error| {
                    AegisError::InvalidConfig(format!(
                        "bad --cuda-prefill-chunk-size `{value}`: {error}"
                    ))
                })?)
            }
            other => {
                return Err(AegisError::InvalidConfig(format!(
                    "unknown engine flag `{other}`"
                )));
            }
        }
        i += 1;
    }

    let model_path: PathBuf = model_path.ok_or_else(|| {
        AegisError::InvalidConfig(
            "engine command requires --model <path> or --config <path>".into(),
        )
    })?;
    let inventory = HardwareInventory::detect();
    let loaded_from_config = loaded_policy.is_some();
    let mut policy = loaded_policy.unwrap_or_else(|| PlacementPolicy::auto_for(&inventory));
    if cuda_device_explicit && !loaded_from_config {
        force_cuda_policy(&mut policy, cuda_device);
    }
    if cuda_device_explicit {
        crate::params::retarget_cuda_policy(&mut policy, cuda_device);
    }
    if !loaded_from_config || ctx_size_explicit {
        policy.context_size = ctx_size;
    }
    if !loaded_from_config || kv_quantization_explicit {
        policy.kv_quantization = kv_quantization;
    }
    if let Some(choice) = linear_layout {
        policy.linear_layout.cpu = choice;
        policy.linear_layout.cuda = choice;
    }
    if let Some(choice) = cpu_linear_layout {
        policy.linear_layout.cpu = choice;
    }
    if let Some(choice) = cuda_linear_layout {
        policy.linear_layout.cuda = choice;
    }
    if let Some(materialization) = linear_materialize {
        policy.linear_layout.materialization = materialization;
    }
    if let Some(store) = weights_store.as_deref() {
        policy.weights_store = crate::params::parse_storage(store, cuda_device)?;
    }
    if let Some(compute) = weights_compute.as_deref() {
        let compute = crate::params::parse_compute(compute, cuda_device)?;
        policy.weights_compute = compute;
        policy.spill_compute = compute;
    }
    policy.spill_store = StoragePlacement::Mmap;
    if let Some(store) = kv_store.as_deref() {
        policy.kv_store = crate::params::parse_storage(store, cuda_device)?;
    }
    if let Some(compute) = kv_compute.as_deref() {
        policy.kv_compute = crate::params::parse_compute(compute, cuda_device)?;
    }
    let explicit_global_weight_placement = weights_store.is_some() || weights_compute.is_some();
    if let Some(n) = n_gpu_layers {
        policy.rules.clear();
        if n > 0 {
            policy.rules.push(PlacementRule {
                selector: LayerSelector::FirstN { n },
                store: Some(StoragePlacement::Vram {
                    device: cuda_device,
                }),
                compute: Some(ComputePlacement::Cuda {
                    device: cuda_device,
                }),
            });
        }
    } else if explicit_global_weight_placement {
        policy.rules.clear();
    }

    Ok(ParsedEngineFlags {
        model_path,
        policy,
        cuda: cuda_runtime,
        generation: loaded_generation.unwrap_or_default(),
    })
}

fn force_cuda_policy(policy: &mut PlacementPolicy, device: usize) {
    policy.weights_store = StoragePlacement::Vram { device };
    policy.weights_compute = ComputePlacement::Cuda { device };
    policy.spill_compute = ComputePlacement::Cuda { device };
    policy.kv_store = StoragePlacement::Vram { device };
    policy.kv_compute = ComputePlacement::Cuda { device };
    if policy.reserve_vram_bytes == 0 {
        policy.reserve_vram_bytes = 1024 * 1024 * 1024;
    }
}

pub(super) fn is_engine_flag(flag: &str) -> bool {
    matches!(
        flag,
        "--model"
            | "--config"
            | "--parameters"
            | "--ctx-size"
            | "--cuda-device"
            | "--n-gpu-layers"
            | "--weights-store"
            | "--weights-compute"
            | "--kv-store"
            | "--kv-compute"
            | "--cache-type-k"
            | "--cache-type-v"
            | "--kv-quant"
            | "--linear-layout"
            | "--cpu-linear-layout"
            | "--cuda-linear-layout"
            | "--linear-materialize"
            | "--native-mxfp4-repack"
            | "--cutlass-nvfp4-repack"
            | "--native-mxfp4-inference"
            | "--cuda-stage-timings"
            | "--cuda-prefill-attention"
            | "--cuda-prefill-chunk-size"
    )
}

pub(super) fn flag_takes_value(flag: &str) -> bool {
    is_engine_flag(flag)
        && !matches!(
            flag,
            "--native-mxfp4-repack"
                | "--cutlass-nvfp4-repack"
                | "--native-mxfp4-inference"
                | "--cuda-stage-timings"
        )
}

pub(super) fn take_value(args: &[String], i: &mut usize, flag: &str) -> Result<String> {
    *i += 1;
    args.get(*i)
        .cloned()
        .ok_or_else(|| AegisError::InvalidConfig(format!("missing value for {flag}")))
}

pub(super) fn parse_value<T: std::str::FromStr>(
    args: &[String],
    i: &mut usize,
    flag: &str,
) -> Result<T> {
    let raw = take_value(args, i, flag)?;
    raw.parse::<T>()
        .map_err(|_| AegisError::InvalidConfig(format!("invalid value `{raw}` for {flag}")))
}

fn cuda_device_from_policy(policy: &PlacementPolicy) -> Option<usize> {
    storage_cuda_device(policy.weights_store)
        .or_else(|| compute_cuda_device(policy.weights_compute))
        .or_else(|| storage_cuda_device(policy.kv_store))
        .or_else(|| compute_cuda_device(policy.kv_compute))
        .or_else(|| {
            policy.rules.iter().find_map(|rule| {
                rule.store
                    .and_then(storage_cuda_device)
                    .or_else(|| rule.compute.and_then(compute_cuda_device))
            })
        })
}

fn storage_cuda_device(storage: StoragePlacement) -> Option<usize> {
    match storage {
        StoragePlacement::Vram { device } => Some(device),
        StoragePlacement::Ram | StoragePlacement::Mmap => None,
    }
}

fn compute_cuda_device(compute: ComputePlacement) -> Option<usize> {
    match compute {
        ComputePlacement::Cuda { device } => Some(device),
        ComputePlacement::Cpu | ComputePlacement::Wgpu { .. } => None,
    }
}

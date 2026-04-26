use super::helpers::{
    deterministic_input, find_cuda_linear, first_cuda_nvfp4_region, resident_layout_for_region,
};
use crate::cuda::CUDA_PREFILL_CHUNK_MAX;
use crate::engine::quality::run_quality_smoke;
use crate::engine::{AegisEngine, EngineConfig};
use crate::error::{AegisError, Result};
use crate::executor::readiness_for_plan;
use crate::generation::{GenerateOutput, GenerateRequest, SamplingConfig};
use crate::graph::TensorRole;
use crate::hardware::HardwareInventory;
use crate::planning::materialization::LinearMaterializationCache;
use crate::planning::placement::{ComputePlacement, StoragePlacement};
use crate::tensor::storage::{HostTensorStorage, TensorResidencyPlan, TensorStorageLoader};

pub(super) fn inspect_hardware() {
    let inventory = HardwareInventory::detect();
    println!(
        "cpu: {} physical_cores={} logical_threads={} ram_total={} ram_available={}",
        inventory.cpu.model_name,
        inventory.cpu.physical_cores,
        inventory.cpu.logical_threads,
        inventory.cpu.ram_total_bytes,
        inventory
            .cpu
            .ram_available_bytes
            .map(|value| value.to_string())
            .unwrap_or_else(|| "?".into())
    );
    for gpu in inventory.gpus {
        println!(
            "gpu: index={} name={} arch={:?} cc={} vram_total={} vram_free={} fp4={} fp8={}",
            gpu.index,
            gpu.name,
            gpu.architecture,
            gpu.compute_capability.as_deref().unwrap_or("?"),
            gpu.vram_total_bytes,
            gpu.vram_free_bytes
                .map(|value| value.to_string())
                .unwrap_or_else(|| "?".into()),
            gpu.supports_fp4(),
            gpu.supports_fp8()
        );
    }
}

pub(super) fn mvp_check(config: EngineConfig) -> Result<()> {
    let executor_config = EngineConfig {
        enable_executor: true,
        ..config.clone()
    };
    let engine = AegisEngine::build(EngineConfig {
        enable_executor: false,
        ..config
    })?;
    let readiness = readiness_for_plan(&engine.placement, &engine.runtime);
    println!(
        "mvp-check: runnable={} selected={} planned_cpu_regions={} planned_cuda_regions={}",
        readiness.runnable,
        readiness.selected_backend,
        readiness.planned_cpu_regions,
        readiness.planned_cuda_regions,
    );
    for limitation in &readiness.limitations {
        println!("mvp-limitation: {limitation}");
    }
    if !readiness.runnable {
        return Err(AegisError::Unsupported(format!(
            "mvp-check failed: {}",
            readiness.limitations.join("; ")
        )));
    }
    let executor_probe = AegisEngine::build(executor_config)?;
    executor_probe.probe_executor()?;
    let info = executor_probe.executor_info().ok_or_else(|| {
        AegisError::InvalidPlan("mvp-check executor probe missing executor".into())
    })?;
    println!(
        "mvp-check-executor: build=ok backend={} capabilities={}",
        info.name,
        info.capabilities.len()
    );
    Ok(())
}

pub(super) fn quality_smoke(config: EngineConfig) -> Result<()> {
    for result in run_quality_smoke(config)? {
        println!(
            "quality-smoke: case={} finish={} prompt_tokens={} completion_tokens={} text={:?}",
            result.case.name,
            result.output.finish_reason,
            result.output.prompt_tokens,
            result.output.completion_tokens,
            result.output.text
        );
    }
    Ok(())
}

pub(super) fn cuda_prefill_compare(config: EngineConfig) -> Result<()> {
    let configured_chunk = config
        .cuda
        .prefill_chunk_size
        .unwrap_or(128)
        .clamp(1, CUDA_PREFILL_CHUNK_MAX);
    cuda_prefill_compare_one_chunk(config, configured_chunk)
}

pub(super) fn cuda_prefill_sweep(config: EngineConfig) -> Result<()> {
    let chunks = [
        1,
        2,
        3,
        7,
        8,
        16,
        31,
        32,
        64,
        128,
        512,
        2048,
        CUDA_PREFILL_CHUNK_MAX,
    ];
    for chunk in chunks {
        let mut chunk_config = config.clone();
        chunk_config.cuda.prefill_chunk_size = Some(chunk);
        cuda_prefill_compare_one_chunk(chunk_config, chunk)?;
    }
    Ok(())
}

fn cuda_prefill_compare_one_chunk(config: EngineConfig, configured_chunk: usize) -> Result<()> {
    let mut token_config = config.clone();
    token_config.cuda.prefill_chunk_size = Some(1);

    let prompts = [
        "The capital of France is",
        "Привет, кратко объясни что такое трансформер.",
    ];
    let requests = prompts
        .iter()
        .map(|prompt| GenerateRequest {
            prompt: (*prompt).into(),
            max_tokens: 1,
            sampling: SamplingConfig {
                temperature: 0.0,
                top_k: 1,
                top_p: 1.0,
            },
        })
        .collect::<Vec<_>>();

    let token_outputs = {
        let token_engine = AegisEngine::build(token_config)?;
        requests
            .iter()
            .map(|request| token_engine.generate(request.clone()))
            .collect::<Result<Vec<_>>>()?
    };
    let (chunk_outputs, attention_requested, attention_logical, attention_effective) = {
        let chunk_engine = AegisEngine::build(config)?;
        let outputs = requests
            .iter()
            .map(|request| chunk_engine.generate(request.clone()))
            .collect::<Result<Vec<_>>>()?;
        let compute_capability = chunk_engine
            .inventory
            .gpus
            .first()
            .and_then(|gpu| gpu.compute_capability.as_deref());
        let selection_context_tokens = outputs
            .iter()
            .map(|output| output.prompt_tokens)
            .max()
            .unwrap_or(0);
        let selection = chunk_engine.cuda.prefill_attention_selection(
            compute_capability,
            selection_context_tokens,
            chunk_engine.graph.head_dim,
        );
        (
            outputs,
            selection.requested.canonical_name(),
            selection.logical_backend.canonical_name(),
            selection.effective_path.canonical_name(),
        )
    };

    for ((prompt, token_output), chunk_output) in prompts
        .iter()
        .zip(token_outputs.iter())
        .zip(chunk_outputs.iter())
    {
        ensure_prefill_match(prompt, token_output, chunk_output)?;
        println!(
            "cuda-prefill-compare: chunk_size={} requested={} logical_backend={} effective_path={} prompt_tokens={} completion_tokens={} text={:?}",
            configured_chunk,
            attention_requested,
            attention_logical,
            attention_effective,
            chunk_output.prompt_tokens,
            chunk_output.completion_tokens,
            chunk_output.text
        );
    }
    Ok(())
}

fn ensure_prefill_match(
    prompt: &str,
    token_output: &GenerateOutput,
    chunk_output: &GenerateOutput,
) -> Result<()> {
    if token_output.text != chunk_output.text
        || token_output.finish_reason != chunk_output.finish_reason
        || token_output.completion_tokens != chunk_output.completion_tokens
    {
        return Err(AegisError::InvalidPlan(format!(
            "chunked CUDA prefill diverged from token-by-token for prompt {:?}: token text={:?} finish={} completion={}, chunk text={:?} finish={} completion={}",
            prompt,
            token_output.text,
            token_output.finish_reason,
            token_output.completion_tokens,
            chunk_output.text,
            chunk_output.finish_reason,
            chunk_output.completion_tokens
        )));
    }
    Ok(())
}

pub(super) fn storage_smoke(config: EngineConfig) -> Result<()> {
    let engine = AegisEngine::build(EngineConfig {
        enable_executor: false,
        ..config
    })?;
    let mut selected = Vec::new();
    for tensor in &engine.storage.tensors {
        if let Some(index) =
            selected
                .iter()
                .position(|existing: &&crate::tensor::storage::TensorStoragePlan| {
                    existing.residency == tensor.residency
                })
        {
            if tensor.bytes < selected[index].bytes {
                selected[index] = tensor;
            }
        } else {
            selected.push(tensor);
        }
    }
    let mut loader = TensorStorageLoader::new();
    for tensor_plan in selected {
        let tensor = engine
            .artifact
            .tensors
            .get(&tensor_plan.name)
            .ok_or_else(|| {
                AegisError::InvalidPlan(format!(
                    "storage plan references missing tensor `{}`",
                    tensor_plan.name
                ))
            })?;
        match tensor_plan.store {
            StoragePlacement::Vram { device } => {
                println!(
                    "storage-smoke: tensor={} residency={:?} store=vram:{} compute={} bytes={} host_load=deferred",
                    tensor_plan.name,
                    tensor_plan.residency,
                    device,
                    tensor_plan.compute,
                    tensor_plan.bytes
                );
            }
            StoragePlacement::Ram | StoragePlacement::Mmap => {
                let loaded = loader.load_for_store(tensor, tensor_plan.store)?;
                let source = match &loaded.storage {
                    HostTensorStorage::Ram(_) => "ram",
                    HostTensorStorage::Mmap { .. } => "mmap",
                };
                println!(
                    "storage-smoke: tensor={} residency={:?} store={} compute={} bytes={} host_load={} loaded_bytes={}",
                    tensor_plan.name,
                    tensor_plan.residency,
                    tensor_plan.store,
                    tensor_plan.compute,
                    tensor_plan.bytes,
                    source,
                    loaded.len()
                );
            }
        }
    }
    Ok(())
}

fn first_cpu_nvfp4_region<'a>(
    engine: &'a AegisEngine,
    command_name: &'static str,
) -> Result<(
    &'a crate::graph::GraphRegion,
    &'a crate::planning::placement::RegionPlacement,
)> {
    let region_placements = engine.placement.region_map();
    let region = engine
        .graph
        .regions
        .iter()
        .find(|region| {
            let Some(placement) = region_placements.get(&region.id) else {
                return false;
            };
            matches!(placement.compute, ComputePlacement::Cpu)
                && region.tensors.iter().any(|tensor| {
                    matches!(
                        tensor.role,
                        TensorRole::Query
                            | TensorRole::Key
                            | TensorRole::Value
                            | TensorRole::Output
                            | TensorRole::Gate
                            | TensorRole::Up
                            | TensorRole::Down
                    ) && tensor.info.dtype == crate::tensor::TensorDType::U8
                })
        })
        .ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "{command_name} needs at least one cpu-computed NVFP4 linear region"
            ))
        })?;
    let placement = region_placements.get(&region.id).ok_or_else(|| {
        AegisError::InvalidPlan(format!("missing placement for region `{}`", region.id.0))
    })?;
    Ok((region, placement))
}

pub(super) fn cpu_smoke(config: EngineConfig) -> Result<()> {
    let engine = AegisEngine::build(EngineConfig {
        enable_executor: false,
        ..config
    })?;
    let (region, placement) = first_cpu_nvfp4_region(&engine, "cpu-smoke")?;
    let resident_layout = resident_layout_for_region(&engine, &region.id);
    let mut materialization = LinearMaterializationCache::new();
    let linear = materialization
        .load_first_cpu_region_nvfp4_linear(
            &engine.artifact,
            region,
            placement,
            resident_layout,
            engine.placement.linear_layout.materialization,
        )?
        .ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "no NVFP4 linear tensor found in region `{}`",
                region.id.0
            ))
        })?;
    let input = deterministic_input(linear.cols);
    let output = linear.matvec(&input)?;
    let checksum: f32 = output.iter().take(32).copied().sum();
    let max_abs = output.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
    let materialization_stats = materialization.stats();
    println!(
        "cpu-smoke: linear={} rows={} cols={} packed_bytes={} scale_bytes={} input_scale={} output_scale={} store={} residency={:?} layout={} materialize={} cache_entries={} cache_hits={} cache_misses={} materialized_extra_bytes={} output_len={} checksum32={:.6} first={:.6} max_abs={:.6}",
        linear.name,
        linear.rows,
        linear.cols,
        linear.packed_bytes,
        linear.scale_bytes,
        linear.input_scale,
        linear.output_scale,
        linear.store,
        linear.residency,
        linear.resident_layout,
        engine.placement.linear_layout.materialization,
        materialization_stats.entries,
        materialization_stats.hits,
        materialization_stats.misses,
        materialization_stats.materialized_extra_bytes,
        output.len(),
        checksum,
        output.first().copied().unwrap_or(0.0),
        max_abs
    );
    Ok(())
}

pub(super) fn cpu_materialize_smoke(config: EngineConfig) -> Result<()> {
    let engine = AegisEngine::build(EngineConfig {
        enable_executor: false,
        ..config
    })?;
    let (region, placement) = first_cpu_nvfp4_region(&engine, "cpu-materialize-smoke")?;
    let resident_layout = resident_layout_for_region(&engine, &region.id);
    let mut materialization = LinearMaterializationCache::new();
    let first = materialization
        .load_first_cpu_region_nvfp4_linear(
            &engine.artifact,
            region,
            placement,
            resident_layout,
            engine.placement.linear_layout.materialization,
        )?
        .ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "no NVFP4 linear tensor found in region `{}`",
                region.id.0
            ))
        })?;
    let second = materialization
        .load_first_cpu_region_nvfp4_linear(
            &engine.artifact,
            region,
            placement,
            resident_layout,
            engine.placement.linear_layout.materialization,
        )?
        .ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "no NVFP4 linear tensor found in region `{}`",
                region.id.0
            ))
        })?;
    let input = deterministic_input(second.cols);
    let output = second.matvec(&input)?;
    let checksum: f32 = output.iter().take(32).copied().sum();
    let stats = materialization.stats();
    println!(
        "cpu-materialize-smoke: linear={} layout={} materialize={} entries={} hits={} misses={} uncached={} materialized_extra_bytes={} first_load_extra={} second_checksum32={:.6}",
        second.name,
        second.resident_layout,
        engine.placement.linear_layout.materialization,
        stats.entries,
        stats.hits,
        stats.misses,
        stats.uncached,
        stats.materialized_extra_bytes,
        first.materialized_extra_bytes(),
        checksum,
    );
    Ok(())
}

pub(super) fn cuda_smoke(config: EngineConfig) -> Result<()> {
    let cuda_config = config.cuda;
    let engine = AegisEngine::build(EngineConfig {
        enable_executor: false,
        ..config
    })?;
    let device = first_cuda_device(&engine, "cuda-smoke")?;
    let cuda = crate::cuda::CudaRuntime::new_with_config(device, cuda_config)?;
    let cuda_weights = cuda.weight_loader();
    let region_placements = engine.placement.region_map();
    let region = engine
        .graph
        .regions
        .iter()
        .find(|region| {
            let Some(placement) = region_placements.get(&region.id) else {
                return false;
            };
            matches!(
                placement.compute,
                ComputePlacement::Cuda {
                    device: compute_device
                } if compute_device == device
            ) && region.tensors.iter().any(|tensor| {
                matches!(
                    tensor.role,
                    TensorRole::Query
                        | TensorRole::Key
                        | TensorRole::Value
                        | TensorRole::Output
                        | TensorRole::Gate
                        | TensorRole::Up
                        | TensorRole::Down
                ) && tensor.info.dtype == crate::tensor::TensorDType::U8
            })
        })
        .ok_or_else(|| {
            AegisError::InvalidPlan("no NVFP4 linear region with compute=cuda found".into())
        })?;
    let placement = region_placements.get(&region.id).ok_or_else(|| {
        AegisError::InvalidPlan(format!("missing placement for region `{}`", region.id.0))
    })?;
    let resident_layout = resident_layout_for_region(&engine, &region.id);
    let linears = cuda_weights.load_placed_region_nvfp4_linears_with_layout(
        &engine.artifact,
        region,
        placement,
        resident_layout,
    )?;
    let mut launch_abi = "none";
    if let Some(first) = linears.first() {
        if first.kernel_family == crate::planning::runtime::KernelFamily::CudaNativeFp4TensorCores {
            cuda.probe_blackwell_nvfp4_linear_abi(first)?;
            launch_abi = "native-probe-ok";
        } else {
            let input = deterministic_input(first.cols);
            let _ = cuda.matvec_nvfp4_reference_host(first, &input)?;
            launch_abi = "reference-matvec-ok";
        }
    }
    for linear in &linears {
        println!(
            "cuda-smoke: device={} linear={} rows={} cols={} packed_bytes={} scale_bytes={} native_mxfp4_bytes={} native_mxfp4_blocks_per_row={} input_scale={} output_scale={} family={:?} residency={:?} layout={} launch_abi={}",
            cuda.device_index(),
            linear.name,
            linear.rows,
            linear.cols,
            linear.packed_bytes,
            linear.scale_bytes,
            linear.native_mxfp4_bytes(),
            linear.native_mxfp4_blocks_per_row(),
            linear.input_scale,
            linear.output_scale,
            linear.kernel_family,
            linear.residency,
            linear.resident_layout,
            launch_abi
        );
    }
    Ok(())
}

fn first_cuda_device(engine: &AegisEngine, command_name: &'static str) -> Result<usize> {
    match engine
        .placement
        .region_placements
        .iter()
        .find_map(|region| {
            if matches!(region.compute, ComputePlacement::Cuda { .. }) {
                Some(region.compute)
            } else {
                None
            }
        }) {
        Some(ComputePlacement::Cuda { device }) => Ok(device),
        _ => Err(AegisError::InvalidPlan(format!(
            "{command_name} needs at least one cuda-computed region"
        ))),
    }
}

pub(super) fn cuda_dense_smoke(config: EngineConfig) -> Result<()> {
    let cuda_config = config.cuda;
    let engine = AegisEngine::build(EngineConfig {
        enable_executor: false,
        ..config
    })?;
    let device = first_cuda_device(&engine, "cuda-dense-smoke")?;
    let region_placements = engine.placement.region_map();
    let (region, tensor) = engine
        .graph
        .regions
        .iter()
        .find_map(|region| {
            let placement = region_placements.get(&region.id)?;
            if !matches!(
                placement.compute,
                ComputePlacement::Cuda {
                    device: compute_device
                } if compute_device == device
            ) {
                return None;
            }
            region.tensors.iter().find_map(|tensor| {
                (matches!(tensor.role, TensorRole::TokenEmbedding | TensorRole::LmHead)
                    && tensor.info.dtype == crate::tensor::TensorDType::BF16
                    && tensor.info.shape.len() == 2)
                    .then_some((region, &tensor.info))
            })
        })
        .ok_or_else(|| AegisError::InvalidPlan("no BF16 dense matrix on cuda plan found".into()))?;
    let placement = region_placements.get(&region.id).ok_or_else(|| {
        AegisError::InvalidPlan(format!("missing placement for region `{}`", region.id.0))
    })?;
    let residency = match placement.store {
        StoragePlacement::Vram {
            device: store_device,
        } if store_device == device => TensorResidencyPlan::VramResident { device },
        StoragePlacement::Ram | StoragePlacement::Mmap => {
            return Err(AegisError::Unsupported(format!(
                "cuda-dense-smoke cannot load {} as a CUDA resident matrix until streaming H2D transfer nodes exist",
                placement.store
            )));
        }
        StoragePlacement::Vram {
            device: store_device,
        } => {
            return Err(AegisError::Unsupported(format!(
                "cuda-dense-smoke cannot load cross-device matrix store=vram:{store_device} compute=cuda:{device}"
            )));
        }
    };
    let cuda = crate::cuda::CudaRuntime::new_with_config(device, cuda_config)?;
    let cuda_weights = cuda.weight_loader();
    let mut loader = TensorStorageLoader::new();
    let matrix = cuda_weights.load_bf16_matrix_with_store(
        tensor,
        placement.store,
        residency,
        &mut loader,
    )?;
    let input = deterministic_input(matrix.cols);
    let output = cuda.matvec_bf16_reference_host(&matrix, &input)?;
    let checksum: f32 = output.iter().take(32).copied().sum();
    let max_abs = output.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);
    println!(
        "cuda-dense-smoke: device={} matrix={} rows={} cols={} residency={:?} output_len={} checksum32={:.6} first={:.6} max_abs={:.6}",
        cuda.device_index(),
        matrix.name,
        matrix.rows,
        matrix.cols,
        matrix.residency,
        output.len(),
        checksum,
        output.first().copied().unwrap_or(0.0),
        max_abs,
    );
    Ok(())
}

pub(super) fn cuda_chain_smoke(config: EngineConfig) -> Result<()> {
    let cuda_config = config.cuda;
    let engine = AegisEngine::build(EngineConfig {
        enable_executor: false,
        ..config
    })?;
    let (device, region, placement) = first_cuda_nvfp4_region(&engine)?;
    let cuda = crate::cuda::CudaRuntime::new_with_config(device, cuda_config)?;
    let cuda_weights = cuda.weight_loader();
    let resident_layout = resident_layout_for_region(&engine, &region.id);
    let linears = cuda_weights.load_placed_region_nvfp4_linears_with_layout(
        &engine.artifact,
        region,
        placement,
        resident_layout,
    )?;
    let q_proj = find_cuda_linear(&linears, ".self_attn.q_proj")?;
    let o_proj = find_cuda_linear(&linears, ".self_attn.o_proj")?;
    let gate_proj = find_cuda_linear(&linears, ".mlp.gate_proj")?;
    let down_proj = find_cuda_linear(&linears, ".mlp.down_proj")?;
    if q_proj.rows != o_proj.cols {
        return Err(AegisError::InvalidPlan(format!(
            "cuda-chain-smoke q->o shape mismatch: q rows={} o cols={}",
            q_proj.rows, o_proj.cols
        )));
    }
    if gate_proj.rows != down_proj.cols {
        return Err(AegisError::InvalidPlan(format!(
            "cuda-chain-smoke gate->down shape mismatch: gate rows={} down cols={}",
            gate_proj.rows, down_proj.cols
        )));
    }

    let input = deterministic_input(q_proj.cols);
    let input_dev = cuda.upload_f32(&input)?;
    let mut q_dev = cuda.alloc_f32(q_proj.rows)?;
    cuda.matvec_nvfp4_reference_device(q_proj, &input_dev, &mut q_dev)?;
    let mut o_dev = cuda.alloc_f32(o_proj.rows)?;
    cuda.matvec_nvfp4_reference_device(o_proj, &q_dev, &mut o_dev)?;

    let mut gate_dev = cuda.alloc_f32(gate_proj.rows)?;
    cuda.matvec_nvfp4_reference_device(gate_proj, &input_dev, &mut gate_dev)?;
    let mut down_dev = cuda.alloc_f32(down_proj.rows)?;
    cuda.matvec_nvfp4_reference_device(down_proj, &gate_dev, &mut down_dev)?;

    let o_host = cuda.download_f32(&o_dev)?;
    let down_host = cuda.download_f32(&down_dev)?;
    let checksum_o: f32 = o_host.iter().take(32).copied().sum();
    let checksum_down: f32 = down_host.iter().take(32).copied().sum();
    println!(
        "cuda-chain-smoke: device={} region={} layout={} q_to_o_len={} gate_to_down_len={} checksum_o32={:.6} checksum_down32={:.6} first_o={:.6} first_down={:.6}",
        cuda.device_index(),
        region.id.0,
        resident_layout,
        o_host.len(),
        down_host.len(),
        checksum_o,
        checksum_down,
        o_host.first().copied().unwrap_or(0.0),
        down_host.first().copied().unwrap_or(0.0),
    );
    Ok(())
}

pub(super) fn cuda_compare(config: EngineConfig) -> Result<()> {
    let cuda_config = config.cuda;
    let engine = AegisEngine::build(EngineConfig {
        enable_executor: false,
        ..config
    })?;
    let (device, region, placement) = first_cuda_nvfp4_region(&engine)?;
    let cuda = crate::cuda::CudaRuntime::new_with_config(device, cuda_config)?;
    let cuda_weights = cuda.weight_loader();
    let resident_layout = resident_layout_for_region(&engine, &region.id);
    let linears = cuda_weights.load_placed_region_nvfp4_linears_with_layout(
        &engine.artifact,
        region,
        placement,
        resident_layout,
    )?;
    if linears.is_empty() {
        return Err(AegisError::InvalidPlan(format!(
            "no NVFP4 linear tensor found in region `{}`",
            region.id.0
        )));
    }
    let mut cpu_loader = TensorStorageLoader::new();
    let cpu = crate::cpu::CpuRuntime::new();
    for linear in &linears {
        cuda_compare_linear(&cuda, &cpu, &engine, linear, &mut cpu_loader)?;
    }
    Ok(())
}

fn cuda_compare_linear(
    cuda: &crate::cuda::CudaRuntime,
    cpu: &crate::cpu::CpuRuntime,
    engine: &AegisEngine,
    linear: &crate::cuda::DeviceNvfp4Linear,
    cpu_loader: &mut TensorStorageLoader,
) -> Result<()> {
    let cpu_linear = cpu.load_nvfp4_linear_with_store(
        &engine.artifact,
        &linear.name,
        StoragePlacement::Mmap,
        TensorResidencyPlan::FileBackedMmap,
        cpu_loader,
    )?;
    let input = deterministic_input(linear.cols);
    let gpu_output = cuda.matvec_nvfp4_reference_host(linear, &input)?;
    let cpu_output = cpu_linear.matvec(&input)?;
    let mut max_abs_diff = 0.0_f32;
    let mut mean_abs_diff = 0.0_f32;
    for (&gpu, &cpu) in gpu_output.iter().zip(cpu_output.iter()) {
        let diff = (gpu - cpu).abs();
        max_abs_diff = max_abs_diff.max(diff);
        mean_abs_diff += diff;
    }
    mean_abs_diff /= gpu_output.len().max(1) as f32;
    println!(
        "cuda-compare: device={} linear={} rows={} cols={} residency={:?} layout={} output_len={} max_abs_diff={:.8} mean_abs_diff={:.8} gpu_first={:.6} cpu_first={:.6}",
        cuda.device_index(),
        linear.name,
        linear.rows,
        linear.cols,
        linear.residency,
        linear.resident_layout,
        gpu_output.len(),
        max_abs_diff,
        mean_abs_diff,
        gpu_output.first().copied().unwrap_or(0.0),
        cpu_output.first().copied().unwrap_or(0.0),
    );
    if linear.native_mxfp4_bytes() > 0 {
        let input_dev = cuda.upload_f32(&input)?;
        let mut native_dev = cuda.alloc_f32(linear.rows)?;
        cuda.matvec_mxfp4_native_device(linear, &input_dev, &mut native_dev)?;
        let native_output = cuda.download_f32(&native_dev)?;
        let mut native_max_abs_diff = 0.0_f32;
        let mut native_mean_abs_diff = 0.0_f32;
        for (&native, &reference) in native_output.iter().zip(gpu_output.iter()) {
            let diff = (native - reference).abs();
            native_max_abs_diff = native_max_abs_diff.max(diff);
            native_mean_abs_diff += diff;
        }
        native_mean_abs_diff /= native_output.len().max(1) as f32;
        println!(
            "cuda-native-compare: native_mxfp4_bytes={} blocks_per_row={} max_abs_diff_vs_ref={:.8} mean_abs_diff_vs_ref={:.8} native_first={:.6} ref_first={:.6}",
            linear.native_mxfp4_bytes(),
            linear.native_mxfp4_blocks_per_row(),
            native_max_abs_diff,
            native_mean_abs_diff,
            native_output.first().copied().unwrap_or(0.0),
            gpu_output.first().copied().unwrap_or(0.0),
        );
        if !native_max_abs_diff.is_finite()
            || !native_mean_abs_diff.is_finite()
            || native_max_abs_diff > 1.0
            || native_mean_abs_diff > 0.08
        {
            return Err(AegisError::InvalidPlan(format!(
                "native MXFP4 compare failed for `{}`: max_abs_diff={native_max_abs_diff:.8} mean_abs_diff={native_mean_abs_diff:.8}",
                linear.name
            )));
        }
    }
    Ok(())
}

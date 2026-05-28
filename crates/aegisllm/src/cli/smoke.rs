use super::helpers::{
    deterministic_input, find_cuda_linear, first_cuda_nvfp4_region, resident_layout_for_region,
};
use aegisllm_cuda::cuda::CUDA_PREFILL_CHUNK_MAX;
use crate::engine::quality::run_quality_smoke;
use crate::engine::{AegisEngine, EngineConfig};
use aegisllm_base::error::{AegisError, Result};
use crate::executor::readiness_for_plan;
use aegisllm_base::generation::{GenerateOutput, GenerateRequest, SamplingConfig};
use aegisllm_base::graph::TensorRole;
use aegisllm_base::hardware::HardwareInventory;
use aegisllm_cpu::materialization::LinearMaterializationCache;
use aegisllm_base::planning::placement::{ComputePlacement, StoragePlacement};
use aegisllm_base::tensor::storage::{HostTensorStorage, TensorResidencyPlan, TensorStorageLoader};

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
                min_p: 0.0,
            },
            stop_token_ids: Vec::new(),
            image_injection: None,
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
            chunk_engine.graph.num_attention_heads,
            chunk_engine.graph.num_kv_heads,
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

/// Stage A.3 correctness oracle. Prefills a fixed short prompt twice on the
/// real model — run 1 with the `reference_kernel` backend (default
/// `CudaPrefillAttentionKernel::Reference`, the f32 oracle; selectable via
/// `--reference`), run 2 with a chosen fast backend (the engine config's
/// `prefill_attention`, default = Auto, selectable via
/// `--cuda-prefill-attention`) — and reports per-layer post-attention
/// hidden-state diffs plus a final-logits diff.
///
/// The default reference path is the scalar f32 online-softmax prefill kernel
/// (`aegis_attention_prefill_batched` in attention_prefill_dense_wmma.cu),
/// reached because `prefill_attention=Reference` makes every fast-path branch
/// in `attention_prefill_dense_compat_device` fall through to
/// `attention_prefill_batched_device`. It is correct for short prompts: at
/// context < the sliding window (1024) full causal attention equals windowed
/// attention, so the absence of a window mask in the reference kernel is a
/// no-op.
///
/// Passing `--reference auto` (or another fast backend) makes run 1 use the
/// DEFAULT WMMA dispatch instead — this isolates a fast backend's own
/// algorithmic divergence from the f32-scalar-vs-bf16-tensor-core precision
/// difference, because both runs then share the same bf16 precision regime.
pub(super) fn cuda_attn_compare(
    config: EngineConfig,
    prompt: Option<String>,
    reference_kernel: aegisllm_cuda::cuda::CudaPrefillAttentionKernel,
) -> Result<()> {
    // A short, fixed default prompt — long enough to exercise attention but
    // small enough that the whole compare finishes in seconds.
    let prompt = prompt.unwrap_or_else(|| {
        "The capital of France is Paris, a city on the river Seine.".to_string()
    });

    let fast_kernel = config.cuda.prefill_attention;
    println!(
        "cuda-attn-compare: prompt={prompt:?} reference={} fast={}",
        reference_kernel.canonical_name(),
        fast_kernel.canonical_name(),
    );

    // Run 1: the chosen reference backend (default = f32 reference oracle).
    let mut ref_config = config.clone();
    ref_config.cuda.prefill_attention = reference_kernel;
    let (ref_layers, ref_logits) = run_attn_compare_prefill(ref_config, &prompt)?;

    // Run 2: the chosen fast backend (engine default unless overridden).
    let (fast_layers, fast_logits) = run_attn_compare_prefill(config, &prompt)?;

    if ref_layers.len() != fast_layers.len() {
        return Err(AegisError::InvalidPlan(format!(
            "cuda-attn-compare: layer count mismatch: reference={} fast={}",
            ref_layers.len(),
            fast_layers.len()
        )));
    }
    if ref_layers.is_empty() {
        return Err(AegisError::InvalidPlan(
            "cuda-attn-compare: no per-layer hidden states captured — \
             prefill did not run through the chunked CUDA path".into(),
        ));
    }

    println!(
        "  {:>5}  {:>14}  {:>14}  {:>12}",
        "layer", "max_abs_diff", "mean_abs_diff", "cosine_sim",
    );
    let mut worst_max = 0.0_f32;
    let mut worst_cos = 1.0_f32;
    for (idx, (r, f)) in ref_layers.iter().zip(fast_layers.iter()).enumerate() {
        let (max_abs, mean_abs, cos) = diff_stats(r, f);
        worst_max = worst_max.max(max_abs);
        worst_cos = worst_cos.min(cos);
        println!(
            "  {idx:>5}  {max_abs:>14.6e}  {mean_abs:>14.6e}  {cos:>12.8}",
        );
    }
    let (logit_max, logit_mean, logit_cos) = diff_stats(&ref_logits, &fast_logits);
    println!(
        "  final logits: len={} max_abs_diff={logit_max:.6e} mean_abs_diff={logit_mean:.6e} cosine_sim={logit_cos:.8}",
        ref_logits.len(),
    );
    let ref_argmax = argmax(&ref_logits);
    let fast_argmax = argmax(&fast_logits);
    println!(
        "  argmax: reference[{}]={ref_argmax} fast[{}]={fast_argmax} match={}",
        reference_kernel.canonical_name(),
        fast_kernel.canonical_name(),
        ref_argmax == fast_argmax,
    );
    println!(
        "cuda-attn-compare: layers={} worst_max_abs_diff={worst_max:.6e} worst_cosine_sim={worst_cos:.8}",
        ref_layers.len(),
    );
    Ok(())
}

/// Prefill `prompt` on a freshly built engine and return
/// `(per_layer_post_attn_hidden, final_logits)`. The per-layer hidden states
/// are the row-0 post-attention residual for each transformer layer; the
/// final logits are for the position after the last prompt token.
fn run_attn_compare_prefill(
    mut config: EngineConfig,
    prompt: &str,
) -> Result<(Vec<Vec<f32>>, Vec<f32>)> {
    // Use a prefill chunk large enough that any short compare prompt fits in
    // a single chunk (so each layer's capture hook fires exactly once), but
    // small enough that the prefill scratch buffers stay modest — sizing
    // scratch for CUDA_PREFILL_CHUNK_MAX would OOM a second engine build.
    const ATTN_COMPARE_CHUNK: usize = 512;
    config.cuda.prefill_chunk_size = Some(ATTN_COMPARE_CHUNK);
    let engine = AegisEngine::build(config)?;
    let executor = engine.executor().ok_or_else(|| {
        AegisError::Unsupported("cuda-attn-compare: engine built without executor".into())
    })?;
    let backend = executor.as_primitives();

    let tokens = backend.encode_prompt(prompt)?;
    if tokens.len() < 2 {
        return Err(AegisError::InvalidPlan(format!(
            "cuda-attn-compare: prompt tokenizes to {} token(s); need ≥2",
            tokens.len()
        )));
    }
    if tokens.len() > ATTN_COMPARE_CHUNK {
        return Err(AegisError::InvalidPlan(format!(
            "cuda-attn-compare: prompt tokenizes to {} tokens, exceeding the \
             single-chunk limit of {ATTN_COMPARE_CHUNK}; use a shorter prompt",
            tokens.len()
        )));
    }
    let (&last, prefix) = tokens.split_last().expect("len ≥ 2 checked above");

    let greedy = aegisllm_base::generation::SamplingConfig {
        temperature: 0.0,
        top_k: 1,
        top_p: 1.0,
        min_p: 0.0,
    };
    let mut state = backend.new_sequence_state()?;

    // Arm per-layer capture only around the prefill so unrelated thread-local
    // state stays clean. `prefill_prompt` runs the chunked CUDA prefill which
    // invokes the capture hook once per layer.
    aegisllm_cuda::layer_capture::arm();
    let prefill_result = backend.prefill_prompt(state.as_mut(), prefix, &greedy);
    let layers = aegisllm_cuda::layer_capture::take();
    aegisllm_cuda::layer_capture::disarm();
    prefill_result?;

    // Logits for the position after the full prompt.
    let logits = backend.forward_logits(state.as_mut(), last)?;
    Ok((layers, logits))
}

/// `(max_abs_diff, mean_abs_diff, cosine_similarity)` between two vectors.
fn diff_stats(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    let n = a.len().min(b.len());
    if n == 0 {
        return (0.0, 0.0, 1.0);
    }
    let mut max_abs = 0.0_f32;
    let mut sum_abs = 0.0_f64;
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    for i in 0..n {
        let (x, y) = (a[i], b[i]);
        let d = (x - y).abs();
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
        dot += (x as f64) * (y as f64);
        norm_a += (x as f64) * (x as f64);
        norm_b += (y as f64) * (y as f64);
    }
    let cos = if norm_a > 0.0 && norm_b > 0.0 {
        (dot / (norm_a.sqrt() * norm_b.sqrt())) as f32
    } else {
        1.0
    };
    (max_abs, (sum_abs / n as f64) as f32, cos)
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
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
    let cuda = aegisllm_cuda::cuda::CudaRuntime::new_with_config(device, cuda_config)?;
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

/// Vision-tower load smoke (Stage I.1). Opens the artifact, builds a
/// CudaRuntime + WeightLoader the same way `cuda_smoke` does, then calls
/// `VisionTower::from_artifact` to load all 356 vision-tower + projector
/// tensors into VRAM. Reports counts and a quick sanity check.
pub(super) fn vision_load_smoke(config: EngineConfig) -> Result<()> {
    use aegisllm_cuda::executor::vision::{VisionEncoderShape, VisionTower};
    use aegisllm_base::tensor::storage::TensorStorageLoader;
    use aegisllm_base::modalities::image_preprocess::ImageProcessor;
    use std::path::Path;

    let cuda_config = config.cuda;
    let engine = AegisEngine::build(EngineConfig {
        enable_executor: false,
        ..config
    })?;
    let device = first_cuda_device(&engine, "vision-load-smoke")?;
    let cuda = aegisllm_cuda::cuda::CudaRuntime::new_with_config(device, cuda_config)?;
    let cuda_weights = cuda.weight_loader();
    let mut loader = TensorStorageLoader::new();

    let t0 = std::time::Instant::now();
    let shape = VisionEncoderShape::from_artifact(&engine.artifact)?;
    eprintln!(
        "vision-load-smoke: arch={} vision\n  hidden={} intermediate={} layers={} heads={} head_dim={}\n  patch={} pool={} pos_table={} standardize={}\n  loading vision tower...",
        engine.artifact.config.model_type,
        shape.hidden_size, shape.intermediate_size, shape.num_layers,
        shape.num_attention_heads, shape.head_dim,
        shape.patch_size, shape.pooling_kernel_size, shape.position_embedding_size,
        shape.standardize,
    );
    let tower = VisionTower::from_artifact(
        &engine.artifact, shape, &cuda_weights, device, &mut loader,
    )?;
    let dt = t0.elapsed();
    eprintln!(
        "vision-load-smoke: OK — loaded in {:.2}s",
        dt.as_secs_f64()
    );
    eprintln!(
        "  patch_embed:    {}x{}",
        tower.patch_embed.rows, tower.patch_embed.cols
    );
    eprintln!(
        "  position_table: {}x{}",
        tower.position_table.rows, tower.position_table.cols
    );
    if let Some(ref std) = tower.std {
        eprintln!(
            "  std_scale:      len={}     std_bias: len={}",
            std.scale.len(), std.bias.len()
        );
    } else {
        eprintln!("  std:            (omitted — vision_config.standardize=false)");
    }
    eprintln!("  layers:         {} loaded", tower.layers.len());
    eprintln!(
        "  layer[0]:       q_proj={}x{} mlp_gate={}x{}",
        tower.layers[0].q_proj.rows, tower.layers[0].q_proj.cols,
        tower.layers[0].mlp_gate.rows, tower.layers[0].mlp_gate.cols,
    );
    eprintln!(
        "  projector:      {}x{}  (vision_hidden -> text_hidden)",
        tower.projector.rows, tower.projector.cols
    );

    // ── Optional forward pass smoke: if AEGIS_VISION_IMAGE is set, load that
    // image and run the full vision-tower forward, report output token count
    // + a few projected embedding values.
    if let Ok(path) = std::env::var("AEGIS_VISION_IMAGE") {
        eprintln!("\nvision-load-smoke: running FORWARD on {path}");
        let vc = engine.artifact.config.vision_config.as_ref()
            .ok_or_else(|| aegisllm_base::error::AegisError::InvalidPlan(
                "vision-load-smoke: config.json missing vision_config".into()
            ))?;
        let max_soft = engine.artifact.config.vision_soft_tokens_per_image
            .ok_or_else(|| aegisllm_base::error::AegisError::InvalidPlan(
                "vision-load-smoke: config.json missing vision_soft_tokens_per_image".into()
            ))?;
        let processor = ImageProcessor::from_artifact_vision(vc, max_soft);
        let img = processor.load(Path::new(&path))?;
        eprintln!(
            "  preprocessed:   {}x{} → {} patches ({}x{}) → {} tokens ({}x{})",
            img.height, img.width,
            img.num_patches(), img.num_patches_h, img.num_patches_w,
            img.num_tokens(), img.num_tokens_h, img.num_tokens_w
        );
        let t1 = std::time::Instant::now();
        let embeds = tower.forward_gpu(&cuda, &img.patches, img.num_patches_h, img.num_patches_w)?;
        let dt1 = t1.elapsed();
        let text_hidden = tower.projector.rows;
        let n_tokens = img.num_tokens();
        let expected = n_tokens * text_hidden;
        if embeds.len() != expected {
            return Err(aegisllm_base::error::AegisError::InvalidPlan(format!(
                "vision forward produced {} f32, expected {}", embeds.len(), expected
            )));
        }
        let sum: f32 = embeds.iter().sum();
        let mean = sum / embeds.len() as f32;
        let mut max = f32::MIN;
        let mut min = f32::MAX;
        for &x in &embeds {
            if x > max { max = x; }
            if x < min { min = x; }
        }
        eprintln!(
            "  forward OK:     {:.2}s  output [{}, {}] mean={:.4} min={:.4} max={:.4}",
            dt1.as_secs_f64(), n_tokens, text_hidden, mean, min, max
        );
        eprintln!("  embeds[0][0..8]: {:?}", &embeds[..8.min(embeds.len())]);
    }

    println!("vision-load-smoke: PASS");
    Ok(())
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
    let cuda = aegisllm_cuda::cuda::CudaRuntime::new_with_config(device, cuda_config)?;
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
    let cuda = aegisllm_cuda::cuda::CudaRuntime::new_with_config(device, cuda_config)?;
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
    let cuda = aegisllm_cuda::cuda::CudaRuntime::new_with_config(device, cuda_config)?;
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
    let cpu = aegisllm_cpu::cpu::CpuRuntime::new();
    for linear in &linears {
        cuda_compare_linear(&cuda, &cpu, &engine, linear, &mut cpu_loader)?;
    }
    Ok(())
}

fn cuda_compare_linear(
    cuda: &aegisllm_cuda::cuda::CudaRuntime,
    cpu: &aegisllm_cpu::cpu::CpuRuntime,
    engine: &AegisEngine,
    linear: &aegisllm_cuda::cuda::DeviceNvfp4Linear,
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

/// Standalone smoke test for the CUTLASS NVFP4 grouped GEMM bridge.
/// No model load required — exercises the three new helpers + grouped
/// kernel on synthetic 4-expert data with M ∈ {128, 256, 512, 1024},
/// N=K=128, and compares against a CPU NVFP4 dequant + f32 matmul
/// reference. Pass criteria: cos_sim ≥ 0.998 per expert.
pub(super) fn cuda_cutlass_nvfp4_smoke() -> Result<()> {
    // Pick CUDA device 0. The CUTLASS NVFP4 grouped kernel only targets
    // sm_120+; if the device isn't Blackwell, can_implement() inside the
    // kernel will reject and we surface that here.
    use aegisllm_base::cuda_config::CudaRuntimeConfig;
    let runtime = aegisllm_cuda::cuda::CudaRuntime::new_with_config(
        0,
        CudaRuntimeConfig::from_env(),
    )?;
    if !aegisllm_cuda::cuda::CudaRuntime::cutlass_nvfp4_moe_grouped_built() {
        eprintln!(
            "cuda-cutlass-nvfp4-smoke FAIL: bridge not compiled — rebuild with \
             AEGIS_CUTLASS_NVFP4_GROUPED_BUILD=1"
        );
        return Err(AegisError::Unsupported(
            "AEGIS_CUTLASS_NVFP4_GROUPED_BUILD=1 required".into(),
        ));
    }
    let report = runtime.cutlass_nvfp4_moe_grouped_smoke()?;
    println!(
        "cuda-cutlass-nvfp4-smoke: device={} experts={} workspace_bytes={} gemm_ms={:.3} threshold={:.4}",
        runtime.device_index(),
        report.num_experts,
        report.workspace_bytes,
        report.gemm_ms,
        report.cos_sim_threshold,
    );
    for (g, e) in report.experts.iter().enumerate() {
        let (sfa_b, sfb_b) = report.sfa_sfb_bytes[g];
        let verdict = if e.cos_sim >= report.cos_sim_threshold { "PASS" } else { "FAIL" };
        println!(
            "  expert[{g}] m={} n={} k={} alpha={:.3} sfa_bytes={} sfb_bytes={} cos_sim={:.8} scale_ratio={:.6} abs_max_err={:.4e} ref_abs_max={:.4e} ref_l2={:.4e} {verdict}",
            e.m, e.n, e.k, e.output_scale, sfa_b, sfb_b, e.cos_sim, e.scale_ratio, e.abs_max_err, e.ref_abs_max, e.ref_l2
        );
    }
    if report.passed {
        println!("cuda-cutlass-nvfp4-smoke: PASS");
        Ok(())
    } else {
        eprintln!("cuda-cutlass-nvfp4-smoke: FAIL");
        Err(AegisError::Unsupported(
            "cuda-cutlass-nvfp4-smoke failed: at least one expert below cos_sim threshold".into(),
        ))
    }
}

/// Standalone smoke test for the SM120 FP8 e4m3 `m16n8k32` tensor-core MMA.
/// No model load — exercises the bare MMA primitive, a tiled FP8 GEMM, and a
/// tiny synthetic FP8 attention against CPU references in escalating stages.
/// This de-risks the raw `mma.sync.aligned.kind::f8f6f4.m16n8k32` instruction
/// before a from-scratch FP8 FlashAttention kernel is written.
pub(super) fn cuda_attn_fp8_smoke() -> Result<()> {
    use aegisllm_base::cuda_config::CudaRuntimeConfig;
    let runtime = aegisllm_cuda::cuda::CudaRuntime::new_with_config(
        0,
        CudaRuntimeConfig::from_env(),
    )?;
    let report = runtime.fp8_mma_smoke()?;
    println!(
        "cuda-attn-fp8-smoke: device={} compute_capability={}",
        report.device_index, report.compute_capability,
    );
    for s in &report.stages {
        let verdict = if s.passed { "PASS" } else { "FAIL" };
        println!(
            "  {} [{}] cos_sim={:.8} abs_max_err={:.4e} ref_abs_max={:.4e} \
             deterministic={} bar={:.4} {verdict}",
            s.name, s.shape, s.cos_sim, s.abs_max_err, s.ref_abs_max, s.deterministic, s.bar,
        );
    }
    if report.passed {
        println!("cuda-attn-fp8-smoke: PASS");
        Ok(())
    } else {
        eprintln!("cuda-attn-fp8-smoke: FAIL");
        Err(AegisError::Unsupported(
            "cuda-attn-fp8-smoke failed: at least one stage below its cos_sim bar".into(),
        ))
    }
}

/// Standalone correctness check for the GPU f32 reference attention kernel
/// (`aegis_attention_prefill_batched`). Validates it against the INDEPENDENT
/// CPU f32 reference (`reference_attention_prefill_f32_into`) on identical
/// synthetic Q/K/V inputs — no model load.
///
/// The GPU reference reads a full-f32 query but reads K/V as f16 bits (the KV
/// cache dtype). To make the comparison a pure algorithm check (not a
/// precision check) we round the synthetic K/V to f16 FIRST, then feed those
/// f16-exact values to BOTH references: the GPU kernel reads the f16 bits,
/// the CPU reference reads the same values widened back to f32. Q is f32 for
/// both. With identical numeric inputs, any output divergence is an algorithm
/// bug (wrong GQA mapping, causal range, head_dim handling), not rounding.
///
/// Acceptance: cosine ≥ 0.9999 and a tiny max-abs diff on every case.
pub(super) fn cuda_attn_ref_check() -> Result<()> {
    use aegisllm_base::cuda_config::{CudaPrefillAttentionKernel, CudaRuntimeConfig};
    use aegisllm_cpu::{ReferenceAttentionPrefillRequest, reference_attention_prefill_f32_into};
    use half::f16;

    // Force the GPU reference kernel: `prefill_attention = Reference` +
    // `start_position = 0` routes `attention_prefill_batched_device` to the
    // `CacheOnly` path = `aegis_attention_prefill_batched`, the kernel under
    // test. (Auto would pick the Warp kernel for head_dim <= 256.)
    let mut config = CudaRuntimeConfig::from_env();
    config.prefill_attention = CudaPrefillAttentionKernel::Reference;
    let runtime = aegisllm_cuda::cuda::CudaRuntime::new_with_config(0, config)?;

    // Deterministic pseudo-random generator (small magnitudes so f16 rounding
    // of K/V stays well-conditioned and softmax does not overflow).
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || -> f32 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let bits = (seed >> 33) as u32;
        (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
    };

    // (label, q_heads, kv_heads, head_dim, batch). Covers GQA groups 1/4/8
    // and head_dim 256 + 512, causal multi-token batches.
    let cases: &[(&str, usize, usize, usize, usize)] = &[
        ("hdim256 gqa1", 4, 4, 256, 17),
        ("hdim256 gqa4", 8, 2, 256, 33),
        ("hdim512 gqa1", 2, 2, 512, 19),
        ("hdim512 gqa4", 8, 2, 512, 40),
        ("hdim512 gqa8", 8, 1, 512, 48),
    ];

    println!("cuda-attn-ref-check: device={} GPU-f32-ref vs CPU-f32-ref", runtime.device_index());
    println!(
        "  {:<16} {:>7} {:>8} {:>9} {:>6}  {:>14} {:>14} {:>12}  verdict",
        "case", "q_heads", "kv_heads", "head_dim", "batch", "max_abs_diff", "mean_abs_diff",
        "cosine_sim",
    );

    const COS_BAR: f32 = 0.9999;
    let mut all_pass = true;

    for &(label, q_heads, kv_heads, head_dim, batch) in cases {
        let q_width = q_heads * head_dim;
        let kv_width = kv_heads * head_dim;

        // Synthetic query (f32) and K/V (rounded to f16, then widened to f32
        // so both references consume bit-identical numeric values).
        let query: Vec<f32> = (0..batch * q_width).map(|_| next()).collect();
        let keys_f16: Vec<u16> = (0..batch * kv_width)
            .map(|_| f16::from_f32(next()).to_bits())
            .collect();
        let values_f16: Vec<u16> = (0..batch * kv_width)
            .map(|_| f16::from_f32(next()).to_bits())
            .collect();
        let keys_f32: Vec<f32> = keys_f16
            .iter()
            .map(|&b| f16::from_bits(b).to_f32())
            .collect();
        let values_f32: Vec<f32> = values_f16
            .iter()
            .map(|&b| f16::from_bits(b).to_f32())
            .collect();

        // --- CPU reference ---
        let mut cpu_out = vec![0.0_f32; batch * q_width];
        reference_attention_prefill_f32_into(
            ReferenceAttentionPrefillRequest {
                keys: &keys_f32,
                values: &values_f32,
                start_position: 0,
                batch,
                query: &query,
                num_attention_heads: q_heads,
                num_kv_heads: kv_heads,
                head_dim,
            },
            &mut cpu_out,
        )?;

        // --- GPU reference ---
        let d_keys = runtime.upload_u16(&keys_f16)?;
        let d_values = runtime.upload_u16(&values_f16)?;
        let d_query = runtime.upload_f32(&query)?;
        // CacheOnly path ignores key_chunk/value_chunk but the shape check
        // still requires them to be at least `batch * kv_width` long.
        let d_key_chunk = runtime.upload_f32(&vec![0.0_f32; batch * kv_width])?;
        let d_value_chunk = runtime.upload_f32(&vec![0.0_f32; batch * kv_width])?;
        let mut d_out = runtime.alloc_f32(batch * q_width)?;
        runtime.attention_prefill_batched_device(
            &d_keys,
            &d_values,
            &d_key_chunk,
            &d_value_chunk,
            &d_query,
            0,
            batch,
            q_heads,
            kv_heads,
            head_dim,
            &mut d_out,
        )?;
        runtime.synchronize()?;
        let gpu_out = runtime.download_f32(&d_out)?;

        let (max_abs, mean_abs, cos) = diff_stats(&cpu_out, &gpu_out);
        let pass = cos >= COS_BAR && max_abs < 1.0e-2;
        all_pass &= pass;
        println!(
            "  {label:<16} {q_heads:>7} {kv_heads:>8} {head_dim:>9} {batch:>6}  \
             {max_abs:>14.6e} {mean_abs:>14.6e} {cos:>12.8}  {}",
            if pass { "PASS" } else { "FAIL" },
        );
    }

    if all_pass {
        println!(
            "cuda-attn-ref-check: PASS — GPU f32 reference is algorithmically correct \
             (cosine >= {COS_BAR} on every case)"
        );
        Ok(())
    } else {
        eprintln!("cuda-attn-ref-check: FAIL — GPU f32 reference diverges from CPU f32 reference");
        Err(AegisError::Unsupported(
            "cuda-attn-ref-check failed: GPU f32 reference attention disagrees with the CPU \
             f32 reference on identical inputs".into(),
        ))
    }
}

use super::{AegisEngine, EngineReport};
use crate::cuda::CudaRuntimeConfig;
use crate::executor::{ExecutorGraphPlan, readiness_for_plan};
use crate::planning::memory::AllocationPool;
use crate::planning::placement::{
    ComputePlacement, ResolvedPlacement, StoragePlacement, TransferPolicy,
};
use crate::planning::runtime::{KernelFamily, RuntimePlan};
use crate::tensor::layout::LinearResidentLayout;
use crate::tensor::quant::QuantFormat;

impl AegisEngine {
    pub fn report(&self) -> EngineReport {
        let mut lines = Vec::new();
        lines.push(format!("model: {}", self.placement.model));
        lines.push(format!(
            "shape: layers={} hidden={} heads={} kv_heads={} head_dim={} vocab={}",
            self.graph.num_layers,
            self.graph.hidden_size,
            self.graph.num_attention_heads,
            self.graph.num_kv_heads,
            self.graph.head_dim,
            self.graph
                .vocab_size
                .map(|v| v.to_string())
                .unwrap_or_else(|| "?".into())
        ));
        lines.push(format!(
            "weights: quant={} bytes={}",
            self.graph.weight_quantization,
            self.graph.total_weight_bytes()
        ));
        lines.push(format!(
            "product-mode: {}",
            product_mode_summary(&self.placement)
        ));
        lines.push(format!(
            "hardware: cpu={} threads={} ram_total={} ram_usable={}",
            self.inventory.cpu.model_name,
            self.inventory.cpu.logical_threads,
            self.inventory.cpu.ram_total_bytes,
            self.placement.budget.ram_usable_bytes
        ));
        for gpu in &self.inventory.gpus {
            lines.push(format!(
                "hardware: cuda:{} name={} arch={:?} cc={} vram_total={} vram_free={}",
                gpu.index,
                gpu.name,
                gpu.architecture,
                gpu.compute_capability.as_deref().unwrap_or("?"),
                gpu.vram_total_bytes,
                gpu.vram_free_bytes
                    .map(|bytes| bytes.to_string())
                    .unwrap_or_else(|| "?".into())
            ));
        }
        for backend in self.backends.iter() {
            lines.push(format!(
                "backend: {:?} flash={} paged={} fp4={} fp8={}",
                backend.kind,
                backend.supports_flash_attention,
                backend.supports_paged_attention,
                backend.supports_fp4,
                backend.supports_fp8
            ));
        }
        push_memory_report(self, &mut lines);
        push_storage_report(self, &mut lines);
        push_runtime_report(self, &mut lines);
        push_executor_report(self, &mut lines);
        lines.extend(region_summary(&self.placement));
        for warning in &self.memory.warnings {
            lines.push(format!("warning: {warning}"));
        }
        for warning in &self.runtime.warnings {
            lines.push(format!("runtime-warning: {warning}"));
        }
        EngineReport { lines }
    }
}

impl std::fmt::Display for EngineReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for line in &self.lines {
            writeln!(f, "{line}")?;
        }
        Ok(())
    }
}

fn product_mode_summary(placement: &ResolvedPlacement) -> String {
    let all_cpu = placement
        .region_placements
        .iter()
        .all(|region| region.compute == ComputePlacement::Cpu)
        && placement.kv_cache.compute == ComputePlacement::Cpu;
    let cuda_device = placement.region_placements.iter().find_map(|region| {
        if let ComputePlacement::Cuda { device } = region.compute {
            Some(device)
        } else {
            None
        }
    });
    let all_one_cuda = cuda_device.is_some_and(|device| {
        placement.region_placements.iter().all(|region| {
            matches!(region.compute, ComputePlacement::Cuda { device: compute } if compute == device)
        }) && placement.kv_cache.compute == (ComputePlacement::Cuda { device })
    });

    if all_cpu {
        let host_store = placement
            .region_placements
            .iter()
            .all(|region| matches!(region.store, StoragePlacement::Ram | StoragePlacement::Mmap));
        let vram_store = placement
            .region_placements
            .iter()
            .any(|region| matches!(region.store, StoragePlacement::Vram { .. }));
        return if host_store {
            "ram/mmap+cpu".into()
        } else if vram_store {
            "vram+cpu (eager D2H materialization)".into()
        } else {
            "cpu".into()
        };
    }

    if all_one_cuda {
        let device = cuda_device.unwrap_or(0);
        let all_vram = placement.region_placements.iter().all(|region| {
            matches!(region.store, StoragePlacement::Vram { device: store } if store == device)
        });
        let any_host = placement
            .region_placements
            .iter()
            .any(|region| matches!(region.store, StoragePlacement::Ram | StoragePlacement::Mmap));
        return if all_vram {
            format!("vram+gpu cuda:{device}")
        } else if any_host {
            format!("ram/mmap+gpu cuda:{device} (eager H2D materialization)")
        } else {
            format!("gpu cuda:{device}")
        };
    }

    "hybrid".into()
}

fn push_memory_report(engine: &AegisEngine, lines: &mut Vec<String>) {
    let persistent_vram0 = engine
        .memory
        .footprint
        .persistent_vram_bytes
        .iter()
        .find(|(device, _)| *device == 0)
        .map(|(_, bytes)| *bytes)
        .unwrap_or(0);
    let peak_device0_staging = engine
        .memory
        .footprint
        .peak_device_staging_bytes
        .iter()
        .find(|(device, _)| *device == 0)
        .map(|(_, bytes)| *bytes)
        .unwrap_or(0);
    let prefill_scratch = engine
        .memory
        .allocations
        .iter()
        .find(|allocation| allocation.name == "cuda_prefill_scratch_per_sequence")
        .map(|allocation| allocation.bytes)
        .unwrap_or(0);
    lines.push(format!(
        "memory: persistent_ram={} file_backed_mmap={} persistent_vram0={} peak_host_staging={} peak_device0_staging={}",
        engine.memory.footprint.persistent_ram_bytes,
        engine.memory.footprint.file_backed_mmap_bytes,
        persistent_vram0,
        engine.memory.footprint.peak_host_staging_bytes,
        peak_device0_staging
    ));
    lines.push(format!(
        "memory-detail: ram_pool={} mmap_pool={} vram0_pool={} kv_cache={} cuda_prefill_scratch_per_sequence={}",
        engine.memory.bytes_in_pool(AllocationPool::Ram),
        engine.memory.bytes_in_pool(AllocationPool::Mmap),
        engine
            .memory
            .bytes_in_pool(AllocationPool::Vram { device: 0 }),
        engine.placement.kv_cache.estimated_bytes,
        prefill_scratch
    ));
    for (device, bytes) in &engine.memory.footprint.persistent_vram_bytes {
        let staging = engine
            .memory
            .footprint
            .peak_device_staging_bytes
            .iter()
            .find(|(staging_device, _)| staging_device == device)
            .map(|(_, bytes)| *bytes)
            .unwrap_or(0);
        lines.push(format!(
            "memory-device: vram:{device} persistent={} peak_staging={}",
            bytes, staging
        ));
    }
}

fn push_storage_report(engine: &AegisEngine, lines: &mut Vec<String>) {
    let storage_vram0 = engine
        .storage
        .totals
        .vram_resident_bytes
        .iter()
        .find(|(device, _)| *device == 0)
        .map(|(_, bytes)| *bytes)
        .unwrap_or(0);
    let staged_h2d0 = engine
        .storage
        .totals
        .staged_host_to_device_peak_bytes
        .iter()
        .find(|(device, _)| *device == 0)
        .map(|(_, bytes)| *bytes)
        .unwrap_or(0);
    lines.push(format!(
        "storage: tensor_ram_resident={} tensor_mmap_file_backed={} tensor_vram0_resident={} tensor_h2d0_staging_peak={}",
        engine.storage.totals.ram_resident_bytes,
        engine.storage.totals.mmap_file_backed_bytes,
        storage_vram0,
        staged_h2d0,
    ));
    for (device, bytes) in &engine.storage.totals.vram_resident_bytes {
        let staging = engine
            .storage
            .totals
            .staged_host_to_device_peak_bytes
            .iter()
            .find(|(staging_device, _)| staging_device == device)
            .map(|(_, bytes)| *bytes)
            .unwrap_or(0);
        lines.push(format!(
            "storage-device: vram:{device} tensor_resident={} h2d_staging_peak={}",
            bytes, staging
        ));
    }
}

fn push_runtime_report(engine: &AegisEngine, lines: &mut Vec<String>) {
    lines.push(format!(
        "kv_cache: store={} compute={} quant={} context={}",
        engine.placement.kv_cache.store,
        engine.placement.kv_cache.compute,
        engine.placement.kv_cache.quantization,
        engine.placement.kv_cache.context_size
    ));
    lines.push(format!(
        "linear-layout-policy: cpu={} cuda={} materialize={} max_extra={}",
        engine.placement.linear_layout.cpu,
        engine.placement.linear_layout.cuda,
        engine.placement.linear_layout.materialization,
        engine
            .placement
            .linear_layout
            .max_extra_memory_bytes
            .map(|bytes| bytes.to_string())
            .unwrap_or_else(|| "auto".into())
    ));
    lines.push(format!(
        "cuda-runtime-config: native_mxfp4_repack={} cutlass_nvfp4_repack={} native_mxfp4_inference={} prefill_attention={:?} prefill_chunk_size={}",
        engine.cuda.native_mxfp4_repack,
        engine.cuda.cutlass_nvfp4_repack,
        engine.cuda.native_mxfp4_inference,
        engine.cuda.prefill_attention,
        engine
            .cuda
            .prefill_chunk_size
            .map(|chunk| chunk.to_string())
            .unwrap_or_else(|| "auto".into()),
    ));
    lines.push(format!(
        "runtime: planned_native_fp4_tc_regions={} planned_cutlass_fp4_tc_regions={} planned_native_fp8_tc_regions={} cuda_dense_tc_regions={} cuda_quant_ref_regions={} cpu_regions={}",
        engine
            .runtime
            .count_family(KernelFamily::CudaNativeFp4TensorCores),
        engine
            .runtime
            .count_family(KernelFamily::CudaCutlassFp4TensorCores),
        engine
            .runtime
            .count_family(KernelFamily::CudaNativeFp8TensorCores),
        engine.runtime.count_family(KernelFamily::CudaDenseTensorCores),
        engine.runtime.count_family(KernelFamily::CudaQuantizedReference),
        engine.runtime.count_family(KernelFamily::CpuScalar)
            + engine.runtime.count_family(KernelFamily::CpuSimd),
    ));
    let effective_native_mxfp4 = effective_native_mxfp4_regions(&engine.runtime, engine.cuda);
    let effective_cutlass_nvfp4 = effective_cutlass_nvfp4_regions(&engine.runtime, engine.cuda);
    let effective_reference_nvfp4 =
        effective_cuda_nvfp4_reference_regions(&engine.runtime, engine.cuda);
    lines.push(format!(
        "runtime-effective: cuda_native_mxfp4_regions={} cuda_cutlass_nvfp4_regions={} cuda_nvfp4_reference_regions={}",
        effective_native_mxfp4, effective_cutlass_nvfp4, effective_reference_nvfp4
    ));
    lines.push(format!(
        "runtime-formats: nvfp4_regions={} fp8_regions={} dense_regions={}",
        engine.runtime.count_format(QuantFormat::Nvfp4),
        engine.runtime.count_format(QuantFormat::Fp8E4M3Block),
        engine.runtime.count_format(QuantFormat::DenseF32)
            + engine.runtime.count_format(QuantFormat::F16)
            + engine.runtime.count_format(QuantFormat::Bf16),
    ));
    lines.push(format!(
        "runtime-layouts: packed_source={} native_tc={} cuda_r4f_e2m1_ue4m3={} dense_tc={} repacked_fp8={} unpacked_i8={} repacked_int4={} extra_weight_bytes={}",
        engine
            .runtime
            .count_resident_layout(LinearResidentLayout::PackedSource),
        engine
            .runtime
            .count_resident_layout(LinearResidentLayout::NativeTensorCore),
        engine
            .runtime
            .count_resident_layout(LinearResidentLayout::CudaR4fE2m1Ue4m3),
        engine
            .runtime
            .count_resident_layout(LinearResidentLayout::DenseTensorCore),
        engine
            .runtime
            .count_resident_layout(LinearResidentLayout::RepackedFp8),
        engine
            .runtime
            .count_resident_layout(LinearResidentLayout::UnpackedI8Scales),
        engine
            .runtime
            .count_resident_layout(LinearResidentLayout::RepackedInt4),
        engine.runtime.extra_layout_weight_bytes(),
    ));
}

fn push_executor_report(engine: &AegisEngine, lines: &mut Vec<String>) {
    let readiness = readiness_for_plan(&engine.placement, &engine.runtime);
    lines.push(format!(
        "executor: selected={} runnable={} planned_cpu_regions={} planned_cuda_regions={}",
        readiness.selected_backend,
        readiness.runnable,
        readiness.planned_cpu_regions,
        readiness.planned_cuda_regions,
    ));
    let executor_graph = ExecutorGraphPlan::from_resolved_placement(&engine.placement);
    lines.push(format!(
        "executor-graph: nodes={} activation_transfers={} weight_transfers={} kv_cache_nodes={}",
        executor_graph.nodes.len(),
        executor_graph.activation_transfers().count(),
        executor_graph.weight_transfers().count(),
        usize::from(executor_graph.kv_cache().is_some())
    ));
    if let Some(executor) = &engine.executor {
        let info = executor.info();
        lines.push(format!(
            "executor-backend: name={} backends={:?} capabilities={}",
            info.name,
            info.backends,
            info.capabilities.len(),
        ));
        for limitation in &info.limitations {
            lines.push(format!("executor-backend-note: {limitation}"));
        }
    }
    for limitation in &readiness.limitations {
        lines.push(format!("executor-limitation: {limitation}"));
    }
}

fn effective_native_mxfp4_regions(runtime: &RuntimePlan, cuda: CudaRuntimeConfig) -> usize {
    if !(cuda.native_mxfp4_repack && cuda.native_mxfp4_inference) {
        return 0;
    }
    runtime.count_family(KernelFamily::CudaNativeFp4TensorCores)
}

fn effective_cutlass_nvfp4_regions(runtime: &RuntimePlan, cuda: CudaRuntimeConfig) -> usize {
    if !cuda.cutlass_nvfp4_repack {
        return 0;
    }
    runtime.count_family(KernelFamily::CudaCutlassFp4TensorCores)
}

fn effective_cuda_nvfp4_reference_regions(runtime: &RuntimePlan, cuda: CudaRuntimeConfig) -> usize {
    let planned_nvfp4_cuda = runtime
        .kernels
        .iter()
        .filter(|kernel| matches!(kernel.device, crate::backend::BackendKind::Cuda { .. }))
        .filter(|kernel| kernel.quant_format == QuantFormat::Nvfp4)
        .count();
    planned_nvfp4_cuda
        .saturating_sub(effective_native_mxfp4_regions(runtime, cuda))
        .saturating_sub(effective_cutlass_nvfp4_regions(runtime, cuda))
}

fn region_summary(placement: &ResolvedPlacement) -> Vec<String> {
    let mut lines = Vec::new();
    let mut current: Option<(
        StoragePlacement,
        ComputePlacement,
        TransferPolicy,
        usize,
        usize,
    )> = None;

    for region in &placement.region_placements {
        let Some(layer) = region.layer_index else {
            lines.push(format!(
                "region: {} kind={:?} store={} compute={} transfer={:?} bytes={}",
                region.region_id.0,
                region.kind,
                region.store,
                region.compute,
                region.transfer,
                region.weight_bytes
            ));
            continue;
        };

        let key = (region.store, region.compute, region.transfer);
        match current.as_mut() {
            Some((store, compute, transfer, start, end))
                if (*store, *compute, *transfer) == key && *end + 1 == layer =>
            {
                *end = layer;
            }
            Some((store, compute, transfer, start, end)) => {
                lines.push(format!(
                    "layers: {}..={} store={} compute={} transfer={:?}",
                    start, end, store, compute, transfer
                ));
                current = Some((region.store, region.compute, region.transfer, layer, layer));
            }
            None => current = Some((region.store, region.compute, region.transfer, layer, layer)),
        }
    }
    if let Some((store, compute, transfer, start, end)) = current {
        lines.push(format!(
            "layers: {}..={} store={} compute={} transfer={:?}",
            start, end, store, compute, transfer
        ));
    }

    lines
}

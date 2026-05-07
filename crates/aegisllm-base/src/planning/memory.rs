use crate::backend::BackendKind;
use crate::cuda_config::{
    CUDA_PREFILL_CHUNK_MAX, CUDA_PREFILL_DENSE_SPLIT_K_TOKENS, CudaRuntimeConfig,
};
use crate::graph::ModelGraph;
use crate::hardware::HardwareInventory;
use crate::planning::placement::{
    ComputePlacement, PlacementPolicy, ResolvedPlacement, StoragePlacement, TransferPolicy,
};
use crate::planning::runtime::{KernelFamily, RuntimePlan};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryBudget {
    pub ram_total_bytes: u64,
    pub ram_usable_bytes: u64,
    pub vram: Vec<MemoryPool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryPool {
    pub device: usize,
    pub total_bytes: u64,
    pub usable_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryPlan {
    pub allocations: Vec<PlannedAllocation>,
    pub transfers: Vec<PlannedTransfer>,
    pub footprint: MemoryFootprint,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedAllocation {
    pub name: String,
    pub pool: AllocationPool,
    pub bytes: u64,
    pub file_backed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedTransfer {
    pub name: String,
    pub policy: TransferPolicy,
    pub bytes: u64,
    pub source: StoragePlacement,
    pub compute: ComputePlacement,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MemoryFootprint {
    pub persistent_ram_bytes: u64,
    pub file_backed_mmap_bytes: u64,
    pub persistent_vram_bytes: Vec<(usize, u64)>,
    pub peak_host_staging_bytes: u64,
    pub peak_device_staging_bytes: Vec<(usize, u64)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AllocationPool {
    Ram,
    Mmap,
    Vram { device: usize },
}

impl MemoryBudget {
    pub fn from_inventory(inventory: &HardwareInventory, policy: &PlacementPolicy) -> Self {
        let ram_available = inventory
            .cpu
            .ram_available_bytes
            .unwrap_or(inventory.cpu.ram_total_bytes);
        let ram_usable_bytes = ram_available.saturating_sub(policy.reserve_ram_bytes);
        let vram = inventory
            .gpus
            .iter()
            .map(|gpu| {
                let available = gpu.vram_free_bytes.unwrap_or(gpu.vram_total_bytes);
                MemoryPool {
                    device: gpu.index,
                    total_bytes: gpu.vram_total_bytes,
                    usable_bytes: available.saturating_sub(policy.reserve_vram_bytes),
                }
            })
            .collect();

        Self {
            ram_total_bytes: inventory.cpu.ram_total_bytes,
            ram_usable_bytes,
            vram,
        }
    }

    pub fn first_vram_usable(&self) -> Option<(usize, u64)> {
        self.vram
            .first()
            .map(|pool| (pool.device, pool.usable_bytes))
    }

    pub fn vram_usable(&self, device: usize) -> Option<u64> {
        self.vram
            .iter()
            .find(|pool| pool.device == device)
            .map(|pool| pool.usable_bytes)
    }
}

impl MemoryPlan {
    pub fn from_placement(placement: &ResolvedPlacement) -> Self {
        Self::from_placement_and_runtime(placement, None)
    }

    pub fn from_placement_and_runtime(
        placement: &ResolvedPlacement,
        runtime: Option<&RuntimePlan>,
    ) -> Self {
        Self::from_placement_runtime_and_cuda(placement, runtime, CudaRuntimeConfig::from_env())
    }

    pub fn from_placement_runtime_and_cuda(
        placement: &ResolvedPlacement,
        runtime: Option<&RuntimePlan>,
        cuda: CudaRuntimeConfig,
    ) -> Self {
        Self::from_placement_runtime_graph_and_cuda(placement, runtime, None, cuda)
    }

    pub fn from_placement_runtime_graph_and_cuda(
        placement: &ResolvedPlacement,
        runtime: Option<&RuntimePlan>,
        graph: Option<&ModelGraph>,
        cuda: CudaRuntimeConfig,
    ) -> Self {
        let mut allocations = Vec::new();
        let mut transfers = Vec::new();
        for region in &placement.region_placements {
            allocations.push(PlannedAllocation {
                name: region.region_id.0.clone(),
                pool: pool_for_store(region.store),
                bytes: region.weight_bytes,
                file_backed: matches!(region.store, StoragePlacement::Mmap),
            });
            if region.transfer != TransferPolicy::None {
                transfers.push(PlannedTransfer {
                    name: region.region_id.0.clone(),
                    policy: region.transfer,
                    bytes: region.weight_bytes,
                    source: region.store,
                    compute: region.compute,
                });
            }
        }
        allocations.push(PlannedAllocation {
            name: "kv_cache".into(),
            pool: pool_for_store(placement.kv_cache.store),
            bytes: placement.kv_cache.estimated_bytes,
            file_backed: false,
        });
        if let Some(runtime) = runtime {
            // Device-resident kernels: each repacked copy lives persistently in VRAM → SUM.
            // Host/mapped-staged kernels: the executor reuses ONE staging buffer (max across
            // all host kernels per device), mirroring LinearStagingPool.new(max_*) → MAX.
            let mut host_staged_max: std::collections::BTreeMap<usize, u64> =
                std::collections::BTreeMap::new();
            for kernel in &runtime.kernels {
                let bytes = effective_layout_extra_bytes(kernel, cuda);
                if bytes == 0 {
                    continue;
                }
                let BackendKind::Cuda { device } = kernel.device else { continue };
                match kernel.residency {
                    crate::planning::runtime::TensorResidency::Device => {
                        allocations.push(PlannedAllocation {
                            name: format!("{}:layout_extra", kernel.name),
                            pool: AllocationPool::Vram { device },
                            bytes,
                            file_backed: false,
                        });
                    }
                    crate::planning::runtime::TensorResidency::Host
                    | crate::planning::runtime::TensorResidency::MappedHostToDevice => {
                        let entry = host_staged_max.entry(device).or_insert(0);
                        *entry = (*entry).max(bytes);
                    }
                }
            }
            for (device, bytes) in host_staged_max {
                allocations.push(PlannedAllocation {
                    name: "host_staged:layout_extra_max".into(),
                    pool: AllocationPool::Vram { device },
                    bytes,
                    file_backed: false,
                });
            }
        }
        if let Some(graph) = graph
            && let Some((device, bytes)) = cuda_prefill_scratch_bytes(placement, graph, cuda)
        {
            allocations.push(PlannedAllocation {
                name: "cuda_prefill_scratch_per_sequence".into(),
                pool: AllocationPool::Vram { device },
                bytes,
                file_backed: false,
            });
        }
        let footprint = footprint_from(&allocations, &transfers, placement);

        let mut warnings = placement.warnings.clone();
        let ram = allocations
            .iter()
            .filter(|a| matches!(a.pool, AllocationPool::Ram))
            .map(|a| a.bytes)
            .sum::<u64>();
        if ram > placement.budget.ram_usable_bytes {
            warnings.push(format!(
                "ram allocation exceeds usable budget: planned={} usable={}",
                ram, placement.budget.ram_usable_bytes
            ));
        }
        for pool in &placement.budget.vram {
            let vram = allocations
                .iter()
                .filter(|a| {
                    a.pool
                        == AllocationPool::Vram {
                            device: pool.device,
                        }
                })
                .map(|a| a.bytes)
                .sum::<u64>();
            if vram > pool.usable_bytes {
                warnings.push(format!(
                    "vram:{} allocation exceeds usable budget: planned={} usable={}",
                    pool.device, vram, pool.usable_bytes
                ));
            }
        }

        Self {
            allocations,
            transfers,
            footprint,
            warnings,
        }
    }

    pub fn bytes_in_pool(&self, pool: AllocationPool) -> u64 {
        self.allocations
            .iter()
            .filter(|allocation| allocation.pool == pool)
            .map(|allocation| allocation.bytes)
            .sum()
    }
}

fn cuda_prefill_scratch_bytes(
    placement: &ResolvedPlacement,
    graph: &ModelGraph,
    cuda: CudaRuntimeConfig,
) -> Option<(usize, u64)> {
    let chunk = cuda
        .prefill_chunk_size
        .unwrap_or(128)
        .clamp(1, CUDA_PREFILL_CHUNK_MAX);
    if chunk <= 1 {
        return None;
    }
    let device = placement
        .region_placements
        .iter()
        .find_map(|region| match region.compute {
            ComputePlacement::Cuda { device } => Some(device),
            _ => None,
        })
        .or(match placement.kv_cache.compute {
            ComputePlacement::Cuda { device } => Some(device),
            _ => None,
        })?;
    let hidden = graph.hidden_size as u64;
    let intermediate = graph.intermediate_size.unwrap_or(graph.hidden_size) as u64;
    let q_width = (graph.num_attention_heads * graph.head_dim) as u64;
    let kv_width = (graph.num_kv_heads * graph.head_dim) as u64;
    let chunk = chunk as u64;

    let cutlass_prefill = cuda.cutlass_nvfp4_repack;
    let intermediate_f32 = if cutlass_prefill {
        // CUTLASS prefill keeps gate and up activations, then quantizes SwiGLU
        // directly for the down projection. The fallback path additionally
        // needs full-size quant_intermediate. SwiGLU fallback is in-place in
        // gate, so there is no separate full-size swiglu scratch.
        2 * intermediate
    } else {
        3 * intermediate
    };
    let hidden_f32 = if cutlass_prefill {
        // hidden plus input_normed, which is reused for o_proj and down_proj
        // outputs. q/qkv/k/v scratch is accounted below; separate
        // attn_context/attn_out/mlp_out buffers are not resident on the
        // CUTLASS hot path.
        2 * hidden
    } else {
        // fallback keeps quant_hidden, but still reuses input_normed for
        // projection outputs.
        3 * hidden
    };
    // qkv is reused as attention context after split. Q reuses the gate
    // buffer and K reuses the up buffer until MLP starts, so only V needs a
    // separate KV-width scratch buffer.
    let qkv_f32 = q_width + 2 * kv_width + kv_width;
    let f32_elements = chunk * (hidden_f32 + qkv_f32 + intermediate_f32);
    let mxfp4_hidden = if cutlass_prefill {
        0
    } else {
        chunk * mxfp4_vector_bytes_estimate(graph.hidden_size) as u64
    };
    let mxfp4_intermediate = if cutlass_prefill {
        0
    } else {
        chunk
            * mxfp4_vector_bytes_estimate(graph.intermediate_size.unwrap_or(graph.hidden_size))
                as u64
    };
    let metadata_u32 = chunk * 3 + 3;
    let token_bytes = metadata_u32 * std::mem::size_of::<u32>() as u64;
    let split_attention_enabled =
        std::env::var_os("AEGISLLM_CUDA_DISABLE_SPLIT_K_ATTENTION").is_none()
            && std::env::var_os("AEGISLLM_CUDA_EXPERIMENTAL_SPLIT_K_ATTENTION").is_some();
    let split_attention_f32 = if split_attention_enabled {
        let q_block = 16_u64;
        let split_tokens = CUDA_PREFILL_DENSE_SPLIT_K_TOKENS as u64;
        let q_blocks = chunk.div_ceil(q_block);
        let splits = (placement.kv_cache.context_size as u64)
            .div_ceil(split_tokens)
            .max(1);
        let rows = q_blocks * graph.num_attention_heads as u64 * splits * q_block;
        rows * (graph.head_dim as u64 + 2)
    } else {
        0
    };
    Some((
        device,
        (f32_elements + split_attention_f32) * std::mem::size_of::<f32>() as u64
            + mxfp4_hidden
            + mxfp4_intermediate
            + token_bytes,
    ))
}

fn mxfp4_vector_bytes_estimate(len: usize) -> usize {
    (len / 64) * 36
}

fn effective_layout_extra_bytes(
    kernel: &crate::planning::runtime::KernelPlan,
    cuda: CudaRuntimeConfig,
) -> u64 {
    if kernel.family == KernelFamily::CudaNativeFp4TensorCores
        && !(cuda.native_mxfp4_repack || cuda.cutlass_nvfp4_repack)
    {
        return 0;
    }
    kernel.linear_layout.extra_weight_bytes
}

fn pool_for_backend_extra(device: BackendKind) -> AllocationPool {
    match device {
        BackendKind::Cpu | BackendKind::Wgpu { .. } => AllocationPool::Ram,
        BackendKind::Cuda { device } => AllocationPool::Vram { device },
    }
}

fn pool_for_store(store: StoragePlacement) -> AllocationPool {
    match store {
        StoragePlacement::Ram => AllocationPool::Ram,
        StoragePlacement::Mmap => AllocationPool::Mmap,
        StoragePlacement::Vram { device } => AllocationPool::Vram { device },
    }
}

fn footprint_from(
    allocations: &[PlannedAllocation],
    transfers: &[PlannedTransfer],
    placement: &ResolvedPlacement,
) -> MemoryFootprint {
    let persistent_ram_bytes = allocations
        .iter()
        .filter(|a| a.pool == AllocationPool::Ram)
        .map(|a| a.bytes)
        .sum();
    let file_backed_mmap_bytes = allocations
        .iter()
        .filter(|a| a.pool == AllocationPool::Mmap)
        .map(|a| a.bytes)
        .sum();
    let persistent_vram_bytes = placement
        .budget
        .vram
        .iter()
        .map(|pool| {
            let bytes = allocations
                .iter()
                .filter(|a| {
                    a.pool
                        == AllocationPool::Vram {
                            device: pool.device,
                        }
                })
                .map(|a| a.bytes)
                .sum();
            (pool.device, bytes)
        })
        .collect();
    let peak_host_staging_bytes = transfers
        .iter()
        .filter(|t| matches!(t.policy, TransferPolicy::HostToDeviceEachUse))
        .map(|t| t.bytes)
        .max()
        .unwrap_or(0);
    let peak_device_staging_bytes = placement
        .budget
        .vram
        .iter()
        .map(|pool| {
            let bytes = transfers
                .iter()
                .filter(|t| transfer_targets_device(t, pool.device))
                .map(|t| t.bytes)
                .max()
                .unwrap_or(0);
            (pool.device, bytes)
        })
        .collect();

    MemoryFootprint {
        persistent_ram_bytes,
        file_backed_mmap_bytes,
        persistent_vram_bytes,
        peak_host_staging_bytes,
        peak_device_staging_bytes,
    }
}

fn transfer_targets_device(transfer: &PlannedTransfer, device: usize) -> bool {
    match (transfer.policy, transfer.source, transfer.compute) {
        (
            TransferPolicy::HostToDeviceEachUse | TransferPolicy::CrossDevice,
            _,
            ComputePlacement::Cuda {
                device: compute_device,
            },
        ) => compute_device == device,
        (
            TransferPolicy::DeviceToHostEachUse,
            StoragePlacement::Vram {
                device: source_device,
            },
            ComputePlacement::Cpu,
        ) => source_device == device,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{GraphRegionKind, RegionId};
    use crate::planning::placement::{
        ComputePlacement, KvCachePlacement, RegionPlacement, TransferPolicy,
    };
    use crate::planning::runtime::{KernelPlan, SyncPolicy, TensorResidency};
    use crate::tensor::layout::{LinearLayoutPlan, LinearResidentLayout, MaterializationPolicy};
    use crate::tensor::quant::{KvCacheQuantization, QuantFormat, WeightQuantization};

    #[test]
    fn native_fp4_layout_extra_counts_only_when_repack_is_enabled() {
        let placement = ResolvedPlacement {
            model: "m".into(),
            weight_quantization: WeightQuantization::Nvfp4,
            region_placements: vec![RegionPlacement {
                region_id: RegionId("layer.0".into()),
                kind: GraphRegionKind::TransformerBlock,
                layer_index: Some(0),
                weight_bytes: 100,
                store: StoragePlacement::Vram { device: 0 },
                compute: ComputePlacement::Cuda { device: 0 },
                transfer: TransferPolicy::None,
            }],
            kv_cache: KvCachePlacement {
                store: StoragePlacement::Vram { device: 0 },
                compute: ComputePlacement::Cuda { device: 0 },
                quantization: KvCacheQuantization::F16,
                context_size: 1,
                estimated_bytes: 10,
                first_n_layers: None,
                first_store: None,
            },
            budget: MemoryBudget {
                ram_total_bytes: 1024,
                ram_usable_bytes: 1024,
                vram: vec![MemoryPool {
                    device: 0,
                    total_bytes: 1024,
                    usable_bytes: 1024,
                }],
            },
            linear_layout: crate::tensor::layout::LinearLayoutPolicy::default(),
            warnings: Vec::new(),
            attention_quantization: crate::planning::placement::WeightQuantOverride::Default,
            shared_mlp_quantization: crate::planning::placement::WeightQuantOverride::Default,
        };
        let runtime = RuntimePlan {
            kernels: vec![KernelPlan {
                name: "layer.0".into(),
                device: BackendKind::Cuda { device: 0 },
                quant_format: QuantFormat::Nvfp4,
                linear_layout: LinearLayoutPlan {
                    source_format: QuantFormat::Nvfp4,
                    resident_layout: LinearResidentLayout::NativeTensorCore,
                    materialization: MaterializationPolicy::Lazy,
                    extra_weight_bytes: 50,
                    notes: Vec::new(),
                },
                family: KernelFamily::CudaNativeFp4TensorCores,
                residency: TensorResidency::Device,
                sync: SyncPolicy::StreamOrdered,
            }],
            warnings: Vec::new(),
        };

        let disabled = MemoryPlan::from_placement_runtime_and_cuda(
            &placement,
            Some(&runtime),
            CudaRuntimeConfig {
                native_mxfp4_repack: false,
                cutlass_nvfp4_repack: false,
                native_mxfp4_inference: false,
                prefill_attention: Default::default(),
                prefill_chunk_size: None,
                prefill_stage_timings: false,
            },
        );
        let enabled = MemoryPlan::from_placement_runtime_and_cuda(
            &placement,
            Some(&runtime),
            CudaRuntimeConfig {
                native_mxfp4_repack: true,
                cutlass_nvfp4_repack: false,
                native_mxfp4_inference: false,
                prefill_attention: Default::default(),
                prefill_chunk_size: None,
                prefill_stage_timings: false,
            },
        );
        let cutlass_sidecar = MemoryPlan::from_placement_runtime_and_cuda(
            &placement,
            Some(&runtime),
            CudaRuntimeConfig {
                native_mxfp4_repack: false,
                cutlass_nvfp4_repack: true,
                native_mxfp4_inference: false,
                prefill_attention: Default::default(),
                prefill_chunk_size: None,
                prefill_stage_timings: false,
            },
        );

        assert_eq!(
            disabled.bytes_in_pool(AllocationPool::Vram { device: 0 }),
            110
        );
        assert_eq!(
            enabled.bytes_in_pool(AllocationPool::Vram { device: 0 }),
            160
        );
        assert_eq!(
            cutlass_sidecar.bytes_in_pool(AllocationPool::Vram { device: 0 }),
            160
        );
    }
}

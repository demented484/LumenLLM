mod attention;
pub(super) mod block;
mod forward;
mod full;
mod linear_ops;
mod loader;
mod mlp;
mod planning;
mod prefill;
mod provider;
mod rope;
mod state;

pub(super) use planning::cuda_kernel_limitations;
pub use provider::CudaExecutorProvider;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendKind;
    use crate::executor::cuda::planning::validate_cuda_placement;
    use crate::planning::placement::{
        ComputePlacement, KvCachePlacement, RegionPlacement, ResolvedPlacement, StoragePlacement,
        TransferPolicy,
    };
    use crate::planning::runtime::RuntimePlan;
    use crate::planning::runtime::{KernelFamily, KernelPlan, SyncPolicy, TensorResidency};
    use crate::tensor::layout::LinearResidentLayout;
    use crate::tensor::quant::{KvCacheQuantization, WeightQuantization};

    #[test]
    fn cuda_provider_is_runnable_with_reference_kernel_fallback() {
        let placement = ResolvedPlacement {
            model: "m".into(),
            weight_quantization: WeightQuantization::Nvfp4,
            region_placements: vec![RegionPlacement {
                region_id: crate::graph::RegionId("layer.0".into()),
                kind: crate::graph::GraphRegionKind::TransformerBlock,
                layer_index: Some(0),
                weight_bytes: 1,
                store: StoragePlacement::Vram { device: 0 },
                compute: ComputePlacement::Cuda { device: 0 },
                transfer: TransferPolicy::None,
            }],
            kv_cache: KvCachePlacement {
                store: StoragePlacement::Vram { device: 0 },
                compute: ComputePlacement::Cuda { device: 0 },
                quantization: KvCacheQuantization::F16,
                context_size: 1,
                estimated_bytes: 1,
            },
            budget: crate::planning::memory::MemoryBudget {
                ram_total_bytes: 1,
                ram_usable_bytes: 1,
                vram: vec![crate::planning::memory::MemoryPool {
                    device: 0,
                    total_bytes: 1,
                    usable_bytes: 1,
                }],
            },
            linear_layout: crate::tensor::layout::LinearLayoutPolicy::default(),
            warnings: Vec::new(),
        };
        let runtime = RuntimePlan {
            kernels: vec![KernelPlan {
                name: "layer.0".into(),
                device: BackendKind::Cuda { device: 0 },
                quant_format: crate::tensor::quant::QuantFormat::Nvfp4,
                linear_layout: crate::tensor::layout::LinearLayoutPlan {
                    source_format: crate::tensor::quant::QuantFormat::Nvfp4,
                    resident_layout: LinearResidentLayout::NativeTensorCore,
                    materialization: crate::tensor::layout::MaterializationPolicy::Lazy,
                    extra_weight_bytes: 0,
                    notes: Vec::new(),
                },
                family: KernelFamily::CudaNativeFp4TensorCores,
                residency: TensorResidency::Device,
                sync: SyncPolicy::StreamOrdered,
            }],
            warnings: Vec::new(),
        };

        let plan = CudaExecutorProvider::plan(&placement, &runtime).unwrap();
        assert!(plan.runnable);
        assert!(plan.limitations.is_empty());
    }

    #[test]
    fn cuda_generate_accepts_host_stored_regions_with_eager_upload() {
        let placement = ResolvedPlacement {
            model: "m".into(),
            weight_quantization: WeightQuantization::Nvfp4,
            region_placements: vec![RegionPlacement {
                region_id: crate::graph::RegionId("layer.0".into()),
                kind: crate::graph::GraphRegionKind::TransformerBlock,
                layer_index: Some(0),
                weight_bytes: 1,
                store: StoragePlacement::Mmap,
                compute: ComputePlacement::Cuda { device: 0 },
                transfer: TransferPolicy::HostToDeviceEachUse,
            }],
            kv_cache: KvCachePlacement {
                store: StoragePlacement::Vram { device: 0 },
                compute: ComputePlacement::Cuda { device: 0 },
                quantization: KvCacheQuantization::F16,
                context_size: 1,
                estimated_bytes: 1,
            },
            budget: crate::planning::memory::MemoryBudget {
                ram_total_bytes: 1,
                ram_usable_bytes: 1,
                vram: vec![crate::planning::memory::MemoryPool {
                    device: 0,
                    total_bytes: 1,
                    usable_bytes: 1,
                }],
            },
            linear_layout: crate::tensor::layout::LinearLayoutPolicy::default(),
            warnings: Vec::new(),
        };

        let runtime = RuntimePlan {
            kernels: Vec::new(),
            warnings: Vec::new(),
        };
        let plan = CudaExecutorProvider::plan(&placement, &runtime).unwrap();
        assert!(plan.runnable);
        assert!(plan.limitations.is_empty());

        validate_cuda_placement(&placement, 0).unwrap();
    }
}

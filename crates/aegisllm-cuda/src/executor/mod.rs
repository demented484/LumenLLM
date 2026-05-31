mod attention;
pub mod block;
mod cache_cleanup;
mod forward;
mod full;
mod gdn;
pub mod layer_capture;
mod linear_ops;
mod load_progress;
mod loader;
mod mlp;
mod mtp;
mod planning;
mod ple;
mod prefill;
pub mod prefix_cache;
mod provider;
mod rope;
mod speculative;
pub(crate) mod state;
pub mod vision;
pub mod audio;

pub use planning::cuda_kernel_limitations;
pub use provider::CudaExecutorProvider;

#[cfg(test)]
mod tests {
    use super::*;
    use aegisllm_base::backend::BackendKind;
    use crate::executor::planning::validate_cuda_placement;
    use aegisllm_base::planning::placement::{
        ComputePlacement, KvCachePlacement, RegionPlacement, ResolvedPlacement, StoragePlacement,
        TransferPolicy,
    };
    use aegisllm_base::planning::runtime::RuntimePlan;
    use aegisllm_base::planning::runtime::{KernelFamily, KernelPlan, SyncPolicy, TensorResidency};
    use aegisllm_base::tensor::layout::LinearResidentLayout;
    use aegisllm_base::tensor::quant::{KvCacheQuantization, WeightQuantization};

    #[test]
    fn cuda_provider_is_runnable_with_reference_kernel_fallback() {
        let placement = ResolvedPlacement {
            model: "m".into(),
            weight_quantization: WeightQuantization::Nvfp4,
            region_placements: vec![RegionPlacement {
                region_id: aegisllm_base::graph::RegionId("layer.0".into()),
                kind: aegisllm_base::graph::GraphRegionKind::TransformerBlock,
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
                first_n_layers: None,
                first_store: None,
            },
            budget: aegisllm_base::planning::memory::MemoryBudget {
                ram_total_bytes: 1,
                ram_usable_bytes: 1,
                vram: vec![aegisllm_base::planning::memory::MemoryPool {
                    device: 0,
                    total_bytes: 1,
                    usable_bytes: 1,
                }],
            },
            linear_layout: aegisllm_base::tensor::layout::LinearLayoutPolicy::default(),
            warnings: Vec::new(),
            attention_quantization: aegisllm_base::planning::placement::WeightQuantOverride::Default,
            shared_mlp_quantization: aegisllm_base::planning::placement::WeightQuantOverride::Default,
            attention_store_override: None,
        };
        let runtime = RuntimePlan {
            kernels: vec![KernelPlan {
                name: "layer.0".into(),
                device: BackendKind::Cuda { device: 0 },
                quant_format: aegisllm_base::tensor::quant::QuantFormat::Nvfp4,
                linear_layout: aegisllm_base::tensor::layout::LinearLayoutPlan {
                    source_format: aegisllm_base::tensor::quant::QuantFormat::Nvfp4,
                    resident_layout: LinearResidentLayout::NativeTensorCore,
                    materialization: aegisllm_base::tensor::layout::MaterializationPolicy::Lazy,
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
                region_id: aegisllm_base::graph::RegionId("layer.0".into()),
                kind: aegisllm_base::graph::GraphRegionKind::TransformerBlock,
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
                first_n_layers: None,
                first_store: None,
            },
            budget: aegisllm_base::planning::memory::MemoryBudget {
                ram_total_bytes: 1,
                ram_usable_bytes: 1,
                vram: vec![aegisllm_base::planning::memory::MemoryPool {
                    device: 0,
                    total_bytes: 1,
                    usable_bytes: 1,
                }],
            },
            linear_layout: aegisllm_base::tensor::layout::LinearLayoutPolicy::default(),
            warnings: Vec::new(),
            attention_quantization: aegisllm_base::planning::placement::WeightQuantOverride::Default,
            shared_mlp_quantization: aegisllm_base::planning::placement::WeightQuantOverride::Default,
            attention_store_override: None,
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

mod activation;
mod kv;
mod plan;
mod primitive;
mod region;
mod transfer;

pub use activation::{ActivationResidency, ActivationTransferNode};
pub use kv::KvCacheNode;
pub use plan::{ExecutionNode, ExecutorGraphPlan};
pub use primitive::{BackendPrimitiveKind, BackendPrimitiveNode, BackendPrimitivePlan};
pub use region::RegionExecutionNode;
pub use transfer::WeightTransferNode;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{GraphRegionKind, RegionId};
    use crate::planning::memory::{MemoryBudget, MemoryPool};
    use crate::planning::placement::{
        ComputePlacement, KvCachePlacement, RegionPlacement, ResolvedPlacement, StoragePlacement,
        TransferPolicy,
    };
    use crate::tensor::layout::LinearLayoutPolicy;
    use crate::tensor::quant::{KvCacheQuantization, WeightQuantization};

    #[test]
    fn executor_graph_plan_emits_activation_weight_and_kv_nodes() {
        let placement = ResolvedPlacement {
            model: "m".into(),
            weight_quantization: WeightQuantization::Nvfp4,
            region_placements: vec![
                RegionPlacement {
                    region_id: RegionId("embed".into()),
                    kind: GraphRegionKind::TokenEmbedding,
                    layer_index: None,
                    weight_bytes: 1,
                    store: StoragePlacement::Ram,
                    compute: ComputePlacement::Cpu,
                    transfer: TransferPolicy::None,
                },
                RegionPlacement {
                    region_id: RegionId("layer.0".into()),
                    kind: GraphRegionKind::TransformerBlock,
                    layer_index: Some(0),
                    weight_bytes: 1,
                    store: StoragePlacement::Mmap,
                    compute: ComputePlacement::Cuda { device: 0 },
                    transfer: TransferPolicy::HostToDeviceEachUse,
                },
            ],
            kv_cache: KvCachePlacement {
                store: StoragePlacement::Ram,
                compute: ComputePlacement::Cpu,
                quantization: KvCacheQuantization::F16,
                context_size: 4,
                estimated_bytes: 1,
            },
            budget: MemoryBudget {
                ram_total_bytes: 1,
                ram_usable_bytes: 1,
                vram: vec![MemoryPool {
                    device: 0,
                    total_bytes: 1,
                    usable_bytes: 1,
                }],
            },
            linear_layout: LinearLayoutPolicy::default(),
            warnings: Vec::new(),
        };

        let plan = ExecutorGraphPlan::from_resolved_placement(&placement);
        assert!(plan.kv_cache().is_some());
        assert_eq!(plan.activation_transfers().count(), 1);
        assert_eq!(plan.weight_transfers().count(), 1);
        let primitives = plan.backend_primitives();
        assert_eq!(primitives.count(BackendPrimitiveKind::KvCache), 1);
        assert_eq!(primitives.count(BackendPrimitiveKind::WeightTransfer), 1);
        assert_eq!(
            primitives.count(BackendPrimitiveKind::ActivationTransfer),
            1
        );
        assert!(matches!(plan.nodes[0], ExecutionNode::KvCache(_)));
        assert!(matches!(plan.nodes[1], ExecutionNode::Region(_)));
        assert!(matches!(plan.nodes[2], ExecutionNode::WeightTransfer(_)));
        assert!(matches!(
            plan.nodes[3],
            ExecutionNode::ActivationTransfer(_)
        ));
        assert!(matches!(plan.nodes[4], ExecutionNode::Region(_)));
        assert_eq!(
            plan.cuda_regions_with_host_kv(&placement.region_placements),
            1
        );
    }
}

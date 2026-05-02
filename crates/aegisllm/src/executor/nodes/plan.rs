use aegisllm_base::planning::placement::{ComputePlacement, ResolvedPlacement, TransferPolicy};

use super::{
    ActivationResidency, ActivationTransferNode, BackendPrimitiveNode, BackendPrimitivePlan,
    KvCacheNode, RegionExecutionNode, WeightTransferNode,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionNode {
    Region(RegionExecutionNode),
    ActivationTransfer(ActivationTransferNode),
    WeightTransfer(WeightTransferNode),
    KvCache(KvCacheNode),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutorGraphPlan {
    pub nodes: Vec<ExecutionNode>,
}

impl ExecutorGraphPlan {
    pub fn from_resolved_placement(placement: &ResolvedPlacement) -> Self {
        let mut nodes = Vec::new();
        nodes.push(ExecutionNode::KvCache(KvCacheNode::from_placement(
            &placement.kv_cache,
        )));

        let mut previous_region: Option<&crate::planning::placement::RegionPlacement> = None;
        for region in &placement.region_placements {
            if region.transfer != TransferPolicy::None {
                nodes.push(ExecutionNode::WeightTransfer(WeightTransferNode {
                    region_id: region.region_id.clone(),
                    store: region.store,
                    compute: region.compute,
                    transfer: region.transfer,
                }));
            }
            if let Some(previous) = previous_region {
                let from = ActivationResidency::from_compute(previous.compute);
                let to = ActivationResidency::from_compute(region.compute);
                if from != to {
                    nodes.push(ExecutionNode::ActivationTransfer(ActivationTransferNode {
                        after_region: previous.region_id.clone(),
                        from,
                        to,
                    }));
                }
            }
            nodes.push(ExecutionNode::Region(RegionExecutionNode {
                region_id: region.region_id.clone(),
                kind: region.kind,
                compute: region.compute,
            }));
            previous_region = Some(region);
        }

        Self { nodes }
    }

    pub fn activation_transfers(&self) -> impl Iterator<Item = &ActivationTransferNode> {
        self.nodes.iter().filter_map(|node| match node {
            ExecutionNode::ActivationTransfer(node) => Some(node),
            _ => None,
        })
    }

    pub fn weight_transfers(&self) -> impl Iterator<Item = &WeightTransferNode> {
        self.nodes.iter().filter_map(|node| match node {
            ExecutionNode::WeightTransfer(node) => Some(node),
            _ => None,
        })
    }

    pub fn kv_cache(&self) -> Option<&KvCacheNode> {
        self.nodes.iter().find_map(|node| match node {
            ExecutionNode::KvCache(node) => Some(node),
            _ => None,
        })
    }

    pub fn cuda_regions_with_host_kv(
        &self,
        regions: &[crate::planning::placement::RegionPlacement],
    ) -> usize {
        let Some(kv) = self.kv_cache() else {
            return 0;
        };
        if kv.residency() != ActivationResidency::Host {
            return 0;
        }
        regions
            .iter()
            .filter(|region| matches!(region.compute, ComputePlacement::Cuda { .. }))
            .count()
    }

    pub fn backend_primitives(&self) -> BackendPrimitivePlan {
        BackendPrimitivePlan {
            primitives: self.nodes.iter().map(BackendPrimitiveNode::from).collect(),
        }
    }
}

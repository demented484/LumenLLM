use aegisllm_base::graph::RegionId;
use aegisllm_base::planning::placement::{ComputePlacement, StoragePlacement, TransferPolicy};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightTransferNode {
    pub region_id: RegionId,
    pub store: StoragePlacement,
    pub compute: ComputePlacement,
    pub transfer: TransferPolicy,
}

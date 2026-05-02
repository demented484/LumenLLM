use aegisllm_base::graph::{GraphRegionKind, RegionId};
use aegisllm_base::planning::placement::ComputePlacement;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionExecutionNode {
    pub region_id: RegionId,
    pub kind: GraphRegionKind,
    pub compute: ComputePlacement,
}

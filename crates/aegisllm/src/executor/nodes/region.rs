use crate::graph::{GraphRegionKind, RegionId};
use crate::planning::placement::ComputePlacement;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionExecutionNode {
    pub region_id: RegionId,
    pub kind: GraphRegionKind,
    pub compute: ComputePlacement,
}

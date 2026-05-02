use aegisllm_base::graph::RegionId;
use aegisllm_base::planning::placement::ComputePlacement;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivationResidency {
    Host,
    Device { device: usize },
}

impl ActivationResidency {
    pub fn from_compute(compute: ComputePlacement) -> Self {
        match compute {
            ComputePlacement::Cpu => Self::Host,
            ComputePlacement::Cuda { device } => Self::Device { device },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivationTransferNode {
    pub after_region: RegionId,
    pub from: ActivationResidency,
    pub to: ActivationResidency,
}

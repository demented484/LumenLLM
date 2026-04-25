use super::{
    ActivationTransferNode, ExecutionNode, KvCacheNode, RegionExecutionNode, WeightTransferNode,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendPrimitiveKind {
    Region,
    ActivationTransfer,
    WeightTransfer,
    KvCache,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendPrimitiveNode {
    Region(RegionExecutionNode),
    ActivationTransfer(ActivationTransferNode),
    WeightTransfer(WeightTransferNode),
    KvCache(KvCacheNode),
}

impl BackendPrimitiveNode {
    pub fn kind(&self) -> BackendPrimitiveKind {
        match self {
            Self::Region(_) => BackendPrimitiveKind::Region,
            Self::ActivationTransfer(_) => BackendPrimitiveKind::ActivationTransfer,
            Self::WeightTransfer(_) => BackendPrimitiveKind::WeightTransfer,
            Self::KvCache(_) => BackendPrimitiveKind::KvCache,
        }
    }
}

impl From<&ExecutionNode> for BackendPrimitiveNode {
    fn from(node: &ExecutionNode) -> Self {
        match node {
            ExecutionNode::Region(node) => Self::Region(node.clone()),
            ExecutionNode::ActivationTransfer(node) => Self::ActivationTransfer(node.clone()),
            ExecutionNode::WeightTransfer(node) => Self::WeightTransfer(node.clone()),
            ExecutionNode::KvCache(node) => Self::KvCache(node.clone()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendPrimitivePlan {
    pub primitives: Vec<BackendPrimitiveNode>,
}

impl BackendPrimitivePlan {
    pub fn count(&self, kind: BackendPrimitiveKind) -> usize {
        self.primitives
            .iter()
            .filter(|primitive| primitive.kind() == kind)
            .count()
    }
}

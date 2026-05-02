use aegisllm_base::planning::placement::{ComputePlacement, KvCachePlacement, StoragePlacement};
use aegisllm_base::tensor::quant::KvCacheQuantization;

use super::ActivationResidency;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvCacheNode {
    pub store: StoragePlacement,
    pub compute: ComputePlacement,
    pub quantization: KvCacheQuantization,
    pub context_size: usize,
}

impl KvCacheNode {
    pub fn from_placement(kv: &KvCachePlacement) -> Self {
        Self {
            store: kv.store,
            compute: kv.compute,
            quantization: kv.quantization,
            context_size: kv.context_size,
        }
    }

    pub fn residency(&self) -> ActivationResidency {
        ActivationResidency::from_compute(self.compute)
    }
}

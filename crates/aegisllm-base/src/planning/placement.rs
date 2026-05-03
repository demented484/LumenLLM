use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};

use crate::error::{AegisError, Result};
use crate::graph::{GraphRegion, GraphRegionKind, ModelGraph, RegionId};
use crate::hardware::{ComputeDevice, HardwareInventory};
use crate::planning::memory::MemoryBudget;
use crate::tensor::layout::LinearLayoutPolicy;
use crate::tensor::quant::{KvCacheQuantization, WeightQuantization};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StorageTier {
    Ram,
    Vram { device: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StoragePlacement {
    Ram,
    Vram { device: usize },
    Mmap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComputePlacement {
    Cpu,
    Cuda { device: usize },
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlacementPolicy {
    pub weights_store: StoragePlacement,
    pub weights_compute: ComputePlacement,
    pub spill_store: StoragePlacement,
    pub spill_compute: ComputePlacement,
    pub kv_store: StoragePlacement,
    pub kv_compute: ComputePlacement,
    pub kv_quantization: KvCacheQuantization,
    pub context_size: usize,
    pub reserve_ram_bytes: u64,
    pub reserve_vram_bytes: u64,
    pub linear_layout: LinearLayoutPolicy,
    pub rules: Vec<PlacementRule>,
    /// First `kv_vram_layers` layers keep KV in VRAM; remaining use `kv_store`.
    pub kv_vram_layers: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlacementRule {
    pub selector: LayerSelector,
    pub store: Option<StoragePlacement>,
    pub compute: Option<ComputePlacement>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerSelector {
    FirstN { n: usize },
    LastN { n: usize },
    Range { start: usize, end: usize },
    Kind(GraphRegionKind),
    Region(String),
    All,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedPlacement {
    pub model: String,
    pub weight_quantization: WeightQuantization,
    pub region_placements: Vec<RegionPlacement>,
    pub kv_cache: KvCachePlacement,
    pub budget: MemoryBudget,
    pub linear_layout: LinearLayoutPolicy,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RegionPlacement {
    pub region_id: RegionId,
    pub kind: GraphRegionKind,
    pub layer_index: Option<usize>,
    pub weight_bytes: u64,
    pub store: StoragePlacement,
    pub compute: ComputePlacement,
    pub transfer: TransferPolicy,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KvCachePlacement {
    pub store: StoragePlacement,
    pub compute: ComputePlacement,
    pub quantization: KvCacheQuantization,
    pub context_size: usize,
    pub estimated_bytes: u64,
    /// First `vram_layers` layers keep KV in VRAM; layers >= this index use `store`.
    pub vram_layers: Option<usize>,
}

impl KvCachePlacement {
    /// Returns the resolved KV storage for the given layer index.
    /// If `vram_layers` is set, layers 0..vram_layers use VRAM; the rest use `store`.
    pub fn store_for_layer(&self, layer_idx: usize) -> StoragePlacement {
        match self.vram_layers {
            Some(n) if layer_idx < n => match self.compute {
                ComputePlacement::Cuda { device } => StoragePlacement::Vram { device },
                _ => self.store,
            },
            _ => self.store,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferPolicy {
    None,
    HostToDeviceEachUse,
    DeviceToHostEachUse,
    CrossDevice,
}

impl PlacementPolicy {
    pub fn auto_for(inventory: &HardwareInventory) -> Self {
        let cuda = inventory.preferred_cuda().map(|gpu| gpu.index);
        match cuda {
            Some(device) => Self {
                weights_store: StoragePlacement::Vram { device },
                weights_compute: ComputePlacement::Cuda { device },
                spill_store: StoragePlacement::Mmap,
                spill_compute: ComputePlacement::Cuda { device },
                kv_store: StoragePlacement::Vram { device },
                kv_compute: ComputePlacement::Cuda { device },
                kv_quantization: KvCacheQuantization::F16,
                context_size: 8192,
                reserve_ram_bytes: 4 * 1024 * 1024 * 1024,
                reserve_vram_bytes: 1024 * 1024 * 1024,
                linear_layout: LinearLayoutPolicy::default(),
                rules: Vec::new(),
                kv_vram_layers: None,
            },
            None => Self {
                weights_store: StoragePlacement::Mmap,
                weights_compute: ComputePlacement::Cpu,
                spill_store: StoragePlacement::Mmap,
                spill_compute: ComputePlacement::Cpu,
                kv_store: StoragePlacement::Ram,
                kv_compute: ComputePlacement::Cpu,
                kv_quantization: KvCacheQuantization::F16,
                context_size: 8192,
                reserve_ram_bytes: 2 * 1024 * 1024 * 1024,
                reserve_vram_bytes: 0,
                linear_layout: LinearLayoutPolicy::default(),
                rules: Vec::new(),
                kv_vram_layers: None,
            },
        }
    }
}

impl ResolvedPlacement {
    pub fn plan(
        model_name: impl Into<String>,
        graph: &ModelGraph,
        inventory: &HardwareInventory,
        policy: &PlacementPolicy,
    ) -> Result<Self> {
        if policy.context_size == 0 {
            return Err(AegisError::InvalidPlan(
                "kv cache context_size must be greater than zero".into(),
            ));
        }
        let mut warnings = Vec::new();
        let budget = MemoryBudget::from_inventory(inventory, policy);
        let kv_cache = KvCachePlacement {
            store: policy.kv_store,
            compute: policy.kv_compute,
            quantization: policy.kv_quantization,
            context_size: policy.context_size,
            estimated_bytes: estimate_kv_cache_bytes(graph, policy),
            vram_layers: policy.kv_vram_layers,
        };
        let mut region_placements = graph
            .regions
            .iter()
            .map(|region| {
                let (store, compute) = apply_rules(region, graph.num_layers, policy);
                RegionPlacement {
                    region_id: region.id.clone(),
                    kind: region.kind,
                    layer_index: region.layer_index,
                    weight_bytes: region.weight_bytes(),
                    store,
                    compute,
                    transfer: transfer_policy(store, compute),
                }
            })
            .collect::<Vec<_>>();

        trim_vram_to_budget(
            &mut region_placements,
            &kv_cache,
            &budget,
            policy,
            &mut warnings,
        );

        Ok(Self {
            model: model_name.into(),
            weight_quantization: graph.weight_quantization,
            region_placements,
            kv_cache,
            budget,
            linear_layout: policy.linear_layout.clone(),
            warnings,
        })
    }

    pub fn region_map(&self) -> BTreeMap<&RegionId, &RegionPlacement> {
        self.region_placements
            .iter()
            .map(|placement| (&placement.region_id, placement))
            .collect()
    }

    pub fn vram_weight_bytes(&self, device: usize) -> u64 {
        self.region_placements
            .iter()
            .filter(|p| p.store == StoragePlacement::Vram { device })
            .map(|p| p.weight_bytes)
            .sum()
    }

    pub fn ram_weight_bytes(&self) -> u64 {
        self.region_placements
            .iter()
            .filter(|p| matches!(p.store, StoragePlacement::Ram | StoragePlacement::Mmap))
            .map(|p| p.weight_bytes)
            .sum()
    }
}

impl Display for StoragePlacement {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ram => f.write_str("ram"),
            Self::Mmap => f.write_str("mmap"),
            Self::Vram { device } => write!(f, "vram:{device}"),
        }
    }
}

impl Display for ComputePlacement {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => f.write_str("cpu"),
            Self::Cuda { device } => write!(f, "cuda:{device}"),
        }
    }
}

fn apply_rules(
    region: &GraphRegion,
    num_layers: usize,
    policy: &PlacementPolicy,
) -> (StoragePlacement, ComputePlacement) {
    let mut store = policy.weights_store;
    let mut compute = policy.weights_compute;
    for rule in &policy.rules {
        if selector_matches(&rule.selector, region, num_layers) {
            if let Some(next) = rule.store {
                store = next;
            }
            if let Some(next) = rule.compute {
                compute = next;
            }
        }
    }
    (store, compute)
}

fn selector_matches(selector: &LayerSelector, region: &GraphRegion, num_layers: usize) -> bool {
    match selector {
        LayerSelector::All => true,
        LayerSelector::Kind(kind) => region.kind == *kind,
        LayerSelector::Region(id) => region.id.0 == *id,
        LayerSelector::FirstN { n } => region.layer_index.is_some_and(|i| i < *n),
        LayerSelector::LastN { n } => region
            .layer_index
            .is_some_and(|i| i >= num_layers.saturating_sub(*n)),
        LayerSelector::Range { start, end } => {
            region.layer_index.is_some_and(|i| i >= *start && i < *end)
        }
    }
}

fn transfer_policy(store: StoragePlacement, compute: ComputePlacement) -> TransferPolicy {
    match (store, compute) {
        (StoragePlacement::Ram | StoragePlacement::Mmap, ComputePlacement::Cpu) => {
            TransferPolicy::None
        }
        (StoragePlacement::Vram { device: a }, ComputePlacement::Cuda { device: b }) if a == b => {
            TransferPolicy::None
        }
        (StoragePlacement::Ram | StoragePlacement::Mmap, ComputePlacement::Cuda { .. }) => {
            TransferPolicy::HostToDeviceEachUse
        }
        (StoragePlacement::Vram { .. }, ComputePlacement::Cpu) => {
            TransferPolicy::DeviceToHostEachUse
        }
        (StoragePlacement::Vram { .. }, ComputePlacement::Cuda { .. }) => {
            TransferPolicy::CrossDevice
        }
    }
}

fn estimate_kv_cache_bytes(graph: &ModelGraph, policy: &PlacementPolicy) -> u64 {
    let elem = (policy.kv_quantization.bytes_per_element() * 2.0).ceil() as u64;
    let values = graph
        .num_layers
        .saturating_mul(2)
        .saturating_mul(graph.num_kv_heads)
        .saturating_mul(graph.head_dim)
        .saturating_mul(policy.context_size) as u64;
    values.saturating_mul(elem).div_ceil(2)
}

fn trim_vram_to_budget(
    placements: &mut [RegionPlacement],
    kv_cache: &KvCachePlacement,
    budget: &MemoryBudget,
    policy: &PlacementPolicy,
    warnings: &mut Vec<String>,
) {
    for pool in &budget.vram {
        let device = pool.device;
        let usable = pool.usable_bytes;
        let runtime_vram = match kv_cache.store {
            StoragePlacement::Vram { device: kv_device } if kv_device == device => {
                kv_cache.estimated_bytes
            }
            _ => 0,
        };
        let mut used = runtime_vram
            + placements
                .iter()
                .filter(|p| p.store == (StoragePlacement::Vram { device }))
                .map(|p| p.weight_bytes)
                .sum::<u64>();
        if used <= usable {
            continue;
        }

        for placement in placements.iter_mut().rev() {
            if used <= usable {
                break;
            }
            if placement.store == (StoragePlacement::Vram { device }) {
                used = used.saturating_sub(placement.weight_bytes);
                placement.store = policy.spill_store;
                placement.compute = policy.spill_compute;
                placement.transfer = transfer_policy(placement.store, placement.compute);
            }
        }
        warnings.push(format!(
            "vram:{device} budget forced storage spill: final_persistent={} usable={} runtime_reserved={}",
            used, usable, runtime_vram
        ));
    }
}

impl From<ComputePlacement> for ComputeDevice {
    fn from(value: ComputePlacement) -> Self {
        match value {
            ComputePlacement::Cpu => Self::Cpu,
            ComputePlacement::Cuda { device } => Self::Cuda { index: device },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{CpuInfo, GpuArchitecture, GpuInfo};
    use crate::tensor::quant::WeightQuantization;
    use crate::tensor::{TensorDType, TensorInfo};
    use std::path::PathBuf;

    #[test]
    fn selectors_match_first_n_layers_only() {
        let region = GraphRegion {
            id: RegionId("layer.1".into()),
            kind: GraphRegionKind::TransformerBlock,
            layer_index: Some(1),
            tensors: Vec::new(),
        };
        assert!(selector_matches(
            &LayerSelector::FirstN { n: 2 },
            &region,
            32
        ));
        assert!(!selector_matches(
            &LayerSelector::FirstN { n: 1 },
            &region,
            32
        ));
    }

    #[test]
    fn auto_cuda_policy_places_all_runtime_regions_on_vram() {
        let inventory = HardwareInventory {
            cpu: CpuInfo {
                model_name: "test-cpu".into(),
                physical_cores: 1,
                logical_threads: 1,
                ram_total_bytes: 32 * 1024 * 1024 * 1024,
                ram_available_bytes: Some(32 * 1024 * 1024 * 1024),
                avx2: false,
                avx512: false,
                bf16: false,
            },
            gpus: vec![GpuInfo {
                index: 0,
                name: "test-blackwell".into(),
                driver_version: "test".into(),
                compute_capability: Some("12.0".into()),
                vram_total_bytes: 16 * 1024 * 1024 * 1024,
                vram_free_bytes: Some(16 * 1024 * 1024 * 1024),
                architecture: GpuArchitecture::Blackwell,
            }],
        };

        let policy = PlacementPolicy::auto_for(&inventory);
        assert_eq!(policy.weights_store, StoragePlacement::Vram { device: 0 });
        assert_eq!(policy.weights_compute, ComputePlacement::Cuda { device: 0 });
        assert!(policy.rules.is_empty());
    }

    #[test]
    fn vram_spill_applies_spill_compute_policy() {
        let inventory = HardwareInventory {
            cpu: CpuInfo {
                model_name: "test-cpu".into(),
                physical_cores: 1,
                logical_threads: 1,
                ram_total_bytes: 1024,
                ram_available_bytes: Some(1024),
                avx2: false,
                avx512: false,
                bf16: false,
            },
            gpus: vec![GpuInfo {
                index: 0,
                name: "tiny-gpu".into(),
                driver_version: "test".into(),
                compute_capability: Some("12.0".into()),
                vram_total_bytes: 64,
                vram_free_bytes: Some(64),
                architecture: GpuArchitecture::Blackwell,
            }],
        };
        let graph = ModelGraph {
            model_type: "llama".into(),
            architecture: "llama".into(),
            hidden_size: 1,
            intermediate_size: None,
            num_layers: 1,
            num_attention_heads: 1,
            num_kv_heads: 1,
            head_dim: 1,
            vocab_size: None,
            weight_quantization: WeightQuantization::Nvfp4,
            regions: vec![GraphRegion {
                id: RegionId("layer.0".into()),
                kind: GraphRegionKind::TransformerBlock,
                layer_index: Some(0),
                tensors: vec![crate::graph::GraphTensor {
                    role: crate::graph::TensorRole::Query,
                    info: TensorInfo {
                        name: "model.layers.0.self_attn.q_proj.weight".into(),
                        dtype: TensorDType::U8,
                        shape: vec![100],
                        num_elements: 100,
                        data_offsets: (0, 100),
                        file_offsets: (0, 100),
                        shard_name: "model.safetensors".into(),
                        shard_path: PathBuf::from("model.safetensors"),
                    },
                }],
            }],
            layer_metadata: vec![crate::graph::LayerMetadata {
                layer_idx: 0,
                kind: crate::model::LayerKind::DenseDecoder,
                attention_pattern: crate::model::AttentionPattern::FullCausal,
            }],
            norm_pattern: crate::model::NormPattern::PreOnly,
            lm_head_softcap: None,
            attn_logit_softcap: None,
            is_sliced: false,
            text_prefix: "model.".into(),
        };
        let mut policy = PlacementPolicy::auto_for(&inventory);
        policy.reserve_vram_bytes = 0;
        policy.kv_store = StoragePlacement::Ram;
        policy.kv_compute = ComputePlacement::Cpu;
        policy.spill_store = StoragePlacement::Mmap;
        policy.spill_compute = ComputePlacement::Cpu;

        let placement = ResolvedPlacement::plan("m", &graph, &inventory, &policy)
            .expect("placement should be planned");

        let region = &placement.region_placements[0];
        assert_eq!(region.store, StoragePlacement::Mmap);
        assert_eq!(region.compute, ComputePlacement::Cpu);
        assert_eq!(region.transfer, TransferPolicy::None);
    }
}

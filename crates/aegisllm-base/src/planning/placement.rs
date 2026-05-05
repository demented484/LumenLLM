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
    /// First `kv_first_n_layers` layers use `kv_first_store` (or VRAM derived from
    /// `kv_compute` if `kv_first_store` is `None`); remaining layers use `kv_store`.
    pub kv_first_n_layers: Option<usize>,
    /// Storage tier for the first-N KV layers. When `None` and `kv_first_n_layers`
    /// is `Some`, executor falls back to VRAM derived from `kv_compute` (legacy
    /// behavior preserved for the old `vram_layers` use case).
    pub kv_first_store: Option<StoragePlacement>,
    /// Per-layer attention (Q/K/V/O) placement override. When set, the
    /// loader uses this for attention weights regardless of the layer's
    /// region store. Set by the `attention` section of the parameters
    /// file to keep attention BF16 in RAM independently of the layer's
    /// MoE/MLP weights.
    pub attention_store_override: Option<StoragePlacement>,
    pub attention_compute_override: Option<ComputePlacement>,
    /// Per-load-time quantization for attention Q/K/V/O. `Default` keeps
    /// the checkpoint's storage format (BF16 for our Gemma-4-26B-NVFP4).
    /// Other values run a load-time per-block-absmax quantizer so the
    /// resulting weights are smaller and use the matching tensor-core
    /// GEMM during inference.
    pub attention_quantization: WeightQuantOverride,
    /// Per-load-time quantization for the shared expert (always-active
    /// MLP — `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj` in the
    /// checkpoint). Same semantics as `attention_quantization`.
    pub shared_mlp_quantization: WeightQuantOverride,
}

/// Available load-time weight-quantization formats. `Default` means "keep
/// whatever the checkpoint stored" (no on-load quantization). The four
/// non-default formats split into:
///   * **Float** (`Mxfp4`, `Fp8`) — preserve dynamic range, no
///     calibration required, ~0.1–0.5% quality cost.
///   * **Integer** (`Mxint4`, `Int4`, `Int8`) — fixed-point, smaller for
///     equal bit-width (e.g. INT4 vs MXFP4) but more sensitive to
///     outliers without calibration data; runtime path uses cuBLASLt
///     INT8 / CUTLASS INT4 GEMM kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightQuantOverride {
    /// Keep the checkpoint's storage format (no load-time quant).
    Default,
    /// Microsoft / OCP MXFP4: 4-bit float, group_size=32, E8M0 scales.
    Mxfp4,
    /// 8-bit float (E4M3 by default).
    Fp8,
    /// Microsoft / OCP MXINT4: 4-bit signed int, group_size=32, E8M0 scales.
    Mxint4,
    /// Plain INT4 with per-row or per-group scales (no MX framing).
    Int4,
    /// Plain INT8 with per-row scale.
    Int8,
}

impl WeightQuantOverride {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            // `bf16` is accepted as an alias for `default` since the
            // checkpoint stores attention/shared-MLP as BF16 and saying
            // "BF16" reads naturally for users who don't know the
            // checkpoint's native precision.
            "default" | "bf16" | "" => Some(Self::Default),
            "mxfp4" => Some(Self::Mxfp4),
            "fp8" | "fp8_e4m3" | "fp8-e4m3" => Some(Self::Fp8),
            "mxint4" => Some(Self::Mxint4),
            "int4" => Some(Self::Int4),
            "int8" => Some(Self::Int8),
            _ => None,
        }
    }
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
    /// Carried through from the source `PlacementPolicy` for the CUDA
    /// executor to consult when loading attention / shared-expert
    /// weights. `Default` means "store as the checkpoint stored it".
    pub attention_quantization: WeightQuantOverride,
    pub shared_mlp_quantization: WeightQuantOverride,
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
    /// Tail tier: where KV for layers `[first_n_layers..)` lives. If `first_n_layers`
    /// is `None`, this is the storage for ALL layers' KV.
    pub store: StoragePlacement,
    pub compute: ComputePlacement,
    pub quantization: KvCacheQuantization,
    pub context_size: usize,
    pub estimated_bytes: u64,
    /// First `first_n_layers` layers use `first_store`; remaining layers use `store`.
    /// When `None`, all layers use `store`.
    pub first_n_layers: Option<usize>,
    /// Storage tier for the first-N layers. When `None` and `first_n_layers` is set,
    /// the executor falls back to `Vram { device }` derived from `compute` (legacy
    /// behavior preserved for the `vram_layers` use case).
    pub first_store: Option<StoragePlacement>,
}

impl KvCachePlacement {
    /// Returns the resolved KV storage for the given layer index.
    /// If `first_n_layers` is set, layers `0..first_n_layers` use `first_store` (or
    /// VRAM derived from `compute` if `first_store` is `None`); the rest use `store`.
    pub fn store_for_layer(&self, layer_idx: usize) -> StoragePlacement {
        match self.first_n_layers {
            Some(n) if layer_idx < n => self.first_store.unwrap_or_else(|| match self.compute {
                ComputePlacement::Cuda { device } => StoragePlacement::Vram { device },
                _ => self.store,
            }),
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
                kv_first_n_layers: None,
                kv_first_store: None,
                attention_store_override: None,
                attention_compute_override: None,
                attention_quantization: WeightQuantOverride::Default,
                shared_mlp_quantization: WeightQuantOverride::Default,
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
                kv_first_n_layers: None,
                kv_first_store: None,
                attention_store_override: None,
                attention_compute_override: None,
                attention_quantization: WeightQuantOverride::Default,
                shared_mlp_quantization: WeightQuantOverride::Default,
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
            first_n_layers: policy.kv_first_n_layers,
            first_store: policy.kv_first_store,
        };
        let mut region_placements = graph
            .regions
            .iter()
            .map(|region| {
                let (store, compute) = apply_rules(region, graph.num_layers, policy);
                // User-facing `ram` paired with a CUDA compute target is internally
                // represented as `Mmap`: weights are file-backed and only the active
                // working set lives in pinned host memory while the rest is paged
                // in/out by the kernel under memory pressure. This avoids reserving
                // the full model footprint in pinned RAM, which would not fit on
                // memory-tight hosts even when the model itself fits in VRAM budgets.
                let store = match (store, compute) {
                    (StoragePlacement::Ram, ComputePlacement::Cuda { .. }) => {
                        StoragePlacement::Mmap
                    }
                    other => other.0,
                };
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
            attention_quantization: policy.attention_quantization,
            shared_mlp_quantization: policy.shared_mlp_quantization,
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
                head_dim: 128,
            }],
            norm_pattern: crate::model::NormPattern::PreOnly,
            lm_head_softcap: None,
            attn_logit_softcap: None,
            embed_scale: None,
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

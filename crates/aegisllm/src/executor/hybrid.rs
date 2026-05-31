use std::collections::BTreeSet;

use aegisllm_cpu::cpu::block::CpuLayerBlockExecutor;
use aegisllm_cpu::cpu::state::CpuLlamaState;
use aegisllm_cpu::{G4CpuExecutor, G4CpuState};
use aegisllm_cuda::executor::block::{CudaLayerBlockExecutor, CudaLayerBlockState};
use aegisllm_cuda::executor::cuda_kernel_limitations;
use crate::executor::nodes::ExecutorGraphPlan;
use aegisllm_base::executor::traits::{
    ExecutorBackendInfo, ExecutorCapability, ExecutorProviderPlan, GenerationBackendPrimitives,
    GenerationState, ModelExecutorBackend,
};
use aegisllm_base::artifact::ModelArtifact;
use aegisllm_base::backend::BackendKind;
use aegisllm_base::model::{detect_architecture, LayerKind};
use aegisllm_cuda::cuda::CudaRuntimeConfig;
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::generation::SamplingConfig;
use aegisllm_base::graph::ModelGraph;
use aegisllm_base::planning::placement::{
    ComputePlacement, ResolvedPlacement, StoragePlacement, TransferPolicy,
};
use aegisllm_base::planning::runtime::RuntimePlan;
use aegisllm_base::text::TextProcessor;

/// The CPU side of a hybrid forward. Llama text decoders use the per-layer
/// `CpuLayerBlockExecutor`; Gemma-4 DENSE (E2B/E4B) uses the architecture-correct
/// `G4CpuExecutor` (PrePost norms, per-head q/k/v norm, partial RoPE, per-layer
/// head_dim/kv, PLE). Gemma-4 MoE (26B) is rejected upstream.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
enum HybridCpu {
    Llama(CpuLayerBlockExecutor),
    Gemma4(Box<G4CpuExecutor>),
}

#[derive(Debug)]
pub struct HybridExecutorProvider {
    backends: Vec<BackendKind>,
    limitations: Vec<String>,
    text: Option<TextProcessor>,
    cpu: Option<HybridCpu>,
    cuda: Option<CudaLayerBlockExecutor>,
    schedule: Vec<HybridLayerBackend>,
}

impl HybridExecutorProvider {
    pub fn new(backends: Vec<BackendKind>, limitations: Vec<String>) -> Self {
        Self {
            backends,
            limitations,
            text: None,
            cpu: None,
            cuda: None,
            schedule: Vec::new(),
        }
    }

    pub fn plan(
        placement: &ResolvedPlacement,
        runtime: &RuntimePlan,
    ) -> Option<ExecutorProviderPlan> {
        let topology = HybridTopology::from_placement(placement);
        if !topology.is_hybrid() {
            return None;
        }
        let limitations = hybrid_limitations(placement, runtime, &topology);
        let info = hybrid_backend_info(&topology, limitations.clone());
        Some(ExecutorProviderPlan {
            info,
            runnable: limitations.is_empty(),
            limitations,
        })
    }

    pub fn from_artifact(
        artifact: &ModelArtifact,
        graph: &ModelGraph,
        placement: &ResolvedPlacement,
        runtime: &RuntimePlan,
        cuda_config: CudaRuntimeConfig,
    ) -> Result<Self> {
        let plan = Self::plan(placement, runtime).ok_or_else(|| {
            AegisError::InvalidPlan("hybrid executor requested for non-hybrid placement".into())
        })?;
        if !plan.runnable {
            return Err(AegisError::Unsupported(format!(
                "hybrid executor plan is not runnable yet: {}",
                plan.limitations.join("; ")
            )));
        }
        let topology = HybridTopology::from_placement(placement);
        let cuda_device = topology
            .cuda_devices
            .iter()
            .next()
            .copied()
            .ok_or_else(|| {
                AegisError::InvalidPlan("hybrid executor selected without CUDA device".into())
            })?;
        let mut cpu_layers = BTreeSet::new();
        let mut cuda_layers = BTreeSet::new();
        let mut schedule = vec![HybridLayerBackend::Cpu; graph.num_layers];
        for region in &placement.region_placements {
            let Some(layer) = region.layer_index else {
                continue;
            };
            match region.compute {
                ComputePlacement::Cpu => {
                    cpu_layers.insert(layer);
                    schedule[layer] = HybridLayerBackend::Cpu;
                }
                ComputePlacement::Cuda { device } if device == cuda_device => {
                    cuda_layers.insert(layer);
                    schedule[layer] = HybridLayerBackend::Cuda;
                }
                ComputePlacement::Cuda { device } => {
                    return Err(AegisError::Unsupported(format!(
                        "hybrid executor cannot schedule layer.{layer} on cuda:{device}; selected cuda:{cuda_device}"
                    )));
                }
                ComputePlacement::Wgpu { device } => {
                    return Err(AegisError::Unsupported(format!(
                        "hybrid executor cannot schedule layer.{layer} on wgpu:{device}; \
                         hybrid path is currently CPU+CUDA only"
                    )));
                }
            }
        }

        // Architecture dispatch: Gemma-4 DENSE (E2B/E4B) uses the G4 CPU path
        // (PrePost norms / q-k-v norms / partial RoPE / PLE) on the CPU side and
        // the PLE-aware CUDA per-layer path on the GPU side. Llama/Qwen text keep
        // the existing per-layer Llama block path. Gemma-4 MoE (26B) is rejected.
        let is_gemma4 = detect_architecture(&artifact.config)
            .map(|arch| arch.name() == "gemma4")
            .unwrap_or(false);

        let cuda = CudaLayerBlockExecutor::from_artifact(
            artifact,
            graph,
            placement,
            runtime,
            cuda_device,
            cuda_config,
            &cuda_layers,
        )?;

        let cpu = if is_gemma4 {
            // Reject MoE Gemma-4 (26B-A4B): the hybrid is dense-only. Detect any
            // MoE layer from the graph metadata.
            let has_moe = graph
                .layer_metadata
                .iter()
                .any(|m| matches!(m.kind, LayerKind::MoEDecoder { .. }));
            if has_moe {
                return Err(AegisError::Unsupported(
                    "hybrid CPU+GPU execution for Gemma-4 is implemented for DENSE models \
                     (E2B/E4B) only; the 26B-A4B MoE hybrid is out of scope — run it fully \
                     on one device."
                        .into(),
                ));
            }
            // The G4 CPU executor loads ALL layers (dense indexing keeps KV-share
            // parent indices valid); only the CPU-scheduled layers are RUN. Its
            // global PLE apparatus + embed are also used for the once-per-token
            // PLE feed shared with the GPU layers.
            let g4 = G4CpuExecutor::from_artifact(artifact, graph, placement, runtime)?;
            // Cross-device KV-share validation: every KV-share child must be
            // co-located with its parent on the SAME backend, so the parent's KV
            // cache (which lives with whichever device ran it) is reachable. A
            // prefix/suffix split that separates a shared layer from its parent
            // is rejected with actionable guidance.
            for layer in 0..graph.num_layers {
                if let Some(parent) = g4.kv_shared_parent(layer) {
                    if schedule[layer] != schedule[parent] {
                        return Err(AegisError::Unsupported(format!(
                            "hybrid Gemma-4: layer {layer} shares its KV cache with layer \
                             {parent}, but they are scheduled on different backends \
                             ({:?} vs {:?}). Place the split so every KV-share child and its \
                             parent are on the same device — for E2B the lowest valid \
                             cuda/cpu boundary is layer 13 (parents 13 & 14 must co-locate \
                             with shared layers 15..).",
                            schedule[layer], schedule[parent]
                        )));
                    }
                }
            }
            HybridCpu::Gemma4(Box::new(g4))
        } else {
            HybridCpu::Llama(CpuLayerBlockExecutor::from_artifact(
                artifact,
                graph,
                placement,
                runtime,
                &cpu_layers,
            )?)
        };

        Ok(Self {
            backends: plan.info.backends,
            limitations: vec![
                "hybrid scheduler uses synchronous host activation boundaries; pinned/async transfer nodes are pending".into(),
                "hybrid scheduler keeps final logits/sampling on CPU for correctness-first MVP".into(),
            ],
            text: Some(TextProcessor::from_artifact(artifact)?),
            cpu: Some(cpu),
            cuda: Some(cuda),
            schedule,
        })
    }
}

impl GenerationBackendPrimitives for HybridExecutorProvider {
    fn encode_prompt(&self, prompt: &str) -> Result<Vec<usize>> {
        self.text
            .as_ref()
            .ok_or_else(|| self.not_initialized())?
            .encode_prompt(prompt)
    }

    fn decode_tokens(&self, tokens: &[usize]) -> Result<String> {
        self.text
            .as_ref()
            .ok_or_else(|| self.not_initialized())?
            .decode_tokens(tokens)
    }

    fn is_eos(&self, _token: usize) -> bool {
        self.text
            .as_ref()
            .map(|text| text.is_eos(_token))
            .unwrap_or(false)
    }

    fn new_sequence_state(&self) -> Result<Box<dyn GenerationState>> {
        let cpu = self.cpu.as_ref().ok_or_else(|| self.not_initialized())?;
        let cuda = self.cuda.as_ref().ok_or_else(|| self.not_initialized())?;
        let cpu_state = match cpu {
            HybridCpu::Llama(m) => HybridCpuState::Llama(m.new_state()),
            HybridCpu::Gemma4(m) => HybridCpuState::Gemma4(Box::new(m.new_state())),
        };
        Ok(Box::new(HybridSequenceState {
            position: 0,
            hidden: None,
            cpu: cpu_state,
            cuda: cuda.new_state()?,
        }))
    }

    fn forward_hidden(&self, state: &mut dyn GenerationState, token_id: usize) -> Result<()> {
        let state = hybrid_state_mut(state)?;
        let hidden = self.forward_hidden_host(state, token_id)?;
        state.hidden = Some(hidden);
        Ok(())
    }

    fn forward_logits(&self, state: &mut dyn GenerationState, token_id: usize) -> Result<Vec<f32>> {
        let state = hybrid_state_mut(state)?;
        let hidden = self.forward_hidden_host(state, token_id)?;
        self.final_logits_host(&mut state.cpu, &hidden)
    }

    fn prefill_prompt(
        &self,
        state: &mut dyn GenerationState,
        prompt_tokens: &[usize],
        sampling: &SamplingConfig,
    ) -> Result<usize> {
        let Some((&last, prefix)) = prompt_tokens.split_last() else {
            return Err(AegisError::InvalidConfig(
                "prompt produced no tokens".into(),
            ));
        };
        let state = hybrid_state_mut(state)?;
        for &token in prefix {
            let _hidden = self.forward_hidden_host(state, token)?;
        }
        let hidden = self.forward_hidden_host(state, last)?;
        let logits = self.final_logits_host(&mut state.cpu, &hidden)?;
        aegisllm_base::executor::generation::sample_next_token(&logits, sampling)
    }

    fn forward_next_token(
        &self,
        state: &mut dyn GenerationState,
        token_id: usize,
        sampling: &SamplingConfig,
    ) -> Result<usize> {
        let logits = self.forward_logits(state, token_id)?;
        aegisllm_base::executor::generation::sample_next_token(&logits, sampling)
    }
}

impl ModelExecutorBackend for HybridExecutorProvider {
    fn info(&self) -> ExecutorBackendInfo {
        let topology = HybridTopology::from_backends(&self.backends);
        hybrid_backend_info(&topology, self.limitations.clone())
    }

    fn probe(&self) -> Result<()> {
        let _state = self.new_sequence_state()?;
        Ok(())
    }
}

impl HybridExecutorProvider {
    fn not_initialized(&self) -> AegisError {
        AegisError::Unsupported(format!(
            "hybrid executor provider is registered but not runnable yet: {}",
            self.limitations.join("; ")
        ))
    }

    fn forward_hidden_host(
        &self,
        state: &mut HybridSequenceState,
        token_id: usize,
    ) -> Result<Vec<f32>> {
        let cpu = self.cpu.as_ref().ok_or_else(|| self.not_initialized())?;
        let cuda = self.cuda.as_ref().ok_or_else(|| self.not_initialized())?;
        let position = state.position;
        match (cpu, &mut state.cpu) {
            (HybridCpu::Llama(cpu), HybridCpuState::Llama(cpu_state)) => {
                let mut hidden = cpu.embed_token(token_id)?;
                for (layer, backend) in self.schedule.iter().copied().enumerate() {
                    hidden = match backend {
                        HybridLayerBackend::Cpu => {
                            cpu.forward_layer_host(cpu_state, layer, position, &hidden)?
                        }
                        HybridLayerBackend::Cuda => {
                            cuda.forward_layer_host(&mut state.cuda, layer, position, &hidden)?
                        }
                    };
                }
                state.position += 1;
                Ok(hidden)
            }
            (HybridCpu::Gemma4(g4), HybridCpuState::Gemma4(g4_state)) => {
                // Token entry (embed + scale + PLE feed) runs ONCE on the CPU.
                // The resulting per_layer_inputs is shared with the GPU layers
                // (uploaded per GPU layer), so every layer's PLE additive uses a
                // bit-identical feed regardless of which device computes it.
                let mut hidden = g4.token_entry_host(g4_state, token_id)?;
                // Clone the shared PLE feed once for this token (empty for
                // non-PLE models). GPU layers upload it; CPU layers read it from
                // g4_state directly inside forward_dense_layer_host.
                let per_layer_inputs: Vec<f32> =
                    g4.per_layer_inputs_snapshot(g4_state).to_vec();
                // Walk the schedule, dispatching each maximal CONTIGUOUS run of
                // GPU layers as ONE block (upload hidden once, run the whole run
                // on-device, download once) instead of per-layer host round-trips
                // — the per-layer path made the GPU layers ~as slow as CPU because
                // every layer re-uploaded + downloaded the hidden with a blocking
                // sync. CPU layers run one at a time (they already work on the host
                // hidden; no transfer to batch).
                let n = self.schedule.len();
                let mut layer = 0usize;
                while layer < n {
                    match self.schedule[layer] {
                        HybridLayerBackend::Cpu => {
                            hidden = g4.forward_dense_layer_host(g4_state, layer, position, &hidden)?;
                            layer += 1;
                        }
                        HybridLayerBackend::Cuda => {
                            let start = layer;
                            while layer < n
                                && matches!(self.schedule[layer], HybridLayerBackend::Cuda)
                            {
                                layer += 1;
                            }
                            let block: Vec<usize> = (start..layer).collect();
                            hidden = cuda.forward_g4_layers_block_host(
                                &mut state.cuda,
                                &block,
                                position,
                                &hidden,
                                &per_layer_inputs,
                            )?;
                        }
                    }
                }
                g4.advance_position(g4_state);
                state.position += 1;
                Ok(hidden)
            }
            _ => Err(AegisError::InvalidPlan(
                "hybrid CPU executor/state variant mismatch".into(),
            )),
        }
    }

    /// final_norm → lm_head (+ Gemma-4 logit softcap) on the CPU, dispatching on
    /// the active CPU backend. Logits/sampling stay on CPU in the MVP.
    fn final_logits_host(
        &self,
        cpu_state: &mut HybridCpuState,
        hidden: &[f32],
    ) -> Result<Vec<f32>> {
        let cpu = self.cpu.as_ref().ok_or_else(|| self.not_initialized())?;
        match (cpu, cpu_state) {
            (HybridCpu::Llama(cpu), HybridCpuState::Llama(cpu_state)) => {
                cpu.final_logits_host_with_state(cpu_state, hidden)
            }
            (HybridCpu::Gemma4(g4), HybridCpuState::Gemma4(_)) => g4.final_logits_host(hidden),
            _ => Err(AegisError::InvalidPlan(
                "hybrid CPU executor/state variant mismatch".into(),
            )),
        }
    }
}

#[derive(Debug)]
struct HybridSequenceState {
    position: usize,
    hidden: Option<Vec<f32>>,
    cpu: HybridCpuState,
    cuda: CudaLayerBlockState,
}

/// CPU-side per-sequence state, matching the active `HybridCpu` variant.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
enum HybridCpuState {
    Llama(CpuLlamaState),
    Gemma4(Box<G4CpuState>),
}

fn hybrid_state_mut(state: &mut dyn GenerationState) -> Result<&mut HybridSequenceState> {
    state
        .as_any_mut()
        .downcast_mut::<HybridSequenceState>()
        .ok_or_else(|| AegisError::InvalidPlan("hybrid executor received foreign state".into()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HybridLayerBackend {
    Cpu,
    Cuda,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HybridTopology {
    has_cpu: bool,
    cuda_devices: BTreeSet<usize>,
}

impl HybridTopology {
    fn from_placement(placement: &ResolvedPlacement) -> Self {
        let mut topology = Self {
            has_cpu: matches!(placement.kv_cache.compute, ComputePlacement::Cpu),
            cuda_devices: BTreeSet::new(),
        };
        if let ComputePlacement::Cuda { device } = placement.kv_cache.compute {
            topology.cuda_devices.insert(device);
        }
        for region in &placement.region_placements {
            match region.compute {
                ComputePlacement::Cpu => topology.has_cpu = true,
                ComputePlacement::Cuda { device } => {
                    topology.cuda_devices.insert(device);
                }
                ComputePlacement::Wgpu { .. } => {
                    // Hybrid topology currently only models CPU+CUDA.
                    // Wgpu placements are deferred to the wgpu provider
                    // (which the hybrid scheduler does not call).
                }
            }
        }
        topology
    }

    fn from_backends(backends: &[BackendKind]) -> Self {
        let mut topology = Self {
            has_cpu: false,
            cuda_devices: BTreeSet::new(),
        };
        for backend in backends {
            match backend {
                BackendKind::Cpu | BackendKind::Wgpu { .. } => topology.has_cpu = true,
                BackendKind::Cuda { device } => {
                    topology.cuda_devices.insert(*device);
                }
            }
        }
        topology
    }

    fn is_hybrid(&self) -> bool {
        (self.has_cpu && !self.cuda_devices.is_empty()) || self.cuda_devices.len() > 1
    }

    fn backends(&self) -> Vec<BackendKind> {
        let mut backends = Vec::new();
        if self.has_cpu {
            backends.push(BackendKind::Cpu);
        }
        backends.extend(
            self.cuda_devices
                .iter()
                .copied()
                .map(|device| BackendKind::Cuda { device }),
        );
        backends
    }

    fn compute_placements(&self) -> Vec<ComputePlacement> {
        let mut placements = Vec::new();
        if self.has_cpu {
            placements.push(ComputePlacement::Cpu);
        }
        placements.extend(
            self.cuda_devices
                .iter()
                .copied()
                .map(|device| ComputePlacement::Cuda { device }),
        );
        placements
    }

    fn storage_placements(&self) -> Vec<StoragePlacement> {
        let mut placements = vec![StoragePlacement::Ram, StoragePlacement::Mmap];
        placements.extend(
            self.cuda_devices
                .iter()
                .copied()
                .map(|device| StoragePlacement::Vram { device }),
        );
        placements
    }
}

fn hybrid_backend_info(topology: &HybridTopology, limitations: Vec<String>) -> ExecutorBackendInfo {
    ExecutorBackendInfo {
        name: "hybrid",
        backends: topology.backends(),
        weight_store: topology.storage_placements(),
        weight_compute: topology.compute_placements(),
        kv_compute: topology.compute_placements(),
        capabilities: vec![
            ExecutorCapability::Tokenize,
            ExecutorCapability::DenseEmbedding,
            ExecutorCapability::DenseLmHead,
            ExecutorCapability::RmsNorm,
            ExecutorCapability::Rope,
            ExecutorCapability::Attention,
            ExecutorCapability::Mlp,
            ExecutorCapability::Nvfp4Linear,
            ExecutorCapability::KvCache,
        ],
        limitations,
    }
}

fn hybrid_limitations(
    placement: &ResolvedPlacement,
    runtime: &RuntimePlan,
    topology: &HybridTopology,
) -> Vec<String> {
    let mut limitations = cuda_kernel_limitations(runtime);

    if topology.cuda_devices.len() > 1 {
        limitations.push(
            "multi-CUDA-device hybrid scheduling and peer transfer nodes are not wired yet".into(),
        );
    }

    let executor_graph = ExecutorGraphPlan::from_resolved_placement(placement);

    let _activation_boundaries = executor_graph.activation_transfers().count();
    let _host_to_device_weight_regions = executor_graph
        .weight_transfers()
        .filter(|node| node.transfer == TransferPolicy::HostToDeviceEachUse)
        .count();

    let device_to_host_weight_regions = executor_graph
        .weight_transfers()
        .filter(|node| node.transfer == TransferPolicy::DeviceToHostEachUse)
        .count();
    if device_to_host_weight_regions > 0 {
        limitations.push(format!(
            "{device_to_host_weight_regions} CUDA-to-host weight staging nodes are not wired into the hybrid scheduler"
        ));
    }

    let cross_device_weight_regions = executor_graph
        .weight_transfers()
        .filter(|node| node.transfer == TransferPolicy::CrossDevice)
        .count();
    if cross_device_weight_regions > 0 {
        limitations.push(format!(
            "{cross_device_weight_regions} cross-device weight transfer nodes are not wired into the hybrid scheduler"
        ));
    }

    let non_cpu_bookend_regions = placement
        .region_placements
        .iter()
        .filter(|region| region.layer_index.is_none())
        .filter(|region| region.compute != ComputePlacement::Cpu)
        .count();
    if non_cpu_bookend_regions > 0 {
        limitations.push(format!(
            "{non_cpu_bookend_regions} non-layer regions are not compute=cpu; hybrid MVP keeps embedding/final logits on CPU"
        ));
    }
    limitations.sort();
    limitations.dedup();
    limitations
}

#[cfg(test)]
mod tests {
    use super::*;
    use aegisllm_base::graph::{GraphRegionKind, RegionId};
    use aegisllm_base::planning::memory::{MemoryBudget, MemoryPool};
    use aegisllm_base::planning::placement::{KvCachePlacement, RegionPlacement};
    use aegisllm_base::planning::runtime::{KernelPlan, SyncPolicy, TensorResidency};
    use aegisllm_base::tensor::layout::{LinearLayoutPlan, LinearResidentLayout, MaterializationPolicy};
    use aegisllm_base::tensor::quant::{KvCacheQuantization, QuantFormat, WeightQuantization};

    #[test]
    fn hybrid_provider_keeps_mixed_cpu_cuda_distinct() {
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
                    store: StoragePlacement::Vram { device: 0 },
                    compute: ComputePlacement::Cuda { device: 0 },
                    transfer: TransferPolicy::None,
                },
            ],
            kv_cache: KvCachePlacement {
                store: StoragePlacement::Ram,
                compute: ComputePlacement::Cpu,
                quantization: KvCacheQuantization::F16,
                context_size: 1,
                estimated_bytes: 1,
                first_n_layers: None,
                first_store: None,
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
            linear_layout: Default::default(),
            warnings: Vec::new(),
            attention_quantization: aegisllm_base::planning::placement::WeightQuantOverride::Default,
            shared_mlp_quantization: aegisllm_base::planning::placement::WeightQuantOverride::Default,
            attention_store_override: None,
            experts_compute_override: None,
            experts_store_override: None,
        };
        let runtime = RuntimePlan {
            kernels: vec![
                KernelPlan {
                    name: "embed".into(),
                    device: BackendKind::Cpu,
                    quant_format: QuantFormat::Bf16,
                    linear_layout: LinearLayoutPlan {
                        source_format: QuantFormat::Bf16,
                        resident_layout: LinearResidentLayout::PackedSource,
                        materialization: MaterializationPolicy::Lazy,
                        extra_weight_bytes: 0,
                        notes: Vec::new(),
                    },
                    family: crate::planning::runtime::KernelFamily::CpuSimd,
                    residency: TensorResidency::Host,
                    sync: SyncPolicy::StreamOrdered,
                },
                KernelPlan {
                    name: "layer.0".into(),
                    device: BackendKind::Cuda { device: 0 },
                    quant_format: QuantFormat::Nvfp4,
                    linear_layout: LinearLayoutPlan {
                        source_format: QuantFormat::Nvfp4,
                        resident_layout: LinearResidentLayout::NativeTensorCore,
                        materialization: MaterializationPolicy::Lazy,
                        extra_weight_bytes: 0,
                        notes: Vec::new(),
                    },
                    family: crate::planning::runtime::KernelFamily::CudaNativeFp4TensorCores,
                    residency: TensorResidency::Device,
                    sync: SyncPolicy::StreamOrdered,
                },
            ],
            warnings: Vec::new(),
        };

        let plan = HybridExecutorProvider::plan(&placement, &runtime).unwrap();
        assert_eq!(plan.info.name, "hybrid");
        assert!(plan.runnable);
        assert!(plan.info.backends.contains(&BackendKind::Cpu));
        assert!(
            plan.info
                .backends
                .contains(&BackendKind::Cuda { device: 0 })
        );
        assert!(plan.limitations.is_empty());
    }
}

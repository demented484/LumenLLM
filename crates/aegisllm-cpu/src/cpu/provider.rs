use super::g4::{G4CpuExecutor, G4CpuState};
use super::state::{CpuLlamaExecutor, CpuLlamaState};
use aegisllm_base::artifact::ModelArtifact;
use aegisllm_base::backend::BackendKind;
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::executor::traits::{
    ExecutorBackendInfo, ExecutorCapability, ExecutorProviderPlan, GenerationBackendPrimitives,
    GenerationState, ModelExecutorBackend,
};
use aegisllm_base::graph::ModelGraph;
use aegisllm_base::model::detect_architecture;
use aegisllm_base::planning::placement::{ComputePlacement, ResolvedPlacement, StoragePlacement};
use aegisllm_base::planning::runtime::RuntimePlan;
use aegisllm_base::text::TextProcessor;

/// Architecture-specific CPU backend. Selected at load time from the model
/// descriptor — Gemma-4 routes to the new `G4CpuExecutor`; everything else
/// (Llama / Qwen / Nemotron text) keeps the existing Llama-style path.
/// The Gemma-4 variants are boxed because they carry substantially more
/// per-layer state (PrePost norms, q/k/v norms, PLE/MoE), keeping the enum
/// small for the common Llama path.
#[derive(Debug)]
// Llama is the established inline hot path; the Gemma-4 variant is already
// boxed. The residual size delta is the inline Llama executor itself, which we
// deliberately keep un-boxed to avoid touching the existing path.
#[allow(clippy::large_enum_variant)]
enum CpuBackend {
    Llama(CpuLlamaExecutor),
    Gemma4(Box<G4CpuExecutor>),
}

/// Runtime state for the active backend variant.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
enum CpuState {
    Llama(CpuLlamaState),
    Gemma4(Box<G4CpuState>),
}

#[derive(Debug)]
pub struct CpuReferenceExecutor {
    text: TextProcessor,
    cpu: CpuBackend,
}

impl CpuReferenceExecutor {
    pub fn plan(placement: &ResolvedPlacement) -> ExecutorProviderPlan {
        let limitations = cpu_reference_limitations(placement);
        ExecutorProviderPlan {
            info: cpu_backend_info(Vec::new()),
            runnable: limitations.is_empty(),
            limitations,
        }
    }

    pub fn from_artifact(
        artifact: &ModelArtifact,
        graph: &ModelGraph,
        placement: &ResolvedPlacement,
        runtime: &RuntimePlan,
    ) -> Result<Self> {
        validate_cpu_placement(placement)?;
        // Architecture is decided here (downstream of executor selection, which
        // is purely placement-driven). Gemma-4 uses the PrePost/PLE/MoE forward;
        // all other text decoders keep the Llama-style path.
        let is_gemma4 = detect_architecture(&artifact.config)
            .map(|arch| arch.name() == "gemma4")
            .unwrap_or(false);
        let cpu = if is_gemma4 {
            CpuBackend::Gemma4(Box::new(G4CpuExecutor::from_artifact(
                artifact, graph, placement, runtime,
            )?))
        } else {
            CpuBackend::Llama(CpuLlamaExecutor::from_artifact(artifact, graph, placement, runtime)?)
        };
        Ok(Self {
            text: TextProcessor::from_artifact(artifact)?,
            cpu,
        })
    }
}

impl GenerationBackendPrimitives for CpuReferenceExecutor {
    fn encode_prompt(&self, prompt: &str) -> Result<Vec<usize>> {
        self.text.encode_prompt(prompt)
    }

    fn decode_tokens(&self, tokens: &[usize]) -> Result<String> {
        self.text.decode_tokens(tokens)
    }

    fn is_eos(&self, token: usize) -> bool {
        self.text.is_eos(token)
    }

    fn new_sequence_state(&self) -> Result<Box<dyn GenerationState>> {
        Ok(match &self.cpu {
            CpuBackend::Llama(m) => Box::new(CpuState::Llama(m.new_state())),
            CpuBackend::Gemma4(m) => Box::new(CpuState::Gemma4(Box::new(m.new_state()))),
        })
    }

    fn forward_hidden(&self, state: &mut dyn GenerationState, token_id: usize) -> Result<()> {
        match (&self.cpu, cpu_state_mut(state)?) {
            (CpuBackend::Llama(m), CpuState::Llama(s)) => m.forward_hidden(s, token_id).map(|_| ()),
            (CpuBackend::Gemma4(m), CpuState::Gemma4(s)) => m.forward_hidden(s, token_id).map(|_| ()),
            _ => Err(state_mismatch()),
        }
    }

    fn forward_logits(&self, state: &mut dyn GenerationState, token_id: usize) -> Result<Vec<f32>> {
        match (&self.cpu, cpu_state_mut(state)?) {
            (CpuBackend::Llama(m), CpuState::Llama(s)) => m.forward_logits(s, token_id),
            (CpuBackend::Gemma4(m), CpuState::Gemma4(s)) => m.forward_logits(s, token_id),
            _ => Err(state_mismatch()),
        }
    }

    fn prefill_prompt(
        &self,
        state: &mut dyn GenerationState,
        prompt_tokens: &[usize],
        sampling: &aegisllm_base::generation::SamplingConfig,
    ) -> Result<usize> {
        match (&self.cpu, cpu_state_mut(state)?) {
            (CpuBackend::Llama(m), CpuState::Llama(s)) => m.prefill_prompt(s, prompt_tokens, sampling),
            (CpuBackend::Gemma4(m), CpuState::Gemma4(s)) => m.prefill_prompt(s, prompt_tokens, sampling),
            _ => Err(state_mismatch()),
        }
    }
}

fn state_mismatch() -> AegisError {
    AegisError::InvalidPlan("CPU executor backend/state variant mismatch".into())
}

impl ModelExecutorBackend for CpuReferenceExecutor {
    fn info(&self) -> ExecutorBackendInfo {
        cpu_backend_info(vec![
            "CPU reference provider is correctness-first, not the final performance path".into(),
        ])
    }
}

fn cpu_state_mut(state: &mut dyn GenerationState) -> Result<&mut CpuState> {
    state
        .as_any_mut()
        .downcast_mut::<CpuState>()
        .ok_or_else(|| AegisError::InvalidPlan("CPU executor received foreign state".into()))
}

fn cpu_backend_info(limitations: Vec<String>) -> ExecutorBackendInfo {
    ExecutorBackendInfo {
        name: "cpu-reference",
        backends: vec![BackendKind::Cpu],
        weight_store: vec![StoragePlacement::Ram, StoragePlacement::Mmap],
        weight_compute: vec![ComputePlacement::Cpu],
        kv_compute: vec![ComputePlacement::Cpu],
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

fn cpu_reference_limitations(placement: &ResolvedPlacement) -> Vec<String> {
    let mut limitations = Vec::new();
    let non_cpu_compute_regions = placement
        .region_placements
        .iter()
        .filter(|region| !matches!(region.compute, ComputePlacement::Cpu))
        .count();
    if non_cpu_compute_regions > 0 {
        limitations.push(format!(
            "{non_cpu_compute_regions} regions require non-CPU executor providers"
        ));
    }
    if !matches!(placement.kv_cache.compute, ComputePlacement::Cpu) {
        limitations.push(format!(
            "kv-cache compute={} needs a matching executor provider",
            placement.kv_cache.compute
        ));
    }
    limitations
}

fn validate_cpu_placement(placement: &ResolvedPlacement) -> Result<()> {
    for region in &placement.region_placements {
        if !matches!(region.compute, ComputePlacement::Cpu) {
            return Err(AegisError::Unsupported(format!(
                "generate CPU executor cannot run region `{}` with compute={}",
                region.region_id.0, region.compute
            )));
        }
    }
    if !matches!(placement.kv_cache.compute, ComputePlacement::Cpu) {
        return Err(AegisError::Unsupported(format!(
            "generate CPU executor requires kv compute=cpu, got {}",
            placement.kv_cache.compute
        )));
    }
    Ok(())
}

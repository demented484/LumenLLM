use super::state::{CpuLlamaExecutor, CpuLlamaState};
use aegisllm_base::artifact::ModelArtifact;
use aegisllm_base::backend::BackendKind;
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::executor::traits::{
    ExecutorBackendInfo, ExecutorCapability, ExecutorProviderPlan, GenerationBackendPrimitives,
    GenerationState, ModelExecutorBackend,
};
use aegisllm_base::graph::ModelGraph;
use aegisllm_base::planning::placement::{ComputePlacement, ResolvedPlacement, StoragePlacement};
use aegisllm_base::planning::runtime::RuntimePlan;
use aegisllm_base::text::TextProcessor;

#[derive(Debug)]
pub struct CpuReferenceExecutor {
    text: TextProcessor,
    cpu: CpuLlamaExecutor,
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
        Ok(Self {
            text: TextProcessor::from_artifact(artifact)?,
            cpu: CpuLlamaExecutor::from_artifact(artifact, graph, placement, runtime)?,
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
        Ok(Box::new(self.cpu.new_state()))
    }

    fn forward_hidden(&self, state: &mut dyn GenerationState, token_id: usize) -> Result<()> {
        let state = cpu_state_mut(state)?;
        self.cpu.forward_hidden(state, token_id).map(|_| ())
    }

    fn forward_logits(&self, state: &mut dyn GenerationState, token_id: usize) -> Result<Vec<f32>> {
        let state = cpu_state_mut(state)?;
        self.cpu.forward_logits(state, token_id)
    }

    fn prefill_prompt(
        &self,
        state: &mut dyn GenerationState,
        prompt_tokens: &[usize],
        sampling: &aegisllm_base::generation::SamplingConfig,
    ) -> Result<usize> {
        let state = cpu_state_mut(state)?;
        self.cpu.prefill_prompt(state, prompt_tokens, sampling)
    }
}

impl ModelExecutorBackend for CpuReferenceExecutor {
    fn info(&self) -> ExecutorBackendInfo {
        cpu_backend_info(vec![
            "CPU reference provider is correctness-first, not the final performance path".into(),
        ])
    }
}

fn cpu_state_mut(state: &mut dyn GenerationState) -> Result<&mut CpuLlamaState> {
    state
        .as_any_mut()
        .downcast_mut::<CpuLlamaState>()
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

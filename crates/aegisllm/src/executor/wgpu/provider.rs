use super::loader::WgpuContext;
use super::state::WgpuLlamaState;
use crate::artifact::ModelArtifact;
use crate::backend::BackendKind;
use crate::error::{AegisError, Result};
use crate::executor::traits::{
    ExecutorBackendInfo, ExecutorCapability, ExecutorProviderPlan, GenerationBackendPrimitives,
    GenerationState, ModelExecutorBackend,
};
use crate::generation::SamplingConfig;
use crate::planning::placement::{ComputePlacement, StoragePlacement};
use crate::text::TextProcessor;

#[derive(Debug)]
pub struct WgpuExecutorProvider {
    device: usize,
    limitations: Vec<String>,
    text: Option<TextProcessor>,
    #[allow(dead_code)]
    ctx: Option<WgpuContext>,
}

impl WgpuExecutorProvider {
    pub fn new(device: usize) -> Self {
        Self {
            device,
            limitations: vec!["wgpu skeleton; forward not implemented".into()],
            text: None,
            ctx: None,
        }
    }

    pub fn plan() -> ExecutorProviderPlan {
        let limitations = vec!["wgpu skeleton; forward not implemented".into()];
        ExecutorProviderPlan {
            info: wgpu_backend_info(0, limitations.clone()),
            runnable: false,
            limitations,
        }
    }

    pub fn probe_adapters() -> Result<()> {
        let instance = wgpu::Instance::default();
        let adapters = instance.enumerate_adapters(wgpu::Backends::PRIMARY);
        if adapters.is_empty() {
            return Err(AegisError::Unsupported("no wgpu adapter available".into()));
        }
        Ok(())
    }

    pub fn from_artifact(artifact: &ModelArtifact, device_index: usize) -> Result<Self> {
        let ctx = WgpuContext::new(device_index)?;
        Ok(Self {
            device: device_index,
            limitations: vec!["wgpu skeleton; forward not implemented".into()],
            text: Some(TextProcessor::from_artifact(artifact)?),
            ctx: Some(ctx),
        })
    }
}

impl GenerationBackendPrimitives for WgpuExecutorProvider {
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

    fn is_eos(&self, token: usize) -> bool {
        self.text
            .as_ref()
            .map(|t| t.is_eos(token))
            .unwrap_or(false)
    }

    fn new_sequence_state(&self) -> Result<Box<dyn GenerationState>> {
        Ok(Box::new(WgpuLlamaState))
    }

    fn forward_hidden(&self, _state: &mut dyn GenerationState, _token_id: usize) -> Result<()> {
        Err(AegisError::Unsupported(
            "wgpu forward not implemented".into(),
        ))
    }

    fn forward_logits(
        &self,
        _state: &mut dyn GenerationState,
        _token_id: usize,
    ) -> Result<Vec<f32>> {
        Err(AegisError::Unsupported(
            "wgpu forward not implemented".into(),
        ))
    }

    fn prefill_prompt(
        &self,
        _state: &mut dyn GenerationState,
        _prompt_tokens: &[usize],
        _sampling: &SamplingConfig,
    ) -> Result<usize> {
        Err(AegisError::Unsupported(
            "wgpu forward not implemented".into(),
        ))
    }
}

impl ModelExecutorBackend for WgpuExecutorProvider {
    fn info(&self) -> ExecutorBackendInfo {
        wgpu_backend_info(self.device, self.limitations.clone())
    }
}

impl WgpuExecutorProvider {
    fn not_initialized(&self) -> AegisError {
        AegisError::Unsupported(format!(
            "wgpu executor not initialized: {}",
            self.limitations.join("; ")
        ))
    }
}

fn wgpu_backend_info(device: usize, limitations: Vec<String>) -> ExecutorBackendInfo {
    ExecutorBackendInfo {
        name: "wgpu",
        backends: vec![BackendKind::Wgpu { device }],
        weight_store: vec![StoragePlacement::Ram, StoragePlacement::Mmap],
        weight_compute: vec![ComputePlacement::Cpu],
        kv_compute: vec![ComputePlacement::Cpu],
        capabilities: vec![ExecutorCapability::Tokenize],
        limitations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgpu_plan_is_never_runnable_in_skeleton() {
        let plan = WgpuExecutorProvider::plan();
        assert!(!plan.runnable);
        assert!(!plan.limitations.is_empty());
    }

    #[test]
    fn wgpu_provider_returns_unsupported_for_forward() {
        let provider = WgpuExecutorProvider::new(0);
        let mut state = WgpuLlamaState;
        let err = provider
            .forward_hidden(&mut state as &mut dyn GenerationState, 0)
            .unwrap_err();
        assert!(err.to_string().contains("wgpu forward not implemented"));
    }
}

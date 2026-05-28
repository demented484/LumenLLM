use std::any::Any;
use std::fmt::Debug;

use crate::backend::BackendKind;
use crate::error::Result;
use crate::generation::{
    GenerateOutput, GenerateRequest, PrefillStageTimings, SamplingConfig, TimedGenerateOutput,
};
use crate::planning::placement::{ComputePlacement, StoragePlacement};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutorStage {
    Prefill,
    Decode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutorCapability {
    Tokenize,
    DenseEmbedding,
    DenseLmHead,
    RmsNorm,
    Rope,
    Attention,
    Mlp,
    Nvfp4Linear,
    KvCache,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutorBackendInfo {
    pub name: &'static str,
    pub backends: Vec<BackendKind>,
    pub weight_store: Vec<StoragePlacement>,
    pub weight_compute: Vec<ComputePlacement>,
    pub kv_compute: Vec<ComputePlacement>,
    pub capabilities: Vec<ExecutorCapability>,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutorProviderPlan {
    pub info: ExecutorBackendInfo,
    pub runnable: bool,
    pub limitations: Vec<String>,
}

pub trait GenerationState: Debug + Send {
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T> GenerationState for T
where
    T: Debug + Send + 'static,
{
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

pub trait GenerationBackendPrimitives: Debug + Send + Sync {
    fn encode_prompt(&self, prompt: &str) -> Result<Vec<usize>>;
    /// Tokenize raw text — bypasses any chat template. BOS still
    /// prepended. Used by perplexity / pretrain-quality tooling. Default
    /// impl falls back to `encode_prompt` so backends without a chat
    /// template don't have to override.
    fn encode_text_raw(&self, text: &str) -> Result<Vec<usize>> {
        self.encode_prompt(text)
    }
    fn decode_tokens(&self, tokens: &[usize]) -> Result<String>;
    fn is_eos(&self, token: usize) -> bool;
    fn new_sequence_state(&self) -> Result<Box<dyn GenerationState>>;
    fn forward_hidden(&self, state: &mut dyn GenerationState, token_id: usize) -> Result<()>;
    fn forward_logits(&self, state: &mut dyn GenerationState, token_id: usize) -> Result<Vec<f32>>;
    fn prefill_prompt(
        &self,
        state: &mut dyn GenerationState,
        prompt_tokens: &[usize],
        sampling: &SamplingConfig,
    ) -> Result<usize> {
        super::generation::prefill_prompt_token_by_token(self, state, prompt_tokens, sampling)
    }
    fn forward_next_token(
        &self,
        state: &mut dyn GenerationState,
        token_id: usize,
        sampling: &SamplingConfig,
    ) -> Result<usize> {
        let logits = self.forward_logits(state, token_id)?;
        super::generation::sample_next_token(&logits, sampling)
    }
    fn prefill_stage_timings(
        &self,
        _state: &mut dyn GenerationState,
    ) -> Option<PrefillStageTimings> {
        None
    }

    /// Attach multimodal image embeddings for the next prefill. The prefill
    /// embed step splices them into the embedding stream at every position
    /// whose token id matches `injection.image_token_id`. Default no-op.
    fn set_image_injection(
        &self,
        _state: &mut dyn GenerationState,
        _injection: &crate::generation::ImageInjection,
    ) -> Result<()> {
        Ok(())
    }

    /// Attach audio-encoder soft-token embeddings for the next prefill. The
    /// prefill embed step splices them into the embedding stream at every
    /// position whose token id matches `injection.audio_token_id`. Default no-op.
    fn set_audio_injection(
        &self,
        _state: &mut dyn GenerationState,
        _injection: &crate::generation::AudioInjection,
    ) -> Result<()> {
        Ok(())
    }
}

pub trait ModelExecutorBackend: GenerationBackendPrimitives {
    fn info(&self) -> ExecutorBackendInfo;
    fn probe(&self) -> Result<()> {
        Ok(())
    }
    fn generate(&self, request: &GenerateRequest) -> Result<GenerateOutput> {
        super::generation::generate_with_backend(self, request)
    }
    fn generate_timed(&self, request: &GenerateRequest) -> Result<TimedGenerateOutput> {
        super::generation::generate_with_backend_timed(self, request)
    }
}

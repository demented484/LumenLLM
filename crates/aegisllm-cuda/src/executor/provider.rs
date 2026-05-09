use std::sync::Mutex;

use super::planning::{cuda_limitations, dominant_cuda_device};
use super::state::{CudaLlamaExecutor, CudaLlamaState};
use aegisllm_base::artifact::ModelArtifact;
use aegisllm_base::backend::BackendKind;
use crate::cuda::{CudaPrefillAttentionKernel, CudaRuntimeConfig};
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::executor::traits::{
    ExecutorBackendInfo, ExecutorCapability, ExecutorProviderPlan, GenerationBackendPrimitives,
    GenerationState, ModelExecutorBackend,
};
use aegisllm_base::generation::{PrefillStageTimings, SamplingConfig};
use aegisllm_base::graph::ModelGraph;
use aegisllm_base::planning::placement::{ComputePlacement, ResolvedPlacement, StoragePlacement};
use aegisllm_base::planning::runtime::RuntimePlan;
use aegisllm_base::text::TextProcessor;

#[derive(Debug)]
pub struct CudaExecutorProvider {
    device: usize,
    limitations: Vec<String>,
    text: Option<TextProcessor>,
    cuda: Option<CudaLlamaExecutor>,
    /// Warmed sequence state allocated at load time (KV cache + scratch).
    /// Handed out to the first `new_sequence_state()` caller; subsequent
    /// callers allocate fresh. This moves the ~10 GiB KV-cache `cudaMalloc`
    /// off the first prompt's critical path.
    prepared_state: Mutex<Option<CudaLlamaState>>,
}

impl CudaExecutorProvider {
    pub fn new(device: usize, limitations: Vec<String>) -> Self {
        Self {
            device,
            limitations,
            text: None,
            cuda: None,
            prepared_state: Mutex::new(None),
        }
    }

    pub fn from_artifact(
        artifact: &ModelArtifact,
        graph: &ModelGraph,
        placement: &ResolvedPlacement,
        runtime: &RuntimePlan,
        cuda_config: CudaRuntimeConfig,
    ) -> Result<Self> {
        let plan = Self::plan(placement, runtime).ok_or_else(|| {
            AegisError::InvalidPlan("CUDA executor requested for non-CUDA placement".into())
        })?;
        if !plan.runnable {
            return Err(AegisError::Unsupported(format!(
                "CUDA executor plan is not runnable yet: {}",
                plan.limitations.join("; ")
            )));
        }
        let BackendKind::Cuda { device } = plan.info.backends[0] else {
            return Err(AegisError::InvalidPlan(
                "CUDA provider selected without CUDA backend".into(),
            ));
        };
        let kernel_note = if cuda_config.native_mxfp4_repack && cuda_config.native_mxfp4_inference {
            "CUDA executor uses experimental native MXFP4 inference where native resident linears are materialized".into()
        } else {
            "CUDA executor uses prequantized scalar matvec kernels for NVFP4 linears unless native MXFP4 repack+inference are explicitly enabled".into()
        };
        let attention_note = match cuda_config.prefill_attention {
            CudaPrefillAttentionKernel::Auto => {
                "CUDA prefill attention is auto-selected from the architecture policy, with correctness-preserving fallback to reference or Aegis paged-varlen paths until production FA kernels are available".into()
            }
            CudaPrefillAttentionKernel::Off => {
                "CUDA fast prefill attention is disabled; the reference path is used".into()
            }
            CudaPrefillAttentionKernel::FlashAttention2 => {
                "CUDA prefill attention requests the FA2 backend for Ampere/Ada-class GPUs".into()
            }
            CudaPrefillAttentionKernel::FlashAttention3 => {
                "CUDA prefill attention requests the FA3 backend for Hopper-class GPUs".into()
            }
            CudaPrefillAttentionKernel::FlashAttention4 => {
                "CUDA prefill attention requests the Blackwell FA4 backend".into()
            }
            CudaPrefillAttentionKernel::AegisVarlen => {
                "CUDA prefill attention uses the Aegis paged-varlen online-softmax path".into()
            }
            CudaPrefillAttentionKernel::WarpFlash => {
                "CUDA prefill attention prefers the warp cache-only kernel for eligible first chunks and falls back to bounded continuation otherwise".into()
            }
            CudaPrefillAttentionKernel::Reference => {
                "CUDA prefill attention uses the reference scalar kernel".into()
            }
            CudaPrefillAttentionKernel::Continuation => {
                "CUDA prefill attention uses the varlen continuation kernel with bounded shared memory".into()
            }
        };
        let timing_enabled = std::env::var("AEGIS_LOAD_TIMING").is_ok();
        let t0 = std::time::Instant::now();
        let cuda_executor = CudaLlamaExecutor::from_artifact(
            artifact,
            graph,
            placement,
            runtime,
            device,
            cuda_config,
        )?;
        if timing_enabled {
            eprintln!(
                "load-timing: from_artifact total     {:>6.2}s",
                t0.elapsed().as_secs_f64()
            );
        }
        // Pre-allocate the per-sequence state (KV cache, scratch, sampled-token
        // buffer, etc.) so the first prompt doesn't pay for a ~10 GiB cudaMalloc
        // on its critical path. Cached in `prepared_state` and consumed by the
        // first `new_sequence_state()` caller; later callers allocate fresh.
        let t1 = std::time::Instant::now();
        let warmed = cuda_executor.new_state()?;
        if timing_enabled {
            eprintln!(
                "load-timing: warmed new_state        {:>6.2}s",
                t1.elapsed().as_secs_f64()
            );
        }
        Ok(Self {
            device,
            limitations: vec![
                "CUDA reference executor stores KV cache as f16; q8/fp8 KV kernels are not implemented yet".into(),
                kernel_note,
                attention_note,
            ],
            text: Some(TextProcessor::from_artifact(artifact)?),
            cuda: Some(cuda_executor),
            prepared_state: Mutex::new(Some(warmed)),
        })
    }

    pub fn plan(
        placement: &ResolvedPlacement,
        runtime: &RuntimePlan,
    ) -> Option<ExecutorProviderPlan> {
        let device = dominant_cuda_device(placement)?;
        let limitations = cuda_limitations(placement, runtime, device);
        let info = cuda_backend_info(device, limitations.clone());
        Some(ExecutorProviderPlan {
            info,
            runnable: limitations.is_empty(),
            limitations,
        })
    }
}

impl GenerationBackendPrimitives for CudaExecutorProvider {
    fn encode_prompt(&self, prompt: &str) -> Result<Vec<usize>> {
        let text = self.text.as_ref().ok_or_else(|| {
            AegisError::Unsupported(format!(
                "CUDA executor provider is registered but not initialized: {}",
                self.limitations.join("; ")
            ))
        })?;
        text.encode_prompt(prompt)
    }

    fn encode_text_raw(&self, prompt: &str) -> Result<Vec<usize>> {
        let text = self.text.as_ref().ok_or_else(|| {
            AegisError::Unsupported(format!(
                "CUDA executor provider is registered but not initialized: {}",
                self.limitations.join("; ")
            ))
        })?;
        text.encode_text_raw(prompt)
    }

    fn decode_tokens(&self, tokens: &[usize]) -> Result<String> {
        let text = self.text.as_ref().ok_or_else(|| {
            AegisError::Unsupported(format!(
                "CUDA executor provider is registered but not initialized: {}",
                self.limitations.join("; ")
            ))
        })?;
        text.decode_tokens(tokens)
    }

    fn is_eos(&self, token: usize) -> bool {
        self.text
            .as_ref()
            .map(|text| text.is_eos(token))
            .unwrap_or(false)
    }

    fn new_sequence_state(&self) -> Result<Box<dyn GenerationState>> {
        let cuda = self.cuda.as_ref().ok_or_else(|| {
            AegisError::Unsupported(format!(
                "CUDA executor provider is registered but not initialized: {}",
                self.limitations.join("; ")
            ))
        })?;
        // First call after load consumes the state warmed in `from_artifact`.
        // Concurrent callers (or a second sequential request) allocate fresh.
        if let Some(prepared) = self
            .prepared_state
            .lock()
            .ok()
            .and_then(|mut g| g.take())
        {
            return Ok(Box::new(prepared));
        }
        Ok(Box::new(cuda.new_state()?))
    }

    fn forward_hidden(&self, state: &mut dyn GenerationState, token_id: usize) -> Result<()> {
        let cuda = self.cuda.as_ref().ok_or_else(|| {
            AegisError::Unsupported(format!(
                "CUDA executor provider is registered but not initialized: {}",
                self.limitations.join("; ")
            ))
        })?;
        cuda.forward_hidden(cuda_state_mut(state)?, token_id)
    }

    fn forward_logits(&self, state: &mut dyn GenerationState, token_id: usize) -> Result<Vec<f32>> {
        let cuda = self.cuda.as_ref().ok_or_else(|| {
            AegisError::Unsupported(format!(
                "CUDA executor provider is registered but not initialized: {}",
                self.limitations.join("; ")
            ))
        })?;
        cuda.forward_logits(cuda_state_mut(state)?, token_id)
    }

    fn prefill_prompt(
        &self,
        state: &mut dyn GenerationState,
        prompt_tokens: &[usize],
        sampling: &SamplingConfig,
    ) -> Result<usize> {
        let cuda = self.cuda.as_ref().ok_or_else(|| {
            AegisError::Unsupported(format!(
                "CUDA executor provider is registered but not initialized: {}",
                self.limitations.join("; ")
            ))
        })?;
        cuda.prefill_prompt(cuda_state_mut(state)?, prompt_tokens, sampling)
    }

    fn forward_next_token(
        &self,
        state: &mut dyn GenerationState,
        token_id: usize,
        sampling: &SamplingConfig,
    ) -> Result<usize> {
        let cuda = self.cuda.as_ref().ok_or_else(|| {
            AegisError::Unsupported(format!(
                "CUDA executor provider is registered but not initialized: {}",
                self.limitations.join("; ")
            ))
        })?;
        cuda.forward_next_token(cuda_state_mut(state)?, token_id, sampling)
    }

    fn prefill_stage_timings(
        &self,
        state: &mut dyn GenerationState,
    ) -> Option<PrefillStageTimings> {
        cuda_state_mut(state)
            .ok()
            .and_then(|state| state.prefill_timings.snapshot())
    }
}

impl ModelExecutorBackend for CudaExecutorProvider {
    fn info(&self) -> ExecutorBackendInfo {
        cuda_backend_info(self.device, self.limitations.clone())
    }

    fn probe(&self) -> Result<()> {
        let cuda = self.cuda.as_ref().ok_or_else(|| {
            AegisError::Unsupported(format!(
                "CUDA executor provider is registered but not initialized: {}",
                self.limitations.join("; ")
            ))
        })?;
        let _state = cuda.new_state()?;
        Ok(())
    }
}

fn cuda_state_mut(state: &mut dyn GenerationState) -> Result<&mut CudaLlamaState> {
    state
        .as_any_mut()
        .downcast_mut::<CudaLlamaState>()
        .ok_or_else(|| AegisError::InvalidPlan("CUDA executor received foreign state".into()))
}

fn cuda_backend_info(device: usize, limitations: Vec<String>) -> ExecutorBackendInfo {
    ExecutorBackendInfo {
        name: "cuda",
        backends: vec![BackendKind::Cuda { device }],
        weight_store: vec![StoragePlacement::Vram { device }],
        weight_compute: vec![ComputePlacement::Cuda { device }],
        kv_compute: vec![ComputePlacement::Cuda { device }],
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

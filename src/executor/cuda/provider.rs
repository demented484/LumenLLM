use super::planning::{cuda_limitations, dominant_cuda_device};
use super::state::{CudaLlamaExecutor, CudaLlamaState};
use crate::artifact::ModelArtifact;
use crate::backend::BackendKind;
use crate::cuda::{CudaPrefillAttentionKernel, CudaRuntimeConfig};
use crate::error::{AegisError, Result};
use crate::executor::traits::{
    ExecutorBackendInfo, ExecutorCapability, ExecutorProviderPlan, GenerationBackendPrimitives,
    GenerationState, ModelExecutorBackend,
};
use crate::generation::SamplingConfig;
use crate::graph::ModelGraph;
use crate::planning::placement::{ComputePlacement, ResolvedPlacement, StoragePlacement};
use crate::planning::runtime::RuntimePlan;
use crate::text::TextProcessor;

#[derive(Debug)]
pub struct CudaExecutorProvider {
    device: usize,
    limitations: Vec<String>,
    text: Option<TextProcessor>,
    cuda: Option<CudaLlamaExecutor>,
}

impl CudaExecutorProvider {
    pub fn new(device: usize, limitations: Vec<String>) -> Self {
        Self {
            device,
            limitations,
            text: None,
            cuda: None,
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
                "CUDA prefill attention is auto-selected; reference attention is used for short correctness-sensitive chunks and paged varlen FlashAttention is used for longer chunks".into()
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
            CudaPrefillAttentionKernel::FlashVarlen => {
                "CUDA prefill attention uses the paged varlen online-softmax FlashAttention path".into()
            }
            CudaPrefillAttentionKernel::FlashAttention4 => {
                "CUDA prefill attention uses the Blackwell FA4-style tiled paged-varlen path".into()
            }
        };
        Ok(Self {
            device,
            limitations: vec![
                "CUDA reference executor stores KV cache as f16; q8/fp8 KV kernels are not implemented yet".into(),
                kernel_note,
                attention_note,
            ],
            text: Some(TextProcessor::from_artifact(artifact)?),
            cuda: Some(CudaLlamaExecutor::from_artifact(
                artifact,
                graph,
                placement,
                runtime,
                device,
                cuda_config,
            )?),
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

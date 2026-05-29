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
        Self::from_artifact_with_draft(artifact, graph, placement, runtime, cuda_config, None)
    }

    /// Like `from_artifact` but optionally attaches an EAGLE/MTP speculative
    /// decoding draft model (`(draft_model_path, num_draft_tokens)`). When the
    /// draft is `None`, behaves identically to `from_artifact`.
    pub fn from_artifact_with_draft(
        artifact: &ModelArtifact,
        graph: &ModelGraph,
        placement: &ResolvedPlacement,
        runtime: &RuntimePlan,
        cuda_config: CudaRuntimeConfig,
        draft: Option<(&std::path::Path, usize)>,
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
            CudaPrefillAttentionKernel::Fp8 => {
                "CUDA prefill attention requests the FP8 native-MMA backend (requires an FP8 KV cache)".into()
            }
            CudaPrefillAttentionKernel::Mma => {
                "CUDA prefill attention requests the Stage D.1 hand-tuned mma.sync BF16 backend (head_dim=512)".into()
            }
            CudaPrefillAttentionKernel::Continuation => {
                "CUDA prefill attention uses the varlen continuation kernel with bounded shared memory".into()
            }
        };
        // `--cuda-prefill-attention fp8` (CLI) and `attention.compute-quantization:
        // fp8` (config) both resolve to the FP8 attention backend, which reads
        // FP8 K/V directly. The config-field path is validated in the params
        // parser; the CLI flag is applied after parsing, so re-check here that
        // an FP8 attention backend has an FP8 KV cache and reject cleanly if not.
        if cuda_config.resolve_attention_backend()
            == aegisllm_base::cuda_config::AttentionComputeBackend::Fp8
            && placement.kv_cache.quantization
                != aegisllm_base::tensor::quant::KvCacheQuantization::Fp8
        {
            return Err(AegisError::InvalidConfig(format!(
                "FP8 prefill attention was selected (--cuda-prefill-attention fp8 or \
                 attention.compute-quantization=fp8 / AEGIS_ATTN_FP8=1), but the KV cache \
                 resolves to `{}`. The FP8 attention kernel reads FP8 K/V directly — set \
                 the KV cache `type-k`/`type-v` to `fp8`, or use a non-FP8 attention backend.",
                placement.kv_cache.quantization.label()
            )));
        }
        let t0 = std::time::Instant::now();
        let mut cuda_executor = CudaLlamaExecutor::from_artifact(
            artifact,
            graph,
            placement,
            runtime,
            device,
            cuda_config,
        )?;
        eprintln!(
            "load-timing: from_artifact total     {:>6.2}s",
            t0.elapsed().as_secs_f64()
        );
        // Attach the EAGLE/MTP speculative-decoding draft model, if requested.
        // ~135 MiB extra VRAM; shares the target's KV cache (no duplication).
        if let Some((draft_path, num_draft_tokens)) = draft {
            let td = std::time::Instant::now();
            cuda_executor.attach_draft_model(draft_path, num_draft_tokens)?;
            eprintln!(
                "load-timing: draft model attach      {:>6.2}s  (num_draft_tokens={})",
                td.elapsed().as_secs_f64(),
                num_draft_tokens,
            );
        }
        // Pre-allocate the per-sequence state (KV cache, scratch, sampled-token
        // buffer, etc.) so the first prompt doesn't pay for a ~10 GiB cudaMalloc
        // on its critical path. Cached in `prepared_state` and consumed by the
        // first `new_sequence_state()` caller; later callers allocate fresh.
        // When a draft is attached this also allocates the (tiny) draft scratch.
        let t1 = std::time::Instant::now();
        let warmed = cuda_executor.new_spec_state()?;
        eprintln!(
            "load-timing: warmed new_state        {:>6.2}s",
            t1.elapsed().as_secs_f64()
        );
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

    /// Speculative-decoding generate (greedy). Encodes the prompt, allocates a
    /// spec-decode state, runs the draft-propose / target-verify loop, and
    /// decodes the accepted tokens. Used only when a draft model is attached.
    fn generate_speculative(
        &self,
        request: &aegisllm_base::generation::GenerateRequest,
    ) -> Result<aegisllm_base::generation::GenerateOutput> {
        let cuda = self.cuda.as_ref().ok_or_else(|| {
            AegisError::Unsupported("CUDA executor not initialized for spec-decode".into())
        })?;
        let prompt_tokens = self.encode_prompt(&request.prompt)?;
        if prompt_tokens.is_empty() {
            return Err(AegisError::InvalidConfig("prompt produced no tokens".into()));
        }
        // Draft-trace harness: dump the EXACT chat-templated prompt token ids so the
        // vLLM reference can be driven with prompt_token_ids and feed the target an
        // identical context (otherwise input_state_hidden diverges at stage 0).
        // No-op unless AEGIS_DRAFT_TRACE is set.
        if let Ok(dir) = std::env::var("AEGIS_DRAFT_TRACE") {
            let _ = std::fs::create_dir_all(&dir);
            let ids: Vec<String> = prompt_tokens.iter().map(|t| t.to_string()).collect();
            let path = format!("{dir}/prompt_token_ids.json");
            match std::fs::write(&path, format!("[{}]", ids.join(","))) {
                Ok(()) => eprintln!("[draft-trace] wrote {path} ({} tokens)", prompt_tokens.len()),
                Err(e) => eprintln!("[draft-trace] failed to write {path}: {e}"),
            }
        }
        let mut state = self.new_sequence_state()?;
        if let Some(ref injection) = request.image_injection {
            self.set_image_injection(state.as_mut(), injection)?;
        }
        let cuda_state = cuda_state_mut(state.as_mut())?;
        let generated = cuda.generate_speculative_greedy(
            cuda_state,
            &prompt_tokens,
            request,
            &|t| self.is_eos(t),
        )?;
        Ok(aegisllm_base::generation::GenerateOutput {
            text: self.decode_tokens(&generated)?,
            prompt_tokens: prompt_tokens.len(),
            completion_tokens: generated.len(),
            finish_reason: "length".into(),
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
        // When a draft is attached, allocate the spec-decode state (includes the
        // draft scratch); otherwise the plain state.
        Ok(Box::new(cuda.new_spec_state()?))
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

    fn set_image_injection(
        &self,
        state: &mut dyn GenerationState,
        injection: &aegisllm_base::generation::ImageInjection,
    ) -> Result<()> {
        let cuda = self.cuda.as_ref().ok_or_else(|| {
            AegisError::Unsupported("CUDA executor not initialized".into())
        })?;
        let s = cuda_state_mut(state)?;
        // Upload [n_tokens, hidden] row-major f32 to VRAM.
        let buf = cuda.upload_f32(&injection.data)?;
        s.image_embeds = Some(buf);
        s.image_token_id = injection.image_token_id as u32;
        s.image_n_tokens = injection.n_tokens;
        Ok(())
    }

    fn set_audio_injection(
        &self,
        state: &mut dyn GenerationState,
        injection: &aegisllm_base::generation::AudioInjection,
    ) -> Result<()> {
        let cuda = self.cuda.as_ref().ok_or_else(|| {
            AegisError::Unsupported("CUDA executor not initialized".into())
        })?;
        let s = cuda_state_mut(state)?;
        // Upload [n_tokens, hidden] row-major f32 to VRAM.
        let buf = cuda.upload_f32(&injection.data)?;
        s.audio_embeds = Some(buf);
        s.audio_token_id = injection.audio_token_id as u32;
        s.audio_n_tokens = injection.n_tokens;
        Ok(())
    }
}

impl ModelExecutorBackend for CudaExecutorProvider {
    fn info(&self) -> ExecutorBackendInfo {
        cuda_backend_info(self.device, self.limitations.clone())
    }

    /// Route through the speculative-decoding loop when a draft model is
    /// attached; otherwise fall back to the default generic decode loop.
    fn generate(
        &self,
        request: &aegisllm_base::generation::GenerateRequest,
    ) -> Result<aegisllm_base::generation::GenerateOutput> {
        if self.cuda.as_ref().is_some_and(|c| c.has_draft()) {
            return self.generate_speculative(request);
        }
        aegisllm_base::executor::generation::generate_with_backend(self, request)
    }

    fn generate_timed(
        &self,
        request: &aegisllm_base::generation::GenerateRequest,
    ) -> Result<aegisllm_base::generation::TimedGenerateOutput> {
        if self.cuda.as_ref().is_some_and(|c| c.has_draft()) {
            // Spec-decode timing is not yet split into prefill/decode buckets;
            // report the whole run as decode. TODO(gpu-verify): proper timing.
            let start = std::time::Instant::now();
            let output = self.generate_speculative(request)?;
            return Ok(aegisllm_base::generation::TimedGenerateOutput {
                output,
                tokenize_elapsed: std::time::Duration::ZERO,
                prefill_elapsed: std::time::Duration::ZERO,
                decode_elapsed: start.elapsed(),
                total_elapsed: start.elapsed(),
                prefill_stage_timings: None,
            });
        }
        aegisllm_base::executor::generation::generate_with_backend_timed(self, request)
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

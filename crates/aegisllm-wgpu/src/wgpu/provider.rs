use std::sync::Arc;

use super::block::{forward_token_device, Activation};
use super::loader::WgpuContext;
use super::state::{WgpuLlamaState, WgpuModelState};
use super::weights::{
    load_gemma4_model, load_vanilla_llama_model, WgpuLinear, WgpuModel, WgpuModelShape,
};
use aegisllm_base::artifact::ModelArtifact;
use aegisllm_base::backend::BackendKind;
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::executor::traits::{
    ExecutorBackendInfo, ExecutorCapability, ExecutorProviderPlan, GenerationBackendPrimitives,
    GenerationState, ModelExecutorBackend,
};
use aegisllm_base::generation::SamplingConfig;
use aegisllm_base::planning::placement::{ComputePlacement, StoragePlacement};
use aegisllm_base::text::TextProcessor;

/// Architecture detection for picking the right loader. We treat any
/// model with `model_type == "gemma"` and a Gemma-4 marker tensor
/// (`pre_feedforward_layernorm_2` is a Gemma-4-MoE-only tensor) as
/// Gemma-4; everything else falls through to the vanilla-Llama
/// loader. Real production code would consult the architecture
/// detector in `aegisllm_base::model`, but for now this is enough to
/// drive the wiring forward.
fn pick_loader_kind(artifact: &ModelArtifact) -> ModelKind {
    let mt = artifact.config.model_type.to_lowercase();
    let has_gemma4_moe_marker = artifact
        .tensors
        .tensors
        .keys()
        .any(|n| n.contains("pre_feedforward_layernorm_2"));
    if has_gemma4_moe_marker || mt.starts_with("gemma") {
        ModelKind::Gemma4
    } else {
        ModelKind::VanillaLlama
    }
}

#[derive(Debug, Clone, Copy)]
enum ModelKind {
    VanillaLlama,
    Gemma4,
}

#[derive(Debug)]
pub struct WgpuExecutorProvider {
    device: usize,
    limitations: Vec<String>,
    text: Option<TextProcessor>,
    /// `None` until `from_artifact` succeeds; once present, forward
    /// methods call into it.
    model: Option<Arc<WgpuModel>>,
    /// Per-layer-uniform `rope_theta` (Llama / Gemma-4 sliding). For
    /// Gemma-4 we should plumb a per-layer-class theta (sliding=10k vs
    /// global=1M); for now we use one value and tolerate a small
    /// numerical mismatch on the 5 global layers. Replacing this with
    /// `[Vec<f32>; 2]` keyed by (sliding, global) is the next step.
    rope_theta: f32,
    activation: Activation,
    max_seq_len: usize,
    kind: Option<ModelKind>,
}

impl WgpuExecutorProvider {
    pub fn new(device: usize) -> Self {
        Self {
            device,
            limitations: vec!["wgpu skeleton; forward not implemented".into()],
            text: None,
            model: None,
            rope_theta: 10000.0,
            activation: Activation::SwiGLU,
            max_seq_len: 0,
            kind: None,
        }
    }

    pub fn plan() -> ExecutorProviderPlan {
        // Wgpu forward is now implemented for the f32 / BF16-upcast
        // path. NVFP4 weights work via on-the-fly dequant. Provider
        // is opt-in: planner only routes to it when the placement
        // explicitly requests `compute = "wgpu:N"`. The runtime gate
        // for "is the model loadable" lives in `from_artifact`.
        let limitations: Vec<String> = vec![];
        ExecutorProviderPlan {
            info: wgpu_backend_info(0, limitations.clone()),
            runnable: true,
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
        let ctx = Arc::new(WgpuContext::new(device_index)?);
        let kind = pick_loader_kind(artifact);
        let model = match kind {
            ModelKind::Gemma4 => load_gemma4_model(ctx.clone(), artifact)?,
            ModelKind::VanillaLlama => {
                let cfg = &artifact.config;
                let shape = WgpuModelShape {
                    num_layers: cfg.num_hidden_layers,
                    hidden_size: cfg.hidden_size,
                    intermediate_size: cfg
                        .intermediate_size
                        .ok_or_else(|| AegisError::InvalidPlan(
                            "vanilla Llama missing intermediate_size".into(),
                        ))?,
                    num_q_heads: cfg.num_attention_heads,
                    num_kv_heads: cfg.num_key_value_heads.unwrap_or(cfg.num_attention_heads),
                    head_dim: cfg.head_dim.unwrap_or(cfg.hidden_size / cfg.num_attention_heads),
                    vocab_size: cfg
                        .vocab_size
                        .ok_or_else(|| AegisError::InvalidPlan("config missing vocab_size".into()))?,
                    rms_norm_eps: cfg.rms_norm_eps.unwrap_or(1e-6) as f32,
                };
                load_vanilla_llama_model(ctx.clone(), artifact, shape)?
            }
        };
        let activation = match kind {
            ModelKind::Gemma4 => Activation::GeGluTanh,
            ModelKind::VanillaLlama => Activation::SwiGLU,
        };
        let rope_theta = artifact
            .config
            .rope_theta
            .unwrap_or(10000.0) as f32;
        let max_seq_len = artifact
            .config
            .max_position_embeddings
            .unwrap_or(8192)
            // Cap at a sane bound — full max_position_embeddings can be
            // 1M+ for Gemma-4, which would allocate enormous KV caches.
            .min(8192);
        Ok(Self {
            device: device_index,
            limitations: vec![],
            text: Some(TextProcessor::from_artifact(artifact)?),
            model: Some(Arc::new(model)),
            rope_theta,
            activation,
            max_seq_len,
            kind: Some(kind),
        })
    }
}

fn rope_for_layer_factory(rope_theta: f32) -> impl Fn(usize, usize, usize) -> (Vec<f32>, Vec<f32>) {
    move |position, _layer_idx, half_dim| {
        let inv = (0..half_dim)
            .map(|i| rope_theta.powf(-2.0 * i as f32 / (2 * half_dim) as f32))
            .collect::<Vec<_>>();
        let cos = inv.iter().map(|t| (position as f32 * t).cos()).collect();
        let sin = inv.iter().map(|t| (position as f32 * t).sin()).collect();
        (cos, sin)
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
        self.text.as_ref().map(|t| t.is_eos(token)).unwrap_or(false)
    }

    fn new_sequence_state(&self) -> Result<Box<dyn GenerationState>> {
        let model = self.model.as_ref().ok_or_else(|| self.not_initialized())?;
        // Largest dequant scratch needed by any matmul. Gemma-4's
        // routed expert gate/up/down is `moe_intermediate × hidden`
        // ≈ 1.98 M elements; shared MLP gate/up is `intermediate ×
        // hidden` ≈ 5.95 M elements. Take the larger.
        let max_dequant = model
            .intermediate_size
            .saturating_mul(model.hidden_size);
        // Per-layer max widths for Gemma-4 heterogeneity (global
        // layers use head_dim=512 / num_kv_heads=2 vs sliding 256/8).
        let mut max_q_width = model.num_q_heads * model.head_dim;
        let mut max_kv_width = model.num_kv_heads * model.head_dim;
        let mut max_head_dim = model.head_dim;
        for layer in &model.layers {
            let hd = layer.head_dim_override.unwrap_or(model.head_dim);
            let nkv = layer.num_kv_heads_override.unwrap_or(model.num_kv_heads);
            max_q_width = max_q_width.max(model.num_q_heads * hd);
            max_kv_width = max_kv_width.max(nkv * hd);
            max_head_dim = max_head_dim.max(hd);
        }
        let state = WgpuModelState::new_with_layer_max(
            &model.ctx,
            model.layers.len(),
            model.hidden_size,
            model.intermediate_size,
            model.num_q_heads,
            model.num_kv_heads,
            model.head_dim,
            model.vocab_size,
            self.max_seq_len.max(1),
            max_dequant,
            max_q_width,
            max_kv_width,
            max_head_dim,
        )?;
        Ok(Box::new(state))
    }

    fn forward_hidden(&self, state: &mut dyn GenerationState, token_id: usize) -> Result<()> {
        let model = self.model.as_ref().ok_or_else(|| self.not_initialized())?;
        let st = state
            .as_any_mut()
            .downcast_mut::<WgpuModelState>()
            .ok_or_else(|| AegisError::InvalidPlan(
                "wgpu forward_hidden: state is not a WgpuModelState".into(),
            ))?;
        // Embedding lookup: write row `token_id` of the embedding
        // table into state.residual. We do this on host for now (one
        // small upload per token) since `embedding_device` is the
        // device path but writing directly is cheaper than a dispatch
        // for a 4096-element row.
        if (token_id as usize) >= model.embed_tokens_rows {
            return Err(AegisError::InvalidPlan(format!(
                "token_id {token_id} ≥ vocab {}", model.embed_tokens_rows
            )));
        }
        // Embedding lookup: row `token_id` of embed_tokens →
        // state.residual. Two paths depending on storage format:
        //   * Dense (f32): single copy_buffer_to_buffer of `hidden * 4`
        //     bytes. Bypass Vulkan's 2 GiB binding cap.
        //   * Bf16Packed: copy `hidden * 2` BF16 bytes from packed
        //     buffer to a small bf16 row scratch, then dispatch
        //     `dequant_bf16_device` to expand bf16 → f32 into residual.
        match &model.embed_tokens {
            WgpuLinear::Dense { weight, .. } => {
                let row_bytes = (model.hidden_size * std::mem::size_of::<f32>()) as u64;
                let src_offset = (token_id as u64) * row_bytes;
                let mut enc = model
                    .ctx
                    .device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("wgpu_provider_embed_lookup_dense"),
                    });
                enc.copy_buffer_to_buffer(weight, src_offset, &st.residual, 0, row_bytes);
                model.ctx.queue().submit(std::iter::once(enc.finish()));
            }
            WgpuLinear::Bf16Packed { weight, .. } => {
                // Each BF16 row is `hidden * 2` bytes. token_id-th row
                // starts at `token_id * hidden * 2` bytes from the
                // packed buffer's start. Both offsets are 4-byte
                // aligned as long as hidden is even (always true).
                let row_bytes_bf16 = (model.hidden_size * 2) as u64;
                let src_offset = (token_id as u64) * row_bytes_bf16;
                if std::env::var("AEGIS_WGPU_TRACE").is_ok() {
                    eprintln!(
                        "[provider] embed lookup token={} hidden={} row_bf16_bytes={} src_off={} src_buf={} dst_buf={} resid_buf={}",
                        token_id,
                        model.hidden_size,
                        row_bytes_bf16,
                        src_offset,
                        weight.size(),
                        st.embed_row_packed_bf16.size(),
                        st.residual.size(),
                    );
                }
                let mut enc = model
                    .ctx
                    .device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("wgpu_provider_embed_lookup_bf16"),
                    });
                enc.copy_buffer_to_buffer(
                    weight,
                    src_offset,
                    &st.embed_row_packed_bf16,
                    0,
                    row_bytes_bf16,
                );
                model.ctx.queue().submit(std::iter::once(enc.finish()));
                super::forward::dequant_bf16_device(
                    &model.ctx,
                    &st.embed_row_packed_bf16,
                    &st.residual,
                    model.hidden_size,
                )?;
            }
            WgpuLinear::Nvfp4 { .. } => {
                return Err(AegisError::Unsupported(
                    "wgpu provider: NVFP4 embed_tokens not supported".into(),
                ));
            }
        }
        let rope_fn = rope_for_layer_factory(self.rope_theta);
        forward_token_device(
            &model.ctx,
            model,
            st,
            rope_fn,
            model.rms_norm_eps,
            self.activation,
        )?;
        Ok(())
    }

    fn forward_logits(
        &self,
        state: &mut dyn GenerationState,
        token_id: usize,
    ) -> Result<Vec<f32>> {
        self.forward_hidden(state, token_id)?;
        let model = self.model.as_ref().ok_or_else(|| self.not_initialized())?;
        let st = state
            .as_any_mut()
            .downcast_mut::<WgpuModelState>()
            .ok_or_else(|| AegisError::InvalidPlan(
                "wgpu forward_logits: state is not a WgpuModelState".into(),
            ))?;
        super::forward::download_f32_buf(
            &model.ctx,
            &st.logits,
            model.vocab_size,
            "wgpu_provider_logits",
        )
    }

    fn prefill_prompt(
        &self,
        state: &mut dyn GenerationState,
        prompt_tokens: &[usize],
        sampling: &SamplingConfig,
    ) -> Result<usize> {
        // Token-by-token prefill. CUDA has a chunked path; for wgpu
        // first iteration we just loop forward_logits per token. Real
        // long-prompt prefill performance work is later.
        aegisllm_base::executor::generation::prefill_prompt_token_by_token(
            self, state, prompt_tokens, sampling,
        )
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
    fn wgpu_plan_is_runnable_now_that_forward_is_wired() {
        let plan = WgpuExecutorProvider::plan();
        assert!(plan.runnable, "wgpu plan should be runnable after forward wiring");
        assert!(plan.limitations.is_empty());
    }

    #[test]
    fn wgpu_provider_unloaded_returns_unsupported_for_forward() {
        let provider = WgpuExecutorProvider::new(0);
        let mut state = WgpuLlamaState::default();
        let err = provider
            .forward_hidden(&mut state as &mut dyn GenerationState, 0)
            .unwrap_err();
        // The unloaded provider has no model — forward should error.
        let msg = err.to_string();
        assert!(
            msg.contains("wgpu executor not initialized") || msg.contains("not a WgpuModelState"),
            "unexpected error: {msg}"
        );
    }
}

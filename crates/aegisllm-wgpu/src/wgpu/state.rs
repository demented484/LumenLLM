use aegisllm_base::error::{AegisError, Result};

use super::forward::alloc_storage;
use super::loader::WgpuContext;

/// Per-sequence decode state owned by the wgpu backend.
///
/// Holds the persistent device buffers that need to survive across
/// `forward_*` calls within one generation session. Sized once at
/// construction from the model's hidden/intermediate dimensions and
/// max sequence length; per-call kernels read/write these buffers
/// without ever round-tripping to host.
///
/// `Default` is provided so the unsupported-forward stub paths in the
/// provider can hand back an empty state — those callers must not invoke
/// any device kernel that expects allocated buffers. Real forward callers
/// build a state via [`WgpuLlamaState::new_for_dense_mlp`] (this skeleton)
/// or [`WgpuLlamaState::new_for_model`] (once attention/KV are wired).
#[derive(Default)]
pub struct WgpuLlamaState {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub max_seq_len: usize,
    pub position: usize,
    /// Live activation: the layer-input residual stream (`[hidden_size]`).
    /// Forward primitives chain via this buffer + scratch.
    pub residual: Option<wgpu::Buffer>,
    /// Scratch for the post-norm output.
    pub post_normed: Option<wgpu::Buffer>,
    /// Scratch for the gate-projection output (`[intermediate_size]`).
    pub gate: Option<wgpu::Buffer>,
    /// Scratch for the up-projection output (`[intermediate_size]`).
    pub up: Option<wgpu::Buffer>,
    /// Scratch for the SwiGLU output (`[intermediate_size]`).
    pub swiglu_out: Option<wgpu::Buffer>,
    /// Scratch for the down-projection output (`[hidden_size]`).
    pub mlp_out: Option<wgpu::Buffer>,
}

impl std::fmt::Debug for WgpuLlamaState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuLlamaState")
            .field("hidden_size", &self.hidden_size)
            .field("intermediate_size", &self.intermediate_size)
            .field("max_seq_len", &self.max_seq_len)
            .field("position", &self.position)
            .field("buffers_allocated", &self.residual.is_some())
            .finish()
    }
}

impl WgpuLlamaState {
    /// Allocate state buffers sized for a dense (non-MoE) MLP block.
    ///
    /// Allocates: `residual[hidden]`, `post_normed[hidden]`,
    /// `gate[intermediate]`, `up[intermediate]`, `swiglu_out[intermediate]`,
    /// `mlp_out[hidden]`. KV cache is **not** allocated here — this
    /// constructor exists for the dense-MLP-block forward skeleton; full
    /// model state (attention + KV) ships separately once the attention
    /// device path lands.
    pub fn new_for_dense_mlp(
        ctx: &WgpuContext,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Self> {
        if hidden_size == 0 || intermediate_size == 0 {
            return Err(AegisError::InvalidPlan(
                "WgpuLlamaState requires non-zero hidden_size and intermediate_size".into(),
            ));
        }
        let h_bytes = (hidden_size * std::mem::size_of::<f32>()) as u64;
        let i_bytes = (intermediate_size * std::mem::size_of::<f32>()) as u64;
        Ok(Self {
            hidden_size,
            intermediate_size,
            max_seq_len: 0,
            position: 0,
            residual: Some(alloc_storage(ctx, h_bytes, "wgpu state residual")),
            post_normed: Some(alloc_storage(ctx, h_bytes, "wgpu state post_normed")),
            gate: Some(alloc_storage(ctx, i_bytes, "wgpu state gate")),
            up: Some(alloc_storage(ctx, i_bytes, "wgpu state up")),
            swiglu_out: Some(alloc_storage(ctx, i_bytes, "wgpu state swiglu")),
            mlp_out: Some(alloc_storage(ctx, h_bytes, "wgpu state mlp_out")),
        })
    }

    /// Reset position counter without dropping buffers — for sequential
    /// independent sequences that reuse the same state.
    pub fn reset_position(&mut self) {
        self.position = 0;
    }
}

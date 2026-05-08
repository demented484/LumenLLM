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

    // ── Attention-block fields (only populated when state is built via
    // `new_for_full_layer`) ─────────────────────────────────────────────
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// `[num_q_heads * head_dim]`. Holds the Q projection of the current
    /// token; mutated in-place by RoPE before attention.
    pub attn_q: Option<wgpu::Buffer>,
    /// Persistent KV cache. Layout: keys live at `[0, max_seq_len *
    /// kv_width)` (kv_width = num_kv_heads * head_dim), values live at
    /// `[max_seq_len * kv_width, 2 * max_seq_len * kv_width)`. Each
    /// decode token writes its K/V at offset `position * kv_width` in
    /// the respective region.
    pub attn_kv_cache: Option<wgpu::Buffer>,
    /// `[num_q_heads * head_dim]`. Holds the attention output before O
    /// projection.
    pub attn_out: Option<wgpu::Buffer>,
    /// `[num_kv_heads * head_dim]`. Holds the K projection of the current
    /// token before it's written to the cache + RoPE'd.
    pub attn_k_new: Option<wgpu::Buffer>,
    /// `[num_kv_heads * head_dim]`. Holds the V projection of the current
    /// token before it's written to the cache.
    pub attn_v_new: Option<wgpu::Buffer>,
    /// `[head_dim / 2]`. RoPE cosine table for the current position.
    /// Re-uploaded per token by the attention-block forward.
    pub rope_cos: Option<wgpu::Buffer>,
    /// `[head_dim / 2]`. RoPE sine table for the current position.
    pub rope_sin: Option<wgpu::Buffer>,
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
            num_q_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            attn_q: None,
            attn_kv_cache: None,
            attn_out: None,
            attn_k_new: None,
            attn_v_new: None,
            rope_cos: None,
            rope_sin: None,
            residual: Some(alloc_storage(ctx, h_bytes, "wgpu state residual")),
            post_normed: Some(alloc_storage(ctx, h_bytes, "wgpu state post_normed")),
            gate: Some(alloc_storage(ctx, i_bytes, "wgpu state gate")),
            up: Some(alloc_storage(ctx, i_bytes, "wgpu state up")),
            swiglu_out: Some(alloc_storage(ctx, i_bytes, "wgpu state swiglu")),
            mlp_out: Some(alloc_storage(ctx, h_bytes, "wgpu state mlp_out")),
        })
    }

    /// Allocate state buffers sized for a full Llama-style layer (attention
    /// + dense MLP). KV cache is per-state for now (single layer); a real
    /// model with N layers would need N caches, which a future
    /// `WgpuModelState` will own.
    ///
    /// `max_seq_len` bounds the persistent KV cache; passing more than
    /// this many decode tokens will be rejected by the attention forward.
    #[allow(clippy::too_many_arguments)]
    pub fn new_for_full_layer(
        ctx: &WgpuContext,
        hidden_size: usize,
        intermediate_size: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        if hidden_size == 0 || intermediate_size == 0 || max_seq_len == 0 {
            return Err(AegisError::InvalidPlan(
                "WgpuLlamaState::new_for_full_layer requires non-zero shapes".into(),
            ));
        }
        if num_q_heads == 0 || num_kv_heads == 0 || head_dim == 0 {
            return Err(AegisError::InvalidPlan(
                "WgpuLlamaState::new_for_full_layer requires non-zero head shapes".into(),
            ));
        }
        if num_q_heads % num_kv_heads != 0 {
            return Err(AegisError::InvalidPlan(format!(
                "num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )));
        }
        if head_dim % 2 != 0 {
            return Err(AegisError::InvalidPlan(format!(
                "RoPE requires even head_dim, got {head_dim}"
            )));
        }
        let mut s = Self::new_for_dense_mlp(ctx, hidden_size, intermediate_size)?;
        let kv_width = num_kv_heads * head_dim;
        let q_width = num_q_heads * head_dim;
        let q_bytes = (q_width * std::mem::size_of::<f32>()) as u64;
        let kv_bytes = (kv_width * std::mem::size_of::<f32>()) as u64;
        let cache_bytes = (2 * max_seq_len * kv_width * std::mem::size_of::<f32>()) as u64;
        let half = head_dim / 2;
        let half_bytes = (half * std::mem::size_of::<f32>()) as u64;

        s.num_q_heads = num_q_heads;
        s.num_kv_heads = num_kv_heads;
        s.head_dim = head_dim;
        s.max_seq_len = max_seq_len;
        s.attn_q = Some(alloc_storage(ctx, q_bytes, "wgpu state attn_q"));
        s.attn_out = Some(alloc_storage(ctx, q_bytes, "wgpu state attn_out"));
        s.attn_k_new = Some(alloc_storage(ctx, kv_bytes, "wgpu state attn_k_new"));
        s.attn_v_new = Some(alloc_storage(ctx, kv_bytes, "wgpu state attn_v_new"));
        s.attn_kv_cache = Some(alloc_storage(ctx, cache_bytes, "wgpu state attn_kv_cache"));
        s.rope_cos = Some(alloc_storage(ctx, half_bytes, "wgpu state rope_cos"));
        s.rope_sin = Some(alloc_storage(ctx, half_bytes, "wgpu state rope_sin"));
        Ok(s)
    }

    /// Reset position counter without dropping buffers — for sequential
    /// independent sequences that reuse the same state.
    pub fn reset_position(&mut self) {
        self.position = 0;
    }
}

/// Multi-layer decode state for a whole model. Owns the shared
/// per-layer scratch (residual, norms, QKV scratch, MLP scratch) plus
/// one persistent KV cache per layer plus the final-stage outputs
/// (final-norm scratch + logits buffer).
///
/// The forward orchestration (`forward_layer_device` in `block.rs`)
/// reads the layer-specific KV cache by index and uses the shared
/// scratch in-place across all N layers — kernels are sequenced on the
/// wgpu queue, so no cross-layer aliasing concerns.
pub struct WgpuModelState {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    /// Decode token counter (0-indexed). Bumped by the caller after a
    /// layer-block forward sequence completes.
    pub position: usize,

    // ── Shared activations / scratch (re-used per layer) ──────────────────
    pub residual: wgpu::Buffer,         // [hidden_size]
    pub post_normed: wgpu::Buffer,      // [hidden_size]
    pub mlp_out: wgpu::Buffer,          // [hidden_size]
    pub attn_q: wgpu::Buffer,           // [num_q_heads * head_dim]
    pub attn_k_new: wgpu::Buffer,       // [num_kv_heads * head_dim]
    pub attn_v_new: wgpu::Buffer,       // [num_kv_heads * head_dim]
    pub attn_out: wgpu::Buffer,         // [num_q_heads * head_dim]
    pub gate: wgpu::Buffer,             // [intermediate_size]
    pub up: wgpu::Buffer,               // [intermediate_size]
    pub swiglu_out: wgpu::Buffer,       // [intermediate_size]
    pub rope_cos: wgpu::Buffer,         // [head_dim / 2]
    pub rope_sin: wgpu::Buffer,         // [head_dim / 2]

    // ── Per-layer KV caches ───────────────────────────────────────────────
    /// `kv_caches[L]` is layer L's persistent cache: keys at
    /// `[0, max_seq_len * kv_width)`, values at
    /// `[max_seq_len * kv_width, 2 * max_seq_len * kv_width)`,
    /// kv_width = num_kv_heads * head_dim.
    pub kv_caches: Vec<wgpu::Buffer>,

    // ── Final-stage outputs ──────────────────────────────────────────────
    pub final_normed: wgpu::Buffer,     // [hidden_size]
    pub logits: wgpu::Buffer,           // [vocab_size]
}

impl std::fmt::Debug for WgpuModelState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuModelState")
            .field("hidden_size", &self.hidden_size)
            .field("intermediate_size", &self.intermediate_size)
            .field("num_q_heads", &self.num_q_heads)
            .field("num_kv_heads", &self.num_kv_heads)
            .field("head_dim", &self.head_dim)
            .field("vocab_size", &self.vocab_size)
            .field("max_seq_len", &self.max_seq_len)
            .field("position", &self.position)
            .field("num_layer_caches", &self.kv_caches.len())
            .finish()
    }
}

impl WgpuModelState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &WgpuContext,
        num_layers: usize,
        hidden_size: usize,
        intermediate_size: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        vocab_size: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        if num_layers == 0
            || hidden_size == 0
            || intermediate_size == 0
            || num_q_heads == 0
            || num_kv_heads == 0
            || head_dim == 0
            || vocab_size == 0
            || max_seq_len == 0
        {
            return Err(AegisError::InvalidPlan(
                "WgpuModelState requires all shapes to be non-zero".into(),
            ));
        }
        if num_q_heads % num_kv_heads != 0 {
            return Err(AegisError::InvalidPlan(format!(
                "num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )));
        }
        if head_dim % 2 != 0 {
            return Err(AegisError::InvalidPlan(format!(
                "RoPE requires even head_dim, got {head_dim}"
            )));
        }
        let h_bytes = (hidden_size * 4) as u64;
        let i_bytes = (intermediate_size * 4) as u64;
        let q_bytes = (num_q_heads * head_dim * 4) as u64;
        let kv_width = num_kv_heads * head_dim;
        let kv_bytes = (kv_width * 4) as u64;
        let cache_bytes = (2 * max_seq_len * kv_width * 4) as u64;
        let half = head_dim / 2;
        let half_bytes = (half * 4) as u64;
        let v_bytes = (vocab_size * 4) as u64;

        let kv_caches = (0..num_layers)
            .map(|_| alloc_storage(ctx, cache_bytes, "model state kv_cache"))
            .collect::<Vec<_>>();

        Ok(Self {
            hidden_size,
            intermediate_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            vocab_size,
            max_seq_len,
            position: 0,
            residual: alloc_storage(ctx, h_bytes, "model state residual"),
            post_normed: alloc_storage(ctx, h_bytes, "model state post_normed"),
            mlp_out: alloc_storage(ctx, h_bytes, "model state mlp_out"),
            attn_q: alloc_storage(ctx, q_bytes, "model state attn_q"),
            attn_k_new: alloc_storage(ctx, kv_bytes, "model state attn_k_new"),
            attn_v_new: alloc_storage(ctx, kv_bytes, "model state attn_v_new"),
            attn_out: alloc_storage(ctx, q_bytes, "model state attn_out"),
            gate: alloc_storage(ctx, i_bytes, "model state gate"),
            up: alloc_storage(ctx, i_bytes, "model state up"),
            swiglu_out: alloc_storage(ctx, i_bytes, "model state swiglu"),
            rope_cos: alloc_storage(ctx, half_bytes, "model state rope_cos"),
            rope_sin: alloc_storage(ctx, half_bytes, "model state rope_sin"),
            kv_caches,
            final_normed: alloc_storage(ctx, h_bytes, "model state final_normed"),
            logits: alloc_storage(ctx, v_bytes, "model state logits"),
        })
    }

    pub fn reset_position(&mut self) {
        self.position = 0;
    }
}

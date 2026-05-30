//! Gemma-4 CPU executor & runtime-state structs. Mirrors the data the CUDA
//! `CudaLlamaExecutor` / `CudaLayer` / `PleGlobal` / `CudaMoE` carry, but in
//! host f32 / BF16 form (`crates/aegisllm-cuda/src/executor/state.rs`).

use super::linear::CpuLinear;
use super::rope::G4RopeConfig;
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::executor::tensors::Bf16Matrix;

/// Top-level Gemma-4 CPU executor.
///
/// Public so the hybrid (CPU+GPU heterogeneous) executor in the `aegisllm`
/// crate can hold one and drive its per-layer block API (`token_entry_host`,
/// `forward_dense_layer_host`, `final_logits_host`) for the CPU-scheduled
/// layers of a Gemma-4 dense forward. Fields stay crate-private; the hybrid
/// only touches the public method surface.
#[derive(Debug)]
pub struct G4CpuExecutor {
    pub(crate) hidden_size: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) rms_norm_eps: f32,
    /// Multiplicative embed scale = sqrt(hidden_size).
    pub(crate) embed_scale: f32,
    /// Final lm_head logit softcap (30.0); applied as cap*tanh(logits/cap).
    pub(crate) lm_head_softcap: Option<f32>,
    pub(crate) embed_tokens: Bf16Matrix,
    pub(crate) final_norm: Vec<f32>,
    pub(crate) lm_head: Bf16Matrix,
    pub(crate) layers: Vec<G4CpuLayer>,
    pub(crate) kv_context_size: usize,
    /// PLE global apparatus (E4B / E2B); None for dense 31B and 26B-A4B.
    pub(crate) ple: Option<G4PleGlobal>,
    /// Largest expert/dense intermediate size, for batched-prefill scratch
    /// sizing (`new_prefill_scratch`).
    pub(crate) max_intermediate: usize,
}

#[derive(Debug)]
pub(crate) struct G4CpuLayer {
    // ── Norms (Gemma-4 PrePost) ────────────────────────────────────────
    pub(crate) input_norm_weight: Vec<f32>,
    /// HF `pre_feedforward_layernorm` (pre-MLP norm). Named to match the CUDA
    /// `post_attention_norm_weight` field role.
    pub(crate) pre_mlp_norm_weight: Vec<f32>,
    /// Post-attention sublayer norm (Gemma-4 `post_attention_layernorm`).
    pub(crate) post_attn_sublayer_norm: Option<Vec<f32>>,
    /// Post-MLP sublayer norm (Gemma-4 `post_feedforward_layernorm`).
    pub(crate) post_mlp_sublayer_norm: Option<Vec<f32>>,

    // ── Attention projections ──────────────────────────────────────────
    pub(crate) q_proj: CpuLinear,
    pub(crate) k_proj: CpuLinear,
    pub(crate) v_proj: CpuLinear,
    pub(crate) o_proj: CpuLinear,
    pub(crate) q_norm_weight: Option<Vec<f32>>,
    pub(crate) k_norm_weight: Option<Vec<f32>>,

    // ── Dense MLP (non-MoE layers / not used for MoE) ──────────────────
    pub(crate) mlp: Option<G4DenseMlp>,
    /// MoE block (26B-A4B); replaces the dense MLP when present.
    pub(crate) moe: Option<G4MoeLayer>,

    // ── Per-layer geometry ─────────────────────────────────────────────
    pub(crate) layer_head_dim: usize,
    pub(crate) layer_num_kv_heads: usize,
    pub(crate) window_size: usize,
    pub(crate) partial_dim: usize,
    pub(crate) rope: G4RopeConfig,
    pub(crate) layer_scalar: Option<f32>,
    /// KV-share parent layer index (E4B last layers); None = own KV.
    pub(crate) kv_shared_from: Option<usize>,

    // ── Per-layer PLE weights (E4B/E2B) ────────────────────────────────
    pub(crate) ple: Option<G4PleLayer>,
}

#[derive(Debug)]
pub(crate) struct G4DenseMlp {
    pub(crate) gate_proj: CpuLinear,
    pub(crate) up_proj: CpuLinear,
    pub(crate) down_proj: CpuLinear,
}

#[derive(Debug)]
pub(crate) struct G4MoeExpert {
    pub(crate) gate_proj: CpuLinear,
    pub(crate) up_proj: CpuLinear,
    pub(crate) down_proj: CpuLinear,
}

#[derive(Debug)]
pub(crate) struct G4MoeLayer {
    /// Router weight matrix [num_experts, hidden_size], BF16.
    pub(crate) router: Bf16Matrix,
    /// Gemma-4 per-input-dim scale applied to router input BEFORE projection.
    pub(crate) router_input_scale: Option<Vec<f32>>,
    /// Gemma-4 per-expert scale applied AFTER softmax+topk+renorm.
    pub(crate) router_per_expert_scale: Option<Vec<f32>>,
    pub(crate) experts: Vec<G4MoeExpert>,
    pub(crate) shared_expert: Option<G4DenseMlp>,
    pub(crate) top_k: usize,
    pub(crate) num_experts: usize,
    pub(crate) expert_intermediate_size: usize,
    /// MoE sublayer norms (Gemma-4 26B two-stream combine).
    pub(crate) post_feedforward_layernorm_1: Option<Vec<f32>>,
    pub(crate) pre_feedforward_layernorm_2: Option<Vec<f32>>,
    pub(crate) post_feedforward_layernorm_2: Option<Vec<f32>>,
}

/// PLE global apparatus (token-entry compute).
#[derive(Debug)]
pub(crate) struct G4PleGlobal {
    /// `embed_tokens_per_layer.weight` — `[vocab, num_layers * ple_dim]` BF16.
    pub(crate) embed_table: Bf16Matrix,
    /// `per_layer_model_projection.weight` — `[num_layers*ple_dim, hidden]` BF16.
    pub(crate) model_projection: Bf16Matrix,
    /// `per_layer_projection_norm.weight` — `[ple_dim]` f32.
    pub(crate) projection_norm: Vec<f32>,
    pub(crate) ple_dim: usize,
    pub(crate) embed_scale_per_layer: f32,   // sqrt(ple_dim)
    pub(crate) model_projection_scale: f32,  // 1/sqrt(hidden)
    pub(crate) combine_scale: f32,           // 1/sqrt(2)
}

/// Per-layer PLE weights.
#[derive(Debug)]
pub(crate) struct G4PleLayer {
    /// `per_layer_input_gate.weight` — `[ple_dim, hidden]` BF16.
    pub(crate) input_gate: Bf16Matrix,
    /// `per_layer_projection.weight` — `[hidden, ple_dim]` BF16.
    pub(crate) projection: Bf16Matrix,
    /// `post_per_layer_input_norm.weight` — `[hidden]` f32.
    pub(crate) post_norm: Vec<f32>,
}

// ── Runtime state ──────────────────────────────────────────────────────────

/// Per-sequence runtime state for `G4CpuExecutor`. Public for the hybrid
/// executor (which owns the CPU-side state for a heterogeneous forward).
/// Fields stay crate-private.
#[derive(Debug)]
pub struct G4CpuState {
    pub(crate) position: usize,
    pub(crate) layers: Vec<G4LayerKvState>,
    /// `[num_layers * ple_dim]` per-token PLE feed; empty for non-PLE models.
    pub(crate) per_layer_inputs: Vec<f32>,
}

/// Per-layer KV cache (linear, position-indexed; correctness-first, no ring
/// buffer). Width is per-layer (`layer_num_kv_heads * layer_head_dim`).
#[derive(Debug)]
pub(crate) struct G4LayerKvState {
    pub(crate) keys: Vec<f32>,
    pub(crate) values: Vec<f32>,
    pub(crate) seq_len: usize,
    pub(crate) kv_width: usize,
}

/// Reusable per-chunk scratch for batched prefill. Every buffer is allocated
/// ONCE (sized to `PREFILL_MAX_BATCH * width`) and reused across ALL layers and
/// chunks, so the hot prefill loop performs ZERO per-layer/per-chunk buffer
/// allocation or zero-init. Each field is a DISTINCT pool slot; buffers that are
/// live simultaneously in the same expression occupy different slots (verified
/// in `forward.rs`). Every region a slot exposes is FULLY OVERWRITTEN by the
/// first writer before any read (matmul_into / rms_norm_into / geglu_into /
/// add_into all overwrite their whole output), so stale content from a prior
/// layer/chunk is never observed — the reuse is semantically identical to a
/// fresh `vec![0.0; ..]`.
#[derive(Debug)]
pub(crate) struct G4PrefillScratch {
    /// Ping-pong "main" buffers (hidden-width per token). `main_a` / `main_b`
    /// alternate as the per-layer hidden input and the attention `residual` /
    /// MLP `hidden_out` output.
    pub(crate) main_a: Vec<f32>,
    pub(crate) main_b: Vec<f32>,
    /// hidden-width: attention `input_normed` and MLP `post_normed`.
    pub(crate) normed: Vec<f32>,
    /// q_width: Q projection output (post norm/RoPE).
    pub(crate) q: Vec<f32>,
    /// kv_width: K projection output.
    pub(crate) k: Vec<f32>,
    /// kv_width: V projection output.
    pub(crate) v: Vec<f32>,
    /// q_width: per-position attention context.
    pub(crate) attn_context: Vec<f32>,
    /// hidden-width: o_proj output (`attn_out`) and MLP down-proj output (`mlp_out`).
    pub(crate) proj_out: Vec<f32>,
    /// intermediate-width: MLP gate projection.
    pub(crate) gate: Vec<f32>,
    /// intermediate-width: MLP up projection.
    pub(crate) up: Vec<f32>,
    /// intermediate-width: GeGLU activation output.
    pub(crate) swiglu: Vec<f32>,
}

impl G4LayerKvState {
    pub(crate) fn push(&mut self, key: &[f32], value: &[f32]) -> Result<()> {
        if key.len() != self.kv_width || value.len() != self.kv_width {
            return Err(AegisError::InvalidPlan(format!(
                "g4 kv push shape mismatch: expected {}, got key={} value={}",
                self.kv_width,
                key.len(),
                value.len()
            )));
        }
        self.keys.extend_from_slice(key);
        self.values.extend_from_slice(value);
        self.seq_len += 1;
        Ok(())
    }
}

impl G4CpuExecutor {
    pub fn new_state(&self) -> G4CpuState {
        let cap_tokens = self.kv_context_size.min(256);
        let layers = self
            .layers
            .iter()
            .map(|layer| {
                let kv_width = layer.layer_num_kv_heads * layer.layer_head_dim;
                G4LayerKvState {
                    keys: Vec::with_capacity(cap_tokens * kv_width),
                    values: Vec::with_capacity(cap_tokens * kv_width),
                    seq_len: 0,
                    kv_width,
                }
            })
            .collect();
        let ple_len = self
            .ple
            .as_ref()
            .map(|p| self.layers.len() * p.ple_dim)
            .unwrap_or(0);
        G4CpuState {
            position: 0,
            layers,
            per_layer_inputs: vec![0.0; ple_len],
        }
    }

    /// Allocate the batched-prefill scratch pool ONCE, sized to the max width any
    /// layer needs at `max_batch` tokens. Reused across all layers and chunks of
    /// a single `forward_batched` call so the hot loop never allocates or
    /// zero-inits a per-layer buffer. Widths:
    ///   * main_a/main_b/normed/proj_out — `hidden_size`
    ///   * q/attn_context               — `max(num_attention_heads * head_dim)`
    ///   * k/v                          — `max(num_kv_heads * head_dim)`
    ///   * gate/up/swiglu               — `max_intermediate`
    pub(crate) fn new_prefill_scratch(&self, max_batch: usize) -> G4PrefillScratch {
        let hidden = self.hidden_size;
        let mut q_width = 0usize;
        let mut kv_width = 0usize;
        for layer in &self.layers {
            q_width = q_width.max(self.num_attention_heads * layer.layer_head_dim);
            kv_width = kv_width.max(layer.layer_num_kv_heads * layer.layer_head_dim);
        }
        let inter = self.max_intermediate;
        let z = |w: usize| vec![0.0_f32; max_batch * w];
        G4PrefillScratch {
            main_a: z(hidden),
            main_b: z(hidden),
            normed: z(hidden),
            q: z(q_width),
            k: z(kv_width),
            v: z(kv_width),
            attn_context: z(q_width),
            proj_out: z(hidden),
            gate: z(inter),
            up: z(inter),
            swiglu: z(inter),
        }
    }
}

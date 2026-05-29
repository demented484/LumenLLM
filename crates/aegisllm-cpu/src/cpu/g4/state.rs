//! Gemma-4 CPU executor & runtime-state structs. Mirrors the data the CUDA
//! `CudaLlamaExecutor` / `CudaLayer` / `PleGlobal` / `CudaMoE` carry, but in
//! host f32 / BF16 form (`crates/aegisllm-cuda/src/executor/state.rs`).

use super::linear::CpuLinear;
use super::rope::G4RopeConfig;
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::executor::tensors::Bf16Matrix;

/// Top-level Gemma-4 CPU executor.
#[derive(Debug)]
pub(crate) struct G4CpuExecutor {
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
    /// Largest expert/dense intermediate size, for scratch sizing (reserved
    /// for the persistent-scratch / batched-prefill follow-up).
    #[allow(dead_code)]
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
    pub(crate) fn new_state(&self) -> G4CpuState {
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
}

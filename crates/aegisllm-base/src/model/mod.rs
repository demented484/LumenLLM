//! Model architecture registry and traits.
//!
//! Each supported model family implements [`ModelArchitecture`], which drives
//! graph construction and per-layer dispatch decisions throughout the engine.

mod gemma4;
mod llama;
mod nemotron3;
mod qwen3;

pub use gemma4::Gemma4Architecture;
pub use llama::LlamaArchitecture;
pub use nemotron3::Nemotron3Architecture;
pub use qwen3::Qwen3Architecture;

use crate::artifact::{HfConfig, ModelArtifact};
use crate::error::{AegisError, Result};
use crate::graph::ModelGraph;

// ── Core enums ───────────────────────────────────────────────────────────────

/// Describes the computation kind at each transformer layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerKind {
    /// Standard dense self-attention + MLP (Llama, Qwen 3.5 dense, Gemma 4 full).
    DenseDecoder,
    /// Mixture-of-Experts with top-k routing.
    MoEDecoder {
        num_experts: usize,
        top_k: usize,
        /// Whether a "shared expert" runs in addition to the routed top-k.
        has_shared_expert: bool,
    },
    /// Gated DeltaNet linear attention (Qwen 3.5/3.6 hybrid linear layers).
    LinearAttentionDecoder,
    /// Mamba selective state-space (Nemotron 3).
    MambaDecoder,
    /// Sliding-window attention (Gemma 4 local layers).
    SlidingWindowDecoder {
        window: usize,
    },
    /// Full global attention when interleaved with sliding-window layers.
    GlobalDecoder,
}

/// Describes the attention pattern at a specific layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionPattern {
    /// Standard causal full attention.
    FullCausal,
    /// Attend only to the most recent `size` tokens.
    SlidingWindow { size: usize },
    /// Gated DeltaNet linear recurrent attention.
    LinearGatedDeltaNet,
    /// Selective state-space (Mamba); no cross-token attention at all.
    Mamba,
    /// No attention (pure MLP-only layer, rare).
    None,
}

/// Whether the model uses a second (post) RMSNorm after each sublayer output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormPattern {
    /// Single pre-norm before each sublayer (Llama, Qwen 3.x).
    PreOnly,
    /// Pre-norm before + post-norm after each sublayer (Gemma 4).
    PrePost,
}

/// RoPE configuration extracted from the model config.
#[derive(Debug, Clone, PartialEq)]
pub struct RopeConfig {
    pub theta: f32,
    /// Fraction of head_dim that receives positional encoding.
    /// 1.0 = full RoPE (default). Less = partial / p-RoPE (Gemma 4 global).
    pub partial_factor: f32,
    pub scaling: Option<RopeScalingConfig>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RopeScalingConfig {
    pub rope_type: String,
    pub factor: f32,
    pub low_freq_factor: Option<f32>,
    pub high_freq_factor: Option<f32>,
    pub original_max_position_embeddings: Option<usize>,
}

impl Default for RopeConfig {
    fn default() -> Self {
        Self {
            theta: 10_000.0,
            partial_factor: 1.0,
            scaling: None,
        }
    }
}

// ── Trait ────────────────────────────────────────────────────────────────────

/// Implemented by each supported model family. Controls how the engine builds
/// the model graph and dispatches per-layer computation.
pub trait ModelArchitecture: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &'static str;

    /// Build the full `ModelGraph` from the artifact. Called once at load time.
    fn build_graph(&self, artifact: &ModelArtifact) -> Result<ModelGraph>;

    /// Returns the computation kind for layer `layer_idx`.
    fn layer_kind(&self, layer_idx: usize, config: &HfConfig) -> LayerKind;

    /// Returns the attention pattern for layer `layer_idx`.
    fn attention_pattern(&self, layer_idx: usize, config: &HfConfig) -> AttentionPattern;

    /// Pre-only or pre+post norm style.
    fn norm_pattern(&self) -> NormPattern;

    /// Logit soft-cap applied to `lm_head` output (None = no capping).
    fn lm_head_softcap(&self, config: &HfConfig) -> Option<f32>;

    /// Positional encoding config for this architecture.
    fn rope_config(&self, config: &HfConfig) -> RopeConfig;
}

// ── Detection ────────────────────────────────────────────────────────────────

/// Detect and return the correct `ModelArchitecture` from a `config.json`.
pub fn detect_architecture(config: &HfConfig) -> Result<Box<dyn ModelArchitecture>> {
    // Check the `architectures` list first (most reliable).
    if let Some(archs) = &config.architectures {
        for arch in archs {
            if let Some(a) = try_from_architecture_name(arch) {
                return Ok(a);
            }
        }
    }
    // Fall back to model_type string.
    if let Some(a) = try_from_model_type(&config.model_type) {
        return Ok(a);
    }
    Err(AegisError::Unsupported(format!(
        "unknown architecture — architectures={:?} model_type=`{}`",
        config.architectures, config.model_type
    )))
}

fn try_from_architecture_name(name: &str) -> Option<Box<dyn ModelArchitecture>> {
    match name {
        n if n.contains("Llama") || n.contains("llama") => {
            Some(Box::new(LlamaArchitecture))
        }
        n if n.contains("Mistral") || n.contains("mistral") => {
            // Mistral is structurally Llama — same tensor names, sliding window.
            Some(Box::new(LlamaArchitecture))
        }
        n if n.contains("Qwen3") || n.contains("Qwen2") => {
            Some(Box::new(Qwen3Architecture))
        }
        n if n.contains("Gemma4") || n.contains("Gemma3") => {
            Some(Box::new(Gemma4Architecture))
        }
        n if n.contains("Nemotron") || n.contains("nemotron") => {
            Some(Box::new(Nemotron3Architecture))
        }
        _ => None,
    }
}

fn try_from_model_type(model_type: &str) -> Option<Box<dyn ModelArchitecture>> {
    let lower = model_type.to_ascii_lowercase();
    match lower.as_str() {
        t if t.contains("llama") || t.contains("mistral") => {
            Some(Box::new(LlamaArchitecture))
        }
        t if t.contains("qwen3") || t.contains("qwen2") || t.starts_with("qwen") => {
            Some(Box::new(Qwen3Architecture))
        }
        t if t.contains("gemma4") || t.contains("gemma3") || t.starts_with("gemma") => {
            Some(Box::new(Gemma4Architecture))
        }
        t if t.contains("nemotron") || t.contains("mamba") => {
            Some(Box::new(Nemotron3Architecture))
        }
        _ => None,
    }
}

// CPU reference backend for aegisllm: forward pass + reference attention helpers.

pub mod attention;
pub mod cpu;
pub mod materialization;

pub use attention::{
    ReferenceAttentionDecodeRequest, ReferenceAttentionPrefillRequest,
    reference_attention_decode_f32_into, reference_attention_prefill_f32_into,
};
pub use cpu::{CpuNvfp4Linear, CpuReferenceExecutor, CpuRuntime};
pub use materialization::{
    LinearMaterializationCache, LinearMaterializationKey, LinearMaterializationStats,
};

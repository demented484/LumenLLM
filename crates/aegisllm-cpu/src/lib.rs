// CPU reference backend for aegisllm: forward pass + reference attention helpers.

pub mod attention;
pub mod cpu;
pub mod materialization;
pub mod nvfp4_gemv;
pub mod persistent_pool;

pub use attention::{
    ReferenceAttentionDecodeRequest, ReferenceAttentionPrefillRequest,
    reference_attention_decode_f32_into, reference_attention_prefill_f32_into,
};
pub use cpu::{CpuNvfp4Linear, CpuReferenceExecutor, CpuRuntime, G4CpuExecutor, G4CpuState};
pub use materialization::{
    LinearMaterializationCache, LinearMaterializationKey, LinearMaterializationStats,
};
pub use nvfp4_gemv::{
    CpuMoeExpert, MoeLayerScratch, PackedWeights, gemm_into, gemv_into, moe_layer_experts_into,
};

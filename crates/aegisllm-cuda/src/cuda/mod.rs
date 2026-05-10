mod compile;
mod cutlass_bridge;
mod functions;
mod kernels;
pub(crate) mod loader;
pub(crate) mod owned_pinned;
pub(crate) mod registered_shards;
mod repack;
pub(crate) mod runtime;
pub(crate) mod staging;
mod types;

pub use aegisllm_base::cuda_config::{
    CUDA_PREFILL_CHUNK_MAX, CUDA_PREFILL_DENSE_SPLIT_K_TOKENS, CudaAttentionBackend,
    CudaPrefillAttentionKernel, CudaRuntimeConfig,
};
pub use aegisllm_base::backend_types::AttentionDType;
pub use loader::{CudaWeightLoader, LoadStatusSink};
pub use runtime::CudaRuntime;
/// Maximum sequence length for CUDA Graph-captured decode attention.
/// Must match CUDA_GRAPH_ATTN_MAX_SEQ_LEN in runtime/attention/decode.rs.
pub(crate) const CUDA_GRAPH_ATTN_MAX_SEQ_LEN: usize = 8192;
/// Number of position-chunks the split-K decode attention divides seq_len into.
/// Benchmarks show 16 is optimal for typical contexts (< 3000 tokens) on RTX 5070 Ti.
/// 16 → grid (32, 16)=512 blocks; 3072-byte shared mem fits 32 blocks/SM.
pub(crate) const DECODE_SPLIT_K: usize = 16;
/// Maximum positions per chunk: CUDA_GRAPH_ATTN_MAX_SEQ_LEN / DECODE_SPLIT_K.
pub(crate) const DECODE_MAX_CHUNK_LEN: usize = CUDA_GRAPH_ATTN_MAX_SEQ_LEN / DECODE_SPLIT_K;
pub use types::{
    CudaAttentionRequest, CudaAttentionSplitScratch,
    DensePrefillMetadataProof, DeviceBf16Matrix, DeviceBuffer,
    DeviceNvfp4Linear, DeviceRopeConfig, StandaloneFp8Linear,
};

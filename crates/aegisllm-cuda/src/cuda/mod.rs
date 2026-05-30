mod compile;
mod cutlass_bridge;
mod functions;
pub(crate) mod host_arena;
mod kernels;
pub(crate) mod loader;
pub(crate) mod owned_pinned;
#[cfg(test)]
mod pcie_bench;
pub(crate) mod registered_shards;
mod repack;
pub(crate) mod runtime;
pub(crate) mod staging;
mod types;

pub use aegisllm_base::cuda_config::{
    AttentionComputeQuant, CUDA_PREFILL_CHUNK_MAX, CUDA_PREFILL_DENSE_SPLIT_K_TOKENS,
    CudaAttentionBackend, CudaPrefillAttentionKernel, CudaRuntimeConfig,
};
pub use aegisllm_base::backend_types::AttentionDType;
pub use loader::{CudaWeightLoader, LoadStatusSink};
pub use runtime::CudaRuntime;
pub use runtime::cutlass_moe_smoke::{CutlassMoeSmokeExpert, CutlassMoeSmokeReport};
pub use runtime::fp8_mma_smoke::{Fp8MmaSmokeReport, Fp8MmaStageResult};
/// Maximum sequence length for CUDA Graph-captured decode attention.
/// Must match CUDA_GRAPH_ATTN_MAX_SEQ_LEN in runtime/attention/decode.rs.
pub(crate) const CUDA_GRAPH_ATTN_MAX_SEQ_LEN: usize = 8192;
/// Number of position-chunks the split-K decode attention divides seq_len into.
/// 16 is the baseline (and what graph-capture replay uses); for the eager path
/// at long ctx we grow split_k so per-block chunk_len stays bounded — mirrors
/// vLLM PARTITION_SIZE=512 and TRT-LLM kMinHistoryTokensPerBlock=128 patterns.
/// The kernel's `split_k` arg is runtime-variable; only the partial-buffer
/// allocation needs to size for the worst case.
pub(crate) const DECODE_SPLIT_K: usize = 16;
/// Upper bound on adaptive split_k. At ctx=32768 with target chunk_len=256,
/// split_k=128. Bigger ctx widens chunk_len rather than split_k beyond this,
/// to bound launch / reduction overhead and partial-buffer size.
pub(crate) const DECODE_SPLIT_K_MAX: usize = 128;
/// Target K-positions processed per (chunk × head) block in the eager path.
/// 512 matches vLLM's PARTITION_SIZE: at ctx ≤ 8192 with split_k=16 the chunk
/// is already ≤ 512 so adaptive split doesn't activate (no regression at
/// short/mid ctx). Beyond 8192 split_k grows so chunk_len stays bounded at
/// ~512, bounding per-block work and flattening the long-ctx slope.
pub(crate) const DECODE_TARGET_CHUNK_LEN: usize = 512;
/// Maximum positions per chunk in the graph-capture envelope (baseline
/// split_k=16). Used to size shared-mem in the captured kernel; eager-path
/// long ctx widens dynamically via split_k.
pub(crate) const DECODE_MAX_CHUNK_LEN: usize = CUDA_GRAPH_ATTN_MAX_SEQ_LEN / DECODE_SPLIT_K;
/// Stage G: auto-select the head-dim-partitioned single-pass decode kernel
/// (`..._hdpart`) at/above this context. Below 64k the 2-pass kernel's
/// scores[chunk_len]+vsum shared is small (chunk_len<=head_dim) and its
/// cp.async pipeline wins; at/above 64k split_k caps at 128 so chunk_len grows,
/// the 2-pass shared bloats and caps occupancy at 33%, while hdpart's <1 KiB
/// shared sustains higher occupancy. Measured A/B @256k: hdpart +~9% decode;
/// @16k hdpart is -5% (kept on the 2-pass kernel). Bit-equivalent precision
/// (greedy output identical at 9.5k ctx), so auto-selection changes no quality.
/// Force on at any context with AEGIS_DECODE_HDPART=1.
pub(crate) const DECODE_HDPART_MIN_CONTEXT: usize = 65536;
pub use types::{
    CudaAttentionRequest, CudaAttentionSplitScratch,
    DensePrefillMetadataProof, DeviceBf16Matrix, DeviceBuffer,
    DeviceNvfp4Linear, DeviceRopeConfig, StandaloneFp8Linear,
};

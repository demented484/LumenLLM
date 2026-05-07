pub mod artifact;
pub mod backend;
pub mod cuda_config;
pub mod backend_types;
pub mod error;
pub mod executor;
pub mod generation;
pub mod graph;
pub mod hardware;
pub mod model;
pub mod modalities;
pub mod planning;
pub mod tensor;
pub mod text;

pub use artifact::{HfConfig, ModelArtifact, ModelArtifactSummary};
pub use backend::{BackendDescriptor, BackendKind, BackendRegistry};
pub use cuda_config::{
    CUDA_PREFILL_CHUNK_MAX, CUDA_PREFILL_DENSE_SPLIT_K_TOKENS, CUDA_PREFILL_VARLEN_MIN_CONTEXT,
    CudaPrefillAttentionKernel, CudaRuntimeConfig,
};
pub use backend_types::AttentionDType;
pub use error::{AegisError, Result};
pub use generation::{GenerateOutput, GenerateRequest, SamplingConfig};
pub use graph::{GraphRegion, GraphRegionKind, LayerMetadata, ModelGraph, RegionId, TensorRole};
pub use model::{AttentionPattern, LayerKind, ModelArchitecture, NormPattern, RopeConfig};
pub use hardware::{ComputeDevice, CpuInfo, GpuArchitecture, GpuInfo, HardwareInventory};
pub use tensor::{TensorDType, TensorInfo, TensorRegistry};
pub use text::TextProcessor;

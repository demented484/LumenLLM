// Top-level aegisllm crate: engine, executor orchestration, server, CLI.
// Most of the model code lives in focused workspace crates:
//   aegisllm-base   — shared types (error, tensor, planning, executor traits)
//   aegisllm-cuda   — CUDA backend (runtime + kernels + executor)
//   aegisllm-cpu    — CPU reference backend (forward, materialization)
//   aegisllm-wgpu   — wgpu skeleton backend

pub mod cli;
pub mod engine;
pub mod executor;
pub mod params;
pub mod server;

pub use aegisllm_base::{
    AegisError, BackendKind, BackendRegistry, ComputeDevice, CpuInfo, GenerateOutput,
    GenerateRequest, GpuArchitecture, GpuInfo, GraphRegion, GraphRegionKind, HardwareInventory,
    HfConfig, ModelArtifact, ModelArtifactSummary, ModelGraph, RegionId, Result, SamplingConfig,
    TensorDType, TensorInfo, TensorRegistry, TensorRole, TextProcessor,
};
pub use aegisllm_base::artifact;
pub use aegisllm_base::backend;
pub use aegisllm_base::error;
pub use aegisllm_base::generation;
pub use aegisllm_base::graph;
pub use aegisllm_base::hardware;
pub use aegisllm_base::planning;
pub use aegisllm_base::tensor;
pub use aegisllm_base::text;
pub use aegisllm_base::executor::traits::{
    ExecutorBackendInfo, ExecutorCapability, ExecutorProviderPlan, ExecutorStage,
    GenerationBackendPrimitives, GenerationState, ModelExecutorBackend,
};
pub use aegisllm_base::cuda_config::{CudaPrefillAttentionKernel, CudaRuntimeConfig};
pub use aegisllm_base::cuda_types::CudaAttentionDType;

pub use aegisllm_cpu::{
    CpuNvfp4Linear, CpuReferenceExecutor, CpuRuntime, LinearMaterializationCache,
    LinearMaterializationKey, LinearMaterializationStats,
};
pub use aegisllm_cuda::{CudaExecutorProvider, CudaRuntime, DeviceBf16Matrix, DeviceBuffer, DeviceNvfp4Linear};
pub use aegisllm_wgpu::{
    decode_attention_gpu, embedding_gpu, matmul_f32_gpu, residual_add_gpu, rms_norm_gpu, rope_gpu,
    swiglu_gpu, WgpuContext, WgpuExecutorProvider,
};

pub use engine::{AegisEngine, EngineConfig, EngineReport};
pub use executor::{
    Executor, ExecutorReadiness, HybridExecutorProvider, readiness_for_plan,
};
pub use params::{ParametersFile, ServeConfig};

pub mod layout {
    pub use aegisllm_base::tensor::layout::*;
}

pub mod materialization {
    pub use aegisllm_cpu::materialization::*;
}

pub mod memory {
    pub use aegisllm_base::planning::memory::*;
}

pub mod placement {
    pub use aegisllm_base::planning::placement::*;
}

pub mod quant {
    pub use aegisllm_base::tensor::quant::*;
}

pub mod runtime {
    pub use aegisllm_base::planning::runtime::*;
}

pub mod storage {
    pub use aegisllm_base::tensor::storage::*;
}

pub use aegisllm_base::planning::runtime::cuda_nvfp4_kernel_family_for_layout;

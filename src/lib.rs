pub mod artifact;
pub mod backend;
pub mod cli;
pub mod cpu;
pub mod cuda;
pub mod engine;
pub mod error;
pub mod executor;
pub mod generation;
pub mod graph;
pub mod hardware;
pub mod params;
pub mod planning;
pub mod server;
pub mod tensor;
pub mod text;

pub mod layout {
    pub use crate::tensor::layout::*;
}

pub mod materialization {
    pub use crate::planning::materialization::*;
}

pub mod memory {
    pub use crate::planning::memory::*;
}

pub mod placement {
    pub use crate::planning::placement::*;
}

pub mod quant {
    pub use crate::tensor::quant::*;
}

pub mod runtime {
    pub use crate::planning::runtime::*;
}

pub mod storage {
    pub use crate::tensor::storage::*;
}

pub use artifact::{HfConfig, ModelArtifact, ModelArtifactSummary};
pub use backend::{BackendKind, BackendRegistry};
pub use cpu::{CpuNvfp4Linear, CpuRuntime};
pub use cuda::{CudaRuntime, CudaRuntimeConfig, DeviceBf16Matrix, DeviceBuffer, DeviceNvfp4Linear};
pub use engine::{AegisEngine, EngineConfig, EngineReport};
pub use error::{AegisError, Result};
pub use executor::{
    Executor, ExecutorBackendInfo, ExecutorCapability, ExecutorReadiness, ExecutorStage,
    ModelExecutorBackend, readiness_for_plan,
};
pub use generation::{GenerateOutput, GenerateRequest, SamplingConfig};
pub use graph::{GraphRegion, GraphRegionKind, ModelGraph, TensorRole};
pub use hardware::{ComputeDevice, CpuInfo, GpuArchitecture, GpuInfo, HardwareInventory};
pub use params::{ParametersFile, ServeConfig};
pub use planning::materialization::{
    LinearMaterializationCache, LinearMaterializationKey, LinearMaterializationStats,
    cuda_nvfp4_kernel_family_for_layout,
};
pub use planning::memory::{MemoryBudget, MemoryPlan, MemoryPool, PlannedAllocation};
pub use planning::placement::{
    ComputePlacement, LayerSelector, PlacementPolicy, PlacementRule, ResolvedPlacement,
    StoragePlacement, StorageTier,
};
pub use planning::runtime::{
    KernelCandidate, KernelFamily, KernelPlan, KernelRegistry, RuntimePlan,
};
pub use tensor::layout::{
    LinearLayoutChoice, LinearLayoutPlan, LinearLayoutPolicy, LinearResidentLayout,
    MaterializationPolicy,
};
pub use tensor::quant::{
    KvCacheQuantization, Nvfp4LinearSpec, QuantFormat, QuantFormatDescriptor, TensorCorePrecision,
    WeightQuantization,
};
pub use tensor::storage::{StoragePlan, StorageTotals, TensorResidencyPlan, TensorStoragePlan};
pub use text::TextProcessor;

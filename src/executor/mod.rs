mod attention;
mod cpu;
mod cuda;
mod generation;
mod hybrid;
mod nodes;
mod tensors;
mod traits;

use crate::artifact::ModelArtifact;
use crate::backend::BackendKind;
use crate::cuda::CudaRuntimeConfig;
use crate::error::{AegisError, Result};
use crate::generation::{GenerateOutput, GenerateRequest, TimedGenerateOutput};
use crate::graph::ModelGraph;
use crate::planning::placement::{ComputePlacement, ResolvedPlacement};
use crate::planning::runtime::RuntimePlan;

pub use cpu::CpuReferenceExecutor;
pub use cuda::CudaExecutorProvider;
pub use hybrid::HybridExecutorProvider;
pub use nodes::{
    ActivationResidency, ActivationTransferNode, BackendPrimitiveKind, BackendPrimitiveNode,
    BackendPrimitivePlan, ExecutionNode, ExecutorGraphPlan, KvCacheNode, RegionExecutionNode,
    WeightTransferNode,
};
pub use traits::{
    ExecutorBackendInfo, ExecutorCapability, ExecutorProviderPlan, ExecutorStage,
    GenerationBackendPrimitives, GenerationState, ModelExecutorBackend,
};

#[derive(Debug)]
pub struct Executor {
    backend: Box<dyn ModelExecutorBackend>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutorReadiness {
    pub selected_backend: &'static str,
    pub runnable: bool,
    pub planned_cpu_regions: usize,
    pub planned_cuda_regions: usize,
    pub limitations: Vec<String>,
}

impl Executor {
    pub fn build_native(
        artifact: &ModelArtifact,
        graph: &ModelGraph,
        placement: &ResolvedPlacement,
        runtime: RuntimePlan,
        cuda: CudaRuntimeConfig,
    ) -> Result<Self> {
        let readiness = readiness_for_plan(placement, &runtime);
        if !readiness.runnable {
            return Err(AegisError::Unsupported(format!(
                "executor plan is not runnable yet: {}",
                readiness.limitations.join("; ")
            )));
        }
        match readiness.selected_backend {
            "cpu-reference" => Ok(Self {
                backend: Box::new(CpuReferenceExecutor::from_artifact(
                    artifact, graph, placement, &runtime,
                )?),
            }),
            "cuda" => Ok(Self {
                backend: Box::new(CudaExecutorProvider::from_artifact(
                    artifact, graph, placement, &runtime, cuda,
                )?),
            }),
            "hybrid" => Ok(Self {
                backend: Box::new(HybridExecutorProvider::from_artifact(
                    artifact, graph, placement, &runtime, cuda,
                )?),
            }),
            other => Err(AegisError::Unsupported(format!(
                "{other} executor plan is marked runnable but construction is not wired"
            ))),
        }
    }

    pub fn generate(&self, request: &GenerateRequest) -> Result<GenerateOutput> {
        self.backend.generate(request)
    }

    pub fn generate_timed(&self, request: &GenerateRequest) -> Result<TimedGenerateOutput> {
        self.backend.generate_timed(request)
    }

    pub fn info(&self) -> ExecutorBackendInfo {
        self.backend.info()
    }

    pub fn probe(&self) -> Result<()> {
        self.backend.probe()
    }
}

pub fn readiness_for_plan(
    placement: &ResolvedPlacement,
    runtime: &RuntimePlan,
) -> ExecutorReadiness {
    let planned_cuda_regions = runtime
        .kernels
        .iter()
        .filter(|kernel| matches!(kernel.device, BackendKind::Cuda { .. }))
        .count();
    let planned_cpu_regions = runtime
        .kernels
        .iter()
        .filter(|kernel| kernel.device == BackendKind::Cpu)
        .count();

    let plan = select_provider_plan(
        placement,
        runtime,
        planned_cpu_regions,
        planned_cuda_regions,
    );
    let mut limitations = plan
        .as_ref()
        .map(|plan| plan.limitations.clone())
        .unwrap_or_else(|| vec!["no executor provider matched the resolved placement".into()]);
    limitations.sort();
    limitations.dedup();

    ExecutorReadiness {
        selected_backend: plan.as_ref().map(|plan| plan.info.name).unwrap_or("none"),
        runnable: limitations.is_empty(),
        planned_cpu_regions,
        planned_cuda_regions,
        limitations,
    }
}

fn select_provider_plan(
    placement: &ResolvedPlacement,
    runtime: &RuntimePlan,
    planned_cpu_regions: usize,
    planned_cuda_regions: usize,
) -> Option<ExecutorProviderPlan> {
    if planned_cuda_regions == 0 {
        return Some(CpuReferenceExecutor::plan(placement));
    }

    if planned_cpu_regions > 0 || matches!(placement.kv_cache.compute, ComputePlacement::Cpu) {
        return HybridExecutorProvider::plan(placement, runtime);
    }

    CudaExecutorProvider::plan(placement, runtime)
}

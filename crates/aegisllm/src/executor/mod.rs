// Executor orchestration: top-level facade + hybrid provider + node graph.

mod hybrid;
pub mod nodes;

use aegisllm_base::artifact::ModelArtifact;
use aegisllm_base::backend::BackendKind;
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::generation::{GenerateOutput, GenerateRequest, TimedGenerateOutput};
use aegisllm_base::graph::ModelGraph;
use aegisllm_base::planning::placement::{ComputePlacement, ResolvedPlacement};
use aegisllm_base::planning::runtime::RuntimePlan;
use aegisllm_base::cuda_config::CudaRuntimeConfig;

pub use aegisllm_base::executor::traits::{
    ExecutorBackendInfo, ExecutorCapability, ExecutorProviderPlan, ExecutorStage,
    GenerationBackendPrimitives, GenerationState, ModelExecutorBackend,
};
pub use aegisllm_cpu::CpuReferenceExecutor;
pub use aegisllm_cuda::executor::CudaExecutorProvider;
pub use aegisllm_wgpu::WgpuExecutorProvider;
pub use hybrid::HybridExecutorProvider;
pub use nodes::{
    ActivationResidency, ActivationTransferNode, BackendPrimitiveKind, BackendPrimitiveNode,
    BackendPrimitivePlan, ExecutionNode, ExecutorGraphPlan, KvCacheNode, RegionExecutionNode,
    WeightTransferNode,
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

    pub fn generate_streaming(
        &self,
        request: &GenerateRequest,
        callback: &mut dyn FnMut(usize, &str) -> std::ops::ControlFlow<()>,
    ) -> Result<GenerateOutput> {
        aegisllm_base::executor::generation::generate_streaming_with_backend(
            self.as_primitives(),
            request,
            callback,
        )
    }

    pub fn info(&self) -> ExecutorBackendInfo {
        self.backend.info()
    }

    pub fn probe(&self) -> Result<()> {
        self.backend.probe()
    }

    pub fn as_primitives(&self) -> &dyn GenerationBackendPrimitives {
        self.backend.as_ref()
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
    // Wgpu has highest priority when ANY region or KV cache is placed on
    // wgpu — surface "wgpu skeleton: forward not implemented" instead of
    // silently falling through to a CPU/CUDA provider that can't read
    // wgpu-bound bytes anyway.
    let any_wgpu = matches!(placement.kv_cache.compute, ComputePlacement::Wgpu { .. })
        || placement
            .region_placements
            .iter()
            .any(|r| matches!(r.compute, ComputePlacement::Wgpu { .. }));
    if any_wgpu {
        return Some(WgpuExecutorProvider::plan());
    }

    if planned_cuda_regions == 0 {
        return Some(CpuReferenceExecutor::plan(placement));
    }

    if planned_cpu_regions > 0 || matches!(placement.kv_cache.compute, ComputePlacement::Cpu) {
        return HybridExecutorProvider::plan(placement, runtime);
    }

    CudaExecutorProvider::plan(placement, runtime)
}

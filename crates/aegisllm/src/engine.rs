use std::path::PathBuf;

pub mod bench;
pub mod quality;

mod report;

use aegisllm_base::artifact::ModelArtifact;
use aegisllm_base::backend::BackendRegistry;
use aegisllm_cuda::cuda::CudaRuntimeConfig;
use aegisllm_base::error::{AegisError, Result};
use crate::executor::{Executor, ExecutorBackendInfo};
use aegisllm_base::generation::{GenerateOutput, GenerateRequest, TimedGenerateOutput};
use aegisllm_base::graph::ModelGraph;
use aegisllm_base::hardware::HardwareInventory;
use aegisllm_base::planning::memory::MemoryPlan;
use aegisllm_base::planning::placement::{PlacementPolicy, ResolvedPlacement};
use aegisllm_base::planning::runtime::{KernelFamily, RuntimePlan};
use aegisllm_base::tensor::storage::StoragePlan;

#[derive(Debug, Clone, PartialEq)]
pub struct EngineConfig {
    pub model_path: PathBuf,
    pub policy: PlacementPolicy,
    pub enable_executor: bool,
    pub cuda: CudaRuntimeConfig,
}

#[derive(Debug)]
pub struct AegisEngine {
    pub artifact: ModelArtifact,
    pub graph: ModelGraph,
    pub inventory: HardwareInventory,
    pub backends: BackendRegistry,
    pub placement: ResolvedPlacement,
    pub memory: MemoryPlan,
    pub runtime: RuntimePlan,
    pub storage: StoragePlan,
    pub cuda: CudaRuntimeConfig,
    executor: Option<Executor>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EngineReport {
    pub lines: Vec<String>,
}

impl EngineConfig {
    pub fn auto(model_path: impl Into<PathBuf>) -> Self {
        let inventory = HardwareInventory::detect();
        Self {
            model_path: model_path.into(),
            policy: PlacementPolicy::auto_for(&inventory),
            enable_executor: true,
            cuda: CudaRuntimeConfig::from_env(),
        }
    }
}

impl AegisEngine {
    pub fn build(config: EngineConfig) -> Result<Self> {
        let artifact = ModelArtifact::from_local_path(&config.model_path)?;
        let graph = ModelGraph::from_artifact(&artifact)?;
        let inventory = HardwareInventory::detect();
        let backends = BackendRegistry::from_inventory(&inventory);
        validate_policy_backends(&backends, &config.policy)?;
        let placement = ResolvedPlacement::plan(
            artifact
                .root
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("model"),
            &graph,
            &inventory,
            &config.policy,
        )?;
        let runtime = RuntimePlan::build(&graph, &placement, &backends)?;
        let memory = MemoryPlan::from_placement_runtime_graph_and_cuda(
            &placement,
            Some(&runtime),
            Some(&graph),
            config.cuda,
        );
        validate_memory_budget(&memory)?;
        validate_cuda_runtime_config(config.cuda, &inventory, &runtime)?;
        let storage = StoragePlan::from_graph_and_placement(&graph, &placement);
        let executor = if config.enable_executor {
            Some(Executor::build_native(
                &artifact,
                &graph,
                &placement,
                runtime.clone(),
                config.cuda,
            )?)
        } else {
            None
        };

        Ok(Self {
            artifact,
            graph,
            inventory,
            backends,
            placement,
            memory,
            runtime,
            storage,
            cuda: config.cuda,
            executor,
        })
    }

    pub fn generate(&self, request: GenerateRequest) -> Result<GenerateOutput> {
        let executor = self
            .executor
            .as_ref()
            .ok_or_else(|| AegisError::Unsupported("engine was built without executor".into()))?;
        executor.generate(&request)
    }

    pub fn generate_timed(&self, request: GenerateRequest) -> Result<TimedGenerateOutput> {
        let executor = self
            .executor
            .as_ref()
            .ok_or_else(|| AegisError::Unsupported("engine was built without executor".into()))?;
        executor.generate_timed(&request)
    }

    pub fn executor_info(&self) -> Option<ExecutorBackendInfo> {
        self.executor.as_ref().map(Executor::info)
    }

    pub fn probe_executor(&self) -> Result<()> {
        let executor = self
            .executor
            .as_ref()
            .ok_or_else(|| AegisError::Unsupported("engine was built without executor".into()))?;
        executor.probe()
    }
}

fn validate_memory_budget(memory: &MemoryPlan) -> Result<()> {
    let exceeded = memory
        .warnings
        .iter()
        .find(|warning| warning.contains("allocation exceeds usable budget"));
    if let Some(warning) = exceeded {
        return Err(AegisError::InvalidPlan(format!(
            "memory budget gate failed: {warning}"
        )));
    }
    Ok(())
}

fn validate_cuda_runtime_config(
    cuda: CudaRuntimeConfig,
    inventory: &HardwareInventory,
    runtime: &RuntimePlan,
) -> Result<()> {
    if cuda.native_mxfp4_inference && !cuda.native_mxfp4_repack {
        return Err(AegisError::InvalidConfig(
            "cuda.native-mxfp4-inference=true requires cuda.native-mxfp4-repack=true".into(),
        ));
    }
    if !cuda.native_mxfp4_inference {
        return Ok(());
    }

    let native_regions = runtime
        .kernels
        .iter()
        .filter(|kernel| kernel.family == KernelFamily::CudaNativeFp4TensorCores)
        .count();
    if native_regions == 0 {
        return Err(AegisError::InvalidConfig(
            "cuda.native-mxfp4-inference=true but the runtime plan has no native FP4 CUDA regions"
                .into(),
        ));
    }

    for kernel in runtime
        .kernels
        .iter()
        .filter(|kernel| kernel.family == KernelFamily::CudaNativeFp4TensorCores)
    {
        let crate::backend::BackendKind::Cuda { device } = kernel.device else {
            continue;
        };
        let Some(gpu) = inventory.gpus.iter().find(|gpu| gpu.index == device) else {
            return Err(AegisError::InvalidPlan(format!(
                "native MXFP4 kernel `{}` targets missing cuda:{device}",
                kernel.name
            )));
        };
        if !gpu.supports_fp4() {
            return Err(AegisError::Unsupported(format!(
                "native MXFP4 kernel `{}` requires a Blackwell/FP4 CUDA backend, got cuda:{device} {}",
                kernel.name, gpu.name
            )));
        }
    }
    Ok(())
}

fn validate_policy_backends(backends: &BackendRegistry, policy: &PlacementPolicy) -> Result<()> {
    for placement in [policy.weights_compute, policy.kv_compute] {
        if !backends.contains_compute(placement) {
            return Err(AegisError::InvalidPlan(format!(
                "requested compute backend `{placement}` is not available"
            )));
        }
    }
    Ok(())
}

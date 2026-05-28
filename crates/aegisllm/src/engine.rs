use std::path::PathBuf;

pub mod bench;
pub mod eval_mmlu_pro;
pub mod perplexity;
pub mod quality;
pub mod sample_diversity;

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
    /// EAGLE/MTP speculative-decoding draft model path. `None` = no spec-decode.
    pub draft_model: Option<PathBuf>,
    /// Tokens proposed per spec-decode round (default 4).
    pub num_draft_tokens: usize,
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
    /// EAGLE/MTP draft model path (carried so `with_executor` can attach it
    /// when promoting a preview engine to a full executor for `serve`).
    pub draft_model: Option<PathBuf>,
    pub num_draft_tokens: usize,
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
            draft_model: std::env::var_os("AEGIS_DRAFT_MODEL").map(PathBuf::from),
            num_draft_tokens: std::env::var("AEGIS_NUM_DRAFT_TOKENS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&n| n >= 1)
                .unwrap_or(4),
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
                config.draft_model.as_deref(),
                config.num_draft_tokens,
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
            draft_model: config.draft_model,
            num_draft_tokens: config.num_draft_tokens,
            executor,
        })
    }

    /// Build the heavy native executor (CUDA weight load) on top of an
    /// already-built preview engine, reusing the parsed `artifact`,
    /// `graph`, `placement`, and `runtime` instead of re-running
    /// `ModelArtifact::from_local_path` and replanning. Used by the
    /// `serve` CLI: it builds a preview without an executor to compute
    /// readiness, then promotes that preview here when the plan is
    /// runnable. Avoids a second 17 GiB-shard scan and a duplicated
    /// `parse_lfs_pointer` pass through every safetensors file.
    pub fn with_executor(mut self) -> Result<Self> {
        if self.executor.is_some() {
            return Ok(self);
        }
        let executor = Executor::build_native(
            &self.artifact,
            &self.graph,
            &self.placement,
            self.runtime.clone(),
            self.cuda,
            self.draft_model.as_deref(),
            self.num_draft_tokens,
        )?;
        self.executor = Some(executor);
        Ok(self)
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

    pub fn generate_streaming(
        &self,
        request: &GenerateRequest,
        callback: &mut dyn FnMut(usize, &str) -> std::ops::ControlFlow<()>,
    ) -> Result<GenerateOutput> {
        let executor = self
            .executor
            .as_ref()
            .ok_or_else(|| AegisError::Unsupported("engine was built without executor".into()))?;
        executor.generate_streaming(request, callback)
    }

    pub fn executor_info(&self) -> Option<ExecutorBackendInfo> {
        self.executor.as_ref().map(Executor::info)
    }

    pub fn executor(&self) -> Option<&Executor> {
        self.executor.as_ref()
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
        if std::env::var("AEGIS_BYPASS_BUDGET_GATE").is_ok() {
            eprintln!(
                "[aegis] memory budget gate bypassed: {warning} (AEGIS_BYPASS_BUDGET_GATE=1)"
            );
            return Ok(());
        }
        return Err(AegisError::InvalidPlan(format!(
            "memory budget gate failed: {warning}. \n\
             Options:\n\
               * Free host RAM. If a previous aegisllm run left page-cache \
                 behind, `sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches` \
                 reclaims it.\n\
               * Close other applications competing for RAM.\n\
               * Switch the offending region to `store: mmap` in your config to \
                 trade decode speed for lower RAM footprint (each H2D pays the \
                 CUDA driver's internal pinned-staging copy).\n\
               * Set AEGIS_BYPASS_BUDGET_GATE=1 to skip the check (useful for \
                 wgpu where ram-stored weights upload to GPU and don't stay \
                 resident in host)."
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

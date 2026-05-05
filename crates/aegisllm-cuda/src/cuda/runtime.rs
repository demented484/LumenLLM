use std::sync::{Arc, OnceLock};

use cudarc::cublaslt::CudaBlasLT;
use cudarc::driver::{CudaContext, CudaEvent, CudaStream, sys};

use aegisllm_base::cuda_config::CudaRuntimeConfig;
use super::expert_cache::CacheHandle;
use super::functions::CudaKernelFunctions;
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::hardware::HardwareInventory;

mod attention;
mod blackwell;
mod cublaslt;
mod cutlass;
mod gemm;
mod graph;
mod kv;
mod linear;
mod memory;
mod ops;
mod quant;
mod sampling;
pub(super) mod state_cache;

#[derive(Debug)]
pub struct CudaRuntime {
    device_index: usize,
    compute_capability: Option<String>,
    config: CudaRuntimeConfig,
    pub(super) stream: Arc<CudaStream>,
    /// Dedicated stream for asynchronous PCIe H2D/D2H transfers (KV cache staging).
    /// Allows overlapping next-layer KV upload with current-layer compute.
    /// Synchronization with the compute stream is managed manually via CudaEvent.
    pub(super) transfer_stream: Arc<CudaStream>,
    kernels: CudaKernelFunctions,
    /// cuBLASLt handle for BF16 tensor-core GEMM (attention Q/K/V/O, shared MLP, lm_head).
    /// Bound to the compute stream so launches order with custom kernels.
    pub(super) cublas_lt: CudaBlasLT,
    /// VRAM expert weight cache (Phase 4 of perf overhaul). Set after the
    /// executor finishes loading host-resident NVFP4 weights and knows how
    /// much VRAM is free. Cache hits skip per-call H2D bandwidth.
    expert_cache: OnceLock<CacheHandle>,
}

impl CudaRuntime {
    pub fn new(device_index: usize) -> Result<Self> {
        Self::new_with_config(device_index, CudaRuntimeConfig::from_env())
    }

    pub fn new_with_config(device_index: usize, config: CudaRuntimeConfig) -> Result<Self> {
        let context =
            CudaContext::new(device_index).map_err(map_cuda_err("create cuda context"))?;
        // Disable event-based cross-stream sync tracking. We use a single non-default
        // stream for all operations, so cudarc's automatic event tracking would only
        // insert spurious cross-stream waits that break CUDA Graph capture.
        // Safety: we manage ordering ourselves (single stream, single thread).
        unsafe { context.disable_event_tracking() };
        // Use a non-default stream so that CUDA Graph capture is supported.
        // Stream 0 (default/legacy stream) does not allow begin_capture().
        let stream = context.new_stream().map_err(map_cuda_err("create cuda stream"))?;
        // Separate stream for async PCIe transfers. Manual event-based sync.
        let transfer_stream = context
            .new_stream()
            .map_err(map_cuda_err("create cuda transfer stream"))?;
        let kernels = CudaKernelFunctions::load(&context, device_index)?;
        let compute_capability = HardwareInventory::detect()
            .gpus
            .iter()
            .find(|gpu| gpu.index == device_index)
            .and_then(|gpu| gpu.compute_capability.clone());
        // cuBLASLt is bound to the compute stream so its kernels share the same launch order
        // as the custom kernels we issue. Workspace is auto-sized for SM_120 (32 MiB) by cudarc.
        let cublas_lt = CudaBlasLT::new(stream.clone()).map_err(|e| {
            AegisError::Unsupported(format!("create cuBLASLt handle failed: {e:?}"))
        })?;

        Ok(Self {
            device_index,
            compute_capability,
            config,
            stream,
            transfer_stream,
            kernels,
            cublas_lt,
            expert_cache: OnceLock::new(),
        })
    }

    /// Install the VRAM expert cache. Called once by the executor after all
    /// host-resident NVFP4 weights have been loaded into the pinned arena —
    /// at that point the runtime knows how much VRAM is free and can size
    /// the cache. Subsequent inference dispatch checks `expert_cache()` for
    /// a hit before falling through to the staging path.
    pub(crate) fn install_expert_cache(&self, cache: CacheHandle) -> Result<()> {
        self.expert_cache
            .set(cache)
            .map_err(|_| AegisError::InvalidPlan("expert cache already installed".into()))
    }

    pub(crate) fn expert_cache(&self) -> Option<&CacheHandle> {
        self.expert_cache.get()
    }

    pub fn device_index(&self) -> usize {
        self.device_index
    }

    pub fn config(&self) -> CudaRuntimeConfig {
        self.config
    }

    pub fn compute_capability(&self) -> Option<&str> {
        self.compute_capability.as_deref()
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(map_cuda_err("synchronize cuda stream"))
    }

    pub(crate) fn stream(&self) -> &std::sync::Arc<CudaStream> {
        &self.stream
    }

    /// Dedicated transfer stream — used by `LinearStagingPool::prepare_async`
    /// and the KV-cache async upload path so PCIe traffic can overlap with
    /// compute.
    pub(crate) fn transfer_stream(&self) -> &std::sync::Arc<CudaStream> {
        &self.transfer_stream
    }

    /// Synchronize the transfer stream (block CPU until pending transfers complete).
    /// Used at the end of a decode/prefill step to ensure D2H writebacks are visible
    /// to the host before the next step's H2D reads from the same host buffers.
    pub(crate) fn synchronize_transfer(&self) -> Result<()> {
        self.transfer_stream
            .synchronize()
            .map_err(map_cuda_err("synchronize cuda transfer stream"))
    }

    /// Record an event on the compute stream. Used to signal "kernel finished writing
    /// to staging" so the transfer stream can issue D2H writeback or reuse the slot.
    pub(crate) fn record_compute_event(&self) -> Result<CudaEvent> {
        self.stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING))
            .map_err(map_cuda_err("record compute event"))
    }

    /// Record an event on the transfer stream. Used to signal "H2D upload finished"
    /// so the compute stream can read the staging slot.
    pub(crate) fn record_transfer_event(&self) -> Result<CudaEvent> {
        self.transfer_stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING))
            .map_err(map_cuda_err("record transfer event"))
    }

    /// Make the compute stream wait until `event` has been signaled.
    /// All future kernels on the compute stream will not start until then.
    pub(crate) fn compute_wait_event(&self, event: &CudaEvent) -> Result<()> {
        self.stream
            .wait(event)
            .map_err(map_cuda_err("compute stream wait event"))
    }

    /// Make the transfer stream wait until `event` has been signaled.
    /// All future transfers on the transfer stream will not start until then.
    pub(crate) fn transfer_wait_event(&self, event: &CudaEvent) -> Result<()> {
        self.transfer_stream
            .wait(event)
            .map_err(map_cuda_err("transfer stream wait event"))
    }

    /// Allocate a pre-recorded `CudaEvent` for reuse. Events are expensive to
    /// create/destroy (each is a kernel-mode driver call); call this once at
    /// startup and re-record the event in the hot path via `record_into`.
    pub(crate) fn alloc_event(&self) -> Result<CudaEvent> {
        self.stream
            .context()
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DISABLE_TIMING))
            .map_err(map_cuda_err("alloc cuda event"))
    }

    /// Re-record an existing event with the compute stream's current state.
    /// Cheaper than `record_compute_event` because no new event object is
    /// allocated.
    pub(crate) fn record_into_compute(&self, event: &CudaEvent) -> Result<()> {
        event
            .record(&self.stream)
            .map_err(map_cuda_err("record event on compute stream"))
    }

    /// Re-record an existing event with the transfer stream's current state.
    pub(crate) fn record_into_transfer(&self, event: &CudaEvent) -> Result<()> {
        event
            .record(&self.transfer_stream)
            .map_err(map_cuda_err("record event on transfer stream"))
    }
}

fn ceil_div(value: u32, divisor: u32) -> u32 {
    value.div_ceil(divisor)
}

pub(crate) fn map_cuda_err(
    stage: &'static str,
) -> impl FnOnce(cudarc::driver::DriverError) -> AegisError {
    move |error| AegisError::Unsupported(format!("cuda stage `{stage}` failed: {error:?}"))
}

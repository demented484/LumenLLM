use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream};

use aegisllm_base::cuda_config::CudaRuntimeConfig;
use super::functions::CudaKernelFunctions;
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::hardware::HardwareInventory;

mod attention;
mod blackwell;
mod cutlass;
mod gemm;
mod graph;
mod kv;
mod linear;
mod memory;
mod ops;
mod quant;
mod sampling;

#[derive(Debug)]
pub struct CudaRuntime {
    device_index: usize,
    compute_capability: Option<String>,
    config: CudaRuntimeConfig,
    pub(super) stream: Arc<CudaStream>,
    kernels: CudaKernelFunctions,
}

impl CudaRuntime {
    pub fn new(device_index: usize) -> Result<Self> {
        Self::new_with_config(device_index, CudaRuntimeConfig::from_env())
    }

    pub fn new_with_config(device_index: usize, config: CudaRuntimeConfig) -> Result<Self> {
        let context =
            CudaContext::new(device_index).map_err(map_cuda_err("create cuda context"))?;
        let stream = context.default_stream();
        let kernels = CudaKernelFunctions::load(&context, device_index)?;
        let compute_capability = HardwareInventory::detect()
            .gpus
            .iter()
            .find(|gpu| gpu.index == device_index)
            .and_then(|gpu| gpu.compute_capability.clone());

        Ok(Self {
            device_index,
            compute_capability,
            config,
            stream,
            kernels,
        })
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
}

fn ceil_div(value: u32, divisor: u32) -> u32 {
    value.div_ceil(divisor)
}

pub(super) fn map_cuda_err(
    stage: &'static str,
) -> impl FnOnce(cudarc::driver::DriverError) -> AegisError {
    move |error| AegisError::Unsupported(format!("cuda stage `{stage}` failed: {error:?}"))
}

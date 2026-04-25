use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream};

use super::config::CudaRuntimeConfig;
use super::functions::CudaKernelFunctions;
use crate::error::{AegisError, Result};

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

        Ok(Self {
            device_index,
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

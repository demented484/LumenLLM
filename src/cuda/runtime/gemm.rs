use crate::cuda::{CudaRuntime, DeviceNvfp4Linear};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum CudaGemmDType {
    Bf16,
    Fp16,
    Fp8,
    Nvfp4,
    Mxfp4,
    F32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum CudaPrefillGemmKernel {
    BlackwellMxFp4TensorCores,
    HopperFp8TensorCores,
    Bf16TensorCores,
    Fp16TensorCores,
    Nvfp4ScalarReference,
    F32ScalarReference,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub struct CudaPrefillGemmShape {
    pub m_tokens: usize,
    pub n_output_channels: usize,
    pub k_hidden: usize,
}

#[allow(dead_code)]
impl CudaPrefillGemmShape {
    pub fn flops(self) -> f64 {
        2.0 * self.m_tokens as f64 * self.n_output_channels as f64 * self.k_hidden as f64
    }

    pub fn tflops(self, elapsed_micros: u128) -> f64 {
        if elapsed_micros == 0 {
            return 0.0;
        }
        self.flops() / (elapsed_micros as f64 / 1_000_000.0) / 1.0e12
    }
}

impl CudaRuntime {
    pub fn select_prefill_linear_gemm_kernel(
        &self,
        linear: &DeviceNvfp4Linear,
    ) -> CudaPrefillGemmKernel {
        if self.native_mxfp4_inference_enabled_for(linear) {
            CudaPrefillGemmKernel::BlackwellMxFp4TensorCores
        } else {
            CudaPrefillGemmKernel::Nvfp4ScalarReference
        }
    }
}

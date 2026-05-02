use cudarc::driver::{LaunchConfig, PushKernelArg};

use super::{CudaRuntime, map_cuda_err};
use crate::cuda::{DeviceBuffer, DeviceNvfp4Linear};
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::hardware::{GpuArchitecture, GpuInfo};

impl CudaRuntime {
    pub fn launch_blackwell_nvfp4_linear(
        &self,
        linear: &DeviceNvfp4Linear,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        self.matvec_mxfp4_native_device(linear, input, output)
    }

    pub fn probe_blackwell_nvfp4_linear_abi(&self, linear: &DeviceNvfp4Linear) -> Result<()> {
        let input = self.alloc_f32(linear.cols)?;
        let mut output = self.alloc_f32(linear.rows)?;
        self.launch_blackwell_nvfp4_probe(linear, &input, &mut output)
    }

    fn launch_blackwell_nvfp4_probe(
        &self,
        linear: &DeviceNvfp4Linear,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if input.len() != linear.cols || output.len() != linear.rows {
            return Err(AegisError::InvalidPlan(format!(
                "nvfp4 linear shape mismatch for {}: expected input={} output={}, got input={} output={}",
                linear.name,
                linear.cols,
                linear.rows,
                input.len(),
                output.len()
            )));
        }

        let rows = linear.rows as u32;
        let cols = linear.cols as u32;
        let input_scale = linear.input_scale;
        let output_scale = linear.output_scale;
        let cfg = LaunchConfig {
            grid_dim: (linear.rows as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.blackwell_fp4)
                .arg(&linear.packed)
                .arg(&linear.scales)
                .arg(&input.slice)
                .arg(&rows)
                .arg(&cols)
                .arg(&input_scale)
                .arg(&output_scale)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch blackwell nvfp4 linear"))?;
        self.stream
            .synchronize()
            .map_err(map_cuda_err("synchronize blackwell nvfp4 linear"))?;
        Ok(())
    }

    pub fn supports_native_nvfp4(gpu: &GpuInfo) -> bool {
        matches!(gpu.architecture, GpuArchitecture::Blackwell)
    }
}

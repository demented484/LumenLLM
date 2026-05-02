use cudarc::driver::{LaunchConfig, PushKernelArg};

use super::{CudaRuntime, ceil_div, map_cuda_err};
use crate::cuda::DeviceBuffer;
use aegisllm_base::error::{AegisError, Result};

fn u32_arg(name: &str, value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        AegisError::InvalidPlan(format!(
            "CUDA quant argument {name} exceeds u32 range: {value}"
        ))
    })
}

fn checked_len(label: &str, lhs: usize, rhs: usize) -> Result<usize> {
    lhs.checked_mul(rhs).ok_or_else(|| {
        AegisError::InvalidPlan(format!("CUDA quant {label} length overflow: {lhs} * {rhs}"))
    })
}

impl CudaRuntime {
    pub fn quantize_nvfp4_input_device(
        &self,
        input: &DeviceBuffer<f32>,
        input_scale: f32,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if input.len() != output.len() {
            return Err(AegisError::InvalidPlan(format!(
                "nvfp4 input quantization shape mismatch: input={} output={}",
                input.len(),
                output.len()
            )));
        }
        let len = input.len() as u32;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(len, 16), 1, 1),
            block_dim: (16, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.nvfp4_quantize_input)
                .arg(&input.slice)
                .arg(&len)
                .arg(&input_scale)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch nvfp4 input quantization"))?;
        Ok(())
    }

    pub fn quantize_nvfp4_input_batched_device(
        &self,
        input: &DeviceBuffer<f32>,
        input_scale: f32,
        batch: usize,
        len: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if input.len() < batch * len || output.len() < batch * len {
            return Err(AegisError::InvalidPlan(format!(
                "batched nvfp4 input quant shape mismatch: input={} output={} batch={} len={}",
                input.len(),
                output.len(),
                batch,
                len
            )));
        }
        let batch = batch as u32;
        let len = len as u32;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(len, 16), batch, 1),
            block_dim: (16, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.nvfp4_quantize_input_batched)
                .arg(&input.slice)
                .arg(&batch)
                .arg(&len)
                .arg(&input_scale)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch batched nvfp4 input quantize"))?;
        Ok(())
    }

    pub fn mxfp4_vector_bytes(len: usize) -> Result<usize> {
        if len % 64 != 0 {
            return Err(AegisError::InvalidPlan(format!(
                "native MXFP4 vector quantization requires len divisible by 64, got {len}"
            )));
        }
        Ok((len / 64) * 36)
    }

    pub fn quantize_mxfp4_input_device(
        &self,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<u8>,
    ) -> Result<()> {
        let expected = Self::mxfp4_vector_bytes(input.len())?;
        if output.len() != expected {
            return Err(AegisError::InvalidPlan(format!(
                "native MXFP4 vector buffer mismatch: expected {expected} bytes for input len={}, got {}",
                input.len(),
                output.len()
            )));
        }
        let len = input.len() as u32;
        let cfg = LaunchConfig {
            grid_dim: ((input.len() / 64) as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.mxfp4_quantize_input)
                .arg(&input.slice)
                .arg(&len)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch native mxfp4 input quantization"))?;
        Ok(())
    }

    pub fn quantize_mxfp4_input_batched_device(
        &self,
        input: &DeviceBuffer<f32>,
        batch: usize,
        len: usize,
        output: &mut DeviceBuffer<u8>,
    ) -> Result<()> {
        let values = checked_len("batched native MXFP4 input", batch, len)?;
        if input.len() < values {
            return Err(AegisError::InvalidPlan(format!(
                "batched native MXFP4 input shape mismatch: input={} batch={} len={}",
                input.len(),
                batch,
                len
            )));
        }
        let row_bytes = Self::mxfp4_vector_bytes(len)?;
        let expected = checked_len("batched native MXFP4 output", batch, row_bytes)?;
        if output.len() < expected {
            return Err(AegisError::InvalidPlan(format!(
                "batched native MXFP4 vector buffer mismatch: expected at least {expected} bytes for batch={batch} len={len}, got {}",
                output.len()
            )));
        }
        let batch_u32 = u32_arg("batch", batch)?;
        let len_u32 = u32_arg("len", len)?;
        let cfg = LaunchConfig {
            grid_dim: (len_u32 / 64, batch_u32, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.mxfp4_quantize_input)
                .arg(&input.slice)
                .arg(&len_u32)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err(
            "launch batched native mxfp4 input quantization",
        ))?;
        Ok(())
    }

    pub fn swiglu_mxfp4_quantize_batched_device(
        &self,
        gate: &DeviceBuffer<f32>,
        up: &DeviceBuffer<f32>,
        batch: usize,
        len: usize,
        output: &mut DeviceBuffer<u8>,
    ) -> Result<()> {
        let values = checked_len("batched SwiGLU MXFP4 input", batch, len)?;
        if gate.len() < values || up.len() < values {
            return Err(AegisError::InvalidPlan(format!(
                "batched SwiGLU MXFP4 input shape mismatch: gate={} up={} batch={} len={}",
                gate.len(),
                up.len(),
                batch,
                len
            )));
        }
        let row_bytes = Self::mxfp4_vector_bytes(len)?;
        let expected = checked_len("batched SwiGLU MXFP4 output", batch, row_bytes)?;
        if output.len() < expected {
            return Err(AegisError::InvalidPlan(format!(
                "batched SwiGLU MXFP4 output too small: output={} expected={}",
                output.len(),
                expected
            )));
        }
        let batch_u32 = u32_arg("batch", batch)?;
        let len_u32 = u32_arg("len", len)?;
        let cfg = LaunchConfig {
            grid_dim: (len_u32 / 64, batch_u32, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.swiglu_mxfp4_quantize_batched)
                .arg(&gate.slice)
                .arg(&up.slice)
                .arg(&batch_u32)
                .arg(&len_u32)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch batched SwiGLU MXFP4 quantization"))?;
        Ok(())
    }
}

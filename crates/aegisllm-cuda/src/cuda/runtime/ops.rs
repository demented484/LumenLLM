use cudarc::driver::{LaunchConfig, PushKernelArg};

use super::{CudaRuntime, ceil_div, map_cuda_err};
use crate::cuda::{DeviceBuffer, DeviceRopeConfig};
use aegisllm_base::error::{AegisError, Result};

fn u32_arg(name: &str, value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        AegisError::InvalidPlan(format!(
            "CUDA ops argument {name} exceeds u32 range: {value}"
        ))
    })
}

fn checked_len(label: &str, lhs: usize, rhs: usize) -> Result<usize> {
    lhs.checked_mul(rhs).ok_or_else(|| {
        AegisError::InvalidPlan(format!("CUDA ops {label} length overflow: {lhs} * {rhs}"))
    })
}

fn validate_rope_shape(label: &str, num_heads: usize, head_dim: usize) -> Result<()> {
    if num_heads == 0 || head_dim == 0 || head_dim % 2 != 0 || head_dim > 256 {
        return Err(AegisError::InvalidPlan(format!(
            "{label} requires non-zero heads and even head_dim <= 256: heads={} head_dim={}",
            num_heads, head_dim
        )));
    }
    Ok(())
}

impl CudaRuntime {
    pub fn f32_to_f16_device(
        &self,
        input: &DeviceBuffer<f32>,
        len: usize,
        output: &mut DeviceBuffer<u16>,
    ) -> Result<()> {
        if input.len() < len || output.len() < len {
            return Err(AegisError::InvalidPlan(format!(
                "f32->f16 conversion shape mismatch: input={} output={} len={}",
                input.len(),
                output.len(),
                len
            )));
        }
        let len = len as u32;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(len, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.f32_to_f16)
                .arg(&input.slice)
                .arg(&len)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch f32 to f16"))?;
        Ok(())
    }

    pub fn rms_norm_device(
        &self,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        eps: f32,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if input.len() != weight.len() || input.len() != output.len() {
            return Err(AegisError::InvalidPlan(format!(
                "rms norm shape mismatch: input={} weight={} output={}",
                input.len(),
                weight.len(),
                output.len()
            )));
        }
        let len = input.len() as u32;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.rms_norm)
                .arg(&input.slice)
                .arg(&weight.slice)
                .arg(&len)
                .arg(&eps)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch rms norm"))?;
        Ok(())
    }

    pub fn rms_norm_batched_device(
        &self,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        batch: usize,
        eps: f32,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let len = weight.len();
        if input.len() < batch * len || output.len() < batch * len {
            return Err(AegisError::InvalidPlan(format!(
                "batched rms norm shape mismatch: input={} output={} batch={} len={}",
                input.len(),
                output.len(),
                batch,
                len
            )));
        }
        let batch = batch as u32;
        let len = len as u32;
        let cfg = LaunchConfig {
            grid_dim: (batch, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.rms_norm_batched)
                .arg(&input.slice)
                .arg(&weight.slice)
                .arg(&batch)
                .arg(&len)
                .arg(&eps)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch batched rms norm"))?;
        Ok(())
    }

    pub fn rms_norm_quant_nvfp4_device(
        &self,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        eps: f32,
        input_scale: f32,
        normed_output: &mut DeviceBuffer<f32>,
        quantized_output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if input.len() != weight.len()
            || input.len() != normed_output.len()
            || input.len() != quantized_output.len()
        {
            return Err(AegisError::InvalidPlan(format!(
                "rms norm nvfp4 quant shape mismatch: input={} weight={} normed={} quantized={}",
                input.len(),
                weight.len(),
                normed_output.len(),
                quantized_output.len()
            )));
        }
        let len = input.len() as u32;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.rms_norm_quant_nvfp4)
                .arg(&input.slice)
                .arg(&weight.slice)
                .arg(&len)
                .arg(&eps)
                .arg(&input_scale)
                .arg(&mut normed_output.slice)
                .arg(&mut quantized_output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch rms norm nvfp4 quant"))?;
        Ok(())
    }

    pub fn rms_norm_quant_nvfp4_batched_device(
        &self,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        batch: usize,
        eps: f32,
        input_scale: f32,
        normed_output: &mut DeviceBuffer<f32>,
        quantized_output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let len = weight.len();
        if input.len() < batch * len
            || normed_output.len() < batch * len
            || quantized_output.len() < batch * len
        {
            return Err(AegisError::InvalidPlan(format!(
                "batched rms norm nvfp4 quant shape mismatch: input={} normed={} quantized={} batch={} len={}",
                input.len(),
                normed_output.len(),
                quantized_output.len(),
                batch,
                len
            )));
        }
        let batch = batch as u32;
        let len = len as u32;
        let cfg = LaunchConfig {
            grid_dim: (batch, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.rms_norm_quant_nvfp4_batched)
                .arg(&input.slice)
                .arg(&weight.slice)
                .arg(&batch)
                .arg(&len)
                .arg(&eps)
                .arg(&input_scale)
                .arg(&mut normed_output.slice)
                .arg(&mut quantized_output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch batched rms norm nvfp4 quant"))?;
        Ok(())
    }

    pub fn add_device(
        &self,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        self.add_device_len(a, b, output, a.len())
    }

    pub fn add_device_len(
        &self,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        len: usize,
    ) -> Result<()> {
        if a.len() < len || b.len() < len || output.len() < len {
            return Err(AegisError::InvalidPlan(format!(
                "vector add shape mismatch: a={} b={} output={} len={}",
                a.len(),
                b.len(),
                output.len(),
                len
            )));
        }
        let len = len as u32;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(len, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.add)
                .arg(&a.slice)
                .arg(&b.slice)
                .arg(&len)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch vector add"))?;
        Ok(())
    }

    pub fn add_inplace_device_len(
        &self,
        a: &mut DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        len: usize,
    ) -> Result<()> {
        if a.len() < len || b.len() < len {
            return Err(AegisError::InvalidPlan(format!(
                "in-place vector add shape mismatch: a={} b={} len={}",
                a.len(),
                b.len(),
                len
            )));
        }
        let len = len as u32;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(len, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.add_inplace)
                .arg(&mut a.slice)
                .arg(&b.slice)
                .arg(&len)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch in-place vector add"))?;
        Ok(())
    }

    pub fn swiglu_device(
        &self,
        gate: &DeviceBuffer<f32>,
        up: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        self.swiglu_device_len(gate, up, output, gate.len())
    }

    pub fn swiglu_device_len(
        &self,
        gate: &DeviceBuffer<f32>,
        up: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        len: usize,
    ) -> Result<()> {
        if gate.len() < len || up.len() < len || output.len() < len {
            return Err(AegisError::InvalidPlan(format!(
                "swiglu shape mismatch: gate={} up={} output={} len={}",
                gate.len(),
                up.len(),
                output.len(),
                len
            )));
        }
        let len = len as u32;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(len, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.swiglu)
                .arg(&gate.slice)
                .arg(&up.slice)
                .arg(&len)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch swiglu"))?;
        Ok(())
    }

    pub fn swiglu_inplace_gate_device_len(
        &self,
        gate_and_output: &mut DeviceBuffer<f32>,
        up: &DeviceBuffer<f32>,
        len: usize,
    ) -> Result<()> {
        if gate_and_output.len() < len || up.len() < len {
            return Err(AegisError::InvalidPlan(format!(
                "in-place swiglu shape mismatch: gate={} up={} len={}",
                gate_and_output.len(),
                up.len(),
                len
            )));
        }
        let len = len as u32;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(len, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.swiglu_inplace_gate)
                .arg(&mut gate_and_output.slice)
                .arg(&up.slice)
                .arg(&len)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch in-place gate swiglu"))?;
        Ok(())
    }

    pub fn apply_rope_device(
        &self,
        values: &mut DeviceBuffer<f32>,
        position: usize,
        num_heads: usize,
        head_dim: usize,
        rope: DeviceRopeConfig,
    ) -> Result<()> {
        validate_rope_shape("rope", num_heads, head_dim)?;
        let expected_values = checked_len("rope values", num_heads, head_dim)?;
        if values.len() != expected_values {
            return Err(AegisError::InvalidPlan(format!(
                "rope shape mismatch: values={} expected={}",
                values.len(),
                expected_values
            )));
        }
        let position = u32_arg("position", position)?;
        let num_heads = u32_arg("num_heads", num_heads)?;
        let head_dim = u32_arg("head_dim", head_dim)?;
        let cfg = LaunchConfig {
            grid_dim: (num_heads, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.rope)
                .arg(&mut values.slice)
                .arg(&position)
                .arg(&num_heads)
                .arg(&head_dim)
                .arg(&rope.theta)
                .arg(&rope.factor)
                .arg(&rope.low_freq_factor)
                .arg(&rope.high_freq_factor)
                .arg(&rope.original_max_position_embeddings)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch rope"))?;
        Ok(())
    }

    pub fn apply_rope_batched_device(
        &self,
        values: &mut DeviceBuffer<f32>,
        start_position: usize,
        batch: usize,
        num_heads: usize,
        head_dim: usize,
        rope: DeviceRopeConfig,
    ) -> Result<()> {
        validate_rope_shape("batched rope", num_heads, head_dim)?;
        let expected_values = checked_len("batched rope batch/head", batch, num_heads)
            .and_then(|len| checked_len("batched rope values", len, head_dim))?;
        if values.len() < expected_values {
            return Err(AegisError::InvalidPlan(format!(
                "batched rope shape mismatch: values={} expected={}",
                values.len(),
                expected_values
            )));
        }
        let start_position = u32_arg("start_position", start_position)?;
        let batch = u32_arg("batch", batch)?;
        let num_heads = u32_arg("num_heads", num_heads)?;
        let head_dim = u32_arg("head_dim", head_dim)?;
        let cfg = LaunchConfig {
            grid_dim: (num_heads, batch, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.rope_batched)
                .arg(&mut values.slice)
                .arg(&start_position)
                .arg(&batch)
                .arg(&num_heads)
                .arg(&head_dim)
                .arg(&rope.theta)
                .arg(&rope.factor)
                .arg(&rope.low_freq_factor)
                .arg(&rope.high_freq_factor)
                .arg(&rope.original_max_position_embeddings)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch batched rope"))?;
        Ok(())
    }

    pub fn apply_rope_positions_batched_device(
        &self,
        values: &mut DeviceBuffer<f32>,
        positions: &DeviceBuffer<u32>,
        batch: usize,
        num_heads: usize,
        head_dim: usize,
        rope: DeviceRopeConfig,
    ) -> Result<()> {
        validate_rope_shape("positions batched rope", num_heads, head_dim)?;
        let expected_values = batch
            .checked_mul(num_heads)
            .and_then(|len| len.checked_mul(head_dim))
            .ok_or_else(|| {
                AegisError::InvalidPlan(format!(
                    "positions batched rope length overflow: batch={} heads={} head_dim={}",
                    batch, num_heads, head_dim
                ))
            })?;
        if values.len() < expected_values || positions.len() < batch {
            return Err(AegisError::InvalidPlan(format!(
                "positions batched rope shape mismatch: values={} positions={} expected_values={} batch={}",
                values.len(),
                positions.len(),
                expected_values,
                batch
            )));
        }
        let batch = u32_arg("batch", batch)?;
        let num_heads = u32_arg("num_heads", num_heads)?;
        let head_dim = u32_arg("head_dim", head_dim)?;
        let cfg = LaunchConfig {
            grid_dim: (num_heads, batch, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.rope_positions_batched)
                .arg(&mut values.slice)
                .arg(&positions.slice)
                .arg(&batch)
                .arg(&num_heads)
                .arg(&head_dim)
                .arg(&rope.theta)
                .arg(&rope.factor)
                .arg(&rope.low_freq_factor)
                .arg(&rope.high_freq_factor)
                .arg(&rope.original_max_position_embeddings)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch positions batched rope"))?;
        Ok(())
    }

    pub fn apply_rope_positions_batched_f16_out_device(
        &self,
        values: &mut DeviceBuffer<f32>,
        positions: &DeviceBuffer<u32>,
        batch: usize,
        num_heads: usize,
        head_dim: usize,
        rope: DeviceRopeConfig,
        output: &mut DeviceBuffer<u16>,
    ) -> Result<()> {
        validate_rope_shape("positions batched rope f16 output", num_heads, head_dim)?;
        let expected_values = batch
            .checked_mul(num_heads)
            .and_then(|len| len.checked_mul(head_dim))
            .ok_or_else(|| {
                AegisError::InvalidPlan(format!(
                    "positions batched rope f16 output length overflow: batch={} heads={} head_dim={}",
                    batch, num_heads, head_dim
                ))
            })?;
        if values.len() < expected_values
            || output.len() < expected_values
            || positions.len() < batch
        {
            return Err(AegisError::InvalidPlan(format!(
                "positions batched rope f16 output shape mismatch: values={} output={} positions={} expected_values={} batch={}",
                values.len(),
                output.len(),
                positions.len(),
                expected_values,
                batch
            )));
        }
        let batch = u32_arg("batch", batch)?;
        let num_heads = u32_arg("num_heads", num_heads)?;
        let head_dim = u32_arg("head_dim", head_dim)?;
        let cfg = LaunchConfig {
            grid_dim: (num_heads, batch, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.rope_positions_batched_f16_out)
                .arg(&mut values.slice)
                .arg(&positions.slice)
                .arg(&batch)
                .arg(&num_heads)
                .arg(&head_dim)
                .arg(&rope.theta)
                .arg(&rope.factor)
                .arg(&rope.low_freq_factor)
                .arg(&rope.high_freq_factor)
                .arg(&rope.original_max_position_embeddings)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch positions batched rope f16 output"))?;
        Ok(())
    }

    pub fn copy_row_f32_device(
        &self,
        input: &DeviceBuffer<f32>,
        row: usize,
        cols: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if input.len() < (row + 1) * cols || output.len() != cols {
            return Err(AegisError::InvalidPlan(format!(
                "copy row shape mismatch: input={} row={} cols={} output={}",
                input.len(),
                row,
                cols,
                output.len()
            )));
        }
        let row = row as u32;
        let cols = cols as u32;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(cols, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.copy_row_f32)
                .arg(&input.slice)
                .arg(&row)
                .arg(&cols)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch copy row f32"))?;
        Ok(())
    }
}

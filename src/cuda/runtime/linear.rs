use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use super::{CudaRuntime, ceil_div, map_cuda_err};
use crate::cuda::{DeviceBf16Matrix, DeviceBuffer, DeviceNvfp4Linear};
use crate::error::{AegisError, Result};
use crate::planning::runtime::KernelFamily;

fn u32_arg(name: &str, value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        AegisError::InvalidPlan(format!(
            "CUDA linear argument {name} exceeds u32 range: {value}"
        ))
    })
}

fn checked_len(label: &str, lhs: usize, rhs: usize) -> Result<usize> {
    lhs.checked_mul(rhs).ok_or_else(|| {
        AegisError::InvalidPlan(format!(
            "CUDA linear {label} length overflow: {lhs} * {rhs}"
        ))
    })
}

impl CudaRuntime {
    pub fn matvec_nvfp4_reference_host(
        &self,
        linear: &DeviceNvfp4Linear,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        if input.len() != linear.cols {
            return Err(AegisError::InvalidPlan(format!(
                "nvfp4 linear shape mismatch for {}: expected input={}, got input={}",
                linear.name,
                linear.cols,
                input.len()
            )));
        }
        let input_dev = self.upload_f32(input)?;
        let mut output_dev = self.alloc_f32(linear.rows)?;
        self.matvec_nvfp4_reference_device(linear, &input_dev, &mut output_dev)?;
        self.download_f32(&output_dev)
    }

    pub fn matvec_nvfp4_reference_device(
        &self,
        linear: &DeviceNvfp4Linear,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        self.launch_nvfp4_reference(linear, &input.slice, &mut output.slice)
    }

    pub fn matvec_nvfp4_prequantized_device(
        &self,
        linear: &DeviceNvfp4Linear,
        quantized_input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        self.launch_nvfp4_prequantized(linear, &quantized_input.slice, &mut output.slice)
    }

    pub fn matvec_nvfp4_reference_device_with_scratch(
        &self,
        linear: &DeviceNvfp4Linear,
        input: &DeviceBuffer<f32>,
        quantized_input: &mut DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        self.quantize_nvfp4_input_device(input, linear.input_scale, quantized_input)?;
        self.matvec_nvfp4_prequantized_device(linear, quantized_input, output)
    }

    pub fn matvec_bf16_reference_host(
        &self,
        matrix: &DeviceBf16Matrix,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        if input.len() != matrix.cols {
            return Err(AegisError::InvalidPlan(format!(
                "bf16 matrix shape mismatch for {}: expected input={}, got input={}",
                matrix.name,
                matrix.cols,
                input.len()
            )));
        }
        let input_dev = self.upload_f32(input)?;
        let mut output_dev = self.alloc_f32(matrix.rows)?;
        self.matvec_bf16_reference_device(matrix, &input_dev, &mut output_dev)?;
        self.download_f32(&output_dev)
    }

    pub fn matvec_bf16_reference_device(
        &self,
        matrix: &DeviceBf16Matrix,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        self.launch_bf16_matvec_reference(matrix, &input.slice, &mut output.slice)
    }

    pub fn bf16_row_to_f32_device(
        &self,
        matrix: &DeviceBf16Matrix,
        row: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if row >= matrix.rows || output.len() != matrix.cols {
            return Err(AegisError::InvalidPlan(format!(
                "bf16 row shape mismatch for {}: row={} rows={} output={} cols={}",
                matrix.name,
                row,
                matrix.rows,
                output.len(),
                matrix.cols
            )));
        }
        let row = row as u32;
        let cols = matrix.cols as u32;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(matrix.cols as u32, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.bf16_row)
                .arg(&matrix.values)
                .arg(&row)
                .arg(&cols)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch bf16 row to f32"))?;
        Ok(())
    }

    pub fn bf16_rows_to_f32_device(
        &self,
        matrix: &DeviceBf16Matrix,
        rows: &DeviceBuffer<u32>,
        batch: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let output_len = batch.checked_mul(matrix.cols).ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "bf16 rows output length overflow for {}: batch={} cols={}",
                matrix.name, batch, matrix.cols
            ))
        })?;
        if rows.len() < batch || output.len() < output_len {
            return Err(AegisError::InvalidPlan(format!(
                "bf16 rows shape mismatch for {}: rows={} batch={} output={} expected={}",
                matrix.name,
                rows.len(),
                batch,
                output.len(),
                output_len
            )));
        }
        let batch = u32_arg("batch", batch)?;
        let rows_total = u32_arg("rows", matrix.rows)?;
        let cols = u32_arg("cols", matrix.cols)?;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(cols, 256), batch, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.bf16_rows)
                .arg(&matrix.values)
                .arg(&rows.slice)
                .arg(&batch)
                .arg(&rows_total)
                .arg(&cols)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch bf16 rows to f32"))?;
        Ok(())
    }

    pub fn matvec_nvfp4_reference_batched_device(
        &self,
        linear: &DeviceNvfp4Linear,
        input: &DeviceBuffer<f32>,
        batch: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if input.len() < batch * linear.cols || output.len() < batch * linear.rows {
            return Err(AegisError::InvalidPlan(format!(
                "batched nvfp4 linear shape mismatch for {}: input={} expected={} output={} expected={}",
                linear.name,
                input.len(),
                batch * linear.cols,
                output.len(),
                batch * linear.rows
            )));
        }
        let rows = linear.rows as u32;
        let cols = linear.cols as u32;
        let batch = batch as u32;
        let cfg = LaunchConfig {
            grid_dim: (rows, batch, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.nvfp4_reference_batched)
                .arg(&linear.packed)
                .arg(&linear.scales)
                .arg(&input.slice)
                .arg(&rows)
                .arg(&cols)
                .arg(&linear.input_scale)
                .arg(&linear.output_scale)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch batched nvfp4 matvec reference"))?;
        Ok(())
    }

    pub fn matvec_nvfp4_prequantized_batched_device(
        &self,
        linear: &DeviceNvfp4Linear,
        input: &DeviceBuffer<f32>,
        batch: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if input.len() < batch * linear.cols || output.len() < batch * linear.rows {
            return Err(AegisError::InvalidPlan(format!(
                "batched prequantized nvfp4 linear shape mismatch for {}: input={} expected={} output={} expected={}",
                linear.name,
                input.len(),
                batch * linear.cols,
                output.len(),
                batch * linear.rows
            )));
        }
        let rows = linear.rows as u32;
        let cols = linear.cols as u32;
        let batch = batch as u32;
        let cfg = LaunchConfig {
            grid_dim: (rows, batch, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.nvfp4_prequant_batched)
                .arg(&linear.packed)
                .arg(&linear.scales)
                .arg(&input.slice)
                .arg(&rows)
                .arg(&cols)
                .arg(&linear.output_scale)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch batched nvfp4 matvec prequantized"))?;
        Ok(())
    }

    pub fn matvec_mxfp4_native_device(
        &self,
        linear: &DeviceNvfp4Linear,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let mut input_mxfp4 = self.alloc_u8(Self::mxfp4_vector_bytes(input.len())?)?;
        self.quantize_mxfp4_input_device(input, &mut input_mxfp4)?;
        self.matvec_mxfp4_native_prepacked_device(linear, &input_mxfp4, output)
    }

    pub fn matvec_mxfp4_native_prepacked_device(
        &self,
        linear: &DeviceNvfp4Linear,
        input_mxfp4: &DeviceBuffer<u8>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let expected_input_bytes = Self::mxfp4_vector_bytes(linear.cols)?;
        if input_mxfp4.len() != expected_input_bytes || output.len() != linear.rows {
            return Err(AegisError::InvalidPlan(format!(
                "native mxfp4 matvec shape mismatch for {}: expected input_bytes={} output={}, got input_bytes={} output={}",
                linear.name,
                expected_input_bytes,
                linear.rows,
                input_mxfp4.len(),
                output.len()
            )));
        }
        let Some(native) = linear.native_mxfp4.as_ref() else {
            return Err(AegisError::InvalidPlan(format!(
                "native mxfp4 inference requested for `{}`, but no native MXFP4 resident layout was materialized; enable CudaRuntimeConfig.native_mxfp4_repack",
                linear.name
            )));
        };
        if linear.cols % 64 != 0 {
            return Err(AegisError::InvalidPlan(format!(
                "native mxfp4 matvec for `{}` requires cols divisible by 64, got {}",
                linear.name, linear.cols
            )));
        }
        let rows = linear.rows as u32;
        let cols = linear.cols as u32;
        let blocks_per_row = native.blocks_per_row as u32;
        let output_scale = linear.output_scale;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(rows, 16), 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.mxfp4_matvec)
                .arg(&native.data)
                .arg(&input_mxfp4.slice)
                .arg(&rows)
                .arg(&cols)
                .arg(&blocks_per_row)
                .arg(&output_scale)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch native mxfp4 matvec"))?;
        Ok(())
    }

    pub fn matvec_mxfp4_native_prepacked_batched_device(
        &self,
        linear: &DeviceNvfp4Linear,
        input_mxfp4: &DeviceBuffer<u8>,
        batch: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let row_bytes = Self::mxfp4_vector_bytes(linear.cols)?;
        let expected_input_bytes = checked_len("batched native mxfp4 input", batch, row_bytes)?;
        let expected_output = checked_len("batched native mxfp4 output", batch, linear.rows)?;
        if input_mxfp4.len() < expected_input_bytes || output.len() < expected_output {
            return Err(AegisError::InvalidPlan(format!(
                "batched native mxfp4 matvec shape mismatch for {}: expected input_bytes>={} output>={}, got input_bytes={} output={}",
                linear.name,
                expected_input_bytes,
                expected_output,
                input_mxfp4.len(),
                output.len()
            )));
        }
        let Some(native) = linear.native_mxfp4.as_ref() else {
            return Err(AegisError::InvalidPlan(format!(
                "batched native mxfp4 inference requested for `{}`, but no native MXFP4 resident layout was materialized; enable CudaRuntimeConfig.native_mxfp4_repack",
                linear.name
            )));
        };
        if linear.cols % 64 != 0 {
            return Err(AegisError::InvalidPlan(format!(
                "batched native mxfp4 matvec for `{}` requires cols divisible by 64, got {}",
                linear.name, linear.cols
            )));
        }
        let rows = u32_arg("rows", linear.rows)?;
        let cols = u32_arg("cols", linear.cols)?;
        let blocks_per_row = u32_arg("blocks_per_row", native.blocks_per_row)?;
        let batch_u32 = u32_arg("batch", batch)?;
        let output_scale = linear.output_scale;
        let use_prefill_tile_kernel = batch >= 16;
        let use_n8_kernel = batch > 1 && !use_prefill_tile_kernel;
        if use_prefill_tile_kernel {
            let use_n64_tile = rows >= 64;
            let row_tile = if use_n64_tile { 64 } else { 32 };
            let block_dim = if use_n64_tile { 256 } else { 128 };
            let kernel = if use_n64_tile {
                &self.kernels.mxfp4_matmul_tile_m16n64
            } else {
                &self.kernels.mxfp4_matmul_tile_m16n32
            };
            let cfg = LaunchConfig {
                grid_dim: (ceil_div(rows, row_tile), ceil_div(batch_u32, 16), 1),
                block_dim: (block_dim, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.stream
                    .launch_builder(kernel)
                    .arg(&native.data)
                    .arg(&input_mxfp4.slice)
                    .arg(&rows)
                    .arg(&cols)
                    .arg(&blocks_per_row)
                    .arg(&batch_u32)
                    .arg(&output_scale)
                    .arg(&mut output.slice)
                    .launch(cfg)
            }
            .map_err(map_cuda_err("launch tiled native mxfp4 prefill gemm"))?;
        } else if use_n8_kernel {
            let cfg = LaunchConfig {
                grid_dim: (ceil_div(rows, 16), ceil_div(batch_u32, 8), 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.stream
                    .launch_builder(&self.kernels.mxfp4_matmul_n8)
                    .arg(&native.data)
                    .arg(&input_mxfp4.slice)
                    .arg(&rows)
                    .arg(&cols)
                    .arg(&blocks_per_row)
                    .arg(&batch_u32)
                    .arg(&output_scale)
                    .arg(&mut output.slice)
                    .launch(cfg)
            }
            .map_err(map_cuda_err("launch batched native mxfp4 matmul n8"))?;
        } else {
            let cfg = LaunchConfig {
                grid_dim: (ceil_div(rows, 16), batch_u32, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.stream
                    .launch_builder(&self.kernels.mxfp4_matvec)
                    .arg(&native.data)
                    .arg(&input_mxfp4.slice)
                    .arg(&rows)
                    .arg(&cols)
                    .arg(&blocks_per_row)
                    .arg(&output_scale)
                    .arg(&mut output.slice)
                    .launch(cfg)
            }
            .map_err(map_cuda_err("launch batched native mxfp4 matvec"))?;
        }
        Ok(())
    }

    pub fn matmul_mxfp4_native_prepacked_prefill_device(
        &self,
        linear: &DeviceNvfp4Linear,
        input_mxfp4: &DeviceBuffer<u8>,
        tokens: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        self.matvec_mxfp4_native_prepacked_batched_device(linear, input_mxfp4, tokens, output)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matmul_mxfp4_native_qkv_prefill_device(
        &self,
        q_proj: &DeviceNvfp4Linear,
        k_proj: &DeviceNvfp4Linear,
        v_proj: &DeviceNvfp4Linear,
        input_mxfp4: &DeviceBuffer<u8>,
        batch: usize,
        q_output: &mut DeviceBuffer<f32>,
        k_output: &mut DeviceBuffer<f32>,
        v_output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if batch == 1 {
            self.matmul_mxfp4_native_prepacked_prefill_device(
                q_proj,
                input_mxfp4,
                batch,
                q_output,
            )?;
            self.matmul_mxfp4_native_prepacked_prefill_device(
                k_proj,
                input_mxfp4,
                batch,
                k_output,
            )?;
            return self.matmul_mxfp4_native_prepacked_prefill_device(
                v_proj,
                input_mxfp4,
                batch,
                v_output,
            );
        }
        if q_proj.cols != k_proj.cols || q_proj.cols != v_proj.cols || k_proj.rows != v_proj.rows {
            return Err(AegisError::InvalidPlan(format!(
                "grouped qkv mxfp4 prefill shape mismatch: q={}x{} k={}x{} v={}x{}",
                q_proj.rows, q_proj.cols, k_proj.rows, k_proj.cols, v_proj.rows, v_proj.cols
            )));
        }
        let input_row_bytes = Self::mxfp4_vector_bytes(q_proj.cols)?;
        let expected_input_bytes = checked_len("grouped qkv mxfp4 input", batch, input_row_bytes)?;
        let expected_q_output = checked_len("grouped qkv q output", batch, q_proj.rows)?;
        let expected_k_output = checked_len("grouped qkv k output", batch, k_proj.rows)?;
        let expected_v_output = checked_len("grouped qkv v output", batch, v_proj.rows)?;
        if input_mxfp4.len() < expected_input_bytes
            || q_output.len() < expected_q_output
            || k_output.len() < expected_k_output
            || v_output.len() < expected_v_output
        {
            return Err(AegisError::InvalidPlan(format!(
                "grouped qkv mxfp4 prefill buffers too small: input={} expected_input={} q_out={} k_out={} v_out={} batch={}",
                input_mxfp4.len(),
                expected_input_bytes,
                q_output.len(),
                k_output.len(),
                v_output.len(),
                batch
            )));
        }
        let (Some(q_native), Some(k_native), Some(v_native)) = (
            q_proj.native_mxfp4.as_ref(),
            k_proj.native_mxfp4.as_ref(),
            v_proj.native_mxfp4.as_ref(),
        ) else {
            return Err(AegisError::InvalidPlan(
                "grouped qkv mxfp4 prefill requires native MXFP4 resident layouts".into(),
            ));
        };
        if q_proj.cols % 64 != 0 {
            return Err(AegisError::InvalidPlan(format!(
                "grouped qkv mxfp4 prefill requires cols divisible by 64, got {}",
                q_proj.cols
            )));
        }
        let q_rows = u32_arg("q_rows", q_proj.rows)?;
        let kv_rows = u32_arg("kv_rows", k_proj.rows)?;
        let cols = u32_arg("cols", q_proj.cols)?;
        let q_blocks = u32_arg("q_blocks_per_row", q_native.blocks_per_row)?;
        let k_blocks = u32_arg("k_blocks_per_row", k_native.blocks_per_row)?;
        let v_blocks = u32_arg("v_blocks_per_row", v_native.blocks_per_row)?;
        let batch_u32 = u32_arg("batch", batch)?;
        let max_rows = q_rows.max(kv_rows);
        let use_n64_tile = max_rows >= 64;
        let row_tile = if use_n64_tile { 64 } else { 32 };
        let block_dim = if use_n64_tile { 256 } else { 128 };
        let kernel = if use_n64_tile {
            &self.kernels.mxfp4_matmul_qkv_tile_m16n64
        } else {
            &self.kernels.mxfp4_matmul_qkv_tile_m16n32
        };
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(max_rows, row_tile), ceil_div(batch_u32, 16), 3),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(kernel)
                .arg(&q_native.data)
                .arg(&k_native.data)
                .arg(&v_native.data)
                .arg(&input_mxfp4.slice)
                .arg(&q_rows)
                .arg(&kv_rows)
                .arg(&cols)
                .arg(&q_blocks)
                .arg(&k_blocks)
                .arg(&v_blocks)
                .arg(&batch_u32)
                .arg(&q_proj.output_scale)
                .arg(&k_proj.output_scale)
                .arg(&v_proj.output_scale)
                .arg(&mut q_output.slice)
                .arg(&mut k_output.slice)
                .arg(&mut v_output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch grouped qkv native mxfp4 prefill gemm"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matmul_mxfp4_native_gate_up_prefill_device(
        &self,
        gate_proj: &DeviceNvfp4Linear,
        up_proj: &DeviceNvfp4Linear,
        input_mxfp4: &DeviceBuffer<u8>,
        batch: usize,
        gate_output: &mut DeviceBuffer<f32>,
        up_output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if batch == 1 {
            self.matmul_mxfp4_native_prepacked_prefill_device(
                gate_proj,
                input_mxfp4,
                batch,
                gate_output,
            )?;
            return self.matmul_mxfp4_native_prepacked_prefill_device(
                up_proj,
                input_mxfp4,
                batch,
                up_output,
            );
        }
        if gate_proj.cols != up_proj.cols {
            return Err(AegisError::InvalidPlan(format!(
                "grouped gate/up mxfp4 prefill shape mismatch: gate={}x{} up={}x{}",
                gate_proj.rows, gate_proj.cols, up_proj.rows, up_proj.cols
            )));
        }
        let input_row_bytes = Self::mxfp4_vector_bytes(gate_proj.cols)?;
        let expected_input_bytes =
            checked_len("grouped gate/up mxfp4 input", batch, input_row_bytes)?;
        let expected_gate_output =
            checked_len("grouped gate/up gate output", batch, gate_proj.rows)?;
        let expected_up_output = checked_len("grouped gate/up up output", batch, up_proj.rows)?;
        if input_mxfp4.len() < expected_input_bytes
            || gate_output.len() < expected_gate_output
            || up_output.len() < expected_up_output
        {
            return Err(AegisError::InvalidPlan(format!(
                "grouped gate/up mxfp4 prefill buffers too small: input={} expected_input={} gate_out={} up_out={} batch={}",
                input_mxfp4.len(),
                expected_input_bytes,
                gate_output.len(),
                up_output.len(),
                batch
            )));
        }
        let (Some(gate_native), Some(up_native)) = (
            gate_proj.native_mxfp4.as_ref(),
            up_proj.native_mxfp4.as_ref(),
        ) else {
            return Err(AegisError::InvalidPlan(
                "grouped gate/up mxfp4 prefill requires native MXFP4 resident layouts".into(),
            ));
        };
        if gate_proj.cols % 64 != 0 {
            return Err(AegisError::InvalidPlan(format!(
                "grouped gate/up mxfp4 prefill requires cols divisible by 64, got {}",
                gate_proj.cols
            )));
        }
        let gate_rows = u32_arg("gate_rows", gate_proj.rows)?;
        let up_rows = u32_arg("up_rows", up_proj.rows)?;
        let cols = u32_arg("cols", gate_proj.cols)?;
        let gate_blocks = u32_arg("gate_blocks_per_row", gate_native.blocks_per_row)?;
        let up_blocks = u32_arg("up_blocks_per_row", up_native.blocks_per_row)?;
        let batch_u32 = u32_arg("batch", batch)?;
        let max_rows = gate_rows.max(up_rows);
        let use_n64_tile = max_rows >= 64;
        let row_tile = if use_n64_tile { 64 } else { 32 };
        let block_dim = if use_n64_tile { 256 } else { 128 };
        let kernel = if use_n64_tile {
            &self.kernels.mxfp4_matmul_gate_up_tile_m16n64
        } else {
            &self.kernels.mxfp4_matmul_gate_up_tile_m16n32
        };
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(max_rows, row_tile), ceil_div(batch_u32, 16), 2),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(kernel)
                .arg(&gate_native.data)
                .arg(&up_native.data)
                .arg(&input_mxfp4.slice)
                .arg(&gate_rows)
                .arg(&up_rows)
                .arg(&cols)
                .arg(&gate_blocks)
                .arg(&up_blocks)
                .arg(&batch_u32)
                .arg(&gate_proj.output_scale)
                .arg(&up_proj.output_scale)
                .arg(&mut gate_output.slice)
                .arg(&mut up_output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err(
            "launch grouped gate/up native mxfp4 prefill gemm",
        ))?;
        Ok(())
    }

    pub fn prefill_linear_tflops(
        elapsed: std::time::Duration,
        tokens: usize,
        output_channels: usize,
        hidden: usize,
    ) -> f64 {
        let seconds = elapsed.as_secs_f64();
        if seconds == 0.0 {
            return 0.0;
        }
        let flops = 2.0 * tokens as f64 * output_channels as f64 * hidden as f64;
        flops / seconds / 1.0e12
    }

    fn launch_bf16_matvec_reference(
        &self,
        matrix: &DeviceBf16Matrix,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<()> {
        if input.len() != matrix.cols || output.len() != matrix.rows {
            return Err(AegisError::InvalidPlan(format!(
                "bf16 matvec shape mismatch for {}: expected input={} output={}, got input={} output={}",
                matrix.name,
                matrix.cols,
                matrix.rows,
                input.len(),
                output.len()
            )));
        }
        let rows = matrix.rows as u32;
        let cols = matrix.cols as u32;
        let cfg = LaunchConfig {
            grid_dim: (matrix.rows as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 128 * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.bf16_matvec)
                .arg(&matrix.values)
                .arg(input)
                .arg(&rows)
                .arg(&cols)
                .arg(output)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch bf16 matvec reference"))?;
        Ok(())
    }

    fn launch_nvfp4_reference(
        &self,
        linear: &DeviceNvfp4Linear,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<()> {
        if input.len() != linear.cols || output.len() != linear.rows {
            return Err(AegisError::InvalidPlan(format!(
                "nvfp4 reference shape mismatch for {}: expected input={} output={}, got input={} output={}",
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
            block_dim: (128, 1, 1),
            shared_mem_bytes: 128 * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.nvfp4_reference)
                .arg(&linear.packed)
                .arg(&linear.scales)
                .arg(input)
                .arg(&rows)
                .arg(&cols)
                .arg(&input_scale)
                .arg(&output_scale)
                .arg(output)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch nvfp4 reference linear"))?;
        Ok(())
    }

    fn launch_nvfp4_prequantized(
        &self,
        linear: &DeviceNvfp4Linear,
        quantized_input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<()> {
        if quantized_input.len() != linear.cols || output.len() != linear.rows {
            return Err(AegisError::InvalidPlan(format!(
                "nvfp4 prequantized shape mismatch for {}: expected input={} output={}, got input={} output={}",
                linear.name,
                linear.cols,
                linear.rows,
                quantized_input.len(),
                output.len()
            )));
        }
        let rows = linear.rows as u32;
        let cols = linear.cols as u32;
        let output_scale = linear.output_scale;
        let cfg = LaunchConfig {
            grid_dim: (linear.rows as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 128 * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.nvfp4_prequant)
                .arg(&linear.packed)
                .arg(&linear.scales)
                .arg(quantized_input)
                .arg(&rows)
                .arg(&cols)
                .arg(&output_scale)
                .arg(output)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch nvfp4 prequantized linear"))?;
        Ok(())
    }

    pub fn native_mxfp4_inference_enabled_for(&self, linear: &DeviceNvfp4Linear) -> bool {
        linear.kernel_family == KernelFamily::CudaNativeFp4TensorCores
            && linear.native_mxfp4.is_some()
            && self.config.native_mxfp4_inference
    }
}

impl DeviceNvfp4Linear {
    pub fn native_mxfp4_bytes(&self) -> usize {
        self.native_mxfp4
            .as_ref()
            .map(|native| {
                debug_assert_eq!(native.data.len(), native.bytes);
                native.bytes
            })
            .unwrap_or(0)
    }

    pub fn native_mxfp4_blocks_per_row(&self) -> usize {
        self.native_mxfp4
            .as_ref()
            .map(|native| native.blocks_per_row)
            .unwrap_or(0)
    }

    pub fn cutlass_nvfp4_payload_bytes(&self) -> usize {
        self.cutlass_nvfp4
            .as_ref()
            .map(|resident| resident.payload_e2m1.len())
            .unwrap_or(0)
    }

    pub fn cutlass_nvfp4_scale_bytes(&self) -> usize {
        self.cutlass_nvfp4
            .as_ref()
            .map(|resident| resident.scales_ue4m3.len())
            .unwrap_or(0)
    }

    pub fn cutlass_nvfp4_scale_shape(&self) -> Option<(usize, usize)> {
        self.cutlass_nvfp4
            .as_ref()
            .map(|resident| (resident.layout.scale_rows, resident.layout.scale_cols))
    }
}

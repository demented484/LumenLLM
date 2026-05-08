use cudarc::driver::{CudaSlice, CudaView, LaunchConfig, PushKernelArg};

use super::{CudaRuntime, ceil_div, map_cuda_err};
use crate::cuda::{DeviceBf16Matrix, DeviceBuffer, DeviceNvfp4Linear};
use crate::cuda::staging::LinearStagingPool;
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::planning::runtime::KernelFamily;

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
    #[allow(clippy::too_many_arguments)]
    pub fn split_qkv_scaled_device(
        &self,
        qkv: &DeviceBuffer<f32>,
        batch: usize,
        q_rows: usize,
        kv_rows: usize,
        q_output_scale: f32,
        k_output_scale: f32,
        v_output_scale: f32,
        q_output: &mut DeviceBuffer<f32>,
        k_output: &mut DeviceBuffer<f32>,
        v_output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let qkv_rows = q_rows
            .checked_add(kv_rows)
            .and_then(|rows| rows.checked_add(kv_rows))
            .ok_or_else(|| AegisError::InvalidPlan("split qkv rows overflow".into()))?;
        let expected_qkv = checked_len("split qkv input", batch, qkv_rows)?;
        let expected_q = checked_len("split q output", batch, q_rows)?;
        let expected_kv = checked_len("split kv output", batch, kv_rows)?;
        if qkv.len() < expected_qkv
            || q_output.len() < expected_q
            || k_output.len() < expected_kv
            || v_output.len() < expected_kv
        {
            return Err(AegisError::InvalidPlan(format!(
                "split qkv buffers too small: qkv={} expected_qkv={} q={} expected_q={} k={} v={} expected_kv={}",
                qkv.len(),
                expected_qkv,
                q_output.len(),
                expected_q,
                k_output.len(),
                v_output.len(),
                expected_kv
            )));
        }
        let total = expected_qkv;
        let block = 256u32;
        let grid = ceil_div(u32_arg("split qkv elements", total)?, block).clamp(1, 65535);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let batch = u32_arg("batch", batch)?;
        let q_rows = u32_arg("q_rows", q_rows)?;
        let kv_rows = u32_arg("kv_rows", kv_rows)?;
        unsafe {
            self.stream
                .launch_builder(&self.kernels.split_qkv_scaled)
                .arg(&qkv.slice)
                .arg(&batch)
                .arg(&q_rows)
                .arg(&kv_rows)
                .arg(&q_output_scale)
                .arg(&k_output_scale)
                .arg(&v_output_scale)
                .arg(&mut q_output.slice)
                .arg(&mut k_output.slice)
                .arg(&mut v_output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch split qkv scaled"))?;
        Ok(())
    }

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
        if input.len() < matrix.cols {
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
        if matrix.is_host_resident() {
            return self.matvec_bf16_host_resident_device(matrix, input, output);
        }
        self.launch_bf16_matvec_reference(matrix, &input.slice, &mut output.slice)
    }

    /// Quantize a VRAM-resident BF16 matrix to FP8 E4M3 with per-row FP32 scales.
    /// Used at load time by `load_bf16_as_fp8_linear`. The output buffers must be
    /// pre-allocated to `rows*cols` bytes (fp8) and `rows` floats (scales).
    pub fn quantize_bf16_to_fp8_per_row_device(
        &self,
        bf16: &DeviceBuffer<u16>,
        rows: usize,
        cols: usize,
        fp8_out: &mut DeviceBuffer<u8>,
        row_scales_out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if bf16.len() < rows * cols {
            return Err(AegisError::InvalidPlan(format!(
                "fp8 quantize: bf16 buffer too small: have {} need {}*{}={}",
                bf16.len(), rows, cols, rows * cols
            )));
        }
        if fp8_out.len() < rows * cols {
            return Err(AegisError::InvalidPlan(format!(
                "fp8 quantize: output buffer too small: have {} need {}",
                fp8_out.len(), rows * cols
            )));
        }
        if row_scales_out.len() < rows {
            return Err(AegisError::InvalidPlan(format!(
                "fp8 quantize: scales buffer too small: have {} need {}",
                row_scales_out.len(), rows
            )));
        }
        let rows_u32 = u32_arg("rows", rows)?;
        let cols_u32 = u32_arg("cols", cols)?;
        let block_dim = 256u32;
        let cfg = LaunchConfig {
            grid_dim: (rows_u32, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: block_dim * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.fp8_quantize_bf16_per_row)
                .arg(&bf16.slice)
                .arg(&mut fp8_out.slice)
                .arg(&mut row_scales_out.slice)
                .arg(&rows_u32)
                .arg(&cols_u32)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch fp8 quantize bf16 per row"))?;
        Ok(())
    }

    /// Single-token matvec against a standalone FP8 weight (decode path).
    pub fn matvec_fp8_standalone_device(
        &self,
        linear: &super::super::StandaloneFp8Linear,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if input.len() < linear.cols || output.len() < linear.rows {
            return Err(AegisError::InvalidPlan(format!(
                "fp8 matvec shape mismatch for {}: input={} need {}, output={} need {}",
                linear.name, input.len(), linear.cols, output.len(), linear.rows
            )));
        }
        let rows = u32_arg("rows", linear.rows)?;
        let cols = u32_arg("cols", linear.cols)?;
        let block_dim = 256u32;
        let cfg = LaunchConfig {
            grid_dim: (rows, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: block_dim * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.fp8_matvec)
                .arg(linear.data_slice())
                .arg(linear.row_scales_slice())
                .arg(&input.slice)
                .arg(&rows)
                .arg(&cols)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch fp8 matvec"))?;
        Ok(())
    }

    /// Dequantize an FP8 weight matrix into a BF16 scratch buffer in-place.
    /// Used by the cuBLASLt-backed FP8 prefill path: dequant once per call,
    /// then run BF16×BF16 GEMM via tensor cores.
    pub fn dequant_fp8_to_bf16_device(
        &self,
        linear: &super::super::StandaloneFp8Linear,
        bf16_out: &mut DeviceBuffer<u16>,
    ) -> Result<()> {
        let total = linear.rows * linear.cols;
        if bf16_out.len() < total {
            return Err(AegisError::InvalidPlan(format!(
                "fp8→bf16 dequant: scratch too small for `{}`: have {} need {}",
                linear.name, bf16_out.len(), total
            )));
        }
        let rows = u32_arg("rows", linear.rows)?;
        let cols = u32_arg("cols", linear.cols)?;
        let block_dim = 256u32;
        let cfg = LaunchConfig {
            grid_dim: (rows, ceil_div(cols, block_dim), 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.fp8_dequant_to_bf16)
                .arg(linear.data_slice())
                .arg(linear.row_scales_slice())
                .arg(&mut bf16_out.slice)
                .arg(&rows)
                .arg(&cols)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch fp8 dequant to bf16"))?;
        Ok(())
    }

    /// Batched matmul against a standalone FP8 weight (prefill path).
    pub fn matmul_fp8_standalone_batched_device(
        &self,
        linear: &super::super::StandaloneFp8Linear,
        input: &DeviceBuffer<f32>,
        batch: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let total_in = checked_len("fp8 matmul input", batch, linear.cols)?;
        let total_out = checked_len("fp8 matmul output", batch, linear.rows)?;
        if input.len() < total_in || output.len() < total_out {
            return Err(AegisError::InvalidPlan(format!(
                "fp8 matmul shape mismatch for {}: input={} need {}, output={} need {}",
                linear.name, input.len(), total_in, output.len(), total_out
            )));
        }
        let rows = u32_arg("rows", linear.rows)?;
        let cols = u32_arg("cols", linear.cols)?;
        let batch_u32 = u32_arg("batch", batch)?;
        let block_dim = 128u32;
        let cfg = LaunchConfig {
            grid_dim: (rows, batch_u32, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: block_dim * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.fp8_matmul_batched)
                .arg(linear.data_slice())
                .arg(linear.row_scales_slice())
                .arg(&input.slice)
                .arg(&rows)
                .arg(&cols)
                .arg(&batch_u32)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch fp8 matmul batched"))?;
        Ok(())
    }

    /// Batched BF16 GEMM-like matmul over `batch` token rows. Requires the matrix
    /// to be VRAM-resident (host-resident BF16 hot-path is the slow CPU rayon
    /// fallback and is intentionally not supported here — chunked prefill always
    /// runs on VRAM-resident weights).
    pub fn matmul_bf16_reference_batched_device(
        &self,
        matrix: &DeviceBf16Matrix,
        input: &DeviceBuffer<f32>,
        batch: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if matrix.is_host_resident() {
            return Err(AegisError::InvalidPlan(format!(
                "batched bf16 matmul does not support host-resident matrix `{}`; load to VRAM",
                matrix.name
            )));
        }
        let total_in = checked_len("bf16 matmul input", batch, matrix.cols)?;
        let total_out = checked_len("bf16 matmul output", batch, matrix.rows)?;
        if input.len() < total_in || output.len() < total_out {
            return Err(AegisError::InvalidPlan(format!(
                "batched bf16 matmul shape mismatch for {}: input.len()={} need {}*{}={}, output.len()={} need {}*{}={}",
                matrix.name, input.len(), batch, matrix.cols, total_in,
                output.len(), batch, matrix.rows, total_out
            )));
        }
        let rows = u32_arg("rows", matrix.rows)?;
        let cols = u32_arg("cols", matrix.cols)?;
        let batch_u32 = u32_arg("batch", batch)?;
        let block_dim = 128u32;
        let cfg = LaunchConfig {
            grid_dim: (rows, batch_u32, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: block_dim * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.bf16_matmul_reference_batched)
                .arg(&matrix.values)
                .arg(&input.slice)
                .arg(&rows)
                .arg(&cols)
                .arg(&batch_u32)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch bf16_matmul_reference_batched"))?;
        Ok(())
    }

    /// CPU-side matvec for host-resident BF16 matrices: avoid having lm_head (~1 GB)
    /// permanently in VRAM at the cost of one D2H download (input) + ~30ms CPU compute
    /// + one H2D upload (logits) per decode step. Saves ~1 GB VRAM.
    fn matvec_bf16_host_resident_device(
        &self,
        matrix: &DeviceBf16Matrix,
        input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let host = matrix.host_values.as_ref().ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "host-resident matvec called on non-host-resident `{}`",
                matrix.name
            ))
        })?;
        if input.len() < matrix.cols || output.len() < matrix.rows {
            return Err(AegisError::InvalidPlan(format!(
                "bf16 host matvec shape mismatch for {}: input={} cols={} output={} rows={}",
                matrix.name, input.len(), matrix.cols, output.len(), matrix.rows
            )));
        }
        use rayon::prelude::*;
        let input_host = self.download_f32(input)?;
        let weights = host
            .values
            .as_slice()
            .map_err(map_cuda_err("read pinned bf16 weights"))?;
        let cols = matrix.cols;
        let rows = matrix.rows;
        let mut result = vec![0.0_f32; rows];
        result
            .par_iter_mut()
            .enumerate()
            .for_each(|(row, slot)| {
                let row_base = row * cols;
                let mut acc = 0.0_f32;
                for c in 0..cols {
                    let bf16_bits = weights[row_base + c];
                    let f = f32::from_bits((bf16_bits as u32) << 16);
                    acc += f * input_host[c];
                }
                *slot = acc;
            });
        let mut out_dev = output.slice.slice_mut(0..rows);
        self.stream
            .memcpy_htod(&result, &mut out_dev)
            .map_err(map_cuda_err("upload host bf16 matvec result"))?;
        Ok(())
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
        if let Some(host) = matrix.host_values.as_ref() {
            // Host-resident: extract just the requested row from pinned RAM, convert
            // BF16→f32 on host (cols × 4 bytes ≈ 16 KB), upload to GPU. Tiny copy.
            let weights = host
                .values
                .as_slice()
                .map_err(map_cuda_err("read pinned bf16 row"))?;
            let row_base = row * matrix.cols;
            let row_f32: Vec<f32> = weights[row_base..row_base + matrix.cols]
                .iter()
                .map(|&bits| f32::from_bits((bits as u32) << 16))
                .collect();
            let mut dst = output.slice.slice_mut(0..matrix.cols);
            self.stream
                .memcpy_htod(&row_f32, &mut dst)
                .map_err(map_cuda_err("upload host bf16 row"))?;
            return Ok(());
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
        if let Some(host) = matrix.host_values.as_ref() {
            // Host-resident: download row indices, gather requested rows on host
            // (batch × cols × 4 bytes — small for prefill chunk), upload as f32.
            let row_indices: Vec<u32> = self
                .stream
                .memcpy_dtov(&rows.slice.slice(0..batch))
                .map_err(map_cuda_err("download bf16 row indices"))?;
            let weights = host
                .values
                .as_slice()
                .map_err(map_cuda_err("read pinned bf16 rows"))?;
            let mut gathered = Vec::with_capacity(output_len);
            for &idx in &row_indices {
                let idx = idx as usize;
                if idx >= matrix.rows {
                    return Err(AegisError::InvalidPlan(format!(
                        "bf16 row index out of bounds: idx={} rows={}",
                        idx, matrix.rows
                    )));
                }
                let base = idx * matrix.cols;
                gathered.extend(
                    weights[base..base + matrix.cols]
                        .iter()
                        .map(|&bits| f32::from_bits((bits as u32) << 16)),
                );
            }
            let mut dst = output.slice.slice_mut(0..output_len);
            self.stream
                .memcpy_htod(&gathered, &mut dst)
                .map_err(map_cuda_err("upload host bf16 rows"))?;
            return Ok(());
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

    /// Grouped MoE NVFP4 GEMM (Phase B.2): single launch processes all
    /// active experts of one projection. Replaces the per-expert dispatch
    /// loop's inner kernel call.
    ///
    /// Inputs are concatenated VRAM buffers built by the caller:
    ///   * `packed_base` / `scales_base` — contiguous bytes of all active
    ///     experts' weights for this projection (gate / up / down).
    ///   * `*_offsets` / `output_scales` / `expert_token_offsets` — per-active-
    ///     expert metadata uploaded once per projection.
    ///   * `permuted_input` — `[total_tokens, cols]` activations sorted by
    ///     expert (built by `permute_gather_f32_device`).
    ///   * `permuted_output` — written; later un-permuted by
    ///     `unpermute_scatter_add_f32_device`.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_nvfp4_grouped_prequant_wmma_bf16_device(
        &self,
        packed_base: &DeviceBuffer<u8>,
        scales_base: &DeviceBuffer<u8>,
        packed_offsets: &DeviceBuffer<u32>,
        scales_offsets: &DeviceBuffer<u32>,
        output_scales: &DeviceBuffer<f32>,
        expert_token_offsets: &DeviceBuffer<u32>,
        permuted_input: &DeviceBuffer<f32>,
        rows: usize,
        cols: usize,
        max_tokens_per_expert: usize,
        num_active_experts: usize,
        permuted_output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if num_active_experts == 0 || max_tokens_per_expert == 0 {
            return Ok(());
        }
        if rows % 16 != 0 || cols % 16 != 0 {
            return Err(AegisError::InvalidPlan(format!(
                "grouped wmma gemm requires rows%16==0 and cols%16==0, got rows={} cols={}",
                rows, cols,
            )));
        }
        let rows_u32 = u32_arg("rows", rows)?;
        let cols_u32 = u32_arg("cols", cols)?;
        let grid_z = u32_arg("num_active_experts", num_active_experts)?;

        // Prefer the 32x32-output-tile (4-warp) kernel: same FP4 dequant
        // semantics, identical per-element WMMA accumulation order, but
        // amortizes shared-memory loads across 4 mma.sync calls per K-iter
        // and raises tensor-core utilization. Eligibility: rows is a
        // multiple of 32 (ensures each block's M-tile is fully populated for
        // the 16x16+16x16 sub-tile pair). The 16x16 fallback kernel handles
        // odd rows for forward-compatibility with future dim choices, and
        // also covers the small-batch decode path that calls this.
        let t32_disabled = std::env::var("AEGIS_NVFP4_GROUPED_T32_DISABLE").is_ok();
        let use_t32 = !t32_disabled
            && rows % 32 == 0
            && max_tokens_per_expert >= 16;
        if use_t32 {
            let grid_x = (rows / 32) as u32;
            let grid_y = ((max_tokens_per_expert + 31) / 32) as u32;
            let cfg = LaunchConfig {
                grid_dim: (grid_x, grid_y, grid_z),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.stream
                    .launch_builder(&self.kernels.nvfp4_grouped_prequant_gemm_wmma_bf16_t32)
                    .arg(&packed_base.slice)
                    .arg(&scales_base.slice)
                    .arg(&packed_offsets.slice)
                    .arg(&scales_offsets.slice)
                    .arg(&output_scales.slice)
                    .arg(&expert_token_offsets.slice)
                    .arg(&permuted_input.slice)
                    .arg(&rows_u32)
                    .arg(&cols_u32)
                    .arg(&mut permuted_output.slice)
                    .launch(cfg)
            }
            .map_err(map_cuda_err("launch grouped nvfp4 wmma gemm t32"))?;
            return Ok(());
        }

        let grid_x = (rows / 16) as u32;
        let grid_y = ((max_tokens_per_expert + 15) / 16) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, grid_z),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.nvfp4_grouped_prequant_gemm_wmma_bf16)
                .arg(&packed_base.slice)
                .arg(&scales_base.slice)
                .arg(&packed_offsets.slice)
                .arg(&scales_offsets.slice)
                .arg(&output_scales.slice)
                .arg(&expert_token_offsets.slice)
                .arg(&permuted_input.slice)
                .arg(&rows_u32)
                .arg(&cols_u32)
                .arg(&mut permuted_output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch grouped nvfp4 wmma gemm"))?;
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
        let rows_u32 = linear.rows as u32;
        let cols_u32 = linear.cols as u32;
        let batch_u32 = batch as u32;
        if batch > 1 {
            // Prefer the BF16 WMMA tensor-core path when shapes align to 16
            // AND batch is large enough to keep SM grid saturated (~32 N-tiles
            // worth). For per-expert MoE dispatch with batch~8, the scalar
            // 8-warp kernel actually wins because its 256-thread blocks fill
            // SMs better than WMMA's 32-thread blocks. WMMA pays off once
            // grouped MoE delivers larger per-call batches.
            let wmma_disabled = std::env::var("AEGIS_NVFP4_WMMA_DISABLE").is_ok();
            let wmma_eligible = !wmma_disabled
                && linear.rows % 16 == 0
                && linear.cols % 16 == 0
                && batch >= 32;
            if wmma_eligible {
                let grid_x = (linear.rows / 16) as u32;
                let grid_y = ((batch + 15) / 16) as u32;
                let cfg = LaunchConfig {
                    grid_dim: (grid_x, grid_y, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,  // sh_a/sh_b/sh_c are static __shared__
                };
                unsafe {
                    self.stream
                        .launch_builder(&self.kernels.nvfp4_prequant_batched_gemm_wmma_bf16)
                        .arg(&linear.packed)
                        .arg(&linear.scales)
                        .arg(&input.slice)
                        .arg(&rows_u32)
                        .arg(&cols_u32)
                        .arg(&batch_u32)
                        .arg(&linear.output_scale)
                        .arg(&mut output.slice)
                        .launch(cfg)
                }
                .map_err(map_cuda_err("launch batched nvfp4 gemm wmma bf16"))?;
                return Ok(());
            }
            let grid_y = ((batch + 7) / 8) as u32;
            let shared = (linear.cols / 2 + linear.cols / 16) as u32;
            let cfg = LaunchConfig {
                grid_dim: (rows_u32, grid_y, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: shared,
            };
            unsafe {
                self.stream
                    .launch_builder(&self.kernels.nvfp4_prequant_batched_gemm)
                    .arg(&linear.packed)
                    .arg(&linear.scales)
                    .arg(&input.slice)
                    .arg(&rows_u32)
                    .arg(&cols_u32)
                    .arg(&batch_u32)
                    .arg(&linear.output_scale)
                    .arg(&mut output.slice)
                    .launch(cfg)
            }
            .map_err(map_cuda_err("launch batched nvfp4 gemm prequantized"))?;
        } else {
            let cfg = LaunchConfig {
                grid_dim: (rows_u32, batch_u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 256 * std::mem::size_of::<f32>() as u32,
            };
            unsafe {
                self.stream
                    .launch_builder(&self.kernels.nvfp4_prequant_batched)
                    .arg(&linear.packed)
                    .arg(&linear.scales)
                    .arg(&input.slice)
                    .arg(&rows_u32)
                    .arg(&cols_u32)
                    .arg(&linear.output_scale)
                    .arg(&mut output.slice)
                    .launch(cfg)
            }
            .map_err(map_cuda_err("launch batched nvfp4 matvec prequantized"))?;
        }
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
        if !linear.cols.is_multiple_of(64) {
            return Err(AegisError::InvalidPlan(format!(
                "native mxfp4 matvec for `{}` requires cols divisible by 64, got {}",
                linear.name, linear.cols
            )));
        }
        let rows = linear.rows as u32;
        let cols = linear.cols as u32;
        let blocks_per_row = native.blocks_per_row as u32;
        let output_scale = linear.output_scale;
        // Choose kernel by divisibility: 16-warp → 4-warp → 1-warp, highest occupancy first.
        // 16-warp (512 threads): ~92-100% SM occupancy for all LLaMA projection sizes.
        // 4-warp (128 threads): ~80% for gate/up, ~23% for small matrices.
        // 1-warp (32 threads): fallback for odd col counts.
        let (block_dim, kernel, tag) = if linear.cols.is_multiple_of(64 * 16) {
            (512u32, &self.kernels.mxfp4_matvec_16warp, "16warp")
        } else if linear.cols.is_multiple_of(64 * 4) {
            (128u32, &self.kernels.mxfp4_matvec_4warp, "4warp")
        } else {
            (32u32, &self.kernels.mxfp4_matvec, "1warp")
        };
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(rows, 16), 1, 1),
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
                .arg(&output_scale)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err(match tag {
            "16warp" => "launch native mxfp4 matvec 16warp",
            "4warp"  => "launch native mxfp4 matvec 4warp",
            _        => "launch native mxfp4 matvec 1warp",
        }))?;
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
        if !linear.cols.is_multiple_of(64) {
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
        if !q_proj.cols.is_multiple_of(64) {
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
        if !gate_proj.cols.is_multiple_of(64) {
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
        if input.len() < matrix.cols || output.len() < matrix.rows {
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

    // -----------------------------------------------------------------------
    // Staging-path launchers: accept CudaView<u8> for packed/scales so callers
    // can pass views into a shared staging VRAM buffer instead of owned slices.
    // -----------------------------------------------------------------------

    /// Reference NVFP4 matvec where packed/scales come from a staging view.
    pub(crate) fn launch_nvfp4_reference_views(
        &self,
        packed: &CudaView<u8>,
        scales: &CudaView<u8>,
        rows: usize,
        cols: usize,
        input_scale: f32,
        output_scale: f32,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 128 * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.nvfp4_reference)
                .arg(packed)
                .arg(scales)
                .arg(input)
                .arg(&rows_u32)
                .arg(&cols_u32)
                .arg(&input_scale)
                .arg(&output_scale)
                .arg(output)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch staged nvfp4 reference"))?;
        Ok(())
    }

    /// Pre-quantized NVFP4 matvec where packed/scales come from a staging view.
    pub(crate) fn launch_nvfp4_prequantized_views(
        &self,
        packed: &CudaView<u8>,
        scales: &CudaView<u8>,
        rows: usize,
        cols: usize,
        output_scale: f32,
        quantized_input: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 128 * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.nvfp4_prequant)
                .arg(packed)
                .arg(scales)
                .arg(quantized_input)
                .arg(&rows_u32)
                .arg(&cols_u32)
                .arg(&output_scale)
                .arg(output)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch staged nvfp4 prequantized"))?;
        Ok(())
    }

    /// Pre-quantized batched NVFP4 matvec where packed/scales come from a staging view.
    pub(crate) fn launch_nvfp4_prequantized_batched_views(
        &self,
        packed: &CudaView<u8>,
        scales: &CudaView<u8>,
        rows: usize,
        cols: usize,
        output_scale: f32,
        quantized_input: &CudaSlice<f32>,
        batch: usize,
        output: &mut CudaSlice<f32>,
    ) -> Result<()> {
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let batch_u32 = batch as u32;
        if batch > 1 {
            // Same WMMA gating as the device-buffer path (see
            // matvec_nvfp4_prequantized_batched_device). Routed-expert
            // weights stream through here, so this is the hottest dispatch
            // for prefill — but per-expert batch (~8 for typical chunk_size)
            // is too small for WMMA's 32-thread blocks to saturate SMs.
            // Gate on batch>=32; grouped MoE will deliver larger batches.
            let wmma_disabled = std::env::var("AEGIS_NVFP4_WMMA_DISABLE").is_ok();
            let wmma_eligible = !wmma_disabled
                && rows % 16 == 0
                && cols % 16 == 0
                && batch >= 32;
            if wmma_eligible {
                let grid_x = (rows / 16) as u32;
                let grid_y = ((batch + 15) / 16) as u32;
                let cfg = LaunchConfig {
                    grid_dim: (grid_x, grid_y, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    self.stream
                        .launch_builder(&self.kernels.nvfp4_prequant_batched_gemm_wmma_bf16)
                        .arg(packed)
                        .arg(scales)
                        .arg(quantized_input)
                        .arg(&rows_u32)
                        .arg(&cols_u32)
                        .arg(&batch_u32)
                        .arg(&output_scale)
                        .arg(output)
                        .launch(cfg)
                }
                .map_err(map_cuda_err("launch staged nvfp4 wmma bf16 batched gemm"))?;
                return Ok(());
            }
            let grid_y = ((batch + 7) / 8) as u32;
            let shared = (cols / 2 + cols / 16) as u32;
            let cfg = LaunchConfig {
                grid_dim: (rows_u32, grid_y, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: shared,
            };
            unsafe {
                self.stream
                    .launch_builder(&self.kernels.nvfp4_prequant_batched_gemm)
                    .arg(packed)
                    .arg(scales)
                    .arg(quantized_input)
                    .arg(&rows_u32)
                    .arg(&cols_u32)
                    .arg(&batch_u32)
                    .arg(&output_scale)
                    .arg(output)
                    .launch(cfg)
            }
            .map_err(map_cuda_err("launch staged nvfp4 prequantized batched gemm"))?;
        } else {
            let cfg = LaunchConfig {
                grid_dim: (rows_u32, batch_u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 256 * std::mem::size_of::<f32>() as u32,
            };
            unsafe {
                self.stream
                    .launch_builder(&self.kernels.nvfp4_prequant_batched)
                    .arg(packed)
                    .arg(scales)
                    .arg(quantized_input)
                    .arg(&rows_u32)
                    .arg(&cols_u32)
                    .arg(&output_scale)
                    .arg(output)
                    .launch(cfg)
            }
            .map_err(map_cuda_err("launch staged nvfp4 prequantized batched"))?;
        }
        Ok(())
    }

    /// Native MXFP4 batched matvec where the weight data comes from a staging view
    /// (CudaView<u8>) rather than the owned DeviceNvfp4Linear.native_mxfp4.data slice.
    /// Used for host-resident layers whose repacked data was staged just before this call.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch_native_mxfp4_batched_views(
        &self,
        weight_data: &CudaView<u8>,
        input_mxfp4: &DeviceBuffer<u8>,
        rows: usize,
        cols: usize,
        blocks_per_row: usize,
        batch: usize,
        output_scale: f32,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let rows = u32_arg("rows", rows)?;
        let cols = u32_arg("cols", cols)?;
        let blocks_per_row = u32_arg("blocks_per_row", blocks_per_row)?;
        let batch_u32 = u32_arg("batch", batch)?;
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
                    .arg(weight_data)
                    .arg(&input_mxfp4.slice)
                    .arg(&rows)
                    .arg(&cols)
                    .arg(&blocks_per_row)
                    .arg(&batch_u32)
                    .arg(&output_scale)
                    .arg(&mut output.slice)
                    .launch(cfg)
            }
            .map_err(map_cuda_err("launch staged native mxfp4 prefill tile"))?;
        } else if use_n8_kernel {
            let cfg = LaunchConfig {
                grid_dim: (ceil_div(rows, 16), ceil_div(batch_u32, 8), 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.stream
                    .launch_builder(&self.kernels.mxfp4_matmul_n8)
                    .arg(weight_data)
                    .arg(&input_mxfp4.slice)
                    .arg(&rows)
                    .arg(&cols)
                    .arg(&blocks_per_row)
                    .arg(&batch_u32)
                    .arg(&output_scale)
                    .arg(&mut output.slice)
                    .launch(cfg)
            }
            .map_err(map_cuda_err("launch staged native mxfp4 n8"))?;
        } else {
            let cfg = LaunchConfig {
                grid_dim: (ceil_div(rows, 16), batch_u32, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.stream
                    .launch_builder(&self.kernels.mxfp4_matvec)
                    .arg(weight_data)
                    .arg(&input_mxfp4.slice)
                    .arg(&rows)
                    .arg(&cols)
                    .arg(&blocks_per_row)
                    .arg(&output_scale)
                    .arg(&mut output.slice)
                    .launch(cfg)
            }
            .map_err(map_cuda_err("launch staged native mxfp4 matvec"))?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // High-level staging helpers: take DeviceBuffer + LinearStagingPool, do
    // the H2D copy and kernel dispatch in one call.  Callers in executor/ use
    // these because they cannot access DeviceNvfp4Linear::host_weights directly
    // (it is pub(super) within cuda/).
    // -----------------------------------------------------------------------

    /// Staged decode matvec (M=1): H2D copy host weights to staging VRAM, then run
    /// the pre-quantized kernel.  `quantized_input` must already hold the fp4-
    /// quantized activations (caller ran `quantize_nvfp4_input_device` first).
    pub(crate) fn matvec_nvfp4_staged_prequantized_device(
        &self,
        linear: &DeviceNvfp4Linear,
        staging: &mut LinearStagingPool,
        quantized_input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let hw = linear.host_weights.as_deref().ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "staged matvec called on non-host-resident linear `{}`",
                linear.name
            ))
        })?;
        let slot = staging.prepare_async(self, hw, linear.packed_bytes, linear.scale_bytes)?;
        let result = {
            let packed_view = staging.packed_view(slot, linear.packed_bytes);
            let scales_view = staging.scales_view(slot, linear.scale_bytes);
            self.launch_nvfp4_prequantized_views(
                &packed_view,
                &scales_view,
                linear.rows,
                linear.cols,
                linear.output_scale,
                &quantized_input.slice,
                &mut output.slice,
            )
        };
        staging.mark_kernel_launched(self, slot)?;
        result
    }

    /// Staged prefill matvec (M=batch): same H2D staging, batched kernel.
    pub(crate) fn matvec_nvfp4_staged_prequantized_batched_device(
        &self,
        linear: &DeviceNvfp4Linear,
        staging: &mut LinearStagingPool,
        quantized_input: &DeviceBuffer<f32>,
        batch: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let hw = linear.host_weights.as_deref().ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "staged batched matvec called on non-host-resident linear `{}`",
                linear.name
            ))
        })?;
        let slot = staging.prepare_async(self, hw, linear.packed_bytes, linear.scale_bytes)?;
        let result = {
            let packed_view = staging.packed_view(slot, linear.packed_bytes);
            let scales_view = staging.scales_view(slot, linear.scale_bytes);
            self.launch_nvfp4_prequantized_batched_views(
                &packed_view,
                &scales_view,
                linear.rows,
                linear.cols,
                linear.output_scale,
                &quantized_input.slice,
                batch,
                &mut output.slice,
            )
        };
        staging.mark_kernel_launched(self, slot)?;
        result
    }

    /// Staged decode matvec (M=1) using native MXFP4 tensor cores.
    /// Stages the repacked weight data, then runs the Blackwell tensor-core kernel.
    pub(crate) fn matvec_native_mxfp4_staged_device(
        &self,
        linear: &DeviceNvfp4Linear,
        staging: &mut LinearStagingPool,
        input_mxfp4: &DeviceBuffer<u8>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let hw = linear.host_weights.as_deref().ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "staged native mxfp4 matvec on non-host-resident linear `{}`",
                linear.name
            ))
        })?;
        let mxfp4 = hw.native_mxfp4.as_ref().ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "staged native mxfp4 matvec: no repacked data for `{}`; set native_mxfp4_repack=true",
                linear.name
            ))
        })?;
        // Stage packed/scales into a fresh slot, then layer the native-mxfp4
        // bytes into the same slot (consumed by the same kernel launch).
        let slot = staging.prepare_async(self, hw, linear.packed_bytes, linear.scale_bytes)?;
        staging.prepare_native_mxfp4_into_last(self, mxfp4)?;
        let result = {
            let weight_view = staging
                .native_mxfp4_view(slot, mxfp4.data.len())
                .ok_or_else(|| {
                    AegisError::InvalidPlan("native MXFP4 staging buffer not allocated".into())
                })?;
            self.launch_native_mxfp4_batched_views(
                &weight_view,
                input_mxfp4,
                linear.rows,
                linear.cols,
                mxfp4.blocks_per_row,
                1,
                linear.output_scale,
                output,
            )
        };
        staging.mark_kernel_launched(self, slot)?;
        result
    }

    /// Staged prefill matmul (M=batch) using native MXFP4 tensor cores.
    pub(crate) fn matvec_native_mxfp4_staged_batched_device(
        &self,
        linear: &DeviceNvfp4Linear,
        staging: &mut LinearStagingPool,
        input_mxfp4: &DeviceBuffer<u8>,
        batch: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let hw = linear.host_weights.as_deref().ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "staged native mxfp4 batched matvec on non-host-resident linear `{}`",
                linear.name
            ))
        })?;
        let mxfp4 = hw.native_mxfp4.as_ref().ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "staged native mxfp4 batched matvec: no repacked data for `{}`; set native_mxfp4_repack=true",
                linear.name
            ))
        })?;
        let slot = staging.prepare_async(self, hw, linear.packed_bytes, linear.scale_bytes)?;
        staging.prepare_native_mxfp4_into_last(self, mxfp4)?;
        let result = {
            let weight_view = staging
                .native_mxfp4_view(slot, mxfp4.data.len())
                .ok_or_else(|| {
                    AegisError::InvalidPlan("native MXFP4 staging buffer not allocated".into())
                })?;
            self.launch_native_mxfp4_batched_views(
                &weight_view,
                input_mxfp4,
                linear.rows,
                linear.cols,
                mxfp4.blocks_per_row,
                batch,
                linear.output_scale,
                output,
            )
        };
        staging.mark_kernel_launched(self, slot)?;
        result
    }
}

impl DeviceNvfp4Linear {
    /// Returns `true` if this layer is host-resident AND has native MXFP4 repacked data,
    /// meaning inference can use staged tensor-core path instead of software NVfp4.
    pub fn is_host_resident_with_native_mxfp4(&self) -> bool {
        self.host_weights
            .as_ref()
            .is_some_and(|hw| hw.native_mxfp4.is_some())
    }

    /// Byte count of the native MXFP4 repacked data in host RAM (0 if absent).
    pub fn host_resident_native_mxfp4_bytes(&self) -> usize {
        self.host_weights
            .as_ref()
            .and_then(|hw| hw.native_mxfp4.as_ref())
            .map(|m| m.data.len())
            .unwrap_or(0)
    }

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

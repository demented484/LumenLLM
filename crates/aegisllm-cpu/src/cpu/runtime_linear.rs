use super::simd;
use super::{CpuNvfp4Data, CpuNvfp4Linear};
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::tensor::quant::{
    QK_NVFP4_SUB, decode_nvfp4_nibble_i8, decode_ue4m3_with_half_lut, quantize_input_nvfp4,
};
use rayon::prelude::*;
use std::cell::RefCell;

thread_local! {
    /// Per-rayon-worker scratch for dequantized weight rows.
    /// Reused across calls to avoid re-allocating on every matvec row.
    static DEQUANT_ROW: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

impl CpuNvfp4Linear {
    pub fn materialized_extra_bytes(&self) -> u64 {
        match &self.data {
            CpuNvfp4Data::Packed { .. } => 0,
            CpuNvfp4Data::UnpackedI8 { weights, scales } => {
                weights.len() as u64 + (scales.len() * std::mem::size_of::<f32>()) as u64
            }
        }
    }

    pub fn matvec(&self, input: &[f32]) -> Result<Vec<f32>> {
        let mut output = vec![0.0_f32; self.rows];
        self.matvec_into(input, &mut output)?;
        Ok(output)
    }

    /// Batched matvec: process `batch` input vectors through the same weight matrix.
    /// Layout: input[token * cols + i], output[token * rows + r]. Dequantizes each row
    /// once and reuses across the whole batch — much faster than calling matvec_into in
    /// a loop because the weight row stays in L1/L2 cache.
    pub fn matmul_into(&self, input: &[f32], batch: usize, output: &mut [f32]) -> Result<()> {
        if batch == 0 {
            return Ok(());
        }
        if input.len() != batch * self.cols || output.len() != batch * self.rows {
            return Err(AegisError::InvalidPlan(format!(
                "nvfp4 batched matmul shape mismatch for {}: expected input={} output={} (batch={} rows={} cols={}), got input={} output={}",
                self.name,
                batch * self.cols,
                batch * self.rows,
                batch,
                self.rows,
                self.cols,
                input.len(),
                output.len()
            )));
        }
        // Quantize each input row to NVfp4 representable values (per-token).
        let mut quantized_inputs: Option<Vec<f32>> = None;
        if self.input_scale > 0.0 {
            let mut q = Vec::with_capacity(batch * self.cols);
            for token in 0..batch {
                let row = &input[token * self.cols..(token + 1) * self.cols];
                if let Some(q_row) = quantize_input_nvfp4(row, self.input_scale) {
                    q.extend_from_slice(&q_row);
                } else {
                    q.extend_from_slice(row);
                }
            }
            quantized_inputs = Some(q);
        }
        let input_view = quantized_inputs.as_deref().unwrap_or(input);
        let scale_cols = self.cols / QK_NVFP4_SUB;
        let cols = self.cols;
        let rows = self.rows;
        let output_scale = self.output_scale;

        // Parallelize over rows. Each row dequantizes once into thread-local scratch
        // and SIMD-dots against all batch tokens.
        match &self.data {
            CpuNvfp4Data::Packed { packed, scales } => {
                let packed = packed.as_bytes();
                let scales = scales.as_bytes();
                let packed_cols = cols / 2;
                // Output is batch-major (token outer, row inner): rewrite as transposed accumulator.
                // For cache locality on rows, accumulate per row across all tokens, then scatter.
                let row_outputs: Vec<Vec<f32>> = (0..rows)
                    .into_par_iter()
                    .map(|row| {
                        let packed_row = &packed[row * packed_cols..(row + 1) * packed_cols];
                        let scale_row = &scales[row * scale_cols..(row + 1) * scale_cols];
                        DEQUANT_ROW.with(|scratch| {
                            let mut scratch = scratch.borrow_mut();
                            scratch.clear();
                            scratch.resize(cols, 0.0_f32);
                            for (block_idx, &bs_raw) in scale_row.iter().enumerate() {
                                let block_scale = decode_ue4m3_with_half_lut(bs_raw);
                                let input_base = block_idx * QK_NVFP4_SUB;
                                let packed_base = block_idx * (QK_NVFP4_SUB / 2);
                                for j in 0..(QK_NVFP4_SUB / 2) {
                                    let byte = packed_row[packed_base + j];
                                    let lo = input_base + j * 2;
                                    let hi = lo + 1;
                                    scratch[lo] = decode_nvfp4_nibble_i8(byte & 0x0f) as f32 * block_scale;
                                    scratch[hi] = decode_nvfp4_nibble_i8(byte >> 4) as f32 * block_scale;
                                }
                            }
                            // SIMD dot against each batch token.
                            let mut row_out = vec![0.0_f32; batch];
                            for token in 0..batch {
                                let in_row = &input_view[token * cols..(token + 1) * cols];
                                row_out[token] = simd::dot_f32(&scratch, in_row) * output_scale;
                            }
                            row_out
                        })
                    })
                    .collect();
                // Scatter row_outputs[row][token] → output[token * rows + row]
                for (row, row_out) in row_outputs.iter().enumerate() {
                    for token in 0..batch {
                        output[token * rows + row] = row_out[token];
                    }
                }
            }
            CpuNvfp4Data::UnpackedI8 { weights, scales } => {
                let row_outputs: Vec<Vec<f32>> = (0..rows)
                    .into_par_iter()
                    .map(|row| {
                        let weight_row = &weights[row * cols..(row + 1) * cols];
                        let scale_row = &scales[row * scale_cols..(row + 1) * scale_cols];
                        DEQUANT_ROW.with(|scratch| {
                            let mut scratch = scratch.borrow_mut();
                            scratch.clear();
                            scratch.resize(cols, 0.0_f32);
                            for (block_idx, &block_scale) in scale_row.iter().enumerate() {
                                let input_base = block_idx * QK_NVFP4_SUB;
                                for k in 0..QK_NVFP4_SUB {
                                    scratch[input_base + k] =
                                        weight_row[input_base + k] as f32 * block_scale;
                                }
                            }
                            let mut row_out = vec![0.0_f32; batch];
                            for token in 0..batch {
                                let in_row = &input_view[token * cols..(token + 1) * cols];
                                row_out[token] = simd::dot_f32(&scratch, in_row) * output_scale;
                            }
                            row_out
                        })
                    })
                    .collect();
                for (row, row_out) in row_outputs.iter().enumerate() {
                    for token in 0..batch {
                        output[token * rows + row] = row_out[token];
                    }
                }
            }
        }
        Ok(())
    }

    pub fn matvec_into(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != self.cols || output.len() != self.rows {
            return Err(AegisError::InvalidPlan(format!(
                "nvfp4 linear shape mismatch for {}: expected input={} output={}, got input={} output={}",
                self.name,
                self.cols,
                self.rows,
                input.len(),
                output.len()
            )));
        }
        let quantized_input = quantize_input_nvfp4(input, self.input_scale);
        let input = quantized_input.as_deref().unwrap_or(input);
        let scale_cols = self.cols / QK_NVFP4_SUB;

        match &self.data {
            CpuNvfp4Data::Packed { packed, scales } => {
                let packed = packed.as_bytes();
                let scales = scales.as_bytes();
                if packed.len() != self.packed_bytes || scales.len() != self.scale_bytes {
                    return Err(AegisError::InvalidPlan(format!(
                        "nvfp4 linear `{}` byte mismatch: expected packed={} scales={}, got packed={} scales={}",
                        self.name,
                        self.packed_bytes,
                        self.scale_bytes,
                        packed.len(),
                        scales.len()
                    )));
                }
                let packed_cols = self.cols / 2;
                let cols = self.cols;
                let output_scale = self.output_scale;
                output.par_iter_mut().enumerate().for_each(|(row, slot)| {
                    let packed_row = &packed[row * packed_cols..(row + 1) * packed_cols];
                    let scale_row = &scales[row * scale_cols..(row + 1) * scale_cols];
                    DEQUANT_ROW.with(|scratch| {
                        let mut scratch = scratch.borrow_mut();
                        scratch.clear();
                        scratch.resize(cols, 0.0_f32);
                        // Dequantize entire row: scratch[col] = decode(nibble) * block_scale.
                        // This runs scalar dequant + SIMD dot — much faster than scalar fused loop
                        // because the dot reduction vectorizes 8/16-wide.
                        for (block_idx, &block_scale_raw) in scale_row.iter().enumerate() {
                            let block_scale = decode_ue4m3_with_half_lut(block_scale_raw);
                            let input_base = block_idx * QK_NVFP4_SUB;
                            let packed_base = block_idx * (QK_NVFP4_SUB / 2);
                            for j in 0..(QK_NVFP4_SUB / 2) {
                                let byte = packed_row[packed_base + j];
                                let lo_col = input_base + j * 2;
                                let hi_col = lo_col + 1;
                                scratch[lo_col] = decode_nvfp4_nibble_i8(byte & 0x0f) as f32
                                    * block_scale;
                                scratch[hi_col] = decode_nvfp4_nibble_i8(byte >> 4) as f32
                                    * block_scale;
                            }
                        }
                        *slot = simd::dot_f32(&scratch, input) * output_scale;
                    });
                });
            }
            CpuNvfp4Data::UnpackedI8 { weights, scales } => {
                if weights.len() != self.rows * self.cols || scales.len() != self.rows * scale_cols
                {
                    return Err(AegisError::InvalidPlan(format!(
                        "unpacked nvfp4 linear `{}` size mismatch",
                        self.name
                    )));
                }
                let cols = self.cols;
                let output_scale = self.output_scale;
                output.par_iter_mut().enumerate().for_each(|(row, slot)| {
                    let weight_row = &weights[row * cols..(row + 1) * cols];
                    let scale_row = &scales[row * scale_cols..(row + 1) * scale_cols];
                    DEQUANT_ROW.with(|scratch| {
                        let mut scratch = scratch.borrow_mut();
                        scratch.clear();
                        scratch.resize(cols, 0.0_f32);
                        // Widen i8 → f32 and apply per-block scale into scratch.
                        for (block_idx, &block_scale) in scale_row.iter().enumerate() {
                            let input_base = block_idx * QK_NVFP4_SUB;
                            for k in 0..QK_NVFP4_SUB {
                                scratch[input_base + k] =
                                    weight_row[input_base + k] as f32 * block_scale;
                            }
                        }
                        *slot = simd::dot_f32(&scratch, input) * output_scale;
                    });
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::runtime_loader::unpack_nvfp4_for_test;
    use aegisllm_base::planning::placement::StoragePlacement;
    use aegisllm_base::tensor::layout::LinearResidentLayout;
    use aegisllm_base::tensor::quant::{Nvfp4LinearSpec, QK_NVFP4_SUB};
    use aegisllm_base::tensor::storage::{HostTensorStorage, LoadedHostTensor, TensorResidencyPlan};

    fn tiny_linear(input_scale: f32) -> CpuNvfp4Linear {
        tiny_linear_with_layout(input_scale, LinearResidentLayout::PackedSource)
    }

    fn tiny_linear_with_layout(
        input_scale: f32,
        resident_layout: LinearResidentLayout,
    ) -> CpuNvfp4Linear {
        let rows = 2;
        let cols = 64;
        let packed_bytes = rows * cols / 2;
        let scale_bytes = rows * cols / QK_NVFP4_SUB;
        let packed = LoadedHostTensor {
            name: "tiny.weight".into(),
            storage: HostTensorStorage::Ram(vec![0x21; packed_bytes]),
            shard_path: std::path::PathBuf::new(),
        };
        let scales = LoadedHostTensor {
            name: "tiny.weight_scale".into(),
            storage: HostTensorStorage::Ram(vec![0x40; scale_bytes]),
            shard_path: std::path::PathBuf::new(),
        };
        let data = match resident_layout {
            LinearResidentLayout::PackedSource => CpuNvfp4Data::Packed { packed, scales },
            LinearResidentLayout::UnpackedI8Scales => {
                let spec = Nvfp4LinearSpec {
                    name: "tiny".into(),
                    rows,
                    cols,
                    packed_bytes,
                    scale_bytes,
                    input_scale,
                    output_scale: 0.5,
                };
                let (weights, scales) =
                    unpack_nvfp4_for_test(packed.as_bytes(), scales.as_bytes(), &spec).unwrap();
                CpuNvfp4Data::UnpackedI8 { weights, scales }
            }
            other => panic!("unsupported test layout {other:?}"),
        };
        CpuNvfp4Linear {
            name: "tiny".into(),
            rows,
            cols,
            packed_bytes,
            scale_bytes,
            input_scale,
            output_scale: 0.5,
            residency: TensorResidencyPlan::RamResident,
            store: StoragePlacement::Ram,
            resident_layout,
            data,
        }
    }

    #[test]
    fn nvfp4_matvec_decodes_packed_rows() {
        let linear = tiny_linear(0.0);
        let output = linear.matvec(&vec![1.0; linear.cols]).unwrap();
        assert_eq!(output, vec![48.0, 48.0]);
    }

    #[test]
    fn nvfp4_matvec_quantized_zero_input_stays_zero() {
        let linear = tiny_linear(1.0);
        let output = linear.matvec(&vec![0.0; linear.cols]).unwrap();
        assert_eq!(output, vec![0.0, 0.0]);
    }

    #[test]
    fn nvfp4_matmul_into_matches_matvec_loop() {
        let linear = tiny_linear(0.1);
        let batch = 4;
        let cols = linear.cols;
        let mut inputs = Vec::with_capacity(batch * cols);
        for token in 0..batch {
            for i in 0..cols {
                inputs.push((token as f32 + i as f32 / 13.0) * 0.5 - 1.0);
            }
        }
        let mut batched = vec![0.0_f32; batch * linear.rows];
        linear.matmul_into(&inputs, batch, &mut batched).unwrap();

        let mut expected = vec![0.0_f32; batch * linear.rows];
        for token in 0..batch {
            let in_row = &inputs[token * cols..(token + 1) * cols];
            let out_row = &mut expected[token * linear.rows..(token + 1) * linear.rows];
            linear.matvec_into(in_row, out_row).unwrap();
        }
        for i in 0..batched.len() {
            assert!(
                (batched[i] - expected[i]).abs() < 1e-5,
                "mismatch at {i}: batched={} expected={}",
                batched[i],
                expected[i]
            );
        }
    }

    #[test]
    fn nvfp4_unpacked_i8_matches_packed_layout() {
        let packed = tiny_linear_with_layout(0.0, LinearResidentLayout::PackedSource);
        let unpacked = tiny_linear_with_layout(0.0, LinearResidentLayout::UnpackedI8Scales);
        let input = (0..packed.cols)
            .map(|idx| idx as f32 / 17.0 - 1.0)
            .collect::<Vec<_>>();
        assert_eq!(
            packed.matvec(&input).unwrap(),
            unpacked.matvec(&input).unwrap()
        );
    }
}

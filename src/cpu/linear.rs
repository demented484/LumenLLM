use super::{CpuNvfp4Data, CpuNvfp4Linear};
use crate::error::{AegisError, Result};
use crate::tensor::quant::{
    QK_NVFP4_SUB, decode_nvfp4_nibble_i8, decode_ue4m3_with_half_lut, quantize_input_nvfp4,
};
use rayon::prelude::*;

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
                output.par_iter_mut().enumerate().for_each(|(row, slot)| {
                    let packed_row = &packed[row * packed_cols..(row + 1) * packed_cols];
                    let scale_row = &scales[row * scale_cols..(row + 1) * scale_cols];
                    let mut acc = 0.0_f32;
                    for block_idx in 0..scale_cols {
                        let block_scale = decode_ue4m3_with_half_lut(scale_row[block_idx]);
                        let input_base = block_idx * QK_NVFP4_SUB;
                        let packed_base = block_idx * (QK_NVFP4_SUB / 2);
                        for j in 0..(QK_NVFP4_SUB / 2) {
                            let byte = packed_row[packed_base + j];
                            let lo_col = input_base + j * 2;
                            let hi_col = lo_col + 1;
                            acc += decode_nvfp4_nibble_i8(byte & 0x0f) as f32
                                * block_scale
                                * input[lo_col];
                            acc += decode_nvfp4_nibble_i8(byte >> 4) as f32
                                * block_scale
                                * input[hi_col];
                        }
                    }
                    *slot = acc * self.output_scale;
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
                output.par_iter_mut().enumerate().for_each(|(row, slot)| {
                    let weight_row = &weights[row * self.cols..(row + 1) * self.cols];
                    let scale_row = &scales[row * scale_cols..(row + 1) * scale_cols];
                    let mut acc = 0.0_f32;
                    for block_idx in 0..scale_cols {
                        let block_scale = scale_row[block_idx];
                        let input_base = block_idx * QK_NVFP4_SUB;
                        let weight_block = &weight_row[input_base..input_base + QK_NVFP4_SUB];
                        let input_block = &input[input_base..input_base + QK_NVFP4_SUB];
                        for (&weight, &value) in weight_block.iter().zip(input_block.iter()) {
                            acc += weight as f32 * block_scale * value;
                        }
                    }
                    *slot = acc * self.output_scale;
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::loader::unpack_nvfp4_for_test;
    use crate::planning::placement::StoragePlacement;
    use crate::tensor::layout::LinearResidentLayout;
    use crate::tensor::quant::{Nvfp4LinearSpec, QK_NVFP4_SUB};
    use crate::tensor::storage::{HostTensorStorage, LoadedHostTensor, TensorResidencyPlan};

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
        };
        let scales = LoadedHostTensor {
            name: "tiny.weight_scale".into(),
            storage: HostTensorStorage::Ram(vec![0x40; scale_bytes]),
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

use rayon::prelude::*;

use crate::artifact::ModelArtifact;
use crate::error::{AegisError, Result};
use crate::planning::placement::StoragePlacement;
use crate::tensor::storage::{LoadedHostTensor, TensorStorageLoader};
use crate::tensor::{TensorDType, TensorInfo};

#[derive(Debug, Clone)]
pub struct Bf16Matrix {
    name: String,
    pub rows: usize,
    pub cols: usize,
    tensor: LoadedHostTensor,
}

impl Bf16Matrix {
    pub fn from_named_tensor(
        artifact: &ModelArtifact,
        name: &str,
        store: StoragePlacement,
        loader: &mut TensorStorageLoader,
    ) -> Result<Self> {
        let tensor = require_tensor(artifact, name)?;
        Self::from_tensor(tensor, store, loader)
    }

    pub fn from_first_existing_tensor(
        artifact: &ModelArtifact,
        names: &[&str],
        store: StoragePlacement,
        loader: &mut TensorStorageLoader,
    ) -> Result<Self> {
        let tensor = names
            .iter()
            .find_map(|name| artifact.tensors.get(name))
            .ok_or_else(|| {
                AegisError::InvalidPlan(format!(
                    "missing any of dense matrix tensors: {}",
                    names.join(", ")
                ))
            })?;
        Self::from_tensor(tensor, store, loader)
    }

    pub fn from_tensor(
        tensor: &TensorInfo,
        store: StoragePlacement,
        loader: &mut TensorStorageLoader,
    ) -> Result<Self> {
        if tensor.dtype != TensorDType::BF16 || tensor.shape.len() != 2 {
            return Err(AegisError::InvalidPlan(format!(
                "`{}` must be a BF16 matrix",
                tensor.name
            )));
        }
        Ok(Self {
            name: tensor.name.clone(),
            rows: tensor.shape[0],
            cols: tensor.shape[1],
            tensor: loader.load_for_store(tensor, store)?,
        })
    }

    pub fn row(&self, row: usize) -> Result<Vec<f32>> {
        if row >= self.rows {
            return Err(AegisError::InvalidPlan(format!(
                "token id {row} is out of range for `{}` rows={}",
                self.name, self.rows
            )));
        }
        let bytes = self.tensor.as_bytes();
        let start = row * self.cols * 2;
        let row_bytes = &bytes[start..start + self.cols * 2];
        let mut out = vec![0.0; self.cols];
        for (dst, chunk) in out.iter_mut().zip(row_bytes.chunks_exact(2)) {
            *dst = bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
        }
        Ok(out)
    }

    pub fn matvec_into(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != self.cols {
            return Err(AegisError::InvalidPlan(format!(
                "BF16 matrix `{}` input mismatch: expected {}, got {}",
                self.name,
                self.cols,
                input.len()
            )));
        }
        if output.len() != self.rows {
            return Err(AegisError::InvalidPlan(format!(
                "BF16 matrix `{}` output mismatch: expected {}, got {}",
                self.name,
                self.rows,
                output.len()
            )));
        }
        let bytes = self.tensor.as_bytes();
        output.par_iter_mut().enumerate().for_each(|(row, slot)| {
            let start = row * self.cols * 2;
            let row_bytes = &bytes[start..start + self.cols * 2];
            *slot = dot_bf16_bytes_f32(row_bytes, input);
        });
        Ok(())
    }
}

pub fn require_tensor<'a>(
    artifact: &'a ModelArtifact,
    name: &str,
) -> Result<&'a TensorInfo> {
    artifact
        .tensors
        .get(name)
        .ok_or_else(|| AegisError::InvalidPlan(format!("missing tensor `{name}`")))
}

pub fn read_dense_vector(
    tensor: &TensorInfo,
    store: StoragePlacement,
    loader: &mut TensorStorageLoader,
) -> Result<Vec<f32>> {
    if tensor.shape.len() != 1 {
        return Err(AegisError::InvalidPlan(format!(
            "`{}` must be a dense vector",
            tensor.name
        )));
    }
    let loaded = loader.load_for_store(tensor, store)?;
    let bytes = loaded.as_bytes();
    match tensor.dtype {
        TensorDType::BF16 => Ok(bytes
            .chunks_exact(2)
            .map(|chunk| bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
            .collect()),
        TensorDType::F32 => Ok(bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()),
        other => Err(AegisError::InvalidPlan(format!(
            "`{}` must be BF16 or F32 vector, got {:?}",
            tensor.name, other
        ))),
    }
}

fn dot_bf16_bytes_f32(row: &[u8], input: &[f32]) -> f32 {
    row.chunks_exact(2)
        .zip(input.iter())
        .map(|(chunk, &value)| bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])) * value)
        .sum()
}

#[inline(always)]
pub fn bf16_to_f32(value: u16) -> f32 {
    f32::from_bits((value as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_decode_roundtrips_one() {
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
    }
}

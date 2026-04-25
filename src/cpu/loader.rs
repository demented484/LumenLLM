use super::{CpuNvfp4Data, CpuNvfp4Linear, CpuRuntime};
use crate::artifact::ModelArtifact;
use crate::error::{AegisError, Result};
use crate::graph::{GraphRegion, TensorRole};
use crate::planning::placement::{ComputePlacement, RegionPlacement, StoragePlacement};
use crate::tensor::layout::LinearResidentLayout;
use crate::tensor::quant::{
    Nvfp4LinearSpec, QK_NVFP4_SUB, decode_nvfp4_nibble_i8, decode_ue4m3_with_half_lut,
};
use crate::tensor::storage::{TensorResidencyPlan, TensorStorageLoader};
use crate::tensor::{TensorDType, TensorInfo};

impl CpuRuntime {
    pub fn load_nvfp4_linear_with_store(
        &self,
        artifact: &ModelArtifact,
        prefix: &str,
        store: StoragePlacement,
        residency: TensorResidencyPlan,
        loader: &mut TensorStorageLoader,
    ) -> Result<CpuNvfp4Linear> {
        self.load_nvfp4_linear_with_layout(
            artifact,
            prefix,
            store,
            residency,
            LinearResidentLayout::PackedSource,
            loader,
        )
    }

    pub fn load_nvfp4_linear_with_layout(
        &self,
        artifact: &ModelArtifact,
        prefix: &str,
        store: StoragePlacement,
        residency: TensorResidencyPlan,
        resident_layout: LinearResidentLayout,
        loader: &mut TensorStorageLoader,
    ) -> Result<CpuNvfp4Linear> {
        let weight = artifact
            .tensors
            .get(&format!("{prefix}.weight"))
            .ok_or_else(|| AegisError::InvalidPlan(format!("missing `{prefix}.weight`")))?;
        let scales = artifact
            .tensors
            .get(&format!("{prefix}.weight_scale"))
            .ok_or_else(|| AegisError::InvalidPlan(format!("missing `{prefix}.weight_scale`")))?;
        let output_scale = artifact
            .tensors
            .get(&format!("{prefix}.weight_scale_2"))
            .map(|tensor| read_scalar_f32_with_loader(loader, tensor, store))
            .transpose()?
            .unwrap_or(1.0);
        let input_scale = artifact
            .tensors
            .get(&format!("{prefix}.input_scale"))
            .map(|tensor| read_scalar_f32_with_loader(loader, tensor, store))
            .transpose()?
            .unwrap_or(1.0);
        let spec =
            Nvfp4LinearSpec::from_tensors(prefix, weight, scales, input_scale, output_scale)?;
        let packed = loader.load_for_store(weight, store)?;
        let scale_bytes = loader.load_for_store(scales, store)?;
        let data = match resident_layout {
            LinearResidentLayout::PackedSource => CpuNvfp4Data::Packed {
                packed,
                scales: scale_bytes,
            },
            LinearResidentLayout::UnpackedI8Scales => {
                let (weights, scales) =
                    unpack_nvfp4_linear(packed.as_bytes(), scale_bytes.as_bytes(), &spec)?;
                CpuNvfp4Data::UnpackedI8 { weights, scales }
            }
            other => {
                return Err(AegisError::Unsupported(format!(
                    "CPU NVFP4 resident layout `{other}` is not implemented for `{prefix}`"
                )));
            }
        };

        Ok(CpuNvfp4Linear {
            name: spec.name,
            rows: spec.rows,
            cols: spec.cols,
            packed_bytes: spec.packed_bytes,
            scale_bytes: spec.scale_bytes,
            input_scale: spec.input_scale,
            output_scale: spec.output_scale,
            residency,
            store,
            resident_layout,
            data,
        })
    }

    pub fn load_placed_region_nvfp4_linears(
        &self,
        artifact: &ModelArtifact,
        region: &GraphRegion,
        placement: &RegionPlacement,
    ) -> Result<Vec<CpuNvfp4Linear>> {
        if placement.region_id != region.id {
            return Err(AegisError::InvalidPlan(format!(
                "placement `{}` does not match graph region `{}`",
                placement.region_id.0, region.id.0
            )));
        }
        if !matches!(placement.compute, ComputePlacement::Cpu) {
            return Err(AegisError::Unsupported(format!(
                "region `{}` is compute={}; CPU loader refused to load it",
                region.id.0, placement.compute
            )));
        }
        let residency = match placement.store {
            StoragePlacement::Ram => TensorResidencyPlan::RamResident,
            StoragePlacement::Mmap => TensorResidencyPlan::FileBackedMmap,
            StoragePlacement::Vram { device } => TensorResidencyPlan::StagedDeviceToHost { device },
        };

        let mut linears = Vec::new();
        let mut loader = TensorStorageLoader::new();
        for prefix in nvfp4_linear_prefixes(region) {
            linears.push(self.load_nvfp4_linear_with_store(
                artifact,
                prefix,
                placement.store,
                residency,
                &mut loader,
            )?);
        }
        Ok(linears)
    }

    pub fn load_first_placed_region_nvfp4_linear(
        &self,
        artifact: &ModelArtifact,
        region: &GraphRegion,
        placement: &RegionPlacement,
    ) -> Result<Option<CpuNvfp4Linear>> {
        self.load_first_placed_region_nvfp4_linear_with_layout(
            artifact,
            region,
            placement,
            LinearResidentLayout::PackedSource,
        )
    }

    pub fn load_first_placed_region_nvfp4_linear_with_layout(
        &self,
        artifact: &ModelArtifact,
        region: &GraphRegion,
        placement: &RegionPlacement,
        resident_layout: LinearResidentLayout,
    ) -> Result<Option<CpuNvfp4Linear>> {
        if placement.region_id != region.id {
            return Err(AegisError::InvalidPlan(format!(
                "placement `{}` does not match graph region `{}`",
                placement.region_id.0, region.id.0
            )));
        }
        if !matches!(placement.compute, ComputePlacement::Cpu) {
            return Err(AegisError::Unsupported(format!(
                "region `{}` is compute={}; CPU loader refused to load it",
                region.id.0, placement.compute
            )));
        }
        let residency = match placement.store {
            StoragePlacement::Ram => TensorResidencyPlan::RamResident,
            StoragePlacement::Mmap => TensorResidencyPlan::FileBackedMmap,
            StoragePlacement::Vram { device } => TensorResidencyPlan::StagedDeviceToHost { device },
        };

        let mut loader = TensorStorageLoader::new();
        if let Some(prefix) = nvfp4_linear_prefixes(region).into_iter().next() {
            Ok(Some(self.load_nvfp4_linear_with_layout(
                artifact,
                prefix,
                placement.store,
                residency,
                resident_layout,
                &mut loader,
            )?))
        } else {
            Ok(None)
        }
    }
}

fn unpack_nvfp4_linear(
    packed: &[u8],
    scale_bytes: &[u8],
    spec: &Nvfp4LinearSpec,
) -> Result<(Vec<i8>, Vec<f32>)> {
    if packed.len() != spec.packed_bytes || scale_bytes.len() != spec.scale_bytes {
        return Err(AegisError::InvalidPlan(format!(
            "nvfp4 linear `{}` byte mismatch while unpacking",
            spec.name
        )));
    }
    let packed_cols = spec.packed_cols();
    let scale_cols = spec.scale_cols();
    let mut weights = vec![0_i8; spec.rows * spec.cols];
    let mut scales = vec![0.0_f32; spec.rows * scale_cols];
    for row in 0..spec.rows {
        let packed_row = &packed[row * packed_cols..(row + 1) * packed_cols];
        let scale_row = &scale_bytes[row * scale_cols..(row + 1) * scale_cols];
        let dst_weight_row = &mut weights[row * spec.cols..(row + 1) * spec.cols];
        let dst_scale_row = &mut scales[row * scale_cols..(row + 1) * scale_cols];
        for block_idx in 0..scale_cols {
            dst_scale_row[block_idx] = decode_ue4m3_with_half_lut(scale_row[block_idx]);
            let packed_base = block_idx * (QK_NVFP4_SUB / 2);
            let weight_base = block_idx * QK_NVFP4_SUB;
            for j in 0..(QK_NVFP4_SUB / 2) {
                let byte = packed_row[packed_base + j];
                let lo_col = weight_base + j * 2;
                let hi_col = lo_col + 1;
                dst_weight_row[lo_col] = decode_nvfp4_nibble_i8(byte & 0x0f);
                dst_weight_row[hi_col] = decode_nvfp4_nibble_i8(byte >> 4);
            }
        }
    }
    Ok((weights, scales))
}

pub(crate) fn nvfp4_linear_prefixes(region: &GraphRegion) -> Vec<&str> {
    region
        .tensors
        .iter()
        .filter(|tensor| {
            matches!(
                tensor.role,
                TensorRole::Query
                    | TensorRole::Key
                    | TensorRole::Value
                    | TensorRole::Output
                    | TensorRole::Gate
                    | TensorRole::Up
                    | TensorRole::Down
            ) && tensor.info.dtype == TensorDType::U8
        })
        .filter_map(|tensor| tensor.info.name.strip_suffix(".weight"))
        .collect()
}

fn read_scalar_f32_with_loader(
    loader: &mut TensorStorageLoader,
    tensor: &TensorInfo,
    store: StoragePlacement,
) -> Result<f32> {
    if tensor.dtype != TensorDType::F32 || tensor.data_len_bytes() != 4 {
        return Err(AegisError::InvalidPlan(format!(
            "`{}` must be a scalar F32 tensor",
            tensor.name
        )));
    }
    let loaded = loader.load_for_store(tensor, store)?;
    Ok(f32::from_le_bytes(loaded.as_bytes().try_into().map_err(
        |_| AegisError::InvalidPlan(format!("bad scalar F32 tensor `{}`", tensor.name)),
    )?))
}

#[cfg(test)]
pub(super) fn unpack_nvfp4_for_test(
    packed: &[u8],
    scale_bytes: &[u8],
    spec: &Nvfp4LinearSpec,
) -> Result<(Vec<i8>, Vec<f32>)> {
    unpack_nvfp4_linear(packed, scale_bytes, spec)
}

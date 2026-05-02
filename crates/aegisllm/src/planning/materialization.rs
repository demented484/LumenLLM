use std::collections::HashMap;

use crate::artifact::ModelArtifact;
use crate::executor::cpu::runtime::nvfp4_linear_prefixes;
use crate::executor::cpu::{CpuNvfp4Linear, CpuRuntime};
use crate::error::{AegisError, Result};
use crate::graph::GraphRegion;
use crate::planning::placement::{ComputePlacement, RegionPlacement, StoragePlacement};
use crate::planning::runtime::KernelFamily;
use crate::tensor::layout::{LinearResidentLayout, MaterializationPolicy};
use crate::tensor::storage::{TensorResidencyPlan, TensorStorageLoader};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct LinearMaterializationKey {
    pub name: String,
    pub store: StoragePlacement,
    pub residency: TensorResidencyPlan,
    pub resident_layout: LinearResidentLayout,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LinearMaterializationStats {
    pub hits: u64,
    pub misses: u64,
    pub uncached: u64,
    pub entries: usize,
    pub materialized_extra_bytes: u64,
}

#[derive(Debug, Default)]
pub struct LinearMaterializationCache {
    cpu: CpuRuntime,
    loader: TensorStorageLoader,
    cpu_nvfp4: HashMap<LinearMaterializationKey, CpuNvfp4Linear>,
    stats: LinearMaterializationStats,
}

impl LinearMaterializationCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn stats(&self) -> LinearMaterializationStats {
        let mut stats = self.stats;
        stats.entries = self.cpu_nvfp4.len();
        stats
    }

    pub fn clear(&mut self) {
        self.cpu_nvfp4.clear();
        self.stats = LinearMaterializationStats::default();
    }

    pub fn load_cpu_nvfp4_linear(
        &mut self,
        artifact: &ModelArtifact,
        prefix: &str,
        store: StoragePlacement,
        residency: TensorResidencyPlan,
        resident_layout: LinearResidentLayout,
        materialization: MaterializationPolicy,
    ) -> Result<CpuNvfp4Linear> {
        if materialization == MaterializationPolicy::EachUse {
            self.stats.uncached += 1;
            return self.cpu.load_nvfp4_linear_with_layout(
                artifact,
                prefix,
                store,
                residency,
                resident_layout,
                &mut self.loader,
            );
        }

        let key = LinearMaterializationKey {
            name: prefix.into(),
            store,
            residency,
            resident_layout,
        };
        if let Some(linear) = self.cpu_nvfp4.get(&key) {
            self.stats.hits += 1;
            return Ok(linear.clone());
        }

        self.stats.misses += 1;
        let linear = self.cpu.load_nvfp4_linear_with_layout(
            artifact,
            prefix,
            store,
            residency,
            resident_layout,
            &mut self.loader,
        )?;
        self.stats.materialized_extra_bytes += linear.materialized_extra_bytes();
        self.cpu_nvfp4.insert(key, linear.clone());
        Ok(linear)
    }

    pub fn preload_cpu_region_nvfp4_linears(
        &mut self,
        artifact: &ModelArtifact,
        region: &GraphRegion,
        placement: &RegionPlacement,
        resident_layout: LinearResidentLayout,
        materialization: MaterializationPolicy,
    ) -> Result<Vec<CpuNvfp4Linear>> {
        let residency = cpu_residency_for_region(region, placement)?;
        let mut linears = Vec::new();
        for prefix in nvfp4_linear_prefixes(region) {
            linears.push(self.load_cpu_nvfp4_linear(
                artifact,
                prefix,
                placement.store,
                residency,
                resident_layout,
                materialization,
            )?);
        }
        Ok(linears)
    }

    pub fn load_first_cpu_region_nvfp4_linear(
        &mut self,
        artifact: &ModelArtifact,
        region: &GraphRegion,
        placement: &RegionPlacement,
        resident_layout: LinearResidentLayout,
        materialization: MaterializationPolicy,
    ) -> Result<Option<CpuNvfp4Linear>> {
        if materialization == MaterializationPolicy::OnLoad {
            return Ok(self
                .preload_cpu_region_nvfp4_linears(
                    artifact,
                    region,
                    placement,
                    resident_layout,
                    materialization,
                )?
                .into_iter()
                .next());
        }

        let residency = cpu_residency_for_region(region, placement)?;
        let Some(prefix) = nvfp4_linear_prefixes(region).into_iter().next() else {
            return Ok(None);
        };
        self.load_cpu_nvfp4_linear(
            artifact,
            prefix,
            placement.store,
            residency,
            resident_layout,
            materialization,
        )
        .map(Some)
    }
}

pub fn cuda_nvfp4_kernel_family_for_layout(
    prefix: &str,
    resident_layout: LinearResidentLayout,
) -> Result<KernelFamily> {
    match resident_layout {
        LinearResidentLayout::PackedSource => Ok(KernelFamily::CudaQuantizedReference),
        LinearResidentLayout::NativeTensorCore => Ok(KernelFamily::CudaNativeFp4TensorCores),
        LinearResidentLayout::CudaR4fE2m1Ue4m3 => Ok(KernelFamily::CudaCutlassFp4TensorCores),
        LinearResidentLayout::RepackedFp8 => Err(AegisError::Unsupported(format!(
            "CUDA NVFP4 -> FP8 materialization for `{prefix}` is planned but the repack kernel is not implemented yet"
        ))),
        LinearResidentLayout::RepackedInt4 => Err(AegisError::Unsupported(format!(
            "CUDA NVFP4 -> INT4 materialization for `{prefix}` is planned but the repack kernel is not implemented yet"
        ))),
        LinearResidentLayout::DenseTensorCore => Err(AegisError::Unsupported(format!(
            "CUDA dense tensor-core materialization is not valid for NVFP4 linear `{prefix}`"
        ))),
        LinearResidentLayout::UnpackedI8Scales => Err(AegisError::Unsupported(format!(
            "CPU unpacked-i8 materialization cannot be used by CUDA linear `{prefix}`"
        ))),
    }
}

fn cpu_residency_for_region(
    region: &GraphRegion,
    placement: &RegionPlacement,
) -> Result<TensorResidencyPlan> {
    if placement.region_id != region.id {
        return Err(AegisError::InvalidPlan(format!(
            "placement `{}` does not match graph region `{}`",
            placement.region_id.0, region.id.0
        )));
    }
    if !matches!(placement.compute, ComputePlacement::Cpu) {
        return Err(AegisError::Unsupported(format!(
            "region `{}` is compute={}; CPU materializer refused to load it",
            region.id.0, placement.compute
        )));
    }

    match placement.store {
        StoragePlacement::Ram => Ok(TensorResidencyPlan::RamResident),
        StoragePlacement::Mmap => Ok(TensorResidencyPlan::FileBackedMmap),
        StoragePlacement::Vram { device } => Ok(TensorResidencyPlan::StagedDeviceToHost { device }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_nvfp4_native_layout_selects_native_family() {
        let family =
            cuda_nvfp4_kernel_family_for_layout("x", LinearResidentLayout::NativeTensorCore)
                .unwrap();
        assert_eq!(family, KernelFamily::CudaNativeFp4TensorCores);
    }

    #[test]
    fn cuda_nvfp4_cutlass_layout_selects_cutlass_family() {
        let family =
            cuda_nvfp4_kernel_family_for_layout("x", LinearResidentLayout::CudaR4fE2m1Ue4m3)
                .unwrap();
        assert_eq!(family, KernelFamily::CudaCutlassFp4TensorCores);
    }

    #[test]
    fn cuda_nvfp4_fp8_repack_is_explicitly_pending() {
        let error = cuda_nvfp4_kernel_family_for_layout("x", LinearResidentLayout::RepackedFp8)
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("repack kernel is not implemented")
        );
    }
}

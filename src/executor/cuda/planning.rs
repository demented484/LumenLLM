use std::collections::BTreeSet;

use crate::error::{AegisError, Result};
use crate::planning::placement::{ComputePlacement, ResolvedPlacement, StoragePlacement};
use crate::planning::runtime::RuntimePlan;
use crate::tensor::layout::LinearResidentLayout;
use crate::tensor::quant::KvCacheQuantization;

pub(super) fn validate_cuda_placement(placement: &ResolvedPlacement, device: usize) -> Result<()> {
    for region in &placement.region_placements {
        match region.compute {
            ComputePlacement::Cuda {
                device: compute_device,
            } if compute_device == device => {}
            other => {
                return Err(AegisError::Unsupported(format!(
                    "generate CUDA executor cannot run region `{}` with compute={other}",
                    region.region_id.0
                )));
            }
        }
        if let StoragePlacement::Vram {
            device: store_device,
        } = region.store
            && store_device != device
        {
            return Err(AegisError::Unsupported(format!(
                "generate CUDA executor cannot read region `{}` from vram:{store_device} on cuda:{device}",
                region.region_id.0
            )));
        }
    }
    match placement.kv_cache.compute {
        ComputePlacement::Cuda {
            device: compute_device,
        } if compute_device == device => Ok(()),
        other => Err(AegisError::Unsupported(format!(
            "generate CUDA executor requires kv compute=cuda:{device}, got {other}"
        ))),
    }
}

pub(super) fn dominant_cuda_device(placement: &ResolvedPlacement) -> Option<usize> {
    let mut devices = BTreeSet::new();
    for region in &placement.region_placements {
        if let ComputePlacement::Cuda { device } = region.compute {
            devices.insert(device);
        }
        if let StoragePlacement::Vram { device } = region.store {
            devices.insert(device);
        }
    }
    if let ComputePlacement::Cuda { device } = placement.kv_cache.compute {
        devices.insert(device);
    }
    devices.into_iter().next()
}

pub(super) fn cuda_limitations(
    placement: &ResolvedPlacement,
    runtime: &RuntimePlan,
    device: usize,
) -> Vec<String> {
    let mut limitations = Vec::new();

    let foreign_compute = placement
        .region_placements
        .iter()
        .filter(|region| {
            !matches!(
                region.compute,
                ComputePlacement::Cuda {
                    device: compute_device
                } if compute_device == device
            )
        })
        .count();
    if foreign_compute > 0 {
        limitations.push(format!(
            "{foreign_compute} regions are not compute=cuda:{device}; hybrid transfer/execution nodes are not wired yet"
        ));
    }
    match placement.kv_cache.compute {
        ComputePlacement::Cuda {
            device: compute_device,
        } if compute_device == device => {}
        other => limitations.push(format!(
            "kv-cache compute={other} does not match CUDA provider cuda:{device}"
        )),
    }
    if placement.kv_cache.quantization != KvCacheQuantization::F16 {
        limitations.push(format!(
            "CUDA executor currently supports kv-cache=f16 only, got {}",
            placement.kv_cache.quantization
        ));
    }

    limitations.extend(cuda_kernel_limitations(runtime));
    limitations.sort();
    limitations.dedup();
    limitations
}

pub(in crate::executor) fn cuda_kernel_limitations(runtime: &RuntimePlan) -> Vec<String> {
    let mut limitations = Vec::new();

    if runtime.count_resident_layout(LinearResidentLayout::RepackedFp8) > 0 {
        limitations.push(
            "NVFP4 -> FP8 CUDA materialization is planned but the repack kernel is not implemented"
                .into(),
        );
    }
    if runtime.count_resident_layout(LinearResidentLayout::RepackedInt4) > 0 {
        limitations.push(
            "CUDA INT4 materialization is planned but the repack kernel is not implemented".into(),
        );
    }
    limitations.sort();
    limitations.dedup();
    limitations
}

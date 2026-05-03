use crate::engine::AegisEngine;
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::graph::TensorRole;
use aegisllm_base::planning::placement::ComputePlacement;
use aegisllm_base::tensor::layout::LinearResidentLayout;

pub(super) fn deterministic_input(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let centered = (i % 37) as f32 - 18.0;
            centered / 64.0
        })
        .collect()
}

pub(super) fn resident_layout_for_region(
    engine: &AegisEngine,
    region_id: &crate::graph::RegionId,
) -> LinearResidentLayout {
    engine
        .runtime
        .kernels
        .iter()
        .find(|kernel| kernel.name == region_id.0)
        .map(|kernel| kernel.linear_layout.resident_layout)
        .unwrap_or(LinearResidentLayout::PackedSource)
}

pub(super) fn first_cuda_nvfp4_region(
    engine: &AegisEngine,
) -> Result<(
    usize,
    &crate::graph::GraphRegion,
    &crate::planning::placement::RegionPlacement,
)> {
    let device = match engine
        .placement
        .region_placements
        .iter()
        .find_map(|region| {
            if matches!(region.compute, ComputePlacement::Cuda { .. }) {
                Some(region.compute)
            } else {
                None
            }
        }) {
        Some(ComputePlacement::Cuda { device }) => device,
        _ => {
            return Err(AegisError::InvalidPlan(
                "cuda command needs at least one cuda-computed region".into(),
            ));
        }
    };
    let region_placements = engine.placement.region_map();
    let region = engine
        .graph
        .regions
        .iter()
        .find(|region| {
            let Some(placement) = region_placements.get(&region.id) else {
                return false;
            };
            matches!(
                placement.compute,
                ComputePlacement::Cuda {
                    device: compute_device
                } if compute_device == device
            ) && region.tensors.iter().any(|tensor| {
                matches!(
                    tensor.role,
                    TensorRole::Query
                        | TensorRole::Key
                        | TensorRole::Value
                        | TensorRole::Output
                        | TensorRole::Gate
                        | TensorRole::Up
                        | TensorRole::Down
                ) && tensor.info.dtype == crate::tensor::TensorDType::U8
            })
        })
        .ok_or_else(|| {
            AegisError::InvalidPlan("no NVFP4 linear region with compute=cuda found".into())
        })?;
    let placement = region_placements.get(&region.id).ok_or_else(|| {
        AegisError::InvalidPlan(format!("missing placement for region `{}`", region.id.0))
    })?;
    Ok((device, region, placement))
}

pub(super) fn find_cuda_linear<'a>(
    linears: &'a [aegisllm_cuda::cuda::DeviceNvfp4Linear],
    suffix: &str,
) -> Result<&'a aegisllm_cuda::cuda::DeviceNvfp4Linear> {
    linears
        .iter()
        .find(|linear| linear.name.ends_with(suffix))
        .ok_or_else(|| AegisError::InvalidPlan(format!("missing CUDA linear `{suffix}`")))
}

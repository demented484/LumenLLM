use std::collections::BTreeSet;

use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::planning::placement::{ComputePlacement, ResolvedPlacement, StoragePlacement};
use aegisllm_base::planning::runtime::{KernelFamily, RuntimePlan};
use aegisllm_base::tensor::layout::LinearResidentLayout;
use aegisllm_base::tensor::quant::KvCacheQuantization;

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
    match placement.kv_cache.quantization {
        KvCacheQuantization::F16 | KvCacheQuantization::Bf16 => {}
        KvCacheQuantization::Fp8 => limitations.push(
            "kv-cache fp8: store/decode kernels + dual-buffer prefill infrastructure are \
             in place, but unscaled FP8 E4M3 produces ~10% per-element error which \
             destroys attention quality at typical Gemma-4 K/V magnitudes (smoke test \
             outputs gibberish at scale=1.0; needs per-tensor scale calibration before \
             this can be lifted). See /tmp/fp8_kv_postmortem.md.".into(),
        ),
        KvCacheQuantization::Q8_0 => limitations.push(
            "kv-cache q8_0 support is planned (Phase 3.4) but the store/load kernel is not yet implemented".into(),
        ),
        KvCacheQuantization::Nvfp4 => limitations.push(
            "kv-cache nvfp4 (Blackwell FP4) support is planned (Phase 3.5) but the store/load kernel is not yet implemented".into(),
        ),
    }

    limitations.extend(cuda_kernel_limitations(runtime));
    limitations.sort();
    limitations.dedup();
    limitations
}

pub fn cuda_kernel_limitations(runtime: &RuntimePlan) -> Vec<String> {
    let mut limitations = Vec::new();

    if runtime.count_resident_layout(LinearResidentLayout::RepackedFp8) > 0 {
        limitations.push(
            "NVFP4 -> FP8 CUDA materialization is planned but the repack kernel is not implemented"
                .into(),
        );
    }
    if runtime.count_resident_layout(LinearResidentLayout::RepackedInt4) > 0 {
        limitations.push(
            "CUDA INT4 weight-only GEMM is planned (Phase 3.6) but the kernel is not yet implemented".into(),
        );
    }
    if runtime.count_resident_layout(LinearResidentLayout::UnpackedI8Scales) > 0 {
        limitations.push(
            "CUDA INT8 weight-only GEMM is planned (Phase 3.5) but the kernel is not yet implemented".into(),
        );
    }
    if runtime.count_family(KernelFamily::CudaGatedDeltaNet) > 0 {
        limitations.push(
            "Gated DeltaNet (linear attention) kernel is planned (Phase 6) but not yet implemented".into(),
        );
    }
    if runtime.count_family(KernelFamily::CudaMambaScan) > 0 {
        limitations.push(
            "Mamba selective-scan kernel is planned (Phase 7) but not yet implemented".into(),
        );
    }
    limitations.sort();
    limitations.dedup();
    limitations
}

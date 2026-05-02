use crate::backend::{BackendDescriptor, BackendKind, BackendRegistry};
use crate::error::{AegisError, Result};
use crate::graph::{GraphRegion, GraphRegionKind, ModelGraph, TensorRole};
use crate::planning::placement::{ResolvedPlacement, StoragePlacement, TransferPolicy};
use crate::tensor::layout::{LinearLayoutPlan, LinearResidentLayout};
use crate::tensor::quant::{QuantFormat, TensorCorePrecision, WeightQuantization};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimePlan {
    pub kernels: Vec<KernelPlan>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelPlan {
    pub name: String,
    pub device: BackendKind,
    pub quant_format: QuantFormat,
    pub linear_layout: LinearLayoutPlan,
    pub family: KernelFamily,
    pub residency: TensorResidency,
    pub sync: SyncPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelFamily {
    CpuScalar,
    CpuSimd,
    CudaDenseTensorCores,
    CudaQuantizedReference,
    CudaNativeFp4TensorCores,
    CudaCutlassFp4TensorCores,
    CudaNativeFp8TensorCores,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorResidency {
    Host,
    Device,
    MappedHostToDevice,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncPolicy {
    StreamOrdered,
    ExplicitBoundary,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelRegistry {
    candidates: Vec<KernelCandidate>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelCandidate {
    pub family: KernelFamily,
    pub device_class: KernelDeviceClass,
    pub format_match: QuantFormatMatch,
    pub required_precision: Option<TensorCorePrecision>,
    pub priority: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelDeviceClass {
    Cpu,
    Cuda,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantFormatMatch {
    Exact(QuantFormat),
    Dense,
    Quantized,
}

impl RuntimePlan {
    pub fn build(
        graph: &ModelGraph,
        placement: &ResolvedPlacement,
        backends: &BackendRegistry,
    ) -> Result<Self> {
        let mut warnings = Vec::new();
        let mut kernels = Vec::new();
        let registry = KernelRegistry::default();

        let graph_regions = graph.regions_by_id();
        for region in &placement.region_placements {
            let graph_region = graph_regions
                .get(&region.region_id)
                .copied()
                .ok_or_else(|| {
                    AegisError::InvalidPlan(format!(
                        "runtime placement references unknown graph region `{}`",
                        region.region_id.0
                    ))
                })?;
            let quant_format = region_quant_format(graph_region);
            let backend_kind = BackendKind::from(region.compute);
            let backend = backends.get(backend_kind).ok_or_else(|| {
                AegisError::InvalidPlan(format!(
                    "compute backend `{:?}` was selected but no backend exists",
                    backend_kind
                ))
            })?;
            let candidate = registry.select(backend, quant_format).ok_or_else(|| {
                AegisError::InvalidPlan(format!(
                    "no kernel candidate for backend `{:?}` and quant format `{}`",
                    backend.kind, quant_format
                ))
            })?;
            let selected_family = candidate.family;
            let linear_layout = placement.linear_layout.plan(
                backend,
                quant_format,
                selected_family,
                region.weight_bytes,
            );
            let family =
                effective_family_for_layout(selected_family, linear_layout.resident_layout);
            warnings.extend(
                linear_layout
                    .notes
                    .iter()
                    .map(|note| format!("region `{}` layout note: {}", region.region_id.0, note)),
            );
            let residency = match (region.store, region.transfer) {
                (StoragePlacement::Vram { .. }, TransferPolicy::None) => TensorResidency::Device,
                (StoragePlacement::Mmap, TransferPolicy::HostToDeviceEachUse) => {
                    TensorResidency::MappedHostToDevice
                }
                _ => TensorResidency::Host,
            };
            let sync = match family {
                KernelFamily::CudaNativeFp4TensorCores
                | KernelFamily::CudaCutlassFp4TensorCores
                | KernelFamily::CudaNativeFp8TensorCores
                | KernelFamily::CudaDenseTensorCores => SyncPolicy::StreamOrdered,
                KernelFamily::CudaQuantizedReference => SyncPolicy::ExplicitBoundary,
                KernelFamily::CpuScalar | KernelFamily::CpuSimd => SyncPolicy::StreamOrdered,
            };
            kernels.push(KernelPlan {
                name: region.region_id.0.clone(),
                device: backend_kind,
                quant_format,
                linear_layout,
                family,
                residency,
                sync,
            });
        }

        let native_fp4_regions = kernels
            .iter()
            .filter(|kernel| {
                matches!(
                    kernel.family,
                    KernelFamily::CudaNativeFp4TensorCores
                        | KernelFamily::CudaCutlassFp4TensorCores
                )
            })
            .count();
        let planned_cuda_quantized_regions = kernels
            .iter()
            .filter(|kernel| matches!(kernel.device, BackendKind::Cuda { .. }))
            .filter(|kernel| kernel.quant_format.is_quantized())
            .count();
        let repacked_fp8_regions = kernels
            .iter()
            .filter(|kernel| {
                kernel.linear_layout.resident_layout == LinearResidentLayout::RepackedFp8
            })
            .count();
        if graph.weight_quantization == WeightQuantization::Nvfp4
            && planned_cuda_quantized_regions > 0
            && native_fp4_regions == 0
            && repacked_fp8_regions == 0
        {
            warnings.push("NVFP4 model has no native FP4 tensor-core regions in the plan".into());
        }
        let quantized_reference_regions = kernels
            .iter()
            .filter(|kernel| kernel.family == KernelFamily::CudaQuantizedReference)
            .count();
        if quantized_reference_regions > 0 {
            warnings.push(format!(
                "{quantized_reference_regions} CUDA quantized regions are using reference kernels; add a native candidate for the format/backend to optimize them"
            ));
        }
        if repacked_fp8_regions > 0 {
            warnings.push(format!(
                "{repacked_fp8_regions} regions are planned for FP8 materialization; CUDA repack kernels are not implemented yet"
            ));
        }
        if planned_cuda_quantized_regions > 0
            && graph.weight_quantization.format_hint().is_quantized()
            && kernels
                .iter()
                .filter(|kernel| matches!(kernel.device, BackendKind::Cuda { .. }))
                .filter(|kernel| kernel.quant_format == graph.weight_quantization.format_hint())
                .all(|kernel| {
                    !matches!(
                        kernel.family,
                        KernelFamily::CudaNativeFp4TensorCores
                            | KernelFamily::CudaCutlassFp4TensorCores
                            | KernelFamily::CudaNativeFp8TensorCores
                    )
                })
        {
            warnings.push(
                "quantized model is planned without a native tensor-core candidate for its format"
                    .into(),
            );
        }
        if placement
            .region_placements
            .iter()
            .any(|region| region.kind == GraphRegionKind::TransformerBlock)
            && placement.kv_cache.compute != placement.region_placements[0].compute
        {
            warnings.push(
                "kv-cache compute device differs from transformer compute device; runtime will need transfer nodes"
                    .into(),
            );
        }

        Ok(Self { kernels, warnings })
    }

    pub fn count_family(&self, family: KernelFamily) -> usize {
        self.kernels
            .iter()
            .filter(|kernel| kernel.family == family)
            .count()
    }

    pub fn count_format(&self, format: QuantFormat) -> usize {
        self.kernels
            .iter()
            .filter(|kernel| kernel.quant_format == format)
            .count()
    }

    pub fn count_resident_layout(&self, layout: LinearResidentLayout) -> usize {
        self.kernels
            .iter()
            .filter(|kernel| kernel.linear_layout.resident_layout == layout)
            .count()
    }

    pub fn extra_layout_weight_bytes(&self) -> u64 {
        self.kernels
            .iter()
            .map(|kernel| kernel.linear_layout.extra_weight_bytes)
            .sum()
    }
}

impl Default for KernelRegistry {
    fn default() -> Self {
        Self {
            candidates: vec![
                KernelCandidate {
                    family: KernelFamily::CudaNativeFp4TensorCores,
                    device_class: KernelDeviceClass::Cuda,
                    format_match: QuantFormatMatch::Exact(QuantFormat::Nvfp4),
                    required_precision: Some(TensorCorePrecision::Fp4),
                    priority: 100,
                },
                KernelCandidate {
                    family: KernelFamily::CudaNativeFp8TensorCores,
                    device_class: KernelDeviceClass::Cuda,
                    format_match: QuantFormatMatch::Exact(QuantFormat::Fp8E4M3Block),
                    required_precision: Some(TensorCorePrecision::Fp8),
                    priority: 100,
                },
                KernelCandidate {
                    family: KernelFamily::CudaDenseTensorCores,
                    device_class: KernelDeviceClass::Cuda,
                    format_match: QuantFormatMatch::Dense,
                    required_precision: None,
                    priority: 50,
                },
                KernelCandidate {
                    family: KernelFamily::CudaQuantizedReference,
                    device_class: KernelDeviceClass::Cuda,
                    format_match: QuantFormatMatch::Quantized,
                    required_precision: None,
                    priority: 10,
                },
                KernelCandidate {
                    family: KernelFamily::CpuSimd,
                    device_class: KernelDeviceClass::Cpu,
                    format_match: QuantFormatMatch::Dense,
                    required_precision: None,
                    priority: 50,
                },
                KernelCandidate {
                    family: KernelFamily::CpuScalar,
                    device_class: KernelDeviceClass::Cpu,
                    format_match: QuantFormatMatch::Quantized,
                    required_precision: None,
                    priority: 10,
                },
            ],
        }
    }
}

impl KernelRegistry {
    pub fn select(
        &self,
        backend: &BackendDescriptor,
        format: QuantFormat,
    ) -> Option<&KernelCandidate> {
        self.candidates
            .iter()
            .filter(|candidate| candidate.matches_backend(backend))
            .filter(|candidate| candidate.matches_format(format))
            .filter(|candidate| {
                candidate
                    .required_precision
                    .is_none_or(|precision| backend.supports_tensor_core_precision(precision))
            })
            .max_by_key(|candidate| candidate.priority)
    }
}

impl KernelCandidate {
    fn matches_backend(&self, backend: &BackendDescriptor) -> bool {
        match (self.device_class, backend.kind) {
            (KernelDeviceClass::Cpu, BackendKind::Cpu) => true,
            (KernelDeviceClass::Cuda, BackendKind::Cuda { .. }) => true,
            (KernelDeviceClass::Cpu, BackendKind::Cuda { .. })
            | (KernelDeviceClass::Cuda, BackendKind::Cpu)
            | (_, BackendKind::Wgpu { .. }) => false,
        }
    }

    fn matches_format(&self, format: QuantFormat) -> bool {
        match self.format_match {
            QuantFormatMatch::Exact(expected) => expected == format,
            QuantFormatMatch::Dense => format.is_dense(),
            QuantFormatMatch::Quantized => format.is_quantized(),
        }
    }
}

fn region_quant_format(region: &GraphRegion) -> QuantFormat {
    if region_has_nvfp4_linear(region) {
        return QuantFormat::Nvfp4;
    }
    if region_has_linear_dtype(region, crate::tensor::TensorDType::F8E4M3) {
        return QuantFormat::Fp8E4M3Block;
    }
    if region_has_linear_dtype(region, crate::tensor::TensorDType::BF16) {
        return QuantFormat::Bf16;
    }
    if region_has_linear_dtype(region, crate::tensor::TensorDType::F16) {
        return QuantFormat::F16;
    }
    if region_has_linear_dtype(region, crate::tensor::TensorDType::F32) {
        return QuantFormat::DenseF32;
    }
    QuantFormat::DenseF32
}

fn effective_family_for_layout(
    selected_family: KernelFamily,
    resident_layout: LinearResidentLayout,
) -> KernelFamily {
    match resident_layout {
        LinearResidentLayout::RepackedFp8 => KernelFamily::CudaNativeFp8TensorCores,
        LinearResidentLayout::DenseTensorCore => KernelFamily::CudaDenseTensorCores,
        LinearResidentLayout::CudaR4fE2m1Ue4m3 => KernelFamily::CudaCutlassFp4TensorCores,
        LinearResidentLayout::PackedSource
        | LinearResidentLayout::NativeTensorCore
        | LinearResidentLayout::UnpackedI8Scales
        | LinearResidentLayout::RepackedInt4 => selected_family,
    }
}

fn region_has_nvfp4_linear(region: &GraphRegion) -> bool {
    let has_packed_weight = region.tensors.iter().any(|tensor| {
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
    });
    let has_scale = region
        .tensors
        .iter()
        .any(|tensor| tensor.role == TensorRole::WeightScale);
    has_packed_weight && has_scale
}

fn region_has_linear_dtype(region: &GraphRegion, dtype: crate::tensor::TensorDType) -> bool {
    region.tensors.iter().any(|tensor| {
        matches!(
            tensor.role,
            TensorRole::Query
                | TensorRole::Key
                | TensorRole::Value
                | TensorRole::Output
                | TensorRole::Gate
                | TensorRole::Up
                | TensorRole::Down
                | TensorRole::LmHead
                | TensorRole::TokenEmbedding
        ) && tensor.info.dtype == dtype
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_prefers_native_fp4_when_backend_supports_it() {
        let backend = BackendDescriptor {
            kind: BackendKind::Cuda { device: 0 },
            label: "cuda".into(),
            supports_fp4: true,
            supports_fp8: true,
            supports_flash_attention: true,
            supports_paged_attention: true,
            attention_dtypes: vec![crate::cuda_types::CudaAttentionDType::F16],
            tensor_core_precisions: vec![TensorCorePrecision::Fp4],
        };
        let registry = KernelRegistry::default();
        let candidate = registry.select(&backend, QuantFormat::Nvfp4).unwrap();
        assert_eq!(candidate.family, KernelFamily::CudaNativeFp4TensorCores);
    }

    #[test]
    fn registry_falls_back_for_quantized_cuda_without_native_precision() {
        let backend = BackendDescriptor {
            kind: BackendKind::Cuda { device: 0 },
            label: "cuda".into(),
            supports_fp4: false,
            supports_fp8: false,
            supports_flash_attention: true,
            supports_paged_attention: true,
            attention_dtypes: vec![crate::cuda_types::CudaAttentionDType::F16],
            tensor_core_precisions: Vec::new(),
        };
        let registry = KernelRegistry::default();
        let candidate = registry.select(&backend, QuantFormat::Nvfp4).unwrap();
        assert_eq!(candidate.family, KernelFamily::CudaQuantizedReference);
    }

    #[test]
    fn fp8_repack_layout_targets_fp8_tensor_core_family() {
        assert_eq!(
            effective_family_for_layout(
                KernelFamily::CudaQuantizedReference,
                LinearResidentLayout::RepackedFp8,
            ),
            KernelFamily::CudaNativeFp8TensorCores
        );
    }

    #[test]
    fn cutlass_fp4_layout_targets_cutlass_family() {
        assert_eq!(
            effective_family_for_layout(
                KernelFamily::CudaNativeFp4TensorCores,
                LinearResidentLayout::CudaR4fE2m1Ue4m3,
            ),
            KernelFamily::CudaCutlassFp4TensorCores
        );
    }
}

pub fn cuda_nvfp4_kernel_family_for_layout(
    prefix: &str,
    resident_layout: crate::tensor::layout::LinearResidentLayout,
) -> crate::error::Result<KernelFamily> {
    use crate::tensor::layout::LinearResidentLayout;
    match resident_layout {
        LinearResidentLayout::PackedSource => Ok(KernelFamily::CudaQuantizedReference),
        LinearResidentLayout::NativeTensorCore => Ok(KernelFamily::CudaNativeFp4TensorCores),
        LinearResidentLayout::CudaR4fE2m1Ue4m3 => Ok(KernelFamily::CudaCutlassFp4TensorCores),
        LinearResidentLayout::RepackedFp8 => Err(crate::error::AegisError::Unsupported(format!(
            "CUDA NVFP4 -> FP8 materialization for `{prefix}` is planned but the repack kernel is not implemented yet"
        ))),
        LinearResidentLayout::RepackedInt4 => Err(crate::error::AegisError::Unsupported(format!(
            "CUDA NVFP4 -> INT4 materialization for `{prefix}` is planned but the repack kernel is not implemented yet"
        ))),
        LinearResidentLayout::DenseTensorCore => Err(crate::error::AegisError::Unsupported(format!(
            "CUDA dense tensor-core materialization is not valid for NVFP4 linear `{prefix}`"
        ))),
        LinearResidentLayout::UnpackedI8Scales => Err(crate::error::AegisError::Unsupported(format!(
            "CPU unpacked-i8 materialization cannot be used by CUDA linear `{prefix}`"
        ))),
    }
}

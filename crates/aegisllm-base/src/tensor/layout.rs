use std::fmt::{Display, Formatter};

use crate::backend::{BackendDescriptor, BackendKind};
use crate::error::{AegisError, Result};
use crate::planning::runtime::KernelFamily;
use crate::tensor::quant::QuantFormat;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearLayoutPolicy {
    pub cpu: LinearLayoutChoice,
    pub cuda: LinearLayoutChoice,
    pub materialization: MaterializationPolicy,
    pub max_extra_memory_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LinearLayoutChoice {
    Auto,
    Packed,
    Native,
    CutlassFp4,
    RepackFp8,
    UnpackedI8,
    RepackInt4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MaterializationPolicy {
    Lazy,
    OnLoad,
    EachUse,
    Cache,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LinearResidentLayout {
    PackedSource,
    NativeTensorCore,
    CudaR4fE2m1Ue4m3,
    DenseTensorCore,
    RepackedFp8,
    UnpackedI8Scales,
    RepackedInt4,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearLayoutPlan {
    pub source_format: QuantFormat,
    pub resident_layout: LinearResidentLayout,
    pub materialization: MaterializationPolicy,
    pub extra_weight_bytes: u64,
    pub notes: Vec<String>,
}

impl Default for LinearLayoutPolicy {
    fn default() -> Self {
        Self {
            cpu: LinearLayoutChoice::Packed,
            cuda: LinearLayoutChoice::Auto,
            materialization: MaterializationPolicy::Lazy,
            max_extra_memory_bytes: None,
        }
    }
}

impl LinearLayoutPolicy {
    pub fn plan(
        &self,
        backend: &BackendDescriptor,
        source_format: QuantFormat,
        family: KernelFamily,
        source_bytes: u64,
    ) -> LinearLayoutPlan {
        let choice = match backend.kind {
            BackendKind::Cpu | BackendKind::Wgpu { .. } => self.cpu,
            BackendKind::Cuda { .. } => self.cuda,
        };
        let (resident_layout, mut notes) =
            plan_resident_layout(choice, backend, source_format, family);
        let extra_weight_bytes = estimate_extra_bytes(source_format, resident_layout, source_bytes);
        if let Some(limit) = self.max_extra_memory_bytes
            && extra_weight_bytes > limit
        {
            notes.push(format!(
                "layout `{}` estimates extra_weight_bytes={} above max_extra_memory_bytes={}",
                resident_layout, extra_weight_bytes, limit
            ));
        }
        LinearLayoutPlan {
            source_format,
            resident_layout,
            materialization: self.materialization,
            extra_weight_bytes,
            notes,
        }
    }
}

impl LinearLayoutChoice {
    pub fn parse(value: &str) -> Result<Self> {
        match normalize(value).as_str() {
            "auto" | "native-or-repack" | "native_or_repack" => Ok(Self::Auto),
            "packed" | "source" | "source-packed" | "source_packed" => Ok(Self::Packed),
            "native" | "native-compatible" | "native_compatible" => Ok(Self::Native),
            "cutlass-fp4"
            | "cutlass_fp4"
            | "cublaslt-fp4"
            | "cublaslt_fp4"
            | "cuda-r4f-e2m1-ue4m3"
            | "cuda_r4f_e2m1_ue4m3"
            | "nvfp4-tc"
            | "nvfp4_tc" => Ok(Self::CutlassFp4),
            "fp8" | "repack-fp8" | "repack_fp8" => Ok(Self::RepackFp8),
            "unpacked-i8" | "unpacked_i8" | "i8" | "i8-scales" | "i8_scales" => {
                Ok(Self::UnpackedI8)
            }
            "int4" | "repack-int4" | "repack_int4" => Ok(Self::RepackInt4),
            _ => Err(AegisError::InvalidConfig(format!(
                "unsupported linear layout `{value}`"
            ))),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Packed => "packed",
            Self::Native => "native",
            Self::CutlassFp4 => "cutlass-fp4",
            Self::RepackFp8 => "repack-fp8",
            Self::UnpackedI8 => "unpacked-i8",
            Self::RepackInt4 => "repack-int4",
        }
    }
}

impl MaterializationPolicy {
    pub fn parse(value: &str) -> Result<Self> {
        match normalize(value).as_str() {
            "lazy" | "on-first-use" | "on_first_use" => Ok(Self::Lazy),
            "on-load" | "on_load" | "load" => Ok(Self::OnLoad),
            "each-use" | "each_use" | "stream" => Ok(Self::EachUse),
            "cache" | "cached" => Ok(Self::Cache),
            _ => Err(AegisError::InvalidConfig(format!(
                "unsupported linear materialization policy `{value}`"
            ))),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Lazy => "lazy",
            Self::OnLoad => "on-load",
            Self::EachUse => "each-use",
            Self::Cache => "cache",
        }
    }
}

impl LinearResidentLayout {
    pub fn label(self) -> &'static str {
        match self {
            Self::PackedSource => "packed-source",
            Self::NativeTensorCore => "native-tensor-core",
            Self::CudaR4fE2m1Ue4m3 => "cuda-r4f-e2m1-ue4m3",
            Self::DenseTensorCore => "dense-tensor-core",
            Self::RepackedFp8 => "repacked-fp8",
            Self::UnpackedI8Scales => "unpacked-i8-scales",
            Self::RepackedInt4 => "repacked-int4",
        }
    }
}

impl Display for LinearLayoutChoice {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

impl Display for MaterializationPolicy {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

impl Display for LinearResidentLayout {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

fn plan_resident_layout(
    choice: LinearLayoutChoice,
    backend: &BackendDescriptor,
    source_format: QuantFormat,
    family: KernelFamily,
) -> (LinearResidentLayout, Vec<String>) {
    match backend.kind {
        BackendKind::Cpu | BackendKind::Wgpu { .. } => plan_cpu_layout(choice, source_format),
        BackendKind::Cuda { .. } => plan_cuda_layout(choice, backend, source_format, family),
    }
}

fn plan_cpu_layout(
    choice: LinearLayoutChoice,
    source_format: QuantFormat,
) -> (LinearResidentLayout, Vec<String>) {
    match choice {
        LinearLayoutChoice::Auto | LinearLayoutChoice::Packed => {
            (LinearResidentLayout::PackedSource, Vec::new())
        }
        LinearLayoutChoice::UnpackedI8 if source_format == QuantFormat::Nvfp4 => {
            (LinearResidentLayout::UnpackedI8Scales, Vec::new())
        }
        LinearLayoutChoice::RepackInt4 if source_format.is_quantized() => {
            (LinearResidentLayout::RepackedInt4, Vec::new())
        }
        other => (
            LinearResidentLayout::PackedSource,
            vec![format!(
                "layout choice `{}` is not implemented for CPU format `{}`; using packed source",
                other, source_format
            )],
        ),
    }
}

fn plan_cuda_layout(
    choice: LinearLayoutChoice,
    backend: &BackendDescriptor,
    source_format: QuantFormat,
    family: KernelFamily,
) -> (LinearResidentLayout, Vec<String>) {
    match choice {
        LinearLayoutChoice::Auto => match family {
            KernelFamily::CudaNativeFp4TensorCores | KernelFamily::CudaNativeFp8TensorCores => {
                (LinearResidentLayout::NativeTensorCore, Vec::new())
            }
            KernelFamily::CudaDenseTensorCores => {
                (LinearResidentLayout::DenseTensorCore, Vec::new())
            }
            _ => (LinearResidentLayout::PackedSource, Vec::new()),
        },
        LinearLayoutChoice::Native => match family {
            KernelFamily::CudaNativeFp4TensorCores | KernelFamily::CudaNativeFp8TensorCores => {
                (LinearResidentLayout::NativeTensorCore, Vec::new())
            }
            KernelFamily::CudaDenseTensorCores => {
                (LinearResidentLayout::DenseTensorCore, Vec::new())
            }
            _ => (
                LinearResidentLayout::PackedSource,
                vec![format!(
                    "native layout requested for `{}` but no native kernel candidate was selected; using packed source",
                    source_format
                )],
            ),
        },
        LinearLayoutChoice::RepackFp8 if source_format.is_quantized() && backend.supports_fp8 => {
            (LinearResidentLayout::RepackedFp8, Vec::new())
        }
        LinearLayoutChoice::CutlassFp4
            if source_format == QuantFormat::Nvfp4 && backend.supports_fp4 =>
        {
            (LinearResidentLayout::CudaR4fE2m1Ue4m3, Vec::new())
        }
        LinearLayoutChoice::RepackInt4 if source_format.is_quantized() => {
            (LinearResidentLayout::RepackedInt4, Vec::new())
        }
        LinearLayoutChoice::Packed => (LinearResidentLayout::PackedSource, Vec::new()),
        other => (
            LinearResidentLayout::PackedSource,
            vec![format!(
                "layout choice `{}` is not implemented for CUDA format `{}`; using packed source",
                other, source_format
            )],
        ),
    }
}

fn estimate_extra_bytes(
    source_format: QuantFormat,
    resident_layout: LinearResidentLayout,
    source_bytes: u64,
) -> u64 {
    match (source_format, resident_layout) {
        (_, LinearResidentLayout::PackedSource) | (_, LinearResidentLayout::DenseTensorCore) => 0,
        // FP8 block-scaled (DeepSeek-style) weights stay packed in VRAM and are
        // dequant-on-the-fly in `aegis_fp8_block_matvec` — they never materialize
        // a tensor-core copy, so 0 extra bytes regardless of the nominal layout.
        // (Without this the planner over-counts ~source_bytes/weight as phantom
        // NativeTensorCore materialization and falsely trips the VRAM budget gate.)
        (QuantFormat::Fp8E4M3Block, _) => 0,
        (QuantFormat::Nvfp4, LinearResidentLayout::NativeTensorCore) => source_bytes,
        (QuantFormat::Nvfp4, LinearResidentLayout::CudaR4fE2m1Ue4m3) => source_bytes,
        (_, LinearResidentLayout::NativeTensorCore) => source_bytes,
        (_, LinearResidentLayout::CudaR4fE2m1Ue4m3) => source_bytes,
        (QuantFormat::Nvfp4, LinearResidentLayout::UnpackedI8Scales) => {
            source_bytes.saturating_mul(2)
        }
        (QuantFormat::Nvfp4, LinearResidentLayout::RepackedFp8) => source_bytes,
        (_, LinearResidentLayout::RepackedFp8)
        | (_, LinearResidentLayout::RepackedInt4)
        | (_, LinearResidentLayout::UnpackedI8Scales) => source_bytes,
    }
}

fn normalize(value: &str) -> String {
    value.trim().to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendKind;
    use crate::backend_types::AttentionDType;
    use crate::tensor::quant::TensorCorePrecision;

    fn cuda_backend(fp4: bool, fp8: bool) -> BackendDescriptor {
        let mut tensor_core_precisions = Vec::new();
        if fp4 {
            tensor_core_precisions.push(TensorCorePrecision::Fp4);
        }
        if fp8 {
            tensor_core_precisions.push(TensorCorePrecision::Fp8);
        }
        BackendDescriptor {
            kind: BackendKind::Cuda { device: 0 },
            label: "cuda".into(),
            ready_for_auto: true,
            supports_fp4: fp4,
            supports_fp8: fp8,
            supports_flash_attention: true,
            supports_paged_attention: true,
            attention_dtypes: vec![AttentionDType::F16],
            tensor_core_precisions,
        }
    }

    #[test]
    fn auto_cuda_layout_uses_native_for_selected_fp4_kernel() {
        let policy = LinearLayoutPolicy::default();
        let plan = policy.plan(
            &cuda_backend(true, true),
            QuantFormat::Nvfp4,
            KernelFamily::CudaNativeFp4TensorCores,
            1024,
        );
        assert_eq!(plan.resident_layout, LinearResidentLayout::NativeTensorCore);
        assert_eq!(plan.extra_weight_bytes, 1024);
    }

    #[test]
    fn explicit_cutlass_fp4_layout_uses_cuda_r4f_resident_contract() {
        let policy = LinearLayoutPolicy {
            cuda: LinearLayoutChoice::CutlassFp4,
            ..Default::default()
        };
        let plan = policy.plan(
            &cuda_backend(true, true),
            QuantFormat::Nvfp4,
            KernelFamily::CudaNativeFp4TensorCores,
            1024,
        );
        assert_eq!(plan.resident_layout, LinearResidentLayout::CudaR4fE2m1Ue4m3);
    }

    #[test]
    fn auto_cuda_reference_layout_keeps_packed_source_on_fp8_only_gpu() {
        let policy = LinearLayoutPolicy::default();
        let plan = policy.plan(
            &cuda_backend(false, true),
            QuantFormat::Nvfp4,
            KernelFamily::CudaQuantizedReference,
            1024,
        );
        assert_eq!(plan.resident_layout, LinearResidentLayout::PackedSource);
        assert_eq!(plan.extra_weight_bytes, 0);
        assert!(plan.notes.is_empty());
    }

    #[test]
    fn cpu_unpacked_i8_estimates_extra_memory() {
        let policy = LinearLayoutPolicy {
            cpu: LinearLayoutChoice::UnpackedI8,
            ..Default::default()
        };
        let backend = BackendDescriptor {
            kind: BackendKind::Cpu,
            label: "cpu".into(),
            ready_for_auto: true,
            supports_fp4: false,
            supports_fp8: false,
            supports_flash_attention: false,
            supports_paged_attention: false,
            attention_dtypes: vec![AttentionDType::F32],
            tensor_core_precisions: Vec::new(),
        };
        let plan = policy.plan(&backend, QuantFormat::Nvfp4, KernelFamily::CpuScalar, 1024);
        assert_eq!(plan.resident_layout, LinearResidentLayout::UnpackedI8Scales);
        assert_eq!(plan.extra_weight_bytes, 2048);
    }
}

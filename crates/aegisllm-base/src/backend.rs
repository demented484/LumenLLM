use std::collections::BTreeMap;

use crate::hardware::{GpuArchitecture, HardwareInventory};
use crate::planning::placement::ComputePlacement;
use crate::backend_types::AttentionDType;
use crate::tensor::quant::{QuantFormat, TensorCorePrecision};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BackendKind {
    Cpu,
    Cuda { device: usize },
    Wgpu { device: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendDescriptor {
    pub kind: BackendKind,
    pub label: String,
    /// Auto planner only selects this backend when `true`.
    /// Set to `true` once the backend passes Gates v1 for this backend×dtype pair.
    pub ready_for_auto: bool,
    pub supports_fp4: bool,
    pub supports_fp8: bool,
    pub supports_flash_attention: bool,
    pub supports_paged_attention: bool,
    pub attention_dtypes: Vec<AttentionDType>,
    pub tensor_core_precisions: Vec<TensorCorePrecision>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendRegistry {
    backends: BTreeMap<BackendKind, BackendDescriptor>,
}

impl BackendRegistry {
    pub fn from_inventory(inventory: &HardwareInventory) -> Self {
        let mut backends = BTreeMap::new();
        backends.insert(
            BackendKind::Cpu,
            BackendDescriptor {
                kind: BackendKind::Cpu,
                label: format!(
                    "cpu avx2={} avx512={} bf16={}",
                    inventory.cpu.avx2, inventory.cpu.avx512, inventory.cpu.bf16
                ),
                ready_for_auto: true,
                supports_fp4: false,
                supports_fp8: false,
                supports_flash_attention: false,
                supports_paged_attention: false,
                attention_dtypes: vec![AttentionDType::F32],
                tensor_core_precisions: Vec::new(),
            },
        );
        for gpu in &inventory.gpus {
            let tensor_core_precisions = gpu_tensor_core_precisions(gpu.architecture);
            backends.insert(
                BackendKind::Cuda { device: gpu.index },
                BackendDescriptor {
                    kind: BackendKind::Cuda { device: gpu.index },
                    label: format!("cuda:{} {} {:?}", gpu.index, gpu.name, gpu.architecture),
                    ready_for_auto: true,
                    supports_fp4: tensor_core_precisions.contains(&TensorCorePrecision::Fp4),
                    supports_fp8: tensor_core_precisions.contains(&TensorCorePrecision::Fp8),
                    supports_flash_attention: true,
                    supports_paged_attention: true,
                    attention_dtypes: cuda_attention_dtypes(gpu.architecture),
                    tensor_core_precisions,
                },
            );
        }
        let wgpu_instance = wgpu::Instance::default();
        for (idx, adapter) in wgpu_instance
            .enumerate_adapters(wgpu::Backends::PRIMARY)
            .into_iter()
            .enumerate()
        {
            let info = adapter.get_info();
            backends.insert(
                BackendKind::Wgpu { device: idx },
                BackendDescriptor {
                    kind: BackendKind::Wgpu { device: idx },
                    label: format!("wgpu:{} {} {:?}", idx, info.name, info.backend),
                    // wgpu skeleton: not runnable for inference yet
                    ready_for_auto: false,
                    supports_fp4: false,
                    supports_fp8: false,
                    supports_flash_attention: false,
                    supports_paged_attention: false,
                    attention_dtypes: vec![AttentionDType::F32],
                    tensor_core_precisions: vec![],
                },
            );
        }
        Self { backends }
    }

    pub fn get(&self, kind: BackendKind) -> Option<&BackendDescriptor> {
        self.backends.get(&kind)
    }

    pub fn contains_compute(&self, placement: ComputePlacement) -> bool {
        self.get(placement.into()).is_some()
    }

    pub fn iter(&self) -> impl Iterator<Item = &BackendDescriptor> {
        self.backends.values()
    }
}

impl BackendDescriptor {
    pub fn supports_tensor_core_precision(&self, precision: TensorCorePrecision) -> bool {
        self.tensor_core_precisions.contains(&precision)
    }

    pub fn supports_native_quant_format(&self, format: QuantFormat) -> bool {
        format
            .descriptor()
            .native_tensor_core_precision
            .is_some_and(|precision| self.supports_tensor_core_precision(precision))
    }

    pub fn supports_attention_dtype(&self, dtype: AttentionDType) -> bool {
        self.attention_dtypes.contains(&dtype)
    }
}

impl From<ComputePlacement> for BackendKind {
    fn from(value: ComputePlacement) -> Self {
        match value {
            ComputePlacement::Cpu => Self::Cpu,
            ComputePlacement::Cuda { device } => Self::Cuda { device },
            ComputePlacement::Wgpu { device } => Self::Wgpu { device },
        }
    }
}

fn gpu_tensor_core_precisions(architecture: GpuArchitecture) -> Vec<TensorCorePrecision> {
    match architecture {
        GpuArchitecture::Blackwell => vec![
            TensorCorePrecision::Tf32,
            TensorCorePrecision::F16,
            TensorCorePrecision::Bf16,
            TensorCorePrecision::Fp8,
            TensorCorePrecision::Fp4,
            TensorCorePrecision::Int8,
        ],
        GpuArchitecture::Hopper => vec![
            TensorCorePrecision::Tf32,
            TensorCorePrecision::F16,
            TensorCorePrecision::Bf16,
            TensorCorePrecision::Fp8,
            TensorCorePrecision::Int8,
        ],
        GpuArchitecture::Ada | GpuArchitecture::Ampere => vec![
            TensorCorePrecision::Tf32,
            TensorCorePrecision::F16,
            TensorCorePrecision::Bf16,
            TensorCorePrecision::Int8,
        ],
        GpuArchitecture::Unknown => vec![TensorCorePrecision::F16],
    }
}

fn cuda_attention_dtypes(architecture: GpuArchitecture) -> Vec<AttentionDType> {
    match architecture {
        GpuArchitecture::Blackwell => vec![
            AttentionDType::F32,
            AttentionDType::F16,
            AttentionDType::Bf16,
            AttentionDType::Fp8E4M3,
            AttentionDType::Fp8E5M2,
            AttentionDType::Fp4E2M1,
            AttentionDType::Int8,
            AttentionDType::Int4,
        ],
        GpuArchitecture::Hopper => vec![
            AttentionDType::F32,
            AttentionDType::F16,
            AttentionDType::Bf16,
            AttentionDType::Fp8E4M3,
            AttentionDType::Fp8E5M2,
            AttentionDType::Int8,
            AttentionDType::Int4,
        ],
        GpuArchitecture::Ada | GpuArchitecture::Ampere => vec![
            AttentionDType::F32,
            AttentionDType::F16,
            AttentionDType::Bf16,
            AttentionDType::Int8,
            AttentionDType::Int4,
        ],
        GpuArchitecture::Unknown => vec![AttentionDType::F32, AttentionDType::F16],
    }
}

#[cfg(test)]
mod tests {
    use super::{cuda_attention_dtypes, BackendDescriptor, BackendKind};
    use crate::backend_types::AttentionDType;
    use crate::hardware::GpuArchitecture;

    #[test]
    fn attention_dtype_capabilities_are_not_fp4_only() {
        let blackwell = cuda_attention_dtypes(GpuArchitecture::Blackwell);
        assert!(blackwell.contains(&AttentionDType::F16));
        assert!(blackwell.contains(&AttentionDType::Bf16));
        assert!(blackwell.contains(&AttentionDType::Fp8E4M3));
        assert!(blackwell.contains(&AttentionDType::Fp4E2M1));
        assert!(blackwell.contains(&AttentionDType::Int4));

        let ampere = cuda_attention_dtypes(GpuArchitecture::Ampere);
        assert!(ampere.contains(&AttentionDType::F16));
        assert!(ampere.contains(&AttentionDType::Bf16));
        assert!(ampere.contains(&AttentionDType::Int8));
        assert!(!ampere.contains(&AttentionDType::Fp4E2M1));
    }

    #[test]
    fn backend_descriptor_checks_attention_dtype_explicitly() {
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
        assert!(backend.supports_attention_dtype(AttentionDType::F32));
        assert!(!backend.supports_attention_dtype(AttentionDType::Fp4E2M1));
    }
}

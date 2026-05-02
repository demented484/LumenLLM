pub(crate) use super::runtime_loader::nvfp4_linear_prefixes;

use crate::planning::placement::StoragePlacement;
use crate::tensor::layout::LinearResidentLayout;
use crate::tensor::storage::{LoadedHostTensor, TensorResidencyPlan};

#[derive(Debug, Clone, Default)]
pub struct CpuRuntime;

#[derive(Debug, Clone)]
pub struct CpuNvfp4Linear {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub packed_bytes: usize,
    pub scale_bytes: usize,
    pub input_scale: f32,
    pub output_scale: f32,
    pub residency: TensorResidencyPlan,
    pub store: StoragePlacement,
    pub resident_layout: LinearResidentLayout,
    pub(crate) data: CpuNvfp4Data,
}

#[derive(Debug, Clone)]
pub(crate) enum CpuNvfp4Data {
    Packed {
        packed: LoadedHostTensor,
        scales: LoadedHostTensor,
    },
    UnpackedI8 {
        weights: Vec<i8>,
        scales: Vec<f32>,
    },
}

impl CpuRuntime {
    pub fn new() -> Self {
        Self
    }
}

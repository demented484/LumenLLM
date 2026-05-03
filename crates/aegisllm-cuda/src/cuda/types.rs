use std::ffi::c_void;

use cudarc::driver::{CudaSlice, PinnedHostSlice};

use super::repack::CutlassNvfp4LinearLayout;
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::planning::runtime::KernelFamily;
use aegisllm_base::tensor::layout::LinearResidentLayout;
use aegisllm_base::tensor::storage::TensorResidencyPlan;

#[derive(Debug)]
pub struct DeviceBuffer<T> {
    pub(super) slice: CudaSlice<T>,
}

#[derive(Debug)]
pub struct CudaAttentionSplitScratch<'a> {
    pub acc: &'a mut DeviceBuffer<f32>,
    pub m: &'a mut DeviceBuffer<f32>,
    pub l: &'a mut DeviceBuffer<f32>,
}

#[derive(Debug)]
pub struct CudaAttentionRequest<'a> {
    pub q: &'a DeviceBuffer<f32>,
    pub q_half: Option<&'a DeviceBuffer<u16>>,
    pub k_cache: &'a DeviceBuffer<u16>,
    pub v_cache: &'a DeviceBuffer<u16>,
    pub cu_q: &'a DeviceBuffer<u32>,
    pub cu_k: &'a DeviceBuffer<u32>,
    pub context_lens: &'a DeviceBuffer<u32>,
    pub slot_mapping: &'a DeviceBuffer<u32>,
    pub block_tables: &'a DeviceBuffer<u32>,
    pub split_scratch: Option<CudaAttentionSplitScratch<'a>>,
    pub output: &'a mut DeviceBuffer<f32>,
    pub num_sequences: usize,
    pub num_prefill_tokens: usize,
    pub num_decode_tokens: usize,
    pub max_q: usize,
    pub max_k: usize,
    pub block_table_stride: usize,
    pub head_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub causal: bool,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum CudaAttentionMode {
    Decode = 0,
    Prefill = 1,
    Varlen = 2,
    Mixed = 3,
}

pub use aegisllm_base::cuda_types::CudaAttentionDType;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct CudaAttentionParamsV1 {
    pub abi_version: u32,
    pub mode: CudaAttentionMode,
    pub flags: u32,
    pub num_sequences: u32,
    pub num_prefill_tokens: u32,
    pub num_decode_tokens: u32,
    pub max_q: u32,
    pub max_k: u32,
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub page_tokens: u32,
    pub block_table_stride: u32,
    pub physical_slots: u32,
    pub softmax_scale: f32,
    pub q_dtype: CudaAttentionDType,
    pub k_dtype: CudaAttentionDType,
    pub v_dtype: CudaAttentionDType,
    pub output_dtype: CudaAttentionDType,
    pub accum_dtype: CudaAttentionDType,
    pub reserved0: u32,
    pub reserved1: u32,
    pub reserved2: u32,
    pub q: *const c_void,
    pub k_cache: *const c_void,
    pub v_cache: *const c_void,
    pub output: *mut c_void,
    pub cu_q: *const u32,
    pub cu_k: *const u32,
    pub context_lens: *const u32,
    pub slot_mapping: *const u32,
    pub block_tables: *const u32,
}

#[allow(dead_code)]
impl CudaAttentionParamsV1 {
    pub const ABI_VERSION: u32 = 1;
    pub const FLAG_CAUSAL: u32 = 1 << 0;
    pub const FLAG_PAGED_KV: u32 = 1 << 1;
    pub const FLAG_GQA: u32 = 1 << 2;
}

/// Weight bytes kept in host RAM for a heterogeneous (StagedHostToDevice) linear.
/// The `packed` and `scales` fields of the parent `DeviceNvfp4Linear` are 1-byte
/// placeholders when this is `Some`; actual data lives here in CUDA-pinned host RAM.
///
/// Pinned (page-locked) memory enables:
///   1. True async DMA H2D copies (no internal driver staging buffer).
///   2. ~3-5× higher PCIe transfer throughput vs pageable Vec<u8>.
/// Each `PinnedHostSlice<u8>` is sized exactly to its content; staging hands the
/// whole slice to `memcpy_htod` so cudarc preserves the pinned semantics
/// (slicing into a pinned buffer demotes it to pageable in the safe API).
#[derive(Debug)]
pub(super) struct HostResidentWeights {
    pub packed: PinnedHostSlice<u8>,
    pub scales: PinnedHostSlice<u8>,
    /// Native MXFP4 repacked layout, if available (requires native_mxfp4_repack=true).
    /// When present, inference stages this into VRAM and uses tensor-core kernels.
    pub native_mxfp4: Option<HostResidentMxfp4>,
}

#[derive(Debug)]
pub(super) struct HostResidentMxfp4 {
    pub data: PinnedHostSlice<u8>,
    pub blocks_per_row: usize,
}

#[derive(Debug)]
pub struct DeviceNvfp4Linear {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub packed_bytes: usize,
    pub scale_bytes: usize,
    pub input_scale: f32,
    pub output_scale: f32,
    pub kernel_family: KernelFamily,
    pub resident_layout: LinearResidentLayout,
    pub residency: TensorResidencyPlan,
    pub(super) packed: CudaSlice<u8>,
    pub(super) scales: CudaSlice<u8>,
    pub(super) native_mxfp4: Option<DeviceMxfp4Linear>,
    pub(super) cutlass_nvfp4: Option<DeviceCutlassNvfp4Linear>,
    /// Non-None for `StagedHostToDevice` layers: weights live in host RAM.
    /// When set, `packed`/`scales` above are 1-byte stubs (no real VRAM).
    pub(super) host_weights: Option<Box<HostResidentWeights>>,
}

impl DeviceNvfp4Linear {
    /// Returns `true` if this linear's weights live in host RAM (StagedHostToDevice residency).
    /// In that case a staging VRAM pool must be used for H2D transfer at inference time.
    pub fn is_host_resident(&self) -> bool {
        self.host_weights.is_some()
    }
}

#[derive(Debug)]
pub(super) struct DeviceMxfp4Linear {
    pub bytes: usize,
    pub blocks_per_row: usize,
    pub data: CudaSlice<u8>,
}

#[derive(Debug)]
pub(super) struct DeviceCutlassNvfp4Linear {
    pub layout: CutlassNvfp4LinearLayout,
    pub payload_e2m1: CudaSlice<u8>,
    pub scales_ue4m3: CudaSlice<u8>,
}

/// BF16 matrix stored in CUDA-pinned host RAM for `StagedHostToDevice` residency.
/// Rows are streamed to a small VRAM scratch buffer for embedding lookup, or the
/// whole matrix is staged for matvec (used by lm_head).
#[derive(Debug)]
pub(super) struct HostBf16Weights {
    pub values: cudarc::driver::PinnedHostSlice<u16>,
}

#[derive(Debug)]
pub struct DeviceBf16Matrix {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub residency: TensorResidencyPlan,
    /// Tiny VRAM stub when host-resident; full matrix when VRAM-resident.
    pub(super) values: CudaSlice<u16>,
    /// Set for `StagedHostToDevice` BF16 matrices (e.g. embed_tokens, lm_head when
    /// `model.store=ram`). When present, callers must use the staging-aware paths.
    pub(super) host_values: Option<Box<HostBf16Weights>>,
}

impl DeviceBf16Matrix {
    /// Host-resident BF16 matrices live in pinned host RAM and stage to VRAM per use.
    pub fn is_host_resident(&self) -> bool {
        self.host_values.is_some()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DeviceRopeConfig {
    pub theta: f32,
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: u32,
    /// 0 = full head_dim (standard RoPE); >0 = first N dims get RoPE (Gemma 4 p-RoPE).
    pub partial_dim: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DensePrefillMetadataProof {
    start_position: usize,
    batch: usize,
    context_len: usize,
}

impl DensePrefillMetadataProof {
    pub fn new_identity(
        start_position: usize,
        batch: usize,
        context_size: usize,
        positions: &[u32],
        slot_mapping: &[u32],
        cu_q: &[u32],
        context_lens: &[u32],
    ) -> Result<Self> {
        if batch == 0 {
            return Err(AegisError::InvalidPlan(
                "dense prefill metadata proof requires a non-empty batch".into(),
            ));
        }
        let expected_context_len = start_position.checked_add(batch).ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "dense prefill metadata proof overflows: start={} batch={}",
                start_position, batch
            ))
        })?;
        if batch > u32::MAX as usize
            || expected_context_len > u32::MAX as usize
            || context_size > u32::MAX as usize
        {
            return Err(AegisError::InvalidPlan(format!(
                "dense prefill metadata proof requires u32 metadata: batch={} context_len={} context_size={}",
                batch, expected_context_len, context_size
            )));
        }
        if positions.len() != batch
            || slot_mapping.len() != batch
            || cu_q != [0u32, batch as u32]
            || context_lens != [expected_context_len as u32]
        {
            return Err(AegisError::InvalidPlan(format!(
                "dense prefill metadata proof requires identity metadata: positions={} slots={} cu_q={:?} context_lens={:?} start={} batch={} expected_context_len={}",
                positions.len(),
                slot_mapping.len(),
                cu_q,
                context_lens,
                start_position,
                batch,
                expected_context_len
            )));
        }
        let mut previous_slot = None;
        for idx in 0..batch {
            let expected = start_position.checked_add(idx).ok_or_else(|| {
                AegisError::InvalidPlan(format!(
                    "dense prefill metadata proof position overflow: start={} idx={}",
                    start_position, idx
                ))
            })?;
            if expected > u32::MAX as usize
                || positions[idx] as usize != expected
                || slot_mapping[idx] as usize != expected
            {
                return Err(AegisError::InvalidPlan(format!(
                    "dense prefill metadata proof found non-identity entry: idx={} position={} slot={} expected={}",
                    idx, positions[idx], slot_mapping[idx], expected
                )));
            }
            if previous_slot.is_some_and(|slot| slot >= slot_mapping[idx]) {
                return Err(AegisError::InvalidPlan(format!(
                    "dense prefill metadata proof requires strictly increasing slots: prev={:?} current={}",
                    previous_slot, slot_mapping[idx]
                )));
            }
            previous_slot = Some(slot_mapping[idx]);
        }
        if expected_context_len > context_size {
            return Err(AegisError::InvalidPlan(format!(
                "dense prefill metadata proof exceeds context: start={} batch={} context_len={} context_size={}",
                start_position, batch, expected_context_len, context_size
            )));
        }
        Ok(Self {
            start_position,
            batch,
            context_len: expected_context_len,
        })
    }

    pub fn start_position(self) -> usize {
        self.start_position
    }

    pub fn batch(self) -> usize {
        self.batch
    }

    pub fn context_len(self) -> usize {
        self.context_len
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CudaAttentionDType, CudaAttentionMode, CudaAttentionParamsV1, DensePrefillMetadataProof,
    };

    #[test]
    fn attention_params_are_c_stable_enough_for_cuda_ffi() {
        assert_eq!(CudaAttentionParamsV1::ABI_VERSION, 1);
        assert_eq!(CudaAttentionMode::Decode as u32, 0);
        assert_eq!(CudaAttentionMode::Prefill as u32, 1);
        assert_eq!(CudaAttentionMode::Varlen as u32, 2);
        assert_eq!(CudaAttentionMode::Mixed as u32, 3);
        assert_eq!(CudaAttentionDType::F32 as u32, 0);
        assert_eq!(CudaAttentionDType::F16 as u32, 1);
        assert_eq!(CudaAttentionDType::Bf16 as u32, 2);
        assert_eq!(CudaAttentionDType::Fp8E4M3 as u32, 3);
        assert_eq!(CudaAttentionDType::Fp8E5M2 as u32, 4);
        assert_eq!(CudaAttentionDType::Fp4E2M1 as u32, 5);
        assert_eq!(CudaAttentionDType::Int8 as u32, 6);
        assert_eq!(CudaAttentionDType::Int4 as u32, 7);
        assert_eq!(CudaAttentionParamsV1::FLAG_CAUSAL, 1);
        assert_eq!(CudaAttentionParamsV1::FLAG_PAGED_KV, 2);
        assert_eq!(CudaAttentionParamsV1::FLAG_GQA, 4);
        assert_eq!(
            std::mem::size_of::<CudaAttentionParamsV1>() % std::mem::size_of::<usize>(),
            0
        );
    }

    #[test]
    fn dense_prefill_metadata_proof_accepts_identity_span() {
        let proof = DensePrefillMetadataProof::new_identity(
            5,
            3,
            16,
            &[5, 6, 7],
            &[5, 6, 7],
            &[0, 3],
            &[8],
        )
        .unwrap();
        assert_eq!(proof.start_position(), 5);
        assert_eq!(proof.batch(), 3);
        assert_eq!(proof.context_len(), 8);
    }

    #[test]
    fn dense_prefill_metadata_proof_rejects_non_identity_slot_mapping() {
        assert!(
            DensePrefillMetadataProof::new_identity(
                5,
                3,
                16,
                &[5, 6, 7],
                &[5, 7, 6],
                &[0, 3],
                &[8],
            )
            .is_err()
        );
    }

    #[test]
    fn dense_prefill_metadata_proof_rejects_bad_cu_q() {
        assert!(
            DensePrefillMetadataProof::new_identity(
                5,
                3,
                16,
                &[5, 6, 7],
                &[5, 6, 7],
                &[0, 2],
                &[8],
            )
            .is_err()
        );
    }
}

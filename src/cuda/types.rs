use std::ffi::c_void;

use cudarc::driver::CudaSlice;

use super::repack::CutlassNvfp4LinearLayout;
use crate::error::{AegisError, Result};
use crate::planning::runtime::KernelFamily;
use crate::tensor::layout::LinearResidentLayout;
use crate::tensor::storage::TensorResidencyPlan;

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
pub enum CudaSdpaMode {
    Decode = 0,
    Prefill = 1,
    Varlen = 2,
    Mixed = 3,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct CudaSdpaParamsV1 {
    pub abi_version: u32,
    pub mode: CudaSdpaMode,
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
    pub reserved0: u32,
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
impl CudaSdpaParamsV1 {
    pub const ABI_VERSION: u32 = 1;
    pub const FLAG_CAUSAL: u32 = 1 << 0;
    pub const FLAG_PAGED_KV: u32 = 1 << 1;
    pub const FLAG_GQA: u32 = 1 << 2;
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

#[derive(Debug)]
pub struct DeviceBf16Matrix {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub residency: TensorResidencyPlan,
    pub(super) values: CudaSlice<u16>,
}

#[derive(Debug, Clone, Copy)]
pub struct DeviceRopeConfig {
    pub theta: f32,
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: u32,
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
    use super::{CudaSdpaMode, CudaSdpaParamsV1, DensePrefillMetadataProof};

    #[test]
    fn sdpa_params_are_c_stable_enough_for_cuda_ffi() {
        assert_eq!(CudaSdpaParamsV1::ABI_VERSION, 1);
        assert_eq!(CudaSdpaMode::Decode as u32, 0);
        assert_eq!(CudaSdpaMode::Prefill as u32, 1);
        assert_eq!(CudaSdpaMode::Varlen as u32, 2);
        assert_eq!(CudaSdpaMode::Mixed as u32, 3);
        assert_eq!(CudaSdpaParamsV1::FLAG_CAUSAL, 1);
        assert_eq!(CudaSdpaParamsV1::FLAG_PAGED_KV, 2);
        assert_eq!(CudaSdpaParamsV1::FLAG_GQA, 4);
        assert_eq!(
            std::mem::size_of::<CudaSdpaParamsV1>() % std::mem::size_of::<usize>(),
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

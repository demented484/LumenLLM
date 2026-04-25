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
    use super::DensePrefillMetadataProof;

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

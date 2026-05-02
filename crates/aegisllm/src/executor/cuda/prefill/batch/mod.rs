use crate::cuda::{CudaRuntime, DensePrefillMetadataProof};
use crate::error::{AegisError, Result};
use crate::executor::cuda::state::CudaPrefillScratch;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub(super) enum CudaPrefillBatchKind {
    FirstPrefill,
    ContinuationPrefill,
    Decode,
    Mixed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CudaPrefillBatch {
    pub(super) start_position: usize,
    pub(super) num_sequences: usize,
    pub(super) num_prefill_tokens: usize,
    pub(super) num_decode_tokens: usize,
    pub(super) max_q: usize,
    pub(super) max_k: usize,
    pub(super) kind: CudaPrefillBatchKind,
    pub(super) dense_metadata: DensePrefillMetadataProof,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct HostPrefillBatchDescriptor {
    pub(super) request_ids: Vec<u32>,
    pub(super) seq_ids: Vec<u32>,
    pub(super) token_ids: Vec<u32>,
    pub(super) positions: Vec<u32>,
    pub(super) slot_mapping: Vec<u32>,
    pub(super) cu_q: Vec<u32>,
    pub(super) cu_k: Vec<u32>,
    pub(super) context_lens: Vec<u32>,
    pub(super) block_tables: Vec<u32>,
    pub(super) max_q: usize,
    pub(super) max_k: usize,
    pub(super) num_prefill_tokens: usize,
    pub(super) num_decode_tokens: usize,
    pub(super) kind: CudaPrefillBatchKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct HostPrefillSequenceDescriptor {
    pub(super) request_id: u32,
    pub(super) seq_id: u32,
    pub(super) token_ids: Vec<u32>,
    pub(super) start_position: usize,
    pub(super) context_len: usize,
    pub(super) block_table: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct HostDecodeSequenceDescriptor {
    pub(super) request_id: u32,
    pub(super) seq_id: u32,
    pub(super) token_id: u32,
    pub(super) position: usize,
    pub(super) context_len: usize,
    pub(super) physical_slot: u32,
    pub(super) block_table: Vec<u32>,
}

impl CudaPrefillScratch {
    pub(super) fn prepare_dense_batch(
        &mut self,
        runtime: &CudaRuntime,
        tokens: &[usize],
        start_position: usize,
        context_size: usize,
        vocab_size: usize,
    ) -> Result<CudaPrefillBatch> {
        let host = HostPrefillBatchDescriptor::dense_single_sequence(
            tokens,
            start_position,
            context_size,
            vocab_size,
        )?;
        self.validate_capacity(&host)?;
        self.copy_host_descriptor(&host);
        let dense_metadata = DensePrefillMetadataProof::new_identity(
            start_position,
            host.num_prefill_tokens,
            context_size,
            &self.position_host,
            &self.slot_mapping_host,
            &self.cu_q_host,
            &self.context_lens_host,
        )?;

        runtime.copy_u32_to_device(&self.token_host, &mut self.tokens)?;
        runtime.build_dense_prefill_metadata_device(
            start_position,
            host.num_prefill_tokens,
            &mut self.positions,
            &mut self.slot_mapping,
        )?;
        runtime.copy_u32_to_device(&self.cu_q_host, &mut self.cu_q)?;
        runtime.copy_u32_to_device(&self.cu_k_host, &mut self.cu_k)?;
        runtime.copy_u32_to_device(&self.context_lens_host, &mut self.context_lens)?;
        runtime.copy_u32_to_device(&self.block_tables_host, &mut self.block_tables)?;

        Ok(CudaPrefillBatch {
            start_position,
            num_sequences: 1,
            num_prefill_tokens: host.num_prefill_tokens,
            num_decode_tokens: host.num_decode_tokens,
            max_q: host.max_q,
            max_k: host.max_k,
            kind: host.kind,
            dense_metadata,
        })
    }

    fn validate_capacity(&self, host: &HostPrefillBatchDescriptor) -> Result<()> {
        let total_query_tokens = host.num_prefill_tokens + host.num_decode_tokens;
        if total_query_tokens == 0 || total_query_tokens > self.chunk_size {
            return Err(AegisError::InvalidPlan(format!(
                "bad CUDA prefill batch size: prefill_tokens={} decode_tokens={} chunk_size={}",
                host.num_prefill_tokens, host.num_decode_tokens, self.chunk_size
            )));
        }
        if host.request_ids.len() > self.max_sequences
            || host.seq_ids.len() > self.max_sequences
            || host.cu_q.len() > self.max_sequences + 1
            || host.cu_k.len() > self.max_sequences + 1
            || host.context_lens.len() > self.max_sequences
            || host.block_tables.len() > self.block_table_capacity
        {
            return Err(AegisError::InvalidPlan(format!(
                "CUDA prefill metadata capacity exceeded: seqs={} max_seqs={} blocks={} block_capacity={}",
                host.request_ids.len(),
                self.max_sequences,
                host.block_tables.len(),
                self.block_table_capacity
            )));
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub(super) fn upload_varlen_descriptor(
        &mut self,
        runtime: &CudaRuntime,
        host: &HostPrefillBatchDescriptor,
    ) -> Result<()> {
        self.validate_capacity(host)?;
        self.copy_host_descriptor(host);
        runtime.copy_u32_to_device(&self.request_ids_host, &mut self.request_ids)?;
        runtime.copy_u32_to_device(&self.seq_ids_host, &mut self.seq_ids)?;
        runtime.copy_u32_to_device(&self.token_host, &mut self.tokens)?;
        runtime.copy_u32_to_device(&self.position_host, &mut self.positions)?;
        runtime.copy_u32_to_device(&self.slot_mapping_host, &mut self.slot_mapping)?;
        runtime.copy_u32_to_device(&self.cu_q_host, &mut self.cu_q)?;
        runtime.copy_u32_to_device(&self.cu_k_host, &mut self.cu_k)?;
        runtime.copy_u32_to_device(&self.context_lens_host, &mut self.context_lens)?;
        runtime.copy_u32_to_device(&self.block_tables_host, &mut self.block_tables)?;
        Ok(())
    }

    fn copy_host_descriptor(&mut self, host: &HostPrefillBatchDescriptor) {
        self.request_ids_host.clear();
        self.seq_ids_host.clear();
        self.token_host.clear();
        self.position_host.clear();
        self.slot_mapping_host.clear();
        self.cu_q_host.clear();
        self.cu_k_host.clear();
        self.context_lens_host.clear();
        self.block_tables_host.clear();

        self.request_ids_host.extend_from_slice(&host.request_ids);
        self.seq_ids_host.extend_from_slice(&host.seq_ids);
        self.token_host.extend_from_slice(&host.token_ids);
        self.position_host.extend_from_slice(&host.positions);
        self.slot_mapping_host.extend_from_slice(&host.slot_mapping);
        self.cu_q_host.extend_from_slice(&host.cu_q);
        self.cu_k_host.extend_from_slice(&host.cu_k);
        self.context_lens_host.extend_from_slice(&host.context_lens);
        self.block_tables_host.extend_from_slice(&host.block_tables);
    }
}

mod builder;

#[cfg(test)]
mod tests {
    use super::builder::validate_dense_prefill_tokens;
    use super::{
        CudaPrefillBatchKind, HostDecodeSequenceDescriptor, HostPrefillBatchDescriptor,
        HostPrefillSequenceDescriptor,
    };

    #[test]
    fn dense_prefill_token_validation_rejects_oob_vocab_id() {
        assert!(validate_dense_prefill_tokens(&[0, 4], 4).is_err());
    }

    #[test]
    fn dense_prefill_token_validation_accepts_vocab_ids() {
        validate_dense_prefill_tokens(&[0, 3], 4).unwrap();
    }

    #[test]
    fn dense_descriptor_exposes_varlen_prefill_fields() {
        let descriptor =
            HostPrefillBatchDescriptor::dense_single_sequence(&[1, 2, 3], 5, 32, 16).unwrap();
        assert_eq!(descriptor.request_ids, [0]);
        assert_eq!(descriptor.seq_ids, [0]);
        assert_eq!(descriptor.positions, [5, 6, 7]);
        assert_eq!(descriptor.slot_mapping, [5, 6, 7]);
        assert_eq!(descriptor.cu_q, [0, 3]);
        assert_eq!(descriptor.cu_k, [0, 8]);
        assert_eq!(descriptor.context_lens, [8]);
        assert_eq!(descriptor.max_q, 3);
        assert_eq!(descriptor.max_k, 8);
        assert_eq!(descriptor.num_prefill_tokens, 3);
        assert_eq!(descriptor.num_decode_tokens, 0);
        assert_eq!(descriptor.kind, CudaPrefillBatchKind::ContinuationPrefill);
        assert_eq!(descriptor.block_tables, [0]);
    }

    #[test]
    fn dense_descriptor_marks_first_prefill() {
        let descriptor =
            HostPrefillBatchDescriptor::dense_single_sequence(&[1, 2], 0, 32, 16).unwrap();
        assert_eq!(descriptor.kind, CudaPrefillBatchKind::FirstPrefill);
    }

    #[test]
    fn paged_multi_sequence_descriptor_maps_physical_slots() {
        let descriptor = HostPrefillBatchDescriptor::paged_multi_sequence(
            &[
                HostPrefillSequenceDescriptor {
                    request_id: 7,
                    seq_id: 70,
                    token_ids: vec![1, 2],
                    start_position: 0,
                    context_len: 2,
                    block_table: vec![5],
                },
                HostPrefillSequenceDescriptor {
                    request_id: 8,
                    seq_id: 80,
                    token_ids: vec![3, 4, 5],
                    start_position: 2,
                    context_len: 5,
                    block_table: vec![9, 2],
                },
            ],
            4,
            16,
        )
        .unwrap();
        assert_eq!(descriptor.request_ids, [7, 8]);
        assert_eq!(descriptor.seq_ids, [70, 80]);
        assert_eq!(descriptor.cu_q, [0, 2, 5]);
        assert_eq!(descriptor.cu_k, [0, 2, 7]);
        assert_eq!(descriptor.context_lens, [2, 5]);
        assert_eq!(descriptor.slot_mapping, [20, 21, 38, 39, 8]);
        assert_eq!(descriptor.max_q, 3);
        assert_eq!(descriptor.max_k, 5);
        assert_eq!(descriptor.num_prefill_tokens, 5);
    }

    #[test]
    fn mixed_descriptor_keeps_prefill_and_decode_counts_explicit() {
        let descriptor = HostPrefillBatchDescriptor::mixed_paged(
            &[HostPrefillSequenceDescriptor {
                request_id: 1,
                seq_id: 10,
                token_ids: vec![6, 7],
                start_position: 4,
                context_len: 6,
                block_table: vec![1, 3],
            }],
            &[HostDecodeSequenceDescriptor {
                request_id: 2,
                seq_id: 20,
                token_id: 8,
                position: 9,
                context_len: 10,
                physical_slot: 42,
                block_table: vec![4, 5, 6],
            }],
            4,
            16,
        )
        .unwrap();
        assert_eq!(descriptor.kind, CudaPrefillBatchKind::Mixed);
        assert_eq!(descriptor.num_prefill_tokens, 2);
        assert_eq!(descriptor.num_decode_tokens, 1);
        assert_eq!(descriptor.cu_q, [0, 2, 3]);
        assert_eq!(descriptor.cu_k, [0, 6, 16]);
        assert_eq!(descriptor.slot_mapping, [12, 13, 42]);
        assert_eq!(descriptor.context_lens, [6, 10]);
    }

    #[test]
    fn paged_descriptor_rejects_context_without_pages() {
        let error = HostPrefillBatchDescriptor::paged_multi_sequence(
            &[HostPrefillSequenceDescriptor {
                request_id: 9,
                seq_id: 90,
                token_ids: vec![1],
                start_position: 0,
                context_len: 9,
                block_table: vec![2],
            }],
            4,
            16,
        )
        .unwrap_err();
        assert!(error.to_string().contains("block table stride too small"));
    }
}

use super::{
    CudaPrefillBatchKind, HostDecodeSequenceDescriptor, HostPrefillBatchDescriptor,
    HostPrefillSequenceDescriptor,
};
use aegisllm_base::error::{AegisError, Result};
use crate::executor::prefill::kv::PREFILL_KV_PAGE_TOKENS;

impl HostPrefillBatchDescriptor {
    pub(in crate::executor::prefill) fn dense_single_sequence(
        tokens: &[usize],
        start_position: usize,
        context_size: usize,
        vocab_size: usize,
    ) -> Result<Self> {
        validate_dense_prefill_tokens(tokens, vocab_size)?;
        let end_position = start_position.checked_add(tokens.len()).ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "prefill batch position overflow: start={} batch={}",
                start_position,
                tokens.len()
            ))
        })?;
        if end_position > context_size {
            return Err(AegisError::InvalidPlan(format!(
                "prefill batch exceeds context: start={} batch={} context={}",
                start_position,
                tokens.len(),
                context_size
            )));
        }
        if end_position > u32::MAX as usize || context_size > u32::MAX as usize {
            return Err(AegisError::InvalidPlan(format!(
                "CUDA prefill dense adapter requires u32 positions: end={} context={}",
                end_position, context_size
            )));
        }
        let kind = if start_position == 0 {
            CudaPrefillBatchKind::FirstPrefill
        } else {
            CudaPrefillBatchKind::ContinuationPrefill
        };
        let page_count = end_position.div_ceil(PREFILL_KV_PAGE_TOKENS).max(1);
        let out = Self {
            request_ids: vec![0],
            seq_ids: vec![0],
            token_ids: tokens.iter().map(|&token| token as u32).collect(),
            positions: (0..tokens.len())
                .map(|idx| (start_position + idx) as u32)
                .collect(),
            slot_mapping: (0..tokens.len())
                .map(|idx| (start_position + idx) as u32)
                .collect(),
            cu_q: vec![0, tokens.len() as u32],
            cu_k: vec![0, end_position as u32],
            context_lens: vec![end_position as u32],
            block_tables: (0..page_count).map(|block| block as u32).collect(),
            max_q: tokens.len(),
            max_k: end_position,
            num_prefill_tokens: tokens.len(),
            num_decode_tokens: 0,
            kind,
        };
        out.validate_varlen_contract(PREFILL_KV_PAGE_TOKENS)?;
        Ok(out)
    }

    #[allow(dead_code)]
    pub(super) fn paged_multi_sequence(
        sequences: &[HostPrefillSequenceDescriptor],
        page_tokens: usize,
        vocab_size: usize,
    ) -> Result<Self> {
        validate_page_tokens(page_tokens)?;
        if sequences.is_empty() {
            return Err(AegisError::InvalidPlan(
                "paged prefill descriptor requires at least one sequence".into(),
            ));
        }
        let max_blocks = sequences
            .iter()
            .map(|sequence| sequence.block_table.len())
            .max()
            .unwrap_or(1)
            .max(1);
        let mut out = Self::empty(CudaPrefillBatchKind::FirstPrefill);
        out.cu_q.push(0);
        out.cu_k.push(0);
        for sequence in sequences {
            validate_dense_prefill_tokens_usize(
                sequence
                    .token_ids
                    .iter()
                    .copied()
                    .map(|token| token as usize),
                vocab_size,
            )?;
            out.request_ids.push(sequence.request_id);
            out.seq_ids.push(sequence.seq_id);
            out.context_lens.push(sequence.context_len as u32);
            out.max_q = out.max_q.max(sequence.token_ids.len());
            out.max_k = out.max_k.max(sequence.context_len);
            for (idx, &token) in sequence.token_ids.iter().enumerate() {
                let logical_position = sequence.start_position + idx;
                out.token_ids.push(token);
                out.positions.push(logical_position as u32);
                out.slot_mapping.push(physical_slot_from_table(
                    &sequence.block_table,
                    page_tokens,
                    logical_position,
                )?);
            }
            let next_q = out
                .cu_q
                .last()
                .copied()
                .unwrap_or(0)
                .checked_add(sequence.token_ids.len() as u32)
                .ok_or_else(|| AegisError::InvalidPlan("cu_q overflow".into()))?;
            out.cu_q.push(next_q);
            push_cu_k(&mut out, sequence.context_len)?;
            out.block_tables
                .extend(padded_block_table(&sequence.block_table, max_blocks));
        }
        out.num_prefill_tokens = out.token_ids.len();
        out.kind = if sequences
            .iter()
            .all(|sequence| sequence.start_position == 0)
        {
            CudaPrefillBatchKind::FirstPrefill
        } else {
            CudaPrefillBatchKind::ContinuationPrefill
        };
        out.validate_varlen_contract(page_tokens)?;
        Ok(out)
    }

    #[allow(dead_code)]
    pub(super) fn mixed_paged(
        prefill: &[HostPrefillSequenceDescriptor],
        decode: &[HostDecodeSequenceDescriptor],
        page_tokens: usize,
        vocab_size: usize,
    ) -> Result<Self> {
        validate_page_tokens(page_tokens)?;
        if prefill.is_empty() && decode.is_empty() {
            return Err(AegisError::InvalidPlan(
                "mixed descriptor requires at least one query row".into(),
            ));
        }
        let max_blocks = prefill
            .iter()
            .map(|sequence| sequence.block_table.len())
            .chain(decode.iter().map(|sequence| sequence.block_table.len()))
            .max()
            .unwrap_or(1)
            .max(1);
        let mut out = Self::empty(if prefill.is_empty() {
            CudaPrefillBatchKind::Decode
        } else if decode.is_empty() {
            CudaPrefillBatchKind::ContinuationPrefill
        } else {
            CudaPrefillBatchKind::Mixed
        });
        out.cu_q.push(0);
        out.cu_k.push(0);
        for sequence in prefill {
            validate_dense_prefill_tokens_usize(
                sequence
                    .token_ids
                    .iter()
                    .copied()
                    .map(|token| token as usize),
                vocab_size,
            )?;
            push_prefill_sequence(&mut out, sequence, page_tokens, max_blocks)?;
        }
        out.num_prefill_tokens = out.token_ids.len();
        for sequence in decode {
            validate_dense_prefill_tokens_usize([sequence.token_id as usize], vocab_size)?;
            out.request_ids.push(sequence.request_id);
            out.seq_ids.push(sequence.seq_id);
            out.token_ids.push(sequence.token_id);
            out.positions.push(sequence.position as u32);
            out.slot_mapping.push(sequence.physical_slot);
            out.context_lens.push(sequence.context_len as u32);
            out.max_q = out.max_q.max(1);
            out.max_k = out.max_k.max(sequence.context_len);
            let next_q = out
                .cu_q
                .last()
                .copied()
                .unwrap_or(0)
                .checked_add(1)
                .ok_or_else(|| AegisError::InvalidPlan("cu_q overflow".into()))?;
            out.cu_q.push(next_q);
            push_cu_k(&mut out, sequence.context_len)?;
            out.block_tables
                .extend(padded_block_table(&sequence.block_table, max_blocks));
        }
        out.num_decode_tokens = decode.len();
        out.validate_varlen_contract(page_tokens)?;
        Ok(out)
    }

    fn validate_varlen_contract(&self, page_tokens: usize) -> Result<()> {
        validate_page_tokens(page_tokens)?;
        let total_query_tokens = self
            .num_prefill_tokens
            .checked_add(self.num_decode_tokens)
            .ok_or_else(|| {
                AegisError::InvalidPlan("prefill descriptor token count overflow".into())
            })?;
        let num_sequences = self.request_ids.len();
        if num_sequences == 0
            || self.seq_ids.len() != num_sequences
            || self.context_lens.len() != num_sequences
            || self.cu_q.len() != num_sequences + 1
            || self.cu_k.len() != num_sequences + 1
        {
            return Err(AegisError::InvalidPlan(format!(
                "bad prefill descriptor sequence metadata: seqs={} seq_ids={} context_lens={} cu_q={} cu_k={}",
                num_sequences,
                self.seq_ids.len(),
                self.context_lens.len(),
                self.cu_q.len(),
                self.cu_k.len()
            )));
        }
        if self.token_ids.len() != total_query_tokens
            || self.positions.len() != total_query_tokens
            || self.slot_mapping.len() != total_query_tokens
        {
            return Err(AegisError::InvalidPlan(format!(
                "bad prefill descriptor token metadata: tokens={} positions={} slots={} expected={}",
                self.token_ids.len(),
                self.positions.len(),
                self.slot_mapping.len(),
                total_query_tokens
            )));
        }
        validate_cumulative_offsets("cu_q", &self.cu_q, total_query_tokens)?;
        let total_k = self
            .context_lens
            .iter()
            .try_fold(0usize, |acc, &len| acc.checked_add(len as usize))
            .ok_or_else(|| AegisError::InvalidPlan("cu_k total overflow".into()))?;
        validate_cumulative_offsets("cu_k", &self.cu_k, total_k)?;
        for seq in 0..num_sequences {
            let q_len = (self.cu_q[seq + 1] - self.cu_q[seq]) as usize;
            let k_len = (self.cu_k[seq + 1] - self.cu_k[seq]) as usize;
            let context_len = self.context_lens[seq] as usize;
            if k_len != context_len {
                return Err(AegisError::InvalidPlan(format!(
                    "prefill descriptor cu_k/context mismatch at sequence {seq}: cu_k_delta={} context_len={}",
                    k_len, context_len
                )));
            }
            if q_len > self.max_q || context_len > self.max_k {
                return Err(AegisError::InvalidPlan(format!(
                    "prefill descriptor max_q/max_k too small at sequence {seq}: q_len={} max_q={} context_len={} max_k={}",
                    q_len, self.max_q, context_len, self.max_k
                )));
            }
        }
        let block_table_stride = self
            .block_tables
            .len()
            .checked_div(num_sequences)
            .ok_or_else(|| {
                AegisError::InvalidPlan("prefill descriptor block table division failed".into())
            })?;
        if block_table_stride == 0 || block_table_stride * num_sequences != self.block_tables.len()
        {
            return Err(AegisError::InvalidPlan(format!(
                "bad prefill descriptor block table shape: blocks={} seqs={}",
                self.block_tables.len(),
                num_sequences
            )));
        }
        for seq in 0..num_sequences {
            let context_len = self.context_lens[seq] as usize;
            let required_pages = context_len.div_ceil(page_tokens).max(1);
            if required_pages > block_table_stride {
                return Err(AegisError::InvalidPlan(format!(
                    "prefill descriptor block table stride too small at sequence {seq}: required_pages={} stride={}",
                    required_pages, block_table_stride
                )));
            }
            let row_start = seq * block_table_stride;
            for page_idx in 0..required_pages {
                if self.block_tables[row_start + page_idx] == u32::MAX {
                    return Err(AegisError::InvalidPlan(format!(
                        "prefill descriptor has missing physical page at sequence {seq} page {page_idx}"
                    )));
                }
            }
        }
        if self.slot_mapping.contains(&u32::MAX) {
            return Err(AegisError::InvalidPlan(
                "prefill descriptor slot_mapping contains an unmapped sentinel".into(),
            ));
        }
        Ok(())
    }

    fn empty(kind: CudaPrefillBatchKind) -> Self {
        Self {
            request_ids: Vec::new(),
            seq_ids: Vec::new(),
            token_ids: Vec::new(),
            positions: Vec::new(),
            slot_mapping: Vec::new(),
            cu_q: Vec::new(),
            cu_k: Vec::new(),
            context_lens: Vec::new(),
            block_tables: Vec::new(),
            max_q: 0,
            max_k: 0,
            num_prefill_tokens: 0,
            num_decode_tokens: 0,
            kind,
        }
    }
}

fn push_prefill_sequence(
    out: &mut HostPrefillBatchDescriptor,
    sequence: &HostPrefillSequenceDescriptor,
    page_tokens: usize,
    max_blocks: usize,
) -> Result<()> {
    out.request_ids.push(sequence.request_id);
    out.seq_ids.push(sequence.seq_id);
    out.context_lens.push(sequence.context_len as u32);
    out.max_q = out.max_q.max(sequence.token_ids.len());
    out.max_k = out.max_k.max(sequence.context_len);
    for (idx, &token) in sequence.token_ids.iter().enumerate() {
        let logical_position = sequence.start_position + idx;
        out.token_ids.push(token);
        out.positions.push(logical_position as u32);
        out.slot_mapping.push(physical_slot_from_table(
            &sequence.block_table,
            page_tokens,
            logical_position,
        )?);
    }
    let next_q = out
        .cu_q
        .last()
        .copied()
        .unwrap_or(0)
        .checked_add(sequence.token_ids.len() as u32)
        .ok_or_else(|| AegisError::InvalidPlan("cu_q overflow".into()))?;
    out.cu_q.push(next_q);
    push_cu_k(out, sequence.context_len)?;
    out.block_tables
        .extend(padded_block_table(&sequence.block_table, max_blocks));
    Ok(())
}

fn push_cu_k(out: &mut HostPrefillBatchDescriptor, context_len: usize) -> Result<()> {
    let context_len = u32::try_from(context_len).map_err(|_| {
        AegisError::InvalidPlan(format!(
            "prefill descriptor context length exceeds u32: {context_len}"
        ))
    })?;
    let next_k = out
        .cu_k
        .last()
        .copied()
        .unwrap_or(0)
        .checked_add(context_len)
        .ok_or_else(|| AegisError::InvalidPlan("cu_k overflow".into()))?;
    out.cu_k.push(next_k);
    Ok(())
}

fn validate_cumulative_offsets(name: &str, offsets: &[u32], expected_total: usize) -> Result<()> {
    let expected_total = u32::try_from(expected_total).map_err(|_| {
        AegisError::InvalidPlan(format!(
            "bad prefill descriptor {name}: expected total exceeds u32"
        ))
    })?;
    if offsets.first().copied() != Some(0) || offsets.last().copied() != Some(expected_total) {
        return Err(AegisError::InvalidPlan(format!(
            "bad prefill descriptor {name}: first={:?} last={:?} expected_total={}",
            offsets.first(),
            offsets.last(),
            expected_total
        )));
    }
    for window in offsets.windows(2) {
        if window[0] > window[1] {
            return Err(AegisError::InvalidPlan(format!(
                "bad prefill descriptor {name}: offsets are not monotonic: {:?}",
                offsets
            )));
        }
    }
    Ok(())
}

fn padded_block_table(block_table: &[u32], max_blocks: usize) -> impl Iterator<Item = u32> + '_ {
    block_table
        .iter()
        .copied()
        .chain(std::iter::repeat(u32::MAX))
        .take(max_blocks)
}

fn physical_slot_from_table(
    block_table: &[u32],
    page_tokens: usize,
    logical_position: usize,
) -> Result<u32> {
    let logical_page = logical_position / page_tokens;
    let page_offset = logical_position % page_tokens;
    let Some(&physical_page) = block_table.get(logical_page) else {
        return Err(AegisError::InvalidPlan(format!(
            "paged descriptor missing page: logical_position={} page_tokens={} table_len={}",
            logical_position,
            page_tokens,
            block_table.len()
        )));
    };
    physical_page
        .checked_mul(page_tokens as u32)
        .and_then(|slot| slot.checked_add(page_offset as u32))
        .ok_or_else(|| AegisError::InvalidPlan("physical slot overflow".into()))
}

fn validate_page_tokens(page_tokens: usize) -> Result<()> {
    if page_tokens == 0 || page_tokens > u32::MAX as usize {
        return Err(AegisError::InvalidPlan(format!(
            "paged descriptor requires page_tokens in 1..=u32::MAX, got {page_tokens}"
        )));
    }
    Ok(())
}

pub(super) fn validate_dense_prefill_tokens(tokens: &[usize], vocab_size: usize) -> Result<()> {
    validate_dense_prefill_tokens_usize(tokens.iter().copied(), vocab_size)
}

fn validate_dense_prefill_tokens_usize(
    tokens: impl IntoIterator<Item = usize>,
    vocab_size: usize,
) -> Result<()> {
    let tokens = tokens.into_iter().collect::<Vec<_>>();
    if tokens.is_empty() {
        return Err(AegisError::InvalidPlan(
            "CUDA prefill batch cannot be empty".into(),
        ));
    }
    if vocab_size > u32::MAX as usize {
        return Err(AegisError::InvalidPlan(format!(
            "CUDA prefill dense adapter requires u32 vocab size: vocab_size={vocab_size}"
        )));
    }
    for (idx, token) in tokens.into_iter().enumerate() {
        if token >= vocab_size || token > u32::MAX as usize {
            return Err(AegisError::InvalidPlan(format!(
                "CUDA prefill token id out of range: chunk_idx={} token={} vocab_size={}",
                idx, token, vocab_size
            )));
        }
    }
    Ok(())
}

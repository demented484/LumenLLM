use cudarc::driver::{LaunchConfig, PushKernelArg};

use super::{CudaRuntime, map_cuda_err};
use crate::cuda::{CudaPrefillAttentionKernel, DensePrefillMetadataProof, DeviceBuffer};
use crate::error::{AegisError, Result};

fn u32_arg(name: &str, value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        AegisError::InvalidPlan(format!(
            "CUDA attention argument {name} exceeds u32 range: {value}"
        ))
    })
}

fn checked_len(label: &str, lhs: usize, rhs: usize) -> Result<usize> {
    lhs.checked_mul(rhs).ok_or_else(|| {
        AegisError::InvalidPlan(format!(
            "CUDA attention {label} length overflow: {lhs} * {rhs}"
        ))
    })
}

fn checked_sum(label: &str, lhs: usize, rhs: usize) -> Result<usize> {
    lhs.checked_add(rhs).ok_or_else(|| {
        AegisError::InvalidPlan(format!(
            "CUDA attention {label} length overflow: {lhs} + {rhs}"
        ))
    })
}

const FLASH_COMPAT_PAGE_TOKENS: usize = 256;

impl CudaRuntime {
    pub fn attention_decode_device(
        &self,
        key_cache: &DeviceBuffer<u16>,
        value_cache: &DeviceBuffer<u16>,
        query: &DeviceBuffer<f32>,
        seq_len: usize,
        num_attention_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if num_kv_heads == 0 || num_attention_heads == 0 || head_dim == 0 {
            return Err(AegisError::InvalidPlan(format!(
                "attention dimensions must be non-zero: q_heads={} kv_heads={} head_dim={}",
                num_attention_heads, num_kv_heads, head_dim
            )));
        }
        let query_len = checked_len("decode query", num_attention_heads, head_dim)?;
        let kv_width = checked_len("decode kv width", num_kv_heads, head_dim)?;
        if query.len() != query_len || output.len() != query_len {
            return Err(AegisError::InvalidPlan(
                "attention query/output shape mismatch".into(),
            ));
        }
        if seq_len == 0
            || key_cache.len() < checked_len("decode key cache", seq_len, kv_width)?
            || value_cache.len() < checked_len("decode value cache", seq_len, kv_width)?
        {
            return Err(AegisError::InvalidPlan(format!(
                "attention kv shape mismatch: seq_len={} kv_width={} key_cache={} value_cache={}",
                seq_len,
                kv_width,
                key_cache.len(),
                value_cache.len()
            )));
        }
        if num_attention_heads % num_kv_heads != 0 {
            return Err(AegisError::InvalidPlan(
                "attention heads must be divisible by kv heads".into(),
            ));
        }
        let seq_len = u32_arg("seq_len", seq_len)?;
        let num_attention_heads = u32_arg("num_attention_heads", num_attention_heads)?;
        let num_kv_heads = u32_arg("num_kv_heads", num_kv_heads)?;
        let head_dim = u32_arg("head_dim", head_dim)?;
        let block_dim = 128u32;
        let legacy_shared_bytes = seq_len as usize * std::mem::size_of::<f32>()
            + block_dim as usize * std::mem::size_of::<f32>();
        let streaming = legacy_shared_bytes > 48 * 1024;
        let cfg = LaunchConfig {
            grid_dim: (num_attention_heads, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: if streaming {
                ((block_dim as usize + head_dim as usize + 3) * std::mem::size_of::<f32>()) as u32
            } else {
                legacy_shared_bytes as u32
            },
        };
        let kernel = if streaming {
            &self.kernels.attention_decode_streaming
        } else {
            &self.kernels.attention
        };
        unsafe {
            self.stream
                .launch_builder(kernel)
                .arg(&key_cache.slice)
                .arg(&value_cache.slice)
                .arg(&query.slice)
                .arg(&seq_len)
                .arg(&num_attention_heads)
                .arg(&num_kv_heads)
                .arg(&head_dim)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch attention decode"))?;
        Ok(())
    }

    pub fn attention_prefill_batched_device(
        &self,
        key_cache: &DeviceBuffer<u16>,
        value_cache: &DeviceBuffer<u16>,
        key_chunk: &DeviceBuffer<f32>,
        value_chunk: &DeviceBuffer<f32>,
        query: &DeviceBuffer<f32>,
        start_position: usize,
        batch: usize,
        num_attention_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if num_kv_heads == 0 || num_attention_heads == 0 || head_dim == 0 {
            return Err(AegisError::InvalidPlan(format!(
                "prefill attention dimensions must be non-zero: q_heads={} kv_heads={} head_dim={}",
                num_attention_heads, num_kv_heads, head_dim
            )));
        }
        let query_len = checked_len("prefill query width", num_attention_heads, head_dim)?;
        let kv_width = checked_len("prefill kv width", num_kv_heads, head_dim)?;
        let query_batch_len = checked_len("prefill query batch", batch, query_len)?;
        let kv_batch_len = checked_len("prefill kv batch", batch, kv_width)?;
        if query.len() < query_batch_len || output.len() < query_batch_len {
            return Err(AegisError::InvalidPlan(
                "batched attention query/output shape mismatch".into(),
            ));
        }
        if key_chunk.len() < kv_batch_len || value_chunk.len() < kv_batch_len {
            return Err(AegisError::InvalidPlan(
                "batched attention current kv chunk shape mismatch".into(),
            ));
        }
        let max_seq_len = checked_sum("prefill max seq", start_position, batch)?;
        if batch == 0
            || key_cache.len() < checked_len("prefill key cache", max_seq_len, kv_width)?
            || value_cache.len() < checked_len("prefill value cache", max_seq_len, kv_width)?
        {
            return Err(AegisError::InvalidPlan(format!(
                "batched attention kv shape mismatch: start={} batch={} kv_width={} key_cache={} value_cache={}",
                start_position,
                batch,
                kv_width,
                key_cache.len(),
                value_cache.len()
            )));
        }
        if num_attention_heads % num_kv_heads != 0 {
            return Err(AegisError::InvalidPlan(
                "attention heads must be divisible by kv heads".into(),
            ));
        }
        let start_position = u32_arg("start_position", start_position)?;
        let batch = u32_arg("batch", batch)?;
        let num_attention_heads = u32_arg("num_attention_heads", num_attention_heads)?;
        let num_kv_heads = u32_arg("num_kv_heads", num_kv_heads)?;
        let head_dim = u32_arg("head_dim", head_dim)?;
        let warp_eligible = start_position == 0 && head_dim % 32 == 0 && head_dim <= 256;
        let warp_parallel = match self.config.prefill_attention {
            CudaPrefillAttentionKernel::Auto => false,
            CudaPrefillAttentionKernel::WarpFlash => warp_eligible,
            CudaPrefillAttentionKernel::Reference => false,
            CudaPrefillAttentionKernel::Continuation => false,
            CudaPrefillAttentionKernel::FlashVarlen => false,
        };
        let block_dim = if warp_parallel { 128 } else { 128 };
        let legacy_shared_bytes = (max_seq_len + block_dim as usize) * std::mem::size_of::<f32>();
        if matches!(
            self.config.prefill_attention,
            CudaPrefillAttentionKernel::Reference
        ) && legacy_shared_bytes > 48 * 1024
        {
            return Err(AegisError::InvalidPlan(format!(
                "CUDA reference prefill attention requires {} bytes of dynamic shared memory; use cuda.prefill-attention=auto or continuation for long prefixes",
                legacy_shared_bytes
            )));
        }
        let continuation = matches!(
            self.config.prefill_attention,
            CudaPrefillAttentionKernel::Continuation
        ) || (!matches!(
            self.config.prefill_attention,
            CudaPrefillAttentionKernel::Reference
        ) && !warp_parallel
            && legacy_shared_bytes > 48 * 1024);
        let cfg = LaunchConfig {
            grid_dim: (num_attention_heads, batch, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: if continuation {
                ((block_dim as usize + head_dim as usize + 3) * std::mem::size_of::<f32>()) as u32
            } else {
                (max_seq_len * std::mem::size_of::<f32>()
                    + block_dim as usize * std::mem::size_of::<f32>()) as u32
            },
        };
        let cache_only = !continuation || warp_parallel;
        let kernel = if warp_parallel {
            &self.kernels.attention_prefill_batched_warp
        } else if continuation {
            &self.kernels.attention_prefill_continuation
        } else {
            &self.kernels.attention_prefill_batched
        };
        if cache_only {
            unsafe {
                self.stream
                    .launch_builder(kernel)
                    .arg(&key_cache.slice)
                    .arg(&value_cache.slice)
                    .arg(&query.slice)
                    .arg(&start_position)
                    .arg(&batch)
                    .arg(&num_attention_heads)
                    .arg(&num_kv_heads)
                    .arg(&head_dim)
                    .arg(&mut output.slice)
                    .launch(cfg)
            }
        } else {
            unsafe {
                self.stream
                    .launch_builder(kernel)
                    .arg(&key_cache.slice)
                    .arg(&value_cache.slice)
                    .arg(&key_chunk.slice)
                    .arg(&value_chunk.slice)
                    .arg(&query.slice)
                    .arg(&start_position)
                    .arg(&batch)
                    .arg(&num_attention_heads)
                    .arg(&num_kv_heads)
                    .arg(&head_dim)
                    .arg(&mut output.slice)
                    .launch(cfg)
            }
        }
        .map_err(map_cuda_err("launch batched attention prefill"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_prefill_dense_compat_device(
        &self,
        key_cache: &DeviceBuffer<u16>,
        value_cache: &DeviceBuffer<u16>,
        key_chunk: &DeviceBuffer<f32>,
        value_chunk: &DeviceBuffer<f32>,
        query: &DeviceBuffer<f32>,
        query_half: &mut DeviceBuffer<u16>,
        slot_mapping: &DeviceBuffer<u32>,
        cu_q: &DeviceBuffer<u32>,
        cu_k: &DeviceBuffer<u32>,
        context_lens: &DeviceBuffer<u32>,
        block_tables: &DeviceBuffer<u32>,
        num_sequences: usize,
        start_position: usize,
        batch: usize,
        num_attention_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        output: &mut DeviceBuffer<f32>,
        dense_metadata: DensePrefillMetadataProof,
    ) -> Result<()> {
        let _ = u32_arg("num_sequences", num_sequences)?;
        let _ = u32_arg("start_position", start_position)?;
        let _ = u32_arg("batch", batch)?;
        if num_sequences != 1
            || cu_q.len() < 2
            || context_lens.len() < 1
            || slot_mapping.len() < batch
            || dense_metadata.start_position() != start_position
            || dense_metadata.batch() != batch
            || dense_metadata.context_len()
                != start_position.checked_add(batch).ok_or_else(|| {
                    AegisError::InvalidPlan(format!(
                        "dense varlen prefill adapter position overflow: start={start_position} batch={batch}"
                    ))
                })?
        {
            return Err(AegisError::InvalidPlan(format!(
                "dense varlen prefill adapter requires proven identity metadata: seqs={} cu_q={} context_lens={} slots={} start={} batch={} proof_start={} proof_batch={} proof_context={}",
                num_sequences,
                cu_q.len(),
                context_lens.len(),
                slot_mapping.len(),
                start_position,
                batch,
                dense_metadata.start_position(),
                dense_metadata.batch(),
                dense_metadata.context_len()
            )));
        }
        let flash_varlen_min_context = 128;
        let use_flash_varlen = match self.config.prefill_attention {
            CudaPrefillAttentionKernel::FlashVarlen => {
                dense_metadata.context_len() >= flash_varlen_min_context
            }
            CudaPrefillAttentionKernel::Auto => {
                dense_metadata.context_len() >= flash_varlen_min_context
            }
            CudaPrefillAttentionKernel::Reference
            | CudaPrefillAttentionKernel::WarpFlash
            | CudaPrefillAttentionKernel::Continuation => false,
        };
        if use_flash_varlen {
            let q_len = checked_len(
                "dense varlen query half conversion",
                batch,
                checked_len(
                    "dense varlen query half width",
                    num_attention_heads,
                    head_dim,
                )?,
            )?;
            self.f32_to_f16_device(query, q_len, query_half)?;
            let block_table_stride = dense_metadata
                .context_len()
                .div_ceil(FLASH_COMPAT_PAGE_TOKENS)
                .max(1);
            return self.attention_prefill_paged_varlen_device(
                key_cache,
                value_cache,
                query,
                Some(query_half),
                slot_mapping,
                cu_q,
                cu_k,
                context_lens,
                block_tables,
                num_sequences,
                batch,
                0,
                batch,
                dense_metadata.context_len(),
                block_table_stride,
                num_attention_heads,
                num_kv_heads,
                head_dim,
                output,
            );
        }
        self.attention_prefill_batched_device(
            key_cache,
            value_cache,
            key_chunk,
            value_chunk,
            query,
            start_position,
            batch,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            output,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_prefill_paged_varlen_device(
        &self,
        key_cache: &DeviceBuffer<u16>,
        value_cache: &DeviceBuffer<u16>,
        query: &DeviceBuffer<f32>,
        query_half: Option<&DeviceBuffer<u16>>,
        slot_mapping: &DeviceBuffer<u32>,
        cu_q: &DeviceBuffer<u32>,
        cu_k: &DeviceBuffer<u32>,
        context_lens: &DeviceBuffer<u32>,
        block_tables: &DeviceBuffer<u32>,
        num_sequences: usize,
        num_prefill_tokens: usize,
        num_decode_tokens: usize,
        max_q: usize,
        max_k: usize,
        block_table_stride: usize,
        num_attention_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if num_kv_heads == 0 || num_attention_heads == 0 || head_dim == 0 {
            return Err(AegisError::InvalidPlan(format!(
                "paged varlen prefill dimensions must be non-zero: q_heads={} kv_heads={} head_dim={}",
                num_attention_heads, num_kv_heads, head_dim
            )));
        }
        if num_sequences == 0 || max_q == 0 || max_k == 0 {
            return Err(AegisError::InvalidPlan(format!(
                "paged varlen prefill requires non-empty metadata: seqs={} max_q={} max_k={}",
                num_sequences, max_q, max_k
            )));
        }
        let page_tokens_usize = FLASH_COMPAT_PAGE_TOKENS;
        let required_pages_per_sequence = max_k.div_ceil(page_tokens_usize).max(1);
        if block_table_stride < required_pages_per_sequence {
            return Err(AegisError::InvalidPlan(format!(
                "paged varlen prefill block table stride too small: stride={} required={} max_k={} page_tokens={}",
                block_table_stride, required_pages_per_sequence, max_k, page_tokens_usize
            )));
        }
        if cu_q.len() < num_sequences + 1
            || cu_k.len() < num_sequences + 1
            || context_lens.len() < num_sequences
            || block_tables.len() < num_sequences.saturating_mul(block_table_stride)
            || slot_mapping.len() < num_prefill_tokens + num_decode_tokens
        {
            return Err(AegisError::InvalidPlan(format!(
                "paged varlen prefill metadata too small: seqs={} cu_q={} cu_k={} context_lens={} block_tables={} stride={} slots={} tokens={}",
                num_sequences,
                cu_q.len(),
                cu_k.len(),
                context_lens.len(),
                block_tables.len(),
                block_table_stride,
                slot_mapping.len(),
                num_prefill_tokens + num_decode_tokens
            )));
        }
        let q_width = checked_len("paged varlen q width", num_attention_heads, head_dim)?;
        let _ = checked_len("paged varlen kv width", num_kv_heads, head_dim)?;
        let q_tokens = checked_len("paged varlen query tokens", num_prefill_tokens, q_width)?;
        if output.len() < q_tokens
            || query
                .len()
                .max(query_half.map(|buffer| buffer.len()).unwrap_or(0))
                < q_tokens
        {
            return Err(AegisError::InvalidPlan(
                "paged varlen prefill query/output shape mismatch".into(),
            ));
        }
        let kv_width = checked_len("paged varlen kv width", num_kv_heads, head_dim)?;
        let physical_slots = key_cache.len() / kv_width;
        if key_cache.len() % kv_width != 0
            || value_cache.len() != key_cache.len()
            || physical_slots < max_k
        {
            return Err(AegisError::InvalidPlan(format!(
                "paged varlen prefill kv cache too small or misaligned: max_k={} kv_width={} key_cache={} value_cache={}",
                max_k,
                kv_width,
                key_cache.len(),
                value_cache.len()
            )));
        }
        if num_attention_heads % num_kv_heads != 0 {
            return Err(AegisError::InvalidPlan(
                "paged varlen attention heads must be divisible by kv heads".into(),
            ));
        }
        let head_dim_usize = head_dim;
        let use_halfq_block4 = query_half.is_some()
            && num_sequences == 1
            && num_decode_tokens == 0
            && num_prefill_tokens >= 4
            && head_dim_usize <= 256;
        let num_sequences = u32_arg("num_sequences", num_sequences)?;
        let total_q = u32_arg("num_prefill_tokens", num_prefill_tokens)?;
        let num_attention_heads = u32_arg("num_attention_heads", num_attention_heads)?;
        let num_kv_heads = u32_arg("num_kv_heads", num_kv_heads)?;
        let head_dim = u32_arg("head_dim", head_dim)?;
        let page_tokens = u32_arg("page_tokens", page_tokens_usize)?;
        let block_table_stride = u32_arg("block_table_stride", block_table_stride)?;
        let physical_slots = u32_arg("physical_slots", physical_slots)?;
        let warp_eligible = head_dim_usize <= 256 && head_dim_usize % 32 == 0;
        let use_warp = matches!(
            self.config.prefill_attention,
            CudaPrefillAttentionKernel::WarpFlash
        ) && warp_eligible;
        let block_dim = if use_halfq_block4 { 64_u32 } else { 128_u32 };
        let mut shared_floats = if use_warp {
            (block_dim / 32) as usize * 3 + head_dim_usize + 4
        } else if use_halfq_block4 {
            let q_block = 4_usize;
            q_block * block_dim as usize + (q_block * 2 + 2) * head_dim_usize + q_block * 4
        } else {
            block_dim as usize + head_dim_usize + 4
        };
        if query_half.is_some() && !use_halfq_block4 {
            shared_floats += head_dim_usize;
        }
        let grid_q = if use_halfq_block4 {
            total_q.div_ceil(4)
        } else {
            total_q
        };
        let cfg = LaunchConfig {
            grid_dim: (num_attention_heads, grid_q, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: (shared_floats * std::mem::size_of::<f32>()) as u32,
        };
        if let Some(query_half) = query_half {
            let kernel = if use_halfq_block4 {
                &self.kernels.attention_prefill_paged_varlen_halfq_block4
            } else {
                &self.kernels.attention_prefill_paged_varlen_halfq
            };
            unsafe {
                self.stream
                    .launch_builder(kernel)
                    .arg(&key_cache.slice)
                    .arg(&value_cache.slice)
                    .arg(&query_half.slice)
                    .arg(&slot_mapping.slice)
                    .arg(&cu_q.slice)
                    .arg(&context_lens.slice)
                    .arg(&block_tables.slice)
                    .arg(&num_sequences)
                    .arg(&total_q)
                    .arg(&num_attention_heads)
                    .arg(&num_kv_heads)
                    .arg(&head_dim)
                    .arg(&page_tokens)
                    .arg(&block_table_stride)
                    .arg(&physical_slots)
                    .arg(&mut output.slice)
                    .launch(cfg)
            }
        } else {
            let kernel = if use_warp {
                &self.kernels.attention_prefill_paged_varlen_warp
            } else {
                &self.kernels.attention_prefill_paged_varlen
            };
            unsafe {
                self.stream
                    .launch_builder(kernel)
                    .arg(&key_cache.slice)
                    .arg(&value_cache.slice)
                    .arg(&query.slice)
                    .arg(&slot_mapping.slice)
                    .arg(&cu_q.slice)
                    .arg(&context_lens.slice)
                    .arg(&block_tables.slice)
                    .arg(&num_sequences)
                    .arg(&total_q)
                    .arg(&num_attention_heads)
                    .arg(&num_kv_heads)
                    .arg(&head_dim)
                    .arg(&page_tokens)
                    .arg(&block_table_stride)
                    .arg(&physical_slots)
                    .arg(&mut output.slice)
                    .launch(cfg)
            }
        }
        .map_err(map_cuda_err("launch paged varlen prefill attention"))?;
        Ok(())
    }
}

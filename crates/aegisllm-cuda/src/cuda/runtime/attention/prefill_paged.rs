use cudarc::driver::{LaunchConfig, PushKernelArg, sys::CUfunction_attribute_enum};

use super::*;
use crate::cuda::{
    CudaAttentionRequest, CudaAttentionSplitScratch, DeviceBuffer,
};
use aegisllm_base::error::{AegisError, Result};

impl CudaRuntime {
    pub(super) fn attention_prefill_paged_varlen_halfq_wmma_hdim128_gqa4_device(
        &self,
        key_cache: &DeviceBuffer<u16>,
        value_cache: &DeviceBuffer<u16>,
        query_half: &DeviceBuffer<u16>,
        slot_mapping: &DeviceBuffer<u32>,
        cu_q: &DeviceBuffer<u32>,
        context_lens: &DeviceBuffer<u32>,
        block_tables: &DeviceBuffer<u32>,
        num_sequences: usize,
        total_query_tokens: usize,
        max_q: usize,
        block_table_stride: usize,
        physical_slots: usize,
        num_attention_heads: usize,
        num_kv_heads: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let head_dim = 128usize;
        if num_kv_heads == 0 || num_attention_heads % num_kv_heads != 0 {
            return Err(AegisError::InvalidPlan(
                "paged gqa4 wmma attention heads must be divisible by kv heads".into(),
            ));
        }
        let group = num_attention_heads / num_kv_heads;
        if group < PAGED_WMMA_GQA4_HEADS {
            return Err(AegisError::InvalidPlan(format!(
                "paged gqa4 wmma attention requires GQA group >= {}, got {group}",
                PAGED_WMMA_GQA4_HEADS
            )));
        }
        let group_tiles = group.div_ceil(PAGED_WMMA_GQA4_HEADS);
        let q_rows = PAGED_WMMA_GQA4_Q_TOKENS * PAGED_WMMA_GQA4_HEADS;
        let q_width = checked_len("paged gqa4 wmma q width", num_attention_heads, head_dim)?;
        let q_tokens = checked_len("paged gqa4 wmma query tokens", total_query_tokens, q_width)?;
        if query_half.len() < q_tokens || output.len() < q_tokens {
            return Err(AegisError::InvalidPlan(format!(
                "paged gqa4 wmma attention q/output shape mismatch: query_half={} output={} required={}",
                query_half.len(),
                output.len(),
                q_tokens
            )));
        }
        let kv_width = checked_len("paged gqa4 wmma kv width", num_kv_heads, head_dim)?;
        let cache_len = checked_len("paged gqa4 wmma kv cache", physical_slots, kv_width)?;
        if key_cache.len() < cache_len || value_cache.len() < cache_len {
            return Err(AegisError::InvalidPlan(format!(
                "paged gqa4 wmma attention kv cache shape mismatch: key_cache={} value_cache={} required={}",
                key_cache.len(),
                value_cache.len(),
                cache_len
            )));
        }
        if slot_mapping.len() < total_query_tokens
            || cu_q.len() < num_sequences + 1
            || context_lens.len() < num_sequences
            || block_tables.len() < num_sequences.saturating_mul(block_table_stride)
        {
            return Err(AegisError::InvalidPlan(format!(
                "paged gqa4 wmma metadata too small: seqs={} slots={} cu_q={} context_lens={} block_tables={} stride={} total_q={}",
                num_sequences,
                slot_mapping.len(),
                cu_q.len(),
                context_lens.len(),
                block_tables.len(),
                block_table_stride,
                total_query_tokens
            )));
        }
        let half_values =
            q_rows * head_dim + 2 * DENSE_WMMA_K_TILE * head_dim + q_rows * (DENSE_WMMA_K_TILE + 8);
        let float_values = q_rows * (DENSE_WMMA_K_TILE + 8) + q_rows * (head_dim + 8) + q_rows * 3;
        let shared_mem_bytes =
            half_values * std::mem::size_of::<u16>() + float_values * std::mem::size_of::<f32>();
        self.kernels
            .attention_prefill_paged_varlen_halfq_wmma_hdim128_gqa4
            .set_attribute(
                CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                i32::try_from(shared_mem_bytes).map_err(|_| {
                    AegisError::InvalidPlan(format!(
                        "paged gqa4 padded wmma shared memory exceeds i32: {shared_mem_bytes}"
                    ))
                })?,
            )
            .map_err(map_cuda_err(
                "set paged gqa4 padded wmma max dynamic shared memory",
            ))?;
        let cfg = LaunchConfig {
            grid_dim: (
                u32_arg(
                    "paged gqa4 wmma kv/group blocks",
                    checked_len("paged gqa4 wmma group blocks", num_kv_heads, group_tiles)?,
                )?,
                u32_arg(
                    "paged gqa4 wmma q blocks",
                    max_q.div_ceil(PAGED_WMMA_GQA4_Q_TOKENS),
                )?,
                u32_arg("paged gqa4 wmma sequences", num_sequences)?,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: u32_arg("paged gqa4 padded wmma shared memory", shared_mem_bytes)?,
        };
        let num_sequences = u32_arg("num_sequences", num_sequences)?;
        let total_q = u32_arg("total_query_tokens", total_query_tokens)?;
        let num_attention_heads = u32_arg("num_attention_heads", num_attention_heads)?;
        let num_kv_heads = u32_arg("num_kv_heads", num_kv_heads)?;
        let head_dim = u32_arg("head_dim", head_dim)?;
        let page_tokens = u32_arg("page_tokens", FLASH_COMPAT_PAGE_TOKENS)?;
        let block_table_stride = u32_arg("block_table_stride", block_table_stride)?;
        let physical_slots = u32_arg("physical_slots", physical_slots)?;
        unsafe {
            self.stream
                .launch_builder(
                    &self
                        .kernels
                        .attention_prefill_paged_varlen_halfq_wmma_hdim128_gqa4,
                )
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
        .map_err(map_cuda_err(
            "launch paged gqa4 halfq wmma prefill attention",
        ))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]

    pub fn attention_prefill_paged_varlen_device(
        &self,
        key_cache: &DeviceBuffer<u16>,
        value_cache: &DeviceBuffer<u16>,
        query: &DeviceBuffer<f32>,
        query_half: Option<&DeviceBuffer<u16>>,
        split_scratch: Option<(
            &mut DeviceBuffer<f32>,
            &mut DeviceBuffer<f32>,
            &mut DeviceBuffer<f32>,
        )>,
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
        let mut request = CudaAttentionRequest {
            q: query,
            q_half: query_half,
            k_cache: key_cache,
            v_cache: value_cache,
            cu_q,
            cu_k,
            context_lens,
            slot_mapping,
            block_tables,
            split_scratch: split_scratch.map(|(acc, m, l)| CudaAttentionSplitScratch { acc, m, l }),
            output,
            num_sequences,
            num_prefill_tokens,
            num_decode_tokens,
            max_q,
            max_k,
            block_table_stride,
            head_dim,
            num_q_heads: num_attention_heads,
            num_kv_heads,
            causal: true,
        };
        self.attention_prefill_request_device(&mut request)
    }

}

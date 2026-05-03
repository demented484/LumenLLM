use cudarc::driver::{LaunchConfig, PushKernelArg};

use super::*;
use crate::cuda::{CudaAttentionBackend, CudaAttentionRequest, CudaPrefillAttentionKernel};
use aegisllm_base::cuda_config::CUDA_PREFILL_VARLEN_MIN_CONTEXT;
use aegisllm_base::error::{AegisError, Result};

impl CudaRuntime {
    pub(super) fn select_prefill_attention_backend(
        &self,
        context_len: usize,
        _head_dim: usize,
    ) -> Result<CudaAttentionBackend> {
        match self.config.prefill_attention {
            CudaPrefillAttentionKernel::Auto => {
                let target = CudaAttentionBackend::auto_target_for_compute_capability(
                    self.compute_capability(),
                );
                Ok(match target {
                    // Production FA2/FA3/FA4 kernels are separate future backends. Until they
                    // pass correctness and throughput gates, auto falls back to the fastest
                    // available Aegis implementation for long prefixes and reference for short ones.
                    CudaAttentionBackend::FlashAttention2
                    | CudaAttentionBackend::FlashAttention3
                    | CudaAttentionBackend::FlashAttention4
                        if context_len >= CUDA_PREFILL_VARLEN_MIN_CONTEXT =>
                    {
                        CudaAttentionBackend::AegisVarlen
                    }
                    _ => CudaAttentionBackend::Reference,
                })
            }
            CudaPrefillAttentionKernel::Off | CudaPrefillAttentionKernel::Reference => {
                Ok(CudaAttentionBackend::Reference)
            }
            CudaPrefillAttentionKernel::FlashAttention2 => Err(AegisError::Unsupported(
                "cuda.prefill-attention=fa2 is reserved for the production Ampere/Ada FA2 backend; use aegis-varlen, off, or auto until that kernel lands".into(),
            )),
            CudaPrefillAttentionKernel::FlashAttention3 => Err(AegisError::Unsupported(
                "cuda.prefill-attention=fa3 is reserved for the production Hopper FA3 backend; use aegis-varlen, off, or auto until that kernel lands".into(),
            )),
            CudaPrefillAttentionKernel::FlashAttention4 => {
                if !self
                    .compute_capability()
                    .is_some_and(|compute_capability| compute_capability.starts_with("12."))
                {
                    return Err(AegisError::Unsupported(format!(
                        "cuda.prefill-attention=fa4 requires a Blackwell/SM12.x CUDA device; device {} is not reported as Blackwell",
                        self.device_index
                    )));
                }
                Ok(CudaAttentionBackend::FlashAttention4)
            }
            CudaPrefillAttentionKernel::AegisVarlen => Ok(CudaAttentionBackend::AegisVarlen),
            CudaPrefillAttentionKernel::WarpFlash | CudaPrefillAttentionKernel::Continuation => {
                Ok(CudaAttentionBackend::Reference)
            }
        }
    }


    pub fn attention_prefill_request_device(
        &self,
        request: &mut CudaAttentionRequest<'_>,
    ) -> Result<()> {
        let key_cache = request.k_cache;
        let value_cache = request.v_cache;
        let query = request.q;
        let query_half = request.q_half;
        let slot_mapping = request.slot_mapping;
        let cu_q = request.cu_q;
        let cu_k = request.cu_k;
        let context_lens = request.context_lens;
        let block_tables = request.block_tables;
        let num_sequences = request.num_sequences;
        let num_prefill_tokens = request.num_prefill_tokens;
        let num_decode_tokens = request.num_decode_tokens;
        let max_q = request.max_q;
        let max_k = request.max_k;
        let block_table_stride = request.block_table_stride;
        let num_attention_heads = request.num_q_heads;
        let num_kv_heads = request.num_kv_heads;
        let head_dim = request.head_dim;
        if !request.causal {
            return Err(AegisError::Unsupported(
                "CUDA prefill attention ABI currently requires causal=true".into(),
            ));
        }
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
        if num_prefill_tokens == 0 {
            return Err(AegisError::InvalidPlan(
                "paged varlen prefill requires at least one prefill query token; decode-only attention needs the decode ABI".into(),
            ));
        }
        if num_decode_tokens > 0 {
            return Err(AegisError::Unsupported(format!(
                "paged varlen mixed prefill+decode is not implemented yet: prefill_tokens={} decode_tokens={}; scheduler descriptors can express mixed batches, but Aegis kernels currently execute prefill rows only",
                num_prefill_tokens, num_decode_tokens
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
        let selected_backend = self.select_prefill_attention_backend(max_k, head_dim)?;
        if !matches!(
            selected_backend,
            CudaAttentionBackend::AegisVarlen | CudaAttentionBackend::FlashAttention4
        ) {
            return Err(AegisError::InvalidPlan(format!(
                "paged attention ABI was invoked for backend {}; expected aegis-varlen or fa4",
                selected_backend.canonical_name()
            )));
        }
        let total_query_tokens = num_prefill_tokens;
        let q_width = checked_len("paged varlen q width", num_attention_heads, head_dim)?;
        let _ = checked_len("paged varlen kv width", num_kv_heads, head_dim)?;
        let q_tokens = checked_len("paged varlen query tokens", total_query_tokens, q_width)?;
        if request.output.len() < q_tokens || query.len() < q_tokens {
            return Err(AegisError::InvalidPlan(
                "paged varlen prefill query/output shape mismatch".into(),
            ));
        }
        if let Some(query_half) = query_half
            && query_half.len() < q_tokens
        {
            return Err(AegisError::InvalidPlan(format!(
                "paged varlen prefill q_half shape mismatch: required={} actual={}",
                q_tokens,
                query_half.len()
            )));
        }
        let kv_width = checked_len("paged varlen kv width", num_kv_heads, head_dim)?;
        let physical_slots = key_cache.len() / kv_width;
        if !key_cache.len().is_multiple_of(kv_width)
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
        if !num_attention_heads.is_multiple_of(num_kv_heads) {
            return Err(AegisError::InvalidPlan(
                "paged varlen attention heads must be divisible by kv heads".into(),
            ));
        }
        let head_dim_usize = head_dim;
        let q_blocks_usize = total_query_tokens.div_ceil(FLASH_SPLIT_Q_BLOCK);
        let split_count_usize = max_k.div_ceil(FLASH_SPLIT_K_TOKENS).max(1);
        let split_rows = q_blocks_usize
            .checked_mul(num_attention_heads)
            .and_then(|value| value.checked_mul(split_count_usize))
            .and_then(|value| value.checked_mul(FLASH_SPLIT_Q_BLOCK))
            .ok_or_else(|| AegisError::InvalidPlan("split-K attention scratch overflow".into()))?;
        let split_acc_len = split_rows
            .checked_mul(head_dim_usize)
            .ok_or_else(|| AegisError::InvalidPlan("split-K attention acc overflow".into()))?;
        let split_scratch_ready = request.split_scratch.as_ref().is_some_and(|scratch| {
            scratch.acc.len() >= split_acc_len
                && scratch.m.len() >= split_rows
                && scratch.l.len() >= split_rows
        });
        let native_query_half = query_half;
        let use_halfq_single_sequence = native_query_half.is_some()
            && num_sequences == 1
            && num_decode_tokens == 0
            && head_dim_usize <= 256;
        let gqa_group = num_attention_heads / num_kv_heads;
        let use_paged_wmma_gqa4 = native_query_half.is_some()
            && matches!(selected_backend, CudaAttentionBackend::AegisVarlen)
            && num_decode_tokens == 0
            && head_dim_usize == 128
            && gqa_group >= PAGED_WMMA_GQA4_HEADS
            && num_prefill_tokens >= PAGED_WMMA_GQA4_Q_TOKENS;
        let use_halfq_block4 =
            use_halfq_single_sequence && num_prefill_tokens >= TILED_HALFQ_Q_BLOCK;
        let mut use_fa4_hdim128 = use_halfq_single_sequence
            && head_dim_usize == 128
            && matches!(selected_backend, CudaAttentionBackend::FlashAttention4);
        if matches!(selected_backend, CudaAttentionBackend::FlashAttention4) && !use_fa4_hdim128 {
            if matches!(
                self.config.prefill_attention,
                CudaPrefillAttentionKernel::FlashAttention4
            ) {
                return Err(AegisError::Unsupported(format!(
                    "cuda.prefill-attention=fa4 currently supports only single-sequence causal prefill with q_half, paged f16 KV cache, head_dim=128, and no decode tokens; got seqs={} prefill={} decode={} head_dim={} q_half={}",
                    request.num_sequences,
                    request.num_prefill_tokens,
                    request.num_decode_tokens,
                    request.head_dim,
                    request.q_half.is_some()
                )));
            }
            use_fa4_hdim128 = false;
        }
        let use_halfq_block4_split = use_halfq_block4
            && !use_fa4_hdim128
            && split_scratch_ready
            && split_count_usize > 1
            && max_k >= 4096;
        if use_paged_wmma_gqa4 && !use_halfq_block4_split {
            let Some(query_half) = native_query_half else {
                return Err(AegisError::InvalidPlan(
                    "paged GQA4 WMMA attention requires query_half".into(),
                ));
            };
            return self.attention_prefill_paged_varlen_halfq_wmma_hdim128_gqa4_device(
                key_cache,
                value_cache,
                query_half,
                slot_mapping,
                cu_q,
                context_lens,
                block_tables,
                num_sequences,
                total_query_tokens,
                max_q,
                block_table_stride,
                physical_slots,
                num_attention_heads,
                num_kv_heads,
                request.output,
            );
        }
        let num_sequences = u32_arg("num_sequences", num_sequences)?;
        let total_q = u32_arg("total_query_tokens", total_query_tokens)?;
        let num_attention_heads = u32_arg("num_attention_heads", num_attention_heads)?;
        let num_kv_heads = u32_arg("num_kv_heads", num_kv_heads)?;
        let head_dim = u32_arg("head_dim", head_dim)?;
        let page_tokens = u32_arg("page_tokens", page_tokens_usize)?;
        let split_tokens = u32_arg("split_tokens", FLASH_SPLIT_K_TOKENS)?;
        let split_count = u32_arg("split_count", split_count_usize)?;
        let block_table_stride = u32_arg("block_table_stride", block_table_stride)?;
        let physical_slots = u32_arg("physical_slots", physical_slots)?;
        let warp_eligible = head_dim_usize <= 256 && head_dim_usize.is_multiple_of(32);
        let use_warp = matches!(
            self.config.prefill_attention,
            CudaPrefillAttentionKernel::WarpFlash
        ) && warp_eligible;
        let block_dim = 128_u32;
        let mut shared_floats = if use_warp {
            (block_dim / 32) as usize * 3 + head_dim_usize + 4
        } else if use_fa4_hdim128 {
            let nwarps = (block_dim / 32) as usize;
            FA4_HDIM128_Q_BLOCK * FA4_HDIM128_K_TILE * (nwarps + 1)
                + FA4_HDIM128_Q_BLOCK * head_dim_usize
                + FA4_HDIM128_K_TILE * head_dim_usize * 2
                + FA4_HDIM128_Q_BLOCK * head_dim_usize
                + FA4_HDIM128_Q_BLOCK * 4
                + FA4_HDIM128_K_TILE
        } else if use_halfq_block4 {
            let q_block = if use_halfq_block4_split {
                FLASH_SPLIT_Q_BLOCK
            } else {
                TILED_HALFQ_Q_BLOCK
            };
            let nwarps = (block_dim / 32) as usize;
            q_block * nwarps + (q_block * 2 + 2) * head_dim_usize + q_block * 4
        } else {
            block_dim as usize + head_dim_usize + 4
        };
        if native_query_half.is_some() && !use_halfq_block4 && !use_fa4_hdim128 {
            shared_floats += head_dim_usize;
        }
        let grid_q = if use_fa4_hdim128 {
            total_q.div_ceil(FA4_HDIM128_Q_BLOCK as u32)
        } else if use_halfq_block4_split {
            total_q.div_ceil(FLASH_SPLIT_Q_BLOCK as u32)
        } else if use_halfq_block4 {
            total_q.div_ceil(TILED_HALFQ_Q_BLOCK as u32)
        } else {
            total_q
        };
        let cfg = LaunchConfig {
            grid_dim: (num_attention_heads, grid_q, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: validate_dynamic_shared_bytes(
                "prefill_paged_varlen",
                shared_floats * std::mem::size_of::<f32>(),
            )?,
        };
        if use_halfq_block4_split {
            let Some(query_half) = query_half else {
                return Err(AegisError::InvalidPlan(
                    "split-K halfq attention requires query_half".into(),
                ));
            };
            let Some(split_scratch) = request.split_scratch.as_mut() else {
                return Err(AegisError::InvalidPlan(
                    "split-K halfq attention requires scratch buffers".into(),
                ));
            };
            unsafe {
                self.stream
                    .launch_builder(
                        &self
                            .kernels
                            .attention_prefill_paged_varlen_halfq_block4_split,
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
                    .arg(&split_tokens)
                    .arg(&split_count)
                    .arg(&block_table_stride)
                    .arg(&physical_slots)
                    .arg(&mut split_scratch.acc.slice)
                    .arg(&mut split_scratch.m.slice)
                    .arg(&mut split_scratch.l.slice)
                    .launch(LaunchConfig {
                        grid_dim: (num_attention_heads, grid_q, split_count),
                        block_dim: (block_dim, 1, 1),
                        shared_mem_bytes: (shared_floats * std::mem::size_of::<f32>()) as u32,
                    })
            }
            .map_err(map_cuda_err(
                "launch split-K paged varlen prefill attention",
            ))?;
            unsafe {
                self.stream
                    .launch_builder(
                        &self
                            .kernels
                            .attention_prefill_paged_varlen_halfq_block4_combine,
                    )
                    .arg(&split_scratch.acc.slice)
                    .arg(&split_scratch.m.slice)
                    .arg(&split_scratch.l.slice)
                    .arg(&total_q)
                    .arg(&num_attention_heads)
                    .arg(&head_dim)
                    .arg(&split_count)
                    .arg(&mut request.output.slice)
                    .launch(LaunchConfig {
                        grid_dim: (num_attention_heads, grid_q, 1),
                        block_dim: (block_dim, 1, 1),
                        shared_mem_bytes: ((FLASH_SPLIT_Q_BLOCK * head_dim_usize
                            + FLASH_SPLIT_Q_BLOCK * 4)
                            * std::mem::size_of::<f32>())
                            as u32,
                    })
            }
            .map_err(map_cuda_err(
                "launch combine split-K paged varlen prefill attention",
            ))?;
            return Ok(());
        }
        if let Some(query_half) = native_query_half {
            let kernel = if use_halfq_block4 {
                if use_fa4_hdim128 {
                    &self.kernels.attention_prefill_paged_varlen_fa4_hdim128
                } else {
                    &self.kernels.attention_prefill_paged_varlen_halfq_block4
                }
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
                    .arg(&mut request.output.slice)
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
                    .arg(&mut request.output.slice)
                    .launch(cfg)
            }
        }
        .map_err(map_cuda_err("launch paged varlen prefill attention"))?;
        Ok(())
    }
}

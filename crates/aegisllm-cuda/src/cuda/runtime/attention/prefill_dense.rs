use cudarc::driver::{LaunchConfig, PushKernelArg, sys::CUfunction_attribute_enum};

use super::*;
use crate::cuda::{
    CudaAttentionBackend, CudaPrefillAttentionKernel, DensePrefillMetadataProof, DeviceBuffer,
};
use aegisllm_base::cuda_config::CUDA_PREFILL_VARLEN_MIN_CONTEXT;
use aegisllm_base::error::{AegisError, Result};

impl CudaRuntime {
    #[allow(clippy::too_many_arguments)]
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
        if !num_attention_heads.is_multiple_of(num_kv_heads) {
            return Err(AegisError::InvalidPlan(
                "attention heads must be divisible by kv heads".into(),
            ));
        }
        let start_position = u32_arg("start_position", start_position)?;
        let batch = u32_arg("batch", batch)?;
        let num_attention_heads = u32_arg("num_attention_heads", num_attention_heads)?;
        let num_kv_heads = u32_arg("num_kv_heads", num_kv_heads)?;
        let head_dim = u32_arg("head_dim", head_dim)?;
        let cache_capacity_u32 = u32_arg(
            "cache_capacity",
            key_cache.len() / (num_kv_heads as usize * head_dim as usize),
        )?;
        let block_dim = CUDA_ATTENTION_BLOCK_DIM;
        let legacy_shared_bytes = (max_seq_len + block_dim as usize) * std::mem::size_of::<f32>();
        let selected_kernel = select_prefill_batched_kernel(
            self.config.prefill_attention,
            start_position as usize,
            head_dim as usize,
            legacy_shared_bytes,
        )?;
        let continuation = matches!(selected_kernel, PrefillBatchedKernel::Continuation);
        let cfg = LaunchConfig {
            grid_dim: (num_attention_heads, batch, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: if continuation {
                validate_dynamic_shared_bytes(
                    "prefill_dense_online",
                    (block_dim as usize + head_dim as usize + 3) * std::mem::size_of::<f32>(),
                )?
            } else {
                validate_dynamic_shared_bytes(
                    "prefill_dense_cache",
                    max_seq_len * std::mem::size_of::<f32>()
                        + block_dim as usize * std::mem::size_of::<f32>(),
                )?
            },
        };
        let cache_resident_signature = matches!(
            selected_kernel,
            PrefillBatchedKernel::CacheOnly | PrefillBatchedKernel::Warp
        );
        let kernel = match selected_kernel {
            PrefillBatchedKernel::Warp => &self.kernels.attention_prefill_batched_warp,
            PrefillBatchedKernel::Continuation => &self.kernels.attention_prefill_continuation,
            PrefillBatchedKernel::CacheOnly => &self.kernels.attention_prefill_batched,
        };
        if cache_resident_signature {
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
                    .arg(&cache_capacity_u32)
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
                    .arg(&cache_capacity_u32)
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
        query_half_ready: bool,
        split_acc: &mut DeviceBuffer<f32>,
        split_m: &mut DeviceBuffer<f32>,
        split_l: &mut DeviceBuffer<f32>,
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
            || context_lens.is_empty()
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
        if num_kv_heads == 0 || !num_attention_heads.is_multiple_of(num_kv_heads) {
            return Err(AegisError::InvalidPlan(
                "dense prefill attention heads must be divisible by kv heads".into(),
            ));
        }
        let legacy_shared_bytes = (dense_metadata.context_len()
            + CUDA_ATTENTION_BLOCK_DIM as usize)
            * std::mem::size_of::<f32>();
        // hdim=256 path (Gemma-4 sliding layers): the parametric WMMA
        // kernel handles it via the templated impl. Other hdim≠128 paths
        // still fall through to the slow paged kernel until we add their
        // optimized fa/gqa4/etc variants.
        if matches!(
            self.config.prefill_attention,
            CudaPrefillAttentionKernel::Auto
                | CudaPrefillAttentionKernel::AegisVarlen
                | CudaPrefillAttentionKernel::WarpFlash
        ) && head_dim == 256
            && batch >= DENSE_WARP_TILE_Q_BLOCK
        {
            let q_len = checked_len(
                "dense wmma hdim256 query half width",
                batch,
                checked_len(
                    "dense wmma hdim256 q width",
                    num_attention_heads,
                    head_dim,
                )?,
            )?;
            if !query_half_ready {
                self.f32_to_f16_device(query, q_len, query_half)?;
            }
            return self.attention_prefill_dense_halfq_wmma_hdim_device(
                key_cache,
                value_cache,
                query_half,
                start_position,
                batch,
                dense_metadata.context_len(),
                num_attention_heads,
                num_kv_heads,
                head_dim,
                output,
            );
        }
        if matches!(
            self.config.prefill_attention,
            CudaPrefillAttentionKernel::Auto
                | CudaPrefillAttentionKernel::AegisVarlen
                | CudaPrefillAttentionKernel::WarpFlash
        ) && head_dim == 128
            && batch >= DENSE_WARP_TILE_Q_BLOCK
        {
            let q_len = checked_len(
                "dense warp-tile query half conversion",
                batch,
                checked_len(
                    "dense warp-tile query half width",
                    num_attention_heads,
                    head_dim,
                )?,
            )?;
            if !query_half_ready {
                self.f32_to_f16_device(query, q_len, query_half)?;
            } else if query_half.len() < q_len {
                return Err(AegisError::InvalidPlan(format!(
                    "dense warp-tile q_half shape mismatch: required={} actual={}",
                    q_len,
                    query_half.len()
                )));
            }
            return if matches!(
                self.config.prefill_attention,
                CudaPrefillAttentionKernel::WarpFlash
            ) {
                self.attention_prefill_dense_halfq_warp_tile_hdim128_device(
                    key_cache,
                    value_cache,
                    query_half,
                    start_position,
                    batch,
                    dense_metadata.context_len(),
                    num_attention_heads,
                    num_kv_heads,
                    output,
                )
            } else if dense_wmma_split_k_enabled()
                && dense_metadata.context_len() >= DENSE_WMMA_SPLIT_K_TOKENS * 2
                && num_attention_heads / num_kv_heads >= DENSE_WMMA_GQA4_HEADS
                && dense_wmma_split_scratch_ready(
                    split_acc,
                    split_m,
                    split_l,
                    batch,
                    dense_metadata.context_len(),
                    num_attention_heads,
                    head_dim,
                )
            {
                self.attention_prefill_dense_halfq_wmma_hdim128_gqa4_split_device(
                    key_cache,
                    value_cache,
                    query_half,
                    split_acc,
                    split_m,
                    split_l,
                    start_position,
                    batch,
                    dense_metadata.context_len(),
                    num_attention_heads,
                    num_kv_heads,
                    output,
                )
            } else if dense_wmma_split_k_enabled()
                && dense_metadata.context_len() >= DENSE_WMMA_SPLIT_K_TOKENS * 2
                && dense_wmma_split_scratch_ready(
                    split_acc,
                    split_m,
                    split_l,
                    batch,
                    dense_metadata.context_len(),
                    num_attention_heads,
                    head_dim,
                )
            {
                self.attention_prefill_dense_halfq_wmma_hdim128_split_device(
                    key_cache,
                    value_cache,
                    query_half,
                    split_acc,
                    split_m,
                    split_l,
                    start_position,
                    batch,
                    dense_metadata.context_len(),
                    num_attention_heads,
                    num_kv_heads,
                    output,
                )
            } else if dense_wmma_cluster2_enabled() && dense_metadata.context_len() >= 1024 {
                self.attention_prefill_dense_halfq_wmma_hdim128_cluster2_device(
                    key_cache,
                    value_cache,
                    query_half,
                    start_position,
                    batch,
                    dense_metadata.context_len(),
                    num_attention_heads,
                    num_kv_heads,
                    output,
                )
            } else if dense_wmma_q32_enabled() && dense_metadata.context_len() >= 1024 {
                self.attention_prefill_dense_halfq_wmma_hdim128_q32_device(
                    key_cache,
                    value_cache,
                    query_half,
                    start_position,
                    batch,
                    dense_metadata.context_len(),
                    num_attention_heads,
                    num_kv_heads,
                    output,
                )
            } else if num_attention_heads / num_kv_heads >= DENSE_WMMA_GQA4_HEADS {
                self.attention_prefill_dense_halfq_wmma_hdim128_gqa4_device(
                    key_cache,
                    value_cache,
                    query_half,
                    start_position,
                    batch,
                    dense_metadata.context_len(),
                    num_attention_heads,
                    num_kv_heads,
                    output,
                )
            } else if dense_wmma_legacy_enabled() {
                self.attention_prefill_dense_halfq_wmma_hdim_device(
                    key_cache,
                    value_cache,
                    query_half,
                    start_position,
                    batch,
                    dense_metadata.context_len(),
                    num_attention_heads,
                    num_kv_heads,
                    head_dim,
                    output,
                )
            } else {
                self.attention_prefill_dense_halfq_wmma_hdim128_fa_device(
                    key_cache,
                    value_cache,
                    query_half,
                    start_position,
                    batch,
                    dense_metadata.context_len(),
                    num_attention_heads,
                    num_kv_heads,
                    output,
                )
            };
        }
        if matches!(
            self.config.prefill_attention,
            CudaPrefillAttentionKernel::Auto
                | CudaPrefillAttentionKernel::AegisVarlen
                | CudaPrefillAttentionKernel::WarpFlash
        ) && start_position == 0
            && head_dim.is_multiple_of(32)
            && head_dim <= 256
            && legacy_shared_bytes <= 48 * 1024
        {
            return self.attention_prefill_batched_device(
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
            );
        }
        let selected_backend =
            self.select_prefill_attention_backend(dense_metadata.context_len(), head_dim)?;
        let use_varlen_attention = match selected_backend {
            CudaAttentionBackend::AegisVarlen => {
                dense_metadata.context_len() >= CUDA_PREFILL_VARLEN_MIN_CONTEXT
            }
            CudaAttentionBackend::FlashAttention4 => {
                dense_metadata.context_len() >= CUDA_PREFILL_VARLEN_MIN_CONTEXT
            }
            CudaAttentionBackend::Reference => false,
            CudaAttentionBackend::FlashAttention2 | CudaAttentionBackend::FlashAttention3 => {
                unreachable!("FA2/FA3 should not be selected until their kernels are implemented")
            }
        };
        if use_varlen_attention {
            let q_len = checked_len(
                "dense varlen query half conversion",
                batch,
                checked_len(
                    "dense varlen query half width",
                    num_attention_heads,
                    head_dim,
                )?,
            )?;
            if !query_half_ready {
                self.f32_to_f16_device(query, q_len, query_half)?;
            } else if query_half.len() < q_len {
                return Err(AegisError::InvalidPlan(format!(
                    "dense varlen prefill q_half shape mismatch: required={} actual={}",
                    q_len,
                    query_half.len()
                )));
            }
            if matches!(selected_backend, CudaAttentionBackend::AegisVarlen)
                && query_half_ready
                && num_sequences == 1
                && head_dim <= 256
                && batch >= TILED_HALFQ_Q_BLOCK
            {
                return self.attention_prefill_dense_halfq_block4_device(
                    key_cache,
                    value_cache,
                    query_half,
                    start_position,
                    batch,
                    dense_metadata.context_len(),
                    num_attention_heads,
                    num_kv_heads,
                    head_dim,
                    output,
                );
            }
            let block_table_stride = dense_metadata
                .context_len()
                .div_ceil(FLASH_COMPAT_PAGE_TOKENS)
                .max(1);
            return self.attention_prefill_paged_varlen_device(
                key_cache,
                value_cache,
                query,
                Some(query_half),
                Some((split_acc, split_m, split_l)),
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
    pub(super) fn attention_prefill_dense_halfq_block4_device(
        &self,
        key_cache: &DeviceBuffer<u16>,
        value_cache: &DeviceBuffer<u16>,
        query_half: &DeviceBuffer<u16>,
        start_position: usize,
        batch: usize,
        context_len: usize,
        num_attention_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let q_width = checked_len("dense halfq q width", num_attention_heads, head_dim)?;
        let q_tokens = checked_len("dense halfq q tokens", batch, q_width)?;
        let kv_width = checked_len("dense halfq kv width", num_kv_heads, head_dim)?;
        let cache_len = checked_len("dense halfq kv cache", context_len, kv_width)?;
        if query_half.len() < q_tokens || output.len() < q_tokens {
            return Err(AegisError::InvalidPlan(format!(
                "dense halfq attention q/output shape mismatch: query_half={} output={} required={}",
                query_half.len(),
                output.len(),
                q_tokens
            )));
        }
        // Sliding-window layers may allocate cache to `window_size * kv_width`
        // (smaller than `context_len * kv_width`); the kernel addresses it via
        // ring-buffer slots = `pos % cache_capacity`. We only require that
        // key and value caches have the same size and are aligned to kv_width.
        let _ = cache_len;
        if key_cache.len() != value_cache.len() || key_cache.len() % kv_width != 0 || key_cache.is_empty() {
            return Err(AegisError::InvalidPlan(format!(
                "dense halfq attention kv cache shape mismatch: key_cache={} value_cache={} required={}",
                key_cache.len(),
                value_cache.len(),
                cache_len
            )));
        }
        if !num_attention_heads.is_multiple_of(num_kv_heads) {
            return Err(AegisError::InvalidPlan(
                "dense halfq attention heads must be divisible by kv heads".into(),
            ));
        }
        let q_block = TILED_HALFQ_Q_BLOCK;
        let block_dim = 64_u32;
        let nwarps = (block_dim / 32) as usize;
        let shared_floats = q_block * nwarps + (q_block * 2 + 2) * head_dim + q_block * 4;
        let cfg = LaunchConfig {
            grid_dim: (
                u32_arg("num_attention_heads", num_attention_heads)?,
                u32_arg("dense halfq q blocks", batch.div_ceil(q_block))?,
                1,
            ),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: validate_dynamic_shared_bytes(
                "prefill_dense_halfq_block4",
                shared_floats * std::mem::size_of::<f32>(),
            )?,
        };
        let start_position = u32_arg("start_position", start_position)?;
        let total_q = u32_arg("total_query_tokens", batch)?;
        let context_len = u32_arg("context_len", context_len)?;
        let num_attention_heads = u32_arg("num_attention_heads", num_attention_heads)?;
        let num_kv_heads = u32_arg("num_kv_heads", num_kv_heads)?;
        let head_dim = u32_arg("head_dim", head_dim)?;
        let cache_capacity_u32 = u32_arg(
            "cache_capacity",
            key_cache.len() / (num_kv_heads as usize * head_dim as usize),
        )?;
        unsafe {
            self.stream
                .launch_builder(&self.kernels.attention_prefill_dense_halfq_block4)
                .arg(&key_cache.slice)
                .arg(&value_cache.slice)
                .arg(&query_half.slice)
                .arg(&start_position)
                .arg(&total_q)
                .arg(&context_len)
                .arg(&num_attention_heads)
                .arg(&num_kv_heads)
                .arg(&head_dim)
                .arg(&cache_capacity_u32)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch dense halfq varlen prefill attention"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_prefill_dense_halfq_warp_tile_hdim128_device(
        &self,
        key_cache: &DeviceBuffer<u16>,
        value_cache: &DeviceBuffer<u16>,
        query_half: &DeviceBuffer<u16>,
        start_position: usize,
        batch: usize,
        context_len: usize,
        num_attention_heads: usize,
        num_kv_heads: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let head_dim = 128usize;
        let q_width = checked_len("dense warp-tile q width", num_attention_heads, head_dim)?;
        let q_tokens = checked_len("dense warp-tile q tokens", batch, q_width)?;
        let kv_width = checked_len("dense warp-tile kv width", num_kv_heads, head_dim)?;
        let cache_len = checked_len("dense warp-tile kv cache", context_len, kv_width)?;
        if query_half.len() < q_tokens || output.len() < q_tokens {
            return Err(AegisError::InvalidPlan(format!(
                "dense warp-tile attention q/output shape mismatch: query_half={} output={} required={}",
                query_half.len(),
                output.len(),
                q_tokens
            )));
        }
        // Sliding-window layers may allocate cache to `window_size * kv_width`
        // (smaller than `context_len * kv_width`); the kernel addresses it via
        // ring-buffer slots = `pos % cache_capacity`. We only require that
        // key and value caches have the same size and are aligned to kv_width.
        let _ = cache_len;
        if key_cache.len() != value_cache.len() || key_cache.len() % kv_width != 0 || key_cache.is_empty() {
            return Err(AegisError::InvalidPlan(format!(
                "dense warp-tile attention kv cache shape mismatch: key_cache={} value_cache={} required={}",
                key_cache.len(),
                value_cache.len(),
                cache_len
            )));
        }
        if !num_attention_heads.is_multiple_of(num_kv_heads) {
            return Err(AegisError::InvalidPlan(
                "dense warp-tile attention heads must be divisible by kv heads".into(),
            ));
        }
        let cfg = LaunchConfig {
            grid_dim: (
                u32_arg("num_attention_heads", num_attention_heads)?,
                u32_arg(
                    "dense warp-tile q blocks",
                    batch.div_ceil(DENSE_WARP_TILE_Q_BLOCK),
                )?,
                1,
            ),
            block_dim: (512, 1, 1),
            shared_mem_bytes: validate_dynamic_shared_bytes(
                "prefill_dense_halfq_warp_tile_hdim128",
                DENSE_WARP_TILE_K_TILE * head_dim * 2 * std::mem::size_of::<u16>(),
            )?,
        };
        let start_position = u32_arg("start_position", start_position)?;
        let total_q = u32_arg("total_query_tokens", batch)?;
        let context_len = u32_arg("context_len", context_len)?;
        let num_attention_heads = u32_arg("num_attention_heads", num_attention_heads)?;
        let num_kv_heads = u32_arg("num_kv_heads", num_kv_heads)?;
        let head_dim = u32_arg("head_dim", head_dim)?;
        let cache_capacity_u32 = u32_arg(
            "cache_capacity",
            key_cache.len() / (num_kv_heads as usize * head_dim as usize),
        )?;
        unsafe {
            self.stream
                .launch_builder(&self.kernels.attention_prefill_dense_halfq_warp_tile_hdim128)
                .arg(&key_cache.slice)
                .arg(&value_cache.slice)
                .arg(&query_half.slice)
                .arg(&start_position)
                .arg(&total_q)
                .arg(&context_len)
                .arg(&num_attention_heads)
                .arg(&num_kv_heads)
                .arg(&head_dim)
                .arg(&cache_capacity_u32)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err(
            "launch dense halfq warp-tile prefill attention",
        ))?;
        Ok(())
    }

    /// Dense WMMA prefill attention with parametric head_dim.
    /// Selects the kernel instantiation by `head_dim` (currently 128 / 256;
    /// 512 is not yet templated and falls back to the slow paged path
    /// upstream). Block size scales as `(head_dim/16) * 32` threads so the
    /// output P*V WMMA distributes one 16-column slice per warp.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_prefill_dense_halfq_wmma_hdim_device(
        &self,
        key_cache: &DeviceBuffer<u16>,
        value_cache: &DeviceBuffer<u16>,
        query_half: &DeviceBuffer<u16>,
        start_position: usize,
        batch: usize,
        context_len: usize,
        num_attention_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let kernel = match head_dim {
            128 => &self.kernels.attention_prefill_dense_halfq_wmma_hdim128,
            256 => &self.kernels.attention_prefill_dense_halfq_wmma_hdim256,
            other => return Err(AegisError::Unsupported(format!(
                "dense wmma prefill attention requires head_dim ∈ {{128, 256}}; got {other}",
            ))),
        };
        let q_width = checked_len("dense wmma q width", num_attention_heads, head_dim)?;
        let q_tokens = checked_len("dense wmma q tokens", batch, q_width)?;
        let kv_width = checked_len("dense wmma kv width", num_kv_heads, head_dim)?;
        let cache_len = checked_len("dense wmma kv cache", context_len, kv_width)?;
        if query_half.len() < q_tokens || output.len() < q_tokens {
            return Err(AegisError::InvalidPlan(format!(
                "dense wmma attention q/output shape mismatch: query_half={} output={} required={}",
                query_half.len(),
                output.len(),
                q_tokens
            )));
        }
        // Sliding-window layers may allocate cache to `window_size * kv_width`
        // (smaller than `context_len * kv_width`); the kernel addresses it via
        // ring-buffer slots = `pos % cache_capacity`. We only require that
        // key and value caches have the same size and are aligned to kv_width.
        let _ = cache_len;
        if key_cache.len() != value_cache.len() || key_cache.len() % kv_width != 0 || key_cache.is_empty() {
            return Err(AegisError::InvalidPlan(format!(
                "dense wmma attention kv cache shape mismatch: key_cache={} value_cache={} required={}",
                key_cache.len(),
                value_cache.len(),
                cache_len
            )));
        }
        if !num_attention_heads.is_multiple_of(num_kv_heads) {
            return Err(AegisError::InvalidPlan(
                "dense wmma attention heads must be divisible by kv heads".into(),
            ));
        }
        let half_values = DENSE_WMMA_Q_BLOCK * head_dim
            + 2 * DENSE_WMMA_K_TILE * head_dim
            + DENSE_WMMA_Q_BLOCK * DENSE_WMMA_K_TILE;
        let float_values = DENSE_WMMA_Q_BLOCK * DENSE_WMMA_K_TILE
            + DENSE_WMMA_Q_BLOCK * head_dim
            + DENSE_WMMA_Q_BLOCK * head_dim
            + DENSE_WMMA_Q_BLOCK * 3;
        // Block size = (head_dim/16) * 32 — one warp per output 16-col
        // slice for the P*V WMMA stage. hdim128 → 256, hdim256 → 512.
        let output_warps = (head_dim / 16) as u32;
        let block_threads = output_warps * 32;
        let cfg = LaunchConfig {
            grid_dim: (
                u32_arg("num_attention_heads", num_attention_heads)?,
                u32_arg("dense wmma q blocks", batch.div_ceil(DENSE_WMMA_Q_BLOCK))?,
                1,
            ),
            block_dim: (block_threads, 1, 1),
            // hdim=256 exceeds the default 48 KiB shared-mem cap; the
            // function has been opted into 96 KiB at load time
            // (functions.rs). Use the higher cap.
            shared_mem_bytes: super::validate_dynamic_shared_bytes_with_cap(
                "prefill_dense_halfq_wmma",
                half_values * std::mem::size_of::<u16>()
                    + float_values * std::mem::size_of::<f32>(),
                96 * 1024,
            )?,
        };
        let start_position = u32_arg("start_position", start_position)?;
        let total_q = u32_arg("total_query_tokens", batch)?;
        let context_len = u32_arg("context_len", context_len)?;
        let num_attention_heads = u32_arg("num_attention_heads", num_attention_heads)?;
        let num_kv_heads = u32_arg("num_kv_heads", num_kv_heads)?;
        let head_dim = u32_arg("head_dim", head_dim)?;
        let cache_capacity_u32 = u32_arg(
            "cache_capacity",
            key_cache.len() / (num_kv_heads as usize * head_dim as usize),
        )?;
        unsafe {
            self.stream
                .launch_builder(kernel)
                .arg(&key_cache.slice)
                .arg(&value_cache.slice)
                .arg(&query_half.slice)
                .arg(&start_position)
                .arg(&total_q)
                .arg(&context_len)
                .arg(&num_attention_heads)
                .arg(&num_kv_heads)
                .arg(&head_dim)
                .arg(&cache_capacity_u32)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch dense halfq wmma prefill attention"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_prefill_dense_halfq_wmma_hdim128_fa_device(
        &self,
        key_cache: &DeviceBuffer<u16>,
        value_cache: &DeviceBuffer<u16>,
        query_half: &DeviceBuffer<u16>,
        start_position: usize,
        batch: usize,
        context_len: usize,
        num_attention_heads: usize,
        num_kv_heads: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let head_dim = 128usize;
        let q_width = checked_len("dense fa wmma q width", num_attention_heads, head_dim)?;
        let q_tokens = checked_len("dense fa wmma q tokens", batch, q_width)?;
        let kv_width = checked_len("dense fa wmma kv width", num_kv_heads, head_dim)?;
        let cache_len = checked_len("dense fa wmma kv cache", context_len, kv_width)?;
        if query_half.len() < q_tokens || output.len() < q_tokens {
            return Err(AegisError::InvalidPlan(format!(
                "dense fa wmma attention q/output shape mismatch: query_half={} output={} required={}",
                query_half.len(),
                output.len(),
                q_tokens
            )));
        }
        // Sliding-window layers may allocate cache to `window_size * kv_width`
        // (smaller than `context_len * kv_width`); the kernel addresses it via
        // ring-buffer slots = `pos % cache_capacity`. We only require that
        // key and value caches have the same size and are aligned to kv_width.
        let _ = cache_len;
        if key_cache.len() != value_cache.len() || key_cache.len() % kv_width != 0 || key_cache.is_empty() {
            return Err(AegisError::InvalidPlan(format!(
                "dense fa wmma attention kv cache shape mismatch: key_cache={} value_cache={} required={}",
                key_cache.len(),
                value_cache.len(),
                cache_len
            )));
        }
        if !num_attention_heads.is_multiple_of(num_kv_heads) {
            return Err(AegisError::InvalidPlan(
                "dense fa wmma attention heads must be divisible by kv heads".into(),
            ));
        }
        let half_values = DENSE_WMMA_FA_Q_BLOCK * head_dim
            + 2 * DENSE_WMMA_K_TILE * head_dim
            + DENSE_WMMA_FA_Q_BLOCK * DENSE_WMMA_K_TILE;
        let float_values = DENSE_WMMA_FA_Q_BLOCK * DENSE_WMMA_K_TILE
            + DENSE_WMMA_FA_Q_BLOCK * head_dim
            + DENSE_WMMA_FA_Q_BLOCK * 3;
        let cfg = LaunchConfig {
            grid_dim: (
                u32_arg("num_attention_heads", num_attention_heads)?,
                u32_arg(
                    "dense fa wmma q blocks",
                    batch.div_ceil(DENSE_WMMA_FA_Q_BLOCK),
                )?,
                1,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: validate_dynamic_shared_bytes(
                "prefill_dense_halfq_wmma_hdim128_fa",
                half_values * std::mem::size_of::<u16>()
                    + float_values * std::mem::size_of::<f32>(),
            )?,
        };
        let start_position = u32_arg("start_position", start_position)?;
        let total_q = u32_arg("total_query_tokens", batch)?;
        let context_len = u32_arg("context_len", context_len)?;
        let num_attention_heads = u32_arg("num_attention_heads", num_attention_heads)?;
        let num_kv_heads = u32_arg("num_kv_heads", num_kv_heads)?;
        let head_dim = u32_arg("head_dim", head_dim)?;
        let cache_capacity_u32 = u32_arg(
            "cache_capacity",
            key_cache.len() / (num_kv_heads as usize * head_dim as usize),
        )?;
        unsafe {
            self.stream
                .launch_builder(&self.kernels.attention_prefill_dense_halfq_wmma_hdim128_fa)
                .arg(&key_cache.slice)
                .arg(&value_cache.slice)
                .arg(&query_half.slice)
                .arg(&start_position)
                .arg(&total_q)
                .arg(&context_len)
                .arg(&num_attention_heads)
                .arg(&num_kv_heads)
                .arg(&head_dim)
                .arg(&cache_capacity_u32)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch dense fa halfq wmma prefill attention"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_prefill_dense_halfq_wmma_hdim128_gqa4_device(
        &self,
        key_cache: &DeviceBuffer<u16>,
        value_cache: &DeviceBuffer<u16>,
        query_half: &DeviceBuffer<u16>,
        start_position: usize,
        batch: usize,
        context_len: usize,
        num_attention_heads: usize,
        num_kv_heads: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let head_dim = 128usize;
        if num_kv_heads == 0 || !num_attention_heads.is_multiple_of(num_kv_heads) {
            return Err(AegisError::InvalidPlan(
                "dense gqa4 wmma attention heads must be divisible by kv heads".into(),
            ));
        }
        let group = num_attention_heads / num_kv_heads;
        if group < DENSE_WMMA_GQA4_HEADS {
            return Err(AegisError::InvalidPlan(format!(
                "dense gqa4 wmma attention requires GQA group >= {}, got {group}",
                DENSE_WMMA_GQA4_HEADS
            )));
        }
        let group_tiles = group.div_ceil(DENSE_WMMA_GQA4_HEADS);
        let q_rows = DENSE_WMMA_GQA4_Q_TOKENS * DENSE_WMMA_GQA4_HEADS;
        let q_width = checked_len("dense gqa4 wmma q width", num_attention_heads, head_dim)?;
        let q_tokens = checked_len("dense gqa4 wmma q tokens", batch, q_width)?;
        let kv_width = checked_len("dense gqa4 wmma kv width", num_kv_heads, head_dim)?;
        let cache_len = checked_len("dense gqa4 wmma kv cache", context_len, kv_width)?;
        if query_half.len() < q_tokens || output.len() < q_tokens {
            return Err(AegisError::InvalidPlan(format!(
                "dense gqa4 wmma attention q/output shape mismatch: query_half={} output={} required={}",
                query_half.len(),
                output.len(),
                q_tokens
            )));
        }
        // Sliding-window layers may allocate cache to `window_size * kv_width`
        // (smaller than `context_len * kv_width`); the kernel addresses it via
        // ring-buffer slots = `pos % cache_capacity`. We only require that
        // key and value caches have the same size and are aligned to kv_width.
        let _ = cache_len;
        if key_cache.len() != value_cache.len() || key_cache.len() % kv_width != 0 || key_cache.is_empty() {
            return Err(AegisError::InvalidPlan(format!(
                "dense gqa4 wmma attention kv cache shape mismatch: key_cache={} value_cache={} required={}",
                key_cache.len(),
                value_cache.len(),
                cache_len
            )));
        }
        let score_stride = DENSE_WMMA_K_TILE + 8;
        let half_values = q_rows * head_dim + 2 * DENSE_WMMA_K_TILE * head_dim + q_rows * score_stride;
        let float_values = q_rows * score_stride + q_rows * (head_dim + 8) + q_rows * 3;
        let shared_mem_bytes =
            half_values * std::mem::size_of::<u16>() + float_values * std::mem::size_of::<f32>();
        self.kernels
            .attention_prefill_dense_halfq_wmma_hdim128_gqa4
            .set_attribute(
                CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                i32::try_from(shared_mem_bytes).map_err(|_| {
                    AegisError::InvalidPlan(format!(
                        "dense gqa4 padded wmma shared memory exceeds i32: {shared_mem_bytes}"
                    ))
                })?,
            )
            .map_err(map_cuda_err(
                "set gqa4 padded wmma max dynamic shared memory",
            ))?;
        let cfg = LaunchConfig {
            grid_dim: (
                u32_arg(
                    "dense gqa4 wmma kv/group blocks",
                    checked_len("dense gqa4 wmma group blocks", num_kv_heads, group_tiles)?,
                )?,
                u32_arg(
                    "dense gqa4 wmma q blocks",
                    batch.div_ceil(DENSE_WMMA_GQA4_Q_TOKENS),
                )?,
                1,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: u32_arg("dense gqa4 padded wmma shared memory", shared_mem_bytes)?,
        };
        let start_position = u32_arg("start_position", start_position)?;
        let total_q = u32_arg("total_query_tokens", batch)?;
        let context_len = u32_arg("context_len", context_len)?;
        let num_attention_heads = u32_arg("num_attention_heads", num_attention_heads)?;
        let num_kv_heads = u32_arg("num_kv_heads", num_kv_heads)?;
        let head_dim = u32_arg("head_dim", head_dim)?;
        let cache_capacity_u32 = u32_arg(
            "cache_capacity",
            key_cache.len() / (num_kv_heads as usize * head_dim as usize),
        )?;
        unsafe {
            self.stream
                .launch_builder(&self.kernels.attention_prefill_dense_halfq_wmma_hdim128_gqa4)
                .arg(&key_cache.slice)
                .arg(&value_cache.slice)
                .arg(&query_half.slice)
                .arg(&start_position)
                .arg(&total_q)
                .arg(&context_len)
                .arg(&num_attention_heads)
                .arg(&num_kv_heads)
                .arg(&head_dim)
                .arg(&cache_capacity_u32)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err(
            "launch dense gqa4 halfq wmma prefill attention",
        ))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_prefill_dense_halfq_wmma_hdim128_gqa4_split_device(
        &self,
        key_cache: &DeviceBuffer<u16>,
        value_cache: &DeviceBuffer<u16>,
        query_half: &DeviceBuffer<u16>,
        split_acc: &mut DeviceBuffer<f32>,
        split_m: &mut DeviceBuffer<f32>,
        split_l: &mut DeviceBuffer<f32>,
        start_position: usize,
        batch: usize,
        context_len: usize,
        num_attention_heads: usize,
        num_kv_heads: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let head_dim = 128usize;
        if num_kv_heads == 0 || !num_attention_heads.is_multiple_of(num_kv_heads) {
            return Err(AegisError::InvalidPlan(
                "dense gqa4 split wmma attention heads must be divisible by kv heads".into(),
            ));
        }
        let group = num_attention_heads / num_kv_heads;
        if group < DENSE_WMMA_GQA4_HEADS {
            return Err(AegisError::InvalidPlan(format!(
                "dense gqa4 split wmma attention requires GQA group >= {}, got {group}",
                DENSE_WMMA_GQA4_HEADS
            )));
        }
        let group_tiles = group.div_ceil(DENSE_WMMA_GQA4_HEADS);
        let split_count = context_len.div_ceil(DENSE_WMMA_SPLIT_K_TOKENS).max(1);
        if !dense_wmma_split_scratch_ready(
            split_acc,
            split_m,
            split_l,
            batch,
            context_len,
            num_attention_heads,
            head_dim,
        ) {
            return Err(AegisError::InvalidPlan(
                "dense gqa4 split wmma attention scratch buffers are too small".into(),
            ));
        }
        let q_rows = DENSE_WMMA_GQA4_SPLIT_Q_TOKENS * DENSE_WMMA_GQA4_HEADS;
        let q_width = checked_len(
            "dense gqa4 split wmma q width",
            num_attention_heads,
            head_dim,
        )?;
        let q_tokens = checked_len("dense gqa4 split wmma q tokens", batch, q_width)?;
        let kv_width = checked_len("dense gqa4 split wmma kv width", num_kv_heads, head_dim)?;
        let cache_len = checked_len("dense gqa4 split wmma kv cache", context_len, kv_width)?;
        if query_half.len() < q_tokens || output.len() < q_tokens {
            return Err(AegisError::InvalidPlan(format!(
                "dense gqa4 split wmma attention q/output shape mismatch: query_half={} output={} required={}",
                query_half.len(),
                output.len(),
                q_tokens
            )));
        }
        // Sliding-window layers may allocate cache to `window_size * kv_width`
        // (smaller than `context_len * kv_width`); the kernel addresses it via
        // ring-buffer slots = `pos % cache_capacity`. We only require that
        // key and value caches have the same size and are aligned to kv_width.
        let _ = cache_len;
        if key_cache.len() != value_cache.len() || key_cache.len() % kv_width != 0 || key_cache.is_empty() {
            return Err(AegisError::InvalidPlan(format!(
                "dense gqa4 split wmma attention kv cache shape mismatch: key_cache={} value_cache={} required={}",
                key_cache.len(),
                value_cache.len(),
                cache_len
            )));
        }
        let half_values =
            q_rows * head_dim + 2 * DENSE_WMMA_K_TILE * head_dim + q_rows * DENSE_WMMA_K_TILE;
        let float_values = q_rows * DENSE_WMMA_K_TILE + q_rows * head_dim + q_rows * 3;
        let shared_mem_bytes = validate_dynamic_shared_bytes(
            "prefill_dense_halfq_wmma_hdim128_gqa4_split",
            half_values * std::mem::size_of::<u16>() + float_values * std::mem::size_of::<f32>(),
        )?;
        let start_position = u32_arg("start_position", start_position)?;
        let total_q = u32_arg("total_query_tokens", batch)?;
        let context_len = u32_arg("context_len", context_len)?;
        let num_attention_heads_u32 = u32_arg("num_attention_heads", num_attention_heads)?;
        let num_kv_heads = u32_arg("num_kv_heads", num_kv_heads)?;
        let head_dim = u32_arg("head_dim", head_dim)?;
        let cache_capacity_u32 = u32_arg(
            "cache_capacity",
            key_cache.len() / (num_kv_heads as usize * head_dim as usize),
        )?;
        let split_tokens = u32_arg(
            "dense gqa4 split wmma split tokens",
            DENSE_WMMA_SPLIT_K_TOKENS,
        )?;
        let split_count_u32 = u32_arg("dense gqa4 split wmma split count", split_count)?;
        unsafe {
            self.stream
                .launch_builder(
                    &self
                        .kernels
                        .attention_prefill_dense_halfq_wmma_hdim128_gqa4_split,
                )
                .arg(&key_cache.slice)
                .arg(&value_cache.slice)
                .arg(&query_half.slice)
                .arg(&start_position)
                .arg(&total_q)
                .arg(&context_len)
                .arg(&num_attention_heads_u32)
                .arg(&num_kv_heads)
                .arg(&head_dim)
                .arg(&cache_capacity_u32)
                .arg(&split_tokens)
                .arg(&split_count_u32)
                .arg(&mut split_acc.slice)
                .arg(&mut split_m.slice)
                .arg(&mut split_l.slice)
                .launch(LaunchConfig {
                    grid_dim: (
                        u32_arg(
                            "dense gqa4 split wmma kv/group blocks",
                            checked_len(
                                "dense gqa4 split wmma group blocks",
                                num_attention_heads / group,
                                group_tiles,
                            )?,
                        )?,
                        u32_arg(
                            "dense gqa4 split wmma q blocks",
                            batch.div_ceil(DENSE_WMMA_GQA4_SPLIT_Q_TOKENS),
                        )?,
                        split_count_u32,
                    ),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes,
                })
        }
        .map_err(map_cuda_err(
            "launch split-K dense gqa4 halfq wmma prefill attention",
        ))?;
        let grid_q = batch.div_ceil(DENSE_WMMA_Q_BLOCK);
        unsafe {
            self.stream
                .launch_builder(
                    &self
                        .kernels
                        .attention_prefill_dense_halfq_wmma_hdim128_combine,
                )
                .arg(&split_acc.slice)
                .arg(&split_m.slice)
                .arg(&split_l.slice)
                .arg(&total_q)
                .arg(&num_attention_heads_u32)
                .arg(&head_dim)
                .arg(&split_count_u32)
                .arg(&mut output.slice)
                .launch(LaunchConfig {
                    grid_dim: (
                        num_attention_heads_u32,
                        u32_arg("dense gqa4 split wmma combine q blocks", grid_q)?,
                        1,
                    ),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: validate_dynamic_shared_bytes(
                        "prefill_dense_halfq_wmma_hdim128_gqa4_combine",
                        (DENSE_WMMA_Q_BLOCK * 128 + DENSE_WMMA_Q_BLOCK * 3)
                            * std::mem::size_of::<f32>(),
                    )?,
                })
        }
        .map_err(map_cuda_err(
            "launch combine split-K dense gqa4 halfq wmma prefill attention",
        ))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_prefill_dense_halfq_wmma_hdim128_cluster2_device(
        &self,
        key_cache: &DeviceBuffer<u16>,
        value_cache: &DeviceBuffer<u16>,
        query_half: &DeviceBuffer<u16>,
        start_position: usize,
        batch: usize,
        context_len: usize,
        num_attention_heads: usize,
        num_kv_heads: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let head_dim = 128usize;
        let cluster_blocks = 2usize;
        let q_width = checked_len("dense cluster2 wmma q width", num_attention_heads, head_dim)?;
        let q_tokens = checked_len("dense cluster2 wmma q tokens", batch, q_width)?;
        let kv_width = checked_len("dense cluster2 wmma kv width", num_kv_heads, head_dim)?;
        let cache_len = checked_len("dense cluster2 wmma kv cache", context_len, kv_width)?;
        if query_half.len() < q_tokens || output.len() < q_tokens {
            return Err(AegisError::InvalidPlan(format!(
                "dense cluster2 wmma attention q/output shape mismatch: query_half={} output={} required={}",
                query_half.len(),
                output.len(),
                q_tokens
            )));
        }
        // Sliding-window layers may allocate cache to `window_size * kv_width`
        // (smaller than `context_len * kv_width`); the kernel addresses it via
        // ring-buffer slots = `pos % cache_capacity`. We only require that
        // key and value caches have the same size and are aligned to kv_width.
        let _ = cache_len;
        if key_cache.len() != value_cache.len() || key_cache.len() % kv_width != 0 || key_cache.is_empty() {
            return Err(AegisError::InvalidPlan(format!(
                "dense cluster2 wmma attention kv cache shape mismatch: key_cache={} value_cache={} required={}",
                key_cache.len(),
                value_cache.len(),
                cache_len
            )));
        }
        if !num_attention_heads.is_multiple_of(num_kv_heads) {
            return Err(AegisError::InvalidPlan(
                "dense cluster2 wmma attention heads must be divisible by kv heads".into(),
            ));
        }
        let half_values = DENSE_WMMA_FA_Q_BLOCK * head_dim
            + 2 * DENSE_WMMA_K_TILE * head_dim
            + DENSE_WMMA_FA_Q_BLOCK * DENSE_WMMA_K_TILE;
        let float_values = DENSE_WMMA_FA_Q_BLOCK * DENSE_WMMA_K_TILE
            + DENSE_WMMA_FA_Q_BLOCK * head_dim
            + DENSE_WMMA_FA_Q_BLOCK * 3;
        let shared_mem_bytes = validate_dynamic_shared_bytes(
            "prefill_dense_halfq_wmma_hdim128_cluster2",
            half_values * std::mem::size_of::<u16>() + float_values * std::mem::size_of::<f32>(),
        )?;
        let start_position = u32_arg("start_position", start_position)?;
        let total_q = u32_arg("total_query_tokens", batch)?;
        let context_len = u32_arg("context_len", context_len)?;
        let num_attention_heads_u32 = u32_arg("num_attention_heads", num_attention_heads)?;
        let num_kv_heads = u32_arg("num_kv_heads", num_kv_heads)?;
        let head_dim = u32_arg("head_dim", head_dim)?;
        let cache_capacity_u32 = u32_arg(
            "cache_capacity",
            key_cache.len() / (num_kv_heads as usize * head_dim as usize),
        )?;
        unsafe {
            self.stream
                .launch_builder(
                    &self
                        .kernels
                        .attention_prefill_dense_halfq_wmma_hdim128_cluster2,
                )
                .arg(&key_cache.slice)
                .arg(&value_cache.slice)
                .arg(&query_half.slice)
                .arg(&start_position)
                .arg(&total_q)
                .arg(&context_len)
                .arg(&num_attention_heads_u32)
                .arg(&num_kv_heads)
                .arg(&head_dim)
                .arg(&cache_capacity_u32)
                .arg(&mut output.slice)
                .launch(LaunchConfig {
                    grid_dim: (
                        u32_arg(
                            "dense cluster2 wmma head blocks",
                            checked_len(
                                "dense cluster2 wmma head cluster blocks",
                                num_attention_heads,
                                cluster_blocks,
                            )?,
                        )?,
                        u32_arg(
                            "dense cluster2 wmma q blocks",
                            batch.div_ceil(DENSE_WMMA_FA_Q_BLOCK),
                        )?,
                        1,
                    ),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes,
                })
        }
        .map_err(map_cuda_err(
            "launch dense cluster2 halfq wmma prefill attention",
        ))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_prefill_dense_halfq_wmma_hdim128_q32_device(
        &self,
        key_cache: &DeviceBuffer<u16>,
        value_cache: &DeviceBuffer<u16>,
        query_half: &DeviceBuffer<u16>,
        start_position: usize,
        batch: usize,
        context_len: usize,
        num_attention_heads: usize,
        num_kv_heads: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let head_dim = 128usize;
        let q_width = checked_len("dense q32 wmma q width", num_attention_heads, head_dim)?;
        let q_tokens = checked_len("dense q32 wmma q tokens", batch, q_width)?;
        let kv_width = checked_len("dense q32 wmma kv width", num_kv_heads, head_dim)?;
        let cache_len = checked_len("dense q32 wmma kv cache", context_len, kv_width)?;
        if query_half.len() < q_tokens || output.len() < q_tokens {
            return Err(AegisError::InvalidPlan(format!(
                "dense q32 wmma attention q/output shape mismatch: query_half={} output={} required={}",
                query_half.len(),
                output.len(),
                q_tokens
            )));
        }
        // Sliding-window layers may allocate cache to `window_size * kv_width`
        // (smaller than `context_len * kv_width`); the kernel addresses it via
        // ring-buffer slots = `pos % cache_capacity`. We only require that
        // key and value caches have the same size and are aligned to kv_width.
        let _ = cache_len;
        if key_cache.len() != value_cache.len() || key_cache.len() % kv_width != 0 || key_cache.is_empty() {
            return Err(AegisError::InvalidPlan(format!(
                "dense q32 wmma attention kv cache shape mismatch: key_cache={} value_cache={} required={}",
                key_cache.len(),
                value_cache.len(),
                cache_len
            )));
        }
        let half_values = DENSE_WMMA_Q32_BLOCK * head_dim
            + 2 * DENSE_WMMA_K_TILE * head_dim
            + DENSE_WMMA_Q32_BLOCK * DENSE_WMMA_K_TILE;
        let float_values = DENSE_WMMA_Q32_BLOCK * DENSE_WMMA_K_TILE
            + DENSE_WMMA_Q32_BLOCK * head_dim
            + DENSE_WMMA_Q32_BLOCK * head_dim
            + DENSE_WMMA_Q32_BLOCK * 3;
        let shared_mem_bytes =
            half_values * std::mem::size_of::<u16>() + float_values * std::mem::size_of::<f32>();
        self.kernels
            .attention_prefill_dense_halfq_wmma_hdim128_q32
            .set_attribute(
                CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                i32::try_from(shared_mem_bytes).map_err(|_| {
                    AegisError::InvalidPlan(format!(
                        "dense q32 wmma shared memory exceeds i32: {shared_mem_bytes}"
                    ))
                })?,
            )
            .map_err(map_cuda_err("set q32 wmma max dynamic shared memory"))?;
        let start_position = u32_arg("start_position", start_position)?;
        let total_q = u32_arg("total_query_tokens", batch)?;
        let context_len = u32_arg("context_len", context_len)?;
        let num_attention_heads = u32_arg("num_attention_heads", num_attention_heads)?;
        let num_kv_heads = u32_arg("num_kv_heads", num_kv_heads)?;
        let head_dim = u32_arg("head_dim", head_dim)?;
        let cache_capacity_u32 = u32_arg(
            "cache_capacity",
            key_cache.len() / (num_kv_heads as usize * head_dim as usize),
        )?;
        unsafe {
            self.stream
                .launch_builder(&self.kernels.attention_prefill_dense_halfq_wmma_hdim128_q32)
                .arg(&key_cache.slice)
                .arg(&value_cache.slice)
                .arg(&query_half.slice)
                .arg(&start_position)
                .arg(&total_q)
                .arg(&context_len)
                .arg(&num_attention_heads)
                .arg(&num_kv_heads)
                .arg(&head_dim)
                .arg(&cache_capacity_u32)
                .arg(&mut output.slice)
                .launch(LaunchConfig {
                    grid_dim: (
                        num_attention_heads,
                        u32_arg(
                            "dense q32 wmma q blocks",
                            batch.div_ceil(DENSE_WMMA_Q32_BLOCK),
                        )?,
                        1,
                    ),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: u32::try_from(shared_mem_bytes).map_err(|_| {
                        AegisError::InvalidPlan(format!(
                            "dense q32 wmma shared memory exceeds u32: {shared_mem_bytes}"
                        ))
                    })?,
                })
        }
        .map_err(map_cuda_err(
            "launch dense q32 halfq wmma prefill attention",
        ))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_prefill_dense_halfq_wmma_hdim128_split_device(
        &self,
        key_cache: &DeviceBuffer<u16>,
        value_cache: &DeviceBuffer<u16>,
        query_half: &DeviceBuffer<u16>,
        split_acc: &mut DeviceBuffer<f32>,
        split_m: &mut DeviceBuffer<f32>,
        split_l: &mut DeviceBuffer<f32>,
        start_position: usize,
        batch: usize,
        context_len: usize,
        num_attention_heads: usize,
        num_kv_heads: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let head_dim = 128usize;
        let q_width = checked_len("dense split wmma q width", num_attention_heads, head_dim)?;
        let q_tokens = checked_len("dense split wmma q tokens", batch, q_width)?;
        let kv_width = checked_len("dense split wmma kv width", num_kv_heads, head_dim)?;
        let cache_len = checked_len("dense split wmma kv cache", context_len, kv_width)?;
        if query_half.len() < q_tokens || output.len() < q_tokens {
            return Err(AegisError::InvalidPlan(format!(
                "dense split wmma attention q/output shape mismatch: query_half={} output={} required={}",
                query_half.len(),
                output.len(),
                q_tokens
            )));
        }
        // Sliding-window layers may allocate cache to `window_size * kv_width`
        // (smaller than `context_len * kv_width`); the kernel addresses it via
        // ring-buffer slots = `pos % cache_capacity`. We only require that
        // key and value caches have the same size and are aligned to kv_width.
        let _ = cache_len;
        if key_cache.len() != value_cache.len() || key_cache.len() % kv_width != 0 || key_cache.is_empty() {
            return Err(AegisError::InvalidPlan(format!(
                "dense split wmma attention kv cache shape mismatch: key_cache={} value_cache={} required={}",
                key_cache.len(),
                value_cache.len(),
                cache_len
            )));
        }
        if !num_attention_heads.is_multiple_of(num_kv_heads) {
            return Err(AegisError::InvalidPlan(
                "dense split wmma attention heads must be divisible by kv heads".into(),
            ));
        }
        let split_count = context_len.div_ceil(DENSE_WMMA_SPLIT_K_TOKENS).max(1);
        if !dense_wmma_split_scratch_ready(
            split_acc,
            split_m,
            split_l,
            batch,
            context_len,
            num_attention_heads,
            head_dim,
        ) {
            return Err(AegisError::InvalidPlan(
                "dense split wmma attention scratch buffers are too small".into(),
            ));
        }
        let half_values = DENSE_WMMA_Q_BLOCK * head_dim
            + 2 * DENSE_WMMA_K_TILE * head_dim
            + DENSE_WMMA_Q_BLOCK * DENSE_WMMA_K_TILE;
        let float_values = DENSE_WMMA_Q_BLOCK * DENSE_WMMA_K_TILE
            + DENSE_WMMA_Q_BLOCK * head_dim
            + DENSE_WMMA_Q_BLOCK * head_dim
            + DENSE_WMMA_Q_BLOCK * 3;
        let shared_mem_bytes = validate_dynamic_shared_bytes(
            "prefill_dense_halfq_wmma_hdim128_split",
            half_values * std::mem::size_of::<u16>() + float_values * std::mem::size_of::<f32>(),
        )?;
        let grid_q = batch.div_ceil(DENSE_WMMA_Q_BLOCK);
        let start_position = u32_arg("start_position", start_position)?;
        let total_q = u32_arg("total_query_tokens", batch)?;
        let context_len = u32_arg("context_len", context_len)?;
        let num_attention_heads_u32 = u32_arg("num_attention_heads", num_attention_heads)?;
        let num_kv_heads = u32_arg("num_kv_heads", num_kv_heads)?;
        let head_dim = u32_arg("head_dim", head_dim)?;
        let cache_capacity_u32 = u32_arg(
            "cache_capacity",
            key_cache.len() / (num_kv_heads as usize * head_dim as usize),
        )?;
        let split_tokens = u32_arg("dense split wmma split tokens", DENSE_WMMA_SPLIT_K_TOKENS)?;
        let split_count_u32 = u32_arg("dense split wmma split count", split_count)?;
        unsafe {
            self.stream
                .launch_builder(
                    &self
                        .kernels
                        .attention_prefill_dense_halfq_wmma_hdim128_split,
                )
                .arg(&key_cache.slice)
                .arg(&value_cache.slice)
                .arg(&query_half.slice)
                .arg(&start_position)
                .arg(&total_q)
                .arg(&context_len)
                .arg(&num_attention_heads_u32)
                .arg(&num_kv_heads)
                .arg(&head_dim)
                .arg(&cache_capacity_u32)
                .arg(&split_tokens)
                .arg(&split_count_u32)
                .arg(&mut split_acc.slice)
                .arg(&mut split_m.slice)
                .arg(&mut split_l.slice)
                .launch(LaunchConfig {
                    grid_dim: (
                        num_attention_heads_u32,
                        u32_arg("dense split wmma q blocks", grid_q)?,
                        split_count_u32,
                    ),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes,
                })
        }
        .map_err(map_cuda_err(
            "launch split-K dense halfq wmma prefill attention",
        ))?;
        unsafe {
            self.stream
                .launch_builder(
                    &self
                        .kernels
                        .attention_prefill_dense_halfq_wmma_hdim128_combine,
                )
                .arg(&split_acc.slice)
                .arg(&split_m.slice)
                .arg(&split_l.slice)
                .arg(&total_q)
                .arg(&num_attention_heads_u32)
                .arg(&head_dim)
                .arg(&split_count_u32)
                .arg(&mut output.slice)
                .launch(LaunchConfig {
                    grid_dim: (
                        num_attention_heads_u32,
                        u32_arg("dense split wmma combine q blocks", grid_q)?,
                        1,
                    ),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: validate_dynamic_shared_bytes(
                        "prefill_dense_halfq_wmma_hdim128_combine",
                        (DENSE_WMMA_Q_BLOCK * 128 + DENSE_WMMA_Q_BLOCK * 3)
                            * std::mem::size_of::<f32>(),
                    )?,
                })
        }
        .map_err(map_cuda_err(
            "launch combine split-K dense halfq wmma prefill attention",
        ))?;
        Ok(())
    }
}

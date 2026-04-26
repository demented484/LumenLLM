use cudarc::driver::{LaunchConfig, PushKernelArg, sys::CUfunction_attribute_enum};

use super::{CudaRuntime, map_cuda_err};
use crate::cuda::{
    CudaAttentionBackend, CudaAttentionRequest, CudaAttentionSplitScratch,
    CudaPrefillAttentionKernel, DensePrefillMetadataProof, DeviceBuffer,
    config::CUDA_PREFILL_VARLEN_MIN_CONTEXT,
};
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

fn validate_dynamic_shared_bytes(kernel: &str, bytes: usize) -> Result<u32> {
    if bytes > 48 * 1024 {
        return Err(AegisError::InvalidPlan(format!(
            "CUDA kernel `{kernel}` requires {bytes} bytes of dynamic shared memory, exceeding the conservative 48KiB launch limit"
        )));
    }
    Ok(bytes as u32)
}

const FLASH_COMPAT_PAGE_TOKENS: usize = 256;
const FLASH_SPLIT_K_TOKENS: usize = 256;
const FLASH_SPLIT_Q_BLOCK: usize = 4;
const TILED_HALFQ_Q_BLOCK: usize = 4;
const DENSE_WARP_TILE_Q_BLOCK: usize = 16;
const DENSE_WARP_TILE_K_TILE: usize = 32;
const DENSE_WMMA_Q_BLOCK: usize = 16;
const DENSE_WMMA_FA_Q_BLOCK: usize = 16;
const DENSE_WMMA_GQA4_Q_TOKENS: usize = 8;
const DENSE_WMMA_GQA4_HEADS: usize = 4;
const DENSE_WMMA_Q32_BLOCK: usize = 32;
const DENSE_WMMA_K_TILE: usize = 32;
const DENSE_WMMA_SPLIT_K_TOKENS: usize = 2048;
const FA4_HDIM128_Q_BLOCK: usize = 8;
const FA4_HDIM128_K_TILE: usize = 32;
const CUDA_ATTENTION_BLOCK_DIM: u32 = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrefillBatchedKernel {
    CacheOnly,
    Continuation,
    Warp,
}

fn select_prefill_batched_kernel(
    config: CudaPrefillAttentionKernel,
    start_position: usize,
    head_dim: usize,
    legacy_shared_bytes: usize,
) -> Result<PrefillBatchedKernel> {
    let warp_eligible = start_position == 0 && head_dim % 32 == 0 && head_dim <= 256;
    if matches!(
        config,
        CudaPrefillAttentionKernel::Auto
            | CudaPrefillAttentionKernel::AegisVarlen
            | CudaPrefillAttentionKernel::WarpFlash
    ) && warp_eligible
    {
        return Ok(PrefillBatchedKernel::Warp);
    }
    if matches!(config, CudaPrefillAttentionKernel::Continuation) {
        return Ok(PrefillBatchedKernel::Continuation);
    }
    if matches!(
        config,
        CudaPrefillAttentionKernel::Off | CudaPrefillAttentionKernel::Reference
    ) && legacy_shared_bytes > 48 * 1024
    {
        return Err(AegisError::InvalidPlan(format!(
            "CUDA reference prefill attention requires {} bytes of dynamic shared memory; use cuda.prefill-attention=aegis-varlen, auto, or continuation for long prefixes",
            legacy_shared_bytes
        )));
    }
    if !matches!(
        config,
        CudaPrefillAttentionKernel::Off | CudaPrefillAttentionKernel::Reference
    ) && legacy_shared_bytes > 48 * 1024
    {
        return Ok(PrefillBatchedKernel::Continuation);
    }
    Ok(PrefillBatchedKernel::CacheOnly)
}

impl CudaRuntime {
    fn select_prefill_attention_backend(
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
        let block_dim = CUDA_ATTENTION_BLOCK_DIM;
        let legacy_shared_bytes = seq_len as usize * std::mem::size_of::<f32>()
            + block_dim as usize * std::mem::size_of::<f32>();
        let streaming = legacy_shared_bytes > 48 * 1024;
        let cfg = LaunchConfig {
            grid_dim: (num_attention_heads, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: if streaming {
                validate_dynamic_shared_bytes(
                    "attention_decode_streaming",
                    (block_dim as usize + head_dim as usize + 3) * std::mem::size_of::<f32>(),
                )?
            } else {
                validate_dynamic_shared_bytes("attention_decode", legacy_shared_bytes)?
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
        let legacy_shared_bytes = (dense_metadata.context_len()
            + CUDA_ATTENTION_BLOCK_DIM as usize)
            * std::mem::size_of::<f32>();
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
                self.attention_prefill_dense_halfq_wmma_hdim128_device(
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
            && head_dim % 32 == 0
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
    fn attention_prefill_dense_halfq_block4_device(
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
        if key_cache.len() < cache_len || value_cache.len() < cache_len {
            return Err(AegisError::InvalidPlan(format!(
                "dense halfq attention kv cache shape mismatch: key_cache={} value_cache={} required={}",
                key_cache.len(),
                value_cache.len(),
                cache_len
            )));
        }
        if num_attention_heads % num_kv_heads != 0 {
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
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch dense halfq varlen prefill attention"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_prefill_dense_halfq_warp_tile_hdim128_device(
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
        if key_cache.len() < cache_len || value_cache.len() < cache_len {
            return Err(AegisError::InvalidPlan(format!(
                "dense warp-tile attention kv cache shape mismatch: key_cache={} value_cache={} required={}",
                key_cache.len(),
                value_cache.len(),
                cache_len
            )));
        }
        if num_attention_heads % num_kv_heads != 0 {
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
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err(
            "launch dense halfq warp-tile prefill attention",
        ))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_prefill_dense_halfq_wmma_hdim128_device(
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
        if key_cache.len() < cache_len || value_cache.len() < cache_len {
            return Err(AegisError::InvalidPlan(format!(
                "dense wmma attention kv cache shape mismatch: key_cache={} value_cache={} required={}",
                key_cache.len(),
                value_cache.len(),
                cache_len
            )));
        }
        if num_attention_heads % num_kv_heads != 0 {
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
        let cfg = LaunchConfig {
            grid_dim: (
                u32_arg("num_attention_heads", num_attention_heads)?,
                u32_arg("dense wmma q blocks", batch.div_ceil(DENSE_WMMA_Q_BLOCK))?,
                1,
            ),
            block_dim: (256, 1, 1),
            shared_mem_bytes: validate_dynamic_shared_bytes(
                "prefill_dense_halfq_wmma_hdim128",
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
        unsafe {
            self.stream
                .launch_builder(&self.kernels.attention_prefill_dense_halfq_wmma_hdim128)
                .arg(&key_cache.slice)
                .arg(&value_cache.slice)
                .arg(&query_half.slice)
                .arg(&start_position)
                .arg(&total_q)
                .arg(&context_len)
                .arg(&num_attention_heads)
                .arg(&num_kv_heads)
                .arg(&head_dim)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch dense halfq wmma prefill attention"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_prefill_dense_halfq_wmma_hdim128_fa_device(
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
        if key_cache.len() < cache_len || value_cache.len() < cache_len {
            return Err(AegisError::InvalidPlan(format!(
                "dense fa wmma attention kv cache shape mismatch: key_cache={} value_cache={} required={}",
                key_cache.len(),
                value_cache.len(),
                cache_len
            )));
        }
        if num_attention_heads % num_kv_heads != 0 {
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
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch dense fa halfq wmma prefill attention"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_prefill_dense_halfq_wmma_hdim128_gqa4_device(
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
        if key_cache.len() < cache_len || value_cache.len() < cache_len {
            return Err(AegisError::InvalidPlan(format!(
                "dense gqa4 wmma attention kv cache shape mismatch: key_cache={} value_cache={} required={}",
                key_cache.len(),
                value_cache.len(),
                cache_len
            )));
        }
        if num_attention_heads % num_kv_heads != 0 {
            return Err(AegisError::InvalidPlan(
                "dense gqa4 wmma attention heads must be divisible by kv heads".into(),
            ));
        }
        let half_values =
            q_rows * head_dim + 2 * DENSE_WMMA_K_TILE * head_dim + q_rows * DENSE_WMMA_K_TILE;
        let float_values = q_rows * DENSE_WMMA_K_TILE + q_rows * head_dim + q_rows * 3;
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
            shared_mem_bytes: validate_dynamic_shared_bytes(
                "prefill_dense_halfq_wmma_hdim128_gqa4",
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
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err(
            "launch dense gqa4 halfq wmma prefill attention",
        ))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_prefill_dense_halfq_wmma_hdim128_gqa4_split_device(
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
        let q_rows = DENSE_WMMA_GQA4_Q_TOKENS * DENSE_WMMA_GQA4_HEADS;
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
        if key_cache.len() < cache_len || value_cache.len() < cache_len {
            return Err(AegisError::InvalidPlan(format!(
                "dense gqa4 split wmma attention kv cache shape mismatch: key_cache={} value_cache={} required={}",
                key_cache.len(),
                value_cache.len(),
                cache_len
            )));
        }
        if num_attention_heads % num_kv_heads != 0 {
            return Err(AegisError::InvalidPlan(
                "dense gqa4 split wmma attention heads must be divisible by kv heads".into(),
            ));
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
                            batch.div_ceil(DENSE_WMMA_GQA4_Q_TOKENS),
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
    fn attention_prefill_dense_halfq_wmma_hdim128_cluster2_device(
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
        if key_cache.len() < cache_len || value_cache.len() < cache_len {
            return Err(AegisError::InvalidPlan(format!(
                "dense cluster2 wmma attention kv cache shape mismatch: key_cache={} value_cache={} required={}",
                key_cache.len(),
                value_cache.len(),
                cache_len
            )));
        }
        if num_attention_heads % num_kv_heads != 0 {
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
    fn attention_prefill_dense_halfq_wmma_hdim128_q32_device(
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
        if key_cache.len() < cache_len || value_cache.len() < cache_len {
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
    fn attention_prefill_dense_halfq_wmma_hdim128_split_device(
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
        if key_cache.len() < cache_len || value_cache.len() < cache_len {
            return Err(AegisError::InvalidPlan(format!(
                "dense split wmma attention kv cache shape mismatch: key_cache={} value_cache={} required={}",
                key_cache.len(),
                value_cache.len(),
                cache_len
            )));
        }
        if num_attention_heads % num_kv_heads != 0 {
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
        let warp_eligible = head_dim_usize <= 256 && head_dim_usize % 32 == 0;
        let use_warp = matches!(
            self.config.prefill_attention,
            CudaPrefillAttentionKernel::WarpFlash
        ) && warp_eligible;
        let block_dim = if use_fa4_hdim128 {
            128_u32
        } else if use_halfq_block4 {
            128_u32
        } else {
            128_u32
        };
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

fn dense_wmma_split_k_enabled() -> bool {
    std::env::var_os("AEGISLLM_CUDA_EXPERIMENTAL_SPLIT_K_ATTENTION").is_some()
}

fn dense_wmma_q32_enabled() -> bool {
    std::env::var_os("AEGISLLM_CUDA_EXPERIMENTAL_PERSISTENT_ATTENTION").is_some()
}

fn dense_wmma_legacy_enabled() -> bool {
    std::env::var_os("AEGISLLM_CUDA_LEGACY_WMMA_ATTENTION").is_some()
}

fn dense_wmma_cluster2_enabled() -> bool {
    std::env::var_os("AEGISLLM_CUDA_EXPERIMENTAL_CLUSTER_ATTENTION").is_some()
}

fn dense_wmma_split_scratch_ready(
    split_acc: &DeviceBuffer<f32>,
    split_m: &DeviceBuffer<f32>,
    split_l: &DeviceBuffer<f32>,
    batch: usize,
    context_len: usize,
    num_attention_heads: usize,
    head_dim: usize,
) -> bool {
    let split_count = context_len.div_ceil(DENSE_WMMA_SPLIT_K_TOKENS).max(1);
    let rows = batch
        .div_ceil(DENSE_WMMA_Q_BLOCK)
        .checked_mul(num_attention_heads)
        .and_then(|value| value.checked_mul(split_count))
        .and_then(|value| value.checked_mul(DENSE_WMMA_Q_BLOCK));
    let Some(rows) = rows else {
        return false;
    };
    let Some(acc_len) = rows.checked_mul(head_dim) else {
        return false;
    };
    split_acc.len() >= acc_len && split_m.len() >= rows && split_l.len() >= rows
}

#[cfg(test)]
mod tests {
    use super::{PrefillBatchedKernel, select_prefill_batched_kernel};
    use crate::cuda::CudaPrefillAttentionKernel;

    #[test]
    fn reference_prefill_rejects_oversized_shared_memory() {
        let error = select_prefill_batched_kernel(
            CudaPrefillAttentionKernel::Reference,
            0,
            128,
            256 * 1024,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("cuda.prefill-attention=aegis-varlen")
        );
    }

    #[test]
    fn warp_flash_still_prefers_warp_kernel_when_eligible() {
        assert_eq!(
            select_prefill_batched_kernel(CudaPrefillAttentionKernel::WarpFlash, 0, 128, 1024)
                .unwrap(),
            PrefillBatchedKernel::Warp
        );
    }

    #[test]
    fn varlen_first_prefill_uses_warp_specialization_when_dense() {
        assert_eq!(
            select_prefill_batched_kernel(CudaPrefillAttentionKernel::AegisVarlen, 0, 128, 1024)
                .unwrap(),
            PrefillBatchedKernel::Warp
        );
    }
}

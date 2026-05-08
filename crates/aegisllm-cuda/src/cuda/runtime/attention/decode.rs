use cudarc::driver::{LaunchConfig, PushKernelArg};

use super::*;
use crate::cuda::DeviceBuffer;
use aegisllm_base::error::{AegisError, Result};

use crate::cuda::{CUDA_GRAPH_ATTN_MAX_SEQ_LEN, DECODE_MAX_CHUNK_LEN, DECODE_SPLIT_K};

impl CudaRuntime {
    /// Like `attention_decode_device` but reads `seq_len` from a device buffer (index 0).
    /// Uses the non-streaming kernel with pre-allocated shared memory for CUDA Graph replay.
    /// The caller must ensure that the actual seq_len (at replay time) ≤ CUDA_GRAPH_ATTN_MAX_SEQ_LEN.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_decode_ptr_device(
        &self,
        key_cache: &DeviceBuffer<u16>,
        value_cache: &DeviceBuffer<u16>,
        query: &DeviceBuffer<f32>,
        p_seq_len: &DeviceBuffer<u32>,
        num_attention_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        window_size: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if num_kv_heads == 0 || num_attention_heads == 0 || head_dim == 0 {
            return Err(AegisError::InvalidPlan(format!(
                "attention_ptr dimensions must be non-zero: q_heads={} kv_heads={} head_dim={}",
                num_attention_heads, num_kv_heads, head_dim
            )));
        }
        let query_len = checked_len("decode_ptr query", num_attention_heads, head_dim)?;
        if query.len() < query_len || output.len() < query_len {
            return Err(AegisError::InvalidPlan(
                "attention_ptr query/output shape mismatch".into(),
            ));
        }
        if !num_attention_heads.is_multiple_of(num_kv_heads) {
            return Err(AegisError::InvalidPlan(
                "attention_ptr heads must be divisible by kv heads".into(),
            ));
        }
        let num_attention_heads = u32_arg("num_attention_heads", num_attention_heads)?;
        let num_kv_heads = u32_arg("num_kv_heads", num_kv_heads)?;
        let head_dim = u32_arg("head_dim", head_dim)?;
        let window_size = u32_arg("window_size", window_size)?;
        let block_dim = CUDA_ATTENTION_BLOCK_DIM;
        // Pre-allocate shared memory for the worst-case seq_len at graph capture time.
        // The kernel reads the actual seq_len from *p_seq_len at runtime.
        let max_shared_bytes = validate_dynamic_shared_bytes(
            "attention_decode_ptr",
            (CUDA_GRAPH_ATTN_MAX_SEQ_LEN + block_dim as usize) * std::mem::size_of::<f32>(),
        )?;
        let cfg = LaunchConfig {
            grid_dim: (num_attention_heads, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: max_shared_bytes,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.attention_ptr)
                .arg(&key_cache.slice)
                .arg(&value_cache.slice)
                .arg(&query.slice)
                .arg(&p_seq_len.slice)
                .arg(&num_attention_heads)
                .arg(&num_kv_heads)
                .arg(&head_dim)
                .arg(&window_size)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch attention_decode_ptr"))?;
        Ok(())
    }

    /// FlashDecoding split-K decode attention for CUDA Graph replay.
    /// Replaces `attention_decode_ptr_device` with 2 kernels:
    ///   1. split: grid (heads, DECODE_SPLIT_K), tiny shared mem, much higher occupancy
    ///   2. combine: grid (heads, 1), no shared mem
    ///
    /// `partial_acc/m/l` must be pre-allocated in CudaScratch (see state.rs).
    ///
    /// `seq_len_hint` is the host-known seq_len (kv_position + 1) for the
    /// current decode step. The kernel re-reads the authoritative value from
    /// `*p_seq_len` at runtime; the hint is used solely to size the dynamic
    /// shared memory allocation, which must hold `scores[chunk_len]` where
    /// `chunk_len = ceil(seq_len / DECODE_SPLIT_K)`. For seq_len ≤
    /// `CUDA_GRAPH_ATTN_MAX_SEQ_LEN` the captured-graph hot path uses the
    /// fixed `DECODE_MAX_CHUNK_LEN` allocation; for longer seqs (eager
    /// path only — graph capture is skipped above 8k) we widen shared mem
    /// to fit. Without this fix, decode at seq_len > 8k overruns the
    /// fixed 512-slot `scores` array into adjacent shared memory and
    /// (above ~16k) past the dynamic-shared allocation entirely, surfacing
    /// as `CUDA_ERROR_ILLEGAL_ADDRESS` on the next kernel launch.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_decode_split_ptr_device(
        &self,
        key_cache: &DeviceBuffer<u16>,
        value_cache: &DeviceBuffer<u16>,
        query: &DeviceBuffer<f32>,
        p_seq_len: &DeviceBuffer<u32>,
        num_attention_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        window_size: usize,
        seq_len_hint: usize,
        partial_acc: &mut DeviceBuffer<f32>,
        partial_m: &mut DeviceBuffer<f32>,
        partial_l: &mut DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if num_kv_heads == 0 || num_attention_heads == 0 || head_dim == 0 {
            return Err(AegisError::InvalidPlan(format!(
                "attention_split_ptr dimensions must be non-zero: q_heads={} kv_heads={} head_dim={}",
                num_attention_heads, num_kv_heads, head_dim
            )));
        }
        let query_len = checked_len("decode_split query", num_attention_heads, head_dim)?;
        if query.len() < query_len || output.len() < query_len {
            return Err(AegisError::InvalidPlan(format!(
                "attention_split_ptr query/output too small: query={} output={} expected_min={}",
                query.len(), output.len(), query_len
            )));
        }
        let partial_len = checked_len("decode_split partial", num_attention_heads, DECODE_SPLIT_K)?;
        let partial_acc_len = checked_len("decode_split partial_acc", partial_len, head_dim)?;
        if partial_acc.len() < partial_acc_len || partial_m.len() < partial_len || partial_l.len() < partial_len {
            return Err(AegisError::InvalidPlan(format!(
                "attention_split_ptr partial buffers too small: acc={} m={} l={} required acc={} ml={}",
                partial_acc.len(), partial_m.len(), partial_l.len(), partial_acc_len, partial_len
            )));
        }
        if !num_attention_heads.is_multiple_of(num_kv_heads) {
            return Err(AegisError::InvalidPlan(
                "attention_split_ptr heads must be divisible by kv heads".into(),
            ));
        }
        let num_heads_u32 = u32_arg("num_attention_heads", num_attention_heads)?;
        let num_kv_heads_u32 = u32_arg("num_kv_heads", num_kv_heads)?;
        let head_dim_u32 = u32_arg("head_dim", head_dim)?;
        let split_k_u32 = u32_arg("split_k", DECODE_SPLIT_K)?;
        // For the captured-graph hot path (seq_len ≤ CUDA_GRAPH_ATTN_MAX_SEQ_LEN),
        // chunk_len ≤ DECODE_MAX_CHUNK_LEN and we use the fixed allocation.
        // For longer seqs we widen `max_chunk_len` to fit the actual chunk size
        // so `scores[pos]` writes stay in-bounds. The kernel uses this both as
        // the shared-mem layout offset for warp_partial/vsum AND as the
        // implicit upper bound on `pos`.
        let chunk_len_for_seq = seq_len_hint.div_ceil(DECODE_SPLIT_K).max(1);
        let effective_max_chunk_len = DECODE_MAX_CHUNK_LEN.max(chunk_len_for_seq);
        let max_chunk_len_u32 = u32_arg("max_chunk_len", effective_max_chunk_len)?;
        let window_size_u32 = u32_arg("window_size", window_size)?;
        // Cache capacity (in tokens) is inferred from the key buffer size:
        // sliding-window layers will pass a smaller cache, the kernel uses
        // `slot = pos % cache_capacity` to index it. Global layers pass
        // `cache_capacity == context_size`, which makes the wrap a no-op for
        // any `pos < context_size`.
        let cache_capacity = key_cache.len() / (num_kv_heads * head_dim);
        let cache_capacity_u32 = u32_arg("cache_capacity", cache_capacity)?;
        let block_dim = CUDA_ATTENTION_BLOCK_DIM;
        // Shared memory layout: scores[effective_max_chunk_len] +
        // warp_partial[4] + vsum[4*head_dim], all f32.
        // Capped at the kernel's MAX_DYNAMIC_SHARED_SIZE_BYTES opt-in
        // (96 KiB) at load time.
        let split_shared_bytes_usize =
            (effective_max_chunk_len + 4 + 4 * head_dim) * std::mem::size_of::<f32>();
        let split_shared_bytes = super::validate_dynamic_shared_bytes_with_cap(
            "attention_decode_ptr_split",
            split_shared_bytes_usize,
            96 * 1024,
        )?;
        let split_cfg = LaunchConfig {
            grid_dim: (num_heads_u32, split_k_u32, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: split_shared_bytes,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.attention_decode_ptr_split)
                .arg(&key_cache.slice)
                .arg(&value_cache.slice)
                .arg(&query.slice)
                .arg(&p_seq_len.slice)
                .arg(&num_heads_u32)
                .arg(&num_kv_heads_u32)
                .arg(&head_dim_u32)
                .arg(&split_k_u32)
                .arg(&max_chunk_len_u32)
                .arg(&window_size_u32)
                .arg(&cache_capacity_u32)
                .arg(&mut partial_acc.slice)
                .arg(&mut partial_m.slice)
                .arg(&mut partial_l.slice)
                .launch(split_cfg)
        }
        .map_err(map_cuda_err("launch attention_decode_ptr_split"))?;

        let combine_cfg = LaunchConfig {
            grid_dim: (num_heads_u32, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.attention_decode_ptr_combine)
                .arg(&partial_acc.slice)
                .arg(&partial_m.slice)
                .arg(&partial_l.slice)
                .arg(&head_dim_u32)
                .arg(&split_k_u32)
                .arg(&mut output.slice)
                .launch(combine_cfg)
        }
        .map_err(map_cuda_err("launch attention_decode_ptr_combine"))?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // FP8 E4M3 attention decode methods
    // -----------------------------------------------------------------------

    /// FP8 KV cache decode attention — reads `seq_len` from device memory
    /// (CUDA Graph friendly).  Shares the combine kernel with the F16 split-K path.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_decode_ptr_fp8_device(
        &self,
        key_cache: &DeviceBuffer<u8>,
        value_cache: &DeviceBuffer<u8>,
        query: &DeviceBuffer<f32>,
        p_seq_len: &DeviceBuffer<u32>,
        num_attention_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        window_size: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if num_kv_heads == 0 || num_attention_heads == 0 || head_dim == 0 {
            return Err(AegisError::InvalidPlan(format!(
                "attention_ptr_fp8: non-zero dims required q_heads={} kv_heads={} head_dim={}",
                num_attention_heads, num_kv_heads, head_dim
            )));
        }
        let query_len = checked_len("fp8_ptr query", num_attention_heads, head_dim)?;
        if query.len() < query_len || output.len() < query_len {
            return Err(AegisError::InvalidPlan("attention_ptr_fp8 query/output shape mismatch".into()));
        }
        if !num_attention_heads.is_multiple_of(num_kv_heads) {
            return Err(AegisError::InvalidPlan("attention_ptr_fp8 heads not divisible by kv heads".into()));
        }
        let num_attention_heads = u32_arg("num_attention_heads", num_attention_heads)?;
        let num_kv_heads        = u32_arg("num_kv_heads", num_kv_heads)?;
        let head_dim            = u32_arg("head_dim", head_dim)?;
        let window_size         = u32_arg("window_size", window_size)?;
        let block_dim           = CUDA_ATTENTION_BLOCK_DIM;
        let max_shared_bytes = validate_dynamic_shared_bytes(
            "attention_decode_ptr_fp8",
            (CUDA_GRAPH_ATTN_MAX_SEQ_LEN + block_dim as usize) * std::mem::size_of::<f32>(),
        )?;
        let cfg = LaunchConfig {
            grid_dim: (num_attention_heads, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: max_shared_bytes,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.attention_decode_ptr_fp8)
                .arg(&key_cache.slice)
                .arg(&value_cache.slice)
                .arg(&query.slice)
                .arg(&p_seq_len.slice)
                .arg(&num_attention_heads)
                .arg(&num_kv_heads)
                .arg(&head_dim)
                .arg(&window_size)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch attention_decode_ptr_fp8"))?;
        Ok(())
    }

    /// FP8 KV FlashDecoding split-K for CUDA Graph.  The combine pass is
    /// shared with the F16 path (it never reads the KV cache).
    ///
    /// `seq_len_hint`: see `attention_decode_split_ptr_device` — used solely
    /// to size dynamic shared memory for `scores[chunk_len]`. Required for
    /// correct execution above `CUDA_GRAPH_ATTN_MAX_SEQ_LEN`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_decode_split_ptr_fp8_device(
        &self,
        key_cache: &DeviceBuffer<u8>,
        value_cache: &DeviceBuffer<u8>,
        query: &DeviceBuffer<f32>,
        p_seq_len: &DeviceBuffer<u32>,
        num_attention_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        window_size: usize,
        seq_len_hint: usize,
        partial_acc: &mut DeviceBuffer<f32>,
        partial_m: &mut DeviceBuffer<f32>,
        partial_l: &mut DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if num_kv_heads == 0 || num_attention_heads == 0 || head_dim == 0 {
            return Err(AegisError::InvalidPlan(format!(
                "attention_split_fp8: non-zero dims required q_heads={} kv_heads={} head_dim={}",
                num_attention_heads, num_kv_heads, head_dim
            )));
        }
        let query_len = checked_len("fp8_split query", num_attention_heads, head_dim)?;
        if query.len() < query_len || output.len() < query_len {
            return Err(AegisError::InvalidPlan("attention_split_fp8 query/output shape mismatch".into()));
        }
        let partial_len     = checked_len("fp8_split partial", num_attention_heads, DECODE_SPLIT_K)?;
        let partial_acc_len = checked_len("fp8_split partial_acc", partial_len, head_dim)?;
        if partial_acc.len() < partial_acc_len || partial_m.len() < partial_len || partial_l.len() < partial_len {
            return Err(AegisError::InvalidPlan(format!(
                "attention_split_fp8 partial buffers too small: acc={} m={} l={} req_acc={} req_ml={}",
                partial_acc.len(), partial_m.len(), partial_l.len(), partial_acc_len, partial_len
            )));
        }
        if !num_attention_heads.is_multiple_of(num_kv_heads) {
            return Err(AegisError::InvalidPlan("attention_split_fp8 heads not divisible by kv heads".into()));
        }
        let num_heads_u32    = u32_arg("num_attention_heads", num_attention_heads)?;
        let num_kv_heads_u32 = u32_arg("num_kv_heads", num_kv_heads)?;
        let head_dim_u32     = u32_arg("head_dim", head_dim)?;
        let split_k_u32      = u32_arg("split_k", DECODE_SPLIT_K)?;
        // Dynamic-size shared mem to fit `scores[chunk_len]` at long seqs.
        // See `attention_decode_split_ptr_device` for full rationale.
        let chunk_len_for_seq = seq_len_hint.div_ceil(DECODE_SPLIT_K).max(1);
        let effective_max_chunk_len = DECODE_MAX_CHUNK_LEN.max(chunk_len_for_seq);
        let max_chunk_len_u32 = u32_arg("max_chunk_len", effective_max_chunk_len)?;
        let window_size_u32  = u32_arg("window_size", window_size)?;
        let cache_capacity = key_cache.len() / (num_kv_heads * head_dim);
        let cache_capacity_u32 = u32_arg("cache_capacity", cache_capacity)?;
        let block_dim        = CUDA_ATTENTION_BLOCK_DIM;
        let split_shared_bytes_usize =
            (effective_max_chunk_len + 4 + 4 * head_dim) * std::mem::size_of::<f32>();
        let split_shared_bytes = super::validate_dynamic_shared_bytes_with_cap(
            "attention_decode_ptr_split_fp8",
            split_shared_bytes_usize,
            96 * 1024,
        )?;
        let split_cfg = LaunchConfig {
            grid_dim: (num_heads_u32, split_k_u32, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: split_shared_bytes,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.attention_decode_ptr_split_fp8)
                .arg(&key_cache.slice)
                .arg(&value_cache.slice)
                .arg(&query.slice)
                .arg(&p_seq_len.slice)
                .arg(&num_heads_u32)
                .arg(&num_kv_heads_u32)
                .arg(&head_dim_u32)
                .arg(&split_k_u32)
                .arg(&max_chunk_len_u32)
                .arg(&window_size_u32)
                .arg(&cache_capacity_u32)
                .arg(&mut partial_acc.slice)
                .arg(&mut partial_m.slice)
                .arg(&mut partial_l.slice)
                .launch(split_cfg)
        }
        .map_err(map_cuda_err("launch attention_decode_ptr_split_fp8"))?;

        let combine_cfg = LaunchConfig {
            grid_dim: (num_heads_u32, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.attention_decode_ptr_combine)
                .arg(&partial_acc.slice)
                .arg(&partial_m.slice)
                .arg(&partial_l.slice)
                .arg(&head_dim_u32)
                .arg(&split_k_u32)
                .arg(&mut output.slice)
                .launch(combine_cfg)
        }
        .map_err(map_cuda_err("launch attention_decode_ptr_combine (fp8)"))?;
        Ok(())
    }

    /// FP8 KV non-graph decode — uses simple or streaming kernel depending on
    /// shared-memory budget, same heuristic as the F16 `attention_decode_device`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention_decode_fp8_device(
        &self,
        key_cache: &DeviceBuffer<u8>,
        value_cache: &DeviceBuffer<u8>,
        query: &DeviceBuffer<f32>,
        seq_len: usize,
        num_attention_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        window_size: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if num_kv_heads == 0 || num_attention_heads == 0 || head_dim == 0 {
            return Err(AegisError::InvalidPlan(format!(
                "attention_fp8 dimensions non-zero required q_heads={} kv_heads={} head_dim={}",
                num_attention_heads, num_kv_heads, head_dim
            )));
        }
        let query_len = checked_len("fp8 decode query", num_attention_heads, head_dim)?;
        let kv_width  = checked_len("fp8 decode kv width", num_kv_heads, head_dim)?;
        if query.len() < query_len || output.len() < query_len {
            return Err(AegisError::InvalidPlan("attention_fp8 query/output shape mismatch".into()));
        }
        if seq_len == 0
            || key_cache.len() < checked_len("fp8 key cache", seq_len, kv_width)?
            || value_cache.len() < checked_len("fp8 value cache", seq_len, kv_width)?
        {
            return Err(AegisError::InvalidPlan(format!(
                "attention_fp8 cache shape mismatch: seq={} kv_width={} key={} value={}",
                seq_len, kv_width, key_cache.len(), value_cache.len()
            )));
        }
        if !num_attention_heads.is_multiple_of(num_kv_heads) {
            return Err(AegisError::InvalidPlan("attention_fp8 heads not divisible by kv heads".into()));
        }
        let seq_len_u32         = u32_arg("seq_len", seq_len)?;
        let num_attention_heads = u32_arg("num_attention_heads", num_attention_heads)?;
        let num_kv_heads        = u32_arg("num_kv_heads", num_kv_heads)?;
        let head_dim_u32        = u32_arg("head_dim", head_dim)?;
        let window_size         = u32_arg("window_size", window_size)?;
        let block_dim           = CUDA_ATTENTION_BLOCK_DIM;
        let legacy_shared = seq_len * std::mem::size_of::<f32>()
            + block_dim as usize * std::mem::size_of::<f32>();
        let streaming = legacy_shared > 48 * 1024;
        let cfg = LaunchConfig {
            grid_dim: (num_attention_heads, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: if streaming {
                validate_dynamic_shared_bytes(
                    "attention_decode_streaming_fp8",
                    (block_dim as usize + head_dim + 3) * std::mem::size_of::<f32>(),
                )?
            } else {
                validate_dynamic_shared_bytes("attention_decode_fp8", legacy_shared)?
            },
        };
        let kernel = if streaming {
            &self.kernels.attention_decode_streaming_fp8
        } else {
            &self.kernels.attention_decode_fp8
        };
        unsafe {
            self.stream
                .launch_builder(kernel)
                .arg(&key_cache.slice)
                .arg(&value_cache.slice)
                .arg(&query.slice)
                .arg(&seq_len_u32)
                .arg(&num_attention_heads)
                .arg(&num_kv_heads)
                .arg(&head_dim_u32)
                .arg(&window_size)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch attention_decode_fp8"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention_decode_device(
        &self,
        key_cache: &DeviceBuffer<u16>,
        value_cache: &DeviceBuffer<u16>,
        query: &DeviceBuffer<f32>,
        seq_len: usize,
        num_attention_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        window_size: usize,
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
        if query.len() < query_len || output.len() < query_len {
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
        if !num_attention_heads.is_multiple_of(num_kv_heads) {
            return Err(AegisError::InvalidPlan(
                "attention heads must be divisible by kv heads".into(),
            ));
        }
        let seq_len = u32_arg("seq_len", seq_len)?;
        let num_attention_heads = u32_arg("num_attention_heads", num_attention_heads)?;
        let num_kv_heads = u32_arg("num_kv_heads", num_kv_heads)?;
        let head_dim = u32_arg("head_dim", head_dim)?;
        let window_size = u32_arg("window_size", window_size)?;
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
                .arg(&window_size)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch attention decode"))?;
        Ok(())
    }

}

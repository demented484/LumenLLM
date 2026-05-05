use cudarc::driver::{LaunchConfig, PinnedHostSlice, PushKernelArg};

use super::{CudaRuntime, ceil_div, map_cuda_err};
use crate::cuda::{DensePrefillMetadataProof, DeviceBuffer, DeviceRopeConfig};
use aegisllm_base::error::{AegisError, Result};

fn u32_arg(name: &str, value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        AegisError::InvalidPlan(format!(
            "CUDA KV argument {name} exceeds u32 range: {value}"
        ))
    })
}

fn checked_len(label: &str, lhs: usize, rhs: usize) -> Result<usize> {
    lhs.checked_mul(rhs).ok_or_else(|| {
        AegisError::InvalidPlan(format!("CUDA KV {label} length overflow: {lhs} * {rhs}"))
    })
}

fn checked_sum(label: &str, lhs: usize, rhs: usize) -> Result<usize> {
    lhs.checked_add(rhs).ok_or_else(|| {
        AegisError::InvalidPlan(format!("CUDA KV {label} length overflow: {lhs} + {rhs}"))
    })
}

impl CudaRuntime {
    pub fn build_dense_prefill_metadata_device(
        &self,
        start_position: usize,
        batch: usize,
        positions: &mut DeviceBuffer<u32>,
        slot_mapping: &mut DeviceBuffer<u32>,
    ) -> Result<()> {
        let end = checked_sum("dense metadata end", start_position, batch)?;
        if batch == 0 || positions.len() < batch || slot_mapping.len() < batch {
            return Err(AegisError::InvalidPlan(format!(
                "dense prefill metadata buffers too small: start={} batch={} positions={} slots={}",
                start_position,
                batch,
                positions.len(),
                slot_mapping.len()
            )));
        }
        let start_position = u32_arg("start_position", start_position)?;
        let _ = u32_arg("end_position", end)?;
        let batch_u32 = u32_arg("batch", batch)?;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(batch_u32, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.build_dense_prefill_metadata)
                .arg(&start_position)
                .arg(&batch_u32)
                .arg(&mut positions.slice)
                .arg(&mut slot_mapping.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch dense prefill metadata builder"))?;
        Ok(())
    }

    /// Like `store_kv_device` but reads `position` from a device buffer at index 0.
    /// Use this inside CUDA Graph captures so `position` can vary per replay.
    #[allow(clippy::too_many_arguments)]
    pub fn store_kv_ptr_device(
        &self,
        key_cache: &mut DeviceBuffer<u16>,
        value_cache: &mut DeviceBuffer<u16>,
        key: &DeviceBuffer<f32>,
        value: &DeviceBuffer<f32>,
        p_position: &DeviceBuffer<u32>,
        kv_width: usize,
        context_size: usize,
    ) -> Result<()> {
        let cache_len = checked_len("cache", context_size, kv_width)?;
        // `key`/`value` are scratch buffers sized for the largest layer's kv_width
        // (Gemma 4 has heterogeneous global vs sliding kv head counts), so we only
        // require the buffer to hold at least `kv_width` elements.
        if key.len() < kv_width || value.len() < kv_width {
            return Err(AegisError::InvalidPlan(
                "kv_store_ptr vector shape mismatch".into(),
            ));
        }
        if key_cache.len() != cache_len || value_cache.len() != cache_len {
            return Err(AegisError::InvalidPlan(format!(
                "kv_store_ptr cache shape mismatch: key_cache={} value_cache={} context={} width={}",
                key_cache.len(),
                value_cache.len(),
                context_size,
                kv_width
            )));
        }
        let width = u32_arg("kv_width", kv_width)?;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(width, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.kv_store_ptr)
                .arg(&mut key_cache.slice)
                .arg(&mut value_cache.slice)
                .arg(&key.slice)
                .arg(&value.slice)
                .arg(&p_position.slice)
                .arg(&width)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch kv_store_ptr"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn store_kv_device(
        &self,
        key_cache: &mut DeviceBuffer<u16>,
        value_cache: &mut DeviceBuffer<u16>,
        key: &DeviceBuffer<f32>,
        value: &DeviceBuffer<f32>,
        position: usize,
        kv_width: usize,
        context_size: usize,
    ) -> Result<()> {
        let cache_len = checked_len("cache", context_size, kv_width)?;
        if key.len() != kv_width || value.len() != kv_width {
            return Err(AegisError::InvalidPlan(
                "kv store vector shape mismatch".into(),
            ));
        }
        if key_cache.len() != cache_len
            || value_cache.len() != cache_len
            || position >= context_size
        {
            return Err(AegisError::InvalidPlan(format!(
                "kv cache shape mismatch: key_cache={} value_cache={} position={} context={} width={}",
                key_cache.len(),
                value_cache.len(),
                position,
                context_size,
                kv_width
            )));
        }
        let position = u32_arg("position", position)?;
        let width = u32_arg("kv_width", kv_width)?;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(width, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.kv_store)
                .arg(&mut key_cache.slice)
                .arg(&mut value_cache.slice)
                .arg(&key.slice)
                .arg(&value.slice)
                .arg(&position)
                .arg(&width)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch kv store"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn store_kv_batched_device(
        &self,
        key_cache: &mut DeviceBuffer<u16>,
        value_cache: &mut DeviceBuffer<u16>,
        key: &DeviceBuffer<f32>,
        value: &DeviceBuffer<f32>,
        start_position: usize,
        batch: usize,
        kv_width: usize,
        context_size: usize,
    ) -> Result<()> {
        let vector_len = checked_len("batched key/value", batch, kv_width)?;
        let cache_len = checked_len("batched cache", context_size, kv_width)?;
        let end_position = checked_sum("batched end", start_position, batch)?;
        if key.len() < vector_len || value.len() < vector_len {
            return Err(AegisError::InvalidPlan(
                "batched kv store vector shape mismatch".into(),
            ));
        }
        if key_cache.len() != cache_len
            || value_cache.len() != cache_len
            || end_position > context_size
        {
            return Err(AegisError::InvalidPlan(format!(
                "batched kv cache shape mismatch: key_cache={} value_cache={} start={} batch={} context={} width={}",
                key_cache.len(),
                value_cache.len(),
                start_position,
                batch,
                context_size,
                kv_width
            )));
        }
        let start_position = u32_arg("start_position", start_position)?;
        let batch = u32_arg("batch", batch)?;
        let width = u32_arg("kv_width", kv_width)?;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(width, 256), batch, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.kv_store_batched)
                .arg(&mut key_cache.slice)
                .arg(&mut value_cache.slice)
                .arg(&key.slice)
                .arg(&value.slice)
                .arg(&start_position)
                .arg(&batch)
                .arg(&width)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch batched kv store"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn store_kv_slots_batched_device(
        &self,
        key_cache: &mut DeviceBuffer<u16>,
        value_cache: &mut DeviceBuffer<u16>,
        key: &DeviceBuffer<f32>,
        value: &DeviceBuffer<f32>,
        slot_mapping: &DeviceBuffer<u32>,
        batch: usize,
        kv_width: usize,
        context_size: usize,
        dense_metadata: DensePrefillMetadataProof,
    ) -> Result<()> {
        let vector_len = checked_len("slot-mapped key/value", batch, kv_width)?;
        let cache_len = checked_len("slot-mapped cache", context_size, kv_width)?;
        if dense_metadata.batch() != batch || dense_metadata.context_len() > context_size {
            return Err(AegisError::InvalidPlan(format!(
                "slot-mapped kv store requires matching dense identity proof: batch={} context={} proof_batch={} proof_context={}",
                batch,
                context_size,
                dense_metadata.batch(),
                dense_metadata.context_len()
            )));
        }
        if key.len() < vector_len || value.len() < vector_len {
            return Err(AegisError::InvalidPlan(
                "slot-mapped kv store vector shape mismatch".into(),
            ));
        }
        if slot_mapping.len() < batch
            || key_cache.len() != cache_len
            || value_cache.len() != cache_len
        {
            return Err(AegisError::InvalidPlan(format!(
                "slot-mapped kv cache shape mismatch: key_cache={} value_cache={} slots={} batch={} context={} width={}",
                key_cache.len(),
                value_cache.len(),
                slot_mapping.len(),
                batch,
                context_size,
                kv_width
            )));
        }
        let batch = u32_arg("batch", batch)?;
        let width = u32_arg("kv_width", kv_width)?;
        let context_size = u32_arg("context_size", context_size)?;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(width, 256), batch, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.kv_store_slots_batched)
                .arg(&mut key_cache.slice)
                .arg(&mut value_cache.slice)
                .arg(&key.slice)
                .arg(&value.slice)
                .arg(&slot_mapping.slice)
                .arg(&batch)
                .arg(&width)
                .arg(&context_size)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch slot-mapped batched kv store"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn store_kv_slots_batched_rope_key_device(
        &self,
        key_cache: &mut DeviceBuffer<u16>,
        value_cache: &mut DeviceBuffer<u16>,
        key: &mut DeviceBuffer<f32>,
        value: &DeviceBuffer<f32>,
        positions: &DeviceBuffer<u32>,
        slot_mapping: &DeviceBuffer<u32>,
        batch: usize,
        num_heads: usize,
        head_dim: usize,
        context_size: usize,
        dense_metadata: DensePrefillMetadataProof,
        rope: DeviceRopeConfig,
    ) -> Result<()> {
        // The kernel handles each kv element with its own thread — head_dim is not
        // capped by the threadblock size. Only require evenness for the rope split.
        if num_heads == 0 || head_dim == 0 || !head_dim.is_multiple_of(2) {
            return Err(AegisError::InvalidPlan(format!(
                "slot-mapped roped kv store requires non-zero heads and even head_dim: heads={} head_dim={}",
                num_heads, head_dim
            )));
        }
        let kv_width = checked_len("slot-mapped roped kv width", num_heads, head_dim)?;
        let vector_len = checked_len("slot-mapped roped key/value", batch, kv_width)?;
        let cache_len = checked_len("slot-mapped roped cache", context_size, kv_width)?;
        if dense_metadata.batch() != batch || dense_metadata.context_len() > context_size {
            return Err(AegisError::InvalidPlan(format!(
                "slot-mapped roped kv store requires matching dense identity proof: batch={} context={} proof_batch={} proof_context={}",
                batch,
                context_size,
                dense_metadata.batch(),
                dense_metadata.context_len()
            )));
        }
        if key.len() < vector_len || value.len() < vector_len {
            return Err(AegisError::InvalidPlan(
                "slot-mapped roped kv store vector shape mismatch".into(),
            ));
        }
        if positions.len() < batch
            || slot_mapping.len() < batch
            || key_cache.len() != cache_len
            || value_cache.len() != cache_len
        {
            return Err(AegisError::InvalidPlan(format!(
                "slot-mapped roped kv cache shape mismatch: key_cache={} value_cache={} key={} value={} positions={} slots={} batch={} context={} width={}",
                key_cache.len(),
                value_cache.len(),
                key.len(),
                value.len(),
                positions.len(),
                slot_mapping.len(),
                batch,
                context_size,
                kv_width
            )));
        }
        let batch = u32_arg("batch", batch)?;
        let num_heads = u32_arg("num_heads", num_heads)?;
        let head_dim = u32_arg("head_dim", head_dim)?;
        let kv_width = u32_arg("kv_width", kv_width)?;
        let context_size = u32_arg("context_size", context_size)?;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(kv_width, 256), batch, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.rope_kv_store_slots_batched)
                .arg(&mut key_cache.slice)
                .arg(&mut value_cache.slice)
                .arg(&mut key.slice)
                .arg(&value.slice)
                .arg(&positions.slice)
                .arg(&slot_mapping.slice)
                .arg(&batch)
                .arg(&num_heads)
                .arg(&head_dim)
                .arg(&context_size)
                .arg(&rope.theta)
                .arg(&rope.factor)
                .arg(&rope.low_freq_factor)
                .arg(&rope.high_freq_factor)
                .arg(&rope.original_max_position_embeddings)
                .arg(&rope.partial_dim)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch slot-mapped roped batched kv store"))?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // FP8 E4M3 KV store methods
    // -----------------------------------------------------------------------
    // These mirror the F16 variants above but use `DeviceBuffer<u8>` caches
    // and the `aegis_kv_store_fp8*` CUDA kernels.  Conversion accuracy is
    // round-to-nearest; tolerance for the long-context-32k gate is 5e-3.

    #[allow(clippy::too_many_arguments)]
    pub fn store_kv_fp8_ptr_device(
        &self,
        key_cache: &mut DeviceBuffer<u8>,
        value_cache: &mut DeviceBuffer<u8>,
        key: &DeviceBuffer<f32>,
        value: &DeviceBuffer<f32>,
        p_position: &DeviceBuffer<u32>,
        kv_width: usize,
        context_size: usize,
    ) -> Result<()> {
        let cache_len = checked_len("fp8 cache", context_size, kv_width)?;
        if key.len() != kv_width || value.len() != kv_width {
            return Err(AegisError::InvalidPlan("kv_store_fp8_ptr vector shape mismatch".into()));
        }
        if key_cache.len() != cache_len || value_cache.len() != cache_len {
            return Err(AegisError::InvalidPlan(format!(
                "kv_store_fp8_ptr cache shape mismatch: key={} value={} context={} width={}",
                key_cache.len(), value_cache.len(), context_size, kv_width
            )));
        }
        let width = u32_arg("kv_width", kv_width)?;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(width, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.kv_store_fp8_ptr)
                .arg(&mut key_cache.slice)
                .arg(&mut value_cache.slice)
                .arg(&key.slice)
                .arg(&value.slice)
                .arg(&p_position.slice)
                .arg(&width)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch kv_store_fp8_ptr"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn store_kv_fp8_device(
        &self,
        key_cache: &mut DeviceBuffer<u8>,
        value_cache: &mut DeviceBuffer<u8>,
        key: &DeviceBuffer<f32>,
        value: &DeviceBuffer<f32>,
        position: usize,
        kv_width: usize,
        context_size: usize,
    ) -> Result<()> {
        let cache_len = checked_len("fp8 cache", context_size, kv_width)?;
        if key.len() != kv_width || value.len() != kv_width {
            return Err(AegisError::InvalidPlan("kv_store_fp8 vector shape mismatch".into()));
        }
        if key_cache.len() != cache_len || value_cache.len() != cache_len || position >= context_size {
            return Err(AegisError::InvalidPlan(format!(
                "kv_store_fp8 cache shape mismatch: pos={} context={} width={}",
                position, context_size, kv_width
            )));
        }
        let position = u32_arg("position", position)?;
        let width = u32_arg("kv_width", kv_width)?;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(width, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.kv_store_fp8)
                .arg(&mut key_cache.slice)
                .arg(&mut value_cache.slice)
                .arg(&key.slice)
                .arg(&value.slice)
                .arg(&position)
                .arg(&width)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch kv_store_fp8"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn store_kv_fp8_batched_device(
        &self,
        key_cache: &mut DeviceBuffer<u8>,
        value_cache: &mut DeviceBuffer<u8>,
        key: &DeviceBuffer<f32>,
        value: &DeviceBuffer<f32>,
        start_position: usize,
        batch: usize,
        kv_width: usize,
        context_size: usize,
    ) -> Result<()> {
        let vector_len = checked_len("fp8 batched kv", batch, kv_width)?;
        let cache_len  = checked_len("fp8 batched cache", context_size, kv_width)?;
        let end_pos    = checked_sum("fp8 batched end", start_position, batch)?;
        if key.len() < vector_len || value.len() < vector_len {
            return Err(AegisError::InvalidPlan("kv_store_fp8_batched vector shape mismatch".into()));
        }
        if key_cache.len() != cache_len || value_cache.len() != cache_len || end_pos > context_size {
            return Err(AegisError::InvalidPlan(format!(
                "kv_store_fp8_batched cache mismatch: start={} batch={} context={} width={}",
                start_position, batch, context_size, kv_width
            )));
        }
        let start_position = u32_arg("start_position", start_position)?;
        let batch          = u32_arg("batch", batch)?;
        let width          = u32_arg("kv_width", kv_width)?;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(width, 256), batch, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.kv_store_fp8_batched)
                .arg(&mut key_cache.slice)
                .arg(&mut value_cache.slice)
                .arg(&key.slice)
                .arg(&value.slice)
                .arg(&start_position)
                .arg(&batch)
                .arg(&width)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch kv_store_fp8_batched"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn store_kv_fp8_slots_batched_device(
        &self,
        key_cache: &mut DeviceBuffer<u8>,
        value_cache: &mut DeviceBuffer<u8>,
        key: &DeviceBuffer<f32>,
        value: &DeviceBuffer<f32>,
        slot_mapping: &DeviceBuffer<u32>,
        batch: usize,
        kv_width: usize,
        context_size: usize,
        dense_metadata: DensePrefillMetadataProof,
    ) -> Result<()> {
        let vector_len = checked_len("fp8 slot kv", batch, kv_width)?;
        let cache_len  = checked_len("fp8 slot cache", context_size, kv_width)?;
        if dense_metadata.batch() != batch || dense_metadata.context_len() > context_size {
            return Err(AegisError::InvalidPlan(format!(
                "kv_store_fp8_slots: proof mismatch batch={} ctx={} proof_batch={} proof_ctx={}",
                batch, context_size, dense_metadata.batch(), dense_metadata.context_len()
            )));
        }
        if key.len() < vector_len || value.len() < vector_len {
            return Err(AegisError::InvalidPlan("kv_store_fp8_slots vector shape mismatch".into()));
        }
        if slot_mapping.len() < batch || key_cache.len() != cache_len || value_cache.len() != cache_len {
            return Err(AegisError::InvalidPlan(format!(
                "kv_store_fp8_slots cache mismatch: batch={} context={} width={}",
                batch, context_size, kv_width
            )));
        }
        let batch        = u32_arg("batch", batch)?;
        let width        = u32_arg("kv_width", kv_width)?;
        let context_size = u32_arg("context_size", context_size)?;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(width, 256), batch, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.kv_store_fp8_slots_batched)
                .arg(&mut key_cache.slice)
                .arg(&mut value_cache.slice)
                .arg(&key.slice)
                .arg(&value.slice)
                .arg(&slot_mapping.slice)
                .arg(&batch)
                .arg(&width)
                .arg(&context_size)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch kv_store_fp8_slots_batched"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn store_kv_fp8_slots_batched_rope_key_device(
        &self,
        key_cache: &mut DeviceBuffer<u8>,
        value_cache: &mut DeviceBuffer<u8>,
        key: &mut DeviceBuffer<f32>,
        value: &DeviceBuffer<f32>,
        positions: &DeviceBuffer<u32>,
        slot_mapping: &DeviceBuffer<u32>,
        batch: usize,
        num_heads: usize,
        head_dim: usize,
        context_size: usize,
        dense_metadata: DensePrefillMetadataProof,
        rope: DeviceRopeConfig,
    ) -> Result<()> {
        if num_heads == 0 || head_dim == 0 || !head_dim.is_multiple_of(2) || head_dim > 256 {
            return Err(AegisError::InvalidPlan(format!(
                "fp8 roped kv: invalid heads={} head_dim={}",
                num_heads, head_dim
            )));
        }
        let kv_width   = checked_len("fp8 roped kv width", num_heads, head_dim)?;
        let vector_len = checked_len("fp8 roped kv vectors", batch, kv_width)?;
        let cache_len  = checked_len("fp8 roped cache", context_size, kv_width)?;
        if dense_metadata.batch() != batch || dense_metadata.context_len() > context_size {
            return Err(AegisError::InvalidPlan(format!(
                "fp8 roped kv: proof mismatch batch={} ctx={}", batch, context_size
            )));
        }
        if key.len() < vector_len || value.len() < vector_len {
            return Err(AegisError::InvalidPlan("fp8 roped kv: vector shape mismatch".into()));
        }
        if positions.len() < batch || slot_mapping.len() < batch
            || key_cache.len() != cache_len || value_cache.len() != cache_len {
            return Err(AegisError::InvalidPlan(format!(
                "fp8 roped kv: cache shape mismatch context={} width={}", context_size, kv_width
            )));
        }
        let batch        = u32_arg("batch", batch)?;
        let num_heads    = u32_arg("num_heads", num_heads)?;
        let head_dim     = u32_arg("head_dim", head_dim)?;
        let kv_width     = u32_arg("kv_width", kv_width)?;
        let context_size = u32_arg("context_size", context_size)?;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(kv_width, 256), batch, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.rope_kv_store_fp8_slots_batched)
                .arg(&mut key_cache.slice)
                .arg(&mut value_cache.slice)
                .arg(&mut key.slice)
                .arg(&value.slice)
                .arg(&positions.slice)
                .arg(&slot_mapping.slice)
                .arg(&batch)
                .arg(&num_heads)
                .arg(&head_dim)
                .arg(&context_size)
                .arg(&rope.theta)
                .arg(&rope.factor)
                .arg(&rope.low_freq_factor)
                .arg(&rope.high_freq_factor)
                .arg(&rope.original_max_position_embeddings)
                .arg(&rope.partial_dim)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch rope_kv_store_fp8_slots_batched"))?;
        Ok(())
    }

    /// H2D: upload `count` u16 elements from a pinned host KV slice into VRAM staging.
    /// Used before each decode/prefill attention step when KV is host-resident.
    /// No-op when `count == 0` (first decode step has no prior KV).
    pub fn upload_kv_slice_device(
        &self,
        staging: &mut DeviceBuffer<u16>,
        host: &PinnedHostSlice<u16>,
        count: usize,
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        if count > staging.len() {
            return Err(AegisError::InvalidPlan(format!(
                "upload_kv_slice overflow: count={count} staging_len={}",
                staging.len()
            )));
        }
        let host_slice = host.as_slice().map_err(map_cuda_err("mmap kv host for upload"))?;
        if count > host_slice.len() {
            return Err(AegisError::InvalidPlan(format!(
                "upload_kv_slice host overflow: count={count} host_len={}",
                host_slice.len()
            )));
        }
        let mut dst = staging.slice.slice_mut(0..count);
        self.stream
            .memcpy_htod(&host_slice[..count], &mut dst)
            .map_err(map_cuda_err("kv h2d upload"))
    }

    /// Async H2D: upload `count` u16 elements from pinned host into VRAM staging
    /// using the transfer stream. Caller must record an event on the transfer
    /// stream and have the compute stream wait on it before reading staging.
    pub fn upload_kv_slice_async(
        &self,
        staging: &mut DeviceBuffer<u16>,
        host: &PinnedHostSlice<u16>,
        count: usize,
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        if count > staging.len() {
            return Err(AegisError::InvalidPlan(format!(
                "upload_kv_slice_async overflow: count={count} staging_len={}",
                staging.len()
            )));
        }
        let host_slice = host
            .as_slice()
            .map_err(map_cuda_err("mmap kv host for async upload"))?;
        if count > host_slice.len() {
            return Err(AegisError::InvalidPlan(format!(
                "upload_kv_slice_async host overflow: count={count} host_len={}",
                host_slice.len()
            )));
        }
        let mut dst = staging.slice.slice_mut(0..count);
        self.transfer_stream
            .memcpy_htod(&host_slice[..count], &mut dst)
            .map_err(map_cuda_err("kv h2d async upload"))
    }

    /// Async D2H: writeback `kv_width` u16 elements at slot `slot_idx` from VRAM
    /// staging into pinned host. Issued on the transfer stream — caller must
    /// synchronize the transfer stream before reading the host slice on the CPU.
    /// Returns immediately; the copy proceeds asynchronously on the transfer stream.
    pub fn writeback_kv_slot_async(
        &self,
        host: &mut PinnedHostSlice<u16>,
        staging: &DeviceBuffer<u16>,
        slot_idx: usize,
        kv_width: usize,
    ) -> Result<()> {
        let start = slot_idx.checked_mul(kv_width).ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "kv writeback slot overflow: idx={slot_idx} width={kv_width}"
            ))
        })?;
        let end = start.checked_add(kv_width).ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "kv writeback end overflow: start={start} width={kv_width}"
            ))
        })?;
        if end > staging.len() {
            return Err(AegisError::InvalidPlan(format!(
                "kv writeback async staging overflow: end={end} staging_len={}",
                staging.len()
            )));
        }
        let host_len = host.len();
        if end > host_len {
            return Err(AegisError::InvalidPlan(format!(
                "kv writeback async host overflow: end={end} host_len={host_len}"
            )));
        }
        let host_slice = host
            .as_mut_slice()
            .map_err(map_cuda_err("mmap kv host for async writeback"))?;
        let dst = &mut host_slice[start..end];
        let src = staging.slice.slice(start..end);
        self.transfer_stream
            .memcpy_dtoh(&src, dst)
            .map_err(map_cuda_err("kv d2h async writeback"))
    }

    /// Async D2H batched writeback: writeback `batch * kv_width` u16 elements starting
    /// at slot `start_slot`. Issued on the transfer stream.
    pub fn writeback_kv_batch_async(
        &self,
        host: &mut PinnedHostSlice<u16>,
        staging: &DeviceBuffer<u16>,
        start_slot: usize,
        batch: usize,
        kv_width: usize,
    ) -> Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let start = start_slot.checked_mul(kv_width).ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "kv batch async writeback start overflow: slot={start_slot} width={kv_width}"
            ))
        })?;
        let count = batch.checked_mul(kv_width).ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "kv batch async writeback count overflow: batch={batch} width={kv_width}"
            ))
        })?;
        let end = start.checked_add(count).ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "kv batch async writeback end overflow: start={start} count={count}"
            ))
        })?;
        if end > staging.len() {
            return Err(AegisError::InvalidPlan(format!(
                "kv batch async writeback staging overflow: end={end} staging_len={}",
                staging.len()
            )));
        }
        let host_len = host.len();
        if end > host_len {
            return Err(AegisError::InvalidPlan(format!(
                "kv batch async writeback host overflow: end={end} host_len={host_len}"
            )));
        }
        let host_slice = host
            .as_mut_slice()
            .map_err(map_cuda_err("mmap kv host for async batch writeback"))?;
        let dst = &mut host_slice[start..end];
        let src = staging.slice.slice(start..end);
        self.transfer_stream
            .memcpy_dtoh(&src, dst)
            .map_err(map_cuda_err("kv d2h async batch writeback"))
    }

    /// D2H: writeback `kv_width` u16 elements at slot `slot_idx` from VRAM staging to host.
    /// Synchronous: blocks until the GPU copy completes (cudarc `memcpy_dtov` is sync).
    pub fn writeback_kv_slot_device(
        &self,
        host: &mut PinnedHostSlice<u16>,
        staging: &DeviceBuffer<u16>,
        slot_idx: usize,
        kv_width: usize,
    ) -> Result<()> {
        let start = slot_idx.checked_mul(kv_width).ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "kv writeback slot overflow: idx={slot_idx} width={kv_width}"
            ))
        })?;
        let end = start.checked_add(kv_width).ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "kv writeback end overflow: start={start} width={kv_width}"
            ))
        })?;
        if end > staging.len() {
            return Err(AegisError::InvalidPlan(format!(
                "kv writeback staging overflow: end={end} staging_len={}",
                staging.len()
            )));
        }
        let downloaded = self
            .stream
            .clone_dtoh(&staging.slice.slice(start..end))
            .map_err(map_cuda_err("kv d2h writeback"))?;
        let host_slice = host
            .as_mut_slice()
            .map_err(map_cuda_err("mmap kv host for writeback"))?;
        if end > host_slice.len() {
            return Err(AegisError::InvalidPlan(format!(
                "kv writeback host overflow: end={end} host_len={}",
                host_slice.len()
            )));
        }
        host_slice[start..end].copy_from_slice(&downloaded);
        Ok(())
    }

    /// D2H: writeback `batch * kv_width` u16 elements starting at slot `start_slot`
    /// from VRAM staging to host. Used after prefill KV store for a full chunk.
    pub fn writeback_kv_batch_device(
        &self,
        host: &mut PinnedHostSlice<u16>,
        staging: &DeviceBuffer<u16>,
        start_slot: usize,
        batch: usize,
        kv_width: usize,
    ) -> Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let start = start_slot.checked_mul(kv_width).ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "kv batch writeback start overflow: slot={start_slot} width={kv_width}"
            ))
        })?;
        let count = batch.checked_mul(kv_width).ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "kv batch writeback count overflow: batch={batch} width={kv_width}"
            ))
        })?;
        let end = start.checked_add(count).ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "kv batch writeback end overflow: start={start} count={count}"
            ))
        })?;
        if end > staging.len() {
            return Err(AegisError::InvalidPlan(format!(
                "kv batch writeback staging overflow: end={end} staging_len={}",
                staging.len()
            )));
        }
        let downloaded = self
            .stream
            .clone_dtoh(&staging.slice.slice(start..end))
            .map_err(map_cuda_err("kv batch d2h writeback"))?;
        let host_slice = host
            .as_mut_slice()
            .map_err(map_cuda_err("mmap kv host for batch writeback"))?;
        if end > host_slice.len() {
            return Err(AegisError::InvalidPlan(format!(
                "kv batch writeback host overflow: end={end} host_len={}",
                host_slice.len()
            )));
        }
        host_slice[start..end].copy_from_slice(&downloaded);
        Ok(())
    }
}

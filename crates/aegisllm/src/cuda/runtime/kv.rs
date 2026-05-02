use cudarc::driver::{LaunchConfig, PushKernelArg};

use super::{CudaRuntime, ceil_div, map_cuda_err};
use crate::cuda::{DensePrefillMetadataProof, DeviceBuffer, DeviceRopeConfig};
use crate::error::{AegisError, Result};

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
        if num_heads == 0 || head_dim == 0 || head_dim % 2 != 0 || head_dim > 256 {
            return Err(AegisError::InvalidPlan(format!(
                "slot-mapped roped kv store requires non-zero heads and even head_dim <= 256: heads={} head_dim={}",
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
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch slot-mapped roped batched kv store"))?;
        Ok(())
    }
}

use cudarc::driver::{LaunchConfig, PushKernelArg};

use super::*;
use crate::cuda::DeviceBuffer;
use aegisllm_base::error::{AegisError, Result};

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

}

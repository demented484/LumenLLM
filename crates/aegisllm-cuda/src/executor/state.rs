use super::rope::RopeConfig;
use crate::cuda::{CudaRuntime, DeviceBf16Matrix, DeviceBuffer, DeviceNvfp4Linear};
use aegisllm_base::generation::PrefillStageTimings;

#[derive(Debug)]
pub(super) struct CudaLlamaExecutor {
    pub(super) runtime: CudaRuntime,
    pub(super) hidden_size: usize,
    pub(super) num_attention_heads: usize,
    pub(super) num_kv_heads: usize,
    pub(super) head_dim: usize,
    pub(super) rms_norm_eps: f32,
    pub(super) rope: RopeConfig,
    pub(super) embed_tokens: DeviceBf16Matrix,
    pub(super) final_norm: DeviceBuffer<f32>,
    pub(super) lm_head: DeviceBf16Matrix,
    pub(super) layers: Vec<CudaLayer>,
    pub(super) kv_context_size: usize,
    pub(super) prefill_chunk_size: usize,
    pub(super) prefill_stage_timings_enabled: bool,
}

#[derive(Debug)]
pub(super) struct CudaLayer {
    pub(super) input_norm_weight: DeviceBuffer<f32>,
    pub(super) post_attention_norm_weight: DeviceBuffer<f32>,
    pub(super) q_proj: DeviceNvfp4Linear,
    pub(super) k_proj: DeviceNvfp4Linear,
    pub(super) v_proj: DeviceNvfp4Linear,
    pub(super) qkv_proj: Option<DeviceNvfp4Linear>,
    pub(super) o_proj: DeviceNvfp4Linear,
    pub(super) gate_proj: DeviceNvfp4Linear,
    pub(super) up_proj: DeviceNvfp4Linear,
    pub(super) down_proj: DeviceNvfp4Linear,
}

#[derive(Debug)]
pub struct CudaLlamaState {
    pub(super) position: usize,
    pub(super) hidden: DeviceBuffer<f32>,
    pub(super) logits: DeviceBuffer<f32>,
    pub(super) sampled_token: DeviceBuffer<u32>,
    pub(super) layers: Vec<CudaLayerState>,
    pub(super) scratch: CudaScratch,
    pub(super) prefill: Option<CudaPrefillScratch>,
    pub(super) prefill_timings: CudaPrefillStageTimings,
}

#[derive(Debug)]
pub(super) struct CudaLayerState {
    pub(super) kv: CudaKvCache,
}

#[derive(Debug)]
#[allow(dead_code)]
pub(super) struct CudaKvCache {
    pub(super) layout: CudaKvCacheLayout,
    pub(super) keys: DeviceBuffer<u16>,
    pub(super) values: DeviceBuffer<u16>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub(super) enum CudaKvCacheLayout {
    Dense {
        context_size: usize,
        kv_width: usize,
    },
    Paged {
        block_size: usize,
        num_blocks: usize,
        kv_width: usize,
    },
}

impl CudaKvCache {
    pub(super) fn dense(
        runtime: &CudaRuntime,
        context_size: usize,
        kv_width: usize,
    ) -> aegisllm_base::error::Result<Self> {
        let len = context_size.checked_mul(kv_width).ok_or_else(|| {
            aegisllm_base::error::AegisError::InvalidPlan(format!(
                "CUDA dense KV cache length overflow: context={} kv_width={}",
                context_size, kv_width
            ))
        })?;
        Ok(Self {
            layout: CudaKvCacheLayout::Dense {
                context_size,
                kv_width,
            },
            keys: runtime.alloc_u16(len)?,
            values: runtime.alloc_u16(len)?,
        })
    }
}

#[derive(Debug)]
pub(super) struct CudaScratch {
    pub(super) input_normed: DeviceBuffer<f32>,
    pub(super) quant_hidden: DeviceBuffer<f32>,
    pub(super) quant_intermediate: DeviceBuffer<f32>,
    pub(super) mxfp4_hidden: DeviceBuffer<u8>,
    pub(super) mxfp4_intermediate: DeviceBuffer<u8>,
    pub(super) cutlass_payload: DeviceBuffer<u8>,
    pub(super) cutlass_scales: DeviceBuffer<u8>,
    pub(super) cutlass_workspace: DeviceBuffer<u8>,
    pub(super) q: DeviceBuffer<f32>,
    pub(super) k: DeviceBuffer<f32>,
    pub(super) v: DeviceBuffer<f32>,
    pub(super) attn_context: DeviceBuffer<f32>,
    pub(super) attn_out: DeviceBuffer<f32>,
    pub(super) residual: DeviceBuffer<f32>,
    pub(super) post_normed: DeviceBuffer<f32>,
    pub(super) gate: DeviceBuffer<f32>,
    pub(super) up: DeviceBuffer<f32>,
    pub(super) swiglu: DeviceBuffer<f32>,
    pub(super) mlp_out: DeviceBuffer<f32>,
    pub(super) hidden_out: DeviceBuffer<f32>,
    pub(super) final_hidden: DeviceBuffer<f32>,
    pub(super) argmax_block_values: DeviceBuffer<f32>,
    pub(super) argmax_block_indices: DeviceBuffer<u32>,
}

#[derive(Debug)]
#[allow(dead_code)]
pub(super) struct CudaPrefillScratch {
    pub(super) chunk_size: usize,
    pub(super) max_sequences: usize,
    pub(super) block_table_capacity: usize,
    pub(super) request_ids_host: Vec<u32>,
    pub(super) seq_ids_host: Vec<u32>,
    pub(super) token_host: Vec<u32>,
    pub(super) position_host: Vec<u32>,
    pub(super) slot_mapping_host: Vec<u32>,
    pub(super) cu_q_host: Vec<u32>,
    pub(super) cu_k_host: Vec<u32>,
    pub(super) context_lens_host: Vec<u32>,
    pub(super) block_tables_host: Vec<u32>,
    pub(super) request_ids: DeviceBuffer<u32>,
    pub(super) seq_ids: DeviceBuffer<u32>,
    pub(super) tokens: DeviceBuffer<u32>,
    pub(super) positions: DeviceBuffer<u32>,
    pub(super) slot_mapping: DeviceBuffer<u32>,
    pub(super) cu_q: DeviceBuffer<u32>,
    pub(super) cu_k: DeviceBuffer<u32>,
    pub(super) context_lens: DeviceBuffer<u32>,
    pub(super) block_tables: DeviceBuffer<u32>,
    pub(super) hidden: DeviceBuffer<f32>,
    pub(super) input_normed: DeviceBuffer<f32>,
    pub(super) quant_hidden: DeviceBuffer<f32>,
    pub(super) quant_intermediate: DeviceBuffer<f32>,
    pub(super) mxfp4_hidden: DeviceBuffer<u8>,
    pub(super) mxfp4_intermediate: DeviceBuffer<u8>,
    pub(super) cutlass_payload: DeviceBuffer<u8>,
    pub(super) cutlass_scales: DeviceBuffer<u8>,
    pub(super) cutlass_workspace: DeviceBuffer<u8>,
    pub(super) qkv: DeviceBuffer<f32>,
    pub(super) q: DeviceBuffer<f32>,
    pub(super) q_half: DeviceBuffer<u16>,
    pub(super) attn_split_acc: DeviceBuffer<f32>,
    pub(super) attn_split_m: DeviceBuffer<f32>,
    pub(super) attn_split_l: DeviceBuffer<f32>,
    pub(super) k: DeviceBuffer<f32>,
    pub(super) v: DeviceBuffer<f32>,
    pub(super) attn_context: DeviceBuffer<f32>,
    pub(super) attn_out: DeviceBuffer<f32>,
    pub(super) gate: DeviceBuffer<f32>,
    pub(super) up: DeviceBuffer<f32>,
    pub(super) swiglu: DeviceBuffer<f32>,
    pub(super) mlp_out: DeviceBuffer<f32>,
}

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct CudaPrefillStageTimings {
    pub(super) enabled: bool,
    pub(super) chunks: usize,
    pub(super) prepare_us: u128,
    pub(super) embed_us: u128,
    pub(super) qkv_us: u128,
    pub(super) qkv_tflops: f64,
    pub(super) rope_us: u128,
    pub(super) kv_store_us: u128,
    pub(super) attention_us: u128,
    pub(super) o_proj_us: u128,
    pub(super) mlp_us: u128,
    pub(super) mlp_tflops: f64,
    pub(super) layer_total_us: u128,
    pub(super) sample_us: u128,
}

impl CudaPrefillStageTimings {
    pub(super) fn from_enabled(enabled: bool) -> Self {
        Self {
            enabled,
            ..Self::default()
        }
    }

    pub(super) fn reset(&mut self) {
        let enabled = self.enabled;
        *self = Self {
            enabled,
            ..Self::default()
        };
    }

    pub(super) fn snapshot(self) -> Option<PrefillStageTimings> {
        self.enabled.then_some(PrefillStageTimings {
            chunks: self.chunks,
            prepare_us: self.prepare_us,
            embed_us: self.embed_us,
            qkv_us: self.qkv_us,
            qkv_tflops: self.qkv_tflops,
            rope_us: self.rope_us,
            kv_store_us: self.kv_store_us,
            attention_us: self.attention_us,
            o_proj_us: self.o_proj_us,
            mlp_us: self.mlp_us,
            mlp_tflops: self.mlp_tflops,
            layer_total_us: self.layer_total_us,
            sample_us: self.sample_us,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::CudaPrefillStageTimings;

    #[test]
    fn prefill_stage_timings_reset_preserves_enabled_flag() {
        let mut timings = CudaPrefillStageTimings {
            enabled: true,
            chunks: 3,
            prepare_us: 11,
            embed_us: 7,
            ..CudaPrefillStageTimings::default()
        };
        timings.reset();
        assert!(timings.enabled);
        assert_eq!(timings.chunks, 0);
        assert_eq!(timings.prepare_us, 0);
        assert_eq!(timings.embed_us, 0);
    }
}

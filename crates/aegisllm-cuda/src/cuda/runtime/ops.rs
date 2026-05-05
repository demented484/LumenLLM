use cudarc::driver::{LaunchConfig, PushKernelArg};

use super::{CudaRuntime, ceil_div, map_cuda_err};
use crate::cuda::{DeviceBuffer, DeviceRopeConfig};
use aegisllm_base::error::{AegisError, Result};

fn u32_arg(name: &str, value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        AegisError::InvalidPlan(format!(
            "CUDA ops argument {name} exceeds u32 range: {value}"
        ))
    })
}

fn checked_len(label: &str, lhs: usize, rhs: usize) -> Result<usize> {
    lhs.checked_mul(rhs).ok_or_else(|| {
        AegisError::InvalidPlan(format!("CUDA ops {label} length overflow: {lhs} * {rhs}"))
    })
}

fn validate_rope_shape(label: &str, num_heads: usize, head_dim: usize) -> Result<()> {
    validate_rope_shape_with_partial(label, num_heads, head_dim, 0)
}

/// When partial_dim > 0, only the first `partial_dim` elements per head are rotated
/// so the kernel only needs `partial_dim/2` threads — the 256-element constraint
/// applies to the active dims, not the full head_dim.
fn validate_rope_shape_with_partial(
    label: &str,
    num_heads: usize,
    head_dim: usize,
    partial_dim: u32,
) -> Result<()> {
    let active_dim = if partial_dim > 0 { partial_dim as usize } else { head_dim };
    if num_heads == 0 || head_dim == 0 || !head_dim.is_multiple_of(2) || active_dim > 256 {
        return Err(AegisError::InvalidPlan(format!(
            "{label} requires non-zero heads and even active_dim <= 256: \
             heads={num_heads} head_dim={head_dim} partial_dim={partial_dim}"
        )));
    }
    Ok(())
}

impl CudaRuntime {
    pub fn f32_to_f16_device(
        &self,
        input: &DeviceBuffer<f32>,
        len: usize,
        output: &mut DeviceBuffer<u16>,
    ) -> Result<()> {
        if input.len() < len || output.len() < len {
            return Err(AegisError::InvalidPlan(format!(
                "f32->f16 conversion shape mismatch: input={} output={} len={}",
                input.len(),
                output.len(),
                len
            )));
        }
        let len = len as u32;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(len, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.f32_to_f16)
                .arg(&input.slice)
                .arg(&len)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch f32 to f16"))?;
        Ok(())
    }

    pub fn f32_to_bf16_device(
        &self,
        input: &DeviceBuffer<f32>,
        len: usize,
        output: &mut DeviceBuffer<u16>,
    ) -> Result<()> {
        if input.len() < len || output.len() < len {
            return Err(AegisError::InvalidPlan(format!(
                "f32->bf16 conversion shape mismatch: input={} output={} len={}",
                input.len(),
                output.len(),
                len
            )));
        }
        let len = len as u32;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(len, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.f32_to_bf16)
                .arg(&input.slice)
                .arg(&len)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch f32 to bf16"))?;
        Ok(())
    }

    pub fn bf16_to_f32_device(
        &self,
        input: &DeviceBuffer<u16>,
        len: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if input.len() < len || output.len() < len {
            return Err(AegisError::InvalidPlan(format!(
                "bf16->f32 conversion shape mismatch: input={} output={} len={}",
                input.len(),
                output.len(),
                len
            )));
        }
        let len = len as u32;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(len, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.bf16_to_f32)
                .arg(&input.slice)
                .arg(&len)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch bf16 to f32"))?;
        Ok(())
    }

    /// MoE router: per-token softmax + top-k selection with per-expert scaling
    /// and renormalisation. Output: device-resident `[batch, top_k]` expert
    /// indices and routing weights summing to 1 per token.
    ///
    /// Replaces the host-roundtrip pattern (download logits, softmax/top-k on
    /// CPU, upload indices/weights) with one device kernel — saves a sync
    /// stall per MoE layer.
    ///
    /// `per_expert_scale` must be a device buffer of length `num_experts`.
    /// Callers whose model has no per-expert scale should provide an
    /// all-ones identity buffer.
    pub fn router_softmax_topk_device(
        &self,
        logits: &DeviceBuffer<f32>,
        per_expert_scale: &DeviceBuffer<f32>,
        batch: usize,
        num_experts: usize,
        top_k: usize,
        out_idx: &mut DeviceBuffer<u32>,
        out_weights: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let total_logits = batch
            .checked_mul(num_experts)
            .ok_or_else(|| AegisError::InvalidPlan("router logits len overflow".into()))?;
        let total_topk = batch
            .checked_mul(top_k)
            .ok_or_else(|| AegisError::InvalidPlan("router top-k len overflow".into()))?;
        if logits.len() < total_logits
            || out_idx.len() < total_topk
            || out_weights.len() < total_topk
            || per_expert_scale.len() < num_experts
        {
            return Err(AegisError::InvalidPlan(format!(
                "router top-k shape mismatch: logits={} need {}, scale={} need {}, out_idx={} out_weights={} need {}",
                logits.len(), total_logits,
                per_expert_scale.len(), num_experts,
                out_idx.len(), out_weights.len(), total_topk,
            )));
        }
        let batch_u32 = batch as u32;
        let num_experts_u32 = num_experts as u32;
        let top_k_u32 = top_k as u32;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(batch_u32, 64), 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.router_softmax_topk)
                .arg(&logits.slice)
                .arg(&per_expert_scale.slice)
                .arg(&batch_u32)
                .arg(&num_experts_u32)
                .arg(&top_k_u32)
                .arg(&mut out_idx.slice)
                .arg(&mut out_weights.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch router softmax topk"))?;
        Ok(())
    }

    /// Zero an `expert_counts[num_experts]` buffer in preparation for the
    /// `router_bucket_sort_device` scatter.
    pub fn router_zero_expert_counts_device(
        &self,
        expert_counts: &mut DeviceBuffer<u32>,
        num_experts: usize,
    ) -> Result<()> {
        if expert_counts.len() < num_experts {
            return Err(AegisError::InvalidPlan(format!(
                "router_zero_expert_counts: buffer={} need {}",
                expert_counts.len(), num_experts,
            )));
        }
        let n_u32 = num_experts as u32;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(n_u32, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.router_zero_expert_counts)
                .arg(&mut expert_counts.slice)
                .arg(&n_u32)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch router zero expert counts"))?;
        Ok(())
    }

    /// Compute device-resident `expert_offsets[num_experts + 1]` (CSR prefix
    /// sum of `expert_counts`). Used by the grouped-NVFP4 matvec to translate
    /// `(expert, batch_in_expert)` → row index in the permuted activation
    /// buffer without a host roundtrip.
    pub fn router_expert_offsets_device(
        &self,
        expert_counts: &DeviceBuffer<u32>,
        num_experts: usize,
        expert_offsets: &mut DeviceBuffer<u32>,
    ) -> Result<()> {
        if expert_counts.len() < num_experts || expert_offsets.len() < num_experts + 1 {
            return Err(AegisError::InvalidPlan(format!(
                "router_expert_offsets shape: counts={} need {}, offsets={} need {}",
                expert_counts.len(),
                num_experts,
                expert_offsets.len(),
                num_experts + 1,
            )));
        }
        let n = num_experts as u32;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.router_expert_offsets)
                .arg(&expert_counts.slice)
                .arg(&n)
                .arg(&mut expert_offsets.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch router expert offsets"))?;
        Ok(())
    }

    /// Permute-gather: scatter `src[batch, hidden]` rows into the expert-
    /// sorted layout `permuted[total_assignments, hidden]`. Replaces the
    /// per-expert `gather_rows_f32` calls in the legacy MoE dispatch loop —
    /// one launch handles all experts.
    #[allow(clippy::too_many_arguments)]
    pub fn permute_gather_f32_device(
        &self,
        src: &DeviceBuffer<f32>,
        expert_token_lists: &DeviceBuffer<u32>,
        expert_counts: &DeviceBuffer<u32>,
        expert_first_token_off: &DeviceBuffer<u32>,
        stride: usize,                   // tokens-per-expert stride in expert_token_lists
        num_experts: usize,
        max_per_expert: usize,           // grid.y bound
        hidden: usize,
        permuted: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let stride_u32 = stride as u32;
        let hidden_u32 = hidden as u32;
        const BLOCK_DIM_X: u32 = 256;
        let cfg = LaunchConfig {
            grid_dim: (
                ceil_div(hidden_u32, BLOCK_DIM_X),
                max_per_expert as u32,
                num_experts as u32,
            ),
            block_dim: (BLOCK_DIM_X, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.permute_gather_f32)
                .arg(&src.slice)
                .arg(&expert_token_lists.slice)
                .arg(&expert_counts.slice)
                .arg(&expert_first_token_off.slice)
                .arg(&stride_u32)
                .arg(&hidden_u32)
                .arg(&mut permuted.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch permute gather f32"))?;
        Ok(())
    }

    /// Unpermute-scatter-add: read `permuted[total_assignments, hidden]`,
    /// multiply each row by its routing weight, atomically add into
    /// `moe_acc[src_token, hidden]`. Replaces the per-expert
    /// `scatter_add_weighted` calls.
    #[allow(clippy::too_many_arguments)]
    pub fn unpermute_scatter_add_f32_device(
        &self,
        permuted: &DeviceBuffer<f32>,
        expert_token_lists: &DeviceBuffer<u32>,
        expert_weight_lists: &DeviceBuffer<f32>,
        expert_counts: &DeviceBuffer<u32>,
        expert_first_token_off: &DeviceBuffer<u32>,
        stride: usize,
        num_experts: usize,
        max_per_expert: usize,
        hidden: usize,
        moe_acc: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let stride_u32 = stride as u32;
        let hidden_u32 = hidden as u32;
        const BLOCK_DIM_X: u32 = 256;
        let cfg = LaunchConfig {
            grid_dim: (
                ceil_div(hidden_u32, BLOCK_DIM_X),
                max_per_expert as u32,
                num_experts as u32,
            ),
            block_dim: (BLOCK_DIM_X, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.unpermute_scatter_add_f32)
                .arg(&permuted.slice)
                .arg(&expert_token_lists.slice)
                .arg(&expert_weight_lists.slice)
                .arg(&expert_counts.slice)
                .arg(&expert_first_token_off.slice)
                .arg(&stride_u32)
                .arg(&hidden_u32)
                .arg(&mut moe_acc.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch unpermute scatter add f32"))?;
        Ok(())
    }

    /// Grouped NVFP4 matvec: one launch processes ALL experts of a layer for
    /// one matmul-position. `base_packed`/`base_scales` are typically the
    /// VRAM expert cache buffer; `packed_offsets`/`scales_offsets` give per-
    /// expert byte offsets inside it. Output is in the permuted layout
    /// matching `permuted_input`.
    #[allow(clippy::too_many_arguments)]
    pub fn nvfp4_grouped_matvec_packed_device(
        &self,
        base_packed: &DeviceBuffer<u8>,
        packed_offsets: &DeviceBuffer<u32>,
        base_scales: &DeviceBuffer<u8>,
        scales_offsets: &DeviceBuffer<u32>,
        expert_counts: &DeviceBuffer<u32>,
        expert_first_token_off: &DeviceBuffer<u32>,
        rows: usize,
        cols: usize,
        output_scale: f32,
        permuted_input: &DeviceBuffer<f32>,
        permuted_output: &mut DeviceBuffer<f32>,
        num_experts: usize,
        max_per_expert: usize,
    ) -> Result<()> {
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const BLOCK_DIM: u32 = 256;
        let cfg = LaunchConfig {
            grid_dim: (rows_u32, max_per_expert as u32, num_experts as u32),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: BLOCK_DIM * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.nvfp4_grouped_matvec_packed)
                .arg(&base_packed.slice)
                .arg(&packed_offsets.slice)
                .arg(&base_scales.slice)
                .arg(&scales_offsets.slice)
                .arg(&expert_counts.slice)
                .arg(&expert_first_token_off.slice)
                .arg(&rows_u32)
                .arg(&cols_u32)
                .arg(&output_scale)
                .arg(&permuted_input.slice)
                .arg(&mut permuted_output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch nvfp4 grouped matvec packed"))?;
        Ok(())
    }

    /// Scatter (token, expert, weight) triples into per-expert lists. Atomic
    /// `expert_counts[expert]` provides each token's slot inside the per-
    /// expert list. After this kernel, the host downloads `expert_counts`
    /// (`num_experts * 4` bytes — small), then dispatches a per-expert
    /// matmul reading from `expert_token_lists`/`expert_weight_lists`.
    pub fn router_bucket_sort_device(
        &self,
        topk_idx: &DeviceBuffer<u32>,
        topk_weights: &DeviceBuffer<f32>,
        batch: usize,
        top_k: usize,
        stride: usize, // max tokens per expert (typically batch * top_k)
        expert_token_lists: &mut DeviceBuffer<u32>,
        expert_weight_lists: &mut DeviceBuffer<f32>,
        expert_counts: &mut DeviceBuffer<u32>,
    ) -> Result<()> {
        let total_slots = batch
            .checked_mul(top_k)
            .ok_or_else(|| AegisError::InvalidPlan("router bucket sort total slots overflow".into()))?;
        let batch_u32 = batch as u32;
        let top_k_u32 = top_k as u32;
        let stride_u32 = stride as u32;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(total_slots as u32, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.router_bucket_sort)
                .arg(&topk_idx.slice)
                .arg(&topk_weights.slice)
                .arg(&batch_u32)
                .arg(&top_k_u32)
                .arg(&stride_u32)
                .arg(&mut expert_token_lists.slice)
                .arg(&mut expert_weight_lists.slice)
                .arg(&mut expert_counts.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch router bucket sort"))?;
        Ok(())
    }

    pub fn rms_norm_device(
        &self,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        eps: f32,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if input.len() != weight.len() || input.len() != output.len() {
            return Err(AegisError::InvalidPlan(format!(
                "rms norm shape mismatch: input={} weight={} output={}",
                input.len(),
                weight.len(),
                output.len()
            )));
        }
        let len = input.len() as u32;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.rms_norm)
                .arg(&input.slice)
                .arg(&weight.slice)
                .arg(&len)
                .arg(&eps)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch rms norm"))?;
        Ok(())
    }

    pub fn rms_norm_batched_device(
        &self,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        batch: usize,
        eps: f32,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let len = weight.len();
        if input.len() < batch * len || output.len() < batch * len {
            return Err(AegisError::InvalidPlan(format!(
                "batched rms norm shape mismatch: input={} output={} batch={} len={}",
                input.len(),
                output.len(),
                batch,
                len
            )));
        }
        let batch = batch as u32;
        let len = len as u32;
        let cfg = LaunchConfig {
            grid_dim: (batch, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.rms_norm_batched)
                .arg(&input.slice)
                .arg(&weight.slice)
                .arg(&batch)
                .arg(&len)
                .arg(&eps)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch batched rms norm"))?;
        Ok(())
    }

    pub fn rms_norm_quant_nvfp4_device(
        &self,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        eps: f32,
        input_scale: f32,
        normed_output: &mut DeviceBuffer<f32>,
        quantized_output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if input.len() != weight.len()
            || input.len() != normed_output.len()
            || input.len() != quantized_output.len()
        {
            return Err(AegisError::InvalidPlan(format!(
                "rms norm nvfp4 quant shape mismatch: input={} weight={} normed={} quantized={}",
                input.len(),
                weight.len(),
                normed_output.len(),
                quantized_output.len()
            )));
        }
        let len = input.len() as u32;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.rms_norm_quant_nvfp4)
                .arg(&input.slice)
                .arg(&weight.slice)
                .arg(&len)
                .arg(&eps)
                .arg(&input_scale)
                .arg(&mut normed_output.slice)
                .arg(&mut quantized_output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch rms norm nvfp4 quant"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_quant_nvfp4_batched_device(
        &self,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        batch: usize,
        eps: f32,
        input_scale: f32,
        normed_output: &mut DeviceBuffer<f32>,
        quantized_output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let len = weight.len();
        if input.len() < batch * len
            || normed_output.len() < batch * len
            || quantized_output.len() < batch * len
        {
            return Err(AegisError::InvalidPlan(format!(
                "batched rms norm nvfp4 quant shape mismatch: input={} normed={} quantized={} batch={} len={}",
                input.len(),
                normed_output.len(),
                quantized_output.len(),
                batch,
                len
            )));
        }
        let batch = batch as u32;
        let len = len as u32;
        let cfg = LaunchConfig {
            grid_dim: (batch, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.rms_norm_quant_nvfp4_batched)
                .arg(&input.slice)
                .arg(&weight.slice)
                .arg(&batch)
                .arg(&len)
                .arg(&eps)
                .arg(&input_scale)
                .arg(&mut normed_output.slice)
                .arg(&mut quantized_output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch batched rms norm nvfp4 quant"))?;
        Ok(())
    }

    pub fn add_device(
        &self,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        self.add_device_len(a, b, output, a.len())
    }

    pub fn add_device_len(
        &self,
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        len: usize,
    ) -> Result<()> {
        if a.len() < len || b.len() < len || output.len() < len {
            return Err(AegisError::InvalidPlan(format!(
                "vector add shape mismatch: a={} b={} output={} len={}",
                a.len(),
                b.len(),
                output.len(),
                len
            )));
        }
        let len = len as u32;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(len, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.add)
                .arg(&a.slice)
                .arg(&b.slice)
                .arg(&len)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch vector add"))?;
        Ok(())
    }

    pub fn add_inplace_device(&self, a: &mut DeviceBuffer<f32>, b: &DeviceBuffer<f32>) -> Result<()> {
        self.add_inplace_device_len(a, b, a.len())
    }

    /// Element-wise copy: `dst = src`.  Implemented as `zero(dst); dst += src`.
    /// Copy the first `len` f32 elements from `src` to `dst`. Both buffers must hold at
    /// least `len` elements. Useful when source and destination have different total sizes
    /// (e.g. scratch holds the max width across all layers, but we only need to write the
    /// active layer's prefix).
    pub fn copy_prefix_f32_device(
        &self,
        src: &DeviceBuffer<f32>,
        dst: &mut DeviceBuffer<f32>,
        len: usize,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        if src.len() < len || dst.len() < len {
            return Err(AegisError::InvalidPlan(format!(
                "copy_prefix_f32 size mismatch: src={} dst={} len={}",
                src.len(),
                dst.len(),
                len
            )));
        }
        // dst[..len] = 0; dst[..len] += src[..len]  (avoids needing a dedicated copy kernel)
        self.zero_f32_device_len(dst, len)?;
        self.add_inplace_device_len(dst, src, len)?;
        Ok(())
    }

    pub fn copy_f32_device(&self, src: &DeviceBuffer<f32>, dst: &mut DeviceBuffer<f32>) -> Result<()> {
        self.zero_f32_device(dst)?;
        self.add_inplace_device_len(dst, src, src.len())
    }

    pub fn add_inplace_device_len(
        &self,
        a: &mut DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        len: usize,
    ) -> Result<()> {
        if a.len() < len || b.len() < len {
            return Err(AegisError::InvalidPlan(format!(
                "in-place vector add shape mismatch: a={} b={} len={}",
                a.len(),
                b.len(),
                len
            )));
        }
        let len = len as u32;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(len, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.add_inplace)
                .arg(&mut a.slice)
                .arg(&b.slice)
                .arg(&len)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch in-place vector add"))?;
        Ok(())
    }

    pub fn swiglu_device(
        &self,
        gate: &DeviceBuffer<f32>,
        up: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        self.swiglu_device_len(gate, up, output, gate.len())
    }

    /// RMS norm with no learned weight (used by Gemma 4 v_norm and the router pre-norm).
    /// `output[batch_idx, i] = input[batch_idx, i] * rsqrt(mean(input[batch_idx]^2) + eps)`.
    pub fn rms_norm_batched_no_weight_device(
        &self,
        input: &DeviceBuffer<f32>,
        batch: usize,
        len: usize,
        eps: f32,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let total = batch.saturating_mul(len);
        if input.len() < total || output.len() < total {
            return Err(AegisError::InvalidPlan(format!(
                "rms_norm_batched_no_weight shape mismatch: input={} output={} batch={} len={}",
                input.len(), output.len(), batch, len
            )));
        }
        let batch_u32 = u32_arg("batch", batch)?;
        let len_u32 = u32_arg("len", len)?;
        let cfg = LaunchConfig {
            grid_dim: (batch_u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.rms_norm_batched_no_weight)
                .arg(&input.slice)
                .arg(&batch_u32)
                .arg(&len_u32)
                .arg(&eps)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch rms_norm_batched_no_weight"))?;
        Ok(())
    }

    /// In-place GeGLU: `up_inout[i] = gelu_tanh(gate[i]) * up_inout[i]`.
    /// Used in chunked MoE prefill where the up buffer doubles as the output
    /// to save a copy (read-then-write of `up[idx]` is per-thread safe).
    pub fn geglu_tanh_in_place_device(
        &self,
        gate: &DeviceBuffer<f32>,
        up_inout: &mut DeviceBuffer<f32>,
        len: usize,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        if gate.len() < len || up_inout.len() < len {
            return Err(AegisError::InvalidPlan(format!(
                "geglu_in_place size mismatch: gate={} up={} need {}",
                gate.len(),
                up_inout.len(),
                len
            )));
        }
        let len_u32 = u32_arg("len", len)?;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(len_u32, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: same buffer is bound twice — once as the `up` input and once
        // as `output`. Each thread reads `up[idx]` once before writing
        // `output[idx]`, so the per-thread read-then-write is safe with input
        // and output aliased. Different threads touch disjoint indices.
        let up_ptr: *mut crate::cuda::DeviceBuffer<f32> = up_inout as *mut _;
        unsafe {
            let up_for_input = &(*up_ptr).slice;
            let up_for_output = &mut (*up_ptr).slice;
            self.stream
                .launch_builder(&self.kernels.geglu_tanh)
                .arg(&gate.slice)
                .arg(up_for_input)
                .arg(&len_u32)
                .arg(up_for_output)
                .launch(cfg)
                .map_err(map_cuda_err("launch geglu_tanh in-place"))
        }?;
        Ok(())
    }

    /// Gemma 3/4 gated activation: `output[i] = gelu_tanh(gate[i]) * up[i]`.
    pub fn geglu_tanh_device(
        &self,
        gate: &DeviceBuffer<f32>,
        up: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let len = gate.len();
        if up.len() < len || output.len() < len {
            return Err(AegisError::InvalidPlan(format!(
                "geglu shape mismatch: gate={} up={} output={}",
                gate.len(), up.len(), output.len()
            )));
        }
        let len_u32 = u32_arg("len", len)?;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(len_u32, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.geglu_tanh)
                .arg(&gate.slice)
                .arg(&up.slice)
                .arg(&len_u32)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch geglu_tanh"))?;
        Ok(())
    }

    pub fn swiglu_device_len(
        &self,
        gate: &DeviceBuffer<f32>,
        up: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        len: usize,
    ) -> Result<()> {
        if gate.len() < len || up.len() < len || output.len() < len {
            return Err(AegisError::InvalidPlan(format!(
                "swiglu shape mismatch: gate={} up={} output={} len={}",
                gate.len(),
                up.len(),
                output.len(),
                len
            )));
        }
        let len = len as u32;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(len, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.swiglu)
                .arg(&gate.slice)
                .arg(&up.slice)
                .arg(&len)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch swiglu"))?;
        Ok(())
    }

    pub fn swiglu_inplace_gate_device_len(
        &self,
        gate_and_output: &mut DeviceBuffer<f32>,
        up: &DeviceBuffer<f32>,
        len: usize,
    ) -> Result<()> {
        if gate_and_output.len() < len || up.len() < len {
            return Err(AegisError::InvalidPlan(format!(
                "in-place swiglu shape mismatch: gate={} up={} len={}",
                gate_and_output.len(),
                up.len(),
                len
            )));
        }
        let len = len as u32;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(len, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.swiglu_inplace_gate)
                .arg(&mut gate_and_output.slice)
                .arg(&up.slice)
                .arg(&len)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch in-place gate swiglu"))?;
        Ok(())
    }

    /// Like `apply_rope_device` but reads `position` from a device buffer at index 0.
    /// Use this inside CUDA Graph captures so `position` can vary per replay.
    pub fn apply_rope_ptr_device(
        &self,
        values: &mut DeviceBuffer<f32>,
        p_position: &DeviceBuffer<u32>,
        num_heads: usize,
        head_dim: usize,
        rope: DeviceRopeConfig,
    ) -> Result<()> {
        validate_rope_shape_with_partial("rope_ptr", num_heads, head_dim, rope.partial_dim)?;
        let expected_values = checked_len("rope_ptr values", num_heads, head_dim)?;
        if values.len() < expected_values {
            return Err(AegisError::InvalidPlan(format!(
                "rope_ptr shape mismatch: values={} expected_min={}",
                values.len(),
                expected_values
            )));
        }
        let num_heads = u32_arg("num_heads", num_heads)?;
        let head_dim = u32_arg("head_dim", head_dim)?;
        let cfg = LaunchConfig {
            grid_dim: (num_heads, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.rope_ptr)
                .arg(&mut values.slice)
                .arg(&p_position.slice)
                .arg(&num_heads)
                .arg(&head_dim)
                .arg(&rope.theta)
                .arg(&rope.factor)
                .arg(&rope.low_freq_factor)
                .arg(&rope.high_freq_factor)
                .arg(&rope.original_max_position_embeddings)
                .arg(&rope.partial_dim)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch rope_ptr"))?;
        Ok(())
    }

    pub fn apply_rope_device(
        &self,
        values: &mut DeviceBuffer<f32>,
        position: usize,
        num_heads: usize,
        head_dim: usize,
        rope: DeviceRopeConfig,
    ) -> Result<()> {
        validate_rope_shape("rope", num_heads, head_dim)?;
        let expected_values = checked_len("rope values", num_heads, head_dim)?;
        if values.len() < expected_values {
            return Err(AegisError::InvalidPlan(format!(
                "rope shape mismatch: values={} expected_min={}",
                values.len(),
                expected_values
            )));
        }
        let position = u32_arg("position", position)?;
        let num_heads = u32_arg("num_heads", num_heads)?;
        let head_dim = u32_arg("head_dim", head_dim)?;
        let cfg = LaunchConfig {
            grid_dim: (num_heads, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.rope)
                .arg(&mut values.slice)
                .arg(&position)
                .arg(&num_heads)
                .arg(&head_dim)
                .arg(&rope.theta)
                .arg(&rope.factor)
                .arg(&rope.low_freq_factor)
                .arg(&rope.high_freq_factor)
                .arg(&rope.original_max_position_embeddings)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch rope"))?;
        Ok(())
    }

    pub fn apply_rope_batched_device(
        &self,
        values: &mut DeviceBuffer<f32>,
        start_position: usize,
        batch: usize,
        num_heads: usize,
        head_dim: usize,
        rope: DeviceRopeConfig,
    ) -> Result<()> {
        validate_rope_shape_with_partial("batched rope", num_heads, head_dim, rope.partial_dim)?;
        let expected_values = checked_len("batched rope batch/head", batch, num_heads)
            .and_then(|len| checked_len("batched rope values", len, head_dim))?;
        if values.len() < expected_values {
            return Err(AegisError::InvalidPlan(format!(
                "batched rope shape mismatch: values={} expected={}",
                values.len(),
                expected_values
            )));
        }
        let start_position = u32_arg("start_position", start_position)?;
        let batch = u32_arg("batch", batch)?;
        let num_heads = u32_arg("num_heads", num_heads)?;
        let head_dim = u32_arg("head_dim", head_dim)?;
        let cfg = LaunchConfig {
            grid_dim: (num_heads, batch, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.rope_batched)
                .arg(&mut values.slice)
                .arg(&start_position)
                .arg(&batch)
                .arg(&num_heads)
                .arg(&head_dim)
                .arg(&rope.theta)
                .arg(&rope.factor)
                .arg(&rope.low_freq_factor)
                .arg(&rope.high_freq_factor)
                .arg(&rope.original_max_position_embeddings)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch batched rope"))?;
        Ok(())
    }

    pub fn apply_rope_positions_batched_device(
        &self,
        values: &mut DeviceBuffer<f32>,
        positions: &DeviceBuffer<u32>,
        batch: usize,
        num_heads: usize,
        head_dim: usize,
        rope: DeviceRopeConfig,
    ) -> Result<()> {
        validate_rope_shape_with_partial("positions batched rope", num_heads, head_dim, rope.partial_dim)?;
        let expected_values = batch
            .checked_mul(num_heads)
            .and_then(|len| len.checked_mul(head_dim))
            .ok_or_else(|| {
                AegisError::InvalidPlan(format!(
                    "positions batched rope length overflow: batch={} heads={} head_dim={}",
                    batch, num_heads, head_dim
                ))
            })?;
        if values.len() < expected_values || positions.len() < batch {
            return Err(AegisError::InvalidPlan(format!(
                "positions batched rope shape mismatch: values={} positions={} expected_values={} batch={}",
                values.len(),
                positions.len(),
                expected_values,
                batch
            )));
        }
        let batch = u32_arg("batch", batch)?;
        let num_heads = u32_arg("num_heads", num_heads)?;
        let head_dim = u32_arg("head_dim", head_dim)?;
        let cfg = LaunchConfig {
            grid_dim: (num_heads, batch, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.rope_positions_batched)
                .arg(&mut values.slice)
                .arg(&positions.slice)
                .arg(&batch)
                .arg(&num_heads)
                .arg(&head_dim)
                .arg(&rope.theta)
                .arg(&rope.factor)
                .arg(&rope.low_freq_factor)
                .arg(&rope.high_freq_factor)
                .arg(&rope.original_max_position_embeddings)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch positions batched rope"))?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn apply_rope_positions_batched_f16_out_device(
        &self,
        values: &mut DeviceBuffer<f32>,
        positions: &DeviceBuffer<u32>,
        batch: usize,
        num_heads: usize,
        head_dim: usize,
        rope: DeviceRopeConfig,
        output: &mut DeviceBuffer<u16>,
    ) -> Result<()> {
        validate_rope_shape_with_partial(
            "positions batched rope f16 output",
            num_heads,
            head_dim,
            rope.partial_dim,
        )?;
        let expected_values = batch
            .checked_mul(num_heads)
            .and_then(|len| len.checked_mul(head_dim))
            .ok_or_else(|| {
                AegisError::InvalidPlan(format!(
                    "positions batched rope f16 output length overflow: batch={} heads={} head_dim={}",
                    batch, num_heads, head_dim
                ))
            })?;
        if values.len() < expected_values
            || output.len() < expected_values
            || positions.len() < batch
        {
            return Err(AegisError::InvalidPlan(format!(
                "positions batched rope f16 output shape mismatch: values={} output={} positions={} expected_values={} batch={}",
                values.len(),
                output.len(),
                positions.len(),
                expected_values,
                batch
            )));
        }
        let batch = u32_arg("batch", batch)?;
        let num_heads = u32_arg("num_heads", num_heads)?;
        let head_dim = u32_arg("head_dim", head_dim)?;
        let cfg = LaunchConfig {
            grid_dim: (num_heads, batch, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.rope_positions_batched_f16_out)
                .arg(&mut values.slice)
                .arg(&positions.slice)
                .arg(&batch)
                .arg(&num_heads)
                .arg(&head_dim)
                .arg(&rope.theta)
                .arg(&rope.factor)
                .arg(&rope.low_freq_factor)
                .arg(&rope.high_freq_factor)
                .arg(&rope.original_max_position_embeddings)
                .arg(&mut output.slice)
                .arg(&rope.partial_dim)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch positions batched rope f16 output"))?;
        Ok(())
    }

    pub fn copy_row_f32_device(
        &self,
        input: &DeviceBuffer<f32>,
        row: usize,
        cols: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if input.len() < (row + 1) * cols || output.len() != cols {
            return Err(AegisError::InvalidPlan(format!(
                "copy row shape mismatch: input={} row={} cols={} output={}",
                input.len(),
                row,
                cols,
                output.len()
            )));
        }
        let row = row as u32;
        let cols = cols as u32;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(cols, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.copy_row_f32)
                .arg(&input.slice)
                .arg(&row)
                .arg(&cols)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch copy row f32"))?;
        Ok(())
    }

    /// Accumulate: `out[i] += alpha * src[i]`. Used for MoE weighted expert combine.
    pub fn axpy_f32_device(
        &self,
        alpha: f32,
        src: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if src.len() != out.len() {
            return Err(AegisError::InvalidPlan(format!(
                "axpy_f32 shape mismatch: src={} out={}",
                src.len(),
                out.len()
            )));
        }
        let len = u32_arg("len", src.len())?;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(src.len() as u32, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.axpy_f32)
                .arg(&mut out.slice)
                .arg(&src.slice)
                .arg(&alpha)
                .arg(&len)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch axpy_f32"))?;
        Ok(())
    }

    /// Zero a float buffer. Used to initialise the MoE accumulator before expert dispatch.
    pub fn zero_f32_device(&self, out: &mut DeviceBuffer<f32>) -> Result<()> {
        let total = out.len();
        self.zero_f32_device_len(out, total)
    }

    /// Zero only the first `len` elements of a float buffer.
    pub fn zero_f32_device_len(&self, out: &mut DeviceBuffer<f32>, len: usize) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        if out.len() < len {
            return Err(AegisError::InvalidPlan(format!(
                "zero_f32 len exceeds buffer: out={} requested={}",
                out.len(),
                len
            )));
        }
        let len_u32 = u32_arg("len", len)?;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(len_u32, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.zero_f32)
                .arg(&mut out.slice)
                .arg(&len_u32)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch zero_f32"))?;
        Ok(())
    }

    /// In-place element-wise scale: `out[i] *= scale`. Used for Gemma 4 per-layer `layer_scalar`.
    pub fn scale_f32_device(&self, scale: f32, out: &mut DeviceBuffer<f32>) -> Result<()> {
        let total = out.len();
        self.scale_f32_device_len(scale, out, total)
    }

    /// Like `scale_f32_device` but only operates on the first `len` elements.
    pub fn scale_f32_device_len(
        &self,
        scale: f32,
        out: &mut DeviceBuffer<f32>,
        len: usize,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        if out.len() < len {
            return Err(AegisError::InvalidPlan(format!(
                "scale_f32 len exceeds buffer: out={} requested={}",
                out.len(),
                len
            )));
        }
        let len_u32 = u32_arg("len", len)?;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(len_u32, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.scale_f32)
                .arg(&mut out.slice)
                .arg(&scale)
                .arg(&len_u32)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch scale_f32"))?;
        Ok(())
    }

    /// MoE chunked prefill helper: gather `count` rows of `cols` floats from `src`
    /// into `dst` per the `indices` mapping. `dst[r, c] = src[indices[r], c]`.
    pub fn gather_rows_f32_device(
        &self,
        src: &DeviceBuffer<f32>,
        indices: &DeviceBuffer<u32>,
        count: usize,
        cols: usize,
        dst: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if count == 0 || cols == 0 {
            return Ok(());
        }
        if dst.len() < count * cols {
            return Err(AegisError::InvalidPlan(format!(
                "gather_rows_f32 dst too small: have {}, need {}*{}={}",
                dst.len(), count, cols, count * cols
            )));
        }
        if indices.len() < count {
            return Err(AegisError::InvalidPlan(format!(
                "gather_rows_f32 indices too small: have {}, need {}",
                indices.len(), count
            )));
        }
        let count_u32 = u32_arg("count", count)?;
        let cols_u32 = u32_arg("cols", cols)?;
        let cfg = LaunchConfig {
            grid_dim: (count_u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.gather_rows_f32)
                .arg(&src.slice)
                .arg(&indices.slice)
                .arg(&count_u32)
                .arg(&cols_u32)
                .arg(&mut dst.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch gather_rows_f32"))?;
        Ok(())
    }

    /// MoE chunked prefill helper: scatter-add with per-row weight. Multiple rows
    /// may target the same destination row (different top-k positions for the
    /// same source token), so atomicAdd is used.
    /// `out[indices[r], c] += weights[r] * src[r, c]`.
    pub fn scatter_add_weighted_f32_device(
        &self,
        src: &DeviceBuffer<f32>,
        indices: &DeviceBuffer<u32>,
        weights: &DeviceBuffer<f32>,
        count: usize,
        cols: usize,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if count == 0 || cols == 0 {
            return Ok(());
        }
        if src.len() < count * cols {
            return Err(AegisError::InvalidPlan(format!(
                "scatter_add_weighted_f32 src too small: have {}, need {}*{}={}",
                src.len(), count, cols, count * cols
            )));
        }
        if indices.len() < count || weights.len() < count {
            return Err(AegisError::InvalidPlan(format!(
                "scatter_add_weighted_f32 indices/weights too small: indices={} weights={} need {}",
                indices.len(), weights.len(), count
            )));
        }
        let count_u32 = u32_arg("count", count)?;
        let cols_u32 = u32_arg("cols", cols)?;
        let cfg = LaunchConfig {
            grid_dim: (count_u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.scatter_add_weighted_f32)
                .arg(&src.slice)
                .arg(&indices.slice)
                .arg(&weights.slice)
                .arg(&count_u32)
                .arg(&cols_u32)
                .arg(&mut out.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch scatter_add_weighted_f32"))?;
        Ok(())
    }

    /// Element-wise multiply in-place: `out[i] *= scale[i]`. Lengths must match.
    pub fn mul_vec_inplace_device(
        &self,
        out: &mut DeviceBuffer<f32>,
        scale: &DeviceBuffer<f32>,
    ) -> Result<()> {
        if out.is_empty() {
            return Ok(());
        }
        if out.len() != scale.len() {
            return Err(AegisError::InvalidPlan(format!(
                "mul_vec shape mismatch: out={} scale={}",
                out.len(),
                scale.len()
            )));
        }
        let len = u32_arg("len", out.len())?;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(out.len() as u32, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.mul_vec_inplace_f32)
                .arg(&mut out.slice)
                .arg(&scale.slice)
                .arg(&len)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch mul_vec_inplace_f32"))?;
        Ok(())
    }
}

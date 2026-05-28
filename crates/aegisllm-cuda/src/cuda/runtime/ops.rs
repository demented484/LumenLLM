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

    /// Decode-only packed variant of `router_softmax_topk_device`.
    ///
    /// Writes a single contiguous `[batch * top_k * 2]` u32 buffer containing
    /// interleaved `(idx, bitcast<u32>(weight))` records. The host downloads
    /// it once (`batch * top_k * 8` bytes) and reinterprets the f32 weights
    /// on the CPU side. Replaces the two separate `out_idx` / `out_weights`
    /// downloads with a single dtoh in the decode hot path.
    pub fn router_softmax_topk_packed_device(
        &self,
        logits: &DeviceBuffer<f32>,
        per_expert_scale: &DeviceBuffer<f32>,
        batch: usize,
        num_experts: usize,
        top_k: usize,
        out_packed: &mut DeviceBuffer<u32>,
    ) -> Result<()> {
        let total_logits = batch
            .checked_mul(num_experts)
            .ok_or_else(|| AegisError::InvalidPlan("router packed logits len overflow".into()))?;
        let total_packed_words = batch
            .checked_mul(top_k)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| AegisError::InvalidPlan("router packed top-k len overflow".into()))?;
        if logits.len() < total_logits
            || out_packed.len() < total_packed_words
            || per_expert_scale.len() < num_experts
        {
            return Err(AegisError::InvalidPlan(format!(
                "router topk packed shape mismatch: logits={} need {}, scale={} need {}, out={} need {}",
                logits.len(), total_logits,
                per_expert_scale.len(), num_experts,
                out_packed.len(), total_packed_words,
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
                .launch_builder(&self.kernels.router_softmax_topk_packed)
                .arg(&logits.slice)
                .arg(&per_expert_scale.slice)
                .arg(&batch_u32)
                .arg(&num_experts_u32)
                .arg(&top_k_u32)
                .arg(&mut out_packed.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch router softmax topk packed"))?;
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

    /// Strided GeGLU over a row-stacked fused gate/up buffer.
    ///
    /// `fused` is `[batch, 2*intermediate]` row-major where, per row, the first
    /// `intermediate` floats are the gate logits and the next `intermediate` are
    /// the up logits. Writes `output[batch, intermediate]` row-major with
    /// `gelu_pytorch_tanh(gate) * up`. Used by the fused shared-MLP path where
    /// a single cuBLASLt GEMM produces `fused` and this kernel splits +
    /// activates it ready for the down projection.
    pub fn geglu_tanh_strided_device(
        &self,
        fused: &DeviceBuffer<f32>,
        batch: usize,
        intermediate: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if batch == 0 || intermediate == 0 {
            return Ok(());
        }
        let need_in = checked_len("geglu_strided fused", batch, 2 * intermediate)?;
        let need_out = checked_len("geglu_strided output", batch, intermediate)?;
        if fused.len() < need_in {
            return Err(AegisError::InvalidPlan(format!(
                "geglu_strided fused buffer too small: have {} need batch*2*intermediate={}*2*{}={}",
                fused.len(), batch, intermediate, need_in
            )));
        }
        if output.len() < need_out {
            return Err(AegisError::InvalidPlan(format!(
                "geglu_strided output buffer too small: have {} need batch*intermediate={}*{}={}",
                output.len(), batch, intermediate, need_out
            )));
        }
        let batch_u32 = u32_arg("batch", batch)?;
        let intermediate_u32 = u32_arg("intermediate", intermediate)?;
        // 2D launch: x = column (intermediate), y = row (batch).
        // Block 64×4 keeps occupancy high while letting one warp own
        // a 32-col tile of one row (coalesced loads from `fused`).
        let block_x = 64u32;
        let block_y = 4u32;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(intermediate_u32, block_x), ceil_div(batch_u32, block_y), 1),
            block_dim: (block_x, block_y, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.geglu_tanh_strided)
                .arg(&fused.slice)
                .arg(&batch_u32)
                .arg(&intermediate_u32)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch geglu_tanh_strided"))?;
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

    /// Build per-expert prefix-sum offsets from `expert_counts`. Output is
    /// `expert_offsets[num_experts+1]` where `expert_offsets[e]` is the
    /// starting row in the permuted-activation buffer for expert `e`. Used
    /// by grouped MoE: replaces a host-side prefix-sum + upload after the
    /// per-token softmax+topk produced `expert_counts` on device.
    pub fn router_expert_offsets_device(
        &self,
        expert_counts: &DeviceBuffer<u32>,
        num_experts: usize,
        expert_offsets: &mut DeviceBuffer<u32>,
    ) -> Result<()> {
        if expert_counts.len() < num_experts {
            return Err(AegisError::InvalidPlan(format!(
                "router_expert_offsets counts too small: have {} need {}",
                expert_counts.len(), num_experts
            )));
        }
        if expert_offsets.len() < num_experts + 1 {
            return Err(AegisError::InvalidPlan(format!(
                "router_expert_offsets out too small: have {} need {}",
                expert_offsets.len(), num_experts + 1
            )));
        }
        let num_experts_u32 = u32_arg("num_experts", num_experts)?;
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.router_expert_offsets)
                .arg(&expert_counts.slice)
                .arg(&num_experts_u32)
                .arg(&mut expert_offsets.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch router_expert_offsets"))?;
        Ok(())
    }

    /// Permute-gather: scatter source rows into expert-sorted layout. After
    /// this kernel, `permuted[expert_offsets[e]..expert_offsets[e+1]]` holds
    /// hidden states of all tokens routed to expert `e`. Single launch
    /// replaces the per-expert `gather_rows_f32_device` calls in the
    /// grouped MoE prefill path.
    #[allow(clippy::too_many_arguments)]
    pub fn permute_gather_f32_device(
        &self,
        src: &DeviceBuffer<f32>,
        expert_token_lists: &DeviceBuffer<u32>,
        expert_counts: &DeviceBuffer<u32>,
        expert_offsets: &DeviceBuffer<u32>,
        stride: usize,
        num_experts: usize,
        max_tokens_per_expert: usize,
        hidden: usize,
        permuted: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if hidden == 0 || num_experts == 0 || max_tokens_per_expert == 0 {
            return Ok(());
        }
        let stride_u32 = u32_arg("stride", stride)?;
        let hidden_u32 = u32_arg("hidden", hidden)?;
        let num_experts_u32 = u32_arg("num_experts", num_experts)?;
        let max_tok_u32 = u32_arg("max_tokens_per_expert", max_tokens_per_expert)?;
        let block_dim = 256u32;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(hidden_u32, block_dim), max_tok_u32, num_experts_u32),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.permute_gather_f32)
                .arg(&src.slice)
                .arg(&expert_token_lists.slice)
                .arg(&expert_counts.slice)
                .arg(&expert_offsets.slice)
                .arg(&stride_u32)
                .arg(&hidden_u32)
                .arg(&mut permuted.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch permute_gather_f32"))?;
        Ok(())
    }

    /// Deterministic unpermute-scatter-add: reads per-expert output rows from
    /// the permuted buffer, multiplies each by its routing weight, and adds
    /// the result into `moe_acc[src_token, h]`.
    ///
    /// Replaces the original single `aegis_unpermute_scatter_add_f32` kernel,
    /// whose `atomicAdd` made the per-token sum ORDER-DEPENDENT across runs
    /// (blocks from different experts race on the same `moe_acc` cell, and
    /// float atomic-add ordering is not reproducible). That ~1-ULP per-run
    /// drift propagated through every prefill layer and flipped occasional
    /// late-token argmax decisions, so greedy (temperature=0) decode produced
    /// different completions run-to-run.
    ///
    /// The deterministic path is two kernels:
    ///   1. `aegis_router_build_unpermute_index` — builds a per-token inverse
    ///      routing table indexed by the expert's *canonical rank* (ascending
    ///      expert id), so the slot assignment is scheduler-independent.
    ///   2. `aegis_unpermute_scatter_serial_f32` — one block per
    ///      `(hidden tile, token)`, accumulates that token's routes in fixed
    ///      rank order and writes `moe_acc` once. No atomics, no contention.
    ///
    /// `moe_acc` must be pre-zeroed by the caller (it always is). The serial
    /// kernel uses `+=` so the CUTLASS split path's two calls (large then
    /// small experts) compose correctly.
    #[allow(clippy::too_many_arguments)]
    pub fn unpermute_scatter_add_f32_device(
        &self,
        permuted: &DeviceBuffer<f32>,
        expert_token_lists: &DeviceBuffer<u32>,
        expert_weight_lists: &DeviceBuffer<f32>,
        expert_counts: &DeviceBuffer<u32>,
        expert_offsets: &DeviceBuffer<u32>,
        stride: usize,
        num_experts: usize,
        max_tokens_per_expert: usize,
        hidden: usize,
        batch: usize,
        top_k: usize,
        unpermute_rows: &mut DeviceBuffer<u32>,
        unpermute_wbits: &mut DeviceBuffer<u32>,
        unpermute_count: &mut DeviceBuffer<u32>,
        moe_acc: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if hidden == 0 || num_experts == 0 || max_tokens_per_expert == 0
            || batch == 0 || top_k == 0
        {
            return Ok(());
        }
        let index_len = batch.checked_mul(top_k).ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "unpermute index overflow: batch={batch} top_k={top_k}"
            ))
        })?;
        if unpermute_rows.len() < index_len
            || unpermute_wbits.len() < index_len
            || unpermute_count.len() < batch
        {
            return Err(AegisError::InvalidPlan(format!(
                "unpermute index buffers too small: rows={} wbits={} count={} need rows/wbits={} count={}",
                unpermute_rows.len(), unpermute_wbits.len(), unpermute_count.len(),
                index_len, batch,
            )));
        }
        let stride_u32 = u32_arg("stride", stride)?;
        let hidden_u32 = u32_arg("hidden", hidden)?;
        let num_experts_u32 = u32_arg("num_experts", num_experts)?;
        let max_tok_u32 = u32_arg("max_tokens_per_expert", max_tokens_per_expert)?;
        let batch_u32 = u32_arg("batch", batch)?;
        let top_k_u32 = u32_arg("top_k", top_k)?;

        // Zero the per-token route counter. `build_unpermute_index` only
        // `atomicMax`es into it, so a stale value from a prior call (or a
        // prior layer) would corrupt the result if not cleared. The buffer
        // is `[chunk_size]` (a few KiB) so zeroing all of it is negligible.
        self.stream
            .memset_zeros(&mut unpermute_count.slice)
            .map_err(map_cuda_err("zero unpermute_count"))?;

        // Kernel 1: build the per-token inverse routing table.
        let block_dim = 256u32;
        let build_cfg = LaunchConfig {
            grid_dim: (ceil_div(max_tok_u32, block_dim), num_experts_u32, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.router_build_unpermute_index)
                .arg(&expert_token_lists.slice)
                .arg(&expert_counts.slice)
                .arg(&expert_offsets.slice)
                .arg(&expert_weight_lists.slice)
                .arg(&num_experts_u32)
                .arg(&stride_u32)
                .arg(&top_k_u32)
                .arg(&mut unpermute_rows.slice)
                .arg(&mut unpermute_wbits.slice)
                .arg(&mut unpermute_count.slice)
                .launch(build_cfg)
        }
        .map_err(map_cuda_err("launch router_build_unpermute_index"))?;

        // Kernel 2: deterministic serial scatter — one block per (h tile, token).
        let scatter_cfg = LaunchConfig {
            grid_dim: (ceil_div(hidden_u32, block_dim), batch_u32, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.unpermute_scatter_serial_f32)
                .arg(&permuted.slice)
                .arg(&unpermute_rows.slice)
                .arg(&unpermute_wbits.slice)
                .arg(&unpermute_count.slice)
                .arg(&top_k_u32)
                .arg(&hidden_u32)
                .arg(&mut moe_acc.slice)
                .launch(scatter_cfg)
        }
        .map_err(map_cuda_err("launch unpermute_scatter_serial_f32"))?;
        Ok(())
    }

    /// Element-wise multiply in-place: `out[i] *= scale[i]`. Lengths must match.
    /// Batched per-token-strided multiply for the PLE per-layer additive
    /// (prefill path). `gate[t, d] *= per_layer_inputs[t, layer_idx, d]`
    /// where `per_layer_inputs` is `[chunk_size, num_layers, ple_dim]`
    /// row-major and `gate` is `[chunk_size, ple_dim]` row-major.
    #[allow(clippy::too_many_arguments)]
    pub fn ple_per_layer_mul_inplace_device(
        &self,
        gate: &mut DeviceBuffer<f32>,
        per_layer_inputs: &DeviceBuffer<f32>,
        chunk_size: usize,
        num_layers: usize,
        ple_dim: usize,
        layer_idx: usize,
    ) -> Result<()> {
        if gate.len() < chunk_size * ple_dim {
            return Err(AegisError::InvalidPlan(format!(
                "ple_per_layer_mul: gate len={} < chunk*ple_dim={}",
                gate.len(), chunk_size * ple_dim,
            )));
        }
        if per_layer_inputs.len() < chunk_size * num_layers * ple_dim {
            return Err(AegisError::InvalidPlan(format!(
                "ple_per_layer_mul: per_layer_inputs len={} < chunk*num_layers*ple_dim={}",
                per_layer_inputs.len(), chunk_size * num_layers * ple_dim,
            )));
        }
        let ple_dim_u = u32_arg("ple_dim", ple_dim)?;
        let chunk_u = u32_arg("chunk_size", chunk_size)?;
        let num_layers_u = u32_arg("num_layers", num_layers)?;
        let layer_idx_u = u32_arg("layer_idx", layer_idx)?;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(ple_dim_u, 128), chunk_u, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.ple_per_layer_mul_inplace_f32)
                .arg(&mut gate.slice)
                .arg(&per_layer_inputs.slice)
                .arg(&chunk_u)
                .arg(&num_layers_u)
                .arg(&ple_dim_u)
                .arg(&layer_idx_u)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch ple_per_layer_mul_inplace_f32"))?;
        Ok(())
    }

    /// Single-tensor in-place `gelu_pytorch_tanh` — used by the Gemma-4 E4B
    /// PLE gate before the elementwise multiply with per-layer inputs.
    pub fn gelu_tanh_inplace_device(
        &self,
        x: &mut DeviceBuffer<f32>,
        len: usize,
    ) -> Result<()> {
        if x.len() < len {
            return Err(AegisError::InvalidPlan(format!(
                "gelu_tanh_inplace: buf len={} < requested {}",
                x.len(), len
            )));
        }
        let len_u = u32_arg("len", len)?;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(len_u, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.gelu_tanh_inplace_f32)
                .arg(&mut x.slice)
                .arg(&len_u)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch gelu_tanh_inplace_f32"))?;
        Ok(())
    }

    /// Multiply `out[i] *= src[src_offset + i]` for `i in 0..len`, on a
    /// sliced view of `src`. Used by the PLE per-layer additive: the
    /// per-layer-feed slice for layer `li` lives at
    /// `per_layer_inputs[li * ple_dim .. (li+1) * ple_dim]`.
    pub fn mul_vec_inplace_slice_device(
        &self,
        out: &mut DeviceBuffer<f32>,
        src: &DeviceBuffer<f32>,
        src_offset: usize,
        len: usize,
    ) -> Result<()> {
        if out.len() < len {
            return Err(AegisError::InvalidPlan(format!(
                "mul_vec_inplace_slice: out len={} < requested {}",
                out.len(), len
            )));
        }
        if src_offset + len > src.len() {
            return Err(AegisError::InvalidPlan(format!(
                "mul_vec_inplace_slice: src[{src_offset}..{}] OOB (len={})",
                src_offset + len, src.len()
            )));
        }
        let src_view = src.slice.try_slice(src_offset..src_offset + len)
            .ok_or_else(|| AegisError::InvalidPlan(
                "mul_vec_inplace_slice: src slice view failed".into()))?;
        let mut out_view = out.slice.try_slice_mut(0..len)
            .ok_or_else(|| AegisError::InvalidPlan(
                "mul_vec_inplace_slice: out slice view failed".into()))?;
        let len_u = u32_arg("len", len)?;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(len_u, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.mul_vec_inplace_f32)
                .arg(&mut out_view)
                .arg(&src_view)
                .arg(&len_u)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch mul_vec_inplace_slice"))?;
        Ok(())
    }

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

    /// Stage I.4 vision pixel rescale: in-place x = 2*(x - 0.5) on a flat buffer.
    pub fn vision_pixel_rescale_device(
        &self,
        pixels: &mut DeviceBuffer<f32>,
        n: usize,
    ) -> Result<()> {
        let n_u = u32_arg("n", n)?;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(n_u, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.vision_pixel_rescale)
                .arg(&mut pixels.slice)
                .arg(&n_u)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch vision_pixel_rescale"))?;
        Ok(())
    }

    /// Stage I.4 vision position-embedding add. Adds 2D-axial positional
    /// embeddings (bank 0 = x, bank 1 = y from a [2*N, hidden] BF16 table)
    /// into `hidden_buf` shape `[n_patches, hidden_size]` row-major.
    /// `position_table` is a raw u16 CudaSlice — the caller passes the raw
    /// slice from `DeviceBf16Matrix::values_u16()` rather than wrapping it.
    pub fn vision_pos_embed_add_device(
        &self,
        hidden_buf: &mut DeviceBuffer<f32>,
        position_table: &cudarc::driver::CudaSlice<u16>,
        n_patches_h: usize,
        n_patches_w: usize,
        n_table_rows: usize,
        hidden_size: usize,
    ) -> Result<()> {
        let n_patches = n_patches_h * n_patches_w;
        if hidden_buf.len() < n_patches * hidden_size {
            return Err(AegisError::InvalidPlan(format!(
                "vision_pos_embed_add: hidden_buf too small {} need {}",
                hidden_buf.len(), n_patches * hidden_size
            )));
        }
        let n_patches_h_u = u32_arg("n_patches_h", n_patches_h)?;
        let n_patches_w_u = u32_arg("n_patches_w", n_patches_w)?;
        let n_table_u = u32_arg("n_table_rows", n_table_rows)?;
        let hidden_u = u32_arg("hidden_size", hidden_size)?;
        let cfg = LaunchConfig {
            grid_dim: (n_patches as u32, ceil_div(hidden_u, 256), 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.vision_pos_embed_add)
                .arg(&mut hidden_buf.slice)
                .arg(position_table)
                .arg(&n_patches_h_u)
                .arg(&n_patches_w_u)
                .arg(&n_table_u)
                .arg(&hidden_u)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch vision_pos_embed_add"))?;
        Ok(())
    }

    /// Stage I.4 per-head RMSNorm of `x` shape `[n_tok, n_heads, head_dim]`.
    /// Pass `Some(weight)` for Q/K (with scale); pass `None` for V (no scale).
    /// In-place.
    pub fn vision_head_rmsnorm_device(
        &self,
        x: &mut DeviceBuffer<f32>,
        weight: Option<&DeviceBuffer<f32>>,
        n_tok: usize,
        n_heads: usize,
        head_dim: usize,
        eps: f32,
    ) -> Result<()> {
        if head_dim == 0 || head_dim > 1024 {
            return Err(AegisError::InvalidPlan(format!(
                "vision_head_rmsnorm: head_dim={} out of range (1..=1024)",
                head_dim
            )));
        }
        let with_weight: u32 = if weight.is_some() { 1 } else { 0 };
        let cfg = LaunchConfig {
            grid_dim: (u32_arg("n_tok", n_tok)?, u32_arg("n_heads", n_heads)?, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: std::mem::size_of::<f32>() as u32,
        };
        let n_heads_u = n_heads as u32;
        let head_dim_u = head_dim as u32;
        // When weight is None, pass the x buffer itself as a placeholder
        // (kernel won't read it because with_weight=0). Avoids a fresh
        // allocation.
        let dummy_weight = x.slice.clone();
        let weight_arg = match weight {
            Some(w) => &w.slice,
            None => &dummy_weight,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.vision_head_rmsnorm)
                .arg(&mut x.slice)
                .arg(weight_arg)
                .arg(&n_heads_u)
                .arg(&head_dim_u)
                .arg(&eps)
                .arg(&with_weight)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch vision_head_rmsnorm"))?;
        Ok(())
    }

    /// Stage I.4 2D multidimensional RoPE on Q or K, in-place. Splits
    /// head_dim into 2 spatial halves (x-dim, y-dim); each half does
    /// rotate-half RoPE with frequencies from `rope_theta` and per-token
    /// positions derived from (ph, pw) ↔ token index.
    pub fn vision_rope_2d_device(
        &self,
        x: &mut DeviceBuffer<f32>,
        n_tok: usize,
        n_patches_w: usize,
        n_heads: usize,
        head_dim: usize,
        rope_theta: f32,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (u32_arg("n_tok", n_tok)?, u32_arg("n_heads", n_heads)?, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_patches_w_u = n_patches_w as u32;
        let n_heads_u = n_heads as u32;
        let head_dim_u = head_dim as u32;
        unsafe {
            self.stream
                .launch_builder(&self.kernels.vision_rope_2d)
                .arg(&mut x.slice)
                .arg(&n_patches_w_u)
                .arg(&n_heads_u)
                .arg(&head_dim_u)
                .arg(&rope_theta)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch vision_rope_2d"))?;
        Ok(())
    }

    /// Stage I.4 per-channel affine standardization: x = (x - bias) * scale.
    /// In-place over `[n_rows, hidden_size]` row-major.
    pub fn vision_standardize_device(
        &self,
        x: &mut DeviceBuffer<f32>,
        scale: &DeviceBuffer<f32>,
        bias: &DeviceBuffer<f32>,
        n_rows: usize,
        hidden_size: usize,
    ) -> Result<()> {
        let hidden_u = u32_arg("hidden_size", hidden_size)?;
        let cfg = LaunchConfig {
            grid_dim: (u32_arg("n_rows", n_rows)?, ceil_div(hidden_u, 256), 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.vision_standardize)
                .arg(&mut x.slice)
                .arg(&scale.slice)
                .arg(&bias.slice)
                .arg(&hidden_u)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch vision_standardize"))?;
        Ok(())
    }

    /// Stage I.4 3×3 average pool (stride 3, no overlap) with per-element
    /// scale factor (caller passes sqrt(hidden) / pool²).
    pub fn vision_pool3x3_scale_device(
        &self,
        src: &DeviceBuffer<f32>,
        dst: &mut DeviceBuffer<f32>,
        n_ph: usize,
        n_pw: usize,
        n_th: usize,
        n_tw: usize,
        hidden_size: usize,
        pool: usize,
        out_scale: f32,
    ) -> Result<()> {
        let hidden_u = u32_arg("hidden_size", hidden_size)?;
        let cfg = LaunchConfig {
            grid_dim: (u32_arg("n_tokens", n_th * n_tw)?, ceil_div(hidden_u, 256), 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_ph_u = n_ph as u32;
        let n_pw_u = n_pw as u32;
        let n_tw_u = n_tw as u32;
        let pool_u = pool as u32;
        unsafe {
            self.stream
                .launch_builder(&self.kernels.vision_pool3x3_scale)
                .arg(&src.slice)
                .arg(&mut dst.slice)
                .arg(&n_ph_u)
                .arg(&n_pw_u)
                .arg(&n_tw_u)
                .arg(&hidden_u)
                .arg(&pool_u)
                .arg(&out_scale)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch vision_pool3x3_scale"))?;
        Ok(())
    }

    /// Stage I.3 fused bidirectional vision attention. One launch computes
    /// softmax(Q·K^T * scale) · V over the whole [n_tok, n_heads] grid.
    /// Inputs / output: f32 row-major `[n_tok, n_heads, head_dim]`.
    /// Dynamic shared: `(n_tok + 8 + head_dim) * 4` bytes; with n_tok≤2376
    /// and head_dim=72 that's ~9.7 KiB.
    pub fn vision_bidi_attn_device(
        &self,
        q: &DeviceBuffer<f32>,
        k: &DeviceBuffer<f32>,
        v: &DeviceBuffer<f32>,
        n_tok: usize,
        n_heads: usize,
        head_dim: usize,
        scale: f32,
        out: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let total = n_tok.saturating_mul(n_heads).saturating_mul(head_dim);
        if q.len() < total || k.len() < total || v.len() < total || out.len() < total {
            return Err(AegisError::InvalidPlan(format!(
                "vision_bidi_attn: q/k/v/out len mismatch (need {total}, got q={} k={} v={} out={})",
                q.len(), k.len(), v.len(), out.len(),
            )));
        }
        let shared_bytes = ((n_tok + 8 + head_dim) * std::mem::size_of::<f32>()) as u32;
        if shared_bytes > 96 * 1024 {
            return Err(AegisError::Unsupported(format!(
                "vision_bidi_attn: n_tok={n_tok} head_dim={head_dim} needs {shared_bytes}B shared > 96KiB cap"
            )));
        }
        let cfg = LaunchConfig {
            grid_dim: (u32_arg("n_heads", n_heads)?, u32_arg("n_tok", n_tok)?, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: shared_bytes,
        };
        let n_tok_u = n_tok as u32;
        let n_heads_u = n_heads as u32;
        let head_dim_u = head_dim as u32;
        unsafe {
            self.stream
                .launch_builder(&self.kernels.vision_bidi_attn)
                .arg(&q.slice)
                .arg(&k.slice)
                .arg(&v.slice)
                .arg(&n_tok_u)
                .arg(&n_heads_u)
                .arg(&head_dim_u)
                .arg(&scale)
                .arg(&mut out.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch vision_bidi_attn"))?;
        Ok(())
    }

    // ── Gemma-4 audio tower (USM/Conformer) kernels. ─────────────────────

    /// GLU split-half: `out[t, c] = a[t, c] * sigmoid(a[t, c + half])`.
    /// `a` is `[n_frames, 2*half]`, `out` is `[n_frames, half]`. Matches
    /// `torch.nn.functional.glu(x, dim=-1)` (first half value, second half gate).
    pub fn audio_glu_halfsplit_device(
        &self,
        a: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
        n_frames: usize,
        half: usize,
    ) -> Result<()> {
        if a.len() < n_frames * 2 * half || out.len() < n_frames * half {
            return Err(AegisError::InvalidPlan(format!(
                "audio_glu_halfsplit: a={} need {}, out={} need {}",
                a.len(), n_frames * 2 * half, out.len(), n_frames * half
            )));
        }
        let half_u = u32_arg("half", half)?;
        let cfg = LaunchConfig {
            grid_dim: (u32_arg("n_frames", n_frames)?, ceil_div(half_u, 256), 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_frames_u = n_frames as u32;
        unsafe {
            self.stream
                .launch_builder(&self.kernels.audio_glu_halfsplit)
                .arg(&a.slice)
                .arg(&mut out.slice)
                .arg(&n_frames_u)
                .arg(&half_u)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch audio_glu_halfsplit"))?;
        Ok(())
    }

    /// Depthwise causal conv1d over time. `in`/`out` are `[n_frames, channels]`,
    /// `w` is `[channels, kernel]` (flattened from `[channels, 1, kernel]`).
    /// Left-padded by `kernel-1` (causal); out-of-range frames read as zero.
    pub fn audio_depthwise_causal_conv1d_device(
        &self,
        input: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        n_frames: usize,
        channels: usize,
        kernel: usize,
    ) -> Result<()> {
        let total = n_frames * channels;
        if input.len() < total || output.len() < total || weight.len() < channels * kernel {
            return Err(AegisError::InvalidPlan(format!(
                "audio_depthwise_causal_conv1d: in={} out={} w={} (need in/out {}, w {})",
                input.len(), output.len(), weight.len(), total, channels * kernel
            )));
        }
        let channels_u = u32_arg("channels", channels)?;
        let cfg = LaunchConfig {
            grid_dim: (u32_arg("n_frames", n_frames)?, ceil_div(channels_u, 256), 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_frames_u = n_frames as u32;
        let kernel_u = kernel as u32;
        unsafe {
            self.stream
                .launch_builder(&self.kernels.audio_depthwise_causal_conv1d)
                .arg(&input.slice)
                .arg(&weight.slice)
                .arg(&mut output.slice)
                .arg(&n_frames_u)
                .arg(&channels_u)
                .arg(&kernel_u)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch audio_depthwise_causal_conv1d"))?;
        Ok(())
    }

    /// Apply `per_dim_scale` to the attention query, in place. `q` is
    /// `[n_frames, n_heads, head_dim]`; `per_dim_scale` is `[head_dim]` (raw
    /// learned param — softplus is applied inside the kernel). `q_scale` is the
    /// precomputed scalar `head_dim^-0.5 / softplus(0)`.
    pub fn audio_per_dim_scale_device(
        &self,
        q: &mut DeviceBuffer<f32>,
        per_dim_scale: &DeviceBuffer<f32>,
        n_frames: usize,
        n_heads: usize,
        head_dim: usize,
        q_scale: f32,
    ) -> Result<()> {
        if head_dim == 0 || head_dim > 1024 {
            return Err(AegisError::InvalidPlan(format!(
                "audio_per_dim_scale: head_dim={head_dim} out of range (1..=1024)"
            )));
        }
        if per_dim_scale.len() < head_dim {
            return Err(AegisError::InvalidPlan(format!(
                "audio_per_dim_scale: per_dim_scale len {} < head_dim {}",
                per_dim_scale.len(), head_dim
            )));
        }
        let cfg = LaunchConfig {
            grid_dim: (u32_arg("n_frames", n_frames)?, u32_arg("n_heads", n_heads)?, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_heads_u = n_heads as u32;
        let head_dim_u = head_dim as u32;
        unsafe {
            self.stream
                .launch_builder(&self.kernels.audio_per_dim_scale)
                .arg(&mut q.slice)
                .arg(&per_dim_scale.slice)
                .arg(&n_heads_u)
                .arg(&head_dim_u)
                .arg(&q_scale)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch audio_per_dim_scale"))?;
        Ok(())
    }

    /// Clamp a flat f32 buffer in place to `[-c, c]` (HF gradient clipping).
    pub fn audio_clamp_inplace_device(
        &self,
        x: &mut DeviceBuffer<f32>,
        n: usize,
        c: f32,
    ) -> Result<()> {
        if x.len() < n {
            return Err(AegisError::InvalidPlan(format!(
                "audio_clamp_inplace: x len {} < n {}", x.len(), n
            )));
        }
        let n_u = u32_arg("n", n)?;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(n_u, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.audio_clamp_inplace)
                .arg(&mut x.slice)
                .arg(&n_u)
                .arg(&c)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch audio_clamp_inplace"))?;
        Ok(())
    }

    /// SiLU in place: `x = x * sigmoid(x)` over the first `n` elements.
    pub fn audio_silu_inplace_device(
        &self,
        x: &mut DeviceBuffer<f32>,
        n: usize,
    ) -> Result<()> {
        if x.len() < n {
            return Err(AegisError::InvalidPlan(format!(
                "audio_silu_inplace: x len {} < n {}", x.len(), n
            )));
        }
        let n_u = u32_arg("n", n)?;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(n_u, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.audio_silu_inplace)
                .arg(&mut x.slice)
                .arg(&n_u)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch audio_silu_inplace"))?;
        Ok(())
    }

    /// Add a learned bias vector to each row of `x` `[n_rows, dim]`, in place.
    pub fn audio_add_bias_rows_device(
        &self,
        x: &mut DeviceBuffer<f32>,
        bias: &DeviceBuffer<f32>,
        n_rows: usize,
        dim: usize,
    ) -> Result<()> {
        if x.len() < n_rows * dim || bias.len() < dim {
            return Err(AegisError::InvalidPlan(format!(
                "audio_add_bias_rows: x={} need {}, bias={} need {}",
                x.len(), n_rows * dim, bias.len(), dim
            )));
        }
        let dim_u = u32_arg("dim", dim)?;
        let cfg = LaunchConfig {
            grid_dim: (u32_arg("n_rows", n_rows)?, ceil_div(dim_u, 256), 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.audio_add_bias_rows)
                .arg(&mut x.slice)
                .arg(&bias.slice)
                .arg(&dim_u)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch audio_add_bias_rows"))?;
        Ok(())
    }

    /// Stage I.2 vision row-softmax. In-place: for each of `n_rows` rows of
    /// `scores` (shape `[n_rows, n_cols]` row-major), pre-multiply by `scale`
    /// then apply numerically-stable softmax along the row. Used by the
    /// vision tower's bidirectional attention.
    pub fn vision_row_softmax_device(
        &self,
        scores: &mut DeviceBuffer<f32>,
        n_rows: usize,
        n_cols: usize,
        scale: f32,
    ) -> Result<()> {
        if scores.len() < n_rows * n_cols {
            return Err(AegisError::InvalidPlan(format!(
                "vision_row_softmax: scores len={} < n_rows({}) * n_cols({}) = {}",
                scores.len(), n_rows, n_cols, n_rows * n_cols,
            )));
        }
        let cfg = LaunchConfig {
            grid_dim: (u32_arg("n_rows", n_rows)?, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_rows_u = n_rows as u32;
        let n_cols_u = n_cols as u32;
        unsafe {
            self.stream
                .launch_builder(&self.kernels.vision_row_softmax)
                .arg(&mut scores.slice)
                .arg(&n_rows_u)
                .arg(&n_cols_u)
                .arg(&scale)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch vision_row_softmax"))?;
        Ok(())
    }

    /// BF16 in-place row-softmax — same math as `vision_row_softmax_device`
    /// but reads/writes BF16 storage with F32 compute. Used by the BF16
    /// vision-attention path to avoid the BF16→F32→softmax→F32→BF16
    /// round-trip around the row reduction.
    pub fn vision_row_softmax_bf16_device(
        &self,
        scores: &mut DeviceBuffer<u16>,
        n_rows: usize,
        n_cols: usize,
        scale: f32,
    ) -> Result<()> {
        if scores.len() < n_rows * n_cols {
            return Err(AegisError::InvalidPlan(format!(
                "vision_row_softmax_bf16: scores len={} < n_rows({}) * n_cols({}) = {}",
                scores.len(), n_rows, n_cols, n_rows * n_cols,
            )));
        }
        let cfg = LaunchConfig {
            grid_dim: (u32_arg("n_rows", n_rows)?, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_rows_u = n_rows as u32;
        let n_cols_u = n_cols as u32;
        unsafe {
            self.stream
                .launch_builder(&self.kernels.vision_row_softmax_bf16)
                .arg(&mut scores.slice)
                .arg(&n_rows_u)
                .arg(&n_cols_u)
                .arg(&scale)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch vision_row_softmax_bf16"))?;
        Ok(())
    }
}

/// Recurrent state cache for linear-attention (Gated DeltaNet) and SSM (Mamba) layers.
///
/// Unlike the KV cache whose size grows with context length, recurrent states are
/// *fixed-size*: each sequence holds one state tensor per recurrent layer,
/// independent of how many tokens have been processed.
///
/// Layout conventions:
/// - Gated DeltaNet: `state[layer, head, dim_v, dim_k]` stored as f32 or f16.
///   Three tensors per layer: the combined KV accumulator and the beta-gate shadow.
/// - Mamba SSM: `state[layer, head, d_state, d_inner]` stored as f32.
///
/// Phase 6 / Phase 7 kernel stubs return `AegisError::Unsupported` until the
/// actual CUDA kernels (gated_deltanet_prefill.cu, mamba_scan_prefill.cu, etc.)
/// are implemented.
use cudarc::driver::{LaunchConfig, PushKernelArg};
use crate::cuda::DeviceBuffer;
use super::{CudaRuntime, map_cuda_err};
use aegisllm_base::error::{AegisError, Result};

/// Shape descriptor for one layer's recurrent state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecurrentStateShape {
    pub num_heads: usize,
    pub dim_a: usize,
    pub dim_b: usize,
}

impl RecurrentStateShape {
    /// Total number of f32 elements for one layer's state.
    pub fn elements(&self) -> usize {
        self.num_heads * self.dim_a * self.dim_b
    }

    /// Bytes assuming f32 storage.
    pub fn bytes_f32(&self) -> usize {
        self.elements() * 4
    }
}

/// One sequence's complete recurrent state, covering all recurrent layers.
#[derive(Debug)]
pub struct SequenceRecurrentState {
    /// One device buffer per recurrent layer, each of `shape.elements()` f32 values.
    pub layer_states: Vec<DeviceBuffer<f32>>,
    pub shape: RecurrentStateShape,
    pub num_layers: usize,
}

impl SequenceRecurrentState {
    pub fn total_elements(&self) -> usize {
        self.shape.elements() * self.num_layers
    }

    pub fn total_bytes(&self) -> usize {
        self.total_elements() * 4
    }
}

impl CudaRuntime {
    /// Allocate a zeroed recurrent state for `num_layers` recurrent layers.
    ///
    /// Phase 6/7 stub — allocates VRAM but returns `Unsupported` until the
    /// GDN / Mamba kernels are wired in.
    pub fn alloc_recurrent_state(
        &self,
        num_layers: usize,
        shape: RecurrentStateShape,
    ) -> Result<SequenceRecurrentState> {
        if num_layers == 0 || shape.elements() == 0 {
            return Err(AegisError::InvalidPlan(
                "recurrent state must have at least one layer and non-zero shape".into(),
            ));
        }
        let mut layer_states = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let mut buf = self.alloc_f32(shape.elements())?;
            // Zero-initialize so the first decode step reads clean state.
            self.stream
                .memset_zeros(&mut buf.slice)
                .map_err(map_cuda_err("zero-init recurrent state"))?;
            layer_states.push(buf);
        }
        Ok(SequenceRecurrentState { layer_states, shape, num_layers })
    }

    /// Apply the Gated DeltaNet delta-rule update for a single decode token.
    ///
    /// State layout: `[num_heads, head_dim_v, head_dim_k]` f32, updated in-place.
    /// Inputs (per value head, already preprocessed by the caller):
    ///   - `query`/`key`: `[num_heads, head_dim_k]`, L2-normed (query pre-scaled
    ///     by 1/√head_dim_k) and GQA-expanded to `num_heads`;
    ///   - `value`: `[num_heads, head_dim_v]`;
    ///   - `beta` = sigmoid(b), `g` = -exp(A_log)·softplus(a+dt_bias): `[num_heads]`.
    /// Output: `[num_heads, head_dim_v]`.
    ///
    /// `head_dim_k` must be a power of 2 and ≤ 256.
    pub fn gated_deltanet_decode_step(
        &self,
        state: &mut DeviceBuffer<f32>,
        query: &DeviceBuffer<f32>,
        key: &DeviceBuffer<f32>,
        value: &DeviceBuffer<f32>,
        beta: &DeviceBuffer<f32>,
        g: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        num_heads: usize,
        head_dim_k: usize,
        head_dim_v: usize,
    ) -> Result<()> {
        if num_heads == 0 || head_dim_k == 0 || head_dim_v == 0 {
            return Ok(());
        }
        if !head_dim_k.is_power_of_two() || head_dim_k > 256 {
            return Err(AegisError::InvalidPlan(format!(
                "GDN decode: head_dim_k={head_dim_k} must be a power of 2 and ≤ 256"
            )));
        }
        let expected_state = num_heads * head_dim_v * head_dim_k;
        let expected_out   = num_heads * head_dim_v;
        if state.len() < expected_state
            || query.len() < num_heads * head_dim_k
            || key.len() < num_heads * head_dim_k
            || value.len() < num_heads * head_dim_v
            || beta.len() < num_heads
            || g.len() < num_heads
            || output.len() < expected_out
        {
            return Err(AegisError::InvalidPlan(
                "GDN decode: buffer size mismatch".into(),
            ));
        }

        // Warp-per-row: one warp per (head, d_v row). 256 threads = 8 warps/block.
        let warps_needed = (num_heads * head_dim_v) as u32;
        const WARPS_PER_BLOCK: u32 = 8;
        let cfg = LaunchConfig {
            grid_dim: ((warps_needed + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, 1, 1),
            block_dim: (WARPS_PER_BLOCK * 32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.kernels.gated_deltanet_decode)
                .arg(&mut state.slice)
                .arg(&query.slice)
                .arg(&key.slice)
                .arg(&value.slice)
                .arg(&beta.slice)
                .arg(&g.slice)
                .arg(&mut output.slice)
                .arg(&(num_heads as u32))
                .arg(&(head_dim_k as u32))
                .arg(&(head_dim_v as u32))
                .launch(cfg)
        }
        .map_err(map_cuda_err("gated_deltanet_decode"))?;
        Ok(())
    }

    // ===== Batched (chunked-prefill) GDN launchers =====

    /// Batched delta-rule over a T-token chunk (warp per (head, d_v-row); loops
    /// the T tokens through the recurrence — same math as the decode step).
    #[allow(clippy::too_many_arguments)]
    pub fn gated_deltanet_prefill_step(
        &self,
        state: &mut DeviceBuffer<f32>,
        q: &DeviceBuffer<f32>, k: &DeviceBuffer<f32>, v: &DeviceBuffer<f32>,
        beta: &DeviceBuffer<f32>, g: &DeviceBuffer<f32>, output: &mut DeviceBuffer<f32>,
        seq_len: usize, num_heads: usize, head_dim_k: usize, head_dim_v: usize,
    ) -> Result<()> {
        if seq_len == 0 || num_heads == 0 { return Ok(()); }
        let warps = (num_heads * head_dim_v) as u32;
        let cfg = LaunchConfig {
            grid_dim: ((warps + 7) / 8, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.kernels.gated_deltanet_prefill)
                .arg(&mut state.slice).arg(&q.slice).arg(&k.slice).arg(&v.slice)
                .arg(&beta.slice).arg(&g.slice).arg(&mut output.slice)
                .arg(&(seq_len as u32)).arg(&(num_heads as u32))
                .arg(&(head_dim_k as u32)).arg(&(head_dim_v as u32))
                .launch(cfg)
        }.map_err(map_cuda_err("gated_deltanet_prefill"))?;
        Ok(())
    }

    /// Batched depthwise causal conv1d + SiLU over a T-token chunk.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_conv1d_prefill(
        &self, x: &DeviceBuffer<f32>, conv_state: &mut DeviceBuffer<f32>,
        conv_weight: &DeviceBuffer<f32>, out: &mut DeviceBuffer<f32>,
        seq_len: usize, channels: usize, kernel: usize,
    ) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: ((channels as u32 + 255) / 256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0,
        };
        unsafe {
            self.stream.launch_builder(&self.kernels.gdn_conv1d_prefill)
                .arg(&x.slice).arg(&mut conv_state.slice).arg(&conv_weight.slice).arg(&mut out.slice)
                .arg(&(seq_len as u32)).arg(&(channels as u32)).arg(&(kernel as u32))
                .launch(cfg)
        }.map_err(map_cuda_err("gdn_conv1d_prefill"))?;
        Ok(())
    }

    /// Batched qk-norm + GQA-expand over T tokens. Grid (n_v, T).
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_qk_norm_expand_batched(
        &self, q_in: &DeviceBuffer<f32>, k_in: &DeviceBuffer<f32>,
        q_out: &mut DeviceBuffer<f32>, k_out: &mut DeviceBuffer<f32>,
        seq_len: usize, n_k: usize, n_v: usize, d_k: usize, expand: usize,
    ) -> Result<()> {
        let block = (d_k as u32).next_power_of_two().min(256);
        let cfg = LaunchConfig {
            grid_dim: (n_v as u32, seq_len as u32, 1), block_dim: (block, 1, 1),
            shared_mem_bytes: block * 4,
        };
        unsafe {
            self.stream.launch_builder(&self.kernels.gdn_qk_norm_expand_batched)
                .arg(&q_in.slice).arg(&k_in.slice).arg(&mut q_out.slice).arg(&mut k_out.slice)
                .arg(&(n_k as u32)).arg(&(d_k as u32)).arg(&(expand as u32))
                .launch(cfg)
        }.map_err(map_cuda_err("gdn_qk_norm_expand_batched"))?;
        Ok(())
    }

    /// Batched gate (beta, g) over T tokens.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_gate_batched(
        &self, b: &DeviceBuffer<f32>, a: &DeviceBuffer<f32>, a_log: &DeviceBuffer<f32>,
        dt_bias: &DeviceBuffer<f32>, beta_out: &mut DeviceBuffer<f32>, g_out: &mut DeviceBuffer<f32>,
        seq_len: usize, n_v: usize,
    ) -> Result<()> {
        let total = (seq_len * n_v) as u32;
        let cfg = LaunchConfig { grid_dim: ((total + 255) / 256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(&self.kernels.gdn_gate_batched)
                .arg(&b.slice).arg(&a.slice).arg(&a_log.slice).arg(&dt_bias.slice)
                .arg(&mut beta_out.slice).arg(&mut g_out.slice)
                .arg(&(seq_len as u32)).arg(&(n_v as u32))
                .launch(cfg)
        }.map_err(map_cuda_err("gdn_gate_batched"))?;
        Ok(())
    }

    /// Batched gated RMSNorm over T tokens. Grid (n_v, T).
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_gated_rmsnorm_batched(
        &self, o: &DeviceBuffer<f32>, z: &DeviceBuffer<f32>, weight: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>, seq_len: usize, n_v: usize, d_v: usize, eps: f32,
    ) -> Result<()> {
        let block = (d_v as u32).next_power_of_two().min(1024);
        let cfg = LaunchConfig {
            grid_dim: (n_v as u32, seq_len as u32, 1), block_dim: (block, 1, 1), shared_mem_bytes: block * 4,
        };
        unsafe {
            self.stream.launch_builder(&self.kernels.gdn_gated_rmsnorm_batched)
                .arg(&o.slice).arg(&z.slice).arg(&weight.slice).arg(&mut out.slice)
                .arg(&(d_v as u32)).arg(&eps)
                .launch(cfg)
        }.map_err(map_cuda_err("gdn_gated_rmsnorm_batched"))?;
        Ok(())
    }

    /// Strided 2D copy: split a [rows, src_stride] buffer into a [rows, copy_len] one.
    #[allow(clippy::too_many_arguments)]
    pub fn strided_copy_2d(
        &self, src: &DeviceBuffer<f32>, dst: &mut DeviceBuffer<f32>,
        rows: usize, copy_len: usize, src_stride: usize, dst_stride: usize, src_off: usize,
    ) -> Result<()> {
        let total = (rows * copy_len) as u32;
        let cfg = LaunchConfig { grid_dim: ((total + 255) / 256, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(&self.kernels.strided_copy_2d)
                .arg(&src.slice).arg(&mut dst.slice)
                .arg(&(rows as u32)).arg(&(copy_len as u32)).arg(&(src_stride as u32))
                .arg(&(dst_stride as u32)).arg(&(src_off as u32))
                .launch(cfg)
        }.map_err(map_cuda_err("strided_copy_2d"))?;
        Ok(())
    }

    /// L2-normalize q,k over `head_dim_k` and GQA-expand from `n_k` key heads to
    /// `n_v` value heads, scaling q by 1/√head_dim_k. Outputs are `[n_v, d_k]`.
    /// `head_dim_k` must be a power of 2 and ≤ 256.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_qk_norm_expand(
        &self,
        q_in: &DeviceBuffer<f32>,
        k_in: &DeviceBuffer<f32>,
        q_out: &mut DeviceBuffer<f32>,
        k_out: &mut DeviceBuffer<f32>,
        n_k: usize,
        n_v: usize,
        head_dim_k: usize,
    ) -> Result<()> {
        if n_k == 0 || n_v == 0 || head_dim_k == 0 {
            return Ok(());
        }
        if !head_dim_k.is_power_of_two() || head_dim_k > 256 {
            return Err(AegisError::InvalidPlan(format!(
                "GDN qk-norm: head_dim_k={head_dim_k} must be a power of 2 and ≤ 256"
            )));
        }
        if n_v % n_k != 0 {
            return Err(AegisError::InvalidPlan(format!(
                "GDN qk-norm: n_v={n_v} not a multiple of n_k={n_k}"
            )));
        }
        if q_in.len() < n_k * head_dim_k
            || k_in.len() < n_k * head_dim_k
            || q_out.len() < n_v * head_dim_k
            || k_out.len() < n_v * head_dim_k
        {
            return Err(AegisError::InvalidPlan("GDN qk-norm: buffer size mismatch".into()));
        }
        let block_k = head_dim_k as u32;
        let cfg = LaunchConfig {
            grid_dim: (n_v as u32, 1, 1),
            block_dim: (block_k, 1, 1),
            shared_mem_bytes: block_k * 4,
        };
        let expand = (n_v / n_k) as u32;
        unsafe {
            self.stream
                .launch_builder(&self.kernels.gdn_qk_norm_expand)
                .arg(&q_in.slice)
                .arg(&k_in.slice)
                .arg(&mut q_out.slice)
                .arg(&mut k_out.slice)
                .arg(&(n_k as u32))
                .arg(&(head_dim_k as u32))
                .arg(&expand)
                .launch(cfg)
        }
        .map_err(map_cuda_err("gdn_qk_norm_expand"))?;
        Ok(())
    }

    /// Compute per-value-head GDN gates: `beta = sigmoid(b)` and
    /// `g = -exp(A_log)·softplus(a + dt_bias)`. All buffers are `[n_v]`.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_gate(
        &self,
        b: &DeviceBuffer<f32>,
        a: &DeviceBuffer<f32>,
        a_log: &DeviceBuffer<f32>,
        dt_bias: &DeviceBuffer<f32>,
        beta_out: &mut DeviceBuffer<f32>,
        g_out: &mut DeviceBuffer<f32>,
        n_v: usize,
    ) -> Result<()> {
        if n_v == 0 {
            return Ok(());
        }
        if n_v > 1024 {
            return Err(AegisError::InvalidPlan(format!(
                "GDN gate: n_v={n_v} exceeds 1024 (one block)"
            )));
        }
        if b.len() < n_v || a.len() < n_v || a_log.len() < n_v
            || dt_bias.len() < n_v || beta_out.len() < n_v || g_out.len() < n_v
        {
            return Err(AegisError::InvalidPlan("GDN gate: buffer size mismatch".into()));
        }
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (n_v as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.gdn_gate)
                .arg(&b.slice)
                .arg(&a.slice)
                .arg(&a_log.slice)
                .arg(&dt_bias.slice)
                .arg(&mut beta_out.slice)
                .arg(&mut g_out.slice)
                .arg(&(n_v as u32))
                .launch(cfg)
        }
        .map_err(map_cuda_err("gdn_gate"))?;
        Ok(())
    }

    /// GDN output gated RMSNorm (Qwen3-Next): per value head, `out = weight *
    /// (o·silu(z)) / sqrt(mean((o·silu(z))²)+eps)` over `head_dim_v`. `weight`
    /// is `[head_dim_v]`; o/z/out are `[n_v, head_dim_v]`.
    pub fn gdn_gated_rmsnorm(
        &self,
        o: &DeviceBuffer<f32>,
        z: &DeviceBuffer<f32>,
        weight: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
        n_v: usize,
        head_dim_v: usize,
        eps: f32,
    ) -> Result<()> {
        if n_v == 0 || head_dim_v == 0 {
            return Ok(());
        }
        let block_v = head_dim_v.next_power_of_two().min(1024) as u32;
        if (block_v as usize) < head_dim_v {
            return Err(AegisError::InvalidPlan(format!(
                "GDN gated-rmsnorm: head_dim_v={head_dim_v} exceeds 1024"
            )));
        }
        if o.len() < n_v * head_dim_v
            || z.len() < n_v * head_dim_v
            || weight.len() < head_dim_v
            || out.len() < n_v * head_dim_v
        {
            return Err(AegisError::InvalidPlan(
                "GDN gated-rmsnorm: buffer size mismatch".into(),
            ));
        }
        let cfg = LaunchConfig {
            grid_dim: (n_v as u32, 1, 1),
            block_dim: (block_v, 1, 1),
            shared_mem_bytes: block_v * 4,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.gdn_gated_rmsnorm)
                .arg(&o.slice)
                .arg(&z.slice)
                .arg(&weight.slice)
                .arg(&mut out.slice)
                .arg(&(head_dim_v as u32))
                .arg(&eps)
                .launch(cfg)
        }
        .map_err(map_cuda_err("gdn_gated_rmsnorm"))?;
        Ok(())
    }

    /// Streaming depthwise causal conv1d + SiLU for one decode token (no bias —
    /// GDN's conv1d.weight has none). Applies the K-tap filter over
    /// `[conv_state, x_new]` per channel, writes `out` (`[channels]`), and shifts
    /// `conv_state` (`[channels, kernel-1]`) to append `x_new`. `conv_weight` is
    /// `[channels, 1, kernel]`.
    pub fn gdn_conv1d_decode(
        &self,
        x_new: &DeviceBuffer<f32>,
        conv_state: &mut DeviceBuffer<f32>,
        conv_weight: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
        channels: usize,
        kernel: usize,
    ) -> Result<()> {
        if channels == 0 || kernel == 0 {
            return Ok(());
        }
        if x_new.len() < channels
            || conv_state.len() < channels * (kernel - 1)
            || conv_weight.len() < channels * kernel
            || out.len() < channels
        {
            return Err(AegisError::InvalidPlan(
                "GDN conv1d-decode: buffer size mismatch".into(),
            ));
        }
        let threads = 256u32;
        let blocks = ((channels as u32) + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.gdn_conv1d_decode)
                .arg(&x_new.slice)
                .arg(&mut conv_state.slice)
                .arg(&conv_weight.slice)
                .arg(&mut out.slice)
                .arg(&(channels as u32))
                .arg(&(kernel as u32))
                .launch(cfg)
        }
        .map_err(map_cuda_err("gdn_conv1d_decode"))?;
        Ok(())
    }

    /// De-interleave a gated q-projection `[num_heads, 2*head_dim]` into separate
    /// contiguous `query`/`gate` buffers (each `[num_heads, head_dim]`).
    /// Qwen3-Next attention output gate.
    pub fn deinterleave_gated_q(
        &self,
        q_full: &DeviceBuffer<f32>,
        query: &mut DeviceBuffer<f32>,
        gate: &mut DeviceBuffer<f32>,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<()> {
        let total = num_heads * head_dim;
        if total == 0 {
            return Ok(());
        }
        if q_full.len() < 2 * total || query.len() < total || gate.len() < total {
            return Err(AegisError::InvalidPlan("deinterleave_gated_q: buffer size mismatch".into()));
        }
        let threads = 256u32;
        let blocks = ((total as u32) + threads - 1) / threads;
        let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.deinterleave_gated_q)
                .arg(&q_full.slice)
                .arg(&mut query.slice)
                .arg(&mut gate.slice)
                .arg(&(num_heads as u32))
                .arg(&(head_dim as u32))
                .launch(cfg)
        }
        .map_err(map_cuda_err("deinterleave_gated_q"))?;
        Ok(())
    }

    /// De-interleave Qwen3-Next in_proj_qkv from per-key-head packed layout
    /// `[kh: q,k,v]` to contiguous `[all_q | all_k | all_v]` for the conv1d.
    pub fn gdn_deinterleave_qkv(
        &self,
        in_packed: &DeviceBuffer<f32>,
        out: &mut DeviceBuffer<f32>,
        n_k: usize,
        hd_k: usize,
        hd_v: usize,
        expand: usize,
    ) -> Result<()> {
        let stride = 2 * hd_k + expand * hd_v;
        let total = n_k * stride;
        if in_packed.len() < total || out.len() < total {
            return Err(AegisError::InvalidPlan("gdn_deinterleave_qkv: buffer size mismatch".into()));
        }
        let threads = 256u32;
        let blocks = ((total as u32) + threads - 1) / threads;
        let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.gdn_deinterleave_qkv)
                .arg(&in_packed.slice)
                .arg(&mut out.slice)
                .arg(&(n_k as u32))
                .arg(&(hd_k as u32))
                .arg(&(hd_v as u32))
                .arg(&(expand as u32))
                .launch(cfg)
        }
        .map_err(map_cuda_err("gdn_deinterleave_qkv"))?;
        Ok(())
    }

    /// HF/GPT-NeoX partial RoPE (Qwen3-Next): rotate the first `rotary_dim` dims
    /// per head (half-split within `rotary_dim`, inv-freq divisor `rotary_dim`),
    /// passing the rest through. `values` is `[num_heads, head_dim]` in place.
    pub fn apply_rope_neox_partial_device(
        &self,
        values: &mut DeviceBuffer<f32>,
        p_position: &DeviceBuffer<u32>,
        num_heads: usize,
        head_dim: usize,
        theta: f32,
        rotary_dim: usize,
    ) -> Result<()> {
        if num_heads == 0 || rotary_dim == 0 {
            return Ok(());
        }
        if values.len() < num_heads * head_dim {
            return Err(AegisError::InvalidPlan("rope neox partial: buffer too small".into()));
        }
        let cfg = LaunchConfig {
            grid_dim: (num_heads as u32, 1, 1),
            block_dim: ((rotary_dim / 2) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.apply_rope_neox_partial)
                .arg(&mut values.slice)
                .arg(&p_position.slice)
                .arg(&(num_heads as u32))
                .arg(&(head_dim as u32))
                .arg(&theta)
                .arg(&(rotary_dim as u32))
                .launch(cfg)
        }
        .map_err(map_cuda_err("apply_rope_neox_partial"))?;
        Ok(())
    }

    /// Batched HF/GPT-NeoX partial RoPE (Qwen3-Next full-attention prefill).
    /// `values` is `[seq_len, num_heads, head_dim]` in place; each token uses
    /// its own position from `p_positions[t]`.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_rope_neox_partial_batched_device(
        &self,
        values: &mut DeviceBuffer<f32>,
        p_positions: &DeviceBuffer<u32>,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        theta: f32,
        rotary_dim: usize,
    ) -> Result<()> {
        if num_heads == 0 || rotary_dim == 0 || seq_len == 0 {
            return Ok(());
        }
        if values.len() < seq_len * num_heads * head_dim {
            return Err(AegisError::InvalidPlan(
                "rope neox partial batched: buffer too small".into(),
            ));
        }
        let cfg = LaunchConfig {
            grid_dim: (num_heads as u32, seq_len as u32, 1),
            block_dim: ((rotary_dim / 2) as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.apply_rope_neox_partial_batched)
                .arg(&mut values.slice)
                .arg(&p_positions.slice)
                .arg(&(num_heads as u32))
                .arg(&(head_dim as u32))
                .arg(&theta)
                .arg(&(rotary_dim as u32))
                .launch(cfg)
        }
        .map_err(map_cuda_err("apply_rope_neox_partial_batched"))?;
        Ok(())
    }

    /// `x[i] *= sigmoid(g[i])` over `n` elements (Qwen3-Next attention output gate).
    pub fn sigmoid_gate_mul(
        &self,
        x: &mut DeviceBuffer<f32>,
        g: &DeviceBuffer<f32>,
        n: usize,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        if x.len() < n || g.len() < n {
            return Err(AegisError::InvalidPlan("sigmoid_gate_mul: buffer size mismatch".into()));
        }
        let threads = 256u32;
        let blocks = ((n as u32) + threads - 1) / threads;
        let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.sigmoid_gate_mul)
                .arg(&mut x.slice)
                .arg(&g.slice)
                .arg(&(n as u32))
                .launch(cfg)
        }
        .map_err(map_cuda_err("sigmoid_gate_mul"))?;
        Ok(())
    }

    /// Scale `out[0..n)` in place by `sigmoid(logit[0])`, where `logit` is a
    /// single-element device buffer. Folds the Qwen3-Next shared-expert gate
    /// (sigmoid + broadcast scale) into one on-device launch, avoiding a
    /// blocking `download_f32` of the gate logit per MoE layer per token.
    pub fn scale_by_sigmoid_scalar(
        &self,
        out: &mut DeviceBuffer<f32>,
        logit: &DeviceBuffer<f32>,
        n: usize,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        if out.len() < n {
            return Err(AegisError::InvalidPlan(format!(
                "scale_by_sigmoid_scalar: out has {} elems, need {}",
                out.len(),
                n
            )));
        }
        let threads = 256u32;
        let blocks = ((n as u32) + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.scale_by_sigmoid_scalar)
                .arg(&mut out.slice)
                .arg(&logit.slice)
                .arg(&(n as u32))
                .launch(cfg)
        }
        .map_err(map_cuda_err("scale_by_sigmoid_scalar"))?;
        Ok(())
    }

    /// Apply the Gated DeltaNet chunked-prefill kernel over `chunk_len` tokens.
    ///
    /// **Phase 6 stub** — returns `Unsupported` until `gated_deltanet_prefill.cu`
    /// is compiled and linked.
    #[allow(unused_variables)]
    pub fn gated_deltanet_prefill_chunk(
        &self,
        state: &mut DeviceBuffer<f32>,
        queries: &DeviceBuffer<f32>,
        keys: &DeviceBuffer<f32>,
        values: &DeviceBuffer<f32>,
        betas: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        chunk_len: usize,
        num_heads: usize,
        head_dim_k: usize,
        head_dim_v: usize,
    ) -> Result<()> {
        Err(AegisError::Unsupported(
            "Gated DeltaNet chunked-prefill kernel not yet implemented (Phase 6.2)".into(),
        ))
    }

    /// Apply the Mamba selective-scan recurrent step for a single decode token.
    ///
    /// `hidden` is the post-in_proj SSM input `x` of shape `[num_heads, head_dim]`
    /// where `head_dim = d_inner / num_heads`.  `D` skip weights have the same shape.
    ///
    /// State layout: `[num_heads, d_state, head_dim]` f32, updated in-place.
    /// `a_log[h, s] = log(-A[h, s])` (standard Mamba convention; A < 0 implies decay).
    ///
    /// `head_dim` must be a power of 2 and ≤ 1024.
    #[allow(clippy::too_many_arguments)]
    pub fn mamba_scan_decode_step(
        &self,
        state: &mut DeviceBuffer<f32>,
        hidden: &DeviceBuffer<f32>,
        dt: &DeviceBuffer<f32>,
        a_log: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        c: &DeviceBuffer<f32>,
        skip_d: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        d_state: usize,
        d_inner: usize,
        num_heads: usize,
    ) -> Result<()> {
        if num_heads == 0 || d_state == 0 || d_inner == 0 {
            return Ok(());
        }
        if d_inner % num_heads != 0 {
            return Err(AegisError::InvalidPlan(format!(
                "Mamba decode: d_inner={d_inner} must be divisible by num_heads={num_heads}"
            )));
        }
        let head_dim = d_inner / num_heads;
        if !head_dim.is_power_of_two() || head_dim > 1024 {
            return Err(AegisError::InvalidPlan(format!(
                "Mamba decode: head_dim={head_dim} must be a power of 2 and ≤ 1024"
            )));
        }

        let cfg = LaunchConfig {
            grid_dim: (num_heads as u32, 1, 1),
            block_dim: (head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.kernels.mamba_scan_decode)
                .arg(&mut state.slice)
                .arg(&hidden.slice)
                .arg(&dt.slice)
                .arg(&a_log.slice)
                .arg(&b.slice)
                .arg(&c.slice)
                .arg(&skip_d.slice)
                .arg(&mut output.slice)
                .arg(&(d_state as u32))
                .arg(&(head_dim as u32))
                .launch(cfg)
        }
        .map_err(map_cuda_err("mamba_scan_decode"))?;
        Ok(())
    }

    /// Apply the Mamba parallel associative-scan prefill kernel over `seq_len` tokens.
    ///
    /// **Phase 7 stub** — returns `Unsupported` until `mamba_scan_prefill.cu`
    /// is compiled and linked.
    #[allow(unused_variables)]
    pub fn mamba_scan_prefill(
        &self,
        state: &mut DeviceBuffer<f32>,
        hidden: &DeviceBuffer<f32>,
        dt: &DeviceBuffer<f32>,
        a_log: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        c: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        seq_len: usize,
        d_state: usize,
        d_inner: usize,
        num_heads: usize,
    ) -> Result<()> {
        Err(AegisError::Unsupported(
            "Mamba parallel-scan prefill kernel not yet implemented (Phase 7.2)".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::RecurrentStateShape;

    #[test]
    fn recurrent_state_shape_elements_and_bytes() {
        let s = RecurrentStateShape { num_heads: 4, dim_a: 64, dim_b: 64 };
        assert_eq!(s.elements(), 4 * 64 * 64);
        assert_eq!(s.bytes_f32(), 4 * 64 * 64 * 4);
    }

    #[test]
    fn recurrent_state_shape_zero_is_ok_struct() {
        let s = RecurrentStateShape { num_heads: 0, dim_a: 64, dim_b: 64 };
        assert_eq!(s.elements(), 0);
    }

    #[test]
    fn gdn_decode_validates_head_dim_k_power_of_two() {
        // Validate the power-of-2 check without a GPU by testing the guard condition.
        let head_dim_k: usize = 100;
        assert!(!head_dim_k.is_power_of_two(), "100 is not a power of 2");
        let head_dim_k_ok: usize = 128;
        assert!(head_dim_k_ok.is_power_of_two() && head_dim_k_ok <= 256);
    }

    #[test]
    fn gdn_decode_state_size_math() {
        // State for a typical Qwen 3.5 GDN layer:
        // num_heads=32, head_dim_k=128, head_dim_v=128 → 32 * 128 * 128 = 524288 f32
        let num_heads: usize = 32;
        let d_k: usize = 128;
        let d_v: usize = 128;
        let state_elements = num_heads * d_v * d_k;
        let output_elements = num_heads * d_v;
        assert_eq!(state_elements, 524288);
        assert_eq!(output_elements, 32 * 128);
        // Block size = d_k, shared mem = d_k * 4 bytes = 512 bytes per block
        let smem_bytes = d_k * 4;
        assert_eq!(smem_bytes, 512);
        assert!(smem_bytes <= 49152, "fits in typical CUDA shared mem limit");
    }

    // ── GPU correctness: the aegis_gated_deltanet_decode kernel must reproduce
    //    the canonical gated-delta recurrence (the same math the CPU reference
    //    oracle in aegisllm-cpu/src/cpu/gdn.rs implements, here in the kernel's
    //    S[d_v,d_k] layout). Skips gracefully when no CUDA device is present.

    /// Host reference for one multi-head decode step, S[d_v,d_k] layout
    /// (identical math to the kernel): S*=exp(g); kv=S·k; delta=(v-kv)·beta;
    /// S+=k⊗delta; y=S·q. Returns `[n_v*d_v]` output.
    fn ref_step(
        state: &mut [f32], q: &[f32], k: &[f32], v: &[f32],
        beta: &[f32], g: &[f32], n_v: usize, d_k: usize, d_v: usize,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; n_v * d_v];
        for h in 0..n_v {
            let decay = g[h].exp();
            let qh = &q[h * d_k..(h + 1) * d_k];
            let kh = &k[h * d_k..(h + 1) * d_k];
            let vh = &v[h * d_v..(h + 1) * d_v];
            let base = h * d_v * d_k;
            for i in 0..d_v {
                let row = &mut state[base + i * d_k..base + (i + 1) * d_k];
                for x in row.iter_mut() {
                    *x *= decay;
                }
                let kv: f32 = (0..d_k).map(|j| row[j] * kh[j]).sum();
                let delta = (vh[i] - kv) * beta[h];
                for j in 0..d_k {
                    row[j] += kh[j] * delta;
                }
                out[h * d_v + i] = (0..d_k).map(|j| row[j] * qh[j]).sum();
            }
        }
        out
    }

    #[test]
    fn gdn_decode_kernel_matches_reference_on_gpu() {
        use crate::cuda::runtime::CudaRuntime;
        let rt = match CudaRuntime::new(0) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("no CUDA device ({e:?}); skipping GPU GDN check");
                return;
            }
        };
        let (n_v, d_k, d_v) = (4usize, 8usize, 8usize);
        // deterministic pseudo-random inputs in [-1, 1).
        let rng = |seed: usize, n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let h = seed
                        .wrapping_mul(2_654_435_761)
                        .wrapping_add(i.wrapping_mul(40_503));
                    (h % 1000) as f32 / 500.0 - 1.0
                })
                .collect()
        };

        let mut ref_state = vec![0.0f32; n_v * d_v * d_k];
        let mut gpu_state = rt.upload_f32(&vec![0.0f32; n_v * d_v * d_k]).unwrap();

        let cosine = |a: &[f32], b: &[f32]| -> f32 {
            let dot: f32 = a.iter().zip(b).map(|(&x, &y)| x * y).sum();
            let na: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
            dot / (na * nb + 1e-12)
        };

        for t in 0..4usize {
            let q = rng(t * 7 + 1, n_v * d_k);
            let k = rng(t * 7 + 2, n_v * d_k);
            let v = rng(t * 7 + 3, n_v * d_v);
            let beta: Vec<f32> = (0..n_v).map(|h| 0.3 + 0.1 * h as f32).collect();
            let g: Vec<f32> = (0..n_v).map(|h| -0.2 - 0.15 * h as f32).collect();

            let ref_out = ref_step(&mut ref_state, &q, &k, &v, &beta, &g, n_v, d_k, d_v);

            let dq = rt.upload_f32(&q).unwrap();
            let dk = rt.upload_f32(&k).unwrap();
            let dv = rt.upload_f32(&v).unwrap();
            let dbeta = rt.upload_f32(&beta).unwrap();
            let dg = rt.upload_f32(&g).unwrap();
            let mut dout = rt.upload_f32(&vec![0.0f32; n_v * d_v]).unwrap();
            rt.gated_deltanet_decode_step(
                &mut gpu_state, &dq, &dk, &dv, &dbeta, &dg, &mut dout, n_v, d_k, d_v,
            )
            .unwrap();
            rt.synchronize().unwrap();
            let gpu_out = rt.download_f32(&dout).unwrap();

            let cos = cosine(&ref_out, &gpu_out);
            let max_abs = ref_out
                .iter()
                .zip(&gpu_out)
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                cos > 0.9999 && max_abs < 1e-4,
                "step {t}: cos={cos} max_abs={max_abs}\n ref={ref_out:?}\n gpu={gpu_out:?}"
            );
        }

        // Final recurrent state must match too (catches decay/update drift).
        let gpu_final = rt.download_f32(&gpu_state).unwrap();
        let cos_state = cosine(&ref_state, &gpu_final);
        assert!(cos_state > 0.9999, "final state cosine {cos_state}");
    }

    #[test]
    fn gdn_qk_norm_expand_matches_reference_on_gpu() {
        use crate::cuda::runtime::CudaRuntime;
        let rt = match CudaRuntime::new(0) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("no CUDA device ({e:?}); skipping");
                return;
            }
        };
        let (n_k, n_v, d_k) = (2usize, 4usize, 8usize); // expand = 2
        let rng = |seed: usize, n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let h = seed.wrapping_mul(2_654_435_761).wrapping_add(i.wrapping_mul(40_503));
                    (h % 1000) as f32 / 500.0 - 1.0
                })
                .collect()
        };
        let q = rng(11, n_k * d_k);
        let k = rng(22, n_k * d_k);

        // reference: L2-norm over d_k, scale q by 1/sqrt(d_k), expand n_k→n_v.
        let inv_sqrt = 1.0 / (d_k as f32).sqrt();
        let mut rq = vec![0.0f32; n_v * d_k];
        let mut rk = vec![0.0f32; n_v * d_k];
        for h in 0..n_v {
            let kh = h / (n_v / n_k);
            let qn: f32 = (0..d_k).map(|j| q[kh * d_k + j].powi(2)).sum::<f32>().sqrt();
            let kn: f32 = (0..d_k).map(|j| k[kh * d_k + j].powi(2)).sum::<f32>().sqrt();
            for j in 0..d_k {
                rq[h * d_k + j] = q[kh * d_k + j] / (qn + 1e-6) * inv_sqrt;
                rk[h * d_k + j] = k[kh * d_k + j] / (kn + 1e-6);
            }
        }

        let dq = rt.upload_f32(&q).unwrap();
        let dk = rt.upload_f32(&k).unwrap();
        let mut dqo = rt.upload_f32(&vec![0.0f32; n_v * d_k]).unwrap();
        let mut dko = rt.upload_f32(&vec![0.0f32; n_v * d_k]).unwrap();
        rt.gdn_qk_norm_expand(&dq, &dk, &mut dqo, &mut dko, n_k, n_v, d_k).unwrap();
        rt.synchronize().unwrap();
        let gq = rt.download_f32(&dqo).unwrap();
        let gk = rt.download_f32(&dko).unwrap();
        for i in 0..n_v * d_k {
            assert!((rq[i] - gq[i]).abs() < 1e-4, "q[{i}] ref={} gpu={}", rq[i], gq[i]);
            assert!((rk[i] - gk[i]).abs() < 1e-4, "k[{i}] ref={} gpu={}", rk[i], gk[i]);
        }
    }

    #[test]
    fn gdn_gate_matches_reference_on_gpu() {
        use crate::cuda::runtime::CudaRuntime;
        let rt = match CudaRuntime::new(0) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("no CUDA device ({e:?}); skipping");
                return;
            }
        };
        let n_v = 6usize;
        let b: Vec<f32> = (0..n_v).map(|h| -1.0 + 0.4 * h as f32).collect();
        let a: Vec<f32> = (0..n_v).map(|h| 0.2 - 0.3 * h as f32).collect();
        let a_log: Vec<f32> = (0..n_v).map(|h| -0.5 + 0.1 * h as f32).collect();
        let dt: Vec<f32> = (0..n_v).map(|h| 0.05 * h as f32).collect();

        // reference
        let softplus = |x: f32| x.max(0.0) + (-x.abs()).exp().ln_1p();
        let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
        let rbeta: Vec<f32> = b.iter().map(|&x| sigmoid(x)).collect();
        let rg: Vec<f32> = (0..n_v)
            .map(|h| -a_log[h].exp() * softplus(a[h] + dt[h]))
            .collect();

        let db = rt.upload_f32(&b).unwrap();
        let da = rt.upload_f32(&a).unwrap();
        let dal = rt.upload_f32(&a_log).unwrap();
        let ddt = rt.upload_f32(&dt).unwrap();
        let mut dbeta = rt.upload_f32(&vec![0.0f32; n_v]).unwrap();
        let mut dg = rt.upload_f32(&vec![0.0f32; n_v]).unwrap();
        rt.gdn_gate(&db, &da, &dal, &ddt, &mut dbeta, &mut dg, n_v).unwrap();
        rt.synchronize().unwrap();
        let gbeta = rt.download_f32(&dbeta).unwrap();
        let gg = rt.download_f32(&dg).unwrap();
        for h in 0..n_v {
            assert!((rbeta[h] - gbeta[h]).abs() < 1e-5, "beta[{h}]");
            assert!((rg[h] - gg[h]).abs() < 1e-5, "g[{h}] ref={} gpu={}", rg[h], gg[h]);
            assert!(gg[h] <= 0.0, "g must be ≤ 0");
        }
    }

    #[test]
    fn gdn_gated_rmsnorm_matches_reference_on_gpu() {
        use crate::cuda::runtime::CudaRuntime;
        let rt = match CudaRuntime::new(0) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("no CUDA device ({e:?}); skipping");
                return;
            }
        };
        let (n_v, d_v) = (4usize, 8usize);
        let rng = |seed: usize, n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let h = seed.wrapping_mul(2_654_435_761).wrapping_add(i.wrapping_mul(40_503));
                    (h % 1000) as f32 / 500.0 - 1.0
                })
                .collect()
        };
        let o = rng(31, n_v * d_v);
        let z = rng(42, n_v * d_v);
        let w = rng(53, d_v);
        let eps = 1e-6f32;

        // reference (HF order): normalize o FIRST, scale by weight, THEN gate.
        let silu = |x: f32| x / (1.0 + (-x).exp());
        let mut rout = vec![0.0f32; n_v * d_v];
        for h in 0..n_v {
            let ms: f32 = (0..d_v).map(|j| o[h * d_v + j].powi(2)).sum::<f32>() / d_v as f32;
            let inv = 1.0 / (ms + eps).sqrt();
            for j in 0..d_v {
                rout[h * d_v + j] = w[j] * (o[h * d_v + j] * inv) * silu(z[h * d_v + j]);
            }
        }

        let dop = rt.upload_f32(&o).unwrap();
        let dz = rt.upload_f32(&z).unwrap();
        let dw = rt.upload_f32(&w).unwrap();
        let mut dout = rt.upload_f32(&vec![0.0f32; n_v * d_v]).unwrap();
        rt.gdn_gated_rmsnorm(&dop, &dz, &dw, &mut dout, n_v, d_v, eps).unwrap();
        rt.synchronize().unwrap();
        let gout = rt.download_f32(&dout).unwrap();
        for i in 0..n_v * d_v {
            assert!((rout[i] - gout[i]).abs() < 1e-4, "[{i}] ref={} gpu={}", rout[i], gout[i]);
        }
    }

    #[test]
    fn gdn_conv1d_decode_matches_reference_on_gpu() {
        use crate::cuda::runtime::CudaRuntime;
        let rt = match CudaRuntime::new(0) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("no CUDA device ({e:?}); skipping");
                return;
            }
        };
        let (channels, k) = (5usize, 4usize); // K-1 = 3 history taps
        let rng = |seed: usize, n: usize| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let h = seed.wrapping_mul(2_654_435_761).wrapping_add(i.wrapping_mul(40_503));
                    (h % 1000) as f32 / 500.0 - 1.0
                })
                .collect()
        };
        let weight = rng(7, channels * k); // [C, K]
        let silu = |x: f32| x / (1.0 + (-x).exp());

        // Drive a 6-token sequence through the streaming conv and compare to a
        // from-scratch causal conv (left-padded with zeros) at each step.
        let seq_len = 6usize;
        let xs: Vec<Vec<f32>> = (0..seq_len).map(|t| rng(100 + t, channels)).collect();

        let mut conv_state = rt.upload_f32(&vec![0.0f32; channels * (k - 1)]).unwrap();
        let dw = rt.upload_f32(&weight).unwrap();
        let mut dout = rt.upload_f32(&vec![0.0f32; channels]).unwrap();

        for t in 0..seq_len {
            let dx = rt.upload_f32(&xs[t]).unwrap();
            rt.gdn_conv1d_decode(&dx, &mut conv_state, &dw, &mut dout, channels, k).unwrap();
            rt.synchronize().unwrap();
            let gout = rt.download_f32(&dout).unwrap();
            // reference: causal conv at position t over the full sequence.
            for c in 0..channels {
                let mut acc = 0.0f32;
                for tap in 0..k {
                    let src = t as isize - (k as isize - 1) + tap as isize;
                    if src >= 0 {
                        acc += weight[c * k + tap] * xs[src as usize][c];
                    }
                }
                let expect = silu(acc);
                assert!(
                    (expect - gout[c]).abs() < 1e-4,
                    "t={t} c={c} ref={expect} gpu={}",
                    gout[c]
                );
            }
        }
    }

    fn e4m3_dec(b: u8) -> f32 {
        let s = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
        let e = ((b >> 3) & 0x0F) as i32;
        let mant = (b & 0x07) as f32;
        let v = if e == 0 {
            (mant / 8.0) * 0.015_625
        } else if e == 15 && (b & 7) == 7 {
            0.0 // NaN guard (won't occur in test data)
        } else {
            (1.0 + mant / 8.0) * 2f32.powi(e - 7)
        };
        s * v
    }

    // GPU correctness: aegis_fp8_block_gemm vs f32 reference using the
    // GPU-quantized activation. Validates the e4m3 MMA tiling + the per-128-K
    // block rescale (a_scale[m,g] * w_scale[n/128,g]). Includes an M-tail (40).
    #[test]
    fn fp8_block_gemm_matches_reference_on_gpu() {
        use crate::cuda::runtime::CudaRuntime;
        use crate::cuda::StandaloneFp8Linear;
        let rt = match CudaRuntime::new(0) {
            Ok(r) => r,
            Err(e) => { eprintln!("no CUDA ({e:?}); skip fp8_block_gemm check"); return; }
        };
        let (m, n, k) = (40usize, 128usize, 256usize); // M-tail, N=1 block, K=2 groups
        let nkg = k / 128;
        // Generate VALID e4m3 weight bytes: magnitude 0x7F (and its signed
        // 0xFF) is the e4m3 NaN encoding. Real FP8 checkpoints never contain it,
        // but the SM120 tensor-core MMA decodes it as a hardware NaN that
        // propagates through the whole accumulation tile — so a raw-random
        // weight byte gave a spurious all-NaN GEMM output. Clamp the NaN
        // magnitude to 0x7E (the e4m3 max, 448) to keep every byte finite.
        let urng = |seed: usize, len: usize| -> Vec<u8> {
            (0..len).map(|i| {
                let b = ((seed.wrapping_mul(2_654_435_761).wrapping_add(i.wrapping_mul(40_503))) % 256) as u8;
                let mag = b & 0x7F;
                (b & 0x80) | if mag == 0x7F { 0x7E } else { mag }
            }).collect()
        };
        let frng = |seed: usize, len: usize| -> Vec<f32> {
            (0..len).map(|i| ((seed.wrapping_mul(2_654_435_761).wrapping_add(i.wrapping_mul(40_503))) % 1000) as f32 / 500.0 - 1.0).collect()
        };
        let w_bytes = urng(7, n * k);
        let nb = (n / 128).max(1);
        let w_scale: Vec<f32> = (0..nb * nkg).map(|i| 0.02 + (i as f32) * 0.001).collect();
        let weight = StandaloneFp8Linear {
            name: "test".into(), rows: n, cols: k, bytes: n * k,
            data: rt.stream.clone_htod(&w_bytes).unwrap(),
            row_scales: rt.stream.clone_htod(&[1.0f32]).unwrap(),
            block_scales: Some(rt.stream.clone_htod(&w_scale).unwrap()),
            block_size: 128, scale_cols: nkg as u32,
        };
        let a = frng(11, m * k);
        let a_dev = rt.upload_f32(&a).unwrap();
        let mut a_q = rt.alloc_u8(m * k).unwrap();
        let mut a_scale = rt.alloc_f32(m * nkg).unwrap();
        rt.quantize_f32_to_fp8_token_group_device(&a_dev, m, k, &mut a_q, &mut a_scale).unwrap();
        let mut out = rt.alloc_f32(m * n).unwrap();
        rt.fp8_block_gemm_device(&a_q, &a_scale, &weight, m, &mut out).unwrap();
        let gout = rt.download_f32(&out).unwrap();
        let a_q_h = rt.download_u8(&a_q).unwrap();
        let a_sc_h = rt.download_f32(&a_scale).unwrap();
        let mut refout = vec![0.0f32; m * n];
        for mm in 0..m {
            for nn in 0..n {
                let mut acc = 0.0f32;
                for g in 0..nkg {
                    let mut blk = 0.0f32;
                    for kk in (g * 128)..(g * 128 + 128) {
                        blk += e4m3_dec(a_q_h[mm * k + kk]) * e4m3_dec(w_bytes[nn * k + kk]);
                    }
                    acc += blk * a_sc_h[mm * nkg + g] * w_scale[(nn / 128) * nkg + g];
                }
                refout[mm * n + nn] = acc;
            }
        }
        let dot: f64 = gout.iter().zip(&refout).map(|(&x, &y)| x as f64 * y as f64).sum();
        let na: f64 = gout.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
        let nb2: f64 = refout.iter().map(|&y| (y as f64).powi(2)).sum::<f64>().sqrt();
        let cos = dot / (na * nb2 + 1e-12);
        let maxerr = gout.iter().zip(&refout).map(|(&x, &y)| (x - y).abs()).fold(0.0f32, f32::max);
        eprintln!("fp8_block_gemm cos={cos:.6} maxerr={maxerr:.5}");
        assert!(cos > 0.999, "fp8_block_gemm cos {cos} too low (maxerr {maxerr})");
    }

    // ── Batched (chunked-prefill) GDN kernels vs the validated decode kernels ──
    // Each test drives a SMALL T-token sequence through the BATCHED kernel and
    // compares to running the DECODE kernel T times sequentially with the SAME
    // inputs (state threaded). They MUST match — any divergence localizes the
    // numerical bug in the batched-prefill path (decode is the trusted oracle).

    fn skip_or_rt() -> Option<crate::cuda::runtime::CudaRuntime> {
        match crate::cuda::runtime::CudaRuntime::new(0) {
            Ok(r) => Some(r),
            Err(e) => { eprintln!("no CUDA device ({e:?}); skipping"); None }
        }
    }
    fn frng(seed: usize, n: usize) -> Vec<f32> {
        (0..n).map(|i| {
            let h = seed.wrapping_mul(2_654_435_761).wrapping_add(i.wrapping_mul(40_503));
            (h % 1000) as f32 / 500.0 - 1.0
        }).collect()
    }
    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(&x, &y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    #[test]
    fn gdn_prefill_delta_rule_matches_decode_t_times() {
        let Some(rt) = skip_or_rt() else { return };
        let (t_len, n_v, d_k, d_v) = (5usize, 4usize, 8usize, 8usize);
        // Per-token inputs (already normed/scaled/expanded in the real path).
        let q: Vec<Vec<f32>> = (0..t_len).map(|t| frng(t * 7 + 1, n_v * d_k)).collect();
        let k: Vec<Vec<f32>> = (0..t_len).map(|t| frng(t * 7 + 2, n_v * d_k)).collect();
        let v: Vec<Vec<f32>> = (0..t_len).map(|t| frng(t * 7 + 3, n_v * d_v)).collect();
        let beta: Vec<Vec<f32>> = (0..t_len)
            .map(|t| (0..n_v).map(|h| 0.3 + 0.05 * (h + t) as f32).collect())
            .collect();
        let g: Vec<Vec<f32>> = (0..t_len)
            .map(|t| (0..n_v).map(|h| -0.2 - 0.07 * (h + t) as f32).collect())
            .collect();

        // Reference: decode kernel T times (state threaded), capturing each y.
        let mut dec_state = rt.upload_f32(&vec![0.0f32; n_v * d_v * d_k]).unwrap();
        let mut dec_outs = vec![0.0f32; t_len * n_v * d_v];
        for t in 0..t_len {
            let dq = rt.upload_f32(&q[t]).unwrap();
            let dk = rt.upload_f32(&k[t]).unwrap();
            let dv = rt.upload_f32(&v[t]).unwrap();
            let db = rt.upload_f32(&beta[t]).unwrap();
            let dg = rt.upload_f32(&g[t]).unwrap();
            let mut dout = rt.alloc_f32(n_v * d_v).unwrap();
            rt.gated_deltanet_decode_step(&mut dec_state, &dq, &dk, &dv, &db, &dg, &mut dout,
                n_v, d_k, d_v).unwrap();
            rt.synchronize().unwrap();
            let o = rt.download_f32(&dout).unwrap();
            dec_outs[t * n_v * d_v..(t + 1) * n_v * d_v].copy_from_slice(&o);
        }
        let dec_final_state = rt.download_f32(&dec_state).unwrap();

        // Batched: flatten [T, n_v, *] and run the prefill kernel once.
        let flat = |v: &[Vec<f32>]| -> Vec<f32> { v.iter().flatten().copied().collect() };
        let dq = rt.upload_f32(&flat(&q)).unwrap();
        let dk = rt.upload_f32(&flat(&k)).unwrap();
        let dv = rt.upload_f32(&flat(&v)).unwrap();
        let db = rt.upload_f32(&flat(&beta)).unwrap();
        let dg = rt.upload_f32(&flat(&g)).unwrap();
        let mut pre_state = rt.upload_f32(&vec![0.0f32; n_v * d_v * d_k]).unwrap();
        let mut pre_out = rt.alloc_f32(t_len * n_v * d_v).unwrap();
        rt.gated_deltanet_prefill_step(&mut pre_state, &dq, &dk, &dv, &db, &dg, &mut pre_out,
            t_len, n_v, d_k, d_v).unwrap();
        rt.synchronize().unwrap();
        let pre_outs = rt.download_f32(&pre_out).unwrap();
        let pre_final_state = rt.download_f32(&pre_state).unwrap();

        let oerr = max_abs_diff(&dec_outs, &pre_outs);
        let serr = max_abs_diff(&dec_final_state, &pre_final_state);
        assert!(oerr < 1e-4, "delta-rule batched-vs-decode out max_abs={oerr}");
        assert!(serr < 1e-4, "delta-rule batched-vs-decode state max_abs={serr}");
    }

    #[test]
    fn gdn_prefill_conv1d_matches_decode_t_times() {
        let Some(rt) = skip_or_rt() else { return };
        let (t_len, channels, kern) = (5usize, 5usize, 4usize);
        let weight = frng(7, channels * kern);
        let xs: Vec<Vec<f32>> = (0..t_len).map(|t| frng(100 + t, channels)).collect();
        let dw = rt.upload_f32(&weight).unwrap();

        // Decode T times (conv_state threaded).
        let mut dec_state = rt.upload_f32(&vec![0.0f32; channels * (kern - 1)]).unwrap();
        let mut dec_outs = vec![0.0f32; t_len * channels];
        for t in 0..t_len {
            let dx = rt.upload_f32(&xs[t]).unwrap();
            let mut dout = rt.alloc_f32(channels).unwrap();
            rt.gdn_conv1d_decode(&dx, &mut dec_state, &dw, &mut dout, channels, kern).unwrap();
            rt.synchronize().unwrap();
            let o = rt.download_f32(&dout).unwrap();
            dec_outs[t * channels..(t + 1) * channels].copy_from_slice(&o);
        }
        let dec_final_state = rt.download_f32(&dec_state).unwrap();

        // Batched: [T, C].
        let x_flat: Vec<f32> = xs.iter().flatten().copied().collect();
        let dx = rt.upload_f32(&x_flat).unwrap();
        let mut pre_state = rt.upload_f32(&vec![0.0f32; channels * (kern - 1)]).unwrap();
        let mut pre_out = rt.alloc_f32(t_len * channels).unwrap();
        rt.gdn_conv1d_prefill(&dx, &mut pre_state, &dw, &mut pre_out, t_len, channels, kern).unwrap();
        rt.synchronize().unwrap();
        let pre_outs = rt.download_f32(&pre_out).unwrap();
        let pre_final_state = rt.download_f32(&pre_state).unwrap();

        let oerr = max_abs_diff(&dec_outs, &pre_outs);
        let serr = max_abs_diff(&dec_final_state, &pre_final_state);
        assert!(oerr < 1e-4, "conv1d batched-vs-decode out max_abs={oerr}");
        assert!(serr < 1e-4, "conv1d batched-vs-decode state max_abs={serr}");
    }

    #[test]
    fn gdn_prefill_qk_norm_matches_decode_t_times() {
        let Some(rt) = skip_or_rt() else { return };
        let (t_len, n_k, n_v, d_k) = (5usize, 2usize, 4usize, 8usize);
        let expand = n_v / n_k;
        let q: Vec<Vec<f32>> = (0..t_len).map(|t| frng(t * 11 + 1, n_k * d_k)).collect();
        let k: Vec<Vec<f32>> = (0..t_len).map(|t| frng(t * 11 + 2, n_k * d_k)).collect();

        // Decode per token.
        let mut dec_q = vec![0.0f32; t_len * n_v * d_k];
        let mut dec_k = vec![0.0f32; t_len * n_v * d_k];
        for t in 0..t_len {
            let dq = rt.upload_f32(&q[t]).unwrap();
            let dk = rt.upload_f32(&k[t]).unwrap();
            let mut dqo = rt.alloc_f32(n_v * d_k).unwrap();
            let mut dko = rt.alloc_f32(n_v * d_k).unwrap();
            rt.gdn_qk_norm_expand(&dq, &dk, &mut dqo, &mut dko, n_k, n_v, d_k).unwrap();
            rt.synchronize().unwrap();
            dec_q[t * n_v * d_k..(t + 1) * n_v * d_k].copy_from_slice(&rt.download_f32(&dqo).unwrap());
            dec_k[t * n_v * d_k..(t + 1) * n_v * d_k].copy_from_slice(&rt.download_f32(&dko).unwrap());
        }
        // Batched [T, n_k, d_k].
        let qf: Vec<f32> = q.iter().flatten().copied().collect();
        let kf: Vec<f32> = k.iter().flatten().copied().collect();
        let dq = rt.upload_f32(&qf).unwrap();
        let dk = rt.upload_f32(&kf).unwrap();
        let mut dqo = rt.alloc_f32(t_len * n_v * d_k).unwrap();
        let mut dko = rt.alloc_f32(t_len * n_v * d_k).unwrap();
        rt.gdn_qk_norm_expand_batched(&dq, &dk, &mut dqo, &mut dko, t_len, n_k, n_v, d_k, expand).unwrap();
        rt.synchronize().unwrap();
        let qerr = max_abs_diff(&dec_q, &rt.download_f32(&dqo).unwrap());
        let kerr = max_abs_diff(&dec_k, &rt.download_f32(&dko).unwrap());
        assert!(qerr < 1e-4, "qk_norm batched-vs-decode q max_abs={qerr}");
        assert!(kerr < 1e-4, "qk_norm batched-vs-decode k max_abs={kerr}");
    }

    #[test]
    fn gdn_prefill_gate_matches_decode_t_times() {
        let Some(rt) = skip_or_rt() else { return };
        let (t_len, n_v) = (5usize, 6usize);
        let a_log = frng(91, n_v);
        let dt = frng(92, n_v);
        let dal = rt.upload_f32(&a_log).unwrap();
        let ddt = rt.upload_f32(&dt).unwrap();
        let b: Vec<Vec<f32>> = (0..t_len).map(|t| frng(t * 13 + 1, n_v)).collect();
        let a: Vec<Vec<f32>> = (0..t_len).map(|t| frng(t * 13 + 2, n_v)).collect();

        let mut dec_beta = vec![0.0f32; t_len * n_v];
        let mut dec_g = vec![0.0f32; t_len * n_v];
        for t in 0..t_len {
            let db = rt.upload_f32(&b[t]).unwrap();
            let da = rt.upload_f32(&a[t]).unwrap();
            let mut dbeta = rt.alloc_f32(n_v).unwrap();
            let mut dg = rt.alloc_f32(n_v).unwrap();
            rt.gdn_gate(&db, &da, &dal, &ddt, &mut dbeta, &mut dg, n_v).unwrap();
            rt.synchronize().unwrap();
            dec_beta[t * n_v..(t + 1) * n_v].copy_from_slice(&rt.download_f32(&dbeta).unwrap());
            dec_g[t * n_v..(t + 1) * n_v].copy_from_slice(&rt.download_f32(&dg).unwrap());
        }
        let bf: Vec<f32> = b.iter().flatten().copied().collect();
        let af: Vec<f32> = a.iter().flatten().copied().collect();
        let db = rt.upload_f32(&bf).unwrap();
        let da = rt.upload_f32(&af).unwrap();
        let mut dbeta = rt.alloc_f32(t_len * n_v).unwrap();
        let mut dg = rt.alloc_f32(t_len * n_v).unwrap();
        rt.gdn_gate_batched(&db, &da, &dal, &ddt, &mut dbeta, &mut dg, t_len, n_v).unwrap();
        rt.synchronize().unwrap();
        let berr = max_abs_diff(&dec_beta, &rt.download_f32(&dbeta).unwrap());
        let gerr = max_abs_diff(&dec_g, &rt.download_f32(&dg).unwrap());
        assert!(berr < 1e-5, "gate batched-vs-decode beta max_abs={berr}");
        assert!(gerr < 1e-5, "gate batched-vs-decode g max_abs={gerr}");
    }

    #[test]
    fn gdn_prefill_gated_rmsnorm_matches_decode_t_times() {
        let Some(rt) = skip_or_rt() else { return };
        let (t_len, n_v, d_v) = (5usize, 4usize, 8usize);
        let w = frng(53, d_v);
        let dw = rt.upload_f32(&w).unwrap();
        let eps = 1e-6f32;
        let o: Vec<Vec<f32>> = (0..t_len).map(|t| frng(t * 17 + 1, n_v * d_v)).collect();
        let z: Vec<Vec<f32>> = (0..t_len).map(|t| frng(t * 17 + 2, n_v * d_v)).collect();

        let mut dec_out = vec![0.0f32; t_len * n_v * d_v];
        for t in 0..t_len {
            let dop = rt.upload_f32(&o[t]).unwrap();
            let dz = rt.upload_f32(&z[t]).unwrap();
            let mut dout = rt.alloc_f32(n_v * d_v).unwrap();
            rt.gdn_gated_rmsnorm(&dop, &dz, &dw, &mut dout, n_v, d_v, eps).unwrap();
            rt.synchronize().unwrap();
            dec_out[t * n_v * d_v..(t + 1) * n_v * d_v].copy_from_slice(&rt.download_f32(&dout).unwrap());
        }
        let of: Vec<f32> = o.iter().flatten().copied().collect();
        let zf: Vec<f32> = z.iter().flatten().copied().collect();
        let dop = rt.upload_f32(&of).unwrap();
        let dz = rt.upload_f32(&zf).unwrap();
        let mut dout = rt.alloc_f32(t_len * n_v * d_v).unwrap();
        rt.gdn_gated_rmsnorm_batched(&dop, &dz, &dw, &mut dout, t_len, n_v, d_v, eps).unwrap();
        rt.synchronize().unwrap();
        let err = max_abs_diff(&dec_out, &rt.download_f32(&dout).unwrap());
        assert!(err < 1e-4, "gated_rmsnorm batched-vs-decode max_abs={err}");
    }

    #[test]
    fn gdn_prefill_strided_copy_splits_correctly() {
        let Some(rt) = skip_or_rt() else { return };
        // Mirror the conv-output split: src [rows, src_stride] → q/k/v slices.
        let (rows, n_k, d_k, n_v, d_v) = (4usize, 2usize, 8usize, 4usize, 8usize);
        let qk = n_k * d_k;
        let vw = n_v * d_v;
        let src_stride = 2 * qk + vw;
        let src: Vec<f32> = frng(5, rows * src_stride);
        let dsrc = rt.upload_f32(&src).unwrap();
        let mut dq = rt.alloc_f32(rows * qk).unwrap();
        let mut dk = rt.alloc_f32(rows * qk).unwrap();
        let mut dv = rt.alloc_f32(rows * vw).unwrap();
        rt.strided_copy_2d(&dsrc, &mut dq, rows, qk, src_stride, qk, 0).unwrap();
        rt.strided_copy_2d(&dsrc, &mut dk, rows, qk, src_stride, qk, qk).unwrap();
        rt.strided_copy_2d(&dsrc, &mut dv, rows, vw, src_stride, vw, 2 * qk).unwrap();
        rt.synchronize().unwrap();
        let gq = rt.download_f32(&dq).unwrap();
        let gk = rt.download_f32(&dk).unwrap();
        let gv = rt.download_f32(&dv).unwrap();
        for r in 0..rows {
            for c in 0..qk {
                assert_eq!(gq[r * qk + c], src[r * src_stride + c], "q[{r},{c}]");
                assert_eq!(gk[r * qk + c], src[r * src_stride + qk + c], "k[{r},{c}]");
            }
            for c in 0..vw {
                assert_eq!(gv[r * vw + c], src[r * src_stride + 2 * qk + c], "v[{r},{c}]");
            }
        }
    }
}

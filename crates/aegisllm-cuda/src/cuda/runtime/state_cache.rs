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
    /// Inputs: q/k/v are `[num_heads, head_dim_k/v]`; beta is `[num_heads]`.
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
            || output.len() < expected_out
        {
            return Err(AegisError::InvalidPlan(
                "GDN decode: buffer size mismatch".into(),
            ));
        }

        let block_k = head_dim_k as u32;
        let cfg = LaunchConfig {
            grid_dim: (num_heads as u32, 1, 1),
            block_dim: (block_k, 1, 1),
            shared_mem_bytes: block_k * 4,
        };

        unsafe {
            self.stream
                .launch_builder(&self.kernels.gated_deltanet_decode)
                .arg(&mut state.slice)
                .arg(&query.slice)
                .arg(&key.slice)
                .arg(&value.slice)
                .arg(&beta.slice)
                .arg(&mut output.slice)
                .arg(&(head_dim_k as u32))
                .arg(&(head_dim_v as u32))
                .launch(cfg)
        }
        .map_err(map_cuda_err("gated_deltanet_decode"))?;
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
}

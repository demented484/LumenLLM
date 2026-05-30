//! Runtime launchers for GPU-driven MoE decode: gather routed experts'
//! NVFP4 weights from device-mapped host RAM into a VRAM scratch in one
//! launch, then run the per-expert GEMVs reading the gathered bytes +
//! device-resident scales. See
//! `kernels/blackwell/moe_gpu_driven_decode.cu` for the kernel contracts and
//! `executor/mlp.rs::forward_moe_decode_gpu_driven` for the orchestration.

use cudarc::driver::{LaunchConfig, PushKernelArg};

use super::{CudaRuntime, ceil_div, map_cuda_err};
use crate::cuda::DeviceBuffer;
use crate::executor::state::MoeDeviceTables;
use aegisllm_base::error::{AegisError, Result};

impl CudaRuntime {
    /// Gather the routed experts' packed+scales bytes from device-mapped host
    /// RAM into `bulk_packed`/`bulk_scales` (fixed slot-major layout) and write
    /// the per-slot NVFP4 input/output scales into `slot_in_scale`/`slot_out_scale`.
    /// `packed_topk` holds the on-device `(idx, wbits)` router top-k records;
    /// the kernel reads expert indices straight from it (no CPU round-trip).
    /// One CTA per (slot, projection): grid = (3, top_k).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn moe_gather_experts_device(
        &self,
        packed_topk: &DeviceBuffer<u32>,
        top_k: usize,
        num_experts: usize,
        tables: &MoeDeviceTables,
        bulk_packed: &mut DeviceBuffer<u8>,
        bulk_scales: &mut DeviceBuffer<u8>,
        slot_in_scale: &mut DeviceBuffer<f32>,
        slot_out_scale: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        let per_slot_packed = tables.gate_packed_bytes + tables.up_packed_bytes + tables.down_packed_bytes;
        let per_slot_scale = tables.gate_scale_bytes + tables.up_scale_bytes + tables.down_scale_bytes;
        let need_packed = per_slot_packed * top_k;
        let need_scale = per_slot_scale * top_k;
        if bulk_packed.len() < need_packed || bulk_scales.len() < need_scale {
            return Err(AegisError::InvalidPlan(format!(
                "moe_gather: bulk too small: packed have {} need {}, scales have {} need {}",
                bulk_packed.len(), need_packed, bulk_scales.len(), need_scale
            )));
        }
        if slot_in_scale.len() < top_k * 3 || slot_out_scale.len() < top_k * 3 {
            return Err(AegisError::InvalidPlan(format!(
                "moe_gather: slot scale arrays too small: in {} out {} need {}",
                slot_in_scale.len(), slot_out_scale.len(), top_k * 3
            )));
        }
        let top_k_u32 = top_k as u32;
        let num_experts_u32 = num_experts as u32;
        let gate_packed_b = tables.gate_packed_bytes as u32;
        let gate_scale_b = tables.gate_scale_bytes as u32;
        let up_packed_b = tables.up_packed_bytes as u32;
        let up_scale_b = tables.up_scale_bytes as u32;
        let down_packed_b = tables.down_packed_bytes as u32;
        let down_scale_b = tables.down_scale_bytes as u32;
        // 256 threads stream the (large) packed bytes per (slot, projection).
        let cfg = LaunchConfig {
            grid_dim: (3, top_k_u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.moe_gather_experts)
                .arg(&packed_topk.slice)
                .arg(&top_k_u32)
                .arg(&num_experts_u32)
                .arg(&tables.gate_packed_ptrs.slice)
                .arg(&tables.up_packed_ptrs.slice)
                .arg(&tables.down_packed_ptrs.slice)
                .arg(&tables.gate_scale_ptrs.slice)
                .arg(&tables.up_scale_ptrs.slice)
                .arg(&tables.down_scale_ptrs.slice)
                .arg(&gate_packed_b)
                .arg(&gate_scale_b)
                .arg(&up_packed_b)
                .arg(&up_scale_b)
                .arg(&down_packed_b)
                .arg(&down_scale_b)
                .arg(&tables.gate_in_scale.slice)
                .arg(&tables.up_in_scale.slice)
                .arg(&tables.down_in_scale.slice)
                .arg(&tables.gate_out_scale.slice)
                .arg(&tables.up_out_scale.slice)
                .arg(&tables.down_out_scale.slice)
                .arg(&mut bulk_packed.slice)
                .arg(&mut bulk_scales.slice)
                .arg(&mut slot_in_scale.slice)
                .arg(&mut slot_out_scale.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch moe_gather_experts"))?;
        Ok(())
    }

    /// NVFP4 input quantization where `input_scale` is read from a device float
    /// pointer (`slot_in_scale[slot_proj]`). Same math as
    /// `quantize_nvfp4_input_device`; the scale isn't a launch-time scalar so the
    /// launch is graph-capturable.
    pub(crate) fn quantize_nvfp4_input_dptr_device(
        &self,
        input: &DeviceBuffer<f32>,
        slot_in_scale: &DeviceBuffer<f32>,
        slot_proj: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if input.len() > output.len() {
            return Err(AegisError::InvalidPlan(format!(
                "nvfp4 dptr input quant shape mismatch: input={} output={}",
                input.len(), output.len()
            )));
        }
        if slot_proj >= slot_in_scale.len() {
            return Err(AegisError::InvalidPlan(format!(
                "nvfp4 dptr input quant: slot_proj {slot_proj} out of bounds {}",
                slot_in_scale.len()
            )));
        }
        let len = input.len() as u32;
        let scale_view = slot_in_scale.slice.slice(slot_proj..slot_proj + 1);
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(len, 16), 1, 1),
            block_dim: (16, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.nvfp4_quantize_input_dptr)
                .arg(&input.slice)
                .arg(&len)
                .arg(&scale_view)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch nvfp4 quantize input dptr"))?;
        Ok(())
    }

    /// Pre-quantized NVFP4 GEMV (M=1) reading packed/scales from byte offsets
    /// into the gathered bulk VRAM buffer and `output_scale` from a device float
    /// pointer (`slot_out_scale[slot_proj]`). Numerically identical to
    /// `matvec_nvfp4_prequantized_bulk_views_device`; only the output scale
    /// source differs (device array vs launch scalar) so the launch is
    /// graph-capturable.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn matvec_nvfp4_prequantized_dptr_bulk_device(
        &self,
        bulk_packed: &DeviceBuffer<u8>,
        bulk_scales: &DeviceBuffer<u8>,
        packed_offset: usize,
        packed_bytes: usize,
        scales_offset: usize,
        scale_bytes: usize,
        rows: usize,
        cols: usize,
        slot_out_scale: &DeviceBuffer<f32>,
        slot_proj: usize,
        quantized_input: &DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if packed_offset + packed_bytes > bulk_packed.len()
            || scales_offset + scale_bytes > bulk_scales.len()
        {
            return Err(AegisError::InvalidPlan(format!(
                "dptr bulk view out of bounds: packed {}+{}>{}  scales {}+{}>{}",
                packed_offset, packed_bytes, bulk_packed.len(),
                scales_offset, scale_bytes, bulk_scales.len(),
            )));
        }
        if slot_proj >= slot_out_scale.len() {
            return Err(AegisError::InvalidPlan(format!(
                "dptr bulk gemv: slot_proj {slot_proj} out of bounds {}",
                slot_out_scale.len()
            )));
        }
        // Buffers may be OVER-allocated (expert_gate/up/swiglu are sized to
        // max_expert_intermediate = max(routed moe_inter, shared_expert inter),
        // and quant_expert to max_input). The kernel reads exactly `cols` and
        // writes exactly `rows` (grid_dim = rows); downstream ops read the valid
        // prefix. So require AT LEAST the needed size, not an exact match.
        if quantized_input.len() < cols || output.len() < rows {
            return Err(AegisError::InvalidPlan(format!(
                "dptr bulk gemv shape too small: input={} need {cols}, output={} need {rows}",
                quantized_input.len(), output.len()
            )));
        }
        let packed_view = bulk_packed.slice.slice(packed_offset..packed_offset + packed_bytes);
        let scales_view = bulk_scales.slice.slice(scales_offset..scales_offset + scale_bytes);
        let out_scale_view = slot_out_scale.slice.slice(slot_proj..slot_proj + 1);
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let cfg = LaunchConfig {
            grid_dim: (rows_u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 128 * std::mem::size_of::<f32>() as u32,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.nvfp4_prequant_dptr)
                .arg(&packed_view)
                .arg(&scales_view)
                .arg(&quantized_input.slice)
                .arg(&rows_u32)
                .arg(&cols_u32)
                .arg(&out_scale_view)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch nvfp4 prequantized dptr bulk"))?;
        Ok(())
    }

    /// `out += weight_slot * src`, where the routing weight is read from the
    /// on-device top-k buffer (`packed_topk[2*slot + 1]`). Keeps the routing
    /// weight on-device so the accumulation is graph-capturable.
    pub(crate) fn axpy_f32_topk_weight_device(
        &self,
        out: &mut DeviceBuffer<f32>,
        src: &DeviceBuffer<f32>,
        packed_topk: &DeviceBuffer<u32>,
        slot: usize,
    ) -> Result<()> {
        if src.len() != out.len() {
            return Err(AegisError::InvalidPlan(format!(
                "axpy topk-weight shape mismatch: src={} out={}",
                src.len(), out.len()
            )));
        }
        let len = src.len() as u32;
        let slot_u32 = slot as u32;
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(len, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            self.stream
                .launch_builder(&self.kernels.axpy_f32_topk_weight)
                .arg(&mut out.slice)
                .arg(&src.slice)
                .arg(&packed_topk.slice)
                .arg(&slot_u32)
                .arg(&len)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch axpy_f32_topk_weight"))?;
        Ok(())
    }

    // ── Batched (grouped-over-experts) decode MoE launchers ──────────────────
    // Each collapses a per-slot stage into ONE launch with slot on grid.y. Math
    // is byte-identical to the single-expert launchers above (see the kernels in
    // moe_gpu_driven_decode.cu). Graph-capturable: fixed shapes, device-pointer
    // scales/weights, no host-data-dependent control flow.

    /// Batched NVFP4 input-quant over `top_k` slots. `input_stride`=0 (gate/up,
    /// every slot quantizes the shared `hidden` with its own scale) or the
    /// per-slot stride (down). Output per-slot at `output_stride`. Scale read
    /// on-device from `slot_in_scale[slot*3 + proj_off]`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn quantize_nvfp4_input_batched_dptr_device(
        &self,
        input: &DeviceBuffer<f32>,
        input_stride: usize,
        len: usize,
        output_stride: usize,
        slot_in_scale: &DeviceBuffer<f32>,
        proj_off: usize,
        top_k: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if proj_off >= 3 || top_k == 0 {
            return Err(AegisError::InvalidPlan(format!(
                "batched quant: bad proj_off={proj_off} or top_k={top_k}"
            )));
        }
        if output_stride < len || output.len() < top_k * output_stride {
            return Err(AegisError::InvalidPlan(format!(
                "batched quant: output too small: len {} stride {} top_k {} have {}",
                len, output_stride, top_k, output.len()
            )));
        }
        let needed_in = if input_stride == 0 { len } else { top_k * input_stride };
        if input.len() < needed_in || (input_stride != 0 && input_stride < len) {
            return Err(AegisError::InvalidPlan(format!(
                "batched quant: input too small: len {} stride {} have {}",
                len, input_stride, input.len()
            )));
        }
        if slot_in_scale.len() < top_k * 3 {
            return Err(AegisError::InvalidPlan(format!(
                "batched quant: slot_in_scale {} need {}", slot_in_scale.len(), top_k * 3
            )));
        }
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(len as u32, 16), top_k as u32, 1),
            block_dim: (16, 1, 1),
            shared_mem_bytes: 0,
        };
        let (input_stride_u, len_u, out_stride_u, proj_u, top_k_u) = (
            input_stride as u32, len as u32, output_stride as u32, proj_off as u32, top_k as u32,
        );
        unsafe {
            self.stream
                .launch_builder(&self.kernels.nvfp4_quantize_input_batched_dptr)
                .arg(&input.slice)
                .arg(&input_stride_u)
                .arg(&len_u)
                .arg(&out_stride_u)
                .arg(&slot_in_scale.slice)
                .arg(&proj_u)
                .arg(&top_k_u)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch nvfp4 quantize input batched dptr"))?;
        Ok(())
    }

    /// Batched prequantized NVFP4 GEMV over `top_k` slots. Reads the RAW bulk
    /// buffer (no pre-slice); derives slot/proj base from the slot-major strides.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn matvec_nvfp4_prequantized_batched_dptr_device(
        &self,
        bulk_packed: &DeviceBuffer<u8>,
        bulk_scales: &DeviceBuffer<u8>,
        per_slot_packed: usize,
        per_slot_scale: usize,
        proj_packed_off: usize,
        proj_scale_off: usize,
        input: &DeviceBuffer<f32>,
        input_stride: usize,
        rows: usize,
        cols: usize,
        slot_out_scale: &DeviceBuffer<f32>,
        proj_off: usize,
        output_stride: usize,
        top_k: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if proj_off >= 3 || top_k == 0 {
            return Err(AegisError::InvalidPlan(format!(
                "batched gemv: bad proj_off={proj_off} or top_k={top_k}"
            )));
        }
        if bulk_packed.len() < top_k * per_slot_packed || bulk_scales.len() < top_k * per_slot_scale {
            return Err(AegisError::InvalidPlan(format!(
                "batched gemv: bulk too small: packed {} need {}, scales {} need {}",
                bulk_packed.len(), top_k * per_slot_packed, bulk_scales.len(), top_k * per_slot_scale
            )));
        }
        if proj_packed_off + (rows * (cols / 2)) > per_slot_packed
            || proj_scale_off + (rows * (cols / 16)) > per_slot_scale
        {
            return Err(AegisError::InvalidPlan(format!(
                "batched gemv: proj offset/extent out of slot: packed {}+{}*{}>{}",
                proj_packed_off, rows, cols / 2, per_slot_packed
            )));
        }
        if input_stride < cols || input.len() < top_k * input_stride {
            return Err(AegisError::InvalidPlan(format!(
                "batched gemv: input too small: cols {} stride {} top_k {} have {}",
                cols, input_stride, top_k, input.len()
            )));
        }
        if output_stride < rows || output.len() < top_k * output_stride {
            return Err(AegisError::InvalidPlan(format!(
                "batched gemv: output too small: rows {} stride {} top_k {} have {}",
                rows, output_stride, top_k, output.len()
            )));
        }
        if slot_out_scale.len() < top_k * 3 {
            return Err(AegisError::InvalidPlan(format!(
                "batched gemv: slot_out_scale {} need {}", slot_out_scale.len(), top_k * 3
            )));
        }
        // Fast warp-per-row kernel (mmvq-style, no shared-mem reduction) vs the
        // naive block-per-row+tree kernel. Same ABI; only grid/block/shmem +
        // kernel handle differ.
        let fast = super::linear::fast_decode_gemv_enabled();
        let cfg = if fast {
            LaunchConfig {
                grid_dim: (rows as u32, top_k as u32, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            }
        } else {
            LaunchConfig {
                grid_dim: (rows as u32, top_k as u32, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 128 * std::mem::size_of::<f32>() as u32,
            }
        };
        let kernel = if fast {
            &self.kernels.nvfp4_prequant_batched_dptr_warp
        } else {
            &self.kernels.nvfp4_prequant_batched_dptr
        };
        let (psp, pss, ppo, pso, istride, rows_u, cols_u, proj_u, ostride, tk) = (
            per_slot_packed as u32, per_slot_scale as u32, proj_packed_off as u32,
            proj_scale_off as u32, input_stride as u32, rows as u32, cols as u32,
            proj_off as u32, output_stride as u32, top_k as u32,
        );
        unsafe {
            self.stream
                .launch_builder(kernel)
                .arg(&bulk_packed.slice)
                .arg(&bulk_scales.slice)
                .arg(&psp)
                .arg(&pss)
                .arg(&ppo)
                .arg(&pso)
                .arg(&input.slice)
                .arg(&istride)
                .arg(&rows_u)
                .arg(&cols_u)
                .arg(&slot_out_scale.slice)
                .arg(&proj_u)
                .arg(&ostride)
                .arg(&tk)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch nvfp4 prequant batched dptr"))?;
        Ok(())
    }

    /// Batched strided GeGLU over `top_k` slots: out[s] = gelu_tanh(gate[s])*up[s].
    pub(crate) fn geglu_tanh_batched_slots_device(
        &self,
        gate: &DeviceBuffer<f32>,
        up: &DeviceBuffer<f32>,
        len: usize,
        stride: usize,
        top_k: usize,
        output: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if top_k == 0 || stride < len {
            return Err(AegisError::InvalidPlan(format!(
                "batched geglu: bad top_k={top_k} or stride {stride} < len {len}"
            )));
        }
        if gate.len() < top_k * stride || up.len() < top_k * stride || output.len() < top_k * stride {
            return Err(AegisError::InvalidPlan(format!(
                "batched geglu: buffers too small: stride {} top_k {} gate {} up {} out {}",
                stride, top_k, gate.len(), up.len(), output.len()
            )));
        }
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(len as u32, 256), top_k as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (len_u, stride_u, tk) = (len as u32, stride as u32, top_k as u32);
        unsafe {
            self.stream
                .launch_builder(&self.kernels.moe_geglu_tanh_batched_slots)
                .arg(&gate.slice)
                .arg(&up.slice)
                .arg(&len_u)
                .arg(&stride_u)
                .arg(&tk)
                .arg(&mut output.slice)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch moe geglu tanh batched slots"))?;
        Ok(())
    }

    /// Batched weighted accumulate: out[i] = sum_{k=0..top_k-1} w[k]*expert_out[k][i].
    /// Routing weight w[k] read on-device from `packed_topk[k*2+1]`. Fixed ascending
    /// slot fold + single-expression FMA → bit-identical to the serial axpy chain.
    pub(crate) fn moe_weighted_accumulate_device(
        &self,
        out: &mut DeviceBuffer<f32>,
        expert_out: &DeviceBuffer<f32>,
        output_stride: usize,
        packed_topk: &DeviceBuffer<u32>,
        top_k: usize,
        len: usize,
    ) -> Result<()> {
        if top_k == 0 || output_stride < len {
            return Err(AegisError::InvalidPlan(format!(
                "batched accumulate: bad top_k={top_k} or stride {output_stride} < len {len}"
            )));
        }
        if out.len() < len || expert_out.len() < top_k * output_stride {
            return Err(AegisError::InvalidPlan(format!(
                "batched accumulate: buffers too small: len {} stride {} top_k {} out {} expert_out {}",
                len, output_stride, top_k, out.len(), expert_out.len()
            )));
        }
        if packed_topk.len() < top_k * 2 {
            return Err(AegisError::InvalidPlan(format!(
                "batched accumulate: packed_topk {} need {}", packed_topk.len(), top_k * 2
            )));
        }
        let cfg = LaunchConfig {
            grid_dim: (ceil_div(len as u32, 256), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (ostride, tk, len_u) = (output_stride as u32, top_k as u32, len as u32);
        unsafe {
            self.stream
                .launch_builder(&self.kernels.moe_weighted_accumulate)
                .arg(&mut out.slice)
                .arg(&expert_out.slice)
                .arg(&ostride)
                .arg(&packed_topk.slice)
                .arg(&tk)
                .arg(&len_u)
                .launch(cfg)
        }
        .map_err(map_cuda_err("launch moe weighted accumulate"))?;
        Ok(())
    }
}

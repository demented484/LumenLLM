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
}

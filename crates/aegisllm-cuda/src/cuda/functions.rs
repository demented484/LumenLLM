use std::{env, path::Path, sync::Arc};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule};
use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};

use super::compile::nvrtc_arch_for_device;
use super::kernels::BLACKWELL_FP4_KERNEL_SRC;
use aegisllm_base::error::{AegisError, Result};

#[derive(Debug)]
pub(crate) struct CudaKernelFunctions {
    pub(crate) _module: Arc<CudaModule>,
    pub(crate) blackwell_fp4: CudaFunction,
    pub(crate) mxfp4_matvec: CudaFunction,
    pub(crate) mxfp4_matvec_4warp: CudaFunction,
    pub(crate) mxfp4_matvec_16warp: CudaFunction,
    pub(crate) mxfp4_matmul_n8: CudaFunction,
    pub(crate) mxfp4_matmul_tile_m16n32: CudaFunction,
    pub(crate) mxfp4_matmul_tile_m16n64: CudaFunction,
    pub(crate) mxfp4_matmul_qkv_tile_m16n32: CudaFunction,
    pub(crate) mxfp4_matmul_qkv_tile_m16n64: CudaFunction,
    pub(crate) mxfp4_matmul_gate_up_tile_m16n32: CudaFunction,
    pub(crate) mxfp4_matmul_gate_up_tile_m16n64: CudaFunction,
    pub(crate) split_qkv_scaled: CudaFunction,
    pub(crate) mxfp4_quantize_input: CudaFunction,
    pub(crate) swiglu_mxfp4_quantize_batched: CudaFunction,
    pub(crate) fp8_quantize_bf16_per_row: CudaFunction,
    pub(crate) fp8_matvec: CudaFunction,
    pub(crate) fp8_matmul_batched: CudaFunction,
    pub(crate) fp8_dequant_to_bf16: CudaFunction,
    pub(crate) nvfp4_reference: CudaFunction,
    pub(crate) nvfp4_reference_batched: CudaFunction,
    pub(crate) nvfp4_prequant: CudaFunction,
    pub(crate) nvfp4_prequant_batched: CudaFunction,
    pub(crate) nvfp4_prequant_batched_gemm: CudaFunction,
    pub(crate) nvfp4_prequant_batched_gemm_wmma_bf16: CudaFunction,
    pub(crate) nvfp4_grouped_prequant_gemm_wmma_bf16: CudaFunction,
    pub(crate) nvfp4_grouped_prequant_gemm_wmma_bf16_t32: CudaFunction,
    // ── MoE/NVFP4 fused entries (Phase B.4 Round 2; do not reorder, attention
    //    work is appended below). ──
    pub(crate) nvfp4_grouped_prequant_gemm_wmma_bf16_t32_dual: CudaFunction,
    // ── MoE/NVFP4 cp.async pipelined entry (Phase B.4 Round 3). Opt-in via
    //    AEGIS_NVFP4_GROUPED_T32_PIPELINE=1. ──
    pub(crate) nvfp4_grouped_prequant_gemm_wmma_bf16_t32_pipeline: CudaFunction,
    // ── MoE/NVFP4 64×64 output-tile entry (Phase B.4 Round 4). Opt-in via
    //    AEGIS_NVFP4_GROUPED_T32_BIG_ENABLE=1. 8 warps, 4×2 warp grid, 2 c_frags
    //    per warp. Eligibility: rows%64==0 AND max_tokens_per_expert>=64. ──
    pub(crate) nvfp4_grouped_prequant_gemm_wmma_bf16_t32_big: CudaFunction,
    // ── MoE/NVFP4 64×64 output-tile + cp.async pipelined B (Phase B.4 Round 5).
    //    Opt-in via AEGIS_NVFP4_GROUPED_T32_BIG_PIPELINE=1. Same eligibility
    //    as `_t32_big`; pipelines B-tile load via cp.async double-buffer. ──
    pub(crate) nvfp4_grouped_prequant_gemm_wmma_bf16_t32_big_pipeline: CudaFunction,
    pub(crate) nvfp4_quantize_input: CudaFunction,
    pub(crate) nvfp4_quantize_input_batched: CudaFunction,
    pub(crate) bf16_matvec: CudaFunction,
    pub(crate) bf16_row: CudaFunction,
    pub(crate) bf16_rows: CudaFunction,
    pub(crate) rms_norm: CudaFunction,
    pub(crate) rms_norm_batched: CudaFunction,
    pub(crate) rms_norm_quant_nvfp4: CudaFunction,
    pub(crate) rms_norm_quant_nvfp4_batched: CudaFunction,
    pub(crate) add: CudaFunction,
    pub(crate) add_inplace: CudaFunction,
    pub(crate) swiglu: CudaFunction,
    pub(crate) swiglu_inplace_gate: CudaFunction,
    pub(crate) geglu_tanh: CudaFunction,
    pub(crate) geglu_tanh_strided: CudaFunction,
    pub(crate) rms_norm_batched_no_weight: CudaFunction,
    pub(crate) rope: CudaFunction,
    pub(crate) rope_ptr: CudaFunction,
    pub(crate) rope_batched: CudaFunction,
    pub(crate) rope_positions_batched: CudaFunction,
    pub(crate) rope_positions_batched_f16_out: CudaFunction,
    pub(crate) build_dense_prefill_metadata: CudaFunction,
    pub(crate) f32_to_f16: CudaFunction,
    pub(crate) f32_to_bf16: CudaFunction,
    pub(crate) bf16_to_f32: CudaFunction,
    pub(crate) router_softmax_topk: CudaFunction,
    /// Stage I.2 vision row-softmax (bidirectional attention's softmax pass).
    pub(crate) vision_row_softmax: CudaFunction,
    /// BF16 in-place row-softmax — skips the BF16↔F32 round-trip on the
    /// vision tower's BF16 attention path.
    pub(crate) vision_row_softmax_bf16: CudaFunction,
    /// Single-tensor in-place gelu_pytorch_tanh — Gemma-4 E4B PLE gate.
    pub(crate) gelu_tanh_inplace_f32: CudaFunction,
    /// Batched per-token-strided multiply for the prefill PLE additive.
    pub(crate) ple_per_layer_mul_inplace_f32: CudaFunction,
    /// Stage I.3 fused bidirectional vision attention (QK·softmax·PV in one launch).
    pub(crate) vision_bidi_attn: CudaFunction,
    /// Stage I.4 GPU-only vision forward kernels.
    pub(crate) vision_pixel_rescale: CudaFunction,
    pub(crate) vision_pos_embed_add: CudaFunction,
    pub(crate) vision_head_rmsnorm: CudaFunction,
    pub(crate) vision_rope_2d: CudaFunction,
    pub(crate) vision_standardize: CudaFunction,
    pub(crate) vision_pool3x3_scale: CudaFunction,
    /// Gemma-4 audio tower (USM/Conformer) per-token kernels.
    pub(crate) audio_glu_halfsplit: CudaFunction,
    pub(crate) audio_depthwise_causal_conv1d: CudaFunction,
    pub(crate) audio_per_dim_scale: CudaFunction,
    pub(crate) audio_clamp_inplace: CudaFunction,
    pub(crate) audio_silu_inplace: CudaFunction,
    pub(crate) audio_add_bias_rows: CudaFunction,
    pub(crate) router_softmax_topk_packed: CudaFunction,
    pub(crate) router_zero_expert_counts: CudaFunction,
    pub(crate) router_bucket_sort: CudaFunction,
    pub(crate) router_expert_offsets: CudaFunction,
    pub(crate) permute_gather_f32: CudaFunction,
    pub(crate) unpermute_scatter_add_f32: CudaFunction,
    pub(crate) router_build_unpermute_index: CudaFunction,
    pub(crate) unpermute_scatter_serial_f32: CudaFunction,
    pub(crate) kv_store: CudaFunction,
    pub(crate) kv_store_ptr: CudaFunction,
    pub(crate) kv_store_batched: CudaFunction,
    pub(crate) kv_store_slots_batched: CudaFunction,
    pub(crate) rope_kv_store_slots_batched: CudaFunction,
    pub(crate) kv_store_fp8: CudaFunction,
    pub(crate) kv_store_fp8_ptr: CudaFunction,
    pub(crate) kv_store_fp8_batched: CudaFunction,
    pub(crate) kv_store_fp8_slots_batched: CudaFunction,
    pub(crate) rope_kv_store_fp8_slots_batched: CudaFunction,
    pub(crate) attention_decode_fp8: CudaFunction,
    pub(crate) attention_decode_ptr_fp8: CudaFunction,
    pub(crate) attention_decode_ptr_split_fp8: CudaFunction,
    pub(crate) attention_decode_streaming_fp8: CudaFunction,
    pub(crate) attention: CudaFunction,
    pub(crate) attention_ptr: CudaFunction,
    pub(crate) attention_decode_ptr_split: CudaFunction,
    pub(crate) attention_decode_ptr_split_hdpart: CudaFunction,
    pub(crate) attention_decode_ptr_split_hdpart_fp8: CudaFunction,
    pub(crate) attention_decode_ptr_combine: CudaFunction,
    pub(crate) attention_decode_streaming: CudaFunction,
    pub(crate) attention_prefill_batched: CudaFunction,
    pub(crate) attention_prefill_continuation: CudaFunction,
    pub(crate) attention_prefill_batched_warp: CudaFunction,
    pub(crate) attention_prefill_paged_varlen: CudaFunction,
    pub(crate) attention_prefill_paged_varlen_halfq: CudaFunction,
    pub(crate) attention_prefill_paged_varlen_halfq_block4: CudaFunction,
    pub(crate) attention_prefill_paged_varlen_halfq_wmma_hdim128_gqa4: CudaFunction,
    pub(crate) attention_prefill_dense_halfq_block4: CudaFunction,
    pub(crate) attention_prefill_dense_halfq_warp_tile_hdim128: CudaFunction,
    pub(crate) attention_prefill_dense_halfq_wmma_hdim128: CudaFunction,
    pub(crate) attention_prefill_dense_halfq_wmma_hdim256: CudaFunction,
    pub(crate) attention_prefill_dense_halfq_wmma_hdim512: CudaFunction,
    pub(crate) attention_prefill_dense_halfq_wmma_hdim512_regacc: CudaFunction,
    // Q_BLOCK=32 twin of `..._hdim512_regacc`. Halves K/V HBM bandwidth per
    // output token at long context. Opt-in via `AEGIS_HDIM512_Q32_ENABLE=1`.
    pub(crate) attention_prefill_dense_halfq_wmma_hdim512_q32_regacc: CudaFunction,
    // cp.async K-only pipelined twin of `..._hdim512_q32_regacc`. Double-
    // buffers the K tile (32 KiB extra shmem); V stays synchronous. Opt-in
    // via `AEGIS_HDIM512_Q32_PIPELINE_ENABLE=1`.
    pub(crate) attention_prefill_dense_halfq_wmma_hdim512_q32_regacc_pipeline: CudaFunction,
    // ===== Round 3 attention pipeline (cp.async K/V double-buffer) =====
    // Numerical-twin of `..._hdim512_regacc`; opt-in via env var.
    pub(crate) attention_prefill_dense_halfq_wmma_hdim512_regacc_pipeline: CudaFunction,
    // ===================================================================
    // FlashAttention-2 style hdim=512 prefill kernel. kv_block=64 (4x the old
    // k_tile=16), register-resident O accumulator, hdim-slab streamed K/V with
    // cp.async double-buffering. Opt-in via `AEGIS_ATTN_FA2=1`.
    pub(crate) attention_prefill_dense_fa2_hdim512: CudaFunction,
    pub(crate) attention_prefill_dense_fa2_hdim512_q64: CudaFunction,
    // ===================================================================
    // Stage H.4 mma4: register-softmax 8-warp/32-KV hd=512 prefill kernel.
    // Auto-default for ctx ∈ [16k, 64k] (FA-2 takes ctx > 64k). Force on/off
    // with AEGIS_MMA4=1 / AEGIS_MMA4=0.
    pub(crate) attention_prefill_dense_mma4_hdim512: CudaFunction,
    // FP8-E4M3 KV-cache variant of the FA-2 hdim=512 q32 kernel. Reads the
    // persistent e4m3 cache directly (half the KV HBM traffic), dequants
    // e4m3->half in shared, runs the identical BF16 WMMA math. Opt-in via
    // `AEGIS_ATTN_FP8=1` + KV cache quant=Fp8 + head_dim=512.
    pub(crate) attention_prefill_dense_fa2_hdim512_fp8: CudaFunction,
    // Native FP8 e4m3 MMA variant of the FA-2 hdim=512 kernel. Keeps K/V e4m3
    // in shared memory and feeds the bytes straight into the SM120
    // `kind::f8f6f4.m16n8k32` tensor-core MMA (no half-slab dequant). The
    // halved smem footprint fits 2 thread-blocks per SM. Opt-in via
    // `AEGIS_ATTN_FP8=1` + KV cache quant=Fp8 + head_dim=512.
    pub(crate) attention_prefill_dense_fa2_hdim512_fp8_mma: CudaFunction,
    pub(crate) attention_prefill_dense_halfq_wmma_hdim128_fa: CudaFunction,
    pub(crate) attention_prefill_dense_halfq_wmma_hdim128_gqa4: CudaFunction,
    pub(crate) attention_prefill_dense_halfq_wmma_hdim128_gqa4_split: CudaFunction,
    pub(crate) attention_prefill_dense_halfq_wmma_hdim128_cluster2: CudaFunction,
    pub(crate) attention_prefill_dense_halfq_wmma_hdim128_q32: CudaFunction,
    pub(crate) attention_prefill_dense_halfq_wmma_hdim128_split: CudaFunction,
    pub(crate) attention_prefill_dense_halfq_wmma_hdim128_combine: CudaFunction,
    pub(crate) attention_prefill_paged_varlen_fa4_hdim128: CudaFunction,
    pub(crate) attention_prefill_paged_varlen_halfq_block4_split: CudaFunction,
    pub(crate) attention_prefill_paged_varlen_halfq_block4_combine: CudaFunction,
    pub(crate) attention_prefill_paged_varlen_warp: CudaFunction,
    pub(crate) copy_row_f32: CudaFunction,
    pub(crate) argmax_blocks: CudaFunction,
    pub(crate) argmax_finalize: CudaFunction,
    /// Speculative decoding: sparse lm_head matvec over an explicit candidate-row
    /// list (centroid-masked draft head). Registered alongside the sampling kernels.
    pub(crate) spec_sparse_lm_head_matvec: CudaFunction,
    pub(crate) axpy_f32: CudaFunction,
    // ── GPU-driven MoE decode (device-mapped-host expert gather) ──
    pub(crate) moe_gather_experts: CudaFunction,
    pub(crate) nvfp4_quantize_input_dptr: CudaFunction,
    pub(crate) nvfp4_prequant_dptr: CudaFunction,
    pub(crate) axpy_f32_topk_weight: CudaFunction,
    // Batched (grouped-over-experts) decode MoE — slot on grid.y.
    // Fast M=1 NVFP4 GEMV (warp-per-row, no shared-mem reduction).
    pub(crate) nvfp4_gemv_warp: CudaFunction,
    pub(crate) nvfp4_quantize_input_batched_dptr: CudaFunction,
    pub(crate) nvfp4_prequant_batched_dptr: CudaFunction,
    pub(crate) nvfp4_prequant_batched_dptr_warp: CudaFunction,
    pub(crate) moe_geglu_tanh_batched_slots: CudaFunction,
    pub(crate) moe_weighted_accumulate: CudaFunction,
    pub(crate) zero_f32: CudaFunction,
    pub(crate) scale_f32: CudaFunction,
    pub(crate) mul_vec_inplace_f32: CudaFunction,
    pub(crate) gather_rows_f32: CudaFunction,
    pub(crate) scatter_add_weighted_f32: CudaFunction,
    pub(crate) bf16_matmul_reference_batched: CudaFunction,
    pub(crate) gated_deltanet_decode: CudaFunction,
    pub(crate) mamba_scan_decode: CudaFunction,
}

impl CudaKernelFunctions {
    pub(crate) fn load(context: &Arc<CudaContext>, device_index: usize) -> Result<Self> {
        let ptx = compile_ptx_with_opts(
            BLACKWELL_FP4_KERNEL_SRC,
            CompileOptions {
                arch: Some(nvrtc_arch_for_device(device_index)),
                name: Some("aegis_blackwell_nvfp4.cu".into()),
                include_paths: cuda_include_paths(),
                ..Default::default()
            },
        )
        .map_err(|error| {
            AegisError::Unsupported(format!(
                "compile blackwell fp4 cuda kernels failed: {error}"
            ))
        })?;
        let module = context
            .load_module(Ptx::from_src(ptx.to_src()))
            .map_err(map_cuda_err("load blackwell fp4 module"))?;

        Ok(Self {
            blackwell_fp4: load(&module, "aegis_blackwell_nvfp4_linear_probe")?,
            mxfp4_matvec: load(&module, "aegis_mxfp4_matvec_native")?,
            mxfp4_matvec_4warp: load(&module, "aegis_mxfp4_matvec_native_4warp")?,
            mxfp4_matvec_16warp: load(&module, "aegis_mxfp4_matvec_native_16warp")?,
            mxfp4_matmul_n8: load(&module, "aegis_mxfp4_matmul_native_n8")?,
            mxfp4_matmul_tile_m16n32: load(&module, "aegis_mxfp4_matmul_native_tile_m16n32")?,
            mxfp4_matmul_tile_m16n64: load(&module, "aegis_mxfp4_matmul_native_tile_m16n64")?,
            mxfp4_matmul_qkv_tile_m16n32: load(
                &module,
                "aegis_mxfp4_matmul_qkv_native_tile_m16n32",
            )?,
            mxfp4_matmul_qkv_tile_m16n64: load(
                &module,
                "aegis_mxfp4_matmul_qkv_native_tile_m16n64",
            )?,
            mxfp4_matmul_gate_up_tile_m16n32: load(
                &module,
                "aegis_mxfp4_matmul_gate_up_native_tile_m16n32",
            )?,
            mxfp4_matmul_gate_up_tile_m16n64: load(
                &module,
                "aegis_mxfp4_matmul_gate_up_native_tile_m16n64",
            )?,
            split_qkv_scaled: load(&module, "aegis_split_qkv_scaled")?,
            mxfp4_quantize_input: load(&module, "aegis_mxfp4_quantize_vector")?,
            swiglu_mxfp4_quantize_batched: load(&module, "aegis_swiglu_mxfp4_quantize_batched")?,
            fp8_quantize_bf16_per_row: load(&module, "aegis_quantize_bf16_to_fp8_per_row")?,
            fp8_matvec: load(&module, "aegis_fp8_matvec")?,
            fp8_matmul_batched: load(&module, "aegis_fp8_matmul_batched")?,
            fp8_dequant_to_bf16: load(&module, "aegis_dequant_fp8_to_bf16")?,
            nvfp4_reference: load(&module, "aegis_nvfp4_linear_reference")?,
            nvfp4_reference_batched: load(&module, "aegis_nvfp4_linear_reference_batched")?,
            nvfp4_prequant: load(&module, "aegis_nvfp4_linear_prequantized")?,
            nvfp4_prequant_batched: load(&module, "aegis_nvfp4_linear_prequantized_batched")?,
            nvfp4_prequant_batched_gemm: load(
                &module,
                "aegis_nvfp4_linear_prequantized_batched_gemm",
            )?,
            nvfp4_prequant_batched_gemm_wmma_bf16: load(
                &module,
                "aegis_nvfp4_linear_prequantized_batched_gemm_wmma_bf16",
            )?,
            nvfp4_grouped_prequant_gemm_wmma_bf16: load(
                &module,
                "aegis_nvfp4_grouped_prequant_gemm_wmma_bf16",
            )?,
            nvfp4_grouped_prequant_gemm_wmma_bf16_t32: load(
                &module,
                "aegis_nvfp4_grouped_prequant_gemm_wmma_bf16_t32",
            )?,
            // ── MoE/NVFP4 fused entries (Phase B.4 Round 2). ──
            nvfp4_grouped_prequant_gemm_wmma_bf16_t32_dual: load(
                &module,
                "aegis_nvfp4_grouped_prequant_gemm_wmma_bf16_t32_dual",
            )?,
            // ── MoE/NVFP4 cp.async pipelined entry (Phase B.4 Round 3). ──
            nvfp4_grouped_prequant_gemm_wmma_bf16_t32_pipeline: load(
                &module,
                "aegis_nvfp4_grouped_prequant_gemm_wmma_bf16_t32_pipeline",
            )?,
            // ── MoE/NVFP4 64×64 output-tile entry (Phase B.4 Round 4). ──
            nvfp4_grouped_prequant_gemm_wmma_bf16_t32_big: load(
                &module,
                "aegis_nvfp4_grouped_prequant_gemm_wmma_bf16_t32_big",
            )?,
            // ── MoE/NVFP4 64×64 output-tile + cp.async B-pipeline (Phase B.4 Round 5). ──
            nvfp4_grouped_prequant_gemm_wmma_bf16_t32_big_pipeline: load(
                &module,
                "aegis_nvfp4_grouped_prequant_gemm_wmma_bf16_t32_big_pipeline",
            )?,
            nvfp4_quantize_input: load(&module, "aegis_nvfp4_quantize_input")?,
            nvfp4_quantize_input_batched: load(&module, "aegis_nvfp4_quantize_input_batched")?,
            bf16_matvec: load(&module, "aegis_bf16_matvec_reference")?,
            bf16_row: load(&module, "aegis_bf16_row_to_f32")?,
            bf16_rows: load(&module, "aegis_bf16_rows_to_f32")?,
            rms_norm: load(&module, "aegis_rms_norm")?,
            rms_norm_batched: load(&module, "aegis_rms_norm_batched")?,
            rms_norm_quant_nvfp4: load(&module, "aegis_rms_norm_quant_nvfp4")?,
            rms_norm_quant_nvfp4_batched: load(&module, "aegis_rms_norm_quant_nvfp4_batched")?,
            add: load(&module, "aegis_vector_add")?,
            add_inplace: load(&module, "aegis_vector_add_inplace")?,
            swiglu: load(&module, "aegis_swiglu")?,
            swiglu_inplace_gate: load(&module, "aegis_swiglu_inplace_gate")?,
            geglu_tanh: load(&module, "aegis_geglu_tanh")?,
            geglu_tanh_strided: load(&module, "aegis_geglu_tanh_strided")?,
            rms_norm_batched_no_weight: load(&module, "aegis_rms_norm_batched_no_weight")?,
            rope: load(&module, "aegis_apply_rope")?,
            rope_ptr: load(&module, "aegis_apply_rope_ptr")?,
            rope_batched: load(&module, "aegis_apply_rope_batched")?,
            rope_positions_batched: load(&module, "aegis_apply_rope_positions_batched")?,
            rope_positions_batched_f16_out: load(
                &module,
                "aegis_apply_rope_positions_batched_f16_out",
            )?,
            build_dense_prefill_metadata: load(&module, "aegis_build_dense_prefill_metadata")?,
            f32_to_f16: load(&module, "aegis_f32_to_f16")?,
            f32_to_bf16: load(&module, "aegis_f32_to_bf16")?,
            bf16_to_f32: load(&module, "aegis_bf16_to_f32")?,
            router_softmax_topk: load(&module, "aegis_router_softmax_topk")?,
            vision_row_softmax: load(&module, "aegis_vision_row_softmax")?,
            vision_row_softmax_bf16: load(&module, "aegis_vision_row_softmax_bf16")?,
            gelu_tanh_inplace_f32: load(&module, "aegis_gelu_tanh_inplace_f32")?,
            ple_per_layer_mul_inplace_f32: load(&module, "aegis_ple_per_layer_mul_inplace_f32")?,
            vision_bidi_attn: {
                // Needs dynamic shared scaled by max n_tok in flight. We use
                // 96 KiB cap which fits scores[n_tok≤2376] + 8 warpred + Q[hd].
                let f = load(&module, "aegis_vision_bidi_attn")?;
                f.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum
                        ::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    96 * 1024,
                )
                .map_err(|e| AegisError::Unsupported(format!(
                    "set max dynamic shared mem on vision_bidi_attn: {e:?}"
                )))?;
                f
            },
            vision_pixel_rescale: load(&module, "aegis_vision_pixel_rescale")?,
            vision_pos_embed_add: load(&module, "aegis_vision_pos_embed_add")?,
            vision_head_rmsnorm: load(&module, "aegis_vision_head_rmsnorm")?,
            vision_rope_2d: load(&module, "aegis_vision_rope_2d")?,
            vision_standardize: load(&module, "aegis_vision_standardize")?,
            vision_pool3x3_scale: load(&module, "aegis_vision_pool3x3_scale")?,
            audio_glu_halfsplit: load(&module, "aegis_audio_glu_halfsplit")?,
            audio_depthwise_causal_conv1d: load(
                &module,
                "aegis_audio_depthwise_causal_conv1d",
            )?,
            audio_per_dim_scale: load(&module, "aegis_audio_per_dim_scale")?,
            audio_clamp_inplace: load(&module, "aegis_audio_clamp_inplace")?,
            audio_silu_inplace: load(&module, "aegis_audio_silu_inplace")?,
            audio_add_bias_rows: load(&module, "aegis_audio_add_bias_rows")?,
            router_softmax_topk_packed: load(&module, "aegis_router_softmax_topk_packed")?,
            router_zero_expert_counts: load(&module, "aegis_router_zero_expert_counts")?,
            router_bucket_sort: load(&module, "aegis_router_bucket_sort")?,
            router_expert_offsets: load(&module, "aegis_router_expert_offsets")?,
            permute_gather_f32: load(&module, "aegis_permute_gather_f32")?,
            unpermute_scatter_add_f32: load(&module, "aegis_unpermute_scatter_add_f32")?,
            router_build_unpermute_index: load(&module, "aegis_router_build_unpermute_index")?,
            unpermute_scatter_serial_f32: load(&module, "aegis_unpermute_scatter_serial_f32")?,
            kv_store: load(&module, "aegis_kv_store")?,
            kv_store_ptr: load(&module, "aegis_kv_store_ptr")?,
            kv_store_batched: load(&module, "aegis_kv_store_batched")?,
            kv_store_slots_batched: load(&module, "aegis_kv_store_slots_batched")?,
            rope_kv_store_slots_batched: load(&module, "aegis_rope_kv_store_slots_batched")?,
            kv_store_fp8: load(&module, "aegis_kv_store_fp8")?,
            kv_store_fp8_ptr: load(&module, "aegis_kv_store_fp8_ptr")?,
            kv_store_fp8_batched: load(&module, "aegis_kv_store_fp8_batched")?,
            kv_store_fp8_slots_batched: load(&module, "aegis_kv_store_fp8_slots_batched")?,
            rope_kv_store_fp8_slots_batched: load(&module, "aegis_rope_kv_store_fp8_slots_batched")?,
            attention_decode_fp8: load(&module, "aegis_attention_decode_fp8")?,
            attention_decode_ptr_fp8: load(&module, "aegis_attention_decode_ptr_fp8")?,
            attention_decode_ptr_split_fp8: {
                // Same long-context dynamic-shared opt-in as the F16
                // attention_decode_ptr_split path — see comment there.
                let f = load(&module, "aegis_attention_decode_ptr_split_fp8")?;
                f.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum
                        ::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    96 * 1024,
                )
                .map_err(|e| AegisError::Unsupported(format!(
                    "set max dynamic shared mem on attention_decode_ptr_split_fp8: {e:?}"
                )))?;
                f
            },
            attention_decode_streaming_fp8: load(&module, "aegis_attention_decode_streaming_fp8")?,
            attention: load(&module, "aegis_attention_decode")?,
            attention_ptr: load(&module, "aegis_attention_decode_ptr")?,
            attention_decode_ptr_split: {
                // Opt the split-decode kernel into the 96 KiB dynamic shared
                // pool so it can size `scores[chunk_len]` from the actual
                // seq_len at long contexts. The captured-graph hot path
                // (seq_len ≤ CUDA_GRAPH_ATTN_MAX_SEQ_LEN) still allocates
                // only `DECODE_MAX_CHUNK_LEN`; the larger pool is used only
                // for the eager long-context path. See decode.rs.
                let f = load(&module, "aegis_attention_decode_ptr_split")?;
                f.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum
                        ::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    96 * 1024,
                )
                .map_err(|e| AegisError::Unsupported(format!(
                    "set max dynamic shared mem on attention_decode_ptr_split: {e:?}"
                )))?;
                f
            },
            // Stage G head-dim-partitioned single-pass decode kernels. Tiny
            // shared (KQ[128]+scratch, <1 KiB) so no opt-in cap needed.
            attention_decode_ptr_split_hdpart:
                load(&module, "aegis_attention_decode_ptr_split_hdpart")?,
            attention_decode_ptr_split_hdpart_fp8:
                load(&module, "aegis_attention_decode_ptr_split_hdpart_fp8")?,
            attention_decode_ptr_combine: load(&module, "aegis_attention_decode_ptr_combine")?,
            attention_decode_streaming: load(&module, "aegis_attention_decode_streaming")?,
            attention_prefill_batched: load(&module, "aegis_attention_prefill_batched")?,
            attention_prefill_continuation: load(&module, "aegis_attention_prefill_continuation")?,
            attention_prefill_batched_warp: load(&module, "aegis_attention_prefill_batched_warp")?,
            attention_prefill_paged_varlen: load(&module, "aegis_attention_prefill_paged_varlen")?,
            attention_prefill_paged_varlen_halfq: load(
                &module,
                "aegis_attention_prefill_paged_varlen_halfq",
            )?,
            attention_prefill_paged_varlen_halfq_block4: load(
                &module,
                "aegis_attention_prefill_paged_varlen_halfq_block4",
            )?,
            attention_prefill_paged_varlen_halfq_wmma_hdim128_gqa4: load(
                &module,
                "aegis_attention_prefill_paged_varlen_halfq_wmma_hdim128_gqa4",
            )?,
            attention_prefill_dense_halfq_block4: load(
                &module,
                "aegis_attention_prefill_dense_halfq_block4",
            )?,
            attention_prefill_dense_halfq_warp_tile_hdim128: load(
                &module,
                "aegis_attention_prefill_dense_halfq_warp_tile_hdim128",
            )?,
            attention_prefill_dense_halfq_wmma_hdim128: load(
                &module,
                "aegis_attention_prefill_dense_halfq_wmma_hdim128",
            )?,
            attention_prefill_dense_halfq_wmma_hdim256: {
                let f = load(&module, "aegis_attention_prefill_dense_halfq_wmma_hdim256")?;
                // hdim=256 needs ~75 KiB dynamic shared memory; default cap
                // is 48 KiB. Opt into the larger pool (Blackwell supports up
                // to 228 KiB per SM). Set once at load time; persists for the
                // lifetime of the function.
                f.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum
                        ::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    96 * 1024,
                )
                .map_err(|e| AegisError::Unsupported(format!(
                    "set max dynamic shared mem on hdim256 kernel: {e:?}"
                )))?;
                f
            },
            attention_prefill_dense_halfq_wmma_hdim512: {
                let f = load(&module, "aegis_attention_prefill_dense_halfq_wmma_hdim512")?;
                // hdim=512 needs ~83 KiB dynamic shared memory after
                // dropping the tile_acc double-buffer (see the kernel's
                // comment). 96 KiB is comfortably above that and sits
                // within the sm_120 100 KiB per-block cap.
                f.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum
                        ::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    96 * 1024,
                )
                .map_err(|e| AegisError::Unsupported(format!(
                    "set max dynamic shared mem on hdim512 kernel: {e:?}"
                )))?;
                f
            },
            attention_prefill_dense_halfq_wmma_hdim512_regacc: {
                let f = load(&module, "aegis_attention_prefill_dense_halfq_wmma_hdim512_regacc")?;
                // Register-resident-acc variant. Shared memory drops
                // to ~50 KiB (no acc buffer); register pressure rises
                // because two persistent f32 c_frags per warp now live
                // in registers. 64 KiB cap is plenty.
                f.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum
                        ::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    64 * 1024,
                )
                .map_err(|e| AegisError::Unsupported(format!(
                    "set max dynamic shared mem on hdim512_regacc kernel: {e:?}"
                )))?;
                f
            },
            attention_prefill_dense_halfq_wmma_hdim512_q32_regacc: {
                let f = load(
                    &module,
                    "aegis_attention_prefill_dense_halfq_wmma_hdim512_q32_regacc",
                )?;
                // Q_BLOCK=32 twin. Shared mem ~67.5 KiB (q_shared 32 KiB +
                // k_shared 16 KiB + v_shared 16 KiB + scores 2 KiB +
                // weights_half 1 KiB + scalars 0.4 KiB). Use sm_120's 96
                // KiB opt-in dynamic-shared cap. Register pressure higher
                // than the q_block=16 twin: 4 persistent f32 c_frags per
                // warp (vs 2). Expect 1 block/SM residency.
                f.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum
                        ::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    96 * 1024,
                )
                .map_err(|e| AegisError::Unsupported(format!(
                    "set max dynamic shared mem on hdim512_q32_regacc kernel: {e:?}"
                )))?;
                f
            },
            attention_prefill_dense_halfq_wmma_hdim512_q32_regacc_pipeline: {
                let f = load(
                    &module,
                    "aegis_attention_prefill_dense_halfq_wmma_hdim512_q32_regacc_pipeline",
                )?;
                // cp.async K-only pipelined Q_BLOCK=32 twin. q_shared 32 KiB
                // + k_shared[2] 32 KiB + v_shared 16 KiB + scores 2 KiB +
                // weights_half 1 KiB + scalars 0.4 KiB = ~83.4 KiB, within
                // sm_120's 96 KiB opt-in dynamic-shared cap. Same 1 block/SM
                // residency as the synchronous q32 twin (no occupancy loss).
                f.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum
                        ::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    96 * 1024,
                )
                .map_err(|e| AegisError::Unsupported(format!(
                    "set max dynamic shared mem on hdim512_q32_regacc_pipeline kernel: {e:?}"
                )))?;
                f
            },
            // ===== Round 3 attention pipeline (cp.async K-only double-buffer) =====
            attention_prefill_dense_halfq_wmma_hdim512_regacc_pipeline: {
                let f = load(
                    &module,
                    "aegis_attention_prefill_dense_halfq_wmma_hdim512_regacc_pipeline",
                )?;
                // cp.async-pipelined K-only twin: doubles K tile shared-mem
                // (16 KiB extra) and adds a dedicated 16 KiB acc_scratch
                // (since k_shared no longer overlays acc). V stays single-
                // buffered and synchronous. Total ~82 KiB, within sm_120's
                // 96 KiB opt-in dynamic-shared cap. Pipelining V too would
                // push us back over the cap (see kernel comment). The 96 KiB
                // cap is the empirically-safe ceiling on consumer Blackwell.
                f.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum
                        ::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    96 * 1024,
                )
                .map_err(|e| AegisError::Unsupported(format!(
                    "set max dynamic shared mem on hdim512_regacc_pipeline kernel: {e:?}"
                )))?;
                f
            },
            // ===================================================================
            attention_prefill_dense_fa2_hdim512: {
                let f = load(
                    &module,
                    "aegis_attention_prefill_dense_fa2_hdim512",
                )?;
                // FA-2 hdim=512 kernel. Shared mem ~76.5 KiB (q_shared 32 KiB
                // + kv_slab[2] 32 KiB + s_shared 8 KiB + weights_h 4 KiB +
                // scalars 0.4 KiB). Use sm_120's 96 KiB opt-in dynamic-shared
                // cap. 1 block/SM expected (16 persistent o_frags, kv_block=64).
                f.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum
                        ::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    96 * 1024,
                )
                .map_err(|e| AegisError::Unsupported(format!(
                    "set max dynamic shared mem on fa2_hdim512 kernel: {e:?}"
                )))?;
                f
            },
            // ===================================================================
            attention_prefill_dense_fa2_hdim512_q64: {
                let f = load(
                    &module,
                    "aegis_attention_prefill_dense_fa2_hdim512_q64",
                )?;
                // FA-2 hdim=512 q_block=64 variant (Lever A: 2x arithmetic
                // intensity, halved KV HBM re-reads). Shared mem ~92.75 KiB
                // (q_shared 64 KiB + kv_slab[2] 16 KiB + s_shared 8 KiB +
                // weights_h 4 KiB + scalars 0.75 KiB). 96 KiB opt-in cap.
                f.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum
                        ::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    96 * 1024,
                )
                .map_err(|e| AegisError::Unsupported(format!(
                    "set max dynamic shared mem on fa2_hdim512_q64 kernel: {e:?}"
                )))?;
                f
            },
            // Stage H.4 mma4: register-softmax 8-warp/32-KV hd=512 prefill.
            // ~43 KiB shared (no s_shared S spill; tiny xwarp_max/sum buffers).
            // Uses sm_120's 96 KiB opt-in dynamic-shared cap.
            attention_prefill_dense_mma4_hdim512: {
                let f = load(&module, "aegis_attention_prefill_dense_mma4_hdim512")?;
                f.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum
                        ::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    96 * 1024,
                )
                .map_err(|e| AegisError::Unsupported(format!(
                    "set max dynamic shared mem on mma4_hdim512 kernel: {e:?}"
                )))?;
                f
            },
            // ===================================================================
            attention_prefill_dense_fa2_hdim512_fp8: {
                let f = load(
                    &module,
                    "aegis_attention_prefill_dense_fa2_hdim512_fp8",
                )?;
                // FP8-E4M3 FA-2 hdim=512 kernel. Shared mem ~76.4 KiB
                // (q_shared 32 KiB + e4m3_stage[2] 16 KiB + half_slab 16 KiB
                // + s_shared 8 KiB + weights_h 4 KiB + scalars 0.4 KiB).
                // Use sm_120's 96 KiB opt-in dynamic-shared cap.
                f.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum
                        ::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    96 * 1024,
                )
                .map_err(|e| AegisError::Unsupported(format!(
                    "set max dynamic shared mem on fa2_hdim512_fp8 kernel: {e:?}"
                )))?;
                f
            },
            // ===================================================================
            attention_prefill_dense_fa2_hdim512_fp8_mma: {
                let f = load(
                    &module,
                    "aegis_attention_prefill_dense_fa2_hdim512_fp8_mma",
                )?;
                // Native FP8 e4m3 MMA FA-2 hdim=512 kernel. Shared mem ~42.5 KiB
                // (q_e4m3 16 + kv_e4m3[2] 16 + s_shared 8 + p_e4m3 2 + scalars
                // 0.4 + q_scale 0.1). Capped at 48 KiB: that is < 100 KiB / 2,
                // so the driver can co-resident 2 thread-blocks per SM — the
                // occupancy target of this kernel. (__launch_bounds__(512, 2)
                // also requests the 2-block hint at compile time.)
                f.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute_enum
                        ::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    48 * 1024,
                )
                .map_err(|e| AegisError::Unsupported(format!(
                    "set max dynamic shared mem on fa2_hdim512_fp8_mma kernel: {e:?}"
                )))?;
                f
            },
            attention_prefill_dense_halfq_wmma_hdim128_fa: load(
                &module,
                "aegis_attention_prefill_dense_halfq_wmma_hdim128_fa",
            )?,
            attention_prefill_dense_halfq_wmma_hdim128_gqa4: load(
                &module,
                "aegis_attention_prefill_dense_halfq_wmma_hdim128_gqa4",
            )?,
            attention_prefill_dense_halfq_wmma_hdim128_gqa4_split: load(
                &module,
                "aegis_attention_prefill_dense_halfq_wmma_hdim128_gqa4_split",
            )?,
            attention_prefill_dense_halfq_wmma_hdim128_cluster2: load(
                &module,
                "aegis_attention_prefill_dense_halfq_wmma_hdim128_cluster2",
            )?,
            attention_prefill_dense_halfq_wmma_hdim128_q32: load(
                &module,
                "aegis_attention_prefill_dense_halfq_wmma_hdim128_q32",
            )?,
            attention_prefill_dense_halfq_wmma_hdim128_split: load(
                &module,
                "aegis_attention_prefill_dense_halfq_wmma_hdim128_split",
            )?,
            attention_prefill_dense_halfq_wmma_hdim128_combine: load(
                &module,
                "aegis_attention_prefill_dense_halfq_wmma_hdim128_combine",
            )?,
            attention_prefill_paged_varlen_fa4_hdim128: load(
                &module,
                "aegis_attention_prefill_paged_varlen_fa4_hdim128",
            )?,
            attention_prefill_paged_varlen_halfq_block4_split: load(
                &module,
                "aegis_attention_prefill_paged_varlen_halfq_block4_split",
            )?,
            attention_prefill_paged_varlen_halfq_block4_combine: load(
                &module,
                "aegis_attention_prefill_paged_varlen_halfq_block4_combine",
            )?,
            attention_prefill_paged_varlen_warp: load(
                &module,
                "aegis_attention_prefill_paged_varlen_warp",
            )?,
            copy_row_f32: load(&module, "aegis_copy_row_f32")?,
            argmax_blocks: load(&module, "aegis_argmax_f32_blocks")?,
            argmax_finalize: load(&module, "aegis_argmax_f32_finalize")?,
            spec_sparse_lm_head_matvec: load(&module, "aegis_spec_sparse_lm_head_matvec")?,
            axpy_f32: load(&module, "aegis_axpy_f32")?,
            moe_gather_experts: load(&module, "aegis_moe_gather_experts")?,
            nvfp4_quantize_input_dptr: load(&module, "aegis_nvfp4_quantize_input_dptr")?,
            nvfp4_prequant_dptr: load(&module, "aegis_nvfp4_linear_prequantized_dptr")?,
            axpy_f32_topk_weight: load(&module, "aegis_axpy_f32_topk_weight")?,
            nvfp4_gemv_warp: load(&module, "aegis_nvfp4_gemv_warp")?,
            nvfp4_quantize_input_batched_dptr: load(&module, "aegis_nvfp4_quantize_input_batched_dptr")?,
            nvfp4_prequant_batched_dptr: load(&module, "aegis_nvfp4_linear_prequantized_batched_dptr")?,
            nvfp4_prequant_batched_dptr_warp: load(&module, "aegis_nvfp4_linear_prequantized_batched_dptr_warp")?,
            moe_geglu_tanh_batched_slots: load(&module, "aegis_moe_geglu_tanh_batched_slots")?,
            moe_weighted_accumulate: load(&module, "aegis_moe_weighted_accumulate")?,
            zero_f32: load(&module, "aegis_zero_f32")?,
            scale_f32: load(&module, "aegis_scale_f32")?,
            mul_vec_inplace_f32: load(&module, "aegis_mul_vec_inplace_f32")?,
            gather_rows_f32: load(&module, "aegis_gather_rows_f32")?,
            scatter_add_weighted_f32: load(&module, "aegis_scatter_add_weighted_f32")?,
            bf16_matmul_reference_batched: load(&module, "aegis_bf16_matmul_reference_batched")?,
            gated_deltanet_decode: load(&module, "aegis_gated_deltanet_decode")?,
            mamba_scan_decode: load(&module, "aegis_mamba_scan_decode")?,
            _module: module,
        })
    }
}

fn cuda_include_paths() -> Vec<String> {
    let mut candidates = Vec::new();
    for var in ["CUDA_PATH", "CUDA_HOME"] {
        if let Ok(root) = env::var(var) {
            candidates.push(format!("{root}/include"));
            candidates.push(format!("{root}/targets/x86_64-linux/include"));
            candidates.push(format!("{root}/targets/x86_64-linux/include/cccl"));
        }
    }
    candidates.extend(
        [
            "/opt/cuda/targets/x86_64-linux/include",
            "/opt/cuda/targets/x86_64-linux/include/cccl",
            "/usr/local/cuda/include",
            "/usr/local/cuda/targets/x86_64-linux/include/cccl",
            "/usr/include",
        ]
        .into_iter()
        .map(str::to_owned),
    );
    candidates
        .into_iter()
        .filter(|path| {
            let path = Path::new(path);
            path.join("cuda_fp16.h").exists() || path.join("cuda/std/type_traits").exists()
        })
        .collect()
}

fn load(module: &Arc<CudaModule>, name: &'static str) -> Result<CudaFunction> {
    module.load_function(name).map_err(move |error| {
        AegisError::Unsupported(format!("load cuda function `{name}` failed: {error:?}"))
    })
}

fn map_cuda_err(stage: &'static str) -> impl FnOnce(cudarc::driver::DriverError) -> AegisError {
    move |error| AegisError::Unsupported(format!("cuda stage `{stage}` failed: {error:?}"))
}

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
    pub(crate) router_zero_expert_counts: CudaFunction,
    pub(crate) router_bucket_sort: CudaFunction,
    pub(crate) router_expert_offsets: CudaFunction,
    pub(crate) permute_gather_f32: CudaFunction,
    pub(crate) unpermute_scatter_add_f32: CudaFunction,
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
    pub(crate) axpy_f32: CudaFunction,
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
            router_zero_expert_counts: load(&module, "aegis_router_zero_expert_counts")?,
            router_bucket_sort: load(&module, "aegis_router_bucket_sort")?,
            router_expert_offsets: load(&module, "aegis_router_expert_offsets")?,
            permute_gather_f32: load(&module, "aegis_permute_gather_f32")?,
            unpermute_scatter_add_f32: load(&module, "aegis_unpermute_scatter_add_f32")?,
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
            axpy_f32: load(&module, "aegis_axpy_f32")?,
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

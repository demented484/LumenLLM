use std::{env, path::Path, sync::Arc};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule};
use cudarc::nvrtc::{CompileOptions, Ptx, compile_ptx_with_opts};

use super::compile::nvrtc_arch_for_device;
use super::kernels::BLACKWELL_FP4_KERNEL_SRC;
use crate::error::{AegisError, Result};

#[derive(Debug)]
pub(crate) struct CudaKernelFunctions {
    pub(crate) _module: Arc<CudaModule>,
    pub(crate) blackwell_fp4: CudaFunction,
    pub(crate) mxfp4_matvec: CudaFunction,
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
    pub(crate) nvfp4_reference: CudaFunction,
    pub(crate) nvfp4_reference_batched: CudaFunction,
    pub(crate) nvfp4_prequant: CudaFunction,
    pub(crate) nvfp4_prequant_batched: CudaFunction,
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
    pub(crate) swiglu: CudaFunction,
    pub(crate) rope: CudaFunction,
    pub(crate) rope_batched: CudaFunction,
    pub(crate) rope_positions_batched: CudaFunction,
    pub(crate) rope_positions_batched_f16_out: CudaFunction,
    pub(crate) build_dense_prefill_metadata: CudaFunction,
    pub(crate) f32_to_f16: CudaFunction,
    pub(crate) kv_store: CudaFunction,
    pub(crate) kv_store_batched: CudaFunction,
    pub(crate) kv_store_slots_batched: CudaFunction,
    pub(crate) rope_kv_store_slots_batched: CudaFunction,
    pub(crate) attention: CudaFunction,
    pub(crate) attention_decode_streaming: CudaFunction,
    pub(crate) attention_prefill_batched: CudaFunction,
    pub(crate) attention_prefill_continuation: CudaFunction,
    pub(crate) attention_prefill_batched_warp: CudaFunction,
    pub(crate) attention_prefill_paged_varlen: CudaFunction,
    pub(crate) attention_prefill_paged_varlen_halfq: CudaFunction,
    pub(crate) attention_prefill_paged_varlen_halfq_block4: CudaFunction,
    pub(crate) attention_prefill_dense_halfq_block4: CudaFunction,
    pub(crate) attention_prefill_dense_halfq_warp_tile_hdim128: CudaFunction,
    pub(crate) attention_prefill_dense_halfq_wmma_hdim128: CudaFunction,
    pub(crate) attention_prefill_dense_halfq_wmma_hdim128_fa: CudaFunction,
    pub(crate) attention_prefill_dense_halfq_wmma_hdim128_gqa4: CudaFunction,
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
            nvfp4_reference: load(&module, "aegis_nvfp4_linear_reference")?,
            nvfp4_reference_batched: load(&module, "aegis_nvfp4_linear_reference_batched")?,
            nvfp4_prequant: load(&module, "aegis_nvfp4_linear_prequantized")?,
            nvfp4_prequant_batched: load(&module, "aegis_nvfp4_linear_prequantized_batched")?,
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
            swiglu: load(&module, "aegis_swiglu")?,
            rope: load(&module, "aegis_apply_rope")?,
            rope_batched: load(&module, "aegis_apply_rope_batched")?,
            rope_positions_batched: load(&module, "aegis_apply_rope_positions_batched")?,
            rope_positions_batched_f16_out: load(
                &module,
                "aegis_apply_rope_positions_batched_f16_out",
            )?,
            build_dense_prefill_metadata: load(&module, "aegis_build_dense_prefill_metadata")?,
            f32_to_f16: load(&module, "aegis_f32_to_f16")?,
            kv_store: load(&module, "aegis_kv_store")?,
            kv_store_batched: load(&module, "aegis_kv_store_batched")?,
            kv_store_slots_batched: load(&module, "aegis_kv_store_slots_batched")?,
            rope_kv_store_slots_batched: load(&module, "aegis_rope_kv_store_slots_batched")?,
            attention: load(&module, "aegis_attention_decode")?,
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
            attention_prefill_dense_halfq_wmma_hdim128_fa: load(
                &module,
                "aegis_attention_prefill_dense_halfq_wmma_hdim128_fa",
            )?,
            attention_prefill_dense_halfq_wmma_hdim128_gqa4: load(
                &module,
                "aegis_attention_prefill_dense_halfq_wmma_hdim128_gqa4",
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

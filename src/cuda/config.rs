use crate::error::{AegisError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CudaRuntimeConfig {
    pub native_mxfp4_repack: bool,
    pub cutlass_nvfp4_repack: bool,
    pub native_mxfp4_inference: bool,
    pub prefill_attention: CudaPrefillAttentionKernel,
    pub prefill_chunk_size: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CudaPrefillAttentionKernel {
    #[default]
    Auto,
    Reference,
    WarpFlash,
    Continuation,
}

impl CudaRuntimeConfig {
    pub fn from_env() -> Self {
        Self {
            native_mxfp4_repack: std::env::var_os("AEGISLLM_NATIVE_MXFP4_REPACK").is_some(),
            cutlass_nvfp4_repack: std::env::var_os("AEGISLLM_CUTLASS_NVFP4_REPACK").is_some(),
            native_mxfp4_inference: std::env::var_os("AEGISLLM_NATIVE_MXFP4_INFERENCE").is_some(),
            prefill_attention: std::env::var("AEGISLLM_CUDA_PREFILL_ATTENTION")
                .ok()
                .and_then(|value| CudaPrefillAttentionKernel::parse(&value).ok())
                .unwrap_or_default(),
            prefill_chunk_size: std::env::var("AEGIS_CUDA_PREFILL_CHUNK")
                .ok()
                .and_then(|value| value.parse::<usize>().ok()),
        }
    }
}

impl CudaPrefillAttentionKernel {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "reference" | "scalar" | "legacy" | "off" | "false" => Ok(Self::Reference),
            "warp" | "warp-flash" | "flash" | "flash-attention" | "on" | "true" => {
                Ok(Self::WarpFlash)
            }
            "continuation" | "varlen" | "online" | "online-softmax" | "flash-varlen" => {
                Ok(Self::Continuation)
            }
            other => Err(AegisError::InvalidConfig(format!(
                "unsupported CUDA prefill attention kernel `{other}`"
            ))),
        }
    }
}

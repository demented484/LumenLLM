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
    Off,
    Sdpa,
    FlashAttention2,
    FlashAttention3,
    FlashAttention4,
    AegisVarlen,
    Reference,
    WarpFlash,
    Continuation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaAttentionBackend {
    Reference,
    Sdpa,
    FlashAttention2,
    FlashAttention3,
    FlashAttention4,
    AegisVarlen,
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
            "off" | "false" | "none" | "disabled" => Ok(Self::Off),
            "reference" | "scalar" | "legacy" => Ok(Self::Reference),
            "sdpa" | "spda" | "scaled-dot-product" | "scaled-dot-product-attention" => {
                Ok(Self::Sdpa)
            }
            "fa2" | "flash2" | "flash-attention-2" | "flashattention2" => Ok(Self::FlashAttention2),
            "fa3" | "flash3" | "flash-attention-3" | "flashattention3" => Ok(Self::FlashAttention3),
            "fa4" | "flash4" | "flash-attention-4" | "flashattention4" => Ok(Self::FlashAttention4),
            "aegis-varlen" | "aegis-paged" | "paged-online" | "paged-varlen" | "flash-varlen"
            | "fa-varlen" | "flash-varlen-paged" | "varlen" => Ok(Self::AegisVarlen),
            "warp" | "warp-flash" | "flash" | "flash-attention" | "on" | "true" => {
                Ok(Self::WarpFlash)
            }
            "continuation" | "online" | "online-softmax" => Ok(Self::Continuation),
            other => Err(AegisError::InvalidConfig(format!(
                "unsupported CUDA prefill attention kernel `{other}`"
            ))),
        }
    }

    pub fn canonical_name(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Off => "off",
            Self::Sdpa => "sdpa",
            Self::FlashAttention2 => "fa2",
            Self::FlashAttention3 => "fa3",
            Self::FlashAttention4 => "fa4",
            Self::AegisVarlen => "aegis-varlen",
            Self::Reference => "reference",
            Self::WarpFlash => "warp-flash",
            Self::Continuation => "continuation",
        }
    }
}

impl CudaAttentionBackend {
    pub fn auto_target_for_compute_capability(compute_capability: Option<&str>) -> Self {
        let cc = compute_capability.unwrap_or_default();
        if cc.starts_with("12.") {
            Self::FlashAttention4
        } else if cc.starts_with("9.") {
            Self::FlashAttention3
        } else if cc.starts_with("8.") {
            Self::FlashAttention2
        } else {
            Self::Sdpa
        }
    }

    pub fn canonical_name(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::Sdpa => "sdpa",
            Self::FlashAttention2 => "fa2",
            Self::FlashAttention3 => "fa3",
            Self::FlashAttention4 => "fa4",
            Self::AegisVarlen => "aegis-varlen",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CudaPrefillAttentionKernel;

    #[test]
    fn parses_flash_attention_4_aliases() {
        for alias in ["fa4", "flash4", "flash-attention-4", "flashattention4"] {
            assert_eq!(
                CudaPrefillAttentionKernel::parse(alias).expect("alias should parse"),
                CudaPrefillAttentionKernel::FlashAttention4
            );
        }
    }

    #[test]
    fn parses_new_attention_backend_names() {
        assert_eq!(
            CudaPrefillAttentionKernel::parse("off").unwrap(),
            CudaPrefillAttentionKernel::Off
        );
        assert_eq!(
            CudaPrefillAttentionKernel::parse("sdpa").unwrap(),
            CudaPrefillAttentionKernel::Sdpa
        );
        assert_eq!(
            CudaPrefillAttentionKernel::parse("fa2").unwrap(),
            CudaPrefillAttentionKernel::FlashAttention2
        );
        assert_eq!(
            CudaPrefillAttentionKernel::parse("fa3").unwrap(),
            CudaPrefillAttentionKernel::FlashAttention3
        );
        assert_eq!(
            CudaPrefillAttentionKernel::parse("flash-varlen").unwrap(),
            CudaPrefillAttentionKernel::AegisVarlen
        );
        assert_eq!(
            CudaPrefillAttentionKernel::parse("aegis-varlen").unwrap(),
            CudaPrefillAttentionKernel::AegisVarlen
        );
    }

    #[test]
    fn auto_backend_policy_tracks_cuda_generation() {
        use super::CudaAttentionBackend;

        assert_eq!(
            CudaAttentionBackend::auto_target_for_compute_capability(Some("8.9")),
            CudaAttentionBackend::FlashAttention2
        );
        assert_eq!(
            CudaAttentionBackend::auto_target_for_compute_capability(Some("9.0")),
            CudaAttentionBackend::FlashAttention3
        );
        assert_eq!(
            CudaAttentionBackend::auto_target_for_compute_capability(Some("12.0")),
            CudaAttentionBackend::FlashAttention4
        );
        assert_eq!(
            CudaAttentionBackend::auto_target_for_compute_capability(None),
            CudaAttentionBackend::Sdpa
        );
    }
}

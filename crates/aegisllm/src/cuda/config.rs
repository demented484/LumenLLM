use crate::error::{AegisError, Result};

pub(crate) const CUDA_PREFILL_VARLEN_MIN_CONTEXT: usize = 128;
pub(crate) const CUDA_PREFILL_CHUNK_MAX: usize = 8192;
pub(crate) const CUDA_PREFILL_DENSE_SPLIT_K_TOKENS: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CudaRuntimeConfig {
    pub native_mxfp4_repack: bool,
    pub cutlass_nvfp4_repack: bool,
    pub native_mxfp4_inference: bool,
    pub prefill_attention: CudaPrefillAttentionKernel,
    pub prefill_chunk_size: Option<usize>,
    pub prefill_stage_timings: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CudaPrefillAttentionKernel {
    #[default]
    Auto,
    Off,
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
    FlashAttention2,
    FlashAttention3,
    FlashAttention4,
    AegisVarlen,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaAttentionEffectivePath {
    ReferenceCacheOnly,
    ReferenceContinuation,
    AegisDenseWarpTile,
    AegisDenseWmmaTile,
    AegisDenseWmmaFaPipeline,
    AegisDenseWmmaGqa4,
    AegisDenseWmmaGqa4SplitK,
    AegisDenseWmmaCluster2,
    AegisDenseWmmaPersistentQ32,
    AegisDenseWmmaSplitK,
    AegisPagedVarlen,
    AegisPagedWmmaGqa4,
    FlashAttention4PagedVarlen,
    WarpFlash,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaPrefillAttentionSelection {
    pub requested: CudaPrefillAttentionKernel,
    pub auto_target: Option<CudaAttentionBackend>,
    pub logical_backend: CudaAttentionBackend,
    pub effective_path: CudaAttentionEffectivePath,
    pub reason: &'static str,
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
            prefill_stage_timings: std::env::var_os("AEGISLLM_CUDA_STAGE_TIMINGS").is_some(),
        }
    }

    pub fn prefill_attention_selection(
        self,
        compute_capability: Option<&str>,
        context_len: usize,
        head_dim: usize,
        num_attention_heads: usize,
        num_kv_heads: usize,
    ) -> CudaPrefillAttentionSelection {
        CudaPrefillAttentionSelection::select(
            self.prefill_attention,
            compute_capability,
            context_len,
            head_dim,
            num_attention_heads,
            num_kv_heads,
        )
    }
}

impl CudaPrefillAttentionKernel {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "off" | "false" | "none" | "disabled" => Ok(Self::Off),
            "reference" | "scalar" | "legacy" => Ok(Self::Reference),
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
            Self::AegisVarlen
        }
    }

    pub fn canonical_name(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::FlashAttention2 => "fa2",
            Self::FlashAttention3 => "fa3",
            Self::FlashAttention4 => "fa4",
            Self::AegisVarlen => "aegis-varlen",
        }
    }
}

impl CudaAttentionEffectivePath {
    pub fn canonical_name(self) -> &'static str {
        match self {
            Self::ReferenceCacheOnly => "reference/cache-only",
            Self::ReferenceContinuation => "reference/continuation",
            Self::AegisDenseWarpTile => "aegis-varlen/dense-warp-tile",
            Self::AegisDenseWmmaTile => "aegis-varlen/dense-wmma-tile",
            Self::AegisDenseWmmaFaPipeline => "aegis-varlen/dense-wmma-fa-pipeline",
            Self::AegisDenseWmmaGqa4 => "aegis-varlen/dense-wmma-gqa4",
            Self::AegisDenseWmmaGqa4SplitK => "aegis-varlen/dense-wmma-gqa4-stream-k",
            Self::AegisDenseWmmaCluster2 => "aegis-varlen/dense-wmma-cluster2",
            Self::AegisDenseWmmaPersistentQ32 => "aegis-varlen/dense-wmma-persistent-q32",
            Self::AegisDenseWmmaSplitK => "aegis-varlen/dense-wmma-split-k",
            Self::AegisPagedVarlen => "aegis-varlen/paged-varlen",
            Self::AegisPagedWmmaGqa4 => "aegis-varlen/paged-wmma-gqa4",
            Self::FlashAttention4PagedVarlen => "fa4/paged-varlen",
            Self::WarpFlash => "warp-flash/cache-only",
        }
    }
}

impl CudaPrefillAttentionSelection {
    pub fn select(
        requested: CudaPrefillAttentionKernel,
        compute_capability: Option<&str>,
        context_len: usize,
        head_dim: usize,
        num_attention_heads: usize,
        num_kv_heads: usize,
    ) -> Self {
        let legacy_shared_bytes = (context_len + 128) * std::mem::size_of::<f32>();
        let oversized_dense_scores = legacy_shared_bytes > 48 * 1024;
        let gqa_group = if num_kv_heads == 0 {
            1
        } else {
            num_attention_heads / num_kv_heads
        };
        match requested {
            CudaPrefillAttentionKernel::Auto => {
                let auto_target =
                    CudaAttentionBackend::auto_target_for_compute_capability(compute_capability);
                let dense_warp_tile_eligible =
                    head_dim == 128 && context_len >= CUDA_PREFILL_VARLEN_MIN_CONTEXT;
                let dense_gqa4_eligible = dense_warp_tile_eligible && gqa_group >= 4;
                let dense_split_k_eligible = dense_split_k_enabled() && head_dim == 128;
                let dense_q32_experimental =
                    std::env::var_os("AEGISLLM_CUDA_EXPERIMENTAL_PERSISTENT_ATTENTION").is_some()
                        && head_dim == 128
                        && context_len >= 1024;
                let dense_cluster2_experimental =
                    std::env::var_os("AEGISLLM_CUDA_EXPERIMENTAL_CLUSTER_ATTENTION").is_some()
                        && head_dim == 128
                        && context_len >= 1024;
                let warp_eligible =
                    head_dim % 32 == 0 && head_dim <= 256 && !oversized_dense_scores;
                let (logical_backend, effective_path, reason) = if dense_split_k_eligible {
                    if dense_gqa4_eligible {
                        (
                            CudaAttentionBackend::AegisVarlen,
                            CudaAttentionEffectivePath::AegisDenseWmmaGqa4SplitK,
                            "auto selected explicitly enabled GQA4 split-K dense WMMA-tiled prefill attention",
                        )
                    } else {
                        (
                            CudaAttentionBackend::AegisVarlen,
                            CudaAttentionEffectivePath::AegisDenseWmmaSplitK,
                            "auto selected explicitly enabled split-K dense WMMA-tiled prefill attention",
                        )
                    }
                } else if dense_cluster2_experimental {
                    (
                        CudaAttentionBackend::AegisVarlen,
                        CudaAttentionEffectivePath::AegisDenseWmmaCluster2,
                        "auto selected experimental cluster2 dense WMMA-tiled prefill attention",
                    )
                } else if dense_q32_experimental {
                    (
                        CudaAttentionBackend::AegisVarlen,
                        CudaAttentionEffectivePath::AegisDenseWmmaPersistentQ32,
                        "auto selected experimental q32 persistent dense WMMA-tiled prefill attention",
                    )
                } else if dense_gqa4_eligible {
                    (
                        CudaAttentionBackend::AegisVarlen,
                        CudaAttentionEffectivePath::AegisDenseWmmaGqa4,
                        "auto selected GQA4 fused Aegis FA-style dense WMMA-tiled prefill attention",
                    )
                } else if dense_warp_tile_eligible {
                    (
                        CudaAttentionBackend::AegisVarlen,
                        CudaAttentionEffectivePath::AegisDenseWmmaFaPipeline,
                        "auto selected Aegis FA-style dense WMMA-tiled prefill attention for head_dim=128",
                    )
                } else if warp_eligible {
                    (
                        CudaAttentionBackend::Reference,
                        CudaAttentionEffectivePath::WarpFlash,
                        "auto selected dense first-prefill warp specialization; continuation chunks fall back to paged-varlen or bounded reference as needed",
                    )
                } else {
                    match auto_target {
                        CudaAttentionBackend::FlashAttention4
                            if context_len >= CUDA_PREFILL_VARLEN_MIN_CONTEXT =>
                        {
                            (
                                CudaAttentionBackend::AegisVarlen,
                                CudaAttentionEffectivePath::AegisPagedVarlen,
                                "auto selected Blackwell-class attention; using Aegis paged-varlen path until FA4 is promoted",
                            )
                        }
                        CudaAttentionBackend::FlashAttention2
                        | CudaAttentionBackend::FlashAttention3
                            if context_len >= CUDA_PREFILL_VARLEN_MIN_CONTEXT =>
                        {
                            (
                                CudaAttentionBackend::AegisVarlen,
                                CudaAttentionEffectivePath::AegisPagedVarlen,
                                "auto selected flash attention generation; using Aegis paged-varlen path for long context",
                            )
                        }
                        _ if oversized_dense_scores => (
                            CudaAttentionBackend::Reference,
                            CudaAttentionEffectivePath::ReferenceContinuation,
                            "dense score buffer exceeds bounded shared-memory policy",
                        ),
                        _ => (
                            CudaAttentionBackend::Reference,
                            CudaAttentionEffectivePath::ReferenceCacheOnly,
                            "short context uses dense cache-only reference path",
                        ),
                    }
                };
                Self {
                    requested,
                    auto_target: Some(auto_target),
                    logical_backend,
                    effective_path,
                    reason,
                }
            }
            CudaPrefillAttentionKernel::Off | CudaPrefillAttentionKernel::Reference => Self {
                requested,
                auto_target: None,
                logical_backend: CudaAttentionBackend::Reference,
                effective_path: CudaAttentionEffectivePath::ReferenceCacheOnly,
                reason: if oversized_dense_scores {
                    "reference requested but dense scores exceed bounded shared-memory policy; runtime rejects this shape unless aegis-varlen, auto, or continuation is requested"
                } else {
                    "reference requested"
                },
            },
            CudaPrefillAttentionKernel::FlashAttention4 => Self {
                requested,
                auto_target: None,
                logical_backend: CudaAttentionBackend::FlashAttention4,
                effective_path: CudaAttentionEffectivePath::FlashAttention4PagedVarlen,
                reason: "fa4 requested explicitly",
            },
            CudaPrefillAttentionKernel::AegisVarlen => Self {
                requested,
                auto_target: None,
                logical_backend: CudaAttentionBackend::AegisVarlen,
                effective_path: if head_dim == 128 && context_len >= CUDA_PREFILL_VARLEN_MIN_CONTEXT
                {
                    if dense_split_k_enabled() {
                        if gqa_group >= 4 {
                            CudaAttentionEffectivePath::AegisDenseWmmaGqa4SplitK
                        } else {
                            CudaAttentionEffectivePath::AegisDenseWmmaSplitK
                        }
                    } else if std::env::var_os("AEGISLLM_CUDA_EXPERIMENTAL_CLUSTER_ATTENTION")
                        .is_some()
                        && context_len >= 1024
                    {
                        CudaAttentionEffectivePath::AegisDenseWmmaCluster2
                    } else if std::env::var_os("AEGISLLM_CUDA_EXPERIMENTAL_PERSISTENT_ATTENTION")
                        .is_some()
                        && context_len >= 1024
                    {
                        CudaAttentionEffectivePath::AegisDenseWmmaPersistentQ32
                    } else if gqa_group >= 4 {
                        CudaAttentionEffectivePath::AegisDenseWmmaGqa4
                    } else {
                        CudaAttentionEffectivePath::AegisDenseWmmaFaPipeline
                    }
                } else if head_dim % 32 == 0 && head_dim <= 256 && !oversized_dense_scores {
                    CudaAttentionEffectivePath::WarpFlash
                } else {
                    CudaAttentionEffectivePath::AegisPagedVarlen
                },
                reason: "aegis-varlen requested; dense identity head_dim=128 prefill uses WMMA-tiled online attention, paged batches use paged-varlen",
            },
            CudaPrefillAttentionKernel::WarpFlash => Self {
                requested,
                auto_target: None,
                logical_backend: CudaAttentionBackend::Reference,
                effective_path: if head_dim % 32 == 0 && head_dim <= 256 {
                    CudaAttentionEffectivePath::WarpFlash
                } else if oversized_dense_scores {
                    CudaAttentionEffectivePath::ReferenceContinuation
                } else {
                    CudaAttentionEffectivePath::ReferenceCacheOnly
                },
                reason: "warp-flash requested; falls back when the head dimension is not warp-friendly",
            },
            CudaPrefillAttentionKernel::Continuation => Self {
                requested,
                auto_target: None,
                logical_backend: CudaAttentionBackend::Reference,
                effective_path: CudaAttentionEffectivePath::ReferenceContinuation,
                reason: "online continuation requested explicitly",
            },
            CudaPrefillAttentionKernel::FlashAttention2 => Self {
                requested,
                auto_target: None,
                logical_backend: CudaAttentionBackend::FlashAttention2,
                effective_path: CudaAttentionEffectivePath::AegisPagedVarlen,
                reason: "fa2 frontend is reserved; runtime reports unsupported before launch",
            },
            CudaPrefillAttentionKernel::FlashAttention3 => Self {
                requested,
                auto_target: None,
                logical_backend: CudaAttentionBackend::FlashAttention3,
                effective_path: CudaAttentionEffectivePath::AegisPagedVarlen,
                reason: "fa3 frontend is reserved; runtime reports unsupported before launch",
            },
        }
    }
}

fn dense_split_k_enabled() -> bool {
    std::env::var_os("AEGISLLM_CUDA_DISABLE_SPLIT_K_ATTENTION").is_none()
        && std::env::var_os("AEGISLLM_CUDA_EXPERIMENTAL_SPLIT_K_ATTENTION").is_some()
}

#[cfg(test)]
mod tests {
    use super::{
        CudaAttentionBackend, CudaAttentionEffectivePath, CudaPrefillAttentionKernel,
        CudaPrefillAttentionSelection,
    };

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
            CudaAttentionBackend::AegisVarlen
        );
    }

    #[test]
    fn auto_policy_reports_blackwell_target_and_validated_path() {
        let selection = CudaPrefillAttentionSelection::select(
            CudaPrefillAttentionKernel::Auto,
            Some("12.0"),
            128,
            128,
            32,
            32,
        );
        assert_eq!(
            selection.auto_target,
            Some(CudaAttentionBackend::FlashAttention4)
        );
        assert_eq!(selection.logical_backend, CudaAttentionBackend::AegisVarlen);
        assert_eq!(
            selection.effective_path,
            CudaAttentionEffectivePath::AegisDenseWmmaFaPipeline
        );
    }

    #[test]
    fn explicit_varlen_reports_dense_first_prefill_specialization() {
        let selection = CudaPrefillAttentionSelection::select(
            CudaPrefillAttentionKernel::AegisVarlen,
            Some("12.0"),
            1024,
            128,
            32,
            32,
        );
        assert_eq!(selection.logical_backend, CudaAttentionBackend::AegisVarlen);
        assert_eq!(
            selection.effective_path,
            CudaAttentionEffectivePath::AegisDenseWmmaFaPipeline
        );
    }

    #[test]
    fn auto_policy_reports_gqa_fused_path_for_grouped_query_models() {
        let selection = CudaPrefillAttentionSelection::select(
            CudaPrefillAttentionKernel::Auto,
            Some("12.0"),
            1024,
            128,
            32,
            8,
        );
        assert_eq!(selection.logical_backend, CudaAttentionBackend::AegisVarlen);
        assert_eq!(
            selection.effective_path,
            CudaAttentionEffectivePath::AegisDenseWmmaGqa4
        );
    }
}

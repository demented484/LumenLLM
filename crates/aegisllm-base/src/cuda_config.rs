use crate::error::{AegisError, Result};

pub const CUDA_PREFILL_VARLEN_MIN_CONTEXT: usize = 128;
pub const CUDA_PREFILL_CHUNK_MAX: usize = 8192;
pub const CUDA_PREFILL_DENSE_SPLIT_K_TOKENS: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CudaRuntimeConfig {
    pub native_mxfp4_repack: bool,
    pub cutlass_nvfp4_repack: bool,
    pub native_mxfp4_inference: bool,
    pub prefill_attention: CudaPrefillAttentionKernel,
    pub prefill_chunk_size: Option<usize>,
    pub prefill_stage_timings: bool,
    /// Precision the prefill/decode attention KERNEL runs in. This is the
    /// attention *compute* path, distinct from `attention-quantization`
    /// (which re-quantizes the Q/K/V/O *weights* at load time) and from
    /// `kv-cache.type-k/type-v` (which sets the K/V cache storage dtype).
    ///
    /// Set by `attention.compute-quantization` in `parameters.*.json`.
    /// Converges with the `AEGIS_ATTN_FP8` / `AEGIS_ATTN_FA2` env gates:
    /// the attention dispatch enables a path if EITHER the env var is set
    /// OR this field requests it (env var stays a working override).
    pub attention_compute_quant: AttentionComputeQuant,
}

/// Precision selector for the attention compute kernels (prefill + decode).
///
/// `Default` is bit-equivalent to main: the dispatch keeps honoring only the
/// `AEGIS_ATTN_*` env gates. The non-default variants drive those same gates
/// from config so the user does not have to export env vars.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttentionComputeQuant {
    /// Keep the historical default dispatch (env gates only).
    #[default]
    Default,
    /// Run the BF16 (half-precision) attention kernels — the legacy WMMA
    /// path. Equivalent to `Default` for dispatch, named for clarity in
    /// configs that want to be explicit about precision.
    Bf16,
    /// Run the BF16 FlashAttention-2 rewrite (head_dim=512 path). Drives
    /// the same gate as `AEGIS_ATTN_FA2=1`.
    Bf16Fa2,
    /// Run the FP8 (E4M3) attention kernels. Drives the same gate as
    /// `AEGIS_ATTN_FP8=1`. Requires the KV cache to be FP8 because the
    /// FP8 attention kernel reads FP8 K/V directly.
    Fp8,
}

impl AttentionComputeQuant {
    /// Parse the `attention.compute-quantization` config value.
    /// `default` / `bf16` / `bf16-fa2` / `fp8` (with a few aliases).
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "default" | "auto" | "" => Ok(Self::Default),
            "bf16" | "bfloat16" | "half" => Ok(Self::Bf16),
            "bf16-fa2" | "bf16_fa2" | "fa2" | "flash-attention-2" => Ok(Self::Bf16Fa2),
            "fp8" | "f8" | "fp8-e4m3" | "fp8_e4m3" => Ok(Self::Fp8),
            other => Err(AegisError::InvalidConfig(format!(
                "unsupported attention compute-quantization `{other}` \
                 (use one of: default, bf16, bf16-fa2, fp8)"
            ))),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Bf16 => "bf16",
            Self::Bf16Fa2 => "bf16-fa2",
            Self::Fp8 => "fp8",
        }
    }

    /// Whether this selection requests the FP8 attention compute path.
    pub fn wants_fp8(self) -> bool {
        matches!(self, Self::Fp8)
    }

    /// Whether this selection requests the BF16 FlashAttention-2 path.
    pub fn wants_fa2(self) -> bool {
        matches!(self, Self::Bf16Fa2)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CudaPrefillAttentionKernel {
    #[default]
    Auto,
    Off,
    FlashAttention2,
    FlashAttention3,
    FlashAttention4,
    /// FP8 (E4M3) native-MMA prefill attention. Routes the head_dim=512 path
    /// through the `attention_prefill_fa2_fp8_mma` kernels. Requires an FP8 KV
    /// cache — selecting this without an FP8 config is rejected at engine build.
    Fp8,
    AegisVarlen,
    Reference,
    WarpFlash,
    Continuation,
}

/// The single, resolved top-level attention compute backend for a whole
/// prefill. This is the output of [`CudaRuntimeConfig::resolve_attention_backend`]
/// — the ONE decision point that unifies the `CudaPrefillAttentionKernel` enum
/// (`--cuda-prefill-attention`) with the `AEGIS_ATTN_FA2` / `AEGIS_ATTN_FP8`
/// env shortcuts. Every site that used to consult `attention_fa2_enabled()` /
/// `attention_fp8_enabled()` independently now funnels through this so an
/// explicit enum value (especially `Reference`) always wins over the env vars.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionComputeBackend {
    /// f32 scalar reference oracle — forced for ALL layers, env overrides
    /// suppressed. A `reference` run is a true oracle.
    Reference,
    /// The historical default WMMA half-precision dispatch (env-neutral).
    Bf16,
    /// BF16 FlashAttention-2 rewrite (head_dim=512 `attention_prefill_dense_fa2_*`).
    Fa2,
    /// FP8 (E4M3) native-MMA prefill attention (`attention_prefill_fa2_fp8_mma`).
    Fp8,
}

impl AttentionComputeBackend {
    pub fn canonical_name(self) -> &'static str {
        match self {
            Self::Reference => "reference",
            Self::Bf16 => "bf16",
            Self::Fa2 => "fa2",
            Self::Fp8 => "fp8",
        }
    }
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
            // `from_env` keeps `Default` here: the historical `AEGIS_ATTN_FP8` /
            // `AEGIS_ATTN_FA2` env gates are read directly at the attention
            // dispatch site, so leaving this `Default` preserves bit-equivalent
            // behavior when no config drives it. The config path sets this
            // field explicitly in `into_engine_fragment`.
            attention_compute_quant: AttentionComputeQuant::Default,
        }
    }

    /// THE single top-level attention-backend decision point.
    ///
    /// Unifies the two historically independent mechanisms:
    ///   * the `CudaPrefillAttentionKernel` enum (`--cuda-prefill-attention`),
    ///   * the `AEGIS_ATTN_FA2` / `AEGIS_ATTN_FP8` env shortcuts (and the
    ///     equivalent `attention.compute-quantization` config field).
    ///
    /// Resolution order — an explicit enum value ALWAYS wins over the env
    /// shortcuts, so `reference` is a true oracle even with `AEGIS_ATTN_FA2=1`
    /// exported:
    ///   * `Reference` / `Off`  → [`AttentionComputeBackend::Reference`]
    ///     (env FA-2/FP8 overrides SUPPRESSED).
    ///   * `FlashAttention2`    → [`AttentionComputeBackend::Fa2`].
    ///   * `Fp8`                → [`AttentionComputeBackend::Fp8`].
    ///   * `Auto` (and the other enum values that don't pin a compute
    ///     precision) → FP8 if requested (`compute-quantization: fp8` /
    ///     `AEGIS_ATTN_FP8`); else legacy [`AttentionComputeBackend::Bf16`] if
    ///     explicitly opted in (`compute-quantization: bf16`); else the
    ///     DEFAULT [`AttentionComputeBackend::Fa2`] (FA-2 for hdim=512).
    ///
    /// `prefill_dense.rs` and `layer.rs` both consult this; nothing reads the
    /// `AEGIS_ATTN_*` env vars or `compute_quant` directly any more.
    pub fn resolve_attention_backend(self) -> AttentionComputeBackend {
        match self.prefill_attention {
            // Explicit enum values pin the backend and suppress the env vars.
            CudaPrefillAttentionKernel::Reference | CudaPrefillAttentionKernel::Off => {
                AttentionComputeBackend::Reference
            }
            CudaPrefillAttentionKernel::FlashAttention2 => AttentionComputeBackend::Fa2,
            CudaPrefillAttentionKernel::Fp8 => AttentionComputeBackend::Fp8,
            // Auto / AegisVarlen / WarpFlash / Continuation / FA3 / FA4 do not
            // pin a compute precision: the env shortcuts (and the equivalent
            // config field) are honored here, funnelled through this one site.
            _ => {
                let env_fp8 = std::env::var("AEGIS_ATTN_FP8").as_deref() == Ok("1");
                if self.attention_compute_quant.wants_fp8() || env_fp8 {
                    AttentionComputeBackend::Fp8
                } else if matches!(self.attention_compute_quant, AttentionComputeQuant::Bf16) {
                    // Explicit opt-out (`compute-quantization: bf16`) → the
                    // legacy WMMA hdim-512 kernel.
                    AttentionComputeBackend::Bf16
                } else {
                    // DEFAULT: FA-2 for hdim=512 — validated equal-accuracy to
                    // the legacy WMMA kernel (per-layer cosine in-band, greedy
                    // output character-identical via quality-smoke, GPU f32
                    // reference oracle confirmed correct) and +32% prefill
                    // @256k. hdim 256/128 are unaffected: the downstream
                    // `use_fa2` gate is `head_dim == 512`. `AEGIS_ATTN_FA2` and
                    // the `bf16-fa2` config value are now redundant (still
                    // accepted — they resolve here too).
                    AttentionComputeBackend::Fa2
                }
            }
        }
    }

    /// Effective FP8-attention decision. Delegates to the single
    /// [`resolve_attention_backend`](Self::resolve_attention_backend) gate so
    /// `reference`/`fa2`/explicit-enum selections override the env shortcut.
    pub fn attention_fp8_enabled(self) -> bool {
        matches!(self.resolve_attention_backend(), AttentionComputeBackend::Fp8)
    }

    /// Effective BF16 FlashAttention-2 decision. Delegates to the single
    /// [`resolve_attention_backend`](Self::resolve_attention_backend) gate.
    pub fn attention_fa2_enabled(self) -> bool {
        matches!(self.resolve_attention_backend(), AttentionComputeBackend::Fa2)
    }

    /// Whether the resolved backend forces the f32 reference oracle for all
    /// layers. When true, the FA-2/FP8 env shortcuts are SUPPRESSED.
    pub fn attention_reference_forced(self) -> bool {
        matches!(
            self.resolve_attention_backend(),
            AttentionComputeBackend::Reference
        )
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
            "fp8" | "f8" | "fp8-e4m3" | "fp8_e4m3" | "fp8-mma" => Ok(Self::Fp8),
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
            Self::Fp8 => "fp8",
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
                    head_dim.is_multiple_of(32) && head_dim <= 256 && !oversized_dense_scores;
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
                } else if head_dim.is_multiple_of(32) && head_dim <= 256 && !oversized_dense_scores {
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
                effective_path: if head_dim.is_multiple_of(32) && head_dim <= 256 {
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
                effective_path: CudaAttentionEffectivePath::AegisDenseWmmaFaPipeline,
                reason: "fa2 requested explicitly; head_dim=512 routes through the FA-2 dense rewrite",
            },
            CudaPrefillAttentionKernel::Fp8 => Self {
                requested,
                auto_target: None,
                logical_backend: CudaAttentionBackend::AegisVarlen,
                effective_path: CudaAttentionEffectivePath::AegisDenseWmmaFaPipeline,
                reason: "fp8 requested explicitly; head_dim=512 routes through the FP8 native-MMA dense path",
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

    use super::{AttentionComputeQuant, CudaRuntimeConfig};

    #[test]
    fn attention_compute_quant_parses_canonical_values() {
        assert_eq!(
            AttentionComputeQuant::parse("default").unwrap(),
            AttentionComputeQuant::Default
        );
        assert_eq!(
            AttentionComputeQuant::parse("bf16").unwrap(),
            AttentionComputeQuant::Bf16
        );
        assert_eq!(
            AttentionComputeQuant::parse("bf16-fa2").unwrap(),
            AttentionComputeQuant::Bf16Fa2
        );
        assert_eq!(
            AttentionComputeQuant::parse("fp8").unwrap(),
            AttentionComputeQuant::Fp8
        );
        // empty string and `auto` fold to Default.
        assert_eq!(
            AttentionComputeQuant::parse("").unwrap(),
            AttentionComputeQuant::Default
        );
        assert_eq!(
            AttentionComputeQuant::parse("AUTO").unwrap(),
            AttentionComputeQuant::Default
        );
    }

    #[test]
    fn attention_compute_quant_rejects_unknown() {
        assert!(AttentionComputeQuant::parse("int3").is_err());
        assert!(AttentionComputeQuant::parse("nvfp4").is_err());
    }

    #[test]
    fn attention_compute_quant_default_is_default() {
        assert_eq!(
            AttentionComputeQuant::default(),
            AttentionComputeQuant::Default
        );
    }

    #[test]
    fn attention_fp8_enabled_follows_config_field() {
        // Config requesting FP8 enables the FP8 path regardless of env.
        let cfg = CudaRuntimeConfig {
            attention_compute_quant: AttentionComputeQuant::Fp8,
            ..CudaRuntimeConfig::default()
        };
        assert!(cfg.attention_fp8_enabled());
        // Default config does not enable FP8 (env var not set in test env).
        let cfg_default = CudaRuntimeConfig::default();
        assert!(!cfg_default.attention_fp8_enabled());
    }

    #[test]
    fn attention_fa2_enabled_follows_config_field() {
        let cfg = CudaRuntimeConfig {
            attention_compute_quant: AttentionComputeQuant::Bf16Fa2,
            ..CudaRuntimeConfig::default()
        };
        assert!(cfg.attention_fa2_enabled());
        // Since Stage B, FA-2 is the resolved default (hdim=512 prefill is
        // context-gated downstream). The explicit `bf16` opt-out disables it.
        assert!(CudaRuntimeConfig::default().attention_fa2_enabled());
        let cfg_bf16 = CudaRuntimeConfig {
            attention_compute_quant: AttentionComputeQuant::Bf16,
            ..CudaRuntimeConfig::default()
        };
        assert!(!cfg_bf16.attention_fa2_enabled());
    }

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
    fn resolve_backend_explicit_enum_overrides_env() {
        use super::AttentionComputeBackend;
        // `Reference` resolves to Reference regardless of the compute-quant
        // field — the unified gate makes an explicit enum win. (Env vars are
        // not exercised here to keep the test parallel-safe; the runtime
        // gates #4/#5 verify the env-suppression end to end.)
        let cfg_ref = CudaRuntimeConfig {
            prefill_attention: CudaPrefillAttentionKernel::Reference,
            attention_compute_quant: AttentionComputeQuant::Bf16Fa2,
            ..CudaRuntimeConfig::default()
        };
        assert_eq!(
            cfg_ref.resolve_attention_backend(),
            AttentionComputeBackend::Reference
        );
        assert!(!cfg_ref.attention_fa2_enabled());
        assert!(cfg_ref.attention_reference_forced());

        // `FlashAttention2` enum pins Fa2 even when the compute-quant field
        // asks for FP8 — the enum is the top-level override.
        let cfg_fa2 = CudaRuntimeConfig {
            prefill_attention: CudaPrefillAttentionKernel::FlashAttention2,
            attention_compute_quant: AttentionComputeQuant::Fp8,
            ..CudaRuntimeConfig::default()
        };
        assert_eq!(
            cfg_fa2.resolve_attention_backend(),
            AttentionComputeBackend::Fa2
        );

        // `Fp8` enum pins Fp8.
        let cfg_fp8 = CudaRuntimeConfig {
            prefill_attention: CudaPrefillAttentionKernel::Fp8,
            ..CudaRuntimeConfig::default()
        };
        assert_eq!(
            cfg_fp8.resolve_attention_backend(),
            AttentionComputeBackend::Fp8
        );

        // Auto + default compute-quant + no env → Fa2 (FA-2 is the default
        // prefill attention for hdim=512 since Stage B).
        let cfg_auto = CudaRuntimeConfig::default();
        assert_eq!(
            cfg_auto.resolve_attention_backend(),
            AttentionComputeBackend::Fa2
        );
        // `compute-quantization: bf16` is the explicit opt-out to the legacy
        // WMMA hdim-512 kernel.
        let cfg_bf16 = CudaRuntimeConfig {
            attention_compute_quant: AttentionComputeQuant::Bf16,
            ..CudaRuntimeConfig::default()
        };
        assert_eq!(
            cfg_bf16.resolve_attention_backend(),
            AttentionComputeBackend::Bf16
        );
    }

    #[test]
    fn fp8_kernel_alias_parses() {
        for alias in ["fp8", "f8", "fp8-e4m3", "fp8-mma"] {
            assert_eq!(
                CudaPrefillAttentionKernel::parse(alias).unwrap(),
                CudaPrefillAttentionKernel::Fp8
            );
        }
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

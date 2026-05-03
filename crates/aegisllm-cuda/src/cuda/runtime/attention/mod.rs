use super::{CudaRuntime, map_cuda_err};
use crate::cuda::{CudaPrefillAttentionKernel, DeviceBuffer};
use aegisllm_base::cuda_config::CUDA_PREFILL_DENSE_SPLIT_K_TOKENS;
use aegisllm_base::error::{AegisError, Result};

mod decode;
mod dispatch;
mod prefill_dense;
mod prefill_paged;

pub(super) fn u32_arg(name: &str, value: usize) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        AegisError::InvalidPlan(format!(
            "CUDA attention argument {name} exceeds u32 range: {value}"
        ))
    })
}

pub(super) fn checked_len(label: &str, lhs: usize, rhs: usize) -> Result<usize> {
    lhs.checked_mul(rhs).ok_or_else(|| {
        AegisError::InvalidPlan(format!(
            "CUDA attention {label} length overflow: {lhs} * {rhs}"
        ))
    })
}

pub(super) fn checked_sum(label: &str, lhs: usize, rhs: usize) -> Result<usize> {
    lhs.checked_add(rhs).ok_or_else(|| {
        AegisError::InvalidPlan(format!(
            "CUDA attention {label} length overflow: {lhs} + {rhs}"
        ))
    })
}

pub(super) fn validate_dynamic_shared_bytes(kernel: &str, bytes: usize) -> Result<u32> {
    if bytes > 48 * 1024 {
        return Err(AegisError::InvalidPlan(format!(
            "CUDA kernel `{kernel}` requires {bytes} bytes of dynamic shared memory, exceeding the conservative 48KiB launch limit"
        )));
    }
    Ok(bytes as u32)
}

pub(super) const FLASH_COMPAT_PAGE_TOKENS: usize = 256;
pub(super) const FLASH_SPLIT_K_TOKENS: usize = 256;
pub(super) const FLASH_SPLIT_Q_BLOCK: usize = 4;
pub(super) const TILED_HALFQ_Q_BLOCK: usize = 4;
pub(super) const DENSE_WARP_TILE_Q_BLOCK: usize = 16;
pub(super) const DENSE_WARP_TILE_K_TILE: usize = 32;
pub(super) const DENSE_WMMA_Q_BLOCK: usize = 16;
pub(super) const DENSE_WMMA_FA_Q_BLOCK: usize = 16;
pub(super) const DENSE_WMMA_GQA4_Q_TOKENS: usize = 8;
pub(super) const DENSE_WMMA_GQA4_HEADS: usize = 4;
pub(super) const DENSE_WMMA_GQA4_SPLIT_Q_TOKENS: usize = 8;
pub(super) const PAGED_WMMA_GQA4_Q_TOKENS: usize = 8;
pub(super) const PAGED_WMMA_GQA4_HEADS: usize = 4;
pub(super) const DENSE_WMMA_Q32_BLOCK: usize = 32;
pub(super) const DENSE_WMMA_K_TILE: usize = 32;
pub(super) const DENSE_WMMA_SPLIT_K_TOKENS: usize = CUDA_PREFILL_DENSE_SPLIT_K_TOKENS;
pub(super) const FA4_HDIM128_Q_BLOCK: usize = 8;
pub(super) const FA4_HDIM128_K_TILE: usize = 32;
pub(super) const CUDA_ATTENTION_BLOCK_DIM: u32 = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PrefillBatchedKernel {
    CacheOnly,
    Continuation,
    Warp,
}

pub(super) fn select_prefill_batched_kernel(
    config: CudaPrefillAttentionKernel,
    start_position: usize,
    head_dim: usize,
    legacy_shared_bytes: usize,
) -> Result<PrefillBatchedKernel> {
    let warp_eligible = start_position == 0 && head_dim.is_multiple_of(32) && head_dim <= 256;
    if matches!(
        config,
        CudaPrefillAttentionKernel::Auto
            | CudaPrefillAttentionKernel::AegisVarlen
            | CudaPrefillAttentionKernel::WarpFlash
    ) && warp_eligible
    {
        return Ok(PrefillBatchedKernel::Warp);
    }
    if matches!(config, CudaPrefillAttentionKernel::Continuation) {
        return Ok(PrefillBatchedKernel::Continuation);
    }
    if matches!(
        config,
        CudaPrefillAttentionKernel::Off | CudaPrefillAttentionKernel::Reference
    ) && legacy_shared_bytes > 48 * 1024
    {
        return Err(AegisError::InvalidPlan(format!(
            "CUDA reference prefill attention requires {} bytes of dynamic shared memory; use cuda.prefill-attention=aegis-varlen, auto, or continuation for long prefixes",
            legacy_shared_bytes
        )));
    }
    if !matches!(
        config,
        CudaPrefillAttentionKernel::Off | CudaPrefillAttentionKernel::Reference
    ) && legacy_shared_bytes > 48 * 1024
    {
        return Ok(PrefillBatchedKernel::Continuation);
    }
    Ok(PrefillBatchedKernel::CacheOnly)
}

pub(super) fn dense_wmma_split_k_enabled() -> bool {
    std::env::var_os("AEGISLLM_CUDA_DISABLE_SPLIT_K_ATTENTION").is_none()
        && std::env::var_os("AEGISLLM_CUDA_EXPERIMENTAL_SPLIT_K_ATTENTION").is_some()
}

pub(super) fn dense_wmma_q32_enabled() -> bool {
    std::env::var_os("AEGISLLM_CUDA_EXPERIMENTAL_PERSISTENT_ATTENTION").is_some()
}

pub(super) fn dense_wmma_legacy_enabled() -> bool {
    std::env::var_os("AEGISLLM_CUDA_LEGACY_WMMA_ATTENTION").is_some()
}

pub(super) fn dense_wmma_cluster2_enabled() -> bool {
    std::env::var_os("AEGISLLM_CUDA_EXPERIMENTAL_CLUSTER_ATTENTION").is_some()
}

pub(super) fn dense_wmma_split_scratch_ready(
    split_acc: &DeviceBuffer<f32>,
    split_m: &DeviceBuffer<f32>,
    split_l: &DeviceBuffer<f32>,
    batch: usize,
    context_len: usize,
    num_attention_heads: usize,
    head_dim: usize,
) -> bool {
    let split_count = context_len.div_ceil(DENSE_WMMA_SPLIT_K_TOKENS).max(1);
    let rows = batch
        .div_ceil(DENSE_WMMA_Q_BLOCK)
        .checked_mul(num_attention_heads)
        .and_then(|value| value.checked_mul(split_count))
        .and_then(|value| value.checked_mul(DENSE_WMMA_Q_BLOCK));
    let Some(rows) = rows else {
        return false;
    };
    let Some(acc_len) = rows.checked_mul(head_dim) else {
        return false;
    };
    split_acc.len() >= acc_len && split_m.len() >= rows && split_l.len() >= rows
}

#[cfg(test)]
mod tests {
    use super::{PrefillBatchedKernel, select_prefill_batched_kernel};
    use crate::cuda::CudaPrefillAttentionKernel;

    #[test]
    fn reference_prefill_rejects_oversized_shared_memory() {
        let error = select_prefill_batched_kernel(
            CudaPrefillAttentionKernel::Reference,
            0,
            128,
            256 * 1024,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("cuda.prefill-attention=aegis-varlen")
        );
    }

    #[test]
    fn warp_flash_still_prefers_warp_kernel_when_eligible() {
        assert_eq!(
            select_prefill_batched_kernel(CudaPrefillAttentionKernel::WarpFlash, 0, 128, 1024)
                .unwrap(),
            PrefillBatchedKernel::Warp
        );
    }

    #[test]
    fn varlen_first_prefill_uses_warp_specialization_when_dense() {
        assert_eq!(
            select_prefill_batched_kernel(CudaPrefillAttentionKernel::AegisVarlen, 0, 128, 1024)
                .unwrap(),
            PrefillBatchedKernel::Warp
        );
    }
}

use std::ffi::c_void;

use cudarc::driver::CudaSlice;

use super::owned_pinned::OwnedPinnedBuf;
use super::repack::CutlassNvfp4LinearLayout;
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::planning::runtime::KernelFamily;
use aegisllm_base::tensor::layout::LinearResidentLayout;
use aegisllm_base::tensor::storage::{HostTensorStorage, LoadedHostTensor, TensorResidencyPlan};

#[derive(Debug)]
pub struct DeviceBuffer<T> {
    pub(super) slice: CudaSlice<T>,
}

#[derive(Debug)]
pub struct CudaAttentionSplitScratch<'a> {
    pub acc: &'a mut DeviceBuffer<f32>,
    pub m: &'a mut DeviceBuffer<f32>,
    pub l: &'a mut DeviceBuffer<f32>,
}

#[derive(Debug)]
pub struct CudaAttentionRequest<'a> {
    pub q: &'a DeviceBuffer<f32>,
    pub q_half: Option<&'a DeviceBuffer<u16>>,
    pub k_cache: &'a DeviceBuffer<u16>,
    pub v_cache: &'a DeviceBuffer<u16>,
    pub cu_q: &'a DeviceBuffer<u32>,
    pub cu_k: &'a DeviceBuffer<u32>,
    pub context_lens: &'a DeviceBuffer<u32>,
    pub slot_mapping: &'a DeviceBuffer<u32>,
    pub block_tables: &'a DeviceBuffer<u32>,
    pub split_scratch: Option<CudaAttentionSplitScratch<'a>>,
    pub output: &'a mut DeviceBuffer<f32>,
    pub num_sequences: usize,
    pub num_prefill_tokens: usize,
    pub num_decode_tokens: usize,
    pub max_q: usize,
    pub max_k: usize,
    pub block_table_stride: usize,
    pub head_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub causal: bool,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum CudaAttentionMode {
    Decode = 0,
    Prefill = 1,
    Varlen = 2,
    Mixed = 3,
}

pub use aegisllm_base::backend_types::AttentionDType;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct CudaAttentionParamsV1 {
    pub abi_version: u32,
    pub mode: CudaAttentionMode,
    pub flags: u32,
    pub num_sequences: u32,
    pub num_prefill_tokens: u32,
    pub num_decode_tokens: u32,
    pub max_q: u32,
    pub max_k: u32,
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub page_tokens: u32,
    pub block_table_stride: u32,
    pub physical_slots: u32,
    pub softmax_scale: f32,
    pub q_dtype: AttentionDType,
    pub k_dtype: AttentionDType,
    pub v_dtype: AttentionDType,
    pub output_dtype: AttentionDType,
    pub accum_dtype: AttentionDType,
    pub reserved0: u32,
    pub reserved1: u32,
    pub reserved2: u32,
    pub q: *const c_void,
    pub k_cache: *const c_void,
    pub v_cache: *const c_void,
    pub output: *mut c_void,
    pub cu_q: *const u32,
    pub cu_k: *const u32,
    pub context_lens: *const u32,
    pub slot_mapping: *const u32,
    pub block_tables: *const u32,
}

#[allow(dead_code)]
impl CudaAttentionParamsV1 {
    pub const ABI_VERSION: u32 = 1;
    pub const FLAG_CAUSAL: u32 = 1 << 0;
    pub const FLAG_PAGED_KV: u32 = 1 << 1;
    pub const FLAG_GQA: u32 = 1 << 2;
}

/// Host-resident weight bytes for a single tensor. Two variants now:
///
///   * `Mmap` — file-backed safetensors region, the **primary** storage for
///     all host-resident weights. Pages are owned by the kernel page cache,
///     so they don't count against process RSS and are evictable under
///     memory pressure (matches llama.cpp's default loader behavior).
///   * `Pinned` — small process-owned pinned alloc. Used only for repacked
///     bytes generated at load time (native-MXFP4 layout, MatFormer
///     submatrix slices) where the source data doesn't have a file-backed
///     region to point at.
///
/// Both expose bytes via `as_bytes()`; staging feeds the slice to
/// `memcpy_htod`. Pinned source → direct DMA; Mmap → per-slot CPU bounce
/// then DMA (handled by the staging pool).
#[derive(Debug)]
pub(crate) enum HostWeightBytes {
    Pinned(OwnedPinnedBuf),
    Mmap(LoadedHostTensor),
}

impl HostWeightBytes {
    pub fn as_bytes(&self) -> Result<&[u8]> {
        match self {
            Self::Pinned(p) => Ok(p.as_slice()),
            Self::Mmap(t) => Ok(t.as_bytes()),
        }
    }
    pub fn len(&self) -> usize {
        match self {
            Self::Pinned(p) => p.len(),
            Self::Mmap(t) => t.as_bytes().len(),
        }
    }
    /// `true` when bytes live in CUDA-pinned host memory and can be DMA'd
    /// directly without an intermediate CPU memcpy. The staging pool uses this
    /// to skip the bounce-buffer fast-path overhead for pinned sources.
    pub fn is_pinned(&self) -> bool {
        matches!(self, Self::Pinned(_))
    }
}

#[derive(Debug)]
pub(crate) struct HostResidentWeights {
    pub packed: HostWeightBytes,
    pub scales: HostWeightBytes,
    /// Native MXFP4 repacked layout, if available (requires native_mxfp4_repack=true).
    /// When present, inference stages this into VRAM and uses tensor-core kernels.
    pub native_mxfp4: Option<HostResidentMxfp4>,
}

#[derive(Debug)]
pub(super) struct HostResidentMxfp4 {
    pub data: HostWeightBytes,
    pub blocks_per_row: usize,
}

#[derive(Debug)]
pub struct DeviceNvfp4Linear {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub packed_bytes: usize,
    pub scale_bytes: usize,
    pub input_scale: f32,
    pub output_scale: f32,
    pub kernel_family: KernelFamily,
    pub resident_layout: LinearResidentLayout,
    pub residency: TensorResidencyPlan,
    pub(super) packed: CudaSlice<u8>,
    pub(super) scales: CudaSlice<u8>,
    pub(super) native_mxfp4: Option<DeviceMxfp4Linear>,
    pub(super) cutlass_nvfp4: Option<DeviceCutlassNvfp4Linear>,
    /// Non-None for `StagedHostToDevice` layers: weights live in host RAM.
    /// When set, `packed`/`scales` above are 1-byte stubs (no real VRAM).
    pub(super) host_weights: Option<Box<HostResidentWeights>>,
}

impl DeviceNvfp4Linear {
    /// Returns `true` if this linear's weights live in host RAM (StagedHostToDevice
    /// residency). In that case the inference dispatcher routes through the
    /// staging path (which transparently uses cache views or H2D-streams from
    /// `host_weights` depending on whether the weight is in the VRAM cache).
    ///
    /// We key on `residency` rather than `host_weights.is_some()` because the
    /// loader teardown drops `host_weights` for weights that are in the VRAM
    /// cache to free the host arena — those weights are still
    /// `StagedHostToDevice` for dispatch purposes (cache-views replace the
    /// arena copy without changing the kernel choice).
    pub fn is_host_resident(&self) -> bool {
        matches!(self.residency, TensorResidencyPlan::StagedHostToDevice { .. })
    }

    /// Borrow the packed + scales host bytes for a host-resident weight.
    /// Returns `None` for VRAM-resident weights or when `host_weights` was
    /// dropped (e.g. weights now live in the VRAM expert cache). Used by
    /// grouped MoE bulk staging to concatenate active experts' weights
    /// into the bulk VRAM buffers.
    pub fn host_packed_scales_bytes(&self) -> Option<Result<(&[u8], &[u8])>> {
        let host = self.host_weights.as_ref()?;
        Some(match (host.packed.as_bytes(), host.scales.as_bytes()) {
            (Ok(p), Ok(s)) => Ok((p, s)),
            (Err(e), _) | (_, Err(e)) => Err(e),
        })
    }
}

#[derive(Debug)]
pub(super) struct DeviceMxfp4Linear {
    pub bytes: usize,
    pub blocks_per_row: usize,
    pub data: CudaSlice<u8>,
}

#[derive(Debug)]
pub(super) struct DeviceCutlassNvfp4Linear {
    pub layout: CutlassNvfp4LinearLayout,
    pub payload_e2m1: CudaSlice<u8>,
    pub scales_ue4m3: CudaSlice<u8>,
}

/// BF16 matrix stored as a file-backed mmap region for `StagedHostToDevice`
/// residency. Pages live in the kernel page cache (counted as Cached, not
/// RSS) and are evictable under memory pressure — peak host RAM no longer
/// scales with model size. Rows are streamed to a small VRAM scratch
/// buffer for embedding lookup, or the whole matrix is uploaded to a
/// transient VRAM buffer for matvec.
#[derive(Debug)]
pub(super) struct HostBf16Weights {
    /// The underlying `LoadedHostTensor` keeps the shard mmap alive
    /// for the lifetime of this struct; on drop, its destructor hints
    /// the kernel to release the touched range from page cache, which
    /// is what we want at executor teardown (not during steady-state
    /// inference, when the loaded tensor stays alive).
    loaded: LoadedHostTensor,
    /// Element count (NOT byte count): `rows * cols`.
    len_u16: usize,
}

impl HostBf16Weights {
    pub(super) fn from_loaded(loaded: LoadedHostTensor) -> Result<Self> {
        let HostTensorStorage::Mmap { offset, len, .. } = &loaded.storage else {
            return Err(AegisError::InvalidPlan(
                "HostBf16Weights requires mmap-backed storage".into(),
            ));
        };
        if len % 2 != 0 {
            return Err(AegisError::InvalidPlan(format!(
                "BF16 mmap region has odd byte length {len}"
            )));
        }
        // Safetensors aligns tensor data to 8 bytes per spec, so 2-byte
        // alignment of `offset` is guaranteed; bail loudly if a shard
        // ever violates that (the u16 reinterpret below would UB).
        if offset % 2 != 0 {
            return Err(AegisError::InvalidPlan(format!(
                "BF16 mmap offset {offset} is not 2-byte aligned"
            )));
        }
        let len_u16 = len / 2;
        Ok(Self { loaded, len_u16 })
    }

    /// View the bytes as `&[u16]` (BF16 stored as u16). The shard mmap
    /// is page-aligned (≥ 4 KiB) and the safetensors offset within it
    /// is checked to be 2-byte aligned, so the reinterpret is sound.
    pub fn values(&self) -> &[u16] {
        let bytes = self.loaded.as_bytes();
        // SAFETY: alignment validated in `from_loaded`; `bytes.len()` is
        // exactly `self.len_u16 * 2` so the resulting slice covers the
        // same range with the same lifetime as `self.loaded`.
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u16, self.len_u16) }
    }
}

#[derive(Debug)]
pub struct DeviceBf16Matrix {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub residency: TensorResidencyPlan,
    /// Tiny VRAM stub when host-resident; full matrix when VRAM-resident.
    pub(super) values: CudaSlice<u16>,
    /// Set for `StagedHostToDevice` BF16 matrices (e.g. embed_tokens, lm_head when
    /// `model.store=ram`). When present, callers must use the staging-aware paths.
    pub(super) host_values: Option<Box<HostBf16Weights>>,
}

impl DeviceBf16Matrix {
    /// Host-resident BF16 matrices live in pinned host RAM and stage to VRAM per use.
    pub fn is_host_resident(&self) -> bool {
        self.host_values.is_some()
    }
}

/// Standalone FP8 E4M3 linear weight, produced by the load-time
/// `bf16 → fp8` quantizer when the user sets
/// `shared-MLP-quantization = "fp8"` (or `attention-quantization = "fp8"`).
///
/// Layout: `data` is `rows × cols` bytes of E4M3 (NVIDIA convention,
/// NaN=0x7f/0xff, max=448). `row_scales[r]` is the FP32 per-row dequant
/// scale: the original BF16 value at `(r, c)` ≈
/// `fp8_e4m3_bits_to_float(data[r*cols + c]) * row_scales[r]`. Per-row
/// (rather than per-group) trades a small amount of accuracy for
/// simplicity and a tiny scale buffer (`rows * 4` bytes). VRAM-resident
/// only; the load-time quantizer writes both buffers directly to the
/// device.
#[derive(Debug)]
pub struct StandaloneFp8Linear {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub bytes: usize,
    pub(super) data: CudaSlice<u8>,
    pub(super) row_scales: CudaSlice<f32>,
}

impl StandaloneFp8Linear {
    pub(super) fn data_slice(&self) -> &CudaSlice<u8> { &self.data }
    pub(super) fn row_scales_slice(&self) -> &CudaSlice<f32> { &self.row_scales }
}

#[derive(Debug, Clone, Copy)]
pub struct DeviceRopeConfig {
    pub theta: f32,
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: u32,
    /// 0 = full head_dim (standard RoPE); >0 = first N dims get RoPE (Gemma 4 p-RoPE).
    pub partial_dim: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DensePrefillMetadataProof {
    start_position: usize,
    batch: usize,
    context_len: usize,
}

impl DensePrefillMetadataProof {
    pub fn new_identity(
        start_position: usize,
        batch: usize,
        context_size: usize,
        positions: &[u32],
        slot_mapping: &[u32],
        cu_q: &[u32],
        context_lens: &[u32],
    ) -> Result<Self> {
        if batch == 0 {
            return Err(AegisError::InvalidPlan(
                "dense prefill metadata proof requires a non-empty batch".into(),
            ));
        }
        let expected_context_len = start_position.checked_add(batch).ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "dense prefill metadata proof overflows: start={} batch={}",
                start_position, batch
            ))
        })?;
        if batch > u32::MAX as usize
            || expected_context_len > u32::MAX as usize
            || context_size > u32::MAX as usize
        {
            return Err(AegisError::InvalidPlan(format!(
                "dense prefill metadata proof requires u32 metadata: batch={} context_len={} context_size={}",
                batch, expected_context_len, context_size
            )));
        }
        if positions.len() != batch
            || slot_mapping.len() != batch
            || cu_q != [0u32, batch as u32]
            || context_lens != [expected_context_len as u32]
        {
            return Err(AegisError::InvalidPlan(format!(
                "dense prefill metadata proof requires identity metadata: positions={} slots={} cu_q={:?} context_lens={:?} start={} batch={} expected_context_len={}",
                positions.len(),
                slot_mapping.len(),
                cu_q,
                context_lens,
                start_position,
                batch,
                expected_context_len
            )));
        }
        let mut previous_slot = None;
        for idx in 0..batch {
            let expected = start_position.checked_add(idx).ok_or_else(|| {
                AegisError::InvalidPlan(format!(
                    "dense prefill metadata proof position overflow: start={} idx={}",
                    start_position, idx
                ))
            })?;
            if expected > u32::MAX as usize
                || positions[idx] as usize != expected
                || slot_mapping[idx] as usize != expected
            {
                return Err(AegisError::InvalidPlan(format!(
                    "dense prefill metadata proof found non-identity entry: idx={} position={} slot={} expected={}",
                    idx, positions[idx], slot_mapping[idx], expected
                )));
            }
            if previous_slot.is_some_and(|slot| slot >= slot_mapping[idx]) {
                return Err(AegisError::InvalidPlan(format!(
                    "dense prefill metadata proof requires strictly increasing slots: prev={:?} current={}",
                    previous_slot, slot_mapping[idx]
                )));
            }
            previous_slot = Some(slot_mapping[idx]);
        }
        if expected_context_len > context_size {
            return Err(AegisError::InvalidPlan(format!(
                "dense prefill metadata proof exceeds context: start={} batch={} context_len={} context_size={}",
                start_position, batch, expected_context_len, context_size
            )));
        }
        Ok(Self {
            start_position,
            batch,
            context_len: expected_context_len,
        })
    }

    pub fn start_position(self) -> usize {
        self.start_position
    }

    pub fn batch(self) -> usize {
        self.batch
    }

    pub fn context_len(self) -> usize {
        self.context_len
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AttentionDType, CudaAttentionMode, CudaAttentionParamsV1, DensePrefillMetadataProof,
    };

    #[test]
    fn attention_params_are_c_stable_enough_for_cuda_ffi() {
        assert_eq!(CudaAttentionParamsV1::ABI_VERSION, 1);
        assert_eq!(CudaAttentionMode::Decode as u32, 0);
        assert_eq!(CudaAttentionMode::Prefill as u32, 1);
        assert_eq!(CudaAttentionMode::Varlen as u32, 2);
        assert_eq!(CudaAttentionMode::Mixed as u32, 3);
        assert_eq!(AttentionDType::F32 as u32, 0);
        assert_eq!(AttentionDType::F16 as u32, 1);
        assert_eq!(AttentionDType::Bf16 as u32, 2);
        assert_eq!(AttentionDType::Fp8E4M3 as u32, 3);
        assert_eq!(AttentionDType::Fp8E5M2 as u32, 4);
        assert_eq!(AttentionDType::Fp4E2M1 as u32, 5);
        assert_eq!(AttentionDType::Int8 as u32, 6);
        assert_eq!(AttentionDType::Int4 as u32, 7);
        assert_eq!(CudaAttentionParamsV1::FLAG_CAUSAL, 1);
        assert_eq!(CudaAttentionParamsV1::FLAG_PAGED_KV, 2);
        assert_eq!(CudaAttentionParamsV1::FLAG_GQA, 4);
        assert_eq!(
            std::mem::size_of::<CudaAttentionParamsV1>() % std::mem::size_of::<usize>(),
            0
        );
    }

    #[test]
    fn dense_prefill_metadata_proof_accepts_identity_span() {
        let proof = DensePrefillMetadataProof::new_identity(
            5,
            3,
            16,
            &[5, 6, 7],
            &[5, 6, 7],
            &[0, 3],
            &[8],
        )
        .unwrap();
        assert_eq!(proof.start_position(), 5);
        assert_eq!(proof.batch(), 3);
        assert_eq!(proof.context_len(), 8);
    }

    #[test]
    fn dense_prefill_metadata_proof_rejects_non_identity_slot_mapping() {
        assert!(
            DensePrefillMetadataProof::new_identity(
                5,
                3,
                16,
                &[5, 6, 7],
                &[5, 7, 6],
                &[0, 3],
                &[8],
            )
            .is_err()
        );
    }

    #[test]
    fn dense_prefill_metadata_proof_rejects_bad_cu_q() {
        assert!(
            DensePrefillMetadataProof::new_identity(
                5,
                3,
                16,
                &[5, 6, 7],
                &[5, 6, 7],
                &[0, 2],
                &[8],
            )
            .is_err()
        );
    }
}

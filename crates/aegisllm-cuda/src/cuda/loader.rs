use super::host_arena::ArenaHandle;
use super::repack::{
    cached_repack_nvfp4_to_cutlass_e2m1_ue4m3_host, cached_repack_nvfp4_to_mxfp4_host,
    repack_nvfp4_to_cutlass_e2m1_ue4m3_host,
};
use super::owned_pinned::OwnedPinnedBuf;
use super::runtime::{CudaRuntime, map_cuda_err};
use super::types::{
    DeviceBf16Matrix, DeviceBuffer, DeviceCutlassNvfp4Linear, DeviceMxfp4Linear, DeviceNvfp4Linear,
    HostBf16Weights, HostResidentMxfp4, HostResidentWeights, HostWeightBytes,
};
use aegisllm_base::artifact::ModelArtifact;
use aegisllm_base::error::{AegisError, Result};

/// Number of concurrent `pread` workers used by `read_chunked_par`.
/// 8 saturates a Gen4 NVMe (4-5 GB/s) without contention; more workers
/// just add `File::open` overhead per call. Empirically tuned at commit
/// 67b9b0f when 4-way (~3.2 GB/s on a 1.4 GiB BF16 tensor) was bumped
/// to 8-way (~4.4 GB/s) on the user's 5070 Ti host.
const PREAD_CHUNK_COUNT: usize = 8;

/// Allocate a process-owned pinned host buffer and copy `bytes` into it.
/// Used only for small, generated-at-load-time data (native MXFP4 repack
/// output, MatFormer submatrix slices) where the source is a transient
/// `Vec<u8>` with no file-backed mmap to point at.
fn alloc_pinned_from_bytes(
    _runtime: &CudaRuntime,
    bytes: &[u8],
    _label: &'static str,
) -> Result<OwnedPinnedBuf> {
    let mut pinned = OwnedPinnedBuf::new(bytes.len())?;
    pinned.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
    Ok(pinned)
}

/// Parallel chunked `pread` of a contiguous file range into `dst`.
/// Splits into `PREAD_CHUNK_COUNT` sub-ranges, opens that many
/// independent `File` handles, and dispatches concurrent `read_at`
/// calls on rayon workers. Each thread writes a disjoint sub-slice
/// so there's no aliasing.
///
/// Single-thread `read_exact` on a 1.4 GiB BF16 tensor runs at ~440
/// MB/s (NVMe queue depth = 1, page-cache copy + memcpy serialized).
/// 8-way pread saturates the link at ~4.4 GB/s on the user's Gen4
/// drive — about a 10× win on big tensors.
fn read_chunked_par(path: &std::path::Path, file_offset: u64, dst: &mut [u8]) -> Result<()> {
    use rayon::prelude::*;
    use std::os::unix::fs::FileExt;
    use aegisllm_base::tensor::storage::fadvise_dont_need;
    let len = dst.len();
    if len == 0 {
        return Ok(());
    }
    // Tiny tensors: single read is faster than splitting. Threshold
    // chosen so we don't spawn 8 workers for a 4 KiB scalar.
    if len < 1 << 20 {
        let file = std::fs::File::open(path)?;
        file.read_exact_at(dst, file_offset)?;
        // Evict the just-read range from the page cache: the bytes are now
        // in the arena (anonymous RAM), so the file-cache copy is dead
        // weight. Loading ~12 GiB of experts via pread otherwise balloons
        // the page cache to ~9 GiB mid-load (it counts against the cgroup
        // memory limit and inflates `MemoryCurrent`). pread pages are not
        // mmap-mapped, so POSIX_FADV_DONTNEED evicts them immediately.
        fadvise_dont_need(&file, file_offset, len as u64);
        return Ok(());
    }
    let chunk_size = len.div_ceil(PREAD_CHUNK_COUNT);
    let dst_ptr = dst.as_mut_ptr() as usize;
    (0..PREAD_CHUNK_COUNT)
        .into_par_iter()
        .try_for_each(|i| -> Result<()> {
            let chunk_start = i * chunk_size;
            if chunk_start >= len {
                return Ok(());
            }
            let chunk_end = (chunk_start + chunk_size).min(len);
            let chunk_len = chunk_end - chunk_start;
            let file = std::fs::File::open(path)?;
            // SAFETY: chunks are disjoint by construction (i × chunk_size,
            // non-overlapping ranges); `dst_ptr + chunk_start` is in-bounds
            // because `chunk_end ≤ len`.
            let chunk_dst = unsafe {
                std::slice::from_raw_parts_mut(
                    (dst_ptr as *mut u8).add(chunk_start),
                    chunk_len,
                )
            };
            let chunk_file_off = file_offset + chunk_start as u64;
            file.read_exact_at(chunk_dst, chunk_file_off)?;
            Ok(())
        })?;
    // One page-cache evict for the whole tensor range, after every chunk
    // read has completed. The bytes are in the arena (anonymous RAM) now,
    // so the file-cache copy is dead weight; loading ~12 GiB of experts
    // via pread otherwise balloons the page cache to ~9 GiB mid-load
    // (it counts against the cgroup memory limit). One fadvise per tensor
    // — not per chunk — keeps it off the concurrent-read critical path.
    if let Ok(file) = std::fs::File::open(path) {
        fadvise_dont_need(&file, file_offset, len as u64);
    }
    Ok(())
}

/// Read a tensor's raw bytes from disk into the pinned host arena via
/// 8-way parallel `pread`. Returns the byte offset inside the arena
/// where bytes were placed; combined with the byte length this
/// uniquely identifies the slice for downstream `HostWeightBytes::Arena`.
fn read_tensor_into_arena(
    arena: &ArenaHandle,
    tensor: &TensorInfo,
) -> Result<HostWeightBytes> {
    let len = tensor.data_len_bytes() as usize;
    // Reserve the arena slot first (atomic bump). The actual read into
    // the slot can then run in parallel with other tensors' reads
    // because slots are disjoint by construction.
    let offset = arena.reserve(len)?;
    // SAFETY: `arena.reserve` exclusively claimed [offset, offset+len);
    // we have unique write access until we release it.
    let dst = unsafe { arena.slice_mut(offset, len) };
    read_chunked_par(&tensor.shard_path, tensor.file_offsets.0, dst)?;
    Ok(HostWeightBytes::Arena {
        arena: arena.clone(),
        offset,
        len,
    })
}
use aegisllm_base::graph::{GraphRegion, TensorRole};
use aegisllm_base::planning::cuda_nvfp4_kernel_family_for_layout;
use aegisllm_base::planning::placement::{ComputePlacement, RegionPlacement, StoragePlacement};
use aegisllm_base::planning::runtime::KernelFamily;
use aegisllm_base::tensor::layout::LinearResidentLayout;
use aegisllm_base::tensor::quant::{Nvfp4LinearSpec, QK_NVFP4, QK_NVFP4_SUB};
use aegisllm_base::tensor::storage::{
    LoadedHostTensor, NestedParamSlice, TensorResidencyPlan, TensorStorageLoader,
};
use aegisllm_base::tensor::{TensorDType, TensorInfo};

/// Callback invoked by the loader to report fine-grained sub-step progress
/// (e.g. "layer 5 expert 64/128") between coarse `step` advances. The executor
/// wires this to its TTY progress indicator; cross-crate callers that don't
/// care about progress just leave it `None`.
pub type LoadStatusSink = std::sync::Arc<dyn Fn(&str) + Send + Sync>;

pub struct CudaWeightLoader<'a> {
    runtime: &'a CudaRuntime,
    /// Pinned-host arena for host-resident NVFP4 + BF16 weights. When
    /// set, host-resident loaders read tensor bytes directly into this
    /// arena (single big `cuMemHostRegister`'d allocation, sub-allocated
    /// by atomic bump). When `None`, host-resident paths cannot be
    /// taken — the executor must always build the loader via
    /// `weight_loader_with_arena(...)` for any config that has
    /// host-resident weights.
    arena: Option<ArenaHandle>,
    /// Reusable pinned-host bounce buffer for VRAM-resident BF16
    /// uploads. Replaces the prior `loader.load_for_store(_, Mmap)` →
    /// `clone_htod` path that mmap'd the shard and filled the kernel
    /// page cache by the tensor size on every read — for Gemma-4-26B
    /// with embed + lm_head VRAM-resident that's ~3 GiB of page cache
    /// retained until end of load, on top of the growing 12-14 GiB
    /// arena. Combined that pushed peak host RAM near the system
    /// limit and triggered OOM during the layer loop on 32 GiB hosts.
    /// With this buffer, each VRAM-resident BF16 read goes directly
    /// from disk into pinned memory (no page-cache fill) and DMAs to
    /// device with a single pinned→device memcpy. Sized to the
    /// largest tensor seen so far, reused across loads.
    bounce: std::cell::RefCell<Option<OwnedPinnedBuf>>,
    /// Optional callback for fine-grained sub-step progress. Heavy inner
    /// loops (MoE experts, big BF16 uploads) call `report_status(...)` so
    /// the user's progress indicator can refresh between coarse `step`
    /// advances.
    status_sink: Option<LoadStatusSink>,
}

impl CudaRuntime {
    pub fn weight_loader(&self) -> CudaWeightLoader<'_> {
        CudaWeightLoader {
            runtime: self,
            arena: None,
            bounce: std::cell::RefCell::new(None),
            status_sink: None,
        }
    }

    /// Create a weight loader bound to a pre-allocated pinned-host arena.
    /// Required for any config with host-resident weights; non-host-resident
    /// loads (VRAM-resident BF16 / NVFP4) ignore the arena.
    pub fn weight_loader_with_arena(&self, arena: ArenaHandle) -> CudaWeightLoader<'_> {
        CudaWeightLoader {
            runtime: self,
            arena: Some(arena),
            bounce: std::cell::RefCell::new(None),
            status_sink: None,
        }
    }
}

impl CudaWeightLoader<'_> {
    pub fn device_index(&self) -> usize {
        self.runtime.device_index()
    }

    /// Attach a sub-step status callback (overwrites any prior sink). The
    /// loader invokes it from heavy inner loops via `report_status`.
    pub fn with_status_sink(mut self, sink: LoadStatusSink) -> Self {
        self.status_sink = Some(sink);
        self
    }

    /// Emit a sub-step status line via the attached sink, if any. Cheap
    /// no-op when no sink is set, so call-site clutter is the only cost
    /// for callers that don't wire one in.
    pub fn report_status(&self, label: &str) {
        if let Some(sink) = self.status_sink.as_ref() {
            sink(label);
        }
    }

    /// Borrow the underlying runtime for callers that need direct access to
    /// allocator / upload primitives during loading. Used by the executor's
    /// loader to populate `router_per_expert_scale_device` and similar
    /// accompanying device-resident metadata.
    pub fn runtime(&self) -> &CudaRuntime {
        self.runtime
    }

    /// Borrow the loader's pinned-host arena. Returns `None` if the
    /// loader was built bare via `weight_loader()` (no host-resident
    /// path supported). The executor inspects this to know whether
    /// `pin_now()` is needed at end of load.
    pub fn arena(&self) -> Option<&ArenaHandle> {
        self.arena.as_ref()
    }

    /// Drop the VRAM-upload bounce buffer. The bounce is high-water-
    /// mark sized to the largest VRAM-resident BF16 tensor (typically
    /// lm_head ≈ 1.4 GiB on Gemma-4-26B). After all VRAM-resident
    /// BF16 weights have been loaded (which happens BEFORE the layer
    /// loop), the bounce is dead weight that competes with the
    /// growing arena for host RAM. Callers must drop it explicitly
    /// to keep the load-time peak inside the arena's footprint.
    pub fn release_bounce(&self) {
        *self.bounce.borrow_mut() = None;
    }

    /// Ensure the loader's pinned bounce is at least `len` bytes,
    /// then read `tensor`'s bytes directly from disk into it. Uses
    /// `pread`-equivalent reads (no `seek` then `read`, just
    /// `read_exact_at`) to bypass the kernel's mmap page cache —
    /// for VRAM-resident BF16 (embed/lm_head when configured to
    /// stream once at load) this is the difference between holding
    /// ~3 GiB of page-cache pages alongside the growing 14 GiB
    /// arena (→ OOM on 32 GiB hosts) and bounded ~tensor-size
    /// transient anon pages that get reused across loads.
    fn read_tensor_into_bounce(&self, tensor: &TensorInfo) -> Result<()> {
        let len = tensor.data_len_bytes() as usize;
        let need_realloc = self
            .bounce
            .borrow()
            .as_ref()
            .map(|b| b.len() < len)
            .unwrap_or(true);
        if need_realloc {
            // Drop the old bounce first so peak memory is just the
            // new size, not old + new during construction.
            *self.bounce.borrow_mut() = None;
            *self.bounce.borrow_mut() = Some(OwnedPinnedBuf::new(len)?);
        }
        let mut bounce_ref = self.bounce.borrow_mut();
        let bounce = bounce_ref.as_mut().expect("bounce just ensured to exist");
        let bytes = bounce.as_mut_slice();
        // 8-way parallel pread saturates Gen4 NVMe (~4.4 GB/s on a
        // 1.4 GiB BF16 tensor) instead of the ~440 MB/s single-thread
        // ceiling. Big tensors (lm_head, embed) used to dominate the
        // load timeline; with chunked pread they finish in <0.5s.
        read_chunked_par(&tensor.shard_path, tensor.file_offsets.0, &mut bytes[..len])?;
        Ok(())
    }

    /// Parallel-prefetch many host-resident NVFP4 tensor pairs
    /// (`{prefix}.weight`, `{prefix}.weight_scale`) into the loader's
    /// arena. Each pair runs on a rayon worker that opens its shard,
    /// `pread`s into the arena's atomic-bump-allocated slot, and
    /// returns the host bytes for the consumer to attach to its
    /// `DeviceNvfp4Linear`. Per-tensor reads are themselves chunked
    /// (8-way) so even a single big tensor saturates the NVMe; the
    /// outer rayon parallelism hides per-call open/seek overhead
    /// across the ~384 expert tensors of one MoE layer.
    ///
    /// Returns `(packed, scales)` pairs in the same order as `prefixes`,
    /// so the caller can match them with serial CUDA-side metadata
    /// construction.
    pub fn prefetch_host_nvfp4_pairs_par<'a>(
        &self,
        artifact: &'a ModelArtifact,
        prefixes: &[String],
    ) -> Result<Vec<(HostWeightBytes, HostWeightBytes)>> {
        use rayon::prelude::*;
        let arena = self.arena.as_ref().ok_or_else(|| {
            AegisError::InvalidPlan(
                "prefetch_host_nvfp4_pairs_par requires loader built with weight_loader_with_arena".into(),
            )
        })?;
        prefixes
            .par_iter()
            .map(|prefix| -> Result<(HostWeightBytes, HostWeightBytes)> {
                let weight = artifact
                    .tensors
                    .get(&format!("{prefix}.weight"))
                    .ok_or_else(|| {
                        AegisError::InvalidPlan(format!("missing `{prefix}.weight`"))
                    })?;
                let scales = artifact
                    .tensors
                    .get(&format!("{prefix}.weight_scale"))
                    .ok_or_else(|| {
                        AegisError::InvalidPlan(format!("missing `{prefix}.weight_scale`"))
                    })?;
                let packed = read_tensor_into_arena(arena, weight)?;
                let s = read_tensor_into_arena(arena, scales)?;
                Ok((packed, s))
            })
            .collect::<Result<Vec<_>>>()
    }

    /// Build a `DeviceNvfp4Linear` for a host-resident layer using
    /// bytes already prefetched into the arena. This is the finalize
    /// half of the parallel-prefetch flow: file I/O ran on rayon
    /// workers via `prefetch_host_nvfp4_pairs_par`, and now the main
    /// thread does the cheap CUDA-side stub allocs and metadata
    /// construction (which must be serial — single CUDA stream).
    /// `loader` is consulted only for the optional small scalar
    /// metadata (`weight_scale_2`, `input_scale`).
    pub fn finalize_host_nvfp4_with_prefetched(
        &self,
        artifact: &ModelArtifact,
        prefix: &str,
        residency: TensorResidencyPlan,
        store: StoragePlacement,
        packed_bytes: HostWeightBytes,
        scales_bytes: HostWeightBytes,
        loader: &mut TensorStorageLoader,
    ) -> Result<DeviceNvfp4Linear> {
        let kernel_family = cuda_nvfp4_kernel_family_for_layout(
            prefix,
            aegisllm_base::tensor::layout::LinearResidentLayout::PackedSource,
        )?;
        let weight = artifact
            .tensors
            .get(&format!("{prefix}.weight"))
            .ok_or_else(|| AegisError::InvalidPlan(format!("missing `{prefix}.weight`")))?;
        let scales = artifact
            .tensors
            .get(&format!("{prefix}.weight_scale"))
            .ok_or_else(|| AegisError::InvalidPlan(format!("missing `{prefix}.weight_scale`")))?;
        let output_scale = artifact
            .tensors
            .get(&format!("{prefix}.weight_scale_2"))
            .map(|t| read_scalar_f32_with_loader(loader, t, store))
            .transpose()?
            .unwrap_or(1.0);
        let input_scale = artifact
            .tensors
            .get(&format!("{prefix}.input_scale"))
            .map(|t| read_scalar_f32_with_loader(loader, t, store))
            .transpose()?
            .unwrap_or(1.0);
        let spec = Nvfp4LinearSpec::from_tensors(prefix, weight, scales, input_scale, output_scale)?;
        let stub_packed = self
            .runtime
            .stream
            .clone_htod(&[0u8])
            .map_err(map_cuda_err("htod nvfp4 host-resident stub packed"))?;
        let stub_scales = self
            .runtime
            .stream
            .clone_htod(&[0u8])
            .map_err(map_cuda_err("htod nvfp4 host-resident stub scales"))?;
        Ok(DeviceNvfp4Linear {
            name: spec.name,
            rows: spec.rows,
            cols: spec.cols,
            packed_bytes: spec.packed_bytes,
            scale_bytes: spec.scale_bytes,
            input_scale: spec.input_scale,
            output_scale: spec.output_scale,
            kernel_family,
            resident_layout: aegisllm_base::tensor::layout::LinearResidentLayout::PackedSource,
            residency,
            packed: stub_packed,
            scales: stub_scales,
            native_mxfp4: None,
            cutlass_nvfp4: None,
            host_weights: Some(Box::new(HostResidentWeights {
                packed: packed_bytes,
                scales: scales_bytes,
                native_mxfp4: None,
            })),
        })
    }

    pub fn load_dense_vector_with_store(
        &self,
        tensor: &TensorInfo,
        store: StoragePlacement,
        loader: &mut TensorStorageLoader,
    ) -> Result<DeviceBuffer<f32>> {
        if tensor.shape.len() != 1 {
            return Err(AegisError::InvalidPlan(format!(
                "`{}` must be a dense vector",
                tensor.name
            )));
        }
        let loaded = loader.load_for_store(tensor, store)?;
        let bytes = loaded.as_bytes();
        let values = match tensor.dtype {
            TensorDType::BF16 => bytes
                .chunks_exact(2)
                .map(|chunk| {
                    f32::from_bits((u16::from_le_bytes([chunk[0], chunk[1]]) as u32) << 16)
                })
                .collect::<Vec<_>>(),
            TensorDType::F32 => bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect::<Vec<_>>(),
            other => {
                return Err(AegisError::InvalidPlan(format!(
                    "`{}` must be BF16 or F32 vector, got {:?}",
                    tensor.name, other
                )));
            }
        };
        self.runtime.upload_f32(&values)
    }

    pub fn load_bf16_matrix_with_store(
        &self,
        tensor: &TensorInfo,
        _store: StoragePlacement,
        residency: TensorResidencyPlan,
        loader: &mut TensorStorageLoader,
    ) -> Result<DeviceBf16Matrix> {
        if tensor.dtype != TensorDType::BF16 || tensor.shape.len() < 2 {
            return Err(AegisError::InvalidPlan(format!(
                "`{}` must be a BF16 matrix (>= 2-D; got dtype={:?} shape={:?})",
                tensor.name, tensor.dtype, tensor.shape,
            )));
        }
        // For N-D tensors (e.g. vision position_embedding_table [2, 10240, 1152]),
        // collapse all leading dims into `rows`; the last dim stays `cols`.
        let last_dim = tensor.shape[tensor.shape.len() - 1];
        let row_product: usize = tensor.shape[..tensor.shape.len() - 1].iter().product();
        let effective_rows = row_product;
        let effective_cols = last_dim;
        let is_host_resident = matches!(residency, TensorResidencyPlan::StagedHostToDevice { .. });

        if is_host_resident {
            // Read tensor bytes from disk straight into the pinned host
            // arena via 8-way parallel `pread`. Big BF16 tensors
            // (embed ≈ 1.4 GiB) saturate Gen4 NVMe at ~4.4 GB/s
            // instead of the ~440 MB/s single-thread ceiling. After
            // `pin_now()` at end-of-load the arena is
            // `cuMemHostRegister`'d, so DMA from any sub-slice takes
            // the direct-pinned path.
            let arena = self.arena.as_ref().ok_or_else(|| {
                AegisError::InvalidPlan(format!(
                    "host-resident BF16 `{}` requires loader built with weight_loader_with_arena(...)",
                    tensor.name
                ))
            })?;
            let len = tensor.data_len_bytes() as usize;
            let offset = arena.reserve(len)?;
            // SAFETY: `reserve` exclusively claimed [offset, offset+len);
            // `read_chunked_par` writes disjoint sub-slices.
            let dst = unsafe { arena.slice_mut(offset, len) };
            read_chunked_par(&tensor.shard_path, tensor.file_offsets.0, dst)?;
            let stub = self
                .runtime
                .stream
                .clone_htod(&[0u16])
                .map_err(map_cuda_err("htod bf16 host-resident stub"))?;
            let host_weights = HostBf16Weights::from_arena(arena.clone(), offset, len)?;
            return Ok(DeviceBf16Matrix {
                name: tensor.name.clone(),
                rows: effective_rows,
                cols: effective_cols,
                residency,
                values: stub,
                host_values: Some(Box::new(host_weights)),
            });
        }

        // VRAM-resident BF16: read directly from disk into the
        // loader's reusable pinned bounce, then DMA bounce → VRAM.
        // Bypasses the mmap page-cache fill that would otherwise
        // hold ~tensor-size of shard pages alongside the growing
        // arena and trigger OOM on 32 GiB hosts. `loader` arg is
        // unused on this branch but stays in the signature for the
        // host-resident branch above.
        let _ = loader;
        let len_bytes = tensor.data_len_bytes() as usize;
        if len_bytes % 2 != 0 {
            return Err(AegisError::InvalidPlan(format!(
                "BF16 tensor `{}` has odd byte length {}", tensor.name, len_bytes
            )));
        }
        let len_u16 = len_bytes / 2;
        self.read_tensor_into_bounce(tensor)?;
        let mut buffer = self.runtime.alloc_u16(len_u16)?;
        let bounce_ref = self.bounce.borrow();
        let bounce = bounce_ref.as_ref().expect("bounce populated above");
        // SAFETY: bounce is a process-owned mmap (page-aligned, so
        // 2-byte alignment is satisfied); even-byte-length checked.
        // Slice to exactly `len_u16` because the bounce may be
        // larger when reused across smaller tensors.
        let src_u16: &[u16] = &bounce.as_u16_slice()[..len_u16];
        self.runtime
            .stream
            .memcpy_htod(src_u16, &mut buffer.slice)
            .map_err(map_cuda_err("htod bf16 matrix from bounce"))?;
        // `memcpy_htod` from a pinned source is async and returns
        // before the DMA finishes; if the next loader call clobbers
        // the bounce mid-transfer it'd corrupt the upload silently.
        // Synchronise here so the bounce is safe to reuse.
        self.runtime
            .synchronize()
            .map_err(|e| AegisError::Unsupported(format!("sync after bf16 bounce htod: {e}")))?;
        Ok(DeviceBf16Matrix {
            name: tensor.name.clone(),
            rows: effective_rows,
            cols: effective_cols,
            residency,
            values: buffer.slice,
            host_values: None,
        })
    }

    /// Fuse a pair of already-loaded VRAM-resident BF16 gate/up matrices into
    /// one row-stacked `[2*rows, cols]` matrix and drop the originals' VRAM.
    ///
    /// Both inputs must be VRAM-resident and have identical `(rows, cols)` —
    /// the standard Gemma 4 / Llama gate/up convention. Returns:
    ///   * the fused matrix (rows = `2 * orig_rows`, cols unchanged), VRAM-
    ///     resident, ready for a single cuBLASLt BF16 GEMM that produces
    ///     `[batch, 2*orig_rows]` row-major. Per token the first `orig_rows`
    ///     floats are the gate logits and the next `orig_rows` are the up
    ///     logits — exactly the layout expected by `geglu_tanh_strided_device`.
    ///   * a pair of `DeviceBf16Matrix` stubs that retain the original
    ///     `(name, rows, cols, residency)` metadata but hold only a 1-element
    ///     placeholder VRAM slice. Returned so downstream code that still calls
    ///     `.rows()` / `.cols()` / `.name()` keeps working without further
    ///     refactors. **Stubs must not be matmul'd against** — `cublaslt_bf16_enabled_for`
    ///     still returns `true` (their `host_values` is `None`) but their
    ///     1-element `values` slice would corrupt any GEMM. Callers route the
    ///     matmul through the fused matrix instead.
    ///
    /// The fusion path uses a 1-shot device alloc + two D2D `memcpy_dtod`
    /// calls. Original gate/up VRAM is freed on return (the two stubs hold tiny
    /// `clone_htod(&[0u16])` slices), saving `2 * rows * cols * 2` bytes per
    /// fused layer versus keeping both originals alive.
    pub fn fuse_bf16_gate_up(
        &self,
        mut gate: DeviceBf16Matrix,
        mut up: DeviceBf16Matrix,
    ) -> Result<(DeviceBf16Matrix, DeviceBf16Matrix, DeviceBf16Matrix)> {
        if gate.is_host_resident() || up.is_host_resident() {
            return Err(AegisError::InvalidPlan(format!(
                "fuse_bf16_gate_up requires VRAM-resident inputs; got gate.host_resident={} up.host_resident={}",
                gate.is_host_resident(),
                up.is_host_resident()
            )));
        }
        if gate.rows != up.rows || gate.cols != up.cols {
            return Err(AegisError::InvalidPlan(format!(
                "fuse_bf16_gate_up shape mismatch: gate=({}, {}) up=({}, {})",
                gate.rows, gate.cols, up.rows, up.cols
            )));
        }
        let rows = gate.rows;
        let cols = gate.cols;
        let per_mat = rows
            .checked_mul(cols)
            .ok_or_else(|| AegisError::InvalidPlan("fuse gate/up elem count overflow".into()))?;
        let fused_len = per_mat
            .checked_mul(2)
            .ok_or_else(|| AegisError::InvalidPlan("fuse gate/up doubled count overflow".into()))?;
        if gate.values.len() < per_mat || up.values.len() < per_mat {
            return Err(AegisError::InvalidPlan(format!(
                "fuse_bf16_gate_up: input values too small: gate={} up={} need {}",
                gate.values.len(), up.values.len(), per_mat
            )));
        }

        // Allocate the row-stacked destination in VRAM and D2D-copy each
        // input matrix into its sub-range. cuBLASLt sees a single
        // `[2*rows, cols]` row-major weight, equivalent to stacking gate
        // on top of up (so the GEMM output per token is
        // `[gate_logits, up_logits]` row-major — the layout consumed by
        // `aegis_geglu_tanh_strided`).
        let mut fused_slice = unsafe { self.runtime.stream.alloc::<u16>(fused_len) }
            .map_err(map_cuda_err("alloc fused gate/up bf16"))?;
        {
            let src_view = gate.values.slice(0..per_mat);
            let mut dst_view = fused_slice.slice_mut(0..per_mat);
            self.runtime
                .stream
                .memcpy_dtod(&src_view, &mut dst_view)
                .map_err(map_cuda_err("d2d fused gate copy"))?;
        }
        {
            let src_view = up.values.slice(0..per_mat);
            let mut dst_view = fused_slice.slice_mut(per_mat..fused_len);
            self.runtime
                .stream
                .memcpy_dtod(&src_view, &mut dst_view)
                .map_err(map_cuda_err("d2d fused up copy"))?;
        }
        // Synchronise so the source buffers can safely be dropped below.
        self.runtime
            .synchronize()
            .map_err(|e| AegisError::Unsupported(format!("sync after fuse gate/up d2d: {e}")))?;

        let fused_name = format!(
            "{}|fused|{}",
            gate.name.trim_end_matches(".weight"),
            up.name.trim_end_matches(".weight")
        );
        let fused = DeviceBf16Matrix {
            name: fused_name,
            rows: 2 * rows,
            cols,
            residency: gate.residency.clone(),
            values: fused_slice,
            host_values: None,
        };

        // Replace each original's `values` with a tiny 1-element stub. This
        // drops the full-size VRAM allocation (Rust's `Drop` on `CudaSlice`
        // returns the bytes to the pool) while keeping `(name, rows, cols,
        // residency)` intact for downstream introspection.
        let gate_stub_slice = self
            .runtime
            .stream
            .clone_htod(&[0u16])
            .map_err(map_cuda_err("htod gate stub"))?;
        let up_stub_slice = self
            .runtime
            .stream
            .clone_htod(&[0u16])
            .map_err(map_cuda_err("htod up stub"))?;
        gate.values = gate_stub_slice;
        up.values = up_stub_slice;

        Ok((fused, gate, up))
    }

    /// Load a BF16 matrix from `tensor` and quantize it to FP8 E4M3 with
    /// per-row FP32 scales at load time. Used by
    /// `shared-MLP-quantization = "fp8"` (and `attention-quantization = "fp8"`
    /// once that path lands).
    ///
    /// Path: stage BF16 directly into a temporary VRAM buffer, then run the
    /// `aegis_quantize_bf16_to_fp8_per_row` kernel which computes per-row
    /// absmax → scale = amax / 448 → encode each element as E4M3. The
    /// transient BF16 buffer is dropped on return; only FP8 + scales
    /// remain (~2× smaller than BF16). No host-side quantizer arithmetic;
    /// reuses the device-side `float_to_fp8_e4m3_bits` helper.
    pub fn load_bf16_as_fp8_linear(
        &self,
        tensor: &TensorInfo,
        loader: &mut TensorStorageLoader,
    ) -> Result<crate::cuda::StandaloneFp8Linear> {
        if tensor.dtype != TensorDType::BF16 || tensor.shape.len() != 2 {
            return Err(AegisError::InvalidPlan(format!(
                "`{}` must be a 2D BF16 matrix to be quantized to FP8",
                tensor.name
            )));
        }
        let rows = tensor.shape[0];
        let cols = tensor.shape[1];
        let total = rows.checked_mul(cols).ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "`{}` BF16→FP8 element count overflow: rows={} cols={}",
                tensor.name, rows, cols
            ))
        })?;

        // Stage BF16 into a transient VRAM buffer.
        let loaded = loader.load_for_store(tensor, StoragePlacement::Ram)?;
        let bf16_host: Vec<u16> = loaded
            .as_bytes()
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();
        if bf16_host.len() != total {
            return Err(AegisError::InvalidPlan(format!(
                "`{}` BF16→FP8: bf16 source size mismatch: got {} u16, expected {}",
                tensor.name, bf16_host.len(), total
            )));
        }
        let bf16_dev_slice = self
            .runtime
            .stream
            .clone_htod(&bf16_host)
            .map_err(map_cuda_err("htod bf16 transient for fp8 quantize"))?;
        let bf16_dev = DeviceBuffer { slice: bf16_dev_slice };

        // Allocate FP8 + per-row scales output buffers.
        let mut fp8_dev = self.runtime.alloc_u8(total)?;
        let mut row_scales_dev = self.runtime.alloc_f32(rows)?;

        // Run the quantize kernel. After this returns, `bf16_dev` is no
        // longer needed and is dropped at end of scope.
        self.runtime
            .quantize_bf16_to_fp8_per_row_device(&bf16_dev, rows, cols, &mut fp8_dev, &mut row_scales_dev)?;

        Ok(crate::cuda::StandaloneFp8Linear {
            name: tensor.name.clone(),
            rows,
            cols,
            bytes: total,
            data: fp8_dev.slice,
            row_scales: row_scales_dev.slice,
        })
    }

    pub fn load_nvfp4_linear(
        &self,
        artifact: &ModelArtifact,
        prefix: &str,
    ) -> Result<DeviceNvfp4Linear> {
        let mut loader = TensorStorageLoader::new();
        self.load_nvfp4_linear_with_store(
            artifact,
            prefix,
            StoragePlacement::Vram {
                device: self.runtime.device_index(),
            },
            TensorResidencyPlan::VramResident {
                device: self.runtime.device_index(),
            },
            &mut loader,
        )
    }

    pub fn load_nvfp4_linear_with_store(
        &self,
        artifact: &ModelArtifact,
        prefix: &str,
        store: StoragePlacement,
        residency: TensorResidencyPlan,
        loader: &mut TensorStorageLoader,
    ) -> Result<DeviceNvfp4Linear> {
        self.load_nvfp4_linear_with_layout(
            artifact,
            prefix,
            store,
            residency,
            LinearResidentLayout::NativeTensorCore,
            loader,
        )
    }

    pub fn load_nvfp4_linear_with_layout(
        &self,
        artifact: &ModelArtifact,
        prefix: &str,
        store: StoragePlacement,
        residency: TensorResidencyPlan,
        resident_layout: LinearResidentLayout,
        loader: &mut TensorStorageLoader,
    ) -> Result<DeviceNvfp4Linear> {
        let kernel_family = cuda_nvfp4_kernel_family_for_layout(prefix, resident_layout)?;
        let weight = artifact
            .tensors
            .get(&format!("{prefix}.weight"))
            .ok_or_else(|| AegisError::InvalidPlan(format!("missing `{prefix}.weight`")))?;
        let scales = artifact
            .tensors
            .get(&format!("{prefix}.weight_scale"))
            .ok_or_else(|| AegisError::InvalidPlan(format!("missing `{prefix}.weight_scale`")))?;
        let output_scale = artifact
            .tensors
            .get(&format!("{prefix}.weight_scale_2"))
            .map(|tensor| read_scalar_f32_with_loader(loader, tensor, store))
            .transpose()?
            .unwrap_or(1.0);
        let input_scale = artifact
            .tensors
            .get(&format!("{prefix}.input_scale"))
            .map(|tensor| read_scalar_f32_with_loader(loader, tensor, store))
            .transpose()?
            .unwrap_or(1.0);
        let spec =
            Nvfp4LinearSpec::from_tensors(prefix, weight, scales, input_scale, output_scale)?;
        let is_host_resident = matches!(residency, TensorResidencyPlan::StagedHostToDevice { .. });

        // We need the mmap'd bytes up-front whenever the call site has to
        // read them on this thread: VRAM upload (clone_htod below), or
        // load-time repacking into native MXFP4 / cutlass NVFP4. For the
        // plain host-resident NVFP4 path the staging pool reads from the
        // shard mmap on demand at inference time, so we don't fault the
        // pages here.
        let needs_mmap = !is_host_resident
            || self.should_repack_native_mxfp4(prefix, kernel_family)
            || self.should_repack_cutlass_nvfp4(prefix, kernel_family, resident_layout);
        let (packed_host, scales_host) = if needs_mmap {
            (
                Some(loader.load_for_store(weight, store)?),
                Some(loader.load_for_store(scales, store)?),
            )
        } else {
            (None, None)
        };

        let native_mxfp4 = if !is_host_resident && self.should_repack_native_mxfp4(prefix, kernel_family) {
            if spec.cols % 64 != 0 {
                return Err(AegisError::InvalidPlan(format!(
                    "native MXFP4 tensor-core layout for `{}` requires cols divisible by 64, got {}",
                    spec.name, spec.cols
                )));
            }
            let repacked = cached_repack_nvfp4_to_mxfp4_host(
                &artifact.root,
                &spec,
                weight,
                scales,
                packed_host.as_ref().unwrap().as_bytes(),
                scales_host.as_ref().unwrap().as_bytes(),
            )?;
            Some(DeviceMxfp4Linear {
                bytes: repacked.len(),
                blocks_per_row: spec.cols / 32,
                data: self
                    .runtime
                    .stream
                    .clone_htod(&repacked)
                    .map_err(map_cuda_err("htod native mxfp4 weights"))?,
            })
        } else {
            None
        };
        let cutlass_nvfp4 =
            if !is_host_resident && self.should_repack_cutlass_nvfp4(prefix, kernel_family, resident_layout) {
                let repacked = cached_repack_nvfp4_to_cutlass_e2m1_ue4m3_host(
                    &artifact.root,
                    &spec,
                    weight,
                    scales,
                    packed_host.as_ref().unwrap().as_bytes(),
                    scales_host.as_ref().unwrap().as_bytes(),
                )?;
                Some(DeviceCutlassNvfp4Linear {
                    layout: repacked.layout,
                    payload_e2m1: self
                        .runtime
                        .stream
                        .clone_htod(&repacked.payload_e2m1)
                        .map_err(map_cuda_err("htod cutlass nvfp4 payload"))?,
                    scales_ue4m3: self
                        .runtime
                        .stream
                        .clone_htod(&repacked.scales_ue4m3)
                        .map_err(map_cuda_err("htod cutlass nvfp4 scales"))?,
                })
            } else {
                None
            };

        if is_host_resident {
            // Weights stay in CUDA-pinned host RAM; tiny VRAM stubs keep the type system intact.
            // Weights are read directly from the safetensors file into pinned memory so no
            // intermediate mmap/pageable copy is created in the kernel page cache.
            let host_native_mxfp4 =
                if self.should_repack_native_mxfp4(prefix, kernel_family) {
                    if spec.cols % 64 != 0 {
                        return Err(AegisError::InvalidPlan(format!(
                            "native MXFP4 tensor-core layout for `{}` requires cols divisible by 64, got {}",
                            spec.name, spec.cols
                        )));
                    }
                    let repacked = cached_repack_nvfp4_to_mxfp4_host(
                        &artifact.root,
                        &spec,
                        weight,
                        scales,
                        packed_host.as_ref().unwrap().as_bytes(),
                        scales_host.as_ref().unwrap().as_bytes(),
                    )?;
                    let pinned = alloc_pinned_from_bytes(
                        self.runtime,
                        &repacked,
                        "alloc pinned host native mxfp4",
                    )?;
                    Some(HostResidentMxfp4 {
                        blocks_per_row: spec.cols / 32,
                        data: HostWeightBytes::Pinned(pinned),
                    })
                } else {
                    None
                };
            // Host-resident weights live in the pinned-host arena: one
            // big `cuMemHostRegister`'d allocation, sub-allocated per
            // tensor by atomic bump. Per-token inference does direct
            // DMA from the arena — no CPU memcpy through a staging
            // bounce. RAM cost: ~12-14 GiB anonymous-mapped pinned
            // pages for Gemma-4-26B.
            let arena = self.arena.as_ref().ok_or_else(|| {
                AegisError::InvalidPlan(format!(
                    "host-resident NVFP4 `{}` requires loader built with weight_loader_with_arena(...)",
                    spec.name
                ))
            })?;
            let mmap_packed = read_tensor_into_arena(arena, weight)?;
            let mmap_scales = read_tensor_into_arena(arena, scales)?;
            let _ = store;
            let stub_packed = self
                .runtime
                .stream
                .clone_htod(&[0u8])
                .map_err(map_cuda_err("htod nvfp4 host-resident stub packed"))?;
            let stub_scales = self
                .runtime
                .stream
                .clone_htod(&[0u8])
                .map_err(map_cuda_err("htod nvfp4 host-resident stub scales"))?;
            return Ok(DeviceNvfp4Linear {
                name: spec.name,
                rows: spec.rows,
                cols: spec.cols,
                packed_bytes: spec.packed_bytes,
                scale_bytes: spec.scale_bytes,
                input_scale: spec.input_scale,
                output_scale: spec.output_scale,
                kernel_family,
                resident_layout: aegisllm_base::tensor::layout::LinearResidentLayout::PackedSource,
                residency,
                packed: stub_packed,
                scales: stub_scales,
                native_mxfp4: None,
                cutlass_nvfp4: None,
                host_weights: Some(Box::new(HostResidentWeights {
                    packed: mmap_packed,
                    scales: mmap_scales,
                    native_mxfp4: host_native_mxfp4,
                })),
            });
        }

        Ok(DeviceNvfp4Linear {
            name: spec.name,
            rows: spec.rows,
            cols: spec.cols,
            packed_bytes: spec.packed_bytes,
            scale_bytes: spec.scale_bytes,
            input_scale: spec.input_scale,
            output_scale: spec.output_scale,
            kernel_family,
            resident_layout,
            residency,
            packed: self
                .runtime
                .stream
                .clone_htod(packed_host.as_ref().unwrap().as_bytes())
                .map_err(map_cuda_err("htod nvfp4 packed weights"))?,
            scales: self
                .runtime
                .stream
                .clone_htod(scales_host.as_ref().unwrap().as_bytes())
                .map_err(map_cuda_err("htod nvfp4 scales"))?,
            native_mxfp4,
            cutlass_nvfp4,
            host_weights: None,
        })
    }

    /// Load an NVFP4 linear by slicing the leading `eff_rows × eff_logical_cols` block
    /// from a MatFormer nested-param checkpoint.
    ///
    /// Unlike the full-tensor loader, this always uses `LinearResidentLayout::PackedSource`
    /// (no repacking), so the kernel falls back to the unpacked NVFP4 path.
    #[allow(clippy::too_many_arguments)]
    pub fn load_nvfp4_linear_sliced_with_layout(
        &self,
        artifact: &ModelArtifact,
        prefix: &str,
        store: StoragePlacement,
        residency: TensorResidencyPlan,
        _resident_layout: LinearResidentLayout,
        eff_rows: usize,
        eff_logical_cols: usize,
        loader: &mut TensorStorageLoader,
    ) -> Result<DeviceNvfp4Linear> {
        let kernel_family =
            cuda_nvfp4_kernel_family_for_layout(prefix, LinearResidentLayout::PackedSource)?;
        let weight = artifact
            .tensors
            .get(&format!("{prefix}.weight"))
            .ok_or_else(|| AegisError::InvalidPlan(format!("missing `{prefix}.weight`")))?;
        let scales_info = artifact
            .tensors
            .get(&format!("{prefix}.weight_scale"))
            .ok_or_else(|| AegisError::InvalidPlan(format!("missing `{prefix}.weight_scale`")))?;
        let output_scale = artifact
            .tensors
            .get(&format!("{prefix}.weight_scale_2"))
            .map(|t| read_scalar_f32_with_loader(loader, t, store))
            .transpose()?
            .unwrap_or(1.0);
        let input_scale = artifact
            .tensors
            .get(&format!("{prefix}.input_scale"))
            .map(|t| read_scalar_f32_with_loader(loader, t, store))
            .transpose()?
            .unwrap_or(1.0);

        if eff_logical_cols == 0 || eff_logical_cols % QK_NVFP4 != 0 {
            return Err(AegisError::InvalidPlan(format!(
                "NVFP4 sliced `{prefix}` eff_logical_cols={eff_logical_cols} must be non-zero and divisible by {QK_NVFP4}"
            )));
        }
        let eff_packed_cols = eff_logical_cols / 2;
        let eff_scale_cols = eff_logical_cols / QK_NVFP4 * (QK_NVFP4 / QK_NVFP4_SUB);

        let packed_host = loader.load_submatrix(
            weight,
            NestedParamSlice::submatrix(eff_rows, eff_packed_cols),
        )?;
        let scales_host = loader.load_submatrix(
            scales_info,
            NestedParamSlice::submatrix(eff_rows, eff_scale_cols),
        )?;

        let is_host_resident = matches!(residency, TensorResidencyPlan::StagedHostToDevice { .. });
        if is_host_resident {
            let pinned_packed = alloc_pinned_from_bytes(
                self.runtime,
                packed_host.as_bytes(),
                "alloc pinned sliced host packed",
            )?;
            let pinned_scales = alloc_pinned_from_bytes(
                self.runtime,
                scales_host.as_bytes(),
                "alloc pinned sliced host scales",
            )?;
            let stub = self
                .runtime
                .stream
                .clone_htod(&[0u8])
                .map_err(map_cuda_err("htod sliced host-resident stub"))?;
            let stub2 = self
                .runtime
                .stream
                .clone_htod(&[0u8])
                .map_err(map_cuda_err("htod sliced host-resident stub2"))?;
            return Ok(DeviceNvfp4Linear {
                name: prefix.to_string(),
                rows: eff_rows,
                cols: eff_logical_cols,
                packed_bytes: packed_host.len(),
                scale_bytes: scales_host.len(),
                input_scale,
                output_scale,
                kernel_family,
                resident_layout: LinearResidentLayout::PackedSource,
                residency,
                packed: stub,
                scales: stub2,
                native_mxfp4: None,
                cutlass_nvfp4: None,
                host_weights: Some(Box::new(HostResidentWeights {
                    packed: HostWeightBytes::Pinned(pinned_packed),
                    scales: HostWeightBytes::Pinned(pinned_scales),
                    native_mxfp4: None,
                })),
            });
        }

        Ok(DeviceNvfp4Linear {
            name: prefix.to_string(),
            rows: eff_rows,
            cols: eff_logical_cols,
            packed_bytes: packed_host.len(),
            scale_bytes: scales_host.len(),
            input_scale,
            output_scale,
            kernel_family,
            resident_layout: LinearResidentLayout::PackedSource,
            residency,
            packed: self
                .runtime
                .stream
                .clone_htod(packed_host.as_bytes())
                .map_err(map_cuda_err("htod nvfp4 sliced packed weights"))?,
            scales: self
                .runtime
                .stream
                .clone_htod(scales_host.as_bytes())
                .map_err(map_cuda_err("htod nvfp4 sliced scales"))?,
            native_mxfp4: None,
            cutlass_nvfp4: None,
            host_weights: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_cutlass_qkv_group_with_layout(
        &self,
        artifact: &ModelArtifact,
        q_prefix: &str,
        k_prefix: &str,
        v_prefix: &str,
        store: StoragePlacement,
        residency: TensorResidencyPlan,
        resident_layout: LinearResidentLayout,
        loader: &mut TensorStorageLoader,
    ) -> Result<Option<DeviceNvfp4Linear>> {
        if !self.runtime.config().cutlass_nvfp4_repack {
            return Ok(None);
        }
        let kernel_family = cuda_nvfp4_kernel_family_for_layout(q_prefix, resident_layout)?;
        if !matches!(
            kernel_family,
            KernelFamily::CudaCutlassFp4TensorCores | KernelFamily::CudaNativeFp4TensorCores
        ) {
            return Ok(None);
        }

        let q = load_nvfp4_linear_host_parts(artifact, q_prefix, store, loader)?;
        let k = load_nvfp4_linear_host_parts(artifact, k_prefix, store, loader)?;
        let v = load_nvfp4_linear_host_parts(artifact, v_prefix, store, loader)?;
        if q.spec.cols != k.spec.cols || q.spec.cols != v.spec.cols {
            return Err(AegisError::InvalidPlan(format!(
                "CUTLASS QKV group shape mismatch: q={}x{} k={}x{} v={}x{}",
                q.spec.rows, q.spec.cols, k.spec.rows, k.spec.cols, v.spec.rows, v.spec.cols
            )));
        }
        if (q.spec.input_scale - k.spec.input_scale).abs() > 1.0e-12
            || (q.spec.input_scale - v.spec.input_scale).abs() > 1.0e-12
        {
            return Err(AegisError::InvalidPlan(format!(
                "CUTLASS QKV group requires equal input scales: q={} k={} v={}",
                q.spec.input_scale, k.spec.input_scale, v.spec.input_scale
            )));
        }

        let rows = q
            .spec
            .rows
            .checked_add(k.spec.rows)
            .and_then(|rows| rows.checked_add(v.spec.rows))
            .ok_or_else(|| AegisError::InvalidPlan("CUTLASS QKV group rows overflow".into()))?;
        let packed_bytes = q
            .spec
            .packed_bytes
            .checked_add(k.spec.packed_bytes)
            .and_then(|bytes| bytes.checked_add(v.spec.packed_bytes))
            .ok_or_else(|| {
                AegisError::InvalidPlan("CUTLASS QKV group packed bytes overflow".into())
            })?;
        let scale_bytes = q
            .spec
            .scale_bytes
            .checked_add(k.spec.scale_bytes)
            .and_then(|bytes| bytes.checked_add(v.spec.scale_bytes))
            .ok_or_else(|| {
                AegisError::InvalidPlan("CUTLASS QKV group scale bytes overflow".into())
            })?;
        let group_spec = Nvfp4LinearSpec {
            name: format!("{q_prefix}+{k_prefix}+{v_prefix}"),
            rows,
            cols: q.spec.cols,
            packed_bytes,
            scale_bytes,
            input_scale: q.spec.input_scale,
            // The fused GEMM writes an unscaled accumulator. A tiny split kernel
            // applies per-projection output scales while scattering to q/k/v.
            output_scale: 1.0,
        };

        let mut packed = Vec::with_capacity(packed_bytes);
        packed.extend_from_slice(q.packed.as_bytes());
        packed.extend_from_slice(k.packed.as_bytes());
        packed.extend_from_slice(v.packed.as_bytes());
        let mut scales = Vec::with_capacity(scale_bytes);
        scales.extend_from_slice(q.scales.as_bytes());
        scales.extend_from_slice(k.scales.as_bytes());
        scales.extend_from_slice(v.scales.as_bytes());
        let repacked = repack_nvfp4_to_cutlass_e2m1_ue4m3_host(&group_spec, &packed, &scales)?;

        Ok(Some(DeviceNvfp4Linear {
            name: group_spec.name,
            rows: group_spec.rows,
            cols: group_spec.cols,
            packed_bytes: group_spec.packed_bytes,
            scale_bytes: group_spec.scale_bytes,
            input_scale: group_spec.input_scale,
            output_scale: group_spec.output_scale,
            kernel_family,
            resident_layout,
            residency,
            packed: self
                .runtime
                .stream
                .clone_htod(&packed)
                .map_err(map_cuda_err("htod qkv group nvfp4 packed weights"))?,
            scales: self
                .runtime
                .stream
                .clone_htod(&scales)
                .map_err(map_cuda_err("htod qkv group nvfp4 scales"))?,
            native_mxfp4: None,
            cutlass_nvfp4: Some(DeviceCutlassNvfp4Linear {
                layout: repacked.layout,
                payload_e2m1: self
                    .runtime
                    .stream
                    .clone_htod(&repacked.payload_e2m1)
                    .map_err(map_cuda_err("htod qkv group cutlass nvfp4 payload"))?,
                scales_ue4m3: self
                    .runtime
                    .stream
                    .clone_htod(&repacked.scales_ue4m3)
                    .map_err(map_cuda_err("htod qkv group cutlass nvfp4 scales"))?,
            }),
            host_weights: None,
        }))
    }

    pub fn load_region_nvfp4_linears(
        &self,
        artifact: &ModelArtifact,
        region: &GraphRegion,
    ) -> Result<Vec<DeviceNvfp4Linear>> {
        let mut linears = Vec::new();
        let mut loader = TensorStorageLoader::new();
        for tensor in &region.tensors {
            if !is_nvfp4_linear_weight(tensor) {
                continue;
            }
            let prefix = tensor.info.name.strip_suffix(".weight").ok_or_else(|| {
                AegisError::InvalidPlan(format!("bad linear tensor name `{}`", tensor.info.name))
            })?;
            linears.push(self.load_nvfp4_linear_with_store(
                artifact,
                prefix,
                StoragePlacement::Vram {
                    device: self.runtime.device_index(),
                },
                TensorResidencyPlan::VramResident {
                    device: self.runtime.device_index(),
                },
                &mut loader,
            )?);
        }
        Ok(linears)
    }

    pub fn load_region_nvfp4_linears_with_store(
        &self,
        artifact: &ModelArtifact,
        region: &GraphRegion,
        store: StoragePlacement,
        residency: TensorResidencyPlan,
    ) -> Result<Vec<DeviceNvfp4Linear>> {
        self.load_region_nvfp4_linears_with_layout(
            artifact,
            region,
            store,
            residency,
            LinearResidentLayout::NativeTensorCore,
        )
    }

    pub fn load_region_nvfp4_linears_with_layout(
        &self,
        artifact: &ModelArtifact,
        region: &GraphRegion,
        store: StoragePlacement,
        residency: TensorResidencyPlan,
        resident_layout: LinearResidentLayout,
    ) -> Result<Vec<DeviceNvfp4Linear>> {
        let mut linears = Vec::new();
        let mut loader = TensorStorageLoader::new();
        for tensor in &region.tensors {
            if !is_nvfp4_linear_weight(tensor) {
                continue;
            }
            let prefix = tensor.info.name.strip_suffix(".weight").ok_or_else(|| {
                AegisError::InvalidPlan(format!("bad linear tensor name `{}`", tensor.info.name))
            })?;
            linears.push(self.load_nvfp4_linear_with_layout(
                artifact,
                prefix,
                store,
                residency,
                resident_layout,
                &mut loader,
            )?);
        }
        Ok(linears)
    }

    pub fn load_placed_region_nvfp4_linears(
        &self,
        artifact: &ModelArtifact,
        region: &GraphRegion,
        placement: &RegionPlacement,
    ) -> Result<Vec<DeviceNvfp4Linear>> {
        self.load_placed_region_nvfp4_linears_with_layout(
            artifact,
            region,
            placement,
            LinearResidentLayout::NativeTensorCore,
        )
    }

    pub fn load_placed_region_nvfp4_linears_with_layout(
        &self,
        artifact: &ModelArtifact,
        region: &GraphRegion,
        placement: &RegionPlacement,
        resident_layout: LinearResidentLayout,
    ) -> Result<Vec<DeviceNvfp4Linear>> {
        if placement.region_id != region.id {
            return Err(AegisError::InvalidPlan(format!(
                "placement `{}` does not match graph region `{}`",
                placement.region_id.0, region.id.0
            )));
        }
        match (placement.store, placement.compute) {
            (
                StoragePlacement::Vram {
                    device: store_device,
                },
                ComputePlacement::Cuda {
                    device: compute_device,
                },
            ) if store_device == self.runtime.device_index()
                && compute_device == self.runtime.device_index() =>
            {
                self.load_region_nvfp4_linears_with_layout(
                    artifact,
                    region,
                    placement.store,
                    TensorResidencyPlan::VramResident {
                        device: self.runtime.device_index(),
                    },
                    resident_layout,
                )
            }
            (StoragePlacement::Ram | StoragePlacement::Mmap, ComputePlacement::Cuda { device })
                if device == self.runtime.device_index() =>
            {
                self.load_region_nvfp4_linears_with_layout(
                    artifact,
                    region,
                    placement.store,
                    TensorResidencyPlan::StagedHostToDevice {
                        device: self.runtime.device_index(),
                    },
                    resident_layout,
                )
            }
            (StoragePlacement::Ram | StoragePlacement::Mmap, ComputePlacement::Cuda { device }) => {
                Err(AegisError::Unsupported(format!(
                    "region `{}` is compute=cuda:{device}, but this CUDA runtime is cuda:{}",
                    region.id.0,
                    self.runtime.device_index()
                )))
            }
            (StoragePlacement::Vram { device }, ComputePlacement::Cuda { device: compute }) => {
                Err(AegisError::Unsupported(format!(
                    "region `{}` has cross-device placement store=vram:{device} compute=cuda:{compute}; cross-device loaders are not implemented yet",
                    region.id.0
                )))
            }
            (_, ComputePlacement::Cpu) => Err(AegisError::Unsupported(format!(
                "region `{}` is compute=cpu; CUDA resident loader refused to load it",
                region.id.0
            ))),
            (_, ComputePlacement::Wgpu { device }) => Err(AegisError::Unsupported(format!(
                "region `{}` is compute=wgpu:{device}; CUDA resident loader refused to load it",
                region.id.0
            ))),
        }
    }

    pub fn load_first_placed_region_nvfp4_linear(
        &self,
        artifact: &ModelArtifact,
        region: &GraphRegion,
        placement: &RegionPlacement,
    ) -> Result<Option<DeviceNvfp4Linear>> {
        self.load_first_placed_region_nvfp4_linear_with_layout(
            artifact,
            region,
            placement,
            LinearResidentLayout::NativeTensorCore,
        )
    }

    pub fn load_first_placed_region_nvfp4_linear_with_layout(
        &self,
        artifact: &ModelArtifact,
        region: &GraphRegion,
        placement: &RegionPlacement,
        resident_layout: LinearResidentLayout,
    ) -> Result<Option<DeviceNvfp4Linear>> {
        if placement.region_id != region.id {
            return Err(AegisError::InvalidPlan(format!(
                "placement `{}` does not match graph region `{}`",
                placement.region_id.0, region.id.0
            )));
        }
        let Some(prefix) = first_nvfp4_linear_prefix(region) else {
            return Ok(None);
        };
        let mut loader = TensorStorageLoader::new();
        match (placement.store, placement.compute) {
            (
                StoragePlacement::Vram {
                    device: store_device,
                },
                ComputePlacement::Cuda {
                    device: compute_device,
                },
            ) if store_device == self.runtime.device_index()
                && compute_device == self.runtime.device_index() =>
            {
                self.load_nvfp4_linear_with_layout(
                    artifact,
                    prefix,
                    placement.store,
                    TensorResidencyPlan::VramResident {
                        device: self.runtime.device_index(),
                    },
                    resident_layout,
                    &mut loader,
                )
                .map(Some)
            }
            (StoragePlacement::Ram | StoragePlacement::Mmap, ComputePlacement::Cuda { device })
                if device == self.runtime.device_index() =>
            {
                self.load_nvfp4_linear_with_layout(
                    artifact,
                    prefix,
                    placement.store,
                    TensorResidencyPlan::StagedHostToDevice {
                        device: self.runtime.device_index(),
                    },
                    resident_layout,
                    &mut loader,
                )
                .map(Some)
            }
            (StoragePlacement::Ram | StoragePlacement::Mmap, ComputePlacement::Cuda { device }) => {
                Err(AegisError::Unsupported(format!(
                    "region `{}` is compute=cuda:{device}, but this CUDA runtime is cuda:{}",
                    region.id.0,
                    self.runtime.device_index()
                )))
            }
            (StoragePlacement::Vram { device }, ComputePlacement::Cuda { device: compute }) => {
                Err(AegisError::Unsupported(format!(
                    "region `{}` has cross-device placement store=vram:{device} compute=cuda:{compute}; cross-device loaders are not implemented yet",
                    region.id.0
                )))
            }
            (_, ComputePlacement::Cpu) => Err(AegisError::Unsupported(format!(
                "region `{}` is compute=cpu; CUDA resident loader refused to load it",
                region.id.0
            ))),
            (_, ComputePlacement::Wgpu { device }) => Err(AegisError::Unsupported(format!(
                "region `{}` is compute=wgpu:{device}; CUDA resident loader refused to load it",
                region.id.0
            ))),
        }
    }

    fn should_repack_native_mxfp4(&self, prefix: &str, kernel_family: KernelFamily) -> bool {
        kernel_family == KernelFamily::CudaNativeFp4TensorCores
            && self.runtime.config().native_mxfp4_repack
            && !(self.runtime.config().cutlass_nvfp4_repack
                && native_layout_cutlass_prefill_sidecar(prefix))
    }

    fn should_repack_cutlass_nvfp4(
        &self,
        prefix: &str,
        kernel_family: KernelFamily,
        resident_layout: LinearResidentLayout,
    ) -> bool {
        resident_layout == LinearResidentLayout::CudaR4fE2m1Ue4m3
            || kernel_family == KernelFamily::CudaCutlassFp4TensorCores
            || (kernel_family == KernelFamily::CudaNativeFp4TensorCores
                && self.runtime.config().cutlass_nvfp4_repack
                && native_layout_cutlass_prefill_sidecar(prefix))
    }
}

impl CudaWeightLoader<'_> {
    /// Returns a 1×1 NvFP4 placeholder that satisfies the type system but is never used for
    /// computation. MoE layers hold real per-expert linears in `CudaMoE`; the `CudaLayer`
    /// gate/up/down fields are dummies so the struct is always fully initialised.
    pub fn alloc_dummy_nvfp4_linear(&self, name: &str) -> Result<DeviceNvfp4Linear> {
        let stub = self
            .runtime
            .stream
            .clone_htod(&[0u8])
            .map_err(map_cuda_err("htod dummy nvfp4 stub"))?;
        let stub2 = self
            .runtime
            .stream
            .clone_htod(&[0u8])
            .map_err(map_cuda_err("htod dummy nvfp4 stub2"))?;
        Ok(DeviceNvfp4Linear {
            name: name.to_string(),
            rows: 1,
            cols: 1,
            packed_bytes: 1,
            scale_bytes: 1,
            input_scale: 1.0,
            output_scale: 1.0,
            kernel_family: KernelFamily::CpuScalar,
            resident_layout: LinearResidentLayout::PackedSource,
            residency: TensorResidencyPlan::VramResident {
                device: self.runtime.device_index(),
            },
            packed: stub,
            scales: stub2,
            native_mxfp4: None,
            cutlass_nvfp4: None,
            host_weights: None,
        })
    }
}

fn native_layout_cutlass_prefill_sidecar(prefix: &str) -> bool {
    prefix.ends_with(".self_attn.o_proj")
        || prefix.ends_with(".mlp.gate_proj")
        || prefix.ends_with(".mlp.up_proj")
        || prefix.ends_with(".mlp.down_proj")
}

struct Nvfp4LinearHostParts {
    spec: Nvfp4LinearSpec,
    packed: LoadedHostTensor,
    scales: LoadedHostTensor,
}

fn load_nvfp4_linear_host_parts(
    artifact: &ModelArtifact,
    prefix: &str,
    store: StoragePlacement,
    loader: &mut TensorStorageLoader,
) -> Result<Nvfp4LinearHostParts> {
    let weight = artifact
        .tensors
        .get(&format!("{prefix}.weight"))
        .ok_or_else(|| AegisError::InvalidPlan(format!("missing `{prefix}.weight`")))?;
    let scales = artifact
        .tensors
        .get(&format!("{prefix}.weight_scale"))
        .ok_or_else(|| AegisError::InvalidPlan(format!("missing `{prefix}.weight_scale`")))?;
    let output_scale = artifact
        .tensors
        .get(&format!("{prefix}.weight_scale_2"))
        .map(|tensor| read_scalar_f32_with_loader(loader, tensor, store))
        .transpose()?
        .unwrap_or(1.0);
    let input_scale = artifact
        .tensors
        .get(&format!("{prefix}.input_scale"))
        .map(|tensor| read_scalar_f32_with_loader(loader, tensor, store))
        .transpose()?
        .unwrap_or(1.0);
    let spec = Nvfp4LinearSpec::from_tensors(prefix, weight, scales, input_scale, output_scale)?;
    let packed = loader.load_for_store(weight, store)?;
    let scales = loader.load_for_store(scales, store)?;
    Ok(Nvfp4LinearHostParts {
        spec,
        packed,
        scales,
    })
}

pub(crate) fn read_scalar_f32_with_loader(
    loader: &mut TensorStorageLoader,
    tensor: &TensorInfo,
    store: StoragePlacement,
) -> Result<f32> {
    let loaded: LoadedHostTensor = loader.load_for_store(tensor, store)?;
    let bytes = loaded.as_bytes();
    match tensor.dtype {
        TensorDType::F32 if bytes.len() == 4 => Ok(f32::from_le_bytes(bytes.try_into().map_err(
            |_| AegisError::InvalidPlan(format!("bad scalar F32 tensor `{}`", tensor.name)),
        )?)),
        TensorDType::BF16 if bytes.len() == 2 => {
            let bits = u16::from_le_bytes(bytes.try_into().map_err(
                |_| AegisError::InvalidPlan(format!("bad scalar BF16 tensor `{}`", tensor.name)),
            )?);
            // BF16 is the top 16 bits of an F32; shift left by 16 to recover f32 bits.
            Ok(f32::from_bits((bits as u32) << 16))
        }
        _ => Err(AegisError::InvalidPlan(format!(
            "`{}` must be a scalar F32 or BF16 tensor, got {:?} ({} bytes)",
            tensor.name, tensor.dtype, bytes.len()
        ))),
    }
}

fn first_nvfp4_linear_prefix(region: &GraphRegion) -> Option<&str> {
    region
        .tensors
        .iter()
        .find(|tensor| is_nvfp4_linear_weight(tensor))
        .and_then(|tensor| tensor.info.name.strip_suffix(".weight"))
}

fn is_nvfp4_linear_weight(tensor: &aegisllm_base::graph::GraphTensor) -> bool {
    matches!(
        tensor.role,
        TensorRole::Query
            | TensorRole::Key
            | TensorRole::Value
            | TensorRole::Output
            | TensorRole::Gate
            | TensorRole::Up
            | TensorRole::Down
    ) && tensor.info.dtype == TensorDType::U8
}

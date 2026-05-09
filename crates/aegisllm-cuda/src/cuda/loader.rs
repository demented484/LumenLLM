use super::repack::{
    cached_repack_nvfp4_to_cutlass_e2m1_ue4m3_host, cached_repack_nvfp4_to_mxfp4_host,
    repack_nvfp4_to_cutlass_e2m1_ue4m3_host,
};
use super::runtime::{CudaRuntime, map_cuda_err};
use super::types::{
    DeviceBf16Matrix, DeviceBuffer, DeviceCutlassNvfp4Linear, DeviceMxfp4Linear, DeviceNvfp4Linear,
    HostBf16Weights, HostResidentMxfp4, HostResidentWeights, HostWeightBytes,
};
use aegisllm_base::artifact::ModelArtifact;
use aegisllm_base::error::{AegisError, Result};
use cudarc::driver::PinnedHostSlice;
use std::io::{Read, Seek, SeekFrom};

/// Allocate CUDA-pinned host memory and copy an in-memory byte slice into it.
/// Only used for repacked data (native MXFP4) that is generated in a Vec<u8> at runtime.
/// For tensors loaded from disk, prefer `alloc_pinned_u8_from_file` to avoid the
/// intermediate mmap/pageable copy.
fn alloc_pinned_from_bytes(
    runtime: &CudaRuntime,
    bytes: &[u8],
    label: &'static str,
) -> Result<PinnedHostSlice<u8>> {
    let mut pinned = unsafe { runtime.stream.context().alloc_pinned::<u8>(bytes.len()) }
        .map_err(map_cuda_err(label))?;
    pinned
        .as_mut_slice()
        .map_err(map_cuda_err(label))?
        .copy_from_slice(bytes);
    Ok(pinned)
}

/// Read a tensor's bytes directly from the safetensors file into a freshly allocated
/// CUDA-pinned host buffer.  Avoids the mmap-then-copy double-allocation that occurs
/// when weights are staged from disk: only one copy of the data exists in RAM (the
/// pinned buffer), instead of the kernel page-cache pages + a separate pinned copy.
fn alloc_pinned_u8_from_file(
    runtime: &CudaRuntime,
    tensor: &TensorInfo,
    label: &'static str,
) -> Result<PinnedHostSlice<u8>> {
    let len = tensor.data_len_bytes() as usize;
    let mut pinned = unsafe { runtime.stream.context().alloc_pinned::<u8>(len) }
        .map_err(map_cuda_err(label))?;
    {
        let dst = pinned.as_mut_slice().map_err(map_cuda_err(label))?;
        let mut file = std::fs::File::open(&tensor.shard_path)?;
        file.seek(SeekFrom::Start(tensor.file_offsets.0))?;
        file.read_exact(dst)?;
        aegisllm_base::tensor::storage::fadvise_dont_need(
            &file,
            tensor.file_offsets.0,
            len as u64,
        );
    }
    Ok(pinned)
}

/// Read a tensor's bytes into the shared pinned-host arena via direct file
/// I/O. Uses `read_exact` straight into the arena slice — sequential reads
/// inside a shard let the OS aggressively prefetch.
///
/// Per-call file open is cheap relative to the file read itself (NVMe at
/// ~3-5 GB/s dominates), but for very many tensors the open + seek overhead
/// adds up. We accept this for code simplicity; if it shows up as a hot
/// spot, the next step is a per-shard `File` handle cache or a single
/// shard-wide read into a temporary buffer.
fn read_tensor_into_arena(
    arena: &super::host_arena::ArenaHandle,
    tensor: &TensorInfo,
) -> Result<super::types::HostWeightBytes> {
    let len = tensor.data_len_bytes() as usize;
    let mut file = std::fs::File::open(&tensor.shard_path)?;
    file.seek(SeekFrom::Start(tensor.file_offsets.0))?;
    let offset = arena.alloc_and_fill(&mut file, len)?;
    aegisllm_base::tensor::storage::fadvise_dont_need(
        &file,
        tensor.file_offsets.0,
        len as u64,
    );
    Ok(super::types::HostWeightBytes::Arena {
        arena: arena.clone(),
        offset,
        len,
    })
}

/// Read a BF16 tensor directly from disk into a CUDA-pinned u16 buffer.
/// Safetensors uses little-endian byte order, same as x86/ARM, so the raw bytes
/// can be reinterpreted as u16 values in-place without endian conversion.
fn alloc_pinned_u16_from_file(
    runtime: &CudaRuntime,
    tensor: &TensorInfo,
    label: &'static str,
) -> Result<PinnedHostSlice<u16>> {
    let len_bytes = tensor.data_len_bytes() as usize;
    if len_bytes % 2 != 0 {
        return Err(AegisError::InvalidPlan(format!(
            "`{}` BF16 byte length is not even: {len_bytes}",
            tensor.name
        )));
    }
    let len_u16 = len_bytes / 2;
    let mut pinned = unsafe { runtime.stream.context().alloc_pinned::<u16>(len_u16) }
        .map_err(map_cuda_err(label))?;
    {
        let dst_u16 = pinned.as_mut_slice().map_err(map_cuda_err(label))?;
        // Safety: u16 and u8 have the same alignment requirements here; we treat the
        // pinned u16 buffer as a raw byte buffer for the initial file read.
        let dst_u8 = unsafe {
            std::slice::from_raw_parts_mut(dst_u16.as_mut_ptr() as *mut u8, len_bytes)
        };
        let mut file = std::fs::File::open(&tensor.shard_path)?;
        file.seek(SeekFrom::Start(tensor.file_offsets.0))?;
        file.read_exact(dst_u8)?;
        aegisllm_base::tensor::storage::fadvise_dont_need(
            &file,
            tensor.file_offsets.0,
            len_bytes as u64,
        );
    }
    Ok(pinned)
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

pub struct CudaWeightLoader<'a> {
    runtime: &'a CudaRuntime,
    /// Shared pinned-host arena for staged-NVFP4 weight residency. When set,
    /// host-resident NVFP4 weights are read directly into this arena instead
    /// of allocating a separate `cuMemAllocHost` per tensor — collapses ~7700
    /// pinned allocations into one. `None` for paths that don't need host
    /// residency (used for backwards-compatible callers).
    arena: Option<super::host_arena::ArenaHandle>,
    /// Single reusable pinned-host bounce buffer for VRAM-resident weight
    /// uploads. Replaces the prior `mmap shard → clone_htod` path which
    /// (a) ballooned the kernel page cache by the full ~17 GiB model size
    /// and (b) bounced through the CUDA driver's internal pinned staging on
    /// every memcpy from a pageable mmap source. With this buffer, each
    /// VRAM-resident tensor is read directly from disk into pinned memory
    /// (no page-cache fill) and then DMA'd to the device with a single
    /// pinned→device memcpy. Sized to the largest tensor seen so far —
    /// grows on demand and stays at high-water mark for the rest of the
    /// load. Wrapped in `RefCell` because `&CudaWeightLoader` is used by
    /// many `&self` methods and only the inner buffer needs interior
    /// mutability.
    bounce: std::cell::RefCell<Option<PinnedHostSlice<u8>>>,
}

impl CudaRuntime {
    pub fn weight_loader(&self) -> CudaWeightLoader<'_> {
        CudaWeightLoader {
            runtime: self,
            arena: None,
            bounce: std::cell::RefCell::new(None),
        }
    }

    /// Create a weight loader bound to a pre-allocated pinned-host arena.
    /// The arena is consumed by host-resident NVFP4 weights as they are loaded;
    /// non-host-resident weights ignore it.
    pub fn weight_loader_with_arena(
        &self,
        arena: super::host_arena::ArenaHandle,
    ) -> CudaWeightLoader<'_> {
        CudaWeightLoader {
            runtime: self,
            arena: Some(arena),
            bounce: std::cell::RefCell::new(None),
        }
    }
}

impl CudaWeightLoader<'_> {
    pub fn device_index(&self) -> usize {
        self.runtime.device_index()
    }

    /// Borrow the underlying runtime for callers that need direct access to
    /// allocator / upload primitives during loading. Used by the executor's
    /// loader to populate `router_per_expert_scale_device` and similar
    /// accompanying device-resident metadata.
    pub fn runtime(&self) -> &CudaRuntime {
        self.runtime
    }

    /// Borrow the loader's pinned arena (set when the loader was built via
    /// `weight_loader_with_arena`). Returns `None` for the bare-loader
    /// variant. Used by the parallel host-NVFP4 prefetch helper below.
    pub fn arena(&self) -> Option<&super::host_arena::ArenaHandle> {
        self.arena.as_ref()
    }

    /// Parallel-read pairs of NVFP4 tensors (`{prefix}.weight`,
    /// `{prefix}.weight_scale`) into the loader's pinned host arena.
    ///
    /// Each `(weight, scales)` pair runs on a rayon worker that opens the
    /// safetensors shard file, seeks to the tensor's byte range, and reads
    /// directly into the arena via `arena.alloc_and_fill`. The arena bump
    /// pointer is atomic (`SeqCst fetch_add`) so concurrent calls produce
    /// disjoint regions. Returns the host bytes in input order so the
    /// serial consumer can attach them to the matching `DeviceNvfp4Linear`
    /// without re-reading.
    ///
    /// Used by the MoE expert loader to overlap the per-tensor file I/O
    /// across all 128 routed experts × 3 projections instead of issuing
    /// one read at a time. NVMe benefits from ≥4 concurrent in-flight
    /// reads; rayon's default pool (≈ #cores) is plenty.
    pub fn prefetch_host_nvfp4_pairs_par<'a>(
        &self,
        artifact: &'a ModelArtifact,
        prefixes: &[String],
    ) -> Result<Vec<(super::types::HostWeightBytes, super::types::HostWeightBytes)>> {
        use rayon::prelude::*;
        let arena = self.arena.as_ref().ok_or_else(|| {
            AegisError::InvalidPlan(
                "prefetch_host_nvfp4_pairs_par requires loader built with arena".into(),
            )
        })?;
        prefixes
            .par_iter()
            .map(|prefix| -> Result<(super::types::HostWeightBytes, super::types::HostWeightBytes)> {
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

    /// Build a `DeviceNvfp4Linear` for a host-resident layer using bytes
    /// already read into the arena by `prefetch_host_nvfp4_pairs_par`.
    ///
    /// This is the **finalize** half of the parallel-prefetch flow: the
    /// expensive part (file I/O into pinned arena) ran on a rayon worker,
    /// and now the main thread does the cheap CUDA-side work (two 1-byte
    /// stub allocs and the metadata struct construction). `loader` is
    /// still consulted for the optional `weight_scale_2` / `input_scale`
    /// scalars which are tiny and stay on the mmap path.
    ///
    /// Skips the repack branches of the full
    /// `load_nvfp4_linear_with_layout` because those run only for
    /// VRAM-resident layouts (native MXFP4 / cutlass NVFP4), which the
    /// prefetch path is not currently used for.
    pub fn finalize_host_nvfp4_with_prefetched(
        &self,
        artifact: &ModelArtifact,
        prefix: &str,
        residency: TensorResidencyPlan,
        store: StoragePlacement,
        packed_bytes: super::types::HostWeightBytes,
        scales_bytes: super::types::HostWeightBytes,
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
            host_weights: Some(Box::new(super::types::HostResidentWeights {
                packed: packed_bytes,
                scales: scales_bytes,
                native_mxfp4: None,
            })),
        })
    }

    /// Ensure the loader's pinned bounce is ≥ `min_len` bytes, then read
    /// `tensor`'s bytes directly from disk into it. Returns a guard that
    /// callers use to access the populated pinned slice; on guard drop,
    /// the bounce stays allocated for reuse by the next tensor.
    ///
    /// This is the load-time replacement for `loader.load_for_store(...,
    /// Vram) → mmap → clone_htod`. Two payoffs vs the mmap path:
    ///
    ///   1. No kernel page-cache fill. The destination is pinned host
    ///      memory; reading into it does not populate the page cache for
    ///      the source file. Combined with the `fadvise(DONTNEED)` on the
    ///      file range afterwards, the kernel evicts the just-read pages
    ///      eagerly instead of accumulating ~17 GiB of cached weights and
    ///      then discarding them in 8 GiB sawtooth bursts.
    ///   2. No implicit pageable→pinned bounce inside `cudaMemcpy`. The
    ///      source is already pinned, so the H2D is a single DMA hop.
    fn read_tensor_into_pinned(
        &self,
        tensor: &TensorInfo,
    ) -> Result<()> {
        let len = tensor.data_len_bytes() as usize;
        let need_realloc = self
            .bounce
            .borrow()
            .as_ref()
            .map(|b| b.len() < len)
            .unwrap_or(true);
        if need_realloc {
            *self.bounce.borrow_mut() = None;
            let pinned = self.runtime.alloc_pinned_u8(len)?;
            *self.bounce.borrow_mut() = Some(pinned);
        }
        let mut bounce_ref = self.bounce.borrow_mut();
        let bounce = bounce_ref.as_mut().expect("bounce just ensured to exist");
        let bytes = bounce
            .as_mut_slice()
            .map_err(map_cuda_err("bounce as_mut_slice"))?;
        let mut file = std::fs::File::open(&tensor.shard_path)?;
        file.seek(SeekFrom::Start(tensor.file_offsets.0))?;
        file.read_exact(&mut bytes[..len])?;
        aegisllm_base::tensor::storage::fadvise_dont_need(
            &file,
            tensor.file_offsets.0,
            len as u64,
        );
        Ok(())
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
        store: StoragePlacement,
        residency: TensorResidencyPlan,
        loader: &mut TensorStorageLoader,
    ) -> Result<DeviceBf16Matrix> {
        if tensor.dtype != TensorDType::BF16 || tensor.shape.len() != 2 {
            return Err(AegisError::InvalidPlan(format!(
                "`{}` must be a BF16 matrix",
                tensor.name
            )));
        }
        // Residency is now strictly config-driven: `store=ram` → host-pinned;
        // `store=vram` → device-resident. There is no force_vram override.
        // If the host-resident matvec path is too slow for a given workload
        // (e.g. lm_head over WRITECOMBINED RAM is ~30× slower than the VRAM
        // kernel), set `output-layer.store = vram` in parameters.json.
        let is_host_resident = matches!(residency, TensorResidencyPlan::StagedHostToDevice { .. });
        if is_host_resident {
            // Read directly from file into pinned u16 memory — avoids the mmap page-cache
            // copy and the intermediate Vec<u16>; only one copy of the data exists in RAM.
            let pinned = alloc_pinned_u16_from_file(
                self.runtime,
                tensor,
                "alloc pinned bf16 host",
            )?;
            let stub = self
                .runtime
                .stream
                .clone_htod(&[0u16])
                .map_err(map_cuda_err("htod bf16 host-resident stub"))?;
            return Ok(DeviceBf16Matrix {
                name: tensor.name.clone(),
                rows: tensor.shape[0],
                cols: tensor.shape[1],
                residency,
                values: stub,
                host_values: Some(Box::new(HostBf16Weights { values: pinned })),
            });
        }
        // VRAM-resident BF16: bypass the mmap path entirely. Read the
        // tensor's bytes from disk into the loader's pinned bounce buffer,
        // then DMA from pinned → fresh u16-typed VRAM. Avoids both the
        // kernel page-cache fill and the implicit pageable→pinned bounce
        // inside `cudaMemcpy` from the old `mmap → clone_htod` path.
        // `loader` is unused on this branch but kept in the signature
        // because the host-resident branch above still consults it.
        let _ = loader;
        let len_bytes = tensor.data_len_bytes() as usize;
        if len_bytes % 2 != 0 {
            return Err(AegisError::InvalidPlan(format!(
                "BF16 tensor `{}` has odd byte length {}", tensor.name, len_bytes
            )));
        }
        let len_u16 = len_bytes / 2;
        self.read_tensor_into_pinned(tensor)?;
        let mut buffer = self.runtime.alloc_u16(len_u16)?;
        let bounce_ref = self.bounce.borrow();
        let bounce = bounce_ref
            .as_ref()
            .expect("bounce populated by read_tensor_into_pinned");
        let bounce_bytes = bounce
            .as_slice()
            .map_err(map_cuda_err("bounce as_slice for bf16 htod"))?;
        // SAFETY: cuMemAllocHost returns page-aligned memory (so 2-byte
        // alignment is satisfied), and `len_bytes` is even-checked above.
        let bounce_u16: &[u16] = unsafe {
            std::slice::from_raw_parts(bounce_bytes.as_ptr() as *const u16, len_u16)
        };
        self.runtime
            .stream
            .memcpy_htod(bounce_u16, &mut buffer.slice)
            .map_err(map_cuda_err("htod bf16 matrix from pinned"))?;
        Ok(DeviceBf16Matrix {
            name: tensor.name.clone(),
            rows: tensor.shape[0],
            cols: tensor.shape[1],
            residency,
            values: buffer.slice,
            host_values: None,
        })
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

        // Load mmap data only when it is needed: VRAM upload, or repacking for tensor cores.
        // For plain host-resident layers (staged streaming) we read straight into pinned RAM
        // via alloc_pinned_u8_from_file, avoiding the kernel page-cache copy entirely.
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
            // Host-resident NVFP4 weights are read directly into the shared
            // pinned-host arena. One `cuMemAllocHost` covers all weights;
            // each tensor sub-allocates by atomic offset bump. This keeps the
            // hot inference path with **zero CPU memcpy** (source is pinned →
            // GPU DMA pulls directly) at the cost of locking ~total_model_size
            // RAM. See `host_arena.rs` for the rationale and trade-off vs the
            // earlier mmap+bounce approach.
            let arena = self.arena.as_ref().ok_or_else(|| {
                AegisError::InvalidPlan(format!(
                    "host-resident NVFP4 layer `{}` requires the loader to be built \
                     with `weight_loader_with_arena(...)`; got bare `weight_loader()`",
                    spec.name,
                ))
            })?;
            let packed_arena = read_tensor_into_arena(arena, weight)?;
            let scales_arena = read_tensor_into_arena(arena, scales)?;
            let mmap_packed = packed_arena;
            let mmap_scales = scales_arena;
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

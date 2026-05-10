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

/// Allocate a process-owned pinned host buffer and copy `bytes` into it.
/// Used only for small, generated-at-load-time data (native MXFP4 repack
/// output, MatFormer submatrix slices) where the source is a transient
/// `Vec<u8>` with no file-backed mmap to point at. The general path keeps
/// weights in shard mmaps; this helper covers the few exceptions.
fn alloc_pinned_from_bytes(
    _runtime: &CudaRuntime,
    bytes: &[u8],
    _label: &'static str,
) -> Result<OwnedPinnedBuf> {
    let mut pinned = OwnedPinnedBuf::new(bytes.len())?;
    pinned.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
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

/// Callback invoked by the loader to report fine-grained sub-step progress
/// (e.g. "layer 5 expert 64/128") between coarse `step` advances. The executor
/// wires this to its TTY progress indicator; cross-crate callers that don't
/// care about progress just leave it `None`.
pub type LoadStatusSink = std::sync::Arc<dyn Fn(&str) + Send + Sync>;

pub struct CudaWeightLoader<'a> {
    runtime: &'a CudaRuntime,
    /// Optional callback for fine-grained sub-step progress. Heavy inner
    /// loops (MoE experts, big BF16 uploads) call `report_status(...)` so
    /// the user's progress indicator can refresh between coarse `step`
    /// advances.
    status_sink: Option<LoadStatusSink>,
    /// Set of safetensors shard paths from which at least one
    /// host-resident weight has been loaded. Populated as a side effect
    /// of `load_for_store(_, Mmap)` for host-resident NVFP4 / BF16
    /// branches; the executor reads it after load to register those
    /// shard mmaps with `cuMemHostRegister`, so per-token streaming
    /// takes the direct-DMA fast path.
    host_resident_shards: std::cell::RefCell<std::collections::HashSet<std::path::PathBuf>>,
}

impl CudaRuntime {
    pub fn weight_loader(&self) -> CudaWeightLoader<'_> {
        CudaWeightLoader {
            runtime: self,
            status_sink: None,
            host_resident_shards: std::cell::RefCell::new(std::collections::HashSet::new()),
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

    /// Snapshot the set of safetensors shard paths from which at least
    /// one host-resident weight has been loaded. Called by the executor
    /// at end-of-load to drive `cuMemHostRegister` on those shards.
    pub fn host_resident_shards(&self) -> std::collections::HashSet<std::path::PathBuf> {
        self.host_resident_shards.borrow().clone()
    }

    /// Mark a tensor's shard as containing host-resident bytes. Called
    /// from the host-resident NVFP4 / BF16 branches inside the loader.
    fn mark_host_resident_shard(&self, tensor: &TensorInfo) {
        self.host_resident_shards
            .borrow_mut()
            .insert(tensor.shard_path.clone());
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
        if tensor.dtype != TensorDType::BF16 || tensor.shape.len() != 2 {
            return Err(AegisError::InvalidPlan(format!(
                "`{}` must be a BF16 matrix",
                tensor.name
            )));
        }
        let is_host_resident = matches!(residency, TensorResidencyPlan::StagedHostToDevice { .. });

        // Both branches read through the shared shard mmap (cached in
        // `TensorStorageLoader`). The shard mmap IS the storage — host-
        // resident weights borrow into it directly (file-backed pages,
        // counted as "Cached", evictable under pressure), and VRAM-
        // resident uploads memcpy from it once. No anonymous-RAM
        // pinned copy, no per-tensor `cuMemAllocHost`, no bounce buffer
        // — the load-time peak collapses to whatever the kernel chooses
        // to keep in page cache, which is reclaimable.
        let loaded = loader.load_for_store(tensor, StoragePlacement::Mmap)?;

        if is_host_resident {
            // Synchronously fault every page so "model loaded" actually
            // means the BF16 matrix is in RAM, not lazy-on-first-access.
            loaded.prefault();
            // Mark the shard so the executor can `cuMemHostRegister`
            // it after load — restores direct-DMA from the mmap on
            // subsequent inference accesses.
            self.mark_host_resident_shard(tensor);
            let stub = self
                .runtime
                .stream
                .clone_htod(&[0u16])
                .map_err(map_cuda_err("htod bf16 host-resident stub"))?;
            let host_weights = HostBf16Weights::from_loaded(loaded)?;
            return Ok(DeviceBf16Matrix {
                name: tensor.name.clone(),
                rows: tensor.shape[0],
                cols: tensor.shape[1],
                residency,
                values: stub,
                host_values: Some(Box::new(host_weights)),
            });
        }

        // VRAM-resident BF16: H2D from the shard mmap straight into a
        // fresh u16-typed VRAM buffer. The driver pays a one-time
        // pageable→pinned bounce internally; that's fine for load-time
        // (vs the prior 1.4 GiB pinned-anon bounce buffer). After this
        // call returns, the loader's drop hooks call
        // `posix_fadvise(DONTNEED)` on the touched range so the kernel
        // can reclaim the page-cache pages immediately.
        let len_bytes = tensor.data_len_bytes() as usize;
        if len_bytes % 2 != 0 {
            return Err(AegisError::InvalidPlan(format!(
                "BF16 tensor `{}` has odd byte length {}", tensor.name, len_bytes
            )));
        }
        let len_u16 = len_bytes / 2;
        let bytes = loaded.as_bytes();
        // Reinterpret as &[u16] — safetensors aligns tensor data to 8 bytes,
        // so 2-byte alignment of the file offset is guaranteed.
        if (bytes.as_ptr() as usize) % 2 != 0 {
            return Err(AegisError::InvalidPlan(format!(
                "BF16 mmap source for `{}` is not 2-byte aligned",
                tensor.name
            )));
        }
        // SAFETY: bytes from a shard mmap, aligned check above; len_u16 * 2 == len_bytes.
        let src_u16: &[u16] = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const u16, len_u16)
        };
        let mut buffer = self.runtime.alloc_u16(len_u16)?;
        self.runtime
            .stream
            .memcpy_htod(src_u16, &mut buffer.slice)
            .map_err(map_cuda_err("htod bf16 matrix from mmap"))?;
        // The `memcpy_htod` is synchronous from a pageable source (driver
        // bounces internally and waits), so by the time we return the
        // VRAM buffer is fully populated and `loaded` can drop safely.
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
            // Host-resident weights point straight into the shard mmap
            // (cached in `TensorStorageLoader`). The mmap IS the
            // storage — file-backed pages, counted as "Cached" (not
            // RSS), reclaimable under memory pressure. Per-token
            // inference does a CPU memcpy mmap → per-slot pinned
            // bounce → DMA (handled by the staging pool) instead of
            // the prior zero-copy arena DMA, which is slightly slower
            // but eliminates the ~12 GiB anonymous-RAM pinned arena
            // that used to make peak host RAM scale with model size.
            let packed_loaded = loader.load_for_store(weight, StoragePlacement::Mmap)?;
            let scales_loaded = loader.load_for_store(scales, StoragePlacement::Mmap)?;
            // Synchronously fault every page in. Without this, the mmap
            // pages stay un-faulted until first inference access, so
            // "model loaded" wouldn't actually mean "all weights are in
            // RAM" — the first request would block on disk reads.
            packed_loaded.prefault();
            scales_loaded.prefault();
            // Mark the shards so the executor can `cuMemHostRegister`
            // them after load — restores direct-DMA from the mmap on
            // per-token expert streaming.
            self.mark_host_resident_shard(weight);
            self.mark_host_resident_shard(scales);
            let (mmap_packed, mmap_scales) = (
                HostWeightBytes::Mmap(packed_loaded),
                HostWeightBytes::Mmap(scales_loaded),
            );
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

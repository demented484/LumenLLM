use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::Arc;

use memmap2::{Advice, Mmap, MmapOptions, UncheckedAdvice};
use std::os::unix::io::AsRawFd;

use crate::error::{AegisError, Result};
use crate::graph::{ModelGraph, RegionId, TensorRole};
use crate::planning::placement::{
    ComputePlacement, ResolvedPlacement, StoragePlacement, TransferPolicy,
};
use crate::tensor::{TensorDType, TensorInfo};

/// Specifies a submatrix slice to load from a nested-param (MatFormer) weight.
///
/// MatFormer-style models (Gemma 4 E2B / E4B) store nested matrices where
/// the smaller variant uses only the leading rows × leading cols of each
/// weight.  `NestedParamSlice` describes the slice to extract.
///
/// Supported layouts:
/// - **Row slice**: `cols_end == None` — take leading `rows_end` rows,
///   all columns (e.g. output projection select-columns).
/// - **Column slice**: `rows_end == None` — take all rows, leading `cols_end`
///   columns (e.g. input projection select-features).
/// - **Submatrix**: both set — take the leading `rows_end × cols_end` block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NestedParamSlice {
    pub rows_end: Option<usize>,
    pub cols_end: Option<usize>,
}

impl NestedParamSlice {
    /// Take only the leading `rows` rows (all columns).
    pub fn rows(rows: usize) -> Self {
        Self { rows_end: Some(rows), cols_end: None }
    }

    /// Take only the leading `cols` columns (all rows).
    pub fn cols(cols: usize) -> Self {
        Self { rows_end: None, cols_end: Some(cols) }
    }

    /// Take the leading `rows × cols` submatrix.
    pub fn submatrix(rows: usize, cols: usize) -> Self {
        Self { rows_end: Some(rows), cols_end: Some(cols) }
    }

    /// Compute the effective shape and byte count for this slice applied to `full_shape`.
    /// Returns `(effective_rows, effective_cols, byte_count)`.
    pub fn effective_shape(
        &self,
        full_shape: &[usize],
        dtype: TensorDType,
    ) -> Result<(usize, usize, usize)> {
        if full_shape.len() != 2 {
            return Err(AegisError::InvalidPlan(format!(
                "NestedParamSlice requires a 2-D tensor, got shape {:?}",
                full_shape
            )));
        }
        let full_rows = full_shape[0];
        let full_cols = full_shape[1];
        let eff_rows = self.rows_end.map_or(full_rows, |r| r.min(full_rows));
        let eff_cols = self.cols_end.map_or(full_cols, |c| c.min(full_cols));
        if eff_rows == 0 || eff_cols == 0 {
            return Err(AegisError::InvalidPlan(
                "NestedParamSlice would produce an empty tensor".into(),
            ));
        }
        let bytes_per_element = dtype.bytes_per_element();
        let byte_count = eff_rows * eff_cols * bytes_per_element;
        Ok((eff_rows, eff_cols, byte_count))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StoragePlan {
    pub tensors: Vec<TensorStoragePlan>,
    pub totals: StorageTotals,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorStoragePlan {
    pub name: String,
    pub region_id: RegionId,
    pub role: TensorRole,
    pub bytes: u64,
    pub store: StoragePlacement,
    pub compute: ComputePlacement,
    pub residency: TensorResidencyPlan,
    pub transfer: TransferPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorResidencyPlan {
    RamResident,
    FileBackedMmap,
    VramResident {
        device: usize,
    },
    StagedHostToDevice {
        device: usize,
    },
    StagedDeviceToHost {
        device: usize,
    },
    CrossDevice {
        store_device: usize,
        compute_device: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StorageTotals {
    pub ram_resident_bytes: u64,
    pub mmap_file_backed_bytes: u64,
    pub vram_resident_bytes: Vec<(usize, u64)>,
    pub staged_host_to_device_peak_bytes: Vec<(usize, u64)>,
}

#[derive(Debug, Default)]
pub struct TensorStorageLoader {
    mmaps: BTreeMap<PathBuf, Arc<Mmap>>,
}

#[derive(Debug, Clone)]
pub struct LoadedHostTensor {
    pub name: String,
    pub storage: HostTensorStorage,
    /// Path to the safetensors shard backing this tensor. Used by
    /// `advise_release_pages` to call `posix_fadvise(POSIX_FADV_DONTNEED)` on
    /// the file's byte range so the kernel evicts the just-uploaded pages
    /// from the page cache instead of letting them accumulate until the
    /// system hits memory pressure (which manifests as a sawtooth in
    /// `MemAvailable` during model load).
    pub shard_path: PathBuf,
}

#[derive(Debug, Clone)]
pub enum HostTensorStorage {
    Ram(Vec<u8>),
    Mmap {
        map: Arc<Mmap>,
        offset: usize,
        len: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostLoadMode {
    RamResident,
    MmapFileBacked,
}

impl StoragePlan {
    pub fn from_graph_and_placement(graph: &ModelGraph, placement: &ResolvedPlacement) -> Self {
        let region_placements = placement.region_map();
        let mut tensors = Vec::new();
        for region in &graph.regions {
            let Some(region_placement) = region_placements.get(&region.id) else {
                continue;
            };
            for tensor in &region.tensors {
                let residency = residency_for(region_placement.store, region_placement.compute);
                tensors.push(TensorStoragePlan {
                    name: tensor.info.name.clone(),
                    region_id: region.id.clone(),
                    role: tensor.role,
                    bytes: tensor.info.data_len_bytes(),
                    store: region_placement.store,
                    compute: region_placement.compute,
                    residency,
                    transfer: region_placement.transfer,
                });
            }
        }
        let totals = StorageTotals::from_tensors(&tensors);
        Self { tensors, totals }
    }

    pub fn tensors_in_region<'a>(
        &'a self,
        region_id: &'a RegionId,
    ) -> impl Iterator<Item = &'a TensorStoragePlan> + 'a {
        self.tensors
            .iter()
            .filter(move |tensor| &tensor.region_id == region_id)
    }
}

impl TensorStorageLoader {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn load_for_store(
        &mut self,
        tensor: &TensorInfo,
        store: StoragePlacement,
    ) -> Result<LoadedHostTensor> {
        let mode = match store {
            StoragePlacement::Ram => HostLoadMode::RamResident,
            StoragePlacement::Mmap | StoragePlacement::Vram { .. } => HostLoadMode::MmapFileBacked,
        };
        self.load(tensor, mode)
    }

    pub fn load(&mut self, tensor: &TensorInfo, mode: HostLoadMode) -> Result<LoadedHostTensor> {
        let storage = match mode {
            HostLoadMode::RamResident => HostTensorStorage::Ram(read_tensor_bytes(tensor)?),
            HostLoadMode::MmapFileBacked => {
                let map = self.mmap_shard(&tensor.shard_path)?;
                HostTensorStorage::Mmap {
                    map,
                    offset: tensor.file_offsets.0 as usize,
                    len: tensor.data_len_bytes() as usize,
                }
            }
        };
        Ok(LoadedHostTensor {
            name: tensor.name.clone(),
            storage,
            shard_path: tensor.shard_path.clone(),
        })
    }

    /// Load a submatrix slice of a 2-D weight tensor (MatFormer / nested-param).
    ///
    /// Reads only the bytes that belong to the effective rows × cols block.
    /// If `slice` covers the full tensor, this is equivalent to a normal load.
    ///
    /// Always allocates RAM for the result (copying is required to produce a
    /// contiguous submatrix when `cols_end` < full_cols).
    pub fn load_submatrix(
        &mut self,
        tensor: &TensorInfo,
        slice: NestedParamSlice,
    ) -> Result<LoadedHostTensor> {
        let (eff_rows, eff_cols, byte_count) =
            slice.effective_shape(&tensor.shape, tensor.dtype)?;
        let full_cols = tensor.shape[1];
        let bytes_per_element = tensor.dtype.bytes_per_element();
        let full_col_bytes = full_cols * bytes_per_element;
        let eff_col_bytes = eff_cols * bytes_per_element;

        // Fast path: full rows and full cols → standard load.
        if eff_rows == tensor.shape[0] && eff_cols == full_cols {
            return self.load(tensor, HostLoadMode::RamResident);
        }

        // Read the full tensor bytes (mmap or file read), then extract the submatrix.
        let map = self.mmap_shard(&tensor.shard_path)?;
        let file_start = tensor.file_offsets.0 as usize;
        let data_start = file_start + tensor.data_offsets.0 as usize;

        let mut out = Vec::with_capacity(byte_count);
        for row in 0..eff_rows {
            let row_offset = data_start + row * full_col_bytes;
            out.extend_from_slice(&map[row_offset..row_offset + eff_col_bytes]);
        }
        Ok(LoadedHostTensor {
            name: tensor.name.clone(),
            storage: HostTensorStorage::Ram(out),
            shard_path: tensor.shard_path.clone(),
        })
    }

    fn mmap_shard(&mut self, path: &PathBuf) -> Result<Arc<Mmap>> {
        if let Some(map) = self.mmaps.get(path) {
            return Ok(map.clone());
        }
        let file = File::open(path)?;
        let map = Arc::new(unsafe { MmapOptions::new().map(&file)? });
        self.mmaps.insert(path.clone(), map.clone());
        Ok(map)
    }

    /// Force the kernel to drop every page the loader paged in. Linux's
    /// `MADV_DONTNEED` on a file-backed mmap removes the mapping from this
    /// process's page tables but does NOT reliably evict the page-cache
    /// entries — the kernel keeps them around in case any other process (or
    /// a future open) needs them. After loading a 14 GiB model, that's 14
    /// GiB of "Cached" memory that's effectively dead weight on memory-tight
    /// hosts (32 GiB systems, OS, IDE, etc.). `posix_fadvise(DONTNEED)` on
    /// the file descriptor is the canonical knob to evict file-backed cache
    /// pages: it tells the kernel "this is the last access for this range,
    /// you can drop it". We re-`File::open` per shard since the original
    /// `File` was consumed by `MmapOptions::map`; the open is cheap.
    pub fn release_page_cache(&self) {
        for path in self.mmaps.keys() {
            if let Ok(file) = File::open(path) {
                let len = file.metadata().map(|m| m.len()).unwrap_or(0);
                if len > 0 {
                    fadvise_dont_need(&file, 0, len);
                }
            }
        }
    }
}

impl Drop for TensorStorageLoader {
    fn drop(&mut self) {
        // Best-effort page-cache eviction at the end of weight loading.
        // Safe to skip on errors (we just keep the cache; not fatal).
        self.release_page_cache();
    }
}

impl LoadedHostTensor {
    pub fn as_bytes(&self) -> &[u8] {
        self.storage.as_bytes()
    }

    pub fn len(&self) -> usize {
        self.as_bytes().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl HostTensorStorage {
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Ram(bytes) => bytes,
            Self::Mmap { map, offset, len } => &map[*offset..*offset + *len],
        }
    }
}

impl LoadedHostTensor {
    /// Hint the kernel to drop the pages backing this tensor from the page cache.
    /// Called automatically on `Drop` for mmap-backed tensors so that streaming
    /// many GB of weights from disk into VRAM does not balloon the page cache and
    /// trigger the OOM killer on memory-tight hosts.
    ///
    /// Two-step: `MADV_DONTNEED` removes the pages from this process's VMA, then
    /// `POSIX_FADV_DONTNEED` on the underlying shard file evicts them from the
    /// kernel's page cache too. Without the second call, file-backed clean
    /// pages stay in the cache indefinitely (until OS-wide memory pressure),
    /// producing a sawtooth in `MemAvailable` during model load.
    pub fn advise_release_pages(&self) {
        if let HostTensorStorage::Mmap { map, offset, len } = &self.storage {
            // SAFETY: the underlying mmap is a read-only mapping of an immutable
            // safetensors shard; MADV_DONTNEED only drops cached pages, no data is lost.
            unsafe {
                let _ = map.unchecked_advise_range(UncheckedAdvice::DontNeed, *offset, *len);
            }
            if let Ok(file) = File::open(&self.shard_path) {
                fadvise_dont_need(&file, *offset as u64, *len as u64);
            }
        }
    }

    /// Tell the kernel we will need these pages soon — schedules async readahead
    /// from disk into the page cache. Used for long-term host-resident weights
    /// (mmap-backed) so that inference doesn't pay page-fault cost on first access.
    pub fn advise_prefault(&self) {
        if let HostTensorStorage::Mmap { map, offset, len } = &self.storage {
            let _ = map.advise_range(Advice::WillNeed, *offset, *len);
        }
    }
}

impl Drop for LoadedHostTensor {
    fn drop(&mut self) {
        self.advise_release_pages();
    }
}

/// Hint the kernel to drop pages in `[offset, offset+len)` of `file` from the page cache.
/// Used after `read_exact` so that one-shot bulk reads (weights → pinned RAM / VRAM) do
/// not balloon the page cache and trigger the OOM killer on memory-tight hosts.
pub fn fadvise_dont_need(file: &File, offset: u64, len: u64) {
    let fd = file.as_raw_fd();
    unsafe {
        libc::posix_fadvise(
            fd,
            offset as libc::off_t,
            len as libc::off_t,
            libc::POSIX_FADV_DONTNEED,
        );
    }
}

impl StorageTotals {
    fn from_tensors(tensors: &[TensorStoragePlan]) -> Self {
        let mut totals = Self::default();
        for tensor in tensors {
            match tensor.store {
                StoragePlacement::Ram => {
                    totals.ram_resident_bytes += tensor.bytes;
                }
                StoragePlacement::Mmap => {
                    totals.mmap_file_backed_bytes += tensor.bytes;
                }
                StoragePlacement::Vram { device } => {
                    add_bytes(&mut totals.vram_resident_bytes, device, tensor.bytes);
                }
            }
            match tensor.residency {
                TensorResidencyPlan::StagedHostToDevice { device } => {
                    set_peak(
                        &mut totals.staged_host_to_device_peak_bytes,
                        device,
                        tensor.bytes,
                    );
                }
                TensorResidencyPlan::RamResident
                | TensorResidencyPlan::FileBackedMmap
                | TensorResidencyPlan::VramResident { .. }
                | TensorResidencyPlan::StagedDeviceToHost { .. }
                | TensorResidencyPlan::CrossDevice { .. } => {}
            }
        }
        totals
    }
}

fn residency_for(store: StoragePlacement, compute: ComputePlacement) -> TensorResidencyPlan {
    // Treat Cuda and Wgpu identically here: both are GPU-class devices
    // with VRAM-resident or staged-host-to-device residency. The actual
    // backend is chosen at executor construction; the residency plan just
    // describes where bytes live and whether they need staging.
    match (store, compute) {
        (StoragePlacement::Ram, ComputePlacement::Cpu) => TensorResidencyPlan::RamResident,
        (StoragePlacement::Mmap, ComputePlacement::Cpu) => TensorResidencyPlan::FileBackedMmap,
        (StoragePlacement::Vram { device: store_device },
         ComputePlacement::Cuda { device } | ComputePlacement::Wgpu { device })
            if store_device == device =>
        {
            TensorResidencyPlan::VramResident { device }
        }
        (StoragePlacement::Ram | StoragePlacement::Mmap,
         ComputePlacement::Cuda { device } | ComputePlacement::Wgpu { device }) =>
        {
            TensorResidencyPlan::StagedHostToDevice { device }
        }
        (StoragePlacement::Vram { device }, ComputePlacement::Cpu) => {
            TensorResidencyPlan::StagedDeviceToHost { device }
        }
        (StoragePlacement::Vram { device: store_device },
         ComputePlacement::Cuda { device: compute_device }
         | ComputePlacement::Wgpu { device: compute_device }) =>
        {
            TensorResidencyPlan::CrossDevice { store_device, compute_device }
        }
    }
}

fn add_bytes(values: &mut Vec<(usize, u64)>, device: usize, bytes: u64) {
    if let Some((_, total)) = values.iter_mut().find(|(entry, _)| *entry == device) {
        *total += bytes;
    } else {
        values.push((device, bytes));
    }
}

fn set_peak(values: &mut Vec<(usize, u64)>, device: usize, bytes: u64) {
    if let Some((_, peak)) = values.iter_mut().find(|(entry, _)| *entry == device) {
        *peak = (*peak).max(bytes);
    } else {
        values.push((device, bytes));
    }
}

fn read_tensor_bytes(tensor: &TensorInfo) -> Result<Vec<u8>> {
    let len = tensor.data_len_bytes() as usize;
    let mut file = File::open(&tensor.shard_path)?;
    file.seek(SeekFrom::Start(tensor.file_offsets.0))?;
    let mut bytes = vec![0_u8; len];
    file.read_exact(&mut bytes)?;
    fadvise_dont_need(&file, tensor.file_offsets.0, len as u64);
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::NestedParamSlice;
    use crate::tensor::TensorDType;

    #[test]
    fn nested_param_slice_row_only() {
        let s = NestedParamSlice::rows(2);
        let (r, c, b) = s.effective_shape(&[4, 8], TensorDType::F32).unwrap();
        assert_eq!(r, 2);
        assert_eq!(c, 8);
        assert_eq!(b, 2 * 8 * 4);
    }

    #[test]
    fn nested_param_slice_col_only() {
        let s = NestedParamSlice::cols(4);
        let (r, c, b) = s.effective_shape(&[8, 16], TensorDType::F16).unwrap();
        assert_eq!(r, 8);
        assert_eq!(c, 4);
        assert_eq!(b, 8 * 4 * 2);
    }

    #[test]
    fn nested_param_slice_submatrix() {
        let s = NestedParamSlice::submatrix(3, 5);
        let (r, c, b) = s.effective_shape(&[6, 10], TensorDType::F32).unwrap();
        assert_eq!(r, 3);
        assert_eq!(c, 5);
        assert_eq!(b, 3 * 5 * 4);
    }

    #[test]
    fn nested_param_slice_clamps_to_full_size() {
        // slice larger than tensor → clamp to full
        let s = NestedParamSlice::submatrix(100, 100);
        let (r, c, _) = s.effective_shape(&[4, 8], TensorDType::F32).unwrap();
        assert_eq!(r, 4);
        assert_eq!(c, 8);
    }

    #[test]
    fn nested_param_slice_rejects_non_2d() {
        let s = NestedParamSlice::rows(2);
        assert!(s.effective_shape(&[4, 8, 2], TensorDType::F32).is_err());
        assert!(s.effective_shape(&[4], TensorDType::F32).is_err());
    }
}

use cudarc::driver::PinnedHostSlice;
use cudarc::driver::result::{device as cu_device, mem_pool as cu_mempool};
use cudarc::driver::sys::CUmemPool_attribute;

use super::{CudaRuntime, map_cuda_err};
use crate::cuda::DeviceBuffer;
use aegisllm_base::error::{AegisError, Result};

/// Snapshot of the device's default async-mem-pool occupancy. Both numbers
/// are in bytes. `reserved_current` is the total VRAM the pool has
/// requested from the driver (including unused-but-cached blocks).
/// `used_current` is what's actually live for callers.
#[derive(Debug, Clone, Copy)]
pub struct MemPoolStats {
    pub reserved_current: usize,
    pub used_current: usize,
}

impl MemPoolStats {
    pub fn cached_bytes(&self) -> usize {
        self.reserved_current.saturating_sub(self.used_current)
    }
}

/// Read the device's default mempool occupancy. Used to diagnose VRAM gaps
/// where the pool grows above peak-live usage during loading and never
/// shrinks back (the classic cause of "we save 800 MiB by quantizing but
/// only see 400 MiB drop in nvidia-smi").
pub fn read_mempool_stats(device_index: usize) -> Result<MemPoolStats> {
    let dev = cu_device::get(device_index as i32)
        .map_err(map_cuda_err("cuDeviceGet for mempool stats"))?;
    let pool = unsafe {
        cu_device::get_default_mem_pool(dev)
            .map_err(map_cuda_err("cuDeviceGetDefaultMemPool"))?
    };
    let mut reserved: u64 = 0;
    let mut used: u64 = 0;
    unsafe {
        cu_mempool::get_attribute(
            pool,
            CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
            &mut reserved as *mut u64 as *mut _,
        )
        .map_err(map_cuda_err("cuMemPoolGetAttribute reserved"))?;
        cu_mempool::get_attribute(
            pool,
            CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
            &mut used as *mut u64 as *mut _,
        )
        .map_err(map_cuda_err("cuMemPoolGetAttribute used"))?;
    }
    Ok(MemPoolStats {
        reserved_current: reserved as usize,
        used_current: used as usize,
    })
}

/// Release unused blocks in the device's default async-mempool back to the
/// driver. Pass `min_bytes_to_keep = 0` to release everything that isn't
/// currently allocated. Safe and effectively free (no-op if the pool is
/// already trimmed). Should be called after the load phase peaks and
/// before steady-state inference begins.
pub fn trim_default_mempool(device_index: usize, min_bytes_to_keep: usize) -> Result<()> {
    let dev = cu_device::get(device_index as i32)
        .map_err(map_cuda_err("cuDeviceGet for mempool trim"))?;
    let pool = unsafe {
        cu_device::get_default_mem_pool(dev)
            .map_err(map_cuda_err("cuDeviceGetDefaultMemPool"))?
    };
    unsafe {
        cu_mempool::trim_to(pool, min_bytes_to_keep)
            .map_err(map_cuda_err("cuMemPoolTrimTo"))?;
    }
    Ok(())
}

impl CudaRuntime {
    pub fn alloc_f32(&self, len: usize) -> Result<DeviceBuffer<f32>> {
        // Production kernels fully write these state/scratch buffers before reading them.
        Ok(DeviceBuffer {
            slice: unsafe { self.stream.alloc::<f32>(len) }
                .map_err(map_cuda_err("alloc cuda f32 buffer"))?,
        })
    }

    pub fn alloc_u16(&self, len: usize) -> Result<DeviceBuffer<u16>> {
        Ok(DeviceBuffer {
            slice: unsafe { self.stream.alloc::<u16>(len) }
                .map_err(map_cuda_err("alloc cuda u16 buffer"))?,
        })
    }

    pub fn alloc_u8(&self, len: usize) -> Result<DeviceBuffer<u8>> {
        Ok(DeviceBuffer {
            slice: unsafe { self.stream.alloc::<u8>(len) }
                .map_err(map_cuda_err("alloc cuda u8 buffer"))?,
        })
    }

    pub fn alloc_u32(&self, len: usize) -> Result<DeviceBuffer<u32>> {
        Ok(DeviceBuffer {
            slice: unsafe { self.stream.alloc::<u32>(len) }
                .map_err(map_cuda_err("alloc cuda u32 buffer"))?,
        })
    }

    pub fn alloc_u64(&self, len: usize) -> Result<DeviceBuffer<u64>> {
        Ok(DeviceBuffer {
            slice: unsafe { self.stream.alloc::<u64>(len) }
                .map_err(map_cuda_err("alloc cuda u64 buffer"))?,
        })
    }

    pub fn upload_u64_slice_to_device(
        &self,
        values: &[u64],
        buffer: &mut DeviceBuffer<u64>,
    ) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }
        if buffer.len() < values.len() {
            return Err(AegisError::InvalidPlan(format!(
                "upload_u64_slice_to_device buffer too small: have {}, need {}",
                buffer.len(),
                values.len(),
            )));
        }
        let mut dst = buffer.slice.slice_mut(0..values.len());
        self.stream
            .memcpy_htod(values, &mut dst)
            .map_err(map_cuda_err("htod u64 slice"))
    }

    pub fn upload_f32(&self, values: &[f32]) -> Result<DeviceBuffer<f32>> {
        Ok(DeviceBuffer {
            slice: self
                .stream
                .clone_htod(values)
                .map_err(map_cuda_err("htod f32 buffer"))?,
        })
    }

    /// Upload a u16 slice (used for f16-bits KV caches) to a fresh device
    /// buffer. Used by the attention-reference correctness smoke.
    pub fn upload_u16(&self, values: &[u16]) -> Result<DeviceBuffer<u16>> {
        Ok(DeviceBuffer {
            slice: self
                .stream
                .clone_htod(values)
                .map_err(map_cuda_err("htod u16 buffer"))?,
        })
    }

    pub fn copy_u32_to_device(&self, values: &[u32], buffer: &mut DeviceBuffer<u32>) -> Result<()> {
        self.stream
            .memcpy_htod(values, &mut buffer.slice)
            .map_err(map_cuda_err("htod u32 buffer"))
    }

    /// Upload an arbitrary-length f32 slice into the head of an existing device
    /// buffer (length need not equal the buffer's full capacity). Used by
    /// chunked MoE prefill to push per-expert routing weight slices.
    pub fn upload_f32_slice_to_device(
        &self,
        values: &[f32],
        buffer: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }
        if buffer.len() < values.len() {
            return Err(AegisError::InvalidPlan(format!(
                "upload_f32_slice_to_device buffer too small: have {}, need {}",
                buffer.len(),
                values.len(),
            )));
        }
        let mut dst = buffer.slice.slice_mut(0..values.len());
        self.stream
            .memcpy_htod(values, &mut dst)
            .map_err(map_cuda_err("htod f32 slice"))
    }

    /// Same as `copy_u32_to_device` but accepts shorter slices than the buffer's
    /// capacity (only the prefix is overwritten).
    pub fn upload_u32_slice_to_device(
        &self,
        values: &[u32],
        buffer: &mut DeviceBuffer<u32>,
    ) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }
        if buffer.len() < values.len() {
            return Err(AegisError::InvalidPlan(format!(
                "upload_u32_slice_to_device buffer too small: have {}, need {}",
                buffer.len(),
                values.len(),
            )));
        }
        let mut dst = buffer.slice.slice_mut(0..values.len());
        self.stream
            .memcpy_htod(values, &mut dst)
            .map_err(map_cuda_err("htod u32 slice"))
    }

    pub fn download_f32(&self, buffer: &DeviceBuffer<f32>) -> Result<Vec<f32>> {
        self.stream
            .clone_dtoh(&buffer.slice)
            .map_err(map_cuda_err("dtoh f32 buffer"))
    }

    pub fn download_u32(&self, buffer: &DeviceBuffer<u32>) -> Result<Vec<u32>> {
        self.stream
            .clone_dtoh(&buffer.slice)
            .map_err(map_cuda_err("dtoh u32 buffer"))
    }

    pub fn download_u8(&self, buffer: &DeviceBuffer<u8>) -> Result<Vec<u8>> {
        self.stream
            .clone_dtoh(&buffer.slice)
            .map_err(map_cuda_err("dtoh u8 buffer"))
    }

    pub fn upload_u8_slice_to_device(
        &self,
        values: &[u8],
        buffer: &mut DeviceBuffer<u8>,
    ) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }
        if buffer.len() < values.len() {
            return Err(AegisError::InvalidPlan(format!(
                "upload_u8_slice_to_device buffer too small: have {}, need {}",
                buffer.len(),
                values.len()
            )));
        }
        let mut dst = buffer.slice.slice_mut(0..values.len());
        self.stream
            .memcpy_htod(values, &mut dst)
            .map_err(map_cuda_err("htod u8 slice"))
    }

    pub fn upload_u64_slice(&self, values: &[u64]) -> Result<DeviceBuffer<u64>> {
        Ok(DeviceBuffer {
            slice: self
                .stream
                .clone_htod(values)
                .map_err(map_cuda_err("htod u64 buffer"))?,
        })
    }

    /// Device-to-device copy of `len` elements from `src[src_offset..]` into
    /// `dst[dst_offset..]`. Used by the GPU router-bucket dispatch loop to
    /// pull a per-expert slice out of the bucket-sort output without going
    /// through host memory.
    pub fn copy_u32_d2d_range(
        &self,
        src: &DeviceBuffer<u32>,
        src_offset: usize,
        dst: &mut DeviceBuffer<u32>,
        dst_offset: usize,
        len: usize,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        if src_offset.saturating_add(len) > src.len() || dst_offset.saturating_add(len) > dst.len()
        {
            return Err(AegisError::InvalidPlan(format!(
                "copy_u32_d2d_range out of bounds: src.len={} src_off={} dst.len={} dst_off={} len={}",
                src.len(), src_offset, dst.len(), dst_offset, len
            )));
        }
        let src_view = src.slice.slice(src_offset..src_offset + len);
        let mut dst_view = dst.slice.slice_mut(dst_offset..dst_offset + len);
        self.stream
            .memcpy_dtod(&src_view, &mut dst_view)
            .map_err(map_cuda_err("d2d u32 range"))
    }

    /// Allocate `len` bytes of CUDA-pinned (page-locked) host memory bound
    /// to this runtime's primary context. Used by grouped MoE bulk staging
    /// for the bounce buffers.
    pub fn alloc_pinned_u8(&self, len: usize) -> Result<cudarc::driver::PinnedHostSlice<u8>> {
        unsafe { self.stream.context().alloc_pinned::<u8>(len) }
            .map_err(map_cuda_err("alloc pinned u8"))
    }

    /// Copy from a pinned host byte slice into a u8 device buffer (full
    /// length defined by `len`). Used by grouped MoE bulk staging to do
    /// one big DMA transfer per projection instead of many small ones.
    pub fn copy_pinned_u8_to_device(
        &self,
        src: &cudarc::driver::PinnedHostSlice<u8>,
        len: usize,
        dst: &mut DeviceBuffer<u8>,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        if dst.len() < len {
            return Err(AegisError::InvalidPlan(format!(
                "copy_pinned_u8_to_device dst too small: have {} need {}",
                dst.len(), len
            )));
        }
        let src_full = src
            .as_slice()
            .map_err(map_cuda_err("pinned u8 as_slice"))?;
        let src_slice = src_full.get(..len).ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "copy_pinned_u8_to_device src too small: have {} need {}",
                src_full.len(), len
            ))
        })?;
        let mut dst_view = dst.slice.slice_mut(0..len);
        self.stream
            .memcpy_htod(src_slice, &mut dst_view)
            .map_err(map_cuda_err("h2d pinned u8"))
    }

    /// Same as `copy_host_u8_to_device_at_offset` but issues the H2D on the
    /// transfer stream instead of the compute stream. Used by grouped MoE
    /// bulk staging to overlap weight uploads with kernel execution: the
    /// caller records a transfer event afterwards and makes the compute
    /// stream `wait` on that event before launching the consumer GEMM.
    pub fn copy_host_u8_to_device_at_offset_async(
        &self,
        src: &[u8],
        dst: &mut DeviceBuffer<u8>,
        dst_offset: usize,
    ) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        if dst_offset.saturating_add(src.len()) > dst.len() {
            return Err(AegisError::InvalidPlan(format!(
                "copy_host_u8_to_device_at_offset_async out of bounds: dst.len={} dst_off={} src.len={}",
                dst.len(), dst_offset, src.len()
            )));
        }
        let mut dst_view = dst.slice.slice_mut(dst_offset..dst_offset + src.len());
        self.transfer_stream
            .memcpy_htod(src, &mut dst_view)
            .map_err(map_cuda_err("h2d u8 transfer-stream"))
    }

    /// Same as `upload_u32_slice_to_device` but on the transfer stream.
    pub fn upload_u32_slice_to_device_async(
        &self,
        values: &[u32],
        buffer: &mut DeviceBuffer<u32>,
    ) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }
        if buffer.len() < values.len() {
            return Err(AegisError::InvalidPlan(format!(
                "upload_u32_slice_to_device_async buffer too small: have {} need {}",
                buffer.len(), values.len()
            )));
        }
        let mut dst = buffer.slice.slice_mut(0..values.len());
        self.transfer_stream
            .memcpy_htod(values, &mut dst)
            .map_err(map_cuda_err("htod u32 transfer-stream"))
    }

    /// Same as `upload_f32_slice_to_device` but on the transfer stream.
    pub fn upload_f32_slice_to_device_async(
        &self,
        values: &[f32],
        buffer: &mut DeviceBuffer<f32>,
    ) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }
        if buffer.len() < values.len() {
            return Err(AegisError::InvalidPlan(format!(
                "upload_f32_slice_to_device_async buffer too small: have {} need {}",
                buffer.len(), values.len()
            )));
        }
        let mut dst = buffer.slice.slice_mut(0..values.len());
        self.transfer_stream
            .memcpy_htod(values, &mut dst)
            .map_err(map_cuda_err("htod f32 transfer-stream"))
    }

    /// Copy a host byte slice into a u8 device buffer at `dst_offset`.
    /// Used by grouped MoE bulk staging to concatenate per-expert weight
    /// bytes into a single contiguous VRAM buffer.
    pub fn copy_host_u8_to_device_at_offset(
        &self,
        src: &[u8],
        dst: &mut DeviceBuffer<u8>,
        dst_offset: usize,
    ) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        if dst_offset.saturating_add(src.len()) > dst.len() {
            return Err(AegisError::InvalidPlan(format!(
                "copy_host_u8_to_device_at_offset out of bounds: dst.len={} dst_off={} src.len={}",
                dst.len(), dst_offset, src.len()
            )));
        }
        let mut dst_view = dst.slice.slice_mut(dst_offset..dst_offset + src.len());
        self.stream
            .memcpy_htod(src, &mut dst_view)
            .map_err(map_cuda_err("h2d u8 range"))
    }

    pub fn copy_f32_d2d_range(
        &self,
        src: &DeviceBuffer<f32>,
        src_offset: usize,
        dst: &mut DeviceBuffer<f32>,
        dst_offset: usize,
        len: usize,
    ) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        if src_offset.saturating_add(len) > src.len() || dst_offset.saturating_add(len) > dst.len()
        {
            return Err(AegisError::InvalidPlan(format!(
                "copy_f32_d2d_range out of bounds: src.len={} src_off={} dst.len={} dst_off={} len={}",
                src.len(), src_offset, dst.len(), dst_offset, len
            )));
        }
        let src_view = src.slice.slice(src_offset..src_offset + len);
        let mut dst_view = dst.slice.slice_mut(dst_offset..dst_offset + len);
        self.stream
            .memcpy_dtod(&src_view, &mut dst_view)
            .map_err(map_cuda_err("d2d f32 range"))
    }

    /// Allocate CUDA-pinned (page-locked) host memory for `len` u16 elements.
    /// The returned slice is zero-initialized; CPU reads are uncached (WRITECOMBINED flag
    /// is NOT used here — KV data is read by the CPU during D2H writeback).
    pub fn alloc_pinned_u16(&self, len: usize) -> Result<PinnedHostSlice<u16>> {
        unsafe { self.stream.context().alloc_pinned::<u16>(len) }
            .map_err(map_cuda_err("alloc pinned u16 for kv host"))
    }

    /// Allocate CUDA-pinned (page-locked) host memory for `len` u32 elements.
    /// Used for the decode MoE packed top-k dtoh destination so the async copy
    /// on the transfer stream takes the direct-DMA fast path.
    pub fn alloc_pinned_u32(&self, len: usize) -> Result<PinnedHostSlice<u32>> {
        unsafe { self.stream.context().alloc_pinned::<u32>(len) }
            .map_err(map_cuda_err("alloc pinned u32"))
    }

    /// Issue an async D2H copy from `src` (`u32` device buffer) into `dst`
    /// (CUDA-pinned host slice) on the **transfer stream**. Caller must record
    /// a transfer event afterwards and host-synchronize on it before reading
    /// `dst`.
    ///
    /// Internally goes through `CudaStream::memcpy_dtoh`, which on a
    /// PinnedHostSlice destination enqueues a wait on the slice's internal
    /// event (no host stall) and re-records it once the copy lands — providing
    /// the producer/consumer ordering between successive layers without us
    /// having to track an extra event per layer.
    pub fn download_u32_to_pinned_async(
        &self,
        src: &DeviceBuffer<u32>,
        dst: &mut PinnedHostSlice<u32>,
        count: usize,
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        if src.len() < count || dst.len() < count {
            return Err(AegisError::InvalidPlan(format!(
                "download_u32_to_pinned_async out of bounds: src={} dst={} count={count}",
                src.len(), dst.len(),
            )));
        }
        let src_view = src.slice.slice(0..count);
        // cudarc's memcpy_dtoh handles PinnedHostSlice destinations via
        // `stream_synced_mut_slice`: enqueues an async wait on the slice's
        // internal event (no host block) and re-records it on completion.
        // We cannot pass a `&mut [u32]` subslice because that requires
        // synchronously calling `as_mut_slice()` first (which would block).
        // Instead pass the full slice and rely on `src` length being the
        // copy-size driver (memcpy_dtoh uses `src.len()` for the byte count).
        if count != src.len() {
            // Defensive: cudarc's memcpy_dtoh sizes from src; we ensure src
            // has exactly `count` elements by using a sliced view above.
            // `src_view.len() == count` here, so this branch is unreachable
            // in practice, kept for clarity.
        }
        self.transfer_stream
            .memcpy_dtoh(&src_view, dst)
            .map_err(map_cuda_err("d2h u32 packed topk async"))
    }
}

impl<T> DeviceBuffer<T> {
    pub fn len(&self) -> usize {
        self.slice.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slice.is_empty()
    }
}

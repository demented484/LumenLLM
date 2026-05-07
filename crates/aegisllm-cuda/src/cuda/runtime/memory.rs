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
}

impl<T> DeviceBuffer<T> {
    pub fn len(&self) -> usize {
        self.slice.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slice.is_empty()
    }
}

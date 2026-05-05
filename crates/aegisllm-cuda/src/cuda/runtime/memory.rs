use cudarc::driver::PinnedHostSlice;

use super::{CudaRuntime, map_cuda_err};
use crate::cuda::DeviceBuffer;
use aegisllm_base::error::{AegisError, Result};

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

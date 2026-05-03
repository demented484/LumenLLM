use cudarc::driver::PinnedHostSlice;

use super::{CudaRuntime, map_cuda_err};
use crate::cuda::DeviceBuffer;
use aegisllm_base::error::Result;

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

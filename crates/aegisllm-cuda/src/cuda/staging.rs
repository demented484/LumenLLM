use cudarc::driver::{CudaSlice, CudaView};

use super::runtime::map_cuda_err;
use super::types::{HostResidentMxfp4, HostResidentWeights};
use aegisllm_base::error::{AegisError, Result};

/// Single VRAM staging buffer used for heterogeneous (host-resident) layers.
///
/// At decode time, each layer with `host_weights` does an H2D memcpy into this
/// buffer before its kernel launches.  Only one layer is in-flight at a time
/// (single compute stream), so a single slot is sufficient for correctness.
/// Double-buffering (transfer stream overlapping compute stream) can be added later.
pub(crate) struct LinearStagingPool {
    /// Pre-allocated VRAM: max `packed_bytes` across all staged layers.
    packed: CudaSlice<u8>,
    /// Pre-allocated VRAM: max `scale_bytes` across all staged layers.
    scales: CudaSlice<u8>,
    /// Optional native MXFP4 staging VRAM: allocated when any layer has repacked data.
    native_mxfp4: Option<CudaSlice<u8>>,
}

impl std::fmt::Debug for LinearStagingPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LinearStagingPool")
            .field("packed_cap", &self.packed.len())
            .field("scales_cap", &self.scales.len())
            .field("native_mxfp4_cap", &self.native_mxfp4.as_ref().map(|s| s.len()))
            .finish()
    }
}

impl LinearStagingPool {
    pub(crate) fn new(
        max_packed_bytes: usize,
        max_scale_bytes: usize,
        max_native_mxfp4_bytes: usize,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    ) -> Result<Self> {
        let cap_p = max_packed_bytes.max(1);
        let cap_s = max_scale_bytes.max(1);
        let packed = unsafe { stream.alloc::<u8>(cap_p) }
            .map_err(map_cuda_err("alloc staging packed buffer"))?;
        let scales = unsafe { stream.alloc::<u8>(cap_s) }
            .map_err(map_cuda_err("alloc staging scales buffer"))?;
        let native_mxfp4 = if max_native_mxfp4_bytes > 0 {
            Some(
                unsafe { stream.alloc::<u8>(max_native_mxfp4_bytes) }
                    .map_err(map_cuda_err("alloc staging native mxfp4 buffer"))?,
            )
        } else {
            None
        };
        Ok(Self { packed, scales, native_mxfp4 })
    }

    /// H2D copy host-resident weights into staging VRAM. Each `PinnedHostSlice<u8>` is
    /// passed as a whole to `memcpy_htod` so cudarc records the pinned event correctly,
    /// enabling true async DMA at full PCIe bandwidth.
    /// `packed_bytes` and `scale_bytes` are validated against the pinned slice sizes.
    pub(crate) fn prepare(
        &mut self,
        hw: &HostResidentWeights,
        packed_bytes: usize,
        scale_bytes: usize,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    ) -> Result<()> {
        if packed_bytes > self.packed.len() {
            return Err(AegisError::InvalidPlan(format!(
                "staging packed overflow: layer needs {} bytes, pool has {}",
                packed_bytes,
                self.packed.len()
            )));
        }
        if scale_bytes > self.scales.len() {
            return Err(AegisError::InvalidPlan(format!(
                "staging scales overflow: layer needs {} bytes, pool has {}",
                scale_bytes,
                self.scales.len()
            )));
        }
        if hw.packed.len() != packed_bytes || hw.scales.len() != scale_bytes {
            return Err(AegisError::InvalidPlan(format!(
                "pinned host slice size mismatch: packed expected={} got={} scales expected={} got={}",
                packed_bytes, hw.packed.len(), scale_bytes, hw.scales.len()
            )));
        }
        {
            let mut dst = self.packed.slice_mut(0..packed_bytes);
            stream
                .memcpy_htod(&hw.packed, &mut dst)
                .map_err(map_cuda_err("staging h2d packed"))?;
        }
        {
            let mut dst = self.scales.slice_mut(0..scale_bytes);
            stream
                .memcpy_htod(&hw.scales, &mut dst)
                .map_err(map_cuda_err("staging h2d scales"))?;
        }
        Ok(())
    }

    /// H2D copy native MXFP4 repacked data from pinned host buffer into staging VRAM.
    pub(crate) fn prepare_native_mxfp4(
        &mut self,
        mxfp4: &HostResidentMxfp4,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    ) -> Result<()> {
        let buf = self.native_mxfp4.as_mut().ok_or_else(|| {
            AegisError::InvalidPlan(
                "native MXFP4 staging buffer not allocated; set native_mxfp4_repack=true".into(),
            )
        })?;
        if mxfp4.data.len() > buf.len() {
            return Err(AegisError::InvalidPlan(format!(
                "staging native mxfp4 overflow: layer needs {} bytes, pool has {}",
                mxfp4.data.len(),
                buf.len()
            )));
        }
        let mut dst = buf.slice_mut(0..mxfp4.data.len());
        stream
            .memcpy_htod(&mxfp4.data, &mut dst)
            .map_err(map_cuda_err("staging h2d native mxfp4"))?;
        Ok(())
    }

    pub(crate) fn packed_view(&self, len: usize) -> CudaView<'_, u8> {
        self.packed.slice(0..len)
    }

    pub(crate) fn scales_view(&self, len: usize) -> CudaView<'_, u8> {
        self.scales.slice(0..len)
    }

    pub(crate) fn native_mxfp4_view(&self, len: usize) -> Option<CudaView<'_, u8>> {
        self.native_mxfp4.as_ref().map(|s| s.slice(0..len))
    }
}

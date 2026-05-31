use cudarc::driver::{CudaEvent, CudaSlice, CudaView};

use super::owned_pinned::OwnedPinnedBuf;
use super::runtime::{CudaRuntime, map_cuda_err};
use super::types::{HostResidentMxfp4, HostResidentWeights};
use aegisllm_base::error::{AegisError, Result};

/// Diagnostic: total bytes the staging pool has issued via H2D (packed + scales
/// + native-mxfp4). Incremented on every transfer. Read deltas around a decode
/// step to measure actual per-token streaming volume. Gated by the reader, not
/// here, so the counter is always live (a relaxed add is free).
pub(crate) static STAGING_H2D_BYTES: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

#[inline]
fn count_h2d(bytes: usize) {
    STAGING_H2D_BYTES.fetch_add(bytes as u64, std::sync::atomic::Ordering::Relaxed);
}

/// One physical staging slot in VRAM. Holds the packed/scales (and optionally
/// repacked native-MXFP4) bytes for ONE in-flight host-resident layer.
///
/// Each slot also owns a small **pinned host bounce buffer** of the same size.
/// When weights are mmap'd (unpinned), `prepare_async` first CPU-copies into
/// the bounce, then issues the async H2D from the bounce. This restores the
/// fast pinned→VRAM DMA path that vanishes when memcpy_htod is called with
/// an unpinned source (driver internal bouncing serialises and is slow for
/// many small calls).
pub(crate) struct StagingSlot {
    packed: CudaSlice<u8>,
    scales: CudaSlice<u8>,
    /// Single contiguous `[packed_cap + scales_cap]` device buffer. When the
    /// host packed+scales are adjacent in the pinned arena (the contiguous
    /// loader layout), `prepare_async` issues ONE H2D into this buffer and the
    /// consuming kernel reads packed at `[0..packed_bytes]` and scales at
    /// `[packed_bytes..]`. Halves the per-projection H2D call count on the
    /// CPU-issue-bound decode path. `packed`/`scales` above remain for the
    /// non-contiguous fallback (two copies) and the native-MXFP4 path.
    combined: CudaSlice<u8>,
    native_mxfp4: Option<CudaSlice<u8>>,
    bounce_packed: OwnedPinnedBuf,
    bounce_scales: OwnedPinnedBuf,
    bounce_native_mxfp4: Option<OwnedPinnedBuf>,
    /// Reusable compute-stream event re-recorded after every kernel that reads
    /// this slot. Transfer waits on this before overwriting. Pre-allocated
    /// once — re-recording is cheap, allocating a fresh event each call is not.
    compute_event: CudaEvent,
    /// True after the first kernel has been launched against this slot. Until
    /// then `compute_event` has no recorded workload and waiting on it is a
    /// no-op (CUDA semantics) but we skip the wait call entirely.
    primed: bool,
    /// Reusable transfer-stream event re-recorded after every H2D into this
    /// slot. Compute stream waits on this before launching its kernel.
    transfer_event: CudaEvent,
    /// When the most recent `prepare_async` used the single-copy combined path,
    /// this is `Some(packed_bytes)` — the offset at which scales begin inside
    /// `combined`. `packed_view`/`scales_view` then carve views out of
    /// `combined`. `None` means the last prepare used the two-buffer path.
    combined_packed_bytes: Option<usize>,
}

/// Double-buffered VRAM staging pool used for streaming host-resident layers
/// from pinned RAM into VRAM with H2D / compute overlap.
///
/// Layout: two physical slots (`slots[0]` and `slots[1]`). Each `prepare_async`
/// call cycles to the *other* slot, so while the compute stream reads slot N
/// the transfer stream is already filling slot N⊕1 with the next layer's
/// weights. CudaEvents enforce the only ordering constraints that matter:
///
///  * transfer stream **waits** on the slot's `last_compute_event` before
///    overwriting it (so the kernel that read it has finished).
///  * compute stream **waits** on the most recent transfer-done event before
///    launching its kernel (so the H2D is complete before the kernel reads).
///
/// This structure was previously a single-slot, single-stream pool: every
/// H2D blocked the compute stream and every kernel waited for the H2D, fully
/// serialising 700+ small transfers per token.
/// Default number of staging slots in the pool. Larger pools allow more H2D
/// transfers in flight on the transfer stream while compute eats earlier
/// slots — useful when per-matvec compute is short relative to H2D, since
/// the transfer stream can fill ahead and hide PCIe latency more aggressively.
///
/// ## Copy-engine saturation (offloaded-MoE decode)
/// At decode the routed experts of every MoE layer stream `top_k × 3`
/// (gate/up/down) tiny H2D transfers from RAM. With only 4 slots, slot `i` is
/// reused after 4 transfers, so the 5th projection's H2D must `transfer_wait`
/// on the 1st projection's GEMV completion (slot WAR fence) — the transfer
/// stream stalls on compute *within the same token*, idling the copy engine
/// between bursts and capping PCIe at ~28 GB/s on a ~56 GB/s link.
///
/// Raising the slot count to ≥ `top_k × 3` gives every projection of a layer
/// its own slot, so all of a layer's H2Ds issue back-to-back on the transfer
/// stream with NO same-token compute fence between them — the copy engine
/// sees one saturated burst per layer. Cross-layer WAR fences still apply (a
/// slot reused next layer waits on the prior layer's GEMV) but by then that
/// GEMV is long done, so the wait is a no-op. Output is bit-identical: same
/// slots, same kernels, same order — only the *number* of physical slots
/// changes the overlap, never the math.
///
/// Opt-in via `AEGIS_STAGING_SLOTS=<n>` (default 4 → unchanged behaviour).
const DEFAULT_STAGING_SLOT_COUNT: usize = 4;

/// Resolve the configured slot count from `AEGIS_STAGING_SLOTS`, falling back
/// to [`DEFAULT_STAGING_SLOT_COUNT`]. Clamped to ≥ 2 (double-buffering is the
/// minimum for any transfer/compute overlap) and a sane upper bound so a typo
/// can't allocate gigabytes of staging.
fn configured_slot_count() -> usize {
    std::env::var("AEGIS_STAGING_SLOTS")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .map(|n| n.clamp(2, 64))
        .unwrap_or(DEFAULT_STAGING_SLOT_COUNT)
}

pub(crate) struct LinearStagingPool {
    slots: Vec<StagingSlot>,
    /// Index of the slot the next `prepare_async` call will write into.
    next_slot: usize,
    /// Slot index returned from the most recent `prepare_async`. The caller is
    /// expected to launch a kernel reading that slot then call
    /// `mark_kernel_launched` so this pool can record the post-kernel event.
    last_prepared_slot: Option<usize>,
}

impl std::fmt::Debug for LinearStagingPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LinearStagingPool")
            .field("slot_count", &self.slots.len())
            .field("slot_packed_cap", &self.slots[0].packed.len())
            .field("slot_scales_cap", &self.slots[0].scales.len())
            .field(
                "native_mxfp4_cap",
                &self.slots[0].native_mxfp4.as_ref().map(|s| s.len()),
            )
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
        let alloc_slot = || -> Result<StagingSlot> {
            let packed = unsafe { stream.alloc::<u8>(cap_p) }
                .map_err(map_cuda_err("alloc staging packed buffer"))?;
            let scales = unsafe { stream.alloc::<u8>(cap_s) }
                .map_err(map_cuda_err("alloc staging scales buffer"))?;
            let combined = unsafe { stream.alloc::<u8>(cap_p + cap_s) }
                .map_err(map_cuda_err("alloc staging combined buffer"))?;
            let bounce_packed = OwnedPinnedBuf::new(cap_p)?;
            let bounce_scales = OwnedPinnedBuf::new(cap_s)?;
            let (native_mxfp4, bounce_native_mxfp4) = if max_native_mxfp4_bytes > 0 {
                let dev = unsafe { stream.alloc::<u8>(max_native_mxfp4_bytes) }
                    .map_err(map_cuda_err("alloc staging native mxfp4 buffer"))?;
                let bounce = OwnedPinnedBuf::new(max_native_mxfp4_bytes)?;
                (Some(dev), Some(bounce))
            } else {
                (None, None)
            };
            let compute_event = stream
                .context()
                .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING))
                .map_err(map_cuda_err("alloc staging compute event"))?;
            let transfer_event = stream
                .context()
                .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING))
                .map_err(map_cuda_err("alloc staging transfer event"))?;
            Ok(StagingSlot {
                packed,
                scales,
                combined,
                native_mxfp4,
                bounce_packed,
                bounce_scales,
                bounce_native_mxfp4,
                compute_event,
                primed: false,
                transfer_event,
                combined_packed_bytes: None,
            })
        };
        let slot_count = configured_slot_count();
        let mut slots: Vec<StagingSlot> = Vec::with_capacity(slot_count);
        for _ in 0..slot_count {
            slots.push(alloc_slot()?);
        }
        Ok(Self {
            slots,
            next_slot: 0,
            last_prepared_slot: None,
        })
    }

    /// Stage one host-resident linear's weights into the next free slot. Returns
    /// the slot index — callers must use the `*_view(slot)` accessors with this
    /// index when launching the consuming kernel, then call
    /// `mark_kernel_launched(slot)` immediately after the launch.
    ///
    /// Internals:
    ///   * if the slot was previously read by a kernel, transfer stream waits
    ///     for that kernel's compute event before re-using the slot.
    ///   * H2D copies are issued on the **transfer stream**, not the compute
    ///     stream, so they can overlap with kernels still reading the *other*
    ///     slot.
    ///   * after issuing H2Ds, a transfer event is recorded and the compute
    ///     stream is told to wait on it before launching consuming kernels.
    pub(crate) fn prepare_async(
        &mut self,
        runtime: &CudaRuntime,
        hw: &HostResidentWeights,
        packed_bytes: usize,
        scale_bytes: usize,
    ) -> Result<usize> {
        let slot_idx = self.next_slot;
        self.next_slot = (self.next_slot + 1) % self.slots.len();
        let slot = &mut self.slots[slot_idx];

        if packed_bytes > slot.packed.len() {
            return Err(AegisError::InvalidPlan(format!(
                "staging packed overflow: layer needs {} bytes, slot has {}",
                packed_bytes,
                slot.packed.len()
            )));
        }
        if scale_bytes > slot.scales.len() {
            return Err(AegisError::InvalidPlan(format!(
                "staging scales overflow: layer needs {} bytes, slot has {}",
                scale_bytes,
                slot.scales.len()
            )));
        }
        if hw.packed.len() != packed_bytes || hw.scales.len() != scale_bytes {
            return Err(AegisError::InvalidPlan(format!(
                "pinned host slice size mismatch: packed expected={} got={} scales expected={} got={}",
                packed_bytes,
                hw.packed.len(),
                scale_bytes,
                hw.scales.len()
            )));
        }

        count_h2d(packed_bytes + scale_bytes);

        // Block transfer stream until the kernel that previously read this slot
        // has completed. Without this the H2D could clobber bytes mid-read.
        // Skip on the very first use (no kernel has read this slot yet).
        if slot.primed {
            runtime.transfer_wait_event(&slot.compute_event)?;
        }

        // ── Single-copy combined path (decode CPU-op-count fix) ──────────────
        //
        // When packed+scales are physically adjacent in the same pinned arena
        // (the contiguous loader layout), issue ONE `cuMemcpyHtoDAsync` for the
        // whole `[packed || scales]` block into this slot's `combined` device
        // buffer. The consuming kernel then reads packed at `[0..packed_bytes]`
        // and scales at `[packed_bytes..]` via `packed_view`/`scales_view`.
        // Halves the per-projection H2D call count on the CPU-issue-bound
        // decode path. Bit-identical: same bytes (the combined slice IS the two
        // back-to-back regions), same kernel, same order — only the *number* of
        // host-issued copies changes. Falls through to the two-copy path below
        // when not arena-adjacent (mmap/pinned fallback, or pre-contiguous
        // layouts). Native-MXFP4 is unaffected (separate buffer/path).
        let transfer_stream = runtime.transfer_stream();
        if let Some(combined) = hw.contiguous_packed_scales() {
            let combined = combined?;
            let total = packed_bytes + scale_bytes;
            debug_assert_eq!(combined.len(), total, "combined slice spans packed+scales");
            if total > slot.combined.len() {
                return Err(AegisError::InvalidPlan(format!(
                    "staging combined overflow: layer needs {} bytes, slot has {}",
                    total,
                    slot.combined.len()
                )));
            }
            let mut dst = slot.combined.slice_mut(0..total);
            transfer_stream
                .memcpy_htod(&combined[..total], &mut dst)
                .map_err(map_cuda_err("staging async h2d combined (pinned src)"))?;
            slot.combined_packed_bytes = Some(packed_bytes);
            runtime.record_into_transfer(&slot.transfer_event)?;
            runtime.compute_wait_event(&slot.transfer_event)?;
            self.last_prepared_slot = Some(slot_idx);
            return Ok(slot_idx);
        }
        slot.combined_packed_bytes = None;
        // Source-type dispatch:
        //
        //   * Pinned source (Arena / Pinned variant) — feed straight into
        //     `memcpy_htod`. The driver detects pinned host memory and uses a
        //     direct DMA path with **zero CPU work** in the inference thread.
        //     This is the fast path used by the main NVFP4 expert-streaming
        //     workload (host-resident NVFP4 weights live in the shared arena).
        //
        //   * Unpinned source (Mmap variant) — CPU memcpy mmap → pinned bounce
        //     first, then DMA from bounce. Without the bounce, `memcpy_htod`
        //     internally bounces through the driver's small staging area,
        //     which serialises and is much slower for high-frequency calls.
        if hw.packed.is_pinned() {
            let src = hw.packed.as_bytes()?;
            let mut dst = slot.packed.slice_mut(0..packed_bytes);
            transfer_stream
                .memcpy_htod(&src[..packed_bytes], &mut dst)
                .map_err(map_cuda_err("staging async h2d packed (pinned src)"))?;
        } else {
            let src = hw.packed.as_bytes()?;
            let bounce = slot.bounce_packed.as_mut_slice();
            bounce[..packed_bytes].copy_from_slice(&src[..packed_bytes]);
            let mut dst = slot.packed.slice_mut(0..packed_bytes);
            // Slice the bounce to exactly `packed_bytes`. Passing the whole
            // `&OwnedPinnedBuf` would expose its full capacity (sized for
            // the largest layer) and cudarc's memcpy_htod asserts
            // `dst.len() >= src.len()` → panic on any smaller layer.
            let src_pinned: &[u8] = &slot.bounce_packed.as_slice()[..packed_bytes];
            transfer_stream
                .memcpy_htod(src_pinned, &mut dst)
                .map_err(map_cuda_err("staging async h2d packed (bounce)"))?;
        }
        if hw.scales.is_pinned() {
            let src = hw.scales.as_bytes()?;
            let mut dst = slot.scales.slice_mut(0..scale_bytes);
            transfer_stream
                .memcpy_htod(&src[..scale_bytes], &mut dst)
                .map_err(map_cuda_err("staging async h2d scales (pinned src)"))?;
        } else {
            let src = hw.scales.as_bytes()?;
            let bounce = slot.bounce_scales.as_mut_slice();
            bounce[..scale_bytes].copy_from_slice(&src[..scale_bytes]);
            let mut dst = slot.scales.slice_mut(0..scale_bytes);
            let src_pinned: &[u8] = &slot.bounce_scales.as_slice()[..scale_bytes];
            transfer_stream
                .memcpy_htod(src_pinned, &mut dst)
                .map_err(map_cuda_err("staging async h2d scales (bounce)"))?;
        }

        // Re-record the slot's transfer event with the just-queued H2D, then
        // make compute stream wait on it. Reusing the event avoids the
        // ~ms-scale cost of cuEventCreate on each call.
        runtime.record_into_transfer(&slot.transfer_event)?;
        runtime.compute_wait_event(&slot.transfer_event)?;

        self.last_prepared_slot = Some(slot_idx);
        Ok(slot_idx)
    }

    /// Stage native-MXFP4 repacked bytes into the most-recently-prepared slot.
    /// Must be called *before* `mark_kernel_launched` and after the matching
    /// `prepare_async`. The native-MXFP4 buffer reuses the same slot/event as
    /// the packed/scales buffers since they're consumed by the same kernel.
    pub(crate) fn prepare_native_mxfp4_into_last(
        &mut self,
        runtime: &CudaRuntime,
        mxfp4: &HostResidentMxfp4,
    ) -> Result<()> {
        let slot_idx = self.last_prepared_slot.ok_or_else(|| {
            AegisError::InvalidPlan(
                "prepare_native_mxfp4 called without a preceding prepare_async".into(),
            )
        })?;
        let slot = &mut self.slots[slot_idx];
        let buf = slot.native_mxfp4.as_mut().ok_or_else(|| {
            AegisError::InvalidPlan(
                "native MXFP4 staging buffer not allocated; set native_mxfp4_repack=true".into(),
            )
        })?;
        if mxfp4.data.len() > buf.len() {
            return Err(AegisError::InvalidPlan(format!(
                "staging native mxfp4 overflow: layer needs {} bytes, slot has {}",
                mxfp4.data.len(),
                buf.len()
            )));
        }
        let transfer_stream = runtime.transfer_stream();
        let len = mxfp4.data.len();
        if mxfp4.data.is_pinned() {
            let src = mxfp4.data.as_bytes()?;
            let mut dst = buf.slice_mut(0..len);
            transfer_stream
                .memcpy_htod(&src[..len], &mut dst)
                .map_err(map_cuda_err("staging async h2d native mxfp4 (pinned src)"))?;
        } else {
            let src = mxfp4.data.as_bytes()?;
            let bounce = slot.bounce_native_mxfp4.as_mut().ok_or_else(|| {
                AegisError::InvalidPlan(
                    "native MXFP4 bounce buffer not allocated; pool must be sized with native_mxfp4_bytes>0"
                        .into(),
                )
            })?;
            let bounce_slice = bounce.as_mut_slice();
            bounce_slice[..len].copy_from_slice(&src[..len]);
            let mut dst = buf.slice_mut(0..len);
            let src_pinned: &[u8] = &bounce.as_slice()[..len];
            transfer_stream
                .memcpy_htod(src_pinned, &mut dst)
                .map_err(map_cuda_err("staging async h2d native mxfp4 (bounce)"))?;
        }
        // Re-record the same transfer event so the next compute_wait_event
        // call (issued by the matvec wrapper after this) covers both the
        // packed/scales H2D and the native-mxfp4 H2D.
        runtime.record_into_transfer(&slot.transfer_event)?;
        runtime.compute_wait_event(&slot.transfer_event)?;
        Ok(())
    }

    /// Re-record the slot's compute event after the consuming kernel has been
    /// launched. The transfer stream will wait on this event before
    /// overwriting the slot in a subsequent `prepare_async`.
    pub(crate) fn mark_kernel_launched(
        &mut self,
        runtime: &CudaRuntime,
        slot_idx: usize,
    ) -> Result<()> {
        let slot = &mut self.slots[slot_idx];
        runtime.record_into_compute(&slot.compute_event)?;
        slot.primed = true;
        Ok(())
    }

    pub(crate) fn packed_view(&self, slot_idx: usize, len: usize) -> CudaView<'_, u8> {
        let slot = &self.slots[slot_idx];
        // Combined single-copy path: packed lives at [0..len) inside `combined`.
        if slot.combined_packed_bytes.is_some() {
            return slot.combined.slice(0..len);
        }
        slot.packed.slice(0..len)
    }

    pub(crate) fn scales_view(&self, slot_idx: usize, len: usize) -> CudaView<'_, u8> {
        let slot = &self.slots[slot_idx];
        // Combined single-copy path: scales follow packed inside `combined`.
        if let Some(packed_bytes) = slot.combined_packed_bytes {
            return slot.combined.slice(packed_bytes..packed_bytes + len);
        }
        slot.scales.slice(0..len)
    }

    pub(crate) fn native_mxfp4_view(
        &self,
        slot_idx: usize,
        len: usize,
    ) -> Option<CudaView<'_, u8>> {
        self.slots[slot_idx]
            .native_mxfp4
            .as_ref()
            .map(|s| s.slice(0..len))
    }

    // ── Backwards-compatible single-slot API ────────────────────────────────
    //
    // The codebase has many `staging.prepare(...)` + `staging.packed_view(len)`
    // call sites that pass NO slot index. To avoid touching ~20 sites in this
    // patch, the legacy methods route through the async pool but use slot 0
    // exclusively (i.e., no overlap, but correctness preserved). Migrate
    // call-sites to `prepare_async + mark_kernel_launched` for actual speedup.

    pub(crate) fn prepare(
        &mut self,
        hw: &HostResidentWeights,
        packed_bytes: usize,
        scale_bytes: usize,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    ) -> Result<()> {
        let slot = &mut self.slots[0];
        // Legacy path writes the separate packed/scales buffers; clear any
        // combined marker a prior prepare_async on slot 0 may have left so the
        // view accessors read the right (separate) buffers.
        slot.combined_packed_bytes = None;
        if packed_bytes > slot.packed.len() {
            return Err(AegisError::InvalidPlan(format!(
                "staging packed overflow: layer needs {} bytes, pool has {}",
                packed_bytes,
                slot.packed.len()
            )));
        }
        if scale_bytes > slot.scales.len() {
            return Err(AegisError::InvalidPlan(format!(
                "staging scales overflow: layer needs {} bytes, pool has {}",
                scale_bytes,
                slot.scales.len()
            )));
        }
        if hw.packed.len() != packed_bytes || hw.scales.len() != scale_bytes {
            return Err(AegisError::InvalidPlan(format!(
                "pinned host slice size mismatch: packed expected={} got={} scales expected={} got={}",
                packed_bytes,
                hw.packed.len(),
                scale_bytes,
                hw.scales.len()
            )));
        }
        count_h2d(packed_bytes + scale_bytes);
        // Synchronous (single-stream) path with the same source-type dispatch
        // as `prepare_async`: pinned source → direct DMA, unpinned (mmap) →
        // CPU memcpy through the bounce buffer first.
        if hw.packed.is_pinned() {
            let src = hw.packed.as_bytes()?;
            let mut dst = slot.packed.slice_mut(0..packed_bytes);
            stream
                .memcpy_htod(&src[..packed_bytes], &mut dst)
                .map_err(map_cuda_err("staging h2d packed (pinned src)"))?;
        } else {
            let src = hw.packed.as_bytes()?;
            let bounce = slot.bounce_packed.as_mut_slice();
            bounce[..packed_bytes].copy_from_slice(&src[..packed_bytes]);
            let mut dst = slot.packed.slice_mut(0..packed_bytes);
            let src_pinned: &[u8] = &slot.bounce_packed.as_slice()[..packed_bytes];
            stream
                .memcpy_htod(src_pinned, &mut dst)
                .map_err(map_cuda_err("staging h2d packed (bounce)"))?;
        }
        if hw.scales.is_pinned() {
            let src = hw.scales.as_bytes()?;
            let mut dst = slot.scales.slice_mut(0..scale_bytes);
            stream
                .memcpy_htod(&src[..scale_bytes], &mut dst)
                .map_err(map_cuda_err("staging h2d scales (pinned src)"))?;
        } else {
            let src = hw.scales.as_bytes()?;
            let bounce = slot.bounce_scales.as_mut_slice();
            bounce[..scale_bytes].copy_from_slice(&src[..scale_bytes]);
            let mut dst = slot.scales.slice_mut(0..scale_bytes);
            let src_pinned: &[u8] = &slot.bounce_scales.as_slice()[..scale_bytes];
            stream
                .memcpy_htod(src_pinned, &mut dst)
                .map_err(map_cuda_err("staging h2d scales (bounce)"))?;
        }
        self.last_prepared_slot = Some(0);
        Ok(())
    }

    pub(crate) fn prepare_native_mxfp4(
        &mut self,
        mxfp4: &HostResidentMxfp4,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    ) -> Result<()> {
        let slot = &mut self.slots[0];
        let buf = slot.native_mxfp4.as_mut().ok_or_else(|| {
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
        let src = mxfp4.data.as_bytes()?;
        let len = mxfp4.data.len();
        let mut dst = buf.slice_mut(0..len);
        stream
            .memcpy_htod(&src[..len], &mut dst)
            .map_err(map_cuda_err("staging h2d native mxfp4"))?;
        Ok(())
    }
}

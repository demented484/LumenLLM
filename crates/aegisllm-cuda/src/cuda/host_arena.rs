//! Single-allocation pinned host arena for streaming weights.
//!
//! Replaces the per-tensor `cuMemAllocHost` pattern (one CUDA-pinned alloc per
//! safetensors weight, ~7700 calls × ~10 ms each = 77 s of mlock + driver
//! overhead) with **one big pinned allocation up-front**, sub-allocated by
//! offset. Loading time drops from ~80 s to ~5–10 s for the same 18 GB model.
//!
//! The arena exposes:
//!   * `new` — single `cuMemAllocHost(capacity)` at construction.
//!   * `alloc_and_fill` — atomic bump-pointer sub-allocation that reads bytes
//!     directly from a `Read` source (typically a safetensors file handle).
//!   * `slice(offset, len)` — zero-cost read-only view used by the staging
//!     pool and `memcpy_htod` (the source bytes are pinned, so DMA is fast).
//!
//! Concurrency model: the arena is single-writer during the load phase
//! (`alloc_and_fill`) and many-reader after load (`slice`). The `used` counter
//! is atomic so multiple loaders could bump-allocate from different threads,
//! though current callers are single-threaded.
//!
//! Memory model: callers obtain raw `&[u8]` views into the arena. The arena
//! is held by the executor inside `Arc<PinnedArena>` and shared by every
//! weight that lives inside it; the pinned allocation is freed only when the
//! last `Arc` drops, which is the executor's drop time.

use std::io::Read;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use cudarc::driver::PinnedHostSlice;

use super::runtime::{CudaRuntime, map_cuda_err};
use aegisllm_base::error::{AegisError, Result};

/// One contiguous CUDA-pinned host allocation, sub-allocated by offset.
pub(crate) struct PinnedArena {
    /// Owns the pinned allocation. Kept alive for the arena's entire lifetime;
    /// `slice` views borrow into the same memory via the cached `base_ptr`.
    _backing: PinnedHostSlice<u8>,
    /// Cached base pointer. Captured once at construction so per-call reads
    /// don't pay cudarc's `event.synchronize` cost on every `slice`.
    base_ptr: *const u8,
    capacity: usize,
    used: AtomicUsize,
}

// SAFETY: the pinned allocation is locked at a stable host address; raw
// pointers into it remain valid for the arena's lifetime. Multiple readers
// of disjoint regions are fine; bump-allocation uses atomic ordering.
unsafe impl Send for PinnedArena {}
unsafe impl Sync for PinnedArena {}

impl PinnedArena {
    /// Allocate a single pinned-host buffer of `capacity` bytes. Use a
    /// generous over-estimate at construction; sub-allocations bump from 0.
    pub(crate) fn new(runtime: &CudaRuntime, capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(AegisError::InvalidPlan(
                "PinnedArena requires non-zero capacity".into(),
            ));
        }
        let mut backing = unsafe { runtime.stream.context().alloc_pinned::<u8>(capacity) }
            .map_err(map_cuda_err("alloc pinned arena"))?;
        let base_ptr = backing
            .as_mut_slice()
            .map_err(map_cuda_err("pinned arena init slice"))?
            .as_mut_ptr() as *const u8;
        Ok(Self {
            _backing: backing,
            base_ptr,
            capacity,
            used: AtomicUsize::new(0),
        })
    }

    /// Reserve `len` bytes and read them from `reader`. Returns the offset
    /// where the bytes were placed; combined with `len` this identifies the
    /// `slice` view callers will later use as the H2D source.
    ///
    /// Safety: the returned region is exclusively owned by this call's
    /// caller until `slice` views it. There must be no concurrent reader of
    /// the same region during the fill — which holds in current callers
    /// (load phase strictly precedes inference).
    pub(crate) fn alloc_and_fill<R: Read>(&self, mut reader: R, len: usize) -> Result<usize> {
        let offset = self.used.fetch_add(len, Ordering::SeqCst);
        if offset.saturating_add(len) > self.capacity {
            // Roll back the over-shoot so capacity reporting stays accurate.
            self.used.store(offset, Ordering::SeqCst);
            return Err(AegisError::InvalidPlan(format!(
                "PinnedArena overflow: requested {len} at offset {offset}, capacity {capacity}",
                capacity = self.capacity,
            )));
        }
        // SAFETY: `[offset..offset+len)` is exclusively claimed by this call
        // (atomic bump above) and inside the pinned allocation.
        let dst = unsafe {
            std::slice::from_raw_parts_mut((self.base_ptr as *mut u8).add(offset), len)
        };
        reader
            .read_exact(dst)
            .map_err(|e| AegisError::Unsupported(format!("pinned arena read_exact: {e}")))?;
        Ok(offset)
    }

    /// Read-only view of `len` bytes at `offset`. The returned slice is
    /// pinned host memory and is the right kind of source for fast
    /// `memcpy_htod` DMA.
    pub(crate) fn slice(&self, offset: usize, len: usize) -> &[u8] {
        debug_assert!(
            offset.saturating_add(len) <= self.capacity,
            "PinnedArena::slice out of bounds (offset={offset} len={len} cap={cap})",
            cap = self.capacity,
        );
        // SAFETY: bounds checked in debug; offsets only ever come from a
        // prior `alloc_and_fill` on this same arena.
        unsafe { std::slice::from_raw_parts(self.base_ptr.add(offset), len) }
    }

    pub(crate) fn used(&self) -> usize {
        self.used.load(Ordering::Relaxed)
    }

    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }
}

impl std::fmt::Debug for PinnedArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedArena")
            .field("capacity_bytes", &self.capacity)
            .field("used_bytes", &self.used())
            .finish()
    }
}

/// Convenience: shared handle to an arena. Each host-resident weight clones
/// one of these; the arena is dropped when the last clone goes away
/// (= when the executor is dropped).
pub(crate) type ArenaHandle = Arc<PinnedArena>;

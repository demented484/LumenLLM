//! Single-allocation pinned host arena for host-resident weights.
//!
//! Instead of letting host-resident weights live in shard mmap pages
//! (the post-873d765 architecture), this arena allocates one big
//! `OwnedPinnedBuf` upfront, sub-allocates per-tensor regions by
//! atomic bump, and reads tensor bytes directly from disk into them.
//! The whole arena is page-locked with `cuMemHostRegister` once
//! after the load loop finishes — DMA from any sub-region takes the
//! direct-pinned-DMA path, no per-token CPU memcpy through a
//! staging-pool bounce.
//!
//! Memory profile: ~12-14 GiB anonymous-mapped pinned RAM for
//! Gemma-4-26B's host-resident expert weights. Backed by our own
//! `OwnedPinnedBuf` (mmap+cuMemHostRegister via the
//! process), so on `kill aegisllm` the kernel unmaps and the driver
//! release callback returns pages to the kernel free list — no NVIDIA
//! driver pool retention.
//!
//! Concurrency model: single-writer during load (`alloc_and_fill`),
//! many-reader after load (`slice`). The `used` counter is atomic so
//! multiple loaders could bump-allocate from different threads, though
//! current callers are single-threaded.

use std::io::Read;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::owned_pinned::OwnedPinnedBuf;
use super::runtime::CudaRuntime;
use aegisllm_base::error::{AegisError, Result};

/// One contiguous pinned host allocation, sub-allocated by offset.
pub(crate) struct PinnedArena {
    /// Owns the pinned allocation. Pages are mmap'd by us (process-
    /// owned) and registered with CUDA for fast DMA via
    /// `cuMemHostRegister`. On drop, `cuMemHostUnregister` + `munmap`
    /// returns pages to the kernel immediately.
    _backing: OwnedPinnedBuf,
    /// Cached base pointer.
    base_ptr: *const u8,
    capacity: usize,
    used: AtomicUsize,
}

// SAFETY: the pinned allocation is locked at a stable host address;
// raw pointers into it remain valid for the arena's lifetime.
// Multiple readers of disjoint regions are fine; bump-allocation uses
// atomic ordering.
unsafe impl Send for PinnedArena {}
unsafe impl Sync for PinnedArena {}

impl PinnedArena {
    /// Allocate a single pinned-host buffer of `capacity` bytes. Use
    /// a generous over-estimate at construction; sub-allocations bump
    /// from 0. Pages are demand-faulted as `alloc_and_fill` writes
    /// tensor data, so RSS grows gradually with load progress instead
    /// of jumping by the full capacity (~14 GiB on Gemma-4-26B) at
    /// construction time.
    pub(crate) fn new(_runtime: &CudaRuntime, capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(AegisError::InvalidPlan(
                "PinnedArena requires non-zero capacity".into(),
            ));
        }
        let mut backing = OwnedPinnedBuf::new_unpinned(capacity)?;
        let base_ptr = backing.as_mut_ptr() as *const u8;
        let actual_capacity = backing.len();
        Ok(Self {
            _backing: backing,
            base_ptr,
            capacity: actual_capacity,
            used: AtomicUsize::new(0),
        })
    }

    /// Page-lock the entire arena with the CUDA driver so subsequent
    /// `memcpy_htod` calls from arena slices take the direct-DMA fast
    /// path. Call this once after the load loop has finished writing
    /// every tensor — pages are already committed by the writes, so
    /// the registration just locks them in place without extra
    /// physical-memory cost. Takes `&self` so the call can be made
    /// through an `Arc<PinnedArena>` after host-resident weights have
    /// already cloned their references.
    pub(crate) fn pin_now(&self) -> Result<()> {
        self._backing.pin_now()
    }

    /// Reserve `len` bytes and read them from `reader`. Returns the
    /// offset where the bytes were placed; combined with `len` this
    /// identifies the `slice` view callers will later use as the H2D
    /// source.
    pub(crate) fn alloc_and_fill<R: Read>(&self, mut reader: R, len: usize) -> Result<usize> {
        let offset = self.reserve(len)?;
        // SAFETY: `reserve` exclusively claimed `[offset, offset+len)`.
        let dst = unsafe { self.slice_mut(offset, len) };
        reader
            .read_exact(dst)
            .map_err(|e| AegisError::Unsupported(format!("pinned arena read_exact: {e}")))?;
        Ok(offset)
    }

    /// Reserve `len` bytes via atomic bump-pointer alloc. Returns the
    /// claimed offset. The caller is responsible for filling
    /// `[offset, offset+len)` exactly once before the bytes are
    /// observed by any other path. Use this when filling happens out
    /// of band — e.g. via parallel `pread` workers each writing a
    /// disjoint sub-region. Pair with `slice_mut` to get the write
    /// destination.
    pub(crate) fn reserve(&self, len: usize) -> Result<usize> {
        let offset = self.used.fetch_add(len, Ordering::SeqCst);
        if offset.saturating_add(len) > self.capacity {
            // Roll back the over-shoot so capacity reporting stays accurate.
            self.used.store(offset, Ordering::SeqCst);
            return Err(AegisError::InvalidPlan(format!(
                "PinnedArena overflow: requested {len} at offset {offset}, capacity {capacity}",
                capacity = self.capacity,
            )));
        }
        Ok(offset)
    }

    /// Mutable view of `[offset, offset+len)` for a slot the caller
    /// has just reserved. The caller asserts that no other thread
    /// holds a reference into the same range — the arena does NOT
    /// track per-slot ownership; safety is on the caller. Used by
    /// parallel `pread` paths where each worker writes a disjoint
    /// sub-slice.
    ///
    /// # Safety
    /// `offset..offset+len` must lie within the arena and must be
    /// exclusively owned by this caller (no other reader or writer).
    pub(crate) unsafe fn slice_mut(&self, offset: usize, len: usize) -> &mut [u8] {
        debug_assert!(
            offset.saturating_add(len) <= self.capacity,
            "PinnedArena::slice_mut out of bounds (offset={offset} len={len} cap={cap})",
            cap = self.capacity,
        );
        unsafe { std::slice::from_raw_parts_mut((self.base_ptr as *mut u8).add(offset), len) }
    }

    /// Read-only view of `len` bytes at `offset`.
    pub(crate) fn slice(&self, offset: usize, len: usize) -> &[u8] {
        debug_assert!(
            offset.saturating_add(len) <= self.capacity,
            "PinnedArena::slice out of bounds (offset={offset} len={len} cap={cap})",
            cap = self.capacity,
        );
        // SAFETY: bounds checked in debug; offsets only ever come
        // from a prior `alloc_and_fill` on this same arena.
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

/// Convenience: shared handle to an arena. Each host-resident weight
/// clones one of these; the arena is dropped when the last clone goes
/// away (= when the executor is dropped).
pub(crate) type ArenaHandle = Arc<PinnedArena>;

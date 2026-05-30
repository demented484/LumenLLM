//! Pinned host memory we OWN, instead of borrowing from CUDA's driver pool.
//!
//! ## Why this exists
//! `cuMemHostAlloc` (cudarc's `alloc_pinned`) asks the NVIDIA driver to
//! allocate AND pin pages. The pages are owned by the driver's internal
//! pool. On free / process exit, those pages return to the pool, NOT to
//! the kernel's free list. The pool only releases under shrinker
//! pressure (which on Linux requires `drop_caches=2` — root-only, or
//! organic memory pressure that may take minutes to land).
//!
//! Empirically (verified with a CPU-only Rust program that allocates
//! 15 GiB `Vec<u8>`): plain anonymous memory returns to the kernel
//! IMMEDIATELY on `kill`. So the leftover-RAM problem is specific to
//! `cuMemHostAlloc`, not a generic Linux mm subsystem behavior.
//!
//! ## What this does instead
//! We allocate pages ourselves via `mmap(MAP_ANONYMOUS|MAP_PRIVATE)`.
//! The pages are owned by our process's mm. We then call
//! `cuMemHostRegister` to ask the driver to PIN those pages for fast
//! DMA — but the driver doesn't own them. On `Drop` we
//! `cuMemHostUnregister` + `munmap`, returning the pages to the kernel
//! immediately.
//!
//! On process kill (any signal, including `_exit` from our SIGTERM
//! handler): the kernel auto-tears down the VMA, which triggers the
//! driver's release callback on `/dev/nvidia*` to unpin. Pages return
//! to the kernel's free list right away. No driver pool, no shrinker
//! dance, no `drop_caches` needed.
//!
//! ## API
//! `OwnedPinnedBuf` is a drop-in replacement for cudarc's
//! `PinnedHostSlice<u8>` for our use cases — exposes `as_slice()` /
//! `as_mut_slice()` returning `&[u8]` / `&mut [u8]`, plus `as_ptr()` /
//! `as_mut_ptr()` and `len()`. `&[u8]` already implements cudarc's
//! `HostSlice<u8>` trait, so it can be passed straight to
//! `memcpy_htod` as a pinned source — the driver detects the
//! registered pinning and uses the fast direct-DMA path.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use aegisllm_base::error::{AegisError, Result};
use cudarc::driver::{CudaStream, HostSlice, SyncOnDrop, sys};

#[derive(Debug)]
pub(crate) struct OwnedPinnedBuf {
    ptr: *mut u8,
    len: usize,
    /// `true` iff `cuMemHostRegister` succeeded. Atomic so `pin_now`
    /// can be called through an `Arc<...>` (no &mut required); the
    /// underlying CUDA call is a pointer-based syscall, the only Rust
    /// state we mutate is this flag.
    registered: AtomicBool,
    /// Byte offset where the cuMemHostRegister'd range starts.
    /// `pin_range` may register a sub-range starting > 0 (e.g. when
    /// the arena's used portion doesn't begin at offset 0), and
    /// `cuMemHostUnregister` must be called with the exact pointer
    /// that was registered. `pin_now`/full-buffer pin leaves this 0.
    register_start: AtomicUsize,
    /// Device-accessible pointer to the registered range, obtained via
    /// `cuMemHostGetDevicePointer_v2` when the buffer is pinned with
    /// `pin_range_devicemap`. 0 when the buffer was registered without
    /// `CU_MEMHOSTREGISTER_DEVICEMAP` (the default `pin_now`/`pin_range`
    /// path). This `CUdeviceptr` corresponds to the **host** address
    /// `ptr + register_start`; callers that want the device pointer for an
    /// arbitrary offset `o >= register_start` add `(o - register_start)`.
    device_ptr: AtomicU64,
}

// SAFETY: the underlying pages are pinned for the lifetime of the
// struct. The pointer remains valid until Drop.
unsafe impl Send for OwnedPinnedBuf {}
unsafe impl Sync for OwnedPinnedBuf {}

impl OwnedPinnedBuf {
    /// Allocate `len` bytes via anonymous mmap AND immediately register
    /// them with CUDA for DMA pinning. The register call page-locks
    /// (i.e. commits) every page in the range up-front. Use this when
    /// the buffer is small (~MB) or when you'll write to all of it
    /// shortly; for larger buffers that get filled gradually, prefer
    /// `new_unpinned` + `pin_now` so the pages commit lazily as you
    /// write and the registration only locks down the final committed
    /// set, avoiding a 10+ GiB instantaneous RSS jump at construction.
    pub(crate) fn new(len: usize) -> Result<Self> {
        let mut buf = Self::new_unpinned(len)?;
        buf.pin_now()?;
        Ok(buf)
    }

    /// Allocate `len` bytes via anonymous mmap WITHOUT registering with
    /// CUDA. Pages are demand-paged: virtual address space is reserved
    /// but no physical memory is committed until the caller writes to
    /// it. Use this for the big load-time arena so `RSS` grows
    /// gradually as tensors are read in, instead of jumping by the
    /// full arena capacity at allocation time (which would temporarily
    /// double the visible host-RAM peak during load and trigger a
    /// freeze on memory-tight hosts).
    pub(crate) fn new_unpinned(len: usize) -> Result<Self> {
        if len == 0 {
            return Err(AegisError::InvalidPlan(
                "OwnedPinnedBuf requires non-zero size".into(),
            ));
        }
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let len_aligned = len.div_ceil(page_size) * page_size;
        // `MAP_NORESERVE` is **load-bearing** for the 14 GiB arena on
        // 32 GiB hosts. Without it, mmap counts the full virtual range
        // against `Committed_AS` immediately — the kernel reserves swap
        // backing for every page even though we'll never touch some of
        // them. With strict overcommit (or even default heuristic when
        // swap is small), the next big allocation (KV cache during
        // `new_state`, ~6 GiB) trips `Committed_AS > CommitLimit` and
        // OOM-killer terminates us silently. CUDA's `cuMemHostAlloc`
        // path bypasses this check entirely (driver-managed pool, not
        // VMA accounting), which is why ba3b5b8 didn't OOM and the
        // mmap-backed arena does. With `MAP_NORESERVE`, only physically-
        // touched pages count — exactly what we want for a lazily-
        // committed arena.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len_aligned,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            let err = std::io::Error::last_os_error();
            return Err(AegisError::Unsupported(format!(
                "OwnedPinnedBuf: mmap({len_aligned}) failed: {err}"
            )));
        }
        Ok(Self {
            ptr: ptr as *mut u8,
            len: len_aligned,
            registered: AtomicBool::new(false),
            register_start: AtomicUsize::new(0),
            device_ptr: AtomicU64::new(0),
        })
    }

    /// Register our pages with the CUDA driver so subsequent
    /// `memcpy_htod` calls take the direct-pinned-DMA fast path.
    /// Idempotent. Takes `&self` (not `&mut`) so it can be invoked
    /// through an `Arc<...>` after handles have been cloned out — the
    /// only Rust state that changes is an atomic flag, and the CUDA
    /// driver's `cuMemHostRegister` is itself thread-safe per the API
    /// docs. Call this AFTER the buffer has been fully written, so
    /// the register call only page-locks already-committed pages
    /// (no extra physical-memory commits).
    pub(crate) fn pin_now(&self) -> Result<()> {
        self.pin_range(0, self.len)
    }

    /// Register only the first `len` bytes (rounded up to a page).
    /// Used by `PinnedArena::pin_now` to lock just the actually-
    /// written portion of the arena, not the trailing unused pages
    /// that `compute_host_arena_capacity` over-estimated. With
    /// `MAP_NORESERVE` set, those trailing pages are uncommitted —
    /// registering them would force the kernel to commit AND lock
    /// pages we'll never read, costing free RAM that's not actually
    /// needed for inference.
    pub(crate) fn pin_range(&self, offset: usize, len: usize) -> Result<()> {
        self.pin_range_with_flags(offset, len, sys::CU_MEMHOSTREGISTER_PORTABLE, false)
    }

    /// Like `pin_range`, but also passes `CU_MEMHOSTREGISTER_DEVICEMAP` so the
    /// GPU can read the host pages directly, and then resolves the
    /// device-accessible pointer for the registered range via
    /// `cuMemHostGetDevicePointer_v2`. After this call `device_ptr()` returns a
    /// non-zero `CUdeviceptr`. Used for the host-resident expert weight arena so
    /// a GPU gather kernel can stream the selected experts' bytes from mapped
    /// host RAM into a VRAM scratch in one launch — no CPU round-trip.
    pub(crate) fn pin_range_devicemap(&self, offset: usize, len: usize) -> Result<()> {
        self.pin_range_with_flags(
            offset,
            len,
            sys::CU_MEMHOSTREGISTER_PORTABLE | sys::CU_MEMHOSTREGISTER_DEVICEMAP,
            true,
        )
    }

    fn pin_range_with_flags(
        &self,
        offset: usize,
        len: usize,
        flags: u32,
        devicemap: bool,
    ) -> Result<()> {
        if self.registered.load(Ordering::Acquire) {
            return Ok(());
        }
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        // Round the registered range OUT to page boundaries (down on
        // start, up on end) — `cuMemHostRegister` requires page
        // alignment. Clamp to the buffer's bounds.
        let start = (offset / page_size) * page_size;
        let end = ((offset + len).div_ceil(page_size) * page_size).min(self.len);
        let registered_len = end.saturating_sub(start);
        if registered_len == 0 {
            self.registered.store(true, Ordering::Release);
            return Ok(());
        }
        let host_start = unsafe { (self.ptr as *mut u8).add(start) };
        let r = unsafe { sys::cuMemHostRegister_v2(host_start as *mut _, registered_len, flags) };
        if r != sys::CUresult::CUDA_SUCCESS {
            return Err(AegisError::Unsupported(format!(
                "OwnedPinnedBuf::pin_range({start}, {registered_len}, flags={flags:#x}): cuMemHostRegister failed: {r:?}"
            )));
        }
        if devicemap {
            // Resolve the device-accessible pointer for the registered host
            // range. With UVA on a 64-bit Linux + recent driver this is
            // typically equal to the host VA, but we must not assume that —
            // always go through the API.
            let mut dptr: sys::CUdeviceptr = 0;
            let r = unsafe {
                sys::cuMemHostGetDevicePointer_v2(&mut dptr, host_start as *mut _, 0)
            };
            if r != sys::CUresult::CUDA_SUCCESS {
                // Roll back the registration so Drop doesn't double-unregister
                // a range we can't actually use.
                unsafe {
                    let _ = sys::cuMemHostUnregister(host_start as *mut _);
                }
                return Err(AegisError::Unsupported(format!(
                    "OwnedPinnedBuf::pin_range_devicemap({start}, {registered_len}): cuMemHostGetDevicePointer_v2 failed: {r:?}"
                )));
            }
            self.device_ptr.store(dptr, Ordering::Release);
        }
        // Track the registered start so Drop can unregister at the
        // exact pointer we passed (cuMemHostUnregister wants the same
        // address that was registered).
        self.registered.store(true, Ordering::Release);
        self.register_start
            .store(start, std::sync::atomic::Ordering::Release);
        Ok(())
    }

    /// Device-accessible pointer (`CUdeviceptr`) for the registered range's
    /// start (= host `ptr + register_start`). Returns 0 when the buffer was not
    /// pinned with `pin_range_devicemap`. The device pointer for an arbitrary
    /// host offset `o` is `device_ptr_base() + (o - register_start())`.
    pub(crate) fn device_ptr_base(&self) -> u64 {
        self.device_ptr.load(Ordering::Acquire)
    }

    /// Byte offset of the registered range's start within the buffer (the
    /// page-aligned-down `start` from `pin_range*`). Combined with
    /// `device_ptr_base()` this maps any host offset to its device pointer.
    pub(crate) fn register_start(&self) -> usize {
        self.register_start.load(Ordering::Acquire)
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn as_ptr(&self) -> *const u8 {
        self.ptr as *const u8
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        // SAFETY: pointer is valid for `len` bytes for the lifetime
        // of `self`; pages are pinned so the kernel cannot move or
        // unmap them.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: same as as_slice; &mut self gives exclusive access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Reinterpret the underlying bytes as `&[u16]`. mmap-allocated
    /// memory is always page-aligned (≥ 4 KiB) so 2-byte alignment is
    /// satisfied. Length must be even (debug-asserted).
    pub(crate) fn as_u16_slice(&self) -> &[u16] {
        debug_assert!(self.len % 2 == 0, "OwnedPinnedBuf::as_u16_slice on odd-byte buffer");
        // SAFETY: page-aligned ptr → 2-byte aligned. len/2 elements
        // each occupying 2 bytes covers exactly the same memory as
        // `self.len` bytes.
        unsafe { std::slice::from_raw_parts(self.ptr as *const u16, self.len / 2) }
    }
}

/// Direct `HostSlice<u8>` impl so `OwnedPinnedBuf` can be passed
/// straight to `stream.memcpy_htod(&buf, &mut dst)` — same call site
/// pattern as `&PinnedHostSlice<u8>`. We don't carry per-buffer
/// synchronisation events (callers do stream-level event work
/// elsewhere), so we return `SyncOnDrop::Sync(None)` exactly like
/// the `Vec<T>` and `[T]` impls in cudarc.
impl HostSlice<u8> for OwnedPinnedBuf {
    fn len(&self) -> usize {
        self.len
    }
    unsafe fn stream_synced_slice<'a>(
        &'a self,
        _stream: &'a CudaStream,
    ) -> (&'a [u8], SyncOnDrop<'a>) {
        (
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) },
            SyncOnDrop::Sync(None),
        )
    }
    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        _stream: &'a CudaStream,
    ) -> (&'a mut [u8], SyncOnDrop<'a>) {
        (
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) },
            SyncOnDrop::Sync(None),
        )
    }
}

impl Drop for OwnedPinnedBuf {
    fn drop(&mut self) {
        unsafe {
            if self.registered.load(Ordering::Acquire) {
                let start = self.register_start.load(Ordering::Acquire);
                // cuMemHostUnregister wants the exact pointer that was
                // registered via cuMemHostRegister — for partial pins
                // (`pin_range`) that's `ptr + start`, not `ptr`.
                let _ = sys::cuMemHostUnregister(self.ptr.add(start) as *mut _);
            }
            libc::munmap(self.ptr as *mut _, self.len);
        }
    }
}

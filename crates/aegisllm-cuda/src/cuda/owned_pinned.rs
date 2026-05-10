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

use std::sync::atomic::{AtomicBool, Ordering};

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
        if self.registered.load(Ordering::Acquire) {
            return Ok(());
        }
        let r = unsafe {
            sys::cuMemHostRegister_v2(
                self.ptr as *mut _,
                self.len,
                sys::CU_MEMHOSTREGISTER_PORTABLE,
            )
        };
        if r != sys::CUresult::CUDA_SUCCESS {
            return Err(AegisError::Unsupported(format!(
                "OwnedPinnedBuf::pin_now: cuMemHostRegister failed: {r:?}"
            )));
        }
        self.registered.store(true, Ordering::Release);
        Ok(())
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
                let _ = sys::cuMemHostUnregister(self.ptr as *mut _);
            }
            libc::munmap(self.ptr as *mut _, self.len);
        }
    }
}

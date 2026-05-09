//! Best-effort page-cache eviction on graceful shutdown.
//!
//! On a normal kill (`SIGTERM` from `kill aegisllm` / Ctrl-C in an
//! attached terminal / docker stop), evict the safetensors shard pages
//! from the kernel page cache before exit. The runtime arena holds the
//! working copy in pinned memory, so the page cache is wasted bytes from
//! the moment the load finishes — leaving it stuck means `kill aegisllm`
//! frees < 17 GiB less than the user expects.
//!
//! Doesn't help against `SIGKILL` (`kill -9`) — those are uncatchable
//! and there's nothing user-space can do. The full-shard fadvise sweep
//! at the end of `from_artifact` is the SIGKILL fallback: by the time
//! the process is "killable," the cache is already evicted.
//!
//! Implementation notes:
//!   - Uses `libc::sigaction` directly (no `signal_hook` / `nix`).
//!   - Stores shard paths as `CString` in a static `OnceLock` so the
//!     handler can use them without allocating.
//!   - Signal handler only calls async-signal-safe libc primitives
//!     (`open`, `posix_fadvise`, `close`, `_exit`) — no Rust
//!     allocations, no `println!`.

use std::ffi::CString;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::sync::OnceLock;

static SHARD_PATHS: OnceLock<Vec<CString>> = OnceLock::new();

/// Register the safetensors shard paths and install handlers for
/// `SIGTERM` / `SIGINT` so they run a fadvise sweep before exit. Idempotent
/// — only the first call wins, subsequent calls are no-ops (the path list
/// shouldn't change between loads in a single process).
pub(super) fn install(paths: impl IntoIterator<Item = std::path::PathBuf>) {
    let cstrings: Vec<CString> = paths
        .into_iter()
        .filter_map(|p| CString::new(p.as_path().as_os_str().as_bytes()).ok())
        .collect();
    if SHARD_PATHS.set(cstrings).is_err() {
        return; // already installed
    }
    unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = handler as usize;
        sa.sa_flags = libc::SA_RESTART;
        libc::sigaction(libc::SIGTERM, &sa, std::ptr::null_mut());
        libc::sigaction(libc::SIGINT, &sa, std::ptr::null_mut());
        libc::sigaction(libc::SIGHUP, &sa, std::ptr::null_mut());
    }
}

extern "C" fn handler(signum: libc::c_int) {
    if let Some(paths) = SHARD_PATHS.get() {
        for path in paths {
            unsafe {
                let fd = libc::open(path.as_ptr(), libc::O_RDONLY);
                if fd >= 0 {
                    // (offset=0, len=0) means "the whole file" per POSIX.
                    let _ = libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED);
                    let _ = libc::close(fd);
                }
            }
        }
    }
    // Standard "exit code = 128 + signum" convention so callers can tell
    // signal exits apart from normal returns.
    unsafe { libc::_exit(128 + signum) }
}

#[doc(hidden)]
pub(super) fn _path_helper(p: &Path) -> Option<CString> {
    CString::new(p.as_os_str().as_bytes()).ok()
}

//! Auto-eviction of safetensors page-cache pages after aegisllm exits.
//!
//! ## Problem
//! With `store=mmap` the executor keeps each shard mmap'd
//! (`Arc<Mmap>` per host-resident NVFP4 weight) for the whole serve
//! lifetime so the H2D streaming path can read directly from those
//! VMAs. After inference touches a chunk of weights, the kernel's
//! page cache is populated with several GiB of shard data. When the
//! user kills the server those pages remain in `Cached` indefinitely
//! — Linux's `posix_fadvise(POSIX_FADV_DONTNEED)` is best-effort and
//! on an idle system tends to leave warm pages alone. The user used
//! to have to run `sync && echo 3 > /proc/sys/vm/drop_caches` (which
//! requires root) to recover the memory.
//!
//! ## Fix
//! At the end of weight load we `fork()` once and **immediately
//! `execv`** the `aegisllm-evict` helper binary in sidecar mode:
//!
//!     aegisllm-evict --wait-fd <N> --parent-pid <PID> <shard1> ...
//!
//! `execv` discards the inherited address space, so:
//!   - The sidecar shows up in `ps` as `aegisllm-evict`, not as a
//!     duplicate `aegisllm` (no confusion in `pgrep`/`top`).
//!   - It doesn't carry CoW copies of the parent's ~17 GiB shard
//!     VMAs, so its VSZ is tiny.
//!   - The eviction syscalls run in a fresh address space — empirically
//!     the only configuration where `posix_fadvise(DONTNEED)` actually
//!     frees the pages. (CoW'd children manage to free only a couple
//!     hundred MiB of a ~6 GiB inferred-warm page cache.)
//!
//! The helper:
//!   1. Reads the inherited pipe fd until EOF (parent exit closes it).
//!   2. Polls `/proc/<parent>/status` until it disappears, ensuring
//!      `exit_mm()` has finished tearing down the parent's mappings.
//!   3. Runs `mmap+madvise(DONTNEED)+munmap+posix_fadvise(DONTNEED)`
//!      on each shard.
//!   4. Exits.
//!
//! Works for graceful kills (SIGTERM/SIGINT/SIGHUP) and SIGKILL
//! alike — both close the parent's fds, which closes the parent's
//! pipe write end, which gives the helper its EOF.

use std::ffi::{CString, OsString};
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::PathBuf;
use std::sync::OnceLock;

static INSTALLED: OnceLock<()> = OnceLock::new();

/// Spawn the eviction sidecar and install the signal-handler stub for
/// the `128 + signum` exit-code convention. Idempotent.
pub(super) fn install(paths: impl IntoIterator<Item = PathBuf>) {
    if INSTALLED.set(()).is_err() {
        return;
    }
    let path_bufs: Vec<PathBuf> = paths.into_iter().collect();
    if path_bufs.is_empty() {
        return;
    }
    spawn_sidecar(&path_bufs);
    install_signal_handlers();
}

fn spawn_sidecar(paths: &[PathBuf]) {
    // Locate the helper binary co-located with the running aegisllm
    // executable (covers `cargo run` and side-by-side install
    // layouts). Fall back to plain `aegisllm-evict` for $PATH lookup.
    let helper_path = locate_helper().unwrap_or_else(|| OsString::from("aegisllm-evict"));
    let helper_cstr = match CString::new(helper_path.as_bytes()) {
        Ok(c) => c,
        Err(_) => return,
    };
    let parent_pid_str = std::process::id().to_string();
    // Build owned argv: [helper, --wait-fd, <N>, --parent-pid, <PID>, paths...]
    // The fd value goes in after the pipe is created, so we patch it
    // in place after fork.
    let mut argv_strs: Vec<CString> = Vec::with_capacity(paths.len() + 5);
    argv_strs.push(helper_cstr.clone());
    argv_strs.push(CString::new("--wait-fd").unwrap());
    argv_strs.push(CString::new("0").unwrap()); // placeholder; fixed up to real fd after pipe()
    argv_strs.push(CString::new("--parent-pid").unwrap());
    argv_strs.push(CString::new(parent_pid_str.as_str()).unwrap());
    for path in paths {
        match CString::new(path.as_os_str().as_bytes()) {
            Ok(c) => argv_strs.push(c),
            Err(_) => return,
        }
    }

    unsafe {
        let mut pipefd = [0i32; 2];
        // The pipe write end stays in the parent for the rest of its
        // life. CLOEXEC on it so any subsequent `execv` from the
        // parent (e.g. NVIDIA / Vulkan / wgpu loading shaders or
        // child workers) doesn't accidentally inherit it and keep
        // the helper's `read()` hanging after the parent dies.
        // The read end goes to the helper via execv — we can't set
        // CLOEXEC on it (it'd close on exec). Instead, we set it on
        // both ends with pipe2, then clear it on the read end before
        // execv.
        if libc::pipe2(pipefd.as_mut_ptr(), libc::O_CLOEXEC) != 0 {
            return;
        }
        // Patch the placeholder fd in argv to the real read-end fd.
        let fd_str = match CString::new(pipefd[0].to_string()) {
            Ok(c) => c,
            Err(_) => {
                libc::close(pipefd[0]);
                libc::close(pipefd[1]);
                return;
            }
        };
        argv_strs[2] = fd_str;

        let pid = libc::fork();
        if pid < 0 {
            libc::close(pipefd[0]);
            libc::close(pipefd[1]);
            return;
        }
        if pid == 0 {
            // ===== CHILD =====
            // Move into our own process group so a foreground `Ctrl+C`
            // (SIGINT delivered to the parent's pgrp) doesn't reach us
            // and prevent eviction.
            libc::setpgid(0, 0);
            // Close write end (parent owns it) and clear CLOEXEC on the
            // read end so it survives the upcoming execv.
            libc::close(pipefd[1]);
            let flags = libc::fcntl(pipefd[0], libc::F_GETFD);
            if flags >= 0 {
                let _ = libc::fcntl(pipefd[0], libc::F_SETFD, flags & !libc::FD_CLOEXEC);
            }
            // Build the argv pointer array. Owned by us; freed on _exit.
            let mut argv_ptrs: Vec<*const libc::c_char> =
                argv_strs.iter().map(|c| c.as_ptr()).collect();
            argv_ptrs.push(std::ptr::null());
            libc::execv(helper_cstr.as_ptr(), argv_ptrs.as_ptr());
            // execv only returns on failure; try $PATH lookup.
            let basename = CString::new("aegisllm-evict").unwrap();
            argv_ptrs[0] = basename.as_ptr();
            libc::execvp(basename.as_ptr(), argv_ptrs.as_ptr());
            // If both attempts failed, exit silently. Cache stays
            // populated until the OS reclaims under memory pressure;
            // not worse than not having the sidecar at all.
            libc::_exit(127);
        }
        // ===== PARENT =====
        libc::close(pipefd[0]);
    }
}

/// Return the path of `aegisllm-evict` co-located with `/proc/self/exe`.
fn locate_helper() -> Option<OsString> {
    let exe = std::fs::read_link("/proc/self/exe").ok()?;
    let mut path: Vec<u8> = exe.parent()?.as_os_str().as_bytes().to_vec();
    path.push(b'/');
    path.extend_from_slice(b"aegisllm-evict");
    Some(OsString::from_vec(path))
}

fn install_signal_handlers() {
    unsafe {
        let mut sa: libc::sigaction = std::mem::zeroed();
        // Block all other signals while the handler runs so a second
        // SIGTERM/SIGINT/SIGHUP can't reenter `_exit` mid-execution.
        // `std::mem::zeroed` doesn't initialise `sa_mask` portably.
        libc::sigfillset(&mut sa.sa_mask);
        sa.sa_sigaction = exit_handler as usize;
        sa.sa_flags = libc::SA_RESTART;
        libc::sigaction(libc::SIGTERM, &sa, std::ptr::null_mut());
        libc::sigaction(libc::SIGINT, &sa, std::ptr::null_mut());
        libc::sigaction(libc::SIGHUP, &sa, std::ptr::null_mut());
    }
}

/// Just `_exit(128 + signum)` so callers can distinguish signal exits
/// from normal returns. The sidecar handles cache eviction.
extern "C" fn exit_handler(signum: libc::c_int) {
    unsafe { libc::_exit(128 + signum) }
}

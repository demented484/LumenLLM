//! Standalone helper that evicts a list of files from the kernel page
//! cache. Used by `aegisllm` itself (the main `serve` binary) to clear
//! the model's safetensors shards out of `Cached` after the user kills
//! the server, so `free -h` doesn't keep showing several GiB of
//! inherited cache requiring `sudo drop_caches` to release.
//!
//! ## Two modes
//!
//!   - `aegisllm-evict <path>...`
//!       Evict each path immediately and exit. Used by sysadmins who
//!       just want to clear the cache by hand.
//!
//!   - `aegisllm-evict --wait-fd <N> --parent-pid <PID> <path>...`
//!       Sidecar mode: block reading from inherited pipe fd <N> until
//!       it returns EOF (which happens when the aegisllm parent dies
//!       and the kernel closes its end of the pipe). Then poll
//!       /proc/<PID>/status until the parent is fully reaped — Linux
//!       closes fds before tearing down VMAs in `do_exit`, so we'd
//!       race the parent's `exit_mm` if we evicted on EOF directly —
//!       and finally evict.
//!
//! The sidecar mode is what the main aegisllm binary uses: it spawns
//! us via `fork()` immediately after weight load, then immediately
//! `execv`s us so the sidecar lives as a fresh process with no
//! inherited address space (which is necessary for the eviction
//! syscalls to actually free the pages — running them from a CoW'd
//! child empirically frees ~230 MiB out of 6 GiB; running them from a
//! fresh address space frees the lot).

use std::ffi::CString;

fn main() {
    let mut args = std::env::args().skip(1).peekable();
    let mut wait_fd: Option<i32> = None;
    let mut parent_pid: Option<i32> = None;
    let mut paths: Vec<String> = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--wait-fd" => {
                wait_fd = args.next().and_then(|s| s.parse().ok());
            }
            "--parent-pid" => {
                parent_pid = args.next().and_then(|s| s.parse().ok());
            }
            other => paths.push(other.to_string()),
        }
    }
    if paths.is_empty() {
        eprintln!("aegisllm-evict: no shard paths given");
        std::process::exit(2);
    }

    if let (Some(fd), Some(ppid)) = (wait_fd, parent_pid) {
        wait_for_parent_exit(fd, ppid);
    }

    for path in &paths {
        if let Ok(c) = CString::new(path.as_str()) {
            evict(&c);
        }
    }

    // Trigger NVIDIA-driver and kernel slab shrinkers via gentle
    // memory pressure. `cuMemAllocHost` allocates from a per-driver
    // pinned-page pool; when the parent process dies, the driver
    // does NOT return pool pages to the system free list — they
    // stay reserved for fast re-allocation. Only `drop_caches=2`
    // (which iterates registered shrinkers and calls NVIDIA's
    // shrinker callback) or organic memory pressure releases them.
    // Without root we can't `drop_caches`, so we manufacture
    // pressure: allocate anon RAM in small chunks, touch every
    // page, watch MemAvailable, stop at a safe cliff, and release.
    //
    // Opt-in via env var: AEGIS_RECLAIM_PINNED=1. Off by default
    // because briefly consuming most of RAM can disturb other
    // apps (latency spike, swap activity) on systems where the
    // user doesn't actually care about post-kill memory.
    if std::env::var_os("AEGIS_RECLAIM_PINNED").is_some() {
        force_driver_shrinker();
    }

    // Last-resort: call `sudo -n sysctl vm.drop_caches=3` if the user
    // explicitly opted in via AEGIS_DROP_CACHES_VIA_SUDO=1 AND has
    // configured passwordless sudo for the command. Empirically this is
    // the only reliable way to release the NVIDIA driver's pinned-page
    // pool (~14 GiB for a Gemma-4-26B `store=ram` load) on a non-root
    // user without root-level invasive changes.
    //
    // Required one-time sudoers config (run once as root):
    //
    //   echo "daniil ALL=(root) NOPASSWD: /usr/bin/sysctl vm.drop_caches=3" \
    //       | sudo EDITOR='tee' visudo -f /etc/sudoers.d/aegisllm-drop-caches
    //
    // After that, set AEGIS_DROP_CACHES_VIA_SUDO=1 in your env and the
    // sidecar will reclaim the NVIDIA pool automatically on every kill.
    // `sudo -n` (non-interactive) means we never prompt for a password —
    // if sudoers isn't configured, the call fails silently and we exit.
    if std::env::var_os("AEGIS_DROP_CACHES_VIA_SUDO").is_some() {
        // sync first so any in-flight writeback finishes before drop_caches.
        let _ = std::process::Command::new("sync").status();
        let _ = std::process::Command::new("sudo")
            .args(["-n", "sysctl", "vm.drop_caches=3"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
    }
}

/// Ramp anon-memory pressure to coax the kernel into running its
/// slab shrinkers, which in turn invoke the NVIDIA driver module's
/// shrinker callback to release its pinned-page pool.
///
/// Safety budget: we stop allocating as soon as MemAvailable falls
/// below `min_avail_mib` (default 384 MiB) — that's far enough into
/// pressure that shrinkers fire, but with enough headroom that other
/// processes don't fail allocations or get OOM-killed. We also bump
/// our own `oom_score_adj` to the max so if the kernel does need to
/// kill something, it's us.
fn force_driver_shrinker() {
    let _ = std::fs::write("/proc/self/oom_score_adj", "1000");

    // Read MemAvailable ceiling we should not cross.
    let starting_avail_kb = match read_meminfo("MemAvailable:") {
        Some(v) => v,
        None => return,
    };
    if starting_avail_kb < 1024 * 1024 {
        // Less than 1 GiB — already tight, don't risk it.
        return;
    }

    let chunk_bytes: usize = 256 * 1024 * 1024; // 256 MiB
    let page_size: usize = 4096;
    let min_avail_kb: u64 = 384 * 1024; // never let MemAvailable go below 384 MiB
    let max_chunks: usize = 80; // hard cap: 80 * 256 MiB = 20 GiB
    let mut chunks: Vec<(*mut libc::c_void, usize)> = Vec::with_capacity(max_chunks);

    for _ in 0..max_chunks {
        let cur_avail = read_meminfo("MemAvailable:").unwrap_or(0);
        if cur_avail < min_avail_kb {
            break;
        }
        let addr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                chunk_bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_POPULATE,
                -1,
                0,
            )
        };
        if addr == libc::MAP_FAILED {
            break;
        }
        // MAP_POPULATE asks the kernel to pre-fault, but is advisory;
        // touch one byte per page to guarantee commit.
        let p = addr as *mut u8;
        let mut off: usize = 0;
        while off < chunk_bytes {
            unsafe { std::ptr::write_volatile(p.add(off), 0u8) };
            off += page_size;
        }
        chunks.push((addr, chunk_bytes));
    }

    // Hold for ~50 ms so shrinkers and PCP drain finish their
    // passes before we release. (Shrinkers run synchronously when
    // the alloc path enters reclaim, but background drain can
    // continue briefly.)
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 50_000_000 };
    unsafe { libc::nanosleep(&mut ts, std::ptr::null_mut()); }

    // Release every chunk. MADV_DONTNEED tells the kernel to drop
    // the pages immediately rather than caching them in PCP for
    // our (nonexistent) future use.
    for (addr, len) in chunks {
        unsafe {
            libc::madvise(addr, len, libc::MADV_DONTNEED);
            libc::munmap(addr, len);
        }
    }
}

fn read_meminfo(key: &str) -> Option<u64> {
    std::fs::read_to_string("/proc/meminfo").ok().and_then(|s| {
        s.lines()
            .find(|l| l.starts_with(key))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|n| n.parse::<u64>().ok())
    })
}

/// Block until the parent at `parent_pid` has fully exited and been
/// reaped by init (its `/proc/<pid>` directory disappears). The pipe
/// fd `fd` returns EOF when the parent's last write end closes —
/// which happens when its fd table is torn down at the START of
/// `do_exit()`, BEFORE `exit_mm()`. Polling /proc/<pid>/status until
/// the entry disappears ensures `exit_mm()` has finished and our
/// eviction sees a fully un-mapped inode.
///
/// Cap the poll at 30 s so a stuck parent doesn't leave the helper
/// pinned forever.
fn wait_for_parent_exit(fd: i32, parent_pid: i32) {
    // Block reading from the inherited pipe fd. EOF arrives the moment
    // the kernel closes the parent's last write end (which it does at
    // the start of `do_exit()` before tearing down the parent's mm).
    let mut buf = [0u8; 64];
    loop {
        let n = unsafe { libc::read(fd, buf.as_mut_ptr() as *mut _, buf.len()) };
        if n == 0 {
            break; // EOF — parent exited
        }
        if n < 0 {
            let err = unsafe { *libc::__errno_location() };
            if err == libc::EINTR {
                continue;
            }
            // Unexpected error; fall through to the /proc poll which
            // will eventually time out.
            break;
        }
        // Bytes received unexpectedly; ignore.
    }
    unsafe { libc::close(fd); }

    // Then poll /proc/<parent>/status until the directory disappears
    // (kernel removes it once the task is fully reaped). This ensures
    // `exit_mm()` has finished and the shard pages are no longer
    // mapped by the parent. Cap at 30 s so a stuck parent doesn't
    // pin us forever.
    let path = format!("/proc/{parent_pid}/status");
    let start = std::time::Instant::now();
    while std::path::Path::new(&path).exists() {
        if start.elapsed() > std::time::Duration::from_secs(30) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

fn evict(path: &std::ffi::CStr) {
    unsafe {
        let fd = libc::open(path.as_ptr(), libc::O_RDONLY);
        if fd < 0 {
            return;
        }
        let len = libc::lseek(fd, 0, libc::SEEK_END);
        let _ = libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED);
        if len > 0 {
            let len_sz = len as libc::size_t;
            let addr = libc::mmap(
                std::ptr::null_mut(),
                len_sz,
                libc::PROT_READ,
                libc::MAP_SHARED,
                fd,
                0,
            );
            if addr != libc::MAP_FAILED {
                let _ = libc::madvise(addr, len_sz, libc::MADV_DONTNEED);
                let _ = libc::munmap(addr, len_sz);
            }
        }
        let _ = libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED);
        let _ = libc::close(fd);
    }
}

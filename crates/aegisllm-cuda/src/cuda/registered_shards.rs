//! Page-lock the safetensors shard mmaps that hold host-resident
//! weights with `cuMemHostRegister`, so per-token H2D streaming during
//! inference takes the direct-DMA fast path instead of bouncing
//! through a per-slot CPU memcpy.
//!
//! ## Why this exists
//!
//! The shard-mmap-as-storage refactor (one commit back) put expert
//! weights and the BF16 embed table in file-backed pages instead of
//! a pre-pinned anonymous arena. That collapsed peak host RAM (no
//! more 12-14 GiB anon allocation) but made every per-token H2D pay
//! a CPU `memcpy mmap_slice → per-slot pinned bounce → memcpy_htod`
//! round-trip — about ~30 ms/token for a top-K=8 MoE on Gemma-4-26B,
//! halving decode tps.
//!
//! Registering the shard mmaps with the CUDA driver tells it "these
//! pages are host-pinned for DMA" — `memcpy_htod` from any pointer
//! inside a registered range becomes a single DMA hop, no CPU memcpy.
//! Pages stay in the kernel page cache (file-backed, counted as
//! `Cached`), but `cuMemHostRegister` mlocks them so they don't get
//! evicted. Net memory cost ≈ the prior pinned arena, just file-
//! backed instead of anonymous.
//!
//! ## What's registered
//!
//! Only shards that actually contain at least one host-resident
//! weight. Tracked by the loader as it builds `HostWeightBytes::Mmap`
//! variants; shards holding only VRAM-resident bytes (e.g. an
//! lm_head-only shard) stay unregistered and the kernel can reclaim
//! their pages under memory pressure once the load-time upload is
//! done.
//!
//! ## Lifetime
//!
//! The `RegisteredShards` struct is stored in the executor alongside
//! the loaded weights. On drop (executor teardown), it calls
//! `cuMemHostUnregister` on each shard, then drops the `Arc<Mmap>`
//! clones — which drops the kernel mmap if no other reference is
//! held. Both the unregister and the munmap happen before process
//! exit so no NVIDIA driver state leaks.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use cudarc::driver::sys;
use memmap2::Mmap;

use aegisllm_base::error::Result;
use aegisllm_base::tensor::storage::TensorStorageLoader;

#[derive(Debug, Default)]
pub(crate) struct RegisteredShards {
    /// Held shard mmaps; on drop, each is unregistered and its
    /// reference is released. The `Arc` stays alive past the loader
    /// because the per-tensor `LoadedHostTensor`s also clone it.
    shards: Vec<Arc<Mmap>>,
}

impl RegisteredShards {
    pub(crate) fn empty() -> Self {
        Self { shards: Vec::new() }
    }

    /// Register every shard in `paths` that the loader has cached.
    /// Skips paths the loader doesn't know about (no-op if nothing
    /// host-resident loaded from that shard). On per-shard
    /// registration failure, logs to stderr and skips that shard —
    /// inference will fall back to the bounce path for those bytes.
    pub(crate) fn register(
        loader: &TensorStorageLoader,
        paths: &HashSet<PathBuf>,
    ) -> Result<Self> {
        let mut shards: Vec<Arc<Mmap>> = Vec::with_capacity(paths.len());
        let mut total_bytes: usize = 0;
        for path in paths {
            let Some(map) = loader.shard_mmap(path) else {
                continue;
            };
            let ptr = map.as_ptr();
            let len = map.len();
            if len == 0 {
                continue;
            }
            // cuMemHostRegister requires the range to be page-aligned.
            // mmap allocations are always page-aligned at the start;
            // the length must be rounded up to a page multiple. The
            // kernel-mapped trailing slack (between EOF and the next
            // page boundary) is part of our VMA, so registering it is
            // safe — those bytes are zero-filled and we never read
            // them as tensor data.
            let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
            let len_aligned = len.div_ceil(page_size) * page_size;
            let r = unsafe {
                sys::cuMemHostRegister_v2(
                    ptr as *mut _,
                    len_aligned,
                    sys::CU_MEMHOSTREGISTER_PORTABLE,
                )
            };
            if r != sys::CUresult::CUDA_SUCCESS {
                eprintln!(
                    "registered_shards: cuMemHostRegister on `{}` ({:.1} MiB) failed: {:?}; \
                     inference will use the bounce path for this shard",
                    path.display(),
                    len_aligned as f64 / (1024.0 * 1024.0),
                    r,
                );
                continue;
            }
            total_bytes += len_aligned;
            shards.push(map);
        }
        eprintln!(
            "load-timing: registered {} shard mmap(s) with cuMemHostRegister ({:.2} GiB)",
            shards.len(),
            total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        );
        Ok(Self { shards })
    }

    /// Number of shards we have an active registration on. Used by
    /// debug logging.
    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        self.shards.len()
    }
}

impl Drop for RegisteredShards {
    fn drop(&mut self) {
        for map in &self.shards {
            // SAFETY: matched with the `cuMemHostRegister_v2` in
            // `register`. cudarc's sys layer tolerates a failing
            // unregister (e.g. if the driver is already shut down on
            // process tear-down); we don't surface the error since
            // there's nothing the caller can do at drop time.
            unsafe {
                let _ = sys::cuMemHostUnregister(map.as_ptr() as *mut _);
            }
        }
    }
}

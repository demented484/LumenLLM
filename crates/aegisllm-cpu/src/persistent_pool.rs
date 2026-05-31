//! A persistent, hot, core-pinned worker pool for the experts-on-CPU MoE
//! dispatch.
//!
//! # Why not just rayon?
//!
//! The fused NVFP4 expert kernel ([`crate::nvfp4_gemv::moe_layer_experts_into`])
//! is DRAM-bandwidth-bound. Run STANDALONE in a tight loop it sustains ~37 GB/s
//! (15 ms for the 540 MiB/token active set) because rayon's worker threads stay
//! hot between the back-to-back parallel regions. But in the INTEGRATED decode
//! path, ~0.3 ms of CUDA FFI (the blocking input download, the kernel launches,
//! the result upload) runs on the calling thread BETWEEN each MoE layer's two
//! parallel regions. That idle gap is long enough for rayon's workers to go to
//! sleep, so every one of the 80 parallel regions/token (2 per layer × 40
//! layers) pays a futex wake-up on ~11 workers. Measured: the SAME kernel that
//! hits 37 GB/s standalone realizes only ~22 GB/s integrated — a 67% slowdown
//! that is pure thread wake-up latency, not the kernel and not DRAM.
//!
//! This pool fixes that. Worker threads are spawned once, pinned to distinct
//! CPUs, and **spin** on a generation counter for a bounded window (covering the
//! inter-layer FFI gap) before parking. So across a token's 80 regions the
//! workers never actually sleep — they busy-wait through the short FFI gaps and
//! are already running when the next region is posted. Between tokens (a longer
//! gap) they park, and the first region of the next token re-wakes them once
//! (cheap, amortized over 80 regions).
//!
//! # The work model
//!
//! [`PersistentPool::dispatch`] runs a single "parallel broadcast": the caller
//! supplies a total job count `n` and a `Fn(usize) + Sync` closure; the pool
//! statically partitions `[0, n)` into one near-equal contiguous shard per
//! worker (including the calling thread, which participates) and each thread
//! calls the closure for every index in its shard. Static contiguous
//! partitioning suits the uniform-cost row jobs the MoE dispatch posts (every
//! row is the same shape) and avoids per-index atomic contention. The call is
//! synchronous: `dispatch` returns only after every index has been processed.
//!
//! # Safety
//!
//! The closure is type-erased to a `*const (dyn Fn(usize) + Sync)` so it can
//! cross the thread boundary without an allocation per dispatch. SAFETY: the
//! pointer is valid for the entire `dispatch` call (the closure outlives the
//! synchronous wait), `dispatch` blocks until all workers have finished reading
//! it, and the closure is `Sync` so concurrent `&` access from the workers is
//! sound. No worker touches the pointer outside the generation it was posted
//! for (guarded by the per-generation `pending` counter).

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread;

/// Type-erased parallel-for job: process index `i` in `[0, n)`.
type JobFn<'a> = dyn Fn(usize) + Sync + 'a;

/// Shared state every worker spins on. One generation per `dispatch`.
struct Shared {
    /// Bumped once per `dispatch`; workers spin until it changes, then run.
    generation: AtomicU64,
    /// Number of worker threads still running the current generation's shard
    /// (the main thread waits for this to hit 0).
    pending: AtomicUsize,
    /// Total job count for the current generation.
    n: AtomicUsize,
    /// Number of participating lanes (worker threads + 1 for the caller).
    lanes: AtomicUsize,
    /// Raw pointer to the current generation's `JobFn`. SAFETY: see module docs.
    job: AtomicPtr,
    /// Set once at shutdown so workers exit their spin loop.
    shutdown: std::sync::atomic::AtomicBool,
    /// Worker thread handles, populated once at spawn, for the deep-idle wakeup.
    /// After a worker spins then yields past its budget without a new
    /// generation it `park`s; `dispatch` `unpark`s each one. `unpark` carries a
    /// token (an unpark that races ahead of the park makes the next park return
    /// immediately), so unlike a condvar this CANNOT lose a wakeup — which is
    /// why we use it for the hot re-entry path. Wrapped in `OnceLock` because
    /// the handles only exist after the threads are spawned (which needs the
    /// `Arc<Shared>` they capture).
    worker_threads: OnceLock<Vec<thread::Thread>>,
    /// Number of workers currently in (or about to enter) the deep `park`.
    /// `dispatch` only issues the unpark loop when this is non-zero, so the
    /// hot intra-token path (workers spinning) pays a single relaxed load
    /// instead of one `unpark` syscall-ish call per worker per region.
    parked: AtomicUsize,
}

/// `AtomicPtr`-like cell for the type-erased fat pointer to the job closure.
/// A `dyn` pointer is two words; we store it as two `usize` atomics (data +
/// vtable). Only ever read inside a generation the worker has observed.
struct AtomicPtr {
    data: AtomicUsize,
    vtable: AtomicUsize,
}

impl AtomicPtr {
    const fn new() -> Self {
        Self { data: AtomicUsize::new(0), vtable: AtomicUsize::new(0) }
    }
    #[inline]
    fn store(&self, p: *const JobFn<'_>) {
        // Decompose the fat pointer into (data, vtable). `*const dyn` is two
        // usize-sized words on every supported target.
        let raw: (usize, usize) = unsafe { std::mem::transmute::<*const JobFn<'_>, (usize, usize)>(p) };
        self.vtable.store(raw.1, Ordering::Relaxed);
        // data published last with Release so a worker that reads it sees vtable.
        self.data.store(raw.0, Ordering::Release);
    }
    #[inline]
    fn load(&self) -> *const JobFn<'static> {
        let data = self.data.load(Ordering::Acquire);
        let vtable = self.vtable.load(Ordering::Relaxed);
        unsafe { std::mem::transmute::<(usize, usize), *const JobFn<'static>>((data, vtable)) }
    }
}

/// A persistent, core-pinned, spin-then-park worker pool.
pub struct PersistentPool {
    shared: Arc<Shared>,
    workers: Vec<thread::JoinHandle<()>>,
    /// Total lanes (worker threads + the calling thread).
    lanes: usize,
}

/// Bounded spin budget (iterations) a worker burns waiting for the next
/// generation before it falls back to `thread::yield_now`. Sized so the busy
/// window comfortably covers the ~0.3 ms inter-layer CUDA FFI gap on this class
/// of CPU; after the budget the worker yields (cheap) rather than parking, so
/// re-entry stays sub-microsecond during a token while not pegging a core
/// forever between tokens.
/// Hot-spin budget (iterations) before a worker drops to `yield_now`. Kept
/// SMALL on purpose: a long 100%-busy spin between the sub-millisecond
/// inter-layer gaps both burns the CPU's all-core turbo budget (slowing the
/// bandwidth kernel itself) AND contends with the CUDA driver's helper threads
/// that the GPU-issuing main thread depends on. A short spin keeps re-entry
/// latency at a few hundred ns without holding a core hot the whole gap; the
/// `yield_now` fallback then lets the scheduler reclaim the core for the driver.
/// Tunable via `AEGIS_CPU_MOE_SPIN` (iterations) for A/B sweeps.
fn spin_budget() -> u64 {
    static V: OnceLock<u64> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("AEGIS_CPU_MOE_SPIN").ok().and_then(|s| s.parse().ok()).unwrap_or(2_048)
    })
}

/// After the hot-spin budget, a worker cooperatively `yield_now`s for this many
/// iterations before deep-parking. Covers a brief lull (e.g. the per-token
/// sampling tail) without parking, while bounding the busy window so an idle
/// engine drops to a real `park` within a few ms instead of pegging a core.
/// Tunable via `AEGIS_CPU_MOE_YIELD` (iterations) for A/B sweeps.
fn yield_budget() -> u64 {
    static V: OnceLock<u64> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("AEGIS_CPU_MOE_YIELD").ok().and_then(|s| s.parse().ok()).unwrap_or(2_000_000)
    })
}

impl PersistentPool {
    /// Create a pool with `lanes` total participating threads (the caller +
    /// `lanes - 1` spawned workers). `lanes` is clamped to `[1, ncpus]`.
    fn new(lanes: usize) -> Self {
        let lanes = lanes.max(1);
        let shared = Arc::new(Shared {
            generation: AtomicU64::new(0),
            pending: AtomicUsize::new(0),
            n: AtomicUsize::new(0),
            lanes: AtomicUsize::new(lanes),
            job: AtomicPtr::new(),
            shutdown: std::sync::atomic::AtomicBool::new(false),
            worker_threads: OnceLock::new(),
            parked: AtomicUsize::new(0),
        });
        let mut workers = Vec::with_capacity(lanes.saturating_sub(1));
        let mut thread_handles = Vec::with_capacity(lanes.saturating_sub(1));
        // Worker lane indices 1..lanes (lane 0 is the calling thread). Pin each
        // to its lane's CPU so the L2/DRAM access pattern stays stable.
        for lane in 1..lanes {
            let sh = Arc::clone(&shared);
            let handle = thread::Builder::new()
                .name(format!("aegis-moe-{lane}"))
                .spawn(move || worker_loop(sh, lane))
                .expect("spawn aegis-moe worker");
            thread_handles.push(handle.thread().clone());
            workers.push(handle);
        }
        // Publish the thread handles for the deep-idle unpark path.
        let _ = shared.worker_threads.set(thread_handles);
        // Pin the calling (GPU-issuing) lane-0 thread to CPU 0 (default ON;
        // opt out with AEGIS_CPU_MOE_PIN_MAIN=0). Measured DECISIVE on the
        // 6-core test host: without it the scheduler intermittently co-locates
        // the GPU-issuing thread on a CPU already running a spinning worker,
        // which halves decode throughput (~26 vs ~37 tps, bimodal). Pinning
        // lane 0 → CPU 0 and the workers → CPUs 1..lanes gives each physical
        // core exactly one hot thread → stable. (An earlier note found pinning
        // slower, but that was before the kernel software-prefetch lifted the
        // compute to the DRAM ceiling — at the higher core utilization the
        // collision became the dominant variance source.)
        let pin_main = std::env::var("AEGIS_CPU_MOE_PIN_MAIN").map(|v| v != "0").unwrap_or(true);
        if pin_main {
            pin_to_cpu(0);
        }
        Self { shared, workers, lanes }
    }

    /// Number of participating lanes (worker threads + the caller).
    #[inline]
    pub fn lanes(&self) -> usize {
        self.lanes
    }

    /// Run `f(i)` for every `i` in `[0, n)` across the pool, returning only once
    /// all indices have been processed. The calling thread participates as lane
    /// 0 (so a 1-lane pool just runs the loop inline with no threading at all).
    pub fn dispatch<F>(&self, n: usize, f: F)
    where
        F: Fn(usize) + Sync,
    {
        if n == 0 {
            return;
        }
        let lanes = self.lanes;
        if lanes <= 1 {
            for i in 0..n {
                f(i);
            }
            return;
        }
        // Publish the job. Workers that observe the new generation will run
        // their shard; we (lane 0) run shard 0 directly, then wait.
        let job: &JobFn<'_> = &f;
        let job_ptr: *const JobFn<'_> = job;
        self.shared.n.store(n, Ordering::Relaxed);
        self.shared.lanes.store(lanes, Ordering::Relaxed);
        self.shared.job.store(job_ptr);
        // pending = worker lanes (lane 0 = caller does not count itself).
        self.shared.pending.store(lanes - 1, Ordering::Release);
        // Bump generation last → workers wake and read a fully-published job.
        self.shared.generation.fetch_add(1, Ordering::Release);
        // Wake any workers that deep-parked during an idle gap. `unpark` carries
        // a wakeup token, so even if a worker is between bumping `parked` and its
        // `park` call the token makes the next `park` return at once — no lost
        // wakeup. The `parked > 0` guard means the hot intra-token path (workers
        // still spinning) skips the unpark loop entirely after one relaxed load.
        // The Acquire pairs with the worker's Release bump of `parked` so a
        // non-zero read here cannot miss a worker that has committed to parking.
        if self.shared.parked.load(Ordering::Acquire) != 0 {
            if let Some(threads) = self.shared.worker_threads.get() {
                for t in threads {
                    t.unpark();
                }
            }
        }

        // Lane 0 runs its own shard inline.
        run_shard(&f, 0, lanes, n);

        // Wait for all worker lanes to finish this generation. Spin (workers are
        // hot); the regions are sub-millisecond so a spin-wait is cheapest.
        let mut spins = 0u64;
        while self.shared.pending.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
            spins += 1;
            if spins.is_multiple_of(1024) {
                std::thread::yield_now();
            }
        }
    }
}

impl Drop for PersistentPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        // Wake workers out of their spin/park so they observe shutdown.
        self.shared.generation.fetch_add(1, Ordering::Release);
        if let Some(threads) = self.shared.worker_threads.get() {
            for t in threads {
                t.unpark();
            }
        }
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

/// Compute and run lane `lane`'s contiguous shard of `[0, n)`.
#[inline]
fn run_shard<F: Fn(usize)>(f: &F, lane: usize, lanes: usize, n: usize) {
    // Near-equal contiguous split: first `rem` lanes get `base + 1` indices.
    let base = n / lanes;
    let rem = n % lanes;
    let (start, len) = if lane < rem {
        (lane * (base + 1), base + 1)
    } else {
        (rem * (base + 1) + (lane - rem) * base, base)
    };
    for i in start..start + len {
        f(i);
    }
}

/// Worker thread body: pin, then spin-then-yield waiting for each new
/// generation, run this lane's shard, mark done, repeat until shutdown.
fn worker_loop(shared: Arc<Shared>, lane: usize) {
    pin_to_cpu(lane);
    // Start from the pool's construction-time generation (0). We must NOT load
    // the live counter here: a dispatch can bump it before this thread finishes
    // spawning, and loading the already-bumped value would make the worker skip
    // that generation's shard (leaving `pending` stuck → the dispatcher would
    // spin forever). Since `new()` returns the only handle through which a
    // dispatch can occur, every generation the pool ever posts starts at 1, and
    // each worker observes them in order starting from 0.
    let mut last_gen: u64 = 0;
    let spin_budget = spin_budget();
    let yield_budget = yield_budget();
    loop {
        // Wait for the next generation in three escalating phases so we stay hot
        // through a token's sub-millisecond inter-layer gaps but do NOT peg a
        // core when the engine is idle between requests:
        //   1. hot spin (covers the ~0.3 ms inter-layer CUDA FFI gap),
        //   2. cooperative yield (a few ms of light idle),
        //   3. deep park on the condvar (engine idle → sleep until notified).
        let mut spins = 0u64;
        let new_gen = loop {
            let g = shared.generation.load(Ordering::Acquire);
            if g != last_gen {
                break g;
            }
            spins += 1;
            if spins < spin_budget {
                std::hint::spin_loop();
            } else if spins < spin_budget + yield_budget {
                std::thread::yield_now();
            } else {
                // Deep idle: park (with a timeout backstop). Announce intent to
                // park (Release) BEFORE the final generation re-check so a
                // concurrent `dispatch` that bumps the generation will observe
                // `parked != 0` and unpark us; the unpark token then makes our
                // `park_timeout` return at once (no lost wakeup). Re-check the
                // generation AFTER bumping `parked` to also catch a generation
                // that advanced just before we committed to parking.
                shared.parked.fetch_add(1, Ordering::Release);
                if shared.generation.load(Ordering::Acquire) == last_gen
                    && !shared.shutdown.load(Ordering::Acquire)
                {
                    std::thread::park_timeout(std::time::Duration::from_millis(10));
                }
                shared.parked.fetch_sub(1, Ordering::Release);
            }
        };
        last_gen = new_gen;
        if shared.shutdown.load(Ordering::Acquire) {
            return;
        }
        // Read the published job + dims for this generation.
        let n = shared.n.load(Ordering::Relaxed);
        let lanes = shared.lanes.load(Ordering::Relaxed);
        let job_ptr = shared.job.load();
        if lane < lanes {
            // SAFETY: the main thread keeps the closure alive until `pending`
            // reaches 0 (it spins on it); the closure is `Sync` so a shared `&`
            // is sound; `job_ptr` was published with Release before the
            // generation bump we just observed with Acquire.
            let f: &JobFn<'static> = unsafe { &*job_ptr };
            run_shard_dyn(f, lane, lanes, n);
        }
        // Mark this lane done for the generation.
        shared.pending.fetch_sub(1, Ordering::Release);
    }
}

/// `run_shard` for the type-erased `dyn Fn`.
#[inline]
fn run_shard_dyn(f: &JobFn<'_>, lane: usize, lanes: usize, n: usize) {
    let base = n / lanes;
    let rem = n % lanes;
    let (start, len) = if lane < rem {
        (lane * (base + 1), base + 1)
    } else {
        (rem * (base + 1) + (lane - rem) * base, base)
    };
    for i in start..start + len {
        f(i);
    }
}

/// Best-effort pin of the current thread to `cpu` (Linux only; no-op elsewhere
/// or on failure). Keeps the worker's DRAM/L2 access pattern stable across
/// dispatches without depending on a thread-affinity crate.
#[cfg(target_os = "linux")]
fn pin_to_cpu(cpu: usize) {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu, &mut set);
        // tid 0 = current thread.
        let _ = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

#[cfg(not(target_os = "linux"))]
fn pin_to_cpu(_cpu: usize) {}

/// The process-wide MoE dispatch pool. Lazily created on first use, sized to
/// the number of PHYSICAL-ish cores: a DRAM-bandwidth-bound kernel saturates
/// memory channels well before it needs SMT siblings, and pinning one lane per
/// distinct core keeps the channels fed without intra-core contention. We use
/// `min(ncpus, 6)`-style sizing via the env override or a default that targets
/// the dispatch's measured sweet spot.
static POOL: OnceLock<PersistentPool> = OnceLock::new();

/// Default lane count: half the logical CPUs (one per physical core on an
/// SMT-2 machine), clamped to `[1, ncpus]`. Overridable with
/// `AEGIS_CPU_MOE_LANES` for tuning. The kernel is DRAM-bound, so adding SMT
/// siblings does not add bandwidth — one lane per physical core is the sweet
/// spot and leaves a sibling free for the GPU-issuing thread.
fn default_lanes() -> usize {
    if let Ok(v) = std::env::var("AEGIS_CPU_MOE_LANES") {
        if let Ok(n) = v.parse::<usize>() {
            return n.max(1);
        }
    }
    let ncpus = available_parallelism();
    // One lane per physical core (assume SMT-2); at least 1.
    (ncpus / 2).max(1)
}

fn available_parallelism() -> usize {
    thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// A/B switch: when `AEGIS_CPU_MOE_RAYON=1` the MoE kernel dispatches its two
/// parallel regions on rayon instead of this pool. Kept for measurement and as
/// the safe fallback on hosts where the pool's spin nets a regression.
pub fn use_rayon_fallback() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("AEGIS_CPU_MOE_RAYON").map(|v| v == "1").unwrap_or(false))
}

/// Access the process-wide persistent MoE dispatch pool, creating it on first
/// call. Safe to call from the decode hot path (the `OnceLock` fast path is a
/// single relaxed load after init).
pub fn global_pool() -> &'static PersistentPool {
    POOL.get_or_init(|| PersistentPool::new(default_lanes()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn dispatch_covers_every_index_exactly_once() {
        let pool = PersistentPool::new(4);
        for &n in &[0usize, 1, 3, 7, 16, 33, 1000] {
            let hits: Vec<AtomicU64> = (0..n).map(|_| AtomicU64::new(0)).collect();
            pool.dispatch(n, |i| {
                hits[i].fetch_add(1, Ordering::Relaxed);
            });
            for (i, h) in hits.iter().enumerate() {
                assert_eq!(h.load(Ordering::Relaxed), 1, "index {i} (n={n})");
            }
        }
    }

    #[test]
    fn dispatch_sums_correctly_under_repeated_use() {
        let pool = PersistentPool::new(6);
        // Re-dispatch many times to exercise the generation counter + spin path.
        for round in 0..50 {
            let n = 257;
            let acc = AtomicU64::new(0);
            pool.dispatch(n, |i| {
                acc.fetch_add((i as u64) + round, Ordering::Relaxed);
            });
            let expected: u64 = (0..n as u64).map(|i| i + round).sum();
            assert_eq!(acc.load(Ordering::Relaxed), expected, "round {round}");
        }
    }

    #[test]
    fn single_lane_runs_inline() {
        let pool = PersistentPool::new(1);
        let acc = AtomicU64::new(0);
        pool.dispatch(100, |i| {
            acc.fetch_add(i as u64, Ordering::Relaxed);
        });
        assert_eq!(acc.load(Ordering::Relaxed), (0..100u64).sum::<u64>());
    }
}

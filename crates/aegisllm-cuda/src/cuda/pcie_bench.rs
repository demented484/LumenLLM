//! STEP 0 microbench — raw copy-engine H2D ceiling.
//!
//! Answers THE gating question for the offloaded-MoE decode perf work: when we
//! issue back-to-back `cuMemcpyHtoDAsync` from **pinned** host RAM to VRAM on a
//! single (transfer) stream with NO interleaved compute / no per-transfer
//! event fences, can the PCIe-5 copy engine exceed the ~28 GB/s the decode
//! path measures today, and how close does it get to the ~56 GB/s link
//! ceiling?
//!
//! The decode path's 28 GB/s is suspected to be a *stalling* artifact: the
//! 4-slot `LinearStagingPool` fences every transfer on the prior compute
//! (slot reuse) so the copy engine idles between bursts. This bench removes
//! every fence and every kernel — it is the pure DMA ceiling. If it lands near
//! 56, the FALLBACK (saturate the copy engine: more slots, decouple transfer
//! from compute) is worthwhile. If it caps near 28, the link/driver is the
//! limit and only a fundamentally different transfer (e.g. GPU-driven gather,
//! already a dead end) could go faster — meaning we should NOT chase the
//! ceiling on this path.
//!
//! Size classes mirror the real workload:
//!   * 3.2 MiB × many  — one NVFP4 expert projection's packed bytes; the decode
//!     path issues `top_k × 3` of these per MoE layer.
//!   * 25 MiB          — roughly one full layer's active-expert packed bytes
//!     (top_k=8 × 3 projections), i.e. the burst we'd like to issue back-to-back.
//!   * 256 MiB         — large enough to amortize all per-call overhead and show
//!     the asymptotic link bandwidth.
//!
//! Run (needs the GPU; ignored by default so `cargo test` stays host-only):
//!   CUTLASS_DIR=/home/daniil/LM-experements/cutlass CUDA_HOME=/opt/cuda \
//!     cargo test -p aegisllm-cuda --release pcie_h2d_ceiling -- --ignored --nocapture

#[cfg(test)]
mod tests {
    use crate::cuda::owned_pinned::OwnedPinnedBuf;
    use cudarc::driver::{CudaContext, sys};

    /// Issue `iters` back-to-back HtoDAsync copies of `bytes` each from one
    /// pinned source into one VRAM dest on `stream`, then synchronize once.
    /// Returns achieved GB/s over the whole burst (total bytes / wall time).
    ///
    /// We reuse a single src/dst pair so the copy engine sees a steady stream
    /// of same-size transfers — exactly the shape the decode path issues, minus
    /// the per-transfer compute fence we're trying to eliminate. Timing is a
    /// CPU `Instant` around a single trailing `cuStreamSynchronize`: the async
    /// launches queue cheaply, so wall time ≈ device transfer time for bursts
    /// of this magnitude.
    unsafe fn time_h2d_gbps(
        stream: sys::CUstream,
        dst: sys::CUdeviceptr,
        src: *const u8,
        bytes: usize,
        iters: usize,
    ) -> f64 {
        // Warm-up: first copy after alloc pays page-table / driver setup cost.
        let rc = unsafe {
            sys::cuMemcpyHtoDAsync_v2(dst, src as *const std::ffi::c_void, bytes, stream)
        };
        assert_eq!(rc, sys::CUresult::CUDA_SUCCESS, "warmup HtoDAsync failed: {rc:?}");
        let rc = unsafe { sys::cuStreamSynchronize(stream) };
        assert_eq!(rc, sys::CUresult::CUDA_SUCCESS, "warmup sync failed: {rc:?}");

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            let rc = unsafe {
                sys::cuMemcpyHtoDAsync_v2(dst, src as *const std::ffi::c_void, bytes, stream)
            };
            assert_eq!(rc, sys::CUresult::CUDA_SUCCESS, "HtoDAsync failed: {rc:?}");
        }
        let rc = unsafe { sys::cuStreamSynchronize(stream) };
        assert_eq!(rc, sys::CUresult::CUDA_SUCCESS, "sync failed: {rc:?}");
        let secs = t0.elapsed().as_secs_f64();
        let total = (bytes as f64) * (iters as f64);
        (total / secs) / 1.0e9
    }

    #[test]
    #[ignore = "needs a CUDA device; run with --ignored"]
    fn pcie_h2d_ceiling() {
        let Ok(ctx) = CudaContext::new(0) else {
            eprintln!("skip: no CUDA device");
            return;
        };
        // A dedicated user stream (legacy/default stream would serialize against
        // everything). bind the context as current for the raw driver calls.
        let stream = ctx.new_stream().expect("new stream");
        let cu_stream = stream.cu_stream();

        // One pinned source big enough for the largest class (256 MiB). We copy
        // sub-ranges of it for the smaller classes — same pinned pages, so all
        // three classes exercise the identical fast DMA path.
        const MAX: usize = 256 * 1024 * 1024;
        let src = OwnedPinnedBuf::new(MAX).expect("alloc pinned source");
        // Touch the pages so they're committed + the pin actually locks real
        // memory (new() registers, but MAP_NORESERVE means commit-on-touch).
        // SAFETY: we own the buffer for MAX bytes.
        unsafe {
            std::ptr::write_bytes(src.as_ptr() as *mut u8, 0xA5u8, MAX);
        }
        let src_ptr = src.as_ptr();

        // One VRAM dest of MAX bytes via the driver allocator.
        let mut dst: sys::CUdeviceptr = 0;
        let rc = unsafe { sys::cuMemAlloc_v2(&mut dst, MAX) };
        assert_eq!(rc, sys::CUresult::CUDA_SUCCESS, "cuMemAlloc failed: {rc:?}");

        // (label, per-copy bytes, iters) — iters chosen so each class moves
        // ~2 GiB total (enough to average out jitter, quick enough for CI).
        let mib = 1024 * 1024;
        let classes: [(&str, usize, usize); 3] = [
            ("3.2 MiB x640", 3_355_443, 640), //   ~3.2 MiB packed expert projection
            ("25 MiB x80", 25 * mib, 80),     //   ~full layer active-expert burst
            ("256 MiB x8", 256 * mib, 8),     //   asymptotic link bandwidth
        ];

        eprintln!("\n=== PCIe-5 H2D copy-engine ceiling (pinned -> VRAM, no fences) ===");
        for (label, bytes, iters) in classes {
            let gbps =
                unsafe { time_h2d_gbps(cu_stream, dst, src_ptr, bytes, iters) };
            eprintln!(
                "  {label:>16}: {gbps:6.1} GB/s   ({iters} x {:.2} MiB)",
                bytes as f64 / mib as f64
            );
        }
        eprintln!("=== decode path measures ~28 GB/s; link ceiling ~56 GB/s ===\n");

        unsafe {
            let _ = sys::cuMemFree_v2(dst);
        }
    }
}

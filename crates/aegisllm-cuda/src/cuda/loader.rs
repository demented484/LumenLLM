use super::host_arena::ArenaHandle;
use super::repack::{
    cached_repack_nvfp4_to_cutlass_e2m1_ue4m3_host, cached_repack_nvfp4_to_mxfp4_host,
    repack_nvfp4_to_cutlass_e2m1_ue4m3_host,
};
use super::owned_pinned::OwnedPinnedBuf;
use super::runtime::{CudaRuntime, map_cuda_err};
use super::types::{
    DeviceBf16Matrix, DeviceBuffer, DeviceCutlassNvfp4Linear, DeviceMxfp4Linear, DeviceNvfp4Linear,
    HostBf16Weights, HostResidentMxfp4, HostResidentWeights, HostWeightBytes,
};
use aegisllm_base::artifact::ModelArtifact;
use aegisllm_base::error::{AegisError, Result};

/// Number of concurrent `pread` workers used by `read_chunked_par`.
/// 8 saturates a Gen4 NVMe (4-5 GB/s) without contention; more workers
/// just add `File::open` overhead per call. Empirically tuned at commit
/// 67b9b0f when 4-way (~3.2 GB/s on a 1.4 GiB BF16 tensor) was bumped
/// to 8-way (~4.4 GB/s) on the user's 5070 Ti host.
const PREAD_CHUNK_COUNT: usize = 8;

/// Allocate a process-owned pinned host buffer and copy `bytes` into it.
/// Used only for small, generated-at-load-time data (native MXFP4 repack
/// output, MatFormer submatrix slices) where the source is a transient
/// `Vec<u8>` with no file-backed mmap to point at.
fn alloc_pinned_from_bytes(
    _runtime: &CudaRuntime,
    bytes: &[u8],
    _label: &'static str,
) -> Result<OwnedPinnedBuf> {
    let mut pinned = OwnedPinnedBuf::new(bytes.len())?;
    pinned.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
    Ok(pinned)
}

/// Parallel chunked `pread` of a contiguous file range into `dst`.
/// Splits into `PREAD_CHUNK_COUNT` sub-ranges, opens that many
/// independent `File` handles, and dispatches concurrent `read_at`
/// calls on rayon workers. Each thread writes a disjoint sub-slice
/// so there's no aliasing.
///
/// Single-thread `read_exact` on a 1.4 GiB BF16 tensor runs at ~440
/// MB/s (NVMe queue depth = 1, page-cache copy + memcpy serialized).
/// 8-way pread saturates the link at ~4.4 GB/s on the user's Gen4
/// drive — about a 10× win on big tensors.
fn read_chunked_par(path: &std::path::Path, file_offset: u64, dst: &mut [u8]) -> Result<()> {
    use rayon::prelude::*;
    use std::os::unix::fs::FileExt;
    use aegisllm_base::tensor::storage::fadvise_dont_need;
    let len = dst.len();
    if len == 0 {
        return Ok(());
    }
    // Tiny tensors: single read is faster than splitting. Threshold
    // chosen so we don't spawn 8 workers for a 4 KiB scalar.
    if len < 1 << 20 {
        let file = std::fs::File::open(path)?;
        file.read_exact_at(dst, file_offset)?;
        // Evict the just-read range from the page cache: the bytes are now
        // in the arena (anonymous RAM), so the file-cache copy is dead
        // weight. Loading ~12 GiB of experts via pread otherwise balloons
        // the page cache to ~9 GiB mid-load (it counts against the cgroup
        // memory limit and inflates `MemoryCurrent`). pread pages are not
        // mmap-mapped, so POSIX_FADV_DONTNEED evicts them immediately.
        fadvise_dont_need(&file, file_offset, len as u64);
        return Ok(());
    }
    let chunk_size = len.div_ceil(PREAD_CHUNK_COUNT);
    let dst_ptr = dst.as_mut_ptr() as usize;
    (0..PREAD_CHUNK_COUNT)
        .into_par_iter()
        .try_for_each(|i| -> Result<()> {
            let chunk_start = i * chunk_size;
            if chunk_start >= len {
                return Ok(());
            }
            let chunk_end = (chunk_start + chunk_size).min(len);
            let chunk_len = chunk_end - chunk_start;
            let file = std::fs::File::open(path)?;
            // SAFETY: chunks are disjoint by construction (i × chunk_size,
            // non-overlapping ranges); `dst_ptr + chunk_start` is in-bounds
            // because `chunk_end ≤ len`.
            let chunk_dst = unsafe {
                std::slice::from_raw_parts_mut(
                    (dst_ptr as *mut u8).add(chunk_start),
                    chunk_len,
                )
            };
            let chunk_file_off = file_offset + chunk_start as u64;
            file.read_exact_at(chunk_dst, chunk_file_off)?;
            Ok(())
        })?;
    // One page-cache evict for the whole tensor range, after every chunk
    // read has completed. The bytes are in the arena (anonymous RAM) now,
    // so the file-cache copy is dead weight; loading ~12 GiB of experts
    // via pread otherwise balloons the page cache to ~9 GiB mid-load
    // (it counts against the cgroup memory limit). One fadvise per tensor
    // — not per chunk — keeps it off the concurrent-read critical path.
    if let Ok(file) = std::fs::File::open(path) {
        fadvise_dont_need(&file, file_offset, len as u64);
    }
    Ok(())
}

/// Decode a signed OCP FP8 E4M3 byte to f32 (1 sign, 4 exp bias-7, 3 mantissa).
/// E4M3 has no infinities; the lone NaN is S.1111.111. Built as a 256-entry LUT
/// so the per-element dequant loop is a single table lookup.
fn build_e4m3_signed_lut() -> [f32; 256] {
    let mut lut = [0f32; 256];
    for (b, slot) in lut.iter_mut().enumerate() {
        let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
        let exp = ((b >> 3) & 0x0F) as i32;
        let mant = (b & 0x07) as f32;
        let v = if exp == 0 {
            // Subnormal: 2^(1-bias) * (mant/8), bias = 7 → 2^-6 = 0.015625.
            (mant / 8.0) * 0.015_625
        } else if exp == 15 && (b & 0x07) == 7 {
            f32::NAN
        } else {
            (1.0 + mant / 8.0) * 2f32.powi(exp - 7)
        };
        *slot = sign * v;
    }
    lut
}

/// Round-to-nearest-even f32 → bf16 (matches PyTorch's `.bfloat16()`), so the
/// dequanted weights bit-match a torch/vLLM reference rather than truncate-biased.
#[inline]
fn f32_to_bf16_round(x: f32) -> u16 {
    let bits = x.to_bits();
    if bits & 0x7FFF_FFFF > 0x7F80_0000 {
        return 0x7FC0; // canonical bf16 NaN
    }
    let rounding_bias = 0x0000_7FFF + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

/// Dequant a DeepSeek-style FP8 block-scaled weight straight into a BF16
/// destination byte buffer (LE u16 per element), parallel over output rows.
/// `w_bf16[i,j] = e4m3(fp8[i,j]) * scale[i/blk_r, j/blk_c]`.
#[allow(clippy::too_many_arguments)]
fn dequant_fp8_block_into_bf16(
    dst: &mut [u8],
    fp8: &[u8],
    scales: &[f32],
    rows: usize,
    cols: usize,
    s_cols: usize,
    blk_r: usize,
    blk_c: usize,
    lut: &[f32; 256],
) {
    use rayon::prelude::*;
    dst.par_chunks_mut(cols * 2).enumerate().for_each(|(i, row)| {
        if i >= rows {
            return;
        }
        let sr = (i / blk_r) * s_cols;
        let base = i * cols;
        for j in 0..cols {
            let w = lut[fp8[base + j] as usize];
            let sc = scales[sr + j / blk_c];
            let bytes = f32_to_bf16_round(w * sc).to_le_bytes();
            row[2 * j] = bytes[0];
            row[2 * j + 1] = bytes[1];
        }
    });
}

/// Reserve ONE contiguous arena slot holding a projection's packed weight
/// immediately followed by its scales, and read both into it from disk via
/// parallel `pread`. Returns `(packed, scales)` arena views with the invariant
/// `scales.offset == packed.offset + packed.len` — i.e. the two are physically
/// adjacent in the pinned arena.
///
/// This adjacency lets the decode staging pool issue ONE `cuMemcpyHtoDAsync`
/// covering both packed+scales (into one contiguous device slot, viewed at
/// sub-offsets) instead of two separate copies — halving the per-expert-
/// projection H2D call count on the CPU-issue-bound decode path. Bit-identical:
/// same bytes, same order, same kernel; only the *number* of host-issued copies
/// changes. Other consumers (prefill bulk, GPU-driven gather) keep using the
/// two views independently — adjacency is a superset guarantee, transparent to
/// them.
fn read_packed_scales_contiguous_into_arena(
    arena: &ArenaHandle,
    packed: &TensorInfo,
    scales: &TensorInfo,
) -> Result<(HostWeightBytes, HostWeightBytes)> {
    let plen = packed.data_len_bytes() as usize;
    let slen = scales.data_len_bytes() as usize;
    let base = arena.reserve(plen + slen)?;
    // SAFETY: `reserve` exclusively claimed [base, base+plen+slen); the two
    // sub-ranges are disjoint and exclusively ours until released.
    let pdst = unsafe { arena.slice_mut(base, plen) };
    read_chunked_par(&packed.shard_path, packed.file_offsets.0, pdst)?;
    let sdst = unsafe { arena.slice_mut(base + plen, slen) };
    read_chunked_par(&scales.shard_path, scales.file_offsets.0, sdst)?;
    Ok((
        HostWeightBytes::Arena { arena: arena.clone(), offset: base, len: plen },
        HostWeightBytes::Arena { arena: arena.clone(), offset: base + plen, len: slen },
    ))
}

/// Like [`read_packed_scales_contiguous_into_arena`] but copies from an
/// in-memory shard buffer (the whole-shard prefetch path) rather than reading
/// from disk. Same adjacency guarantee.
fn copy_packed_scales_contiguous_into_arena(
    arena: &ArenaHandle,
    packed_src: &[u8],
    scales_src: &[u8],
) -> Result<(HostWeightBytes, HostWeightBytes)> {
    let plen = packed_src.len();
    let slen = scales_src.len();
    let base = arena.reserve(plen + slen)?;
    // SAFETY: `reserve` exclusively claimed [base, base+plen+slen); disjoint.
    let pdst = unsafe { arena.slice_mut(base, plen) };
    pdst.copy_from_slice(packed_src);
    let sdst = unsafe { arena.slice_mut(base + plen, slen) };
    sdst.copy_from_slice(scales_src);
    Ok((
        HostWeightBytes::Arena { arena: arena.clone(), offset: base, len: plen },
        HostWeightBytes::Arena { arena: arena.clone(), offset: base + plen, len: slen },
    ))
}

/// Free host RAM (Linux `MemAvailable`), used as a guard before buffering a
/// whole ~9 GiB shard so a memory-tight host (32 GiB, OS + IDE + the growing
/// pinned arena) doesn't OOM. `None` if /proc/meminfo can't be read.
fn mem_available_bytes() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: u64 = rest.trim().trim_end_matches("kB").trim().parse().ok()?;
            return Some(kb.saturating_mul(1024));
        }
    }
    None
}
use aegisllm_base::graph::{GraphRegion, TensorRole};
use aegisllm_base::planning::cuda_nvfp4_kernel_family_for_layout;
use aegisllm_base::planning::placement::{ComputePlacement, RegionPlacement, StoragePlacement};
use aegisllm_base::planning::runtime::KernelFamily;
use aegisllm_base::tensor::layout::LinearResidentLayout;
use aegisllm_base::tensor::quant::{Nvfp4LinearSpec, QK_NVFP4, QK_NVFP4_SUB};
use aegisllm_base::tensor::storage::{
    LoadedHostTensor, NestedParamSlice, TensorResidencyPlan, TensorStorageLoader,
};
use aegisllm_base::tensor::{TensorDType, TensorInfo};

/// Callback invoked by the loader to report fine-grained sub-step progress
/// (e.g. "layer 5 expert 64/128") between coarse `step` advances. The executor
/// wires this to its TTY progress indicator; cross-crate callers that don't
/// care about progress just leave it `None`.
pub type LoadStatusSink = std::sync::Arc<dyn Fn(&str) + Send + Sync>;

pub struct CudaWeightLoader<'a> {
    runtime: &'a CudaRuntime,
    /// Pinned-host arena for host-resident NVFP4 + BF16 weights. When
    /// set, host-resident loaders read tensor bytes directly into this
    /// arena (single big `cuMemHostRegister`'d allocation, sub-allocated
    /// by atomic bump). When `None`, host-resident paths cannot be
    /// taken — the executor must always build the loader via
    /// `weight_loader_with_arena(...)` for any config that has
    /// host-resident weights.
    arena: Option<ArenaHandle>,
    /// Reusable pinned-host bounce buffer for VRAM-resident BF16
    /// uploads. Replaces the prior `loader.load_for_store(_, Mmap)` →
    /// `clone_htod` path that mmap'd the shard and filled the kernel
    /// page cache by the tensor size on every read — for Gemma-4-26B
    /// with embed + lm_head VRAM-resident that's ~3 GiB of page cache
    /// retained until end of load, on top of the growing 12-14 GiB
    /// arena. Combined that pushed peak host RAM near the system
    /// limit and triggered OOM during the layer loop on 32 GiB hosts.
    /// With this buffer, each VRAM-resident BF16 read goes directly
    /// from disk into pinned memory (no page-cache fill) and DMAs to
    /// device with a single pinned→device memcpy. Sized to the
    /// largest tensor seen so far, reused across loads.
    bounce: std::cell::RefCell<Option<OwnedPinnedBuf>>,
    /// Optional callback for fine-grained sub-step progress. Heavy inner
    /// loops (MoE experts, big BF16 uploads) call `report_status(...)` so
    /// the user's progress indicator can refresh between coarse `step`
    /// advances.
    status_sink: Option<LoadStatusSink>,
    /// Single-shard read buffer for the expert prefetch. A layer's expert
    /// tensors are SCATTERED across the whole shard (the safetensors layout is
    /// not layer-grouped), so reading them per-tensor makes the NVMe seek
    /// between non-adjacent offsets (~400 MB/s). Instead we read the WHOLE shard
    /// once, sequentially (read_chunked_par over a contiguous file = a few large
    /// sequential sub-reads → ~2 GB/s) into this buffer and memcpy each tensor
    /// out of it. Holds one shard at a time (the old buffer is dropped before
    /// the next is read); layers load in order so the cache hits across all
    /// layers in a shard. RefCell: only touched serially (before the prefetch's
    /// rayon fan-out, which captures the resulting `Arc` by value).
    shard_cache: std::cell::RefCell<Option<(std::path::PathBuf, std::sync::Arc<Vec<u8>>)>>,
}

impl CudaRuntime {
    pub fn weight_loader(&self) -> CudaWeightLoader<'_> {
        CudaWeightLoader {
            runtime: self,
            arena: None,
            bounce: std::cell::RefCell::new(None),
            status_sink: None,
            shard_cache: std::cell::RefCell::new(None),
        }
    }

    /// Create a weight loader bound to a pre-allocated pinned-host arena.
    /// Required for any config with host-resident weights; non-host-resident
    /// loads (VRAM-resident BF16 / NVFP4) ignore the arena.
    pub fn weight_loader_with_arena(&self, arena: ArenaHandle) -> CudaWeightLoader<'_> {
        CudaWeightLoader {
            runtime: self,
            arena: Some(arena),
            bounce: std::cell::RefCell::new(None),
            status_sink: None,
            shard_cache: std::cell::RefCell::new(None),
        }
    }
}

impl CudaWeightLoader<'_> {
    pub fn device_index(&self) -> usize {
        self.runtime.device_index()
    }

    /// Attach a sub-step status callback (overwrites any prior sink). The
    /// loader invokes it from heavy inner loops via `report_status`.
    pub fn with_status_sink(mut self, sink: LoadStatusSink) -> Self {
        self.status_sink = Some(sink);
        self
    }

    /// Emit a sub-step status line via the attached sink, if any. Cheap
    /// no-op when no sink is set, so call-site clutter is the only cost
    /// for callers that don't wire one in.
    pub fn report_status(&self, label: &str) {
        if let Some(sink) = self.status_sink.as_ref() {
            sink(label);
        }
    }

    /// Borrow the underlying runtime for callers that need direct access to
    /// allocator / upload primitives during loading. Used by the executor's
    /// loader to populate `router_per_expert_scale_device` and similar
    /// accompanying device-resident metadata.
    pub fn runtime(&self) -> &CudaRuntime {
        self.runtime
    }

    /// Borrow the loader's pinned-host arena. Returns `None` if the
    /// loader was built bare via `weight_loader()` (no host-resident
    /// path supported). The executor inspects this to know whether
    /// `pin_now()` is needed at end of load.
    pub fn arena(&self) -> Option<&ArenaHandle> {
        self.arena.as_ref()
    }

    /// Drop the VRAM-upload bounce buffer. The bounce is high-water-
    /// mark sized to the largest VRAM-resident BF16 tensor (typically
    /// lm_head ≈ 1.4 GiB on Gemma-4-26B). After all VRAM-resident
    /// BF16 weights have been loaded (which happens BEFORE the layer
    /// loop), the bounce is dead weight that competes with the
    /// growing arena for host RAM. Callers must drop it explicitly
    /// to keep the load-time peak inside the arena's footprint.
    pub fn release_bounce(&self) {
        *self.bounce.borrow_mut() = None;
    }

    /// Ensure the loader's pinned bounce is at least `len` bytes,
    /// then read `tensor`'s bytes directly from disk into it. Uses
    /// `pread`-equivalent reads (no `seek` then `read`, just
    /// `read_exact_at`) to bypass the kernel's mmap page cache —
    /// for VRAM-resident BF16 (embed/lm_head when configured to
    /// stream once at load) this is the difference between holding
    /// ~3 GiB of page-cache pages alongside the growing 14 GiB
    /// arena (→ OOM on 32 GiB hosts) and bounded ~tensor-size
    /// transient anon pages that get reused across loads.
    fn read_tensor_into_bounce(&self, tensor: &TensorInfo) -> Result<()> {
        let len = tensor.data_len_bytes() as usize;
        let need_realloc = self
            .bounce
            .borrow()
            .as_ref()
            .map(|b| b.len() < len)
            .unwrap_or(true);
        if need_realloc {
            // Drop the old bounce first so peak memory is just the
            // new size, not old + new during construction.
            *self.bounce.borrow_mut() = None;
            *self.bounce.borrow_mut() = Some(OwnedPinnedBuf::new(len)?);
        }
        let mut bounce_ref = self.bounce.borrow_mut();
        let bounce = bounce_ref.as_mut().expect("bounce just ensured to exist");
        let bytes = bounce.as_mut_slice();
        // 8-way parallel pread saturates Gen4 NVMe (~4.4 GB/s on a
        // 1.4 GiB BF16 tensor) instead of the ~440 MB/s single-thread
        // ceiling. Big tensors (lm_head, embed) used to dominate the
        // load timeline; with chunked pread they finish in <0.5s.
        read_chunked_par(&tensor.shard_path, tensor.file_offsets.0, &mut bytes[..len])?;
        Ok(())
    }

    /// Parallel-prefetch many host-resident NVFP4 tensor pairs
    /// (`{prefix}.weight`, `{prefix}.weight_scale`) into the loader's
    /// arena. Each pair runs on a rayon worker that opens its shard,
    /// `pread`s into the arena's atomic-bump-allocated slot, and
    /// returns the host bytes for the consumer to attach to its
    /// `DeviceNvfp4Linear`. Per-tensor reads are themselves chunked
    /// (8-way) so even a single big tensor saturates the NVMe; the
    /// outer rayon parallelism hides per-call open/seek overhead
    /// across the ~384 expert tensors of one MoE layer.
    ///
    /// Returns `(packed, scales)` pairs in the same order as `prefixes`,
    /// so the caller can match them with serial CUDA-side metadata
    /// construction.
    pub fn prefetch_host_nvfp4_pairs_par<'a>(
        &self,
        artifact: &'a ModelArtifact,
        prefixes: &[String],
    ) -> Result<Vec<(HostWeightBytes, HostWeightBytes)>> {
        use rayon::prelude::*;
        let arena = self.arena.as_ref().ok_or_else(|| {
            AegisError::InvalidPlan(
                "prefetch_host_nvfp4_pairs_par requires loader built with weight_loader_with_arena".into(),
            )
        })?;
        // Gather (weight, scale) TensorInfo pairs in `prefixes` order.
        let pairs: Vec<(&TensorInfo, &TensorInfo)> = prefixes
            .iter()
            .map(|prefix| -> Result<(&TensorInfo, &TensorInfo)> {
                let weight = artifact.tensors.get(&format!("{prefix}.weight")).ok_or_else(|| {
                    AegisError::InvalidPlan(format!("missing `{prefix}.weight`"))
                })?;
                let scales =
                    artifact.tensors.get(&format!("{prefix}.weight_scale")).ok_or_else(|| {
                        AegisError::InvalidPlan(format!("missing `{prefix}.weight_scale`"))
                    })?;
                Ok((weight, scales))
            })
            .collect::<Result<Vec<_>>>()?;

        // FAST PATH — whole-shard buffer. A layer's expert tensors live in one
        // shard but are SCATTERED across it (the safetensors layout is not
        // layer-grouped), so per-tensor reads make the NVMe seek (~400 MB/s).
        // Read the whole shard ONCE sequentially (cached across all layers in
        // that shard) and memcpy each tensor out of the buffer. ~2-4x faster
        // cold expert load. Falls back to per-tensor reads on multi-shard layers
        // or when host RAM is too tight to buffer the shard.
        if !pairs.is_empty() {
            let shard = pairs[0].0.shard_path.clone();
            let one_shard =
                pairs.iter().all(|(w, s)| w.shard_path == shard && s.shard_path == shard);
            if one_shard {
                if let Some(buf) = self.ensure_shard_buffered(&shard)? {
                    return pairs
                        .par_iter()
                        .map(|(w, s)| -> Result<(HostWeightBytes, HostWeightBytes)> {
                            let span = |t: &TensorInfo| -> (usize, usize) {
                                (t.file_offsets.0 as usize, t.data_len_bytes() as usize)
                            };
                            let (wstart, wlen) = span(w);
                            let (sstart, slen) = span(s);
                            // Contiguous packed||scales arena slot → one H2D at decode.
                            copy_packed_scales_contiguous_into_arena(
                                arena,
                                &buf[wstart..wstart + wlen],
                                &buf[sstart..sstart + slen],
                            )
                        })
                        .collect::<Result<Vec<_>>>();
                }
            }
        }

        // FALLBACK — per-tensor parallel reads (multi-shard layer or low RAM).
        // Still places each projection's packed||scales contiguously in the
        // arena (one combined reserve), preserving the decode single-H2D win.
        pairs
            .par_iter()
            .map(|(weight, scales)| -> Result<(HostWeightBytes, HostWeightBytes)> {
                read_packed_scales_contiguous_into_arena(arena, weight, scales)
            })
            .collect::<Result<Vec<_>>>()
    }

    /// Ensure `path`'s whole shard is buffered in `shard_cache`; return the
    /// buffer (or `None` to decline → caller falls back to per-tensor reads when
    /// free host RAM is too tight). Reads the WHOLE shard sequentially via
    /// read_chunked_par (a few large sequential sub-reads → ~NVMe ceiling). The
    /// previous shard's buffer is dropped first so we never hold two at once.
    fn ensure_shard_buffered(
        &self,
        path: &std::path::Path,
    ) -> Result<Option<std::sync::Arc<Vec<u8>>>> {
        if let Some((p, buf)) = self.shard_cache.borrow().as_ref() {
            if p == path {
                return Ok(Some(buf.clone()));
            }
        }
        // Switching shards: drop the old buffer FIRST so its ~9 GiB is freed
        // before the RAM check + the new allocation (never hold two at once, and
        // don't let the old one's footprint trip the guard for the new shard).
        *self.shard_cache.borrow_mut() = None;
        let len = std::fs::metadata(path)
            .map_err(|e| AegisError::InvalidPlan(format!("stat shard {}: {e}", path.display())))?
            .len() as usize;
        // RAM guard. The shard buffer (`len`) is held SIMULTANEOUSLY with the
        // experts being copied OUT of it into the persistent pinned arena (which
        // accumulates ~this shard's expert bytes), so the true peak is ≈ 2×len,
        // NOT len. The old `len + 4 GiB` guard under-counted by a whole shard and
        // let a 20.9 GiB shard buffer + a 17.5 GiB arena fill OOM a 32 GiB host.
        // Require room for both copies + an OS/IDE margin, else fall back to
        // per-tensor reads (one tensor in flight — peak is just the arena).
        const MARGIN: u64 = 4 * 1024 * 1024 * 1024;
        if let Some(avail) = mem_available_bytes() {
            if avail < 2 * len as u64 + MARGIN {
                return Ok(None);
            }
        }
        let mut buf = vec![0u8; len];
        let t = std::time::Instant::now();
        read_chunked_par(path, 0, &mut buf)?;
        if !std::io::IsTerminal::is_terminal(&std::io::stderr()) {
            let s = t.elapsed().as_secs_f64().max(1e-9);
            eprintln!(
                "load-timing:   shard buffered {:.2} GiB in {:.2}s ({:.0} MiB/s) [{}]",
                len as f64 / (1024.0 * 1024.0 * 1024.0),
                s,
                (len as f64 / (1024.0 * 1024.0)) / s,
                path.file_name().and_then(|n| n.to_str()).unwrap_or("?"),
            );
        }
        let arc = std::sync::Arc::new(buf);
        *self.shard_cache.borrow_mut() = Some((path.to_path_buf(), arc.clone()));
        Ok(Some(arc))
    }

    /// Build a `DeviceNvfp4Linear` for a host-resident layer using
    /// bytes already prefetched into the arena. This is the finalize
    /// half of the parallel-prefetch flow: file I/O ran on rayon
    /// workers via `prefetch_host_nvfp4_pairs_par`, and now the main
    /// thread does the cheap CUDA-side stub allocs and metadata
    /// construction (which must be serial — single CUDA stream).
    /// `loader` is consulted only for the optional small scalar
    /// metadata (`weight_scale_2`, `input_scale`).
    pub fn finalize_host_nvfp4_with_prefetched(
        &self,
        artifact: &ModelArtifact,
        prefix: &str,
        residency: TensorResidencyPlan,
        store: StoragePlacement,
        packed_bytes: HostWeightBytes,
        scales_bytes: HostWeightBytes,
        loader: &mut TensorStorageLoader,
    ) -> Result<DeviceNvfp4Linear> {
        let kernel_family = cuda_nvfp4_kernel_family_for_layout(
            prefix,
            aegisllm_base::tensor::layout::LinearResidentLayout::PackedSource,
        )?;
        let weight = artifact
            .tensors
            .get(&format!("{prefix}.weight"))
            .ok_or_else(|| AegisError::InvalidPlan(format!("missing `{prefix}.weight`")))?;
        let scales = artifact
            .tensors
            .get(&format!("{prefix}.weight_scale"))
            .ok_or_else(|| AegisError::InvalidPlan(format!("missing `{prefix}.weight_scale`")))?;
        let output_scale = read_nvfp4_pertensor_scale(
            artifact, prefix, "weight_scale_2", "weight_global_scale", loader, store,
        )?.unwrap_or(1.0);
        let input_scale = read_nvfp4_pertensor_scale(
            artifact, prefix, "input_scale", "input_global_scale", loader, store,
        )?.unwrap_or(1.0);
        let spec = Nvfp4LinearSpec::from_tensors(prefix, weight, scales, input_scale, output_scale)?;
        let stub_packed = self
            .runtime
            .stream
            .clone_htod(&[0u8])
            .map_err(map_cuda_err("htod nvfp4 host-resident stub packed"))?;
        let stub_scales = self
            .runtime
            .stream
            .clone_htod(&[0u8])
            .map_err(map_cuda_err("htod nvfp4 host-resident stub scales"))?;
        Ok(DeviceNvfp4Linear {
            name: spec.name,
            rows: spec.rows,
            cols: spec.cols,
            packed_bytes: spec.packed_bytes,
            scale_bytes: spec.scale_bytes,
            input_scale: spec.input_scale,
            output_scale: spec.output_scale,
            kernel_family,
            resident_layout: aegisllm_base::tensor::layout::LinearResidentLayout::PackedSource,
            residency,
            packed: stub_packed,
            scales: stub_scales,
            native_mxfp4: None,
            cutlass_nvfp4: None,
            host_weights: Some(Box::new(HostResidentWeights {
                packed: packed_bytes,
                scales: scales_bytes,
                native_mxfp4: None,
            })),
        })
    }

    pub fn load_dense_vector_with_store(
        &self,
        tensor: &TensorInfo,
        store: StoragePlacement,
        loader: &mut TensorStorageLoader,
    ) -> Result<DeviceBuffer<f32>> {
        // Vectors (norms, biases) are 1-D; GDN's depthwise conv1d weight is
        // `[channels, 1, kernel]` (3-D), loaded flattened row-major. Accept any
        // shape and flatten to `num_elements` f32 values below.
        if tensor.num_elements == 0 {
            return Err(AegisError::InvalidPlan(format!(
                "`{}` is empty (cannot load as a dense vector)",
                tensor.name
            )));
        }
        let loaded = loader.load_for_store(tensor, store)?;
        let bytes = loaded.as_bytes();
        let values = match tensor.dtype {
            TensorDType::BF16 => bytes
                .chunks_exact(2)
                .map(|chunk| {
                    f32::from_bits((u16::from_le_bytes([chunk[0], chunk[1]]) as u32) << 16)
                })
                .collect::<Vec<_>>(),
            TensorDType::F32 => bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect::<Vec<_>>(),
            other => {
                return Err(AegisError::InvalidPlan(format!(
                    "`{}` must be BF16 or F32 vector, got {:?}",
                    tensor.name, other
                )));
            }
        };
        self.runtime.upload_f32(&values)
    }

    /// Adds 1.0 to every element of a loaded norm-weight buffer (one-time, at
    /// load). Qwen3-Next's `Qwen3NextRMSNorm` computes `norm(x) * (1 + weight)`
    /// (zero-centered weights), while the engine's RMSNorm kernels apply a plain
    /// `weight` — folding the +1 here makes the plain kernel exact without
    /// touching the hot path or other architectures.
    pub fn plus_one_norm(&self, buf: DeviceBuffer<f32>) -> Result<DeviceBuffer<f32>> {
        let mut v = self.runtime.download_f32(&buf)?;
        for x in &mut v {
            *x += 1.0;
        }
        self.runtime.upload_f32(&v)
    }

    pub fn load_bf16_matrix_with_store(
        &self,
        tensor: &TensorInfo,
        _store: StoragePlacement,
        residency: TensorResidencyPlan,
        loader: &mut TensorStorageLoader,
    ) -> Result<DeviceBf16Matrix> {
        if tensor.dtype != TensorDType::BF16 || tensor.shape.len() < 2 {
            return Err(AegisError::InvalidPlan(format!(
                "`{}` must be a BF16 matrix (>= 2-D; got dtype={:?} shape={:?})",
                tensor.name, tensor.dtype, tensor.shape,
            )));
        }
        // For N-D tensors (e.g. vision position_embedding_table [2, 10240, 1152]),
        // collapse all leading dims into `rows`; the last dim stays `cols`.
        let last_dim = tensor.shape[tensor.shape.len() - 1];
        let row_product: usize = tensor.shape[..tensor.shape.len() - 1].iter().product();
        let effective_rows = row_product;
        let effective_cols = last_dim;
        let is_host_resident = matches!(residency, TensorResidencyPlan::StagedHostToDevice { .. });

        if is_host_resident {
            // Read tensor bytes from disk straight into the pinned host
            // arena via 8-way parallel `pread`. Big BF16 tensors
            // (embed ≈ 1.4 GiB) saturate Gen4 NVMe at ~4.4 GB/s
            // instead of the ~440 MB/s single-thread ceiling. After
            // `pin_now()` at end-of-load the arena is
            // `cuMemHostRegister`'d, so DMA from any sub-slice takes
            // the direct-pinned path.
            let arena = self.arena.as_ref().ok_or_else(|| {
                AegisError::InvalidPlan(format!(
                    "host-resident BF16 `{}` requires loader built with weight_loader_with_arena(...)",
                    tensor.name
                ))
            })?;
            let len = tensor.data_len_bytes() as usize;
            let offset = arena.reserve(len)?;
            // SAFETY: `reserve` exclusively claimed [offset, offset+len);
            // `read_chunked_par` writes disjoint sub-slices.
            let dst = unsafe { arena.slice_mut(offset, len) };
            read_chunked_par(&tensor.shard_path, tensor.file_offsets.0, dst)?;
            let stub = self
                .runtime
                .stream
                .clone_htod(&[0u16])
                .map_err(map_cuda_err("htod bf16 host-resident stub"))?;
            let host_weights = HostBf16Weights::from_arena(arena.clone(), offset, len)?;
            return Ok(DeviceBf16Matrix {
                name: tensor.name.clone(),
                rows: effective_rows,
                cols: effective_cols,
                residency,
                values: stub,
                host_values: Some(Box::new(host_weights)),
            });
        }

        // VRAM-resident BF16: read directly from disk into the
        // loader's reusable pinned bounce, then DMA bounce → VRAM.
        // Bypasses the mmap page-cache fill that would otherwise
        // hold ~tensor-size of shard pages alongside the growing
        // arena and trigger OOM on 32 GiB hosts. `loader` arg is
        // unused on this branch but stays in the signature for the
        // host-resident branch above.
        let _ = loader;
        let len_bytes = tensor.data_len_bytes() as usize;
        if len_bytes % 2 != 0 {
            return Err(AegisError::InvalidPlan(format!(
                "BF16 tensor `{}` has odd byte length {}", tensor.name, len_bytes
            )));
        }
        let len_u16 = len_bytes / 2;
        self.read_tensor_into_bounce(tensor)?;
        let mut buffer = self.runtime.alloc_u16(len_u16)?;
        let bounce_ref = self.bounce.borrow();
        let bounce = bounce_ref.as_ref().expect("bounce populated above");
        // SAFETY: bounce is a process-owned mmap (page-aligned, so
        // 2-byte alignment is satisfied); even-byte-length checked.
        // Slice to exactly `len_u16` because the bounce may be
        // larger when reused across smaller tensors.
        let src_u16: &[u16] = &bounce.as_u16_slice()[..len_u16];
        self.runtime
            .stream
            .memcpy_htod(src_u16, &mut buffer.slice)
            .map_err(map_cuda_err("htod bf16 matrix from bounce"))?;
        // `memcpy_htod` from a pinned source is async and returns
        // before the DMA finishes; if the next loader call clobbers
        // the bounce mid-transfer it'd corrupt the upload silently.
        // Synchronise here so the bounce is safe to reuse.
        self.runtime
            .synchronize()
            .map_err(|e| AegisError::Unsupported(format!("sync after bf16 bounce htod: {e}")))?;
        Ok(DeviceBf16Matrix {
            name: tensor.name.clone(),
            rows: effective_rows,
            cols: effective_cols,
            residency,
            values: buffer.slice,
            host_values: None,
        })
    }

    /// Load a BF16 tensor into a VRAM-resident `DeviceBf16Matrix` with the
    /// caller's explicit `(rows, cols)` shape, regardless of the checkpoint's
    /// stored N-D shape. Used for tensors whose logical 2-D linear shape isn't
    /// "all-leading-dims × last-dim" — e.g. Qwen's Conv3d patch-embed
    /// `[1152, 3, 2, 16, 16]` which is the linear `[1152, 1536]` (rows = first
    /// dim, cols = product of the rest). `rows * cols` must equal the tensor's
    /// element count.
    pub fn load_bf16_matrix_explicit_dims(
        &self,
        tensor: &TensorInfo,
        rows: usize,
        cols: usize,
        loader: &mut TensorStorageLoader,
    ) -> Result<DeviceBf16Matrix> {
        if tensor.dtype != TensorDType::BF16 {
            return Err(AegisError::InvalidPlan(format!(
                "`{}` must be BF16 for explicit-dims load (got {:?})",
                tensor.name, tensor.dtype
            )));
        }
        if rows * cols != tensor.num_elements {
            return Err(AegisError::InvalidPlan(format!(
                "`{}`: explicit rows*cols={} != num_elements={}",
                tensor.name, rows * cols, tensor.num_elements
            )));
        }
        let loaded = loader.load_for_store(tensor, StoragePlacement::Ram)?;
        let bytes = loaded.as_bytes();
        let values: Vec<u16> = bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        self.bf16_matrix_from_host_u16(&tensor.name, rows, cols, &values)
    }

    /// Build a VRAM-resident `DeviceBf16Matrix` directly from a host `&[u16]`
    /// (BF16 bit-pattern) slice of length `rows * cols`. Used to materialize
    /// per-expert weight matrices sliced out of a stacked/fused expert tensor
    /// (e.g. the Qwen3.6 MTP head's `experts.gate_up_proj [E, 2*I, H]` and
    /// `experts.down_proj [E, H, I]`), which the per-tensor checkpoint loader
    /// can't address as individual experts.
    pub fn bf16_matrix_from_host_u16(
        &self,
        name: &str,
        rows: usize,
        cols: usize,
        values: &[u16],
    ) -> Result<DeviceBf16Matrix> {
        if values.len() != rows * cols {
            return Err(AegisError::InvalidPlan(format!(
                "bf16_matrix_from_host_u16(`{name}`): values.len()={} != rows*cols={}",
                values.len(),
                rows * cols
            )));
        }
        let device = self.device_index();
        let buffer = self.runtime.upload_u16(values)?;
        Ok(DeviceBf16Matrix {
            name: name.to_string(),
            rows,
            cols,
            residency: TensorResidencyPlan::VramResident { device },
            values: buffer.slice,
            host_values: None,
        })
    }

    /// Load a DeepSeek-style FP8 **block-scaled** linear and dequantize it to
    /// BF16 on the host so it runs through the already-validated BF16 matvec
    /// path. The checkpoint stores `weight` as F8_E4M3 `[rows, cols]` plus a
    /// `weight_scale_inv` BF16 block table `[ceil(rows/128), ceil(cols/128)]`;
    /// dequant is `w_bf16[i,j] = e4m3(w[i,j]) * scale_inv[i/blk, j/blk]`.
    /// Placement (VRAM vs pinned host arena) mirrors `load_bf16_matrix_with_store`.
    pub fn load_fp8_block_as_bf16_matrix(
        &self,
        weight: &TensorInfo,
        scale: &TensorInfo,
        _store: StoragePlacement,
        residency: TensorResidencyPlan,
        _loader: &mut TensorStorageLoader,
    ) -> Result<DeviceBf16Matrix> {
        if weight.dtype != TensorDType::F8E4M3 || weight.shape.len() != 2 {
            return Err(AegisError::InvalidPlan(format!(
                "`{}` must be a 2-D F8_E4M3 matrix (got dtype={:?} shape={:?})",
                weight.name, weight.dtype, weight.shape
            )));
        }
        if scale.shape.len() != 2 {
            return Err(AegisError::InvalidPlan(format!(
                "`{}` FP8 block-scale must be 2-D (got {:?})",
                scale.name, scale.shape
            )));
        }
        let rows = weight.shape[0];
        let cols = weight.shape[1];
        let s_rows = scale.shape[0];
        let s_cols = scale.shape[1];
        let blk_r = rows.div_ceil(s_rows.max(1));
        let blk_c = cols.div_ceil(s_cols.max(1));

        // Read raw bytes: FP8 weight (1 B/elem) + BF16 block scales (2 B/elem).
        let mut fp8 = vec![0u8; rows * cols];
        read_chunked_par(&weight.shard_path, weight.file_offsets.0, &mut fp8)?;
        let mut s_bytes = vec![0u8; s_rows * s_cols * 2];
        read_chunked_par(&scale.shard_path, scale.file_offsets.0, &mut s_bytes)?;
        let scales: Vec<f32> = (0..s_rows * s_cols)
            .map(|k| {
                let bits = u16::from_le_bytes([s_bytes[2 * k], s_bytes[2 * k + 1]]);
                f32::from_bits((bits as u32) << 16)
            })
            .collect();
        let lut = build_e4m3_signed_lut();

        let is_host_resident = matches!(residency, TensorResidencyPlan::StagedHostToDevice { .. });
        if is_host_resident {
            let arena = self.arena.as_ref().ok_or_else(|| {
                AegisError::InvalidPlan(format!(
                    "host-resident FP8 `{}` requires loader built with weight_loader_with_arena(...)",
                    weight.name
                ))
            })?;
            let len = rows * cols * 2;
            let offset = arena.reserve(len)?;
            // SAFETY: `reserve` exclusively claimed [offset, offset+len);
            // `dequant_fp8_block_into_bf16` writes disjoint per-row sub-slices.
            let dst = unsafe { arena.slice_mut(offset, len) };
            dequant_fp8_block_into_bf16(dst, &fp8, &scales, rows, cols, s_cols, blk_r, blk_c, &lut);
            let stub = self
                .runtime
                .stream
                .clone_htod(&[0u16])
                .map_err(map_cuda_err("htod fp8-block bf16 host-resident stub"))?;
            let host_weights = HostBf16Weights::from_arena(arena.clone(), offset, len)?;
            return Ok(DeviceBf16Matrix {
                name: weight.name.clone(),
                rows,
                cols,
                residency,
                values: stub,
                host_values: Some(Box::new(host_weights)),
            });
        }

        // VRAM-resident: dequant to a host Vec<u16>, then DMA → VRAM.
        let mut out = vec![0u16; rows * cols];
        {
            use rayon::prelude::*;
            out.par_chunks_mut(cols).enumerate().for_each(|(i, row)| {
                let sr = (i / blk_r) * s_cols;
                let base = i * cols;
                for (j, slot) in row.iter_mut().enumerate() {
                    let w = lut[fp8[base + j] as usize];
                    let sc = scales[sr + j / blk_c];
                    *slot = f32_to_bf16_round(w * sc);
                }
            });
        }
        let mut buffer = self.runtime.alloc_u16(rows * cols)?;
        self.runtime
            .stream
            .memcpy_htod(&out, &mut buffer.slice)
            .map_err(map_cuda_err("htod fp8-block bf16 matrix"))?;
        self.runtime
            .synchronize()
            .map_err(|e| AegisError::Unsupported(format!("sync after fp8-block htod: {e}")))?;
        Ok(DeviceBf16Matrix {
            name: weight.name.clone(),
            rows,
            cols,
            residency,
            values: buffer.slice,
            host_values: None,
        })
    }

    /// Fuse a pair of already-loaded VRAM-resident BF16 gate/up matrices into
    /// one row-stacked `[2*rows, cols]` matrix and drop the originals' VRAM.
    ///
    /// Both inputs must be VRAM-resident and have identical `(rows, cols)` —
    /// the standard Gemma 4 / Llama gate/up convention. Returns:
    ///   * the fused matrix (rows = `2 * orig_rows`, cols unchanged), VRAM-
    ///     resident, ready for a single cuBLASLt BF16 GEMM that produces
    ///     `[batch, 2*orig_rows]` row-major. Per token the first `orig_rows`
    ///     floats are the gate logits and the next `orig_rows` are the up
    ///     logits — exactly the layout expected by `geglu_tanh_strided_device`.
    ///   * a pair of `DeviceBf16Matrix` stubs that retain the original
    ///     `(name, rows, cols, residency)` metadata but hold only a 1-element
    ///     placeholder VRAM slice. Returned so downstream code that still calls
    ///     `.rows()` / `.cols()` / `.name()` keeps working without further
    ///     refactors. **Stubs must not be matmul'd against** — `cublaslt_bf16_enabled_for`
    ///     still returns `true` (their `host_values` is `None`) but their
    ///     1-element `values` slice would corrupt any GEMM. Callers route the
    ///     matmul through the fused matrix instead.
    ///
    /// The fusion path uses a 1-shot device alloc + two D2D `memcpy_dtod`
    /// calls. Original gate/up VRAM is freed on return (the two stubs hold tiny
    /// `clone_htod(&[0u16])` slices), saving `2 * rows * cols * 2` bytes per
    /// fused layer versus keeping both originals alive.
    pub fn fuse_bf16_gate_up(
        &self,
        mut gate: DeviceBf16Matrix,
        mut up: DeviceBf16Matrix,
    ) -> Result<(DeviceBf16Matrix, DeviceBf16Matrix, DeviceBf16Matrix)> {
        if gate.is_host_resident() || up.is_host_resident() {
            return Err(AegisError::InvalidPlan(format!(
                "fuse_bf16_gate_up requires VRAM-resident inputs; got gate.host_resident={} up.host_resident={}",
                gate.is_host_resident(),
                up.is_host_resident()
            )));
        }
        if gate.rows != up.rows || gate.cols != up.cols {
            return Err(AegisError::InvalidPlan(format!(
                "fuse_bf16_gate_up shape mismatch: gate=({}, {}) up=({}, {})",
                gate.rows, gate.cols, up.rows, up.cols
            )));
        }
        let rows = gate.rows;
        let cols = gate.cols;
        let per_mat = rows
            .checked_mul(cols)
            .ok_or_else(|| AegisError::InvalidPlan("fuse gate/up elem count overflow".into()))?;
        let fused_len = per_mat
            .checked_mul(2)
            .ok_or_else(|| AegisError::InvalidPlan("fuse gate/up doubled count overflow".into()))?;
        if gate.values.len() < per_mat || up.values.len() < per_mat {
            return Err(AegisError::InvalidPlan(format!(
                "fuse_bf16_gate_up: input values too small: gate={} up={} need {}",
                gate.values.len(), up.values.len(), per_mat
            )));
        }

        // Allocate the row-stacked destination in VRAM and D2D-copy each
        // input matrix into its sub-range. cuBLASLt sees a single
        // `[2*rows, cols]` row-major weight, equivalent to stacking gate
        // on top of up (so the GEMM output per token is
        // `[gate_logits, up_logits]` row-major — the layout consumed by
        // `aegis_geglu_tanh_strided`).
        let mut fused_slice = unsafe { self.runtime.stream.alloc::<u16>(fused_len) }
            .map_err(map_cuda_err("alloc fused gate/up bf16"))?;
        {
            let src_view = gate.values.slice(0..per_mat);
            let mut dst_view = fused_slice.slice_mut(0..per_mat);
            self.runtime
                .stream
                .memcpy_dtod(&src_view, &mut dst_view)
                .map_err(map_cuda_err("d2d fused gate copy"))?;
        }
        {
            let src_view = up.values.slice(0..per_mat);
            let mut dst_view = fused_slice.slice_mut(per_mat..fused_len);
            self.runtime
                .stream
                .memcpy_dtod(&src_view, &mut dst_view)
                .map_err(map_cuda_err("d2d fused up copy"))?;
        }
        // Synchronise so the source buffers can safely be dropped below.
        self.runtime
            .synchronize()
            .map_err(|e| AegisError::Unsupported(format!("sync after fuse gate/up d2d: {e}")))?;

        let fused_name = format!(
            "{}|fused|{}",
            gate.name.trim_end_matches(".weight"),
            up.name.trim_end_matches(".weight")
        );
        let fused = DeviceBf16Matrix {
            name: fused_name,
            rows: 2 * rows,
            cols,
            residency: gate.residency.clone(),
            values: fused_slice,
            host_values: None,
        };

        // Replace each original's `values` with a tiny 1-element stub. This
        // drops the full-size VRAM allocation (Rust's `Drop` on `CudaSlice`
        // returns the bytes to the pool) while keeping `(name, rows, cols,
        // residency)` intact for downstream introspection.
        let gate_stub_slice = self
            .runtime
            .stream
            .clone_htod(&[0u16])
            .map_err(map_cuda_err("htod gate stub"))?;
        let up_stub_slice = self
            .runtime
            .stream
            .clone_htod(&[0u16])
            .map_err(map_cuda_err("htod up stub"))?;
        gate.values = gate_stub_slice;
        up.values = up_stub_slice;

        Ok((fused, gate, up))
    }

    /// Load a BF16 matrix from `tensor` and quantize it to FP8 E4M3 with
    /// per-row FP32 scales at load time. Used by
    /// `shared-MLP-quantization = "fp8"` (and `attention-quantization = "fp8"`
    /// once that path lands).
    ///
    /// Path: stage BF16 directly into a temporary VRAM buffer, then run the
    /// `aegis_quantize_bf16_to_fp8_per_row` kernel which computes per-row
    /// absmax → scale = amax / 448 → encode each element as E4M3. The
    /// transient BF16 buffer is dropped on return; only FP8 + scales
    /// remain (~2× smaller than BF16). No host-side quantizer arithmetic;
    /// reuses the device-side `float_to_fp8_e4m3_bits` helper.
    pub fn load_bf16_as_fp8_linear(
        &self,
        tensor: &TensorInfo,
        loader: &mut TensorStorageLoader,
    ) -> Result<crate::cuda::StandaloneFp8Linear> {
        if tensor.dtype != TensorDType::BF16 || tensor.shape.len() != 2 {
            return Err(AegisError::InvalidPlan(format!(
                "`{}` must be a 2D BF16 matrix to be quantized to FP8",
                tensor.name
            )));
        }
        let rows = tensor.shape[0];
        let cols = tensor.shape[1];
        let total = rows.checked_mul(cols).ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "`{}` BF16→FP8 element count overflow: rows={} cols={}",
                tensor.name, rows, cols
            ))
        })?;

        // Stage BF16 into a transient VRAM buffer.
        let loaded = loader.load_for_store(tensor, StoragePlacement::Ram)?;
        let bf16_host: Vec<u16> = loaded
            .as_bytes()
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();
        if bf16_host.len() != total {
            return Err(AegisError::InvalidPlan(format!(
                "`{}` BF16→FP8: bf16 source size mismatch: got {} u16, expected {}",
                tensor.name, bf16_host.len(), total
            )));
        }
        let bf16_dev_slice = self
            .runtime
            .stream
            .clone_htod(&bf16_host)
            .map_err(map_cuda_err("htod bf16 transient for fp8 quantize"))?;
        let bf16_dev = DeviceBuffer { slice: bf16_dev_slice };

        // Allocate FP8 + per-row scales output buffers.
        let mut fp8_dev = self.runtime.alloc_u8(total)?;
        let mut row_scales_dev = self.runtime.alloc_f32(rows)?;

        // Run the quantize kernel. After this returns, `bf16_dev` is no
        // longer needed and is dropped at end of scope.
        self.runtime
            .quantize_bf16_to_fp8_per_row_device(&bf16_dev, rows, cols, &mut fp8_dev, &mut row_scales_dev)?;

        Ok(crate::cuda::StandaloneFp8Linear {
            name: tensor.name.clone(),
            rows,
            cols,
            bytes: total,
            data: fp8_dev.slice,
            row_scales: row_scales_dev.slice,
            block_scales: None,
            block_size: 0,
            scale_cols: 0,
        })
    }

    /// Load a pre-quantized DeepSeek-style FP8 **block-scaled** linear into VRAM:
    /// F8_E4M3 `weight [rows, cols]` + BF16 `weight_scale_inv [rows/blk, cols/blk]`.
    /// The 9 GB of FP8 weights fit the 16 GB GPU directly (vs 18 GB dequanted to
    /// BF16), and the fused `aegis_fp8_block_matvec` dequant-on-the-fly avoids ever
    /// materializing BF16. Always VRAM-resident (caller's store/residency ignored).
    pub fn load_fp8_block_linear(
        &self,
        weight: &TensorInfo,
        scale: &TensorInfo,
        loader: &mut TensorStorageLoader,
    ) -> Result<crate::cuda::StandaloneFp8Linear> {
        if weight.dtype != TensorDType::F8E4M3 || weight.shape.len() != 2 {
            return Err(AegisError::InvalidPlan(format!(
                "`{}` must be a 2-D F8_E4M3 matrix (got dtype={:?} shape={:?})",
                weight.name, weight.dtype, weight.shape
            )));
        }
        if scale.shape.len() != 2 {
            return Err(AegisError::InvalidPlan(format!(
                "`{}` FP8 block-scale must be 2-D (got {:?})",
                scale.name, scale.shape
            )));
        }
        let rows = weight.shape[0];
        let cols = weight.shape[1];
        let total = rows.checked_mul(cols).ok_or_else(|| {
            AegisError::InvalidPlan(format!("`{}` FP8 element-count overflow", weight.name))
        })?;
        let s_rows = scale.shape[0];
        let s_cols = scale.shape[1];
        let blk_r = rows.div_ceil(s_rows.max(1));
        let blk_c = cols.div_ceil(s_cols.max(1));
        if blk_r != blk_c {
            return Err(AegisError::InvalidPlan(format!(
                "`{}` FP8 block scale expects square blocks, got {blk_r}x{blk_c}",
                weight.name
            )));
        }

        // FP8 weight bytes → VRAM (1 byte/elem).
        let w_loaded = loader.load_for_store(weight, StoragePlacement::Ram)?;
        let fp8_bytes = w_loaded.as_bytes();
        if fp8_bytes.len() < total {
            return Err(AegisError::InvalidPlan(format!(
                "`{}` FP8 weight short read: {} < {total}",
                weight.name,
                fp8_bytes.len()
            )));
        }
        let data = self
            .runtime
            .stream
            .clone_htod(&fp8_bytes[..total])
            .map_err(map_cuda_err("htod fp8 block weight"))?;

        // BF16 block scales → f32 → VRAM.
        let s_loaded = loader.load_for_store(scale, StoragePlacement::Ram)?;
        let scales_f32: Vec<f32> = s_loaded
            .as_bytes()
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect();
        let block_scales = self
            .runtime
            .stream
            .clone_htod(&scales_f32)
            .map_err(map_cuda_err("htod fp8 block scales"))?;
        // Unused in the block path, but the struct requires a non-empty slice.
        let row_scales = self
            .runtime
            .stream
            .clone_htod(&[1.0f32])
            .map_err(map_cuda_err("htod fp8 block row-scale stub"))?;

        Ok(crate::cuda::StandaloneFp8Linear {
            name: weight.name.clone(),
            rows,
            cols,
            bytes: total,
            data,
            row_scales,
            block_scales: Some(block_scales),
            block_size: blk_r as u32,
            scale_cols: s_cols as u32,
        })
    }

    pub fn load_nvfp4_linear(
        &self,
        artifact: &ModelArtifact,
        prefix: &str,
    ) -> Result<DeviceNvfp4Linear> {
        let mut loader = TensorStorageLoader::new();
        self.load_nvfp4_linear_with_store(
            artifact,
            prefix,
            StoragePlacement::Vram {
                device: self.runtime.device_index(),
            },
            TensorResidencyPlan::VramResident {
                device: self.runtime.device_index(),
            },
            &mut loader,
        )
    }

    pub fn load_nvfp4_linear_with_store(
        &self,
        artifact: &ModelArtifact,
        prefix: &str,
        store: StoragePlacement,
        residency: TensorResidencyPlan,
        loader: &mut TensorStorageLoader,
    ) -> Result<DeviceNvfp4Linear> {
        self.load_nvfp4_linear_with_layout(
            artifact,
            prefix,
            store,
            residency,
            LinearResidentLayout::NativeTensorCore,
            loader,
        )
    }

    pub fn load_nvfp4_linear_with_layout(
        &self,
        artifact: &ModelArtifact,
        prefix: &str,
        store: StoragePlacement,
        residency: TensorResidencyPlan,
        resident_layout: LinearResidentLayout,
        loader: &mut TensorStorageLoader,
    ) -> Result<DeviceNvfp4Linear> {
        let kernel_family = cuda_nvfp4_kernel_family_for_layout(prefix, resident_layout)?;
        let weight = artifact
            .tensors
            .get(&format!("{prefix}.weight"))
            .ok_or_else(|| AegisError::InvalidPlan(format!("missing `{prefix}.weight`")))?;
        let scales = artifact
            .tensors
            .get(&format!("{prefix}.weight_scale"))
            .ok_or_else(|| AegisError::InvalidPlan(format!("missing `{prefix}.weight_scale`")))?;
        let output_scale = read_nvfp4_pertensor_scale(
            artifact, prefix, "weight_scale_2", "weight_global_scale", loader, store,
        )?.unwrap_or(1.0);
        let input_scale = read_nvfp4_pertensor_scale(
            artifact, prefix, "input_scale", "input_global_scale", loader, store,
        )?.unwrap_or(1.0);
        let spec =
            Nvfp4LinearSpec::from_tensors(prefix, weight, scales, input_scale, output_scale)?;
        let is_host_resident = matches!(residency, TensorResidencyPlan::StagedHostToDevice { .. });

        // We need the mmap'd bytes up-front whenever the call site has to
        // read them on this thread: VRAM upload (clone_htod below), or
        // load-time repacking into native MXFP4 / cutlass NVFP4. For the
        // plain host-resident NVFP4 path the staging pool reads from the
        // shard mmap on demand at inference time, so we don't fault the
        // pages here.
        let needs_mmap = !is_host_resident
            || self.should_repack_native_mxfp4(prefix, kernel_family)
            || self.should_repack_cutlass_nvfp4(prefix, kernel_family, resident_layout);
        let (packed_host, scales_host) = if needs_mmap {
            (
                Some(loader.load_for_store(weight, store)?),
                Some(loader.load_for_store(scales, store)?),
            )
        } else {
            (None, None)
        };

        let native_mxfp4 = if !is_host_resident && self.should_repack_native_mxfp4(prefix, kernel_family) {
            if spec.cols % 64 != 0 {
                return Err(AegisError::InvalidPlan(format!(
                    "native MXFP4 tensor-core layout for `{}` requires cols divisible by 64, got {}",
                    spec.name, spec.cols
                )));
            }
            let repacked = cached_repack_nvfp4_to_mxfp4_host(
                &artifact.root,
                &spec,
                weight,
                scales,
                packed_host.as_ref().unwrap().as_bytes(),
                scales_host.as_ref().unwrap().as_bytes(),
            )?;
            Some(DeviceMxfp4Linear {
                bytes: repacked.len(),
                blocks_per_row: spec.cols / 32,
                data: self
                    .runtime
                    .stream
                    .clone_htod(&repacked)
                    .map_err(map_cuda_err("htod native mxfp4 weights"))?,
            })
        } else {
            None
        };
        let cutlass_nvfp4 =
            if !is_host_resident && self.should_repack_cutlass_nvfp4(prefix, kernel_family, resident_layout) {
                let repacked = cached_repack_nvfp4_to_cutlass_e2m1_ue4m3_host(
                    &artifact.root,
                    &spec,
                    weight,
                    scales,
                    packed_host.as_ref().unwrap().as_bytes(),
                    scales_host.as_ref().unwrap().as_bytes(),
                )?;
                Some(DeviceCutlassNvfp4Linear {
                    layout: repacked.layout,
                    payload_e2m1: self
                        .runtime
                        .stream
                        .clone_htod(&repacked.payload_e2m1)
                        .map_err(map_cuda_err("htod cutlass nvfp4 payload"))?,
                    scales_ue4m3: self
                        .runtime
                        .stream
                        .clone_htod(&repacked.scales_ue4m3)
                        .map_err(map_cuda_err("htod cutlass nvfp4 scales"))?,
                })
            } else {
                None
            };

        if is_host_resident {
            // Weights stay in CUDA-pinned host RAM; tiny VRAM stubs keep the type system intact.
            // Weights are read directly from the safetensors file into pinned memory so no
            // intermediate mmap/pageable copy is created in the kernel page cache.
            let host_native_mxfp4 =
                if self.should_repack_native_mxfp4(prefix, kernel_family) {
                    if spec.cols % 64 != 0 {
                        return Err(AegisError::InvalidPlan(format!(
                            "native MXFP4 tensor-core layout for `{}` requires cols divisible by 64, got {}",
                            spec.name, spec.cols
                        )));
                    }
                    let repacked = cached_repack_nvfp4_to_mxfp4_host(
                        &artifact.root,
                        &spec,
                        weight,
                        scales,
                        packed_host.as_ref().unwrap().as_bytes(),
                        scales_host.as_ref().unwrap().as_bytes(),
                    )?;
                    let pinned = alloc_pinned_from_bytes(
                        self.runtime,
                        &repacked,
                        "alloc pinned host native mxfp4",
                    )?;
                    Some(HostResidentMxfp4 {
                        blocks_per_row: spec.cols / 32,
                        data: HostWeightBytes::Pinned(pinned),
                    })
                } else {
                    None
                };
            // Host-resident weights live in the pinned-host arena: one
            // big `cuMemHostRegister`'d allocation, sub-allocated per
            // tensor by atomic bump. Per-token inference does direct
            // DMA from the arena — no CPU memcpy through a staging
            // bounce. RAM cost: ~12-14 GiB anonymous-mapped pinned
            // pages for Gemma-4-26B.
            let arena = self.arena.as_ref().ok_or_else(|| {
                AegisError::InvalidPlan(format!(
                    "host-resident NVFP4 `{}` requires loader built with weight_loader_with_arena(...)",
                    spec.name
                ))
            })?;
            // Contiguous packed||scales arena slot (adjacency lets decode issue
            // one combined H2D per projection; see the contiguous helper).
            let (mmap_packed, mmap_scales) =
                read_packed_scales_contiguous_into_arena(arena, weight, scales)?;
            let _ = store;
            let stub_packed = self
                .runtime
                .stream
                .clone_htod(&[0u8])
                .map_err(map_cuda_err("htod nvfp4 host-resident stub packed"))?;
            let stub_scales = self
                .runtime
                .stream
                .clone_htod(&[0u8])
                .map_err(map_cuda_err("htod nvfp4 host-resident stub scales"))?;
            return Ok(DeviceNvfp4Linear {
                name: spec.name,
                rows: spec.rows,
                cols: spec.cols,
                packed_bytes: spec.packed_bytes,
                scale_bytes: spec.scale_bytes,
                input_scale: spec.input_scale,
                output_scale: spec.output_scale,
                kernel_family,
                resident_layout: aegisllm_base::tensor::layout::LinearResidentLayout::PackedSource,
                residency,
                packed: stub_packed,
                scales: stub_scales,
                native_mxfp4: None,
                cutlass_nvfp4: None,
                host_weights: Some(Box::new(HostResidentWeights {
                    packed: mmap_packed,
                    scales: mmap_scales,
                    native_mxfp4: host_native_mxfp4,
                })),
            });
        }

        Ok(DeviceNvfp4Linear {
            name: spec.name,
            rows: spec.rows,
            cols: spec.cols,
            packed_bytes: spec.packed_bytes,
            scale_bytes: spec.scale_bytes,
            input_scale: spec.input_scale,
            output_scale: spec.output_scale,
            kernel_family,
            resident_layout,
            residency,
            packed: self
                .runtime
                .stream
                .clone_htod(packed_host.as_ref().unwrap().as_bytes())
                .map_err(map_cuda_err("htod nvfp4 packed weights"))?,
            scales: self
                .runtime
                .stream
                .clone_htod(scales_host.as_ref().unwrap().as_bytes())
                .map_err(map_cuda_err("htod nvfp4 scales"))?,
            native_mxfp4,
            cutlass_nvfp4,
            host_weights: None,
        })
    }

    /// Load an NVFP4 linear by slicing the leading `eff_rows × eff_logical_cols` block
    /// from a MatFormer nested-param checkpoint.
    ///
    /// Unlike the full-tensor loader, this always uses `LinearResidentLayout::PackedSource`
    /// (no repacking), so the kernel falls back to the unpacked NVFP4 path.
    #[allow(clippy::too_many_arguments)]
    pub fn load_nvfp4_linear_sliced_with_layout(
        &self,
        artifact: &ModelArtifact,
        prefix: &str,
        store: StoragePlacement,
        residency: TensorResidencyPlan,
        _resident_layout: LinearResidentLayout,
        eff_rows: usize,
        eff_logical_cols: usize,
        loader: &mut TensorStorageLoader,
    ) -> Result<DeviceNvfp4Linear> {
        let kernel_family =
            cuda_nvfp4_kernel_family_for_layout(prefix, LinearResidentLayout::PackedSource)?;
        let weight = artifact
            .tensors
            .get(&format!("{prefix}.weight"))
            .ok_or_else(|| AegisError::InvalidPlan(format!("missing `{prefix}.weight`")))?;
        let scales_info = artifact
            .tensors
            .get(&format!("{prefix}.weight_scale"))
            .ok_or_else(|| AegisError::InvalidPlan(format!("missing `{prefix}.weight_scale`")))?;
        let output_scale = read_nvfp4_pertensor_scale(
            artifact, prefix, "weight_scale_2", "weight_global_scale", loader, store,
        )?.unwrap_or(1.0);
        let input_scale = read_nvfp4_pertensor_scale(
            artifact, prefix, "input_scale", "input_global_scale", loader, store,
        )?.unwrap_or(1.0);

        if eff_logical_cols == 0 || eff_logical_cols % QK_NVFP4 != 0 {
            return Err(AegisError::InvalidPlan(format!(
                "NVFP4 sliced `{prefix}` eff_logical_cols={eff_logical_cols} must be non-zero and divisible by {QK_NVFP4}"
            )));
        }
        let eff_packed_cols = eff_logical_cols / 2;
        let eff_scale_cols = eff_logical_cols / QK_NVFP4 * (QK_NVFP4 / QK_NVFP4_SUB);

        let packed_host = loader.load_submatrix(
            weight,
            NestedParamSlice::submatrix(eff_rows, eff_packed_cols),
        )?;
        let scales_host = loader.load_submatrix(
            scales_info,
            NestedParamSlice::submatrix(eff_rows, eff_scale_cols),
        )?;

        let is_host_resident = matches!(residency, TensorResidencyPlan::StagedHostToDevice { .. });
        if is_host_resident {
            let pinned_packed = alloc_pinned_from_bytes(
                self.runtime,
                packed_host.as_bytes(),
                "alloc pinned sliced host packed",
            )?;
            let pinned_scales = alloc_pinned_from_bytes(
                self.runtime,
                scales_host.as_bytes(),
                "alloc pinned sliced host scales",
            )?;
            let stub = self
                .runtime
                .stream
                .clone_htod(&[0u8])
                .map_err(map_cuda_err("htod sliced host-resident stub"))?;
            let stub2 = self
                .runtime
                .stream
                .clone_htod(&[0u8])
                .map_err(map_cuda_err("htod sliced host-resident stub2"))?;
            return Ok(DeviceNvfp4Linear {
                name: prefix.to_string(),
                rows: eff_rows,
                cols: eff_logical_cols,
                packed_bytes: packed_host.len(),
                scale_bytes: scales_host.len(),
                input_scale,
                output_scale,
                kernel_family,
                resident_layout: LinearResidentLayout::PackedSource,
                residency,
                packed: stub,
                scales: stub2,
                native_mxfp4: None,
                cutlass_nvfp4: None,
                host_weights: Some(Box::new(HostResidentWeights {
                    packed: HostWeightBytes::Pinned(pinned_packed),
                    scales: HostWeightBytes::Pinned(pinned_scales),
                    native_mxfp4: None,
                })),
            });
        }

        Ok(DeviceNvfp4Linear {
            name: prefix.to_string(),
            rows: eff_rows,
            cols: eff_logical_cols,
            packed_bytes: packed_host.len(),
            scale_bytes: scales_host.len(),
            input_scale,
            output_scale,
            kernel_family,
            resident_layout: LinearResidentLayout::PackedSource,
            residency,
            packed: self
                .runtime
                .stream
                .clone_htod(packed_host.as_bytes())
                .map_err(map_cuda_err("htod nvfp4 sliced packed weights"))?,
            scales: self
                .runtime
                .stream
                .clone_htod(scales_host.as_bytes())
                .map_err(map_cuda_err("htod nvfp4 sliced scales"))?,
            native_mxfp4: None,
            cutlass_nvfp4: None,
            host_weights: None,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_cutlass_qkv_group_with_layout(
        &self,
        artifact: &ModelArtifact,
        q_prefix: &str,
        k_prefix: &str,
        v_prefix: &str,
        store: StoragePlacement,
        residency: TensorResidencyPlan,
        resident_layout: LinearResidentLayout,
        loader: &mut TensorStorageLoader,
    ) -> Result<Option<DeviceNvfp4Linear>> {
        if !self.runtime.config().cutlass_nvfp4_repack {
            return Ok(None);
        }
        let kernel_family = cuda_nvfp4_kernel_family_for_layout(q_prefix, resident_layout)?;
        if !matches!(
            kernel_family,
            KernelFamily::CudaCutlassFp4TensorCores | KernelFamily::CudaNativeFp4TensorCores
        ) {
            return Ok(None);
        }

        let q = load_nvfp4_linear_host_parts(artifact, q_prefix, store, loader)?;
        let k = load_nvfp4_linear_host_parts(artifact, k_prefix, store, loader)?;
        let v = load_nvfp4_linear_host_parts(artifact, v_prefix, store, loader)?;
        if q.spec.cols != k.spec.cols || q.spec.cols != v.spec.cols {
            return Err(AegisError::InvalidPlan(format!(
                "CUTLASS QKV group shape mismatch: q={}x{} k={}x{} v={}x{}",
                q.spec.rows, q.spec.cols, k.spec.rows, k.spec.cols, v.spec.rows, v.spec.cols
            )));
        }
        if (q.spec.input_scale - k.spec.input_scale).abs() > 1.0e-12
            || (q.spec.input_scale - v.spec.input_scale).abs() > 1.0e-12
        {
            return Err(AegisError::InvalidPlan(format!(
                "CUTLASS QKV group requires equal input scales: q={} k={} v={}",
                q.spec.input_scale, k.spec.input_scale, v.spec.input_scale
            )));
        }

        let rows = q
            .spec
            .rows
            .checked_add(k.spec.rows)
            .and_then(|rows| rows.checked_add(v.spec.rows))
            .ok_or_else(|| AegisError::InvalidPlan("CUTLASS QKV group rows overflow".into()))?;
        let packed_bytes = q
            .spec
            .packed_bytes
            .checked_add(k.spec.packed_bytes)
            .and_then(|bytes| bytes.checked_add(v.spec.packed_bytes))
            .ok_or_else(|| {
                AegisError::InvalidPlan("CUTLASS QKV group packed bytes overflow".into())
            })?;
        let scale_bytes = q
            .spec
            .scale_bytes
            .checked_add(k.spec.scale_bytes)
            .and_then(|bytes| bytes.checked_add(v.spec.scale_bytes))
            .ok_or_else(|| {
                AegisError::InvalidPlan("CUTLASS QKV group scale bytes overflow".into())
            })?;
        let group_spec = Nvfp4LinearSpec {
            name: format!("{q_prefix}+{k_prefix}+{v_prefix}"),
            rows,
            cols: q.spec.cols,
            packed_bytes,
            scale_bytes,
            input_scale: q.spec.input_scale,
            // The fused GEMM writes an unscaled accumulator. A tiny split kernel
            // applies per-projection output scales while scattering to q/k/v.
            output_scale: 1.0,
        };

        let mut packed = Vec::with_capacity(packed_bytes);
        packed.extend_from_slice(q.packed.as_bytes());
        packed.extend_from_slice(k.packed.as_bytes());
        packed.extend_from_slice(v.packed.as_bytes());
        let mut scales = Vec::with_capacity(scale_bytes);
        scales.extend_from_slice(q.scales.as_bytes());
        scales.extend_from_slice(k.scales.as_bytes());
        scales.extend_from_slice(v.scales.as_bytes());
        let repacked = repack_nvfp4_to_cutlass_e2m1_ue4m3_host(&group_spec, &packed, &scales)?;

        Ok(Some(DeviceNvfp4Linear {
            name: group_spec.name,
            rows: group_spec.rows,
            cols: group_spec.cols,
            packed_bytes: group_spec.packed_bytes,
            scale_bytes: group_spec.scale_bytes,
            input_scale: group_spec.input_scale,
            output_scale: group_spec.output_scale,
            kernel_family,
            resident_layout,
            residency,
            packed: self
                .runtime
                .stream
                .clone_htod(&packed)
                .map_err(map_cuda_err("htod qkv group nvfp4 packed weights"))?,
            scales: self
                .runtime
                .stream
                .clone_htod(&scales)
                .map_err(map_cuda_err("htod qkv group nvfp4 scales"))?,
            native_mxfp4: None,
            cutlass_nvfp4: Some(DeviceCutlassNvfp4Linear {
                layout: repacked.layout,
                payload_e2m1: self
                    .runtime
                    .stream
                    .clone_htod(&repacked.payload_e2m1)
                    .map_err(map_cuda_err("htod qkv group cutlass nvfp4 payload"))?,
                scales_ue4m3: self
                    .runtime
                    .stream
                    .clone_htod(&repacked.scales_ue4m3)
                    .map_err(map_cuda_err("htod qkv group cutlass nvfp4 scales"))?,
            }),
            host_weights: None,
        }))
    }

    pub fn load_region_nvfp4_linears(
        &self,
        artifact: &ModelArtifact,
        region: &GraphRegion,
    ) -> Result<Vec<DeviceNvfp4Linear>> {
        let mut linears = Vec::new();
        let mut loader = TensorStorageLoader::new();
        for tensor in &region.tensors {
            if !is_nvfp4_linear_weight(tensor) {
                continue;
            }
            let prefix = tensor.info.name.strip_suffix(".weight").ok_or_else(|| {
                AegisError::InvalidPlan(format!("bad linear tensor name `{}`", tensor.info.name))
            })?;
            linears.push(self.load_nvfp4_linear_with_store(
                artifact,
                prefix,
                StoragePlacement::Vram {
                    device: self.runtime.device_index(),
                },
                TensorResidencyPlan::VramResident {
                    device: self.runtime.device_index(),
                },
                &mut loader,
            )?);
        }
        Ok(linears)
    }

    pub fn load_region_nvfp4_linears_with_store(
        &self,
        artifact: &ModelArtifact,
        region: &GraphRegion,
        store: StoragePlacement,
        residency: TensorResidencyPlan,
    ) -> Result<Vec<DeviceNvfp4Linear>> {
        self.load_region_nvfp4_linears_with_layout(
            artifact,
            region,
            store,
            residency,
            LinearResidentLayout::NativeTensorCore,
        )
    }

    pub fn load_region_nvfp4_linears_with_layout(
        &self,
        artifact: &ModelArtifact,
        region: &GraphRegion,
        store: StoragePlacement,
        residency: TensorResidencyPlan,
        resident_layout: LinearResidentLayout,
    ) -> Result<Vec<DeviceNvfp4Linear>> {
        let mut linears = Vec::new();
        let mut loader = TensorStorageLoader::new();
        for tensor in &region.tensors {
            if !is_nvfp4_linear_weight(tensor) {
                continue;
            }
            let prefix = tensor.info.name.strip_suffix(".weight").ok_or_else(|| {
                AegisError::InvalidPlan(format!("bad linear tensor name `{}`", tensor.info.name))
            })?;
            linears.push(self.load_nvfp4_linear_with_layout(
                artifact,
                prefix,
                store,
                residency,
                resident_layout,
                &mut loader,
            )?);
        }
        Ok(linears)
    }

    pub fn load_placed_region_nvfp4_linears(
        &self,
        artifact: &ModelArtifact,
        region: &GraphRegion,
        placement: &RegionPlacement,
    ) -> Result<Vec<DeviceNvfp4Linear>> {
        self.load_placed_region_nvfp4_linears_with_layout(
            artifact,
            region,
            placement,
            LinearResidentLayout::NativeTensorCore,
        )
    }

    pub fn load_placed_region_nvfp4_linears_with_layout(
        &self,
        artifact: &ModelArtifact,
        region: &GraphRegion,
        placement: &RegionPlacement,
        resident_layout: LinearResidentLayout,
    ) -> Result<Vec<DeviceNvfp4Linear>> {
        if placement.region_id != region.id {
            return Err(AegisError::InvalidPlan(format!(
                "placement `{}` does not match graph region `{}`",
                placement.region_id.0, region.id.0
            )));
        }
        match (placement.store, placement.compute) {
            (
                StoragePlacement::Vram {
                    device: store_device,
                },
                ComputePlacement::Cuda {
                    device: compute_device,
                },
            ) if store_device == self.runtime.device_index()
                && compute_device == self.runtime.device_index() =>
            {
                self.load_region_nvfp4_linears_with_layout(
                    artifact,
                    region,
                    placement.store,
                    TensorResidencyPlan::VramResident {
                        device: self.runtime.device_index(),
                    },
                    resident_layout,
                )
            }
            (StoragePlacement::Ram | StoragePlacement::Mmap, ComputePlacement::Cuda { device })
                if device == self.runtime.device_index() =>
            {
                self.load_region_nvfp4_linears_with_layout(
                    artifact,
                    region,
                    placement.store,
                    TensorResidencyPlan::StagedHostToDevice {
                        device: self.runtime.device_index(),
                    },
                    resident_layout,
                )
            }
            (StoragePlacement::Ram | StoragePlacement::Mmap, ComputePlacement::Cuda { device }) => {
                Err(AegisError::Unsupported(format!(
                    "region `{}` is compute=cuda:{device}, but this CUDA runtime is cuda:{}",
                    region.id.0,
                    self.runtime.device_index()
                )))
            }
            (StoragePlacement::Vram { device }, ComputePlacement::Cuda { device: compute }) => {
                Err(AegisError::Unsupported(format!(
                    "region `{}` has cross-device placement store=vram:{device} compute=cuda:{compute}; cross-device loaders are not implemented yet",
                    region.id.0
                )))
            }
            (_, ComputePlacement::Cpu) => Err(AegisError::Unsupported(format!(
                "region `{}` is compute=cpu; CUDA resident loader refused to load it",
                region.id.0
            ))),
            (_, ComputePlacement::Wgpu { device }) => Err(AegisError::Unsupported(format!(
                "region `{}` is compute=wgpu:{device}; CUDA resident loader refused to load it",
                region.id.0
            ))),
        }
    }

    pub fn load_first_placed_region_nvfp4_linear(
        &self,
        artifact: &ModelArtifact,
        region: &GraphRegion,
        placement: &RegionPlacement,
    ) -> Result<Option<DeviceNvfp4Linear>> {
        self.load_first_placed_region_nvfp4_linear_with_layout(
            artifact,
            region,
            placement,
            LinearResidentLayout::NativeTensorCore,
        )
    }

    pub fn load_first_placed_region_nvfp4_linear_with_layout(
        &self,
        artifact: &ModelArtifact,
        region: &GraphRegion,
        placement: &RegionPlacement,
        resident_layout: LinearResidentLayout,
    ) -> Result<Option<DeviceNvfp4Linear>> {
        if placement.region_id != region.id {
            return Err(AegisError::InvalidPlan(format!(
                "placement `{}` does not match graph region `{}`",
                placement.region_id.0, region.id.0
            )));
        }
        let Some(prefix) = first_nvfp4_linear_prefix(region) else {
            return Ok(None);
        };
        let mut loader = TensorStorageLoader::new();
        match (placement.store, placement.compute) {
            (
                StoragePlacement::Vram {
                    device: store_device,
                },
                ComputePlacement::Cuda {
                    device: compute_device,
                },
            ) if store_device == self.runtime.device_index()
                && compute_device == self.runtime.device_index() =>
            {
                self.load_nvfp4_linear_with_layout(
                    artifact,
                    prefix,
                    placement.store,
                    TensorResidencyPlan::VramResident {
                        device: self.runtime.device_index(),
                    },
                    resident_layout,
                    &mut loader,
                )
                .map(Some)
            }
            (StoragePlacement::Ram | StoragePlacement::Mmap, ComputePlacement::Cuda { device })
                if device == self.runtime.device_index() =>
            {
                self.load_nvfp4_linear_with_layout(
                    artifact,
                    prefix,
                    placement.store,
                    TensorResidencyPlan::StagedHostToDevice {
                        device: self.runtime.device_index(),
                    },
                    resident_layout,
                    &mut loader,
                )
                .map(Some)
            }
            (StoragePlacement::Ram | StoragePlacement::Mmap, ComputePlacement::Cuda { device }) => {
                Err(AegisError::Unsupported(format!(
                    "region `{}` is compute=cuda:{device}, but this CUDA runtime is cuda:{}",
                    region.id.0,
                    self.runtime.device_index()
                )))
            }
            (StoragePlacement::Vram { device }, ComputePlacement::Cuda { device: compute }) => {
                Err(AegisError::Unsupported(format!(
                    "region `{}` has cross-device placement store=vram:{device} compute=cuda:{compute}; cross-device loaders are not implemented yet",
                    region.id.0
                )))
            }
            (_, ComputePlacement::Cpu) => Err(AegisError::Unsupported(format!(
                "region `{}` is compute=cpu; CUDA resident loader refused to load it",
                region.id.0
            ))),
            (_, ComputePlacement::Wgpu { device }) => Err(AegisError::Unsupported(format!(
                "region `{}` is compute=wgpu:{device}; CUDA resident loader refused to load it",
                region.id.0
            ))),
        }
    }

    fn should_repack_native_mxfp4(&self, prefix: &str, kernel_family: KernelFamily) -> bool {
        kernel_family == KernelFamily::CudaNativeFp4TensorCores
            && self.runtime.config().native_mxfp4_repack
            && !(self.runtime.config().cutlass_nvfp4_repack
                && native_layout_cutlass_prefill_sidecar(prefix))
    }

    fn should_repack_cutlass_nvfp4(
        &self,
        prefix: &str,
        kernel_family: KernelFamily,
        resident_layout: LinearResidentLayout,
    ) -> bool {
        resident_layout == LinearResidentLayout::CudaR4fE2m1Ue4m3
            || kernel_family == KernelFamily::CudaCutlassFp4TensorCores
            || (kernel_family == KernelFamily::CudaNativeFp4TensorCores
                && self.runtime.config().cutlass_nvfp4_repack
                && native_layout_cutlass_prefill_sidecar(prefix))
    }
}

impl CudaWeightLoader<'_> {
    /// Returns a 1×1 NvFP4 placeholder that satisfies the type system but is never used for
    /// computation. MoE layers hold real per-expert linears in `CudaMoE`; the `CudaLayer`
    /// gate/up/down fields are dummies so the struct is always fully initialised.
    pub fn alloc_dummy_nvfp4_linear(&self, name: &str) -> Result<DeviceNvfp4Linear> {
        let stub = self
            .runtime
            .stream
            .clone_htod(&[0u8])
            .map_err(map_cuda_err("htod dummy nvfp4 stub"))?;
        let stub2 = self
            .runtime
            .stream
            .clone_htod(&[0u8])
            .map_err(map_cuda_err("htod dummy nvfp4 stub2"))?;
        Ok(DeviceNvfp4Linear {
            name: name.to_string(),
            rows: 1,
            cols: 1,
            packed_bytes: 1,
            scale_bytes: 1,
            input_scale: 1.0,
            output_scale: 1.0,
            kernel_family: KernelFamily::CpuScalar,
            resident_layout: LinearResidentLayout::PackedSource,
            residency: TensorResidencyPlan::VramResident {
                device: self.runtime.device_index(),
            },
            packed: stub,
            scales: stub2,
            native_mxfp4: None,
            cutlass_nvfp4: None,
            host_weights: None,
        })
    }
}

fn native_layout_cutlass_prefill_sidecar(prefix: &str) -> bool {
    prefix.ends_with(".self_attn.o_proj")
        || prefix.ends_with(".mlp.gate_proj")
        || prefix.ends_with(".mlp.up_proj")
        || prefix.ends_with(".mlp.down_proj")
}

struct Nvfp4LinearHostParts {
    spec: Nvfp4LinearSpec,
    packed: LoadedHostTensor,
    scales: LoadedHostTensor,
}

fn load_nvfp4_linear_host_parts(
    artifact: &ModelArtifact,
    prefix: &str,
    store: StoragePlacement,
    loader: &mut TensorStorageLoader,
) -> Result<Nvfp4LinearHostParts> {
    let weight = artifact
        .tensors
        .get(&format!("{prefix}.weight"))
        .ok_or_else(|| AegisError::InvalidPlan(format!("missing `{prefix}.weight`")))?;
    let scales = artifact
        .tensors
        .get(&format!("{prefix}.weight_scale"))
        .ok_or_else(|| AegisError::InvalidPlan(format!("missing `{prefix}.weight_scale`")))?;
    let output_scale = read_nvfp4_pertensor_scale(
        artifact, prefix, "weight_scale_2", "weight_global_scale", loader, store,
    )?.unwrap_or(1.0);
    let input_scale = read_nvfp4_pertensor_scale(
        artifact, prefix, "input_scale", "input_global_scale", loader, store,
    )?.unwrap_or(1.0);
    let spec = Nvfp4LinearSpec::from_tensors(prefix, weight, scales, input_scale, output_scale)?;
    let packed = loader.load_for_store(weight, store)?;
    let scales = loader.load_for_store(scales, store)?;
    Ok(Nvfp4LinearHostParts {
        spec,
        packed,
        scales,
    })
}

/// Reads an NVFP4 per-tensor scale, normalizing the two checkpoint conventions
/// to a MULTIPLIER (what the dequant kernel expects). Gemma stores
/// `weight_scale_2`/`input_scale` as a multiplier directly. compressed-tensors
/// (Qwen3-Next) stores `weight_global_scale`/`input_global_scale` as a DIVISOR
/// (`≈ FP4_MAX*FP8_MAX/amax`, e.g. 45312) — invert it so `nibble·block·scale`
/// matches. Returns `None` if neither name is present.
pub(crate) fn read_nvfp4_pertensor_scale(
    artifact: &ModelArtifact,
    prefix: &str,
    gemma_suffix: &str,
    ct_suffix: &str,
    loader: &mut TensorStorageLoader,
    store: StoragePlacement,
) -> Result<Option<f32>> {
    if let Some(t) = artifact.tensors.get(&format!("{prefix}.{gemma_suffix}")) {
        Ok(Some(read_scalar_f32_with_loader(loader, t, store)?))
    } else if let Some(t) = artifact.tensors.get(&format!("{prefix}.{ct_suffix}")) {
        let v = read_scalar_f32_with_loader(loader, t, store)?;
        Ok(Some(if v.abs() > f32::MIN_POSITIVE { 1.0 / v } else { 1.0 }))
    } else {
        Ok(None)
    }
}

pub(crate) fn read_scalar_f32_with_loader(
    loader: &mut TensorStorageLoader,
    tensor: &TensorInfo,
    store: StoragePlacement,
) -> Result<f32> {
    let loaded: LoadedHostTensor = loader.load_for_store(tensor, store)?;
    let bytes = loaded.as_bytes();
    match tensor.dtype {
        TensorDType::F32 if bytes.len() == 4 => Ok(f32::from_le_bytes(bytes.try_into().map_err(
            |_| AegisError::InvalidPlan(format!("bad scalar F32 tensor `{}`", tensor.name)),
        )?)),
        TensorDType::BF16 if bytes.len() == 2 => {
            let bits = u16::from_le_bytes(bytes.try_into().map_err(
                |_| AegisError::InvalidPlan(format!("bad scalar BF16 tensor `{}`", tensor.name)),
            )?);
            // BF16 is the top 16 bits of an F32; shift left by 16 to recover f32 bits.
            Ok(f32::from_bits((bits as u32) << 16))
        }
        _ => Err(AegisError::InvalidPlan(format!(
            "`{}` must be a scalar F32 or BF16 tensor, got {:?} ({} bytes)",
            tensor.name, tensor.dtype, bytes.len()
        ))),
    }
}

fn first_nvfp4_linear_prefix(region: &GraphRegion) -> Option<&str> {
    region
        .tensors
        .iter()
        .find(|tensor| is_nvfp4_linear_weight(tensor))
        .and_then(|tensor| tensor.info.name.strip_suffix(".weight"))
}

fn is_nvfp4_linear_weight(tensor: &aegisllm_base::graph::GraphTensor) -> bool {
    matches!(
        tensor.role,
        TensorRole::Query
            | TensorRole::Key
            | TensorRole::Value
            | TensorRole::Output
            | TensorRole::Gate
            | TensorRole::Up
            | TensorRole::Down
    ) && tensor.info.dtype == TensorDType::U8
}

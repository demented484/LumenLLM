//! VRAM expert weight cache (Phase 4 of perf overhaul).
//!
//! On Gemma-4-26B-A4B (16 GB VRAM target), after BF16/embed/lm_head/KV/scratch
//! reside in VRAM there is typically ~9 GB free. NVFP4 expert weights total
//! ~12 GB. Caching ~75% of experts in VRAM lets inference skip the per-call
//! H2D copy from the pinned-host arena for cache hits — the dominant
//! bandwidth and overhead cost of the MoE forward path.
//!
//! Cache layout:
//!   * One contiguous `DeviceBuffer<u8>` of `capacity_bytes`, populated at
//!     load time (after the runtime knows how much free VRAM remains).
//!   * Per-weight metadata keyed by weight name (e.g.
//!     `model.layers.5.mlp.experts.42.gate_proj`) → byte offsets for packed
//!     data and FP8 scales inside the buffer.
//!
//! Population strategy (this implementation): static, in load order, until
//! capacity is exhausted. Future iterations can add LRU/LFU eviction or
//! profile-guided cache populate.
//!
//! Cache hit path: dispatch reads `(buffer + packed_off..)` and
//! `(buffer + scales_off..)` directly via `CudaSlice::slice` views and
//! launches the existing `aegis_nvfp4_linear_prequantized_batched_*` kernel.
//! Zero CPU work, zero PCIe traffic for cached experts.
//!
//! Cache miss path: fall through to the existing staging-pool path
//! (host-pinned arena → VRAM staging slot → kernel).

use std::collections::HashMap;
use std::sync::Arc;

use super::runtime::{CudaRuntime, map_cuda_err};
use super::types::{DeviceBuffer, DeviceNvfp4Linear};
use aegisllm_base::error::{AegisError, Result};

/// Per-weight cache entry. Offsets are relative to the cache buffer's start.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CacheEntry {
    pub(crate) packed_offset: usize,
    pub(crate) packed_bytes: usize,
    pub(crate) scales_offset: usize,
    pub(crate) scales_bytes: usize,
}

/// VRAM expert cache. One per CudaRuntime, populated after loading.
pub(crate) struct VramExpertCache {
    /// Single backing VRAM allocation that holds many experts' bytes.
    buffer: DeviceBuffer<u8>,
    /// Total capacity (bytes).
    capacity_bytes: usize,
    /// Bytes already used by inserted entries.
    used_bytes: usize,
    /// Map from weight name (e.g. NVFP4 linear.name) to its cache entry.
    entries: HashMap<String, CacheEntry>,
}

impl VramExpertCache {
    pub(crate) fn new(runtime: &CudaRuntime, capacity_bytes: usize) -> Result<Self> {
        let buffer = runtime.alloc_u8(capacity_bytes)?;
        Ok(Self {
            buffer,
            capacity_bytes,
            used_bytes: 0,
            entries: HashMap::new(),
        })
    }

    /// Try to add an expert's packed/scales bytes to the cache. Returns
    /// `Ok(true)` on insert, `Ok(false)` if the cache is full and the entry
    /// didn't fit. `Err` for I/O / driver problems.
    pub(crate) fn try_insert(
        &mut self,
        runtime: &CudaRuntime,
        name: &str,
        packed_src: &[u8],
        scales_src: &[u8],
    ) -> Result<bool> {
        let total = packed_src.len() + scales_src.len();
        if self.used_bytes.saturating_add(total) > self.capacity_bytes {
            return Ok(false);
        }
        let packed_offset = self.used_bytes;
        let scales_offset = packed_offset + packed_src.len();
        // H2D copy bytes into the cache buffer at the assigned offset. Source
        // is the pinned-host arena (PinnedHostSlice), so this hits fast pinned
        // DMA. Done on the compute stream synchronously during load — no need
        // for the staging-pool dance here since the buffer is permanently
        // owned by the cache.
        {
            let mut packed_view = self
                .buffer
                .slice
                .slice_mut(packed_offset..packed_offset + packed_src.len());
            runtime
                .stream
                .memcpy_htod(packed_src, &mut packed_view)
                .map_err(map_cuda_err("vram cache htod packed"))?;
        }
        {
            let mut scales_view = self
                .buffer
                .slice
                .slice_mut(scales_offset..scales_offset + scales_src.len());
            runtime
                .stream
                .memcpy_htod(scales_src, &mut scales_view)
                .map_err(map_cuda_err("vram cache htod scales"))?;
        }
        self.used_bytes += total;
        self.entries.insert(
            name.to_string(),
            CacheEntry {
                packed_offset,
                packed_bytes: packed_src.len(),
                scales_offset,
                scales_bytes: scales_src.len(),
            },
        );
        Ok(true)
    }

    pub(crate) fn get(&self, name: &str) -> Option<&CacheEntry> {
        self.entries.get(name)
    }

    pub(crate) fn buffer(&self) -> &DeviceBuffer<u8> {
        &self.buffer
    }

    pub(crate) fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    pub(crate) fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }

    pub(crate) fn entry_count(&self) -> usize {
        self.entries.len()
    }
}

impl std::fmt::Debug for VramExpertCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VramExpertCache")
            .field("capacity_bytes", &self.capacity_bytes)
            .field("used_bytes", &self.used_bytes)
            .field("entries", &self.entries.len())
            .finish()
    }
}

pub(crate) type CacheHandle = Arc<VramExpertCache>;

/// Query free VRAM and pick a cache budget. Reserves a safety margin so
/// later allocations (CUDA Graph capture, cuBLASLt workspace expansion) have
/// breathing room.
pub(crate) fn pick_cache_capacity(runtime: &CudaRuntime, safety_margin_bytes: usize) -> Result<usize> {
    let (free, _total) = runtime
        .stream
        .context()
        .mem_get_info()
        .map_err(map_cuda_err("query free vram"))?;
    if free <= safety_margin_bytes {
        return Err(AegisError::InvalidPlan(format!(
            "insufficient free VRAM for expert cache: free={free} margin={safety_margin_bytes}"
        )));
    }
    Ok(free - safety_margin_bytes)
}

/// Try to insert a single host-resident NVFP4 expert weight into the cache.
/// Returns `Ok(true)` if it fit, `Ok(false)` if the cache is full.
pub(crate) fn try_cache_nvfp4_expert(
    cache: &mut VramExpertCache,
    runtime: &CudaRuntime,
    expert: &DeviceNvfp4Linear,
) -> Result<bool> {
    let Some(host) = expert.host_weights.as_deref() else {
        return Ok(false);
    };
    let packed_bytes = host.packed.as_bytes()?;
    let scales_bytes = host.scales.as_bytes()?;
    cache.try_insert(runtime, &expert.name, packed_bytes, scales_bytes)
}

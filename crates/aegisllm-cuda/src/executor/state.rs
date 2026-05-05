use std::fmt;

use cudarc::driver::{CudaGraph, PinnedHostSlice};

use crate::cuda::{CudaRuntime, DeviceBf16Matrix, DeviceBuffer, DeviceNvfp4Linear};
use crate::cuda::staging::LinearStagingPool;
use aegisllm_base::planning::placement::StoragePlacement;

/// Wraps either an NVFP4 or BF16 linear projection.
#[derive(Debug)]
pub(super) enum CudaLinear {
    Nvfp4(DeviceNvfp4Linear),
    Bf16(DeviceBf16Matrix),
}

impl CudaLinear {
    pub(super) fn rows(&self) -> usize {
        match self { Self::Nvfp4(l) => l.rows, Self::Bf16(m) => m.rows }
    }
    pub(super) fn cols(&self) -> usize {
        match self { Self::Nvfp4(l) => l.cols, Self::Bf16(m) => m.cols }
    }
    pub(super) fn name(&self) -> &str {
        match self { Self::Nvfp4(l) => &l.name, Self::Bf16(m) => &m.name }
    }
    pub(super) fn is_host_resident(&self) -> bool {
        match self { Self::Nvfp4(l) => l.is_host_resident(), Self::Bf16(m) => m.is_host_resident() }
    }
    pub(super) fn as_nvfp4(&self) -> Option<&DeviceNvfp4Linear> {
        match self { Self::Nvfp4(l) => Some(l), _ => None }
    }
    pub(super) fn as_bf16(&self) -> Option<&DeviceBf16Matrix> {
        match self { Self::Bf16(m) => Some(m), _ => None }
    }
    pub(super) fn cutlass_nvfp4_enabled(&self, runtime: &CudaRuntime) -> bool {
        match self { Self::Nvfp4(l) => runtime.cutlass_nvfp4_inference_enabled_for(l), _ => false }
    }
    pub(super) fn native_mxfp4_enabled(&self, runtime: &CudaRuntime) -> bool {
        match self { Self::Nvfp4(l) => runtime.native_mxfp4_inference_enabled_for(l), _ => false }
    }
}

/// Per-routed-expert weights (gate, up, down projections).
#[derive(Debug)]
pub(super) struct CudaMoEExpert {
    pub(super) gate_proj: DeviceNvfp4Linear,
    pub(super) up_proj: DeviceNvfp4Linear,
    pub(super) down_proj: DeviceNvfp4Linear,
}

/// Shared (always-active) expert weights — present in some MoE models (e.g. Nemotron 3, Gemma 4).
#[derive(Debug)]
pub(super) struct CudaMoEShared {
    pub(super) gate_proj: CudaLinear,
    pub(super) up_proj: CudaLinear,
    pub(super) down_proj: CudaLinear,
}

/// MoE data for one transformer layer.
#[derive(Debug)]
pub(super) struct CudaMoE {
    /// Router weight matrix: [num_experts, hidden_size] in BF16.
    pub(super) router: DeviceBf16Matrix,
    /// Gemma 4: per-input-dim scale applied to the router input BEFORE projection.
    /// Stored as a [hidden_size] BF16 vector at `{prefix}.router.scale`.
    pub(super) router_input_scale: Option<DeviceBuffer<f32>>,
    /// Gemma 4: per-expert scale applied to top-k routing weights AFTER softmax+topk
    /// (transformers Gemma4TextRouter applies it as `top_k_weights *= per_expert_scale[idx]`).
    /// Cached as a host `Vec<f32>` for legacy callers that still run top-k on CPU
    /// (decode path); the GPU top-k path uses `router_per_expert_scale_device`.
    pub(super) router_per_expert_scale_host: Option<Vec<f32>>,
    /// Always-populated device buffer of `[num_experts]` per-expert scales.
    /// When the model has no per-expert scale, this holds an identity (all 1.0)
    /// so the GPU `router_softmax_topk_device` kernel can branch-free always
    /// apply scaling.
    pub(super) router_per_expert_scale_device: DeviceBuffer<f32>,
    pub(super) experts: Vec<CudaMoEExpert>,
    pub(super) shared_expert: Option<CudaMoEShared>,
    pub(super) top_k: usize,
    pub(super) num_experts: usize,
    pub(super) expert_intermediate_size: usize,
}

/// Extra scratch buffers allocated only when the model contains MoE layers.
#[derive(Debug)]
pub(super) struct CudaMoEScratch {
    pub(super) router_logits: DeviceBuffer<f32>,
    /// Gemma 4: scratch holding the router input scaled by `router.scale`.
    /// Sized to `hidden_size`; only used when `router_input_scale` is present.
    pub(super) router_input_scratch: DeviceBuffer<f32>,
    pub(super) moe_acc: DeviceBuffer<f32>,
    pub(super) expert_gate: DeviceBuffer<f32>,
    pub(super) expert_up: DeviceBuffer<f32>,
    pub(super) expert_swiglu: DeviceBuffer<f32>,
    pub(super) expert_out: DeviceBuffer<f32>,
    pub(super) quant_expert: DeviceBuffer<f32>,
    pub(super) mxfp4_expert: DeviceBuffer<u8>,
}

/// Wraps `CudaGraph` so that `CudaLlamaState` satisfies `Send`.
/// Safety: `CudaLlamaState` is used from a single thread per generation session.
pub(super) struct SendCudaGraph(pub(super) CudaGraph);
unsafe impl Send for SendCudaGraph {}
impl fmt::Debug for SendCudaGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaGraph").finish()
    }
}
use aegisllm_base::generation::PrefillStageTimings;

#[derive(Debug)]
pub(super) struct CudaLlamaExecutor {
    pub(super) runtime: CudaRuntime,
    pub(super) hidden_size: usize,
    pub(super) num_attention_heads: usize,
    pub(super) num_kv_heads: usize,
    pub(super) head_dim: usize,
    pub(super) rms_norm_eps: f32,
    pub(super) embed_tokens: DeviceBf16Matrix,
    pub(super) final_norm: DeviceBuffer<f32>,
    pub(super) lm_head: DeviceBf16Matrix,
    pub(super) layers: Vec<CudaLayer>,
    pub(super) kv_context_size: usize,
    pub(super) prefill_chunk_size: usize,
    pub(super) prefill_stage_timings_enabled: bool,
    /// Gemma 4: `cap * tanh(logits / cap)` applied to lm_head output before sampling.
    pub(super) lm_head_softcap: Option<f32>,
    /// Multiplicative scale applied to token embeddings after lookup (Gemma 4 = sqrt(hidden_size)).
    pub(super) embed_scale: Option<f32>,
    /// True when any layer has host-resident (StagedHostToDevice) weights.
    /// Used to inhibit CUDA Graph capture (H2D copies cannot be in a captured graph).
    pub(super) has_staged_layers: bool,
    /// True when any layer has host-resident KV; inhibits CUDA Graph capture.
    pub(super) has_staged_kv: bool,
    /// Tail tier: KV store for layers >= `kv_first_n_layers` (or all layers when
    /// `kv_first_n_layers` is `None`).
    pub(super) kv_store: StoragePlacement,
    /// First-N count and tier. Layers `0..kv_first_n_layers` use `kv_first_store`.
    /// `kv_first_store=None` with `kv_first_n_layers=Some(_)` means "VRAM derived
    /// from compute" (legacy behavior preserved for the simple force-VRAM-first-N case).
    pub(super) kv_first_n_layers: Option<usize>,
    pub(super) kv_first_store: Option<StoragePlacement>,
}

#[derive(Debug)]
pub(super) struct CudaLayer {
    pub(super) input_norm_weight: DeviceBuffer<f32>,
    /// Pre-MLP norm (HuggingFace: `post_attention_layernorm.weight`).
    pub(super) post_attention_norm_weight: DeviceBuffer<f32>,
    /// Gemma 4 PrePost: applied to attention output before residual add.
    pub(super) post_attn_sublayer_norm: Option<DeviceBuffer<f32>>,
    /// Gemma 4 PrePost: applied to MLP output before residual add.
    pub(super) post_mlp_sublayer_norm: Option<DeviceBuffer<f32>>,
    /// Gemma 4 MoE: post-norm on shared-MLP stream before combining with experts.
    pub(super) post_feedforward_layernorm_1: Option<DeviceBuffer<f32>>,
    /// Gemma 4 MoE: separate pre-norm for expert inputs (pre_feedforward_layernorm_2).
    pub(super) pre_feedforward_layernorm_2: Option<DeviceBuffer<f32>>,
    /// Gemma 4 MoE: post-norm on routed-expert stream before combining with shared MLP.
    pub(super) post_feedforward_layernorm_2: Option<DeviceBuffer<f32>>,
    /// Gemma 4: per-layer scalar multiplier applied after each layer's residual add.
    pub(super) layer_scalar: Option<f32>,
    pub(super) q_proj: CudaLinear,
    pub(super) k_proj: CudaLinear,
    pub(super) v_proj: CudaLinear,
    pub(super) qkv_proj: Option<CudaLinear>,
    pub(super) o_proj: CudaLinear,
    /// Gemma 4: RMS norm applied per-head to Q after the projection, before RoPE.
    pub(super) q_norm_weight: Option<DeviceBuffer<f32>>,
    /// Gemma 4: RMS norm applied per-head to K after the projection, before RoPE.
    pub(super) k_norm_weight: Option<DeviceBuffer<f32>>,
    pub(super) gate_proj: DeviceNvfp4Linear,
    pub(super) up_proj: DeviceNvfp4Linear,
    pub(super) down_proj: DeviceNvfp4Linear,
    /// 0 = full causal; >0 = sliding-window (Gemma 4 local layers, Mistral).
    pub(super) window_size: usize,
    /// Per-layer RoPE config with the correct partial_dim baked in.
    pub(super) rope: crate::cuda::DeviceRopeConfig,
    /// Present only for MoE layers (e.g. Gemma 4 26B, Qwen 3 MoE).
    pub(super) moe: Option<Box<CudaMoE>>,
    /// Per-layer head_dim. Differs from model-wide for Gemma 4 global layers (512 vs 256).
    pub(super) layer_head_dim: usize,
    /// Per-layer KV head count. Differs from model-wide for Gemma 4 global layers (2 vs 8).
    pub(super) layer_num_kv_heads: usize,
}

#[derive(Debug)]
pub struct CudaLlamaState {
    pub(super) position: usize,
    pub(super) hidden: DeviceBuffer<f32>,
    pub(super) logits: DeviceBuffer<f32>,
    pub(super) sampled_token: DeviceBuffer<u32>,
    pub(super) layers: Vec<CudaLayerState>,
    pub(super) scratch: CudaScratch,
    pub(super) prefill: Option<CudaPrefillScratch>,
    pub(super) prefill_timings: CudaPrefillStageTimings,
    /// Device buffers holding the current decode `position` and `seq_len` (= position + 1).
    /// Kept here (not in CudaScratch) so we can borrow them alongside `&mut scratch`.
    /// Updated before each decode step (outside graph capture), read by the ptr-based kernels.
    pub(super) decode_position: DeviceBuffer<u32>,
    pub(super) decode_seq_len: DeviceBuffer<u32>,
    /// Captured CUDA Graph for decode steps. Set after the first decode step.
    /// When set, each subsequent step updates decode_position/decode_seq_len then replays
    /// this graph instead of issuing ~645 kernel launches.
    pub(super) decode_graph: Option<SendCudaGraph>,
}

#[derive(Debug)]
pub(super) struct CudaLayerState {
    pub(super) kv: CudaKvCache,
}

/// KV weights stored in CUDA-pinned host RAM for `kv-cache.store=ram` configs.
/// CPU reads are safe (no WRITECOMBINED flag), enabling D2H writeback after each token.
#[derive(Debug)]
pub(super) struct HostKvWeights {
    pub(super) keys: PinnedHostSlice<u16>,
    pub(super) values: PinnedHostSlice<u16>,
}

/// VRAM staging slot for host-resident KV layers.
/// Before each attention step, existing KV entries are H2D-copied here from pinned RAM,
/// the store kernel appends the new token, attention reads from here, then D2H writeback
/// copies the new slot back to pinned RAM. Two slots in `KvStagingPool` enable async
/// transfer pipelining (next-layer H2D overlaps current-layer compute).
#[derive(Debug)]
pub(super) struct KvStagingSlot {
    pub(super) keys: DeviceBuffer<u16>,
    pub(super) values: DeviceBuffer<u16>,
    pub(super) context_size: usize,
    pub(super) kv_width: usize,
}

/// Double-buffered staging pool. Layer L uses `slots[L % 2]` so that while layer L
/// is computing on slots[0], the transfer stream can prefetch H2D for layer L+1
/// into slots[1]. Per-slot CudaEvents track the most recent compute on that slot,
/// so the transfer stream waits for compute to finish before reusing a slot.
#[derive(Debug)]
pub(super) struct KvStagingPool {
    pub(super) slots: [KvStagingSlot; 2],
    /// Last compute event recorded for each slot (cleared after transfer-side wait).
    pub(super) last_compute_event: [Option<cudarc::driver::CudaEvent>; 2],
}

#[derive(Debug)]
#[allow(dead_code)]
pub(super) struct CudaKvCache {
    pub(super) layout: CudaKvCacheLayout,
    /// VRAM: full-sized for Dense, 1-element stub for HostResident.
    pub(super) keys: DeviceBuffer<u16>,
    /// VRAM: full-sized for Dense, 1-element stub for HostResident.
    pub(super) values: DeviceBuffer<u16>,
    /// Non-None when `kv-cache.store=ram`: actual KV lives in pinned host RAM.
    pub(super) host: Option<Box<HostKvWeights>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub(super) enum CudaKvCacheLayout {
    Dense {
        context_size: usize,
        kv_width: usize,
    },
    Paged {
        block_size: usize,
        num_blocks: usize,
        kv_width: usize,
    },
}

impl CudaKvCache {
    pub(super) fn dense(
        runtime: &CudaRuntime,
        context_size: usize,
        kv_width: usize,
    ) -> aegisllm_base::error::Result<Self> {
        let len = context_size.checked_mul(kv_width).ok_or_else(|| {
            aegisllm_base::error::AegisError::InvalidPlan(format!(
                "CUDA dense KV cache length overflow: context={} kv_width={}",
                context_size, kv_width
            ))
        })?;
        Ok(Self {
            layout: CudaKvCacheLayout::Dense {
                context_size,
                kv_width,
            },
            keys: runtime.alloc_u16(len)?,
            values: runtime.alloc_u16(len)?,
            host: None,
        })
    }

    /// Allocate host-resident KV: full-size pinned RAM + 1-element VRAM stubs.
    /// The shared `KvStagingSlot` in `CudaScratch` is used at inference time.
    pub(super) fn staged_host(
        runtime: &CudaRuntime,
        context_size: usize,
        kv_width: usize,
    ) -> aegisllm_base::error::Result<Self> {
        use aegisllm_base::error::AegisError;
        let len = context_size.checked_mul(kv_width).ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "CUDA host-resident KV length overflow: context={context_size} kv_width={kv_width}"
            ))
        })?;
        let keys_host = runtime.alloc_pinned_u16(len)?;
        let values_host = runtime.alloc_pinned_u16(len)?;
        // 1-element VRAM stubs (staging slot is used for all actual GPU operations).
        Ok(Self {
            layout: CudaKvCacheLayout::Dense {
                context_size,
                kv_width,
            },
            keys: runtime.alloc_u16(1)?,
            values: runtime.alloc_u16(1)?,
            host: Some(Box::new(HostKvWeights {
                keys: keys_host,
                values: values_host,
            })),
        })
    }

    pub(super) fn is_host_resident(&self) -> bool {
        self.host.is_some()
    }
}

#[derive(Debug)]
pub(super) struct CudaScratch {
    pub(super) input_normed: DeviceBuffer<f32>,
    pub(super) quant_hidden: DeviceBuffer<f32>,
    pub(super) quant_intermediate: DeviceBuffer<f32>,
    pub(super) mxfp4_hidden: DeviceBuffer<u8>,
    pub(super) mxfp4_intermediate: DeviceBuffer<u8>,
    pub(super) cutlass_payload: DeviceBuffer<u8>,
    pub(super) cutlass_scales: DeviceBuffer<u8>,
    pub(super) cutlass_workspace: DeviceBuffer<u8>,
    pub(super) q: DeviceBuffer<f32>,
    pub(super) k: DeviceBuffer<f32>,
    pub(super) v: DeviceBuffer<f32>,
    /// Gemma 4: scratch for per-head q_norm/k_norm output (rms_norm cannot run in-place).
    /// Sized to hold the larger of q (`max_q_width`) or k (`max_kv_width`).
    pub(super) qk_norm_scratch: DeviceBuffer<f32>,
    pub(super) attn_split_acc: DeviceBuffer<f32>,
    pub(super) attn_split_m: DeviceBuffer<f32>,
    pub(super) attn_split_l: DeviceBuffer<f32>,
    pub(super) attn_context: DeviceBuffer<f32>,
    pub(super) attn_out: DeviceBuffer<f32>,
    pub(super) residual: DeviceBuffer<f32>,
    pub(super) post_normed: DeviceBuffer<f32>,
    pub(super) gate: DeviceBuffer<f32>,
    pub(super) up: DeviceBuffer<f32>,
    pub(super) swiglu: DeviceBuffer<f32>,
    pub(super) mlp_out: DeviceBuffer<f32>,
    pub(super) hidden_out: DeviceBuffer<f32>,
    pub(super) final_hidden: DeviceBuffer<f32>,
    pub(super) argmax_block_values: DeviceBuffer<f32>,
    pub(super) argmax_block_indices: DeviceBuffer<u32>,
    /// Allocated only when model has MoE layers.
    pub(super) moe: Option<Box<CudaMoEScratch>>,
    /// Allocated only when the model has any host-resident (StagedHostToDevice) linears.
    /// Sized to hold the largest staged layer's packed + scales in one slot.
    pub(super) staging_pool: Option<Box<LinearStagingPool>>,
    /// Allocated only when any layer has host-resident KV. Two VRAM slots sized
    /// `context_size × kv_width` each, used in ping-pong to overlap async H2D
    /// (next layer) with compute (current layer) via the dedicated transfer stream.
    pub(super) kv_staging: Option<Box<KvStagingPool>>,
}

#[derive(Debug)]
#[allow(dead_code)]
pub(super) struct CudaPrefillScratch {
    pub(super) chunk_size: usize,
    pub(super) max_sequences: usize,
    pub(super) block_table_capacity: usize,
    pub(super) request_ids_host: Vec<u32>,
    pub(super) seq_ids_host: Vec<u32>,
    pub(super) token_host: Vec<u32>,
    pub(super) position_host: Vec<u32>,
    pub(super) slot_mapping_host: Vec<u32>,
    pub(super) cu_q_host: Vec<u32>,
    pub(super) cu_k_host: Vec<u32>,
    pub(super) context_lens_host: Vec<u32>,
    pub(super) block_tables_host: Vec<u32>,
    pub(super) request_ids: DeviceBuffer<u32>,
    pub(super) seq_ids: DeviceBuffer<u32>,
    pub(super) tokens: DeviceBuffer<u32>,
    pub(super) positions: DeviceBuffer<u32>,
    pub(super) slot_mapping: DeviceBuffer<u32>,
    pub(super) cu_q: DeviceBuffer<u32>,
    pub(super) cu_k: DeviceBuffer<u32>,
    pub(super) context_lens: DeviceBuffer<u32>,
    pub(super) block_tables: DeviceBuffer<u32>,
    pub(super) hidden: DeviceBuffer<f32>,
    pub(super) input_normed: DeviceBuffer<f32>,
    pub(super) quant_hidden: DeviceBuffer<f32>,
    pub(super) quant_intermediate: DeviceBuffer<f32>,
    pub(super) mxfp4_hidden: DeviceBuffer<u8>,
    pub(super) mxfp4_intermediate: DeviceBuffer<u8>,
    pub(super) cutlass_payload: DeviceBuffer<u8>,
    pub(super) cutlass_scales: DeviceBuffer<u8>,
    pub(super) cutlass_workspace: DeviceBuffer<u8>,
    pub(super) qkv: DeviceBuffer<f32>,
    pub(super) q: DeviceBuffer<f32>,
    pub(super) q_half: DeviceBuffer<u16>,
    pub(super) attn_split_acc: DeviceBuffer<f32>,
    pub(super) attn_split_m: DeviceBuffer<f32>,
    pub(super) attn_split_l: DeviceBuffer<f32>,
    pub(super) k: DeviceBuffer<f32>,
    pub(super) v: DeviceBuffer<f32>,
    pub(super) attn_context: DeviceBuffer<f32>,
    pub(super) attn_out: DeviceBuffer<f32>,
    pub(super) gate: DeviceBuffer<f32>,
    pub(super) up: DeviceBuffer<f32>,
    pub(super) swiglu: DeviceBuffer<f32>,
    pub(super) mlp_out: DeviceBuffer<f32>,
    /// BF16 scratch for cuBLASLt input (F32→BF16 staging). Sized for
    /// `chunk_size * max(hidden, intermediate, max_q_width, q_width+2*kv_width)`.
    pub(super) bf16_in_scratch: DeviceBuffer<u16>,
    /// BF16 scratch for cuBLASLt output (before BF16→F32 conversion). Same size.
    pub(super) bf16_out_scratch: DeviceBuffer<u16>,
    /// MoE prefill scratch (allocated only when the model has MoE layers).
    pub(super) moe: Option<Box<CudaMoEPrefillScratch>>,
}

/// Per-chunk scratch for chunked MoE prefill. Sized for `chunk_size` tokens.
#[derive(Debug)]
pub(super) struct CudaMoEPrefillScratch {
    /// `[chunk_size, num_experts]` — router logits per token (host download).
    pub(super) router_logits: DeviceBuffer<f32>,
    /// `[chunk_size, hidden_size]` — input to router after norm + scale + root_size.
    pub(super) router_input: DeviceBuffer<f32>,
    /// `[chunk_size, hidden_size]` — pre_feedforward_layernorm_2(residual) for experts.
    pub(super) expert_input: DeviceBuffer<f32>,
    /// `[chunk_size, hidden_size]` — accumulator for routed-expert outputs.
    pub(super) moe_acc: DeviceBuffer<f32>,
    /// `[chunk_size, hidden_size]` — stream1 (post_feedforward_layernorm_1(shared MLP output)).
    pub(super) stream1: DeviceBuffer<f32>,
    /// Gather buffer: `[max_active_tokens, hidden_size]` for one expert's batch input.
    pub(super) gather_input: DeviceBuffer<f32>,
    /// Gather buffer for intermediate (gate / up): `[max_active_tokens, expert_intermediate]`.
    pub(super) gather_intermediate: DeviceBuffer<f32>,
    /// Gather buffer for swiglu output: `[max_active_tokens, expert_intermediate]`.
    pub(super) gather_swiglu: DeviceBuffer<f32>,
    /// Gather buffer for down_proj output: `[max_active_tokens, hidden_size]`.
    pub(super) gather_out: DeviceBuffer<f32>,
    /// Quantized input scratch for NVFP4 expert matmuls: `[max_active_tokens, max_dim]`.
    pub(super) gather_quant: DeviceBuffer<f32>,
    /// MXFP4 quantized input scratch for native MXFP4 path.
    pub(super) gather_mxfp4: DeviceBuffer<u8>,
    /// Indices buffer: `[max_active_tokens]` source-token indices for current expert.
    pub(super) gather_indices: DeviceBuffer<u32>,
    /// Weights buffer: `[max_active_tokens]` per-token routing weight for current expert.
    pub(super) gather_weights: DeviceBuffer<f32>,
    // ── GPU router top-k scratch (Phase 1 of perf overhaul) ────────────────
    /// Device-resident top-k expert indices: `[chunk_size, top_k]`.
    pub(super) topk_idx: DeviceBuffer<u32>,
    /// Device-resident top-k normalized routing weights: `[chunk_size, top_k]`.
    pub(super) topk_weights: DeviceBuffer<f32>,
    /// Per-expert token list buffer (after device-side bucket sort):
    /// `[num_experts, max_per_expert]` where `max_per_expert = chunk_size * top_k`.
    pub(super) expert_token_lists: DeviceBuffer<u32>,
    /// Per-expert routing-weight buffer (parallel to `expert_token_lists`).
    pub(super) expert_weight_lists: DeviceBuffer<f32>,
    /// Per-expert token count after bucket sort: `[num_experts]`. Downloaded
    /// (small, ~512 bytes for 128 experts) to drive the per-expert matmul
    /// dispatch loop.
    pub(super) expert_counts: DeviceBuffer<u32>,
    /// Stride between per-expert lists (= `max_per_expert` = chunk_size * top_k).
    pub(super) expert_list_stride: usize,
    // ── Phase 2 grouped-MoE scratch (used when VRAM expert cache is on) ────
    /// CSR prefix-sum over `expert_counts`: `expert_offsets[e]` is the start
    /// row in the permuted activation buffer for expert `e`. Length is
    /// `num_experts + 1`; last entry is `total_assignments`.
    pub(super) expert_offsets: DeviceBuffer<u32>,
    /// Per-chunk packed-cached-counts upload buffer. For experts cached in
    /// VRAM the value is the original `expert_counts[e]`; for uncached the
    /// value is 0 (so the grouped GEMM skips them — they go through the
    /// per-expert fallback path that still uses staging).
    pub(super) cached_counts: DeviceBuffer<u32>,
    /// Per-layer per-matmul-position byte offsets into the VRAM cache buffer.
    /// Built on host per chunk by looking up each expert's weight name in the
    /// cache, uploaded before each grouped matvec call. 6 small buffers per
    /// chunk: (gate, up, down) × (packed, scales). Each is `[num_experts]` u32.
    // u64 byte offsets — the VRAM expert cache exceeds 4 GB on Gemma-4-26B,
    // so 32-bit offsets silently wrap around layer 10 and corrupt weights.
    pub(super) gate_packed_offsets: DeviceBuffer<u64>,
    pub(super) gate_scales_offsets: DeviceBuffer<u64>,
    pub(super) up_packed_offsets: DeviceBuffer<u64>,
    pub(super) up_scales_offsets: DeviceBuffer<u64>,
    pub(super) down_packed_offsets: DeviceBuffer<u64>,
    pub(super) down_scales_offsets: DeviceBuffer<u64>,
    /// Per-expert input/output scales per matmul position. Each is
    /// `[num_experts]` f32, uploaded before each grouped matvec call.
    pub(super) gate_input_scales: DeviceBuffer<f32>,
    pub(super) gate_output_scales: DeviceBuffer<f32>,
    pub(super) up_input_scales: DeviceBuffer<f32>,
    pub(super) up_output_scales: DeviceBuffer<f32>,
    pub(super) down_input_scales: DeviceBuffer<f32>,
    pub(super) down_output_scales: DeviceBuffer<f32>,
    /// Permuted activation buffer for the grouped MoE pipeline:
    /// `[chunk_size * top_k, hidden]`.
    pub(super) permuted_input: DeviceBuffer<f32>,
    /// Per-expert NVFP4-prequantized copy of the permuted activation. Written
    /// by `nvfp4_quantize_input_per_expert_device` (different `input_scale`
    /// per expert) and consumed by the grouped prequant GEMM. Sized to fit
    /// either the gate/up input (hidden) or the down input (intermediate),
    /// using the larger of the two.
    pub(super) permuted_input_quant: DeviceBuffer<f32>,
    /// Permuted gate-projection output: `[chunk_size * top_k, expert_intermediate]`.
    pub(super) permuted_gate: DeviceBuffer<f32>,
    /// Permuted up-projection output (reused as SwiGLU/GeGLU input/output).
    pub(super) permuted_up: DeviceBuffer<f32>,
    /// Permuted down-projection output: `[chunk_size * top_k, hidden]`.
    pub(super) permuted_down: DeviceBuffer<f32>,
}

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct CudaPrefillStageTimings {
    pub(super) enabled: bool,
    pub(super) chunks: usize,
    pub(super) prepare_us: u128,
    pub(super) embed_us: u128,
    pub(super) qkv_us: u128,
    pub(super) qkv_tflops: f64,
    pub(super) rope_us: u128,
    pub(super) kv_store_us: u128,
    pub(super) attention_us: u128,
    pub(super) o_proj_us: u128,
    pub(super) mlp_us: u128,
    pub(super) mlp_tflops: f64,
    pub(super) layer_total_us: u128,
    pub(super) sample_us: u128,
}

impl CudaPrefillStageTimings {
    pub(super) fn from_enabled(enabled: bool) -> Self {
        Self {
            enabled,
            ..Self::default()
        }
    }

    pub(super) fn reset(&mut self) {
        let enabled = self.enabled;
        *self = Self {
            enabled,
            ..Self::default()
        };
    }

    pub(super) fn snapshot(self) -> Option<PrefillStageTimings> {
        self.enabled.then_some(PrefillStageTimings {
            chunks: self.chunks,
            prepare_us: self.prepare_us,
            embed_us: self.embed_us,
            qkv_us: self.qkv_us,
            qkv_tflops: self.qkv_tflops,
            rope_us: self.rope_us,
            kv_store_us: self.kv_store_us,
            attention_us: self.attention_us,
            o_proj_us: self.o_proj_us,
            mlp_us: self.mlp_us,
            mlp_tflops: self.mlp_tflops,
            layer_total_us: self.layer_total_us,
            sample_us: self.sample_us,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::CudaPrefillStageTimings;

    #[test]
    fn prefill_stage_timings_reset_preserves_enabled_flag() {
        let mut timings = CudaPrefillStageTimings {
            enabled: true,
            chunks: 3,
            prepare_us: 11,
            embed_us: 7,
            ..CudaPrefillStageTimings::default()
        };
        timings.reset();
        assert!(timings.enabled);
        assert_eq!(timings.chunks, 0);
        assert_eq!(timings.prepare_us, 0);
        assert_eq!(timings.embed_us, 0);
    }
}

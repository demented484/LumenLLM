use std::fmt;

use cudarc::driver::{CudaGraph, PinnedHostSlice};

use crate::cuda::{CudaRuntime, DeviceBf16Matrix, DeviceBuffer, DeviceNvfp4Linear, StandaloneFp8Linear};
use crate::cuda::staging::LinearStagingPool;
use aegisllm_base::planning::placement::StoragePlacement;

/// Global PLE state shared across the whole model. Per-layer `input_gate`,
/// `projection`, and `post_norm` live on `CudaLayer::ple`.
#[derive(Debug)]
pub(super) struct PleGlobal {
    /// `model.embed_tokens_per_layer.weight` — `[vocab, num_layers * ple_dim]`
    /// BF16. Host-resident (~5.4 GiB at E4B); per-token slice streamed on
    /// demand. The slice for token `t` reshapes to `[num_layers, ple_dim]`.
    pub(super) embed_table: DeviceBf16Matrix,
    /// `model.per_layer_model_projection.weight` — `[num_layers * ple_dim,
    /// hidden]` BF16. Projects each token's hidden state into a parallel
    /// per-layer feed.
    pub(super) model_projection: DeviceBf16Matrix,
    /// `model.per_layer_projection_norm.weight` — `[ple_dim]` f32 weights
    /// used by an RMSNorm over the projection output.
    pub(super) projection_norm: DeviceBuffer<f32>,
    /// Feature dim per layer (Gemma-4 E4B: 256).
    pub(super) ple_dim: usize,
    /// `embed_scale_per_layer = sqrt(ple_dim)`.
    pub(super) embed_scale_per_layer: f32,
    /// `per_layer_model_projection_scale = 1 / sqrt(hidden_size)`.
    pub(super) model_projection_scale: f32,
    /// Combine weight = `1 / sqrt(2)` per HF Gemma-4.
    pub(super) combine_scale: f32,
}

/// Per-layer PLE weights, attached to each `CudaLayer` when PLE is enabled.
#[derive(Debug)]
pub(super) struct PleLayer {
    /// `per_layer_input_gate.weight` — `[ple_dim, hidden]` BF16.
    pub(super) input_gate: DeviceBf16Matrix,
    /// `per_layer_projection.weight` — `[hidden, ple_dim]` BF16.
    pub(super) projection: DeviceBf16Matrix,
    /// `post_per_layer_input_norm.weight` — `[hidden]` f32 RMSNorm weight.
    pub(super) post_norm: DeviceBuffer<f32>,
}

/// Wraps a linear projection in NVFP4, BF16, or FP8 storage.
#[derive(Debug)]
pub(super) enum CudaLinear {
    Nvfp4(DeviceNvfp4Linear),
    Bf16(DeviceBf16Matrix),
    Fp8(StandaloneFp8Linear),
}

impl CudaLinear {
    pub(super) fn rows(&self) -> usize {
        match self {
            Self::Nvfp4(l) => l.rows,
            Self::Bf16(m) => m.rows,
            Self::Fp8(m) => m.rows,
        }
    }
    pub(super) fn cols(&self) -> usize {
        match self {
            Self::Nvfp4(l) => l.cols,
            Self::Bf16(m) => m.cols,
            Self::Fp8(m) => m.cols,
        }
    }
    pub(super) fn name(&self) -> &str {
        match self {
            Self::Nvfp4(l) => &l.name,
            Self::Bf16(m) => &m.name,
            Self::Fp8(m) => &m.name,
        }
    }
    pub(super) fn is_host_resident(&self) -> bool {
        match self {
            Self::Nvfp4(l) => l.is_host_resident(),
            Self::Bf16(m) => m.is_host_resident(),
            // FP8 standalone: load-time quantizer writes directly to VRAM.
            Self::Fp8(_) => false,
        }
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
    /// Optional row-stacked `[2 * intermediate, hidden_size]` BF16 fused
    /// gate+up matrix. When `Some`, the shared-MLP path runs a single fused
    /// cuBLASLt BF16 GEMM into a `[batch, 2*intermediate]` row-major buffer,
    /// then a strided GeGLU kernel that consumes the fused layout directly —
    /// saving one BF16 GEMM launch per MoE layer per prefill chunk (and one
    /// per token at decode). The original `gate_proj`/`up_proj` are kept as
    /// metadata-only stubs (real `(name, rows, cols)`, 1-element VRAM
    /// placeholder) so callers that introspect shapes still work; the actual
    /// matmul **must** be routed through `gate_up_fused` when it is `Some`.
    /// Built at load time by `fuse_bf16_gate_up` for the BF16 `Wq::Default`
    /// shared-MLP path; left `None` for FP8 quantized shared MLP.
    pub(super) gate_up_fused: Option<DeviceBf16Matrix>,
    /// Qwen3-Next `shared_expert_gate` weight `[1, hidden]` BF16. When present,
    /// the shared-expert output is scaled by `sigmoid(gate · x)` before being
    /// added to the routed experts. `None` for Gemma (ungated shared MLP).
    pub(super) shared_gate: Option<DeviceBf16Matrix>,
}

/// GPU-driven MoE decode tables (built at load when the expert arena is
/// device-mapped via `AEGIS_GPU_DRIVEN_MOE=1`). Per-projection device arrays of
/// length `num_experts` holding each expert's device-mapped-host base pointer
/// (packed + scales) and its NVFP4 input/output scales. The gather kernel reads
/// the on-device router top-k index buffer, indexes these tables, and streams
/// the selected experts' bytes into a fixed VRAM scratch in one launch — no CPU
/// round-trip, graph-capturable. Per-projection byte strides are uniform across
/// experts within a layer.
#[derive(Debug)]
pub(crate) struct MoeDeviceTables {
    pub(crate) gate_packed_ptrs: DeviceBuffer<u64>,
    pub(crate) up_packed_ptrs: DeviceBuffer<u64>,
    pub(crate) down_packed_ptrs: DeviceBuffer<u64>,
    pub(crate) gate_scale_ptrs: DeviceBuffer<u64>,
    pub(crate) up_scale_ptrs: DeviceBuffer<u64>,
    pub(crate) down_scale_ptrs: DeviceBuffer<u64>,
    pub(crate) gate_in_scale: DeviceBuffer<f32>,
    pub(crate) up_in_scale: DeviceBuffer<f32>,
    pub(crate) down_in_scale: DeviceBuffer<f32>,
    pub(crate) gate_out_scale: DeviceBuffer<f32>,
    pub(crate) up_out_scale: DeviceBuffer<f32>,
    pub(crate) down_out_scale: DeviceBuffer<f32>,
    pub(crate) gate_packed_bytes: usize,
    pub(crate) gate_scale_bytes: usize,
    pub(crate) up_packed_bytes: usize,
    pub(crate) up_scale_bytes: usize,
    pub(crate) down_packed_bytes: usize,
    pub(crate) down_scale_bytes: usize,
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
    /// GPU-driven decode tables; `Some` only when the expert arena was
    /// device-mapped at load and this layer's experts are host-resident NVFP4.
    /// `None` → decode uses the host-streamed staging-pool path.
    pub(super) device_tables: Option<MoeDeviceTables>,
    /// Config-driven (`hidden-layers.experts.compute = cpu`): when `true` the
    /// routed-expert decode GEMV runs on the CPU (read in place from the host
    /// arena via `aegisllm-cpu::moe_layer_experts_into`) instead of streaming
    /// the NVFP4 weights over PCIe. Set at load from `experts_compute_override`;
    /// `false` (the default for every config without an `experts` section) keeps
    /// the unchanged GPU path. The shared expert / GDN / attention / router are
    /// NOT affected — they always run on the layer's `cuda:0` compute.
    pub(super) cpu_experts: bool,
}

/// Extra scratch buffers allocated only when the model contains MoE layers.
#[derive(Debug)]
pub(super) struct CudaMoEScratch {
    pub(super) router_logits: DeviceBuffer<f32>,
    /// Gemma 4: scratch holding the router input scaled by `router.scale`.
    /// Sized to `hidden_size`; only used when `router_input_scale` is present.
    pub(super) router_input_scratch: DeviceBuffer<f32>,
    pub(super) moe_acc: DeviceBuffer<f32>,
    /// Decode async-overlap router: separate routed-expert accumulator so the
    /// shared-MLP output can be staged into `moe_acc` in parallel with the
    /// router top-k dtoh, before routed experts begin to accumulate. Without
    /// this split, shared MLP and routed-experts both write into `moe_acc`
    /// and must serialize. Sized to `hidden_size`.
    pub(super) routed_acc: DeviceBuffer<f32>,
    pub(super) expert_gate: DeviceBuffer<f32>,
    pub(super) expert_up: DeviceBuffer<f32>,
    pub(super) expert_swiglu: DeviceBuffer<f32>,
    pub(super) expert_out: DeviceBuffer<f32>,
    // BATCHED decode MoE (AEGIS_BATCHED_DECODE_MOE): [max_top_k * width] staging
    // for all top_k experts at once (slot on grid.y). Separate from the per-slot
    // buffers above so the default path is untouched. See executor/mlp.rs.
    pub(super) expert_gate_b: DeviceBuffer<f32>,
    pub(super) expert_up_b: DeviceBuffer<f32>,
    pub(super) expert_swiglu_b: DeviceBuffer<f32>,
    pub(super) expert_out_b: DeviceBuffer<f32>,
    pub(super) quant_b: DeviceBuffer<f32>,
    /// Decode-side counterpart to `CudaMoEPrefillScratch.gather_shared_gate_up_fused`.
    /// Sized to `2 * max_expert_intermediate` floats — fits one token's worth
    /// of `[gate_logits, up_logits]` produced by the fused matvec when the
    /// shared MLP has a `gate_up_fused` matrix.
    pub(super) shared_gate_up_fused: DeviceBuffer<f32>,
    /// Qwen3-Next shared-expert gate logit ([1] device scalar). Persistent so
    /// the decode path computes the sigmoid gate fully on-device (no per-layer
    /// blocking `download_f32`, no per-call alloc).
    pub(super) shared_gate_logit: DeviceBuffer<f32>,
    pub(super) quant_expert: DeviceBuffer<f32>,
    /// Decode-only: persistent NVFP4-quantized copy of the routed-expert input
    /// hidden, quantized ONCE per MoE layer with the gate/up `input_scale`.
    ///
    /// Every routed expert's gate_proj and up_proj read the SAME `hidden_out`
    /// and (in the Qwen3.x NVFP4 checkpoint) the SAME `input_scale` — verified
    /// constant across all 256 experts (0.007634) AND gate==up within each
    /// expert. So the fp4-quantized input is byte-identical for every expert's
    /// gate/up GEMV. Quantizing it once per layer (into this buffer) instead of
    /// once per expert (into `quant_expert`, which the down_proj quant clobbers
    /// each iteration) removes `top_k-1` redundant identical quantize launches
    /// per MoE layer — bit-identical output (same bytes feed the same GEMVs).
    /// Sized like `quant_expert` (`max_input`).
    pub(super) quant_gate_up: DeviceBuffer<f32>,
    pub(super) mxfp4_expert: DeviceBuffer<u8>,
    /// Decode async-overlap router scratch.
    ///
    /// `packed_topk_device` is a `[max_top_k * 2]` u32 buffer that
    /// `router_softmax_topk_packed_device` fills with interleaved
    /// `(idx, bitcast<u32>(weight))` records. `packed_topk_pinned` is the
    /// pinned host destination of the single fused dtoh issued on the
    /// transfer stream; reads are gated by the pinned slice's internal event,
    /// which `as_slice()` synchronizes on. `event_topk_ready` is an event
    /// recorded on the compute stream after the packed top-k kernel; the
    /// transfer stream waits on it before launching the dtoh.
    ///
    /// `router_probs`/`router_indexed` remain only for prefill / tests; the
    /// decode hot path no longer touches them. `router_top_indices`/
    /// `router_top_weights` are the parsed host-side outputs of the dtoh
    /// reused across MoE layers.
    pub(super) packed_topk_device: DeviceBuffer<u32>,
    pub(super) packed_topk_pinned: PinnedHostSlice<u32>,
    pub(super) event_topk_ready: cudarc::driver::CudaEvent,
    pub(super) router_probs: Vec<f32>,
    pub(super) router_indexed: Vec<(usize, f32)>,
    pub(super) router_top_indices: Vec<usize>,
    pub(super) router_top_weights: Vec<f32>,
    /// Coalesced expert-weight staging for decode. After the router picks the
    /// top-k expert indices, the active experts' NVFP4 packed/scales bytes for
    /// ALL three projections (gate/up/down) are concatenated into these two
    /// contiguous VRAM buffers via back-to-back `copy_host_u8_to_device_at_offset_async`
    /// transfers on the transfer stream — ONE saturated H2D burst per MoE layer
    /// instead of `top_k × 3` tiny interleaved transfers via the staging pool.
    /// The per-expert GEMVs then read views into these buffers, producing
    /// bit-identical output to the per-expert staged path (same weights, same
    /// kernel, same order). `None` when no MoE layer has host-resident
    /// (StagedHostToDevice) experts (e.g. VRAM-resident expert cache) — in that
    /// case decode keeps the per-expert path (no H2D to coalesce).
    pub(super) bulk_expert_packed: Option<DeviceBuffer<u8>>,
    pub(super) bulk_expert_scales: Option<DeviceBuffer<u8>>,
    /// Transfer→compute fence: recorded on the transfer stream after a layer's
    /// bulk H2D burst; the compute stream waits on it before the expert GEMVs.
    pub(super) bulk_expert_event: cudarc::driver::CudaEvent,
    /// Compute→transfer fence (WAR hazard): recorded on the compute stream after
    /// a layer's expert GEMVs finish reading the bulk buffer; the NEXT layer's
    /// burst makes the transfer stream wait on it before overwriting the shared
    /// buffer. `bulk_expert_primed` guards the first-layer wait (the event has
    /// no recorded workload until the first GEMV pass completes).
    pub(super) bulk_expert_compute_event: cudarc::driver::CudaEvent,
    pub(super) bulk_expert_primed: bool,
    /// GPU-driven decode: per-slot NVFP4 input/output scales (`[top_k * 3]`,
    /// gate/up/down per slot) written by the gather kernel from the device
    /// scale tables, then read by the device-scalar quantize + GEMV kernels.
    /// Keeps the per-expert scales on-device so the launch sequence is fixed
    /// (graph-capturable). 1-element stubs when GPU-driven decode is not armed.
    pub(super) slot_in_scale: DeviceBuffer<f32>,
    pub(super) slot_out_scale: DeviceBuffer<f32>,
    /// Experts-on-CPU decode (config `hidden-layers.experts.compute = cpu`):
    /// reusable host buffers + the
    /// CPU kernel's per-layer scratch. `cpu_expert_input` holds the per-layer
    /// routed-expert input hidden downloaded from `hidden_out`; `cpu_routed_acc`
    /// holds the CPU kernel's `Σ_k w_k·expert_k` result before it is uploaded
    /// back into `routed_acc`. `cpu_moe_scratch` is the gate/up/swiglu/down
    /// slabs + flattened rayon job tables (zero per-token alloc once warmed).
    pub(super) cpu_expert_input: Vec<f32>,
    pub(super) cpu_routed_acc: Vec<f32>,
    pub(super) cpu_moe_scratch: aegisllm_cpu::MoeLayerScratch,
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
    /// PLE (Per-Layer Embeddings) apparatus, present only on Gemma-4 dense
    /// checkpoints with `hidden_size_per_layer_input` set (E4B, E2B). When
    /// `None`, every PLE step in the forward pass is a no-op.
    pub(super) ple: Option<PleGlobal>,
    /// True when any layer has host-resident (StagedHostToDevice) weights.
    /// Used to inhibit CUDA Graph capture (H2D copies cannot be in a captured graph).
    pub(super) has_staged_layers: bool,
    /// True when any layer has host-resident KV; inhibits CUDA Graph capture.
    pub(super) has_staged_kv: bool,
    /// True when the ONLY thing making layers "staged" is GPU-driven MoE experts
    /// (device-mapped-host gather, not host H2D), and KV is VRAM-resident. When
    /// set, the decode graph is allowed to capture even though `has_staged_layers`
    /// is true — the per-token control flow is a fixed kernel sequence reading
    /// the on-device top-k, so it replays correctly. See `forward.rs` gate.
    pub(super) moe_decode_gpu_driven_graphable: bool,
    /// Tail tier: KV store for layers >= `kv_first_n_layers` (or all layers when
    /// `kv_first_n_layers` is `None`).
    pub(super) kv_store: StoragePlacement,
    /// First-N count and tier. Layers `0..kv_first_n_layers` use `kv_first_store`.
    /// `kv_first_store=None` with `kv_first_n_layers=Some(_)` means "VRAM derived
    /// from compute" (legacy behavior preserved for the simple force-VRAM-first-N case).
    pub(super) kv_first_n_layers: Option<usize>,
    pub(super) kv_first_store: Option<StoragePlacement>,
    /// Storage dtype for the KV cache (f16/bf16/fp8). Per-layer is uniform.
    pub(super) kv_quantization: aegisllm_base::tensor::quant::KvCacheQuantization,
    /// `cuMemHostRegister` registrations on safetensors shard mmaps that
    /// hold host-resident weights. Kept alive for the executor's lifetime
    /// so per-token H2D streaming pulls directly from the registered
    /// pages (DMA fast path); on drop, every shard is unregistered before
    /// its mmap is unmapped.
    pub(super) registered_shards: crate::cuda::registered_shards::RegisteredShards,
    /// EAGLE/MTP speculative-decoding draft model. `Some` only when the engine
    /// was loaded with `--draft-model <path>` (or `AEGIS_DRAFT_MODEL`). When
    /// `None`, generation runs exactly as before (no spec-decode path is taken).
    pub(super) draft: Option<Box<DraftModel>>,
    /// Number of tokens the draft proposes per spec-decode round (default 4).
    pub(super) num_draft_tokens: usize,
    /// Qwen3.6 EAGLE/MTP speculative-decoding head (in-checkpoint, full-attn +
    /// MoE). `Some` only when the MTP head was attached (`AEGIS_MTP=1` and the
    /// checkpoint has `mtp.fc.weight`). Distinct from `draft` (the Gemma-4
    /// external draft model); only one of the two is ever set.
    pub(super) mtp: Option<Box<super::mtp::MtpHead>>,
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
    /// Dense MLP gate/up/down projections. `CudaLinear` enum so a single
    /// dense decoder can hold NVFP4 (NVIDIA prequantized checkpoints), BF16
    /// (vanilla HF releases like Gemma-4-E4B-it), or load-time-quantized FP8
    /// (`shared-MLP-quantization = "fp8"`). The variant is chosen by
    /// `load_cuda_linear` at load time based on whether `{prefix}.weight_scale`
    /// is present in the checkpoint and the `shared-MLP-quantization` config
    /// override.
    pub(super) gate_proj: CudaLinear,
    pub(super) up_proj: CudaLinear,
    pub(super) down_proj: CudaLinear,
    /// Dense MLP activation. Most models use SwiGLU (silu(gate) * up); Gemma-4
    /// (E4B and 26B-A4B) uses GeGLU-tanh (gelu_pytorch_tanh(gate) * up).
    /// NVFP4 dense MLPs ignore this — they fuse the activation into
    /// kernel-specific fast paths that always run SwiGLU; setting this to
    /// GeluTanh on an NVFP4 layer is a load-time error (unsupported combo).
    pub(super) dense_activation: super::mlp::DenseActivation,
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
    /// Per-layer PLE weights (E4B / E2B). `None` when the model has no PLE.
    pub(super) ple: Option<PleLayer>,
    /// KV-cache sharing (Gemma-4 E4B / E2B): when `Some(parent_idx)`, this
    /// layer's K/V projections + cache writes are skipped at runtime; the
    /// attention kernel reads K/V from `layers[parent_idx].kv` instead. The
    /// parent is selected at load time as the most recent layer of the same
    /// `layer_type` (sliding vs full) before the shared-layer boundary
    /// `num_hidden_layers - num_kv_shared_layers`. E4B: layers 24..41 are
    /// shared; layers 22 (sliding) and 23 (full) are the parents.
    pub(super) kv_shared_from: Option<usize>,
    /// Qwen3-Next Gated DeltaNet mixer weights. `Some` replaces the
    /// self-attention sublayer with the GDN recurrence; the MLP sublayer
    /// (dense or MoE) is unaffected. `None` for every standard attention layer.
    pub(super) gdn: Option<Box<super::gdn::CudaGdn>>,
    /// Qwen3-Next full-attention output gate: when true, `q_proj` outputs
    /// `[num_heads, 2*head_dim]` (query interleaved with a gate); the attention
    /// output is multiplied by `sigmoid(gate)` per head before `o_proj`.
    pub(super) attn_output_gate: bool,
}

/// EAGLE/MTP speculative-decoding draft model (Gemma-4 E4B-it-assistant).
///
/// The draft is a tiny (4-layer, hidden=256) Q-ONLY decoder that re-uses the
/// TARGET model's per-layer K/V cache: each draft layer computes only `q_proj`
/// + `q_norm` + RoPE and attends against a target layer's KV buffer (exactly
/// the `kv_shared_override` machinery the target's own shared layers use, but
/// cross-model). The draft input is
/// `concat(draft_embed(token), target_backbone_hidden)` → `pre_projection` →
/// 4 Q-only layers → `final_norm` → `post_projection` (back to backbone width,
/// fed forward as the next step's `target_backbone_hidden`) and, in parallel,
/// the centroid-masked sparse LM head produces the proposed token.
///
/// All weights are VRAM-resident (~135 MiB total) and the draft NEVER allocates
/// its own KV cache — it reads the target's.
#[derive(Debug)]
pub(super) struct DraftModel {
    /// `pre_projection.weight` — `[draft_hidden, 2*backbone_hidden]` BF16
    /// (checkpoint: `[256, 5120]` = 256 × (2 × 2560)). Input is
    /// `concat(token_embed[backbone_hidden], target_backbone_hidden[backbone_hidden])`
    /// → `draft_hidden`. NOTE: the token-embed half uses the TARGET model's
    /// backbone embedding (2560-wide), NOT the draft's own 256-wide
    /// `embed_tokens` (which is the tied sparse LM head). TODO(gpu-verify):
    /// confirm the token-embed source (target embed_tokens vs a draft-side
    /// 2560-wide embed) and the concat order (`[embed, hidden]` vs `[hidden,
    /// embed]`) against the HF `Gemma4AssistantModel` reference.
    pub(super) pre_projection: DeviceBf16Matrix,
    /// `post_projection.weight` — `[backbone_hidden, draft_hidden]` BF16.
    /// Projects the draft's post-`final_norm` hidden back to backbone width so
    /// it can seed the next draft step as `target_backbone_hidden`.
    pub(super) post_projection: DeviceBf16Matrix,
    /// `model.embed_tokens.weight` — `[vocab, draft_hidden]` BF16. Tied to the
    /// sparse LM head (the draft scores candidate token rows of THIS matrix).
    /// Used ONLY by the centroid sparse head, not by `pre_projection`.
    pub(super) embed_tokens: DeviceBf16Matrix,
    /// `model.norm.weight` — `[draft_hidden]` f32 RMSNorm.
    pub(super) final_norm: DeviceBuffer<f32>,
    /// 4 Q-only decoder layers. Built as full `CudaLayer`s (with stub k/v/o
    /// where unused) so the existing `forward_attention_device` + `forward_mlp_device`
    /// can run them unchanged when `kv_shared_override` is always `Some`.
    pub(super) layers: Vec<CudaLayer>,
    /// Per draft layer: index into the TARGET's `state.layers` whose KV cache
    /// this draft layer attends. Selected at load time as the most recent
    /// KV-owning target layer of matching attention type (sliding ↔ sliding,
    /// global ↔ global). TODO(gpu-verify): confirm cross-model parent choice.
    pub(super) target_kv_layer: Vec<usize>,
    /// Centroid-masked sparse LM head apparatus.
    pub(super) centroid_head: CentroidHead,
    pub(super) draft_hidden: usize,
    pub(super) backbone_hidden: usize,
    pub(super) num_attention_heads: usize,
    pub(super) rms_norm_eps: f32,
    pub(super) draft_hidden_for_scratch: usize,
    pub(super) intermediate_for_scratch: usize,
    pub(super) num_centroids: usize,
    pub(super) max_candidates: usize,
}

impl DraftModel {
    pub(super) fn draft_hidden(&self) -> usize { self.draft_hidden }
}

/// Centroid-masked sparse LM head. The full `[vocab, draft_hidden]` head is
/// never scored densely; the draft hidden is first scored against
/// `num_centroids` centroids, the top-`top_k` centroids are kept, and each
/// maps (via `token_ordering`) to a contiguous block of candidate token ids.
/// The dense head is then evaluated only over the gathered candidate rows.
#[derive(Debug)]
pub(super) struct CentroidHead {
    /// `masked_embedding.centroids.weight` — `[num_centroids, draft_hidden]` BF16.
    pub(super) centroids: DeviceBf16Matrix,
    /// `masked_embedding.token_ordering` — `[vocab]` i64 (loaded as u32). The
    /// permutation that groups the vocab by centroid: token ids sorted so each
    /// centroid owns a contiguous slice. TODO(gpu-verify): confirm the exact
    /// per-centroid slice boundaries (uniform vocab/num_centroids vs an offsets
    /// table) against the HF `Gemma4MaskedEmbedding` reference.
    pub(super) token_ordering: Vec<u32>,
    pub(super) num_centroids: usize,
    /// `centroid_intermediate_top_k` — number of centroids kept (=32).
    pub(super) top_k: usize,
    pub(super) vocab_size: usize,
}

/// Draft-only scratch buffers (all small — draft hidden is 256).
#[derive(Debug)]
pub(super) struct DraftScratch {
    /// `[2*backbone_hidden]` — the pre_projection input concat buffer
    /// `[token_embed(backbone), target_backbone_hidden(backbone)]`.
    pub(super) pre_proj_input: DeviceBuffer<f32>,
    /// `[backbone_hidden]` — target-embedding lookup of the current draft token
    /// (first half of the pre_projection input).
    pub(super) draft_embed: DeviceBuffer<f32>,
    /// `[draft_hidden]` — running draft hidden (post pre_projection / per layer).
    pub(super) hidden: DeviceBuffer<f32>,
    /// `[draft_hidden]` — final-normed draft hidden (input to head + post_projection).
    pub(super) final_hidden: DeviceBuffer<f32>,
    /// `[backbone_hidden]` — post_projection output (seeds next step's target hidden).
    pub(super) backbone_out: DeviceBuffer<f32>,
    /// `[num_centroids]` — centroid scores.
    pub(super) centroid_scores: DeviceBuffer<f32>,
    /// `[max_candidates]` — candidate token-id list (device).
    pub(super) candidate_rows: DeviceBuffer<u32>,
    /// `[max_candidates]` — candidate logits.
    pub(super) candidate_logits: DeviceBuffer<f32>,
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
    /// Stage I.2 image injection — VRAM-resident image-soft-token embeddings
    /// `[image_n_tokens, hidden_size]`. The prefill embed step overwrites
    /// every input position whose token id equals `image_token_id` with
    /// consecutive rows from this buffer.
    pub image_embeds: Option<DeviceBuffer<f32>>,
    pub image_token_id: u32,
    pub image_n_tokens: usize,
    /// Qwen3-VL M-RoPE: per-image pre-merge grid `(grid_t, grid_h, grid_w)` +
    /// `spatial_merge_size`. `Some` enables 3-component M-RoPE position ids in
    /// prefill (built from the prompt token sequence via `get_rope_index`).
    /// `None` → ordinary 1-D RoPE (Gemma / text-only), no behaviour change.
    pub mrope_grid: Option<(usize, usize, usize, usize)>,
    /// M-RoPE decode position delta = `max(position)+1 - seq_len`, set after
    /// the prompt is processed. Decode positions = `seq_len + delta + step` in
    /// all 3 axes (they re-align post-image), so decode stays on the 1-D path
    /// with this shifted scalar position.
    pub mrope_decode_delta: i64,
    /// Full prompt M-RoPE position ids `[3][seq]` (T,H,W), computed once at the
    /// start of prefill from the prompt token sequence + `mrope_grid`. Per-chunk
    /// the relevant slice is uploaded to the prefill scratch buffers. Empty when
    /// no image / non-M-RoPE model.
    pub mrope_positions: Option<[Vec<u32>; 3]>,
    /// Audio soft-token embeddings `[audio_n_tokens, hidden_size]` (VRAM).
    /// The prefill embed step overwrites every input position whose token id
    /// equals `audio_token_id` with consecutive rows from this buffer. Mirrors
    /// the image-injection path.
    pub audio_embeds: Option<DeviceBuffer<f32>>,
    pub audio_token_id: u32,
    pub audio_n_tokens: usize,
    /// Device buffers holding the current decode `position` and `seq_len` (= position + 1).
    /// Kept here (not in CudaScratch) so we can borrow them alongside `&mut scratch`.
    /// Updated before each decode step (outside graph capture), read by the ptr-based kernels.
    pub(super) decode_position: DeviceBuffer<u32>,
    pub(super) decode_seq_len: DeviceBuffer<u32>,
    /// Captured CUDA Graph for decode steps. Set after the first decode step.
    /// When set, each subsequent step updates decode_position/decode_seq_len then replays
    /// this graph instead of issuing ~645 kernel launches.
    pub(super) decode_graph: Option<SendCudaGraph>,
    /// Speculative-decoding draft scratch. `Some` only when the executor has a
    /// draft model attached and this state was allocated for spec-decode. Holds
    /// the draft's per-step buffers + the draft decoder `CudaScratch` (sized to
    /// the draft's small 256-wide layers). All mutation of the draft happens
    /// through the STATE (mirrors how the target's scratch lives on the state),
    /// so the executor's `&self` weight references stay immutable.
    pub(super) draft: Option<Box<DraftState>>,
    /// Per-sequence MTP head state (KV + scratch). `Some` only when an MTP head
    /// is attached and this state was allocated for spec-decode.
    pub(super) mtp: Option<Box<super::mtp::MtpState>>,
}

/// Per-sequence speculative-decoding scratch (lives on `CudaLlamaState`).
#[derive(Debug)]
pub(super) struct DraftState {
    /// Small per-step buffers (embed concat, centroid scores, candidate lists).
    pub(super) scratch: DraftScratch,
    /// Full `CudaScratch` sized to the draft's small decoder widths, reused by
    /// `forward_attention_device` / `forward_mlp_device` for the draft's 4
    /// Q-only layers. Separate from the target's scratch because the draft's
    /// RMSNorm kernels require exact-length (256-wide) buffers.
    pub(super) decoder_scratch: CudaScratch,
    /// Throwaway `CudaLayerState` handed to `forward_attention_device` as its
    /// `&mut layer_state` argument. The draft always passes
    /// `kv_shared_override = Some(target_kv)`, and the override branch NEVER
    /// reads or writes `layer_state.kv` — so this exists purely to satisfy the
    /// function signature without aliasing the target's KV buffers. Its KV is a
    /// 1-element stub.
    pub(super) dummy_layer_state: CudaLayerState,
}

#[derive(Debug)]
pub(super) struct CudaLayerState {
    pub(super) kv: CudaKvCache,
    /// Gated DeltaNet recurrent state `[n_v, d_v, d_k]` f32 (Qwen3-Next GDN
    /// layers). Persists across the whole sequence; `None` for attention layers.
    pub(super) recurrent: Option<DeviceBuffer<f32>>,
    /// GDN depthwise-conv rolling state `[conv_channels, kernel-1]` f32.
    pub(super) conv_state: Option<DeviceBuffer<f32>>,
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
    /// Storage dtype: F16/BF16 → u16 buffer (2 bytes/elem), FP8 → u8 buffer
    /// (1 byte/elem). The `keys`/`values` enums hold the actual VRAM
    /// allocation; downstream kernel dispatch matches on this to pick the
    /// right per-token store + attention kernel pair.
    pub(super) quantization: aegisllm_base::tensor::quant::KvCacheQuantization,
    /// VRAM: full-sized for Dense, 1-element stub for HostResident.
    pub(super) keys: KvBuffer,
    /// VRAM: full-sized for Dense, 1-element stub for HostResident.
    pub(super) values: KvBuffer,
    /// Non-None when `kv-cache.store=ram`: actual KV lives in pinned host RAM.
    pub(super) host: Option<Box<HostKvWeights>>,
    /// Auxiliary f16 KV scratch used only when `quantization == Fp8`. Prefill
    /// attention kernels read f16 cache lines, so we maintain a parallel f16
    /// buffer that mirrors the FP8 cache during prefill. After prefill ends,
    /// decode reads the FP8 buffer exclusively. The f16 scratch is allocated
    /// at the same effective capacity as the FP8 cache.
    pub(super) prefill_f16_keys: Option<DeviceBuffer<u16>>,
    pub(super) prefill_f16_values: Option<DeviceBuffer<u16>>,
}

/// VRAM-resident KV-cache half (keys or values). The bit-width matches the
/// configured `KvCacheQuantization` of the parent `CudaKvCache`.
#[derive(Debug)]
pub(super) enum KvBuffer {
    /// F16 / BF16 — 2 bytes per element. Same wire format as our existing
    /// `kv_store_*` and `attention_decode_*` kernels.
    F16(DeviceBuffer<u16>),
    /// FP8 (E4M3) — 1 byte per element. Dispatches to the `*_fp8_*`
    /// runtime methods.
    Fp8(DeviceBuffer<u8>),
}

impl KvBuffer {
    pub(super) fn as_f16(&self) -> Option<&DeviceBuffer<u16>> {
        match self { Self::F16(b) => Some(b), _ => None }
    }
    pub(super) fn as_f16_mut(&mut self) -> Option<&mut DeviceBuffer<u16>> {
        match self { Self::F16(b) => Some(b), _ => None }
    }
    pub(super) fn as_fp8(&self) -> Option<&DeviceBuffer<u8>> {
        match self { Self::Fp8(b) => Some(b), _ => None }
    }
    pub(super) fn as_fp8_mut(&mut self) -> Option<&mut DeviceBuffer<u8>> {
        match self { Self::Fp8(b) => Some(b), _ => None }
    }
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
        quantization: aegisllm_base::tensor::quant::KvCacheQuantization,
        effective_capacity: usize,
        is_sliding: bool,
    ) -> aegisllm_base::error::Result<Self> {
        use aegisllm_base::error::AegisError;
        use aegisllm_base::tensor::quant::KvCacheQuantization;
        // `effective_capacity` is the number of token slots actually
        // allocated. Sliding-window layers pass `window_size` (cache wraps
        // every `window_size` tokens via `slot = pos % window_size`).
        // Global / full-attention layers pass `context_size`.
        let len = effective_capacity.checked_mul(kv_width).ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "CUDA dense KV cache length overflow: cap={} kv_width={}",
                effective_capacity, kv_width
            ))
        })?;
        let (keys, values, prefill_f16_keys, prefill_f16_values) = match quantization {
            KvCacheQuantization::F16 | KvCacheQuantization::Bf16 => (
                KvBuffer::F16(runtime.alloc_u16(len)?),
                KvBuffer::F16(runtime.alloc_u16(len)?),
                None,
                None,
            ),
            KvCacheQuantization::Fp8 => {
                // Auxiliary f16 cache for prefill attention. Kept ONLY for
                // SLIDING (windowed) layers, where `effective_capacity` is the
                // small window (~1024 slots) so the 2 B/elem cost is tiny and
                // the non-FP8-fast-path compat attention kernel still reads it.
                //
                // GLOBAL (full-attention, head_dim=512) layers have
                // `effective_capacity == context_size` (262144 at long ctx) ->
                // a full f16 KV cache on top of the FP8 one, the source of the
                // 262144 OOM. Stage C.1 routes EVERY global prefill chunk
                // through an FP8-KV-reading kernel (the FP8-MMA kernel under
                // fp8 compute, the option-b dequant kernel under bf16 compute),
                // so the global aux is never read -> not allocated here.
                let (aux_k, aux_v) = if is_sliding {
                    (
                        Some(runtime.alloc_u16(len)?),
                        Some(runtime.alloc_u16(len)?),
                    )
                } else {
                    (None, None)
                };
                (
                    KvBuffer::Fp8(runtime.alloc_u8(len)?),
                    KvBuffer::Fp8(runtime.alloc_u8(len)?),
                    aux_k,
                    aux_v,
                )
            }
            other => return Err(AegisError::Unsupported(format!(
                "kv-cache quantization {other:?} not yet wired into CUDA executor; supported: f16, bf16, fp8"
            ))),
        };
        Ok(Self {
            layout: CudaKvCacheLayout::Dense {
                context_size,
                kv_width,
            },
            quantization,
            keys,
            values,
            host: None,
            prefill_f16_keys,
            prefill_f16_values,
        })
    }

    /// Allocate host-resident KV: full-size pinned RAM + 1-element VRAM stubs.
    /// The shared `KvStagingSlot` in `CudaScratch` is used at inference time.
    pub(super) fn staged_host(
        runtime: &CudaRuntime,
        context_size: usize,
        kv_width: usize,
        quantization: aegisllm_base::tensor::quant::KvCacheQuantization,
    ) -> aegisllm_base::error::Result<Self> {
        use aegisllm_base::error::AegisError;
        use aegisllm_base::tensor::quant::KvCacheQuantization;
        if !matches!(
            quantization,
            KvCacheQuantization::F16 | KvCacheQuantization::Bf16
        ) {
            return Err(AegisError::Unsupported(format!(
                "host-resident KV cache only supports f16/bf16 today, got {quantization:?}"
            )));
        }
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
            quantization,
            keys: KvBuffer::F16(runtime.alloc_u16(1)?),
            values: KvBuffer::F16(runtime.alloc_u16(1)?),
            host: Some(Box::new(HostKvWeights {
                keys: keys_host,
                values: values_host,
            })),
            prefill_f16_keys: None,
            prefill_f16_values: None,
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
    /// Allocated only when the model has Gated DeltaNet (linear-attention)
    /// layers. Persistent per-decode-step scratch so the GDN mixer reuses one
    /// buffer set across all GDN layers + tokens instead of ~16 fresh
    /// `alloc_f32`s per layer per token.
    pub(super) gdn_decode: Option<Box<super::gdn::GdnDecodeScratch>>,
    /// Allocated only when the model has any host-resident (StagedHostToDevice) linears.
    /// Sized to hold the largest staged layer's packed + scales in one slot.
    pub(super) staging_pool: Option<Box<LinearStagingPool>>,
    /// Allocated only when any layer has host-resident KV. Two VRAM slots sized
    /// `context_size × kv_width` each, used in ping-pong to overlap async H2D
    /// (next layer) with compute (current layer) via the dedicated transfer stream.
    pub(super) kv_staging: Option<Box<KvStagingPool>>,
    /// PLE per-layer feed: `[num_layers, ple_dim]` f32. Computed once per
    /// decode step (at token entry) by combining `embed_tokens_per_layer`
    /// lookup with the `per_layer_model_projection` of the current hidden
    /// state, then consumed inside each layer's MLP forward to produce the
    /// per-layer additive contribution. Sized 1 when the model has no PLE.
    pub(super) per_layer_inputs: DeviceBuffer<f32>,
    /// PLE projection scratch `[num_layers * ple_dim]` — output of the BF16
    /// `hidden @ per_layer_model_projection.T` GEMM before RMSNorm/combine.
    pub(super) ple_projection: DeviceBuffer<f32>,
    /// PLE projection normed scratch `[num_layers * ple_dim]` — after RMSNorm.
    pub(super) ple_projection_normed: DeviceBuffer<f32>,
    /// PLE gate-projection output `[ple_dim]` — per-layer scratch consumed
    /// inside the decoder block's PLE additive contribution.
    pub(super) ple_gate: DeviceBuffer<f32>,
    /// PLE projection output `[hidden]` — per-layer scratch.
    pub(super) ple_contrib: DeviceBuffer<f32>,
    /// PLE projection output `[hidden]` after RMSNorm.
    pub(super) ple_contrib_normed: DeviceBuffer<f32>,
    /// PLE BF16 staging for the lookup row + projection input.
    pub(super) ple_bf16_in: DeviceBuffer<u16>,
    /// PLE BF16 GEMM output staging.
    pub(super) ple_bf16_out: DeviceBuffer<u16>,
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
    /// Qwen3-VL M-RoPE 3-component position buffers (T,H,W), each sized like
    /// `positions`. Populated only when the prompt carries an image and the
    /// model uses M-RoPE; the prefill RoPE then reads these instead of the
    /// 1-D `positions`. `mrope_active` gates the swap so text-only/non-Qwen
    /// prefill is byte-for-byte unchanged.
    pub(super) mrope_pos_t: DeviceBuffer<u32>,
    pub(super) mrope_pos_h: DeviceBuffer<u32>,
    pub(super) mrope_pos_w: DeviceBuffer<u32>,
    pub(super) mrope_active: bool,
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
    /// BF16 scratch for FP8-weight dequant. Sized for the largest projection:
    /// `max(hidden*hidden, hidden*intermediate, hidden*q_width, hidden*kv_width)`
    /// elements. Reused per-call across all FP8 prefill GEMMs (one shared
    /// scratch, not per-layer). Empty (zero-length) when no FP8 weights are
    /// loaded — initialized lazily by the FP8 prefill path.
    pub(super) fp8_dequant_scratch: DeviceBuffer<u16>,
    /// MoE prefill scratch (allocated only when the model has MoE layers).
    pub(super) moe: Option<Box<CudaMoEPrefillScratch>>,
    /// Throwaway f16 KV target for GLOBAL (head_dim=512) layers under FP8 KV.
    /// Stage C drops the full-context `prefill_f16_keys/values` aux for global
    /// layers, but the proven `store_kv_slots_batched_rope_key_device` kernel
    /// applies RoPE in-place to the K tile AND writes an f16 cache line in one
    /// pass. We keep using it for the in-place RoPE and redirect its f16 cache
    /// writes here — a `chunk_size * kv_width` buffer (one chunk, not the full
    /// 262144-token context). The kernel's `slot % cache_capacity` wrap keeps
    /// every write in-bounds; the contents are never read (the FP8 mirror
    /// store + the FP8-direct attention kernels carry the real K/V). Sized
    /// `chunk_size * max_kv_width` each; zero-length (stub) for non-FP8 configs.
    pub(super) prefill_global_kv_f16_scratch_k: DeviceBuffer<u16>,
    pub(super) prefill_global_kv_f16_scratch_v: DeviceBuffer<u16>,
    /// PLE per-token-per-layer feed for the chunked prefill pass.
    /// `[chunk_size, num_layers, ple_dim]` row-major f32. Computed once
    /// per chunk in `prefill/mod.rs::prefill_prompt_chunked` (after embed
    /// scale + image injection) by `compute_per_layer_inputs_prefill_chunk`,
    /// consumed inside each prefill layer's MLP forward by
    /// `apply_ple_contribution_prefill_chunk`. Sized 1 for non-PLE models.
    pub(super) per_layer_inputs: DeviceBuffer<f32>,
    /// PLE projection scratch — `[chunk_size, num_layers * ple_dim]` f32.
    pub(super) ple_projection: DeviceBuffer<f32>,
    /// PLE projection-normed scratch — same size.
    pub(super) ple_projection_normed: DeviceBuffer<f32>,
    /// PLE per-layer gate `[chunk_size, ple_dim]` f32.
    pub(super) ple_gate: DeviceBuffer<f32>,
    /// PLE per-layer contrib `[chunk_size, hidden]` f32.
    pub(super) ple_contrib: DeviceBuffer<f32>,
    /// PLE per-layer contrib-normed `[chunk_size, hidden]` f32.
    pub(super) ple_contrib_normed: DeviceBuffer<f32>,
    /// PLE BF16 staging — sized for the larger of {chunk × num_layers × ple_dim}
    /// (lookup row gather) and {chunk × hidden} (cuBLASLt input/output).
    pub(super) ple_bf16_in: DeviceBuffer<u16>,
    pub(super) ple_bf16_out: DeviceBuffer<u16>,
    // ── Native FP8 block-scaled prefill GEMM scratch (Qwen3.5 FP8) ──────────
    /// Per-(token,128-K-group) e4m3 quantized activation, `[chunk * max_fp8_K]`
    /// u8. Shared by every `matmul_fp8_block_native_batched` call in the
    /// chunked prefill (GDN in/out_proj, full-attn q/k/v/o, dense gate/up/down).
    /// Zero-length stub for non-FP8-block configs.
    pub(super) fp8_a_q: DeviceBuffer<u8>,
    /// Per-(token,128-K-group) activation scale, `[chunk * ceil(max_fp8_K/128)]` f32.
    pub(super) fp8_a_scale: DeviceBuffer<f32>,
    // ── Batched GDN chunked-prefill scratch (Qwen3-Next linear_attention) ───
    /// `[chunk * conv_channels]` — in_proj_qkv output / conv1d output (split into q/k/v).
    pub(super) gdn_qkv: DeviceBuffer<f32>,
    /// `[chunk * conv_channels]` — conv1d output (SiLU'd), before the q/k/v split.
    pub(super) gdn_conv_out: DeviceBuffer<f32>,
    /// `[chunk * v_width]` — in_proj_z output (gate stream for the gated RMSNorm).
    pub(super) gdn_z: DeviceBuffer<f32>,
    /// `[chunk * n_v]` — in_proj_b output.
    pub(super) gdn_b: DeviceBuffer<f32>,
    /// `[chunk * n_v]` — in_proj_a output.
    pub(super) gdn_a: DeviceBuffer<f32>,
    /// `[chunk * qk_width]` — q split out of conv output (pre-norm).
    pub(super) gdn_q_raw: DeviceBuffer<f32>,
    /// `[chunk * qk_width]` — k split out of conv output (pre-norm).
    pub(super) gdn_k_raw: DeviceBuffer<f32>,
    /// `[chunk * v_width]` — v split out of conv output.
    pub(super) gdn_v: DeviceBuffer<f32>,
    /// `[chunk * n_v * d_k]` — normed + GQA-expanded q.
    pub(super) gdn_q_n: DeviceBuffer<f32>,
    /// `[chunk * n_v * d_k]` — normed + GQA-expanded k.
    pub(super) gdn_k_n: DeviceBuffer<f32>,
    /// `[chunk * n_v]` — beta = sigmoid(b).
    pub(super) gdn_beta: DeviceBuffer<f32>,
    /// `[chunk * n_v]` — g = -exp(a_log)*softplus(a+dt_bias).
    pub(super) gdn_g: DeviceBuffer<f32>,
    /// `[chunk * v_width]` — delta-rule output o.
    pub(super) gdn_o: DeviceBuffer<f32>,
    /// `[chunk * v_width]` — gated-RMSNorm(o, z) → out_proj input.
    pub(super) gdn_o_norm: DeviceBuffer<f32>,
    /// `[chunk * hidden]` — out_proj output (mixer_out), added to the residual.
    pub(super) gdn_mixer_out: DeviceBuffer<f32>,
    // ── Qwen3-Next full-attention prefill scratch (gated q output) ──────────
    /// `[chunk * 2 * q_width]` — gated q_proj output (query interleaved with gate).
    pub(super) attn_q_full: DeviceBuffer<f32>,
    /// `[chunk * q_width]` — de-interleaved attention output gate (sigmoid-mul'd in).
    pub(super) attn_gate: DeviceBuffer<f32>,
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
    /// Output buffer for the fused shared-MLP gate+up GEMM:
    /// `[chunk_size, 2 * max_shared_intermediate]` row-major. Per token the
    /// first `intermediate` floats are gate logits, the next `intermediate`
    /// are up logits. Consumed by `geglu_tanh_strided_device`. Only the
    /// `cs * 2 * shared_intermediate` prefix is written/read on each call.
    /// Sized to `cs * 2 * max_expert_intermediate` (an upper bound across
    /// MoE layers; shared intermediate is usually the larger of the two).
    pub(super) gather_shared_gate_up_fused: DeviceBuffer<f32>,
    /// Qwen3-Next shared-expert gate logits, `[chunk_size]`. Per-token logit
    /// produced by the `shared_gate` `[1, hidden]` batched matvec; the shared
    /// MLP output rows are scaled by `sigmoid(logit[token])` before being added
    /// to the routed experts (mirrors `CudaMoEScratch.shared_gate_logit` for
    /// decode but holds one logit per chunk token, not a single scalar).
    /// `None`/unused for models without a shared-expert gate (Gemma).
    pub(super) shared_gate_logit: DeviceBuffer<f32>,
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
    // ── Permuted MoE scratch (grouped path) ────────────────────────────────
    /// Per-expert prefix-sum offsets: `[num_experts + 1]`. Built by
    /// `aegis_router_expert_offsets` from `expert_counts`. Defines slice
    /// boundaries in the permuted-activation buffers.
    pub(super) expert_offsets: DeviceBuffer<u32>,
    /// Permuted input: `[chunk_size * top_k, hidden_size]`. Filled by
    /// `aegis_permute_gather_f32`: tokens grouped by expert, in order of
    /// `expert_offsets`. Each routed-expert call reads its slice
    /// `[expert_offsets[e]..expert_offsets[e+1])` directly — no per-call
    /// `gather_rows`.
    pub(super) permuted_input: DeviceBuffer<f32>,
    /// Permuted gate output: `[chunk_size * top_k, expert_intermediate]`.
    pub(super) permuted_intermediate: DeviceBuffer<f32>,
    /// Permuted up output (also reused as GeGLU output for down input).
    pub(super) permuted_swiglu: DeviceBuffer<f32>,
    /// Permuted down_proj output: `[chunk_size * top_k, hidden_size]`. Read
    /// by the deterministic unpermute-scatter to write back into `moe_acc`.
    pub(super) permuted_output: DeviceBuffer<f32>,
    // ── Deterministic unpermute-scatter inverse index ──────────────────────
    /// Per-token inverse routing table, `[chunk_size * top_k]`. Slot
    /// `token*top_k + k` holds the permuted source row of the expert that is
    /// `token`'s k-th (by ascending expert id) route. Built by
    /// `aegis_router_build_unpermute_index`, consumed by
    /// `aegis_unpermute_scatter_serial_f32`. Replaces the nondeterministic
    /// `atomicAdd`-based scatter so greedy decode is bit-reproducible.
    pub(super) unpermute_rows: DeviceBuffer<u32>,
    /// Parallel to `unpermute_rows`: `bitcast<u32>(routing weight)` per slot.
    pub(super) unpermute_wbits: DeviceBuffer<u32>,
    /// Per-token count of routed experts, `[chunk_size]`. Zeroed before each
    /// `build_unpermute_index` launch so multiple calls (CUTLASS split path)
    /// do not mix subsets.
    pub(super) unpermute_count: DeviceBuffer<u32>,
    // ── Bulk expert weight staging (grouped GEMM path) ─────────────────────
    /// Three projections per layer (gate / up / down) need three independent
    /// staging slots so the transfer stream can stage projection N+1 while
    /// the compute stream's grouped-GEMM kernel for projection N is still
    /// reading from its own slot. Each slot is sized for the worst-case
    /// (all 128 experts active) projection footprint (~127 MiB packed +
    /// ~16 MiB scales + tiny metadata). 3-slot total: ~430 MiB transient
    /// VRAM. Required for Phase B.3 (dual-stream H2D/compute overlap).
    pub(super) bulk_slots: [GroupedStagingSlot; 3],
    /// Per-active-expert prefix-sum of token counts: `expert_token_offsets[ae+1]
    /// - expert_token_offsets[ae]` is the number of tokens routed to the
    /// ae-th active expert. Length = `num_experts + 1`. Independent of
    /// projection so a single buffer is fine — written once per layer
    /// before any GEMM and not modified between projections.
    pub(super) bulk_token_offsets: DeviceBuffer<u32>,
    /// CUTLASS NVFP4 grouped GEMM scratch. `None` unless the build was
    /// compiled with `AEGIS_CUTLASS_NVFP4_GROUPED_BUILD=1` AND the runtime
    /// flag `AEGIS_CUTLASS_NVFP4_GROUPED=1` was set at executor init.
    /// Holds per-expert quantized-input + swizzled-weight-scale buffers,
    /// per-group metadata blobs (strides, layouts, problem shapes), and
    /// the pointer arrays the CUTLASS kernel expects. Sized for the
    /// worst-case (all routed experts active, max chunk_size*top_k tokens).
    pub(super) cutlass: Option<Box<CutlassMoeScratch>>,
}

/// Per-projection CUTLASS staging slot (gate / up / down each get one
/// so the swizzled weight scales for projection N+1 can be computed
/// while CUTLASS for projection N is still reading from its own slot).
#[derive(Debug)]
pub(super) struct CutlassMoeProjSlot {
    /// Swizzled weight scales — sized for `max_experts * sfb_per_group`
    /// at the worst-case (n, k) for this projection (gate/up = (intermediate, hidden),
    /// down = (hidden, intermediate)).
    pub(super) weight_sfb: DeviceBuffer<u8>,
    /// Per-active-expert source offsets into `bulk_scales` (input to swizzle).
    pub(super) src_offsets: DeviceBuffer<u64>,
    /// Per-active-expert destination offsets into `weight_sfb`.
    pub(super) dst_offsets: DeviceBuffer<u64>,
    /// Per-active-expert SFB pointers passed to CUTLASS (device array of u64).
    pub(super) sfb_ptrs: DeviceBuffer<u64>,
    /// Per-active-expert A/B/D pointers (one per group). A/B point to
    /// quantized activations and bulk_packed; D points into permuted output.
    pub(super) a_ptrs: DeviceBuffer<u64>,
    pub(super) b_ptrs: DeviceBuffer<u64>,
    pub(super) d_ptrs: DeviceBuffer<u64>,
    /// Per-active-expert SFA pointers (shared across gate/up since SFA depends
    /// only on K = hidden, but down has K = intermediate so SFA differs;
    /// keeping per-slot for simplicity).
    pub(super) sfa_ptrs: DeviceBuffer<u64>,
    /// Per-active-expert pointers into the per-group alpha f32 array.
    pub(super) alpha_ptrs: DeviceBuffer<u64>,
    /// Workspace for CUTLASS (size depends on problem shapes; sized for worst case).
    pub(super) workspace: DeviceBuffer<u8>,
}

/// CUTLASS NVFP4 grouped GEMM scratch shared across all three projections.
#[derive(Debug)]
pub(super) struct CutlassMoeScratch {
    /// Quantized A bytes per group concatenated. Sized for hidden K = max(K_gate_up, K_down).
    /// We use TWO buffers: one sized for K=hidden (gate/up activations),
    /// one for K=intermediate (down activations) — different K means
    /// different per-row byte width.
    pub(super) input_packed_hidden: DeviceBuffer<u8>,
    pub(super) input_sfa_hidden: DeviceBuffer<u8>,
    pub(super) input_packed_intermediate: DeviceBuffer<u8>,
    pub(super) input_sfa_intermediate: DeviceBuffer<u8>,
    /// Per-group offsets (u64 array) for `input_packed_*` / `input_sfa_*`.
    /// Built per-call from the sorted active_experts slice (large-only).
    pub(super) payload_offsets: DeviceBuffer<u64>,
    pub(super) sfa_offsets: DeviceBuffer<u64>,
    /// Per-projection slots.
    pub(super) slots: [CutlassMoeProjSlot; 3],
    /// Per-group stride blobs (sized at runtime via blob_sizes query).
    pub(super) stride_a: DeviceBuffer<u8>,
    pub(super) stride_b: DeviceBuffer<u8>,
    pub(super) stride_d: DeviceBuffer<u8>,
    pub(super) layout_sfa: DeviceBuffer<u8>,
    pub(super) layout_sfb: DeviceBuffer<u8>,
    /// Per-group problem-shape blob (3×i32 + padding).
    pub(super) problem_shapes: DeviceBuffer<u8>,
    /// Per-group alpha values (f32, one per active expert) — uploaded per call.
    pub(super) alpha_values: DeviceBuffer<f32>,
    /// CUTLASS blob sizes (queried once at construction).
    pub(super) blob_stride_a: usize,
    pub(super) blob_stride_b: usize,
    pub(super) blob_stride_d: usize,
    pub(super) blob_layout_sfa: usize,
    pub(super) blob_layout_sfb: usize,
    pub(super) blob_problem_shape: usize,
    /// Active-expert token-count prefix-sum buffer (u32, num_experts+1), used
    /// by the per-group quantize-input call. Reused per projection.
    pub(super) token_offsets: DeviceBuffer<u32>,
    /// Single-entry u64 scratch for per-group quantize_input calls (avoids
    /// per-iter device allocations).
    pub(super) quant_payload_off_scratch: DeviceBuffer<u64>,
    pub(super) quant_sfa_off_scratch: DeviceBuffer<u64>,
}

/// One physical staging slot for a single MoE projection (gate / up /
/// down). Holds the packed/scales bytes plus per-active-expert metadata
/// so the grouped-GEMM kernel can dispatch the projection independently
/// of the other two slots. The kernel reads `*_offsets` and
/// `output_scales` at execution time, so each projection needs its own
/// metadata to avoid races with the next projection's H2D into the
/// shared bulk buffer.
#[derive(Debug)]
pub(super) struct GroupedStagingSlot {
    pub(super) bulk_packed: DeviceBuffer<u8>,
    pub(super) bulk_scales: DeviceBuffer<u8>,
    pub(super) bulk_packed_offsets: DeviceBuffer<u32>,
    pub(super) bulk_scales_offsets: DeviceBuffer<u32>,
    pub(super) bulk_output_scales: DeviceBuffer<f32>,
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

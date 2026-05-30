use std::collections::{BTreeMap, BTreeSet};

use super::loader::{CudaLayerShape, load_cuda_layer, load_ple_global, runtime_layouts_by_region};
use super::state::{CudaLayer, CudaLayerState, CudaScratch, PleGlobal};
use aegisllm_base::artifact::ModelArtifact;
use crate::cuda::{DECODE_SPLIT_K_MAX, CudaRuntime, CudaRuntimeConfig, DeviceBuffer};
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::graph::{ModelGraph, RegionId};
use aegisllm_base::model::AttentionPattern;
use aegisllm_base::planning::placement::{ComputePlacement, ResolvedPlacement, StoragePlacement};
use aegisllm_base::planning::runtime::RuntimePlan;
use aegisllm_base::tensor::layout::LinearResidentLayout;
use aegisllm_base::tensor::storage::TensorStorageLoader;

#[derive(Debug)]
#[allow(dead_code)]
pub struct CudaLayerBlockExecutor {
    pub(super) runtime: CudaRuntime,
    pub(super) hidden_size: usize,
    pub(super) num_attention_heads: usize,
    pub(super) num_kv_heads: usize,
    pub(super) head_dim: usize,
    pub(super) rms_norm_eps: f32,
    pub(super) layers: BTreeMap<usize, CudaLayer>,
    pub(super) kv_context_size: usize,
    /// KV-cache storage dtype (mirrors the full executor). The hybrid Gemma-4
    /// path allocates per-layer KV with this quantization.
    pub(super) kv_quantization: aegisllm_base::tensor::quant::KvCacheQuantization,
    /// Total decoder layer count of the model (NOT just the selected subset).
    /// Needed to size the shared PLE feed `[num_layers * ple_dim]`.
    pub(super) total_num_layers: usize,
    /// Gemma-4 E4B/E2B PLE global apparatus, loaded when the model has
    /// `hidden_size_per_layer_input`. `None` for non-PLE models (Llama, Qwen,
    /// dense Gemma without PLE). When present, the hybrid uploads the shared
    /// `per_layer_inputs` (computed once on CPU) and each GPU layer applies the
    /// per-layer PLE additive on-device via the full-model `forward_mlp_device`.
    pub(super) ple: Option<PleGlobal>,
    /// Multiplicative embed scale (Gemma-4 = sqrt(hidden)); carried for parity
    /// with the full executor (the hybrid does the embed on CPU, so this is
    /// informational, but kept so a future GPU-side embed path can reuse it).
    pub(super) embed_scale: Option<f32>,
    /// Pinned host arena backing any host-resident weights this executor loads
    /// (the Gemma-4 PLE table + any store=ram GPU layer). Held so the arena —
    /// and its `cuMemHostRegister` pinning — outlives the weights that DMA from it.
    _host_arena: std::sync::Arc<crate::cuda::host_arena::PinnedArena>,
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct CudaLayerBlockState {
    pub(super) hidden: DeviceBuffer<f32>,
    pub(super) layers: BTreeMap<usize, CudaLayerState>,
    pub(super) scratch: CudaScratch,
    /// Reusable 1-element u32 device buffers for the per-layer
    /// `position` / `seq_len` kernel args. Pre-allocated once at state
    /// construction and overwritten per `forward_layer_device` call,
    /// instead of `alloc_u32(1)` × 2 fresh allocations per layer per
    /// token (each round-trips through the cudaMallocAsync pool).
    pub(super) p_position: DeviceBuffer<u32>,
    pub(super) p_seq_len: DeviceBuffer<u32>,
}

impl CudaLayerBlockExecutor {
    #[allow(dead_code)]
    pub fn from_artifact(
        artifact: &ModelArtifact,
        graph: &ModelGraph,
        placement: &ResolvedPlacement,
        runtime: &RuntimePlan,
        device: usize,
        cuda_config: CudaRuntimeConfig,
        selected_layers: &BTreeSet<usize>,
    ) -> Result<Self> {
        let cuda = CudaRuntime::new_with_config(device, cuda_config)?;
        let region_placements = placement.region_map();
        // Host arena for any host-resident weights THIS CUDA executor loads: the
        // always-host-resident Gemma-4 PLE table + any selected GPU layer with
        // store=ram. Sized only for this executor's regions (the selected GPU
        // layers) — the CPU layers and CPU bookends (embed/final_norm/lm_head)
        // are loaded by the CPU executor, not here. Without the arena, the
        // host-resident BF16 PLE-table load errors in the weight loader (this was
        // the P3 hybrid blocker).
        let arena_regions: std::collections::BTreeMap<
            &RegionId,
            &aegisllm_base::planning::placement::RegionPlacement,
        > = region_placements
            .iter()
            .filter(|(id, _)| selected_layers.iter().any(|&l| id.0 == format!("layer.{l}")))
            .map(|(id, pl)| (*id, *pl))
            .collect();
        let host_arena_capacity =
            super::full::compute_host_arena_capacity(artifact, graph, &arena_regions);
        let host_arena = std::sync::Arc::new(
            crate::cuda::host_arena::PinnedArena::new(&cuda, host_arena_capacity)?,
        );
        let cuda_weights = cuda.weight_loader_with_arena(host_arena.clone());
        let runtime_layouts = runtime_layouts_by_region(runtime);
        let mut loader = TensorStorageLoader::new();
        let mut layers = BTreeMap::new();

        for &layer in selected_layers {
            let region_id = RegionId(format!("layer.{layer}"));
            let region = graph
                .regions
                .iter()
                .find(|region| region.id == region_id)
                .ok_or_else(|| {
                    AegisError::InvalidPlan(format!("missing graph region `{}`", region_id.0))
                })?;
            let placement = region_placements.get(&region_id).ok_or_else(|| {
                AegisError::InvalidPlan(format!("missing placement for `{}`", region_id.0))
            })?;
            match placement.compute {
                ComputePlacement::Cuda {
                    device: compute_device,
                } if compute_device == device => {}
                other => {
                    return Err(AegisError::InvalidPlan(format!(
                        "selected CUDA hybrid layer `{}` has compute={other}",
                        region_id.0
                    )));
                }
            }
            let resident_layout = runtime_layouts
                .get(&region_id.0)
                .copied()
                .unwrap_or(LinearResidentLayout::PackedSource);
            let layer_meta = graph.layer(layer);
            let window_size = layer_meta
                .and_then(|meta| match meta.attention_pattern {
                    AttentionPattern::SlidingWindow { size } => Some(size),
                    _ => None,
                })
                .unwrap_or(0);
            // Per-layer head_dim / num_kv_heads (differ for Gemma-4 global vs
            // sliding layers, e.g. global 512/2 vs sliding 256/8). Mirrors the
            // full executor (full.rs:214-246) so the per-layer block path runs
            // Gemma-4 dense layers with the SAME geometry as all-GPU.
            let layer_head_dim = layer_meta.map(|m| m.head_dim).unwrap_or(graph.head_dim);
            let layer_num_kv_heads =
                layer_meta.map(|m| m.num_kv_heads).unwrap_or(graph.num_kv_heads);
            let partial_dim = artifact.config.partial_rotary_factor
                .map(|factor| {
                    // Partial RoPE only on global (FullCausal) layers; uses the
                    // per-layer head_dim (512 for Gemma-4 global, not 256).
                    let is_global = matches!(
                        layer_meta.map(|m| &m.attention_pattern),
                        Some(AttentionPattern::FullCausal)
                    );
                    if is_global && factor < 1.0 {
                        (factor as f64 * layer_head_dim as f64).round() as usize
                    } else {
                        0
                    }
                })
                .unwrap_or(0);
            let layer_kind = layer_meta
                .map(|m| m.kind)
                .unwrap_or(aegisllm_base::model::LayerKind::DenseDecoder);
            layers.insert(
                layer,
                load_cuda_layer(
                    &cuda_weights,
                    artifact,
                    layer,
                    region.kind,
                    layer_kind,
                    placement,
                    resident_layout,
                    CudaLayerShape {
                        hidden_size: graph.hidden_size,
                        intermediate_size: graph.intermediate_size,
                        num_attention_heads: graph.num_attention_heads,
                        num_kv_heads: layer_num_kv_heads,
                        head_dim: layer_head_dim,
                        is_sliced: graph.is_sliced,
                        text_prefix: graph.text_prefix.clone(),
                    },
                    window_size,
                    partial_dim,
                    aegisllm_base::planning::placement::WeightQuantOverride::Default,
                    aegisllm_base::planning::placement::WeightQuantOverride::Default,
                    None,
                    &mut loader,
                )?,
            );
        }

        // KV-share post-load pass (Gemma-4 E4B / E2B): set each selected shared
        // layer's `kv_shared_from` to its parent (the most recent pre-boundary
        // layer of matching attention type). Mirrors full.rs:260-299 exactly.
        // The hybrid VALIDATES upstream that a shared layer is co-located with
        // its KV parent on the same backend; this only fills the field so the
        // attention path reads the parent's (same-device) KV cache.
        if let Some(n_shared) = artifact.config.num_kv_shared_layers {
            if n_shared > 0 && n_shared < graph.num_layers {
                let first_shared = graph.num_layers - n_shared;
                let arch = aegisllm_base::model::detect_architecture(&artifact.config)?;
                let layer_is_global: Vec<bool> = (0..graph.num_layers)
                    .map(|i| matches!(
                        arch.attention_pattern(i, &artifact.config),
                        AttentionPattern::FullCausal
                    ))
                    .collect();
                for (&li, cuda_layer) in layers.iter_mut() {
                    if li < first_shared {
                        continue;
                    }
                    let need_global = layer_is_global[li];
                    let parent = (0..first_shared)
                        .rev()
                        .find(|&k| layer_is_global[k] == need_global);
                    match parent {
                        Some(p) => cuda_layer.kv_shared_from = Some(p),
                        None => {
                            return Err(AegisError::InvalidPlan(format!(
                                "hybrid KV-share: no pre-boundary parent of {} type for layer {li}",
                                if need_global { "global" } else { "sliding" }
                            )));
                        }
                    }
                }
            }
        }

        // PLE global apparatus (E4B / E2B). Loaded once; shared by all GPU
        // layers. `None` for non-PLE models.
        let ple = load_ple_global(&cuda_weights, artifact, device, &graph.text_prefix, &mut loader)?;
        let embed_scale = graph
            .embed_scale
            .or_else(|| ple.as_ref().map(|_| (graph.hidden_size as f32).sqrt()));

        // Register the now-filled arena with cuMemHostRegister so staging-pool
        // DMAs (and the PLE table reads) take the direct-pinned path. Mirrors
        // full.rs's post-load pin. No device-map: the hybrid GPU layers don't use
        // the GPU-driven MoE gather (that path is full-executor only).
        host_arena.pin_now()?;

        Ok(Self {
            runtime: cuda,
            hidden_size: graph.hidden_size,
            num_attention_heads: graph.num_attention_heads,
            num_kv_heads: graph.num_kv_heads,
            head_dim: graph.head_dim,
            rms_norm_eps: artifact.config.rms_norm_eps.unwrap_or(1e-5) as f32,
            layers,
            kv_context_size: placement.kv_cache.context_size,
            kv_quantization: placement.kv_cache.quantization,
            total_num_layers: graph.num_layers,
            ple,
            embed_scale,
            _host_arena: host_arena,
        })
    }

    #[allow(dead_code)]
    pub fn new_state(&self) -> Result<CudaLayerBlockState> {
        // Scratch q/k/v widths must fit the LARGEST per-layer geometry among the
        // selected layers. Gemma-4 global layers use head_dim=512 / 2 kv-heads
        // (q_width=num_heads*512), sliding 256/8 — so model-wide num_kv_heads *
        // head_dim under-sizes the global-layer buffers. Take the per-layer max.
        let max_q_width = self
            .layers
            .values()
            .map(|l| self.num_attention_heads * l.layer_head_dim)
            .max()
            .unwrap_or(self.num_attention_heads * self.head_dim);
        let max_kv_width = self
            .layers
            .values()
            .map(|l| l.layer_num_kv_heads * l.layer_head_dim)
            .max()
            .unwrap_or(self.num_kv_heads * self.head_dim);
        let max_head_dim = self
            .layers
            .values()
            .map(|l| l.layer_head_dim)
            .max()
            .unwrap_or(self.head_dim);
        let kv_width = max_kv_width;
        let intermediate = self
            .layers
            .values()
            .filter(|l| l.moe.is_none())
            .map(|l| l.gate_proj.rows())
            .max()
            .unwrap_or(self.hidden_size);
        let max_cutlass_input = self.hidden_size.max(intermediate);
        let cutlass_payload =
            CudaRuntime::cutlass_nvfp4_activation_payload_bytes(1, max_cutlass_input)
                .unwrap_or(1)
                .max(1);
        let cutlass_scales =
            CudaRuntime::cutlass_nvfp4_activation_scale_bytes(1, max_cutlass_input)
                .unwrap_or(1)
                .max(1);
        let cutlass_workspace = self
            .layers
            .values()
            .flat_map(|layer| {
                // All 7 dense projections are now CudaLinear; only NVFP4
                // variants need cuBLASLt-CUTLASS scratch. Filter via
                // `as_nvfp4()` before sizing.
                let mut nvfp4s: Vec<&crate::cuda::DeviceNvfp4Linear> = Vec::new();
                for cl in [&layer.gate_proj, &layer.up_proj, &layer.down_proj,
                           &layer.q_proj, &layer.k_proj, &layer.v_proj, &layer.o_proj] {
                    if let Some(l) = cl.as_nvfp4() { nvfp4s.push(l); }
                }
                nvfp4s
            })
            .filter(|linear| self.runtime.cutlass_nvfp4_inference_enabled_for(linear))
            .map(|linear| {
                self.runtime
                    .cutlass_nvfp4_workspace_bytes(1, linear.rows, linear.cols)
            })
            .try_fold(1usize, |max_bytes, bytes| {
                bytes.map(|bytes| max_bytes.max(bytes))
            })?
            .max(1);
        Ok(CudaLayerBlockState {
            hidden: self.runtime.alloc_f32(self.hidden_size)?,
            layers: self
                .layers
                .iter()
                .map(|(&layer, cuda_layer)| {
                    // Per-layer KV geometry (Gemma-4 global 512/2 vs sliding
                    // 256/8) and per-layer effective capacity (sliding layers
                    // ring-buffer `window_size` slots; global use full context).
                    // Mirrors full.rs:996-1029. Own-KV layers get a real cache;
                    // shared layers (kv_shared_from = Some) read the parent's
                    // cache and only need a 1-slot stub, but we allocate a small
                    // real cache anyway (1-element width is unsafe for the store
                    // kernel which is skipped on shared layers — so size by the
                    // layer's own width to stay valid even though it's unused).
                    let layer_kv_width = cuda_layer.layer_num_kv_heads * cuda_layer.layer_head_dim;
                    let layer_kv_capacity = if cuda_layer.window_size > 0 {
                        cuda_layer.window_size.min(self.kv_context_size)
                    } else {
                        self.kv_context_size
                    };
                    Ok((
                        layer,
                        CudaLayerState {
                            kv: super::state::CudaKvCache::dense(
                                &self.runtime,
                                self.kv_context_size,
                                layer_kv_width,
                                self.kv_quantization,
                                layer_kv_capacity,
                                cuda_layer.window_size > 0,
                            )?,
                        },
                    ))
                })
                .collect::<Result<BTreeMap<_, _>>>()?,
            scratch: CudaScratch {
                input_normed: self.runtime.alloc_f32(self.hidden_size)?,
                quant_hidden: self.runtime.alloc_f32(self.hidden_size)?,
                quant_intermediate: self.runtime.alloc_f32(intermediate)?,
                mxfp4_hidden: self
                    .runtime
                    .alloc_u8(CudaRuntime::mxfp4_vector_bytes(self.hidden_size)?)?,
                mxfp4_intermediate: self
                    .runtime
                    .alloc_u8(CudaRuntime::mxfp4_vector_bytes(intermediate)?)?,
                cutlass_payload: self.runtime.alloc_u8(cutlass_payload)?,
                cutlass_scales: self.runtime.alloc_u8(cutlass_scales)?,
                cutlass_workspace: self.runtime.alloc_u8(cutlass_workspace)?,
                q: self.runtime.alloc_f32(max_q_width)?,
                k: self.runtime.alloc_f32(kv_width)?,
                v: self.runtime.alloc_f32(kv_width)?,
                qk_norm_scratch: self.runtime.alloc_f32(max_q_width.max(kv_width))?,
                attn_split_acc: self
                    .runtime
                    .alloc_f32(self.num_attention_heads * DECODE_SPLIT_K_MAX * max_head_dim)?,
                attn_split_m: self
                    .runtime
                    .alloc_f32(self.num_attention_heads * DECODE_SPLIT_K_MAX)?,
                attn_split_l: self
                    .runtime
                    .alloc_f32(self.num_attention_heads * DECODE_SPLIT_K_MAX)?,
                attn_context: self.runtime.alloc_f32(max_q_width)?,
                attn_out: self.runtime.alloc_f32(self.hidden_size)?,
                residual: self.runtime.alloc_f32(self.hidden_size)?,
                post_normed: self.runtime.alloc_f32(self.hidden_size)?,
                gate: self.runtime.alloc_f32(intermediate)?,
                up: self.runtime.alloc_f32(intermediate)?,
                swiglu: self.runtime.alloc_f32(intermediate)?,
                mlp_out: self.runtime.alloc_f32(self.hidden_size)?,
                hidden_out: self.runtime.alloc_f32(self.hidden_size)?,
                final_hidden: self.runtime.alloc_f32(self.hidden_size)?,
                argmax_block_values: self.runtime.alloc_f32(1)?,
                argmax_block_indices: self.runtime.alloc_u32(1)?,
                moe: None,
                staging_pool: None,
                kv_staging: None,
                // PLE decode scratch — sized for the hybrid Gemma-4 dense path
                // (E4B/E2B). `per_layer_inputs` is `[total_num_layers * ple_dim]`
                // (the full model's per-layer feed; the hybrid uploads the same
                // vector the CPU computed at token entry). The remaining buffers
                // mirror the full executor's decode PLE scratch (full.rs:1117).
                // Stub size 1 when the model has no PLE (Llama / Qwen / dense
                // Gemma without PLE) — these layers run the Llama-style block path
                // and never touch the PLE scratch.
                per_layer_inputs: self.runtime.alloc_f32(self.ple.as_ref()
                    .map(|p| self.total_num_layers * p.ple_dim).unwrap_or(1))?,
                ple_projection: self.runtime.alloc_f32(self.ple.as_ref()
                    .map(|p| self.total_num_layers * p.ple_dim).unwrap_or(1))?,
                ple_projection_normed: self.runtime.alloc_f32(self.ple.as_ref()
                    .map(|p| self.total_num_layers * p.ple_dim).unwrap_or(1))?,
                ple_gate: self.runtime.alloc_f32(self.ple.as_ref()
                    .map(|p| p.ple_dim).unwrap_or(1))?,
                ple_contrib: self.runtime.alloc_f32(self.ple.as_ref()
                    .map(|_| self.hidden_size).unwrap_or(1))?,
                ple_contrib_normed: self.runtime.alloc_f32(self.ple.as_ref()
                    .map(|_| self.hidden_size).unwrap_or(1))?,
                ple_bf16_in: self.runtime.alloc_u16(self.ple.as_ref()
                    .map(|p| (self.total_num_layers * p.ple_dim).max(self.hidden_size))
                    .unwrap_or(1))?,
                ple_bf16_out: self.runtime.alloc_u16(self.ple.as_ref()
                    .map(|p| (self.total_num_layers * p.ple_dim).max(self.hidden_size))
                    .unwrap_or(1))?,
            },
            p_position: self.runtime.alloc_u32(1)?,
            p_seq_len: self.runtime.alloc_u32(1)?,
        })
    }
}

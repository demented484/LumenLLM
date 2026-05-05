use super::linear_ops::native_mxfp4_enabled;
use super::loader::{
    CudaLayerShape, cuda_residency_for_store, first_existing_tensor, load_cuda_layer,
    runtime_layouts_by_region,
};
use super::planning::validate_cuda_placement;
use super::state::{
    CudaKvCache, CudaLayer, CudaLayerState, CudaLlamaExecutor, CudaLlamaState, CudaMoEScratch,
    CudaPrefillScratch, CudaScratch, KvStagingPool, KvStagingSlot,
};
use crate::cuda::staging::LinearStagingPool;
use aegisllm_base::artifact::ModelArtifact;
use crate::cuda::{
    CUDA_PREFILL_CHUNK_MAX, CUDA_PREFILL_DENSE_SPLIT_K_TOKENS, DECODE_SPLIT_K, CudaRuntime,
    CudaRuntimeConfig,
};
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::executor::tensors::require_tensor;
use aegisllm_base::graph::{ModelGraph, RegionId};
use aegisllm_base::planning::placement::{ResolvedPlacement, StoragePlacement};
use aegisllm_base::planning::runtime::RuntimePlan;
use aegisllm_base::tensor::layout::LinearResidentLayout;
use aegisllm_base::tensor::storage::TensorStorageLoader;

const FLASH_COMPAT_PREFILL_KV_PAGE_TOKENS: usize = 256;
const PREFILL_SPLIT_K_TOKENS: usize = CUDA_PREFILL_DENSE_SPLIT_K_TOKENS;
const PREFILL_SPLIT_Q_BLOCK: usize = 16;

impl CudaLlamaExecutor {
    pub(super) fn from_artifact(
        artifact: &ModelArtifact,
        graph: &ModelGraph,
        placement: &ResolvedPlacement,
        runtime: &RuntimePlan,
        device: usize,
        cuda_config: CudaRuntimeConfig,
    ) -> Result<Self> {
        validate_cuda_placement(placement, device)?;
        if graph.num_kv_heads == 0 || !graph.num_attention_heads.is_multiple_of(graph.num_kv_heads) {
            return Err(AegisError::InvalidPlan(format!(
                "CUDA executor requires attention heads divisible by kv heads, got heads={} kv_heads={}",
                graph.num_attention_heads, graph.num_kv_heads
            )));
        }
        let cuda = CudaRuntime::new_with_config(device, cuda_config)?;
        let region_placements = placement.region_map();
        // Pre-size the pinned-host arena to **only** what will actually live in
        // it — NVFP4 weights (`.weight` + companion `.weight_scale`) inside
        // regions whose placement is host-resident (store=Ram or Mmap). BF16
        // weights inside the same regions get force-VRAM'd by the loader and
        // never use the arena, so including them here would waste 3-4 GB of
        // pinned RAM. Saves ~17 GB → ~14 GB locked for Gemma-4-26B.
        let host_arena_capacity = compute_host_arena_capacity(artifact, graph, &region_placements);
        let host_arena = std::sync::Arc::new(
            crate::cuda::host_arena::PinnedArena::new(&cuda, host_arena_capacity)?,
        );
        let cuda_weights = cuda.weight_loader_with_arena(host_arena.clone());
        let runtime_layouts = runtime_layouts_by_region(runtime);
        let mut loader = TensorStorageLoader::new();

        let embed_region = region_placements
            .get(&RegionId("embed".into()))
            .ok_or_else(|| AegisError::InvalidPlan("missing embed placement".into()))?;
        let final_norm_region = region_placements
            .get(&RegionId("final_norm".into()))
            .ok_or_else(|| AegisError::InvalidPlan("missing final_norm placement".into()))?;
        let lm_head_region = region_placements
            .get(&RegionId("lm_head".into()))
            .ok_or_else(|| AegisError::InvalidPlan("missing lm_head placement".into()))?;

        let embed_name = format!("{}embed_tokens.weight", graph.text_prefix);
        let embed_tokens = cuda_weights.load_bf16_matrix_with_store(
            first_existing_tensor(artifact, &[&embed_name, "model.embed_tokens.weight"])?,
            embed_region.store,
            cuda_residency_for_store(embed_region.store, device)?,
            &mut loader,
        )?;
        let final_norm_name = format!("{}norm.weight", graph.text_prefix);
        let final_norm = cuda_weights.load_dense_vector_with_store(
            first_existing_tensor(artifact, &[&final_norm_name, "model.norm.weight"])?,
            final_norm_region.store,
            &mut loader,
        )?;
        let lm_head_tensor = first_existing_tensor(
            artifact,
            &["lm_head.weight", &embed_name, "model.embed_tokens.weight"],
        )?;
        // lm_head: force-VRAM regardless of `store=ram` config because the matvec path
        // against host-pinned BF16 (WRITECOMBINED-uncached for CPU reads) is 30× slower
        // than the VRAM kernel. Cost: ~1 GB VRAM kept resident even in hetero mode.
        // embed_tokens (above) honors host-residency since per-token row lookup is cheap.
        let lm_head = cuda_weights.load_bf16_matrix_with_store_opts(
            lm_head_tensor,
            lm_head_region.store,
            cuda_residency_for_store(lm_head_region.store, device)?,
            &mut loader,
            true,
        )?;

        let mut layers = Vec::with_capacity(graph.num_layers);
        for layer in 0..graph.num_layers {
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
            let resident_layout = runtime_layouts
                .get(&region_id.0)
                .copied()
                .unwrap_or(LinearResidentLayout::PackedSource);
            let layer_meta = graph.layer(layer);
            let window_size = layer_meta
                .and_then(|meta| match meta.attention_pattern {
                    aegisllm_base::model::AttentionPattern::SlidingWindow { size } => Some(size),
                    _ => None,
                })
                .unwrap_or(0);
            let layer_kind = graph
                .layer(layer)
                .map(|m| m.kind)
                .unwrap_or(aegisllm_base::model::LayerKind::DenseDecoder);
            // Per-layer head_dim and num_kv_heads (differ for Gemma 4 global vs sliding).
            let layer_head_dim = layer_meta.map(|m| m.head_dim).unwrap_or(graph.head_dim);
            let layer_num_kv_heads = layer_meta.map(|m| m.num_kv_heads).unwrap_or(graph.num_kv_heads);
            let partial_dim = artifact.config.partial_rotary_factor
                .map(|factor| {
                    // Only global layers (FullCausal in Gemma 4) use partial RoPE.
                    // Use layer_head_dim (e.g. 512 for Gemma4 global) not graph.head_dim (256).
                    let is_global = matches!(
                        layer_meta.map(|m| &m.attention_pattern),
                        Some(aegisllm_base::model::AttentionPattern::FullCausal)
                    );
                    if is_global && factor < 1.0 {
                        (factor as f64 * layer_head_dim as f64).round() as usize
                    } else {
                        0
                    }
                })
                .unwrap_or(0);
            layers.push(load_cuda_layer(
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
                &mut loader,
            )?);
        }

        let has_staged_layers = layers.iter().any(|layer| {
            [&layer.q_proj, &layer.k_proj, &layer.v_proj, &layer.o_proj]
                .iter().any(|l| l.is_host_resident())
            || [&layer.gate_proj, &layer.up_proj, &layer.down_proj]
                .iter().any(|l| l.is_host_resident())
            || layer.qkv_proj.as_ref().is_some_and(|l| l.is_host_resident())
            || layer.moe.as_ref().is_some_and(|moe| {
                moe.experts.iter().any(|e| {
                    e.gate_proj.is_host_resident()
                        || e.up_proj.is_host_resident()
                        || e.down_proj.is_host_resident()
                }) || moe.shared_expert.as_ref().is_some_and(|se| {
                    se.gate_proj.is_host_resident()
                        || se.up_proj.is_host_resident()
                        || se.down_proj.is_host_resident()
                })
            })
        });

        // ── Phase 4: VRAM expert cache (env-gated) ─────────────────────────
        // After all loading completes, query free VRAM and pre-populate as
        // many host-resident NVFP4 expert weights as fit. Cache hits during
        // inference skip the per-call H2D bandwidth.
        //
        // WHY OPT-IN: empirically, on its own the cache regresses prefill
        // ~7% — eliminating the H2D path on the transfer stream removes the
        // implicit compute/transfer-stream parallelism that double-buffered
        // staging provides, and the per-matvec kernel time dominates so
        // skipping H2D doesn't claw it back. Combining the cache with a
        // grouped-GEMM dispatch (Phase 2) is expected to unlock the win
        // because per-layer kernel count drops from ~150 to ~3 and per-call
        // staging overhead becomes dominant. Until Phase 2 lands, set
        // `AEGIS_VRAM_EXPERT_CACHE=1` to enable for experimentation.
        if has_staged_layers && std::env::var("AEGIS_VRAM_EXPERT_CACHE").is_ok() {
            let safety_margin: usize = 1usize << 30; // 1 GB headroom
            match crate::cuda::expert_cache::pick_cache_capacity(&cuda, safety_margin) {
                Ok(capacity_bytes) if capacity_bytes > 0 => {
                    let mut cache = crate::cuda::expert_cache::VramExpertCache::new(
                        &cuda, capacity_bytes,
                    )?;
                    let mut inserted: usize = 0;
                    let mut skipped: usize = 0;
                    // Iterate experts in load order. Static cache; first-fit fills
                    // the buffer until capacity is exhausted. Once the cache is
                    // full we stop — remaining experts use the per-call staging
                    // pool path.
                    'outer: for layer in &layers {
                        if let Some(moe) = layer.moe.as_ref() {
                            for expert in &moe.experts {
                                for proj in [
                                    &expert.gate_proj,
                                    &expert.up_proj,
                                    &expert.down_proj,
                                ] {
                                    let placed = crate::cuda::expert_cache::try_cache_nvfp4_expert(
                                        &mut cache, &cuda, proj,
                                    )?;
                                    if placed { inserted += 1; }
                                    else      { skipped  += 1;  break 'outer; }
                                }
                            }
                        }
                    }
                    eprintln!(
                        "vram-expert-cache: capacity={} MB used={} MB inserted={} skipped={}",
                        cache.capacity_bytes() / (1usize << 20),
                        cache.used_bytes() / (1usize << 20),
                        inserted,
                        skipped,
                    );
                    cuda.install_expert_cache(std::sync::Arc::new(cache))?;
                }
                Ok(_) => {
                    eprintln!("vram-expert-cache: skipped (capacity 0 after safety margin)");
                }
                Err(e) => {
                    eprintln!("vram-expert-cache: skipped: {e:?}");
                }
            }
        }

        let kv_store = placement.kv_cache.store;
        let kv_first_n_layers = placement.kv_cache.first_n_layers;
        let kv_first_store = placement.kv_cache.first_store;
        let num_layers = graph.num_layers;
        // Heterogeneous KV placement → some layer's KV is host-resident → inhibit
        // CUDA Graph capture. The check covers either (a) tail tier is RAM, or
        // (b) first-N tier exists and either tier is RAM (incl. when first_store
        // is unset and the implicit fallback to VRAM is fine, but tail is RAM).
        let first_store_implies_staging = kv_first_store
            .map(|s| matches!(s, StoragePlacement::Ram))
            .unwrap_or(false);
        let has_staged_kv = matches!(kv_store, StoragePlacement::Ram)
            || (kv_first_n_layers.is_some_and(|n| n < num_layers)
                && (matches!(kv_store, StoragePlacement::Ram) || first_store_implies_staging));
        let effective_prefill_chunk_size = cuda_prefill_chunk_size(cuda_config);
        Ok(Self {
            runtime: cuda,
            hidden_size: graph.hidden_size,
            num_attention_heads: graph.num_attention_heads,
            num_kv_heads: graph.num_kv_heads,
            head_dim: graph.head_dim,
            rms_norm_eps: artifact.config.rms_norm_eps.unwrap_or(1e-5) as f32,
            embed_tokens,
            final_norm,
            lm_head,
            layers,
            kv_context_size: placement.kv_cache.context_size,
            prefill_chunk_size: effective_prefill_chunk_size,
            prefill_stage_timings_enabled: cuda_config.prefill_stage_timings,
            lm_head_softcap: graph.lm_head_softcap,
            embed_scale: graph.embed_scale,
            has_staged_layers,
            has_staged_kv,
            kv_store,
            kv_first_n_layers,
            kv_first_store,
        })
    }

    pub(super) fn new_state(&self) -> Result<CudaLlamaState> {
        // Use per-layer max values for scratch buffer sizing to handle heterogeneous layers
        // (e.g. Gemma 4 global vs sliding attention have different head_dim / num_kv_heads).
        let max_q_width = self.layers.iter()
            .map(|l| self.num_attention_heads * l.layer_head_dim)
            .max().unwrap_or(self.num_attention_heads * self.head_dim);
        let max_kv_width = self.layers.iter()
            .map(|l| l.layer_num_kv_heads * l.layer_head_dim)
            .max().unwrap_or(self.num_kv_heads * self.head_dim);
        let max_head_dim = self.layers.iter()
            .map(|l| l.layer_head_dim)
            .max().unwrap_or(self.head_dim);
        let kv_width = max_kv_width;
        let qkv_width = max_q_width + 2 * max_kv_width;
        // Dense intermediate size: max across non-MoE layers (MoE layers have a 1-row dummy).
        let intermediate = self
            .layers
            .iter()
            .filter(|l| l.moe.is_none())
            .map(|l| l.gate_proj.rows)
            .max()
            .unwrap_or(self.hidden_size);
        // MoE expert intermediate size: max across MoE layers (0 if no MoE layers).
        let moe_intermediate = self
            .layers
            .iter()
            .filter_map(|l| l.moe.as_ref())
            .map(|m| m.expert_intermediate_size)
            .max()
            .unwrap_or(0);
        // Shared expert intermediate size (e.g. Gemma4 always-active expert has a larger
        // intermediate than routed experts). Scratch gate/up/swiglu must fit both.
        let shared_expert_intermediate = self
            .layers
            .iter()
            .filter_map(|l| l.moe.as_ref())
            .filter_map(|m| m.shared_expert.as_ref())
            .map(|se| se.gate_proj.rows())
            .max()
            .unwrap_or(0);
        let max_expert_intermediate = moe_intermediate.max(shared_expert_intermediate);
        let needs_fallback_down_scratch = self.layers.iter().any(|layer| {
            !self
                .runtime
                .cutlass_nvfp4_inference_enabled_for(&layer.down_proj)
                && !native_mxfp4_enabled(&self.runtime, &layer.down_proj)
        });
        let needs_quant_hidden_scratch = self
            .layers
            .iter()
            .any(|layer| prefill_layer_needs_quant_hidden_scratch(&self.runtime, layer));
        let needs_mxfp4_hidden_scratch = self
            .layers
            .iter()
            .any(|layer| prefill_layer_needs_mxfp4_hidden_scratch(&self.runtime, layer));
        let needs_mxfp4_down_scratch = needs_fallback_down_scratch
            || self
                .layers
                .iter()
                .any(|layer| linear_on_native_mxfp4_path(&self.runtime, &layer.down_proj));
        let quant_hidden_len = if needs_quant_hidden_scratch {
            self.prefill_chunk_size * self.hidden_size
        } else {
            1
        };
        let mxfp4_hidden_len = if needs_mxfp4_hidden_scratch {
            self.prefill_chunk_size * CudaRuntime::mxfp4_vector_bytes(self.hidden_size)?
        } else {
            1
        };
        let quant_intermediate_len = if needs_fallback_down_scratch {
            self.prefill_chunk_size * intermediate
        } else {
            1
        };
        let mxfp4_intermediate_len = if needs_mxfp4_down_scratch {
            self.prefill_chunk_size * CudaRuntime::mxfp4_vector_bytes(intermediate)?
        } else {
            1
        };
        let cutlass_prefill_scratch =
            cutlass_prefill_scratch_bytes(self, self.prefill_chunk_size, intermediate)?;
        let cutlass_decode_scratch = cutlass_prefill_scratch_bytes(self, 1, intermediate)?;
        let prefill_attention_scratch =
            prefill_attention_split_scratch(self, self.prefill_chunk_size)?;
        let prefill = if self.prefill_chunk_size > 1 {
            let prefill_max_sequences = 1;
            let prefill_block_table_capacity = self
                .kv_context_size
                .div_ceil(FLASH_COMPAT_PREFILL_KV_PAGE_TOKENS)
                .max(1);
            Some(CudaPrefillScratch {
                chunk_size: self.prefill_chunk_size,
                max_sequences: prefill_max_sequences,
                block_table_capacity: prefill_block_table_capacity,
                request_ids_host: Vec::with_capacity(prefill_max_sequences),
                seq_ids_host: Vec::with_capacity(prefill_max_sequences),
                token_host: Vec::with_capacity(self.prefill_chunk_size),
                position_host: Vec::with_capacity(self.prefill_chunk_size),
                slot_mapping_host: Vec::with_capacity(self.prefill_chunk_size),
                cu_q_host: Vec::with_capacity(prefill_max_sequences + 1),
                cu_k_host: Vec::with_capacity(prefill_max_sequences + 1),
                context_lens_host: Vec::with_capacity(prefill_max_sequences),
                block_tables_host: Vec::with_capacity(prefill_block_table_capacity),
                request_ids: self.runtime.alloc_u32(prefill_max_sequences)?,
                seq_ids: self.runtime.alloc_u32(prefill_max_sequences)?,
                tokens: self.runtime.alloc_u32(self.prefill_chunk_size)?,
                positions: self.runtime.alloc_u32(self.prefill_chunk_size)?,
                slot_mapping: self.runtime.alloc_u32(self.prefill_chunk_size)?,
                cu_q: self.runtime.alloc_u32(prefill_max_sequences + 1)?,
                cu_k: self.runtime.alloc_u32(prefill_max_sequences + 1)?,
                context_lens: self.runtime.alloc_u32(prefill_max_sequences)?,
                block_tables: self.runtime.alloc_u32(prefill_block_table_capacity)?,
                hidden: self
                    .runtime
                    .alloc_f32(self.prefill_chunk_size * self.hidden_size)?,
                input_normed: self
                    .runtime
                    .alloc_f32(self.prefill_chunk_size * self.hidden_size)?,
                quant_hidden: self.runtime.alloc_f32(quant_hidden_len)?,
                quant_intermediate: self.runtime.alloc_f32(quant_intermediate_len)?,
                mxfp4_hidden: self.runtime.alloc_u8(mxfp4_hidden_len)?,
                mxfp4_intermediate: self.runtime.alloc_u8(mxfp4_intermediate_len)?,
                cutlass_payload: self
                    .runtime
                    .alloc_u8(cutlass_prefill_scratch.payload_bytes)?,
                cutlass_scales: self.runtime.alloc_u8(cutlass_prefill_scratch.scale_bytes)?,
                cutlass_workspace: self
                    .runtime
                    .alloc_u8(cutlass_prefill_scratch.workspace_bytes)?,
                qkv: self
                    .runtime
                    .alloc_f32(self.prefill_chunk_size * qkv_width)?,
                // Q reuses the gate buffer until MLP starts.
                q: self.runtime.alloc_f32(1)?,
                // Size q_half to fit Q for the WIDEST layer's head_dim, not the
                // model-wide one. Gemma 4 has heterogeneous heads (sliding=256,
                // global=512); using model-wide head_dim was a buffer-overflow
                // bug that silently corrupted attention state on global layers.
                q_half: self
                    .runtime
                    .alloc_u16(self.prefill_chunk_size * max_q_width)?,
                attn_split_acc: self.runtime.alloc_f32(prefill_attention_scratch.acc_f32)?,
                attn_split_m: self
                    .runtime
                    .alloc_f32(prefill_attention_scratch.stats_f32)?,
                attn_split_l: self
                    .runtime
                    .alloc_f32(prefill_attention_scratch.stats_f32)?,
                // K reuses the up buffer until MLP starts.
                k: self.runtime.alloc_f32(1)?,
                v: self.runtime.alloc_f32(self.prefill_chunk_size * max_kv_width)?,
                // Reused output now lives in qkv after QKV split has consumed it.
                attn_context: self.runtime.alloc_f32(1)?,
                // Reused output now lives in input_normed after QKV/attention.
                attn_out: self.runtime.alloc_f32(1)?,
                // For MoE-only models, `intermediate` is 1 (dense MLP path is dead), but
                // `gate` doubles as the Q output buffer during chunked attention and `up`
                // doubles as the K output buffer. Size them to fit attention dims so the
                // BF16 chunked attention path works.
                gate: self
                    .runtime
                    .alloc_f32(self.prefill_chunk_size * intermediate.max(max_q_width))?,
                up: self
                    .runtime
                    .alloc_f32(self.prefill_chunk_size * intermediate.max(max_kv_width))?,
                swiglu: self.runtime.alloc_f32(1)?,
                // Reused output now lives in input_normed after gate/up.
                mlp_out: self.runtime.alloc_f32(1)?,
                // cuBLASLt BF16 scratch — size for the largest single matmul. Worst
                // case is shared MLP / down_proj: chunk_size * max(intermediate, hidden,
                // max_q_width). For Gemma 4 26B: 64 * max(2112, 2816, 8192) = 64*8192 ≈ 1MB.
                bf16_in_scratch: self.runtime.alloc_u16(
                    self.prefill_chunk_size * intermediate.max(self.hidden_size).max(max_q_width)
                )?,
                bf16_out_scratch: self.runtime.alloc_u16(
                    self.prefill_chunk_size * intermediate.max(self.hidden_size).max(max_q_width)
                )?,
                moe: if moe_intermediate > 0 {
                    let cs = self.prefill_chunk_size;
                    let max_experts = self.layers.iter()
                        .filter_map(|l| l.moe.as_ref())
                        .map(|m| m.num_experts)
                        .max()
                        .unwrap_or(1);
                    let max_top_k = self.layers.iter()
                        .filter_map(|l| l.moe.as_ref())
                        .map(|m| m.top_k)
                        .max()
                        .unwrap_or(1);
                    // Intermediate buffer needs to fit either expert-MLP intermediate
                    // (moe_intermediate) or the down_proj input. moe_intermediate is
                    // smaller (704 for Gemma 4), hidden_size is bigger (2816).
                    let max_dim = self.hidden_size.max(max_expert_intermediate);
                    let max_per_expert = cs * max_top_k; // worst case all routed to 1
                    Some(Box::new(super::state::CudaMoEPrefillScratch {
                        router_logits: self.runtime.alloc_f32(cs * max_experts)?,
                        router_input: self.runtime.alloc_f32(cs * self.hidden_size)?,
                        expert_input: self.runtime.alloc_f32(cs * self.hidden_size)?,
                        moe_acc: self.runtime.alloc_f32(cs * self.hidden_size)?,
                        stream1: self.runtime.alloc_f32(cs * self.hidden_size)?,
                        gather_input: self.runtime.alloc_f32(cs * self.hidden_size)?,
                        gather_intermediate: self.runtime.alloc_f32(cs * max_expert_intermediate)?,
                        gather_swiglu: self.runtime.alloc_f32(cs * max_expert_intermediate)?,
                        gather_out: self.runtime.alloc_f32(cs * self.hidden_size)?,
                        gather_quant: self.runtime.alloc_f32(cs * max_dim)?,
                        gather_mxfp4: self.runtime.alloc_u8(
                            cs * CudaRuntime::mxfp4_vector_bytes(max_dim)?,
                        )?,
                        gather_indices: self.runtime.alloc_u32(cs * max_top_k)?,
                        gather_weights: self.runtime.alloc_f32(cs * max_top_k)?,
                        // GPU router top-k scratch (Phase 1 of perf overhaul).
                        topk_idx: self.runtime.alloc_u32(cs * max_top_k)?,
                        topk_weights: self.runtime.alloc_f32(cs * max_top_k)?,
                        expert_token_lists: self.runtime.alloc_u32(max_experts * max_per_expert)?,
                        expert_weight_lists: self.runtime.alloc_f32(max_experts * max_per_expert)?,
                        expert_counts: self.runtime.alloc_u32(max_experts)?,
                        expert_list_stride: max_per_expert,
                        // Phase 2 grouped-MoE scratch (used when VRAM cache is on).
                        expert_offsets: self.runtime.alloc_u32(max_experts + 1)?,
                        cached_counts: self.runtime.alloc_u32(max_experts)?,
                        permuted_input: self.runtime.alloc_f32(cs * max_top_k * self.hidden_size)?,
                        permuted_gate: self.runtime.alloc_f32(cs * max_top_k * max_expert_intermediate)?,
                        permuted_up: self.runtime.alloc_f32(cs * max_top_k * max_expert_intermediate)?,
                        permuted_down: self.runtime.alloc_f32(cs * max_top_k * self.hidden_size)?,
                        gate_packed_offsets: self.runtime.alloc_u32(max_experts)?,
                        gate_scales_offsets: self.runtime.alloc_u32(max_experts)?,
                        up_packed_offsets: self.runtime.alloc_u32(max_experts)?,
                        up_scales_offsets: self.runtime.alloc_u32(max_experts)?,
                        down_packed_offsets: self.runtime.alloc_u32(max_experts)?,
                        down_scales_offsets: self.runtime.alloc_u32(max_experts)?,
                        gate_input_scales: self.runtime.alloc_f32(max_experts)?,
                        gate_output_scales: self.runtime.alloc_f32(max_experts)?,
                        up_input_scales: self.runtime.alloc_f32(max_experts)?,
                        up_output_scales: self.runtime.alloc_f32(max_experts)?,
                        down_input_scales: self.runtime.alloc_f32(max_experts)?,
                        down_output_scales: self.runtime.alloc_f32(max_experts)?,
                    }))
                } else {
                    None
                },
            })
        } else {
            None
        };

        Ok(CudaLlamaState {
            position: 0,
            hidden: self.runtime.alloc_f32(self.hidden_size)?,
            logits: self.runtime.alloc_f32(self.lm_head.rows)?,
            sampled_token: self.runtime.alloc_u32(1)?,
            layers: self.layers.iter().enumerate()
                .map(|(layer_idx, layer)| {
                    // Per-layer KV width (differs for Gemma 4 global vs sliding layers).
                    let layer_kv_width = layer.layer_num_kv_heads * layer.layer_head_dim;
                    // Resolve per-layer KV store: first_store for layers < first_n_layers,
                    // else the tail kv_store. first_store=None with first_n_layers=Some(_)
                    // means "VRAM derived from compute" — preserve legacy behavior by
                    // defaulting to the dense (VRAM) cache.
                    let layer_store = match self.kv_first_n_layers {
                        Some(n) if layer_idx < n => self.kv_first_store
                            .unwrap_or_else(|| {
                                // Legacy: implicit VRAM. Use the runtime's device.
                                StoragePlacement::Vram { device: self.runtime.device_index() }
                            }),
                        _ => self.kv_store,
                    };
                    let kv = match layer_store {
                        StoragePlacement::Vram { .. } => {
                            CudaKvCache::dense(&self.runtime, self.kv_context_size, layer_kv_width)?
                        }
                        StoragePlacement::Ram | StoragePlacement::Mmap => {
                            CudaKvCache::staged_host(&self.runtime, self.kv_context_size, layer_kv_width)?
                        }
                    };
                    Ok(CudaLayerState { kv })
                })
                .collect::<Result<Vec<_>>>()?,
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
                cutlass_payload: self
                    .runtime
                    .alloc_u8(cutlass_decode_scratch.payload_bytes)?,
                cutlass_scales: self.runtime.alloc_u8(cutlass_decode_scratch.scale_bytes)?,
                cutlass_workspace: self
                    .runtime
                    .alloc_u8(cutlass_decode_scratch.workspace_bytes)?,
                q: self.runtime.alloc_f32(max_q_width)?,
                k: self.runtime.alloc_f32(max_kv_width)?,
                v: self.runtime.alloc_f32(max_kv_width)?,
                qk_norm_scratch: self.runtime.alloc_f32(max_q_width.max(max_kv_width))?,
                attn_split_acc: self
                    .runtime
                    .alloc_f32(self.num_attention_heads * DECODE_SPLIT_K * max_head_dim)?,
                attn_split_m: self
                    .runtime
                    .alloc_f32(self.num_attention_heads * DECODE_SPLIT_K)?,
                attn_split_l: self
                    .runtime
                    .alloc_f32(self.num_attention_heads * DECODE_SPLIT_K)?,
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
                argmax_block_values: self.runtime.alloc_f32(self.lm_head.rows.div_ceil(256))?,
                argmax_block_indices: self.runtime.alloc_u32(self.lm_head.rows.div_ceil(256))?,
                staging_pool: if self.has_staged_layers {
                    let (max_packed, max_scale, max_native_mxfp4) = self.max_staged_layer_bytes();
                    Some(Box::new(LinearStagingPool::new(
                        max_packed,
                        max_scale,
                        max_native_mxfp4,
                        self.runtime.stream(),
                    )?))
                } else {
                    None
                },
                kv_staging: if self.has_staged_kv {
                    let make_slot = || -> Result<KvStagingSlot> {
                        Ok(KvStagingSlot {
                            keys: self.runtime.alloc_u16(self.kv_context_size * kv_width)?,
                            values: self.runtime.alloc_u16(self.kv_context_size * kv_width)?,
                            context_size: self.kv_context_size,
                            kv_width,
                        })
                    };
                    Some(Box::new(KvStagingPool {
                        slots: [make_slot()?, make_slot()?],
                        last_compute_event: [None, None],
                    }))
                } else {
                    None
                },
                moe: if moe_intermediate > 0 {
                    let max_input = self.hidden_size.max(max_expert_intermediate);
                    Some(Box::new(CudaMoEScratch {
                        router_logits: self.runtime.alloc_f32(
                            self.layers
                                .iter()
                                .filter_map(|l| l.moe.as_ref())
                                .map(|m| m.num_experts)
                                .max()
                                .unwrap_or(1),
                        )?,
                        router_input_scratch: self.runtime.alloc_f32(self.hidden_size)?,
                        moe_acc: self.runtime.alloc_f32(self.hidden_size)?,
                        expert_gate: self.runtime.alloc_f32(max_expert_intermediate)?,
                        expert_up: self.runtime.alloc_f32(max_expert_intermediate)?,
                        expert_swiglu: self.runtime.alloc_f32(max_expert_intermediate)?,
                        expert_out: self.runtime.alloc_f32(self.hidden_size)?,
                        quant_expert: self.runtime.alloc_f32(max_input)?,
                        mxfp4_expert: self
                            .runtime
                            .alloc_u8(CudaRuntime::mxfp4_vector_bytes(max_input)?)?,
                    }))
                } else {
                    None
                },
            },
            prefill,
            prefill_timings: super::state::CudaPrefillStageTimings::from_enabled(
                self.prefill_stage_timings_enabled,
            ),
            decode_position: self.runtime.alloc_u32(1)?,
            decode_seq_len: self.runtime.alloc_u32(1)?,
            decode_graph: None,
        })
    }

    /// Returns `(max_packed_bytes, max_scale_bytes, max_native_mxfp4_bytes)` across all
    /// host-resident linears in the model.
    fn max_staged_layer_bytes(&self) -> (usize, usize, usize) {
        let mut max_p = 0usize;
        let mut max_s = 0usize;
        let mut max_m = 0usize;
        for layer in &self.layers {
            let nvfp4_linears: Vec<&crate::cuda::DeviceNvfp4Linear> = {
                let mut v: Vec<&crate::cuda::DeviceNvfp4Linear> = vec![
                    &layer.gate_proj, &layer.up_proj, &layer.down_proj,
                ];
                // Add q/k/v/o if they're NVFP4
                for cl in [&layer.q_proj, &layer.k_proj, &layer.v_proj, &layer.o_proj] {
                    if let Some(l) = cl.as_nvfp4() { v.push(l); }
                }
                if let Some(ref qkv) = layer.qkv_proj {
                    if let Some(l) = qkv.as_nvfp4() { v.push(l); }
                }
                if let Some(ref moe) = layer.moe {
                    for e in &moe.experts {
                        v.push(&e.gate_proj);
                        v.push(&e.up_proj);
                        v.push(&e.down_proj);
                    }
                    if let Some(ref se) = moe.shared_expert {
                        if let Some(l) = se.gate_proj.as_nvfp4() { v.push(l); }
                        if let Some(l) = se.up_proj.as_nvfp4() { v.push(l); }
                        if let Some(l) = se.down_proj.as_nvfp4() { v.push(l); }
                    }
                }
                v
            };
            for l in nvfp4_linears {
                if l.is_host_resident() {
                    max_p = max_p.max(l.packed_bytes);
                    max_s = max_s.max(l.scale_bytes);
                    max_m = max_m.max(l.host_resident_native_mxfp4_bytes());
                }
            }
        }
        (max_p, max_s, max_m)
    }
}

/// True if this linear will use the native MXFP4 tensor-core path at inference time.
/// Covers both VRAM-resident repacked layers and host-resident layers with repacked data.
fn linear_on_native_mxfp4_path(runtime: &CudaRuntime, linear: &crate::cuda::DeviceNvfp4Linear) -> bool {
    native_mxfp4_enabled(runtime, linear) || linear.is_host_resident_with_native_mxfp4()
}

fn prefill_layer_needs_quant_hidden_scratch(
    runtime: &CudaRuntime,
    layer: &CudaLayer,
) -> bool {
    // Uses native_mxfp4_enabled (VRAM only) intentionally: the else branch in layer.rs
    // still calls rms_norm_quant_nvfp4_batched_device for host-resident layers even when
    // they have repacked native MXFP4 data in RAM.  quant_hidden must be large enough for
    // that write even though the data is subsequently ignored by the staged native MXFP4 path.
    //
    // BF16 attention layers don't need quant_hidden for QKV (rms_norm path), but the
    // qkv_fallback check for BF16 layers resolves to false since none of the NVFP4 methods apply.
    let qkv_group_cutlass = layer
        .qkv_proj
        .as_ref()
        .is_some_and(|linear| linear.cutlass_nvfp4_enabled(runtime));
    let qkv_all_cutlass = [&layer.q_proj, &layer.k_proj, &layer.v_proj]
        .into_iter()
        .all(|linear| linear.cutlass_nvfp4_enabled(runtime));
    let qkv_all_native = [&layer.q_proj, &layer.k_proj, &layer.v_proj]
        .into_iter()
        .all(|linear| linear.native_mxfp4_enabled(runtime));
    let qkv_fallback_needs_quant = !qkv_group_cutlass
        && !qkv_all_cutlass
        && !qkv_all_native
        && [&layer.q_proj, &layer.k_proj, &layer.v_proj]
            .into_iter()
            .any(|linear| !linear.native_mxfp4_enabled(runtime));

    let o_needs_quant = !layer.o_proj.cutlass_nvfp4_enabled(runtime)
        && !layer.o_proj.native_mxfp4_enabled(runtime)
        // BF16 o_proj doesn't need quant_hidden scratch
        && layer.o_proj.as_nvfp4().is_some();

    let gate_up_all_cutlass = [&layer.gate_proj, &layer.up_proj]
        .into_iter()
        .all(|linear| runtime.cutlass_nvfp4_inference_enabled_for(linear));
    let gate_up_all_native = [&layer.gate_proj, &layer.up_proj]
        .into_iter()
        .all(|linear| native_mxfp4_enabled(runtime, linear));
    let gate_up_fallback_needs_quant = !gate_up_all_cutlass
        && !gate_up_all_native
        && [&layer.gate_proj, &layer.up_proj]
            .into_iter()
            .any(|linear| !native_mxfp4_enabled(runtime, linear));

    qkv_fallback_needs_quant || o_needs_quant || gate_up_fallback_needs_quant
}

fn prefill_layer_needs_mxfp4_hidden_scratch(
    runtime: &CudaRuntime,
    layer: &CudaLayer,
) -> bool {
    let qkv_group_cutlass = layer
        .qkv_proj
        .as_ref()
        .is_some_and(|linear| linear.cutlass_nvfp4_enabled(runtime));
    let qkv_all_cutlass = [&layer.q_proj, &layer.k_proj, &layer.v_proj]
        .into_iter()
        .all(|linear| linear.cutlass_nvfp4_enabled(runtime));
    // BF16 projections never use the mxfp4 path
    let qkv_needs_mxfp4 = !qkv_group_cutlass
        && !qkv_all_cutlass
        && [&layer.q_proj, &layer.k_proj, &layer.v_proj]
            .into_iter()
            .any(|linear| {
                linear.as_nvfp4().is_some_and(|l| linear_on_native_mxfp4_path(runtime, l))
            });

    let o_needs_mxfp4 = !layer.o_proj.cutlass_nvfp4_enabled(runtime)
        && layer.o_proj.as_nvfp4().is_some_and(|l| linear_on_native_mxfp4_path(runtime, l));

    let gate_up_all_cutlass = [&layer.gate_proj, &layer.up_proj]
        .into_iter()
        .all(|linear| runtime.cutlass_nvfp4_inference_enabled_for(linear));
    let gate_up_needs_mxfp4 = !gate_up_all_cutlass
        && [&layer.gate_proj, &layer.up_proj]
            .into_iter()
            .any(|linear| linear_on_native_mxfp4_path(runtime, linear));

    qkv_needs_mxfp4 || o_needs_mxfp4 || gate_up_needs_mxfp4
}

#[derive(Debug, Clone, Copy)]
struct PrefillAttentionSplitScratchBytes {
    acc_f32: usize,
    stats_f32: usize,
}

fn prefill_attention_split_scratch(
    executor: &CudaLlamaExecutor,
    chunk_size: usize,
) -> Result<PrefillAttentionSplitScratchBytes> {
    let split_enabled = std::env::var_os("AEGISLLM_CUDA_DISABLE_SPLIT_K_ATTENTION").is_none()
        && std::env::var_os("AEGISLLM_CUDA_EXPERIMENTAL_SPLIT_K_ATTENTION").is_some();
    if !split_enabled {
        return Ok(PrefillAttentionSplitScratchBytes {
            acc_f32: 1,
            stats_f32: 1,
        });
    }
    let q_blocks = chunk_size.div_ceil(PREFILL_SPLIT_Q_BLOCK);
    // Split-K scratch must cover the largest KV span a prefill chunk can see, not
    // just the number of query rows in the current chunk.
    let splits = executor
        .kv_context_size
        .div_ceil(PREFILL_SPLIT_K_TOKENS)
        .max(1);
    let rows = q_blocks
        .checked_mul(executor.num_attention_heads)
        .and_then(|value| value.checked_mul(splits))
        .and_then(|value| value.checked_mul(PREFILL_SPLIT_Q_BLOCK))
        .ok_or_else(|| {
            AegisError::InvalidPlan("prefill split attention scratch overflow".into())
        })?;
    // Use the widest layer's head_dim; Gemma 4 global layers have head_dim=512
    // even when graph.head_dim is 256 (sliding default). Sizing to model-wide
    // head_dim would underflow the buffer when global layers run.
    let max_head_dim = executor
        .layers
        .iter()
        .map(|l| l.layer_head_dim)
        .max()
        .unwrap_or(executor.head_dim);
    let acc_f32 = rows
        .checked_mul(max_head_dim)
        .ok_or_else(|| AegisError::InvalidPlan("prefill split attention acc overflow".into()))?;
    Ok(PrefillAttentionSplitScratchBytes {
        acc_f32: acc_f32.max(1),
        stats_f32: rows.max(1),
    })
}

/// Sum the bytes that will land in the pinned-host arena: NVFP4 weights and
/// their `.weight_scale` companions inside host-resident regions (placement
/// store is RAM or Mmap, not VRAM).
///
/// We skip:
/// * tensors in VRAM-store regions (embed/lm_head/final_norm in our config) —
///   they go straight to VRAM, not through the arena;
/// * BF16 weights without a `.weight_scale` companion (attention Q/K/V/O,
///   shared MLP, norms) — the loader force-VRAMs these even when their
///   region's store is RAM, so they never touch the arena;
/// * input_scale / output_scale scalars — read once into f32 at load time, no
///   pinned host residency needed.
///
/// Returned value is at least 1 to satisfy `PinnedArena::new`'s non-zero
/// capacity check (some configs may have no host-resident NVFP4 weights at
/// all — for those the arena is allocated tiny and never used).
fn compute_host_arena_capacity(
    artifact: &aegisllm_base::artifact::ModelArtifact,
    graph: &aegisllm_base::graph::ModelGraph,
    region_placements: &std::collections::BTreeMap<
        &aegisllm_base::graph::RegionId,
        &aegisllm_base::planning::placement::RegionPlacement,
    >,
) -> usize {
    use aegisllm_base::planning::placement::StoragePlacement;
    let mut total: usize = 0;
    for region in &graph.regions {
        let placement = match region_placements.get(&region.id) {
            Some(p) => p,
            None => continue,
        };
        // Only host-resident regions contribute to the arena.
        let host_resident = !matches!(placement.store, StoragePlacement::Vram { .. });
        if !host_resident {
            continue;
        }
        for graph_tensor in &region.tensors {
            let name = &graph_tensor.info.name;
            // NVFP4 quantised weight: paired `.weight` + `.weight_scale`.
            if let Some(stem) = name.strip_suffix(".weight") {
                let scale_name = format!("{stem}.weight_scale");
                if artifact.tensors.has(&scale_name) {
                    total = total.saturating_add(graph_tensor.info.data_len_bytes() as usize);
                }
            } else if name.ends_with(".weight_scale") {
                total = total.saturating_add(graph_tensor.info.data_len_bytes() as usize);
            }
            // Scalar `.input_scale` / `.weight_scale_2` and unpaired `.weight`
            // (BF16 attention etc.) are intentionally skipped — they don't go
            // through the arena.
        }
    }
    total.max(1)
}

fn cuda_prefill_chunk_size(config: CudaRuntimeConfig) -> usize {
    config
        .prefill_chunk_size
        .unwrap_or(128)
        .clamp(1, CUDA_PREFILL_CHUNK_MAX)
}

struct CutlassPrefillScratchBytes {
    payload_bytes: usize,
    scale_bytes: usize,
    workspace_bytes: usize,
}

fn cutlass_prefill_scratch_bytes(
    executor: &CudaLlamaExecutor,
    chunk_size: usize,
    intermediate: usize,
) -> Result<CutlassPrefillScratchBytes> {
    let max_input = executor.hidden_size.max(intermediate);
    let payload_bytes =
        CudaRuntime::cutlass_nvfp4_activation_payload_bytes(chunk_size, max_input).unwrap_or(1);
    let scale_bytes =
        CudaRuntime::cutlass_nvfp4_activation_scale_bytes(chunk_size, max_input).unwrap_or(1);
    // Collect NVFP4 linears for cutlass workspace computation.
    // q/k/v/o/qkv are CudaLinear (may be BF16); gate/up/down are always DeviceNvfp4Linear.
    let any_cutlass = executor.layers.iter().any(|layer| {
        [&layer.q_proj, &layer.k_proj, &layer.v_proj, &layer.o_proj]
            .iter()
            .any(|cl| cl.cutlass_nvfp4_enabled(&executor.runtime))
        || layer.qkv_proj.as_ref().is_some_and(|cl| cl.cutlass_nvfp4_enabled(&executor.runtime))
        || [&layer.gate_proj, &layer.up_proj, &layer.down_proj]
            .iter()
            .any(|l| executor.runtime.cutlass_nvfp4_inference_enabled_for(*l))
    });
    let workspace_bytes = if any_cutlass {
        executor
            .layers
            .iter()
            .flat_map(|layer| {
                // gate/up/down are always DeviceNvfp4Linear
                let mut nvfp4s: Vec<&crate::cuda::DeviceNvfp4Linear> = vec![
                    &layer.gate_proj, &layer.up_proj, &layer.down_proj,
                ];
                for cl in [&layer.q_proj, &layer.k_proj, &layer.v_proj, &layer.o_proj] {
                    if let Some(l) = cl.as_nvfp4() { nvfp4s.push(l); }
                }
                if let Some(ref qkv) = layer.qkv_proj {
                    if let Some(l) = qkv.as_nvfp4() { nvfp4s.push(l); }
                }
                nvfp4s
            })
            .filter(|linear| executor.runtime.cutlass_nvfp4_inference_enabled_for(linear))
            .map(|linear| {
                executor
                    .runtime
                    .cutlass_nvfp4_workspace_bytes(chunk_size, linear.rows, linear.cols)
            })
            .try_fold(1usize, |max_bytes, bytes| {
                bytes.map(|bytes| max_bytes.max(bytes))
            })?
    } else {
        1
    };
    Ok(CutlassPrefillScratchBytes {
        payload_bytes: payload_bytes.max(1),
        scale_bytes: scale_bytes.max(1),
        workspace_bytes: workspace_bytes.max(1),
    })
}

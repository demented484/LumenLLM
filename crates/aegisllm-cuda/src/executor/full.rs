use super::linear_ops::native_mxfp4_enabled;
use super::load_progress::LoadProgress;
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
        let attention_store_override = placement.attention_store_override;
        // Reject load-time quantization overrides that don't yet have a
        // runtime implementation. The schema parses all of mxfp4 / fp8 /
        // mxint4 / int4 / int8, but only the formats below have a wired
        // loader path today; the rest fail with a clear error.
        use aegisllm_base::planning::placement::WeightQuantOverride as Wq;
        // attention-quantization: `default` (BF16) and `fp8` (load-time
        // BF16→E4M3 with per-row FP32 scales) are wired. NVFP4 layers (with
        // checkpoint-side weight_scale tensors) ignore this knob — they're
        // already 4-bit.
        match placement.attention_quantization {
            Wq::Default | Wq::Fp8 => {}
            other => return Err(AegisError::Unsupported(format!(
                "attention-quantization={other:?} not yet wired up; supported today: \
                 default, fp8. Roadmap: mxint4, int4, int8."
            ))),
        }
        // shared-MLP-quantization: `default` (BF16, force-VRAM) and `fp8`
        // (load-time BF16→E4M3 with per-row FP32 scales) are wired.
        match placement.shared_mlp_quantization {
            Wq::Default | Wq::Fp8 => {}
            other => return Err(AegisError::Unsupported(format!(
                "shared-MLP-quantization={other:?} not yet wired up; supported today: \
                 default, fp8. Roadmap: mxint4, int4, int8."
            ))),
        }
        if graph.num_kv_heads == 0 || !graph.num_attention_heads.is_multiple_of(graph.num_kv_heads) {
            return Err(AegisError::InvalidPlan(format!(
                "CUDA executor requires attention heads divisible by kv heads, got heads={} kv_heads={}",
                graph.num_attention_heads, graph.num_kv_heads
            )));
        }
        let cuda = CudaRuntime::new_with_config(device, cuda_config)?;
        let region_placements = placement.region_map();
        // Pre-size a pinned-host arena to fit every byte that will land
        // in host-resident NVFP4 weights + their `weight_scale` companions
        // + host-resident BF16 matrices (embed/lm_head when configured
        // host-resident). The arena is one big anonymous-mapped pinned
        // allocation; sub-allocated per-tensor by atomic bump. After
        // load, `pin_now()` registers it with `cuMemHostRegister` so
        // staging-pool DMAs from sub-slices take the direct-pinned path.
        let host_arena_capacity = compute_host_arena_capacity(artifact, graph, &region_placements);
        let host_arena = std::sync::Arc::new(
            crate::cuda::host_arena::PinnedArena::new(&cuda, host_arena_capacity)?,
        );
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

        // 3 non-layer regions (embed, final_norm, lm_head) + N layers.
        let progress = std::sync::Arc::new(LoadProgress::new(3 + graph.num_layers));
        // Wire the loader's sub-step sink to the progress indicator so MoE
        // expert loops and other long inner stages emit fine-grained ticks
        // (TTY only — non-TTY skips ticks to keep logs readable).
        let cuda_weights = {
            let p = progress.clone();
            let sink: crate::cuda::LoadStatusSink =
                std::sync::Arc::new(move |label: &str| p.tick(label));
            cuda.weight_loader_with_arena(host_arena.clone()).with_status_sink(sink)
        };
        let load_start = std::time::Instant::now();
        // Per-stage timings: only print as their own lines when stderr is
        // NOT a TTY (i.e. redirected to a log/pipe). On a TTY the progress
        // bar updates in place — adding a fresh `load-timing:` line per
        // stage pushes the bar off-screen on every step. The final
        // `weights total` / `from_artifact total` summaries below still
        // print unconditionally so the totals are visible everywhere.
        let progress_for_stage = progress.clone();
        let stage_t = move |label: &str, t0: std::time::Instant| {
            if !progress_for_stage.is_tty() {
                eprintln!(
                    "load-timing: {label:<22} {:>6.2}s  (cumulative {:>6.2}s)",
                    t0.elapsed().as_secs_f64(),
                    load_start.elapsed().as_secs_f64()
                );
            }
        };

        let embed_name = format!("{}embed_tokens.weight", graph.text_prefix);
        let t0 = std::time::Instant::now();
        let embed_tokens = cuda_weights.load_bf16_matrix_with_store(
            first_existing_tensor(artifact, &[&embed_name, "model.embed_tokens.weight"])?,
            embed_region.store,
            cuda_residency_for_store(embed_region.store, device)?,
            &mut loader,
        )?;
        stage_t("embed", t0);
        progress.step("embed");
        let final_norm_name = format!("{}norm.weight", graph.text_prefix);
        let t0 = std::time::Instant::now();
        let final_norm = cuda_weights.load_dense_vector_with_store(
            first_existing_tensor(artifact, &[&final_norm_name, "model.norm.weight"])?,
            final_norm_region.store,
            &mut loader,
        )?;
        stage_t("final_norm", t0);
        progress.step("final_norm");
        let lm_head_tensor = first_existing_tensor(
            artifact,
            &["lm_head.weight", &embed_name, "model.embed_tokens.weight"],
        )?;
        let t0 = std::time::Instant::now();
        let lm_head = cuda_weights.load_bf16_matrix_with_store(
            lm_head_tensor,
            lm_head_region.store,
            cuda_residency_for_store(lm_head_region.store, device)?,
            &mut loader,
        )?;
        // lm_head is the last VRAM-resident BF16 weight loaded; the
        // layer loop below only fills the host arena. Drop the
        // ~1.4 GiB pinned bounce buffer NOW so it doesn't compete
        // with the growing arena for host RAM during the layer load
        // and push us over the OOM line on memory-tight hosts.
        cuda_weights.release_bounce();
        stage_t("lm_head", t0);
        progress.step("lm_head");

        let mut layers = Vec::with_capacity(graph.num_layers);
        let shared_mlp_q = placement.shared_mlp_quantization;
        let attention_q = placement.attention_quantization;
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
            progress.tick(&format!("layer {layer}: starting"));
            let t0 = std::time::Instant::now();
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
                shared_mlp_q,
                attention_q,
                attention_store_override,
                &mut loader,
            )?);
            stage_t(&format!("layer {layer}"), t0);
            progress.step(&format!("layer {layer}"));
        }
        eprintln!(
            "load-timing: weights total            {:>6.2}s",
            load_start.elapsed().as_secs_f64()
        );

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

        // Fork the post-exit sidecar BEFORE `pin_now()` so the child
        // inherits a tiny, unregistered VMA tree. Forking after the
        // 14 GiB arena is `cuMemHostRegister`'d would force the
        // kernel to walk and CoW-protect every PTE in the pinned
        // range — page-table allocation alone is ~28 MiB at 4 KiB
        // granularity, and on memory-tight hosts the cumulative
        // pressure (pinned arena + KV-cache about-to-be-allocated +
        // page-table duplication) is enough to trip OOM-killer.
        // Doing this before pin_now keeps the child's fork cost flat.
        let shards: std::collections::HashSet<std::path::PathBuf> = artifact
            .tensors
            .tensors
            .values()
            .map(|t| t.shard_path.clone())
            .collect();
        let mut evict_list: std::collections::HashSet<std::path::PathBuf> = shards;
        if let Ok(entries) = std::fs::read_dir(&artifact.root) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    evict_list.insert(path);
                }
            }
        }
        super::cache_cleanup::install(evict_list);

        // All host-resident weights are now in the arena (NVFP4 packed/
        // scales + BF16 host matrices). Page-lock the whole arena with
        // `cuMemHostRegister` so per-token H2D streaming during inference
        // takes the direct-DMA path. The pages are already committed by
        // the load loop's writes, so the registration only locks them in
        // place — no extra physical-memory cost. We deliberately
        // deferred the registration from arena construction to here so
        // the load-time RSS curve grows incrementally with each tensor
        // written, instead of jumping by the full ~14 GiB capacity at
        // once and freezing the desktop on memory-tight hosts.
        let pin_t = std::time::Instant::now();
        host_arena.pin_now()?;
        eprintln!(
            "load-timing: arena pin (cuMemHostRegister) {:>6.2}s",
            pin_t.elapsed().as_secs_f64(),
        );
        // Empty `RegisteredShards` placeholder — the executor still
        // owns the field for type-system reasons but we don't register
        // any shards (host-resident weights are in the arena instead).
        let registered_shards = crate::cuda::registered_shards::RegisteredShards::empty();

        // Drop the loader and our local arena clone now that load is
        // complete. The arena `Arc<PinnedArena>` is still cloned inside
        // every host-resident weight's `HostWeightBytes::Arena { arena, .. }`,
        // so these drops just decrement the refcount — the pinned bytes
        // stay alive for streamed inference. Bookkeeping only.
        drop(cuda_weights);
        drop(host_arena);
        let _ = has_staged_layers;

        // Trim the device's default cudaMallocAsync memory pool back to its
        // live working set. Loading layers in sequence builds up the pool's
        // peak reservation, but most of those allocations are short-lived
        // (host-side tensor buffers, transient repacked weights). Without
        // this trim, the pool retains those blocks indefinitely, inflating
        // nvidia-smi VRAM usage by 500-1000 MiB above what the model
        // actually needs at steady state.
        let mp_before = crate::cuda::runtime::memory::read_mempool_stats(device).ok();
        if crate::cuda::runtime::memory::trim_default_mempool(device, 0).is_ok() {
            if std::env::var("AEGIS_VRAM_BREAKDOWN").is_ok() {
                let mp_after = crate::cuda::runtime::memory::read_mempool_stats(device).ok();
                let mb = |b: usize| b as f64 / (1024.0 * 1024.0);
                if let (Some(b), Some(a)) = (mp_before, mp_after) {
                    eprintln!(
                        "vram-breakdown: mempool BEFORE trim: reserved={:.0} MiB, used={:.0} MiB, cached={:.0} MiB",
                        mb(b.reserved_current), mb(b.used_current), mb(b.cached_bytes()),
                    );
                    eprintln!(
                        "vram-breakdown: mempool AFTER  trim: reserved={:.0} MiB, used={:.0} MiB, cached={:.0} MiB",
                        mb(a.reserved_current), mb(a.used_current), mb(a.cached_bytes()),
                    );
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

        // Per-category VRAM accounting. Keep this debug print until the
        // memory layout stabilises — it's the only honest answer to "where
        // did the GiBs go".
        if std::env::var("AEGIS_VRAM_BREAKDOWN").is_ok() {
            let hd = graph.head_dim;
            let nh = graph.num_attention_heads;
            let nkv = graph.num_kv_heads;
            let nl = graph.num_layers;
            let h = graph.hidden_size;
            let q_w = nh * hd;
            let kv_w = nkv * hd;
            let bf16 = 2usize;
            // Attention Q/K/V/O per layer (BF16, force-VRAM).
            let attn_per_layer = q_w * h * bf16          // q
                               + kv_w * h * bf16          // k
                               + kv_w * h * bf16          // v
                               + h * q_w * bf16;          // o
            let attn_total = attn_per_layer * nl;
            let embed_lm_head = artifact.tensors.tensors.get("model.embed_tokens.weight")
                .map(|t| t.data_len_bytes() as usize).unwrap_or(0);
            // Try to find shared MLP shapes for one layer, scale by nl.
            let shared_per_layer = artifact.tensors.tensors.values()
                .filter(|t: &&aegisllm_base::tensor::core::TensorInfo| {
                    let n = &t.name;
                    n.contains("layers.0.mlp.gate_proj.weight")
                        || n.contains("layers.0.mlp.up_proj.weight")
                        || n.contains("layers.0.mlp.down_proj.weight")
                })
                .map(|t: &aegisllm_base::tensor::core::TensorInfo| t.data_len_bytes() as usize)
                .sum::<usize>();
            let shared_total = shared_per_layer * nl;
            // Routers per layer.
            let router_per_layer = artifact.tensors.tensors.values()
                .filter(|t: &&aegisllm_base::tensor::core::TensorInfo| {
                    let n = &t.name;
                    n.contains("layers.0.mlp.router.weight")
                        || n.contains("layers.0.mlp.router.scale")
                })
                .map(|t: &aegisllm_base::tensor::core::TensorInfo| t.data_len_bytes() as usize)
                .sum::<usize>();
            let router_total = router_per_layer * nl;
            // KV cache: 2 (k+v) × ctx × kv_w per layer × fp16.
            let ctx = placement.kv_cache.context_size;
            let kv_total = 2 * ctx * kv_w * 2 * nl;
            let mb = |b: usize| b as f64 / (1024.0 * 1024.0);
            eprintln!(
                "vram-breakdown: attn(BF16,force-vram)={:.0} MiB | embed+lm_head(BF16)={:.0} MiB | \
                 shared_mlp(BF16,force-vram)={:.0} MiB | routers={:.0} MiB | kv_cache(fp16,ctx={})={:.0} MiB",
                mb(attn_total), mb(embed_lm_head * 2), mb(shared_total),
                mb(router_total), ctx, mb(kv_total),
            );
            eprintln!(
                "vram-breakdown: nl={} hidden={} q_width={} kv_width={} head_dim={}",
                nl, h, q_w, kv_w, hd,
            );
        }
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
            kv_quantization: placement.kv_cache.quantization,
            registered_shards,
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
                // FP8 weight-dequant scratch — sized for the largest single
                // FP8 projection in the model. Allocated only when at least
                // one FP8 weight is present (saves ~46 MiB on BF16-only
                // configs); otherwise zero-length.
                fp8_dequant_scratch: {
                    use crate::executor::state::CudaLinear as CL;
                    let max_fp8_elems = self
                        .layers
                        .iter()
                        .flat_map(|l| {
                            let attn_projs = [&l.q_proj, &l.k_proj, &l.v_proj, &l.o_proj];
                            let shared_iter = l
                                .moe
                                .as_ref()
                                .and_then(|m| m.shared_expert.as_ref())
                                .into_iter()
                                .flat_map(|se| {
                                    [&se.gate_proj, &se.up_proj, &se.down_proj].into_iter()
                                });
                            attn_projs.into_iter().chain(shared_iter)
                        })
                        .filter(|p| matches!(p, CL::Fp8(_)))
                        .map(|p| p.rows() * p.cols())
                        .max()
                        .unwrap_or(0);
                    self.runtime.alloc_u16(max_fp8_elems.max(1))?
                },
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
                        // Permuted MoE scratch. Total assignments per chunk
                        // = chunk_size * top_k (each token routes to top_k
                        // experts). For Gemma-4-26B at chunk=1024, top_k=8:
                        // ~92 MiB input/output, ~23 MiB intermediate/swiglu.
                        expert_offsets: self.runtime.alloc_u32(max_experts + 1)?,
                        permuted_input: self.runtime.alloc_f32(cs * max_top_k * self.hidden_size)?,
                        permuted_intermediate: self
                            .runtime
                            .alloc_f32(cs * max_top_k * max_expert_intermediate)?,
                        permuted_swiglu: self
                            .runtime
                            .alloc_f32(cs * max_top_k * max_expert_intermediate)?,
                        permuted_output: self.runtime.alloc_f32(cs * max_top_k * self.hidden_size)?,
                        // 3-slot grouped staging (gate / up / down each get
                        // their own slot). Allows transfer stream to fill
                        // projection N+1's slot while compute stream's
                        // grouped-GEMM kernel for projection N is still
                        // reading from its own slot. ~143 MiB per slot on
                        // Gemma-4-26B; total ~430 MiB transient VRAM. The
                        // win is ~30% reduction in MoE per-layer time when
                        // H2D and compute would otherwise serialize.
                        bulk_slots: {
                            let max_packed = self
                                .layers
                                .iter()
                                .filter_map(|l| l.moe.as_ref())
                                .flat_map(|m| m.experts.iter())
                                .flat_map(|e| {
                                    [&e.gate_proj, &e.up_proj, &e.down_proj].into_iter()
                                })
                                .map(|p| p.packed_bytes)
                                .max()
                                .unwrap_or(0);
                            let max_scales = self
                                .layers
                                .iter()
                                .filter_map(|l| l.moe.as_ref())
                                .flat_map(|m| m.experts.iter())
                                .flat_map(|e| {
                                    [&e.gate_proj, &e.up_proj, &e.down_proj].into_iter()
                                })
                                .map(|p| p.scale_bytes)
                                .max()
                                .unwrap_or(0);
                            let mut slots = Vec::with_capacity(3);
                            for _ in 0..3 {
                                slots.push(super::state::GroupedStagingSlot {
                                    bulk_packed: self
                                        .runtime
                                        .alloc_u8(max_experts * max_packed.max(1))?,
                                    bulk_scales: self
                                        .runtime
                                        .alloc_u8(max_experts * max_scales.max(1))?,
                                    bulk_packed_offsets: self.runtime.alloc_u32(max_experts)?,
                                    bulk_scales_offsets: self.runtime.alloc_u32(max_experts)?,
                                    bulk_output_scales: self.runtime.alloc_f32(max_experts)?,
                                });
                            }
                            slots.try_into().map_err(|_: Vec<_>| {
                                AegisError::Unsupported(
                                    "internal: bulk staging slot vec→array mismatch".into(),
                                )
                            })?
                        },
                        bulk_token_offsets: self.runtime.alloc_u32(max_experts + 1)?,
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
                    // Per-layer effective KV capacity. Sliding-window layers
                    // need only `window_size` slots (ring buffer); global /
                    // full-attention layers (window_size == 0) need the full
                    // context. For Gemma-4-26B-A4B with 25 sliding layers
                    // (window=1024) + 5 global at ctx=32768, this drops the
                    // KV-cache VRAM from ~7.7 GiB to ~1.5 GiB.
                    let layer_kv_capacity = if layer.window_size > 0 {
                        layer.window_size.min(self.kv_context_size)
                    } else {
                        self.kv_context_size
                    };
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
                            CudaKvCache::dense(
                                &self.runtime,
                                self.kv_context_size,
                                layer_kv_width,
                                self.kv_quantization,
                                layer_kv_capacity,
                            )?
                        }
                        StoragePlacement::Ram | StoragePlacement::Mmap => {
                            CudaKvCache::staged_host(
                                &self.runtime,
                                self.kv_context_size,
                                layer_kv_width,
                                self.kv_quantization,
                            )?
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

/// Sum the bytes that will land in the pinned-host arena: every tensor
/// in a host-resident region (NVFP4 packed + companion `.weight_scale`
/// + the BF16 matrices that go to host RAM, e.g. embed/lm_head when
/// configured `store=ram`).
///
/// Returned value is at least 1 to satisfy `PinnedArena::new`'s non-zero
/// capacity check. When no region is host-resident (everything fits in
/// VRAM) the arena is allocated tiny and never used.
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
        // Only host-resident regions feed the arena.
        if !matches!(placement.store, StoragePlacement::Ram | StoragePlacement::Mmap) {
            continue;
        }
        for graph_tensor in &region.tensors {
            let name = &graph_tensor.info.name;
            // NVFP4 quantised weight: paired `.weight` + `.weight_scale`.
            if let Some(stem) = name.strip_suffix(".weight") {
                let scale_name = format!("{stem}.weight_scale");
                if artifact.tensors.has(&scale_name) {
                    total = total.saturating_add(graph_tensor.info.data_len_bytes() as usize);
                    continue;
                }
                // Unpaired `.weight` (BF16 matrix like embed/lm_head when host-resident).
                if matches!(graph_tensor.info.dtype, aegisllm_base::tensor::TensorDType::BF16) {
                    total = total.saturating_add(graph_tensor.info.data_len_bytes() as usize);
                }
            } else if name.ends_with(".weight_scale") {
                total = total.saturating_add(graph_tensor.info.data_len_bytes() as usize);
            }
            // Scalars (.input_scale, .weight_scale_2) are tiny f32; we
            // overshoot by ignoring them rather than tracking precisely.
        }
    }
    total.max(1)
}

fn cuda_prefill_chunk_size(config: CudaRuntimeConfig) -> usize {
    // Default 2048: matches llama.cpp's typical ubatch=2048. Halves H2D
    // amortization vs chunk=1024 for routed-expert weight streaming. The
    // hdim256 WMMA attention kernel handles the larger Q-tile range
    // correctly (sliding-window ring-buffer KV cache transparently).
    // Slightly slower than chunk=1024 on tiny prompts (≤1024 tokens)
    // but wins at every realistic length.
    config
        .prefill_chunk_size
        .unwrap_or(2048)
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

//! EAGLE/MTP speculative decoding with the Gemma-4 E4B-it-assistant draft.
//!
//! ## What this is
//!
//! The draft (`gemma-4-E4B-it-assistant`) is a 4-layer, hidden=256 **Q-only**
//! decoder. Per draft step it:
//!   1. embeds the current draft token (`embed_tokens`, 256-wide),
//!   2. concatenates `[draft_embed, target_backbone_hidden]` and feeds it
//!      through `pre_projection` → 256-wide draft hidden,
//!   3. runs 4 Q-only decoder layers that attend the TARGET model's per-layer
//!      K/V cache (NO k_proj/v_proj — exactly the `kv_shared_override` path the
//!      target's own shared layers use, but cross-model),
//!   4. RMSNorms (`final_norm`) and:
//!        a. projects back to backbone width (`post_projection`) → the next
//!           step's `target_backbone_hidden`, and
//!        b. runs the centroid-masked sparse LM head → the proposed token.
//!
//! ## Speculative loop (greedy, this pass)
//!
//! The draft proposes `K = num_draft_tokens` tokens autoregressively (each fed
//! into the next draft step). The target then VERIFIES the proposals: it runs
//! one forward per proposed position (over the existing decode KV-share path),
//! takes greedy argmax at each, and accepts the longest prefix where the
//! target's argmax equals the draft's proposal. On the first mismatch the
//! target's own argmax token is emitted and the draft is re-seeded.
//!
//! ## Status
//!
//! Compile-complete, structurally faithful. Verification here uses the proven
//! single-token decode forward per proposed position (NOT a single batched
//! chunked-prefill forward). Both are numerically equivalent for greedy accept;
//! the batched path is a perf optimization deferred to GPU-verify time.
//! TODO(gpu-verify): every numeric detail flagged inline (centroid math,
//! cross-model KV pointer, accept indexing, draft RoPE / q_norm, pre/post
//! projection concat order).

use std::path::Path;

use aegisllm_base::artifact::ModelArtifact;
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::executor::tensors::require_tensor;
use aegisllm_base::generation::{GenerateRequest, SamplingConfig};
use aegisllm_base::planning::placement::StoragePlacement;
use aegisllm_base::tensor::storage::TensorStorageLoader;

use crate::cuda::{CudaWeightLoader, DeviceBuffer};

use super::loader::cuda_residency_for_store;
use super::mlp::DenseActivation;
use super::rope::RopeConfig;
use super::state::{
    CentroidHead, CudaKvCache, CudaLayer, CudaLayerState, CudaLinear, CudaLlamaExecutor,
    CudaLlamaState, CudaScratch, DraftModel, DraftScratch,
};

/// Default number of draft tokens proposed per spec-decode round.
pub(super) fn default_num_draft_tokens() -> usize {
    std::env::var("AEGIS_NUM_DRAFT_TOKENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(4)
}

// ───────────────────────── draft numeric-trace dump ──────────────────────────
//
// WORKSTREAM A (draft-trace): when `AEGIS_DRAFT_TRACE=<dir>` is set, the FIRST
// `draft_step` call (round 0, step 0) writes raw little-endian f32 `.bin` of
// every intermediate draft hidden state into `<dir>`. Paired with the vLLM
// gemma4_mtp.py `GEMMA4_MTP_TRACE` dump and `bench/draft_trace_compare.py`,
// which cosine/max-rel diffs each stage and prints the FIRST one that diverges
// (localizing the numeric bug: pre_projection vs a specific layer vs
// attention-within-a-layer). NO-op unless the env var is set.

thread_local! {
    /// Counts `draft_step` invocations on this thread. The dump fires only when
    /// this is 0 (the very first draft step = round 0, step 0).
    static DRAFT_STEP_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Returns `Some(dir)` only for round-0/step-0 when `AEGIS_DRAFT_TRACE` is set,
/// and increments the per-thread step counter. Call EXACTLY once at the top of
/// `draft_step`.
fn draft_trace_dir_first_step() -> Option<String> {
    let n = DRAFT_STEP_CALLS.with(|c| {
        let v = c.get();
        c.set(v + 1);
        v
    });
    if n != 0 {
        return None;
    }
    std::env::var("AEGIS_DRAFT_TRACE").ok()
}

/// Download a device f32 buffer's first `len` elements and write them as raw
/// little-endian f32 to `<dir>/<name>.bin`. No-op when `dir` is None. Mirrors
/// `audio::dump_stage` but downloads from device first and truncates to `len`
/// (device scratch buffers are often over-allocated past the logical width).
fn draft_dump_device(
    dir: &Option<String>,
    name: &str,
    rt: &crate::cuda::CudaRuntime,
    buf: &DeviceBuffer<f32>,
    len: usize,
) {
    let Some(dir) = dir else { return };
    let host = match rt.download_f32(buf) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("[draft-trace] download {name} failed: {e}");
            return;
        }
    };
    let n = len.min(host.len());
    draft_dump_slice(&Some(dir.clone()), name, &host[..n]);
}

/// Write a host f32 slice as raw little-endian f32 to `<dir>/<name>.bin`.
fn draft_dump_slice(dir: &Option<String>, name: &str, data: &[f32]) {
    let Some(dir) = dir else { return };
    let _ = std::fs::create_dir_all(dir);
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let path = format!("{dir}/{name}.bin");
    if let Err(e) = std::fs::write(&path, &bytes) {
        eprintln!("[draft-trace] failed to write {path}: {e}");
    } else {
        eprintln!("[draft-trace] wrote {path} ({} f32)", data.len());
    }
}

/// Raw fields parsed from the draft's `config.json`. The draft uses a
/// `gemma4_assistant` model_type whose nested `text_config` carries the
/// decoder shape; the top-level carries the centroid head config. We parse
/// the raw JSON directly rather than extending `HfConfig` (the assistant is a
/// self-contained second model the engine never plans/places).
#[derive(Debug, Clone)]
struct DraftConfig {
    draft_hidden: usize,
    intermediate: usize,
    num_attention_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    global_head_dim: usize,
    num_hidden_layers: usize,
    sliding_window: usize,
    rms_norm_eps: f32,
    /// `text_config.rope_parameters.full_attention.partial_rotary_factor`.
    partial_rotary_factor: f32,
    rope_theta_sliding: f32,
    rope_theta_global: f32,
    layer_types: Vec<String>,
    backbone_hidden: usize,
    num_centroids: usize,
    centroid_top_k: usize,
    vocab_size: usize,
    hidden_activation: String,
}

fn parse_draft_config(root: &Path) -> Result<DraftConfig> {
    let path = root.join("config.json");
    let bytes = std::fs::read(&path).map_err(|e| {
        AegisError::InvalidConfig(format!("draft: cannot read {}: {e}", path.display()))
    })?;
    let json: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| {
        AegisError::InvalidConfig(format!("draft: bad config.json: {e}"))
    })?;
    let tc = json
        .get("text_config")
        .ok_or_else(|| AegisError::InvalidConfig("draft config missing text_config".into()))?;
    let g_usize = |v: &serde_json::Value, k: &str| -> Option<usize> {
        v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize)
    };
    let g_f32 = |v: &serde_json::Value, k: &str| -> Option<f32> {
        v.get(k).and_then(|x| x.as_f64()).map(|x| x as f32)
    };
    let req_usize = |v: &serde_json::Value, k: &str| -> Result<usize> {
        g_usize(v, k).ok_or_else(|| {
            AegisError::InvalidConfig(format!("draft text_config missing `{k}`"))
        })
    };

    // RoPE per-attention-type theta + partial factor (mirrors target loader).
    let rope = tc.get("rope_parameters");
    let rope_theta_sliding = rope
        .and_then(|r| r.get("sliding_attention"))
        .and_then(|s| g_f32(s, "rope_theta"))
        .unwrap_or(10_000.0);
    let rope_theta_global = rope
        .and_then(|r| r.get("full_attention"))
        .and_then(|s| g_f32(s, "rope_theta"))
        .unwrap_or(1_000_000.0);
    let partial_rotary_factor = rope
        .and_then(|r| r.get("full_attention"))
        .and_then(|s| g_f32(s, "partial_rotary_factor"))
        .unwrap_or(0.25);

    let layer_types = tc
        .get("layer_types")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    Ok(DraftConfig {
        draft_hidden: req_usize(tc, "hidden_size")?,
        intermediate: req_usize(tc, "intermediate_size")?,
        num_attention_heads: req_usize(tc, "num_attention_heads")?,
        num_kv_heads: req_usize(tc, "num_key_value_heads")?,
        head_dim: req_usize(tc, "head_dim")?,
        global_head_dim: g_usize(tc, "global_head_dim").unwrap_or_else(|| {
            g_usize(tc, "head_dim").unwrap_or(256)
        }),
        num_hidden_layers: req_usize(tc, "num_hidden_layers")?,
        sliding_window: g_usize(tc, "sliding_window").unwrap_or(512),
        rms_norm_eps: g_f32(tc, "rms_norm_eps").unwrap_or(1e-6),
        partial_rotary_factor,
        rope_theta_sliding,
        rope_theta_global,
        layer_types,
        backbone_hidden: g_usize(&json, "backbone_hidden_size").unwrap_or(2560),
        num_centroids: g_usize(&json, "num_centroids").unwrap_or(2048),
        centroid_top_k: g_usize(&json, "centroid_intermediate_top_k").unwrap_or(32),
        vocab_size: g_usize(tc, "vocab_size").unwrap_or(262144),
        hidden_activation: tc
            .get("hidden_activation")
            .and_then(|v| v.as_str())
            .unwrap_or("gelu_pytorch_tanh")
            .to_string(),
    })
}

/// Whether draft layer `idx` is a global (full-attention) layer.
fn layer_is_global(cfg: &DraftConfig, idx: usize) -> bool {
    cfg.layer_types
        .get(idx)
        .map(|t| t == "full_attention")
        .unwrap_or_else(|| {
            // Fallback: Gemma-4 pattern — last layer is always global.
            idx + 1 == cfg.num_hidden_layers
        })
}

/// Build a single Q-only draft `CudaLayer`. The draft checkpoint stores
/// `self_attn.q_proj` + `self_attn.q_norm` + `self_attn.o_proj` + the dense
/// MLP + the pre/post/input/feedforward norms + a per-layer `layer_scalar`,
/// but NO k_proj/v_proj/k_norm. We load q/o/mlp/norms for real and stub the
/// k/v slots (never invoked because the attention path always takes the
/// `kv_shared_override` branch for the draft).
#[allow(clippy::too_many_arguments)]
fn load_draft_layer(
    cuda: &CudaWeightLoader<'_>,
    artifact: &ModelArtifact,
    layer: usize,
    cfg: &DraftConfig,
    rope_base: &RopeConfig,
    loader: &mut TensorStorageLoader,
) -> Result<CudaLayer> {
    let device = cuda.device_index();
    let store = StoragePlacement::Vram { device };
    let residency = cuda_residency_for_store(store, device)?;
    let prefix = format!("model.layers.{layer}");

    let is_global = layer_is_global(cfg, layer);
    let layer_head_dim = if is_global { cfg.global_head_dim } else { cfg.head_dim };
    let q_width = cfg.num_attention_heads * layer_head_dim;
    let window_size = if is_global { 0 } else { cfg.sliding_window };

    // partial RoPE dim — only global layers use p-RoPE (matches target loader).
    let partial_dim = if is_global && cfg.partial_rotary_factor < 1.0 {
        (cfg.partial_rotary_factor as f64 * layer_head_dim as f64).round() as usize
    } else {
        0
    };
    let theta_override = if is_global {
        Some(cfg.rope_theta_global)
    } else {
        Some(cfg.rope_theta_sliding)
    };
    let rope = rope_base.to_device_with_partial_dim_and_theta(partial_dim, theta_override)?;

    let q_proj = cuda
        .load_bf16_matrix_with_store(
            require_tensor(artifact, &format!("{prefix}.self_attn.q_proj.weight"))?,
            store,
            residency.clone(),
            loader,
        )
        .map(CudaLinear::Bf16)?;
    let o_proj = cuda
        .load_bf16_matrix_with_store(
            require_tensor(artifact, &format!("{prefix}.self_attn.o_proj.weight"))?,
            store,
            residency.clone(),
            loader,
        )
        .map(CudaLinear::Bf16)?;
    let q_norm_weight = Some(cuda.load_dense_vector_with_store(
        require_tensor(artifact, &format!("{prefix}.self_attn.q_norm.weight"))?,
        store,
        loader,
    )?);

    // Stub k/v projections — never invoked (kv_shared_override is always Some
    // for the draft). 1-element NVFP4 dummies keep the CudaLinear enum happy.
    let k_proj = CudaLinear::Nvfp4(cuda.alloc_dummy_nvfp4_linear(&format!("{prefix}.k_proj.stub"))?);
    let v_proj = CudaLinear::Nvfp4(cuda.alloc_dummy_nvfp4_linear(&format!("{prefix}.v_proj.stub"))?);

    // Dense MLP (GeGLU-tanh).
    let gate_proj = cuda
        .load_bf16_matrix_with_store(
            require_tensor(artifact, &format!("{prefix}.mlp.gate_proj.weight"))?,
            store,
            residency.clone(),
            loader,
        )
        .map(CudaLinear::Bf16)?;
    let up_proj = cuda
        .load_bf16_matrix_with_store(
            require_tensor(artifact, &format!("{prefix}.mlp.up_proj.weight"))?,
            store,
            residency.clone(),
            loader,
        )
        .map(CudaLinear::Bf16)?;
    let down_proj = cuda
        .load_bf16_matrix_with_store(
            require_tensor(artifact, &format!("{prefix}.mlp.down_proj.weight"))?,
            store,
            residency.clone(),
            loader,
        )
        .map(CudaLinear::Bf16)?;

    // Norms. Gemma-4 PrePost layout (same as target E4B):
    //   input_layernorm           → input_norm_weight (pre-attn)
    //   post_attention_layernorm  → post_attn_sublayer_norm (post-attn, pre-residual)
    //   pre_feedforward_layernorm → post_attention_norm_weight (pre-MLP)
    //   post_feedforward_layernorm→ post_mlp_sublayer_norm (post-MLP, pre-residual)
    let input_norm_weight = cuda.load_dense_vector_with_store(
        require_tensor(artifact, &format!("{prefix}.input_layernorm.weight"))?,
        store,
        loader,
    )?;
    let post_attention_norm_weight = cuda.load_dense_vector_with_store(
        require_tensor(artifact, &format!("{prefix}.pre_feedforward_layernorm.weight"))?,
        store,
        loader,
    )?;
    let post_attn_sublayer_norm = Some(cuda.load_dense_vector_with_store(
        require_tensor(artifact, &format!("{prefix}.post_attention_layernorm.weight"))?,
        store,
        loader,
    )?);
    let post_mlp_sublayer_norm = Some(cuda.load_dense_vector_with_store(
        require_tensor(artifact, &format!("{prefix}.post_feedforward_layernorm.weight"))?,
        store,
        loader,
    )?);

    let layer_scalar = {
        use crate::cuda::loader::read_scalar_f32_with_loader;
        artifact
            .tensors
            .get(&format!("{prefix}.layer_scalar"))
            .map(|t| read_scalar_f32_with_loader(loader, t, store))
            .transpose()?
    };

    let dense_activation = match cfg.hidden_activation.as_str() {
        "silu" | "swiglu" => DenseActivation::Swiglu,
        "gelu_pytorch_tanh" | "gelu_tanh" | "gelu" => DenseActivation::GeluTanh,
        other => {
            return Err(AegisError::InvalidPlan(format!(
                "draft: unsupported hidden_activation `{other}`"
            )))
        }
    };

    let _ = q_width;
    Ok(CudaLayer {
        input_norm_weight,
        post_attention_norm_weight,
        post_attn_sublayer_norm,
        post_mlp_sublayer_norm,
        post_feedforward_layernorm_1: None,
        pre_feedforward_layernorm_2: None,
        post_feedforward_layernorm_2: None,
        layer_scalar,
        q_proj,
        k_proj,
        v_proj,
        qkv_proj: None,
        o_proj,
        q_norm_weight,
        // No k_norm on the draft (Q-only). The kv_shared_override branch skips
        // k_norm anyway, but leave None for clarity.
        k_norm_weight: None,
        gate_proj,
        up_proj,
        down_proj,
        dense_activation,
        window_size,
        rope,
        moe: None,
        layer_head_dim,
        layer_num_kv_heads: cfg.num_kv_heads,
        ple: None,
        // The draft NEVER uses its own KV cache field; cross-model KV sharing
        // is resolved at forward time against the TARGET's layer states (see
        // `DraftModel::target_kv_layer`). We always pass kv_shared_override.
        kv_shared_from: None,
        gdn: None,
        attn_output_gate: false,
    })
}

/// Resolve, for each draft layer, the TARGET layer index whose K/V cache the
/// draft layer attends. The draft's attention hyperparameters match the
/// target's text config, so a draft sliding layer reads the target's most
/// recent KV-owning sliding layer and a draft global layer reads the target's
/// most recent KV-owning global layer.
///
/// "KV-owning" = a target layer that actually allocated a KV buffer (i.e. its
/// own `kv_shared_from` is None). The target's shared (tail) layers are stubs.
///
/// TODO(gpu-verify): the vLLM Gemma4 MTP reference ties the draft to the
/// target's LAST layer's K/V (a single global cache). Confirm whether the draft
/// sliding layers should read the target's last sliding cache or the same last
/// global cache. Here we match by attention type, which is the conservative
/// interpretation; if the reference uses last-global for all draft layers,
/// change the sliding branch to point at the global parent too.
fn resolve_target_kv_layers(
    exec: &CudaLlamaExecutor,
    cfg: &DraftConfig,
) -> Result<Vec<usize>> {
    let n = exec.layers.len();
    // Classify each target layer as global vs sliding from its window_size
    // (global layers are full-attention → window_size == 0).
    let target_is_global: Vec<bool> = exec.layers.iter().map(|l| l.window_size == 0).collect();
    let owns_kv: Vec<bool> = exec.layers.iter().map(|l| l.kv_shared_from.is_none()).collect();

    let most_recent = |want_global: bool| -> Option<usize> {
        (0..n)
            .rev()
            .find(|&i| owns_kv[i] && target_is_global[i] == want_global)
    };
    let global_parent = most_recent(true);
    let sliding_parent = most_recent(false);

    let mut out = Vec::with_capacity(cfg.num_hidden_layers);
    for li in 0..cfg.num_hidden_layers {
        let want_global = layer_is_global(cfg, li);
        let parent = if want_global { global_parent } else { sliding_parent };
        let parent = parent.ok_or_else(|| {
            AegisError::InvalidPlan(format!(
                "draft layer {li}: no target {} KV-owning parent layer found",
                if want_global { "global" } else { "sliding" }
            ))
        })?;
        out.push(parent);
    }
    Ok(out)
}

/// Build the draft-only `CudaScratch` sized to the draft's (small) widths.
/// No MoE, no PLE, no staging, no CUTLASS — the draft is dense BF16.
fn build_draft_scratch(exec: &CudaLlamaExecutor, draft: &DraftModel) -> Result<CudaScratch> {
    let rt = &exec.runtime;
    let hidden = draft.draft_hidden_for_scratch;
    let num_heads = draft.num_attention_heads;
    let max_head_dim = draft
        .layers
        .iter()
        .map(|l| l.layer_head_dim)
        .max()
        .unwrap_or(hidden);
    let max_num_kv_heads = draft
        .layers
        .iter()
        .map(|l| l.layer_num_kv_heads)
        .max()
        .unwrap_or(1);
    let max_q_width = num_heads * max_head_dim;
    let max_kv_width = max_num_kv_heads * max_head_dim;
    let intermediate = draft.intermediate_for_scratch;
    use crate::cuda::{DECODE_SPLIT_K_MAX, CudaRuntime};
    Ok(CudaScratch {
        input_normed: rt.alloc_f32(hidden)?,
        quant_hidden: rt.alloc_f32(hidden)?,
        quant_intermediate: rt.alloc_f32(intermediate)?,
        mxfp4_hidden: rt.alloc_u8(CudaRuntime::mxfp4_vector_bytes(hidden)?)?,
        mxfp4_intermediate: rt.alloc_u8(CudaRuntime::mxfp4_vector_bytes(intermediate)?)?,
        cutlass_payload: rt.alloc_u8(1)?,
        cutlass_scales: rt.alloc_u8(1)?,
        cutlass_workspace: rt.alloc_u8(1)?,
        q: rt.alloc_f32(max_q_width)?,
        k: rt.alloc_f32(max_kv_width)?,
        v: rt.alloc_f32(max_kv_width)?,
        qk_norm_scratch: rt.alloc_f32(max_q_width.max(max_kv_width))?,
        attn_split_acc: rt.alloc_f32(num_heads * DECODE_SPLIT_K_MAX * max_head_dim)?,
        attn_split_m: rt.alloc_f32(num_heads * DECODE_SPLIT_K_MAX)?,
        attn_split_l: rt.alloc_f32(num_heads * DECODE_SPLIT_K_MAX)?,
        attn_context: rt.alloc_f32(max_q_width)?,
        attn_out: rt.alloc_f32(hidden)?,
        residual: rt.alloc_f32(hidden)?,
        post_normed: rt.alloc_f32(hidden)?,
        gate: rt.alloc_f32(intermediate)?,
        up: rt.alloc_f32(intermediate)?,
        swiglu: rt.alloc_f32(intermediate)?,
        mlp_out: rt.alloc_f32(hidden)?,
        hidden_out: rt.alloc_f32(hidden)?,
        final_hidden: rt.alloc_f32(hidden)?,
        argmax_block_values: rt.alloc_f32(1)?,
        argmax_block_indices: rt.alloc_u32(1)?,
        moe: None,
        // GDN decode scratch — built from the draft's GDN dims if it has any
        // (current EAGLE/MTP drafts are plain attention, so this is None).
        gdn_decode: match draft.layers.iter().find_map(|l| l.gdn.as_ref().map(|g| g.dims)) {
            Some(dims) => Some(Box::new(super::gdn::GdnDecodeScratch::new(rt, dims, hidden)?)),
            None => None,
        },
        staging_pool: None,
        kv_staging: None,
        per_layer_inputs: rt.alloc_f32(1)?,
        ple_projection: rt.alloc_f32(1)?,
        ple_projection_normed: rt.alloc_f32(1)?,
        ple_gate: rt.alloc_f32(1)?,
        ple_contrib: rt.alloc_f32(1)?,
        ple_contrib_normed: rt.alloc_f32(1)?,
        ple_bf16_in: rt.alloc_u16(1)?,
        ple_bf16_out: rt.alloc_u16(1)?,
    })
}

/// Load the EAGLE/MTP draft model. All weights VRAM-resident (~135 MiB).
pub(super) fn load_draft_model(
    exec: &CudaLlamaExecutor,
    draft_path: &Path,
) -> Result<DraftModel> {
    let artifact = ModelArtifact::from_local_path(draft_path)?;
    let cfg = parse_draft_config(draft_path)?;

    // Sanity: draft attention hyperparams must let the draft Q index the
    // target K/V (same KV-head count + head_dim per attention type).
    if cfg.num_kv_heads != exec.num_kv_heads {
        return Err(AegisError::InvalidPlan(format!(
            "draft num_kv_heads ({}) must match target ({}) for cross-model KV share",
            cfg.num_kv_heads, exec.num_kv_heads
        )));
    }

    let device = exec.runtime.device_index();
    let store = StoragePlacement::Vram { device };
    let residency = cuda_residency_for_store(store, device)?;
    let host_arena = std::sync::Arc::new(crate::cuda::host_arena::PinnedArena::new(
        &exec.runtime,
        // Draft is all VRAM-resident, so the arena is only needed for the
        // loader's bounce buffer; size it to the largest single tensor
        // (embed_tokens 262144 × 256 BF16 = 134 MiB).
        cfg.vocab_size * cfg.draft_hidden * 2,
    )?);
    let cuda = exec.runtime.weight_loader_with_arena(host_arena.clone());
    let mut loader = TensorStorageLoader::new();

    // pre_projection [draft_hidden, 2*backbone_hidden], post_projection
    // [backbone_hidden, draft_hidden]. Both BF16, VRAM-resident.
    let pre_projection = cuda.load_bf16_matrix_with_store(
        require_tensor(&artifact, "pre_projection.weight")?,
        store,
        residency.clone(),
        &mut loader,
    )?;
    let post_projection = cuda.load_bf16_matrix_with_store(
        require_tensor(&artifact, "post_projection.weight")?,
        store,
        residency.clone(),
        &mut loader,
    )?;
    let embed_tokens = cuda.load_bf16_matrix_with_store(
        require_tensor(&artifact, "model.embed_tokens.weight")?,
        store,
        residency.clone(),
        &mut loader,
    )?;
    let final_norm = cuda.load_dense_vector_with_store(
        require_tensor(&artifact, "model.norm.weight")?,
        store,
        &mut loader,
    )?;

    // Centroid head.
    let centroids = cuda.load_bf16_matrix_with_store(
        require_tensor(&artifact, "masked_embedding.centroids.weight")?,
        store,
        residency.clone(),
        &mut loader,
    )?;
    // token_ordering: i64 [vocab]. Load to host as u32 (token ids fit in u32).
    let token_ordering = load_token_ordering(&artifact, store, &mut loader)?;

    // Decoder layers.
    let rope_base = RopeConfig::from_artifact(&artifact);
    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for li in 0..cfg.num_hidden_layers {
        layers.push(load_draft_layer(&cuda, &artifact, li, &cfg, &rope_base, &mut loader)?);
    }

    let target_kv_layer = resolve_target_kv_layers(exec, &cfg)?;

    // Pin the (tiny) arena so the loader's host-resident path (unused here, but
    // safe) and any bounce DMAs are page-locked.
    host_arena.pin_now()?;
    drop(cuda);
    drop(host_arena);

    // Candidate-set upper bound: top_k centroids each owning a vocab slice.
    // With uniform partitioning each centroid owns ceil(vocab / num_centroids)
    // tokens; cap total candidates at top_k * that.
    let per_centroid = cfg.vocab_size.div_ceil(cfg.num_centroids);
    let max_candidates = (cfg.centroid_top_k * per_centroid).min(cfg.vocab_size).max(1);

    Ok(DraftModel {
        pre_projection,
        post_projection,
        embed_tokens,
        final_norm,
        layers,
        target_kv_layer,
        centroid_head: CentroidHead {
            centroids,
            token_ordering,
            num_centroids: cfg.num_centroids,
            top_k: cfg.centroid_top_k,
            vocab_size: cfg.vocab_size,
        },
        draft_hidden: cfg.draft_hidden,
        backbone_hidden: cfg.backbone_hidden,
        num_attention_heads: cfg.num_attention_heads,
        rms_norm_eps: cfg.rms_norm_eps,
        draft_hidden_for_scratch: cfg.draft_hidden,
        intermediate_for_scratch: cfg.intermediate,
        num_centroids: cfg.num_centroids,
        max_candidates,
    })
}

/// Build the per-sequence `DraftState` (scratch) for a draft model. Called by
/// the executor's spec-decode state allocator. Sized to the draft's small
/// widths (256 hidden, 2048 intermediate).
pub(super) fn build_draft_state(exec: &CudaLlamaExecutor) -> Result<super::state::DraftState> {
    let draft = exec
        .draft
        .as_ref()
        .ok_or_else(|| AegisError::InvalidPlan("build_draft_state without a draft model".into()))?;
    let rt = &exec.runtime;
    let scratch = DraftScratch {
        pre_proj_input: rt.alloc_f32(draft.pre_projection.cols)?,
        draft_embed: rt.alloc_f32(draft.backbone_hidden)?,
        hidden: rt.alloc_f32(draft.draft_hidden)?,
        final_hidden: rt.alloc_f32(draft.draft_hidden)?,
        backbone_out: rt.alloc_f32(draft.backbone_hidden)?,
        centroid_scores: rt.alloc_f32(draft.num_centroids)?,
        candidate_rows: rt.alloc_u32(draft.max_candidates)?,
        candidate_logits: rt.alloc_f32(draft.max_candidates)?,
    };
    let decoder_scratch = build_draft_scratch(exec, draft)?;
    // 1-slot stub KV for the unused `&mut layer_state` arg (override branch
    // never touches it). F16, ctx=1, kv_width=1.
    let dummy_kv = CudaKvCache::dense(
        rt,
        1,
        1,
        aegisllm_base::tensor::quant::KvCacheQuantization::F16,
        1,
        false,
    )?;
    Ok(super::state::DraftState {
        scratch,
        decoder_scratch,
        dummy_layer_state: CudaLayerState { kv: dummy_kv, recurrent: None, conv_state: None },
    })
}

/// Load `masked_embedding.token_ordering` (i64 [vocab]) into a host `Vec<u32>`.
fn load_token_ordering(
    artifact: &ModelArtifact,
    store: StoragePlacement,
    loader: &mut TensorStorageLoader,
) -> Result<Vec<u32>> {
    use aegisllm_base::tensor::TensorDType;
    let tensor = require_tensor(artifact, "masked_embedding.token_ordering")?;
    let loaded = loader.load_for_store(tensor, store)?;
    let bytes = loaded.as_bytes();
    let out = match tensor.dtype {
        TensorDType::I64 => bytes
            .chunks_exact(8)
            .map(|c| {
                let v = i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
                v as u32
            })
            .collect::<Vec<u32>>(),
        TensorDType::I32 => bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<u32>>(),
        other => {
            return Err(AegisError::InvalidPlan(format!(
                "draft token_ordering must be i64/i32, got {other:?}"
            )))
        }
    };
    Ok(out)
}


// ───────────────────────── draft forward + spec loop ─────────────────────────

impl CudaLlamaExecutor {
    /// True when a draft model is attached (spec-decode available).
    pub(super) fn has_draft(&self) -> bool {
        self.draft.is_some()
    }

    /// Allocate a `CudaLlamaState` with the draft scratch attached. Wraps the
    /// normal `new_state()` and fills `state.draft` when a draft is present.
    pub(super) fn new_spec_state(&self) -> Result<CudaLlamaState> {
        let mut state = self.new_state()?;
        if self.draft.is_some() {
            state.draft = Some(Box::new(build_draft_state(self)?));
        }
        if self.mtp.is_some() {
            state.mtp = Some(Box::new(super::mtp::build_mtp_state(self)?));
        }
        Ok(state)
    }

    /// Run one draft step. `state.hidden` holds the current backbone hidden
    /// (target's last hidden on the first step; the previous draft step's
    /// `post_projection` output on later steps). Produces the proposed token
    /// (greedy argmax over the centroid-masked sparse head) and writes the next
    /// step's backbone hidden into `state.draft.scratch.backbone_out`.
    fn draft_step(
        &self,
        state: &mut CudaLlamaState,
        token: usize,
        draft_position: usize,
    ) -> Result<usize> {
        let draft = self
            .draft
            .as_ref()
            .ok_or_else(|| AegisError::InvalidPlan("draft_step without a draft model".into()))?;
        let rt = &self.runtime;
        let eps = draft.rms_norm_eps;
        let bb = draft.backbone_hidden;
        let dh = draft.draft_hidden;

        // WORKSTREAM A (draft-trace): active only on round-0/step-0 when
        // AEGIS_DRAFT_TRACE=<dir> is set. Increments the per-thread call counter
        // exactly once here.
        let trace = draft_trace_dir_first_step();

        // Borrow disjoint state fields.
        let CudaLlamaState {
            hidden,
            layers,
            decode_position,
            decode_seq_len,
            draft: draft_state_field,
            ..
        } = state;
        let dstate = draft_state_field
            .as_mut()
            .ok_or_else(|| AegisError::InvalidPlan("draft_step without draft state".into()))?;

        // 1. Token embedding (TARGET backbone embedding, backbone-wide).
        //    TODO(gpu-verify): confirm the pre_projection token-embed half is
        //    the TARGET embed_tokens and whether the Gemma sqrt(hidden) embed
        //    scale belongs here.
        // TRACE: the target feedback hidden (state.hidden) is the conditioning
        // input to pre_projection's second half — dump it BEFORE it is consumed.
        draft_dump_device(&trace, "input_state_hidden", rt, hidden, bb);
        rt.bf16_row_to_f32_device(&self.embed_tokens, token, &mut dstate.scratch.draft_embed)?;
        if let Some(scale) = self.embed_scale {
            rt.scale_f32_device(scale, &mut dstate.scratch.draft_embed)?;
        }
        // TRACE: the (target-embed, scaled) draft token embedding — first half of
        // the pre_projection input. vLLM `inputs_embeds` (= embed_tokens*normalizer).
        draft_dump_device(&trace, "draft_embed", rt, &dstate.scratch.draft_embed, bb);

        // 2. pre_projection input = concat(token_embed, target_hidden).
        rt.copy_f32_d2d_range(&dstate.scratch.draft_embed, 0, &mut dstate.scratch.pre_proj_input, 0, bb)?;
        rt.copy_f32_d2d_range(hidden, 0, &mut dstate.scratch.pre_proj_input, bb, bb)?;
        // TRACE: the 5120-wide concat fed into pre_projection (vLLM `combined`).
        draft_dump_device(&trace, "pre_proj_input", rt, &dstate.scratch.pre_proj_input, 2 * bb);
        rt.matvec_bf16_reference_device(
            &draft.pre_projection,
            &dstate.scratch.pre_proj_input,
            &mut dstate.scratch.hidden,
        )?;
        // TRACE: pre_projection output = the initial draft hidden (vLLM
        // `hidden_states` post `pre_projection`, before any layer).
        draft_dump_device(&trace, "pre_proj_out", rt, &dstate.scratch.hidden, dh);

        // 3. Decode position/seq_len, then the 4 Q-only layers vs target KV.
        //
        // TODO(gpu-verify): seq_len off-by-one. The draft is Q-ONLY — it never
        // writes its own K/V — so the target KV slot at `draft_position` is NOT
        // populated at draft time (the target only writes it later, during
        // verification, when it forwards this token). The attention kernel
        // attends keys at positions [0, seq_len). For a draft token whose query
        // sits at `draft_position`, the valid (already-written) target KV spans
        // positions [0, draft_position), i.e. seq_len SHOULD be `draft_position`
        // (= base_pos + k), NOT `draft_position + 1`. Using `+1` here makes the
        // draft attend one KV slot of stale/garbage ring-buffer data. We keep
        // `+1` to mirror the normal decode convention (where the current token's
        // KV *is* written before attention) but this is the single most likely
        // numeric bug in the spec path; verify against a reference EAGLE trace
        // and change to `draft_position` if the draft must exclude the
        // not-yet-written current position.
        // Constant seq_len = draft_position (the last target position). The draft
        // is Q-only and never writes the slot at `draft_position`, so it attends
        // the target KV span [0, draft_position) — the fully-written committed
        // context. (Was draft_position + 1, which attended one not-yet-written
        // stale ring-buffer slot.) Paired with the constant-position change above.
        let seq_len = draft_position;
        rt.copy_u32_to_device(&[draft_position as u32], decode_position)?;
        rt.copy_u32_to_device(&[seq_len as u32], decode_seq_len)?;
        draft_forward_layers(
            self, draft, dstate, layers, decode_position, decode_seq_len, draft_position, seq_len,
            &trace,
        )?;

        // 4. final_norm → final_hidden, then post_projection + sparse head.
        rt.rms_norm_device(&dstate.scratch.hidden, &draft.final_norm, eps, &mut dstate.scratch.final_hidden)?;
        // TRACE: final-normed draft hidden (vLLM `draft_hidden_states` = norm output).
        draft_dump_device(&trace, "final_hidden", rt, &dstate.scratch.final_hidden, dh);
        rt.matvec_bf16_reference_device(
            &draft.post_projection,
            &dstate.scratch.final_hidden,
            &mut dstate.scratch.backbone_out,
        )?;
        // TRACE: post_projection output, the backbone-width feedback for the next
        // step (vLLM `backbone_hidden_states`).
        draft_dump_device(&trace, "backbone_out", rt, &dstate.scratch.backbone_out, bb);
        draft_sparse_head_argmax(rt, draft, dstate)
    }

    /// Greedy speculative-decoding generation. Mirrors
    /// `generate_with_backend_timed`'s control flow (prefill → decode) but,
    /// after each accepted token, runs a draft round that proposes
    /// `num_draft_tokens` continuations and verifies them in the target.
    /// Returns the produced token ids (caller decodes to text).
    ///
    /// EXACT-MATCH greedy only this pass.
    ///
    /// KV bookkeeping: verification runs ONE proven `forward_hidden` per
    /// proposed position and STOPS at the first mismatch, so the target KV is
    /// only ever written for positions that are accepted (or the single
    /// mismatch-corrected token) — strictly in order. Rejected proposals never
    /// reach a target forward, so there is NO speculative KV to roll back and
    /// `state.position` is always exactly `base_pos + (#forwards)`. A future
    /// batched-verify optimization (one forward over all K proposals) WOULD
    /// write K positions up front and need an explicit position/KV rewind on
    /// rejection — that variant is deferred.
    ///
    /// TODO(gpu-verify): temp>0 rejection sampling deferred; for the batched
    /// (non-per-token) verify path, the accept/reject position rewind semantics
    /// need a reference trace. The per-token path used here needs no rewind.
    pub(super) fn generate_speculative_greedy(
        &self,
        state: &mut CudaLlamaState,
        prompt_tokens: &[usize],
        request: &GenerateRequest,
        is_eos: &dyn Fn(usize) -> bool,
    ) -> Result<Vec<usize>> {
        let greedy = SamplingConfig { temperature: 0.0, top_k: 1, top_p: 1.0, min_p: 0.0 };
        let num_draft = self.num_draft_tokens.max(1);
        let spec_stats = std::env::var("AEGIS_SPEC_STATS").is_ok();
        let (mut stat_rounds, mut stat_proposed, mut stat_accepted) = (0usize, 0usize, 0usize);
        let (mut stat_draft_us, mut stat_verify_us) = (0u64, 0u64);
        let stat_t0 = std::time::Instant::now();

        let mut next = self.prefill_prompt(state, prompt_tokens, &greedy)?;
        let mut generated: Vec<usize> = Vec::new();

        'outer: while generated.len() < request.max_tokens {
            if is_eos(next) {
                break;
            }
            if request.stop_token_ids.contains(&next) {
                generated.push(next);
                break;
            }
            generated.push(next);
            if generated.len() >= request.max_tokens {
                break;
            }

            // ── Draft round: propose `num_draft` tokens autoregressively. ──
            // The first draft input is `next`; its backbone hidden is
            // state.hidden (set by the target forward that produced `next`).
            // Each draft step writes the next backbone hidden into
            // scratch.backbone_out; we copy it into state.hidden for the
            // following draft step's pre_projection. state.hidden is restored
            // by the verifying target forwards below (they recompute it).
            let base_pos = state.position;
            let mut proposals: Vec<usize> = Vec::with_capacity(num_draft);
            let mut draft_input = next;
            // Feed the draft the POST-final-norm target hidden. The gemma4-MTP draft's
            // pre_projection was trained on the normed last hidden: vLLM feeds
            // final_norm(target_hidden) (verified: cos 0.988 vs HF post-norm), while
            // aegisllm was feeding the PRE-norm residual (cos 0.44 vs vLLM, rms 1.1 vs
            // 4.0) → ~38% accept. Normalize state.hidden IN PLACE for this round's
            // FIRST draft step; subsequent steps overwrite state.hidden with the
            // draft's backbone_out feedback (already in normed/predicted space), so we
            // normalize only the round-entry target hidden. state.hidden is recomputed
            // by verify_batched after the round, so this in-place edit is safe.
            {
                let CudaLlamaState { ref mut hidden, ref mut scratch, .. } = *state;
                self.runtime.rms_norm_device(
                    hidden, &self.final_norm, self.rms_norm_eps, &mut scratch.final_hidden,
                )?;
                self.runtime.copy_prefix_f32_device(
                    &scratch.final_hidden, hidden, self.hidden_size,
                )?;
            }
            let t_draft = std::time::Instant::now();
            for _k in 0..num_draft {
                // CONSTANT draft position (vLLM gemma4 MTP `constant_draft_positions`):
                // every draft step predicts from the SAME (last target) position —
                // positions/seq_lens do NOT advance between steps. The EAGLE
                // hidden-state feedback (backbone_out → state.hidden) is what carries
                // "how far ahead", not the position. (Was base_pos + k, which made
                // the draft RoPE/attention-window wrong → ~12% accept rate.)
                let proposed = self.draft_step(state, draft_input, base_pos)?;
                // Seed next draft step's backbone hidden.
                let bb = self.draft.as_ref().unwrap().backbone_hidden;
                // SAFETY of split: backbone_out lives on state.draft, hidden is
                // state.hidden — distinct fields. Use a scoped destructure.
                {
                    let CudaLlamaState { ref mut hidden, ref draft, .. } = *state;
                    let dstate = draft.as_ref().unwrap();
                    self.runtime.copy_prefix_f32_device(
                        &dstate.scratch.backbone_out,
                        hidden,
                        bb.min(hidden.len()),
                    )?;
                }
                proposals.push(proposed);
                draft_input = proposed;
                if is_eos(proposed) {
                    break;
                }
            }

            // ── BATCHED target verification: ONE forward over [next, prop0..]. ──
            // state.position is still base_pos. verify_batched runs the K+1-token
            // batched (chunked-prefill) forward at positions [base_pos, base_pos+K],
            // writing K+1 target KV slots, and returns the K+1 per-position greedy
            // argmaxes. preds[i] is the target's prediction AFTER verify_tokens[i],
            // i.e. the token for position base_pos+i+1.
            if spec_stats { self.runtime.synchronize()?; stat_draft_us += t_draft.elapsed().as_micros() as u64; }
            let t_verify = std::time::Instant::now();
            let mut verify_tokens = Vec::with_capacity(proposals.len() + 1);
            verify_tokens.push(next);
            verify_tokens.extend_from_slice(&proposals);
            let preds = self.verify_batched(state, &verify_tokens, base_pos)?;
            if spec_stats { stat_verify_us += t_verify.elapsed().as_micros() as u64; }
            let kk = proposals.len();

            // Accept length m = longest prefix where target argmax == proposal.
            // preds[m] is the correction (m<kk) or the free bonus (m==kk).
            let mut m = 0usize;
            while m < kk && preds[m] == proposals[m] {
                m += 1;
            }
            if spec_stats {
                stat_rounds += 1;
                stat_proposed += kk;
                stat_accepted += m;
                if stat_rounds <= 3 {
                    eprintln!(
                        "[spec-dbg] round {} base_pos={} proposals={:?} target_preds={:?} accepted={}",
                        stat_rounds, base_pos, proposals, &preds[..kk.min(preds.len())], m,
                    );
                }
            }

            // KV/position rewind: only positions [base_pos, base_pos+m] are valid
            // (next + m accepted proposals); the rejected tail [base_pos+m+1 ..
            // base_pos+K] is stale and gets positionally overwritten next round.
            // No truncate op needed — KV is position-addressed. The NEXT token to
            // write is the correction at base_pos+m+1.
            state.position = base_pos + m + 1;
            // Next draft round's backbone hidden = row m (the correction's
            // conditioning hidden). Wrong row only lowers accept rate, not
            // correctness — but feed the right one.
            {
                let CudaLlamaState { ref mut hidden, ref prefill, .. } = *state;
                let prefill = prefill.as_ref().ok_or_else(|| {
                    AegisError::InvalidPlan("spec verify: prefill scratch missing post-verify".into())
                })?;
                self.runtime
                    .copy_row_f32_device(&prefill.hidden, m, self.hidden_size, hidden)?;
            }

            // Emit committed tokens: proposals[0..m] (accepted), then preds[m]
            // (correction/bonus) left UNPUSHED in `next` (the outer loop pushes it
            // next iteration — preserves the push-once contract). Same EOS/stop
            // semantics as the per-token path.
            for j in 0..=m {
                let tok = if j < m { proposals[j] } else { preds[m] };
                if is_eos(tok) {
                    // EOS: surface as pending `next`; outer loop terminates without emit.
                    next = tok;
                    break;
                }
                if request.stop_token_ids.contains(&tok) {
                    generated.push(tok);
                    break 'outer;
                }
                if j == m {
                    // Final token of the round — hand to `next`, UNPUSHED.
                    next = tok;
                    break;
                }
                // Accepted proposal strictly followed by another verified token.
                if generated.len() >= request.max_tokens {
                    break 'outer;
                }
                generated.push(tok);
            }
        }
        if spec_stats {
            let dt = stat_t0.elapsed().as_secs_f64();
            let acc_rate = if stat_proposed > 0 { stat_accepted as f64 / stat_proposed as f64 } else { 0.0 };
            let toks_per_round = if stat_rounds > 0 { (generated.len() as f64) / stat_rounds as f64 } else { 0.0 };
            eprintln!(
                "[spec-stats] generated={} rounds={} proposed={} accepted={} accept_rate={:.1}% \
                 tokens/round={:.2} decode={:.2}s {:.1} tok/s (num_draft={}) | draft={:.1}ms/rd verify={:.1}ms/rd",
                generated.len(), stat_rounds, stat_proposed, stat_accepted, acc_rate * 100.0,
                toks_per_round, dt, generated.len() as f64 / dt, num_draft,
                stat_draft_us as f64 / 1000.0 / stat_rounds.max(1) as f64,
                stat_verify_us as f64 / 1000.0 / stat_rounds.max(1) as f64,
            );
        }
        Ok(generated)
    }
}

/// Run the draft's 4 Q-only decoder layers against the TARGET's KV cache. Each
/// layer attends `target_layers[target_kv_layer[li]].kv` via the existing
/// `kv_shared_override` path (q_proj + q_norm + RoPE + attention + o_proj + MLP;
/// NO k/v projection or KV store).
///
/// TODO(gpu-verify): cross-model KV pointer sharing — confirm the draft Q (4
/// heads) GQA-attends the target's 2 KV heads, and the draft global layer's
/// p-RoPE (head_dim=512, factor 0.25) matches the target's global RoPE.
#[allow(clippy::too_many_arguments)]
fn draft_forward_layers(
    exec: &CudaLlamaExecutor,
    draft: &DraftModel,
    dstate: &mut super::state::DraftState,
    target_layers: &mut [CudaLayerState],
    decode_position: &DeviceBuffer<u32>,
    decode_seq_len: &DeviceBuffer<u32>,
    position: usize,
    seq_len: usize,
    // WORKSTREAM A (draft-trace): Some(dir) only on round-0/step-0. Each layer's
    // post-attention residual and final output are dumped, so the compare can
    // localize a divergence down to attention-vs-MLP within a single layer.
    trace: &Option<String>,
) -> Result<()> {
    let rt = &exec.runtime;
    let eps = draft.rms_norm_eps;
    let num_heads = draft.num_attention_heads;
    let draft_hidden = draft.draft_hidden();
    for (li, layer) in draft.layers.iter().enumerate() {
        let parent_idx = draft.target_kv_layer[li];
        // Immutable borrow of the target parent's KV used as override; the
        // `&mut layer_state` is a disjoint dummy on the draft state, so no
        // aliasing of the target KV.
        let parent_kv: &CudaKvCache = &target_layers[parent_idx].kv;
        super::attention::forward_attention_device(
            rt,
            layer,
            &mut dstate.dummy_layer_state,
            Some(parent_kv),
            &dstate.scratch.hidden,
            &mut dstate.decoder_scratch,
            decode_position,
            decode_seq_len,
            eps,
            num_heads,
            layer.layer_num_kv_heads,
            layer.layer_head_dim,
            exec.kv_context_size,
            layer.rope,
            None,
            position,
            seq_len,
        )?;
        // TRACE: post-attention residual = hidden + (post-normed) attn-out, i.e.
        // vLLM's `hidden_states = post_attention_layernorm(attn) + residual`
        // (gemma4_mtp.py L333-334). Diverging HERE but not at pre_proj isolates
        // the bug to this layer's ATTENTION (q_proj/q_norm/RoPE/attn/o_proj).
        draft_dump_device(
            trace,
            &format!("layer{li:02}_post_attn"),
            rt,
            &dstate.decoder_scratch.residual,
            draft_hidden,
        );
        super::mlp::forward_mlp_device(rt, layer, li, None, &mut dstate.decoder_scratch, eps)?;
        // TRACE: full layer output (post-MLP, post-scalar) = vLLM layer return
        // value (gemma4_mtp.py L341-343). Diverging HERE but matching at
        // post_attn isolates the bug to this layer's MLP.
        draft_dump_device(
            trace,
            &format!("layer{li:02}_out"),
            rt,
            &dstate.decoder_scratch.hidden_out,
            draft_hidden,
        );
        // Move the layer output (decoder_scratch.hidden_out) into the running
        // draft hidden for the next layer.
        rt.copy_prefix_f32_device(
            &dstate.decoder_scratch.hidden_out,
            &mut dstate.scratch.hidden,
            draft_hidden,
        )?;
    }
    Ok(())
}

/// Centroid-masked sparse LM head over the draft's `final_hidden`. Returns the
/// greedy argmax token id.
///
/// 1. score `final_hidden` vs `num_centroids` centroids,
/// 2. CPU top-`top_k` centroids,
/// 3. materialize candidate token ids each centroid owns (via `token_ordering`),
/// 4. dense (tied) lm_head over candidate rows only,
/// 5. argmax → global token id.
///
/// TODO(gpu-verify): the centroid→token slice mapping. Here we assume a UNIFORM
/// partition: centroid `c` owns `token_ordering[c*per_centroid ..
/// (c+1)*per_centroid)` with `per_centroid = ceil(vocab/num_centroids)`. The HF
/// `Gemma4MaskedEmbedding` may store explicit per-centroid offsets — replace
/// the slice math if so.
fn draft_sparse_head_argmax(
    rt: &crate::cuda::CudaRuntime,
    draft: &DraftModel,
    dstate: &mut super::state::DraftState,
) -> Result<usize> {
    let head = &draft.centroid_head;

    // A/B DIAGNOSTIC (AEGIS_DRAFT_DENSE_HEAD=1): bypass the centroid mask and do a
    // FULL dense tied-lm_head argmax over the whole vocab. If accept rate jumps vs
    // the sparse head, the centroid SELECTION is the bug (right token missing from
    // the top-k candidate set). If it stays ~equal, the centroid head is fine and
    // the draft-hidden quality (RoPE/attention/layers) is the bug.
    if std::env::var("AEGIS_DRAFT_DENSE_HEAD").is_ok() {
        let vocab = draft.embed_tokens.rows;
        let mut logits = rt.alloc_f32(vocab)?;
        rt.matvec_bf16_reference_device(
            &draft.embed_tokens, &dstate.scratch.final_hidden, &mut logits,
        )?;
        let h = rt.download_f32(&logits)?;
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &v) in h.iter().enumerate() {
            if v > bv { bv = v; best = i; }
        }
        return Ok(best);
    }

    // 1. Centroid scores.
    rt.matvec_bf16_reference_device(
        &head.centroids,
        &dstate.scratch.final_hidden,
        &mut dstate.scratch.centroid_scores,
    )?;
    let scores = rt.download_f32(&dstate.scratch.centroid_scores)?;

    // 2. Top-k centroids.
    let mut idx: Vec<usize> = (0..head.num_centroids.min(scores.len())).collect();
    let top_k = head.top_k.min(idx.len());
    idx.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    idx.truncate(top_k);

    // 3. Gather candidate token ids.
    let per_centroid = head.vocab_size.div_ceil(head.num_centroids);
    let mut candidates: Vec<u32> = Vec::with_capacity(top_k * per_centroid);
    for &c in &idx {
        let start = c * per_centroid;
        let end = (start + per_centroid).min(head.token_ordering.len());
        if start < end {
            candidates.extend_from_slice(&head.token_ordering[start..end]);
        }
    }
    if candidates.is_empty() {
        return Err(AegisError::InvalidPlan(
            "draft centroid head produced no candidate tokens".into(),
        ));
    }
    let cap = dstate.scratch.candidate_rows.len();
    if candidates.len() > cap {
        candidates.truncate(cap);
    }
    let num_candidates = candidates.len();

    // 4. Sparse lm_head over candidate rows.
    rt.upload_u32_slice_to_device(&candidates, &mut dstate.scratch.candidate_rows)?;
    rt.spec_sparse_lm_head_matvec_device(
        &draft.embed_tokens,
        &dstate.scratch.final_hidden,
        &dstate.scratch.candidate_rows,
        num_candidates,
        &mut dstate.scratch.candidate_logits,
    )?;
    let logits = rt.download_f32(&dstate.scratch.candidate_logits)?;

    // 5. Argmax → global token id (tie-break to lower token id).
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for i in 0..num_candidates {
        let v = logits[i];
        if v > best_val || (v == best_val && candidates[i] < candidates[best]) {
            best_val = v;
            best = i;
        }
    }
    Ok(candidates[best] as usize)
}

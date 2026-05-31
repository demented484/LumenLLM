//! Qwen3.6 (Qwen3-Next) EAGLE / MTP speculative-decoding head.
//!
//! Unlike the Gemma-4 draft (a separate small Q-only model with a centroid
//! head — see `speculative.rs`), the Qwen3.6 MTP head is a **single full
//! decoder layer that lives in the SAME checkpoint** (`model_mtp.safetensors`,
//! indexed by `model.safetensors.index.json` under the `mtp.*` prefix). It
//! shares the target's `embed_tokens`, `lm_head` and uses the target's
//! `final_norm` to normalize the conditioning hidden.
//!
//! Per the vLLM reference (`vllm/model_executor/models/qwen3_5_mtp.py`,
//! `Qwen3_5MultiTokenPredictor.forward`):
//!
//! ```text
//!   e = embed_tokens(token)                 # shared main embedding
//!   e = pre_fc_norm_embedding(e)            # RMSNorm (1+w)
//!   h = pre_fc_norm_hidden(target_hidden)   # RMSNorm (1+w) on POST-final-norm
//!   x = fc( cat([e, h]) )                   # [2H] -> [H]
//!   x = decoder_layer(x)                    # full-attn (gated) + MoE, PreOnly
//!   x = mtp.norm(x)                         # RMSNorm (1+w)
//!   logit = lm_head(x)                      # shared main lm_head -> argmax token
//! ```
//!
//! The `decoder_layer` is a standard Qwen3-Next full-attention + sparse-MoE
//! decoder, so we reuse `forward_attention_device` + `forward_mlp_device`
//! unchanged. The MTP layer's experts are stored as **stacked/fused** BF16
//! tensors (`experts.gate_up_proj [E, 2I, H]`, `experts.down_proj [E, H, I]`),
//! which we slice into per-expert `DeviceBf16Matrix` at load so the existing
//! per-expert decode MoE GEMV path runs them as-is.
//!
//! ## POST-final-norm conditioning (the validated MTP lesson)
//!
//! The MTP head consumes the **post-final-norm** target hidden (vLLM applies
//! `pre_fc_norm_hidden` to the target's pre-lm-head hidden, which is itself the
//! output of the model's `norm`). Feeding the pre-norm residual collapses the
//! accept rate — the same bug fixed for Gemma-4 (see
//! `aegisllm_draft_trace_harness`). We normalize `state.hidden` with the
//! target's `final_norm` before the draft round (mirroring `speculative.rs`),
//! so `pre_fc_norm_hidden` receives the post-final-norm hidden.

use aegisllm_base::artifact::ModelArtifact;
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::executor::tensors::require_tensor;
use aegisllm_base::generation::{GenerateRequest, SamplingConfig};
use aegisllm_base::planning::placement::StoragePlacement;
use aegisllm_base::tensor::storage::TensorStorageLoader;
use aegisllm_base::tensor::TensorDType;

use crate::cuda::{CudaWeightLoader, DeviceBuffer};

use super::loader::cuda_residency_for_store;
use super::mlp::DenseActivation;
use super::rope::RopeConfig;
use super::state::{
    CudaKvCache, CudaLayer, CudaLayerState, CudaLinear, CudaLlamaExecutor, CudaLlamaState,
};

/// One MTP routed expert (all BF16 — the MTP head ships unquantized weights).
#[derive(Debug)]
pub(super) struct MtpExpert {
    pub(super) gate_proj: CudaLinear,
    pub(super) up_proj: CudaLinear,
    pub(super) down_proj: CudaLinear,
}

/// The MTP MoE block (BF16 routed + shared experts). Kept self-contained
/// (NOT the target's NVFP4-typed `CudaMoE`) because the MTP experts are BF16;
/// the dedicated `forward_mtp_moe_decode` runs a simple per-expert BF16 GEMV
/// loop (M=1 draft step, top-8 of 256 → cheap) instead of the NVFP4 streaming
/// decode path.
#[derive(Debug)]
pub(super) struct MtpMoe {
    pub(super) router: crate::cuda::DeviceBf16Matrix,
    pub(super) experts: Vec<MtpExpert>,
    pub(super) shared_gate_proj: CudaLinear,
    pub(super) shared_up_proj: CudaLinear,
    pub(super) shared_down_proj: CudaLinear,
    /// `shared_expert_gate.weight` [1, hidden] BF16 — sigmoid gate on shared out.
    pub(super) shared_gate: Option<crate::cuda::DeviceBf16Matrix>,
    pub(super) top_k: usize,
    pub(super) num_experts: usize,
    pub(super) intermediate: usize,
    /// `[num_experts]` identity per-expert scale (the GPU router kernel always
    /// multiplies by this; Qwen3-Next has no per-expert calibration).
    pub(super) per_expert_scale: DeviceBuffer<f32>,
}

/// The loaded Qwen3.6 MTP head. Small (~1.7 GiB BF16), VRAM-resident.
#[derive(Debug)]
pub(super) struct MtpHead {
    /// `mtp.fc.weight` [hidden, 2*hidden] BF16 — projects cat[e_norm; h_norm].
    pub(super) fc: crate::cuda::DeviceBf16Matrix,
    /// `mtp.pre_fc_norm_embedding.weight` [hidden] (folded +1 for Qwen).
    pub(super) pre_fc_norm_embedding: DeviceBuffer<f32>,
    /// `mtp.pre_fc_norm_hidden.weight` [hidden] (folded +1 for Qwen).
    pub(super) pre_fc_norm_hidden: DeviceBuffer<f32>,
    /// `mtp.norm.weight` [hidden] (folded +1 for Qwen) — final norm before lm_head.
    pub(super) norm: DeviceBuffer<f32>,
    /// The single MTP decoder layer's full-attention sublayer (as a `CudaLayer`
    /// with stub MLP — the MoE is held separately in `moe`).
    pub(super) layer: CudaLayer,
    /// The MTP decoder layer's MoE sublayer (BF16).
    pub(super) moe: MtpMoe,
    pub(super) hidden_size: usize,
    pub(super) rms_norm_eps: f32,
    pub(super) num_attention_heads: usize,
    pub(super) num_kv_heads: usize,
    pub(super) head_dim: usize,
}

/// Read a checkpoint tensor's raw bytes (host) for slicing. The MTP tensors are
/// indexed by the main `model.safetensors.index.json` (pointing at
/// `model_mtp.safetensors`), so they live in the already-loaded artifact.
fn read_tensor_bytes(
    artifact: &ModelArtifact,
    name: &str,
    loader: &mut TensorStorageLoader,
) -> Result<Vec<u8>> {
    let tensor = require_tensor(artifact, name)?;
    let store = StoragePlacement::Mmap;
    let loaded = loader.load_for_store(tensor, store)?;
    Ok(loaded.as_bytes().to_vec())
}

/// Load a norm vector, folding the +1 (Qwen3-Next zero-centered RMSNorm) when
/// `qwen_unit_norm`.
fn load_mtp_norm(
    cuda: &CudaWeightLoader<'_>,
    artifact: &ModelArtifact,
    store: StoragePlacement,
    loader: &mut TensorStorageLoader,
    name: &str,
    qwen_unit_norm: bool,
) -> Result<DeviceBuffer<f32>> {
    let b = cuda.load_dense_vector_with_store(require_tensor(artifact, name)?, store, loader)?;
    if qwen_unit_norm { cuda.plus_one_norm(b) } else { Ok(b) }
}

/// Reinterpret a BF16 byte buffer as `&[u16]`.
fn bytes_to_u16(bytes: &[u8]) -> Result<Vec<u16>> {
    if bytes.len() % 2 != 0 {
        return Err(AegisError::InvalidPlan(format!(
            "MTP expert tensor has odd byte length {}",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

/// Load the MTP head from the artifact. Reuses target hyperparameters
/// (`hidden_size`, attention dims, rms_norm_eps, RoPE) — the MTP layer is a
/// Qwen3-Next full-attention layer (`layer_type="full_attention"` per vLLM).
pub(super) fn load_mtp_head(
    exec: &CudaLlamaExecutor,
    artifact: &ModelArtifact,
) -> Result<MtpHead> {
    let device = exec.runtime.device_index();
    let store = StoragePlacement::Vram { device };
    let residency = cuda_residency_for_store(store, device)?;
    let hidden = exec.hidden_size;

    // Arena is only needed for the host-resident loader path; the MTP head is
    // fully VRAM-resident. Size it to the largest single tensor we DMA through
    // the bounce (fc = hidden*2*hidden BF16).
    let host_arena = std::sync::Arc::new(crate::cuda::host_arena::PinnedArena::new(
        &exec.runtime,
        (hidden * 2 * hidden * 2).max(1),
    )?);
    let cuda = exec.runtime.weight_loader_with_arena(host_arena.clone());
    let mut loader = TensorStorageLoader::new();

    // Qwen3-Next zero-centered RMSNorm (norm·(1+weight)): fold +1 at load.
    let qwen_unit_norm = {
        let mt = artifact.config.model_type.as_str();
        mt.contains("qwen3_5") || mt.contains("qwen3_next")
    };

    let fc = cuda.load_bf16_matrix_with_store(
        require_tensor(artifact, "mtp.fc.weight")?,
        store,
        residency.clone(),
        &mut loader,
    )?;
    if fc.rows != hidden || fc.cols != 2 * hidden {
        return Err(AegisError::InvalidPlan(format!(
            "mtp.fc.weight shape [{}, {}] != expected [{hidden}, {}]",
            fc.rows, fc.cols, 2 * hidden
        )));
    }
    let pre_fc_norm_embedding =
        load_mtp_norm(&cuda, artifact, store, &mut loader, "mtp.pre_fc_norm_embedding.weight", qwen_unit_norm)?;
    let pre_fc_norm_hidden =
        load_mtp_norm(&cuda, artifact, store, &mut loader, "mtp.pre_fc_norm_hidden.weight", qwen_unit_norm)?;
    let norm = load_mtp_norm(&cuda, artifact, store, &mut loader, "mtp.norm.weight", qwen_unit_norm)?;

    let layer = load_mtp_decoder_layer(&cuda, artifact, exec, store, &mut loader)?;
    let moe = load_mtp_moe(&cuda, artifact, exec, store, &mut loader)?;

    host_arena.pin_now()?;
    drop(cuda);
    drop(host_arena);

    Ok(MtpHead {
        fc,
        pre_fc_norm_embedding,
        pre_fc_norm_hidden,
        norm,
        layer,
        moe,
        hidden_size: hidden,
        rms_norm_eps: exec.rms_norm_eps,
        num_attention_heads: exec.num_attention_heads,
        num_kv_heads: exec.num_kv_heads,
        head_dim: exec.head_dim,
    })
}

/// Build the single MTP decoder layer as a standard `CudaLayer` (Qwen3-Next
/// full-attention with output gate + sparse MoE, both BF16). Mirrors the main
/// loader's full-attn + MoE construction but with the `mtp.layers.0.` prefix
/// and stacked-expert slicing.
fn load_mtp_decoder_layer(
    cuda: &CudaWeightLoader<'_>,
    artifact: &ModelArtifact,
    exec: &CudaLlamaExecutor,
    store: StoragePlacement,
    loader: &mut TensorStorageLoader,
) -> Result<CudaLayer> {
    let device = cuda.device_index();
    let residency = cuda_residency_for_store(store, device)?;
    let prefix = "mtp.layers.0";
    let num_kv_heads = exec.num_kv_heads;
    let head_dim = exec.head_dim;

    let qwen_unit_norm = {
        let mt = artifact.config.model_type.as_str();
        mt.contains("qwen3_5") || mt.contains("qwen3_next")
    };

    // ── Full-attention (gated) projections, all BF16. ──
    // q_proj is double-width ([query|gate] per head) for attn_output_gate.
    let gated = artifact.config.attn_output_gate == Some(true);
    let q_proj = CudaLinear::Bf16(cuda.load_bf16_matrix_with_store(
        require_tensor(artifact, &format!("{prefix}.self_attn.q_proj.weight"))?,
        store,
        residency.clone(),
        loader,
    )?);
    let k_proj = CudaLinear::Bf16(cuda.load_bf16_matrix_with_store(
        require_tensor(artifact, &format!("{prefix}.self_attn.k_proj.weight"))?,
        store,
        residency.clone(),
        loader,
    )?);
    let v_proj = CudaLinear::Bf16(cuda.load_bf16_matrix_with_store(
        require_tensor(artifact, &format!("{prefix}.self_attn.v_proj.weight"))?,
        store,
        residency.clone(),
        loader,
    )?);
    let o_proj = CudaLinear::Bf16(cuda.load_bf16_matrix_with_store(
        require_tensor(artifact, &format!("{prefix}.self_attn.o_proj.weight"))?,
        store,
        residency.clone(),
        loader,
    )?);
    let q_norm_weight = Some(load_mtp_norm(
        cuda, artifact, store, loader, &format!("{prefix}.self_attn.q_norm.weight"), qwen_unit_norm,
    )?);
    let k_norm_weight = Some(load_mtp_norm(
        cuda, artifact, store, loader, &format!("{prefix}.self_attn.k_norm.weight"), qwen_unit_norm,
    )?);

    // Norms (Qwen PreOnly: input_layernorm + post_attention_layernorm = pre-MLP).
    let input_norm_weight = load_mtp_norm(
        cuda, artifact, store, loader, &format!("{prefix}.input_layernorm.weight"), qwen_unit_norm,
    )?;
    let post_attention_norm_weight = load_mtp_norm(
        cuda, artifact, store, loader, &format!("{prefix}.post_attention_layernorm.weight"), qwen_unit_norm,
    )?;

    // RoPE: full-attention layers use rope_theta_global + partial_rotary.
    let partial_dim = {
        let factor = artifact.config.partial_rotary_factor.unwrap_or(1.0);
        if factor < 1.0 {
            (factor as f64 * head_dim as f64).round() as usize
        } else {
            0
        }
    };
    let theta_override = artifact.config.rope_theta_global.map(|v| v as f32);
    let rope = RopeConfig::from_artifact(artifact)
        .to_device_with_partial_dim_and_theta(partial_dim, theta_override)?;

    Ok(CudaLayer {
        input_norm_weight,
        post_attention_norm_weight,
        post_attn_sublayer_norm: None,
        post_mlp_sublayer_norm: None,
        post_feedforward_layernorm_1: None,
        pre_feedforward_layernorm_2: None,
        post_feedforward_layernorm_2: None,
        layer_scalar: None,
        q_proj,
        k_proj,
        v_proj,
        qkv_proj: None,
        o_proj,
        q_norm_weight,
        k_norm_weight,
        // Dense MLP slots are stubs (MoE layer uses `moe`).
        gate_proj: CudaLinear::Nvfp4(cuda.alloc_dummy_nvfp4_linear("mtp.mlp.gate_proj.stub")?),
        up_proj: CudaLinear::Nvfp4(cuda.alloc_dummy_nvfp4_linear("mtp.mlp.up_proj.stub")?),
        down_proj: CudaLinear::Nvfp4(cuda.alloc_dummy_nvfp4_linear("mtp.mlp.down_proj.stub")?),
        dense_activation: DenseActivation::Swiglu,
        window_size: 0,
        rope,
        moe: None,
        layer_head_dim: head_dim,
        layer_num_kv_heads: num_kv_heads,
        ple: None,
        kv_shared_from: None,
        gdn: None,
        attn_output_gate: gated,
    })
}

/// Build the MTP MoE from stacked BF16 expert tensors. The checkpoint stores
/// `experts.gate_up_proj [E, 2I, H]` (gate stacked on up along dim 1) and
/// `experts.down_proj [E, H, I]`; we slice per-expert into `CudaMoEExpert`
/// BF16 matrices so the existing decode MoE GEMV path runs them as-is.
fn load_mtp_moe(
    cuda: &CudaWeightLoader<'_>,
    artifact: &ModelArtifact,
    exec: &CudaLlamaExecutor,
    store: StoragePlacement,
    loader: &mut TensorStorageLoader,
) -> Result<MtpMoe> {
    let device = cuda.device_index();
    let residency = cuda_residency_for_store(store, device)?;
    let prefix = "mtp.layers.0";
    let hidden = exec.hidden_size;
    let num_experts = artifact.config.num_experts.ok_or_else(|| {
        AegisError::InvalidPlan("MTP MoE: missing num_experts in config".into())
    })?;
    let top_k = artifact.config.num_experts_per_tok.ok_or_else(|| {
        AegisError::InvalidPlan("MTP MoE: missing num_experts_per_tok in config".into())
    })?;

    // Router [num_experts, hidden] BF16.
    let router = cuda.load_bf16_matrix_with_store(
        require_tensor(artifact, &format!("{prefix}.mlp.gate.weight"))?,
        store,
        residency.clone(),
        loader,
    )?;

    // Stacked experts → slice per-expert.
    let gate_up_tensor = require_tensor(artifact, &format!("{prefix}.mlp.experts.gate_up_proj"))?;
    let down_tensor = require_tensor(artifact, &format!("{prefix}.mlp.experts.down_proj"))?;
    if gate_up_tensor.dtype != TensorDType::BF16 || down_tensor.dtype != TensorDType::BF16 {
        return Err(AegisError::InvalidPlan(
            "MTP stacked experts must be BF16".into(),
        ));
    }
    // gate_up_proj [E, 2I, H]; down_proj [E, H, I].
    let two_i = gate_up_tensor.shape[1];
    let h_gu = gate_up_tensor.shape[2];
    let inter = two_i / 2;
    let h_dn = down_tensor.shape[1];
    let inter_dn = down_tensor.shape[2];
    if gate_up_tensor.shape[0] != num_experts
        || down_tensor.shape[0] != num_experts
        || h_gu != hidden
        || h_dn != hidden
        || two_i % 2 != 0
        || inter != inter_dn
    {
        return Err(AegisError::InvalidPlan(format!(
            "MTP expert shapes mismatch: gate_up={:?} down={:?} (num_experts={num_experts}, hidden={hidden})",
            gate_up_tensor.shape, down_tensor.shape
        )));
    }

    let gate_up_bytes = read_tensor_bytes(artifact, &format!("{prefix}.mlp.experts.gate_up_proj"), loader)?;
    let down_bytes = read_tensor_bytes(artifact, &format!("{prefix}.mlp.experts.down_proj"), loader)?;
    let gate_up_u16 = bytes_to_u16(&gate_up_bytes)?;
    let down_u16 = bytes_to_u16(&down_bytes)?;
    let gu_per_expert = two_i * hidden; // [2I, H]
    let dn_per_expert = hidden * inter; // [H, I]

    let mut experts = Vec::with_capacity(num_experts);
    for e in 0..num_experts {
        let gu = &gate_up_u16[e * gu_per_expert..(e + 1) * gu_per_expert];
        // gate = rows [0, I) of [2I, H]; up = rows [I, 2I).
        let gate_slice = &gu[0..inter * hidden];
        let up_slice = &gu[inter * hidden..two_i * hidden];
        let dn = &down_u16[e * dn_per_expert..(e + 1) * dn_per_expert];
        let gate_proj = CudaLinear::Bf16(cuda.bf16_matrix_from_host_u16(
            &format!("{prefix}.mlp.experts.{e}.gate_proj"),
            inter,
            hidden,
            gate_slice,
        )?);
        let up_proj = CudaLinear::Bf16(cuda.bf16_matrix_from_host_u16(
            &format!("{prefix}.mlp.experts.{e}.up_proj"),
            inter,
            hidden,
            up_slice,
        )?);
        let down_proj = CudaLinear::Bf16(cuda.bf16_matrix_from_host_u16(
            &format!("{prefix}.mlp.experts.{e}.down_proj"),
            hidden,
            inter,
            dn,
        )?);
        experts.push(MtpExpert { gate_proj, up_proj, down_proj });
    }

    // Shared expert (BF16) + sigmoid gate.
    let sp = format!("{prefix}.mlp.shared_expert");
    let shared_gate_proj = CudaLinear::Bf16(cuda.load_bf16_matrix_with_store(
        require_tensor(artifact, &format!("{sp}.gate_proj.weight"))?,
        store,
        residency.clone(),
        loader,
    )?);
    let shared_up_proj = CudaLinear::Bf16(cuda.load_bf16_matrix_with_store(
        require_tensor(artifact, &format!("{sp}.up_proj.weight"))?,
        store,
        residency.clone(),
        loader,
    )?);
    let shared_down_proj = CudaLinear::Bf16(cuda.load_bf16_matrix_with_store(
        require_tensor(artifact, &format!("{sp}.down_proj.weight"))?,
        store,
        residency.clone(),
        loader,
    )?);
    let shared_gate = artifact
        .tensors
        .get(&format!("{prefix}.mlp.shared_expert_gate.weight"))
        .map(|t| cuda.load_bf16_matrix_with_store(t, store, residency.clone(), loader))
        .transpose()?;

    // No per-expert / router-input scaling for Qwen3-Next (plain softmax→topk→renorm).
    let mut per_expert_scale = cuda.runtime().alloc_f32(num_experts)?;
    cuda.runtime()
        .upload_f32_slice_to_device(&vec![1.0_f32; num_experts], &mut per_expert_scale)?;

    Ok(MtpMoe {
        router,
        experts,
        shared_gate_proj,
        shared_up_proj,
        shared_down_proj,
        shared_gate,
        top_k,
        num_experts,
        intermediate: inter,
        per_expert_scale,
    })
}

// ───────────────────────── per-sequence MTP state ──────────────────────────

/// Per-sequence scratch + KV for the MTP head. Allocated alongside the target's
/// `CudaLlamaState` when the head is attached.
#[derive(Debug)]
pub(super) struct MtpState {
    /// Decoder scratch sized to the (target = MTP) hidden width. Reused by the
    /// MTP layer's `forward_attention_device` for its q/k/v/attn buffers and
    /// the `residual` post-attention stream.
    pub(super) scratch: super::state::CudaScratch,
    /// The MTP layer's OWN KV cache (full-attention layer writes its own K/V).
    pub(super) layer_state: CudaLayerState,
    /// `[hidden]` — normalized token embedding (first half of fc input).
    pub(super) emb_normed: DeviceBuffer<f32>,
    /// `[hidden]` — normalized conditioning hidden (second half of fc input).
    pub(super) hid_normed: DeviceBuffer<f32>,
    /// `[2*hidden]` — fc input concat `[emb_normed; hid_normed]`.
    pub(super) fc_input: DeviceBuffer<f32>,
    /// `[hidden]` — fc output / running mixer hidden (the MTP layer's residual).
    pub(super) hidden: DeviceBuffer<f32>,
    /// `[hidden]` — post-mtp.norm hidden (input to lm_head).
    pub(super) normed: DeviceBuffer<f32>,
    /// `[vocab]` — lm_head logits for argmax.
    pub(super) logits: DeviceBuffer<f32>,
    // ── MoE scratch (dedicated BF16 path) ──
    /// `[num_experts]` router logits.
    pub(super) router_logits: DeviceBuffer<f32>,
    /// `[top_k*2]` packed (idx, weight) router top-k records.
    pub(super) packed_topk: DeviceBuffer<u32>,
    /// `[hidden]` — post-attention-layernorm'd hidden (MoE input).
    pub(super) moe_input: DeviceBuffer<f32>,
    /// `[hidden]` — running MoE accumulator (shared + routed).
    pub(super) moe_acc: DeviceBuffer<f32>,
    /// `[intermediate]` per-expert gate/up/swiglu scratch.
    pub(super) expert_gate: DeviceBuffer<f32>,
    pub(super) expert_up: DeviceBuffer<f32>,
    pub(super) expert_swiglu: DeviceBuffer<f32>,
    /// `[hidden]` per-expert down-projection output.
    pub(super) expert_out: DeviceBuffer<f32>,
    /// `[1]` shared-expert sigmoid-gate logit.
    pub(super) shared_gate_logit: DeviceBuffer<f32>,
    /// GDN recurrent + conv-state SNAPSHOT, one entry per target layer that has
    /// a GDN mixer. Used to roll back the recurrent state on partial accept:
    /// `verify_batched` advances every GDN layer by K+1; if only j<K proposals
    /// accept we restore the snapshot (state at `base_pos`) and re-run the
    /// committed (j+1)-token prefix to re-advance to `base_pos+j+1` EXACTLY.
    /// `gdn_layer_indices[s]` is the target layer index for snapshot slot `s`.
    pub(super) gdn_snapshot: Vec<(DeviceBuffer<f32>, DeviceBuffer<f32>)>,
    pub(super) gdn_layer_indices: Vec<usize>,
}

/// Build the per-sequence MTP state. `kv_context_size` matches the target so
/// the MTP layer's KV cache addresses the same positions.
pub(super) fn build_mtp_state(exec: &CudaLlamaExecutor) -> Result<MtpState> {
    let head = exec
        .mtp
        .as_ref()
        .ok_or_else(|| AegisError::InvalidPlan("build_mtp_state without an MTP head".into()))?;
    let rt = &exec.runtime;
    let hidden = head.hidden_size;
    let inter = head.moe.intermediate;
    let top_k = head.moe.top_k;
    let num_experts = head.moe.num_experts;
    let vocab = exec.lm_head.rows;

    // GDN snapshot buffers: one (recurrent, conv_state) pair per GDN layer,
    // sized to that layer's state. Used for partial-accept rollback.
    let mut gdn_snapshot = Vec::new();
    let mut gdn_layer_indices = Vec::new();
    for (li, layer) in exec.layers.iter().enumerate() {
        if let Some(gdn) = layer.gdn.as_ref() {
            let d = gdn.dims;
            gdn_snapshot.push((
                rt.alloc_f32(d.state_elems())?,
                rt.alloc_f32(d.conv_state_elems())?,
            ));
            gdn_layer_indices.push(li);
        }
    }

    let scratch = build_mtp_decoder_scratch(exec, head)?;
    let kv = CudaKvCache::dense(
        rt,
        exec.kv_context_size,
        head.num_kv_heads * head.head_dim,
        exec.kv_quantization,
        exec.kv_context_size,
        false,
    )?;
    Ok(MtpState {
        scratch,
        layer_state: CudaLayerState { kv, recurrent: None, conv_state: None },
        emb_normed: rt.alloc_f32(hidden)?,
        hid_normed: rt.alloc_f32(hidden)?,
        fc_input: rt.alloc_f32(2 * hidden)?,
        hidden: rt.alloc_f32(hidden)?,
        normed: rt.alloc_f32(hidden)?,
        logits: rt.alloc_f32(vocab)?,
        router_logits: rt.alloc_f32(num_experts)?,
        packed_topk: rt.alloc_u32(top_k * 2)?,
        moe_input: rt.alloc_f32(hidden)?,
        moe_acc: rt.alloc_f32(hidden)?,
        expert_gate: rt.alloc_f32(inter)?,
        expert_up: rt.alloc_f32(inter)?,
        expert_swiglu: rt.alloc_f32(inter)?,
        expert_out: rt.alloc_f32(hidden)?,
        shared_gate_logit: rt.alloc_f32(1)?,
        gdn_snapshot,
        gdn_layer_indices,
    })
}

/// Build a minimal target-width `CudaScratch` for the MTP attention sublayer.
/// Mirrors `speculative::build_draft_scratch` but sized to the target hidden
/// width (the MTP layer is full-width). Only the attention-path fields are
/// used (q/k/v/attn_*/residual/post_normed/input_normed/quant_*); MoE is
/// handled by the dedicated `forward_mtp_moe_decode`, so `moe`/`gdn`/`ple`
/// scratch are stubs.
fn build_mtp_decoder_scratch(
    exec: &CudaLlamaExecutor,
    head: &MtpHead,
) -> Result<super::state::CudaScratch> {
    let rt = &exec.runtime;
    let hidden = head.hidden_size;
    let num_heads = head.num_attention_heads;
    let head_dim = head.head_dim;
    // Gated attention: q_proj emits 2×q_width; the de-interleave temp lands in
    // `q` then the gate stash. Size q to the gated width to be safe.
    let q_width = num_heads * head_dim;
    let kv_width = head.num_kv_heads * head_dim;
    use crate::cuda::{CudaRuntime, DECODE_SPLIT_K_MAX};
    Ok(super::state::CudaScratch {
        input_normed: rt.alloc_f32(hidden)?,
        quant_hidden: rt.alloc_f32(hidden.max(2 * q_width))?,
        quant_intermediate: rt.alloc_f32(1)?,
        mxfp4_hidden: rt.alloc_u8(CudaRuntime::mxfp4_vector_bytes(hidden.max(2 * q_width))?)?,
        mxfp4_intermediate: rt.alloc_u8(1)?,
        cutlass_payload: rt.alloc_u8(1)?,
        cutlass_scales: rt.alloc_u8(1)?,
        cutlass_workspace: rt.alloc_u8(1)?,
        q: rt.alloc_f32(2 * q_width)?,
        k: rt.alloc_f32(kv_width)?,
        v: rt.alloc_f32(kv_width)?,
        qk_norm_scratch: rt.alloc_f32((2 * q_width).max(kv_width))?,
        attn_split_acc: rt.alloc_f32(num_heads * DECODE_SPLIT_K_MAX * head_dim)?,
        attn_split_m: rt.alloc_f32(num_heads * DECODE_SPLIT_K_MAX)?,
        attn_split_l: rt.alloc_f32(num_heads * DECODE_SPLIT_K_MAX)?,
        attn_context: rt.alloc_f32(q_width)?,
        attn_out: rt.alloc_f32(hidden)?,
        residual: rt.alloc_f32(hidden)?,
        post_normed: rt.alloc_f32(hidden)?,
        gate: rt.alloc_f32(1)?,
        up: rt.alloc_f32(1)?,
        swiglu: rt.alloc_f32(1)?,
        mlp_out: rt.alloc_f32(hidden)?,
        hidden_out: rt.alloc_f32(hidden)?,
        final_hidden: rt.alloc_f32(hidden)?,
        argmax_block_values: rt.alloc_f32(1)?,
        argmax_block_indices: rt.alloc_u32(1)?,
        moe: None,
        gdn_decode: None,
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

// ───────────────────────── MTP forward + draft loop ─────────────────────────

impl CudaLlamaExecutor {
    /// One MTP draft step. Given the current draft `token` and the conditioning
    /// `cond_hidden` (POST-final-norm target hidden), runs the MTP head forward
    /// at `position` (writing the MTP layer's KV) and returns the greedy argmax
    /// proposed token. Also leaves `mtp_state.hidden` holding the post-mtp.norm
    /// hidden so the NEXT draft step can use it as conditioning.
    ///
    /// The MTP head shares the target's `embed_tokens` and `lm_head`.
    fn mtp_step(
        &self,
        target: &mut CudaLlamaState,
        token: usize,
        cond_hidden: &DeviceBuffer<f32>,
        position: usize,
    ) -> Result<usize> {
        let head = self
            .mtp
            .as_ref()
            .ok_or_else(|| AegisError::InvalidPlan("mtp_step without an MTP head".into()))?;
        let rt = &self.runtime;
        let eps = head.rms_norm_eps;
        let hidden = head.hidden_size;

        let mtp = target
            .mtp
            .as_mut()
            .ok_or_else(|| AegisError::InvalidPlan("mtp_step without MTP state".into()))?;

        // 1. embed token (shared target embed_tokens) → emb; pre_fc_norm_embedding.
        rt.bf16_row_to_f32_device(&self.embed_tokens, token, &mut mtp.hidden)?;
        if let Some(scale) = self.embed_scale {
            rt.scale_f32_device(scale, &mut mtp.hidden)?;
        }
        rt.rms_norm_device(&mtp.hidden, &head.pre_fc_norm_embedding, eps, &mut mtp.emb_normed)?;

        // 2. pre_fc_norm_hidden(cond_hidden) → hid_normed.
        rt.rms_norm_device(cond_hidden, &head.pre_fc_norm_hidden, eps, &mut mtp.hid_normed)?;

        // 3. fc(cat[emb_normed, hid_normed]) → hidden (the layer residual input).
        rt.copy_f32_d2d_range(&mtp.emb_normed, 0, &mut mtp.fc_input, 0, hidden)?;
        rt.copy_f32_d2d_range(&mtp.hid_normed, 0, &mut mtp.fc_input, hidden, hidden)?;
        rt.matvec_bf16_reference_device(&head.fc, &mtp.fc_input, &mut mtp.hidden)?;

        // 4. MTP decoder layer (full-attn gated + MoE), PreOnly residual stream.
        let seq_len = position + 1;
        rt.copy_u32_to_device(&[position as u32], &mut target.decode_position)?;
        rt.copy_u32_to_device(&[seq_len as u32], &mut target.decode_seq_len)?;
        // 4a. attention sublayer: writes scratch.residual = hidden + attn_out.
        {
            let hidden_in = &mtp.hidden as *const DeviceBuffer<f32>;
            // SAFETY: `hidden` (read) is distinct from `scratch`/`layer_state`.
            super::attention::forward_attention_device(
                rt,
                &head.layer,
                &mut mtp.layer_state,
                None,
                unsafe { &*hidden_in },
                &mut mtp.scratch,
                &target.decode_position,
                &target.decode_seq_len,
                eps,
                head.num_attention_heads,
                head.num_kv_heads,
                head.head_dim,
                self.kv_context_size,
                head.layer.rope,
                None,
                position,
                seq_len,
            )?;
        }
        // 4b. MoE sublayer on scratch.residual → mtp.hidden (= residual + moe_out).
        forward_mtp_moe_decode(rt, head, mtp, eps)?;

        // 5. mtp.norm(hidden) → normed; lm_head(normed) → argmax.
        rt.rms_norm_device(&mtp.hidden, &head.norm, eps, &mut mtp.normed)?;
        rt.matvec_bf16_reference_device(&self.lm_head, &mtp.normed, &mut mtp.logits)?;
        // Greedy argmax (no softcap on Qwen3-Next).
        let h = rt.download_f32(&mtp.logits)?;
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in h.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best = i;
            }
        }
        Ok(best)
    }

    /// Propose `num_draft` tokens autoregressively from the MTP head. The first
    /// draft input is `first_token` (the target's just-committed token); its
    /// conditioning hidden is the POST-final-norm target hidden in `target.hidden`.
    ///
    /// Returns the proposed token ids and leaves the MTP layer KV advanced over
    /// the draft positions `[base_pos, base_pos + num_draft)`. Those KV slots are
    /// transient (the MTP head is re-seeded each round from the target's verified
    /// hidden), so no rollback of the MTP KV is needed — the next round overwrites
    /// them positionally.
    pub(super) fn mtp_propose(
        &self,
        target: &mut CudaLlamaState,
        first_token: usize,
        base_pos: usize,
        num_draft: usize,
        is_eos: &dyn Fn(usize) -> bool,
    ) -> Result<Vec<usize>> {
        let hidden = self.hidden_size;
        // Conditioning for step 0 = POST-final-norm of the target's last hidden.
        // (The validated MTP lesson: feed final_norm(target_hidden), NOT the raw
        // pre-norm residual — see aegisllm_draft_trace_harness.) We compute it
        // into the MTP state's hid scratch via the target final_norm, leaving
        // `target.hidden` untouched (the verify path recomputes it anyway, but we
        // avoid mutating it so the post-round bookkeeping stays simple).
        let mut proposals = Vec::with_capacity(num_draft);
        let mut draft_token = first_token;
        // cond_hidden buffer: reuse the MTP state's `normed` field is risky
        // (overwritten in step); use a dedicated round-local normed copy.
        // We normalize target.hidden into mtp.hid_normed-sized scratch held in
        // `cond` (a fresh per-round buffer kept small — hidden-wide).
        let mut cond = self.runtime.alloc_f32(hidden)?;
        {
            let CudaLlamaState { hidden: ref th, .. } = *target;
            self.runtime
                .rms_norm_device(th, &self.final_norm, self.rms_norm_eps, &mut cond)?;
        }
        for k in 0..num_draft {
            let position = base_pos + k;
            let proposed = self.mtp_step(target, draft_token, &cond, position)?;
            // The next step's conditioning hidden = this step's post-mtp.norm
            // hidden? No — EAGLE feeds the predicted-hidden forward. vLLM feeds
            // the MTP layer's OUTPUT hidden (pre-final-norm) as the next step's
            // conditioning, normalized by pre_fc_norm_hidden inside mtp_step. We
            // copy the MTP layer residual output (mtp.hidden) into `cond` for the
            // next step. (pre_fc_norm_hidden re-normalizes it.)
            {
                let mtp = target.mtp.as_ref().unwrap();
                self.runtime.copy_prefix_f32_device(&mtp.hidden, &mut cond, hidden)?;
            }
            proposals.push(proposed);
            draft_token = proposed;
            if is_eos(proposed) {
                break;
            }
        }
        Ok(proposals)
    }
}

/// Dedicated MTP MoE decode forward (BF16). Reads `mtp.scratch.residual` (the
/// post-attention residual stream), writes `residual + moe_out` into
/// `mtp.hidden`. Qwen3-Next MoE: router on post-attn-layernorm'd hidden →
/// softmax→top_k→renorm; shared expert (SwiGLU) scaled by sigmoid(shared_gate);
/// routed experts (SwiGLU) weighted-summed.
fn forward_mtp_moe_decode(
    rt: &crate::cuda::CudaRuntime,
    head: &MtpHead,
    mtp: &mut MtpState,
    eps: f32,
) -> Result<()> {
    let moe = &head.moe;
    let hidden = head.hidden_size;

    // post_attention_layernorm(residual) → moe_input (router + experts input).
    rt.rms_norm_device(
        &mtp.scratch.residual,
        &head.layer.post_attention_norm_weight,
        eps,
        &mut mtp.moe_input,
    )?;

    // ── shared expert (always-active), SwiGLU, sigmoid-gated. ──
    matvec_cuda_linear_with_scratch_noscratch(rt, &moe.shared_gate_proj, &mtp.moe_input, &mut mtp.expert_gate)?;
    matvec_cuda_linear_with_scratch_noscratch(rt, &moe.shared_up_proj, &mtp.moe_input, &mut mtp.expert_up)?;
    rt.swiglu_device(&mtp.expert_gate, &mtp.expert_up, &mut mtp.expert_swiglu)?;
    matvec_cuda_linear_with_scratch_noscratch(rt, &moe.shared_down_proj, &mtp.expert_swiglu, &mut mtp.moe_acc)?;
    if let Some(ref sgate) = moe.shared_gate {
        rt.matvec_bf16_reference_device(sgate, &mtp.moe_input, &mut mtp.shared_gate_logit)?;
        let n = mtp.moe_acc.len();
        let logit = &mtp.shared_gate_logit as *const DeviceBuffer<f32>;
        // SAFETY: shared_gate_logit (read) and moe_acc (write) are distinct fields.
        rt.scale_by_sigmoid_scalar(&mut mtp.moe_acc, unsafe { &*logit }, n)?;
    }

    // ── router → top_k → renorm. ──
    rt.matvec_bf16_reference_device(&moe.router, &mtp.moe_input, &mut mtp.router_logits)?;
    rt.router_softmax_topk_packed_device(
        &mtp.router_logits,
        &moe.per_expert_scale,
        1,
        moe.num_experts,
        moe.top_k,
        &mut mtp.packed_topk,
    )?;
    let packed = rt.download_u32(&mtp.packed_topk)?;

    // ── routed experts: Σ_k w[k] · expert_k(moe_input). ──
    for k in 0..moe.top_k {
        let idx = packed[k * 2] as usize;
        let weight = f32::from_bits(packed[k * 2 + 1]);
        if idx >= moe.experts.len() {
            return Err(AegisError::InvalidPlan(format!(
                "MTP router selected expert {idx} >= {}",
                moe.experts.len()
            )));
        }
        let expert = &moe.experts[idx];
        matvec_cuda_linear_with_scratch_noscratch(rt, &expert.gate_proj, &mtp.moe_input, &mut mtp.expert_gate)?;
        matvec_cuda_linear_with_scratch_noscratch(rt, &expert.up_proj, &mtp.moe_input, &mut mtp.expert_up)?;
        rt.swiglu_device(&mtp.expert_gate, &mtp.expert_up, &mut mtp.expert_swiglu)?;
        matvec_cuda_linear_with_scratch_noscratch(rt, &expert.down_proj, &mtp.expert_swiglu, &mut mtp.expert_out)?;
        // moe_acc += weight * expert_out.
        rt.axpy_f32_device(weight, &mtp.expert_out, &mut mtp.moe_acc)?;
    }

    // residual add: mtp.hidden = residual + moe_acc.
    rt.add_device(&mtp.scratch.residual, &mtp.moe_acc, &mut mtp.hidden)?;
    let _ = hidden;
    Ok(())
}

/// BF16/FP8 matvec for an MTP linear (experts are always BF16 → reference
/// matvec; no quant scratch). Kept separate from
/// `linear_ops::matvec_cuda_linear_with_scratch` to avoid threading the NVFP4
/// quant scratch (never used here).
fn matvec_cuda_linear_with_scratch_noscratch(
    rt: &crate::cuda::CudaRuntime,
    linear: &CudaLinear,
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
) -> Result<()> {
    match linear {
        CudaLinear::Bf16(m) => rt.matvec_bf16_reference_device(m, input, output),
        CudaLinear::Fp8(m) => rt.matvec_fp8_standalone_device(m, input, output),
        CudaLinear::Nvfp4(_) => Err(AegisError::Unsupported(
            "MTP MoE expert must be BF16/FP8".into(),
        )),
    }
}

// ───────────────────── MTP speculative generate (greedy) ────────────────────

impl CudaLlamaExecutor {
    /// Snapshot every GDN layer's recurrent + conv state into the MTP state's
    /// backup buffers (called BEFORE a batched verify so a partial accept can
    /// roll the GDN recurrent state back to the pre-verify position).
    fn gdn_snapshot_save(&self, state: &mut CudaLlamaState) -> Result<()> {
        let rt = &self.runtime;
        // Disjoint borrow: snapshot buffers live on state.mtp; GDN states on
        // state.layers. Pull the layer index list out first.
        let indices: Vec<usize> = {
            let mtp = state.mtp.as_ref().ok_or_else(|| {
                AegisError::InvalidPlan("gdn_snapshot_save without MTP state".into())
            })?;
            mtp.gdn_layer_indices.clone()
        };
        for (slot, &li) in indices.iter().enumerate() {
            // SAFETY-free split: read from state.layers[li], write to
            // state.mtp.gdn_snapshot[slot] — different fields of `state`.
            let (rec_src, conv_src) = {
                let ls = &state.layers[li];
                (
                    ls.recurrent.as_ref().map(|b| b as *const DeviceBuffer<f32>),
                    ls.conv_state.as_ref().map(|b| b as *const DeviceBuffer<f32>),
                )
            };
            let mtp = state.mtp.as_mut().unwrap();
            let (rec_dst, conv_dst) = &mut mtp.gdn_snapshot[slot];
            if let Some(src) = rec_src {
                rt.copy_f32_device(unsafe { &*src }, rec_dst)?;
            }
            if let Some(src) = conv_src {
                rt.copy_f32_device(unsafe { &*src }, conv_dst)?;
            }
        }
        Ok(())
    }

    /// Restore every GDN layer's recurrent + conv state from the snapshot
    /// (called on a PARTIAL accept before re-running the committed prefix).
    fn gdn_snapshot_restore(&self, state: &mut CudaLlamaState) -> Result<()> {
        let rt = &self.runtime;
        let indices: Vec<usize> = {
            let mtp = state.mtp.as_ref().ok_or_else(|| {
                AegisError::InvalidPlan("gdn_snapshot_restore without MTP state".into())
            })?;
            mtp.gdn_layer_indices.clone()
        };
        for (slot, &li) in indices.iter().enumerate() {
            let (rec_src, conv_src) = {
                let mtp = state.mtp.as_ref().unwrap();
                let (r, c) = &mtp.gdn_snapshot[slot];
                (r as *const DeviceBuffer<f32>, c as *const DeviceBuffer<f32>)
            };
            let ls = &mut state.layers[li];
            if let Some(dst) = ls.recurrent.as_mut() {
                rt.copy_f32_device(unsafe { &*rec_src }, dst)?;
            }
            if let Some(dst) = ls.conv_state.as_mut() {
                rt.copy_f32_device(unsafe { &*conv_src }, dst)?;
            }
        }
        Ok(())
    }

    /// Greedy MTP speculative decoding. Lossless by construction: every emitted
    /// token equals the TARGET model's greedy argmax at that position (the
    /// batched verify produces the same per-position argmaxes the target would
    /// emit), and rejected draft tokens are discarded.
    ///
    /// GDN RECURRENT-STATE ROLLBACK (the correctness crux): a batched verify of
    /// K+1 tokens advances every GDN layer's recurrent + conv state by K+1. On a
    /// partial accept (m<K) we restore the pre-verify snapshot and re-run the
    /// committed (m+1)-token prefix, re-advancing the GDN state to EXACTLY
    /// position `base_pos+m+1`. Full-attention KV is position-addressed, so the
    /// rejected tail is overwritten next round (the prefix re-run also rewrites
    /// the committed KV). When m==K (full accept) the state is already correct.
    pub(super) fn generate_speculative_mtp_greedy(
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

            let base_pos = state.position;
            // ── Draft round: MTP proposes `num_draft` tokens autoregressively. ──
            // Conditioning hidden = POST-final-norm of the target's last hidden
            // (computed inside mtp_propose). The MTP layer's own KV is advanced
            // over the draft positions; it is re-seeded each round so no rollback
            // of the MTP KV is needed.
            let proposals = self.mtp_propose(state, next, base_pos, num_draft, is_eos)?;

            // ── Snapshot GDN state, then batched verify over [next, prop0..]. ──
            self.gdn_snapshot_save(state)?;
            let mut verify_tokens = Vec::with_capacity(proposals.len() + 1);
            verify_tokens.push(next);
            verify_tokens.extend_from_slice(&proposals);
            let preds = self.verify_batched(state, &verify_tokens, base_pos)?;
            let kk = proposals.len();

            // Accept length m = longest prefix where target argmax == proposal.
            let mut m = 0usize;
            while m < kk && preds[m] == proposals[m] {
                m += 1;
            }

            // ── GDN rollback on partial accept. ──
            // Committed = m+1 tokens (positions base_pos..=base_pos+m). On a full
            // accept (m==kk) the verify already advanced GDN state to the correct
            // base_pos+kk+1 = base_pos+m+1. On a partial accept, restore the
            // snapshot and re-run the committed prefix to re-advance GDN state
            // (and rewrite committed KV + prefill.hidden) to base_pos+m+1 exactly.
            let preds_final: Vec<usize>;
            if m < kk {
                self.gdn_snapshot_restore(state)?;
                preds_final = self.verify_batched(state, &verify_tokens[..m + 1], base_pos)?;
            } else {
                preds_final = preds;
            }

            if spec_stats {
                stat_rounds += 1;
                stat_proposed += kk;
                stat_accepted += m;
                if stat_rounds <= 5 {
                    eprintln!(
                        "[mtp-spec] round {} base_pos={} proposals={:?} target_preds={:?} accepted={}/{}",
                        stat_rounds, base_pos, proposals,
                        &preds_final[..(m + 1).min(preds_final.len())], m, kk,
                    );
                }
            }

            // state.position advances to base_pos + m + 1 (next + m accepted).
            state.position = base_pos + m + 1;
            // Next round's conditioning hidden = the verified hidden at row m (the
            // correction/bonus token's conditioning). prefill.hidden holds the
            // committed rows from the verify (full-accept) or the prefix re-run.
            {
                let CudaLlamaState { ref mut hidden, ref prefill, .. } = *state;
                let prefill = prefill.as_ref().ok_or_else(|| {
                    AegisError::InvalidPlan("mtp verify: prefill scratch missing post-verify".into())
                })?;
                self.runtime.copy_row_f32_device(&prefill.hidden, m, self.hidden_size, hidden)?;
            }

            // Emit committed tokens: proposals[0..m] (accepted), then preds[m]
            // (correction/bonus) handed to `next` UNPUSHED (outer loop pushes it).
            for j in 0..=m {
                let tok = if j < m { proposals[j] } else { preds_final[m] };
                if is_eos(tok) {
                    next = tok;
                    break;
                }
                if request.stop_token_ids.contains(&tok) {
                    generated.push(tok);
                    break 'outer;
                }
                if j == m {
                    next = tok;
                    break;
                }
                if generated.len() >= request.max_tokens {
                    break 'outer;
                }
                generated.push(tok);
            }
        }
        if spec_stats {
            let dt = stat_t0.elapsed().as_secs_f64();
            let acc_rate = if stat_proposed > 0 {
                stat_accepted as f64 / stat_proposed as f64
            } else {
                0.0
            };
            let toks_per_round = if stat_rounds > 0 {
                generated.len() as f64 / stat_rounds as f64
            } else {
                0.0
            };
            eprintln!(
                "[mtp-spec-stats] generated={} rounds={} proposed={} accepted={} accept_rate={:.1}% \
                 tokens/round={:.2} decode={:.2}s {:.1} tok/s (num_draft={})",
                generated.len(), stat_rounds, stat_proposed, stat_accepted,
                acc_rate * 100.0, toks_per_round, dt, generated.len() as f64 / dt, num_draft,
            );
        }
        Ok(generated)
    }
}

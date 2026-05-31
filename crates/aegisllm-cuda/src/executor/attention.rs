use super::linear_ops::{
    matvec_cuda_linear_with_scratch, matvec_nvfp4_device_with_scratch,
    matvec_nvfp4_prepared_device_reuse, native_mxfp4_enabled, prepare_nvfp4_input,
};
use super::state::{CudaLayer, CudaLayerState, CudaLinear, CudaScratch};
use crate::cuda::{CudaRuntime, DeviceBuffer, DeviceRopeConfig};
use aegisllm_base::error::Result;

/// Forward attention for a single layer.
///
/// `staging_slot_idx`: when `Some(idx)`, the layer's KV is host-resident and the
/// caller has pre-uploaded the prior KV onto `scratch.kv_staging.slots[idx]` via
/// the transfer stream and event-synchronized the compute stream against it.
/// This function only runs store_kv + attention against the slot; the caller is
/// responsible for scheduling the post-compute D2H writeback on the transfer stream.
/// When `None`, the layer's KV is fully VRAM-resident.
#[allow(clippy::too_many_arguments)]
pub(super) fn forward_attention_device(
    runtime: &CudaRuntime,
    layer: &CudaLayer,
    layer_state: &mut CudaLayerState,
    // KV-cache override (Gemma-4 E4B / E2B shared layers). When Some,
    // this layer's K/V projections, K-RoPE, and cache writes are skipped;
    // attention reads K/V from the override (parent layer's cache slot).
    // None for every existing target model — full attention pipeline runs.
    kv_shared_override: Option<&crate::executor::state::CudaKvCache>,
    hidden: &DeviceBuffer<f32>,
    scratch: &mut CudaScratch,
    p_position: &DeviceBuffer<u32>,
    p_seq_len: &DeviceBuffer<u32>,
    rms_norm_eps: f32,
    num_attention_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_context_size: usize,
    rope: DeviceRopeConfig,
    staging_slot_idx: Option<usize>,
    _position: usize,
    _seq_len: usize,
) -> Result<()> {
    let kv_width = num_kv_heads * head_dim;
    // Sliding-window layers allocate only `window_size` KV slots and the
    // cache wraps via `slot = position % cache_capacity`. The kernel uses
    // the `cache_capacity` arg as the wrap modulus AND as a bounds check
    // (`if (slot < context_size)` inside `aegis_*_kv_store_*_batched`),
    // so we MUST pass the per-layer effective capacity here, not the
    // engine's full `kv_context_size`. Passing 262144 for a 1024-slot
    // sliding layer makes the modulo a no-op and decode writes go
    // out-of-bounds at position >= 1024 — visible as a deterministic
    // collapse to gibberish ~920 tokens into generation.
    let effective_kv_capacity = if layer.window_size > 0 {
        layer.window_size.min(kv_context_size)
    } else {
        kv_context_size
    };

    // Qwen3-Next attention output gate: q_proj is double-width; we de-interleave
    // the query into `scratch.q` and stash the gate here, applying
    // `attn_context *= sigmoid(gate)` after attention, before o_proj.
    let q_width = num_attention_heads * head_dim;
    let mut gate_buf: Option<DeviceBuffer<f32>> = None;

    if let (Some(q), Some(k), Some(v)) = (layer.q_proj.as_nvfp4(), layer.k_proj.as_nvfp4(), layer.v_proj.as_nvfp4()) {
        // NVFP4 path (Llama, Qwen, etc.)
        runtime.rms_norm_quant_nvfp4_device(
            hidden,
            &layer.input_norm_weight,
            rms_norm_eps,
            q.input_scale,
            &mut scratch.input_normed,
            &mut scratch.quant_hidden,
        )?;
        let mut quant_scale = Some(q.input_scale);
        // Q projection: quantize input_normed to mxfp4_hidden (native path) or quant_hidden (legacy).
        // Gated layers project to 2×q_width then de-interleave query→scratch.q, gate→gate_buf.
        let mxfp4_valid = if layer.attn_output_gate {
            let mut q_full = runtime.alloc_f32(2 * q_width)?;
            let valid = matvec_nvfp4_prepared_device_reuse(
                runtime,
                q,
                &scratch.input_normed,
                &scratch.quant_hidden,
                &mut scratch.mxfp4_hidden,
                false,
                &mut q_full,
                scratch.staging_pool.as_deref_mut(),
            )?;
            let mut gate = runtime.alloc_f32(q_width)?;
            runtime.deinterleave_gated_q(&q_full, &mut scratch.q, &mut gate, num_attention_heads, head_dim)?;
            gate_buf = Some(gate);
            valid
        } else {
            matvec_nvfp4_prepared_device_reuse(
                runtime,
                q,
                &scratch.input_normed,
                &scratch.quant_hidden,
                &mut scratch.mxfp4_hidden,
                false,
                &mut scratch.q,
                scratch.staging_pool.as_deref_mut(),
            )?
        };
        // K/V projections share the same input_normed — skip MXFP4 re-quantize in native path.
        prepare_nvfp4_input(
            runtime,
            k,
            &scratch.input_normed,
            &mut quant_scale,
            &mut scratch.quant_hidden,
        )?;
        matvec_nvfp4_prepared_device_reuse(
            runtime,
            k,
            &scratch.input_normed,
            &scratch.quant_hidden,
            &mut scratch.mxfp4_hidden,
            mxfp4_valid,
            &mut scratch.k,
            scratch.staging_pool.as_deref_mut(),
        )?;
        prepare_nvfp4_input(
            runtime,
            v,
            &scratch.input_normed,
            &mut quant_scale,
            &mut scratch.quant_hidden,
        )?;
        matvec_nvfp4_prepared_device_reuse(
            runtime,
            v,
            &scratch.input_normed,
            &scratch.quant_hidden,
            &mut scratch.mxfp4_hidden,
            mxfp4_valid,
            &mut scratch.v,
            scratch.staging_pool.as_deref_mut(),
        )?;
    } else {
        // BF16 path (Gemma4 attention)
        runtime.rms_norm_device(hidden, &layer.input_norm_weight, rms_norm_eps, &mut scratch.input_normed)?;
        if layer.attn_output_gate {
            let mut q_full = runtime.alloc_f32(2 * q_width)?;
            matvec_cuda_linear_with_scratch(runtime, &layer.q_proj, &scratch.input_normed,
                &mut scratch.quant_hidden, &mut scratch.mxfp4_hidden, &mut q_full,
                scratch.staging_pool.as_deref_mut())?;
            let mut gate = runtime.alloc_f32(q_width)?;
            runtime.deinterleave_gated_q(&q_full, &mut scratch.q, &mut gate, num_attention_heads, head_dim)?;
            gate_buf = Some(gate);
        } else {
            matvec_cuda_linear_with_scratch(runtime, &layer.q_proj, &scratch.input_normed,
                &mut scratch.quant_hidden, &mut scratch.mxfp4_hidden, &mut scratch.q,
                scratch.staging_pool.as_deref_mut())?;
        }
        // Shared layers (Gemma-4 E4B last 18 of 42) skip K/V projection +
        // norm + RoPE + cache write — their attention reads K/V from the
        // parent layer's cache slot via `kv_shared_override`. Q is still
        // computed and normed/RoPE'd normally so the shared layer can
        // attend with its own query against the parent's keys/values.
        if kv_shared_override.is_none() {
            matvec_cuda_linear_with_scratch(runtime, &layer.k_proj, &scratch.input_normed,
                &mut scratch.quant_hidden, &mut scratch.mxfp4_hidden, &mut scratch.k,
                scratch.staging_pool.as_deref_mut())?;
            matvec_cuda_linear_with_scratch(runtime, &layer.v_proj, &scratch.input_normed,
                &mut scratch.quant_hidden, &mut scratch.mxfp4_hidden, &mut scratch.v,
                scratch.staging_pool.as_deref_mut())?;
        }
    }
    // Gemma 4: per-head RMS norm on Q and K, applied between projection and RoPE.
    // The norm weight has length `head_dim`; the kernel processes `num_heads` rows in parallel.
    // RMS norm cannot run in-place (the kernel re-reads its input in the second pass), so
    // write to the scratch buffer then copy back. The scratch buffer holds enough elements
    // for the largest (q_width / kv_width) across all layers.
    if let Some(ref qnw) = layer.q_norm_weight {
        runtime.rms_norm_batched_device(
            &scratch.q,
            qnw,
            num_attention_heads,
            rms_norm_eps,
            &mut scratch.qk_norm_scratch,
        )?;
        runtime.copy_prefix_f32_device(
            &scratch.qk_norm_scratch,
            &mut scratch.q,
            num_attention_heads * head_dim,
        )?;
    }
    if kv_shared_override.is_none() && layer.k_norm_weight.is_some() {
        let knw = layer.k_norm_weight.as_ref().unwrap();
        runtime.rms_norm_batched_device(
            &scratch.k,
            knw,
            num_kv_heads,
            rms_norm_eps,
            &mut scratch.qk_norm_scratch,
        )?;
        runtime.copy_prefix_f32_device(
            &scratch.qk_norm_scratch,
            &mut scratch.k,
            num_kv_heads * head_dim,
        )?;
    }
    // Gemma 4: V is RMS-normed per-head with NO learned weight (with_scale=False).
    // This applies whenever `q_norm` is present (Gemma 4 always pairs q/k/v norms).
    // Qwen3-Next (attn_output_gate) has QK-norm but NO V-norm — exclude it.
    if kv_shared_override.is_none() && layer.q_norm_weight.is_some() && !layer.attn_output_gate {
        runtime.rms_norm_batched_no_weight_device(
            &scratch.v,
            num_kv_heads,
            head_dim,
            rms_norm_eps,
            &mut scratch.qk_norm_scratch,
        )?;
        runtime.copy_prefix_f32_device(
            &scratch.qk_norm_scratch,
            &mut scratch.v,
            num_kv_heads * head_dim,
        )?;
    }
    if let Ok(tag) = std::env::var("AEGIS_DUMP_QKV") {
        let q = runtime.download_f32(&scratch.q).unwrap();
        let k = runtime.download_f32(&scratch.k).unwrap();
        let v = runtime.download_f32(&scratch.v).unwrap();
        eprintln!("[DUMP {tag} Q post-norm] first8={:?}", &q[0..8]);
        eprintln!("[DUMP {tag} K post-norm] first8={:?}", &k[0..8]);
        eprintln!("[DUMP {tag} V post-norm] first8={:?}", &v[0..8]);
    }
    if layer.attn_output_gate {
        // Qwen3-Next: HF/GPT-NeoX partial RoPE (rotate first rotary_dim with
        // half-split within rotary_dim, divisor rotary_dim).
        let rd = rope.partial_dim as usize;
        runtime.apply_rope_neox_partial_device(
            &mut scratch.q, p_position, num_attention_heads, head_dim, rope.theta, rd,
        )?;
        if kv_shared_override.is_none() {
            runtime.apply_rope_neox_partial_device(
                &mut scratch.k, p_position, num_kv_heads, head_dim, rope.theta, rd,
            )?;
        }
    } else {
        runtime.apply_rope_ptr_device(
            &mut scratch.q,
            p_position,
            num_attention_heads,
            head_dim,
            rope,
        )?;
        if kv_shared_override.is_none() {
            runtime.apply_rope_ptr_device(&mut scratch.k, p_position, num_kv_heads, head_dim, rope)?;
        }
    }
    if let Ok(tag) = std::env::var("AEGIS_DUMP_QROPE") {
        thread_local! { static C: std::cell::RefCell<usize> = std::cell::RefCell::new(0); }
        let idx = C.with(|c| { let v = *c.borrow(); *c.borrow_mut() = v + 1; v });
        if idx == 0 || idx == 30 {
            // First call (BOS layer 0) and 30th call (Hi layer 0).
            let q = runtime.download_f32(&scratch.q).unwrap();
            let pos = runtime.download_u32(p_position).unwrap();
            eprintln!("[DUMP {tag} call#{} pos={:?} Q post-rope] {:?}", idx, pos, &q[0..8]);
        }
    }
    // Gemma 4 attention uses scaling=1.0 (NOT 1/sqrt(d)). Our attention kernels hardcode
    // softmax scale = rsqrt(head_dim). Pre-multiply Q by sqrt(head_dim) so the kernel's
    // built-in scaling cancels out and the effective Q·K^T is unscaled.
    // Qwen3-Next (attn_output_gate) uses STANDARD 1/sqrt(d) scaling — skip this.
    if layer.q_norm_weight.is_some() && !layer.attn_output_gate {
        let sqrt_d = (head_dim as f32).sqrt();
        runtime.scale_f32_device_len(sqrt_d, &mut scratch.q, num_attention_heads * head_dim)?;
    }

    if let Some(idx) = staging_slot_idx {
        // Host-resident KV: caller has pre-uploaded prior KV onto the staging slot
        // and the compute stream is synchronized against the H2D event. Run store +
        // attention against this slot; caller will schedule the D2H writeback.
        let pool = scratch.kv_staging.as_mut().ok_or_else(|| {
            aegisllm_base::error::AegisError::InvalidPlan(
                "host-resident KV cache requires kv_staging pool".into(),
            )
        })?;
        let staging = &mut pool.slots[idx];
        runtime.store_kv_ptr_device(
            &mut staging.keys,
            &mut staging.values,
            &scratch.k,
            &scratch.v,
            p_position,
            kv_width,
            effective_kv_capacity,
        )?;
        runtime.attention_decode_split_ptr_device(
            &staging.keys,
            &staging.values,
            &scratch.q,
            p_seq_len,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            layer.window_size,
            _seq_len,
            &mut scratch.attn_split_acc,
            &mut scratch.attn_split_m,
            &mut scratch.attn_split_l,
            &mut scratch.attn_context,
        )?;
    } else if let Some(parent_kv) = kv_shared_override {
        // Shared-layer attention: skip store_kv (parent already wrote at
        // this position); read directly from parent's KV cache.
        use crate::executor::state::KvBuffer;
        match (&parent_kv.keys, &parent_kv.values) {
            (KvBuffer::F16(keys), KvBuffer::F16(values)) => {
                runtime.attention_decode_split_ptr_device(
                    keys, values, &scratch.q, p_seq_len, num_attention_heads, num_kv_heads,
                    head_dim, layer.window_size, _seq_len,
                    &mut scratch.attn_split_acc,
                    &mut scratch.attn_split_m,
                    &mut scratch.attn_split_l,
                    &mut scratch.attn_context,
                )?;
            }
            (KvBuffer::Fp8(keys), KvBuffer::Fp8(values)) => {
                runtime.attention_decode_split_ptr_fp8_device(
                    keys, values, &scratch.q, p_seq_len, num_attention_heads, num_kv_heads,
                    head_dim, layer.window_size, _seq_len,
                    &mut scratch.attn_split_acc,
                    &mut scratch.attn_split_m,
                    &mut scratch.attn_split_l,
                    &mut scratch.attn_context,
                )?;
            }
            _ => return Err(aegisllm_base::error::AegisError::InvalidPlan(
                "KV-share: parent KV keys/values dtype mismatch".into(),
            )),
        }
    } else {
        use crate::executor::state::KvBuffer;
        // Branch on KV cache dtype: F16/BF16 (u16-backed) → existing kernels;
        // FP8 (u8-backed) → fp8 store + decode_split_fp8 kernels. The math is
        // identical, only the wire format of the KV stored bytes differs.
        match (&mut layer_state.kv.keys, &mut layer_state.kv.values) {
            (KvBuffer::F16(keys), KvBuffer::F16(values)) => {
                runtime.store_kv_ptr_device(
                    keys, values, &scratch.k, &scratch.v, p_position, kv_width, effective_kv_capacity,
                )?;
                runtime.attention_decode_split_ptr_device(
                    keys, values, &scratch.q, p_seq_len, num_attention_heads, num_kv_heads,
                    head_dim, layer.window_size,
                    _seq_len,
                    &mut scratch.attn_split_acc,
                    &mut scratch.attn_split_m,
                    &mut scratch.attn_split_l,
                    &mut scratch.attn_context,
                )?;
            }
            (KvBuffer::Fp8(keys), KvBuffer::Fp8(values)) => {
                runtime.store_kv_fp8_ptr_device(
                    keys, values, &scratch.k, &scratch.v, p_position, kv_width, effective_kv_capacity,
                )?;
                runtime.attention_decode_split_ptr_fp8_device(
                    keys, values, &scratch.q, p_seq_len, num_attention_heads, num_kv_heads,
                    head_dim, layer.window_size,
                    _seq_len,
                    &mut scratch.attn_split_acc,
                    &mut scratch.attn_split_m,
                    &mut scratch.attn_split_l,
                    &mut scratch.attn_context,
                )?;
            }
            _ => return Err(aegisllm_base::error::AegisError::InvalidPlan(
                "KV cache keys/values dtype mismatch (one F16, one FP8)".into(),
            )),
        }
    }
    if let Ok(tag) = std::env::var("AEGIS_DUMP_ATTNOUT") {
        thread_local! { static C2: std::cell::RefCell<usize> = std::cell::RefCell::new(0); }
        let target = std::env::var("AEGIS_DUMP_ATTNOUT_LAYER")
            .ok().and_then(|s| s.parse::<usize>().ok());
        let idx = C2.with(|c| { let v = *c.borrow(); *c.borrow_mut() = v + 1; v });
        // Decode-style call counter: layer L for token T = call#(T*num_layers + L). Default to first 2 tokens layer 0.
        let layer_match = target.map(|t| idx % 30 == t).unwrap_or(idx == 0 || idx == 30);
        if layer_match {
            let q = runtime.download_f32(&scratch.attn_context).unwrap();
            let pos = runtime.download_u32(p_position).unwrap();
            eprintln!("[DUMP {tag} call#{} pos={:?} attn_output] {:?}", idx, pos, &q[0..8]);
        }
    }
    // Qwen3-Next attention output gate: multiply the attention context by
    // sigmoid(gate) per head before the output projection.
    if let Some(ref gate) = gate_buf {
        runtime.sigmoid_gate_mul(&mut scratch.attn_context, gate, q_width)?;
    }
    match &layer.o_proj {
        CudaLinear::Nvfp4(o) => {
            if native_mxfp4_enabled(runtime, o) {
                matvec_nvfp4_device_with_scratch(
                    runtime, o, &scratch.attn_context,
                    &mut scratch.quant_hidden, &mut scratch.mxfp4_hidden,
                    &mut scratch.attn_out, scratch.staging_pool.as_deref_mut(),
                )?;
            } else if runtime.cutlass_nvfp4_inference_enabled_for(o) {
                runtime.matmul_cutlass_nvfp4_prefill_device(
                    o, &scratch.attn_context, 1,
                    &mut scratch.cutlass_payload, &mut scratch.cutlass_scales,
                    &mut scratch.cutlass_workspace, &mut scratch.attn_out,
                )?;
            } else {
                matvec_nvfp4_device_with_scratch(
                    runtime, o, &scratch.attn_context,
                    &mut scratch.quant_hidden, &mut scratch.mxfp4_hidden,
                    &mut scratch.attn_out, scratch.staging_pool.as_deref_mut(),
                )?;
            }
        }
        CudaLinear::Bf16(o) => {
            runtime.matvec_bf16_reference_device(o, &scratch.attn_context, &mut scratch.attn_out)?;
        }
        CudaLinear::Fp8(o) => {
            runtime.matvec_fp8_standalone_device(o, &scratch.attn_context, &mut scratch.attn_out)?;
        }
    }
    if let Some(ref post_norm) = layer.post_attn_sublayer_norm {
        // Gemma 4 PrePost: normalize attention output before adding to residual.
        runtime.rms_norm_device(&scratch.attn_out, post_norm, rms_norm_eps, &mut scratch.post_normed)?;
        runtime.add_device(hidden, &scratch.post_normed, &mut scratch.residual)?;
    } else {
        runtime.add_device(hidden, &scratch.attn_out, &mut scratch.residual)?;
    }
    Ok(())
}

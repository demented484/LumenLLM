use std::fs;
use std::path::Path;

use aegisllm_cuda::cuda::{CudaPrefillAttentionKernel, CudaRuntimeConfig};
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::generation::SamplingConfig;
use aegisllm_base::planning::placement::{
    ComputePlacement, LayerSelector, PlacementPolicy, PlacementRule, StoragePlacement,
    WeightQuantOverride,
};
use aegisllm_base::tensor::layout::{LinearLayoutChoice, LinearLayoutPolicy, MaterializationPolicy};
use aegisllm_base::tensor::quant::KvCacheQuantization;

use super::file::*;
use super::runtime::{EngineConfigFragment, ServeConfig};

impl ParametersFile {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        Ok(serde_json::from_slice(&fs::read(path)?)?)
    }

    pub fn into_serve_config(self, default_policy: PlacementPolicy) -> Result<ServeConfig> {
        let host = self
            .server
            .as_ref()
            .and_then(|server| server.host.clone())
            .unwrap_or_else(|| "127.0.0.1".into());
        let port = self
            .server
            .as_ref()
            .and_then(|server| server.port)
            .unwrap_or(1337);
        let api = self
            .server
            .as_ref()
            .and_then(|server| server.server_api.clone())
            .unwrap_or_else(|| "openai".into());

        Ok(ServeConfig {
            host,
            port,
            api,
            engine: self.into_engine_fragment(default_policy)?,
        })
    }

    pub fn into_engine_fragment(self, mut policy: PlacementPolicy) -> Result<EngineConfigFragment> {
        let cuda_device = self.cuda.as_ref().and_then(|cuda| cuda.device).unwrap_or(0);
        if let Some(device) = self.cuda.as_ref().and_then(|cuda| cuda.device) {
            retarget_cuda_policy(&mut policy, device);
        }
        let mut cuda_runtime = CudaRuntimeConfig::from_env();
        let mut explicit_cuda_prefill_attention = false;
        if let Some(cuda) = &self.cuda {
            if let Some(value) = cuda.native_mxfp4_repack {
                cuda_runtime.native_mxfp4_repack = value;
            }
            if let Some(value) = cuda.cutlass_nvfp4_repack {
                cuda_runtime.cutlass_nvfp4_repack = value;
            }
            if let Some(value) = cuda.native_mxfp4_inference {
                cuda_runtime.native_mxfp4_inference = value;
            }
            if let Some(value) = cuda.prefill_attention.as_deref() {
                cuda_runtime.prefill_attention = CudaPrefillAttentionKernel::parse(value)?;
                explicit_cuda_prefill_attention = true;
            }
            if let Some(value) = cuda.prefill_chunk_size {
                cuda_runtime.prefill_chunk_size = Some(value);
            }
            if let Some(value) = cuda.prefill_stage_timings {
                cuda_runtime.prefill_stage_timings = value;
            }
        }
        let model_path = self.model.path;

        // ── `model.{store, compute}` — defaults for non-block weights (embed, lm_head,
        //    final_norm). Also serves as the fallback tier for hidden-layers sub-sections
        //    when they don't specify their own `fallback-store/compute`.
        let mmap_enabled = self.model.mmap.unwrap_or(true);
        let model_store: Option<StoragePlacement> = self.model.store
            .as_deref()
            .map(|s| parse_storage(s, cuda_device))
            .transpose()?;
        let model_compute: Option<ComputePlacement> = self.model.compute
            .as_deref()
            .map(|c| parse_compute(c, cuda_device))
            .transpose()?;
        if let Some(store) = model_store {
            policy.weights_store = store;
            policy.spill_store = store;
            policy.kv_store = store;
        } else if !mmap_enabled && policy.weights_store == StoragePlacement::Mmap {
            policy.weights_store = StoragePlacement::Ram;
            policy.spill_store = StoragePlacement::Ram;
        }
        if let Some(compute) = model_compute {
            policy.weights_compute = compute;
            policy.spill_compute = compute;
            policy.kv_compute = compute;
        }

        // ── `input-layer` — embed_tokens placement override. Embed is a row
        //    lookup, so RAM placement is cheap (no matmul to stage).
        if let Some(input) = self.input_layer {
            let store = input.store.as_deref()
                .map(|s| parse_storage(s, cuda_device)).transpose()?;
            let compute = input.compute.as_deref()
                .map(|c| parse_compute(c, cuda_device)).transpose()?;
            if store.is_some() || compute.is_some() {
                policy.rules.push(PlacementRule {
                    selector: LayerSelector::Region("embed".into()),
                    store,
                    compute,
                });
            }
        }

        // ── `output-layer` — lm_head placement override.
        //
        // Note on perf: BF16 lm_head in RAM routes through the CPU-rayon
        // matvec at decode time (`matvec_bf16_host_resident_device`),
        // which is ~30ms/token vs ~1ms VRAM. Acceptable trade for 1 GiB
        // VRAM saved on memory-constrained configs; if you want faster
        // decode keep `output-layer.store = vram`.
        if let Some(output) = self.output_layer {
            let store = output.store.as_deref()
                .map(|s| parse_storage(s, cuda_device)).transpose()?;
            let compute = output.compute.as_deref()
                .map(|c| parse_compute(c, cuda_device)).transpose()?;
            if store.is_some() || compute.is_some() {
                policy.rules.push(PlacementRule {
                    selector: LayerSelector::Region("lm_head".into()),
                    store,
                    compute,
                });
            }
        }

        // ── `attention` — Q/K/V/O placement *within* each layer. Stored on
        //    the policy as a separate `attention_store_override`; the loader
        //    consults it before deciding to force-VRAM the BF16 attention
        //    weights. `mechanism` maps to `cuda.prefill_attention`.
        if let Some(attn) = self.attention {
            if let Some(mech) = attn.mechanism.as_deref() {
                cuda_runtime.prefill_attention =
                    CudaPrefillAttentionKernel::parse(mech)?;
                explicit_cuda_prefill_attention = true;
            }
            if let Some(store) = attn.store.as_deref() {
                policy.attention_store_override = Some(parse_storage(store, cuda_device)?);
            }
            if let Some(compute) = attn.compute.as_deref() {
                policy.attention_compute_override = Some(parse_compute(compute, cuda_device)?);
            }
            if let Some(q) = attn.attention_quantization.as_deref() {
                policy.attention_quantization = WeightQuantOverride::parse(q).ok_or_else(|| {
                    AegisError::InvalidConfig(format!(
                        "unsupported attention-quantization `{q}` \
                         (use one of: default, fp8, mxint4, int4, int8)"
                    ))
                })?;
            }
        }

        // ── `hidden-layers` — per-block weights and per-block KV cache.
        if let Some(hidden_layers) = self.hidden_layers {
            let parent_compute: Option<ComputePlacement> = hidden_layers.compute
                .as_deref()
                .map(|c| parse_compute(c, cuda_device))
                .transpose()?;
            // Shorthand `hidden-layers.store=...`: treat as weights.store
            // applied to all hidden layers when no nested `weights` block
            // is present.
            let parent_store: Option<StoragePlacement> = hidden_layers.store
                .as_deref()
                .map(|s| parse_storage(s, cuda_device))
                .transpose()?;
            if hidden_layers.weights.is_none() {
                // `hidden-layers.{store,compute}` shorthand applies to LAYER
                // regions only — embed, lm_head, and final_norm are governed
                // by `input-layer`, `output-layer`, and `model.*` respectively.
                // `LayerSelector::All` would (silently) overwrite those.
                let layer_selector = LayerSelector::Range { start: 0, end: usize::MAX };
                if let Some(store) = parent_store {
                    policy.rules.push(PlacementRule {
                        selector: layer_selector,
                        store: Some(store),
                        compute: parent_compute,
                    });
                } else if let Some(compute) = parent_compute {
                    policy.rules.push(PlacementRule {
                        selector: layer_selector,
                        store: None,
                        compute: Some(compute),
                    });
                }
            }
            if let Some(q) = hidden_layers.shared_mlp_quantization.as_deref() {
                policy.shared_mlp_quantization = WeightQuantOverride::parse(q).ok_or_else(|| {
                    AegisError::InvalidConfig(format!(
                        "unsupported shared-MLP-quantization `{q}` \
                         (use one of: default, fp8, mxint4, int4, int8)"
                    ))
                })?;
            }

            // ── weights sub-section ────────────────────────────────────────────
            if let Some(weights) = hidden_layers.weights {
                let primary_store = weights.store
                    .as_deref()
                    .map(|s| parse_storage(s, cuda_device))
                    .transpose()?;
                let primary_compute = weights.compute
                    .as_deref()
                    .map(|c| parse_compute(c, cuda_device))
                    .transpose()?
                    .or(parent_compute);
                let fallback_store = weights.fallback_store
                    .as_deref()
                    .map(|s| parse_storage(s, cuda_device))
                    .transpose()?;
                let fallback_compute = weights.fallback_compute
                    .as_deref()
                    .map(|c| parse_compute(c, cuda_device))
                    .transpose()?;

                // Tail (layers >= number) is described by the policy default
                // (weights_store/compute). Override it with fallback if specified.
                if let Some(store) = fallback_store {
                    policy.weights_store = store;
                }
                if let Some(compute) = fallback_compute {
                    policy.weights_compute = compute;
                }

                // First-N rule (or All if number is omitted).
                let selector = match weights.number {
                    Some(n) => LayerSelector::FirstN { n },
                    None => LayerSelector::All,
                };
                if primary_store.is_some() || primary_compute.is_some() {
                    policy.rules.push(PlacementRule {
                        selector,
                        store: primary_store,
                        compute: primary_compute,
                    });
                }
            }

            // ── kv-cache sub-section ───────────────────────────────────────────
            if let Some(kv) = hidden_layers.kv_cache {
                if let Some(context_size) = kv.context_size {
                    policy.context_size = context_size;
                }
                let primary_store = kv.store
                    .as_deref()
                    .map(|s| parse_storage(s, cuda_device))
                    .transpose()?;
                let fallback_store = kv.fallback_store
                    .as_deref()
                    .map(|s| parse_storage(s, cuda_device))
                    .transpose()?;
                if let Some(value) = kv.type_k.or(kv.type_v) {
                    policy.kv_quantization = KvCacheQuantization::parse(&value).ok_or_else(|| {
                        AegisError::InvalidConfig(format!(
                            "unsupported kv cache quantization `{value}`"
                        ))
                    })?;
                }

                // KV cache compute is implicit from the matching layer's weights compute
                // (set above). We only manage storage here.
                //
                // Mapping to policy fields:
                //   * `kv-cache.{number=N, store=A}`: first N layers use A; tail uses
                //     `fallback-store` if set, else `model.store`.
                //   * `kv-cache.{store=A}` (no number): ALL layers use A; first_n_layers=None.
                match (kv.number, primary_store, fallback_store) {
                    (Some(n), Some(first), fallback) => {
                        policy.kv_first_n_layers = Some(n);
                        policy.kv_first_store = Some(first);
                        policy.kv_store = fallback.unwrap_or(policy.kv_store);
                    }
                    (None, Some(store), _) => {
                        policy.kv_store = store;
                        policy.kv_first_n_layers = None;
                        policy.kv_first_store = None;
                    }
                    (Some(_), None, _) => {
                        return Err(AegisError::InvalidConfig(
                            "hidden-layers.kv-cache.number set but `store` missing; \
                             specify the store for the first-N tier or remove `number`."
                                .into(),
                        ));
                    }
                    (None, None, _) => {
                        // Nothing to do — keeps inherited values from `model.store`.
                    }
                }
            }
        }

        if let Some(layout) = self.linear_layout {
            apply_linear_layout_section(&mut policy.linear_layout, layout)?;
        }
        if let Some(other) = &self.other {
            if let Some(value) = other.cpu_linear_layout.as_deref() {
                policy.linear_layout.cpu = LinearLayoutChoice::parse(value)?;
            }
            if let Some(value) = other.cuda_linear_layout.as_deref() {
                policy.linear_layout.cuda = LinearLayoutChoice::parse(value)?;
            }
            if let Some(value) = other.linear_materialize.as_deref() {
                policy.linear_layout.materialization = MaterializationPolicy::parse(value)?;
            }
            if !explicit_cuda_prefill_attention && let Some(value) = other.flash_attention {
                cuda_runtime.prefill_attention = if value {
                    CudaPrefillAttentionKernel::Auto
                } else {
                    CudaPrefillAttentionKernel::Reference
                };
            }
            if cuda_runtime.prefill_chunk_size.is_none() {
                cuda_runtime.prefill_chunk_size = other.ubatch_size.or(other.batch_size);
            }
        }
        let mut generation = SamplingConfig::default();
        if let Some(other) = &self.other {
            if let Some(value) = other.temperature {
                generation.temperature = value;
            }
            if let Some(value) = other.top_p {
                generation.top_p = value;
            }
            if let Some(value) = other.top_k {
                generation.top_k = value;
            }
            if let Some(value) = other.min_p {
                generation.min_p = value;
            }
        }

        Ok(EngineConfigFragment {
            model_path,
            policy,
            cuda: cuda_runtime,
            generation,
        })
    }
}

pub(crate) fn retarget_cuda_policy(policy: &mut PlacementPolicy, device: usize) {
    policy.weights_store = retarget_store(policy.weights_store, device);
    policy.spill_store = retarget_store(policy.spill_store, device);
    policy.kv_store = retarget_store(policy.kv_store, device);
    policy.weights_compute = retarget_compute(policy.weights_compute, device);
    policy.spill_compute = retarget_compute(policy.spill_compute, device);
    policy.kv_compute = retarget_compute(policy.kv_compute, device);
    for rule in &mut policy.rules {
        rule.store = rule.store.map(|store| retarget_store(store, device));
        rule.compute = rule
            .compute
            .map(|compute| retarget_compute(compute, device));
    }
}

fn retarget_store(store: StoragePlacement, device: usize) -> StoragePlacement {
    match store {
        StoragePlacement::Vram { .. } => StoragePlacement::Vram { device },
        other => other,
    }
}

fn retarget_compute(compute: ComputePlacement, device: usize) -> ComputePlacement {
    match compute {
        ComputePlacement::Cuda { .. } => ComputePlacement::Cuda { device },
        other => other,
    }
}

fn apply_linear_layout_section(
    policy: &mut LinearLayoutPolicy,
    layout: LinearLayoutSection,
) -> Result<()> {
    if let Some(mode) = layout.mode.as_deref() {
        let choice = LinearLayoutChoice::parse(mode)?;
        policy.cpu = choice;
        policy.cuda = choice;
    }
    if let Some(cpu) = layout.cpu.as_deref() {
        policy.cpu = LinearLayoutChoice::parse(cpu)?;
    }
    if let Some(cuda) = layout.cuda.as_deref() {
        policy.cuda = LinearLayoutChoice::parse(cuda)?;
    }
    if let Some(materialize) = layout.materialize.as_deref() {
        policy.materialization = MaterializationPolicy::parse(materialize)?;
    }
    if let Some(value) = layout.max_extra_memory {
        policy.max_extra_memory_bytes = parse_optional_bytes(value)?;
    }
    Ok(())
}

fn parse_optional_bytes(value: serde_json::Value) -> Result<Option<u64>> {
    match value {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::String(value) if value.eq_ignore_ascii_case("auto") => Ok(None),
        serde_json::Value::String(value) => parse_bytes_string(&value).map(Some),
        serde_json::Value::Number(value) => value.as_u64().map(Some).ok_or_else(|| {
            AegisError::InvalidConfig("max-extra-memory must be a positive integer".into())
        }),
        other => Err(AegisError::InvalidConfig(format!(
            "unsupported max-extra-memory value `{other}`"
        ))),
    }
}

fn parse_bytes_string(value: &str) -> Result<u64> {
    let raw = value.trim().to_ascii_lowercase();
    let (digits, multiplier) = if let Some(number) = raw.strip_suffix("gib") {
        (number.trim(), 1024_u64.pow(3))
    } else if let Some(number) = raw.strip_suffix("gb") {
        (number.trim(), 1_000_000_000)
    } else if let Some(number) = raw.strip_suffix("mib") {
        (number.trim(), 1024_u64.pow(2))
    } else if let Some(number) = raw.strip_suffix("mb") {
        (number.trim(), 1_000_000)
    } else {
        (raw.as_str(), 1)
    };
    let value = digits
        .parse::<u64>()
        .map_err(|_| AegisError::InvalidConfig(format!("invalid byte value `{}`", value.trim())))?;
    value
        .checked_mul(multiplier)
        .ok_or_else(|| AegisError::InvalidConfig(format!("byte value `{}` overflows", value)))
}

pub fn parse_storage(value: &str, default_device: usize) -> Result<StoragePlacement> {
    match value.to_ascii_lowercase().as_str() {
        // "ram" is the user-facing name for host memory; the engine picks the most
        // memory-efficient internal residency (pinned vs mmap) based on the compute target.
        "ram" => Ok(StoragePlacement::Ram),
        "vram" | "gpu" => Ok(StoragePlacement::Vram {
            device: default_device,
        }),
        other if other.starts_with("vram:") => Ok(StoragePlacement::Vram {
            device: other
                .trim_start_matches("vram:")
                .parse::<usize>()
                .map_err(|_| AegisError::InvalidConfig(format!("invalid storage `{value}`")))?,
        }),
        "mmap" => Err(AegisError::InvalidConfig(
            "`mmap` is no longer accepted as a storage placement; use `ram` (the engine \
             will mmap weights internally when staging to a GPU)".into()
        )),
        _ => Err(AegisError::InvalidConfig(format!(
            "unsupported storage placement `{value}` (use `ram` or `vram`)"
        ))),
    }
}

pub fn parse_compute(value: &str, default_device: usize) -> Result<ComputePlacement> {
    match value.to_ascii_lowercase().as_str() {
        "cpu" => Ok(ComputePlacement::Cpu),
        "cuda" | "gpu" => Ok(ComputePlacement::Cuda {
            device: default_device,
        }),
        "wgpu" => Ok(ComputePlacement::Wgpu {
            device: default_device,
        }),
        other if other.starts_with("cuda:") => Ok(ComputePlacement::Cuda {
            device: other
                .trim_start_matches("cuda:")
                .parse::<usize>()
                .map_err(|_| AegisError::InvalidConfig(format!("invalid compute `{value}`")))?,
        }),
        other if other.starts_with("wgpu:") => Ok(ComputePlacement::Wgpu {
            device: other
                .trim_start_matches("wgpu:")
                .parse::<usize>()
                .map_err(|_| AegisError::InvalidConfig(format!("invalid compute `{value}`")))?,
        }),
        _ => Err(AegisError::InvalidConfig(format!(
            "unsupported compute placement `{value}`"
        ))),
    }
}

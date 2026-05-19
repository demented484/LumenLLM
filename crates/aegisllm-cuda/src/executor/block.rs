use std::collections::{BTreeMap, BTreeSet};

use super::loader::{CudaLayerShape, load_cuda_layer, runtime_layouts_by_region};
use super::state::{CudaLayer, CudaLayerState, CudaScratch};
use aegisllm_base::artifact::ModelArtifact;
use crate::cuda::{DECODE_SPLIT_K_MAX, CudaRuntime, CudaRuntimeConfig, DeviceBuffer};
use aegisllm_base::error::{AegisError, Result};
use aegisllm_base::graph::{ModelGraph, RegionId};
use aegisllm_base::planning::placement::{ComputePlacement, ResolvedPlacement};
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
        let cuda_weights = cuda.weight_loader();
        let region_placements = placement.region_map();
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
                    aegisllm_base::model::AttentionPattern::SlidingWindow { size } => Some(size),
                    _ => None,
                })
                .unwrap_or(0);
            let partial_dim = artifact.config.partial_rotary_factor
                .map(|factor| {
                    let is_global = matches!(
                        layer_meta.map(|m| &m.attention_pattern),
                        Some(aegisllm_base::model::AttentionPattern::FullCausal)
                    );
                    if is_global && factor < 1.0 {
                        (factor as f64 * graph.head_dim as f64).round() as usize
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
                        num_kv_heads: graph.num_kv_heads,
                        head_dim: graph.head_dim,
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

        Ok(Self {
            runtime: cuda,
            hidden_size: graph.hidden_size,
            num_attention_heads: graph.num_attention_heads,
            num_kv_heads: graph.num_kv_heads,
            head_dim: graph.head_dim,
            rms_norm_eps: artifact.config.rms_norm_eps.unwrap_or(1e-5) as f32,
            layers,
            kv_context_size: placement.kv_cache.context_size,
        })
    }

    #[allow(dead_code)]
    pub fn new_state(&self) -> Result<CudaLayerBlockState> {
        let kv_width = self.num_kv_heads * self.head_dim;
        let intermediate = self
            .layers
            .values()
            .filter(|l| l.moe.is_none())
            .map(|l| l.gate_proj.rows)
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
                // gate/up/down are always DeviceNvfp4Linear; q/k/v/o are CudaLinear
                let mut nvfp4s: Vec<&crate::cuda::DeviceNvfp4Linear> = vec![
                    &layer.gate_proj, &layer.up_proj, &layer.down_proj,
                ];
                for cl in [&layer.q_proj, &layer.k_proj, &layer.v_proj, &layer.o_proj] {
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
                .keys()
                .map(|&layer| {
                    Ok((
                        layer,
                        CudaLayerState {
                            kv: super::state::CudaKvCache::dense(
                                &self.runtime,
                                self.kv_context_size,
                                kv_width,
                                aegisllm_base::tensor::quant::KvCacheQuantization::F16,
                                self.kv_context_size,
                                // F16 quant -> no aux cache allocated regardless;
                                // this block path uses full-context capacity.
                                false,
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
                q: self
                    .runtime
                    .alloc_f32(self.num_attention_heads * self.head_dim)?,
                k: self.runtime.alloc_f32(kv_width)?,
                v: self.runtime.alloc_f32(kv_width)?,
                qk_norm_scratch: self
                    .runtime
                    .alloc_f32((self.num_attention_heads * self.head_dim).max(kv_width))?,
                attn_split_acc: self
                    .runtime
                    .alloc_f32(self.num_attention_heads * DECODE_SPLIT_K_MAX * self.head_dim)?,
                attn_split_m: self
                    .runtime
                    .alloc_f32(self.num_attention_heads * DECODE_SPLIT_K_MAX)?,
                attn_split_l: self
                    .runtime
                    .alloc_f32(self.num_attention_heads * DECODE_SPLIT_K_MAX)?,
                attn_context: self
                    .runtime
                    .alloc_f32(self.num_attention_heads * self.head_dim)?,
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
            },
            p_position: self.runtime.alloc_u32(1)?,
            p_seq_len: self.runtime.alloc_u32(1)?,
        })
    }
}

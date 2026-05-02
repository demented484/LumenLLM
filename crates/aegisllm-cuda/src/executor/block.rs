use std::collections::{BTreeMap, BTreeSet};

use super::loader::{CudaLayerShape, load_cuda_layer, runtime_layouts_by_region};
use super::rope::RopeConfig;
use super::state::{CudaLayer, CudaLayerState, CudaScratch};
use aegisllm_base::artifact::ModelArtifact;
use crate::cuda::{CudaRuntime, CudaRuntimeConfig, DeviceBuffer};
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
    pub(super) rope: RopeConfig,
    pub(super) layers: BTreeMap<usize, CudaLayer>,
    pub(super) kv_context_size: usize,
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct CudaLayerBlockState {
    pub(super) hidden: DeviceBuffer<f32>,
    pub(super) layers: BTreeMap<usize, CudaLayerState>,
    pub(super) scratch: CudaScratch,
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
            layers.insert(
                layer,
                load_cuda_layer(
                    &cuda_weights,
                    artifact,
                    layer,
                    region.kind,
                    placement,
                    resident_layout,
                    CudaLayerShape {
                        hidden_size: graph.hidden_size,
                        intermediate_size: graph.intermediate_size,
                        num_attention_heads: graph.num_attention_heads,
                        num_kv_heads: graph.num_kv_heads,
                        head_dim: graph.head_dim,
                    },
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
            rope: RopeConfig::from_artifact(artifact),
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
            .next()
            .map(|layer| layer.gate_proj.rows)
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
                [
                    &layer.q_proj,
                    &layer.k_proj,
                    &layer.v_proj,
                    &layer.o_proj,
                    &layer.gate_proj,
                    &layer.up_proj,
                    &layer.down_proj,
                ]
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
            },
        })
    }
}

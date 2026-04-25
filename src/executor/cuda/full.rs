use super::loader::{
    CudaLayerShape, cuda_residency_for_store, first_existing_tensor, load_cuda_layer,
    runtime_layouts_by_region,
};
use super::planning::validate_cuda_placement;
use super::rope::RopeConfig;
use super::state::{
    CudaLayerState, CudaLlamaExecutor, CudaLlamaState, CudaPrefillScratch, CudaScratch,
};
use crate::artifact::ModelArtifact;
use crate::cuda::{CudaRuntime, CudaRuntimeConfig};
use crate::error::{AegisError, Result};
use crate::executor::tensors::require_tensor;
use crate::graph::{ModelGraph, RegionId};
use crate::planning::placement::ResolvedPlacement;
use crate::planning::runtime::RuntimePlan;
use crate::tensor::layout::LinearResidentLayout;
use crate::tensor::storage::TensorStorageLoader;

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
        if graph.num_kv_heads == 0 || graph.num_attention_heads % graph.num_kv_heads != 0 {
            return Err(AegisError::InvalidPlan(format!(
                "CUDA executor requires attention heads divisible by kv heads, got heads={} kv_heads={}",
                graph.num_attention_heads, graph.num_kv_heads
            )));
        }
        let cuda = CudaRuntime::new_with_config(device, cuda_config)?;
        let cuda_weights = cuda.weight_loader();
        let region_placements = placement.region_map();
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

        let embed_tokens = cuda_weights.load_bf16_matrix_with_store(
            require_tensor(artifact, "model.embed_tokens.weight")?,
            embed_region.store,
            cuda_residency_for_store(embed_region.store, device)?,
            &mut loader,
        )?;
        let final_norm = cuda_weights.load_dense_vector_with_store(
            require_tensor(artifact, "model.norm.weight")?,
            final_norm_region.store,
            &mut loader,
        )?;
        let lm_head_tensor =
            first_existing_tensor(artifact, &["lm_head.weight", "model.embed_tokens.weight"])?;
        let lm_head = cuda_weights.load_bf16_matrix_with_store(
            lm_head_tensor,
            lm_head_region.store,
            cuda_residency_for_store(lm_head_region.store, device)?,
            &mut loader,
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
            layers.push(load_cuda_layer(
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
            )?);
        }

        Ok(Self {
            runtime: cuda,
            hidden_size: graph.hidden_size,
            num_attention_heads: graph.num_attention_heads,
            num_kv_heads: graph.num_kv_heads,
            head_dim: graph.head_dim,
            rms_norm_eps: artifact.config.rms_norm_eps.unwrap_or(1e-5) as f32,
            rope: RopeConfig::from_artifact(artifact),
            embed_tokens,
            final_norm,
            lm_head,
            layers,
            kv_context_size: placement.kv_cache.context_size,
            prefill_chunk_size: cuda_prefill_chunk_size(cuda_config),
        })
    }

    pub(super) fn new_state(&self) -> Result<CudaLlamaState> {
        let kv_width = self.num_kv_heads * self.head_dim;
        let intermediate = self
            .layers
            .first()
            .map(|layer| layer.gate_proj.rows)
            .unwrap_or(self.hidden_size);
        let prefill = if self.prefill_chunk_size > 1 {
            let prefill_max_sequences = 1;
            let prefill_block_table_capacity = self.kv_context_size.div_ceil(16).max(1);
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
                hidden_out: self
                    .runtime
                    .alloc_f32(self.prefill_chunk_size * self.hidden_size)?,
                input_normed: self
                    .runtime
                    .alloc_f32(self.prefill_chunk_size * self.hidden_size)?,
                quant_hidden: self
                    .runtime
                    .alloc_f32(self.prefill_chunk_size * self.hidden_size)?,
                quant_intermediate: self
                    .runtime
                    .alloc_f32(self.prefill_chunk_size * intermediate)?,
                mxfp4_hidden: self.runtime.alloc_u8(
                    self.prefill_chunk_size * CudaRuntime::mxfp4_vector_bytes(self.hidden_size)?,
                )?,
                mxfp4_intermediate: self.runtime.alloc_u8(
                    self.prefill_chunk_size * CudaRuntime::mxfp4_vector_bytes(intermediate)?,
                )?,
                q: self.runtime.alloc_f32(
                    self.prefill_chunk_size * self.num_attention_heads * self.head_dim,
                )?,
                k: self.runtime.alloc_f32(self.prefill_chunk_size * kv_width)?,
                v: self.runtime.alloc_f32(self.prefill_chunk_size * kv_width)?,
                attn_context: self.runtime.alloc_f32(
                    self.prefill_chunk_size * self.num_attention_heads * self.head_dim,
                )?,
                attn_out: self
                    .runtime
                    .alloc_f32(self.prefill_chunk_size * self.hidden_size)?,
                residual: self
                    .runtime
                    .alloc_f32(self.prefill_chunk_size * self.hidden_size)?,
                post_normed: self
                    .runtime
                    .alloc_f32(self.prefill_chunk_size * self.hidden_size)?,
                gate: self
                    .runtime
                    .alloc_f32(self.prefill_chunk_size * intermediate)?,
                up: self
                    .runtime
                    .alloc_f32(self.prefill_chunk_size * intermediate)?,
                swiglu: self
                    .runtime
                    .alloc_f32(self.prefill_chunk_size * intermediate)?,
                mlp_out: self
                    .runtime
                    .alloc_f32(self.prefill_chunk_size * self.hidden_size)?,
            })
        } else {
            None
        };

        Ok(CudaLlamaState {
            position: 0,
            hidden: self.runtime.alloc_f32(self.hidden_size)?,
            logits: self.runtime.alloc_f32(self.lm_head.rows)?,
            sampled_token: self.runtime.alloc_u32(1)?,
            layers: (0..self.layers.len())
                .map(|_| {
                    Ok(CudaLayerState {
                        kv: super::state::CudaKvCache::dense(
                            &self.runtime,
                            self.kv_context_size,
                            kv_width,
                        )?,
                    })
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
                argmax_block_values: self.runtime.alloc_f32(self.lm_head.rows.div_ceil(256))?,
                argmax_block_indices: self.runtime.alloc_u32(self.lm_head.rows.div_ceil(256))?,
            },
            prefill,
            prefill_timings: super::state::CudaPrefillStageTimings::from_env(),
        })
    }
}

fn cuda_prefill_chunk_size(config: CudaRuntimeConfig) -> usize {
    config.prefill_chunk_size.unwrap_or(128).clamp(1, 2048)
}

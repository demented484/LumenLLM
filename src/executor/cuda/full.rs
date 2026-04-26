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

const FLASH_COMPAT_PREFILL_KV_PAGE_TOKENS: usize = 256;
const PREFILL_SPLIT_K_TOKENS: usize = 256;
const PREFILL_SPLIT_Q_BLOCK: usize = 4;

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
            prefill_stage_timings_enabled: cuda_config.prefill_stage_timings,
        })
    }

    pub(super) fn new_state(&self) -> Result<CudaLlamaState> {
        let kv_width = self.num_kv_heads * self.head_dim;
        let intermediate = self
            .layers
            .first()
            .map(|layer| layer.gate_proj.rows)
            .unwrap_or(self.hidden_size);
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
                cutlass_payload: self
                    .runtime
                    .alloc_u8(cutlass_prefill_scratch.payload_bytes)?,
                cutlass_scales: self.runtime.alloc_u8(cutlass_prefill_scratch.scale_bytes)?,
                cutlass_workspace: self
                    .runtime
                    .alloc_u8(cutlass_prefill_scratch.workspace_bytes)?,
                q: self.runtime.alloc_f32(
                    self.prefill_chunk_size * self.num_attention_heads * self.head_dim,
                )?,
                q_half: self.runtime.alloc_u16(
                    self.prefill_chunk_size * self.num_attention_heads * self.head_dim,
                )?,
                attn_split_acc: self.runtime.alloc_f32(prefill_attention_scratch.acc_f32)?,
                attn_split_m: self
                    .runtime
                    .alloc_f32(prefill_attention_scratch.stats_f32)?,
                attn_split_l: self
                    .runtime
                    .alloc_f32(prefill_attention_scratch.stats_f32)?,
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
                cutlass_payload: self
                    .runtime
                    .alloc_u8(cutlass_decode_scratch.payload_bytes)?,
                cutlass_scales: self.runtime.alloc_u8(cutlass_decode_scratch.scale_bytes)?,
                cutlass_workspace: self
                    .runtime
                    .alloc_u8(cutlass_decode_scratch.workspace_bytes)?,
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
            prefill_timings: super::state::CudaPrefillStageTimings::from_enabled(
                self.prefill_stage_timings_enabled,
            ),
        })
    }
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
    if std::env::var_os("AEGISLLM_CUDA_EXPERIMENTAL_SPLIT_K_ATTENTION").is_none() {
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
    let acc_f32 = rows
        .checked_mul(executor.head_dim)
        .ok_or_else(|| AegisError::InvalidPlan("prefill split attention acc overflow".into()))?;
    Ok(PrefillAttentionSplitScratchBytes {
        acc_f32: acc_f32.max(1),
        stats_f32: rows.max(1),
    })
}

fn cuda_prefill_chunk_size(config: CudaRuntimeConfig) -> usize {
    config.prefill_chunk_size.unwrap_or(128).clamp(1, 2048)
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
    let workspace_bytes = if executor.layers.iter().any(|layer| {
        [
            &layer.q_proj,
            &layer.k_proj,
            &layer.v_proj,
            &layer.o_proj,
            &layer.gate_proj,
            &layer.up_proj,
            &layer.down_proj,
        ]
        .into_iter()
        .any(|linear| executor.runtime.cutlass_nvfp4_inference_enabled_for(linear))
    }) {
        executor
            .layers
            .iter()
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

// wgpu (Vulkan/Metal/D3D12) backend skeleton for aegisllm.

pub mod wgpu;

pub use wgpu::{
    // Provider + context.
    WgpuContext, WgpuExecutorProvider,
    // Per-layer state (legacy single-layer constructor used by attention-block test).
    WgpuLlamaState,
    // Multi-layer state (used by forward_token_device).
    WgpuModelState,
    // Layer / model weight types.
    WgpuAttentionWeightsFull, WgpuDenseMlpWeights, WgpuLayerWeights, WgpuLinear,
    WgpuMlpWeightsFull, WgpuModel, WgpuModelShape, WgpuAttentionWeights, WgpuMoeExpert,
    WgpuMoeWeights,
    // Layer block forward fns + activation enum.
    forward_attention_block_device, forward_dense_mlp_block_device, forward_layer_device,
    forward_moe_block_device, forward_token_device, Activation,
    // Loader entry points.
    load_gemma4_model, load_vanilla_llama_model,
    // Host-API primitive wrappers (for unit tests of individual kernels).
    decode_attention_gpu, dequant_nvfp4_gpu, embedding_gpu, matmul_f32_gpu, residual_add_gpu,
    rms_norm_gpu, rope_gpu, swiglu_gpu,
    // Device-resident primitive wrappers.
    alloc_storage, decode_attention_device, dequant_nvfp4_device, download_f32_buf,
    embedding_device, geglu_tanh_device, matmul_f32_device, residual_add_device,
    rms_norm_batched_device, rms_norm_device, rope_device, scale_f32_device, swiglu_device,
    upload_f32_buf, upload_padded_u8_buf,
};

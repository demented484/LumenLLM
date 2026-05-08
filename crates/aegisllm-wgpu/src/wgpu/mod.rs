mod block;
mod forward;
mod loader;
mod provider;
mod state;
mod weights;

pub use weights::{
    load_gemma4_model, load_vanilla_llama_model, WgpuAttentionWeightsFull, WgpuLayerWeights,
    WgpuLinear, WgpuMlpWeightsFull, WgpuModel, WgpuModelShape, WgpuMoeExpert, WgpuMoeWeights,
};

pub use block::{
    forward_attention_block_device, forward_dense_mlp_block_device, forward_layer_device,
    forward_moe_block_device, forward_token_device, Activation, WgpuAttentionWeights,
    WgpuDenseMlpWeights,
};
pub use state::{WgpuLlamaState, WgpuModelState};

pub use forward::{
    decode_attention_gpu, dequant_nvfp4_gpu, embedding_gpu, matmul_f32_gpu, residual_add_gpu,
    rms_norm_gpu, rope_gpu, swiglu_gpu,
};
pub use forward::{
    alloc_storage, decode_attention_device, decode_attention_device_full,
    decode_attention_device_strided, dequant_bf16_device, dequant_nvfp4_device,
    download_f32_buf, embedding_device, geglu_tanh_device, matmul_bf16_device,
    matmul_f32_device, residual_add_device, rms_norm_batched_device, rms_norm_device,
    rope_device, scale_f32_device, swiglu_device, upload_bf16_packed_buf, upload_f32_buf,
    upload_padded_u8_buf,
};
pub use loader::WgpuContext;
pub use provider::WgpuExecutorProvider;

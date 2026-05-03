// wgpu (Vulkan/Metal/D3D12) backend skeleton for aegisllm.

pub mod wgpu;

pub use wgpu::{
    decode_attention_gpu, embedding_gpu, matmul_f32_gpu, residual_add_gpu, rms_norm_gpu, rope_gpu,
    swiglu_gpu, WgpuContext, WgpuExecutorProvider, WgpuLlamaState,
};

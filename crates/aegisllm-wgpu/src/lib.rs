// wgpu (Vulkan/Metal/D3D12) backend skeleton for aegisllm.

pub mod wgpu;

pub use wgpu::{rms_norm_gpu, WgpuContext, WgpuExecutorProvider, WgpuLlamaState};

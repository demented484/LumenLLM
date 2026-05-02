// CUDA-related enum types that need to be visible to non-cuda crates
// (planning, tensor, backend registry). The actual CUDA runtime types
// (DeviceBuffer, CudaRuntime, etc.) live in aegisllm-cuda.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum CudaAttentionDType {
    F32 = 0,
    F16 = 1,
    Bf16 = 2,
    Fp8E4M3 = 3,
    Fp8E5M2 = 4,
    Fp4E2M1 = 5,
    Int8 = 6,
    Int4 = 7,
}

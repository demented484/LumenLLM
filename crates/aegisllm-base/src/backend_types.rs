// Backend-agnostic enum types that need to be visible to all crates
// (planning, tensor, backend registry, every executor backend). Backend-
// specific runtime types (CUDA's DeviceBuffer / CudaRuntime, wgpu's
// device handles, CPU's tensor views) live in their own crates.
//
// Used to be `cuda_types::CudaAttentionDType` — renamed because every
// backend (CPU, wgpu, CUDA) needs to advertise which attention compute
// dtypes it supports, and the prior naming wrongly implied CUDA-only.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum AttentionDType {
    F32 = 0,
    F16 = 1,
    Bf16 = 2,
    Fp8E4M3 = 3,
    Fp8E5M2 = 4,
    Fp4E2M1 = 5,
    Int8 = 6,
    Int4 = 7,
}

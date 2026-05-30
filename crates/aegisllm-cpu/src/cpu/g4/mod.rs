//! Gemma-4 CPU forward path. Architecture-driven (NOT hardcoded): selected
//! by `detect_architecture(...).name() == "gemma4"` in `cpu/provider.rs`.
//!
//! Implements the missing Gemma-4 primitives — per-head q/k norm + no-weight
//! v-norm, partial RoPE, sliding-window attention with scale folded to 1.0,
//! per-layer head_dim/num_kv_heads from `ModelGraph.layer_metadata`,
//! embed_scale, PLE, MoE router+experts — plus the forward orchestration.
//!
//! Correctness-first: BF16 weights convert to f32 lazily per matvec; a fast
//! blocked SIMD BF16 GEMM is a follow-up. The op-order matches the CUDA decode
//! path exactly (see file-level docs in each submodule).

mod attention;
mod forward;
mod linear;
mod loader;
mod moe;
mod norm;
mod ple;
mod rope;
mod state;

// `G4CpuExecutor` / `G4CpuState` are `pub` (their fields stay crate-private) so
// the hybrid (CPU+GPU) executor in the `aegisllm` crate can own one and drive
// the per-layer block API (`token_entry_host`, `forward_dense_layer_host`, …).
pub use state::{G4CpuExecutor, G4CpuState};

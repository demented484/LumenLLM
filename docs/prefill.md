# Prefill Completion Plan

This is the definition of done for making CUDA prefill a product-quality path.

## Correctness

- Chunked CUDA prefill must match token-by-token CUDA generation for deterministic sampling across chunk sizes `1`, `2`, `7`, `8`, `64`, `128`, `512`, and the configured production chunk.
- The comparison must cover `start_position = 0` and continuation after an existing prefix.
- Attention tests must cover causal masking inside the fresh chunk: query row `i` can see all prefix tokens and fresh rows `0..=i`, never future fresh rows.
- Native MXFP4 prefill must have a separate tolerance contract from scalar NVFP4 reference because the resident layout requantizes weights into MXFP4.

## Memory

- The prefill hot path must not allocate per chunk on the host or device.
- Scratch kernels must operate on active `batch * width`, not full scratch capacity.
- Attention `auto` must never exceed safe dynamic shared-memory limits; long prefixes use bounded-memory continuation.
- Planning/reporting must account for per-sequence prefill scratch in addition to persistent weights and KV cache.

## Throughput

- Q/K/V, gate/up, and down projections should use batched tensor-core FP4 paths when native MXFP4 inference is enabled.
- Current `m16n8k64` native path is the minimum viable batched tile. The next target is tile-major resident weights plus GEMM-friendly activation layout so one CTA computes larger `M x N` tiles.
- Prefill stage timings now include observed QKV/MLP TFLOPS when `AEGISLLM_CUDA_STAGE_TIMINGS=1` is set.
- `bench-generate` supports `--format text|json|csv` so prompt/chunk sweeps can be consumed by scripts.
- Release benchmarks should include short prompt (`~400 tokens`), medium prompt (`~2k`), and long prompt (`~4k+`).

## Attention Roadmap

- Current `auto` uses cache-only dense attention while shared memory is safe and falls back to bounded-memory continuation for long prefixes.
- The host/device prefill descriptor now carries `request_ids`, `seq_ids`, `token_ids`, `positions`, `slot_mapping`, `cu_q`, `cu_k`, `context_lens`, `block_tables`, `max_q`, `max_k`, and prefill/decode token counts. The current CUDA kernel still consumes the dense compatibility subset.
- Paged KV has an allocator/slot-mapping scaffold. Production work remains: GPU block-table reads, eviction, prefix reuse, and request-owned lifetime cleanup.
- A true FlashAttention-class path needs tiled online softmax across K blocks, not one CTA per query row scanning the whole prefix serially.
- CUDA-specific kernels must remain behind backend primitives so ROCm, oneAPI, CPU, and future NPU backends can provide their own attention implementation.

## Current Production Gaps

- Native MXFP4 prefill still dispatches through the existing batched `m16n8k64` path. It has a real `M=tokens, N=rows, K=cols` API and TFLOPS accounting now, but the next speed jump requires a larger tile-major GEMM kernel or CUTLASS/cuBLASLt-backed implementation.
- FlashAttention is not production FlashAttention yet. The ABI is varlen-shaped and the scheduler/KV descriptors exist, but the kernel does not yet consume paged block tables with online softmax.
- Continuous batching has a budgeted scheduler scaffold only. It still needs integration with the HTTP server request lifecycle and per-request KV ownership.
- CUDA Graph policy types exist so replay can be added without changing the public runtime shape, but no graph capture/replay is active yet.

## Exit Criteria

- `cargo test`, `quality-smoke`, `cuda-compare`, and prefill correctness smoke pass.
- `cuda-prefill-sweep` passes chunk sizes `1,2,3,7,8,16,31,32,64,128,512,2048` against token-by-token CUDA.
- No per-chunk host allocation remains in the CUDA prefill loop.
- Short-prompt native FP4 prefill is consistently above the previous scalar/native-matvec baseline.
- Long-prompt prefill uses bounded shared memory and has a clear next bottleneck documented by benchmark output.

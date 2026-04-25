# Prefill Architecture Closure Plan

This document tracks the full prefill work, not just one CUDA kernel.

## Phase 1: Dense Adapter Over Batch Metadata

- Keep the public single-request `prefill_prompt(&[usize])` API stable.
- Internally build a `CudaPrefillBatch` for every chunk.
- Dense identity metadata:
  - `positions[i] = start_position + i`
  - `slot_mapping[i] = start_position + i`
  - `cu_q = [0, batch]`
  - `context_lens = [start_position + batch]`
  - `num_sequences = 1`
- All metadata must be uploaded once per chunk and reused from pre-allocated scratch buffers.
- Positions and slots must be bounds-checked before casting to CUDA `u32`.

## Phase 2: Slot-Mapped KV Store

- Prefill KV writes go through `slot_mapping`, not `start_position`.
- Dense mode uses identity slots, so existing decode remains compatible.
- Invalid/out-of-range slots are rejected before launch for host-known dense batches.
- Non-identity slots are not enabled until attention reads through slot/block metadata.

## Phase 3: Varlen-Shaped Attention ABI

- Runtime accepts varlen-shaped metadata even while the first implementation is a dense adapter.
- Required ABI fields:
  - `slot_mapping`
  - `cu_q`
  - `context_lens`
  - `num_sequences`
  - future `block_table`, `block_size`, `max_q`, `max_k`
- Dense adapter carries a host-side `DensePrefillMetadataProof` and is allowed to delegate to existing kernels only when invariants prove identity layout and `num_sequences == 1`.
- Real varlen kernels must read K/V through logical-to-physical mapping.

## Phase 4: Pooled/Paged KV

- Wrap per-layer `keys`/`values` in a `CudaKvCache` handle.
- Start with a flat dense pool: `capacity = context_size` slots.
- Add `Paged` layout metadata without enabling it in attention yet.
- Future paged mode allocates pages `[num_blocks, block_size, kv_heads, head_dim]` and uses `block_table` plus `slot_mapping`.

## Phase 5: Positions-Based RoPE

- Batched RoPE must consume explicit `positions`, not assume contiguous `start_position + row`.
- Dense mode still supplies contiguous positions.
- Multi-sequence varlen mode only becomes legal after this stage.

## Phase 6: Stage-Level Timings

- Debug timings are opt-in through `AEGISLLM_CUDA_STAGE_TIMINGS`.
- Timings synchronize the CUDA stream after each stage, so they are diagnostic timings, not normal benchmark timings.
- `prepare_us` covers dense metadata validation and H2D uploads before embedding starts.
- Stages:
  - prepare
  - embed
  - QKV
  - RoPE
  - KV store
  - attention
  - O projection
  - MLP
  - sample/final head

## Phase 7: Correctness And Benchmark Gates

- `cargo test`
- `cuda-prefill-compare --config ...`
- `quality-smoke --config ...`
- `bench-generate` short/medium/long prompt sweeps
- Required future tests:
  - chunk sizes `1,2,7,8,64,128,512`
  - first prefill and continuation
  - identity slot mapping
  - rejected non-identity mapping until mapped attention exists
  - GQA/MQA valid shapes and invalid head divisibility

## Current Closure Boundary

The current implementation closes the dense single-sequence adapter and prepares the internal ABI. Full vLLM-style continuous batching is the next layer: request scheduler, multiple live sequences per step, paged KV allocation, and mapped attention kernels.

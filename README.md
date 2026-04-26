# aegisllm.rs

Rust inference-engine rewrite focused on explicit storage and compute placement.

The important architectural split is:

- `hardware`: discovers CPU/RAM and CUDA/VRAM inventory.
- `artifact` + `tensor`: reads Hugging Face safetensors metadata without loading all weights.
- `graph`: turns model tensors into stable regions such as embedding, transformer blocks, final norm and lm head.
- `placement`: resolves manual policy into per-region `store` and `compute` decisions.
- `memory`: dry-runs persistent RAM/VRAM allocations, file-backed mmap, KV cache, and staging peaks separately.
- `backend`: records available CPU/CUDA backend capabilities.
- `layout` + `materialization`: decide whether linear weights stay packed, become native tensor-core resident, or are explicitly repacked for a backend.
- `runtime`: selects kernel families from backend capabilities and tensor quantization instead of hardcoding one GPU generation.
- `executor`: provider boundary for `cpu-reference`, `cuda`, and `hybrid` execution plans.
- `cuda` + `cpu`: backend runtimes with explicit resident tensor loaders and matvec primitives. CUDA runtime config is explicit and can be supplied from `parameters.json`; environment variables are only compatibility defaults for CLI/manual use.
- `engine`: ties the above together, reports the plan, and only builds an executor for real generation.

The current runnable paths are all-CPU, all-CUDA, and correctness-first CPU/CUDA hybrid. CUDA generation keeps weights and activations device-resident, supports BF16 dense tensors, NVFP4 linears, RoPE, RMSNorm, MLP, decode attention, chunked prefill, native Blackwell MXFP4 linear dispatch, and OpenAI/Anthropic/Google-compatible serving adapters.

`mvp-check` is intentionally stricter than a dry plan: after readiness says a plan is runnable, it builds the real executor so CUDA module compilation, tensor loading, and backend construction failures are caught before serving.

Useful commands:

```bash
cargo run -- inspect-hardware
cargo run -- serve --config ../parameters.json
cargo run -- show-plan --config ../parameters.json
cargo run -- mvp-check --config ../parameters.minimal.json
cargo run -- cuda-smoke --config ../parameters.cuda.layer1.json
cargo run -- cuda-dense-smoke --config ../parameters.json
cargo run -- cuda-chain-smoke --config ../parameters.cuda.layer1.json
cargo run -- cuda-compare --config ../parameters.cuda.layer1.json
cargo run -- cuda-prefill-sweep --config ../parameters.json
cargo run --release -- generate --config ../parameters.minimal.json --prompt "Привет" --max-tokens 1
cargo run --release -- bench-generate --config ../parameters.minimal.json --prompt "Привет" --prompt-repeat 8 --max-tokens 16 --temperature 0 --format json
python3 tools/bench_vllm_generate.py --model ../models/Llama-3.1-8B-Instruct-NVFP4 --prompt "Привет" --prompt-repeat 8 --max-tokens 16 --temperature 0
```

`serve` exposes `/health` and `/v1/models` for all compatibility modes. With `server-api=openai` it exposes `/v1/completions` and `/v1/chat/completions`; with `server-api=anthropic` it exposes `/v1/messages`; with `server-api=google` it exposes `/v1beta/models/{model}:generateContent`. If the selected provider is not runnable yet, the server still starts in degraded mode and generation requests return a structured `executor_not_ready` error with the current readiness limitations.

CUDA notes:

- Blackwell FP4 prefill can use the CUTLASS resident layout (`CUDA_R_4F_E2M1`
  payload plus UE4M3 scales) for Q/K/V/O and MLP projections. The CUTLASS
  bridge does not own CUDA allocations; Rust allocates and lifetime-manages
  payloads, scales, workspaces and outputs.
- Experimental native MXFP4 repack/inference is opt-in via:
  - `cuda.native-mxfp4-repack`
  - `cuda.native-mxfp4-inference`
  - CLI equivalents: `--native-mxfp4-repack` and `--native-mxfp4-inference`
- Native MXFP4 repack writes a per-model cache under `.aegis-cache/mxfp4-v1` so repeated benchmark runs do not spend most of their time repacking weights on the host. Set `AEGISLLM_NATIVE_MXFP4_CACHE=0` to disable it.
- Chunked CUDA prefill uses `AEGIS_CUDA_PREFILL_CHUNK` for the token chunk size. The default is `128`; values are clamped to `1..=2048`.
- CUDA prefill attention is selected by `cuda.prefill-attention` (`auto`, `fa4`, `flash-varlen`, `warp-flash`, `continuation`, or `reference`) or the legacy `other-parameters.flash-attention` boolean. Explicit `cuda.prefill-attention` wins over the legacy flag. `auto` keeps the correctness-first reference kernel for shorter chunks and switches to the paged varlen online-softmax path for longer chunks; `fa4` is an explicit Blackwell-only tiled paged-varlen prototype and is not selected automatically until it beats the current fast path on correctness and throughput; `warp-flash` remains an explicit experimental opt-in.
- The paged varlen prefill path uses an FA-compatible 256-token page table and
  a transient f16 query view while keeping f32 outputs for the surrounding
  residual/MLP path. The current fast path is a single-sequence block-Q
  online-softmax kernel with shared K/V reuse and warp reductions; multi-request
  and mixed scheduler paths still fall back to the more general varlen kernel.
- An experimental split-K prefill attention scaffold exists for future
  long-context tuning. It is opt-in with
  `AEGISLLM_CUDA_EXPERIMENTAL_SPLIT_K_ATTENTION=1`; by default its large partial
  accumulator scratch is not allocated.
- CUTLASS FP4 MLP prefill fuses SwiGLU directly into the down-projection FP4
  activation layout, avoiding a transient f32 SwiGLU buffer before the down GEMM.
- Set `AEGISLLM_CUDA_STAGE_TIMINGS=1` to print per-stage prefill timings plus QKV/MLP TFLOPS estimates.
- Recent RTX 5070 Ti smoke numbers with the CUTLASS FP4 config and one generated
  token: prompt-repeat 16 / 202 prompt tokens ≈ 4.4k prefill tok/s,
  prompt-repeat 64 / 778 prompt tokens ≈ 2.6k prefill tok/s,
  prompt-repeat 128 / 1546 prompt tokens ≈ 1.7k prefill tok/s. The long-context
  drop is expected until the attention kernel grows a true block-K split/reduce
  FlashAttention path.
- Reports distinguish planned families from effective dispatch. With native MXFP4 disabled, planned native FP4 regions still run through the CUDA NVFP4 reference path.
- KV cache defaults to f16 and is stored as f16 in the CUDA reference executor. q8/fp8 KV kernels are not implemented yet and are rejected by CUDA readiness.
- The explicit storage/compute plan is still authoritative: full CUDA, full CPU, and host-orchestrated hybrid plans run without silently falling back to one backend.

Memory policy notes:

- Prefer `mmap` for cold/spilled weights so CPU RAM is not consumed by a second copy.
- Count KV cache before deciding how many layers can live in VRAM.
- Report staging peaks separately from persistent allocations.
- Treat usable RAM/VRAM budget overflows as engine-build errors for MVP paths.
- Use `weights-store vram` only when the whole selected residency fits; otherwise let the planner spill to `mmap`.

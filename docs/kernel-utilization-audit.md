# Prefill Kernel-Utilization Audit

**Scope:** every prefill compute stage of aegisllm.rs, model Gemma-4-26B-A4B-NVFP4,
RTX 5070 Ti (SM120, ~175-180 TFLOPS dense BF16 tensor-core peak, NVFP4 path higher).
**Method:** static code reading only — no engine runs. Measured timings supplied by the user.
**Repo:** `main @ 9545f21`. **Branch:** `audit/kernel-utilization`.

## Critical dispatch fact

Gemma-4-26B-A4B has **BF16 attention weights** (q/k/v/o) and a **BF16 shared MLP +
BF16 router**; only the 128 **routed experts are NVFP4**. In `prefill/layer.rs` the
`layer.q_proj.as_nvfp4()` test returns `None`, so QKV/o_proj take the **BF16/FP8
branch** → `matmul_bf16_cublaslt_device`. The `matmul_bf16_reference_batched_device`
and `matvec_bf16_reference_device` kernels (the "reference" red flags) are **only
reached when a weight is host-resident** (`cublaslt_bf16_enabled_for` returns false).
All of Gemma-4's dense BF16 weights are force-VRAM-resident, so **the "reference"
kernels are NOT on the prefill hot path** — `matvec_bf16_reference_device` is
decode-only (`executor/mlp.rs`). This single fact removes the two biggest suspected
hand-rolled offenders.

## Per-stage table

FLOP arithmetic uses: hidden=2560, 30 layers (5 global hd=512 / 25 sliding hd=256),
T≈15000 prefill tokens, routed-expert intermediate≈2816, top_k=8, 128 experts.
`flops = 2·M·N·K`. BF16 peak 175 TFLOPS; NVFP4 peak ≈350 TFLOPS.

### 1. QKV projection — 498,000 us

- **(a) Kernel:** `matmul_bf16_cublaslt_device` (`cuda/runtime/cublaslt.rs`), with the
  shared-input variant `matmul_bf16_cublaslt_with_input_bf16_device` (single
  f32→bf16 quant reused across Q/K/V). The loader's fused `qkv_proj` CUTLASS group is
  **NVFP4-only** and is skipped for Gemma-4 (BF16 weights). Three separate cuBLASLt
  GEMMs per layer.
- **(b) Classification:** **library-grade (cuBLASLt)**. Not hand-rolled.
- **(c) Util:** total qkv FLOP ≈ 19.7 TFLOP (mixed-layer) → aggregate **≈38-47 TFLOPS**
  (`19.7e12 / 0.498s`). The engine's reported 56.5 is a per-layer `max()` (best global
  layer with q_out=8192). Either way **~22-32% of BF16 peak.**
- **(d) Library replacement:** already library-grade. The low number is *not* kernel
  quality — it is **launch/conversion overhead on small per-GEMM M**. Each chunk's M
  is ~1875 tokens split into 3 thin GEMMs plus an f32→bf16 and bf16→f32 conversion
  kernel each. cuBLASLt at M≈1875, K=2560, N=1024-8192 is launch-bound, not
  flop-bound.
- **(e) Recoverable:** moderate. A fused QKV GEMM (one cuBLASLt call, N=10240) would
  cut 2 of 3 launch overheads and the redundant epilogue conversions. Estimated
  **~80-130k us saved** (~2-3% of layer_total). Requires materializing a fused BF16
  QKV weight at load (the NVFP4 path already does this; BF16 does not).

### 2. o_proj — 187,000 us

- **(a) Kernel:** `matmul_bf16_cublaslt_device` (BF16 branch of the o_proj `match`).
- **(b) Classification:** **library-grade (cuBLASLt).**
- **(c) Util:** FLOP ≈ 11.0 TFLOP → **≈59-100 TFLOPS** aggregate (single GEMM, larger
  effective M than the split QKV → better util). **~34-57% of peak.**
- **(d/e) Replacement:** none warranted — already a single library GEMM, decent util.
  **Not actionable.**

### 3. Attention — 1,739,000 us (39% of layer_total) — THE dominant stage

- **(a) Kernel:** `aegis_attention_prefill_dense_fa2_hdim512` (global hd=512 layers,
  `attention_prefill_fa2.cu`) and `aegis_attention_prefill_dense_halfq_wmma_impl`
  hd=256 (sliding layers). Dispatched in `attention/prefill_dense.rs` under
  `AEGIS_ATTN_FA2=1`.
- **(b) Classification:** **hand-rolled** (BF16 WMMA 16×16×16, FA-2 tiling, online
  softmax, cp.async double-buffered KV slabs).
- **(c) Util:** ~14% tensor util (~20-25 TFLOPS), per the prior FA-2 audit
  (`aegisllm_fa2_prefill_attn.md`), confirmed by the q_block 32→64 diagnostic
  (memory-bound would have shown ~50% change; saw ~10% → latency/sync-bound).
- **(d) Library replacement: GENUINELY ABSENT on SM120.** CUTLASS FMHA
  (`77_blackwell_fmha`) and FA-3/FA-4 are **SM100-only** (need TMA + 228 KiB shared).
  flash-attention SM80 caps at head_dim=256 — cannot serve the hd=512 global layers.
  SM120 has only 100 KiB shared; head_dim=512 BF16 K+V double-buffered + S/P scratch
  does not fit, which is the structural ceiling. **This is the one true no-library
  exception** (see dedicated section below).
- **(e) Recoverable:** the cheap wins (FA-2 rewrite, q64, barrier reduction) are
  already banked. Remaining headroom needs FP8 K/V (accuracy cost, user-rejected) or
  >100 KiB shared hardware. **Not a library swap. Not actionable here.**

### 4. Routed experts (`experts_done`) — 1,520,000 us (34%)

- **(a) Kernels (split dispatch, `prefill/moe.rs`
  `forward_moe_cutlass_split_routed_experts`):**
  - Large experts (M ≥ 128): **CUTLASS NVFP4 grouped GEMM** via
    `cutlass_moe_nvfp4_grouped_run` (`ThreadBlockShape<128,128,128>`).
  - Small experts (M < 128): **hand-rolled** `aegis_nvfp4_grouped_prequant_gemm_wmma_bf16_t32_big`
    (`linear_nvfp4_packed.cu` — 64×64 tile, 8-warp, BF16 WMMA inner).
- **(b) Classification:** **mixed** — large bucket library-grade (CUTLASS), small
  bucket hand-rolled (t32_big).
- **(c) Util:** routed-expert FLOP ≈ 156 TFLOP → **≈102 TFLOPS aggregate** (~29% of
  NVFP4 peak, ~58% of BF16 peak). Per the M-distribution memo, the 10-15 M≥256
  experts do >95% of compute and go through CUTLASS; the ~10-20 M<64 experts do <5%
  of compute but each costs a launch + permute/quant overhead. The 102 TFLOPS
  *aggregate* is therefore CUTLASS-dominated already; the small-expert t32_big tail
  is a fixed-overhead drag, not a flop underperformer.
- **(d) Library replacement:** the small bucket cannot go to the same CUTLASS template
  (CUTLASS rejects M<128). A second `<64,128,128>` (or smaller) CUTLASS template
  *could* absorb the 64-127 band, but the long tail is M<64 (often M=1) — CUTLASS
  has no useful tile there. The t32_big subset path is already the right structure
  (one grouped GEMM per projection for the whole small subset).
- **(e) Recoverable:** small. The small bucket is <5% of MoE compute; even halving its
  overhead is sub-1% of layer_total. The CUTLASS large path is already the win
  (`aegisllm_cutlass_nvfp4_grouped_landed.md`: +70-82% prefill). **Mostly not
  actionable**; a `<64,128,128>` template is a marginal tuning lever, not a cheap win.

### 5. Shared MLP (`shared_mlp_done`) — 223,000 us

- **(a) Kernel:** `matmul_bf16_cublaslt_device` ×3 (gate, up, down), or one fused
  gate+up cuBLASLt GEMM + `geglu_tanh_strided` when `shared.gate_up_fused` exists
  (`prefill/moe.rs` step 2).
- **(b) Classification:** **library-grade (cuBLASLt).** The shared MLP weights are
  force-VRAM-resident, so the cuBLASLt branch always applies; the
  `matmul_bf16_reference_batched_device` fallback is dead code here. The name
  "reference" appears only in the decode helper `matvec_bf16_reference_device`, which
  this stage does not call.
- **(c) Util:** gate+up+down FLOP ≈ 14-18 TFLOP (shared intermediate uncertain;
  2048→2560 range) → **≈63-80 TFLOPS** (~36-46% of BF16 peak).
- **(d/e) Replacement:** already library-grade and fused where the loader supplies a
  fused weight. **Not actionable** beyond ensuring `gate_up_fused` is materialized
  (it apparently is — the code path exists).

### 6. Router (`router_done`) — 71,000 us

- **(a) Kernel:** `matmul_bf16_cublaslt_device` (`prefill/moe.rs` step 6 — the code
  explicitly checks `cublaslt_bf16_enabled_for(&moe.router)` first; the
  `matmul_bf16_reference_batched_device` is the host-resident fallback only).
- **(b) Classification:** **library-grade (cuBLASLt).** NOTE: prefill uses cuBLASLt;
  *decode* uses `matvec_bf16_reference_device` — but that is out of this audit's scope.
- **(c) Util:** router GEMM is [128 experts × 2560 hidden], FLOP ≈ 0.30 TFLOP →
  **≈4 TFLOPS.** Looks abysmal, but N=128 is a tiny GEMM — this stage is **100%
  launch/overhead-bound**, not compute-bound. 71k us / (30 layers × 8 chunks) ≈ 296 us
  per call for an f32→bf16 + GEMM + bf16→f32 + the rms-norm/scale preceding it.
- **(d) Library replacement:** cuBLASLt is already the library. The router can never
  be flop-efficient at N=128 — it is fundamentally a thin GEMM.
- **(e) Recoverable:** 71k us is 1.6% of layer_total and is overhead, not kernel
  quality. Could be trimmed by fusing the f32→bf16/bf16→f32 conversions into the
  rms-norm/epilogue, but it is small. **Low-priority micro-opt, not a library swap.**

### 7. Small buckets — rope / kv_store / embed / topk / MoE-misc

| stage | us | kernel | class | note |
|-------|-----|--------|-------|------|
| rope | 44,000 | `apply_rope_positions_batched_f16_out` (`norm_rope_kv.cu`) | hand-rolled | small, fine |
| kv_store | 26,000 | `store_kv_slots_batched_rope_key` (+FP8 mirror) | hand-rolled | small, fine |
| embed | 30,000 | embedding gather | hand-rolled | small, fine |
| topk | 15,000 | `router_softmax_topk` + `router_bucket_sort` (`router_topk.cu`) | hand-rolled | small, fine |
| MoE misc | ~113,000 | `geglu_tanh`, `permute_gather_f32`, `unpermute_scatter_add_f32`, `zero_f32`, scatter (`router_topk.cu`, `norm_rope_kv.cu`) | hand-rolled | **surprisingly large** |

**MoE-misc ~113k us (2.5% of layer_total)** is the only small-bucket surprise. It is
memory-bound data-movement (gather/scatter/permute of [total_routed × 2560] f32
buffers) + GeGLU. No library exists for permute/scatter — these are inherently custom.
The `zero_f32` calls on `permuted_intermediate/swiglu/output` in the CUTLASS split path
(needed to 0-fill small-expert rows) are pure overhead and a candidate for a masked
GEMM epilogue. Minor.

## Ranked actionable swaps (biggest prefill gain first)

There are **NO mediocre-hand-rolled-with-a-library-alternative** stages. Every dense
GEMM is already on cuBLASLt; the large MoE bucket is already on CUTLASS. The only
real levers are *fusion/overhead* reductions, not library swaps:

1. **Fused BF16 QKV GEMM** — replace 3 cuBLASLt calls (+2 redundant epilogue
   conversions) with one N=10240 cuBLASLt call. Reuses the already-validated cuBLASLt
   path → **zero accuracy risk**. Requires a fused BF16 QKV weight materialized at
   load. Est. **~80-130k us / 4.44M = ~2-3% prefill.** *Best available win.*
2. **Fuse f32↔bf16 conversions into rms-norm / epilogues** for QKV, o_proj, router,
   shared MLP — removes ~2 conversion-kernel launches per GEMM. Reuses validated math
   (bit-identical). Est. **~1-2% prefill** aggregate. Low risk, broad but small.
3. **Eliminate the `zero_f32` pre-fills in the CUTLASS split MoE path** via masked
   scatter. Est. **<1% prefill.** Marginal.
4. *(Tuning, not a swap)* second CUTLASS `<64,128,128>` template for the MoE 64-127 M
   band. Est. **<1% prefill** (band is <5% of MoE compute). Not worth the compile cost.

## Genuine no-library exceptions

- **Attention head_dim=512 (global layers)** — `aegis_attention_prefill_dense_fa2_hdim512`.
  CUTLASS FMHA / FA-3 / FA-4 are SM100-only; flash-attention SM80 caps at hd=256.
  SM120's 100 KiB shared cap structurally prevents head_dim=512 K+V double-buffering.
  The ~14% util is the BF16 ceiling, not a defect. **No library swap exists. Not
  recoverable without FP8 K/V (accuracy cost) or new hardware.** This stage is 39% of
  prefill and is correctly already the team's focus — but it is *not* a hand-rolling
  failure, it is a hardware limit.
- **rope / kv_store / embed / permute / scatter / topk** — no library primitives
  exist for these layouts; hand-rolled is correct and they are small (<6% combined).

## Verdict on the user's hypothesis

**"Is the engine fundamentally limited by hand-rolling kernels?" — No.**

Of the 7 major prefill stages, **5 are already library-grade**: QKV, o_proj, shared
MLP, and router all run on **cuBLASLt**; the large-expert MoE bucket (>95% of MoE
compute) runs on **CUTLASS NVFP4 grouped GEMM**. The only genuinely hand-rolled
*compute-heavy* stages are **(a) attention** — for which **no SM120 library exists**
(a hardware fact, not an engineering shortfall) — and **(b) the small-expert MoE
t32_big tail**, which carries <5% of MoE FLOPs.

The measured "low" TFLOPS numbers (qkv ~40, router ~4) are **launch- and
conversion-overhead artifacts on thin GEMMs**, not kernel-quality failures — cuBLASLt
*is* the best available kernel; it simply cannot be flop-efficient at M≈1875 / N=128.
The total prefill recoverable by realistic *library-adjacent* work (QKV fusion +
conversion fusion) is **~3-5% of layer_total** — real but modest. The dominant cost,
attention (39%), is a genuine SM120 hardware ceiling with no library escape.

**Honest bottom line:** the engine is **not** hand-rolling its way into mediocrity.
The dense and large-MoE paths are already on vendor libraries. The remaining gains are
fusion micro-optimizations (~3-5%), and the big structural cost (attention) is
hardware-bound, not library-bound. There are ~2 cheap fusion wins; everything else is
either already fine or genuinely hard.

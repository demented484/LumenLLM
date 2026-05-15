// Router softmax + top-k + bucket-sort kernels for MoE dispatch.
//
// Replaces the old host-side roundtrip (download logits → softmax+top-k+bucket
// on CPU → upload indices+weights). Eliminates one device sync per MoE layer
// in the inference path; data-flow stays entirely on the device until the
// host needs `expert_counts` to size per-expert dispatch dimensions.
//
// Three kernels:
//   * `aegis_router_softmax_topk`         — per-token softmax + top-k selection
//                                            with optional per-expert scale
//                                            and renormalisation
//   * `aegis_router_zero_expert_counts`   — clear the atomic bucket counters
//   * `aegis_router_bucket_sort`          — scatter (token, expert, weight)
//                                            triples into per-expert lists via
//                                            `atomicAdd` on the count array

// Per-token softmax + top-k selection. One thread handles one token; loops
// over `num_experts` (up to ~128 for current models) keep all logits in
// thread-local arrays. Selection sort over `top_k` entries is fine for small
// `top_k` (typically 2-8).
//
// Numerical stability: subtract the max before `expf`. After top-k selection,
// optionally multiply weights by `per_expert_scale[expert_idx]` and
// re-normalise so `sum(top_k_weights) == 1.0` (matches the host
// `softmax_top_k_normalized` behaviour byte-for-byte under ideal arithmetic).
extern "C" __global__ void aegis_router_softmax_topk(
    const float* __restrict__ logits,         // [batch, num_experts]
    const float* __restrict__ per_expert_scale, // [num_experts] or null
    const unsigned int batch,
    const unsigned int num_experts,
    const unsigned int top_k,
    unsigned int*  __restrict__ out_idx,      // [batch, top_k]
    float*         __restrict__ out_weights   // [batch, top_k]
) {
    const unsigned int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= batch) return;

    // 128 is the max num_experts we expect; static array keeps everything
    // in registers/local memory without dynamic shared-memory plumbing.
    constexpr unsigned int MAX_EXPERTS = 256;
    if (num_experts > MAX_EXPERTS) {
        // Out-of-bounds; should never happen given current model configs.
        return;
    }

    float local_logits[MAX_EXPERTS];
    const float* row = logits + (size_t)token * num_experts;

    // Pass 1: load + max
    float max_val = -3.402823466e38f;
    for (unsigned int i = 0; i < num_experts; ++i) {
        const float v = row[i];
        local_logits[i] = v;
        if (v > max_val) max_val = v;
    }

    // Pass 2: exp + sum (softmax denominator)
    float sum = 0.0f;
    for (unsigned int i = 0; i < num_experts; ++i) {
        const float e = __expf(local_logits[i] - max_val);
        local_logits[i] = e;
        sum += e;
    }
    const float inv_sum = 1.0f / sum;

    // Pass 3: top-k selection by selection sort. For top_k ≤ 8 this is fine.
    constexpr unsigned int MAX_TOP_K = 16;
    unsigned int picked_idx[MAX_TOP_K];
    float        picked_w[MAX_TOP_K];
    if (top_k > MAX_TOP_K) return;

    for (unsigned int k = 0; k < top_k; ++k) {
        float best_w = -1.0f;
        unsigned int best_i = 0;
        for (unsigned int i = 0; i < num_experts; ++i) {
            const float w = local_logits[i];
            if (w > best_w) {
                best_w = w;
                best_i = i;
            }
        }
        picked_idx[k] = best_i;
        picked_w[k]   = best_w * inv_sum;
        // Mark as picked so it isn't re-selected.
        local_logits[best_i] = -1.0f;
    }

    // Pass 4: renormalise top-k probabilities so they sum to 1.
    // ORDER MATTERS: HF Gemma4TextRouter.forward renormalises THEN multiplies
    // by per_expert_scale — and does NOT renormalise again. Doing the scale
    // before the renormalise (the previous order here) silently undoes the
    // per-expert weighting because dividing by `sum(scaled)` gives back the
    // unscaled proportions. Mirrors the CPU `softmax_top_k_normalized` path
    // in `executor/mlp.rs`.
    float renorm_sum = 0.0f;
    for (unsigned int k = 0; k < top_k; ++k) renorm_sum += picked_w[k];
    const float inv_renorm = (renorm_sum > 0.0f) ? (1.0f / renorm_sum) : 0.0f;
    for (unsigned int k = 0; k < top_k; ++k) {
        picked_w[k] *= inv_renorm;
    }

    // Pass 5: optional per-expert scale (no further renormalise).
    if (per_expert_scale != nullptr) {
        for (unsigned int k = 0; k < top_k; ++k) {
            picked_w[k] *= per_expert_scale[picked_idx[k]];
        }
    }

    unsigned int* out_i = out_idx     + (size_t)token * top_k;
    float*        out_w = out_weights + (size_t)token * top_k;
    for (unsigned int k = 0; k < top_k; ++k) {
        out_i[k] = picked_idx[k];
        out_w[k] = picked_w[k];
    }
}

// Decode-only variant: per-token softmax + top-k that writes a SINGLE packed
// output buffer with `(u32 idx, f32 weight)` records interleaved as raw u32
// words. Used by `forward_moe_decode_device` so the host can issue ONE small
// dtoh (top_k * 8 bytes) instead of two separate downloads. Otherwise byte-
// identical to `aegis_router_softmax_topk`. Layout per token:
//   packed[k*2 + 0] = expert_idx           (u32)
//   packed[k*2 + 1] = bitcast<u32>(weight) (f32 bits as u32)
extern "C" __global__ void aegis_router_softmax_topk_packed(
    const float* __restrict__ logits,           // [batch, num_experts]
    const float* __restrict__ per_expert_scale, // [num_experts] or null
    const unsigned int batch,
    const unsigned int num_experts,
    const unsigned int top_k,
    unsigned int* __restrict__ out_packed       // [batch * top_k * 2] u32 words
) {
    const unsigned int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= batch) return;

    constexpr unsigned int MAX_EXPERTS = 256;
    if (num_experts > MAX_EXPERTS) return;

    float local_logits[MAX_EXPERTS];
    const float* row = logits + (size_t)token * num_experts;

    float max_val = -3.402823466e38f;
    for (unsigned int i = 0; i < num_experts; ++i) {
        const float v = row[i];
        local_logits[i] = v;
        if (v > max_val) max_val = v;
    }

    float sum = 0.0f;
    for (unsigned int i = 0; i < num_experts; ++i) {
        const float e = __expf(local_logits[i] - max_val);
        local_logits[i] = e;
        sum += e;
    }
    const float inv_sum = 1.0f / sum;

    constexpr unsigned int MAX_TOP_K = 16;
    unsigned int picked_idx[MAX_TOP_K];
    float        picked_w[MAX_TOP_K];
    if (top_k > MAX_TOP_K) return;

    for (unsigned int k = 0; k < top_k; ++k) {
        float best_w = -1.0f;
        unsigned int best_i = 0;
        for (unsigned int i = 0; i < num_experts; ++i) {
            const float w = local_logits[i];
            if (w > best_w) {
                best_w = w;
                best_i = i;
            }
        }
        picked_idx[k] = best_i;
        picked_w[k]   = best_w * inv_sum;
        local_logits[best_i] = -1.0f;
    }

    float renorm_sum = 0.0f;
    for (unsigned int k = 0; k < top_k; ++k) renorm_sum += picked_w[k];
    const float inv_renorm = (renorm_sum > 0.0f) ? (1.0f / renorm_sum) : 0.0f;
    for (unsigned int k = 0; k < top_k; ++k) picked_w[k] *= inv_renorm;

    if (per_expert_scale != nullptr) {
        for (unsigned int k = 0; k < top_k; ++k) {
            picked_w[k] *= per_expert_scale[picked_idx[k]];
        }
    }

    unsigned int* out = out_packed + (size_t)token * top_k * 2u;
    for (unsigned int k = 0; k < top_k; ++k) {
        out[k * 2u + 0u] = picked_idx[k];
        // Bit-cast f32 → u32 so the host can reinterpret without a copy.
        out[k * 2u + 1u] = __float_as_uint(picked_w[k]);
    }
}

// Zero `expert_counts[num_experts]`. Called before `bucket_sort` because
// `bucket_sort` `atomicAdd`s into the same array.
extern "C" __global__ void aegis_router_zero_expert_counts(
    unsigned int* __restrict__ expert_counts,
    const unsigned int num_experts
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_experts) {
        expert_counts[idx] = 0u;
    }
}

// Compute prefix sum (CSR-style offsets) over `expert_counts[num_experts]`.
// Output: `expert_offsets[num_experts + 1]` where `expert_offsets[e]` is the
// starting row in the permuted activation buffer for expert `e`, and
// `expert_offsets[num_experts]` is the total number of routed assignments.
//
// Single-block kernel; serial scan. `num_experts` is small (≤256) so this is
// fine — sub-microsecond on any modern GPU. Replaces the host-side
// "build per-expert offsets" loop that grouped-GEMM dispatch would otherwise
// have to do after a `download_u32(expert_counts)`.
extern "C" __global__ void aegis_router_expert_offsets(
    const unsigned int* __restrict__ expert_counts,  // [num_experts]
    const unsigned int num_experts,
    unsigned int* __restrict__ expert_offsets         // [num_experts + 1]
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    unsigned int sum = 0;
    expert_offsets[0] = 0;
    for (unsigned int e = 0; e < num_experts; ++e) {
        sum += expert_counts[e];
        expert_offsets[e + 1] = sum;
    }
}

// Scatter (token, expert, weight) triples into per-expert lists. Each thread
// handles one (token, k) slot from the top-k tables. `atomicAdd` on the
// expert's count slot gives the position to write to.
//
// Layout of outputs:
//   * `expert_token_lists[expert_idx * stride + pos]` = source token index
//   * `expert_weight_lists[expert_idx * stride + pos]` = routing weight
//   * `expert_counts[expert_idx]` = number of tokens routed to that expert
//
// `stride` is `max_tokens_per_expert`, set on the host to `batch * top_k`
// (the worst case where every (token, k) slot picks the same expert).
// Permute-gather: scatter input rows into the expert-sorted (permuted)
// layout. After this kernel, `permuted[expert_offsets[e]..expert_offsets[e+1]]`
// holds the hidden states of all tokens routed to expert `e`, in order.
//
// Replaces the per-expert `gather_rows_f32` calls in the existing dispatch
// loop with one kernel that does all experts at once. Output layout is
// what the grouped-NVFP4-GEMM kernel below consumes.
extern "C" __global__ void aegis_permute_gather_f32(
    const float* __restrict__ src,                          // [batch, hidden]
    const unsigned int* __restrict__ expert_token_lists,    // [num_experts × stride]
    const unsigned int* __restrict__ expert_counts,         // [num_experts]
    const unsigned int* __restrict__ expert_first_token_off,// [num_experts + 1]
    const unsigned int stride,
    const unsigned int hidden,
    float* __restrict__ permuted                            // [total_assignments, hidden]
) {
    const unsigned int expert = blockIdx.z;
    const unsigned int batch_in_expert = blockIdx.y;
    const unsigned int hidden_base = blockIdx.x * blockDim.x;
    const unsigned int tid = threadIdx.x;

    const unsigned int count = expert_counts[expert];
    if (batch_in_expert >= count) return;
    const unsigned int h = hidden_base + tid;
    if (h >= hidden) return;

    const unsigned int src_token = expert_token_lists[expert * stride + batch_in_expert];
    const unsigned int dst_row = expert_first_token_off[expert] + batch_in_expert;

    permuted[(size_t)dst_row * hidden + h] = src[(size_t)src_token * hidden + h];
}

// Unpermute-scatter-add: reads the per-expert output rows from the permuted
// buffer, multiplies each by its routing weight, and atomically adds into
// `moe_acc[src_token, h]`. Multiple experts may write to the same source
// token (top_k > 1), which is why scatter is atomic.
//
// DETERMINISM NOTE: this kernel's `atomicAdd` makes the per-token sum
// ORDER-DEPENDENT across runs — blocks from different experts (`blockIdx.z`)
// race on the same `moe_acc[src_token, h]` cell, and atomic-add ordering of
// floats is not reproducible. The resulting ~1-ULP per-run drift propagates
// through every prefill layer and flips occasional late-token argmax
// decisions in greedy decode. It is kept only for reference / A-B testing;
// the inference path uses the deterministic two-kernel pair below
// (`aegis_router_build_unpermute_index` + `aegis_unpermute_scatter_serial_f32`).
extern "C" __global__ void aegis_unpermute_scatter_add_f32(
    const float* __restrict__ permuted,                     // [total_assignments, hidden]
    const unsigned int* __restrict__ expert_token_lists,    // [num_experts × stride]
    const float*        __restrict__ expert_weight_lists,   // [num_experts × stride]
    const unsigned int* __restrict__ expert_counts,
    const unsigned int* __restrict__ expert_first_token_off,
    const unsigned int stride,
    const unsigned int hidden,
    float* __restrict__ moe_acc                             // [batch, hidden]
) {
    const unsigned int expert = blockIdx.z;
    const unsigned int batch_in_expert = blockIdx.y;
    const unsigned int hidden_base = blockIdx.x * blockDim.x;
    const unsigned int tid = threadIdx.x;

    const unsigned int count = expert_counts[expert];
    if (batch_in_expert >= count) return;
    const unsigned int h = hidden_base + tid;
    if (h >= hidden) return;

    const unsigned int src_token = expert_token_lists[expert * stride + batch_in_expert];
    const float weight = expert_weight_lists[expert * stride + batch_in_expert];
    const unsigned int src_row = expert_first_token_off[expert] + batch_in_expert;

    const float v = permuted[(size_t)src_row * hidden + h] * weight;
    atomicAdd(&moe_acc[(size_t)src_token * hidden + h], v);
}

// ── Deterministic unpermute-scatter, kernel 1 of 2 ──────────────────────────
//
// Builds a per-token inverse routing table so the scatter (kernel 2) can be a
// race-free, fixed-order serial accumulation.
//
// One thread per routed assignment `(expert e, slot b)` with `b < count[e]`.
// For each assignment it computes the source token `t` and the assignment's
// *canonical rank* among `t`'s experts — the number of experts `e' < e` that
// also routed `t`. That rank `k ∈ [0, top_k)` gives the assignment a
// deterministic slot in `t`'s row of the inverse table, INDEPENDENT of block
// scheduling. The table stores, per `(t, k)`, the packed pair
//   out_rows  [t*top_k + k] = permuted source row  (expert_first_token_off[e] + b)
//   out_wbits [t*top_k + k] = bitcast<u32>(routing weight)
// and `out_count[t]` is set to the number of experts routing `t` (== top_k in
// practice, but computed so a token with fewer routes is still correct).
//
// Rank computation scans experts `0..e`; total work is
// `num_experts * total_assignments` — a few million ops per layer, negligible
// next to the grouped GEMMs.
extern "C" __global__ void aegis_router_build_unpermute_index(
    const unsigned int* __restrict__ expert_token_lists,    // [num_experts × stride]
    const unsigned int* __restrict__ expert_counts,         // [num_experts]
    const unsigned int* __restrict__ expert_first_token_off,// [num_experts + 1]
    const float*        __restrict__ expert_weight_lists,   // [num_experts × stride]
    const unsigned int num_experts,
    const unsigned int stride,
    const unsigned int top_k,
    unsigned int* __restrict__ out_rows,                    // [batch × top_k]
    unsigned int* __restrict__ out_wbits,                   // [batch × top_k]
    unsigned int* __restrict__ out_count                    // [batch]
) {
    const unsigned int expert = blockIdx.y;
    if (expert >= num_experts) return;
    const unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int count = expert_counts[expert];
    if (slot >= count) return;

    const unsigned int token = expert_token_lists[expert * stride + slot];
    const float        weight = expert_weight_lists[expert * stride + slot];
    const unsigned int src_row = expert_first_token_off[expert] + slot;

    // Canonical rank of this assignment among `token`'s experts: count experts
    // with a smaller id that also routed `token`. Deterministic, scheduler-
    // independent. Each such expert routes `token` at most once.
    unsigned int rank = 0u;
    for (unsigned int e = 0u; e < expert; ++e) {
        const unsigned int ce = expert_counts[e];
        const unsigned int* list = expert_token_lists + (size_t)e * stride;
        for (unsigned int j = 0u; j < ce; ++j) {
            if (list[j] == token) { ++rank; break; }
        }
    }
    if (rank >= top_k) return;  // defensive: never expected (≤ top_k routes/token)

    out_rows [(size_t)token * top_k + rank] = src_row;
    out_wbits[(size_t)token * top_k + rank] = __float_as_uint(weight);
    // `out_count[token]` is the number of experts routing `token`. The highest
    // rank wins the max; every contributing assignment writes rank+1.
    atomicMax(&out_count[token], rank + 1u);
}

// ── Deterministic unpermute-scatter, kernel 2 of 2 ──────────────────────────
//
// One block per `(hidden tile, source token)`. Each thread owns one hidden
// channel `h` and accumulates that token's expert contributions in a FIXED
// rank order `k = 0..in_count[token]` into a register, then adds the result
// into `moe_acc[token, h]` exactly once. Each output cell is touched by
// exactly one block → no atomics, no cross-block contention → bit-identical
// across runs.
//
// The write is `+=` (not `=`) so the kernel composes when called more than
// once into the same — pre-zeroed — `moe_acc` (the CUTLASS split path issues
// one call for large experts and one for small experts). The read-modify-
// write is race-free because the `(h_tile, token)` → block mapping is a
// bijection over output cells within a single launch.
extern "C" __global__ void aegis_unpermute_scatter_serial_f32(
    const float* __restrict__ permuted,                     // [total_assignments, hidden]
    const unsigned int* __restrict__ in_rows,               // [batch × top_k]
    const unsigned int* __restrict__ in_wbits,              // [batch × top_k]
    const unsigned int* __restrict__ in_count,              // [batch]
    const unsigned int top_k,
    const unsigned int hidden,
    float* __restrict__ moe_acc                             // [batch, hidden]
) {
    const unsigned int token = blockIdx.y;
    const unsigned int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= hidden) return;

    const unsigned int n = in_count[token];
    if (n == 0u) return;
    float acc = 0.0f;
    for (unsigned int k = 0u; k < n; ++k) {
        const unsigned int src_row = in_rows[(size_t)token * top_k + k];
        const float weight = __uint_as_float(in_wbits[(size_t)token * top_k + k]);
        acc += permuted[(size_t)src_row * hidden + h] * weight;
    }
    moe_acc[(size_t)token * hidden + h] += acc;
}

extern "C" __global__ void aegis_router_bucket_sort(
    const unsigned int* __restrict__ topk_idx,     // [batch, top_k]
    const float*        __restrict__ topk_weights, // [batch, top_k]
    const unsigned int batch,
    const unsigned int top_k,
    const unsigned int stride,                     // max tokens per expert
    unsigned int* __restrict__ expert_token_lists, // [num_experts, stride]
    float*        __restrict__ expert_weight_lists,// [num_experts, stride]
    unsigned int* __restrict__ expert_counts       // [num_experts]
) {
    const unsigned int total = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total_slots = batch * top_k;
    if (total >= total_slots) return;

    const unsigned int token = total / top_k;
    const unsigned int slot  = total % top_k;
    const unsigned int expert = topk_idx[(size_t)token * top_k + slot];
    const float        weight = topk_weights[(size_t)token * top_k + slot];

    const unsigned int pos = atomicAdd(&expert_counts[expert], 1u);
    if (pos < stride) {
        expert_token_lists [(size_t)expert * stride + pos] = token;
        expert_weight_lists[(size_t)expert * stride + pos] = weight;
    }
}

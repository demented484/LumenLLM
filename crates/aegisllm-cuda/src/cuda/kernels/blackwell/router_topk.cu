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

    // Pass 4: optional per-expert scale + renormalise so weights sum to 1.
    if (per_expert_scale != nullptr) {
        for (unsigned int k = 0; k < top_k; ++k) {
            picked_w[k] *= per_expert_scale[picked_idx[k]];
        }
    }
    float renorm_sum = 0.0f;
    for (unsigned int k = 0; k < top_k; ++k) renorm_sum += picked_w[k];
    const float inv_renorm = (renorm_sum > 0.0f) ? (1.0f / renorm_sum) : 0.0f;

    unsigned int* out_i = out_idx     + (size_t)token * top_k;
    float*        out_w = out_weights + (size_t)token * top_k;
    for (unsigned int k = 0; k < top_k; ++k) {
        out_i[k] = picked_idx[k];
        out_w[k] = picked_w[k] * inv_renorm;
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

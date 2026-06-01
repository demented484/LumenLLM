extern "C" __global__ void aegis_bf16_matvec_reference(
    const unsigned short* matrix,
    const float* input,
    const unsigned int rows,
    const unsigned int cols,
    float* output
) {
    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }

    extern __shared__ float partial[];
    float sum = 0.0f;
    const unsigned short* matrix_row = matrix + size_t(row) * cols;
    for (unsigned int col = tid; col < cols; col += blockDim.x) {
        sum += bf16_to_float(matrix_row[col]) * input[col];
    }

    partial[tid] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0u) {
        output[row] = partial[0];
    }
}

// Fast BF16 M=1 matvec: same 128 threads/row + coalesced loads as
// aegis_bf16_matvec_reference, but the 8-step __syncthreads tree is replaced by a
// per-warp shuffle reduction + a single-barrier combine of the 4 warp partials.
// f32 accumulate (numerically equivalent; only the reduction order differs).
// grid=(rows,1,1) block=(128,1,1) shmem=0. Used for the dense/shared-MLP and
// lm_head matvecs (every model + dense E4B).
extern "C" __global__ void aegis_bf16_matvec_warp(
    const unsigned short* matrix,
    const float* input,
    const unsigned int rows,
    const unsigned int cols,
    float* output
) {
    const unsigned int row = blockIdx.x;
    const unsigned int tid = threadIdx.x;             // 0..127
    if (row >= rows) {
        return;
    }
    const unsigned short* matrix_row = matrix + size_t(row) * cols;
    float sum = 0.0f;
    if ((cols & 7u) == 0u) {
        // Vectorized: 128-bit loads — 8 bf16 weights (uint4) + 8 f32 inputs
        // (2x float4) per iteration. matrix_row is 16-byte aligned when cols%8==0
        // (base is 256-aligned, row*cols is a multiple of 8 u16 = 16 bytes).
        const unsigned int n8 = cols >> 3;
        const uint4* mrow4 = reinterpret_cast<const uint4*>(matrix_row);
        const float4* in4 = reinterpret_cast<const float4*>(input);
        for (unsigned int g = tid; g < n8; g += 128u) {
            const uint4 w = mrow4[g];
            const float4 a = in4[g * 2u];
            const float4 b = in4[g * 2u + 1u];
            sum += bf16_to_float((unsigned short)(w.x & 0xFFFFu)) * a.x;
            sum += bf16_to_float((unsigned short)(w.x >> 16))     * a.y;
            sum += bf16_to_float((unsigned short)(w.y & 0xFFFFu)) * a.z;
            sum += bf16_to_float((unsigned short)(w.y >> 16))     * a.w;
            sum += bf16_to_float((unsigned short)(w.z & 0xFFFFu)) * b.x;
            sum += bf16_to_float((unsigned short)(w.z >> 16))     * b.y;
            sum += bf16_to_float((unsigned short)(w.w & 0xFFFFu)) * b.z;
            sum += bf16_to_float((unsigned short)(w.w >> 16))     * b.w;
        }
    } else {
        for (unsigned int col = tid; col < cols; col += 128u) {
            sum += bf16_to_float(matrix_row[col]) * input[col];
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, (unsigned int)offset, 32);
    }
    __shared__ float warp_sums[4];
    const unsigned int warp_id = tid >> 5;
    if ((tid & 31u) == 0u) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();
    if (tid == 0u) {
        output[row] = warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
    }
}

extern "C" __global__ void aegis_argmax_f32_blocks(
    const float* input,
    const unsigned int len,
    float* block_values,
    unsigned int* block_indices
) {
    __shared__ float values[256];
    __shared__ unsigned int indices[256];
    const unsigned int tid = threadIdx.x;
    const unsigned int idx = blockIdx.x * blockDim.x + tid;
    float value = -3.402823466e38f;
    unsigned int out_idx = 0xffffffffu;
    if (idx < len) {
        value = input[idx];
        out_idx = idx;
    }
    values[tid] = value;
    indices[tid] = out_idx;
    __syncthreads();

    for (unsigned int stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            const float other_value = values[tid + stride];
            const unsigned int other_idx = indices[tid + stride];
            const bool take_other = other_value > values[tid]
                || (other_value == values[tid] && other_idx < indices[tid]);
            if (take_other) {
                values[tid] = other_value;
                indices[tid] = other_idx;
            }
        }
        __syncthreads();
    }

    if (tid == 0u) {
        block_values[blockIdx.x] = values[0];
        block_indices[blockIdx.x] = indices[0];
    }
}

extern "C" __global__ void aegis_argmax_f32_finalize(
    const float* block_values,
    const unsigned int* block_indices,
    const unsigned int num_blocks,
    unsigned int* output_token
) {
    float best_value = -3.402823466e38f;
    unsigned int best_idx = 0u;
    for (unsigned int idx = 0u; idx < num_blocks; ++idx) {
        const float value = block_values[idx];
        const unsigned int token = block_indices[idx];
        if (value > best_value || (value == best_value && token < best_idx)) {
            best_value = value;
            best_idx = token;
        }
    }
    output_token[0] = best_idx;
}

// Batched BF16 matmul: output[batch, row] = sum_c bf16(matrix[row, c]) * input[batch, c].
// Used by chunked prefill for BF16 weights (router, shared MLP, lm_head).
// Grid: (rows, batch). Block: (128, 1, 1). Shared mem: 128 * sizeof(float).
extern "C" __global__ void aegis_bf16_matmul_reference_batched(
    const unsigned short* matrix,
    const float* input,
    const unsigned int rows,
    const unsigned int cols,
    const unsigned int batch,
    float* output
) {
    const unsigned int row = blockIdx.x;
    const unsigned int batch_idx = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    if (row >= rows || batch_idx >= batch) {
        return;
    }
    extern __shared__ float partial[];
    float sum = 0.0f;
    const unsigned short* matrix_row = matrix + size_t(row) * cols;
    const float* input_row = input + size_t(batch_idx) * cols;
    for (unsigned int col = tid; col < cols; col += blockDim.x) {
        sum += bf16_to_float(matrix_row[col]) * input_row[col];
    }
    partial[tid] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0u) {
        output[size_t(batch_idx) * rows + row] = partial[0];
    }
}

// ── GPU multinomial sampler (temperature → top-k → top-p → min-p → draw) ─────
//
// Replaces the per-token "download all 248K vocab logits to the host + sort +
// top-k/top-p/min-p on the CPU" path with ONE fused single-block kernel that
// keeps the logits on device and downloads only the sampled token id.
//
// Design: a single block of `BLOCK` threads, launched once per decode token.
//   * Top-k by iterated block-parallel argmax: for each of the `k` rounds
//     (k = min(top_k, KCAP) — NOT a fixed 64; we only ever do as many rounds as
//     the config's top_k), the whole block cooperatively finds the best logit
//     strictly worse than the previous round's winner (lexicographic-descending
//     order: value desc, index asc). This naturally handles duplicate logits
//     (index breaks ties) without per-element bookkeeping and matches the CPU's
//     stable sort+truncate ranking exactly. Cost ≈ k full-vocab parallel
//     reductions; with BLOCK=512 and 248K vocab each round is ~485 reads/thread
//     plus a log2(BLOCK) shared-mem reduction.
//   * Thread 0 then applies temperature → top-p → min-p → renormalise → draw
//     over the ≤k winners (a tiny set) with the host-supplied uniform `u ∈
//     [0,1)` — the SAME `rand::random::<f32>()` draw the CPU path uses.
//
// SEMANTICS — byte-for-byte the CPU `sample_next_token` order:
//   * top-k is a pure logit ranking (pre-exp), exactly like the CPU sort+truncate.
//     Ties broken by SMALLER index (matches `total_cmp` desc + the stable sort:
//     equal logits keep ascending index, so the lower index ranks first).
//   * weight = expf((logit - max_logit) / temperature)  (max == the round-0
//     winner = the global max over the top-k set, same as the CPU).
//   * top-p: walk the desc weights, keep the shortest prefix whose cumulative
//     weight ≥ top_p * total; keep at least 1.
//   * min-p: keep weights ≥ min_p * weight_max (weight_max == weights[0]).
//   * draw over the post-min-p `total` (NOT a forced 1.0): `draw = u * total;
//     for each survivor: if draw <= w return idx; draw -= w`. This is the
//     identical event to the CPU's `u * total_raw <= w_raw` walk given the same
//     `u`, so the selected token is the same.
//
// SAMPLER_KCAP bounds the top-k we support. The config uses top_k ≤ 50.
#define SAMPLER_KCAP 64

// Block-parallel argmax-with-threshold for one round. Each thread scans its
// strided slice of [0,vocab) keeping the best element STRICTLY worse than
// (thr_v,thr_i) under lexicographic-descending order, then the block reduces to
// a single winner via shared memory. `have_thr=false` (round 0) accepts all.
// Returns the winner in (out_v,out_i) for every thread (broadcast via smem[0]).
__device__ __forceinline__ void aegis_sampler_round_argmax(
    const float* __restrict__ logits,
    unsigned int vocab,
    bool have_thr,
    float thr_v,
    unsigned int thr_i,
    float* red_val,            // [blockDim.x] shared scratch
    unsigned int* red_idx,     // [blockDim.x] shared scratch
    float* out_v,
    unsigned int* out_i)
{
    const unsigned int tid = threadIdx.x;
    const unsigned int nthreads = blockDim.x;
    float best = -3.402823466e38f;
    unsigned int besti = 0xffffffffu;
    for (unsigned int i = tid; i < vocab; i += nthreads) {
        const float v = logits[i];
        if (have_thr) {
            const bool worse_than_thr = v < thr_v || (v == thr_v && i > thr_i);
            if (!worse_than_thr) continue;
        }
        if (v > best || (v == best && i < besti)) { best = v; besti = i; }
    }
    red_val[tid] = best;
    red_idx[tid] = besti;
    __syncthreads();
    for (unsigned int stride = nthreads >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            const float ov = red_val[tid + stride];
            const unsigned int oi = red_idx[tid + stride];
            const bool take = ov > red_val[tid] || (ov == red_val[tid] && oi < red_idx[tid]);
            if (take) { red_val[tid] = ov; red_idx[tid] = oi; }
        }
        __syncthreads();
    }
    *out_v = red_val[0];
    *out_i = red_idx[0];
    __syncthreads();   // protect red_* before the next round overwrites it
}

// Single fused sampler kernel — grid (1), block (BLOCK threads). Selects the
// global top-`k` (k = min(top_k, KCAP)) by iterated parallel argmax, then thread
// 0 does temperature/top-p/min-p/draw over the ≤k winners with uniform `u`.
extern "C" __global__ void aegis_sampler_topk_fused(
    const float* __restrict__ logits,
    const unsigned int vocab,
    const unsigned int top_k,        // requested k; 0 or >KCAP => KCAP
    const float temperature,
    const float top_p,
    const float min_p,
    const float u,                   // uniform draw in [0,1)
    unsigned int* __restrict__ out_token
) {
    extern __shared__ float aegis_sampler_smem[];
    float* red_val = aegis_sampler_smem;                       // blockDim
    unsigned int* red_idx = (unsigned int*)(red_val + blockDim.x);
    // Winners live in shared mem so thread 0 can read them after the rounds.
    float* sel_v = (float*)(red_idx + blockDim.x);             // KCAP
    unsigned int* sel_i = (unsigned int*)(sel_v + SAMPLER_KCAP);// KCAP

    unsigned int k = top_k;
    if (k == 0u || k > SAMPLER_KCAP) k = SAMPLER_KCAP;
    if (k > vocab) k = vocab;

    float thr_v = 3.402823466e38f;
    unsigned int thr_i = 0u;
    bool have_thr = false;
    for (unsigned int r = 0; r < k; ++r) {
        float win_v;
        unsigned int win_i;
        aegis_sampler_round_argmax(logits, vocab, have_thr, thr_v, thr_i,
                                   red_val, red_idx, &win_v, &win_i);
        if (threadIdx.x == 0) {
            sel_v[r] = win_v;
            sel_i[r] = win_i;
        }
        thr_v = win_v;
        thr_i = win_i;
        have_thr = true;
    }
    __syncthreads();

    // Serial finish over the tiny ≤k winner set on thread 0.
    if (threadIdx.x != 0) return;
    if (k == 0u) { out_token[0] = 0u; return; }

    const float temp = temperature > 1e-6f ? temperature : 1e-6f;
    const float max_logit = sel_v[0];   // round-0 winner == global max over top-k
    float w[SAMPLER_KCAP];
    unsigned int idx[SAMPLER_KCAP];
    unsigned int nw = 0;
    for (unsigned int r = 0; r < k; ++r) {
        // Precise libm expf (NOT fast __expf) so the post-temperature weights
        // match the CPU reference's `f32::exp()` — the top-p/min-p cutoffs and
        // the draw walk compare these weights, so a fast-math approximation
        // would diverge from the CPU sampler at boundary candidates.
        const float e = expf((sel_v[r] - max_logit) / temp);
        if (isfinite(e) && e > 0.0f) {
            w[nw] = e;
            idx[nw] = sel_i[r];
            ++nw;
        }
    }
    if (nw == 0u) { out_token[0] = sel_i[0]; return; }

    // top-p (nucleus): keep shortest desc prefix with cumulative >= top_p*total.
    float total = 0.0f;
    for (unsigned int r = 0; r < nw; ++r) total += w[r];
    if (top_p > 0.0f && top_p < 1.0f && total > 0.0f) {
        const float cutoff = total * top_p;
        float cum = 0.0f;
        unsigned int keep = 0;
        for (unsigned int r = 0; r < nw; ++r) {
            cum += w[r];
            ++keep;
            if (cum >= cutoff) break;
        }
        if (keep < 1u) keep = 1u;
        nw = keep;
    }

    // min-p: keep w >= min_p * w_max  (w_max == w[0], the global max weight).
    if (min_p > 0.0f) {
        const float thresh = w[0] * min_p;
        unsigned int m = 0;
        for (unsigned int r = 0; r < nw; ++r) {
            if (w[r] >= thresh) { w[m] = w[r]; idx[m] = idx[r]; ++m; }
        }
        nw = m;
        if (nw == 0u) { out_token[0] = sel_i[0]; return; }
    }

    // Multinomial draw — identical to the CPU cumulative walk over `total`.
    float total2 = 0.0f;
    for (unsigned int r = 0; r < nw; ++r) total2 += w[r];
    if (total2 <= 0.0f) { out_token[0] = sel_i[0]; return; }
    float draw = u * total2;
    for (unsigned int r = 0; r < nw; ++r) {
        if (draw <= w[r]) { out_token[0] = idx[r]; return; }
        draw -= w[r];
    }
    // FP-rounding fallthrough — the CPU returns argmax(logits) == sel_i[0].
    out_token[0] = sel_i[0];
}

// ── Fast top-k sampler (radix-threshold select) ─────────────────────────────
//
// Selects the IDENTICAL top-k set as `aegis_sampler_topk_fused` (the iterated
// block-argmax above) — same lexicographic-descending order (value desc, index
// asc) — but in ~4-6 full-vocab passes instead of `k` serial argmax rounds
// (k=50 in production). Then runs the SAME thread-0 temperature/top-p/min-p/draw
// finish over the ≤k winners, so a given uniform `u` selects the SAME token.
//
// Method (single block, BLOCK threads):
//   * Order-preserving f32→u32 key: for finite x, `vkey = (bits>>31) ? ~bits :
//     (bits | 0x80000000)`. Larger float ⇒ larger vkey (handles ±0). The top-k
//     by value == top-k by vkey.
//   * Phase 1 (find tau_vkey): byte-radix select. For each of the 4 bytes
//     (MSB→LSB), build a 256-bin histogram over candidates whose higher bytes
//     already match the running prefix, scan the bins from the top to find which
//     bin contains the k-th-largest vkey (by multiplicity), and fix that byte of
//     `tau_vkey`. After 4 passes `tau_vkey` is the EXACT 32-bit key of the
//     k-th-largest value (with multiplicity). `n_gt = #(vkey > tau_vkey) < k`.
//   * Phase 2 (resolve the value-tie group by index): among the `n_eq` elements
//     with `vkey == tau_vkey` we must keep the `k - n_gt` SMALLEST indices
//     (index asc tie-break). Byte-radix select the `(k-n_gt)`-th smallest index
//     within the tie group → `tau_idx`. Then the selected set is exactly
//     `{vkey > tau_vkey} ∪ {vkey == tau_vkey AND index <= tau_idx}` — exactly k
//     elements, identical to the iterated-argmax set.
//   * Gather the ≤k winners into shared mem, insertion-sort them by (value desc,
//     index asc) on thread 0 (k≤64), then run the identical finish.
//
// Only finite logits are considered (matches the CPU `is_finite` filter): a
// non-finite logit maps to a key but we mask it out in every histogram/count so
// it can never be selected (the iterated-argmax path likewise can pick a +inf,
// but the CPU reference filters non-finite BEFORE the sort — the production
// logits are always finite, and the same-u test never feeds non-finite values).

__device__ __forceinline__ unsigned int aegis_f32_to_orderkey(float x) {
    unsigned int b = __float_as_uint(x);
    return (b >> 31) ? ~b : (b | 0x80000000u);
}

// Single fused FAST sampler — drop-in replacement for aegis_sampler_topk_fused.
// grid (1), block (BLOCK threads). Shared mem layout (see runtime/sampling.rs):
//   hist[256] (u32) + sel_v[KCAP] (f32) + sel_i[KCAP] (u32) + scalars.
extern "C" __global__ void aegis_sampler_topk_fast(
    const float* __restrict__ logits,
    const unsigned int vocab,
    const unsigned int top_k,        // requested k; 0 or >KCAP => KCAP
    const float temperature,
    const float top_p,
    const float min_p,
    const float u,                   // uniform draw in [0,1)
    unsigned int* __restrict__ out_token
) {
    extern __shared__ unsigned int aegis_fast_smem[];
    unsigned int* hist = aegis_fast_smem;                 // [256] u32
    float* sel_v = (float*)(hist + 256);                  // [KCAP] f32
    unsigned int* sel_i = (unsigned int*)(sel_v + SAMPLER_KCAP); // [KCAP] u32
    __shared__ unsigned int s_count;     // generic shared counter/accumulator
    __shared__ unsigned int s_nsel;      // # gathered winners
    // Running radix-select state, shared so EVERY thread sees the prefix that
    // thread 0 fixes each pass (these MUST be shared, not per-thread registers —
    // otherwise only thread 0's histogram would be correctly restricted).
    __shared__ unsigned int s_prefix;
    __shared__ unsigned int s_mask;
    __shared__ unsigned int s_need;

    const unsigned int tid = threadIdx.x;
    const unsigned int nthreads = blockDim.x;

    unsigned int k = top_k;
    if (k == 0u || k > SAMPLER_KCAP) k = SAMPLER_KCAP;
    if (k > vocab) k = vocab;
    if (k == 0u) { if (tid == 0u) out_token[0] = 0u; return; }

    // ── Phase 1: byte-radix select of tau_vkey (k-th largest value key) ──
    // Invariant entering byte b: s_prefix holds the already-fixed higher bytes;
    // s_need = the rank still to locate within candidates whose higher bytes ==
    // s_prefix.
    if (tid == 0u) { s_prefix = 0u; s_mask = 0u; s_need = k; }
    __syncthreads();
    for (int byte = 3; byte >= 0; --byte) {
        const unsigned int shift = (unsigned int)byte * 8u;
        const unsigned int prefix = s_prefix;
        const unsigned int mask = s_mask;
        const unsigned int need = s_need;
        // Zero the histogram.
        for (unsigned int i = tid; i < 256u; i += nthreads) hist[i] = 0u;
        __syncthreads();
        // Histogram this byte over candidates matching the running prefix.
        for (unsigned int i = tid; i < vocab; i += nthreads) {
            const float x = logits[i];
            if (!isfinite(x)) continue;
            const unsigned int vk = aegis_f32_to_orderkey(x);
            if ((vk & mask) != prefix) continue;
            const unsigned int bin = (vk >> shift) & 0xFFu;
            atomicAdd(&hist[bin], 1u);
        }
        __syncthreads();
        // Thread 0 scans bins from high to low to find the bin holding the
        // `need`-th element (1-based, counting from the largest key).
        if (tid == 0u) {
            unsigned int acc = 0u;
            int sel_bin = 0;
            unsigned int below = 0u; // count strictly above the selected bin
            for (int b = 255; b >= 0; --b) {
                const unsigned int c = hist[(unsigned int)b];
                if (acc + c >= need) { sel_bin = b; below = acc; break; }
                acc += c;
            }
            // Fix this byte; narrow `need` to the rank within the selected bin.
            s_prefix = prefix | (((unsigned int)sel_bin) << shift);
            s_mask = mask | (0xFFu << shift);
            s_need = need - below;   // rank within the selected bin (next byte)
        }
        __syncthreads();
    }
    const unsigned int tau_vkey = s_prefix;

    // Count strictly-greater elements (n_gt) for the tie split.
    if (tid == 0u) s_count = 0u;
    __syncthreads();
    {
        unsigned int local = 0u;
        for (unsigned int i = tid; i < vocab; i += nthreads) {
            const float x = logits[i];
            if (!isfinite(x)) continue;
            if (aegis_f32_to_orderkey(x) > tau_vkey) ++local;
        }
        atomicAdd(&s_count, local);
    }
    __syncthreads();
    const unsigned int n_gt = s_count;
    const unsigned int need_eq = (k > n_gt) ? (k - n_gt) : 0u;  // ties to keep

    // ── Phase 2: among vkey==tau_vkey, find the need_eq-th SMALLEST index. ──
    // Byte-radix select on the index (ascending: scan bins from low to high).
    unsigned int tau_idx = 0xFFFFFFFFu;
    if (need_eq > 0u) {
        if (tid == 0u) { s_prefix = 0u; s_mask = 0u; s_need = need_eq; }
        __syncthreads();
        for (int byte = 3; byte >= 0; --byte) {
            const unsigned int shift = (unsigned int)byte * 8u;
            const unsigned int iprefix = s_prefix;
            const unsigned int imask = s_mask;
            const unsigned int ineed = s_need;
            for (unsigned int i = tid; i < 256u; i += nthreads) hist[i] = 0u;
            __syncthreads();
            for (unsigned int i = tid; i < vocab; i += nthreads) {
                const float x = logits[i];
                if (!isfinite(x)) continue;
                if (aegis_f32_to_orderkey(x) != tau_vkey) continue;
                if ((i & imask) != iprefix) continue;
                const unsigned int bin = (i >> shift) & 0xFFu;
                atomicAdd(&hist[bin], 1u);
            }
            __syncthreads();
            if (tid == 0u) {
                unsigned int acc = 0u;
                int sel_bin = 0;
                unsigned int below = 0u;
                for (int b = 0; b < 256; ++b) {           // ascending for index
                    const unsigned int c = hist[(unsigned int)b];
                    if (acc + c >= ineed) { sel_bin = b; below = acc; break; }
                    acc += c;
                }
                s_prefix = iprefix | (((unsigned int)sel_bin) << shift);
                s_mask = imask | (0xFFu << shift);
                s_need = ineed - below;
            }
            __syncthreads();
        }
        tau_idx = s_prefix;  // the need_eq-th smallest index in the tie group
    }

    // ── Gather the exact top-k set into shared mem. ──
    // Selected: vkey > tau_vkey, OR (vkey == tau_vkey AND index <= tau_idx).
    if (tid == 0u) s_nsel = 0u;
    __syncthreads();
    for (unsigned int i = tid; i < vocab; i += nthreads) {
        const float x = logits[i];
        if (!isfinite(x)) continue;
        const unsigned int vk = aegis_f32_to_orderkey(x);
        bool take = false;
        if (vk > tau_vkey) take = true;
        else if (vk == tau_vkey && i <= tau_idx) take = true;
        if (take) {
            const unsigned int slot = atomicAdd(&s_nsel, 1u);
            if (slot < SAMPLER_KCAP) { sel_v[slot] = x; sel_i[slot] = i; }
        }
    }
    __syncthreads();
    unsigned int nsel = s_nsel;
    if (nsel > SAMPLER_KCAP) nsel = SAMPLER_KCAP;  // defensive; should == k

    // Thread 0: sort the ≤k winners by (value desc, index asc), then run the
    // identical temperature/top-p/min-p/draw finish as aegis_sampler_topk_fused.
    if (tid != 0u) return;

    // Insertion sort (k ≤ 64).
    for (unsigned int a = 1u; a < nsel; ++a) {
        const float vv = sel_v[a];
        const unsigned int ii = sel_i[a];
        int j = (int)a - 1;
        while (j >= 0) {
            const bool jworse = sel_v[j] < vv || (sel_v[j] == vv && sel_i[j] > ii);
            if (!jworse) break;
            sel_v[j + 1] = sel_v[j];
            sel_i[j + 1] = sel_i[j];
            --j;
        }
        sel_v[j + 1] = vv;
        sel_i[j + 1] = ii;
    }

    const float temp = temperature > 1e-6f ? temperature : 1e-6f;
    const float max_logit = sel_v[0];   // global max over the top-k set
    float w[SAMPLER_KCAP];
    unsigned int idx[SAMPLER_KCAP];
    unsigned int nw = 0;
    for (unsigned int r = 0; r < nsel; ++r) {
        const float e = expf((sel_v[r] - max_logit) / temp);
        if (isfinite(e) && e > 0.0f) {
            w[nw] = e;
            idx[nw] = sel_i[r];
            ++nw;
        }
    }
    if (nw == 0u) { out_token[0] = sel_i[0]; return; }

    float total = 0.0f;
    for (unsigned int r = 0; r < nw; ++r) total += w[r];
    if (top_p > 0.0f && top_p < 1.0f && total > 0.0f) {
        const float cutoff = total * top_p;
        float cum = 0.0f;
        unsigned int keep = 0;
        for (unsigned int r = 0; r < nw; ++r) {
            cum += w[r];
            ++keep;
            if (cum >= cutoff) break;
        }
        if (keep < 1u) keep = 1u;
        nw = keep;
    }

    if (min_p > 0.0f) {
        const float thresh = w[0] * min_p;
        unsigned int m = 0;
        for (unsigned int r = 0; r < nw; ++r) {
            if (w[r] >= thresh) { w[m] = w[r]; idx[m] = idx[r]; ++m; }
        }
        nw = m;
        if (nw == 0u) { out_token[0] = sel_i[0]; return; }
    }

    float total2 = 0.0f;
    for (unsigned int r = 0; r < nw; ++r) total2 += w[r];
    if (total2 <= 0.0f) { out_token[0] = sel_i[0]; return; }
    float draw = u * total2;
    for (unsigned int r = 0; r < nw; ++r) {
        if (draw <= w[r]) { out_token[0] = idx[r]; return; }
        draw -= w[r];
    }
    out_token[0] = sel_i[0];
}

// Batched element-wise GeGLU (gelu_pytorch_tanh): out[i] = gelu_tanh(gate[i]) * up[i].
// Used by chunked prefill for the routed-expert GeGLU step on gathered token batches.
extern "C" __global__ void aegis_geglu_tanh_batched(
    const float* gate,
    const float* up,
    const unsigned int len,
    float* output
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        const float x = gate[idx];
        const float k = 0.7978845608028654f;
        const float k2 = 0.044715f;
        const float inner = k * (x + k2 * x * x * x);
        const float gelu = 0.5f * x * (1.0f + tanhf(inner));
        output[idx] = gelu * up[idx];
    }
}

// Speculative-decoding (EAGLE/MTP draft) kernels.
//
// The draft model produces a 256-dim hidden per step. Its LM head is
// "centroid-masked": the full 262144-row embed/lm_head matrix is never
// scored densely. Instead the draft hidden is scored against `num_centroids`
// (=2048) cluster centroids, the top-`centroid_intermediate_top_k` (=32)
// centroids are kept, and each centroid maps (via `token_ordering`) to a
// contiguous block of candidate token ids. The dense lm_head is then evaluated
// ONLY over the gathered candidate rows (~4K), turning a 262144-wide GEMV into
// a few-thousand-wide one.
//
// This file holds the device kernel that scores an explicit candidate-row list
// against the (shared) BF16 lm_head matrix. Centroid scoring itself reuses the
// existing dense `aegis_bf16_matvec_reference` kernel (2048 rows is cheap), and
// the top-k centroid selection + candidate-row materialization happen on the
// host (the candidate set is tiny). See `runtime/speculative.rs`.
//
// TODO(gpu-verify): confirm the exact centroid→token mapping semantics against
// the assistant checkpoint's `masked_embedding.token_ordering` tensor — this
// kernel only consumes a precomputed candidate-row index list, so the mapping
// math lives on the host and is the thing most likely to need correction.

// Sparse lm_head matvec over an explicit list of candidate rows.
//
//   for each i in [0, num_candidates):
//       r = candidate_rows[i]
//       logits[i] = sum_c bf16(lm_head[r, c]) * hidden[c]
//
// `lm_head` is row-major [vocab, cols] BF16; `hidden` is [cols] f32. One block
// per candidate, block-stride reduction over `cols` (mirrors
// `aegis_bf16_matvec_reference`). `bf16_to_float` comes from linear_utils.cuh.
extern "C" __global__ void aegis_spec_sparse_lm_head_matvec(
    const unsigned short* lm_head,      // [vocab, cols] BF16, row-major
    const float* hidden,                // [cols] f32
    const unsigned int* candidate_rows, // [num_candidates] token ids
    const unsigned int num_candidates,
    const unsigned int cols,
    float* logits                       // [num_candidates] f32 (out)
) {
    const unsigned int cand = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (cand >= num_candidates) {
        return;
    }
    const unsigned int row = candidate_rows[cand];

    extern __shared__ float partial[];
    float sum = 0.0f;
    const unsigned short* matrix_row = lm_head + size_t(row) * cols;
    for (unsigned int col = tid; col < cols; col += blockDim.x) {
        sum += bf16_to_float(matrix_row[col]) * hidden[col];
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
        logits[cand] = partial[0];
    }
}

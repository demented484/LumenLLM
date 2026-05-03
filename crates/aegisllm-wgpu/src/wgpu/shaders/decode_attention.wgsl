// Single-token (M=1) attention with online softmax for portable inference.
// One workgroup per attention head; each workgroup processes the full sequence
// sequentially in its single thread. Slow but correct; FlashAttention-style
// tiling would parallelize within the workgroup later.
//
// Layout:
//   q[num_q_heads, head_dim]
//   k[seq_len, num_kv_heads, head_dim]
//   v[seq_len, num_kv_heads, head_dim]
//   out[num_q_heads, head_dim]
// GQA: kv_head = q_head / (num_q_heads / num_kv_heads).
//
// Bindings:
//   0 (rw):  out
//   1 (ro):  q
//   2 (ro):  kv (concatenated [k_data, v_data]); we encode this as one buffer with kv_offset_v
//   3 (uniform): params

struct Params {
    num_q_heads  : u32,
    num_kv_heads : u32,
    head_dim     : u32,
    seq_len      : u32,
    kv_offset_v  : u32,  // float index where the V section starts in `kv` (== seq_len * num_kv_heads * head_dim)
    _pad0        : u32,
    _pad1        : u32,
    _pad2        : u32,
}

@group(0) @binding(0) var<storage, read_write> out    : array<f32>;
@group(0) @binding(1) var<storage, read>       q      : array<f32>;
@group(0) @binding(2) var<storage, read>       kv     : array<f32>;
@group(0) @binding(3) var<uniform>             params : Params;

@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) wid: vec3<u32>) {
    let q_head = wid.x;
    if (q_head >= params.num_q_heads) { return; }
    let group_size = params.num_q_heads / params.num_kv_heads;
    let kv_head = q_head / group_size;
    let head_dim = params.head_dim;
    let kv_width = params.num_kv_heads * head_dim;
    let scale = 1.0 / sqrt(f32(head_dim));

    // Inv-INFINITY surrogate that's safe across all backends.
    var max_score: f32 = -3.4e38;
    var sum: f32 = 0.0;
    // Local accumulator. Bounded by head_dim ≤ 256 in target models.
    var acc: array<f32, 256>;
    for (var i: u32 = 0u; i < head_dim; i = i + 1u) {
        acc[i] = 0.0;
    }

    let q_base = q_head * head_dim;
    for (var pos: u32 = 0u; pos < params.seq_len; pos = pos + 1u) {
        let k_base = pos * kv_width + kv_head * head_dim;
        // dot(q[q_head], k[pos, kv_head])
        var dot: f32 = 0.0;
        for (var i: u32 = 0u; i < head_dim; i = i + 1u) {
            dot = dot + q[q_base + i] * kv[k_base + i];
        }
        let score = dot * scale;
        // Online softmax: rescale if new max.
        if (score > max_score) {
            let rescale = exp(max_score - score);
            for (var i: u32 = 0u; i < head_dim; i = i + 1u) {
                acc[i] = acc[i] * rescale;
            }
            sum = sum * rescale;
            max_score = score;
        }
        let weight = exp(score - max_score);
        sum = sum + weight;
        let v_base = params.kv_offset_v + pos * kv_width + kv_head * head_dim;
        for (var i: u32 = 0u; i < head_dim; i = i + 1u) {
            acc[i] = acc[i] + weight * kv[v_base + i];
        }
    }

    let inv_sum = select(0.0, 1.0 / sum, sum > 0.0);
    for (var i: u32 = 0u; i < head_dim; i = i + 1u) {
        out[q_base + i] = acc[i] * inv_sum;
    }
}

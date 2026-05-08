// Per-row RMS norm: applies the SAME `weight[len]` vector to each of
// `batch` independent rows of `input[batch * len]`. Used by Gemma-4 for
// per-head Q/K/V norms applied between projection and RoPE.
//
//   for h in 0..batch:
//     row = input[h*len .. (h+1)*len]
//     mean_sq = sum(row^2) / len
//     inv_rms = 1 / sqrt(mean_sq + eps)
//     output[h*len .. (h+1)*len] = row * inv_rms * weight
//
// To use this for the V "no-weight" variant (Gemma-4 V norm has no
// learned weight, just a per-head unit-variance normalisation), bind an
// all-ones `weight` buffer of length `len` — the multiply by 1 collapses.

struct Params {
    batch: u32,
    len: u32,
    eps: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>       input  : array<f32>;
@group(0) @binding(1) var<storage, read>       weight : array<f32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;
@group(0) @binding(3) var<uniform>             params : Params;

var<workgroup> partial : array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let head = wid.x;
    if (head >= params.batch) { return; }
    let tid = lid.x;
    let len = params.len;
    let base = head * len;

    // Pass 1: sum of squares with stride-256 loop.
    var acc: f32 = 0.0;
    var i = tid;
    loop {
        if (i >= len) { break; }
        let v = input[base + i];
        acc = acc + v * v;
        i = i + 256u;
    }
    partial[tid] = acc;
    workgroupBarrier();

    // Tree reduction.
    var stride: u32 = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) {
            partial[tid] = partial[tid] + partial[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inv_rms = 1.0 / sqrt(partial[0] / f32(len) + params.eps);

    // Pass 2: normalise + per-element weight multiply.
    var j = tid;
    loop {
        if (j >= len) { break; }
        output[base + j] = input[base + j] * inv_rms * weight[j];
        j = j + 256u;
    }
}

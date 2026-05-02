struct Params { len: u32, eps: f32 }

@group(0) @binding(0) var<storage, read>       input  : array<f32>;
@group(0) @binding(1) var<storage, read>       weight : array<f32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;
@group(0) @binding(3) var<uniform>             params : Params;

var<workgroup> partial : array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    var acc: f32 = 0.0;
    var i = tid;
    loop { if (i >= params.len) { break; } let v = input[i]; acc = acc + v*v; i = i + 256u; }
    partial[tid] = acc;
    workgroupBarrier();

    var stride: u32 = 128u;
    loop {
        if (stride == 0u) { break; }
        if (tid < stride) { partial[tid] = partial[tid] + partial[tid + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inv_rms = 1.0 / sqrt(partial[0] / f32(params.len) + params.eps);

    var j = tid;
    loop { if (j >= params.len) { break; } output[j] = input[j] * inv_rms * weight[j]; j = j + 256u; }
}

// RoPE: rotates pairs (x[i], x[i+half_dim]) by precomputed cos/sin tables.
// Layout: input has shape [num_heads, head_dim]. cos/sin tables have length half_dim.
// The tables are computed by the host once per position (or per batch of positions).
// Operation:  x'[i]        = x[i] * cos[i] - x[i + half] * sin[i]
//             x'[i + half] = x[i] * sin[i] + x[i + half] * cos[i]

struct Params {
    num_heads : u32,
    head_dim  : u32,
    half_dim  : u32,
    _pad      : u32,
}

@group(0) @binding(0) var<storage, read_write> values    : array<f32>;
@group(0) @binding(1) var<storage, read>       cos_table : array<f32>;
@group(0) @binding(2) var<storage, read>       sin_table : array<f32>;
@group(0) @binding(3) var<uniform>             params    : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let head = gid.y;
    let i = gid.x;
    if (head >= params.num_heads || i >= params.half_dim) { return; }
    let base = head * params.head_dim;
    let x0 = values[base + i];
    let x1 = values[base + i + params.half_dim];
    let c = cos_table[i];
    let s = sin_table[i];
    values[base + i]                     = x0 * c - x1 * s;
    values[base + i + params.half_dim]   = x0 * s + x1 * c;
}

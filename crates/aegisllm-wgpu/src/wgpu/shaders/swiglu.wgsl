// SwiGLU activation: out[i] = silu(gate[i]) * up[i] where silu(x) = x / (1 + exp(-x)).
// Used in MLP between gate_proj and down_proj.

struct Params { len: u32 }

@group(0) @binding(0) var<storage, read>       gate   : array<f32>;
@group(0) @binding(1) var<storage, read>       up     : array<f32>;
@group(0) @binding(2) var<storage, read_write> out    : array<f32>;
@group(0) @binding(3) var<uniform>             params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.len) { return; }
    let g = gate[i];
    let silu = g / (1.0 + exp(-g));
    out[i] = silu * up[i];
}

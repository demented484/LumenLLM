// In-place element-wise scalar multiply: `data[i] *= scale`.
// Used for Gemma-4 `embed_scale = sqrt(hidden_size)` after embedding
// lookup, per-layer `layer_scalar`, and the post-RoPE Q scaling that
// cancels Gemma-4's attention-scaling-=1.0 quirk against our kernel's
// hardcoded 1/sqrt(d) softmax scale.
//
// Bindings match the standard 4-storage layout used by the rest of the
// dispatcher: bindings 0 and 1 are read-only (we ignore them), binding 2
// is read-write (the actual data we mutate), binding 3 is uniform.
// Caller passes `data` as the "out" slot when invoking
// `dispatch_three_storage_device`.

struct Params {
    len: u32,
    scale: f32,
}

@group(0) @binding(0) var<storage, read_write> data   : array<f32>;
@group(0) @binding(1) var<uniform>             params : Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.len) {
        return;
    }
    data[i] = data[i] * params.scale;
}

// Embedding lookup: out[i] = embed_table[token_id, i].
// embed_table is row-major [vocab_size, hidden_size]; token_id selects one row.
// One workgroup per output element across `hidden_size` lanes.

struct Params {
    token_id    : u32,
    hidden_size : u32,
}

@group(0) @binding(0) var<storage, read>       embed_table : array<f32>;
@group(0) @binding(1) var<storage, read>       _unused     : array<f32>; // pad to standard 4-binding layout
@group(0) @binding(2) var<storage, read_write> out         : array<f32>;
@group(0) @binding(3) var<uniform>             params      : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.hidden_size) { return; }
    out[i] = embed_table[params.token_id * params.hidden_size + i];
}

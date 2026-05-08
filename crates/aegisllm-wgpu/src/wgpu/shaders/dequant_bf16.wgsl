// BF16 → f32 dequant. WGSL has no native bf16 type, so we store BF16
// weights as `array<u32>` where each u32 holds two consecutive bf16
// values: bits[0..16) = even-index value, bits[16..32) = odd-index.
// Each output f32 is reconstructed by zero-extending the bf16 bits to
// the high 16 bits of an f32 (the `bitcast<f32>(bf16_bits << 16)`
// trick — exact for all BF16-representable values).
//
// Input length-in-bf16-values = `len`; the input buffer holds
// `(len + 1) / 2` u32 words.

struct Params {
    len: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>       packed : array<u32>;
@group(0) @binding(1) var<storage, read>       _unused : array<u32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;
@group(0) @binding(3) var<uniform>             params : Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.len) {
        return;
    }
    let word = packed[i / 2u];
    let bf16_bits: u32 = select(word >> 16u, word & 0xFFFFu, (i & 1u) == 0u);
    output[i] = bitcast<f32>(bf16_bits << 16u);
}

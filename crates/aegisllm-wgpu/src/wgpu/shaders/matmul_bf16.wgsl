// Matmul with BF16-packed weights: C[M,N] = A[M,K] @ B^T[N,K] where
// B is stored as `array<u32>` packing 2 BF16 values per word (lo
// bits = even index, hi bits = odd). Decodes one BF16 → f32 in the
// inner loop via the standard `(bf16_bits << 16) bitcast f32` trick.
//
// Avoids the need to dequant the entire weight matrix into a giant
// scratch buffer before the matmul — critical for large lm_head
// against tied embeddings (2.95 GiB of f32 wouldn't fit alongside
// model weights in 16 GiB VRAM).

struct MatMulParams {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>       a      : array<f32>;
@group(0) @binding(1) var<storage, read>       b      : array<u32>;  // BF16 packed
@group(0) @binding(2) var<storage, read_write> c      : array<f32>;
@group(0) @binding(3) var<uniform>             params : MatMulParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    let col = gid.y;
    if (row >= params.m || col >= params.n) {
        return;
    }
    var acc: f32 = 0.0;
    let b_base = col * params.k;  // index in BF16 elements, not u32 words
    var i: u32 = 0u;
    loop {
        if (i >= params.k) { break; }
        let bf16_idx = b_base + i;
        let word = b[bf16_idx / 2u];
        let bf16_bits: u32 = select(word >> 16u, word & 0xFFFFu, (bf16_idx & 1u) == 0u);
        let b_val = bitcast<f32>(bf16_bits << 16u);
        acc = acc + a[row * params.k + i] * b_val;
        i = i + 1u;
    }
    c[row * params.n + col] = acc;
}

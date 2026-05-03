// f32 matrix multiplication: C[M,N] = A[M,K] @ B^T[N,K]  (B stored row-major as [N,K]).
// Each invocation computes one output element. Workgroup size 8x8.

struct MatMulParams {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>       a      : array<f32>;
@group(0) @binding(1) var<storage, read>       b      : array<f32>;
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
    let a_base = row * params.k;
    let b_base = col * params.k;
    var i: u32 = 0u;
    loop {
        if (i >= params.k) { break; }
        acc = acc + a[a_base + i] * b[b_base + i];
        i = i + 1u;
    }
    c[row * params.n + col] = acc;
}

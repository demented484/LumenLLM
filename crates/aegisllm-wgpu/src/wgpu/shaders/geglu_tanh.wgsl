// GeGLU (tanh-approximation) activation, used by Gemma-4 routed
// experts and shared MLP. `out[i] = gelu_tanh(gate[i]) * up[i]` where
//
//   gelu_tanh(x) = 0.5 * x * (1 + tanh( sqrt(2/π) * (x + 0.044715 * x³) ))
//
// This is the `pytorch_tanh` GELU approximation that
// `transformers.activations.GELUTanh` uses; matches Gemma-4's
// HuggingFace reference exactly.

struct LenParams {
    len: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>       gate   : array<f32>;
@group(0) @binding(1) var<storage, read>       up     : array<f32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;
@group(0) @binding(3) var<uniform>             params : LenParams;

const SQRT_2_OVER_PI : f32 = 0.7978845608028654;
const GELU_COEFF     : f32 = 0.044715;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.len) {
        return;
    }
    let g = gate[i];
    let inner = SQRT_2_OVER_PI * (g + GELU_COEFF * g * g * g);
    let act = 0.5 * g * (1.0 + tanh(inner));
    output[i] = act * up[i];
}

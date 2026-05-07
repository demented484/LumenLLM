// NVFP4 → f32 dequantization shader.
//
// NVFP4 layout (matches the CUDA-side helpers in
// `aegisllm-cuda/src/cuda/kernels/blackwell/linear_utils.cuh`):
//
//   * Packed: row-major `[rows, cols/2]` bytes. Each byte holds two 4-bit
//     elements (low nibble = even col, high nibble = odd col). The
//     decoded codepoint is given by `decode_nvfp4_nibble` below.
//   * Scales: row-major `[rows, cols/16]` bytes (UE4M3-half encoding,
//     decoded by `decode_ue4m3_half`). One scale covers a 16-element block
//     along the K (cols) axis. NVFP4 carries an extra 0.5× factor in the
//     scale relative to standard E4M3.
//   * Output scale: per-tensor f32 multiplier (the `output_scale` field on
//     `DeviceNvfp4Linear`). Applied after nibble × block scale.
//
// Output: row-major `[rows, cols]` f32. Suitable as the `b` matrix for
// `matmul_f32_gpu` (transposed-K GEMM convention used by the rest of the
// wgpu forward path).
//
// All buffers are `array<u32>` because WGSL storage buffers don't expose
// u8 arrays directly. We unpack bytes from the u32 word at runtime via
// shifts. Output `output_scale` is passed as `bitcast<u32>(f32)` so it
// fits a uniform struct without alignment/padding gymnastics.

struct DequantParams {
    rows: u32,
    cols: u32,
    output_scale_bits: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>       packed_u32 : array<u32>;
@group(0) @binding(1) var<storage, read>       scales_u32 : array<u32>;
@group(0) @binding(2) var<storage, read_write> output_f32 : array<f32>;
@group(0) @binding(3) var<uniform>             params     : DequantParams;

// Returns the signed integer codepoint for a 4-bit NVFP4 nibble.
// Matches `decode_nvfp4_nibble()` in linear_utils.cuh exactly so the
// WGPU dequant produces the same f32 values as the CUDA reference path.
fn decode_nvfp4_nibble(nibble: u32) -> i32 {
    var v: i32 = 0;
    switch (nibble & 0xFu) {
        case 0u:  { v =   0; }
        case 1u:  { v =   1; }
        case 2u:  { v =   2; }
        case 3u:  { v =   3; }
        case 4u:  { v =   4; }
        case 5u:  { v =   6; }
        case 6u:  { v =   8; }
        case 7u:  { v =  12; }
        case 8u:  { v =   0; }
        case 9u:  { v =  -1; }
        case 10u: { v =  -2; }
        case 11u: { v =  -3; }
        case 12u: { v =  -4; }
        case 13u: { v =  -6; }
        case 14u: { v =  -8; }
        default:  { v = -12; }
    }
    return v;
}

// UE4M3-half decoder. Mirrors `decode_ue4m3_half()` in linear_utils.cuh
// — including the `* 0.5` NVFP4-specific tail.
fn decode_ue4m3_half(byte: u32) -> f32 {
    let masked = byte & 0x7Fu;
    if (masked == 0u || masked == 0x7Fu) {
        return 0.0;
    }
    let exponent = i32((masked >> 3u) & 0x0Fu);
    let mantissa = f32(masked & 0x07u);
    var raw: f32;
    if (exponent == 0) {
        // Subnormal: mantissa * 2^(-9).
        raw = mantissa * 0.001953125;
    } else {
        // Normal: (1 + mantissa/8) * 2^(exp-7).
        raw = (1.0 + mantissa * 0.125) * exp2(f32(exponent - 7));
    }
    return raw * 0.5;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let elem = gid.x;
    let total = params.rows * params.cols;
    if (elem >= total) {
        return;
    }
    let row = elem / params.cols;
    let col = elem % params.cols;

    // Packed: each row has cols/2 bytes; low nibble = even col.
    let packed_byte_idx = row * (params.cols / 2u) + (col / 2u);
    let packed_word = packed_u32[packed_byte_idx / 4u];
    let byte_shift = (packed_byte_idx % 4u) * 8u;
    let byte = (packed_word >> byte_shift) & 0xFFu;
    var nibble: u32;
    if ((col & 1u) == 0u) {
        nibble = byte & 0x0Fu;
    } else {
        nibble = byte >> 4u;
    }

    // Scales: each row has cols/16 bytes; one scale per 16-element block.
    let scale_byte_idx = row * (params.cols / 16u) + (col / 16u);
    let scale_word = scales_u32[scale_byte_idx / 4u];
    let scale_shift = (scale_byte_idx % 4u) * 8u;
    let scale_byte = (scale_word >> scale_shift) & 0xFFu;
    let block_scale = decode_ue4m3_half(scale_byte);

    let value = f32(decode_nvfp4_nibble(nibble)) * block_scale;
    let output_scale = bitcast<f32>(params.output_scale_bits);
    output_f32[row * params.cols + col] = value * output_scale;
}

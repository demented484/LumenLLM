use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt as _;

use super::loader::{KernelPipeline, WgpuContext};
use aegisllm_base::error::{AegisError, Result};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    len: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct LenParams {
    len: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MatMulParams {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct RopeParams {
    num_heads: u32,
    head_dim: u32,
    half_dim: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct EmbeddingParams {
    token_id: u32,
    hidden_size: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct DecodeAttentionParams {
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    kv_offset_v: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Run the standard 3-storage-buffer kernel pattern: read in1, in2, write out, plus uniform.
/// Used for rms_norm, swiglu, residual_add, matmul.
#[allow(clippy::too_many_arguments)]
fn run_three_storage_kernel(
    ctx: &WgpuContext,
    pipe: &KernelPipeline,
    in1: &[f32],
    in2: &[f32],
    out_len: usize,
    uniform_bytes: &[u8],
    workgroups: (u32, u32, u32),
    label: &'static str,
) -> Result<Vec<f32>> {
    let out_byte_len = (out_len * std::mem::size_of::<f32>()) as u64;

    let in1_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(in1),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let in2_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(in2),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: out_byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let uniform_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: uniform_bytes,
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let staging_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: out_byte_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &pipe.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: in1_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: in2_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: uniform_buf.as_entire_binding() },
        ],
    });

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some(label),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipe.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    }
    encoder.copy_buffer_to_buffer(&out_buf, 0, &staging_buf, 0, out_byte_len);
    ctx.queue.submit(std::iter::once(encoder.finish()));

    readback_f32(&ctx.device, &staging_buf, out_byte_len, label)
}

fn readback_f32(
    device: &wgpu::Device,
    staging: &wgpu::Buffer,
    byte_len: u64,
    label: &'static str,
) -> Result<Vec<f32>> {
    let _ = byte_len;
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| { tx.send(result).ok(); });
    device.poll(wgpu::Maintain::Wait);
    rx.recv()
        .map_err(|_| AegisError::Unsupported(format!("wgpu {label} readback channel closed")))?
        .map_err(|e| AegisError::Unsupported(format!("wgpu {label} map_async failed: {e:?}")))?;
    let data = slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    Ok(result)
}

pub fn rms_norm_gpu(
    ctx: &WgpuContext,
    input: &[f32],
    weight: &[f32],
    eps: f32,
) -> Result<Vec<f32>> {
    let len = input.len().min(weight.len());
    let params = Params { len: len as u32, eps };
    run_three_storage_kernel(
        ctx,
        &ctx.rms_norm,
        &input[..len],
        &weight[..len],
        len,
        bytemuck::bytes_of(&params),
        (1, 1, 1),
        "rms_norm",
    )
}

/// SwiGLU on wgpu: out[i] = silu(gate[i]) * up[i].
pub fn swiglu_gpu(
    ctx: &WgpuContext,
    gate: &[f32],
    up: &[f32],
) -> Result<Vec<f32>> {
    let len = gate.len().min(up.len());
    let params = LenParams { len: len as u32, _pad: 0 };
    let groups = ((len + 63) / 64) as u32;
    run_three_storage_kernel(
        ctx,
        &ctx.swiglu,
        &gate[..len],
        &up[..len],
        len,
        bytemuck::bytes_of(&params),
        (groups, 1, 1),
        "swiglu",
    )
}

/// Element-wise add on wgpu: out[i] = a[i] + b[i].
pub fn residual_add_gpu(
    ctx: &WgpuContext,
    a: &[f32],
    b: &[f32],
) -> Result<Vec<f32>> {
    let len = a.len().min(b.len());
    let params = LenParams { len: len as u32, _pad: 0 };
    let groups = ((len + 63) / 64) as u32;
    run_three_storage_kernel(
        ctx,
        &ctx.residual_add,
        &a[..len],
        &b[..len],
        len,
        bytemuck::bytes_of(&params),
        (groups, 1, 1),
        "residual_add",
    )
}

/// Matrix multiplication on wgpu: C[M, N] = A[M, K] @ B^T[N, K] (B stored as row-major [N, K]).
pub fn matmul_f32_gpu(
    ctx: &WgpuContext,
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Vec<f32>> {
    if a.len() != m * k || b.len() != n * k {
        return Err(AegisError::InvalidPlan(format!(
            "matmul shape mismatch: a={} expected={} b={} expected={} (m={} n={} k={})",
            a.len(), m * k, b.len(), n * k, m, n, k
        )));
    }
    let params = MatMulParams { m: m as u32, n: n as u32, k: k as u32, _pad: 0 };
    let groups_x = ((m + 7) / 8) as u32;
    let groups_y = ((n + 7) / 8) as u32;
    run_three_storage_kernel(
        ctx,
        &ctx.matmul,
        a,
        b,
        m * n,
        bytemuck::bytes_of(&params),
        (groups_x, groups_y, 1),
        "matmul",
    )
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct DequantParams {
    rows: u32,
    cols: u32,
    output_scale_bits: u32,
    _pad: u32,
}

/// Dequantize a row-major NVFP4 weight `[rows, cols]` to f32 on the GPU.
///
/// `packed_bytes`: `rows * cols / 2` bytes (low nibble = even col).
/// `scales_bytes`: `rows * cols / 16` bytes (UE4M3-half, NVFP4 0.5× tail
/// included via shader's `decode_ue4m3_half`).
/// `output_scale`: per-tensor f32 multiplier (the `output_scale` field of
/// `DeviceNvfp4Linear`); `1.0` if the tensor has none.
///
/// Returns a `Vec<f32>` of length `rows * cols`. Suitable as the `b`
/// matrix for `matmul_f32_gpu`. This is the bridge that lets the wgpu
/// forward path consume Gemma-4 NVFP4 weights without committing to an
/// f32 weight upload.
pub fn dequant_nvfp4_gpu(
    ctx: &WgpuContext,
    packed_bytes: &[u8],
    scales_bytes: &[u8],
    rows: usize,
    cols: usize,
    output_scale: f32,
) -> Result<Vec<f32>> {
    if cols % 16 != 0 {
        return Err(AegisError::InvalidPlan(format!(
            "dequant_nvfp4: cols ({cols}) must be a multiple of 16",
        )));
    }
    let expected_packed = rows * cols / 2;
    let expected_scales = rows * cols / 16;
    if packed_bytes.len() != expected_packed || scales_bytes.len() != expected_scales {
        return Err(AegisError::InvalidPlan(format!(
            "dequant_nvfp4 size mismatch: packed={} expected={} scales={} expected={}",
            packed_bytes.len(), expected_packed, scales_bytes.len(), expected_scales,
        )));
    }
    // wgpu storage buffers are read as `array<u32>` in WGSL; pad input
    // byte arrays to u32 alignment so the shader's word-shifting is safe
    // for any row/col combination.
    let pad_to_u32 = |bytes: &[u8]| -> Vec<u8> {
        let pad = (4 - bytes.len() % 4) % 4;
        let mut v = Vec::with_capacity(bytes.len() + pad);
        v.extend_from_slice(bytes);
        v.extend(std::iter::repeat(0u8).take(pad));
        v
    };
    let packed_u32 = pad_to_u32(packed_bytes);
    let scales_u32 = pad_to_u32(scales_bytes);
    let out_len = rows * cols;
    let out_bytes = (out_len * std::mem::size_of::<f32>()) as u64;

    let packed_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dequant_nvfp4 packed"),
        contents: &packed_u32,
        usage: wgpu::BufferUsages::STORAGE,
    });
    let scales_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dequant_nvfp4 scales"),
        contents: &scales_u32,
        usage: wgpu::BufferUsages::STORAGE,
    });
    let out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dequant_nvfp4 out"),
        size: out_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let params = DequantParams {
        rows: rows as u32,
        cols: cols as u32,
        output_scale_bits: output_scale.to_bits(),
        _pad: 0,
    };
    let uniform_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("dequant_nvfp4 uniform"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let staging_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("dequant_nvfp4 staging"),
        size: out_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("dequant_nvfp4 bind"),
        layout: &ctx.dequant_nvfp4.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: packed_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: scales_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: uniform_buf.as_entire_binding() },
        ],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("dequant_nvfp4 enc") });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("dequant_nvfp4 pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.dequant_nvfp4.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let groups = ((out_len + 63) / 64) as u32;
        pass.dispatch_workgroups(groups, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out_buf, 0, &staging_buf, 0, out_bytes);
    ctx.queue.submit(Some(encoder.finish()));

    readback_f32(&ctx.device, &staging_buf, out_bytes, "dequant_nvfp4")
}

/// Embedding lookup on wgpu: returns row `token_id` of `embed_table` (shape [vocab, hidden]).
/// Returns the hidden_size-element row as a Vec<f32>.
pub fn embedding_gpu(
    ctx: &WgpuContext,
    embed_table: &[f32],
    token_id: u32,
    hidden_size: usize,
) -> Result<Vec<f32>> {
    if embed_table.len() < ((token_id as usize) + 1) * hidden_size {
        return Err(AegisError::InvalidPlan(format!(
            "embedding lookup out of range: token_id={token_id} hidden_size={hidden_size} table_len={}",
            embed_table.len()
        )));
    }
    let params = EmbeddingParams { token_id, hidden_size: hidden_size as u32 };
    // Reuse the standard kernel runner: pass the embed_table as in1, a 1-element dummy as in2.
    let dummy = [0.0_f32];
    let groups = ((hidden_size + 63) / 64) as u32;
    run_three_storage_kernel(
        ctx,
        &ctx.embedding,
        embed_table,
        &dummy,
        hidden_size,
        bytemuck::bytes_of(&params),
        (groups, 1, 1),
        "embedding",
    )
}

/// Single-token attention on wgpu (M=1 / decode). Online softmax across `seq_len` keys/values.
/// `q` shape: [num_q_heads, head_dim]; `keys` & `values`: [seq_len, num_kv_heads, head_dim].
/// GQA: kv_head = q_head / (num_q_heads / num_kv_heads).
pub fn decode_attention_gpu(
    ctx: &WgpuContext,
    q: &[f32],
    keys: &[f32],
    values: &[f32],
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Result<Vec<f32>> {
    if num_q_heads % num_kv_heads != 0 {
        return Err(AegisError::InvalidPlan(format!(
            "decode_attention: num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        )));
    }
    if head_dim > 256 {
        return Err(AegisError::Unsupported(format!(
            "decode_attention WGSL kernel hard-codes max head_dim=256, got {head_dim}"
        )));
    }
    let q_len = num_q_heads * head_dim;
    let kv_width = num_kv_heads * head_dim;
    let kv_len = seq_len * kv_width;
    if q.len() != q_len {
        return Err(AegisError::InvalidPlan(format!(
            "decode_attention: q.len()={} expected={q_len}", q.len()
        )));
    }
    if keys.len() != kv_len || values.len() != kv_len {
        return Err(AegisError::InvalidPlan(format!(
            "decode_attention: keys/values len mismatch: keys={} values={} expected={}",
            keys.len(), values.len(), kv_len
        )));
    }
    // Concatenate keys + values into a single buffer for binding 2.
    let mut kv_concat = Vec::with_capacity(kv_len * 2);
    kv_concat.extend_from_slice(keys);
    kv_concat.extend_from_slice(values);

    let params = DecodeAttentionParams {
        num_q_heads: num_q_heads as u32,
        num_kv_heads: num_kv_heads as u32,
        head_dim: head_dim as u32,
        seq_len: seq_len as u32,
        kv_offset_v: kv_len as u32,
        _pad0: 0, _pad1: 0, _pad2: 0,
    };
    let byte_len = (q_len * std::mem::size_of::<f32>()) as u64;

    let out_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("attn_out"),
        size: byte_len,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let q_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("attn_q"),
        contents: bytemuck::cast_slice(q),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let kv_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("attn_kv"),
        contents: bytemuck::cast_slice(&kv_concat),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let uniform_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("attn_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let staging_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("attn_staging"),
        size: byte_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("attn_bg"),
        layout: &ctx.decode_attention.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: out_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: q_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: kv_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: uniform_buf.as_entire_binding() },
        ],
    });

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("attn_enc"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("attn_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.decode_attention.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(num_q_heads as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&out_buf, 0, &staging_buf, 0, byte_len);
    ctx.queue.submit(std::iter::once(encoder.finish()));

    readback_f32(&ctx.device, &staging_buf, byte_len, "decode_attention")
}

/// RoPE in place on wgpu: rotates pairs (x[i], x[i + half_dim]) by precomputed cos/sin.
/// Returns the rotated values as a new Vec (the input is consumed by buffer copy).
pub fn rope_gpu(
    ctx: &WgpuContext,
    values: &[f32],
    cos_table: &[f32],
    sin_table: &[f32],
    num_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    let half_dim = head_dim / 2;
    if values.len() != num_heads * head_dim {
        return Err(AegisError::InvalidPlan(format!(
            "rope shape mismatch: values={} expected={}", values.len(), num_heads * head_dim
        )));
    }
    if cos_table.len() != half_dim || sin_table.len() != half_dim {
        return Err(AegisError::InvalidPlan(format!(
            "rope table size mismatch: cos={} sin={} expected={}",
            cos_table.len(), sin_table.len(), half_dim
        )));
    }
    let params = RopeParams {
        num_heads: num_heads as u32,
        head_dim: head_dim as u32,
        half_dim: half_dim as u32,
        _pad: 0,
    };
    let byte_len = (values.len() * std::mem::size_of::<f32>()) as u64;

    // Read-write storage holding the rotated values.
    let values_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rope_values"),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let cos_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rope_cos"),
        contents: bytemuck::cast_slice(cos_table),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let sin_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rope_sin"),
        contents: bytemuck::cast_slice(sin_table),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let uniform_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("rope_params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let staging_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rope_staging"),
        size: byte_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rope_bg"),
        layout: &ctx.rope.bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: values_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: cos_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: sin_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: uniform_buf.as_entire_binding() },
        ],
    });

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("rope_enc"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rope_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&ctx.rope.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        let groups_x = ((half_dim + 63) / 64) as u32;
        let groups_y = num_heads as u32;
        pass.dispatch_workgroups(groups_x, groups_y, 1);
    }
    encoder.copy_buffer_to_buffer(&values_buf, 0, &staging_buf, 0, byte_len);
    ctx.queue.submit(std::iter::once(encoder.finish()));

    readback_f32(&ctx.device, &staging_buf, byte_len, "rope")
}

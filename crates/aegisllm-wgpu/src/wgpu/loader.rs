use aegisllm_base::error::{AegisError, Result};

impl std::fmt::Debug for WgpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuContext").finish_non_exhaustive()
    }
}

/// Holds device, queue, and all compute pipelines for the wgpu backend.
/// Pipelines are created upfront so per-call overhead is just buffer setup + dispatch.
pub struct WgpuContext {
    pub(super) device: wgpu::Device,
    pub(super) queue: wgpu::Queue,
    // Shaders that use the standard 4-binding layout: (storage, storage, storage rw, uniform).
    pub(super) rms_norm: KernelPipeline,
    /// Per-row RMS norm — Gemma-4's per-head Q/K/V norms.
    pub(super) rms_norm_batched: KernelPipeline,
    /// In-place scalar multiply — Gemma-4's embed_scale, layer_scalar,
    /// and post-RoPE Q scale.
    pub(super) scale_f32: KernelPipeline,
    pub(super) swiglu: KernelPipeline,
    /// Gemma-4 uses GeGLU (tanh-approximation GELU) instead of SwiGLU.
    pub(super) geglu_tanh: KernelPipeline,
    pub(super) residual_add: KernelPipeline,
    pub(super) embedding: KernelPipeline,
    // Matmul: same 4-binding layout but different uniform shape (m,n,k,_pad).
    pub(super) matmul: KernelPipeline,
    /// NVFP4 → f32 dequantization (`shaders/dequant_nvfp4.wgsl`). Bridges
    /// quantized weight buffers (Gemma-4 routed experts, attention proj
    /// in NVFP4 source) to the f32-only matmul kernel. Same 4-binding
    /// layout as everyone else: packed (ro), scales (ro), output (rw),
    /// uniform.
    pub(super) dequant_nvfp4: KernelPipeline,
    // RoPE: storage rw + 2 storage read + uniform.
    pub(super) rope: KernelPipeline,
    // Decode attention: storage rw (out) + 2 storage read (q, kv) + uniform.
    pub(super) decode_attention: KernelPipeline,
}

/// A compute pipeline together with the bind group layout it expects.
pub(super) struct KernelPipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl WgpuContext {
    /// Accessor for the underlying `wgpu::Device`. Public so integration
    /// tests and downstream callers can build their own command encoders
    /// against the same device the primitives use.
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Accessor for the underlying `wgpu::Queue`. Public for the same
    /// reason as [`WgpuContext::device`] — letting callers submit
    /// command buffers and write to buffers built from primitives'
    /// output.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn new(device_index: usize) -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapters: Vec<wgpu::Adapter> =
            instance.enumerate_adapters(wgpu::Backends::PRIMARY);
        let adapter = adapters.into_iter().nth(device_index).ok_or_else(|| {
            AegisError::Unsupported("no wgpu adapter available".into())
        })?;

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("aegis-wgpu"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .map_err(|e| AegisError::Unsupported(format!("wgpu request_device failed: {e}")))?;

        // Standard 4-binding layout for kernels with (in1, in2, out, uniform).
        let standard_4_layout = make_standard_4_layout(&device, "standard_4");
        let rms_norm = build_kernel(
            &device,
            include_str!("shaders/rms_norm.wgsl"),
            &standard_4_layout,
            "rms_norm",
        );
        let rms_norm_batched = build_kernel(
            &device,
            include_str!("shaders/rms_norm_batched.wgsl"),
            &standard_4_layout,
            "rms_norm_batched",
        );
        let scale_f32 = build_kernel(
            &device,
            include_str!("shaders/scale_f32.wgsl"),
            &standard_4_layout,
            "scale_f32",
        );
        let swiglu = build_kernel(
            &device,
            include_str!("shaders/swiglu.wgsl"),
            &standard_4_layout,
            "swiglu",
        );
        let geglu_tanh = build_kernel(
            &device,
            include_str!("shaders/geglu_tanh.wgsl"),
            &standard_4_layout,
            "geglu_tanh",
        );
        let residual_add = build_kernel(
            &device,
            include_str!("shaders/residual_add.wgsl"),
            &standard_4_layout,
            "residual_add",
        );
        let matmul = build_kernel(
            &device,
            include_str!("shaders/matmul_f32.wgsl"),
            &standard_4_layout,
            "matmul",
        );
        let dequant_nvfp4 = build_kernel(
            &device,
            include_str!("shaders/dequant_nvfp4.wgsl"),
            &standard_4_layout,
            "dequant_nvfp4",
        );
        let embedding = build_kernel(
            &device,
            include_str!("shaders/embedding.wgsl"),
            &standard_4_layout,
            "embedding",
        );
        let rope_layout = make_rope_layout(&device);
        let rope = build_kernel(
            &device,
            include_str!("shaders/rope.wgsl"),
            &rope_layout,
            "rope",
        );
        // decode_attention has same binding shape as rope (rw, ro, ro, uniform).
        let decode_attention = build_kernel(
            &device,
            include_str!("shaders/decode_attention.wgsl"),
            &rope_layout,
            "decode_attention",
        );

        Ok(Self {
            device,
            queue,
            rms_norm,
            rms_norm_batched,
            scale_f32,
            swiglu,
            geglu_tanh,
            residual_add,
            embedding,
            matmul,
            dequant_nvfp4,
            rope,
            decode_attention,
        })
    }
}

fn standard_4_entries(read_only_first: bool) -> [wgpu::BindGroupLayoutEntry; 4] {
    [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: read_only_first },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
    ]
}

fn make_standard_4_layout(device: &wgpu::Device, label: &str) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &standard_4_entries(true),
    })
}

fn make_rope_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    // RoPE: binding 0 is read-write storage (values rotated in place),
    // bindings 1-2 are read-only storage (cos/sin tables), binding 3 is uniform.
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("rope"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

fn build_kernel(
    device: &wgpu::Device,
    source: &str,
    bind_group_layout: &wgpu::BindGroupLayout,
    label: &str,
) -> KernelPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    // Use the pipeline's actual layout (handle to BGL #0). This is guaranteed to
    // match what the shader expects at bind group creation time.
    let bgl = pipeline.get_bind_group_layout(0);
    KernelPipeline {
        pipeline,
        bind_group_layout: bgl,
    }
}

//! Math-backend module connecting GPU execution via wgpu and providing a CPU fallback.

use std::sync::Arc;
use std::sync::OnceLock;
use anyhow::{anyhow, Result};
use parking_lot::Mutex as ParkingMutex;

/// Maximum FFN intermediate dimension supported. Sized for Mixtral-8x7B (d_ff=14336).
/// Increase if using models with larger d_ff.
const MAX_EXPERT_D_FF: usize = 16_384;

// Embed WGSL shaders using include_str
const MATMUL_SHADER: &str = include_str!("wgpu_shaders/matmul.wgsl");
const MATMUL_Q4_0_SHADER: &str = include_str!("wgpu_shaders/matmul_q4_0.wgsl");
const SWIGLU_SHADER: &str = include_str!("wgpu_shaders/swiglu.wgsl");
const SOFTMAX_SHADER: &str = include_str!("wgpu_shaders/softmax.wgsl");
const ATTENTION_SHADER: &str = include_str!("wgpu_shaders/attention.wgsl");

/// Zero-copy view of a f16 tensor borrowed from the caller.
#[derive(Copy, Clone, Debug)]
pub struct TensorView<'a> {
    pub data: &'a [half::f16],
    pub rows: usize,
    pub cols: usize,
}

/// Zero-copy mutable view of a f16 tensor borrowed from the caller.
#[derive(Debug)]
pub struct TensorViewMut<'a> {
    pub data: &'a mut [half::f16],
    pub rows: usize,
    pub cols: usize,
}

/// Abstraction over GPU-resident storage for expert weight buffers.
///
/// `GpuResident` (host bytes) implements this returning `None` for
/// `as_wgpu_buffer`.  `VramExpertEntry` (fully promoted) returns `Some`.
/// A future CUDA backend would add a third implementor wrapping a device
/// pointer opaquely without leaking it here.
pub trait GpuStorage: Send + Sync + 'static {
    /// Total byte length of the weight payload.
    fn byte_len(&self) -> usize;
    /// VRAM buffer handle, if this storage is device-resident.
    /// Returns `None` for host-only (CPU-tier) storage.
    fn as_wgpu_buffer(&self) -> Option<&wgpu::Buffer>;
}

/// Minimal contract every math backend must satisfy.
pub trait Backend: Send + Sync + 'static {
    fn device_name(&self) -> &str;
    fn is_gpu(&self) -> bool {
        false
    }
    fn matmul_into(&self, a: TensorView, b: TensorView, out: &mut TensorViewMut) -> Result<()>;
    fn swiglu_into(&self, gate: TensorView, up: TensorView, out: &mut TensorViewMut) -> Result<()>;
    fn softmax(&self, x: &mut TensorViewMut) -> Result<()>;
    fn kv_cache_insert(
        &self,
        layer: usize,
        position: usize,
        k: TensorView,
        v: TensorView,
    ) -> Result<()>;
    fn kv_attend(
        &self,
        layer: usize,
        q: TensorView,
        seq_len: usize,
        out: &mut TensorViewMut,
    ) -> Result<()>;

    /// Execute one MoE expert FFN from VRAM when the expert is GPU-resident,
    /// or fall back to the CPU path. On the GPU path the weight bytes are
    /// already in VRAM and no PCIe upload is needed.
    ///
    /// `x`       : hidden state input  [d_model]
    /// `d_model` : hidden dimension
    /// `d_ff`    : FFN intermediate dimension
    /// `out`     : output buffer        [d_model]
    fn expert_matmul(
        &self,
        layer_idx: usize,
        expert_id: u32,
        x:        TensorView<'_>,
        d_model:  usize,
        d_ff:     usize,
        out:      &mut TensorViewMut<'_>,
    ) -> Result<()>;
}

// =====================================================================
// Push Constants structs (POD, 16 bytes max, byte-identical to WGSL)
// =====================================================================

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MatmulPushConstants {
    m: u32,
    n: u32,
    k: u32,
    /// Unused (zero) for the dense F32 `matmul_main` entry point. For
    /// the Q4_0 inline-dequant entry point (`matmul_q4_0_main`) this
    /// carries the projection's first-block index inside the packed
    /// expert weight buffer — see `wgpu_shaders/matmul_q4_0.wgsl`.
    w_block_off: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SwigluPushConstants {
    n_elements: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftmaxPushConstants {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AttentionPushConstants {
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    /// Offset of this layer's K slice in the KV buffer, in **f32
    /// elements** (not bytes — a byte offset would overflow u32 for
    /// deep models with large KV slices).
    layer_offset: u32,
}

// =====================================================================
// GPU VRAM KV Cache
// =====================================================================

pub struct GpuKvCache {
    pub buffer: wgpu::Buffer,
    pub num_layers: usize,
    pub max_seq_len: usize,
    pub kv_dim: usize,
}

impl GpuKvCache {
    pub fn offset_bytes(&self, layer: usize, kv: usize, seq_pos: usize) -> u64 {
        let idx = ((layer * 2 + kv) * self.max_seq_len + seq_pos) * self.kv_dim;
        (idx * 4) as u64
    }
}

// =====================================================================
// GPU Backend using wgpu
// =====================================================================

/// How the bytes inside a [`VramExpertEntry`] weight buffer are encoded,
/// and therefore which matmul pipeline the FFN passes must dispatch.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum VramWeightLayout {
    /// Dense little-endian f32: `[gate | up | down]`, each projection
    /// `d_ff × d_model × 4` bytes. Bind groups slice the buffer per
    /// projection; `matmul_main` consumes it.
    F32,
    /// Native GGUF Q4_0 blocks (18 bytes / 32 weights), gate, up and
    /// down concatenated back-to-back with no padding. The whole
    /// buffer is bound at offset 0 (18-byte blocks cannot honour
    /// storage-offset alignment), and each pass selects its projection
    /// via the `w_block_off` push constant; `matmul_q4_0_main`
    /// dequantises inline.
    Q4_0,
}

/// A fully-initialized VRAM expert: weight buffer + pre-built bind groups
/// for the four FFN dispatch passes. Created once per expert on first
/// promotion; reused on every subsequent token.
struct VramExpertEntry {
    /// Raw weight buffer in VRAM. Layout: [gate_proj | up_proj | down_proj],
    /// either dense f32 LE or packed Q4_0 blocks — see [`VramWeightLayout`].
    /// gate_proj: [d_ff, d_model], up_proj: [d_ff, d_model], down_proj: [d_model, d_ff].
    weight_buf: wgpu::Buffer,
    /// Bind group for Pass 1: gate matmul.
    /// Bindings: [weight_buf@gate_slice (read), work_a/x (read), work_mid_1 (write)]
    gate_bg:   wgpu::BindGroup,
    /// Bind group for Pass 2: up matmul.
    /// Bindings: [weight_buf@up_slice (read), work_a/x (read), work_mid_2 (write)]
    up_bg:     wgpu::BindGroup,
    /// Bind group for Pass 3: SwiGLU.
    /// Bindings: [work_mid_1/gate (read), work_mid_2/up (read), work_out/swiglu_out (write)]
    swiglu_bg: wgpu::BindGroup,
    /// Bind group for Pass 4: down matmul.
    /// Bindings: [weight_buf@down_slice (read), work_out/swiglu_out (read), work_mid_1/final_out (write)]
    down_bg:   wgpu::BindGroup,
    /// Cached shape parameters.
    d_model:   usize,
    d_ff:      usize,
    /// Weight encoding → which matmul pipeline the passes dispatch.
    layout:    VramWeightLayout,
    /// First-block index of the up projection (Q4_0 layout only; 0 for F32).
    up_block_off:   u32,
    /// First-block index of the down projection (Q4_0 layout only; 0 for F32).
    down_block_off: u32,
}

impl GpuStorage for VramExpertEntry {
    fn byte_len(&self) -> usize {
        self.weight_buf.size() as usize
    }
    fn as_wgpu_buffer(&self) -> Option<&wgpu::Buffer> {
        Some(&self.weight_buf)
    }
}

pub struct GpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    device_name: String,

    matmul_pipeline: wgpu::ComputePipeline,
    /// Q4_0 inline-dequant GEMV pipeline (`matmul_q4_0.wgsl`) for expert
    /// FFN passes whose weights stay in native GGUF Q4_0 blocks in VRAM.
    matmul_q4_0_pipeline: wgpu::ComputePipeline,
    swiglu_pipeline: wgpu::ComputePipeline,
    softmax_pipeline: wgpu::ComputePipeline,
    attention_pipeline: wgpu::ComputePipeline,

    work_a: wgpu::Buffer,
    work_b: wgpu::Buffer,
    work_out: wgpu::Buffer,
    /// Pre-allocated intermediate [d_ff] buffer for the expert gate projection output.
    work_mid_1: wgpu::Buffer,
    /// Pre-allocated intermediate [d_ff] buffer for the expert up projection output.
    work_mid_2: wgpu::Buffer,

    _staging_up: wgpu::Buffer,
    staging_dn: wgpu::Buffer,

    kv_cache: GpuKvCache,

    matmul_bind_group: wgpu::BindGroup,
    swiglu_bind_group: wgpu::BindGroup,
    softmax_bind_group: wgpu::BindGroup,
    attention_bind_group: wgpu::BindGroup,

    /// f16→f32 staging scratch for buffer uploads. A real `Mutex` (not
    /// `UnsafeCell`) so concurrent callers — e.g. two Tokio tasks that
    /// share the same `Arc<GpuBackend>` when the batch scheduler fuses
    /// requests — serialize instead of aliasing `&mut` into the same
    /// buffer. The lock only covers a host-side convert + `write_buffer`
    /// (microseconds), so contention is negligible.
    conversion_scratch: ParkingMutex<Vec<f32>>,

    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,

    /// VRAM buffers for hot experts. Keyed by expert_id. Populated lazily
    /// on first access after GpuExpertCache promotes an expert.
    vram_expert_bufs: ParkingMutex<std::collections::HashMap<u32, VramExpertEntry>>,

    /// Reference to the VRAM expert cache. Used to check whether an expert
    /// is GPU-resident before falling back to the NVMe → CPU path.
    gpu_expert_cache: Arc<crate::expert_cache::GpuExpertCache>,
    /// Serializes the expert FFN path, which reuses shared staging buffers.
    expert_execution_lock: ParkingMutex<()>,
}

impl GpuBackend {
    pub async fn try_new(
        num_layers: usize,
        max_seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        gpu_expert_cache: Arc<crate::expert_cache::GpuExpertCache>,
    ) -> Result<Self> {
        // GQA models have num_kv_heads < num_heads; 0 means MHA.
        let num_kv_heads = if num_kv_heads == 0 { num_heads } else { num_kv_heads };
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| anyhow!("No HighPerformance adapter found"))?;

        let info = adapter.get_info();
        let device_name = format!("wgpu-{:?}-{}", info.backend, info.name);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("MER-GpuBackend"),
                    required_features: wgpu::Features::PUSH_CONSTANTS,
                    required_limits: wgpu::Limits {
                        max_push_constant_size: 32,
                        ..wgpu::Limits::default()
                    },
                },
                None,
            )
            .await?;

        // Compile modules
        let matmul_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul_shader"),
            source: wgpu::ShaderSource::Wgsl(MATMUL_SHADER.into()),
        });

        let matmul_q4_0_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("matmul_q4_0_shader"),
            source: wgpu::ShaderSource::Wgsl(MATMUL_Q4_0_SHADER.into()),
        });

        let swiglu_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("swiglu_shader"),
            source: wgpu::ShaderSource::Wgsl(SWIGLU_SHADER.into()),
        });

        let softmax_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("softmax_shader"),
            source: wgpu::ShaderSource::Wgsl(SOFTMAX_SHADER.into()),
        });

        // Compile attention shader, injecting dynamic MAX_SEQ_LEN
        let attention_src = ATTENTION_SHADER.replace(
            "const MAX_SEQ_LEN: u32 = 4096u;",
            &format!("const MAX_SEQ_LEN: u32 = {}u;", max_seq_len),
        ).replace(
            "const MAX_HEAD_DIM: u32 = 256u;",
            &format!("const MAX_HEAD_DIM: u32 = {}u;", head_dim),
        );
        let attention_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("attention_shader"),
            source: wgpu::ShaderSource::Wgsl(attention_src.into()),
        });

        // Setup layouts manually for pipelines since push constants are used
        let layout_3_buffers = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("layout_3_buffers"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
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
            ],
        });

        let layout_1_buffer = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("layout_1_buffer"),
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
            ],
        });

        let matmul_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("matmul_pipeline_layout"),
            bind_group_layouts: &[&layout_3_buffers],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..16,
            }],
        });

        let swiglu_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("swiglu_pipeline_layout"),
            bind_group_layouts: &[&layout_3_buffers],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..16,
            }],
        });

        let softmax_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("softmax_pipeline_layout"),
            bind_group_layouts: &[&layout_1_buffer],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..16,
            }],
        });

        let attention_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("attention_pipeline_layout"),
            bind_group_layouts: &[&layout_3_buffers],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                // 5 × u32 = 20 bytes; padded to the 32-byte limit
                // requested in `required_limits`.
                range: 0..32,
            }],
        });

        // Compute pipelines
        let matmul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matmul_pipeline"),
            layout: Some(&matmul_pipeline_layout),
            module: &matmul_module,
            entry_point: "matmul_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        // Same bind-group shape (read, read, read-write) and the same
        // 16-byte push-constant block as the dense pipeline, so the
        // pipeline layout is shared; only the module/entry differ.
        let matmul_q4_0_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matmul_q4_0_pipeline"),
            layout: Some(&matmul_pipeline_layout),
            module: &matmul_q4_0_module,
            entry_point: "matmul_q4_0_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let swiglu_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("swiglu_pipeline"),
            layout: Some(&swiglu_pipeline_layout),
            module: &swiglu_module,
            entry_point: "swiglu_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let softmax_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("softmax_pipeline"),
            layout: Some(&softmax_pipeline_layout),
            module: &softmax_module,
            entry_point: "softmax_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let attention_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("attention_pipeline"),
            layout: Some(&attention_pipeline_layout),
            module: &attention_module,
            entry_point: "attention_main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        // Pre-allocated buffers
        const MAX_ELEM: usize = 4096 * 4096;
        let work_a = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("work_a"),
            size: (MAX_ELEM * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let work_b = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("work_b"),
            size: (MAX_ELEM * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let work_out = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("work_out"),
            size: (MAX_ELEM * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let work_mid_1 = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("work_mid_1"),
            size:               (MAX_EXPERT_D_FF * 4) as u64,
            usage:              wgpu::BufferUsages::STORAGE
                              | wgpu::BufferUsages::COPY_DST
                              | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let work_mid_2 = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("work_mid_2"),
            size:               (MAX_EXPERT_D_FF * 4) as u64,
            usage:              wgpu::BufferUsages::STORAGE
                              | wgpu::BufferUsages::COPY_DST
                              | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_up = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_up"),
            size: (MAX_ELEM * 4) as u64,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_dn = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_dn"),
            size: (MAX_ELEM * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // KV cache — stride is the *KV* width (num_kv_heads × head_dim),
        // which is narrower than the query width for GQA models.
        let kv_dim = num_kv_heads * head_dim;
        let kv_cache_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("kv_cache"),
            size: (num_layers * 2 * max_seq_len * kv_dim * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let kv_cache = GpuKvCache {
            buffer: kv_cache_buffer,
            num_layers,
            max_seq_len,
            kv_dim,
        };

        // Pre-build bind groups
        let matmul_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_bind_group"),
            layout: &layout_3_buffers,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: work_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: work_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: work_out.as_entire_binding() },
            ],
        });

        let swiglu_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("swiglu_bind_group"),
            layout: &layout_3_buffers,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: work_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: work_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: work_out.as_entire_binding() },
            ],
        });

        let softmax_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("softmax_bind_group"),
            layout: &layout_1_buffer,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: work_a.as_entire_binding() },
            ],
        });

        let attention_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("attention_bind_group"),
            layout: &layout_3_buffers,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: work_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: kv_cache.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: work_out.as_entire_binding() },
            ],
        });

        let conversion_scratch = ParkingMutex::new(vec![0.0f32; MAX_ELEM]);

        Ok(Self {
            device,
            queue,
            device_name,
            matmul_pipeline,
            matmul_q4_0_pipeline,
            swiglu_pipeline,
            softmax_pipeline,
            attention_pipeline,
            work_a,
            work_b,
            work_out,
            work_mid_1,
            work_mid_2,
            _staging_up: staging_up,
            staging_dn,
            kv_cache,
            matmul_bind_group,
            swiglu_bind_group,
            softmax_bind_group,
            attention_bind_group,
            conversion_scratch,
            num_heads,
            num_kv_heads,
            head_dim,
            vram_expert_bufs: ParkingMutex::new(std::collections::HashMap::new()),
            gpu_expert_cache,
            expert_execution_lock: ParkingMutex::new(()),
        })
    }

    /// Upload expert weight bytes to VRAM and pre-build the four bind groups
    /// needed for expert FFN dispatch.
    ///
    /// Weight layout (verify against `dispatch_expert_forward` before shipping):
    ///   gate_proj bytes: [0,                   d_ff * d_model * 4)
    ///   up_proj bytes:   [d_ff * d_model * 4,  2 * d_ff * d_model * 4)
    ///   down_proj bytes: [2 * d_ff * d_model * 4, 3 * d_ff * d_model * 4)
    /// All weights are f32 little-endian.
    fn build_expert_entry(
        &self,
        weight_bytes: &[u8],
        d_model:      usize,
        d_ff:         usize,
    ) -> anyhow::Result<VramExpertEntry> {
        use std::num::NonZeroU64;

        let proj_bytes = d_ff * d_model * 4;  // bytes per projection matrix
        anyhow::ensure!(
            proj_bytes > 0,
            "invalid expert shape: d_ff={} d_model={} produces zero-byte projections",
            d_ff,
            d_model
        );
        anyhow::ensure!(
            weight_bytes.len() >= 3 * proj_bytes,
            "expert weight buffer too small: got {} bytes, need {} (3 × d_ff={} × d_model={} × 4)",
            weight_bytes.len(), 3 * proj_bytes, d_ff, d_model
        );
        anyhow::ensure!(
            d_ff <= MAX_EXPERT_D_FF,
            "d_ff={} exceeds MAX_EXPERT_D_FF={}; increase the constant",
            d_ff, MAX_EXPERT_D_FF
        );

        // ── Upload weights to VRAM ────────────────────────────────────────────
        let weight_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("vram_expert_weights"),
            size:               weight_bytes.len() as u64,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&weight_buf, 0, weight_bytes);

        // ── Sub-range offsets ─────────────────────────────────────────────────
        let gate_offset  = 0u64;
        let up_offset    = proj_bytes as u64;
        let down_offset  = 2 * proj_bytes as u64;
        let min_align = (self.device.limits().min_storage_buffer_offset_alignment as u64).max(1);
        anyhow::ensure!(
            up_offset % min_align == 0,
            "expert projection slice offset {} is not aligned to min_storage_buffer_offset_alignment={}",
            up_offset,
            min_align
        );
        anyhow::ensure!(
            down_offset % min_align == 0,
            "expert projection slice offset {} is not aligned to min_storage_buffer_offset_alignment={}",
            down_offset,
            min_align
        );
        let slice_size   = NonZeroU64::new(proj_bytes as u64).unwrap();

        // ── Bind group layout (3 buffers: read, read, read-write) ─────────────
        // Reuse the pipeline's auto-layout by pulling it from the existing pipeline.
        let matmul_bgl  = self.matmul_pipeline.get_bind_group_layout(0);
        let swiglu_bgl  = self.swiglu_pipeline.get_bind_group_layout(0);

        // ── Pass 1: gate matmul — weight_buf[gate] × x → work_mid_1 ─────────
        let gate_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("expert_gate_bg"),
            layout:  &matmul_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &weight_buf,
                        offset: gate_offset,
                        size:   Some(slice_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: self.work_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  2,
                    resource: self.work_mid_1.as_entire_binding(),
                },
            ],
        });

        // ── Pass 2: up matmul — weight_buf[up] × x → work_mid_2 ─────────────
        let up_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("expert_up_bg"),
            layout:  &matmul_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &weight_buf,
                        offset: up_offset,
                        size:   Some(slice_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: self.work_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  2,
                    resource: self.work_mid_2.as_entire_binding(),
                },
            ],
        });

        // ── Pass 3: SwiGLU — work_mid_1, work_mid_2 → work_out ──────────────
        let swiglu_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("expert_swiglu_bg"),
            layout:  &swiglu_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: self.work_mid_1.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: self.work_mid_2.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  2,
                    resource: self.work_out.as_entire_binding(),
                },
            ],
        });

        // ── Pass 4: down matmul — weight_buf[down] × work_out → work_mid_1 ──
        let down_slice_size = NonZeroU64::new(proj_bytes as u64).unwrap(); // same shape
        let down_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("expert_down_bg"),
            layout:  &matmul_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &weight_buf,
                        offset: down_offset,
                        size:   Some(down_slice_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: self.work_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  2,
                    resource: self.work_mid_1.as_entire_binding(),
                },
            ],
        });

        Ok(VramExpertEntry {
            weight_buf,
            gate_bg,
            up_bg,
            swiglu_bg,
            down_bg,
            d_model,
            d_ff,
            layout: VramWeightLayout::F32,
            up_block_off: 0,
            down_block_off: 0,
        })
    }

    /// Upload a **native Q4_0** expert weight buffer to VRAM and pre-build
    /// the four FFN bind groups against the inline-dequant pipeline
    /// (`matmul_q4_0.wgsl`). Unlike [`Self::build_expert_entry`] the bytes
    /// are *not* dequantised first: the GGUF Q4_0 blocks (18 bytes per 32
    /// weights) cross PCIe and live in VRAM as-is — ~8× fewer bytes than
    /// the dense F32 stream — and each compute pass unpacks blocks inline.
    ///
    /// Expected layout (matching `OwnedExpertWeights::from_bytes_q4_0`):
    /// gate, up and down block streams concatenated back-to-back, each
    /// `(d_ff·d_model / 32) × 18` bytes. Both `d_model` and `d_ff` must be
    /// multiples of the 32-element Q4_0 block (the caller guarantees this
    /// via `Engine::gpu_eligible_dtype`), so every matrix row starts on a
    /// block boundary. Buffers short by at most one page are zero-padded,
    /// mirroring the CPU loader's `q4_expert_bytes_with_tolerance`.
    fn build_expert_entry_q4_0(
        &self,
        weight_bytes: &[u8],
        d_model:      usize,
        d_ff:         usize,
    ) -> anyhow::Result<VramExpertEntry> {
        use crate::inference::{Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMS};

        anyhow::ensure!(
            d_model > 0 && d_ff > 0,
            "invalid expert shape: d_ff={} d_model={}",
            d_ff, d_model
        );
        anyhow::ensure!(
            d_model % Q4_0_BLOCK_ELEMS == 0 && d_ff % Q4_0_BLOCK_ELEMS == 0,
            "Q4_0 GPU path requires block-aligned dims: d_model={} d_ff={} (block={})",
            d_model, d_ff, Q4_0_BLOCK_ELEMS
        );
        anyhow::ensure!(
            d_ff <= MAX_EXPERT_D_FF,
            "d_ff={} exceeds MAX_EXPERT_D_FF={}; increase the constant",
            d_ff, MAX_EXPERT_D_FF
        );

        let blocks_per_proj = (d_ff * d_model) / Q4_0_BLOCK_ELEMS;
        let proj_bytes = blocks_per_proj * Q4_0_BLOCK_BYTES;
        let need = 3 * proj_bytes;
        let tol = crate::inference::EXPERT_SIZE_TOLERANCE_BYTES;
        anyhow::ensure!(
            weight_bytes.len() >= need
                || (need > tol && need - weight_bytes.len() <= tol),
            "Q4_0 expert weight buffer too small: got {} bytes, need {} (3 × {} blocks × {}B)",
            weight_bytes.len(), need, blocks_per_proj, Q4_0_BLOCK_BYTES
        );
        anyhow::ensure!(
            2 * blocks_per_proj <= u32::MAX as usize,
            "Q4_0 expert block count {} exceeds u32 push-constant range",
            2 * blocks_per_proj
        );

        // wgpu requires buffer sizes / write lengths to be 4-byte
        // multiples; `need` is only guaranteed even (18-byte blocks), so
        // round up and zero-fill the tail (also covers the ≤ one-page
        // shortfall tolerance above).
        let padded_len = need.div_ceil(4) * 4;
        let weight_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("vram_expert_weights_q4_0"),
            size:               padded_len as u64,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let avail = weight_bytes.len().min(need);
        if avail == padded_len {
            self.queue.write_buffer(&weight_buf, 0, &weight_bytes[..need]);
        } else {
            let mut padded = Vec::with_capacity(padded_len);
            padded.extend_from_slice(&weight_bytes[..avail]);
            padded.resize(padded_len, 0);
            self.queue.write_buffer(&weight_buf, 0, &padded);
        }

        // The projection base is selected via the `w_block_off` push
        // constant (Q4_0 blocks are 18 bytes and cannot honour
        // min_storage_buffer_offset_alignment), so every matmul pass
        // binds the entire weight buffer.
        let matmul_bgl = self.matmul_q4_0_pipeline.get_bind_group_layout(0);
        let swiglu_bgl = self.swiglu_pipeline.get_bind_group_layout(0);

        let make_matmul_bg = |label: &str, x_buf: &wgpu::Buffer, out_buf: &wgpu::Buffer| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label:   Some(label),
                layout:  &matmul_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: weight_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: x_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: out_buf.as_entire_binding() },
                ],
            })
        };

        // Pass 1: gate GEMV — W[gate blocks] × x → work_mid_1.
        let gate_bg = make_matmul_bg("expert_gate_bg_q4_0", &self.work_a, &self.work_mid_1);
        // Pass 2: up GEMV — W[up blocks] × x → work_mid_2.
        let up_bg = make_matmul_bg("expert_up_bg_q4_0", &self.work_a, &self.work_mid_2);
        // Pass 3: SwiGLU — work_mid_1, work_mid_2 → work_out (dense, unchanged).
        let swiglu_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("expert_swiglu_bg_q4_0"),
            layout:  &swiglu_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.work_mid_1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.work_mid_2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.work_out.as_entire_binding() },
            ],
        });
        // Pass 4: down GEMV — W[down blocks] × work_out → work_mid_1.
        let down_bg = make_matmul_bg("expert_down_bg_q4_0", &self.work_out, &self.work_mid_1);

        Ok(VramExpertEntry {
            weight_buf,
            gate_bg,
            up_bg,
            swiglu_bg,
            down_bg,
            d_model,
            d_ff,
            layout: VramWeightLayout::Q4_0,
            up_block_off: blocks_per_proj as u32,
            down_block_off: (2 * blocks_per_proj) as u32,
        })
    }

    /// Dispatch a SwiGLU expert FFN where the weight buffer is already
    /// VRAM-resident. Uploads only `x` (hidden state, ~8 KB); the weights
    /// never cross PCIe.
    ///
    /// The weight layout assumed is `[gate_proj || up_proj || down_proj]`
    /// matching `ExpertWeights::from_bytes` / the SwiGLU forward convention.
    fn expert_matmul_from_vram(
        &self,
        entry: &VramExpertEntry,
        x:     TensorView<'_>,
        out:   &mut TensorViewMut<'_>,
    ) -> Result<()> {
        let _exec_guard = self.expert_execution_lock.lock();
        let d_model = entry.d_model;
        let d_ff    = entry.d_ff;

        debug_assert_eq!(x.data.len(),   d_model);
        debug_assert_eq!(out.data.len(), d_model);

        // ── Upload x to work_a ────────────────────────────────────────────────
        let mut scratch = self.conversion_scratch.lock();
        assert!(d_model <= scratch.len());
        for i in 0..d_model {
            scratch[i] = x.data[i].to_f32();
        }
        self.queue.write_buffer(&self.work_a, 0, bytemuck::cast_slice(&scratch[..d_model]));

        // ── Single command buffer: 4 sequential compute passes ───────────────
        // The GPU executes these in order; no host-side synchronization needed
        // between passes. One submit = one PCIe round-trip for the whole FFN.
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("expert_ffn_encoder"),
        });

        // Pass 1: gate_proj × x → work_mid_1   (M=d_ff, K=d_model, N=1)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label:            Some("expert_gate_pass"),
                timestamp_writes: None,
            });
            match entry.layout {
                VramWeightLayout::F32 => {
                    cpass.set_pipeline(&self.matmul_pipeline);
                    cpass.set_bind_group(0, &entry.gate_bg, &[]);
                    cpass.set_push_constants(0, bytemuck::bytes_of(&MatmulPushConstants {
                        m: d_ff as u32, n: 1, k: d_model as u32, w_block_off: 0,
                    }));
                    cpass.dispatch_workgroups(1, (d_ff as u32 + 15) / 16, 1);
                }
                VramWeightLayout::Q4_0 => {
                    cpass.set_pipeline(&self.matmul_q4_0_pipeline);
                    cpass.set_bind_group(0, &entry.gate_bg, &[]);
                    cpass.set_push_constants(0, bytemuck::bytes_of(&MatmulPushConstants {
                        m: d_ff as u32, n: 1, k: d_model as u32, w_block_off: 0,
                    }));
                    cpass.dispatch_workgroups((d_ff as u32 + 63) / 64, 1, 1);
                }
            }
        }

        // Pass 2: up_proj × x → work_mid_2   (M=d_ff, K=d_model, N=1)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label:            Some("expert_up_pass"),
                timestamp_writes: None,
            });
            match entry.layout {
                VramWeightLayout::F32 => {
                    cpass.set_pipeline(&self.matmul_pipeline);
                    cpass.set_bind_group(0, &entry.up_bg, &[]);
                    cpass.set_push_constants(0, bytemuck::bytes_of(&MatmulPushConstants {
                        m: d_ff as u32, n: 1, k: d_model as u32, w_block_off: 0,
                    }));
                    cpass.dispatch_workgroups(1, (d_ff as u32 + 15) / 16, 1);
                }
                VramWeightLayout::Q4_0 => {
                    cpass.set_pipeline(&self.matmul_q4_0_pipeline);
                    cpass.set_bind_group(0, &entry.up_bg, &[]);
                    cpass.set_push_constants(0, bytemuck::bytes_of(&MatmulPushConstants {
                        m: d_ff as u32, n: 1, k: d_model as u32,
                        w_block_off: entry.up_block_off,
                    }));
                    cpass.dispatch_workgroups((d_ff as u32 + 63) / 64, 1, 1);
                }
            }
        }

        // Pass 3: SwiGLU(work_mid_1, work_mid_2) → work_out   (n_elements=d_ff)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label:            Some("expert_swiglu_pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.swiglu_pipeline);
            cpass.set_bind_group(0, &entry.swiglu_bg, &[]);
            cpass.set_push_constants(0, bytemuck::bytes_of(&SwigluPushConstants {
                n_elements: d_ff as u32, _pad0: 0, _pad1: 0, _pad2: 0,
            }));
            let wg_x = (d_ff as u32 + 255) / 256;
            cpass.dispatch_workgroups(wg_x, 1, 1);
        }

        // Pass 4: down_proj × work_out → work_mid_1   (M=d_model, K=d_ff, N=1)
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label:            Some("expert_down_pass"),
                timestamp_writes: None,
            });
            match entry.layout {
                VramWeightLayout::F32 => {
                    cpass.set_pipeline(&self.matmul_pipeline);
                    cpass.set_bind_group(0, &entry.down_bg, &[]);
                    cpass.set_push_constants(0, bytemuck::bytes_of(&MatmulPushConstants {
                        m: d_model as u32, n: 1, k: d_ff as u32, w_block_off: 0,
                    }));
                    cpass.dispatch_workgroups(1, (d_model as u32 + 15) / 16, 1);
                }
                VramWeightLayout::Q4_0 => {
                    cpass.set_pipeline(&self.matmul_q4_0_pipeline);
                    cpass.set_bind_group(0, &entry.down_bg, &[]);
                    cpass.set_push_constants(0, bytemuck::bytes_of(&MatmulPushConstants {
                        m: d_model as u32, n: 1, k: d_ff as u32,
                        w_block_off: entry.down_block_off,
                    }));
                    cpass.dispatch_workgroups((d_model as u32 + 63) / 64, 1, 1);
                }
            }
        }

        // ── Readback work_mid_1 → out ─────────────────────────────────────────
        let out_bytes = (d_model * 4) as u64;
        encoder.copy_buffer_to_buffer(&self.work_mid_1, 0, &self.staging_dn, 0, out_bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = self.staging_dn.slice(0..out_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| { let _ = tx.send(res); });
        self.device.poll(wgpu::Maintain::Wait);

        rx.recv()
            .map_err(|e| anyhow::anyhow!("channel error on expert readback: {e:?}"))?
            .map_err(|e| anyhow::anyhow!("buffer map error on expert readback: {e:?}"))?;

        {
            let view   = slice.get_mapped_range();
            let floats: &[f32] = bytemuck::cast_slice(&view);
            for i in 0..d_model {
                out.data[i] = half::f16::from_f32(floats[i]);
            }
        }
        self.staging_dn.unmap();
        Ok(())
    }
}

impl Backend for GpuBackend {
    fn device_name(&self) -> &str {
        &self.device_name
    }

    fn is_gpu(&self) -> bool {
        true
    }

    fn matmul_into(&self, a: TensorView, b: TensorView, out: &mut TensorViewMut) -> Result<()> {
        let a_len = a.data.len();
        let b_len = b.data.len();
        let out_len = out.rows * out.cols;

        let mut scratch = self.conversion_scratch.lock();
        assert!(a_len <= scratch.len());
        assert!(b_len <= scratch.len());
        assert!(out_len <= scratch.len());

        // Upload A
        for i in 0..a_len {
            scratch[i] = a.data[i].to_f32();
        }
        self.queue.write_buffer(&self.work_a, 0, bytemuck::cast_slice(&scratch[..a_len]));

        // Upload B
        for i in 0..b_len {
            scratch[i] = b.data[i].to_f32();
        }
        self.queue.write_buffer(&self.work_b, 0, bytemuck::cast_slice(&scratch[..b_len]));

        // Dispatch
        let pcs = MatmulPushConstants {
            m: a.rows as u32,
            n: b.cols as u32,
            k: a.cols as u32,
            w_block_off: 0,
        };

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("matmul_encoder"),
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.matmul_pipeline);
            compute_pass.set_bind_group(0, &self.matmul_bind_group, &[]);
            compute_pass.set_push_constants(0, bytemuck::bytes_of(&pcs));
            compute_pass.dispatch_workgroups(
                (b.cols as u32 + 15) / 16,
                (a.rows as u32 + 15) / 16,
                1,
            );
        }

        // Readback
        let out_bytes = (out_len * 4) as u64;
        encoder.copy_buffer_to_buffer(&self.work_out, 0, &self.staging_dn, 0, out_bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = self.staging_dn.slice(0..out_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);

        rx.recv()
            .map_err(|e| anyhow!("Channel error on GPU readback: {:?}", e))?
            .map_err(|e| anyhow!("Buffer map error on GPU readback: {:?}", e))?;

        {
            let view = slice.get_mapped_range();
            let floats: &[f32] = bytemuck::cast_slice(&view);
            for i in 0..out_len {
                out.data[i] = half::f16::from_f32(floats[i]);
            }
        }
        self.staging_dn.unmap();
        Ok(())
    }

    fn swiglu_into(&self, gate: TensorView, up: TensorView, out: &mut TensorViewMut) -> Result<()> {
        let len = gate.data.len();
        let out_len = out.rows * out.cols;
        assert_eq!(up.data.len(), len);
        assert_eq!(out_len, len);

        let mut scratch = self.conversion_scratch.lock();
        assert!(len <= scratch.len());

        // Upload gate
        for i in 0..len {
            scratch[i] = gate.data[i].to_f32();
        }
        self.queue.write_buffer(&self.work_a, 0, bytemuck::cast_slice(&scratch[..len]));

        // Upload up
        for i in 0..len {
            scratch[i] = up.data[i].to_f32();
        }
        self.queue.write_buffer(&self.work_b, 0, bytemuck::cast_slice(&scratch[..len]));

        // Dispatch
        let pcs = SwigluPushConstants {
            n_elements: len as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("swiglu_encoder"),
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("swiglu_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.swiglu_pipeline);
            compute_pass.set_bind_group(0, &self.swiglu_bind_group, &[]);
            compute_pass.set_push_constants(0, bytemuck::bytes_of(&pcs));
            compute_pass.dispatch_workgroups((len as u32 + 255) / 256, 1, 1);
        }

        // Readback
        let out_bytes = (len * 4) as u64;
        encoder.copy_buffer_to_buffer(&self.work_out, 0, &self.staging_dn, 0, out_bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = self.staging_dn.slice(0..out_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);

        rx.recv()
            .map_err(|e| anyhow!("Channel error on GPU readback: {:?}", e))?
            .map_err(|e| anyhow!("Buffer map error on GPU readback: {:?}", e))?;

        {
            let view = slice.get_mapped_range();
            let floats: &[f32] = bytemuck::cast_slice(&view);
            for i in 0..len {
                out.data[i] = half::f16::from_f32(floats[i]);
            }
        }
        self.staging_dn.unmap();
        Ok(())
    }

    fn softmax(&self, x: &mut TensorViewMut) -> Result<()> {
        let len = x.data.len();

        let mut scratch = self.conversion_scratch.lock();
        assert!(len <= scratch.len());

        // Upload x
        for i in 0..len {
            scratch[i] = x.data[i].to_f32();
        }
        self.queue.write_buffer(&self.work_a, 0, bytemuck::cast_slice(&scratch[..len]));

        // Dispatch
        let pcs = SoftmaxPushConstants {
            rows: x.rows as u32,
            cols: x.cols as u32,
            _pad0: 0,
            _pad1: 0,
        };

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("softmax_encoder"),
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("softmax_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.softmax_pipeline);
            compute_pass.set_bind_group(0, &self.softmax_bind_group, &[]);
            compute_pass.set_push_constants(0, bytemuck::bytes_of(&pcs));
            compute_pass.dispatch_workgroups(x.rows as u32, 1, 1);
        }

        // Readback from work_a (in-place)
        let out_bytes = (len * 4) as u64;
        encoder.copy_buffer_to_buffer(&self.work_a, 0, &self.staging_dn, 0, out_bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = self.staging_dn.slice(0..out_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);

        rx.recv()
            .map_err(|e| anyhow!("Channel error on GPU readback: {:?}", e))?
            .map_err(|e| anyhow!("Buffer map error on GPU readback: {:?}", e))?;

        {
            let view = slice.get_mapped_range();
            let floats: &[f32] = bytemuck::cast_slice(&view);
            for i in 0..len {
                x.data[i] = half::f16::from_f32(floats[i]);
            }
        }
        self.staging_dn.unmap();
        Ok(())
    }

    fn kv_cache_insert(
        &self,
        layer: usize,
        position: usize,
        k: TensorView,
        v: TensorView,
    ) -> Result<()> {
        let k_len = k.data.len();
        let v_len = v.data.len();

        let mut scratch = self.conversion_scratch.lock();
        assert!(k_len <= scratch.len());
        assert!(v_len <= scratch.len());

        // Upload K
        for i in 0..k_len {
            scratch[i] = k.data[i].to_f32();
        }
        let offset_k = self.kv_cache.offset_bytes(layer, 0, position);
        self.queue.write_buffer(
            &self.kv_cache.buffer,
            offset_k,
            bytemuck::cast_slice(&scratch[..k_len]),
        );

        // Upload V
        for i in 0..v_len {
            scratch[i] = v.data[i].to_f32();
        }
        let offset_v = self.kv_cache.offset_bytes(layer, 1, position);
        self.queue.write_buffer(
            &self.kv_cache.buffer,
            offset_v,
            bytemuck::cast_slice(&scratch[..v_len]),
        );

        Ok(())
    }

    fn kv_attend(
        &self,
        layer: usize,
        q: TensorView,
        seq_len: usize,
        out: &mut TensorViewMut,
    ) -> Result<()> {
        let q_len = q.data.len();
        let out_len = out.rows * out.cols;

        let mut scratch = self.conversion_scratch.lock();
        assert!(q_len <= scratch.len());
        assert!(out_len <= scratch.len());

        // Upload Q
        for i in 0..q_len {
            scratch[i] = q.data[i].to_f32();
        }
        self.queue.write_buffer(&self.work_a, 0, bytemuck::cast_slice(&scratch[..q_len]));

        // Dispatch
        // Pass the layer offset in f32 *elements*: a byte offset cast to
        // u32 silently wraps past 4 GiB for deep models with large KV
        // slices. Guard the (4× larger) element range explicitly.
        let layer_off_elems = self.kv_cache.offset_bytes(layer, 0, 0) / 4;
        if layer_off_elems > u32::MAX as u64 {
            return Err(anyhow!(
                "KV layer offset {layer_off_elems} elements exceeds u32 push-constant range"
            ));
        }
        let pcs = AttentionPushConstants {
            num_heads: self.num_heads as u32,
            num_kv_heads: self.num_kv_heads as u32,
            head_dim: self.head_dim as u32,
            seq_len: seq_len as u32,
            layer_offset: layer_off_elems as u32,
        };

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("attention_encoder"),
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("attention_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.attention_pipeline);
            compute_pass.set_bind_group(0, &self.attention_bind_group, &[]);
            compute_pass.set_push_constants(0, bytemuck::bytes_of(&pcs));
            compute_pass.dispatch_workgroups(self.num_heads as u32, 1, 1);
        }

        // Readback
        let out_bytes = (out_len * 4) as u64;
        encoder.copy_buffer_to_buffer(&self.work_out, 0, &self.staging_dn, 0, out_bytes);
        self.queue.submit(Some(encoder.finish()));

        let slice = self.staging_dn.slice(0..out_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);

        rx.recv()
            .map_err(|e| anyhow!("Channel error on GPU readback: {:?}", e))?
            .map_err(|e| anyhow!("Buffer map error on GPU readback: {:?}", e))?;

        {
            let view = slice.get_mapped_range();
            let floats: &[f32] = bytemuck::cast_slice(&view);
            for i in 0..out_len {
                out.data[i] = half::f16::from_f32(floats[i]);
            }
        }
        self.staging_dn.unmap();
        Ok(())
    }

    fn expert_matmul(
        &self,
        layer_idx: usize,
        expert_id: u32,
        x:        TensorView<'_>,
        d_model:  usize,
        d_ff:     usize,
        out:      &mut TensorViewMut<'_>,
    ) -> Result<()> {
        use crate::expert_cache::GpuLookup;
        let _ = layer_idx;

        // ── Fast path: expert already in VRAM with pre-built bind groups ──────
        {
            let bufs = self.vram_expert_bufs.lock();
            if let Some(entry) = bufs.get(&expert_id) {
                return self.expert_matmul_from_vram(entry, x, out);
            }
        }

        // Slow path: evict stale entries whose GpuExpertCache slot was reclaimed.
        self.vram_expert_bufs
            .lock()
            .retain(|id, _| self.gpu_expert_cache.contains(*id));

        // ── Promote from GpuExpertCache bytes → VRAM entry ────────────────────
        match self.gpu_expert_cache.get(expert_id) {
            GpuLookup::AnchorHit(r) | GpuLookup::LruHit(r) => {
                // Native Q4_0 residents go through the inline-dequant
                // pipeline; everything else is the dense F32 layout.
                let entry = match r.dtype() {
                    crate::inference::WeightDtype::Q4_0 => {
                        self.build_expert_entry_q4_0(r.data(), d_model, d_ff)?
                    }
                    _ => self.build_expert_entry(r.data(), d_model, d_ff)?,
                };
                let result = self.expert_matmul_from_vram(&entry, x, out);
                self.vram_expert_bufs.lock().insert(expert_id, entry);
                result
            }
            GpuLookup::Miss => {
                anyhow::bail!(
                    "expert {} not VRAM-resident; caller must fall back to CPU path",
                    expert_id
                )
            }
        }
    }
}

// =====================================================================
// Candle CPU Fallback Backend
// =====================================================================

#[derive(Clone, Default)]
pub struct CandleBackend;

impl CandleBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Backend for CandleBackend {
    fn device_name(&self) -> &str {
        "cpu-fallback"
    }

    fn is_gpu(&self) -> bool {
        false
    }

    fn matmul_into(&self, a: TensorView, b: TensorView, out: &mut TensorViewMut) -> Result<()> {
        let m = a.rows;
        let k = a.cols;
        let n = b.cols;
        assert_eq!(b.rows, k);
        assert_eq!(out.rows, m);
        assert_eq!(out.cols, n);

        for val in out.data.iter_mut() {
            *val = half::f16::ZERO;
        }

        let tile_size = 32;
        for i_outer in (0..m).step_by(tile_size) {
            let i_end = (i_outer + tile_size).min(m);
            for k_outer in (0..k).step_by(tile_size) {
                let k_end = (k_outer + tile_size).min(k);
                for j_outer in (0..n).step_by(tile_size) {
                    let j_end = (j_outer + tile_size).min(n);

                    for i in i_outer..i_end {
                        let out_row_offset = i * n;
                        for k_inner in k_outer..k_end {
                            let a_val = a.data[i * k + k_inner].to_f32();
                            if a_val == 0.0 {
                                continue;
                            }
                            let b_row_offset = k_inner * n;
                            for j in j_outer..j_end {
                                let b_val = b.data[b_row_offset + j].to_f32();
                                let out_idx = out_row_offset + j;
                                let cur = out.data[out_idx].to_f32();
                                out.data[out_idx] = half::f16::from_f32(cur + a_val * b_val);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn swiglu_into(&self, gate: TensorView, up: TensorView, out: &mut TensorViewMut) -> Result<()> {
        let len = gate.data.len();
        assert_eq!(up.data.len(), len);
        assert_eq!(out.data.len(), len);

        for i in 0..len {
            let g = gate.data[i].to_f32();
            let u = up.data[i].to_f32();
            let silu_g = g / (1.0 + (-g).exp());
            out.data[i] = half::f16::from_f32(silu_g * u);
        }
        Ok(())
    }

    fn softmax(&self, x: &mut TensorViewMut) -> Result<()> {
        let rows = x.rows;
        let cols = x.cols;
        for r in 0..rows {
            let row_slice = &mut x.data[r * cols..(r + 1) * cols];
            if row_slice.is_empty() {
                continue;
            }
            let mut maxv = f32::NEG_INFINITY;
            for &v in row_slice.iter() {
                let vf = v.to_f32();
                if vf > maxv {
                    maxv = vf;
                }
            }
            let mut sum = 0.0f32;
            for v in row_slice.iter_mut() {
                let vf = v.to_f32();
                let ev = (vf - maxv).exp();
                *v = half::f16::from_f32(ev);
                sum += ev;
            }
            if sum > 0.0 {
                for v in row_slice.iter_mut() {
                    *v = half::f16::from_f32(v.to_f32() / sum);
                }
            }
        }
        Ok(())
    }

    fn kv_cache_insert(
        &self,
        _layer: usize,
        _position: usize,
        _k: TensorView,
        _v: TensorView,
    ) -> Result<()> {
        // Managed on the CPU path directly in transformer.rs
        Ok(())
    }

    fn kv_attend(
        &self,
        _layer: usize,
        _q: TensorView,
        _seq_len: usize,
        _out: &mut TensorViewMut,
    ) -> Result<()> {
        // Managed on the CPU path directly in transformer.rs
        Ok(())
    }

    fn expert_matmul(
        &self,
        _layer_idx: usize,
        _expert_id: u32,
        _x:        TensorView<'_>,
        _d_model:  usize,
        _d_ff:     usize,
        _out:      &mut TensorViewMut<'_>,
    ) -> Result<()> {
        anyhow::bail!("expert_matmul should not be called on CPU backend; use direct NVMe streaming path instead")
    }
}

// =====================================================================
// BackendBox Dispatch Enum (Zero-cost dispatch, no dyn/vtable)
// =====================================================================

pub enum BackendBox {
    Gpu(GpuBackend),
    Cpu(CandleBackend),
}

impl BackendBox {
    pub async fn init(
        num_layers: usize,
        max_seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        gpu_expert_cache: Arc<crate::expert_cache::GpuExpertCache>,
    ) -> Self {
        match GpuBackend::try_new(num_layers, max_seq_len, num_heads, num_kv_heads, head_dim, gpu_expert_cache).await {
            Ok(gpu) => BackendBox::Gpu(gpu),
            Err(e) => {
                tracing::warn!(reason = %e, "GPU init failed — activating CPU fallback");
                BackendBox::Cpu(CandleBackend::new())
            }
        }
    }

    pub fn init_blocking(
        num_layers: usize,
        max_seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        gpu_expert_cache: Arc<crate::expert_cache::GpuExpertCache>,
    ) -> Self {
        pollster::block_on(Self::init(
            num_layers,
            max_seq_len,
            num_heads,
            num_kv_heads,
            head_dim,
            gpu_expert_cache,
        ))
    }
}

impl Backend for BackendBox {
    fn device_name(&self) -> &str {
        match self {
            BackendBox::Gpu(gpu) => gpu.device_name(),
            BackendBox::Cpu(cpu) => cpu.device_name(),
        }
    }

    fn is_gpu(&self) -> bool {
        match self {
            BackendBox::Gpu(gpu) => gpu.is_gpu(),
            BackendBox::Cpu(cpu) => cpu.is_gpu(),
        }
    }

    fn matmul_into(&self, a: TensorView, b: TensorView, out: &mut TensorViewMut) -> Result<()> {
        match self {
            BackendBox::Gpu(gpu) => gpu.matmul_into(a, b, out),
            BackendBox::Cpu(cpu) => cpu.matmul_into(a, b, out),
        }
    }

    fn swiglu_into(&self, gate: TensorView, up: TensorView, out: &mut TensorViewMut) -> Result<()> {
        match self {
            BackendBox::Gpu(gpu) => gpu.swiglu_into(gate, up, out),
            BackendBox::Cpu(cpu) => cpu.swiglu_into(gate, up, out),
        }
    }

    fn softmax(&self, x: &mut TensorViewMut) -> Result<()> {
        match self {
            BackendBox::Gpu(gpu) => gpu.softmax(x),
            BackendBox::Cpu(cpu) => cpu.softmax(x),
        }
    }

    fn kv_cache_insert(
        &self,
        layer: usize,
        position: usize,
        k: TensorView,
        v: TensorView,
    ) -> Result<()> {
        match self {
            BackendBox::Gpu(gpu) => gpu.kv_cache_insert(layer, position, k, v),
            BackendBox::Cpu(cpu) => cpu.kv_cache_insert(layer, position, k, v),
        }
    }

    fn kv_attend(
        &self,
        layer: usize,
        q: TensorView,
        seq_len: usize,
        out: &mut TensorViewMut,
    ) -> Result<()> {
        match self {
            BackendBox::Gpu(gpu) => gpu.kv_attend(layer, q, seq_len, out),
            BackendBox::Cpu(cpu) => cpu.kv_attend(layer, q, seq_len, out),
        }
    }

    fn expert_matmul(
        &self,
        layer_idx: usize,
        expert_id: u32,
        x:        TensorView<'_>,
        d_model:  usize,
        d_ff:     usize,
        out:      &mut TensorViewMut<'_>,
    ) -> Result<()> {
        match self {
            Self::Gpu(g) => g.expert_matmul(layer_idx, expert_id, x, d_model, d_ff, out),
            Self::Cpu(c) => c.expert_matmul(layer_idx, expert_id, x, d_model, d_ff, out),
        }
    }
}

// =====================================================================
// Global active backend Registry
// =====================================================================

static BACKEND: OnceLock<Arc<BackendBox>> = OnceLock::new();

/// Install `b` as the process-wide active backend. Returns `Err` if a
/// backend has already been installed.
pub fn set_backend(b: Arc<BackendBox>) -> Result<(), &'static str> {
    BACKEND
        .set(b)
        .map_err(|_| "backend already installed; call before any token is generated")
}

/// Install the default backend (`CandleBackend`) if none has been set yet.
pub fn install_default() {
    let _ = BACKEND.set(Arc::new(BackendBox::Cpu(CandleBackend::new())));
}

/// Active backend. Falls back to a CPU reference backend when nothing has
/// been installed.
pub fn current() -> Arc<BackendBox> {
    BACKEND
        .get()
        .cloned()
        .unwrap_or_else(|| Arc::new(BackendBox::Cpu(CandleBackend::new())))
}

// =====================================================================
// Operator-facing ComputeOffload Enum
// =====================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ComputeOffload {
    Cpu,
    Gpu,
}

impl Default for ComputeOffload {
    fn default() -> Self {
        Self::Cpu
    }
}

// =====================================================================
// Unit Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_matmul_correctness() {
        let backend = CandleBackend::new();
        let a_data = [
            half::f16::from_f32(1.0),
            half::f16::from_f32(2.0),
            half::f16::from_f32(3.0),
            half::f16::from_f32(4.0),
        ];
        let b_data = [
            half::f16::from_f32(5.0),
            half::f16::from_f32(6.0),
            half::f16::from_f32(7.0),
            half::f16::from_f32(8.0),
        ];
        let mut out_data = [half::f16::ZERO; 4];

        let a = TensorView {
            data: &a_data,
            rows: 2,
            cols: 2,
        };
        let b = TensorView {
            data: &b_data,
            rows: 2,
            cols: 2,
        };
        let mut out = TensorViewMut {
            data: &mut out_data,
            rows: 2,
            cols: 2,
        };

        backend.matmul_into(a, b, &mut out).unwrap();

        // Expected:
        // [1*5 + 2*7, 1*6 + 2*8] = [19, 22]
        // [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
        assert_eq!(out_data[0].to_f32(), 19.0);
        assert_eq!(out_data[1].to_f32(), 22.0);
        assert_eq!(out_data[2].to_f32(), 43.0);
        assert_eq!(out_data[3].to_f32(), 50.0);
    }

    #[test]
    fn test_candle_swiglu_correctness() {
        let backend = CandleBackend::new();
        let gate_data = [half::f16::from_f32(0.0), half::f16::from_f32(1.0)];
        let up_data = [half::f16::from_f32(2.0), half::f16::from_f32(3.0)];
        let mut out_data = [half::f16::ZERO; 2];

        let gate = TensorView {
            data: &gate_data,
            rows: 1,
            cols: 2,
        };
        let up = TensorView {
            data: &up_data,
            rows: 1,
            cols: 2,
        };
        let mut out = TensorViewMut {
            data: &mut out_data,
            rows: 1,
            cols: 2,
        };

        backend.swiglu_into(gate, up, &mut out).unwrap();

        // Expected:
        // out[0] = silu(0) * 2 = 0 * 2 = 0
        // out[1] = silu(1) * 3 = (1 / (1 + exp(-1))) * 3 = 0.7310586 * 3 = 2.1931758
        assert!((out_data[0].to_f32() - 0.0).abs() < 1e-4);
        assert!((out_data[1].to_f32() - 2.1931758).abs() < 1e-3);
    }

    #[test]
    fn test_candle_softmax_correctness() {
        let backend = CandleBackend::new();
        let mut data = [
            half::f16::from_f32(1.0),
            half::f16::from_f32(2.0),
            half::f16::from_f32(3.0),
            half::f16::from_f32(-1.0),
            half::f16::from_f32(0.0),
            half::f16::from_f32(4.0),
        ];
        let mut out = TensorViewMut {
            data: &mut data,
            rows: 2,
            cols: 3,
        };

        backend.softmax(&mut out).unwrap();

        // Row 1 sum: exp(1-3) + exp(2-3) + exp(3-3) = exp(-2) + exp(-1) + 1.0 = 0.1353 + 0.3679 + 1.0 = 1.5032
        // Row 1 values: exp(-2)/1.5032 = 0.0900, exp(-1)/1.5032 = 0.2447, 1.0/1.5032 = 0.6653
        // Sum of Row 1 should be 1.0
        let sum1 = data[0].to_f32() + data[1].to_f32() + data[2].to_f32();
        assert!((sum1 - 1.0).abs() < 1e-3);

        // Row 2 sum: exp(-1-4) + exp(0-4) + exp(4-4) = exp(-5) + exp(-4) + 1.0 = 0.0067 + 0.0183 + 1.0 = 1.0250
        // Sum of Row 2 should be 1.0
        let sum2 = data[3].to_f32() + data[4].to_f32() + data[5].to_f32();
        assert!((sum2 - 1.0).abs() < 1e-3);
    }
}

#[cfg(test)]
mod q4_0_shader_logic_tests {
    //! Host-side mirror of `wgpu_shaders/matmul_q4_0.wgsl`.
    //!
    //! The riskiest part of the inline-dequant GEMV shader is the byte
    //! arithmetic: 18-byte Q4_0 blocks bound as `array<u32>`, per-byte
    //! extraction with shifts, the f16 scale decode and the
    //! low-nibble-first weight order. These tests re-implement that
    //! exact logic in Rust (keep in sync with the WGSL!) and check it
    //! against the canonical CPU dequantiser
    //! [`crate::inference::dequantize_q4_0_block`], so a nibble-order
    //! or offset mistake in the shader's math shows up in CI without
    //! needing a GPU adapter.

    use crate::inference::{
        dequantize_q4_0_block, quantize_q4_0_block, Q4_0_BLOCK_BYTES, Q4_0_BLOCK_ELEMS,
    };

    /// Mirror of the WGSL `read_byte` helper.
    fn read_byte(w: &[u32], off: usize) -> u32 {
        (w[off >> 2] >> ((off & 3) * 8)) & 0xff
    }

    /// Pack a little-endian byte stream into the `array<u32>` view the
    /// shader binds, zero-padding to a 4-byte boundary exactly like
    /// `build_expert_entry_q4_0` does.
    fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
        let mut padded = bytes.to_vec();
        padded.resize(bytes.len().div_ceil(4) * 4, 0);
        padded
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Mirror of the WGSL `matmul_q4_0_main` body for one output row.
    fn shader_row_dot(w: &[u32], w_block_off: usize, row: usize, k: usize, x: &[f32]) -> f32 {
        let blocks_per_row = k / Q4_0_BLOCK_ELEMS;
        let mut byte_off = (w_block_off + row * blocks_per_row) * Q4_0_BLOCK_BYTES;
        let mut x_base = 0usize;
        let mut sum = 0.0f32;
        for _ in 0..blocks_per_row {
            let s_lo = read_byte(w, byte_off);
            let s_hi = read_byte(w, byte_off + 1);
            let d = half::f16::from_bits((s_lo | (s_hi << 8)) as u16).to_f32();
            let mut partial = 0.0f32;
            for j in 0..16 {
                let q = read_byte(w, byte_off + 2 + j);
                let w0 = (q & 0xf) as f32 - 8.0;
                let w1 = (q >> 4) as f32 - 8.0;
                partial += w0 * x[x_base + 2 * j] + w1 * x[x_base + 2 * j + 1];
            }
            sum += d * partial;
            byte_off += Q4_0_BLOCK_BYTES;
            x_base += Q4_0_BLOCK_ELEMS;
        }
        sum
    }

    /// Deterministic pseudo-random weights that exercise the full
    /// nibble range, both signs and varying block scales.
    fn synth_weights(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).max(1);
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state % 2000) as f32 - 1000.0) / 250.0
            })
            .collect()
    }

    /// Quantise an `m × k` row-major matrix into a tight Q4_0 block
    /// stream (rows start on block boundaries because `k % 32 == 0`).
    fn quantize_matrix(weights: &[f32]) -> Vec<u8> {
        let mut out = Vec::new();
        for chunk in weights.chunks(Q4_0_BLOCK_ELEMS) {
            let mut blk = [0u8; Q4_0_BLOCK_BYTES];
            quantize_q4_0_block(chunk, &mut blk);
            out.extend_from_slice(&blk);
        }
        out
    }

    #[test]
    fn shader_byte_extraction_matches_canonical_block_dequant() {
        // One block whose start lands on a *non*-4-byte-aligned offset
        // (block 1 starts at byte 18), the case the `array<u32>` shift
        // logic exists for.
        let src_a = synth_weights(Q4_0_BLOCK_ELEMS, 7);
        let src_b = synth_weights(Q4_0_BLOCK_ELEMS, 11);
        let mut bytes = Vec::new();
        for src in [&src_a, &src_b] {
            let mut blk = [0u8; Q4_0_BLOCK_BYTES];
            quantize_q4_0_block(src, &mut blk);
            bytes.extend_from_slice(&blk);
        }
        let words = bytes_to_words(&bytes);

        for (bi, range) in [(0usize, 0..Q4_0_BLOCK_BYTES), (1, Q4_0_BLOCK_BYTES..2 * Q4_0_BLOCK_BYTES)] {
            let mut expected = [0.0f32; Q4_0_BLOCK_ELEMS];
            dequantize_q4_0_block(&bytes[range], &mut expected);
            // Dot with a one-hot x isolates each dequantised weight.
            for i in 0..Q4_0_BLOCK_ELEMS {
                let mut x = vec![0.0f32; Q4_0_BLOCK_ELEMS];
                x[i] = 1.0;
                let got = shader_row_dot(&words, bi, 0, Q4_0_BLOCK_ELEMS, &x);
                assert!(
                    (got - expected[i]).abs() < 1e-6,
                    "block {bi} elem {i}: shader logic {got} != canonical {expected:?}"
                );
            }
        }
    }

    #[test]
    fn shader_gemv_matches_cpu_dequant_gemv_with_block_offset() {
        // Small m × k matrix behind a non-zero `w_block_off`, mimicking
        // the up/down projections inside the packed [gate|up|down]
        // expert buffer.
        let (m, k) = (4usize, 64usize);
        let lead_blocks = 3usize; // "gate" blocks preceding this projection
        let lead = synth_weights(lead_blocks * Q4_0_BLOCK_ELEMS, 23);
        let mat = synth_weights(m * k, 42);
        let x = synth_weights(k, 99);

        let mut bytes = quantize_matrix(&lead);
        bytes.extend_from_slice(&quantize_matrix(&mat));
        let words = bytes_to_words(&bytes);

        // Expected: canonical block dequant, then a plain dot per row.
        let mat_bytes = quantize_matrix(&mat);
        let mut dequant = vec![0.0f32; m * k];
        for (b, blk) in mat_bytes.chunks_exact(Q4_0_BLOCK_BYTES).enumerate() {
            dequantize_q4_0_block(blk, &mut dequant[b * Q4_0_BLOCK_ELEMS..(b + 1) * Q4_0_BLOCK_ELEMS]);
        }
        for row in 0..m {
            let expected: f32 = (0..k).map(|c| dequant[row * k + c] * x[c]).sum();
            let got = shader_row_dot(&words, lead_blocks, row, k, &x);
            assert!(
                (got - expected).abs() < 1e-4 * expected.abs().max(1.0),
                "row {row}: shader logic {got} != cpu {expected}"
            );
        }
    }
}

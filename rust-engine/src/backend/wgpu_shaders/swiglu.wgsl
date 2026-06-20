struct PushConstants {
    n_elements: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};
var<push_constant> pc: PushConstants;

@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn swiglu_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < pc.n_elements) {
        let g = gate[idx];
        let u = up[idx];
        // TODO: apply swiglu_limit (GPT-OSS) in WGSL. The CPU path clamps
        // `g` to [-limit, limit] before the sigmoid; replicating it here
        // needs the limit threaded in as a push constant / uniform, which
        // would require a pipeline rebuild. The GPU compute path is not used
        // in production yet, so the clamp is applied on the CPU path only.
        let silu_g = g / (1.0 + exp(-g));
        out[idx] = silu_g * u;
    }
}

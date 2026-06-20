struct PushConstants {
    n_elements: u32,
    // GPT-OSS SwiGLU gate clamp: `g` is clamped to [-swiglu_limit, swiglu_limit]
    // before the sigmoid. Callers pass `+inf` (a positive infinity) when no
    // clamp is active, which makes `clamp(g, -inf, inf)` a bit-exact no-op so
    // every non-GPT-OSS architecture is unaffected.
    swiglu_limit: f32,
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
        // GPT-OSS `swiglu_limit`: clamp the gate to [-limit, limit] before the
        // sigmoid, matching the CPU reference (`kernels::scalar::swiglu_f32_clamped`).
        let limit = pc.swiglu_limit;
        let clamped_g = clamp(g, -limit, limit);
        let silu_g = clamped_g / (1.0 + exp(-clamped_g));
        out[idx] = silu_g * u;
    }
}

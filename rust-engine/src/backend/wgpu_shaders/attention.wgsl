// attention.wgsl
// Push constants: {num_heads, head_dim, seq_len, layer_offset: u32}
struct PushConstants {
    num_heads: u32,
    head_dim: u32,
    seq_len: u32,
    layer_offset: u32,
};
var<push_constant> pc: PushConstants;

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> KV: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;

var<workgroup> shared_dot: array<f32, 32>;
var<workgroup> shared_out: array<f32, MAX_HEAD_DIM>;
var<workgroup> wg_m: f32;
var<workgroup> wg_d: f32;

// MAX_SEQ_LEN is defined at compile time and will be replaced dynamically at runtime
const MAX_SEQ_LEN: u32 = 4096u;
const MAX_HEAD_DIM: u32 = 256u;

@compute @workgroup_size(32, 1, 1)
fn attention_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let h = wg_id.x;
    let tid = local_id.x;

    if (h >= pc.num_heads) {
        return;
    }

    // Initialize shared memory
    if (tid == 0u) {
        wg_m = -999999999.0;
        wg_d = 0.0;
    }
    for (var j = tid; j < pc.head_dim; j = j + 32u) {
        shared_out[j] = 0.0;
    }
    workgroupBarrier();

    let kv_dim = pc.num_heads * pc.head_dim;
    let f32_offset = pc.layer_offset / 4u;
    let scale = 1.0 / sqrt(f32(pc.head_dim));

    for (var t = 0u; t < pc.seq_len; t = t + 1u) {
        // 1. Compute dot product Q_h . K_h,t
        let key_offset = f32_offset + t * kv_dim + h * pc.head_dim;
        var partial = 0.0;
        for (var j = tid; j < pc.head_dim; j = j + 32u) {
            partial = partial + Q[h * pc.head_dim + j] * KV[key_offset + j];
        }
        shared_dot[tid] = partial;
        workgroupBarrier();

        // Reduce dot product
        for (var stride = 16u; stride > 0u; stride = stride / 2u) {
            if (tid < stride) {
                shared_dot[tid] = shared_dot[tid] + shared_dot[tid + stride];
            }
            workgroupBarrier();
        }

        let s_t = shared_dot[0] * scale;

        // 2. Update online softmax parameters
        let old_m = wg_m;
        let new_m = max(old_m, s_t);
        let factor_old = exp(old_m - new_m);
        let factor_new = exp(s_t - new_m);
        let new_d = wg_d * factor_old + factor_new;

        // 3. Update output accumulator
        let val_offset = f32_offset + MAX_SEQ_LEN * kv_dim + t * kv_dim + h * pc.head_dim;
        for (var j = tid; j < pc.head_dim; j = j + 32u) {
            let v_val = KV[val_offset + j];
            shared_out[j] = shared_out[j] * factor_old + factor_new * v_val;
        }

        if (tid == 0u) {
            wg_m = new_m;
            wg_d = new_d;
        }
        workgroupBarrier();
    }

    // Final normalization
    let final_d = wg_d;
    for (var j = tid; j < pc.head_dim; j = j + 32u) {
        let out_idx = h * pc.head_dim + j;
        if (final_d > 0.0) {
            OUT[out_idx] = shared_out[j] / final_d;
        } else {
            OUT[out_idx] = 0.0;
        }
    }
}

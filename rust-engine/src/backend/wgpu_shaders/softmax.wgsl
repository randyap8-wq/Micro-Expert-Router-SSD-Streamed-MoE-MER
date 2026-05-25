struct PushConstants {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
};
var<push_constant> pc: PushConstants;

@group(0) @binding(0) var<storage, read_write> X: array<f32>;

var<workgroup> shared_max: array<f32, 64>;
var<workgroup> shared_sum: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn softmax_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let row = wg_id.x;
    let tid = local_id.x;

    if (row >= pc.rows) {
        return;
    }

    // Step 1: Find max value of the row
    var local_max = -999999999.0;
    for (var col = tid; col < pc.cols; col = col + 64u) {
        let val = X[row * pc.cols + col];
        if (val > local_max) {
            local_max = val;
        }
    }
    shared_max[tid] = local_max;
    workgroupBarrier();

    // Reduce shared_max
    for (var stride = 32u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            if (shared_max[tid + stride] > shared_max[tid]) {
                shared_max[tid] = shared_max[tid + stride];
            }
        }
        workgroupBarrier();
    }
    let row_max = shared_max[0];

    // Step 2: Compute sum of exp(val - row_max)
    var local_sum = 0.0;
    for (var col = tid; col < pc.cols; col = col + 64u) {
        let idx = row * pc.cols + col;
        let val = exp(X[idx] - row_max);
        X[idx] = val; // Store exp temporarily
        local_sum = local_sum + val;
    }
    shared_sum[tid] = local_sum;
    workgroupBarrier();

    // Reduce shared_sum
    for (var stride = 32u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
        }
        workgroupBarrier();
    }
    let row_sum = shared_sum[0];

    // Step 3: Normalise
    for (var col = tid; col < pc.cols; col = col + 64u) {
        let idx = row * pc.cols + col;
        if (row_sum > 0.0) {
            X[idx] = X[idx] / row_sum;
        } else {
            X[idx] = 0.0;
        }
    }
}

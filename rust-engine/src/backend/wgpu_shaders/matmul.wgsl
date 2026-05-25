struct PushConstants {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
};
var<push_constant> pc: PushConstants;

@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;

var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn matmul_main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let row = global_id.y;
    let col = global_id.x;
    let local_row = local_id.y;
    let local_col = local_id.x;

    var sum = 0.0;
    let num_tiles = (pc.k + 15u) / 16u;

    for (var t = 0u; t < num_tiles; t = t + 1u) {
        // Load tile of A
        let a_col = t * 16u + local_col;
        if (row < pc.m && a_col < pc.k) {
            tile_a[local_row][local_col] = A[row * pc.k + a_col];
        } else {
            tile_a[local_row][local_col] = 0.0;
        }

        // Load tile of B
        let b_row = t * 16u + local_row;
        if (b_row < pc.k && col < pc.n) {
            tile_b[local_row][local_col] = B[b_row * pc.n + col];
        } else {
            tile_b[local_row][local_col] = 0.0;
        }

        workgroupBarrier();

        // Compute dot product for this tile
        for (var i = 0u; i < 16u; i = i + 1u) {
            sum = sum + tile_a[local_row][i] * tile_b[i][local_col];
        }

        workgroupBarrier();
    }

    if (row < pc.m && col < pc.n) {
        OUT[row * pc.n + col] = sum;
    }
}

// Q4_0 inline-dequant GEMV: y[row] = Σ_col dequant(W)[row, col] · x[col]
//
// The weight matrix stays in its native GGUF Q4_0 block format in VRAM —
// 18 bytes per 32 weights (f16 scale + 16 nibble bytes) — and each block
// is unpacked *inside* the shader, right next to the multiply-accumulate.
// Compared to the F32 `matmul.wgsl` path this cuts both the PCIe upload
// and the per-dispatch VRAM bandwidth by ~8× (4.5 bits/weight instead of
// 32), which is exactly the bandwidth budget the dense F32 shader burned
// on pre-dequantised weights.
//
// Block layout (must match `inference::dequantize_q4_0_block`):
//   d  : f16 little-endian (2 bytes)   — block scale
//   qs : 16 bytes                      — 32× 4-bit weights, low nibble
//                                        first: elem 2j = qs[j] & 0xF,
//                                        elem 2j+1 = qs[j] >> 4; both
//                                        biased by -8.
//
// The expert FFN only ever needs N == 1 (a per-token GEMV), so this is a
// row-per-invocation kernel rather than a tiled GEMM: each invocation
// owns one output row and walks that row's k/32 blocks sequentially.
// Q4_0 blocks are 18 bytes — *not* 4-byte aligned — so the buffer is
// bound as `array<u32>` and bytes are extracted with shifts. Rows do
// start on block boundaries because the host guarantees k % 32 == 0
// (`Engine::gpu_eligible_dtype`).

struct PushConstants {
    // Rows of the weight matrix (output length).
    m: u32,
    // Unused (always 1 for the expert GEMV); kept so the push-constant
    // block matches `MatmulPushConstants` on the host.
    n: u32,
    // Columns of the weight matrix == len(x). Must be a multiple of 32.
    k: u32,
    // Block index of this projection's first Q4_0 block inside W. The
    // whole [gate || up || down] expert buffer is bound at offset 0
    // (18-byte blocks cannot honour storage-offset alignment rules),
    // so the projection base is selected here instead of via the
    // bind-group offset.
    w_block_off: u32,
};
var<push_constant> pc: PushConstants;

@group(0) @binding(0) var<storage, read> W: array<u32>;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;

const BLOCK_BYTES: u32 = 18u;
const BLOCK_ELEMS: u32 = 32u;

fn read_byte(off: u32) -> u32 {
    return (W[off >> 2u] >> ((off & 3u) * 8u)) & 0xffu;
}

@compute @workgroup_size(64, 1, 1)
fn matmul_q4_0_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= pc.m) {
        return;
    }
    let blocks_per_row = pc.k / BLOCK_ELEMS;
    var byte_off = (pc.w_block_off + row * blocks_per_row) * BLOCK_BYTES;
    var x_base = 0u;
    var sum = 0.0;

    for (var b = 0u; b < blocks_per_row; b = b + 1u) {
        // f16 LE block scale from the first two bytes.
        let s_lo = read_byte(byte_off);
        let s_hi = read_byte(byte_off + 1u);
        let d = unpack2x16float(s_lo | (s_hi << 8u)).x;

        // 16 nibble bytes → 32 weights. Accumulate the un-scaled dot
        // product per block and apply the scale once at the end.
        var partial = 0.0;
        for (var j = 0u; j < 16u; j = j + 1u) {
            let q = read_byte(byte_off + 2u + j);
            let w0 = f32(q & 0xfu) - 8.0;
            let w1 = f32(q >> 4u) - 8.0;
            partial = partial + w0 * X[x_base + 2u * j] + w1 * X[x_base + 2u * j + 1u];
        }
        sum = sum + d * partial;

        byte_off = byte_off + BLOCK_BYTES;
        x_base = x_base + BLOCK_ELEMS;
    }

    OUT[row] = sum;
}

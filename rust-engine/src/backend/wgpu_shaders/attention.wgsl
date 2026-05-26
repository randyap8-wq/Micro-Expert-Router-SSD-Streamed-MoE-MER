// attention.wgsl — v2: seq-tiled parallel attention
//
// One workgroup per attention head (dispatch: [num_heads, 1, 1]).
// WG = 32 threads per workgroup.
//
// v1 processed one seq position per outer iteration (seq_len barriers).
// v2 processes WG=32 positions per tile (ceil(seq_len/32) barriers).
//
// Per tile:
//   Step 1: Each thread tid computes score s[tid] = dot(Q_h, K_h[t]) * scale
//           where t = tile*WG + tid. Serial over head_dim per thread; no
//           intra-tile reduction needed for the dot product.
//   Step 2: 5-step binary tree reduces the 32 scores to (tile_m, tile_d)
//           using the Milakov & Gimelshein stable merge:
//             merge(m_a,d_a, m_b,d_b) -> m=max(m_a,m_b),
//                                        d=d_a*exp(m_a-m)+d_b*exp(m_b-m)
//   Step 3: V accumulation. Thread tid handles j = tid, tid+WG, tid+2*WG,...
//           For each j, iterates over all t_count valid positions in the tile:
//             shared_out[j] = shared_out[j] * factor_old
//                           + sum_{t_local} exp(s[t_local]-new_m) * KV_V[t][j]
//           This is correct because each thread owns a disjoint j-range, so
//           there are no write conflicts in shared_out.
//   Step 4: Thread 0 updates running (run_m, run_d).
//
// Push constants: {num_heads, head_dim, seq_len, layer_offset: u32}
// KV layout: [layer][kv=0(K)/kv=1(V)][seq_pos][kv_dim] stored as f32.
//   K base for position t: f32_off + t * kv_dim + h * head_dim
//   V base for position t: f32_off + MAX_SEQ_LEN * kv_dim + t * kv_dim + h * head_dim
//
// MAX_SEQ_LEN and MAX_HEAD_DIM are replaced by GpuBackend::try_new() before
// shader compilation; the literals here are defaults that must not be used
// at runtime without substitution.

struct PushConstants {
    num_heads:    u32,
    head_dim:     u32,
    seq_len:      u32,
    layer_offset: u32,
}
var<push_constant> pc: PushConstants;

@group(0) @binding(0) var<storage, read>       Q:   array<f32>;
@group(0) @binding(1) var<storage, read>       KV:  array<f32>;
@group(0) @binding(2) var<storage, read_write> OUT: array<f32>;

// ── Runtime-substituted constants ─────────────────────────────────────────────
const MAX_SEQ_LEN: u32 = 4096u;   // replaced by GpuBackend::try_new
const MAX_HEAD_DIM: u32 = 256u;   // replaced by GpuBackend::try_new

// ── Workgroup size ─────────────────────────────────────────────────────────────
const WG: u32 = 32u;

// ── Workgroup shared memory ────────────────────────────────────────────────────

// Running online-softmax state across all tiles.
var<workgroup> run_m: f32;   // running global max
var<workgroup> run_d: f32;   // running denominator  = sum_t exp(s_t - run_m)

// Per-tile scratch: one score per thread + tree-reduction buffers.
var<workgroup> shared_scores: array<f32, 32u>;
var<workgroup> shared_m:      array<f32, 32u>;
var<workgroup> shared_d:      array<f32, 32u>;

// Running weighted-V accumulator across all tiles [head_dim].
var<workgroup> shared_out: array<f32, MAX_HEAD_DIM>;

// ── Kernel ─────────────────────────────────────────────────────────────────────

@compute @workgroup_size(32, 1, 1)
fn attention_main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id)        wg_id:    vec3<u32>,
) {
    let h   = wg_id.x;
    let tid = local_id.x;

    if h >= pc.num_heads { return; }

    // ── Initialise shared state ───────────────────────────────────────────────
    if tid == 0u {
        run_m = -3.402823e+38;
        run_d = 0.0;
    }
    for (var j = tid; j < pc.head_dim; j += WG) {
        shared_out[j] = 0.0;
    }
    workgroupBarrier();

    let kv_dim    = pc.num_heads * pc.head_dim;
    let f32_off   = pc.layer_offset / 4u;
    let scale     = 1.0 / sqrt(f32(pc.head_dim));
    let num_tiles = (pc.seq_len + WG - 1u) / WG;

    // ── Main tile loop ─────────────────────────────────────────────────────────
    for (var tile = 0u; tile < num_tiles; tile++) {

        let t       = tile * WG + tid;
        let valid   = t < pc.seq_len;
        // Number of valid positions in this tile (1..WG; always WG except last tile).
        let t_count = min(WG, pc.seq_len - tile * WG);

        // ── Step 1: Thread tid computes dot(Q_h, K_h[t]) * scale ─────────────
        // Each thread handles its own seq position independently — no reduction
        // needed here, which eliminates one barrier vs v1.
        var s = -3.402823e+38;
        if valid {
            s = 0.0;
            let key_off = f32_off + t * kv_dim + h * pc.head_dim;
            for (var j = 0u; j < pc.head_dim; j++) {
                s += Q[h * pc.head_dim + j] * KV[key_off + j];
            }
            s *= scale;
        }
        shared_scores[tid] = s;

        // Initialise tree-reduction buffers.
        //   valid position:  (m=s, d=1.0)  [d = exp(s-s) = 1]
        //   padding:         (m=-inf, d=0.0) [contributes nothing to the merge]
        shared_m[tid] = s;
        shared_d[tid] = select(0.0, 1.0, valid);
        workgroupBarrier();

        // ── Step 2: 5-step binary tree — produces (tile_m, tile_d) ───────────
        // Merge rule for two online-softmax accumulators (a) and (b):
        //   m_new = max(m_a, m_b)
        //   d_new = d_a * exp(m_a - m_new) + d_b * exp(m_b - m_new)
        for (var step = 0u; step < 5u; step++) {
            let half = 16u >> step;   // 16, 8, 4, 2, 1
            if tid < half {
                let m_a   = shared_m[tid];
                let d_a   = shared_d[tid];
                let m_b   = shared_m[tid + half];
                let d_b   = shared_d[tid + half];
                let m_new = max(m_a, m_b);
                shared_m[tid] = m_new;
                shared_d[tid] = d_a * exp(m_a - m_new) + d_b * exp(m_b - m_new);
            }
            workgroupBarrier();
        }
        // shared_m[0] = tile_m,  shared_d[0] = tile_d

        // ── Step 3: V accumulation ────────────────────────────────────────────
        // Compute new running max and the rescaling factor for the old accumulator.
        let old_run_m  = run_m;
        let tile_m     = shared_m[0];
        let new_run_m  = max(old_run_m, tile_m);
        let factor_old = exp(old_run_m - new_run_m);  // rescale factor for shared_out

        // Thread tid owns j = tid, tid+WG, tid+2*WG, ... of shared_out.
        // For each j, loops over all t_count valid positions in the tile and
        // accumulates the weighted V contribution. Because each thread owns a
        // disjoint j-range, there are no write conflicts in shared_out.
        for (var j = tid; j < pc.head_dim; j += WG) {
            var v_acc = shared_out[j] * factor_old;
            for (var tl = 0u; tl < t_count; tl++) {
                let w     = exp(shared_scores[tl] - new_run_m);
                let t_abs = tile * WG + tl;
                let v_off = f32_off
                          + MAX_SEQ_LEN * kv_dim    // jump past K slice
                          + t_abs * kv_dim           // seq position stride
                          + h * pc.head_dim          // head offset
                          + j;                       // element index
                v_acc += w * KV[v_off];
            }
            shared_out[j] = v_acc;
        }
        workgroupBarrier();

        // ── Step 4: Update running state (thread 0 only) ─────────────────────
        if tid == 0u {
            let tile_d = shared_d[0];
            run_d = run_d * factor_old + tile_d * exp(tile_m - new_run_m);
            run_m = new_run_m;
        }
        workgroupBarrier();
    }

    // ── Final normalisation ────────────────────────────────────────────────────
    let inv_d = select(0.0, 1.0 / run_d, run_d > 0.0);
    for (var j = tid; j < pc.head_dim; j += WG) {
        OUT[h * pc.head_dim + j] = shared_out[j] * inv_d;
    }
}

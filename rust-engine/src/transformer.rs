//! Scalar Rust implementation of the dense pieces of a Mixtral / Llama-style
//! transformer decoder layer:
//!
//! * [`RmsNorm`] — RMSNorm normalisation.
//! * [`apply_rope_inplace`] — rotary positional embedding for one head.
//! * [`MultiHeadSelfAttention`] — scalar causal multi-head self attention
//!   with a per-layer KV cache. Grouped-query attention (GQA) supported by
//!   passing `num_kv_heads < num_heads`.
//! * [`TransformerLayer`] — wires `attention -> residual -> rmsnorm -> moe`.
//!
//! These are the pieces the gist asks for in **Phase 2**. They are
//! deliberately written in plain `f32` Rust — no BLAS, no SIMD intrinsic
//! crates — for two reasons:
//!
//! 1. The whole engine builds on stable Rust with zero non-Rust toolchain
//!    requirements (the existing scalar SwiGLU expert kernel in
//!    [`crate::inference`] is the same shape).
//! 2. The dense weights are *resident* (small relative to the per-layer
//!    expert weights), so the dense compute is not the bottleneck the
//!    engine is built to optimise — it's the SSD-streamed expert FFN. The
//!    interesting work of this codebase is the streaming cache, not the
//!    matmul kernel.
//!
//! The matmul shape and call sites match what `candle_transformers::models::mixtral`
//! does, so a future PR can swap each method for a `candle::Tensor` op
//! without changing call sites.
//!
//! These types are exercised by unit tests below; production wiring (a
//! full forward pass through stacked `TransformerLayer`s with loaded
//! Mixtral weights) lands in a follow-up PR — the [`crate::server`]
//! generation loop currently drives [`crate::engine::Engine::generate`]
//! directly, which is enough to exercise the SSD-streaming substrate
//! end-to-end. Allow `dead_code` here so the public API is greppable
//! without forcing a stub call site.
#![allow(dead_code)]


use crate::expert_cache::ExpertResident;
use crate::inference::{ExpertWeights, HiddenState};
use half::f16;

/// RMSNorm: `y = x * rsqrt(mean(x^2) + eps) * weight`.
///
/// Used before attention and before the MoE block in Llama / Mixtral-style
/// architectures. The `weight` parameter is a learnable per-channel scale
/// of length `d_model`.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    pub weight: Vec<f32>,
    pub eps: f32,
}

impl RmsNorm {
    pub fn new(weight: Vec<f32>, eps: f32) -> Self {
        Self { weight, eps }
    }

    /// In-place RMSNorm. `x.len()` must equal `weight.len()`.
    pub fn forward_inplace(&self, x: &mut [f32]) {
        debug_assert_eq!(x.len(), self.weight.len(), "RMSNorm dim mismatch");
        let n = x.len() as f32;
        let mut sq_sum = 0.0f32;
        for &v in x.iter() {
            sq_sum += v * v;
        }
        let mean_sq = sq_sum / n;
        let scale = 1.0 / (mean_sq + self.eps).sqrt();
        for (xi, wi) in x.iter_mut().zip(self.weight.iter()) {
            *xi = *xi * scale * *wi;
        }
    }

    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut y = x.to_vec();
        self.forward_inplace(&mut y);
        y
    }
}

/// Apply rotary positional embedding to a single head's `(q or k)` vector
/// **in place** at absolute position `pos`.
///
/// Layout: contiguous `head_dim` floats; rotated as `head_dim/2` complex
/// pairs at frequencies `1 / base^(2i / head_dim)`. Matches the rotary
/// convention used by Llama-2/3 and Mixtral.
pub fn apply_rope_inplace(v: &mut [f32], pos: usize, base: f32) {
    let head_dim = v.len();
    debug_assert!(head_dim % 2 == 0, "RoPE requires even head_dim");
    let half = head_dim / 2;
    let pos_f = pos as f32;
    for i in 0..half {
        // Inverse-frequency for this pair.
        let inv_freq = 1.0 / base.powf(2.0 * i as f32 / head_dim as f32);
        let theta = pos_f * inv_freq;
        let (sin_t, cos_t) = theta.sin_cos();
        // Pair up dim `i` and dim `i + half` (Llama convention; matches
        // candle-transformers' Mixtral implementation).
        let a = v[i];
        let b = v[i + half];
        v[i] = a * cos_t - b * sin_t;
        v[i + half] = a * sin_t + b * cos_t;
    }
}

/// One layer's **paged** KV cache (per-layer). Stores keys and values
/// in fixed-size blocks of [`PAGED_BLOCK_TOKENS`] tokens × `kv_dim`
/// floats each, indexed by a block table.
///
/// Replaces the original `Vec<f32>` flat layout (which `extend_from_slice`d
/// — and therefore reallocated the entire backing storage — every time
/// the sequence grew past the `Vec`'s capacity) with vLLM-style
/// PagedAttention block allocation: each block is a separately-owned
/// boxed slice and the cache grows by *appending one new block to the
/// block table* when the trailing block fills up. The block table
/// (`Vec<Box<[f32]>>` here, conceptually a `Vec<u32>` of block ids in a
/// shared pool) is the indirection layer that decouples per-request
/// sequence growth from a single contiguous allocation per request.
///
/// Public surface compatibility: the original `KvCache` exposed
/// `keys: Vec<f32>` / `values: Vec<f32>` as `pub` fields used only
/// inside this module via `key(i)` / `value(i)` accessors and a single
/// `extend_from_slice` call inside `append`. All external callers go
/// through `KvCache::new(kv_dim)` plus `append`, `reset`, `seq_len` and
/// `kv_dim`, all of which keep their original semantics.
#[derive(Debug, Clone, Default)]
pub struct KvCache {
    /// Block table: each entry holds [`PAGED_BLOCK_TOKENS`] tokens'
    /// worth of `kv_dim` floats laid out as `[token_in_block, kv_dim]`
    /// row-major. The last block may be partially filled (the unused
    /// tail is never read because `seq_len` bounds every iteration).
    keys_blocks: Vec<Box<[f32]>>,
    /// Mirrors `keys_blocks` for the value half of the cache.
    values_blocks: Vec<Box<[f32]>>,
    pub seq_len: usize,
    pub kv_dim: usize,
}

/// Number of tokens per PagedAttention block. Matches the spec value
/// (16) and the vLLM default. Larger blocks waste more memory at the
/// tail; smaller blocks add more block-table indirections per
/// attention sweep. 16 is a known sweet spot for Mixtral / Llama
/// shapes and keeps each block at `16 * kv_dim * 4` bytes — well
/// under one OS page for any realistic `kv_dim`.
pub const PAGED_BLOCK_TOKENS: usize = 16;

impl KvCache {
    pub fn new(kv_dim: usize) -> Self {
        Self {
            keys_blocks: Vec::new(),
            values_blocks: Vec::new(),
            seq_len: 0,
            kv_dim,
        }
    }

    pub fn append(&mut self, k: &[f32], v: &[f32]) {
        debug_assert_eq!(k.len(), self.kv_dim);
        debug_assert_eq!(v.len(), self.kv_dim);
        let pos = self.seq_len;
        let block_idx = pos / PAGED_BLOCK_TOKENS;
        let in_block = pos % PAGED_BLOCK_TOKENS;
        // Allocate a fresh block when crossing a block boundary. This
        // is the *only* allocation point in the per-token path — the
        // existing block bytes are written in place.
        if in_block == 0 {
            debug_assert_eq!(self.keys_blocks.len(), block_idx);
            let block_floats = PAGED_BLOCK_TOKENS * self.kv_dim;
            self.keys_blocks
                .push(vec![0.0f32; block_floats].into_boxed_slice());
            self.values_blocks
                .push(vec![0.0f32; block_floats].into_boxed_slice());
        }
        let start = in_block * self.kv_dim;
        let end = start + self.kv_dim;
        // Borrow the freshly-allocated (or current) trailing block and
        // write directly into its in-place slot.
        let kb = self
            .keys_blocks
            .last_mut()
            .expect("block must exist after append");
        let vb = self
            .values_blocks
            .last_mut()
            .expect("block must exist after append");
        kb[start..end].copy_from_slice(k);
        vb[start..end].copy_from_slice(v);
        self.seq_len += 1;
    }

    pub fn reset(&mut self) {
        self.keys_blocks.clear();
        self.values_blocks.clear();
        self.seq_len = 0;
    }

    /// Overwrite every cached K/V byte with zero before discarding the
    /// cache, then truncate the block tables. Called from the session
    /// store's `DELETE /v1/sessions/{id}` handler so that a tenant's
    /// (potentially sensitive) attention state cannot be read by a
    /// subsequent allocation that lands in the same heap region.
    ///
    /// We use `std::ptr::write_volatile` to perform the zeroing
    /// writes: volatile stores are defined to have observable side
    /// effects and may not be elided by the optimiser, which is the
    /// guarantee `Vec::fill` (followed by a drop) does *not* provide
    /// — without volatile semantics the stdlib `fill` of a value that
    /// is dropped immediately afterwards is, in principle, eligible
    /// for dead-store elimination.
    pub fn zeroize(&mut self) {
        zeroize_blocks(&mut self.keys_blocks);
        zeroize_blocks(&mut self.values_blocks);
        self.reset();
    }

    /// Number of allocated blocks. Useful for telemetry — matches
    /// the vLLM `block_tables` length.
    pub fn num_blocks(&self) -> usize {
        self.keys_blocks.len()
    }

    /// Get the i-th cached key as a slice of length `kv_dim`.
    fn key(&self, i: usize) -> &[f32] {
        let block_idx = i / PAGED_BLOCK_TOKENS;
        let in_block = i % PAGED_BLOCK_TOKENS;
        let start = in_block * self.kv_dim;
        &self.keys_blocks[block_idx][start..start + self.kv_dim]
    }

    fn value(&self, i: usize) -> &[f32] {
        let block_idx = i / PAGED_BLOCK_TOKENS;
        let in_block = i % PAGED_BLOCK_TOKENS;
        let start = in_block * self.kv_dim;
        &self.values_blocks[block_idx][start..start + self.kv_dim]
    }
}

/// Zero every `f32` of every block via `ptr::write_volatile` so the
/// optimiser cannot elide the stores even though the underlying
/// `Vec`s are dropped immediately afterwards. The trailing
/// `compiler_fence` prevents the writes from being reordered past
/// the eventual deallocation of the backing buffers.
#[inline(never)]
fn zeroize_blocks(blocks: &mut [Box<[f32]>]) {
    for block in blocks.iter_mut() {
        let ptr = block.as_mut_ptr();
        let len = block.len();
        // Safety: `block` is a `Box<[f32]>` (boxed slice) of length
        // `len`. If `len == 0`, the loop below executes zero times,
        // so `ptr` is never dereferenced; this remains valid even if
        // an empty boxed slice uses a dangling but properly aligned
        // non-null pointer and performs no allocation. If `len > 0`,
        // `ptr` points to the start of the slice's contiguous
        // storage, so every offset `0..len` is in-bounds and
        // properly aligned for `f32`. The slice is uniquely borrowed
        // via `&mut block`, so no other thread or reference can
        // observe or mutate these bytes for the duration of the
        // loop.
        for i in 0..len {
            unsafe { std::ptr::write_volatile(ptr.add(i), 0.0f32) };
        }
    }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

/// Causal multi-head self attention with optional grouped-query attention.
///
/// Weights are stored row-major:
/// * `wq` : `[num_heads * head_dim, d_model]`
/// * `wk` : `[num_kv_heads * head_dim, d_model]`
/// * `wv` : `[num_kv_heads * head_dim, d_model]`
/// * `wo` : `[d_model, num_heads * head_dim]`
#[derive(Debug, Clone)]
pub struct MultiHeadSelfAttention {
    pub d_model: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_base: f32,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    /// Sliding-window attention span (Mixtral default = 4096). When
    /// `Some(w)`, each query position `pos` only attends to KV positions
    /// in `[pos.saturating_sub(w - 1) ..= pos]`. The KV cache itself
    /// still stores all positions (that's required for correctness as
    /// the window slides forward); only the attention sum is restricted.
    /// `None` recovers full causal attention (backward compatible
    /// default used by every existing test).
    pub window_size: Option<usize>,
}

impl MultiHeadSelfAttention {
    pub fn q_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    /// Forward one token at absolute position `pos`. Updates `kv` with
    /// the new K/V for this position. Returns a new hidden state of
    /// length `d_model`.
    ///
    /// The GPU path (selected when `backend.is_gpu()`) dispatches the Q/K/V
    /// and output projections through `backend.matmul_into`, writes K/V
    /// straight into VRAM via `backend.kv_cache_insert`, and runs attention
    /// via `backend.kv_attend` — no PCIe round-trip back to system RAM.
    /// The CPU path is the original paged-attention loop, byte-for-byte.
    pub fn forward(
        &self,
        x: &[f32],
        pos: usize,
        layer_idx: usize,
        kv: &mut KvCache,
        backend: &crate::backend::BackendBox,
    ) -> Vec<f32> {
        use crate::backend::{Backend, TensorView, TensorViewMut};

        debug_assert_eq!(x.len(), self.d_model);
        debug_assert_eq!(kv.kv_dim, self.kv_dim());

        // ── Helpers: f32 ↔ f16 conversion at the backend boundary ────────────
        // These allocations happen once per token (not in the inner loop) and
        // are therefore outside the hot-path allocation budget.
        let to_f16 = |v: &[f32]| -> Vec<f16> {
            v.iter().map(|&f| f16::from_f32(f)).collect()
        };
        let to_f32 = |v: &[f16]| -> Vec<f32> {
            v.iter().map(|h| h.to_f32()).collect()
        };

        let x_f16 = to_f16(x);

        // ── 1) Project Q, K, V via backend ───────────────────────────────────
        let wq_f16 = to_f16(&self.wq);
        let wk_f16 = to_f16(&self.wk);
        let wv_f16 = to_f16(&self.wv);

        let q_dim  = self.q_dim();
        let kv_dim = self.kv_dim();

        let mut q_f16  = vec![f16::ZERO; q_dim];
        let mut k_f16  = vec![f16::ZERO; kv_dim];
        let mut v_f16  = vec![f16::ZERO; kv_dim];

        backend.matmul_into(
            TensorView { data: &wq_f16, rows: q_dim,  cols: self.d_model },
            TensorView { data: &x_f16,  rows: self.d_model, cols: 1 },
            &mut TensorViewMut { data: &mut q_f16, rows: q_dim, cols: 1 },
        ).expect("Q projection failed");

        backend.matmul_into(
            TensorView { data: &wk_f16, rows: kv_dim, cols: self.d_model },
            TensorView { data: &x_f16,  rows: self.d_model, cols: 1 },
            &mut TensorViewMut { data: &mut k_f16, rows: kv_dim, cols: 1 },
        ).expect("K projection failed");

        backend.matmul_into(
            TensorView { data: &wv_f16, rows: kv_dim, cols: self.d_model },
            TensorView { data: &x_f16,  rows: self.d_model, cols: 1 },
            &mut TensorViewMut { data: &mut v_f16, rows: kv_dim, cols: 1 },
        ).expect("V projection failed");

        // ── 2) Apply RoPE in f32 (cheap; stays on CPU regardless of backend) ─
        let mut q = to_f32(&q_f16);
        let mut k = to_f32(&k_f16);

        for h in 0..self.num_heads {
            let s = h * self.head_dim;
            apply_rope_inplace(&mut q[s..s + self.head_dim], pos, self.rope_base);
        }
        for h in 0..self.num_kv_heads {
            let s = h * self.head_dim;
            apply_rope_inplace(&mut k[s..s + self.head_dim], pos, self.rope_base);
        }

        // ── 3) KV insert + attention ──────────────────────────────────────────
        let k_f16_rope = to_f16(&k);
        let v_f16_rope = v_f16; // V is not RoPE'd

        let mut attn_out = vec![0.0f32; q_dim];

        if backend.is_gpu() {
            // GPU path: K and V written directly into VRAM; attention kernel
            // runs over VRAM-resident KV — zero round-trip to system RAM.
            backend.kv_cache_insert(
                layer_idx,
                pos,
                TensorView { data: &k_f16_rope, rows: 1, cols: kv_dim },
                TensorView { data: &v_f16_rope, rows: 1, cols: kv_dim },
            ).expect("kv_cache_insert failed");

            // Keep the CPU-side paged KV cache in sync so any downstream
            // consumer that still reads `kv` directly observes the same
            // sequence length and per-position K/V bytes the GPU sees.
            let v_for_cpu = to_f32(&v_f16_rope);
            kv.append(&k, &v_for_cpu);

            // seq_len after the insert = pos + 1
            let seq_len = pos + 1;
            let q_f16_rope = to_f16(&q);
            let mut out_f16 = vec![f16::ZERO; q_dim];

            backend.kv_attend(
                layer_idx,
                TensorView { data: &q_f16_rope, rows: self.num_heads, cols: self.head_dim },
                seq_len,
                &mut TensorViewMut { data: &mut out_f16, rows: self.num_heads, cols: self.head_dim },
            ).expect("kv_attend failed");

            attn_out = to_f32(&out_f16);
        } else {
            // CPU path: existing paged-attention loop unchanged.
            let v = to_f32(&v_f16_rope);
            kv.append(&k, &v);

            let scale   = 1.0 / (self.head_dim as f32).sqrt();
            let t_max   = kv.seq_len;
            let t_start = match self.window_size {
                Some(w) if w > 0 => t_max.saturating_sub(w),
                _ => 0,
            };

            for h in 0..self.num_heads {
                let kv_head = h * self.num_kv_heads / self.num_heads;
                let q_head  = &q[h * self.head_dim..(h + 1) * self.head_dim];
                let span    = t_max - t_start;
                let mut scores = Vec::with_capacity(span);

                for t in t_start..t_max {
                    let k_t = kv.key(t);
                    let k_h = &k_t[kv_head * self.head_dim..(kv_head + 1) * self.head_dim];
                    let mut s = 0.0f32;
                    for j in 0..self.head_dim { s += q_head[j] * k_h[j]; }
                    scores.push(s * scale);
                }
                softmax_inplace(&mut scores);

                let out_h = &mut attn_out[h * self.head_dim..(h + 1) * self.head_dim];
                for (idx, score) in scores.iter().enumerate() {
                    let t   = t_start + idx;
                    let v_t = kv.value(t);
                    let v_h = &v_t[kv_head * self.head_dim..(kv_head + 1) * self.head_dim];
                    for j in 0..self.head_dim { out_h[j] += score * v_h[j]; }
                }
            }
        }

        // ── 4) Output projection via backend ──────────────────────────────────
        let wo_f16      = to_f16(&self.wo);
        let attn_f16    = to_f16(&attn_out);
        let mut out_f16 = vec![f16::ZERO; self.d_model];

        backend.matmul_into(
            TensorView { data: &wo_f16,   rows: self.d_model, cols: q_dim },
            TensorView { data: &attn_f16, rows: q_dim,        cols: 1 },
            &mut TensorViewMut { data: &mut out_f16, rows: self.d_model, cols: 1 },
        ).expect("output projection failed");

        to_f32(&out_f16)
    }
}

/// Combine the per-token outputs of `k` selected experts using the gating
/// scores. `outputs[i]` and `scores[i]` must be aligned (same expert).
/// Scores must already be softmax-normalised over the chosen top-K set.
///
/// Thin wrapper over [`crate::inference::combine_outputs`] (the canonical
/// MoE combiner) that ignores the redundant `d_model` argument; kept for
/// backwards compatibility with `TransformerLayer::moe_combine`.
pub fn combine_moe_outputs(outputs: &[HiddenState], scores: &[f32], d_model: usize) -> HiddenState {
    debug_assert!(
        outputs.iter().all(|o| o.len() == d_model),
        "every expert output must have length d_model"
    );
    let _ = d_model;
    crate::inference::combine_outputs(outputs, scores)
}

/// Run one expert FFN by reinterpreting its on-disk bytes (already loaded
/// into the resident buffer) as `[gate || up || down]` SwiGLU weights and
/// applying the SwiGLU forward over `x`. This is the bridge from
/// "bytes streamed from SSD" to "expert output vector" used by the
/// transformer layer's MoE block.
pub fn run_expert_forward(
    resident: &ExpertResident,
    x: &[f32],
    d_model: usize,
    d_ff: usize,
) -> Result<HiddenState, crate::inference::ExpertWeightsError> {
    let weights = ExpertWeights::from_bytes(resident.data(), d_model, d_ff)?;
    Ok(weights.forward(x))
}

/// Row-major matrix-vector multiply: `y = W · x` where `W` is
/// `[rows, cols]` row-major. Returns a fresh `Vec<f32>` of length `rows`.
///
/// **Auto-escalation (gist Task 1).** Three layers of dispatch, in
/// order:
///
/// 1. `--features blas` → BLAS-shaped `matrixmultiply` SGEMV
///    microkernel (the `ndarray`-style tuned path).
/// 2. Otherwise, **runtime row-parallel** when the matrix is large
///    enough to amortise thread-spawn overhead (`rows*cols ≥ 8 KiB`).
///    This path is now always compiled — the `--features simd` flag is
///    no longer required and is retained only for backwards
///    compatibility (it is a no-op).
/// 3. Scalar fallback otherwise.
pub fn matmul_row_major(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    // Note: the historic `simd`/`blas` mutual-exclusion `compile_error!`
    // is no longer needed — `simd` is now a deprecated no-op feature
    // (the runtime row-parallel path is always compiled). The `blas`
    // branch still wins when present.
    #[cfg(feature = "blas")]
    {
        // BLAS-equivalent SGEMV via `matrixmultiply::sgemm`: treat the
        // matrix-vector product as a `(rows × cols) × (cols × 1)`
        // SGEMM. The crate's tuned microkernel (the same one used by
        // `ndarray`'s `dot`) gives ~order-of-magnitude speedups over
        // the scalar loop on AVX2 / NEON for the dense projections in
        // `TransformerLayer`.
        let mut y = vec![0.0f32; rows];
        // SAFETY: matrixmultiply::sgemm is defined as taking pointers
        // to row-major (m × k) and (k × n) matrices and writing into a
        // row-major (m × n) output. We satisfy all aliasing and bounds
        // requirements: `w` is a borrowed slice of exactly `rows*cols`
        // floats, `x` is `cols` floats, and `y` is a fresh `rows`-
        // length buffer that doesn't alias either input. `rsa` / `csa`
        // etc. are the row/col strides for the three matrices; `1`
        // everywhere selects row-major.
        unsafe {
            matrixmultiply::sgemm(
                rows,
                cols,
                1,
                1.0,
                w.as_ptr(),
                cols as isize,
                1,
                x.as_ptr(),
                1,
                1,
                0.0,
                y.as_mut_ptr(),
                1,
                1,
            );
        }
        return y;
    }
    // Runtime row-parallel path. Always compiled (no `#[cfg(feature =
    // "simd")]` gate) so a single binary auto-escalates on any host
    // with enough cores.
    #[cfg(not(feature = "blas"))]
    {
        if rows * cols >= 8 * 1024 {
            return matmul_row_major_parallel(w, x, rows, cols);
        }
    }
    let mut y = vec![0.0f32; rows];
    for i in 0..rows {
        let row = &w[i * cols..(i + 1) * cols];
        let mut acc = 0.0f32;
        for j in 0..cols {
            acc += row[j] * x[j];
        }
        y[i] = acc;
    }
    y
}

/// Row-parallel matmul using `std::thread::scope`. Each worker computes a
/// contiguous block of output rows; no synchronisation is required because
/// the output rows are disjoint. Always compiled — gist Task 1's
/// "auto-escalation" requirement means we can't hide this behind a
/// cargo feature any more.
#[cfg(not(feature = "blas"))]
fn matmul_row_major_parallel(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let nthreads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(rows.max(1));
    if nthreads <= 1 {
        // Fall back to scalar for tiny outputs.
        let mut y = vec![0.0f32; rows];
        for i in 0..rows {
            let row = &w[i * cols..(i + 1) * cols];
            let mut acc = 0.0f32;
            for j in 0..cols {
                acc += row[j] * x[j];
            }
            y[i] = acc;
        }
        return y;
    }
    let mut y = vec![0.0f32; rows];
    let chunk = rows.div_ceil(nthreads);
    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(nthreads);
        for (chunk_idx, out_chunk) in y.chunks_mut(chunk).enumerate() {
            let row_start = chunk_idx * chunk;
            let w_slice = &w[row_start * cols..(row_start + out_chunk.len()) * cols];
            let x_ref = x;
            handles.push(s.spawn(move || {
                for (i, slot) in out_chunk.iter_mut().enumerate() {
                    let row = &w_slice[i * cols..(i + 1) * cols];
                    let mut acc = 0.0f32;
                    for j in 0..cols {
                        acc += row[j] * x_ref[j];
                    }
                    *slot = acc;
                }
            }));
        }
        for h in handles {
            let _ = h.join();
        }
    });
    y
}

/// Final language-modelling head: a linear projection from the residual
/// stream `[d_model]` to per-token logits `[vocab_size]`. In real models
/// this is sometimes weight-tied with the input embedding; we keep them
/// separate so the engine can sanity-check sampling without an embedding
/// matrix.
#[derive(Debug, Clone)]
pub struct LMHead {
    pub weights: Vec<f32>,
    pub vocab_size: usize,
    pub d_model: usize,
}

impl LMHead {
    pub fn new(weights: Vec<f32>, vocab_size: usize, d_model: usize) -> Self {
        assert_eq!(
            weights.len(),
            vocab_size * d_model,
            "lm_head weights must be [vocab_size, d_model]"
        );
        Self { weights, vocab_size, d_model }
    }

    /// Compute logits = `W · hidden`.
    pub fn forward(&self, hidden: &[f32]) -> Vec<f32> {
        matmul_row_major(&self.weights, hidden, self.vocab_size, self.d_model)
    }

    /// One-shot: project `hidden` to logits and sample a next-token id
    /// using the given [`crate::sampling::SamplingParams`]. The
    /// `position` is folded into the sampler's per-step seed so a
    /// `(seed, position)` pair always yields the same token — see
    /// `crate::sampling` for the deterministic-decode contract.
    pub fn sample(
        &self,
        hidden: &[f32],
        params: &crate::sampling::SamplingParams,
        position: u64,
    ) -> u32 {
        let logits = self.forward(hidden);
        crate::sampling::sample(&logits, params, position)
    }
}

/// Element-wise residual add: `y = a + b`.
#[inline]
pub fn add_residual(a: &[f32], b: &[f32]) -> Vec<f32> {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// One Llama / Mixtral-style transformer decoder layer.
///
/// Holds the dense (resident) weights — RMSNorms, attention projections,
/// and the routing gate — but **not** the expert FFN weights themselves.
/// Those are streamed from SSD per token by the engine's
/// [`crate::expert_cache::ExpertCache`] and handed back here as already-
/// loaded `ExpertResident`s for the [`Self::moe_combine`] step.
///
/// The layer is intentionally split into three sync helpers
/// ([`Self::attn_block`], [`Self::moe_pre`], [`Self::moe_combine`])
/// rather than one monolithic `forward` because expert loading is
/// **async** (it issues `pread(2)` to NVMe). The async driver in
/// `crate::model::RealModel::step` calls `attn_block`, then `moe_pre`,
/// then `await`s expert fetches via the engine, then calls
/// `moe_combine` to fold the per-expert FFN outputs back into the
/// residual stream — exactly the pseudocode the gist gives.
#[derive(Debug, Clone)]
pub struct TransformerLayer {
    pub rms_attn: RmsNorm,
    pub attn: MultiHeadSelfAttention,
    pub rms_moe: RmsNorm,
    pub gate: crate::gating::LinearGate,
}

impl TransformerLayer {
    /// `hidden -> rmsnorm -> attention -> residual`. Updates `kv` with
    /// the K/V for this token. `layer_idx` and `backend` are threaded
    /// through to [`MultiHeadSelfAttention::forward`] so the GPU path
    /// can route K/V writes to the correct VRAM layer slice.
    pub fn attn_block(
        &self,
        hidden: &[f32],
        pos: usize,
        layer_idx: usize,
        kv: &mut KvCache,
        backend: &crate::backend::BackendBox,
    ) -> Vec<f32> {
        let normed = self.rms_attn.forward(hidden);
        let attn_out = self.attn.forward(&normed, pos, layer_idx, kv, backend);
        add_residual(hidden, &attn_out)
    }

    /// `hidden -> rmsnorm -> gate.route()`. Returns the normalised
    /// hidden state (which is what every expert FFN should consume) and
    /// the routing decision.
    pub fn moe_pre(&self, hidden: &[f32]) -> (Vec<f32>, crate::gating::RoutingDecision) {
        let normed = self.rms_moe.forward(hidden);
        let routing = self.gate.route(&normed);
        (normed, routing)
    }

    /// Fold the per-expert FFN outputs back into the residual stream:
    /// `hidden + sum_i weights[i] * expert_outputs[i]`. The lengths of
    /// `expert_outputs` and `weights` must match (one per chosen expert);
    /// any expert that failed to materialise on disk should be filtered
    /// out of *both* slices upstream so the weighted sum stays
    /// well-defined.
    pub fn moe_combine(
        &self,
        hidden: &[f32],
        expert_outputs: &[HiddenState],
        weights: &[f32],
    ) -> Vec<f32> {
        let moe = combine_moe_outputs(expert_outputs, weights, self.attn.d_model);
        add_residual(hidden, &moe)
    }
}

/// Numerically-stable softmax, in place.
pub fn softmax_inplace(v: &mut [f32]) {
    if v.is_empty() {
        return;
    }
    let mut max = v[0];
    for &x in v.iter() {
        if x > max {
            max = x;
        }
    }
    let mut sum = 0.0f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
}

// =====================================================================
// Dense backbone abstraction (gist Part 2, fix #6).
// =====================================================================

/// Trait abstraction over the dense ("backbone") compute pieces of a
/// transformer decoder — `attn_block`, `RmsNorm`, and `LMHead`.
///
/// **Why**: the dense backbone is O(N²) attention math plus a couple
/// of dense matmuls. Today every byte of it runs on the CPU through
/// the scalar `transformer.rs` implementation (auto-escalated to
/// AVX-512 by the [`crate::kernels`] dispatcher inside the
/// [`crate::backend::Backend`] math facade). The gist's Part 2 calls
/// for a *clean seam* so an opt-in heterogeneous executor (a
/// `cudarc` / `wgpu` GpuBackend) can take over the dense body while
/// the SSD-streaming MoE path stays CPU-side. Pinned-host residuals
/// cross the host/device boundary exactly once per attention block,
/// not on every row of math.
///
/// **Where it plugs in**: [`crate::backend::Backend`] already owns
/// the per-row matmul / SwiGLU / softmax primitives (the GpuBackend
/// implementation overrides those). `DenseBackbone` is the layer
/// *above* that: it composes those primitives into the named blocks
/// the gist enumerates (`attn_block`, `RmsNorm`, `LMHead`) so a GPU
/// executor can fuse them into a single device-side kernel launch
/// instead of paying per-primitive host/device boundary cost.
///
/// The default implementation [`CpuBackbone`] just delegates to the
/// inherent methods on [`TransformerLayer`], [`RmsNorm`], and
/// [`LMHead`] — i.e. the existing CPU path. A future GPU backbone
/// implementation lives in `backend/mod.rs` next to [`crate::backend::GpuBackend`].
pub trait DenseBackbone: Send + Sync {
    /// Short human-readable name (e.g. `"cpu"`, `"cuda-0"`,
    /// `"wgpu-vulkan"`). Used by the startup log so operators can see
    /// which backbone is live alongside the math [`crate::backend::Backend`]
    /// identifier.
    fn name(&self) -> &'static str;

    /// `hidden → rmsnorm → attention → residual`. Equivalent to
    /// `layer.attn_block(hidden, pos, layer_idx, kv, backend)` on the CPU
    /// path. A GPU implementation can launch a single fused kernel here.
    fn attn_block(
        &self,
        layer: &TransformerLayer,
        hidden: &[f32],
        pos: usize,
        layer_idx: usize,
        kv: &mut KvCache,
        backend: &crate::backend::BackendBox,
    ) -> Vec<f32>;

    /// RMSNorm. Equivalent to `norm.forward(x)`. The default impl
    /// delegates to the inherent method; a device-side
    /// implementation overrides this to keep the residual on-device.
    fn rmsnorm(&self, norm: &RmsNorm, x: &[f32]) -> Vec<f32> {
        norm.forward(x)
    }

    /// Project the final hidden state to logits via the LM head.
    /// Equivalent to `head.forward(hidden)`. A GPU implementation
    /// runs the final `[vocab × d_model]` matmul on-device and only
    /// transfers the `vocab`-long logits vector back to the host.
    fn lm_head(&self, head: &LMHead, hidden: &[f32]) -> Vec<f32> {
        head.forward(hidden)
    }
}

/// Default CPU backbone: every method delegates to the existing
/// inherent implementation. Adding this is a no-op for the CPU
/// runtime — it's the trait wrapper around the existing methods so
/// callers can be ported to `DenseBackbone` without losing behaviour.
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuBackbone;

impl DenseBackbone for CpuBackbone {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn attn_block(
        &self,
        layer: &TransformerLayer,
        hidden: &[f32],
        pos: usize,
        layer_idx: usize,
        kv: &mut KvCache,
        backend: &crate::backend::BackendBox,
    ) -> Vec<f32> {
        layer.attn_block(hidden, pos, layer_idx, kv, backend)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_backend() -> crate::backend::BackendBox {
        crate::backend::BackendBox::Cpu(crate::backend::CandleBackend::new())
    }

    #[test]
    fn rmsnorm_unit_weight_normalises_to_unit_variance() {
        let n = 8;
        let weight = vec![1.0f32; n];
        let norm = RmsNorm::new(weight, 1e-6);
        let mut x: Vec<f32> = (0..n).map(|i| i as f32 - 3.5).collect();
        norm.forward_inplace(&mut x);
        let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
        // After RMSNorm with unit weight and tiny eps, mean(x^2) ≈ 1.
        assert!((mean_sq - 1.0).abs() < 1e-3, "mean_sq={mean_sq}");
    }

    #[test]
    fn rope_pos_zero_is_identity() {
        let mut v: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let original = v.clone();
        apply_rope_inplace(&mut v, 0, 10000.0);
        for (a, b) in v.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5, "rope at pos 0 must be identity");
        }
    }

    #[test]
    fn rope_preserves_norm() {
        let mut v: Vec<f32> = (1..=16).map(|i| i as f32 * 0.1).collect();
        let n_before: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        apply_rope_inplace(&mut v, 7, 10000.0);
        let n_after: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n_before - n_after).abs() < 1e-4);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut v = vec![1.0, 2.0, 3.0, -1.0];
        softmax_inplace(&mut v);
        let sum: f32 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax sum={sum}");
        // Largest input -> largest output.
        assert!(v[2] > v[1] && v[1] > v[0] && v[0] > v[3]);
    }

    #[test]
    fn softmax_handles_empty() {
        let mut v: Vec<f32> = Vec::new();
        softmax_inplace(&mut v);
        assert!(v.is_empty());
    }

    #[test]
    fn attention_shapes_match_and_cache_grows() {
        let d_model = 8;
        let num_heads = 2;
        let head_dim = 4;
        let num_kv_heads = 2;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        // Use deterministic small weights so we exercise the math.
        let mk = |rows: usize, cols: usize| {
            (0..rows * cols).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect()
        };
        let attn = MultiHeadSelfAttention {
            d_model,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_base: 10000.0,
            wq: mk(q_dim, d_model),
            wk: mk(kv_dim, d_model),
            wv: mk(kv_dim, d_model),
            wo: mk(d_model, q_dim),
            window_size: None,
        };
        let mut kv = KvCache::new(kv_dim);
        let x: Vec<f32> = (0..d_model).map(|i| 0.1 * i as f32).collect();
        let y0 = attn.forward(&x, 0, 0, &mut kv, &cpu_backend());
        assert_eq!(y0.len(), d_model);
        assert_eq!(kv.seq_len, 1);
        let y1 = attn.forward(&x, 1, 0, &mut kv, &cpu_backend());
        assert_eq!(y1.len(), d_model);
        assert_eq!(kv.seq_len, 2);
        // Output must be finite.
        assert!(y1.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn combine_moe_outputs_is_weighted_sum() {
        let d = 4;
        let outs = vec![vec![1.0; d], vec![2.0; d], vec![4.0; d]];
        let scores = vec![0.5, 0.25, 0.25];
        let y = combine_moe_outputs(&outs, &scores, d);
        // 0.5*1 + 0.25*2 + 0.25*4 = 2.0
        for v in y {
            assert!((v - 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn lm_head_projects_to_vocab() {
        let d_model = 4;
        let vocab = 6;
        // Identity-ish: first d_model rows are I, rest are zero.
        let mut w = vec![0.0f32; vocab * d_model];
        for i in 0..d_model.min(vocab) {
            w[i * d_model + i] = 1.0;
        }
        let head = LMHead::new(w, vocab, d_model);
        let logits = head.forward(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(logits.len(), vocab);
        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[1], 2.0);
        assert_eq!(logits[2], 3.0);
        assert_eq!(logits[3], 4.0);
        // Rows beyond d_model are all zero.
        assert_eq!(logits[4], 0.0);
        assert_eq!(logits[5], 0.0);
    }

    #[test]
    fn add_residual_is_elementwise() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![10.0, 20.0, 30.0];
        let y = add_residual(&a, &b);
        assert_eq!(y, vec![11.0, 22.0, 33.0]);
    }

    /// Build a tiny `TransformerLayer` with deterministic small weights
    /// so we can exercise the full `attn_block + moe_pre + moe_combine`
    /// path without loading anything from disk.
    fn make_layer(d_model: usize, num_experts: usize, top_k: usize) -> TransformerLayer {
        let head_dim = 4;
        let num_heads = d_model / head_dim;
        let num_kv_heads = num_heads;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let mk = |rows: usize, cols: usize, scale: f32| {
            (0..rows * cols)
                .map(|i| ((i % 7) as f32 - 3.0) * scale)
                .collect::<Vec<f32>>()
        };
        TransformerLayer {
            rms_attn: RmsNorm::new(vec![1.0; d_model], 1e-6),
            attn: MultiHeadSelfAttention {
                d_model,
                num_heads,
                num_kv_heads,
                head_dim,
                rope_base: 10000.0,
                wq: mk(q_dim, d_model, 0.05),
                wk: mk(kv_dim, d_model, 0.05),
                wv: mk(kv_dim, d_model, 0.05),
                wo: mk(d_model, q_dim, 0.05),
                window_size: None,
            },
            rms_moe: RmsNorm::new(vec![1.0; d_model], 1e-6),
            gate: crate::gating::LinearGate::new(
                mk(num_experts, d_model, 0.1),
                num_experts,
                d_model,
                top_k,
            ),
        }
    }

    #[test]
    fn transformer_layer_attn_block_is_finite_and_grows_kv() {
        let d_model = 16;
        let layer = make_layer(d_model, 4, 2);
        let mut kv = KvCache::new(layer.attn.kv_dim());
        let x: Vec<f32> = (0..d_model).map(|i| 0.1 * i as f32 - 0.5).collect();
        let y0 = layer.attn_block(&x, 0, 0, &mut kv, &cpu_backend());
        assert_eq!(y0.len(), d_model);
        assert!(y0.iter().all(|v| v.is_finite()));
        assert_eq!(kv.seq_len, 1);
        let y1 = layer.attn_block(&y0, 1, 0, &mut kv, &cpu_backend());
        assert_eq!(kv.seq_len, 2);
        assert!(y1.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn transformer_layer_moe_pre_routes_top_k_experts() {
        let d_model = 16;
        let layer = make_layer(d_model, 8, 2);
        let x: Vec<f32> = (0..d_model).map(|i| 0.1 * i as f32 - 0.5).collect();
        let (normed, routing) = layer.moe_pre(&x);
        assert_eq!(normed.len(), d_model);
        assert!(normed.iter().all(|v| v.is_finite()));
        assert_eq!(routing.experts.len(), 2);
        assert_eq!(routing.weights.len(), 2);
        let sum: f32 = routing.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn transformer_layer_moe_combine_blends_outputs() {
        let d_model = 8;
        let layer = make_layer(d_model, 4, 2);
        let hidden = vec![1.0; d_model];
        // Two expert outputs of all-ones and all-twos with equal weights:
        // combined moe = 1.5 * 1.0 vector; hidden + moe = 2.5 vector.
        let outs = vec![vec![1.0; d_model], vec![2.0; d_model]];
        let weights = vec![0.5, 0.5];
        let y = layer.moe_combine(&hidden, &outs, &weights);
        for v in y {
            assert!((v - 2.5).abs() < 1e-6);
        }
    }

    /// Build a tiny attention block with controllable window. Uses
    /// identity-like Q/K projections so we can read the V contribution
    /// of position 0 directly.
    fn make_window_attn(window: Option<usize>) -> MultiHeadSelfAttention {
        let d_model = 4;
        let num_heads = 1;
        let num_kv_heads = 1;
        let head_dim = 4;
        // Identity for q/k/v/o (vector of d_model^2 with diagonal 1.0).
        let identity = |dim: usize| -> Vec<f32> {
            let mut w = vec![0.0f32; dim * dim];
            for i in 0..dim {
                w[i * dim + i] = 1.0;
            }
            w
        };
        MultiHeadSelfAttention {
            d_model,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_base: 10000.0,
            wq: identity(d_model),
            wk: identity(d_model),
            wv: identity(d_model),
            wo: identity(d_model),
            window_size: window,
        }
    }

    /// With `window_size = Some(2)`, position 3 must NOT attend to
    /// position 0 — the window covers only positions [2, 3]. The
    /// attention output for a query whose key would otherwise dominate
    /// at t=0 (a unique large signal there) must therefore *not* reflect
    /// that signal once we step past the window.
    #[test]
    fn sliding_window_excludes_positions_outside_span() {
        let attn = make_window_attn(Some(2));
        let mut kv = KvCache::new(attn.kv_dim());
        // Distinct token at position 0; the rest are ~zero.
        let big = vec![10.0f32, 0.0, 0.0, 0.0];
        let small = vec![0.0f32, 0.0, 0.0, 0.0];
        let _ = attn.forward(&big, 0, 0, &mut kv, &cpu_backend());
        let _ = attn.forward(&small, 1, 0, &mut kv, &cpu_backend());
        let _ = attn.forward(&small, 2, 0, &mut kv, &cpu_backend());
        // At pos 3 with window 2 the visible KV span is [2, 3] → t=0 must
        // not contribute. Output should reflect (mostly) the zero tokens.
        let y = attn.forward(&small, 3, 0, &mut kv, &cpu_backend());
        // The big spike at t=0 had a 10.0 in dim 0; if we *were*
        // attending to it the output's dim 0 would be > 1.0. With the
        // window excluding t=0 it must stay near 0.
        assert!(y[0].abs() < 1e-3, "leaked value from outside window: {y:?}");

        // Sanity check: with full attention (window = None), the same
        // pattern leaks the t=0 spike into the output (proves the
        // fixture is non-degenerate).
        let attn_full = make_window_attn(None);
        let mut kv2 = KvCache::new(attn_full.kv_dim());
        let _ = attn_full.forward(&big, 0, &mut kv2);
        let _ = attn_full.forward(&small, 1, &mut kv2);
        let _ = attn_full.forward(&small, 2, &mut kv2);
        let y_full = attn_full.forward(&small, 3, &mut kv2);
        assert!(y_full[0] > 0.5, "full attention should see t=0 spike: {y_full:?}");
    }

    #[test]
    fn sliding_window_inside_span_behaves_like_full_attention() {
        // For a window larger than the sequence, results must match
        // unrestricted attention bit-for-bit.
        let attn_w = make_window_attn(Some(10));
        let attn_n = make_window_attn(None);
        let mut kv1 = KvCache::new(attn_w.kv_dim());
        let mut kv2 = KvCache::new(attn_n.kv_dim());
        let xs = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];
        for (pos, x) in xs.iter().enumerate() {
            let y_w = attn_w.forward(x, pos, &mut kv1);
            let y_n = attn_n.forward(x, pos, &mut kv2);
            for (a, b) in y_w.iter().zip(y_n.iter()) {
                assert!((a - b).abs() < 1e-5);
            }
        }
    }

    // ----------------- PagedAttention block-storage tests -------------

    #[test]
    fn paged_kv_cache_grows_one_block_per_block_tokens() {
        let kv = KvCache::new(8);
        assert_eq!(kv.num_blocks(), 0);
        let mut kv = kv;
        // Insert exactly PAGED_BLOCK_TOKENS tokens — should fit in one block.
        for _ in 0..PAGED_BLOCK_TOKENS {
            kv.append(&[1.0; 8], &[2.0; 8]);
        }
        assert_eq!(kv.seq_len, PAGED_BLOCK_TOKENS);
        assert_eq!(kv.num_blocks(), 1);
        // One more token forces a new block.
        kv.append(&[3.0; 8], &[4.0; 8]);
        assert_eq!(kv.num_blocks(), 2);
        // The just-appended token should round-trip via `key`/`value`.
        let last = kv.seq_len - 1;
        assert_eq!(kv.key(last), &[3.0; 8][..]);
        assert_eq!(kv.value(last), &[4.0; 8][..]);
        // And the first token in the previous block should still match.
        assert_eq!(kv.key(0), &[1.0; 8][..]);
    }

    #[test]
    fn paged_kv_cache_reset_releases_blocks() {
        let mut kv = KvCache::new(4);
        for _ in 0..(PAGED_BLOCK_TOKENS * 2 + 3) {
            kv.append(&[1.0; 4], &[2.0; 4]);
        }
        assert!(kv.num_blocks() >= 3);
        kv.reset();
        assert_eq!(kv.seq_len, 0);
        assert_eq!(kv.num_blocks(), 0);
    }

    #[test]
    fn paged_kv_cache_attention_matches_legacy_layout() {
        // Build an attention block, run a few tokens through it, and
        // verify the per-token output is unchanged from what a flat
        // KV cache would have produced. Since the block layout is
        // accessed only through `key(i)`/`value(i)` — which return
        // slices identical to what the old flat `Vec<f32>` would
        // have — the block index just has to stay correct as we
        // cross block boundaries. Walk past at least one boundary.
        let d_model = 4;
        let head_dim = 2;
        let num_heads = 2;
        let num_kv_heads = 2;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let mk = |rows: usize, cols: usize| -> Vec<f32> {
            (0..rows * cols)
                .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
                .collect()
        };
        let attn = MultiHeadSelfAttention {
            d_model,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_base: 10000.0,
            wq: mk(q_dim, d_model),
            wk: mk(kv_dim, d_model),
            wv: mk(kv_dim, d_model),
            wo: mk(d_model, q_dim),
            window_size: None,
        };
        let mut kv = KvCache::new(kv_dim);
        // Walk past the first block boundary to exercise multi-block
        // indexing.
        let xs: Vec<Vec<f32>> = (0..(PAGED_BLOCK_TOKENS + 3))
            .map(|t| (0..d_model).map(|j| 0.05 * (t as f32) + 0.01 * (j as f32)).collect())
            .collect();
        let mut last = vec![0.0f32; d_model];
        for (pos, x) in xs.iter().enumerate() {
            last = attn.forward(x, pos, 0, &mut kv, &cpu_backend());
        }
        assert_eq!(kv.seq_len, xs.len());
        assert_eq!(kv.num_blocks(), 2);
        assert!(last.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn zeroize_blocks_clears_every_element() {
        // Mix block sizes (including empty and non-power-of-two
        // lengths) to exercise the inner `for i in 0..len` loop's
        // bounds and the iteration over multiple blocks.
        let mut blocks: Vec<Box<[f32]>> = vec![
            vec![1.0f32; 16].into_boxed_slice(),
            vec![-2.5f32; 7].into_boxed_slice(),
            vec![f32::INFINITY; 1].into_boxed_slice(),
            Vec::<f32>::new().into_boxed_slice(),
            vec![std::f32::consts::PI; 33].into_boxed_slice(),
        ];
        // Sanity: at least one non-zero element exists before zeroising.
        assert!(blocks.iter().any(|b| b.iter().any(|&v| v != 0.0)));

        zeroize_blocks(&mut blocks);

        for (i, b) in blocks.iter().enumerate() {
            for (j, &v) in b.iter().enumerate() {
                assert_eq!(
                    v.to_bits(),
                    0.0f32.to_bits(),
                    "block {i} element {j} not zeroised: {v}"
                );
            }
        }
    }
}

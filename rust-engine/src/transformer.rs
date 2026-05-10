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

/// One layer's KV cache (per head). Stores keys and values in
/// row-major `[seq_len, num_kv_heads * head_dim]` layout, growing as new
/// tokens are appended.
#[derive(Debug, Clone, Default)]
pub struct KvCache {
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    pub seq_len: usize,
    pub kv_dim: usize,
}

impl KvCache {
    pub fn new(kv_dim: usize) -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            seq_len: 0,
            kv_dim,
        }
    }

    pub fn append(&mut self, k: &[f32], v: &[f32]) {
        debug_assert_eq!(k.len(), self.kv_dim);
        debug_assert_eq!(v.len(), self.kv_dim);
        self.keys.extend_from_slice(k);
        self.values.extend_from_slice(v);
        self.seq_len += 1;
    }

    pub fn reset(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.seq_len = 0;
    }

    /// Get the i-th cached key as a slice of length `kv_dim`.
    fn key(&self, i: usize) -> &[f32] {
        &self.keys[i * self.kv_dim..(i + 1) * self.kv_dim]
    }

    fn value(&self, i: usize) -> &[f32] {
        &self.values[i * self.kv_dim..(i + 1) * self.kv_dim]
    }
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
    pub fn forward(&self, x: &[f32], pos: usize, kv: &mut KvCache) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.d_model);
        debug_assert_eq!(kv.kv_dim, self.kv_dim());

        // 1) Project Q, K, V.
        let mut q = matmul_row_major(&self.wq, x, self.q_dim(), self.d_model);
        let mut k = matmul_row_major(&self.wk, x, self.kv_dim(), self.d_model);
        let v = matmul_row_major(&self.wv, x, self.kv_dim(), self.d_model);

        // 2) Apply RoPE per-head to Q and K.
        for h in 0..self.num_heads {
            let s = h * self.head_dim;
            apply_rope_inplace(&mut q[s..s + self.head_dim], pos, self.rope_base);
        }
        for h in 0..self.num_kv_heads {
            let s = h * self.head_dim;
            apply_rope_inplace(&mut k[s..s + self.head_dim], pos, self.rope_base);
        }

        // 3) Append to KV cache.
        kv.append(&k, &v);

        // 4) Scaled dot-product attention per head.
        //    GQA: head h queries KV head (h * num_kv_heads / num_heads).
        //    Sliding window: when `self.window_size = Some(w)`, restrict
        //    `t` to the most recent `w` positions.
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let t_max = kv.seq_len; // includes the position we just appended
        let t_start = match self.window_size {
            Some(w) if w > 0 => t_max.saturating_sub(w),
            _ => 0,
        };
        let mut attn_out = vec![0.0f32; self.q_dim()];
        for h in 0..self.num_heads {
            let kv_head = h * self.num_kv_heads / self.num_heads;
            let q_head = &q[h * self.head_dim..(h + 1) * self.head_dim];

            // scores[t] = q · k_t * scale, for t in [t_start, t_max).
            let span = t_max - t_start;
            let mut scores = Vec::with_capacity(span);
            for t in t_start..t_max {
                let k_t = kv.key(t);
                let k_h = &k_t[kv_head * self.head_dim..(kv_head + 1) * self.head_dim];
                let mut s = 0.0f32;
                for j in 0..self.head_dim {
                    s += q_head[j] * k_h[j];
                }
                scores.push(s * scale);
            }
            softmax_inplace(&mut scores);

            // out[h] = sum_t scores[t-t_start] * v_t[kv_head]
            let out_h = &mut attn_out[h * self.head_dim..(h + 1) * self.head_dim];
            for (idx, score) in scores.iter().enumerate() {
                let t = t_start + idx;
                let v_t = kv.value(t);
                let v_h = &v_t[kv_head * self.head_dim..(kv_head + 1) * self.head_dim];
                for j in 0..self.head_dim {
                    out_h[j] += score * v_h[j];
                }
            }
        }

        // 5) Output projection.
        matmul_row_major(&self.wo, &attn_out, self.d_model, self.q_dim())
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
/// The `simd` cargo feature replaces the scalar loop with a row-parallel
/// implementation using `std::thread::scope` (no extra crate dependency),
/// which stands in for the rayon / candle / cudarc backend the gist
/// mentions in Phase 6. The scalar path remains the default — it has zero
/// extra deps and is the same shape every other matmul in this engine
/// uses, so behaviour is bit-for-bit unchanged unless you opt in.
pub fn matmul_row_major(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    // Compile-time guard: `simd` and `blas` are mutually exclusive
    // backend choices for this matmul. Building with both on at once
    // is almost certainly a configuration mistake.
    #[cfg(all(feature = "simd", feature = "blas"))]
    compile_error!(
        "cargo features `simd` and `blas` are mutually exclusive — pick one matmul backend"
    );
    #[cfg(feature = "blas")]
    {
        // BLAS-equivalent SGEMV via `matrixmultiply::sgemm`: treat the
        // matrix-vector product as a `(rows × cols) × (cols × 1)`
        // SGEMM. The crate's tuned microkernel (the same one used by
        // `ndarray`'s `dot`) gives ~order-of-magnitude speedups over
        // the scalar loop on AVX2 / NEON for the dense projections in
        // `TransformerLayer`.
        let mut y = vec![0.0f32; rows];
        // Safety: matrixmultiply::sgemm is defined as taking pointers
        // to row-major (m × k) and (k × n) matrices and writing into a
        // row-major (m × n) output; we satisfy all aliasing and bounds
        // requirements. `rsa` / `csa` etc. are the row/col strides for
        // the three matrices; `1` everywhere selects row-major.
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
    #[cfg(feature = "simd")]
    {
        // Heuristic: only parallelise when the matrix is large enough
        // that thread-spawn overhead doesn't dominate. The break-even
        // point is around ~32 KiB of work; below that the scalar path
        // is faster.
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
/// the output rows are disjoint. Used when the `simd` feature is on. A
/// future PR can swap the body for a `candle::Tensor` op or a CUDA kernel
/// without changing this function's signature — the call sites are
/// already routed through `matmul_row_major`.
#[cfg(feature = "simd")]
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
    /// the K/V for this token.
    pub fn attn_block(&self, hidden: &[f32], pos: usize, kv: &mut KvCache) -> Vec<f32> {
        let normed = self.rms_attn.forward(hidden);
        let attn_out = self.attn.forward(&normed, pos, kv);
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

#[cfg(test)]
mod tests {
    use super::*;

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
        let y0 = attn.forward(&x, 0, &mut kv);
        assert_eq!(y0.len(), d_model);
        assert_eq!(kv.seq_len, 1);
        let y1 = attn.forward(&x, 1, &mut kv);
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
        let y0 = layer.attn_block(&x, 0, &mut kv);
        assert_eq!(y0.len(), d_model);
        assert!(y0.iter().all(|v| v.is_finite()));
        assert_eq!(kv.seq_len, 1);
        let y1 = layer.attn_block(&y0, 1, &mut kv);
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
        let _ = attn.forward(&big, 0, &mut kv);
        let _ = attn.forward(&small, 1, &mut kv);
        let _ = attn.forward(&small, 2, &mut kv);
        // At pos 3 with window 2 the visible KV span is [2, 3] → t=0 must
        // not contribute. Output should reflect (mostly) the zero tokens.
        let y = attn.forward(&small, 3, &mut kv);
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
}

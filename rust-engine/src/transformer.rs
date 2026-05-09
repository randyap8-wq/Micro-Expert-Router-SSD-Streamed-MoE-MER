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
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut attn_out = vec![0.0f32; self.q_dim()];
        for h in 0..self.num_heads {
            let kv_head = h * self.num_kv_heads / self.num_heads;
            let q_head = &q[h * self.head_dim..(h + 1) * self.head_dim];

            // scores[t] = q · k_t * scale, for t in 0..=pos (causal).
            let t_max = kv.seq_len; // includes the position we just appended
            let mut scores = Vec::with_capacity(t_max);
            for t in 0..t_max {
                let k_t = kv.key(t);
                let k_h = &k_t[kv_head * self.head_dim..(kv_head + 1) * self.head_dim];
                let mut s = 0.0f32;
                for j in 0..self.head_dim {
                    s += q_head[j] * k_h[j];
                }
                scores.push(s * scale);
            }
            softmax_inplace(&mut scores);

            // out[h] = sum_t scores[t] * v_t[kv_head]
            let out_h = &mut attn_out[h * self.head_dim..(h + 1) * self.head_dim];
            for (t, score) in scores.iter().enumerate() {
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
pub fn combine_moe_outputs(outputs: &[HiddenState], scores: &[f32], d_model: usize) -> HiddenState {
    debug_assert_eq!(outputs.len(), scores.len());
    let mut y = vec![0.0f32; d_model];
    for (out, &s) in outputs.iter().zip(scores.iter()) {
        debug_assert_eq!(out.len(), d_model);
        for (yi, &oi) in y.iter_mut().zip(out.iter()) {
            *yi += s * oi;
        }
    }
    y
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
pub fn matmul_row_major(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
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
}

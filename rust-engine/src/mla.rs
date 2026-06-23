//! Multi-head Latent Attention (MLA) for DeepSeek-V3 / V3.1.
//!
//! DeepSeek-V3 replaces the standard multi-head attention block with
//! **multi-head latent attention**: instead of caching a full
//! `num_heads * head_dim` key/value tensor per token, it caches a small
//! *latent* vector that the per-head K/V are reconstructed from at
//! attention time. This is the single largest KV-cache memory saving in
//! the architecture and is the reason DeepSeek-V3 needs a bespoke
//! attention path — the generic [`crate::transformer::MultiHeadSelfAttention`]
//! cannot express it because its query and value head dims differ
//! (`qk_nope_head_dim + qk_rope_head_dim` for K vs `v_head_dim` for V),
//! and because the cached entry is the compressed latent rather than the
//! materialised K/V.
//!
//! ## Projection pipeline (per token)
//!
//! ```text
//!   x ─▶ q_a_proj ─▶ q_a_layernorm ─▶ q_b_proj ─▶ q  [n_h·(d_nope+d_rope)]
//!                                                  └─ split per head: q_nope | q_pe(RoPE)
//!
//!   x ─▶ kv_a_proj_with_mqa ─▶ [ compressed_kv (kv_lora) | k_pe (d_rope) ]
//!                                  │                          └─ RoPE (shared across heads)
//!                                  └─ kv_a_layernorm
//!
//!   cache ◀── latent = [ compressed_kv ; k_pe ]          (width kv_lora + d_rope)
//!
//!   per cached t:  kv_b_proj(compressed_kv_t) ─▶ per head: k_nope_t | v_t
//!                  k_t = [ k_nope_t ; k_pe_t ]   (k_pe shared across heads)
//!
//!   attn_h = softmax( q_h · k_t · softmax_scale )_t · v_t   [d_v]
//!   out    = o_proj( concat_h attn_h )                       [d_model]
//! ```
//!
//! When `q_lora_rank == 0` (DeepSeek-V2-Lite) the query path is a single
//! dense `q_proj` instead of the `q_a → layernorm → q_b` low-rank pair;
//! this module handles both shapes.
//!
//! The cache stores the post-layernorm `compressed_kv` and the
//! post-RoPE `k_pe` so that reconstruction is a single `kv_b_proj`
//! matmul; both halves are written into the existing paged
//! [`crate::transformer::KvCache`] key slot (its value slot is unused by
//! MLA), which keeps the per-request, per-layer cache plumbing in
//! `RealModel` unchanged.

use crate::transformer::{
    apply_rope_maybe_scaled, matmul_row_major, yarn_get_mscale, KvCache, RmsNorm, YarnRope,
};

/// Per-token softmax over `scores` in place (numerically stable).
fn softmax_inplace(scores: &mut [f32]) {
    if scores.is_empty() {
        return;
    }
    let mut max = f32::NEG_INFINITY;
    let mut saw_nan = false;
    for &s in scores.iter() {
        if s.is_nan() {
            saw_nan = true;
        } else if s > max {
            max = s;
        }
    }
    // Non-finite fallback: a stray `NaN`, a `+inf`, or a fully-masked row
    // (every score `-inf`, leaving `max == -inf`) cannot yield a meaningful
    // distribution, so emit a uniform distribution rather than letting NaN
    // propagate downstream.
    if saw_nan || !max.is_finite() {
        let uniform = 1.0 / scores.len() as f32;
        scores.iter_mut().for_each(|s| *s = uniform);
        return;
    }
    let mut sum = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - max).exp();
        sum += *s;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for s in scores.iter_mut() {
            *s *= inv;
        }
    }
}

/// Multi-head latent attention block (DeepSeek-V3).
///
/// All projection weights are stored row-major as `[out_features,
/// in_features]`, matching the HuggingFace `*.weight` layout and the
/// engine's [`matmul_row_major`] convention.
#[derive(Debug, Clone)]
pub struct MultiHeadLatentAttention {
    pub d_model: usize,
    pub num_heads: usize,
    /// Low-rank query compression dim. `0` selects the direct `q_proj`
    /// path (no `q_a`/`q_b` LoRA pair).
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub rope_base: f32,
    /// Attention logit scale. Defaults to `1/sqrt(qk_nope+qk_rope)`; a
    /// YaRN-scaled checkpoint folds its `mscale` correction in here.
    pub softmax_scale: f32,
    /// Optional YaRN long-context RoPE scaling applied to the
    /// `qk_rope_head_dim` rotary portion of Q and the shared `k_pe`.
    /// `None` keeps the standard `1/base^(2i/d)` rotation.
    pub rope_yarn: Option<YarnRope>,

    /// `[q_lora_rank, d_model]` — present only when `q_lora_rank > 0`.
    pub q_a_proj: Vec<f32>,
    /// RMSNorm over the `q_lora_rank` latent — present only when
    /// `q_lora_rank > 0`.
    pub q_a_layernorm: Option<RmsNorm>,
    /// `q_lora_rank > 0`: `[num_heads*(qk_nope+qk_rope), q_lora_rank]`.
    /// `q_lora_rank == 0`: `[num_heads*(qk_nope+qk_rope), d_model]`.
    pub q_b_proj: Vec<f32>,

    /// `[kv_lora_rank + qk_rope_head_dim, d_model]`.
    pub kv_a_proj_with_mqa: Vec<f32>,
    /// RMSNorm over the `kv_lora_rank` compressed latent.
    pub kv_a_layernorm: RmsNorm,
    /// `[num_heads*(qk_nope+v_head_dim), kv_lora_rank]`.
    pub kv_b_proj: Vec<f32>,

    /// `[d_model, num_heads*v_head_dim]`.
    pub o_proj: Vec<f32>,
}

impl MultiHeadLatentAttention {
    /// Per-head query/key dim (nope + rope portions concatenated).
    #[inline]
    pub fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// Width of the cached latent: `compressed_kv` followed by the shared
    /// `k_pe`. This is the `kv_dim` the layer's [`KvCache`] must be sized
    /// with.
    #[inline]
    pub fn latent_dim(&self) -> usize {
        self.kv_lora_rank + self.qk_rope_head_dim
    }

    /// Default softmax scale for the configured dims (no YaRN correction).
    pub fn default_softmax_scale(qk_nope_head_dim: usize, qk_rope_head_dim: usize) -> f32 {
        let d = (qk_nope_head_dim + qk_rope_head_dim) as f32;
        1.0 / d.sqrt()
    }

    /// YaRN-corrected softmax scale: the default `1/sqrt(d)` multiplied
    /// by `mscale^2` where `mscale = yarn_get_mscale(factor,
    /// mscale_all_dim)`. Mirrors the DeepSeek-V3 reference attention
    /// (`softmax_scale *= mscale * mscale` when `mscale_all_dim` is
    /// set); falls back to the default scale for non-YaRN configs.
    pub fn yarn_softmax_scale(
        qk_nope_head_dim: usize,
        qk_rope_head_dim: usize,
        scaling: Option<&crate::architecture::RopeScaling>,
    ) -> f32 {
        let base = Self::default_softmax_scale(qk_nope_head_dim, qk_rope_head_dim);
        match scaling {
            Some(s)
                if s.rope_type.eq_ignore_ascii_case("yarn")
                    && s.factor > 1.0
                    && s.mscale_all_dim != 0.0 =>
            {
                let m = yarn_get_mscale(s.factor, s.mscale_all_dim);
                base * m * m
            }
            _ => base,
        }
    }

    /// Project the query into per-head `[q_nope | q_pe]` rows and apply
    /// RoPE to the `q_pe` portion of each head. Returns the full
    /// `num_heads * qk_head_dim` vector.
    fn project_q(&self, x: &[f32], pos: usize) -> Vec<f32> {
        let qk = self.qk_head_dim();
        let q_total = self.num_heads * qk;
        let mut q = if self.q_lora_rank > 0 {
            let mut a = matmul_row_major(&self.q_a_proj, x, self.q_lora_rank, self.d_model);
            if let Some(norm) = self.q_a_layernorm.as_ref() {
                norm.forward_inplace(&mut a);
            }
            matmul_row_major(&self.q_b_proj, &a, q_total, self.q_lora_rank)
        } else {
            matmul_row_major(&self.q_b_proj, x, q_total, self.d_model)
        };
        // RoPE on the rope portion of each head (the trailing
        // `qk_rope_head_dim` slots).
        for h in 0..self.num_heads {
            let base = h * qk + self.qk_nope_head_dim;
            apply_rope_maybe_scaled(
                &mut q[base..base + self.qk_rope_head_dim],
                pos,
                self.rope_base,
                self.rope_yarn.as_ref(),
            );
        }
        q
    }

    /// Project + cache the latent KV for the current token. Returns the
    /// freshly computed `compressed_kv` (post-layernorm) and `k_pe`
    /// (post-RoPE) for symmetry with the cached entries; the caller does
    /// not need them (it re-reads from the cache), but exposing them
    /// keeps the function testable.
    pub fn project_and_cache_kv(&self, x: &[f32], pos: usize, kv: &mut KvCache) {
        let proj_dim = self.kv_lora_rank + self.qk_rope_head_dim;
        let kv_a = matmul_row_major(&self.kv_a_proj_with_mqa, x, proj_dim, self.d_model);
        let mut latent = vec![0.0f32; self.latent_dim()];
        // compressed_kv (post-layernorm)
        {
            let mut compressed = kv_a[..self.kv_lora_rank].to_vec();
            self.kv_a_layernorm.forward_inplace(&mut compressed);
            latent[..self.kv_lora_rank].copy_from_slice(&compressed);
        }
        // k_pe (post-RoPE), shared across heads.
        {
            let mut k_pe = kv_a[self.kv_lora_rank..].to_vec();
            apply_rope_maybe_scaled(&mut k_pe, pos, self.rope_base, self.rope_yarn.as_ref());
            latent[self.kv_lora_rank..].copy_from_slice(&k_pe);
        }
        // The value slot is unused by MLA; store the latent in both to
        // satisfy the cache's symmetric append invariant.
        kv.append(&latent, &latent);
    }

    /// Reconstruct per-head `(k_nope, v)` for a cached latent entry by
    /// running `kv_b_proj` over its `compressed_kv` half. Returns
    /// `(kv_b, k_pe)` where `kv_b` is `num_heads*(qk_nope+v_head_dim)`
    /// row-major and `k_pe` is the shared rope key (`qk_rope_head_dim`).
    fn reconstruct_kv<'a>(&self, latent: &'a [f32]) -> (Vec<f32>, &'a [f32]) {
        let compressed = &latent[..self.kv_lora_rank];
        let k_pe = &latent[self.kv_lora_rank..];
        let out_dim = self.num_heads * (self.qk_nope_head_dim + self.v_head_dim);
        let kv_b = matmul_row_major(&self.kv_b_proj, compressed, out_dim, self.kv_lora_rank);
        (kv_b, k_pe)
    }

    /// Forward one token at absolute position `pos`. Appends this token's
    /// latent to `kv` and returns a new hidden state of length
    /// `d_model`.
    ///
    /// `pos` must equal `kv.seq_len` on entry (strict one-token-at-a-time
    /// decode), mirroring the contract of
    /// [`crate::transformer::MultiHeadSelfAttention::forward`].
    pub fn forward(&self, x: &[f32], pos: usize, kv: &mut KvCache) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.d_model);
        debug_assert_eq!(kv.kv_dim, self.latent_dim());

        let qk = self.qk_head_dim();
        let nope = self.qk_nope_head_dim;
        let d_v = self.v_head_dim;

        // 1) Query projection (+ per-head RoPE on the pe portion).
        let q = self.project_q(x, pos);

        // 2) Latent KV projection + cache append.
        self.project_and_cache_kv(x, pos, kv);
        let t_max = kv.seq_len; // includes the token we just appended

        // 3) Causal attention over all cached tokens.
        let mut attn_out = vec![0.0f32; self.num_heads * d_v];
        // Reconstruct each cached token's per-head K/V once (O(seq)
        // matmuls total) and reuse the reconstructed rows across all heads,
        // rather than re-running `kv_b_proj` per (head, token).
        let kv_b_row = self.qk_nope_head_dim + self.v_head_dim;
        let mut kv_b_rows: Vec<Vec<f32>> = Vec::with_capacity(t_max);
        let mut k_pe_rows: Vec<Vec<f32>> = Vec::with_capacity(t_max);
        for t in 0..t_max {
            let latent = kv.key_at(t);
            let (kv_b, k_pe) = self.reconstruct_kv(latent);
            kv_b_rows.push(kv_b);
            k_pe_rows.push(k_pe.to_vec());
        }
        for h in 0..self.num_heads {
            let q_h = &q[h * qk..(h + 1) * qk];
            let mut scores = Vec::with_capacity(t_max);
            for t in 0..t_max {
                let kv_b = &kv_b_rows[t];
                let k_pe = &k_pe_rows[t];
                let head_base = h * kv_b_row;
                let k_nope = &kv_b[head_base..head_base + nope];
                // score = q_h · [k_nope ; k_pe]
                let mut s = 0.0f32;
                for j in 0..nope {
                    s += q_h[j] * k_nope[j];
                }
                for j in 0..self.qk_rope_head_dim {
                    s += q_h[nope + j] * k_pe[j];
                }
                scores.push(s * self.softmax_scale);
            }
            softmax_inplace(&mut scores);
            let out_h = &mut attn_out[h * d_v..(h + 1) * d_v];
            for (t, score) in scores.iter().enumerate() {
                let head_base = h * kv_b_row;
                let v_t = &kv_b_rows[t][head_base + nope..head_base + nope + d_v];
                for j in 0..d_v {
                    out_h[j] += score * v_t[j];
                }
            }
        }

        // 4) Output projection.
        matmul_row_major(&self.o_proj, &attn_out, self.d_model, self.num_heads * d_v)
    }
}

// ---------------------------------------------------------------------
// FP8 (e4m3) block-wise dequantisation.
//
// DeepSeek-V3 ships its dense + attention weights in FP8 `e4m3`
// (1 sign / 4 exponent / 3 mantissa) accompanied by a companion
// `*.weight_scale_inv` tensor: a per-block f32 reciprocal scale laid out
// as `[ceil(rows/block), ceil(cols/block)]` row-major. Dequantising a
// weight element is `fp8_value * scale_inv[block_row, block_col]`.
// ---------------------------------------------------------------------

/// Decode one FP8 `e4m3` byte (1-4-3, bias 7, no infinities; `0xFF`/`0x7F`
/// are NaN) to f32. Matches the OCP `e4m3` / DeepSeek `float8_e4m3fn`
/// definition: the all-exponent-ones encoding is *not* infinity, the max
/// finite magnitude is 448.
pub fn f8_e4m3_to_f32(b: u8) -> f32 {
    let sign = if (b & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((b >> 3) & 0x0F) as i32;
    let mant = (b & 0x07) as u32;
    if exp == 0 {
        if mant == 0 {
            return sign * 0.0;
        }
        // Subnormal: value = mant/8 * 2^(1-bias), bias = 7.
        let m = mant as f32 / 8.0;
        return sign * m * 2f32.powi(1 - 7);
    }
    if exp == 0x0F && mant == 0x07 {
        // e4m3fn reserves S.1111.111 for NaN; decode to 0.0 (a neutral
        // contribution) rather than propagating NaN or injecting an
        // extreme ±448 outlier into downstream matmuls.
        return 0.0;
    }
    // Normal: value = (1 + mant/8) * 2^(exp-bias).
    let m = 1.0 + mant as f32 / 8.0;
    sign * m * 2f32.powi(exp - 7)
}

/// Block-wise dequantise an FP8 `e4m3` weight matrix to f32.
///
/// * `q` — `rows * cols` FP8 bytes, row-major.
/// * `scale_inv` — `ceil(rows/block) * ceil(cols/block)` f32 reciprocal
///   scales, row-major over the block grid.
/// * `block` — square block edge (DeepSeek uses 128).
///
/// Returns the dequantised `rows * cols` f32 matrix, row-major. Returns
/// an empty vector when the shapes are inconsistent so the caller can
/// fall back to seeded init rather than panic on a malformed checkpoint.
pub fn dequant_fp8_e4m3_blockwise(
    q: &[u8],
    scale_inv: &[f32],
    rows: usize,
    cols: usize,
    block: usize,
) -> Vec<f32> {
    if block == 0 || q.len() != rows * cols {
        return Vec::new();
    }
    let block_cols = cols.div_ceil(block);
    let block_rows = rows.div_ceil(block);
    if scale_inv.len() != block_rows * block_cols {
        return Vec::new();
    }
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let br = r / block;
        for c in 0..cols {
            let bc = c / block;
            let scale = scale_inv[br * block_cols + bc];
            out[r * cols + c] = f8_e4m3_to_f32(q[r * cols + c]) * scale;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_all_neg_inf_is_uniform() {
        // A fully-masked row must degrade to a uniform distribution rather
        // than propagating NaNs from `(-inf) - (-inf)`.
        let mut scores = vec![f32::NEG_INFINITY; 3];
        softmax_inplace(&mut scores);
        let expected = 1.0 / 3.0;
        assert!(
            scores.iter().all(|&s| (s - expected).abs() < 1e-6),
            "got {scores:?}"
        );
    }

    #[test]
    fn softmax_mixed_nan_is_uniform() {
        // A stray NaN with an otherwise-finite max must not poison the row.
        // The previous `!max.is_finite()` guard missed this because
        // `f32::max` ignores NaN, leaving a finite max.
        let mut scores = vec![0.0, f32::NAN, 1.0];
        softmax_inplace(&mut scores);
        let expected = 1.0 / 3.0;
        assert!(
            scores
                .iter()
                .all(|&s| s.is_finite() && (s - expected).abs() < 1e-6),
            "got {scores:?}"
        );
    }

    /// Build a small, deterministic MLA block for shape/behaviour tests.
    fn tiny_mla(q_lora_rank: usize) -> MultiHeadLatentAttention {
        let d_model = 8;
        let num_heads = 2;
        let kv_lora_rank = 4;
        let qk_nope_head_dim = 2;
        let qk_rope_head_dim = 2;
        let v_head_dim = 3;
        let qk = qk_nope_head_dim + qk_rope_head_dim;
        let q_total = num_heads * qk;
        // Deterministic, small weights via a simple LCG.
        let mut seed = 0x1234_5678u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / u32::MAX as f32 - 0.5) * 0.1
        };
        let vecf = |n: usize, f: &mut dyn FnMut() -> f32| (0..n).map(|_| f()).collect::<Vec<_>>();
        let (q_a_proj, q_a_layernorm, q_b_proj) = if q_lora_rank > 0 {
            (
                vecf(q_lora_rank * d_model, &mut next),
                Some(RmsNorm::new(vec![1.0; q_lora_rank], 1e-6)),
                vecf(q_total * q_lora_rank, &mut next),
            )
        } else {
            (Vec::new(), None, vecf(q_total * d_model, &mut next))
        };
        MultiHeadLatentAttention {
            d_model,
            num_heads,
            q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            rope_base: 10000.0,
            rope_yarn: None,
            softmax_scale: MultiHeadLatentAttention::default_softmax_scale(
                qk_nope_head_dim,
                qk_rope_head_dim,
            ),
            q_a_proj,
            q_a_layernorm,
            q_b_proj,
            kv_a_proj_with_mqa: vecf((kv_lora_rank + qk_rope_head_dim) * d_model, &mut next),
            kv_a_layernorm: RmsNorm::new(vec![1.0; kv_lora_rank], 1e-6),
            kv_b_proj: vecf(num_heads * (qk_nope_head_dim + v_head_dim) * kv_lora_rank, &mut next),
            o_proj: vecf(d_model * num_heads * v_head_dim, &mut next),
        }
    }

    #[test]
    fn latent_dim_matches_cache_width() {
        let mla = tiny_mla(6);
        assert_eq!(mla.latent_dim(), mla.kv_lora_rank + mla.qk_rope_head_dim);
        let mut kv = KvCache::new(mla.latent_dim());
        let x = vec![0.05f32; mla.d_model];
        let y = mla.forward(&x, 0, &mut kv);
        assert_eq!(y.len(), mla.d_model);
        assert_eq!(kv.seq_len, 1);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_advances_cache_and_stays_finite() {
        for q_lora in [0usize, 6] {
            let mla = tiny_mla(q_lora);
            let mut kv = KvCache::new(mla.latent_dim());
            for pos in 0..5 {
                let x: Vec<f32> = (0..mla.d_model).map(|i| 0.02 * (i as f32 + pos as f32)).collect();
                let y = mla.forward(&x, pos, &mut kv);
                assert_eq!(y.len(), mla.d_model);
                assert!(y.iter().all(|v| v.is_finite()), "non-finite output at pos {pos}");
            }
            assert_eq!(kv.seq_len, 5);
        }
    }

    #[test]
    fn single_token_attention_is_value_projection() {
        // With one token, softmax over a single score is 1.0, so the
        // attention output is exactly that token's reconstructed V per
        // head, projected through o_proj. Verify against a direct
        // recomputation.
        let mla = tiny_mla(6);
        let mut kv = KvCache::new(mla.latent_dim());
        let x = vec![0.07f32; mla.d_model];
        let y = mla.forward(&x, 0, &mut kv);

        // Recompute expected output independently.
        let latent = kv.key_at(0).to_vec();
        let (kv_b, _k_pe) = mla.reconstruct_kv(&latent);
        let kv_b_row = mla.qk_nope_head_dim + mla.v_head_dim;
        let d_v = mla.v_head_dim;
        let mut attn = vec![0.0f32; mla.num_heads * d_v];
        for h in 0..mla.num_heads {
            let v_t = &kv_b[h * kv_b_row + mla.qk_nope_head_dim..h * kv_b_row + kv_b_row];
            attn[h * d_v..(h + 1) * d_v].copy_from_slice(v_t);
        }
        let expected =
            matmul_row_major(&mla.o_proj, &attn, mla.d_model, mla.num_heads * d_v);
        for (a, b) in y.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "got {a}, expected {b}");
        }
    }

    #[test]
    fn fp8_e4m3_decodes_reference_values() {
        // Exact, well-known e4m3 encodings.
        assert_eq!(f8_e4m3_to_f32(0x00), 0.0); // +0
        assert_eq!(f8_e4m3_to_f32(0x38), 1.0); // exp=7 (bias) mant=0 -> 1.0
        assert_eq!(f8_e4m3_to_f32(0x40), 2.0); // exp=8 -> 2.0
        assert_eq!(f8_e4m3_to_f32(0xB8), -1.0); // sign + 1.0
        assert_eq!(f8_e4m3_to_f32(0x34), 0.75); // exp=6 mant=4 -> (1+0.5)*0.5
    }

    #[test]
    fn fp8_blockwise_dequant_applies_per_block_scale() {
        // 2x2 matrix, block=1 so each element has its own scale.
        let q = vec![0x38u8, 0x40, 0x38, 0x38]; // values 1, 2, 1, 1
        let scale = vec![2.0f32, 3.0, 4.0, 5.0];
        let out = dequant_fp8_e4m3_blockwise(&q, &scale, 2, 2, 1);
        assert_eq!(out, vec![2.0, 6.0, 4.0, 5.0]);
    }

    #[test]
    fn fp8_blockwise_dequant_shares_scale_within_block() {
        // 2x2 matrix, block=2 -> a single shared scale.
        let q = vec![0x38u8, 0x40, 0x38, 0x40]; // 1, 2, 1, 2
        let scale = vec![10.0f32];
        let out = dequant_fp8_e4m3_blockwise(&q, &scale, 2, 2, 2);
        assert_eq!(out, vec![10.0, 20.0, 10.0, 20.0]);
    }

    #[test]
    fn fp8_blockwise_dequant_rejects_bad_shapes() {
        assert!(dequant_fp8_e4m3_blockwise(&[0u8; 3], &[1.0], 2, 2, 1).is_empty());
        assert!(dequant_fp8_e4m3_blockwise(&[0u8; 4], &[1.0], 2, 2, 1).is_empty());
    }

    fn yarn_scaling(factor: f32, mscale: f32, mscale_all_dim: f32) -> crate::architecture::RopeScaling {
        crate::architecture::RopeScaling {
            rope_type: "yarn".to_string(),
            factor,
            original_max_position_embeddings: 4096,
            beta_fast: 32.0,
            beta_slow: 1.0,
            mscale,
            mscale_all_dim,
        }
    }

    #[test]
    fn yarn_softmax_scale_applies_mscale_squared() {
        // DeepSeek-V3 config: factor=40, mscale=1.0, mscale_all_dim=1.0.
        let s = yarn_scaling(40.0, 1.0, 1.0);
        let base = MultiHeadLatentAttention::default_softmax_scale(128, 64);
        let scaled = MultiHeadLatentAttention::yarn_softmax_scale(128, 64, Some(&s));
        let m = yarn_get_mscale(40.0, 1.0);
        assert!((scaled - base * m * m).abs() < 1e-7, "got {scaled}");
        // mscale_all_dim == 0 keeps the default scale (reference impl
        // only corrects when mscale_all_dim is set).
        let s0 = yarn_scaling(40.0, 1.0, 0.0);
        assert_eq!(MultiHeadLatentAttention::yarn_softmax_scale(128, 64, Some(&s0)), base);
        // No scaling config at all keeps the default.
        assert_eq!(MultiHeadLatentAttention::yarn_softmax_scale(128, 64, None), base);
    }

    #[test]
    fn mla_forward_with_yarn_stays_finite_and_differs_from_unscaled() {
        // Larger-magnitude, position-varying inputs so attention scores
        // are far from uniform — the YaRN mscale^2 softmax correction
        // then visibly reweights the mixture.
        let x_at = |d_model: usize, pos: usize| -> Vec<f32> {
            (0..d_model).map(|i| 0.5 * (i as f32 + 1.0) + pos as f32).collect()
        };
        let mut mla = tiny_mla(6);
        let mut kv_plain = KvCache::new(mla.latent_dim());
        let plain: Vec<Vec<f32>> = (0..4)
            .map(|pos| mla.forward(&x_at(mla.d_model, pos), pos, &mut kv_plain))
            .collect();

        let s = yarn_scaling(40.0, 1.0, 1.0);
        mla.rope_yarn = YarnRope::from_scaling(mla.qk_rope_head_dim, mla.rope_base, &s);
        assert!(mla.rope_yarn.is_some());
        mla.softmax_scale = MultiHeadLatentAttention::yarn_softmax_scale(
            mla.qk_nope_head_dim,
            mla.qk_rope_head_dim,
            Some(&s),
        );
        let mut kv_yarn = KvCache::new(mla.latent_dim());
        let mut any_diff = false;
        for pos in 0..4 {
            let y = mla.forward(&x_at(mla.d_model, pos), pos, &mut kv_yarn);
            assert!(y.iter().all(|v| v.is_finite()), "non-finite at pos {pos}");
            if y.iter().zip(plain[pos].iter()).any(|(a, b)| (a - b).abs() > 1e-6) {
                any_diff = true;
            }
        }
        assert!(any_diff, "YaRN must alter MLA attention outputs");
    }
}

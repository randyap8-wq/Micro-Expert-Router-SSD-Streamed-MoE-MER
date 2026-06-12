//! Token sampling for the real-transformer path.
//!
//! Replaces the original deterministic `argmax` next-token selection
//! with a configurable softmax-with-temperature + top-K + top-P
//! (nucleus) sampler. The same primitives drive the OpenAI-compatible
//! HTTP request fields (`temperature`, `top_p`, `top_k`, `seed`) and
//! the TOML `[sampling]` defaults.
//!
//! ### Determinism
//!
//! Sampling is seeded by `(seed, position)`: the same `(prompt, seed,
//! max_tokens)` triple produces the same completion bit-for-bit. This
//! is the same property `tokenizer.json`-driven byte fallback gives,
//! so a benchmarking harness can pin reproducible runs without giving
//! up on stochastic decoding entirely. `temperature == 0.0` forces
//! greedy `argmax` regardless of the other parameters (matching how
//! OpenAI / vLLM treat "deterministic" sampling).
//!
//! ### Budget
//!
//! For Mixtral-scale `vocab_size` (~32k) the top-K + top-P pass is
//! a partial-sort + small softmax; cost is negligible relative to a
//! full transformer step. We keep it scalar (no SIMD / BLAS) — like
//! the rest of `inference.rs` and `transformer.rs`, the SSD-streamed
//! expert FFNs are the bottleneck this engine optimises.

use serde::{Deserialize, Serialize};

/// Knobs that control how next-token logits are turned into a token id.
///
/// All fields have neutral defaults: `temperature = 1.0`, no top-K /
/// top-P truncation, `seed = 0`. With `temperature == 0.0` the sampler
/// degrades to greedy `argmax` — useful for reproducible benchmarks /
/// regression tests, and matches the legacy `RealModel::step` behaviour
/// before this module existed.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Softmax temperature. `0.0` (or any non-positive value) → greedy
    /// argmax. Mainstream values: `0.7` (creative) … `1.0` (default).
    pub temperature: f32,

    /// Top-P (nucleus) cumulative-mass cutoff. `1.0` (default) disables
    /// the truncation. `0.9` keeps the smallest set of tokens whose
    /// cumulative softmax probability is `≥ 0.9`.
    pub top_p: f32,

    /// Top-K truncation: keep only the `K` highest-probability tokens.
    /// `0` (default) disables. Combined with `top_p`, the more
    /// restrictive of the two takes effect.
    pub top_k: usize,

    /// Sampling RNG seed. The same `(seed, position)` pair always
    /// produces the same token, so a request is reproducible.
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self { temperature: 1.0, top_p: 1.0, top_k: 0, seed: 0 }
    }
}

impl SamplingParams {
    /// `temperature = 0` — greedy argmax. Equivalent to the original
    /// `RealModel::step` behaviour and used by every existing test.
    pub fn greedy() -> Self {
        Self { temperature: 0.0, top_p: 1.0, top_k: 0, seed: 0 }
    }

    /// True if this is greedy sampling (any non-positive temperature).
    pub fn is_greedy(&self) -> bool {
        !(self.temperature > 0.0)
    }

    /// Per-step RNG seed, derived from `(seed, position)`. The
    /// `splitmix64` step ensures small `position` deltas produce well-
    /// separated states even when `seed == 0`.
    #[inline]
    pub fn step_seed(&self, position: u64) -> u64 {
        let mut z = self
            .seed
            .wrapping_add(0x9E3779B97F4A7C15)
            .wrapping_add(position.wrapping_mul(0xBF58476D1CE4E5B9));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

/// Sample one token id from `logits` using the given parameters.
///
/// Pipeline:
/// 1. If `is_greedy`, return `argmax(logits)` directly.
/// 2. Apply temperature scaling: `logits[i] /= temperature`.
/// 3. Numerically-stable softmax over scaled logits.
/// 4. If `top_k > 0`, zero out all but the top-K probabilities.
/// 5. If `top_p < 1.0`, keep the smallest prefix (sorted descending)
///    whose cumulative mass `≥ top_p`; zero the rest.
/// 6. Renormalise and draw via inverse-CDF over `step_seed(position)`.
pub fn sample(logits: &[f32], params: &SamplingParams, position: u64) -> u32 {
    if logits.is_empty() {
        return 0;
    }
    if params.is_greedy() {
        return argmax(logits);
    }

    // 1) temperature-scaled softmax (numerically stable).
    let t = params.temperature.max(1e-6);
    let mut max = f32::NEG_INFINITY;
    for &v in logits {
        let s = v / t;
        if s > max {
            max = s;
        }
    }
    let mut probs: Vec<f32> = logits.iter().map(|&v| ((v / t) - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if !(sum > 0.0) {
        return argmax(logits);
    }
    for p in probs.iter_mut() {
        *p /= sum;
    }

    // 2) Build a sorted-descending index permutation. `vocab` ≤ 256k in
    // practice; `O(N log N)` is fine for one token.
    let mut order: Vec<usize> = (0..probs.len()).collect();
    // `total_cmp` provides the total order `sort_unstable_by` requires
    // (Rust 1.81+ panics on comparators that violate it, which the old
    // `partial_cmp().unwrap_or(Equal)` did when NaNs were present).
    // Comparing `b` against `a` keeps the sort descending.
    order.sort_unstable_by(|&a, &b| probs[b].total_cmp(&probs[a]));

    // 3) Top-K: discard everything past index K.
    let k_cut = if params.top_k == 0 { order.len() } else { params.top_k.min(order.len()) };

    // 4) Top-P: walk the sorted list until cumulative mass ≥ top_p.
    //    `top_p < 1.0` only — clamp pathological values into [0, 1].
    let p_cut = if !(params.top_p > 0.0 && params.top_p < 1.0) {
        k_cut
    } else {
        let target = params.top_p.clamp(1e-6, 1.0);
        let mut cum = 0.0f32;
        let mut idx = 0usize;
        // Always keep at least the top-1 token so a degenerate
        // (`top_p = 0`) request still has something to sample from.
        for (i, &o) in order.iter().take(k_cut).enumerate() {
            cum += probs[o];
            idx = i + 1;
            if cum >= target {
                break;
            }
        }
        idx.min(k_cut)
    };

    // 5) Zero out anything beyond the cut and renormalise the rest.
    let kept = &order[..p_cut];
    let kept_sum: f32 = kept.iter().map(|&i| probs[i]).sum();
    if !(kept_sum > 0.0) {
        // Numerical edge case: fall back to argmax.
        return argmax(logits);
    }

    // 6) Inverse-CDF draw using a `splitmix64`-derived uniform `[0, 1)`.
    let bits = params.step_seed(position);
    let u = ((bits >> 40) as u32) as f32 / ((1u32 << 24) as f32) * kept_sum;
    let mut acc = 0.0f32;
    for &i in kept {
        acc += probs[i];
        if u <= acc {
            return i as u32;
        }
    }
    // Fallback: numerical drift — return the last kept index.
    *kept.last().unwrap_or(&0) as u32
}

#[inline]
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i as u32;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_returns_argmax() {
        let logits = vec![0.1, 5.0, 2.0, 4.9];
        let id = sample(&logits, &SamplingParams::greedy(), 0);
        assert_eq!(id, 1);
    }

    #[test]
    fn temperature_zero_is_greedy_even_with_top_p_set() {
        let logits = vec![1.0, 2.0, 3.0];
        let p = SamplingParams { temperature: 0.0, top_p: 0.9, top_k: 2, seed: 7 };
        assert_eq!(sample(&logits, &p, 0), 2);
    }

    #[test]
    fn high_temperature_can_pick_lower_logit() {
        // With very high temperature the distribution flattens — over many
        // positions we should see at least one non-argmax token.
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let p = SamplingParams { temperature: 10.0, top_p: 1.0, top_k: 0, seed: 1 };
        let mut seen_non_argmax = false;
        for pos in 0..200 {
            if sample(&logits, &p, pos) != 3 {
                seen_non_argmax = true;
                break;
            }
        }
        assert!(seen_non_argmax, "high-T sampling never deviated from argmax");
    }

    #[test]
    fn top_k_one_collapses_to_argmax() {
        let logits = vec![0.1, 5.0, 2.0, 4.9];
        let p = SamplingParams { temperature: 1.0, top_p: 1.0, top_k: 1, seed: 42 };
        for pos in 0..32 {
            assert_eq!(sample(&logits, &p, pos), 1);
        }
    }

    #[test]
    fn top_p_truncation_excludes_tail() {
        // With top_p = 0.5 and a sharply-peaked distribution, only the
        // top-1 token's cumulative mass exceeds the cutoff — so even
        // with stochastic sampling we always pick it.
        let logits = vec![1.0, 1.0, 1.0, 10.0]; // softmax(10.0) ≈ 1.0
        let p = SamplingParams { temperature: 1.0, top_p: 0.5, top_k: 0, seed: 9 };
        for pos in 0..32 {
            assert_eq!(sample(&logits, &p, pos), 3);
        }
    }

    #[test]
    fn determinism_for_same_seed_and_position() {
        let logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let p = SamplingParams { temperature: 1.5, top_p: 0.95, top_k: 0, seed: 0xC0FFEE };
        for pos in 0..16 {
            let a = sample(&logits, &p, pos);
            let b = sample(&logits, &p, pos);
            assert_eq!(a, b, "non-deterministic sample at pos={pos}");
        }
    }

    #[test]
    fn empty_logits_returns_zero() {
        let p = SamplingParams::default();
        assert_eq!(sample(&[], &p, 0), 0);
    }

    #[test]
    fn negative_temperature_treated_as_greedy() {
        let logits = vec![1.0, 5.0, 3.0];
        let p = SamplingParams { temperature: -0.1, top_p: 1.0, top_k: 0, seed: 0 };
        assert!(p.is_greedy());
        assert_eq!(sample(&logits, &p, 0), 1);
    }

    #[test]
    fn step_seed_varies_with_position() {
        let p = SamplingParams { temperature: 1.0, top_p: 1.0, top_k: 0, seed: 1 };
        let s0 = p.step_seed(0);
        let s1 = p.step_seed(1);
        let s2 = p.step_seed(2);
        assert_ne!(s0, s1);
        assert_ne!(s1, s2);
    }
}

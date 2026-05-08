//! Mocked Mixtral-style router and Markov predictive prefetcher.
//!
//! In a real MoE the router is a learned gating network. For the purposes of
//! this engine — which is about *I/O*, not modelling — we mock it with two
//! components:
//!
//! 1. [`TopKRouter`] picks `k` distinct expert ids per token. Defaults to
//!    `k = 2` (Mixtral 8x7B). It supports a deterministic mode (used for
//!    smoke tests) and a stochastic mode that samples from a per-token
//!    Dirichlet-ish distribution biased toward a "specialist" subset.
//!
//! 2. [`PredictiveLoader`] is a first-order Markov model over expert ids,
//!    built from observed activations. Given the most recent expert it
//!    samples (weighted by transition probability) the top-N most likely
//!    successors and exposes them for asynchronous prefetch.
//!
//! The model is updated online as the engine runs, so the longer the stream,
//! the better the prefetch hit rate becomes.

use parking_lot::RwLock;
use rand::distributions::{Distribution, WeightedIndex};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::HashMap;

/// Picks the top-K experts for a given token. Distinct ids guaranteed.
pub struct TopKRouter {
    num_experts: u32,
    k: usize,
    rng: parking_lot::Mutex<StdRng>,
}

impl TopKRouter {
    pub fn new(num_experts: u32, k: usize, seed: u64) -> Self {
        assert!(k as u32 <= num_experts, "k must be <= num_experts");
        assert!(k > 0);
        Self {
            num_experts,
            k,
            rng: parking_lot::Mutex::new(StdRng::seed_from_u64(seed)),
        }
    }

    /// Force a specific routing decision (used by the CLI to reproduce the
    /// "Router selects Expert ID 3 and 7" example from the spec).
    pub fn fixed(&self, ids: &[u32]) -> Vec<u32> {
        ids.iter()
            .filter(|i| **i < self.num_experts)
            .copied()
            .collect()
    }

    /// Sample `k` distinct expert ids using a weighted distribution. The
    /// distribution favours a stable "specialist" subset (the first 25% of
    /// experts get higher weight) so prefetch has signal to learn from.
    pub fn route(&self, token_idx: u64) -> Vec<u32> {
        let n = self.num_experts as usize;
        let specialist_cutoff = (n / 4).max(1);
        // Build weights deterministically from token_idx so a run is replayable.
        let mut rng = self.rng.lock();
        let mut weights: Vec<f64> = (0..n)
            .map(|i| {
                let base = if i < specialist_cutoff { 4.0 } else { 1.0 };
                // Slow drift so different tokens activate slightly different experts.
                base + (((token_idx.wrapping_add(i as u64)) % 7) as f64) * 0.1
            })
            .collect();

        let mut chosen = Vec::with_capacity(self.k);
        for _ in 0..self.k {
            let dist = match WeightedIndex::new(&weights) {
                Ok(d) => d,
                Err(_) => break,
            };
            let idx = dist.sample(&mut *rng);
            chosen.push(idx as u32);
            // Zero out so we don't pick the same expert twice.
            weights[idx] = 0.0;
        }
        chosen
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn num_experts(&self) -> u32 {
        self.num_experts
    }
}

/// First-order Markov predictor: P(next | last_expert).
///
/// Transitions are stored as a **sparse** per-row map of observation counts:
/// `rows[from]` holds only the successor ids that have been observed at
/// least once, plus a cached row total. A flat `[N x N]` dense matrix would
/// allocate `O(N²)` u32 cells up front (~1 MiB at N=512, ~64 MiB at
/// N=4096) regardless of how many distinct transitions we actually
/// observe — sparse-by-row scales with the number of *visited* (from, to)
/// pairs instead, which is the natural footprint of an MoE routing trace
/// (each token activates only `k` experts so per-token only `k²` pairs
/// are observed).
///
/// Probabilities are still smoothed with a Laplace prior `prior` per
/// transition so unseen successors have non-zero probability and the
/// predictor doesn't need a warm-up phase. The prior is applied
/// implicitly: an absent map entry counts as `prior`, an entry with
/// stored count `c` counts as `c + prior`, and the row's effective total
/// is `total_observed[from] + num_experts * prior`.
struct Row {
    /// Observed counts beyond the prior; absent keys mean "0 observed".
    counts: HashMap<u32, u32>,
    /// Sum of `counts.values()` (i.e. observations beyond the prior).
    total_observed: u64,
}

impl Row {
    fn new() -> Self {
        Self { counts: HashMap::new(), total_observed: 0 }
    }
}

pub struct PredictiveLoader {
    num_experts: u32,
    /// `rows[from]` — sparse counts of successors of `from`.
    rows: RwLock<Vec<Row>>,
    /// Number of successors to suggest per call.
    fanout: usize,
    /// Probability threshold below which we don't bother prefetching.
    min_prob: f64,
    /// Prior added to every transition to smooth the empty-state case.
    prior: u32,
    rng: parking_lot::Mutex<StdRng>,
}

impl PredictiveLoader {
    pub fn new(num_experts: u32, fanout: usize, min_prob: f64, seed: u64) -> Self {
        let n = num_experts as usize;
        let mut rows = Vec::with_capacity(n);
        for _ in 0..n {
            rows.push(Row::new());
        }
        Self {
            num_experts,
            rows: RwLock::new(rows),
            fanout,
            min_prob,
            prior: 1,
            rng: parking_lot::Mutex::new(StdRng::seed_from_u64(seed.wrapping_add(0xDEAD))),
        }
    }

    /// Record that `to` was activated immediately after `from`.
    pub fn observe(&self, from: u32, to: u32) {
        if from >= self.num_experts || to >= self.num_experts {
            return;
        }
        let mut rows = self.rows.write();
        let row = &mut rows[from as usize];
        let entry = row.counts.entry(to).or_insert(0);
        let new_count = entry.saturating_add(1);
        // Only bump the row total when the cell didn't saturate; once a
        // cell is pinned at u32::MAX, further observations stop counting,
        // which matches the saturating semantics of the previous dense
        // implementation.
        if new_count != *entry {
            row.total_observed = row.total_observed.saturating_add(1);
            *entry = new_count;
        }
    }

    /// Record a whole batch of activations from a single token's expert set.
    /// Each previous-token expert transitions to each current-token expert.
    pub fn observe_step(&self, prev_set: &[u32], next_set: &[u32]) {
        for &p in prev_set {
            for &n in next_set {
                self.observe(p, n);
            }
        }
    }

    /// Effective total for a row including the smoothing prior over all `n`
    /// possible successors.
    #[inline]
    fn effective_total(&self, total_observed: u64) -> u64 {
        total_observed + (self.num_experts as u64) * (self.prior as u64)
    }

    /// Probability of an *unobserved* successor (i.e. one that has never
    /// fired from this `from`).
    #[inline]
    fn prior_prob(&self, total_observed: u64) -> f64 {
        let total = self.effective_total(total_observed);
        if total == 0 {
            0.0
        } else {
            self.prior as f64 / total as f64
        }
    }

    /// Predict up to `fanout` successors for `from`, weighted by transition
    /// probability. Returns `(expert_id, p)` pairs sorted by descending `p`,
    /// filtered by `min_prob`.
    pub fn predict_next(&self, from: u32) -> Vec<(u32, f64)> {
        if from >= self.num_experts || self.fanout == 0 {
            return Vec::new();
        }
        let n = self.num_experts as usize;
        let rows = self.rows.read();
        let row = &rows[from as usize];
        let total = self.effective_total(row.total_observed);
        if total == 0 {
            return Vec::new();
        }
        let total_f = total as f64;
        let prior_f = self.prior as f64;

        // Observed successors get their (count + prior) probability.
        let mut probs: Vec<(u32, f64)> = row
            .counts
            .iter()
            .map(|(&id, &c)| (id, (c as f64 + prior_f) / total_f))
            .filter(|&(_, p)| p >= self.min_prob)
            .collect();

        // If the smoothing prior alone clears `min_prob` and we still have
        // room in the fanout, fill in unseen successors (all tied at
        // `prior_prob`). This matches the dense implementation's behaviour
        // where every cell started at the prior. Iterate in id order for
        // determinism.
        let prior_p = prior_f / total_f;
        if prior_p >= self.min_prob && probs.len() < self.fanout {
            for id in 0..n as u32 {
                if probs.len() >= self.fanout {
                    break;
                }
                if !row.counts.contains_key(&id) {
                    probs.push((id, prior_p));
                }
            }
        }

        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        probs.truncate(self.fanout);
        probs
    }

    /// Sample `fanout` distinct successors of `from` using the weighted
    /// distribution (rather than a deterministic argmax). Useful when you
    /// want exploration in the prefetch policy.
    pub fn sample_next(&self, from: u32) -> Vec<u32> {
        if from >= self.num_experts || self.fanout == 0 {
            return Vec::new();
        }
        let n = self.num_experts as usize;
        let rows = self.rows.read();
        let row = &rows[from as usize];

        // Materialise the dense weight vector for this row only (one row,
        // not the full N×N matrix). `prior` for unseen, `c + prior` for
        // observed. Each call allocates `n` f64 — a bounded transient cost.
        let prior_f = self.prior as f64;
        let mut weights: Vec<f64> = vec![prior_f; n];
        for (&id, &c) in &row.counts {
            weights[id as usize] = c as f64 + prior_f;
        }
        drop(rows);

        let mut chosen = Vec::with_capacity(self.fanout);
        let mut rng = self.rng.lock();
        for _ in 0..self.fanout {
            let dist = match WeightedIndex::new(&weights) {
                Ok(d) => d,
                Err(_) => break,
            };
            let idx = dist.sample(&mut *rng);
            chosen.push(idx as u32);
            weights[idx] = 0.0;
        }
        chosen
    }

    pub fn fanout(&self) -> usize {
        self.fanout
    }

    pub fn min_prob(&self) -> f64 {
        self.min_prob
    }

    /// Number of distinct (from, to) pairs we've observed beyond the prior.
    pub fn observations(&self) -> u64 {
        let rows = self.rows.read();
        rows.iter().map(|r| r.total_observed).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_returns_k_distinct_experts() {
        let r = TopKRouter::new(64, 2, 42);
        for t in 0..1000 {
            let ids = r.route(t);
            assert_eq!(ids.len(), 2);
            assert_ne!(ids[0], ids[1]);
            for id in &ids {
                assert!(*id < 64);
            }
        }
    }

    #[test]
    fn predictor_learns_simple_transition() {
        let p = PredictiveLoader::new(8, 2, 0.0, 1);
        for _ in 0..100 {
            p.observe(0, 3);
        }
        let preds = p.predict_next(0);
        assert!(!preds.is_empty());
        // The most likely successor of 0 must be 3.
        assert_eq!(preds[0].0, 3);
        assert!(preds[0].1 > 0.5);
    }

    #[test]
    fn predictor_respects_min_prob_threshold() {
        let p = PredictiveLoader::new(64, 4, 0.5, 1);
        // No real observations -> uniform prior gives p ~ 1/64, below 0.5.
        let preds = p.predict_next(0);
        assert!(preds.is_empty());
    }

    #[test]
    fn predictor_falls_back_to_prior_when_threshold_is_low() {
        // With N=8 and prior=1 every unobserved successor has p = 1/8 = 0.125.
        // A min_prob of 0.05 must let those through so the prefetcher has
        // candidates to issue during the cold-start phase, even with no
        // observations recorded yet. This exercises the sparse loader's
        // implicit-prior fill-in path.
        let p = PredictiveLoader::new(8, 3, 0.05, 1);
        let preds = p.predict_next(0);
        assert_eq!(preds.len(), 3);
        for (_, prob) in &preds {
            assert!((prob - 1.0 / 8.0).abs() < 1e-9);
        }
    }

    #[test]
    fn predictor_observations_counts_only_real_transitions() {
        // The dense implementation's prior used to inflate the apparent
        // observation count; the sparse one must not. After zero observe
        // calls the count is zero; after K calls it is exactly K.
        let p = PredictiveLoader::new(16, 2, 0.0, 1);
        assert_eq!(p.observations(), 0);
        for _ in 0..5 {
            p.observe(2, 7);
        }
        assert_eq!(p.observations(), 5);
    }

    #[test]
    fn predictor_handles_zero_fanout() {
        // A `--no-prefetch` ablation sets fanout to 0; the predictor must
        // return an empty candidate set without iterating the row.
        let p = PredictiveLoader::new(16, 0, 0.0, 1);
        p.observe(0, 1);
        assert!(p.predict_next(0).is_empty());
        assert!(p.sample_next(0).is_empty());
    }
}

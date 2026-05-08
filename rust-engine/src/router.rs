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
/// Transitions are stored as a flat `[num_experts][num_experts]` matrix of
/// observation counts, smoothed with a small prior so unseen successors have
/// non-zero probability and the predictor doesn't need a warm-up phase.
pub struct PredictiveLoader {
    num_experts: u32,
    /// `counts[from * n + to]` = times we observed `to` immediately after `from`.
    counts: RwLock<Vec<u32>>,
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
        Self {
            num_experts,
            counts: RwLock::new(vec![1; n * n]), // smoothing prior of 1
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
        let n = self.num_experts as usize;
        let idx = from as usize * n + to as usize;
        let mut counts = self.counts.write();
        counts[idx] = counts[idx].saturating_add(1);
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

    /// Predict up to `fanout` successors for `from`, weighted by transition
    /// probability. Returns `(expert_id, p)` pairs sorted by descending `p`,
    /// filtered by `min_prob`.
    pub fn predict_next(&self, from: u32) -> Vec<(u32, f64)> {
        if from >= self.num_experts {
            return Vec::new();
        }
        let n = self.num_experts as usize;
        let counts = self.counts.read();
        let row = &counts[(from as usize) * n..(from as usize + 1) * n];
        let total: u64 = row.iter().map(|&c| c as u64).sum();
        if total == 0 {
            return Vec::new();
        }

        let mut probs: Vec<(u32, f64)> = row
            .iter()
            .enumerate()
            .map(|(i, &c)| (i as u32, c as f64 / total as f64))
            .filter(|&(_, p)| p >= self.min_prob)
            .collect();
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        probs.truncate(self.fanout);
        probs
    }

    /// Sample `fanout` distinct successors of `from` using the weighted
    /// distribution (rather than a deterministic argmax). Useful when you
    /// want exploration in the prefetch policy.
    pub fn sample_next(&self, from: u32) -> Vec<u32> {
        if from >= self.num_experts {
            return Vec::new();
        }
        let n = self.num_experts as usize;
        let counts = self.counts.read();
        let row: Vec<f64> = counts[(from as usize) * n..(from as usize + 1) * n]
            .iter()
            .map(|&c| c as f64)
            .collect();

        let mut weights = row;
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
        let counts = self.counts.read();
        counts
            .iter()
            .map(|&c| c.saturating_sub(self.prior) as u64)
            .sum()
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
}

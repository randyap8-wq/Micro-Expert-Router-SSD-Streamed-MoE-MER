//! Deterministic Markov-chain router and Markov predictive prefetcher.
//!
//! For an MoE engine whose interesting axis is **I/O**, a router that picks
//! experts uniformly at random is actively misleading: it has no temporal
//! correlation, so any prefetch policy looks worthless. Real MoE traces show
//! strong locality — once a stream wanders into a "topic", it tends to keep
//! activating the same handful of experts for many tokens. We therefore
//! model the router as a **first-order Markov chain over expert ids**.
//!
//! Two components live here:
//!
//! 1. [`TopKRouter`] picks `k` distinct expert ids per token by sampling
//!    from `P(next | last_expert)`. The transition matrix is either:
//!    * **Generated** with structured locality — experts are split into
//!      `cluster_count` groups and stay inside their group with a high
//!      probability ([`TopKRouter::clustered`]).
//!    * **Loaded** from a file containing a row-stochastic `N x N` matrix
//!      (whitespace-separated `f64` values, row-major — easy to feed real
//!      routing traces; see [`TopKRouter::from_matrix_file`]).
//!    The router is fully deterministic given a seed.
//!
//! 2. [`PredictiveLoader`] is a first-order Markov *predictor* built
//!    online from observed activations and used by the engine to issue
//!    speculative prefetches. The router is fixed; the predictor learns.

use parking_lot::RwLock;
use rand::distributions::{Distribution, WeightedIndex};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;

/// Picks the top-K experts for a given token from a first-order Markov
/// chain over expert ids. Distinct ids guaranteed.
pub struct TopKRouter {
    num_experts: u32,
    k: usize,
    /// Row-stochastic `N x N` transition matrix: `transition[from][to]`
    /// is `P(next = to | last = from)`.
    transition: Vec<Vec<f64>>,
    /// Initial distribution used when no expert has been activated yet.
    initial: Vec<f64>,
    rng: parking_lot::Mutex<StdRng>,
    /// Last expert id this router emitted; drives the next sample. The
    /// router is therefore *stateful*; given a seed and a stream of
    /// `route()` calls, the output sequence is fully reproducible.
    last_expert: parking_lot::Mutex<Option<u32>>,
}

impl TopKRouter {
    /// Build a router from a precomputed transition matrix. Each row must
    /// have exactly `num_experts` entries and sum to a positive value
    /// (it is normalised to a probability distribution internally).
    pub fn from_matrix(num_experts: u32, k: usize, transition: Vec<Vec<f64>>, seed: u64) -> Self {
        assert!(k as u32 <= num_experts, "k must be <= num_experts");
        assert!(k > 0, "k must be > 0");
        let n = num_experts as usize;
        assert_eq!(
            transition.len(),
            n,
            "transition matrix must have num_experts rows"
        );
        let normalised: Vec<Vec<f64>> = transition
            .into_iter()
            .enumerate()
            .map(|(row_idx, row)| {
                assert_eq!(
                    row.len(),
                    n,
                    "row {row_idx} must have num_experts columns"
                );
                normalise_row(&row, n)
            })
            .collect();
        // Initial distribution = uniform over all experts. (Equivalently
        // we could use the stationary distribution of the chain, but
        // uniform is just as deterministic and obviously cluster-agnostic.)
        let initial = vec![1.0 / n as f64; n];
        Self {
            num_experts,
            k,
            transition: normalised,
            initial,
            rng: parking_lot::Mutex::new(StdRng::seed_from_u64(seed)),
            last_expert: parking_lot::Mutex::new(None),
        }
    }

    /// Build a router whose transition matrix has **structured cluster
    /// locality**: experts are partitioned into `cluster_count` groups
    /// (by `id % cluster_count`), and the chain stays inside the
    /// current cluster with probability `intra_cluster_p` (uniform
    /// across in-cluster experts) and jumps out of cluster with the
    /// remaining mass (uniform across the rest of the model).
    ///
    /// This is the synthetic stand-in used when no real routing trace is
    /// supplied. Defaults: `cluster_count = 4`, `intra_cluster_p = 0.9`,
    /// matching the gist's "4 clusters, mostly stay in cluster" example.
    pub fn clustered(
        num_experts: u32,
        k: usize,
        cluster_count: usize,
        intra_cluster_p: f64,
        seed: u64,
    ) -> Self {
        assert!(num_experts >= 1);
        assert!(cluster_count >= 1);
        assert!(
            (0.0..=1.0).contains(&intra_cluster_p),
            "intra_cluster_p must be in [0, 1], got {intra_cluster_p}"
        );
        let n = num_experts as usize;
        let cluster_count = cluster_count.min(n);
        // Assign each expert id to a cluster by `id % cluster_count` so
        // even small N (e.g. 8 experts in 4 clusters) lands two members
        // per cluster and cluster locality is meaningful.
        let cluster_of = |id: usize| id % cluster_count;
        let mut transition = vec![vec![0.0_f64; n]; n];
        for from in 0..n {
            let from_c = cluster_of(from);
            let in_cluster: Vec<usize> = (0..n).filter(|&j| cluster_of(j) == from_c).collect();
            let out_cluster: Vec<usize> = (0..n).filter(|&j| cluster_of(j) != from_c).collect();
            // Distribute `intra_cluster_p` uniformly across in-cluster
            // experts and `1 - intra_cluster_p` across out-of-cluster
            // experts. Edge case: only one cluster — all mass stays in.
            let p_in = if in_cluster.is_empty() {
                0.0
            } else {
                intra_cluster_p / in_cluster.len() as f64
            };
            let p_out = if out_cluster.is_empty() {
                0.0
            } else {
                (1.0 - intra_cluster_p) / out_cluster.len() as f64
            };
            for &j in &in_cluster {
                transition[from][j] = p_in;
            }
            for &j in &out_cluster {
                transition[from][j] = p_out;
            }
            // If single cluster (out_cluster empty) and intra_cluster_p < 1,
            // fold the remaining mass back into the in-cluster row so the
            // row still sums to 1. `in_cluster` is non-empty here:
            // every `from` is in some cluster, so its own cluster
            // contains at least itself. The `.max(1)` is belt-and-suspenders
            // against `cluster_count > num_experts` configurations.
            if out_cluster.is_empty() {
                let extra = (1.0 - intra_cluster_p) / in_cluster.len().max(1) as f64;
                for &j in &in_cluster {
                    transition[from][j] += extra;
                }
            }
        }
        Self::from_matrix(num_experts, k, transition, seed)
    }

    /// Load a row-stochastic transition matrix from a text file. The
    /// file must contain `num_experts * num_experts` whitespace-separated
    /// `f64` values laid out row-major. Rows are normalised so each sums
    /// to 1; degenerate (all-zero) rows are replaced with the uniform
    /// distribution. Use this to feed a real Mixtral routing trace
    /// without rebuilding.
    pub fn from_matrix_file(
        path: &Path,
        num_experts: u32,
        k: usize,
        seed: u64,
    ) -> io::Result<Self> {
        let n = num_experts as usize;
        let text = fs::read_to_string(path)?;
        let values: Vec<f64> = text
            .split_ascii_whitespace()
            .map(|tok| tok.parse::<f64>())
            .collect::<Result<_, _>>()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
        if values.len() != n * n {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "router matrix file {} has {} values; expected {} (num_experts^2)",
                    path.display(),
                    values.len(),
                    n * n
                ),
            ));
        }
        let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n);
        for r in 0..n {
            rows.push(values[r * n..(r + 1) * n].to_vec());
        }
        Ok(Self::from_matrix(num_experts, k, rows, seed))
    }

    /// Convenience constructor for tests / call sites that don't care
    /// about the matrix shape and just want a router with the gist's
    /// default cluster locality (4 clusters, 90% intra-cluster).
    pub fn new(num_experts: u32, k: usize, seed: u64) -> Self {
        Self::clustered(num_experts, k, 4, 0.9, seed)
    }

    /// Force a specific routing decision (used by the CLI to reproduce
    /// the "Router selects Expert ID 3 and 7" example from the spec).
    /// Updates the chain state to the last forced id so the next
    /// `route()` continues from there.
    pub fn fixed(&self, ids: &[u32]) -> Vec<u32> {
        let chosen: Vec<u32> = ids
            .iter()
            .filter(|i| **i < self.num_experts)
            .copied()
            .collect();
        if let Some(&last) = chosen.last() {
            *self.last_expert.lock() = Some(last);
        }
        chosen
    }

    /// Sample `k` distinct expert ids from `P(next | last_expert)`. The
    /// `_token_idx` argument is preserved so the engine API doesn't
    /// change, but the routing is now driven by the Markov chain
    /// internal state, not the token index. The RNG is seeded once at
    /// construction, so a full run is reproducible.
    pub fn route(&self, _token_idx: u64) -> Vec<u32> {
        let n = self.num_experts as usize;
        let mut last_guard = self.last_expert.lock();
        let row: &[f64] = match *last_guard {
            Some(prev) if (prev as usize) < n => &self.transition[prev as usize],
            _ => &self.initial,
        };
        // Materialise weights so we can zero-out picked experts to
        // guarantee `k` distinct results without re-rolling.
        let mut weights: Vec<f64> = row.to_vec();
        let mut chosen = Vec::with_capacity(self.k);
        let mut rng = self.rng.lock();
        for _ in 0..self.k {
            let dist = match WeightedIndex::new(&weights) {
                Ok(d) => d,
                Err(_) => break,
            };
            let idx = dist.sample(&mut *rng);
            chosen.push(idx as u32);
            weights[idx] = 0.0;
        }
        // Advance chain state. We use the *first* (highest-probability
        // sample on average) chosen id as the next "last expert" so the
        // chain reflects the dominant per-token activation; this keeps
        // cluster locality visible even when k > 1.
        if let Some(&first) = chosen.first() {
            *last_guard = Some(first);
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

/// Normalise a row to a probability distribution. All-zero or negative-sum
/// rows fall back to a uniform distribution so the router never panics on
/// a malformed input matrix.
fn normalise_row(row: &[f64], n: usize) -> Vec<f64> {
    let sum: f64 = row.iter().filter(|&&v| v.is_finite() && v > 0.0).sum();
    if sum > 0.0 && sum.is_finite() {
        row.iter()
            .map(|&v| if v.is_finite() && v > 0.0 { v / sum } else { 0.0 })
            .collect()
    } else {
        vec![1.0 / n as f64; n]
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
        // Saturate at u32::MAX rather than wrapping, matching the dense
        // implementation. Once a cell saturates, further observations
        // stop counting toward the row total too — otherwise the implied
        // probability `(*entry + prior) / total` would drift away from
        // the true frequency as `total` grew past `*entry`.
        if *entry < u32::MAX {
            *entry += 1;
            row.total_observed = row.total_observed.saturating_add(1);
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
        // `prior_p`). Iterate in id order for determinism.
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
    #[allow(dead_code)]
    pub fn sample_next(&self, from: u32) -> Vec<u32> {
        if from >= self.num_experts || self.fanout == 0 {
            return Vec::new();
        }
        let n = self.num_experts as usize;
        let rows = self.rows.read();
        let row = &rows[from as usize];

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

    #[allow(dead_code)]
    pub fn fanout(&self) -> usize {
        self.fanout
    }

    #[allow(dead_code)]
    pub fn min_prob(&self) -> f64 {
        self.min_prob
    }

    /// Total number of transition observations recorded (each `observe`
    /// call counts as one). Note this counts *every* observation,
    /// including repeated `(from, to)` pairs — it is **not** the number
    /// of distinct `(from, to)` pairs visited.
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
    fn clustered_router_prefers_in_cluster_transitions() {
        // 8 experts, 4 clusters → cluster membership is `id % 4`, so
        // experts {0,4} share cluster 0. With a 0.95 intra-cluster
        // probability, the chain has stationary distribution uniform
        // across clusters (~25% each) but the per-*step* property we
        // care about is locality: consecutive routes should land in
        // the same cluster ~95% of the time. That is the actual signal
        // a prefetcher learns from.
        let r = TopKRouter::clustered(8, 1, 4, 0.95, 7);
        r.fixed(&[0]);
        let trials = 4000;
        let mut last_cluster = 0u32;
        let mut same_cluster = 0;
        for t in 0..trials {
            let ids = r.route(t as u64);
            let c = ids[0] % 4;
            if t > 0 && c == last_cluster {
                same_cluster += 1;
            }
            last_cluster = c;
        }
        let frac = same_cluster as f64 / (trials - 1) as f64;
        assert!(
            frac > 0.85,
            "expected >85% step-to-step in-cluster transitions, got {:.3} ({}/{})",
            frac,
            same_cluster,
            trials - 1
        );
    }

    #[test]
    fn router_is_deterministic_given_seed() {
        // Two routers built with the same seed must emit identical
        // sequences. This is the property that makes runs reproducible.
        let a = TopKRouter::new(16, 2, 0xC0FFEE);
        let b = TopKRouter::new(16, 2, 0xC0FFEE);
        for t in 0..200 {
            assert_eq!(a.route(t), b.route(t), "diverged at token {t}");
        }
    }

    #[test]
    fn matrix_file_loads_and_normalises() {
        // 4 experts, "shift-by-one" matrix (each row picks (r+1) mod 4):
        // emit a temp file, load it, verify the chain cycles deterministically.
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "micro-expert-router-matrix-{}-{}.txt",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let body = "\
            0 1 0 0\n\
            0 0 1 0\n\
            0 0 0 1\n\
            1 0 0 0\n";
        std::fs::write(&path, body).unwrap();
        let r = TopKRouter::from_matrix_file(&path, 4, 1, 1).unwrap();
        let _ = std::fs::remove_file(&path);
        r.fixed(&[0]);
        let expected = [1u32, 2, 3, 0, 1, 2, 3, 0];
        for (t, want) in expected.iter().enumerate() {
            let got = r.route(t as u64);
            assert_eq!(got, vec![*want], "step {t}: want {want:?}, got {got:?}");
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
        let p = PredictiveLoader::new(8, 3, 0.05, 1);
        let preds = p.predict_next(0);
        assert_eq!(preds.len(), 3);
        for (_, prob) in &preds {
            assert!((prob - 1.0 / 8.0).abs() < 1e-9);
        }
    }

    #[test]
    fn predictor_observations_counts_only_real_transitions() {
        let p = PredictiveLoader::new(16, 2, 0.0, 1);
        assert_eq!(p.observations(), 0);
        for _ in 0..5 {
            p.observe(2, 7);
        }
        assert_eq!(p.observations(), 5);
    }

    #[test]
    fn predictor_handles_zero_fanout() {
        let p = PredictiveLoader::new(16, 0, 0.0, 1);
        p.observe(0, 1);
        assert!(p.predict_next(0).is_empty());
        assert!(p.sample_next(0).is_empty());
    }
}

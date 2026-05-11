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
use std::collections::{HashMap, VecDeque};
use std::sync::OnceLock;
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
    /// `rows[from]` — sparse counts of successors of `from` (1st-order).
    rows: RwLock<Vec<Row>>,
    /// `rows2[(prev_prev, prev)]` — sparse counts of successors of the
    /// pair `(prev_prev -> prev)`. Stored as a `HashMap` keyed by the
    /// flat index `prev_prev * num_experts + prev` to keep the memory
    /// footprint sparse — only pairs we actually observed are
    /// allocated. Used by [`Self::predict_next`] to blend a 2nd-order
    /// signal with the 1st-order baseline.
    rows2: RwLock<HashMap<u64, Row>>,
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
            rows2: RwLock::new(HashMap::new()),
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
        // Saturate at u32::MAX rather than wrapping. Once a cell
        // saturates, further observations stop counting toward the row
        // total too — otherwise the implied probability
        // `(*entry + prior) / total` would drift away from the true
        // frequency as `total` grew past `*entry`.
        if *entry < u32::MAX {
            *entry += 1;
            row.total_observed = row.total_observed.saturating_add(1);
        }
    }

    /// Record a 2nd-order observation: `(prev_prev -> prev -> to)` was
    /// the actual sequence over three consecutive tokens.
    pub fn observe2(&self, prev_prev: u32, prev: u32, to: u32) {
        let n = self.num_experts;
        if prev_prev >= n || prev >= n || to >= n {
            return;
        }
        let key = (prev_prev as u64) * (n as u64) + prev as u64;
        let mut rows2 = self.rows2.write();
        let row = rows2.entry(key).or_insert_with(Row::new);
        let entry = row.counts.entry(to).or_insert(0);
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

    /// Record both the 1st-order `(prev -> next)` and the 2nd-order
    /// `(prev_prev -> prev -> next)` transitions. `prev_prev_set` may
    /// be empty (e.g. very first token after a cold start), in which
    /// case only the 1st-order observations are recorded.
    pub fn observe_step2(&self, prev_prev_set: &[u32], prev_set: &[u32], next_set: &[u32]) {
        self.observe_step(prev_set, next_set);
        for &pp in prev_prev_set {
            for &p in prev_set {
                for &n in next_set {
                    self.observe2(pp, p, n);
                }
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

    /// 2nd-order variant of [`Self::predict_next`]. When a `(prev_prev,
    /// prev)` row has been observed, blends its distribution with the
    /// 1st-order distribution from `prev` (50/50). Falls back to pure
    /// 1st-order when no 2nd-order data exists for the pair, so the
    /// caller can use this unconditionally without warm-up.
    pub fn predict_next2(&self, prev_prev: u32, prev: u32) -> Vec<(u32, f64)> {
        let baseline = self.predict_next(prev);
        if prev_prev >= self.num_experts || prev >= self.num_experts || self.fanout == 0 {
            return baseline;
        }
        let key = (prev_prev as u64) * (self.num_experts as u64) + prev as u64;
        let rows2 = self.rows2.read();
        let Some(row2) = rows2.get(&key) else {
            return baseline;
        };
        if row2.total_observed == 0 {
            return baseline;
        }
        let total = self.effective_total(row2.total_observed);
        let total_f = total as f64;
        let prior_f = self.prior as f64;
        // Build a combined probability map: blend 1st-order and 2nd-order.
        let mut combined: HashMap<u32, f64> = HashMap::new();
        for (id, p) in &baseline {
            *combined.entry(*id).or_insert(0.0) += 0.5 * *p;
        }
        for (&id, &c) in &row2.counts {
            let p = (c as f64 + prior_f) / total_f;
            *combined.entry(id).or_insert(0.0) += 0.5 * p;
        }
        let mut out: Vec<(u32, f64)> = combined
            .into_iter()
            .filter(|&(_, p)| p >= self.min_prob)
            .collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        out.truncate(self.fanout);
        out
    }

    /// **Unified prediction.** Combines the Markov-chain row predictions
    /// with the [`LocalityMonitor`]'s current hot set and (optionally)
    /// the [`NeuralSpeculator`]'s top-K to produce a single ranked list
    /// of `(expert_id, priority_score)` pairs. This is the API called
    /// out in the design spec under Task 1: *"the predictor should now
    /// be able to return a unified set of (expert_id, priority_score)"*.
    ///
    /// Scoring (in `[0, 1]`):
    /// * Markov contribution: probability from
    ///   [`Self::predict_next2`] (or [`Self::predict_next`] when
    ///   `prev_prev` is `None`), weighted **W_MARKOV**.
    /// * Locality contribution: a flat **W_LOCALITY** for every expert
    ///   in `monitor.get_hot_experts(threshold_pct)` — these are
    ///   temporally stable hot experts and we want them in the
    ///   prefetch set even when the Markov chain is uncertain.
    /// * Speculator contribution: a flat **W_SPECULATOR** for every
    ///   expert in `speculator.predict_topk(hidden, speculator_k)` —
    ///   semantic intent is the strongest single signal in the design.
    ///
    /// The three constants sum to **1.0**, so an expert hit by all
    /// three arms with maximal Markov probability tops out at exactly
    /// `1.0` rather than overshooting (the previous design summed to
    /// 1.2, which made the score ill-defined as a probability and
    /// made cross-arm comparisons against `[0, 1]` thresholds
    /// awkward). Their relative weighting still encodes the design
    /// intent: speculator > Markov > locality.
    ///
    /// The three contributions sum, then the result is sorted by
    /// descending score (ties broken by ascending id for determinism)
    /// and truncated to `self.fanout`. An expert that lights up in
    /// every arm is therefore prioritised over one that lights up in
    /// only one.
    ///
    /// Hidden state is borrowed (not cloned) so this is safe to call
    /// on the hot path; the speculator forward is its own internal
    /// allocation.
    pub fn predict_unified(
        &self,
        prev_prev: Option<u32>,
        prev: u32,
        monitor: Option<&LocalityMonitor>,
        threshold_pct: f32,
        speculator: Option<&NeuralSpeculator>,
        hidden: &[f32],
        speculator_k: usize,
    ) -> Vec<(u32, f32)> {
        // Weights for each predictive arm. Normalised so a unanimous
        // expert (Markov p=1, locality hot, speculator top-K) tops
        // out at exactly 1.0. Relative ordering: the speculator is
        // the strongest signal (semantic intent → likely to be
        // correct), Markov is next (statistical smoothing of
        // observed transitions), and locality is the weakest tie-
        // breaker (a flat "this expert is generally hot lately").
        const W_SPECULATOR: f32 = 0.42;
        const W_MARKOV: f32 = 0.33;
        const W_LOCALITY: f32 = 0.25;
        // Compile-time invariant: the weights sum to 1.0 (within
        // f32 epsilon). Kept as a debug_assert so a future tweak
        // that breaks the contract trips a test rather than
        // silently producing >1 scores.
        debug_assert!((W_SPECULATOR + W_MARKOV + W_LOCALITY - 1.0).abs() < 1e-6);

        let mut combined: HashMap<u32, f32> = HashMap::new();

        // 1) Markov contribution.
        let markov = match prev_prev {
            Some(pp) => self.predict_next2(pp, prev),
            None => self.predict_next(prev),
        };
        for (id, p) in markov {
            *combined.entry(id).or_insert(0.0) += W_MARKOV * (p as f32);
        }

        // 2) Locality contribution.
        if let Some(m) = monitor {
            for id in m.get_hot_experts(threshold_pct) {
                *combined.entry(id).or_insert(0.0) += W_LOCALITY;
            }
        }

        // 3) Speculator contribution.
        if let Some(s) = speculator {
            if speculator_k > 0 {
                for id in s.predict_topk(hidden, speculator_k) {
                    *combined.entry(id).or_insert(0.0) += W_SPECULATOR;
                }
            }
        }

        let mut out: Vec<(u32, f32)> = combined
            .into_iter()
            .filter(|&(_, p)| p > 0.0)
            .collect();
        out.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        // Honour the loader's fanout when set; otherwise return all.
        if self.fanout > 0 {
            out.truncate(self.fanout);
        }
        out
    }

    /// Sample the next predicted experts from the Markov-chain row's
    /// distribution (rather than a deterministic argmax). Useful when
    /// you want exploration in the prefetch policy.
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

/// Sliding-window **Locality Monitor** — the temporal-stability arm of the
/// dual-path predictive controller.
///
/// The monitor observes every routed expert id and maintains a heat map
/// (frequency count) over a sliding window of the most recent `window`
/// observations. The intent is orthogonal to the Markov predictor: where
/// the Markov chain answers "given the last expert id, what tends to
/// fire next?", the locality monitor answers "regardless of *which*
/// expert just fired, which experts have been disproportionately busy
/// over the last few hundred tokens?". An expert that crosses the
/// `threshold_pct` heat ratio is "hot" and can be **pinned** in the
/// expert cache so it is protected from LRU eviction even when the
/// Markov chain wanders elsewhere.
///
/// Internally:
/// * `window` is a `VecDeque<u32>` of expert ids in arrival order; once
///   it reaches capacity each new observation evicts the oldest entry.
/// * `counts` is a flat `Vec<u32>` indexed by expert id holding the
///   number of times that id appears in the window. The flat layout is
///   `O(num_experts)` memory but reads/writes are cache-line tight.
/// * `total` mirrors `window.len()` so `is_hot` doesn't have to lock the
///   deque just to compute the denominator.
///
/// All accessors lock a single `RwLock`; the monitor is small, the
/// critical section is short, and contention is bounded by the rate at
/// which the engine routes tokens.
pub struct LocalityMonitor {
    num_experts: u32,
    capacity: usize,
    inner: RwLock<LocalityInner>,
}

struct LocalityInner {
    window: VecDeque<u32>,
    counts: Vec<u32>,
    total: u64,
}

impl LocalityMonitor {
    /// Default sliding-window length when no explicit value is configured.
    /// 512 tokens matches the value called out in the design spec — long
    /// enough to smooth out per-token noise, short enough to track topic
    /// shifts that warrant repinning.
    pub const DEFAULT_WINDOW: usize = 512;

    /// Default heat threshold (10% of the window) for declaring an
    /// expert "hot". Mirrors the spec's example.
    pub const DEFAULT_THRESHOLD_PCT: f32 = 0.10;

    pub fn new(num_experts: u32, window: usize) -> Self {
        let capacity = window.max(1);
        Self {
            num_experts,
            capacity,
            inner: RwLock::new(LocalityInner {
                window: VecDeque::with_capacity(capacity),
                counts: vec![0u32; num_experts as usize],
                total: 0,
            }),
        }
    }

    /// Observe a single expert activation. Drops the oldest observation
    /// if the window is full.
    pub fn observe_one(&self, id: u32) {
        if id >= self.num_experts {
            return;
        }
        let mut inner = self.inner.write();
        if inner.window.len() == self.capacity {
            if let Some(old) = inner.window.pop_front() {
                let slot = &mut inner.counts[old as usize];
                if *slot > 0 {
                    *slot -= 1;
                }
            }
        }
        // `total` is the cumulative observation counter (saturating
        // semantics promised by [`Self::total_observations`]); count
        // every observation, not just the ones that grew the window.
        inner.total = inner.total.saturating_add(1);
        inner.window.push_back(id);
        inner.counts[id as usize] = inner.counts[id as usize].saturating_add(1);
    }

    /// Observe a batch of expert activations (e.g. one token's top-K).
    pub fn observe(&self, ids: &[u32]) {
        for &id in ids {
            self.observe_one(id);
        }
    }

    /// Window capacity (the maximum number of observations kept).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of observations currently in the window.
    pub fn len(&self) -> usize {
        self.inner.read().window.len()
    }

    /// Whether `id` currently exceeds the heat threshold. `threshold_pct`
    /// is a fraction in `[0, 1]`; e.g. `0.1` means "≥10% of the window".
    /// Always returns `false` when the window is empty (no signal yet).
    pub fn is_hot(&self, id: u32, threshold_pct: f32) -> bool {
        if id >= self.num_experts {
            return false;
        }
        let inner = self.inner.read();
        let len = inner.window.len();
        if len == 0 {
            return false;
        }
        let count = inner.counts[id as usize] as f32;
        let cutoff = threshold_pct.max(0.0) * len as f32;
        // `>=` so a threshold of 0.0 still requires at least one
        // observation (since `count == 0` for absent ids).
        count > 0.0 && count >= cutoff
    }

    /// Snapshot of the current hot set: every id whose count meets the
    /// `threshold_pct` ratio. Sorted descending by heat count, ties
    /// broken by ascending id for determinism.
    pub fn hot_set(&self, threshold_pct: f32) -> Vec<u32> {
        let inner = self.inner.read();
        let len = inner.window.len();
        if len == 0 {
            return Vec::new();
        }
        let cutoff = (threshold_pct.max(0.0) * len as f32).max(1.0);
        let mut out: Vec<(u32, u32)> = inner
            .counts
            .iter()
            .enumerate()
            .filter_map(|(i, &c)| {
                if (c as f32) >= cutoff {
                    Some((i as u32, c))
                } else {
                    None
                }
            })
            .collect();
        out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        out.into_iter().map(|(id, _)| id).collect()
    }

    /// Spec-compliant alias of [`Self::hot_set`]. The "Omniscient
    /// Predictive Architecture" design document calls this method
    /// `get_hot_experts(threshold)`; the original implementation
    /// shipped under the shorter `hot_set` name. Both are kept so
    /// callers can use whichever they prefer.
    #[inline]
    pub fn get_hot_experts(&self, threshold_pct: f32) -> Vec<u32> {
        self.hot_set(threshold_pct)
    }

    /// Heat count for a specific id (0 if `id` is out of range).
    pub fn heat(&self, id: u32) -> u32 {
        if id >= self.num_experts {
            return 0;
        }
        self.inner.read().counts[id as usize]
    }

    /// Aggregate observation count since construction (saturating).
    pub fn total_observations(&self) -> u64 {
        self.inner.read().total
    }

    /// Reset the monitor back to its empty state. Useful for tests and
    /// for re-warming the monitor between unrelated request streams.
    #[allow(dead_code)]
    pub fn clear(&self) {
        let mut inner = self.inner.write();
        inner.window.clear();
        for c in inner.counts.iter_mut() {
            *c = 0;
        }
        inner.total = 0;
    }
}

/// Tiny **Neural Speculator** — the semantic-intent arm of the dual-path
/// predictive controller.
///
/// A two-layer MLP (`d_model -> hidden -> num_experts`) trained online
/// against the gating network's actual top-K decisions. The network is
/// kept intentionally small (default hidden = 128) so it fits in L1/L2
/// cache and so a single SGD step per token is cheap; the goal is not
/// to *replace* the gate but to provide a fast, stateless prediction of
/// "what experts will likely fire next given this hidden state" that
/// can be unioned with the Markov chain's hint to drive prefetch.
///
/// Architecture details:
/// * Layer 1: `W1` of shape `[hidden, d_model]`, bias `b1` of `hidden`,
///   ReLU activation.
/// * Layer 2: `W2` of shape `[num_experts, hidden]`, bias `b2` of
///   `num_experts`. The output is interpreted as a logit vector;
///   `predict_topk` returns the top-K ids by logit, and `train_step`
///   takes a softmax + cross-entropy step against a soft target spread
///   uniformly over the actual top-K ids selected by the real gate.
///
/// Initialisation uses He-uniform scaling (the standard ReLU init) with
/// a deterministic seed so identical runs produce identical predictions.
/// All numerical paths are `f32`; gradient clipping at `±1.0` and a
/// `clamp_finite` guard on every weight write keep a stuck model from
/// producing NaN/Inf — the predictor never tearing down the engine
/// (this is a *prefetch hint*; correctness still flows through the real
/// gate downstream).
pub struct NeuralSpeculator {
    d_model: usize,
    hidden: usize,
    num_experts: u32,
    /// Locked together so a `predict()` always sees a consistent `(W1,
    /// b1, W2, b2)` snapshot — `train_step` writes all four and we
    /// don't want a half-updated set of weights to be read mid-update.
    ///
    /// Read access is **prioritised** in the off-path training worker:
    /// the worker calls [`parking_lot::RwLock::try_write_for`] with a
    /// short timeout and drops the sample if it can't acquire the
    /// lock, so the hot-path `predict_topk` is never blocked by
    /// background SGD compute.
    weights: RwLock<SpeculatorWeights>,
    /// Asynchronous training queue. Set on first call to
    /// [`Self::spawn_training_worker`] (idempotent) and consumed by a
    /// dedicated background thread. The queue is bounded so a
    /// runaway producer can't pin unbounded memory: when full,
    /// [`Self::queue_train`] drops the newest sample (training is a
    /// best-effort signal — the *correct* routing still flows
    /// through the gate downstream).
    train_queue: OnceLock<std::sync::mpsc::SyncSender<TrainSample>>,
}

/// A single hidden-state / actual-routing pair queued for off-path
/// SGD. Fields are owned so the background thread doesn't borrow
/// anything from the hot path.
struct TrainSample {
    x: Vec<f32>,
    actual_top_k: Vec<u32>,
    lr: f32,
}

struct SpeculatorWeights {
    /// `[hidden, d_model]` row-major.
    w1: Vec<f32>,
    /// `[hidden]`.
    b1: Vec<f32>,
    /// `[num_experts, hidden]` row-major.
    w2: Vec<f32>,
    /// `[num_experts]`.
    b2: Vec<f32>,
}

impl NeuralSpeculator {
    /// Default hidden size called out in the design spec.
    pub const DEFAULT_HIDDEN: usize = 128;

    /// Default learning rate for the online SGD step. Small by design:
    /// the speculator is trained continuously over every token and a
    /// large lr would let occasional outlier routings dominate.
    pub const DEFAULT_LR: f32 = 1e-3;

    /// L2 weight-decay coefficient applied per SGD step. Small so it
    /// only nudges idle weights toward zero (preventing unbounded
    /// drift when the same expert id never re-appears) without
    /// fighting the gradient signal on actively-routed experts. The
    /// per-step update applies `w := w * (1 - lr * WEIGHT_DECAY)`
    /// before the gradient step (the standard "decoupled weight
    /// decay" / AdamW-style ordering) so the decay is independent of
    /// the gradient magnitude.
    pub const WEIGHT_DECAY: f32 = 1e-4;

    /// Build a fresh speculator with He-uniform initialisation.
    pub fn new(d_model: usize, hidden: usize, num_experts: u32, seed: u64) -> Self {
        assert!(d_model > 0 && hidden > 0 && num_experts > 0);
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(0xC0DE_FEED));
        // He uniform: U(-sqrt(6/fan_in), +sqrt(6/fan_in)).
        let bound1 = (6.0f32 / d_model as f32).sqrt();
        let bound2 = (6.0f32 / hidden as f32).sqrt();
        use rand::Rng;
        let w1: Vec<f32> = (0..hidden * d_model)
            .map(|_| rng.gen_range(-bound1..bound1))
            .collect();
        let w2: Vec<f32> = (0..(num_experts as usize) * hidden)
            .map(|_| rng.gen_range(-bound2..bound2))
            .collect();
        let b1 = vec![0.0f32; hidden];
        let b2 = vec![0.0f32; num_experts as usize];
        Self {
            d_model,
            hidden,
            num_experts,
            weights: RwLock::new(SpeculatorWeights { w1, b1, w2, b2 }),
            train_queue: OnceLock::new(),
        }
    }

    pub fn d_model(&self) -> usize {
        self.d_model
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    pub fn num_experts(&self) -> u32 {
        self.num_experts
    }

    /// Forward pass returning the full logit vector of length `num_experts`.
    /// Used internally by `predict_topk` and `train_step` and also by
    /// tests that want to inspect the raw logits.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.d_model);
        let w = self.weights.read();
        let mut h = vec![0.0f32; self.hidden];
        for i in 0..self.hidden {
            let row = &w.w1[i * self.d_model..(i + 1) * self.d_model];
            let mut acc = w.b1[i];
            for j in 0..self.d_model {
                acc += row[j] * x[j];
            }
            // ReLU
            h[i] = if acc > 0.0 { acc } else { 0.0 };
        }
        let n = self.num_experts as usize;
        let mut logits = vec![0.0f32; n];
        for i in 0..n {
            let row = &w.w2[i * self.hidden..(i + 1) * self.hidden];
            let mut acc = w.b2[i];
            for j in 0..self.hidden {
                acc += row[j] * h[j];
            }
            logits[i] = acc;
        }
        logits
    }

    /// Predict the top-`k` expert ids by logit. Output is in descending
    /// logit order; ties broken by ascending id so the choice is fully
    /// deterministic. Empty input or `k==0` yields an empty vector.
    pub fn predict_topk(&self, x: &[f32], k: usize) -> Vec<u32> {
        if k == 0 || x.len() != self.d_model {
            return Vec::new();
        }
        let logits = self.forward(x);
        let mut idx: Vec<(u32, f32)> = logits
            .into_iter()
            .enumerate()
            .map(|(i, v)| (i as u32, v))
            .collect();
        idx.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        idx.truncate(k.min(self.num_experts as usize));
        idx.into_iter().map(|(i, _)| i).collect()
    }

    /// Predict the top-`k` expert ids together with their softmax
    /// probabilities. Useful for callers (telemetry, prefetch
    /// budgeting) that want a confidence signal alongside the ranking.
    pub fn predict_topk_with_probs(&self, x: &[f32], k: usize) -> Vec<(u32, f32)> {
        if k == 0 || x.len() != self.d_model {
            return Vec::new();
        }
        let mut logits = self.forward(x);
        // Numerically-stable softmax in place.
        let mut max = f32::NEG_INFINITY;
        for &v in &logits {
            if v > max {
                max = v;
            }
        }
        if !max.is_finite() {
            return Vec::new();
        }
        let mut sum = 0.0f32;
        for v in logits.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for v in logits.iter_mut() {
                *v /= sum;
            }
        }
        let mut idx: Vec<(u32, f32)> = logits
            .into_iter()
            .enumerate()
            .map(|(i, v)| (i as u32, v))
            .collect();
        idx.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        idx.truncate(k.min(self.num_experts as usize));
        idx
    }

    /// One step of online SGD against a soft-target distribution that
    /// places mass `1/|actual|` on each id in `actual_top_k` and 0
    /// elsewhere. The loss is cross-entropy on top of softmax; the
    /// gradient w.r.t. the logits is `softmax(logits) - target`.
    /// Gradient values are clipped to `±1.0` per element and weight
    /// updates are NaN/Inf-guarded.
    ///
    /// Returns the cross-entropy loss for diagnostics. Out-of-range
    /// ids in `actual_top_k` are silently ignored; if every id is
    /// out of range the weights are not updated and the returned loss
    /// is `f32::NAN`.
    pub fn train_step(&self, x: &[f32], actual_top_k: &[u32], lr: f32) -> f32 {
        if x.len() != self.d_model || actual_top_k.is_empty() {
            return f32::NAN;
        }
        let mut w = self.weights.write();
        self.train_step_locked(&mut w, x, actual_top_k, lr)
    }

    /// SGD body with the weight lock held by the caller. Separated
    /// out so the off-path training worker can attempt the write
    /// using [`parking_lot::RwLock::try_write_for`] and bail (rather
    /// than block readers) when the hot path is busy.
    fn train_step_locked(
        &self,
        w: &mut SpeculatorWeights,
        x: &[f32],
        actual_top_k: &[u32],
        lr: f32,
    ) -> f32 {
        // Build the target distribution.
        let n = self.num_experts as usize;
        let mut target = vec![0.0f32; n];
        let mut hits = 0usize;
        for &id in actual_top_k {
            if (id as usize) < n {
                target[id as usize] += 1.0;
                hits += 1;
            }
        }
        if hits == 0 {
            return f32::NAN;
        }
        let inv = 1.0f32 / hits as f32;
        for v in target.iter_mut() {
            *v *= inv;
        }

        // Forward (re-do here so we can capture the hidden activation).
        let mut h_pre = vec![0.0f32; self.hidden];
        let mut h = vec![0.0f32; self.hidden];
        for i in 0..self.hidden {
            let row = &w.w1[i * self.d_model..(i + 1) * self.d_model];
            let mut acc = w.b1[i];
            for j in 0..self.d_model {
                acc += row[j] * x[j];
            }
            h_pre[i] = acc;
            h[i] = if acc > 0.0 { acc } else { 0.0 };
        }
        let mut logits = vec![0.0f32; n];
        for i in 0..n {
            let row = &w.w2[i * self.hidden..(i + 1) * self.hidden];
            let mut acc = w.b2[i];
            for j in 0..self.hidden {
                acc += row[j] * h[j];
            }
            logits[i] = acc;
        }
        // Softmax + cross-entropy loss.
        let mut max = f32::NEG_INFINITY;
        for &v in &logits {
            if v > max {
                max = v;
            }
        }
        if !max.is_finite() {
            return f32::NAN;
        }
        let mut probs = logits.clone();
        let mut sum = 0.0f32;
        for v in probs.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        if !(sum > 0.0 && sum.is_finite()) {
            return f32::NAN;
        }
        let mut loss = 0.0f32;
        for (i, v) in probs.iter_mut().enumerate() {
            *v /= sum;
            if target[i] > 0.0 {
                let p = v.max(1e-12);
                loss -= target[i] * p.ln();
            }
        }

        // Gradient w.r.t. logits = probs - target.
        let mut dlogits = probs;
        for i in 0..n {
            dlogits[i] -= target[i];
            // Per-element gradient clipping.
            if dlogits[i] > 1.0 {
                dlogits[i] = 1.0;
            } else if dlogits[i] < -1.0 {
                dlogits[i] = -1.0;
            } else if !dlogits[i].is_finite() {
                dlogits[i] = 0.0;
            }
        }

        // Backprop into hidden: dh = W2^T · dlogits, masked by ReLU
        // derivative (1 where pre-activation > 0, else 0).
        let mut dh = vec![0.0f32; self.hidden];
        for i in 0..n {
            let g = dlogits[i];
            if g == 0.0 {
                continue;
            }
            let row = &w.w2[i * self.hidden..(i + 1) * self.hidden];
            for j in 0..self.hidden {
                dh[j] += row[j] * g;
            }
        }
        for j in 0..self.hidden {
            if h_pre[j] <= 0.0 {
                dh[j] = 0.0;
            }
            // Clip again post-mask (the multiplication by W2 can blow up).
            if dh[j] > 1.0 {
                dh[j] = 1.0;
            } else if dh[j] < -1.0 {
                dh[j] = -1.0;
            } else if !dh[j].is_finite() {
                dh[j] = 0.0;
            }
        }

        // Decoupled weight decay (AdamW-style): shrink every weight by
        // `(1 - lr * WEIGHT_DECAY)` *before* the gradient step. This
        // bounds the magnitude of weights for experts/features that
        // never appear in `actual_top_k` (whose gradient through this
        // step is zero), so the speculator can't drift toward
        // arbitrarily large logits over an infinite training run.
        // Biases are intentionally not decayed (standard practice;
        // they don't suffer from the same scale-blowup pathology).
        let decay = 1.0 - lr * Self::WEIGHT_DECAY;
        if decay > 0.0 && decay < 1.0 {
            for v in w.w2.iter_mut() {
                let new = *v * decay;
                if new.is_finite() {
                    *v = new;
                }
            }
            for v in w.w1.iter_mut() {
                let new = *v * decay;
                if new.is_finite() {
                    *v = new;
                }
            }
        }

        // Update W2 / b2.
        for i in 0..n {
            let g = dlogits[i];
            if g == 0.0 {
                continue;
            }
            let row = &mut w.w2[i * self.hidden..(i + 1) * self.hidden];
            for j in 0..self.hidden {
                let upd = lr * g * h[j];
                let new = row[j] - upd;
                row[j] = if new.is_finite() { new } else { row[j] };
            }
            let new_b = w.b2[i] - lr * g;
            w.b2[i] = if new_b.is_finite() { new_b } else { w.b2[i] };
        }
        // Update W1 / b1.
        for i in 0..self.hidden {
            let g = dh[i];
            if g == 0.0 {
                continue;
            }
            let row = &mut w.w1[i * self.d_model..(i + 1) * self.d_model];
            for j in 0..self.d_model {
                let upd = lr * g * x[j];
                let new = row[j] - upd;
                row[j] = if new.is_finite() { new } else { row[j] };
            }
            let new_b = w.b1[i] - lr * g;
            w.b1[i] = if new_b.is_finite() { new_b } else { w.b1[i] };
        }

        loss
    }

    /// Idempotently spawn the off-path training worker. Subsequent
    /// calls are no-ops. After this returns, [`Self::queue_train`]
    /// will drop samples onto the worker's bounded channel; the
    /// worker pulls them in FIFO order and applies SGD updates,
    /// preferring readers via [`parking_lot::RwLock::try_write_for`].
    ///
    /// The worker is a plain OS thread (not a tokio task) so it
    /// keeps making progress even if the tokio runtime is
    /// momentarily saturated; it terminates when the [`Arc`] count
    /// of the speculator drops to one (its own ref) — i.e. when no
    /// engine still holds it.
    pub fn spawn_training_worker(self: &std::sync::Arc<Self>) {
        // OnceLock::get_or_init guarantees a single worker even
        // under concurrent installs.
        self.train_queue.get_or_init(|| {
            // Bounded queue so a runaway producer can't pin memory.
            // A few hundred slots is plenty: at typical decoder
            // throughput the worker drains the queue between
            // tokens; the bound only matters during pathological
            // bursts (and in that case we'd rather drop old
            // samples than stall the producer).
            let (tx, rx) = std::sync::mpsc::sync_channel::<TrainSample>(256);
            let weak = std::sync::Arc::downgrade(self);
            std::thread::Builder::new()
                .name("mer-speculator-train".to_string())
                .spawn(move || speculator_training_loop(weak, rx))
                .ok(); // Failing to spawn is non-fatal; fall back to in-line train.
            tx
        });
    }

    /// Push one `(hidden_state, actual_top_k)` sample onto the
    /// off-path training queue. Drops the sample silently when the
    /// queue is full or when [`Self::spawn_training_worker`] was
    /// never called — training is a prefetch-hint signal, not a
    /// correctness-critical path.
    pub fn queue_train(&self, x: &[f32], actual_top_k: &[u32], lr: f32) {
        if x.len() != self.d_model || actual_top_k.is_empty() {
            return;
        }
        if let Some(tx) = self.train_queue.get() {
            let sample = TrainSample {
                x: x.to_vec(),
                actual_top_k: actual_top_k.to_vec(),
                lr,
            };
            // `try_send` is non-blocking: when the worker is
            // behind, the newest sample is dropped (a deliberate
            // back-pressure choice — old samples in the queue still
            // contain useful gradient signal).
            let _ = tx.try_send(sample);
        }
    }
}

/// Background training loop. Pulls samples from the queue and
/// applies SGD updates, preferring readers by using
/// [`parking_lot::RwLock::try_write_for`] — if a hot-path
/// `predict_topk` is currently reading, the worker waits at most a
/// few hundred microseconds and then drops the sample rather than
/// starve the inference critical path.
fn speculator_training_loop(
    weak: std::sync::Weak<NeuralSpeculator>,
    rx: std::sync::mpsc::Receiver<TrainSample>,
) {
    use std::time::Duration;
    // Cap the per-sample wait. The hot path reads in the low-µs
    // range, so 500 µs is generous: if we can't get the write
    // lock in that window, readers are bursty and we'd rather
    // skip this update.
    const TRY_WRITE_BUDGET: Duration = Duration::from_micros(500);

    while let Ok(sample) = rx.recv() {
        let Some(spec) = weak.upgrade() else {
            return; // engine dropped the speculator; quit cleanly.
        };
        if let Some(mut w) = spec.weights.try_write_for(TRY_WRITE_BUDGET) {
            let _loss = spec.train_step_locked(&mut w, &sample.x, &sample.actual_top_k, sample.lr);
            // Loss is intentionally discarded — telemetry covers
            // accuracy via the engine's predict-and-train path.
        } else {
            tracing::trace!(
                "speculator training worker: skipped one sample (reader contention)"
            );
        }
        drop(spec);
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

    #[test]
    fn second_order_predictor_outranks_first_order_when_signal_disagrees() {
        // 1st-order: from `prev=2` we mostly see `5`.
        // 2nd-order: from `(prev_prev=7, prev=2)` we mostly see `9`.
        // `predict_next2(7, 2)` should rank `9` higher than `predict_next(2)` does.
        let p = PredictiveLoader::new(16, 4, 0.0, 1);
        for _ in 0..50 {
            p.observe(2, 5);
        }
        for _ in 0..50 {
            p.observe2(7, 2, 9);
            // observe2 also implies observe(2, 9) is *not* called; the
            // engine wires both via observe_step2. Test only the rows2
            // contribution here.
        }
        let preds1 = p.predict_next(2);
        let preds2 = p.predict_next2(7, 2);
        let rank1 = |id: u32, list: &[(u32, f64)]| {
            list.iter().position(|(e, _)| *e == id)
        };
        let r9_first = rank1(9, &preds1).unwrap_or(usize::MAX);
        let r9_second = rank1(9, &preds2).unwrap_or(usize::MAX);
        assert!(
            r9_second < r9_first || (r9_second == 0),
            "2nd-order should rank 9 at least as high as 1st-order: \
             1st={preds1:?} 2nd={preds2:?}"
        );
    }

    #[test]
    fn observe_step2_falls_back_to_first_order_when_prev_prev_empty() {
        let p = PredictiveLoader::new(8, 2, 0.0, 1);
        // Empty prev_prev means we only get 1st-order observations.
        p.observe_step2(&[], &[1], &[3]);
        let preds = p.predict_next(1);
        assert_eq!(preds[0].0, 3);
    }

    // ---------------------- LocalityMonitor tests ---------------------

    #[test]
    fn locality_monitor_tracks_window_and_heat_counts() {
        let m = LocalityMonitor::new(8, 4);
        m.observe(&[1, 2, 1, 3]);
        assert_eq!(m.len(), 4);
        assert_eq!(m.heat(1), 2);
        assert_eq!(m.heat(2), 1);
        assert_eq!(m.heat(3), 1);
        // Adding a 5th observation evicts the oldest (id=1).
        m.observe_one(5);
        assert_eq!(m.len(), 4);
        assert_eq!(m.heat(1), 1);
        assert_eq!(m.heat(5), 1);
    }

    #[test]
    fn locality_monitor_hot_set_above_threshold() {
        let m = LocalityMonitor::new(8, 10);
        // Expert 4 is hammered; others get one observation each.
        for _ in 0..7 {
            m.observe_one(4);
        }
        m.observe(&[0, 1, 2]);
        // 70% > 10% -> expert 4 is hot.
        assert!(m.is_hot(4, 0.10));
        // 10% threshold: only ids whose count >= 1 (== 10%*10) qualify
        // — i.e. every observed id (0,1,2,4).
        let hot = m.hot_set(0.10);
        assert!(hot.contains(&4), "expected 4 in hot set: {hot:?}");
        // 50% threshold: only expert 4 (>=5 obs) qualifies.
        let hot_strict = m.hot_set(0.50);
        assert_eq!(hot_strict, vec![4]);
    }

    #[test]
    fn locality_monitor_ignores_out_of_range_ids() {
        let m = LocalityMonitor::new(4, 8);
        m.observe(&[0, 1, 99, 2, 100]);
        assert_eq!(m.len(), 3);
        assert!(!m.is_hot(99, 0.0));
    }

    #[test]
    fn locality_monitor_clear_resets_state() {
        let m = LocalityMonitor::new(4, 8);
        m.observe(&[0, 1, 2, 3, 0]);
        assert!(m.len() > 0);
        m.clear();
        assert_eq!(m.len(), 0);
        assert_eq!(m.heat(0), 0);
        assert!(m.hot_set(0.0).is_empty());
    }

    // ---------------------- NeuralSpeculator tests --------------------

    #[test]
    fn speculator_predict_topk_returns_distinct_sorted_ids() {
        let s = NeuralSpeculator::new(16, 32, 8, 1);
        let x: Vec<f32> = (0..16).map(|i| i as f32 * 0.1 - 0.5).collect();
        let top = s.predict_topk(&x, 3);
        assert_eq!(top.len(), 3);
        // distinct
        let mut sorted = top.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 3);
        for id in top {
            assert!(id < 8);
        }
    }

    #[test]
    fn speculator_initial_distribution_is_deterministic() {
        let a = NeuralSpeculator::new(8, 16, 4, 42);
        let b = NeuralSpeculator::new(8, 16, 4, 42);
        let x: Vec<f32> = (0..8).map(|i| (i as f32).sin()).collect();
        assert_eq!(a.forward(&x), b.forward(&x));
    }

    #[test]
    fn speculator_train_step_reduces_loss_on_fixed_target() {
        // Simple test: repeatedly train the speculator against a single
        // (x, target) pair and verify the cross-entropy loss decreases.
        let s = NeuralSpeculator::new(8, 16, 4, 7);
        let x: Vec<f32> = vec![0.5; 8];
        let target = [2u32, 3];
        let initial = s.train_step(&x, &target, 0.1);
        for _ in 0..200 {
            s.train_step(&x, &target, 0.1);
        }
        let after = s.train_step(&x, &target, 0.0); // 0 lr -> just measure loss
        assert!(initial.is_finite());
        assert!(after.is_finite());
        assert!(
            after < initial * 0.9,
            "expected loss to decrease materially: initial={initial} after={after}"
        );
        // After training, the top-2 prediction must include both target ids.
        let preds = s.predict_topk(&x, 2);
        assert!(preds.contains(&2), "preds={preds:?}");
        assert!(preds.contains(&3), "preds={preds:?}");
    }

    #[test]
    fn speculator_handles_empty_or_invalid_input() {
        let s = NeuralSpeculator::new(4, 8, 4, 1);
        assert!(s.predict_topk(&[1.0; 4], 0).is_empty());
        assert!(s.predict_topk(&[], 2).is_empty());
        // Wrong-length input: predict returns empty rather than panicking.
        assert!(s.predict_topk(&[1.0; 8], 2).is_empty());
        // Train with empty actual set -> no-op (NaN sentinel).
        assert!(s.train_step(&[1.0; 4], &[], 0.1).is_nan());
        // Train with all out-of-range ids -> no-op.
        assert!(s.train_step(&[1.0; 4], &[99, 100], 0.1).is_nan());
    }

    #[test]
    fn speculator_predict_topk_with_probs_is_normalised() {
        let s = NeuralSpeculator::new(4, 8, 8, 11);
        let x: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4];
        let preds = s.predict_topk_with_probs(&x, 8);
        assert_eq!(preds.len(), 8);
        let total: f32 = preds.iter().map(|(_, p)| *p).sum();
        assert!((total - 1.0).abs() < 1e-4, "softmax sums to {total}");
        // Sorted descending by probability.
        for w in preds.windows(2) {
            assert!(w[0].1 + 1e-6 >= w[1].1);
        }
    }

    #[test]
    fn locality_monitor_get_hot_experts_aliases_hot_set() {
        // Spec-named alias must be byte-equivalent to `hot_set`.
        let m = LocalityMonitor::new(4, 16);
        for _ in 0..6 { m.observe_one(2); }
        for _ in 0..3 { m.observe_one(0); }
        let from_alias = m.get_hot_experts(0.10);
        let from_orig = m.hot_set(0.10);
        assert_eq!(from_alias, from_orig);
        assert!(from_alias.contains(&2));
    }

    #[test]
    fn predict_unified_blends_three_arms() {
        // 8 experts, fanout 8 so we get every contributing arm in
        // the output. Markov hand-trained to predict 0->1, locality
        // pre-warmed with id=2, speculator predicts id=3.
        let p = PredictiveLoader::new(8, 8, 0.0, 42);
        for _ in 0..20 { p.observe(0, 1); }
        let monitor = LocalityMonitor::new(8, 16);
        for _ in 0..6 { monitor.observe_one(2); }
        let speculator = NeuralSpeculator::new(4, 8, 8, 7);
        let hidden = vec![0.5f32, -0.2, 0.3, 0.1];
        // Force speculator's top-1 to be id=3 by training on id=3
        // many times until that becomes its top output for `hidden`.
        for _ in 0..200 {
            speculator.train_step(&hidden, &[3], 0.1);
        }
        let unified = p.predict_unified(
            None,
            0,
            Some(&monitor),
            0.10,
            Some(&speculator),
            &hidden,
            1,
        );
        // Result is ordered by descending priority.
        for w in unified.windows(2) {
            assert!(
                w[0].1 + 1e-6 >= w[1].1,
                "priority must be sorted descending, got {:?}",
                unified
            );
        }
        // Each of the three contributing ids appears at least once.
        let ids: Vec<u32> = unified.iter().map(|&(id, _)| id).collect();
        for expected in [1u32, 2u32, 3u32] {
            assert!(ids.contains(&expected),
                "predict_unified missing id {expected}: {unified:?}");
        }
        // No score is negative.
        for &(_, s) in &unified {
            assert!(s > 0.0);
        }
    }

    #[test]
    fn predict_unified_handles_no_optional_arms() {
        // With monitor=None and speculator=None the unified ranking
        // should preserve the Markov-only `predict_next` ordering. We
        // use *distinct* observation counts so there are no
        // probability ties — `predict_next` and `predict_unified`
        // break ties slightly differently (the latter uses ascending
        // id as a deterministic tiebreaker, the former is stable in
        // input order), which would otherwise be a benign reordering
        // that this test isn't trying to assert against.
        let p = PredictiveLoader::new(4, 4, 0.0, 42);
        for _ in 0..15 { p.observe(0, 1); }
        for _ in 0..7 { p.observe(0, 2); }
        for _ in 0..3 { p.observe(0, 3); }
        let unified = p.predict_unified(
            None,
            0,
            None,
            0.0,
            None,
            &[],
            0,
        );
        let markov = p.predict_next(0);
        assert_eq!(unified.len(), markov.len());
        for ((id_u, _), (id_m, _)) in unified.iter().zip(markov.iter()) {
            assert_eq!(id_u, id_m);
        }
        // And — the headline guarantee of the normalised weights —
        // a 100%-Markov hit (no arms missing, p≈1) cannot exceed
        // 1.0. Here the top expert has p≈15/25=0.6 and the only
        // contribution is W_MARKOV * p = 0.33 * 0.6 ≈ 0.198, so
        // every score is comfortably below 1.
        for &(_, s) in &unified {
            assert!(s >= 0.0 && s <= 1.0, "score {s} outside [0,1]");
        }
    }

    #[test]
    fn predict_unified_score_is_bounded_by_one() {
        // A unanimous expert (p=1 Markov, hot in locality, top of the
        // speculator) must score ≤ 1.0 under the normalised weights —
        // this is the headline guarantee of the new design.
        let p = PredictiveLoader::new(4, 4, 0.0, 42);
        // Make Markov certain that 0 -> 1 (every observation is the
        // same transition).
        for _ in 0..200 { p.observe(0, 1); }
        let monitor = LocalityMonitor::new(4, 8);
        // Saturate locality so id=1 is well above the 10% threshold.
        for _ in 0..16 { monitor.observe_one(1); }
        let speculator = NeuralSpeculator::new(4, 8, 4, 7);
        let hidden = vec![0.5f32, -0.2, 0.3, 0.1];
        // Train the speculator hard so id=1 dominates the top-K for `hidden`.
        for _ in 0..400 { speculator.train_step(&hidden, &[1], 0.1); }
        let unified = p.predict_unified(
            None,
            0,
            Some(&monitor),
            0.10,
            Some(&speculator),
            &hidden,
            1,
        );
        // id=1 should be the top entry and its score must respect the
        // [0, 1] bound.
        let top = unified.first().expect("non-empty result");
        assert_eq!(top.0, 1);
        assert!(top.1 <= 1.0 + 1e-6, "top score {} exceeded 1.0", top.1);
    }
}

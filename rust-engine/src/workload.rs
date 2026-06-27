//! Workload harness for the streaming benchmark.
//!
//! The default `synthetic` benchmark feeds the engine a fresh
//! `splitmix64` hidden state per token, so the realised routing stream
//! is effectively uniform-i.i.d.: every expert is equally likely and
//! independent of history. On such a stream the best any cache can do is
//! a hit rate equal to its capacity fraction (the "C/E wall"), and a
//! predictor cannot beat chance — which is exactly why the speculative
//! arms look inert on it.
//!
//! Real MoE routing is neither uniform nor memoryless: a handful of
//! experts are activated far more often than the rest (popularity
//! **skew**), and consecutive tokens reuse overlapping expert sets
//! (temporal **correlation**). This module synthesises streams with
//! both properties so the skew-aware (Tier 1 static residency) and
//! correlation-aware (Markov / affinity / Tier 3 pre-gate) machinery is
//! actually *falsifiable*:
//!
//! * [`Workload::Skewed`] — a Zipf popularity distribution over experts
//!   plus a tunable Markov "stay" probability for temporal correlation.
//! * [`Workload::Replay`] — replays a recorded JSONL routing trace (the
//!   `--trace-out` format), so a real model's routing can drive the
//!   cache/predictor harness offline.

use std::path::Path;

/// Which benchmark workload `cmd_run` should drive the engine with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Workload {
    /// Legacy behaviour: the engine routes its own per-token synthetic
    /// hidden state (uniform-i.i.d.), or the `--gate-weights` gate does.
    Synthetic,
    /// Zipf popularity + Markov temporal correlation, generated here and
    /// fed to `moe_step` as an explicit expert set.
    Skewed,
    /// Replay an external JSONL routing trace through `moe_step`.
    Replay,
}

impl Workload {
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "synthetic" | "synth" | "uniform" => Some(Self::Synthetic),
            "skewed" | "skew" | "zipf" => Some(Self::Skewed),
            "replay" | "trace" => Some(Self::Replay),
            _ => None,
        }
    }
}

/// SplitMix64 — the same tiny deterministic PRNG the synthetic hidden
/// state uses, so workloads are reproducible from a seed.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform `f64` in `[0, 1)` with 53 bits of entropy.
#[inline]
fn next_unit(state: &mut u64) -> f64 {
    (splitmix64(state) >> 11) as f64 / ((1u64 << 53) as f64)
}

/// A Zipf-popular, optionally Markov-correlated stream of top-K expert
/// sets. Deterministic given its seed.
#[derive(Debug)]
pub struct SkewedStream {
    top_k: usize,
    /// Probability a token *reuses* the previous token's expert set
    /// (temporal correlation). `0.0` ⇒ i.i.d. draws from the Zipf law.
    rho: f64,
    /// Rank → expert id permutation. Decouples "popularity rank" from
    /// the raw id so the hot set isn't trivially `{0, 1, 2, ...}` (which
    /// could accidentally coincide with on-disk layout).
    rank_to_id: Vec<u32>,
    /// Cumulative distribution over ranks (`cdf[r]` = P(rank <= r)).
    cdf: Vec<f64>,
    state: u64,
    prev: Vec<u32>,
}

impl SkewedStream {
    /// Build a stream over `num_experts` drawing `top_k` distinct
    /// experts per token. `zipf_s` is the Zipf exponent (larger ⇒ more
    /// skew; `1.0` is classic Zipf, `0.0` is uniform). `correlation` is
    /// the Markov stay-probability in `[0, 1]`.
    pub fn new(num_experts: u32, top_k: usize, zipf_s: f64, correlation: f64, seed: u64) -> Self {
        let n = num_experts.max(1) as usize;
        let k = top_k.clamp(1, n);
        let s = zipf_s.max(0.0);

        // Zipf weights w_r = 1 / (r+1)^s over ranks r = 0..n, normalised
        // into a CDF for inverse-transform sampling.
        let mut weights = Vec::with_capacity(n);
        let mut total = 0.0f64;
        for r in 0..n {
            let w = 1.0 / ((r as f64) + 1.0).powf(s);
            total += w;
            weights.push(w);
        }
        let mut cdf = Vec::with_capacity(n);
        let mut acc = 0.0f64;
        for w in &weights {
            acc += w / total;
            cdf.push(acc);
        }
        if let Some(last) = cdf.last_mut() {
            *last = 1.0; // guard against fp drift so the search never falls off the end
        }

        // Deterministic Fisher-Yates shuffle of expert ids → rank map.
        let mut perm_state = seed ^ 0xD1B5_4A32_D192_ED03;
        let mut rank_to_id: Vec<u32> = (0..num_experts.max(1)).collect();
        for i in (1..rank_to_id.len()).rev() {
            let j = (splitmix64(&mut perm_state) % (i as u64 + 1)) as usize;
            rank_to_id.swap(i, j);
        }

        Self {
            top_k: k,
            rho: correlation.clamp(0.0, 1.0),
            rank_to_id,
            cdf,
            state: seed.wrapping_add(0x1234_5678),
            prev: Vec::new(),
        }
    }

    /// Map a uniform `u in [0,1)` to a rank via binary search on the CDF.
    fn sample_rank(&self, u: f64) -> usize {
        match self
            .cdf
            .binary_search_by(|p| p.total_cmp(&u))
        {
            Ok(i) => i,
            Err(i) => i.min(self.cdf.len().saturating_sub(1)),
        }
    }

    /// Produce the next token's top-K expert set.
    pub fn next_experts(&mut self) -> Vec<u32> {
        // Temporal correlation: with probability `rho`, reuse the
        // previous token's set verbatim (a Markov "stay").
        if !self.prev.is_empty() && next_unit(&mut self.state) < self.rho {
            return self.prev.clone();
        }
        let mut chosen: Vec<u32> = Vec::with_capacity(self.top_k);
        let guard_max = self.top_k.saturating_mul(64).max(64);
        let mut guard = 0;
        while chosen.len() < self.top_k && guard < guard_max {
            let u = next_unit(&mut self.state);
            let rank = self.sample_rank(u);
            let id = self.rank_to_id[rank];
            if !chosen.contains(&id) {
                chosen.push(id);
            }
            guard += 1;
        }
        // Degenerate fallback (tiny namespaces / pathological draws):
        // top up deterministically so the set is always full.
        let mut r = 0usize;
        while chosen.len() < self.top_k && r < self.rank_to_id.len() {
            let id = self.rank_to_id[r];
            if !chosen.contains(&id) {
                chosen.push(id);
            }
            r += 1;
        }
        self.prev = chosen.clone();
        chosen
    }
}

/// One replayed routing record from a JSONL trace.
#[derive(Debug, Clone)]
pub struct ReplayRecord {
    pub token: u64,
    pub layer: usize,
    pub experts: Vec<u32>,
}

/// Replays recorded JSONL routing trace records (the `--trace-out` format).
/// Cycles back to the start when exhausted so a short trace can drive an
/// arbitrarily long benchmark.
#[derive(Debug)]
pub struct ReplayStream {
    records: Vec<ReplayRecord>,
    idx: usize,
}

impl ReplayStream {
    /// Load every usable `{"experts":[...]}` record from `path`, in file order.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let mut records = Vec::new();
        let mut record_index = 0u64;
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let value: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            let default_token = record_index;
            record_index += 1;
            if let Some(arr) = value.get("experts").and_then(|e| e.as_array()) {
                let experts: Vec<u32> = arr
                    .iter()
                    .filter_map(|x| x.as_u64())
                    .filter_map(|n| {
                        if n > u32::MAX as u64 {
                            tracing::warn!(
                                expert_id = n,
                                "replay trace expert id exceeds u32::MAX; skipping"
                            );
                            None
                        } else {
                            Some(n as u32)
                        }
                    })
                    .collect();
                if !experts.is_empty() {
                    records.push(ReplayRecord {
                        token: value
                            .get("token")
                            .and_then(|t| t.as_u64())
                            .unwrap_or(default_token),
                        layer: value
                            .get("layer")
                            .and_then(|l| l.as_u64())
                            .unwrap_or(0) as usize,
                        experts,
                    });
                }
            }
        }
        Ok(Self { records, idx: 0 })
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Next replay record, cycling through the trace. `None` only when the
    /// trace contained no usable records.
    pub fn next_record(&mut self) -> Option<ReplayRecord> {
        if self.records.is_empty() {
            return None;
        }
        let rec = self.records[self.idx % self.records.len()].clone();
        self.idx += 1;
        Some(rec)
    }

    /// Next expert set, cycling through the trace. `None` only when the
    /// trace contained no usable records.
    #[allow(dead_code)]
    pub fn next_experts(&mut self) -> Option<Vec<u32>> {
        self.next_record().map(|record| record.experts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn workload_parses_aliases() {
        assert_eq!(Workload::from_str_opt("synthetic"), Some(Workload::Synthetic));
        assert_eq!(Workload::from_str_opt("ZIPF"), Some(Workload::Skewed));
        assert_eq!(Workload::from_str_opt(" replay "), Some(Workload::Replay));
        assert_eq!(Workload::from_str_opt("nope"), None);
    }

    #[test]
    fn skewed_stream_is_actually_skewed() {
        let n = 64u32;
        let mut s = SkewedStream::new(n, 2, /*zipf_s=*/ 1.2, /*correlation=*/ 0.0, 0xABCD);
        let mut counts: HashMap<u32, u64> = HashMap::new();
        for _ in 0..20_000 {
            for id in s.next_experts() {
                *counts.entry(id).or_insert(0) += 1;
            }
        }
        // The most popular rank (rank 0) must dominate the median expert
        // by a wide margin under a Zipf law.
        let hottest_id = s.rank_to_id[0];
        let hottest = *counts.get(&hottest_id).unwrap_or(&0);
        let mut all: Vec<u64> = counts.values().copied().collect();
        all.sort_unstable();
        let median = all[all.len() / 2];
        assert!(
            hottest > median * 5,
            "Zipf hot expert ({hottest}) should dwarf the median ({median})"
        );
    }

    #[test]
    fn skewed_top_set_is_distinct_and_full() {
        let mut s = SkewedStream::new(32, 4, 1.0, 0.0, 7);
        for _ in 0..1000 {
            let e = s.next_experts();
            assert_eq!(e.len(), 4, "always top_k experts");
            let mut uniq = e.clone();
            uniq.sort_unstable();
            uniq.dedup();
            assert_eq!(uniq.len(), e.len(), "experts within a token are distinct");
        }
    }

    #[test]
    fn correlation_increases_consecutive_overlap() {
        // High stay-probability ⇒ many consecutive tokens are identical;
        // zero correlation ⇒ far fewer exact repeats.
        let count_repeats = |rho: f64| {
            let mut s = SkewedStream::new(128, 2, 0.8, rho, 99);
            let mut prev = s.next_experts();
            let mut repeats = 0;
            for _ in 0..5000 {
                let cur = s.next_experts();
                if cur == prev {
                    repeats += 1;
                }
                prev = cur;
            }
            repeats
        };
        let correlated = count_repeats(0.9);
        let independent = count_repeats(0.0);
        assert!(
            correlated > independent * 3,
            "rho=0.9 ({correlated} repeats) must be far more correlated than rho=0 ({independent})"
        );
    }

    #[test]
    fn sample_rank_handles_nan_cdf_entry() {
        let mut s = SkewedStream::new(8, 1, 1.0, 0.0, 11);
        s.cdf[3] = f64::NAN;

        let rank = s.sample_rank(0.5);

        assert!(rank < s.cdf.len(), "rank {rank} should stay within the CDF");
    }

    #[test]
    fn replay_round_trips_a_small_trace() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("replay_test_{}.jsonl", std::process::id()));
        std::fs::write(
            &path,
            "{\"token\":10,\"layer\":2,\"experts\":[3,7]}\n\
             {\"token\":11,\"layer\":3,\"experts\":[1,4,9]}\n\
             {\"experts\":[5]}\n",
        )
        .unwrap();
        let mut r = ReplayStream::load(&path).unwrap();
        assert_eq!(r.len(), 3);
        let first = r.next_record().unwrap();
        assert_eq!(first.token, 10);
        assert_eq!(first.layer, 2);
        assert_eq!(first.experts, vec![3, 7]);
        let second = r.next_record().unwrap();
        assert_eq!(second.token, 11);
        assert_eq!(second.layer, 3);
        assert_eq!(second.experts, vec![1, 4, 9]);
        let third = r.next_record().unwrap();
        assert_eq!(third.token, 2);
        assert_eq!(third.layer, 0);
        assert_eq!(third.experts, vec![5]);
        // Cycles back to the start.
        assert_eq!(r.next_experts(), Some(vec![3, 7]));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn replay_skips_expert_ids_exceeding_u32() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("replay_overflow_{}.jsonl", std::process::id()));
        // 4294967296 == u32::MAX + 1; a silent `as u32` cast would truncate
        // it to 0, so the guard must drop it instead.
        std::fs::write(&path, "{\"experts\":[5,4294967296,7]}\n").unwrap();
        let mut r = ReplayStream::load(&path).unwrap();
        let rec = r.next_record().unwrap();
        assert_eq!(rec.experts, vec![5, 7]);
        let _ = std::fs::remove_file(&path);
    }
}

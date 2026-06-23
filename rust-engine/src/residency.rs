//! Tier 1 — skew-aware **static residency**.
//!
//! ## Why this exists
//!
//! With a uniform-i.i.d. access stream the best any cache can do is a
//! hit rate equal to its capacity fraction (the "C/E wall"): every
//! expert is equally likely, so holding `c` of `e` experts hits `c/e` of
//! the time. Real MoE routing is **not** uniform, though — a handful of
//! experts are activated far more often than the rest. Static residency
//! exploits that skew directly: it permanently pins the hottest
//! `fraction` of experts into the RAM cache so they are *never* streamed
//! from the SSD again, lifting the achievable hit rate above the bare
//! capacity fraction.
//!
//! Two sources of the hot set are supported:
//!
//! * an **offline popularity profile** (`id → count` JSON, e.g. produced
//!   by a previous run's `--profile-out`), applied at startup for an
//!   immediate warm cache, or
//! * an **online** hot set derived from the engine's own
//!   `route_observations` after a warmup window.
//!
//! The whole feature is opt-in (`static_residency_fraction == 0.0`
//! disables it), so default deployments are unchanged.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::AtomicBool;

/// An expert-popularity profile: global expert id → observation count.
///
/// Serialised as a flat JSON object with string keys (`{"0": 1234,
/// "5": 42, ...}`) so it round-trips through `serde_json` without a
/// custom map-key codec and stays human-inspectable.
#[derive(Debug, Clone, Default)]
pub struct ResidencyProfile {
    counts: HashMap<u32, u64>,
}

impl ResidencyProfile {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_counts(counts: HashMap<u32, u64>) -> Self {
        Self { counts }
    }

    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    pub fn len(&self) -> usize {
        self.counts.len()
    }

    /// Observation count for `id` (0 when unseen).
    pub fn get(&self, id: u32) -> u64 {
        self.counts.get(&id).copied().unwrap_or(0)
    }

    /// Record one activation of `id`.
    pub fn observe(&mut self, id: u32) {
        *self.counts.entry(id).or_insert(0) += 1;
    }

    /// Load a profile from a JSON object `{ "<id>": <count>, ... }`.
    pub fn load_json(path: &Path) -> std::io::Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        let string_keyed: HashMap<String, u64> = serde_json::from_str(&raw)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let mut counts = HashMap::with_capacity(string_keyed.len());
        for (k, v) in string_keyed {
            let id: u32 = k
                .parse()
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            counts.insert(id, v);
        }
        Ok(Self { counts })
    }

    /// Serialise to a JSON object `{ "<id>": <count>, ... }`. Keys are
    /// emitted in ascending id order so successive dumps diff cleanly.
    pub fn dump_json(&self, path: &Path) -> std::io::Result<()> {
        let mut ordered: Vec<(u32, u64)> = self.counts.iter().map(|(k, v)| (*k, *v)).collect();
        ordered.sort_unstable_by_key(|(k, _)| *k);
        // Build an order-preserving string map for a stable on-disk form.
        let mut buf = String::from("{");
        for (i, (id, count)) in ordered.iter().enumerate() {
            if i > 0 {
                buf.push(',');
            }
            buf.push_str(&format!("\"{id}\":{count}"));
        }
        buf.push('}');
        std::fs::write(path, buf)
    }

    /// The hottest `ceil(fraction × namespace)` expert ids, most-popular
    /// first. Ties break by ascending id for determinism. `fraction` is
    /// clamped to `[0, 1]`; `namespace` is the total expert count used to
    /// size the pin budget (so the budget is a fraction of *all* experts,
    /// not just the observed ones).
    pub fn hot_set(&self, fraction: f64, namespace: usize) -> Vec<u32> {
        let frac = fraction.clamp(0.0, 1.0);
        if frac == 0.0 || namespace == 0 || self.counts.is_empty() {
            return Vec::new();
        }
        let budget = ((namespace as f64) * frac).ceil() as usize;
        if budget == 0 {
            return Vec::new();
        }
        let mut ranked: Vec<(u32, u64)> = self.counts.iter().map(|(k, v)| (*k, *v)).collect();
        // Sort by count desc, then id asc. Experts never observed are not
        // in the map and are correctly excluded.
        ranked.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        ranked.truncate(budget);
        ranked.into_iter().map(|(id, _)| id).collect()
    }
}

/// Engine-side controller state for static residency. Held behind an
/// `Option` on the speculation struct; `None` means the feature is off.
#[derive(Debug)]
pub struct StaticResidencyState {
    /// Fraction of the global expert namespace to pin (`(0, 1]`).
    pub fraction: f64,
    /// Tokens to observe before deriving an *online* hot set. Ignored
    /// when a seed `profile` is supplied (that is applied immediately).
    pub warmup_tokens: u64,
    /// Total expert count, used to size the pin budget.
    pub namespace: usize,
    /// Optional offline seed profile. When `Some`, its hot set is pinned
    /// at the first opportunity with no warmup; when `None`, the hot set
    /// is derived online from the engine's `route_observations`.
    pub profile: Option<ResidencyProfile>,
    /// One-shot latch: set once the hot set has been pinned so the
    /// engine applies it exactly once.
    pub applied: AtomicBool,
}

impl StaticResidencyState {
    pub fn new(
        fraction: f64,
        warmup_tokens: u64,
        namespace: usize,
        profile: Option<ResidencyProfile>,
    ) -> Self {
        Self {
            fraction,
            warmup_tokens,
            namespace,
            profile,
            applied: AtomicBool::new(false),
        }
    }

    /// Whether this controller derives its hot set online (no seed
    /// profile), and therefore needs `route_observations` populated.
    pub fn is_online(&self) -> bool {
        self.profile.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hot_set_picks_top_fraction_by_count() {
        let mut p = ResidencyProfile::new();
        // ids 0..10, counts increasing with id.
        for id in 0..10u32 {
            for _ in 0..=id {
                p.observe(id);
            }
        }
        // 20% of a 10-expert namespace = 2 experts: the two hottest (9, 8).
        let hot = p.hot_set(0.2, 10);
        assert_eq!(hot, vec![9, 8]);
    }

    #[test]
    fn hot_set_zero_fraction_is_empty() {
        let mut p = ResidencyProfile::new();
        p.observe(1);
        assert!(p.hot_set(0.0, 100).is_empty());
    }

    #[test]
    fn hot_set_ceils_budget() {
        let mut p = ResidencyProfile::new();
        for id in 0..3u32 {
            p.observe(id);
        }
        // ceil(0.1 * 3) = 1.
        assert_eq!(p.hot_set(0.1, 3).len(), 1);
    }

    #[test]
    fn hot_set_breaks_ties_by_ascending_id() {
        let mut p = ResidencyProfile::new();
        // All equal counts → tie broken by id, so the lowest ids win.
        for id in [7u32, 3, 9, 1] {
            p.observe(id);
        }
        let hot = p.hot_set(0.5, 4); // ceil(0.5*4)=2
        assert_eq!(hot, vec![1, 3]);
    }

    #[test]
    fn json_round_trips() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("residency_test_{}.json", std::process::id()));
        let mut p = ResidencyProfile::new();
        p.observe(5);
        p.observe(5);
        p.observe(42);
        p.dump_json(&path).unwrap();
        let loaded = ResidencyProfile::load_json(&path).unwrap();
        assert_eq!(loaded.get(5), 2);
        assert_eq!(loaded.get(42), 1);
        assert_eq!(loaded.get(0), 0);
        let _ = std::fs::remove_file(&path);
    }
}

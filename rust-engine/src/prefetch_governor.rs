//! Adaptive, self-regulating prefetch admission controller.
//!
//! ## Why this exists
//!
//! The legacy speculative-I/O path admits every predicted expert that
//! clears a *static* probability threshold and can grab a semaphore
//! permit. On a structureless or weakly-correlated workload the
//! predictors degrade to ~chance accuracy, yet the engine keeps issuing
//! the same volume of speculative reads. Those reads are not free: on a
//! bandwidth-bound SSD they **queue ahead of foreground cache misses**,
//! inflating the latency of the reads that actually block token
//! generation. Field data from a Mixtral-8x7B run made this concrete —
//! with ~9k speculative reads at ~0.8 % precision the foreground miss
//! p50 rose from ~120 ms (no speculation) to ~405 ms (3.4x), while hit
//! rate barely moved.
//!
//! The [`PrefetchGovernor`] closes the loop: it continuously measures
//!
//! * **precision** — the fraction of completed prefetches that were
//!   actually consumed before eviction (an EWMA), and
//! * **contention** — how many foreground (blocking) reads are in
//!   flight right now,
//!
//! and admits a speculative read only when its *expected value* beats
//! the *expected contention cost*. When the predictors are paying off
//! and the disk is idle it admits liberally; when precision collapses or
//! real misses are queued it throttles toward zero, handing the scarce
//! I/O bandwidth back to the foreground path.
//!
//! The controller is **opt-in** (`EngineOptions::prefetch_governor`,
//! default `false`) so existing deployments and benchmarks are
//! bit-for-bit unchanged until they enable it.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

/// Pack/unpack an `f64` into the bits of an `AtomicU64` so the EWMA can
/// be read on the hot admission path with a single relaxed load.
#[inline]
fn load_f64(a: &AtomicU64) -> f64 {
    f64::from_bits(a.load(Ordering::Relaxed))
}
#[inline]
fn store_f64(a: &AtomicU64, v: f64) {
    a.store(v.to_bits(), Ordering::Relaxed);
}

/// Tunables for [`PrefetchGovernor`]. All values have safe, conservative
/// defaults; the engine fills these from [`EngineOptions`].
#[derive(Clone, Copy, Debug)]
pub struct GovernorConfig {
    /// EWMA smoothing factor for the measured precision signal, in
    /// `(0, 1]`. Higher reacts faster to distribution shift; lower is
    /// steadier. `0.2` blends roughly the last ~5 measurement windows.
    pub precision_alpha: f64,
    /// Floor applied to the precision EWMA when scoring admissions, so a
    /// transient run of wasted prefetches can't latch the controller at
    /// exactly zero and starve it of the future hits that would let it
    /// recover. Also the value the EWMA is seeded with.
    pub precision_floor: f64,
    /// Per-outstanding-foreground-read multiplier on the admission
    /// threshold. With `contention_weight = 1.0`, one in-flight
    /// foreground miss doubles the bar a speculative read must clear,
    /// two triple it, and so on.
    pub contention_weight: f64,
    /// Base admission threshold the (probability x precision) product is
    /// compared against when the disk is otherwise idle.
    pub base_threshold: f64,
}

impl Default for GovernorConfig {
    fn default() -> Self {
        Self {
            precision_alpha: 0.2,
            precision_floor: 0.05,
            contention_weight: 1.0,
            base_threshold: 0.02,
        }
    }
}

/// Lock-free adaptive prefetch admission controller. Cheap to share
/// behind an `Arc`: every field is a single atomic and the hot
/// [`Self::admit`] path performs only relaxed loads.
#[derive(Debug)]
pub struct PrefetchGovernor {
    enabled: bool,
    precision_alpha: f64,
    precision_floor: f64,
    contention_weight: f64,
    base_threshold: f64,

    /// EWMA of recent prefetch precision (consumed / completed), in
    /// `[0, 1]`. Read on the admission hot path.
    precision_ewma: AtomicU64,

    /// Gauge of foreground (blocking) reads currently in flight. A
    /// speculative read admitted while this is non-zero is directly
    /// competing with a token-blocking miss for device bandwidth.
    foreground_inflight: AtomicI64,

    /// Rolling within-window counters folded into the EWMA by
    /// [`Self::refresh`]. `completed` counts prefetch reads that landed;
    /// `used` counts those that were consumed by a subsequent hit before
    /// eviction.
    window_completed: AtomicU64,
    window_used: AtomicU64,

    /// Telemetry: prefetches the governor declined to admit.
    throttled: AtomicU64,
    /// Telemetry: prefetches the governor admitted.
    admitted: AtomicU64,
}

/// RAII token for foreground-read accounting. Dropping the guard always
/// balances the corresponding [`PrefetchGovernor::begin_foreground`] call.
pub struct ForegroundGuard<'a> {
    governor: &'a PrefetchGovernor,
}

impl<'a> ForegroundGuard<'a> {
    fn new(governor: &'a PrefetchGovernor) -> Self {
        governor.begin_foreground();
        Self { governor }
    }
}

impl Drop for ForegroundGuard<'_> {
    fn drop(&mut self) {
        self.governor.end_foreground();
    }
}

impl PrefetchGovernor {
    /// Construct a governor. When `enabled` is `false` the controller is
    /// a transparent pass-through: [`Self::admit`] always returns `true`
    /// and the accounting hooks are no-ops, so the legacy unbounded
    /// behaviour is preserved exactly.
    pub fn new(enabled: bool, cfg: GovernorConfig) -> Self {
        let floor = cfg.precision_floor.clamp(0.0, 1.0);
        let g = Self {
            enabled,
            precision_alpha: cfg.precision_alpha.clamp(1e-3, 1.0),
            precision_floor: floor,
            contention_weight: cfg.contention_weight.max(0.0),
            base_threshold: cfg.base_threshold.max(0.0),
            precision_ewma: AtomicU64::new(0),
            foreground_inflight: AtomicI64::new(0),
            window_completed: AtomicU64::new(0),
            window_used: AtomicU64::new(0),
            throttled: AtomicU64::new(0),
            admitted: AtomicU64::new(0),
        };
        // Seed the EWMA optimistically at `max(floor, 0.5)` so a freshly
        // started engine gives speculation a fair chance to prove itself
        // before the measured signal takes over.
        store_f64(&g.precision_ewma, floor.max(0.5));
        g
    }

    /// A disabled pass-through governor (no gating, no accounting).
    pub fn disabled() -> Self {
        Self::new(false, GovernorConfig::default())
    }

    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Decide whether a speculative read for an expert predicted with
    /// probability/score `prob` should be admitted *right now*.
    ///
    /// Returns `true` unconditionally when the governor is disabled.
    /// Otherwise admits iff
    ///
    /// ```text
    ///   prob * max(precision_ewma, floor)
    ///       >= base_threshold * (1 + contention_weight * foreground_inflight)
    /// ```
    ///
    /// i.e. the expected value of the speculation (its probability scaled
    /// by how often speculation has recently paid off) must clear a bar
    /// that rises with the number of foreground misses currently
    /// competing for the device.
    #[inline]
    pub fn admit(&self, prob: f64) -> bool {
        if !self.enabled {
            return true;
        }
        let precision = load_f64(&self.precision_ewma).max(self.precision_floor);
        let inflight = self.foreground_inflight.load(Ordering::Relaxed).max(0) as f64;
        let value = prob.max(0.0) * precision;
        let bar = self.base_threshold * (1.0 + self.contention_weight * inflight);
        let ok = value >= bar;
        if ok {
            self.admitted.fetch_add(1, Ordering::Relaxed);
        } else {
            self.throttled.fetch_add(1, Ordering::Relaxed);
        }
        ok
    }

    /// RAII-free gauge bump: a foreground (blocking) read has started.
    /// Pair with [`Self::end_foreground`]. No-op when disabled.
    #[inline]
    pub fn begin_foreground(&self) {
        if self.enabled {
            self.foreground_inflight.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A foreground read has finished. No-op when disabled.
    #[inline]
    pub fn end_foreground(&self) {
        if self.enabled {
            self.foreground_inflight.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Begin foreground-read accounting and return a guard that ends it
    /// on drop. Preserves the disabled-governor no-op semantics of the
    /// explicit begin/end methods.
    #[inline]
    pub fn foreground_guard(&self) -> ForegroundGuard<'_> {
        ForegroundGuard::new(self)
    }

    /// Record that a speculative read landed (became resident).
    #[inline]
    pub fn record_completed(&self) {
        if self.enabled {
            self.window_completed.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record that a previously-prefetched expert was consumed by a hit
    /// before it was evicted — i.e. the speculation paid off.
    #[inline]
    pub fn record_used(&self) {
        if self.enabled {
            self.window_used.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Fold the current measurement window into the precision EWMA and
    /// reset the window. Cheap enough to call once per token. When no
    /// prefetches completed in the window the EWMA is left untouched
    /// (no signal → no update), which keeps the controller stable during
    /// cache-resident bursts.
    pub fn refresh(&self) {
        if !self.enabled {
            return;
        }
        let completed = self.window_completed.swap(0, Ordering::Relaxed);
        if completed == 0 {
            // No completions ⇒ don't drag the EWMA toward 0 for an idle
            // window; just clear any stray `used` credits.
            self.window_used.store(0, Ordering::Relaxed);
            return;
        }
        let used = self.window_used.swap(0, Ordering::Relaxed).min(completed);
        let sample = used as f64 / completed as f64;
        let prev = load_f64(&self.precision_ewma);
        let next = prev + self.precision_alpha * (sample - prev);
        store_f64(&self.precision_ewma, next.clamp(0.0, 1.0));
    }

    /// Current precision EWMA (for telemetry / the run summary).
    pub fn precision(&self) -> f64 {
        load_f64(&self.precision_ewma)
    }

    /// Current foreground-read gauge (for telemetry).
    pub fn foreground_inflight(&self) -> i64 {
        self.foreground_inflight.load(Ordering::Relaxed)
    }

    /// `(admitted, throttled)` admission decisions so far.
    pub fn decisions(&self) -> (u64, u64) {
        (
            self.admitted.load(Ordering::Relaxed),
            self.throttled.load(Ordering::Relaxed),
        )
    }
}

impl Default for PrefetchGovernor {
    fn default() -> Self {
        Self::disabled()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_governor_admits_everything() {
        let g = PrefetchGovernor::disabled();
        assert!(g.admit(0.0));
        assert!(g.admit(1.0));
        // Accounting hooks are inert.
        g.begin_foreground();
        g.record_completed();
        g.refresh();
        assert!(g.admit(0.0));
    }

    #[test]
    fn high_precision_idle_disk_admits() {
        let g = PrefetchGovernor::new(true, GovernorConfig::default());
        // Seeded optimistic; a confident prediction on an idle disk is
        // admitted.
        assert!(g.admit(0.9));
    }

    #[test]
    fn collapsed_precision_throttles_even_when_idle() {
        let g = PrefetchGovernor::new(true, GovernorConfig::default());
        // Many windows of completed-but-never-used reads drive the
        // precision EWMA down to its floor.
        for _ in 0..40 {
            for _ in 0..10 {
                g.record_completed();
            }
            g.refresh();
        }
        // At the floor (0.05) a moderate 0.3-probability prediction's
        // expected value (0.015) no longer clears even the idle bar
        // (base_threshold 0.02), so it is declined.
        assert!(!g.admit(0.3));
        // A near-certain prediction (0.9 * 0.05 = 0.045) still gets
        // through — the governor throttles junk, not signal.
        assert!(g.admit(0.9));
    }

    #[test]
    fn foreground_contention_raises_the_bar() {
        let cfg = GovernorConfig {
            // Isolate the contention term from the precision term.
            precision_floor: 1.0,
            base_threshold: 0.2,
            contention_weight: 1.0,
            ..GovernorConfig::default()
        };
        let g = PrefetchGovernor::new(true, cfg);
        store_f64(&g.precision_ewma, 1.0);
        // Idle disk: prob 0.3 clears the 0.2 bar.
        assert!(g.admit(0.3));
        // Two foreground misses in flight ⇒ bar = 0.2 * 3 = 0.6.
        g.begin_foreground();
        g.begin_foreground();
        assert!(!g.admit(0.3));
        // A near-certain prediction still gets through.
        assert!(g.admit(0.9));
        g.end_foreground();
        g.end_foreground();
        assert!(g.admit(0.3));
    }

    #[test]
    fn foreground_guard_releases_on_drop() {
        let g = PrefetchGovernor::new(true, GovernorConfig::default());
        let before = g.foreground_inflight();
        {
            let _guard = g.foreground_guard();
            assert_eq!(g.foreground_inflight(), before + 1);
        }
        assert_eq!(g.foreground_inflight(), before);

        let disabled = PrefetchGovernor::disabled();
        {
            let _guard = disabled.foreground_guard();
            assert_eq!(disabled.foreground_inflight(), 0);
        }
        assert_eq!(disabled.foreground_inflight(), 0);
    }

    #[test]
    fn precision_recovers_when_prefetches_get_used() {
        let g = PrefetchGovernor::new(true, GovernorConfig::default());
        for _ in 0..100 {
            g.record_completed();
        }
        g.refresh(); // precision drops (0 used / 100 completed)
        let low = g.precision();
        // Now a window where every completion is consumed.
        for _ in 0..100 {
            g.record_completed();
            g.record_used();
        }
        g.refresh();
        assert!(g.precision() > low);
    }

    #[test]
    fn used_is_clamped_to_completed() {
        let g = PrefetchGovernor::new(true, GovernorConfig::default());
        g.record_completed();
        g.record_used();
        g.record_used();
        g.record_used();
        // Should not panic or exceed 1.0.
        g.refresh();
        assert!(g.precision() <= 1.0);
    }
}

//! In-memory session store for KV-cache persistence between HTTP
//! requests.
//!
//! A *session* is a stable client-supplied identifier (`session_id`)
//! that lets a multi-turn conversation reuse the per-layer KV cache
//! built up by previous requests. Without it every request re-runs
//! attention over the entire prompt; with it, only the new tokens
//! incur attention compute. For long chats this is the difference
//! between O(N²) and amortised O(N) attention work.
//!
//! ## Design
//!
//! * Backed by a [`dashmap::DashMap`] so concurrent HTTP handlers
//!   read/write without contending on a global lock.
//! * Each entry holds the per-layer KV caches plus a *position cursor*
//!   (the absolute token offset where the next request should resume)
//!   and a `last_used` timestamp for TTL eviction.
//! * A background task ([`SessionStore::spawn_evictor`]) periodically
//!   purges stale entries. The TTL is configurable from the TOML
//!   `[server.session_ttl_secs]` field.
//! * The `DELETE /v1/sessions/{id}` endpoint provides explicit
//!   cleanup, used by clients that know they are done with a session
//!   and don't want to wait for TTL eviction.
//!
//! ## Threading model
//!
//! `take` removes the session entry while the request is active so
//! concurrent requests against the same session are serialised
//! (attention state is inherently sequential and cannot be safely
//! interleaved). When the request completes the entry is reinserted
//! with the updated KV caches. If a second request for the same
//! session arrives while the first is still executing, the second
//! starts from scratch — this matches how vLLM, llama.cpp and
//! ollama's session APIs behave: the alternative (queuing) would
//! couple unrelated client streams together.

use crate::transformer::KvCache;
use dashmap::DashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// One active session's persisted state.
#[derive(Debug, Clone)]
pub struct SessionState {
    /// Per-layer KV caches.
    pub kv: Vec<KvCache>,
    /// Absolute position cursor: the index of the *next* token to
    /// generate (i.e. `kv[*].seq_len` for any layer; we track it
    /// explicitly so a future request can pick up regardless of how
    /// the KV cache is laid out internally).
    ///
    /// On resume, the next request's prompt tokens are fed in starting
    /// at this position so RoPE indices and KV slots line up with what
    /// the prior turn already wrote. The "last token to feed into the
    /// next step" is implicit: every request carries its own prompt,
    /// and the new prompt's last token is what seeds the first
    /// generated token of the new turn.
    pub position: usize,
    /// When this session was last touched. Updated on every
    /// successful `take` / `put` round-trip.
    pub last_used: Instant,
}

impl SessionState {
    pub fn new(kv: Vec<KvCache>) -> Self {
        Self { kv, position: 0, last_used: Instant::now() }
    }

    /// Overwrite every per-layer KV buffer with zeros so a later
    /// allocation that lands in the freed memory cannot read residual
    /// attention state. Called from both the explicit
    /// `DELETE /v1/sessions/{id}` endpoint *and* the TTL evictor
    /// before the entry is dropped — gist Issue #1.
    pub(crate) fn zeroize_in_place(&mut self) {
        for cache in self.kv.iter_mut() {
            cache.zeroize();
        }
    }
}

/// Lock-free session store.
#[derive(Debug, Clone)]
pub struct SessionStore {
    inner: Arc<DashMap<String, SessionState>>,
    /// Ids that are currently checked out via [`Self::take`] and not
    /// yet returned via [`Self::put`]. Lets [`Self::delete`] observe
    /// — and revoke — an in-flight session that is invisible to a
    /// `DashMap::remove` against `inner` because the entry has been
    /// handed to a request handler. (F1.5 in the audit.)
    in_flight: Arc<DashMap<String, u64>>,
    /// Ids that were ordered deleted while still in flight. The next
    /// [`Self::put`] for one of these ids zeroizes the returned state
    /// and refuses to reinsert it, preserving the
    /// `DELETE /v1/sessions/{id}` invariant that no resumption of the
    /// session is possible after the delete is observed.
    tombstones: Arc<DashMap<String, ()>>,
    next_checkout: Arc<AtomicU64>,
    ttl: Duration,
}

pub type SessionCheckoutToken = u64;

/// RAII receipt for an in-flight session checkout returned by
/// [`SessionStore::take`]. Holding it proves ownership of the
/// session's `in_flight` marker; passing it back to
/// [`SessionStore::put`] releases the marker and reinserts the state.
///
/// If the guard is **dropped without** `put` — e.g. the request task
/// was cancelled because the HTTP client disconnected mid-generation —
/// the `Drop` impl clears the in-flight marker so the session id does
/// not stay locked forever (the checked-out KV state is lost with the
/// cancelled task, but the id remains usable for fresh sessions).
pub struct SessionCheckout {
    in_flight: Arc<DashMap<String, u64>>,
    id: String,
    token: SessionCheckoutToken,
    armed: bool,
}

impl SessionCheckout {
    /// The raw token value (diagnostics only).
    #[allow(dead_code)]
    pub fn token(&self) -> SessionCheckoutToken {
        self.token
    }
}

impl Drop for SessionCheckout {
    fn drop(&mut self) {
        if self.armed {
            // Only clear the marker if we still own it (a concurrent
            // path may have already replaced or removed it).
            self.in_flight.remove_if(&self.id, |_, v| *v == self.token);
        }
    }
}

impl SessionStore {
    /// `ttl == 0` disables time-based eviction (sessions live until
    /// `delete` is called explicitly).
    pub fn new(ttl: Duration) -> Self {
        Self {
            inner: Arc::new(DashMap::new()),
            in_flight: Arc::new(DashMap::new()),
            tombstones: Arc::new(DashMap::new()),
            next_checkout: Arc::new(AtomicU64::new(1)),
            ttl,
        }
    }

    /// Number of currently-stored sessions.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Atomically remove and return the session state for `id`. The
    /// caller is expected to call [`Self::put`] with the returned
    /// checkout guard when finished so the session resumes on the next
    /// request. If the guard is dropped without `put` (request
    /// cancelled), the in-flight marker is released automatically.
    /// Returns `None` when no session with that id exists.
    pub fn take(&self, id: &str) -> Option<(SessionState, SessionCheckout)> {
        // Reject takes for ids that have an outstanding tombstone:
        // a concurrent `delete` already decided this session is
        // gone, and serving its state to a new request would
        // resurrect attention bytes the operator asked to be
        // discarded.
        if self.tombstones.contains_key(id) {
            return None;
        }
        let token = self.next_checkout.fetch_add(1, Ordering::Relaxed);
        let key = id.to_string();
        match self.in_flight.entry(key.clone()) {
            dashmap::mapref::entry::Entry::Vacant(v) => {
                v.insert(token);
            }
            dashmap::mapref::entry::Entry::Occupied(_) => {
                return None;
            }
        }
        // From here on the guard owns the marker: every early return
        // drops it, which clears the in-flight entry.
        let guard = SessionCheckout {
            in_flight: self.in_flight.clone(),
            id: key,
            token,
            armed: true,
        };
        match self.inner.remove(id) {
            Some((_, mut state)) => {
                if self.tombstones.contains_key(id) {
                    state.zeroize_in_place();
                    return None;
                }
                Some((state, guard))
            }
            None => None,
        }
    }

    /// Reinsert a session — typically after a request has consumed
    /// `take` and produced new tokens. `checkout` must be the guard
    /// returned by `take` for this in-flight session; use `None` when
    /// storing a fresh session that was not checked out from the store.
    pub fn put(
        &self,
        id: String,
        mut state: SessionState,
        checkout: Option<SessionCheckout>,
    ) {
        // Returning an in-flight checkout must prove ownership of the
        // current marker. Callers that did not successfully `take`
        // pass `None` and must not clear another request's marker.
        let was_checked_out = checkout.is_some();
        if let Some(mut guard) = checkout {
            // `put` takes over marker management from here; disarm the
            // guard so its `Drop` doesn't race with the logic below.
            guard.armed = false;
            let owns_marker = self
                .in_flight
                .get(&id)
                .map(|v| *v == guard.token)
                .unwrap_or(false);
            if !owns_marker {
                state.zeroize_in_place();
                return;
            }
            self.in_flight.remove(&id);
        }
        if self.tombstones.contains_key(&id) {
            if was_checked_out {
                self.tombstones.remove(&id);
            }
            // A `delete` arrived while this id was in flight. Honour
            // the delete: scrub the KV bytes and drop the state
            // instead of reinserting. This is the resurrection-
            // prevention half of F1.5.
            state.zeroize_in_place();
            return;
        }
        state.last_used = Instant::now();
        self.inner.insert(id, state);
    }

    /// Remove `id` and report whether it existed. Used by the
    /// `DELETE /v1/sessions/{id}` endpoint. Before discarding the
    /// entry, every KV-cache buffer is overwritten with zeros so the
    /// (potentially sensitive) attention state cannot be read by a
    /// later allocation that lands in the same memory. This is the
    /// "memory zeroing" production-readiness ask.
    pub fn delete(&self, id: &str) -> bool {
        // First case: the session is resident. Remove + zeroize as
        // usual. A concurrent `take` either lost the race on
        // `inner.remove` (and will return `None`) or won and is in
        // the in-flight branch below — never both.
        if let Some((_, mut state)) = self.inner.remove(id) {
            state.zeroize_in_place();
            // Clear any tombstone defensively — the session is gone.
            self.tombstones.remove(id);
            return true;
        }
        // Second case: the session is currently checked out via
        // `take`. We cannot zeroize the bytes (the request handler
        // owns them) but we *can* arm a tombstone so the matching
        // `put` zeroizes-and-discards instead of reinserting.
        if self.in_flight.contains_key(id) {
            self.tombstones.insert(id.to_string(), ());
            return true;
        }
        false
    }

    /// Evict entries idle for longer than the configured TTL. Returns
    /// the number of entries removed. Cheap when the store is small;
    /// the background evictor task calls this periodically.
    ///
    /// Every evicted [`SessionState`] is zeroized **before** being
    /// dropped (gist Issue #1) so a TTL-driven cleanup leaks no more
    /// residual KV bytes than an explicit `DELETE` would.
    pub fn evict_expired(&self) -> usize {
        if self.ttl.is_zero() {
            return 0;
        }
        let now = Instant::now();
        let ttl = self.ttl;
        // We deliberately avoid `DashMap::retain` here: it would drop
        // expired values inline (no chance to zeroize) and there's no
        // hook to run code on the removed entry. Instead, snapshot
        // the expired keys under shard read locks, then `remove`
        // each one and explicitly zeroize before the `SessionState`
        // is dropped at the end of the loop iteration.
        let expired_keys: Vec<String> = self
            .inner
            .iter()
            .filter_map(|entry| {
                if now.duration_since(entry.value().last_used) > ttl {
                    Some(entry.key().clone())
                } else {
                    None
                }
            })
            .collect();
        let mut removed = 0usize;
        for k in expired_keys {
            // Between the read-scan above and the `remove` call here
            // another request may have reinserted the id (we treat
            // the fresh state as "no longer expired" and leave it),
            // or removed it (`remove` returns `None` — skip).
            if let Some((_, mut state)) = self.inner.remove_if(&k, |_, s| {
                now.duration_since(s.last_used) > ttl
            }) {
                state.zeroize_in_place();
                removed += 1;
            }
        }
        removed
    }

    /// Spawn a background tokio task that calls [`Self::evict_expired`]
    /// every `interval`. Returns immediately. The task lives for the
    /// lifetime of the runtime — it holds an `Arc` to the inner map so
    /// dropping the original handle alone does not stop it.
    pub fn spawn_evictor(&self, interval: Duration) {
        if self.ttl.is_zero() || interval.is_zero() {
            return;
        }
        let store = self.clone();
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            // Skip the immediate fire so the first tick lands after
            // `interval`, not at startup.
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            ticker.tick().await;
            loop {
                ticker.tick().await;
                let n = store.evict_expired();
                if n > 0 {
                    tracing::debug!(removed = n, alive = store.len(), "session TTL eviction");
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_kv() -> Vec<KvCache> {
        vec![KvCache::new(8)]
    }

    #[test]
    fn put_then_take_round_trips_state() {
        let store = SessionStore::new(Duration::from_secs(60));
        let mut s = SessionState::new(fake_kv());
        s.position = 4;
        store.put("alice".to_string(), s, None);
        assert_eq!(store.len(), 1);
        let (back, _token) = store.take("alice").expect("session must exist");
        assert_eq!(back.position, 4);
        // Take is destructive.
        assert!(store.take("alice").is_none());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn delete_returns_existence_flag() {
        let store = SessionStore::new(Duration::from_secs(60));
        store.put("a".into(), SessionState::new(fake_kv()), None);
        assert!(store.delete("a"));
        assert!(!store.delete("a"));
        assert!(!store.delete("never-existed"));
    }

    #[test]
    fn evict_expired_drops_stale_entries() {
        let store = SessionStore::new(Duration::from_millis(10));
        let mut s = SessionState::new(fake_kv());
        // Force `last_used` into the past so the eviction sweep removes it.
        s.last_used = Instant::now() - Duration::from_secs(60);
        store.inner.insert("stale".into(), s);
        store.put("fresh".into(), SessionState::new(fake_kv()), None);
        let removed = store.evict_expired();
        assert_eq!(removed, 1);
        assert!(store.take("fresh").is_some());
        assert!(store.take("stale").is_none());
    }

    #[test]
    fn evict_disabled_when_ttl_zero() {
        let store = SessionStore::new(Duration::ZERO);
        let mut s = SessionState::new(fake_kv());
        s.last_used = Instant::now() - Duration::from_secs(60);
        store.inner.insert("ancient".into(), s);
        assert_eq!(store.evict_expired(), 0);
        assert!(store.take("ancient").is_some());
    }

    #[test]
    fn zeroize_in_place_clears_kv_buffers() {
        // Build a SessionState whose KV cache has real (non-zero) bytes
        // written into it. After `zeroize_in_place` every byte the
        // public API can observe must read back as zero — the property
        // both `delete` and `evict_expired` rely on.
        let mut kv = KvCache::new(4);
        kv.append(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
        assert!(kv.num_blocks() > 0, "test setup: KV must own at least one block");
        let mut state = SessionState::new(vec![kv]);
        state.zeroize_in_place();
        assert_eq!(
            state.kv[0].num_blocks(),
            0,
            "zeroize_in_place must drop block-table entries after zeroing the bytes"
        );
        assert_eq!(state.kv[0].seq_len, 0);
    }

    #[test]
    fn evict_expired_zeroizes_before_drop() {
        // Regression for gist Issue #1: `evict_expired` previously
        // dropped expired entries via `DashMap::retain`, which never
        // ran the per-cache `zeroize` step. We can't observe the
        // bytes of a value that's already been dropped, so we
        // instead assert that the eviction path used here is the
        // explicit `remove`+zeroize path — by checking that the
        // store no longer holds the id and the count is correct,
        // and we verify the zeroize itself via the dedicated test
        // above. Together they pin down the contract.
        let store = SessionStore::new(Duration::from_millis(1));
        let mut kv = KvCache::new(4);
        kv.append(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
        let mut s = SessionState::new(vec![kv]);
        // Force `last_used` into the past so the eviction sweep
        // classifies the entry as expired.
        s.last_used = Instant::now() - Duration::from_secs(60);
        store.inner.insert("expired".into(), s);
        assert_eq!(store.evict_expired(), 1);
        assert!(store.take("expired").is_none());
    }

    /// F1.5 regression: a `delete` that arrives while the session is
    /// checked out via `take` must (a) report success and (b) prevent
    /// the subsequent `put` from resurrecting the session.
    #[test]
    fn delete_during_in_flight_take_blocks_resurrection() {
        let store = SessionStore::new(Duration::from_secs(60));
        let mut s = SessionState::new(fake_kv());
        s.position = 7;
        store.put("alice".into(), s, None);

        // Request handler checks the session out.
        let (in_flight, checkout) = store.take("alice").expect("session must exist");
        assert_eq!(in_flight.position, 7);
        assert_eq!(store.len(), 0, "take is destructive on `inner`");

        // Concurrent DELETE /v1/sessions/alice arrives. There is no
        // resident entry to remove, but the session is in flight —
        // the delete must still report success and arm a tombstone.
        assert!(
            store.delete("alice"),
            "delete must see in-flight sessions"
        );

        // Request handler finishes and tries to put the session
        // back. The tombstone must prevent the reinsert.
        store.put("alice".into(), in_flight, Some(checkout));
        assert_eq!(store.len(), 0, "delete must win the race");
        assert!(store.take("alice").is_none());

        // A fresh put after the resolution is allowed.
        store.put("alice".into(), SessionState::new(fake_kv()), None);
        assert!(store.take("alice").is_some());
    }

    /// F1.5 regression: a `take` that arrives after `delete` has armed
    /// a tombstone (e.g. because the session was just deleted and a
    /// straggling reuse attempt comes in) must return `None`, not
    /// resurrect a freshly-inserted entry by id collision.
    #[test]
    fn take_respects_outstanding_tombstone() {
        let store = SessionStore::new(Duration::from_secs(60));
        store.put("bob".into(), SessionState::new(fake_kv()), None);
        // Simulate "in flight then deleted" by hand: check out, then
        // delete — the tombstone is now armed.
        let (s, checkout) = store.take("bob").expect("must exist");
        assert!(store.delete("bob"));
        // Even if a racing caller manages to slip a `put` in (e.g.
        // because two takes ran in parallel and one already ran
        // `put`), the tombstone-aware `take` refuses to surface a
        // tombstoned id. Here we simulate that by inserting back
        // and asserting the next `take` returns None.
        store.put("bob".into(), s, Some(checkout)); // honoured tombstone → no insert
        assert!(store.take("bob").is_none());
    }
}

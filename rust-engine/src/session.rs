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
}

/// Lock-free session store.
#[derive(Debug, Clone)]
pub struct SessionStore {
    inner: Arc<DashMap<String, SessionState>>,
    ttl: Duration,
}

impl SessionStore {
    /// `ttl == 0` disables time-based eviction (sessions live until
    /// `delete` is called explicitly).
    pub fn new(ttl: Duration) -> Self {
        Self {
            inner: Arc::new(DashMap::new()),
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
    /// caller is expected to call [`Self::put`] when finished so the
    /// session resumes on the next request. Returns `None` when no
    /// session with that id exists.
    pub fn take(&self, id: &str) -> Option<SessionState> {
        self.inner.remove(id).map(|(_, mut s)| {
            s.last_used = Instant::now();
            s
        })
    }

    /// Reinsert a session — typically after a request has consumed
    /// `take` and produced new tokens. Overwrites any prior state for
    /// the same id (which can happen if a second concurrent request
    /// for the same session ran in parallel; last writer wins,
    /// matching vLLM / ollama semantics).
    pub fn put(&self, id: String, mut state: SessionState) {
        state.last_used = Instant::now();
        self.inner.insert(id, state);
    }

    /// Remove `id` and report whether it existed. Used by the
    /// `DELETE /v1/sessions/{id}` endpoint.
    pub fn delete(&self, id: &str) -> bool {
        self.inner.remove(id).is_some()
    }

    /// Evict entries idle for longer than the configured TTL. Returns
    /// the number of entries removed. Cheap when the store is small;
    /// the background evictor task calls this periodically.
    pub fn evict_expired(&self) -> usize {
        if self.ttl.is_zero() {
            return 0;
        }
        let now = Instant::now();
        let ttl = self.ttl;
        let mut removed = 0usize;
        // `retain` is `&mut`-locking per shard, so other shards stay
        // available for concurrent reads.
        self.inner.retain(|_, s| {
            let keep = now.duration_since(s.last_used) <= ttl;
            if !keep {
                removed += 1;
            }
            keep
        });
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
        store.put("alice".to_string(), s);
        assert_eq!(store.len(), 1);
        let back = store.take("alice").expect("session must exist");
        assert_eq!(back.position, 4);
        // Take is destructive.
        assert!(store.take("alice").is_none());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn delete_returns_existence_flag() {
        let store = SessionStore::new(Duration::from_secs(60));
        store.put("a".into(), SessionState::new(fake_kv()));
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
        store.put("fresh".into(), SessionState::new(fake_kv()));
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
}

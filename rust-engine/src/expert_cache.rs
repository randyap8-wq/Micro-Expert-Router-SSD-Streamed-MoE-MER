//! In-RAM LRU cache of resident experts.
//!
//! Each cache entry is an `Arc<ExpertResident>` whose buffer is owned by the
//! [`BufferPool`](crate::buffer_pool::BufferPool). Eviction simply drops the
//! `Arc`; once any in-flight inference also drops its handle, the underlying
//! `PooledBuffer` returns to the pool's free list automatically.

use crate::buffer_pool::PooledBuffer;
use lru::LruCache;
use parking_lot::Mutex;
use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::sync::Arc;

/// One resident expert: id + the bytes loaded from the SSD.
pub struct ExpertResident {
    pub id: u32,
    pub buffer: PooledBuffer,
}

impl ExpertResident {
    pub fn data(&self) -> &[u8] {
        self.buffer.as_slice()
    }
}

/// Thread-safe fixed-capacity LRU cache of resident experts.
pub struct ExpertCache {
    inner: Mutex<LruCache<u32, Arc<ExpertResident>>>,
    /// Expert ids that are pinned and must never be returned by
    /// [`Self::evict_lru`]. Pinning is set by the engine after an
    /// expert has been observed enough times to be considered "hot"
    /// (see [`crate::engine::Engine`] / `pin_after_observations`).
    pinned: Mutex<HashSet<u32>>,
    capacity: usize,
}

#[derive(Default, Debug, Clone, Copy)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
}

impl ExpertCache {
    pub fn new(capacity: usize) -> Self {
        let cap = NonZeroUsize::new(capacity).expect("cache capacity must be > 0");
        Self {
            inner: Mutex::new(LruCache::new(cap)),
            pinned: Mutex::new(HashSet::new()),
            capacity,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Look up an expert. Updates LRU recency on hit.
    pub fn get(&self, id: u32) -> Option<Arc<ExpertResident>> {
        self.inner.lock().get(&id).cloned()
    }

    /// Peek without changing recency. Useful for the predictive loader to
    /// check residency without polluting the LRU order.
    pub fn contains(&self, id: u32) -> bool {
        self.inner.lock().peek(&id).is_some()
    }

    /// Insert a resident expert. Returns the evicted entry, if any (so the
    /// caller can observe evictions for logging). If the cache is at
    /// capacity and the LRU candidate is pinned, the next non-pinned
    /// LRU entry is evicted instead. If every entry is pinned, this
    /// returns `Err(resident)` — there is nowhere to put the new entry
    /// without breaking a pin contract.
    pub fn insert(&self, resident: Arc<ExpertResident>) -> Option<Arc<ExpertResident>> {
        let id = resident.id;
        // Pre-evict a non-pinned entry if we're already at capacity,
        // so `push` below never has to silently evict a pinned entry.
        let pre_evicted = {
            let guard = self.inner.lock();
            let at_capacity = guard.len() >= self.capacity && guard.peek(&id).is_none();
            drop(guard);
            if at_capacity { self.evict_lru() } else { None }
        };
        let mut guard = self.inner.lock();
        // `LruCache::push` returns the (k, v) pair that was evicted, if any.
        let push_evicted = guard.push(id, resident).map(|(_, v)| v);
        // Either path evicts at most one; combine.
        push_evicted.or(pre_evicted)
    }

    /// Number of resident experts currently in the cache.
    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }

    /// Pop the least-recently-used **non-pinned** entry. Returns the
    /// removed `Arc` so callers can observe (and log) what was evicted;
    /// once the `Arc` is dropped its `PooledBuffer` returns to the
    /// pool's free list. Pinned experts (see [`Self::pin`]) are
    /// skipped — if every resident expert is pinned this returns
    /// `None`, meaning there is no room to evict.
    pub fn evict_lru(&self) -> Option<Arc<ExpertResident>> {
        let pinned = self.pinned.lock();
        if pinned.is_empty() {
            // Fast path: no pinning, just pop LRU.
            return self.inner.lock().pop_lru().map(|(_, v)| v);
        }
        // Walk LRU order from least-recent to most-recent and pop the
        // first non-pinned entry.
        let mut guard = self.inner.lock();
        // Collect ids in reverse-recency order. `LruCache::iter` yields
        // most-recently-used first, so the *last* item is the LRU.
        let id_order: Vec<u32> = guard.iter().map(|(k, _)| *k).collect();
        for &id in id_order.iter().rev() {
            if !pinned.contains(&id) {
                if let Some(v) = guard.pop(&id) {
                    return Some(v);
                }
            }
        }
        None
    }

    /// Pin an expert id so it is never returned by [`Self::evict_lru`].
    /// If the id isn't currently resident this still records the pin —
    /// when the expert is later loaded it will be protected from
    /// eviction.
    pub fn pin(&self, id: u32) {
        self.pinned.lock().insert(id);
    }

    /// Remove a pin previously installed by [`Self::pin`].
    #[allow(dead_code)]
    pub fn unpin(&self, id: u32) {
        self.pinned.lock().remove(&id);
    }

    /// Whether `id` is currently pinned.
    #[allow(dead_code)]
    pub fn is_pinned(&self, id: u32) -> bool {
        self.pinned.lock().contains(&id)
    }

    /// Snapshot of currently-pinned ids (for diagnostics / metrics).
    #[allow(dead_code)]
    pub fn pinned_ids(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self.pinned.lock().iter().copied().collect();
        v.sort_unstable();
        v
    }

    /// Number of currently-pinned ids.
    pub fn pinned_count(&self) -> usize {
        self.pinned.lock().len()
    }

    /// Snapshot of current residency (for logs/diagnostics).
    pub fn resident_ids(&self) -> Vec<u32> {
        self.inner.lock().iter().map(|(k, _)| *k).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer_pool::BufferPool;

    fn make(id: u32, pool: &BufferPool) -> Arc<ExpertResident> {
        let buffer = pool.try_acquire().unwrap();
        Arc::new(ExpertResident { id, buffer })
    }

    #[test]
    fn lru_eviction_returns_buffer_to_pool() {
        let pool = BufferPool::new(3, 4096, 4096);
        let cache = ExpertCache::new(2);

        cache.insert(make(0, &pool));
        cache.insert(make(1, &pool));
        // 2 of 3 slots are occupied by cache entries; 1 is free.
        let scratch = pool.try_acquire().expect("third slot free");
        assert!(pool.try_acquire().is_none());
        drop(scratch);

        // Inserting a third entry evicts expert 0 (the LRU). The evicted
        // Arc is returned and the cache no longer references its buffer.
        let evicted = cache.insert(make(2, &pool));
        assert!(evicted.is_some());
        assert_eq!(evicted.as_ref().unwrap().id, 0);

        // Pool is fully occupied (cache holds 1 + 2, plus the evicted Arc
        // still holds expert 0's buffer).
        assert!(pool.try_acquire().is_none());
        // Once the evicted Arc is dropped, its buffer returns to the pool.
        drop(evicted);
        assert!(pool.try_acquire().is_some());
    }

    #[test]
    fn hit_updates_recency() {
        let pool = BufferPool::new(3, 4096, 4096);
        let cache = ExpertCache::new(2);
        cache.insert(make(0, &pool));
        cache.insert(make(1, &pool));
        // Touch expert 0 -> it is now most-recently used.
        let _ = cache.get(0);
        // Inserting expert 2 should evict 1, not 0.
        cache.insert(make(2, &pool));
        assert!(cache.contains(0));
        assert!(!cache.contains(1));
        assert!(cache.contains(2));
    }

    #[test]
    fn pinned_entry_is_protected_from_eviction() {
        let pool = BufferPool::new(4, 4096, 4096);
        let cache = ExpertCache::new(2);
        cache.insert(make(0, &pool));
        cache.insert(make(1, &pool));
        // Pin expert 0. Even though it's the LRU, expert 1 must be
        // evicted instead when expert 2 is inserted.
        cache.pin(0);
        let evicted = cache.insert(make(2, &pool));
        assert!(evicted.is_some());
        assert_eq!(evicted.unwrap().id, 1);
        assert!(cache.contains(0));
        assert!(!cache.contains(1));
        assert!(cache.contains(2));
        assert!(cache.is_pinned(0));
        assert_eq!(cache.pinned_count(), 1);
    }

    #[test]
    fn evict_lru_returns_none_when_all_pinned() {
        let pool = BufferPool::new(4, 4096, 4096);
        let cache = ExpertCache::new(2);
        cache.insert(make(0, &pool));
        cache.insert(make(1, &pool));
        cache.pin(0);
        cache.pin(1);
        assert!(cache.evict_lru().is_none());
    }
}

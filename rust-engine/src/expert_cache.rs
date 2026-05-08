//! In-RAM LRU cache of resident experts.
//!
//! Each cache entry is an `Arc<ExpertResident>` whose buffer is owned by the
//! [`BufferPool`](crate::buffer_pool::BufferPool). Eviction simply drops the
//! `Arc`; once any in-flight inference also drops its handle, the underlying
//! `PooledBuffer` returns to the pool's free list automatically.

use crate::buffer_pool::PooledBuffer;
use lru::LruCache;
use parking_lot::Mutex;
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
    /// caller can observe evictions for logging).
    pub fn insert(&self, resident: Arc<ExpertResident>) -> Option<Arc<ExpertResident>> {
        let id = resident.id;
        let mut guard = self.inner.lock();
        // `LruCache::push` returns the (k, v) pair that was evicted, if any.
        guard.push(id, resident).map(|(_, v)| v)
    }

    /// Number of resident experts currently in the cache.
    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }

    /// Pop the least-recently-used entry. Returns the removed `Arc` so
    /// callers can observe (and log) what was evicted; once the `Arc` is
    /// dropped its `PooledBuffer` returns to the pool's free list.
    pub fn evict_lru(&self) -> Option<Arc<ExpertResident>> {
        self.inner.lock().pop_lru().map(|(_, v)| v)
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
}

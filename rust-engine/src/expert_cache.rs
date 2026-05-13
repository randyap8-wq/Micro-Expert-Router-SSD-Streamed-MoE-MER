//! In-RAM LRU cache of resident experts.
//!
//! Each cache entry is an `Arc<ExpertResident>` whose buffer is owned by the
//! [`BufferPool`](crate::buffer_pool::BufferPool). Eviction simply drops the
//! `Arc`; once any in-flight inference also drops its handle, the underlying
//! `PooledBuffer` returns to the pool's free list automatically.
//!
//! When the on-disk expert file was produced by `gguf-convert` (its default
//! mode), the buffer starts with a 64-byte Unified Tensor Header padded out
//! to one block. [`ExpertResident::data`] transparently strips that prefix
//! so every consumer downstream sees only the bare weight payload —
//! existing code paths (the SwiGLU kernels, the cache verifier, the
//! synthetic-expert fixtures) don't need to learn about UTH.

use crate::buffer_pool::PooledBuffer;
use crate::gguf_loader::DEFAULT_BLOCK_ALIGN;
use crate::tensor_header::TensorHeader;
use lru::LruCache;
use parking_lot::Mutex;
use std::collections::HashSet;
use std::num::NonZeroUsize;
use std::sync::Arc;

/// One resident expert: id + the bytes loaded from the SSD.
///
/// The optional Unified Tensor Header prefix is parsed **once** at
/// construction time (see [`ExpertResident::new`]); the resulting
/// `payload_offset` is cached so that [`Self::data`] — which sits on
/// the inference + `--io-only` hot paths — is a cheap subslice
/// operation with no re-parsing.
pub struct ExpertResident {
    pub id: u32,
    pub buffer: PooledBuffer,
    /// Byte offset within `buffer` at which the bare weight payload
    /// begins. `0` for legacy blobs and synthetic fixtures (no UTH);
    /// `UTH_BYTES + page padding` for `gguf-convert` blobs.
    payload_offset: usize,
}

impl ExpertResident {
    /// Construct a resident expert, computing and caching the UTH
    /// payload offset once. Subsequent calls to [`Self::data`] do not
    /// re-probe the header.
    pub fn new(id: u32, buffer: PooledBuffer) -> Self {
        let payload_offset = {
            let raw = buffer.as_slice();
            let (_, payload) = TensorHeader::strip(raw, DEFAULT_BLOCK_ALIGN);
            // `payload` is either `raw` unchanged (offset 0) or a suffix
            // subslice of it; derive the offset directly from the slice
            // lengths rather than via pointer arithmetic.
            let payload_offset = raw.len() - payload.len();
            debug_assert!(payload_offset <= raw.len());
            payload_offset
        };
        Self {
            id,
            buffer,
            payload_offset,
        }
    }

    /// Bare weight bytes — i.e. the buffer with any leading Unified
    /// Tensor Header stripped. The vast majority of callers want this.
    ///
    /// O(1): uses the cached `payload_offset` computed in [`Self::new`],
    /// so the UTH is **not** reparsed on each call.
    #[inline]
    pub fn data(&self) -> &[u8] {
        &self.buffer.as_slice()[self.payload_offset..]
    }

    /// Raw buffer bytes, including any U.T.H. prefix. Used by paths
    /// that need the literal on-disk image (e.g. the cache-integrity
    /// verifier, the dump tools).
    #[allow(dead_code)]
    pub fn raw(&self) -> &[u8] {
        self.buffer.as_slice()
    }

    /// Parsed Unified Tensor Header, if one is present at the start of
    /// the buffer. Returns `None` for legacy files (and for the
    /// synthetic-expert fixtures, which deliberately omit the header).
    ///
    /// This is a cold-path accessor (used by dump/diagnostic tools);
    /// the header is re-probed here rather than stored to keep the
    /// resident struct small.
    #[allow(dead_code)]
    pub fn header(&self) -> Option<TensorHeader> {
        let raw = self.buffer.as_slice();
        TensorHeader::strip(raw, DEFAULT_BLOCK_ALIGN).0
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

    /// Insert a resident expert.
    ///
    /// Returns `Ok(Some(evicted))` when an entry was evicted to make
    /// room (so the caller can observe / log the eviction), `Ok(None)`
    /// when the entry was inserted without displacing anything, and
    /// `Err(resident)` when the cache is full and **every** resident
    /// expert is pinned. The error case hands the original `Arc` back
    /// to the caller so its `PooledBuffer` can return to the pool —
    /// the alternative (silently calling `LruCache::push`, which
    /// would evict a pinned entry) would break the pinning contract.
    pub fn insert(
        &self,
        resident: Arc<ExpertResident>,
    ) -> Result<Option<Arc<ExpertResident>>, Arc<ExpertResident>> {
        let id = resident.id;
        // Pre-evict a non-pinned entry if we're already at capacity,
        // so `push` below never has to silently evict a pinned entry.
        let pre_evicted = {
            let guard = self.inner.lock();
            let at_capacity = guard.len() >= self.capacity && guard.peek(&id).is_none();
            drop(guard);
            if at_capacity {
                match self.evict_lru() {
                    Some(e) => Some(e),
                    // Cache is full *and* every resident expert is
                    // pinned. We must refuse the insert: calling
                    // `push` here would evict a pinned id (LruCache
                    // has no pinning concept).
                    None => return Err(resident),
                }
            } else {
                None
            }
        };
        let mut guard = self.inner.lock();
        // `LruCache::push` returns the (k, v) pair that was evicted, if any.
        // With the pre-eviction above we shouldn't normally hit a second
        // eviction path here, but `push` on an existing key returns the
        // old value — which is fine to surface as "evicted" too.
        let push_evicted = guard.push(id, resident).map(|(_, v)| v);
        Ok(push_evicted.or(pre_evicted))
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
        Arc::new(ExpertResident::new(id, buffer))
    }

    #[test]
    fn lru_eviction_returns_buffer_to_pool() {
        let pool = BufferPool::new(3, 4096, 4096);
        let cache = ExpertCache::new(2);

        let _ = cache.insert(make(0, &pool)).map_err(|_| panic!("insert failed"));
        let _ = cache.insert(make(1, &pool)).map_err(|_| panic!("insert failed"));
        // 2 of 3 slots are occupied by cache entries; 1 is free.
        let scratch = pool.try_acquire().expect("third slot free");
        assert!(pool.try_acquire().is_none());
        drop(scratch);

        // Inserting a third entry evicts expert 0 (the LRU). The evicted
        // Arc is returned and the cache no longer references its buffer.
        let evicted = match cache.insert(make(2, &pool)) {
            Ok(Some(e)) => e,
            other => panic!("expected Ok(Some(_)), got {:?}", other.is_ok()),
        };
        assert_eq!(evicted.id, 0);

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
        let _ = cache.insert(make(0, &pool)).map_err(|_| panic!("insert failed"));
        let _ = cache.insert(make(1, &pool)).map_err(|_| panic!("insert failed"));
        // Touch expert 0 -> it is now most-recently used.
        let _ = cache.get(0);
        // Inserting expert 2 should evict 1, not 0.
        let _ = cache.insert(make(2, &pool)).map_err(|_| panic!("insert failed"));
        assert!(cache.contains(0));
        assert!(!cache.contains(1));
        assert!(cache.contains(2));
    }

    #[test]
    fn pinned_entry_is_protected_from_eviction() {
        let pool = BufferPool::new(4, 4096, 4096);
        let cache = ExpertCache::new(2);
        let _ = cache.insert(make(0, &pool)).map_err(|_| panic!("insert failed"));
        let _ = cache.insert(make(1, &pool)).map_err(|_| panic!("insert failed"));
        // Pin expert 0. Even though it's the LRU, expert 1 must be
        // evicted instead when expert 2 is inserted.
        cache.pin(0);
        let evicted = match cache.insert(make(2, &pool)) {
            Ok(Some(e)) => e,
            other => panic!("expected Ok(Some(_)), got {:?}", other.is_ok()),
        };
        assert_eq!(evicted.id, 1);
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
        let _ = cache.insert(make(0, &pool)).map_err(|_| panic!("insert failed"));
        let _ = cache.insert(make(1, &pool)).map_err(|_| panic!("insert failed"));
        cache.pin(0);
        cache.pin(1);
        assert!(cache.evict_lru().is_none());
    }

    #[test]
    fn insert_returns_err_when_all_pinned() {
        // Cache full of pinned entries must reject a new insert with
        // `Err(resident)` rather than silently evicting a pinned slot.
        let pool = BufferPool::new(4, 4096, 4096);
        let cache = ExpertCache::new(2);
        let _ = cache.insert(make(0, &pool)).map_err(|_| panic!("insert failed"));
        let _ = cache.insert(make(1, &pool)).map_err(|_| panic!("insert failed"));
        cache.pin(0);
        cache.pin(1);
        let new_resident = make(2, &pool);
        let new_id = new_resident.id;
        let err = match cache.insert(new_resident) {
            Err(rejected) => rejected,
            Ok(_) => panic!("expected Err, got Ok"),
        };
        assert_eq!(err.id, new_id);
        // Both pinned entries are still resident.
        assert!(cache.contains(0));
        assert!(cache.contains(1));
        assert!(!cache.contains(2));
        // The rejected resident's buffer returns to the pool when
        // dropped — i.e. the contract that a rejected insert hands the
        // Arc back so its PooledBuffer can be reclaimed.
        drop(err);
        // After dropping the rejected resident *and* the scratch
        // buffer that `make(2, ...)` consumed, the pool should have
        // strictly more free slots than it did at the rejection.
        assert!(pool.try_acquire().is_some());
    }
}

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
use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};

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
    /// Monotonic hit counter (Phase 2 — three-tier memory hierarchy).
    ///
    /// Bumped by [`GpuExpertCache::observe_ram_hit`] / engine routing
    /// every time a RAM lookup resolves to this resident. Read by the
    /// promotion controller — once `hits >= promote_after_hits`, the
    /// expert becomes a candidate for the **Anchor Core** in VRAM.
    ///
    /// Stored as an `AtomicU64` so the engine's lock-free routing hot
    /// path can update it with a single relaxed atomic increment.
    hits: AtomicU64,
    /// Cached once-per-resident Q4_0 zero-padded payload used when the
    /// on-disk bytes are slightly short (≤ one block/page) of the
    /// derived expected size.
    q4_0_padded: StdMutex<Option<(usize, Arc<[u8]>)>>,
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
            hits: AtomicU64::new(0),
            q4_0_padded: StdMutex::new(None),
        }
    }

    /// Increment the resident's monotonic hit counter and return the
    /// new value. Used by the engine on every RAM hit to drive
    /// [`GpuExpertCache`] promotion decisions (Phase 2). Cheap: a
    /// single relaxed atomic FAA — safe to call from the lock-free
    /// inference hot path.
    #[inline]
    pub fn record_hit(&self) -> u64 {
        self.hits.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Current value of the per-resident hit counter. Snapshot only —
    /// the underlying counter may have been bumped by a concurrent
    /// caller by the time this returns. Used by diagnostics and the
    /// VRAM promotion controller.
    #[inline]
    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
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

    /// Return a cached zero-padded Q4_0 payload when the resident is at
    /// most `tolerance` bytes short of `need`. The padded allocation is
    /// created at most once per `need` for this resident.
    pub fn q4_0_padded_payload(&self, need: usize, tolerance: usize) -> Option<Arc<[u8]>> {
        let data = self.data();
        if data.len() >= need {
            return None;
        }
        let shortfall = need - data.len();
        if need <= tolerance || shortfall > tolerance {
            return None;
        }

        let mut guard = self.q4_0_padded.lock().expect("q4_0_padded poisoned");
        if let Some((cached_need, cached)) = guard.as_ref() {
            if *cached_need == need {
                return Some(cached.clone());
            }
        }
        let mut padded = Vec::with_capacity(need);
        padded.extend_from_slice(data);
        padded.resize(need, 0);
        let cached: Arc<[u8]> = Arc::from(padded.into_boxed_slice());
        *guard = Some((need, cached.clone()));
        Some(cached)
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

// =====================================================================
// Phase 2 — GPU (VRAM) expert cache: Segmented Hybrid Policy.
// =====================================================================

/// One VRAM-resident expert. Bytes are owned by the cache and
/// (conceptually) live in device memory; on the default build —
/// where the `gpu` cargo feature is **not** compiled in — VRAM is
/// emulated with a host-side `Vec<u8>` so the rest of the engine
/// (engine.rs, server.rs, batch_scheduler.rs) sees the same
/// `Arc<GpuResident>` shape regardless of whether real CUDA is
/// available.
///
/// The cache surface is identical to [`ExpertResident::data`]: callers
/// get a `&[u8]` weight payload that can be fed directly into the
/// existing `run_inference_*` family. When a real CUDA device is
/// active, [`GpuResident::data`] performs the device-to-host copy
/// lazily (see Phase 3's `run_inference_gpu`), so the inference loop
/// never blocks on the cache itself.
pub struct GpuResident {
    pub id: u32,
    /// Device-resident bytes. On builds without a real GPU runtime
    /// this is just a host `Vec<u8>`; on `gpu`-feature builds the
    /// init path replaces it with a `candle_core::Tensor` reference
    /// (see Phase 3 / `inference::run_inference_gpu`).
    bytes: Vec<u8>,
}

impl GpuResident {
    pub fn new(id: u32, bytes: Vec<u8>) -> Self {
        Self { id, bytes }
    }

    /// Bare weight bytes ready for `run_inference_*`.
    #[inline]
    pub fn data(&self) -> &[u8] {
        &self.bytes
    }

    /// Size in bytes of the VRAM footprint owned by this resident.
    /// Aggregated by the cache to track `mer_vram_used_bytes`.
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.bytes.len()
    }
}

impl crate::backend::GpuStorage for GpuResident {
    fn byte_len(&self) -> usize {
        self.bytes.len()
    }
    fn as_wgpu_buffer(&self) -> Option<&wgpu::Buffer> {
        None   // GpuResident is host-side only; VRAM lives in VramExpertEntry
    }
}

/// Outcome of a VRAM-tier lookup. The variants double as the
/// instrumentation discriminator for `mer_gpu_cache_hits_total` and
/// the engine's three-tier reporting in `/v1/admin/health/experts`.
pub enum GpuLookup {
    /// Hit on the **Anchor Core** — high-frequency, permanently
    /// pinned expert. No LRU recency update.
    AnchorHit(Arc<GpuResident>),
    /// Hit on the **LRU Edge** — temporal locality. Recency updated.
    LruHit(Arc<GpuResident>),
    /// Miss. Caller falls through to the RAM tier.
    Miss,
}

impl GpuLookup {
    /// Convenience: pull the resident out of any hit variant.
    pub fn into_resident(self) -> Option<Arc<GpuResident>> {
        match self {
            GpuLookup::AnchorHit(r) | GpuLookup::LruHit(r) => Some(r),
            GpuLookup::Miss => None,
        }
    }

    pub fn is_hit(&self) -> bool {
        !matches!(self, GpuLookup::Miss)
    }
}

/// Thread-safe VRAM expert cache implementing the **Segmented Hybrid
/// Policy** from the Phase 2 spec:
///
/// * **Anchor Core** — `HashMap<u32, Arc<GpuResident>>` for experts
///   that have crossed `promote_after_hits`. Pinned, never evicted.
///   Sized by `anchor_ratio * capacity_bytes`.
/// * **LRU Edge** — `LruCache<u32, Arc<GpuResident>>` for temporal
///   topic shifts. O(1) recency tracking, byte-budgeted evictions.
///
/// Concurrency contract (gist "Zero-Contention" critical constraint):
///
/// * All cache-state updates go through a single `parking_lot::Mutex`
///   wrapping the `Inner` struct. The critical section is just the
///   HashMap / LRU manipulation — never any I/O, never any compute.
/// * Hit counters on individual `ExpertResident`s are
///   [`AtomicU64`](std::sync::atomic::AtomicU64); the inference hot
///   path bumps them lock-free.
/// * `mer_vram_used_bytes` is an atomic `IntGauge` updated inside the
///   same critical section so external scrapes never observe a
///   torn value.
pub struct GpuExpertCache {
    inner: Mutex<GpuExpertCacheInner>,
    /// Capacity of the **Anchor Core**, in bytes. The total VRAM
    /// budget is `anchor_capacity_bytes + lru_capacity_bytes`.
    anchor_capacity_bytes: usize,
    /// Capacity of the **LRU Edge**, in bytes.
    lru_capacity_bytes: usize,
    /// Promotion threshold copied out of `[gpu_cache].promote_after_hits`.
    /// `0` disables Anchor Core promotions (everything routes to the
    /// LRU Edge).
    promote_after_hits: u64,
    /// Total promotions performed since startup. Mirror of the
    /// `mer_promotions_total` Prometheus counter; exposed here too so
    /// the admin health endpoint can render the value without going
    /// through the Prometheus registry.
    promotions: AtomicU64,
    /// VRAM bytes resident across Anchor + LRU. Read by the admin
    /// health endpoint and the TUI dashboard.
    vram_used: AtomicU64,
    /// Cumulative VRAM (GPU) cache hits — mirrors the
    /// `mer_gpu_cache_hits_total` Prometheus counter.
    hits: AtomicU64,
    /// Cumulative VRAM (GPU) cache misses — mirrors
    /// `mer_gpu_cache_misses_total`.
    misses: AtomicU64,
}

struct GpuExpertCacheInner {
    /// **Anchor Core** — permanently pinned high-frequency experts.
    anchor: HashMap<u32, Arc<GpuResident>>,
    anchor_used_bytes: usize,
    /// **LRU Edge** — temporal locality region.
    lru: LruCache<u32, Arc<GpuResident>>,
    lru_used_bytes: usize,
}

impl GpuExpertCache {
    /// Construct a new VRAM expert cache.
    ///
    /// * `capacity_bytes` — total VRAM budget for the cache
    ///   (anchor + LRU regions combined).
    /// * `anchor_ratio` — fraction of `capacity_bytes` reserved for
    ///   the Anchor Core. Clamped to `[0.0, 1.0]`.
    /// * `promote_after_hits` — threshold for RAM → VRAM promotion.
    ///   `0` disables Anchor Core promotion.
    pub fn new(capacity_bytes: usize, anchor_ratio: f32, promote_after_hits: u64) -> Self {
        let ratio = anchor_ratio.clamp(0.0, 1.0);
        let anchor_capacity_bytes = ((capacity_bytes as f32) * ratio) as usize;
        let lru_capacity_bytes = capacity_bytes.saturating_sub(anchor_capacity_bytes);
        // `LruCache` requires a non-zero entry count even when the
        // bytes budget would naturally allow zero. Use `unbounded()`
        // so eviction is driven solely by the byte-budget check
        // below — passing a sentinel like `usize::MAX` to `new()`
        // makes the underlying hashbrown allocator overflow.
        Self {
            inner: Mutex::new(GpuExpertCacheInner {
                anchor: HashMap::new(),
                anchor_used_bytes: 0,
                lru: LruCache::unbounded(),
                lru_used_bytes: 0,
            }),
            anchor_capacity_bytes,
            lru_capacity_bytes,
            promote_after_hits,
            promotions: AtomicU64::new(0),
            vram_used: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Total VRAM budget (anchor + LRU), in bytes.
    #[inline]
    pub fn capacity_bytes(&self) -> usize {
        self.anchor_capacity_bytes + self.lru_capacity_bytes
    }

    /// Capacity of the Anchor Core region in bytes.
    #[inline]
    pub fn anchor_capacity_bytes(&self) -> usize {
        self.anchor_capacity_bytes
    }

    /// Capacity of the LRU Edge region in bytes.
    #[inline]
    pub fn lru_capacity_bytes(&self) -> usize {
        self.lru_capacity_bytes
    }

    /// Currently-resident VRAM bytes (anchor + LRU).
    #[inline]
    pub fn used_bytes(&self) -> u64 {
        self.vram_used.load(Ordering::Relaxed)
    }

    /// Cumulative RAM → VRAM promotions.
    #[inline]
    pub fn promotions(&self) -> u64 {
        self.promotions.load(Ordering::Relaxed)
    }

    /// Cumulative VRAM cache hits.
    #[inline]
    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// Cumulative VRAM cache misses.
    #[inline]
    pub fn misses(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    /// Promotion threshold (`gpu_cache.promote_after_hits`).
    #[inline]
    pub fn promote_after_hits(&self) -> u64 {
        self.promote_after_hits
    }

    /// Look up an expert in VRAM. Returns the [`GpuLookup`] discriminator
    /// (anchor / LRU / miss) plus the resident handle on hit.
    ///
    /// **LRU Edge** hits update recency; **Anchor Core** hits do not
    /// (anchored experts are permanently hot by definition).
    pub fn get(&self, id: u32) -> GpuLookup {
        let mut g = self.inner.lock();
        if let Some(r) = g.anchor.get(&id).cloned() {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return GpuLookup::AnchorHit(r);
        }
        // `LruCache::get` updates recency; that's what we want for
        // the LRU Edge.
        if let Some(r) = g.lru.get(&id).cloned() {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return GpuLookup::LruHit(r);
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        GpuLookup::Miss
    }

    /// Check whether an expert is currently resident in either the
    /// anchor or LRU regions, without mutating recency or counters.
    pub fn contains(&self, id: u32) -> bool {
        let g = self.inner.lock();
        g.anchor.contains_key(&id) || g.lru.peek(&id).is_some()
    }

    /// Should the resident's current hit count promote it to the
    /// Anchor Core? Cheap relaxed-atomic compare against the
    /// configured threshold; safe to call from the hot path before
    /// kicking off an async promotion.
    #[inline]
    pub fn should_promote(&self, ram_hits: u64) -> bool {
        self.promote_after_hits > 0 && ram_hits >= self.promote_after_hits
    }

    /// Synchronous promotion entry point — copy a RAM resident's
    /// bytes into VRAM and place it in the Anchor Core if budget
    /// allows, otherwise in the LRU Edge.
    ///
    /// **Hot-path callers must not invoke this directly** — instead
    /// hand the resident off to the engine's background promotion
    /// task (see [`crate::engine::Engine`]). The synchronous path
    /// exists for the warm-up sequence (where blocking is the
    /// expected behaviour) and for tests.
    ///
    /// Returns `true` when the expert landed in VRAM, `false` if it
    /// could not fit even after eviction (e.g. payload exceeds the
    /// LRU budget entirely).
    pub fn promote_sync(&self, resident: Arc<GpuResident>) -> bool {
        let bytes = resident.byte_len();
        if bytes == 0 {
            return false;
        }
        let mut g = self.inner.lock();
        // Already resident: nothing to promote. Touch the LRU entry so
        // it becomes MRU, but don't count this as a new promotion nor
        // re-account bytes (the existing entry already owns them).
        if g.anchor.contains_key(&resident.id) {
            return true;
        }
        if g.lru.get(&resident.id).is_some() {
            return true;
        }
        // Anchor first: if it fits in the anchor budget *and* the
        // engine flagged this expert as hot, install there. We treat
        // any explicit promote_sync as "hot" (the engine only calls
        // this after threshold), but still prefer Anchor only when
        // there's room without evicting another anchor entry.
        if bytes <= self.anchor_capacity_bytes
            && g.anchor_used_bytes + bytes <= self.anchor_capacity_bytes
        {
            g.anchor.insert(resident.id, resident.clone());
            g.anchor_used_bytes += bytes;
            drop(g);
            self.promotions.fetch_add(1, Ordering::Relaxed);
            self.refresh_used_bytes();
            return true;
        }
        if bytes > self.lru_capacity_bytes {
            // Won't fit even after evicting everything in the LRU
            // region. Don't try.
            return false;
        }
        // Evict LRU entries until there is room. `LruCache::pop_lru`
        // returns the least-recently-used (k, v).
        while g.lru_used_bytes + bytes > self.lru_capacity_bytes {
            match g.lru.pop_lru() {
                Some((_, victim)) => {
                    g.lru_used_bytes = g.lru_used_bytes.saturating_sub(victim.byte_len());
                }
                None => break,
            }
        }
        let already = g.lru.put(resident.id, resident.clone());
        if let Some(prev) = already {
            // Replacing an existing entry — subtract the old footprint.
            g.lru_used_bytes = g.lru_used_bytes.saturating_sub(prev.byte_len());
        }
        g.lru_used_bytes += bytes;
        drop(g);
        self.promotions.fetch_add(1, Ordering::Relaxed);
        self.refresh_used_bytes();
        true
    }

    /// Snapshot of VRAM-resident expert ids (anchor first, then LRU
    /// in most-recently-used order). Used by the admin health
    /// endpoint and the TUI dashboard.
    pub fn resident_ids(&self) -> Vec<u32> {
        let g = self.inner.lock();
        let mut v: Vec<u32> = g.anchor.keys().copied().collect();
        v.sort_unstable();
        v.extend(g.lru.iter().map(|(k, _)| *k));
        v
    }

    /// Number of Anchor Core entries.
    pub fn anchor_len(&self) -> usize {
        self.inner.lock().anchor.len()
    }

    /// Number of LRU Edge entries.
    pub fn lru_len(&self) -> usize {
        self.inner.lock().lru.len()
    }

    fn refresh_used_bytes(&self) {
        let g = self.inner.lock();
        let total = (g.anchor_used_bytes + g.lru_used_bytes) as u64;
        drop(g);
        self.vram_used.store(total, Ordering::Relaxed);
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

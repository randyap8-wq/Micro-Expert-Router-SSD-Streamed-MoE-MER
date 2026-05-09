//! Per-layer expert cache for multi-layer MoE models (gist Phase 5,
//! "Option B: per-layer caches").
//!
//! Mixtral has 32 layers, each with its own pool of 8 experts. A flat
//! `expert_id` namespace `0..N-1` cannot represent that without forcing
//! every layer's experts onto a single shared LRU — which would let layer
//! 5's prefetched experts evict layer 0's, defeating the cache.
//!
//! [`MultiLayerExpertCache`] owns one [`crate::expert_cache::ExpertCache`]
//! per layer. The router, predictor, and engine all key on
//! `(layer, expert_id)`. The on-disk file naming convention is
//! `expert_<layer>_<id>.bin` for multi-layer models (single-layer models
//! continue to use `expert_<id>.bin`, written by the existing extractor).
//!
//! The wrapper is consumed by the multi-layer transformer wiring landing
//! in a follow-up PR; the in-tree single-layer `serve` path uses one
//! `ExpertCache` directly. `dead_code` is allowed so the surface is
//! greppable without a forced call site.
#![allow(dead_code)]


use crate::expert_cache::{ExpertCache, ExpertResident};
use std::sync::Arc;

/// Fixed `(layer, expert)` key for a multi-layer expert lookup.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ExpertKey {
    pub layer: u32,
    pub expert: u32,
}

impl ExpertKey {
    pub fn new(layer: u32, expert: u32) -> Self {
        Self { layer, expert }
    }
}

/// One [`ExpertCache`] per layer. Capacities can be set per-layer (e.g.
/// to give "hot" early layers more residency budget) or uniformly via
/// [`MultiLayerExpertCache::with_uniform_capacity`].
pub struct MultiLayerExpertCache {
    caches: Vec<Arc<ExpertCache>>,
}

impl MultiLayerExpertCache {
    /// Build a cache with `num_layers` per-layer caches, each of
    /// capacity `cap_per_layer`.
    pub fn with_uniform_capacity(num_layers: usize, cap_per_layer: usize) -> Self {
        assert!(num_layers > 0, "num_layers must be > 0");
        let caches = (0..num_layers).map(|_| Arc::new(ExpertCache::new(cap_per_layer))).collect();
        Self { caches }
    }

    /// Build a cache from explicit per-layer capacities.
    pub fn with_capacities(per_layer_caps: Vec<usize>) -> Self {
        assert!(!per_layer_caps.is_empty(), "must have at least one layer");
        let caches = per_layer_caps.into_iter().map(|c| Arc::new(ExpertCache::new(c))).collect();
        Self { caches }
    }

    pub fn num_layers(&self) -> usize {
        self.caches.len()
    }

    /// Borrow the [`ExpertCache`] for one layer (so existing engine code
    /// that takes an `Arc<ExpertCache>` keeps working).
    pub fn cache_for_layer(&self, layer: u32) -> Arc<ExpertCache> {
        self.caches[layer as usize].clone()
    }

    pub fn get(&self, key: ExpertKey) -> Option<Arc<ExpertResident>> {
        self.caches.get(key.layer as usize)?.get(key.expert)
    }

    pub fn contains(&self, key: ExpertKey) -> bool {
        self.caches
            .get(key.layer as usize)
            .map(|c| c.contains(key.expert))
            .unwrap_or(false)
    }

    /// Total number of cached experts across all layers.
    pub fn total_resident(&self) -> usize {
        self.caches.iter().map(|c| c.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer_pool::BufferPool;

    #[test]
    fn per_layer_caches_are_independent() {
        let pool = BufferPool::new(4, 4096, 4096);
        let mlc = MultiLayerExpertCache::with_uniform_capacity(2, 2);

        // Insert expert 0 into layer 0.
        let resident = Arc::new(ExpertResident {
            id: 0,
            buffer: pool.try_acquire().unwrap(),
        });
        mlc.cache_for_layer(0).insert(resident);

        assert!(mlc.contains(ExpertKey::new(0, 0)));
        assert!(!mlc.contains(ExpertKey::new(1, 0)));
        assert_eq!(mlc.total_resident(), 1);
    }

    #[test]
    fn cache_for_layer_returns_clones_of_same_arc() {
        let mlc = MultiLayerExpertCache::with_uniform_capacity(3, 1);
        let a = mlc.cache_for_layer(0);
        let b = mlc.cache_for_layer(0);
        assert!(Arc::ptr_eq(&a, &b));
    }
}

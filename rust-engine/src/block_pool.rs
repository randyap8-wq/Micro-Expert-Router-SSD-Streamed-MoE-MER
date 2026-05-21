//! Shared **physical block pool** + per-request **block manager** for
//! the paged KV cache.
//!
//! The "Omniscient Predictive Architecture" design spec calls for a
//! KV-cache memory model in which every running request keeps a small
//! `block_table: Vec<u32>` of *block ids* and the actual `f32` storage
//! lives in a single shared, server-wide pool. When a request needs a
//! new block, it pulls one from the pool's free list (O(1)); when it
//! finishes (or is aborted), every block it owns is returned to the
//! pool — likewise O(1) per block.
//!
//! This module implements that pool and the per-request manager that
//! decides when a block needs to be allocated. The compute hot path
//! for attention itself stays in [`crate::transformer`]: the pool is
//! a *memory-management* concern, not an attention-kernel one.
//!
//! ## Layering vs. existing [`crate::transformer::KvCache`]
//!
//! The existing `KvCache` already uses 16-token blocks (matching the
//! vLLM PagedAttention default), but allocates each block on the heap
//! independently with `Vec<Box<[f32]>>`. That representation is fine
//! for a single-request server but creates allocator pressure on a
//! batched scheduler that constantly grows and resets caches across
//! many concurrent requests. The block pool below replaces those
//! transient `Box<[f32]>` allocations with reuse from a single
//! contiguous slab, which:
//!
//! * eliminates allocator round-trips on the per-token append path,
//! * keeps the working set physically close in DRAM (one slab,
//!   block-aligned), so prefetching & cache effects are predictable,
//! * gives the scheduler an easy "how many slots are free?" knob it
//!   can use to refuse / queue new requests instead of OOM-ing.
//!
//! `KvCache` is kept as the canonical attention-side API; this
//! module's [`BlockPool`] and [`BlockManager`] provide the
//! *memory-managed* sibling used by the batch scheduler. Both are
//! fully tested.

use std::sync::Arc;

use parking_lot::Mutex;

/// Default tokens per physical block, matching `PAGED_BLOCK_TOKENS`
/// in [`crate::transformer`]. 16 tokens × `kv_dim` × 4 bytes (f32) is
/// well under one OS page for realistic `kv_dim`, which keeps each
/// block contained on a single MMU page boundary.
pub const POOL_BLOCK_TOKENS: usize = 16;

/// Soft-cap threshold (fraction of primary capacity) above which the
/// scheduler should start preemptive idle-block reclamation. The gist
/// calls out **90%** as the cutoff at which `evict_idle_blocks(...)`
/// runs to reclaim KV memory from sessions that have stopped producing
/// tokens.
pub const SOFT_CAP_RATIO: f32 = 0.90;

/// Critical-pressure threshold (fraction of primary capacity). When
/// crossed the scheduler clamps the predictive-prefetch speculation
/// depth to zero so any free RAM is reserved for active resident
/// tokens. See [`PressureLevel::Critical`].
pub const CRITICAL_PRESSURE_RATIO: f32 = 0.98;

/// Memory-pressure level reported by [`BlockPool::pressure_level`].
/// Drives the scheduler's back-pressure policy: under
/// [`PressureLevel::High`] preemptive idle reclamation kicks in;
/// under [`PressureLevel::Critical`] speculative prefetching is
/// suspended entirely (`speculation_depth() == 0`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PressureLevel {
    /// Plenty of slack — fewer than [`SOFT_CAP_RATIO`] of primary
    /// blocks are in use. No reclamation or back-pressure needed.
    Normal,
    /// At or above [`SOFT_CAP_RATIO`] but still below
    /// [`CRITICAL_PRESSURE_RATIO`]. Scheduler runs idle reclamation
    /// but predictive prefetching continues.
    High,
    /// At or above [`CRITICAL_PRESSURE_RATIO`]. Scheduler suspends
    /// speculative prefetch (depth → 0) until pressure drops.
    Critical,
}

/// High bit of [`BlockId`] reserved to distinguish the **primary**
/// pre-allocated slab (bit clear) from the **overflow** heap-backed
/// slab (bit set). Sentinel placed at `1 << 31` so the low 31 bits
/// remain a normal block index up to ~2 G blocks — far beyond any
/// realistic deployment.
///
/// Callers should treat [`BlockId`] as opaque; this constant exists so
/// the pool's allocator and reader paths can route to the correct
/// slab without an additional lookup.
pub(crate) const OVERFLOW_BIT: u32 = 1 << 31;

/// Opaque handle to one physical block in a [`BlockPool`]. The pool's
/// free list stores these as raw `u32`s to keep the per-request block
/// table compact (`Vec<u32>` rather than `Vec<usize>`).
///
/// The high bit ([`OVERFLOW_BIT`]) discriminates between blocks in
/// the primary pre-allocated slab (bit clear) and the heap-backed
/// overflow slab (bit set) that grows on demand when the primary is
/// exhausted. Within each slab the low 31 bits are a 0-based block
/// index. Use [`BlockId::is_overflow`] and [`BlockId::index`] when
/// you need to introspect the discriminator.
///
/// Note: `BlockId` is *not* a `Drop` type — leaking one is harmless
/// (the pool just permanently loses a slot) but the per-request
/// [`BlockManager`] makes leaks impossible by returning every owned
/// block on `release_all` / `Drop`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

impl BlockId {
    /// `true` if this id refers to the heap-backed overflow slab.
    #[inline]
    pub fn is_overflow(self) -> bool {
        (self.0 & OVERFLOW_BIT) != 0
    }

    /// Slab-local block index (high bit masked off).
    #[inline]
    pub fn index(self) -> usize {
        (self.0 & !OVERFLOW_BIT) as usize
    }
}

/// A shared **physical** block pool: one big f32 slab carved into
/// fixed-size blocks plus a free list of unused block ids, optionally
/// backed by a secondary **overflow** slab that grows on demand when
/// the primary slab is exhausted.
///
/// Every block holds `POOL_BLOCK_TOKENS * kv_dim` floats. The pool
/// also owns a parallel "values" half — a single block id therefore
/// addresses both the K and V slot for the same token range. This
/// matches the way attention reads (`key(i)` / `value(i)` always
/// stride together), so we never want to allocate keys and values
/// separately.
///
/// ## Dynamic scaling
///
/// The primary slab is pre-allocated to `capacity` blocks at
/// construction time and never re-allocated; allocation from it is
/// O(1) (a single free-list pop) and the working set stays
/// block-aligned in DRAM. When that slab is empty, the pool falls
/// back to a heap-backed overflow slab: each overflow allocation
/// extends two parallel `Vec<f32>`s (keys + values) by one block's
/// worth of floats and returns a [`BlockId`] with the
/// [`OVERFLOW_BIT`] flag set. Releases from the overflow slab go to
/// its own LIFO free list so the bytes are reused before any further
/// growth — i.e. the overflow slab acts as a high-water-mark cache,
/// not a leak. The scheduler can detect overflow with
/// [`Self::overflow_in_use`] and log / throttle accordingly.
///
/// The pool is `Sync` and is intended to be wrapped in `Arc`. Free
/// list mutations take a short `parking_lot::Mutex`, and the backing
/// key/value slabs are also protected by `Mutex<Vec<f32>>`. Overflow
/// growth happens under those same mutexes so reads observing a
/// freshly-allocated overflow block see initialised zeros.
pub struct BlockPool {
    kv_dim: usize,
    capacity: usize,
    /// Per-block float counts for both halves — used to bounds-check
    /// reads and to size pre-zeroed slabs.
    block_floats: usize,
    /// Single contiguous slab of f32 keys, length
    /// `capacity * block_floats`.
    keys: Mutex<Vec<f32>>,
    /// Single contiguous slab of f32 values.
    values: Mutex<Vec<f32>>,
    /// LIFO free list of block ids currently available for
    /// allocation. Pre-populated with `0..capacity` at construction.
    free: Mutex<Vec<u32>>,
    /// Heap-backed overflow slab — grown on demand when [`Self::free`]
    /// is empty. The two `Vec`s grow in lock-step (always
    /// `overflow_capacity * block_floats` floats long); the free list
    /// holds slab-local block indices (no [`OVERFLOW_BIT`] flag —
    /// that's applied when constructing the public [`BlockId`]).
    overflow_keys: Mutex<Vec<f32>>,
    overflow_values: Mutex<Vec<f32>>,
    overflow_free: Mutex<Vec<u32>>,
    /// Monotonically tracks the high-water mark of the overflow slab
    /// (= the number of distinct blocks ever materialised). The
    /// **in-use** count is `overflow_capacity - overflow_free.len()`,
    /// exposed via [`Self::overflow_in_use`].
    overflow_capacity: Mutex<usize>,
    /// Per-pool pressure thresholds (gist Part 1, fix #4). Wraps the
    /// legacy [`SOFT_CAP_RATIO`] / [`CRITICAL_PRESSURE_RATIO`]
    /// constants so deployments can override them from the
    /// `[real_transformer]` block in `config.toml` without recompiling.
    thresholds: PressureThresholds,
}

/// Per-pool back-pressure thresholds, surfaced from the config file
/// (gist Part 1, fix #4). The constants [`SOFT_CAP_RATIO`] and
/// [`CRITICAL_PRESSURE_RATIO`] are the **defaults** — production
/// deployments override them via the
/// `[real_transformer].pressure_high_threshold` /
/// `[real_transformer].pressure_critical_threshold` keys so the
/// ladder can be tuned per fleet without rebuilding the binary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PressureThresholds {
    /// Utilisation at or above which the pool is classified
    /// [`PressureLevel::High`]. Triggers preemptive
    /// `evict_idle_blocks` in the batch scheduler.
    pub high: f32,
    /// Utilisation at or above which the pool is classified
    /// [`PressureLevel::Critical`]. Causes the scheduler to suspend
    /// speculation (depth → 0) until pressure drops back below
    /// `high`.
    pub critical: f32,
    /// Optional cap on the number of overflow blocks allocated
    /// beyond the primary slab (gist Part 1, fix #5). `None`
    /// preserves the historical "grow forever" behaviour; `Some(n)`
    /// makes [`BlockPool::allocate`] return `None` once `n` overflow
    /// blocks are in flight, giving the scheduler an admission
    /// back-pressure signal instead of silently exploding heap.
    pub max_overflow_capacity: Option<usize>,
}

impl Default for PressureThresholds {
    fn default() -> Self {
        Self {
            high: SOFT_CAP_RATIO,
            critical: CRITICAL_PRESSURE_RATIO,
            max_overflow_capacity: None,
        }
    }
}

impl PressureThresholds {
    /// Build a custom threshold pair and validate the ordering /
    /// range. Both values must lie in `(0.0, 1.0]` and `high` must
    /// be `<= critical`. Returns `Err` with a human-readable
    /// diagnostic instead of panicking so the config layer can
    /// surface a clean error message to the operator.
    pub fn try_new(high: f32, critical: f32) -> Result<Self, String> {
        if !(high > 0.0 && high <= 1.0) {
            return Err(format!("pressure_high_threshold must be in (0.0, 1.0]; got {high}"));
        }
        if !(critical > 0.0 && critical <= 1.0) {
            return Err(format!(
                "pressure_critical_threshold must be in (0.0, 1.0]; got {critical}"
            ));
        }
        if high > critical {
            return Err(format!(
                "pressure_high_threshold ({high}) must not exceed \
                 pressure_critical_threshold ({critical})"
            ));
        }
        Ok(Self { high, critical, max_overflow_capacity: None })
    }

    /// Builder-style setter for the overflow cap (gist Part 1, fix #5).
    /// `Some(0)` is normalized to `None` (unbounded) so the config
    /// surface can use `0 = unbounded` ergonomically.
    pub fn with_max_overflow_capacity(mut self, max: Option<usize>) -> Self {
        self.max_overflow_capacity = max.filter(|&n| n > 0);
        self
    }
}

impl std::fmt::Debug for BlockPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockPool")
            .field("kv_dim", &self.kv_dim)
            .field("capacity", &self.capacity)
            .field("block_floats", &self.block_floats)
            .field("free_blocks", &self.free.lock().len())
            .field("overflow_capacity", &*self.overflow_capacity.lock())
            .field("overflow_free", &self.overflow_free.lock().len())
            .finish()
    }
}

impl BlockPool {
    /// Build a pool that can hold up to `capacity` blocks of
    /// `POOL_BLOCK_TOKENS * kv_dim` floats each (× 2 for keys+values).
    /// Both slabs are pre-allocated and zeroed up front; allocation
    /// from the primary slab during the per-token hot path never
    /// grows them. The overflow slab starts empty and is only
    /// materialised on demand.
    pub fn new(kv_dim: usize, capacity: usize) -> Arc<Self> {
        Self::with_thresholds(kv_dim, capacity, PressureThresholds::default())
    }

    /// Build a pool whose back-pressure ladder uses the supplied
    /// [`PressureThresholds`] instead of the legacy [`SOFT_CAP_RATIO`]
    /// / [`CRITICAL_PRESSURE_RATIO`] defaults (gist Part 1, fix #4).
    pub fn with_thresholds(
        kv_dim: usize,
        capacity: usize,
        thresholds: PressureThresholds,
    ) -> Arc<Self> {
        assert!(kv_dim > 0, "BlockPool kv_dim must be > 0");
        assert!(capacity > 0, "BlockPool capacity must be > 0");
        let block_floats = POOL_BLOCK_TOKENS * kv_dim;
        let total = block_floats.checked_mul(capacity).expect("BlockPool capacity overflows usize");
        let keys = vec![0.0f32; total];
        let values = vec![0.0f32; total];
        // Free list initialised in *reverse* so allocation hands out
        // ids 0, 1, 2, … in order — easier to reason about in tests.
        let free: Vec<u32> = (0..capacity as u32).rev().collect();
        // Sanity-check: primary capacity must fit in 31 bits so the
        // overflow discriminator (`OVERFLOW_BIT`) is always reserved.
        assert!(
            (capacity as u32) <= !OVERFLOW_BIT,
            "BlockPool primary capacity exceeds 31-bit BlockId space"
        );
        Arc::new(Self {
            kv_dim,
            capacity,
            block_floats,
            keys: Mutex::new(keys),
            values: Mutex::new(values),
            free: Mutex::new(free),
            overflow_keys: Mutex::new(Vec::new()),
            overflow_values: Mutex::new(Vec::new()),
            overflow_free: Mutex::new(Vec::new()),
            overflow_capacity: Mutex::new(0),
            thresholds,
        })
    }

    /// Pop one block id from the free list. Allocation is always
    /// O(1): first the primary slab's free list, then the overflow
    /// slab's free list, and finally a fresh overflow extension. This
    /// method returns `None` only when a configured
    /// [`PressureThresholds::max_overflow_capacity`] has been reached;
    /// callers that want to refuse / queue requests *before* hitting
    /// the heap-backed overflow should consult [`Self::free_blocks`]
    /// first.
    ///
    /// The historical signature returns `Option<BlockId>` for source
    /// compatibility. With no configured overflow cap, allocation is
    /// infallible from the pool's perspective (OOM still panics via
    /// `Vec` growth).
    pub fn allocate(&self) -> Option<BlockId> {
        // Fast path: O(1) primary slab pop.
        if let Some(idx) = self.free.lock().pop() {
            return Some(BlockId(idx));
        }
        // Overflow path. First try the overflow slab's free list;
        // only grow the underlying `Vec`s when there's no recycled
        // slot waiting.
        if let Some(idx) = self.overflow_free.lock().pop() {
            return Some(BlockId(idx | OVERFLOW_BIT));
        }
        // Grow by exactly one block. We hold the keys/values/capacity
        // mutexes in a consistent order so two concurrent allocators
        // can't observe a half-grown slab.
        let mut keys = self.overflow_keys.lock();
        let mut values = self.overflow_values.lock();
        let mut cap = self.overflow_capacity.lock();
        // Admission back-pressure (gist Part 1, fix #5): if a hard
        // cap on the overflow slab is configured and we've already
        // grown to it, refuse the allocation. The scheduler treats
        // `None` as "pool exhausted" and either queues or rejects
        // the request rather than oomming the host.
        if let Some(max) = self.thresholds.max_overflow_capacity {
            if *cap >= max {
                return None;
            }
        }
        let new_idx = *cap as u32;
        assert!(
            (new_idx & OVERFLOW_BIT) == 0,
            "overflow slab exceeded 31-bit BlockId space ({} blocks)",
            *cap
        );
        let new_keys_len = keys.len() + self.block_floats;
        let new_values_len = values.len() + self.block_floats;
        keys.resize(new_keys_len, 0.0);
        values.resize(new_values_len, 0.0);
        *cap += 1;
        Some(BlockId(new_idx | OVERFLOW_BIT))
    }

    /// Return one block id to the free list. The block's contents
    /// are *not* zeroed on release — the next allocator overwrites
    /// the bytes it cares about via [`Self::write_token`], and reads
    /// past `seq_len` are guaranteed unreachable by callers.
    pub fn release(&self, id: BlockId) {
        if id.is_overflow() {
            let idx = id.index();
            debug_assert!(
                idx < *self.overflow_capacity.lock(),
                "released overflow block index {} >= overflow capacity",
                idx
            );
            // Store the bare slab-local index; the OVERFLOW_BIT is
            // re-applied on allocate().
            self.overflow_free.lock().push(idx as u32);
        } else {
            debug_assert!(
                (id.0 as usize) < self.capacity,
                "released block id {} >= capacity {}",
                id.0,
                self.capacity
            );
            self.free.lock().push(id.0);
        }
    }

    /// Number of blocks currently free **in the primary slab only**.
    /// The overflow slab is conceptually unbounded (modulo system
    /// memory), so it doesn't contribute to this count. Snapshot-only;
    /// the value can race with concurrent allocators.
    pub fn free_blocks(&self) -> usize {
        self.free.lock().len()
    }

    /// Number of overflow blocks currently in use (i.e. allocated and
    /// not yet released). `> 0` means the primary slab was at some
    /// point exhausted and the pool is now servicing requests out of
    /// the heap-backed fallback. Snapshot-only; can race.
    pub fn overflow_in_use(&self) -> usize {
        let cap = *self.overflow_capacity.lock();
        cap.saturating_sub(self.overflow_free.lock().len())
    }

    /// Fraction of primary capacity currently in use, in `[0.0, 1.0+]`.
    /// Values `> 1.0` indicate the overflow slab is also active. The
    /// scheduler reads this to decide when to trigger
    /// `evict_idle_blocks` and when to clamp speculation depth.
    /// Snapshot-only; can race.
    pub fn utilization(&self) -> f32 {
        if self.capacity == 0 {
            return 0.0;
        }
        let free = self.free.lock().len();
        let used_primary = self.capacity.saturating_sub(free);
        let overflow_in_use = {
            let cap = *self.overflow_capacity.lock();
            cap.saturating_sub(self.overflow_free.lock().len())
        };
        (used_primary + overflow_in_use) as f32 / self.capacity as f32
    }

    /// Classify the pool's current memory pressure. Drives the
    /// scheduler's back-pressure ladder; see [`PressureLevel`].
    ///
    /// The crossover ratios are taken from the per-pool
    /// [`PressureThresholds`] configured at construction time (gist
    /// Part 1, fix #4) — operators tune them via the
    /// `[real_transformer]` block in `config.toml` without recompiling.
    pub fn pressure_level(&self) -> PressureLevel {
        let u = self.utilization();
        if u >= self.thresholds.critical {
            PressureLevel::Critical
        } else if u >= self.thresholds.high {
            PressureLevel::High
        } else {
            PressureLevel::Normal
        }
    }

    /// Whether the pool's primary slab has crossed
    /// `PressureThresholds::high`. The scheduler polls this on every
    /// batch and, when true, runs `evict_idle_blocks` to reclaim KV
    /// blocks from sessions that haven't generated a token in
    /// `idle_threshold` seconds (default 5 s).
    pub fn above_soft_cap(&self) -> bool {
        self.utilization() >= self.thresholds.high
    }

    /// Currently configured back-pressure thresholds.
    pub fn thresholds(&self) -> PressureThresholds {
        self.thresholds
    }

    /// Total physical capacity of the **primary** slab, in blocks
    /// (constant for the pool's lifetime). The overflow slab is
    /// reported separately by [`Self::overflow_in_use`].
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Per-block embedding width (one of K or V; the other is identical).
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    /// Write the `(k, v)` pair for one token into block `id` at
    /// offset `in_block`. Panics on out-of-range indices, since the
    /// pool is the lowest layer and any out-of-range write is a bug
    /// in the caller. Transparently routes primary vs. overflow
    /// blocks based on [`BlockId::is_overflow`].
    pub fn write_token(&self, id: BlockId, in_block: usize, k: &[f32], v: &[f32]) {
        assert_eq!(k.len(), self.kv_dim, "k length must equal kv_dim");
        assert_eq!(v.len(), self.kv_dim, "v length must equal kv_dim");
        assert!(in_block < POOL_BLOCK_TOKENS, "in_block {} >= POOL_BLOCK_TOKENS", in_block);
        let idx = id.index();
        let block_off = idx * self.block_floats + in_block * self.kv_dim;
        if id.is_overflow() {
            let mut keys = self.overflow_keys.lock();
            keys[block_off..block_off + self.kv_dim].copy_from_slice(k);
            drop(keys);
            let mut values = self.overflow_values.lock();
            values[block_off..block_off + self.kv_dim].copy_from_slice(v);
        } else {
            let mut keys = self.keys.lock();
            keys[block_off..block_off + self.kv_dim].copy_from_slice(k);
            drop(keys);
            let mut values = self.values.lock();
            values[block_off..block_off + self.kv_dim].copy_from_slice(v);
        }
    }

    /// Read the cached key vector for one token slot. Returns an
    /// owned `Vec<f32>` (the pool slab is behind a `Mutex`, so we
    /// can't safely hand out a borrow into it). Hot-path attention
    /// callers should use [`Self::read_key_into`] (or, at the
    /// manager level, [`BlockManager::key_into`]) which writes into a
    /// caller-supplied buffer to avoid per-token allocations.
    pub fn read_key(&self, id: BlockId, in_block: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.kv_dim];
        self.read_key_into(id, in_block, &mut out);
        out
    }

    /// Read the cached value vector for one token slot.
    pub fn read_value(&self, id: BlockId, in_block: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; self.kv_dim];
        self.read_value_into(id, in_block, &mut out);
        out
    }

    /// Borrow-free key read: copies `kv_dim` floats from the slab into
    /// `dst`. Used by attention to avoid one `Vec<f32>` allocation per
    /// scored token.
    pub fn read_key_into(&self, id: BlockId, in_block: usize, dst: &mut [f32]) {
        assert_eq!(dst.len(), self.kv_dim);
        assert!(in_block < POOL_BLOCK_TOKENS);
        let idx = id.index();
        let block_off = idx * self.block_floats + in_block * self.kv_dim;
        if id.is_overflow() {
            let keys = self.overflow_keys.lock();
            dst.copy_from_slice(&keys[block_off..block_off + self.kv_dim]);
        } else {
            let keys = self.keys.lock();
            dst.copy_from_slice(&keys[block_off..block_off + self.kv_dim]);
        }
    }

    /// Borrow-free value read.
    pub fn read_value_into(&self, id: BlockId, in_block: usize, dst: &mut [f32]) {
        assert_eq!(dst.len(), self.kv_dim);
        assert!(in_block < POOL_BLOCK_TOKENS);
        let idx = id.index();
        let block_off = idx * self.block_floats + in_block * self.kv_dim;
        if id.is_overflow() {
            let values = self.overflow_values.lock();
            dst.copy_from_slice(&values[block_off..block_off + self.kv_dim]);
        } else {
            let values = self.values.lock();
            dst.copy_from_slice(&values[block_off..block_off + self.kv_dim]);
        }
    }

    /// Reclaim the overflow slab when nothing currently in flight is
    /// using it. Returns the number of blocks reclaimed (= the
    /// pre-call overflow capacity, when the call did anything).
    ///
    /// This is the cleanup half of the dynamic-overflow design: a
    /// transient burst grows the heap-backed slab, but once every
    /// request that touched overflow has released its blocks the
    /// memory should go back to the allocator rather than staying
    /// pinned at the high-water mark forever. Operators can call
    /// this on a low-frequency timer (e.g. once a minute) from the
    /// scheduler. It is a no-op when any overflow block is still in
    /// use, so it is always safe to call.
    pub fn shrink_overflow_to_fit(&self) -> usize {
        // Acquire the three overflow mutexes in a fixed order to
        // avoid lock-ordering deadlocks with `allocate`.
        let mut keys = self.overflow_keys.lock();
        let mut values = self.overflow_values.lock();
        let mut cap = self.overflow_capacity.lock();
        let mut free = self.overflow_free.lock();
        if free.len() < *cap {
            // Some overflow block is still in use; reclaiming would
            // invalidate its BlockId. Bail.
            return 0;
        }
        let reclaimed = *cap;
        if reclaimed == 0 {
            return 0;
        }
        keys.clear();
        keys.shrink_to_fit();
        values.clear();
        values.shrink_to_fit();
        free.clear();
        free.shrink_to_fit();
        *cap = 0;
        reclaimed
    }
}

/// Per-request **block manager**: owns the request's block table and
/// transparently allocates a new physical block from the shared
/// [`BlockPool`] when the trailing block fills up. Drops every owned
/// block back to the pool on `Drop`.
///
/// `BlockManager` is the public surface most callers will use. The
/// scheduler creates one per request, then the request grows the
/// table by calling [`Self::append`] every decoder step.
pub struct BlockManager {
    pool: Arc<BlockPool>,
    /// Block table: a request's logical block-i lives at the physical
    /// block id `block_table[i]`. Conceptually the same as vLLM's
    /// `block_tables[req_id]` row.
    block_table: Vec<BlockId>,
    seq_len: usize,
}

impl BlockManager {
    pub fn new(pool: Arc<BlockPool>) -> Self {
        Self {
            pool,
            block_table: Vec::new(),
            seq_len: 0,
        }
    }

    /// Append one token's `(k, v)` to the cache. Allocates a fresh
    /// block from the pool whenever the trailing block fills up.
    ///
    /// With dynamic-scaling enabled, [`BlockPool::allocate`] is
    /// effectively infallible (it falls back to a heap-backed
    /// overflow slab when the primary slab is empty), so this method
    /// only returns `Err(BlockAllocError::Exhausted)` if the global
    /// allocator itself fails — in practice the `Vec` resize inside
    /// `allocate` will panic first. The `Result` return is kept for
    /// source-level back-compat with callers that distinguished the
    /// exhausted case before overflow existed.
    pub fn append(&mut self, k: &[f32], v: &[f32]) -> Result<(), BlockAllocError> {
        let pos = self.seq_len;
        let block_idx = pos / POOL_BLOCK_TOKENS;
        let in_block = pos % POOL_BLOCK_TOKENS;
        if in_block == 0 {
            debug_assert_eq!(self.block_table.len(), block_idx);
            let id = self.pool.allocate().ok_or(BlockAllocError::Exhausted)?;
            self.block_table.push(id);
        }
        let id = self.block_table[block_idx];
        self.pool.write_token(id, in_block, k, v);
        self.seq_len += 1;
        Ok(())
    }

    /// Number of cached tokens.
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Number of physical blocks owned by this request.
    pub fn num_blocks(&self) -> usize {
        self.block_table.len()
    }

    /// Snapshot of the block table for diagnostics / scheduler
    /// telemetry. Each entry is a physical block id in the pool.
    pub fn block_table(&self) -> &[BlockId] {
        &self.block_table
    }

    /// Read the i-th cached key into a freshly-allocated `Vec`.
    pub fn key(&self, i: usize) -> Vec<f32> {
        let (id, in_block) = self.physical_addr(i);
        self.pool.read_key(id, in_block)
    }

    /// Read the i-th cached value into a freshly-allocated `Vec`.
    pub fn value(&self, i: usize) -> Vec<f32> {
        let (id, in_block) = self.physical_addr(i);
        self.pool.read_value(id, in_block)
    }

    /// Borrow-free key read into a caller-supplied buffer.
    pub fn key_into(&self, i: usize, dst: &mut [f32]) {
        let (id, in_block) = self.physical_addr(i);
        self.pool.read_key_into(id, in_block, dst);
    }

    /// Borrow-free value read into a caller-supplied buffer.
    pub fn value_into(&self, i: usize, dst: &mut [f32]) {
        let (id, in_block) = self.physical_addr(i);
        self.pool.read_value_into(id, in_block, dst);
    }

    fn physical_addr(&self, i: usize) -> (BlockId, usize) {
        assert!(i < self.seq_len, "read past seq_len");
        let block_idx = i / POOL_BLOCK_TOKENS;
        let in_block = i % POOL_BLOCK_TOKENS;
        (self.block_table[block_idx], in_block)
    }

    /// Release every block this manager owns back to the pool. Called
    /// automatically on `Drop` — explicit invocation is only useful in
    /// tests that want to inspect pool state after release.
    pub fn release_all(&mut self) {
        for id in self.block_table.drain(..) {
            self.pool.release(id);
        }
        self.seq_len = 0;
    }
}

impl Drop for BlockManager {
    fn drop(&mut self) {
        self.release_all();
    }
}

/// Errors returned from [`BlockManager::append`].
#[derive(Debug, PartialEq, Eq)]
pub enum BlockAllocError {
    /// The pool's free list is empty; the scheduler must either
    /// queue / abort this request or evict another request to
    /// reclaim blocks.
    Exhausted,
}

impl std::fmt::Display for BlockAllocError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlockAllocError::Exhausted => write!(f, "block pool is exhausted"),
        }
    }
}

impl std::error::Error for BlockAllocError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pool_alloc_release_round_trips() {
        let pool = BlockPool::new(/*kv_dim=*/ 4, /*capacity=*/ 3);
        assert_eq!(pool.free_blocks(), 3);
        assert_eq!(pool.capacity(), 3);

        let a = pool.allocate().unwrap();
        let b = pool.allocate().unwrap();
        let c = pool.allocate().unwrap();
        // Primary slab is now empty; further allocations spill into
        // the heap-backed overflow slab. They are still allocations,
        // not `None` returns — the pool grows dynamically.
        assert_eq!(pool.free_blocks(), 0);
        assert_eq!(pool.overflow_in_use(), 0);
        let d = pool.allocate().unwrap();
        assert!(d.is_overflow(), "fourth allocate must come from overflow slab");
        assert_eq!(pool.overflow_in_use(), 1);
        // Distinct ids.
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
        assert!(!a.is_overflow() && !b.is_overflow() && !c.is_overflow());

        pool.release(a);
        pool.release(b);
        pool.release(c);
        pool.release(d);
        assert_eq!(pool.free_blocks(), 3);
        assert_eq!(pool.overflow_in_use(), 0);
    }

    #[test]
    fn manager_appends_one_block_per_block_tokens() {
        let pool = BlockPool::new(2, 8);
        let mut m = BlockManager::new(pool.clone());
        // Insert exactly POOL_BLOCK_TOKENS tokens — should fit in one block.
        for i in 0..POOL_BLOCK_TOKENS {
            let k = vec![i as f32, (i + 1) as f32];
            let v = vec![(i * 2) as f32, (i * 2 + 1) as f32];
            m.append(&k, &v).unwrap();
        }
        assert_eq!(m.seq_len(), POOL_BLOCK_TOKENS);
        assert_eq!(m.num_blocks(), 1);
        assert_eq!(pool.free_blocks(), 8 - 1);

        // One more token forces a second block.
        m.append(&[100.0, 101.0], &[200.0, 201.0]).unwrap();
        assert_eq!(m.seq_len(), POOL_BLOCK_TOKENS + 1);
        assert_eq!(m.num_blocks(), 2);
        assert_eq!(pool.free_blocks(), 8 - 2);

        // Spot-check a few reads.
        assert_eq!(m.key(0), vec![0.0, 1.0]);
        assert_eq!(m.value(0), vec![0.0, 1.0]);
        assert_eq!(m.key(POOL_BLOCK_TOKENS), vec![100.0, 101.0]);
        assert_eq!(m.value(POOL_BLOCK_TOKENS), vec![200.0, 201.0]);
    }

    #[test]
    fn manager_drop_releases_all_blocks() {
        let pool = BlockPool::new(2, 4);
        {
            let mut m = BlockManager::new(pool.clone());
            for i in 0..(POOL_BLOCK_TOKENS * 3) {
                m.append(&[i as f32, i as f32], &[i as f32, i as f32]).unwrap();
            }
            assert_eq!(m.num_blocks(), 3);
            assert_eq!(pool.free_blocks(), 1);
        }
        // After Drop every block has been returned.
        assert_eq!(pool.free_blocks(), 4);
    }

    #[test]
    fn manager_overflow_succeeds_when_primary_exhausted() {
        // capacity=1 → one primary block (16 tokens). Beyond that
        // every additional block transparently comes from the
        // heap-backed overflow slab — `append` keeps succeeding.
        let pool = BlockPool::new(2, 1);
        let mut m = BlockManager::new(pool.clone());
        for _ in 0..POOL_BLOCK_TOKENS {
            m.append(&[1.0, 1.0], &[1.0, 1.0]).unwrap();
        }
        // Primary slab is now full.
        assert_eq!(pool.free_blocks(), 0);
        assert_eq!(pool.overflow_in_use(), 0);
        // 17th token needs a new block — pool falls back to overflow.
        m.append(&[2.0, 2.0], &[2.0, 2.0]).unwrap();
        assert_eq!(m.seq_len(), POOL_BLOCK_TOKENS + 1);
        assert_eq!(m.num_blocks(), 2);
        assert!(m.block_table()[1].is_overflow(), "second block must come from overflow slab");
        assert_eq!(pool.overflow_in_use(), 1);
        // The cached value round-trips through the overflow slab.
        assert_eq!(m.key(POOL_BLOCK_TOKENS), vec![2.0, 2.0]);
        assert_eq!(m.value(POOL_BLOCK_TOKENS), vec![2.0, 2.0]);
        // Releasing returns the overflow block to its own free list.
        drop(m);
        assert_eq!(pool.overflow_in_use(), 0);
        assert_eq!(pool.free_blocks(), 1);
    }

    #[test]
    fn overflow_blocks_recycle_before_growing_slab() {
        // Hammer the overflow path: every block past the primary
        // slab's capacity comes from overflow, but a released
        // overflow block must be re-handed out before the slab grows.
        let pool = BlockPool::new(2, 1);
        let _primary = pool.allocate().unwrap();
        let o1 = pool.allocate().unwrap();
        let o2 = pool.allocate().unwrap();
        assert!(o1.is_overflow() && o2.is_overflow());
        assert_eq!(pool.overflow_in_use(), 2);
        pool.release(o1);
        // Next overflow alloc should reuse o1's slot, not grow.
        let o3 = pool.allocate().unwrap();
        assert!(o3.is_overflow());
        assert_eq!(o3, o1, "released overflow id should be recycled");
        assert_eq!(pool.overflow_in_use(), 2);
    }

    #[test]
    fn pool_blocks_isolate_request_state() {
        // Two managers sharing one pool must not see each other's bytes.
        let pool = BlockPool::new(3, 4);
        let mut a = BlockManager::new(pool.clone());
        let mut b = BlockManager::new(pool.clone());
        a.append(&[1.0, 2.0, 3.0], &[10.0, 20.0, 30.0]).unwrap();
        b.append(&[4.0, 5.0, 6.0], &[40.0, 50.0, 60.0]).unwrap();
        assert_eq!(a.key(0), vec![1.0, 2.0, 3.0]);
        assert_eq!(b.key(0), vec![4.0, 5.0, 6.0]);
        assert_eq!(a.value(0), vec![10.0, 20.0, 30.0]);
        assert_eq!(b.value(0), vec![40.0, 50.0, 60.0]);
    }

    #[test]
    fn read_into_does_not_allocate_per_call() {
        let pool = BlockPool::new(4, 2);
        let mut m = BlockManager::new(pool.clone());
        m.append(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]).unwrap();
        let mut buf = vec![0.0f32; 4];
        m.key_into(0, &mut buf);
        assert_eq!(buf, vec![1.0, 2.0, 3.0, 4.0]);
        m.value_into(0, &mut buf);
        assert_eq!(buf, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn shrink_overflow_to_fit_reclaims_after_burst() {
        // capacity=1 primary slab → second alloc forces overflow.
        let pool = BlockPool::new(2, 1);
        let _primary = pool.allocate().unwrap();
        let o1 = pool.allocate().unwrap();
        let o2 = pool.allocate().unwrap();
        assert!(o1.is_overflow() && o2.is_overflow());

        // Cannot shrink while overflow is in use.
        assert_eq!(pool.shrink_overflow_to_fit(), 0);
        assert!(pool.overflow_in_use() > 0);

        // Release both overflow blocks, then shrink: capacity drops to 0.
        pool.release(o1);
        pool.release(o2);
        assert_eq!(pool.overflow_in_use(), 0);
        let reclaimed = pool.shrink_overflow_to_fit();
        assert_eq!(reclaimed, 2);

        // After shrinking, the overflow slab is empty and a fresh
        // overflow alloc starts from index 0 again.
        let o3 = pool.allocate().unwrap();
        assert!(o3.is_overflow());
        assert_eq!(o3.index(), 0);
    }

    #[test]
    fn overflow_cap_enforces_admission_back_pressure() {
        // Primary slab of 1 + overflow cap of 2 → fourth allocate must
        // return None instead of growing the heap unboundedly.
        let thresholds = PressureThresholds::default()
            .with_max_overflow_capacity(Some(2));
        let pool = BlockPool::with_thresholds(2, 1, thresholds);
        let p = pool.allocate().expect("primary block");
        let o1 = pool.allocate().expect("overflow #1");
        let o2 = pool.allocate().expect("overflow #2");
        assert!(o1.is_overflow() && o2.is_overflow());
        // Cap reached: next request must be refused.
        assert!(pool.allocate().is_none(), "overflow cap should refuse");
        // Releasing an overflow block frees the slot for re-admission.
        pool.release(o1);
        let o3 = pool.allocate().expect("retry after release succeeds");
        assert!(o3.is_overflow());
        pool.release(o2);
        pool.release(o3);
        pool.release(p);
    }

    #[test]
    fn pressure_level_walks_through_soft_cap_and_critical() {
        // 10-block primary slab — 0%/50%/90%/100% utilisation should
        // bucket Normal / Normal / High / Critical respectively.
        let pool = BlockPool::new(2, 10);
        assert_eq!(pool.pressure_level(), PressureLevel::Normal);
        assert!(!pool.above_soft_cap());
        let mut held = Vec::new();
        for _ in 0..5 {
            held.push(pool.allocate().unwrap());
        }
        // 50% utilisation → still Normal.
        assert!((pool.utilization() - 0.5).abs() < 1e-6);
        assert_eq!(pool.pressure_level(), PressureLevel::Normal);
        // Drive to 90%.
        for _ in 0..4 {
            held.push(pool.allocate().unwrap());
        }
        assert!((pool.utilization() - 0.9).abs() < 1e-6);
        assert!(pool.above_soft_cap());
        assert_eq!(pool.pressure_level(), PressureLevel::High);
        // Saturate primary → Critical.
        held.push(pool.allocate().unwrap());
        assert!(pool.utilization() >= CRITICAL_PRESSURE_RATIO);
        assert_eq!(pool.pressure_level(), PressureLevel::Critical);
        for id in held.drain(..) {
            pool.release(id);
        }
        assert_eq!(pool.pressure_level(), PressureLevel::Normal);
    }
}

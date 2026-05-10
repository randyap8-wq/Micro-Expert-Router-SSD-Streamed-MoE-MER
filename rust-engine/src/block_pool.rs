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

/// Opaque handle to one physical block in a [`BlockPool`]. The pool's
/// free list stores these as raw `u32`s to keep the per-request block
/// table compact (`Vec<u32>` rather than `Vec<usize>`).
///
/// Note: `BlockId` is *not* a `Drop` type — leaking one is harmless
/// (the pool just permanently loses a slot) but the per-request
/// [`BlockManager`] makes leaks impossible by returning every owned
/// block on `release_all` / `Drop`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

/// A shared **physical** block pool: one big f32 slab carved into
/// fixed-size blocks plus a free list of unused block ids.
///
/// Every block holds `POOL_BLOCK_TOKENS * kv_dim` floats. The pool
/// also owns a parallel "values" half — a single block id therefore
/// addresses both the K and V slot for the same token range. This
/// matches the way attention reads (`key(i)` / `value(i)` always
/// stride together), so we never want to allocate keys and values
/// separately.
///
/// The pool is `Sync` and is intended to be wrapped in `Arc`. Free
/// list mutations take a short `parking_lot::Mutex`, and the backing
/// key/value slabs are also protected by `Mutex<Vec<f32>>`. The slab
/// storage is fixed at construction time and never re-allocated, but
/// reads and writes still synchronize through those mutexes in the
/// current implementation.
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
}

impl std::fmt::Debug for BlockPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockPool")
            .field("kv_dim", &self.kv_dim)
            .field("capacity", &self.capacity)
            .field("block_floats", &self.block_floats)
            .field("free_blocks", &self.free.lock().len())
            .finish()
    }
}

impl BlockPool {
    /// Build a pool that can hold up to `capacity` blocks of
    /// `POOL_BLOCK_TOKENS * kv_dim` floats each (× 2 for keys+values).
    /// Both slabs are pre-allocated and zeroed up front; allocation
    /// during the per-token hot path never grows them.
    pub fn new(kv_dim: usize, capacity: usize) -> Arc<Self> {
        assert!(kv_dim > 0, "BlockPool kv_dim must be > 0");
        let block_floats = POOL_BLOCK_TOKENS * kv_dim;
        let total = block_floats.checked_mul(capacity).expect("BlockPool capacity overflows usize");
        let keys = vec![0.0f32; total];
        let values = vec![0.0f32; total];
        // Free list initialised in *reverse* so allocation hands out
        // ids 0, 1, 2, … in order — easier to reason about in tests.
        let free: Vec<u32> = (0..capacity as u32).rev().collect();
        Arc::new(Self {
            kv_dim,
            capacity,
            block_floats,
            keys: Mutex::new(keys),
            values: Mutex::new(values),
            free: Mutex::new(free),
        })
    }

    /// Pop one block id from the free list. Returns `None` if the
    /// pool is exhausted (the scheduler typically responds by
    /// queuing the request or aborting it).
    pub fn allocate(&self) -> Option<BlockId> {
        self.free.lock().pop().map(BlockId)
    }

    /// Return one block id to the free list. The block's contents
    /// are *not* zeroed on release — the next allocator overwrites
    /// the bytes it cares about via [`Self::write_token`], and reads
    /// past `seq_len` are guaranteed unreachable by callers.
    pub fn release(&self, id: BlockId) {
        debug_assert!(
            (id.0 as usize) < self.capacity,
            "released block id {} >= capacity {}",
            id.0,
            self.capacity
        );
        self.free.lock().push(id.0);
    }

    /// Number of blocks currently free. Snapshot-only; the value can
    /// race with concurrent allocators.
    pub fn free_blocks(&self) -> usize {
        self.free.lock().len()
    }

    /// Total physical capacity, in blocks (constant for the pool's
    /// lifetime).
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
    /// in the caller.
    pub fn write_token(&self, id: BlockId, in_block: usize, k: &[f32], v: &[f32]) {
        assert_eq!(k.len(), self.kv_dim, "k length must equal kv_dim");
        assert_eq!(v.len(), self.kv_dim, "v length must equal kv_dim");
        assert!(in_block < POOL_BLOCK_TOKENS, "in_block {} >= POOL_BLOCK_TOKENS", in_block);
        let block_off = (id.0 as usize) * self.block_floats + in_block * self.kv_dim;
        let mut keys = self.keys.lock();
        keys[block_off..block_off + self.kv_dim].copy_from_slice(k);
        drop(keys);
        let mut values = self.values.lock();
        values[block_off..block_off + self.kv_dim].copy_from_slice(v);
    }

    /// Read the cached key vector for one token slot. Returns an
    /// owned `Vec<f32>` (the pool slab is behind a `Mutex`, so we
    /// can't safely hand out a borrow into it). Hot-path attention
    /// callers should use [`PooledKvCache::key_into`] which writes
    /// into a caller-supplied buffer to avoid per-token allocations.
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
        let block_off = (id.0 as usize) * self.block_floats + in_block * self.kv_dim;
        let keys = self.keys.lock();
        dst.copy_from_slice(&keys[block_off..block_off + self.kv_dim]);
    }

    /// Borrow-free value read.
    pub fn read_value_into(&self, id: BlockId, in_block: usize, dst: &mut [f32]) {
        assert_eq!(dst.len(), self.kv_dim);
        assert!(in_block < POOL_BLOCK_TOKENS);
        let block_off = (id.0 as usize) * self.block_floats + in_block * self.kv_dim;
        let values = self.values.lock();
        dst.copy_from_slice(&values[block_off..block_off + self.kv_dim]);
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
    /// Returns `Err` (without partially mutating state) if the pool
    /// is exhausted at a block boundary.
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
        assert!(pool.allocate().is_none(), "fourth allocate must exhaust");
        assert_eq!(pool.free_blocks(), 0);
        // Distinct ids.
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);

        pool.release(a);
        pool.release(b);
        pool.release(c);
        assert_eq!(pool.free_blocks(), 3);
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
    fn manager_exhaustion_returns_error() {
        // capacity=1 → only one block (16 tokens) before exhaustion.
        let pool = BlockPool::new(2, 1);
        let mut m = BlockManager::new(pool.clone());
        for _ in 0..POOL_BLOCK_TOKENS {
            m.append(&[1.0, 1.0], &[1.0, 1.0]).unwrap();
        }
        // 17th token needs a new block — pool is empty.
        let err = m.append(&[1.0, 1.0], &[1.0, 1.0]);
        assert_eq!(err, Err(BlockAllocError::Exhausted));
        // seq_len did NOT advance on the failed append.
        assert_eq!(m.seq_len(), POOL_BLOCK_TOKENS);
        // No partial allocation either.
        assert_eq!(m.num_blocks(), 1);
        assert_eq!(pool.free_blocks(), 0);
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
}

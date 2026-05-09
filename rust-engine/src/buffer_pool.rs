//! Fixed-size slab pool of pre-allocated, page-aligned RAM buffers.
//!
//! The spec calls for "hot-swapping experts ... into a **pre-allocated** RAM
//! buffer". A naive implementation would `Vec::with_capacity` per cache miss,
//! which (a) is not aligned for `O_DIRECT`, (b) thrashes the allocator, and
//! (c) makes total RAM use unbounded.
//!
//! Instead, at startup we allocate `slots` aligned buffers of `expert_size`
//! bytes and hand them out as [`PooledBuffer`] guards. When a guard is dropped
//! its buffer is returned to the pool's free list — so the LRU eviction of an
//! expert automatically frees a slot for the next miss.

use crate::aligned_buffer::AlignedBuffer;
use parking_lot::Mutex;
use std::sync::Arc;
use tokio::sync::Notify;

struct Inner {
    free: Mutex<Vec<AlignedBuffer>>,
    notify: Notify,
    slots: usize,
    buffer_size: usize,
    align: usize,
}

/// A bounded pool of aligned RAM buffers.
#[derive(Clone)]
pub struct BufferPool {
    inner: Arc<Inner>,
}

impl BufferPool {
    /// Pre-allocate `slots` buffers of `buffer_size` bytes each, aligned to `align`.
    pub fn new(slots: usize, buffer_size: usize, align: usize) -> Self {
        assert!(slots > 0, "pool must have at least one slot");
        let mut free = Vec::with_capacity(slots);
        for _ in 0..slots {
            free.push(AlignedBuffer::new(buffer_size, align));
        }
        Self {
            inner: Arc::new(Inner {
                free: Mutex::new(free),
                notify: Notify::new(),
                slots,
                buffer_size,
                align,
            }),
        }
    }

    /// Total number of slots in the pool.
    pub fn capacity(&self) -> usize {
        self.inner.slots
    }

    /// Size of each buffer in the pool, in bytes.
    pub fn buffer_size(&self) -> usize {
        self.inner.buffer_size
    }

    /// Required alignment of each buffer (matches `O_DIRECT` requirements).
    pub fn align(&self) -> usize {
        self.inner.align
    }

    /// Snapshot the raw `(ptr, len)` of every currently-free buffer.
    /// Intended for the `io_uring` registered-fixed-buffers path: at
    /// startup, callers register every pool buffer with the kernel
    /// once, then submit reads referencing buffer indices instead of
    /// per-read iovecs. Returned pointers are stable for the lifetime
    /// of the pool (each `AlignedBuffer` owns a heap allocation that
    /// the pool keeps alive).
    ///
    /// **Safety contract:** the caller must not write to these
    /// pointers concurrently with another holder of the corresponding
    /// `PooledBuffer`. In practice this means: register the iovecs
    /// once at startup, before any `acquire()` call, and never modify
    /// the pool's free-list contents while reads are in flight.
    pub fn raw_iovecs(&self) -> Vec<(*mut u8, usize)> {
        let mut g = self.inner.free.lock();
        g.iter_mut()
            .map(|b| (b.as_mut_slice().as_mut_ptr(), b.len()))
            .collect()
    }

    /// Try to pop a free buffer immediately. Returns `None` if the pool is empty.
    pub fn try_acquire(&self) -> Option<PooledBuffer> {
        let buf = self.inner.free.lock().pop()?;
        Some(PooledBuffer {
            buffer: Some(buf),
            pool: self.inner.clone(),
        })
    }

    /// Wait asynchronously until a free buffer is available, then return it.
    ///
    /// This provides natural backpressure: when the cache is full and every
    /// resident expert is referenced, new fetches simply wait for one to be
    /// dropped (i.e. evicted *and* released by the inference layer).
    pub async fn acquire(&self) -> PooledBuffer {
        loop {
            if let Some(b) = self.try_acquire() {
                return b;
            }
            // Register interest *before* re-checking to avoid a lost wakeup.
            let notified = self.inner.notify.notified();
            if let Some(b) = self.try_acquire() {
                return b;
            }
            notified.await;
        }
    }
}

/// RAII guard wrapping an [`AlignedBuffer`] borrowed from a [`BufferPool`].
///
/// Dropping the guard returns the buffer to the pool. The buffer is **not**
/// zeroed on return; callers should treat newly-acquired buffers as
/// uninitialised beyond the byte range they explicitly write.
pub struct PooledBuffer {
    buffer: Option<AlignedBuffer>,
    pool: Arc<Inner>,
}

impl PooledBuffer {
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        self.buffer.as_ref().expect("PooledBuffer must hold a buffer until Drop").as_slice()
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.buffer.as_mut().expect("PooledBuffer must hold a buffer until Drop").as_mut_slice()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.buffer.as_ref().expect("PooledBuffer must hold a buffer until Drop").len()
    }
}

impl std::ops::Deref for PooledBuffer {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsRef<[u8]> for PooledBuffer {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsMut<[u8]> for PooledBuffer {
    fn as_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if let Some(buf) = self.buffer.take() {
            self.pool.free.lock().push(buf);
            self.pool.notify.notify_one();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn acquire_and_release_cycle() {
        let pool = BufferPool::new(2, 4096, 4096);
        let a = pool.acquire().await;
        let b = pool.acquire().await;
        assert!(pool.try_acquire().is_none());
        drop(a);
        assert!(pool.try_acquire().is_some());
        drop(b);
    }

    #[tokio::test]
    async fn acquire_waits_for_release() {
        let pool = BufferPool::new(1, 4096, 4096);
        let held = pool.acquire().await;
        let pool2 = pool.clone();
        let h = tokio::spawn(async move {
            let _b = pool2.acquire().await;
        });
        // Give the spawned task a moment to park.
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        drop(held);
        h.await.unwrap();
    }

    #[test]
    fn raw_iovecs_reports_every_free_buffer() {
        // With nothing acquired, raw_iovecs reports all `slots` buffers.
        let pool = BufferPool::new(3, 4096, 4096);
        let iovecs = pool.raw_iovecs();
        assert_eq!(iovecs.len(), 3);
        for (p, l) in iovecs {
            assert!(!p.is_null());
            assert_eq!(l, 4096);
        }
    }
}

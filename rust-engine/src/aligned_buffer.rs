//! Page-aligned heap buffers for `O_DIRECT`.
//!
//! The Linux kernel rejects `O_DIRECT` reads/writes whose **buffer address**,
//! **length**, and **file offset** are not aligned to the underlying device's
//! logical block size (512 B on legacy disks, 4096 B on modern NVMe; some
//! enterprise drives use 8 KiB or 16 KiB).
//!
//! `Vec<u8>` only guarantees `align_of::<u8>() == 1`, so we cannot use it
//! directly for `O_DIRECT`. This module provides [`AlignedBuffer`], a small
//! wrapper around [`std::alloc::alloc`] with a chosen alignment.

use std::alloc::{self, Layout};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

/// A heap-allocated, page-aligned buffer suitable for `O_DIRECT` I/O.
///
/// The buffer length is fixed at construction time and the contents are
/// initialised to zero so that a partial read leaves observable bytes valid.
pub struct AlignedBuffer {
    ptr: NonNull<u8>,
    len: usize,
    align: usize,
}

// SAFETY: we own the allocation exclusively and there is no interior
// mutability beyond what `&mut self` allows.
unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

impl AlignedBuffer {
    /// Allocate a zero-initialised buffer of `size` bytes, aligned to `align`.
    ///
    /// Panics if `align` is not a power of two, if `size` is not a multiple of
    /// `align` (a hard requirement for `O_DIRECT`), or if allocation fails.
    pub fn new(size: usize, align: usize) -> Self {
        assert!(align.is_power_of_two(), "alignment must be a power of two");
        assert!(size > 0, "buffer size must be > 0");
        assert!(
            size % align == 0,
            "buffer size {size} must be a multiple of alignment {align} for O_DIRECT"
        );

        let layout = Layout::from_size_align(size, align).expect("invalid layout");
        // SAFETY: layout is valid (checked above). alloc_zeroed returns null on OOM.
        let raw = unsafe { alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(raw).unwrap_or_else(|| alloc::handle_alloc_error(layout));

        Self { ptr, len: size, align }
    }

    /// Length of the buffer in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Alignment of the buffer's start address (in bytes).
    #[inline]
    pub fn align(&self) -> usize {
        self.align
    }

    /// Borrow the buffer as a byte slice.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is valid, exclusively owned, len is correct, bytes are
        // initialised (alloc_zeroed) and `u8` has no validity invariants.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Mutably borrow the buffer as a byte slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above; exclusive access is enforced by `&mut self`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        // SAFETY: layout matches the one used in `new`.
        let layout = Layout::from_size_align(self.len, self.align).expect("invalid layout");
        unsafe { alloc::dealloc(self.ptr.as_ptr(), layout) };
    }
}

impl Deref for AlignedBuffer {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl DerefMut for AlignedBuffer {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

impl AsRef<[u8]> for AlignedBuffer {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsMut<[u8]> for AlignedBuffer {
    fn as_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_is_aligned() {
        let buf = AlignedBuffer::new(4096, 4096);
        assert_eq!(buf.as_slice().as_ptr() as usize % 4096, 0);
        assert_eq!(buf.len(), 4096);
        assert!(buf.iter().all(|&b| b == 0));
    }

    #[test]
    #[should_panic]
    fn rejects_non_power_of_two_alignment() {
        let _ = AlignedBuffer::new(1024, 1000);
    }

    #[test]
    #[should_panic]
    fn rejects_size_not_multiple_of_alignment() {
        let _ = AlignedBuffer::new(4097, 4096);
    }

    /// **Gist Task 2 — proptest for `AlignedBuffer` slice
    /// arithmetic.**
    ///
    /// For any (size, align) pair that respects the constructor
    /// preconditions (align is a power of two, size > 0, and size
    /// is a multiple of align), the returned buffer must:
    ///   * be aligned to `align` (the whole point of `O_DIRECT`),
    ///   * expose a slice of exactly `size` bytes,
    ///   * start out zeroed,
    ///   * survive arbitrary in-bounds reads/writes through
    ///     `as_mut_slice()` without buffer overruns. This indirectly
    ///     fuzzes the slice arithmetic used by callers like
    ///     `AlignedKvCache::row_floats`, which does
    ///     `&buf[pos*row_bytes .. pos*row_bytes+row_bytes]` then
    ///     reinterprets as `&[f32]`.
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(ProptestConfig {
                cases: 64,
                ..ProptestConfig::default()
            })]

            #[test]
            fn arbitrary_aligned_buffers_are_aligned_zeroed_and_indexable(
                // Restrict alignment to plausible block sizes
                // (powers of two from 512 B to 16 KiB).
                align_shift in 9u32..15,
                // Multiplier picks `size = align * mul` so the
                // "size must be a multiple of align" precondition
                // always holds.
                mul in 1usize..32,
                // Per-row stride and number of rows for the slice-
                // arithmetic check.
                rows in 1usize..8,
            ) {
                let align = 1usize << align_shift;
                let size = align * mul;
                let mut buf = AlignedBuffer::new(size, align);
                prop_assert_eq!(buf.len(), size);
                prop_assert_eq!(buf.align(), align);
                prop_assert_eq!(buf.as_slice().as_ptr() as usize % align, 0);
                prop_assert!(buf.iter().all(|&b| b == 0), "fresh buffer must be zeroed");

                // Slice-arithmetic exercise: divide the buffer into
                // `rows` equal stripes and write a per-row marker
                // byte, then read it back to confirm in-bounds
                // access never overruns the allocation.
                let row_bytes = size / rows.max(1);
                if row_bytes > 0 {
                    let slice = buf.as_mut_slice();
                    for r in 0..rows {
                        let start = r * row_bytes;
                        let end = start + row_bytes;
                        if end > slice.len() { break; }
                        for b in &mut slice[start..end] {
                            *b = (r as u8).wrapping_add(0x55);
                        }
                    }
                    let read = buf.as_slice();
                    for r in 0..rows {
                        let start = r * row_bytes;
                        let end = start + row_bytes;
                        if end > read.len() { break; }
                        let want = (r as u8).wrapping_add(0x55);
                        prop_assert!(
                            read[start..end].iter().all(|&b| b == want),
                            "row {r} arithmetic failed: stride {row_bytes}",
                        );
                    }
                }
            }
        }
    }
}

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
}

// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ptr::NonNull;

use jxl::api::JxlOutputBuffer;

use crate::conversions::to_rs_endianness;
use crate::types::{
    JXL_TYPE_FLOAT, JXL_TYPE_FLOAT16, JXL_TYPE_UINT8, JXL_TYPE_UINT16, JxlDataType, JxlPixelFormat,
};

pub fn bytes_per_sample(data_type: JxlDataType) -> Option<usize> {
    match data_type {
        JXL_TYPE_UINT8 => Some(1),
        JXL_TYPE_UINT16 | JXL_TYPE_FLOAT16 => Some(2),
        JXL_TYPE_FLOAT => Some(4),
        _ => None,
    }
}

pub fn compute_row_layout(width: usize, format: &JxlPixelFormat) -> Option<(usize, usize)> {
    if !(1..=4).contains(&format.num_channels) {
        return None;
    }
    let _ = to_rs_endianness(format.endianness)?;
    let bps = bytes_per_sample(format.data_type)?;
    let ch = format.num_channels as usize;
    let bytes_per_row = width.checked_mul(ch)?.checked_mul(bps)?;
    let stride = if format.align > 1 {
        bytes_per_row.checked_next_multiple_of(format.align)?
    } else {
        bytes_per_row
    };
    Some((bytes_per_row, stride))
}

pub fn compute_buffer_size(width: usize, height: usize, format: &JxlPixelFormat) -> Option<usize> {
    let (bytes_per_row, stride) = compute_row_layout(width, format)?;
    if width == 0 || height == 0 {
        return Some(0);
    }
    let total = stride.checked_mul(height - 1)?.checked_add(bytes_per_row)?;
    if total > isize::MAX as usize {
        return None;
    }
    Some(total)
}

/// A borrowed C input byte buffer registered via `JxlDecoderSetInput`.
pub(crate) struct InputBuffer {
    // Safety invariant: During calls to functions in this crate (until `self` is dropped), either
    // `buf.len() == 0` or `buf` points to `buf.len()` initialized bytes (within a single
    // allocation of at most `isize::MAX` bytes) that are valid for reads and not modified
    // concurrently.
    buf: NonNull<[u8]>,
}

impl InputBuffer {
    /// Wraps a raw C input pointer and byte length. Returns `None` if `data` is null and
    /// `size > 0`.
    ///
    /// # Safety
    /// If `data` is non-null and `size > 0`, the caller must ensure that during calls to functions
    /// in this crate until this `InputBuffer` is dropped, `data` is valid for reads of `size`
    /// initialized bytes (within a single allocation of at most `isize::MAX` bytes) and the
    /// pointed-to bytes are not modified concurrently.
    pub(crate) unsafe fn new(data: *const u8, size: usize) -> Option<Self> {
        let non_null_ptr = if size == 0 {
            NonNull::<u8>::dangling()
        } else {
            NonNull::new(data.cast_mut())?
        };
        Some(Self {
            buf: NonNull::slice_from_raw_parts(non_null_ptr, size),
        })
    }

    pub(crate) fn remaining(&self) -> usize {
        self.buf.len()
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        // SAFETY: We are inside a function in this crate, and `&self` cannot outlive this call or
        // coexist with `&mut self` methods that release the buffer, so by `InputBuffer`'s safety
        // invariant `self.buf` is valid for reads and not modified for the lifetime of `&self`.
        unsafe { self.buf.as_ref() }
    }

    pub(crate) fn consume(&mut self, amount: usize) {
        self.buf = NonNull::from(&self.as_slice()[amount..]);
    }
}

/// A borrowed C output byte buffer registered via `JxlDecoderSetBoxBuffer`.
pub(crate) struct BoxOutBuffer {
    // Safety invariant: During calls to functions in this crate (until `self` is dropped), either
    // `buf.len() == 0` or `buf` points to `buf.len()` writable bytes (within a single allocation
    // of at most `isize::MAX` bytes) that remain exclusively accessible.
    buf: NonNull<[u8]>,
}

impl BoxOutBuffer {
    /// Wraps a raw C output pointer and byte length. Returns `None` if `data` is null and
    /// `size > 0`.
    ///
    /// # Safety
    /// If `data` is non-null and `size > 0`, the caller must ensure that during calls to functions
    /// in this crate until this `BoxOutBuffer` is dropped, `data` is valid for writes of `size`
    /// bytes (within a single allocation of at most `isize::MAX` bytes) and remains exclusively
    /// accessible.
    pub(crate) unsafe fn new(data: *mut u8, size: usize) -> Option<Self> {
        let non_null_ptr = if size == 0 {
            NonNull::<u8>::dangling()
        } else {
            NonNull::new(data)?
        };
        Some(Self {
            buf: NonNull::slice_from_raw_parts(non_null_ptr, size),
        })
    }

    pub(crate) fn remaining(&self) -> usize {
        self.buf.len()
    }

    pub(crate) fn write(&mut self, src: &[u8]) -> usize {
        let n = self.buf.len().min(src.len());
        let ptr = self.buf.as_ptr().cast::<u8>();
        // SAFETY: Either `n == 0` (with non-null `ptr` and `ptr.add(0) == ptr`), or we are inside
        // a function in this crate so by `BoxOutBuffer`'s safety invariant `ptr` is valid for
        // writes of `n <= self.buf.len()` exclusively accessible bytes within a single
        // `<= isize::MAX` allocation (discharging `copy_nonoverlapping`, `ptr.add(n)`, and
        // `NonNull::new_unchecked`).
        let next_ptr = unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), ptr, n);
            NonNull::new_unchecked(ptr.add(n))
        };
        self.buf = NonNull::slice_from_raw_parts(next_ptr, self.buf.len() - n);
        n
    }
}

/// A validated user-provided output pixel buffer and its format.
#[derive(Debug)]
pub(crate) struct ImageOutBuffer {
    // Safety invariant: During calls to functions in this crate (until `self` is dropped), `ptr`
    // is valid for writes of `size` bytes (within a single allocation) that remain exclusively
    // accessible.
    ptr: NonNull<u8>,
    size: usize,
    format: JxlPixelFormat,
}

impl ImageOutBuffer {
    /// Creates a new `ImageOutBuffer` after verifying `buffer` is non-null and `size >= min_size`.
    ///
    /// # Safety
    /// If `buffer` is non-null and `size >= min_size`, the caller must ensure that during calls to
    /// functions in this crate until this `ImageOutBuffer` is dropped, `buffer` is valid for
    /// writes of `size` bytes (within a single allocation) and remains exclusively accessible.
    pub(crate) unsafe fn new(
        buffer: *mut u8,
        size: usize,
        min_size: usize,
        format: JxlPixelFormat,
    ) -> Option<Self> {
        if size < min_size {
            return None;
        }
        let ptr = NonNull::new(buffer)?;
        Some(Self { ptr, size, format })
    }

    pub(crate) fn format(&self) -> JxlPixelFormat {
        self.format
    }

    /// Prepares a `JxlOutputBuffer` for `jxl-rs` to write into directly.
    pub(crate) fn make_output_buffer(
        &mut self,
        width: usize,
        height: usize,
    ) -> Option<JxlOutputBuffer<'_>> {
        let (bytes_per_row, stride) = compute_row_layout(width, &self.format)?;
        let needed = compute_buffer_size(width, height, &self.format)?;
        if self.size < needed {
            return None;
        }
        // SAFETY: We are inside a function in this crate, so by `ImageOutBuffer`'s safety invariant
        // `self.ptr` points to at least `self.size >= needed` writable, exclusively accessible
        // bytes for the lifetime of `&mut self`, satisfying the preconditions of `new_from_ptr`.
        Some(unsafe {
            JxlOutputBuffer::new_from_ptr(self.ptr.as_ptr(), height, bytes_per_row, stride)
        })
    }
}

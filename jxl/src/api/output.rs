// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(unsafe_code)]

use core::slice;
use std::{marker::PhantomData, mem::MaybeUninit, ops::Range};

use crate::image::{Image, ImageDataType};

pub struct JxlOutputBuffer<'a> {
    // Safety invariants:
    //  - uninit data is never written to `buf`.
    //  - `buf` is valid for writes for all bytes in the range
    //    `buf[i*bytes_between_rows..i*bytes_between_rows+bytes_per_row]` for all values of `i`
    //    from `0` to `num_rows-1`.
    //  - `self` has exclusive (write) access to the bytes in these ranges.
    //  - all the bytes in those ranges (and in between) are part of the same allocated object.
    //  - `num_rows > 0`, `bytes_per_row > 0`, `bytes_per_row <= bytes_between_rows`.
    //  - The computation `E = bytes_between_rows * (num_rows-1) + bytes_per_row` does not
    //    overflow and has a result that is at most `isize::MAX`.
    buf: *mut MaybeUninit<u8>,
    bytes_per_row: usize,
    num_rows: usize,
    bytes_between_rows: usize,
    _ph: PhantomData<&'a mut u8>,
}

impl<'a> JxlOutputBuffer<'a> {
    fn check_vals(num_rows: usize, bytes_per_row: usize, bytes_between_rows: usize) {
        assert!(num_rows > 0);
        assert!(bytes_per_row > 0);
        assert!(bytes_between_rows >= bytes_per_row);
        assert!(
            bytes_between_rows
                .checked_mul(num_rows - 1)
                .unwrap()
                .checked_add(bytes_per_row)
                .unwrap()
                <= isize::MAX as usize
        );
    }

    /// Creates a new JxlOutputBuffer from raw pointers.
    /// It is guaranteed that `buf` will never be used to write uninitialized data.
    ///
    /// # Safety
    /// - `buf` must be valid for writes for all bytes in the range
    ///   `buf[i*bytes_between_rows..i*bytes_between_rows+bytes_per_row]` for all values of `i`
    ///   from `0` to `num_rows-1`.
    /// - The bytes in these ranges must not be accessed as long as the returned `Self` is in scope.
    /// - All the bytes in those ranges (and in between) must be part of the same allocated object.
    pub unsafe fn new_from_ptr(
        buf: *mut MaybeUninit<u8>,
        num_rows: usize,
        bytes_per_row: usize,
        bytes_between_rows: usize,
    ) -> Self {
        Self::check_vals(num_rows, bytes_per_row, bytes_between_rows);
        // Safety: caller guarantees the buf-related requirements, and other safety invariants are
        // checked in check_vals.
        Self {
            buf,
            bytes_per_row,
            bytes_between_rows,
            num_rows,
            _ph: PhantomData,
        }
    }

    pub fn new(buf: &'a mut [u8], num_rows: usize, bytes_per_row: usize) -> Self {
        Self::new_uninit(
            // Safety: `new_uninit` guarantees that no uninit data is ever written to the passed-in
            // slice. Moreover, `T` and `MaybeUninit<T>` have the same memory layout.
            unsafe { slice::from_raw_parts_mut(buf.as_mut_ptr().cast(), buf.len()) },
            num_rows,
            bytes_per_row,
        )
    }

    /// Creates a new JxlOutputBuffer from a slice of uninit data.
    /// It is guaranteed that `buf` will never be used to write uninitalized data.
    pub fn new_uninit(
        buf: &'a mut [MaybeUninit<u8>],
        num_rows: usize,
        bytes_per_row: usize,
    ) -> Self {
        Self::check_vals(num_rows, bytes_per_row, bytes_per_row);
        assert!(buf.len() >= bytes_per_row * num_rows);
        // Safety note: buf-related requirements follow from this function mutably borrowing a
        // slice, and other safety invariants are checked in check_vals.
        Self {
            buf: buf.as_mut_ptr(),
            bytes_per_row,
            bytes_between_rows: bytes_per_row,
            num_rows,
            _ph: PhantomData,
        }
    }

    /// # Safety
    /// No uninit data must be written to the returned slice.
    pub unsafe fn get(&mut self, row: usize, cols: Range<usize>) -> &mut [MaybeUninit<u8>] {
        assert!(row < self.num_rows);
        assert!(cols.start <= cols.end);
        assert!(cols.end <= self.bytes_per_row);
        let start = row * self.bytes_between_rows + cols.start;
        // Safety: `start` is guaranteed to be <= isize::MAX, and `self.buf + start` is guaranteed
        // to fit within the same allocated object, as per safety invariants of this struct.
        // We checked above that `row` and `cols` satisfy the requirements to apply the safety
        // invariant.
        let start = unsafe { self.buf.add(start) };
        // Safety: due to the struct safety invariant, we know the entire slice is in a range of
        // memory valid for writes. Moreover, the caller promises not to write uninitialized data
        // in the returned slice. Finally, as we take self by mutable reference and `self` has
        // exclusive access to the slices described in the safety invariant, we know aliasing
        // rules will not be violated.
        unsafe { slice::from_raw_parts_mut(start, cols.len()) }
    }

    pub(crate) fn from_image<T: ImageDataType>(image: &'a mut Image<T>) -> Self {
        let (width, height) = image.size();
        let buf = image.buf_mut();
        assert!(buf.len() >= width * height);
        let bytes_per_pixel = std::mem::size_of::<T>();
        let bytes_per_row = width * bytes_per_pixel;
        let buf_ptr = buf.as_mut_ptr() as *mut MaybeUninit<u8>;

        // Safety: The pointer comes from a mutable slice of a valid Image,
        // which lives for 'a. The dimensions are derived from the image itself.
        // T is guaranteed to be a bag-of-bits type, so the transmute from
        // a pointer to T to a pointer to u8 is safe.
        // new_from_ptr will check the invariants (e.g. non-empty).
        // The JxlOutputBuffer is guaranteed to not write uninitialized data to
        // the buffer.
        unsafe { Self::new_from_ptr(buf_ptr, height, bytes_per_row, bytes_per_row) }
    }

    pub(crate) fn from_output_buffer(lender: &'a mut JxlOutputBuffer<'_>) -> Self {
        // Safety note: this is effectively equivalent to a reborrow. Moreover, the safety
        // invariants are preserved by virtue of copying all the fields.
        Self {
            _ph: PhantomData,
            ..*lender
        }
    }
}

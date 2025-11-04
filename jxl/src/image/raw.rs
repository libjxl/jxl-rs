// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::{fmt::Debug, marker::PhantomData};

use crate::error::Result;

use super::{Rect, internal::RawImageBuffer};

#[repr(transparent)]
pub struct OwnedRawImage {
    // Safety invariant: all the accessible bytes of `self.data` are initialized, and
    // belongs to a single allocation that lives until `self` is dropped.
    // The data referenced by self.data was allocated by RawImageBuffer::try_allocate.
    // `data.is_aligned(CACHE_LINE_BYTE_SIZE)` is true.
    pub(super) data: RawImageBuffer,
}

impl OwnedRawImage {
    pub fn new_zeroed(byte_size: (usize, usize)) -> Result<Self> {
        Ok(Self {
            // Safety note: the returned memory is initialized and part of a single allocation of
            // the correct length.
            data: RawImageBuffer::try_allocate(byte_size, false)?,
        })
    }

    pub fn as_rect_mut(&mut self) -> RawImageRectMut<'_> {
        RawImageRectMut {
            // Safety note: we are lending exclusive ownership to RawImageRectMut.
            data: self.data,
            _ph: PhantomData,
        }
    }

    pub fn as_rect(&'_ self) -> RawImageRect<'_> {
        RawImageRect {
            // Safety note: correctness ensured by the return value borrowing from `self`.
            data: self.data,
            _ph: PhantomData,
        }
    }

    pub fn byte_size(&self) -> (usize, usize) {
        self.data.byte_size()
    }

    pub fn try_clone(&self) -> Result<OwnedRawImage> {
        Ok(Self {
            // SAFETY: we own the data that self.data references, so it is all accessible.
            // Moreover, it is initialized and try_clone creates a copy, so the resulting data is
            // owned and initialized.
            data: unsafe { self.data.try_clone()? },
        })
    }
}

impl Drop for OwnedRawImage {
    fn drop(&mut self) {
        // SAFETY: we own the data referenced by self.data, and it was allocated by
        // RawImageBuffer::try_allocate.
        unsafe {
            self.data.deallocate();
        }
    }
}

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct RawImageRect<'a> {
    // Safety invariant: all the accessible bytes of `self.data` are initialized.
    pub(super) data: RawImageBuffer,
    _ph: PhantomData<&'a u8>,
}

impl<'a> RawImageRect<'a> {
    #[inline(always)]
    pub fn row(&self, row: usize) -> &[u8] {
        // SAFETY: we have shared access to the accessible bytes of `self.data`.
        let row = unsafe { self.data.row(row) };
        // SAFETY: MaybeUninit<u8> and u8 have the same size and layout, and our safety invariant
        // guarantees the data is initialized.
        unsafe { std::slice::from_raw_parts(row.as_ptr() as *const u8, row.len()) }
    }

    pub fn rect(&self, rect: Rect) -> RawImageRect<'a> {
        Self {
            // Safety note: correctness ensured by the fact that the return value still borrows
            // from the original data source.
            data: self.data.rect(rect),
            _ph: PhantomData,
        }
    }

    pub fn byte_size(&self) -> (usize, usize) {
        self.data.byte_size()
    }
}

#[repr(transparent)]
pub struct RawImageRectMut<'a> {
    // Safety invariant: all the accessible bytes of `self.data` are initialized and we have
    // exclusive access to them.
    pub(super) data: RawImageBuffer,
    _ph: PhantomData<&'a mut u8>,
}

impl<'a> RawImageRectMut<'a> {
    #[inline(always)]
    pub fn row(&mut self, row: usize) -> &mut [u8] {
        // SAFETY: we don't write uninit data to `row`, and we have exclusive access to the accessible
        // bytes of `self.data`.
        let row = unsafe { self.data.row_mut(row) };
        // SAFETY: MaybeUninit<u8> and u8 have the same size and layout, and our safety invariant
        // guarantees the data is initialized.
        unsafe { std::slice::from_raw_parts_mut(row.as_mut_ptr() as *mut u8, row.len()) }
    }

    pub fn rect_mut(&'_ mut self, rect: Rect) -> RawImageRectMut<'_> {
        Self {
            // Safety note: we are lending ownership to the returned RawImageRectMut, and Rust's
            // type system ensures correctness.
            data: self.data.rect(rect),
            _ph: PhantomData,
        }
    }

    pub fn as_rect(&'_ self) -> RawImageRect<'_> {
        RawImageRect {
            // Safety note: correctness ensured by the return value borrowing from self.
            data: self.data,
            _ph: PhantomData,
        }
    }

    pub fn byte_size(&self) -> (usize, usize) {
        self.data.byte_size()
    }
}

impl Debug for OwnedRawImage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "raw {}x{}", self.byte_size().0, self.byte_size().1)
    }
}

impl Debug for RawImageRect<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "raw rect {}x{}", self.byte_size().0, self.byte_size().1)
    }
}

impl Debug for RawImageRectMut<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "raw mutrect {}x{}",
            self.byte_size().0,
            self.byte_size().1
        )
    }
}

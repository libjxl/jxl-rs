// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(unsafe_code)]

use std::{mem::MaybeUninit, ops::DerefMut};

/// Trait to represent image output buffers.
/// This trait is implemented for
/// - [u8], which will be interpreted as contiguous rows of data, starting from row 0 and
///   followed by row 1, 2, ... without any gaps)
/// - [&mut [u8]], which will be interpreted as buffers for distinct rows (starting from
///   the buffer for row 0, then row 1, ...).
pub trait JxlOutputBuffer<'a> {
    /// Returns buffers for each row of the image. Buffers might be uninitialized, but the caller
    /// guarantees that data that *is* initialized in the buffers is not overwritten with
    /// uninitialized data.
    ///
    /// # Safety
    /// The returned buffers must not be populated with uninit data.
    unsafe fn get_row_buffers(
        &'a mut self,
        shape: (usize, usize),
        bytes_per_pixel: usize,
    ) -> impl DerefMut<Target = [&'a mut [MaybeUninit<u8>]]>;
}

impl<'a> JxlOutputBuffer<'a> for [u8] {
    unsafe fn get_row_buffers(
        &'a mut self,
        shape: (usize, usize),
        bytes_per_pixel: usize,
    ) -> impl DerefMut<Target = [&'a mut [MaybeUninit<u8>]]> {
        // Safety: the caller promises to not write uninit data in the returned slices, and
        // MaybeUninit<T> is guaranteed to have the same layout as T.
        unsafe {
            std::mem::transmute::<&'a mut [u8], &'a mut [MaybeUninit<u8>]>(self)
                .get_row_buffers(shape, bytes_per_pixel)
        }
    }
}

impl<'a> JxlOutputBuffer<'a> for [MaybeUninit<u8>] {
    unsafe fn get_row_buffers(
        &'a mut self,
        shape: (usize, usize),
        bytes_per_pixel: usize,
    ) -> impl DerefMut<Target = [&'a mut [MaybeUninit<u8>]]> {
        assert!(self.len() >= shape.0 * shape.1 * bytes_per_pixel);
        let mut ret = vec![];
        let mut slc = self;
        for _ in 0..shape.1 {
            let (part, new_slc) = slc.split_at_mut(shape.0 * bytes_per_pixel);
            ret.push(part);
            slc = new_slc;
        }
        ret
    }
}

impl<'a> JxlOutputBuffer<'a> for [&'a mut [u8]] {
    unsafe fn get_row_buffers(
        &'a mut self,
        shape: (usize, usize),
        bytes_per_pixel: usize,
    ) -> impl DerefMut<Target = [&'a mut [MaybeUninit<u8>]]> {
        assert_eq!(self.len(), shape.1, "Incorrect number of rows provided");
        for row in self.iter() {
            assert!(
                row.len() >= shape.0 * bytes_per_pixel,
                "A row buffer is too short"
            );
        }
        // Safety: the caller promises to not write uninit data in the returned slices, and
        // MaybeUninit<T> is guaranteed to have the same layout as T.
        unsafe {
            std::mem::transmute::<&'a mut [&'a mut [u8]], &'a mut [&'a mut [MaybeUninit<u8>]]>(self)
        }
    }
}

impl<'a> JxlOutputBuffer<'a> for [&'a mut [MaybeUninit<u8>]] {
    unsafe fn get_row_buffers(
        &'a mut self,
        shape: (usize, usize),
        bytes_per_pixel: usize,
    ) -> impl DerefMut<Target = [&'a mut [MaybeUninit<u8>]]> {
        assert_eq!(self.len(), shape.1, "Incorrect number of rows provided");
        for row in self.iter() {
            assert!(
                row.len() >= shape.0 * bytes_per_pixel,
                "A row buffer is too short"
            );
        }
        self
    }
}

// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(unsafe_code)]

use core::slice;
use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut};

/// A fixed-capacity, stack-only vector. No heap fallback, no enum discriminant.
/// Unlike SmallVec, every operation avoids a Stack/Heap match branch.
/// Use when the maximum size is known at compile time and guaranteed to fit.
pub struct StackVec<T, const N: usize> {
    len: usize,
    data: [MaybeUninit<T>; N],
}

impl<T, const N: usize> StackVec<T, N> {
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            len: 0,
            data: [const { MaybeUninit::uninit() }; N],
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub fn push(&mut self, val: T) {
        debug_assert!(self.len < N, "StackVec overflow: len={}, cap={}", self.len, N);
        // SAFETY: we just checked len < N (in debug), and the caller must ensure capacity.
        unsafe {
            self.data.get_unchecked_mut(self.len).write(val);
        }
        self.len += 1;
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        // Drop existing elements
        for i in 0..self.len {
            // SAFETY: elements 0..len are initialized
            unsafe { self.data[i].assume_init_drop() };
        }
        self.len = 0;
    }

    #[inline(always)]
    pub fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for val in iter {
            self.push(val);
        }
    }
}

impl<T, const N: usize> Deref for StackVec<T, N> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &[T] {
        // SAFETY: the first `len` elements are initialized.
        unsafe { slice::from_raw_parts(self.data.as_ptr().cast::<T>(), self.len) }
    }
}

impl<T, const N: usize> DerefMut for StackVec<T, N> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: the first `len` elements are initialized.
        unsafe { slice::from_raw_parts_mut(self.data.as_mut_ptr().cast::<T>(), self.len) }
    }
}

impl<T, const N: usize> Drop for StackVec<T, N> {
    fn drop(&mut self) {
        for i in 0..self.len {
            // SAFETY: by invariant, elements 0..len are initialized.
            unsafe { self.data[i].assume_init_drop() };
        }
    }
}

impl<T, const N: usize> FromIterator<T> for StackVec<T, N> {
    #[inline(always)]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut ret = Self::new();
        ret.extend(iter);
        ret
    }
}

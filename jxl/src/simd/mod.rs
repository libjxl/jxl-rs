// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::{
    fmt::Debug,
    ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};

#[cfg(target_arch = "x86_64")]
mod x86_64;

mod scalar;

#[cfg(target_arch = "x86_64")]
pub(crate) use x86_64::{avx::AvxDescriptor, avx512::Avx512Descriptor, simd_function};

#[cfg(not(target_arch = "x86_64"))]
pub(crate) use scalar::simd_function;

#[cfg(all(test, target_arch = "x86_64"))]
pub(crate) use x86_64::test_all_instruction_sets;

#[cfg(all(test, not(target_arch = "x86_64")))]
pub(crate) use scalar::test_all_instruction_sets;

pub(crate) use scalar::ScalarDescriptor;

const CACHE_LINE_BYTE_SIZE: usize = 64;

pub fn round_up_size_to_two_cache_lines<T>(size: usize) -> usize {
    let elements_per_cache_line = CACHE_LINE_BYTE_SIZE / std::mem::size_of::<T>() * 2;
    size.div_ceil(elements_per_cache_line) * elements_per_cache_line
}

pub trait SimdDescriptor: Sized + Copy + Debug + Send + Sync {
    type F32Vec: F32SimdVec<Descriptor = Self>;

    fn new() -> Option<Self>;
}

pub trait F32SimdVec:
    Sized
    + Copy
    + Debug
    + Send
    + Sync
    + Add<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Sub<Self, Output = Self>
    + AddAssign<Self>
    + MulAssign<Self>
    + SubAssign<Self>
{
    type Descriptor: SimdDescriptor;

    const LEN: usize;

    /// Converts v to an array of v.
    fn splat(d: Self::Descriptor, v: f32) -> Self;

    fn mul_add(self, mul: Self, add: Self) -> Self;

    // Requires `mem.len() >= Self::LEN` or it will panic.
    fn load(d: Self::Descriptor, mem: &[f32]) -> Self;

    // Requires `mem.len() >= Self::LEN` or it will panic.
    fn store(&self, mem: &mut [f32]);
}

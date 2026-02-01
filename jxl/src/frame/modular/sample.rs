// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Sample type abstraction for modular decoding.
//!
//! This module provides the `ModularSample` trait which abstracts over the sample
//! type used in modular decoding. Currently supports i16 and i32, allowing 50%
//! memory bandwidth reduction for 8-16 bit images.

use crate::image::ImageDataType;

/// Trait for sample types used in modular decoding.
///
/// This trait extends `ImageDataType` with arithmetic operations needed for
/// prediction and decoding. Implementations must provide wrapping arithmetic
/// and conversion to/from i64 for prediction calculations.
pub trait ModularSample:
    ImageDataType + std::ops::Add<Output = Self> + std::ops::Sub<Output = Self>
{
    /// Zero value
    const ZERO: Self;

    /// Minimum representable value
    const MIN: Self;

    /// Maximum representable value
    const MAX: Self;

    /// Convert from i64 (wrapping for i32, saturating for i16)
    fn from_i64(v: i64) -> Self;

    /// Convert to i64
    fn to_i64(self) -> i64;

    /// Wrapping addition
    fn wrapping_add(self, other: Self) -> Self;

    /// Wrapping subtraction
    fn wrapping_sub(self, other: Self) -> Self;

    /// Wrapping absolute value
    fn wrapping_abs(self) -> Self;

    /// Convert from i32 (for decoded residual values)
    fn from_i32(v: i32) -> Self;

    /// Convert to i32
    fn to_i32(self) -> i32;
}

impl ModularSample for i32 {
    const ZERO: Self = 0;
    const MIN: Self = i32::MIN;
    const MAX: Self = i32::MAX;

    #[inline(always)]
    fn from_i64(v: i64) -> Self {
        // Direct cast (wrapping) - matches original behavior
        v as i32
    }

    #[inline(always)]
    fn to_i64(self) -> i64 {
        self as i64
    }

    #[inline(always)]
    fn wrapping_add(self, other: Self) -> Self {
        i32::wrapping_add(self, other)
    }

    #[inline(always)]
    fn wrapping_sub(self, other: Self) -> Self {
        i32::wrapping_sub(self, other)
    }

    #[inline(always)]
    fn wrapping_abs(self) -> Self {
        i32::wrapping_abs(self)
    }

    #[inline(always)]
    fn from_i32(v: i32) -> Self {
        v
    }

    #[inline(always)]
    fn to_i32(self) -> i32 {
        self
    }
}

impl ModularSample for i16 {
    const ZERO: Self = 0;
    const MIN: Self = i16::MIN;
    const MAX: Self = i16::MAX;

    #[inline(always)]
    fn from_i64(v: i64) -> Self {
        // Saturating for i16 to prevent overflow for values outside range
        v.clamp(i16::MIN as i64, i16::MAX as i64) as i16
    }

    #[inline(always)]
    fn to_i64(self) -> i64 {
        self as i64
    }

    #[inline(always)]
    fn wrapping_add(self, other: Self) -> Self {
        i16::wrapping_add(self, other)
    }

    #[inline(always)]
    fn wrapping_sub(self, other: Self) -> Self {
        i16::wrapping_sub(self, other)
    }

    #[inline(always)]
    fn wrapping_abs(self) -> Self {
        i16::wrapping_abs(self)
    }

    #[inline(always)]
    fn from_i32(v: i32) -> Self {
        v.clamp(i16::MIN as i32, i16::MAX as i32) as i16
    }

    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_i32_sample() {
        assert_eq!(i32::from_i64(100), 100i32);
        // i32 uses direct cast (wrapping), so large values wrap
        assert_eq!(i32::from_i64(i32::MAX as i64), i32::MAX);
        assert_eq!(i32::from_i64(i32::MIN as i64), i32::MIN);
        assert_eq!(5i32.wrapping_add(3), 8);
        assert_eq!(5i32.wrapping_sub(3), 2);
        assert_eq!((-5i32).wrapping_abs(), 5);
    }

    #[test]
    fn test_i16_sample() {
        assert_eq!(i16::from_i64(100), 100i16);
        // i16 uses saturating conversion
        assert_eq!(i16::from_i64(i64::MAX), i16::MAX);
        assert_eq!(i16::from_i64(i64::MIN), i16::MIN);
        assert_eq!(i16::from_i32(40000), i16::MAX);
        assert_eq!(i16::from_i32(-40000), i16::MIN);
        assert_eq!(5i16.wrapping_add(3), 8);
        assert_eq!(5i16.wrapping_sub(3), 2);
        assert_eq!((-5i16).wrapping_abs(), 5);
    }
}

// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! IEEE 754 half-precision (f16) floating-point support.

/// A 16-bit floating-point type (IEEE 754-2008 binary16).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct F16(u16);

impl F16 {
    /// Creates an f16 from its raw bit representation.
    #[inline(always)]
    pub fn from_bits(bits: u16) -> Self {
        Self(bits)
    }

    /// Returns the raw bit representation of this f16.
    #[inline(always)]
    pub fn to_bits(self) -> u16 {
        self.0
    }

    /// Converts an f32 to f16.
    #[inline]
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let exp = ((bits >> 23) & 0xff) as i32;
        let mantissa = bits & 0x7fffff;

        let h = if exp == 255 {
            // Inf or NaN
            if mantissa == 0 {
                sign | 0x7c00 // Inf
            } else {
                sign | 0x7c00 | (mantissa >> 13) | 1 // NaN (ensure mantissa != 0)
            }
        } else if exp > 142 {
            // Overflow to infinity
            sign | 0x7c00
        } else if exp < 103 {
            // Underflow to zero
            sign
        } else if exp < 113 {
            // Subnormal
            let m = (mantissa | 0x800000) >> (126 - exp);
            sign | (m >> 13)
        } else {
            // Normal
            sign | (((exp - 112) as u32) << 10) | (mantissa >> 13)
        };
        Self(h as u16)
    }

    /// Converts this f16 to f32.
    #[inline]
    pub fn to_f32(self) -> f32 {
        let h = self.0 as u32;
        let sign = (h & 0x8000) << 16;
        let exp = (h >> 10) & 0x1f;
        let mantissa = h & 0x3ff;

        let f = if exp == 0 {
            if mantissa == 0 {
                // Zero
                sign
            } else {
                // Subnormal - normalize it
                let mut m = mantissa;
                let mut e = 113i32;
                while (m & 0x400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                m &= 0x3ff;
                sign | ((e as u32) << 23) | (m << 13)
            }
        } else if exp == 31 {
            // Inf or NaN
            sign | 0x7f800000 | (mantissa << 13)
        } else {
            // Normal
            sign | ((exp + 112) << 23) | (mantissa << 13)
        };
        f32::from_bits(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero() {
        assert_eq!(F16::from_f32(0.0).to_f32(), 0.0);
        assert_eq!(F16::from_f32(-0.0).to_f32().to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn test_one() {
        assert_eq!(F16::from_f32(1.0).to_f32(), 1.0);
        assert_eq!(F16::from_f32(-1.0).to_f32(), -1.0);
    }

    #[test]
    fn test_infinity() {
        assert!(F16::from_f32(f32::INFINITY).to_f32().is_infinite());
        assert!(F16::from_f32(f32::NEG_INFINITY).to_f32().is_infinite());
    }

    #[test]
    fn test_nan() {
        assert!(F16::from_f32(f32::NAN).to_f32().is_nan());
    }

    #[test]
    fn test_roundtrip() {
        // Test some common values
        for &v in &[0.5, 0.25, 2.0, 100.0, 0.001] {
            let h = F16::from_f32(v);
            let back = h.to_f32();
            assert!((back - v).abs() / v < 0.001, "failed for {v}: got {back}");
        }
    }
}

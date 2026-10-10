// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! IEEE 754 half-precision (binary16) floating-point type.

pub use jxl_simd::f16;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero() {
        let z = f16::ZERO;
        assert_eq!(z.to_bits(), 0);
        assert_eq!(z.to_f32(), 0.0);
        assert!(z.is_finite());
    }

    #[test]
    fn test_one() {
        // 1.0 in f16: sign=0, exp=15 (biased), mant=0 -> 0x3C00
        let one = f16::from_bits(0x3C00);
        assert!((one.to_f32() - 1.0).abs() < 1e-6);
        assert!(one.is_finite());
    }

    #[test]
    fn test_negative_one() {
        // -1.0 in f16: sign=1, exp=15, mant=0 -> 0xBC00
        let neg_one = f16::from_bits(0xBC00);
        assert!((neg_one.to_f32() - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_infinity() {
        // +Inf: sign=0, exp=31, mant=0 -> 0x7C00
        let inf = f16::from_bits(0x7C00);
        assert!(inf.to_f32().is_infinite());
        assert!(!inf.is_finite());

        // -Inf: 0xFC00
        let neg_inf = f16::from_bits(0xFC00);
        assert!(neg_inf.to_f32().is_infinite());
        assert!(!neg_inf.is_finite());
    }

    #[test]
    fn test_nan() {
        // NaN: exp=31, mant!=0 -> 0x7C01 (or any mant != 0)
        let nan = f16::from_bits(0x7C01);
        assert!(nan.to_f32().is_nan());
        assert!(!nan.is_finite());
    }

    #[test]
    fn test_denormal() {
        // Smallest positive denormal: 0x0001
        let tiny = f16::from_bits(0x0001);
        let val = tiny.to_f32();
        assert!(val > 0.0);
        assert!(val < 1e-6);
        assert!(tiny.is_finite());
    }

    #[test]
    fn test_denormal_values() {
        // A denormal f16 is mantissa * 2^-24.
        for bits in 1u16..0x0400 {
            let expected = bits as f32 * 2f32.powi(-24);
            assert_eq!(f16::from_bits(bits).to_f32(), expected, "bits {bits:#06x}");
            assert_eq!(f16::from_bits(bits | 0x8000).to_f32(), -expected);
            assert_eq!(
                f16::from_f32(expected).to_bits(),
                bits,
                "value {expected:e}"
            );
            assert_eq!(f16::from_f32(-expected).to_bits(), bits | 0x8000);
        }
        // Largest denormal and smallest normal.
        assert_eq!(f16::from_bits(0x03FF).to_f32(), 1023.0 * 2f32.powi(-24));
        assert_eq!(f16::from_bits(0x0400).to_f32(), 2f32.powi(-14));
    }

    #[test]
    fn test_denormal_rounding() {
        let ulp = 2f32.powi(-24);
        // Ties to even.
        assert_eq!(f16::from_f32(1.5 * ulp).to_bits(), 0x0002);
        assert_eq!(f16::from_f32(2.5 * ulp).to_bits(), 0x0002);
        assert_eq!(f16::from_f32(2.75 * ulp).to_bits(), 0x0003);
        // Rounds up into the smallest normal.
        assert_eq!(f16::from_f32(1023.75 * ulp).to_bits(), 0x0400);
    }

    #[test]
    fn test_roundtrip_normal() {
        let test_values: [f32; 8] = [0.5, 1.0, 2.0, 100.0, 0.001, -0.5, -1.0, -100.0];
        for &v in &test_values {
            let h = f16::from_f32(v);
            let back = h.to_f32();
            // f16 has limited precision, allow ~0.1% error for normal values
            let rel_err = ((v - back) / v).abs();
            assert!(
                rel_err < 0.002,
                "Roundtrip failed for {}: got {}, rel_err {}",
                v,
                back,
                rel_err
            );
        }
    }

    #[test]
    fn test_roundtrip_special() {
        // Zero
        assert_eq!(f16::from_f32(0.0).to_f32(), 0.0);

        // Infinity
        assert!(f16::from_f32(f32::INFINITY).to_f32().is_infinite());
        assert!(f16::from_f32(f32::NEG_INFINITY).to_f32().is_infinite());

        // NaN
        assert!(f16::from_f32(f32::NAN).to_f32().is_nan());
    }

    #[test]
    fn test_overflow_to_infinity() {
        // f16 max is ~65504, values above should overflow to infinity
        let big = f16::from_f32(100000.0);
        assert!(big.to_f32().is_infinite());
    }

    #[test]
    fn test_underflow_to_zero() {
        // Very small values should underflow to zero
        let tiny = f16::from_f32(1e-10);
        assert_eq!(tiny.to_f32(), 0.0);
    }

    #[test]
    fn test_bytes() {
        let h = f16::from_bits(0x1234);
        assert_eq!(h.to_le_bytes(), [0x34, 0x12]);
        assert_eq!(h.to_be_bytes(), [0x12, 0x34]);
    }
}

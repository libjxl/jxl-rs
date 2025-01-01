// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

const SRGB_POWTABLE_UPPER: [u8; 16] = [
    0x00, 0x0a, 0x19, 0x26, 0x32, 0x41, 0x4d, 0x5c, 0x68, 0x75, 0x83, 0x8f, 0xa0, 0xaa, 0xb9, 0xc6,
];

const SRGB_POWTABLE_LOWER: [u8; 16] = [
    0x00, 0xb7, 0x04, 0x0d, 0xcb, 0xe7, 0x41, 0x68, 0x51, 0xd1, 0xeb, 0xf2, 0x00, 0xb7, 0x04, 0x0d,
];

/// Converts the linear samples with the sRGB transfer curve.
// Fast linear to sRGB conversion, ported from libjxl. Max error ~1.7e-4
pub fn linear_to_srgb_fast(samples: &mut [f32]) {
    for s in samples {
        let v = s.to_bits() & 0x7fff_ffff;
        let v_adj = f32::from_bits((v | 0x3e80_0000) & 0x3eff_ffff);
        let pow = 0.059914046f32;
        let pow = pow * v_adj - 0.10889456;
        let pow = pow * v_adj + 0.107963754;
        let pow = pow * v_adj + 0.018092343;

        // `mul` won't be used when `v` is small.
        let idx = (v >> 23).wrapping_sub(118) as usize & 0xf;
        let mul = 0x4000_0000
            | (u32::from(SRGB_POWTABLE_UPPER[idx]) << 18)
            | (u32::from(SRGB_POWTABLE_LOWER[idx]) << 10);

        let v = f32::from_bits(v);
        let small = v * 12.92;
        let acc = pow * f32::from_bits(mul) - 0.055;

        *s = if v <= 0.0031308 { small } else { acc }.copysign(*s);
    }
}

/// Converts the linear samples with the sRGB transfer curve.
// Max error ~5e-7
pub fn linear_to_srgb(samples: &mut [f32]) {
    #[allow(clippy::excessive_precision)]
    const P: [f32; 5] = [
        -5.135152395e-4,
        5.287254571e-3,
        3.903842876e-1,
        1.474205315,
        7.352629620e-1,
    ];

    #[allow(clippy::excessive_precision)]
    const Q: [f32; 5] = [
        1.004519624e-2,
        3.036675394e-1,
        1.340816930,
        9.258482155e-1,
        2.424867759e-2,
    ];

    for x in samples {
        let a = x.abs();
        *x = if a <= 0.0031308 {
            a * 12.92
        } else {
            crate::util::eval_rational_poly(a.sqrt(), P, Q)
        }
        .copysign(*x);
    }
}

/// Converts samples in sRGB transfer curve to linear. Inverse of `linear_to_srgb`.
pub fn srgb_to_linear(samples: &mut [f32]) {
    #[allow(clippy::excessive_precision)]
    const P: [f32; 5] = [
        2.200248328e-4,
        1.043637593e-2,
        1.624820318e-1,
        7.961564959e-1,
        8.210152774e-1,
    ];

    #[allow(clippy::excessive_precision)]
    const Q: [f32; 5] = [
        2.631846970e-1,
        1.076976492,
        4.987528350e-1,
        -5.512498495e-2,
        6.521209011e-3,
    ];

    for x in samples {
        let a = x.abs();
        *x = if a <= 0.04045 {
            a / 12.92
        } else {
            crate::util::eval_rational_poly(a, P, Q)
        }
        .copysign(*x);
    }
}

#[cfg(test)]
mod test {
    use test_log::test;

    use super::*;
    use crate::util::test::assert_all_almost_eq;

    fn arb_samples(
        u: &mut arbtest::arbitrary::Unstructured,
    ) -> arbtest::arbitrary::Result<Vec<f32>> {
        const DENOM: u32 = 1 << 24;

        let len = u.arbitrary_len::<u32>()?;
        let mut samples = Vec::with_capacity(len);

        // uniform distribution in [-1.0, 1.0]
        for _ in 0..len {
            let a: u32 = u.int_in_range(0..=DENOM)?;
            let signed: bool = u.arbitrary()?;
            let x = a as f32 / DENOM as f32;
            samples.push(if signed { -x } else { x });
        }

        Ok(samples)
    }

    #[test]
    fn srgb_roundtrip_arb() {
        arbtest::arbtest(|u| {
            let samples: Vec<f32> = arb_samples(u)?;
            let mut output = samples.clone();

            linear_to_srgb(&mut output);
            srgb_to_linear(&mut output);
            assert_all_almost_eq!(&output, &samples, 2e-6);
            Ok(())
        });
    }

    #[test]
    fn linear_to_srgb_fast_arb() {
        arbtest::arbtest(|u| {
            let mut samples: Vec<f32> = arb_samples(u)?;
            let mut fast = samples.clone();

            linear_to_srgb(&mut samples);
            linear_to_srgb_fast(&mut fast);
            assert_all_almost_eq!(&samples, &fast, 1.7e-4);
            Ok(())
        });
    }
}

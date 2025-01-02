// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(clippy::excessive_precision)]

use super::eval_rational_poly;

const POW2F_NUMER_COEFFS: [f32; 3] = [1.01749063e1, 4.88687798e1, 9.85506591e1];
const POW2F_DENOM_COEFFS: [f32; 4] = [2.10242958e-1, -2.22328856e-2, -1.94414990e1, 9.85506633e1];

#[inline]
fn fast_pow2f(x: f32) -> f32 {
    let x_floor = x.floor();
    let exp = f32::from_bits(((x_floor as i32 + 127) as u32) << 23);
    let frac = x - x_floor;

    let num = frac + POW2F_NUMER_COEFFS[0];
    let num = num * frac + POW2F_NUMER_COEFFS[1];
    let num = num * frac + POW2F_NUMER_COEFFS[2];
    let num = num * exp;

    let den = POW2F_DENOM_COEFFS[0] * frac + POW2F_DENOM_COEFFS[1];
    let den = den * frac + POW2F_DENOM_COEFFS[2];
    let den = den * frac + POW2F_DENOM_COEFFS[3];

    num / den
}

const LOG2F_P: [f32; 3] = [
    -1.8503833400518310e-6,
    1.4287160470083755,
    7.4245873327820566e-1,
];
const LOG2F_Q: [f32; 3] = [
    9.9032814277590719e-1,
    1.0096718572241148,
    1.7409343003366853e-1,
];

#[inline]
fn fast_log2f(x: f32) -> f32 {
    let x_bits = x.to_bits() as i32;
    let exp_bits = x_bits - 0x3f2aaaab;
    let exp_shifted = exp_bits >> 23;
    let mantissa = f32::from_bits((x_bits - (exp_shifted << 23)) as u32);
    let exp_val = exp_shifted as f32;

    let x = mantissa - 1.0;
    eval_rational_poly(x, LOG2F_P, LOG2F_Q) + exp_val
}

// Max relative error: ~3e-5
#[inline]
pub fn fast_powf(base: f32, exp: f32) -> f32 {
    fast_pow2f(fast_log2f(base) * exp)
}

#[cfg(test)]
mod test {
    use test_log::test;

    use super::*;

    #[test]
    fn fast_powf_arb() {
        arbtest::arbtest(|u| {
            // (0.0, 128.0]
            let base = u.int_in_range(1..=1 << 24)? as f32 / (1 << 17) as f32;
            // [-8.0, 8.0]
            let exp = u.int_in_range(-(1i32 << 23)..=1 << 23)? as f32 / (1 << 20) as f32;

            let expected = base.powf(exp);
            let actual = fast_powf(base, exp);
            let abs_error = (actual - expected).abs();
            let rel_error = abs_error / expected;
            assert!(rel_error < 3e-5);
            Ok(())
        });
    }
}

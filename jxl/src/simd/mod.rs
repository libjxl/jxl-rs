// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::{
    fmt::Debug,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
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

pub const CACHE_LINE_BYTE_SIZE: usize = 64;

pub const fn num_per_cache_line<T>() -> usize {
    // Post-mono check that T is smaller than a cache line and has size a power of 2.
    // This prevents some of the silliest mistakes.
    const {
        assert!(std::mem::size_of::<T>() <= CACHE_LINE_BYTE_SIZE);
        assert!(std::mem::size_of::<T>().is_power_of_two());
    }
    CACHE_LINE_BYTE_SIZE / std::mem::size_of::<T>()
}

pub fn round_up_size_to_two_cache_lines<T>(size: usize) -> usize {
    let n = const { num_per_cache_line::<T>() * 2 };
    size.div_ceil(n) * n
}

pub trait SimdDescriptor: Sized + Copy + Debug + Send + Sync {
    type F32Vec: F32SimdVec<Descriptor = Self>;

    fn new() -> Option<Self>;

    fn transpose<const ROWS: usize, const COLS: usize>(self, input: &[f32], output: &mut [f32]);

    /// Calls the given closure within a target feature context.
    /// This enables establishing an unbroken chain of inline functions from the feature-annotated
    /// gateway up to the closure, allowing SIMD intrinsics to be used safely.
    fn call<R>(self, f: impl FnOnce(Self) -> R) -> R;
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
    + Div<Self, Output = Self>
    + AddAssign<Self>
    + MulAssign<Self>
    + SubAssign<Self>
    + DivAssign<Self>
{
    type Descriptor: SimdDescriptor;

    const LEN: usize;

    /// Converts v to an array of v.
    fn splat(d: Self::Descriptor, v: f32) -> Self;

    fn mul_add(self, mul: Self, add: Self) -> Self;

    /// Computes `add - self * mul`, equivalent to `self * (-mul) + add`.
    /// Uses fused multiply-add with negation when available (FMA3 fnmadd).
    fn neg_mul_add(self, mul: Self, add: Self) -> Self;

    // Requires `mem.len() >= Self::LEN` or it will panic.
    fn load(d: Self::Descriptor, mem: &[f32]) -> Self;

    // Requires `mem.len() >= SIZE` or it will panic.
    fn load_partial(d: Self::Descriptor, size: usize, mem: &[f32]) -> Self;

    // Requires `mem.len() >= Self::LEN` or it will panic.
    fn store(&self, mem: &mut [f32]);

    // Requires `mem.len() >= SIZE` or it will panic.
    fn store_partial(&self, size: usize, mem: &mut [f32]);

    fn abs(self) -> Self;

    /// Negates all elements. Currently unused but kept for API completeness.
    #[allow(dead_code)]
    fn neg(self) -> Self;

    fn max(self, other: Self) -> Self;
}

#[cfg(test)]
mod test {
    use arbtest::arbitrary::Unstructured;

    use crate::{
        simd::{F32SimdVec, ScalarDescriptor, SimdDescriptor, test_all_instruction_sets},
        util::test::assert_all_almost_rel_eq,
    };

    enum Distribution {
        Floats,
        NonZeroFloats,
    }

    fn arb_vec<D: SimdDescriptor>(_: D, u: &mut Unstructured, dist: Distribution) -> Vec<f32> {
        let mut res = vec![0.0; D::F32Vec::LEN];
        for v in res.iter_mut() {
            match dist {
                Distribution::Floats => {
                    *v = u.arbitrary::<i32>().unwrap() as f32
                        / (1.0 + u.arbitrary::<u32>().unwrap() as f32)
                }
                Distribution::NonZeroFloats => {
                    let sign = if u.arbitrary::<bool>().unwrap() {
                        1.0
                    } else {
                        -1.0
                    };
                    *v = sign * (1.0 + u.arbitrary::<u32>().unwrap() as f32)
                        / (1.0 + u.arbitrary::<u32>().unwrap() as f32);
                }
            }
        }
        res
    }

    macro_rules! test_instruction {
        ($name:ident, |$a:ident: $a_dist:ident| $block:expr) => {
            fn $name<D: SimdDescriptor>(d: D) {
                fn compute<D: SimdDescriptor>(d: D, a: &[f32]) -> Vec<f32> {
                    let closure = |$a: D::F32Vec| $block;
                    let mut res = vec![0f32; a.len()];
                    for idx in (0..a.len()).step_by(D::F32Vec::LEN) {
                        closure(D::F32Vec::load(d, &a[idx..])).store(&mut res[idx..]);
                    }
                    res
                }
                arbtest::arbtest(|u| {
                    let a = arb_vec(d, u, Distribution::$a_dist);
                    let scalar_res = compute(ScalarDescriptor::new().unwrap(), &a);
                    let simd_res = compute(d, &a);
                    assert_all_almost_rel_eq(&scalar_res, &simd_res, 1e-8);
                    Ok(())
                })
                .size_min(64);
            }
            test_all_instruction_sets!($name);
        };
        ($name:ident, |$a:ident: $a_dist:ident, $b:ident: $b_dist:ident| $block:expr) => {
            fn $name<D: SimdDescriptor>(d: D) {
                fn compute<D: SimdDescriptor>(d: D, a: &[f32], b: &[f32]) -> Vec<f32> {
                    let closure = |$a: D::F32Vec, $b: D::F32Vec| $block;
                    let mut res = vec![0f32; a.len()];
                    for idx in (0..a.len()).step_by(D::F32Vec::LEN) {
                        closure(D::F32Vec::load(d, &a[idx..]), D::F32Vec::load(d, &b[idx..]))
                            .store(&mut res[idx..]);
                    }
                    res
                }
                arbtest::arbtest(|u| {
                    let a = arb_vec(d, u, Distribution::$a_dist);
                    let b = arb_vec(d, u, Distribution::$b_dist);
                    let scalar_res = compute(ScalarDescriptor::new().unwrap(), &a, &b);
                    let simd_res = compute(d, &a, &b);
                    assert_all_almost_rel_eq(&scalar_res, &simd_res, 1e-8);
                    Ok(())
                })
                .size_min(128);
            }
            test_all_instruction_sets!($name);
        };
        ($name:ident, |$a:ident: $a_dist:ident, $b:ident: $b_dist:ident, $c:ident: $c_dist:ident| $block:expr) => {
            fn $name<D: SimdDescriptor>(d: D) {
                fn compute<D: SimdDescriptor>(d: D, a: &[f32], b: &[f32], c: &[f32]) -> Vec<f32> {
                    let closure = |$a: D::F32Vec, $b: D::F32Vec, $c: D::F32Vec| $block;
                    let mut res = vec![0f32; a.len()];
                    for idx in (0..a.len()).step_by(D::F32Vec::LEN) {
                        closure(
                            D::F32Vec::load(d, &a[idx..]),
                            D::F32Vec::load(d, &b[idx..]),
                            D::F32Vec::load(d, &c[idx..]),
                        )
                        .store(&mut res[idx..]);
                    }
                    res
                }
                arbtest::arbtest(|u| {
                    let a = arb_vec(d, u, Distribution::$a_dist);
                    let b = arb_vec(d, u, Distribution::$b_dist);
                    let c = arb_vec(d, u, Distribution::$c_dist);
                    let scalar_res = compute(ScalarDescriptor::new().unwrap(), &a, &b, &c);
                    let simd_res = compute(d, &a, &b, &c);
                    assert_all_almost_rel_eq(&scalar_res, &simd_res, 1e-8);
                    Ok(())
                })
                .size_min(172);
            }
            test_all_instruction_sets!($name);
        };
    }

    test_instruction!(add, |a: Floats, b: Floats| { a + b });
    test_instruction!(mul, |a: Floats, b: Floats| { a * b });
    test_instruction!(sub, |a: Floats, b: Floats| { a - b });
    test_instruction!(div, |a: Floats, b: NonZeroFloats| { a / b });

    test_instruction!(add_assign, |a: Floats, b: Floats| {
        let mut res = a;
        res += b;
        res
    });
    test_instruction!(mul_assign, |a: Floats, b: Floats| {
        let mut res = a;
        res *= b;
        res
    });
    test_instruction!(sub_assign, |a: Floats, b: Floats| {
        let mut res = a;
        res -= b;
        res
    });
    test_instruction!(div_assign, |a: Floats, b: NonZeroFloats| {
        let mut res = a;
        res /= b;
        res
    });

    test_instruction!(mul_add, |a: Floats, b: Floats, c: Floats| {
        a.mul_add(b, c)
    });

    test_instruction!(neg_mul_add, |a: Floats, b: Floats, c: Floats| {
        a.neg_mul_add(b, c)
    });

    // Validate that neg_mul_add computes c - a * b correctly
    fn test_neg_mul_add_correctness<D: SimdDescriptor>(d: D) {
        let a_vals = [
            2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 2.5, 3.5, 4.5, 5.5, 1.0, 2.0, 3.0, 4.0,
        ];
        let b_vals = [
            1.0, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5, 1.5, 2.5, 3.5, 4.5, 0.25, 0.75, 1.25, 1.75,
        ];
        let c_vals = [
            10.0, 20.0, 30.0, 40.0, 5.0, 15.0, 25.0, 35.0, 12.0, 22.0, 32.0, 42.0, 6.0, 16.0, 26.0,
            36.0,
        ];

        let a = D::F32Vec::load(d, &a_vals[..D::F32Vec::LEN]);
        let b = D::F32Vec::load(d, &b_vals[..D::F32Vec::LEN]);
        let c = D::F32Vec::load(d, &c_vals[..D::F32Vec::LEN]);

        let result = a.neg_mul_add(b, c);
        let expected = c - a * b;

        let mut result_vals = [0.0; 16];
        let mut expected_vals = [0.0; 16];
        result.store(&mut result_vals[..D::F32Vec::LEN]);
        expected.store(&mut expected_vals[..D::F32Vec::LEN]);

        for i in 0..D::F32Vec::LEN {
            assert!(
                (result_vals[i] - expected_vals[i]).abs() < 1e-5,
                "neg_mul_add correctness failed at index {}: got {}, expected {}",
                i,
                result_vals[i],
                expected_vals[i]
            );
        }
    }

    test_all_instruction_sets!(test_neg_mul_add_correctness);

    test_instruction!(abs, |a: Floats| { a.abs() });
    test_instruction!(max, |a: Floats, b: Floats| { a.max(b) });

    // Test that the call method works, compiles, and can capture arguments
    fn test_call<D: SimdDescriptor>(d: D) {
        // Test basic call functionality
        let result = d.call(|_d| 42);
        assert_eq!(result, 42);

        // Test with capturing variables
        let multiplier = 3.0f32;
        let addend = 5.0f32;

        // Test SIMD operations inside call with captures
        let input = vec![1.0f32; D::F32Vec::LEN * 4];
        let mut output = vec![0.0f32; D::F32Vec::LEN * 4];

        d.call(|d| {
            let mult_vec = D::F32Vec::splat(d, multiplier);
            let add_vec = D::F32Vec::splat(d, addend);

            for idx in (0..input.len()).step_by(D::F32Vec::LEN) {
                let vec = D::F32Vec::load(d, &input[idx..]);
                let result = vec * mult_vec + add_vec;
                result.store(&mut output[idx..]);
            }
        });

        // Verify results
        for &val in &output {
            assert_eq!(val, 1.0 * multiplier + addend);
        }
    }
    test_all_instruction_sets!(test_call);

    fn test_neg<D: SimdDescriptor>(d: D) {
        // Test negation operation with enough elements for any SIMD size
        let len = D::F32Vec::LEN * 2; // Ensure we have at least 2 full vectors
        let input: Vec<f32> = (0..len)
            .map(|i| if i % 2 == 0 { i as f32 } else { -(i as f32) })
            .collect();
        let expected: Vec<f32> = (0..len)
            .map(|i| if i % 2 == 0 { -(i as f32) } else { i as f32 })
            .collect();
        let mut output = vec![0.0f32; input.len()];

        for idx in (0..input.len()).step_by(D::F32Vec::LEN) {
            let vec = D::F32Vec::load(d, &input[idx..]);
            let negated = vec.neg();
            negated.store(&mut output[idx..]);
        }

        for (i, (&out, &exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                out, exp,
                "Mismatch at index {}: expected {}, got {}",
                i, exp, out
            );
        }
    }
    test_all_instruction_sets!(test_neg);

    fn test_load_store_partial<D: SimdDescriptor>(d: D) {
        // Test partial load/store operations with various sizes
        for size in 1..=D::F32Vec::LEN {
            let input: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let mut output = vec![99.0f32; D::F32Vec::LEN]; // Fill with sentinel value

            let vec = D::F32Vec::load_partial(d, size, &input);
            vec.store_partial(size, &mut output);

            // Verify that the first 'size' elements match
            for i in 0..size {
                assert_eq!(
                    output[i], input[i],
                    "Mismatch at index {} (size={}): expected {}, got {}",
                    i, size, input[i], output[i]
                );
            }

            // Verify that elements beyond 'size' are unchanged (still sentinel)
            for (idx, &val) in output
                .iter()
                .enumerate()
                .skip(size)
                .take(D::F32Vec::LEN - size)
            {
                assert_eq!(
                    val, 99.0,
                    "Element at index {} was modified (size={})",
                    idx, size
                );
            }
        }
    }
    test_all_instruction_sets!(test_load_store_partial);

    fn test_transpose_8x8<D: SimdDescriptor>(d: D) {
        // Test 8x8 matrix transpose
        // Input: sequential values 0..64
        let mut input = vec![0.0f32; 64];
        for (i, val) in input.iter_mut().enumerate() {
            *val = i as f32;
        }

        let mut output = vec![0.0f32; 64];
        d.transpose::<8, 8>(&input, &mut output);

        // Verify transpose: output[i*8+j] should equal input[j*8+i]
        for i in 0..8 {
            for j in 0..8 {
                let expected = input[j * 8 + i];
                let actual = output[i * 8 + j];
                assert_eq!(
                    actual, expected,
                    "Mismatch at position ({}, {}): expected {}, got {}",
                    i, j, expected, actual
                );
            }
        }
    }
    test_all_instruction_sets!(test_transpose_8x8);
}

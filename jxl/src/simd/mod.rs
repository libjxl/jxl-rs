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

    // Requires `mem.len() >= Self::LEN` or it will panic.
    fn load(d: Self::Descriptor, mem: &[f32]) -> Self;

    // Requires `mem.len() >= Self::LEN` or it will panic.
    fn store(&self, mem: &mut [f32]);

    fn abs(self) -> Self;

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

    test_instruction!(abs, |a: Floats| { a.abs() });
    test_instruction!(max, |a: Floats, b: Floats| { a.max(b) });
}

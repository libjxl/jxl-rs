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
    use crate::{
        simd::{
            F32SimdVec, ScalarDescriptor, SimdDescriptor, round_up_size_to_two_cache_lines,
            test_all_instruction_sets,
        },
        util::test::assert_all_almost_eq,
    };

    fn vec_add<D: SimdDescriptor>(d: D, a: &[f32], b: &[f32]) -> Vec<f32> {
        assert!(a.len() == b.len());
        let mut res = vec![0f32; a.len()];
        for idx in (0..a.len()).step_by(D::F32Vec::LEN) {
            (D::F32Vec::load(d, &a[idx..]) + D::F32Vec::load(d, &b[idx..])).store(&mut res[idx..]);
        }
        res
    }

    fn vec_sub<D: SimdDescriptor>(d: D, a: &[f32], b: &[f32]) -> Vec<f32> {
        assert!(a.len() == b.len());
        let mut res = vec![0f32; a.len()];
        for idx in (0..a.len()).step_by(D::F32Vec::LEN) {
            (D::F32Vec::load(d, &a[idx..]) - D::F32Vec::load(d, &b[idx..])).store(&mut res[idx..]);
        }
        res
    }

    fn vec_mul<D: SimdDescriptor>(d: D, a: &[f32], b: &[f32]) -> Vec<f32> {
        assert!(a.len() == b.len());
        let mut res = vec![0f32; a.len()];
        for idx in (0..a.len()).step_by(D::F32Vec::LEN) {
            (D::F32Vec::load(d, &a[idx..]) * D::F32Vec::load(d, &b[idx..])).store(&mut res[idx..]);
        }
        res
    }

    fn vec_add_assign<D: SimdDescriptor>(d: D, a: &[f32], b: &[f32]) -> Vec<f32> {
        assert!(a.len() == b.len());
        let mut res = vec![0f32; a.len()];
        for idx in (0..a.len()).step_by(D::F32Vec::LEN) {
            let mut a_vec = D::F32Vec::load(d, &a[idx..]);
            a_vec += D::F32Vec::load(d, &b[idx..]);
            a_vec.store(&mut res[idx..]);
        }
        res
    }

    fn vec_sub_assign<D: SimdDescriptor>(d: D, a: &[f32], b: &[f32]) -> Vec<f32> {
        assert!(a.len() == b.len());
        let mut res = vec![0f32; a.len()];
        for idx in (0..a.len()).step_by(D::F32Vec::LEN) {
            let mut a_vec = D::F32Vec::load(d, &a[idx..]);
            a_vec -= D::F32Vec::load(d, &b[idx..]);
            a_vec.store(&mut res[idx..]);
        }
        res
    }

    fn vec_mul_assign<D: SimdDescriptor>(d: D, a: &[f32], b: &[f32]) -> Vec<f32> {
        assert!(a.len() == b.len());
        let mut res = vec![0f32; a.len()];
        for idx in (0..a.len()).step_by(D::F32Vec::LEN) {
            let mut a_vec = D::F32Vec::load(d, &a[idx..]);
            a_vec *= D::F32Vec::load(d, &b[idx..]);
            a_vec.store(&mut res[idx..]);
        }
        res
    }

    fn vec_abs<D: SimdDescriptor>(d: D, a: &[f32]) -> Vec<f32> {
        let mut res = vec![0f32; a.len()];
        for idx in (0..a.len()).step_by(D::F32Vec::LEN) {
            D::F32Vec::load(d, &a[idx..]).abs().store(&mut res[idx..]);
        }
        res
    }

    fn vec_max<D: SimdDescriptor>(d: D, a: &[f32], b: &[f32]) -> Vec<f32> {
        assert!(a.len() == b.len());
        let mut res = vec![0f32; a.len()];
        for idx in (0..a.len()).step_by(D::F32Vec::LEN) {
            D::F32Vec::load(d, &a[idx..])
                .max(D::F32Vec::load(d, &b[idx..]))
                .store(&mut res[idx..]);
        }
        res
    }

    fn simd_op_scalar_equivalent<D: SimdDescriptor>(d: D) {
        arbtest::arbtest(|u| {
            let mut a = vec![0.0; round_up_size_to_two_cache_lines::<f32>(D::F32Vec::LEN)];
            let mut b = vec![0.0; round_up_size_to_two_cache_lines::<f32>(D::F32Vec::LEN)];

            for i in 0..D::F32Vec::LEN {
                a[i] = u.arbitrary::<f32>()?;
                b[i] = u.arbitrary::<f32>()?;
            }

            let scalar_add_result = vec_add(ScalarDescriptor::new().unwrap(), &a, &b);
            let d_add_result = vec_add(d, &a, &b);
            assert_all_almost_eq!(scalar_add_result, d_add_result, 1e-8);

            let scalar_sub_result = vec_sub(ScalarDescriptor::new().unwrap(), &a, &b);
            let d_sub_result = vec_sub(d, &a, &b);
            assert_all_almost_eq!(scalar_sub_result, d_sub_result, 1e-8);

            let scalar_mul_result = vec_mul(ScalarDescriptor::new().unwrap(), &a, &b);
            let d_mul_result = vec_mul(d, &a, &b);
            assert_all_almost_eq!(scalar_mul_result, d_mul_result, 1e-8);

            let scalar_add_assign_result = vec_add_assign(ScalarDescriptor::new().unwrap(), &a, &b);
            let d_add_assign_result = vec_add_assign(d, &a, &b);
            assert_all_almost_eq!(scalar_add_assign_result, d_add_assign_result, 1e-8);

            let scalar_sub_assign_result = vec_sub_assign(ScalarDescriptor::new().unwrap(), &a, &b);
            let d_sub_assign_result = vec_sub_assign(d, &a, &b);
            assert_all_almost_eq!(scalar_sub_assign_result, d_sub_assign_result, 1e-8);

            let scalar_mul_assign_result = vec_mul_assign(ScalarDescriptor::new().unwrap(), &a, &b);
            let d_mul_assign_result = vec_mul_assign(d, &a, &b);
            assert_all_almost_eq!(scalar_mul_assign_result, d_mul_assign_result, 1e-8);

            let scalar_abs_result = vec_abs(ScalarDescriptor::new().unwrap(), &a);
            let d_abs_result = vec_abs(d, &a);
            assert_all_almost_eq!(scalar_abs_result, d_abs_result, 1e-8);

            let scalar_max_result = vec_max(ScalarDescriptor::new().unwrap(), &a, &b);
            let d_max_result = vec_max(d, &a, &b);
            assert_all_almost_eq!(scalar_max_result, d_max_result, 1e-8);

            Ok(())
        });
    }

    test_all_instruction_sets!(simd_op_scalar_equivalent);
}

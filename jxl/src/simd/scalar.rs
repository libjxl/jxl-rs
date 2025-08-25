// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

use super::{F32SimdVec, SimdDescriptor};

#[derive(Clone, Copy, Debug)]
pub struct ScalarDescriptor;

impl SimdDescriptor for ScalarDescriptor {
    type F32Vec = F32VecScalar;
    fn new() -> Option<Self> {
        Some(Self)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct F32VecScalar(f32);

impl F32SimdVec for F32VecScalar {
    type Descriptor = ScalarDescriptor;

    const LEN: usize = 1;

    fn load(_d: Self::Descriptor, mem: &[f32]) -> Self {
        Self(mem[0])
    }

    fn store(&self, mem: &mut [f32]) {
        mem[0] = self.0;
    }

    fn mul_add(self, mul: Self, add: Self) -> Self {
        Self(self.0.mul_add(mul.0, add.0))
    }

    fn splat(_d: Self::Descriptor, v: f32) -> Self {
        Self(v)
    }

    fn abs(self) -> Self {
        Self(self.0.abs())
    }

    fn max(self, other: Self) -> Self {
        Self(self.0.max(other.0))
    }
}

impl Add<F32VecScalar> for F32VecScalar {
    type Output = F32VecScalar;
    fn add(self, rhs: F32VecScalar) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Sub<F32VecScalar> for F32VecScalar {
    type Output = F32VecScalar;
    fn sub(self, rhs: F32VecScalar) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl Mul<F32VecScalar> for F32VecScalar {
    type Output = F32VecScalar;
    fn mul(self, rhs: F32VecScalar) -> Self::Output {
        Self(self.0 * rhs.0)
    }
}

impl Div<F32VecScalar> for F32VecScalar {
    type Output = F32VecScalar;
    fn div(self, rhs: F32VecScalar) -> Self::Output {
        Self(self.0 / rhs.0)
    }
}

impl AddAssign<F32VecScalar> for F32VecScalar {
    fn add_assign(&mut self, rhs: F32VecScalar) {
        self.0 += rhs.0;
    }
}

impl SubAssign<F32VecScalar> for F32VecScalar {
    fn sub_assign(&mut self, rhs: F32VecScalar) {
        self.0 -= rhs.0;
    }
}

impl MulAssign<F32VecScalar> for F32VecScalar {
    fn mul_assign(&mut self, rhs: F32VecScalar) {
        self.0 *= rhs.0;
    }
}

impl DivAssign<F32VecScalar> for F32VecScalar {
    fn div_assign(&mut self, rhs: F32VecScalar) {
        self.0 /= rhs.0;
    }
}

#[allow(unused_macros)]
macro_rules! simd_function {
    (
        $dname:ident,
        $descr:ident: $descr_ty:ident,
        $pub:vis fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty )? $body: block
    ) => {
        $pub fn $name<$descr_ty: crate::simd::SimdDescriptor>($descr: $descr_ty, $($arg: $ty),*) $(-> $ret)? $body
        $pub fn $dname($($arg: $ty),*) $(-> $ret)? {
            use crate::simd::SimdDescriptor;
            $name(crate::simd::ScalarDescriptor::new().unwrap(), $($arg),*)
        }
    };
}

#[allow(unused_imports)]
pub(crate) use simd_function;

#[allow(unused_macros)]
macro_rules! test_all_instruction_sets {
    (
        $name:ident
    ) => {
        paste::paste! {
            #[test]
            fn [<$name _scalar>]() {
                use crate::simd::SimdDescriptor;
                $name(crate::simd::ScalarDescriptor::new().unwrap())
            }
        }
    };
}

#[allow(unused_imports)]
pub(crate) use test_all_instruction_sets;

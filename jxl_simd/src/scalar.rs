// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::{F32SimdVec, I32SimdVec, SimdDescriptor, SimdMask};

#[derive(Clone, Copy, Debug)]
pub struct ScalarDescriptor;

impl SimdDescriptor for ScalarDescriptor {
    type F32Vec = f32;
    type I32Vec = i32;
    type Mask = bool;
    fn new() -> Option<Self> {
        Some(Self)
    }

    #[inline(always)]
    fn transpose<const ROWS: usize, const COLS: usize>(self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), ROWS * COLS);
        assert_eq!(output.len(), ROWS * COLS);

        for r in 0..ROWS {
            for c in 0..COLS {
                let input_idx = r * COLS + c;
                let output_idx = c * ROWS + r;
                output[output_idx] = input[input_idx];
            }
        }
    }

    fn call<R>(self, f: impl FnOnce(Self) -> R) -> R {
        // No special features needed for scalar implementation
        f(self)
    }
}

impl F32SimdVec for f32 {
    type Descriptor = ScalarDescriptor;

    const LEN: usize = 1;

    #[inline(always)]
    fn load(_d: Self::Descriptor, mem: &[f32]) -> Self {
        mem[0]
    }

    #[inline(always)]
    fn load_partial(d: Self::Descriptor, size: usize, mem: &[f32]) -> Self {
        assert_eq!(size, 1);
        Self::load(d, mem)
    }

    #[inline(always)]
    fn store(&self, mem: &mut [f32]) {
        mem[0] = *self;
    }

    #[inline(always)]
    fn store_partial(&self, size: usize, mem: &mut [f32]) {
        assert_eq!(size, 1);
        self.store(mem)
    }

    #[inline(always)]
    fn mul_add(self, mul: Self, add: Self) -> Self {
        (self * mul) + add
    }

    #[inline(always)]
    fn neg_mul_add(self, mul: Self, add: Self) -> Self {
        -(self * mul) + add
    }

    #[inline(always)]
    fn splat(_d: Self::Descriptor, v: f32) -> Self {
        v
    }

    #[inline(always)]
    fn zero(_d: Self::Descriptor) -> Self {
        0.0
    }

    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }

    #[inline(always)]
    fn floor(self) -> Self {
        self.floor()
    }

    #[inline(always)]
    fn sqrt(self) -> Self {
        self.sqrt()
    }

    #[inline(always)]
    fn neg(self) -> Self {
        -self
    }

    #[inline(always)]
    fn copysign(self, sign: Self) -> Self {
        self.copysign(sign)
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        self.max(other)
    }

    #[inline(always)]
    fn gt(self, other: Self) -> bool {
        self > other
    }

    #[inline(always)]
    fn as_i32(self) -> i32 {
        self as i32
    }

    #[inline(always)]
    fn bitcast_to_i32(self) -> i32 {
        self.to_bits() as i32
    }
}

impl I32SimdVec for i32 {
    type Descriptor = ScalarDescriptor;

    const LEN: usize = 1;

    #[inline(always)]
    fn splat(_d: Self::Descriptor, v: i32) -> Self {
        v
    }

    #[inline(always)]
    fn load(_d: Self::Descriptor, mem: &[i32]) -> Self {
        mem[0]
    }

    #[inline(always)]
    fn abs(self) -> Self {
        self.abs()
    }

    #[inline(always)]
    fn as_f32(self) -> f32 {
        self as f32
    }

    #[inline(always)]
    fn bitcast_to_f32(self) -> f32 {
        f32::from_bits(self as u32)
    }

    #[inline(always)]
    fn gt(self, other: Self) -> bool {
        self > other
    }
}

impl SimdMask for bool {
    type Descriptor = ScalarDescriptor;

    #[inline(always)]
    fn if_then_else_f32(self, if_true: f32, if_false: f32) -> f32 {
        if self { if_true } else { if_false }
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[macro_export]
macro_rules! simd_function {
    (
        $dname:ident,
        $descr:ident: $descr_ty:ident,
        $(#[$($attr:meta)*])*
        $pub:vis fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty )? $body: block
    ) => {
        $(#[$($attr)*])*
        $pub fn $name<$descr_ty: $crate::SimdDescriptor>($descr: $descr_ty, $($arg: $ty),*) $(-> $ret)? $body
        $(#[$($attr)*])*
        $pub fn $dname($($arg: $ty),*) $(-> $ret)? {
            use $crate::SimdDescriptor;
            $name($crate::ScalarDescriptor::new().unwrap(), $($arg),*)
        }
    };
}

#[cfg(not(target_arch = "x86_64"))]
#[macro_export]
macro_rules! test_all_instruction_sets {
    (
        $name:ident
    ) => {
        paste::paste! {
            #[test]
            fn [<$name _scalar>]() {
                use $crate::SimdDescriptor;
                $name($crate::ScalarDescriptor::new().unwrap())
            }
        }
    };
}

#[cfg(not(target_arch = "x86_64"))]
#[macro_export]
macro_rules! bench_all_instruction_sets {
    (
        $name:ident,
        $criterion:ident
    ) => {
        use $crate::SimdDescriptor;
        $name(
            $crate::ScalarDescriptor::new().unwrap(),
            $criterion,
            &format!("{}_scalar", stringify!($name)),
        );
    };
}

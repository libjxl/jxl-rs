// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::{
    arch::x86_64::{
        __m512, _mm512_add_ps, _mm512_andnot_si512, _mm512_castps_si512, _mm512_castsi512_ps,
        _mm512_div_ps, _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_max_ps, _mm512_mul_ps,
        _mm512_set1_epi32, _mm512_set1_ps, _mm512_storeu_ps, _mm512_sub_ps,
    },
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

use super::super::{F32SimdVec, SimdDescriptor};

// Safety invariant: this type is only ever constructed if avx512f is available.
#[derive(Clone, Copy, Debug)]
pub struct Avx512Descriptor;

impl SimdDescriptor for Avx512Descriptor {
    type F32Vec = F32VecAvx512;
    fn new() -> Option<Self> {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: we just checked avx512f.
            Some(Self)
        } else {
            None
        }
    }
}

// TODO(veluca): retire this macro once we have #[unsafe(target_feature)].
macro_rules! fn_avx {
    (
        $this:ident: $self_ty:ty,
        fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty )? $body: block) => {
        #[inline(always)]
        fn $name(self: $self_ty, $($arg: $ty),*) $(-> $ret)? {
            #[target_feature(enable = "avx512f")]
            #[inline]
            fn inner($this: $self_ty, $($arg: $ty),*) $(-> $ret)? {
                $body
            }
            // SAFETY: `self.1` is constructed iff avx512f is available.
            unsafe { inner(self, $($arg),*) }
        }
    };
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct F32VecAvx512(__m512, Avx512Descriptor);

impl F32SimdVec for F32VecAvx512 {
    type Descriptor = Avx512Descriptor;

    const LEN: usize = 16;

    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[f32]) -> Self {
        assert!(mem.len() >= Self::LEN);
        // SAFETY: we just checked that `mem` has enough space. Moreover, we know avx512f is available
        // from the safety invariant on `d`.
        Self(unsafe { _mm512_loadu_ps(mem.as_ptr()) }, d)
    }

    #[inline(always)]
    fn store(&self, mem: &mut [f32]) {
        assert!(mem.len() >= Self::LEN);
        // SAFETY: we just checked that `mem` has enough space. Moreover, we know avx512f is available
        // from the safety invariant on `self.1`.
        unsafe { _mm512_storeu_ps(mem.as_mut_ptr(), self.0) }
    }

    fn_avx!(this: F32VecAvx512, fn mul_add(mul: F32VecAvx512, add: F32VecAvx512) -> F32VecAvx512 {
        F32VecAvx512(_mm512_fmadd_ps(this.0, mul.0, add.0), this.1)
    });

    fn splat(d: Self::Descriptor, v: f32) -> Self {
        // SAFETY: We know avx512f is available from the safety invariant on `d`.
        unsafe { Self(_mm512_set1_ps(v), d) }
    }

    fn_avx!(this: F32VecAvx512, fn abs() -> F32VecAvx512 {
        F32VecAvx512(
            _mm512_castsi512_ps(_mm512_andnot_si512(
                _mm512_set1_epi32(0b10000000000000000000000000000000u32 as i32),
                _mm512_castps_si512(this.0),
            )),
            this.1)
    });

    fn_avx!(this: F32VecAvx512, fn max(other: F32VecAvx512) -> F32VecAvx512 {
        F32VecAvx512(_mm512_max_ps(this.0, other.0), this.1)
    });
}

impl Add<F32VecAvx512> for F32VecAvx512 {
    type Output = F32VecAvx512;
    fn_avx!(this: F32VecAvx512, fn add(rhs: F32VecAvx512) -> F32VecAvx512 {
        F32VecAvx512(_mm512_add_ps(this.0, rhs.0), this.1)
    });
}

impl Sub<F32VecAvx512> for F32VecAvx512 {
    type Output = F32VecAvx512;
    fn_avx!(this: F32VecAvx512, fn sub(rhs: F32VecAvx512) -> F32VecAvx512 {
        F32VecAvx512(_mm512_sub_ps(this.0, rhs.0), this.1)
    });
}

impl Mul<F32VecAvx512> for F32VecAvx512 {
    type Output = F32VecAvx512;
    fn_avx!(this: F32VecAvx512, fn mul(rhs: F32VecAvx512) -> F32VecAvx512 {
        F32VecAvx512(_mm512_mul_ps(this.0, rhs.0), this.1)
    });
}

impl Div<F32VecAvx512> for F32VecAvx512 {
    type Output = F32VecAvx512;
    fn_avx!(this: F32VecAvx512, fn div(rhs: F32VecAvx512) -> F32VecAvx512 {
        F32VecAvx512(_mm512_div_ps(this.0, rhs.0), this.1)
    });
}

impl AddAssign<F32VecAvx512> for F32VecAvx512 {
    fn_avx!(this: &mut F32VecAvx512, fn add_assign(rhs: F32VecAvx512) {
        this.0 = _mm512_add_ps(this.0, rhs.0)
    });
}

impl SubAssign<F32VecAvx512> for F32VecAvx512 {
    fn_avx!(this: &mut F32VecAvx512, fn sub_assign(rhs: F32VecAvx512) {
        this.0 = _mm512_sub_ps(this.0, rhs.0)
    });
}

impl MulAssign<F32VecAvx512> for F32VecAvx512 {
    fn_avx!(this: &mut F32VecAvx512, fn mul_assign(rhs: F32VecAvx512) {
        this.0 = _mm512_mul_ps(this.0, rhs.0)
    });
}

impl DivAssign<F32VecAvx512> for F32VecAvx512 {
    fn_avx!(this: &mut F32VecAvx512, fn div_assign(rhs: F32VecAvx512) {
        this.0 = _mm512_div_ps(this.0, rhs.0)
    });
}

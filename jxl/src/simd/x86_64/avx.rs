// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::{
    arch::x86_64::{
        __m256, _mm256_add_ps, _mm256_andnot_si256, _mm256_castps_si256, _mm256_castsi256_ps,
        _mm256_div_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_max_ps, _mm256_mul_ps,
        _mm256_set1_epi32, _mm256_set1_ps, _mm256_storeu_ps, _mm256_sub_ps,
    },
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

use super::super::{F32SimdVec, SimdDescriptor};

// Safety invariant: this type is only ever constructed if avx2 and fma are available.
#[derive(Clone, Copy, Debug)]
pub struct AvxDescriptor;

impl SimdDescriptor for AvxDescriptor {
    type F32Vec = F32VecAvx;
    fn new() -> Option<Self> {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: we just checked avx2 and fma.
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
            #[target_feature(enable = "fma,avx2")]
            #[inline]
            fn inner($this: $self_ty, $($arg: $ty),*) $(-> $ret)? {
                $body
            }
            // SAFETY: `self.1` is constructed iff avx2 and fma are available.
            unsafe { inner(self, $($arg),*) }
        }
    };
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct F32VecAvx(__m256, AvxDescriptor);

impl F32SimdVec for F32VecAvx {
    type Descriptor = AvxDescriptor;

    const LEN: usize = 8;

    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[f32]) -> Self {
        assert!(mem.len() >= Self::LEN);
        // SAFETY: we just checked that `mem` has enough space. Moreover, we know avx is available
        // from the safety invariant on `d`.
        Self(unsafe { _mm256_loadu_ps(mem.as_ptr()) }, d)
    }

    #[inline(always)]
    fn store(&self, mem: &mut [f32]) {
        assert!(mem.len() >= Self::LEN);
        // SAFETY: we just checked that `mem` has enough space. Moreover, we know avx is available
        // from the safety invariant on `self.1`.
        unsafe { _mm256_storeu_ps(mem.as_mut_ptr(), self.0) }
    }

    fn_avx!(this: F32VecAvx, fn mul_add(mul: F32VecAvx, add: F32VecAvx) -> F32VecAvx {
        F32VecAvx(_mm256_fmadd_ps(this.0, mul.0, add.0), this.1)
    });

    fn splat(d: Self::Descriptor, v: f32) -> Self {
        // SAFETY: We know avx is available from the safety invariant on `d`.
        unsafe { Self(_mm256_set1_ps(v), d) }
    }

    fn_avx!(this: F32VecAvx, fn abs() -> F32VecAvx {
        F32VecAvx(
            _mm256_castsi256_ps(_mm256_andnot_si256(
                _mm256_set1_epi32(0b10000000000000000000000000000000u32 as i32),
                _mm256_castps_si256(this.0),
            )),
            this.1)
    });

    fn_avx!(this: F32VecAvx, fn max(other: F32VecAvx) -> F32VecAvx {
        F32VecAvx(_mm256_max_ps(this.0, other.0), this.1)
    });
}

impl Add<F32VecAvx> for F32VecAvx {
    type Output = F32VecAvx;
    fn_avx!(this: F32VecAvx, fn add(rhs: F32VecAvx) -> F32VecAvx {
        F32VecAvx(_mm256_add_ps(this.0, rhs.0), this.1)
    });
}

impl Sub<F32VecAvx> for F32VecAvx {
    type Output = F32VecAvx;
    fn_avx!(this: F32VecAvx, fn sub(rhs: F32VecAvx) -> F32VecAvx {
        F32VecAvx(_mm256_sub_ps(this.0, rhs.0), this.1)
    });
}

impl Mul<F32VecAvx> for F32VecAvx {
    type Output = F32VecAvx;
    fn_avx!(this: F32VecAvx, fn mul(rhs: F32VecAvx) -> F32VecAvx {
        F32VecAvx(_mm256_mul_ps(this.0, rhs.0), this.1)
    });
}

impl Div<F32VecAvx> for F32VecAvx {
    type Output = F32VecAvx;
    fn_avx!(this: F32VecAvx, fn div(rhs: F32VecAvx) -> F32VecAvx {
        F32VecAvx(_mm256_div_ps(this.0, rhs.0), this.1)
    });
}

impl AddAssign<F32VecAvx> for F32VecAvx {
    fn_avx!(this: &mut F32VecAvx, fn add_assign(rhs: F32VecAvx) {
        this.0 = _mm256_add_ps(this.0, rhs.0)
    });
}

impl SubAssign<F32VecAvx> for F32VecAvx {
    fn_avx!(this: &mut F32VecAvx, fn sub_assign(rhs: F32VecAvx) {
        this.0 = _mm256_sub_ps(this.0, rhs.0)
    });
}

impl MulAssign<F32VecAvx> for F32VecAvx {
    fn_avx!(this: &mut F32VecAvx, fn mul_assign(rhs: F32VecAvx) {
        this.0 = _mm256_mul_ps(this.0, rhs.0)
    });
}

impl DivAssign<F32VecAvx> for F32VecAvx {
    fn_avx!(this: &mut F32VecAvx, fn div_assign(rhs: F32VecAvx) {
        this.0 = _mm256_div_ps(this.0, rhs.0)
    });
}

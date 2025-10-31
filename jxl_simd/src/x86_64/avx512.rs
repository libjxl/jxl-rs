// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::{
    arch::x86_64::{
        __m512, __m512i, __mmask16, _mm512_abs_epi32, _mm512_add_epi32, _mm512_add_ps,
        _mm512_andnot_si512, _mm512_castps_si512, _mm512_castsi512_ps, _mm512_cmpgt_epi32_mask,
        _mm512_cvtepi32_ps, _mm512_div_ps, _mm512_fmadd_ps, _mm512_fnmadd_ps, _mm512_loadu_epi32,
        _mm512_loadu_ps, _mm512_mask_blend_ps, _mm512_mask_loadu_ps, _mm512_mask_storeu_ps,
        _mm512_max_ps, _mm512_mul_epi32, _mm512_mul_ps, _mm512_set1_epi32, _mm512_set1_ps,
        _mm512_setzero_ps, _mm512_storeu_ps, _mm512_sub_epi32, _mm512_sub_ps, _mm512_xor_si512,
    },
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

use super::super::{AvxDescriptor, F32SimdVec, I32SimdVec, SimdDescriptor, SimdMask};

// Safety invariant: this type is only ever constructed if avx512f is available.
#[derive(Clone, Copy, Debug)]
pub struct Avx512Descriptor(());

#[allow(unused)]
impl Avx512Descriptor {
    /// # Safety
    /// The caller must guarantee that the "avx512f" target feature is available.
    pub unsafe fn new_unchecked() -> Self {
        Self(())
    }
    pub fn as_avx(&self) -> AvxDescriptor {
        // SAFETY: the safety invariant on `self` guarantees avx512f is available, which implies
        // avx2 and fma.
        unsafe { AvxDescriptor::new_unchecked() }
    }
}

impl SimdDescriptor for Avx512Descriptor {
    type F32Vec = F32VecAvx512;
    type I32Vec = I32VecAvx512;
    type Mask = MaskAvx512;
    fn new() -> Option<Self> {
        if is_x86_feature_detected!("avx512f") {
            // SAFETY: we just checked avx512f.
            Some(Self(()))
        } else {
            None
        }
    }

    #[inline(always)]
    fn transpose<const ROWS: usize, const COLS: usize>(self, input: &[f32], output: &mut [f32]) {
        // TODO: implement an AVX-512-specific version
        self.as_avx().transpose::<ROWS, COLS>(input, output)
    }

    fn call<R>(self, f: impl FnOnce(Self) -> R) -> R {
        #[target_feature(enable = "avx512f")]
        unsafe fn inner<R>(d: Avx512Descriptor, f: impl FnOnce(Avx512Descriptor) -> R) -> R {
            f(d)
        }
        // SAFETY: the safety invariant on `self` guarantees avx512f.
        unsafe { inner(self, f) }
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

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct MaskAvx512(__mmask16, Avx512Descriptor);

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
    fn load_partial(d: Self::Descriptor, size: usize, mem: &[f32]) -> Self {
        assert!(Self::LEN >= size);
        assert!(mem.len() >= size);
        // Fast path: avoid mask setup overhead when loading full vectors
        // This optimization skips the expensive mask creation and masked load when size == LEN
        if size == Self::LEN {
            return Self::load(d, mem);
        }
        // SAFETY: we just checked that `mem` has enough space. Moreover, we know avx512f is available
        // from the safety invariant on `d`.
        Self(
            unsafe { _mm512_mask_loadu_ps(_mm512_setzero_ps(), (1u16 << size) - 1, mem.as_ptr()) },
            d,
        )
    }

    #[inline(always)]
    fn store(&self, mem: &mut [f32]) {
        assert!(mem.len() >= Self::LEN);
        // SAFETY: we just checked that `mem` has enough space. Moreover, we know avx512f is available
        // from the safety invariant on `self.1`.
        unsafe { _mm512_storeu_ps(mem.as_mut_ptr(), self.0) }
    }

    #[inline(always)]
    fn store_partial(&self, size: usize, mem: &mut [f32]) {
        assert!(Self::LEN >= size);
        assert!(mem.len() >= size);
        if size == Self::LEN {
            return self.store(mem);
        }
        // SAFETY: we just checked that `mem` has enough space. Moreover, we know avx512f is available
        // from the safety invariant on `self.1`.
        unsafe { _mm512_mask_storeu_ps(mem.as_mut_ptr(), (1u16 << size) - 1, self.0) }
    }

    fn_avx!(this: F32VecAvx512, fn mul_add(mul: F32VecAvx512, add: F32VecAvx512) -> F32VecAvx512 {
        F32VecAvx512(_mm512_fmadd_ps(this.0, mul.0, add.0), this.1)
    });

    fn_avx!(this: F32VecAvx512, fn neg_mul_add(mul: F32VecAvx512, add: F32VecAvx512) -> F32VecAvx512 {
        F32VecAvx512(_mm512_fnmadd_ps(this.0, mul.0, add.0), this.1)
    });

    #[inline(always)]
    fn splat(d: Self::Descriptor, v: f32) -> Self {
        // SAFETY: We know avx512f is available from the safety invariant on `d`.
        unsafe { Self(_mm512_set1_ps(v), d) }
    }

    fn_avx!(this: F32VecAvx512, fn abs() -> F32VecAvx512 {
        F32VecAvx512(
            _mm512_castsi512_ps(_mm512_andnot_si512(
                _mm512_set1_epi32(i32::MIN),
                _mm512_castps_si512(this.0),
            )),
            this.1)
    });

    fn_avx!(this: F32VecAvx512, fn neg() -> F32VecAvx512 {
        F32VecAvx512(
            _mm512_castsi512_ps(_mm512_xor_si512(
                _mm512_set1_epi32(i32::MIN),
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

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct I32VecAvx512(__m512i, Avx512Descriptor);

impl I32SimdVec for I32VecAvx512 {
    type Descriptor = Avx512Descriptor;
    type F32Vec = F32VecAvx512;
    type Mask = MaskAvx512;

    const LEN: usize = 16;

    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[i32]) -> Self {
        assert!(mem.len() >= Self::LEN);
        // SAFETY: we just checked that `mem` has enough space. Moreover, we know avx512f is available
        // from the safety invariant on `d`.
        Self(unsafe { _mm512_loadu_epi32(mem.as_ptr()) }, d)
    }

    #[inline(always)]
    fn splat(d: Self::Descriptor, v: i32) -> Self {
        // SAFETY: We know avx512f is available from the safety invariant on `d`.
        unsafe { Self(_mm512_set1_epi32(v), d) }
    }

    fn_avx!(this: I32VecAvx512, fn as_f32() -> F32VecAvx512 {
         F32VecAvx512(_mm512_cvtepi32_ps(this.0), this.1)
    });

    fn_avx!(this: I32VecAvx512, fn abs() -> I32VecAvx512 {
        I32VecAvx512(
            _mm512_abs_epi32(
                this.0,
            ),
            this.1)
    });

    fn_avx!(this: I32VecAvx512, fn gt(rhs: I32VecAvx512) -> MaskAvx512 {
        MaskAvx512(
            _mm512_cmpgt_epi32_mask(this.0, rhs.0),
            this.1
        )
    });
}

impl Add<I32VecAvx512> for I32VecAvx512 {
    type Output = I32VecAvx512;
    fn_avx!(this: I32VecAvx512, fn add(rhs: I32VecAvx512) -> I32VecAvx512 {
        I32VecAvx512(_mm512_add_epi32(this.0, rhs.0), this.1)
    });
}

impl Sub<I32VecAvx512> for I32VecAvx512 {
    type Output = I32VecAvx512;
    fn_avx!(this: I32VecAvx512, fn sub(rhs: I32VecAvx512) -> I32VecAvx512 {
        I32VecAvx512(_mm512_sub_epi32(this.0, rhs.0), this.1)
    });
}

impl Mul<I32VecAvx512> for I32VecAvx512 {
    type Output = I32VecAvx512;
    fn_avx!(this: I32VecAvx512, fn mul(rhs: I32VecAvx512) -> I32VecAvx512 {
        I32VecAvx512(_mm512_mul_epi32(this.0, rhs.0), this.1)
    });
}

impl AddAssign<I32VecAvx512> for I32VecAvx512 {
    fn_avx!(this: &mut I32VecAvx512, fn add_assign(rhs: I32VecAvx512) {
        this.0 = _mm512_add_epi32(this.0, rhs.0)
    });
}

impl SubAssign<I32VecAvx512> for I32VecAvx512 {
    fn_avx!(this: &mut I32VecAvx512, fn sub_assign(rhs: I32VecAvx512) {
        this.0 = _mm512_sub_epi32(this.0, rhs.0)
    });
}

impl MulAssign<I32VecAvx512> for I32VecAvx512 {
    fn_avx!(this: &mut I32VecAvx512, fn mul_assign(rhs: I32VecAvx512) {
        this.0 = _mm512_mul_epi32(this.0, rhs.0)
    });
}

impl SimdMask for MaskAvx512 {
    type Descriptor = Avx512Descriptor;
    type F32Vec = F32VecAvx512;
    type I32Vec = I32VecAvx512;

    fn_avx!(this: MaskAvx512, fn if_then_else_f32(if_true: F32VecAvx512, if_false: F32VecAvx512) -> F32VecAvx512 {
         F32VecAvx512(_mm512_mask_blend_ps(this.0, if_false.0, if_true.0), this.1)
    });
}

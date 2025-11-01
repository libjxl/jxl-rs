// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::arch::aarch64::*;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Shl, ShlAssign, Shr, ShrAssign, Sub, SubAssign,
};

use super::super::{F32SimdVec, I32SimdVec, ScalarDescriptor, SimdDescriptor, SimdMask};

// Safety invariant: this type is only ever constructed if neon is available.
#[derive(Clone, Copy, Debug)]
pub struct NeonDescriptor(());

impl NeonDescriptor {
    /// # Safety
    /// The caller must guarantee that the "neon" target feature is available.
    pub unsafe fn new_unchecked() -> Self {
        Self(())
    }

    #[target_feature(enable = "neon")]
    #[inline]
    fn transpose4x4f32(
        self,
        input: &[f32],
        input_stride: usize,
        output: &mut [f32],
        output_stride: usize,
    ) {
        assert!(input_stride >= 4);
        assert!(input.len() >= input_stride.checked_mul(3).unwrap().checked_add(4).unwrap());
        assert!(
            output.len()
                >= output_stride
                    .checked_mul(3)
                    .unwrap()
                    .checked_add(4)
                    .unwrap()
        );

        // SAFETY: input is verified to be large enough for this pointer arithmetic.
        let (p0, p1, p2, p3) = unsafe {
            (
                vld1q_f32(input.as_ptr()),
                vld1q_f32(input.as_ptr().add(input_stride)),
                vld1q_f32(input.as_ptr().add(2 * input_stride)),
                vld1q_f32(input.as_ptr().add(3 * input_stride)),
            )
        };

        let tr0 = vreinterpretq_f64_f32(vtrn1q_f32(p0, p1));
        let tr1 = vreinterpretq_f64_f32(vtrn2q_f32(p0, p1));
        let tr2 = vreinterpretq_f64_f32(vtrn1q_f32(p2, p3));
        let tr3 = vreinterpretq_f64_f32(vtrn2q_f32(p2, p3));

        let p0 = vreinterpretq_f32_f64(vzip1q_f64(tr0, tr2));
        let p1 = vreinterpretq_f32_f64(vzip1q_f64(tr1, tr3));
        let p2 = vreinterpretq_f32_f64(vzip2q_f64(tr0, tr2));
        let p3 = vreinterpretq_f32_f64(vzip2q_f64(tr1, tr3));

        // SAFETY: output is verified to be large enough for this pointer arithmetic.
        unsafe {
            vst1q_f32(output.as_mut_ptr(), p0);
            vst1q_f32(output.as_mut_ptr().add(output_stride), p1);
            vst1q_f32(output.as_mut_ptr().add(2 * output_stride), p2);
            vst1q_f32(output.as_mut_ptr().add(3 * output_stride), p3);
        }
    }

    #[target_feature(enable = "neon")]
    #[inline]
    fn transpose4x4f32_contiguous(self, input: &[f32], output: &mut [f32]) {
        assert!(input.len() >= 4 * 4);
        assert!(output.len() >= 4 * 4);

        // SAFETY: input is verified to be large enough for this pointer.
        let float32x4x4_t(p0, p1, p2, p3) = unsafe { vld4q_f32(input.as_ptr()) };

        // SAFETY: output is verified to be large enough for this pointer arithmetic.
        unsafe {
            vst1q_f32(&raw mut output[0], p0);
            vst1q_f32(&raw mut output[4], p1);
            vst1q_f32(&raw mut output[8], p2);
            vst1q_f32(&raw mut output[12], p3);
        }
    }
}

impl SimdDescriptor for NeonDescriptor {
    type F32Vec = F32VecNeon;
    type I32Vec = I32VecNeon;
    type Mask = MaskNeon;

    fn new() -> Option<Self> {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: we just checked neon.
            Some(unsafe { Self::new_unchecked() })
        } else {
            None
        }
    }

    #[inline(always)]
    fn transpose<const ROWS: usize, const COLS: usize>(self, input: &[f32], output: &mut [f32]) {
        assert_eq!(input.len(), ROWS * COLS);
        assert_eq!(output.len(), ROWS * COLS);

        if ROWS == 4 && COLS == 4 {
            // SAFETY: We know neon is available from the safety invariant on `self`.
            unsafe {
                self.transpose4x4f32_contiguous(input, output);
            }
        } else if ROWS.is_multiple_of(4) && COLS.is_multiple_of(4) {
            for r in (0..ROWS).step_by(4) {
                let input_row = &input[r * COLS..];
                for c in (0..COLS).step_by(4) {
                    let output_row = &mut output[c * ROWS..];
                    // SAFETY: We know neon is available from the safety invariant on `self`.
                    unsafe {
                        self.transpose4x4f32(&input_row[c..], COLS, &mut output_row[r..], ROWS);
                    }
                }
            }
        } else {
            let scalar = ScalarDescriptor;
            scalar.transpose::<ROWS, COLS>(input, output);
        }
    }

    fn call<R>(self, f: impl FnOnce(Self) -> R) -> R {
        #[target_feature(enable = "neon")]
        unsafe fn inner<R>(d: NeonDescriptor, f: impl FnOnce(NeonDescriptor) -> R) -> R {
            f(d)
        }
        // SAFETY: the safety invariant on `self` guarantees neon.
        unsafe { inner(self, f) }
    }
}

// TODO: retire this macro once we have #[unsafe(target_feature)].
macro_rules! fn_neon {
    {} => {};
    {$(
        fn $name:ident($this:ident: $self_ty:ty $(, $arg:ident: $ty:ty)* $(,)?) $(-> $ret:ty )?
        $body: block
    )*} => {$(
        #[inline(always)]
        fn $name(self: $self_ty, $($arg: $ty),*) $(-> $ret)? {
            #[target_feature(enable = "neon")]
            #[inline]
            fn inner($this: $self_ty, $($arg: $ty),*) $(-> $ret)? {
                $body
            }
            // SAFETY: `self.1` is constructed iff neon is available.
            unsafe { inner(self, $($arg),*) }
        }
    )*};
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct F32VecNeon(float32x4_t, NeonDescriptor);

impl F32SimdVec for F32VecNeon {
    type Descriptor = NeonDescriptor;

    const LEN: usize = 4;

    #[inline(always)]
    fn splat(d: Self::Descriptor, v: f32) -> Self {
        // SAFETY: We know neon is available from the safety invariant on `d`.
        Self(unsafe { vdupq_n_f32(v) }, d)
    }

    #[inline(always)]
    fn zero(d: Self::Descriptor) -> Self {
        // SAFETY: We know neon is available from the safety invariant on `d`.
        Self(unsafe { vdupq_n_f32(0.0) }, d)
    }

    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[f32]) -> Self {
        assert!(mem.len() >= Self::LEN);
        // SAFETY: we just checked that `mem` has enough space. Moreover, we know neon is available
        // from the safety invariant on `d`.
        Self(unsafe { vld1q_f32(mem.as_ptr()) }, d)
    }

    #[inline(always)]
    fn load_partial(d: Self::Descriptor, size: usize, mem: &[f32]) -> Self {
        debug_assert!(Self::LEN >= size);
        assert!(mem.len() >= size);
        if size == Self::LEN {
            return Self::load(d, mem);
        }

        // SAFETY: we just checked that `mem` has enough space. Moreover, we know neon is available
        // from the safety invariant on `d`.
        unsafe {
            let mut v = vdupq_n_f32(0.0);
            if size > 0 {
                v = vld1q_lane_f32::<0>(&raw const mem[0], v);
            }
            if size > 1 {
                v = vld1q_lane_f32::<1>(&raw const mem[1], v);
            }
            if size > 2 {
                v = vld1q_lane_f32::<2>(&raw const mem[2], v);
            }
            // `size > 3` case is covered by `size == Self::LEN` above
            debug_assert!(size <= 3);

            Self(v, d)
        }
    }

    #[inline(always)]
    fn store(&self, mem: &mut [f32]) {
        assert!(mem.len() >= Self::LEN);
        // SAFETY: we just checked that `mem` has enough space. Moreover, we know neon is available
        // from the safety invariant on `d`.
        unsafe { vst1q_f32(mem.as_mut_ptr(), self.0) }
    }

    #[inline(always)]
    fn store_partial(&self, size: usize, mem: &mut [f32]) {
        debug_assert!(Self::LEN >= size);
        assert!(mem.len() >= size);
        if size == Self::LEN {
            return self.store(mem);
        }

        // SAFETY: we just checked that `mem` has enough space. Moreover, we know neon is available
        // from the safety invariant on `d`.
        unsafe {
            if size > 0 {
                vst1q_lane_f32::<0>(&raw mut mem[0], self.0);
            }
            if size > 1 {
                vst1q_lane_f32::<1>(&raw mut mem[1], self.0);
            }
            if size > 2 {
                vst1q_lane_f32::<2>(&raw mut mem[2], self.0);
            }
            // `size > 3` case is covered by `size == Self::LEN` above
            debug_assert!(size <= 3);
        }
    }

    fn_neon! {
        fn mul_add(this: F32VecNeon, mul: F32VecNeon, add: F32VecNeon) -> F32VecNeon {
            F32VecNeon(vfmaq_f32(add.0, this.0, mul.0), this.1)
        }

        fn neg_mul_add(this: F32VecNeon, mul: F32VecNeon, add: F32VecNeon) -> F32VecNeon {
            F32VecNeon(vfmsq_f32(add.0, this.0, mul.0), this.1)
        }

        fn abs(this: F32VecNeon) -> F32VecNeon {
            F32VecNeon(vabsq_f32(this.0), this.1)
        }

        fn floor(this: F32VecNeon) -> F32VecNeon {
            F32VecNeon(vrndmq_f32(this.0), this.1)
        }

        fn sqrt(this: F32VecNeon) -> F32VecNeon {
            F32VecNeon(vsqrtq_f32(this.0), this.1)
        }

        fn neg(this: F32VecNeon) -> F32VecNeon {
            F32VecNeon(vnegq_f32(this.0), this.1)
        }

        fn copysign(this: F32VecNeon, sign: F32VecNeon) -> F32VecNeon {
            F32VecNeon(
                vbslq_f32(vdupq_n_u32(0x8000_0000), sign.0, this.0),
                this.1,
            )
        }

        fn max(this: F32VecNeon, other: F32VecNeon) -> F32VecNeon {
            F32VecNeon(vmaxq_f32(this.0, other.0), this.1)
        }

        fn gt(this: F32VecNeon, other: F32VecNeon) -> MaskNeon {
            MaskNeon(vcgtq_f32(this.0, other.0), this.1)
        }

        fn as_i32(this: F32VecNeon) -> I32VecNeon {
            I32VecNeon(vcvtq_s32_f32(this.0), this.1)
        }

        fn bitcast_to_i32(this: F32VecNeon) -> I32VecNeon {
            I32VecNeon(vreinterpretq_s32_f32(this.0), this.1)
        }
    }
}

impl Add<F32VecNeon> for F32VecNeon {
    type Output = Self;
    fn_neon! {
        fn add(this: F32VecNeon, rhs: F32VecNeon) -> F32VecNeon {
            F32VecNeon(vaddq_f32(this.0, rhs.0), this.1)
        }
    }
}

impl Sub<F32VecNeon> for F32VecNeon {
    type Output = Self;
    fn_neon! {
        fn sub(this: F32VecNeon, rhs: F32VecNeon) -> F32VecNeon {
            F32VecNeon(vsubq_f32(this.0, rhs.0), this.1)
        }
    }
}

impl Mul<F32VecNeon> for F32VecNeon {
    type Output = Self;
    fn_neon! {
        fn mul(this: F32VecNeon, rhs: F32VecNeon) -> F32VecNeon {
            F32VecNeon(vmulq_f32(this.0, rhs.0), this.1)
        }
    }
}

impl Div<F32VecNeon> for F32VecNeon {
    type Output = Self;
    fn_neon! {
        fn div(this: F32VecNeon, rhs: F32VecNeon) -> F32VecNeon {
            F32VecNeon(vdivq_f32(this.0, rhs.0), this.1)
        }
    }
}

impl AddAssign<F32VecNeon> for F32VecNeon {
    fn_neon! {
        fn add_assign(this: &mut F32VecNeon, rhs: F32VecNeon) {
            this.0 = vaddq_f32(this.0, rhs.0);
        }
    }
}

impl SubAssign<F32VecNeon> for F32VecNeon {
    fn_neon! {
        fn sub_assign(this: &mut F32VecNeon, rhs: F32VecNeon) {
            this.0 = vsubq_f32(this.0, rhs.0);
        }
    }
}

impl MulAssign<F32VecNeon> for F32VecNeon {
    fn_neon! {
        fn mul_assign(this: &mut F32VecNeon, rhs: F32VecNeon) {
            this.0 = vmulq_f32(this.0, rhs.0);
        }
    }
}

impl DivAssign<F32VecNeon> for F32VecNeon {
    fn_neon! {
        fn div_assign(this: &mut F32VecNeon, rhs: F32VecNeon) {
            this.0 = vdivq_f32(this.0, rhs.0);
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct I32VecNeon(int32x4_t, NeonDescriptor);

impl I32SimdVec for I32VecNeon {
    type Descriptor = NeonDescriptor;

    const LEN: usize = 4;

    #[inline(always)]
    fn splat(d: Self::Descriptor, v: i32) -> Self {
        // SAFETY: We know neon is available from the safety invariant on `d`.
        Self(unsafe { vdupq_n_s32(v) }, d)
    }

    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[i32]) -> Self {
        assert!(mem.len() >= Self::LEN);
        // SAFETY: we just checked that `mem` has enough space. Moreover, we know neon is available
        // from the safety invariant on `d`.
        Self(unsafe { vld1q_s32(mem.as_ptr()) }, d)
    }

    fn_neon! {
        fn abs(this: I32VecNeon) -> I32VecNeon {
            I32VecNeon(vabsq_s32(this.0), this.1)
        }

        fn as_f32(this: I32VecNeon) -> F32VecNeon {
            F32VecNeon(vcvtq_f32_s32(this.0), this.1)
        }

        fn bitcast_to_f32(this: I32VecNeon) -> F32VecNeon {
            F32VecNeon(vreinterpretq_f32_s32(this.0), this.1)
        }

        fn gt(this: I32VecNeon, other: I32VecNeon) -> MaskNeon {
            MaskNeon(vcgtq_s32(this.0, other.0), this.1)
        }
    }
}

impl Add<I32VecNeon> for I32VecNeon {
    type Output = I32VecNeon;
    fn_neon! {
        fn add(this: I32VecNeon, rhs: I32VecNeon) -> I32VecNeon {
            I32VecNeon(vaddq_s32(this.0, rhs.0), this.1)
        }
    }
}

impl Sub<I32VecNeon> for I32VecNeon {
    type Output = I32VecNeon;
    fn_neon! {
        fn sub(this: I32VecNeon, rhs: I32VecNeon) -> I32VecNeon {
            I32VecNeon(vsubq_s32(this.0, rhs.0), this.1)
        }
    }
}

impl Mul<I32VecNeon> for I32VecNeon {
    type Output = I32VecNeon;
    fn_neon! {
        fn mul(this: I32VecNeon, rhs: I32VecNeon) -> I32VecNeon {
            I32VecNeon(vmulq_s32(this.0, rhs.0), this.1)
        }
    }
}

impl Shl<I32VecNeon> for I32VecNeon {
    type Output = I32VecNeon;
    fn_neon! {
        fn shl(this: I32VecNeon, rhs: I32VecNeon) -> I32VecNeon {
            I32VecNeon(vshlq_s32(this.0, rhs.0), this.1)
        }
    }
}

impl Shr<I32VecNeon> for I32VecNeon {
    type Output = I32VecNeon;
    fn_neon! {
        fn shr(this: I32VecNeon, rhs: I32VecNeon) -> I32VecNeon {
            I32VecNeon(vshlq_s32(vnegq_s32(this.0), rhs.0), this.1)
        }
    }
}

impl AddAssign<I32VecNeon> for I32VecNeon {
    fn_neon! {
        fn add_assign(this: &mut I32VecNeon, rhs: I32VecNeon) {
            this.0 = vaddq_s32(this.0, rhs.0)
        }
    }
}

impl SubAssign<I32VecNeon> for I32VecNeon {
    fn_neon! {
        fn sub_assign(this: &mut I32VecNeon, rhs: I32VecNeon) {
            this.0 = vsubq_s32(this.0, rhs.0)
        }
    }
}

impl MulAssign<I32VecNeon> for I32VecNeon {
    fn_neon! {
        fn mul_assign(this: &mut I32VecNeon, rhs: I32VecNeon) {
            this.0 = vmulq_s32(this.0, rhs.0)
        }
    }
}

impl ShlAssign<I32VecNeon> for I32VecNeon {
    fn_neon! {
        fn shl_assign(this: &mut I32VecNeon, rhs: I32VecNeon) {
            this.0 = vshlq_s32(this.0, rhs.0)
        }
    }
}

impl ShrAssign<I32VecNeon> for I32VecNeon {
    fn_neon! {
        fn shr_assign(this: &mut I32VecNeon, rhs: I32VecNeon) {
            this.0 = vshlq_s32(vnegq_s32(this.0), rhs.0)
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct MaskNeon(uint32x4_t, NeonDescriptor);

impl SimdMask for MaskNeon {
    type Descriptor = NeonDescriptor;

    fn_neon! {
        fn if_then_else_f32(
            this: MaskNeon,
            if_true: F32VecNeon,
            if_false: F32VecNeon,
        ) -> F32VecNeon {
            F32VecNeon(vbslq_f32(this.0, if_true.0, if_false.0), this.1)
        }
    }
}

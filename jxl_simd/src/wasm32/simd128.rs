// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Most WASM SIMD intrinsics are safe (unlike x86/ARM), making the unsafe blocks in the
// fn_simd128! macro pattern unnecessary. We allow this for consistency with other backends.
#![allow(unused_unsafe)]

use std::{
    arch::wasm32::*,
    mem::MaybeUninit,
    ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div,
        DivAssign, Mul, MulAssign, Neg, Sub, SubAssign,
    },
};

use crate::U32SimdVec;

use super::super::{F32SimdVec, I32SimdVec, SimdDescriptor, SimdMask, U8SimdVec, U16SimdVec};

// Safety invariant: this type is only ever constructed if simd128 is available.
#[derive(Clone, Copy, Debug)]
pub struct WasmSimdDescriptor(());

impl WasmSimdDescriptor {
    /// # Safety
    /// The caller must guarantee that the "simd128" target feature is available.
    pub unsafe fn new_unchecked() -> Self {
        Self(())
    }
}

/// Prepared 8-entry BF16 lookup table for WASM SIMD.
/// Contains 8 BF16 values packed into 16 bytes.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Bf16Table8WasmSimd(v128);

impl SimdDescriptor for WasmSimdDescriptor {
    type F32Vec = F32VecWasmSimd;

    type I32Vec = I32VecWasmSimd;

    type U32Vec = U32VecWasmSimd;

    type U16Vec = U16VecWasmSimd;

    type U8Vec = U8VecWasmSimd;

    type Mask = MaskWasmSimd;
    type Bf16Table8 = Bf16Table8WasmSimd;

    type Descriptor256 = Self;
    type Descriptor128 = Self;

    fn new() -> Option<Self> {
        // On wasm32, simd128 is always available if compiled with the feature.
        // SAFETY: the simd128 feature is required to compile this module.
        Some(unsafe { Self::new_unchecked() })
    }

    fn maybe_downgrade_256bit(self) -> Self {
        self
    }

    fn maybe_downgrade_128bit(self) -> Self {
        self
    }

    fn call<R>(self, f: impl FnOnce(Self) -> R) -> R {
        #[target_feature(enable = "simd128")]
        #[inline(never)]
        unsafe fn inner<R>(d: WasmSimdDescriptor, f: impl FnOnce(WasmSimdDescriptor) -> R) -> R {
            f(d)
        }
        // SAFETY: the safety invariant on `self` guarantees simd128.
        unsafe { inner(self, f) }
    }
}

// TODO: retire this macro once we have #[unsafe(target_feature)].
macro_rules! fn_simd128 {
    {} => {};
    {$(
        fn $name:ident($this:ident: $self_ty:ty $(, $arg:ident: $ty:ty)* $(,)?) $(-> $ret:ty )?
        $body: block
    )*} => {$(
        #[inline(always)]
        fn $name(self: $self_ty, $($arg: $ty),*) $(-> $ret)? {
            #[target_feature(enable = "simd128")]
            #[inline]
            fn inner($this: $self_ty, $($arg: $ty),*) $(-> $ret)? {
                $body
            }
            // SAFETY: `self.1` is constructed iff simd128 is available.
            unsafe { inner(self, $($arg),*) }
        }
    )*};
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct F32VecWasmSimd(v128, WasmSimdDescriptor);

// SAFETY: The methods in this implementation that write to `MaybeUninit` (store_interleaved_*)
// ensure that they write valid data to the output slice without reading uninitialized memory.
unsafe impl F32SimdVec for F32VecWasmSimd {
    type Descriptor = WasmSimdDescriptor;

    const LEN: usize = 4;

    #[inline(always)]
    fn splat(d: Self::Descriptor, v: f32) -> Self {
        // SAFETY: We know simd128 is available from the safety invariant on `d`.
        Self(unsafe { f32x4_splat(v) }, d)
    }

    #[inline(always)]
    fn zero(d: Self::Descriptor) -> Self {
        // SAFETY: We know simd128 is available from the safety invariant on `d`.
        Self(unsafe { f32x4_splat(0.0) }, d)
    }

    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[f32]) -> Self {
        assert!(mem.len() >= Self::LEN);
        // SAFETY: we just checked that `mem` has enough space. Moreover, we know simd128 is
        // available from the safety invariant on `d`. v128_load supports unaligned loads.
        Self(unsafe { v128_load(mem.as_ptr().cast()) }, d)
    }

    #[inline(always)]
    fn store(&self, mem: &mut [f32]) {
        assert!(mem.len() >= Self::LEN);
        // SAFETY: we just checked that `mem` has enough space. Moreover, we know simd128 is
        // available from the safety invariant on `d`. v128_store supports unaligned stores.
        unsafe { v128_store(mem.as_mut_ptr().cast(), self.0) }
    }

    #[inline(always)]
    fn store_interleaved_2_uninit(a: Self, b: Self, dest: &mut [MaybeUninit<f32>]) {
        #[target_feature(enable = "simd128")]
        #[inline]
        fn store_interleaved_2_impl(a: v128, b: v128, dest: &mut [MaybeUninit<f32>]) {
            assert!(dest.len() >= 2 * F32VecWasmSimd::LEN);
            // a = [a0, a1, a2, a3], b = [b0, b1, b2, b3]
            // lo = [a0, b0, a1, b1], hi = [a2, b2, a3, b3]
            let lo = i32x4_shuffle::<0, 4, 1, 5>(a, b);
            let hi = i32x4_shuffle::<2, 6, 3, 7>(a, b);
            // SAFETY: `dest` has enough space and writing to `MaybeUninit<f32>` through
            // `*mut v128` is valid. v128_store supports unaligned stores.
            unsafe {
                let dest_ptr = dest.as_mut_ptr().cast::<v128>();
                v128_store(dest_ptr, lo);
                v128_store(dest_ptr.add(1), hi);
            }
        }

        // SAFETY: simd128 is available from the safety invariant on the descriptor.
        unsafe { store_interleaved_2_impl(a.0, b.0, dest) }
    }

    #[inline(always)]
    fn store_interleaved_3_uninit(a: Self, b: Self, c: Self, dest: &mut [MaybeUninit<f32>]) {
        #[target_feature(enable = "simd128")]
        #[inline]
        fn store_interleaved_3_impl(a: v128, b: v128, c: v128, dest: &mut [MaybeUninit<f32>]) {
            assert!(dest.len() >= 3 * F32VecWasmSimd::LEN);
            // Input vectors:
            // a = [a0, a1, a2, a3]
            // b = [b0, b1, b2, b3]
            // c = [c0, c1, c2, c3]
            //
            // Desired interleaved output:
            // out0 = [a0, b0, c0, a1]
            // out1 = [b1, c1, a2, b2]
            // out2 = [c2, a3, b3, c3]

            let p_ab_lo = i32x4_shuffle::<0, 4, 1, 5>(a, b); // [a0, b0, a1, b1]
            let p_ab_hi = i32x4_shuffle::<2, 6, 3, 7>(a, b); // [a2, b2, a3, b3]

            // out0 = [a0, b0, c0, a1]
            let out0 = i32x4_shuffle::<0, 1, 4, 2>(p_ab_lo, c);

            // out1 = [b1, c1, a2, b2]
            // b1=p_ab_lo[3], c1=c[1], a2=p_ab_hi[0], b2=p_ab_hi[1]
            let tmp1 = i32x4_shuffle::<3, 5, 0, 0>(p_ab_lo, c); // [b1, c1, ?, ?]
            let out1 = i32x4_shuffle::<0, 1, 4, 5>(tmp1, p_ab_hi); // [b1, c1, a2, b2]

            // out2 = [c2, a3, b3, c3]
            // c2=c[2], a3=p_ab_hi[2], b3=p_ab_hi[3], c3=c[3]
            let out2 = i32x4_shuffle::<6, 2, 3, 7>(p_ab_hi, c);

            // SAFETY: `dest` has enough space. v128_store supports unaligned stores.
            unsafe {
                let dest_ptr = dest.as_mut_ptr().cast::<v128>();
                v128_store(dest_ptr, out0);
                v128_store(dest_ptr.add(1), out1);
                v128_store(dest_ptr.add(2), out2);
            }
        }

        // SAFETY: simd128 is available from the safety invariant on the descriptor.
        unsafe { store_interleaved_3_impl(a.0, b.0, c.0, dest) }
    }

    #[inline(always)]
    fn store_interleaved_4_uninit(
        a: Self,
        b: Self,
        c: Self,
        d: Self,
        dest: &mut [MaybeUninit<f32>],
    ) {
        #[target_feature(enable = "simd128")]
        #[inline]
        fn store_interleaved_4_impl(
            a: v128,
            b: v128,
            c: v128,
            d: v128,
            dest: &mut [MaybeUninit<f32>],
        ) {
            assert!(dest.len() >= 4 * F32VecWasmSimd::LEN);
            // First interleave pairs: ab and cd
            let ab_lo = i32x4_shuffle::<0, 4, 1, 5>(a, b); // [a0, b0, a1, b1]
            let ab_hi = i32x4_shuffle::<2, 6, 3, 7>(a, b); // [a2, b2, a3, b3]
            let cd_lo = i32x4_shuffle::<0, 4, 1, 5>(c, d); // [c0, d0, c1, d1]
            let cd_hi = i32x4_shuffle::<2, 6, 3, 7>(c, d); // [c2, d2, c3, d3]

            // Then interleave the pairs using 64-bit shuffles
            let out0 = i64x2_shuffle::<0, 2>(ab_lo, cd_lo); // [a0, b0, c0, d0]
            let out1 = i64x2_shuffle::<1, 3>(ab_lo, cd_lo); // [a1, b1, c1, d1]
            let out2 = i64x2_shuffle::<0, 2>(ab_hi, cd_hi); // [a2, b2, c2, d2]
            let out3 = i64x2_shuffle::<1, 3>(ab_hi, cd_hi); // [a3, b3, c3, d3]

            // SAFETY: `dest` has enough space. v128_store supports unaligned stores.
            unsafe {
                let dest_ptr = dest.as_mut_ptr().cast::<v128>();
                v128_store(dest_ptr, out0);
                v128_store(dest_ptr.add(1), out1);
                v128_store(dest_ptr.add(2), out2);
                v128_store(dest_ptr.add(3), out3);
            }
        }

        // SAFETY: simd128 is available from the safety invariant on the descriptor.
        unsafe { store_interleaved_4_impl(a.0, b.0, c.0, d.0, dest) }
    }

    #[inline(always)]
    fn store_interleaved_8(
        a: Self,
        b: Self,
        c: Self,
        d: Self,
        e: Self,
        f: Self,
        g: Self,
        h: Self,
        dest: &mut [f32],
    ) {
        #[target_feature(enable = "simd128")]
        #[inline]
        fn store_interleaved_8_impl(
            a: v128,
            b: v128,
            c: v128,
            d: v128,
            e: v128,
            f: v128,
            g: v128,
            h: v128,
            dest: &mut [f32],
        ) {
            assert!(dest.len() >= 8 * F32VecWasmSimd::LEN);
            let ab_lo = i32x4_shuffle::<0, 4, 1, 5>(a, b);
            let ab_hi = i32x4_shuffle::<2, 6, 3, 7>(a, b);
            let cd_lo = i32x4_shuffle::<0, 4, 1, 5>(c, d);
            let cd_hi = i32x4_shuffle::<2, 6, 3, 7>(c, d);
            let ef_lo = i32x4_shuffle::<0, 4, 1, 5>(e, f);
            let ef_hi = i32x4_shuffle::<2, 6, 3, 7>(e, f);
            let gh_lo = i32x4_shuffle::<0, 4, 1, 5>(g, h);
            let gh_hi = i32x4_shuffle::<2, 6, 3, 7>(g, h);

            let abcd_0 = i64x2_shuffle::<0, 2>(ab_lo, cd_lo);
            let abcd_1 = i64x2_shuffle::<1, 3>(ab_lo, cd_lo);
            let abcd_2 = i64x2_shuffle::<0, 2>(ab_hi, cd_hi);
            let abcd_3 = i64x2_shuffle::<1, 3>(ab_hi, cd_hi);
            let efgh_0 = i64x2_shuffle::<0, 2>(ef_lo, gh_lo);
            let efgh_1 = i64x2_shuffle::<1, 3>(ef_lo, gh_lo);
            let efgh_2 = i64x2_shuffle::<0, 2>(ef_hi, gh_hi);
            let efgh_3 = i64x2_shuffle::<1, 3>(ef_hi, gh_hi);

            // SAFETY: we just checked that dest has enough space. v128_store supports
            // unaligned stores.
            unsafe {
                let ptr = dest.as_mut_ptr().cast::<v128>();
                v128_store(ptr, abcd_0);
                v128_store(ptr.add(1), efgh_0);
                v128_store(ptr.add(2), abcd_1);
                v128_store(ptr.add(3), efgh_1);
                v128_store(ptr.add(4), abcd_2);
                v128_store(ptr.add(5), efgh_2);
                v128_store(ptr.add(6), abcd_3);
                v128_store(ptr.add(7), efgh_3);
            }
        }

        // SAFETY: simd128 is available from the safety invariant on the descriptor stored in `a`.
        unsafe { store_interleaved_8_impl(a.0, b.0, c.0, d.0, e.0, f.0, g.0, h.0, dest) }
    }

    #[inline(always)]
    fn load_deinterleaved_2(d: Self::Descriptor, src: &[f32]) -> (Self, Self) {
        #[target_feature(enable = "simd128")]
        #[inline]
        fn load_deinterleaved_2_impl(src: &[f32]) -> (v128, v128) {
            assert!(src.len() >= 2 * F32VecWasmSimd::LEN);
            // Input: [a0, b0, a1, b1, a2, b2, a3, b3]
            // Output: a = [a0, a1, a2, a3], b = [b0, b1, b2, b3]
            // SAFETY: we just checked that src has enough space. v128_load supports unaligned
            // loads.
            let (in0, in1) = unsafe {
                (
                    v128_load(src.as_ptr().cast()),
                    v128_load(src.as_ptr().add(4).cast()),
                )
            };

            let a = i32x4_shuffle::<0, 2, 4, 6>(in0, in1); // [a0, a1, a2, a3]
            let b = i32x4_shuffle::<1, 3, 5, 7>(in0, in1); // [b0, b1, b2, b3]

            (a, b)
        }

        // SAFETY: simd128 is available from the safety invariant on the descriptor.
        let (a, b) = unsafe { load_deinterleaved_2_impl(src) };
        (Self(a, d), Self(b, d))
    }

    #[inline(always)]
    fn load_deinterleaved_3(d: Self::Descriptor, src: &[f32]) -> (Self, Self, Self) {
        #[target_feature(enable = "simd128")]
        #[inline]
        fn load_deinterleaved_3_impl(src: &[f32]) -> (v128, v128, v128) {
            assert!(src.len() >= 3 * F32VecWasmSimd::LEN);
            // Input: [a0, b0, c0, a1, b1, c1, a2, b2, c2, a3, b3, c3]
            // Output: a = [a0, a1, a2, a3], b = [b0, b1, b2, b3], c = [c0, c1, c2, c3]
            // SAFETY: we just checked that src has enough space.
            let (in0, in1, in2) = unsafe {
                (
                    v128_load(src.as_ptr().cast()),        // [a0, b0, c0, a1]
                    v128_load(src.as_ptr().add(4).cast()), // [b1, c1, a2, b2]
                    v128_load(src.as_ptr().add(8).cast()), // [c2, a3, b3, c3]
                )
            };

            // a: a0=in0[0], a1=in0[3], a2=in1[2], a3=in2[1]
            let a_lo = i32x4_shuffle::<0, 3, 0, 0>(in0, in0); // [a0, a1, ?, ?]
            let a_hi = i32x4_shuffle::<2, 5, 0, 0>(in1, in2); // [a2, a3, ?, ?]
            let a = i32x4_shuffle::<0, 1, 4, 5>(a_lo, a_hi); // [a0, a1, a2, a3]

            // b: b0=in0[1], b1=in1[0], b2=in1[3], b3=in2[2]
            let b_lo = i32x4_shuffle::<1, 4, 0, 0>(in0, in1); // [b0, b1, ?, ?]
            let b_hi = i32x4_shuffle::<3, 6, 0, 0>(in1, in2); // [b2, b3, ?, ?]
            let b = i32x4_shuffle::<0, 1, 4, 5>(b_lo, b_hi); // [b0, b1, b2, b3]

            // c: c0=in0[2], c1=in1[1], c2=in2[0], c3=in2[3]
            let c_lo = i32x4_shuffle::<2, 5, 0, 0>(in0, in1); // [c0, c1, ?, ?]
            let c_hi = i32x4_shuffle::<0, 3, 0, 0>(in2, in2); // [c2, c3, ?, ?]
            let c = i32x4_shuffle::<0, 1, 4, 5>(c_lo, c_hi); // [c0, c1, c2, c3]

            (a, b, c)
        }

        // SAFETY: simd128 is available from the safety invariant on the descriptor.
        let (a, b, c) = unsafe { load_deinterleaved_3_impl(src) };
        (Self(a, d), Self(b, d), Self(c, d))
    }

    #[inline(always)]
    fn load_deinterleaved_4(d: Self::Descriptor, src: &[f32]) -> (Self, Self, Self, Self) {
        #[target_feature(enable = "simd128")]
        #[inline]
        fn load_deinterleaved_4_impl(src: &[f32]) -> (v128, v128, v128, v128) {
            assert!(src.len() >= 4 * F32VecWasmSimd::LEN);
            // Input: [a0, b0, c0, d0, a1, b1, c1, d1, a2, b2, c2, d2, a3, b3, c3, d3]
            // SAFETY: we just checked that src has enough space.
            let (in0, in1, in2, in3) = unsafe {
                (
                    v128_load(src.as_ptr().cast()),
                    v128_load(src.as_ptr().add(4).cast()),
                    v128_load(src.as_ptr().add(8).cast()),
                    v128_load(src.as_ptr().add(12).cast()),
                )
            };

            // 4x4 matrix transpose
            let t0 = i32x4_shuffle::<0, 4, 1, 5>(in0, in1); // [a0, a1, b0, b1]
            let t1 = i32x4_shuffle::<2, 6, 3, 7>(in0, in1); // [c0, c1, d0, d1]
            let t2 = i32x4_shuffle::<0, 4, 1, 5>(in2, in3); // [a2, a3, b2, b3]
            let t3 = i32x4_shuffle::<2, 6, 3, 7>(in2, in3); // [c2, c3, d2, d3]

            let a = i64x2_shuffle::<0, 2>(t0, t2); // [a0, a1, a2, a3]
            let b = i64x2_shuffle::<1, 3>(t0, t2); // [b0, b1, b2, b3]
            let c = i64x2_shuffle::<0, 2>(t1, t3); // [c0, c1, c2, c3]
            let dv = i64x2_shuffle::<1, 3>(t1, t3); // [d0, d1, d2, d3]

            (a, b, c, dv)
        }

        // SAFETY: simd128 is available from the safety invariant on the descriptor.
        let (a, b, c, dv) = unsafe { load_deinterleaved_4_impl(src) };
        (Self(a, d), Self(b, d), Self(c, d), Self(dv, d))
    }

    #[inline(always)]
    fn transpose_square(d: WasmSimdDescriptor, data: &mut [[f32; 4]], stride: usize) {
        #[target_feature(enable = "simd128")]
        #[inline]
        fn transpose4x4f32(d: WasmSimdDescriptor, data: &mut [[f32; 4]], stride: usize) {
            assert!(data.len() > stride * 3);

            let p0 = F32VecWasmSimd::load_array(d, &data[0]).0;
            let p1 = F32VecWasmSimd::load_array(d, &data[1 * stride]).0;
            let p2 = F32VecWasmSimd::load_array(d, &data[2 * stride]).0;
            let p3 = F32VecWasmSimd::load_array(d, &data[3 * stride]).0;

            let q0 = i32x4_shuffle::<0, 4, 1, 5>(p0, p2);
            let q1 = i32x4_shuffle::<0, 4, 1, 5>(p1, p3);
            let q2 = i32x4_shuffle::<2, 6, 3, 7>(p0, p2);
            let q3 = i32x4_shuffle::<2, 6, 3, 7>(p1, p3);

            let r0 = i32x4_shuffle::<0, 4, 1, 5>(q0, q1);
            let r1 = i32x4_shuffle::<2, 6, 3, 7>(q0, q1);
            let r2 = i32x4_shuffle::<0, 4, 1, 5>(q2, q3);
            let r3 = i32x4_shuffle::<2, 6, 3, 7>(q2, q3);

            F32VecWasmSimd(r0, d).store_array(&mut data[0]);
            F32VecWasmSimd(r1, d).store_array(&mut data[1 * stride]);
            F32VecWasmSimd(r2, d).store_array(&mut data[2 * stride]);
            F32VecWasmSimd(r3, d).store_array(&mut data[3 * stride]);
        }

        // SAFETY: the safety invariant on `d` guarantees simd128.
        unsafe {
            transpose4x4f32(d, data, stride);
        }
    }

    crate::impl_f32_array_interface!();

    fn_simd128! {
        fn mul_add(this: F32VecWasmSimd, mul: F32VecWasmSimd, add: F32VecWasmSimd) -> F32VecWasmSimd {
            // WASM SIMD has no FMA instruction; use separate mul + add.
            F32VecWasmSimd(f32x4_add(f32x4_mul(this.0, mul.0), add.0), this.1)
        }

        fn neg_mul_add(this: F32VecWasmSimd, mul: F32VecWasmSimd, add: F32VecWasmSimd) -> F32VecWasmSimd {
            // Computes add - this * mul
            F32VecWasmSimd(f32x4_sub(add.0, f32x4_mul(this.0, mul.0)), this.1)
        }

        fn abs(this: F32VecWasmSimd) -> F32VecWasmSimd {
            F32VecWasmSimd(f32x4_abs(this.0), this.1)
        }

        fn floor(this: F32VecWasmSimd) -> F32VecWasmSimd {
            F32VecWasmSimd(f32x4_floor(this.0), this.1)
        }

        fn sqrt(this: F32VecWasmSimd) -> F32VecWasmSimd {
            F32VecWasmSimd(f32x4_sqrt(this.0), this.1)
        }

        fn neg(this: F32VecWasmSimd) -> F32VecWasmSimd {
            F32VecWasmSimd(f32x4_neg(this.0), this.1)
        }

        fn copysign(this: F32VecWasmSimd, sign: F32VecWasmSimd) -> F32VecWasmSimd {
            let sign_mask = i32x4_splat(i32::MIN); // 0x80000000
            F32VecWasmSimd(v128_bitselect(sign.0, this.0, sign_mask), this.1)
        }

        fn max(this: F32VecWasmSimd, other: F32VecWasmSimd) -> F32VecWasmSimd {
            F32VecWasmSimd(f32x4_max(this.0, other.0), this.1)
        }

        fn min(this: F32VecWasmSimd, other: F32VecWasmSimd) -> F32VecWasmSimd {
            F32VecWasmSimd(f32x4_min(this.0, other.0), this.1)
        }

        fn gt(this: F32VecWasmSimd, other: F32VecWasmSimd) -> MaskWasmSimd {
            MaskWasmSimd(f32x4_gt(this.0, other.0), this.1)
        }

        fn as_i32(this: F32VecWasmSimd) -> I32VecWasmSimd {
            I32VecWasmSimd(i32x4_trunc_sat_f32x4(this.0), this.1)
        }

        fn bitcast_to_i32(this: F32VecWasmSimd) -> I32VecWasmSimd {
            // v128 is untyped; no conversion needed.
            I32VecWasmSimd(this.0, this.1)
        }

        fn round_store_u8(this: F32VecWasmSimd, dest: &mut [u8]) {
            assert!(dest.len() >= F32VecWasmSimd::LEN);
            let rounded = f32x4_nearest(this.0);
            let i32s = i32x4_trunc_sat_f32x4(rounded);
            // Pack i32 -> u16 -> u8 (use same vector twice, take lower half each time)
            let u16s = u16x8_narrow_i32x4(i32s, i32s);
            let u8s = u8x16_narrow_i16x8(u16s, u16s);
            // Store lower 4 bytes
            // SAFETY: we checked dest has enough space. v128_store32_lane supports unaligned
            // stores.
            unsafe {
                v128_store32_lane::<0>(u8s, dest.as_mut_ptr().cast());
            }
        }

        fn round_store_u16(this: F32VecWasmSimd, dest: &mut [u16]) {
            assert!(dest.len() >= F32VecWasmSimd::LEN);
            let rounded = f32x4_nearest(this.0);
            let i32s = i32x4_trunc_sat_f32x4(rounded);
            // Pack i32 -> u16 (use same vector twice, take lower half)
            let u16s = u16x8_narrow_i32x4(i32s, i32s);
            // Store lower 8 bytes (4 u16s)
            // SAFETY: we checked dest has enough space. v128_store64_lane supports unaligned
            // stores.
            unsafe {
                v128_store64_lane::<0>(u16s, dest.as_mut_ptr().cast());
            }
        }

        fn store_f16_bits(this: F32VecWasmSimd, dest: &mut [u16]) {
            assert!(dest.len() >= F32VecWasmSimd::LEN);
            // WASM SIMD has no hardware f16 conversion; use scalar path.
            let mut tmp = [0.0f32; 4];
            // SAFETY: tmp has exactly 4 f32s = 16 bytes, sufficient for a v128 store.
            unsafe { v128_store(tmp.as_mut_ptr().cast(), this.0); }
            for i in 0..4 {
                dest[i] = crate::f16::from_f32(tmp[i]).to_bits();
            }
        }
    }

    #[inline(always)]
    fn load_f16_bits(d: Self::Descriptor, mem: &[u16]) -> Self {
        assert!(mem.len() >= Self::LEN);
        // WASM SIMD has no hardware f16 conversion; use scalar path.
        let mut result = [0.0f32; 4];
        for i in 0..4 {
            result[i] = crate::f16::from_bits(mem[i]).to_f32();
        }
        Self::load(d, &result)
    }

    #[inline(always)]
    fn prepare_table_bf16_8(_d: WasmSimdDescriptor, table: &[f32; 8]) -> Bf16Table8WasmSimd {
        #[target_feature(enable = "simd128")]
        #[inline]
        fn prepare_impl(table: &[f32; 8]) -> v128 {
            // Convert f32 table to BF16 packed in 128 bits (16 bytes for 8 entries)
            // BF16 is the high 16 bits of f32
            // SAFETY: table has exactly 8 elements and simd128 is available from target_feature.
            let (table_lo, table_hi) = unsafe {
                (
                    v128_load(table.as_ptr().cast()),
                    v128_load(table.as_ptr().add(4).cast()),
                )
            };

            // Shift right by 16 to move high 16 bits to low 16 bits of each lane
            let bf16_lo = u32x4_shr(table_lo, 16);
            let bf16_hi = u32x4_shr(table_hi, 16);

            // Narrow u32 lanes to u16 using shuffle to extract low 2 bytes of each lane
            // table_lo = [w0, x0, y0, z0, w1, x1, y1, z1, w2, x2, y2, z2, w3, x3, y3, z3]
            // After shr16, bf16 values are in bytes [0,1] of each 4-byte lane
            // We want bytes [0,1,4,5,8,9,12,13] from each vector
            i8x16_shuffle::<0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29>(
                bf16_lo, bf16_hi,
            )
        }
        // SAFETY: simd128 is available from the safety invariant on the descriptor.
        Bf16Table8WasmSimd(unsafe { prepare_impl(table) })
    }

    #[inline(always)]
    fn table_lookup_bf16_8(
        d: WasmSimdDescriptor,
        table: Bf16Table8WasmSimd,
        indices: I32VecWasmSimd,
    ) -> Self {
        #[target_feature(enable = "simd128")]
        #[inline]
        fn lookup_impl(bf16_table: v128, indices: v128) -> v128 {
            // Build shuffle mask efficiently using arithmetic on 32-bit indices.
            // For each index i (0-7), we need to select bytes [2*i, 2*i+1] from bf16_table
            // and place them in the high 16 bits of each 32-bit f32 lane (bytes 2,3),
            // with bytes 0,1 set to zero (using out-of-range index which gives 0 in
            // i8x16_swizzle).
            //
            // Output byte pattern per lane (little-endian): [0x80, 0x80, 2*i, 2*i+1]
            // As a 32-bit value: 0x80 | (0x80 << 8) | (2*i << 16) | ((2*i+1) << 24)
            //                  = (i << 17) | (i << 25) | 0x01008080
            let shl17 = i32x4_shl(indices, 17);
            let shl25 = i32x4_shl(indices, 25);
            let base = i32x4_splat(0x01008080u32 as i32);
            let shuffle_mask = v128_or(v128_or(shl17, shl25), base);

            // Perform the table lookup (out of range indices give 0).
            // Result has bf16 in high 16 bits of each 32-bit lane = valid f32.
            i8x16_swizzle(bf16_table, shuffle_mask)
        }
        // SAFETY: simd128 is available from the safety invariant on the descriptor.
        F32VecWasmSimd(unsafe { lookup_impl(table.0, indices.0) }, d)
    }
}

impl Add<F32VecWasmSimd> for F32VecWasmSimd {
    type Output = Self;
    fn_simd128! {
        fn add(this: F32VecWasmSimd, rhs: F32VecWasmSimd) -> F32VecWasmSimd {
            F32VecWasmSimd(f32x4_add(this.0, rhs.0), this.1)
        }
    }
}

impl Sub<F32VecWasmSimd> for F32VecWasmSimd {
    type Output = Self;
    fn_simd128! {
        fn sub(this: F32VecWasmSimd, rhs: F32VecWasmSimd) -> F32VecWasmSimd {
            F32VecWasmSimd(f32x4_sub(this.0, rhs.0), this.1)
        }
    }
}

impl Mul<F32VecWasmSimd> for F32VecWasmSimd {
    type Output = Self;
    fn_simd128! {
        fn mul(this: F32VecWasmSimd, rhs: F32VecWasmSimd) -> F32VecWasmSimd {
            F32VecWasmSimd(f32x4_mul(this.0, rhs.0), this.1)
        }
    }
}

impl Div<F32VecWasmSimd> for F32VecWasmSimd {
    type Output = Self;
    fn_simd128! {
        fn div(this: F32VecWasmSimd, rhs: F32VecWasmSimd) -> F32VecWasmSimd {
            F32VecWasmSimd(f32x4_div(this.0, rhs.0), this.1)
        }
    }
}

impl AddAssign<F32VecWasmSimd> for F32VecWasmSimd {
    fn_simd128! {
        fn add_assign(this: &mut F32VecWasmSimd, rhs: F32VecWasmSimd) {
            this.0 = f32x4_add(this.0, rhs.0);
        }
    }
}

impl SubAssign<F32VecWasmSimd> for F32VecWasmSimd {
    fn_simd128! {
        fn sub_assign(this: &mut F32VecWasmSimd, rhs: F32VecWasmSimd) {
            this.0 = f32x4_sub(this.0, rhs.0);
        }
    }
}

impl MulAssign<F32VecWasmSimd> for F32VecWasmSimd {
    fn_simd128! {
        fn mul_assign(this: &mut F32VecWasmSimd, rhs: F32VecWasmSimd) {
            this.0 = f32x4_mul(this.0, rhs.0);
        }
    }
}

impl DivAssign<F32VecWasmSimd> for F32VecWasmSimd {
    fn_simd128! {
        fn div_assign(this: &mut F32VecWasmSimd, rhs: F32VecWasmSimd) {
            this.0 = f32x4_div(this.0, rhs.0);
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct I32VecWasmSimd(v128, WasmSimdDescriptor);

impl I32SimdVec for I32VecWasmSimd {
    type Descriptor = WasmSimdDescriptor;

    const LEN: usize = 4;

    #[inline(always)]
    fn splat(d: Self::Descriptor, v: i32) -> Self {
        // SAFETY: We know simd128 is available from the safety invariant on `d`.
        Self(unsafe { i32x4_splat(v) }, d)
    }

    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[i32]) -> Self {
        assert!(mem.len() >= Self::LEN);
        // SAFETY: we just checked that `mem` has enough space. Moreover, we know simd128 is
        // available from the safety invariant on `d`.
        Self(unsafe { v128_load(mem.as_ptr().cast()) }, d)
    }

    #[inline(always)]
    fn store(&self, mem: &mut [i32]) {
        assert!(mem.len() >= Self::LEN);
        // SAFETY: we just checked that `mem` has enough space. Moreover, we know simd128 is
        // available from the safety invariant on `self.1`.
        unsafe { v128_store(mem.as_mut_ptr().cast(), self.0) }
    }

    fn_simd128! {
        fn abs(this: I32VecWasmSimd) -> I32VecWasmSimd {
            I32VecWasmSimd(i32x4_abs(this.0), this.1)
        }

        fn as_f32(this: I32VecWasmSimd) -> F32VecWasmSimd {
            F32VecWasmSimd(f32x4_convert_i32x4(this.0), this.1)
        }

        fn bitcast_to_f32(this: I32VecWasmSimd) -> F32VecWasmSimd {
            // v128 is untyped; no conversion needed.
            F32VecWasmSimd(this.0, this.1)
        }

        fn bitcast_to_u32(this: I32VecWasmSimd) -> U32VecWasmSimd {
            // v128 is untyped; no conversion needed.
            U32VecWasmSimd(this.0, this.1)
        }

        fn gt(this: I32VecWasmSimd, other: I32VecWasmSimd) -> MaskWasmSimd {
            MaskWasmSimd(i32x4_gt(this.0, other.0), this.1)
        }

        fn lt_zero(this: I32VecWasmSimd) -> MaskWasmSimd {
            MaskWasmSimd(i32x4_lt(this.0, i32x4_splat(0)), this.1)
        }

        fn eq(this: I32VecWasmSimd, other: I32VecWasmSimd) -> MaskWasmSimd {
            MaskWasmSimd(i32x4_eq(this.0, other.0), this.1)
        }

        fn eq_zero(this: I32VecWasmSimd) -> MaskWasmSimd {
            MaskWasmSimd(i32x4_eq(this.0, i32x4_splat(0)), this.1)
        }

        fn mul_wide_take_high(this: I32VecWasmSimd, rhs: I32VecWasmSimd) -> I32VecWasmSimd {
            // Widening multiply: i32 * i32 -> i64, then take high 32 bits
            let lo = i64x2_extmul_low_i32x4(this.0, rhs.0);  // lanes 0,1 -> i64 results
            let hi = i64x2_extmul_high_i32x4(this.0, rhs.0); // lanes 2,3 -> i64 results
            // Extract high 32 bits from each i64 lane
            // lo = [lo0_lo, lo0_hi, lo1_lo, lo1_hi] as bytes
            // hi = [hi0_lo, hi0_hi, hi1_lo, hi1_hi] as bytes
            // We want [lo0_hi, lo1_hi, hi0_hi, hi1_hi] in i32 lanes
            // That's bytes [4,5,6,7, 12,13,14,15] from lo and [4,5,6,7, 12,13,14,15] from hi
            let result = i8x16_shuffle::<
                4, 5, 6, 7, 12, 13, 14, 15,
                20, 21, 22, 23, 28, 29, 30, 31,
            >(lo, hi);
            I32VecWasmSimd(result, this.1)
        }
    }

    #[inline(always)]
    fn shl<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        // SAFETY: We know simd128 is available from the safety invariant on `self.1`.
        unsafe { Self(i32x4_shl(self.0, AMOUNT_U), self.1) }
    }

    #[inline(always)]
    fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        // SAFETY: We know simd128 is available from the safety invariant on `self.1`.
        unsafe { Self(i32x4_shr(self.0, AMOUNT_U), self.1) }
    }

    #[inline(always)]
    fn store_u16(self, dest: &mut [u16]) {
        assert!(dest.len() >= Self::LEN);
        #[target_feature(enable = "simd128")]
        #[inline]
        fn store_u16_impl(v: v128, dest: &mut [u16]) {
            // Gather the lower 2 bytes of each 4-byte lane (truncation, not saturation)
            let narrowed =
                i8x16_shuffle::<0, 1, 4, 5, 8, 9, 12, 13, 0, 1, 4, 5, 8, 9, 12, 13>(v, v);
            // Store lower 8 bytes (4 u16s)
            // SAFETY: dest has enough space. v128_store64_lane supports unaligned stores.
            unsafe {
                v128_store64_lane::<0>(narrowed, dest.as_mut_ptr().cast());
            }
        }
        // SAFETY: simd128 is available from the safety invariant on `self.1`.
        unsafe { store_u16_impl(self.0, dest) }
    }

    #[inline(always)]
    fn store_u8(self, dest: &mut [u8]) {
        assert!(dest.len() >= Self::LEN);
        #[target_feature(enable = "simd128")]
        #[inline]
        fn store_u8_impl(v: v128, dest: &mut [u8]) {
            // Gather byte 0 of each 4-byte lane (truncation, not saturation)
            let narrowed =
                i8x16_shuffle::<0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12>(v, v);
            // Store lower 4 bytes
            // SAFETY: dest has enough space. v128_store32_lane supports unaligned stores.
            unsafe {
                v128_store32_lane::<0>(narrowed, dest.as_mut_ptr().cast());
            }
        }
        // SAFETY: simd128 is available from the safety invariant on `self.1`.
        unsafe { store_u8_impl(self.0, dest) }
    }
}

impl Add<I32VecWasmSimd> for I32VecWasmSimd {
    type Output = I32VecWasmSimd;
    fn_simd128! {
        fn add(this: I32VecWasmSimd, rhs: I32VecWasmSimd) -> I32VecWasmSimd {
            I32VecWasmSimd(i32x4_add(this.0, rhs.0), this.1)
        }
    }
}

impl Sub<I32VecWasmSimd> for I32VecWasmSimd {
    type Output = I32VecWasmSimd;
    fn_simd128! {
        fn sub(this: I32VecWasmSimd, rhs: I32VecWasmSimd) -> I32VecWasmSimd {
            I32VecWasmSimd(i32x4_sub(this.0, rhs.0), this.1)
        }
    }
}

impl Mul<I32VecWasmSimd> for I32VecWasmSimd {
    type Output = I32VecWasmSimd;
    fn_simd128! {
        fn mul(this: I32VecWasmSimd, rhs: I32VecWasmSimd) -> I32VecWasmSimd {
            I32VecWasmSimd(i32x4_mul(this.0, rhs.0), this.1)
        }
    }
}

impl Neg for I32VecWasmSimd {
    type Output = I32VecWasmSimd;
    fn_simd128! {
        fn neg(this: I32VecWasmSimd) -> I32VecWasmSimd {
            I32VecWasmSimd(i32x4_neg(this.0), this.1)
        }
    }
}

impl BitAnd<I32VecWasmSimd> for I32VecWasmSimd {
    type Output = I32VecWasmSimd;
    fn_simd128! {
        fn bitand(this: I32VecWasmSimd, rhs: I32VecWasmSimd) -> I32VecWasmSimd {
            I32VecWasmSimd(v128_and(this.0, rhs.0), this.1)
        }
    }
}

impl BitOr<I32VecWasmSimd> for I32VecWasmSimd {
    type Output = I32VecWasmSimd;
    fn_simd128! {
        fn bitor(this: I32VecWasmSimd, rhs: I32VecWasmSimd) -> I32VecWasmSimd {
            I32VecWasmSimd(v128_or(this.0, rhs.0), this.1)
        }
    }
}

impl BitXor<I32VecWasmSimd> for I32VecWasmSimd {
    type Output = I32VecWasmSimd;
    fn_simd128! {
        fn bitxor(this: I32VecWasmSimd, rhs: I32VecWasmSimd) -> I32VecWasmSimd {
            I32VecWasmSimd(v128_xor(this.0, rhs.0), this.1)
        }
    }
}

impl AddAssign<I32VecWasmSimd> for I32VecWasmSimd {
    fn_simd128! {
        fn add_assign(this: &mut I32VecWasmSimd, rhs: I32VecWasmSimd) {
            this.0 = i32x4_add(this.0, rhs.0)
        }
    }
}

impl SubAssign<I32VecWasmSimd> for I32VecWasmSimd {
    fn_simd128! {
        fn sub_assign(this: &mut I32VecWasmSimd, rhs: I32VecWasmSimd) {
            this.0 = i32x4_sub(this.0, rhs.0)
        }
    }
}

impl MulAssign<I32VecWasmSimd> for I32VecWasmSimd {
    fn_simd128! {
        fn mul_assign(this: &mut I32VecWasmSimd, rhs: I32VecWasmSimd) {
            this.0 = i32x4_mul(this.0, rhs.0)
        }
    }
}

impl BitAndAssign<I32VecWasmSimd> for I32VecWasmSimd {
    fn_simd128! {
        fn bitand_assign(this: &mut I32VecWasmSimd, rhs: I32VecWasmSimd) {
            this.0 = v128_and(this.0, rhs.0);
        }
    }
}

impl BitOrAssign<I32VecWasmSimd> for I32VecWasmSimd {
    fn_simd128! {
        fn bitor_assign(this: &mut I32VecWasmSimd, rhs: I32VecWasmSimd) {
            this.0 = v128_or(this.0, rhs.0);
        }
    }
}

impl BitXorAssign<I32VecWasmSimd> for I32VecWasmSimd {
    fn_simd128! {
        fn bitxor_assign(this: &mut I32VecWasmSimd, rhs: I32VecWasmSimd) {
            this.0 = v128_xor(this.0, rhs.0);
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U32VecWasmSimd(v128, WasmSimdDescriptor);

impl U32SimdVec for U32VecWasmSimd {
    type Descriptor = WasmSimdDescriptor;

    const LEN: usize = 4;

    fn_simd128! {
        fn bitcast_to_i32(this: U32VecWasmSimd) -> I32VecWasmSimd {
            // v128 is untyped; no conversion needed.
            I32VecWasmSimd(this.0, this.1)
        }
    }

    #[inline(always)]
    fn shr<const AMOUNT_U: u32, const AMOUNT_I: i32>(self) -> Self {
        // SAFETY: We know simd128 is available from the safety invariant on `self.1`.
        unsafe { Self(u32x4_shr(self.0, AMOUNT_U), self.1) }
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U8VecWasmSimd(v128, WasmSimdDescriptor);

// SAFETY: The methods in this implementation that write to `MaybeUninit` (store_interleaved_*)
// ensure that they write valid data to the output slice without reading uninitialized memory.
unsafe impl U8SimdVec for U8VecWasmSimd {
    type Descriptor = WasmSimdDescriptor;
    const LEN: usize = 16;

    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[u8]) -> Self {
        assert!(mem.len() >= Self::LEN);
        // SAFETY: we just checked that `mem` has enough space. Moreover, we know simd128 is
        // available from the safety invariant on `d`. v128_load supports unaligned loads.
        Self(unsafe { v128_load(mem.as_ptr().cast()) }, d)
    }

    #[inline(always)]
    fn splat(d: Self::Descriptor, v: u8) -> Self {
        // SAFETY: We know simd128 is available from the safety invariant on `d`.
        Self(unsafe { u8x16_splat(v) }, d)
    }

    #[inline(always)]
    fn store(&self, mem: &mut [u8]) {
        assert!(mem.len() >= Self::LEN);
        // SAFETY: we just checked that `mem` has enough space. Moreover, we know simd128 is
        // available from the safety invariant on `d`. v128_store supports unaligned stores.
        unsafe { v128_store(mem.as_mut_ptr().cast(), self.0) }
    }

    #[inline(always)]
    fn store_interleaved_2_uninit(a: Self, b: Self, dest: &mut [MaybeUninit<u8>]) {
        assert!(dest.len() >= 2 * Self::LEN);
        #[target_feature(enable = "simd128")]
        #[inline]
        fn interleave_2_impl(a: v128, b: v128, dest: &mut [MaybeUninit<u8>]) {
            // Interleave bytes: [a0,b0,a1,b1,...,a7,b7] and [a8,b8,...,a15,b15]
            let lo = i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(a, b);
            let hi =
                i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(a, b);
            // SAFETY: dest has enough space. v128_store supports unaligned stores.
            unsafe {
                let dest_ptr = dest.as_mut_ptr().cast::<v128>();
                v128_store(dest_ptr, lo);
                v128_store(dest_ptr.add(1), hi);
            }
        }
        // SAFETY: simd128 is available from the safety invariant on the descriptor stored in `a`.
        unsafe { interleave_2_impl(a.0, b.0, dest) }
    }

    #[inline(always)]
    fn store_interleaved_3_uninit(a: Self, b: Self, c: Self, dest: &mut [MaybeUninit<u8>]) {
        assert!(dest.len() >= 3 * Self::LEN);
        #[target_feature(enable = "simd128")]
        #[inline]
        fn interleave_3_impl(a: v128, b: v128, c: v128, dest: &mut [MaybeUninit<u8>]) {
            // Interleave 3 channels of 16 bytes each into 48 bytes:
            // [a0,b0,c0, a1,b1,c1, ..., a15,b15,c15]
            //
            // Stage 1: interleave a and b in byte pairs
            // ab_even = [a0,b0,a1,b1,a2,b2,a3,b3,a4,b4,a5,b5,a6,b6,a7,b7]
            let ab_even =
                i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(a, b);
            // ab_odd = [a8,b8,a9,b9,a10,b10,a11,b11,a12,b12,a13,b13,a14,b14,a15,b15]
            let ab_odd =
                i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(a, b);

            // Stage 2: weave c into every 3rd position
            // out0 = [a0,b0,c0, a1,b1,c1, a2,b2,c2, a3,b3,c3, a4,b4,c4, a5]
            let out0 = i8x16_shuffle::<
                0,
                1,
                16, // a0, b0, c0
                2,
                3,
                17, // a1, b1, c1
                4,
                5,
                18, // a2, b2, c2
                6,
                7,
                19, // a3, b3, c3
                8,
                9,
                20, // a4, b4, c4
                10, // a5
            >(ab_even, c);

            // out1 = [b5,c5, a6,b6,c6, a7,b7,c7, a8,b8,c8, a9,b9,c9, a10,b10]
            // First 8 bytes from ab_even + c, last 8 from ab_odd + c
            let tmp1 = i8x16_shuffle::<
                11,
                21, // b5, c5
                12,
                13,
                22, // a6, b6, c6
                14,
                15,
                23, // a7, b7, c7
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            >(ab_even, c);
            let tmp2 = i8x16_shuffle::<
                0,
                1,
                24, // a8, b8, c8
                2,
                3,
                25, // a9, b9, c9
                4,
                5, // a10, b10
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            >(ab_odd, c);
            let out1 =
                i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(tmp1, tmp2);

            // out2 = [c10, a11,b11,c11, a12,b12,c12, a13,b13,c13, a14,b14,c14, a15,b15,c15]
            let out2 = i8x16_shuffle::<
                26, // c10
                6,
                7,
                27, // a11, b11, c11
                8,
                9,
                28, // a12, b12, c12
                10,
                11,
                29, // a13, b13, c13
                12,
                13,
                30, // a14, b14, c14
                14,
                15,
                31, // a15, b15, c15
            >(ab_odd, c);

            // SAFETY: dest has enough space. v128_store supports unaligned stores.
            unsafe {
                let dest_ptr = dest.as_mut_ptr().cast::<v128>();
                v128_store(dest_ptr, out0);
                v128_store(dest_ptr.add(1), out1);
                v128_store(dest_ptr.add(2), out2);
            }
        }
        // SAFETY: simd128 is available from the safety invariant on the descriptor stored in `a`.
        unsafe { interleave_3_impl(a.0, b.0, c.0, dest) }
    }

    #[inline(always)]
    fn store_interleaved_4_uninit(
        a: Self,
        b: Self,
        c: Self,
        d: Self,
        dest: &mut [MaybeUninit<u8>],
    ) {
        assert!(dest.len() >= 4 * Self::LEN);
        #[target_feature(enable = "simd128")]
        #[inline]
        fn interleave_4_impl(a: v128, b: v128, c: v128, d: v128, dest: &mut [MaybeUninit<u8>]) {
            // Interleave 4 channels: [a0,b0,c0,d0, a1,b1,c1,d1, ..., a15,b15,c15,d15]
            // Stage 1: interleave ab and cd pairs
            let ab_lo =
                i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(a, b);
            let ab_hi =
                i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(a, b);
            let cd_lo =
                i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(c, d);
            let cd_hi =
                i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(c, d);

            // Stage 2: interleave abcd
            // ab_lo = [a0,b0,a1,b1,...,a7,b7], cd_lo = [c0,d0,c1,d1,...,c7,d7]
            let out0 = i8x16_shuffle::<0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23>(
                ab_lo, cd_lo,
            );
            let out1 = i8x16_shuffle::<8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31>(
                ab_lo, cd_lo,
            );
            let out2 = i8x16_shuffle::<0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23>(
                ab_hi, cd_hi,
            );
            let out3 = i8x16_shuffle::<8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31>(
                ab_hi, cd_hi,
            );

            // SAFETY: dest has enough space. v128_store supports unaligned stores.
            unsafe {
                let dest_ptr = dest.as_mut_ptr().cast::<v128>();
                v128_store(dest_ptr, out0);
                v128_store(dest_ptr.add(1), out1);
                v128_store(dest_ptr.add(2), out2);
                v128_store(dest_ptr.add(3), out3);
            }
        }
        // SAFETY: simd128 is available from the safety invariant on the descriptor stored in `a`.
        unsafe { interleave_4_impl(a.0, b.0, c.0, d.0, dest) }
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct U16VecWasmSimd(v128, WasmSimdDescriptor);

// SAFETY: The methods in this implementation that write to `MaybeUninit` (store_interleaved_*)
// ensure that they write valid data to the output slice without reading uninitialized memory.
unsafe impl U16SimdVec for U16VecWasmSimd {
    type Descriptor = WasmSimdDescriptor;
    const LEN: usize = 8;

    #[inline(always)]
    fn load(d: Self::Descriptor, mem: &[u16]) -> Self {
        assert!(mem.len() >= Self::LEN);
        // SAFETY: we just checked that `mem` has enough space. Moreover, we know simd128 is
        // available from the safety invariant on `d`. v128_load supports unaligned loads.
        Self(unsafe { v128_load(mem.as_ptr().cast()) }, d)
    }

    #[inline(always)]
    fn splat(d: Self::Descriptor, v: u16) -> Self {
        // SAFETY: We know simd128 is available from the safety invariant on `d`.
        Self(unsafe { u16x8_splat(v) }, d)
    }

    #[inline(always)]
    fn store(&self, mem: &mut [u16]) {
        assert!(mem.len() >= Self::LEN);
        // SAFETY: we just checked that `mem` has enough space. Moreover, we know simd128 is
        // available from the safety invariant on `d`. v128_store supports unaligned stores.
        unsafe { v128_store(mem.as_mut_ptr().cast(), self.0) }
    }

    #[inline(always)]
    fn store_interleaved_2_uninit(a: Self, b: Self, dest: &mut [MaybeUninit<u16>]) {
        assert!(dest.len() >= 2 * Self::LEN);
        #[target_feature(enable = "simd128")]
        #[inline]
        fn interleave_2_impl(a: v128, b: v128, dest: &mut [MaybeUninit<u16>]) {
            // Interleave u16 pairs: [a0,b0,a1,b1,...] using byte-level shuffles
            let lo = i8x16_shuffle::<0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23>(a, b);
            let hi =
                i8x16_shuffle::<8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31>(a, b);
            // SAFETY: dest has enough space. v128_store supports unaligned stores.
            unsafe {
                let dest_ptr = dest.as_mut_ptr().cast::<v128>();
                v128_store(dest_ptr, lo);
                v128_store(dest_ptr.add(1), hi);
            }
        }
        // SAFETY: simd128 is available from the safety invariant on the descriptor stored in `a`.
        unsafe { interleave_2_impl(a.0, b.0, dest) }
    }

    #[inline(always)]
    fn store_interleaved_3_uninit(a: Self, b: Self, c: Self, dest: &mut [MaybeUninit<u16>]) {
        assert!(dest.len() >= 3 * Self::LEN);
        #[target_feature(enable = "simd128")]
        #[inline]
        fn interleave_3_impl(a: v128, b: v128, c: v128, dest: &mut [MaybeUninit<u16>]) {
            // Interleave 3 channels of 8 u16 each into 24 u16s (48 bytes, 3 v128s)
            // out0 = [a0,b0,c0, a1,b1,c1, a2,b2] (8 u16s)
            // out1 = [c2, a3,b3,c3, a4,b4,c4, a5] (8 u16s)
            // out2 = [b5,c5, a6,b6,c6, a7,b7,c7] (8 u16s)

            // First interleave a and b
            let ab_lo =
                i8x16_shuffle::<0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23>(a, b);
            // ab_lo = [a0,b0,a1,b1,a2,b2,a3,b3] as u16 pairs

            let ab_hi =
                i8x16_shuffle::<8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31>(a, b);
            // ab_hi = [a4,b4,a5,b5,a6,b6,a7,b7] as u16 pairs

            // out0: a0,b0,c0, a1,b1,c1, a2,b2
            // ab_lo bytes: [a0L,a0H, b0L,b0H, a1L,a1H, b1L,b1H, a2L,a2H, b2L,b2H, a3L,a3H, b3L,b3H]
            //  indices:      0   1    2   3    4   5    6   7    8   9   10  11   12  13   14  15
            // c bytes: [c0L,c0H, c1L,c1H, c2L,c2H, c3L,c3H, ...]
            //  indices: 16  17   18  19   20  21   22  23
            let out0 = i8x16_shuffle::<
                0,
                1,
                2,
                3,
                16,
                17, // a0, b0, c0
                4,
                5,
                6,
                7,
                18,
                19, // a1, b1, c1
                8,
                9,
                10,
                11, // a2, b2
            >(ab_lo, c);

            // out1: c2, a3,b3,c3, a4,b4,c4, a5
            let tmp1 = i8x16_shuffle::<
                20,
                21, // c2
                12,
                13,
                14,
                15,
                22,
                23, // a3, b3, c3
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0, // padding
            >(ab_lo, c);
            let tmp2 = i8x16_shuffle::<
                0,
                1,
                2,
                3,
                24,
                25, // a4, b4, c4
                4,
                5, // a5
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0, // padding
            >(ab_hi, c);
            let out1 =
                i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(tmp1, tmp2);

            // out2: b5,c5, a6,b6,c6, a7,b7,c7
            let out2 = i8x16_shuffle::<
                6,
                7,
                26,
                27, // b5, c5
                8,
                9,
                10,
                11,
                28,
                29, // a6, b6, c6
                12,
                13,
                14,
                15,
                30,
                31, // a7, b7, c7
            >(ab_hi, c);

            // SAFETY: dest has enough space. v128_store supports unaligned stores.
            unsafe {
                let dest_ptr = dest.as_mut_ptr().cast::<v128>();
                v128_store(dest_ptr, out0);
                v128_store(dest_ptr.add(1), out1);
                v128_store(dest_ptr.add(2), out2);
            }
        }
        // SAFETY: simd128 is available from the safety invariant on the descriptor stored in `a`.
        unsafe { interleave_3_impl(a.0, b.0, c.0, dest) }
    }

    #[inline(always)]
    fn store_interleaved_4_uninit(
        a: Self,
        b: Self,
        c: Self,
        d: Self,
        dest: &mut [MaybeUninit<u16>],
    ) {
        assert!(dest.len() >= 4 * Self::LEN);
        #[target_feature(enable = "simd128")]
        #[inline]
        fn interleave_4_impl(a: v128, b: v128, c: v128, d: v128, dest: &mut [MaybeUninit<u16>]) {
            // Interleave 4 channels of 8 u16 each into 32 u16s (64 bytes, 4 v128s)
            // Stage 1: interleave ab and cd pairs (u16 level via byte shuffles)
            let ab_lo =
                i8x16_shuffle::<0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23>(a, b);
            let ab_hi =
                i8x16_shuffle::<8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31>(a, b);
            let cd_lo =
                i8x16_shuffle::<0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23>(c, d);
            let cd_hi =
                i8x16_shuffle::<8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31>(c, d);

            // Stage 2: interleave abcd pairs (each pair is 4 bytes = 2 u16)
            let out0 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23>(
                ab_lo, cd_lo,
            );
            let out1 = i8x16_shuffle::<8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31>(
                ab_lo, cd_lo,
            );
            let out2 = i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23>(
                ab_hi, cd_hi,
            );
            let out3 = i8x16_shuffle::<8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31>(
                ab_hi, cd_hi,
            );

            // SAFETY: dest has enough space. v128_store supports unaligned stores.
            unsafe {
                let dest_ptr = dest.as_mut_ptr().cast::<v128>();
                v128_store(dest_ptr, out0);
                v128_store(dest_ptr.add(1), out1);
                v128_store(dest_ptr.add(2), out2);
                v128_store(dest_ptr.add(3), out3);
            }
        }
        // SAFETY: simd128 is available from the safety invariant on the descriptor stored in `a`.
        unsafe { interleave_4_impl(a.0, b.0, c.0, d.0, dest) }
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct MaskWasmSimd(v128, WasmSimdDescriptor);

impl SimdMask for MaskWasmSimd {
    type Descriptor = WasmSimdDescriptor;

    fn_simd128! {
        fn if_then_else_f32(
            this: MaskWasmSimd,
            if_true: F32VecWasmSimd,
            if_false: F32VecWasmSimd,
        ) -> F32VecWasmSimd {
            F32VecWasmSimd(v128_bitselect(if_true.0, if_false.0, this.0), this.1)
        }

        fn if_then_else_i32(
            this: MaskWasmSimd,
            if_true: I32VecWasmSimd,
            if_false: I32VecWasmSimd,
        ) -> I32VecWasmSimd {
            I32VecWasmSimd(v128_bitselect(if_true.0, if_false.0, this.0), this.1)
        }

        fn maskz_i32(this: MaskWasmSimd, v: I32VecWasmSimd) -> I32VecWasmSimd {
            // v & ~self: WASM v128_andnot(a, b) = a & ~b
            I32VecWasmSimd(v128_andnot(v.0, this.0), this.1)
        }

        fn andnot(this: MaskWasmSimd, rhs: MaskWasmSimd) -> MaskWasmSimd {
            // !self & rhs: WASM v128_andnot(a, b) = a & ~b
            MaskWasmSimd(v128_andnot(rhs.0, this.0), this.1)
        }

        fn all(this: MaskWasmSimd) -> bool {
            i32x4_all_true(this.0)
        }
    }
}

impl BitAnd<MaskWasmSimd> for MaskWasmSimd {
    type Output = MaskWasmSimd;
    fn_simd128! {
        fn bitand(this: MaskWasmSimd, rhs: MaskWasmSimd) -> MaskWasmSimd {
            MaskWasmSimd(v128_and(this.0, rhs.0), this.1)
        }
    }
}

impl BitOr<MaskWasmSimd> for MaskWasmSimd {
    type Output = MaskWasmSimd;
    fn_simd128! {
        fn bitor(this: MaskWasmSimd, rhs: MaskWasmSimd) -> MaskWasmSimd {
            MaskWasmSimd(v128_or(this.0, rhs.0), this.1)
        }
    }
}

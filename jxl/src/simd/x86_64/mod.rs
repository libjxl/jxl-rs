// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(unsafe_code)]

pub(super) mod avx;
pub(super) mod avx512;

macro_rules! simd_function {
    (
        $dname:ident,
        $descr:ident: $descr_ty:ident,
        $pub:vis fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty )? $body: block
    ) => {
        #[inline(always)]
        $pub fn $name<$descr_ty: crate::simd::SimdDescriptor>($descr: $descr_ty, $($arg: $ty),*) $(-> $ret)? $body
        #[allow(unsafe_code)]
        $pub fn $dname($($arg: $ty),*) $(-> $ret)? {
            use crate::simd::SimdDescriptor;
            if let Some(d) = crate::simd::Avx512Descriptor::new() {
                #[target_feature(enable = "avx512f")]
                fn inner(d: crate::simd::Avx512Descriptor, $($arg: $ty),*) $(-> $ret)? {
                    $name(d, $($arg),*)
                }
                // SAFETY: we just checked for avx512f.
                return unsafe { inner(d, $($arg),*) };
            }
            if let Some(d) = crate::simd::AvxDescriptor::new() {
                #[target_feature(enable = "avx2,fma")]
                fn inner(d: crate::simd::AvxDescriptor, $($arg: $ty),*) $(-> $ret)? {
                    $name(d, $($arg),*)
                }
                // SAFETY: we just checked for avx2 and fma.
                return unsafe { inner(d, $($arg),*) };
            }
            $name(crate::simd::ScalarDescriptor::new().unwrap(), $($arg),*)
        }
    };
}

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
            #[test]
            fn [<$name _avx>]() {
                use crate::simd::SimdDescriptor;
                let Some(d) = crate::simd::AvxDescriptor::new() else { return; };
                $name(d)
            }
            #[test]
            fn [<$name _avx512>]() {
                use crate::simd::SimdDescriptor;
                let Some(d) = crate::simd::Avx512Descriptor::new() else { return; };
                $name(d)
            }
        }
    };
}

#[allow(unused_imports)]
pub(crate) use test_all_instruction_sets;

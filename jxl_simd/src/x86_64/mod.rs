// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(unsafe_code)]
#![allow(clippy::identity_op)]

pub(super) mod avx;
pub(super) mod avx512;
pub(super) mod sse42;

#[macro_export]
macro_rules! simd_function {
    (
        $dname:ident,
        $descr:ident: $descr_ty:ident,
        $(#[$($attr:meta)*])*
        $pub:vis fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty )? $body: block
    ) => {
        #[inline(always)]
        $(#[$($attr)*])*
        $pub fn $name<$descr_ty: $crate::SimdDescriptor>($descr: $descr_ty, $($arg: $ty),*) $(-> $ret)? $body
        #[allow(unsafe_code)]
        $(#[$($attr)*])*
        $pub fn $dname($($arg: $ty),*) $(-> $ret)? {
            use $crate::SimdDescriptor;
            if let Some(d) = $crate::Avx512Descriptor::new() {
                #[target_feature(enable = "avx512f")]
                fn inner(d: $crate::Avx512Descriptor, $($arg: $ty),*) $(-> $ret)? {
                    $name(d, $($arg),*)
                }
                // SAFETY: we just checked for avx512f.
                return unsafe { inner(d, $($arg),*) };
            }
            if let Some(d) = $crate::AvxDescriptor::new() {
                #[target_feature(enable = "avx2,fma")]
                fn inner(d: $crate::AvxDescriptor, $($arg: $ty),*) $(-> $ret)? {
                    $name(d, $($arg),*)
                }
                // SAFETY: we just checked for avx2 and fma.
                return unsafe { inner(d, $($arg),*) };
            }
            if let Some(d) = $crate::Sse42Descriptor::new() {
                #[target_feature(enable = "sse4.2")]
                fn inner(d: $crate::Sse42Descriptor, $($arg: $ty),*) $(-> $ret)? {
                    $name(d, $($arg),*)
                }
                // SAFETY: we just checked for sse4.2.
                return unsafe { inner(d, $($arg),*) };
            }
            $name($crate::ScalarDescriptor::new().unwrap(), $($arg),*)
        }
    };
}

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
            #[allow(unsafe_code)]
            #[test]
            fn [<$name _sse42>]() {
                use $crate::SimdDescriptor;
                let Some(d) = $crate::Sse42Descriptor::new() else { return; };
                #[target_feature(enable = "sse4.2")]
                fn inner(d: $crate::Sse42Descriptor) {
                    $name(d)
                }
                // SAFETY: we just checked for sse4.2.
                return unsafe { inner(d) };
            }
            #[allow(unsafe_code)]
            #[test]
            fn [<$name _avx>]() {
                use $crate::SimdDescriptor;
                let Some(d) = $crate::AvxDescriptor::new() else { return; };
                #[target_feature(enable = "avx2,fma")]
                fn inner(d: $crate::AvxDescriptor) {
                    $name(d)
                }
                // SAFETY: we just checked for avx2 and fma.
                return unsafe { inner(d) };
            }
            #[allow(unsafe_code)]
            #[test]
            fn [<$name _avx512>]() {
                use $crate::SimdDescriptor;
                let Some(d) = $crate::Avx512Descriptor::new() else { return; };
                #[target_feature(enable = "avx512f")]
                fn inner(d: $crate::Avx512Descriptor) {
                    $name(d)
                }
                // SAFETY: we just checked for avx512f.
                return unsafe { inner(d) };
            }
        }
    };
}

#[macro_export]
macro_rules! bench_all_instruction_sets {
    (
        $name:ident,
        $criterion:ident
    ) => {
        use $crate::SimdDescriptor;
        if let Some(d) = $crate::Avx512Descriptor::new() {
            #[target_feature(enable = "avx512f")]
            fn inner(
                d: $crate::Avx512Descriptor,
                criterion: &mut ::criterion::BenchmarkGroup<
                    '_,
                    impl ::criterion::measurement::Measurement,
                >,
                name: &str,
            ) {
                $name(d, criterion, name)
            }
            // SAFETY: we just checked for avx512f.
            unsafe { inner(d, $criterion, "avx512") };
        }
        if let Some(d) = $crate::AvxDescriptor::new() {
            #[target_feature(enable = "avx2,fma")]
            fn inner(
                d: $crate::AvxDescriptor,
                criterion: &mut ::criterion::BenchmarkGroup<
                    '_,
                    impl ::criterion::measurement::Measurement,
                >,
                name: &str,
            ) {
                $name(d, criterion, name)
            }
            // SAFETY: we just checked for avx2 and fma.
            unsafe { inner(d, $criterion, "avx") };
        }
        if let Some(d) = $crate::Sse42Descriptor::new() {
            #[target_feature(enable = "sse4.2")]
            fn inner(
                d: $crate::Sse42Descriptor,
                criterion: &mut ::criterion::BenchmarkGroup<
                    '_,
                    impl ::criterion::measurement::Measurement,
                >,
                name: &str,
            ) {
                $name(d, criterion, name)
            }
            // SAFETY: we just checked for sse4.2.
            unsafe { inner(d, $criterion, "sse42") };
        }
        $name(
            $crate::ScalarDescriptor::new().unwrap(),
            $criterion,
            "scalar",
        );
    };
}

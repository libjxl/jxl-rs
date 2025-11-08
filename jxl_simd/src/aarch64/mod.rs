// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(unsafe_code)]
#![allow(clippy::identity_op)]

pub(super) mod neon;

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
            if let Some(d) = $crate::NeonDescriptor::new() {
                #[target_feature(enable = "neon")]
                fn inner(d: $crate::NeonDescriptor, $($arg: $ty),*) $(-> $ret)? {
                    $name(d, $($arg),*)
                }
                // SAFETY: we just checked for neon.
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
            fn [<$name _neon>]() {
                use $crate::SimdDescriptor;
                let Some(d) = $crate::NeonDescriptor::new() else { return; };
                #[target_feature(enable = "neon")]
                fn inner(d: $crate::NeonDescriptor) {
                    $name(d)
                }
                // SAFETY: we just checked for neon.
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
        if let Some(d) = $crate::NeonDescriptor::new() {
            #[target_feature(enable = "neon")]
            fn inner(
                d: $crate::NeonDescriptor,
                criterion: &mut ::criterion::BenchmarkGroup<
                    '_,
                    impl ::criterion::measurement::Measurement,
                >,
                name: &str,
            ) {
                $name(d, criterion, name)
            }
            // SAFETY: we just checked for neon.
            unsafe { inner(d, $criterion, "neon") };
        }
        $name(
            $crate::ScalarDescriptor::new().unwrap(),
            $criterion,
            "scalar",
        );
    };
}

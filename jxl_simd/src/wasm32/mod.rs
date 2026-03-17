// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#[cfg(feature = "simd128")]
pub(super) mod simd128;

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
        $(#[$($attr)*])*
        $pub fn $dname($($arg: $ty),*) $(-> $ret)? {
            #[allow(unused)]
            use $crate::SimdDescriptor;
            #[cfg(all(feature = "simd128", target_feature = "simd128"))]
            { $name($crate::Simd128Descriptor::new().unwrap(), $($arg),*) }
            #[cfg(not(all(feature = "simd128", target_feature = "simd128")))]
            { $name($crate::ScalarDescriptor::new().unwrap(), $($arg),*) }
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
        }

        $crate::test_simd128!($name);
    };
}

#[cfg(feature = "simd128")]
#[doc(hidden)]
#[macro_export]
macro_rules! test_simd128 {
    ($name:ident) => {
        paste::paste! {
            #[test]
            fn [<$name _simd128>]() {
                use $crate::SimdDescriptor;
                let Some(d) = $crate::Simd128Descriptor::new() else { return; };
                $name(d)
            }
        }
    };
}

#[cfg(not(feature = "simd128"))]
#[doc(hidden)]
#[macro_export]
macro_rules! test_simd128 {
    ($name:ident) => {};
}

#[macro_export]
macro_rules! bench_all_instruction_sets {
    (
        $name:ident,
        $criterion:ident
    ) => {
        use $crate::SimdDescriptor;
        #[cfg(all(feature = "simd128", target_feature = "simd128"))]
        $name(
            $crate::Simd128Descriptor::new().unwrap(),
            $criterion,
            "simd128",
        );
        #[cfg(not(all(feature = "simd128", target_feature = "simd128")))]
        $name(
            $crate::ScalarDescriptor::new().unwrap(),
            $criterion,
            "scalar",
        );
    };
}

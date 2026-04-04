// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(unsafe_code)]
#![allow(clippy::identity_op)]

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
        #[allow(unsafe_code)]
        $(#[$($attr)*])*
        $pub fn $dname($($arg: $ty),*) $(-> $ret)? {
            #[allow(unused)]
            use $crate::SimdDescriptor;
            $crate::simd_function_body_simd128!($name($($arg: $ty),*) $(-> $ret)?; ($($arg),*));
            $name($crate::ScalarDescriptor::new().unwrap(), $($arg),*)
        }
    };
}

#[cfg(feature = "simd128")]
#[doc(hidden)]
#[macro_export]
macro_rules! simd_function_body_simd128 {
    ($name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty )?; ($($val:expr),* $(,)?)) => {
        if cfg!(target_feature = "simd128") {
            // SAFETY: we just checked for simd128.
            let d = unsafe { $crate::WasmSimdDescriptor::new_unchecked() };
            return $name(d, $($val),*);
        } else if let Some(d) = $crate::WasmSimdDescriptor::new() {
            #[target_feature(enable = "simd128")]
            fn simd128(d: $crate::WasmSimdDescriptor, $($arg: $ty),*) $(-> $ret)? {
                $name(d, $($val),*)
            }
            // SAFETY: we just checked for simd128.
            return unsafe { simd128(d, $($arg),*) };
        }
    };
}

#[cfg(not(feature = "simd128"))]
#[doc(hidden)]
#[macro_export]
macro_rules! simd_function_body_simd128 {
    ($($ignore:tt)*) => {};
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
            #[allow(unsafe_code, unused_unsafe)]
            #[test]
            fn [<$name _simd128>]() {
                use $crate::SimdDescriptor;
                let Some(d) = $crate::WasmSimdDescriptor::new() else { return; };
                #[target_feature(enable = "simd128")]
                fn inner(d: $crate::WasmSimdDescriptor) {
                    $name(d)
                }
                // SAFETY: we just checked for simd128.
                return unsafe { inner(d) };
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
        // `simd_function_body_*` does early return; wrap it with an immediately-invoked closure
        (|| {
            $crate::simd_function_body_simd128!(
                $name($criterion: &mut ::criterion::BenchmarkGroup<'_, impl ::criterion::measurement::Measurement>);
                ($criterion, "simd128")
            );
        })();
        $name(
            $crate::ScalarDescriptor::new().unwrap(),
            $criterion,
            "scalar",
        );
    };
}

// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Ignored by default so feature-enabled `cargo test` stays portable.
//! The AVX-512 CI job runs this target with `--ignored`.

#![cfg(all(target_arch = "x86_64", feature = "avx512"))]

use jxl_simd::{Avx512Descriptor, SimdDescriptor};

#[test]
#[ignore = "requires AVX-512F and AVX-512BW; run by the AVX-512 CI job"]
fn requires_avx512() {
    assert!(
        Avx512Descriptor::new().is_some(),
        "missing AVX-512F and AVX-512BW; Avx512Descriptor::new() returned None, so this runner cannot provide AVX-512 coverage"
    );
}

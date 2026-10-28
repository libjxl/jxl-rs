// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=c/harness.c");
    println!("cargo::rerun-if-changed=c/cxx_harness.cc");

    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if target_arch == "wasm32" {
        return;
    }

    let jxl_include = std::env::var_os("DEP_JXL_INCLUDE")
        .expect("DEP_JXL_INCLUDE must be exported by jxl-libjxl-c-api");

    cc::Build::new()
        .file("c/harness.c")
        .include(&jxl_include)
        .std("c11")
        .warnings(true)
        .extra_warnings(true)
        .warnings_into_errors(true)
        .compile("jxl_c_test_harness");

    cc::Build::new()
        .cpp(true)
        .file("c/cxx_harness.cc")
        .include(&jxl_include)
        .std("c++17")
        .warnings(true)
        .extra_warnings(true)
        .warnings_into_errors(true)
        .compile("jxl_cxx_test_harness");
}

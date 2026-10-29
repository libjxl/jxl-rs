// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::path::PathBuf;

fn main() {
    let manifest_dir =
        PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let include_dir = manifest_dir.join("include");
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=include");
    println!("cargo::metadata=include={}", include_dir.display());
}

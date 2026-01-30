// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use anyhow::Result;
use vergen_gitcl::{Emitter, GitclBuilder, RustcBuilder};

fn main() {
    generate_build_info().unwrap();
}

fn generate_build_info() -> Result<()> {
    let gitcl = GitclBuilder::default().describe(true, true, None).build()?;
    let rustc = RustcBuilder::default().semver(true).build()?;

    Emitter::default()
        .add_instructions(&gitcl)?
        .add_instructions(&rustc)?
        .emit()
}

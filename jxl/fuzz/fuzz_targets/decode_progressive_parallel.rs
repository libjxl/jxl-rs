// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#![no_main]

use jxl_fuzz::{FuzzConfig, fuzz_decode};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = fuzz_decode(
        data,
        FuzzConfig {
            progressive: true,
            parallel: true,
            num_threads: 2,
            flush_intermediate: true,
            ..Default::default()
        },
    );
});

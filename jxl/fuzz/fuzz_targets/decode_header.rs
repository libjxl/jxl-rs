// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#![no_main]

use jxl::api::{JxlDecoder, JxlDecoderOptions};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut data = data;
    let decoder_options = JxlDecoderOptions::default();
    let mut decoder = JxlDecoder::new(decoder_options);
    let _ = decoder.process(&mut data, None, None);
});

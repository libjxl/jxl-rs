// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#![no_main]

use jxl::api::{JxlDecoder, JxlDecoderOptions, states};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Note: cargo-fuzz doesn't support |mut data: &[u8]| here.
    let mut data = data;
    let decoder_options = JxlDecoderOptions::default();
    let initialized_decoder = JxlDecoder::<states::Initialized>::new(decoder_options);
    let _ = initialized_decoder.process(&mut data);
});

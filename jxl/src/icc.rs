// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::bit_reader::*;
use crate::entropy_coding::decode::Histograms;
use crate::error::Error;
use crate::headers::encodings::*;

const ICC_CONTEXTS: usize = 41;

pub fn read_icc(br: &mut BitReader) -> Result<Vec<u8>, Error> {
    let len = u64::read_unconditional(&(), br, &Empty {})?;
    if len > 1u64 << 20 {
        return Err(Error::ICCTooLarge);
    }
    println!("Encoded len: {}", len);

    let histograms = Histograms::decode(ICC_CONTEXTS, br, /*allow_lz77=*/ true)?;

    println!("{:#?}", histograms);

    Ok(vec![])
}

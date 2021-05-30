// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

extern crate jxl_headers_derive;

use jxl_headers_derive::UnconditionalCoder;
use num_derive::FromPrimitive;

use crate::bit_reader::BitReader;
use crate::error::Error;
use crate::headers::bit_depth::*;
use crate::headers::encodings::*;
use crate::headers::size::*;

#[derive(Debug)]
pub struct Signature;

impl Signature {
    pub fn new() -> Signature {
        Signature {}
    }
}

impl crate::headers::encodings::UnconditionalCoder<()> for Signature {
    fn read_unconditional(_: (), br: &mut BitReader) -> Result<Signature, Error> {
        let sig1 = br.read(8)? as u8;
        let sig2 = br.read(8)? as u8;
        if (sig1, sig2) != (0xff, 0x0a) {
            Err(Error::InvalidSignature(sig1, sig2))
        } else {
            Ok(Signature {})
        }
    }
}

#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive)]
enum Orientation {
    Identity = 1,
    FlipHorizontal = 2,
    Rotate180 = 3,
    FlipVertical = 4,
    Transpose = 5,
    Rotate90 = 6,
    AntiTranspose = 7,
    Rotate270 = 8,
}

#[derive(UnconditionalCoder, Debug)]
struct Animation {
    #[coder(u2S(100, 1000, Bits(10) + 1, Bits(30) + 1))]
    tps_numerator: u32,
    #[coder(u2S(1, 1001, Bits(10) + 1, Bits(30) + 1))]
    tps_denominator: u32,
    #[coder(u2S(0, Bits(3), Bits(16), Bits(32)))]
    num_loops: u32,
    have_timecodes: bool,
}

#[derive(UnconditionalCoder, Debug)]
#[trace]
pub struct ImageMetadata {
    #[all_default]
    all_default: bool,
    #[default(false)]
    extra_fields: bool,
    #[condition(extra_fields)]
    #[default(Orientation::Identity)]
    #[coder(Bits(3) + 1)]
    orientation: Orientation,
    #[condition(extra_fields)]
    #[default(false)]
    have_intrinsic_size: bool, // TODO(veluca93): fold have_ fields in Option.
    #[condition(have_intrinsic_size)]
    intrinsic_size: Option<Size>,
    #[condition(extra_fields)]
    #[default(false)]
    have_preview: bool,
    #[condition(have_preview)]
    preview: Option<Preview>,
    #[condition(extra_fields)]
    #[default(false)]
    have_animation: bool,
    #[condition(have_animation)]
    animation: Option<Animation>,
    #[default(BitDepth::default())]
    bit_depth: BitDepth,
    #[default(true)]
    modular_16bit_sufficient: bool,
    // extra_channel_info: Vec<ExtraChannelInfo>,
    // xyb_encoded: bool,
    // color_encoding: ColorEncoding,
    // tone_mapping: ToneMapping,
    // extensions: ???,
}

// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

extern crate jxl_headers_derive;
extern crate num_derive;

use jxl_headers_derive::UnconditionalCoder;
use num_derive::FromPrimitive;

use crate::bit_reader::BitReader;
use crate::error::Error;
use crate::headers::bit_depth::BitDepth;
use crate::headers::encodings::*;

#[allow(clippy::upper_case_acronyms)]
#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive)]
enum ExtraChannel {
    Alpha,
    Depth,
    SpotColor,
    SelectionMask,
    Black,
    CFA,
    Thermal,
    Reserved0,
    Reserved1,
    Reserved2,
    Reserved3,
    Reserved4,
    Reserved5,
    Reserved6,
    Reserved7,
    Unknown,
    Optional,
}

#[derive(UnconditionalCoder, Debug)]
#[validate]
pub struct ExtraChannelInfo {
    #[all_default]
    all_default: bool,
    #[default(ExtraChannel::Alpha)]
    ec_type: ExtraChannel,
    #[default(BitDepth::default())]
    bit_depth: BitDepth,
    #[coder(u2S(0, 3, 4, Bits(3) + 1))]
    #[default(0)]
    dim_shift: u32,
    name: String,
    // TODO(veluca93): if using Option<bool>, this is None when all_default.
    #[condition(ec_type == ExtraChannel::Alpha)]
    #[default(false)]
    alpha_associated: bool,
    #[condition(ec_type == ExtraChannel::SpotColor)]
    spot_color: Option<[f32; 3]>,
    #[condition(ec_type == ExtraChannel::CFA)]
    #[coder(u2S(1, Bits(2), Bits(4) + 3, Bits(8) + 19))]
    cfa_channel: Option<u32>,
}

impl ExtraChannelInfo {
    fn check(&self) -> Result<(), Error> {
        if self.dim_shift > 3 {
            Err(Error::DimShiftTooLarge(self.dim_shift))
        } else {
            Ok(())
        }
    }
}

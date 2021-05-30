// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

extern crate jxl_headers_derive;

use jxl_headers_derive::UnconditionalCoder;
use num_derive::FromPrimitive;

use crate::bit_reader::BitReader;
use crate::error::Error;
use crate::headers::encodings::*;

#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive)]
pub enum ColorSpace {
    RGB,
    Gray,
    XYB,
    Unknown,
}

#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive)]
pub enum WhitePoint {
    D65 = 1,
    Custom = 2,
    E = 10,
    DCI = 11,
}

#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive)]
pub enum Primaries {
    SRGB = 1,
    Custom = 2,
    BT2100 = 9,
    P3 = 11,
}

#[derive(UnconditionalCoder, Debug)]
pub struct CustomXY {
    #[default(0)]
    #[coder(u2S(Bits(19), Bits(19) + 524288, Bits(20) + 1048576, Bits(21) + 2097152))]
    x: i32,
    #[default(0)]
    #[coder(u2S(Bits(19), Bits(19) + 524288, Bits(20) + 1048576, Bits(21) + 2097152))]
    y: i32,
}

#[derive(UnconditionalCoder, Debug)]
pub struct ColorEncoding {
    #[all_default]
    #[default(true)]
    all_default: bool,
    #[default(false)]
    want_icc: bool,
    #[default(ColorSpace::RGB)]
    color_space: ColorSpace,
    #[condition(!want_icc && color_space != ColorSpace::XYB)]
    #[default(WhitePoint::D65)]
    white_point: WhitePoint,
    // TODO(veluca): can this be merged in the enum?
    #[condition(white_point == WhitePoint::Custom)]
    #[default(CustomXY::default())]
    white: CustomXY,
    #[condition(!want_icc && color_space != ColorSpace::XYB && color_space != ColorSpace::Gray)]
    #[default(Primaries::SRGB)]
    primaries: Primaries,
    #[condition(primaries == Primaries::Custom)]
    #[default([CustomXY::default(), CustomXY::default(), CustomXY::default()])]
    custom_primaries: [CustomXY; 3],
    // tf: TransferFunction,
    // rendering_intent: RenderingIntent,
}

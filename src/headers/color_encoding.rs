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

#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive)]
pub enum TransferFunction {
    BT709 = 1,
    Unknown = 2,
    Linear = 8,
    SRGB = 13,
    PQ = 16,
    DCI = 17,
    HLG = 18,
}

#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive)]
pub enum RenderingIntent {
    Perceptual = 0,
    Relative,
    Saturation,
    Absolute,
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

pub struct CustomTransferFunctionNonserialized {
    color_space: ColorSpace,
}

#[derive(UnconditionalCoder, Debug)]
#[nonserialized(CustomTransferFunctionNonserialized)]
#[validate]
pub struct CustomTransferFunction {
    #[condition(nonserialized.color_space != ColorSpace::XYB)]
    #[default(false)]
    have_gamma: bool,
    #[condition(have_gamma)]
    #[default(3333333)] // XYB gamma
    #[coder(Bits(24))]
    gamma: u32,
    #[condition(!have_gamma && nonserialized.color_space != ColorSpace::XYB)]
    #[default(TransferFunction::SRGB)]
    transfer_function: TransferFunction,
}

impl CustomTransferFunction {
    pub fn gamma(&self) -> f32 {
        assert!(self.have_gamma);
        self.gamma as f32 * 0.0000001
    }

    pub fn check(&self) -> Result<(), Error> {
        if self.have_gamma {
            let gamma = self.gamma();
            if gamma > 1.0 || gamma * 8192.0 < 1.0 {
                Err(Error::InvalidGamma(gamma))
            } else {
                Ok(())
            }
        } else {
            Ok(())
        }
    }
}

#[derive(UnconditionalCoder, Debug)]
#[validate]
pub struct ColorEncoding {
    #[all_default]
    #[default(true)]
    all_default: bool,
    #[default(false)]
    pub want_icc: bool,
    #[default(ColorSpace::RGB)]
    pub color_space: ColorSpace,
    #[condition(!want_icc && color_space != ColorSpace::XYB)]
    #[default(WhitePoint::D65)]
    pub white_point: WhitePoint,
    // TODO(veluca): can this be merged in the enum?
    #[condition(white_point == WhitePoint::Custom)]
    #[default(CustomXY::default())]
    pub white: CustomXY,
    #[condition(!want_icc && color_space != ColorSpace::XYB && color_space != ColorSpace::Gray)]
    #[default(Primaries::SRGB)]
    pub primaries: Primaries,
    #[condition(primaries == Primaries::Custom)]
    #[default([CustomXY::default(), CustomXY::default(), CustomXY::default()])]
    pub custom_primaries: [CustomXY; 3],
    #[condition(!want_icc)]
    #[default(CustomTransferFunction::default())]
    #[nonserialized(color_space: color_space)]
    pub tf: CustomTransferFunction,
    #[condition(!want_icc)]
    #[default(RenderingIntent::Relative)]
    pub rendering_intent: RenderingIntent,
}

impl ColorEncoding {
    pub fn check(&self) -> Result<(), Error> {
        if !self.want_icc
            && (self.color_space == ColorSpace::Unknown
                || self.tf.transfer_function == TransferFunction::Unknown)
        {
            Err(Error::InvalidColorEncoding)
        } else {
            Ok(())
        }
    }
}

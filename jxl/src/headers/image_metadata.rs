// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    bit_reader::BitReader,
    error::Error,
    headers::{bit_depth::*, color_encoding::*, encodings::*, extra_channels::*, size::*},
};
use jxl_macros::UnconditionalCoder;
use num_derive::FromPrimitive;

#[derive(Debug, Default, Clone)]
pub struct Signature;

impl Signature {
    pub fn new() -> Signature {
        Signature {}
    }
}

impl crate::headers::encodings::UnconditionalCoder<()> for Signature {
    type Nonserialized = Empty;
    fn read_unconditional(_: &(), br: &mut BitReader, _: &Empty) -> Result<Signature, Error> {
        let sig1 = br.read(8)? as u8;
        let sig2 = br.read(8)? as u8;
        if (sig1, sig2) != (0xff, 0x0a) {
            Err(Error::InvalidSignature)
        } else {
            Ok(Signature {})
        }
    }
}

#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive)]
pub enum Orientation {
    Identity = 1,
    FlipHorizontal = 2,
    Rotate180 = 3,
    FlipVertical = 4,
    Transpose = 5,
    Rotate90 = 6,
    AntiTranspose = 7,
    Rotate270 = 8,
}

impl Orientation {
    pub fn is_transposing(self) -> bool {
        matches!(
            self,
            Orientation::Transpose
                | Orientation::AntiTranspose
                | Orientation::Rotate90
                | Orientation::Rotate270
        )
    }
}

#[derive(UnconditionalCoder, Debug, Clone)]
pub struct Animation {
    #[coder(u2S(100, 1000, Bits(10) + 1, Bits(30) + 1))]
    pub tps_numerator: u32,
    #[coder(u2S(1, 1001, Bits(8) + 1, Bits(10) + 1))]
    pub tps_denominator: u32,
    #[coder(u2S(0, Bits(3), Bits(16), Bits(32)))]
    pub num_loops: u32,
    pub have_timecodes: bool,
}

#[derive(UnconditionalCoder, Debug, Clone)]
#[validate]
pub struct ToneMapping {
    #[all_default]
    pub all_default: bool,
    #[default(255.0)]
    pub intensity_target: f32,
    #[default(0.0)]
    pub min_nits: f32,
    #[default(false)]
    pub relative_to_max_display: bool,
    #[default(0.0)]
    pub linear_below: f32,
}

impl ToneMapping {
    #[cfg(test)]
    pub fn empty() -> ToneMapping {
        ToneMapping {
            all_default: false,
            intensity_target: 0f32,
            min_nits: 0f32,
            relative_to_max_display: false,
            linear_below: 0f32,
        }
    }
    pub fn check(&self, _: &Empty) -> Result<(), Error> {
        if self.intensity_target <= 0.0 {
            Err(Error::InvalidIntensityTarget(self.intensity_target))
        } else if self.min_nits < 0.0 || self.min_nits > self.intensity_target {
            Err(Error::InvalidMinNits(self.min_nits))
        } else if self.linear_below < 0.0
            || (self.relative_to_max_display && self.linear_below > 1.0)
        {
            Err(Error::InvalidLinearBelow(
                self.relative_to_max_display,
                self.linear_below,
            ))
        } else {
            Ok(())
        }
    }
}

// TODO(firsching): remove once we use this!
#[allow(dead_code)]
#[derive(UnconditionalCoder, Debug, Clone)]
pub struct ImageMetadata {
    #[all_default]
    all_default: bool,
    #[default(false)]
    extra_fields: bool,
    #[condition(extra_fields)]
    #[default(Orientation::Identity)]
    #[coder(Bits(3) + 1)]
    pub orientation: Orientation,
    #[condition(extra_fields)]
    #[default(false)]
    have_intrinsic_size: bool, // TODO(veluca93): fold have_ fields in Option.
    #[condition(have_intrinsic_size)]
    pub intrinsic_size: Option<Size>,
    #[condition(extra_fields)]
    #[default(false)]
    have_preview: bool,
    #[condition(have_preview)]
    pub preview: Option<Preview>,
    #[condition(extra_fields)]
    #[default(false)]
    have_animation: bool,
    #[condition(have_animation)]
    pub animation: Option<Animation>,
    #[default(BitDepth::default(&field_nonserialized))]
    pub bit_depth: BitDepth,
    #[default(true)]
    pub modular_16bit_sufficient: bool,
    #[size_coder(implicit(u2S(0, 1, Bits(4) + 2, Bits(12) + 1)))]
    pub extra_channel_info: Vec<ExtraChannelInfo>,
    #[default(true)]
    pub xyb_encoded: bool,
    #[default(ColorEncoding::default(&field_nonserialized))]
    pub color_encoding: ColorEncoding,
    #[condition(extra_fields)]
    #[default(ToneMapping::default(&field_nonserialized))]
    pub tone_mapping: ToneMapping,
    extensions: Option<Extensions>,
}

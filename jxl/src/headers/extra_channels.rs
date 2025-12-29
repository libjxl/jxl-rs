// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    bit_reader::BitReader,
    error::Error,
    headers::{bit_depth::BitDepth, encodings::*},
};
use jxl_macros::UnconditionalCoder;
use num_derive::FromPrimitive;

#[allow(clippy::upper_case_acronyms)]
#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive, Eq)]
pub enum ExtraChannel {
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

// TODO(veluca): figure out if these fields should be unused.
#[allow(dead_code)]
#[derive(UnconditionalCoder, Debug, Clone)]
#[validate]
pub struct ExtraChannelInfo {
    #[all_default]
    all_default: bool,
    #[default(ExtraChannel::Alpha)]
    pub ec_type: ExtraChannel,
    #[default(BitDepth::default(&field_nonserialized))]
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
    pub spot_color: Option<[f32; 4]>,
    #[condition(ec_type == ExtraChannel::CFA)]
    #[coder(u2S(1, Bits(2), Bits(4) + 3, Bits(8) + 19))]
    cfa_channel: Option<u32>,
}

impl ExtraChannelInfo {
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        all_default: bool,
        ec_type: ExtraChannel,
        bit_depth: BitDepth,
        dim_shift: u32,
        name: String,
        alpha_associated: bool,
        spot_color: Option<[f32; 4]>,
        cfa_channel: Option<u32>,
    ) -> ExtraChannelInfo {
        ExtraChannelInfo {
            all_default,
            ec_type,
            bit_depth,
            dim_shift,
            name,
            alpha_associated,
            spot_color,
            cfa_channel,
        }
    }
    pub fn dim_shift(&self) -> u32 {
        self.dim_shift
    }
    pub fn alpha_associated(&self) -> bool {
        self.alpha_associated
    }
    pub fn bit_depth(&self) -> BitDepth {
        self.bit_depth
    }
    fn check(&self, _: &Empty) -> Result<(), Error> {
        if self.dim_shift > 3 {
            Err(Error::DimShiftTooLarge(self.dim_shift))
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::headers::bit_depth::BitDepth;

    /// Test that extra channels can have their own bit depth independent of the image.
    ///
    /// This is important because extra channels (like alpha, depth maps, etc.) may
    /// have different precision requirements than the main color channels. For example,
    /// an image might be 8-bit RGB with a 16-bit alpha channel.
    ///
    /// Previously the render pipeline incorrectly used the image's metadata bit depth
    /// for all extra channels, causing incorrect conversion for channels with different
    /// bit depths.
    #[test]
    fn test_extra_channel_bit_depth() {
        // Create an 8-bit extra channel
        let ec_8bit = ExtraChannelInfo::new(
            false,
            ExtraChannel::Alpha,
            BitDepth::integer_samples(8),
            0,
            "alpha".to_string(),
            false,
            None,
            None,
        );
        assert_eq!(ec_8bit.bit_depth().bits_per_sample(), 8);

        // Create a 16-bit extra channel
        let ec_16bit = ExtraChannelInfo::new(
            false,
            ExtraChannel::Depth,
            BitDepth::integer_samples(16),
            0,
            "depth".to_string(),
            false,
            None,
            None,
        );
        assert_eq!(ec_16bit.bit_depth().bits_per_sample(), 16);

        // Verify they are independent
        assert_ne!(
            ec_8bit.bit_depth().bits_per_sample(),
            ec_16bit.bit_depth().bits_per_sample()
        );
    }

    /// Test that the bit_depth getter returns the correct value for float samples.
    #[test]
    fn test_extra_channel_float_bit_depth() {
        let ec_float = ExtraChannelInfo::new(
            false,
            ExtraChannel::Depth,
            BitDepth::f32(),
            0,
            "depth_float".to_string(),
            false,
            None,
            None,
        );
        assert!(ec_float.bit_depth().floating_point_sample());
        assert_eq!(ec_float.bit_depth().bits_per_sample(), 32);
    }

    /// Test that using the wrong bit depth for conversion produces incorrect values.
    ///
    /// This test demonstrates why the render pipeline MUST use each extra channel's
    /// own bit_depth rather than the image's global bit_depth.
    ///
    /// The modular-to-f32 conversion scale is: 1.0 / ((1 << bits) - 1)
    /// - 8-bit:  scale = 1/255,   so value 255 → 1.0
    /// - 16-bit: scale = 1/65535, so value 255 → ~0.00389
    ///
    /// If an 8-bit extra channel is decoded using a 16-bit scale (the image's bit depth),
    /// the maximum value (255) would map to 0.00389 instead of 1.0 - completely wrong!
    #[test]
    fn test_wrong_bit_depth_produces_wrong_conversion() {
        // Simulate the conversion scale calculation from ConvertModularToF32Stage
        fn conversion_scale(bits: u32) -> f32 {
            1.0 / ((1u64 << bits) - 1) as f32
        }

        let scale_8bit = conversion_scale(8);
        let scale_16bit = conversion_scale(16);

        // Max 8-bit value
        let max_8bit_value = 255i32;

        // Correct conversion: 8-bit channel with 8-bit scale
        let correct_result = max_8bit_value as f32 * scale_8bit;
        assert!(
            (correct_result - 1.0).abs() < 1e-6,
            "8-bit max value should convert to 1.0, got {}",
            correct_result
        );

        // WRONG conversion: 8-bit channel with 16-bit scale (the bug!)
        let wrong_result = max_8bit_value as f32 * scale_16bit;
        assert!(
            (wrong_result - 0.00389).abs() < 0.0001,
            "Using wrong scale, 255 converts to ~0.00389, got {}",
            wrong_result
        );

        // The difference is catastrophic - values would be ~257x too small
        let ratio = correct_result / wrong_result;
        assert!(
            ratio > 250.0,
            "Using wrong bit depth causes ~257x error, ratio was {}",
            ratio
        );
    }
}

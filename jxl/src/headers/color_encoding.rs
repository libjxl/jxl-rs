// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{bit_reader::BitReader, error::Error, headers::encodings::*};
use jxl_macros::UnconditionalCoder;
use num_derive::FromPrimitive;

#[allow(clippy::upper_case_acronyms)]
#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive)]
pub enum ColorSpace {
    RGB,
    Gray,
    XYB,
    Unknown,
}

#[allow(clippy::upper_case_acronyms)]
#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive)]
pub enum WhitePoint {
    D65 = 1,
    Custom = 2,
    E = 10,
    DCI = 11,
}

impl WhitePoint {
    // TODO: implement turning into strings for icc writing?
}

#[allow(clippy::upper_case_acronyms)]
#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive)]
pub enum Primaries {
    SRGB = 1,
    Custom = 2,
    BT2100 = 9,
    P3 = 11,
}

impl Primaries {
    // TODO: implement turning into strings for icc writing?
}

#[allow(clippy::upper_case_acronyms)]
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

impl RenderingIntent {
    // TODO: implement turning into strings for icc writing?
}

#[derive(UnconditionalCoder, Debug, Clone)]
pub struct CustomXY {
    #[default(0)]
    #[coder(u2S(Bits(19), Bits(19) + 524288, Bits(20) + 1048576, Bits(21) + 2097152))]
    pub x: i32,
    #[default(0)]
    #[coder(u2S(Bits(19), Bits(19) + 524288, Bits(20) + 1048576, Bits(21) + 2097152))]
    pub y: i32,
}

pub struct CustomTransferFunctionNonserialized {
    color_space: ColorSpace,
}

#[derive(UnconditionalCoder, Debug, Clone)]
#[nonserialized(CustomTransferFunctionNonserialized)]
#[validate]
pub struct CustomTransferFunction {
    #[condition(nonserialized.color_space != ColorSpace::XYB)]
    #[default(false)]
    pub have_gamma: bool,
    #[condition(have_gamma)]
    #[default(3333333)] // XYB gamma
    #[coder(Bits(24))]
    pub gamma: u32,
    #[condition(!have_gamma && nonserialized.color_space != ColorSpace::XYB)]
    #[default(TransferFunction::SRGB)]
    pub transfer_function: TransferFunction,
}

impl CustomTransferFunction {
    #[cfg(test)]
    pub fn empty() -> CustomTransferFunction {
        CustomTransferFunction {
            have_gamma: false,
            gamma: 0,
            transfer_function: TransferFunction::Unknown,
        }
    }
    pub fn gamma(&self) -> f32 {
        assert!(self.have_gamma);
        self.gamma as f32 * 0.0000001
    }

    pub fn check(&self, _: &CustomTransferFunctionNonserialized) -> Result<(), Error> {
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

/// Writes a u32 value in big-endian format to the slice at the given position.
pub fn write_u32_be(slice: &mut [u8], pos: usize, value: u32) -> Result<(), Error> {
    if pos.checked_add(4).is_none_or(|end| end > slice.len()) {
        return Err(Error::IccWriteOutOfBounds);
    }
    slice[pos..pos + 4].copy_from_slice(&value.to_be_bytes());
    Ok(())
}

/// Writes a u16 value in big-endian format to the slice at the given position.
pub fn write_u16_be(slice: &mut [u8], pos: usize, value: u16) -> Result<(), Error> {
    if pos.checked_add(2).is_none_or(|end| end > slice.len()) {
        return Err(Error::IccWriteOutOfBounds);
    }
    slice[pos..pos + 2].copy_from_slice(&value.to_be_bytes());
    Ok(())
}

/// Writes a u8 value to the slice at the given position.
pub fn write_u8(slice: &mut [u8], pos: usize, value: u8) -> Result<(), Error> {
    if pos.checked_add(1).is_none_or(|end| end > slice.len()) {
        return Err(Error::IccWriteOutOfBounds);
    }
    slice[pos] = value;
    Ok(())
}

/// Writes a 4-character ASCII tag string to the slice at the given position.
pub fn write_icc_tag(slice: &mut [u8], pos: usize, tag_str: &str) -> Result<(), Error> {
    if tag_str.len() != 4 || !tag_str.is_ascii() {
        return Err(Error::IccInvalidTagString(tag_str.to_string()));
    }
    if pos.checked_add(4).is_none_or(|end| end > slice.len()) {
        return Err(Error::IccWriteOutOfBounds);
    }
    slice[pos..pos + 4].copy_from_slice(tag_str.as_bytes());
    Ok(())
}

#[derive(UnconditionalCoder, Debug, Clone)]
#[validate]
pub struct ColorEncoding {
    #[all_default]
    // TODO(firsching): remove once we use this!
    #[allow(dead_code)]
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
    #[default(CustomXY::default(&field_nonserialized))]
    pub white: CustomXY,
    #[condition(!want_icc && color_space != ColorSpace::XYB && color_space != ColorSpace::Gray)]
    #[default(Primaries::SRGB)]
    pub primaries: Primaries,
    #[condition(primaries == Primaries::Custom)]
    #[default([CustomXY::default(&field_nonserialized), CustomXY::default(&field_nonserialized), CustomXY::default(&field_nonserialized)])]
    pub custom_primaries: [CustomXY; 3],
    #[condition(!want_icc)]
    #[default(CustomTransferFunction::default(&field_nonserialized))]
    #[nonserialized(color_space: color_space)]
    pub tf: CustomTransferFunction,
    #[condition(!want_icc)]
    #[default(RenderingIntent::Relative)]
    pub rendering_intent: RenderingIntent,
}

impl ColorEncoding {
    pub fn check(&self, _: &Empty) -> Result<(), Error> {
        if !self.want_icc
            && (self.color_space == ColorSpace::Unknown
                || self.tf.transfer_function == TransferFunction::Unknown)
        {
            Err(Error::InvalidColorEncoding)
        } else {
            Ok(())
        }
    }

    fn can_tone_map_for_icc(&self) -> bool {
        // Placeholder
        // TODO(firsching): implement this function
        false
    }

    pub fn create_icc_header(&self) -> Result<Vec<u8>, Error> {
        let mut header_data = vec![0u8; 128];

        // Profile size - To be filled in at the end of profile creation.
        write_u32_be(&mut header_data, 0, 0)?;
        const CMM_TAG: &str = "jxl ";
        // CMM Type
        write_icc_tag(&mut header_data, 4, CMM_TAG)?;

        // Profile version - ICC v4.4 (0x04400000)
        // Conformance tests have v4.3, libjxl produces v4.4
        write_u32_be(&mut header_data, 8, 0x04400000u32)?;

        let profile_class_str = match self.color_space {
            ColorSpace::XYB => "scnr",
            _ => "mntr",
        };
        write_icc_tag(&mut header_data, 12, profile_class_str)?;

        // Data color space
        let data_color_space_str = match self.color_space {
            ColorSpace::Gray => "GRAY",
            _ => "RGB ",
        };
        write_icc_tag(&mut header_data, 16, data_color_space_str)?;

        // PCS - Profile Connection Space
        // Corresponds to: if (kEnable3DToneMapping && CanToneMap(c))
        // Assuming kEnable3DToneMapping is true for this port for now.
        const K_ENABLE_3D_ICC_TONEMAPPING: bool = true;
        if K_ENABLE_3D_ICC_TONEMAPPING && self.can_tone_map_for_icc() {
            write_icc_tag(&mut header_data, 20, "Lab ")?;
        } else {
            write_icc_tag(&mut header_data, 20, "XYZ ")?;
        }

        // Date and Time - Placeholder values from libjxl
        write_u16_be(&mut header_data, 24, 2019)?; // Year
        write_u16_be(&mut header_data, 26, 12)?; // Month
        write_u16_be(&mut header_data, 28, 1)?; // Day
        write_u16_be(&mut header_data, 30, 0)?; // Hours
        write_u16_be(&mut header_data, 32, 0)?; // Minutes
        write_u16_be(&mut header_data, 34, 0)?; // Seconds

        write_icc_tag(&mut header_data, 36, "acsp")?;
        write_icc_tag(&mut header_data, 40, "APPL")?;

        // Profile flags
        write_u32_be(&mut header_data, 44, 0)?;
        // Device manufacturer
        write_u32_be(&mut header_data, 48, 0)?;
        // Device model
        write_u32_be(&mut header_data, 52, 0)?;
        // Device attributes
        write_u32_be(&mut header_data, 56, 0)?;
        write_u32_be(&mut header_data, 60, 0)?;

        // Rendering Intent
        write_u32_be(&mut header_data, 64, self.rendering_intent as u32)?;

        // Whitepoint is fixed to D50 for ICC.
        write_u32_be(&mut header_data, 68, 0x0000F6D6)?;
        write_u32_be(&mut header_data, 72, 0x00010000)?;
        write_u32_be(&mut header_data, 76, 0x0000D32D)?;

        // Profile Creator
        write_icc_tag(&mut header_data, 80, CMM_TAG)?;

        // Profile ID (MD5 checksum) (offset 84) - 16 bytes.
        // This is calculated at the end of profile creation and written here.

        // Reserved (offset 100-127) - already zeroed here.

        Ok(header_data)
    }

    pub fn maybe_create_profile(&self) -> Result<Option<Vec<u8>>, Error> {
        // TODO can reuse `check` above? or at least simplify logic/dedup somehow?
        if self.color_space == ColorSpace::Unknown
            || self.tf.transfer_function == TransferFunction::Unknown
        {
            return Ok(None);
        }
        if !matches!(
            self.color_space,
            ColorSpace::RGB | ColorSpace::Gray | ColorSpace::XYB
        ) {
            return Err(Error::InvalidColorSpace);
        }

        if self.color_space == ColorSpace::XYB
            && self.rendering_intent != RenderingIntent::Perceptual
        {
            return Err(Error::InvalidRenderingIntent);
        }
        let header = self.create_icc_header()?;
        // TODO
        // generate ICC profile as Vec<u8> here and return instead of just header
        Ok(Some(header))
    }
}

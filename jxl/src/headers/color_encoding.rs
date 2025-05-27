// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{bit_reader::BitReader, error::Error, headers::encodings::*};
use jxl_macros::UnconditionalCoder;
use num_derive::FromPrimitive;
use std::fmt;

#[allow(clippy::upper_case_acronyms)]
#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive)]
pub enum ColorSpace {
    RGB,
    Gray,
    XYB,
    Unknown,
}

impl fmt::Display for ColorSpace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                ColorSpace::RGB => "RGB",
                ColorSpace::Gray => "Gra",
                ColorSpace::XYB => "XYB",
                ColorSpace::Unknown => "CS?",
            }
        )
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive)]
pub enum WhitePoint {
    D65 = 1,
    Custom = 2,
    E = 10,
    DCI = 11,
}

impl fmt::Display for WhitePoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                WhitePoint::D65 => "D65",
                WhitePoint::Custom => "Cst",
                WhitePoint::E => "EER",
                WhitePoint::DCI => "DCI",
            }
        )
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive)]
pub enum Primaries {
    SRGB = 1,
    Custom = 2,
    BT2100 = 9,
    P3 = 11,
}

impl fmt::Display for Primaries {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Primaries::SRGB => "SRG",
                Primaries::Custom => "Cst", // Base string for Custom
                Primaries::BT2100 => "202",
                Primaries::P3 => "DCI",
            }
        )
    }
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

impl fmt::Display for TransferFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                TransferFunction::BT709 => "709",
                TransferFunction::Unknown => "TF?",
                TransferFunction::Linear => "Lin",
                TransferFunction::SRGB => "SRG",
                TransferFunction::PQ => "PeQ",
                TransferFunction::DCI => "DCI",
                TransferFunction::HLG => "HLG",
            }
        )
    }
}

#[derive(UnconditionalCoder, Copy, Clone, PartialEq, Debug, FromPrimitive)]
pub enum RenderingIntent {
    Perceptual = 0,
    Relative,
    Saturation,
    Absolute,
}

impl fmt::Display for RenderingIntent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                RenderingIntent::Perceptual => "Per",
                RenderingIntent::Relative => "Rel",
                RenderingIntent::Saturation => "Sat",
                RenderingIntent::Absolute => "Abs",
            }
        )
    }
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

/// Creates an ICC 'mluc' tag with a single "enUS" record.
///
/// The input `text` must be ASCII, as it will be encoded as UTF-16BE by prepending
/// a null byte to each ASCII character.
pub fn create_icc_mluc_tag(tags: &mut Vec<u8>, text: &str) -> Result<(), Error> {
    // libjxl comments that "The input text must be ASCII".
    // We enforce this.
    if !text.is_ascii() {
        return Err(Error::IccMlucTextNotAscii(text.to_string()));
    }
    // Tag signature 'mluc' (4 bytes)
    tags.extend_from_slice(b"mluc");
    // Reserved, must be 0 (4 bytes)
    tags.extend_from_slice(&0u32.to_be_bytes());
    // Number of records (u32, 4 bytes) - Hardcoded to 1.
    tags.extend_from_slice(&1u32.to_be_bytes());
    // Record size (u32, 4 bytes) - Each record descriptor is 12 bytes.
    // (Language Code [2] + Country Code [2] + String Length [4] + String Offset [4])
    tags.extend_from_slice(&12u32.to_be_bytes());
    // Language Code (2 bytes) - "en" for English
    tags.extend_from_slice(b"en");
    // Country Code (2 bytes) - "US" for United States
    tags.extend_from_slice(b"US");
    // Length of the string (u32, 4 bytes)
    // For ASCII text encoded as UTF-16BE, each char becomes 2 bytes.
    let string_actual_byte_length = text.len() * 2;
    tags.extend_from_slice(&(string_actual_byte_length as u32).to_be_bytes());
    // Offset of the string (u32, 4 bytes)
    // The string data for this record starts at offset 28.
    tags.extend_from_slice(&28u32.to_be_bytes());
    // The actual string data, encoded as UTF-16BE.
    // For ASCII char 'X', UTF-16BE is 0x00 0x58.
    for ascii_char_code in text.as_bytes() {
        tags.push(0u8);
        tags.push(*ascii_char_code);
    }

    Ok(())
}

struct TagInfo {
    signature: [u8; 4],
    // Offset of this tag's data relative to the START of the `tags_data` block
    offset_in_tags_blob: u32,
    // Unpadded size of this tag's actual data content.
    size_unpadded: u32,
}

fn pad_to_4_byte_boundary(data: &mut Vec<u8>) {
    data.resize(data.len().next_multiple_of(4), 0u8);
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

    pub fn get_color_encoding_description(&self) -> String {
        // Helper for formatting custom XY float values.
        // Your CustomXY stores i32, which are float * 1_000_000.
        let format_xy_float = |val: i32| -> String { format!("{:.7}", val as f64 / 1_000_000.0) };
        // Helper for formatting gamma float value.
        let format_gamma_float = |val: f32| -> String { format!("{:.7}", val) };

        // Handle special known color spaces first
        if self.color_space == ColorSpace::RGB && self.white_point == WhitePoint::D65 {
            if self.rendering_intent == RenderingIntent::Perceptual
                && !self.tf.have_gamma
                && self.tf.transfer_function == TransferFunction::SRGB
            {
                if self.primaries == Primaries::SRGB {
                    return "sRGB".to_string();
                }
                if self.primaries == Primaries::P3 {
                    return "DisplayP3".to_string();
                }
            }
            if self.rendering_intent == RenderingIntent::Relative
                && self.primaries == Primaries::BT2100
            {
                if !self.tf.have_gamma && self.tf.transfer_function == TransferFunction::PQ {
                    return "Rec2100PQ".to_string();
                }
                if !self.tf.have_gamma && self.tf.transfer_function == TransferFunction::HLG {
                    return "Rec2100HLG".to_string();
                }
            }
        }

        // Build the string part by part for other case
        let mut d = String::with_capacity(64);

        // Append ColorSpace string
        d.push_str(&self.color_space.to_string());

        let explicit_wp_tf = self.color_space != ColorSpace::XYB;

        if explicit_wp_tf {
            d.push('_');
            if self.white_point == WhitePoint::Custom {
                // For Custom, we append the specific xy values
                d.push_str(&format_xy_float(self.white.x));
                d.push(';');
                d.push_str(&format_xy_float(self.white.y));
            } else {
                d.push_str(&self.white_point.to_string());
            }
        }

        if self.color_space != ColorSpace::Gray && self.color_space != ColorSpace::XYB {
            d.push('_');
            if self.primaries == Primaries::Custom {
                // For Custom, append specific r,g,b xy values
                // Red primaries
                d.push_str(&format_xy_float(self.custom_primaries[0].x));
                d.push(';');
                d.push_str(&format_xy_float(self.custom_primaries[0].y));
                d.push(';');
                // Green primaries
                d.push_str(&format_xy_float(self.custom_primaries[1].x));
                d.push(';');
                d.push_str(&format_xy_float(self.custom_primaries[1].y));
                d.push(';');
                // Blue primaries
                d.push_str(&format_xy_float(self.custom_primaries[2].x));
                d.push(';');
                d.push_str(&format_xy_float(self.custom_primaries[2].y));
            } else {
                d.push_str(&self.primaries.to_string());
            }
        }

        d.push('_');
        d.push_str(&self.rendering_intent.to_string());

        if explicit_wp_tf {
            d.push('_');
            if self.tf.have_gamma {
                d.push('g');
                d.push_str(&format_gamma_float(self.tf.gamma()));
            } else {
                d.push_str(&self.tf.transfer_function.to_string());
            }
        }
        d
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
        let mut tags_data: Vec<u8> = Vec::new();
        let mut collected_tags: Vec<TagInfo> = Vec::new();

        // Create 'desc' (ProfileDescription) tag
        let description_string = self.get_color_encoding_description();

        let desc_tag_start_offset = tags_data.len() as u32; // 0 at this point ...
        create_icc_mluc_tag(&mut tags_data, &description_string)?;
        let desc_tag_unpadded_size = (tags_data.len() as u32) - desc_tag_start_offset;
        pad_to_4_byte_boundary(&mut tags_data);
        collected_tags.push(TagInfo {
            signature: *b"desc",
            offset_in_tags_blob: desc_tag_start_offset,
            size_unpadded: desc_tag_unpadded_size,
        });

        // Create 'cprt' (Copyright) tag
        let copyright_string = "CC0";
        let cprt_tag_start_offset = tags_data.len() as u32;
        create_icc_mluc_tag(&mut tags_data, copyright_string)?;
        let cprt_tag_unpadded_size = (tags_data.len() as u32) - cprt_tag_start_offset;
        pad_to_4_byte_boundary(&mut tags_data);
        collected_tags.push(TagInfo {
            signature: *b"cprt",
            offset_in_tags_blob: cprt_tag_start_offset,
            size_unpadded: cprt_tag_unpadded_size,
        });

        // TODO: add the more tags

        // Construct the Tag Table bytes
        let mut tag_table_bytes: Vec<u8> = Vec::new();
        // First, the number of tags (u32)
        tag_table_bytes.extend_from_slice(&(collected_tags.len() as u32).to_be_bytes());

        let header_size = header.len() as u32;
        // Each entry in the tag table on disk is 12 bytes: signature (4), offset (4), size (4)
        let tag_table_on_disk_size = 4 + (collected_tags.len() as u32 * 12);

        for tag_info in &collected_tags {
            tag_table_bytes.extend_from_slice(&tag_info.signature);
            // The offset in the tag table is absolute from the start of the ICC profile file
            let final_profile_offset_for_tag =
                header_size + tag_table_on_disk_size + tag_info.offset_in_tags_blob;
            tag_table_bytes.extend_from_slice(&final_profile_offset_for_tag.to_be_bytes());
            // In https://www.color.org/specification/ICC.1-2022-05.pdf, section 7.3.5 reads:
            //
            // "The value of the tag data element size shall be the number of actual data
            // bytes and shall not include any padding at the end of the tag data element."
            //
            // The reference from conformance tests and libjxl use the padded size here instead.
            tag_table_bytes.extend_from_slice(&tag_info.size_unpadded.to_be_bytes());
        }

        // Assemble the final ICC profile parts: header + tag_table + tags_data
        let mut final_icc_profile_data: Vec<u8> =
            Vec::with_capacity(header.len() + tag_table_bytes.len() + tags_data.len());
        final_icc_profile_data.extend_from_slice(&header);
        final_icc_profile_data.extend_from_slice(&tag_table_bytes);
        final_icc_profile_data.extend_from_slice(&tags_data);

        // Update the profile size in the header (at offset 0)
        let total_profile_size = final_icc_profile_data.len() as u32;
        write_u32_be(&mut final_icc_profile_data, 0, total_profile_size)?;
        // TODO: MD5 hashing?
        Ok(Some(final_icc_profile_data))
    }
}

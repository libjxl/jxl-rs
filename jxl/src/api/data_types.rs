// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{headers::extra_channels::ExtraChannel, image::DataTypeTag};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JxlColorType {
    Grayscale,
    GrayscaleAlpha,
    Rgb,
    Rgba,
    Bgr,
    Bgra,
}

impl JxlColorType {
    pub fn has_alpha(&self) -> bool {
        match self {
            Self::Grayscale => false,
            Self::GrayscaleAlpha => true,
            Self::Rgb | Self::Bgr => false,
            Self::Rgba | Self::Bgra => true,
        }
    }
    pub fn samples_per_pixel(&self) -> usize {
        match self {
            Self::Grayscale => 1,
            Self::GrayscaleAlpha => 2,
            Self::Rgb | Self::Bgr => 3,
            Self::Rgba | Self::Bgra => 4,
        }
    }
    pub fn is_grayscale(&self) -> bool {
        match self {
            Self::Grayscale => true,
            Self::GrayscaleAlpha => true,
            Self::Rgb | Self::Bgr => false,
            Self::Rgba | Self::Bgra => false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Endianness {
    LittleEndian,
    BigEndian,
}

impl Endianness {
    pub fn native() -> Self {
        #[cfg(target_endian = "little")]
        {
            Endianness::LittleEndian
        }
        #[cfg(target_endian = "big")]
        {
            Endianness::BigEndian
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JxlDataFormat {
    U8 {
        bit_depth: u8,
    },
    U16 {
        endianness: Endianness,
        bit_depth: u8,
    },
    F16 {
        endianness: Endianness,
    },
    F32 {
        endianness: Endianness,
    },
}

impl JxlDataFormat {
    pub fn bytes_per_sample(&self) -> usize {
        match self {
            Self::U8 { .. } => 1,
            Self::U16 { .. } | Self::F16 { .. } => 2,
            Self::F32 { .. } => 4,
        }
    }

    /// Creates a U8 format with 8-bit depth.
    pub fn u8() -> Self {
        Self::U8 { bit_depth: 8 }
    }

    /// Creates a U16 format with native endianness and 16-bit depth.
    pub fn u16() -> Self {
        Self::U16 {
            endianness: Endianness::native(),
            bit_depth: 16,
        }
    }

    /// Creates an F16 format with native endianness.
    pub fn f16() -> Self {
        Self::F16 {
            endianness: Endianness::native(),
        }
    }

    /// Creates an F32 format with native endianness.
    pub fn f32() -> Self {
        Self::F32 {
            endianness: Endianness::native(),
        }
    }

    pub(crate) fn data_type(&self) -> DataTypeTag {
        match self {
            JxlDataFormat::U8 { .. } => DataTypeTag::U8,
            JxlDataFormat::U16 { .. } => DataTypeTag::U16,
            JxlDataFormat::F16 { .. } => DataTypeTag::F16,
            JxlDataFormat::F32 { .. } => DataTypeTag::F32,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JxlPixelFormat {
    pub color_type: JxlColorType,
    // None -> ignore
    pub color_data_format: Option<JxlDataFormat>,
    pub extra_channel_format: Vec<Option<JxlDataFormat>>,
}

impl JxlPixelFormat {
    /// Creates an RGBA8 pixel format.
    ///
    /// The alpha channel (if present) is interleaved with RGB. Any additional
    /// extra channels are ignored.
    ///
    /// `num_extra_channels` should match `basic_info.extra_channels.len()`.
    pub fn rgba8(num_extra_channels: usize) -> Self {
        Self {
            color_type: JxlColorType::Rgba,
            color_data_format: Some(JxlDataFormat::u8()),
            extra_channel_format: vec![None; num_extra_channels],
        }
    }

    /// Creates an RGBA16 pixel format with native endianness.
    ///
    /// The alpha channel (if present) is interleaved with RGB. Any additional
    /// extra channels are ignored.
    ///
    /// `num_extra_channels` should match `basic_info.extra_channels.len()`.
    pub fn rgba16(num_extra_channels: usize) -> Self {
        Self {
            color_type: JxlColorType::Rgba,
            color_data_format: Some(JxlDataFormat::u16()),
            extra_channel_format: vec![None; num_extra_channels],
        }
    }

    /// Creates an RGBA F32 pixel format with native endianness.
    ///
    /// The alpha channel (if present) is interleaved with RGB. Any additional
    /// extra channels are ignored.
    ///
    /// `num_extra_channels` should match `basic_info.extra_channels.len()`.
    pub fn rgba_f32(num_extra_channels: usize) -> Self {
        Self {
            color_type: JxlColorType::Rgba,
            color_data_format: Some(JxlDataFormat::f32()),
            extra_channel_format: vec![None; num_extra_channels],
        }
    }

    /// Creates a BGRA8 pixel format (for native Windows/Skia format).
    ///
    /// The alpha channel (if present) is interleaved with BGR. Any additional
    /// extra channels are ignored.
    ///
    /// `num_extra_channels` should match `basic_info.extra_channels.len()`.
    pub fn bgra8(num_extra_channels: usize) -> Self {
        Self {
            color_type: JxlColorType::Bgra,
            color_data_format: Some(JxlDataFormat::u8()),
            extra_channel_format: vec![None; num_extra_channels],
        }
    }

    /// Creates an RGB8 pixel format (no alpha).
    ///
    /// `num_extra_channels` should match `basic_info.extra_channels.len()`.
    pub fn rgb8(num_extra_channels: usize) -> Self {
        Self {
            color_type: JxlColorType::Rgb,
            color_data_format: Some(JxlDataFormat::u8()),
            extra_channel_format: vec![None; num_extra_channels],
        }
    }

    /// Returns the number of bytes per pixel for this format.
    pub fn bytes_per_pixel(&self) -> Option<usize> {
        self.color_data_format
            .as_ref()
            .map(|df| df.bytes_per_sample() * self.color_type.samples_per_pixel())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum JxlBitDepth {
    Int {
        bits_per_sample: u32,
    },
    Float {
        bits_per_sample: u32,
        exponent_bits_per_sample: u32,
    },
}

impl JxlBitDepth {
    pub fn bits_per_sample(&self) -> u32 {
        match self {
            JxlBitDepth::Int { bits_per_sample: b } => *b,
            JxlBitDepth::Float {
                bits_per_sample: b, ..
            } => *b,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JxlExtraChannel {
    pub ec_type: ExtraChannel,
    pub alpha_associated: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JxlAnimation {
    pub tps_numerator: u32,
    pub tps_denominator: u32,
    pub num_loops: u32,
    pub have_timecodes: bool,
}

#[derive(Clone, Debug)]
pub struct JxlFrameHeader {
    pub name: String,
    pub duration: Option<f64>,
    /// Frame size (width, height)
    pub size: (usize, usize),
}

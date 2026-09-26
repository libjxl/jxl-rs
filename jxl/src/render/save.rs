// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::api::{JxlColorType, JxlDataFormat, JxlOutputBuffer};
use crate::error::{Error, Result};
use crate::headers::Orientation;
use crate::image::DataTypeTag;
use crate::util::{SmallVec, StackOnly};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ChannelConversion {
    None,
    F32ToU8 {
        bit_depth: u8,
        dither_channel: usize,
    },
    I16ToU8 {
        multiplier: i32,
        max: i32,
    },
    I32ToU8 {
        multiplier: i32,
        max: i32,
    },
    F32ToU16 {
        bit_depth: u8,
    },
    F32ToF16 {
        clamp_range: Option<(f32, f32)>,
    },
}

impl ChannelConversion {
    pub fn input_type(&self, default_type: DataTypeTag) -> DataTypeTag {
        match self {
            Self::None => default_type,
            Self::F32ToU8 { .. } | Self::F32ToU16 { .. } | Self::F32ToF16 { .. } => {
                DataTypeTag::F32
            }
            Self::I16ToU8 { .. } => DataTypeTag::I16,
            Self::I32ToU8 { .. } => DataTypeTag::I32,
        }
    }
}

#[derive(Debug)]
pub struct SaveStage {
    pub(super) channels: Vec<usize>,
    pub(super) orientation: Orientation,
    pub(super) output_buffer_index: usize,
    pub(super) color_type: JxlColorType,
    pub(super) data_format: JxlDataFormat,
    /// When true, fill alpha channel with opaque (1.0) values.
    /// Used when RGBA output is requested but image has no alpha channel.
    pub(super) fill_opaque_alpha: bool,
    pub(super) conversions: SmallVec<ChannelConversion, 4, StackOnly>,
}

impl SaveStage {
    pub fn new(
        channels: &[usize],
        orientation: Orientation,
        output_buffer_index: usize,
        mut color_type: JxlColorType,
        data_format: JxlDataFormat,
        fill_opaque_alpha: bool,
        mut conversions: SmallVec<ChannelConversion, 4, StackOnly>,
    ) -> SaveStage {
        let mut channels = channels.to_vec();
        while conversions.len() < channels.len() {
            conversions.push(ChannelConversion::None);
        }
        if color_type == JxlColorType::Bgr {
            color_type = JxlColorType::Rgb;
            channels.swap(0, 2);
            conversions.swap(0, 2);
        }
        if color_type == JxlColorType::Bgra {
            color_type = JxlColorType::Rgba;
            channels.swap(0, 2);
            conversions.swap(0, 2);
        }
        Self {
            channels,
            orientation,
            output_buffer_index,
            color_type,
            data_format,
            fill_opaque_alpha,
            conversions,
        }
    }

    /// Returns the number of output channels (including filled alpha if applicable)
    pub fn output_channels(&self) -> usize {
        self.color_type.samples_per_pixel()
    }

    pub fn uses_channel(&self, c: usize) -> bool {
        self.channels.contains(&c)
    }

    pub fn input_type(&self) -> DataTypeTag {
        self.data_format.data_type()
    }

    pub fn channel_input_type(&self, c: usize) -> DataTypeTag {
        let idx = self.channels.iter().position(|&chan| chan == c);
        match idx {
            Some(i) => self
                .conversions
                .get(i)
                .map(|conv| conv.input_type(self.data_format.data_type()))
                .unwrap_or(self.data_format.data_type()),
            None => self.data_format.data_type(),
        }
    }

    pub fn check_buffer_size(
        &self,
        size: (usize, usize),
        buffer: Option<&JxlOutputBuffer>,
    ) -> Result<()> {
        let Some(buf) = buffer else {
            return Ok(());
        };
        let osize = self.orientation.map_size(size);

        let expected_w = self.output_channels() * self.data_format.bytes_per_sample() * osize.0;

        if buf.byte_size() != (expected_w, osize.1) {
            return Err(Error::InvalidOutputBufferSize(
                buf.byte_size().0,
                buf.byte_size().1,
                osize.0,
                osize.1,
                self.color_type,
                self.data_format,
            ));
        }
        Ok(())
    }
}

impl std::fmt::Display for SaveStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "save channels {:?} (type {:?} {:?})",
            self.channels, self.color_type, self.data_format
        )
    }
}

// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::{
    api::{JxlColorType, JxlDataFormat, JxlOutputBuffer},
    error::{Error, Result},
    headers::Orientation,
    image::DataTypeTag,
};

pub struct SaveStage {
    pub(super) channels: Vec<usize>,
    pub(super) orientation: Orientation,
    pub(super) output_buffer_index: usize,
    pub(super) color_type: JxlColorType,
    pub(super) data_format: JxlDataFormat,
    /// If true, add opaque alpha (1.0) when color_type has alpha but channels doesn't include it.
    pub(super) fill_opaque_alpha: bool,
}

impl SaveStage {
    pub fn new(
        channels: &[usize],
        orientation: Orientation,
        output_buffer_index: usize,
        mut color_type: JxlColorType,
        data_format: JxlDataFormat,
    ) -> SaveStage {
        let mut channels = channels.to_vec();
        if color_type == JxlColorType::Bgr {
            color_type = JxlColorType::Rgb;
            channels.swap(0, 2);
        }
        if color_type == JxlColorType::Bgra {
            color_type = JxlColorType::Rgba;
            channels.swap(0, 2);
        }

        // Check if we need to fill opaque alpha:
        // - color_type has alpha (RGBA, GrayscaleAlpha, BGRA)
        // - but channels doesn't include an alpha source
        let fill_opaque_alpha =
            color_type.has_alpha() && channels.len() < color_type.samples_per_pixel();

        Self {
            channels,
            orientation,
            output_buffer_index,
            color_type,
            data_format,
            fill_opaque_alpha,
        }
    }

    /// Returns the number of output channels (may be more than source channels if filling opaque alpha).
    pub fn output_channels(&self) -> usize {
        if self.fill_opaque_alpha {
            self.color_type.samples_per_pixel()
        } else {
            self.channels.len()
        }
    }

    pub fn uses_channel(&self, c: usize) -> bool {
        self.channels.contains(&c)
    }

    pub fn input_type(&self) -> DataTypeTag {
        self.data_format.data_type()
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

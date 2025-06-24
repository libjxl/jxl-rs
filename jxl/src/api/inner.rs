// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(dead_code, unused_variables)]

use crate::{error::Result, headers::frame_header::FrameHeader};

use super::{
    JxlBasicInfo, JxlBitstreamInput, JxlCms, JxlColorProfile, JxlDecoderOptions, JxlOutputBuffer,
    JxlPixelFormat, ProcessingResult,
};

/// Low-level, less-type-safe API.
pub struct JxlDecoderInner {
    // TODO(veluca): more fields
    pub(super) options: JxlDecoderOptions,
    pub(super) cms: Option<Box<dyn JxlCms>>,
}

impl JxlDecoderInner {
    /// Creates a new decoder with the given options and CMS.
    pub fn new(options: JxlDecoderOptions, cms: Option<impl JxlCms + 'static>) -> Self {
        JxlDecoderInner {
            options,
            cms: cms.map(|cms| Box::new(cms) as Box<dyn JxlCms>),
        }
    }

    /// Process more of the input file.
    /// This function will return when reaching the next decoding stage (i.e. finished decoding
    /// file/frame header, or finished decoding a frame).
    pub fn process<'a, In: JxlBitstreamInput, Out: JxlOutputBuffer<'a> + ?Sized>(
        &mut self,
        input: &mut In,
        buffers: Option<&'a mut [&'a mut Out]>,
    ) -> Result<ProcessingResult<(), ()>> {
        todo!()
    }

    /// Skip the next `count` frames.
    pub fn skip_frames(
        &mut self,
        input: &mut impl JxlBitstreamInput,
        count: usize,
    ) -> Result<ProcessingResult<(), ()>> {
        todo!()
    }

    /// Skip the current frame.
    pub fn skip_frame(
        &mut self,
        input: &mut impl JxlBitstreamInput,
    ) -> Result<ProcessingResult<(), ()>> {
        todo!()
    }

    /// Obtains the image's basic information, if available.
    pub fn basic_info(&self) -> Option<&JxlBasicInfo> {
        todo!()
    }

    /// Retrieves the file's color profile, if available.
    pub fn embedded_color_profile(&self) -> Option<&JxlColorProfile> {
        todo!()
    }

    /// Specifies the preferred color profile to be used for outputting data.
    /// Same semantics as JxlDecoderSetOutputColorProfile.
    pub fn set_output_color_profile(&mut self, profile: &JxlColorProfile) -> Result<()> {
        todo!()
    }

    pub fn current_pixel_format(&self) -> Option<&JxlPixelFormat> {
        todo!()
    }

    pub fn set_pixel_format(&mut self, pixel_format: JxlPixelFormat) {
        todo!()
    }

    // TODO: don't use the raw bitstream type; include name and extra channel blend info.
    pub fn frame_header(&self) -> Option<&FrameHeader> {
        todo!()
    }

    /// Number of passes we have full data for.
    pub fn num_completed_passes(&self) -> Option<usize> {
        todo!()
    }

    /// Draws all the pixels we have data for.
    pub fn flush_pixels(&mut self) -> Result<()> {
        todo!()
    }

    /// Resets entirely a decoder, producing a new decoder with the same settings.
    /// This is faster than creating a new decoder in some cases.
    pub fn reset(&mut self) {
        todo!()
    }

    /// Rewinds a decoder to the start of the file, allowing past frames to be displayed again.
    pub fn rewind(&mut self) {
        todo!()
    }
}

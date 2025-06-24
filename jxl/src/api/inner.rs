// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(dead_code, unused_variables)]

use crate::{
    error::{Error, Result},
    headers::frame_header::FrameHeader,
};

use super::{
    JxlBasicInfo, JxlBitstreamInput, JxlCms, JxlColorProfile, JxlDecoderOptions, JxlOutputBuffer,
    JxlPixelFormat, ProcessingResult,
};

/// Low-level, less-type-safe API.
pub struct JxlDecoderInner {
    pub(super) options: JxlDecoderOptions,
    pub(super) cms: Option<Box<dyn JxlCms>>,
    // These fields are populated once image information is available.
    pub(super) basic_info: Option<JxlBasicInfo>,
    pub(super) embedded_color_profile: Option<JxlColorProfile>,
    pub(super) output_color_profile: Option<JxlColorProfile>,
    pub(super) pixel_format: Option<JxlPixelFormat>,
    // These fields are populated when starting to decode a frame, and cleared once
    // the frame is done.
    pub(super) frame_header: Option<FrameHeader>,
    pub(super) num_completed_passes: Option<usize>, // probably will be in a more nested struct
}

impl JxlDecoderInner {
    /// Creates a new decoder with the given options and, optionally, CMS.
    pub fn new(options: JxlDecoderOptions, cms: Option<Box<dyn JxlCms>>) -> Self {
        JxlDecoderInner {
            options,
            cms,
            basic_info: None,
            embedded_color_profile: None,
            output_color_profile: None,
            pixel_format: None,
            frame_header: None,
            num_completed_passes: None,
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

    /// Draws all the pixels we have data for.
    pub fn flush_pixels(&mut self) -> Result<()> {
        todo!()
    }

    /// Obtains the image's basic information, if available.
    pub fn basic_info(&self) -> Option<&JxlBasicInfo> {
        self.basic_info.as_ref()
    }

    /// Retrieves the file's color profile, if available.
    pub fn embedded_color_profile(&self) -> Option<&JxlColorProfile> {
        self.embedded_color_profile.as_ref()
    }

    /// Retrieves the current output color profile, if available.
    pub fn output_color_profile(&self) -> Option<&JxlColorProfile> {
        self.output_color_profile.as_ref()
    }

    /// Specifies the preferred color profile to be used for outputting data.
    /// Same semantics as JxlDecoderSetOutputColorProfile.
    pub fn set_output_color_profile(&mut self, profile: &JxlColorProfile) -> Result<()> {
        if let (JxlColorProfile::Icc(_), None) = (profile, &self.cms) {
            return Err(Error::ICCOutputNoCMS);
        }
        self.output_color_profile = Some(profile.clone());
        Ok(())
    }

    pub fn current_pixel_format(&self) -> Option<&JxlPixelFormat> {
        self.pixel_format.as_ref()
    }

    pub fn set_pixel_format(&mut self, pixel_format: JxlPixelFormat) {
        self.pixel_format = Some(pixel_format);
    }

    // TODO: don't use the raw bitstream type; include name and extra channel blend info.
    pub fn frame_header(&self) -> Option<&FrameHeader> {
        self.frame_header.as_ref()
    }

    /// Number of passes we have full data for.
    pub fn num_completed_passes(&self) -> Option<usize> {
        self.num_completed_passes
    }

    /// Resets entirely a decoder, producing a new decoder with the same settings.
    /// This is faster than creating a new decoder in some cases.
    pub fn reset(&mut self) {
        self.basic_info = None;
        self.embedded_color_profile = None;
        self.output_color_profile = None;
        self.pixel_format = None;
        self.frame_header = None;
        self.num_completed_passes = None;
    }

    /// Rewinds a decoder to the start of the file, allowing past frames to be displayed again.
    pub fn rewind(&mut self) {
        // TODO(veluca): keep track of frame offsets for skipping.
        self.reset();
    }
}

// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![allow(dead_code)]

use crate::{
    api::JxlFrameHeader,
    error::{Error, Result},
};

use super::{JxlBasicInfo, JxlCms, JxlColorProfile, JxlDecoderOptions, JxlPixelFormat};
use box_parser::BoxParser;
use codestream_parser::CodestreamParser;

mod box_parser;
mod codestream_parser;
mod process;

/// Low-level, less-type-safe API.
pub struct JxlDecoderInner {
    options: JxlDecoderOptions,
    cms: Option<Box<dyn JxlCms>>,
    box_parser: BoxParser,
    codestream_parser: CodestreamParser,
}

impl JxlDecoderInner {
    /// Creates a new decoder with the given options and, optionally, CMS.
    pub fn new(options: JxlDecoderOptions, cms: Option<Box<dyn JxlCms>>) -> Self {
        JxlDecoderInner {
            options,
            cms,
            box_parser: BoxParser::new(),
            codestream_parser: CodestreamParser::new(),
        }
    }

    /// Obtains the image's basic information, if available.
    pub fn basic_info(&self) -> Option<&JxlBasicInfo> {
        self.codestream_parser.basic_info.as_ref()
    }

    /// Retrieves the file's color profile, if available.
    pub fn embedded_color_profile(&self) -> Option<&JxlColorProfile> {
        self.codestream_parser.embedded_color_profile.as_ref()
    }

    /// Retrieves the current output color profile, if available.
    pub fn output_color_profile(&self) -> Option<&JxlColorProfile> {
        self.codestream_parser.output_color_profile.as_ref()
    }

    /// Specifies the preferred color profile to be used for outputting data.
    /// Same semantics as JxlDecoderSetOutputColorProfile.
    pub fn set_output_color_profile(&mut self, profile: &JxlColorProfile) -> Result<()> {
        if let (JxlColorProfile::Icc(_), None) = (profile, &self.cms) {
            return Err(Error::ICCOutputNoCMS);
        }
        unimplemented!()
    }

    pub fn current_pixel_format(&self) -> Option<&JxlPixelFormat> {
        self.codestream_parser.pixel_format.as_ref()
    }

    pub fn set_pixel_format(&mut self, pixel_format: JxlPixelFormat) {
        drop(pixel_format);
        unimplemented!()
    }

    pub fn frame_header(&self) -> Option<JxlFrameHeader> {
        if let Some(file_header) = &self.codestream_parser.file_header {
            if let Some(frame) = &self.codestream_parser.frame {
                let header = frame.header();
                Some(JxlFrameHeader {
                    name: header.name.clone(),
                    duration: file_header
                        .image_metadata
                        .animation
                        .as_ref()
                        .map(|anim| header.duration(anim)),
                })
            } else {
                None
            }
        } else {
            None
        }
    }

    /// Number of passes we have full data for.
    pub fn num_completed_passes(&self) -> Option<usize> {
        None // TODO.
    }

    /// Rewinds a decoder to the start of the file, allowing past frames to be displayed again.
    pub fn rewind(&mut self) {
        // TODO(veluca): keep track of frame offsets for skipping.
        self.box_parser = BoxParser::new();
        self.codestream_parser = CodestreamParser::new();
    }

    pub fn has_more_frames(&self) -> bool {
        self.codestream_parser.has_more_frames
    }
}

// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#[cfg(test)]
use crate::api::FrameCallback;
use crate::{
    api::JxlFrameHeader,
    error::{Error, Result},
};

use super::{JxlBasicInfo, JxlColorProfile, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat, ProcessingResult};
use box_parser::BoxParser;
use codestream_parser::CodestreamParser;

mod box_parser;
mod codestream_parser;
mod process;

/// The current state of the decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecoderState {
    /// Initial state, waiting for image header.
    Initialized,
    /// Image header decoded, waiting for frame header.
    WithImageInfo,
    /// Frame header decoded, ready to decode frame data.
    WithFrameInfo,
}

/// FFI-friendly decoder API without typestate constraints.
///
/// This decoder uses runtime state checking instead of compile-time typestate,
/// making it easier to use from C/C++ FFI bindings.
///
/// # Input Buffer Management
///
/// The decoder has built-in input buffering. Use [`set_input`](Self::set_input)
/// to provide initial data, and [`append_input`](Self::append_input) to add more.
///
/// State transitions:
/// - `Initialized` → (process) → `WithImageInfo`
/// - `WithImageInfo` → (process) → `WithFrameInfo`
/// - `WithFrameInfo` → (decode_frame/skip_frame) → `WithImageInfo`
pub struct JxlDecoderInner {
    options: JxlDecoderOptions,
    box_parser: BoxParser,
    codestream_parser: CodestreamParser,
    state: DecoderState,
    input_buffer: Vec<u8>,
    input_consumed: usize,
    all_input_received: bool,
}

impl JxlDecoderInner {
    /// Creates a new decoder with the given options and, optionally, CMS.
    pub fn new(options: JxlDecoderOptions) -> Self {
        JxlDecoderInner {
            options,
            box_parser: BoxParser::new(),
            codestream_parser: CodestreamParser::new(),
            state: DecoderState::Initialized,
            input_buffer: Vec::new(),
            input_consumed: 0,
            all_input_received: false,
        }
    }

    /// Returns the current decoder state.
    pub fn state(&self) -> DecoderState {
        self.state
    }

    /// Sets input data for decoding, replacing any existing buffer.
    pub fn set_input(&mut self, data: &[u8], all_input: bool) {
        self.input_buffer.clear();
        self.input_buffer.extend_from_slice(data);
        self.input_consumed = 0;
        self.all_input_received = all_input;
    }

    /// Appends additional input data for incremental decoding.
    pub fn append_input(&mut self, data: &[u8], all_input: bool) {
        self.input_buffer.extend_from_slice(data);
        self.all_input_received = all_input;
    }

    /// Returns true if all input has been received.
    pub fn all_input_received(&self) -> bool {
        self.all_input_received
    }

    #[cfg(test)]
    pub fn set_frame_callback(&mut self, callback: Box<FrameCallback>) {
        self.codestream_parser.frame_callback = Some(callback);
    }

    #[cfg(test)]
    pub fn decoded_frames(&self) -> usize {
        self.codestream_parser.decoded_frames
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
    pub fn set_output_color_profile(&mut self, profile: JxlColorProfile) -> Result<()> {
        if let (JxlColorProfile::Icc(_), None) = (&profile, &self.options.cms) {
            return Err(Error::ICCOutputNoCMS);
        }
        self.codestream_parser.output_color_profile = Some(profile);
        Ok(())
    }

    pub fn current_pixel_format(&self) -> Option<&JxlPixelFormat> {
        self.codestream_parser.pixel_format.as_ref()
    }

    pub fn set_pixel_format(&mut self, pixel_format: JxlPixelFormat) {
        self.codestream_parser.pixel_format = Some(pixel_format);
    }

    pub fn frame_header(&self) -> Option<JxlFrameHeader> {
        let frame_header = self.codestream_parser.frame.as_ref()?.header();
        // The render pipeline always adds ExtendToImageDimensionsStage which extends
        // frames to the full image size. So the output size is always the image size,
        // not the frame's upsampled size.
        let size = self.codestream_parser.basic_info.as_ref()?.size;
        Some(JxlFrameHeader {
            name: frame_header.name.clone(),
            duration: self
                .codestream_parser
                .animation
                .as_ref()
                .map(|anim| frame_header.duration(anim)),
            size,
        })
    }

    /// Number of passes we have full data for.
    /// Returns the minimum number of passes completed across all groups.
    pub fn num_completed_passes(&self) -> Option<usize> {
        Some(self.codestream_parser.num_completed_passes())
    }

    /// Rewinds a decoder to the start of the file, allowing past frames to be displayed again.
    ///
    /// This fully resets the decoder. For animation loop playback, consider using
    /// [`rewind_for_animation`](Self::rewind_for_animation) instead.
    pub fn rewind(&mut self) {
        // TODO(veluca): keep track of frame offsets for skipping.
        self.box_parser = BoxParser::new();
        self.codestream_parser = CodestreamParser::new();
        self.state = DecoderState::Initialized;
        self.input_buffer.clear();
        self.input_consumed = 0;
        self.all_input_received = false;
    }

    /// Rewinds for animation loop replay, keeping pixel_format setting.
    ///
    /// This resets the decoder but preserves the pixel_format configuration,
    /// so the caller doesn't need to re-set it after rewinding.
    ///
    /// After calling this, provide input from the beginning of the file.
    /// Headers will be re-parsed, then frames can be decoded again.
    ///
    /// Returns `true` if pixel_format was preserved, `false` if none was set.
    pub fn rewind_for_animation(&mut self) -> bool {
        self.box_parser = BoxParser::new();
        let had_pixel_format = self.codestream_parser.rewind_for_animation().is_some();
        self.state = DecoderState::Initialized;
        self.input_consumed = 0;
        had_pixel_format
    }

    pub fn has_more_frames(&self) -> bool {
        self.codestream_parser.has_more_frames
    }

    /// Process the internal input buffer, advancing the decoder state.
    ///
    /// Returns `Ok(true)` when the state has advanced, `Ok(false)` when more
    /// input is needed.
    pub fn process_buffered(&mut self) -> Result<bool> {
        if self.state == DecoderState::WithFrameInfo {
            return Err(Error::InvalidDecoderState(
                "Use decode_frame_buffered() in WithFrameInfo state",
            ));
        }

        if self.input_consumed >= self.input_buffer.len() && !self.all_input_received {
            return Ok(false);
        }

        // Copy remaining input to avoid borrow conflicts with self.process()
        let remaining_input: Vec<u8> = self.input_buffer[self.input_consumed..].to_vec();
        let mut input = remaining_input.as_slice();
        let input_before = input.len();

        match self.process(&mut input, None)? {
            ProcessingResult::Complete { .. } => {
                self.input_consumed += input_before - input.len();
                self.state = match self.state {
                    DecoderState::Initialized => DecoderState::WithImageInfo,
                    DecoderState::WithImageInfo => DecoderState::WithFrameInfo,
                    DecoderState::WithFrameInfo => unreachable!(),
                };
                Ok(true)
            }
            ProcessingResult::NeedsMoreInput { .. } => {
                self.input_consumed += input_before - input.len();
                Ok(false)
            }
        }
    }

    /// Decode the current frame using the internal input buffer.
    ///
    /// Must be called when state is `WithFrameInfo`. After successful decoding,
    /// state transitions back to `WithImageInfo`.
    pub fn decode_frame_buffered(&mut self, buffer: &mut [u8]) -> Result<bool> {
        if self.state != DecoderState::WithFrameInfo {
            return Err(Error::InvalidDecoderState(
                "decode_frame_buffered() requires WithFrameInfo state",
            ));
        }

        let frame_header = self
            .frame_header()
            .ok_or(Error::InvalidDecoderState("Frame header not available"))?;
        let (width, height) = frame_header.size;
        let pixel_format = self
            .current_pixel_format()
            .ok_or(Error::InvalidDecoderState("Pixel format not set"))?;
        let bytes_per_pixel = pixel_format
            .bytes_per_pixel()
            .ok_or(Error::InvalidDecoderState(
                "Pixel format has no color output",
            ))?;
        let bytes_per_row = width * bytes_per_pixel;

        let output_buffer = JxlOutputBuffer::new(buffer, height, bytes_per_row);

        // Copy remaining input to avoid borrow conflicts with self.process()
        let remaining_input: Vec<u8> = self.input_buffer[self.input_consumed..].to_vec();
        let mut input = remaining_input.as_slice();
        let input_before = input.len();

        match self.process(&mut input, Some(&mut [output_buffer]))? {
            ProcessingResult::Complete { .. } => {
                self.input_consumed += input_before - input.len();
                self.state = DecoderState::WithImageInfo;
                Ok(true)
            }
            ProcessingResult::NeedsMoreInput { .. } => {
                self.input_consumed += input_before - input.len();
                Ok(false)
            }
        }
    }

    /// Skip the current frame using the internal input buffer.
    pub fn skip_frame_buffered(&mut self) -> Result<bool> {
        if self.state != DecoderState::WithFrameInfo {
            return Err(Error::InvalidDecoderState(
                "skip_frame_buffered() requires WithFrameInfo state",
            ));
        }

        // Copy remaining input to avoid borrow conflicts with self.process()
        let remaining_input: Vec<u8> = self.input_buffer[self.input_consumed..].to_vec();
        let mut input = remaining_input.as_slice();
        let input_before = input.len();

        match self.process(&mut input, None)? {
            ProcessingResult::Complete { .. } => {
                self.input_consumed += input_before - input.len();
                self.state = DecoderState::WithImageInfo;
                Ok(true)
            }
            ProcessingResult::NeedsMoreInput { .. } => {
                self.input_consumed += input_before - input.len();
                Ok(false)
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn set_use_simple_pipeline(&mut self, u: bool) {
        self.codestream_parser.set_use_simple_pipeline(u);
    }
}

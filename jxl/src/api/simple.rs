// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! FFI-friendly decoder API without typestate constraints.
//!
//! This module provides [`JxlDecoderSimple`], an alternative to the typestate-based
//! [`JxlDecoder`](super::JxlDecoder) API. It uses runtime state checking instead of
//! compile-time state transitions, making it easier to use from FFI bindings.
//!
//! # Example
//!
//! ```ignore
//! use jxl::api::{JxlDecoderSimple, JxlDecoderOptions, JxlPixelFormat};
//!
//! let mut decoder = JxlDecoderSimple::new(JxlDecoderOptions::default());
//! let mut input = file_data.as_slice();
//!
//! // Process until we have image info
//! while decoder.state() == DecoderState::Initialized {
//!     decoder.process(&mut input)?;
//! }
//!
//! // Configure pixel format
//! let info = decoder.basic_info().unwrap();
//! decoder.set_pixel_format(JxlPixelFormat::rgba8(info.num_extra_channels()));
//!
//! // Process frames
//! while decoder.has_more_frames() {
//!     while decoder.state() == DecoderState::WithImageInfo {
//!         decoder.process(&mut input)?;
//!     }
//!
//!     let mut buffer = vec![0u8; info.buffer_size(&format).unwrap()];
//!     decoder.decode_frame(&mut input, &mut buffer)?;
//! }
//! ```

use crate::error::{Error, Result};

use super::{
    JxlBasicInfo, JxlBitstreamInput, JxlColorProfile, JxlDecoderInner, JxlDecoderOptions,
    JxlFrameHeader, JxlOutputBuffer, JxlPixelFormat, ProcessingResult,
};

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

/// FFI-friendly decoder without typestate constraints.
///
/// This decoder uses runtime state checking instead of compile-time typestate,
/// making it easier to use from C/C++ FFI bindings or in situations where
/// typestate is inconvenient.
///
/// # Input Buffer Management
///
/// The decoder has built-in input buffering. Use [`set_input`](Self::set_input)
/// to provide initial data, and [`append_input`](Self::append_input) to add more.
/// The decoder tracks how much input has been consumed.
///
/// State transitions:
/// - `Initialized` → (process) → `WithImageInfo`
/// - `WithImageInfo` → (process) → `WithFrameInfo`
/// - `WithFrameInfo` → (decode_frame/skip_frame) → `WithImageInfo`
///
/// Use [`rewind()`](Self::rewind) to reset to `Initialized` state.
pub struct JxlDecoderSimple {
    inner: JxlDecoderInner,
    state: DecoderState,
    input_buffer: Vec<u8>,
    input_consumed: usize,
    all_input_received: bool,
}

impl JxlDecoderSimple {
    /// Creates a new decoder with the given options.
    pub fn new(options: JxlDecoderOptions) -> Self {
        Self {
            inner: JxlDecoderInner::new(options),
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

    /// Rewinds the decoder to the initial state.
    ///
    /// This allows decoding the same file again or switching to a new file.
    /// All parsed information (basic_info, pixel_format, etc.) is cleared.
    /// The input buffer is also cleared.
    pub fn rewind(&mut self) {
        self.inner.rewind();
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
    /// After calling this:
    /// - State becomes `Initialized`
    /// - Input consumption is reset (call set_input again with all data)
    /// - Headers will be re-parsed, then frames can be decoded again
    ///
    /// Returns `true` if pixel_format was preserved, `false` if none was set.
    ///
    /// # Example
    /// ```ignore
    /// // After decoding all frames of an animation:
    /// if decoder.rewind_for_animation() {
    ///     decoder.set_input(&file_data, true); // Reset input to beginning
    ///     // pixel_format is still set, just re-parse headers and decode frames
    /// }
    /// ```
    pub fn rewind_for_animation(&mut self) -> bool {
        let had_pixel_format = self.inner.rewind_for_animation();
        self.state = DecoderState::Initialized;
        self.input_consumed = 0;
        had_pixel_format
    }

    /// Sets input data for decoding, replacing any existing buffer.
    ///
    /// # Arguments
    /// * `data` - The input data
    /// * `all_input` - If true, this is all the input data that will be provided
    pub fn set_input(&mut self, data: &[u8], all_input: bool) {
        self.input_buffer.clear();
        self.input_buffer.extend_from_slice(data);
        self.input_consumed = 0;
        self.all_input_received = all_input;
    }

    /// Appends additional input data for incremental decoding.
    ///
    /// # Arguments
    /// * `data` - Additional input data to append
    /// * `all_input` - If true, this completes the input data
    pub fn append_input(&mut self, data: &[u8], all_input: bool) {
        self.input_buffer.extend_from_slice(data);
        self.all_input_received = all_input;
    }

    /// Returns true if all input has been received.
    pub fn all_input_received(&self) -> bool {
        self.all_input_received
    }

    /// Process the internal input buffer, advancing the decoder state.
    ///
    /// Returns `Ok(true)` when the state has advanced, `Ok(false)` when more
    /// input is needed. Use [`set_input`](Self::set_input) or
    /// [`append_input`](Self::append_input) to provide data first.
    ///
    /// # State Transitions
    /// - `Initialized` → `WithImageInfo`: When image header is fully parsed
    /// - `WithImageInfo` → `WithFrameInfo`: When frame header is fully parsed
    /// - `WithFrameInfo`: Use [`decode_frame_buffered`](Self::decode_frame_buffered) instead
    pub fn process_buffered(&mut self) -> Result<bool> {
        if self.state == DecoderState::WithFrameInfo {
            return Err(Error::InvalidDecoderState(
                "Use decode_frame_buffered() or skip_frame_buffered() in WithFrameInfo state",
            ));
        }

        let input_slice = &self.input_buffer[self.input_consumed..];
        if input_slice.is_empty() && !self.all_input_received {
            return Ok(false);
        }

        let mut input = input_slice;
        let input_before = input.len();

        match self.inner.process(&mut input, None)? {
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

    /// Process input data, advancing the decoder state.
    ///
    /// Returns `Ok(true)` when the state has advanced, `Ok(false)` when more
    /// input is needed.
    ///
    /// # State Transitions
    /// - `Initialized` → `WithImageInfo`: When image header is fully parsed
    /// - `WithImageInfo` → `WithFrameInfo`: When frame header is fully parsed
    /// - `WithFrameInfo`: Use [`decode_frame`](Self::decode_frame) instead
    pub fn process(&mut self, input: &mut impl JxlBitstreamInput) -> Result<bool> {
        if self.state == DecoderState::WithFrameInfo {
            return Err(Error::InvalidDecoderState(
                "Use decode_frame() or skip_frame() in WithFrameInfo state",
            ));
        }

        match self.inner.process(input, None)? {
            ProcessingResult::Complete { .. } => {
                self.state = match self.state {
                    DecoderState::Initialized => DecoderState::WithImageInfo,
                    DecoderState::WithImageInfo => DecoderState::WithFrameInfo,
                    DecoderState::WithFrameInfo => unreachable!(),
                };
                Ok(true)
            }
            ProcessingResult::NeedsMoreInput { .. } => Ok(false),
        }
    }

    /// Returns the image's basic information, if available.
    ///
    /// Available after state advances to `WithImageInfo` or later.
    pub fn basic_info(&self) -> Option<&JxlBasicInfo> {
        self.inner.basic_info()
    }

    /// Returns the current frame header, if available.
    ///
    /// Available when state is `WithFrameInfo`.
    pub fn frame_header(&self) -> Option<JxlFrameHeader> {
        if self.state != DecoderState::WithFrameInfo {
            return None;
        }
        self.inner.frame_header()
    }

    /// Retrieves the file's embedded color profile, if available.
    pub fn embedded_color_profile(&self) -> Option<&JxlColorProfile> {
        self.inner.embedded_color_profile()
    }

    /// Retrieves the current output color profile, if available.
    pub fn output_color_profile(&self) -> Option<&JxlColorProfile> {
        self.inner.output_color_profile()
    }

    /// Sets the output color profile.
    ///
    /// Must be called when state is `WithImageInfo` or later.
    pub fn set_output_color_profile(&mut self, profile: JxlColorProfile) -> Result<()> {
        self.inner.set_output_color_profile(profile)
    }

    /// Returns the current pixel format, if set.
    pub fn current_pixel_format(&self) -> Option<&JxlPixelFormat> {
        self.inner.current_pixel_format()
    }

    /// Sets the desired output pixel format.
    ///
    /// Should be called when state is `WithImageInfo`, before processing frames.
    pub fn set_pixel_format(&mut self, pixel_format: JxlPixelFormat) {
        self.inner.set_pixel_format(pixel_format);
    }

    /// Returns true if there are more frames to decode.
    pub fn has_more_frames(&self) -> bool {
        self.inner.has_more_frames()
    }

    /// Number of passes completed for the current frame.
    pub fn num_completed_passes(&self) -> Option<usize> {
        self.inner.num_completed_passes()
    }

    /// Decode the current frame directly into a buffer.
    ///
    /// Must be called when state is `WithFrameInfo`. After successful decoding,
    /// state transitions back to `WithImageInfo`.
    ///
    /// # Arguments
    /// * `input` - Input data source
    /// * `buffer` - Output buffer, must be correctly sized for the pixel format
    ///
    /// # Returns
    /// * `Ok(true)` - Frame decoded successfully, state is now `WithImageInfo`
    /// * `Ok(false)` - More input needed, call again with more data
    pub fn decode_frame(
        &mut self,
        input: &mut impl JxlBitstreamInput,
        buffer: &mut [u8],
    ) -> Result<bool> {
        if self.state != DecoderState::WithFrameInfo {
            return Err(Error::InvalidDecoderState(
                "decode_frame() requires WithFrameInfo state",
            ));
        }

        let frame_header = self
            .inner
            .frame_header()
            .ok_or(Error::InvalidDecoderState("Frame header not available"))?;
        let (width, height) = frame_header.size;
        let pixel_format = self
            .inner
            .current_pixel_format()
            .ok_or(Error::InvalidDecoderState("Pixel format not set"))?;
        let bytes_per_pixel = pixel_format
            .bytes_per_pixel()
            .ok_or(Error::InvalidDecoderState(
                "Pixel format has no color output",
            ))?;
        let bytes_per_row = width * bytes_per_pixel;

        let output_buffer = JxlOutputBuffer::new(buffer, height, bytes_per_row);

        match self.inner.process(input, Some(&mut [output_buffer]))? {
            ProcessingResult::Complete { .. } => {
                self.state = DecoderState::WithImageInfo;
                Ok(true)
            }
            ProcessingResult::NeedsMoreInput { .. } => Ok(false),
        }
    }

    /// Skip the current frame without decoding pixel data.
    ///
    /// Must be called when state is `WithFrameInfo`. After skipping,
    /// state transitions back to `WithImageInfo`.
    ///
    /// # Returns
    /// * `Ok(true)` - Frame skipped, state is now `WithImageInfo`
    /// * `Ok(false)` - More input needed
    pub fn skip_frame(&mut self, input: &mut impl JxlBitstreamInput) -> Result<bool> {
        if self.state != DecoderState::WithFrameInfo {
            return Err(Error::InvalidDecoderState(
                "skip_frame() requires WithFrameInfo state",
            ));
        }

        match self.inner.process(input, None)? {
            ProcessingResult::Complete { .. } => {
                self.state = DecoderState::WithImageInfo;
                Ok(true)
            }
            ProcessingResult::NeedsMoreInput { .. } => Ok(false),
        }
    }

    /// Decode the current frame using the internal input buffer.
    ///
    /// Must be called when state is `WithFrameInfo`. After successful decoding,
    /// state transitions back to `WithImageInfo`.
    ///
    /// # Arguments
    /// * `buffer` - Output buffer, must be correctly sized for the pixel format
    ///
    /// # Returns
    /// * `Ok(true)` - Frame decoded successfully, state is now `WithImageInfo`
    /// * `Ok(false)` - More input needed, call `append_input()` and try again
    pub fn decode_frame_buffered(&mut self, buffer: &mut [u8]) -> Result<bool> {
        if self.state != DecoderState::WithFrameInfo {
            return Err(Error::InvalidDecoderState(
                "decode_frame_buffered() requires WithFrameInfo state",
            ));
        }

        let frame_header = self
            .inner
            .frame_header()
            .ok_or(Error::InvalidDecoderState("Frame header not available"))?;
        let (width, height) = frame_header.size;
        let pixel_format = self
            .inner
            .current_pixel_format()
            .ok_or(Error::InvalidDecoderState("Pixel format not set"))?;
        let bytes_per_pixel = pixel_format
            .bytes_per_pixel()
            .ok_or(Error::InvalidDecoderState(
                "Pixel format has no color output",
            ))?;
        let bytes_per_row = width * bytes_per_pixel;

        let output_buffer = JxlOutputBuffer::new(buffer, height, bytes_per_row);

        let input_slice = &self.input_buffer[self.input_consumed..];
        let mut input = input_slice;
        let input_before = input.len();

        match self.inner.process(&mut input, Some(&mut [output_buffer]))? {
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
    ///
    /// Must be called when state is `WithFrameInfo`. After skipping,
    /// state transitions back to `WithImageInfo`.
    ///
    /// # Returns
    /// * `Ok(true)` - Frame skipped, state is now `WithImageInfo`
    /// * `Ok(false)` - More input needed, call `append_input()` and try again
    pub fn skip_frame_buffered(&mut self) -> Result<bool> {
        if self.state != DecoderState::WithFrameInfo {
            return Err(Error::InvalidDecoderState(
                "skip_frame_buffered() requires WithFrameInfo state",
            ));
        }

        let input_slice = &self.input_buffer[self.input_consumed..];
        let mut input = input_slice;
        let input_before = input.len();

        match self.inner.process(&mut input, None)? {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::JxlPixelFormat;

    #[test]
    fn test_decoder_simple_rgb8() {
        let file = std::fs::read("resources/test/basic.jxl").unwrap();
        let mut decoder = JxlDecoderSimple::new(JxlDecoderOptions::default());
        let mut input = file.as_slice();

        // Initial state
        assert_eq!(decoder.state(), DecoderState::Initialized);
        assert!(decoder.basic_info().is_none());

        // Process to get image info
        while decoder.state() == DecoderState::Initialized {
            let advanced = decoder.process(&mut input).unwrap();
            if !advanced && input.is_empty() {
                panic!("Unexpected end of input");
            }
        }
        assert_eq!(decoder.state(), DecoderState::WithImageInfo);

        // Now we have basic info
        let info = decoder.basic_info().unwrap().clone();
        assert!(info.size.0 > 0 && info.size.1 > 0);

        // Set pixel format
        let format = JxlPixelFormat::rgb8(info.num_extra_channels());
        decoder.set_pixel_format(format.clone());

        // Process to get frame info
        while decoder.state() == DecoderState::WithImageInfo {
            let advanced = decoder.process(&mut input).unwrap();
            if !advanced && input.is_empty() {
                panic!("Unexpected end of input");
            }
        }
        assert_eq!(decoder.state(), DecoderState::WithFrameInfo);

        // Now we have frame header
        let frame_header = decoder.frame_header().unwrap();
        assert_eq!(frame_header.size, info.size);

        // Decode frame
        let buffer_size = info.buffer_size(&format).unwrap();
        let mut buffer = vec![0u8; buffer_size];

        while !decoder.decode_frame(&mut input, &mut buffer).unwrap() {
            if input.is_empty() {
                panic!("Unexpected end of input");
            }
        }

        // State should be back to WithImageInfo
        assert_eq!(decoder.state(), DecoderState::WithImageInfo);

        // Buffer should have data
        assert!(buffer.iter().any(|&b| b != 0));
    }

    #[test]
    fn test_decoder_simple_rgba8_with_alpha() {
        let file = std::fs::read("resources/test/dice.jxl").unwrap();
        let mut decoder = JxlDecoderSimple::new(JxlDecoderOptions::default());
        let mut input = file.as_slice();

        // Process to image info
        while decoder.state() == DecoderState::Initialized {
            decoder.process(&mut input).unwrap();
        }

        let info = decoder.basic_info().unwrap().clone();
        assert!(info.num_extra_channels() > 0);

        // Set RGBA format
        let format = JxlPixelFormat::rgba8(info.num_extra_channels());
        decoder.set_pixel_format(format.clone());

        // Process to frame info
        while decoder.state() == DecoderState::WithImageInfo {
            decoder.process(&mut input).unwrap();
        }

        // Decode frame
        let mut buffer = vec![0u8; info.buffer_size(&format).unwrap()];
        while !decoder.decode_frame(&mut input, &mut buffer).unwrap() {}

        // Verify RGBA data
        assert!(buffer.iter().any(|&b| b != 0));
        assert_eq!(buffer.len(), info.size.0 * info.size.1 * 4);
    }

    #[test]
    fn test_decoder_simple_rewind() {
        let file = std::fs::read("resources/test/basic.jxl").unwrap();
        let mut decoder = JxlDecoderSimple::new(JxlDecoderOptions::default());
        let mut input = file.as_slice();

        // Decode once
        while decoder.state() == DecoderState::Initialized {
            decoder.process(&mut input).unwrap();
        }
        assert!(decoder.basic_info().is_some());

        // Rewind
        decoder.rewind();
        assert_eq!(decoder.state(), DecoderState::Initialized);
        assert!(decoder.basic_info().is_none());

        // Decode again
        input = file.as_slice();
        while decoder.state() == DecoderState::Initialized {
            decoder.process(&mut input).unwrap();
        }
        assert!(decoder.basic_info().is_some());
    }

    #[test]
    fn test_decoder_simple_state_errors() {
        let mut decoder = JxlDecoderSimple::new(JxlDecoderOptions::default());

        // decode_frame should fail in Initialized state
        let mut dummy_buf = vec![0u8; 100];
        let mut empty_input: &[u8] = &[];
        let result = decoder.decode_frame(&mut empty_input, &mut dummy_buf);
        assert!(result.is_err());

        // skip_frame should fail in Initialized state
        let result = decoder.skip_frame(&mut empty_input);
        assert!(result.is_err());
    }

    #[test]
    fn test_decoder_simple_rewind_for_animation() {
        let file = std::fs::read("resources/test/dice.jxl").unwrap();
        let mut decoder = JxlDecoderSimple::new(JxlDecoderOptions::default());
        let mut input = file.as_slice();

        // Decode first time
        while decoder.state() == DecoderState::Initialized {
            decoder.process(&mut input).unwrap();
        }

        let info = decoder.basic_info().unwrap().clone();
        let format = JxlPixelFormat::rgba8(info.num_extra_channels());
        decoder.set_pixel_format(format.clone());

        while decoder.state() == DecoderState::WithImageInfo {
            decoder.process(&mut input).unwrap();
        }

        let mut buffer1 = vec![0u8; info.buffer_size(&format).unwrap()];
        while !decoder.decode_frame(&mut input, &mut buffer1).unwrap() {}

        // Now use rewind_for_animation
        assert!(decoder.rewind_for_animation());
        // State goes back to Initialized (headers need re-parsing)
        assert_eq!(decoder.state(), DecoderState::Initialized);

        // basic_info is cleared (full reset)
        assert!(decoder.basic_info().is_none());

        // pixel_format should still be set (the main benefit)
        assert!(decoder.current_pixel_format().is_some());

        // Decode again with fresh input - need to re-parse headers
        input = file.as_slice();

        // Process through Initialized state (re-parse headers)
        while decoder.state() == DecoderState::Initialized {
            decoder.process(&mut input).unwrap();
        }

        // Process through WithImageInfo to get frame info
        while decoder.state() == DecoderState::WithImageInfo {
            decoder.process(&mut input).unwrap();
        }

        let mut buffer2 = vec![0u8; info.buffer_size(&format).unwrap()];
        while !decoder.decode_frame(&mut input, &mut buffer2).unwrap() {}

        // Both decodes should produce the same result
        assert_eq!(buffer1, buffer2);
    }

    #[test]
    fn test_rewind_for_animation_returns_false_without_pixel_format() {
        let mut decoder = JxlDecoderSimple::new(JxlDecoderOptions::default());

        // Returns false when no pixel_format was set
        assert!(!decoder.rewind_for_animation());
        assert_eq!(decoder.state(), DecoderState::Initialized);
    }

    #[test]
    fn test_decoder_simple_buffered_api() {
        let file = std::fs::read("resources/test/basic.jxl").unwrap();
        let mut decoder = JxlDecoderSimple::new(JxlDecoderOptions::default());

        // Set all input at once
        decoder.set_input(&file, true);

        // Process to get image info using buffered API
        while decoder.state() == DecoderState::Initialized {
            let advanced = decoder.process_buffered().unwrap();
            if !advanced {
                panic!("Should have enough data");
            }
        }
        assert_eq!(decoder.state(), DecoderState::WithImageInfo);

        // Get basic info and set pixel format
        let info = decoder.basic_info().unwrap().clone();
        let format = JxlPixelFormat::rgba8(info.num_extra_channels());
        decoder.set_pixel_format(format.clone());

        // Process to frame info
        while decoder.state() == DecoderState::WithImageInfo {
            decoder.process_buffered().unwrap();
        }
        assert_eq!(decoder.state(), DecoderState::WithFrameInfo);

        // Decode frame using buffered API
        let mut buffer = vec![0u8; info.buffer_size(&format).unwrap()];
        while !decoder.decode_frame_buffered(&mut buffer).unwrap() {}

        // Verify buffer has data
        assert!(buffer.iter().any(|&b| b != 0));
    }

    #[test]
    fn test_decoder_simple_buffered_incremental() {
        let file = std::fs::read("resources/test/basic.jxl").unwrap();
        let mut decoder = JxlDecoderSimple::new(JxlDecoderOptions::default());

        // Feed data in chunks
        let chunk_size = 100;
        let mut pos = 0;

        // Process header
        while decoder.state() == DecoderState::Initialized {
            if pos < file.len() {
                let end = (pos + chunk_size).min(file.len());
                decoder.append_input(&file[pos..end], end == file.len());
                pos = end;
            }
            if !decoder.process_buffered().unwrap() && pos >= file.len() {
                panic!("Not enough data for header");
            }
        }

        let info = decoder.basic_info().unwrap().clone();
        let format = JxlPixelFormat::rgba8(info.num_extra_channels());
        decoder.set_pixel_format(format.clone());

        // Process frame header
        while decoder.state() == DecoderState::WithImageInfo {
            if pos < file.len() {
                let end = (pos + chunk_size).min(file.len());
                decoder.append_input(&file[pos..end], end == file.len());
                pos = end;
            }
            if !decoder.process_buffered().unwrap() && pos >= file.len() {
                panic!("Not enough data for frame header");
            }
        }

        // Decode frame
        let mut buffer = vec![0u8; info.buffer_size(&format).unwrap()];
        while !decoder.decode_frame_buffered(&mut buffer).unwrap() {
            if pos < file.len() {
                let end = (pos + chunk_size).min(file.len());
                decoder.append_input(&file[pos..end], end == file.len());
                pos = end;
            }
        }

        assert!(buffer.iter().any(|&b| b != 0));
    }
}

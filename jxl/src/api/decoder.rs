// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::{
    JxlBasicInfo, JxlBitstreamInput, JxlCms, JxlColorProfile, JxlDecoderInner, JxlDecoderOptions,
    JxlOutputBuffer, JxlPixelFormat, ProcessingResult,
};
use crate::{error::Result, headers::frame_header::FrameHeader};
use states::*;
use std::marker::PhantomData;

pub mod states {
    pub trait JxlState {}
    pub struct Initialized;
    pub struct WithImageInfo;
    pub struct WithFrameInfo;
    impl JxlState for Initialized {}
    impl JxlState for WithImageInfo {}
    impl JxlState for WithFrameInfo {}
}

// Q: do we plan to add support for box decoding?
// If we do, one way is to take a callback &[u8; 4] -> Box<dyn Write>.

/// High level API using the typestate pattern to forbid invalid usage.
pub struct JxlDecoder<State: JxlState> {
    inner: JxlDecoderInner,
    _state: PhantomData<State>,
}

impl<S: JxlState> JxlDecoder<S> {
    fn wrap_inner(inner: JxlDecoderInner) -> Self {
        Self {
            inner,
            _state: PhantomData,
        }
    }

    /// Rewinds a decoder to the start of the file, allowing past frames to be displayed again.
    pub fn rewind(mut self) -> JxlDecoder<Initialized> {
        self.inner.rewind();
        JxlDecoder::wrap_inner(self.inner)
    }

    fn map_inner_processing_result<SuccessState: JxlState>(
        self,
        inner_result: ProcessingResult<(), ()>,
    ) -> ProcessingResult<JxlDecoder<SuccessState>, Self> {
        match inner_result {
            ProcessingResult::Complete { .. } => ProcessingResult::Complete {
                result: JxlDecoder::wrap_inner(self.inner),
            },
            ProcessingResult::NeedsMoreInput { size_hint, .. } => {
                ProcessingResult::NeedsMoreInput {
                    size_hint,
                    fallback: self,
                }
            }
        }
    }
}

impl JxlDecoder<Initialized> {
    pub fn new(options: JxlDecoderOptions) -> Self {
        Self::wrap_inner(JxlDecoderInner::new(options, None))
    }

    pub fn new_with_cms(options: JxlDecoderOptions, cms: impl JxlCms + 'static) -> Self {
        Self::wrap_inner(JxlDecoderInner::new(options, Some(Box::new(cms))))
    }

    pub fn process(
        mut self,
        input: &mut impl JxlBitstreamInput,
    ) -> Result<ProcessingResult<JxlDecoder<WithImageInfo>, Self>> {
        let inner_result = self.inner.process(input, None)?;
        Ok(self.map_inner_processing_result(inner_result))
    }
}

impl JxlDecoder<WithImageInfo> {
    // TODO(veluca): once frame skipping is implemented properly, expose that in the API.

    /// Obtains the image's basic information.
    pub fn basic_info(&self) -> &JxlBasicInfo {
        self.inner.basic_info().unwrap()
    }

    /// Retrieves the file's color profile.
    pub fn embedded_color_profile(&self) -> &JxlColorProfile {
        self.inner.embedded_color_profile().unwrap()
    }

    /// Retrieves the current output color profile.
    pub fn output_color_profile(&self) -> &JxlColorProfile {
        self.inner.output_color_profile().unwrap()
    }

    /// Specifies the preferred color profile to be used for outputting data.
    /// Same semantics as JxlDecoderSetOutputColorProfile.
    pub fn set_output_color_profile(&mut self, profile: &JxlColorProfile) -> Result<()> {
        self.inner.set_output_color_profile(profile)
    }

    pub fn current_pixel_format(&self) -> &JxlPixelFormat {
        self.inner.current_pixel_format().unwrap()
    }

    pub fn set_pixel_format(&mut self, pixel_format: JxlPixelFormat) {
        self.inner.set_pixel_format(pixel_format);
    }

    pub fn process(
        mut self,
        input: &mut impl JxlBitstreamInput,
    ) -> Result<ProcessingResult<JxlDecoder<WithFrameInfo>, Self>> {
        let inner_result = self.inner.process(input, None)?;
        Ok(self.map_inner_processing_result(inner_result))
    }

    pub fn has_more_frames(&self) -> bool {
        self.inner.has_more_frames()
    }
}

impl JxlDecoder<WithFrameInfo> {
    /// Skip the current frame.
    pub fn skip_frame(
        mut self,
        input: &mut impl JxlBitstreamInput,
    ) -> Result<ProcessingResult<JxlDecoder<WithImageInfo>, Self>> {
        let inner_result = self.inner.process(input, None)?;
        Ok(self.map_inner_processing_result(inner_result))
    }

    // TODO: don't use the raw bitstream type; include name and extra channel blend info.
    pub fn frame_header(&self) -> &FrameHeader {
        self.inner.frame_header().unwrap()
    }

    /// Number of passes we have full data for.
    pub fn num_completed_passes(&self) -> usize {
        self.inner.num_completed_passes().unwrap()
    }

    /// Draws all the pixels we have data for.
    pub fn flush_pixels(&mut self, buffers: &mut [JxlOutputBuffer<'_>]) -> Result<()> {
        self.inner.flush_pixels(buffers)
    }

    /// Guarantees to populate exactly the appropriate part of the buffers.
    /// Wants one buffer for each non-ignored pixel type, i.e. color channels and each extra channel.
    pub fn process<In: JxlBitstreamInput>(
        mut self,
        input: &mut In,
        buffers: &mut [JxlOutputBuffer<'_>],
    ) -> Result<ProcessingResult<JxlDecoder<WithImageInfo>, Self>> {
        let inner_result = self.inner.process(input, Some(buffers))?;
        Ok(self.map_inner_processing_result(inner_result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::{JxlColorType::Rgb, JxlDecoderOptions, JxlOutputBuffer};
    use crate::error::Error;
    use jxl_macros::for_each_test_file;
    use std::mem::MaybeUninit;
    use std::path::Path;

    fn decode_test_file(path: &Path) -> Result<(), crate::error::Error> {
        // Load the test image
        let test_data = std::fs::read(path).expect("Failed to read test file");
        let mut input = test_data.as_slice();

        // Create decoder with default options
        let options = JxlDecoderOptions::default();
        let mut initialized_decoder = JxlDecoder::<states::Initialized>::new(options);

        // Process until we have image info
        let mut decoder_with_image_info = loop {
            match initialized_decoder.process(&mut input).unwrap() {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if input.is_empty() {
                        panic!("Unexpected end of input while reading image info");
                    }
                    initialized_decoder = fallback;
                }
            }
        };

        // Get basic info
        let basic_info = decoder_with_image_info.basic_info();
        assert!(basic_info.bit_depth.bits_per_sample() > 0);
        let orientation = basic_info.orientation;

        // Get image dimensions (after upsampling, which is the actual output size)
        let (width, height) = basic_info.size;
        assert!(width > 0);
        assert!(height > 0);

        // Check if orientation transposes dimensions
        let (buffer_width, buffer_height) = if orientation.is_transposing() {
            (height, width)
        } else {
            (width, height)
        };

        // Get pixel format info
        let pixel_format = decoder_with_image_info.current_pixel_format().clone();
        let num_channels = pixel_format.color_type.samples_per_pixel();
        assert!(num_channels > 0);

        let mut frame_count = 0;

        loop {
            // Process until we have frame info
            let mut decoder_with_frame_info = loop {
                match decoder_with_image_info.process(&mut input).unwrap() {
                    ProcessingResult::Complete { result } => break result,
                    ProcessingResult::NeedsMoreInput { fallback, .. } => {
                        if input.is_empty() {
                            panic!("Unexpected end of input while reading frame info");
                        }
                        decoder_with_image_info = fallback;
                    }
                }
            };

            // Prepare output buffers
            let mut output_buffers: Vec<Vec<MaybeUninit<u8>>> = Vec::new();

            // For RGB images, first buffer holds interleaved RGB data
            if pixel_format.color_type == Rgb {
                // First buffer for interleaved RGB (3 channels * 4 bytes per float)
                output_buffers.push(vec![
                    MaybeUninit::uninit();
                    buffer_width * buffer_height * 12
                ]);
                // Additional buffers for extra channels
                for _ in 3..num_channels {
                    output_buffers.push(vec![
                        MaybeUninit::uninit();
                        buffer_width * buffer_height * 4
                    ]);
                }
            } else {
                // For grayscale or other formats, one buffer per channel
                for _ in 0..num_channels {
                    output_buffers.push(vec![
                        MaybeUninit::uninit();
                        buffer_width * buffer_height * 4
                    ]);
                }
            }

            let mut output_slices: Vec<JxlOutputBuffer> = output_buffers
                .iter_mut()
                .enumerate()
                .map(|(i, buffer)| {
                    let bytes_per_pixel = if i == 0 && pixel_format.color_type == Rgb {
                        12 // Interleaved RGB
                    } else {
                        4 // Single channel
                    };
                    JxlOutputBuffer::new_uninit(
                        buffer.as_mut_slice(),
                        buffer_height,
                        bytes_per_pixel * buffer_width,
                    )
                })
                .collect();

            decoder_with_image_info = loop {
                match decoder_with_frame_info
                    .process(&mut input, &mut output_slices)
                    .unwrap()
                {
                    ProcessingResult::Complete { result } => break result,
                    ProcessingResult::NeedsMoreInput { fallback, .. } => {
                        if input.is_empty() {
                            panic!("Unexpected end of input while decoding frame");
                        }
                        decoder_with_frame_info = fallback;
                    }
                }
            };

            // Verify we decoded something
            if pixel_format.color_type == Rgb {
                // For RGB, first buffer contains interleaved RGB data
                assert!(!output_buffers.is_empty());
                assert_eq!(output_buffers[0].len(), width * height * 12); // 3 channels * 4 bytes
                // Additional buffers for extra channels
                for buffer in &output_buffers[1..] {
                    assert_eq!(buffer.len(), width * height * 4);
                }
            } else {
                // For other formats, one buffer per channel
                assert_eq!(output_buffers.len(), num_channels);
                for buffer in &output_buffers {
                    assert_eq!(buffer.len(), width * height * 4);
                }
            }

            frame_count += 1;

            // Check if there are more frames
            if !decoder_with_image_info.has_more_frames() {
                break;
            }
        }

        // Ensure we decoded at least one frame
        assert!(frame_count > 0, "No frames were decoded");

        Ok(())
    }

    for_each_test_file!(decode_test_file);
}

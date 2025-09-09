// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::{
    JxlBasicInfo, JxlBitstreamInput, JxlCms, JxlColorProfile, JxlDecoderInner, JxlDecoderOptions,
    JxlOutputBuffer, JxlPixelFormat, ProcessingResult,
};
#[cfg(test)]
use crate::frame::Frame;
use crate::{api::JxlFrameHeader, error::Result};
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

#[cfg(test)]
pub type FrameCallback = dyn FnMut(&Frame, usize) -> Result<()>;

impl<S: JxlState> JxlDecoder<S> {
    fn wrap_inner(inner: JxlDecoderInner) -> Self {
        Self {
            inner,
            _state: PhantomData,
        }
    }

    /// Sets a callback that processes all frames by calling `callback(frame, frame_index)`.
    #[cfg(test)]
    pub fn set_frame_callback(&mut self, callback: Box<FrameCallback>) {
        self.inner.set_frame_callback(callback);
    }

    #[cfg(test)]
    pub fn decoded_frames(&self) -> usize {
        self.inner.decoded_frames()
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
    pub fn frame_header(&self) -> JxlFrameHeader {
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
pub(crate) mod tests {
    use super::*;
    use crate::api::JxlDecoderOptions;
    use crate::error::Error;
    use crate::image::Image;
    use jxl_macros::for_each_test_file;
    use std::path::Path;

    #[test]
    fn decode_small_chunks() {
        arbtest::arbtest(|u| {
            decode(
                &std::fs::read("resources/test/green_queen_vardct_e3.jxl").unwrap(),
                u.arbitrary::<u8>().unwrap() as usize + 1,
                None,
            )
            .unwrap();
            Ok(())
        });
    }

    #[allow(clippy::type_complexity)]
    pub fn decode(
        mut input: &[u8],
        chunk_size: usize,
        callback: Option<Box<dyn FnMut(&Frame, usize) -> Result<(), Error>>>,
    ) -> Result<usize, Error> {
        let options = JxlDecoderOptions::default();
        let mut initialized_decoder = JxlDecoder::<states::Initialized>::new(options);

        if let Some(callback) = callback {
            initialized_decoder.set_frame_callback(callback);
        }

        let mut chunk_input = &input[0..0];

        macro_rules! advance_decoder {
            ($decoder: ident $(, $extra_arg: expr)?) => {
                loop {
                    chunk_input =
                        &input[..(chunk_input.len().saturating_add(chunk_size)).min(input.len())];
                    let available_before = chunk_input.len();
                    let process_result = $decoder.process(&mut chunk_input $(, $extra_arg)?);
                    input = &input[(available_before - chunk_input.len())..];
                    match process_result.unwrap() {
                        ProcessingResult::Complete { result } => break result,
                        ProcessingResult::NeedsMoreInput { fallback, .. } => {
                            if input.is_empty() {
                                panic!("Unexpected end of input");
                            }
                            $decoder = fallback;
                        }
                    }
                }
            };
        }

        // Process until we have image info
        let mut decoder_with_image_info = advance_decoder!(initialized_decoder);

        // Get basic info
        let basic_info = decoder_with_image_info.basic_info().clone();
        assert!(basic_info.bit_depth.bits_per_sample() > 0);

        // Get image dimensions (after upsampling, which is the actual output size)
        let (buffer_width, buffer_height) = basic_info.size;
        assert!(buffer_width > 0);
        assert!(buffer_height > 0);

        // Get pixel format info
        // TODO(veluca): this relies on the default pixel format using floats. We should not do
        // this, and instead call set_pixel_format, but that is currently not implemented.
        let pixel_format = decoder_with_image_info.current_pixel_format().clone();

        let num_channels = pixel_format.color_type.samples_per_pixel();
        assert!(num_channels > 0);

        loop {
            // Process until we have frame info
            let mut decoder_with_frame_info = advance_decoder!(decoder_with_image_info);

            // First channel is interleaved.
            let mut buffers = vec![Image::new_constant(
                (buffer_width * num_channels, buffer_height),
                f32::NAN,
            )?];

            for ecf in pixel_format.extra_channel_format.iter() {
                if ecf.is_none() {
                    continue;
                }
                buffers.push(Image::new_constant(
                    (buffer_width, buffer_height),
                    f32::NAN,
                )?);
            }

            let mut api_buffers: Vec<_> = buffers
                .iter_mut()
                .map(JxlOutputBuffer::from_image)
                .collect();

            decoder_with_image_info = advance_decoder!(decoder_with_frame_info, &mut api_buffers);

            // All pixels should have been overwritten, so they should no longer be NaNs.
            for buf in buffers {
                let (xs, ys) = buf.size();
                for y in 0..ys {
                    for x in 0..xs {
                        assert!(!buf.as_rect().row(y)[x].is_nan());
                    }
                }
            }

            // Check if there are more frames
            if !decoder_with_image_info.has_more_frames() {
                let decoded_frames = decoder_with_image_info.decoded_frames();

                // Ensure we decoded at least one frame
                assert!(decoded_frames > 0, "No frames were decoded");

                return Ok(decoded_frames);
            }
        }
    }

    fn decode_test_file(path: &Path) -> Result<(), Error> {
        decode(&std::fs::read(path)?, usize::MAX, None)?;
        Ok(())
    }

    for_each_test_file!(decode_test_file);
}

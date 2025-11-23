// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::{
    JxlBasicInfo, JxlBitstreamInput, JxlColorProfile, JxlDecoderInner, JxlDecoderOptions,
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
        Self::wrap_inner(JxlDecoderInner::new(options))
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
    pub fn set_output_color_profile(&mut self, profile: JxlColorProfile) -> Result<()> {
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

    #[cfg(test)]
    pub(crate) fn set_use_simple_pipeline(&mut self, u: bool) {
        self.inner.set_use_simple_pipeline(u);
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
    use crate::image::{Image, Rect};
    use crate::util::test::assert_almost_abs_eq_coords;
    use jxl_macros::for_each_test_file;
    use std::path::Path;

    #[test]
    fn decode_small_chunks() {
        arbtest::arbtest(|u| {
            decode(
                &std::fs::read("resources/test/green_queen_vardct_e3.jxl").unwrap(),
                u.arbitrary::<u8>().unwrap() as usize + 1,
                false,
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
        use_simple_pipeline: bool,
        callback: Option<Box<dyn FnMut(&Frame, usize) -> Result<(), Error>>>,
    ) -> Result<(usize, Vec<Vec<Image<f32>>>), Error> {
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
        decoder_with_image_info.set_use_simple_pipeline(use_simple_pipeline);

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

        let mut frames = vec![];

        loop {
            // Process until we have frame info
            let mut decoder_with_frame_info = advance_decoder!(decoder_with_image_info);

            // First channel is interleaved.
            let mut buffers = vec![Image::new_with_value(
                (buffer_width * num_channels, buffer_height),
                f32::NAN,
            )?];

            for ecf in pixel_format.extra_channel_format.iter() {
                if ecf.is_none() {
                    continue;
                }
                buffers.push(Image::new_with_value(
                    (buffer_width, buffer_height),
                    f32::NAN,
                )?);
            }

            let mut api_buffers: Vec<_> = buffers
                .iter_mut()
                .map(|b| {
                    JxlOutputBuffer::from_image_rect_mut(
                        b.get_rect_mut(Rect {
                            origin: (0, 0),
                            size: b.size(),
                        })
                        .into_raw(),
                    )
                })
                .collect();

            decoder_with_image_info = advance_decoder!(decoder_with_frame_info, &mut api_buffers);

            // All pixels should have been overwritten, so they should no longer be NaNs.
            for buf in buffers.iter() {
                let (xs, ys) = buf.size();
                for y in 0..ys {
                    let row = buf.row(y);
                    for (x, v) in row.iter().enumerate() {
                        assert!(!v.is_nan(), "NaN at {x} {y} (image size {xs}x{ys})");
                    }
                }
            }

            frames.push(buffers);

            // Check if there are more frames
            if !decoder_with_image_info.has_more_frames() {
                let decoded_frames = decoder_with_image_info.decoded_frames();

                // Ensure we decoded at least one frame
                assert!(decoded_frames > 0, "No frames were decoded");

                return Ok((decoded_frames, frames));
            }
        }
    }

    fn decode_test_file(path: &Path) -> Result<(), Error> {
        decode(&std::fs::read(path)?, usize::MAX, false, None)?;
        Ok(())
    }

    for_each_test_file!(decode_test_file);

    fn compare_pipelines(path: &Path) -> Result<(), Error> {
        let file = std::fs::read(path)?;
        let simple_frames = decode(&file, usize::MAX, true, None)?.1;
        let frames = decode(&file, usize::MAX, false, None)?.1;
        assert_eq!(frames.len(), simple_frames.len());
        for (fc, (f, sf)) in frames
            .into_iter()
            .zip(simple_frames.into_iter())
            .enumerate()
        {
            assert_eq!(
                f.len(),
                sf.len(),
                "Frame {fc} has different channels counts",
            );
            for (c, (b, sb)) in f.into_iter().zip(sf.into_iter()).enumerate() {
                assert_eq!(
                    b.size(),
                    sb.size(),
                    "Channel {c} in frame {fc} has different sizes",
                );
                // TODO(veluca): This check actually succeeds if we disable SIMD.
                // With SIMD, the exact output of computations in epf.rs appear to depend on the
                // lane that the computation was done in (???). We should investigate this.
                // b.as_rect().check_equal(sb.as_rect());
                let sz = b.size();
                if false {
                    let f = std::fs::File::create(Path::new("/tmp/").join(format!(
                        "{}_diff_chan{c}.pbm",
                        path.as_os_str().to_string_lossy().replace("/", "_")
                    )))?;
                    use std::io::Write;
                    let mut f = std::io::BufWriter::new(f);
                    writeln!(f, "P1\n{} {}", sz.0, sz.1)?;
                    for y in 0..sz.1 {
                        for x in 0..sz.0 {
                            if (b.row(y)[x] - sb.row(y)[x]).abs() > 1e-8 {
                                write!(f, "1")?;
                            } else {
                                write!(f, "0")?;
                            }
                        }
                    }
                    drop(f);
                }
                for y in 0..sz.1 {
                    for x in 0..sz.0 {
                        assert_almost_abs_eq_coords(b.row(y)[x], sb.row(y)[x], 1e-5, (x, y), c);
                    }
                }
            }
        }
        Ok(())
    }

    for_each_test_file!(compare_pipelines);

    /// Test that flush_pixels works when no frame is currently being decoded.
    #[test]
    fn test_flush_pixels_no_frame() {
        // Load a test file
        let file = std::fs::read("resources/test/green_queen_vardct_e3.jxl").unwrap();
        let mut input = &file[..];
        let options = JxlDecoderOptions::default();
        let decoder = JxlDecoder::<states::Initialized>::new(options);

        // Process until we have image info
        let mut chunk_input = &input[0..0];
        let decoder_with_image_info;
        {
            let mut d = decoder;
            loop {
                chunk_input = &input[..(chunk_input.len().saturating_add(1024)).min(input.len())];
                let available_before = chunk_input.len();
                let process_result = d.process(&mut chunk_input);
                input = &input[(available_before - chunk_input.len())..];
                match process_result.unwrap() {
                    ProcessingResult::Complete { result } => {
                        decoder_with_image_info = result;
                        break;
                    }
                    ProcessingResult::NeedsMoreInput { fallback, .. } => {
                        d = fallback;
                    }
                }
            }
        }

        // Process until we have frame info
        let mut decoder_with_frame_info;
        {
            let mut d = decoder_with_image_info;
            loop {
                chunk_input = &input[..(chunk_input.len().saturating_add(1024)).min(input.len())];
                let available_before = chunk_input.len();
                let process_result = d.process(&mut chunk_input);
                input = &input[(available_before - chunk_input.len())..];
                match process_result.unwrap() {
                    ProcessingResult::Complete { result } => {
                        decoder_with_frame_info = result;
                        break;
                    }
                    ProcessingResult::NeedsMoreInput { fallback, .. } => {
                        d = fallback;
                    }
                }
            }
        }

        // Get the basic info for buffer sizes
        let basic_info = decoder_with_frame_info.inner.basic_info().unwrap().clone();
        let (buffer_width, buffer_height) = basic_info.size;
        let pixel_format = decoder_with_frame_info
            .inner
            .current_pixel_format()
            .unwrap()
            .clone();
        let num_channels = pixel_format.color_type.samples_per_pixel();

        // Create buffer
        let mut buffer = Image::new_with_value(
            (buffer_width * num_channels, buffer_height),
            f32::NAN,
        )
        .unwrap();
        let api_buffer = JxlOutputBuffer::from_image_rect_mut(
            buffer
                .get_rect_mut(Rect {
                    origin: (0, 0),
                    size: buffer.size(),
                })
                .into_raw(),
        );

        // Call flush_pixels - should not panic even though we haven't received any section data yet
        let result = decoder_with_frame_info.flush_pixels(&mut [api_buffer]);
        assert!(result.is_ok());
    }

    /// Test that flush_pixels can be called and returns Ok even with no render pipeline.
    #[test]
    fn test_flush_pixels_inner_no_pipeline() {
        // Test that flush_pixels at the inner level handles no-pipeline case gracefully
        let file = std::fs::read("resources/test/green_queen_vardct_e3.jxl").unwrap();
        let mut input = &file[..];
        let options = JxlDecoderOptions::default();
        let mut inner = crate::api::JxlDecoderInner::new(options);

        // Get image info first
        let mut chunk = &input[..4096.min(input.len())];
        let _ = inner.process(&mut chunk, None);
        input = &input[(4096.min(input.len()) - chunk.len())..];

        // Get frame info
        chunk = &input[..4096.min(input.len())];
        let _ = inner.process(&mut chunk, None);

        // Create a small test buffer using Image
        let mut test_image: Image<f32> = Image::new((4, 4)).unwrap();
        let output_buffer = JxlOutputBuffer::from_image_rect_mut(
            test_image
                .get_rect_mut(Rect {
                    origin: (0, 0),
                    size: test_image.size(),
                })
                .into_raw(),
        );

        // flush_pixels should return Ok even if no pipeline is set up
        let result = inner.flush_pixels(&mut [output_buffer]);
        assert!(result.is_ok());
    }

    /// Test that flush_pixels can be called multiple times without error.
    #[test]
    fn test_flush_pixels_multiple_calls() {
        let file = std::fs::read("resources/test/green_queen_vardct_e3.jxl").unwrap();
        let mut input = &file[..];
        let options = JxlDecoderOptions::default();
        let decoder = JxlDecoder::<states::Initialized>::new(options);

        let mut chunk_input = &input[0..0];
        let decoder_with_image_info;
        {
            let mut d = decoder;
            loop {
                chunk_input = &input[..(chunk_input.len().saturating_add(1024)).min(input.len())];
                let available_before = chunk_input.len();
                let process_result = d.process(&mut chunk_input);
                input = &input[(available_before - chunk_input.len())..];
                match process_result.unwrap() {
                    ProcessingResult::Complete { result } => {
                        decoder_with_image_info = result;
                        break;
                    }
                    ProcessingResult::NeedsMoreInput { fallback, .. } => {
                        d = fallback;
                    }
                }
            }
        }

        let basic_info = decoder_with_image_info.basic_info().clone();
        let (buffer_width, buffer_height) = basic_info.size;
        let pixel_format = decoder_with_image_info.current_pixel_format().clone();
        let num_channels = pixel_format.color_type.samples_per_pixel();

        let mut decoder_with_frame_info;
        {
            let mut d = decoder_with_image_info;
            loop {
                chunk_input = &input[..(chunk_input.len().saturating_add(1024)).min(input.len())];
                let available_before = chunk_input.len();
                let process_result = d.process(&mut chunk_input);
                input = &input[(available_before - chunk_input.len())..];
                match process_result.unwrap() {
                    ProcessingResult::Complete { result } => {
                        decoder_with_frame_info = result;
                        break;
                    }
                    ProcessingResult::NeedsMoreInput { fallback, .. } => {
                        d = fallback;
                    }
                }
            }
        }

        let mut buffer =
            Image::new_with_value((buffer_width * num_channels, buffer_height), f32::NAN).unwrap();

        // Call flush_pixels multiple times - should all succeed
        for _ in 0..5 {
            let api_buffer = JxlOutputBuffer::from_image_rect_mut(
                buffer
                    .get_rect_mut(Rect {
                        origin: (0, 0),
                        size: buffer.size(),
                    })
                    .into_raw(),
            );
            let result = decoder_with_frame_info.flush_pixels(&mut [api_buffer]);
            assert!(result.is_ok(), "flush_pixels should succeed on repeated calls");
        }
    }

    /// Test that flush_pixels works correctly with both pipeline types.
    #[test]
    fn test_flush_pixels_simple_pipeline() {
        let file = std::fs::read("resources/test/green_queen_vardct_e3.jxl").unwrap();
        let mut input = &file[..];
        let options = JxlDecoderOptions::default();
        let decoder = JxlDecoder::<states::Initialized>::new(options);

        let mut chunk_input = &input[0..0];
        let mut decoder_with_image_info;
        {
            let mut d = decoder;
            loop {
                chunk_input = &input[..(chunk_input.len().saturating_add(4096)).min(input.len())];
                let available_before = chunk_input.len();
                let process_result = d.process(&mut chunk_input);
                input = &input[(available_before - chunk_input.len())..];
                match process_result.unwrap() {
                    ProcessingResult::Complete { result } => {
                        decoder_with_image_info = result;
                        break;
                    }
                    ProcessingResult::NeedsMoreInput { fallback, .. } => {
                        d = fallback;
                    }
                }
            }
        }

        // Enable simple pipeline
        decoder_with_image_info.set_use_simple_pipeline(true);

        let basic_info = decoder_with_image_info.basic_info().clone();
        let (buffer_width, buffer_height) = basic_info.size;
        let pixel_format = decoder_with_image_info.current_pixel_format().clone();
        let num_channels = pixel_format.color_type.samples_per_pixel();

        let mut decoder_with_frame_info;
        {
            let mut d = decoder_with_image_info;
            loop {
                chunk_input = &input[..(chunk_input.len().saturating_add(4096)).min(input.len())];
                let available_before = chunk_input.len();
                let process_result = d.process(&mut chunk_input);
                input = &input[(available_before - chunk_input.len())..];
                match process_result.unwrap() {
                    ProcessingResult::Complete { result } => {
                        decoder_with_frame_info = result;
                        break;
                    }
                    ProcessingResult::NeedsMoreInput { fallback, .. } => {
                        d = fallback;
                    }
                }
            }
        }

        let mut buffer =
            Image::new_with_value((buffer_width * num_channels, buffer_height), f32::NAN).unwrap();

        let api_buffer = JxlOutputBuffer::from_image_rect_mut(
            buffer
                .get_rect_mut(Rect {
                    origin: (0, 0),
                    size: buffer.size(),
                })
                .into_raw(),
        );

        // flush_pixels should work with simple pipeline
        let result = decoder_with_frame_info.flush_pixels(&mut [api_buffer]);
        assert!(result.is_ok(), "flush_pixels should work with simple pipeline");
    }

    /// Test flush_pixels on a modular image to verify it works with different encodings.
    #[test]
    fn test_flush_pixels_modular_image() {
        // Use a modular-encoded image
        let file = std::fs::read("resources/test/green_queen_modular_e3.jxl").unwrap();
        let mut input = &file[..];
        let options = JxlDecoderOptions::default();
        let decoder = JxlDecoder::<states::Initialized>::new(options);

        let mut chunk_input = &input[0..0];
        let decoder_with_image_info;
        {
            let mut d = decoder;
            loop {
                chunk_input = &input[..(chunk_input.len().saturating_add(4096)).min(input.len())];
                let available_before = chunk_input.len();
                let process_result = d.process(&mut chunk_input);
                input = &input[(available_before - chunk_input.len())..];
                match process_result.unwrap() {
                    ProcessingResult::Complete { result } => {
                        decoder_with_image_info = result;
                        break;
                    }
                    ProcessingResult::NeedsMoreInput { fallback, .. } => {
                        d = fallback;
                    }
                }
            }
        }

        let basic_info = decoder_with_image_info.basic_info().clone();
        let (buffer_width, buffer_height) = basic_info.size;
        let pixel_format = decoder_with_image_info.current_pixel_format().clone();
        let num_channels = pixel_format.color_type.samples_per_pixel();

        let mut decoder_with_frame_info;
        {
            let mut d = decoder_with_image_info;
            loop {
                chunk_input = &input[..(chunk_input.len().saturating_add(4096)).min(input.len())];
                let available_before = chunk_input.len();
                let process_result = d.process(&mut chunk_input);
                input = &input[(available_before - chunk_input.len())..];
                match process_result.unwrap() {
                    ProcessingResult::Complete { result } => {
                        decoder_with_frame_info = result;
                        break;
                    }
                    ProcessingResult::NeedsMoreInput { fallback, .. } => {
                        d = fallback;
                    }
                }
            }
        }

        let mut buffer =
            Image::new_with_value((buffer_width * num_channels, buffer_height), f32::NAN).unwrap();

        let api_buffer = JxlOutputBuffer::from_image_rect_mut(
            buffer
                .get_rect_mut(Rect {
                    origin: (0, 0),
                    size: buffer.size(),
                })
                .into_raw(),
        );

        let result = decoder_with_frame_info.flush_pixels(&mut [api_buffer]);
        assert!(result.is_ok(), "flush_pixels should work with modular images");
    }

    /// Test that flush_pixels returns consistent results - calling it twice without
    /// new data should not change the buffer.
    #[test]
    fn test_flush_pixels_idempotent() {
        let file = std::fs::read("resources/test/green_queen_vardct_e3.jxl").unwrap();
        let mut input = &file[..];
        let options = JxlDecoderOptions::default();
        let decoder = JxlDecoder::<states::Initialized>::new(options);

        let mut chunk_input = &input[0..0];
        let decoder_with_image_info;
        {
            let mut d = decoder;
            loop {
                chunk_input = &input[..(chunk_input.len().saturating_add(4096)).min(input.len())];
                let available_before = chunk_input.len();
                let process_result = d.process(&mut chunk_input);
                input = &input[(available_before - chunk_input.len())..];
                match process_result.unwrap() {
                    ProcessingResult::Complete { result } => {
                        decoder_with_image_info = result;
                        break;
                    }
                    ProcessingResult::NeedsMoreInput { fallback, .. } => {
                        d = fallback;
                    }
                }
            }
        }

        let basic_info = decoder_with_image_info.basic_info().clone();
        let (buffer_width, buffer_height) = basic_info.size;
        let pixel_format = decoder_with_image_info.current_pixel_format().clone();
        let num_channels = pixel_format.color_type.samples_per_pixel();

        let mut decoder_with_frame_info;
        {
            let mut d = decoder_with_image_info;
            loop {
                chunk_input = &input[..(chunk_input.len().saturating_add(4096)).min(input.len())];
                let available_before = chunk_input.len();
                let process_result = d.process(&mut chunk_input);
                input = &input[(available_before - chunk_input.len())..];
                match process_result.unwrap() {
                    ProcessingResult::Complete { result } => {
                        decoder_with_frame_info = result;
                        break;
                    }
                    ProcessingResult::NeedsMoreInput { fallback, .. } => {
                        d = fallback;
                    }
                }
            }
        }

        let mut buffer1 =
            Image::new_with_value((buffer_width * num_channels, buffer_height), 0.0f32).unwrap();
        let mut buffer2 =
            Image::new_with_value((buffer_width * num_channels, buffer_height), 0.0f32).unwrap();

        // First flush
        {
            let api_buffer = JxlOutputBuffer::from_image_rect_mut(
                buffer1
                    .get_rect_mut(Rect {
                        origin: (0, 0),
                        size: buffer1.size(),
                    })
                    .into_raw(),
            );
            decoder_with_frame_info
                .flush_pixels(&mut [api_buffer])
                .unwrap();
        }

        // Second flush to different buffer
        {
            let api_buffer = JxlOutputBuffer::from_image_rect_mut(
                buffer2
                    .get_rect_mut(Rect {
                        origin: (0, 0),
                        size: buffer2.size(),
                    })
                    .into_raw(),
            );
            decoder_with_frame_info
                .flush_pixels(&mut [api_buffer])
                .unwrap();
        }

        // Both buffers should have identical content
        let (xs, ys) = buffer1.size();
        for y in 0..ys {
            for x in 0..xs {
                let v1 = buffer1.row(y)[x];
                let v2 = buffer2.row(y)[x];
                // Both should be equal (either both NaN or both same value)
                assert!(
                    (v1.is_nan() && v2.is_nan()) || (v1 - v2).abs() < 1e-10,
                    "flush_pixels should be idempotent: at ({x}, {y}): {v1} != {v2}"
                );
            }
        }
    }
}

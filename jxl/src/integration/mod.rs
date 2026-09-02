// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Integration with the [`image`] crate.
//!
//! Call [`register_image_decoding_hook`] once before using `image`'s generic
//! loading APIs. JPEG XL files can then be opened by extension or detected by
//! either of their standard signatures.

use std::io::{BufRead, BufReader, Read, Seek};

use image::error::{DecodingError, ImageFormatHint, LimitError, LimitErrorKind};
use image::hooks::{GenericReader, register_decoding_hook, register_format_detection_hook};
use image::{ColorType, ImageError, ImageResult, LimitSupport, Limits};

use crate::api::{
    CODESTREAM_SIGNATURE, CONTAINER_SIGNATURE, Endianness, JxlBitDepth, JxlColorType,
    JxlDataFormat, JxlDecoder as ApiJxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat,
    ProcessingResult, states,
};
use crate::headers::extra_channels::ExtraChannel;

/// An adapter implementing [`image::ImageDecoder`] for JPEG XL images.
pub struct JxlDecoder<R: Read + Seek> {
    input: BufReader<R>,
    decoder: ApiJxlDecoder<states::WithImageInfo>,
    width: u32,
    height: u32,
    color_type: ColorType,
    icc_profile: Option<Vec<u8>>,
    limits: Limits,
}

impl<R: Read + Seek> JxlDecoder<R> {
    /// Creates an image decoder from a readable, seekable stream.
    pub fn new(reader: R) -> ImageResult<Self> {
        let mut input = BufReader::new(reader);
        let decoder = ApiJxlDecoder::<states::Initialized>::new(JxlDecoderOptions::default());
        let mut decoder = complete_or_truncated(decoder.process(&mut input, None), &mut input)?;

        let info = decoder.basic_info().clone();
        let width = u32::try_from(info.size.0).map_err(|_| dimension_error())?;
        let height = u32::try_from(info.size.1).map_err(|_| dimension_error())?;
        let grayscale = decoder.current_pixel_format().color_type.is_grayscale();
        let has_alpha = info
            .extra_channels
            .iter()
            .any(|channel| channel.ec_type == ExtraChannel::Alpha);

        let (color_type, pixel_color_type, data_format) =
            output_format(&info.bit_depth, grayscale, has_alpha);
        decoder
            .set_pixel_format(JxlPixelFormat {
                color_type: pixel_color_type,
                color_data_format: Some(data_format),
                extra_channel_format: vec![None; info.extra_channels.len()],
            })
            .map_err(image_error)?;

        let icc_profile = decoder
            .output_color_profile()
            .try_as_icc()
            .map(|profile| profile.into_owned());

        Ok(Self {
            input,
            decoder,
            width,
            height,
            color_type,
            icc_profile,
            limits: Limits::no_limits(),
        })
    }

    fn decode_into(mut self, buf: &mut [u8]) -> ImageResult<()> {
        assert_eq!(buf.len() as u64, image::ImageDecoder::total_bytes(&self));

        let frame =
            complete_or_truncated(self.decoder.process(&mut self.input, None), &mut self.input)?;
        let bytes_per_sample = match self.color_type {
            ColorType::L8 | ColorType::La8 | ColorType::Rgb8 | ColorType::Rgba8 => 1,
            ColorType::L16 | ColorType::La16 | ColorType::Rgb16 | ColorType::Rgba16 => 2,
            ColorType::Rgb32F | ColorType::Rgba32F => 4,
            _ => unreachable!("the adapter only produces supported color types"),
        };
        let bytes_per_row = buf.len() / self.height as usize;

        if bytes_per_sample == 1
            || (buf.as_ptr().align_offset(bytes_per_sample) == 0
                && bytes_per_row.is_multiple_of(bytes_per_sample))
        {
            decode_frame(frame, &mut self.input, buf, self.height)?;
        } else if bytes_per_sample == 2 {
            self.limits.reserve_usize(buf.len())?;
            let mut aligned = vec![0u16; buf.len() / 2];
            decode_frame(
                frame,
                &mut self.input,
                bytemuck::cast_slice_mut(&mut aligned),
                self.height,
            )?;
            buf.copy_from_slice(bytemuck::cast_slice(&aligned));
        } else {
            self.limits.reserve_usize(buf.len())?;
            let mut aligned = vec![0f32; buf.len() / 4];
            decode_frame(
                frame,
                &mut self.input,
                bytemuck::cast_slice_mut(&mut aligned),
                self.height,
            )?;
            buf.copy_from_slice(bytemuck::cast_slice(&aligned));
        }

        Ok(())
    }
}

impl<R: Read + Seek> image::ImageDecoder for JxlDecoder<R> {
    fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn color_type(&self) -> ColorType {
        self.color_type
    }

    fn icc_profile(&mut self) -> ImageResult<Option<Vec<u8>>> {
        Ok(self.icc_profile.clone())
    }

    fn set_limits(&mut self, limits: Limits) -> ImageResult<()> {
        limits.check_support(&LimitSupport::default())?;
        limits.check_dimensions(self.width, self.height)?;

        let output_bytes = image::ImageDecoder::total_bytes(self);
        let scratch_bytes = if self.color_type.bytes_per_pixel() as usize
            / self.color_type.channel_count() as usize
            > 1
        {
            output_bytes
        } else {
            0
        };
        if limits
            .max_alloc
            .is_some_and(|max| output_bytes.saturating_add(scratch_bytes) > max)
        {
            return Err(ImageError::Limits(LimitError::from_kind(
                LimitErrorKind::InsufficientMemory,
            )));
        }

        self.limits = limits;
        Ok(())
    }

    fn read_image(self, buf: &mut [u8]) -> ImageResult<()> {
        self.decode_into(buf)
    }

    fn read_image_boxed(self: Box<Self>, buf: &mut [u8]) -> ImageResult<()> {
        (*self).decode_into(buf)
    }
}

fn output_format(
    bit_depth: &JxlBitDepth,
    grayscale: bool,
    alpha: bool,
) -> (ColorType, JxlColorType, JxlDataFormat) {
    match bit_depth {
        JxlBitDepth::Float { .. } => (
            if alpha {
                ColorType::Rgba32F
            } else {
                ColorType::Rgb32F
            },
            if alpha {
                JxlColorType::Rgba
            } else {
                JxlColorType::Rgb
            },
            JxlDataFormat::F32 {
                endianness: Endianness::native(),
            },
        ),
        JxlBitDepth::Int { bits_per_sample } if *bits_per_sample <= 8 => (
            match (grayscale, alpha) {
                (true, false) => ColorType::L8,
                (true, true) => ColorType::La8,
                (false, false) => ColorType::Rgb8,
                (false, true) => ColorType::Rgba8,
            },
            jxl_color_type(grayscale, alpha),
            JxlDataFormat::U8 { bit_depth: 8 },
        ),
        JxlBitDepth::Int { .. } => (
            match (grayscale, alpha) {
                (true, false) => ColorType::L16,
                (true, true) => ColorType::La16,
                (false, false) => ColorType::Rgb16,
                (false, true) => ColorType::Rgba16,
            },
            jxl_color_type(grayscale, alpha),
            JxlDataFormat::U16 {
                endianness: Endianness::native(),
                bit_depth: 16,
            },
        ),
    }
}

fn jxl_color_type(grayscale: bool, alpha: bool) -> JxlColorType {
    match (grayscale, alpha) {
        (true, false) => JxlColorType::Grayscale,
        (true, true) => JxlColorType::GrayscaleAlpha,
        (false, false) => JxlColorType::Rgb,
        (false, true) => JxlColorType::Rgba,
    }
}

fn complete_or_truncated<T, F, R: Read + Seek>(
    mut result: crate::error::Result<ProcessingResult<T, F>>,
    input: &mut BufReader<R>,
) -> ImageResult<T>
where
    F: Retry<T, R>,
{
    loop {
        match result.map_err(image_error)? {
            ProcessingResult::Complete { result } => return Ok(result),
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                if input.fill_buf()?.is_empty() {
                    return Err(truncated_error());
                }
                result = fallback.retry(input);
            }
        }
    }
}

trait Retry<T, R: Read + Seek> {
    fn retry(self, input: &mut BufReader<R>) -> crate::error::Result<ProcessingResult<T, Self>>
    where
        Self: Sized;
}

impl<R: Read + Seek> Retry<ApiJxlDecoder<states::WithImageInfo>, R>
    for ApiJxlDecoder<states::Initialized>
{
    fn retry(
        self,
        input: &mut BufReader<R>,
    ) -> crate::error::Result<ProcessingResult<ApiJxlDecoder<states::WithImageInfo>, Self>> {
        self.process(input, None)
    }
}

impl<R: Read + Seek> Retry<ApiJxlDecoder<states::WithFrameInfo>, R>
    for ApiJxlDecoder<states::WithImageInfo>
{
    fn retry(
        self,
        input: &mut BufReader<R>,
    ) -> crate::error::Result<ProcessingResult<ApiJxlDecoder<states::WithFrameInfo>, Self>> {
        self.process(input, None)
    }
}

fn decode_frame<R: Read + Seek>(
    mut decoder: ApiJxlDecoder<states::WithFrameInfo>,
    input: &mut BufReader<R>,
    buf: &mut [u8],
    height: u32,
) -> ImageResult<()> {
    let bytes_per_row = buf.len() / height as usize;
    let mut output = JxlOutputBuffer::new(buf, height as usize, bytes_per_row);
    loop {
        match decoder
            .process(input, std::slice::from_mut(&mut output), None)
            .map_err(image_error)?
        {
            ProcessingResult::Complete { .. } => return Ok(()),
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                if input.fill_buf()?.is_empty() {
                    return Err(truncated_error());
                }
                decoder = fallback;
            }
        }
    }
}

fn image_error(error: crate::error::Error) -> ImageError {
    ImageError::Decoding(DecodingError::new(jxl_format_hint(), error))
}

fn truncated_error() -> ImageError {
    ImageError::Decoding(DecodingError::new(
        jxl_format_hint(),
        "truncated JPEG XL image",
    ))
}

fn dimension_error() -> ImageError {
    ImageError::Limits(LimitError::from_kind(LimitErrorKind::DimensionError))
}

fn jxl_format_hint() -> ImageFormatHint {
    ImageFormatHint::Name("JPEG XL".to_owned())
}

/// Registers JPEG XL extension-based decoding and signature detection with [`image`].
///
/// Call this once before using APIs such as [`image::open`] or
/// [`image::load_from_memory`]. The registration applies process-wide and recognizes
/// both bare JPEG XL codestreams and JPEG XL containers.
///
/// Returns `true` when the decoder was registered, or `false` if another decoder
/// was already registered for the `jxl` extension.
pub fn register_image_decoding_hook() -> bool {
    let registered = register_decoding_hook(
        "jxl".into(),
        Box::new(|reader: GenericReader<'_>| Ok(Box::new(JxlDecoder::new(reader)?))),
    );
    if registered {
        register_format_detection_hook("jxl".into(), &CODESTREAM_SIGNATURE, None);
        register_format_detection_hook("jxl".into(), &CONTAINER_SIGNATURE, None);
    }
    registered
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use image::ImageDecoder as _;

    use super::*;

    const RGB: &[u8] = include_bytes!("../../resources/test/3x3_srgb_lossless.jxl");
    const RGBA: &[u8] = include_bytes!("../../resources/test/3x3a_srgb_lossless.jxl");
    const RGB16: &[u8] = include_bytes!("../../resources/test/image_integration_rgb16.jxl");

    #[test]
    fn decodes_with_image_adapter() {
        let decoder = JxlDecoder::new(Cursor::new(RGB)).unwrap();
        assert_eq!(decoder.dimensions(), (3, 3));
        assert_eq!(decoder.color_type(), ColorType::Rgb8);

        let mut pixels = vec![0; decoder.total_bytes() as usize];
        decoder.read_image(&mut pixels).unwrap();
        assert!(pixels.iter().any(|&sample| sample != 0));
    }

    #[test]
    fn preserves_alpha() {
        let decoder = JxlDecoder::new(Cursor::new(RGBA)).unwrap();
        assert_eq!(decoder.color_type(), ColorType::Rgba8);
    }

    #[test]
    fn decodes_16_bit_to_unaligned_buffer() {
        let decoder = JxlDecoder::new(Cursor::new(RGB16)).unwrap();
        assert_eq!(decoder.color_type(), ColorType::Rgb16);

        let mut storage = vec![0; decoder.total_bytes() as usize + 1];
        decoder.read_image(&mut storage[1..]).unwrap();
        let expected: Vec<u8> = [u16::MAX, 0, 0, 32768, u16::MAX, 0]
            .into_iter()
            .flat_map(u16::to_ne_bytes)
            .collect();
        assert_eq!(&storage[1..], expected);
    }

    #[test]
    fn registers_image_hooks() {
        assert!(register_image_decoding_hook());
        assert!(!register_image_decoding_hook());

        let image = image::load_from_memory(RGB).unwrap();
        assert_eq!((image.width(), image.height()), (3, 3));

        let path = format!(
            "{}/resources/test/3x3_srgb_lossless.jxl",
            env!("CARGO_MANIFEST_DIR")
        );
        let image = image::open(path).unwrap();
        assert_eq!((image.width(), image.height()), (3, 3));
    }
}

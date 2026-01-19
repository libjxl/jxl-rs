// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::time::{Duration, Instant};

use color_eyre::eyre::{Result, eyre};
use jxl::{
    api::{
        Endianness, JxlAnimation, JxlBitDepth, JxlBitstreamInput, JxlColorProfile, JxlColorType,
        JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat,
        ProcessingResult, states::WithImageInfo,
    },
    image::{Image, ImageDataType, Rect},
    util::f16,
};

pub struct ImageFrame<T: ImageDataType> {
    pub channels: Vec<Image<T>>,
    pub duration: f64,
    pub color_type: JxlColorType,
}

pub struct DecodeOutput<T: ImageDataType> {
    pub size: (usize, usize),
    pub frames: Vec<ImageFrame<T>>,
    pub original_bit_depth: JxlBitDepth,
    pub output_profile: JxlColorProfile,
    pub embedded_profile: JxlColorProfile,
    pub jxl_animation: Option<JxlAnimation>,
}

pub fn decode_header<In: JxlBitstreamInput>(
    input: &mut In,
    decoder_options: JxlDecoderOptions,
) -> Result<JxlDecoder<WithImageInfo>> {
    let initialized_decoder = JxlDecoder::<jxl::api::states::Initialized>::new(decoder_options);

    match initialized_decoder.process(input)? {
        ProcessingResult::Complete { result } => Ok(result),
        ProcessingResult::NeedsMoreInput { .. } => Err(eyre!("Source file truncated")),
    }
}

/// Output data type for decoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputDataType {
    U8,
    U16,
    F16,
    F32,
}

impl OutputDataType {
    /// Parse from string (case-insensitive).
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "u8" => Some(Self::U8),
            "u16" => Some(Self::U16),
            "f16" => Some(Self::F16),
            "f32" => Some(Self::F32),
            _ => None,
        }
    }

    /// Get the JxlDataFormat for this type.
    pub fn to_data_format(self) -> JxlDataFormat {
        match self {
            Self::U8 => JxlDataFormat::U8 { bit_depth: 8 },
            Self::U16 => JxlDataFormat::U16 {
                endianness: Endianness::native(),
                bit_depth: 16,
            },
            Self::F16 => JxlDataFormat::F16 {
                endianness: Endianness::native(),
            },
            Self::F32 => JxlDataFormat::f32(),
        }
    }
}

/// Typed decode output that preserves the original output type.
/// The caller is responsible for converting to f32 when needed for saving.
pub enum TypedDecodeOutput {
    U8(DecodeOutput<u8>),
    U16(DecodeOutput<u16>),
    F16(DecodeOutput<f16>),
    F32(DecodeOutput<f32>),
}

impl TypedDecodeOutput {
    /// Get the image size.
    pub fn size(&self) -> (usize, usize) {
        match self {
            Self::U8(d) => d.size,
            Self::U16(d) => d.size,
            Self::F16(d) => d.size,
            Self::F32(d) => d.size,
        }
    }

    /// Get the output color profile.
    pub fn output_profile(&self) -> &JxlColorProfile {
        match self {
            Self::U8(d) => &d.output_profile,
            Self::U16(d) => &d.output_profile,
            Self::F16(d) => &d.output_profile,
            Self::F32(d) => &d.output_profile,
        }
    }

    /// Get the embedded color profile.
    pub fn embedded_profile(&self) -> &JxlColorProfile {
        match self {
            Self::U8(d) => &d.embedded_profile,
            Self::U16(d) => &d.embedded_profile,
            Self::F16(d) => &d.embedded_profile,
            Self::F32(d) => &d.embedded_profile,
        }
    }

    /// Get the original bit depth.
    pub fn original_bit_depth(&self) -> &JxlBitDepth {
        match self {
            Self::U8(d) => &d.original_bit_depth,
            Self::U16(d) => &d.original_bit_depth,
            Self::F16(d) => &d.original_bit_depth,
            Self::F32(d) => &d.original_bit_depth,
        }
    }

    /// Convert to f32 output for saving to encoders.
    pub fn to_f32(self) -> Result<DecodeOutput<f32>> {
        match self {
            Self::U8(d) => convert_decode_output_to_f32(d),
            Self::U16(d) => convert_decode_output_to_f32(d),
            Self::F16(d) => convert_decode_output_to_f32(d),
            Self::F32(d) => Ok(d),
        }
    }

    /// Truncate to keep only first N frames.
    pub fn truncate_frames(&mut self, len: usize) {
        match self {
            Self::U8(d) => d.frames.truncate(len),
            Self::U16(d) => d.frames.truncate(len),
            Self::F16(d) => d.frames.truncate(len),
            Self::F32(d) => d.frames.truncate(len),
        }
    }

    /// Get the first frame's color type and channel size (for preview handling).
    pub fn first_frame_info(&self) -> Option<(JxlColorType, (usize, usize))> {
        match self {
            Self::U8(d) => d
                .frames
                .first()
                .map(|f| (f.color_type, f.channels[0].size())),
            Self::U16(d) => d
                .frames
                .first()
                .map(|f| (f.color_type, f.channels[0].size())),
            Self::F16(d) => d
                .frames
                .first()
                .map(|f| (f.color_type, f.channels[0].size())),
            Self::F32(d) => d
                .frames
                .first()
                .map(|f| (f.color_type, f.channels[0].size())),
        }
    }

    /// Update the size field.
    pub fn set_size(&mut self, size: (usize, usize)) {
        match self {
            Self::U8(d) => d.size = size,
            Self::U16(d) => d.size = size,
            Self::F16(d) => d.size = size,
            Self::F32(d) => d.size = size,
        }
    }
}

/// Decode a JXL image with a specific output data type.
/// Returns the raw typed output without conversion, so benchmark timing is accurate.
pub fn decode_frames_with_type<In: JxlBitstreamInput>(
    input: &mut In,
    decoder_options: JxlDecoderOptions,
    output_type: OutputDataType,
    linear_output: bool,
) -> Result<(TypedDecodeOutput, Duration)> {
    match output_type {
        OutputDataType::U8 => {
            let (output, duration) =
                decode_frames_typed::<u8, _>(input, decoder_options, output_type, linear_output)?;
            Ok((TypedDecodeOutput::U8(output), duration))
        }
        OutputDataType::U16 => {
            let (output, duration) =
                decode_frames_typed::<u16, _>(input, decoder_options, output_type, linear_output)?;
            Ok((TypedDecodeOutput::U16(output), duration))
        }
        OutputDataType::F16 => {
            let (output, duration) =
                decode_frames_typed::<f16, _>(input, decoder_options, output_type, linear_output)?;
            Ok((TypedDecodeOutput::F16(output), duration))
        }
        OutputDataType::F32 => {
            let (output, duration) =
                decode_frames_typed::<f32, _>(input, decoder_options, output_type, linear_output)?;
            Ok((TypedDecodeOutput::F32(output), duration))
        }
    }
}

/// Generic decoder that decodes to type T.
fn decode_frames_typed<T: ImageDataType, In: JxlBitstreamInput>(
    input: &mut In,
    decoder_options: JxlDecoderOptions,
    output_type: OutputDataType,
    linear_output: bool,
) -> Result<(DecodeOutput<T>, Duration)> {
    let start = Instant::now();

    let mut decoder_with_image_info = decode_header(input, decoder_options)?;

    // Get info and clone what we need before mutating the decoder
    let info = decoder_with_image_info.basic_info().clone();
    let embedded_profile = decoder_with_image_info.embedded_color_profile().clone();

    // If linear output is requested, modify the output profile
    if linear_output
        && let JxlColorProfile::Simple(enc) = decoder_with_image_info.output_color_profile().clone()
    {
        decoder_with_image_info
            .set_output_color_profile(JxlColorProfile::Simple(enc.with_linear_tf()))?;
    }
    let output_profile = decoder_with_image_info.output_color_profile().clone();

    // Set the pixel format to the requested data type
    let current_format = decoder_with_image_info.current_pixel_format().clone();
    let new_format = JxlPixelFormat {
        color_type: current_format.color_type,
        color_data_format: Some(output_type.to_data_format()),
        extra_channel_format: current_format
            .extra_channel_format
            .iter()
            .map(|f| f.as_ref().map(|_| output_type.to_data_format()))
            .collect(),
    };
    decoder_with_image_info.set_pixel_format(new_format);

    let mut image_data = DecodeOutput {
        size: info.size,
        frames: Vec::new(),
        original_bit_depth: info.bit_depth.clone(),
        output_profile,
        embedded_profile,
        jxl_animation: info.animation.clone(),
    };

    let extra_channels = info.extra_channels.len();
    let pixel_format = decoder_with_image_info.current_pixel_format().clone();
    let color_type = pixel_format.color_type;
    let samples_per_pixel = if color_type == JxlColorType::Grayscale {
        1
    } else {
        3
    };

    loop {
        let decoder_with_frame_info = match decoder_with_image_info.process(input)? {
            ProcessingResult::Complete { result } => result,
            ProcessingResult::NeedsMoreInput { .. } => return Err(eyre!("Source file truncated")),
        };

        let frame_header = decoder_with_frame_info.frame_header();
        let frame_size = frame_header.size;

        // Create typed output buffers
        let mut typed_outputs = vec![Image::<T>::new((
            frame_size.0 * samples_per_pixel,
            frame_size.1,
        ))?];

        for _ in 0..extra_channels {
            typed_outputs.push(Image::<T>::new(frame_size)?);
        }

        let mut output_bufs: Vec<JxlOutputBuffer<'_>> = typed_outputs
            .iter_mut()
            .map(|x| {
                let rect = Rect {
                    size: x.size(),
                    origin: (0, 0),
                };
                JxlOutputBuffer::from_image_rect_mut(x.get_rect_mut(rect).into_raw())
            })
            .collect();

        decoder_with_image_info = match decoder_with_frame_info.process(input, &mut output_bufs)? {
            ProcessingResult::Complete { result } => result,
            ProcessingResult::NeedsMoreInput { .. } => return Err(eyre!("Source file truncated")),
        };

        image_data.frames.push(ImageFrame {
            duration: frame_header.duration.unwrap_or(0.0),
            channels: typed_outputs,
            color_type,
        });

        if !decoder_with_image_info.has_more_frames() {
            break;
        }
    }

    Ok((image_data, start.elapsed()))
}

/// Trait for converting a value to f32.
trait ConvertToF32: Copy {
    fn to_f32_normalized(self) -> f32;
}

impl ConvertToF32 for u8 {
    fn to_f32_normalized(self) -> f32 {
        self as f32 / 255.0
    }
}

impl ConvertToF32 for u16 {
    fn to_f32_normalized(self) -> f32 {
        self as f32 / 65535.0
    }
}

impl ConvertToF32 for f16 {
    fn to_f32_normalized(self) -> f32 {
        self.to_f32()
    }
}

/// Convert a DecodeOutput from type T to f32.
fn convert_decode_output_to_f32<T: ImageDataType + ConvertToF32>(
    src: DecodeOutput<T>,
) -> Result<DecodeOutput<f32>> {
    let mut frames = Vec::with_capacity(src.frames.len());
    for frame in src.frames {
        let channels: Vec<Image<f32>> = frame
            .channels
            .into_iter()
            .map(|img| convert_image_to_f32(img))
            .collect::<std::result::Result<_, _>>()?;
        frames.push(ImageFrame {
            channels,
            duration: frame.duration,
            color_type: frame.color_type,
        });
    }
    Ok(DecodeOutput {
        size: src.size,
        frames,
        original_bit_depth: src.original_bit_depth,
        output_profile: src.output_profile,
        embedded_profile: src.embedded_profile,
        jxl_animation: src.jxl_animation,
    })
}

/// Convert an image from type T to f32.
fn convert_image_to_f32<T: ImageDataType + ConvertToF32>(
    src: Image<T>,
) -> std::result::Result<Image<f32>, jxl::error::Error> {
    let size = src.size();
    let mut dst = Image::<f32>::new(size)?;

    for y in 0..size.1 {
        let src_row = src.row(y);
        let dst_row = dst.row_mut(y);
        for x in 0..size.0 {
            dst_row[x] = src_row[x].to_f32_normalized();
        }
    }

    Ok(dst)
}

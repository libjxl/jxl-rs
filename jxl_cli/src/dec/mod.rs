// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::{
    str::FromStr,
    time::{Duration, Instant},
};

use color_eyre::eyre::{Result, eyre};
use jxl::{
    api::{
        Endianness, JxlAnimation, JxlBitDepth, JxlBitstreamInput, JxlColorProfile, JxlColorType,
        JxlDataFormat, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat,
        ProcessingResult, states::WithImageInfo,
    },
    headers::extra_channels::ExtraChannel,
    image::{OwnedRawImage, Rect},
};

pub struct ImageFrame {
    pub channels: Vec<OwnedRawImage>,
    pub duration: f64,
    pub color_type: JxlColorType,
}

pub struct DecodeOutput {
    pub size: (usize, usize),
    pub frames: Vec<ImageFrame>,
    pub data_type: OutputDataType,
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

impl FromStr for OutputDataType {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "u8" => Ok(Self::U8),
            "u16" => Ok(Self::U16),
            "f16" => Ok(Self::F16),
            "f32" => Ok(Self::F32),
            _ => Err(format!("Unknown data type {s}")),
        }
    }
}

impl OutputDataType {
    pub const ALL: &'static [OutputDataType] = &[
        OutputDataType::U8,
        OutputDataType::U16,
        OutputDataType::F16,
        OutputDataType::F32,
    ];

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

    pub fn bits_per_sample(&self) -> usize {
        self.to_data_format().bytes_per_sample() * 8
    }
}

#[allow(clippy::too_many_arguments)]
pub fn decode_frames<In: JxlBitstreamInput>(
    input: &mut In,
    decoder_options: JxlDecoderOptions,
    requested_bit_depth: Option<usize>,
    requested_output_type: Option<OutputDataType>,
    accepted_output_types: &[OutputDataType],
    interleave_alpha: bool,
    linear_output: bool,
    allow_partial_files: bool,
) -> Result<(DecodeOutput, Duration)> {
    let start = Instant::now();

    let mut decoder_with_image_info = decode_header(input, decoder_options)?;

    // Get info and clone what we need before mutating the decoder
    let info = decoder_with_image_info.basic_info().clone();
    let embedded_profile = decoder_with_image_info.embedded_color_profile().clone();

    let output_type = if let Some(ot) = requested_output_type
        && accepted_output_types.contains(&ot)
    {
        ot
    } else {
        if requested_output_type.is_some() {
            eprintln!("Warning: requested output type is not compatible with output format");
        }
        let bit_depth = requested_bit_depth.unwrap_or(info.bit_depth.bits_per_sample() as usize);
        *accepted_output_types
            .iter()
            .find(|x| x.bits_per_sample() >= bit_depth)
            .unwrap_or(accepted_output_types.last().unwrap())
    };

    let main_alpha_channel = info
        .extra_channels
        .iter()
        .enumerate()
        .find(|x| x.1.ec_type == ExtraChannel::Alpha)
        .map(|x| x.0);

    let interleave_alpha = interleave_alpha && main_alpha_channel.is_some();

    // Set the pixel format to the requested data type
    let current_format = decoder_with_image_info.current_pixel_format().clone();
    let new_format = JxlPixelFormat {
        color_type: if interleave_alpha {
            current_format.color_type.add_alpha()
        } else {
            current_format.color_type
        },
        color_data_format: Some(output_type.to_data_format()),
        extra_channel_format: current_format
            .extra_channel_format
            .iter()
            .enumerate()
            .map(|(c, f)| {
                if interleave_alpha && Some(c) == main_alpha_channel {
                    None
                } else {
                    f.as_ref().map(|_| output_type.to_data_format())
                }
            })
            .collect(),
    };
    decoder_with_image_info.set_pixel_format(new_format);

    // If linear output is requested, modify the output profile
    if linear_output
        && let JxlColorProfile::Simple(enc) = decoder_with_image_info.output_color_profile().clone()
    {
        decoder_with_image_info
            .set_output_color_profile(JxlColorProfile::Simple(enc.with_linear_tf()))?;
    }
    let output_profile = decoder_with_image_info.output_color_profile().clone();

    let mut image_data = DecodeOutput {
        size: info.size,
        frames: Vec::new(),
        data_type: output_type,
        original_bit_depth: info.bit_depth.clone(),
        output_profile,
        embedded_profile,
        jxl_animation: info.animation.clone(),
    };

    let extra_channels = info.extra_channels.len() - if interleave_alpha { 1 } else { 0 };
    let pixel_format = decoder_with_image_info.current_pixel_format().clone();
    let color_type = pixel_format.color_type;
    let samples_per_pixel = pixel_format.color_type.samples_per_pixel();

    loop {
        let decoder_with_frame_info = match decoder_with_image_info.process(input)? {
            ProcessingResult::Complete { result } => result,
            ProcessingResult::NeedsMoreInput { .. } => {
                if allow_partial_files && !image_data.frames.is_empty() {
                    break;
                }
                return Err(eyre!("Source file truncated"));
            }
        };

        let frame_header = decoder_with_frame_info.frame_header();
        let frame_size = frame_header.size;
        let frame_size = (
            frame_size.0 * output_type.bits_per_sample() / 8,
            frame_size.1,
        );

        // Create typed output buffers
        let mut outputs = vec![OwnedRawImage::new((
            frame_size.0 * samples_per_pixel,
            frame_size.1,
        ))?];

        for _ in 0..extra_channels {
            outputs.push(OwnedRawImage::new(frame_size)?);
        }

        let mut output_bufs: Vec<JxlOutputBuffer<'_>> = outputs
            .iter_mut()
            .map(|x| {
                let rect = Rect {
                    size: x.byte_size(),
                    origin: (0, 0),
                };
                JxlOutputBuffer::from_image_rect_mut(x.get_rect_mut(rect))
            })
            .collect();

        decoder_with_image_info = match decoder_with_frame_info.process(input, &mut output_bufs)? {
            ProcessingResult::Complete { result } => result,
            ProcessingResult::NeedsMoreInput { mut fallback, .. } => {
                if allow_partial_files {
                    fallback.flush_pixels(&mut output_bufs)?;
                    image_data.frames.push(ImageFrame {
                        duration: frame_header.duration.unwrap_or(0.0),
                        channels: outputs,
                        color_type,
                    });
                    break;
                }
                return Err(eyre!("Source file truncated"));
            }
        };

        image_data.frames.push(ImageFrame {
            duration: frame_header.duration.unwrap_or(0.0),
            channels: outputs,
            color_type,
        });

        if !decoder_with_image_info.has_more_frames() {
            break;
        }
    }

    Ok((image_data, start.elapsed()))
}

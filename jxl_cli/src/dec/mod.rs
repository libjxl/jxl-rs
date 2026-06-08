// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::{
    io::BufReader,
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
    pub partial_renders: Vec<Vec<OwnedRawImage>>,
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

pub trait JxlBitstreamInputExt: JxlBitstreamInput {
    fn with_capped_size<T, F: FnOnce(&mut Self) -> T>(&mut self, size: Option<usize>, f: F) -> T;
}

impl JxlBitstreamInputExt for &[u8] {
    fn with_capped_size<T, F: FnOnce(&mut Self) -> T>(&mut self, size: Option<usize>, f: F) -> T {
        let size = size.unwrap_or(0);
        if size == 0 {
            return f(self);
        }
        let mut slice = &self[..size.min(self.len())];
        let cur = slice.len();
        let r = f(&mut slice);
        *self = &self[cur - slice.len()..];
        r
    }
}

impl<R> JxlBitstreamInputExt for BufReader<R>
where
    BufReader<R>: JxlBitstreamInput,
{
    // noop implementation
    fn with_capped_size<T, F: FnOnce(&mut Self) -> T>(&mut self, _size: Option<usize>, f: F) -> T {
        f(self)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn decode_frames<In: JxlBitstreamInputExt>(
    input: &mut In,
    decoder_options: JxlDecoderOptions,
    requested_bit_depth: Option<usize>,
    requested_output_type: Option<OutputDataType>,
    accepted_output_types: &[OutputDataType],
    interleave_alpha: bool,
    linear_output: bool,
    render_interval: Option<usize>,
    allow_partial_files: bool,
    store_partial_renders: bool,
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

    // If linear output is requested, initialize the CMS transformer
    let mut output_profile = decoder_with_image_info.output_color_profile().clone();
    let mut cms_transformer = None;
    if linear_output && let JxlColorProfile::Simple(ref enc) = output_profile {
        let linear_enc = enc.with_linear_tf();
        let target_profile = JxlColorProfile::Simple(linear_enc.clone());

        let cms = jxl_cms::lcms2::Lcms2Cms;
        use jxl_cms::JxlCms;
        let (_out_chans, mut transformers) = cms
            .initialize_transforms(
                1,
                info.size.0,
                output_profile.clone(),
                target_profile.clone(),
                255.0,
            )
            .map_err(|e| eyre!("CMS initialization failed: {}", e))?;

        cms_transformer = Some(transformers.remove(0));
        output_profile = target_profile;
    }

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
    let color_channels = if interleave_alpha {
        samples_per_pixel - 1
    } else {
        samples_per_pixel
    };

    'frame: loop {
        let image_size = info.size;
        let byte_size = (
            image_size.0 * output_type.bits_per_sample() / 8,
            image_size.1,
        );

        let mut outputs = vec![OwnedRawImage::new((
            byte_size.0 * samples_per_pixel,
            byte_size.1,
        ))?];

        for _ in 0..extra_channels {
            outputs.push(OwnedRawImage::new(byte_size)?);
        }

        let mut partial_renders: Vec<Vec<OwnedRawImage>> = vec![];

        let mut has_rendered_data = false;

        let mut decoder_with_frame_info = 'partial: loop {
            match input
                .with_capped_size(render_interval, |inp| decoder_with_image_info.process(inp))?
            {
                ProcessingResult::Complete { result } => {
                    break 'partial result;
                }
                ProcessingResult::NeedsMoreInput { mut fallback, .. } => {
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

                    // If we have more data but we're feeding it slowly, save the partial
                    // render and retry.
                    if render_interval.is_some() && input.available_bytes()? > 0 {
                        has_rendered_data |= fallback.flush_pixels(&mut output_bufs)?;
                        if has_rendered_data && store_partial_renders {
                            partial_renders.push(
                                outputs
                                    .iter()
                                    .map(|x| x.try_clone())
                                    .collect::<Result<_, _>>()?,
                            );
                        }
                        decoder_with_image_info = fallback;
                        continue 'partial;
                    } else if allow_partial_files {
                        fallback.flush_pixels(&mut output_bufs)?;
                        image_data.frames.push(ImageFrame {
                            partial_renders,
                            duration: 0.0,
                            channels: outputs,
                            color_type,
                        });
                        break 'frame;
                    }
                    return Err(eyre!("Source file truncated"));
                }
            }
        };

        let frame_header = decoder_with_frame_info.frame_header();

        decoder_with_image_info = 'partial: loop {
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

            match input.with_capped_size(render_interval, |inp| {
                decoder_with_frame_info.process(inp, &mut output_bufs)
            })? {
                ProcessingResult::Complete { result } => {
                    break 'partial result;
                }
                ProcessingResult::NeedsMoreInput { mut fallback, .. } => {
                    // If we have more data but we're feeding it slowly, save the partial
                    // render and retry.
                    if render_interval.is_some() && input.available_bytes()? > 0 {
                        has_rendered_data |= fallback.flush_pixels(&mut output_bufs)?;
                        if has_rendered_data && store_partial_renders {
                            partial_renders.push(
                                outputs
                                    .iter()
                                    .map(|x| x.try_clone())
                                    .collect::<Result<_, _>>()?,
                            );
                        }
                        decoder_with_frame_info = fallback;
                        continue 'partial;
                    } else if allow_partial_files {
                        fallback.flush_pixels(&mut output_bufs)?;
                        image_data.frames.push(ImageFrame {
                            partial_renders,
                            duration: frame_header.duration.unwrap_or(0.0),
                            channels: outputs,
                            color_type,
                        });
                        break 'frame;
                    }
                    return Err(eyre!("Source file truncated"));
                }
            };
        };

        image_data.frames.push(ImageFrame {
            partial_renders,
            duration: frame_header.duration.unwrap_or(0.0),
            channels: outputs,
            color_type,
        });

        if !decoder_with_image_info.has_more_frames() {
            break;
        }
    }

    if let Some(ref mut transformer) = cms_transformer {
        for frame in &mut image_data.frames {
            apply_cms(
                &mut frame.channels[0],
                samples_per_pixel,
                color_channels,
                output_type,
                transformer.as_mut(),
                info.size.0,
                info.size.1,
            )?;
            for partial in &mut frame.partial_renders {
                apply_cms(
                    &mut partial[0],
                    samples_per_pixel,
                    color_channels,
                    output_type,
                    transformer.as_mut(),
                    info.size.0,
                    info.size.1,
                )?;
            }
        }
    }

    Ok((image_data, start.elapsed()))
}

fn apply_cms(
    image: &mut OwnedRawImage,
    samples_per_pixel: usize,
    color_channels: usize,
    output_type: OutputDataType,
    transformer: &mut dyn jxl_cms::JxlCmsTransformer,
    width: usize,
    height: usize,
) -> Result<()> {
    let mut row_color_buffer = vec![0.0f32; width * color_channels];
    let mut row_output_buffer = vec![0.0f32; width * color_channels];
    for y in 0..height {
        let row_bytes = image.row(y);
        // 1. Extract and convert to f32
        match output_type {
            OutputDataType::U8 => {
                for x in 0..width {
                    for c in 0..color_channels {
                        let idx = x * samples_per_pixel + c;
                        row_color_buffer[x * color_channels + c] = row_bytes[idx] as f32 / 255.0;
                    }
                }
            }
            OutputDataType::U16 => {
                for x in 0..width {
                    for c in 0..color_channels {
                        let idx = (x * samples_per_pixel + c) * 2;
                        let val = u16::from_ne_bytes([row_bytes[idx], row_bytes[idx + 1]]);
                        row_color_buffer[x * color_channels + c] = val as f32 / 65535.0;
                    }
                }
            }
            OutputDataType::F16 => {
                for x in 0..width {
                    for c in 0..color_channels {
                        let idx = (x * samples_per_pixel + c) * 2;
                        let val = u16::from_ne_bytes([row_bytes[idx], row_bytes[idx + 1]]);
                        row_color_buffer[x * color_channels + c] =
                            jxl::util::f16::from_bits(val).to_f32();
                    }
                }
            }
            OutputDataType::F32 => {
                for x in 0..width {
                    for c in 0..color_channels {
                        let idx = (x * samples_per_pixel + c) * 4;
                        let val = f32::from_ne_bytes([
                            row_bytes[idx],
                            row_bytes[idx + 1],
                            row_bytes[idx + 2],
                            row_bytes[idx + 3],
                        ]);
                        row_color_buffer[x * color_channels + c] = val;
                    }
                }
            }
        }

        // 2. Perform CMS transform
        transformer
            .do_transform(&row_color_buffer, &mut row_output_buffer)
            .map_err(|e| eyre!("CMS transform failed: {}", e))?;

        // 3. Convert back and write to row
        let row_bytes_mut = image.row_mut(y);
        match output_type {
            OutputDataType::U8 => {
                for x in 0..width {
                    for c in 0..color_channels {
                        let idx = x * samples_per_pixel + c;
                        let val = row_output_buffer[x * color_channels + c];
                        row_bytes_mut[idx] = (val * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                    }
                }
            }
            OutputDataType::U16 => {
                for x in 0..width {
                    for c in 0..color_channels {
                        let idx = (x * samples_per_pixel + c) * 2;
                        let val = row_output_buffer[x * color_channels + c];
                        let val_u16 = (val * 65535.0 + 0.5).clamp(0.0, 65535.0) as u16;
                        let u16_bytes = val_u16.to_ne_bytes();
                        row_bytes_mut[idx] = u16_bytes[0];
                        row_bytes_mut[idx + 1] = u16_bytes[1];
                    }
                }
            }
            OutputDataType::F16 => {
                for x in 0..width {
                    for c in 0..color_channels {
                        let idx = (x * samples_per_pixel + c) * 2;
                        let val = row_output_buffer[x * color_channels + c];
                        let val_f16 = jxl::util::f16::from_f32(val);
                        let u16_bytes = val_f16.to_bits().to_ne_bytes();
                        row_bytes_mut[idx] = u16_bytes[0];
                        row_bytes_mut[idx + 1] = u16_bytes[1];
                    }
                }
            }
            OutputDataType::F32 => {
                for x in 0..width {
                    for c in 0..color_channels {
                        let idx = (x * samples_per_pixel + c) * 4;
                        let val = row_output_buffer[x * color_channels + c];
                        let f32_bytes = val.to_ne_bytes();
                        row_bytes_mut[idx] = f32_bytes[0];
                        row_bytes_mut[idx + 1] = f32_bytes[1];
                        row_bytes_mut[idx + 2] = f32_bytes[2];
                        row_bytes_mut[idx + 3] = f32_bytes[3];
                    }
                }
            }
        }
    }
    Ok(())
}

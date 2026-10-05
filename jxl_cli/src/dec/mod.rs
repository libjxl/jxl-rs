// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::io::BufReader;
use std::str::FromStr;
use std::time::{Duration, Instant};

use color_eyre::eyre::{Result, eyre};
use jxl::api::{
    Endianness, Event, ExtraChannel, JxlAnimation, JxlBitDepth, JxlBitstreamInput,
    JxlColorEncoding, JxlColorProfile, JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderOptions,
    JxlOutputBuffer, JxlParallelRunner, JxlParallelRunnerFun, JxlPixelFormat,
};
use jxl::image::{OwnedRawImage, Rect, f16};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub struct PartialRender {
    pub byte_index: usize,
    pub channels: Vec<OwnedRawImage>,
}

pub struct ImageFrame {
    pub partial_renders: Vec<PartialRender>,
    pub channels: Vec<OwnedRawImage>,
    pub duration: f64,
    pub color_type: JxlColorType,
    pub total_bytes: usize,
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

struct RayonParallelRunner;

impl JxlParallelRunner for RayonParallelRunner {
    fn run(&mut self, num: usize, fun: &JxlParallelRunnerFun) -> jxl::error::Result<()> {
        (0..num).into_par_iter().try_for_each(fun)
    }

    fn num_threads(&self) -> usize {
        rayon::current_num_threads()
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
    accepts_cmyk: bool,
    interleave_alpha: bool,
    linear_output: bool,
    render_interval: Option<usize>,
    allow_partial_files: bool,
) -> Result<(DecodeOutput, Duration)> {
    let start = Instant::now();
    let total_bytes = input.available_bytes()?;

    let mut decoder = JxlDecoder::new(decoder_options);
    while input.with_capped_size(render_interval, |inp| decoder.process(inp, None, None))?
        != Event::BasicInfo
    {
        if input.available_bytes()? == 0 {
            return Err(eyre!("Source file truncated"));
        }
    }

    // Get info and clone what we need before mutating the decoder
    let info = decoder.basic_info().unwrap().clone();
    let embedded_profile = decoder.embedded_color_profile().unwrap().clone();

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
    let current_format = decoder.current_pixel_format().unwrap().clone();
    let color_type = if interleave_alpha {
        current_format
            .color_type
            .add_alpha()
            .ok_or_else(|| eyre!("Output color type does not support interleaved alpha"))?
    } else {
        current_format.color_type
    };
    let new_format = JxlPixelFormat {
        color_type,
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
    decoder.set_pixel_format(new_format)?;

    // If linear output is requested, or CMYK output is not supported, initialize the CMS transformer
    let mut output_profile = decoder.output_color_profile().unwrap().clone();
    let mut cms_transformer = None;
    let target_enc = if linear_output && let JxlColorProfile::Simple(ref enc) = output_profile {
        Some(enc.with_linear_tf())
    } else if !accepts_cmyk && output_profile.is_cmyk() {
        if linear_output {
            Some(JxlColorEncoding::linear_srgb(false))
        } else {
            Some(JxlColorEncoding::srgb(false))
        }
    } else {
        None
    };
    if let Some(enc) = target_enc {
        let target_profile = JxlColorProfile::Simple(enc);

        let cms = jxl_cms::moxcms::MoxCms;
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
    let samples_per_pixel = color_type.samples_per_pixel();
    let color_channels = if interleave_alpha {
        samples_per_pixel - 1
    } else {
        samples_per_pixel
    };

    let make_outputs = || -> Result<Vec<OwnedRawImage>> {
        let byte_size = (info.size.0 * output_type.bits_per_sample() / 8, info.size.1);
        let mut outputs = vec![OwnedRawImage::new((
            byte_size.0 * samples_per_pixel,
            byte_size.1,
        ))?];
        for _ in 0..extra_channels {
            outputs.push(OwnedRawImage::new(byte_size)?);
        }
        Ok(outputs)
    };

    let mut outputs = Some(make_outputs()?);
    let mut partial_renders: Vec<PartialRender> = vec![];
    let mut frame_duration = 0.0;

    loop {
        let mut output_bufs: Option<Vec<JxlOutputBuffer<'_>>> = outputs.as_mut().map(|outs| {
            outs.iter_mut()
                .map(|x| {
                    let rect = Rect {
                        size: x.byte_size(),
                        origin: (0, 0),
                    };
                    JxlOutputBuffer::from_image_rect_mut(x.get_rect_mut(rect))
                })
                .collect()
        });

        match input.with_capped_size(render_interval, |inp| {
            decoder.process(
                inp,
                output_bufs.as_deref_mut(),
                Some(&mut RayonParallelRunner),
            )
        })? {
            Event::BasicInfo => unreachable!(),
            Event::FrameHeader => {
                frame_duration = decoder.frame_header().unwrap().duration.unwrap_or(0.0);
            }
            Event::FrameComplete { has_more_frames } => {
                image_data.frames.push(ImageFrame {
                    partial_renders: std::mem::take(&mut partial_renders),
                    duration: std::mem::replace(&mut frame_duration, 0.0),
                    channels: outputs.take().unwrap(),
                    color_type,
                    total_bytes,
                });
                if has_more_frames {
                    outputs = Some(make_outputs()?);
                }
            }
            Event::Complete => break,
            Event::NeedMoreInput { .. } => {
                if render_interval.is_some() && input.available_bytes()? > 0 {
                    if let Some(ref mut output_bufs) = output_bufs {
                        let changed =
                            decoder.flush_pixels(output_bufs, Some(&mut RayonParallelRunner))?;
                        if changed {
                            partial_renders.push(PartialRender {
                                byte_index: total_bytes.saturating_sub(input.available_bytes()?),
                                channels: outputs
                                    .as_ref()
                                    .unwrap()
                                    .iter()
                                    .map(|x| x.try_clone())
                                    .collect::<Result<_, _>>()?,
                            });
                        }
                    }
                    continue;
                } else if allow_partial_files {
                    if let Some(ref mut output_bufs) = output_bufs {
                        decoder.flush_pixels(output_bufs, Some(&mut RayonParallelRunner))?;
                        image_data.frames.push(ImageFrame {
                            partial_renders,
                            duration: frame_duration,
                            channels: outputs.take().unwrap(),
                            color_type,
                            total_bytes,
                        });
                    }
                    break;
                }
                return Err(eyre!("Source file truncated"));
            }
        }
    }

    if let Some(ref mut transformer) = cms_transformer {
        let black_channel = info
            .extra_channels
            .iter()
            .enumerate()
            .find(|x| x.1.ec_type == ExtraChannel::Black)
            .map(|(x, _)| {
                if interleave_alpha && main_alpha_channel.is_some_and(|i| i < x) {
                    x
                } else {
                    x + 1 // + 1 as color channels are at index 0 of output buffer
                }
            });
        for frame in &mut image_data.frames {
            let black_image = black_channel.map(|x| frame.channels.remove(x));
            apply_cms(
                &mut frame.channels[0],
                black_image.as_ref(),
                samples_per_pixel,
                color_channels,
                output_type,
                transformer.as_mut(),
                info.size.0,
                info.size.1,
            )?;
            for partial in &mut frame.partial_renders {
                let black_image = black_channel.map(|x| partial.channels.remove(x));
                apply_cms(
                    &mut partial.channels[0],
                    black_image.as_ref(),
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

#[allow(clippy::too_many_arguments)]
fn apply_cms(
    image: &mut OwnedRawImage,
    black_image: Option<&OwnedRawImage>,
    samples_per_pixel: usize,
    color_channels: usize,
    output_type: OutputDataType,
    transformer: &mut dyn jxl_cms::JxlCmsTransformer,
    width: usize,
    height: usize,
) -> Result<()> {
    let input_channels = color_channels + if black_image.is_some() { 1 } else { 0 };
    let mut row_color_buffer = vec![0.0f32; width * input_channels];
    let mut row_output_buffer = vec![0.0f32; width * color_channels];
    for y in 0..height {
        let row_bytes = image.row(y);
        let black_row_bytes = black_image.map(|img| img.row(y));
        // 1. Extract and convert to f32
        match output_type {
            OutputDataType::U8 => {
                if let Some(black_bytes) = black_row_bytes {
                    for x in 0..width {
                        for c in 0..color_channels {
                            let idx = x * samples_per_pixel + c;
                            row_color_buffer[x * input_channels + c] =
                                row_bytes[idx] as f32 / 255.0;
                        }
                        row_color_buffer[x * input_channels + color_channels] =
                            black_bytes[x] as f32 / 255.0;
                    }
                } else {
                    for x in 0..width {
                        for c in 0..color_channels {
                            let idx = x * samples_per_pixel + c;
                            row_color_buffer[x * color_channels + c] =
                                row_bytes[idx] as f32 / 255.0;
                        }
                    }
                }
            }
            OutputDataType::U16 => {
                if let Some(black_bytes) = black_row_bytes {
                    for x in 0..width {
                        for c in 0..color_channels {
                            let idx = (x * samples_per_pixel + c) * 2;
                            let val = u16::from_ne_bytes([row_bytes[idx], row_bytes[idx + 1]]);
                            row_color_buffer[x * input_channels + c] = val as f32 / 65535.0;
                        }
                        let k_idx = x * 2;
                        let k_val =
                            u16::from_ne_bytes([black_bytes[k_idx], black_bytes[k_idx + 1]]);
                        row_color_buffer[x * input_channels + color_channels] =
                            k_val as f32 / 65535.0;
                    }
                } else {
                    for x in 0..width {
                        for c in 0..color_channels {
                            let idx = (x * samples_per_pixel + c) * 2;
                            let val = u16::from_ne_bytes([row_bytes[idx], row_bytes[idx + 1]]);
                            row_color_buffer[x * color_channels + c] = val as f32 / 65535.0;
                        }
                    }
                }
            }
            OutputDataType::F16 => {
                if let Some(black_bytes) = black_row_bytes {
                    for x in 0..width {
                        for c in 0..color_channels {
                            let idx = (x * samples_per_pixel + c) * 2;
                            let val = u16::from_ne_bytes([row_bytes[idx], row_bytes[idx + 1]]);
                            row_color_buffer[x * input_channels + c] = f16::from_bits(val).to_f32();
                        }
                        let k_idx = x * 2;
                        let k_val =
                            u16::from_ne_bytes([black_bytes[k_idx], black_bytes[k_idx + 1]]);
                        row_color_buffer[x * input_channels + color_channels] =
                            f16::from_bits(k_val).to_f32();
                    }
                } else {
                    for x in 0..width {
                        for c in 0..color_channels {
                            let idx = (x * samples_per_pixel + c) * 2;
                            let val = u16::from_ne_bytes([row_bytes[idx], row_bytes[idx + 1]]);
                            row_color_buffer[x * color_channels + c] = f16::from_bits(val).to_f32();
                        }
                    }
                }
            }
            OutputDataType::F32 => {
                if let Some(black_bytes) = black_row_bytes {
                    for x in 0..width {
                        for c in 0..color_channels {
                            let idx = (x * samples_per_pixel + c) * 4;
                            let val = f32::from_ne_bytes([
                                row_bytes[idx],
                                row_bytes[idx + 1],
                                row_bytes[idx + 2],
                                row_bytes[idx + 3],
                            ]);
                            row_color_buffer[x * input_channels + c] = val;
                        }
                        let k_idx = x * 4;
                        row_color_buffer[x * input_channels + color_channels] =
                            f32::from_ne_bytes([
                                black_bytes[k_idx],
                                black_bytes[k_idx + 1],
                                black_bytes[k_idx + 2],
                                black_bytes[k_idx + 3],
                            ]);
                    }
                } else {
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
                        let val_f16 = f16::from_f32(val);
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

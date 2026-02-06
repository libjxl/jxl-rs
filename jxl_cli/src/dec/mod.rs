// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::{
    io::{Error, IoSliceMut},
    path::Path,
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

/// A wrapper around a byte slice that implements JxlBitstreamInput for streaming decoding.
/// Unlike `&[u8]`, this reports only the "available" bytes we've fed so far, not the total buffer size.
struct StreamingInput<'a> {
    data: &'a [u8],
    position: usize,
    available_limit: usize, // How many bytes we're pretending to have available
}

impl<'a> StreamingInput<'a> {
    fn new(data: &'a [u8], available_limit: usize) -> Self {
        Self {
            data,
            position: 0,
            available_limit,
        }
    }

    fn set_available_limit(&mut self, new_limit: usize) {
        self.available_limit = new_limit.min(self.data.len());
    }
}

impl<'a> JxlBitstreamInput for StreamingInput<'a> {
    fn available_bytes(&mut self) -> std::result::Result<usize, Error> {
        // Report only up to available_limit, not the full data length
        Ok(self.available_limit.saturating_sub(self.position))
    }

    fn read(&mut self, bufs: &mut [IoSliceMut]) -> std::result::Result<usize, Error> {
        let available = self.available_limit.saturating_sub(self.position);
        if available == 0 {
            return Ok(0);
        }

        let mut total_read = 0;
        for buf in bufs {
            let to_read = buf.len().min(available - total_read);
            if to_read == 0 {
                break;
            }
            buf[..to_read].copy_from_slice(&self.data[self.position..self.position + to_read]);
            self.position += to_read;
            total_read += to_read;
        }
        Ok(total_read)
    }

    fn skip(&mut self, bytes: usize) -> std::result::Result<usize, Error> {
        let available = self.available_limit.saturating_sub(self.position);
        let to_skip = bytes.min(available);
        self.position += to_skip;
        Ok(to_skip)
    }
}

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

/// Configuration for progressive decoding
pub struct ProgressiveDecodeConfig<'a> {
    pub override_bitdepth: Option<usize>,
    pub data_type: Option<OutputDataType>,
    pub supported_data_types: &'a [OutputDataType],
    pub interleave_alpha: bool,
    pub linear_output: bool,
    pub allow_partial_files: bool,
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

/// Parse progressive step size from string (either bytes or percentage)
fn parse_step_size(step_str: &str, total_size: usize) -> Result<usize> {
    if let Some(percentage) = step_str.strip_suffix('%') {
        let pct: f64 = percentage
            .parse()
            .map_err(|_| eyre!("Invalid percentage: {}", percentage))?;
        if pct <= 0.0 || pct > 100.0 {
            return Err(eyre!("Percentage must be between 0 and 100"));
        }
        Ok(((total_size as f64 * pct / 100.0) as usize).max(1))
    } else {
        step_str
            .parse::<usize>()
            .map_err(|_| eyre!("Invalid step size: {}", step_str))
    }
}

// Progressive decode with single decoder instance (incremental data)
// This is true streaming decoding: one decoder, feed data incrementally, flush at each step
pub fn decode_frames_progressive<F>(
    data: &[u8],
    output_path: &Path,
    step_size_str: &str,
    make_decoder_options: F,
    config: &ProgressiveDecodeConfig,
) -> Result<(DecodeOutput, Duration)>
where
    F: Fn() -> JxlDecoderOptions,
{
    let step_size = parse_step_size(step_size_str, data.len())?;
    let mut total_duration = Duration::new(0, 0);
    let mut total_process_time = Duration::new(0, 0);
    let mut total_flush_time = Duration::new(0, 0);
    let mut total_write_time = Duration::new(0, 0);

    println!(
        "Progressive decoding (streaming) with step size: {} bytes ({:.1}% of {} total)",
        step_size,
        (step_size as f64 / data.len() as f64) * 100.0,
        data.len()
    );

    // Prepare output filename components
    let output_stem = output_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| eyre!("Invalid output filename"))?;
    let output_ext = output_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("png");
    let output_dir = output_path.parent().unwrap_or_else(|| Path::new("."));

    // Create a single decoder instance with streaming input
    let start = Instant::now();
    let mut position = step_size.min(data.len());
    let mut streaming_input = StreamingInput::new(data, position);

    println!("Step 1: Decoding header with {} bytes...", position);
    let mut decoder_with_image_info = decode_header(&mut streaming_input, make_decoder_options())?;

    // Get info and setup
    let info = decoder_with_image_info.basic_info().clone();
    let embedded_profile = decoder_with_image_info.embedded_color_profile().clone();

    let output_type = if let Some(ot) = config.data_type
        && config.supported_data_types.contains(&ot)
    {
        ot
    } else {
        if config.data_type.is_some() {
            eprintln!("Warning: requested output type is not compatible with output format");
        }
        let bit_depth = config
            .override_bitdepth
            .unwrap_or(info.bit_depth.bits_per_sample() as usize);
        *config
            .supported_data_types
            .iter()
            .find(|x| x.bits_per_sample() >= bit_depth)
            .unwrap_or(config.supported_data_types.last().unwrap())
    };

    let main_alpha_channel = info
        .extra_channels
        .iter()
        .enumerate()
        .find(|x| x.1.ec_type == ExtraChannel::Alpha)
        .map(|x| x.0);

    let interleave_alpha = config.interleave_alpha && main_alpha_channel.is_some();

    // Set the pixel format
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
    if config.linear_output
        && let JxlColorProfile::Simple(enc) = decoder_with_image_info.output_color_profile().clone()
    {
        decoder_with_image_info
            .set_output_color_profile(JxlColorProfile::Simple(enc.with_linear_tf()))?;
    }
    let output_profile = decoder_with_image_info.output_color_profile().clone();

    let extra_channels = info.extra_channels.len() - if interleave_alpha { 1 } else { 0 };
    let pixel_format = decoder_with_image_info.current_pixel_format().clone();
    let color_type = pixel_format.color_type;
    let samples_per_pixel = pixel_format.color_type.samples_per_pixel();

    let mut step_number = 1;
    let mut frame_count = 0;

    // Process frames progressively
    loop {
        // Try to get frame info - may need multiple attempts with more data
        let decoder_with_frame_info = loop {
            match decoder_with_image_info.process(&mut streaming_input)? {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    if position >= data.len() {
                        println!("End of data reached during frame info");
                        total_duration += start.elapsed();

                        // Do final complete decode
                        let mut full_data = data;
                        let (final_output, _) = decode_frames(
                            &mut full_data,
                            make_decoder_options(),
                            config.override_bitdepth,
                            config.data_type,
                            config.supported_data_types,
                            config.interleave_alpha,
                            config.linear_output,
                            config.allow_partial_files,
                        )?;
                        return Ok((final_output, total_duration));
                    }
                    // Need more data for frame info
                    println!(
                        "  Need more data for frame {}, adding more...",
                        frame_count + 1
                    );
                    decoder_with_image_info = fallback;
                    step_number += 1;
                    position = (position + step_size).min(data.len());
                    streaming_input.set_available_limit(position);
                }
            }
        };

        let frame_header = decoder_with_frame_info.frame_header();
        let frame_size = frame_header.size;
        let frame_size = (
            frame_size.0 * output_type.bits_per_sample() / 8,
            frame_size.1,
        );

        // Create output buffers for this frame
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

        // Try to decode frame data, flushing progressively
        let mut current_decoder = decoder_with_frame_info;
        loop {
            println!(
                "Step {}: Decoding frame {} with {} bytes ({:.1}%)...",
                step_number,
                frame_count + 1,
                position,
                (position as f64 / data.len() as f64) * 100.0
            );

            let process_start = Instant::now();
            match current_decoder.process(&mut streaming_input, &mut output_bufs)? {
                ProcessingResult::Complete { result } => {
                    // Frame fully decoded
                    let process_time = process_start.elapsed();
                    total_process_time += process_time;
                    println!(
                        "  Frame {} complete! (process: {:.3}ms)",
                        frame_count + 1,
                        process_time.as_secs_f64() * 1000.0
                    );
                    decoder_with_image_info = result;
                    frame_count += 1;
                    break;
                }
                ProcessingResult::NeedsMoreInput { mut fallback, .. } => {
                    let process_time = process_start.elapsed();
                    total_process_time += process_time;

                    if position >= data.len() {
                        // No more data, flush what we have
                        println!(
                            "  End of data, flushing frame {} (incomplete)...",
                            frame_count + 1
                        );
                        let flush_start = Instant::now();
                        fallback.flush_pixels(&mut output_bufs)?;
                        let flush_time = flush_start.elapsed();
                        total_flush_time += flush_time;
                        println!(
                            "  Times - process: {:.3}ms, flush: {:.3}ms",
                            process_time.as_secs_f64() * 1000.0,
                            flush_time.as_secs_f64() * 1000.0
                        );
                        drop(output_bufs); // Release borrow

                        // Save final incomplete frame
                        let intermediate_path = output_dir
                            .join(format!("{}-{:03}.{}", output_stem, step_number, output_ext));
                        println!(
                            "Saving final incomplete frame to: {}",
                            intermediate_path.display()
                        );

                        let write_start = Instant::now();
                        let temp_output = DecodeOutput {
                            size: info.size,
                            frames: vec![ImageFrame {
                                duration: frame_header.duration.unwrap_or(0.0),
                                channels: outputs,
                                color_type,
                            }],
                            data_type: output_type,
                            original_bit_depth: info.bit_depth.clone(),
                            output_profile: output_profile.clone(),
                            embedded_profile: embedded_profile.clone(),
                            jxl_animation: info.animation.clone(),
                        };

                        let output_format = crate::enc::OutputFormat::from_output_filename(
                            &intermediate_path.to_string_lossy(),
                        )?;
                        output_format.save_image(&temp_output, &intermediate_path)?;
                        total_write_time += write_start.elapsed();

                        total_duration += start.elapsed();
                        println!("\n=== Progressive Decode Summary ===");
                        println!(
                            "Total processing time: {:.3}ms",
                            total_process_time.as_secs_f64() * 1000.0
                        );
                        println!(
                            "Total flushing time:   {:.3}ms",
                            total_flush_time.as_secs_f64() * 1000.0
                        );
                        println!(
                            "Total write time:      {:.3}ms",
                            total_write_time.as_secs_f64() * 1000.0
                        );
                        println!(
                            "Total time:            {:.3}ms",
                            total_duration.as_secs_f64() * 1000.0
                        );
                        return Ok((temp_output, total_duration));
                    }

                    // Flush intermediate result
                    println!(
                        "  Need more data for frame {}, flushing progressive preview...",
                        frame_count + 1
                    );
                    let flush_start = Instant::now();
                    fallback.flush_pixels(&mut output_bufs)?;
                    let flush_time = flush_start.elapsed();
                    total_flush_time += flush_time;
                    println!(
                        "  Times - process: {:.3}ms, flush: {:.3}ms",
                        process_time.as_secs_f64() * 1000.0,
                        flush_time.as_secs_f64() * 1000.0
                    );
                    drop(output_bufs); // Release borrow to allow cloning

                    // Save intermediate result
                    let intermediate_path = output_dir
                        .join(format!("{}-{:03}.{}", output_stem, step_number, output_ext));
                    println!(
                        "  Saving intermediate result to: {}",
                        intermediate_path.display()
                    );

                    let write_start = Instant::now();
                    // Clone outputs for saving
                    let cloned_outputs: Vec<OwnedRawImage> = outputs
                        .iter()
                        .map(|o| o.try_clone())
                        .collect::<std::result::Result<Vec<_>, _>>()?;

                    let temp_output = DecodeOutput {
                        size: info.size,
                        frames: vec![ImageFrame {
                            duration: frame_header.duration.unwrap_or(0.0),
                            channels: cloned_outputs,
                            color_type,
                        }],
                        data_type: output_type,
                        original_bit_depth: info.bit_depth.clone(),
                        output_profile: output_profile.clone(),
                        embedded_profile: embedded_profile.clone(),
                        jxl_animation: info.animation.clone(),
                    };

                    let output_format = crate::enc::OutputFormat::from_output_filename(
                        &intermediate_path.to_string_lossy(),
                    )?;
                    output_format.save_image(&temp_output, &intermediate_path)?;
                    total_write_time += write_start.elapsed();

                    // Recreate output_bufs for next iteration
                    output_bufs = outputs
                        .iter_mut()
                        .map(|x| {
                            let rect = Rect {
                                size: x.byte_size(),
                                origin: (0, 0),
                            };
                            JxlOutputBuffer::from_image_rect_mut(x.get_rect_mut(rect))
                        })
                        .collect();

                    // Feed more data and continue with same decoder
                    step_number += 1;
                    position = (position + step_size).min(data.len());
                    streaming_input.set_available_limit(position);
                    current_decoder = fallback;
                }
            }
        }

        if !decoder_with_image_info.has_more_frames() {
            break;
        }
    }

    total_duration += start.elapsed();

    println!("\n=== Progressive Decode Summary ===");
    println!(
        "Total processing time: {:.3}ms",
        total_process_time.as_secs_f64() * 1000.0
    );
    println!(
        "Total flushing time:   {:.3}ms",
        total_flush_time.as_secs_f64() * 1000.0
    );
    println!(
        "Total write time:      {:.3}ms",
        total_write_time.as_secs_f64() * 1000.0
    );
    println!(
        "Total time:            {:.3}ms",
        total_duration.as_secs_f64() * 1000.0
    );

    // Return final complete decode (we might not have this if we only got partial data)
    // For now, do a fresh final decode to ensure completeness
    println!("\n=== Single Full Decode (for comparison) ===");
    let single_start = Instant::now();
    let mut full_data = data;
    let (final_output, _) = decode_frames(
        &mut full_data,
        make_decoder_options(),
        config.override_bitdepth,
        config.data_type,
        config.supported_data_types,
        config.interleave_alpha,
        config.linear_output,
        config.allow_partial_files,
    )?;
    let single_time = single_start.elapsed();
    println!(
        "Single decode time:    {:.3}ms",
        single_time.as_secs_f64() * 1000.0
    );
    println!(
        "Progressive overhead:  {:.3}ms ({:.1}%)",
        (total_process_time.as_secs_f64() + total_flush_time.as_secs_f64()
            - single_time.as_secs_f64())
            * 1000.0,
        ((total_process_time.as_secs_f64() + total_flush_time.as_secs_f64())
            / single_time.as_secs_f64()
            - 1.0)
            * 100.0
    );

    Ok((final_output, total_duration))
}

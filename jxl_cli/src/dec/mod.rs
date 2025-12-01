// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::time::{Duration, Instant};

use color_eyre::eyre::{Result, eyre};
use jxl::{
    api::{
        JxlAnimation, JxlBitDepth, JxlBitstreamInput, JxlColorProfile, JxlColorType, JxlDecoder,
        JxlDecoderOptions, JxlExtraChannel, JxlOutputBuffer, ProcessingResult, states::WithImageInfo,
    },
    image::{Image, ImageDataType, Rect},
};

pub struct ImageFrame<T: ImageDataType> {
    pub channels: Vec<Image<T>>,
    pub duration: f64,
    pub color_type: JxlColorType,
    pub name: String,
}

pub struct DecodeOutput<T: ImageDataType> {
    pub size: (usize, usize),
    pub frames: Vec<ImageFrame<T>>,
    pub original_bit_depth: JxlBitDepth,
    pub output_profile: JxlColorProfile,
    pub embedded_profile: JxlColorProfile,
    pub jxl_animation: Option<JxlAnimation>,
    pub extra_channels: Vec<JxlExtraChannel>,
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

/// Decode a JXL image from any input that implements JxlBitstreamInput.
/// This works with both byte slices (`&mut &[u8]`) and buffered readers (`&mut BufReader<File>`).
pub fn decode_frames<In: JxlBitstreamInput>(
    input: &mut In,
    decoder_options: JxlDecoderOptions,
) -> Result<(DecodeOutput<f32>, Duration)> {
    let start = Instant::now();

    let mut decoder_with_image_info = decode_header(input, decoder_options)?;

    let info = decoder_with_image_info.basic_info();
    let embedded_profile = decoder_with_image_info.embedded_color_profile().clone();
    let output_profile = decoder_with_image_info.output_color_profile().clone();

    let mut image_data = DecodeOutput {
        size: info.size,
        frames: Vec::new(),
        original_bit_depth: info.bit_depth.clone(),
        output_profile,
        embedded_profile,
        jxl_animation: info.animation.clone(),
        extra_channels: info.extra_channels.clone(),
    };

    let extra_channels = info.extra_channels.len();
    let pixel_format = decoder_with_image_info.current_pixel_format().clone();
    let color_type = pixel_format.color_type;
    // TODO(zond): This is the way the API works right now, let's improve it when the API is cleverer.
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

        let mut outputs = vec![Image::<f32>::new((
            frame_size.0 * samples_per_pixel,
            frame_size.1,
        ))?];

        for _ in 0..extra_channels {
            outputs.push(Image::<f32>::new(frame_size)?);
        }

        let mut output_bufs: Vec<JxlOutputBuffer<'_>> = outputs
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
            channels: outputs,
            color_type,
            name: frame_header.name.clone(),
        });

        if !decoder_with_image_info.has_more_frames() {
            break;
        }
    }

    Ok((image_data, start.elapsed()))
}

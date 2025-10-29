// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use clap::{Arg, Command};
use color_eyre::eyre::{Result, eyre};
use jxl::api::{
    JxlBitDepth, JxlColorEncoding, JxlColorProfile, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer,
    ProcessingResult,
};
use jxl::headers::extra_channels::ExtraChannel;
use jxl::image::Image;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

fn parse_jxl(path: &Path) -> Result<()> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let options = JxlDecoderOptions::default();
    let mut initialized_decoder = JxlDecoder::<jxl::api::states::Initialized>::new(options);

    let mut buffer = vec![0; 16384];
    let mut offset = 0usize;
    let mut len = 0usize;

    macro_rules! advance_decoder {
        ($decoder: ident $(, $extra_arg: expr)?) => {
            loop {
                let mut data = &buffer[offset..(offset+len)];
                let result = $decoder.process(&mut data $(, $extra_arg)?)?;
                offset += len - data.len();
                len = data.len();
                match result {
                    ProcessingResult::Complete { result } => break Ok(result),
                    ProcessingResult::NeedsMoreInput { fallback, .. } => {
                        $decoder = fallback;
                        if len == 0 {
                            offset = 0;
                            len = reader.read(&mut buffer)?;
                            if len == 0 {
                                break Err(eyre!("Source file {:?} truncated", path));
                            }
                        }
                    }
                }
            }
        };
    }

    let mut decoder_with_image_info = advance_decoder!(initialized_decoder)?;

    let info = decoder_with_image_info.basic_info().clone();

    let how_lossy = if info.uses_original_profile {
        "(possibly) lossless"
    } else {
        "lossy"
    };
    let color_space = format!("{}", decoder_with_image_info.embedded_color_profile());
    let alpha_info = if info
        .extra_channels
        .iter()
        .any(|c| c.ec_type == ExtraChannel::Alpha)
    {
        "+Alpha"
    } else {
        ""
    };
    let image_or_animation = if info.animation.is_some() {
        "Animation"
    } else {
        "Image"
    };
    print!(
        "JPEG XL {}, {}x{}, {}, {}-bit, {}{}",
        image_or_animation,
        info.size.0,
        info.size.1,
        how_lossy,
        info.bit_depth.bits_per_sample(),
        color_space,
        alpha_info,
    );
    if let JxlBitDepth::Float {
        bits_per_sample: _,
        exponent_bits_per_sample: ebps,
    } = info.bit_depth
    {
        print!(", float ({} exponent bits)", ebps);
    }
    println!();
    match decoder_with_image_info.output_color_profile() {
        JxlColorProfile::Icc(icc) => match lcms2::Profile::new_icc(icc.as_slice()) {
            Err(_) => println!("with unparseable ICC profile"),
            Ok(profile) => {
                match profile.info(lcms2::InfoType::Description, lcms2::Locale::none()) {
                    None => println!("with undescribed {}-byte ICC profile", icc.len()),
                    Some(description) => {
                        println!(
                            "with {}-byte ICC profile (description: {})",
                            icc.len(),
                            description
                        )
                    }
                }
            }
        },
        JxlColorProfile::Simple(color_encoding) => match color_encoding {
            JxlColorEncoding::GrayscaleColorSpace {
                white_point,
                transfer_function,
                rendering_intent,
            } => {
                println!(
                    "White point: {}, Transfer function: {}, Rendering intent: {}",
                    white_point, transfer_function, rendering_intent
                );
            }
            JxlColorEncoding::RgbColorSpace {
                white_point,
                primaries,
                transfer_function,
                rendering_intent,
            } => {
                println!(
                    "White point: {}, Primaries: {}, Transfer function: {}, Rendering intent: {}",
                    white_point, primaries, transfer_function, rendering_intent
                );
            }
            JxlColorEncoding::XYB { rendering_intent } => {
                println!("Rendering intent: {}", rendering_intent);
            }
        },
    }

    if let Some(animation) = info.animation {
        let pixel_format = decoder_with_image_info.current_pixel_format().clone();
        let num_channels: usize = pixel_format.color_type.samples_per_pixel();
        let mut num_frames = 0;
        let mut total_seconds = 0.0;

        loop {
            let mut decoder_with_frame_info = advance_decoder!(decoder_with_image_info)?;

            let duration = decoder_with_frame_info.frame_header().duration.unwrap();
            total_seconds += duration;
            println!("Frame {}, duration {}s", num_frames, duration);

            let mut outputs = vec![Image::<f32>::new((
                info.size.0 * num_channels,
                info.size.1,
            ))?];

            for _ in 0..info.extra_channels.len() {
                outputs.push(Image::<f32>::new(info.size)?);
            }

            let mut output_bufs: Vec<JxlOutputBuffer<'_>> = outputs
                .iter_mut()
                .map(|x| JxlOutputBuffer::from_image_rect_mut(x.as_rect_mut().into_raw()))
                .collect();

            decoder_with_image_info = advance_decoder!(decoder_with_frame_info, &mut output_bufs)?;

            num_frames += 1;

            if !decoder_with_image_info.has_more_frames() {
                break;
            }
        }

        print!(
            "Animation length: {} frames in {} seconds",
            num_frames,
            total_seconds / 1000.0
        );
        if animation.have_timecodes {
            print!(" with (potentially) individual timecodes");
        }
        if animation.num_loops < 1 {
            println!(" (looping indefinitely)");
        } else if animation.num_loops > 1 {
            println!(
                " ({} loops, in total {} seconds)",
                animation.num_loops,
                animation.num_loops as f64 * total_seconds
            );
        }
    }
    Ok(())
}

fn main() {
    #[cfg(feature = "tracing-subscriber")]
    {
        use tracing_subscriber::{EnvFilter, fmt, prelude::*};
        tracing_subscriber::registry()
            .with(fmt::layer())
            .with(EnvFilter::from_default_env())
            .init();
    }

    let matches = Command::new("jxlinspect")
        .about("Provides info about a JXL file")
        .arg(
            Arg::new("filename")
                .help("The JXL file to analyze")
                .required(true)
                .index(1),
        )
        .get_matches();

    let filename = Path::new(matches.get_one::<String>("filename").unwrap());

    let res = parse_jxl(filename);
    if let Err(err) = res {
        println!("Error parsing JXL codestream: {err}");
    }
}

#[cfg(test)]
mod jxl_cli_test {
    use super::*;
    use jxl_macros::for_each_test_file;

    for_each_test_file!(parse_jxl);
}

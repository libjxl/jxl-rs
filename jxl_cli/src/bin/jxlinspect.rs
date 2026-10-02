// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use clap::{Arg, Command};
use color_eyre::eyre::{Result, eyre};
use jxl::api::{
    Event, ExtraChannel, JxlBitDepth, JxlColorEncoding, JxlColorProfile, JxlDecoder,
    JxlDecoderOptions,
};

fn parse_jxl(path: &Path) -> Result<()> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut options = JxlDecoderOptions::default();
    options.scan_frames_only = true;
    let mut decoder = JxlDecoder::new(options);

    if decoder.process(&mut reader, None, None)? != Event::BasicInfo {
        return Err(eyre!("Source file {:?} truncated", path));
    }

    let info = decoder.basic_info().unwrap().clone();

    let how_lossy = if info.uses_original_profile {
        "(possibly) lossless"
    } else {
        "lossy"
    };
    let color_space = format!("{}", decoder.embedded_color_profile().unwrap());
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
    match decoder.output_color_profile().unwrap() {
        JxlColorProfile::Icc(icc) => match moxcms::ColorProfile::new_from_slice(icc.as_slice()) {
            Err(_) => println!("with unparseable ICC profile"),
            Ok(profile) => {
                let description = match &profile.description {
                    Some(moxcms::ProfileText::PlainString(text)) => Some(text.as_str()),
                    Some(moxcms::ProfileText::Description(text)) => {
                        Some(if text.unicode_string.is_empty() {
                            text.ascii_string.as_str()
                        } else {
                            text.unicode_string.as_str()
                        })
                    }
                    Some(moxcms::ProfileText::Localizable(texts)) => {
                        texts.first().map(|text| text.value.as_str())
                    }
                    None => None,
                };
                match description {
                    None | Some("") => println!("with undescribed {}-byte ICC profile", icc.len()),
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
        let mut num_frames = 0;
        let mut total_seconds = 0.0;

        loop {
            match decoder.process(&mut reader, None, None)? {
                Event::FrameHeader => {
                    let duration = decoder.frame_header().unwrap().duration.unwrap();
                    total_seconds += duration;
                    println!("Frame {}, duration {}ms", num_frames, duration);
                    num_frames += 1;
                }
                Event::FrameComplete { .. } => {}
                Event::Complete => break,
                _ => {
                    return Err(eyre!("Source file {:?} truncated", path));
                }
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
        use tracing_subscriber::prelude::*;
        use tracing_subscriber::{EnvFilter, fmt};
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

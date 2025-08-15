// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use clap::{Arg, Command};
use color_eyre::eyre::{Result, WrapErr};
use jxl::bit_reader::BitReader;
use jxl::container::{ContainerParser, ParseEvent};
use jxl::frame::{DecoderState, Frame};
use jxl::headers::color_encoding::{ColorEncoding, Primaries, WhitePoint};
use jxl::headers::{FileHeader, JxlHeader};
use jxl::icc::IncrementalIccReader;
use std::cmp::Ordering;
use std::fs;
use std::io::Read;
use std::path::Path;

fn print_color_encoding(color_encoding: &ColorEncoding) {
    // Print the color space
    print!("Color Space: {:?}, ", color_encoding.color_space);

    // Print white point, depending on whether it's a custom white point
    match color_encoding.white_point {
        WhitePoint::Custom => {
            print!(
                "White point: custom ({}, {}), ",
                color_encoding.white.x, color_encoding.white.y
            );
        }
        _ => {
            print!("White point: {:?}, ", color_encoding.white_point);
        }
    }

    // Print primaries, check for custom primaries
    if color_encoding.primaries == Primaries::Custom {
        println!("Custom primaries: ");
        for (i, primary) in color_encoding.custom_primaries.iter().enumerate() {
            print!("  primary {}: ({}, {}), ", i + 1, primary.x, primary.y);
        }
    } else {
        print!("Primaries: {:?}, ", color_encoding.primaries);
    }

    // Print transfer function details
    if color_encoding.tf.have_gamma {
        print!(
            "Transfer function: gamma (gamma = {}), ",
            color_encoding.tf.gamma
        );
    } else {
        print!(
            "Transfer function: {:?}, ",
            color_encoding.tf.transfer_function
        );
    }

    println!("Rendering intent: {:?}", color_encoding.rendering_intent);
}

fn parse_jxl_codestream(data: &[u8], verbose: bool) -> Result<()> {
    let mut br = BitReader::new(data);
    let file_header = FileHeader::read(&mut br)?;

    // Non-verbose output
    let how_lossy = if file_header.image_metadata.xyb_encoded {
        "lossy"
    } else {
        "(possibly) lossless"
    };

    let color_space = format!(
        "{:?}",
        file_header.image_metadata.color_encoding.color_space
    );
    let alpha_info = match file_header
        .image_metadata
        .extra_channel_info
        .iter()
        .any(|info| info.alpha_associated())
    {
        true => "+Alpha",
        false => "",
    };
    let image_or_animation = match file_header.image_metadata.animation {
        None => "Image",
        Some(_) => "Animation",
    };
    print!(
        "JPEG XL {}, {}x{}, {}, {}-bit {}{}",
        image_or_animation,
        file_header.size.xsize(),
        file_header.size.ysize(),
        how_lossy,
        file_header.image_metadata.bit_depth.bits_per_sample(),
        color_space,
        alpha_info,
    );
    if file_header
        .image_metadata
        .bit_depth
        .exponent_bits_per_sample()
        != 0
    {
        print!(
            "float ({} exponent bits)",
            file_header
                .image_metadata
                .bit_depth
                .exponent_bits_per_sample()
        );
    }
    println!();
    if file_header.image_metadata.color_encoding.want_icc {
        let mut r = IncrementalIccReader::new(&mut br)?;
        r.read_all(&mut br)?;
        let icc = r.finalize()?;
        match lcms2::Profile::new_icc(icc.as_slice()) {
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
        }
    } else {
        print_color_encoding(&file_header.image_metadata.color_encoding);
    }
    if verbose {
        // Verbose output: Use Debug trait to print the FileHeaders
        println!("{file_header:#?}");
    }
    // TODO(firsching): handle frames which are blended together, also within animations.
    if let Some(ref animation) = file_header.image_metadata.animation {
        let mut total_duration = 0.0f64;
        let mut decoder_state = DecoderState::new(file_header.clone());
        loop {
            let frame = Frame::new(&mut br, decoder_state)?;
            let ms = frame.header().duration(animation);
            total_duration += ms;
            println!(
                "frame: {:?}x{:?} at position ({},{}), duration {ms}ms",
                frame.header().width,
                frame.header().height,
                frame.header().x0,
                frame.header().y0
            );
            br.jump_to_byte_boundary()?;
            br.skip_bits(frame.total_bytes_in_toc() * 8)?;
            if let Some(state) = frame.finalize()?.decoder_state {
                decoder_state = state;
            } else {
                break;
            }
        }
        print!(
            "Animation length: {} seconds",
            total_duration
                * (if animation.num_loops > 1 {
                    animation.num_loops as f64
                } else {
                    1.0f64
                })
                * 0.001
        );
        match animation.num_loops.cmp(&1) {
            Ordering::Greater => println!(
                " ({} loops of {} seconds)",
                animation.num_loops,
                total_duration * 0.001
            ),
            Ordering::Equal => println!(),
            Ordering::Less => println!(" (looping indefinitely)"),
        }
    }
    Ok(())
}

fn parse_jxl(filename: &Path, verbose: bool) -> Result<()> {
    let mut file = fs::File::open(filename)
        .wrap_err_with(|| format!("Failed to read source image from {:?}", filename))?;
    // Set up the container parser and buffers
    let mut parser = ContainerParser::new();
    let mut buf = vec![0u8; 4096];
    let mut buf_valid = 0usize;
    let mut codestream = Vec::new();

    loop {
        let chunk_size = file
            .read(&mut buf[buf_valid..])
            .wrap_err_with(|| format!("Failed reading from {:?}", filename))?;
        if chunk_size == 0 {
            break;
        }
        buf_valid += chunk_size;

        for event in parser.process_bytes(&buf[..buf_valid]) {
            match event? {
                ParseEvent::BitstreamKind(kind) => {
                    println!("Bitstream kind: {kind:?}");
                }
                ParseEvent::Codestream(data) => {
                    codestream.extend_from_slice(data);
                }
            }
        }

        let consumed = parser.previous_consumed_bytes();
        buf.copy_within(consumed..buf_valid, 0);
        buf_valid -= consumed;
    }

    parse_jxl_codestream(&codestream, verbose)
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
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("Provides more verbose output")
                .num_args(0),
        )
        .get_matches();

    let filename = Path::new(matches.get_one::<String>("filename").unwrap());
    let verbose = matches.get_flag("verbose");

    if verbose {
        println!("Processing file: {filename:?}");
    }

    let res = parse_jxl(filename, verbose);
    if let Err(err) = res {
        println!("Error parsing JXL codestream: {err}");
    }
}

#[cfg(test)]
mod jxl_cli_test {
    use color_eyre::eyre::Result;
    use jxl_macros::for_each_test_file;
    use std::path::Path;

    use crate::parse_jxl;

    fn read_file_from_path(path: &Path) -> Result<()> {
        parse_jxl(path, false)?;
        Ok(())
    }

    for_each_test_file!(read_file_from_path);
}

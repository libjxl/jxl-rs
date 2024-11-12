// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use clap::{Arg, Command};
use jxl::bit_reader::BitReader;
use jxl::container::{ContainerParser, ParseEvent};
use jxl::headers::color_encoding::{ColorEncoding, Primaries, WhitePoint};
use jxl::headers::encodings::UnconditionalCoder;
use jxl::headers::frame_header::{FrameHeader, Toc, TocNonserialized};
use jxl::headers::{FileHeader, JxlHeader};
use jxl::icc::read_icc;
use std::cmp::Ordering;
use std::fs;
use std::io::Read;

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

fn parse_jxl_codestream(data: &[u8], verbose: bool) -> Result<(), jxl::error::Error> {
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
        println!("with ICC profile")
    } else {
        print_color_encoding(&file_header.image_metadata.color_encoding);
    }
    if verbose {
        // Verbose output: Use Debug trait to print the FileHeaders
        println!("{:#?}", file_header);
        // TODO(firsching): consider printing more of less information for ICC
        // for verbose/non-verbose cases
        if file_header.image_metadata.color_encoding.want_icc {
            let icc_data = read_icc(&mut br)?;
            println!("ICC profile length: {} bytes", icc_data.len());
        }
    }
    //TODO(firsching): handle frames which are blended together, also within animations.
    if file_header.image_metadata.animation.is_some() {
        let mut total_duration = 0.0f64;
        let animation = file_header
            .image_metadata
            .animation
            .as_ref()
            .expect("This should never fail, it was just check in the if condition above");
        let mut not_is_last = true;
        while not_is_last {
            let frame_header = FrameHeader::read_unconditional(
                &(),
                &mut br,
                &file_header.frame_header_nonserialized(),
            )
            .unwrap();

            let ms = (frame_header.duration as f64) * 1000.0 * (animation.tps_denominator as f64)
                / (animation.tps_numerator as f64);
            total_duration += ms;
            println!(
                "frame: {:?}x{:?} at position ({},{}), duration {ms}ms",
                frame_header.xsize(&file_header),
                frame_header.ysize(&file_header),
                frame_header.x0,
                frame_header.y0
            );
            // Read TOC to skip to next
            let num_toc_entries = frame_header.num_toc_entries(&file_header);
            let toc = Toc::read_unconditional(
                &(),
                &mut br,
                &TocNonserialized {
                    num_entries: num_toc_entries,
                },
            )
            .unwrap();
            let entries = toc.entries;
            let num_bytes_to_skip: u32 = entries.into_iter().sum();
            br.jump_to_byte_boundary()?;
            not_is_last = !frame_header.is_last;
            // TODO: use return value
            br.skip_bits((num_bytes_to_skip * 8) as usize)?;
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

fn main() {
    #[cfg(feature = "tracing-subscriber")]
    {
        use tracing_subscriber::{fmt, prelude::*, EnvFilter};
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

    let filename = matches.get_one::<String>("filename").unwrap();
    let verbose = matches.get_flag("verbose");

    if verbose {
        println!("Processing file: {}", filename);
    }

    let mut file = fs::File::open(filename).expect("Cannot open file");

    // Set up the container parser and buffers
    let mut parser = ContainerParser::new();
    let mut buf = vec![0u8; 4096];
    let mut buf_valid = 0usize;
    let mut codestream = Vec::new();

    loop {
        let count = file
            .read(&mut buf[buf_valid..])
            .expect("Cannot read data from file");
        if count == 0 {
            break;
        }
        buf_valid += count;

        for event in parser.process_bytes(&buf[..buf_valid]) {
            match event {
                Ok(ParseEvent::BitstreamKind(kind)) => {
                    println!("Bitstream kind: {kind:?}");
                }
                Ok(ParseEvent::Codestream(data)) => {
                    codestream.extend_from_slice(data);
                }
                Err(err) => {
                    println!("Error parsing JXL codestream: {err}");
                    return;
                }
            }
        }

        let consumed = parser.previous_consumed_bytes();
        buf.copy_within(consumed..buf_valid, 0);
        buf_valid -= consumed;
    }

    let res = parse_jxl_codestream(&codestream, verbose);
    if let Err(err) = res {
        println!("Error parsing JXL codestream: {err}");
    }
}

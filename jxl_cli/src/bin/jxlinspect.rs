// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use clap::{Arg, Command};
use jxl::bit_reader::BitReader;
use jxl::container::{ContainerParser, ParseEvent};
use jxl::headers::color_encoding::{ColorEncoding, Primaries, WhitePoint};
use jxl::headers::{FileHeader, JxlHeader};
use jxl::icc::read_icc;
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
    let fileheaders = FileHeader::read(&mut br)?;

    // Non-verbose output
    if !verbose {
        let how_lossy = if fileheaders.image_metadata.xyb_encoded {
            "lossy"
        } else {
            "(possibly) lossless"
        };

        let color_space = format!("{:?}", fileheaders.image_metadata.color_encoding.color_space);
        let alpha_info = match fileheaders
            .image_metadata
            .extra_channel_info
            .iter()
            .any(|info| info.alpha_associated())
        {
            true => "+Alpha",
            false => "",
        };
        let image_or_animation = match fileheaders.image_metadata.animation {
            None => "Image",
            Some(_) => "Animation",
        };
        print!(
            "JPEG XL {}, {}x{}, {}, {}-bit {}{}",
            image_or_animation,
            fileheaders.size.xsize(),
            fileheaders.size.ysize(),
            how_lossy,
            fileheaders.image_metadata.bit_depth.bits_per_sample(),
            color_space,
            alpha_info,
        );
        if fileheaders.image_metadata.bit_depth.exponent_bits_per_sample() != 0 {
            print!(
                "float ({} exponent bits)",
                fileheaders.image_metadata.bit_depth.exponent_bits_per_sample()
            );
        }
        println!();
        if fileheaders.image_metadata.color_encoding.want_icc {
            println!("with ICC profile")
        } else {
            print_color_encoding(&fileheaders.image_metadata.color_encoding);
        }
        return Ok(());
    }

    // Verbose output: Use Debug trait to print the FileHeaders
    println!("{:#?}", fileheaders);

    // TODO(firsching): consider printing more of less information for ICC
    // for verbose/non-verbose cases
    if fileheaders.image_metadata.color_encoding.want_icc {
        let icc_data = read_icc(&mut br)?;
        println!("ICC profile length: {} bytes", icc_data.len());
    }
    // TODO(firsching): add frame header parsing for each frame

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

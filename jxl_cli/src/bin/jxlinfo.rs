// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use clap::{Arg, Command};
use jxl::bit_reader::BitReader;
use jxl::container::{ContainerParser, ParseEvent};
use jxl::headers::{FileHeaders, JxlHeader};
use jxl::icc::read_icc;
use std::fs;
use std::io::Read;

fn parse_jxl_codestream(data: &[u8], verbose: bool) -> Result<(), jxl::error::Error> {
    let mut br = BitReader::new(data);
    let fh = FileHeaders::read(&mut br)?;

    // Non-verbose output
    if !verbose {
        let how_lossy = if fh.image_metadata.xyb_encoded {
            "lossy"
        } else {
            "(possibly) lossless"
        };

        let color_space = format!("{:?}", fh.image_metadata.color_encoding.color_space);
        let alpha_info = match fh
            .image_metadata
            .extra_channel_info
            .iter()
            .any(|info| info.alpha_associated())
        {
            true => "+Alpha",
            false => "",
        };
        print!(
            "{}x{}, {}, {}-bit {}{}",
            fh.size.xsize(),
            fh.size.ysize(),
            how_lossy,
            fh.image_metadata.bit_depth.bits_per_sample(),
            color_space,
            alpha_info,
        );
        if fh.image_metadata.bit_depth.exponent_bits_per_sample() != 0 {
            print!(
                "float ({} exponent bits)",
                fh.image_metadata.bit_depth.exponent_bits_per_sample()
            );
        }
        println!();

        // TODO(firsching): print more info on color encoding

        return Ok(());
    }

    // Verbose output: Use Debug trait to print the FileHeaders
    println!("{:#?}", fh);

    // TODO(firsching): consider printing more of less information for ICC
    // for verbose/non-verbose cases
    if fh.image_metadata.color_encoding.want_icc {
        let icc_data = read_icc(&mut br)?;
        println!("ICC profile length: {} bytes", icc_data.len());
    }
    // TODO(firsching): add frame header parsing for each frame

    Ok(())
}

fn main() {
    let matches = Command::new("jxlinfo")
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

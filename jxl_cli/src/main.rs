// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use clap::Parser;
use jxl::container::{ContainerParser, ParseEvent};
use jxl::decode::{DecodeOptions, ImageData};
use jxl::error::Error;
use jxl::headers::bit_depth::BitDepth;
use std::fs;
use std::io::Read;
use std::path::PathBuf;

pub mod enc;

fn save_icc(icc_bytes: Vec<u8>, icc_filename: Option<PathBuf>) -> Result<(), Error> {
    match icc_filename {
        Some(icc_filename) => {
            std::fs::write(icc_filename, icc_bytes).map_err(|_| Error::OutputWriteFailure)
        }
        None => Ok(()),
    }
}

fn save_image(
    image_data: ImageData<f32>,
    bit_depth: BitDepth,
    output_filename: PathBuf,
) -> Result<(), Error> {
    let fn_str: String = String::from(output_filename.to_string_lossy());
    let mut output_bytes: Vec<u8> = vec![];
    if fn_str.ends_with(".ppm") {
        if image_data.frames.len() == 1 {
            assert_eq!(image_data.frames[0].size, image_data.size);
            if let [r, g, b] = &image_data.frames[0].channels[..] {
                output_bytes = enc::pnm::to_ppm_as_8bit(&[r.as_rect(), g.as_rect(), b.as_rect()]);
            }
        }
    } else if fn_str.ends_with(".pgm") {
        if image_data.frames.len() == 1 {
            assert_eq!(image_data.frames[0].size, image_data.size);
            if let [g] = &image_data.frames[0].channels[..] {
                output_bytes = enc::pnm::to_pgm_as_8bit(&g.as_rect());
            }
        }
    } else if fn_str.ends_with(".npy") {
        output_bytes = enc::numpy::to_numpy(image_data)?;
    } else if fn_str.ends_with(".png") {
        output_bytes = enc::png::to_png(image_data, bit_depth)?;
    }
    if output_bytes.is_empty() {
        return Err(Error::OutputFormatNotSupported);
    }
    if std::fs::write(output_filename, output_bytes).is_err() {
        Err(Error::OutputWriteFailure)
    } else {
        Ok(())
    }
}

#[derive(Parser)]
struct Opt {
    /// Input JXL file
    input: PathBuf,

    /// Output image file, should end in .ppm, .pgm, .png or .npy
    output: PathBuf,

    ///  If specified, writes the ICC profile of the decoded image
    #[clap(long)]
    icc_out: Option<PathBuf>,
}

fn main() -> Result<(), Error> {
    #[cfg(feature = "tracing-subscriber")]
    {
        use tracing_subscriber::{EnvFilter, fmt, prelude::*};
        tracing_subscriber::registry()
            .with(fmt::layer())
            .with(EnvFilter::from_default_env())
            .init();
    }

    let opt = Opt::parse();
    let input_filename = opt.input;
    let mut file = match fs::File::open(input_filename.clone()) {
        Ok(file) => file,
        Err(err) => {
            println!("Cannot open file: {err}");
            return Err(Error::FileNotFound(input_filename));
        }
    };

    let mut parser = ContainerParser::new();
    let mut buf = vec![0u8; 4096];
    let mut buf_valid = 0usize;
    let mut codestream = Vec::new();
    loop {
        let chunk_size = match file.read(&mut buf[buf_valid..]) {
            Ok(l) => l,
            Err(err) => {
                return Err(Error::InputReadFailure(err));
            }
        };
        if chunk_size == 0 {
            break;
        }
        buf_valid += chunk_size;

        for event in parser.process_bytes(&buf[..buf_valid]) {
            match event {
                Ok(ParseEvent::BitstreamKind(kind)) => {
                    println!("Bitstream kind: {kind:?}");
                }
                Ok(ParseEvent::Codestream(buf)) => {
                    codestream.extend_from_slice(buf);
                }
                Err(err) => {
                    println!("Error parsing JXL codestream: {err}");
                    return Err(err);
                }
            }
        }

        let consumed = parser.previous_consumed_bytes();
        buf.copy_within(consumed..buf_valid, 0);
        buf_valid -= consumed;
    }

    let mut options = DecodeOptions::new();
    options.xyb_output_linear = String::from(opt.output.to_string_lossy()).ends_with(".npy");
    let (image_data, bit_depth, icc_bytes) =
        jxl::decode::decode_jxl_codestream(options, &codestream)?;

    let icc_result = save_icc(icc_bytes, opt.icc_out);
    let image_result = save_image(image_data, bit_depth, opt.output);

    if let Err(ref err) = icc_result {
        println!("Failed to save ICC profile: {err}");
    }
    if let Err(ref err) = image_result {
        println!("Failed to save image: {err}");
    }

    icc_result?;
    image_result?;

    Ok(())
}

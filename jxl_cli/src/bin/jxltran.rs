// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::fs;
use std::io::IoSliceMut;
use std::path::PathBuf;

use clap::Parser;
use color_eyre::eyre::{Result, WrapErr, bail, ensure};
use jxl::api::{
    BoxParser, CONTAINER_SIGNATURE, CodestreamInput, Event, JxlDecoder, JxlDecoderOptions,
    JxlSignature, ProfileLevel, check_signature,
};
use jxl::error::Error;

const VERSION_STRING: &str = concat!(
    env!("VERGEN_GIT_DESCRIBE"),
    " (rustc ",
    env!("VERGEN_RUSTC_SEMVER"),
    ")"
);

// Standard 20-byte "ftyp" box for a JPEG XL container.
const FTYP_BOX: [u8; 20] = [
    0, 0, 0, 20, b'f', b't', b'y', b'p', b'j', b'x', b'l', b' ', 0, 0, 0, 0, b'j', b'x', b'l', b' ',
];

// 9-byte "jxll" box indicating Main Profile Level 10.
const JXLL_LEVEL10_BOX: [u8; 9] = [0, 0, 0, 9, b'j', b'x', b'l', b'l', 10];

const SMALL_BOX_HEADER_SIZE: u64 = 8;
const LARGE_BOX_HEADER_SIZE: u64 = 16;
const LARGE_BOX_CONTENT_SIZE_THRESHOLD: u64 = 0x1_0000_0000 - SMALL_BOX_HEADER_SIZE;

#[derive(Parser)]
#[command(version = VERSION_STRING)]
struct Opt {
    /// Input JXL file
    input: PathBuf,

    /// Output JXL file
    output: PathBuf,

    /// Pack a JPEG XL codestream into a file in the container file format. Automatically writes a jxll box if the codestream exceeds Level 5 limits
    #[clap(long, action, conflicts_with = "extract")]
    pack: bool,

    /// Force writing a Level 10 jxll box when packing (implies --pack)
    #[clap(long, action, conflicts_with = "extract")]
    level10: bool,

    /// Extract the JXL codestream from a container file
    #[clap(long, action, conflicts_with_all = ["pack", "level10"])]
    extract: bool,
}

fn append_box_header(box_type: &[u8; 4], content_size: usize, output: &mut Vec<u8>) {
    let size = content_size as u64;
    if size >= LARGE_BOX_CONTENT_SIZE_THRESHOLD {
        let box_size = size + LARGE_BOX_HEADER_SIZE;
        output.extend_from_slice(&1u32.to_be_bytes());
        output.extend_from_slice(box_type);
        output.extend_from_slice(&box_size.to_be_bytes());
    } else {
        let box_size = (size + SMALL_BOX_HEADER_SIZE) as u32;
        output.extend_from_slice(&box_size.to_be_bytes());
        output.extend_from_slice(box_type);
    }
}

// Checks whether a bare codestream exceeds Main Profile Level 5 limits and therefore
// requires a Level 10 jxll box when packed into a container.
fn needs_level10(codestream: &[u8]) -> bool {
    let mut options = JxlDecoderOptions::default();
    options.skip_preview = false;

    let mut decoder = JxlDecoder::new(options);
    let mut input = codestream;

    loop {
        match decoder.process(&mut input, None, None) {
            Ok(Event::BasicInfo | Event::FrameHeader | Event::FrameComplete { .. }) => {}
            Ok(Event::Complete | Event::NeedMoreInput { .. }) => return false,
            Err(Error::LevelLimitExceeded {
                level: ProfileLevel::Main5,
                ..
            }) => return true,
            Err(_) => return false,
        }
    }
}

// Packs a bare JPEG XL codestream into a JPEG XL container.
fn pack_codestream(
    input_bytes: &[u8],
    signature: JxlSignature,
    force_level10: bool,
) -> Result<Vec<u8>> {
    ensure!(
        signature == JxlSignature::Codestream,
        "Input file is not a codestream file"
    );

    let use_level10 = force_level10 || needs_level10(input_bytes);
    let jxll_len = if use_level10 {
        JXLL_LEVEL10_BOX.len()
    } else {
        0
    };

    // 12 bytes for "JXL ", 20 bytes for "ftyp", optional 9 bytes for "jxll",
    // and up to 16 bytes for the "jxlc" header.
    let mut container = Vec::with_capacity(
        CONTAINER_SIGNATURE.len()
            + FTYP_BOX.len()
            + jxll_len
            + LARGE_BOX_HEADER_SIZE as usize
            + input_bytes.len(),
    );
    container.extend_from_slice(&CONTAINER_SIGNATURE);
    container.extend_from_slice(&FTYP_BOX);
    if use_level10 {
        container.extend_from_slice(&JXLL_LEVEL10_BOX);
    }
    append_box_header(b"jxlc", input_bytes.len(), &mut container);
    container.extend_from_slice(input_bytes);

    Ok(container)
}

// Extracts the raw JPEG XL codestream from a JPEG XL container.
fn extract_codestream(mut input_bytes: &[u8], signature: JxlSignature) -> Result<Vec<u8>> {
    ensure!(
        signature == JxlSignature::Container,
        "Input file is not a container file."
    );

    let mut parser = BoxParser::with_aux_boxes(None);
    parser.close_input();

    let mut codestream = Vec::new();
    {
        let mut stream = CodestreamInput::new(&mut parser, &mut input_bytes);
        let mut buf = [0u8; 65536];
        loop {
            let n = stream.read(&mut [IoSliceMut::new(&mut buf)])?;
            if n == 0 {
                break;
            }
            codestream.extend_from_slice(&buf[..n]);
        }
    }

    if parser.container_level() == Some(ProfileLevel::Main10) {
        eprintln!("Warning: extracting raw codestream that is greater than level 5");
    }

    Ok(codestream)
}

fn apply_file_format_options(
    input_bytes: &[u8],
    opt: &Opt,
    signature: JxlSignature,
) -> Result<Vec<u8>> {
    if opt.extract {
        extract_codestream(input_bytes, signature)
    } else if opt.pack || opt.level10 {
        pack_codestream(input_bytes, signature, opt.level10)
    } else {
        Ok(input_bytes.to_vec())
    }
}

fn main() -> Result<()> {
    #[cfg(feature = "tracing-subscriber")]
    {
        use tracing_subscriber::prelude::*;
        use tracing_subscriber::{EnvFilter, fmt};
        tracing_subscriber::registry()
            .with(fmt::layer())
            .with(EnvFilter::from_default_env())
            .init();
    }

    let opt = Opt::parse();

    let input_bytes = fs::read(&opt.input)
        .wrap_err_with(|| format!("Failed to read input image {:?}", opt.input))?;

    let signature = check_signature(&input_bytes);
    if signature != JxlSignature::Codestream && signature != JxlSignature::Container {
        bail!("Input file is not a JPEG XL file.");
    }

    let output_bytes = apply_file_format_options(&input_bytes, &opt, signature)?;

    fs::write(&opt.output, &output_bytes)
        .wrap_err_with(|| format!("Failed to write output file {:?}", opt.output))?;

    Ok(())
}

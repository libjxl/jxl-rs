// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::fs;
use std::path::PathBuf;

use clap::Parser;
use color_eyre::eyre::{Result, WrapErr, bail, ensure};
use jxl::api::{JxlSignature, ProfileLevel, check_signature, extract_codestream, pack_codestream};

#[derive(Parser)]
#[command(version = jxl_cli::VERSION_STRING)]
struct Opt {
    /// Input JXL file
    input: PathBuf,

    /// Output JXL file
    output: PathBuf,

    /// Pack a JPEG XL codestream into a file in the container file format.
    /// Automatically writes a jxll box if the codestream exceeds Level 5 limits
    #[clap(long, action, conflicts_with = "extract", verbatim_doc_comment)]
    pack: bool,

    /// How to handle a Level 10 container box (implies --pack):
    ///   0 = never check (don't add jxll box)
    ///   1 = always add level 10 box
    ///   2 = fast check (minimal decode, header check only)
    ///   3 = accurate check (full decode, default when packing)
    #[clap(
        long,
        value_name = "0|1|2|3",
        value_parser = clap::value_parser!(u8).range(0..=3),
        conflicts_with = "extract",
        verbatim_doc_comment
    )]
    level10: Option<u8>,

    /// Extract the JXL codestream from a container file
    #[clap(long, action, conflicts_with_all = ["pack", "level10"])]
    extract: bool,
}

fn apply_file_format_options(
    input_bytes: &[u8],
    opt: &Opt,
    signature: JxlSignature,
) -> Result<Vec<u8>> {
    if opt.extract {
        ensure!(
            signature == JxlSignature::Container,
            "Input file is not a container file."
        );
        let (codestream, level) = extract_codestream(input_bytes)?;
        if level == Some(ProfileLevel::Main10) {
            eprintln!("Warning: extracting raw codestream that is greater than level 5");
        }
        Ok(codestream)
    } else if opt.pack || opt.level10.is_some() {
        ensure!(
            signature == JxlSignature::Codestream,
            "Input file is not a codestream file"
        );
        Ok(pack_codestream(input_bytes, opt.level10.unwrap_or(3))?)
    } else {
        Ok(input_bytes.to_vec())
    }
}

fn main() -> Result<()> {
    jxl_cli::init_tracing();

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

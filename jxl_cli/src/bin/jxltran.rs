// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::fs;
use std::path::PathBuf;

use clap::Parser;
use color_eyre::eyre::{Result, WrapErr, bail, ensure};
use jxl::api::{JxlSignature, ProfileLevel, check_signature, extract_codestream, pack_codestream};

const VERSION_STRING: &str = concat!(
    env!("VERGEN_GIT_DESCRIBE"),
    " (rustc ",
    env!("VERGEN_RUSTC_SEMVER"),
    ")"
);

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
    } else if opt.pack || opt.level10 {
        ensure!(
            signature == JxlSignature::Codestream,
            "Input file is not a codestream file"
        );
        Ok(pack_codestream(input_bytes, opt.level10)?)
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

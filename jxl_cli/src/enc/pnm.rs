// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use color_eyre::eyre::{Result, ensure};
use jxl::api::JxlColorType;
use std::io::Write;

use crate::dec::{DecodeOutput, OutputDataType};

pub fn to_pgm<Writer: Write>(img: &DecodeOutput, writer: &mut Writer) -> Result<()> {
    assert_eq!(img.data_type, OutputDataType::U8);
    ensure!(
        img.frames[0].color_type == JxlColorType::Grayscale,
        "Writing to PPM only supports Grayscale"
    );
    if img.frames.len() > 1 {
        eprintln!("Warning: More than one frame found, saving just the first one.");
    }
    if img.frames[0].channels.len() > 1 {
        eprintln!("Warning: Ignoring extra channels.");
    }
    write!(writer, "P5\n{} {}\n255\n", img.size.0, img.size.1)?;
    for y in 0..img.size.1 {
        writer.write_all(img.frames[0].channels[0].row(y))?;
    }
    Ok(())
}

pub fn to_ppm<Writer: Write>(img: &DecodeOutput, writer: &mut Writer) -> Result<()> {
    assert_eq!(img.data_type, OutputDataType::U8);
    ensure!(
        img.frames[0].color_type == JxlColorType::Rgb,
        "Writing to PPM only supports RGB"
    );
    if img.frames.len() > 1 {
        eprintln!("Warning: More than one frame found, saving just the first one.");
    }
    if img.frames[0].channels.len() > 1 {
        eprintln!("Warning: Ignoring extra channels.");
    }
    write!(writer, "P6\n{} {}\n255\n", img.size.0, img.size.1)?;
    for y in 0..img.size.1 {
        writer.write_all(img.frames[0].channels[0].row(y))?;
    }
    Ok(())
}

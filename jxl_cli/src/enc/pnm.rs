// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use color_eyre::eyre::{Result, ensure};
use jxl::api::JxlColorType;
use std::io::Write;

use crate::dec::{DecodeOutput, OutputDataType};

pub fn to_pgm<Writer: Write>(img: &DecodeOutput, writer: &mut Writer) -> Result<()> {
    assert!(
        img.data_type == OutputDataType::U8 || img.data_type == OutputDataType::U16,
        "PGM output requires U8 or U16 data type"
    );
    ensure!(
        img.frames[0].color_type == JxlColorType::Grayscale,
        "Writing to PGM only supports Grayscale"
    );
    if img.frames.len() > 1 {
        eprintln!("Warning: More than one frame found, saving just the first one.");
    }
    if img.frames[0].channels.len() > 1 {
        eprintln!("Warning: Ignoring extra channels.");
    }
    let maxval = if img.data_type == OutputDataType::U16 {
        65535
    } else {
        255
    };
    write!(writer, "P5\n{} {}\n{maxval}\n", img.size.0, img.size.1)?;
    for y in 0..img.size.1 {
        let row = img.frames[0].channels[0].row(y);
        if img.data_type == OutputDataType::U16 {
            // PPM spec requires big-endian for 16-bit values
            write_row_u16_be(writer, row)?;
        } else {
            writer.write_all(row)?;
        }
    }
    Ok(())
}

pub fn to_ppm<Writer: Write>(img: &DecodeOutput, writer: &mut Writer) -> Result<()> {
    assert!(
        img.data_type == OutputDataType::U8 || img.data_type == OutputDataType::U16,
        "PPM output requires U8 or U16 data type"
    );
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
    let maxval = if img.data_type == OutputDataType::U16 {
        65535
    } else {
        255
    };
    write!(writer, "P6\n{} {}\n{maxval}\n", img.size.0, img.size.1)?;
    for y in 0..img.size.1 {
        let row = img.frames[0].channels[0].row(y);
        if img.data_type == OutputDataType::U16 {
            // PPM spec requires big-endian for 16-bit values
            write_row_u16_be(writer, row)?;
        } else {
            writer.write_all(row)?;
        }
    }
    Ok(())
}

/// Write a row of native-endian U16 pixel data as big-endian bytes (PPM/PGM spec requirement).
fn write_row_u16_be<Writer: Write>(writer: &mut Writer, row: &[u8]) -> Result<()> {
    // Process in chunks for efficiency
    let mut buf = vec![0u8; row.len()];
    for i in (0..row.len()).step_by(2) {
        let val = u16::from_ne_bytes([row[i], row[i + 1]]);
        let be = val.to_be_bytes();
        buf[i] = be[0];
        buf[i + 1] = be[1];
    }
    writer.write_all(&buf)?;
    Ok(())
}

// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

use color_eyre::eyre::{Result, eyre};

use crate::dec::{DecodeOutput, OutputDataType};

#[cfg(feature = "exr")]
pub mod exr;
pub mod numpy;
pub mod png;
pub mod pnm;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Ppm,
    Pgm,
    Npy,
    Png,
    #[cfg(feature = "exr")]
    Exr,
}

impl OutputFormat {
    pub fn from_output_filename(filename: &str) -> Result<Self> {
        #[cfg(feature = "exr")]
        if filename.ends_with(".exr") {
            return Ok(OutputFormat::Exr);
        }
        if filename.ends_with(".ppm") {
            return Ok(OutputFormat::Ppm);
        }
        if filename.ends_with(".pgm") {
            return Ok(OutputFormat::Pgm);
        }
        if filename.ends_with(".npy") {
            return Ok(OutputFormat::Npy);
        }
        if filename.ends_with(".png") || filename.ends_with(".apng") {
            return Ok(OutputFormat::Png);
        }
        Err(eyre!("Output format not supported for {:?}", filename))
    }

    pub fn supported_output_data_types(&self) -> &'static [OutputDataType] {
        match self {
            Self::Ppm | Self::Pgm => &[OutputDataType::U8, OutputDataType::U16],
            Self::Npy => &[OutputDataType::F32],
            Self::Png => &[OutputDataType::U8, OutputDataType::U16],
            #[cfg(feature = "exr")]
            Self::Exr => &[OutputDataType::F16, OutputDataType::F32],
        }
    }

    pub fn should_fold_alpha(&self) -> bool {
        match self {
            Self::Ppm | Self::Pgm | Self::Npy => false,
            Self::Png => true,
            #[cfg(feature = "exr")]
            Self::Exr => true,
        }
    }

    pub fn accepts_cmyk(&self) -> bool {
        match self {
            Self::Ppm | Self::Pgm | Self::Png => false,
            Self::Npy => true,
            #[cfg(feature = "exr")]
            Self::Exr => false,
        }
    }

    pub fn save_image(&self, image_data: &DecodeOutput, output_filename: &PathBuf) -> Result<()> {
        let has_partial_renders = image_data
            .frames
            .iter()
            .any(|x| !x.partial_renders.is_empty());
        if has_partial_renders {
            if image_data.frames.len() != 1 {
                eprintln!("Warning: Ignoring partial renders in animations.");
            } else if *self != Self::Png {
                eprintln!("Warning: Ignoring partial renders with non-PNG output.");
            } else {
                let frame = &image_data.frames[0];
                let dir = output_filename.parent().unwrap_or(std::path::Path::new(""));
                let stem = output_filename
                    .file_stem()
                    .map(|s| s.to_string_lossy())
                    .unwrap_or_else(|| "output".into());
                for (i, partial) in frame.partial_renders.iter().enumerate() {
                    let fname = dir.join(format!("{stem}.partial_{:012}.png", partial.byte_index));
                    let mut writer = BufWriter::new(File::create(fname)?);
                    png::to_png(image_data, &mut writer, Some(i))?;
                }
                let fname = dir.join(format!("{stem}.partial_{:012}.png", frame.total_bytes));
                let mut writer = BufWriter::new(File::create(fname)?);
                png::to_png(image_data, &mut writer, None)?;
            }
        }
        let mut writer = BufWriter::new(File::create(output_filename)?);
        match self {
            Self::Ppm => pnm::to_ppm(image_data, &mut writer)?,
            Self::Pgm => pnm::to_pgm(image_data, &mut writer)?,
            Self::Npy => numpy::to_numpy(image_data, &mut writer)?,
            Self::Png => png::to_png(image_data, &mut writer, None)?,
            #[cfg(feature = "exr")]
            Self::Exr => exr::to_exr(image_data, &mut writer)?,
        };
        Ok(())
    }
}

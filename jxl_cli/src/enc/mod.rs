// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::{fs::File, io::BufWriter, path::PathBuf};

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
            Self::Ppm | Self::Pgm => &[OutputDataType::U8],
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
                let num_partials = image_data.frames[0].partial_renders.len();
                for i in 0..=num_partials {
                    let dir = output_filename.parent().unwrap();
                    let stem = output_filename.file_stem().unwrap().to_string_lossy();
                    let fname = dir.join(format!("{stem}.partial{i:05}.png"));
                    let mut writer = BufWriter::new(File::create(fname)?);
                    png::to_png(
                        image_data,
                        &mut writer,
                        if i < num_partials { Some(i) } else { None },
                    )?
                }
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

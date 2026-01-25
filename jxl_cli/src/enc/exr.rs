// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::io::{Seek, Write};

use color_eyre::eyre::{Result, eyre};
use jxl::api::{JxlColorEncoding, JxlColorProfile, JxlTransferFunction};

use exr::meta::attribute::Chromaticities;
use exr::prelude::*;

use crate::dec::{DecodeOutput, OutputDataType};

pub fn to_exr<Writer: Write + Seek>(image_data: &DecodeOutput, writer: &mut Writer) -> Result<()> {
    let tuple_to_vec2 = |(x, y)| Vec2(x, y);
    let chromaticities = match &image_data.output_profile {
        JxlColorProfile::Simple(JxlColorEncoding::RgbColorSpace {
            white_point,
            primaries,
            transfer_function: JxlTransferFunction::Linear,
            ..
        }) => {
            let [r, g, b] = primaries.to_xy_coords();
            Some(Chromaticities {
                red: tuple_to_vec2(r),
                green: tuple_to_vec2(g),
                blue: tuple_to_vec2(b),
                white: tuple_to_vec2(white_point.to_xy_coords()),
            })
        }
        JxlColorProfile::Simple(JxlColorEncoding::GrayscaleColorSpace {
            white_point,
            transfer_function: JxlTransferFunction::Linear,
            ..
        }) => Some(Chromaticities {
            red: Vec2(0.64, 0.33),
            green: Vec2(0.3, 0.6),
            blue: Vec2(0.15, 0.06),
            white: tuple_to_vec2(white_point.to_xy_coords()),
        }),
        JxlColorProfile::Icc(_) => {
            return Err(eyre!("EXR requires a linear colorspace (got ICC profile)"));
        }
        JxlColorProfile::Simple(encoding) => {
            return Err(eyre!(
                "Writing of images in colorspace {:?} not yet implemented for EXR output",
                encoding.get_color_encoding_description()
            ));
        }
    };

    if image_data.frames.len() > 1 {
        eprintln!("Warning: More than one frame found, saving just the first one.");
    }
    if image_data.frames[0].channels.len() > 1 {
        eprintln!("Warning: Ignoring extra channels.");
    }

    let (width, height) = image_data.size;
    let num_channels = image_data.frames[0].color_type.samples_per_pixel();

    let mut channels = SmallVec::<[AnyChannel<FlatSamples>; 4]>::new();
    let channel_names = match num_channels {
        1 => vec!["Y"],
        2 => vec!["Y", "A"],
        3 => vec!["R", "G", "B"],
        4 => vec!["R", "G", "B", "A"],
        _ => {
            unreachable!()
        }
    };

    // TODO(sboukortt): convert unassociated alpha to associated if necessary
    let buf = &image_data.frames[0].channels[0];
    for (c, name) in channel_names.iter().copied().enumerate() {
        let sample_data = if image_data.data_type == OutputDataType::F32 {
            FlatSamples::F32(
                (0..height)
                    .flat_map(|y| {
                        buf.row(y)
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .skip(c)
                            .step_by(num_channels)
                            .copied()
                            .map(f32::from_ne_bytes)
                    })
                    .collect(),
            )
        } else {
            FlatSamples::F16(
                (0..height)
                    .flat_map(|y| {
                        buf.row(y)
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .skip(c)
                            .step_by(num_channels)
                            .copied()
                            .map(f16::from_ne_bytes)
                    })
                    .collect(),
            )
        };
        channels.push(AnyChannel::<FlatSamples> {
            name: name.into(),
            sample_data,
            quantize_linearly: name == "A",
            sampling: Vec2(1, 1),
        });
    }
    let channels = AnyChannels::sort(channels);
    let mut image = Image::from_channels((width, height), channels);
    image.attributes.chromaticities = chromaticities;
    // TODO(sboukortt): intensity_target -> whiteLuminance

    image.write().to_buffered(writer)?;
    Ok(())
}

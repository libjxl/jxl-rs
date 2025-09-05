// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

pub use jxl_exr::to_exr;

#[cfg(not(feature = "exr"))]
mod jxl_exr {

    use crate::ImageData;
    use jxl::api::JxlColorProfile;
    use jxl::error::{Error, Result};
    use jxl::headers::bit_depth::BitDepth;

    pub fn to_exr(
        _image_data: ImageData<f32>,
        _bit_depth: BitDepth,
        _color_profile: &JxlColorProfile,
    ) -> Result<Vec<u8>> {
        return Err(Error::OutputFormatNotSupported);
    }
}

#[cfg(feature = "exr")]
mod jxl_exr {

    use std::io::Cursor;

    use color_eyre::eyre::{Result, WrapErr, eyre};
    use jxl::api::{JxlColorEncoding, JxlColorProfile, JxlTransferFunction};
    use jxl::error::Error;

    use exr::meta::attribute::Chromaticities;
    use exr::prelude::*;

    use crate::ImageData;

    pub fn to_exr(
        image_data: ImageData<f32>,
        bit_depth: u32,
        color_profile: &JxlColorProfile,
    ) -> Result<Vec<u8>> {
        let tuple_to_vec2 = |(x, y)| Vec2(x, y);
        let chromaticities = match color_profile {
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
                    "Writing of {:?} channels not yet implemented for EXR output",
                    encoding.get_color_encoding_description()
                ));
            }
        };

        if image_data.frames.is_empty()
            || image_data.frames[0].channels.is_empty()
            || image_data.size.0 == 0
            || image_data.size.1 == 0
        {
            return Err(Error::NoFrames).wrap_err("Invalid JXL image");
        }
        let size = image_data.size;
        let (width, height) = size;
        let num_channels = image_data.frames[0].channels.len();

        for (i, frame) in image_data.frames.iter().enumerate() {
            assert_eq!(frame.size, size, "Frame {i} size mismatch");
            assert_eq!(
                frame.channels.len(),
                num_channels,
                "Frame {i} num channels mismatch"
            );
            for (c, channel) in frame.channels.iter().enumerate() {
                assert_eq!(channel.size(), size, "Frame {i} channel {c} size mismatch");
            }
        }

        let mut channels = SmallVec::<[AnyChannel<FlatSamples>; 4]>::new();
        let channel_names = match num_channels {
            1 => vec!["Y"],
            2 => vec!["Y", "A"],
            3 => vec!["R", "G", "B"],
            4 => vec!["R", "G", "B", "A"],
            _ => {
                return Err(eyre!(
                    "Writing of {:?} channels not yet implemented for EXR output",
                    num_channels
                ));
            }
        };
        // TODO(sboukortt): convert unassociated alpha to associated if necessary
        for (channel, channel_name) in image_data.frames[0].channels.iter().zip(channel_names) {
            let sample_data = if bit_depth <= 16 {
                let mut samples = vec![f16::ZERO; height * width];
                for y in 0..height {
                    for x in 0..width {
                        samples[y * width + x] = f16::from_f32(channel.as_rect().row(y)[x] / 255.0);
                    }
                }
                FlatSamples::F16(samples)
            } else {
                let mut samples = vec![0.0; height * width];
                for y in 0..height {
                    for x in 0..width {
                        samples[y * width + x] = channel.as_rect().row(y)[x] / 255.0;
                    }
                }
                FlatSamples::F32(samples)
            };
            channels.push(AnyChannel::<FlatSamples> {
                name: channel_name.into(),
                sample_data,
                quantize_linearly: channel_name == "A",
                sampling: Vec2(1, 1),
            });
        }
        let channels = AnyChannels::sort(channels);
        let mut image = Image::from_channels((width, height), channels);
        image.attributes.chromaticities = chromaticities;
        // TODO(sboukortt): intensity_target -> whiteLuminance

        let mut buf = Cursor::new(Vec::new());
        image.write().to_buffered(&mut buf)?;
        Ok(buf.into_inner())
    }
}

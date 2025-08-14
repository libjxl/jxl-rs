// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl::api::{
    JxlColorEncoding, JxlColorProfile, JxlPrimaries, JxlTransferFunction, JxlWhitePoint,
};
use jxl::decode::ImageData;
use jxl::error::{Error, Result};
use jxl::headers::color_encoding::RenderingIntent;

use std::borrow::Cow;
use std::io::BufWriter;

fn png_color(num_channels: usize) -> Result<png::ColorType> {
    match num_channels {
        1 => Ok(png::ColorType::Grayscale),
        2 => Ok(png::ColorType::GrayscaleAlpha),
        3 => Ok(png::ColorType::Rgb),
        4 => Ok(png::ColorType::Rgba),
        _ => Err(Error::PNGInvalidNumChannels(num_channels)),
    }
}

fn make_cicp(encoding: &JxlColorEncoding) -> Option<png::CodingIndependentCodePoints> {
    let JxlColorEncoding::RgbColorSpace {
        white_point,
        primaries,
        transfer_function,
        ..
    } = encoding
    else {
        return None;
    };

    Some(png::CodingIndependentCodePoints {
        color_primaries: match white_point {
            JxlWhitePoint::DCI => {
                if *primaries == JxlPrimaries::P3 {
                    11
                } else {
                    return None;
                }
            }
            JxlWhitePoint::D65 => match primaries {
                JxlPrimaries::SRGB => 1,
                JxlPrimaries::BT2100 => 9,
                JxlPrimaries::P3 => 12,
                JxlPrimaries::Chromaticities { .. } => return None,
            },
            _ => return None,
        },
        transfer_function: match transfer_function {
            JxlTransferFunction::BT709 => 1,
            JxlTransferFunction::Linear => 8,
            JxlTransferFunction::SRGB => 13,
            JxlTransferFunction::PQ => 16,
            JxlTransferFunction::DCI => 17,
            JxlTransferFunction::HLG => 18,
            JxlTransferFunction::Gamma(_) => return None,
        },
        matrix_coefficients: 0,
        is_video_full_range_image: true,
    })
}

fn encode_png(
    image_data: ImageData<f32>,
    bit_depth: u32,
    color_profile: &JxlColorProfile,
    buf: &mut Vec<u8>,
) -> Result<()> {
    if image_data.frames.is_empty()
        || image_data.frames[0].channels.is_empty()
        || image_data.size.0 == 0
        || image_data.size.1 == 0
    {
        return Err(Error::NoFrames);
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

    let w = BufWriter::new(buf);
    let mut info = png::Info::with_size(width as u32, height as u32);
    match color_profile {
        JxlColorProfile::Simple(JxlColorEncoding::RgbColorSpace {
            white_point: JxlWhitePoint::D65,
            primaries: JxlPrimaries::SRGB,
            transfer_function: JxlTransferFunction::SRGB,
            rendering_intent,
        }) => {
            info.srgb = Some(match rendering_intent {
                RenderingIntent::Absolute => png::SrgbRenderingIntent::AbsoluteColorimetric,
                RenderingIntent::Relative => png::SrgbRenderingIntent::RelativeColorimetric,
                RenderingIntent::Perceptual => png::SrgbRenderingIntent::Perceptual,
                RenderingIntent::Saturation => png::SrgbRenderingIntent::Saturation,
            });
            info.source_gamma = Some(png::ScaledFloat::from_scaled(45455));
            info.source_chromaticities = Some(png::SourceChromaticities {
                white: (
                    png::ScaledFloat::from_scaled(31270),
                    png::ScaledFloat::from_scaled(32900),
                ),
                red: (
                    png::ScaledFloat::from_scaled(64000),
                    png::ScaledFloat::from_scaled(33000),
                ),
                green: (
                    png::ScaledFloat::from_scaled(30000),
                    png::ScaledFloat::from_scaled(60000),
                ),
                blue: (
                    png::ScaledFloat::from_scaled(15000),
                    png::ScaledFloat::from_scaled(6000),
                ),
            });
        }
        JxlColorProfile::Simple(encoding) => {
            info.coding_independent_code_points = make_cicp(encoding);
            let icc_bytes = encoding.maybe_create_profile()?.unwrap();
            info.icc_profile = Some(Cow::from(icc_bytes));
        }
        JxlColorProfile::Icc(icc_bytes) => {
            info.icc_profile = Some(Cow::Borrowed(icc_bytes));
        }
    }
    let mut encoder = png::Encoder::with_info(w, info).unwrap();
    encoder.set_color(png_color(num_channels)?);
    let eight_bits = bit_depth <= 8;
    encoder.set_depth(if eight_bits {
        png::BitDepth::Eight
    } else {
        png::BitDepth::Sixteen
    });
    if image_data.frames.len() > 1 {
        encoder
            .set_animated(image_data.frames.len() as u32, 0)
            .unwrap();
    }
    let mut writer = encoder.write_header().unwrap();
    let num_pixels = height * width * num_channels;
    if eight_bits {
        let mut data: Vec<u8> = vec![0; num_pixels];
        for frame in image_data.frames {
            for y in 0..height {
                for x in 0..width {
                    for c in 0..num_channels {
                        data[(y * width + x) * num_channels + c] =
                            frame.channels[c].as_rect().row(y)[x].clamp(0.0, 255.0) as u8;
                    }
                }
            }
            writer.write_image_data(&data).unwrap();
        }
    } else {
        let mut data: Vec<u8> = vec![0; 2 * num_pixels];
        for frame in image_data.frames {
            for y in 0..height {
                for x in 0..width {
                    for c in 0..num_channels {
                        let pixel = (frame.channels[c].as_rect().row(y)[x] * 65535.0 / 255.0)
                            .clamp(0.0, 65535.0) as u16;
                        let index = 2 * ((y * width + x) * num_channels + c);
                        data[index] = (pixel >> 8) as u8;
                        data[index + 1] = (pixel & 0xFF) as u8;
                    }
                }
            }
            writer.write_image_data(&data).unwrap();
        }
    }
    Ok(())
}

pub fn to_png(
    image_data: ImageData<f32>,
    bit_depth: u32,
    color_profile: &JxlColorProfile,
) -> Result<Vec<u8>> {
    let mut buf = Vec::<u8>::new();
    encode_png(image_data, bit_depth, color_profile, &mut buf)?;
    Ok(buf)
}

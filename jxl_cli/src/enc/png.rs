// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl::api::{
    JxlColorEncoding, JxlColorProfile, JxlPrimaries, JxlTransferFunction, JxlWhitePoint,
};

use crate::DecodeOutput;
use color_eyre::eyre::{Result, WrapErr, eyre};
use jxl::error::Error;
use jxl::headers::color_encoding::RenderingIntent;

use std::borrow::Cow;
use std::io::Write;

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn calculate_apng_delay(duration_ms: f64) -> Result<(u16, u16)> {
    if duration_ms < 0.0 {
        return Err(eyre!("Negative frame duration: {}", duration_ms));
    }
    if duration_ms == 0.0 {
        return Ok((0, 1));
    }

    let mut num = duration_ms.round() as u64;
    let mut den = 1000u64;

    let common = gcd(num, den);
    num /= common;
    den /= common;

    if num > u16::MAX as u64 || den > u16::MAX as u64 {
        Err(eyre!(
            "APNG frame delay overflow after GCD: {}/{}",
            num,
            den
        ))
    } else {
        Ok((num as u16, den as u16))
    }
}

fn png_color(num_channels: usize) -> Result<png::ColorType> {
    match num_channels {
        1 => Ok(png::ColorType::Grayscale),
        2 => Ok(png::ColorType::GrayscaleAlpha),
        3 => Ok(png::ColorType::Rgb),
        4 => Ok(png::ColorType::Rgba),
        _ => Err(eyre!(
            "Invalid number of channels for PNG output {:?}",
            num_channels
        )),
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

pub fn to_png<Writer: Write>(
    image_data: &DecodeOutput<f32>,
    bit_depth: u32,
    buf: &mut Writer,
) -> Result<()> {
    if image_data.frames.is_empty()
        || image_data.frames[0].channels.is_empty()
        || image_data.size.0 == 0
        || image_data.size.1 == 0
    {
        return Err(Error::NoFrames).wrap_err("Invalid JXL image");
    }
    let (width, height) = image_data.size;
    let num_channels = image_data.frames[0].channels.len();

    for (i, frame) in image_data.frames.iter().enumerate() {
        assert_eq!(
            frame.channels.len(),
            num_channels,
            "Frame {i} num channels mismatch"
        );
        for (c, channel) in frame.channels.iter().enumerate() {
            assert_eq!(
                channel.size(),
                image_data.size,
                "Frame {i} channel {c} size mismatch"
            );
        }
    }

    let mut info = png::Info::with_size(width as u32, height as u32);
    match &image_data.output_profile {
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
    let mut encoder = png::Encoder::with_info(buf, info).unwrap();
    encoder.set_color(png_color(num_channels)?);
    let eight_bits = bit_depth <= 8;
    encoder.set_depth(if eight_bits {
        png::BitDepth::Eight
    } else {
        png::BitDepth::Sixteen
    });
    if let Some(anim) = &image_data.jxl_animation {
        if image_data.frames.len() > 1 {
            encoder.set_animated(image_data.frames.len() as u32, anim.num_loops)?;
        }
    } else if image_data.frames.len() > 1 {
        encoder.set_animated(image_data.frames.len() as u32, 0)?;
    }
    let mut writer = encoder.write_header()?;

    let num_pixels = height * width * num_channels;
    if eight_bits {
        let mut data: Vec<u8> = vec![0; num_pixels];
        for (index, frame) in image_data.frames.iter().enumerate() {
            for y in 0..height {
                for x in 0..width {
                    for c in 0..num_channels {
                        // + 0.5 instead of round is fine since we clamp to non-negative
                        data[(y * width + x) * num_channels + c] =
                            ((frame.channels[c].as_rect().row(y)[x] * 255.0).clamp(0.0, 255.0)
                                + 0.5) as u8;
                    }
                }
            }
            writer.write_image_data(&data)?;
            if index + 1 < image_data.frames.len() && image_data.frames.len() > 1 {
                let (delay_num, delay_den) = calculate_apng_delay(frame.duration)?;
                writer.set_frame_delay(delay_num, delay_den)?;
            }
        }
    } else {
        // 16-bit
        let mut data: Vec<u8> = vec![0; 2 * num_pixels];
        for (index, frame) in image_data.frames.iter().enumerate() {
            for y in 0..height {
                for x in 0..width {
                    for c in 0..num_channels {
                        // + 0.5 instead of round is fine since we clamp to non-negative
                        let pixel = ((frame.channels[c].as_rect().row(y)[x] * 65535.0)
                            .clamp(0.0, 65535.0)
                            + 0.5) as u16;
                        let index = 2 * ((y * width + x) * num_channels + c);
                        data[index] = (pixel >> 8) as u8;
                        data[index + 1] = (pixel & 0xFF) as u8;
                    }
                }
            }
            writer.write_image_data(&data)?;
            if index + 1 < image_data.frames.len() && image_data.frames.len() > 1 {
                let (delay_num, delay_den) = calculate_apng_delay(frame.duration)?;
                writer.set_frame_delay(delay_num, delay_den)?;
            }
        }
    }
    Ok(())
}

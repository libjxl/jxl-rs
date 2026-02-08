// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl::api::{
    JxlColorEncoding, JxlColorProfile, JxlPrimaries, JxlTransferFunction, JxlWhitePoint,
};

use crate::dec::{DecodeOutput, OutputDataType};
use color_eyre::eyre::{Result, eyre};
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
    image_data: &DecodeOutput,
    buf: &mut Writer,
    partial_render: Option<usize>,
) -> Result<()> {
    if image_data.frames[0].channels.len() > 1 {
        eprintln!("Warning: Ignoring non-alpha extra channels.");
    }

    let (width, height) = image_data.size;
    let num_channels = image_data.frames[0].color_type.samples_per_pixel();

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
    encoder.set_compression(png::Compression::Fast);
    let eight_bits = image_data.data_type == OutputDataType::U8;
    encoder.set_depth(if eight_bits {
        png::BitDepth::Eight
    } else {
        png::BitDepth::Sixteen
    });
    let animated = image_data.frames.len() > 1;
    if animated {
        let loops = image_data
            .jxl_animation
            .as_ref()
            .map_or(0, |anim| anim.num_loops);
        encoder.set_animated(image_data.frames.len() as u32, loops)?;
    }
    let mut writer = encoder.write_header()?;

    if eight_bits {
        for frame in &image_data.frames {
            if animated {
                let (delay_num, delay_den) = calculate_apng_delay(frame.duration)?;
                writer.set_frame_delay(delay_num, delay_den)?;
            }
            let mut ww = writer.stream_writer()?;
            let chan = if let Some(p) = partial_render {
                &frame.partial_renders[p][0]
            } else {
                &frame.channels[0]
            };
            for y in 0..height {
                ww.write_all(chan.row(y))?;
            }
        }
    } else {
        // 16-bit
        let mut buffer: Vec<u8> = vec![0; 2 * width * num_channels];
        for frame in &image_data.frames {
            if animated {
                let (delay_num, delay_den) = calculate_apng_delay(frame.duration)?;
                writer.set_frame_delay(delay_num, delay_den)?;
            }
            let mut ww = writer.stream_writer()?;
            let chan = if let Some(p) = partial_render {
                &frame.partial_renders[p][0]
            } else {
                &frame.channels[0]
            };
            for y in 0..height {
                let row = chan.row(y);
                if cfg!(target_endian = "big") {
                    ww.write_all(row)?;
                } else {
                    for x in 0..width * num_channels {
                        buffer[x * 2..][..2].copy_from_slice(
                            &u16::from_ne_bytes(row[x * 2..][..2].try_into().unwrap())
                                .to_be_bytes(),
                        );
                    }
                    ww.write_all(&buffer)?;
                }
            }
        }
    }
    Ok(())
}

// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl::decode::ImageData;
use jxl::error::{Error, Result};
use jxl::headers::bit_depth::BitDepth;

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

fn encode_png(
    image_data: ImageData<f32>,
    bit_depth: BitDepth,
    icc_bytes: Option<&[u8]>,
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
    if let Some(icc_bytes) = icc_bytes {
        info.icc_profile = Some(Cow::from(icc_bytes));
    } else {
        info.srgb = Some(png::SrgbRenderingIntent::RelativeColorimetric);
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
    let mut encoder = png::Encoder::with_info(w, info).unwrap();
    encoder.set_color(png_color(num_channels)?);
    let eight_bits = bit_depth.bits_per_sample() <= 8;
    encoder.set_depth(if eight_bits {
        png::BitDepth::Eight
    } else {
        png::BitDepth::Sixteen
    });
    if image_data.frames.len() > 1 {
        // TODO(szabadka): Handle error.
        let _ = encoder.set_animated(image_data.frames.len() as u32, 0);
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
    bit_depth: BitDepth,
    icc_bytes: Option<&[u8]>,
) -> Result<Vec<u8>> {
    let mut buf = Vec::<u8>::new();
    encode_png(image_data, bit_depth, icc_bytes, &mut buf)?;
    Ok(buf)
}

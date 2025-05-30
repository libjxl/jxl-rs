// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::enc::ImageData;
use crate::error::{Error, Result};

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

fn encode_png(image_data: ImageData<f32>, buf: &mut Vec<u8>) -> Result<()> {
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

    for frame in &image_data.frames {
        assert_eq!(frame.size, size);
        assert_eq!(frame.channels.len(), num_channels);
        for channel in &frame.channels {
            assert_eq!(channel.size(), size);
        }
    }

    let w = BufWriter::new(buf);
    let mut encoder = png::Encoder::new(w, width as u32, height as u32);
    encoder.set_color(png_color(num_channels)?);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();
    let num_pixels = height * width * num_channels;
    let mut data: Vec<u8> = vec![0; num_pixels];
    for y in 0..height {
        for x in 0..width {
            for c in 0..num_channels {
                data[(y * width + x) * num_channels + c] =
                    image_data.frames[0].channels[c].as_rect().row(y)[x].clamp(0.0, 255.0) as u8;
            }
        }
    }
    writer.write_image_data(&data).unwrap();
    Ok(())
}

pub fn to_png(image_data: ImageData<f32>) -> Result<Vec<u8>> {
    let mut buf = Vec::<u8>::new();
    encode_png(image_data, &mut buf)?;
    Ok(buf)
}

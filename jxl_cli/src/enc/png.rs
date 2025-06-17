// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl::decode::ImageData;
use jxl::error::{Error, Result};

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
    let mut encoder = png::Encoder::new(w, width as u32, height as u32);
    encoder.set_color(png_color(num_channels)?);
    encoder.set_depth(png::BitDepth::Eight);
    if image_data.frames.len() > 1 {
        // TODO(szabadka): Handle error.
        let _ = encoder.set_animated(image_data.frames.len() as u32, 0);
    }
    let mut writer = encoder.write_header().unwrap();
    let num_pixels = height * width * num_channels;
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
    Ok(())
}

pub fn to_png(image_data: ImageData<f32>) -> Result<Vec<u8>> {
    let mut buf = Vec::<u8>::new();
    encode_png(image_data, &mut buf)?;
    Ok(buf)
}

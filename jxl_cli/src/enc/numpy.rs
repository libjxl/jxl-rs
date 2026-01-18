// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::dec::DecodeOutput;
use jxl::error::{Error, Result};
use std::io::Write;

fn numpy_header<Writer: Write>(
    xsize: usize,
    ysize: usize,
    num_channels: usize,
    num_frames: usize,
    writer: &mut Writer,
) -> Result<()> {
    // The magic string and version for .npy files (Version 1.0)
    let magic_string: [u8; 8] = [0x93, b'N', b'U', b'M', b'P', b'Y', 0x01, 0x00];

    // Construct the header dictionary string.
    // Note the trailing comma in the tuple and the space before the closing brace, and the newline.
    //
    // The dtype '<f4' signifies little-endian 32-bit float.
    let mut header_dict_str = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': \
	 ({num_frames}, {ysize}, {xsize}, {num_channels}), }}"
    );
    // https://github.com/numpy/numpy/blob/main/doc/neps/nep-0001-npy-format.rst:
    // "terminated by a newline ('n') and padded with spaces ('x20') to make the total length of the magic string + 4 + HEADER_LEN be evenly divisible by 16 for alignment purposes"
    // The 4 is a 2 since the major and minor versions are included in the magic string. The extra 1 at the end is for the newline.
    header_dict_str.push_str(
        (0..(16 - ((magic_string.len() + 2 + header_dict_str.len() + 1) % 16) % 16))
            .map(|_| " ")
            .collect::<String>()
            .as_str(),
    );
    header_dict_str.push('\n');
    let header_dict_len = header_dict_str.len();
    assert!(
        header_dict_len <= u16::MAX as usize,
        "header_dict_len ({}) exceeds u16::MAX ({})",
        header_dict_len,
        u16::MAX
    );
    let header_len_bytes = (header_dict_len as u16).to_le_bytes();

    // Assemble the full header.
    writer.write_all(&magic_string)?;
    writer.write_all(&header_len_bytes)?;
    writer.write_all(header_dict_str.as_bytes())?;

    Ok(())
}

fn numpy_bytes<Writer: Write>(
    image_data: &DecodeOutput<f32>,
    num_channels: usize,
    writer: &mut Writer,
) -> Result<()> {
    let (width, height) = image_data.size;

    for frame in &image_data.frames {
        assert_eq!(frame.channels.len(), num_channels);
        for channel in &frame.channels {
            assert_eq!(channel.size(), image_data.size);
        }

        for y in 0..height {
            for x in 0..width {
                for channel in frame.channels.iter() {
                    writer.write_all(&channel.row(y)[x].to_le_bytes())?;
                }
            }
        }
    }
    Ok(())
}

/// Converts image_data to a Vec<u8> in .npy format.
/// The data will be represented as little-endian 32-bit floats ('<f4').
/// The shape of the NumPy array will be (num_frames, height, width, num_channels).
///
pub fn to_numpy<Writer: Write>(image_data: &DecodeOutput<f32>, writer: &mut Writer) -> Result<()> {
    if image_data.frames.is_empty()
        || image_data.frames[0].channels.is_empty()
        || image_data.size.0 == 0
        || image_data.size.1 == 0
    {
        return Err(Error::NoFrames);
    }
    let size = image_data.size;
    let (width, height) = size;
    let num_frames = image_data.frames.len();
    let num_channels = image_data.frames[0].channels.len();

    numpy_header(width, height, num_channels, num_frames, writer)?;
    // Consistent channel sizes are checked inside the call to `to_numpy_bytes`.
    numpy_bytes(image_data, num_channels, writer)?;

    Ok(())
}

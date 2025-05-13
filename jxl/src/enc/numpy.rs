// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#![cfg(feature = "debug_tools")]

use crate::image::ImageRect;

fn numpy_header(xsize: usize, ysize: usize, num_channels: usize, num_frames: usize) -> Vec<u8> {
    // The magic string and version for .npy files (Version 1.0)
    let magic_string: [u8; 8] = [0x93, b'N', b'U', b'M', b'P', b'Y', 0x01, 0x00];

    // Construct the header dictionary string.
    // Note the trailing comma in the tuple and the space before the closing brace, and the newline.
    //
    // TODO(firsching): shouldn't this be padded by "spaces ('x20') to make the total length of the magic
    // string + 4 + HEADER_LEN be evenly divisible by 16"? see
    // https://github.com/numpy/numpy/blob/main/doc/neps/nep-0001-npy-format.rst
    //
    // The dtype '<f4' signifies little-endian 32-bit float.
    let header_dict_str = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}, {}, {}), }}\n",
        num_frames, ysize, xsize, num_channels
    );
    let header_dict_len = header_dict_str.len();
    assert!(
        header_dict_len <= u16::MAX as usize,
        "header_dict_len ({}) exceeds u16::MAX ({})",
        header_dict_len,
        u16::MAX
    );
    let header_len_bytes = (header_dict_len as u16).to_le_bytes();

    // Assemble the full header.
    let mut header: Vec<u8> = Vec::new();
    header.extend_from_slice(&magic_string);
    header.extend_from_slice(&header_len_bytes);
    header.extend_from_slice(header_dict_str.as_bytes());

    header
}

pub fn to_numpy_bytes(img_channels: &[ImageRect<'_, i32>]) -> Vec<u8> {
    if img_channels.is_empty() {
        return Vec::new();
    }

    let mut ret = vec![];
    let size = img_channels[0].size();
    let (width, height) = size;

    for channel in img_channels {
        assert_eq!(channel.size(), size)
    }

    ret.extend(
        (0..height)
            .flat_map(|y| {
                (0..width).flat_map(move |x| [0, 1, 2].map(move |c| img_channels[c].row(y)[x]))
            })
            .flat_map(|x| ((x.clamp(0, 255) as f32) / 255.0f32).to_le_bytes()),
    );
    ret
}

/// Converts frames to a Vec<u8> in .npy format.
/// The data will be represented as little-endian 32-bit floats ('<f4').
/// The shape of the NumPy array will be (1, height, width, num_channels).
///
pub fn to_numpy(frame: Vec<ImageRect<'_, i32>>) -> Vec<u8> {
    if frame.is_empty() {
        panic!("Input frame data is empty, cannot create .npy file.");
    }
    let size = frame[0].size();
    let (width, height) = size;
    let num_channels = frame.len();
    let num_frames = 1;

    let mut npy_file_bytes = Vec::new();
    npy_file_bytes.extend(numpy_header(width, height, num_channels, num_frames));
    // Consistent channel sizes are checked inside the call to `to_numpy_bytes`.
    npy_file_bytes.extend(to_numpy_bytes(&frame));

    npy_file_bytes
}

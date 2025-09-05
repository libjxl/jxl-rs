// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

macro_rules! create_output_buffers {
    ($info:expr, $pixel_format:expr, $bufs:ident, $slices:ident) => {
        use crate::api::{JxlColorType::Rgb, JxlOutputBuffer};
        use std::mem::MaybeUninit;

        let orientation = $info.orientation;
        let (width, height) = $info.size;

        let (buffer_width, buffer_height) = if orientation.is_transposing() {
            (height, width)
        } else {
            (width, height)
        };

        let num_channels = $pixel_format.color_type.samples_per_pixel();

        let mut $bufs: Vec<Vec<MaybeUninit<u8>>> = Vec::new();

        // For RGB images, first buffer holds interleaved RGB data
        match $pixel_format.color_type == Rgb {
            true => {
                $bufs.push(vec![
                    MaybeUninit::uninit();
                    buffer_width * buffer_height * 12
                ]);
                for _ in 3..num_channels {
                    $bufs.push(vec![
                        MaybeUninit::uninit();
                        buffer_width * buffer_height * 4
                    ]);
                }
            }
            false => {
                // For grayscale or other formats, one buffer per channel
                for _ in 0..num_channels {
                    $bufs.push(vec![
                        MaybeUninit::uninit();
                        buffer_width * buffer_height * 4
                    ]);
                }
            }
        }

        let mut $slices: Vec<JxlOutputBuffer> = $bufs
            .iter_mut()
            .enumerate()
            .map(|(i, buffer)| {
                let bytes_per_pixel = if i == 0 && $pixel_format.color_type == Rgb {
                    12 // Interleaved RGB
                } else {
                    4 // Single channel
                };
                JxlOutputBuffer::new_uninit(
                    buffer.as_mut_slice(),
                    buffer_height,
                    bytes_per_pixel * buffer_width,
                )
            })
            .collect();
    };
}
pub(crate) use create_output_buffers;

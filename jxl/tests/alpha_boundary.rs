// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//! Tests for alpha channel handling at group boundaries.
//!
//! Tests boundary positions (x=256, 512) using dice.jxl which has alpha
//! and is large enough to span multiple groups.

use jxl::api::{JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, ProcessingResult};
use jxl::image::{Image, Rect};

#[test]
fn alpha_boundary() {
    let data = std::fs::read("resources/test/dice.jxl").expect("Failed to read dice.jxl");

    let options = JxlDecoderOptions::default();
    let decoder = JxlDecoder::new(options);

    // Process to get image info
    let mut input = data.as_slice();
    let decoder = match decoder.process(&mut input).unwrap() {
        ProcessingResult::Complete { result } => result,
        ProcessingResult::NeedsMoreInput { .. } => panic!("Need more input for header"),
    };

    let basic_info = decoder.basic_info();
    let (width, height) = basic_info.size;
    let num_color_channels = decoder
        .current_pixel_format()
        .color_type
        .samples_per_pixel();

    // Process to get frame info
    let decoder = match decoder.process(&mut input).unwrap() {
        ProcessingResult::Complete { result } => result,
        ProcessingResult::NeedsMoreInput { .. } => panic!("Need more input for frame"),
    };

    // Prepare output buffers
    let mut rgb_buffer =
        Image::<f32>::new_with_value((width * num_color_channels, height), f32::NAN).unwrap();
    let mut alpha_buffer = Image::<f32>::new_with_value((width, height), f32::NAN).unwrap();

    let mut buffers = vec![
        JxlOutputBuffer::from_image_rect_mut(
            rgb_buffer
                .get_rect_mut(Rect {
                    origin: (0, 0),
                    size: (width * num_color_channels, height),
                })
                .into_raw(),
        ),
        JxlOutputBuffer::from_image_rect_mut(
            alpha_buffer
                .get_rect_mut(Rect {
                    origin: (0, 0),
                    size: (width, height),
                })
                .into_raw(),
        ),
    ];

    // Decode the frame
    let _decoder = match decoder.process(&mut input, &mut buffers).unwrap() {
        ProcessingResult::Complete { result } => result,
        ProcessingResult::NeedsMoreInput { .. } => panic!("Need more input for pixels"),
    };

    // Check alpha values at group boundaries (256, 512)
    let y = height / 2;
    let boundary_positions = [254, 255, 256, 257, 510, 511, 512, 513];

    for &x in &boundary_positions {
        if x < width {
            let alpha = alpha_buffer.row(y)[x];
            assert!(
                alpha >= 0.0 && alpha <= 1.0,
                "Alpha at x={} is {} (out of [0,1] range)",
                x,
                alpha
            );
        }
    }
}

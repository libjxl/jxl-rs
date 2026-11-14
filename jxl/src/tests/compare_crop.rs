// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use super::decode::{DecodeParams, decode};
use crate::api::{Event, JxlDecoder, JxlDecoderOptions, ProfileLevel};
use crate::image::Rect;
use crate::util::ShiftRightCeil;

pub fn run(file: &[u8], downsample_factor: u8) {
    let decoder = {
        let mut input = file;
        let options = JxlDecoderOptions {
            max_profile_level: ProfileLevel::Main10,
            adjust_orientation: false,
            ..Default::default()
        };
        let mut decoder = JxlDecoder::new(options);
        assert_eq!(
            decoder.process(&mut input, None, None).unwrap(),
            Event::BasicInfo,
        );
        decoder
    };
    let basic_info = decoder.basic_info().unwrap();
    let (width, height) = basic_info.size;
    let orientation = basic_info.orientation;
    let unoriented_output_size = (
        width.shrc(downsample_factor),
        height.shrc(downsample_factor),
    );
    let output_size = orientation.map_size(unoriented_output_size);

    let full = decode::<f32>(
        file,
        DecodeParams {
            downsample_factor,
            ..Default::default()
        },
    )
    .unwrap();

    arbtest::arbtest(|u| {
        // Align to unoriented coordinates.
        let crop_width = u.int_in_range((width / 8).max(1)..=(width / 2).max(1))?;
        let crop_height = u.int_in_range((height / 8).max(1)..=(height / 2).max(1))?;
        let xlimit = width - crop_width;
        let ylimit = height - crop_height;
        let x0 = u.int_in_range(0..=(xlimit >> downsample_factor))?;
        let y0 = u.int_in_range(0..=(ylimit >> downsample_factor))?;

        let unoriented_crop = Rect {
            origin: (x0 << downsample_factor, y0 << downsample_factor),
            size: (crop_width, crop_height),
        };
        let crop = orientation.display_rect(unoriented_crop, (width, height));

        let unoriented_output_rect = Rect {
            origin: (x0, y0),
            size: (
                crop_width.shrc(downsample_factor),
                crop_height.shrc(downsample_factor),
            ),
        };
        let output_rect = orientation.display_rect(unoriented_output_rect, unoriented_output_size);
        let (x0, y0) = output_rect.origin;
        let (x1, y1) = output_rect.end();

        let cropped = decode::<f32>(
            file,
            DecodeParams {
                downsample_factor,
                crop_region: Some(crop),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(full.len(), cropped.len());

        for (full_frame, crop_frame) in full.iter().zip(&cropped) {
            assert_eq!(full_frame.len(), crop_frame.len());

            for (full_buf, crop_buf) in full_frame.iter().zip(crop_frame) {
                let num_channels = full_buf.size().0 / output_size.0;
                let x0 = x0 * num_channels;
                let x1 = x1 * num_channels;
                assert_eq!(crop_buf.size(), (x1 - x0, y1 - y0));

                for y in 0..y1 - y0 {
                    assert_eq!(crop_buf.row(y), &full_buf.row(y + y0)[x0..x1]);
                }
            }
        }

        Ok(())
    });
}

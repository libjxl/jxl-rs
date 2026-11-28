// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::path::Path;

use super::decode::{as_output_buffers, compare_frames};
use crate::api::{Event, JxlDecoder, JxlDecoderOptions, ProfileLevel, VisibleFrameInfo};
use crate::image::{Image, Rect};

fn fresh_buffers(reference: &[Image<f32>]) -> Vec<Image<f32>> {
    reference
        .iter()
        .map(|b| Image::new_with_value(b.size(), f32::NAN).unwrap())
        .collect()
}

fn decode_single_frame(
    file: &[u8],
    frame_info: &VisibleFrameInfo,
    downsample_factor: u8,
    crop_region: Rect,
) -> Result<Vec<Image<f32>>, crate::error::Error> {
    let options = JxlDecoderOptions {
        max_profile_level: ProfileLevel::Main10,
        ..Default::default()
    };
    let mut decoder = JxlDecoder::new(options);
    decoder.set_downsample_factor(downsample_factor)?;

    let mut input = file;
    assert_eq!(
        decoder.process(&mut input, None, None).unwrap(),
        Event::BasicInfo
    );
    decoder.set_crop_region(Some(crop_region))?;
    let info = decoder.basic_info().unwrap();
    let (w, h) = info.size;

    let offset = decoder.start_new_frame(frame_info.seek_target).unwrap();
    input = &file[offset as usize..];

    let pixel_format = decoder.current_pixel_format().unwrap();
    let num_channels = pixel_format.color_type.samples_per_pixel();
    assert!(num_channels > 0);
    let mut buffers = vec![];
    if pixel_format.color_data_format.is_some() {
        buffers.push(Image::new_with_value((w * num_channels, h), f32::NAN)?);
    }
    for ecf in pixel_format.extra_channel_format.iter() {
        if ecf.is_none() {
            continue;
        }
        buffers.push(Image::new_with_value((w, h), f32::NAN)?);
    }

    let mut out_bufs = as_output_buffers(&mut buffers);
    loop {
        let ev = decoder.process(&mut input, Some(&mut out_bufs), None)?;
        match ev {
            Event::FrameHeader => {}
            Event::FrameComplete { .. } | Event::Complete => break,
            Event::NeedMoreInput { .. } => panic!("unexpected end of input"),
            Event::BasicInfo => unreachable!(),
        }
    }

    Ok(buffers)
}

pub fn run(path: &Path, file: &[u8]) {
    let frames = super::decode::scan_frames(file, usize::MAX);

    let mut decoder = JxlDecoder::new(JxlDecoderOptions {
        enable_repeated_decoding: true,
        max_profile_level: ProfileLevel::Main10,
        ..Default::default()
    });
    let mut input = file;
    assert_eq!(
        decoder.process(&mut input, None, None).unwrap(),
        Event::BasicInfo
    );
    let info = decoder.basic_info().unwrap();
    let orientation = info.orientation;
    let (w, h) = orientation.map_size(info.size);

    arbtest::arbtest(|u| {
        let frame = u.choose(&frames)?;
        let offset = decoder.start_new_frame(frame.seek_target).unwrap();
        input = &file[offset as usize..];
        let mut need_decoding = true;

        u.arbitrary_loop(Some(3), Some(10), |u| {
            let shift = u.int_in_range(0..=3)?;
            let cw = u.int_in_range(1..=w)?;
            let ch = u.int_in_range(1..=h)?;
            let x = u.int_in_range(0..=((w - cw) >> shift))? << shift;
            let y = u.int_in_range(0..=((h - ch) >> shift))? << shift;
            let crop = orientation.display_rect(
                Rect {
                    origin: (x, y),
                    size: (cw, ch),
                },
                (w, h),
            );
            let expected = decode_single_frame(file, frame, shift, crop).unwrap();

            decoder.set_crop_region(None).unwrap();
            decoder.set_downsample_factor(shift).unwrap();
            decoder.set_crop_region(Some(crop)).unwrap();

            let mut actual = fresh_buffers(&expected);
            let mut out_bufs = as_output_buffers(&mut actual);
            if need_decoding {
                loop {
                    let ev = decoder
                        .process(&mut input, Some(&mut out_bufs), None)
                        .unwrap();
                    match ev {
                        Event::FrameHeader => {}
                        Event::FrameComplete { .. } | Event::Complete => break,
                        Event::NeedMoreInput { .. } => panic!("unexpected end of input"),
                        Event::BasicInfo => unreachable!(),
                    }
                }
                need_decoding = false;
            } else {
                decoder.rerender_frame(&mut out_bufs, None).unwrap();
            }

            compare_frames(path, frame.index, &actual, &expected);

            Ok(std::ops::ControlFlow::Continue(()))
        })
    });
}

// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#![no_main]

use jxl::api::{JxlColorType, JxlDecoder, JxlDecoderOptions, ProcessingResult, states};
use jxl::image::{Image, JxlOutputBuffer, Rect};
use libfuzzer_sys::fuzz_target;

fn as_complete<T, U, E>(result: Result<ProcessingResult<T, U>, E>) -> Result<T, ()> {
    match result {
        Ok(ProcessingResult::Complete { result }) => Ok(result),
        _ => Err(()),
    }
}

// Note: This is adapted from jxl_cli/src/dec/mod.rs
fn fuzz_decode(mut data: &[u8]) -> Result<(), ()> {
    let mut decoder_options = JxlDecoderOptions::default();
    decoder_options.pixel_limit = Some(1 << 27);
    let initialized_decoder = JxlDecoder::<states::Initialized>::new(decoder_options);
    let mut decoder_with_image_info = as_complete(initialized_decoder.process(&mut data))?;

    let info = decoder_with_image_info.basic_info();

    let extra_channels = info.extra_channels.len();
    let pixel_format = decoder_with_image_info.current_pixel_format().clone();
    let color_type = pixel_format.color_type;
    // TODO(zond): This is the way the API works right now, let's improve it when the API is cleverer.
    let samples_per_pixel = if color_type == JxlColorType::Grayscale {
        1
    } else {
        3
    };

    loop {
        let decoder_with_frame_info = as_complete(decoder_with_image_info.process(&mut data))?;
        let frame_header = decoder_with_frame_info.frame_header();
        let frame_size = frame_header.size;

        let mut outputs =
            vec![Image::<f32>::new((frame_size.0 * samples_per_pixel, frame_size.1)).unwrap()];

        for _ in 0..extra_channels {
            outputs.push(Image::<f32>::new(frame_size).unwrap());
        }

        let mut output_bufs: Vec<JxlOutputBuffer<'_>> = outputs
            .iter_mut()
            .map(|x| {
                let rect = Rect {
                    size: x.size(),
                    origin: (0, 0),
                };
                JxlOutputBuffer::from_image_rect_mut(x.get_rect_mut(rect).into_raw())
            })
            .collect();

        decoder_with_image_info =
            as_complete(decoder_with_frame_info.process(&mut data, &mut output_bufs))?;

        if !decoder_with_image_info.has_more_frames() {
            break;
        }
    }

    Ok(())
}

fuzz_target!(|data: &[u8]| {
    let _ = fuzz_decode(data);
});

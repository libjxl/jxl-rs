// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::api::{
    Event, JxlColorEncoding, JxlColorProfile, JxlDecoder, JxlDecoderOptions, JxlPixelFormat,
    TestOptions,
};
use crate::headers::frame_header::{BlendingMode, FrameHeader};
use crate::image::Image;
use crate::tests::decode::as_output_buffers;

fn decode_with_profile(
    data: &[u8],
    linear: bool,
    simple: bool,
) -> (Vec<Image<f32>>, Vec<FrameHeader>) {
    let mut decoder = JxlDecoder::new(JxlDecoderOptions {
        test_options: TestOptions {
            use_simple_pipeline: simple,
            ..Default::default()
        },
        ..Default::default()
    });
    let mut input = data;
    assert_eq!(
        decoder.process(&mut input, None, None).unwrap(),
        Event::BasicInfo
    );
    let info = decoder.basic_info().unwrap();
    let (w, h) = info.size;
    let num_extra = info.extra_channels.len();
    let mut format = JxlPixelFormat::rgba_f32(num_extra);
    format.extra_channel_format.fill(None);
    decoder.set_pixel_format(format).unwrap();
    let encoding = if linear {
        JxlColorEncoding::linear_srgb(false)
    } else {
        JxlColorEncoding::srgb(false)
    };
    decoder.set_test_output_color_profile(JxlColorProfile::Simple(encoding));
    let mut buffers = vec![Image::new((w * 4, h)).unwrap()];
    let mut frames = Vec::new();
    let mut headers = Vec::new();
    loop {
        match decoder
            .process(&mut input, Some(&mut as_output_buffers(&mut buffers)), None)
            .unwrap()
        {
            Event::FrameHeader => headers.push(decoder.raw_frame_header().unwrap().clone()),
            Event::FrameComplete { .. } => {
                frames.push(std::mem::replace(
                    &mut buffers[0],
                    Image::new((w * 4, h)).unwrap(),
                ));
            }
            Event::Complete => return (frames, headers),
            event => panic!("Unexpected event {event:?}"),
        }
    }
}

fn assert_linear_output(srgb: &[Image<f32>], linear: &[Image<f32>]) {
    assert_eq!(srgb.len(), linear.len());
    let mut changed = 0;
    for (frame, (srgb, linear)) in srgb.iter().zip(linear).enumerate() {
        assert_eq!(srgb.size(), linear.size());
        for y in 0..srgb.size().1 {
            for (x, (&s, &l)) in srgb.row(y).iter().zip(linear.row(y)).enumerate() {
                if x % 4 == 3 {
                    assert_eq!(s, l);
                    continue;
                }
                let a = s.abs() as f64;
                let expected = if a <= 0.04045 {
                    a / 12.92
                } else {
                    ((a + 0.055) / 1.055).powf(2.4)
                };
                let expected = (expected as f32).copysign(s);
                assert!(
                    (l - expected).abs() < 1e-4 * (1.0 + expected.abs()),
                    "frame {frame}, ({x}, {y}): {l} != {expected} (sRGB {s})"
                );
                changed += usize::from((expected - s).abs() > 0.05);
            }
        }
    }
    assert!(
        changed > 100,
        "fixture must distinguish sRGB from linear samples"
    );
}

#[test]
fn blending_uses_signalled_transfer_function() {
    let data = std::fs::read("resources/test/issue772_blendbug.jxl").unwrap();
    for simple in [false, true] {
        let (srgb, headers) = decode_with_profile(&data, false, simple);
        assert!(
            headers
                .iter()
                .any(|h| h.blending_info.mode == BlendingMode::Blend)
        );
        let (linear, _) = decode_with_profile(&data, true, simple);
        assert_linear_output(&srgb, &linear);
    }
}

#[test]
fn output_conversion_preserves_post_transform_references() {
    let data = std::fs::read("resources/test/animation_seek_noise_references.jxl").unwrap();
    for simple in [false, true] {
        let (srgb, headers) = decode_with_profile(&data, false, simple);
        let saved = headers
            .iter()
            .position(|h| !h.needs_blending() && h.can_be_referenced && !h.save_before_ct)
            .unwrap();
        let slot = headers[saved].save_as_reference;
        assert!(
            headers[saved + 1..].iter().any(
                |h| h.blending_info.source == slot && h.blending_info.mode == BlendingMode::Add
            )
        );
        let (linear, _) = decode_with_profile(&data, true, simple);
        assert_linear_output(&srgb, &linear);
    }
}

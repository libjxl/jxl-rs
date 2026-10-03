// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::path::Path;

use crate::api::{
    JxlColorType, JxlDataFormat, JxlDecoder, JxlDecoderInner, JxlDecoderOptions, JxlPixelFormat,
    JxlTransferFunction, ProcessingResult, VisibleFrameInfo, states,
};
use crate::error::Error;
use crate::image::{Image, ImageDataType, JxlOutputBuffer, Rect};
use crate::tests::decode::{DecodeParams, compare_frames, decode, scan_frames};

// OOO jxlp boxes require any frame to start in a box that has all the logically-before
// boxes physically before it, and all the logically-after boxes physically after it.
// This test file does *not* satisfy this property.
#[test]
fn decode_ooo_jxlp_invalid_animated_container() {
    let data = std::fs::read("resources/test/invalid_animated_ooo_jxlp.jxl").unwrap();
    let res = decode::<f32>(&data, Default::default());
    assert!(
        matches!(res, Err(Error::InvalidBox)),
        "expected error due to frame start in non-valid checkpoint box"
    );
}

#[test]
fn test_preview_size_none_for_regular_files() {
    let file = std::fs::read("resources/test/basic.jxl").unwrap();
    let options = JxlDecoderOptions::default();
    let mut decoder = JxlDecoder::<states::Initialized>::new(options);
    let mut input = file.as_slice();
    let decoder = loop {
        match decoder.process(&mut input, None).unwrap() {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => decoder = fallback,
        }
    };
    assert!(decoder.basic_info().preview_size.is_none());
}

#[test]
fn test_preview_size_some_for_preview_files() {
    let file = std::fs::read("resources/test/with_preview.jxl").unwrap();
    let options = JxlDecoderOptions::default();
    let mut decoder = JxlDecoder::<states::Initialized>::new(options);
    let mut input = file.as_slice();
    let decoder = loop {
        match decoder.process(&mut input, None).unwrap() {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => decoder = fallback,
        }
    };
    assert_eq!(decoder.basic_info().preview_size, Some((16, 16)));
}

#[test]
fn test_set_pixel_format() {
    let file = std::fs::read("resources/test/basic.jxl").unwrap();
    let options = JxlDecoderOptions::default();
    let mut decoder = JxlDecoder::<states::Initialized>::new(options);
    let mut input = file.as_slice();
    let mut decoder = loop {
        match decoder.process(&mut input, None).unwrap() {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => decoder = fallback,
        }
    };
    let default_format = decoder.current_pixel_format().clone();
    assert_eq!(default_format.color_type, JxlColorType::Rgb);

    let new_format = JxlPixelFormat {
        color_type: JxlColorType::Grayscale,
        color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
        extra_channel_format: vec![],
    };
    decoder.set_pixel_format(new_format.clone()).unwrap();
    assert_eq!(decoder.current_pixel_format(), &new_format);
}

#[test]
fn test_default_output_tf_by_pixel_format() {
    let file = std::fs::read("resources/test/lossy_with_icc.jxl").unwrap();
    let options = JxlDecoderOptions::default();
    let mut decoder = JxlDecoder::<states::Initialized>::new(options);
    let mut input = file.as_slice();
    let mut decoder = loop {
        match decoder.process(&mut input, None).unwrap() {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => decoder = fallback,
        }
    };

    assert_eq!(
        *decoder.output_color_profile().transfer_function().unwrap(),
        JxlTransferFunction::Linear,
    );

    decoder.set_pixel_format(JxlPixelFormat::rgba8(0)).unwrap();
    assert_eq!(
        *decoder.output_color_profile().transfer_function().unwrap(),
        JxlTransferFunction::SRGB,
    );

    decoder
        .set_pixel_format(JxlPixelFormat::rgba_f16(0))
        .unwrap();
    assert_eq!(
        *decoder.output_color_profile().transfer_function().unwrap(),
        JxlTransferFunction::Linear,
    );

    decoder.set_pixel_format(JxlPixelFormat::rgba16(0)).unwrap();
    assert_eq!(
        *decoder.output_color_profile().transfer_function().unwrap(),
        JxlTransferFunction::SRGB,
    );
}

#[test]
fn test_fill_opaque_alpha_both_pipelines() {
    let file = std::fs::read("resources/test/basic.jxl").unwrap();

    for use_simple in [true, false] {
        let frames = decode::<f32>(
            &file,
            DecodeParams {
                pixel_format: Some(JxlPixelFormat::rgba_f32(0)),
                use_simple_pipeline: use_simple,
                ..Default::default()
            },
        )
        .unwrap();
        let color_buffer = &frames[0][0];
        let (xs, height) = color_buffer.size();
        let width = xs / 4;

        for y in 0..height {
            let row = color_buffer.row(y);
            for x in 0..width {
                let alpha = row[x * 4 + 3];
                assert_eq!(
                    alpha, 1.0,
                    "Alpha at ({},{}) should be 1.0, got {} (use_simple={})",
                    x, y, alpha, use_simple
                );
            }
        }
    }
}

/// Test that premultiply_output=true produces premultiplied alpha output
/// from a source with straight (non-premultiplied) alpha.
#[test]
fn test_premultiply_output_straight_alpha() {
    let file =
        std::fs::read("resources/test/conformance_test_images/alpha_nonpremultiplied.jxl").unwrap();

    for use_simple in [true, false] {
        let straight_frames = decode::<f32>(
            &file,
            DecodeParams {
                pixel_format: Some(JxlPixelFormat::rgba_f32(1)),
                use_simple_pipeline: use_simple,
                ..Default::default()
            },
        )
        .unwrap();
        let straight_buffer = &straight_frames[0][0];
        let premul_frames = decode::<f32>(
            &file,
            DecodeParams {
                pixel_format: Some(JxlPixelFormat::rgba_f32(1)),
                use_simple_pipeline: use_simple,
                premultiply_output: true,
                ..Default::default()
            },
        )
        .unwrap();
        let premul_buffer = &premul_frames[0][0];
        let (xs, height) = straight_buffer.size();
        let width = xs / 4;

        let mut found_semitransparent = false;
        for y in 0..height {
            let straight_row = straight_buffer.row(y);
            let premul_row = premul_buffer.row(y);
            for x in 0..width {
                let sr = straight_row[x * 4];
                let sg = straight_row[x * 4 + 1];
                let sb = straight_row[x * 4 + 2];
                let sa = straight_row[x * 4 + 3];

                let pr = premul_row[x * 4];
                let pg = premul_row[x * 4 + 1];
                let pb = premul_row[x * 4 + 2];
                let pa = premul_row[x * 4 + 3];

                assert!(
                    (sa - pa).abs() < 1e-5,
                    "Alpha mismatch at ({},{}): straight={}, premul={} (use_simple={})",
                    x,
                    y,
                    sa,
                    pa,
                    use_simple
                );

                let expected_r = sr * sa;
                let expected_g = sg * sa;
                let expected_b = sb * sa;

                let tol = 0.01;
                assert!(
                    (expected_r - pr).abs() < tol,
                    "R mismatch at ({},{}): expected={}, got={} (use_simple={})",
                    x,
                    y,
                    expected_r,
                    pr,
                    use_simple
                );
                assert!(
                    (expected_g - pg).abs() < tol,
                    "G mismatch at ({},{}): expected={}, got={} (use_simple={})",
                    x,
                    y,
                    expected_g,
                    pg,
                    use_simple
                );
                assert!(
                    (expected_b - pb).abs() < tol,
                    "B mismatch at ({},{}): expected={}, got={} (use_simple={})",
                    x,
                    y,
                    expected_b,
                    pb,
                    use_simple
                );

                if sa > 0.01 && sa < 0.99 {
                    found_semitransparent = true;
                }
            }
        }

        assert!(
            found_semitransparent,
            "Test image should have semi-transparent pixels (use_simple={})",
            use_simple
        );
    }
}

/// Test that premultiplied RGBA output from a grayscale image remains gray.
#[test]
fn test_premultiply_output_grayscale_as_rgba() {
    let file = std::fs::read("resources/test/gray_alpha_lossless.jxl").unwrap();
    let frames = decode::<f32>(
        &file,
        DecodeParams {
            pixel_format: Some(JxlPixelFormat::rgba_f32(1)),
            premultiply_output: true,
            ..Default::default()
        },
    )
    .unwrap();
    let rgba = &frames[0][0];
    let (xs, height) = rgba.size();
    let width = xs / 4;

    for y in 0..height {
        let row = rgba.row(y);
        for x in 0..width {
            assert_eq!(row[x * 4], row[x * 4 + 1]);
            assert_eq!(row[x * 4 + 1], row[x * 4 + 2]);
        }
    }
}

/// Test that premultiply_output=true doesn't double-premultiply
/// when the source already has premultiplied alpha (alpha_associated=true).
#[test]
fn test_premultiply_output_already_premultiplied() {
    let file =
        std::fs::read("resources/test/conformance_test_images/alpha_premultiplied.jxl").unwrap();

    for use_simple in [true, false] {
        let without_flag_frames = decode::<f32>(
            &file,
            DecodeParams {
                pixel_format: Some(JxlPixelFormat::rgba_f32(1)),
                use_simple_pipeline: use_simple,
                ..Default::default()
            },
        )
        .unwrap();
        let without_flag_buffer = &without_flag_frames[0][0];
        let with_flag_frames = decode::<f32>(
            &file,
            DecodeParams {
                pixel_format: Some(JxlPixelFormat::rgba_f32(1)),
                use_simple_pipeline: use_simple,
                premultiply_output: true,
                ..Default::default()
            },
        )
        .unwrap();
        let with_flag_buffer = &with_flag_frames[0][0];
        let (xs, height) = without_flag_buffer.size();
        let width = xs / 4;

        for y in 0..height {
            let without_row = without_flag_buffer.row(y);
            let with_row = with_flag_buffer.row(y);
            for x in 0..width {
                for c in 0..4 {
                    let without_val = without_row[x * 4 + c];
                    let with_val = with_row[x * 4 + c];
                    assert!(
                        (without_val - with_val).abs() < 1e-5,
                        "Mismatch at ({},{}) channel {}: without_flag={}, with_flag={} (use_simple={})",
                        x,
                        y,
                        c,
                        without_val,
                        with_val,
                        use_simple
                    );
                }
            }
        }
    }
}

/// Test that animations with reference frames work correctly.
#[test]
fn test_animation_with_reference_frames() {
    let file =
        std::fs::read("resources/test/conformance_test_images/animation_spline.jxl").unwrap();

    let options = JxlDecoderOptions::default();
    let decoder = JxlDecoder::<states::Initialized>::new(options);
    let mut input = file.as_slice();

    let mut decoder = decoder;
    let mut decoder = loop {
        match decoder.process(&mut input, None).unwrap() {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                decoder = fallback;
            }
        }
    };

    let rgb_format = JxlPixelFormat {
        color_type: JxlColorType::Rgb,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![],
    };
    decoder.set_pixel_format(rgb_format).unwrap();

    let basic_info = decoder.basic_info().clone();
    let (width, height) = basic_info.size;

    let mut frame_count = 0;

    loop {
        let mut decoder_frame = loop {
            match decoder.process(&mut input, None).unwrap() {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    decoder = fallback;
                }
            }
        };

        let mut color_buffer = Image::<f32>::new((width * 3, height)).unwrap();
        let mut buffers: Vec<_> = vec![JxlOutputBuffer::from_image_rect_mut(
            color_buffer
                .get_rect_mut(Rect {
                    origin: (0, 0),
                    size: (width * 3, height),
                })
                .into_raw(),
        )];

        decoder = loop {
            match decoder_frame
                .process(&mut input, &mut buffers, None)
                .unwrap()
            {
                ProcessingResult::Complete { result } => break result,
                ProcessingResult::NeedsMoreInput { fallback, .. } => {
                    decoder_frame = fallback;
                }
            }
        };

        frame_count += 1;

        if !decoder.has_more_frames() {
            break;
        }
    }

    assert!(
        frame_count > 1,
        "Expected multiple frames in animation, got {}",
        frame_count
    );
}

#[test]
fn test_skip_frame_then_decode_next() {
    let file =
        std::fs::read("resources/test/conformance_test_images/animation_spline.jxl").unwrap();

    let options = JxlDecoderOptions::default();
    let decoder = JxlDecoder::<states::Initialized>::new(options);
    let mut input = file.as_slice();

    let mut decoder = decoder;
    let mut decoder = loop {
        match decoder.process(&mut input, None).unwrap() {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                decoder = fallback;
            }
        }
    };

    let rgb_format = JxlPixelFormat {
        color_type: JxlColorType::Rgb,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: vec![],
    };
    decoder.set_pixel_format(rgb_format).unwrap();

    let basic_info = decoder.basic_info().clone();
    let (width, height) = basic_info.size;

    let mut decoder_frame = loop {
        match decoder.process(&mut input, None).unwrap() {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                decoder = fallback;
            }
        }
    };

    let mut decoder = loop {
        match decoder_frame.skip_frame(&mut input).unwrap() {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                decoder_frame = fallback;
            }
        }
    };

    assert!(
        decoder.has_more_frames(),
        "Animation should have more frames"
    );

    let mut decoder_frame = loop {
        match decoder.process(&mut input, None).unwrap() {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                decoder = fallback;
            }
        }
    };

    let mut color_buffer = Image::<f32>::new((width * 3, height)).unwrap();
    let mut buffers: Vec<_> = vec![JxlOutputBuffer::from_image_rect_mut(
        color_buffer
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (width * 3, height),
            })
            .into_raw(),
    )];

    let decoder = loop {
        match decoder_frame
            .process(&mut input, &mut buffers, None)
            .unwrap()
        {
            ProcessingResult::Complete { result } => break result,
            ProcessingResult::NeedsMoreInput { fallback, .. } => {
                decoder_frame = fallback;
            }
        }
    };

    let _ = decoder.has_more_frames();
}

fn check_output_format_matches_f32<T: ImageDataType>() {
    use crate::api::Endianness;
    use crate::image::DataTypeTag;

    let (data_format, scale, clamp, tolerance) = match T::DATA_TYPE_ID {
        DataTypeTag::U8 => (JxlDataFormat::U8 { bit_depth: 8 }, 1.0 / 255.0, true, 0.004),
        DataTypeTag::U16 => (
            JxlDataFormat::U16 {
                endianness: Endianness::native(),
                bit_depth: 16,
            },
            1.0 / 65535.0,
            true,
            0.0001,
        ),
        DataTypeTag::F16 => (
            JxlDataFormat::F16 {
                endianness: Endianness::native(),
            },
            1.0,
            false,
            0.002,
        ),
        _ => unreachable!(),
    };

    let file = std::fs::read("resources/test/conformance_test_images/bicycles.jxl").unwrap();

    for color_type in [JxlColorType::Rgb, JxlColorType::Bgra] {
        let f32_format = JxlPixelFormat {
            color_type,
            color_data_format: Some(JxlDataFormat::f32()),
            extra_channel_format: vec![],
        };
        let format = JxlPixelFormat {
            color_type,
            color_data_format: Some(data_format),
            extra_channel_format: vec![],
        };

        let f32_frames = decode::<f32>(
            &file,
            DecodeParams {
                pixel_format: Some(f32_format),
                ..Default::default()
            },
        )
        .unwrap();
        let f32_buffer = &f32_frames[0][0];
        let (xs, height) = f32_buffer.size();

        for use_simple in [true, false] {
            let frames = decode::<T>(
                &file,
                DecodeParams {
                    pixel_format: Some(format.clone()),
                    use_simple_pipeline: use_simple,
                    ..Default::default()
                },
            )
            .unwrap();
            let buffer = &frames[0][0];

            for y in 0..height {
                let f32_row = f32_buffer.row(y);
                let row = buffer.row(y);
                for x in 0..xs {
                    let f32_val = if clamp {
                        f32_row[x].clamp(0.0, 1.0)
                    } else {
                        f32_row[x]
                    };
                    let val = (row[x].to_f64() * scale) as f32;
                    let error = (f32_val - val).abs();
                    assert!(
                        error < tolerance,
                        "{color_type:?} {:?} mismatch at ({x},{y}): f32={f32_val}, got={:?} (scaled={val}), error={error} (use_simple={use_simple})",
                        T::DATA_TYPE_ID,
                        row[x],
                    );
                }
            }
        }
    }
}

/// Test that u8 output matches f32 output within quantization tolerance.
#[test]
fn test_output_format_u8_matches_f32() {
    check_output_format_matches_f32::<u8>();
}

/// Test that u16 output matches f32 output within quantization tolerance.
#[test]
fn test_output_format_u16_matches_f32() {
    check_output_format_matches_f32::<u16>();
}

/// Test that f16 output matches f32 output within f16 precision tolerance.
#[test]
fn test_output_format_f16_matches_f32() {
    check_output_format_matches_f32::<crate::util::f16>();
}

/// CMYK interleaved output matches the RGB color channels for C, M and Y, and
/// the Black extra channel plane for K.
#[test]
fn test_cmyk_pixel_format() {
    let file = std::fs::read("resources/test/conformance_test_images/cmyk_layers.jxl").unwrap();

    // cmyk_layers.jxl has two extra channels: Black (index 0) and Alpha
    // (index 1).
    let cmyk_format = JxlPixelFormat::cmyk8(2);
    let reference_format = JxlPixelFormat {
        color_type: JxlColorType::Rgb,
        color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
        extra_channel_format: vec![Some(JxlDataFormat::U8 { bit_depth: 8 }), None],
    };

    for use_simple in [true, false] {
        let cmyk_frames = decode::<u8>(
            &file,
            DecodeParams {
                pixel_format: Some(cmyk_format.clone()),
                use_simple_pipeline: use_simple,
                ..Default::default()
            },
        )
        .unwrap();
        let reference_frames = decode::<u8>(
            &file,
            DecodeParams {
                pixel_format: Some(reference_format.clone()),
                use_simple_pipeline: use_simple,
                ..Default::default()
            },
        )
        .unwrap();
        let cmyk = &cmyk_frames[0][0];
        let rgb = &reference_frames[0][0];
        let black = &reference_frames[0][1];
        let (width, height) = black.size();

        for y in 0..height {
            let cmyk_row = cmyk.row(y);
            let rgb_row = rgb.row(y);
            let black_row = black.row(y);
            for x in 0..width {
                for c in 0..3 {
                    assert_eq!(
                        cmyk_row[x * 4 + c],
                        rgb_row[x * 3 + c],
                        "CMY mismatch at ({x},{y}) channel {c} (use_simple={use_simple})"
                    );
                }
                assert_eq!(
                    cmyk_row[x * 4 + 3],
                    black_row[x],
                    "K mismatch at ({x},{y}) (use_simple={use_simple})"
                );
            }
        }
    }
}

/// Requesting CMYK output for a non-CMYK image fails.
#[test]
fn test_cmyk_pixel_format_requires_cmyk_image() {
    let file = std::fs::read("resources/test/basic.jxl").unwrap();
    let result = decode::<f32>(
        &file,
        DecodeParams {
            pixel_format: Some(JxlPixelFormat::cmyk8(0)),
            ..Default::default()
        },
    );
    assert!(
        matches!(result, Err(Error::NotCmyk)),
        "expected NotCmyk, got {result:?}"
    );
}

/// Regression test for ClusterFuzz issue 5342436251336704
#[test]
fn test_fuzzer_smallbuffer_overflow() {
    use std::panic;

    let data = include_bytes!("../../tests/testdata/fuzzer_smallbuffer_overflow.jxl");

    let result = panic::catch_unwind(|| {
        let _ = decode::<f32>(
            data,
            DecodeParams {
                chunk_size: 1024,
                ..Default::default()
            },
        );
    });

    if let Err(e) = result {
        let panic_msg = e
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| e.downcast_ref::<String>().cloned())
            .unwrap_or_default();
        assert!(
            !panic_msg.contains("overflow"),
            "Unexpected overflow panic: {}",
            panic_msg
        );
    }
}

/// Regression test for https://issues.chromium.org/issues/541318910: flushing a
/// frame that does not support rendering before the last pass used to force an
/// eager render of an incomplete group.
#[test]
fn flush_without_partial_render_support() {
    let data = std::fs::read("resources/test/squeeze_empty_residual.jxl").unwrap();
    for chunk_size in 1..=16 {
        decode::<f32>(
            &data,
            DecodeParams {
                chunk_size,
                do_flush: true,
                ..Default::default()
            },
        )
        .unwrap();
    }
}

/// Regression test for https://issues.chromium.org/issues/562761172: flushing
/// a truncated image used to panic when a smooth-squeeze upsample step read a
/// tile whose channel had not been decoded yet.
#[test]
fn flush_truncated_squeeze_missing_tiles() {
    let data = include_bytes!("../../tests/testdata/truncated_squeeze_flush_missing_tiles.jxl");
    for chunk_size in [64, 256, usize::MAX] {
        decode::<f32>(
            data,
            DecodeParams {
                chunk_size,
                do_flush: true,
                allow_partial: true,
                ..Default::default()
            },
        )
        .unwrap();
    }
}

#[test]
fn flush_truncated_squeeze_missing_avg() {
    let data = include_bytes!("../../tests/testdata/truncated_squeeze_missing_avg.jxl");
    for chunk_size in [64, 256, usize::MAX] {
        decode::<f32>(
            data,
            DecodeParams {
                chunk_size,
                do_flush: true,
                allow_partial: true,
                ..Default::default()
            },
        )
        .unwrap();
    }
}

/// Regression test: flushing a truncated image with tiled channels whose tile
/// dimension is <= 1 used to panic during smooth-squeeze upsampling.
#[test]
fn flush_truncated_squeeze_small_tiles() {
    let data = include_bytes!("../../tests/testdata/truncated_squeeze_flush_small_tiles.jxl");
    for chunk_size in [64, 256, usize::MAX] {
        decode::<f32>(
            data,
            DecodeParams {
                chunk_size,
                do_flush: true,
                allow_partial: true,
                ..Default::default()
            },
        )
        .unwrap();
    }
}

fn make_box(ty: &[u8; 4], content: &[u8]) -> Vec<u8> {
    let len = (8 + content.len()) as u32;
    let mut buf = Vec::new();
    buf.extend(len.to_be_bytes());
    buf.extend(ty);
    buf.extend(content);
    buf
}

fn add_container_header(container: &mut Vec<u8>) {
    let sig = [
        0x00, 0x00, 0x00, 0x0c, 0x4a, 0x58, 0x4c, 0x20, 0x0d, 0x0a, 0x87, 0x0a,
    ];
    let ftyp = make_box(b"ftyp", b"jxl \x00\x00\x00\x00jxl ");
    container.extend(&sig);
    container.extend(&ftyp);
}

fn wrap_with_jxlp_chunks(codestream: &[u8], chunk_starts: &[usize]) -> Vec<u8> {
    let mut starts = chunk_starts.to_vec();
    starts.sort_unstable();
    starts.dedup();
    if starts.first().copied() != Some(0) {
        starts.insert(0, 0);
    }
    if starts.last().copied() != Some(codestream.len()) {
        starts.push(codestream.len());
    }
    assert!(starts.len() >= 2);

    let mut container = Vec::new();
    add_container_header(&mut container);

    let num_chunks = starts.len() - 1;
    for i in 0..num_chunks {
        let begin = starts[i];
        let end = starts[i + 1];
        assert!(begin <= end && end <= codestream.len());

        let mut payload = Vec::with_capacity(4 + (end - begin));
        let mut index = i as u32;
        if i + 1 == num_chunks {
            index |= 0x8000_0000;
        }
        payload.extend(index.to_be_bytes());
        payload.extend(&codestream[begin..end]);
        container.extend(make_box(b"jxlp", &payload));
    }

    container
}

/// Seeks `decoder` to visible frame `target`, decodes it and compares it with the sequential decode;
/// returns the input that follows the frame.
fn seek_and_compare<'a>(
    decoder: &mut JxlDecoderInner,
    data: &'a [u8],
    scanned_frames: &[VisibleFrameInfo],
    sequential_frames: &[Vec<Image<f32>>],
    target_visible_index: usize,
) -> &'a [u8] {
    let seek_target = scanned_frames[target_visible_index].seek_target;

    let expected = &sequential_frames[target_visible_index];

    decoder.start_new_frame(seek_target);
    let mut input = &data[seek_target.decode_start_file_offset as usize..];

    let result = decoder.process(&mut input, None, None);
    assert!(
        matches!(result, Ok(ProcessingResult::Complete { .. })),
        "decoder.process: {result:?}"
    );

    let basic_info = decoder.basic_info().unwrap().clone();
    let (width, height) = basic_info.size;

    let default_format = decoder.current_pixel_format().unwrap().clone();
    let requested_format = JxlPixelFormat {
        color_type: default_format.color_type,
        color_data_format: Some(JxlDataFormat::f32()),
        extra_channel_format: default_format
            .extra_channel_format
            .iter()
            .map(|_| Some(JxlDataFormat::f32()))
            .collect(),
    };
    decoder.set_pixel_format(requested_format.clone()).unwrap();

    let channels = requested_format.color_type.samples_per_pixel();
    let num_ec = requested_format.extra_channel_format.len();

    let mut color_buffer = Image::<f32>::new((width * channels, height)).unwrap();
    let mut ec_buffers: Vec<Image<f32>> = (0..num_ec)
        .map(|_| Image::<f32>::new((width, height)).unwrap())
        .collect();
    let mut buffers: Vec<JxlOutputBuffer> = vec![JxlOutputBuffer::from_image_rect_mut(
        color_buffer
            .get_rect_mut(Rect {
                origin: (0, 0),
                size: (width * channels, height),
            })
            .into_raw(),
    )];
    for ec in ec_buffers.iter_mut() {
        buffers.push(JxlOutputBuffer::from_image_rect_mut(
            ec.get_rect_mut(Rect {
                origin: (0, 0),
                size: (width, height),
            })
            .into_raw(),
        ));
    }

    assert!(matches!(
        decoder.process(&mut input, Some(&mut buffers), None),
        Ok(ProcessingResult::Complete { .. })
    ));

    let mut seek_decoded = Vec::with_capacity(1 + num_ec);
    seek_decoded.push(color_buffer);
    seek_decoded.extend(ec_buffers);
    compare_frames(
        Path::new("start_new_frame_seek"),
        target_visible_index,
        expected,
        &seek_decoded,
    );
    input
}

fn assert_start_new_frame_matches_sequential(data: &[u8]) {
    let scanned_frames = scan_frames(data, usize::MAX);

    let sequential_frames = decode(data, Default::default()).unwrap();

    arbtest::arbtest(|u| {
        let initial_offset =
            u.int_in_range(scanned_frames[0].file_offset..=data.len() as u64)? as usize;

        let options = JxlDecoderOptions::default();
        let mut decoder = JxlDecoderInner::new(options);
        let mut input = &data[..initial_offset];

        while let ProcessingResult::Complete { .. } =
            decoder.process(&mut input, None, None).unwrap()
        {
            if input.is_empty() {
                break;
            }
        }

        let num_seeks = u.int_in_range(1..=3)?;
        for _ in 0..num_seeks {
            let target_visible_index =
                u.int_in_range(0..=scanned_frames.len() as u64 - 1)? as usize;
            let input = seek_and_compare(
                &mut decoder,
                data,
                &scanned_frames,
                &sequential_frames,
                target_visible_index,
            );

            let available_bytes = input.len();
            let extra_bytes = u.int_in_range(0..=available_bytes as u64)? as usize;
            if extra_bytes == 0 {
                continue;
            }
            let mut extra_input = &input[..extra_bytes];

            while let ProcessingResult::Complete { .. } =
                decoder.process(&mut extra_input, None, None).unwrap()
            {
                if extra_input.is_empty() {
                    break;
                }
            }
        }
        Ok(())
    });
}

/// Seeks to every visible frame, from the last to the first, and compares each with the sequential
/// decode.
fn assert_every_seek_matches_sequential(data: &[u8]) {
    let scanned_frames = scan_frames(data, usize::MAX);
    let sequential_frames = decode(data, Default::default()).unwrap();
    let mut decoder = JxlDecoderInner::new(JxlDecoderOptions::default());
    let mut input = data;
    while let ProcessingResult::Complete { .. } = decoder.process(&mut input, None, None).unwrap() {
        if input.is_empty() {
            break;
        }
    }
    for target in (0..scanned_frames.len()).rev() {
        seek_and_compare(
            &mut decoder,
            data,
            &scanned_frames,
            &sequential_frames,
            target,
        );
    }
}

// An animation with noise on every frame; displayed frames that are saved as references (some of
// them used by later frames' blending), displayed frames that are not, cropped frames, patches,
// and keyframes after the start. Seeking must skip the unsaved frames and still seed the noise of
// each frame as a sequential decode does.
#[test]
fn test_seek_every_frame_noise_references() {
    let data = std::fs::read("resources/test/animation_seek_noise_references.jxl").unwrap();
    assert_every_seek_matches_sequential(&data);
}

#[test]
fn test_start_new_frame_bare_codestream() {
    let data =
        std::fs::read("resources/test/conformance_test_images/animation_icos4d.jxl").unwrap();
    assert_start_new_frame_matches_sequential(&data);
}

#[test]
fn test_start_new_frame_boxed_jxlp_per_visible_frame() {
    let codestream =
        std::fs::read("resources/test/conformance_test_images/animation_icos4d.jxl").unwrap();

    let scanned_frames = scan_frames(&codestream, usize::MAX);
    assert!(scanned_frames.len() > 1, "need multiple frames");

    let mut chunk_starts: Vec<usize> = scanned_frames
        .iter()
        .map(|f| f.file_offset as usize)
        .collect();
    chunk_starts.sort_unstable();
    chunk_starts.dedup();
    assert_eq!(chunk_starts.len(), scanned_frames.len());

    let container = wrap_with_jxlp_chunks(&codestream, &chunk_starts);
    assert_start_new_frame_matches_sequential(&container);
}

#[test]
fn test_start_new_frame_cropped_traffic_light() {
    let data = std::fs::read("resources/test/cropped_traffic_light.jxl").unwrap();
    assert_start_new_frame_matches_sequential(&data);
}

#[test]
fn test_start_new_frame_animation_newtons_cradle() {
    let data = std::fs::read("resources/test/conformance_test_images/animation_newtons_cradle.jxl")
        .unwrap();
    assert_start_new_frame_matches_sequential(&data);
}

#[test]
fn test_start_new_frame_animation_spline() {
    let data =
        std::fs::read("resources/test/conformance_test_images/animation_spline.jxl").unwrap();
    assert_start_new_frame_matches_sequential(&data);
}

#[test]
fn test_scan_still_image() {
    let data = std::fs::read("resources/test/green_queen_vardct_e3.jxl").unwrap();
    let frames = scan_frames(&data, usize::MAX);

    assert_eq!(frames.len(), 1);
    assert!(frames[0].is_last);
    assert!(frames[0].is_keyframe);
    let total_duration_ms: f64 = frames.iter().map(|f| f.duration_ms).sum();
    assert_eq!(total_duration_ms, 0.0);
}

#[test]
fn test_scan_bare_animation() {
    let data =
        std::fs::read("resources/test/conformance_test_images/animation_icos4d_5.jxl").unwrap();
    let frames = scan_frames(&data, usize::MAX);

    assert!(frames.len() > 1, "expected multiple frames");

    for (i, frame) in frames.iter().enumerate() {
        assert_eq!(frame.index, i);
    }

    assert!(frames.last().unwrap().is_last);
    assert!(frames[0].is_keyframe);
    assert_eq!(
        frames[0].seek_target.decode_start_file_offset,
        frames[0].file_offset as u64
    );
}

#[test]
fn test_scan_animation_offsets_increase() {
    let data =
        std::fs::read("resources/test/conformance_test_images/animation_icos4d_5.jxl").unwrap();
    let frames = scan_frames(&data, usize::MAX);

    for i in 1..frames.len() {
        assert!(
            frames[i].file_offset > frames[i - 1].file_offset,
            "frame {} offset {} should be > frame {} offset {}",
            i,
            frames[i].file_offset,
            i - 1,
            frames[i - 1].file_offset,
        );
    }
}

#[test]
fn test_scan_incremental() {
    let data =
        std::fs::read("resources/test/conformance_test_images/animation_icos4d_5.jxl").unwrap();

    let frames = scan_frames(&data, 128);
    assert!(frames.len() > 1);
    assert!(frames.last().unwrap().is_last);
}

#[test]
fn test_scan_keyframe_detection_still() {
    let data = std::fs::read("resources/test/green_queen_vardct_e3.jxl").unwrap();
    let frames = scan_frames(&data, usize::MAX);

    assert_eq!(frames.len(), 1);
    let f = &frames[0];
    assert!(f.is_keyframe);
    assert_eq!(f.seek_target.decode_start_file_offset, f.file_offset as u64);
    assert_eq!(f.seek_target.visible_frames_to_skip, 0);
}

#[test]
fn test_scan_decode_start_file_offset_consistency() {
    let data =
        std::fs::read("resources/test/conformance_test_images/animation_icos4d_5.jxl").unwrap();

    let frames = scan_frames(&data, usize::MAX);

    for frame in &frames {
        assert!(
            frame.seek_target.decode_start_file_offset <= frame.file_offset,
            "frame {}: decode_start_file_offset {} > file_offset {}",
            frame.index,
            frame.seek_target.decode_start_file_offset,
            frame.file_offset,
        );
        assert_eq!(
            frame.is_keyframe,
            frame.seek_target.visible_frames_to_skip == 0,
            "frame {}: keyframe flag should match visible_frames_to_skip",
            frame.index,
        );
    }
}

#[test]
fn test_scan_with_preview() {
    let data = std::fs::read("resources/test/with_preview.jxl");
    if data.is_err() {
        return;
    }
    let data = data.unwrap();
    let frames = scan_frames(&data, usize::MAX);

    assert!(frames.len() <= 1);
}

#[test]
fn test_scan_patches_not_keyframe() {
    let data = std::fs::read("resources/test/grayscale_patches_var_dct.jxl");
    if data.is_err() {
        return;
    }
    let data = data.unwrap();
    let frames = scan_frames(&data, usize::MAX);

    assert!(!frames.is_empty());
}

/// Regression test for Chromium ClusterFuzz issue 474401148.
#[test]
fn test_fuzzer_xyb_icc_no_panic() {
    #[rustfmt::skip]
    let data: &[u8] = &[
        0xff, 0x0a, 0x01, 0x00, 0x00, 0x04, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x11, 0x25, 0x00,
    ];

    let mut decoder = JxlDecoderInner::new(Default::default());
    let mut input = data;

    if let Ok(ProcessingResult::Complete { .. }) = decoder.process(&mut input, None, None)
        && let Some(profile) = decoder.output_color_profile()
    {
        let _ = profile.try_as_icc();
    }
}

/// Regression test for Chromium ClusterFuzz issue 502853162.
#[test]
fn test_scan_frames_only_empty_followup_no_panic_502853162() {
    #[rustfmt::skip]
    let data: &[u8] = &[
        0xff, 0x0a, 0x31, 0xbd, 0xa2, 0xd0, 0x2a, 0x18,
        0x07, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x0f, 0xa0, 0x26, 0x00, 0xff,
    ];

    let opts = JxlDecoderOptions {
        scan_frames_only: true,
        ..Default::default()
    };
    let mut decoder = JxlDecoderInner::new(opts);

    let mut input = data;
    while decoder.has_more_frames() && !input.is_empty() {
        let _ = decoder.process(&mut input, None, None).unwrap();
    }
}

/// Small regression test for issue #728: squeeze transform boundary bug.
#[test]
fn test_squeeze_boundary_minimal() {
    let frames = decode::<f32>(
        &std::fs::read("resources/test/issue728_minimal.jxl").unwrap(),
        Default::default(),
    )
    .unwrap();
    assert_eq!(frames.len(), 1);
    let frame = &frames[0];
    let buf = &frame[0];
    let (xs, ys) = buf.size();
    for y in 0..ys {
        let row = buf.row(y);
        for (x, &v) in row.iter().enumerate().take(xs) {
            assert!(
                v == 0.0 || v == 1.0,
                "pixel ({}, {}) has value {v}, expected 0.0 or 1.0 \
                 (issue #728 squeeze boundary bug - minimal test)",
                x / 3,
                y,
            );
        }
    }
}

/// Regression test for grid boundary bug with odd-width images (issue #728 variant).
#[test]
fn decode_test_strategic_solid_blue_grid_boundary() {
    let frames = decode::<f32>(
        &std::fs::read("resources/test/strategic_solid_blue.jxl").unwrap(),
        Default::default(),
    )
    .unwrap();
    assert_eq!(frames.len(), 1);
    let frame = &frames[0];

    let buf = &frame[0];
    let (xs, ys) = buf.size();

    assert_eq!(xs, 257 * 3);
    assert_eq!(ys, 256);

    for y in 0..ys {
        for x in 0..257 {
            let row = buf.row(y);
            let (r, g, b) = (row[x * 3], row[x * 3 + 1], row[x * 3 + 2]);
            assert_eq!(
                (r, g, b),
                (0.0, 0.0, 1.0),
                "pixel ({}, {}) has value ({}, {}, {}), expected (0.0, 0.0, 1.0)",
                x,
                y,
                r,
                g,
                b,
            );
        }
    }
}

/// Regression test: a grayscale, non-XYB VarDCT frame has no stage consuming colour
/// channels 1 and 2, so those channels end up with no type in the pipeline. VarDCT still
/// decodes all three channels, and asking the pipeline for their scratch buffers used to
/// panic.
#[test]
fn test_fuzzer_vardct_grayscale_unused_channel() {
    let data = include_bytes!("../../tests/testdata/vardct_grayscale_unused_channel.jxl");
    let frames = decode(data, Default::default()).unwrap();
    let simple_frames = decode(
        data,
        DecodeParams {
            use_simple_pipeline: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].len(), 1);
    assert_eq!(frames[0][0].size(), (1, 1));
    compare_frames(
        Path::new("vardct_grayscale_unused_channel.jxl"),
        0,
        &frames[0],
        &simple_frames[0],
    );
    // Streaming input with flushing exercises the low-memory pipeline's partial renders.
    decode::<f32>(
        data,
        DecodeParams {
            chunk_size: 1,
            do_flush: true,
            ..Default::default()
        },
    )
    .unwrap();
}

/// Regression test: a context map with cluster index 255. This shouldn't panic.
#[test]
fn test_fuzzer_context_map_num_histograms_overflow() {
    let data = include_bytes!("../../tests/testdata/context_map_num_histograms_overflow.jxl");
    let _ = decode::<f32>(data, Default::default());
    let _ = decode::<f32>(
        data,
        DecodeParams {
            chunk_size: 1024,
            do_flush: true,
            ..Default::default()
        },
    );
}

/// Regression test: two nested palette transforms, where the inner one has no colors and no
/// deltas and so produces a 0x1 palette channel. `Image` allocates such a channel as 0x0, and
/// applying the outer palette on top of it used to compare the declared 0x1 size against the
/// allocated 0x0 one and panic. The file is malformed further on, so decoding it must fail --
/// but with an error rather than a panic.
#[test]
fn test_fuzzer_modular_palette_empty_meta_channel() {
    let data = include_bytes!("../../tests/testdata/modular_palette_empty_meta_channel.jxl");
    assert!(decode::<f32>(data, Default::default()).is_err());
}

/// Regression test: a frame with patches that declares `upsampling = 4` and `ec_upsampling = [4]`
/// for an extra channel with `dim_shift = 1`. The declared amounts match, so the guard against
/// mixing patches with differing upsampling used to pass, and `postprocess` then shifted the
/// extra channel to an effective 8x. The extra channel is upsampled before the patches stage
/// while the color channels are upsampled after it, so the patches stage (which uses both)
/// saw channels at two different resolutions and tripped an assertion in the low-memory
/// pipeline.
#[test]
fn test_fuzzer_patches_ec_upsampling_dim_shift() {
    let data = include_bytes!("../../tests/testdata/patches_ec_upsampling_dim_shift.jxl");
    let result = decode::<f32>(data, Default::default());
    assert!(
        matches!(result, Err(Error::PatchesUnsupportedMixedUpsampling(..))),
        "expected a mixed upsampling error, got {:?}",
        result.map(|_| "a decoded image")
    );
}

/// Regression test: a Modular stream that disables LZ77, but whose pixel histogram codes the
/// constant symbol 1 with a split-exponent-zero uint config -- the shape `Histograms::is_rle()`
/// used to accept, since without LZ77 it inspected cluster 0 instead of the (nonexistent)
/// distance cluster. Together with a single Gradient leaf and prefix codes, that made
/// `decode_modular_subbitstream()` take the RLE fast path, where `decode_fast_lossless()`
/// unwraps the LZ77 parameters -- all `None` here -- and panicked. The stream is valid and
/// decodes on the normal path, so it must decode rather than merely not panic.
#[test]
fn test_fuzzer_modular_rle_fast_path_without_lz77() {
    let data = include_bytes!("../../tests/testdata/modular_rle_fast_path_without_lz77.jxl");
    let frames = decode::<f32>(data, Default::default()).unwrap();
    assert_eq!(frames.len(), 1);
    // A single 8x8 frame, with its three colour channels interleaved.
    assert_eq!(frames[0][0].size(), (3 * 8, 8));
    // Streaming input with flushing exercises the low-memory pipeline as well.
    decode::<f32>(
        data,
        DecodeParams {
            chunk_size: 1,
            do_flush: true,
            ..Default::default()
        },
    )
    .unwrap();
}

/// The other direction: a stream that is genuinely RLE-coded (LZ77 enabled, every copy at
/// distance 1) still takes the fast path. It codes the same image as
/// `modular_rle_fast_path_without_lz77.jxl`, so the two must decode to the same pixels.
#[test]
fn test_modular_rle_fast_path() {
    let data = include_bytes!("../../tests/testdata/modular_rle_fast_path.jxl");
    let frames = decode(data, Default::default()).unwrap();
    let no_lz77 = include_bytes!("../../tests/testdata/modular_rle_fast_path_without_lz77.jxl");
    let no_lz77_frames = decode(no_lz77, Default::default()).unwrap();
    compare_frames(
        Path::new("modular_rle_fast_path.jxl"),
        0,
        &frames[0],
        &no_lz77_frames[0],
    );
}

/// Decoding with `adjust_orientation: false` must output pixels in
/// codestream order and report the codestream size in the basic info;
/// re-applying the orientation must reproduce the default (oriented) output.
#[test]
fn test_adjust_orientation_disabled() {
    use crate::headers::Orientation;

    let files = [
        ("orientation1_identity.jxl", Orientation::Identity),
        (
            "orientation2_flip_horizontal.jxl",
            Orientation::FlipHorizontal,
        ),
        ("orientation3_rotate_180.jxl", Orientation::Rotate180),
        ("orientation4_flip_vertical.jxl", Orientation::FlipVertical),
        ("orientation5_transpose.jxl", Orientation::Transpose),
        ("orientation6_rotate_90_cw.jxl", Orientation::Rotate90Cw),
        (
            "orientation7_anti_transpose.jxl",
            Orientation::AntiTranspose,
        ),
        ("orientation8_rotate_90_ccw.jxl", Orientation::Rotate90Ccw),
    ];
    const NUM_SAMPLES: usize = 4;

    for (name, orientation) in files {
        let file = std::fs::read(format!("resources/test/{name}")).unwrap();
        for use_simple in [true, false] {
            let oriented_frames = decode::<f32>(
                &file,
                DecodeParams {
                    pixel_format: Some(JxlPixelFormat::rgba_f32(0)),
                    use_simple_pipeline: use_simple,
                    ..Default::default()
                },
            )
            .unwrap();
            let raw_frames = decode::<f32>(
                &file,
                DecodeParams {
                    pixel_format: Some(JxlPixelFormat::rgba_f32(0)),
                    use_simple_pipeline: use_simple,
                    adjust_orientation: false,
                    ..Default::default()
                },
            )
            .unwrap();

            let oriented = &oriented_frames[0][0];
            let raw = &raw_frames[0][0];
            let (ow, oh) = (oriented.size().0 / NUM_SAMPLES, oriented.size().1);
            let (rw, rh) = (raw.size().0 / NUM_SAMPLES, raw.size().1);
            assert_eq!((ow, oh), orientation.map_size((rw, rh)), "{name}");

            for y in 0..rh {
                for x in 0..rw {
                    let (dx, dy) = orientation.display_pixel((x, y), (rw, rh));
                    for s in 0..NUM_SAMPLES {
                        assert_eq!(
                            raw.row(y)[x * NUM_SAMPLES + s],
                            oriented.row(dy)[dx * NUM_SAMPLES + s],
                            "{name} mismatch at ({x},{y}) sample {s} (use_simple={use_simple})"
                        );
                    }
                }
            }
        }
    }
}

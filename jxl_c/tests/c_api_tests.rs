// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ffi::c_void;
use std::path::PathBuf;

use jxl::api::{
    AlphaMode, Endianness, Event, JxlColorType, JxlDataFormat, JxlDecoder as RsDecoder,
    JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat as RsPixelFormat,
};
use jxl_c::*;

fn test_resource(rel: &str) -> Vec<u8> {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("../jxl/resources/test");
    path.push(rel);
    std::fs::read(&path).unwrap_or_else(|e| panic!("failed to read {}: {}", path.display(), e))
}

#[test]
fn test_version_and_signature() {
    assert!(JxlDecoderVersion() > 0);
    assert_eq!(JxlDecoderVersion(), JxlEncoderVersion());

    let codestream = [0xffu8, 0x0a, 0x00, 0x00];
    let container = [
        0x00u8, 0x00, 0x00, 0x0c, b'J', b'X', b'L', b' ', 0x0d, 0x0a, 0x87, 0x0a,
    ];
    let invalid = [0x89u8, b'P', b'N', b'G'];

    // SAFETY: Passing valid slices and lengths to `JxlSignatureCheck`.
    unsafe {
        assert_eq!(
            JxlSignatureCheck(std::ptr::null(), 0),
            JXL_SIG_NOT_ENOUGH_BYTES
        );
        assert_eq!(
            JxlSignatureCheck(codestream.as_ptr(), 1),
            JXL_SIG_NOT_ENOUGH_BYTES
        );
        assert_eq!(
            JxlSignatureCheck(codestream.as_ptr(), codestream.len()),
            JXL_SIG_CODESTREAM
        );
        assert_eq!(
            JxlSignatureCheck(container.as_ptr(), 4),
            JXL_SIG_NOT_ENOUGH_BYTES
        );
        assert_eq!(
            JxlSignatureCheck(container.as_ptr(), container.len()),
            JXL_SIG_CONTAINER
        );
        assert_eq!(
            JxlSignatureCheck(invalid.as_ptr(), invalid.len()),
            JXL_SIG_INVALID
        );
    }
}

#[test]
fn test_basic_info_and_color_profiles() {
    let data = test_resource("basic.jxl");

    // SAFETY: Exercising standard C-ABI decoder lifecycle with valid pointers.
    unsafe {
        let mut dec = JxlDecoderCreate(std::ptr::null());
        assert!(dec.is_some());

        let events = (JXL_DEC_BASIC_INFO.0 as i32) | (JXL_DEC_COLOR_ENCODING.0 as i32);
        assert_eq!(
            JxlDecoderSubscribeEvents(dec.as_deref_mut(), events),
            JXL_DEC_SUCCESS
        );
        assert_eq!(
            JxlDecoderSetInput(dec.as_deref_mut(), data.as_ptr(), data.len()),
            JXL_DEC_SUCCESS
        );
        JxlDecoderCloseInput(dec.as_deref_mut());

        assert_eq!(
            JxlDecoderProcessInput(dec.as_deref_mut()),
            JXL_DEC_BASIC_INFO
        );
        // NULL output pointer is allowed by libjxl to check availability
        assert_eq!(
            JxlDecoderGetBasicInfo(dec.as_deref(), std::ptr::null_mut()),
            JXL_DEC_SUCCESS
        );
        let mut info = std::mem::MaybeUninit::<JxlBasicInfo>::uninit();
        assert_eq!(
            JxlDecoderGetBasicInfo(dec.as_deref(), info.as_mut_ptr()),
            JXL_DEC_SUCCESS
        );
        let info = info.assume_init();
        assert!(info.xsize > 0 && info.ysize > 0);
        assert_eq!(info.num_color_channels, 3);
        assert_eq!(info.have_preview, JXL_FALSE);
        assert_eq!(info.orientation, JXL_ORIENT_IDENTITY);

        assert_eq!(
            JxlDecoderProcessInput(dec.as_deref_mut()),
            JXL_DEC_COLOR_ENCODING
        );
        let mut enc = std::mem::MaybeUninit::<JxlColorEncoding>::uninit();
        assert_eq!(
            JxlDecoderGetColorAsEncodedProfile(
                dec.as_deref(),
                JXL_COLOR_PROFILE_TARGET_DATA,
                enc.as_mut_ptr(),
            ),
            JXL_DEC_SUCCESS
        );
        let enc = enc.assume_init();
        assert_eq!(enc.color_space, JXL_COLOR_SPACE_RGB);
        assert_eq!(enc.transfer_function, JXL_TRANSFER_FUNCTION_SRGB);

        let mut srgb = std::mem::MaybeUninit::<JxlColorEncoding>::uninit();
        JxlColorEncodingSetToSRGB(srgb.as_mut_ptr(), JXL_FALSE);
        assert_eq!(enc, srgb.assume_init());

        let mut icc_size = 0usize;
        assert_eq!(
            JxlDecoderGetICCProfileSize(
                dec.as_deref(),
                JXL_COLOR_PROFILE_TARGET_DATA,
                &mut icc_size
            ),
            JXL_DEC_SUCCESS
        );
        assert!(icc_size > 128);
        let mut icc_buf = vec![0u8; icc_size];
        assert_eq!(
            JxlDecoderGetColorAsICCProfile(
                dec.as_deref(),
                JXL_COLOR_PROFILE_TARGET_DATA,
                icc_buf.as_mut_ptr(),
                icc_buf.len(),
            ),
            JXL_DEC_SUCCESS
        );
        assert_eq!(&icc_buf[36..40], b"acsp");

        // Since neither JXL_DEC_FRAME nor JXL_DEC_FULL_IMAGE was subscribed, the next
        // call to JxlDecoderProcessInput must return JXL_DEC_SUCCESS immediately.
        assert_eq!(JxlDecoderProcessInput(dec.as_deref_mut()), JXL_DEC_SUCCESS);

        let _ = JxlDecoderReleaseInput(dec.as_deref_mut());
        JxlDecoderDestroy(dec);
    }
}

#[test]
fn test_embedded_icc_profile() {
    let data = test_resource("lossy_with_icc.jxl");

    // SAFETY: Exercising C-ABI ICC profile queries with valid buffers.
    unsafe {
        let mut dec = JxlDecoderCreate(std::ptr::null());
        let events = (JXL_DEC_BASIC_INFO.0 as i32) | (JXL_DEC_COLOR_ENCODING.0 as i32);
        assert_eq!(
            JxlDecoderSubscribeEvents(dec.as_deref_mut(), events),
            JXL_DEC_SUCCESS
        );
        assert_eq!(
            JxlDecoderSetInput(dec.as_deref_mut(), data.as_ptr(), data.len()),
            JXL_DEC_SUCCESS
        );
        JxlDecoderCloseInput(dec.as_deref_mut());

        assert_eq!(
            JxlDecoderProcessInput(dec.as_deref_mut()),
            JXL_DEC_BASIC_INFO
        );
        assert_eq!(
            JxlDecoderProcessInput(dec.as_deref_mut()),
            JXL_DEC_COLOR_ENCODING
        );

        // Embedded profile is ICC, so JxlDecoderGetColorAsEncodedProfile for ORIGINAL should fail,
        // while JxlDecoderGetICCProfileSize for ORIGINAL should succeed.
        let mut enc = std::mem::MaybeUninit::<JxlColorEncoding>::uninit();
        assert_eq!(
            JxlDecoderGetColorAsEncodedProfile(
                dec.as_deref(),
                JXL_COLOR_PROFILE_TARGET_ORIGINAL,
                enc.as_mut_ptr(),
            ),
            JXL_DEC_ERROR
        );
        let mut orig_icc_size = 0usize;
        assert_eq!(
            JxlDecoderGetICCProfileSize(
                dec.as_deref(),
                JXL_COLOR_PROFILE_TARGET_ORIGINAL,
                &mut orig_icc_size,
            ),
            JXL_DEC_SUCCESS
        );
        assert!(orig_icc_size > 0);
        let mut orig_icc = vec![0u8; orig_icc_size];
        assert_eq!(
            JxlDecoderGetColorAsICCProfile(
                dec.as_deref(),
                JXL_COLOR_PROFILE_TARGET_ORIGINAL,
                orig_icc.as_mut_ptr(),
                orig_icc.len(),
            ),
            JXL_DEC_SUCCESS
        );

        // Requesting linear sRGB as preferred profile on XYB image should succeed.
        let mut lin_srgb = std::mem::MaybeUninit::<JxlColorEncoding>::uninit();
        JxlColorEncodingSetToLinearSRGB(lin_srgb.as_mut_ptr(), JXL_FALSE);
        let lin_srgb = lin_srgb.assume_init();
        assert_eq!(
            JxlDecoderSetPreferredColorProfile(dec.as_deref_mut(), &lin_srgb),
            JXL_DEC_SUCCESS
        );
        let mut data_enc = std::mem::MaybeUninit::<JxlColorEncoding>::uninit();
        assert_eq!(
            JxlDecoderGetColorAsEncodedProfile(
                dec.as_deref(),
                JXL_COLOR_PROFILE_TARGET_DATA,
                data_enc.as_mut_ptr(),
            ),
            JXL_DEC_SUCCESS
        );
        assert_eq!(data_enc.assume_init(), lin_srgb);

        JxlDecoderDestroy(dec);
    }
}

fn decode_with_c_api(data: &[u8], format: JxlPixelFormat, subscribe_frame: bool) -> Vec<u8> {
    // SAFETY: Valid C-ABI calls with properly sized output buffer.
    unsafe {
        let mut dec = JxlDecoderCreate(std::ptr::null());
        let mut events = (JXL_DEC_BASIC_INFO.0 as i32) | (JXL_DEC_FULL_IMAGE.0 as i32);
        if subscribe_frame {
            events |= JXL_DEC_FRAME.0 as i32;
        }
        assert_eq!(
            JxlDecoderSubscribeEvents(dec.as_deref_mut(), events),
            JXL_DEC_SUCCESS
        );
        assert_eq!(
            JxlDecoderSetInput(dec.as_deref_mut(), data.as_ptr(), data.len()),
            JXL_DEC_SUCCESS
        );
        JxlDecoderCloseInput(dec.as_deref_mut());

        assert_eq!(
            JxlDecoderProcessInput(dec.as_deref_mut()),
            JXL_DEC_BASIC_INFO
        );
        if subscribe_frame {
            assert_eq!(JxlDecoderProcessInput(dec.as_deref_mut()), JXL_DEC_FRAME);
        }
        assert_eq!(
            JxlDecoderProcessInput(dec.as_deref_mut()),
            JXL_DEC_NEED_IMAGE_OUT_BUFFER
        );

        let mut buf_size = 0usize;
        assert_eq!(
            JxlDecoderImageOutBufferSize(dec.as_deref(), &format, &mut buf_size),
            JXL_DEC_SUCCESS
        );
        let mut out = vec![0u8; buf_size];
        assert_eq!(
            JxlDecoderSetImageOutBuffer(
                dec.as_deref_mut(),
                &format,
                out.as_mut_ptr().cast::<c_void>(),
                out.len(),
            ),
            JXL_DEC_SUCCESS
        );

        assert_eq!(
            JxlDecoderProcessInput(dec.as_deref_mut()),
            JXL_DEC_FULL_IMAGE
        );
        assert_eq!(JxlDecoderProcessInput(dec.as_deref_mut()), JXL_DEC_SUCCESS);

        JxlDecoderDestroy(dec);
        out
    }
}

fn decode_with_rs_api(data: &[u8], rs_format: RsPixelFormat) -> Vec<u8> {
    let mut dec = RsDecoder::new(JxlDecoderOptions::default());
    let mut input = data;
    assert_eq!(
        dec.process(&mut input, None, None).unwrap(),
        Event::BasicInfo
    );
    dec.set_pixel_format(rs_format.clone()).unwrap();
    let (w, h) = dec.basic_info().unwrap().size;
    let bps = rs_format.color_data_format.unwrap().bytes_per_sample();
    let ch = rs_format.color_type.samples_per_pixel();
    let bytes_per_row = w * ch * bps;
    let mut out = vec![0u8; h * bytes_per_row];
    assert_eq!(
        dec.process(&mut input, None, None).unwrap(),
        Event::FrameHeader
    );
    let mut bufs = [JxlOutputBuffer::new(&mut out, h, bytes_per_row)];
    assert!(matches!(
        dec.process(&mut input, Some(&mut bufs), None).unwrap(),
        Event::FrameComplete { .. }
    ));
    out
}

#[test]
fn test_pixel_formats_match_native_jxl_rs() {
    let data = test_resource("3x3_srgb_lossless.jxl");

    let cases = [
        (
            JxlPixelFormat {
                num_channels: 3,
                data_type: JXL_TYPE_UINT8,
                endianness: JXL_NATIVE_ENDIAN,
                align: 0,
            },
            RsPixelFormat::rgb8(0),
        ),
        (
            JxlPixelFormat {
                num_channels: 4,
                data_type: JXL_TYPE_UINT8,
                endianness: JXL_NATIVE_ENDIAN,
                align: 0,
            },
            RsPixelFormat::rgba8(0),
        ),
        (
            JxlPixelFormat {
                num_channels: 3,
                data_type: JXL_TYPE_UINT16,
                endianness: JXL_LITTLE_ENDIAN,
                align: 0,
            },
            RsPixelFormat {
                color_type: JxlColorType::Rgb,
                color_data_format: Some(JxlDataFormat::U16 {
                    endianness: Endianness::LittleEndian,
                    bit_depth: 16,
                }),
                extra_channel_format: vec![],
            },
        ),
        (
            JxlPixelFormat {
                num_channels: 3,
                data_type: JXL_TYPE_UINT16,
                endianness: JXL_BIG_ENDIAN,
                align: 0,
            },
            RsPixelFormat {
                color_type: JxlColorType::Rgb,
                color_data_format: Some(JxlDataFormat::U16 {
                    endianness: Endianness::BigEndian,
                    bit_depth: 16,
                }),
                extra_channel_format: vec![],
            },
        ),
        (
            JxlPixelFormat {
                num_channels: 4,
                data_type: JXL_TYPE_FLOAT16,
                endianness: JXL_LITTLE_ENDIAN,
                align: 0,
            },
            RsPixelFormat {
                color_type: JxlColorType::Rgba,
                color_data_format: Some(JxlDataFormat::F16 {
                    endianness: Endianness::LittleEndian,
                }),
                extra_channel_format: vec![],
            },
        ),
        (
            JxlPixelFormat {
                num_channels: 3,
                data_type: JXL_TYPE_FLOAT,
                endianness: JXL_NATIVE_ENDIAN,
                align: 0,
            },
            RsPixelFormat::rgb_f32(0),
        ),
    ];

    for (c_fmt, rs_fmt) in cases {
        let expected = decode_with_rs_api(&data, rs_fmt);
        for subscribe_frame in [false, true] {
            let actual = decode_with_c_api(&data, c_fmt, subscribe_frame);
            assert_eq!(
                actual, expected,
                "mismatch for {:?} (subscribe_frame={})",
                c_fmt, subscribe_frame
            );
        }
    }
}

#[test]
fn test_unaligned_buffer_and_row_stride() {
    let data = test_resource("3x3_srgb_lossless.jxl");
    let fmt_packed = JxlPixelFormat {
        num_channels: 3,
        data_type: JXL_TYPE_FLOAT,
        endianness: JXL_NATIVE_ENDIAN,
        align: 0,
    };
    let expected = decode_with_c_api(&data, fmt_packed, true);

    // Test with 1-byte misaligned destination pointer and 17-byte row alignment.
    let fmt_strided = JxlPixelFormat {
        num_channels: 3,
        data_type: JXL_TYPE_FLOAT,
        endianness: JXL_NATIVE_ENDIAN,
        align: 17,
    };
    // SAFETY: Testing unaligned destination buffer pointer with valid capacity.
    unsafe {
        let mut dec = JxlDecoderCreate(std::ptr::null());
        assert_eq!(
            JxlDecoderSubscribeEvents(dec.as_deref_mut(), JXL_DEC_FULL_IMAGE.0 as i32),
            JXL_DEC_SUCCESS
        );
        assert_eq!(
            JxlDecoderSetInput(dec.as_deref_mut(), data.as_ptr(), data.len()),
            JXL_DEC_SUCCESS
        );
        JxlDecoderCloseInput(dec.as_deref_mut());
        assert_eq!(
            JxlDecoderProcessInput(dec.as_deref_mut()),
            JXL_DEC_NEED_IMAGE_OUT_BUFFER
        );
        let mut size = 0usize;
        assert_eq!(
            JxlDecoderImageOutBufferSize(dec.as_deref(), &fmt_strided, &mut size),
            JXL_DEC_SUCCESS
        );
        let bytes_per_row = 3 * 3 * 4; // 36 bytes
        let stride = 51; // next multiple of 17 >= 36
        assert_eq!(size, 2 * stride + bytes_per_row);

        let mut backing = vec![0u8; size + 8];
        let unaligned_ptr = backing.as_mut_ptr().add(1);
        assert_eq!(
            JxlDecoderSetImageOutBuffer(
                dec.as_deref_mut(),
                &fmt_strided,
                unaligned_ptr.cast::<c_void>(),
                size,
            ),
            JXL_DEC_SUCCESS
        );
        assert_eq!(
            JxlDecoderProcessInput(dec.as_deref_mut()),
            JXL_DEC_FULL_IMAGE
        );
        assert_eq!(JxlDecoderProcessInput(dec.as_deref_mut()), JXL_DEC_SUCCESS);

        for y in 0..3 {
            let row_slice =
                std::slice::from_raw_parts(unaligned_ptr.add(y * stride), bytes_per_row);
            assert_eq!(
                row_slice,
                &expected[y * bytes_per_row..(y + 1) * bytes_per_row]
            );
        }
        JxlDecoderDestroy(dec);
    }
}

#[test]
fn test_preview_frame_decoding() {
    let data = test_resource("with_preview.jxl");

    // SAFETY: Exercising preview + main frame decoding with different pixel formats.
    unsafe {
        let mut dec = JxlDecoderCreate(std::ptr::null());
        let events = (JXL_DEC_BASIC_INFO.0 as i32)
            | (JXL_DEC_PREVIEW_IMAGE.0 as i32)
            | (JXL_DEC_FRAME.0 as i32)
            | (JXL_DEC_FULL_IMAGE.0 as i32);
        assert_eq!(
            JxlDecoderSubscribeEvents(dec.as_deref_mut(), events),
            JXL_DEC_SUCCESS
        );
        assert_eq!(
            JxlDecoderSetInput(dec.as_deref_mut(), data.as_ptr(), data.len()),
            JXL_DEC_SUCCESS
        );
        JxlDecoderCloseInput(dec.as_deref_mut());

        assert_eq!(
            JxlDecoderProcessInput(dec.as_deref_mut()),
            JXL_DEC_BASIC_INFO
        );
        let mut info = std::mem::MaybeUninit::<JxlBasicInfo>::uninit();
        assert_eq!(
            JxlDecoderGetBasicInfo(dec.as_deref(), info.as_mut_ptr()),
            JXL_DEC_SUCCESS
        );
        let info = info.assume_init();
        assert_eq!(info.have_preview, JXL_TRUE);
        assert_eq!(info.preview.xsize, 16);
        assert_eq!(info.preview.ysize, 16);

        assert_eq!(
            JxlDecoderProcessInput(dec.as_deref_mut()),
            JXL_DEC_NEED_PREVIEW_OUT_BUFFER
        );
        let preview_fmt = JxlPixelFormat {
            num_channels: 3,
            data_type: JXL_TYPE_UINT8,
            endianness: JXL_NATIVE_ENDIAN,
            align: 0,
        };
        let mut preview_size = 0usize;
        assert_eq!(
            JxlDecoderPreviewOutBufferSize(dec.as_deref(), &preview_fmt, &mut preview_size),
            JXL_DEC_SUCCESS
        );
        assert_eq!(preview_size, 16 * 16 * 3);
        let mut preview_buf = vec![0u8; preview_size];
        assert_eq!(
            JxlDecoderSetPreviewOutBuffer(
                dec.as_deref_mut(),
                &preview_fmt,
                preview_buf.as_mut_ptr().cast::<c_void>(),
                preview_buf.len(),
            ),
            JXL_DEC_SUCCESS
        );

        assert_eq!(
            JxlDecoderProcessInput(dec.as_deref_mut()),
            JXL_DEC_PREVIEW_IMAGE
        );
        assert!(preview_buf.iter().any(|&b| b != 0));

        // Now decode main frame in RGBA16 (different format from preview!)
        assert_eq!(JxlDecoderProcessInput(dec.as_deref_mut()), JXL_DEC_FRAME);
        assert_eq!(
            JxlDecoderProcessInput(dec.as_deref_mut()),
            JXL_DEC_NEED_IMAGE_OUT_BUFFER
        );
        let main_fmt = JxlPixelFormat {
            num_channels: 4,
            data_type: JXL_TYPE_UINT16,
            endianness: JXL_NATIVE_ENDIAN,
            align: 0,
        };
        let mut main_size = 0usize;
        assert_eq!(
            JxlDecoderImageOutBufferSize(dec.as_deref(), &main_fmt, &mut main_size),
            JXL_DEC_SUCCESS
        );
        assert_eq!(main_size, (info.xsize as usize) * (info.ysize as usize) * 8);
        let mut main_buf = vec![0u8; main_size];
        assert_eq!(
            JxlDecoderSetImageOutBuffer(
                dec.as_deref_mut(),
                &main_fmt,
                main_buf.as_mut_ptr().cast::<c_void>(),
                main_buf.len(),
            ),
            JXL_DEC_SUCCESS
        );
        assert_eq!(
            JxlDecoderProcessInput(dec.as_deref_mut()),
            JXL_DEC_FULL_IMAGE
        );
        assert_eq!(JxlDecoderProcessInput(dec.as_deref_mut()), JXL_DEC_SUCCESS);

        let expected_main = decode_with_rs_api(
            &data,
            RsPixelFormat::rgba16(info.num_extra_channels as usize),
        );
        assert_eq!(main_buf, expected_main);

        JxlDecoderDestroy(dec);
    }
}

#[test]
fn test_extra_channels_and_unpremultiply_alpha() {
    let data = test_resource("3x3a_srgb_lossless.jxl");

    // SAFETY: Testing extra channel info, planar extra channel buffer, and unpremultiply alpha.
    unsafe {
        let mut dec = JxlDecoderCreate(std::ptr::null());
        let events = (JXL_DEC_BASIC_INFO.0 as i32) | (JXL_DEC_FULL_IMAGE.0 as i32);
        assert_eq!(
            JxlDecoderSubscribeEvents(dec.as_deref_mut(), events),
            JXL_DEC_SUCCESS
        );
        assert_eq!(
            JxlDecoderSetUnpremultiplyAlpha(dec.as_deref_mut(), JXL_TRUE),
            JXL_DEC_SUCCESS
        );
        assert_eq!(
            JxlDecoderSetInput(dec.as_deref_mut(), data.as_ptr(), data.len()),
            JXL_DEC_SUCCESS
        );
        JxlDecoderCloseInput(dec.as_deref_mut());

        assert_eq!(
            JxlDecoderProcessInput(dec.as_deref_mut()),
            JXL_DEC_BASIC_INFO
        );
        let mut info = std::mem::MaybeUninit::<JxlBasicInfo>::uninit();
        assert_eq!(
            JxlDecoderGetBasicInfo(dec.as_deref(), info.as_mut_ptr()),
            JXL_DEC_SUCCESS
        );
        let info = info.assume_init();
        assert_eq!(info.num_extra_channels, 1);
        assert!(info.alpha_bits > 0);

        let mut ec_info = std::mem::MaybeUninit::<JxlExtraChannelInfo>::uninit();
        assert_eq!(
            JxlDecoderGetExtraChannelInfo(dec.as_deref(), 0, ec_info.as_mut_ptr()),
            JXL_DEC_SUCCESS
        );
        let ec_info = ec_info.assume_init();
        assert_eq!(ec_info.type_, JXL_CHANNEL_ALPHA);

        assert_eq!(
            JxlDecoderProcessInput(dec.as_deref_mut()),
            JXL_DEC_NEED_IMAGE_OUT_BUFFER
        );

        // Reject 1-channel format on RGB image
        let invalid_gray_fmt = JxlPixelFormat {
            num_channels: 1,
            data_type: JXL_TYPE_UINT8,
            endianness: JXL_NATIVE_ENDIAN,
            align: 0,
        };
        let mut dummy_size = 0usize;
        assert_eq!(
            JxlDecoderImageOutBufferSize(dec.as_deref(), &invalid_gray_fmt, &mut dummy_size),
            JXL_DEC_ERROR
        );

        let color_fmt = JxlPixelFormat {
            num_channels: 3,
            data_type: JXL_TYPE_UINT8,
            endianness: JXL_NATIVE_ENDIAN,
            align: 0,
        };
        let ec_fmt = JxlPixelFormat {
            num_channels: 1,
            data_type: JXL_TYPE_UINT8,
            endianness: JXL_NATIVE_ENDIAN,
            align: 0,
        };
        let mut color_size = 0usize;
        let mut ec_size = 0usize;
        assert_eq!(
            JxlDecoderImageOutBufferSize(dec.as_deref(), &color_fmt, &mut color_size),
            JXL_DEC_SUCCESS
        );
        assert_eq!(
            JxlDecoderExtraChannelBufferSize(dec.as_deref(), &ec_fmt, &mut ec_size, 0),
            JXL_DEC_SUCCESS
        );
        assert_eq!(color_size, 3 * 3 * 3);
        assert_eq!(ec_size, 3 * 3);

        let mut color_buf = vec![0u8; color_size];
        let mut ec_buf = vec![0u8; ec_size];
        // Setting extra channel buffer before main image buffer must still require main buffer!
        assert_eq!(
            JxlDecoderSetExtraChannelBuffer(
                dec.as_deref_mut(),
                &ec_fmt,
                ec_buf.as_mut_ptr().cast::<c_void>(),
                ec_buf.len(),
                0,
            ),
            JXL_DEC_SUCCESS
        );
        assert_eq!(
            JxlDecoderProcessInput(dec.as_deref_mut()),
            JXL_DEC_NEED_IMAGE_OUT_BUFFER
        );
        assert_eq!(
            JxlDecoderSetImageOutBuffer(
                dec.as_deref_mut(),
                &color_fmt,
                color_buf.as_mut_ptr().cast::<c_void>(),
                color_buf.len(),
            ),
            JXL_DEC_SUCCESS
        );

        assert_eq!(
            JxlDecoderProcessInput(dec.as_deref_mut()),
            JXL_DEC_FULL_IMAGE
        );
        assert_eq!(JxlDecoderProcessInput(dec.as_deref_mut()), JXL_DEC_SUCCESS);

        // Compare against native jxl-rs with AlphaMode::Unpremultiplied
        let mut rs_opts = JxlDecoderOptions::default();
        rs_opts.alpha_mode = AlphaMode::Unpremultiplied;
        let mut rs_dec = RsDecoder::new(rs_opts);
        let mut input = data.as_slice();
        assert_eq!(
            rs_dec.process(&mut input, None, None).unwrap(),
            Event::BasicInfo
        );
        rs_dec
            .set_pixel_format(RsPixelFormat {
                color_type: JxlColorType::Rgb,
                color_data_format: Some(JxlDataFormat::U8 { bit_depth: 8 }),
                extra_channel_format: vec![Some(JxlDataFormat::U8 { bit_depth: 8 })],
            })
            .unwrap();
        assert_eq!(
            rs_dec.process(&mut input, None, None).unwrap(),
            Event::FrameHeader
        );
        let mut expected_color = vec![0u8; color_size];
        let mut expected_ec = vec![0u8; ec_size];
        let mut bufs = [
            JxlOutputBuffer::new(&mut expected_color, 3, 9),
            JxlOutputBuffer::new(&mut expected_ec, 3, 3),
        ];
        assert!(matches!(
            rs_dec.process(&mut input, Some(&mut bufs), None).unwrap(),
            Event::FrameComplete { .. }
        ));
        assert_eq!(color_buf, expected_color);
        assert_eq!(ec_buf, expected_ec);

        JxlDecoderDestroy(dec);
    }
}

struct CallbackCapture {
    width: usize,
    bytes_per_pixel: usize,
    data: Vec<u8>,
}

unsafe extern "C" fn simple_out_callback(
    opaque: *mut c_void,
    x: usize,
    y: usize,
    num_pixels: usize,
    pixels: *const c_void,
) {
    // SAFETY: `opaque` points to our `CallbackCapture` and `pixels` has `num_pixels * bpp` bytes.
    unsafe {
        let cap = &mut *(opaque as *mut CallbackCapture);
        let bpp = cap.bytes_per_pixel;
        let row_offset = (y * cap.width + x) * bpp;
        let byte_count = num_pixels * bpp;
        let src = std::slice::from_raw_parts(pixels.cast::<u8>(), byte_count);
        cap.data[row_offset..row_offset + byte_count].copy_from_slice(src);
    }
}

#[test]
fn test_image_out_callback_and_streaming_chunks() {
    let data = test_resource("basic.jxl");
    let fmt = JxlPixelFormat {
        num_channels: 3,
        data_type: JXL_TYPE_UINT8,
        endianness: JXL_NATIVE_ENDIAN,
        align: 0,
    };
    let expected = decode_with_c_api(&data, fmt, false);

    // SAFETY: Testing streaming input in 5-byte chunks with `JxlDecoderSetImageOutCallback`.
    unsafe {
        let mut dec = JxlDecoderCreate(std::ptr::null());
        let events = (JXL_DEC_BASIC_INFO.0 as i32) | (JXL_DEC_FULL_IMAGE.0 as i32);
        assert_eq!(
            JxlDecoderSubscribeEvents(dec.as_deref_mut(), events),
            JXL_DEC_SUCCESS
        );

        let mut pos = 0usize;
        let chunk_size = 5usize;
        let mut capture = CallbackCapture {
            width: 0,
            bytes_per_pixel: 3,
            data: Vec::new(),
        };
        let mut got_full_image = false;

        loop {
            let status = JxlDecoderProcessInput(dec.as_deref_mut());
            match status {
                JXL_DEC_NEED_MORE_INPUT => {
                    let rem = JxlDecoderReleaseInput(dec.as_deref_mut());
                    pos -= rem;
                    assert!(pos < data.len(), "unexpected need more input at EOF");
                    let end = (pos + chunk_size).min(data.len());
                    assert_eq!(
                        JxlDecoderSetInput(dec.as_deref_mut(), data[pos..end].as_ptr(), end - pos),
                        JXL_DEC_SUCCESS
                    );
                    pos = end;
                    if pos == data.len() {
                        JxlDecoderCloseInput(dec.as_deref_mut());
                    }
                }
                JXL_DEC_BASIC_INFO => {
                    let mut info = std::mem::MaybeUninit::<JxlBasicInfo>::uninit();
                    assert_eq!(
                        JxlDecoderGetBasicInfo(dec.as_deref(), info.as_mut_ptr()),
                        JXL_DEC_SUCCESS
                    );
                    let info = info.assume_init();
                    capture.width = info.xsize as usize;
                    capture.data = vec![0u8; (info.xsize as usize) * (info.ysize as usize) * 3];
                }
                JXL_DEC_NEED_IMAGE_OUT_BUFFER => {
                    assert_eq!(
                        JxlDecoderSetImageOutCallback(
                            dec.as_deref_mut(),
                            &fmt,
                            Some(simple_out_callback),
                            (&raw mut capture).cast::<c_void>(),
                        ),
                        JXL_DEC_SUCCESS
                    );
                }
                JXL_DEC_FULL_IMAGE => {
                    got_full_image = true;
                }
                JXL_DEC_SUCCESS => {
                    break;
                }
                other => panic!("unexpected decoder status: {:?}", other),
            }
        }

        assert!(got_full_image);
        assert_eq!(capture.data, expected);
        JxlDecoderDestroy(dec);
    }
}

#[test]
fn test_animation_skip_frames_and_rewind() {
    let data = test_resource("conformance_test_images/animation_icos4d_5.jxl");

    // SAFETY: Testing multi-frame animation, frame skipping, and decoder rewind.
    unsafe {
        let mut dec = JxlDecoderCreate(std::ptr::null());
        let events = (JXL_DEC_BASIC_INFO.0 as i32)
            | (JXL_DEC_FRAME.0 as i32)
            | (JXL_DEC_FULL_IMAGE.0 as i32);
        assert_eq!(
            JxlDecoderSubscribeEvents(dec.as_deref_mut(), events),
            JXL_DEC_SUCCESS
        );
        assert_eq!(
            JxlDecoderSetInput(dec.as_deref_mut(), data.as_ptr(), data.len()),
            JXL_DEC_SUCCESS
        );
        JxlDecoderCloseInput(dec.as_deref_mut());

        assert_eq!(
            JxlDecoderProcessInput(dec.as_deref_mut()),
            JXL_DEC_BASIC_INFO
        );
        let mut info = std::mem::MaybeUninit::<JxlBasicInfo>::uninit();
        assert_eq!(
            JxlDecoderGetBasicInfo(dec.as_deref(), info.as_mut_ptr()),
            JXL_DEC_SUCCESS
        );
        let info = info.assume_init();
        assert_eq!(info.have_animation, JXL_TRUE);

        let fmt = JxlPixelFormat {
            num_channels: 4,
            data_type: JXL_TYPE_UINT8,
            endianness: JXL_NATIVE_ENDIAN,
            align: 0,
        };
        let mut buf_size = 0usize;
        assert_eq!(
            JxlDecoderImageOutBufferSize(dec.as_deref(), &fmt, &mut buf_size),
            JXL_DEC_SUCCESS
        );
        let mut frame_buf = vec![0u8; buf_size];

        let mut frame_count = 0usize;
        let mut need_buf_count = 0usize;
        loop {
            let st = JxlDecoderProcessInput(dec.as_deref_mut());
            match st {
                JXL_DEC_FRAME => {
                    let mut fh = std::mem::MaybeUninit::<JxlFrameHeader>::uninit();
                    assert_eq!(
                        JxlDecoderGetFrameHeader(dec.as_deref(), fh.as_mut_ptr()),
                        JXL_DEC_SUCCESS
                    );
                    let fh = fh.assume_init();
                    assert_eq!(fh.layer_info.xsize, info.xsize);
                    assert_eq!(fh.layer_info.ysize, info.ysize);
                }
                JXL_DEC_NEED_IMAGE_OUT_BUFFER => {
                    need_buf_count += 1;
                    assert_eq!(
                        JxlDecoderSetImageOutBuffer(
                            dec.as_deref_mut(),
                            &fmt,
                            frame_buf.as_mut_ptr().cast::<c_void>(),
                            frame_buf.len(),
                        ),
                        JXL_DEC_SUCCESS
                    );
                }
                JXL_DEC_FULL_IMAGE => {
                    frame_count += 1;
                }
                JXL_DEC_SUCCESS => break,
                other => panic!("unexpected status: {:?}", other),
            }
        }
        assert!(frame_count > 1);
        // Per-frame buffers must be requested before every frame!
        assert_eq!(need_buf_count, frame_count);

        // Rewind and skip the first 2 frames
        JxlDecoderRewind(dec.as_deref_mut());
        JxlDecoderSkipFrames(dec.as_deref_mut(), 2);
        assert_eq!(
            JxlDecoderSetInput(dec.as_deref_mut(), data.as_ptr(), data.len()),
            JXL_DEC_SUCCESS
        );
        JxlDecoderCloseInput(dec.as_deref_mut());
        assert_eq!(
            JxlDecoderProcessInput(dec.as_deref_mut()),
            JXL_DEC_BASIC_INFO
        );

        let mut after_skip_count = 0usize;
        loop {
            match JxlDecoderProcessInput(dec.as_deref_mut()) {
                JXL_DEC_FRAME => {}
                JXL_DEC_NEED_IMAGE_OUT_BUFFER => {
                    assert_eq!(
                        JxlDecoderSetImageOutBuffer(
                            dec.as_deref_mut(),
                            &fmt,
                            frame_buf.as_mut_ptr().cast::<c_void>(),
                            frame_buf.len(),
                        ),
                        JXL_DEC_SUCCESS
                    );
                }
                JXL_DEC_FULL_IMAGE => after_skip_count += 1,
                JXL_DEC_SUCCESS => break,
                other => panic!("unexpected status after skip: {:?}", other),
            }
        }
        assert_eq!(after_skip_count, frame_count - 2);

        JxlDecoderDestroy(dec);
    }
}

#[test]
fn test_thread_and_resizable_parallel_runners() {
    let data = test_resource("basic.jxl");
    let fmt = JxlPixelFormat {
        num_channels: 3,
        data_type: JXL_TYPE_UINT8,
        endianness: JXL_NATIVE_ENDIAN,
        align: 0,
    };
    let expected = decode_with_c_api(&data, fmt, false);

    // SAFETY: Testing `JxlThreadParallelRunner` and `JxlResizableParallelRunner` with decoder.
    unsafe {
        assert!(JxlThreadParallelRunnerDefaultNumWorkerThreads() >= 1);
        let runner = JxlThreadParallelRunnerCreate(std::ptr::null(), 2);
        assert!(!runner.is_null());

        let mut dec = JxlDecoderCreate(std::ptr::null());
        assert_eq!(
            JxlDecoderSetParallelRunner(dec.as_deref_mut(), Some(JxlThreadParallelRunner), runner),
            JXL_DEC_SUCCESS
        );
        assert_eq!(
            JxlDecoderSubscribeEvents(dec.as_deref_mut(), JXL_DEC_FULL_IMAGE.0 as i32),
            JXL_DEC_SUCCESS
        );
        assert_eq!(
            JxlDecoderSetInput(dec.as_deref_mut(), data.as_ptr(), data.len()),
            JXL_DEC_SUCCESS
        );
        JxlDecoderCloseInput(dec.as_deref_mut());
        assert_eq!(
            JxlDecoderProcessInput(dec.as_deref_mut()),
            JXL_DEC_NEED_IMAGE_OUT_BUFFER
        );
        let mut out = vec![0u8; expected.len()];
        assert_eq!(
            JxlDecoderSetImageOutBuffer(
                dec.as_deref_mut(),
                &fmt,
                out.as_mut_ptr().cast::<c_void>(),
                out.len()
            ),
            JXL_DEC_SUCCESS
        );
        assert_eq!(
            JxlDecoderProcessInput(dec.as_deref_mut()),
            JXL_DEC_FULL_IMAGE
        );
        assert_eq!(out, expected);

        JxlDecoderDestroy(dec);
        JxlThreadParallelRunnerDestroy(runner);

        let r_runner = JxlResizableParallelRunnerCreate(std::ptr::null());
        assert!(!r_runner.is_null());
        let suggested = JxlResizableParallelRunnerSuggestThreads(512, 512);
        assert!(suggested >= 1);
        JxlResizableParallelRunnerSetThreads(r_runner, suggested as usize);
        JxlResizableParallelRunnerDestroy(r_runner);
    }
}

#[test]
fn test_gain_map_read_bundle() {
    let structured = include_bytes!("../../jxl-gainmap/tests/testdata/structured.jhgm");
    let icc = include_bytes!("../../jxl-gainmap/tests/testdata/icc.jhgm");
    let malformed = include_bytes!("../../jxl-gainmap/tests/testdata/malformed_color.jhgm");

    // SAFETY: Passing valid buffers and pointers to `JxlGainMapReadBundle`.
    unsafe {
        let mut bundle = std::mem::MaybeUninit::<JxlGainMapBundle>::uninit();
        let mut bytes_read = 0usize;
        assert_eq!(
            JxlGainMapReadBundle(
                bundle.as_mut_ptr(),
                structured.as_ptr(),
                structured.len(),
                &mut bytes_read,
            ),
            JXL_TRUE
        );
        assert_eq!(bytes_read, structured.len());
        let bundle = bundle.assume_init();
        assert_eq!(bundle.jhgm_version, 0);
        assert_eq!(bundle.has_color_encoding, JXL_TRUE);
        assert_eq!(
            bundle.color_encoding.transfer_function,
            JXL_TRANSFER_FUNCTION_LINEAR
        );
        assert_eq!(bundle.alt_icc_size, 0);
        assert!(bundle.gain_map_size > 0);

        let mut bundle_icc = std::mem::MaybeUninit::<JxlGainMapBundle>::uninit();
        assert_eq!(
            JxlGainMapReadBundle(
                bundle_icc.as_mut_ptr(),
                icc.as_ptr(),
                icc.len(),
                std::ptr::null_mut(),
            ),
            JXL_TRUE
        );
        let bundle_icc = bundle_icc.assume_init();
        assert_eq!(bundle_icc.has_color_encoding, JXL_FALSE);
        assert!(bundle_icc.alt_icc_size > 0);

        let mut bundle_bad = std::mem::MaybeUninit::<JxlGainMapBundle>::uninit();
        assert_eq!(
            JxlGainMapReadBundle(
                bundle_bad.as_mut_ptr(),
                malformed.as_ptr(),
                malformed.len(),
                std::ptr::null_mut(),
            ),
            JXL_FALSE
        );
    }
}

#[test]
fn test_unimplemented_features_return_errors() {
    let data = test_resource("basic.jxl");

    // SAFETY: Verifying unimplemented decoder/encoder/gainmap functions and post-process option
    // setters fail cleanly.
    unsafe {
        let mut dec = JxlDecoderCreate(std::ptr::null());
        assert_eq!(
            JxlDecoderSubscribeEvents(dec.as_deref_mut(), JXL_DEC_BOX.0 as i32),
            JXL_DEC_ERROR
        );
        assert_eq!(
            JxlDecoderSubscribeEvents(dec.as_deref_mut(), JXL_DEC_JPEG_RECONSTRUCTION.0 as i32),
            JXL_DEC_ERROR
        );
        assert_eq!(
            JxlDecoderSubscribeEvents(dec.as_deref_mut(), JXL_DEC_FRAME_PROGRESSION.0 as i32),
            JXL_DEC_ERROR
        );
        assert_eq!(
            JxlDecoderSetProgressiveDetail(dec.as_deref_mut(), kFrames),
            JXL_DEC_ERROR
        );
        assert_eq!(
            JxlDecoderSetCms(dec.as_deref_mut(), JxlCmsInterface::default()),
            JXL_DEC_ERROR
        );
        assert_eq!(
            JxlDecoderSetCoalescing(dec.as_deref_mut(), JXL_FALSE),
            JXL_DEC_ERROR
        );
        assert_eq!(
            JxlDecoderSetCoalescing(dec.as_deref_mut(), JXL_TRUE),
            JXL_DEC_SUCCESS
        );
        assert_eq!(
            JxlDecoderSetBoxBuffer(dec.as_deref_mut(), std::ptr::null_mut(), 0),
            JXL_DEC_ERROR
        );
        assert_eq!(
            JxlDecoderSetJPEGBuffer(dec.as_deref_mut(), std::ptr::null_mut(), 0),
            JXL_DEC_ERROR
        );
        assert_eq!(
            JxlDecoderSetDecompressBoxes(dec.as_deref_mut(), JXL_TRUE),
            JXL_DEC_SUCCESS
        );

        // Once ProcessInput has started, decoder-level options return JXL_DEC_ERROR.
        assert_eq!(
            JxlDecoderSubscribeEvents(
                dec.as_deref_mut(),
                (JXL_DEC_BASIC_INFO.0 | JXL_DEC_BOX.0) as i32
            ),
            JXL_DEC_SUCCESS
        );
        assert!(JxlDecoderSizeHintBasicInfo(dec.as_deref()) > 0);
        assert_eq!(
            JxlDecoderSetInput(dec.as_deref_mut(), data.as_ptr(), data.len()),
            JXL_DEC_SUCCESS
        );
        assert_eq!(
            JxlDecoderProcessInput(dec.as_deref_mut()),
            JXL_DEC_BASIC_INFO
        );
        assert_eq!(JxlDecoderSizeHintBasicInfo(dec.as_deref()), 0);
        assert_eq!(
            JxlDecoderSetDecompressBoxes(dec.as_deref_mut(), JXL_TRUE),
            JXL_DEC_ERROR
        );
        assert_eq!(
            JxlDecoderSetUnpremultiplyAlpha(dec.as_deref_mut(), JXL_TRUE),
            JXL_DEC_ERROR
        );
        assert_eq!(
            JxlDecoderSetRenderSpotcolors(dec.as_deref_mut(), JXL_FALSE),
            JXL_DEC_ERROR
        );
        assert_eq!(
            JxlDecoderSetDesiredIntensityTarget(dec.as_deref_mut(), 500.0),
            JXL_DEC_ERROR
        );

        JxlDecoderDestroy(dec);

        assert_eq!(
            JxlGainMapGetBundleSize(std::ptr::null(), std::ptr::null_mut()),
            JXL_FALSE
        );

        let enc = JxlEncoderCreate(std::ptr::null());
        assert!(enc.is_null());
        assert_eq!(
            JxlEncoderGetError(enc.as_mut()),
            JXL_ENC_ERR_NOT_SUPPORTED
        );

        let mut basic_info = std::mem::MaybeUninit::<JxlBasicInfo>::uninit();
        JxlEncoderInitBasicInfo(basic_info.as_mut_ptr());
        let basic_info = basic_info.assume_init();
        assert_eq!(basic_info.bits_per_sample, 8);
        assert_eq!(basic_info.num_color_channels, 3);
        assert_eq!(basic_info.orientation, JXL_ORIENT_IDENTITY);

        let mut fh = std::mem::MaybeUninit::<JxlFrameHeader>::uninit();
        JxlEncoderInitFrameHeader(fh.as_mut_ptr());
        assert_eq!(fh.assume_init().is_last, JXL_TRUE);

        let mut ec_info = std::mem::MaybeUninit::<JxlExtraChannelInfo>::uninit();
        JxlEncoderInitExtraChannelInfo(JXL_CHANNEL_ALPHA, ec_info.as_mut_ptr());
        let ec_info = ec_info.assume_init();
        assert_eq!(ec_info.type_, JXL_CHANNEL_ALPHA);
        assert_eq!(ec_info.bits_per_sample, 8);
    }
}

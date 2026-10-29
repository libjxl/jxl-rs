// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::path::PathBuf;

use jxl::api::{
    AlphaMode, Endianness, Event, JxlColorType, JxlDataFormat, JxlDecoder as RsDecoder,
    JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat as RsPixelFormat,
};
use jxl_libjxl_c_api::*;

fn test_resource(rel: &str) -> Vec<u8> {
    let path = PathBuf::from("../jxl/resources/test").join(rel);
    std::fs::read(&path).unwrap_or_else(|e| panic!("failed to read {}: {}", path.display(), e))
}

fn set_input(dec: Option<&mut JxlDecoder>, data: &[u8]) -> JxlDecoderStatus {
    // SAFETY: `data` is a valid byte slice outliving the decoder step or until released.
    unsafe { JxlDecoderSetInput(dec, data.as_ptr(), data.len()) }
}

fn set_image_out_buffer(
    dec: Option<&mut JxlDecoder>,
    format: &JxlPixelFormat,
    buf: &mut [u8],
) -> JxlDecoderStatus {
    // SAFETY: `buf` is a valid mutable byte slice for the duration of decoding.
    unsafe {
        JxlDecoderSetImageOutBuffer(
            dec,
            Some(format),
            buf.as_mut_ptr().cast::<c_void>(),
            buf.len(),
        )
    }
}

fn set_preview_out_buffer(
    dec: Option<&mut JxlDecoder>,
    format: &JxlPixelFormat,
    buf: &mut [u8],
) -> JxlDecoderStatus {
    // SAFETY: `buf` is a valid mutable byte slice for the duration of preview decoding.
    unsafe {
        JxlDecoderSetPreviewOutBuffer(
            dec,
            Some(format),
            buf.as_mut_ptr().cast::<c_void>(),
            buf.len(),
        )
    }
}

fn set_extra_channel_buffer(
    dec: Option<&mut JxlDecoder>,
    format: &JxlPixelFormat,
    buf: &mut [u8],
    index: u32,
) -> JxlDecoderStatus {
    // SAFETY: `buf` is a valid mutable byte slice for the duration of decoding.
    unsafe {
        JxlDecoderSetExtraChannelBuffer(
            dec,
            Some(format),
            buf.as_mut_ptr().cast::<c_void>(),
            buf.len(),
            index,
        )
    }
}

fn set_box_buffer(dec: Option<&mut JxlDecoder>, buf: &mut [u8]) -> JxlDecoderStatus {
    // SAFETY: `buf` is a valid mutable byte slice until `JxlDecoderReleaseBoxBuffer`.
    unsafe { JxlDecoderSetBoxBuffer(dec, buf.as_mut_ptr(), buf.len()) }
}

fn get_basic_info(dec: Option<&JxlDecoder>) -> JxlBasicInfo {
    let mut info = MaybeUninit::<JxlBasicInfo>::uninit();
    assert_eq!(
        JxlDecoderGetBasicInfo(dec, Some(&mut info)),
        JXL_DEC_SUCCESS
    );
    // SAFETY: `JxlDecoderGetBasicInfo` returned `JXL_DEC_SUCCESS`, initializing `info`.
    unsafe { info.assume_init() }
}

fn get_frame_header(dec: Option<&JxlDecoder>) -> JxlFrameHeader {
    let mut fh = MaybeUninit::<JxlFrameHeader>::uninit();
    assert_eq!(
        JxlDecoderGetFrameHeader(dec, Some(&mut fh)),
        JXL_DEC_SUCCESS
    );
    // SAFETY: `JxlDecoderGetFrameHeader` returned `JXL_DEC_SUCCESS`, initializing `fh`.
    unsafe { fh.assume_init() }
}

fn get_color_encoding(dec: Option<&JxlDecoder>, target: JxlColorProfileTarget) -> JxlColorEncoding {
    let mut enc = MaybeUninit::<JxlColorEncoding>::uninit();
    assert_eq!(
        JxlDecoderGetColorAsEncodedProfile(dec, target, Some(&mut enc)),
        JXL_DEC_SUCCESS
    );
    // SAFETY: `JxlDecoderGetColorAsEncodedProfile` returned `JXL_DEC_SUCCESS`, initializing `enc`.
    unsafe { enc.assume_init() }
}

fn get_icc_profile(dec: Option<&JxlDecoder>, target: JxlColorProfileTarget) -> Vec<u8> {
    let mut size = MaybeUninit::<usize>::uninit();
    assert_eq!(
        JxlDecoderGetICCProfileSize(dec, target, Some(&mut size)),
        JXL_DEC_SUCCESS
    );
    // SAFETY: `JxlDecoderGetICCProfileSize` returned `JXL_DEC_SUCCESS`, initializing `size`.
    let size = unsafe { size.assume_init() };
    assert!(size > 0);
    let mut buf = vec![0u8; size];
    // SAFETY: `buf` has `size` writable bytes as returned by `JxlDecoderGetICCProfileSize`.
    let status =
        unsafe { JxlDecoderGetColorAsICCProfile(dec, target, buf.as_mut_ptr(), buf.len()) };
    assert_eq!(status, JXL_DEC_SUCCESS);
    buf
}

fn get_image_out_buffer_size(dec: Option<&JxlDecoder>, format: &JxlPixelFormat) -> usize {
    let mut size = MaybeUninit::<usize>::uninit();
    assert_eq!(
        JxlDecoderImageOutBufferSize(dec, Some(format), Some(&mut size)),
        JXL_DEC_SUCCESS
    );
    // SAFETY: `JxlDecoderImageOutBufferSize` returned `JXL_DEC_SUCCESS`, initializing `size`.
    unsafe { size.assume_init() }
}

fn get_preview_out_buffer_size(dec: Option<&JxlDecoder>, format: &JxlPixelFormat) -> usize {
    let mut size = MaybeUninit::<usize>::uninit();
    assert_eq!(
        JxlDecoderPreviewOutBufferSize(dec, Some(format), Some(&mut size)),
        JXL_DEC_SUCCESS
    );
    // SAFETY: `JxlDecoderPreviewOutBufferSize` returned `JXL_DEC_SUCCESS`, initializing `size`.
    unsafe { size.assume_init() }
}

fn get_extra_channel_buffer_size(
    dec: Option<&JxlDecoder>,
    format: &JxlPixelFormat,
    index: u32,
) -> usize {
    let mut size = MaybeUninit::<usize>::uninit();
    assert_eq!(
        JxlDecoderExtraChannelBufferSize(dec, Some(format), Some(&mut size), index),
        JXL_DEC_SUCCESS
    );
    // SAFETY: `JxlDecoderExtraChannelBufferSize` returned `JXL_DEC_SUCCESS`, initializing `size`.
    unsafe { size.assume_init() }
}

fn get_box_size_contents(dec: Option<&JxlDecoder>) -> u64 {
    let mut size = MaybeUninit::<u64>::uninit();
    assert_eq!(
        JxlDecoderGetBoxSizeContents(dec, Some(&mut size)),
        JXL_DEC_SUCCESS
    );
    // SAFETY: `JxlDecoderGetBoxSizeContents` returned `JXL_DEC_SUCCESS`, initializing `size`.
    unsafe { size.assume_init() }
}

fn get_box_size_raw(dec: Option<&JxlDecoder>) -> u64 {
    let mut size = MaybeUninit::<u64>::uninit();
    assert_eq!(
        JxlDecoderGetBoxSizeRaw(dec, Some(&mut size)),
        JXL_DEC_SUCCESS
    );
    // SAFETY: `JxlDecoderGetBoxSizeRaw` returned `JXL_DEC_SUCCESS`, initializing `size`.
    unsafe { size.assume_init() }
}

#[allow(clippy::unnecessary_cast)]
fn get_box_type(dec: Option<&mut JxlDecoder>, decompressed: JXL_BOOL) -> [u8; 4] {
    let mut box_type = MaybeUninit::<JxlBoxType>::uninit();
    assert_eq!(
        JxlDecoderGetBoxType(dec, Some(&mut box_type), decompressed),
        JXL_DEC_SUCCESS
    );
    // SAFETY: `JxlDecoderGetBoxType` returned `JXL_DEC_SUCCESS`, initializing `box_type`.
    let raw = unsafe { box_type.assume_init() };
    raw.map(|c| c as u8)
}

fn srgb_encoding(is_gray: JXL_BOOL) -> JxlColorEncoding {
    let mut enc = MaybeUninit::<JxlColorEncoding>::uninit();
    JxlColorEncodingSetToSRGB(Some(&mut enc), is_gray);
    // SAFETY: `JxlColorEncodingSetToSRGB` unconditionally initializes `enc`.
    unsafe { enc.assume_init() }
}

fn linear_srgb_encoding(is_gray: JXL_BOOL) -> JxlColorEncoding {
    let mut enc = MaybeUninit::<JxlColorEncoding>::uninit();
    JxlColorEncodingSetToLinearSRGB(Some(&mut enc), is_gray);
    // SAFETY: `JxlColorEncodingSetToLinearSRGB` unconditionally initializes `enc`.
    unsafe { enc.assume_init() }
}

#[test]
fn version_and_signature() {
    assert!(JxlDecoderVersion() > 0);
    assert_eq!(JxlDecoderVersion(), JxlEncoderVersion());

    let codestream = [0xffu8, 0x0a, 0x00, 0x00];
    let container = [
        0x00u8, 0x00, 0x00, 0x0c, b'J', b'X', b'L', b' ', 0x0d, 0x0a, 0x87, 0x0a,
    ];
    let invalid = [0x89u8, b'P', b'N', b'G'];

    // SAFETY: Passing valid slice pointers and lengths (or null with len 0) to `JxlSignatureCheck`.
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
fn basic_info_and_color_profiles() {
    let data = test_resource("basic.jxl");

    let mut dec = JxlDecoderCreate(None);
    assert!(dec.is_some());

    let events = (JXL_DEC_BASIC_INFO.0 as i32) | (JXL_DEC_COLOR_ENCODING.0 as i32);
    assert_eq!(
        JxlDecoderSubscribeEvents(dec.as_deref_mut(), events),
        JXL_DEC_SUCCESS
    );
    assert_eq!(set_input(dec.as_deref_mut(), &data), JXL_DEC_SUCCESS);
    JxlDecoderCloseInput(dec.as_deref_mut());

    assert_eq!(
        JxlDecoderProcessInput(dec.as_deref_mut()),
        JXL_DEC_BASIC_INFO
    );
    // NULL output pointer is allowed by libjxl to check availability
    assert_eq!(
        JxlDecoderGetBasicInfo(dec.as_deref(), None),
        JXL_DEC_SUCCESS
    );
    let info = get_basic_info(dec.as_deref());
    assert!(info.xsize > 0 && info.ysize > 0);
    assert_eq!(info.num_color_channels, 3);
    assert_eq!(info.have_preview, JXL_FALSE);
    assert_eq!(info.orientation, JXL_ORIENT_IDENTITY);

    assert_eq!(
        JxlDecoderProcessInput(dec.as_deref_mut()),
        JXL_DEC_COLOR_ENCODING
    );
    let enc = get_color_encoding(dec.as_deref(), JXL_COLOR_PROFILE_TARGET_DATA);
    assert_eq!(enc.color_space, JXL_COLOR_SPACE_RGB);
    assert_eq!(enc.transfer_function, JXL_TRANSFER_FUNCTION_SRGB);
    assert_eq!(enc, srgb_encoding(JXL_FALSE));

    let icc_buf = get_icc_profile(dec.as_deref(), JXL_COLOR_PROFILE_TARGET_DATA);
    assert!(icc_buf.len() > 128);
    assert_eq!(&icc_buf[36..40], b"acsp");

    // Since neither JXL_DEC_FRAME nor JXL_DEC_FULL_IMAGE was subscribed, the next
    // call to JxlDecoderProcessInput must return JXL_DEC_SUCCESS immediately.
    assert_eq!(JxlDecoderProcessInput(dec.as_deref_mut()), JXL_DEC_SUCCESS);

    let _ = JxlDecoderReleaseInput(dec.as_deref_mut());
    JxlDecoderDestroy(dec);
}

#[test]
fn embedded_icc_profile() {
    let data = test_resource("lossy_with_icc.jxl");

    let mut dec = JxlDecoderCreate(None);
    let events = (JXL_DEC_BASIC_INFO.0 as i32) | (JXL_DEC_COLOR_ENCODING.0 as i32);
    assert_eq!(
        JxlDecoderSubscribeEvents(dec.as_deref_mut(), events),
        JXL_DEC_SUCCESS
    );
    assert_eq!(set_input(dec.as_deref_mut(), &data), JXL_DEC_SUCCESS);
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
    let mut enc = MaybeUninit::<JxlColorEncoding>::uninit();
    assert_eq!(
        JxlDecoderGetColorAsEncodedProfile(
            dec.as_deref(),
            JXL_COLOR_PROFILE_TARGET_ORIGINAL,
            Some(&mut enc),
        ),
        JXL_DEC_ERROR
    );
    let orig_icc = get_icc_profile(dec.as_deref(), JXL_COLOR_PROFILE_TARGET_ORIGINAL);
    assert!(!orig_icc.is_empty());

    // Requesting linear sRGB as preferred profile on XYB image is a hint and returns
    // JXL_DEC_SUCCESS; JXL_COLOR_PROFILE_TARGET_DATA reflects the actual output profile
    // (sRGB by default before a float pixel format is set).
    let lin_srgb = linear_srgb_encoding(JXL_FALSE);
    assert_eq!(
        JxlDecoderSetPreferredColorProfile(dec.as_deref_mut(), Some(&lin_srgb)),
        JXL_DEC_SUCCESS
    );
    let srgb = srgb_encoding(JXL_FALSE);
    let data_enc = get_color_encoding(dec.as_deref(), JXL_COLOR_PROFILE_TARGET_DATA);
    assert_eq!(data_enc, srgb);

    JxlDecoderDestroy(dec);

    // On a non-XYB image (`uses_original_profile == JXL_TRUE`), setting a preferred
    // color encoding without a CMS succeeds (`JXL_DEC_SUCCESS`) while leaving
    // `JXL_COLOR_PROFILE_TARGET_DATA` in the original embedded profile.
    let non_xyb = test_resource("3x3_srgb_lossless.jxl");
    let mut dec_non_xyb = JxlDecoderCreate(None);
    assert_eq!(
        JxlDecoderSubscribeEvents(dec_non_xyb.as_deref_mut(), events),
        JXL_DEC_SUCCESS
    );
    assert_eq!(
        set_input(dec_non_xyb.as_deref_mut(), &non_xyb),
        JXL_DEC_SUCCESS
    );
    JxlDecoderCloseInput(dec_non_xyb.as_deref_mut());
    assert_eq!(
        JxlDecoderProcessInput(dec_non_xyb.as_deref_mut()),
        JXL_DEC_BASIC_INFO
    );
    assert_eq!(
        JxlDecoderProcessInput(dec_non_xyb.as_deref_mut()),
        JXL_DEC_COLOR_ENCODING
    );
    let non_xyb_orig_enc =
        get_color_encoding(dec_non_xyb.as_deref(), JXL_COLOR_PROFILE_TARGET_ORIGINAL);
    assert_eq!(
        JxlDecoderSetPreferredColorProfile(dec_non_xyb.as_deref_mut(), Some(&lin_srgb)),
        JXL_DEC_SUCCESS
    );
    let non_xyb_data_enc =
        get_color_encoding(dec_non_xyb.as_deref(), JXL_COLOR_PROFILE_TARGET_DATA);
    assert_eq!(non_xyb_data_enc, non_xyb_orig_enc);
    assert_ne!(non_xyb_data_enc, lin_srgb);
    JxlDecoderDestroy(dec_non_xyb);
}

fn decode_with_c_api(data: &[u8], format: JxlPixelFormat, subscribe_frame: bool) -> Vec<u8> {
    let mut dec = JxlDecoderCreate(None);
    let mut events = (JXL_DEC_BASIC_INFO.0 as i32) | (JXL_DEC_FULL_IMAGE.0 as i32);
    if subscribe_frame {
        events |= JXL_DEC_FRAME.0 as i32;
    }
    assert_eq!(
        JxlDecoderSubscribeEvents(dec.as_deref_mut(), events),
        JXL_DEC_SUCCESS
    );
    assert_eq!(set_input(dec.as_deref_mut(), data), JXL_DEC_SUCCESS);
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

    let buf_size = get_image_out_buffer_size(dec.as_deref(), &format);
    let mut out = vec![0u8; buf_size];
    assert_eq!(
        set_image_out_buffer(dec.as_deref_mut(), &format, &mut out),
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
fn pixel_formats_match_native_jxl_rs() {
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
fn unaligned_buffer_and_row_stride() {
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
    let mut dec = JxlDecoderCreate(None);
    assert_eq!(
        JxlDecoderSubscribeEvents(dec.as_deref_mut(), JXL_DEC_FULL_IMAGE.0 as i32),
        JXL_DEC_SUCCESS
    );
    assert_eq!(set_input(dec.as_deref_mut(), &data), JXL_DEC_SUCCESS);
    JxlDecoderCloseInput(dec.as_deref_mut());
    assert_eq!(
        JxlDecoderProcessInput(dec.as_deref_mut()),
        JXL_DEC_NEED_IMAGE_OUT_BUFFER
    );
    let size = get_image_out_buffer_size(dec.as_deref(), &fmt_strided);
    let bytes_per_row = 3 * 3 * 4; // 36 bytes
    let stride = 51; // next multiple of 17 >= 36
    assert_eq!(size, 2 * stride + bytes_per_row);

    let mut backing = vec![0u8; size + 8];
    let unaligned = &mut backing[1..1 + size];
    assert_eq!(
        set_image_out_buffer(dec.as_deref_mut(), &fmt_strided, unaligned),
        JXL_DEC_SUCCESS
    );
    assert_eq!(
        JxlDecoderProcessInput(dec.as_deref_mut()),
        JXL_DEC_FULL_IMAGE
    );
    assert_eq!(JxlDecoderProcessInput(dec.as_deref_mut()), JXL_DEC_SUCCESS);

    for y in 0..3 {
        let row_slice = &unaligned[y * stride..y * stride + bytes_per_row];
        assert_eq!(
            row_slice,
            &expected[y * bytes_per_row..(y + 1) * bytes_per_row]
        );
    }
    JxlDecoderDestroy(dec);
}

#[test]
fn preview_frame_decoding() {
    let data = test_resource("with_preview.jxl");

    let mut dec = JxlDecoderCreate(None);
    let events = (JXL_DEC_BASIC_INFO.0 as i32)
        | (JXL_DEC_PREVIEW_IMAGE.0 as i32)
        | (JXL_DEC_FRAME.0 as i32)
        | (JXL_DEC_FULL_IMAGE.0 as i32);
    assert_eq!(
        JxlDecoderSubscribeEvents(dec.as_deref_mut(), events),
        JXL_DEC_SUCCESS
    );
    assert_eq!(set_input(dec.as_deref_mut(), &data), JXL_DEC_SUCCESS);
    JxlDecoderCloseInput(dec.as_deref_mut());

    assert_eq!(
        JxlDecoderProcessInput(dec.as_deref_mut()),
        JXL_DEC_BASIC_INFO
    );
    let info = get_basic_info(dec.as_deref());
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
    let preview_size = get_preview_out_buffer_size(dec.as_deref(), &preview_fmt);
    assert_eq!(preview_size, 16 * 16 * 3);
    let mut preview_buf = vec![0u8; preview_size];
    assert_eq!(
        set_preview_out_buffer(dec.as_deref_mut(), &preview_fmt, &mut preview_buf),
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
    let main_size = get_image_out_buffer_size(dec.as_deref(), &main_fmt);
    assert_eq!(main_size, (info.xsize as usize) * (info.ysize as usize) * 8);
    let mut main_buf = vec![0u8; main_size];
    assert_eq!(
        set_image_out_buffer(dec.as_deref_mut(), &main_fmt, &mut main_buf),
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

#[test]
fn extra_channels_and_unpremultiply_alpha() {
    let data = test_resource("3x3a_srgb_lossless.jxl");

    let mut dec = JxlDecoderCreate(None);
    let events = (JXL_DEC_BASIC_INFO.0 as i32) | (JXL_DEC_FULL_IMAGE.0 as i32);
    assert_eq!(
        JxlDecoderSubscribeEvents(dec.as_deref_mut(), events),
        JXL_DEC_SUCCESS
    );
    assert_eq!(
        JxlDecoderSetUnpremultiplyAlpha(dec.as_deref_mut(), JXL_TRUE),
        JXL_DEC_SUCCESS
    );
    assert_eq!(set_input(dec.as_deref_mut(), &data), JXL_DEC_SUCCESS);
    JxlDecoderCloseInput(dec.as_deref_mut());

    assert_eq!(
        JxlDecoderProcessInput(dec.as_deref_mut()),
        JXL_DEC_BASIC_INFO
    );
    let info = get_basic_info(dec.as_deref());
    assert_eq!(info.num_extra_channels, 1);
    assert!(info.alpha_bits > 0);

    let mut ec_info = MaybeUninit::<JxlExtraChannelInfo>::uninit();
    assert_eq!(
        JxlDecoderGetExtraChannelInfo(dec.as_deref(), 0, Some(&mut ec_info)),
        JXL_DEC_SUCCESS
    );
    // SAFETY: `JxlDecoderGetExtraChannelInfo` returned `JXL_DEC_SUCCESS`, initializing `ec_info`.
    let ec_info = unsafe { ec_info.assume_init() };
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
    let mut dummy_size = MaybeUninit::<usize>::uninit();
    assert_eq!(
        JxlDecoderImageOutBufferSize(
            dec.as_deref(),
            Some(&invalid_gray_fmt),
            Some(&mut dummy_size)
        ),
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
    let color_size = get_image_out_buffer_size(dec.as_deref(), &color_fmt);
    let ec_size = get_extra_channel_buffer_size(dec.as_deref(), &ec_fmt, 0);
    assert_eq!(color_size, 3 * 3 * 3);
    assert_eq!(ec_size, 3 * 3);

    let mut color_buf = vec![0u8; color_size];
    let mut ec_buf = vec![0u8; ec_size];
    // Setting extra channel buffer before main image buffer must still require main buffer!
    assert_eq!(
        set_extra_channel_buffer(dec.as_deref_mut(), &ec_fmt, &mut ec_buf, 0),
        JXL_DEC_SUCCESS
    );
    assert_eq!(
        JxlDecoderProcessInput(dec.as_deref_mut()),
        JXL_DEC_NEED_IMAGE_OUT_BUFFER
    );
    assert_eq!(
        set_image_out_buffer(dec.as_deref_mut(), &color_fmt, &mut color_buf),
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

#[test]
fn streaming_chunks() {
    let data = test_resource("basic.jxl");
    let fmt = JxlPixelFormat {
        num_channels: 3,
        data_type: JXL_TYPE_UINT8,
        endianness: JXL_NATIVE_ENDIAN,
        align: 0,
    };
    let expected = decode_with_c_api(&data, fmt, false);

    let mut dec = JxlDecoderCreate(None);
    let events = (JXL_DEC_BASIC_INFO.0 as i32) | (JXL_DEC_FULL_IMAGE.0 as i32);
    assert_eq!(
        JxlDecoderSubscribeEvents(dec.as_deref_mut(), events),
        JXL_DEC_SUCCESS
    );

    let mut pos = 0usize;
    let chunk_size = 5usize;
    let mut out_buf = Vec::new();
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
                    set_input(dec.as_deref_mut(), &data[pos..end]),
                    JXL_DEC_SUCCESS
                );
                pos = end;
                if pos == data.len() {
                    JxlDecoderCloseInput(dec.as_deref_mut());
                }
            }
            JXL_DEC_BASIC_INFO => {
                let info = get_basic_info(dec.as_deref());
                out_buf = vec![0u8; (info.xsize as usize) * (info.ysize as usize) * 3];
            }
            JXL_DEC_NEED_IMAGE_OUT_BUFFER => {
                assert_eq!(
                    set_image_out_buffer(dec.as_deref_mut(), &fmt, &mut out_buf),
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
    assert_eq!(out_buf, expected);
    JxlDecoderDestroy(dec);
}

#[test]
fn flush_image_semantics() {
    let data = test_resource("conformance_test_images/bicycles.jxl");
    let fmt = JxlPixelFormat {
        num_channels: 3,
        data_type: JXL_TYPE_UINT8,
        endianness: JXL_NATIVE_ENDIAN,
        align: 0,
    };

    let mut dec = JxlDecoderCreate(None);
    let events = (JXL_DEC_BASIC_INFO.0 as i32) | (JXL_DEC_FULL_IMAGE.0 as i32);
    assert_eq!(
        JxlDecoderSubscribeEvents(dec.as_deref_mut(), events),
        JXL_DEC_SUCCESS
    );

    let mut pos = 0usize;
    let chunk_size = 256usize;
    let mut out_buf = Vec::new();
    let mut flushed_before_data_failed = false;
    let mut flushed_succeeded_count = 0usize;

    loop {
        let status = JxlDecoderProcessInput(dec.as_deref_mut());
        match status {
            JXL_DEC_NEED_MORE_INPUT => {
                if !out_buf.is_empty() {
                    let flush_res = JxlDecoderFlushImage(dec.as_deref_mut());
                    if flush_res == JXL_DEC_SUCCESS {
                        flushed_succeeded_count += 1;
                        // A second consecutive flush with no new input must also return SUCCESS
                        // once flush_pixels() has returned true at least once.
                        assert_eq!(JxlDecoderFlushImage(dec.as_deref_mut()), JXL_DEC_SUCCESS);
                    }
                }
                let rem = JxlDecoderReleaseInput(dec.as_deref_mut());
                pos -= rem;
                let end = (pos + chunk_size).min(data.len());
                assert_eq!(
                    set_input(dec.as_deref_mut(), &data[pos..end]),
                    JXL_DEC_SUCCESS
                );
                pos = end;
                if pos == data.len() {
                    JxlDecoderCloseInput(dec.as_deref_mut());
                }
            }
            JXL_DEC_BASIC_INFO => {
                let info = get_basic_info(dec.as_deref());
                out_buf = vec![0u8; (info.xsize as usize) * (info.ysize as usize) * 3];
            }
            JXL_DEC_NEED_IMAGE_OUT_BUFFER => {
                assert_eq!(
                    set_image_out_buffer(dec.as_deref_mut(), &fmt, &mut out_buf),
                    JXL_DEC_SUCCESS
                );
                // Right at NEED_IMAGE_OUT_BUFFER (before MainDecode / any pixels rendered),
                // FlushImage must return JXL_DEC_ERROR.
                assert_eq!(JxlDecoderFlushImage(dec.as_deref_mut()), JXL_DEC_ERROR);
                flushed_before_data_failed = true;
            }
            JXL_DEC_FULL_IMAGE => {}
            JXL_DEC_SUCCESS => break,
            other => panic!("unexpected status: {:?}", other),
        }
    }

    assert!(flushed_before_data_failed);
    assert!(flushed_succeeded_count > 0);
    JxlDecoderDestroy(dec);
}

#[test]
fn animation_skip_frames_and_rewind() {
    let data = test_resource("conformance_test_images/animation_icos4d_5.jxl");

    let mut dec = JxlDecoderCreate(None);
    let events =
        (JXL_DEC_BASIC_INFO.0 as i32) | (JXL_DEC_FRAME.0 as i32) | (JXL_DEC_FULL_IMAGE.0 as i32);
    assert_eq!(
        JxlDecoderSubscribeEvents(dec.as_deref_mut(), events),
        JXL_DEC_SUCCESS
    );
    assert_eq!(set_input(dec.as_deref_mut(), &data), JXL_DEC_SUCCESS);
    JxlDecoderCloseInput(dec.as_deref_mut());

    assert_eq!(
        JxlDecoderProcessInput(dec.as_deref_mut()),
        JXL_DEC_BASIC_INFO
    );
    let info = get_basic_info(dec.as_deref());
    assert_eq!(info.have_animation, JXL_TRUE);

    let fmt = JxlPixelFormat {
        num_channels: 4,
        data_type: JXL_TYPE_UINT8,
        endianness: JXL_NATIVE_ENDIAN,
        align: 0,
    };
    let buf_size = get_image_out_buffer_size(dec.as_deref(), &fmt);
    let mut frame_buf = vec![0u8; buf_size];

    let mut frame_count = 0usize;
    let mut need_buf_count = 0usize;
    loop {
        let st = JxlDecoderProcessInput(dec.as_deref_mut());
        match st {
            JXL_DEC_FRAME => {
                let fh = get_frame_header(dec.as_deref());
                assert_eq!(fh.layer_info.xsize, info.xsize);
                assert_eq!(fh.layer_info.ysize, info.ysize);
            }
            JXL_DEC_NEED_IMAGE_OUT_BUFFER => {
                need_buf_count += 1;
                assert_eq!(
                    set_image_out_buffer(dec.as_deref_mut(), &fmt, &mut frame_buf),
                    JXL_DEC_SUCCESS
                );
            }
            JXL_DEC_FULL_IMAGE => {
                frame_count += 1;
                // JxlDecoderGetFrameHeader must remain valid at JXL_DEC_FULL_IMAGE!
                let fh = get_frame_header(dec.as_deref());
                assert_eq!(fh.layer_info.xsize, info.xsize);
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
    assert_eq!(set_input(dec.as_deref_mut(), &data), JXL_DEC_SUCCESS);
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
                    set_image_out_buffer(dec.as_deref_mut(), &fmt, &mut frame_buf),
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

#[test]
fn thread_and_resizable_parallel_runners() {
    let data = test_resource("conformance_test_images/bicycles.jxl");
    let fmt = JxlPixelFormat {
        num_channels: 3,
        data_type: JXL_TYPE_UINT8,
        endianness: JXL_NATIVE_ENDIAN,
        align: 0,
    };
    let expected = decode_with_c_api(&data, fmt, false);

    assert!(JxlThreadParallelRunnerDefaultNumWorkerThreads() >= 1);
    let runner = JxlThreadParallelRunnerCreate(None, 2);
    assert!(!runner.is_null());

    let mut dec = JxlDecoderCreate(None);
    assert_eq!(
        JxlDecoderSetParallelRunner(dec.as_deref_mut(), Some(JxlThreadParallelRunner), runner),
        JXL_DEC_SUCCESS
    );
    assert_eq!(
        JxlDecoderSubscribeEvents(dec.as_deref_mut(), JXL_DEC_FULL_IMAGE.0 as i32),
        JXL_DEC_SUCCESS
    );
    assert_eq!(set_input(dec.as_deref_mut(), &data), JXL_DEC_SUCCESS);
    JxlDecoderCloseInput(dec.as_deref_mut());
    assert_eq!(
        JxlDecoderProcessInput(dec.as_deref_mut()),
        JXL_DEC_NEED_IMAGE_OUT_BUFFER
    );
    let mut out = vec![0u8; expected.len()];
    assert_eq!(
        set_image_out_buffer(dec.as_deref_mut(), &fmt, &mut out),
        JXL_DEC_SUCCESS
    );
    assert_eq!(
        JxlDecoderProcessInput(dec.as_deref_mut()),
        JXL_DEC_FULL_IMAGE
    );
    assert_eq!(out, expected);

    JxlDecoderDestroy(dec);
    JxlThreadParallelRunnerDestroy(runner);

    let r_runner = JxlResizableParallelRunnerCreate(None);
    assert!(!r_runner.is_null());
    let suggested = JxlResizableParallelRunnerSuggestThreads(512, 512);
    assert!(suggested >= 1);
    JxlResizableParallelRunnerSetThreads(r_runner, suggested as usize);
    JxlResizableParallelRunnerDestroy(r_runner);
}

#[test]
fn box_extraction_exif() {
    let cases = [
        (
            &include_bytes!("../../jxl/tests/testdata/exif.jxl")[..],
            false,
            170u64,
        ),
        (
            &include_bytes!("../../jxl/tests/testdata/exif_brob.jxl")[..],
            true,
            124u64,
        ),
        (
            &include_bytes!("../../jxl/tests/testdata/exif_trailing_finite.jxl")[..],
            false,
            170u64,
        ),
        (
            &include_bytes!("../../jxl/tests/testdata/exif_brob_trailing_finite.jxl")[..],
            true,
            124u64,
        ),
        (
            &include_bytes!("../../jxl/tests/testdata/exif_trailing_infinite.jxl")[..],
            false,
            0u64,
        ),
        (
            &include_bytes!("../../jxl/tests/testdata/exif_brob_trailing_infinite.jxl")[..],
            true,
            0u64,
        ),
    ];

    let expected_decompress_true = if cfg!(feature = "brotli") {
        JXL_DEC_SUCCESS
    } else {
        JXL_DEC_ERROR
    };
    let mut expected_exif_payload: Option<Vec<u8>> = None;

    let decompress_modes: &[JXL_BOOL] = if cfg!(feature = "brotli") {
        &[JXL_FALSE, JXL_TRUE]
    } else {
        &[JXL_FALSE]
    };

    for &decompress in decompress_modes {
        for (data, is_brob, expected_raw_contents) in cases {
            let mut dec = JxlDecoderCreate(None);
            let events = (JXL_DEC_BOX.0 | JXL_DEC_BOX_COMPLETE.0) as i32;
            assert_eq!(
                JxlDecoderSubscribeEvents(dec.as_deref_mut(), events),
                JXL_DEC_SUCCESS
            );
            assert_eq!(
                JxlDecoderSetDecompressBoxes(dec.as_deref_mut(), JXL_TRUE),
                expected_decompress_true
            );
            assert_eq!(
                JxlDecoderSetDecompressBoxes(dec.as_deref_mut(), decompress),
                JXL_DEC_SUCCESS
            );
            assert_eq!(set_input(dec.as_deref_mut(), data), JXL_DEC_SUCCESS);
            JxlDecoderCloseInput(dec.as_deref_mut());

            assert_eq!(JxlDecoderProcessInput(dec.as_deref_mut()), JXL_DEC_BOX);

            let raw_type = get_box_type(dec.as_deref_mut(), JXL_FALSE);
            let dec_type = get_box_type(dec.as_deref_mut(), JXL_TRUE);
            assert_eq!(&dec_type, b"Exif");
            if is_brob {
                assert_eq!(&raw_type, b"brob");
            } else {
                assert_eq!(&raw_type, b"Exif");
            }

            let contents_size = get_box_size_contents(dec.as_deref());
            let raw_size = get_box_size_raw(dec.as_deref());
            assert_eq!(contents_size, expected_raw_contents);
            assert_eq!(
                raw_size,
                if expected_raw_contents == 0 {
                    0
                } else {
                    expected_raw_contents + 8
                }
            );

            // Read box in small 32-byte chunks to exercise JXL_DEC_BOX_NEED_MORE_OUTPUT.
            let mut collected = Vec::new();
            let mut chunk = [0u8; 32];
            assert_eq!(
                set_box_buffer(dec.as_deref_mut(), &mut chunk),
                JXL_DEC_SUCCESS
            );
            loop {
                let status = JxlDecoderProcessInput(dec.as_deref_mut());
                if status == JXL_DEC_BOX_NEED_MORE_OUTPUT {
                    // Calling ProcessInput again without releasing/setting still returns
                    // JXL_DEC_BOX_NEED_MORE_OUTPUT, matching libjxl.
                    assert_eq!(
                        JxlDecoderProcessInput(dec.as_deref_mut()),
                        JXL_DEC_BOX_NEED_MORE_OUTPUT
                    );
                }
                let rem = JxlDecoderReleaseBoxBuffer(dec.as_deref_mut());
                collected.extend_from_slice(&chunk[..chunk.len() - rem]);
                if status == JXL_DEC_BOX_NEED_MORE_OUTPUT {
                    assert_eq!(
                        set_box_buffer(dec.as_deref_mut(), &mut chunk),
                        JXL_DEC_SUCCESS
                    );
                } else {
                    assert_eq!(status, JXL_DEC_BOX_COMPLETE);
                    break;
                }
            }
            if decompress == JXL_FALSE && is_brob {
                assert_eq!(collected.len(), 124);
                assert_eq!(&collected[..4], b"Exif");
            } else {
                assert_eq!(collected.len(), 170);
                if let Some(ref exp) = expected_exif_payload {
                    assert_eq!(&collected, exp);
                } else {
                    expected_exif_payload = Some(collected);
                }
            }

            assert_eq!(JxlDecoderProcessInput(dec.as_deref_mut()), JXL_DEC_SUCCESS);
            JxlDecoderDestroy(dec);
        }
    }

    // Test a box using the 64-bit extended length field (LBox == 1, 16-byte XLBox header).
    let xlbox_data = include_bytes!("../../jxl/tests/testdata/exif_trailing_finite_xlbox.jxl");
    let mut dec = JxlDecoderCreate(None);
    assert_eq!(
        JxlDecoderSubscribeEvents(dec.as_deref_mut(), JXL_DEC_BOX.0 as i32),
        JXL_DEC_SUCCESS
    );
    assert_eq!(set_input(dec.as_deref_mut(), xlbox_data), JXL_DEC_SUCCESS);
    JxlDecoderCloseInput(dec.as_deref_mut());
    assert_eq!(JxlDecoderProcessInput(dec.as_deref_mut()), JXL_DEC_BOX);
    assert_eq!(get_box_size_contents(dec.as_deref()), 170);
    assert_eq!(get_box_size_raw(dec.as_deref()), 170 + 16);
    JxlDecoderDestroy(dec);

    // Also test FFmpeg's pattern: releasing and re-setting remaining input after every event
    // without calling `JxlDecoderCloseInput`.
    let data = &include_bytes!("../../jxl/tests/testdata/exif.jxl")[..];
    let mut dec = JxlDecoderCreate(None);
    let events = (JXL_DEC_BASIC_INFO.0
        | JXL_DEC_COLOR_ENCODING.0
        | JXL_DEC_FRAME.0
        | JXL_DEC_FULL_IMAGE.0
        | JXL_DEC_BOX.0) as i32;
    assert_eq!(
        JxlDecoderSubscribeEvents(dec.as_deref_mut(), events),
        JXL_DEC_SUCCESS
    );
    let decompress = if cfg!(feature = "brotli") {
        JXL_TRUE
    } else {
        JXL_FALSE
    };
    assert_eq!(
        JxlDecoderSetDecompressBoxes(dec.as_deref_mut(), decompress),
        JXL_DEC_SUCCESS
    );
    let fmt = JxlPixelFormat {
        num_channels: 3,
        data_type: JXL_TYPE_UINT8,
        endianness: JXL_NATIVE_ENDIAN,
        align: 0,
    };
    let mut pkt = data;
    let mut img_buf = Vec::new();
    let mut box_buf = vec![0u8; 4096];
    let mut got_full_image = false;
    let mut got_box = false;
    while !pkt.is_empty() {
        assert_eq!(set_input(dec.as_deref_mut(), pkt), JXL_DEC_SUCCESS);
        let status = JxlDecoderProcessInput(dec.as_deref_mut());
        let remaining = JxlDecoderReleaseInput(dec.as_deref_mut());
        pkt = &pkt[pkt.len() - remaining..];
        match status {
            JXL_DEC_BASIC_INFO => {
                let info = get_basic_info(dec.as_deref());
                img_buf = vec![0u8; (info.xsize as usize) * (info.ysize as usize) * 3];
            }
            JXL_DEC_COLOR_ENCODING | JXL_DEC_FRAME => {}
            JXL_DEC_NEED_IMAGE_OUT_BUFFER => {
                assert_eq!(
                    set_image_out_buffer(dec.as_deref_mut(), &fmt, &mut img_buf),
                    JXL_DEC_SUCCESS
                );
            }
            JXL_DEC_FULL_IMAGE => got_full_image = true,
            JXL_DEC_BOX => {
                assert_eq!(&get_box_type(dec.as_deref_mut(), JXL_TRUE), b"Exif");
                assert_eq!(
                    set_box_buffer(dec.as_deref_mut(), &mut box_buf),
                    JXL_DEC_SUCCESS
                );
                got_box = true;
            }
            JXL_DEC_NEED_MORE_INPUT | JXL_DEC_SUCCESS => break,
            other => panic!("unexpected status: {:?}", other),
        }
    }
    let box_written = box_buf.len() - JxlDecoderReleaseBoxBuffer(dec.as_deref_mut());
    assert!(got_full_image);
    assert!(got_box);
    assert_eq!(
        &box_buf[..box_written],
        expected_exif_payload.as_ref().unwrap().as_slice()
    );
    JxlDecoderDestroy(dec);
}

#[test]
fn unimplemented_features_return_errors() {
    let data = test_resource("basic.jxl");

    let mut dec = JxlDecoderCreate(None);
    assert_eq!(
        JxlDecoderSubscribeEvents(dec.as_deref_mut(), JXL_DEC_BOX_COMPLETE.0 as i32),
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
    let dummy_cms = JxlCmsInterface {
        set_fields_data: std::ptr::null_mut(),
        set_fields_from_icc: None,
        init_data: std::ptr::null_mut(),
        init: None,
        get_src_buf: None,
        get_dst_buf: None,
        run: None,
        destroy: None,
    };
    assert_eq!(
        JxlDecoderSetCms(dec.as_deref_mut(), dummy_cms),
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
    // SAFETY: Passing null pointer to verify `JxlDecoderSetBoxBuffer` rejects null buffers.
    let box_null_status =
        unsafe { JxlDecoderSetBoxBuffer(dec.as_deref_mut(), std::ptr::null_mut(), 16) };
    assert_eq!(box_null_status, JXL_DEC_ERROR);
    assert_eq!(
        JxlDecoderSetJPEGBuffer(dec.as_deref_mut(), std::ptr::null_mut(), 0),
        JXL_DEC_ERROR
    );
    // SAFETY: Passing null/None arguments to verify callback registration rejects invalid inputs.
    let cb_null_status = unsafe {
        JxlDecoderSetImageOutCallback(dec.as_deref_mut(), None, None, std::ptr::null_mut())
    };
    assert_eq!(cb_null_status, JXL_DEC_ERROR);
    // SAFETY: Passing null/None arguments to verify multithreaded callback registration rejects
    // invalid inputs.
    let mt_cb_null_status = unsafe {
        JxlDecoderSetMultithreadedImageOutCallback(
            dec.as_deref_mut(),
            None,
            None,
            None,
            None,
            std::ptr::null_mut(),
        )
    };
    assert_eq!(mt_cb_null_status, JXL_DEC_ERROR);
    let expected_decompress_true = if cfg!(feature = "brotli") {
        JXL_DEC_SUCCESS
    } else {
        JXL_DEC_ERROR
    };
    assert_eq!(
        JxlDecoderSetDecompressBoxes(dec.as_deref_mut(), JXL_TRUE),
        expected_decompress_true
    );
    assert_eq!(
        JxlDecoderSetDecompressBoxes(dec.as_deref_mut(), JXL_FALSE),
        JXL_DEC_SUCCESS
    );
    assert_eq!(
        JxlDecoderSetImageOutBitDepth(
            dec.as_deref_mut(),
            Some(&JxlBitDepth {
                type_: JxlBitDepthType(99),
                bits_per_sample: 8,
                exponent_bits_per_sample: 0,
            }),
        ),
        JXL_DEC_ERROR
    );

    // Custom memory manager structs are accepted (and ignored internally) so callers like FFmpeg
    // and ImageMagick can create decoders and runners; callers must not assume `alloc` was used to
    // allocate the returned handle.
    unsafe extern "C" fn dummy_alloc(_opaque: *mut c_void, _size: usize) -> *mut c_void {
        std::ptr::null_mut()
    }
    unsafe extern "C" fn dummy_free(_opaque: *mut c_void, _address: *mut c_void) {}
    let custom_mm = JxlMemoryManager {
        opaque: std::ptr::null_mut(),
        alloc: Some(dummy_alloc),
        free: Some(dummy_free),
    };
    let dec_mm = JxlDecoderCreate(Some(&custom_mm));
    assert!(dec_mm.is_some());
    JxlDecoderDestroy(dec_mm);
    let runner_mm = JxlThreadParallelRunnerCreate(Some(&custom_mm), 2);
    assert!(!runner_mm.is_null());
    JxlThreadParallelRunnerDestroy(runner_mm);

    // Once ProcessInput has started, decoder-level options return JXL_DEC_ERROR.
    assert_eq!(
        JxlDecoderSubscribeEvents(
            dec.as_deref_mut(),
            (JXL_DEC_BASIC_INFO.0 | JXL_DEC_BOX.0) as i32
        ),
        JXL_DEC_SUCCESS
    );
    assert!(JxlDecoderSizeHintBasicInfo(dec.as_deref()) > 0);
    assert_eq!(set_input(dec.as_deref_mut(), &data), JXL_DEC_SUCCESS);
    assert_eq!(
        JxlDecoderProcessInput(dec.as_deref_mut()),
        JXL_DEC_BASIC_INFO
    );
    assert_eq!(JxlDecoderSizeHintBasicInfo(dec.as_deref()), 0);
    assert_eq!(
        JxlDecoderSetDecompressBoxes(dec.as_deref_mut(), JXL_TRUE),
        expected_decompress_true
    );
    assert_eq!(
        JxlDecoderSetDecompressBoxes(dec.as_deref_mut(), JXL_FALSE),
        JXL_DEC_SUCCESS
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

    assert_eq!(JxlGainMapGetBundleSize(None, None), JXL_FALSE);
    assert_eq!(
        JxlGainMapReadBundle(None, std::ptr::null(), 0, None),
        JXL_FALSE
    );

    let enc = JxlEncoderCreate(None);
    assert!(enc.is_null());
    assert_eq!(JxlEncoderGetError(None), JXL_ENC_ERR_NOT_SUPPORTED);

    let mut basic_info = MaybeUninit::<JxlBasicInfo>::uninit();
    JxlEncoderInitBasicInfo(Some(&mut basic_info));
    // SAFETY: `JxlEncoderInitBasicInfo` unconditionally initializes `basic_info`.
    let basic_info = unsafe { basic_info.assume_init() };
    assert_eq!(basic_info.bits_per_sample, 8);
    assert_eq!(basic_info.num_color_channels, 3);
    assert_eq!(basic_info.orientation, JXL_ORIENT_IDENTITY);

    let mut fh = MaybeUninit::<JxlFrameHeader>::uninit();
    JxlEncoderInitFrameHeader(Some(&mut fh));
    // SAFETY: `JxlEncoderInitFrameHeader` unconditionally initializes `fh`.
    assert_eq!(unsafe { fh.assume_init() }.is_last, JXL_TRUE);

    let mut ec_info = MaybeUninit::<JxlExtraChannelInfo>::uninit();
    JxlEncoderInitExtraChannelInfo(JXL_CHANNEL_ALPHA, Some(&mut ec_info));
    // SAFETY: `JxlEncoderInitExtraChannelInfo` unconditionally initializes `ec_info`.
    let ec_info = unsafe { ec_info.assume_init() };
    assert_eq!(ec_info.type_, JXL_CHANNEL_ALPHA);
    assert_eq!(ec_info.bits_per_sample, 8);
}

#[test]
fn xyb_auto_color_profile() {
    // Test that for an XYB image with ICC (uses_original_profile=false), switching between
    // float and uint8 pixel formats without an explicit preferred color profile updates
    // JXL_COLOR_PROFILE_TARGET_DATA between Linear sRGB and sRGB.
    let icc_xyb = test_resource("lossy_with_icc.jxl");
    let mut dec = JxlDecoderCreate(None);
    let events = (JXL_DEC_BASIC_INFO.0 as i32)
        | (JXL_DEC_COLOR_ENCODING.0 as i32)
        | (JXL_DEC_FULL_IMAGE.0 as i32);
    assert_eq!(
        JxlDecoderSubscribeEvents(dec.as_deref_mut(), events),
        JXL_DEC_SUCCESS
    );
    assert_eq!(set_input(dec.as_deref_mut(), &icc_xyb), JXL_DEC_SUCCESS);
    JxlDecoderCloseInput(dec.as_deref_mut());

    assert_eq!(
        JxlDecoderProcessInput(dec.as_deref_mut()),
        JXL_DEC_BASIC_INFO
    );
    assert_eq!(
        JxlDecoderProcessInput(dec.as_deref_mut()),
        JXL_DEC_COLOR_ENCODING
    );
    // Default before buffer is set is sRGB (since default format is u8)
    assert_eq!(
        get_color_encoding(dec.as_deref(), JXL_COLOR_PROFILE_TARGET_DATA).transfer_function,
        JXL_TRANSFER_FUNCTION_SRGB
    );

    assert_eq!(
        JxlDecoderProcessInput(dec.as_deref_mut()),
        JXL_DEC_NEED_IMAGE_OUT_BUFFER
    );
    // Setting a float buffer updates TARGET_DATA to Linear sRGB
    let fmt_f32 = JxlPixelFormat {
        num_channels: 3,
        data_type: JXL_TYPE_FLOAT,
        endianness: JXL_NATIVE_ENDIAN,
        align: 0,
    };
    let size_f32 = get_image_out_buffer_size(dec.as_deref(), &fmt_f32);
    let mut buf_f32 = vec![0u8; size_f32];
    assert_eq!(
        set_image_out_buffer(dec.as_deref_mut(), &fmt_f32, &mut buf_f32),
        JXL_DEC_SUCCESS
    );
    assert_eq!(
        get_color_encoding(dec.as_deref(), JXL_COLOR_PROFILE_TARGET_DATA).transfer_function,
        JXL_TRANSFER_FUNCTION_LINEAR
    );

    // Switching back to a u8 buffer restores TARGET_DATA to sRGB
    let fmt_u8 = JxlPixelFormat {
        num_channels: 3,
        data_type: JXL_TYPE_UINT8,
        endianness: JXL_NATIVE_ENDIAN,
        align: 0,
    };
    let size_u8 = get_image_out_buffer_size(dec.as_deref(), &fmt_u8);
    let mut buf_u8 = vec![0u8; size_u8];
    assert_eq!(
        set_image_out_buffer(dec.as_deref_mut(), &fmt_u8, &mut buf_u8),
        JXL_DEC_SUCCESS
    );
    assert_eq!(
        get_color_encoding(dec.as_deref(), JXL_COLOR_PROFILE_TARGET_DATA).transfer_function,
        JXL_TRANSFER_FUNCTION_SRGB
    );

    JxlDecoderDestroy(dec);
}

struct SimpleCallbackCapture {
    width: usize,
    height: usize,
    bytes_per_pixel: usize,
    sample_align: usize,
    out: std::sync::Mutex<Vec<u8>>,
    visits: std::sync::Mutex<Vec<u32>>,
}

impl SimpleCallbackCapture {
    fn new(width: usize, height: usize, bytes_per_pixel: usize, sample_align: usize) -> Self {
        Self {
            width,
            height,
            bytes_per_pixel,
            sample_align,
            out: std::sync::Mutex::new(vec![0u8; width * height * bytes_per_pixel]),
            visits: std::sync::Mutex::new(vec![0u32; width * height]),
        }
    }

    fn as_opaque(&self) -> *mut c_void {
        (&raw const *self).cast_mut().cast::<c_void>()
    }

    fn into_verified_pixels(self) -> Vec<u8> {
        assert!(
            self.visits
                .into_inner()
                .unwrap()
                .into_iter()
                .all(|v| v == 1)
        );
        self.out.into_inner().unwrap()
    }
}

unsafe extern "C" fn simple_capture_cb(
    opaque: *mut c_void,
    x: usize,
    y: usize,
    num_pixels: usize,
    pixels: *const c_void,
) {
    // SAFETY: `opaque` points to a valid `SimpleCallbackCapture` outliving decoding.
    let cap = unsafe { &*(opaque.cast::<SimpleCallbackCapture>()) };
    assert!(y < cap.height);
    assert!(x + num_pixels <= cap.width);
    assert_eq!((pixels as usize) % cap.sample_align, 0);
    let byte_len = num_pixels * cap.bytes_per_pixel;
    // SAFETY: `pixels` is valid for reads of `num_pixels * bytes_per_pixel` bytes during this call.
    let src = unsafe { std::slice::from_raw_parts(pixels.cast::<u8>(), byte_len) };

    let start = (y * cap.width + x) * cap.bytes_per_pixel;
    cap.out.lock().unwrap()[start..start + byte_len].copy_from_slice(src);
    let mut visits = cap.visits.lock().unwrap();
    for px in x..x + num_pixels {
        visits[y * cap.width + px] += 1;
    }
}

fn open_decoder_at_need_out_buffer(
    data: &[u8],
    parallel: bool,
) -> (Option<Box<JxlDecoder>>, usize, usize) {
    let mut dec = JxlDecoderCreate(None);
    if parallel {
        let runner = JxlThreadParallelRunnerCreate(None, 4);
        assert_eq!(
            JxlDecoderSetParallelRunner(dec.as_deref_mut(), Some(JxlThreadParallelRunner), runner),
            JXL_DEC_SUCCESS
        );
        JxlThreadParallelRunnerDestroy(runner);
    }
    let events = (JXL_DEC_BASIC_INFO.0 as i32) | (JXL_DEC_FULL_IMAGE.0 as i32);
    assert_eq!(
        JxlDecoderSubscribeEvents(dec.as_deref_mut(), events),
        JXL_DEC_SUCCESS
    );
    assert_eq!(set_input(dec.as_deref_mut(), data), JXL_DEC_SUCCESS);
    JxlDecoderCloseInput(dec.as_deref_mut());
    assert_eq!(
        JxlDecoderProcessInput(dec.as_deref_mut()),
        JXL_DEC_BASIC_INFO
    );
    let info = get_basic_info(dec.as_deref());
    assert_eq!(
        JxlDecoderProcessInput(dec.as_deref_mut()),
        JXL_DEC_NEED_IMAGE_OUT_BUFFER
    );
    (dec, info.xsize as usize, info.ysize as usize)
}

fn finish_single_frame_decode(mut dec: Option<Box<JxlDecoder>>) {
    assert_eq!(
        JxlDecoderProcessInput(dec.as_deref_mut()),
        JXL_DEC_FULL_IMAGE
    );
    assert_eq!(JxlDecoderProcessInput(dec.as_deref_mut()), JXL_DEC_SUCCESS);
    JxlDecoderDestroy(dec);
}

#[test]
fn image_out_callback_formats_and_parallel() {
    let data = test_resource("conformance_test_images/bicycles.jxl");
    let formats = [
        (
            JxlPixelFormat {
                num_channels: 3,
                data_type: JXL_TYPE_UINT8,
                endianness: JXL_NATIVE_ENDIAN,
                align: 0,
            },
            3usize,
            1usize,
        ),
        (
            JxlPixelFormat {
                num_channels: 4,
                data_type: JXL_TYPE_UINT16,
                endianness: JXL_BIG_ENDIAN,
                align: 8,
            },
            8usize,
            2usize,
        ),
        (
            JxlPixelFormat {
                num_channels: 3,
                data_type: JXL_TYPE_FLOAT16,
                endianness: JXL_LITTLE_ENDIAN,
                align: 0,
            },
            6usize,
            2usize,
        ),
        (
            JxlPixelFormat {
                num_channels: 3,
                data_type: JXL_TYPE_FLOAT,
                endianness: JXL_NATIVE_ENDIAN,
                align: 32,
            },
            12usize,
            4usize,
        ),
    ];

    for (fmt, bpp, sample_align) in formats {
        let expected = decode_with_c_api(&data, JxlPixelFormat { align: 0, ..fmt }, false);

        for parallel in [false, true] {
            let (mut dec, width, height) = open_decoder_at_need_out_buffer(&data, parallel);
            let capture = SimpleCallbackCapture::new(width, height, bpp, sample_align);

            assert_eq!(
                // SAFETY: `capture` outlives decoding and `simple_capture_cb` is thread-safe.
                unsafe {
                    JxlDecoderSetImageOutCallback(
                        dec.as_deref_mut(),
                        Some(&fmt),
                        Some(simple_capture_cb),
                        capture.as_opaque(),
                    )
                },
                JXL_DEC_SUCCESS
            );
            finish_single_frame_decode(dec);
            assert_eq!(capture.into_verified_pixels(), expected);
        }
    }
}

struct MtInitContext {
    capture: SimpleCallbackCapture,
    fail_init: bool,
    init_calls: std::sync::atomic::AtomicUsize,
    destroy_calls: std::sync::atomic::AtomicUsize,
}

impl MtInitContext {
    fn new(width: usize, height: usize, bytes_per_pixel: usize, fail_init: bool) -> Self {
        let (alloc_w, alloc_h) = if fail_init { (0, 0) } else { (width, height) };
        Self {
            capture: SimpleCallbackCapture::new(alloc_w, alloc_h, bytes_per_pixel, 1),
            fail_init,
            init_calls: std::sync::atomic::AtomicUsize::new(0),
            destroy_calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn as_opaque(&self) -> *mut c_void {
        (&raw const *self).cast_mut().cast::<c_void>()
    }
}

struct MtRunSession {
    ctx: *const MtInitContext,
    num_threads: usize,
    num_pixels_per_thread: usize,
    thread_in_use: Vec<std::sync::atomic::AtomicBool>,
}

unsafe extern "C" fn mt_init_cb(
    init_opaque: *mut c_void,
    num_threads: usize,
    num_pixels_per_thread: usize,
) -> *mut c_void {
    // SAFETY: `init_opaque` points to a valid `MtInitContext` outliving decoding.
    let ctx = unsafe { &*(init_opaque.cast::<MtInitContext>()) };
    ctx.init_calls
        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    assert!(num_threads >= 1);
    assert!(num_pixels_per_thread >= 1);
    if ctx.fail_init {
        return std::ptr::null_mut();
    }
    let session = Box::new(MtRunSession {
        ctx: init_opaque.cast::<MtInitContext>(),
        num_threads,
        num_pixels_per_thread,
        thread_in_use: (0..num_threads)
            .map(|_| std::sync::atomic::AtomicBool::new(false))
            .collect(),
    });
    Box::into_raw(session).cast::<c_void>()
}

unsafe extern "C" fn mt_run_cb(
    run_opaque: *mut c_void,
    thread_id: usize,
    x: usize,
    y: usize,
    num_pixels: usize,
    pixels: *const c_void,
) {
    // SAFETY: `run_opaque` was returned by `mt_init_cb` and remains valid until `mt_destroy_cb`.
    let session = unsafe { &*(run_opaque.cast::<MtRunSession>()) };
    assert!(thread_id < session.num_threads);
    assert!(num_pixels <= session.num_pixels_per_thread);
    // Verify no two concurrent invocations ever share the same `thread_id`.
    let was_in_use =
        session.thread_in_use[thread_id].swap(true, std::sync::atomic::Ordering::AcqRel);
    assert!(!was_in_use, "concurrent run_callback on same thread_id");

    // SAFETY: `session.ctx` points to a valid `MtInitContext` outliving decoding, and `pixels` is
    // valid for reads of `num_pixels` samples during this callback.
    unsafe {
        let ctx = &*session.ctx;
        simple_capture_cb(ctx.capture.as_opaque(), x, y, num_pixels, pixels);
    }

    session.thread_in_use[thread_id].store(false, std::sync::atomic::Ordering::Release);
}

unsafe extern "C" fn mt_destroy_cb(run_opaque: *mut c_void) {
    assert!(!run_opaque.is_null());
    // SAFETY: `run_opaque` was allocated via `Box::into_raw` in `mt_init_cb` and all `mt_run_cb`
    // calls have finished.
    let session = unsafe { Box::from_raw(run_opaque.cast::<MtRunSession>()) };
    // SAFETY: `session.ctx` points to a valid `MtInitContext` outliving decoding.
    let ctx = unsafe { &*session.ctx };
    ctx.destroy_calls
        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
}

unsafe extern "C" fn mt_stateless_init_cb(
    init_opaque: *mut c_void,
    num_threads: usize,
    num_pixels_per_thread: usize,
) -> *mut c_void {
    assert!(num_threads >= 1);
    assert!(num_pixels_per_thread >= 1);
    init_opaque
}

unsafe extern "C" fn mt_stateless_run_cb(
    run_opaque: *mut c_void,
    _thread_id: usize,
    x: usize,
    y: usize,
    num_pixels: usize,
    pixels: *const c_void,
) {
    // SAFETY: `run_opaque` is `init_opaque`, pointing to a valid `SimpleCallbackCapture`.
    unsafe { simple_capture_cb(run_opaque, x, y, num_pixels, pixels) };
}

#[test]
fn multithreaded_image_out_callback_lifecycle_and_null_destroy() {
    let data = test_resource("conformance_test_images/bicycles.jxl");
    let fmt = JxlPixelFormat {
        num_channels: 3,
        data_type: JXL_TYPE_UINT8,
        endianness: JXL_NATIVE_ENDIAN,
        align: 0,
    };
    let expected = decode_with_c_api(&data, fmt, false);

    // 1. Full init / run / destroy lifecycle in parallel mode
    {
        let (mut dec, w, h) = open_decoder_at_need_out_buffer(&data, true);
        let ctx = MtInitContext::new(w, h, 3, false);
        assert_eq!(
            // SAFETY: `ctx` outlives decoding and `mt_init_cb`/`mt_run_cb`/`mt_destroy_cb` uphold
            // the multithreaded callback contract.
            unsafe {
                JxlDecoderSetMultithreadedImageOutCallback(
                    dec.as_deref_mut(),
                    Some(&fmt),
                    Some(mt_init_cb),
                    Some(mt_run_cb),
                    Some(mt_destroy_cb),
                    ctx.as_opaque(),
                )
            },
            JXL_DEC_SUCCESS
        );
        finish_single_frame_decode(dec);

        assert_eq!(ctx.init_calls.into_inner(), 1);
        assert_eq!(ctx.destroy_calls.into_inner(), 1);
        assert_eq!(ctx.capture.into_verified_pixels(), expected);
    }

    // 2. Nullable destroy_callback (destroy_callback == None)
    {
        let (mut dec, w, h) = open_decoder_at_need_out_buffer(&data, false);
        let capture = SimpleCallbackCapture::new(w, h, 3, 1);
        assert_eq!(
            // SAFETY: `capture` outlives decoding and `destroy_callback` is intentionally `None`.
            unsafe {
                JxlDecoderSetMultithreadedImageOutCallback(
                    dec.as_deref_mut(),
                    Some(&fmt),
                    Some(mt_stateless_init_cb),
                    Some(mt_stateless_run_cb),
                    None,
                    capture.as_opaque(),
                )
            },
            JXL_DEC_SUCCESS
        );
        finish_single_frame_decode(dec);
        assert_eq!(capture.into_verified_pixels(), expected);
    }

    // 3. init_callback returning NULL fails with JXL_DEC_ERROR without calling destroy_callback
    {
        let (mut dec, w, h) = open_decoder_at_need_out_buffer(&data, false);
        let ctx = MtInitContext::new(w, h, 3, true);
        assert_eq!(
            // SAFETY: `ctx` outlives decoding; `mt_init_cb` returns null to test error propagation.
            unsafe {
                JxlDecoderSetMultithreadedImageOutCallback(
                    dec.as_deref_mut(),
                    Some(&fmt),
                    Some(mt_init_cb),
                    Some(mt_run_cb),
                    Some(mt_destroy_cb),
                    ctx.as_opaque(),
                )
            },
            JXL_DEC_SUCCESS
        );
        assert_eq!(JxlDecoderProcessInput(dec.as_deref_mut()), JXL_DEC_ERROR);
        JxlDecoderDestroy(dec);

        assert_eq!(ctx.init_calls.into_inner(), 1);
        assert_eq!(ctx.destroy_calls.into_inner(), 0);
    }
}

#[test]
fn image_out_callback_streaming_and_flush() {
    let data = test_resource("conformance_test_images/bicycles.jxl");
    let fmt = JxlPixelFormat {
        num_channels: 3,
        data_type: JXL_TYPE_UINT8,
        endianness: JXL_NATIVE_ENDIAN,
        align: 0,
    };
    let expected = decode_with_c_api(&data, fmt, false);

    let mut dec = JxlDecoderCreate(None);
    let events = (JXL_DEC_BASIC_INFO.0 as i32) | (JXL_DEC_FULL_IMAGE.0 as i32);
    assert_eq!(
        JxlDecoderSubscribeEvents(dec.as_deref_mut(), events),
        JXL_DEC_SUCCESS
    );

    let mut pos = 0usize;
    let chunk_size = 256usize;
    let mut capture: Option<Box<SimpleCallbackCapture>> = None;
    let mut flush_successes = 0usize;
    let mut got_full_image = false;

    loop {
        match JxlDecoderProcessInput(dec.as_deref_mut()) {
            JXL_DEC_NEED_MORE_INPUT => {
                if let Some(cap) = capture.as_ref() {
                    let total_pixels = (cap.width * cap.height) as u32;
                    let visits_before = cap.visits.lock().unwrap().iter().sum::<u32>();
                    if JxlDecoderFlushImage(dec.as_deref_mut()) == JXL_DEC_SUCCESS {
                        flush_successes += 1;
                        let visits_after_first = cap.visits.lock().unwrap().iter().sum::<u32>();
                        let delta1 = visits_after_first - visits_before;
                        assert!(delta1 == 0 || delta1 == total_pixels);
                        // Per jxl/decode.h, each FlushImage call visits each pixel 0 or 1 times.
                        assert_eq!(JxlDecoderFlushImage(dec.as_deref_mut()), JXL_DEC_SUCCESS);
                        let visits_after_second = cap.visits.lock().unwrap().iter().sum::<u32>();
                        let delta2 = visits_after_second - visits_after_first;
                        assert!(delta2 == 0 || delta2 == total_pixels);
                    }
                }
                pos -= JxlDecoderReleaseInput(dec.as_deref_mut());
                let end = (pos + chunk_size).min(data.len());
                assert_eq!(
                    set_input(dec.as_deref_mut(), &data[pos..end]),
                    JXL_DEC_SUCCESS
                );
                pos = end;
                if pos == data.len() {
                    JxlDecoderCloseInput(dec.as_deref_mut());
                }
            }
            JXL_DEC_BASIC_INFO => {
                let info = get_basic_info(dec.as_deref());
                capture = Some(Box::new(SimpleCallbackCapture::new(
                    info.xsize as usize,
                    info.ysize as usize,
                    3,
                    1,
                )));
            }
            JXL_DEC_NEED_IMAGE_OUT_BUFFER => {
                let cap_opaque = capture.as_ref().unwrap().as_opaque();
                assert_eq!(
                    // SAFETY: `capture` is heap-allocated and outlives `dec`.
                    unsafe {
                        JxlDecoderSetImageOutCallback(
                            dec.as_deref_mut(),
                            Some(&fmt),
                            Some(simple_capture_cb),
                            cap_opaque,
                        )
                    },
                    JXL_DEC_SUCCESS
                );
                assert_eq!(JxlDecoderFlushImage(dec.as_deref_mut()), JXL_DEC_ERROR);
            }
            JXL_DEC_FULL_IMAGE => got_full_image = true,
            JXL_DEC_SUCCESS => break,
            other => panic!("unexpected status: {other:?}"),
        }
    }

    JxlDecoderDestroy(dec);
    assert!(got_full_image);
    assert!(flush_successes > 0);
    assert_eq!(capture.unwrap().out.into_inner().unwrap(), expected);
}

#[test]
fn image_out_callback_with_extra_channels_and_animation() {
    // 1. Callback + JxlDecoderSetExtraChannelBuffer on the same frame
    {
        let data = test_resource("3x3a_srgb_lossless.jxl");
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

        let (mut ref_dec, w, h) = open_decoder_at_need_out_buffer(&data, false);
        let mut ref_color = vec![0u8; w * h * 3];
        let mut ref_ec = vec![0u8; w * h];
        assert_eq!(
            set_image_out_buffer(ref_dec.as_deref_mut(), &color_fmt, &mut ref_color),
            JXL_DEC_SUCCESS
        );
        assert_eq!(
            set_extra_channel_buffer(ref_dec.as_deref_mut(), &ec_fmt, &mut ref_ec, 0),
            JXL_DEC_SUCCESS
        );
        finish_single_frame_decode(ref_dec);

        let (mut dec, w, h) = open_decoder_at_need_out_buffer(&data, false);
        let capture = SimpleCallbackCapture::new(w, h, 3, 1);
        let mut ec_buf = vec![0u8; w * h];
        assert_eq!(
            set_extra_channel_buffer(dec.as_deref_mut(), &ec_fmt, &mut ec_buf, 0),
            JXL_DEC_SUCCESS
        );
        assert_eq!(
            // SAFETY: `capture` outlives `dec`.
            unsafe {
                JxlDecoderSetImageOutCallback(
                    dec.as_deref_mut(),
                    Some(&color_fmt),
                    Some(simple_capture_cb),
                    capture.as_opaque(),
                )
            },
            JXL_DEC_SUCCESS
        );
        finish_single_frame_decode(dec);

        assert_eq!(capture.into_verified_pixels(), ref_color);
        assert_eq!(ec_buf, ref_ec);
    }

    // 2. Alternating callback and buffer across animation frames
    {
        let data = test_resource("conformance_test_images/animation_icos4d_5.jxl");
        let fmt = JxlPixelFormat {
            num_channels: 4,
            data_type: JXL_TYPE_UINT8,
            endianness: JXL_NATIVE_ENDIAN,
            align: 0,
        };

        let mut ref_frames = Vec::new();
        {
            let (mut dec, w, h) = open_decoder_at_need_out_buffer(&data, false);
            let mut buf = vec![0u8; w * h * 4];
            assert_eq!(
                set_image_out_buffer(dec.as_deref_mut(), &fmt, &mut buf),
                JXL_DEC_SUCCESS
            );
            loop {
                match JxlDecoderProcessInput(dec.as_deref_mut()) {
                    JXL_DEC_NEED_IMAGE_OUT_BUFFER => {
                        assert_eq!(
                            set_image_out_buffer(dec.as_deref_mut(), &fmt, &mut buf),
                            JXL_DEC_SUCCESS
                        );
                    }
                    JXL_DEC_FULL_IMAGE => ref_frames.push(buf.clone()),
                    JXL_DEC_SUCCESS => break,
                    other => panic!("unexpected status: {other:?}"),
                }
            }
            JxlDecoderDestroy(dec);
        }

        let (mut dec, w, h) = open_decoder_at_need_out_buffer(&data, false);
        let mut frame_buf = vec![0u8; w * h * 4];
        let capture = SimpleCallbackCapture::new(w, h, 4, 1);

        for (idx, expected_frame) in ref_frames.iter().enumerate() {
            if idx > 0 {
                assert_eq!(
                    JxlDecoderProcessInput(dec.as_deref_mut()),
                    JXL_DEC_NEED_IMAGE_OUT_BUFFER
                );
            }
            if idx.is_multiple_of(2) {
                capture.out.lock().unwrap().fill(0);
                assert_eq!(
                    // SAFETY: `capture` outlives `dec`.
                    unsafe {
                        JxlDecoderSetImageOutCallback(
                            dec.as_deref_mut(),
                            Some(&fmt),
                            Some(simple_capture_cb),
                            capture.as_opaque(),
                        )
                    },
                    JXL_DEC_SUCCESS
                );
            } else {
                frame_buf.fill(0);
                assert_eq!(
                    set_image_out_buffer(dec.as_deref_mut(), &fmt, &mut frame_buf),
                    JXL_DEC_SUCCESS
                );
            }
            assert_eq!(
                JxlDecoderProcessInput(dec.as_deref_mut()),
                JXL_DEC_FULL_IMAGE
            );
            if idx.is_multiple_of(2) {
                assert_eq!(&*capture.out.lock().unwrap(), expected_frame);
            } else {
                assert_eq!(&frame_buf, expected_frame);
            }
        }
        assert_eq!(JxlDecoderProcessInput(dec.as_deref_mut()), JXL_DEC_SUCCESS);
        JxlDecoderDestroy(dec);
    }
}

#[test]
fn image_out_callback_mutual_exclusion_and_errors() {
    let data = test_resource("lossy_with_icc.jxl");
    let fmt_u8 = JxlPixelFormat {
        num_channels: 3,
        data_type: JXL_TYPE_UINT8,
        endianness: JXL_NATIVE_ENDIAN,
        align: 0,
    };
    let fmt_f32 = JxlPixelFormat {
        num_channels: 3,
        data_type: JXL_TYPE_FLOAT,
        endianness: JXL_NATIVE_ENDIAN,
        align: 0,
    };
    let fmt_invalid_gray = JxlPixelFormat {
        num_channels: 1,
        data_type: JXL_TYPE_UINT8,
        endianness: JXL_NATIVE_ENDIAN,
        align: 0,
    };

    let mut dec = JxlDecoderCreate(None);
    let events = (JXL_DEC_BASIC_INFO.0 as i32) | (JXL_DEC_FULL_IMAGE.0 as i32);
    assert_eq!(
        JxlDecoderSubscribeEvents(dec.as_deref_mut(), events),
        JXL_DEC_SUCCESS
    );

    // Calling before BASIC_INFO must fail
    assert_eq!(
        // SAFETY: `simple_capture_cb` is not invoked because setup fails before BASIC_INFO.
        unsafe {
            JxlDecoderSetImageOutCallback(
                dec.as_deref_mut(),
                Some(&fmt_u8),
                Some(simple_capture_cb),
                std::ptr::null_mut(),
            )
        },
        JXL_DEC_ERROR
    );

    assert_eq!(set_input(dec.as_deref_mut(), &data), JXL_DEC_SUCCESS);
    JxlDecoderCloseInput(dec.as_deref_mut());
    assert_eq!(
        JxlDecoderProcessInput(dec.as_deref_mut()),
        JXL_DEC_BASIC_INFO
    );
    let info = get_basic_info(dec.as_deref());
    let (w, h) = (info.xsize as usize, info.ysize as usize);
    assert_eq!(
        JxlDecoderProcessInput(dec.as_deref_mut()),
        JXL_DEC_NEED_IMAGE_OUT_BUFFER
    );

    // Null callback / missing run_callback must fail
    assert_eq!(
        // SAFETY: Passing `None` callbacks to verify error handling.
        unsafe {
            JxlDecoderSetImageOutCallback(
                dec.as_deref_mut(),
                Some(&fmt_u8),
                None,
                std::ptr::null_mut(),
            )
        },
        JXL_DEC_ERROR
    );
    assert_eq!(
        // SAFETY: Passing `None` `run_callback` to verify error handling.
        unsafe {
            JxlDecoderSetMultithreadedImageOutCallback(
                dec.as_deref_mut(),
                Some(&fmt_u8),
                Some(mt_stateless_init_cb),
                None,
                None,
                std::ptr::null_mut(),
            )
        },
        JXL_DEC_ERROR
    );

    let capture = SimpleCallbackCapture::new(w, h, 3, 1);

    // Setting f32 callback updates TARGET_DATA to Linear sRGB
    assert_eq!(
        // SAFETY: `capture` outlives `dec` and will be replaced by `fmt_u8` before decoding.
        unsafe {
            JxlDecoderSetImageOutCallback(
                dec.as_deref_mut(),
                Some(&fmt_f32),
                Some(simple_capture_cb),
                capture.as_opaque(),
            )
        },
        JXL_DEC_SUCCESS
    );
    assert_eq!(
        get_color_encoding(dec.as_deref(), JXL_COLOR_PROFILE_TARGET_DATA).transfer_function,
        JXL_TRANSFER_FUNCTION_LINEAR
    );

    // Switching from callback to buffer on the same frame must fail
    let mut buf = vec![0u8; w * h * 3];
    assert_eq!(
        set_image_out_buffer(dec.as_deref_mut(), &fmt_u8, &mut buf),
        JXL_DEC_ERROR
    );

    // Replacing callback with another valid callback (u8) succeeds and restores sRGB TARGET_DATA
    assert_eq!(
        // SAFETY: `capture` outlives `dec`.
        unsafe {
            JxlDecoderSetImageOutCallback(
                dec.as_deref_mut(),
                Some(&fmt_u8),
                Some(simple_capture_cb),
                capture.as_opaque(),
            )
        },
        JXL_DEC_SUCCESS
    );
    assert_eq!(
        get_color_encoding(dec.as_deref(), JXL_COLOR_PROFILE_TARGET_DATA).transfer_function,
        JXL_TRANSFER_FUNCTION_SRGB
    );

    // Invalid format (1 channel on RGB image) fails without losing the active u8 callback
    assert_eq!(
        // SAFETY: Invalid format is rejected synchronously.
        unsafe {
            JxlDecoderSetImageOutCallback(
                dec.as_deref_mut(),
                Some(&fmt_invalid_gray),
                Some(simple_capture_cb),
                capture.as_opaque(),
            )
        },
        JXL_DEC_ERROR
    );
    assert_eq!(
        JxlDecoderProcessInput(dec.as_deref_mut()),
        JXL_DEC_FULL_IMAGE
    );
    assert_eq!(
        capture.into_verified_pixels(),
        decode_with_c_api(&data, fmt_u8, false)
    );

    // Rewind and verify buffer -> callback on the same frame also fails
    JxlDecoderRewind(dec.as_deref_mut());
    assert_eq!(set_input(dec.as_deref_mut(), &data), JXL_DEC_SUCCESS);
    JxlDecoderCloseInput(dec.as_deref_mut());
    assert_eq!(
        JxlDecoderProcessInput(dec.as_deref_mut()),
        JXL_DEC_BASIC_INFO
    );
    assert_eq!(
        JxlDecoderProcessInput(dec.as_deref_mut()),
        JXL_DEC_NEED_IMAGE_OUT_BUFFER
    );
    assert_eq!(
        set_image_out_buffer(dec.as_deref_mut(), &fmt_u8, &mut buf),
        JXL_DEC_SUCCESS
    );
    assert_eq!(
        // SAFETY: Switching from buffer to callback is rejected synchronously.
        unsafe {
            JxlDecoderSetImageOutCallback(
                dec.as_deref_mut(),
                Some(&fmt_u8),
                Some(simple_capture_cb),
                std::ptr::null_mut(),
            )
        },
        JXL_DEC_ERROR
    );

    JxlDecoderDestroy(dec);
}

// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::ffi::c_void;
use std::path::PathBuf;

use jxl::api::{
    AlphaMode, Endianness, Event, JxlAuxBoxType, JxlColorType, JxlDataFormat,
    JxlDecoder as RsDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat as RsPixelFormat,
};
use jxl_libjxl_emulation::*;

fn test_resource(rel: &str) -> Vec<u8> {
    let rel_path = PathBuf::from("../jxl/resources/test").join(rel);
    if let Ok(bytes) = std::fs::read(&rel_path) {
        return bytes;
    }
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("../jxl/resources/test");
    path.push(rel);
    std::fs::read(&path).unwrap_or_else(|e| panic!("failed to read {}: {}", path.display(), e))
}

fn testdata_resource(rel: &str) -> Vec<u8> {
    let rel_path = PathBuf::from("../jxl/tests/testdata").join(rel);
    if let Ok(bytes) = std::fs::read(&rel_path) {
        return bytes;
    }
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("../jxl/tests/testdata");
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
        let dec_ptr = JxlDecoderCreate(None);
        assert!(!dec_ptr.is_null());
        let dec = &mut *dec_ptr;

        let events = (JXL_DEC_BASIC_INFO.0 as i32) | (JXL_DEC_COLOR_ENCODING.0 as i32);
        assert_eq!(JxlDecoderSubscribeEvents(dec, events), JXL_DEC_SUCCESS);
        // dng_sdk calls JxlDecoderSetDecompressBoxes unconditionally even when JXL_DEC_BOX is not subscribed
        assert_eq!(JxlDecoderSetDecompressBoxes(dec, JXL_TRUE), JXL_DEC_SUCCESS);
        assert_eq!(
            JxlDecoderSetInput(dec, data.as_ptr(), data.len()),
            JXL_DEC_SUCCESS
        );
        JxlDecoderCloseInput(dec);

        assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_BASIC_INFO);
        // None output pointer is allowed by libjxl to check availability
        assert_eq!(JxlDecoderGetBasicInfo(dec, None), JXL_DEC_SUCCESS);

        let mut info = JxlBasicInfo::default();
        assert_eq!(
            JxlDecoderGetBasicInfo(dec, Some(&mut info)),
            JXL_DEC_SUCCESS
        );
        assert!(info.xsize > 0 && info.ysize > 0);
        assert_eq!(info.num_color_channels, 3);
        assert_eq!(info.have_preview, JXL_FALSE);
        assert_eq!(info.orientation, JXL_ORIENT_IDENTITY);

        assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_COLOR_ENCODING);
        let mut enc = JxlColorEncoding::default();
        assert_eq!(
            JxlDecoderGetColorAsEncodedProfile(dec, JXL_COLOR_PROFILE_TARGET_DATA, Some(&mut enc),),
            JXL_DEC_SUCCESS
        );
        assert_eq!(enc.color_space, JXL_COLOR_SPACE_RGB);
        assert_eq!(enc.transfer_function, JXL_TRANSFER_FUNCTION_SRGB);

        let mut srgb = JxlColorEncoding::default();
        JxlColorEncodingSetToSRGB(&mut srgb, JXL_FALSE);
        assert_eq!(enc, srgb);

        let mut icc_size = 0usize;
        assert_eq!(
            JxlDecoderGetICCProfileSize(dec, JXL_COLOR_PROFILE_TARGET_DATA, Some(&mut icc_size)),
            JXL_DEC_SUCCESS
        );
        assert!(icc_size > 128);
        let mut icc_buf = vec![0u8; icc_size];
        assert_eq!(
            JxlDecoderGetColorAsICCProfile(
                dec,
                JXL_COLOR_PROFILE_TARGET_DATA,
                icc_buf.as_mut_ptr(),
                icc_buf.len(),
            ),
            JXL_DEC_SUCCESS
        );
        assert_eq!(&icc_buf[36..40], b"acsp");

        // Since neither JXL_DEC_FRAME nor JXL_DEC_FULL_IMAGE was subscribed, the next
        // call to JxlDecoderProcessInput must return JXL_DEC_SUCCESS immediately.
        assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_SUCCESS);

        let _ = JxlDecoderReleaseInput(dec);
        JxlDecoderDestroy(dec_ptr);
    }
}

#[test]
fn test_embedded_icc_profile() {
    let data = test_resource("lossy_with_icc.jxl");

    // SAFETY: Exercising C-ABI ICC profile queries with valid buffers.
    unsafe {
        let dec_ptr = JxlDecoderCreate(None);
        let dec = &mut *dec_ptr;
        let events = (JXL_DEC_BASIC_INFO.0 as i32) | (JXL_DEC_COLOR_ENCODING.0 as i32);
        assert_eq!(JxlDecoderSubscribeEvents(dec, events), JXL_DEC_SUCCESS);
        assert_eq!(
            JxlDecoderSetInput(dec, data.as_ptr(), data.len()),
            JXL_DEC_SUCCESS
        );
        JxlDecoderCloseInput(dec);

        assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_BASIC_INFO);
        assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_COLOR_ENCODING);

        // Embedded profile is ICC, so JxlDecoderGetColorAsEncodedProfile for ORIGINAL should fail,
        // while JxlDecoderGetICCProfileSize for ORIGINAL should succeed.
        let mut enc = JxlColorEncoding::default();
        assert_eq!(
            JxlDecoderGetColorAsEncodedProfile(
                dec,
                JXL_COLOR_PROFILE_TARGET_ORIGINAL,
                Some(&mut enc),
            ),
            JXL_DEC_ERROR
        );
        let mut orig_icc_size = 0usize;
        assert_eq!(
            JxlDecoderGetICCProfileSize(
                dec,
                JXL_COLOR_PROFILE_TARGET_ORIGINAL,
                Some(&mut orig_icc_size),
            ),
            JXL_DEC_SUCCESS
        );
        assert!(orig_icc_size > 0);
        let mut orig_icc = vec![0u8; orig_icc_size];
        assert_eq!(
            JxlDecoderGetColorAsICCProfile(
                dec,
                JXL_COLOR_PROFILE_TARGET_ORIGINAL,
                orig_icc.as_mut_ptr(),
                orig_icc.len(),
            ),
            JXL_DEC_SUCCESS
        );

        // Requesting linear sRGB as preferred profile on XYB image should succeed.
        let mut lin_srgb = JxlColorEncoding::default();
        JxlColorEncodingSetToLinearSRGB(&mut lin_srgb, JXL_FALSE);
        assert_eq!(
            JxlDecoderSetPreferredColorProfile(dec, &lin_srgb),
            JXL_DEC_SUCCESS
        );
        let mut data_enc = JxlColorEncoding::default();
        assert_eq!(
            JxlDecoderGetColorAsEncodedProfile(
                dec,
                JXL_COLOR_PROFILE_TARGET_DATA,
                Some(&mut data_enc),
            ),
            JXL_DEC_SUCCESS
        );
        assert_eq!(data_enc, lin_srgb);

        JxlDecoderDestroy(dec_ptr);
    }
}

fn decode_with_c_api(data: &[u8], format: JxlPixelFormat, subscribe_frame: bool) -> Vec<u8> {
    // SAFETY: Valid C-ABI calls with properly sized output buffer.
    unsafe {
        let dec_ptr = JxlDecoderCreate(None);
        let dec = &mut *dec_ptr;
        let mut events = (JXL_DEC_BASIC_INFO.0 as i32) | (JXL_DEC_FULL_IMAGE.0 as i32);
        if subscribe_frame {
            events |= JXL_DEC_FRAME.0 as i32;
        }
        assert_eq!(JxlDecoderSubscribeEvents(dec, events), JXL_DEC_SUCCESS);
        assert_eq!(
            JxlDecoderSetInput(dec, data.as_ptr(), data.len()),
            JXL_DEC_SUCCESS
        );
        JxlDecoderCloseInput(dec);

        assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_BASIC_INFO);
        if subscribe_frame {
            assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_FRAME);
        }
        assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_NEED_IMAGE_OUT_BUFFER);

        let mut buf_size = 0usize;
        assert_eq!(
            JxlDecoderImageOutBufferSize(dec, &format, &mut buf_size),
            JXL_DEC_SUCCESS
        );
        let mut out = vec![0u8; buf_size];
        assert_eq!(
            JxlDecoderSetImageOutBuffer(dec, &format, out.as_mut_ptr().cast::<c_void>(), out.len()),
            JXL_DEC_SUCCESS
        );

        assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_FULL_IMAGE);
        assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_SUCCESS);

        JxlDecoderDestroy(dec_ptr);
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
        let dec_ptr = JxlDecoderCreate(None);
        let dec = &mut *dec_ptr;
        assert_eq!(
            JxlDecoderSubscribeEvents(dec, JXL_DEC_FULL_IMAGE.0 as i32),
            JXL_DEC_SUCCESS
        );
        assert_eq!(
            JxlDecoderSetInput(dec, data.as_ptr(), data.len()),
            JXL_DEC_SUCCESS
        );
        JxlDecoderCloseInput(dec);
        assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_NEED_IMAGE_OUT_BUFFER);
        let mut size = 0usize;
        assert_eq!(
            JxlDecoderImageOutBufferSize(dec, &fmt_strided, &mut size),
            JXL_DEC_SUCCESS
        );
        let bytes_per_row = 3 * 3 * 4; // 36 bytes
        let stride = 51; // next multiple of 17 >= 36
        assert_eq!(size, 2 * stride + bytes_per_row);

        let mut backing = vec![0u8; size + 8];
        let unaligned_ptr = backing.as_mut_ptr().add(1);
        assert_eq!(
            JxlDecoderSetImageOutBuffer(dec, &fmt_strided, unaligned_ptr.cast::<c_void>(), size),
            JXL_DEC_SUCCESS
        );
        assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_FULL_IMAGE);
        assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_SUCCESS);

        for y in 0..3 {
            let row_slice =
                std::slice::from_raw_parts(unaligned_ptr.add(y * stride), bytes_per_row);
            assert_eq!(
                row_slice,
                &expected[y * bytes_per_row..(y + 1) * bytes_per_row]
            );
        }
        JxlDecoderDestroy(dec_ptr);
    }
}

#[test]
fn test_extra_channels_and_unpremultiply_alpha() {
    let data = test_resource("3x3a_srgb_lossless.jxl");

    // SAFETY: Testing extra channel info, planar extra channel buffer, and unpremultiply alpha.
    unsafe {
        let dec_ptr = JxlDecoderCreate(None);
        let dec = &mut *dec_ptr;
        let events = (JXL_DEC_BASIC_INFO.0 as i32) | (JXL_DEC_FULL_IMAGE.0 as i32);
        assert_eq!(JxlDecoderSubscribeEvents(dec, events), JXL_DEC_SUCCESS);
        assert_eq!(
            JxlDecoderSetUnpremultiplyAlpha(dec, JXL_TRUE),
            JXL_DEC_SUCCESS
        );
        assert_eq!(
            JxlDecoderSetInput(dec, data.as_ptr(), data.len()),
            JXL_DEC_SUCCESS
        );
        JxlDecoderCloseInput(dec);

        assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_BASIC_INFO);
        let mut info = JxlBasicInfo::default();
        assert_eq!(
            JxlDecoderGetBasicInfo(dec, Some(&mut info)),
            JXL_DEC_SUCCESS
        );
        assert_eq!(info.num_extra_channels, 1);
        assert!(info.alpha_bits > 0);

        let mut ec_info = JxlExtraChannelInfo::default();
        assert_eq!(
            JxlDecoderGetExtraChannelInfo(dec, 0, &mut ec_info),
            JXL_DEC_SUCCESS
        );
        assert_eq!(ec_info.type_, JXL_CHANNEL_ALPHA);

        assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_NEED_IMAGE_OUT_BUFFER);

        // Reject 1-channel format on RGB image
        let invalid_gray_fmt = JxlPixelFormat {
            num_channels: 1,
            data_type: JXL_TYPE_UINT8,
            endianness: JXL_NATIVE_ENDIAN,
            align: 0,
        };
        let mut dummy_size = 0usize;
        assert_eq!(
            JxlDecoderImageOutBufferSize(dec, &invalid_gray_fmt, &mut dummy_size),
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
            JxlDecoderImageOutBufferSize(dec, &color_fmt, &mut color_size),
            JXL_DEC_SUCCESS
        );
        assert_eq!(
            JxlDecoderExtraChannelBufferSize(dec, &ec_fmt, &mut ec_size, 0),
            JXL_DEC_SUCCESS
        );
        assert_eq!(color_size, 3 * 3 * 3);
        assert_eq!(ec_size, 3 * 3);

        let mut color_buf = vec![0u8; color_size];
        let mut ec_buf = vec![0u8; ec_size];
        // Setting extra channel buffer before main image buffer must still require main buffer!
        assert_eq!(
            JxlDecoderSetExtraChannelBuffer(
                dec,
                &ec_fmt,
                ec_buf.as_mut_ptr().cast::<c_void>(),
                ec_buf.len(),
                0,
            ),
            JXL_DEC_SUCCESS
        );
        assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_NEED_IMAGE_OUT_BUFFER);
        assert_eq!(
            JxlDecoderSetImageOutBuffer(
                dec,
                &color_fmt,
                color_buf.as_mut_ptr().cast::<c_void>(),
                color_buf.len(),
            ),
            JXL_DEC_SUCCESS
        );

        assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_FULL_IMAGE);
        assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_SUCCESS);

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

        JxlDecoderDestroy(dec_ptr);
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
        let dec_ptr = JxlDecoderCreate(None);
        let dec = &mut *dec_ptr;
        let events = (JXL_DEC_BASIC_INFO.0 as i32) | (JXL_DEC_FULL_IMAGE.0 as i32);
        assert_eq!(JxlDecoderSubscribeEvents(dec, events), JXL_DEC_SUCCESS);

        let mut pos = 0usize;
        let chunk_size = 5usize;
        let mut capture = CallbackCapture {
            width: 0,
            bytes_per_pixel: 3,
            data: Vec::new(),
        };
        let mut got_full_image = false;

        loop {
            let status = JxlDecoderProcessInput(dec);
            match status {
                JXL_DEC_NEED_MORE_INPUT => {
                    let rem = JxlDecoderReleaseInput(dec);
                    pos -= rem;
                    assert!(pos < data.len(), "unexpected need more input at EOF");
                    let end = (pos + chunk_size).min(data.len());
                    assert_eq!(
                        JxlDecoderSetInput(dec, data[pos..end].as_ptr(), end - pos),
                        JXL_DEC_SUCCESS
                    );
                    pos = end;
                    if pos == data.len() {
                        JxlDecoderCloseInput(dec);
                    }
                }
                JXL_DEC_BASIC_INFO => {
                    let mut info = JxlBasicInfo::default();
                    assert_eq!(
                        JxlDecoderGetBasicInfo(dec, Some(&mut info)),
                        JXL_DEC_SUCCESS
                    );
                    capture.width = info.xsize as usize;
                    capture.data = vec![0u8; (info.xsize as usize) * (info.ysize as usize) * 3];
                }
                JXL_DEC_NEED_IMAGE_OUT_BUFFER => {
                    assert_eq!(
                        JxlDecoderSetImageOutCallback(
                            dec,
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
        JxlDecoderDestroy(dec_ptr);
    }
}

#[test]
fn test_animation_skip_frames_and_rewind() {
    let data = test_resource("conformance_test_images/animation_icos4d_5.jxl");

    // SAFETY: Testing multi-frame animation, frame skipping, and decoder rewind.
    unsafe {
        let dec_ptr = JxlDecoderCreate(None);
        let dec = &mut *dec_ptr;
        let events = (JXL_DEC_BASIC_INFO.0 as i32)
            | (JXL_DEC_FRAME.0 as i32)
            | (JXL_DEC_FULL_IMAGE.0 as i32);
        assert_eq!(JxlDecoderSubscribeEvents(dec, events), JXL_DEC_SUCCESS);
        assert_eq!(
            JxlDecoderSetInput(dec, data.as_ptr(), data.len()),
            JXL_DEC_SUCCESS
        );
        JxlDecoderCloseInput(dec);

        assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_BASIC_INFO);
        let mut info = JxlBasicInfo::default();
        assert_eq!(
            JxlDecoderGetBasicInfo(dec, Some(&mut info)),
            JXL_DEC_SUCCESS
        );
        assert_eq!(info.have_animation, JXL_TRUE);

        let fmt = JxlPixelFormat {
            num_channels: 4,
            data_type: JXL_TYPE_UINT8,
            endianness: JXL_NATIVE_ENDIAN,
            align: 0,
        };
        let mut buf_size = 0usize;
        assert_eq!(
            JxlDecoderImageOutBufferSize(dec, &fmt, &mut buf_size),
            JXL_DEC_SUCCESS
        );
        let mut frame_buf = vec![0u8; buf_size];

        let mut frame_count = 0usize;
        let mut need_buf_count = 0usize;
        loop {
            let st = JxlDecoderProcessInput(dec);
            match st {
                JXL_DEC_FRAME => {
                    let mut fh = JxlFrameHeader::default();
                    assert_eq!(JxlDecoderGetFrameHeader(dec, &mut fh), JXL_DEC_SUCCESS);
                    assert_eq!(fh.layer_info.xsize, info.xsize);
                    assert_eq!(fh.layer_info.ysize, info.ysize);
                }
                JXL_DEC_NEED_IMAGE_OUT_BUFFER => {
                    need_buf_count += 1;
                    assert_eq!(
                        JxlDecoderSetImageOutBuffer(
                            dec,
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
        assert_eq!(need_buf_count, frame_count);

        // Rewind and skip the first 2 frames
        JxlDecoderRewind(dec);
        JxlDecoderSkipFrames(dec, 2);
        assert_eq!(
            JxlDecoderSetInput(dec, data.as_ptr(), data.len()),
            JXL_DEC_SUCCESS
        );
        JxlDecoderCloseInput(dec);
        assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_BASIC_INFO);

        let mut after_skip_count = 0usize;
        loop {
            match JxlDecoderProcessInput(dec) {
                JXL_DEC_FRAME => {}
                JXL_DEC_NEED_IMAGE_OUT_BUFFER => {
                    assert_eq!(
                        JxlDecoderSetImageOutBuffer(
                            dec,
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

        JxlDecoderDestroy(dec_ptr);
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
        let runner = JxlThreadParallelRunnerCreate(None, 2);
        assert!(!runner.is_null());

        let dec_ptr = JxlDecoderCreate(None);
        let dec = &mut *dec_ptr;
        assert_eq!(
            JxlDecoderSetParallelRunner(dec, Some(JxlThreadParallelRunner), runner),
            JXL_DEC_SUCCESS
        );
        assert_eq!(
            JxlDecoderSubscribeEvents(dec, JXL_DEC_FULL_IMAGE.0 as i32),
            JXL_DEC_SUCCESS
        );
        assert_eq!(
            JxlDecoderSetInput(dec, data.as_ptr(), data.len()),
            JXL_DEC_SUCCESS
        );
        JxlDecoderCloseInput(dec);
        assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_NEED_IMAGE_OUT_BUFFER);
        let mut out = vec![0u8; expected.len()];
        assert_eq!(
            JxlDecoderSetImageOutBuffer(dec, &fmt, out.as_mut_ptr().cast::<c_void>(), out.len()),
            JXL_DEC_SUCCESS
        );
        assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_FULL_IMAGE);
        assert_eq!(out, expected);

        JxlDecoderDestroy(dec_ptr);
        JxlThreadParallelRunnerDestroy(runner);

        let r_runner = JxlResizableParallelRunnerCreate(None);
        assert!(!r_runner.is_null());
        let suggested = JxlResizableParallelRunnerSuggestThreads(512, 512);
        assert!(suggested >= 1);
        JxlResizableParallelRunnerSetThreads(r_runner, suggested as usize);
        JxlResizableParallelRunnerDestroy(r_runner);
    }
}

fn read_exif_with_rs(data: &[u8], decompress: bool) -> Vec<u8> {
    let mut opts = JxlDecoderOptions::default();
    opts.request_aux_boxes = vec![JxlAuxBoxType::EXIF];
    opts.scan_frames_only = true;
    let mut dec = RsDecoder::new(opts);
    dec.close_input();
    let mut input = data;
    loop {
        match dec.process(&mut input, None, None).unwrap() {
            Event::Complete => break,
            Event::NeedMoreInput { .. } => panic!("unexpected need more input"),
            _ => {}
        }
    }
    if let Some(tb) = dec.trailing_box() {
        #[cfg(feature = "brotli")]
        if decompress {
            return tb.data(input).unwrap().into_owned();
        }
        assert!(
            !decompress || !tb.is_compressed(),
            "brotli feature disabled"
        );
        let mut v = tb.raw_data().to_vec();
        v.extend_from_slice(input);
        v
    } else {
        let b = &dec.aux_boxes(JxlAuxBoxType::EXIF)[0];
        #[cfg(feature = "brotli")]
        if decompress {
            return b.data(&[]).unwrap().into_owned();
        }
        assert!(!decompress || !b.is_compressed(), "brotli feature disabled");
        b.raw_data().to_vec()
    }
}

#[test]
fn test_box_extraction_and_decompression() {
    let files = [
        ("exif.jxl", false, false),
        ("exif_brob.jxl", true, false),
        ("exif_trailing_finite.jxl", false, false),
        ("exif_brob_trailing_finite.jxl", true, false),
        ("exif_trailing_infinite.jxl", false, true),
        ("exif_brob_trailing_infinite.jxl", true, true),
    ];

    for (name, is_brob, is_infinite) in files {
        let data = testdata_resource(name);
        for decompress in [true, false] {
            if is_brob && decompress && !cfg!(feature = "brotli") {
                continue;
            }
            let expected = read_exif_with_rs(&data, decompress);

            // SAFETY: Exercising `JXL_DEC_BOX` with small incremental output buffers (16 bytes)
            // to test `JXL_DEC_BOX_NEED_MORE_OUTPUT` and `JxlDecoderReleaseBoxBuffer`.
            unsafe {
                let dec_ptr = JxlDecoderCreate(None);
                let dec = &mut *dec_ptr;
                let events = (JXL_DEC_BASIC_INFO.0 as i32) | (JXL_DEC_BOX.0 as i32);
                assert_eq!(JxlDecoderSubscribeEvents(dec, events), JXL_DEC_SUCCESS);
                assert_eq!(
                    JxlDecoderSetDecompressBoxes(
                        dec,
                        if decompress { JXL_TRUE } else { JXL_FALSE },
                    ),
                    JXL_DEC_SUCCESS
                );
                assert_eq!(
                    JxlDecoderSetInput(dec, data.as_ptr(), data.len()),
                    JXL_DEC_SUCCESS
                );
                JxlDecoderCloseInput(dec);

                let mut box_buf = Vec::<u8>::new();
                let mut has_active_box_buf = false;
                let mut seen_boxes = 0usize;

                loop {
                    let st = JxlDecoderProcessInput(dec);
                    match st {
                        JXL_DEC_BASIC_INFO => {
                            let mut info = JxlBasicInfo::default();
                            assert_eq!(
                                JxlDecoderGetBasicInfo(dec, Some(&mut info)),
                                JXL_DEC_SUCCESS
                            );
                            assert_eq!(info.have_container, JXL_TRUE);
                        }
                        JXL_DEC_BOX => {
                            if has_active_box_buf {
                                let rem = JxlDecoderReleaseBoxBuffer(dec);
                                box_buf.truncate(box_buf.len() - rem);
                            }
                            seen_boxes += 1;
                            let mut raw_type: JxlBoxType = [0; 4];
                            assert_eq!(
                                JxlDecoderGetBoxType(dec, &mut raw_type, JXL_FALSE),
                                JXL_DEC_SUCCESS
                            );
                            let raw_u8 = [
                                raw_type[0] as u8,
                                raw_type[1] as u8,
                                raw_type[2] as u8,
                                raw_type[3] as u8,
                            ];
                            assert_eq!(&raw_u8, if is_brob { b"brob" } else { b"Exif" });

                            let mut dec_type: JxlBoxType = [0; 4];
                            assert_eq!(
                                JxlDecoderGetBoxType(dec, &mut dec_type, JXL_TRUE),
                                JXL_DEC_SUCCESS
                            );
                            let dec_u8 = [
                                dec_type[0] as u8,
                                dec_type[1] as u8,
                                dec_type[2] as u8,
                                dec_type[3] as u8,
                            ];
                            assert_eq!(
                                &dec_u8,
                                if is_brob && !decompress {
                                    b"brob"
                                } else {
                                    b"Exif"
                                }
                            );

                            let mut raw_size = 0u64;
                            assert_eq!(
                                JxlDecoderGetBoxSizeRaw(dec, &mut raw_size),
                                JXL_DEC_SUCCESS
                            );
                            if is_infinite {
                                assert_eq!(raw_size, 0);
                            } else {
                                assert!(raw_size > 8);
                            }

                            box_buf.resize(16, 0);
                            assert_eq!(
                                JxlDecoderSetBoxBuffer(dec, box_buf.as_mut_ptr(), box_buf.len()),
                                JXL_DEC_SUCCESS
                            );
                            has_active_box_buf = true;
                        }
                        JXL_DEC_BOX_NEED_MORE_OUTPUT => {
                            let rem = JxlDecoderReleaseBoxBuffer(dec);
                            let written = box_buf.len() - rem;
                            box_buf.resize(written + 16, 0);
                            assert_eq!(
                                JxlDecoderSetBoxBuffer(
                                    dec,
                                    box_buf[written..].as_mut_ptr(),
                                    box_buf.len() - written,
                                ),
                                JXL_DEC_SUCCESS
                            );
                        }
                        JXL_DEC_SUCCESS => {
                            if has_active_box_buf {
                                let rem = JxlDecoderReleaseBoxBuffer(dec);
                                box_buf.truncate(box_buf.len() - rem);
                            }
                            break;
                        }
                        other => panic!(
                            "unexpected status {:?} for {} (decompress={})",
                            other, name, decompress
                        ),
                    }
                }

                assert_eq!(seen_boxes, 1, "file={}", name);
                assert_eq!(
                    box_buf, expected,
                    "mismatch for {} (decompress={})",
                    name, decompress
                );
                JxlDecoderDestroy(dec_ptr);
            }
        }
    }
}

#[test]
fn test_box_complete_and_rewind_resubscribe() {
    let use_brotli = cfg!(feature = "brotli");
    let data = testdata_resource(if use_brotli {
        "exif_brob.jxl"
    } else {
        "exif.jxl"
    });
    let expected = read_exif_with_rs(&data, use_brotli);

    // SAFETY: Exercising JXL_DEC_BOX_COMPLETE and JxlDecoderRewind + JxlDecoderSubscribeEvents.
    unsafe {
        let dec_ptr = JxlDecoderCreate(None);
        assert!(!dec_ptr.is_null());
        let dec = &mut *dec_ptr;

        // Pass 1 (like KImageFormats): subscribe to JXL_DEC_BOX | JXL_DEC_BOX_COMPLETE.
        let events = (JXL_DEC_BOX.0 as i32) | (JXL_DEC_BOX_COMPLETE.0 as i32);
        assert_eq!(JxlDecoderSubscribeEvents(dec, events), JXL_DEC_SUCCESS);
        assert_eq!(
            JxlDecoderSetDecompressBoxes(dec, if use_brotli { JXL_TRUE } else { JXL_FALSE }),
            JXL_DEC_SUCCESS
        );
        assert_eq!(
            JxlDecoderSetInput(dec, data.as_ptr(), data.len()),
            JXL_DEC_SUCCESS
        );
        JxlDecoderCloseInput(dec);

        assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_BOX);
        let mut box_buf = vec![0u8; 16];
        assert_eq!(
            JxlDecoderSetBoxBuffer(dec, box_buf.as_mut_ptr(), box_buf.len()),
            JXL_DEC_SUCCESS
        );

        let mut status = JxlDecoderProcessInput(dec);
        while status == JXL_DEC_BOX_NEED_MORE_OUTPUT {
            let rem = JxlDecoderReleaseBoxBuffer(dec);
            let written = box_buf.len() - rem;
            box_buf.resize(written + 16, 0);
            assert_eq!(
                JxlDecoderSetBoxBuffer(
                    dec,
                    box_buf[written..].as_mut_ptr(),
                    box_buf.len() - written,
                ),
                JXL_DEC_SUCCESS
            );
            status = JxlDecoderProcessInput(dec);
        }
        assert_eq!(status, JXL_DEC_BOX_COMPLETE);
        let rem = JxlDecoderReleaseBoxBuffer(dec);
        box_buf.truncate(box_buf.len() - rem);
        assert_eq!(box_buf, expected);
        assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_SUCCESS);

        // Pass 2 (like libvips / GDAL / GIMP): rewind and subscribe ONLY to JXL_DEC_FULL_IMAGE.
        JxlDecoderReleaseInput(dec);
        JxlDecoderRewind(dec);
        assert_eq!(
            JxlDecoderSubscribeEvents(dec, JXL_DEC_FULL_IMAGE.0 as i32),
            JXL_DEC_SUCCESS
        );
        assert_eq!(
            JxlDecoderSetInput(dec, data.as_ptr(), data.len()),
            JXL_DEC_SUCCESS
        );
        JxlDecoderCloseInput(dec);

        // Must skip straight to JXL_DEC_NEED_IMAGE_OUT_BUFFER (no JXL_DEC_BOX or JXL_DEC_BASIC_INFO).
        assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_NEED_IMAGE_OUT_BUFFER);
        let fmt = JxlPixelFormat {
            num_channels: 3,
            data_type: JXL_TYPE_UINT8,
            endianness: JXL_NATIVE_ENDIAN,
            align: 0,
        };
        let mut size = 0usize;
        assert_eq!(
            JxlDecoderImageOutBufferSize(dec, &fmt, &mut size),
            JXL_DEC_SUCCESS
        );
        let mut pixels = vec![0u8; size];
        assert_eq!(
            JxlDecoderSetImageOutBuffer(dec, &fmt, pixels.as_mut_ptr().cast::<c_void>(), size),
            JXL_DEC_SUCCESS
        );
        assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_FULL_IMAGE);
        assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_SUCCESS);

        JxlDecoderDestroy(dec_ptr);
    }
}

#[test]
fn test_release_input_tracks_parsed_bytes_ffmpeg_loop() {
    let data = test_resource("3x3_srgb_lossless.jxl");
    let fmt = JxlPixelFormat {
        num_channels: 3,
        data_type: JXL_TYPE_UINT8,
        endianness: JXL_NATIVE_ENDIAN,
        align: 0,
    };
    let expected = decode_with_c_api(&data, fmt, true);

    // SAFETY: Mimicking FFmpeg's libjxldec.c loop where JxlDecoderReleaseInput and
    // JxlDecoderSetInput are called after every JxlDecoderProcessInput event.
    unsafe {
        let dec_ptr = JxlDecoderCreate(None);
        let dec = &mut *dec_ptr;
        let events = (JXL_DEC_BASIC_INFO.0 as i32)
            | (JXL_DEC_COLOR_ENCODING.0 as i32)
            | (JXL_DEC_FRAME.0 as i32)
            | (JXL_DEC_FULL_IMAGE.0 as i32);
        assert_eq!(JxlDecoderSubscribeEvents(dec, events), JXL_DEC_SUCCESS);

        let mut remaining_slice = data.as_slice();
        let mut pixels = Vec::new();

        loop {
            assert!(
                !remaining_slice.is_empty(),
                "input prematurely consumed before JXL_DEC_FULL_IMAGE"
            );
            assert_eq!(
                JxlDecoderSetInput(dec, remaining_slice.as_ptr(), remaining_slice.len()),
                JXL_DEC_SUCCESS
            );
            let status = JxlDecoderProcessInput(dec);
            let rem = JxlDecoderReleaseInput(dec);
            remaining_slice = &remaining_slice[remaining_slice.len() - rem..];

            match status {
                JXL_DEC_BASIC_INFO => {
                    assert!(
                        rem > 0 && rem < data.len(),
                        "expected unparsed frame bytes to remain after BasicInfo, got rem={rem}"
                    );
                }
                JXL_DEC_COLOR_ENCODING | JXL_DEC_FRAME => {
                    assert!(rem > 0);
                }
                JXL_DEC_NEED_IMAGE_OUT_BUFFER => {
                    let mut size = 0usize;
                    assert_eq!(
                        JxlDecoderImageOutBufferSize(dec, &fmt, &mut size),
                        JXL_DEC_SUCCESS
                    );
                    pixels.resize(size, 0);
                    assert_eq!(
                        JxlDecoderSetImageOutBuffer(
                            dec,
                            &fmt,
                            pixels.as_mut_ptr().cast::<c_void>(),
                            pixels.len(),
                        ),
                        JXL_DEC_SUCCESS
                    );
                }
                JXL_DEC_FULL_IMAGE => {
                    break;
                }
                other => panic!("unexpected decoder status: {:?}", other),
            }
        }

        assert_eq!(remaining_slice.len(), 0);
        assert_eq!(pixels, expected);
        assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_SUCCESS);
        JxlDecoderDestroy(dec_ptr);
    }
}

#[test]
fn test_unimplemented_features_return_errors() {
    // SAFETY: Verifying unimplemented decoder/encoder/gainmap functions fail cleanly.
    unsafe {
        let dec_ptr = JxlDecoderCreate(None);
        let dec = &mut *dec_ptr;
        assert_eq!(
            JxlDecoderSubscribeEvents(dec, JXL_DEC_NEED_MORE_INPUT.0 as i32),
            JXL_DEC_ERROR
        );
        assert_eq!(
            JxlDecoderSubscribeEvents(
                dec,
                (JXL_DEC_PREVIEW_IMAGE.0
                    | JXL_DEC_FRAME_PROGRESSION.0
                    | JXL_DEC_JPEG_RECONSTRUCTION.0) as i32,
            ),
            JXL_DEC_SUCCESS
        );
        let fmt = JxlPixelFormat::default();
        let mut size = 0usize;
        assert_eq!(
            JxlDecoderPreviewOutBufferSize(dec, &fmt, &mut size),
            JXL_DEC_ERROR
        );
        assert_eq!(
            JxlDecoderSetPreviewOutBuffer(dec, &fmt, std::ptr::null_mut(), 0),
            JXL_DEC_ERROR
        );
        assert_eq!(
            JxlDecoderSetMultithreadedImageOutCallback(
                dec,
                &fmt,
                None,
                None,
                None,
                std::ptr::null_mut(),
            ),
            JXL_DEC_ERROR
        );
        assert_eq!(JxlDecoderFlushImage(dec), JXL_DEC_ERROR);
        assert_eq!(JxlDecoderSetProgressiveDetail(dec, kFrames), JXL_DEC_ERROR);
        assert_eq!(
            JxlDecoderSetJPEGBuffer(dec, std::ptr::null_mut(), 0),
            JXL_DEC_ERROR
        );
        JxlDecoderDestroy(dec_ptr);

        assert_eq!(
            JxlGainMapGetBundleSize(std::ptr::null(), std::ptr::null_mut()),
            JXL_FALSE
        );

        let enc = JxlEncoderCreate(None);
        assert!(enc.is_null());
        assert_eq!(JxlEncoderGetError(enc), JXL_ENC_ERR_NOT_SUPPORTED);

        let mut basic_info = JxlBasicInfo::default();
        JxlEncoderInitBasicInfo(Some(&mut basic_info));
        assert_eq!(basic_info.bits_per_sample, 8);
        assert_eq!(basic_info.num_color_channels, 3);
        assert_eq!(basic_info.orientation, JXL_ORIENT_IDENTITY);

        let mut fh = JxlFrameHeader::default();
        JxlEncoderInitFrameHeader(Some(&mut fh));
        assert_eq!(fh.is_last, JXL_TRUE);

        let mut ec_info = JxlExtraChannelInfo::default();
        JxlEncoderInitExtraChannelInfo(JXL_CHANNEL_ALPHA, Some(&mut ec_info));
        assert_eq!(ec_info.type_, JXL_CHANNEL_ALPHA);
        assert_eq!(ec_info.bits_per_sample, 8);
    }
}

#[test]
fn test_xyb_icc_default_target_data_is_linear_srgb() {
    for (file, expected_space) in [
        ("conformance_test_images/patches.jxl", JXL_COLOR_SPACE_RGB),
        (
            "conformance_test_images/grayscale.jxl",
            JXL_COLOR_SPACE_GRAY,
        ),
    ] {
        let data = test_resource(file);
        // SAFETY: Testing C API color encoding queries on XYB + ICC images.
        unsafe {
            let dec_ptr = JxlDecoderCreate(None);
            let dec = &mut *dec_ptr;
            let events = (JXL_DEC_BASIC_INFO.0 as i32) | (JXL_DEC_COLOR_ENCODING.0 as i32);
            assert_eq!(JxlDecoderSubscribeEvents(dec, events), JXL_DEC_SUCCESS);
            assert_eq!(
                JxlDecoderSetInput(dec, data.as_ptr(), data.len()),
                JXL_DEC_SUCCESS
            );
            JxlDecoderCloseInput(dec);
            assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_BASIC_INFO);
            assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_COLOR_ENCODING);

            let mut orig_enc = JxlColorEncoding::default();
            assert_eq!(
                JxlDecoderGetColorAsEncodedProfile(
                    dec,
                    JXL_COLOR_PROFILE_TARGET_ORIGINAL,
                    Some(&mut orig_enc),
                ),
                JXL_DEC_ERROR
            );

            let mut data_enc = JxlColorEncoding::default();
            assert_eq!(
                JxlDecoderGetColorAsEncodedProfile(
                    dec,
                    JXL_COLOR_PROFILE_TARGET_DATA,
                    Some(&mut data_enc),
                ),
                JXL_DEC_SUCCESS
            );
            assert_eq!(data_enc.color_space, expected_space);
            assert_eq!(data_enc.transfer_function, JXL_TRANSFER_FUNCTION_LINEAR);

            JxlDecoderDestroy(dec_ptr);
        }
    }
}

#[test]
fn test_diff_fuzz_regressions() {
    let data = test_resource("conformance_test_images/animation_spline.jxl");
    let fmt = JxlPixelFormat {
        num_channels: 3,
        data_type: JXL_TYPE_UINT8,
        endianness: JXL_NATIVE_ENDIAN,
        align: 0,
    };

    unsafe extern "C" fn noop_image_out_cb(
        _opaque: *mut c_void,
        _x: usize,
        _y: usize,
        _num_pixels: usize,
        _pixels: *const c_void,
    ) {
    }

    // SAFETY: Exercising C API state machine edge cases found by differential fuzzing.
    unsafe {
        let dec_ptr = JxlDecoderCreate(None);
        let dec = &mut *dec_ptr;
        let events = (JXL_DEC_BASIC_INFO.0 as i32)
            | (JXL_DEC_FRAME.0 as i32)
            | (JXL_DEC_FULL_IMAGE.0 as i32);
        assert_eq!(JxlDecoderSubscribeEvents(dec, events), JXL_DEC_SUCCESS);
        assert_eq!(
            JxlDecoderSetInput(dec, data.as_ptr(), data.len()),
            JXL_DEC_SUCCESS
        );
        JxlDecoderCloseInput(dec);

        assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_BASIC_INFO);
        assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_FRAME);

        // Calling SetImageOutBitDepth before setting an output buffer must not panic.
        let bd = JxlBitDepth {
            type_: JXL_BIT_DEPTH_FROM_PIXEL_FORMAT,
            bits_per_sample: 0,
            exponent_bits_per_sample: 0,
        };
        assert_eq!(JxlDecoderSetImageOutBitDepth(dec, &bd), JXL_DEC_SUCCESS);

        // JxlDecoderSkipCurrentFrame is rejected at JXL_DEC_FRAME (before NEED_IMAGE_OUT_BUFFER).
        assert_eq!(JxlDecoderSkipCurrentFrame(dec), JXL_DEC_ERROR);

        assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_NEED_IMAGE_OUT_BUFFER);
        assert_eq!(
            JxlDecoderSetImageOutCallback(dec, &fmt, Some(noop_image_out_cb), std::ptr::null_mut(),),
            JXL_DEC_SUCCESS
        );
        assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_FULL_IMAGE);

        // Frame 2: callback must be cleared after Frame 1 completed, so NEED_IMAGE_OUT_BUFFER fires.
        assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_FRAME);
        assert_eq!(JxlDecoderProcessInput(dec), JXL_DEC_NEED_IMAGE_OUT_BUFFER);

        // JxlDecoderSkipCurrentFrame is valid at JXL_DEC_NEED_IMAGE_OUT_BUFFER.
        assert_eq!(JxlDecoderSkipCurrentFrame(dec), JXL_DEC_SUCCESS);

        JxlDecoderDestroy(dec_ptr);
    }
}

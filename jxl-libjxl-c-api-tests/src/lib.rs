// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl_libjxl_c_api as _;

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use std::mem::MaybeUninit;
    use std::path::PathBuf;

    use jxl::api::{
        AlphaMode, Endianness, Event, JxlBitDepth, JxlColorType, JxlDataFormat, JxlDecoder,
        JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat,
    };
    use jxl_libjxl_c_api::{
        JXL_BIG_ENDIAN, JXL_LITTLE_ENDIAN, JXL_NATIVE_ENDIAN, JXL_TYPE_FLOAT, JXL_TYPE_FLOAT16,
        JXL_TYPE_UINT8, JXL_TYPE_UINT16, JxlBasicInfo, JxlDataType, JxlEndianness,
        JxlPixelFormat as CJxlPixelFormat,
    };

    #[repr(u32)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum OutputMode {
        Buffer = 0,
        SimpleCallback = 1,
        MultithreadedCallback = 2,
    }

    #[repr(u32)]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum RunnerMode {
        None = 0,
        Thread = 1,
        Resizable = 2,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    struct CHarnessOptions {
        chunk_size: usize,
        format: CJxlPixelFormat,
        output_mode: OutputMode,
        runner_mode: RunnerMode,
        num_threads: usize,
        keep_orientation: bool,
        unpremultiply_alpha: bool,
        decode_ec0: bool,
    }

    impl Default for CHarnessOptions {
        fn default() -> Self {
            Self {
                chunk_size: 0,
                format: CJxlPixelFormat {
                    num_channels: 3,
                    data_type: JXL_TYPE_UINT8,
                    endianness: JXL_NATIVE_ENDIAN,
                    align: 0,
                },
                output_mode: OutputMode::Buffer,
                runner_mode: RunnerMode::None,
                num_threads: 4,
                keep_orientation: false,
                unpremultiply_alpha: false,
                decode_ec0: false,
            }
        }
    }

    #[repr(C)]
    struct CHarnessResult {
        basic_info: JxlBasicInfo,
        frame_count: usize,
        frame_bytes: usize,
        pixels: *mut u8,
        pixels_size: usize,
        ec0_pixels: *mut u8,
        ec0_size: usize,
    }

    struct CHarnessResultGuard(CHarnessResult);

    impl Drop for CHarnessResultGuard {
        fn drop(&mut self) {
            // SAFETY: `self.0` was initialized by a successful `c_harness_decode` call.
            unsafe {
                c_harness_free_result(&raw mut self.0);
            }
        }
    }

    unsafe extern "C" {
        fn c_harness_abi_sizes(opts_size: *mut usize, result_size: *mut usize);
        fn c_harness_decode(
            data: *const u8,
            data_size: usize,
            opts: *const CHarnessOptions,
            out: *mut CHarnessResult,
        ) -> i32;
        fn c_harness_free_result(res: *mut CHarnessResult);
        fn c_harness_exercise_misc_apis() -> i32;

        fn cxx_harness_decode_rgba8(
            data: *const u8,
            data_size: usize,
            out_w: *mut u32,
            out_h: *mut u32,
            out_pixels: *mut *mut u8,
            out_size: *mut usize,
        ) -> i32;
        fn cxx_harness_free(ptr: *mut u8);
    }

    #[derive(Debug, PartialEq, Eq)]
    struct DecodedOutput {
        size: (usize, usize),
        bits_per_sample: u32,
        exponent_bits_per_sample: u32,
        num_extra_channels: usize,
        frame_count: usize,
        pixels: Vec<u8>,
        ec0_pixels: Vec<u8>,
    }

    fn test_resource(rel: &str) -> Vec<u8> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../jxl/resources/test")
            .join(rel);
        std::fs::read(&path).unwrap_or_else(|e| panic!("failed to read {}: {}", path.display(), e))
    }

    /// # Safety
    /// When `len > 0`, `ptr` must be valid for reads of `len` initialized bytes.
    unsafe fn copy_raw_bytes(ptr: *const u8, len: usize) -> Vec<u8> {
        if len == 0 {
            Vec::new()
        } else {
            // SAFETY: Guaranteed by caller contract when `len > 0`.
            unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
        }
    }

    fn run_c_decode(data: &[u8], opts: CHarnessOptions) -> DecodedOutput {
        let mut raw = MaybeUninit::<CHarnessResult>::uninit();
        // SAFETY: `data`, `opts`, and `raw` are valid pointers for the duration of `c_harness_decode`.
        let rc = unsafe {
            c_harness_decode(data.as_ptr(), data.len(), &raw const opts, raw.as_mut_ptr())
        };
        assert_eq!(rc, 0, "c_harness_decode failed with rc={rc} for {opts:?}");

        // SAFETY: `c_harness_decode` returned 0, having fully initialized `*out`.
        let guard = CHarnessResultGuard(unsafe { raw.assume_init() });
        let res = &guard.0;

        // SAFETY: `res.pixels` and `res.ec0_pixels` hold `pixels_size` / `ec0_size` valid bytes
        // until `guard` drops.
        let (pixels, ec0_pixels) = unsafe {
            (
                copy_raw_bytes(res.pixels, res.pixels_size),
                copy_raw_bytes(res.ec0_pixels, res.ec0_size),
            )
        };

        DecodedOutput {
            size: (res.basic_info.xsize as usize, res.basic_info.ysize as usize),
            bits_per_sample: res.basic_info.bits_per_sample,
            exponent_bits_per_sample: res.basic_info.exponent_bits_per_sample,
            num_extra_channels: res.basic_info.num_extra_channels as usize,
            frame_count: res.frame_count,
            pixels,
            ec0_pixels,
        }
    }

    fn c_endianness_to_rs(endianness: JxlEndianness) -> Endianness {
        match endianness {
            JXL_LITTLE_ENDIAN => Endianness::LittleEndian,
            JXL_BIG_ENDIAN => Endianness::BigEndian,
            _ => Endianness::native(),
        }
    }

    fn c_data_format_to_rs(
        data_type: JxlDataType,
        endianness: JxlEndianness,
    ) -> (JxlDataFormat, usize) {
        let endianness = c_endianness_to_rs(endianness);
        match data_type {
            JXL_TYPE_UINT8 => (JxlDataFormat::U8 { bit_depth: 8 }, 1),
            JXL_TYPE_UINT16 => (
                JxlDataFormat::U16 {
                    endianness,
                    bit_depth: 16,
                },
                2,
            ),
            JXL_TYPE_FLOAT16 => (JxlDataFormat::F16 { endianness }, 2),
            JXL_TYPE_FLOAT => (JxlDataFormat::F32 { endianness }, 4),
            other => panic!("unsupported data_type {other:?}"),
        }
    }

    fn run_native_rs_decode(data: &[u8], opts: CHarnessOptions) -> DecodedOutput {
        let mut rs_opts = JxlDecoderOptions::default();
        rs_opts.adjust_orientation = !opts.keep_orientation;
        rs_opts.skip_preview = true;
        if opts.unpremultiply_alpha {
            rs_opts.alpha_mode = AlphaMode::Unpremultiplied;
        }

        let mut dec = JxlDecoder::new(rs_opts);
        dec.close_input();
        let mut input = data;

        assert_eq!(
            dec.process(&mut input, None, None).unwrap(),
            Event::BasicInfo
        );
        let basic_info = dec.basic_info().unwrap().clone();
        let (width, height) = basic_info.size;
        let (bits_per_sample, exponent_bits_per_sample) = match basic_info.bit_depth {
            JxlBitDepth::Int { bits_per_sample } => (bits_per_sample, 0),
            JxlBitDepth::Float {
                bits_per_sample,
                exponent_bits_per_sample,
            } => (bits_per_sample, exponent_bits_per_sample),
        };

        let color_type = match opts.format.num_channels {
            1 => JxlColorType::Grayscale,
            2 => JxlColorType::GrayscaleAlpha,
            3 => JxlColorType::Rgb,
            4 => JxlColorType::Rgba,
            other => panic!("unsupported num_channels {other}"),
        };
        let (data_fmt, bps) = c_data_format_to_rs(opts.format.data_type, opts.format.endianness);
        let mut ec_formats = vec![None; basic_info.extra_channels.len()];
        if opts.decode_ec0 {
            ec_formats[0] = Some(data_fmt);
        }
        dec.set_pixel_format(JxlPixelFormat {
            color_type,
            color_data_format: Some(data_fmt),
            extra_channel_format: ec_formats,
        })
        .unwrap();

        let color_row_bytes = width * (opts.format.num_channels as usize) * bps;
        let color_frame_bytes = color_row_bytes * height;
        let ec0_row_bytes = width * bps;
        let ec0_frame_bytes = ec0_row_bytes * height;

        let mut pixels = Vec::new();
        let mut ec0_pixels = Vec::new();
        let mut frame_count = 0usize;

        loop {
            match dec.process(&mut input, None, None).unwrap() {
                Event::FrameHeader => {
                    let mut frame_color = vec![0u8; color_frame_bytes];
                    let mut frame_ec0 =
                        vec![0u8; if opts.decode_ec0 { ec0_frame_bytes } else { 0 }];
                    let event = if opts.decode_ec0 {
                        let mut bufs = [
                            JxlOutputBuffer::new(&mut frame_color, height, color_row_bytes),
                            JxlOutputBuffer::new(&mut frame_ec0, height, ec0_row_bytes),
                        ];
                        dec.process(&mut input, Some(&mut bufs), None).unwrap()
                    } else {
                        let mut bufs = [JxlOutputBuffer::new(
                            &mut frame_color,
                            height,
                            color_row_bytes,
                        )];
                        dec.process(&mut input, Some(&mut bufs), None).unwrap()
                    };
                    assert!(matches!(event, Event::FrameComplete { .. }));
                    pixels.extend_from_slice(&frame_color);
                    if opts.decode_ec0 {
                        ec0_pixels.extend_from_slice(&frame_ec0);
                    }
                    frame_count += 1;
                }
                Event::Complete => break,
                other => panic!("unexpected jxl-rs event: {other:?}"),
            }
        }

        DecodedOutput {
            size: (width, height),
            bits_per_sample,
            exponent_bits_per_sample,
            num_extra_channels: basic_info.extra_channels.len(),
            frame_count,
            pixels,
            ec0_pixels,
        }
    }

    #[test]
    fn c_harness_misc_apis_and_stubs() {
        let mut c_opts_size = 0usize;
        let mut c_res_size = 0usize;
        // SAFETY: Passing valid pointers to stack `usize` variables.
        unsafe {
            c_harness_abi_sizes(&raw mut c_opts_size, &raw mut c_res_size);
        }
        assert_eq!(c_opts_size, size_of::<CHarnessOptions>());
        assert_eq!(c_res_size, size_of::<CHarnessResult>());

        assert_eq!(
            // SAFETY: `c_harness_exercise_misc_apis` takes no arguments and exercises C API entrypoints.
            unsafe { c_harness_exercise_misc_apis() },
            0
        );
    }

    #[test]
    fn c_harness_pixel_formats_and_modes_match_native_jxl_rs() {
        let data = test_resource("conformance_test_images/bicycles.jxl");
        let formats = [
            CJxlPixelFormat {
                num_channels: 3,
                data_type: JXL_TYPE_UINT8,
                endianness: JXL_NATIVE_ENDIAN,
                align: 0,
            },
            CJxlPixelFormat {
                num_channels: 4,
                data_type: JXL_TYPE_UINT8,
                endianness: JXL_NATIVE_ENDIAN,
                align: 16,
            },
            CJxlPixelFormat {
                num_channels: 3,
                data_type: JXL_TYPE_UINT16,
                endianness: JXL_LITTLE_ENDIAN,
                align: 0,
            },
            CJxlPixelFormat {
                num_channels: 4,
                data_type: JXL_TYPE_UINT16,
                endianness: JXL_BIG_ENDIAN,
                align: 8,
            },
            CJxlPixelFormat {
                num_channels: 3,
                data_type: JXL_TYPE_FLOAT16,
                endianness: JXL_LITTLE_ENDIAN,
                align: 0,
            },
            CJxlPixelFormat {
                num_channels: 3,
                data_type: JXL_TYPE_FLOAT16,
                endianness: JXL_BIG_ENDIAN,
                align: 0,
            },
            CJxlPixelFormat {
                num_channels: 3,
                data_type: JXL_TYPE_FLOAT,
                endianness: JXL_LITTLE_ENDIAN,
                align: 32,
            },
            CJxlPixelFormat {
                num_channels: 4,
                data_type: JXL_TYPE_FLOAT,
                endianness: JXL_BIG_ENDIAN,
                align: 0,
            },
        ];

        for format in formats {
            let base_opts = CHarnessOptions {
                format,
                ..CHarnessOptions::default()
            };
            let expected = run_native_rs_decode(&data, base_opts);

            for (output_mode, runner_mode) in [
                (OutputMode::Buffer, RunnerMode::None),
                (OutputMode::SimpleCallback, RunnerMode::Thread),
                (OutputMode::MultithreadedCallback, RunnerMode::Resizable),
            ] {
                let opts = CHarnessOptions {
                    output_mode,
                    runner_mode,
                    ..base_opts
                };
                let actual = run_c_decode(&data, opts);
                assert_eq!(actual, expected, "mismatch for {opts:?}");
            }
        }
    }

    #[test]
    fn c_harness_grayscale_alpha_orientation_icc_and_animation_equivalency() {
        // 1. Grayscale & Grayscale+Alpha
        let gray_alpha = test_resource("gray_alpha_lossless.jxl");
        for num_channels in [1, 2, 3, 4] {
            let opts = CHarnessOptions {
                format: CJxlPixelFormat {
                    num_channels,
                    data_type: JXL_TYPE_UINT16,
                    endianness: JXL_NATIVE_ENDIAN,
                    align: 0,
                },
                output_mode: OutputMode::SimpleCallback,
                ..CHarnessOptions::default()
            };
            assert_eq!(
                run_c_decode(&gray_alpha, opts),
                run_native_rs_decode(&gray_alpha, opts)
            );
        }

        // 2. Premultiplied alpha + planar extra channel 0 + chunked streaming
        let premul = test_resource("conformance_test_images/alpha_premultiplied.jxl");
        for unpremultiply_alpha in [false, true] {
            let opts = CHarnessOptions {
                chunk_size: 257,
                format: CJxlPixelFormat {
                    num_channels: 4,
                    data_type: JXL_TYPE_UINT8,
                    endianness: JXL_NATIVE_ENDIAN,
                    align: 0,
                },
                output_mode: OutputMode::MultithreadedCallback,
                runner_mode: RunnerMode::Thread,
                unpremultiply_alpha,
                decode_ec0: true,
                ..CHarnessOptions::default()
            };
            assert_eq!(
                run_c_decode(&premul, opts),
                run_native_rs_decode(&premul, opts)
            );
        }

        // 3. EXIF orientation (`keep_orientation` false vs true)
        let oriented = test_resource("orientation6_rotate_90_cw.jxl");
        for keep_orientation in [false, true] {
            let opts = CHarnessOptions {
                keep_orientation,
                ..CHarnessOptions::default()
            };
            assert_eq!(
                run_c_decode(&oriented, opts),
                run_native_rs_decode(&oriented, opts)
            );
        }

        // 4. XYB image with embedded ICC profile in u8 (sRGB) and f32 (Linear sRGB)
        let icc = test_resource("lossy_with_icc.jxl");
        for data_type in [JXL_TYPE_UINT8, JXL_TYPE_FLOAT] {
            let opts = CHarnessOptions {
                chunk_size: 193,
                format: CJxlPixelFormat {
                    num_channels: 3,
                    data_type,
                    endianness: JXL_NATIVE_ENDIAN,
                    align: 0,
                },
                output_mode: OutputMode::SimpleCallback,
                ..CHarnessOptions::default()
            };
            assert_eq!(run_c_decode(&icc, opts), run_native_rs_decode(&icc, opts));
        }

        // 5. Multi-frame animation across buffer and callback modes
        let anim = test_resource("conformance_test_images/animation_icos4d_5.jxl");
        for output_mode in [OutputMode::Buffer, OutputMode::MultithreadedCallback] {
            let opts = CHarnessOptions {
                chunk_size: 512,
                format: CJxlPixelFormat {
                    num_channels: 4,
                    data_type: JXL_TYPE_UINT8,
                    endianness: JXL_NATIVE_ENDIAN,
                    align: 0,
                },
                output_mode,
                runner_mode: RunnerMode::Resizable,
                ..CHarnessOptions::default()
            };
            let actual = run_c_decode(&anim, opts);
            let expected = run_native_rs_decode(&anim, opts);
            assert!(expected.frame_count > 1);
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn cxx_raii_wrappers_match_native_jxl_rs() {
        let data = test_resource("conformance_test_images/bicycles.jxl");
        let expected = run_native_rs_decode(
            &data,
            CHarnessOptions {
                format: CJxlPixelFormat {
                    num_channels: 4,
                    data_type: JXL_TYPE_UINT8,
                    endianness: JXL_NATIVE_ENDIAN,
                    align: 0,
                },
                ..CHarnessOptions::default()
            },
        );

        let mut w = 0u32;
        let mut h = 0u32;
        let mut ptr: *mut u8 = std::ptr::null_mut();
        let mut size = 0usize;
        // SAFETY: All pointers are valid for the duration of `cxx_harness_decode_rgba8`.
        let rc = unsafe {
            cxx_harness_decode_rgba8(
                data.as_ptr(),
                data.len(),
                &raw mut w,
                &raw mut h,
                &raw mut ptr,
                &raw mut size,
            )
        };
        assert_eq!(rc, 0);
        // SAFETY: `cxx_harness_decode_rgba8` returned 0, so `ptr` holds `size` valid bytes until
        // `cxx_harness_free(ptr)`.
        let actual_pixels = unsafe {
            let v = copy_raw_bytes(ptr, size);
            cxx_harness_free(ptr);
            v
        };
        assert_eq!((w as usize, h as usize), expected.size);
        assert_eq!(actual_pixels, expected.pixels);
    }
}

// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

pub mod dec;
pub mod enc;

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use jxl::api::{JxlDecoder, JxlDecoderOptions};

    use crate::dec::{DecodeOutput, OutputDataType, decode_frames};

    fn get_test_file(name: &str) -> PathBuf {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        root.parent().unwrap().join("jxl/resources/test").join(name)
    }

    fn extract_f32_frames(output: &DecodeOutput) -> Vec<Vec<Vec<f32>>> {
        output
            .frames
            .iter()
            .map(|frame| {
                frame
                    .channels
                    .iter()
                    .map(|channel| {
                        let (_, height) = channel.byte_size();
                        // channel.byte_size().0 is width in bytes
                        let bpp = output.data_type.bits_per_sample() / 8;
                        let width_px = channel.byte_size().0 / bpp;

                        let mut pixels = Vec::with_capacity(width_px * height);
                        for y in 0..height {
                            let row_bytes = channel.row(y);
                            for i in 0..width_px {
                                let val = match output.data_type {
                                    OutputDataType::U8 => row_bytes[i] as f32 / 255.0,
                                    OutputDataType::U16 => {
                                        let b = [row_bytes[i * 2], row_bytes[i * 2 + 1]];
                                        u16::from_ne_bytes(b) as f32 / 65535.0
                                    }
                                    OutputDataType::F16 => {
                                        let b = [row_bytes[i * 2], row_bytes[i * 2 + 1]];
                                        half::f16::from_bits(u16::from_ne_bytes(b)).to_f32()
                                    }
                                    OutputDataType::F32 => {
                                        let start = i * 4;
                                        let b = [
                                            row_bytes[start],
                                            row_bytes[start + 1],
                                            row_bytes[start + 2],
                                            row_bytes[start + 3],
                                        ];
                                        f32::from_ne_bytes(b)
                                    }
                                };
                                pixels.push(val);
                            }
                        }
                        pixels
                    })
                    .collect()
            })
            .collect()
    }

    fn do_decode(mut input: &[u8], ty: OutputDataType) -> DecodeOutput {
        let mut decoder = JxlDecoder::new(JxlDecoderOptions::default());
        decode_frames(
            &mut input,
            &mut decoder,
            None,
            None,
            &[ty],
            true,
            true,
            false,
            None,
            false,
            None,
        )
        .unwrap()
        .0
    }

    fn decode_partial(
        mut input: &[u8],
        render_interval: Option<usize>,
    ) -> color_eyre::Result<DecodeOutput> {
        decode_frames(
            &mut input,
            JxlDecoderOptions::default(),
            None,
            None,
            &[OutputDataType::U8],
            true,
            true,
            false,
            render_interval,
            true,
        )
        .map(|(output, _)| output)
    }

    #[test]
    fn test_output_formats_consistency() {
        let test_files = [
            "conformance_test_images/bicycles.jxl",
            "conformance_test_images/bike.jxl",
            "zoltan_tasi_unsplash.jxl",
        ];

        for filename in test_files {
            let path = get_test_file(filename);
            if !path.exists() {
                eprintln!("Skipping {} (not found)", filename);
                continue;
            }
            let file = std::fs::read(&path).unwrap();

            // Decode as f32 (reference)
            let input = file.as_slice();
            let f32_output = do_decode(input, OutputDataType::F32);
            let f32_pixels = extract_f32_frames(&f32_output);

            for (data_type, tolerance, name, clamps_values) in [
                (OutputDataType::U8, 0.004, "u8", true),
                (OutputDataType::U16, 0.0001, "u16", true),
                (OutputDataType::F16, 0.001, "f16", false),
            ] {
                let input = file.as_slice();
                let output = do_decode(input, data_type);
                let pixels = extract_f32_frames(&output);

                assert_eq!(f32_pixels.len(), pixels.len(), "Frame count mismatch");

                for (frame_idx, (ref_frame, test_frame)) in
                    f32_pixels.iter().zip(pixels.iter()).enumerate()
                {
                    assert_eq!(
                        ref_frame.len(),
                        test_frame.len(),
                        "Channel count mismatch frame {}",
                        frame_idx
                    );
                    for (ch_idx, (ref_ch, test_ch)) in
                        ref_frame.iter().zip(test_frame.iter()).enumerate()
                    {
                        assert_eq!(
                            ref_ch.len(),
                            test_ch.len(),
                            "Pixel count mismatch frame {} ch {}",
                            frame_idx,
                            ch_idx
                        );

                        let mut max_diff: f32 = 0.0;
                        for (i, (&ref_val, &test_val)) in
                            ref_ch.iter().zip(test_ch.iter()).enumerate()
                        {
                            let ref_val_clamped = if clamps_values {
                                ref_val.clamp(0.0, 1.0)
                            } else {
                                ref_val
                            };
                            let diff = (ref_val_clamped - test_val).abs();
                            max_diff = max_diff.max(diff);
                            if diff > tolerance {
                                panic!(
                                    "{}: {} mismatch at frame {} ch {} idx {}: ref={}, test={}, diff={}",
                                    filename,
                                    name,
                                    frame_idx,
                                    ch_idx,
                                    i,
                                    ref_val_clamped,
                                    test_val,
                                    diff
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_output_formats_high_precision() {
        let filename = "conformance_test_images/bicycles.jxl";
        let path = get_test_file(filename);
        if !path.exists() {
            return;
        }
        let file = std::fs::read(&path).unwrap();

        let mut options = JxlDecoderOptions::default();
        options.high_precision = true;
        let mut decoder = JxlDecoder::new(options);
        for ty in OutputDataType::ALL {
            let mut input = file.as_slice();
            decode_frames(
                &mut input,
                &mut decoder,
                None,
                None,
                &[*ty],
                true,
                true,
                false,
                None,
                false,
                None,
            )
            .unwrap();
        }
    }

    #[test]
    fn test_partial_file_with_nothing_rendered() {
        let file = std::fs::read(get_test_file("green_queen_vardct_e3.jxl")).unwrap();

        // Past the headers, but well before the LF preview can be rendered.
        let err = decode_partial(&file[..file.len() / 20], None)
            .err()
            .expect("a file with nothing rendered decoded without an error");
        assert!(
            err.to_string().contains("before anything was rendered"),
            "unexpected error: {err}"
        );

        // A fifth of the file renders the LF preview. With frequent partial renders,
        // an earlier flush has already drawn it, and the final flush reports no change.
        for render_interval in [None, Some(1000)] {
            let output = decode_partial(&file[..file.len() / 5], render_interval).unwrap();
            let channel = &output.frames[0].channels[0];
            assert!((0..channel.byte_size().1).any(|y| channel.row(y).iter().any(|&v| v != 0)));
        }

        // A file that ends inside a later frame still returns the frames before it.
        let file = std::fs::read(get_test_file("animation_seek_noise_references.jxl")).unwrap();
        let output = decode_partial(&file[..62], None).unwrap();
        assert_eq!(output.frames.len(), 2);
    }
}

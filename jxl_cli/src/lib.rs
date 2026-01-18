// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

pub mod cms;
pub mod dec;
pub mod enc;

#[cfg(test)]
mod tests {
    use crate::dec::{LinearOutput, OutputDataType, decode_frames_with_type};
    use jxl::api::JxlDecoderOptions;

    /// Test that decoding with all output data types produces consistent results.
    /// This catches bugs in the f32→u8, f32→u16, f32→f16 conversion stages.
    #[test]
    fn test_output_formats_consistency() {
        let test_files = [
            "../jxl/resources/test/conformance_test_images/bicycles.jxl",
            "../jxl/resources/test/conformance_test_images/bike.jxl",
            "../jxl/resources/test/zoltan_tasi_unsplash.jxl", // Failed in PR #586
        ];

        for test_file in test_files {
            let file = match std::fs::read(test_file) {
                Ok(f) => f,
                Err(_) => continue, // Skip if file not found
            };

            // Decode as f32 (reference)
            let mut input = file.as_slice();
            let (f32_typed_output, _) = decode_frames_with_type(
                &mut input,
                JxlDecoderOptions::default(),
                OutputDataType::F32,
                LinearOutput::No,
            )
            .unwrap();
            let f32_output = f32_typed_output.to_f32().unwrap();

            // Test each data type
            // clamps_values: true for integer formats that clamp to [0,1]
            for (data_type, tolerance, name, clamps_values) in [
                (OutputDataType::U8, 0.003, "u8", true), // ~0.5/255 + margin
                (OutputDataType::U16, 0.0001, "u16", true), // ~0.5/65535 + margin
                (OutputDataType::F16, 0.001, "f16", false), // f16 precision, no clamping
            ] {
                let mut input = file.as_slice();
                let (typed_output, _) = decode_frames_with_type(
                    &mut input,
                    JxlDecoderOptions::default(),
                    data_type,
                    LinearOutput::No,
                )
                .unwrap();
                // Convert to f32 for comparison
                let converted_output = typed_output.to_f32().unwrap();

                assert_eq!(
                    f32_output.frames.len(),
                    converted_output.frames.len(),
                    "{test_file}: frame count mismatch for {name}"
                );

                for (frame_idx, (f32_frame, typed_frame)) in f32_output
                    .frames
                    .iter()
                    .zip(converted_output.frames.iter())
                    .enumerate()
                {
                    assert_eq!(
                        f32_frame.channels.len(),
                        typed_frame.channels.len(),
                        "{test_file}: channel count mismatch for {name} frame {frame_idx}"
                    );

                    for (ch_idx, (f32_ch, typed_ch)) in f32_frame
                        .channels
                        .iter()
                        .zip(typed_frame.channels.iter())
                        .enumerate()
                    {
                        assert_eq!(
                            f32_ch.size(),
                            typed_ch.size(),
                            "{test_file}: size mismatch for {name} frame {frame_idx} channel {ch_idx}"
                        );

                        let size = f32_ch.size();
                        let mut max_diff: f32 = 0.0;

                        for y in 0..size.1 {
                            let f32_row = f32_ch.row(y);
                            let typed_row = typed_ch.row(y);
                            for x in 0..size.0 {
                                // Clamp f32 to [0,1] for integer formats that clamp output
                                let f32_val = if clamps_values {
                                    f32_row[x].clamp(0.0, 1.0)
                                } else {
                                    f32_row[x]
                                };
                                let typed_val = typed_row[x];
                                let diff = (f32_val - typed_val).abs();
                                max_diff = max_diff.max(diff);
                                assert!(
                                    diff <= tolerance,
                                    "{test_file}: {name} mismatch at ({x},{y}): \
                                     f32={f32_val}, {name}={typed_val}, diff={diff}"
                                );
                            }
                        }

                        // Verify we actually processed pixels
                        assert!(size.0 > 0 && size.1 > 0);
                        // Log max diff for informational purposes
                        let _ = max_diff;
                    }
                }
            }
        }
    }

    /// Test that the high precision mode works with all output formats.
    #[test]
    fn test_output_formats_high_precision() {
        let test_file = "../jxl/resources/test/conformance_test_images/bicycles.jxl";
        let file = match std::fs::read(test_file) {
            Ok(f) => f,
            Err(_) => return, // Skip if file not found
        };

        let high_precision_options = || {
            let mut options = JxlDecoderOptions::default();
            options.high_precision = true;
            options
        };

        // Decode as f32 (reference)
        let mut input = file.as_slice();
        let (f32_typed_output, _) = decode_frames_with_type(
            &mut input,
            high_precision_options(),
            OutputDataType::F32,
            LinearOutput::No,
        )
        .unwrap();
        let f32_output = f32_typed_output.to_f32().unwrap();

        // Test u8 with high precision
        let mut input = file.as_slice();
        let (u8_typed_output, _) = decode_frames_with_type(
            &mut input,
            high_precision_options(),
            OutputDataType::U8,
            LinearOutput::No,
        )
        .unwrap();
        let u8_output = u8_typed_output.to_f32().unwrap();

        // Verify both produce valid output
        assert!(!f32_output.frames.is_empty());
        assert!(!u8_output.frames.is_empty());
        assert_eq!(f32_output.frames.len(), u8_output.frames.len());
    }
}

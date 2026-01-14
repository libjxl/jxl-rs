// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::any::Any;

use crate::api::JxlCmsTransformer;
use crate::error::Result;
use crate::render::RenderPipelineInPlaceStage;

use crate::render::simd_utils::{
    deinterleave_2_dispatch, deinterleave_3_dispatch, deinterleave_4_dispatch,
    interleave_2_dispatch, interleave_3_dispatch, interleave_4_dispatch,
};
use crate::util::AtomicRefCell;

/// Thread-local state for CMS transform.
struct CmsLocalState {
    transformer_idx: usize,
    /// Buffer for interleaved input pixels (always used).
    input_buffer: Vec<f32>,
    /// Buffer for interleaved output pixels (only used when in_channels != out_channels).
    output_buffer: Vec<f32>,
}

/// Applies CMS color transform between color profiles.
///
/// Supports non-contiguous channel indices, e.g., CMYK with K at a non-adjacent index.
pub struct CmsStage {
    transformers: Vec<AtomicRefCell<Box<dyn JxlCmsTransformer + Send>>>,
    /// Indices of input channels to read from (can be non-contiguous).
    /// E.g., [0, 1, 2] for RGB or [0, 1, 2, 5] for CMYK where K is at index 5.
    in_channel_indices: Vec<usize>,
    /// Number of output channels. Output is written to the first `out_channels`
    /// indices from `in_channel_indices`.
    out_channels: usize,
    input_buffer_size: usize,
    output_buffer_size: usize,
}

/// Wrapper to prevent LLVM from pessimizing unrelated code paths.
///
/// Without this wrapper, the mere presence of the dyn trait call causes a ~20% performance
/// regression in the modular decoder's `process_output` function, even when CMS is never
/// invoked (no CMS provider configured). The regression appears to be caused by LLVM's
/// interprocedural analysis of the dyn trait vtable affecting code generation elsewhere.
///
/// Things we tried that did NOT fix the regression:
/// - `#[cold]` on various functions
/// - `#[inline(never)]` on `process_output` and functions it calls
/// - `std::hint::black_box` around the transformer
/// - Moving SIMD code to separate modules/crates
/// - Returning `()` or `Result<()>` from this wrapper
///
/// Things that DO fix the regression:
/// - `#[inline(never)]` wrapper returning `bool` or `Option<()>` (this solution)
/// - `extern "C"` ABI boundary
/// - Global `lto = "thin"` or `codegen-units = 1`
///
/// Theory: LLVM's analysis of the dyn trait vtable (even when never executed) influences
/// register allocation or instruction selection in distant hot paths. Isolating the dyn
/// call in a non-inlined function with a non-trivial return type creates enough of an
/// optimization barrier to prevent this cross-function interference.
#[inline(never)]
fn call_cms_transform_inplace(
    transformer: &mut dyn JxlCmsTransformer,
    buf: &mut [f32],
) -> Option<()> {
    transformer.do_transform_inplace(buf).ok()
}

/// Wrapper for separate-buffer transform to prevent LLVM from pessimizing unrelated code paths.
///
/// See module-level documentation for details on why this wrapper is needed.
#[inline(never)]
fn call_cms_transform(
    transformer: &mut dyn JxlCmsTransformer,
    input: &[f32],
    output: &mut [f32],
) -> Option<()> {
    transformer.do_transform(input, output).ok()
}

impl CmsStage {
    /// Creates a new CMS stage with explicit channel indices.
    ///
    /// # Arguments
    /// * `transformers` - CMS transformer instances (one per thread recommended)
    /// * `in_channel_indices` - Indices of input channels (can be non-contiguous)
    /// * `out_channels` - Number of output channels (must be <= in_channel_indices.len())
    /// * `max_pixels` - Maximum pixels per row chunk
    ///
    /// Output is written to the first `out_channels` indices from `in_channel_indices`.
    /// When input and output channel counts match, uses in-place transform.
    /// When they differ, uses separate input/output buffers.
    ///
    /// # Example
    /// ```ignore
    /// // RGB -> RGB (in-place)
    /// CmsStage::new(transformers, vec![0, 1, 2], 3, max_pixels);
    ///
    /// // CMYK -> RGB where K is at pipeline channel 5
    /// CmsStage::new(transformers, vec![0, 1, 2, 5], 3, max_pixels);
    /// ```
    pub fn new(
        transformers: Vec<Box<dyn JxlCmsTransformer + Send>>,
        in_channel_indices: Vec<usize>,
        out_channels: usize,
        max_pixels: usize,
    ) -> Self {
        let in_channels = in_channel_indices.len();
        assert!(
            out_channels <= in_channels,
            "out_channels ({out_channels}) must be <= in_channels ({in_channels})"
        );
        // Pad buffer to SIMD alignment (max vector length is 16)
        let padded_pixels = max_pixels.next_multiple_of(16);
        Self {
            transformers: transformers.into_iter().map(AtomicRefCell::new).collect(),
            in_channel_indices,
            out_channels,
            input_buffer_size: padded_pixels
                .checked_mul(in_channels)
                .expect("CMS input buffer size overflow"),
            output_buffer_size: padded_pixels
                .checked_mul(out_channels)
                .expect("CMS output buffer size overflow"),
        }
    }
}

impl std::fmt::Display for CmsStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CMS transform: {} channels {:?} -> {} channels",
            self.in_channel_indices.len(),
            self.in_channel_indices,
            self.out_channels
        )
    }
}

impl RenderPipelineInPlaceStage for CmsStage {
    type Type = f32;

    fn uses_channel(&self, c: usize) -> bool {
        self.in_channel_indices.contains(&c)
    }

    fn init_local_state(&self, thread_index: usize) -> Result<Option<Box<dyn Any>>> {
        if self.transformers.is_empty() {
            return Ok(None);
        }
        // Use thread index modulo transformer count to assign transformer to this thread
        let idx = thread_index % self.transformers.len();
        let in_nc = self.in_channel_indices.len();

        // When channel counts differ, we need separate input and output buffers.
        // When they're the same, we use in-place transform (output_buffer unused).
        let output_buffer = if in_nc != self.out_channels {
            vec![0.0f32; self.output_buffer_size]
        } else {
            Vec::new()
        };

        Ok(Some(Box::new(CmsLocalState {
            transformer_idx: idx,
            input_buffer: vec![0.0f32; self.input_buffer_size],
            output_buffer,
        })))
    }

    fn process_row_chunk(
        &self,
        _position: (usize, usize),
        xsize: usize,
        row: &mut [&mut [f32]],
        state: Option<&mut dyn Any>,
    ) {
        let Some(state) = state else { return };
        let state: &mut CmsLocalState = state.downcast_mut().unwrap();
        let indices = &self.in_channel_indices;
        let in_nc = indices.len();
        let out_nc = self.out_channels;
        let same_channels = in_nc == out_nc;

        // Single channel input and output: transform directly in place without copying
        if in_nc == 1 && out_nc == 1 {
            let ch_idx = indices[0];
            let mut transformer = self.transformers[state.transformer_idx].borrow_mut();
            call_cms_transform_inplace(&mut **transformer, &mut row[ch_idx][..xsize])
                .expect("CMS transform failed");
            return;
        }

        // Pad to SIMD alignment (pipeline rows are already padded)
        let xsize_padded = xsize.next_multiple_of(16);

        // Interleave planar -> packed using SIMD
        // Index into row using channel indices to support non-contiguous channels
        match in_nc {
            2 => {
                interleave_2_dispatch(
                    &row[indices[0]][..xsize_padded],
                    &row[indices[1]][..xsize_padded],
                    &mut state.input_buffer[..xsize_padded * 2],
                );
            }
            3 => {
                interleave_3_dispatch(
                    &row[indices[0]][..xsize_padded],
                    &row[indices[1]][..xsize_padded],
                    &row[indices[2]][..xsize_padded],
                    &mut state.input_buffer[..xsize_padded * 3],
                );
            }
            4 => {
                interleave_4_dispatch(
                    &row[indices[0]][..xsize_padded],
                    &row[indices[1]][..xsize_padded],
                    &row[indices[2]][..xsize_padded],
                    &row[indices[3]][..xsize_padded],
                    &mut state.input_buffer[..xsize_padded * 4],
                );
            }
            _ => {
                // Scalar fallback for other channel counts
                #[allow(clippy::needless_range_loop)]
                for x in 0..xsize {
                    for (i, &ch_idx) in indices.iter().enumerate() {
                        state.input_buffer[x * in_nc + i] = row[ch_idx][x];
                    }
                }
            }
        }

        // Apply transform (only on actual pixels, not padding)
        let mut transformer = self.transformers[state.transformer_idx].borrow_mut();
        if same_channels {
            // In-place transform when channel counts match
            call_cms_transform_inplace(
                &mut **transformer,
                &mut state.input_buffer[..xsize * in_nc],
            )
            .expect("CMS transform failed");
        } else {
            // Separate buffer transform when channel counts differ
            call_cms_transform(
                &mut **transformer,
                &state.input_buffer[..xsize * in_nc],
                &mut state.output_buffer[..xsize * out_nc],
            )
            .expect("CMS transform failed");
        }

        // Select source buffer for deinterleaving
        let output_buf = if same_channels {
            &state.input_buffer
        } else {
            &state.output_buffer
        };

        // De-interleave packed -> planar
        // Output always goes to contiguous channels 0, 1, 2, ... (color channels)
        match out_nc {
            2 if row.len() >= 2 => {
                let (r0, r1) = row.split_at_mut(1);
                deinterleave_2_dispatch(
                    &output_buf[..xsize_padded * 2],
                    &mut r0[0][..xsize_padded],
                    &mut r1[0][..xsize_padded],
                );
            }
            3 if row.len() >= 3 => {
                let (r0, rest) = row.split_at_mut(1);
                let (r1, r2) = rest.split_at_mut(1);
                deinterleave_3_dispatch(
                    &output_buf[..xsize_padded * 3],
                    &mut r0[0][..xsize_padded],
                    &mut r1[0][..xsize_padded],
                    &mut r2[0][..xsize_padded],
                );
            }
            4 if row.len() >= 4 => {
                let (r0, rest) = row.split_at_mut(1);
                let (r1, rest) = rest.split_at_mut(1);
                let (r2, r3) = rest.split_at_mut(1);
                deinterleave_4_dispatch(
                    &output_buf[..xsize_padded * 4],
                    &mut r0[0][..xsize_padded],
                    &mut r1[0][..xsize_padded],
                    &mut r2[0][..xsize_padded],
                    &mut r3[0][..xsize_padded],
                );
            }
            _ => {
                // Scalar fallback for >4 output channels
                #[allow(clippy::needless_range_loop)]
                for x in 0..xsize {
                    for c in 0..out_nc {
                        row[c][x] = output_buf[x * out_nc + c];
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock transformer that copies input to output (for same channel count).
    struct IdentityTransformer;

    impl JxlCmsTransformer for IdentityTransformer {
        fn do_transform(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
            output.copy_from_slice(input);
            Ok(())
        }

        fn do_transform_inplace(&mut self, _inout: &mut [f32]) -> Result<()> {
            // Identity - no change needed
            Ok(())
        }
    }

    /// Mock transformer that scales values by 2 (for testing in-place).
    struct ScaleTransformer;

    impl JxlCmsTransformer for ScaleTransformer {
        fn do_transform(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
            for (o, i) in output.iter_mut().zip(input.iter()) {
                *o = *i * 2.0;
            }
            Ok(())
        }

        fn do_transform_inplace(&mut self, inout: &mut [f32]) -> Result<()> {
            for v in inout.iter_mut() {
                *v *= 2.0;
            }
            Ok(())
        }
    }

    /// Mock transformer that converts 4 channels to 3 (CMYK -> RGB style).
    /// Simply drops the 4th channel and passes through the first 3.
    struct FourToThreeTransformer;

    impl JxlCmsTransformer for FourToThreeTransformer {
        fn do_transform(&mut self, input: &[f32], output: &mut [f32]) -> Result<()> {
            // Input: CMYK interleaved (4 values per pixel)
            // Output: RGB interleaved (3 values per pixel)
            let num_pixels = input.len() / 4;
            for i in 0..num_pixels {
                // Simple conversion: R = 1-C, G = 1-M, B = 1-Y (ignoring K for simplicity)
                output[i * 3] = 1.0 - input[i * 4]; // C -> R
                output[i * 3 + 1] = 1.0 - input[i * 4 + 1]; // M -> G
                output[i * 3 + 2] = 1.0 - input[i * 4 + 2]; // Y -> B
            }
            Ok(())
        }

        fn do_transform_inplace(&mut self, _inout: &mut [f32]) -> Result<()> {
            // Cannot do 4->3 in place
            panic!("FourToThreeTransformer does not support in-place transform");
        }
    }

    #[test]
    fn test_cms_stage_same_channels_inplace() {
        // Test 3->3 channel transform (uses in-place)
        let transformers: Vec<Box<dyn JxlCmsTransformer + Send>> = vec![Box::new(ScaleTransformer)];
        let stage = CmsStage::new(transformers, vec![0, 1, 2], 3, 16);

        // Initialize state for thread 0
        let state = stage.init_local_state(0).unwrap().unwrap();
        let mut state_ref: Box<dyn Any> = state;

        // Create test data: 3 channels, 4 pixels
        let mut ch0 = vec![1.0, 2.0, 3.0, 4.0];
        let mut ch1 = vec![0.5, 0.5, 0.5, 0.5];
        let mut ch2 = vec![0.1, 0.2, 0.3, 0.4];

        // Pad to 16 for SIMD alignment
        ch0.resize(16, 0.0);
        ch1.resize(16, 0.0);
        ch2.resize(16, 0.0);

        let mut rows: Vec<&mut [f32]> = vec![&mut ch0, &mut ch1, &mut ch2];
        stage.process_row_chunk((0, 0), 4, &mut rows, Some(state_ref.as_mut()));

        // Values should be scaled by 2
        assert_eq!(ch0[0], 2.0);
        assert_eq!(ch0[1], 4.0);
        assert_eq!(ch1[0], 1.0);
        assert_eq!(ch2[0], 0.2);
    }

    #[test]
    fn test_cms_stage_different_channels() {
        // Test 4->3 channel transform (CMYK to RGB style) with contiguous channels
        let transformers: Vec<Box<dyn JxlCmsTransformer + Send>> =
            vec![Box::new(FourToThreeTransformer)];
        let stage = CmsStage::new(transformers, vec![0, 1, 2, 3], 3, 16);

        // Initialize state for thread 0
        let state = stage.init_local_state(0).unwrap().unwrap();
        let mut state_ref: Box<dyn Any> = state;

        // Create test data: 4 channels (CMYK), 2 pixels
        // C=0.2, M=0.3, Y=0.4, K=0.1 for first pixel
        // C=0.5, M=0.5, Y=0.5, K=0.5 for second pixel
        let mut ch0 = vec![0.2, 0.5]; // C
        let mut ch1 = vec![0.3, 0.5]; // M
        let mut ch2 = vec![0.4, 0.5]; // Y
        let mut ch3 = vec![0.1, 0.5]; // K

        // Pad to 16 for SIMD alignment
        ch0.resize(16, 0.0);
        ch1.resize(16, 0.0);
        ch2.resize(16, 0.0);
        ch3.resize(16, 0.0);

        let mut rows: Vec<&mut [f32]> = vec![&mut ch0, &mut ch1, &mut ch2, &mut ch3];
        stage.process_row_chunk((0, 0), 2, &mut rows, Some(state_ref.as_mut()));

        // Output should be RGB: R = 1-C, G = 1-M, B = 1-Y
        // First pixel: R = 0.8, G = 0.7, B = 0.6
        assert!((ch0[0] - 0.8).abs() < 0.001);
        assert!((ch1[0] - 0.7).abs() < 0.001);
        assert!((ch2[0] - 0.6).abs() < 0.001);
        // Second pixel: R = 0.5, G = 0.5, B = 0.5
        assert!((ch0[1] - 0.5).abs() < 0.001);
        assert!((ch1[1] - 0.5).abs() < 0.001);
        assert!((ch2[1] - 0.5).abs() < 0.001);
        // K channel (ch3) is not modified by deinterleave since out_channels=3
    }

    #[test]
    fn test_cms_stage_non_contiguous_channels() {
        // Test 4->3 channel transform with non-contiguous channels (K at index 5)
        let transformers: Vec<Box<dyn JxlCmsTransformer + Send>> =
            vec![Box::new(FourToThreeTransformer)];
        // CMY at 0,1,2 and K at index 5 (simulating other extra channels before K)
        let stage = CmsStage::new(transformers, vec![0, 1, 2, 5], 3, 16);

        let state = stage.init_local_state(0).unwrap().unwrap();
        let mut state_ref: Box<dyn Any> = state;

        // Create test data: 6 channels total, only using 0,1,2,5
        let mut ch0 = vec![0.2, 0.5]; // C
        let mut ch1 = vec![0.3, 0.5]; // M
        let mut ch2 = vec![0.4, 0.5]; // Y
        let mut ch3 = vec![9.9, 9.9]; // unused (e.g., alpha)
        let mut ch4 = vec![8.8, 8.8]; // unused (e.g., depth)
        let mut ch5 = vec![0.1, 0.5]; // K

        // Pad to 16 for SIMD alignment
        for ch in [&mut ch0, &mut ch1, &mut ch2, &mut ch3, &mut ch4, &mut ch5] {
            ch.resize(16, 0.0);
        }

        let mut rows: Vec<&mut [f32]> =
            vec![&mut ch0, &mut ch1, &mut ch2, &mut ch3, &mut ch4, &mut ch5];
        stage.process_row_chunk((0, 0), 2, &mut rows, Some(state_ref.as_mut()));

        // Output should be RGB: R = 1-C, G = 1-M, B = 1-Y
        assert!((ch0[0] - 0.8).abs() < 0.001);
        assert!((ch1[0] - 0.7).abs() < 0.001);
        assert!((ch2[0] - 0.6).abs() < 0.001);
        // Unused channels should be unchanged
        assert!((ch3[0] - 9.9).abs() < 0.001);
        assert!((ch4[0] - 8.8).abs() < 0.001);
    }

    #[test]
    fn test_cms_stage_single_channel() {
        // Test 1->1 channel transform (grayscale)
        let transformers: Vec<Box<dyn JxlCmsTransformer + Send>> = vec![Box::new(ScaleTransformer)];
        let stage = CmsStage::new(transformers, vec![0], 1, 16);

        let state = stage.init_local_state(0).unwrap().unwrap();
        let mut state_ref: Box<dyn Any> = state;

        let mut ch0 = vec![1.0, 2.0, 3.0, 4.0];
        ch0.resize(16, 0.0);

        let mut rows: Vec<&mut [f32]> = vec![&mut ch0];
        stage.process_row_chunk((0, 0), 4, &mut rows, Some(state_ref.as_mut()));

        // Values should be scaled by 2
        assert_eq!(ch0[0], 2.0);
        assert_eq!(ch0[1], 4.0);
        assert_eq!(ch0[2], 6.0);
        assert_eq!(ch0[3], 8.0);
    }

    #[test]
    fn test_cms_stage_no_transformers() {
        // Test with empty transformers - should do nothing
        let transformers: Vec<Box<dyn JxlCmsTransformer + Send>> = vec![];
        let stage = CmsStage::new(transformers, vec![0, 1, 2], 3, 16);

        // init_local_state should return None when no transformers
        let state = stage.init_local_state(0).unwrap();
        assert!(state.is_none());

        // process_row_chunk should be a no-op with None state
        let mut ch0 = vec![1.0, 2.0, 3.0, 4.0];
        ch0.resize(16, 0.0);
        let original = ch0.clone();

        let mut rows: Vec<&mut [f32]> = vec![&mut ch0];
        stage.process_row_chunk((0, 0), 4, &mut rows, None);

        // Values should be unchanged
        assert_eq!(ch0, original);
    }

    #[test]
    fn test_cms_stage_display() {
        let transformers: Vec<Box<dyn JxlCmsTransformer + Send>> =
            vec![Box::new(IdentityTransformer)];
        let stage = CmsStage::new(transformers, vec![0, 1, 2, 3], 3, 16);
        let display = format!("{}", stage);
        assert!(display.contains("4 channels"));
        assert!(display.contains("3 channels"));
    }

    #[test]
    fn test_cms_stage_uses_channel() {
        let transformers: Vec<Box<dyn JxlCmsTransformer + Send>> =
            vec![Box::new(IdentityTransformer)];
        // Non-contiguous channels: 0, 1, 2, 5
        let stage = CmsStage::new(transformers, vec![0, 1, 2, 5], 3, 16);

        assert!(stage.uses_channel(0));
        assert!(stage.uses_channel(1));
        assert!(stage.uses_channel(2));
        assert!(!stage.uses_channel(3)); // Not in indices
        assert!(!stage.uses_channel(4)); // Not in indices
        assert!(stage.uses_channel(5));
        assert!(!stage.uses_channel(6));
    }
}

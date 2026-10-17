// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::render::{Channels, ErasedLocalState, RenderPipelineDownsampleStage};
use crate::util::ShiftRightCeil;

/// Box-filter downsampling by `1 << X` horizontally and `1 << Y` vertically.
///
/// Partial batches at image edges average only valid samples. By default all channels
/// are processed; specifying `C` restricts processing to that channel.
#[cfg_attr(not(test), expect(unused))]
pub struct BoxAverageStage<const X: u8, const Y: u8, const C: usize = { usize::MAX }>;

impl<const X: u8, const Y: u8, const C: usize> std::fmt::Display for BoxAverageStage<X, Y, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "box average downsampling (log2 factors {X}, {Y})")
    }
}

impl<const X: u8, const Y: u8, const C: usize> RenderPipelineDownsampleStage
    for BoxAverageStage<X, Y, C>
{
    type InputT = f32;
    type OutputT = f32;
    const DOWNSAMPLE: (u8, u8) = (X, Y);

    fn uses_channel(&self, c: usize) -> bool {
        C == usize::MAX || c == C
    }

    fn process_row_batch(
        &self,
        _position: (usize, usize),
        input_xsize: usize,
        input_rows: &Channels<f32>,
        output: &mut [&mut [f32]],
        _state: Option<&mut ErasedLocalState>,
        _previous_call_was_previous_row: bool,
    ) {
        let xsize = input_xsize.shrc(X);
        let input_ysize = input_rows.num_valid_rows().min(1usize << Y);
        for (c, dst) in output.iter_mut().enumerate() {
            for x in 0..xsize {
                let start = x << X;
                let end = ((x + 1) << X).min(input_xsize);
                let mut sum = 0.0;
                for y in 0..input_ysize {
                    for &sample in &input_rows.row(c, y)[start..end] {
                        sum += sample;
                    }
                }
                dst[x] = sum / ((end - start) * input_ysize) as f32;
            }
        }
    }
}

#[cfg(test)]
mod test {
    use test_log::test;

    use super::*;
    use crate::error::Result;
    use crate::image::Image;
    use crate::render::test::{make_and_run_simple_pipeline, test_stage_consistency};

    #[test]
    fn consistency() -> Result<()> {
        test_stage_consistency(|| BoxAverageStage::<1, 1>, (513, 515), 3)?;
        test_stage_consistency(|| BoxAverageStage::<2, 0>, (513, 515), 3)?;
        test_stage_consistency(|| BoxAverageStage::<0, 2>, (513, 515), 3)?;
        test_stage_consistency(|| BoxAverageStage::<5, 5>, (513, 515), 3)
    }

    #[test]
    fn downsample_edges() -> Result<()> {
        for (size, expected) in [
            ((1, 3), &[1.0][..]),
            ((3, 1), &[1.0][..]),
            ((1, 5), &[1.5, 4.0][..]),
            ((5, 1), &[1.5, 4.0][..]),
            ((5, 5), &[9.0, 11.5, 21.5, 24.0][..]),
        ] {
            let mut input = Image::<f32>::new(size)?;
            for y in 0..size.1 {
                for x in 0..size.0 {
                    input.row_mut(y)[x] = (y * size.0 + x) as f32;
                }
            }
            let output =
                make_and_run_simple_pipeline(BoxAverageStage::<2, 2>, &[input], size, 0, 256)?;
            let output_size = (size.0.shrc(2), size.1.shrc(2));
            for y in 0..output_size.1 {
                assert_eq!(
                    output[0].row(y),
                    &expected[y * output_size.0..(y + 1) * output_size.0]
                );
            }
        }
        Ok(())
    }
}

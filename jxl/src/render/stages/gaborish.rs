// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::render::internal::InOutChannel;
use crate::render::{RenderPipelineInOutStage, RenderPipelineStage};
use crate::simd::{F32SimdVec, simd_function};

/// Apply Gabor-like filter to a channel.
#[derive(Debug)]
pub struct GaborishStage {
    channel: usize,
    kernel_top_bottom: [f32; 3],
    kernel_center: [f32; 3],
}

impl GaborishStage {
    pub fn new(channel: usize, weight1: f32, weight2: f32) -> Self {
        let weight_total = 1.0 + weight1 * 4.0 + weight2 * 4.0;
        let kernel_top_bottom = [weight2, weight1, weight2].map(|x| x / weight_total);
        let kernel_center = [weight1, 1.0, weight1].map(|x| x / weight_total);
        Self {
            channel,
            kernel_top_bottom,
            kernel_center,
        }
    }
}

impl std::fmt::Display for GaborishStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Gaborish filter for channel {}", self.channel)
    }
}

simd_function!(
    gaborish_process_dispatch,
    d: D,
    fn gaborish_process(
        stage: &GaborishStage,
        xsize: usize,
        row: &mut [InOutChannel<f32, f32>],
    ) {
        let (rows_in, ref mut rows_out) = row[0];
        let row_out = &mut rows_out[0];
        for idx in (0..xsize).step_by(D::F32Vec::LEN) {
            let mut sum = D::F32Vec::splat(d, 0f32);
            let row_and_kernel = std::iter::zip(
                rows_in,
                [stage.kernel_top_bottom, stage.kernel_center, stage.kernel_top_bottom],
            );
            for (row_in, kernel) in row_and_kernel {
                for (dx, weight) in kernel.iter().enumerate() {
                    sum = D::F32Vec::load(d, &row_in[idx + dx..]).mul_add(D::F32Vec::splat(d, *weight), sum);
                }
            }
            sum.store(&mut row_out[idx..]);
        }
    }
);

impl RenderPipelineStage for GaborishStage {
    type Type = RenderPipelineInOutStage<f32, f32, 1, 1, 0, 0>;

    fn uses_channel(&self, c: usize) -> bool {
        c == self.channel
    }

    fn process_row_chunk(
        &self,
        _position: (usize, usize),
        xsize: usize,
        row: &mut [InOutChannel<f32, f32>],
        _state: Option<&mut dyn std::any::Any>,
    ) {
        gaborish_process_dispatch(self, xsize, row);
    }
}

#[cfg(test)]
mod test {
    use test_log::test;

    use super::*;
    use crate::error::Result;
    use crate::image::Image;
    use crate::render::test::{create_in_out_rows, make_and_run_simple_pipeline};
    use crate::simd::{ScalarDescriptor, SimdDescriptor, test_all_instruction_sets};
    use crate::util::test::assert_all_almost_abs_eq;

    #[test]
    fn consistency() -> Result<()> {
        crate::render::test::test_stage_consistency::<_, f32, f32>(
            GaborishStage::new(0, 0.115169525, 0.061248592),
            (500, 500),
            1,
        )
    }

    #[test]
    fn checkerboard() -> Result<()> {
        let mut image = Image::new((2, 2))?;
        image.as_rect_mut().row(0).copy_from_slice(&[0.0, 1.0]);
        image.as_rect_mut().row(1).copy_from_slice(&[1.0, 0.0]);

        let stage = GaborishStage::new(0, 0.115169525, 0.061248592);
        let (_, output) =
            make_and_run_simple_pipeline::<_, f32, f32>(stage, &[image], (2, 2), 0, 256)?;
        let output = output[0].as_rect();

        assert_all_almost_abs_eq(output.row(0), &[0.20686048, 0.7931395], 1e-6);
        assert_all_almost_abs_eq(output.row(1), &[0.7931395, 0.20686048], 1e-6);

        Ok(())
    }

    fn gaborish_process_scalar_equivalent<D: SimdDescriptor>(d: D) {
        arbtest::arbtest(|u| {
            let stage = GaborishStage::new(0, 0.115169525, 0.061248592);
            create_in_out_rows!(u, 1, 1, rows, xsize);
            gaborish_process(d, &stage, xsize, rows);
            let simd_result: Vec<Vec<f32>> = rows[0]
                .1
                .iter()
                .map(|inner_slice| inner_slice.to_vec())
                .collect();
            gaborish_process(ScalarDescriptor::new().unwrap(), &stage, xsize, rows);
            let scalar_result: Vec<Vec<f32>> = rows[0]
                .1
                .iter()
                .map(|inner_slice| inner_slice.to_vec())
                .collect();
            for (index, scalar_row) in scalar_result.iter().take(xsize).enumerate() {
                assert_all_almost_abs_eq(scalar_row, &simd_result[index], 1e-8);
            }
            Ok(())
        });
    }

    test_all_instruction_sets!(gaborish_process_scalar_equivalent);
}

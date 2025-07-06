// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::render::{RenderPipelineInOutStage, RenderPipelineStage};

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

impl RenderPipelineStage for GaborishStage {
    type Type = RenderPipelineInOutStage<f32, f32, 1, 1, 0, 0>;

    fn uses_channel(&self, c: usize) -> bool {
        c == self.channel
    }

    fn process_row_chunk(
        &self,
        _position: (usize, usize),
        xsize: usize,
        row: &mut [(&[&[f32]], &mut [&mut [f32]])],
        _state: Option<&mut dyn std::any::Any>,
    ) {
        let (rows_in, ref mut rows_out) = row[0];
        let row_out = &mut rows_out[0][..xsize];

        let kernel_top_bottom = self.kernel_top_bottom;
        let kernel_center = self.kernel_center;

        for (idx, out) in row_out.iter_mut().enumerate() {
            let mut sum = 0f32;
            let row_and_kernel = std::iter::zip(
                rows_in,
                [kernel_top_bottom, kernel_center, kernel_top_bottom],
            );

            for (row_in, kernel) in row_and_kernel {
                for (dx, weight) in kernel.into_iter().enumerate() {
                    sum += row_in[idx + dx] * weight;
                }
            }

            *out = sum;
        }
    }
}

#[cfg(test)]
mod test {
    use test_log::test;

    use super::*;
    use crate::error::Result;
    use crate::image::Image;
    use crate::render::test::make_and_run_simple_pipeline;
    use crate::util::test::assert_all_almost_eq;

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

        assert_all_almost_eq!(output.row(0), &[0.20686048, 0.7931395], 1e-6);
        assert_all_almost_eq!(output.row(1), &[0.7931395, 0.20686048], 1e-6);

        Ok(())
    }
}

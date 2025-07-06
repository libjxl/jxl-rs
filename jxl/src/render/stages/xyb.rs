// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::headers::OpsinInverseMatrix;
use crate::render::{RenderPipelineInPlaceStage, RenderPipelineStage};

/// Convert XYB to linear sRGB, where 1.0 corresponds to `intensity_target` nits.
pub struct XybToLinearSrgbStage {
    first_channel: usize,
    opsin: OpsinInverseMatrix,
    intensity_target: f32,
}

impl XybToLinearSrgbStage {
    pub fn new(first_channel: usize, opsin: OpsinInverseMatrix, intensity_target: f32) -> Self {
        Self {
            first_channel,
            opsin,
            intensity_target,
        }
    }
}

impl std::fmt::Display for XybToLinearSrgbStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let channel = self.first_channel;
        write!(
            f,
            "XYB to linear sRGB for channel [{},{},{}]",
            channel,
            channel + 1,
            channel + 2
        )
    }
}

impl RenderPipelineStage for XybToLinearSrgbStage {
    type Type = RenderPipelineInPlaceStage<f32>;

    fn uses_channel(&self, c: usize) -> bool {
        (self.first_channel..self.first_channel + 3).contains(&c)
    }

    fn process_row_chunk(
        &self,
        _position: (usize, usize),
        xsize: usize,
        row: &mut [&mut [f32]],
        _state: Option<&mut dyn std::any::Any>,
    ) {
        let [row_x, row_y, row_b] = row else {
            panic!(
                "incorrect number of channels; expected 3, found {}",
                row.len()
            );
        };

        let OpsinInverseMatrix {
            inverse_matrix: mat,
            opsin_biases: bias,
            ..
        } = self.opsin;
        let bias_cbrt = bias.map(|x| x.cbrt());
        let intensity_scale = 255.0 / self.intensity_target;

        for idx in 0..xsize {
            let x = row_x[idx];
            let y = row_y[idx];
            let b = row_b[idx];

            // Mix and apply bias
            let l = y + x - bias_cbrt[0];
            let m = y - x - bias_cbrt[1];
            let s = b - bias_cbrt[2];

            // Apply biased inverse gamma and scale (1.0 corresponds to `intensity_target` nits)
            let l = (l * l * l + bias[0]) * intensity_scale;
            let m = (m * m * m + bias[1]) * intensity_scale;
            let s = (s * s * s + bias[2]) * intensity_scale;

            // Apply opsin inverse matrix (linear LMS to linear sRGB)
            let [r, g, b] = crate::util::matmul3_vec(mat, [l, m, s]);
            row_x[idx] = r;
            row_y[idx] = g;
            row_b[idx] = b;
        }
    }
}

#[cfg(test)]
mod test {
    use test_log::test;

    use super::*;
    use crate::error::Result;
    use crate::headers::encodings::Empty;
    use crate::image::Image;
    use crate::render::test::make_and_run_simple_pipeline;
    use crate::util::test::assert_all_almost_eq;

    #[test]
    fn consistency() -> Result<()> {
        crate::render::test::test_stage_consistency::<_, f32, f32>(
            XybToLinearSrgbStage::new(0, OpsinInverseMatrix::default(&Empty {}), 255.0),
            (500, 500),
            3,
        )
    }

    #[test]
    fn srgb_primaries() -> Result<()> {
        let mut input_x = Image::new((3, 1))?;
        let mut input_y = Image::new((3, 1))?;
        let mut input_b = Image::new((3, 1))?;
        input_x
            .as_rect_mut()
            .row(0)
            .copy_from_slice(&[0.028100073, -0.015386105, 0.0]);
        input_y
            .as_rect_mut()
            .row(0)
            .copy_from_slice(&[0.4881882, 0.71478134, 0.2781282]);
        input_b
            .as_rect_mut()
            .row(0)
            .copy_from_slice(&[0.471659, 0.43707693, 0.66613984]);

        let stage = XybToLinearSrgbStage::new(0, OpsinInverseMatrix::default(&Empty {}), 255.0);
        let output = make_and_run_simple_pipeline::<_, f32, f32>(
            stage,
            &[input_x, input_y, input_b],
            (3, 1),
            0,
            256,
        )?
        .1;

        assert_all_almost_eq!(output[0].as_rect().row(0), &[1.0, 0.0, 0.0], 1e-6);
        assert_all_almost_eq!(output[1].as_rect().row(0), &[0.0, 1.0, 0.0], 1e-6);
        assert_all_almost_eq!(output[2].as_rect().row(0), &[0.0, 0.0, 1.0], 1e-6);

        Ok(())
    }
}

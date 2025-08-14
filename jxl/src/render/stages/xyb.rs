// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::headers::OpsinInverseMatrix;
use crate::render::{RenderPipelineInPlaceStage, RenderPipelineStage};
use crate::simd::{F32SimdVec, simd_function};

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

simd_function!(
    xyb_process_dispatch,
    d: D,
    fn xyb_process(
        opsin: &OpsinInverseMatrix,
        intensity_target: f32,
        xsize: usize,
        row_x: &mut [f32],
        row_y: &mut [f32],
        row_b: &mut [f32],
    ) {
        let OpsinInverseMatrix {
            inverse_matrix: mat,
            opsin_biases: bias,
            ..
        } = opsin;
        // TODO(veluca): consider computing the cbrt in advance.
        let bias_cbrt = bias.map(|x| D::F32Vec::splat(d, x.cbrt()));
        let intensity_scale = 255.0 / intensity_target;
        let scaled_bias = bias.map(|x| D::F32Vec::splat(d, x * intensity_scale));
        let mat = mat.map(|x| D::F32Vec::splat(d, x));
        let intensity_scale = D::F32Vec::splat(d, intensity_scale);

        for idx in (0..xsize).step_by(D::F32Vec::LEN) {
            let x = D::F32Vec::load(d, &row_x[idx..]);
            let y = D::F32Vec::load(d, &row_y[idx..]);
            let b = D::F32Vec::load(d, &row_b[idx..]);

            // Mix and apply bias
            let l = y + x - bias_cbrt[0];
            let m = y - x - bias_cbrt[1];
            let s = b - bias_cbrt[2];

            // Apply biased inverse gamma and scale (1.0 corresponds to `intensity_target` nits)
            let l2 = l * l;
            let m2 = m * m;
            let s2 = s * s;
            let scaled_l = l * intensity_scale;
            let scaled_m = m * intensity_scale;
            let scaled_s = s * intensity_scale;
            let l = l2.mul_add(scaled_l, scaled_bias[0]);
            let m = m2.mul_add(scaled_m, scaled_bias[1]);
            let s = s2.mul_add(scaled_s, scaled_bias[2]);

            // Apply opsin inverse matrix (linear LMS to linear sRGB)
            let r = mat[0].mul_add(l, mat[1].mul_add(m, mat[2] * s));
            let g = mat[3].mul_add(l, mat[4].mul_add(m, mat[5] * s));
            let b = mat[6].mul_add(l, mat[7].mul_add(m, mat[8] * s));
            r.store(&mut row_x[idx..]);
            g.store(&mut row_y[idx..]);
            b.store(&mut row_b[idx..]);
        }
    }
);

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

        xyb_process_dispatch(
            &self.opsin,
            self.intensity_target,
            xsize,
            row_x,
            row_y,
            row_b,
        );
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
    use crate::simd::{
        ScalarDescriptor, SimdDescriptor, round_up_size_to_two_cache_lines,
        test_all_instruction_sets,
    };
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

    fn xyb_process_scalar_equivalent<D: SimdDescriptor>(d: D) {
        let opsin = OpsinInverseMatrix::default(&Empty {});
        arbtest::arbtest(|u| {
            let xsize = u.arbitrary_len::<usize>()?;
            let intensity_target = u.arbitrary::<u8>()? as f32 * 2.0 + 1.0;
            let mut row_x = vec![0.0; round_up_size_to_two_cache_lines::<f32>(xsize)];
            let mut row_y = vec![0.0; round_up_size_to_two_cache_lines::<f32>(xsize)];
            let mut row_b = vec![0.0; round_up_size_to_two_cache_lines::<f32>(xsize)];

            for i in 0..xsize {
                row_x[i] = u.arbitrary::<i16>()? as f32 * (1.0 / i16::MAX as f32);
                row_y[i] = u.arbitrary::<i16>()? as f32 * (1.0 / i16::MAX as f32);
                row_b[i] = u.arbitrary::<i16>()? as f32 * (1.0 / i16::MAX as f32);
            }

            let mut scalar_x = row_x.clone();
            let mut scalar_y = row_y.clone();
            let mut scalar_b = row_b.clone();

            xyb_process(
                d,
                &opsin,
                intensity_target,
                xsize,
                &mut row_x,
                &mut row_y,
                &mut row_b,
            );

            xyb_process(
                ScalarDescriptor::new().unwrap(),
                &opsin,
                intensity_target,
                xsize,
                &mut scalar_x,
                &mut scalar_y,
                &mut scalar_b,
            );

            for i in 0..xsize {
                assert!((row_x[i] - scalar_x[i]).abs() < 1e-8);
                assert!((row_y[i] - scalar_y[i]).abs() < 1e-8);
                assert!((row_b[i] - scalar_b[i]).abs() < 1e-8);
            }

            Ok(())
        });
    }

    test_all_instruction_sets!(xyb_process_scalar_equivalent);
}

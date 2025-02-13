// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(firsching): remove once we use this!
#![allow(dead_code)]

use crate::{
    features::spline::Splines,
    render::{RenderPipelineInPlaceStage, RenderPipelineStage},
};

pub struct SplinesStage {
    splines: Splines,
}

impl std::fmt::Display for SplinesStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "splines")
    }
}

impl RenderPipelineStage for SplinesStage {
    type Type = RenderPipelineInPlaceStage<f32>;

    fn uses_channel(&self, c: usize) -> bool {
        c < 3
    }

    fn process_row_chunk(
        &mut self,
        position: (usize, usize),
        xsize: usize,
        row: &mut [&mut [f32]],
    ) {
        self.splines.draw_segments(row, position, xsize);
    }
}

#[cfg(test)]
mod test {
    use crate::features::spline::{Point, QuantizedSpline, Splines};
    use crate::render::test::make_and_run_simple_pipeline;
    use crate::util::test::{self, assert_all_almost_eq, read_pfm};
    use crate::{error::Result, image::Image, render::stages::splines::SplinesStage};
    use test_log::test;

    #[test]
    fn splines_process_row_chunk() -> Result<(), test::Error> {
        let want_image = read_pfm(include_bytes!("../../../resources/test/splines.pfm"))?;
        let target_images = [
            Image::<f32>::new_constant((320, 320), 0.0)?,
            Image::<f32>::new_constant((320, 320), 0.0)?,
            Image::<f32>::new_constant((320, 320), 0.0)?,
        ];
        let size = target_images[0].size();
        let mut splines = Splines::create(
            0,
            vec![QuantizedSpline {
                control_points: vec![
                    (109, 105),
                    (-130, -261),
                    (-66, 193),
                    (227, -52),
                    (-170, 290),
                ],
                color_dct: [
                    [
                        168, 119, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                    ],
                    [
                        9, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0,
                    ],
                    [
                        -10, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                    ],
                ],
                sigma_dct: [
                    4, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0,
                ],
            }],
            vec![Point { x: 9.0, y: 54.0 }],
        );
        splines.initialize_draw_cache(size.0 as u64, size.1 as u64, 0.0, 1.0)?;
        let stage = SplinesStage { splines };
        let output: Vec<Image<f32>> =
            make_and_run_simple_pipeline(stage, &target_images, size, 256)?.1;
        for c in 0..3 {
            for row in 0..size.1 {
                assert_all_almost_eq!(
                    output[c].as_rect().row(row),
                    want_image[c].as_rect().row(row),
                    1e-3
                );
            }
        }
        Ok(())
    }

    #[test]
    fn splines_consistency() -> Result<()> {
        let mut splines = Splines::create(
            0,
            vec![QuantizedSpline {
                control_points: vec![
                    (109, 105),
                    (-130, -261),
                    (-66, 193),
                    (227, -52),
                    (-170, 290),
                ],
                color_dct: [
                    [
                        168, 119, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                    ],
                    [
                        9, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0,
                    ],
                    [
                        -10, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0,
                    ],
                ],
                sigma_dct: [
                    4, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0,
                ],
            }],
            vec![Point { x: 9.0, y: 54.0 }],
        );
        splines.initialize_draw_cache(500, 500, 0.0, 1.0)?;
        let stage = SplinesStage { splines };

        crate::render::test::test_stage_consistency::<_, f32, f32>(stage, (500, 500), 6)
    }
}

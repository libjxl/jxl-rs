// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl_simd::{F32SimdVec, simd_function};

use crate::render::{
    Channels, ChannelsMut, ErasedLocalState, ForEachChunk, RenderPipelineInOutStage,
};

/// Apply Gabor-like filter to color channels (0, 1, 2).
#[derive(Debug)]
pub struct GaborishStage {
    weight0: [f32; 3],
    weight1: [f32; 3],
    weight2: [f32; 3],
}

impl GaborishStage {
    pub fn new(weights: [(f32, f32); 3]) -> Self {
        let mut weight0 = [0.0; 3];
        let mut weight1 = [0.0; 3];
        let mut weight2 = [0.0; 3];
        for c in 0..3 {
            let (w1, w2) = weights[c];
            let weight_total = 1.0 + w1 * 4.0 + w2 * 4.0;
            weight0[c] = 1.0 / weight_total;
            weight1[c] = w1 / weight_total;
            weight2[c] = w2 / weight_total;
        }
        Self {
            weight0,
            weight1,
            weight2,
        }
    }
}

impl std::fmt::Display for GaborishStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Gaborish filter")
    }
}

simd_function!(
    gaborish_process_dispatch,
    d: D,
    fn gaborish_process(
        stage: &GaborishStage,
        xsize: usize,
        input_rows: &Channels<f32>,
        output_rows: &mut ChannelsMut<f32>,
    ) {
        let w0_0 = D::F32Vec::splat(d, stage.weight0[0]);
        let w1_0 = D::F32Vec::splat(d, stage.weight1[0]);
        let w2_0 = D::F32Vec::splat(d, stage.weight2[0]);

        let w0_1 = D::F32Vec::splat(d, stage.weight0[1]);
        let w1_1 = D::F32Vec::splat(d, stage.weight1[1]);
        let w2_1 = D::F32Vec::splat(d, stage.weight2[1]);

        let w0_2 = D::F32Vec::splat(d, stage.weight0[2]);
        let w1_2 = D::F32Vec::splat(d, stage.weight1[2]);
        let w2_2 = D::F32Vec::splat(d, stage.weight2[2]);

        ForEachChunk::<3, 3, 1, 3, 1>::run(
            d,
            xsize,
            input_rows,
            output_rows,
            #[inline(always)]
            |_x, in_view, out_view| {
                macro_rules! filter_channel {
                    ($c:expr, $w0:expr, $w1:expr, $w2:expr) => {{
                        let p00 = in_view.load::<$c, 0, -1>();
                        let p01 = in_view.load::<$c, 0, 0>();
                        let p02 = in_view.load::<$c, 0, 1>();
                        let p10 = in_view.load::<$c, 1, -1>();
                        let p11 = in_view.load::<$c, 1, 0>();
                        let p12 = in_view.load::<$c, 1, 1>();
                        let p20 = in_view.load::<$c, 2, -1>();
                        let p21 = in_view.load::<$c, 2, 0>();
                        let p22 = in_view.load::<$c, 2, 1>();

                        let sum = p11 * $w0;
                        let sum = $w1.mul_add(p01 + p10 + p21 + p12, sum);
                        let sum = $w2.mul_add(p00 + p02 + p20 + p22, sum);
                        out_view.store::<$c, 0>(sum);
                    }};
                }

                filter_channel!(0, w0_0, w1_0, w2_0);
                filter_channel!(1, w0_1, w1_1, w2_1);
                filter_channel!(2, w0_2, w1_2, w2_2);
            },
        );
    }
);

impl RenderPipelineInOutStage for GaborishStage {
    type InputT = f32;
    type OutputT = f32;
    const SHIFT: (u8, u8) = (0, 0);
    const BORDER: (u8, u8) = (1, 1);

    fn uses_channel(&self, c: usize) -> bool {
        c < 3
    }

    fn process_row_chunk(
        &self,
        _position: (usize, usize),
        xsize: usize,
        input_rows: &Channels<f32>,
        output_rows: &mut ChannelsMut<f32>,
        _state: Option<&mut ErasedLocalState>,
        _previous_call_was_previous_row: bool,
    ) {
        gaborish_process_dispatch(self, xsize, input_rows, output_rows);
    }
}

#[cfg(test)]
mod test {
    use test_log::test;

    use super::*;
    use crate::error::Result;
    use crate::image::Image;
    use crate::render::test::make_and_run_simple_pipeline;
    use crate::tests::assert_close;

    #[test]
    fn consistency() -> Result<()> {
        crate::render::test::test_stage_consistency(
            || {
                GaborishStage::new([
                    (0.115169525, 0.061248592),
                    (0.115169525, 0.061248592),
                    (0.115169525, 0.061248592),
                ])
            },
            (500, 500),
            3,
        )
    }

    #[test]
    fn checkerboard() -> Result<()> {
        let mut image0 = Image::new((2, 2))?;
        image0.row_mut(0).copy_from_slice(&[0.0, 1.0]);
        image0.row_mut(1).copy_from_slice(&[1.0, 0.0]);

        let image1 = image0.try_clone()?;
        let image2 = image0.try_clone()?;

        let stage = GaborishStage::new([
            (0.115169525, 0.061248592),
            (0.0, 0.0),
            (0.115169525, 0.061248592),
        ]);
        let output =
            make_and_run_simple_pipeline(stage, &[image0, image1, image2], (2, 2), 0, 256)?;

        // Channels 0 and 2 are filtered
        assert_close!(all, output[0].row(0), &[0.20686048, 0.7931395], 1e-6);
        assert_close!(all, output[0].row(1), &[0.7931395, 0.20686048], 1e-6);
        assert_close!(all, output[2].row(0), &[0.20686048, 0.7931395], 1e-6);
        assert_close!(all, output[2].row(1), &[0.7931395, 0.20686048], 1e-6);

        // Channel 1 is identity (weights 0.0)
        assert_eq!(output[1].row(0), &[0.0, 1.0]);
        assert_eq!(output[1].row(1), &[1.0, 0.0]);

        Ok(())
    }
}

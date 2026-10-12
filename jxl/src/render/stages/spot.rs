// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl_simd::{F16SimdVec, simd_function};

use crate::render::{ErasedLocalState, RenderPipelineInPlaceStage};
use crate::util::f16;

/// Render spot color
pub struct SpotColorStage {
    /// Spot color channel index
    spot_c: usize,
    /// Spot color in linear RGBA
    spot_color: [f32; 4],
}

impl std::fmt::Display for SpotColorStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "spot color stage for channel {}", self.spot_c)
    }
}

impl SpotColorStage {
    #[allow(unused, reason = "remove once we actually use this")]
    pub fn new(spot_c_offset: usize, spot_color: [f32; 4]) -> Self {
        Self {
            spot_c: 3 + spot_c_offset,
            spot_color,
        }
    }
}

simd_function!(
    spot_color_simd_dispatch,
    d: D,
    fn spot_color_simd(
        row_r: &mut [f16],
        row_g: &mut [f16],
        row_b: &mut [f16],
        row_s: &[f16],
        spot_color: [f32; 4],
        xsize: usize,
    ) {
        let scale = D::F16Vec::splat_f32(d, spot_color[3]);
        let one = D::F16Vec::splat_f32(d, 1.0);
        let sc_r = D::F16Vec::splat_f32(d, spot_color[0]);
        let sc_g = D::F16Vec::splat_f32(d, spot_color[1]);
        let sc_b = D::F16Vec::splat_f32(d, spot_color[2]);

        let iter_r = row_r.chunks_exact_mut(D::F16Vec::LEN);
        let iter_g = row_g.chunks_exact_mut(D::F16Vec::LEN);
        let iter_b = row_b.chunks_exact_mut(D::F16Vec::LEN);
        let iter_s = row_s.chunks_exact(D::F16Vec::LEN);
        for (((r_chunk, g_chunk), b_chunk), s_chunk) in iter_r
            .zip(iter_g)
            .zip(iter_b)
            .zip(iter_s)
            .take(xsize.div_ceil(D::F16Vec::LEN))
        {
            let s = D::F16Vec::load(d, s_chunk);
            let mix = scale * s;
            let one_minus_mix = one - mix;

            let r = D::F16Vec::load(d, r_chunk);
            let g = D::F16Vec::load(d, g_chunk);
            let b = D::F16Vec::load(d, b_chunk);

            mix.mul_add(sc_r, one_minus_mix * r).store(r_chunk);
            mix.mul_add(sc_g, one_minus_mix * g).store(g_chunk);
            mix.mul_add(sc_b, one_minus_mix * b).store(b_chunk);
        }
    }
);

impl RenderPipelineInPlaceStage for SpotColorStage {
    type Type = f16;

    fn uses_channel(&self, c: usize) -> bool {
        c < 3 || c == self.spot_c
    }

    // `row` should only contain color channels and the spot channel.
    fn process_row_chunk(
        &self,
        _position: (usize, usize),
        xsize: usize,
        row: &mut [&mut [f16]],
        _state: Option<&mut ErasedLocalState>,
        _previous_call_was_previous_row: bool,
    ) {
        let [row_r, row_g, row_b, row_s] = row else {
            panic!(
                "incorrect number of channels; expected 4, found {}",
                row.len()
            );
        };

        assert!(
            xsize <= row_r.len()
                && xsize <= row_g.len()
                && xsize <= row_b.len()
                && xsize <= row_s.len()
        );
        spot_color_simd_dispatch(row_r, row_g, row_b, row_s, self.spot_color, xsize);
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
    use crate::util::f16;

    #[test]
    fn consistency() -> Result<()> {
        crate::render::test::test_stage_consistency(
            || SpotColorStage::new(0, [0.0; 4]),
            (500, 500),
            4,
        )
    }

    #[test]
    fn srgb_primaries() -> Result<()> {
        let mut input_r = Image::new((3, 1))?;
        let mut input_g = Image::new((3, 1))?;
        let mut input_b = Image::new((3, 1))?;
        let mut input_s = Image::new((3, 1))?;
        input_r
            .row_mut(0)
            .copy_from_slice(&[1.0, 0.0, 0.0].map(f16::from_f32));
        input_g
            .row_mut(0)
            .copy_from_slice(&[0.0, 1.0, 0.0].map(f16::from_f32));
        input_b
            .row_mut(0)
            .copy_from_slice(&[0.0, 0.0, 1.0].map(f16::from_f32));
        input_s
            .row_mut(0)
            .copy_from_slice(&[1.0, 1.0, 1.0].map(f16::from_f32));

        let stage = SpotColorStage::new(0, [0.5; 4]);
        let output = make_and_run_simple_pipeline(
            stage,
            &[input_r, input_g, input_b, input_s],
            (3, 1),
            0,
            256,
        )?;

        assert_close!(
            all,
            &output[0]
                .row(0)
                .iter()
                .map(|x| x.to_f32())
                .collect::<Vec<_>>(),
            &[0.75, 0.25, 0.25],
            1e-6
        );
        assert_close!(
            all,
            &output[1]
                .row(0)
                .iter()
                .map(|x| x.to_f32())
                .collect::<Vec<_>>(),
            &[0.25, 0.75, 0.25],
            1e-6
        );
        assert_close!(
            all,
            &output[2]
                .row(0)
                .iter()
                .map(|x| x.to_f32())
                .collect::<Vec<_>>(),
            &[0.25, 0.25, 0.75],
            1e-6
        );

        Ok(())
    }
}

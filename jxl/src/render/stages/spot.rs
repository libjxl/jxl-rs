// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl_simd::{ScalarDescriptor, SimdDescriptor};

use super::row_chunks::for_each_chunk;
use crate::render::{ErasedLocalState, RenderPipelineInPlaceStage};

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

impl RenderPipelineInPlaceStage for SpotColorStage {
    type Type = f32;

    fn uses_channel(&self, c: usize) -> bool {
        c < 3 || c == self.spot_c
    }

    // `row` should only contain color channels and the spot channel.
    fn process_row_chunk(
        &self,
        _position: (usize, usize),
        xsize: usize,
        row: &mut [&mut [f32]],
        _state: Option<&mut ErasedLocalState>,
        _previous_call_was_previous_row: bool,
    ) {
        let [row_r, row_g, row_b, row_s] = row else {
            panic!(
                "incorrect number of channels; expected 4, found {}",
                row.len()
            );
        };

        let scale = self.spot_color[3];
        let [spot_r, spot_g, spot_b, _] = self.spot_color;
        let d = ScalarDescriptor::new().unwrap();
        for_each_chunk(
            d,
            xsize,
            (&mut **row_r, &mut **row_g, &mut **row_b, &**row_s),
            #[inline(always)]
            |_x, (mut r, mut g, mut b, s)| {
                let mix = scale * s;
                let inv_mix = 1.0 - mix;
                let new_r = mix * spot_r + inv_mix * r.read();
                let new_g = mix * spot_g + inv_mix * g.read();
                let new_b = mix * spot_b + inv_mix * b.read();
                r.write(new_r);
                g.write(new_g);
                b.write(new_b);
            },
        );
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
        input_r.row_mut(0).copy_from_slice(&[1.0, 0.0, 0.0]);
        input_g.row_mut(0).copy_from_slice(&[0.0, 1.0, 0.0]);
        input_b.row_mut(0).copy_from_slice(&[0.0, 0.0, 1.0]);
        input_s.row_mut(0).copy_from_slice(&[1.0, 1.0, 1.0]);

        let stage = SpotColorStage::new(0, [0.5; 4]);
        let output = make_and_run_simple_pipeline(
            stage,
            &[input_r, input_g, input_b, input_s],
            (3, 1),
            0,
            256,
        )?;

        assert_close!(all, output[0].row(0), &[0.75, 0.25, 0.25], 1e-6);
        assert_close!(all, output[1].row(0), &[0.25, 0.75, 0.25], 1e-6);
        assert_close!(all, output[2].row(0), &[0.25, 0.25, 0.75], 1e-6);

        Ok(())
    }
}

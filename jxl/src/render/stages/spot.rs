// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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
        let (row_r, row_g, row_b, row_s) = unsafe {
            // SAFETY: row contains at least 4 channels.
            let ptr = row.as_mut_ptr();
            (
                &mut **ptr,
                &mut **ptr.add(1),
                &mut **ptr.add(2),
                &mut **ptr.add(3),
            )
        };

        let scale = self.spot_color[3];
        for idx in 0..xsize {
            unsafe {
                // SAFETY: idx < xsize <= row buffer length.
                let mix = scale * *row_s.get_unchecked(idx);
                let r = *row_r.get_unchecked(idx);
                let g = *row_g.get_unchecked(idx);
                let b = *row_b.get_unchecked(idx);
                *row_r.get_unchecked_mut(idx) = mix * self.spot_color[0] + (1.0 - mix) * r;
                *row_g.get_unchecked_mut(idx) = mix * self.spot_color[1] + (1.0 - mix) * g;
                *row_b.get_unchecked_mut(idx) = mix * self.spot_color[2] + (1.0 - mix) * b;
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

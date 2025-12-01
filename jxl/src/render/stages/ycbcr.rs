// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use crate::render::RenderPipelineInPlaceStage;
use jxl_simd::{F32SimdVec, simd_function};

/// Convert YCbCr to RGB
pub struct YcbcrToRgbStage {
    first_channel: usize,
}

impl YcbcrToRgbStage {
    pub fn new(first_channel: usize) -> Self {
        Self { first_channel }
    }
}

impl std::fmt::Display for YcbcrToRgbStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let channel = self.first_channel;
        write!(
            f,
            "YCbCr to RGB for channel [{},{},{}]",
            channel,
            channel + 1,
            channel + 2
        )
    }
}

// SIMD YCbCr to RGB conversion matching C++ Highway implementation
simd_function!(
    ycbcr_to_rgb_simd_dispatch,
    d: D,
    fn ycbcr_to_rgb_simd(
        row_cb: &mut [f32],
        row_y: &mut [f32],
        row_cr: &mut [f32],
        xsize: usize,
    ) {
        let simd_width = D::F32Vec::LEN;

        // Precompute constants as SIMD vectors (matches C++ Highway)
        let c128 = D::F32Vec::splat(d, 128.0 / 255.0);
        let cr_to_r = D::F32Vec::splat(d, 1.402);
        let cr_to_g = D::F32Vec::splat(d, -0.299 * 1.402 / 0.587);
        let cb_to_g = D::F32Vec::splat(d, -0.114 * 1.772 / 0.587);
        let cb_to_b = D::F32Vec::splat(d, 1.772);

        // SIMD loop processing SIMD_WIDTH pixels at once
        let mut x = 0;
        while x + simd_width <= xsize {
            // Load Y, Cb, Cr vectors
            let y_vec = D::F32Vec::load(d, &row_y[x..]) + c128;
            let cb_vec = D::F32Vec::load(d, &row_cb[x..]);
            let cr_vec = D::F32Vec::load(d, &row_cr[x..]);

            // Compute RGB using FMA (fused multiply-add) for better performance
            // R = Y + 1.402 * Cr
            let r_vec = cr_vec.mul_add(cr_to_r, y_vec);

            // G = Y - 0.299*1.402/0.587 * Cr - 0.114*1.772/0.587 * Cb
            let g_vec = cr_vec.mul_add(cr_to_g, cb_vec.mul_add(cb_to_g, y_vec));

            // B = Y + 1.772 * Cb
            let b_vec = cb_vec.mul_add(cb_to_b, y_vec);

            // Store back to channels (R→Cb, G→Y, B→Cr to match layout)
            r_vec.store(&mut row_cb[x..]);
            g_vec.store(&mut row_y[x..]);
            b_vec.store(&mut row_cr[x..]);

            x += simd_width;
        }

        // Scalar remainder
        for idx in x..xsize {
            let y = row_y[idx] + 128.0 / 255.0;
            let cb = row_cb[idx];
            let cr = row_cr[idx];

            row_cb[idx] = cr.mul_add(1.402, y);
            row_y[idx] = cr.mul_add(
                -0.299 * 1.402 / 0.587,
                cb.mul_add(-0.114 * 1.772 / 0.587, y),
            );
            row_cr[idx] = cb.mul_add(1.772, y);
        }
    }
);

impl RenderPipelineInPlaceStage for YcbcrToRgbStage {
    type Type = f32;

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
        // pixels are stored in `Cb Y Cr` order to mimic XYB colorspace
        let [row_cb, row_y, row_cr] = row else {
            panic!(
                "incorrect number of channels; expected 3, found {}",
                row.len()
            );
        };

        assert!(xsize <= row_cb.len() && xsize <= row_y.len() && xsize <= row_cr.len());

        // Use SIMD for YCbCr to RGB conversion (matches C++ Highway implementation)
        // Full-range BT.601 as defined by JFIF Clause 7:
        // https://www.itu.int/rec/T-REC-T.871-201105-I/en
        ycbcr_to_rgb_simd_dispatch(row_cb, row_y, row_cr, xsize);
    }
}

#[cfg(test)]
mod test {
    use test_log::test;

    use super::*;
    use crate::error::Result;
    use crate::image::Image;
    use crate::render::test::make_and_run_simple_pipeline;
    use crate::util::test::assert_all_almost_abs_eq;

    #[test]
    fn consistency() -> Result<()> {
        crate::render::test::test_stage_consistency(|| YcbcrToRgbStage::new(0), (500, 500), 3)
    }

    #[test]
    fn srgb_primaries() -> Result<()> {
        let mut input_y = Image::new((3, 1))?;
        let mut input_cb = Image::new((3, 1))?;
        let mut input_cr = Image::new((3, 1))?;
        input_y
            .row_mut(0)
            .copy_from_slice(&[-0.20296079, 0.08503921, -0.3879608]);
        input_cb
            .row_mut(0)
            .copy_from_slice(&[-0.16873589, -0.3312641, 0.5]);
        input_cr
            .row_mut(0)
            .copy_from_slice(&[0.5, -0.41868758, -0.08131241]);

        let stage = YcbcrToRgbStage::new(0);
        let output =
            make_and_run_simple_pipeline(stage, &[input_cb, input_y, input_cr], (3, 1), 0, 256)?;

        assert_all_almost_abs_eq(output[0].row(0), &[1.0, 0.0, 0.0], 1e-6);
        assert_all_almost_abs_eq(output[1].row(0), &[0.0, 1.0, 0.0], 1e-6);
        assert_all_almost_abs_eq(output[2].row(0), &[0.0, 0.0, 1.0], 1e-6);

        Ok(())
    }
}

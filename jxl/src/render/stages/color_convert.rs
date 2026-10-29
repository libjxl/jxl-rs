// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl_simd::{F32SimdVec, simd_function};

use crate::api::{
    JxlColorEncoding, JxlWhitePoint, RenderingIntent, adapt_to_xyz_d50, primaries_to_xyz,
};
use crate::color::tf::TransferFunction;
use crate::error::{Error, Result};
use crate::render::{ErasedLocalState, RenderPipelineInPlaceStage};
use crate::util::{Matrix3x3, inv_3x3_matrix, mul_3x3_matrix};

struct LinearEncoding {
    to_xyz: Matrix3x3<f64>,
    white: JxlWhitePoint,
    tf: TransferFunction,
    intent: RenderingIntent,
    grayscale: bool,
}

impl LinearEncoding {
    fn new(encoding: &JxlColorEncoding, intensity_target: f32) -> Result<Self> {
        let (white, tf, intent, to_xyz, grayscale) = match encoding {
            JxlColorEncoding::RgbColorSpace {
                white_point,
                primaries,
                transfer_function,
                rendering_intent,
            } => {
                let [r, g, b] = primaries.to_xy_coords();
                let (wx, wy) = white_point.to_xy_coords();
                let matrix = primaries_to_xyz(r.0, r.1, g.0, g.1, b.0, b.1, wx, wy)?;
                (
                    white_point,
                    transfer_function,
                    rendering_intent,
                    matrix,
                    false,
                )
            }
            JxlColorEncoding::GrayscaleColorSpace {
                white_point,
                transfer_function,
                rendering_intent,
            } => {
                let (wx, wy) = white_point.to_xy_coords();
                adapt_to_xyz_d50(wx, wy)?;
                // The three internal color slots carry the same luminance for grayscale.
                let matrix = [
                    [wx as f64 / wy as f64, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, (1.0 - wx as f64 - wy as f64) / wy as f64],
                ];
                (
                    white_point,
                    transfer_function,
                    rendering_intent,
                    matrix,
                    true,
                )
            }
            JxlColorEncoding::XYB { .. } => return Err(Error::InvalidColorEncoding),
        };
        let luminances = if grayscale {
            [1.0, 0.0, 0.0]
        } else {
            to_xyz[1].map(|x| x as f32)
        };
        Ok(Self {
            to_xyz,
            white: white.clone(),
            tf: TransferFunction::from_api_tf(tf, intensity_target, luminances),
            intent: *intent,
            grayscale,
        })
    }
}

/// Changes color encoding only; tone and gamut mapping remain in the synthesized ICC profiles.
pub struct ColorConvertStage {
    from_tf: TransferFunction,
    to_tf: TransferFunction,
    matrix: Option<Matrix3x3<f32>>,
    from_grayscale: bool,
}

impl ColorConvertStage {
    pub fn new(
        from: &JxlColorEncoding,
        to: &JxlColorEncoding,
        intensity_target: f32,
    ) -> Result<Self> {
        let from = LinearEncoding::new(from, intensity_target)?;
        let to = LinearEncoding::new(to, intensity_target)?;
        let matrix = if from.to_xyz == to.to_xyz && from.white == to.white {
            None
        } else {
            let mut matrix = from.to_xyz;
            if from.white != to.white && from.intent != RenderingIntent::Absolute {
                let (wx, wy) = from.white.to_xy_coords();
                let src_adapt = adapt_to_xyz_d50(wx, wy)?;
                let (wx, wy) = to.white.to_xy_coords();
                let dst_adapt = adapt_to_xyz_d50(wx, wy)?;
                matrix = mul_3x3_matrix(
                    &mul_3x3_matrix(&inv_3x3_matrix(&dst_adapt)?, &src_adapt),
                    &matrix,
                );
            }
            matrix = if to.grayscale {
                [matrix[1]; 3]
            } else {
                mul_3x3_matrix(&inv_3x3_matrix(&to.to_xyz)?, &matrix)
            };
            Some(matrix.map(|row| row.map(|x| x as f32)))
        };
        Ok(Self {
            from_tf: from.tf,
            to_tf: to.tf,
            matrix,
            from_grayscale: from.grayscale,
        })
    }
}

impl std::fmt::Display for ColorConvertStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Signalled to output color encoding")
    }
}

simd_function!(
    color_convert_dispatch,
    d: D,
    fn color_convert(stage: &ColorConvertStage, xsize: usize, rows: &mut [&mut [f32]]) {
        let [r, g, b] = rows else { unreachable!() };
        let matrix = stage.matrix.map(|m| m.map(|row| row.map(|x| D::F32Vec::splat(d, x))));
        for ((r, g), b) in r.chunks_exact_mut(D::F32Vec::LEN)
            .zip(g.chunks_exact_mut(D::F32Vec::LEN))
            .zip(b.chunks_exact_mut(D::F32Vec::LEN))
            .take(xsize.div_ceil(D::F32Vec::LEN))
        {
            let rgb = if stage.from_grayscale {
                [D::F32Vec::load(d, r); 3]
            } else {
                [D::F32Vec::load(d, r), D::F32Vec::load(d, g), D::F32Vec::load(d, b)]
            };
            let mut rgb = stage.from_tf.apply_to_linear(d, rgb);
            if let Some(matrix) = matrix {
                rgb = matrix.map(|row| row[0].mul_add(rgb[0], row[1].mul_add(rgb[1], row[2] * rgb[2])));
            }
            let [rv, gv, bv] = stage.to_tf.apply_from_linear(d, rgb);
            rv.store(r);
            gv.store(g);
            bv.store(b);
        }
    }
);

impl RenderPipelineInPlaceStage for ColorConvertStage {
    type Type = f32;

    fn uses_channel(&self, c: usize) -> bool {
        c < 3
    }

    fn process_row_chunk(
        &self,
        _position: (usize, usize),
        xsize: usize,
        row: &mut [&mut [f32]],
        _state: Option<&mut ErasedLocalState>,
        _previous_call_was_previous_row: bool,
    ) {
        color_convert_dispatch(self, xsize, row);
    }
}

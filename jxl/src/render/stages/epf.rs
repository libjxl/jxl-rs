// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::sync::Arc;

use crate::{
    BLOCK_DIM, MIN_SIGMA, SIGMA_PADDING,
    image::Image,
    render::{RenderPipelineInOutStage, RenderPipelineStage},
};

/// 5x5 plus-shaped kernel with 5 SADs per pixel (3x3 plus-shaped). So this makes this filter a 7x7 filter.
pub struct Epf0Stage {
    /// Multiplier for sigma in pass 0
    sigma_scale: f32,
    /// (inverse) multiplier for sigma on borders
    border_sad_mul: f32,
    channel_scale: [f32; 3],
    sigma: Arc<Image<f32>>,
}

impl std::fmt::Display for Epf0Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "EPF stage 0 with sigma scale: {}, border_sad_mul: {}",
            self.sigma_scale, self.border_sad_mul
        )
    }
}

impl Epf0Stage {
    #[allow(unused, reason = "remove once we actually use this")]
    pub fn new(
        sigma_scale: f32,
        border_sad_mul: f32,
        channel_scale: [f32; 3],
        sigma: Arc<Image<f32>>,
    ) -> Self {
        Self {
            sigma,
            sigma_scale,
            channel_scale,
            border_sad_mul,
        }
    }
}

impl RenderPipelineStage for Epf0Stage {
    type Type = RenderPipelineInOutStage<f32, f32, 3, 3, 0, 0>;

    fn uses_channel(&self, c: usize) -> bool {
        c < 3
    }

    fn process_row_chunk(
        &self,
        (xpos, ypos): (usize, usize),
        xsize: usize,
        row: &mut [(&[&[f32]], &mut [&mut [f32]])],
        _state: Option<&mut dyn std::any::Any>,
    ) {
        assert!(row.len() == 3, "Expected 3 channels, got {}", row.len());

        let row_sigma = self.sigma.as_rect().row(ypos / BLOCK_DIM + SIGMA_PADDING);

        let sm = self.sigma_scale * 1.65;
        let bsm = sm * self.border_sad_mul;

        let sad_mul = if ypos % BLOCK_DIM == 0 || ypos % BLOCK_DIM == BLOCK_DIM - 1 {
            [bsm; 8] // border
        } else {
            [bsm, sm, sm, sm, sm, sm, sm, bsm] // center
        };

        for x in 0..xsize {
            let bx = (x + xpos + SIGMA_PADDING * BLOCK_DIM) / BLOCK_DIM;
            let ix = (x + xpos) % BLOCK_DIM;

            if row_sigma[bx] < MIN_SIGMA {
                for (input_c, output_c) in row.iter_mut() {
                    output_c[0][x] = input_c[3][3 + x];
                }
                continue;
            }

            // Compute SADs
            let mut sads = [0.0; 12];
            for ((input_c, _), scale) in row.iter_mut().zip(self.channel_scale) {
                let p30 = input_c[0][3 + x];
                let p21 = input_c[1][2 + x];
                let p31 = input_c[1][3 + x];
                let p41 = input_c[1][4 + x];
                let p12 = input_c[2][1 + x];
                let p22 = input_c[2][2 + x];
                let p32 = input_c[2][3 + x];
                let p42 = input_c[2][4 + x];
                let p52 = input_c[2][5 + x];
                let p03 = input_c[3][x];
                let p13 = input_c[3][1 + x];
                let p23 = input_c[3][2 + x];
                let p33 = input_c[3][3 + x];
                let p43 = input_c[3][4 + x];
                let p53 = input_c[3][5 + x];
                let p63 = input_c[3][6 + x];
                let p14 = input_c[4][1 + x];
                let p24 = input_c[4][2 + x];
                let p34 = input_c[4][3 + x];
                let p44 = input_c[4][4 + x];
                let p54 = input_c[4][5 + x];
                let p25 = input_c[5][2 + x];
                let p35 = input_c[5][3 + x];
                let p45 = input_c[5][4 + x];
                let p36 = input_c[6][3 + x];
                let d32_30 = (p32 - p30).abs();
                let d32_21 = (p32 - p21).abs();
                let d32_31 = (p32 - p31).abs();
                let d32_41 = (p32 - p41).abs();
                let d32_12 = (p32 - p12).abs();
                let d32_22 = (p32 - p22).abs();
                let d32_42 = (p32 - p42).abs();
                let d32_52 = (p32 - p52).abs();
                let d32_23 = (p32 - p23).abs();
                let d32_34 = (p32 - p34).abs();
                let d32_43 = (p32 - p43).abs();
                let d32_33 = (p32 - p33).abs();
                let d23_21 = (p23 - p21).abs();
                let d23_12 = (p23 - p12).abs();
                let d23_22 = (p23 - p22).abs();
                let d23_03 = (p23 - p03).abs();
                let d23_13 = (p23 - p13).abs();
                let d23_33 = (p23 - p33).abs();
                let d23_43 = (p23 - p43).abs();
                let d23_14 = (p23 - p14).abs();
                let d23_24 = (p23 - p24).abs();
                let d23_34 = (p23 - p34).abs();
                let d23_25 = (p23 - p25).abs();
                let d33_31 = (p33 - p31).abs();
                let d33_22 = (p33 - p22).abs();
                let d33_42 = (p33 - p42).abs();
                let d33_13 = (p33 - p13).abs();
                let d33_43 = (p33 - p43).abs();
                let d33_53 = (p33 - p53).abs();
                let d33_24 = (p33 - p24).abs();
                let d33_34 = (p33 - p34).abs();
                let d33_44 = (p33 - p44).abs();
                let d33_35 = (p33 - p35).abs();
                let d43_41 = (p43 - p41).abs();
                let d43_42 = (p43 - p42).abs();
                let d43_52 = (p43 - p52).abs();
                let d43_53 = (p43 - p53).abs();
                let d43_63 = (p43 - p63).abs();
                let d43_34 = (p43 - p34).abs();
                let d43_44 = (p43 - p44).abs();
                let d43_54 = (p43 - p54).abs();
                let d43_45 = (p43 - p45).abs();
                let d34_14 = (p34 - p14).abs();
                let d34_24 = (p34 - p24).abs();
                let d34_44 = (p34 - p44).abs();
                let d34_54 = (p34 - p54).abs();
                let d34_25 = (p34 - p25).abs();
                let d34_35 = (p34 - p35).abs();
                let d34_45 = (p34 - p45).abs();
                let d34_36 = (p34 - p36).abs();
                sads[0] += (d32_30 + d23_21 + d33_31 + d43_41 + d32_34) * scale;
                sads[1] += (d32_21 + d23_12 + d33_22 + d32_43 + d23_34) * scale;
                sads[2] += (d32_31 + d23_22 + d32_33 + d43_42 + d33_34) * scale;
                sads[3] += (d32_41 + d32_23 + d33_42 + d43_52 + d43_34) * scale;
                sads[4] += (d32_12 + d23_03 + d33_13 + d23_43 + d34_14) * scale;
                sads[5] += (d32_22 + d23_13 + d23_33 + d33_43 + d34_24) * scale;
                sads[6] += (d32_42 + d23_33 + d33_43 + d43_53 + d34_44) * scale;
                sads[7] += (d32_52 + d23_43 + d33_53 + d43_63 + d34_54) * scale;
                sads[8] += (d32_23 + d23_14 + d33_24 + d43_34 + d34_25) * scale;
                sads[9] += (d32_33 + d23_24 + d33_34 + d43_44 + d34_35) * scale;
                sads[10] += (d32_43 + d23_34 + d33_44 + d43_54 + d34_45) * scale;
                sads[11] += (d32_34 + d23_25 + d33_35 + d43_45 + d34_36) * scale;
            }
            // Compute output based on SADs
            let inv_sigma = row_sigma[bx] * sad_mul[ix];
            let mut w = 1.0;
            for sad in sads.iter_mut() {
                *sad = (*sad * inv_sigma + 1.0).max(0.0);
                w += *sad;
            }
            let inv_w = 1.0 / w;
            for (input_c, output_c) in row.iter_mut() {
                output_c[0][x] = (input_c[3][3 + x]
                    + input_c[1][3 + x] * sads[0]
                    + input_c[2][2 + x] * sads[1]
                    + input_c[2][3 + x] * sads[2]
                    + input_c[2][4 + x] * sads[3]
                    + input_c[3][1 + x] * sads[4]
                    + input_c[3][2 + x] * sads[5]
                    + input_c[3][4 + x] * sads[6]
                    + input_c[3][5 + x] * sads[7]
                    + input_c[4][2 + x] * sads[8]
                    + input_c[4][3 + x] * sads[9]
                    + input_c[4][4 + x] * sads[10]
                    + input_c[5][3 + x] * sads[11])
                    * inv_w;
            }
        }
    }
}

/// 3x3 plus-shaped kernel with 5 SADs per pixel (3x3 plus-shaped). So this makes this filter a 5x5 filter.
pub struct Epf1Stage {
    /// Multiplier for sigma in pass 1
    sigma_scale: f32,
    /// (inverse) multiplier for sigma on borders
    border_sad_mul: f32,
    channel_scale: [f32; 3],
    sigma: Arc<Image<f32>>,
}

impl std::fmt::Display for Epf1Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "EPF stage 1 with sigma scale: {}, border_sad_mul: {}",
            self.sigma_scale, self.border_sad_mul
        )
    }
}

impl Epf1Stage {
    #[allow(unused, reason = "remove once we actually use this")]
    pub fn new(
        sigma_scale: f32,
        border_sad_mul: f32,
        channel_scale: [f32; 3],
        sigma: Arc<Image<f32>>,
    ) -> Self {
        Self {
            sigma,
            sigma_scale,
            channel_scale,
            border_sad_mul,
        }
    }
}

impl RenderPipelineStage for Epf1Stage {
    type Type = RenderPipelineInOutStage<f32, f32, 2, 2, 0, 0>;

    fn uses_channel(&self, c: usize) -> bool {
        c < 3
    }

    fn process_row_chunk(
        &self,
        (xpos, ypos): (usize, usize),
        xsize: usize,
        row: &mut [(&[&[f32]], &mut [&mut [f32]])],
        _state: Option<&mut dyn std::any::Any>,
    ) {
        assert!(row.len() == 3, "Expected 3 channels, got {}", row.len());

        let row_sigma = self.sigma.as_rect().row(ypos / BLOCK_DIM + SIGMA_PADDING);

        let sm = self.sigma_scale * 1.65;
        let bsm = sm * self.border_sad_mul;

        let sad_mul = if ypos % BLOCK_DIM == 0 || ypos % BLOCK_DIM == BLOCK_DIM - 1 {
            [bsm; 8] // border
        } else {
            [bsm, sm, sm, sm, sm, sm, sm, bsm] // center
        };

        for x in 0..xsize {
            let bx = (x + xpos + SIGMA_PADDING * BLOCK_DIM) / BLOCK_DIM;
            let ix = (x + xpos) % BLOCK_DIM;

            if row_sigma[bx] < MIN_SIGMA {
                for (input_c, output_c) in row.iter_mut() {
                    output_c[0][x] = input_c[2][2 + x];
                }
                continue;
            }

            // Compute SADs
            let mut sads = [0.0; 4];
            for ((input_c, _), scale) in row.iter_mut().zip(self.channel_scale) {
                let p20 = input_c[0][2 + x];
                let p11 = input_c[1][1 + x];
                let p21 = input_c[1][2 + x];
                let p31 = input_c[1][3 + x];
                let p02 = input_c[2][x];
                let p12 = input_c[2][1 + x];
                let p22 = input_c[2][2 + x];
                let p32 = input_c[2][3 + x];
                let p42 = input_c[2][4 + x];
                let p13 = input_c[3][1 + x];
                let p23 = input_c[3][2 + x];
                let p33 = input_c[3][3 + x];
                let p24 = input_c[4][2 + x];
                let d20_21 = (p20 - p21).abs();
                let d11_21 = (p11 - p21).abs();
                let d22_21 = (p22 - p21).abs();
                let d31_21 = (p31 - p21).abs();
                let d02_12 = (p02 - p12).abs();
                let d11_12 = (p11 - p12).abs();
                let d12_22 = (p22 - p12).abs();
                let d31_32 = (p31 - p32).abs();
                let d22_32 = (p22 - p32).abs();
                let d42_32 = (p42 - p32).abs();
                let d13_12 = (p13 - p12).abs();
                let d22_23 = (p22 - p23).abs();
                let d13_23 = (p13 - p23).abs();
                let d33_23 = (p33 - p23).abs();
                let d33_32 = (p33 - p32).abs();
                let d24_23 = (p24 - p23).abs();
                sads[0] += (d20_21 + d11_12 + d22_21 + d31_32 + d22_23) * scale;
                sads[1] += (d11_21 + d02_12 + d12_22 + d22_32 + d13_23) * scale;
                sads[2] += (d31_21 + d12_22 + d22_32 + d42_32 + d33_23) * scale;
                sads[3] += (d22_21 + d13_12 + d22_23 + d33_32 + d24_23) * scale;
            }

            // Compute output based on SADs
            let inv_sigma = row_sigma[bx] * sad_mul[ix];
            let mut w = 1.0;
            for sad in sads.iter_mut() {
                *sad = (*sad * inv_sigma + 1.0).max(0.0);
                w += *sad;
            }
            let inv_w = 1.0 / w;
            for (input_c, output_c) in row.iter_mut() {
                output_c[0][x] = (input_c[2][2 + x]
                    + input_c[1][2 + x] * sads[0]
                    + input_c[2][1 + x] * sads[1]
                    + input_c[2][3 + x] * sads[2]
                    + input_c[3][2 + x] * sads[3])
                    * inv_w;
            }
        }
    }
}

/// 3x3 plus-shaped kernel with 1 SAD per pixel. So this makes this filter a 3x3 filter.
pub struct Epf2Stage {
    /// Multiplier for sigma in pass 2
    sigma_scale: f32,
    /// (inverse) multiplier for sigma on borders
    border_sad_mul: f32,
    channel_scale: [f32; 3],
    sigma: Arc<Image<f32>>,
}

impl std::fmt::Display for Epf2Stage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "EPF stage 2 with sigma scale: {}, border_sad_mul: {}",
            self.sigma_scale, self.border_sad_mul
        )
    }
}

impl Epf2Stage {
    #[allow(unused, reason = "remove once we actually use this")]
    pub fn new(
        sigma_scale: f32,
        border_sad_mul: f32,
        channel_scale: [f32; 3],
        sigma: Arc<Image<f32>>,
    ) -> Self {
        Self {
            sigma,
            sigma_scale,
            channel_scale,
            border_sad_mul,
        }
    }
    fn acc_avg_weighted_px(
        &self,
        c: (f32, f32, f32),
        r: (f32, f32, f32),
        inv_sigma: f32,
        acc: (&mut f32, &mut f32, &mut f32, &mut f32),
    ) {
        let (cx, cy, cb) = c;
        let (rx, ry, rb) = r;
        let (x_acc, y_acc, b_acc, w_acc) = acc;
        let sad = (cx - rx).abs() * self.channel_scale[0]
            + (cy - ry).abs() * self.channel_scale[1]
            + (cb - rb).abs() * self.channel_scale[2];
        let weight = (sad * inv_sigma + 1.0).max(0.0);
        *w_acc += weight;
        *x_acc += weight * cx;
        *y_acc += weight * cy;
        *b_acc += weight * cb;
    }
}

impl RenderPipelineStage for Epf2Stage {
    type Type = RenderPipelineInOutStage<f32, f32, 1, 1, 0, 0>;

    fn uses_channel(&self, c: usize) -> bool {
        c < 3
    }

    fn process_row_chunk(
        &self,
        (xpos, ypos): (usize, usize),
        xsize: usize,
        row: &mut [(&[&[f32]], &mut [&mut [f32]])],
        _state: Option<&mut dyn std::any::Any>,
    ) {
        assert!(row.len() == 3, "Expected 3 channels, got {}", row.len());

        let row_sigma = self.sigma.as_rect().row(ypos / BLOCK_DIM + SIGMA_PADDING);

        let sm = self.sigma_scale * 1.65;
        let bsm = sm * self.border_sad_mul;

        let sad_mul = if ypos % BLOCK_DIM == 0 || ypos % BLOCK_DIM == BLOCK_DIM - 1 {
            [bsm; 8] // border
        } else {
            [bsm, sm, sm, sm, sm, sm, sm, bsm] // center
        };

        for x in 0..xsize {
            let bx = (x + xpos + SIGMA_PADDING * BLOCK_DIM) / BLOCK_DIM;
            let ix = (x + xpos) % BLOCK_DIM;

            if row_sigma[bx] < MIN_SIGMA {
                for (input_c, output_c) in row.iter_mut() {
                    output_c[0][x] = input_c[1][1 + x];
                }
                continue;
            }

            let vsm = sad_mul[ix];
            let inv_sigma = row_sigma[bx] * vsm;

            let x_cc = row[0].0[1][1 + x];
            let y_cc = row[1].0[1][1 + x];
            let b_cc = row[2].0[1][1 + x];

            let mut w_acc = 1.0;
            let mut x_acc = x_cc;
            let mut y_acc = y_cc;
            let mut b_acc = b_cc;

            [(0, 1), (1, 0), (1, 2), (2, 1)]
                .iter()
                .for_each(|(y_off, x_off)| {
                    self.acc_avg_weighted_px(
                        (
                            row[0].0[*y_off as usize][x_off + x],
                            row[1].0[*y_off as usize][x_off + x],
                            row[2].0[*y_off as usize][x_off + x],
                        ),
                        (x_cc, y_cc, b_cc),
                        inv_sigma,
                        (&mut x_acc, &mut y_acc, &mut b_acc, &mut w_acc),
                    );
                });

            let inv_w = 1.0 / w_acc;

            row[0].1[0][x] = x_acc * inv_w;
            row[1].1[0][x] = y_acc * inv_w;
            row[2].1[0][x] = b_acc * inv_w;
        }
    }
}

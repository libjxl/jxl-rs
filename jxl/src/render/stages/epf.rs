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
        &mut self,
        (xpos, ypos): (usize, usize),
        xsize: usize,
        row: &mut [(&[&[f32]], &mut [&mut [f32]])],
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
            const SADS_OFF: [[isize; 2]; 12] = [
                [-2, 0],
                [-1, -1],
                [-1, 0],
                [-1, 1],
                [0, -2],
                [0, -1],
                [0, 1],
                [0, 2],
                [1, -1],
                [1, 0],
                [1, 1],
                [2, 0],
            ];
            for ((input_c, _), scale) in row.iter().zip(self.channel_scale) {
                const PLUS_OFF: [[isize; 2]; 5] = [[0, 0], [-1, 0], [0, -1], [1, 0], [0, 1]];
                for (sads_i, sad_off) in sads.iter_mut().zip(SADS_OFF) {
                    let sad = PLUS_OFF.iter().fold(0.0, |acc, off| {
                        let r11 =
                            input_c[(3 + off[0]) as usize][3 + x.saturating_add_signed(off[1])];
                        let c11 = input_c[(3 + sad_off[0] + off[0]) as usize]
                            [3 + x.saturating_add_signed(sad_off[1] + off[1])];
                        acc + (r11 - c11).abs()
                    });
                    *sads_i = sad.mul_add(scale, *sads_i);
                }
            }

            // Compute output based on SADs
            let vsm = sad_mul[ix];
            let inv_sigma = row_sigma[bx] * vsm;
            for (input_c, output_c) in row.iter_mut() {
                let mut cc = input_c[3][3 + x];
                let mut weight = 1.0;
                for (sad, sad_off) in sads.iter().zip(SADS_OFF) {
                    let c =
                        input_c[(3 + sad_off[0]) as usize][3 + x.saturating_add_signed(sad_off[1])];
                    let w = sad.mul_add(inv_sigma, 1.0).max(0.0);

                    weight += w;
                    cc = c.mul_add(w, cc);
                }

                let inv_w = 1.0 / weight;
                output_c[0][x] = cc * inv_w;
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
        &mut self,
        (xpos, ypos): (usize, usize),
        xsize: usize,
        row: &mut [(&[&[f32]], &mut [&mut [f32]])],
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
            const SADS_OFF: [[isize; 2]; 4] = [[-1, 0], [0, -1], [0, 1], [1, 0]];
            for ((input_c, _), scale) in row.iter_mut().zip(self.channel_scale) {
                const PLUS_OFF: [[isize; 2]; 5] = [[0, 0], [-1, 0], [0, -1], [1, 0], [0, 1]];
                for (sads_i, sad_off) in sads.iter_mut().zip(SADS_OFF) {
                    let sad = PLUS_OFF.iter().fold(0.0, |acc, off| {
                        let r11 =
                            input_c[(2 + off[0]) as usize][2 + x.saturating_add_signed(off[1])];
                        let c11 = input_c[(2 + sad_off[0] + off[0]) as usize]
                            [2 + x.saturating_add_signed(sad_off[1] + off[1])];
                        acc + (r11 - c11).abs()
                    });
                    *sads_i = sad.mul_add(scale, *sads_i);
                }
            }

            // Compute output based on SADs
            let vsm = sad_mul[ix];
            let inv_sigma = row_sigma[bx] * vsm;
            for (input_c, output_c) in row.iter_mut() {
                let mut cc = input_c[2][2 + x];
                let mut weight = 1.0;
                for (sad, sad_off) in sads.iter().zip(SADS_OFF) {
                    let c =
                        input_c[(2 + sad_off[0]) as usize][2 + x.saturating_add_signed(sad_off[1])];
                    let w = sad.mul_add(inv_sigma, 1.0).max(0.0);

                    weight += w;
                    cc = c.mul_add(w, cc);
                }

                let inv_w = 1.0 / weight;
                output_c[0][x] = cc * inv_w;
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
        &mut self,
        (xpos, ypos): (usize, usize),
        xsize: usize,
        row: &mut [(&[&[f32]], &mut [&mut [f32]])],
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

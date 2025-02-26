// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use std::sync::Arc;

use crate::{
    image::Image,
    render::{RenderPipelineInOutStage, RenderPipelineStage},
    BLOCK_DIM, MIN_SIGMA, SIGMA_PADDING,
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

        for ((input_c, output_c), scale) in row.iter_mut().zip(self.channel_scale) {
            for x in xpos..xpos + xsize {
                let bx = (x + SIGMA_PADDING * BLOCK_DIM) / BLOCK_DIM;
                let ix = x % BLOCK_DIM;

                if row_sigma[bx] < MIN_SIGMA {
                    output_c[0][x] = input_c[3][x];
                    continue;
                }

                let vsm = sad_mul[ix];
                let inv_sigma = row_sigma[bx] * vsm;

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
                const PLUS_OFF: [[isize; 2]; 5] = [[0, 0], [-1, 0], [0, -1], [1, 0], [0, 1]];

                for (sads_i, sad_off) in sads.iter_mut().zip(SADS_OFF) {
                    let sad = PLUS_OFF.iter().fold(0.0, |acc, off| {
                        let r11 = input_c[(3 + off[0]) as usize][x.saturating_add_signed(off[1])];
                        let c11 = input_c[(3 + sad_off[0] + off[0]) as usize]
                            [x.saturating_add_signed(sad_off[1] + off[1])];
                        acc + (r11 - c11).abs()
                    });
                    *sads_i = sad.mul_add(scale, *sads_i);
                }

                let mut cc = input_c[3][x];
                let mut weight = 1.0;
                for (sad, sad_off) in sads.iter().zip(SADS_OFF) {
                    let c = input_c[(3 + sad_off[0]) as usize][x.saturating_add_signed(sad_off[1])];
                    let w = sad.mul_add(inv_sigma, 1.0).min(0.0);

                    weight += w;
                    cc = c.mul_add(w, cc);
                }

                let inv_w = 1.0 / weight;
                output_c[0][x] = cc * inv_w;
            }
        }
    }
}

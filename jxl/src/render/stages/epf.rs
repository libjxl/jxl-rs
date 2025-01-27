use std::{array, sync::Arc};

use crate::{
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
    sigma: Arc<[f32]>,
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
        sigma: Arc<[f32]>,
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
        let row_sigma = &self.sigma[ypos / BLOCK_DIM + SIGMA_PADDING..];

        let sm = self.sigma_scale * 1.65;
        let bsm = sm * self.border_sad_mul;

        assert!(row.len() == 3, "Expected 3 channels, got {}", row.len());
        let input_row: [[&[f32]; 7]; 3] = array::from_fn(|c| array::from_fn(|i| row[c].0[i]));

        let sad_mul = if ypos % BLOCK_DIM == 0 || ypos % BLOCK_DIM == BLOCK_DIM - 1 {
            [bsm; 8] // border
        } else {
            [bsm, sm, sm, sm, sm, sm, sm, bsm] // center
        };

        for x in xpos..xpos + xsize {
            let bx = (x + SIGMA_PADDING * BLOCK_DIM) / BLOCK_DIM;
            let ix = x % BLOCK_DIM;

            if row_sigma[bx] < MIN_SIGMA {
                for c in 0..3 {
                    row[c].1[0][x] = input_row[c][3][x];
                }
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

            for (r_c, scale) in input_row.iter().zip(self.channel_scale) {
                for (sads_i, sad_off) in sads.iter_mut().zip(SADS_OFF) {
                    let sad = PLUS_OFF.iter().fold(0.0, |acc, off| {
                        let r11 = r_c[(3 + off[0]) as usize][(x as isize + off[1]) as usize];
                        let c11 = r_c[(3 + sad_off[0] + off[0]) as usize]
                            [(x as isize + sad_off[1] + off[1]) as usize];
                        acc + (r11 - c11).abs()
                    });
                    *sads_i = sad.mul_add(scale, *sads_i);
                }
            }
            let x_cc = input_row[0][3][x];
            let y_cc = input_row[1][3][x];
            let b_cc = input_row[2][3][x];

            let mut output = [1.0, x_cc, y_cc, b_cc];
            for (sad, sad_off) in sads.iter().zip(SADS_OFF) {
                add_pixel(
                    (3 + sad_off[0]) as usize,
                    &input_row,
                    (x as isize + sad_off[1]) as usize,
                    *sad,
                    inv_sigma,
                    &mut output,
                );
            }

            let inv_w = 1.0 / output[0];
            row[0].1[0][x] = output[1] * inv_w;
            row[1].1[0][x] = output[2] * inv_w;
            row[2].1[0][x] = output[3] * inv_w;
        }
    }
}

fn add_pixel(
    row: usize,
    rows: &[[&[f32]; 7]; 3],
    i: usize,
    sad: f32,
    inv_sigma: f32,
    [w, x, y, b]: &mut [f32; 4],
) {
    let cx = rows[0][row][i];
    let cy = rows[1][row][i];
    let cb = rows[2][row][i];

    let weight = sad.mul_add(inv_sigma, 1.0).min(0.0);
    *w += weight;
    *x = cx.mul_add(weight, *x);
    *y = cy.mul_add(weight, *y);
    *b = cb.mul_add(weight, *b);
}

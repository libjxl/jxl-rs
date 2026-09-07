// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl_simd::{F32SimdVec, SimdMask, simd_function};

use crate::features::epf::SigmaSource;
use crate::render::stages::epf::common::{get_sigma, prepare_sad_mul_storage};
use crate::render::{Channels, ChannelsMut, ErasedLocalState, RenderPipelineInOutStage};
use crate::util::sync::{Arc, RwLock};
use crate::{BLOCK_DIM, MIN_SIGMA};

/// 3x3 plus-shaped kernel with 5 SADs per pixel (3x3 plus-shaped). So this makes this filter a 5x5 filter.
pub struct Epf1Stage {
    /// Multiplier for sigma in pass 1
    sigma_scale: f32,
    /// (inverse) multiplier for sigma on borders
    border_sad_mul: f32,
    channel_scale: [f32; 3],
    sigma: Arc<RwLock<SigmaSource>>,
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
    pub fn new(
        sigma_scale: f32,
        border_sad_mul: f32,
        channel_scale: [f32; 3],
        sigma: Arc<RwLock<SigmaSource>>,
    ) -> Self {
        Self {
            sigma,
            sigma_scale,
            channel_scale,
            border_sad_mul,
        }
    }
}

simd_function!(
epf1_process_row_chunk_dispatch,
d: D,
fn epf1_process_row_chunk(
    stage: &Epf1Stage,
    pos: (usize, usize),
    xsize: usize,
    input_rows: &Channels<f32>,
    output_rows: &mut ChannelsMut<f32>,
) {
    let (xpos, ypos) = pos;
    assert_eq!(input_rows.len(), 3);
    assert_eq!(output_rows.len(), 3);
    let (input_x, input_y, input_b) = (&input_rows[0], &input_rows[1], &input_rows[2]);
    let (output_x, output_y, output_b) = output_rows.split_first_3_mut();
    let (out_x, out_y, out_b) = (&mut output_x[0], &mut output_y[0], &mut output_b[0]);

    let sigma = stage.sigma.try_read().unwrap();
    let row_sigma = sigma.row(ypos / BLOCK_DIM);

    const { assert!(D::F32Vec::LEN <= 16) };

    let sm = stage.sigma_scale * 1.65;
    let bsm = sm * stage.border_sad_mul;
    let sad_mul_storage = prepare_sad_mul_storage(xpos, ypos, sm, bsm);
    let sad_mul_0 = D::F32Vec::load(d, &sad_mul_storage[0..]);

    let scale_x = D::F32Vec::splat(d, stage.channel_scale[0]);
    let scale_y = D::F32Vec::splat(d, stage.channel_scale[1]);
    let scale_b = D::F32Vec::splat(d, stage.channel_scale[2]);
    let one = D::F32Vec::splat(d, 1.0);
    let zero = D::F32Vec::splat(d, 0.0);
    let min_sigma = D::F32Vec::splat(d, MIN_SIGMA);

    for x in (0..xsize).step_by(D::F32Vec::LEN) {
        let sigma = get_sigma(d, x + xpos, row_sigma);
        let sad_mul = if D::F32Vec::LEN >= 8 {
            sad_mul_0
        } else {
            // SAFETY: sad_mul_storage has size at least 8 + D::F32Vec::LEN.
            unsafe { D::F32Vec::load(d, sad_mul_storage.get_unchecked(x % 8..)) }
        };

        let sigma_mask = min_sigma.gt(sigma);
        if sigma_mask.all() {
            // SAFETY: input and output rows have sufficient length for x + D::F32Vec::LEN.
            unsafe {
                D::F32Vec::load(d, input_x[2].get_unchecked(2 + x..)).store(out_x.get_unchecked_mut(x..));
                D::F32Vec::load(d, input_y[2].get_unchecked(2 + x..)).store(out_y.get_unchecked_mut(x..));
                D::F32Vec::load(d, input_b[2].get_unchecked(2 + x..)).store(out_b.get_unchecked_mut(x..));
            }
            continue;
        }

        let inv_sigma = sigma * sad_mul;

        // SAFETY: input rows have at least xsize + 4 elements due to BORDER=(2, 2), and output rows have at least xsize elements.
        unsafe {
            let x_cc = D::F32Vec::load(d, input_x[2].get_unchecked(2 + x..));
            let y_cc = D::F32Vec::load(d, input_y[2].get_unchecked(2 + x..));
            let b_cc = D::F32Vec::load(d, input_b[2].get_unchecked(2 + x..));

            let mut sad_up = zero;
            let mut sad_left = zero;
            let mut sad_right = zero;
            let mut sad_down = zero;

            macro_rules! compute_sads_channel {
                ($input_c:expr, $p_cc:expr, $scale:expr) => {
                    let p22 = $p_cc;
                    let p21 = D::F32Vec::load(d, $input_c[1].get_unchecked(2 + x..));
                    let p23 = D::F32Vec::load(d, $input_c[3].get_unchecked(2 + x..));
                    let p12 = D::F32Vec::load(d, $input_c[2].get_unchecked(1 + x..));
                    let p32 = D::F32Vec::load(d, $input_c[2].get_unchecked(3 + x..));

                    let d22_21 = (p22 - p21).abs();
                    let d22_23 = (p22 - p23).abs();
                    let d12_22 = (p22 - p12).abs();
                    let d22_32 = (p22 - p32).abs();

                    {
                        let p20 = D::F32Vec::load(d, $input_c[0].get_unchecked(2 + x..));
                        let p11 = D::F32Vec::load(d, $input_c[1].get_unchecked(1 + x..));
                        let p31 = D::F32Vec::load(d, $input_c[1].get_unchecked(3 + x..));
                        let d20_21 = (p20 - p21).abs();
                        let d11_12 = (p11 - p12).abs();
                        let d31_32 = (p31 - p32).abs();
                        sad_up = (d20_21 + d11_12 + d22_21 + d31_32 + d22_23).mul_add($scale, sad_up);
                    }

                    {
                        let p13 = D::F32Vec::load(d, $input_c[3].get_unchecked(1 + x..));
                        let p33 = D::F32Vec::load(d, $input_c[3].get_unchecked(3 + x..));
                        let p24 = D::F32Vec::load(d, $input_c[4].get_unchecked(2 + x..));
                        let d13_12 = (p13 - p12).abs();
                        let d33_32 = (p33 - p32).abs();
                        let d24_23 = (p24 - p23).abs();
                        sad_down = (d22_21 + d13_12 + d22_23 + d33_32 + d24_23).mul_add($scale, sad_down);
                    }

                    {
                        let p11 = D::F32Vec::load(d, $input_c[1].get_unchecked(1 + x..));
                        let p02 = D::F32Vec::load(d, $input_c[2].get_unchecked(x..));
                        let p13 = D::F32Vec::load(d, $input_c[3].get_unchecked(1 + x..));
                        let d11_21 = (p11 - p21).abs();
                        let d02_12 = (p02 - p12).abs();
                        let d13_23 = (p13 - p23).abs();
                        sad_left = (d11_21 + d02_12 + d12_22 + d22_32 + d13_23).mul_add($scale, sad_left);
                    }

                    {
                        let p31 = D::F32Vec::load(d, $input_c[1].get_unchecked(3 + x..));
                        let p42 = D::F32Vec::load(d, $input_c[2].get_unchecked(4 + x..));
                        let p33 = D::F32Vec::load(d, $input_c[3].get_unchecked(3 + x..));
                        let d31_21 = (p31 - p21).abs();
                        let d42_32 = (p42 - p32).abs();
                        let d33_23 = (p33 - p23).abs();
                        sad_right = (d31_21 + d12_22 + d22_32 + d42_32 + d33_23).mul_add($scale, sad_right);
                    }
                };
            }

            compute_sads_channel!(input_x, x_cc, scale_x);
            compute_sads_channel!(input_y, y_cc, scale_y);
            compute_sads_channel!(input_b, b_cc, scale_b);

            // Compute weights based on SADs
            let w_up = sad_up.mul_add(inv_sigma, one).max(zero);
            let w_left = sad_left.mul_add(inv_sigma, one).max(zero);
            let w_right = sad_right.mul_add(inv_sigma, one).max(zero);
            let w_down = sad_down.mul_add(inv_sigma, one).max(zero);
            let w_acc = one + w_up + w_left + w_right + w_down;
            let inv_w = one / w_acc;

            macro_rules! compute_output {
                ($input_c:expr, $p_cc:expr, $output_c:expr) => {
                    let p_up = D::F32Vec::load(d, $input_c[1].get_unchecked(2 + x..));
                    let p_left = D::F32Vec::load(d, $input_c[2].get_unchecked(1 + x..));
                    let p_right = D::F32Vec::load(d, $input_c[2].get_unchecked(3 + x..));
                    let p_down = D::F32Vec::load(d, $input_c[3].get_unchecked(2 + x..));
                    let acc = p_up.mul_add(
                        w_up,
                        p_left.mul_add(
                            w_left,
                            p_right.mul_add(w_right, p_down.mul_add(w_down, $p_cc)),
                        ),
                    );
                    let out = sigma_mask.if_then_else_f32($p_cc, acc * inv_w);
                    out.store($output_c.get_unchecked_mut(x..));
                };
            }

            compute_output!(input_x, x_cc, out_x);
            compute_output!(input_y, y_cc, out_y);
            compute_output!(input_b, b_cc, out_b);
        }
    }
});

impl RenderPipelineInOutStage for Epf1Stage {
    type InputT = f32;
    type OutputT = f32;
    const SHIFT: (u8, u8) = (0, 0);
    const BORDER: (u8, u8) = (2, 2);

    fn uses_channel(&self, c: usize) -> bool {
        c < 3
    }

    fn process_row_chunk(
        &self,
        (xpos, ypos): (usize, usize),
        xsize: usize,
        input_rows: &Channels<f32>,
        output_rows: &mut ChannelsMut<f32>,
        _state: Option<&mut ErasedLocalState>,
        _previous_call_was_previous_row: bool,
    ) {
        epf1_process_row_chunk_dispatch(self, (xpos, ypos), xsize, input_rows, output_rows);
    }
}

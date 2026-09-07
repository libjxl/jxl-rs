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

/// 5x5 plus-shaped kernel with 5 SADs per pixel (3x3 plus-shaped). So this makes this filter a 7x7 filter.
pub struct Epf0Stage {
    /// Multiplier for sigma in pass 0
    sigma_scale: f32,
    /// (inverse) multiplier for sigma on borders
    border_sad_mul: f32,
    channel_scale: [f32; 3],
    sigma: Arc<RwLock<SigmaSource>>,
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
    epf0_process_row_chunk_dispatch,
    d: D,
    fn epf0_process_row_chunk_simd(
    stage: &Epf0Stage,
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
                D::F32Vec::load(d, input_x[3].get_unchecked(3 + x..)).store(out_x.get_unchecked_mut(x..));
                D::F32Vec::load(d, input_y[3].get_unchecked(3 + x..)).store(out_y.get_unchecked_mut(x..));
                D::F32Vec::load(d, input_b[3].get_unchecked(3 + x..)).store(out_b.get_unchecked_mut(x..));
            }
            continue;
        }

        let inv_sigma = sigma * sad_mul;

        // SAFETY: input rows have at least xsize + 6 elements due to BORDER=(3, 3), and output rows have at least xsize elements.
        unsafe {
            let x_cc = D::F32Vec::load(d, input_x[3].get_unchecked(3 + x..));
            let y_cc = D::F32Vec::load(d, input_y[3].get_unchecked(3 + x..));
            let b_cc = D::F32Vec::load(d, input_b[3].get_unchecked(3 + x..));

            let mut w_acc = one;
            let mut x_acc = x_cc;
            let mut y_acc = y_cc;
            let mut b_acc = b_cc;

            // --- Group 1: Vertical neighbors (0, 2, 9, 11) ---
            {
                let mut sad0 = zero;
                let mut sad2 = zero;
                let mut sad9 = zero;
                let mut sad11 = zero;

                macro_rules! compute_sads_grp1 {
                    ($input_c:expr, $p_cc:expr, $scale:expr) => {
                        let p32 = D::F32Vec::load(d, $input_c[2].get_unchecked(3 + x..));
                        let p34 = D::F32Vec::load(d, $input_c[4].get_unchecked(3 + x..));
                        let p33 = $p_cc;

                        let d32_34 = (p32 - p34).abs();
                        let d32_33 = (p32 - p33).abs();
                        let d33_34 = (p33 - p34).abs();

                        {
                            let p30 = D::F32Vec::load(d, $input_c[0].get_unchecked(3 + x..));
                            let p21 = D::F32Vec::load(d, $input_c[1].get_unchecked(2 + x..));
                            let p23 = D::F32Vec::load(d, $input_c[3].get_unchecked(2 + x..));
                            let p31 = D::F32Vec::load(d, $input_c[1].get_unchecked(3 + x..));
                            let p41 = D::F32Vec::load(d, $input_c[1].get_unchecked(4 + x..));
                            let p43 = D::F32Vec::load(d, $input_c[3].get_unchecked(4 + x..));
                            let d32_30 = (p32 - p30).abs();
                            let d23_21 = (p23 - p21).abs();
                            let d33_31 = (p33 - p31).abs();
                            let d43_41 = (p43 - p41).abs();
                            sad0 = $scale.mul_add(d32_30 + d23_21 + d33_31 + d43_41 + d32_34, sad0);
                        }

                        {
                            let p31 = D::F32Vec::load(d, $input_c[1].get_unchecked(3 + x..));
                            let p23 = D::F32Vec::load(d, $input_c[3].get_unchecked(2 + x..));
                            let p22 = D::F32Vec::load(d, $input_c[2].get_unchecked(2 + x..));
                            let p43 = D::F32Vec::load(d, $input_c[3].get_unchecked(4 + x..));
                            let p42 = D::F32Vec::load(d, $input_c[2].get_unchecked(4 + x..));
                            let d32_31 = (p32 - p31).abs();
                            let d23_22 = (p23 - p22).abs();
                            let d43_42 = (p43 - p42).abs();
                            sad2 = $scale.mul_add(d32_31 + d23_22 + d32_33 + d43_42 + d33_34, sad2);
                        }

                        {
                            let p23 = D::F32Vec::load(d, $input_c[3].get_unchecked(2 + x..));
                            let p24 = D::F32Vec::load(d, $input_c[4].get_unchecked(2 + x..));
                            let p43 = D::F32Vec::load(d, $input_c[3].get_unchecked(4 + x..));
                            let p44 = D::F32Vec::load(d, $input_c[4].get_unchecked(4 + x..));
                            let p35 = D::F32Vec::load(d, $input_c[5].get_unchecked(3 + x..));
                            let d23_24 = (p23 - p24).abs();
                            let d43_44 = (p43 - p44).abs();
                            let d34_35 = (p34 - p35).abs();
                            sad9 = $scale.mul_add(d32_33 + d23_24 + d33_34 + d43_44 + d34_35, sad9);
                        }

                        {
                            let p23 = D::F32Vec::load(d, $input_c[3].get_unchecked(2 + x..));
                            let p25 = D::F32Vec::load(d, $input_c[5].get_unchecked(2 + x..));
                            let p35 = D::F32Vec::load(d, $input_c[5].get_unchecked(3 + x..));
                            let p43 = D::F32Vec::load(d, $input_c[3].get_unchecked(4 + x..));
                            let p45 = D::F32Vec::load(d, $input_c[5].get_unchecked(4 + x..));
                            let p36 = D::F32Vec::load(d, $input_c[6].get_unchecked(3 + x..));
                            let d23_25 = (p23 - p25).abs();
                            let d33_35 = (p33 - p35).abs();
                            let d43_45 = (p43 - p45).abs();
                            let d34_36 = (p34 - p36).abs();
                            sad11 = $scale.mul_add(d32_34 + d23_25 + d33_35 + d43_45 + d34_36, sad11);
                        }
                    };
                }

                compute_sads_grp1!(input_x, x_cc, scale_x);
                compute_sads_grp1!(input_y, y_cc, scale_y);
                compute_sads_grp1!(input_b, b_cc, scale_b);

                let w0 = sad0.mul_add(inv_sigma, one).max(zero);
                let w2 = sad2.mul_add(inv_sigma, one).max(zero);
                let w9 = sad9.mul_add(inv_sigma, one).max(zero);
                let w11 = sad11.mul_add(inv_sigma, one).max(zero);
                w_acc += w0 + w2 + w9 + w11;

                macro_rules! accumulate_grp1 {
                    ($input_c:expr, $acc:ident) => {
                        let p31 = D::F32Vec::load(d, $input_c[1].get_unchecked(3 + x..));
                        let p32 = D::F32Vec::load(d, $input_c[2].get_unchecked(3 + x..));
                        let p34 = D::F32Vec::load(d, $input_c[4].get_unchecked(3 + x..));
                        let p35 = D::F32Vec::load(d, $input_c[5].get_unchecked(3 + x..));
                        $acc = p31.mul_add(
                            w0,
                            p32.mul_add(w2, p34.mul_add(w9, p35.mul_add(w11, $acc))),
                        );
                    };
                }

                accumulate_grp1!(input_x, x_acc);
                accumulate_grp1!(input_y, y_acc);
                accumulate_grp1!(input_b, b_acc);
            }

            // --- Group 2: Horizontal neighbors (4, 5, 6, 7) ---
            {
                let mut sad4 = zero;
                let mut sad5 = zero;
                let mut sad6 = zero;
                let mut sad7 = zero;

                macro_rules! compute_sads_grp2 {
                    ($input_c:expr, $p_cc:expr, $scale:expr) => {
                        let p23 = D::F32Vec::load(d, $input_c[3].get_unchecked(2 + x..));
                        let p43 = D::F32Vec::load(d, $input_c[3].get_unchecked(4 + x..));
                        let p33 = $p_cc;

                        let d23_43 = (p23 - p43).abs();
                        let d23_33 = (p23 - p33).abs();
                        let d33_43 = (p33 - p43).abs();

                        {
                            let p32 = D::F32Vec::load(d, $input_c[2].get_unchecked(3 + x..));
                            let p12 = D::F32Vec::load(d, $input_c[2].get_unchecked(1 + x..));
                            let p03 = D::F32Vec::load(d, $input_c[3].get_unchecked(x..));
                            let p13 = D::F32Vec::load(d, $input_c[3].get_unchecked(1 + x..));
                            let p34 = D::F32Vec::load(d, $input_c[4].get_unchecked(3 + x..));
                            let p14 = D::F32Vec::load(d, $input_c[4].get_unchecked(1 + x..));
                            let d32_12 = (p32 - p12).abs();
                            let d23_03 = (p23 - p03).abs();
                            let d33_13 = (p33 - p13).abs();
                            let d34_14 = (p34 - p14).abs();
                            sad4 = $scale.mul_add(d32_12 + d23_03 + d33_13 + d23_43 + d34_14, sad4);
                        }

                        {
                            let p32 = D::F32Vec::load(d, $input_c[2].get_unchecked(3 + x..));
                            let p22 = D::F32Vec::load(d, $input_c[2].get_unchecked(2 + x..));
                            let p13 = D::F32Vec::load(d, $input_c[3].get_unchecked(1 + x..));
                            let p34 = D::F32Vec::load(d, $input_c[4].get_unchecked(3 + x..));
                            let p24 = D::F32Vec::load(d, $input_c[4].get_unchecked(2 + x..));
                            let d32_22 = (p32 - p22).abs();
                            let d23_13 = (p23 - p13).abs();
                            let d34_24 = (p34 - p24).abs();
                            sad5 = $scale.mul_add(d32_22 + d23_13 + d23_33 + d33_43 + d34_24, sad5);
                        }

                        {
                            let p32 = D::F32Vec::load(d, $input_c[2].get_unchecked(3 + x..));
                            let p42 = D::F32Vec::load(d, $input_c[2].get_unchecked(4 + x..));
                            let p53 = D::F32Vec::load(d, $input_c[3].get_unchecked(5 + x..));
                            let p34 = D::F32Vec::load(d, $input_c[4].get_unchecked(3 + x..));
                            let p44 = D::F32Vec::load(d, $input_c[4].get_unchecked(4 + x..));
                            let d32_42 = (p32 - p42).abs();
                            let d43_53 = (p43 - p53).abs();
                            let d34_44 = (p34 - p44).abs();
                            sad6 = $scale.mul_add(d32_42 + d23_33 + d33_43 + d43_53 + d34_44, sad6);
                        }

                        {
                            let p32 = D::F32Vec::load(d, $input_c[2].get_unchecked(3 + x..));
                            let p52 = D::F32Vec::load(d, $input_c[2].get_unchecked(5 + x..));
                            let p53 = D::F32Vec::load(d, $input_c[3].get_unchecked(5 + x..));
                            let p63 = D::F32Vec::load(d, $input_c[3].get_unchecked(6 + x..));
                            let p34 = D::F32Vec::load(d, $input_c[4].get_unchecked(3 + x..));
                            let p54 = D::F32Vec::load(d, $input_c[4].get_unchecked(5 + x..));
                            let d32_52 = (p32 - p52).abs();
                            let d33_53 = (p33 - p53).abs();
                            let d43_63 = (p43 - p63).abs();
                            let d34_54 = (p34 - p54).abs();
                            sad7 = $scale.mul_add(d32_52 + d23_43 + d33_53 + d43_63 + d34_54, sad7);
                        }
                    };
                }

                compute_sads_grp2!(input_x, x_cc, scale_x);
                compute_sads_grp2!(input_y, y_cc, scale_y);
                compute_sads_grp2!(input_b, b_cc, scale_b);

                let w4 = sad4.mul_add(inv_sigma, one).max(zero);
                let w5 = sad5.mul_add(inv_sigma, one).max(zero);
                let w6 = sad6.mul_add(inv_sigma, one).max(zero);
                let w7 = sad7.mul_add(inv_sigma, one).max(zero);
                w_acc += w4 + w5 + w6 + w7;

                macro_rules! accumulate_grp2 {
                    ($input_c:expr, $acc:ident) => {
                        let p13 = D::F32Vec::load(d, $input_c[3].get_unchecked(1 + x..));
                        let p23 = D::F32Vec::load(d, $input_c[3].get_unchecked(2 + x..));
                        let p43 = D::F32Vec::load(d, $input_c[3].get_unchecked(4 + x..));
                        let p53 = D::F32Vec::load(d, $input_c[3].get_unchecked(5 + x..));
                        $acc = p13.mul_add(
                            w4,
                            p23.mul_add(w5, p43.mul_add(w6, p53.mul_add(w7, $acc))),
                        );
                    };
                }

                accumulate_grp2!(input_x, x_acc);
                accumulate_grp2!(input_y, y_acc);
                accumulate_grp2!(input_b, b_acc);
            }

            // --- Group 3: Diagonal neighbors (1, 3, 8, 10) ---
            {
                let mut sad1 = zero;
                let mut sad3 = zero;
                let mut sad8 = zero;
                let mut sad10 = zero;

                macro_rules! compute_sads_grp3 {
                    ($input_c:expr, $p_cc:expr, $scale:expr) => {
                        let p32 = D::F32Vec::load(d, $input_c[2].get_unchecked(3 + x..));
                        let p43 = D::F32Vec::load(d, $input_c[3].get_unchecked(4 + x..));
                        let p23 = D::F32Vec::load(d, $input_c[3].get_unchecked(2 + x..));
                        let p34 = D::F32Vec::load(d, $input_c[4].get_unchecked(3 + x..));
                        let p33 = $p_cc;

                        let d32_43 = (p32 - p43).abs();
                        let d23_34 = (p23 - p34).abs();
                        let d32_23 = (p32 - p23).abs();
                        let d43_34 = (p43 - p34).abs();

                        {
                            let p21 = D::F32Vec::load(d, $input_c[1].get_unchecked(2 + x..));
                            let p12 = D::F32Vec::load(d, $input_c[2].get_unchecked(1 + x..));
                            let p22 = D::F32Vec::load(d, $input_c[2].get_unchecked(2 + x..));
                            let d32_21 = (p32 - p21).abs();
                            let d23_12 = (p23 - p12).abs();
                            let d33_22 = (p33 - p22).abs();
                            sad1 = $scale.mul_add(d32_21 + d23_12 + d33_22 + d32_43 + d23_34, sad1);
                        }

                        {
                            let p41 = D::F32Vec::load(d, $input_c[1].get_unchecked(4 + x..));
                            let p42 = D::F32Vec::load(d, $input_c[2].get_unchecked(4 + x..));
                            let p52 = D::F32Vec::load(d, $input_c[2].get_unchecked(5 + x..));
                            let d32_41 = (p32 - p41).abs();
                            let d33_42 = (p33 - p42).abs();
                            let d43_52 = (p43 - p52).abs();
                            sad3 = $scale.mul_add(d32_41 + d32_23 + d33_42 + d43_52 + d43_34, sad3);
                        }

                        {
                            let p14 = D::F32Vec::load(d, $input_c[4].get_unchecked(1 + x..));
                            let p24 = D::F32Vec::load(d, $input_c[4].get_unchecked(2 + x..));
                            let p25 = D::F32Vec::load(d, $input_c[5].get_unchecked(2 + x..));
                            let d23_14 = (p23 - p14).abs();
                            let d33_24 = (p33 - p24).abs();
                            let d34_25 = (p34 - p25).abs();
                            sad8 = $scale.mul_add(d32_23 + d23_14 + d33_24 + d43_34 + d34_25, sad8);
                        }

                        {
                            let p44 = D::F32Vec::load(d, $input_c[4].get_unchecked(4 + x..));
                            let p54 = D::F32Vec::load(d, $input_c[4].get_unchecked(5 + x..));
                            let p45 = D::F32Vec::load(d, $input_c[5].get_unchecked(4 + x..));
                            let d33_44 = (p33 - p44).abs();
                            let d43_54 = (p43 - p54).abs();
                            let d34_45 = (p34 - p45).abs();
                            sad10 = $scale.mul_add(d32_43 + d23_34 + d33_44 + d43_54 + d34_45, sad10);
                        }
                    };
                }

                compute_sads_grp3!(input_x, x_cc, scale_x);
                compute_sads_grp3!(input_y, y_cc, scale_y);
                compute_sads_grp3!(input_b, b_cc, scale_b);

                let w1 = sad1.mul_add(inv_sigma, one).max(zero);
                let w3 = sad3.mul_add(inv_sigma, one).max(zero);
                let w8 = sad8.mul_add(inv_sigma, one).max(zero);
                let w10 = sad10.mul_add(inv_sigma, one).max(zero);
                w_acc += w1 + w3 + w8 + w10;

                macro_rules! accumulate_grp3 {
                    ($input_c:expr, $acc:ident) => {
                        let p22 = D::F32Vec::load(d, $input_c[2].get_unchecked(2 + x..));
                        let p42 = D::F32Vec::load(d, $input_c[2].get_unchecked(4 + x..));
                        let p24 = D::F32Vec::load(d, $input_c[4].get_unchecked(2 + x..));
                        let p44 = D::F32Vec::load(d, $input_c[4].get_unchecked(4 + x..));
                        $acc = p22.mul_add(
                            w1,
                            p42.mul_add(w3, p24.mul_add(w8, p44.mul_add(w10, $acc))),
                        );
                    };
                }

                accumulate_grp3!(input_x, x_acc);
                accumulate_grp3!(input_y, y_acc);
                accumulate_grp3!(input_b, b_acc);
            }

            let inv_w = one / w_acc;
            let x_out = sigma_mask.if_then_else_f32(x_cc, x_acc * inv_w);
            let y_out = sigma_mask.if_then_else_f32(y_cc, y_acc * inv_w);
            let b_out = sigma_mask.if_then_else_f32(b_cc, b_acc * inv_w);
            x_out.store(out_x.get_unchecked_mut(x..));
            y_out.store(out_y.get_unchecked_mut(x..));
            b_out.store(out_b.get_unchecked_mut(x..));
        }
    }
});

impl RenderPipelineInOutStage for Epf0Stage {
    type InputT = f32;
    type OutputT = f32;
    const SHIFT: (u8, u8) = (0, 0);
    const BORDER: (u8, u8) = (3, 3);

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
        epf0_process_row_chunk_dispatch(self, (xpos, ypos), xsize, input_rows, output_rows);
    }
}

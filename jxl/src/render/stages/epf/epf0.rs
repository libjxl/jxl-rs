// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl_simd::{F32SimdVec, SimdDescriptor, SimdMask, simd_function};

use crate::features::epf::SigmaSource;
use crate::render::stages::epf::common::{get_sigma, prepare_sad_mul_storage};
use crate::render::stages::row_chunks::{Window, for_each_chunk};
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

#[inline(always)]
fn epf0_process_row_chunk_impl<D: SimdDescriptor>(
    d: D,
    stage: &Epf0Stage,
    pos: (usize, usize),
    xsize: usize,
    input_rows: &Channels<f32>,
    output_rows: &mut ChannelsMut<f32>,
) {
    let (xpos, ypos) = pos;
    assert_eq!(input_rows.len(), 3);
    assert_eq!(output_rows.len(), 3);
    let (output_x, output_y, output_b) = output_rows.split_first_3_mut();

    let sigma = stage.sigma.try_read().unwrap();
    let row_sigma = sigma.row(ypos / BLOCK_DIM);

    const { assert!(D::F32Vec::LEN <= 16) };

    let sm = stage.sigma_scale * 1.65;
    let bsm = sm * stage.border_sad_mul;
    let sad_mul_storage = prepare_sad_mul_storage(xpos, ypos, sm, bsm);

    let inputs: [[Window<6>; 7]; 3] =
        std::array::from_fn(|c| std::array::from_fn(|r| Window(input_rows[c][r])));
    let outputs: [&mut [f32]; 3] = [output_x[0], output_y[0], output_b[0]];

    for_each_chunk(
        d,
        xsize,
        (inputs, outputs),
        #[inline(always)]
        |x, (in_chunks, mut out_chunks)| {
            let sigma = get_sigma(d, x + xpos, row_sigma);
            let sad_mul = D::F32Vec::load(d, &sad_mul_storage[x % 8..]);

            let p33_x = in_chunks[0][3].get::<3>();
            let p33_y = in_chunks[1][3].get::<3>();
            let p33_b = in_chunks[2][3].get::<3>();

            let sigma_mask = D::F32Vec::splat(d, MIN_SIGMA).gt(sigma);
            if sigma_mask.all() {
                out_chunks[0].write(p33_x);
                out_chunks[1].write(p33_y);
                out_chunks[2].write(p33_b);
                return;
            }

            let inv_sigma = sigma * sad_mul;

            let mut w_acc = D::F32Vec::splat(d, 1.0);
            let mut out_x = p33_x;
            let mut out_y = p33_y;
            let mut out_b = p33_b;

            // Group A: Vertical Axis (4 neighbors: (0, -2), (0, -1), (0, +1), (0, +2))
            let mut sad0 = D::F32Vec::splat(d, 0.0);
            let mut sad2 = D::F32Vec::splat(d, 0.0);
            let mut sad9 = D::F32Vec::splat(d, 0.0);
            let mut sad11 = D::F32Vec::splat(d, 0.0);
            for (c, &scale) in stage.channel_scale.iter().enumerate() {
                let scale = D::F32Vec::splat(d, scale);

                let p32 = in_chunks[c][2].get::<3>();
                let p23 = in_chunks[c][3].get::<2>();
                let p33 = in_chunks[c][3].get::<3>();
                let p43 = in_chunks[c][3].get::<4>();
                let p34 = in_chunks[c][4].get::<3>();

                let d32_34 = (p32 - p34).abs();
                let shared_2_9 = (p32 - p33).abs() + (p33 - p34).abs();

                // Row 0
                let p30 = in_chunks[c][0].get::<3>();
                let mut d0 = d32_34 + (p32 - p30).abs();

                // Row 1
                let p21 = in_chunks[c][1].get::<2>();
                let p31 = in_chunks[c][1].get::<3>();
                let p41 = in_chunks[c][1].get::<4>();
                d0 += (p23 - p21).abs() + (p33 - p31).abs() + (p43 - p41).abs();
                sad0 = d0.mul_add(scale, sad0);
                let mut d2 = shared_2_9 + (p32 - p31).abs();

                // Row 2
                let p22 = in_chunks[c][2].get::<2>();
                let p42 = in_chunks[c][2].get::<4>();
                d2 += (p23 - p22).abs() + (p43 - p42).abs();
                sad2 = d2.mul_add(scale, sad2);

                // Row 4
                let p24 = in_chunks[c][4].get::<2>();
                let p44 = in_chunks[c][4].get::<4>();
                let mut d9 = shared_2_9 + (p23 - p24).abs() + (p43 - p44).abs();

                // Row 5
                let p25 = in_chunks[c][5].get::<2>();
                let p35 = in_chunks[c][5].get::<3>();
                let p45 = in_chunks[c][5].get::<4>();
                d9 += (p34 - p35).abs();
                sad9 = d9.mul_add(scale, sad9);
                let mut d11 = d32_34 + (p23 - p25).abs() + (p33 - p35).abs() + (p43 - p45).abs();

                // Row 6
                let p36 = in_chunks[c][6].get::<3>();
                d11 += (p34 - p36).abs();
                sad11 = d11.mul_add(scale, sad11);
            }

            let w0 = sad0
                .mul_add(inv_sigma, D::F32Vec::splat(d, 1.0))
                .max(D::F32Vec::splat(d, 0.0));
            let w2 = sad2
                .mul_add(inv_sigma, D::F32Vec::splat(d, 1.0))
                .max(D::F32Vec::splat(d, 0.0));
            let w9 = sad9
                .mul_add(inv_sigma, D::F32Vec::splat(d, 1.0))
                .max(D::F32Vec::splat(d, 0.0));
            let w11 = sad11
                .mul_add(inv_sigma, D::F32Vec::splat(d, 1.0))
                .max(D::F32Vec::splat(d, 0.0));
            w_acc += (w0 + w2) + (w9 + w11);
            out_x = w0.mul_add(
                in_chunks[0][1].get::<3>(),
                w2.mul_add(
                    in_chunks[0][2].get::<3>(),
                    w9.mul_add(
                        in_chunks[0][4].get::<3>(),
                        w11.mul_add(in_chunks[0][5].get::<3>(), out_x),
                    ),
                ),
            );
            out_y = w0.mul_add(
                in_chunks[1][1].get::<3>(),
                w2.mul_add(
                    in_chunks[1][2].get::<3>(),
                    w9.mul_add(
                        in_chunks[1][4].get::<3>(),
                        w11.mul_add(in_chunks[1][5].get::<3>(), out_y),
                    ),
                ),
            );
            out_b = w0.mul_add(
                in_chunks[2][1].get::<3>(),
                w2.mul_add(
                    in_chunks[2][2].get::<3>(),
                    w9.mul_add(
                        in_chunks[2][4].get::<3>(),
                        w11.mul_add(in_chunks[2][5].get::<3>(), out_b),
                    ),
                ),
            );

            // Group B: Horizontal Axis (4 neighbors: (-2, 0), (-1, 0), (+1, 0), (+2, 0))
            let mut sad4 = D::F32Vec::splat(d, 0.0);
            let mut sad5 = D::F32Vec::splat(d, 0.0);
            let mut sad6 = D::F32Vec::splat(d, 0.0);
            let mut sad7 = D::F32Vec::splat(d, 0.0);
            for (c, &scale) in stage.channel_scale.iter().enumerate() {
                let scale = D::F32Vec::splat(d, scale);

                let p32 = in_chunks[c][2].get::<3>();
                let p23 = in_chunks[c][3].get::<2>();
                let p33 = in_chunks[c][3].get::<3>();
                let p43 = in_chunks[c][3].get::<4>();
                let p34 = in_chunks[c][4].get::<3>();

                let d23_43 = (p23 - p43).abs();
                let shared_5_6 = (p23 - p33).abs() + (p33 - p43).abs();

                // Row 2
                let p12 = in_chunks[c][2].get::<1>();
                let p22 = in_chunks[c][2].get::<2>();
                let p42 = in_chunks[c][2].get::<4>();
                let p52 = in_chunks[c][2].get::<5>();
                let mut d4 = d23_43 + (p32 - p12).abs();
                let mut d5 = shared_5_6 + (p32 - p22).abs();
                let mut d6 = shared_5_6 + (p32 - p42).abs();
                let mut d7 = d23_43 + (p32 - p52).abs();

                // Row 3
                let p03 = in_chunks[c][3].get::<0>();
                let p13 = in_chunks[c][3].get::<1>();
                let p53 = in_chunks[c][3].get::<5>();
                let p63 = in_chunks[c][3].get::<6>();
                d4 += (p23 - p03).abs() + (p33 - p13).abs();
                d5 += (p23 - p13).abs();
                d6 += (p43 - p53).abs();
                d7 += (p33 - p53).abs() + (p43 - p63).abs();

                // Row 4
                let p14 = in_chunks[c][4].get::<1>();
                let p24 = in_chunks[c][4].get::<2>();
                let p44 = in_chunks[c][4].get::<4>();
                let p54 = in_chunks[c][4].get::<5>();
                d4 += (p34 - p14).abs();
                d5 += (p34 - p24).abs();
                d6 += (p34 - p44).abs();
                d7 += (p34 - p54).abs();

                sad4 = d4.mul_add(scale, sad4);
                sad5 = d5.mul_add(scale, sad5);
                sad6 = d6.mul_add(scale, sad6);
                sad7 = d7.mul_add(scale, sad7);
            }

            let w4 = sad4
                .mul_add(inv_sigma, D::F32Vec::splat(d, 1.0))
                .max(D::F32Vec::splat(d, 0.0));
            let w5 = sad5
                .mul_add(inv_sigma, D::F32Vec::splat(d, 1.0))
                .max(D::F32Vec::splat(d, 0.0));
            let w6 = sad6
                .mul_add(inv_sigma, D::F32Vec::splat(d, 1.0))
                .max(D::F32Vec::splat(d, 0.0));
            let w7 = sad7
                .mul_add(inv_sigma, D::F32Vec::splat(d, 1.0))
                .max(D::F32Vec::splat(d, 0.0));
            w_acc += (w4 + w5) + (w6 + w7);
            out_x = w4.mul_add(
                in_chunks[0][3].get::<1>(),
                w5.mul_add(
                    in_chunks[0][3].get::<2>(),
                    w6.mul_add(
                        in_chunks[0][3].get::<4>(),
                        w7.mul_add(in_chunks[0][3].get::<5>(), out_x),
                    ),
                ),
            );
            out_y = w4.mul_add(
                in_chunks[1][3].get::<1>(),
                w5.mul_add(
                    in_chunks[1][3].get::<2>(),
                    w6.mul_add(
                        in_chunks[1][3].get::<4>(),
                        w7.mul_add(in_chunks[1][3].get::<5>(), out_y),
                    ),
                ),
            );
            out_b = w4.mul_add(
                in_chunks[2][3].get::<1>(),
                w5.mul_add(
                    in_chunks[2][3].get::<2>(),
                    w6.mul_add(
                        in_chunks[2][3].get::<4>(),
                        w7.mul_add(in_chunks[2][3].get::<5>(), out_b),
                    ),
                ),
            );

            // Group C: Diagonal Axis (4 neighbors: (-1, -1), (+1, -1), (-1, +1), (+1, +1))
            let mut sad1 = D::F32Vec::splat(d, 0.0);
            let mut sad3 = D::F32Vec::splat(d, 0.0);
            let mut sad8 = D::F32Vec::splat(d, 0.0);
            let mut sad10 = D::F32Vec::splat(d, 0.0);
            for (c, &scale) in stage.channel_scale.iter().enumerate() {
                let scale = D::F32Vec::splat(d, scale);

                let p32 = in_chunks[c][2].get::<3>();
                let p23 = in_chunks[c][3].get::<2>();
                let p33 = in_chunks[c][3].get::<3>();
                let p43 = in_chunks[c][3].get::<4>();
                let p34 = in_chunks[c][4].get::<3>();

                let shared_1_10 = (p32 - p43).abs() + (p23 - p34).abs();
                let shared_3_8 = (p32 - p23).abs() + (p43 - p34).abs();

                // Row 1
                let p21 = in_chunks[c][1].get::<2>();
                let p41 = in_chunks[c][1].get::<4>();
                let mut d1 = shared_1_10 + (p32 - p21).abs();
                let mut d3 = shared_3_8 + (p32 - p41).abs();

                // Row 2
                let p12 = in_chunks[c][2].get::<1>();
                let p22 = in_chunks[c][2].get::<2>();
                let p42 = in_chunks[c][2].get::<4>();
                let p52 = in_chunks[c][2].get::<5>();
                d1 += (p23 - p12).abs() + (p33 - p22).abs();
                d3 += (p33 - p42).abs() + (p43 - p52).abs();
                sad1 = d1.mul_add(scale, sad1);
                sad3 = d3.mul_add(scale, sad3);

                // Row 4
                let p14 = in_chunks[c][4].get::<1>();
                let p24 = in_chunks[c][4].get::<2>();
                let p44 = in_chunks[c][4].get::<4>();
                let p54 = in_chunks[c][4].get::<5>();
                let mut d8 = shared_3_8 + (p23 - p14).abs() + (p33 - p24).abs();
                let mut d10 = shared_1_10 + (p33 - p44).abs() + (p43 - p54).abs();

                // Row 5
                let p25 = in_chunks[c][5].get::<2>();
                let p45 = in_chunks[c][5].get::<4>();
                d8 += (p34 - p25).abs();
                d10 += (p34 - p45).abs();
                sad8 = d8.mul_add(scale, sad8);
                sad10 = d10.mul_add(scale, sad10);
            }

            let w1 = sad1
                .mul_add(inv_sigma, D::F32Vec::splat(d, 1.0))
                .max(D::F32Vec::splat(d, 0.0));
            let w3 = sad3
                .mul_add(inv_sigma, D::F32Vec::splat(d, 1.0))
                .max(D::F32Vec::splat(d, 0.0));
            let w8 = sad8
                .mul_add(inv_sigma, D::F32Vec::splat(d, 1.0))
                .max(D::F32Vec::splat(d, 0.0));
            let w10 = sad10
                .mul_add(inv_sigma, D::F32Vec::splat(d, 1.0))
                .max(D::F32Vec::splat(d, 0.0));
            w_acc += (w1 + w3) + (w8 + w10);
            out_x = w1.mul_add(
                in_chunks[0][2].get::<2>(),
                w3.mul_add(
                    in_chunks[0][2].get::<4>(),
                    w8.mul_add(
                        in_chunks[0][4].get::<2>(),
                        w10.mul_add(in_chunks[0][4].get::<4>(), out_x),
                    ),
                ),
            );
            out_y = w1.mul_add(
                in_chunks[1][2].get::<2>(),
                w3.mul_add(
                    in_chunks[1][2].get::<4>(),
                    w8.mul_add(
                        in_chunks[1][4].get::<2>(),
                        w10.mul_add(in_chunks[1][4].get::<4>(), out_y),
                    ),
                ),
            );
            out_b = w1.mul_add(
                in_chunks[2][2].get::<2>(),
                w3.mul_add(
                    in_chunks[2][2].get::<4>(),
                    w8.mul_add(
                        in_chunks[2][4].get::<2>(),
                        w10.mul_add(in_chunks[2][4].get::<4>(), out_b),
                    ),
                ),
            );

            // Normalize and write
            let inv_w = D::F32Vec::splat(d, 1.0) / w_acc;
            let out_x = sigma_mask.if_then_else_f32(p33_x, out_x * inv_w);
            let out_y = sigma_mask.if_then_else_f32(p33_y, out_y * inv_w);
            let out_b = sigma_mask.if_then_else_f32(p33_b, out_b * inv_w);
            out_chunks[0].write(out_x);
            out_chunks[1].write(out_y);
            out_chunks[2].write(out_b);
        },
    );
}

simd_function!(
    epf0_process_row_chunk_dispatch,
    d: D,
    fn epf0_process_row_chunk(
        stage: &Epf0Stage,
        pos: (usize, usize),
        xsize: usize,
        input_rows: &Channels<f32>,
        output_rows: &mut ChannelsMut<f32>,
    ) {
        epf0_process_row_chunk_impl(d, stage, pos, xsize, input_rows, output_rows)
    }
);

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

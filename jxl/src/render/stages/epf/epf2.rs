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

/// 3x3 plus-shaped kernel with 1 SAD per pixel. So this makes this filter a 3x3 filter.
pub struct Epf2Stage {
    /// Multiplier for sigma in pass 2
    sigma_scale: f32,
    /// (inverse) multiplier for sigma on borders
    border_sad_mul: f32,
    channel_scale: [f32; 3],
    sigma: Arc<RwLock<SigmaSource>>,
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
fn epf2_process_row_chunk_impl<D: SimdDescriptor>(
    d: D,
    stage: &Epf2Stage,
    pos: (usize, usize),
    xsize: usize,
    input_rows: &Channels<f32>,
    output_rows: &mut ChannelsMut<f32>,
) {
    let (xpos, ypos) = pos;
    assert_eq!(
        input_rows.len(),
        3,
        "Expected 3 channels, got {}",
        input_rows.len()
    );
    let (output_x, output_y, output_b) = output_rows.split_first_3_mut();

    let sigma = stage.sigma.try_read().unwrap();
    let row_sigma = sigma.row(ypos / BLOCK_DIM);

    const { assert!(D::F32Vec::LEN <= 16) };

    let sm = stage.sigma_scale * 1.65;
    let bsm = sm * stage.border_sad_mul;
    let sad_mul_storage = prepare_sad_mul_storage(xpos, ypos, sm, bsm);

    let inputs: [[Window<2>; 3]; 3] =
        std::array::from_fn(|c| std::array::from_fn(|r| Window(input_rows[c][r])));
    let outputs: [&mut [f32]; 3] = [output_x[0], output_y[0], output_b[0]];

    let scale0 = D::F32Vec::splat(d, stage.channel_scale[0]);
    let scale1 = D::F32Vec::splat(d, stage.channel_scale[1]);
    let scale2 = D::F32Vec::splat(d, stage.channel_scale[2]);

    for_each_chunk(
        d,
        xsize,
        (inputs, outputs),
        #[inline(always)]
        |x, (in_chunks, mut out_chunks)| {
            let sigma = get_sigma(d, x + xpos, row_sigma);
            let sad_mul = D::F32Vec::load(d, &sad_mul_storage[x % 8..]);

            let x_cc = in_chunks[0][1].get::<1>();
            let y_cc = in_chunks[1][1].get::<1>();
            let b_cc = in_chunks[2][1].get::<1>();

            let sigma_mask = D::F32Vec::splat(d, MIN_SIGMA).gt(sigma);
            if sigma_mask.all() {
                out_chunks[0].write(x_cc);
                out_chunks[1].write(y_cc);
                out_chunks[2].write(b_cc);
                return;
            }

            let inv_sigma = sigma * sad_mul;

            let mut w_acc = D::F32Vec::splat(d, 1.0);
            let mut x_acc = x_cc;
            let mut y_acc = y_cc;
            let mut b_acc = b_cc;

            macro_rules! process_neighbor {
                ($cx:expr, $cy:expr, $cb:expr) => {{
                    let cx = $cx;
                    let cy = $cy;
                    let cb = $cb;
                    let sad = (cx - x_cc).abs().mul_add(
                        scale0,
                        (cy - y_cc)
                            .abs()
                            .mul_add(scale1, (cb - b_cc).abs() * scale2),
                    );
                    let weight = sad
                        .mul_add(inv_sigma, D::F32Vec::splat(d, 1.0))
                        .max(D::F32Vec::splat(d, 0.0));
                    w_acc += weight;
                    x_acc = weight.mul_add(cx, x_acc);
                    y_acc = weight.mul_add(cy, y_acc);
                    b_acc = weight.mul_add(cb, b_acc);
                }};
            }

            process_neighbor!(
                in_chunks[0][0].get::<1>(),
                in_chunks[1][0].get::<1>(),
                in_chunks[2][0].get::<1>()
            );
            process_neighbor!(
                in_chunks[0][1].get::<0>(),
                in_chunks[1][1].get::<0>(),
                in_chunks[2][1].get::<0>()
            );
            process_neighbor!(
                in_chunks[0][1].get::<2>(),
                in_chunks[1][1].get::<2>(),
                in_chunks[2][1].get::<2>()
            );
            process_neighbor!(
                in_chunks[0][2].get::<1>(),
                in_chunks[1][2].get::<1>(),
                in_chunks[2][2].get::<1>()
            );

            let inv_w = D::F32Vec::splat(d, 1.0) / w_acc;

            x_acc *= inv_w;
            y_acc *= inv_w;
            b_acc *= inv_w;
            x_acc = sigma_mask.if_then_else_f32(x_cc, x_acc);
            y_acc = sigma_mask.if_then_else_f32(y_cc, y_acc);
            b_acc = sigma_mask.if_then_else_f32(b_cc, b_acc);
            out_chunks[0].write(x_acc);
            out_chunks[1].write(y_acc);
            out_chunks[2].write(b_acc);
        },
    );
}

simd_function!(
    epf2_process_row_chunk_dispatch,
    d: D,
    fn epf2_process_row_chunk(
        stage: &Epf2Stage,
        pos: (usize, usize),
        xsize: usize,
        input_rows: &Channels<f32>,
        output_rows: &mut ChannelsMut<f32>,
    ) {
        epf2_process_row_chunk_impl(d, stage, pos, xsize, input_rows, output_rows)
    }
);

impl RenderPipelineInOutStage for Epf2Stage {
    type InputT = f32;
    type OutputT = f32;
    const SHIFT: (u8, u8) = (0, 0);
    const BORDER: (u8, u8) = (1, 1);

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
        epf2_process_row_chunk_dispatch(self, (xpos, ypos), xsize, input_rows, output_rows);
    }
}

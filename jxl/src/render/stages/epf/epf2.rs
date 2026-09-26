// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

use jxl_simd::{F32SimdVec, SimdDescriptor, SimdMask, simd_function};

use crate::features::epf::SigmaSource;
use crate::render::stages::epf::common::{get_sigma, prepare_sad_mul_storage};
use crate::render::{
    Channels, ChannelsMut, ErasedLocalState, ForEachChunk, RenderPipelineInOutStage,
};
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
fn epf2_process_simd<D: SimdDescriptor>(
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

    let sigma = stage.sigma.try_read().unwrap();
    let row_sigma = sigma.row(ypos / BLOCK_DIM);

    const { assert!(D::F32Vec::LEN <= 16) };

    let sm = stage.sigma_scale * 1.65;
    let bsm = sm * stage.border_sad_mul;
    let sad_mul_storage = prepare_sad_mul_storage(xpos, ypos, sm, bsm);

    let scale0 = D::F32Vec::splat(d, stage.channel_scale[0]);
    let scale1 = D::F32Vec::splat(d, stage.channel_scale[1]);
    let scale2 = D::F32Vec::splat(d, stage.channel_scale[2]);
    let c_one = D::F32Vec::splat(d, 1.0);
    let c_zero = D::F32Vec::splat(d, 0.0);
    let min_sigma = D::F32Vec::splat(d, MIN_SIGMA);

    ForEachChunk::<3, 3, 1, 3, 1>::run(
        d,
        xsize,
        input_rows,
        output_rows,
        #[inline(always)]
        |x, in_view, out_view| {
            let sigma = get_sigma(d, x + xpos, row_sigma);
            let sad_mul = D::F32Vec::load(d, &sad_mul_storage[x % 8..]);

            let sigma_mask = min_sigma.gt(sigma);
            if sigma_mask.all() {
                out_view.store::<0, 0>(in_view.load::<0, 1, 0>());
                out_view.store::<1, 0>(in_view.load::<1, 1, 0>());
                out_view.store::<2, 0>(in_view.load::<2, 1, 0>());
                return;
            }

            let inv_sigma = sigma * sad_mul;

            let x_cc = in_view.load::<0, 1, 0>();
            let y_cc = in_view.load::<1, 1, 0>();
            let b_cc = in_view.load::<2, 1, 0>();

            let mut w_acc = c_one;
            let mut x_acc = x_cc;
            let mut y_acc = y_cc;
            let mut b_acc = b_cc;

            macro_rules! step_neighbor {
                ($row:expr, $col_offset:expr) => {{
                    let cx = in_view.load::<0, $row, $col_offset>();
                    let cy = in_view.load::<1, $row, $col_offset>();
                    let cb = in_view.load::<2, $row, $col_offset>();

                    let sad = (cx - x_cc).abs().mul_add(
                        scale0,
                        (cy - y_cc)
                            .abs()
                            .mul_add(scale1, (cb - b_cc).abs() * scale2),
                    );
                    let weight = sad.mul_add(inv_sigma, c_one).max(c_zero);
                    w_acc += weight;
                    x_acc = weight.mul_add(cx, x_acc);
                    y_acc = weight.mul_add(cy, y_acc);
                    b_acc = weight.mul_add(cb, b_acc);
                }};
            }

            step_neighbor!(0, 0);
            step_neighbor!(1, -1);
            step_neighbor!(1, 1);
            step_neighbor!(2, 0);

            let inv_w = c_one / w_acc;

            x_acc *= inv_w;
            y_acc *= inv_w;
            b_acc *= inv_w;
            x_acc = sigma_mask.if_then_else_f32(x_cc, x_acc);
            y_acc = sigma_mask.if_then_else_f32(y_cc, y_acc);
            b_acc = sigma_mask.if_then_else_f32(b_cc, b_acc);
            out_view.store::<0, 0>(x_acc);
            out_view.store::<1, 0>(y_acc);
            out_view.store::<2, 0>(b_acc);
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
        epf2_process_simd(d, stage, pos, xsize, input_rows, output_rows);
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

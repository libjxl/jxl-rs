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
    let (xpos, ypos) = pos;
    assert_eq!(input_rows.len(), 3, "Expected 3 channels, got {}", input_rows.len());
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
                D::F32Vec::load(d, input_x[1].get_unchecked(1 + x..)).store(out_x.get_unchecked_mut(x..));
                D::F32Vec::load(d, input_y[1].get_unchecked(1 + x..)).store(out_y.get_unchecked_mut(x..));
                D::F32Vec::load(d, input_b[1].get_unchecked(1 + x..)).store(out_b.get_unchecked_mut(x..));
            }
            continue;
        }

        let inv_sigma = sigma * sad_mul;

        // SAFETY: input rows have at least xsize + 2 elements due to BORDER=(1, 1), and output rows have at least xsize elements.
        unsafe {
            let x_cc = D::F32Vec::load(d, input_x[1].get_unchecked(1 + x..));
            let y_cc = D::F32Vec::load(d, input_y[1].get_unchecked(1 + x..));
            let b_cc = D::F32Vec::load(d, input_b[1].get_unchecked(1 + x..));

            let mut w_acc = one;
            let mut x_acc = x_cc;
            let mut y_acc = y_cc;
            let mut b_acc = b_cc;

            macro_rules! add_neighbor {
                ($y_off:expr, $x_off:expr) => {
                    let cx = D::F32Vec::load(d, input_x[$y_off].get_unchecked($x_off + x..));
                    let cy = D::F32Vec::load(d, input_y[$y_off].get_unchecked($x_off + x..));
                    let cb = D::F32Vec::load(d, input_b[$y_off].get_unchecked($x_off + x..));
                    let sad = (cx - x_cc).abs().mul_add(
                        scale_x,
                        (cy - y_cc).abs().mul_add(scale_y, (cb - b_cc).abs() * scale_b),
                    );
                    let weight = sad.mul_add(inv_sigma, one).max(zero);
                    w_acc += weight;
                    x_acc = weight.mul_add(cx, x_acc);
                    y_acc = weight.mul_add(cy, y_acc);
                    b_acc = weight.mul_add(cb, b_acc);
                };
            }

            add_neighbor!(0, 1);
            add_neighbor!(1, 0);
            add_neighbor!(1, 2);
            add_neighbor!(2, 1);

            let inv_w = one / w_acc;

            x_acc *= inv_w;
            y_acc *= inv_w;
            b_acc *= inv_w;
            let x_acc = sigma_mask.if_then_else_f32(x_cc, x_acc);
            let y_acc = sigma_mask.if_then_else_f32(y_cc, y_acc);
            let b_acc = sigma_mask.if_then_else_f32(b_cc, b_acc);
            x_acc.store(out_x.get_unchecked_mut(x..));
            y_acc.store(out_y.get_unchecked_mut(x..));
            b_acc.store(out_b.get_unchecked_mut(x..));
        }
    }
});

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
